# -*- coding: utf-8 -*-
"""
The study matrix on the AI campus reference (design note, section 5; Phase
17): each study run on the drawn campus - the requests the browser sent,
tests/reference/reference_ai_campus.diagram_*payload.json, or that payload
with an element switched - and checked against something independent.

17a, part 1: the load flow's cases (islanded, a supply unit out), the optimal
power flow, contingency and state estimation; and the ANSI earth fault, which
failed on any network with a VSC.
"""
import contextlib
import io
import json
import math
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = 'reference_ai_campus'


def _payload(name=''):
    with open(os.path.join(HERE, 'reference', f'{GRID}.diagram_{name}payload.json'), encoding='utf-8') as handle:
        return json.load(handle)


def _post(client, payload):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        response = client.post('/', json=payload)
    assert response.status_code == 200, response.get_data(as_text=True)[:400]
    out = json.loads(response.get_data(as_text=True))
    assert not out.get('error'), out.get('message') or out.get('exception')
    return out


def _variant(payload, changes):
    """The payload with fields changed on the elements named by their labels."""
    p = json.loads(json.dumps(payload))
    by_label = {v.get('userFriendlyName'): k for k, v in p.items() if isinstance(v, dict)}
    for label, fields in changes.items():
        p[by_label[label]].update(fields)
    return p


def _labels(payload):
    return {v['name']: v.get('userFriendlyName') for v in payload.values() if isinstance(v, dict) and 'name' in v}


def _by_label(payload, rows):
    """Result rows by their element's label: named by cell, or (the OPF) 'label (cell)'."""
    names = _labels(payload)
    out = {}
    for r in rows:
        name = r['name']
        if name not in names and name.endswith(')') and ' (' in name:
            name = name[name.rindex(' (') + 2:-1]
        out[names.get(name, r.get('label', r['name']))] = r
    return out


# --- load flow: islanded, and a supply unit out ---------------------------------------------

def test_islanded_turbines_and_bess_share_by_droop(client):
    """
    Both feeders open: the four grid-forming BESS PCS (5.5 MVA, 2 %) and the
    two turbines' governors (31.25 MVA, TGOV1's 5 % - the spec gives them no
    dynamics) take the island's deficit at one frequency, each on its droop
    line: f = f0 (1 - R dP / S) for every one. The turbines had stayed at
    their set point, and the BESS took it all at 108 % of their rating.
    """
    base = _payload()
    out = _post(client, _variant(base, {'Feeder 1 breaker': {'closed': 'false'},
                                        'Feeder 2 breaker': {'closed': 'false'}}))
    assert not out.get('warnings'), out.get('warnings')
    assert out['externalgrids'][0]['p_mw'] == pytest.approx(0.0, abs=1e-6)
    gens = _by_label(base, out['generators'])
    pcs = {p['label']: p for p in out['pcs']}
    bess = [pcs[f'BESS {n} PCS'] for n in range(1, 5)]
    f = bess[0]['frequency_hz']
    for p in bess:
        assert p['islanded'] and p['frequency_hz'] == pytest.approx(f, abs=1e-9)
        assert f == pytest.approx(50 * (1 - 0.02 * p['p_mw'] / 5.5), abs=1e-6)
    for g in ('Gas turbine 1', 'Gas turbine 2'):
        assert f == pytest.approx(50 * (1 - 0.05 * (gens[g]['p_mw'] - 12.0) / 31.25), abs=1e-6)
    # The deficit shared by weight S / R: the turbines 625 MW/pu each, the BESS 275.
    dp_gt, dp_bess = gens['Gas turbine 1']['p_mw'] - 12.0, bess[0]['p_mw']
    assert dp_gt / dp_bess == pytest.approx(625 / 275, rel=1e-5)
    assert 49.4 < f < 49.6


@pytest.mark.parametrize('unit, tie, partner, rated', [
    ('Hall 1 SST U1', 'DC tie breaker RG1-RG2', 'Hall 1 SST U2', 7.0),
    ('Hall 1 rectifier U3', 'DC tie breaker RG3-RG4', 'Hall 1 rectifier U4', 7.5),
])
def test_a_supply_unit_out_with_its_tie_closed(client, unit, tie, partner, rated):
    """
    A supply unit out and its row group's tie closed: the other unit feeds
    both row groups (12 MW and the tie's I^2 R) and says it is overloaded.
    The far row group sags by the tie's I R, and its battery, in droop on a
    bus another converter holds, gives what its droop gives at that voltage.
    This never converged: two voltage sources joined by a 0.5 mohm tie, and
    a settling tolerance under pandapower's own rounding.
    """
    base = _payload()
    out = _post(client, _variant(base, {unit: {'in_service': 'false'}, tie: {'closed': 'true'}}))
    warnings = out.get('warnings') or []
    assert any(partner.split()[-1] in w and 'loaded to' in w for w in warnings), warnings
    assert not any('not connected' in w or 'had not settled' in w for w in warnings), warnings
    dc = _by_label(base, out['dcbuses'])
    far = 'Hall 1 row group 1' if 'U1' in unit else 'Hall 1 row group 3'
    v_far = dc[far]['vm_pu']
    i_tie = 6.0 / 0.8                                      # kA, the far row group's 6 MW at 800 V
    assert v_far == pytest.approx(1.0 - i_tie * 0.0005 / 0.8, abs=2e-4)
    conv = {c['name']: c for c in out['dcdcconverters']}
    battery = next(c for c in out['dcdcconverters'] if _labels(base).get(c['name'], '').startswith(
        f"Hall 1 RG{'1' if 'U1' in unit else '3'} battery"))
    assert battery['p_out_mw'] == pytest.approx(1.5 * (1.0 - battery['vm_out_pu']) / 0.05, rel=1e-3)


# --- optimal power flow --------------------------------------------------------------------------

def test_optimal_power_flow_in_merit_order_with_the_halls(client):
    """
    The OPF dialog's request: it converges (Auto voltage angles were always
    on in pandapower's OPF, which then failed on the 330-degree transformer
    shifts), keeps each rectifier-fed hall as its AC draw (they vanished with
    the DC network), and dispatches by price: the turbines at 60 under the
    utility's 90 run to their rating, every source within its window.
    """
    base = _payload('opf_')
    out = _post(client, base)
    assert any('Each VSC is the load its AC side draws' in w for w in out.get('warnings', [])), out.get('warnings')
    loads = {r['name']: r['p_mw'] for r in out['loads']}
    rect = [p for n, p in loads.items() if 'AC side' in n]
    assert sorted(round(p, 2) for p in rect) == [6.05, 6.05, 12.11]
    gens = _by_label(base, out['generators'])
    for g in ('Gas turbine 1', 'Gas turbine 2'):
        assert gens[g]['p_mw'] == pytest.approx(25.0, abs=1e-3)
    for n in range(1, 5):
        assert -5.28 - 1e-6 <= gens[f'BESS {n} PCS']['p_mw'] <= 5.0685
    sgens = _by_label(base, out['staticgenerators'])
    for n in (1, 2):
        assert 1.47 - 1e-6 <= sgens[f'SOFC {n} PCS']['p_mw'] <= 4.655 + 1e-6
    # The utility supplies the rest: the AC loads, motors, the halls' draws and losses, less the generation.
    gen = sum(r['p_mw'] for r in out['generators']) + sum(r['p_mw'] for r in out['staticgenerators'])
    assert out['externalgrids'][0]['p_mw'] < 0 < gen                     # cheaper than the tariff: it exports


# --- contingency -------------------------------------------------------------------------------------

def test_contingency_feeder_n_minus_1(client):
    """Either feeder out, the other carries the campus through the tie: twice the base loading, no violation."""
    base = _payload('contingency_')
    out = _post(client, base)
    cases = {c['outage']: c for c in out['contingency_results']}
    assert set(cases) == {'Feeder 1', 'Feeder 2'}
    flow = _post(client, _payload())
    base_loading = {_labels(_payload()).get(r['name'], r['name']): r['loading_percent'] for r in flow['lines']}
    for out_feeder, other in (('Feeder 1', 'Feeder 2'), ('Feeder 2', 'Feeder 1')):
        c = cases[out_feeder]
        assert c['converged'] and not c['violations'] and c['lost_load_mw'] == 0
        lines = {r['name']: r['loading_percent'] for r in c['line_results']}
        assert lines[other] == pytest.approx(2 * base_loading[other], rel=0.05)


# --- state estimation --------------------------------------------------------------------------------

def test_state_estimation_recovers_the_load_flow(client):
    """
    Simulated measurements from the load flow, estimated back: it converges,
    passes the chi-squared test, and lands on the load flow - the angles with
    the transformers' shifts, as the estimator takes them. It failed on the
    campus: plain pandapower's load flow cannot settle the converters, and
    the estimator has no VSC (an IndexError read as an unobservable network).
    """
    p = _payload()
    p['0'] = {'typ': 'StateEstimationPandaPower Parameters', 'user_email': 't@t'}
    s = _post(client, p)['state_estimation']['summary']
    assert s['converged'] and s['chi2_passed'] and s['compared_with_load_flow']
    assert s['max_vm_error_pu'] < 2e-3 and s['max_va_error_degree'] < 0.2


# --- ANSI earth fault -------------------------------------------------------------------------------------

def test_ansi_earth_fault_with_vscs_on_the_network(client):
    """
    ANSI single-phase on the campus: it failed on any network with a VSC
    (pandapower's model built twice on one net read the first build's VSC
    lookups). At the 35 kV buses it is of the IEC study's order.
    """
    from test_reference_grids import ANSI_PARAMS
    p = _payload('sc1ph_')
    p['0'] = {**ANSI_PARAMS, 'fault_type': '1ph', 'user_email': 't@t'}
    rows = _by_label(p, _post(client, p)['busbars'])
    assert 10 < rows['35 kV Bus A']['i_first_sym_ka'] < 15
    assert rows['GT1 13.8 kV']['i_first_sym_ka'] == pytest.approx(0.0, abs=1e-6)   # behind a delta, ungrounded
