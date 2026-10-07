# -*- coding: utf-8 -*-
"""
The study matrix on the AI campus reference (design note, section 5; Phase
17): each study run on the drawn campus - the requests the browser sent,
tests/reference/reference_ai_campus.diagram_*payload.json, or that payload
with an element switched - and checked against something independent.

17a, part 1: the load flow's cases (islanded, a supply unit out), the optimal
power flow, contingency and state estimation; and the ANSI earth fault, which
failed on any network with a VSC.

17b: the fault and protection studies - ANSI and islanded earth faults,
motor starting, a DC fault at a rack, protection, the POI study's grounding
table and arc flash.
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


# --- 17a part 2: the time series and site screening -----------------------------------------------

def test_time_series_a_day_with_dispatch(client):
    """
    The time-series dialog's request: 24 hourly steps with the microgrid
    dispatch, the racks on their AI training cycles, PV on its clear-sky day.
    Each hour's load is the AC loads plus each rack's rating times its
    profile's mean over that hour (the racks ran at their full 48 MW: a load
    with a minimum voltage drew its rated power at every load flow). Every
    step converges and nothing is unserved (an SST's MV draw counted as a
    load read as load not served, the grid-tied campus as an island running
    short). A smoothing store over a step far longer than its filter is held
    at its set point (its proportional term emptied the supercapacitors below
    0 %); a battery's energy drawn is what it lost.
    """
    import load_profiles_electrisim as lp
    payload = _payload('timeseries_')
    assert payload['0']['microgrid_dispatch'] in (True, 'true')
    out = _post(client, payload)
    mg = out['microgrid']
    steps = mg['steps']
    assert len(steps) == 24 and all(s['converged'] for s in steps)
    assert mg['unserved_mwh'] == pytest.approx(0.0, abs=1e-9) and not any('ran short' in n for n in mg['notes'])
    library, _ = lp.library_from_params(payload['0'])
    rows = [v for v in payload.values() if isinstance(v, dict)]
    ac = sum(float(v['p_mw']) for v in rows if str(v.get('typ', '')).startswith('Load') and not str(v['typ']).startswith('Load DC'))
    racks = [(float(v['p_mw']), v['load_profile_id']) for v in rows if str(v.get('typ', '')).startswith('Load DC')]
    for k, s in enumerate(steps):
        dc = 0.0
        for p, pid in racks:
            prof = library[pid]
            dc += p * lp.average_profile(prof['t'] - prof['t'][0], prof['p'], k * 3600.0, (k + 1) * 3600.0, True)
        assert s['load_mw'] == pytest.approx(ac + dc, abs=1e-6), k
    for st in mg['stores']:
        if st['kind'] == 'Supercapacitor' and 'Rack 10' not in st['label']:
            assert st['soc_min_percent'] == pytest.approx(50.0, abs=1e-3) and st['soc_max_percent'] == pytest.approx(50.0, abs=1e-3), st
        if st['kind'] == 'Battery':
            assert st['drawn_mwh'] == pytest.approx(st['stored_start_mwh'] - st['stored_end_mwh'], abs=1e-9), st['label']
    pv = [r['p_mw'] for r in mg['pcs'] if r['label'] == 'PV 1 PCS']
    sky = library['clear_sky']['p']
    assert all(p == pytest.approx(0.0, abs=1e-9) for p, g in zip(pv, sky) if g == 0)
    assert max(range(24), key=lambda h: pv[h]) == max(range(24), key=lambda h: sky[h])


def test_site_screening_a_campus_board(client):
    """
    The AC IT A board as a 2 MW site, screened against the feeders' outages:
    it fits, base and N-1, with some 2.06 MW of headroom. Every size "did not
    converge" on the campus - its load flow was pandapower's alone, which
    cannot settle the converters. The load flow agrees: at 2 MW (the site's
    total) within 0.95-1.05 pu and nothing above 100 %; at 2.3 MW past a limit.
    """
    p = _payload()
    p['0'] = {'typ': 'DataCenterSiteScreeningPandaPower Parameters', 'site_load_ids': 'AC IT A', 'mw_sizes': '2',
              'power_factor': '0.95', 'include_n11': 'false', 'element_type': 'line', 'voltage_limits': 'true',
              'thermal_limits': 'true', 'min_vm_pu': '0.95', 'max_vm_pu': '1.05', 'max_loading_percent': '100',
              'user_email': 't@t', 'rpc_stream': True}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        response = client.post('/', json=p)
    result = json.loads(response.get_data(as_text=True).strip().splitlines()[-1])
    assert result['type'] == 'result', result
    (site,) = result['data']['screening_results']
    assert site['headroom_mw'] >= 2.0 and site['base_violations'] == 0 and site['worst_n1_violations'] == 0, site
    assert not site['upgrade_likely']
    assert 2.0 <= site['headroom_mw'] < 2.3

    def worst(p_mw):
        q = p_mw * math.tan(math.acos(0.95))
        flow = _post(client, _variant(_payload(), {'AC IT A': {'p_mw': str(p_mw), 'q_mvar': str(q)}}))
        return min(b['vm_pu'] for b in flow['busbars']), max(t['loading_percent'] for t in flow['transformers'])
    vm, loading = worst(2.0)
    assert vm >= 0.95 and loading <= 100.0
    vm, loading = worst(2.3)
    assert vm < 0.95 or loading > 100.0


# --- 17b: faults and protection ------------------------------------------------------------

R_N_13 = 13.8e3 / math.sqrt(3) / 400
X0_ZIGZAG = 0.12 * 35e3 / math.sqrt(3) / 400


def _golden():
    with open(os.path.join(HERE, 'golden', f'{GRID}.json'), encoding='utf-8') as handle:
        return json.load(handle)


def _zigzag_neutral_ka(c=1.0):
    """A zigzag's earth-fault current alone: 3 c V_ph / |Z0 + 3 R_N|, its Z0 12 % on 400 A with X/R 10."""
    return 3 * c * 35 / math.sqrt(3) / abs(complex(0.1 * X0_ZIGZAG + 3 * 50, X0_ZIGZAG))


def test_ansi_earth_faults_through_the_neutral_resistors(client):
    """
    ANSI with the short-circuit dialog's request (the load flow's carries no
    neutral resistors): a 13.8 kV earth fault is held by its 400 A resistor -
    3 V_ph / |2 Z1 + Z0 + 3 R_N|, a little under 400 A at 1.0 pu; at 35 kV the
    IEC study's current over its c of 1.1, within the two standards' X/R.
    """
    from test_reference_grids import ANSI_PARAMS
    p = _payload('sc1ph_')
    p['0'] = {**ANSI_PARAMS, 'fault_type': '1ph', 'user_email': 't@t'}
    ansi = _by_label(p, _post(client, p)['busbars'])
    golden = _golden()['bus_sc_1ph_max']
    assert 0.39 < ansi['Campus A 13.8 kV']['i_first_sym_ka'] < 13.8 / math.sqrt(3) / R_N_13
    assert ansi['35 kV Bus A']['i_first_sym_ka'] == pytest.approx(golden['BUS_A']['ikss_ka'] / 1.1, rel=0.08)


def test_islanded_earth_faults_held_by_the_zigzags(client):
    """
    Islanded - both feeders open - a 35 kV earth fault has the zigzags for its
    only zero-sequence path: each in service adds its c x 400 A or so; the
    turbines and the PCS feed the rest. It was refused: the study took every
    bus for isolated with no grid, the turbines not counted as sources.
    """
    base = _payload('sc1ph_')
    island = _variant(base, {'Feeder 1 breaker': {'closed': 'false'}, 'Feeder 2 breaker': {'closed': 'false'}})
    both = _by_label(island, _post(client, island)['busbars'])
    one = _variant(island, {'Zigzag B breaker': {'closed': 'false'}})
    one = _by_label(one, _post(client, one)['busbars'])
    gain = both['35 kV Bus A']['ikss_ka'] - one['35 kV Bus A']['ikss_ka']
    assert gain == pytest.approx(_zigzag_neutral_ka(1.1), rel=0.05)
    assert both['35 kV Bus A']['ikss_ka'] < 0.2 * _golden()['bus_sc_1ph_max']['BUS_A']['ikss_ka']
    assert both['Campus A 13.8 kV']['ikss_ka'] > 1.0


def test_motor_starting_one_chiller(client):
    """
    One 2 MW chiller started across the line: its locked-rotor current
    6 x I_rated, drawn at its dipped voltage; the dip at 13.8 kV about its
    starting MVA over the bus's short-circuit MVA plus it, within the 15 %
    limit. It failed before the start: plain pandapower cannot solve the
    campus.
    """
    p = _payload()
    cid = {v.get('userFriendlyName'): v.get('id') for v in p.values() if isinstance(v, dict)}
    p['0'] = {'typ': 'MotorStartingPandaPower Parameters', 'mode': 'steady', 'motor_ids': cid['Chiller A1'],
              'starting_method': 'dol', 'voltage_limit_percent': '15', 'thermal_limit_percent': '100',
              'frequency': '50', 'user_email': 't@t'}
    out = _post(client, p)
    (m,) = out['motors']
    i_rated = 2.0 / 0.96 / 0.9 / (math.sqrt(3) * 13.8)
    assert m['i_rated_ka'] == pytest.approx(i_rated, rel=1e-9)
    assert m['i_start_nominal_ka'] == pytest.approx(6 * i_rated, rel=1e-9)
    dip = out['summary']['worst_dip_percent']
    assert m['i_start_ka'] == pytest.approx(m['i_start_nominal_ka'] * (1 - dip / 100), rel=0.02)
    assert out['summary']['n_fail_voltage'] == 0 and 2 < dip < 15
    s_start = math.sqrt(3) * 13.8 * 6 * i_rated
    s_sc = math.sqrt(3) * 13.8 * _golden()['bus_sc_3ph_max']['CA_13']['ikss_ka'] / 1.1
    assert dip == pytest.approx(100 * s_start / (s_sc + s_start), rel=0.3)


def test_dc_fault_at_the_54_v_rack(client):
    """
    A bolted fault on rack 10's 54 V bus: its 130 F supercapacitor module
    discharges through its 4 mohm ESR, i(t) = V0 / ESR e^(-t / ESR C) -
    13.5 kA at once - while its DC/DC shelf blocks; over the 60 ms run (three
    AC periods) it barely decays. The AC short circuit behind the converters
    solves: a grid-forming PCS's missing machine data made it NaN.
    """
    from test_dc_fault import STUDY
    p = _payload()
    cell = {v.get('userFriendlyName'): v['name'] for v in p.values() if isinstance(v, dict) and 'name' in v}
    p['0'] = {**STUDY, 'fault_bus': cell['C1 rack 10 54 V'], 'duration_ms': '20'}
    out = _post(client, p)
    assert not any('did not solve' in w for w in out['warnings']), out['warnings']
    (f,) = out['dcfault']['faults']
    assert f['ip_ka'] == pytest.approx(54 / 0.004 / 1e3, rel=0.01)
    assert f['ik_ka'] == pytest.approx(f['ip_ka'] * math.exp(-0.060 / (0.004 * 130)), rel=0.03)


@pytest.mark.parametrize('fault', ['3ph', '1ph'])
def test_protection_feeder_faults(client, fault):
    """
    Overcurrent relays on every breaker, set automatically: I> the rated
    current x the CT factor 1.2 x the overload factor 1.25 - a feeder's
    1.2 kA, a turbine's or a campus's step-up's at 35 kV. A fault halfway
    along a feeder trips its breaker first, at I>>'s 0.3 s, and nothing
    miscoordinates. Earth faults failed: the faulted line's new half had no
    zero sequence, and pandapower's zero sequence cannot take a VSC.
    """
    with open(os.path.join(HERE, 'reference', 'reference_radial.diagram_protection_payload.json'),
              encoding='utf-8') as handle:
        params = json.load(handle)['0']
    p = _payload('sc1ph_')
    for v in p.values():
        if isinstance(v, dict) and str(v.get('typ', '')).startswith('Switch'):
            v.update({'protection_type': 'ocr', 'curve_type': 'standard_inverse', 'tms': '1.0', 't_grade': '0.5',
                      't_gg': '0.3', 't_g': '0.8', 't_diff': '0.3', 'pickup_mode': 'auto'})
    p['0'] = {**params, 'fault_type': fault, 'user_email': 't@t'}
    out = _post(client, p)
    settings = {d['switch_name']: d['settings'] for d in out['devices']}
    assert settings['Feeder 1 breaker']['I_g_a'] == pytest.approx(1200 * 1.2 * 1.25)
    assert settings['GT1 breaker']['I_g_a'] == pytest.approx(32e3 / (math.sqrt(3) * 35) * 1.2 * 1.25)
    assert settings['Campus A breaker']['I_g_a'] == pytest.approx(16e3 / (math.sqrt(3) * 35) * 1.2 * 1.25)
    assert out['summary']['n_scenarios'] == 2
    assert out['summary']['n_miscoordination'] == 0 and out['summary']['n_unwanted_trips'] == 0
    for sc in out['scenarios']:
        assert sc['short_circuit']['ikss_ka'] > 1.0, sc['fault_label']
        trips = sorted((t['t_trip_s'], t['switch_name']) for t in sc['trip'] if t['tripped'])
        assert trips[0] == (0.3, sc['fault_label'].split(',')[0] + ' breaker'), trips


def test_poi_study_grounding_rows_for_the_zigzags(client):
    """
    The POI study's grounding table: a zigzag's row is the earth fault at the
    bus it grounds - its LV is its own unloaded delta, which read 0 kA - and
    its own neutral current, 3 V_ph / |Z0 + 3 R_N|; the campus step-downs'
    13.8 kV resistors at 400 A.
    """
    from test_reference_grids import POI_PARAMS
    p = _payload('sc1ph_')
    p['0'] = {**POI_PARAMS, 'user_email': 't@t'}
    rows = {r['trafo_name']: r for r in _post(client, p)['grounding']}
    for z in ('Zigzag A', 'Zigzag B'):
        assert rows[z]['rn_ohm'] == 50.0
        assert rows[z]['neutral_i_ka'] == pytest.approx(_zigzag_neutral_ka(), rel=1e-6)
        assert rows[z]['slg_i_ka_lv'] == pytest.approx(_golden()['bus_sc_1ph_max']['BUS_A']['ikss_ka'], rel=1e-4)
    assert rows['Campus A transformer']['rn_ohm'] == pytest.approx(R_N_13, rel=1e-4)


def test_arc_flash_by_voltage_class(client):
    """
    Each bus as the typical equipment of its class: 13.8 and 0.48 kV by IEEE
    1584-2018 for the same inputs, 35 kV by Ralph Lee; the 0.69 kV PCS buses,
    at 85 kA bolted, beyond the standard's 65 kA and said so.
    """
    from test_reference_grids import _ieee1584, _lee
    p = _payload()
    p['0'] = {'typ': 'ArcFlashPandaPower Parameters', 'electrode_config': 'VCB', 'equipment_mode': 'by_voltage',
              'working_distance_mm': '455', 'conductor_gap_mm': '25', 'enclosure_height_mm': '508',
              'enclosure_width_mm': '508', 'enclosure_depth_mm': '508', 'clearing_time_s': '0.2',
              'clearing_time_min_s': '0.2', 'user_email': 't@t'}
    out = _post(client, p)
    rows = {r['name']: r for r in out['arc_flash']}
    for name in ('Campus A 13.8 kV', 'Campus A 0.48 kV'):
        row = rows[name]
        gap, distance, enclosure = (32, 610, (508, 508, 508)) if row['vn_kv'] <= 0.6 else (152, 910, (1143, 762, 762))
        assert row['method'] == 'IEEE1584-2018'
        assert row['incident_energy_cal_cm2'] == pytest.approx(_ieee1584(row, gap, distance, enclosure), rel=1e-6), name
    lee = rows['35 kV Bus A']
    assert lee['method'] == 'RalphLee'
    assert lee['incident_energy_cal_cm2'] == pytest.approx(_lee(lee, 910)[0], rel=1e-6)
    assert any('BESS 2 0.69 kV' in w and '65 kA' in w for w in out['warnings'])
