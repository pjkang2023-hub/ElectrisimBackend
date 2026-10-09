# -*- coding: utf-8 -*-
"""
The study matrix on the 800 VDC AI factory reference (design note, section 5;
Phase 20): each study run on the drawn factory - the requests the browser
sent, tests/reference/reference_ai_factory_800vdc.diagram_*payload.json, or
that request with an element switched - and checked against something
independent.

20a, part 1: the load flow's cases - a rectifier out (N+1), a lineup lost and
the catcher picking up through the diodes, an SST out (2+1), a transformer out
with the tie closed - the conversion losses of Hall A against Hall B, and
contingency.

20a, part 2: the optimal power flow in merit order, a day's time series with
the microgrid dispatch, and the back-up case - the grid lost and the turbine
off, ten minutes on storage.
"""
import contextlib
import io
import json
import math
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = 'reference_ai_factory_800vdc'
V0 = 0.8                                    # the 800 V buses, kV


def _payload(name=''):
    with open(os.path.join(HERE, 'reference', f'{GRID}.diagram_{name}payload.json'), encoding='utf-8') as handle:
        return json.load(handle)


def _spec_rows():
    with open(os.path.join(HERE, 'reference', f'{GRID}.spec.json'), encoding='utf-8') as handle:
        spec = json.load(handle)
    return {r.get('name', r['id']): r for k, rows in spec.items() if isinstance(rows, list) for r in rows if 'id' in r}


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


def _step_loads(payload, dt, n):
    """Each step's load by hand: the AC loads, and each rack's rating times its cycle's mean over the step."""
    import load_profiles_electrisim as lp
    library, _ = lp.library_from_params(payload['0'])
    rows = [v for v in payload.values() if isinstance(v, dict) and str(v.get('in_service', 'true')) != 'false']
    ac = sum(float(v['p_mw']) for v in rows
             if str(v.get('typ', '')).startswith('Load') and not str(v['typ']).startswith('Load DC'))
    out = []
    for k in range(n):
        dc = 0.0
        for v in rows:
            if str(v.get('typ', '')).startswith('Load DC'):
                prof = library[v['load_profile_id']]
                dc += float(v['p_mw']) * lp.average_profile(prof['t'] - prof['t'][0], prof['p'], k * dt, (k + 1) * dt, True)
        out.append(ac + dc)
    return out


def _by_label(payload, rows):
    names = {v['name']: v.get('userFriendlyName') for v in payload.values() if isinstance(v, dict) and 'name' in v}
    return {names.get(r['name'], r['name']): r for r in rows}


@pytest.fixture(scope='module')
def base(client):
    payload = _payload()
    out = _post(client, payload)
    assert not out.get('warnings'), out.get('warnings')
    return payload, out


def _dcdc_input(p_out, eta, p_nl):
    return p_out / eta + p_nl if p_out >= 0 else p_out * eta + p_nl


def _sst_mv(row, p_dc):
    link = _dcdc_input(p_dc, row['dcdc_efficiency_percent'] / 100, row['dcdc_no_load_kw'] / 1e3)
    return _dcdc_input(link, row['rect_efficiency_percent'] / 100, row['rect_no_load_kw'] / 1e3)


# --- 20a, part 1: the load flow's cases ------------------------------------------------------------

def test_a_rectifier_out_the_other_three_share(client, base):
    """
    N+1: Lineup A's rectifier 1 out, its other three hold the 800 V and share
    what the four did, a third each (2.02 MW, within their 2.2 MVA); the
    shelves see nothing.
    """
    payload, before = base
    out = _post(client, _variant(payload, {'Lineup A rectifier 1': {'in_service': 'false'}}))
    assert not out.get('warnings'), out.get('warnings')
    was, now = _by_label(payload, before['vscs']), _by_label(payload, out['vscs'])
    total = -sum(was[f'Lineup A rectifier {k}']['p_dc_mw'] for k in range(1, 5))
    assert now['Lineup A rectifier 1']['p_dc_mw'] == pytest.approx(0.0, abs=1e-9)
    for k in (2, 3, 4):
        assert -now[f'Lineup A rectifier {k}']['p_dc_mw'] == pytest.approx(total / 3, abs=1e-6), k
    assert 2.0 < total / 3 < 2.2
    dc_was, dc_now = _by_label(payload, before['dcbuses']), _by_label(payload, out['dcbuses'])
    for name, row in dc_was.items():
        assert dc_now[name]['vm_pu'] == pytest.approx(row['vm_pu'], abs=1e-8), name


def test_a_lineup_lost_the_catcher_and_its_store_pick_up(client, base):
    """
    Lineup A's breaker open (the paper's Figure 4): its 0.48 kV bus and its
    rectifiers are named as unsupplied, and every Lineup A shelf is carried
    through both its diodes - from the catcher at 784 V, and from Lineup A's
    store in droop, which gives what its droop gives at its bus's voltage,
    P = P_r (V0 - V) V / (droop V0^2), some 1.1 MW, the catcher the rest. Each
    shelf served at its power; Lineup B as it was. This never solved: the
    rectifiers stayed on their dead bus, then the store and the catcher's
    diodes chased each other.
    """
    payload, before = base
    rows = _spec_rows()
    out = _post(client, _variant(payload, {'Lineup A breaker': {'closed': 'false'}}))
    warnings = out.get('warnings') or []
    assert len(warnings) == 2, warnings
    assert any(w.startswith('Bus Lineup A 0.48 kV has no supply') for w in warnings), warnings
    assert any(w.startswith('Lineup A rectifier 1, Lineup A rectifier 2, Lineup A rectifier 3, Lineup A rectifier 4 '
                            'have no AC supply') for w in warnings), warnings
    dc, diodes = _by_label(payload, out['dcbuses']), _by_label(payload, out['dcdiodes'])
    conv, vsc = _by_label(payload, out['dcdcconverters']), _by_label(payload, out['vscs'])
    # The store's droop: its set voltage behind droop V0^2 / P_r.
    store = rows['Lineup A DC Store converter']
    v = dc['Lineup A 800 V']['vm_pu'] * V0
    p_droop = store['rated_mw'] * (V0 - v) * v / (store['droop_percent'] / 100 * V0 * V0)
    assert conv['Lineup A DC Store converter']['p_out_mw'] == pytest.approx(p_droop, rel=1e-4)
    assert 0.9 < p_droop < store['rated_mw']
    # Each Lineup A shelf through both its diodes, its demand met.
    demand = {f'Shelf A{k}': 1.0 for k in range(2, 7)}
    demand['Shelf A1'] = _by_label(payload, before['dcdcconverters'])['Rack A1 power supply 800/50 V']['p_in_mw']
    for shelf, p in demand.items():
        a, c = diodes[f'{shelf} diode from lineup A'], diodes[f'{shelf} diode from the catcher']
        assert a['conducting'] and c['conducting'], shelf
        v_shelf = dc[f'{shelf} 800 V']['vm_pu'] * V0
        assert (a['i_ka'] + c['i_ka']) * v_shelf == pytest.approx(p, rel=1e-3), shelf
        assert v_shelf > 0.97 * V0
    # The catcher's four rectifiers share the rest; Lineup A's carry nothing; Lineup B's as before.
    catcher = [-vsc[f'Catcher B rectifier {k}']['p_dc_mw'] for k in range(1, 5)]
    assert max(catcher) - min(catcher) < 1e-5 and 1.1 < catcher[0] < 1.4     # the diodes settled to 1 mV
    assert sum(catcher) + p_droop == pytest.approx(sum(demand.values()), rel=0.01)
    for k in range(1, 5):
        assert vsc[f'Lineup A rectifier {k}']['p_dc_mw'] == pytest.approx(0.0, abs=1e-12)
    was = _by_label(payload, before['dcbuses'])
    for k in range(1, 7):
        assert dc[f'Shelf B{k} 800 V']['vm_pu'] == pytest.approx(was[f'Shelf B{k} 800 V']['vm_pu'], abs=1e-6)


def test_an_sst_out_the_other_two_carry_hall_b_past_their_rating(client, base):
    """
    2+1: Hall B's SST 1 out, its other two carry the hall's 10 MW, 5 MW of DC
    each - their stages' efficiencies by hand - and say so: their rectifier
    stages 102.4 % loaded and their DC/DC stages 101.2 %. The 2+1 of the note
    needs SSTs a few percent above 5 MW.
    """
    payload, _ = base
    rows = _spec_rows()
    out = _post(client, _variant(payload, {'Hall B SST 1': {'in_service': 'false'}}))
    ssts = _by_label(payload, out['ssts'])
    assert ssts['Hall B SST 1']['p_mv_mw'] == pytest.approx(0.0, abs=1e-9)
    warnings = out.get('warnings') or []
    for k in (2, 3):
        row = rows[f'Hall B SST {k}']
        p_mv = _sst_mv(row, 5.0)
        assert ssts[f'Hall B SST {k}']['p_mv_mw'] == pytest.approx(p_mv, abs=1e-6)
        link = _dcdc_input(5.0, row['dcdc_efficiency_percent'] / 100, row['dcdc_no_load_kw'] / 1e3)
        for stage, loading in (('rectifier', 100 * p_mv / row['rect_rated_mw']),
                               ('dcdc', 100 * link / row['dcdc_rated_mw'])):
            assert f"'Hall B SST {k}': its {stage} stage is loaded to {loading:.1f} %" in ' '.join(warnings), stage
    assert len(warnings) == 4, warnings


def test_a_transformer_out_the_tie_closed(client, base):
    """
    T1's breaker open and the bus tie closed: T2 carries the site, the two
    34.5 kV buses one, T1 energized from 245 kV at no load. The turbine's
    20 MW on Bus 2 leaves T2 lightly loaded: the import as before.
    """
    payload, before = base
    out = _post(client, _variant(payload, {'T1 breaker': {'closed': 'false'}, 'Bus tie': {'closed': 'true'}}))
    assert not out.get('warnings'), out.get('warnings')
    tr, buses = _by_label(payload, out['transformers']), _by_label(payload, out['busbars'])
    assert buses['34.5 kV Bus 1']['vm_pu'] == pytest.approx(buses['34.5 kV Bus 2']['vm_pu'], abs=1e-9)
    assert tr['T1 245/34.5 kV']['p_hv_mw'] == pytest.approx(0.6 * 40 / 1e3, rel=0.05)        # its iron, pfe 24 kW
    grid = out['externalgrids'][0]['p_mw']
    assert tr['T2 245/34.5 kV']['p_hv_mw'] == pytest.approx(grid - tr['T1 245/34.5 kV']['p_hv_mw'], abs=1e-6)
    assert grid == pytest.approx(before['externalgrids'][0]['p_mw'], abs=0.1)
    s_t2 = math.hypot(tr['T2 245/34.5 kV']['p_hv_mw'], tr['T2 245/34.5 kV']['q_hv_mvar'])
    assert tr['T2 245/34.5 kV']['loading_percent'] == pytest.approx(100 * s_t2 / 40, rel=0.02)


def test_conversion_losses_hall_a_against_hall_b(base):
    """
    From 34.5 kV to the IT at the same 1 MW racks: Hall A through its lineup
    transformers, rectifiers (I^2 R each side) and diodes (v_f I + r_on I^2,
    the paper's 0.2 %), Hall B through its SSTs (two 99 % stages and their
    no-load losses) - each stage by hand within the paper's "about 98 %" or
    better. Hall A 97.6 %, Hall B 97.4 %: the transformer and rectifier ahead
    by 0.2 points at this load.
    """
    payload, out = base
    rows = _spec_rows()
    tr, vsc, dio = _by_label(payload, out['transformers']), _by_label(payload, out['vscs']), _by_label(payload, out['dcdiodes'])
    ssts, dc = _by_label(payload, out['ssts']), _by_label(payload, out['dcbuses'])
    lineups = ('Lineup A', 'Lineup B')
    # The rectifiers: DC power plus I^2 R on each side.
    rect_ac = sum(v['p_mw'] for k, v in vsc.items() if k.startswith(lineups))
    rect_dc = -sum(v['p_dc_mw'] for k, v in vsc.items() if k.startswith(lineups))
    by_hand = 0.0
    for k, v in vsc.items():
        if k.startswith(lineups):
            r = rows[k]
            i_dc = -v['p_dc_mw'] / V0
            i_ac = v['p_mw'] / (math.sqrt(3) * 0.48 * _by_label(payload, out['busbars'])[f"{k[:8]} 0.48 kV"]['vm_pu'])
            by_hand += i_dc * i_dc * r['r_dc_ohm'] + 3 * i_ac * i_ac * r['r_ohm']
    assert rect_ac - rect_dc == pytest.approx(by_hand, rel=2e-3)
    # The diodes: v_f I + r_on I^2.
    lineup_diodes = {k: d for k, d in dio.items() if 'from lineup' in k}
    loss = sum(d['loss_kw'] for d in lineup_diodes.values()) / 1e3
    assert loss == pytest.approx(sum(1.6e-3 * d['i_ka'] + 1e-4 * d['i_ka'] ** 2 for d in lineup_diodes.values()), rel=1e-3)
    delivered = sum(d['p_mw'] for d in lineup_diodes.values()) - loss
    # Hall A from 34.5 kV: its lineups' transformers less their AC loads (0.6 MW each).
    p_a = tr['Lineup A transformer']['p_hv_mw'] + tr['Lineup B transformer']['p_hv_mw'] - 1.2
    eta_a = delivered / p_a
    # Hall B: three SSTs sharing its 10 MW.
    p_b = sum(s['p_mv_mw'] for s in ssts.values())
    assert p_b == pytest.approx(sum(_sst_mv(rows[k], 10.0 / 3) for k in ssts), abs=1e-6)
    eta_b = 10.0 / p_b
    stages = {'rectifier': rect_dc / rect_ac, 'diode': delivered / (delivered + loss),
              'lineup transformer': rect_ac / p_a, 'SST': eta_b}
    for name, eta in stages.items():
        assert 0.97 < eta < 0.999, (name, eta)
    assert stages['diode'] == pytest.approx(1 - 1.73 / 800, abs=2e-4)          # 0.2 %
    assert eta_a == pytest.approx(0.976, abs=1e-3) and eta_b == pytest.approx(0.974, abs=1e-3)
    assert eta_a > eta_b


def test_contingency_each_transformer_and_source(client, base):
    """
    The contingency dialog's request, every transformer and source: all
    converge. T1 or T2 out with the tie open cuts off its bus's side - the
    AC loads and SSTs there by hand (the tie's transfer is not in a static
    study); a lineup transformer its 0.6 MW AC load (its shelves carried by
    the catcher); the cooling plant's its 4 MW; the turbine, the BESS PCS or
    the eSTATCOM nothing - the grid takes them up. Each SST its MV draw.
    """
    payload, flow = base
    out = _post(client, _payload('contingency_'))
    cases = {c['outage']: c for c in out['contingency_results']}
    assert all(c['converged'] for c in cases.values())
    ssts = _by_label(payload, flow['ssts'])
    lost = {
        'T1 245/34.5 kV': 0.6 + 0.6 + ssts['Hall B SST 1']['p_mv_mw'],
        'T2 245/34.5 kV': 0.6 + 4.0 + ssts['Hall B SST 2']['p_mv_mw'] + ssts['Hall B SST 3']['p_mv_mw'],
        'Lineup A transformer': 0.6, 'Lineup B transformer': 0.6, 'Catcher B transformer': 0.6,
        'Cooling plant transformer': 4.0, 'Gas turbine GSU': 0.0, 'Substation BESS transformer': 0.0,
        'eSTATCOM transformer': 0.0, 'Gas turbine': 0.0,
    }
    for name, mw in lost.items():
        assert cases[name]['lost_load_mw'] == pytest.approx(mw, abs=2e-3), name
    for name, c in cases.items():
        if name not in lost:                              # the PCS
            assert 'PCS' in name or 'eSTATCOM' in name, name
            assert not c['violations'] and c['lost_load_mw'] == 0
    assert not cases['Gas turbine']['violations']
    t1 = sorted(v['element'] for v in cases['T1 245/34.5 kV']['violations'] if v['type'] == 'supply')
    assert t1 == ['Bus_34.5 kV Bus 1', 'Bus_Catcher B 0.48 kV', 'Bus_Lineup A 0.48 kV', 'Bus_Substation BESS 0.69 kV']


# --- 20a, part 2: the optimal power flow, the time series, the back-up case ---------------------------

def test_optimal_power_flow_in_merit_order(client):
    """
    The OPF dialog's request: the turbine at 60 under the grid's 70 runs to its
    35 MW (43.75 MVA at 0.8); the back-up gensets, out of service, give nothing
    (the OPF's request left that out, and dispatched them); the BESS PCS at
    their window's top, alike; the eSTATCOM, on 7.5 MW-s of supercapacitors,
    no active power (free to the OPF, it exported 14.7 MW) - its rating for
    reactive power. The total cost each source's marginal cost times its power.
    """
    payload = _payload('opf_')
    out = _post(client, payload)
    assert out['opf_converged'] is True
    gens = _by_label(payload, out['generators'])

    def g(label):
        return next(r for k, r in gens.items() if k.startswith(label))
    assert g('Gas turbine')['p_mw'] == pytest.approx(35.0, abs=1e-4)
    for x in ('Lineup A', 'Lineup B', 'Catcher B'):
        assert g(f'{x} back-up genset')['p_mw'] == pytest.approx(0.0, abs=1e-9)
    bess = [g(f'Substation BESS PCS {k}')['p_mw'] for k in range(1, 5)]
    assert max(bess) - min(bess) < 1e-6 and 2.4 < bess[0] < 2.75
    est = g('eSTATCOM')
    assert est['p_mw'] == pytest.approx(0.0, abs=1e-6) and abs(est['q_mvar']) <= 15.0
    grid = out['externalgrids'][0]
    assert grid['p_mw'] < 0                                     # the turbine cheaper than the tariff: it exports
    rows = [grid] + [r for r in out['generators'] if r.get('marginal_cost') is not None]
    assert out['total_cost'] == pytest.approx(sum(r['marginal_cost'] * r['p_mw'] for r in rows), rel=1e-6)


def test_time_series_a_day_with_dispatch(client):
    """
    The time-series dialog's request at 60 Hz: 24 hourly steps with the
    microgrid dispatch, every rack on Figure 5's cycle. Each hour's load is the
    AC loads and each rack's rating times the cycle's mean over it (1.1 s does
    not divide an hour: each hour's differs a little); every step
    converges and nothing goes unserved. The eSTATCOM's supercapacitors hold
    their charge - its PCS draws their 182 W of leakage from the grid (at
    none they ended the day below their minimum); each lineup's store gives
    nothing and loses its converter's 2 kW no-load loss.
    """
    payload = _payload('timeseries_')
    assert payload['0']['microgrid_dispatch'] in (True, 'true') and payload['0']['frequency'] == '60'
    out = _post(client, payload)
    mg = out['microgrid']
    steps = mg['steps']
    assert len(steps) == 24 and all(st['converged'] for st in steps)
    assert mg['unserved_mwh'] == pytest.approx(0.0, abs=1e-9)
    for st, load in zip(steps, _step_loads(payload, 3600.0, 24)):
        assert st['load_mw'] == pytest.approx(load, abs=1e-6), st['time_step']
    stores = {st['label']: st for st in mg['stores']}
    est = stores['eSTATCOM supercapacitors']
    assert est['soc_min_percent'] == pytest.approx(est['soc_max_percent'], abs=1e-4)
    assert est['soc_min_percent'] == pytest.approx(100 * (0.9 ** 2 - 0.5 ** 2) / (1 - 0.5 ** 2), abs=1e-4)
    assert est['delivered_mwh'] == pytest.approx(-1350.0 ** 2 / 1e4 * 24 / 1e6, rel=1e-3)
    for x in ('Lineup A', 'Lineup B', 'Catcher B'):
        st = stores[f'{x} DC Store']
        assert st['drawn_mwh'] == pytest.approx(st['stored_start_mwh'] - st['stored_end_mwh'], abs=1e-9)
        assert st['drawn_mwh'] == pytest.approx(0.002 * 24, rel=1e-3)


def test_backup_ten_minutes_on_storage(client):
    """
    The paper's back-up case: the grid lost, the turbine off, the bus tie
    closed - ten one-minute steps. The island's grid-forming PCS share it by
    droop within what each can give for the step: the four substation BESS at
    their window's top, 2.53 MW each; the eSTATCOM, its 7.5 MW-s gone in the
    first minute, nothing after; the island at the frequency the BESS reached
    their limits, f0 (1 - P_max droop / S). The rest is shed: some 6.7 of
    20.5 MW served. Shared by weight alone, the eSTATCOM took 58 % of the
    island, every load was shed and no step after the first solved. The
    800 V stores give nothing: the island holds the AC, the rectifiers the
    800 V (they have no power limit here).
    """
    payload = _variant(_payload('timeseries_'), {'Grid': {'in_service': 'false'},
                                                 'Gas turbine': {'in_service': 'false'},
                                                 'Bus tie': {'closed': 'true'}})
    payload['0'].update(time_steps='10', time_step_s='60')
    out = _post(client, payload)
    mg = out['microgrid']
    steps = mg['steps']
    assert len(steps) == 10 and all(st['converged'] for st in steps)
    for st, load in zip(steps, _step_loads(payload, 60.0, 10)):
        assert st['load_mw'] + st['unserved_mw'] == pytest.approx(load, abs=1e-6), st['time_step']
        assert 6.5 < st['load_mw'] < 7.0
    assert any('ran short' in n for n in mg['notes'])
    rows = _spec_rows()
    by_step = {}
    for r in mg['pcs']:
        by_step.setdefault(r['time_step'], {})[r['label']] = r
    row = rows['Substation BESS PCS 1']
    for t, pcs in by_step.items():
        bess = [pcs[f'Substation BESS PCS {k}']['p_mw'] for k in range(1, 5)]
        assert max(bess) - min(bess) < 1e-6 and 2.52 < bess[0] < 2.54, t        # their window, falling with their voltage
        f = 60.0 * (1 - bess[0] * row['droop_pf_percent'] / 100 / row['s_rated_mva'])
        for r in pcs.values():
            assert r['frequency_hz'] == pytest.approx(f, abs=1e-6), (t, r['label'])
        if t > 0:
            assert pcs['eSTATCOM']['p_mw'] == pytest.approx(0.0, abs=1e-3), t
    stores = {st['label']: st for st in mg['stores']}
    assert stores['eSTATCOM supercapacitors']['soc_max_percent'] < 1.0
    assert stores['Lineup A DC Store']['drawn_mwh'] < 1e-3 and stores['Hall B DC Store 1']['drawn_mwh'] < 1e-3
    st = stores['Substation DC Store 1']
    assert st['drawn_mwh'] == pytest.approx(st['stored_start_mwh'] - st['stored_end_mwh'], abs=1e-9)
