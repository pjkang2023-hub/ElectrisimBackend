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

20b: faults and protection - ANSI earth faults through the 400 A neutral
resistors, DC faults at a lineup's bus, a shelf, the catcher's bus and the
rack's 50 V bus, a shelf fault's selectivity, arc flash by voltage class, and
the harmonic study.

20c: the power swing in the transient stability study (ANDES) - the racks'
training cycle reaching the machines through the converters' AC draw, its
swing at the grid and the turbine, its filtering by the storage tiers, the
torsional screen - and ride-through and islanding.
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


def test_contingency_each_transformer_source_and_converter(client, base):
    """
    The contingency dialog's request, every transformer, source and converter:
    all converge. T1 or T2 out: the bus tie closes, as the site's transfer
    would, and nothing is lost - the other carries both buses within its
    rating. A lineup transformer: its 0.6 MW AC load (its shelves carried by
    the catcher); the cooling plant's its 4 MW; the turbine, the BESS PCS or
    the eSTATCOM nothing - the grid takes them up. Each rectifier, SST and DC/DC
    converter: N+1 everywhere but the rack's own 50/12 V stage, whose 1 MW of
    GPUs is lost - DC load counted now, as AC load was.
    """
    payload, flow = base
    out = _post(client, _payload('contingency_'))
    cases = {c['outage']: c for c in out['contingency_results']}
    assert all(c['converged'] for c in cases.values())
    for name in ('T1 245/34.5 kV', 'T2 245/34.5 kV'):
        assert cases[name]['ties_closed'] == ['Bus tie'] and cases[name]['lost_load_mw'] == pytest.approx(0.0, abs=1e-9)
        assert cases[name]['max_loading_percent'] < 100.0
    lost = {
        'Lineup A transformer': 0.6, 'Lineup B transformer': 0.6, 'Catcher B transformer': 0.6,
        'Cooling plant transformer': 4.0, 'Gas turbine GSU': 0.0, 'Substation BESS transformer': 0.0,
        'eSTATCOM transformer': 0.0, 'Gas turbine': 0.0,
    }
    for name, mw in lost.items():
        assert cases[name]['lost_load_mw'] == pytest.approx(mw, abs=2e-3), name
        assert cases[name]['ties_closed'] == [], name
    converters = [l for l in (f'Lineup {x} rectifier {k}' for x in 'AB' for k in range(1, 5))] + [
        f'Catcher B rectifier {k}' for k in range(1, 5)] + [f'Hall B SST {k}' for k in range(1, 4)] + [
        'Lineup A DC Store converter', 'Lineup B DC Store converter', 'Catcher B DC Store converter',
        'DC bus coupler A-B', 'Rack A1 power supply 800/50 V', 'Hall B DC Store 1 converter',
        'Hall B DC Store 2 converter', 'Hall B supercapacitors converter']
    for name in converters:
        assert cases[name]['lost_load_mw'] == pytest.approx(0.0, abs=1e-6), name
    vrm = cases['Rack A1 50/12 V converters']
    assert vrm['lost_dc_load_mw'] == pytest.approx(1.0, abs=1e-6) and vrm['lost_load_mw'] == pytest.approx(1.0, abs=1e-6)
    assert [v['element'] for v in vrm['violations'] if v['type'] == 'supply'] == ['DC_Bus_Rack A1 12 V']
    for name, c in cases.items():
        if name not in lost and name not in converters and not name.startswith(('T1', 'T2', 'Rack A1 50/12')):
            assert 'PCS' in name or 'eSTATCOM' in name, name
            assert not c['violations'] and c['lost_load_mw'] == 0


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


# --- 20b: faults and protection ------------------------------------------------------------------------

R_N = 34.5e3 / math.sqrt(3) / 400                  # the 34.5 kV neutral resistors, 400 A


def _dc_fault(client, bus, duration_ms='20'):
    from test_dc_fault import STUDY
    p = _payload()
    cell = {v.get('userFriendlyName'): v['name'] for v in p.values() if isinstance(v, dict) and 'name' in v}
    p['0'] = {**STUDY, 'fault_bus': cell[bus], 'duration_ms': duration_ms, 'frequency': '60'}
    out = _post(client, p)
    (f,) = out['dcfault']['faults']
    return out, f


def test_ansi_earth_faults_through_the_neutral_resistors(client):
    """
    ANSI on the short-circuit request: a 34.5 kV earth fault is held by its
    transformer's 400 A resistor - 3 V_ph / |2 Z1 + Z0 + 3 R_N|, a little under
    V_ph / R_N at 1.0 pu (ANSI adds no PCS current, as IEC does). The buses
    behind an ungrounded winding - the BESS's, the eSTATCOM's, the turbine's
    delta - have none; the 0.48 kV lineups, solidly grounded, some 130 kA.
    """
    from test_reference_grids import ANSI_PARAMS
    p = _payload('sc1ph_')
    p['0'] = {**ANSI_PARAMS, 'fault_type': '1ph', 'frequency_hz': 60, 'user_email': 't@t'}
    rows = _by_label(p, _post(client, p)['busbars'])
    for bus in ('34.5 kV Bus 1', '34.5 kV Bus 2'):
        assert 0.39 < rows[bus]['i_first_sym_ka'] < 34.5 / math.sqrt(3) / R_N, bus
    for bus in ('Substation BESS 0.69 kV', 'eSTATCOM 0.69 kV', 'Gas turbine 13.8 kV'):
        assert rows[bus]['i_first_sym_ka'] == pytest.approx(0.0, abs=1e-9), bus
    for bus in ('Lineup A 0.48 kV', 'Lineup B 0.48 kV', 'Catcher B 0.48 kV'):
        assert 100 < rows[bus]['i_first_sym_ka'] < 150, bus


def test_dc_fault_on_a_lineup_bus_the_diodes_block(client):
    """
    A bolted fault on Lineup A's 800 V bus: its four rectifiers' DC links, its
    store's converter's output capacitor and the bus coupler's input capacitor
    discharge into it, and the rectifiers' diode bridges feed it alike; the
    shelves behind their diodes give nothing back - what is behind shelf A1
    (the rack's converters and filter) only trades current among itself - and
    the catcher, behind the shelves' other diodes, none. The study lists every
    DC element; those beyond the faulted section carry only their own small
    transients, under 1 % of the fault's peak.
    """
    out, f = _dc_fault(client, 'Lineup A 800 V')
    contrib = {(c['kind'], c['name']): c for c in f['contributions']}
    rect = [c for (k, n), c in contrib.items() if n.startswith('Lineup A rectifier')]
    assert len(rect) == 8                                   # each its DC link and its diode bridge
    links = [c for c in rect if 'DC-link' in c['kind']]
    assert len(links) == 4 and max(c['ip_ka'] for c in links) - min(c['ip_ka'] for c in links) < 1e-6
    assert ('DC/DC output capacitor', 'Lineup A DC Store converter') in contrib
    assert ('DC/DC input capacitor', 'DC bus coupler A-B') in contrib
    for (k, n), c in contrib.items():
        if n.startswith(('Catcher B rectifier', 'Shelf A')):          # behind the diodes: nothing
            assert abs(c['ip_ka']) < 1e-3 and abs(c['at_peak_ka']) < 1e-3, n
        elif not n.startswith(('Lineup A', 'DC bus coupler', 'Rack A1')):
            assert abs(c['ip_ka']) < 0.01 * f['ip_ka'], n
    behind = sum(c['at_peak_ka'] for (k, n), c in contrib.items() if n.startswith('Rack A1'))
    assert abs(behind) < 1e-3 * f['ip_ka']
    bridges = [c['ik_ka'] for c in rect if 'diodes' in c['kind']]
    assert f['ik_ka'] == pytest.approx(sum(bridges), rel=0.01) and 100 < f['ik_ka'] < 120


def test_dc_fault_on_the_catcher_bus(client):
    """
    A bolted fault on the catcher's 784 V bus: its four rectifiers and its
    store's converter feed it; the shelves, behind their diodes from it, give
    nothing back - neither do Lineups A and B, which reach it only through them.
    """
    out, f = _dc_fault(client, 'Catcher B 800 V')
    names = {c['name'] for c in f['contributions'] if abs(c['at_peak_ka']) > 1e-3 * f['ip_ka']}
    assert {f'Catcher B rectifier {k}' for k in range(1, 5)} <= names
    assert 'Catcher B DC Store converter' in names
    for c in f['contributions']:
        if c['name'].startswith('Shelf '):                                 # behind their diodes: nothing
            assert abs(c['ip_ka']) < 1e-3, c['name']
        elif c['name'].startswith(('Lineup A', 'Lineup B')):
            assert abs(c['ip_ka']) < 0.01 * f['ip_ka'] and abs(c['ik_ka']) < 1e-3, c['name']
    assert f['v_prefault_kv'] == pytest.approx(0.784, abs=1e-6)


def test_dc_fault_at_the_rack_50_v(client):
    """
    A bolted fault on rack A1's 50 V bus: its supercapacitors - four 130 F
    modules behind 4 mohm each, 1 mohm together - give V0 / ESR, 50 kA, at
    once and decay with ESR C = 0.52 s: some 46 kA over the run's last 60 Hz
    period. The 800 / 50 V supply and the 50 / 12 V converters block, their
    capacitors discharging.
    """
    out, f = _dc_fault(client, 'Rack A1 50 V')
    sc = next(c for c in f['contributions'] if c['name'] == 'Rack A1 supercapacitors')
    assert sc['ip_ka'] == pytest.approx(50 / 0.001 / 1e3, rel=0.01)
    t_mid = 0.050 - 0.5 / 60                                  # the middle of the run's last period
    assert sc['ik_ka'] == pytest.approx(50 * math.exp(-t_mid / (0.001 * 520)), rel=0.02)
    assert f['ik_ka'] == pytest.approx(sc['ik_ka'], rel=1e-3)
    kinds = {(c['kind'], c['name']) for c in f['contributions'] if c['ip_ka'] > 1.0}
    assert ('DC/DC output capacitor', 'Rack A1 power supply 800/50 V') in kinds
    assert ('DC/DC input capacitor', 'Rack A1 50/12 V converters') in kinds


def test_a_shelf_fault_trips_its_breaker_and_its_catcher_group(client):
    """
    A bolted fault on shelf A2, fed forward through both its diodes - from
    Lineup A's rectifiers and the catcher's. Its own breaker sees 147 kA
    prospective, far over its 3 kA trip, and opens in 10 us at some 2 kA,
    within its 30 kA. The catcher's side has a breaker per group, not per
    shelf: catcher group A's sees 137 kA over its 12 kA trip and must open
    too, taking the catcher from all six Lineup A shelves - the shelf's own
    breaker does not clear it alone. Every other breaker carries its load.
    """
    out, f = _dc_fault(client, 'Shelf A2 800 V', duration_ms='60')
    feeds = {c['name'] for c in f['contributions'] if 'diodes' in c['kind']}
    assert {f'Lineup A rectifier {k}' for k in range(1, 5)} | {f'Catcher B rectifier {k}' for k in range(1, 5)} <= feeds
    rows = _spec_rows()
    trips = {b['label']: b['ip_ka'] > rows[b['label']]['trip_current_ka'] for b in out['dcfault']['breakers']}
    assert {k for k, v in trips.items() if v} == {'Shelf A2 breaker', 'Catcher group A breaker'}
    own = next(b for b in out['dcfault']['breakers'] if b['label'] == 'Shelf A2 breaker')
    assert own['i_open_ka'] < own['breaking_capacity_ka'] and not own['exceeds'] and own['ip_ka'] > 100


def test_arc_flash_by_voltage_class(client):
    """
    Each bus as the typical equipment of its class: 34.5 and 245 kV by Ralph
    Lee, the rest by IEEE 1584-2018. The 0.48 kV lineups' 127-143 kA and the
    0.69 kV PCS buses' 112-146 kA are past the standard's 106 and 65 kA: each
    is studied at the limit and said so - the lineups' 8 MVA transformers at
    6 % give fault levels beyond IEEE 1584.
    """
    from test_reference_grids import _ieee1584, _lee
    p = _payload()
    p['0'] = {'typ': 'ArcFlashPandaPower Parameters', 'electrode_config': 'VCB', 'equipment_mode': 'by_voltage',
              'working_distance_mm': '455', 'conductor_gap_mm': '25', 'enclosure_height_mm': '508',
              'enclosure_width_mm': '508', 'enclosure_depth_mm': '508', 'clearing_time_s': '0.2',
              'clearing_time_min_s': '0.2', 'user_email': 't@t'}
    out = _post(client, p)
    rows = {r['name']: r for r in out['arc_flash']}
    for bus in ('34.5 kV Bus 1', '34.5 kV Bus 2', 'Grid 245 kV'):
        assert rows[bus]['method'] == 'RalphLee'
        assert rows[bus]['incident_energy_cal_cm2'] == pytest.approx(_lee(rows[bus], 910)[0], rel=1e-6), bus
    for bus in ('Lineup A 0.48 kV', 'Lineup B 0.48 kV', 'Catcher B 0.48 kV'):
        row = rows[bus]
        assert row['method'] == 'IEEE1584-2018' and row['ikss_ka'] > 106
        capped = dict(row, ikss_ka=106.0)
        assert row['incident_energy_cal_cm2'] == pytest.approx(_ieee1584(capped, 32, 610, (508, 508, 508)), rel=1e-6)
        assert any(w.startswith(f'Bus {bus}: Ibf=') and 'LV model max 106 kA' in w for w in out['warnings'])
    for bus in ('Substation BESS 0.69 kV', 'eSTATCOM 0.69 kV'):
        assert any(w.startswith(f'Bus {bus}: Ibf=') and '65 kA' in w for w in out['warnings'])


def _harmonics(client, tmp_path, spectrum=None):
    import opendssdirect as dss
    with open(os.path.join(HERE, 'reference', 'reference_radial.diagram_harmonic_payload.json'),
              encoding='utf-8') as handle:
        params = json.load(handle)['0']
    p = _payload()
    p['0'] = {**params, 'frequency': '60'}
    if spectrum:
        for v in p.values():
            if isinstance(v, dict) and str(v.get('typ', '')).startswith('VSC'):
                v['spectrum'] = spectrum
    before = dss.Basic.DataPath()
    dss.Basic.DataPath(str(tmp_path))
    try:
        out = _post(client, p)
    finally:
        dss.Basic.DataPath(before)
    assert out['harmonic_analysis']['executed']
    return p, _by_label(p, out['busbars'])


def test_harmonics_the_converters_inject_and_ieee_519_is_judged(client, tmp_path):
    """
    The harmonic study (OpenDSS) at 60 Hz: each rectifier and SST injects an
    active front end's spectrum by default (generic: 2 % fifth, 1.5 % seventh,
    falling after), and every bus is judged against IEEE 519-2022's voltage
    limits for its class - 245 kV 1.5 % THD and 1 % a harmonic, 1-69 kV 5 %
    and 3 %, under 1 kV 8 % and 5 %. Before, every bus read 0 % THD: not for
    want of a spectrum alone - a converter carrying nothing (the catcher's
    rectifiers at rest) was a load of no power, its 0/0 harmonic current made
    every solve NaN. With active front ends all pass; with six-pulse diode
    bridges the lineups' 0.48 kV buses fail on a single harmonic.
    """
    p, buses = _harmonics(client, tmp_path)
    assert len(buses) == 10
    for name, b in buses.items():
        assert 0.0 < b['vthd_percent'] < 3.0, name
        assert b['ieee519']['ok'], name
    grid = buses['Grid 245 kV']['ieee519']
    assert (grid['thd_limit_percent'], grid['individual_limit_percent']) == (1.5, 1.0)
    assert (buses['34.5 kV Bus 1']['ieee519']['thd_limit_percent'],
            buses['Lineup A 0.48 kV']['ieee519']['thd_limit_percent']) == (5.0, 8.0)

    _, six = _harmonics(client, tmp_path, spectrum='six_pulse')
    for name in ('Lineup A 0.48 kV', 'Lineup B 0.48 kV'):
        assert not six[name]['ieee519']['ok'] and six[name]['ieee519']['max_individual_percent'] > 5.0, name
    assert six['Grid 245 kV']['ieee519']['ok']
    assert six['Grid 245 kV']['vthd_percent'] > 2 * buses['Grid 245 kV']['vthd_percent']


def _tds(client, off=(), **params):
    """The transient stability study (ANDES) on the factory, 10 s, no fault; the elements named in off out of service."""
    p = _payload('tds_')
    if off:
        p = _variant(p, {label: {'in_service': 'false'} for label in off})
    p['0'].update({'fault_enabled': 'false', 'tf': '10'}, **params)
    return p, _post(client, p)


PCS = ('Substation BESS PCS 1', 'Substation BESS PCS 2', 'Substation BESS PCS 3', 'Substation BESS PCS 4', 'eSTATCOM')
SMOOTHING = ('Hall B supercapacitors converter',)


def _cycle_s(payload):
    import load_profiles_electrisim as lp
    library, _ = lp.library_from_params(payload['0'])
    prof = library['ai_training']
    t = prof['t'] - prof['t'][0]
    return float(t[-1] + (t[-1] - t[-2]))           # a repeating profile's period: one step past its last sample


def _racks_mw(payload, prefix=''):
    return sum(float(v['p_mw']) for v in payload.values() if isinstance(v, dict)
               and str(v.get('typ', '')).startswith('Load DC') and str(v.get('userFriendlyName')).startswith(prefix))


def _fundamental(row, f_hz):
    """A swing's amplitude at the cycle's own frequency: its strongest line within a bin of it."""
    return max(d['amplitude_mw'] for d in row['dominant'] if abs(d['f_hz'] - f_hz) < 0.12)


def _window_mean(t, x, a, b):
    import numpy as np
    t = np.asarray(t, dtype=float)
    k = (t >= a) & (t <= b)
    return float(np.trapezoid(np.asarray(x, dtype=float)[k], t[k]) / (t[k][-1] - t[k][0]))


def test_tds_the_training_cycle_reaches_the_machines(client):
    """
    Each converter's AC draw follows the DC loads it reaches - the racks on
    their 1.1 s training cycle, 40 % to 100 % - so the swing reaches the grid
    and the turbine: before, the converters were held at the load flow's.
    Hall B's racks are smoothed by its supercapacitors (2 s), so its SSTs
    draw a swing far smaller than the lineups' rectifiers. The power drawn
    matches the power given (losses under 2 %), and the swing's strongest
    frequencies are the cycle's own and its harmonics.
    """
    p, out = _tds(client)
    assert any('each following the profiles of the DC loads it reaches' in w
               and 'less the fast part their smoothing stores take (Hall B supercapacitors converter)' in w
               for w in out['warnings'])
    following = {r['load']: r for r in out['load_profiles'] if r['profile'] == "its DC loads' profiles"}
    rectifiers = {f'Lineup {x} rectifier {i} (DC network)' for x in 'AB' for i in range(1, 5)}
    ssts = {f'Hall B SST {i} (MV) (DC network)' for i in range(1, 4)}
    assert set(following) == rectifiers | ssts
    for label in rectifiers:
        r = following[label]['p_mw']
        assert min(r) / max(r) == pytest.approx(0.4, abs=0.01), label
    for label in ssts:
        r = following[label]['p_mw']
        assert min(r) / max(r) > 0.85, label

    ac = sum(float(v['p_mw']) for v in p.values() if isinstance(v, dict) and str(v.get('typ', '')).startswith('Load')
             and not str(v['typ']).startswith('Load DC') and str(v.get('in_service', 'true')) != 'false')
    motors = sum(float(v.get('pn_mech_mw') or 0.0) for v in p.values()
                 if isinstance(v, dict) and str(v.get('typ', '')).startswith('Motor'))
    drawn = ac + motors + sum(sum(r['p_mw']) / len(r['p_mw']) for r in following.values())
    swing = out['swing']
    given = sum(r['mean_mw'] for r in swing['grid'] + swing['machines']) + sum(
        sum(r['p_mw']) / len(r['p_mw']) for r in out['pcs'])
    assert drawn < given < drawn * 1.02

    f_cycle = 1.0 / _cycle_s(p)
    grid, = swing['grid']
    turbine, = swing['machines']
    assert turbine['name'] == 'Gas turbine'
    assert grid['peak_to_peak_mw'] > 5.0 and turbine['peak_to_peak_mw'] > 1.5
    resolution = 1.0 / (out['tf'] - swing['from_s'])
    for row in (grid, turbine):
        for d in row['dominant'][:3]:
            k = d['f_hz'] / f_cycle
            assert abs(k - round(k)) * f_cycle <= resolution, (row['name'], d)


def test_tds_power_swing_filtering_by_tier(client):
    """
    The swing at the grid with no storage, each tier, and all. Tier 2, Hall
    B's supercapacitors smoothing its racks: at the cycle's frequency a first-
    order filter of 2 s leaves 1 / sqrt(1 + (w tau)^2) of Hall B's swing, the
    rest of the racks' as it was - the fundamental at the grid falls by that
    share. Tier 3, the substation's grid-forming PCS on a stiff grid: the
    1.1 s swing passes - the grid holds the frequency their droops answer -
    within 5 %. All: about tier 2's.
    """
    p = _payload('tds_')
    f_cycle = 1.0 / _cycle_s(p)
    cases = {}
    for name, off in (('none', PCS + SMOOTHING), ('tier 2', PCS), ('tier 3', SMOOTHING), ('all', ())):
        _, out = _tds(client, off=off)
        cases[name] = out['swing']['grid'][0]
    none = _fundamental(cases['none'], f_cycle)
    hall_b, total = _racks_mw(p, 'Hall B rack'), _racks_mw(p)
    left = 1.0 / math.sqrt(1.0 + (2 * math.pi * f_cycle * 2.0) ** 2)
    assert _fundamental(cases['tier 2'], f_cycle) == pytest.approx(none * (total - hall_b + hall_b * left) / total,
                                                                   rel=0.1)
    assert _fundamental(cases['tier 3'], f_cycle) == pytest.approx(none, rel=0.05)
    assert _fundamental(cases['all'], f_cycle) == pytest.approx(_fundamental(cases['tier 2'], f_cycle), rel=0.05)
    pp = {k: v['peak_to_peak_mw'] for k, v in cases.items()}
    assert pp['all'] < pp['tier 2'] < pp['tier 3'] < pp['none'] and pp['all'] < 0.6 * pp['none']


def test_tds_turbine_torsional_screen(client):
    """
    The swing's spectrum at the turbine against an assumed first torsional
    mode, 20 Hz (the 15-25 Hz band): at a 5 ms step the band is resolved. The
    1.1 s cycle's steps put a little of it there; the storage tiers take most
    of it away - the substation's grid-forming PCS the steps' sharp edges,
    Hall B's supercapacitors its swing.
    """
    runs = {}
    for name, off in (('none', PCS + SMOOTHING), ('all', ())):
        _, out = _tds(client, off=off, tstep='0.005')
        assert not any('torsional band' in w for w in out['warnings'])
        turbine, = out['swing']['machines']
        assert turbine['band_resolved'] and turbine['band_hz'] == [15.0, 25.0] and turbine['nyquist_hz'] >= 99
        runs[name] = turbine
    assert 0 < runs['none']['band_rms_mw'] < 0.2 * runs['none']['total_rms_mw']
    assert runs['all']['band_rms_mw'] < 0.15 * runs['none']['band_rms_mw']
    # Its ramp depends on the pieces the profiles are applied in (each a step): only roughly.
    assert runs['all']['max_ramp_mw_per_s'] < 0.5 * runs['none']['max_ramp_mw_per_s']


def test_tds_islanding_the_droops_share(client):
    """
    The grid lost at 2 s: the turbine's governor (4 %, 43.75 MVA) and the
    grid-forming PCS (2 %, 4 x 2.75 and 15 MVA) take what the grid gave, each
    by its droop, P = S / R x df / f0 at the island's mean frequency over
    whole cycles. The eSTATCOM's supercapacitors empty some 4.5 s on and it
    stops; the rest take its share, so the frequency settles lower by their
    gains' ratio.
    """
    payload = _payload('tds_')
    grid = next(k for k, v in payload.items() if isinstance(v, dict) and v.get('userFriendlyName') == 'Grid')
    p, out = _tds(client, tf='20', toggle_line=payload[grid]['name'], toggle_t='2.0', max_points='4000')
    t, f = out['time'], out['frequency_hz']
    gt = out['generator_p_mw'][0]['values']
    pcs = {r['label']: r for r in out['pcs']}
    cycle = _cycle_s(p)
    stop = pcs['eSTATCOM']['stopped_s']
    assert pcs['eSTATCOM']['stopped_because'] == 'emptied' and 5.5 < stop < 7.5
    assert all(pcs[k]['stopped_s'] is None for k in PCS[:4])
    assert any("PCS 'eSTATCOM': its supercapacitors emptied" in w for w in out['warnings'])

    gains = {'Gas turbine': 43.75 / 0.04, **{k: 2.75 / 0.02 for k in PCS[:4]}, 'eSTATCOM': 15.0 / 0.02}
    before = (2.0 - cycle, 2.0)
    dfs = []
    for window, running in (((3.2, 3.2 + 2 * cycle), gains),
                            ((20.0 - 4 * cycle, 20.0), {k: g for k, g in gains.items() if k != 'eSTATCOM'})):
        df = (60.0 - _window_mean(t, f, *window)) / 60.0
        dfs.append((df, sum(running.values())))
        assert _window_mean(t, gt, *window) - _window_mean(t, gt, *before) == pytest.approx(
            gains['Gas turbine'] * df, rel=0.03)
        for k in PCS:
            share = running[k] * df if k in running else 0.0
            got = _window_mean(t, pcs[k]['p_mw'], *window) - _window_mean(t, pcs[k]['p_mw'], *before)
            assert got == pytest.approx(share, rel=0.03, abs=0.01), k
    (df1, g1), (df2, g2) = dfs
    assert df2 / df1 == pytest.approx(g1 / g2, rel=0.05)


def test_tds_ride_through_ieee2800_the_turbine_slips_at_the_pcs_limits(client):
    """
    IEEE 2800's envelope at 245 kV from 1 s (0.05 pu in ANDES for 0.32 s,
    then 0.25 pu): the grid-forming PCS reach their current limit, 1.2 pu, at
    once, and their virtual impedance holds them there. So held, they no
    longer prop up 34.5 kV Bus 2: the turbine's rotor angle runs on past
    150 degrees and the run stops - it slips a pole. Without the limits the
    eSTATCOM went far past its rating and the turbine rode through: the ride-
    through rested on current the PCS cannot give.
    """
    import numpy as np
    _, out = _tds(client, grid_voltage_profile='ieee2800', grid_voltage_start_s='1')
    assert out['converged'] is False and 1.5 < out['time'][-1] < 2.2
    assert any(w.startswith('The simulation stopped at t = ') for w in out['warnings'])
    t = np.asarray(out['time'])
    for r in out['pcs']:
        assert 1.0 <= r['current_limited_from_s'] < 1.05, r['label']
        held = np.asarray(r['current_pu'])[(t > 1.1) & (t < 1.3)]
        assert held.max() <= 1.2 * 1.05, r['label']
    delta, = (s for s in out['delta'] if s['name'] == 'Gas turbine')
    assert math.degrees(max(delta['values']) - delta['values'][0]) > 130

    _, free = _tds(client, grid_voltage_profile='ieee2800', grid_voltage_start_s='1', pcs_limits='false')
    assert free['converged'] is True and free['time'][-1] == pytest.approx(10.0)
    est, = (r for r in free['pcs'] if r['label'] == 'eSTATCOM')
    assert max(est['current_pu']) > 2.0 and max(est['q_mvar']) > 15.0


def test_emt_the_rack_holds_its_12_v(client):
    """
    The EMT study with no disturbance, 20 ms: rack A1's 12 V bus holds from
    t = 0, as the load flow has it. Its 50 / 12 V converters pass the GPUs'
    1 MW at 83 kA, within their limit (1.2 pu, 110 kA), and none of the
    rack's converters limits or blocks. At 12 V their output capacitor's ESR
    and ESL and the GPUs' filter's R and L are the 800 V ones scaled by
    (12 / 800)^2: at 0.1 mohm, an 800 V converter's - 0.76 pu here - their
    ESR dropped 8.3 V at 83 kA, and their bus fell to some 2 V within 5 ms.
    """
    p = _payload()
    p['0'] = {'typ': 'EmtStudy Parameters', 'user_email': 't@t', 'duration_ms': '20', 'frequency': '60'}
    emt = _post(client, p)['emt']
    buses = {b['label']: b for b in emt['buses']}
    for label, band in (('Rack A1 12 V', 0.001), ('Rack A1 50 V', 0.01)):
        assert 1 - band < buses[label]['v_min_pu'] and buses[label]['v_max_pu'] < 1 + band, label
    gpus, = (l for l in emt['loads'] if l['label'] == 'Rack A1 GPUs 12 V')
    assert gpus['t_lost_ms'] is None and gpus['v_min_pu'] > 0.99
    conv = {c['label']: c for c in emt['converters']}
    vrm, psu = conv['Rack A1 50/12 V converters'], conv['Rack A1 power supply 800/50 V']
    for c in (vrm, psu):
        assert c['blocked_ms'] is None and c['limited_ms'] == 0, c['label']
    assert vrm['v_dc_min_kv'] == pytest.approx(0.012, rel=1e-3)
    assert vrm['i_peak_ka'] == pytest.approx(1.0 / 0.012, rel=0.02) and vrm['i_peak_ka'] < vrm['current_limit_ka']
    assert not emt['converters_blocked']


def test_emt_ride_through_ieee2800_the_800_v_where_it_holds(client):
    """
    IEEE 2800's envelope at 245 kV in the EMT study, 0 pu from 20 ms: 100 ms
    sees every loss. Hall B's SSTs, on 34.5 kV Bus 2 at some 0.4 pu, cannot
    carry its racks within their current limit and block within 5 ms; its
    stores' converters, short of the load step, block at 0.8 pu after them
    - its racks lose their 800 V within 15 ms. Lineup A and the catcher, on
    Bus 1 with the grid at 0 pu, have only their stores, 2.5 MW each at its
    current limit against some 6 MW: their buses fall through 0.8 pu, all on
    them blocks, and the shelves lose it too, before 100 ms. Lineup B, on Bus 2, rides
    through on its rectifiers at their current limit. Where the 800 V is
    lost, it is for want of power, not energy: no store gives a thousandth
    of what it holds.
    """
    p = _payload()
    p['0'] = {'typ': 'EmtStudy Parameters', 'user_email': 't@t', 'duration_ms': '100', 'frequency': '60',
              'grid_voltage_profile': 'ieee2800', 'grid_voltage_start_ms': '20'}
    emt = _post(client, p)['emt']
    loads = {l['label']: l for l in emt['loads']}
    for i in range(1, 11):
        assert loads[f'Hall B rack {i}']['verdict'] == 'lost supply' and 30 < loads[f'Hall B rack {i}']['t_lost_ms'] < 40
    for i in range(2, 7):
        assert loads[f'Shelf A{i} racks']['verdict'] == 'lost supply' and 55 < loads[f'Shelf A{i} racks']['t_lost_ms'] < 100
    for i in range(1, 7):
        assert loads[f'Shelf B{i} racks']['t_lost_ms'] is None and loads[f'Shelf B{i} racks']['v_min_pu'] > 0.85
    blocked = {b['label']: b['t_ms'] for b in emt['converters_blocked']}
    assert all(20 < blocked[f'Hall B SST {i} rectifier'] < 26 for i in range(1, 4))
    assert {f'Lineup A rectifier {i}' for i in range(1, 5)} <= set(blocked)
    assert not any(label.startswith('Lineup B') for label in blocked)
    for d in emt['ders']:
        if d['kind'] == 'Battery':
            assert d['energy_given_mj'] < 1e-3 * 5000 * 3.6, d['label']        # 5 MWh each


def test_emt_the_rack_supply_damps_its_input_filter(client):
    """
    Shelf A1's breaker limits through 10 uH; the rack's power supply behind
    it has 6.9 mF at its input and draws a constant 1 MW, a negative
    resistance of -V^2 / P = -0.64 ohm. The LC rings at 1 / 2 pi sqrt(L C),
    some 606 Hz, and is damped only when its series resistance passes
    L / (C |R|) = 2.3 mOhm (Middlebrook): at 0.1 mOhm its input capacitor's
    ESR left a 5.4 % ring on the shelf's 800 V. At 5 mOhm it settles, and the
    shelves beside it with it.
    """
    import numpy as np
    l_h, c_f, r = 10e-6, 6.9e-3, 0.8 ** 2 / 1.0
    assert 1 / (2 * math.pi * math.sqrt(l_h * c_f)) == pytest.approx(606, abs=1)
    assert l_h / (c_f * r) * 1e3 == pytest.approx(2.26, abs=0.01)
    p = _payload()
    psu = next(v for v in p.values() if isinstance(v, dict) and v.get('userFriendlyName') == 'Rack A1 power supply 800/50 V')
    assert float(psu['c_in_esr_mohm']) == 5.0
    p['0'] = {'typ': 'EmtStudy Parameters', 'user_email': 't@t', 'duration_ms': '60', 'frequency': '60'}
    emt = _post(client, p)['emt']
    for b in emt['buses']:
        if b['label'] in ('Shelf A1 800 V', 'Shelf A2 800 V'):
            t, v = np.asarray(b['waveform']['t_ms']), np.asarray(b['waveform']['v_kv']) / b['vn_kv']
            assert np.ptp(v[t >= t[-1] - 20]) < 0.002, b['label']
    loads = {l['label']: l['verdict'] for l in emt['loads']}
    assert all(loads[f'Shelf A{i} racks'] == 'settles' for i in range(2, 7))


def test_two_rectifiers_out_the_other_two_at_their_rating_the_store_picks_up(client):
    """
    Lineup A with two of its four rectifiers out: the two left would carry
    some 6 MW, past their 2.2 MVA. Each delivers its rating instead, its
    800 V bus sags, and the lineup's store in droop picks up what its droop
    gives at that voltage, P = P_r (V0 - V) V / (d V0^2); the catcher's
    diodes, 2 % lower, take a little of Shelf A1. Before, a rectifier held
    its 800 V whatever it carried and its store never helped.
    """
    p = _variant(_payload(), {'Lineup A rectifier 3': {'in_service': 'false'},
                              'Lineup A rectifier 4': {'in_service': 'false'}})
    out = _post(client, p)
    vscs = _by_label(p, out['vscs'])
    for k in (1, 2):
        assert vscs[f'Lineup A rectifier {k}']['p_mw'] == pytest.approx(2.2, rel=1e-6)
        assert any(w.startswith(f"VSC 'Lineup A rectifier {k}' is at its rating, 2.2 MVA") for w in out['warnings'])
    v = _by_label(p, out['dcbuses'])['Lineup A 800 V']['vm_pu']
    assert 0.95 < v < 0.98
    row = _spec_rows()['Lineup A DC Store converter']
    droop = row['rated_mw'] * (1.0 - v) * v / (row['droop_percent'] / 100.0)
    store = _by_label(p, out['dcdcconverters'])['Lineup A DC Store converter']
    assert store['p_out_mw'] == pytest.approx(droop, rel=0.01)
    assert _by_label(p, out['dcdiodes'])['Shelf A1 diode from the catcher']['p_mw'] > 0.05


# --- The design findings, closed as variants: each the fix, against the finding ------------------

def test_variant_ssts_of_5_2_mw_are_a_true_two_plus_one(client):
    """
    Found in 20a: with one SST out the other two carry Hall B past their 5 MW.
    Rated 5.2 MW, the two carry it within their rating.
    """
    p = _variant(_payload(), {**{f'Hall B SST {k}': {'rect_rated_mw': '5.2', 'dcdc_rated_mw': '5.2'} for k in (1, 2, 3)},
                              'Hall B SST 1': {'rect_rated_mw': '5.2', 'dcdc_rated_mw': '5.2', 'in_service': 'false'}})
    ssts = _by_label(p, _post(client, p)['ssts'])
    for k in (2, 3):
        assert 5.0 < ssts[f'Hall B SST {k}']['p_mv_mw'] < 5.2


def test_variant_a_breaker_per_shelf_on_the_catcher_side_is_selective(client):
    """
    Found in 20b: the catcher's group breaker sat in every shelf fault's path.
    With a breaker per shelf on the catcher's side instead (its 3 kA trip, as
    the shelves' own), a fault on shelf A2 trips its two breakers only; the
    catcher stays on the other eleven shelves.
    """
    import copy
    from test_dc_fault import STUDY
    p = _payload()
    cell = {v.get('userFriendlyName'): v['name'] for v in p.values() if isinstance(v, dict) and 'name' in v}
    key = {v.get('userFriendlyName'): k for k, v in p.items() if isinstance(v, dict)}
    proto = p[key['Shelf A2 breaker']]
    for g in 'AB':
        del p[key[f'Catcher group {g} breaker']]
        for k in range(1, 7):
            b = copy.deepcopy(proto)
            b.update(name=f'catcher_brk_{g}{k}', id=f'catcher-brk-{g}{k}', userFriendlyName=f'Shelf {g}{k} catcher breaker',
                     element=cell[f'Shelf {g}{k} diode from the catcher'], bus=cell[f'Catcher group {g} 800 V'])
            p[b['name']] = b
    p['0'] = {**STUDY, 'fault_bus': cell['Shelf A2 800 V'], 'duration_ms': '60', 'frequency': '60'}
    out = _post(client, p)
    trip = float(proto['trip_current_ka'])
    trips = {b['label'] for b in out['dcfault']['breakers'] if b['ip_ka'] > trip}
    assert trips == {'Shelf A2 breaker', 'Shelf A2 catcher breaker'}
    assert all(not b['exceeds'] for b in out['dcfault']['breakers'])


def test_variant_lineup_transformers_at_10_percent_bring_the_lv_into_ieee_1584(client):
    """
    Found in 20b: the 0.48 kV lineups at 127-143 kA, past IEEE 1584's 106 kA.
    Their 8 MVA transformers at 10 % rather than 6 % bring them inside it.
    """
    p = _variant(_payload('sc_'), {t: {'vk_percent': '10', 'vkr_percent': '1'}
                                   for t in ('Lineup A transformer', 'Lineup B transformer', 'Catcher B transformer')})
    buses = _by_label(p, _post(client, p)['busbars'])
    for b in ('Lineup A 0.48 kV', 'Lineup B 0.48 kV', 'Catcher B 0.48 kV'):
        assert 60 < buses[b]['ikss_ka'] < 106, b


def test_variant_some_24_mw_of_grid_forming_storage_carries_the_backup(client):
    """
    Found in 20a: grid lost and turbine off, the substation storage carried
    some 6.7 of 20.5 MW. Four PCS of 7.5 MVA on 6 MWh stores at 1.5 C carry
    the whole site, each some 6.1 MW at the island's droop frequency.
    """
    payload = _variant(_payload('timeseries_'), {
        'Grid': {'in_service': 'false'}, 'Gas turbine': {'in_service': 'false'}, 'Bus tie': {'closed': 'true'},
        'Substation BESS transformer': {'sn_mva': '32'},
        **{f'Substation BESS PCS {k}': {'s_rated_mva': '7.5'} for k in range(1, 5)},
        **{f'Substation DC Store {k}': {'capacity_kwh': '6000', 'c_rate_discharge': '1.5', 'c_rate_charge': '1.5'}
           for k in range(1, 5)}})
    payload['0'].update(time_steps='2', time_step_s='60')
    mg = _post(client, payload)['microgrid']
    for st, load in zip(mg['steps'], _step_loads(payload, 60.0, 2)):
        assert st['converged'] and st['unserved_mw'] == pytest.approx(0.0, abs=1e-6)
        assert st['load_mw'] == pytest.approx(load, abs=1e-6)
    assert not any('ran short' in n for n in mg['notes'])


def test_variant_a_40_mva_estatcom_keeps_the_turbine_in_step(client):
    """
    Found in 20c: held to their 1.2 pu, the substation's PCS could not keep
    the turbine in step through IEEE 2800's dip at 245 kV. A 40 MVA eSTATCOM
    (its supercapacitors scaled with it, from 75 %) at the same limit does:
    the run reaches 10 s and the turbine's angle swings some 60 degrees.
    """
    k = 40.0 / 15.0
    p = _variant(_payload('tds_'), {'eSTATCOM': {'s_rated_mva': '40'}, 'eSTATCOM transformer': {'sn_mva': '44'},
                                    'eSTATCOM supercapacitors': {'c_f': str(8.8889 * k), 'p_rated_kw': str(15000 * k),
                                                                 'v0_percent': '75'}})
    p['0'].update(fault_enabled='false', tf='10', grid_voltage_profile='ieee2800', grid_voltage_start_s='1')
    out = _post(client, p)
    assert out['converged'] is True and out['time'][-1] == pytest.approx(10.0)
    delta, = (s for s in out['delta'] if s['name'] == 'Gas turbine')
    assert math.degrees(max(delta['values']) - delta['values'][0]) < 90
    est, = (r for r in out['pcs'] if r['label'] == 'eSTATCOM')
    assert est['stopped_s'] is None and est['current_limited_from_s'] is not None


def test_emt_variant_bus_stores_sized_for_their_bus_ride_through(client):
    """
    Found in 20c: through IEEE 2800's 0 pu, Lineup A's, the catcher's and
    Hall B's racks lost their 800 V - their stores' converters, 2.5 MW
    against some 6 MW, blocked at 0.8 pu, their batteries behind 62.5 mOhm
    sagging under the current. Converters of 7 MW (6 MW in Hall B) on banks
    of 5 mOhm hold every 800 V bus above 0.94 pu; only the SSTs block, their
    MV gone, and Hall B's stores carry it.
    """
    stores = ('Lineup A DC Store', 'Lineup B DC Store', 'Catcher B DC Store', 'Hall B DC Store 1', 'Hall B DC Store 2')
    p = _variant(_payload(), {
        **{f'{s} converter': {'rated_mw': '6' if s.startswith('Hall B') else '7'} for s in stores},
        **{s: {'r0_mohm': '5'} for s in stores}})
    p['0'] = {'typ': 'EmtStudy Parameters', 'user_email': 't@t', 'duration_ms': '100', 'frequency': '60',
              'grid_voltage_profile': 'ieee2800', 'grid_voltage_start_ms': '20'}
    emt = _post(client, p)['emt']
    assert all(l['t_lost_ms'] is None for l in emt['loads']), [l['label'] for l in emt['loads'] if l['t_lost_ms']]
    assert {b['label'] for b in emt['converters_blocked']} == {
        f'Hall B SST {k} {stage}' for k in (1, 2, 3) for stage in ('rectifier', 'DC/DC')}
    for b in emt['buses']:
        if b['label'].endswith('800 V'):
            assert b['v_min_pu'] > 0.94, b['label']
