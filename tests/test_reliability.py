"""
The reliability study (reliability_electrisim): contingency enumeration over the protection zones.

A small island worked by hand: two 33 kV boards joined by a breaker; on the first, a 5 MW grid-forming battery
station (0.254 MWh above its set-point) and a 10 MW SOFC group; on the second, a second 10 MW group and 15 MW of IT.
Each source behind its own transformer and breaker. Then the C003 rev C.2 reference through the app's route.
"""
import base64
import contextlib
import io
import json
import os

import openpyxl
import pytest

import reliability_electrisim as rel

REF = os.path.join(os.path.dirname(__file__), 'reference')
H_REST = 5 / 60


def _island():
    bus = lambda b, kv=33.0: {'id': b, 'v_nom_kv': kv}  # noqa: E731
    cb = lambda i, f, t: {'id': i, 'from_bus': f, 'to_bus': t, 'normally_closed': True}  # noqa: E731
    tx = lambda i, hv, lv: {'id': i, 'hv_bus': hv, 'lv_bus': lv}  # noqa: E731
    return {
        'id': 'TWO_BOARDS', 'buses': [bus('MV1'), bus('MV2'), bus('ST_HV'), bus('G1_HV'), bus('G2_HV'),
                                      bus('ST_LV', 0.69), bus('G1_LV', 0.69), bus('G2_LV', 0.69)],
        'branches': [], 'switches': [cb('CB_ST', 'ST_HV', 'MV1'), cb('CB_G1', 'G1_HV', 'MV1'),
                                     cb('CB_G2', 'G2_HV', 'MV2'), cb('CB_12', 'MV1', 'MV2')],
        'transformers': [tx('T_ST', 'ST_HV', 'ST_LV'), tx('T_G1', 'G1_HV', 'G1_LV'), tx('T_G2', 'G2_HV', 'G2_LV')],
        'converters': [{'id': 'PCS1', 'bus': 'ST_LV', 's_rated_mva': 5.0, 'control': 'grid_forming_vsm'},
                       {'id': 'G1', 'bus': 'G1_LV', 's_rated_mva': 10.0, 'control': 'grid_following'},
                       {'id': 'G2', 'bus': 'G2_LV', 's_rated_mva': 10.0, 'control': 'grid_following'}],
        'machines': [], 'loads': [{'id': 'IT', 'bus': 'MV2', 'p_mw': 15.0, 'critical': True}],
        'islands': [{'id': 'NORMAL', 'open_switches': [], 'grid_connected': False,
                     'energized_source_ids': ['PCS1', 'G1', 'G2']}],
    }


PARAMS = {'storage_mwh': 0.254, 'p_cold': 0.0, 'it_loads': ['IT']}


def _events(**kw):
    events, zones, p = rel.enumerate_events(_island(), {**PARAMS, **kw})
    return events, {z['id']: z for z in zones}, p


def test_zones_are_cut_by_the_breakers():
    _, zones, _ = _events()
    kinds = {tuple(z['buses']): (z['kind'], tuple(z['members'])) for z in zones.values()}
    assert kinds == {('MV1',): ('87B', ('CB_12', 'CB_G1', 'CB_ST')), ('MV2',): ('87B', ('CB_12', 'CB_G2')),
                     ('ST_HV',): ('87T', ('CB_ST',)), ('G1_HV',): ('87T', ('CB_G1',)), ('G2_HV',): ('87T', ('CB_G2',))}


def test_a_group_lost_the_station_bridges_until_its_storage_runs_out():
    """Group 1 lost: 10 MW sustains 15 MW of IT, 5 MW short - the station bridges it from 0.254 MWh for the 5 min
    of restoration (0.417 MWh needed: 0.163 MWh not served), then nothing bridges the 47.9 h to the repair."""
    events, _, p = _events()
    (e,) = [x for x in events if x['class'] == 'zone' and x['zone'] == 'G1_HV']
    assert e['freq'] == pytest.approx(p['lam_zone'] + p['lam_tx'])
    assert e['it_mw'] == pytest.approx(5.0)
    assert e['it_hot'] == pytest.approx(5 * H_REST - 0.254 + 5 * (48 - H_REST))


def test_an_island_with_no_forming_source_is_dead():
    """The station's zone lost: the two groups cannot form, so all the IT goes, for the repair."""
    events, _, _ = _events()
    (e,) = [x for x in events if x['class'] == 'zone' and x['zone'] == 'ST_HV']
    assert e['it_mw'] == pytest.approx(15.0)
    assert e['it_hot'] == pytest.approx(15.0 * 48)


def test_a_stuck_breaker_opens_the_zone_beyond_until_it_is_isolated():
    """Group 1's breaker sticks: breaker failure opens board 1 too, so the IT board is left with group 2 alone and
    no forming source - dead until restored; counted beyond the plain group fault."""
    events, _, p = _events()
    (e,) = [x for x in events if x['class'] == 'stuck' and x['zone'] == 'G1_HV' and x['detail'] == 'CB_G1']
    assert e['freq'] == pytest.approx((p['lam_zone'] + p['lam_tx']) * p['p_bf'])
    assert e['also'] == ['CB_12', 'CB_G1', 'CB_ST']
    assert e['it_hot'] == pytest.approx(15.0 * H_REST - (5 * H_REST - 0.254))


def test_maintenance_overlap_can_be_switched_off():
    on, _, _ = _events()
    off, _, _ = _events(include_maintenance=False)
    assert any(e['class'] == 'maint' for e in on)
    assert not any(e['class'] == 'maint' for e in off)


def test_two_group_faults_overlapping():
    """Both groups lost at once: each alone leaves 5 MW short once the storage is spent, both together 15 MW - the
    overlap adds 5 MW for the mean overlap of two 48 h repairs (24 h), at lam^2 (48 + 48) / 8760."""
    events, _, p = _events()
    (e,) = [x for x in events if x['class'] == 'overlap' and {x['zone'], x['detail']} == {'G1_HV', 'G2_HV'}]
    lam = p['lam_zone'] + p['lam_tx']
    assert e['freq'] == pytest.approx(lam * lam * 96 / 8760)
    assert e['it_mw'] == pytest.approx(5.0)
    assert e['it_hot'] == pytest.approx(5.0 * 24)
    assert not any(x['class'] == 'overlap' for x in _events(include_second_order=False)[0])


def _with_a_tie():
    """The island with a third board: a second 5 MW battery station on MV3, joined to the IT board by a
    normally-open tie."""
    site = _island()
    site['buses'] += [{'id': 'MV3', 'v_nom_kv': 33.0}, {'id': 'ST2_HV', 'v_nom_kv': 33.0}, {'id': 'ST2_LV', 'v_nom_kv': 0.69}]
    site['switches'] += [{'id': 'CB_ST2', 'from_bus': 'ST2_HV', 'to_bus': 'MV3', 'normally_closed': True},
                         {'id': 'CB_23', 'from_bus': 'MV2', 'to_bus': 'MV3', 'normally_closed': False}]
    site['transformers'].append({'id': 'T_ST2', 'hv_bus': 'ST2_HV', 'lv_bus': 'ST2_LV'})
    site['converters'].append({'id': 'PCS2', 'bus': 'ST2_LV', 's_rated_mva': 5.0, 'control': 'grid_forming_vsm'})
    site['islands'][0]['energized_source_ids'].append('PCS2')
    return site


def test_a_tie_that_fails_to_close():
    """Board 1 faults: the IT board is left with group 2 and no forming source. Closing the tie to station 2 brings
    it back short only 5 MW; if the tie fails to close, all 15 MW stay out to the repair - 10 MW more for 48 h less
    the restoration, at lam_zone x p_fc."""
    events, _, p = rel.enumerate_events(_with_a_tie(), PARAMS)
    (e,) = [x for x in events if x['class'] == 'close' and x['zone'] == 'MV1']
    assert e['freq'] == pytest.approx(p['lam_zone'] * p['p_fc'])
    assert e['also'] == ['CB_23']
    assert e['it_hot'] == pytest.approx(10.0 * (48 - H_REST))


def test_the_workbook_rebuilds_each_frequency_from_the_inputs():
    result = rel.evaluate(_island(), PARAMS)
    wb = openpyxl.load_workbook(io.BytesIO(rel.workbook(result, 'two boards')))
    assert wb.sheetnames == ['Inputs', 'Events', 'Summary']
    ev = wb['Events']
    rows = {ev.cell(row=r, column=1).value: ev.cell(row=r, column=9).value for r in range(2, ev.max_row + 1)
            if ev.cell(row=r, column=1).value}
    assert rows['zone'].startswith('=(D') and 'Inputs!$B$' in rows['zone'] and 'ISNUMBER(H' in rows['stuck']
    assert ev.cell(row=2, column=13).value.startswith('=I2*((1-Inputs!')


def _evaluate_with(data):
    events, _, p = rel.enumerate_events(_island(), {**PARAMS, 'element_data': data})
    return {(e['class'], e['zone'], e['detail']): e for e in events}, p


def test_an_elements_own_rate_and_repair_replace_the_defaults():
    """Group 1's transformer at 0.05 /yr and 200 h to repair: its zone fails at the default bus section plus that
    rate, and stays out 200 h; a bus with its own rate replaces the zone's default bus section."""
    ev, p = _evaluate_with({'T_G1': {'rate': 0.05, 'repair_h': 200}, 'MV2': {'rate': 0.004}})
    g1 = ev[('zone', 'G1_HV', '87T')]
    assert g1['freq'] == pytest.approx(p['lam_zone'] + 0.05)
    assert (g1['own_rate'], g1['bus_default'], g1['tx_default']) == (0.05, 1, 0)
    assert g1['it_hot'] == pytest.approx(5 * H_REST - 0.254 + 5 * (200 - H_REST))
    assert ev[('zone', 'MV2', '87B')]['freq'] == pytest.approx(0.004)


def test_a_breakers_own_failure_to_open():
    ev, p = _evaluate_with({'CB_G1': {'p_bf': 0.02}})
    lam = p['lam_zone'] + p['lam_tx']
    assert ev[('stuck', 'G1_HV', 'CB_G1')]['freq'] == pytest.approx(lam * 0.02)
    assert ev[('stuck', 'MV1', 'CB_ST')]['freq'] == pytest.approx(p['lam_zone'] * p['p_bf'])


def test_the_workbook_carries_the_own_rates():
    result = rel.evaluate(_island(), {**PARAMS, 'element_data': {'T_G1': {'rate': 0.05}, 'CB_G1': {'p_bf': 0.02}}})
    ev = openpyxl.load_workbook(io.BytesIO(rel.workbook(result)))['Events']
    rows = [[ev.cell(row=r, column=c).value for c in range(1, 9)] for r in range(2, ev.max_row + 1)]
    g1 = [r for r in rows if r[0] == 'zone' and r[1] == 'G1_HV']
    stuck = [r for r in rows if r[0] == 'stuck' and r[2] == 'CB_G1']
    assert g1 and g1[0][3:7] == [0.05, 1, 0, 0.0]
    assert stuck and stuck[0][7] == 0.02


def _c003_study(**params):
    pay = json.load(open(os.path.join(REF, 'reference_c003_revc2.diagram_payload.json'), encoding='utf-8'))
    site = json.load(open(os.path.join(REF, 'c003_revC2_site_corrected.json'), encoding='utf-8'))
    key = next(k for k, v in pay.items() if isinstance(v, dict) and 'Parameters' in str(v.get('typ', '')))
    pay[key] = {'typ': 'ReliabilityElectrisim Parameters', 'user_email': 't@t', **{k: str(v) for k, v in params.items()}}
    return pay, site


def test_c003_reference_through_the_route(client, quiet):
    """C003 rev C.2 in its NORMAL state: every event's cells are the drawing's own; the workbook comes back."""
    pay, site = _c003_study(transfer_groups='IT_cordA+IT_cordB')
    with quiet():
        out = json.loads(client.post('/', json={**pay, 'mgtwin_state': {'site': site, 'state_id': 'NORMAL'}})
                         .get_data(as_text=True))
    assert out['error'] is False and out['state'] == 'NORMAL'
    assert out['zones'] == 46
    assert out['by_class']['blackout']['it_mwh_yr'] == pytest.approx(0.05 * (1 - 0.98 ** 3) * 25.0 * 2.0)
    assert out['it_mwh_yr'] == pytest.approx(0.1532, abs=0.002)
    ids = {v.get('id') for v in pay.values() if isinstance(v, dict)}
    assert all(c in ids for e in out['events'] for c in e['cells'])
    assert any(e['cells'] for e in out['events'])
    wb = openpyxl.load_workbook(io.BytesIO(base64.b64decode(out['workbook_xlsx_b64'])))
    ev = wb['Events']
    classes = [ev.cell(row=r, column=1).value for r in range(2, ev.max_row + 1)]
    assert sum(c in ('zone', 'stuck', 'relay', 'maint', 'close', 'overlap', 'blackout') for c in classes) == len(out['events'])


def _two_designs(capex_b=1.0):
    """The island as drawn, and the same with twice the storage at a capex: B loses less."""
    return [{'name': 'A', 'site': _island(), 'state': 'NORMAL', 'capex_meur': 0.0, 'params': {}},
            {'name': 'B', 'site': _island(), 'state': 'NORMAL', 'capex_meur': capex_b, 'params': {'storage_mwh': 0.508}}]


def test_designs_on_one_table():
    out = rel.compare(_two_designs(), PARAMS, crf=0.08)
    a, b = out['designs']
    assert b['it_mwh_yr'] < a['it_mwh_yr']
    assert b['capex_keur_yr'] == pytest.approx(80.0)
    assert b['total_vs_base_keur_yr'] == pytest.approx(b['cost_keur_yr'] + 80.0 - a['cost_keur_yr'])
    assert b['break_even_capex_meur'] == pytest.approx((a['cost_keur_yr'] - b['cost_keur_yr']) / 80.0)
    assert out['ranking'] == ['A', 'B'] and (a['rank'], b['rank']) == (1, 2)


def test_a_sweep_finds_where_two_designs_tie():
    """Swept on the capital recovery factor - pricing only: B's saving pays its annualised capex at
    crf = saving / (capex x 1000), and the ranking turns there."""
    base = rel.compare(_two_designs(), PARAMS, crf=0.08)['designs']
    saving = base[0]['cost_keur_yr'] - base[1]['cost_keur_yr']
    out = rel.compare(_two_designs(), PARAMS, {'param': 'crf', 'values': [0.0, 0.08]}, crf=0.08)
    (tp,) = out['sweep']['tie_points']
    assert tp['value'] == pytest.approx(saving / 1000)
    assert (tp['below'], tp['above']) == ('B', 'A')
    assert [r['ranking'] for r in out['sweep']['rows']] == [['B', 'A'], ['A', 'B']]


def test_a_swept_input_holds_for_every_design():
    out = rel.compare(_two_designs(), PARAMS, {'param': 'storage_mwh', 'values': [0.1, 1.0]}, crf=0.08)
    for row in out['sweep']['rows']:
        assert row['designs']['A']['it_mwh_yr'] == pytest.approx(row['designs']['B']['it_mwh_yr'])


def test_sweep_values():
    assert rel.sweep_values('storage_mwh', '', 0.254) == pytest.approx([0.127, 0.254, 0.508])
    assert rel.sweep_values('lam_tx', '0.02, x0.5, 0.01', 0.01) == pytest.approx([0.005, 0.01, 0.02])
    assert rel.sweep_values('dup_87', 'anything', True) == [0.0, 1.0]


def test_c003_reference_compared_and_swept_through_the_route(client, quiet):
    """C003 rev C.2 beside a copy of itself at a capex, the storage swept: the copy loses what the base loses, its
    break-even capex is nil; the workbook gains its Comparison and Sweep sheets, the base's energy from Summary."""
    pay, site = _c003_study(transfer_groups='IT_cordA+IT_cordB', sweep_param='storage_mwh', sweep_values='x1, x4',
                            crf=0.1, capex_meur=0)
    copy = {k: v for k, v in pay.items() if 'Parameters' not in str(v.get('typ', '') if isinstance(v, dict) else '')}
    key = next(k for k, v in pay.items() if isinstance(v, dict) and 'Parameters' in str(v.get('typ', '')))
    pay[key]['designs'] = [{'name': 'Copy', 'capex_meur': 2.0, 'network': copy, 'own_inputs': 'black_start_steps=3; transfer_groups=IT_cordA+IT_cordB'}]
    with quiet():
        out = json.loads(client.post('/', json={**pay, 'mgtwin_state': {'site': site, 'state_id': 'NORMAL'}})
                         .get_data(as_text=True))
    assert out['error'] is False
    cmp_ = out['comparison']
    base, other = cmp_['designs']
    assert base['it_mwh_yr'] == pytest.approx(out['it_mwh_yr'])
    assert other['name'] == 'Copy' and other['capex_keur_yr'] == pytest.approx(200.0)
    assert other['it_mwh_yr'] == pytest.approx(base['it_mwh_yr']) and other['notes'] == base['notes'] == []
    assert other['break_even_capex_meur'] == pytest.approx(0.0, abs=1e-9)
    rows = cmp_['sweep']['rows']
    assert [r['value'] for r in rows] == pytest.approx([0.254, 1.016])
    assert rows[1]['designs']['Copy']['it_mwh_yr'] <= rows[0]['designs']['Copy']['it_mwh_yr']
    wb = openpyxl.load_workbook(io.BytesIO(base64.b64decode(out['workbook_xlsx_b64'])))
    assert wb.sheetnames == ['Inputs', 'Events', 'Summary', 'Comparison', 'Sweep']
    assert wb['Comparison']['C5'].value == '=Summary!$C$11'
    assert wb['Comparison']['J6'].value.startswith('=RANK(H6')


def test_a_design_whose_inputs_name_none_of_its_loads_is_flagged():
    site = _island()
    assert rel._design_notes(site, {'transfer_groups': [['X_A', 'X_B']], 'it_loads': ['X_A']}) == [
        'none of the IT loads named is on it',
        'its dual-cord pairs name none of its loads: no cord moves to its partner']
    assert rel._design_notes(site, {'it_loads': ['IT']}) == []


def test_monte_carlo_with_flat_profiles_is_the_enumeration():
    """No profile, fixed repairs, storage back at once: the expectation is the enumeration's, exactly, and the
    simulated mean falls within four standard errors of it; the same seed gives the same years."""
    a = rel.monte_carlo(_island(), PARAMS, years=20000, seed=7, repair='fixed')
    assert a['expected']['it_mwh_yr'] == pytest.approx(a['analytic']['it_mwh_yr'], rel=1e-9)
    assert a['expected']['aux_mwh_yr'] == pytest.approx(a['analytic']['aux_mwh_yr'], rel=1e-9, abs=1e-12)
    sim = a['simulated']
    assert abs(sim['it_mwh_yr'] - a['analytic']['it_mwh_yr']) < 4 * sim['it_se']
    assert rel.monte_carlo(_island(), PARAMS, years=2000, seed=7)['simulated'] == \
        rel.monte_carlo(_island(), PARAMS, years=2000, seed=7)['simulated']
    assert sum(b['years'] for b in sim['histogram']) == 20000


def test_monte_carlo_profiles_move_the_expectation():
    """The IT at half its drawn load every hour is the enumeration of a half-loaded island; the storage empty every
    hour is the enumeration without storage."""
    half = _island()
    half['loads'][0]['p_mw'] = 7.5
    mc = rel.monte_carlo(_island(), PARAMS, years=10, it_profile=[0.5] * 24)
    assert mc['expected']['it_mwh_yr'] == pytest.approx(rel.evaluate(half, PARAMS)['it_mwh_yr'], rel=1e-9)
    mc = rel.monte_carlo(_island(), PARAMS, years=10, storage_profile=[0.0])
    assert mc['expected']['it_mwh_yr'] == pytest.approx(rel.evaluate(_island(), {**PARAMS, 'storage_mwh': 0.0})['it_mwh_yr'],
                                                        rel=1e-9)


def test_monte_carlo_storage_spent_by_one_event_is_missing_for_the_next():
    """Faults often enough to follow one another, the same years drawn twice (one seed): with the storage never
    recharged, every event after the first meets it empty, so the years lose more than with it back at once - by
    what the storage is worth in the enumeration (without storage less with it), within a few per cent."""
    often = {**PARAMS, 'lam_tx': 2.0, 'lam_zone': 0.5, 'include_maintenance': False, 'include_second_order': False}
    worth = rel.evaluate(_island(), {**often, 'storage_mwh': 0.0})['it_mwh_yr'] - rel.evaluate(_island(), often)['it_mwh_yr']
    assert worth > 1.0
    never = rel.monte_carlo(_island(), often, years=3000, seed=3, repair='fixed', recharge_h=1e12)['simulated']
    at_once = rel.monte_carlo(_island(), often, years=3000, seed=3, repair='fixed', recharge_h=0)['simulated']
    assert never['it_mwh_yr'] - at_once['it_mwh_yr'] == pytest.approx(worth, rel=0.05)


def test_c003_reference_monte_carlo_through_the_route(client, quiet):
    """C003 rev C.2 with a year-long Monte Carlo, the IT on a daily profile pasted as numbers: the enumeration it
    reports is the study's own, the profile lowers the expectation (the IT below its drawn load at night), and the
    workbook gains its Monte Carlo sheet."""
    day = ' '.join(['0.8'] * 8 + ['1.0'] * 16)
    pay, site = _c003_study(transfer_groups='IT_cordA+IT_cordB', mc_years=2000, mc_seed=5, mc_it_profile=day,
                            mc_recharge_h=2)
    with quiet():
        out = json.loads(client.post('/', json={**pay, 'mgtwin_state': {'site': site, 'state_id': 'NORMAL'}})
                         .get_data(as_text=True))
    assert out['error'] is False
    mc = out['monte_carlo']
    assert mc['years'] == 2000 and mc['recharge_h'] == 2.0
    assert mc['analytic']['it_mwh_yr'] == pytest.approx(out['it_mwh_yr'], rel=1e-9)
    assert mc['expected']['it_mwh_yr'] < mc['analytic']['it_mwh_yr']
    wb = openpyxl.load_workbook(io.BytesIO(base64.b64decode(out['workbook_xlsx_b64'])))
    assert wb.sheetnames[-1] == 'Monte Carlo'
