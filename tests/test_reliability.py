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
    assert sum(c in ('zone', 'stuck', 'relay', 'maint', 'blackout') for c in classes) == len(out['events'])
