"""
The EMT study's timed events and the loads and inverter behaviour they were added for (C003 Phase 3): a
grid-following SOFC inverter blocked and resumed, with its power ramped back; a load scaled; a load behind a
UPS riding off a fault and walking back in; over-frequency curtailment (LFSM-O) along its droop, suspended
while its inverter rides through a dip.
"""
import json

import pytest

from test_pcs import _der, _pcs, _two_buses


def _emt(**kw):
    return {'typ': 'EmtStudy Parameters', 'user_email': 't@t',
            **{k: (v if k == 'events' else str(v)) for k, v in kw.items()}}


def _post(client, quiet, request):
    with quiet():
        result = json.loads(client.post('/', json=request).get_data(as_text=True))
    assert 'emt' in result, result.get('message')
    return result['emt']


def _island(params, **load):
    """An island: a 1 MVA grid-forming battery on A; on B a 0.5 MVA grid-following SOFC at 0.4 MW and the 0.6 MW
    load LD (its fields updated by load)."""
    request = _two_buses(
        _der('Battery', 'ba', capacity_kwh=1000), _pcs('ga', 'a', 'ba', control='grid_forming', s_rated_mva=1.0),
        _der('SOFC', 'fc', p_rated_kw=500, p_set_kw=400), _pcs('p2', 'b', 'fc', s_rated_mva=0.5),
        grid=False, params=params)
    for v in request.values():
        if isinstance(v, dict) and v.get('name') == 'ld':
            v.update({k: str(x) for k, x in load.items()})
    return request


def _conv(emt):
    return {c['label']: c for c in emt['converters']}


def test_inverter_blocked_and_resumed_with_its_power_ramped_back(client, quiet):
    """Blocked at 30 ms and resumed at 60 ms over a 0.2 s ramp: at 160 ms it is about half way back."""
    emt = _post(client, quiet, _island(_emt(time_step_us=20, duration_ms=160, events=[
        {'t': 0.03, 'action': 'trip', 'element': 'P2'},
        {'t': 0.06, 'action': 'close', 'element': 'P2', 'ramp_s': 0.2}])))
    p2 = _conv(emt)['P2']
    assert p2['blocked_ms'] is None
    assert [r['blocks_ms'] for r in emt['resumed'] if r['label'] == 'P2'] == [[[30.0, 60.0]]]
    assert 0.35 < p2['p_end_mw'] / p2['p_start_mw'] < 0.6
    assert not emt['converters_blocked']


def test_load_scaled_by_an_event(client, quiet):
    """LD scaled to half at 30 ms: the grid-forming battery takes the change, the SOFC holds its power."""
    emt = _post(client, quiet, _island(_emt(time_step_us=20, duration_ms=200, events=[
        {'t': 0.03, 'action': 'scale', 'element': 'LD', 'value': 0.5}])))
    (ld,) = [x for x in emt['changing_loads'] if x['label'].startswith('LD')]
    assert ld['share_end'] == pytest.approx(0.5, rel=0.02)
    conv = _conv(emt)
    assert conv['P2']['p_end_mw'] == pytest.approx(conv['P2']['p_start_mw'], rel=0.02)
    assert conv['GA']['p_end_mw'] == pytest.approx(conv['GA']['p_start_mw'] - 0.3, abs=0.02)


def test_load_behind_a_ups_rides_off_a_fault_and_walks_back_in(client, quiet):
    """On the grid, a fault at B from 20 to 70 ms: LD behind its UPS (0.85 pu, floor 0, back over 0.2 s) goes onto
    its battery at the fault and is walking back in at 170 ms."""
    request = _two_buses(params=_emt(time_step_us=20, duration_ms=170, ac_fault_bus='b', ac_fault_type='abcg',
                                     ac_fault_time_ms=20, ac_fault_duration_ms=50, ac_fault_resistance_ohm=0.01))
    request['4'].update({'uv_ease_pu': '0.85', 'uv_floor_pu': '0', 'uv_restore_s': '0.2'})
    emt = _post(client, quiet, request)
    (ld,) = [x for x in emt['changing_loads'] if x['label'].startswith('LD')]
    assert 20 <= ld['ups_from_ms'] < 35
    assert ld['ups_min'] == 0
    assert 0.2 < ld['ups_end'] < 0.6


@pytest.mark.parametrize('lfsm', [False, True])
def test_over_frequency_curtailment_along_its_droop(client, quiet, lfsm):
    """
    LD cut to a sixth at 30 ms: the island's surplus drives the grid-forming battery's frequency up its droop.
    Without LFSM-O it settles near 50.5 Hz; with a 2 % droop from 50.2 Hz the SOFC cuts its power to the point
    on its line, k = 1 - (f - 50.2) / (0.02 x 50), and the frequency settles lower.
    """
    params = _emt(time_step_us=20, duration_ms=400, events=[{'t': 0.03, 'action': 'scale', 'element': 'LD', 'value': 1 / 6}])
    if lfsm:
        params.update({'gfl_lfsm_droop_pct': '2', 'gfl_lfsm_f_start_hz': '50.2'})
    emt = _post(client, quiet, _island(params))
    conv = _conv(emt)
    ga, p2 = conv['GA'], conv['P2']
    if not lfsm:
        assert 'lfsm_end_pu' not in p2
        assert ga['f_end_hz'] > 50.4
        return
    assert ga['f_end_hz'] > 50.2
    assert p2['lfsm_end_pu'] == pytest.approx(1 - (p2['pll_f_max_hz'] - 50.2) / 1.0, abs=0.05)
    assert p2['lfsm_end_pu'] == pytest.approx(1 - (ga['f_end_hz'] - 50.2) / 1.0, abs=0.03)
    assert p2['lfsm_end_pu'] < 0.95
    assert p2['p_end_mw'] == pytest.approx(p2['lfsm_end_pu'] * p2['p_start_mw'], rel=0.05)


@pytest.mark.parametrize('hold_s', [0.0, 0.1])
def test_curtailment_suspended_through_a_dip(client, quiet, hold_s):
    """
    A fault at B from 20 to 70 ms with LFSM-O on. Below 0.85 pu the SOFC inverter does not curtail on its PLL's
    meaningless frequency; but as the voltage returns its PLL overshoots (to about 52.8 Hz), and without a hold
    it curtails on that. Held 0.1 s from the voltage's return, as from a resume, it never curtails.
    """
    emt = _post(client, quiet, _island(_emt(time_step_us=20, duration_ms=200, gfl_lfsm_droop_pct=2,
                                            gfl_lfsm_f_start_hz=50.2, gfl_lfsm_resume_hold_s=hold_s,
                                            ac_fault_bus='b', ac_fault_type='abcg', ac_fault_time_ms=20,
                                            ac_fault_duration_ms=50, ac_fault_resistance_ohm=0.05)))
    p2 = _conv(emt)['P2']
    if hold_s:
        assert p2['lfsm_min_pu'] == pytest.approx(1.0, abs=1e-6)
    else:
        # its filtered frequency's range is kept only while LFSM-O acts, held off through the dip
        assert p2['pll_f_max_hz'] > 51.5
        assert p2['lfsm_min_pu'] < 0.5 and p2['lfsm_min_at_ms'] > 70
    assert p2['lfsm_end_pu'] == pytest.approx(1.0, abs=1e-6)
