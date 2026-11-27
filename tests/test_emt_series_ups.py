"""
The series-reactor UPS in the EMT study: a 100 MW system at 33 kV, a 0.30 pu reactor
between its source and critical buses and a grid-forming PCS on the critical bus (its
battery at 60 kV: the PCS has no transformer here, so its DC link must reach 33 kV).
Against hand phasors for the reactor's steady state, and the mode sequence through
sags, a source loss and a converter trip.
"""
import json
import math

import pytest

from emt_series_ups import class1_check, curve_1_allowed_s
from test_pcs import _bus, _der, _pcs, GRID, LOAD, LINE

KV, MW, PF, X_PU, XR = 33.0, 100.0, 0.98, 0.30, 50.0
MVA = MW / PF
ZB = KV ** 2 / MVA


def _system(**params):
    x = X_PU * ZB
    q = MW * math.tan(math.acos(PF))
    return {'0': {'typ': 'EmtStudy Parameters', 'user_email': 't@t', 'series_ups_line': 'r', 'series_ups_pcs': 'conv',
                  **{k: str(v) for k, v in params.items()}},
            '1': _bus('src', KV), '2': _bus('crit', KV),
            '3': {**GRID, 'bus': 'src', 's_sc_max_mva': '20000', 's_sc_min_mva': '20000', 'rx_max': '0.1', 'rx_min': '0.1'},
            '4': {**LINE, 'name': 'r', 'id': 'cell-r', 'userFriendlyName': 'R', 'busFrom': 'src', 'busTo': 'crit', 'length_km': '1',
                  'r_ohm_per_km': str(x / XR), 'x_ohm_per_km': str(x), 'r0_ohm_per_km': str(x / XR), 'x0_ohm_per_km': str(x),
                  'max_i_ka': '6'},
            '5': {**LOAD, 'bus': 'crit', 'p_mw': str(MW), 'q_mvar': str(q)},
            '6': _der('Battery', 'b', sizing='ratings', vn_v=60000, capacity_kwh=50000, r0_mohm=1, soc_percent=60, soc_min_percent=5,
                      soc_max_percent=95, c_rate_discharge=20, c_rate_charge=20, coulombic_efficiency_percent=99, in_service='true'),
            '7': _pcs('conv', 'crit', 'b', control='grid_forming', s_rated_mva=round(MVA, 2), vn_ac_kv=KV, p_set_mw=0, q_mode='q',
                      q_set_mvar=0, pf=1, vm_set_pu=1, droop_pf_percent=1, droop_qv_percent=1, current_limit_pu=1.6, in_service='true')}


def _run(client, quiet, **params):
    params = {'time_step_us': 20, **params}
    with quiet():
        result = json.loads(client.post('/', json=_system(**params)).get_data(as_text=True))
    assert 'emt' in result, result.get('message')
    return result['emt']


def _modes(ups):
    return [m['mode'] for m in ups['modes']]


def _at(ups, mode):
    return next(m['t_ms'] for m in ups['modes'] if m['mode'] == mode)


def test_steady_state_on_grid_against_the_hand_phasor(client, quiet):
    """At 1.0 pu both sides the converter supplies the reactor's and load's reactive power, and no active power."""
    emt = _run(client, quiet, duration_ms=100)
    ups = emt['series_ups'][0]
    assert _modes(ups) == ['on_grid'] and ups['mode_end'] == 'on_grid'
    assert ups['v_critical_min_pu'] == pytest.approx(1.0, abs=0.01)
    assert ups['v_critical_max_pu'] == pytest.approx(1.0, abs=0.01)
    conv = emt['converters'][0]
    assert conv['p_end_mw'] == pytest.approx(0.0, abs=1.0)
    # The hand phasor: V = E = 1 pu, |P| = sin(delta) / X, Q at each end (1 - cos(delta)) / X, less the load's vars.
    x = X_PU
    delta = math.asin(0.98 * x)
    q_source = (1.0 - math.cos(delta)) / x
    q_conv = (q_source + math.tan(math.acos(PF)) * 0.98) * MVA
    assert conv['q_end_mvar'] == pytest.approx(q_conv, rel=0.05)


def test_a_shallow_sag_stays_on_grid_and_the_converter_supports_it(client, quiet):
    """A sag to 0.8 pu for 100 ms: no transfer, the critical bus well above the source."""
    emt = _run(client, quiet, duration_ms=200, grid_voltage_profile='custom', grid_voltage_table='0, 0.8; 0.1, 1', grid_voltage_start_ms=40)
    ups = emt['series_ups'][0]
    assert _modes(ups) == ['on_grid'] and not ups['openings']
    assert ups['v_source_min_pu'] == pytest.approx(0.8, abs=0.01)
    assert ups['v_critical_min_pu'] > 0.9


def test_a_deep_sag_transfers_at_current_zeros_and_comes_back(client, quiet):
    """To 0.3 pu for 150 ms: transfer in about a half cycle plus its zeros, then sync, close and hand back."""
    emt = _run(client, quiet, duration_ms=1000, series_ups_handback_s=0.2, grid_voltage_profile='custom',
               grid_voltage_table='0, 0.3; 0.15, 1', grid_voltage_start_ms=40)
    ups = emt['series_ups'][0]
    modes = _modes(ups)
    assert modes[:4] == ['on_grid', 'transfer', 'on_storage', 'sync'] and modes[4] == 'on_grid'
    assert 40.0 < _at(ups, 'transfer') < 40.0 + 15.0
    opening = ups['openings'][0]
    assert all(t is not None and t <= opening['t_start_ms'] + 20.0 for t in opening['t_phase_ms'])
    # Each phase opened at its zero: within a time step's change of current, far under the 2 kA it carried.
    assert all(i < 100.0 for i in opening['i_phase_a'])
    assert ups['closings'] and abs(ups['closings'][0]['angle_deg']) <= 10.0
    assert ups['v_critical_max_pu'] < 1.25
    assert not emt['converters_blocked']
    # The source has the load back and the converter has handed it over.
    conv = emt['converters'][0]
    assert abs(conv['p_end_mw']) < 0.2 * MW


def test_loss_of_the_source_stays_on_storage(client, quiet):
    """The source gone to the run's end: on storage, the critical bus held, the converter carrying the load."""
    emt = _run(client, quiet, duration_ms=500, grid_voltage_profile='custom', grid_voltage_table='0, 0.0', grid_voltage_start_ms=40)
    ups = emt['series_ups'][0]
    assert ups['mode_end'] == 'on_storage'
    conv = emt['converters'][0]
    assert conv['p_end_mw'] == pytest.approx(MW, rel=0.1)
    last = ups['trace']['v_critical_low_pu'][-1]
    assert last == pytest.approx(1.0, abs=0.05)


def test_a_converter_trip_goes_to_bypass(client, quiet):
    """The converter tripped on a healthy source: the bypass closes at once and the critical bus follows the source."""
    emt = _run(client, quiet, duration_ms=200, events=json.dumps([{'t': 0.05, 'action': 'trip', 'element': 'CONV'}]))
    ups = emt['series_ups'][0]
    assert ups['mode_end'] == 'bypass'
    assert _at(ups, 'bypass') == pytest.approx(50.0, abs=0.5)
    assert ups['trace']['v_critical_low_pu'][-1] == pytest.approx(ups['trace']['v_source_pu'][-1], abs=0.01)


@pytest.mark.parametrize('level, allowed_ms', [(0.30, 0.1), (0.22, 10.0), (0.14, 20.0), (0.13, 40.0), (0.115, 60.0), (0.105, 100.0)])
def test_curve_1_as_drawn(level, allowed_ms):
    assert curve_1_allowed_s(level) * 1e3 == pytest.approx(allowed_ms, rel=0.001)


def test_curve_1_is_unlimited_under_ten_percent():
    assert curve_1_allowed_s(0.099) == math.inf


@pytest.mark.parametrize('depth, ms, inside', [(0.25, 7.0, True), (0.25, 9.0, False), (0.13, 39.0, True), (0.13, 45.0, False),
                                               (0.09, 500.0, True), (0.35, 0.05, True), (0.35, 0.5, False)])
def test_class1_check_against_a_square_dip(depth, ms, inside):
    t = [k * 5e-5 for k in range(40000)]
    dev = [depth if 0.1 <= x < 0.1 + ms * 1e-3 else 0.0 for x in t]
    assert class1_check(t, {'dip': dev})['inside'] is inside


def test_a_steady_run_is_inside_class_1(client, quiet):
    ups = _run(client, quiet, duration_ms=100)['series_ups'][0]
    assert ups['class1']['inside'] and ups['class1']['worst'] is None


def test_a_source_fault_takes_the_critical_bus_outside_class_1(client, quiet):
    """A three-phase fault on the source bus: with the transfer's delay the critical bus leaves the curve."""
    ups = _run(client, quiet, duration_ms=120, ac_fault_bus='src', ac_fault_type='abcg', ac_fault_time_ms=40, ac_fault_duration_ms=60)['series_ups'][0]
    assert ups['mode_end'] == 'on_storage' or ups['closings']
    assert not ups['class1']['inside']


def test_a_sustained_backfeed_is_the_converter_limit_once_its_offset_is_out(client, quiet):
    """No transfer, the fault held: the symmetrical current into the fault is the converter's limit, less its load's."""
    ups = _run(client, quiet, duration_ms=200, series_ups_v_transfer_pu=0, ac_fault_bus='src', ac_fault_type='abcg',
               ac_fault_time_ms=40, ac_fault_duration_ms=0)['series_ups'][0]
    limit_ka = 1.6 * MVA / (math.sqrt(3.0) * KV)   # kA rms
    assert 0.5 * limit_ka < ups['backfeed_symmetrical_ka'] <= 1.05 * limit_ka
    assert ups['backfeed_peak_ka'] > ups['backfeed_symmetrical_ka'] * math.sqrt(2.0)


def test_forced_opening_is_faster_and_clamped(client, quiet):
    """Fast detection and a turn-off switch: all phases open together, sooner than the zeros allow."""
    sag = dict(grid_voltage_profile='custom', grid_voltage_table='0, 0.3; 0.15, 1', grid_voltage_start_ms=40)
    ups = _run(client, quiet, duration_ms=120, series_ups_detection='fast', series_ups_opening='forced', **sag)['series_ups'][0]
    opening = ups['openings'][0]
    assert len(set(opening['t_phase_ms'])) == 1 and opening['t_start_ms'] < 40.0 + 4.0
    assert max(opening['i_phase_a']) > 100.0
    assert _modes(ups)[:3] == ['on_grid', 'transfer', 'on_storage']


def test_retransfer_from_bypass(client, quiet):
    """The converter trips, then returns by an event: the disconnect closes, the bypass opens at its zeros, on grid."""
    events = json.dumps([{'t': 0.05, 'action': 'trip', 'element': 'CONV'}, {'t': 0.15, 'action': 'close', 'element': 'CONV'}])
    ups = _run(client, quiet, duration_ms=500, events=events)['series_ups'][0]
    assert _modes(ups) == ['on_grid', 'bypass', 'retransfer', 'on_grid']
    assert ups['openings'][-1]['what'] == 'bypass' and all(i < 100.0 for i in ups['openings'][-1]['i_phase_a'])
    assert ups['trace']['v_critical_low_pu'][-1] == pytest.approx(1.0, abs=0.03)


def test_a_dc_link_that_cannot_reach_the_critical_bus_is_named(client, quiet):
    request = _system(time_step_us=20, duration_ms=30)
    request['6']['vn_v'] = '1500'
    with quiet():
        result = json.loads(client.post('/', json=request).get_data(as_text=True))
    assert any('cannot reach the critical bus' in w for w in result['warnings'])


def test_without_a_grid_forming_pcs_the_reactor_is_a_plain_line(client, quiet):
    request = _system(time_step_us=20, duration_ms=60)
    request['0']['series_ups_pcs'] = 'nothing'
    with quiet():
        result = json.loads(client.post('/', json=request).get_data(as_text=True))
    assert result['emt']['series_ups'] == []
    assert any('Series-reactor UPS' in w for w in result['warnings'])
