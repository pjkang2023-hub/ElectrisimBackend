"""
A grid-forming PCS with a filter capacitor and a fast voltage loop in the EMT study, on the
double-conversion and series-reactor UPS test systems: its steady state, a load's sudden fall
against the same run without the loop, its current limit, a source fault and retransfer.
"""
import json

import pytest

import test_emt_double_conversion as DC
import test_emt_series_ups as SU

STEP = 0.1


def _ramp(share, ms=1.0, n=2, t=STEP):
    return [{'t': t + ms * 1e-3 * (k + 1) / n, 'action': 'scale', 'element': 'LD', 'value': 1 + (share - 1) * (k + 1) / n} for k in range(n)]


def _post(client, quiet, request):
    with quiet():
        result = json.loads(client.post('/', json=request).get_data(as_text=True))
    assert 'emt' in result, result.get('message')
    return result['emt']


def _double(client, quiet, loop=True, **params):
    if loop:
        params['gfm_voltage_loop_pcs'] = 'inv'
    return _post(client, quiet, DC._system(time_step_us=20, **params))


def _series(client, quiet, **params):
    return _post(client, quiet, SU._system(time_step_us=20, gfm_voltage_loop_pcs='conv', **params))


def _conv(emt, label='INV'):
    return next(c for c in emt['converters'] if c['label'] == label)


def test_it_starts_in_its_steady_state_and_stays_there(client, quiet):
    """The critical bus and the inverter's power hold from the first step, with no transient from the capacitor."""
    emt = _double(client, quiet, duration_ms=150)
    ups = emt['double_conversion'][0]
    assert ups['class1']['worst'] is None
    assert ups['v_critical_min_pu'] > 0.99 and ups['v_critical_max_pu'] < 1.02
    assert _conv(emt)['p_end_mw'] == pytest.approx(DC.MW, rel=0.02)
    assert _conv(emt)['limited_ms'] == 0.0
    assert not emt['converters_blocked']


def test_a_sudden_load_fall_kicks_the_bus_far_less_than_without_the_loop(client, quiet):
    """70 % of the load off within a millisecond: the bus's overshoot falls by more than half."""
    events = json.dumps(_ramp(0.3))
    without = _double(client, quiet, loop=False, duration_ms=200, events=events)['double_conversion'][0]
    looped = _double(client, quiet, duration_ms=200, events=events)['double_conversion'][0]
    assert looped['v_critical_max_pu'] - 1.0 < 0.5 * (without['v_critical_max_pu'] - 1.0)
    assert looped['class1']['worst']['deviation_peak_pct'] < without['class1']['worst']['deviation_peak_pct']


def test_it_comes_back_to_its_voltage_after_a_step(client, quiet):
    """The voltage settles within 2 % after the load's rise, with no oscillation left at the run's end."""
    events = json.dumps([{'t': 0.1, 'action': 'scale', 'element': 'LD', 'value': 0.5}, {'t': 0.2, 'action': 'scale', 'element': 'LD', 'value': 1.0}])
    ups = _double(client, quiet, duration_ms=450, events=events)['double_conversion'][0]
    lows, highs = ups['trace']['v_critical_low_pu'][-40:], ups['trace']['v_critical_high_pu'][-40:]
    assert min(lows) > 0.98 and max(highs) < 1.02


def test_it_holds_its_current_limit_at_a_critical_bus_fault(client, quiet):
    """A three-phase fault on the critical bus: the inverter stays at about its limit, unblocked."""
    emt = _double(client, quiet, duration_ms=200, ac_fault_bus='crit', ac_fault_type='abcg', ac_fault_time_ms=60, ac_fault_duration_ms=60, ac_fault_resistance_ohm=0.1)
    conv = _conv(emt)
    assert conv['limited_ms'] > 30.0
    assert conv['blocked_ms'] is None
    assert emt['double_conversion'][0]['trace']['v_critical_low_pu'][-1] > 0.9


def test_the_series_ups_rides_a_source_fault_and_comes_back(client, quiet):
    emt = _series(client, quiet, duration_ms=800, ac_fault_bus='src', ac_fault_type='abcg', ac_fault_time_ms=60, ac_fault_duration_ms=60)
    ups = emt['series_ups'][0]
    assert [m['mode'] for m in ups['modes']][:4] == ['on_grid', 'transfer', 'on_storage', 'sync']
    assert ups['mode_end'] == 'on_grid'
    assert not emt['converters_blocked']
    assert ups['trace']['v_critical_low_pu'][-1] == pytest.approx(1.0, abs=0.03)


def test_the_series_ups_retransfers_from_bypass(client, quiet):
    """Tripped and returned by events: at the bus's voltage with its integrators cleared, it takes over without a large jump."""
    events = json.dumps([{'t': 0.05, 'action': 'trip', 'element': 'CONV'}, {'t': 0.15, 'action': 'close', 'element': 'CONV'}])
    emt = _series(client, quiet, duration_ms=500, events=events)
    ups = emt['series_ups'][0]
    assert [m['mode'] for m in ups['modes']] == ['on_grid', 'bypass', 'retransfer', 'on_grid']
    assert ups['trace']['v_critical_low_pu'][-1] == pytest.approx(1.0, abs=0.03)
    assert ups['v_critical_max_pu'] < 1.25
