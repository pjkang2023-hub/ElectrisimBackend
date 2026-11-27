"""
AC breakers operated by the EMT study's timed events (params "events"): a bus coupler, or a switch at a line's end,
tripped - each pole opening at its own current's zero after the command, as a circuit breaker interrupts - and closed
again, all poles at once. Found benchmarking against PSCAD's tutorial `simpleac` (its breaker Brk1 opens at 0.26 s and
recloses at 0.31 s): before, an event naming a switch was dropped with a warning, so the breaker stayed closed.
"""
import json

import numpy as np
import pytest

F = 50.0
PERIOD_MS = 1e3 / F


def _bus(name, kv=20):
    return {'typ': 'Bus0', 'name': name, 'id': f'cell-{name}', 'userFriendlyName': name.upper(), 'vn_kv': str(kv)}


def _request(events, switch_et='b'):
    """A 20 kV grid at A, a 10 km line A-B, breaker CB from B to C (a bus coupler) or at the line's B end, 8 MW at C."""
    line = {'typ': 'Line0', 'name': 'l1', 'id': 'cell-l1', 'userFriendlyName': 'L1', 'busFrom': 'a', 'busTo': 'b',
            'length_km': '10', 'parallel': '1', 'df': '1', 'in_service': 'true', 'r_ohm_per_km': '0.1',
            'x_ohm_per_km': '0.35', 'c_nf_per_km': '10', 'g_us_per_km': '0', 'max_i_ka': '1', 'type': 'ol',
            'r0_ohm_per_km': '0.3', 'x0_ohm_per_km': '1.0', 'c0_nf_per_km': '5', 'endtemp_degree': '80'}
    grid = {'typ': 'External Grid0', 'name': 'g', 'id': 'cell-g', 'userFriendlyName': 'G', 'bus': 'a', 'vm_pu': '1',
            'va_degree': '0', 's_sc_max_mva': '500', 's_sc_min_mva': '500', 'rx_max': '0.1', 'rx_min': '0.1',
            'r0x0_max': '0.1', 'x0x_max': '1', 'r0x0_min': '0.1', 'x0x_min': '1', 'in_service': 'true'}
    load_bus = 'c' if switch_et == 'b' else 'b'
    load = {'typ': 'Load0', 'name': 'ld', 'id': 'cell-ld', 'userFriendlyName': 'LD', 'bus': load_bus, 'p_mw': '8',
            'q_mvar': '2', 'const_z_percent': '100', 'const_i_percent': '0', 'sn_mva': '0', 'scaling': '1',
            'type': 'wye', 'in_service': 'true'}
    if switch_et == 'b':
        cb = {'typ': 'Switch0', 'name': 'cb', 'id': 'cell-cb', 'userFriendlyName': 'CB', 'bus': 'b', 'element': 'c',
              'et': 'b', 'closed': 'true', 'type': 'CB'}
    else:
        cb = {'typ': 'Switch0', 'name': 'cb', 'id': 'cell-cb', 'userFriendlyName': 'CB', 'bus': 'b', 'element': 'l1',
              'et': 'l', 'closed': 'true', 'type': 'CB'}
    study = {'typ': 'EmtStudy Parameters', 'user_email': 't@t', 'frequency': str(F), 'time_step_us': '50',
             'duration_ms': '160', 'events': json.dumps(events)}
    rows = [_bus('a'), _bus('b'), line, grid, load, cb] + ([_bus('c')] if switch_et == 'b' else [])
    return {'0': study, **{str(k + 1): r for k, r in enumerate(rows)}}


def _emt(client, quiet, request):
    with quiet():
        out = json.loads(client.post('/', json=request).get_data(as_text=True))
    assert 'emt' in out, out.get('message')
    return out['emt'], out.get('warnings') or []


def _wave(emt, bus):
    w = next(b for b in emt['ac']['buses'] if b['name'] == bus)['waveform']
    return np.array(w['t_ms']), np.array([w[f'v_{p}_kv'] for p in 'abc'])


def _breaker_current(emt):
    br = next(b for b in emt['ac']['branches'] if b['kind'] == 'Breaker')
    return np.array(br['t_ms']), np.array([br[f'i_{p}_ka'] for p in 'abc'])


def _peak(t, v, t0, t1):
    m = (t >= t0) & (t <= t1)
    return float(np.max(np.abs(v[:, m])))


TRIP_CLOSE = [{'t': 0.04, 'action': 'trip', 'element': 'cb'}, {'t': 0.1, 'action': 'close', 'element': 'cb'}]


def test_a_tripped_bus_coupler_opens_each_pole_at_its_current_zero(client, quiet):
    emt, warnings = _emt(client, quiet, _request(TRIP_CLOSE))
    assert not [w for w in warnings if 'not applied' in w]
    (brk,) = emt['ac_breakers']
    trip, close = brk['operations']
    assert trip['action'] == 'trip' and trip['t_ms'] == pytest.approx(40.0)
    # each pole at its own zero: all within half a cycle of the command, and not all at once
    assert all(40.0 <= p <= 40.0 + PERIOD_MS / 2 + 0.1 for p in trip['poles_ms'])
    assert max(trip['poles_ms']) - min(trip['poles_ms']) > 1.0
    t, i = _breaker_current(emt)
    for k, opened in enumerate(trip['poles_ms']):
        # it was passing zero: 0.2 ms before, a 50 Hz current is within sin(3.6 deg) = 6 % of its peak
        assert abs(np.interp(opened - 0.2, t, i[k])) < 0.1 * np.max(np.abs(i[k, t < 40.0]))
        assert np.max(np.abs(i[k, (t > opened + 0.1) & (t < 99.0)])) < 1e-4            # and stayed open
    # the load bus dying away: only the current its inductance held decays, through its resistance (L / R = P / (Q w),
    # 13 ms)
    t, v = _wave(emt, 'c')
    assert _peak(t, v, 90.0, 99.0) < 0.03 * _peak(t, v, 0.0, 39.0)


def test_a_closed_bus_coupler_brings_its_load_back(client, quiet):
    emt, _ = _emt(client, quiet, _request(TRIP_CLOSE))
    close = emt['ac_breakers'][0]['operations'][1]
    assert close['action'] == 'close' and close['poles_ms'] == [100.0, 100.0, 100.0]
    t, v = _wave(emt, 'c')
    assert _peak(t, v, 120.0, 160.0) == pytest.approx(_peak(t, v, 0.0, 39.0), rel=0.02)


def test_a_switch_at_a_lines_end_trips_that_end_only(client, quiet):
    """The breaker at the line's B end: tripped, its load is cut off while the line stays energised from A, carrying
    only its own charging current."""
    emt, warnings = _emt(client, quiet, _request(TRIP_CLOSE[:1], switch_et='l'))
    assert not [w for w in warnings if 'not applied' in w]
    assert all(p is not None for p in emt['ac_breakers'][0]['operations'][0]['poles_ms'])
    t, v = _wave(emt, 'b')
    assert _peak(t, v, 130.0, 160.0) < 0.01 * _peak(t, v, 0.0, 39.0)
    t, va = _wave(emt, 'a')
    assert _peak(t, va, 80.0, 160.0) == pytest.approx(_peak(t, va, 0.0, 39.0), rel=0.05)


def test_without_events_a_coupler_is_still_a_plain_tie(client, quiet):
    emt, _ = _emt(client, quiet, _request([]))
    assert 'ac_breakers' not in emt
    assert not [b for b in emt['ac']['branches'] if b['kind'] == 'Breaker']
