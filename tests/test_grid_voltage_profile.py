"""
The grid's voltage following a profile in the dynamic studies: IEEE 2800's
low-voltage ride-through envelope (GE Vernova's 800 V AI factory paper,
Figure 7), or a table - in ANDES as steps of the External Grid slack's set
voltage, in EMT as steps of its source's electromotive force. And the EMT
study at 60 Hz, which ran at 50 Hz whatever the network.
"""
import json

import numpy as np
import pytest

import grid_voltage_profile as gvp
from test_dc_elements import _drawn_request
from test_reference_grids import ANDES_PARAMS, _study_request


def _post(client, quiet, request):
    with quiet():
        out = json.loads(client.post('/', json=request).get_data(as_text=True))
    assert not out.get('error'), out.get('message')
    return out


def test_profile_points():
    """The preset; a table by newline or semicolon, sorted; what is wrong with one named."""
    assert gvp.profile_points({})[0] == []
    assert gvp.profile_points({'grid_voltage_profile': 'ieee2800'})[0] == [
        (0.0, 0.0), (0.32, 0.25), (1.2, 0.5), (3.0, 0.7), (6.0, 0.9)]
    points, problems = gvp.profile_points({'grid_voltage_profile': 'custom',
                                           'grid_voltage_table': '0.5, 1.0; 0, 0.4\n0.2\t0.6'})
    assert points == [(0.0, 0.4), (0.2, 0.6), (0.5, 1.0)] and not problems
    _, problems = gvp.profile_points({'grid_voltage_profile': 'custom', 'grid_voltage_table': '0, low; -1, 0.5'})
    assert len(problems) == 2 and 'line 1' in problems[0] and 'negative' in problems[1]
    assert 'not known' in gvp.profile_points({'grid_voltage_profile': 'brownout'})[1][0]


def test_andes_grid_follows_ieee_2800(client, quiet):
    """
    The transmission grid's 110 kV supply through IEEE 2800's envelope from
    1 s: its bus at each step's share of the grid's 1.02 pu, the first floored
    at 0.05 pu (ANDES's converters draw P / V); the machines ride it through.
    """
    request = _study_request('reference_transmission', {
        'typ': 'TransientStabilityAndes Parameters', **ANDES_PARAMS, 'tf': '9', 'tstep': '0',
        'fault_enabled': 'false', 'fault_bus': '', 'toggle_line': '', 'toggle_gen': '',
        'grid_voltage_profile': 'ieee2800', 'grid_voltage_start_s': '1'})
    out = _post(client, quiet, request)
    assert out['converged'] is True
    t = np.asarray(out['time'])
    v = np.asarray(next(b['values'] for b in out['bus_voltage'] if b['name'] == '110 kV busbar A'))
    for at, share in ((0.9, 1.0), (1.2, 0.05), (1.5, 0.25), (2.5, 0.5), (5.0, 0.7), (8.0, 0.9)):
        assert np.interp(at, t, v) == pytest.approx(1.02 * share, abs=2e-3), at
    assert any('below 0.05 pu' in w for w in out['warnings'])
    assert any('0.25 pu at 1.32 s' in w for w in out['warnings'])


def test_emt_grid_dips_and_recovers(client, quiet):
    """
    The radial grid's 110 kV source at half its voltage from 40 ms for 50 ms:
    its own bus follows it (the grid's current through its impedance aside),
    the network below it too, and all come back.
    """
    request = _drawn_request()
    request['0'] = {'typ': 'EmtStudy Parameters', 'user_email': 't@t', 'time_step_us': '20', 'duration_ms': '160',
                    'grid_voltage_profile': 'custom', 'grid_voltage_table': '0, 0.5; 0.05, 1.0',
                    'grid_voltage_start_ms': '40'}
    out = _post(client, quiet, request)
    buses = {b['label']: b for b in out['emt']['ac']['buses']}
    assert buses['110 kV supply']['v_rms_min_pu'] == pytest.approx(0.5, abs=0.01)
    assert 40 < buses['110 kV supply']['t_min_ms'] < 92
    for label in ('20 kV substation', 'B1', 'LV network A'):
        assert 0.4 < buses[label]['v_rms_min_pu'] < 0.6, label
        assert buses[label]['v_rms_final_pu'] > 0.9, label
    assert any('0.5 pu at 0.04 s' in w for w in out['warnings'])


@pytest.mark.parametrize('f_hz', [50, 60])
def test_emt_runs_at_the_frequency_asked(client, quiet, f_hz):
    """The EMT study's AC network at the dialog's frequency: it ran at 50 Hz with no field for it."""
    request = _drawn_request()
    request['0'] = {'typ': 'EmtStudy Parameters', 'user_email': 't@t', 'time_step_us': '20', 'duration_ms': '100',
                    'frequency': str(f_hz)}
    out = _post(client, quiet, request)
    w = next(b['waveform'] for b in out['emt']['ac']['buses'] if b['label'] == '110 kV supply')
    t, v = np.asarray(w['t_ms']), np.asarray(w['v_a_kv'])
    late = t > 20
    up = np.nonzero((v[late][:-1] < 0) & (v[late][1:] >= 0))[0]
    period_ms = np.diff(t[late][up]).mean()
    assert 1e3 / period_ms == pytest.approx(f_hz, rel=0.02)
