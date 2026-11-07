"""
EMT harmonics against waveforms whose harmonics are known and against the impedance scan's own V = I Z;
the PV curve's first thermal limit against the exact answer; parameter sweeps of two kinds of study.
"""
import json
import math

import numpy as np
import pytest

import emt_harmonics as h
import test_emt_impedance_scan as S
import test_voltage_stability as V

F1 = 50.0


def _wave(amps, n=20000, cycles=10, offset=0.0, inter=None):
    t = np.arange(n) * cycles / F1 / n
    y = sum(a * math.sqrt(2.0) * np.cos(2 * math.pi * k * F1 * t + 0.3 * k) for k, a in amps.items()) + offset
    if inter:
        y = y + inter[1] * math.sqrt(2.0) * np.cos(2 * math.pi * inter[0] * t)
    return t, y


def test_known_harmonics_are_read_back():
    t, y = _wave({1: 100.0, 5: 4.0, 7: 3.0, 11: 1.0})
    r = h.analyse(t, y, F1, cycles=4)
    assert r['fundamental_rms'] == pytest.approx(100.0, rel=1e-3)
    assert r['thd_percent'] == pytest.approx(math.sqrt(16 + 9 + 1), rel=1e-3)
    rows = {x['h']: x['percent'] for x in r['harmonics']}
    assert rows[5] == pytest.approx(4.0, rel=1e-3) and rows[7] == pytest.approx(3.0, rel=1e-3)
    assert r['interharmonic_percent'] < 0.01


def test_a_tone_between_the_harmonics_is_reported_apart():
    t, y = _wave({1: 100.0}, inter=(62.5, 2.0))
    r = h.analyse(t, y, F1, cycles=4)
    assert r['thd_percent'] < 0.01
    assert r['interharmonic_percent'] == pytest.approx(2.0, rel=0.01)


def test_dc_is_not_distortion():
    t, y = _wave({1: 100.0}, offset=7.0)
    r = h.analyse(t, y, F1, cycles=4)
    assert r['dc'] == pytest.approx(7.0, rel=1e-3) and r['thd_percent'] < 0.01


def test_too_short_a_run_gives_nothing():
    t, y = _wave({1: 1.0}, n=200, cycles=2)
    assert h.analyse(t, y, F1, cycles=4) is None


def test_the_step_limits_the_harmonics_read():
    t, y = _wave({1: 100.0}, n=400, cycles=10)     # 40 samples a cycle: to about the 19th
    assert h.analyse(t, y, F1, cycles=4, h_max=50)['h_max'] < 20


def test_on_a_run_the_harmonics_are_the_scan_current_times_its_impedance(client, quiet):
    """The scan's tones through the grid: each harmonic of the bus's voltage is amp x Z(f) from the same run."""
    request = S._system(load=False, scan_f_min_hz=50, scan_f_max_hz=1000, scan_points_per_decade=10, scan_settle_ms=250, scan_amp_pu=0.05, harmonics_cycles=1, harmonics_h_max=20)
    with quiet():
        result = json.loads(client.post('/', json=request).get_data(as_text=True))
    emt = result['emt']
    scan, bus = emt['impedance_scan'], emt['harmonics']['buses'][0]
    assert bus['label'] == 'A'
    rows = {x['h']: x['rms'] for x in bus['harmonics']}
    amp_rms = scan['amp_ka_per_tone'] / math.sqrt(2.0)     # kA rms per tone
    checked = 0
    for f, z in zip(scan['f_hz'], scan['z_ohm']):
        k = int(round(f / S.F1))
        if abs(f / S.F1 - k) < 1e-6 and k > 1 and k in rows:
            assert rows[k] == pytest.approx(amp_rms * z, rel=0.03), f
            checked += 1
    assert checked >= 3


def test_it_warns_when_an_event_falls_in_the_window(client, quiet):
    request = S._system(load=False, duration_ms=100, ac_fault_bus='a', ac_fault_type='abcg', ac_fault_time_ms=90, ac_fault_duration_ms=5, ac_fault_resistance_ohm=1, harmonics_cycles=2)
    request['0'].pop('scan_bus')
    with quiet():
        result = json.loads(client.post('/', json=request).get_data(as_text=True))
    assert any('harmonics' in w.lower() and 'event' in w.lower() for w in result['warnings'])


# --- The PV curve's first thermal limit ---------------------------------------------------

def test_the_first_thermal_limit_is_the_exact_one(client, quiet):
    """A 0.5 kA line: the load at which its current reaches 0.5 kA on the exact upper branch."""
    request = V._system(10.0, 2.0, curve='pv', step=0.5, tolerance=0.01)
    next(v for v in request.values() if str(v.get('typ', '')).startswith('Line'))['max_i_ka'] = '0.5'
    r = V._post(client, quiet, request)
    limit = r['first_thermal_limit']
    best = None
    for p in np.linspace(5.0, 40.0, 3501):
        q = 0.2 * p
        c = 2.0 * q * V.X - V.KV ** 2
        v = math.sqrt((-c + math.sqrt(c * c - 4.0 * V.X ** 2 * (p * p + q * q))) / 2.0)
        if math.hypot(p, q) / (math.sqrt(3.0) * v) >= 0.5:
            best = p
            break
    assert limit['p_total_mw'] == pytest.approx(best, rel=0.02)
    assert limit['element'] == 'L1'
    assert limit['p_total_mw'] < r['nose']['p_total_mw']
    assert limit['margin_mw'] == pytest.approx(limit['p_total_mw'] - 10.0, rel=0.01)


def test_a_line_over_at_the_start_says_so(client, quiet):
    request = V._system(10.0, 2.0, curve='pv', step=0.5, tolerance=0.01)
    next(v for v in request.values() if str(v.get('typ', '')).startswith('Line'))['max_i_ka'] = '0.05'
    limit = V._post(client, quiet, request)['first_thermal_limit']
    assert limit['lambda'] == 1.0 and limit['already_over'] is True


# --- Sweeps ------------------------------------------------------------------------------

def _sweep(client, quiet, request, **sweep):
    request = dict(request)
    request['0'] = {'typ': 'ParameterSweep Parameters', 'user_email': 't@t', **sweep}
    with quiet():
        result = json.loads(client.post('/', json=request).get_data(as_text=True))
    return result['sweep']


def test_a_sweep_of_a_load_flow_study_against_the_exact_nose(client, quiet):
    """The line's reactance swept: each nose is E^2 / 2 X, through the application's own entry point."""
    request = V._system(20.0, 0.0)
    name = next(v for v in request.values() if str(v.get('typ', '')).startswith('Line'))['name']
    s = _sweep(client, quiet, request, study_typ='VoltageStabilityPandaPower Parameters', study_params={'curve': 'pv', 'step': '0.2', 'tolerance': '0.001'},
               axes=[{'element': name, 'field': 'x_ohm_per_km', 'values': [5.0, 10.0, 20.0]}], metrics=['voltage_stability.nose.p_total_mw'])
    assert s['n'] == 3 and s['n_failed'] == 0
    for row in s['rows']:
        x = row['values'][f'{name}.x_ohm_per_km']
        assert row['metrics']['voltage_stability.nose.p_total_mw'] == pytest.approx(V.KV ** 2 / (2.0 * x), rel=0.005)
    assert s['summary']['voltage_stability.nose.p_total_mw']['max'] == pytest.approx(V.KV ** 2 / 10.0, rel=0.005)


def test_every_combination_of_two_axes_runs_including_study_settings(client, quiet):
    request = V._system(20.0, 0.0)
    s = _sweep(client, quiet, request, study_typ='VoltageStabilityPandaPower Parameters', study_params=json.dumps({'curve': 'pv'}),
               axes=json.dumps([{'element': '@study', 'field': 'step', 'values': [0.5, 1.0]}, {'element': 'ld', 'field': 'p_mw', 'values': [10.0, 20.0, 30.0]}]),
               metrics=json.dumps(['voltage_stability.p_base_mw']))
    assert s['n'] == 6
    assert sorted({r['metrics']['voltage_stability.p_base_mw'] for r in s['rows']}) == pytest.approx([10.0, 20.0, 30.0], rel=1e-3)


def test_an_emt_sweep(client, quiet):
    """The scan's tone size swept: the impedance does not move, the run is the EMT's own."""
    request = S._system(load=False, scan_f_min_hz=100, scan_f_max_hz=500, scan_settle_ms=250)
    study = {k: v for k, v in request['0'].items() if k != 'typ'}
    s = _sweep(client, quiet, request, study_typ='EmtStudy Parameters', study_params=study, axes=[{'element': '@study', 'field': 'scan_amp_pu', 'values': [0.005, 0.01]}],
               metrics=['emt.impedance_scan.z_ohm.0'])
    a, b = [r['metrics']['emt.impedance_scan.z_ohm.0'] for r in s['rows']]
    assert a == pytest.approx(b, rel=0.005)


@pytest.mark.parametrize('sweep, word', [
    (dict(study_typ='VoltageStabilityPandaPower Parameters', axes=[{'element': 'nowhere', 'field': 'x', 'values': [1]}], metrics=['a']), 'nowhere'),
    (dict(study_typ='ParameterSweep Parameters', axes=[{'element': '@study', 'field': 'x', 'values': [1]}], metrics=['a']), 'sweep'),
    (dict(study_typ='VoltageStabilityPandaPower Parameters', axes=[{'element': '@study', 'field': 'x', 'values': list(range(30))}, {'element': '@study', 'field': 'y', 'values': list(range(30))}], metrics=['a']), 'limit'),
    (dict(axes=[], metrics=[]), 'study_typ'),
])
def test_a_bad_sweep_is_refused_by_name(client, quiet, sweep, word):
    s = _sweep(client, quiet, V._system(10.0, 0.0), **sweep)
    assert s['error'] is True and word in s['message']


def test_a_run_that_fails_does_not_stop_the_rest(client, quiet):
    """A load past any solution makes the study say its load flow does not solve; the sweep goes on."""
    s = _sweep(client, quiet, V._system(10.0, 0.0), study_typ='VoltageStabilityPandaPower Parameters', study_params={'curve': 'pv', 'step': '0.5'},
               axes=[{'element': 'ld', 'field': 'p_mw', 'values': [10.0, 500.0, 20.0]}], metrics=['voltage_stability.p_base_mw', 'voltage_stability.error'])
    assert s['n'] == 3
    assert s['rows'][0]['metrics']['voltage_stability.p_base_mw'] == pytest.approx(10.0, rel=1e-3)
    assert s['rows'][1]['metrics']['voltage_stability.error'] is True
    assert s['rows'][2]['metrics']['voltage_stability.p_base_mw'] == pytest.approx(20.0, rel=1e-3)
