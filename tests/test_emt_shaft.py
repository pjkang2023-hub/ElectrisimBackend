"""
The turbine-generator shaft (emt_shaft) against what has an answer: two masses' free torsional frequency,
sqrt(w0 K (1/2H1 + 1/2H2)) / 2 pi; the IEEE First Benchmark Model's shaft ringing at the frequencies of its own
spring and inertia matrices; its steady twist carrying the turbine's torque with nothing moving; and through the
study, a fault exciting its modes, reported with the torques each section reached.
"""
import json
import math

import numpy as np
import pytest

import emt_shaft
import test_emt_machine_park as T

W0 = 2.0 * math.pi * 60.0
# The IEEE First Benchmark Model's shaft (1977): HP, IP, LPA, LPB, generator, exciter.
FBM = {'h': [0.092897, 0.155589, 0.858670, 0.884215, 0.868495, 0.0342165], 'k': [19.303, 34.929, 52.038, 70.858, 2.822],
       'f': [0.30, 0.26, 0.22, 0.22, 0.0, 0.0], 'gen': 4}


def _ring(shaft, kick, t_end=2.0, dt=2e-5):
    """Free: no torque in or out, one mass given a push; the generator's speed over time."""
    shaft.w[kick] += 1e-3
    t, w = [], []
    for k in range(int(t_end / dt)):
        shaft.step(dt, 0.0, 0.0)
        t.append(k * dt)
        w.append(shaft.w[shaft.gen])
    return np.array(t), np.array(w)


def _peaks(t, y, n):
    y = (y - y.mean()) * np.hanning(len(y))
    spec = np.abs(np.fft.rfft(y, 16 * len(y)))
    f = np.fft.rfftfreq(16 * len(y), t[1] - t[0])
    found = []
    for k in np.argsort(spec)[::-1]:
        if f[k] > 2.0 and all(abs(f[k] - x) > 1.0 for x in found):
            found.append(float(f[k]))
        if len(found) == n:
            break
    return sorted(found)


def test_two_masses_ring_at_their_hand_frequency():
    shaft = emt_shaft.Shaft([1.0, 2.0], [20.0], [1.0, 0.0], 1, W0, 0.0)
    hand = math.sqrt(W0 * 20.0 * (1.0 / 2.0 + 1.0 / 4.0)) / (2.0 * math.pi)
    assert shaft.frequencies()[0] == pytest.approx(hand, rel=1e-9)
    t, w = _ring(shaft, 0)
    assert _peaks(t, w, 1)[0] == pytest.approx(hand, rel=0.005)


def test_the_benchmark_shaft_rings_at_its_matrices_frequencies():
    shaft = emt_shaft.Shaft(**FBM, w0=W0, tm0=0.0)
    modes = shaft.frequencies()
    assert len(modes) == 5
    # The published FBM modes are the system's (15.7, 20.2, 25.6, 32.3, 47.5 Hz); free of the grid, a little apart.
    assert [round(m) for m in modes] == pytest.approx([16, 20, 26, 32, 47], abs=2)
    t, w = _ring(shaft, 0, t_end=3.0)
    found = _peaks(t, w, 4)
    for x in found:
        assert min(abs(x - m) for m in modes) < 0.05 * x, (x, modes)


def test_it_starts_twisted_to_carry_its_load_with_nothing_moving():
    shaft = emt_shaft.Shaft(**FBM, w0=W0, tm0=0.8)
    # Each section carries what the turbine stages before it put in.
    assert shaft.torque0[:4] == pytest.approx(0.8 * np.cumsum([0.30, 0.26, 0.22, 0.22]), rel=1e-9)
    assert shaft.torque0[4] == pytest.approx(0.0, abs=1e-12)
    for _ in range(20000):
        shaft.step(2e-5, 0.8, 0.8)
    assert np.max(np.abs(shaft.w - 1.0)) < 1e-12


def test_through_the_study_a_fault_rings_the_shaft(client, quiet):
    shaft = {'dyn_shaft_H': ', '.join(map(str, FBM['h'])), 'dyn_shaft_K': ', '.join(map(str, FBM['k'])), 'dyn_shaft_F': ', '.join(map(str, FBM['f'])),
             'dyn_shaft_gen': '4'}
    request = T._request(duration_ms=600, ac_fault_bus='a', ac_fault_type='abc', ac_fault_time_ms=100, ac_fault_duration_ms=60, ac_fault_resistance_ohm=0.5)
    next(v for v in request.values() if v.get('typ') == 'Generator0').update(shaft)
    emt, warnings = T._emt(client, quiet, request)
    (m,) = emt['machines']
    assert m['shaft']['modes_hz'] == pytest.approx(emt_shaft.Shaft(**FBM, w0=2 * math.pi * T.F, tm0=0.0).frequencies(), rel=1e-6)
    start, peak = np.array(m['shaft']['torque_start_pu']), np.array(m['shaft']['torque_max_pu'])
    assert np.all(peak >= np.abs(start) - 1e-9) and peak[3] > 1.5 * abs(start[3])
