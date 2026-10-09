"""
The power swing's metrics in the transient stability study: peak to peak,
the steepest ramp, the strongest frequencies and the RMS in a band about a
machine's torsional mode - checked on sines whose answers are known.
"""
import math

import numpy as np
import pytest

import andes_electrisim as ae


def _sine(dt, amplitude=3.0, f=20.0, mean=10.0, tf=6.0):
    t = np.arange(0.0, tf + dt / 2, dt)
    return t, mean + amplitude * np.sin(2 * math.pi * f * t)


def test_a_sine_in_the_torsional_band():
    t, x = _sine(0.002)
    row = ae._swing_one(t, x, 1.0, (15.0, 25.0))
    assert row['mean_mw'] == pytest.approx(10.0, abs=1e-3)
    assert row['peak_to_peak_mw'] == pytest.approx(6.0, rel=0.01)       # sampled, 25 to a cycle
    assert row['max_ramp_mw_per_s'] == pytest.approx(2 * math.pi * 20.0 * 3.0, rel=0.01)
    assert row['dominant'][0]['f_hz'] == pytest.approx(20.0, abs=0.2)
    assert row['dominant'][0]['amplitude_mw'] == pytest.approx(3.0, rel=0.02)
    assert row['band_resolved']
    assert row['band_rms_mw'] == pytest.approx(3.0 / math.sqrt(2), rel=0.02)
    assert row['total_rms_mw'] == pytest.approx(3.0 / math.sqrt(2), rel=0.02)


def test_a_sine_outside_the_band():
    t, x = _sine(0.002, f=2.0)
    row = ae._swing_one(t, x, 1.0, (15.0, 25.0))
    assert row['dominant'][0]['f_hz'] == pytest.approx(2.0, abs=0.2)
    assert row['band_rms_mw'] < 0.01 * row['total_rms_mw']


def test_a_band_past_nyquist_is_not_seen():
    """A 25 ms step resolves 20 Hz: the band to 25 Hz is not screened, and says so rather than read 0."""
    t, x = _sine(0.025, f=2.0)
    row = ae._swing_one(t, x, 1.0, (15.0, 25.0))
    assert row['nyquist_hz'] == pytest.approx(20.0)
    assert not row['band_resolved'] and row['band_rms_mw'] is None


def test_the_swing_starts_where_asked():
    t = np.linspace(0.0, 5.0, 2001)
    x = np.where(t < 1.0, 100.0, 5.0)         # the run settling, then flat
    row = ae._swing_one(t, x, 1.0, (15.0, 25.0))
    assert row['peak_to_peak_mw'] == pytest.approx(0.0, abs=1e-9)
