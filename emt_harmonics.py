"""
Harmonics from an EMT run's waveforms: each bus's voltage and each branch's current, by a Fourier
transform over its last whole cycles. The study's parameters: harmonics_cycles (its window; 0 or
unset, none) and harmonics_h_max. Each row reports the phase with the highest distortion.

A window of n cycles resolves f1 / n, so what falls between the harmonics is reported apart as
interharmonic distortion: a run not settled in its window shows there. THD is against the
fundamental, not against the rated current as IEEE 519's TDD would be.
"""
import math

import numpy as np


def analyse(t, y, f1, cycles=4, h_max=50):
    """One waveform's last ``cycles`` cycles: its fundamental, THD and harmonics, or None if the run is too short."""
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    span = cycles / f1
    if len(t) < 16 or t[-1] - t[0] < span * (1.0 - 1e-9):
        return None
    step = float(np.median(np.diff(t[t >= t[-1] - span])))
    n = int(round(span / max(step, 1e-9)))
    if n < 2 * cycles * (h_max + 1):
        h_max = max(n // (2 * cycles) - 1, 1)     # what the run's step resolves
    grid = t[-1] - span + span * np.arange(n) / n
    spec = np.fft.rfft(np.interp(grid, t, y)) / n
    amp = np.abs(spec) * math.sqrt(2.0)           # rms of each bin
    amp[0] = abs(spec[0])
    fund = float(amp[cycles])
    if fund <= 0.0:
        return {'fundamental_rms': 0.0, 'thd_percent': None, 'interharmonic_percent': None, 'dc': float(spec[0].real), 'harmonics': [], 'h_max': h_max}
    bins = np.arange(len(amp))
    top = min(cycles * h_max, len(amp) - 1)
    wanted = (bins >= cycles) & (bins <= top)
    whole = wanted & (bins % cycles == 0) & (bins != cycles)
    inter = wanted & (bins % cycles != 0)
    rows = [{'h': int(k // cycles), 'rms': float(amp[k]), 'percent': float(100.0 * amp[k] / fund)} for k in bins[whole]]
    return {'fundamental_rms': fund, 'thd_percent': float(100.0 * math.sqrt(np.sum(amp[whole] ** 2)) / fund),
            'interharmonic_percent': float(100.0 * math.sqrt(np.sum(amp[inter] ** 2)) / fund), 'dc': float(spec[0].real), 'harmonics': rows, 'h_max': h_max}


def worst_phase(t, phases, f1, cycles, h_max, scale, floor=0.05):
    """Of three waveforms, the one with the highest THD, reported in the scale's units, its small harmonics left out."""
    results = [analyse(t, y, f1, cycles, h_max) for y in phases]
    results = [(k, r) for k, r in enumerate(results) if r is not None and r['thd_percent'] is not None]
    if not results:
        return None
    k, r = max(results, key=lambda x: x[1]['thd_percent'])
    return {'phase': 'abc'[k], 'fundamental_rms': round(r['fundamental_rms'] * scale, 6), 'thd_percent': round(r['thd_percent'], 4),
            'interharmonic_percent': round(r['interharmonic_percent'], 4), 'dc': round(r['dc'] * scale, 6), 'h_max': r['h_max'],
            'harmonics': [{'h': x['h'], 'rms': round(x['rms'] * scale, 6), 'percent': round(x['percent'], 4)} for x in r['harmonics'] if x['percent'] >= floor]}
