"""
An impedance-based stability check of constant-power loads on a converter-fed
AC network (Middlebrook's criterion, Nyquist on the minor loop).

The source: the network's output impedance at the loads' bus, Z_out(s) =
R + s L - for grid-forming converters, their filter reactors and the
transformers behind them in parallel (the reactor bounds their impedance
from above; their voltage loops only lower it).

The loads: constant power within their power control's bandwidth. A converter
holding its output power draws more current as its voltage falls: its
incremental input admittance is a negative conductance, -P / V^2, out to its
control's bandwidth f_c (a first-order roll-off), and above it its input
filter's positive admittance, 1 / (s L_in), takes over.

The minor loop gain T(s) = Z_out(s) Y_in(s). The network and its loads are
stable together when T stays clear of -1: here |T| < 1 wherever its phase
reaches 180 deg. Over the frequencies where the loads are negative
resistances |T| ~ |Z_out| P / V^2 grows with Z_out's inductance; the check
gives its peak, its gain margin, and the highest bandwidth the loads' power
control may have for a margin of 6 dB.
"""

import math

import numpy as np


def loop_gain(f_hz, r_ohm, l_h, p_w, v_ll, f_c_hz, l_in_h=None):
    """T(j 2 pi f) = Z_out Y_in at frequencies f_hz (array), per phase on a line-to-line voltage v_ll."""
    s = 2j * math.pi * np.asarray(f_hz, dtype=float)
    z_out = r_ohm + s * l_h
    g_cpl = p_w / v_ll ** 2                           # its negative conductance, per phase (star)
    y_in = -g_cpl / (1.0 + s / (2 * math.pi * f_c_hz))
    if l_in_h:
        y_in = y_in + 1.0 / (s * l_in_h)
    return z_out * y_in


def check(r_ohm, l_h, p_w, v_ll, f_c_hz, l_in_h=None, f_hz=None):
    """
    The minor loop's verdict: {'peak_gain', 'gain_margin_db', 'stable', 'f_c_max_hz'}
    - its largest |T| where its phase passes 180 deg (the negative-resistance
    band), its margin to 1, and the loads' power-control bandwidth for 6 dB.
    """
    f = np.logspace(-1, 4, 4000) if f_hz is None else np.asarray(f_hz, dtype=float)
    t = loop_gain(f, r_ohm, l_h, p_w, v_ll, f_c_hz, l_in_h)
    neg = np.real(t) < 0                               # where it can reach -1
    peak = float(np.max(np.abs(t[neg]))) if neg.any() else 0.0
    # Its peak with an ideal power control: |Z_out(f_c)| P / V^2 at the bandwidth f_c; for 6 dB, 0.5.
    g = p_w / v_ll ** 2
    x_target = 0.5 / g
    f_max = math.sqrt(max(x_target ** 2 - r_ohm ** 2, 0.0)) / (2 * math.pi * l_h) if l_h > 0 else math.inf
    return {'peak_gain': peak, 'gain_margin_db': -20 * math.log10(peak) if peak > 0 else math.inf,
            'stable': peak < 1.0, 'f_c_max_hz': f_max}
