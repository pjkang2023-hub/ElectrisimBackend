"""
An impedance scan in the EMT study: a bus's positive-sequence impedance against frequency,
with its converters' controls in the loop, as a frequency scan in an EMT program gives it.

A small balanced current is injected at the bus, the sum of many tones at multiples of the
lowest frequency, each with its own phase so that their sum stays small. Once the run has
settled, the voltage's and current's space vectors are correlated with each tone over one
period of the lowest frequency, and Z(f) = V(f) / I(f). A converter's controls couple a
tone at f to one at 2 f1 - f, so no tone is scanned with its mirror, and none near f1.

Settings (the study's parameters):
- scan_bus: the bus's name;
- scan_f_min_hz, scan_f_max_hz: the band; the lowest frequency is rounded to divide f1;
- scan_points_per_decade: how many tones;
- scan_amp_pu: each tone's current, per unit of scan_base_mva at the bus's voltage;
- scan_settle_ms: how long before the correlation starts.

A low frequency takes a long run: 1 / f_min of simulated time. Scan a band at a time.
Linear only as far as the injection is small: raise and lower scan_amp_pu to check.
"""
import math

import numpy as np

A = complex(math.cos(2.0 * math.pi / 3.0), math.sin(2.0 * math.pi / 3.0))


def _f(value, default):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def tones(f_min, f_max, per_decade, f1):
    """The tones: multiples of a base dividing f1, log-spaced; none within a base of f1, and none with its mirror."""
    base = f1 / max(round(f1 / max(f_min, 1e-3)), 1)
    n_max = int(math.floor(f_max / base))
    count = max(int(math.ceil(math.log10(max(n_max, 1)) * per_decade)) + 1, 2)
    multiples = sorted({int(round(x)) for x in np.logspace(0, math.log10(max(n_max, 1)), count)})
    n1 = int(round(f1 / base))
    chosen = []
    for m in multiples:
        if abs(m - n1) <= 1:
            continue
        if abs(2 * n1 - m) in chosen:
            continue          # its mirror is already scanned
        chosen.append(m)
    return base, np.array(chosen, dtype=float) * base


class ImpedanceScan:
    """Its current sources are built into the circuit here; ``control`` runs each step."""

    def __init__(self, ckt, nodes, vn, params, f1, dt, label):
        self.label, self.nodes, self.dt = label, list(nodes), dt
        self.base, self.f = tones(_f(params.get('scan_f_min_hz'), 20.0), _f(params.get('scan_f_max_hz'), 2000.0),
                                  _f(params.get('scan_points_per_decade'), 10.0), f1)
        self.w = 2.0 * math.pi * self.f
        s_base = _f(params.get('scan_base_mva'), 100.0) * 1e6
        self.i_base = math.sqrt(2.0) * s_base / (math.sqrt(3.0) * vn)
        self.z_base = vn * vn / s_base
        self.amp = _f(params.get('scan_amp_pu'), 0.01) * self.i_base
        # Newman's phases keep the peak of a sum of many tones near that of noise.
        n = len(self.f)
        self.phase = math.pi * np.arange(n) ** 2 / max(n, 1)
        self.settle = _f(params.get('scan_settle_ms'), 300.0) * 1e-3
        self.window = 1.0 / self.base
        self.t_end = self.settle + self.window
        self.ks = [ckt.add_isrc(0, node) for node in self.nodes]    # into the bus
        self.sv = np.zeros(n, dtype=complex)
        self.si = np.zeros(n, dtype=complex)
        self.i_now = np.zeros(3)
        self.count = 0
        self.v_max = 0.0

    def _injection(self, t):
        x = np.sum(np.cos(self.w * t + self.phase))
        y = np.sum(np.sin(self.w * t + self.phase))
        z = complex(x, y) * self.amp          # its space vector: positive sequence
        return np.array([(z * A ** (-k)).real for k in range(3)])

    def control(self, t, state):
        # The source's value set now holds for the step ending at t + dt.
        if self.settle <= t < self.t_end - 1e-12:
            v = state.v[self.nodes]
            v_sv = (v[0] + A * v[1] + A * A * v[2]) * 2.0 / 3.0
            i = self.i_now
            i_sv = (i[0] + A * i[1] + A * A * i[2]) * 2.0 / 3.0
            rotate = np.exp(-1j * self.w * t)
            self.sv += v_sv * rotate
            self.si += i_sv * rotate
            self.count += 1
        i = self._injection(t + self.dt)
        self.i_now = i
        for k, ks in enumerate(self.ks):
            state.set_source_value(ks, float(i[k]))
        return False

    def result(self):
        z = self.sv / np.where(np.abs(self.si) > 0, self.si, 1.0)
        negative = [round(float(f), 3) for f, x in zip(self.f, z) if x.real < 0.0]
        return {'bus': self.label, 'f_hz': [round(float(x), 4) for x in self.f], 'f_base_hz': round(self.base, 5),
                'r_ohm': [round(float(x.real), 6) for x in z], 'x_ohm': [round(float(x.imag), 6) for x in z],
                'z_ohm': [round(float(abs(x)), 6) for x in z], 'angle_deg': [round(math.degrees(float(np.angle(x))), 3) for x in z],
                'z_pu': [round(float(abs(x)) / self.z_base, 6) for x in z], 'z_base_ohm': round(self.z_base, 6),
                'negative_resistance_hz': negative, 'samples': self.count, 'window_s': round(self.window, 6),
                'amp_ka_per_tone': round(self.amp * 1e-3, 6)}
