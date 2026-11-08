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
Linear only as far as the injection is small: raise and lower scan_amp_pu to check. scan_settle_ms
must outlast the slowest thing in the network, not the lowest tone: a load's L against a grid's R can
take seconds, and a slow control longer (an inverter's PLL and power loops).

So that a scan says when it has not settled: its tones come in over the first half of scan_settle_ms
(a raised cosine), exciting less of the slow transient than a step would; and it reads scan_windows
windows one after another (2 by default), each tone's impedance from the last, its change from the one
before (``change``) its settledness - within scan_settle_tol (2 %) it is settled, and the tones that are
not are listed (``unsettled_hz``): raise scan_settle_ms or scan_windows until none is.
"""
import math

import numpy as np

# The series scan for Middlebrook's criterion (SeriesScan): middlebrook_line, a line's name; the same band and settling
# settings; middlebrook_amp_pu its tones' voltage, per unit of the line's phase voltage.

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


class _Windows:
    """Correlations over consecutive windows: each tone's sums in each, and what the last two say of settling."""

    def __init__(self, params, n, settle, window):
        self.k = max(int(_f(params.get('scan_windows'), 2.0)), 1)
        self.tol = _f(params.get('scan_settle_tol'), 0.02)
        self.settle, self.window = settle, window
        self.t_end = settle + self.k * window
        self.ramp = 0.5 * settle
        self.sums = np.zeros((self.k, 3, n), dtype=complex)
        self.count = np.zeros(self.k, dtype=int)

    def gain(self, t):
        """The tones' share at t: a raised cosine over the first half of the settling time."""
        return 1.0 if t >= self.ramp or self.ramp <= 0 else 0.5 * (1.0 - math.cos(math.pi * t / self.ramp))

    def add(self, t, rotate, *signals):
        if not self.settle <= t < self.t_end - 1e-12:
            return
        w = min(int((t - self.settle) / self.window), self.k - 1)
        for j, x in enumerate(signals):
            self.sums[w, j] += x * rotate
        self.count[w] += 1

    def settling(self, z_of):
        """z from the last window, its change from the one before (None with one window), the tones beyond tolerance."""
        z = z_of(self.sums[-1])
        if self.k < 2:
            return z, None
        prev = z_of(self.sums[-2])
        return z, np.abs(z - prev) / np.maximum(np.abs(z), 1e-300)

    def report(self, f, change):
        if change is None:
            return {'windows': self.k, 'change': None, 'unsettled_hz': None}
        return {'windows': self.k, 'change': [round(float(x), 5) for x in change],
                'unsettled_hz': [round(float(x), 4) for x, c in zip(f, change) if c > self.tol], 'settle_tol': self.tol}


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
        self.windows = _Windows(params, n, self.settle, self.window)
        self.t_end = self.windows.t_end
        self.ks = [ckt.add_isrc(0, node) for node in self.nodes]    # into the bus
        self.sv = np.zeros(n, dtype=complex)
        self.si = np.zeros(n, dtype=complex)
        self.i_now = np.zeros(3)
        self.count = 0
        self.v_max = 0.0

    def _injection(self, t):
        x = np.sum(np.cos(self.w * t + self.phase))
        y = np.sum(np.sin(self.w * t + self.phase))
        z = complex(x, y) * self.amp * self.windows.gain(t)          # its space vector: positive sequence
        return np.array([(z * A ** (-k)).real for k in range(3)])

    def control(self, t, state):
        # The source's value set now holds for the step ending at t + dt.
        if self.settle <= t < self.t_end - 1e-12:
            v = state.v[self.nodes]
            v_sv = (v[0] + A * v[1] + A * A * v[2]) * 2.0 / 3.0
            i = self.i_now
            i_sv = (i[0] + A * i[1] + A * A * i[2]) * 2.0 / 3.0
            self.windows.add(t, np.exp(-1j * self.w * t), v_sv, i_sv)
            self.count += 1
        i = self._injection(t + self.dt)
        self.i_now = i
        for k, ks in enumerate(self.ks):
            state.set_source_value(ks, float(i[k]))
        return False

    def result(self):
        z, change = self.windows.settling(lambda sums: sums[0] / np.where(np.abs(sums[1]) > 0, sums[1], 1.0))
        negative = [round(float(f), 3) for f, x in zip(self.f, z) if x.real < 0.0]
        return {'bus': self.label, 'f_hz': [round(float(x), 4) for x in self.f], 'f_base_hz': round(self.base, 5),
                'r_ohm': [round(float(x.real), 6) for x in z], 'x_ohm': [round(float(x.imag), 6) for x in z],
                'z_ohm': [round(float(abs(x)), 6) for x in z], 'angle_deg': [round(math.degrees(float(np.angle(x))), 3) for x in z],
                'z_pu': [round(float(abs(x)) / self.z_base, 6) for x in z], 'z_base_ohm': round(self.z_base, 6),
                'negative_resistance_hz': negative, 'samples': self.count, 'window_s': round(self.window, 6),
                'amp_ka_per_tone': round(self.amp * 1e-3, 6), **self.windows.report(self.f, change)}


class SeriesScan:
    """
    Middlebrook's criterion at a line's from end: small tones of voltage in series between its bus and the line, so
    that one run measures both sides. With the current i into the line and the voltages v_s at the bus and v_l past
    the source, at each tone

        Z_source = -v_s / i   (the network behind the bus),     Z_load = v_l / i   (the line and what it feeds)

    and the minor-loop gain Z_source / Z_load. Its interface is stable where that gain stays clear of -1; inside
    |Z_source / Z_load| < 1 at every tone (Middlebrook's sufficient condition) it is stable whatever the phases.
    ``nodes_bus``, ``nodes_line``: the phase nodes either side of the series sources ``ks``.
    """

    def __init__(self, nodes_bus, nodes_line, ks, vn, params, f1, dt, label):
        self.label, self.dt = label, dt
        self.nodes_bus, self.nodes_line, self.ks = list(nodes_bus), list(nodes_line), list(ks)
        self.base, self.f = tones(_f(params.get('scan_f_min_hz'), 20.0), _f(params.get('scan_f_max_hz'), 2000.0),
                                  _f(params.get('scan_points_per_decade'), 10.0), f1)
        self.w = 2.0 * math.pi * self.f
        self.amp = _f(params.get('middlebrook_amp_pu'), 0.002) * math.sqrt(2.0 / 3.0) * vn
        n = len(self.f)
        self.phase = math.pi * np.arange(n) ** 2 / max(n, 1)
        self.settle = _f(params.get('scan_settle_ms'), 300.0) * 1e-3
        self.window = 1.0 / self.base
        self.windows = _Windows(params, n, self.settle, self.window)
        self.t_end = self.windows.t_end
        self.vs = np.zeros(n, dtype=complex)
        self.vl = np.zeros(n, dtype=complex)
        self.ii = np.zeros(n, dtype=complex)
        self.count = 0

    def _sources(self, t):
        z = complex(np.sum(np.cos(self.w * t + self.phase)), np.sum(np.sin(self.w * t + self.phase))) * self.amp * self.windows.gain(t)
        return [(z * A ** (-k)).real for k in range(3)]

    def control(self, t, state):
        if self.settle <= t < self.t_end - 1e-12:
            sv = lambda x: (x[0] + A * x[1] + A * A * x[2]) * 2.0 / 3.0
            self.windows.add(t, np.exp(-1j * self.w * t), sv(state.v[self.nodes_bus]), sv(state.v[self.nodes_line]),
                             sv(np.array([state.i_rl[k] for k in self.ks])))
            self.count += 1
        e = self._sources(t + self.dt)
        for k, ks in enumerate(self.ks):
            state.set_emf(ks, float(e[k]))
        return False

    def result(self):
        def ratio(sums):
            i_ = np.where(np.abs(sums[2]) > 0, sums[2], 1.0)
            zl_ = sums[1] / i_
            return (-sums[0] / i_) / np.where(np.abs(zl_) > 0, zl_, 1.0)
        loop, change = self.windows.settling(ratio)
        last = self.windows.sums[-1]
        i = np.where(np.abs(last[2]) > 0, last[2], 1.0)
        zs, zl = -last[0] / i, last[1] / i
        over = [round(float(f), 3) for f, x in zip(self.f, loop) if abs(x) >= 1.0]
        k = int(np.argmin(np.abs(1.0 + loop))) if len(loop) else 0
        return {'line': self.label, 'f_hz': [round(float(x), 4) for x in self.f],
                'z_source_ohm': [round(float(abs(x)), 6) for x in zs], 'z_source_angle_deg': [round(math.degrees(float(np.angle(x))), 3) for x in zs],
                'z_load_ohm': [round(float(abs(x)), 6) for x in zl], 'z_load_angle_deg': [round(math.degrees(float(np.angle(x))), 3) for x in zl],
                'r_source_ohm': [round(float(x.real), 6) for x in zs], 'x_source_ohm': [round(float(x.imag), 6) for x in zs],
                'r_load_ohm': [round(float(x.real), 6) for x in zl], 'x_load_ohm': [round(float(x.imag), 6) for x in zl],
                'ratio': [round(float(abs(x)), 6) for x in loop], 'ratio_angle_deg': [round(math.degrees(float(np.angle(x))), 3) for x in loop],
                'ratio_max': round(float(np.max(np.abs(loop))), 6) if len(loop) else None,
                'sufficient': not over, 'ratio_over_one_hz': over,
                'distance_to_minus_one': round(float(abs(1.0 + loop[k])), 6) if len(loop) else None,
                'closest_hz': round(float(self.f[k]), 3) if len(loop) else None,
                'samples': self.count, 'window_s': round(self.window, 6), **self.windows.report(self.f, change)}
