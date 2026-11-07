"""
A frequency-dependent line in the EMT study: travelling waves with their delay, and R and L
that change with frequency (skin effect, earth return), as J. Marti's model has them. For a
transposed overhead line, by its three modes: one ground mode and two equal aerial modes.

Its per-km Z and Y come from its geometry: conductors in a flat row at a height and spacing,
their radius and DC resistance, and the earth's resistivity:

    Z self = Z internal + j w u0 / 2 pi  ln(2 (h + p) / r),   p = sqrt(rho / (j w u0)) (Deri's complex depth)
    Z mutual = j w u0 / 2 pi  ln(sqrt((2 h + 2 p)^2 + d^2) / d),   Z1 = Zs - Zm,   Z0 = Zs + 2 Zm

and the capacitances from Maxwell's potentials. The transposition averages the three pairs.

In each mode, with Zc = sqrt(Z / Y) and A = exp(-sqrt(Z Y) l) = A' exp(-s tau), each end's
current into the line follows from what arrives from the other end:

    b1(t) = A' * f2(t - tau),   f2(t) = v2(t) + Zc * i2(t),   i1(t) = (v1(t) - b1(t)) / Zc,   * a convolution

Zc and A' are fitted with real poles, Zc's residues all positive so that it stays passive, and
each convolution runs recursively by the trapezoid rule. So each end is a conductance and a
current source the controller sets each step, as long as tau is over a step: refused otherwise.

It starts in its own steady state at the fundamental (emt_solver's AC block).
"""
import math

import numpy as np
from scipy.optimize import nnls
from scipy.special import ive

U0 = 4e-7 * math.pi
E0 = 8.854187817e-12
SQ3 = math.sqrt(3.0)
# Phase to mode: orthonormal, so its inverse is its transpose. Row 0 is the ground mode.
T = np.array([[1.0 / SQ3, 1.0 / SQ3, 1.0 / SQ3],
              [math.sqrt(2.0 / 3.0), -1.0 / math.sqrt(6.0), -1.0 / math.sqrt(6.0)],
              [0.0, 1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)]])
FIELDS = ('height_m', 'spacing_m', 'radius_m', 'rdc_ohm_per_km', 'rho_ohm_m')


def _internal(w, r, rdc_per_m):
    """A solid round conductor's internal impedance (ohm/m), its skin effect: Bessel functions of its radius."""
    rho = rdc_per_m * math.pi * r * r
    k = np.sqrt(1j * w * U0 / rho)
    x = k * r
    return rho * k / (2.0 * math.pi * r) * ive(0, x) / ive(1, x)


def per_km(f, geometry):
    """Z and Y per km of the aerial and ground modes at frequencies ``f``: (z1, y1, z0, y0)."""
    h, d, r = geometry['height_m'], geometry['spacing_m'], geometry['radius_m']
    rdc, rho = geometry['rdc_ohm_per_km'] * 1e-3, geometry['rho_ohm_m']
    w = 2.0 * math.pi * np.asarray(f, dtype=float)
    p = np.sqrt(rho / (1j * w * U0))
    zs = _internal(w, r, rdc) + 1j * w * U0 / (2.0 * math.pi) * np.log(2.0 * (h + p) / r)
    zm = sum(1j * w * U0 / (2.0 * math.pi) * np.log(np.sqrt((2.0 * h + 2.0 * p) ** 2 + x * x) / x) for x in (d, d, 2.0 * d)) / 3.0
    ps = math.log(2.0 * h / r) / (2.0 * math.pi * E0)
    pm = sum(math.log(math.sqrt(4.0 * h * h + x * x) / x) for x in (d, d, 2.0 * d)) / 3.0 / (2.0 * math.pi * E0)
    y1, y0 = 1j * w / (ps - pm), 1j * w / (ps + 2.0 * pm)
    return (zs - zm) * 1e3, y1 * 1e3, (zs + 2.0 * zm) * 1e3, y0 * 1e3


def _poles(lo_hz, hi_hz, n):
    return 2.0 * math.pi * np.logspace(math.log10(lo_hz), math.log10(hi_hz), n)


class ModeFit:
    """One mode's fit: Zc = k0 + sum k / (s + p), A' = sum c / (s + q), and tau."""

    def __init__(self, z, y, km, f):
        s = 2j * math.pi * f
        zc = np.sqrt(z / y)
        gamma = np.sqrt(z * y)
        # Zc: non-negative residues, so that it stays passive.
        self.p = _poles(f[0], f[-1], 30)
        basis = 1.0 / (s[:, None] + self.p)
        rows = np.hstack([np.ones((len(s), 1), dtype=complex), basis]) / np.abs(zc)[:, None]
        scale = np.abs(rows).max(axis=0)
        sol, _ = nnls(np.vstack([(rows / scale).real, (rows / scale).imag]), np.concatenate([(zc / np.abs(zc)).real, (zc / np.abs(zc)).imag]))
        sol = sol / scale
        self.k0, self.k = float(sol[0]), sol[1:]
        # A': the delay that leaves the smoothest remainder, near the phase's slope at the top of the band.
        total = gamma * km
        keep = (np.exp(-total.real) > 1e-4) & (f >= 0.1)
        top = np.max(np.where(keep)[0])
        tau0 = total.imag[top] / (2.0 * math.pi * f[top])
        best = None
        self.q = _poles(0.1, f[top] * 3.0, 24)
        basis = 1.0 / (s[keep, None] + self.q)
        for tau in tau0 * np.linspace(0.97, 1.0, 13):
            target = np.exp(-total[keep] + s[keep] * tau)
            weights = np.ones(len(target))
            rows = basis * weights[:, None]
            scale = np.abs(rows).max(axis=0)
            big = np.vstack([(rows / scale).real, (rows / scale).imag, 1e-5 * np.eye(len(self.q))])
            rhs = np.concatenate([(target * weights).real, (target * weights).imag, np.zeros(len(self.q))])
            sol, *_ = np.linalg.lstsq(big, rhs, rcond=None)
            error = float(np.max(np.abs(rows @ (sol / scale) - target * weights)))
            if best is None or error < best[0]:
                best = (error, tau, sol / scale)
        self.fit_error, self.tau, self.c = best

    def zc(self, s):
        return self.k0 + np.sum(self.k / (np.asarray(s)[..., None] + self.p), axis=-1)

    def a(self, s):
        return np.sum(self.c / (np.asarray(s)[..., None] + self.q), axis=-1) * np.exp(-np.asarray(s) * self.tau)


def fits(geometry, km):
    f = np.logspace(-3, 6, 361)
    z1, y1, z0, y0 = per_km(f, geometry)
    return ModeFit(z0, y0, km, f), ModeFit(z1, y1, km, f)


class _Mode:
    """One mode's recursive convolutions at step ``dt``, at both ends of the line."""

    def __init__(self, fit, dt):
        self.fit, self.dt = fit, dt
        d = 1.0 + fit.p * dt / 2.0
        self.az, self.bz = (1.0 - fit.p * dt / 2.0) / d, fit.k * dt / 2.0 / d
        d = 1.0 + fit.q * dt / 2.0
        self.aa, self.ba = (1.0 - fit.q * dt / 2.0) / d, fit.c * dt / 2.0 / d
        self.req = fit.k0 + float(np.sum(self.bz))
        self.delay = fit.tau / dt
        self.size = int(math.ceil(self.delay)) + 4
        self.f_buf = np.zeros((2, self.size))
        self.s = np.zeros((2, len(fit.p)))
        self.a = np.zeros((2, len(fit.q)))
        self.i = np.zeros(2)
        self.b = np.zeros(2)
        self.hz = np.zeros(2)
        self.n = 0

    def _discrete(self, w):
        """Its Zc and A at ``w`` as the recursion has them, the trapezoid's frequency warping included."""
        z = np.exp(1j * w * self.dt)
        zc = self.fit.k0 + np.sum(self.bz * (1.0 + 1.0 / z) / (1.0 - self.az / z))
        a = np.sum(self.ba * (1.0 + 1.0 / z) / (1.0 - self.aa / z)) * np.exp(-1j * w * self.fit.tau)
        return zc, a

    def admittance(self, w):
        """Its two-port admittance at ``w``: i = Y v, into the line at each end."""
        zc, a = self._discrete(w)
        # b1 = A (2 v2 - b2), b2 = A (2 v1 - b1)
        m = np.array([[1.0, a], [a, 1.0]], dtype=complex)
        b_of_v = np.linalg.solve(m, np.array([[0.0, 2.0 * a], [2.0 * a, 0.0]], dtype=complex))
        return (np.eye(2) - b_of_v) / zc

    def start(self, v, w):
        """Its state at t = 0 from the end voltage phasors ``v`` (two), as if it had always run at ``w``."""
        y = self.admittance(w)
        i = y @ v
        zc, a = self._discrete(w)
        f = v + zc * i
        z = np.exp(1j * w * self.dt)
        for e in range(2):
            self.i[e] = i[e].real
            self.s[e] = (self.bz * (1.0 + 1.0 / z) / (1.0 - self.az / z) * i[e]).real
            self.a[e] = (self.ba * (1.0 + 1.0 / z) / (1.0 - self.aa / z) * f[1 - e] * np.exp(-1j * w * self.fit.tau)).real
            # Its past, the latest at slot n = 0.
            self.f_buf[e] = (f[e] * np.exp(-1j * w * self.dt * ((-np.arange(self.size)) % self.size))).real
        self._prepare()

    def _f_at(self, e, index):
        """f at end e at a step ``index`` (possibly between steps, and in the past)."""
        lo = math.floor(index)
        frac = index - lo
        return (1.0 - frac) * self.f_buf[e, lo % self.size] + frac * self.f_buf[e, (lo + 1) % self.size]

    def _prepare(self):
        """What the next step's solve needs, from all that is known at this step."""
        for e in range(2):
            self.hz[e] = float(np.sum(self.az * self.s[e] + self.bz * self.i[e]))
            now = self._f_at(1 - e, self.n + 1 - self.delay) + self._f_at(1 - e, self.n - self.delay)
            self.a[e] = self.aa * self.a[e] + self.ba * now
            self.b[e] = float(np.sum(self.a[e]))

    def step(self, v):
        """The end voltages ``v`` at the new step: its currents and f there, then what the following step needs."""
        for e in range(2):
            i = (v[e] - self.hz[e] - self.b[e]) / self.req
            self.s[e] = self.az * self.s[e] + self.bz * (i + self.i[e])
            self.i[e] = i
        self.n += 1
        for e in range(2):
            self.f_buf[e, self.n % self.size] = v[e] + self.fit.k0 * self.i[e] + float(np.sum(self.s[e]))
        self._prepare()

    def source(self, e):
        """The current into the line at end e beyond v / req."""
        return -(self.hz[e] + self.b[e]) / self.req


class FdLine:
    """``a``, ``b``: its two ends' phase nodes. Raises ValueError if a mode's delay is not over a step, or its conductances cannot be a network."""

    def __init__(self, ckt, a, b, geometry, km, dt, w0, label):
        self.label, self.a, self.b = label, list(a), list(b)
        fit0, fit1 = fits(geometry, km)
        self.tau_ms = (fit0.tau * 1e3, fit1.tau * 1e3)
        if min(fit0.tau, fit1.tau) <= dt:
            raise ValueError(f'its travel time, {min(fit0.tau, fit1.tau) * 1e6:.1f} us, is not over the {dt * 1e6:g} us step')
        self.modes = [_Mode(fit0, dt), _Mode(fit1, dt), _Mode(fit1, dt)]
        g0, g1 = 1.0 / self.modes[0].req, 1.0 / self.modes[1].req
        if g1 < g0:
            raise ValueError('its aerial surge impedance is above its ground mode\'s')
        # The Norton conductance as a network: phase to ground G0, phase to phase (G1 - G0) / 3.
        for nodes in (self.a, self.b):
            for ph in range(3):
                ckt.add_r(nodes[ph], 0, 1.0 / g0)
                ckt.add_r(nodes[ph], nodes[(ph + 1) % 3], 3.0 / (g1 - g0) if g1 > g0 else 1e12)
        self.ks = [[ckt.add_isrc(nodes[ph], 0) for ph in range(3)] for nodes in (self.a, self.b)]
        self.w0 = w0
        self.ckt = ckt
        ckt.add_ac_block(self.a + self.b, self._ac_correction(), self._start)

    def _phase_y(self, w):
        """Its 6 x 6 phase admittance at ``w``, as the recursion has it."""
        y = np.zeros((6, 6), dtype=complex)
        modal = [m.admittance(w) for m in self.modes]
        for e1 in range(2):
            for e2 in range(2):
                block = T.T @ np.diag([modal[m][e1, e2] for m in range(3)]) @ T
                y[3 * e1:3 * e1 + 3, 3 * e2:3 * e2 + 3] = block
        return y

    def _ac_correction(self):
        """What the phasor solve must add to the Norton resistors to make them the whole line at the fundamental."""
        y = self._phase_y(self.w0)
        norton = T.T @ np.diag([1.0 / self.modes[m].req for m in range(3)]) @ T
        y[:3, :3] -= norton
        y[3:, 3:] -= norton
        return y

    def _start(self, v):
        """The phasor solve's end voltages ``v``: the state at t = 0, and the sources for the first step."""
        va, vb = T @ v[:3], T @ v[3:]
        for m, mode in enumerate(self.modes):
            mode.start(np.array([va[m], vb[m]]), self.w0)
        self._set(lambda k, value: self.ckt.isrc[k].__setitem__(2, value))

    def _set(self, setter):
        for e in range(2):
            i_ph = T.T @ np.array([mode.source(e) for mode in self.modes])
            for ph in range(3):
                # add_isrc(node, 0): a current from the node into the line.
                setter(self.ks[e][ph], float(i_ph[ph]))

    def control(self, t, state):
        va, vb = T @ state.v[self.a], T @ state.v[self.b]
        for m, mode in enumerate(self.modes):
            mode.step((va[m], vb[m]))
        self._set(state.set_source_value)
        return False

    def currents(self):
        """Its present current into the line at each end, phases, in A."""
        return [T.T @ np.array([mode.i[e] for mode in self.modes]) for e in range(2)]
