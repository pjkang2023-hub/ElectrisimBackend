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

Untransposed, or more than one circuit on its towers (geometry 'conductors', each one's (x, y) in metres; or
'transposed' False on the flat row): its N conductors' Z and Y in full,

    Z ii = Z internal + j w u0 / 2 pi  ln(2 (y_i + p) / r),   Z ij = j w u0 / 2 pi  ln(sqrt((y_i + y_j + 2 p)^2 + dx^2) / sqrt((y_i - y_j)^2 + dx^2))

and its modes by J. Marti's constant real transformation: the eigenvectors of Z Y at 'transform_hz' (1 kHz by
default), their real parts, Q = their inverse, so that v_mode = Q v and i = Q^T i_mode, each mode Q Z Q^T and
Q^-T Y Q^-1's diagonal at every frequency, fitted on its own. What the real, fixed Q leaves coupled between
modes is left out: small on a flat or vertical line, as in PSCAD's and EMTP's own models of the kind.
Conductors tied to the same node at its ends (two circuits in parallel between the same buses) are folded
into it. Identical circuits with no geometry of their own ('parallel' > 1 on a transposed row) are taken
far enough apart not to couple: Zc over their number.

A cable (fd_kind 'cable'): each phase a coaxial core inside its sheath, the sheath at ground potential, as
with solid bonding on a run short against the sheath's own impedance. The phases are then uncoupled, each one
mode, so the same model runs with the identity for its transform:

    Z = Z core internal + j w u0 / 2 pi  ln(r_sheath / r_core) + Z sheath inner surface,   Y = j w 2 pi e0 er / ln(r_sheath / r_core)

the sheath a thin tube, its inner surface's impedance rho m / (2 pi r) coth(m t), m = sqrt(j w u0 / rho): its
DC resistance at low frequency, the surface impedance at high. So the sheath carries each core's whole
return: right for the coaxial waves of energising and switching, but at the fundamental it overstates the
loss of cables whose sheaths are cross-bonded or bonded at one end, which carry little sheath current; the
study warns where the row's own impedance disagrees. Left out: sheath voltages, cross-bonding, the phases'
coupling through the ground, and semiconducting screens.
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
CABLE_FIELDS = ('core_radius_m', 'core_rdc_ohm_per_km', 'insulation_radius_m', 'sheath_thickness_m', 'sheath_rdc_ohm_per_km', 'eps_r')


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


def per_km_cable(f, g):
    """Z and Y per km of a coaxial core and sheath at frequencies ``f``."""
    w = 2.0 * math.pi * np.asarray(f, dtype=float)
    rc, rs, t = g['core_radius_m'], g['insulation_radius_m'], g['sheath_thickness_m']
    rho_s = g['sheath_rdc_ohm_per_km'] * 1e-3 * 2.0 * math.pi * rs * t
    m = np.sqrt(1j * w * U0 / rho_s)
    sheath = rho_s * m / (2.0 * math.pi * rs) / np.tanh(m * t)
    z = _internal(w, rc, g['core_rdc_ohm_per_km'] * 1e-3) + 1j * w * U0 / (2.0 * math.pi) * math.log(rs / rc) + sheath
    y = 1j * w * 2.0 * math.pi * E0 * g['eps_r'] / math.log(rs / rc)
    return z * 1e3, y * 1e3


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


def fits(geometry, km, par=1.0):
    f = np.logspace(-3, 6, 361)
    z1, y1, z0, y0 = per_km(f, geometry)
    return ModeFit(z0 / par, y0 * par, km, f), ModeFit(z1 / par, y1 * par, km, f)


def conductors(geometry):
    """Its conductors' (x, y), m: as given, or the flat row's three at its height and spacing."""
    xy = geometry.get('conductors')
    if xy:
        return [(float(x), float(y)) for x, y in xy]
    d, h = geometry['spacing_m'], geometry['height_m']
    return [(-d, h), (0.0, h), (d, h)]


def parse_conductors(text):
    """'x,y; x,y; ...' to [(x, y)], or None."""
    raw = str(text or '').strip()
    if not raw:
        return None
    try:
        out = [tuple(float(v) for v in pair.split(',')) for pair in raw.split(';') if pair.strip()]
    except ValueError:
        return None
    return out if out and all(len(x) == 2 for x in out) else None


def per_km_matrix(f, geometry):
    """Z and Y per km of its conductors at frequencies ``f``: arrays (len(f), N, N)."""
    xy = conductors(geometry)
    r, rdc, rho = geometry['radius_m'], geometry['rdc_ohm_per_km'] * 1e-3, geometry['rho_ohm_m']
    w = 2.0 * math.pi * np.asarray(f, dtype=float)
    p = np.sqrt(rho / (1j * w * U0))
    n = len(xy)
    z = np.zeros((len(w), n, n), dtype=complex)
    pot = np.zeros((n, n))
    zi = _internal(w, r, rdc)
    for i, (xi, yi) in enumerate(xy):
        for j, (xj, yj) in enumerate(xy):
            if i == j:
                z[:, i, i] = zi + 1j * w * U0 / (2.0 * math.pi) * np.log(2.0 * (yi + p) / r)
                pot[i, i] = math.log(2.0 * yi / r) / (2.0 * math.pi * E0)
            else:
                dx, near = xi - xj, math.hypot(xi - xj, yi - yj)
                z[:, i, j] = 1j * w * U0 / (2.0 * math.pi) * np.log(np.sqrt((yi + yj + 2.0 * p) ** 2 + dx * dx) / near)
                pot[i, j] = math.log(math.hypot(yi + yj, dx) / near) / (2.0 * math.pi * E0)
    y = 1j * w[:, None, None] * np.linalg.inv(pot)[None, :, :]
    return z * 1e3, y * 1e3


def modal(z, y, f, transform_hz=1000.0):
    """
    J. Marti's constant real transformation: Q, and each mode's z and y per km at every frequency
    (arrays (len(f), N)). Its modes ordered slowest (the ground mode) first.
    """
    k = int(np.argmin(np.abs(np.log(np.asarray(f) / transform_hz))))
    lam, vec = np.linalg.eig(z[k] @ y[k])
    tv = vec.real.copy()
    tv /= np.linalg.norm(tv, axis=0)
    order = np.argsort(-lam.real)               # the largest gamma^2: the slowest, lossiest mode
    tv = tv[:, order]
    q = np.linalg.inv(tv)
    qi = np.linalg.inv(q)
    zm = np.einsum('ij,fjk,lk->fil', q, z, q)
    ym = np.einsum('ji,fjk,kl->fil', qi, y, qi)
    return q, np.diagonal(zm, axis1=1, axis2=2), np.diagonal(ym, axis1=1, axis2=2)


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

    def __init__(self, ckt, a, b, geometry, km, dt, w0, label, par=1.0):
        """``a``, ``b``: each conductor's node at each end (repeated where conductors are tied to one node)."""
        self.label = label
        self.kind = 'cable' if geometry.get('kind') == 'cable' else 'overhead'
        self.untransposed = self.kind == 'overhead' and (bool(geometry.get('conductors')) or geometry.get('transposed') is False)
        if self.kind == 'cable':
            f = np.logspace(-3, 6, 361)
            z, y = per_km_cable(f, geometry)
            fit = ModeFit(z / par, y * par, km, f)
            fitted = [fit, fit, fit]
            self.T = np.eye(3)                 # its phases uncoupled: each its own mode
        elif self.untransposed:
            f = np.logspace(-3, 6, 361)
            z, y = per_km_matrix(f, geometry)
            self.T, zm, ym = modal(z, y, f, float(geometry.get('transform_hz') or 1000.0))
            fitted = [ModeFit(zm[:, m], ym[:, m], km, f) for m in range(zm.shape[1])]
        else:
            fit0, fit1 = fits(geometry, km, par)
            fitted = [fit0, fit1, fit1]
            self.T = T
        n = len(fitted)
        if len(a) != n or len(b) != n:
            raise ValueError(f'its {n} conductors need {n} nodes at each end')
        taus = [x.tau for x in fitted]
        self.tau_ms = (max(taus) * 1e3, min(taus) * 1e3)
        self.tau_modes_ms = [x * 1e3 for x in taus]
        if min(taus) <= dt:
            raise ValueError(f'its travel time, {min(taus) * 1e6:.1f} us, is not over the {dt * 1e6:g} us step')
        self.modes = [_Mode(x, dt) for x in fitted]
        # Its conductors to the nodes they are tied to: P, conductors x nodes, at each end.
        self.a, self.pa = self._fold(a)
        self.b, self.pb = self._fold(b)
        g = np.diag([1.0 / m.req for m in self.modes])
        for nodes, pm in ((self.a, self.pa), (self.b, self.pb)):
            self._network(ckt, nodes, pm.T @ self.T.T @ g @ self.T @ pm)
        self.ks = [[ckt.add_isrc(x, 0) for x in nodes] for nodes in (self.a, self.b)]
        self.w0 = w0
        self.ckt = ckt
        ckt.add_ac_block(self.a + self.b, self._ac_correction(), self._start)

    @staticmethod
    def _fold(nodes):
        unique = list(dict.fromkeys(int(x) for x in nodes))
        pm = np.zeros((len(nodes), len(unique)))
        for k, x in enumerate(nodes):
            pm[k, unique.index(int(x))] = 1.0
        return unique, pm

    def _network(self, ckt, nodes, g):
        """
        The Norton conductance matrix ``g`` among ``nodes``: as resistors where they can all be positive (a transposed
        line's always can), else as a conductance matrix (a cable's cores and sheaths).
        """
        tol = 1e-9 * float(np.max(np.abs(g)))
        ground = g.sum(axis=1)
        off = -(g - np.diag(np.diag(g)))
        if np.any(ground < -tol) or np.any(off < -tol):
            ckt.add_gmatrix(nodes, g)
            return
        for i in range(len(nodes)):
            if ground[i] > tol:
                ckt.add_r(nodes[i], 0, 1.0 / ground[i])
            for j in range(i + 1, len(nodes)):
                if off[i, j] > tol:
                    ckt.add_r(nodes[i], nodes[j], 1.0 / off[i, j])

    def _phase_y(self, w):
        """Its admittance among its end nodes at ``w``, as the recursion has it."""
        modal_y = [m.admittance(w) for m in self.modes]
        blocks = [[self.T.T @ np.diag([modal_y[m][e1, e2] for m in range(len(self.modes))]) @ self.T for e2 in range(2)] for e1 in range(2)]
        pms = (self.pa, self.pb)
        return np.block([[pms[e1].T @ blocks[e1][e2] @ pms[e2] for e2 in range(2)] for e1 in range(2)])

    def _ac_correction(self):
        """What the phasor solve must add to the Norton resistors to make them the whole line at the fundamental."""
        y = self._phase_y(self.w0)
        norton = self.T.T @ np.diag([1.0 / m.req for m in self.modes]) @ self.T
        na = len(self.a)
        y[:na, :na] -= self.pa.T @ norton @ self.pa
        y[na:, na:] -= self.pb.T @ norton @ self.pb
        return y

    def _start(self, v):
        """The phasor solve's end voltages ``v``: the state at t = 0, and the sources for the first step."""
        na = len(self.a)
        va, vb = self.T @ (self.pa @ v[:na]), self.T @ (self.pb @ v[na:])
        for m, mode in enumerate(self.modes):
            mode.start(np.array([va[m], vb[m]]), self.w0)
        self._set(lambda k, value: self.ckt.isrc[k].__setitem__(2, value))

    def _set(self, setter):
        for e, pm in enumerate((self.pa, self.pb)):
            i_node = pm.T @ (self.T.T @ np.array([mode.source(e) for mode in self.modes]))
            for k, value in enumerate(i_node):
                # add_isrc(node, 0): a current from the node into the line.
                setter(self.ks[e][k], float(value))

    def control(self, t, state):
        va, vb = self.T @ (self.pa @ state.v[self.a]), self.T @ (self.pb @ state.v[self.b])
        for m, mode in enumerate(self.modes):
            mode.step((va[m], vb[m]))
        self._set(state.set_source_value)
        return False

    def currents(self):
        """Its present current into the line at each end, per conductor, in A."""
        return [self.T.T @ np.array([mode.i[e] for mode in self.modes]) for e in range(2)]
