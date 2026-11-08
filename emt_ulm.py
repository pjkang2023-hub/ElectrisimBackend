"""
The universal line model (Morched, Gustavsen and Tartibi, 1999) in the EMT study: a multiconductor line or cable
in the phase domain, for what a constant modal transformation cannot carry - a cable's cores and sheaths, whose
modes turn with frequency.

Its characteristic admittance Yc = Z^-1 sqrt(Z Y) and its current wave's propagation H = exp(-sqrt(Y Z) l), both
N x N and changing with frequency, are fitted in the phase domain:

    Yc(s) = D + sum_k R_k / (s + q_k),        H(s) = sum_g exp(-s tau_g) sum_k R_gk / (s + q_gk)

the delays tau_g those of its modes, grouped where they are alike (a cable's three coaxial modes travel
together), each mode's from where its wave is still above 1 % at the far end, tuned for the best fit of its
group's own propagation; the poles real, spread over the band; the residues by least squares. At each end, with
i into the line,

    i_k = Yc * v_k - 2 i_ki,    i_ki = H * i_mr(t - tau),    i_mr = i_m + i_mi

each convolution recursive by the trapezoid rule, so that each end is a conductance matrix Geq = D + sum b_k R_k
(emt_solver.add_gmatrix) and a current the controller sets every step, as long as every delay is over a step.
It starts in its own steady state at the fundamental.
"""
import math

import numpy as np
from scipy.optimize import linear_sum_assignment


def _sqrtm_modes(m):
    lam, vec = np.linalg.eig(m)
    g = np.sqrt(lam)
    g = np.where(g.real < 0, -g, g)
    return g, vec


def _poles(lo_hz, hi_hz, n):
    return 2.0 * math.pi * np.logspace(math.log10(lo_hz), math.log10(hi_hz), n)


def _lstsq(basis, target, ridge=1e-6):
    """Real coefficients x with basis x ~ target (complex, columns of target fitted at once), scaled and ridged."""
    scale = np.maximum(np.abs(basis).max(axis=0), 1e-300)
    b = basis / scale
    k = b.shape[1]
    a = np.vstack([b.real, b.imag, ridge * np.eye(k)])
    rhs = np.vstack([target.real, target.imag, np.zeros((k, target.shape[1]))])
    x, *_ = np.linalg.lstsq(a, rhs, rcond=None)
    return x / scale[:, None]


class UlmFit:
    """The fitted Yc and H of a line ``km`` long with per-km z and y (arrays (len(f), N, N))."""

    def __init__(self, z, y, km, f, n_yc=30, n_h=24, group_tol=0.05):
        f = np.asarray(f, dtype=float)
        s = 2j * math.pi * f
        n = z.shape[1]
        self.n = n
        yc = np.zeros_like(z)
        h = np.zeros_like(z)
        gam = np.zeros((len(f), n), dtype=complex)
        prev = None
        # Its modes, followed from the top of the band down by their eigenvectors, so each keeps its own.
        for i in range(len(f) - 1, -1, -1):
            g_zy, v_zy = _sqrtm_modes(z[i] @ y[i])
            yc[i] = np.linalg.solve(z[i], v_zy @ np.diag(g_zy) @ np.linalg.inv(v_zy))
            g_yz, v_yz = _sqrtm_modes(y[i] @ z[i])
            h[i] = v_yz @ np.diag(np.exp(-g_yz * km)) @ np.linalg.inv(v_yz)
            vn = v_yz / np.linalg.norm(v_yz, axis=0)
            if prev is not None:
                _, cols = linear_sum_assignment(-np.abs(prev.conj().T @ vn))
                vn, g_yz = vn[:, cols], g_yz[cols]
            prev = vn
            gam[i] = g_yz
        self.f, self.yc_exact, self.h_exact = f, yc, h
        # Yc: D and residue matrices on fixed real poles.
        self.q = _poles(max(f[0], 1e-3), f[-1], n_yc)
        basis = np.hstack([np.ones((len(f), 1)), 1.0 / (s[:, None] + self.q)])
        iu = np.triu_indices(n)
        weight = 1.0 / np.linalg.norm(yc, axis=(1, 2))
        x = _lstsq(basis * weight[:, None], yc[:, iu[0], iu[1]] * weight[:, None])
        self.d = np.zeros((n, n))
        self.r = np.zeros((n_yc, n, n))
        self.d[iu], self.d[(iu[1], iu[0])] = x[0], x[0]
        for k in range(n_yc):
            self.r[k][iu], self.r[k][(iu[1], iu[0])] = x[1 + k], x[1 + k]
        # Each mode's delay, from where its wave is still above 1 % at the far end.
        att = np.abs(np.exp(-gam * km))
        taus = []
        tops = []
        for m in range(n):
            ok = np.where((att[:, m] > 1e-2) & (f >= 1.0))[0]
            top = int(ok.max()) if len(ok) else 0
            tops.append(top)
            taus.append((gam[top, m] * km).imag / (2.0 * math.pi * f[top]))
        order = np.argsort(taus)
        groups = []
        for m in order:
            if groups and taus[m] <= groups[-1]['tau0'] * (1.0 + group_tol):
                groups[-1]['modes'].append(int(m))
            else:
                groups.append({'tau0': taus[m], 'modes': [int(m)]})
        # Each group's delay: the one that fits its own modes' propagation best; its poles, over its band.
        for gr in groups:
            top = max(tops[m] for m in gr['modes'])
            sel = (f >= 0.1) & (np.arange(len(f)) <= top)
            gr['q'] = _poles(0.1, max(f[top] * 3.0, 1.0), n_h)
            target = np.sum(np.exp(-gam[:, gr['modes']] * km), axis=1)
            best = None
            for scale in np.linspace(0.9, 1.0, 21):
                tau = gr['tau0'] * scale
                b = np.exp(-s[sel, None] * tau) / (s[sel, None] + gr['q'])
                c = _lstsq(b, target[sel, None], 1e-5)
                err = float(np.max(np.abs(b @ c[:, 0] - target[sel])))
                if best is None or err < best[0]:
                    best = (err, tau)
            gr['tau'] = best[1]
        self.groups = groups
        # H: residue matrices for every group's poles at once, with their delays - every element, H not being
        # symmetric (its transpose is the voltage wave's).
        keep = (f >= 0.1) & (np.arange(len(f)) <= max(tops))
        cols = [np.exp(-s[keep, None] * gr['tau']) / (s[keep, None] + gr['q']) for gr in groups]
        basis = np.hstack(cols)
        # Ridged hard: the residues of groups whose delays differ little can grow large and cancel, and what they
        # leave between the frequencies fitted can make the line active (the cable's 1 km minor section was).
        x = _lstsq(basis, h[keep].reshape(int(keep.sum()), n * n), 1e-3)
        k0 = 0
        for gr in groups:
            nk = len(gr['q'])
            gr['r'] = x[k0:k0 + nk].reshape(nk, n, n)
            k0 += nk
        self.err_yc = float(np.max(np.abs(self.yc(s) - yc) / np.linalg.norm(yc, axis=(1, 2))[:, None, None]))
        self.err_h = float(np.max(np.abs(self.h(s[keep]) - h[keep])))

    def yc(self, s):
        s = np.atleast_1d(s)
        return self.d[None] + np.einsum('fk,kij->fij', 1.0 / (s[:, None] + self.q), self.r)

    def h(self, s):
        s = np.atleast_1d(s)
        out = 0
        for gr in self.groups:
            out = out + np.exp(-s * gr['tau'])[:, None, None] * np.einsum('fk,kij->fij', 1.0 / (s[:, None] + gr['q']), gr['r'])
        return out


class UlmLine:
    """
    The fitted line between nodes ``a`` and ``b`` (each conductor's, repeated where conductors are tied to one node).
    Raises ValueError if a delay is not over a step or its conductance at a step is not positive.
    """

    def __init__(self, ckt, a, b, fit, dt, w0, label):
        self.label, self.fit, self.dt, self.w0, self.ckt = label, fit, dt, w0, ckt
        n = fit.n
        if len(a) != n or len(b) != n:
            raise ValueError(f'its {n} conductors need {n} nodes at each end')
        taus = [gr['tau'] for gr in fit.groups]
        self.tau_ms = (max(taus) * 1e3, min(taus) * 1e3)
        self.tau_modes_ms = [x * 1e3 for x in taus]
        if min(taus) <= dt:
            raise ValueError(f'its travel time, {min(taus) * 1e6:.1f} us, is not over the {dt * 1e6:g} us step')
        d_ = 1.0 + fit.q * dt / 2.0
        self.al, self.be = (1.0 - fit.q * dt / 2.0) / d_, dt / 2.0 / d_
        for gr in fit.groups:
            dg = 1.0 + gr['q'] * dt / 2.0
            gr['al'], gr['be'] = (1.0 - gr['q'] * dt / 2.0) / dg, dt / 2.0 / dg
            gr['delay'] = gr['tau'] / dt
        self.geq = fit.d + np.einsum('k,kij->ij', self.be, fit.r)
        if np.min(np.linalg.eigvalsh(0.5 * (self.geq + self.geq.T))) <= 0:
            raise ValueError('its fitted conductance at the step is not positive')
        # Passive, as the recursion has it, from 1 Hz to near the step's Nyquist frequency: else it could grow.
        self.passivity = min(self._passivity(fr) for fr in np.logspace(0.0, math.log10(0.45 / dt), 200))
        if self.passivity < -1e-3:
            raise ValueError(f'its fitted model is not passive ({self.passivity:.3g} of its admittance)')
        import emt_fd_line
        self.a, self.pa = emt_fd_line.FdLine._fold(a)
        self.b, self.pb = emt_fd_line.FdLine._fold(b)
        for nodes, pm in ((self.a, self.pa), (self.b, self.pb)):
            ckt.add_gmatrix(nodes, pm.T @ self.geq @ pm)
        self.ks = [[ckt.add_isrc(x, 0) for x in nodes] for nodes in (self.a, self.b)]
        self.size = int(math.ceil(max(gr['delay'] for gr in fit.groups))) + 4
        self.buf = np.zeros((2, self.size, n))
        self.s = np.zeros((2, len(fit.q), n))
        self.w = [[np.zeros((len(gr['q']), n)) for gr in fit.groups] for _ in range(2)]
        self.u_prev = [[np.zeros(n) for _ in fit.groups] for _ in range(2)]
        self.v_prev = np.zeros((2, n))
        self.i = np.zeros((2, n))
        self.i_i = np.zeros((2, n))
        self.hist = np.zeros((2, n))
        self.n_step = 0
        ckt.add_ac_block(self.a + self.b, self._ac_correction(), self._start)

    # --- its response at a frequency, as the recursion has it ---

    def _factor(self, al, be, w):
        z = np.exp(1j * w * self.dt)
        return be * (1.0 + 1.0 / z) / (1.0 - al / z)

    def _discrete(self, w):
        yc = self.fit.d + np.einsum('k,kij->ij', self._factor(self.al, self.be, w), self.fit.r)
        h = 0
        for gr in self.fit.groups:
            h = h + np.exp(-1j * w * gr['tau']) * np.einsum('k,kij->ij', self._factor(gr['al'], gr['be'], w), gr['r'])
        return yc, h

    def _waves(self, v1, v2, w):
        """The incident currents at both ends for end voltages v1, v2 at ``w``."""
        yc, h = self._discrete(w)
        n = self.fit.n
        m = np.block([[np.eye(n), h], [h, np.eye(n)]])
        rhs = np.concatenate([h @ yc @ v2, h @ yc @ v1])
        x = np.linalg.solve(m, rhs)
        return x[:n], x[n:], yc, h

    def admittance(self, w):
        """Its 2N x 2N two-port admittance at ``w`` (conductors)."""
        n = self.fit.n
        out = np.zeros((2 * n, 2 * n), dtype=complex)
        for k in range(2 * n):
            v = np.zeros(2 * n, dtype=complex)
            v[k] = 1.0
            i1i, i2i, yc, _ = self._waves(v[:n], v[n:], w)
            out[:, k] = np.concatenate([yc @ v[:n] - 2.0 * i1i, yc @ v[n:] - 2.0 * i2i])
        return out

    def _passivity(self, f_hz):
        """The least eigenvalue of its two-port's Hermitian part at f_hz, as a share of the two-port's largest entry."""
        y = self.admittance(2.0 * math.pi * f_hz)
        return float(np.linalg.eigvalsh(0.5 * (y + y.conj().T)).min() / np.abs(y).max())

    def _ac_correction(self):
        y = self.admittance(self.w0)
        n = self.fit.n
        big = np.block([[self.pa, np.zeros_like(self.pb)], [np.zeros_like(self.pa), self.pb]])
        y = big.T @ y @ big
        na = len(self.a)
        y[:na, :na] -= self.pa.T @ self.geq @ self.pa
        y[na:, na:] -= self.pb.T @ self.geq @ self.pb
        return y

    # --- running ---

    def _start(self, v):
        """The phasor solve's end voltages: its state at t = 0 as if it had always run at the fundamental."""
        na = len(self.a)
        w = self.w0
        v1, v2 = self.pa @ v[:na], self.pb @ v[na:]
        i1i, i2i, yc, _ = self._waves(v1, v2, w)
        vs, iis = (v1, v2), (i1i, i2i)
        ir = [yc @ vs[e] - iis[e] for e in range(2)]      # i_r = i + i_i = Yc v - i_i
        fk = self._factor(self.al, self.be, w)
        for e in range(2):
            self.v_prev[e] = vs[e].real
            self.i_i[e] = iis[e].real
            self.i[e] = (yc @ vs[e] - 2.0 * iis[e]).real
            for k in range(len(self.fit.q)):
                self.s[e, k] = (fk[k] * self.fit.r[k] @ vs[e]).real
            for j in range(self.size):
                self.buf[e, (-j) % self.size] = (ir[e] * np.exp(-1j * w * self.dt * j)).real
        for e in range(2):
            other = 1 - e
            for g, gr in enumerate(self.fit.groups):
                u = ir[other] * np.exp(-1j * w * gr['tau'])
                fg = self._factor(gr['al'], gr['be'], w)
                self.w[e][g] = np.array([(fg[k] * gr['r'][k] @ u).real for k in range(len(gr['q']))])
                self.u_prev[e][g] = u.real
        self.n_step = 0
        self._prepare(1)
        self._set(lambda k, value: self.ckt.isrc[k].__setitem__(2, value))

    def _delayed(self, e, index):
        lo = math.floor(index)
        frac = index - lo
        return (1.0 - frac) * self.buf[e, lo % self.size] + frac * self.buf[e, (lo + 1) % self.size]

    def _prepare(self, m):
        """What step m's solve needs: each end's incident current and its history."""
        for e in range(2):
            other = 1 - e
            total = np.zeros(self.fit.n)
            for g, gr in enumerate(self.fit.groups):
                u = self._delayed(other, m - gr['delay'])
                self.w[e][g] = gr['al'][:, None] * self.w[e][g] + gr['be'][:, None] * np.einsum('kij,j->ki', gr['r'], u + self.u_prev[e][g])
                self.u_prev[e][g] = u
                total += self.w[e][g].sum(axis=0)
            self.i_i[e] = total
            past = self.al[:, None] * self.s[e] + self.be[:, None] * np.einsum('kij,j->ki', self.fit.r, self.v_prev[e])
            self.hist[e] = past.sum(axis=0) - 2.0 * self.i_i[e]

    def _set(self, setter):
        for e, pm in enumerate((self.pa, self.pb)):
            for k, value in enumerate(pm.T @ self.hist[e]):
                setter(self.ks[e][k], float(value))

    def control(self, t, state):
        m = self.n_step + 1
        vs = (self.pa @ state.v[self.a], self.pb @ state.v[self.b])
        for e in range(2):
            v = vs[e]
            self.i[e] = self.geq @ v + self.hist[e]
            self.s[e] = self.al[:, None] * self.s[e] + self.be[:, None] * np.einsum('kij,j->ki', self.fit.r, v + self.v_prev[e])
            self.v_prev[e] = v
            self.buf[e, m % self.size] = self.i[e] + self.i_i[e]
        self.n_step = m
        self._prepare(m + 1)
        self._set(state.set_source_value)
        return False

    def currents(self):
        """Its present current into the line at each end, per conductor, in A."""
        return [self.i[0].copy(), self.i[1].copy()]
