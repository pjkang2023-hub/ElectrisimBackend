"""
Electrisim's time-domain circuit solver, for the DC fault and EMT studies.

Nodal analysis of two-terminal branches, integrated by the trapezoidal rule,
with two half steps of backward Euler (critical damping adjustment) on the
first step, on a step where an element switches, and on the step after an
event - where the trapezoidal rule would ring.
Node 0 is the DC negative pole (the reference).

Branches, each with its current from node a to node b:
- RL: v_a - v_b + e(t) = R i + L di/dt, with e = e0 + E cos(w t + phi);
- C:  i = C d(v_a - v_b)/dt;
- R:  i = (v_a - v_b) / R;
- diode, anode a and cathode b: R_ON conducting, R_OFF not;
- switch: R closed or open, set by events and controllers;
- surge arrester: R_OFF until |v| reaches its clamping voltage, then that
  voltage behind a small resistance (a piecewise-linear metal-oxide arrester);
- nonlinear: i = f(v, t), Newton-iterated within each step (constant-power loads).

The state at t = 0 is each RL branch's current and each capacitor's voltage.
"""

import math

import numpy as np
from scipy.linalg import lu_factor, lu_solve

R_FLOOR = 1e-6           # ohm: a branch with no R or L
R_ON, R_OFF = 1e-5, 1e6  # ohm: a diode conducting / blocking
R_BIAS = 1e6             # ohm: holds a converter's AC side at its DC link's midpoint
SW_R_ON, SW_R_OFF = 1e-5, 1e7      # ohm: a switch closed / open
ARR_R_ON, ARR_R_OFF = 1e-3, 1e7    # ohm: an arrester clamping / not


class Circuit:
    def __init__(self):
        self.labels = ['DC negative pole']
        self.v0 = [0.0]           # each node's voltage at t = 0: the first guess for nonlinear elements
        self.rl, self.c, self.r, self.d = [], [], [], []
        self.sw, self.arr, self.nl = [], [], []
        self.events = []          # (t, action(state))
        self.controllers = []     # fn(t, state) -> True when it changed something

    # --- building -----------------------------------------------------------------

    def node(self, label, v0=0.0):
        self.labels.append(label)
        self.v0.append(float(v0))
        return len(self.labels) - 1

    def add_rl(self, a, b, r, l, i0=0.0, e0=0.0, ac=None):
        """``ac``: (amplitude, omega, phase) of an AC electromotive force."""
        if r <= 0 and l <= 0:
            r = R_FLOOR
        amp, om, ph = ac or (0.0, 0.0, 0.0)
        self.rl.append([a, b, max(r, 0.0), max(l, 0.0), i0, e0, amp, om, ph])
        return len(self.rl) - 1

    def add_c(self, a, b, c, w0=0.0):
        self.c.append((a, b, c, w0))
        return len(self.c) - 1

    def add_r(self, a, b, r):
        self.r.append((a, b, max(r, R_FLOOR)))
        return len(self.r) - 1

    def add_diode(self, anode, cathode):
        self.d.append((anode, cathode))
        return len(self.d) - 1

    def add_switch(self, a, b, closed=True, r_on=SW_R_ON, r_off=SW_R_OFF):
        self.sw.append([a, b, bool(closed), r_on, r_off])
        return len(self.sw) - 1

    def add_arrester(self, a, b, v_clamp, r_on=ARR_R_ON, r_off=ARR_R_OFF):
        self.arr.append((a, b, float(v_clamp), r_on, r_off))
        return len(self.arr) - 1

    def add_nonlinear(self, a, b, func):
        """``func(v_ab, t)`` -> (current a to b, its derivative by v_ab)."""
        self.nl.append((a, b, func))
        return len(self.nl) - 1

    def at(self, t, action):
        """Run ``action(state)`` once the simulation reaches ``t``; the step after it is backward Euler."""
        self.events.append((float(t), action))

    def add_controller(self, fn):
        """``fn(t, state)`` after every step; it returns True when it changed a switch or a load."""
        self.controllers.append(fn)

    # --- simulating ---------------------------------------------------------------

    def simulate(self, t_end, dt, dt_coarse=None, t_fine=None, max_switch_iter=40, nl_tol=1e-6):
        """
        Returns times and, at each, node voltages and branch currents (i_rl,
        i_c, i_r, i_d, i_sw, i_arr, i_nl), and each switch's state (sw_closed).
        """
        n = len(self.labels)
        rl = np.array(self.rl, dtype=float).reshape(-1, 9)
        cc = np.array(self.c, dtype=float).reshape(-1, 4)
        rr = np.array(self.r, dtype=float).reshape(-1, 3)
        dd = np.array(self.d, dtype=int).reshape(-1, 2)
        ra, rb = rl[:, 0].astype(int), rl[:, 1].astype(int)
        R, L = rl[:, 2], rl[:, 3]
        e0, eamp, eom, eph = rl[:, 5], rl[:, 6], rl[:, 7], rl[:, 8]
        ca, cb, C = cc[:, 0].astype(int), cc[:, 1].astype(int), cc[:, 2]
        xa, xb, Gr = rr[:, 0].astype(int), rr[:, 1].astype(int), 1.0 / rr[:, 2]
        da, db = dd[:, 0], dd[:, 1]
        sa = np.array([s[0] for s in self.sw], dtype=int)
        sb = np.array([s[1] for s in self.sw], dtype=int)
        s_on = np.array([1.0 / s[3] for s in self.sw])
        s_off = np.array([1.0 / s[4] for s in self.sw])
        aa = np.array([x[0] for x in self.arr], dtype=int)
        ab = np.array([x[1] for x in self.arr], dtype=int)
        a_vc = np.array([x[2] for x in self.arr])
        a_gon = np.array([1.0 / x[3] for x in self.arr])
        a_goff = np.array([1.0 / x[4] for x in self.arr])
        na_, nb_ = np.array([x[0] for x in self.nl], dtype=int), np.array([x[1] for x in self.nl], dtype=int)

        # The time grid, landing on each event.
        event_times = sorted({t for t, _ in self.events if 0 < t < t_end})
        steps = []
        t = 0.0
        t_fine = t_end if t_fine is None else t_fine
        k_ev = 0
        while t < t_end - 1e-15:
            h = dt if t < t_fine - 1e-15 else (dt_coarse or dt)
            h = min(h, t_end - t)
            while k_ev < len(event_times) and event_times[k_ev] <= t + 1e-15:
                k_ev += 1
            if k_ev < len(event_times) and t + h > event_times[k_ev] + 1e-15:
                h = event_times[k_ev] - t
            steps.append(h)
            t += h
        N = len(steps)
        times = np.concatenate([[0.0], np.cumsum(steps)])

        out = {
            't': times, 'v': np.zeros((N + 1, n)), 'i_rl': np.zeros((N + 1, len(R))),
            'i_c': np.zeros((N + 1, len(C))), 'i_r': np.zeros((N + 1, len(Gr))),
            'i_d': np.zeros((N + 1, len(da))), 'i_sw': np.zeros((N + 1, len(sa))),
            'i_arr': np.zeros((N + 1, len(aa))), 'i_nl': np.zeros((N + 1, len(self.nl))),
            'sw_closed': np.zeros((N + 1, len(sa)), dtype=bool),
        }

        def emf(tt):
            return e0 + eamp * np.cos(eom * tt + eph)

        state = _State(self, n)
        state.sw_closed = np.array([s[2] for s in self.sw], dtype=bool)
        i_now = rl[:, 4].copy()
        w_now = cc[:, 3].copy()
        ic_now = np.zeros(len(C))
        v_now = np.array(self.v0, dtype=float)
        d_on = np.zeros(len(da), dtype=bool)
        a_st = np.zeros(len(aa), dtype=int)      # 0 off, +1 / -1 clamping
        out['i_rl'][0] = i_now
        out['v'][0] = v_now
        out['sw_closed'][0] = state.sw_closed
        cache = {}

        def stamp(a, b, g, Y):
            np.add.at(Y, (a, a), g)
            np.add.at(Y, (b, b), g)
            np.add.at(Y, (a, b), -g)
            np.add.at(Y, (b, a), -g)

        def base_matrix(method, h, d_state, sw_state, a_state):
            key = (method, h, d_state.tobytes(), sw_state.tobytes(), a_state.tobytes())
            if key not in cache:
                k = 2.0 if method == 'tr' else 1.0
                Y = np.zeros((n, n))
                stamp(ra, rb, 1.0 / (R + k * L / h), Y)
                stamp(ca, cb, k * C / h, Y)
                stamp(xa, xb, Gr, Y)
                stamp(da, db, np.where(d_state, 1.0 / R_ON, 1.0 / R_OFF), Y)
                stamp(sa, sb, np.where(sw_state, s_on, s_off), Y)
                stamp(aa, ab, np.where(a_state != 0, a_gon, a_goff), Y)
                if len(cache) > 512:
                    cache.clear()
                factor = None if self.nl else lu_factor(Y[1:, 1:], check_finite=False)
                cache[key] = (Y, factor)
            return cache[key]

        iterations = np.zeros(N, dtype=int)   # solves per step, for diagnosis
        out['iterations'] = iterations
        pending = sorted(self.events, key=lambda e: e[0])
        k_pending = 0
        be_next = True
        t = 0.0
        for s, h in enumerate(steps):
            # Events due now change switches or loads; the step after them is backward Euler.
            while k_pending < len(pending) and pending[k_pending][0] <= t + 1e-15:
                state.t = t
                pending[k_pending][1](state)
                k_pending += 1
                be_next = True
            t1 = t + h

            def solve(method, h_, t0):
                """One step of length h_ from t0, with diodes, arresters and loads iterated to agree."""
                e_now, e_next = emf(t0), emf(t0 + h_)
                d_state, a_state = d_on.copy(), a_st.copy()
                v_guess = v_now.copy()
                k = 2.0 if method == 'tr' else 1.0
                G = 1.0 / (R + k * L / h_)
                if method == 'tr':
                    u_now = v_now[ra] - v_now[rb] + e_now
                    J = G * e_next + G * (u_now + (2.0 * L / h_ - R) * i_now)
                    Gc = 2.0 * C / h_
                    Jc = -Gc * w_now - ic_now
                else:
                    J = G * e_next + G * (L / h_) * i_now
                    Gc = C / h_
                    Jc = -Gc * w_now
                rhs0 = np.bincount(rb, J, n) - np.bincount(ra, J, n)
                rhs0 += np.bincount(cb, Jc, n) - np.bincount(ca, Jc, n)
                for _it in range(max_switch_iter):
                    iterations[s] += 1
                    Ja = np.where(a_state > 0, -a_vc * a_gon, np.where(a_state < 0, a_vc * a_gon, 0.0))
                    rhs = rhs0 + np.bincount(ab, Ja, n) - np.bincount(aa, Ja, n)
                    Y, factor = base_matrix(method, h_, d_state, state.sw_closed, a_state)
                    if self.nl:
                        Yn = Y.copy()
                        vn = v_guess[na_] - v_guess[nb_]
                        gn, Jn = np.zeros(len(self.nl)), np.zeros(len(self.nl))
                        for j, (_, _, func) in enumerate(self.nl):
                            i0, g0 = func(vn[j], t0 + h_)
                            gn[j], Jn[j] = g0, i0 - g0 * vn[j]
                        stamp(na_, nb_, gn, Yn)
                        rhs = rhs + np.bincount(nb_, Jn, n) - np.bincount(na_, Jn, n)
                        factor = lu_factor(Yn[1:, 1:], check_finite=False)
                    v = np.zeros(n)
                    v[1:] = lu_solve(factor, rhs[1:], check_finite=False)
                    vd = v[da] - v[db]
                    new_d = np.where(d_state, vd > 0.0, vd > 1e-9)
                    va = v[aa] - v[ab]
                    new_a = np.where(a_state > 0, np.where(va > a_vc, 1, 0),
                                     np.where(a_state < 0, np.where(va < -a_vc, -1, 0),
                                              np.where(va > a_vc, 1, np.where(va < -a_vc, -1, 0))))
                    if self.nl:
                        dv = (v[na_] - v[nb_]) - vn
                        nl_done = np.max(np.abs(dv)) <= nl_tol * max(1.0, np.max(np.abs(vn)))
                    else:
                        nl_done = True
                    if np.array_equal(new_d, d_state) and np.array_equal(new_a, a_state) and nl_done:
                        break
                    d_state, a_state, v_guess = new_d, new_a, v
                i_next = G * (v[ra] - v[rb]) + J
                w_next = v[ca] - v[cb]
                return v, d_state, a_state, i_next, w_next, Gc * w_next + Jc

            if not be_next:
                sol = solve('tr', h, t)
                if not (np.array_equal(sol[1], d_on) and np.array_equal(sol[2], a_st)):
                    be_next = True    # an element switched: this step is taken again, damped
            damp_next = False
            if be_next:
                # Critical damping adjustment: two half steps of backward Euler. The
                # trapezoidal rule would ring - undamped - on a mode faster than the
                # step, as an inductor's current just interrupted behind an open switch.
                d0, a0 = d_on.copy(), a_st.copy()
                v_now, d_on, a_st, i_now, w_now, ic_now = solve('be', h / 2, t)
                sol = solve('be', h / 2, t + h / 2)
                # An element that switched within this step leaves the next one to damp too.
                damp_next = not (np.array_equal(sol[1], d0) and np.array_equal(sol[2], a0))
            v, d_on, a_st, i_now, w_now, ic_now = sol
            v_now = v
            t = t1
            out['v'][s + 1] = v
            out['i_rl'][s + 1] = i_now
            out['i_c'][s + 1] = ic_now
            out['i_r'][s + 1] = Gr * (v[xa] - v[xb])
            out['i_d'][s + 1] = np.where(d_on, 1.0 / R_ON, 1.0 / R_OFF) * (v[da] - v[db])
            out['i_sw'][s + 1] = np.where(state.sw_closed, s_on, s_off) * (v[sa] - v[sb])
            va = v[aa] - v[ab]
            out['i_arr'][s + 1] = np.where(a_st != 0, a_gon, a_goff) * va + np.where(
                a_st > 0, -a_vc * a_gon, np.where(a_st < 0, a_vc * a_gon, 0.0))
            if self.nl:
                vn = v[na_] - v[nb_]
                out['i_nl'][s + 1] = [func(vn[j], t)[0] for j, (_, _, func) in enumerate(self.nl)]
            be_next = damp_next
            state.t, state.v = t, v
            state.i_rl, state.i_sw = out['i_rl'][s + 1], out['i_sw'][s + 1]
            for fn in self.controllers:
                if fn(t, state):
                    be_next = True
            out['sw_closed'][s + 1] = state.sw_closed
        return out


class _State:
    """What events and controllers see and change: the time, node voltages, currents, switch states."""

    def __init__(self, circuit, n):
        self.circuit = circuit
        self.t = 0.0
        self.v = np.zeros(n)
        self.i_rl = None
        self.i_sw = None
        self.sw_closed = None

    def set_switch(self, k, closed):
        self.sw_closed = self.sw_closed.copy()
        self.sw_closed[k] = bool(closed)


def dc_load_current(p_mw, vn_kv, share_p, share_i, share_r, v_min_pu):
    """
    A DC load's current (A) as a function of its voltage (V), with its
    derivative - the load flow's model in time: constant-power, constant-
    current and constant-resistance shares of its power, the constant-power
    part drawing constant current below v_min_pu. A rectifier front end draws
    no current at reverse voltage, so the power and current parts stop below
    0 - reached linearly over the last 5 % of nominal voltage, where a step
    would leave Newton's method flipping between the two.
    ``p_mw`` is a one-element list, so an event can step it.
    """
    vn = vn_kv * 1e3
    v_eps = _RAMP_PU * vn

    def func(v, t):
        p = p_mw[0] * 1e6
        g_r = share_r * p / (vn * vn)
        i, g = g_r * v, g_r
        if v <= 0:
            return i, g
        # The constant-power and constant-current parts at v, before the ramp to zero.
        i_pc, g_pc = 0.0, 0.0
        if share_p:
            v_knee = max(v_min_pu * vn, v_eps)
            if v >= v_knee:
                i_pc, g_pc = share_p * p / v, -share_p * p / (v * v)
            else:
                i_pc = share_p * p / v_knee
        if share_i:
            i_pc += share_i * p / vn
        if v < v_eps:
            return i + i_pc * v / v_eps, g + i_pc / v_eps
        return i + i_pc, g + g_pc
    return func


_RAMP_PU = 0.05
