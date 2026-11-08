"""
A modular multilevel converter in the EMT study: six arms, each a stack of half-bridge submodules
behind its arm reactor, modelled by arm (its capacitors kept balanced within the arm).

Each arm is a voltage n vS: n the share of its submodules inserted, vS the sum of their capacitor
voltages, which its own current charges: d(1/2 C vS^2)/dt = n vS i, C its submodule's capacitance over
their number. With ``levels`` > 0, n moves in steps of 1 / levels (nearest level), so its harmonics are
the staircase's; at 0 it is continuous (the arm-averaged model).

Each phase's arms sit across the DC link: v_upper = vdc / 2 - e, v_lower = vdc / 2 + e, and its AC side
sees e behind half an arm's impedance, then the transformer. The arms' common current, i_c = (i_u + i_l) / 2,
carries the DC side; the controls hold it at I_dc / 3, which takes out the second harmonic that circulates
otherwise.

Controls: a PLL on its AC bus; dq current control behind its reactances (current_loop_hz); its active power
set, or its DC voltage held (mode 'vdc'); the energy in its arms held by the balance between its two sides:
in mode 'p' by what it draws from the DC side, in mode 'vdc' by what it takes from the AC side. A limit on
its AC current.

Blocking, on an arm's current past block_pu of its AC current limit or a trip event: every switch off, so each
arm is what a blocked half-bridge stack is, its diodes. A current into the stack (P to midpoint in an upper arm,
midpoint to N in a lower) charges its capacitors through their upper diodes, so the arm stands at vS against it;
a current the other way passes the lower diodes at no voltage. So a DC fault is still fed from the AC side
through the bypass diodes, as an uncontrolled rectifier, and the capacitors keep their charge. Each arm carries
that path beside its controlled voltage: a switch, shut while it runs, bypasses it; the solver's own diodes
decide which way current flows once it is open.

Full-bridge submodules (cell 'full'): each can be inserted either way, so an arm's voltage runs from -vS to vS;
blocked, a current either way charges its capacitors through its diodes, so each arm stands at vS against it
both ways: a DC fault is no longer fed from the AC side (whose line-to-line peak is under 2 vS) and its current
dies away.

Submodule by submodule (submodules N > 0), the detailed equivalent model: each arm's N capacitors kept apart,
nearest-level insertion (round(order / their mean voltage) of them), and sorting to balance them as is usual -
with the arm's current charging, its least charged inserted, discharging, its most charged; with balancing off,
always the same ones, so they drift apart. Each submodule's capacitance is the arm's times N. Its switching (each
submodule's insertions and removals) is counted.

Not modelled: submodule failures; each switch's own on-state and switching losses.

It starts in its load-flow steady state, its arms' energy ripple included. ``v_ph``, ``i_ph``: its AC bus's
voltage and the current it delivers there, phase a peak phasors.
"""
import cmath
import math

import numpy as np

A120 = 2.0 * math.pi / 3.0
A = cmath.exp(1j * A120)
SQ2, SQ3 = math.sqrt(2.0), math.sqrt(3.0)


def _space(x):
    return (x[0] + A * x[1] + A * A * x[2]) * 2.0 / 3.0


class Mmc:
    def __init__(self, ckt, label, ac_nodes, p_node, w0, v_ph, i_ph, s_rated, vn_ac, v_dc, *, x_t_pu=0.15, r_t_pu=0.005, x_arm_pu=0.15, r_arm_pu=0.005,
                 energy_kj_per_mva=30.0, levels=0, mode='p', limit_pu=1.2, current_loop_hz=200.0, circulating_hz=300.0, pll_hz=20.0, feed_forward_hz=200.0,
                 block_pu=2.0, cell='half', submodules=0, balancing=True, sm_spread=0.0, r_bias=1e6):
        self.label, self.model, self.w0, self.s_rated = label, 'mmc', w0, s_rated
        self.full = str(cell).strip().lower() == 'full'
        self.n_sm, self.balancing = max(int(submodules), 0), bool(balancing)
        self.ac_nodes, self.p_node, self.levels, self.mode = list(ac_nodes), p_node, int(levels), str(mode)
        z = vn_ac * vn_ac / s_rated
        self.r_arm, self.l_arm = r_arm_pu * z, x_arm_pu * z / w0
        self.r_t, self.l_t = r_t_pu * z, x_t_pu * z / w0
        self.r_ac, self.l_ac = self.r_t + self.r_arm / 2.0, self.l_t + self.l_arm / 2.0
        self.i_max = limit_pu * SQ2 * s_rated / (SQ3 * vn_ac)
        self.v_dc0 = v_dc
        # Its energy: W = 6 (1/2 C vS^2) at vS = v_dc.
        self.w_nom = energy_kj_per_mva * 1e3 * s_rated / 1e6
        self.c_arm = self.w_nom / (3.0 * v_dc * v_dc)
        self.e_ph = v_ph + complex(self.r_ac, w0 * self.l_ac) * i_ph
        p_mid = 1.5 * (self.e_ph * i_ph.conjugate()).real
        # The DC side's current: its power and its arms' resistance on the common current.
        i_dc = p_mid / v_dc
        for _ in range(5):
            i_dc = (p_mid + 2.0 * self.r_arm * i_dc * i_dc / 3.0) / v_dc
        self.i_dc0 = i_dc
        self.p_set = 1.5 * (v_ph * i_ph.conjugate()).real
        self.q_set = 1.5 * (v_ph * i_ph.conjugate()).imag
        v_arm_dc = v_dc / 2.0 - self.r_arm * i_dc / 3.0
        # Its nodes: each phase's arms' midpoint, floating behind the isolating transformer's star.
        star = ckt.node(f'{label} star point', v_dc / 2.0)
        ckt.add_r(star, 0, r_bias)
        ckt.add_r(star, p_node, r_bias)
        l_mag = 1000.0 * self.l_t
        r_m = np.diag([0.0, self.r_t])
        l_m = np.array([[l_mag, l_mag], [l_mag, l_mag + self.l_t]])
        self.mid = []
        self.k_u, self.k_l = [], []
        self.k_sw, self.k_src = [[], []], [[], []]      # each arm's bypass switch and its blocked stack's source, upper then lower
        self.ckt = ckt
        r_src = 1e-3 * z
        self.k_src2 = [[], []]          # a full-bridge stack's other way, against its capacitors too
        for k in range(3):
            x = ckt.node(f'{label} midpoint {"abc"[k]}', v_dc / 2.0)
            self.mid.append(x)
            ckt.add_coupled([self.ac_nodes[k], x], [0, star], r_m, l_m)
            phase = cmath.phase(self.e_ph) - k * A120
            for side, (a, b, v_b, sign) in enumerate(((p_node, x, v_dc / 2.0, 0.0), (x, 0, 0.0, math.pi))):
                inner = ckt.node(f'{label} arm {"ul"[side]}{"abc"[k]}', v_b)
                stack = ckt.node(f'{label} stack {"ul"[side]}{"abc"[k]}', v_b + v_dc)
                arm = ckt.add_rl(a, inner, self.r_arm, self.l_arm, i0=i_dc / 3.0, e0=-v_arm_dc, ac=(abs(self.e_ph), w0, phase + sign), controlled=True)
                (self.k_u if side == 0 else self.k_l).append(arm)
                self.k_sw[side].append(ckt.add_switch(inner, b, closed=True))
                # Blocked: into the stack through its upper diodes, against its capacitors; out of it through its lower ones.
                # A volt forward: with none they would switch with every zero of the current beside the shut switch, and each
                # switching makes the solver's next step a damped one.
                ckt.add_diode(inner, stack, v_f=1.0)
                self.k_src[side].append(ckt.add_rl(stack, b, r_src, 0.0, e0=-v_dc, controlled=True))
                if self.full:
                    # Out of the stack: through the other diodes, against its capacitors as well.
                    back = ckt.node(f'{label} stack back {"ul"[side]}{"abc"[k]}', v_b + v_dc)
                    ckt.add_diode(b, back, v_f=1.0)
                    self.k_src2[side].append(ckt.add_rl(back, inner, r_src, 0.0, e0=-v_dc, controlled=True))
                else:
                    ckt.add_diode(b, inner, v_f=1.0)
        self.p_set_now = self.p_set
        self.w_u = np.zeros(3)
        self.w_l = np.zeros(3)
        self.vs_u = np.zeros(3)
        self.vs_l = np.zeros(3)
        self.v_u = np.full(3, v_arm_dc)
        self.v_l = np.full(3, v_arm_dc)
        self._start_energy(i_ph, i_dc, v_dc)
        if self.n_sm:
            # Each submodule's capacitor, its share of its arm's voltage (spread by sm_spread, to show the balancing).
            self.c_sm = self.c_arm * self.n_sm
            spread = 1.0 + sm_spread * np.linspace(-1.0, 1.0, self.n_sm)
            spread /= spread.mean()
            self.vc = [np.outer(vs / self.n_sm, spread) for vs in (self.vs_u, self.vs_l)]
            self.pol = [np.zeros((3, self.n_sm)), np.zeros((3, self.n_sm))]
            for side, vs in enumerate((self.vs_u, self.vs_l)):
                for k in range(3):
                    self.pol[side][k], _ = self._choose(v_arm_dc, self.vc[side][k], 1.0, self.pol[side][k])
            self.switchings = 0
            self.spread_max = 0.0
            self._sm_state()
        # Controls.
        self.theta0 = cmath.phase(v_ph)
        self.pll_angle = self.theta0
        wn = 2.0 * math.pi * pll_hz
        v_mag = abs(v_ph)
        self.pll_kp, self.pll_ki = 2.0 * 0.7 * wn / v_mag, wn * wn / v_mag
        self.pll_int = 0.0
        wc = 2.0 * math.pi * current_loop_hz
        self.kp, self.ki = self.l_ac * wc, self.r_ac * wc
        self.int_dq = 0j
        # Its voltage feed-forward, filtered: at full bandwidth it would cancel its bus's voltage and hide its reactors.
        self.a_ff = 2.0 * math.pi * feed_forward_hz
        self.v_ff = complex(v_mag, 0.0)
        self.i_u_prev = np.array([i_dc / 3.0 + (i_ph * cmath.exp(-1j * k * A120)).real / 2.0 for k in range(3)])
        self.i_l_prev = np.array([i_dc / 3.0 - (i_ph * cmath.exp(-1j * k * A120)).real / 2.0 for k in range(3)])
        self.kc = self.l_arm * 2.0 * math.pi * circulating_hz
        self.k_w = 2.0 * math.pi * 5.0
        self.vdc_set = v_dc
        wv = 2.0 * math.pi * 10.0
        self.vdc_kp, self.vdc_ki = 2.0 * 0.7 * wv * 1e-3, wv * wv * 1e-3     # replaced by set_dc_capacitance
        self.vdc_int = i_dc
        self.t_last = 0.0
        self.limited_time, self.blocked_at, self.f_sw = 0.0, None, None
        self.block_i = block_pu * self.i_max
        self.trace = []              # (t, P W, Q var, v_dc V, i_dc-side current A, W J)
        self.i_dc_now = i_dc

    def set_dc_capacitance(self, c):
        """Mode 'vdc': its DC voltage loop's gains from the capacitance on its DC bus, at 10 Hz."""
        wv = 2.0 * math.pi * 10.0
        self.vdc_kp, self.vdc_ki = 2.0 * 0.7 * wv * c, wv * wv * c

    def _start_energy(self, i_ph, i_dc, v_dc):
        """
        Each arm's energy in the steady state, W + the integral of its power: for the upper arm
        (vdc / 2 - e)(I / 3 + i / 2) a ripple at the fundamental, (vdc I / 4 - I_dc E / 3) / (j w), and at
        twice it, -E I / (8 j w); for the lower, the first with its sign changed.
        """
        w = self.w0
        one = (v_dc * i_ph / 4.0 - i_dc * self.e_ph / 3.0) / (1j * w)
        two = -self.e_ph * i_ph / 4.0 / (2j * w)
        w_mean = self.w_nom / 6.0
        for k in range(3):
            r1, r2 = cmath.exp(-1j * k * A120), cmath.exp(-2j * k * A120)
            self.w_u[k] = w_mean + (one * r1).real + (two * r2).real
            self.w_l[k] = w_mean - (one * r1).real + (two * r2).real
        self.vs_u = np.sqrt(2.0 * self.w_u / self.c_arm)
        self.vs_l = np.sqrt(2.0 * self.w_l / self.c_arm)

    def _insert(self, order, vs):
        """Its arm's actual voltage for an ordered one: the share inserted, in levels if it has them."""
        n = np.clip(order / np.maximum(vs, 1.0), -1.0 if self.full else 0.0, 1.0)
        if self.levels > 0:
            n = np.round(n * self.levels) / self.levels
        return n * vs

    def _choose(self, order, vc, current, before):
        """
        One arm's submodules for an ordered voltage: how many, nearest level; which, by sorting. Returns their
        insertion (+1, -1 the other way round in a full bridge, 0 bypassed) and its voltage.
        """
        n = int(round(order / max(float(vc.mean()), 1e-9)))
        n = max(min(n, self.n_sm), -self.n_sm if self.full else 0)
        pol = np.zeros(self.n_sm)
        m, sign = abs(n), (1.0 if n >= 0 else -1.0)
        if m:
            if self.balancing:
                order_by = np.argsort(vc)
                # Charging (its current into them as inserted): the least charged; discharging: the most.
                chosen = order_by[:m] if current * sign > 0 else order_by[::-1][:m]
            else:
                chosen = np.arange(m)
            pol[chosen] = sign
        return pol, float(np.dot(pol, vc))

    def _sm_state(self):
        """Its arms' sums and energies from their submodules'."""
        self.vs_u, self.vs_l = self.vc[0].sum(axis=1), self.vc[1].sum(axis=1)
        self.w_u = 0.5 * self.c_sm * (self.vc[0] ** 2).sum(axis=1)
        self.w_l = 0.5 * self.c_sm * (self.vc[1] ** 2).sum(axis=1)
        mean = np.concatenate([x.mean(axis=1) for x in self.vc])
        spread = np.concatenate([np.ptp(x, axis=1) for x in self.vc]) / np.maximum(mean, 1.0)
        self.spread_now = float(np.max(spread))

    def sm_result(self, t_run):
        """Its submodules at the end: their voltages' widest spread in an arm (share of its mean), the widest
        through the run after its first cycle, and each submodule's mean switching frequency."""
        if not self.n_sm:
            return None
        return {'submodules': self.n_sm, 'balancing': self.balancing, 'spread_end': self.spread_now, 'spread_max': self.spread_max,
                'switching_hz': self.switchings / 2.0 / (6.0 * self.n_sm) / max(t_run, 1e-9)}

    def control(self, t, state):
        h = t - self.t_last
        if h <= 0:
            return False
        self.t_last = t
        i_u = np.array([state.i_rl[k] for k in self.k_u])
        i_l = np.array([state.i_rl[k] for k in self.k_l])
        v = state.v[self.ac_nodes]
        v_dc = float(state.v[self.p_node])
        if self.blocked_at is not None:
            return self._blocked(t, h, state, i_u, i_l, v, v_dc)
        if max(float(np.max(np.abs(i_u))), float(np.max(np.abs(i_l)))) > self.block_i:
            return self._block(t, state)
        # Its arms' energy: what each has taken over the step, its voltage held and its current the trapezoid's.
        mean_u, mean_l = 0.5 * (i_u + self.i_u_prev), 0.5 * (i_l + self.i_l_prev)
        if self.n_sm:
            # Each inserted submodule's capacitor charged by its arm's current, the way it is inserted.
            for side, mean_i in enumerate((mean_u, mean_l)):
                self.vc[side] = np.maximum(self.vc[side] + h * self.pol[side] * mean_i[:, None] / self.c_sm, 1e-3)
            self._sm_state()
            if t > 1.0 / (self.w0 / (2.0 * math.pi)):
                self.spread_max = max(self.spread_max, self.spread_now)
        else:
            self.w_u = np.maximum(self.w_u + h * self.v_u * mean_u, 1.0)
            self.w_l = np.maximum(self.w_l + h * self.v_l * mean_l, 1.0)
            self.vs_u = np.sqrt(2.0 * self.w_u / self.c_arm)
            self.vs_l = np.sqrt(2.0 * self.w_l / self.c_arm)
        self.i_u_prev, self.i_l_prev = i_u, i_l
        w_total = float(np.sum(self.w_u) + np.sum(self.w_l))
        # Its PLL.
        v_s = _space(v)
        theta = self.w0 * t + self.pll_angle
        rot = complex(math.cos(theta), -math.sin(theta))
        v_dq = v_s * rot
        self.pll_int += self.pll_ki * v_dq.imag * h
        dw = self.pll_kp * v_dq.imag + self.pll_int
        self.pll_angle += dw * h
        i_ac = i_u - i_l
        # The currents its loops act on are those two-step means: the trapezoid leaves a ring at the step's own period undamped.
        i_dq = _space(mean_u - mean_l) * rot
        # Its reference from the filtered voltage: on the measured one, that ring would pass straight through to its gain.
        self.v_ff += (v_dq - self.v_ff) * (1.0 - math.exp(-self.a_ff * h))
        vd = max(abs(self.v_ff), 1.0)
        # Its active current: its power set, or (holding its DC voltage) what keeps its energy.
        if self.mode == 'vdc':
            p_ref = self.k_w * (w_total - self.w_nom) + v_dc * float(np.sum(i_u + i_l)) / 2.0
        else:
            p_ref = self.p_set_now
        i_ref = complex(p_ref, -self.q_set) / (1.5 * vd)
        mag = abs(i_ref)
        if mag > self.i_max:
            i_ref *= self.i_max / mag
            self.limited_time += h
        err = i_ref - i_dq
        self.int_dq += self.ki * err * h
        e_dq = self.v_ff + complex(self.r_ac, (self.w0 + dw) * self.l_ac) * i_dq + self.kp * err + self.int_dq
        reach = 0.5 * v_dc
        if abs(e_dq) > reach:
            self.int_dq -= self.ki * err * h         # held while limited
            e_dq *= reach / abs(e_dq)
            self.limited_time += h
        # Its voltages over the next step.
        th_next = theta + (self.w0 + dw) * h / 2.0
        e_s = e_dq * complex(math.cos(th_next), math.sin(th_next))
        e = np.array([(e_s * cmath.exp(-1j * k * A120)).real for k in range(3)])
        # Its DC side's current: what its AC side delivers and what its energy needs, or what holds its DC voltage.
        p_ac = float(np.dot(e, i_ac))
        if self.mode == 'vdc':
            error = v_dc - self.vdc_set
            self.vdc_int += self.vdc_ki * error * h
            i_dc_ref = self.vdc_kp * error + self.vdc_int
        else:
            i_dc_ref = (p_ac + self.k_w * (self.w_nom - w_total)) / max(v_dc, 1.0)
        i_c = (mean_u + mean_l) / 2.0
        common = v_dc / 2.0 - self.r_arm * i_dc_ref / 3.0 + self.kc * (i_c - i_dc_ref / 3.0)
        if self.n_sm:
            for side, (order, i_now) in enumerate(((common - e, i_u), (common + e, i_l))):
                volts = np.zeros(3)
                for k in range(3):
                    pol, volts[k] = self._choose(order[k], self.vc[side][k], i_now[k], self.pol[side][k])
                    self.switchings += int(np.count_nonzero(pol != self.pol[side][k]))
                    self.pol[side][k] = pol
                if side == 0:
                    self.v_u = volts
                else:
                    self.v_l = volts
        else:
            self.v_u = self._insert(common - e, self.vs_u)
            self.v_l = self._insert(common + e, self.vs_l)
        for k in range(3):
            state.set_emf(self.k_u[k], float(-self.v_u[k]))
            state.set_emf(self.k_l[k], float(-self.v_l[k]))
        self._hold_stacks(state)
        self.i_dc_now = float(np.sum(i_c))
        q = ((v[1] - v[2]) * i_ac[0] + (v[2] - v[0]) * i_ac[1] + (v[0] - v[1]) * i_ac[2]) / SQ3
        self.trace.append((t, float(np.dot(v, i_ac)), q, v_dc, self.i_dc_now, w_total))
        return False

    def _hold_stacks(self, state):
        """Each blocked-stack source at its arm's capacitor voltage, so that it is ready the moment the arm blocks."""
        for side, vs in enumerate((self.vs_u, self.vs_l)):
            for k in range(3):
                state.set_emf(self.k_src[side][k], float(-vs[k]))
                if self.full:
                    state.set_emf(self.k_src2[side][k], float(-vs[k]))

    def _blocked(self, t, h, state, i_u, i_l, v, v_dc):
        """Blocked: only the current into each stack charges it, at its capacitors' voltage."""
        for side, w in enumerate((self.w_u, self.w_l)):
            vs = self.vs_u if side == 0 else self.vs_l
            charging = np.array([max(float(state.i_rl[self.k_src[side][k]]), 0.0) for k in range(3)])
            if self.full:
                charging += np.array([max(float(state.i_rl[self.k_src2[side][k]]), 0.0) for k in range(3)])
            if self.n_sm:
                # Blocked, every submodule of the arm in the current's path: each charged alike.
                self.vc[side] += h * charging[:, None] / self.c_sm
            else:
                w += h * vs * charging
        if self.n_sm:
            self._sm_state()
        else:
            self.vs_u = np.sqrt(2.0 * self.w_u / self.c_arm)
            self.vs_l = np.sqrt(2.0 * self.w_l / self.c_arm)
        self._hold_stacks(state)
        i_ac = i_u - i_l
        self.i_dc_now = float(np.sum(i_u + i_l)) / 2.0
        q = ((v[1] - v[2]) * i_ac[0] + (v[2] - v[0]) * i_ac[1] + (v[0] - v[1]) * i_ac[2]) / SQ3
        self.trace.append((t, float(np.dot(v, i_ac)), q, v_dc, self.i_dc_now, float(np.sum(self.w_u) + np.sum(self.w_l))))
        return False

    def _block(self, t, state):
        """Every switch off: its controlled voltages to nothing, each arm left to its stack's diodes."""
        if self.blocked_at is not None:
            return False
        self.blocked_at = t
        for k in range(3):
            state.set_emf(self.k_u[k], 0.0)
            state.set_emf(self.k_l[k], 0.0)
        for side in range(2):
            for k in range(3):
                state.set_switch(self.k_sw[side][k], False)
        self._hold_stacks(state)
        return True

    def resume(self, t, state, ramp_s=0.0):
        """Deblocked: its arms back under control, its current loop starting afresh."""
        if self.blocked_at is None:
            return False
        self.blocked_at = None
        self.int_dq = 0j
        for side in range(2):
            for k in range(3):
                state.set_switch(self.k_sw[side][k], True)
        return True
