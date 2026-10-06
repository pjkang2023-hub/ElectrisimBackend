"""
Converters in the EMT study: a VSC - average-value or switching - and its controls.

Its bridge feeds its AC bus through its reactor, the leakage of an isolating
converter transformer that keeps the AC ground and the DC negative pole apart.
- Average-value: each phase is a voltage the controller sets, from its DC
  link's midpoint. Its DC side draws what its AC side delivers - a current
  P / v_dc into its DC-link capacitor.
- Switching: a two-level bridge, each leg's switches joining its phase to the
  DC link's positive or negative pole by regular-sampled PWM - a triangular
  carrier at its switching frequency, its reference updated at each of the
  carrier's peaks and valleys, with min-max zero-sequence injection (as space
  vector modulation, reaching v_dc / sqrt 3). Each leg switches once per half
  period, at an instant computed exactly and landed on by the solver. Its DC
  current is the switches' own.
Its diodes stay in the circuit: they conduct only beyond the DC rails, as when
it blocks.

The controller samples twice per carrier period in both models - so the
average-value model is the switching model's exact average over each half
period - and holds its output until the next sample. It takes its AC
currents at the sample (at the carrier's peaks and valleys: their average over
the switching ripple), and its voltages and DC load current as their mean
since the last sample (an anti-aliasing filter): at a sample all its legs are
on one pole, so its AC bus, behind no filter capacitor, would read the
inductive divider between its reactor and the grid, and its DC current the
exchange between its DC link and the DC network's capacitors.

Controls (grid-following, per unit of its rating where it matters):
- a synchronous-reference-frame PLL on its AC bus (20 Hz);
- dq current control behind its reactor (500 Hz), with decoupling and
  voltage feed-forward, its output held within the DC link's reach (v_dc / sqrt 3);
- an outer loop by its load-flow control: its DC voltage (30 Hz, with the DC
  load's current fed forward) or its active power; its reactive power at its
  load-flow value, or its AC voltage (10 Hz);
- a current limit (1.2 pu by default), active current first, integrators
  held while limited;
- blocking - its voltages and DC current set to zero, its diodes left - when
  its DC voltage falls below a threshold or its current passes 2.5 times its
  limit.

It starts in its load-flow steady state: its voltages, currents, PLL angle and
integrators set to hold it; the switching model's currents start at its
average-value model's, its ripple building from there.
"""

import math

import numpy as np

from emt_solver import R_BIAS

SQ2, SQ3 = math.sqrt(2.0), math.sqrt(3.0)
A120 = 2.0 * math.pi / 3.0


def _f(value, default=0.0):
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def park(v, theta):
    """Amplitude-invariant dq of a three-phase set at angle theta (phase a = cos theta)."""
    c = np.cos(theta - np.arange(3) * A120)
    s = np.sin(theta - np.arange(3) * A120)
    return 2.0 / 3.0 * float(np.dot(v, c)), -2.0 / 3.0 * float(np.dot(v, s))


def inverse_park(d, q, theta):
    ang = theta - np.arange(3) * A120
    return d * np.cos(ang) - q * np.sin(ang)


def _power(v, i):
    """Instantaneous active power of three phases."""
    return float(np.dot(v, i))


def _breakpoint(state):
    """An event that changes nothing: it lands the time grid on a controller's sample."""


class Vsc:
    """One VSC, built into the circuit, with its controller (``control(t, state)``)."""

    def __init__(self, builder, vi, label, bus_dc, term, r_dc, block_pu):
        net, ckt, ac = builder.net, builder.ckt, builder.ac
        self.label = label
        self.w0 = ac.w
        ac_bus = int(net.vsc.at[vi, 'bus'])
        self.ac_nodes = ac.nodes[ac_bus]
        p, q = (_f(net.res_vsc.at[vi, 'p_mw']), _f(net.res_vsc.at[vi, 'q_mvar'])) if vi in net.res_vsc.index else (0.0, 0.0)
        col = lambda c, d: _f(net.vsc.at[vi, c], d) if c in net.vsc.columns else d
        s_rated = col('rated_mva', 0.0)
        if s_rated <= 0:
            s_rated = max(1.25 * math.hypot(p, q), 0.05)
        self.s_rated = s_rated * 1e6
        v_ll = float(net.bus.at[ac_bus, 'vn_kv']) * 1e3
        self.v_nom_peak = SQ2 * v_ll / SQ3
        limit = col('current_limit_pu', 1.2)
        self.i_max = (limit if limit > 0 else 1.2) * SQ2 * self.s_rated / (SQ3 * v_ll)
        self.v_dc_nom = builder.vn[bus_dc]
        v_bus = builder.v_bus[bus_dc]
        c_link = col('dc_link_mf', 0.0) * 1e-3
        if c_link <= 0:
            c_link = 8e-3 * self.s_rated / self.v_dc_nom ** 2     # 4 ms of its rating stored
        self.r = max(col('r_ohm', 0.0), 1e-6)
        self.l = max(col('x_ohm', 0.0), 1e-6) / self.w0
        self.mode_dc = str(net.vsc.at[vi, 'control_mode_dc'] if 'control_mode_dc' in net.vsc.columns else 'vm_pu')
        self.mode_ac = str(net.vsc.at[vi, 'control_mode_ac'] if 'control_mode_ac' in net.vsc.columns else 'q_mvar')
        self.block_v = block_pu * self.v_dc_nom
        self.block_i = 2.5 * self.i_max
        self.model = str(net.vsc.at[vi, 'emt_model']) if 'emt_model' in net.vsc.columns else 'average'
        if self.model not in ('average', 'switching'):
            self.model = 'average'
        f_khz = col('switching_khz', 5.0)
        self.f_sw = (f_khz if f_khz > 0 else 5.0) * 1e3
        self.t_sample = 0.5 / self.f_sw
        if self.model == 'switching':
            # Its current ripple, about v_dc / (6 L f_sw) peak to peak, against its rated current.
            ripple = self.v_dc_nom / (6.0 * self.l * self.f_sw) / (SQ2 * self.s_rated / (SQ3 * v_ll))
            if ripple > 0.5:
                x_pu = self.w0 * self.l * self.s_rated / v_ll ** 2
                builder.warnings.append(
                    f'VSC {label}: switching at {self.f_sw / 1e3:g} kHz, its reactor ({x_pu:.3f} pu) leaves a current '
                    f'ripple of some {100 * ripple:.0f} % of its rated current, which its controls cannot follow. '
                    'A converter reactor is some 0.1 to 0.2 pu; or raise its switching frequency.')

        # Its steady state: the AC current it delivers, and the voltage behind its reactor.
        i_ph = ac._injection_phasor(ac_bus, -p, -q)        # its results use the load convention
        v_ph = ac._v_phasor(ac_bus)
        e_ph = v_ph + complex(self.r, self.w0 * self.l) * i_ph
        # Its DC current, as its load flow has it at its DC bus: what its AC side
        # takes, less its reactor's losses.
        p_dc = _f(net.res_vsc.at[vi, 'p_dc_mw']) if vi in net.res_vsc.index else p
        i_dc = -p_dc * 1e6 / max(v_bus, 1.0)
        v_link = v_bus + r_dc * i_dc

        # The circuit: DC link, its DC current (averaged only), the isolating transformer, the bridge.
        switching = self.model == 'switching'
        self.p_node = ckt.node(f'{label} DC+', v_link)
        ckt.add_c(0, self.p_node, c_link, w0=-v_link)
        self.k_dc_out = ckt.add_rl(self.p_node, term, r_dc, 0.0, i0=i_dc)
        self.k_src = ckt.add_isrc(0, self.p_node, i0=0.0 if switching else i_dc)
        mid = ckt.node(f'{label} bridge star point', v_link / 2)
        ckt.add_r(mid, self.p_node, R_BIAS)
        ckt.add_r(mid, 0, R_BIAS)
        # Its isolating transformer's magnetising: 1000 pu of its rating, 0.1 % of its current.
        l_mag = 1000.0 * v_ll * v_ll / self.s_rated / self.w0
        r_m = np.diag([0.0, self.r])
        l_m = np.array([[l_mag, l_mag], [l_mag, l_mag + self.l]])
        amp, ph = ac._three(e_ph)
        self.k_e, self.k_block, self.legs = [], [], []
        for k in range(3):
            x = ckt.node(f'{label} bridge {"abc"[k]}', v_link / 2)
            ckt.add_coupled([self.ac_nodes[k], x], [0, mid], r_m, l_m)
            if switching:
                # Its leg: to either pole by its switches. The branch to the
                # transformer measures its current; in the AC steady state it is
                # the bridge's average voltage, the legs' switches all closed.
                leg = ckt.node(f'{label} leg {"abc"[k]}', v_link / 2)
                k_e = ckt.add_rl(leg, x, 1e-6, 0.0, ac=(amp[k], self.w0, ph[k]), controlled=True)
                ckt.hold_emf(k_e, 0.0)
                self.k_e.append(k_e)
                self.legs.append((ckt.add_switch(leg, self.p_node, closed=False, phasor_closed=True),
                                  ckt.add_switch(leg, 0, closed=False, phasor_closed=True)))
                ckt.add_diode(leg, self.p_node)
                ckt.add_diode(0, leg)
            else:
                y = ckt.node(f'{label} switches {"abc"[k]}', v_link / 2)
                self.k_e.append(ckt.add_rl(mid, y, 1e-6, 0.0, ac=(amp[k], self.w0, ph[k]), controlled=True))
                self.k_block.append(ckt.add_switch(y, x, closed=True))
                ckt.add_diode(x, self.p_node)
                ckt.add_diode(0, x)

        # Its controller, started in that steady state, and its first half period.
        self._start(v_ph, i_ph, e_ph, c_link, v_link, i_dc)
        self.prime(ckt, e_ph, v_link)

    @classmethod
    def standalone(cls, **kw):
        """
        A controller for a circuit built by hand (the benchmarks): the
        attributes control() uses - w0, ac_nodes, k_e, k_block, k_src,
        k_dc_out, p_node, r, l, i_max, v_nom_peak, block_v, block_i, mode_dc,
        mode_ac, label, s_rated, model, t_sample (None: every step) and, for
        the switching model, legs (each leg's switches to the positive and
        negative poles) - and the steady state it starts in: v_ph, i_ph, e_ph
        (phase a, amplitude), c_link, v_link, i_dc. Prime its circuit
        (prime) before running it.
        """
        self = cls.__new__(cls)
        start = {k: kw.pop(k) for k in ('v_ph', 'i_ph', 'e_ph', 'c_link', 'v_link', 'i_dc')}
        self.model, self.t_sample, self.legs, self.k_block = 'average', None, [], []
        for key, value in kw.items():
            setattr(self, key, value)
        self._start(**start)
        return self

    def prime(self, ckt, e_ph, v_link):
        """
        Its first half period, from the steady state: the voltages it holds
        until its first sample (at the half period's midpoint angle) - the
        averaged phases' e, or its legs' switch states and switching instants -
        and the event that lands the time grid on that sample.
        """
        if self.t_sample is None:
            return
        self.t_next = self.t_sample
        ckt.at(self.t_next, _breakpoint)
        ang = np.angle(e_ph) + 0.5 * self.w0 * self.t_sample - np.arange(3) * A120
        e = abs(e_ph) * np.cos(ang)
        if self.model == 'switching':
            for k, (up, dn), on, t_x in self._pwm_plan(0.0, e, v_link):
                ckt.sw[up][2], ckt.sw[dn][2] = on, not on
                if t_x is not None:
                    ckt.at(t_x, lambda st, up=up, dn=dn, on=not on: self._set_leg(st, up, dn, on))
        else:
            self.e_held = e
            for k in range(3):
                ckt.hold_emf(self.k_e[k], float(e[k]))

    def _start(self, v_ph, i_ph, e_ph, c_link, v_link, i_dc):
        """The controller's state for the steady state given: PLL angle, integrators, references."""
        self.theta = float(np.angle(v_ph))                   # its angle at the last step: t = 0
        self.w = self.w0
        v_pk = abs(v_ph)
        self.kp_pll, self.ki_pll = 2 * 0.7 * (2 * math.pi * 20) / v_pk, (2 * math.pi * 20) ** 2 / v_pk
        self.int_pll = 0.0
        a_c = 2 * math.pi * 500
        # Its PI's zero at a tenth of its bandwidth: alpha * R would leave a near-lossless reactor's integrator idle.
        self.kp_i = a_c * self.l
        self.ki_i = self.kp_i * a_c / 10.0
        i_d, i_q = abs(i_ph) * math.cos(np.angle(i_ph) - self.theta), abs(i_ph) * math.sin(np.angle(i_ph) - self.theta)
        e_d, e_q = abs(e_ph) * math.cos(np.angle(e_ph) - self.theta), abs(e_ph) * math.sin(np.angle(e_ph) - self.theta)
        v_d, v_q = v_pk, 0.0
        self.int_d = e_d - (v_d + self.r * i_d - self.w0 * self.l * i_q)
        self.int_q = e_q - (v_q + self.r * i_q + self.w0 * self.l * i_d)
        w_v = 2 * math.pi * 30
        self.kp_v, self.ki_v = 2 * 0.7 * w_v * c_link, w_v * w_v * c_link
        self.v_dc_ref = v_link
        self.p_ref = -1.5 * v_d * i_d                        # into the DC side
        self.q_out_ref = -1.5 * v_d * i_q
        i_load0 = i_dc
        self.int_v = -1.5 * v_d * i_d / v_link - i_load0     # holds i_d at the start
        self.v_ac_ref = v_pk
        # Its AC voltage loop: full current for a 10 % voltage error, some 10 Hz.
        self.kp_vac, self.ki_vac = 0.0, 2 * math.pi * 10 * self.i_max / (0.1 * self.v_nom_peak)
        self.int_vac = -i_q
        self.t_last = 0.0
        self.t_next = 0.0
        self.t_step = 0.0                                    # its voltages' integrals since the last sample
        self.v_prev = None
        self.acc_v, self.acc_vdc, self.acc_load = np.zeros(3), 0.0, 0.0
        self.acc_p = 0.0                                     # its power since the last sample, for its results
        self.e_held = None                                   # the averaged phases' voltages, held until the next sample
        self.blocked_at = None
        self.limited_time = 0.0
        self.trace = []                                      # (t, P out, Q out, v_dc, |i|) for the results

    def _pwm_plan(self, t, e, v_dc):
        """
        Each leg's switch state from t, the start of a half period of its
        carrier, and the instant it changes within it (None: it holds): its
        upper switch is on while its reference is above the carrier, which
        rises from -1 to 1 over the half periods that start at its valleys.
        """
        m = np.asarray(e, dtype=float) / max(0.5 * v_dc, 1.0)
        m = np.clip(m - 0.5 * (np.max(m) + np.min(m)), -1.0, 1.0)     # min-max zero-sequence injection
        rising = int(round(t / self.t_sample)) % 2 == 0
        plan = []
        for k, (up, dn) in enumerate(self.legs):
            if rising:
                on, frac = bool(m[k] > -1.0), 0.5 * (m[k] + 1.0)
            else:
                on, frac = bool(m[k] >= 1.0), 0.5 * (1.0 - m[k])
            t_x = t + frac * self.t_sample if 0.0 < frac < 1.0 else None
            plan.append((k, (up, dn), on, t_x))
        return plan

    def _set_leg(self, state, up, dn, on):
        if self.blocked_at is None:
            state.set_switch(up, on)
            state.set_switch(dn, not on)

    def _block(self, t, state):
        self.blocked_at = t
        for k in self.k_block:
            state.set_switch(k, False)
        for up, dn in self.legs:
            state.set_switch(up, False)
            state.set_switch(dn, False)
        state.set_source_value(self.k_src, 0.0)

    def control(self, t, state):
        if self.blocked_at is not None:
            return False
        v = state.v[self.ac_nodes]
        i = np.array([state.i_rl[k] for k in self.k_e])
        v_dc = float(state.v[self.p_node])
        i_load = float(state.i_rl[self.k_dc_out])
        # Block, checked every step: its DC voltage too low, or its current too high.
        if v_dc < self.block_v or float(np.max(np.abs(i))) > self.block_i:
            self._block(t, state)
            return True
        # Its controller runs at its samples (twice per carrier period), or every step.
        p_now = _power(v, i)
        if self.model == 'average' and self.e_held is not None:
            # Its DC side takes what its averaged phases deliver, step by step: e . i / v_dc.
            state.set_source_value(self.k_src, -float(np.dot(self.e_held, i)) / max(v_dc, 1.0))
        if self.t_sample is not None:
            v_prev, vdc_prev, load_prev, p_prev = (v, v_dc, i_load, p_now) if self.v_prev is None else self.v_prev
            h = t - self.t_step
            self.acc_v += 0.5 * (v_prev + v) * h
            self.acc_vdc += 0.5 * (vdc_prev + v_dc) * h
            self.acc_load += 0.5 * (load_prev + i_load) * h
            self.acc_p += 0.5 * (p_prev + p_now) * h
            self.v_prev, self.t_step = (v, v_dc, i_load, p_now), t
            # Its next sample: now, if within the solver's sliver of it.
            if t < self.t_next - 1.5 * getattr(state, 'tol', 0.0) - 1e-12:
                return False
        elapsed = t - self.t_last
        self.t_last = t
        if elapsed <= 0:
            return False
        if self.t_sample is not None:
            # Its voltages and DC load current: their mean since the last sample.
            v, v_dc, i_load = self.acc_v / elapsed, self.acc_vdc / elapsed, self.acc_load / elapsed
            p_now = self.acc_p / elapsed
            self.acc_v, self.acc_vdc, self.acc_load, self.acc_p = np.zeros(3), 0.0, 0.0, 0.0
            t_s, dt = self.t_next, self.t_sample                 # its sample, on its own clock
        else:
            t_s, dt = t, elapsed
        # Measured at the angle of their own time; the PLL then moves it on.
        theta = self.theta + self.w * dt
        v_d, v_q = park(v, theta)
        i_d, i_q = park(i, theta)
        self.int_pll += self.ki_pll * v_q * dt
        self.w = self.w0 + self.kp_pll * v_q + self.int_pll
        vd_ = max(v_d, 0.05 * self.v_nom_peak)
        # Outer loops.
        if self.mode_dc.startswith('vm'):
            err = self.v_dc_ref - v_dc
            p_in = v_dc * (i_load + self.kp_v * err + self.int_v)
        else:
            err = 0.0
            p_in = self.p_ref
        id_ref = -p_in / (1.5 * vd_)
        if self.mode_ac.startswith('vm'):
            e_ac = self.v_ac_ref - v_d
            iq_ref = -(self.kp_vac * e_ac + self.int_vac)
        else:
            e_ac = 0.0
            iq_ref = -self.q_out_ref / (1.5 * vd_)
        # Current limit, active current first.
        limited = False
        if abs(id_ref) > self.i_max:
            id_ref, limited = math.copysign(self.i_max, id_ref), True
        iq_max = math.sqrt(max(self.i_max ** 2 - id_ref ** 2, 0.0))
        if abs(iq_ref) > iq_max:
            iq_ref, limited = math.copysign(iq_max, iq_ref), True
        if limited:
            self.limited_time += dt
        else:
            if self.mode_dc.startswith('vm'):
                self.int_v += self.ki_v * err * dt
            if self.mode_ac.startswith('vm'):
                self.int_vac += self.ki_vac * e_ac * dt
        # Current control.
        ed = v_d + self.r * i_d - self.w * self.l * i_q + self.kp_i * (id_ref - i_d) + self.int_d
        eq = v_q + self.r * i_q + self.w * self.l * i_d + self.kp_i * (iq_ref - i_q) + self.int_q
        e_max = max(v_dc, 0.0) / SQ3
        mag = math.hypot(ed, eq)
        if mag > e_max:
            ed, eq = ed * e_max / mag, eq * e_max / mag
        else:
            self.int_d += self.ki_i * (id_ref - i_d) * dt
            self.int_q += self.ki_i * (iq_ref - i_q) * dt
        hold = self.t_sample or dt
        e = inverse_park(ed, eq, theta + 0.5 * self.w * hold)    # held until the next sample: its midpoint's angle
        self.theta = theta
        changed = False
        if self.model == 'switching':
            for k, (up, dn), on, t_x in self._pwm_plan(t_s, e, v_dc):
                changed = changed or bool(state.sw_closed[up]) != on
                self._set_leg(state, up, dn, on)
                if t_x is not None:
                    state.at(t_x, lambda st, up=up, dn=dn, on=not on: self._set_leg(st, up, dn, on))
        else:
            for k in range(3):
                state.set_emf(self.k_e[k], float(e[k]))
            # Its DC side takes what its AC side delivers.
            self.e_held = e
            state.set_source_value(self.k_src, -float(np.dot(e, i)) / max(float(state.v[self.p_node]), 1.0))
        if self.t_sample is not None:
            self.t_next = t_s + self.t_sample
            state.at(self.t_next, _breakpoint)
        # Its power: P its mean since the last sample (every step's, if it samples each); Q at the fundamental, as its controls hold it.
        q_out = 1.5 * (v_q * i_d - v_d * i_q)
        self.trace.append((t_s, p_now, q_out, v_dc, math.hypot(i_d, i_q)))
        return changed
