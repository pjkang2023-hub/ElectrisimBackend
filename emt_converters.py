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
V_F = 1.0     # V: a converter diode's forward voltage
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


def stage_input(p_out, eta, p_nl):
    """What a converter stage draws at its input for ``p_out`` delivered: losses either way (as the load flow)."""
    return p_out / eta + p_nl if p_out >= 0 else p_out * eta + p_nl


def stage_output(p_in, eta, p_nl):
    """What a converter stage delivers for ``p_in`` drawn at its input: stage_input's inverse."""
    p = p_in - p_nl
    return p * eta if p >= 0 else p / eta


# --- The converters' ratings and capacitors (the DC fault study's too) -----------------

def vsc_rating(rated_mva, p_mw, q_mvar):
    """A VSC's rating (VA): as given, or 1.25 times its load-flow power."""
    return (rated_mva if rated_mva > 0 else max(1.25 * math.hypot(p_mw, q_mvar), 0.05)) * 1e6


def vsc_dc_link(c_link_mf, s_rated, v_dc_nom):
    """A VSC's DC-link capacitance (F): as given, or 4 ms of its rating stored."""
    return c_link_mf * 1e-3 if c_link_mf > 0 else 8e-3 * s_rated / v_dc_nom ** 2


def dcdc_rating(rated_mw, p_out_mw):
    """A DC/DC converter's rating (W): as given, or 1.25 times its load-flow output."""
    return rated_mw * 1e6 if rated_mw > 0 else max(1.25 * abs(p_out_mw) * 1e6, 1e4)


def dcdc_capacitors(c_out_mf, rated, vn_in, vn_out):
    """A DC/DC converter's input and output capacitances (F): each 2 ms of its rating stored, its output's unless given."""
    c_out = c_out_mf * 1e-3 if c_out_mf > 0 else 4e-3 * rated / vn_out ** 2
    return 4e-3 * rated / vn_in ** 2, c_out


class Vsc:
    """One VSC, built into the circuit, with its controller (``control(t, state)``)."""

    @classmethod
    def from_net(cls, builder, vi, label, bus_dc, term, r_dc, block_pu):
        """A VSC of the network's VSC table, from its row and its load-flow results."""
        net = builder.net
        res = vi in net.res_vsc.index
        col = lambda c, d: _f(net.vsc.at[vi, c], d) if c in net.vsc.columns else d
        p, q = (_f(net.res_vsc.at[vi, 'p_mw']), _f(net.res_vsc.at[vi, 'q_mvar'])) if res else (0.0, 0.0)
        p_dc = _f(net.res_vsc.at[vi, 'p_dc_mw']) if res else p
        return cls(builder, label, int(net.vsc.at[vi, 'bus']), bus_dc, term, r_dc, block_pu, p=p, q=q, p_dc=p_dc,
                   rated_mva=col('rated_mva', 0.0), limit_pu=col('current_limit_pu', 1.2),
                   c_link_mf=col('dc_link_mf', 0.0), r_ohm=col('r_ohm', 0.0), x_ohm=col('x_ohm', 0.0),
                   mode_dc=str(net.vsc.at[vi, 'control_mode_dc'] if 'control_mode_dc' in net.vsc.columns else 'vm_pu'),
                   mode_ac=str(net.vsc.at[vi, 'control_mode_ac'] if 'control_mode_ac' in net.vsc.columns else 'q_mvar'),
                   model=str(net.vsc.at[vi, 'emt_model']) if 'emt_model' in net.vsc.columns else 'average',
                   switching_khz=col('switching_khz', 5.0))

    def __init__(self, builder, label, ac_bus, bus_dc, term, r_dc, block_pu, *, p, q, p_dc, rated_mva=0.0,
                 limit_pu=1.2, c_link_mf=0.0, r_ohm=0.0, x_ohm=0.0, mode_dc='vm_pu', mode_ac='q_mvar',
                 model='average', switching_khz=5.0, eta=1.0, p_nl_mw=0.0, input_side='dc'):
        """
        ``p``, ``q``: its load-flow power at its AC bus (the load convention);
        ``p_dc``: at its DC bus (the load convention). ``eta``, ``p_nl_mw``: a
        converter stage's efficiency and no-load loss, its input on its
        ``input_side`` ('ac': a rectifier stage; 'dc': an inverter stage) -
        losses its DC side carries.
        """
        ckt, ac = builder.ckt, builder.ac
        self.label = label
        self.w0 = ac.w
        self.ac_nodes = ac.nodes[ac_bus]
        self.eta, self.p_nl, self.input_side = (eta if 0 < eta <= 1 else 1.0), p_nl_mw * 1e6, input_side
        self.s_rated = vsc_rating(rated_mva, p, q)
        v_ll = float(builder.net.bus.at[ac_bus, 'vn_kv']) * 1e3
        self.v_nom_peak = SQ2 * v_ll / SQ3
        self.i_max = (limit_pu if limit_pu > 0 else 1.2) * SQ2 * self.s_rated / (SQ3 * v_ll)
        self.v_dc_nom = builder.vn[bus_dc]
        v_bus = builder.v_bus[bus_dc]
        c_link = vsc_dc_link(c_link_mf, self.s_rated, self.v_dc_nom)
        self.r = max(r_ohm, 1e-6)
        self.l = max(x_ohm, 1e-6) / self.w0
        self.mode_dc, self.mode_ac = str(mode_dc), str(mode_ac)
        self.block_v = block_pu * self.v_dc_nom
        self.block_i = 2.5 * self.i_max
        self.model = model if model in ('average', 'switching') else 'average'
        self.f_sw = (switching_khz if switching_khz > 0 else 5.0) * 1e3
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
        i_dc = -p_dc * 1e6 / max(v_bus, 1.0)
        v_link = v_bus + r_dc * i_dc

        # The circuit: DC link, its DC current (averaged only), the isolating transformer, the bridge.
        switching = self.model == 'switching'
        self.p_node = ckt.node(f'{label} DC+', v_link)
        ckt.add_c(0, self.p_node, c_link, w0=-v_link)
        self.k_dc_out = ckt.add_rl(self.p_node, term, r_dc, 0.0, i0=i_dc)
        # Averaged, its DC current; switching, the current its losses draw (the bridge's is its switches').
        p_e0 = 1.5 * (e_ph * i_ph.conjugate()).real
        self.k_src = ckt.add_isrc(0, self.p_node, i0=-(self._dc_drawn(p_e0) - p_e0) / v_link if switching else i_dc)
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
                ckt.add_diode(leg, self.p_node, V_F)
                ckt.add_diode(0, leg, V_F)
            else:
                y = ckt.node(f'{label} switches {"abc"[k]}', v_link / 2)
                self.k_e.append(ckt.add_rl(mid, y, 1e-6, 0.0, ac=(amp[k], self.w0, ph[k]), controlled=True))
                self.k_block.append(ckt.add_switch(y, x, closed=True))
                ckt.add_diode(x, self.p_node, V_F)
                ckt.add_diode(0, x, V_F)

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
        self.eta, self.p_nl, self.input_side = 1.0, 0.0, 'dc'
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
        ckt.at(self.t_next, _breakpoint, soft=None)
        ang = np.angle(e_ph) + 0.5 * self.w0 * self.t_sample - np.arange(3) * A120
        e = abs(e_ph) * np.cos(ang)
        if self.model == 'switching':
            for k, (up, dn), on, t_x in self._pwm_plan(0.0, e, v_link):
                ckt.sw[up][2], ckt.sw[dn][2] = on, not on
                if t_x is not None:
                    ckt.at(t_x, lambda st, up=up, dn=dn, on=not on: self._set_leg(st, up, dn, on), soft=True)
        else:
            self.e_held = e
            for k in range(3):
                ckt.hold_emf(self.k_e[k], float(e[k]))

    def _start(self, v_ph, i_ph, e_ph, c_link, v_link, i_dc):
        """The controller's state for the steady state given: PLL angle, integrators, references."""
        self.v_ph0, self.i_ph0, self.e_ph0 = v_ph, i_ph, e_ph   # its steady state, for a grid-forming control
        self.v_dc_ref_fn = getattr(self, 'v_dc_ref_fn', None)  # (t, v_dc, i_load) -> its DC voltage set point (MPPT)
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

    def _dc_drawn(self, p_e):
        """The power its DC side draws for ``p_e`` its bridge delivers to its AC side: its stage losses added."""
        if self.eta == 1.0 and self.p_nl == 0.0:
            return p_e
        if self.input_side == 'dc':
            return stage_input(p_e, self.eta, self.p_nl)
        return -stage_output(-p_e, self.eta, self.p_nl)

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
            state.set_source_value(self.k_src, -self._dc_drawn(float(np.dot(self.e_held, i))) / max(v_dc, 1.0))
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
            if self.v_dc_ref_fn is not None:
                self.v_dc_ref = self.v_dc_ref_fn(t_s, v_dc, i_load)
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
            if self.eta != 1.0 or self.p_nl != 0.0:
                # Its losses, by the power it passed since its last sample.
                state.set_source_value(self.k_src, -(self._dc_drawn(p_now) - p_now) / max(v_dc, 1.0))
            for k, (up, dn), on, t_x in self._pwm_plan(t_s, e, v_dc):
                changed = changed or bool(state.sw_closed[up]) != on
                self._set_leg(state, up, dn, on)
                if t_x is not None:
                    state.at(t_x, lambda st, up=up, dn=dn, on=not on: self._set_leg(st, up, dn, on), soft=True)
        else:
            for k in range(3):
                state.set_emf(self.k_e[k], float(e[k]))
            # Its DC side takes what its AC side delivers.
            self.e_held = e
            state.set_source_value(self.k_src, -self._dc_drawn(float(np.dot(e, i))) / max(float(state.v[self.p_node]), 1.0))
        if self.t_sample is not None:
            self.t_next = t_s + self.t_sample
            state.at(self.t_next, _breakpoint, soft=None)
        # Its power: P its mean since the last sample (every step's, if it samples each); Q at the fundamental, as its controls hold it.
        q_out = 1.5 * (v_q * i_d - v_d * i_q)
        self.trace.append((t_s, p_now, q_out, v_dc, math.hypot(i_d, i_q)))
        return 'soft' if changed else False


class GridFormingVsc(Vsc):
    """
    A grid-forming VSC: it sets its bridge's voltage itself - no PLL, no
    current loop - at the frequency and amplitude its droops give from the
    power and reactive power it delivers, each measured through a first-order
    filter (tau, 20 ms):
        w = w0 - m_p (P_f - P_set),   m_p = droop_pf w0 / S_rated
        E = E0 - n_q (Q_f - Q_set),   n_q = droop_qv V_peak / S_rated
    its angle the integral of w. P_set, Q_set and E0 are its load flow's, so it
    starts at w0 and holds its steady state. Above its current limit the
    voltage across its reactor is scaled back by the limit over its current,
    holding its current there (a virtual impedance growing with the overload):
    the reactor's current that its reference voltage would drive, estimated
    from the reactor's impedance.
    Its DC side draws what its bridge delivers, as an averaged VSC's.
    """

    def __init__(self, builder, label, ac_bus, bus_dc, term, r_dc, block_pu, *, droop_pf=0.02, droop_qv=0.05,
                 tau_f=0.02, **kw):
        super().__init__(builder, label, ac_bus, bus_dc, term, r_dc, block_pu, **kw)
        self._gf_start(droop_pf, droop_qv, tau_f)

    @classmethod
    def standalone(cls, droop_pf=0.02, droop_qv=0.05, tau_f=0.02, **kw):
        self = super().standalone(**kw)
        self._gf_start(droop_pf, droop_qv, tau_f)
        return self

    def _gf_start(self, droop_pf, droop_qv, tau_f):
        e_ph, v_ph, i_ph = self.e_ph0, self.v_ph0, self.i_ph0
        self.th_e = float(np.angle(e_ph))
        self.e0 = abs(e_ph)
        s0 = 1.5 * v_ph * np.conj(i_ph)
        self.p_set, self.q_set = float(s0.real), float(s0.imag)
        self.p_f, self.q_f = self.p_set, self.q_set
        self.m_p = droop_pf * self.w0 / self.s_rated
        self.n_q = droop_qv * self.v_nom_peak / self.s_rated
        self.tau_f = tau_f
        self.w_gf = self.w0
        self.freq_trace = []                 # (t, f)

    def control(self, t, state):
        if self.blocked_at is not None:
            return False
        v = state.v[self.ac_nodes]
        i = np.array([state.i_rl[k] for k in self.k_e])
        v_dc = float(state.v[self.p_node])
        i_load = float(state.i_rl[self.k_dc_out])
        if v_dc < self.block_v or float(np.max(np.abs(i))) > self.block_i:
            self._block(t, state)
            return True
        p_now = _power(v, i)
        if self.model == 'average' and self.e_held is not None:
            state.set_source_value(self.k_src, -self._dc_drawn(float(np.dot(self.e_held, i))) / max(v_dc, 1.0))
        if self.t_sample is not None:
            v_prev, vdc_prev, load_prev, p_prev = (v, v_dc, i_load, p_now) if self.v_prev is None else self.v_prev
            h = t - self.t_step
            self.acc_v += 0.5 * (v_prev + v) * h
            self.acc_vdc += 0.5 * (vdc_prev + v_dc) * h
            self.acc_p += 0.5 * (p_prev + p_now) * h
            self.v_prev, self.t_step = (v, v_dc, i_load, p_now), t
            if t < self.t_next - 1.5 * getattr(state, 'tol', 0.0) - 1e-12:
                return False
        elapsed = t - self.t_last
        self.t_last = t
        if elapsed <= 0:
            return False
        if self.t_sample is not None:
            v, v_dc = self.acc_v / elapsed, self.acc_vdc / elapsed
            p_now = self.acc_p / elapsed
            self.acc_v, self.acc_vdc, self.acc_load, self.acc_p = np.zeros(3), 0.0, 0.0, 0.0
            t_s, dt = self.t_next, self.t_sample
        else:
            t_s, dt = t, elapsed
        # Its power and reactive power, filtered; its droops.
        q_now = ((v[1] - v[2]) * i[0] + (v[2] - v[0]) * i[1] + (v[0] - v[1]) * i[2]) / SQ3
        a = 1.0 - math.exp(-dt / self.tau_f)
        self.p_f += (p_now - self.p_f) * a
        self.q_f += (q_now - self.q_f) * a
        self.w_gf = self.w0 - self.m_p * (self.p_f - self.p_set)
        theta = self.th_e + self.w_gf * dt
        e_mag = self.e0 - self.n_q * (self.q_f - self.q_set)
        e_d, e_q = e_mag, 0.0
        # Its current limit: the voltage across its reactor scaled back to what drives its limit through it -
        # the current its reference would drive, |e - v| / |r + j w l|, not the one it measured (which the last
        # sample's scaling set, and would chase).
        i_d, i_q = park(i, theta)
        i_mag = math.hypot(i_d, i_q)
        v_d, v_q = park(v, theta)
        i_est = math.hypot(e_d - v_d, e_q - v_q) / math.hypot(self.r, self.w_gf * self.l)
        if i_est > self.i_max:
            k = self.i_max / i_est
            e_d, e_q = v_d + (e_d - v_d) * k, v_q + (e_q - v_q) * k
            self.limited_time += dt
        e_max = max(v_dc, 0.0) / SQ3
        mag = math.hypot(e_d, e_q)
        if mag > e_max:
            e_d, e_q = e_d * e_max / mag, e_q * e_max / mag
        hold = self.t_sample or dt
        e = inverse_park(e_d, e_q, theta + 0.5 * self.w_gf * hold)
        self.th_e = theta
        changed = False
        if self.model == 'switching':
            for k_, (up, dn), on, t_x in self._pwm_plan(t_s, e, v_dc):
                changed = changed or bool(state.sw_closed[up]) != on
                self._set_leg(state, up, dn, on)
                if t_x is not None:
                    state.at(t_x, lambda st, up=up, dn=dn, on=not on: self._set_leg(st, up, dn, on), soft=True)
        else:
            for k_ in range(3):
                state.set_emf(self.k_e[k_], float(e[k_]))
            self.e_held = e
            state.set_source_value(self.k_src, -self._dc_drawn(float(np.dot(e, i))) / max(float(state.v[self.p_node]), 1.0))
        if self.t_sample is not None:
            self.t_next = t_s + self.t_sample
            state.at(self.t_next, _breakpoint, soft=None)
        self.trace.append((t_s, p_now, q_now, v_dc, i_mag))
        self.freq_trace.append((t_s, self.w_gf / (2 * math.pi)))
        return 'soft' if changed else False


class DcDc:
    """
    A DC/DC converter - a dual active bridge - built into the circuit, with its
    controller (``control(t, state)``).

    Two full bridges and a high-frequency transformer (ratio n = its input's
    nominal voltage to its output's, its leakage L on its output side, sized
    for its rating at a 30 degree phase shift): each bridge a square wave at
    its switching frequency, its output bridge lagging by the phase shift phi,
    so the power it passes is V1' V2 phi (pi - |phi|) / (2 pi^2 f L), at most
    at 90 degrees (some 1.8 times its rating).
    - Average-value: its output current V1' phi (pi - |phi|) / (2 pi^2 f L) into
      its output capacitor, its input drawing that power and its losses.
    - Switching: the two bridges switched, each edge at its exact instant, the
      transformer's currents starting in their periodic steady state; its
      losses a current its input draws. Its output bridge's edges follow the
      mean of the phase shift before and after each sample: a step in the
      phase shift would leave the leakage current a DC offset - one its
      windings' small resistance hardly damps - which the half step avoids.
    Its control: its output voltage (a PI loop, its output current fed
    forward), lowered with its power in droop; or its set power - or the
    power a reference function gives at each sample (``p_ref``: its PV
    array's MPPT, its SOFC's operating power, its store's smoothing power);
    the current its output bridge delivers limited
    (to its limit, and to zero in reverse unless it is bidirectional), sampled
    at each of its bridges' half periods, its voltages and output current as
    their mean since the last sample. It blocks - its bridges' switches open,
    their diodes left - when its input or output voltage falls below its
    blocking threshold.
    """

    def __init__(self, builder, label, bus_in, bus_out, term_in, term_out, *, p_in_mw, p_out_mw, mode='voltage',
                 vm_out_pu=1.0, p_set_mw=0.0, rated_mw=0.0, eta=1.0, p_nl_mw=0.0, bidirectional=False,
                 limit_pu=1.2, model='average', switching_khz=20.0, c_out_mf=0.0, block_pu=0.8, r_in=0.0, r_out=0.0,
                 every_step=False, droop_pu=0.0, p_ref=None, p_in_ref=None):
        """
        ``r_in``, ``r_out``: its terminals' resistance (0: the solver's floor).
        ``every_step``: its controller acting every step, not twice per period
        (average-value only: the benchmarks).
        ``droop_pu``: in voltage mode, its output voltage set point falls by
        this at its rated power. ``p_ref(t, state)``: in power mode, its
        output power at each sample; ``p_in_ref(t, v_in, v_out, i_in)``: the
        power its input is to draw (MPPT), its output that less its losses.
        """
        ckt = builder.ckt
        self.label = label
        self.model = model if model in ('average', 'switching') else 'average'
        self.mode = 'power' if str(mode).startswith('p') else 'voltage'
        self.eta, self.p_nl = (eta if 0 < eta <= 1 else 1.0), p_nl_mw * 1e6
        self.bidirectional = bool(bidirectional)
        vn_in, vn_out = builder.vn[bus_in], builder.vn[bus_out]
        v_in, v_out = builder.v_bus[bus_in], builder.v_bus[bus_out]
        p_out = p_out_mw * 1e6
        rated = dcdc_rating(rated_mw, p_out_mw)
        self.rated = rated
        self.i_max = (limit_pu if limit_pu > 0 else 1.2) * rated / vn_out
        self.f_sw = (switching_khz if switching_khz > 0 else 20.0) * 1e3
        self.t_sample = None if every_step and self.model == 'average' else 0.5 / self.f_sw
        self.n = vn_in / vn_out
        self.l_lk = vn_out ** 2 * 5.0 / (72.0 * self.f_sw * rated)       # its rating at 30 degrees
        self.vn_out, self.v_ref = vn_out, vm_out_pu * vn_out
        self.p_set = p_set_mw * 1e6
        self.droop, self.p_ref, self.p_in_ref = droop_pu, p_ref, p_in_ref
        self.block_in, self.block_out = block_pu * vn_in, block_pu * vn_out
        c_in, c_out = dcdc_capacitors(c_out_mf, rated, vn_in, vn_out)
        self.c_out, self.c_in = c_out, c_in
        w_v = 2 * math.pi * 300.0                            # its output voltage loop
        self.kp_v, self.ki_v = 2 * 0.7 * w_v * c_out, w_v * w_v * c_out

        # Its steady state: the load flow's power through it.
        i_out0 = p_out / max(v_out, 1.0)
        i_in0 = p_in_mw * 1e6 / max(v_in, 1.0)
        self.phi0 = self._phi(i_out0, v_in)[0]

        # Its terminals: input and output capacitors, the currents to its buses.
        self.in_node = ckt.node(f'{label} input', v_in)
        self.out_node = ckt.node(f'{label} output', v_out)
        self.k_in = ckt.add_rl(term_in, self.in_node, r_in, 0.0, i0=i_in0)
        ckt.add_c(0, self.in_node, c_in, w0=-v_in)
        self.k_out = ckt.add_rl(self.out_node, term_out, r_out, 0.0, i0=i_out0)
        ckt.add_c(0, self.out_node, c_out, w0=-v_out)
        switching = self.model == 'switching'
        # Averaged: what its input draws; switching: the current its losses draw.
        self.k_src_in = ckt.add_isrc(self.in_node, 0, i0=i_in0 - (p_out / max(v_in, 1.0) if switching else 0.0))
        self.k_src_out = None if switching else ckt.add_isrc(0, self.out_node, i0=i_out0)
        self.bridges = []
        if switching:
            # Its bridges: legs A, B on its input, C, D on its output; each leg's switches to either pole.
            legs = {}
            for name, pole, v0 in (('A', self.in_node, v_in), ('B', self.in_node, v_in),
                                   ('C', self.out_node, v_out), ('D', self.out_node, v_out)):
                x = ckt.node(f'{label} leg {name}', v0 / 2)
                legs[name] = (x, ckt.add_switch(x, pole, closed=False), ckt.add_switch(x, 0, closed=False))
                ckt.add_diode(x, pole, V_F)
                ckt.add_diode(0, x, V_F)
            self.bridges = [((legs['A'][1], legs['A'][2]), (legs['B'][1], legs['B'][2])),
                            ((legs['C'][1], legs['C'][2]), (legs['D'][1], legs['D'][2]))]
            # Its transformer: magnetising 1000 times its leakage (referred to its input), winding resistance 1e-4 pu.
            l_m = 1000.0 * self.n ** 2 * self.l_lk
            r2 = 1e-4 * vn_out ** 2 / rated
            n = self.n
            i1, i2 = self._steady_currents(v_in, v_out, self.phi0, l_m)
            ckt.add_coupled([legs['A'][0], legs['C'][0]], [legs['B'][0], legs['D'][0]],
                            np.diag([n * n * r2, r2]), np.array([[l_m, l_m / n], [l_m / n, l_m / n ** 2 + self.l_lk]]),
                            i0=[i1, i2])
        self._start(v_in, v_out, i_out0)
        self._prime(ckt)

    # --- its phase shift --------------------------------------------------------------

    def _phi(self, i_ref, v_in):
        """The phase shift for an output current ``i_ref`` at input voltage ``v_in``, and whether it saturated."""
        v1 = max(v_in, 1.0) / self.n
        x = 2.0 * math.pi ** 2 * self.f_sw * self.l_lk * abs(i_ref) / v1
        if x >= math.pi ** 2 / 4.0:
            return math.copysign(math.pi / 2.0, i_ref), True
        return math.copysign(0.5 * (math.pi - math.sqrt(math.pi ** 2 - 4.0 * x)), i_ref), False

    def _current(self, phi, v_in):
        """Its average output current at phase shift ``phi`` and input voltage ``v_in``."""
        return max(v_in, 0.0) / self.n * phi * (math.pi - abs(phi)) / (2.0 * math.pi ** 2 * self.f_sw * self.l_lk)

    def _steady_currents(self, v_in, v_out, phi, l_m):
        """
        Its transformer's currents at t = 0, its input bridge's rising edge, in
        their periodic steady state: the leakage current's change over a half
        period, which half-wave symmetry halves and negates; the magnetising
        current at its trough.
        """
        half = 0.5 / self.f_sw
        delta = phi / (2.0 * math.pi) * (2.0 * half)
        v1 = v_in / self.n
        if delta >= 0:      # its output bridge still negative until delta
            change = ((v1 + v_out) * delta + (v1 - v_out) * (half - delta)) / self.l_lk
        else:               # its output bridge positive until half + delta
            change = ((v1 - v_out) * (half + delta) + (v1 + v_out) * (-delta)) / self.l_lk
        # L di2/dt = v2 - v1': i2 falls by the change above.
        i2 = 0.5 * change
        i_m = -v_in * half / (2.0 * l_m)
        return -i2 / self.n + i_m, i2

    # --- its controller ------------------------------------------------------------------

    def _start(self, v_in, v_out, i_out0):
        self.int_v = 0.0
        self.phi = self.phi_applied = self.phi0
        self.t_last, self.t_next, self.t_step = 0.0, self.t_sample or 0.0, 0.0
        self.prev = None
        self.acc = np.zeros(4)                               # v_in, v_out, its output and input currents: their integrals
        self.blocked_at = None
        self.limited_time = 0.0
        self.trace = []                                      # (t, P in, P out, v_in, v_out, its output bridge's current)
        self.i_bridge = self._current(self.phi, v_in)
        self.v_out_last = v_out                              # at its last sample: its output capacitor's charge since

    def _set_bridge(self, state, k, positive):
        if self.blocked_at is not None:
            return
        (a_up, a_dn), (b_up, b_dn) = self.bridges[k]
        state.set_switch(a_up, positive)
        state.set_switch(a_dn, not positive)
        state.set_switch(b_up, not positive)
        state.set_switch(b_dn, positive)

    def _secondary_plan(self, t, half, phi):
        """Its output bridge's next edge: (time, its polarity then)."""
        delta = phi / (2.0 * math.pi) * (2.0 * self.t_sample)
        if delta >= 0:
            return t + delta, half % 2 == 0
        return t + self.t_sample + delta, (half + 1) % 2 == 0

    def _prime(self, ckt):
        """Its first half period: its sample's breakpoint and, switching, its bridges' states and output edge."""
        if self.t_sample is None:
            return
        ckt.at(self.t_next, _breakpoint, soft=None)
        if self.model != 'switching':
            return
        (a_up, a_dn), (b_up, b_dn) = self.bridges[0]
        for k, on in ((a_up, True), (a_dn, False), (b_up, False), (b_dn, True)):
            ckt.sw[k][2] = on
        t_x, pos = self._secondary_plan(0.0, 0, self.phi)
        (c_up, c_dn), (d_up, d_dn) = self.bridges[1]
        start = not pos                                      # before its edge, the other polarity
        for k, on in ((c_up, start), (c_dn, not start), (d_up, not start), (d_dn, start)):
            ckt.sw[k][2] = on
        ckt.at(t_x, lambda st, pos=pos: self._set_bridge(st, 1, pos), soft=True)

    def _block(self, t, state):
        self.blocked_at = t
        for bridge in self.bridges:
            for up, dn in bridge:
                state.set_switch(up, False)
                state.set_switch(dn, False)
        state.set_source_value(self.k_src_in, 0.0)
        if self.k_src_out is not None:
            state.set_source_value(self.k_src_out, 0.0)

    def control(self, t, state):
        if self.blocked_at is not None:
            return False
        v_in, v_out = float(state.v[self.in_node]), float(state.v[self.out_node])
        i_out = float(state.i_rl[self.k_out])
        if v_in < self.block_in or v_out < self.block_out:
            self._block(t, state)
            return True
        self.i_in_now = float(state.i_rl[self.k_in])
        changed = self._sample(t, state, v_in, v_out, i_out)
        changed = 'soft' if changed else False
        if self.k_src_out is not None:
            # Averaged, step by step: its output current at its phase shift, its input drawing that power and its losses.
            self.i_bridge = self._current(self.phi, v_in)
            state.set_source_value(self.k_src_out, self.i_bridge)
            state.set_source_value(self.k_src_in, stage_input(v_out * self.i_bridge, self.eta, self.p_nl) / max(v_in, 1.0))
        return changed

    def _sample(self, t, state, v_in, v_out, i_out):
        """Its controller, at its samples: its phase shift, and, switching, its bridges' next edges."""
        if self.t_sample is None:
            elapsed = t - self.t_last
            self.t_last = t
            if elapsed <= 0:
                return False
            v_in_m, v_out_m, i_out_m, i_in_m = v_in, v_out, i_out, self.i_in_now
            i_bridge_m = self.i_bridge
            t_s, dt = t, elapsed
        else:
            now = np.array([v_in, v_out, i_out, self.i_in_now])
            prev = now if self.prev is None else self.prev
            self.acc += 0.5 * (prev + now) * (t - self.t_step)
            self.prev, self.t_step = now, t
            if t < self.t_next - 1.5 * getattr(state, 'tol', 0.0) - 1e-12:
                return False
            elapsed = t - self.t_last
            self.t_last = t
            if elapsed <= 0:
                return False
            v_in_m, v_out_m, i_out_m, i_in_m = self.acc / elapsed
            self.acc = np.zeros(4)
            # Its output bridge's mean current: what reached its bus, and what charged its output capacitor.
            i_bridge_m = i_out_m + self.c_out * (v_out - self.v_out_last) / elapsed
            t_s, dt = self.t_next, self.t_sample
        self.v_out_last = v_out
        # Its output current: for its output voltage, or its set power; limited.
        if self.mode == 'voltage':
            # In droop its set point falls with the power it delivers.
            v_ref = self.v_ref * (1.0 - self.droop * v_out_m * i_out_m / self.rated) if self.droop else self.v_ref
            err = v_ref - v_out_m
            i_ref = i_out_m + self.kp_v * err + self.int_v
        else:
            err = 0.0
            if self.p_in_ref is not None:
                self.p_set = stage_output(self.p_in_ref(t_s, v_in_m, v_out_m, i_in_m), self.eta, self.p_nl)
            elif self.p_ref is not None:
                self.p_set = self.p_ref(t_s, state)
            i_ref = self.p_set / max(v_out_m, 0.05 * self.vn_out)
        lo = -self.i_max if self.bidirectional else 0.0
        limited = not lo <= i_ref <= self.i_max
        i_ref = min(max(i_ref, lo), self.i_max)
        self.phi, saturated = self._phi(i_ref, v_in_m)
        limited = limited or saturated
        if limited:
            self.limited_time += dt
        elif self.mode == 'voltage':
            self.int_v += self.ki_v * err * dt
        changed = False
        if self.model == 'switching':
            # Its losses, by the power it passed since its last sample.
            p_out = v_out_m * i_bridge_m
            state.set_source_value(self.k_src_in, (stage_input(p_out, self.eta, self.p_nl) - p_out) / max(v_in, 1.0))
            half = int(round(t_s / self.t_sample))
            self._set_bridge(state, 0, half % 2 == 0)
            changed = True
            t_x, pos = self._secondary_plan(t_s, half, 0.5 * (self.phi_applied + self.phi))
            self.phi_applied = self.phi
            state.at(t_x, lambda st, pos=pos: self._set_bridge(st, 1, pos), soft=True)
        if self.t_sample is not None:
            self.t_next = t_s + self.t_sample
            state.at(self.t_next, _breakpoint, soft=None)
        p_out = v_out_m * i_bridge_m
        self.trace.append((t_s, stage_input(p_out, self.eta, self.p_nl), p_out, v_in_m, v_out_m, i_bridge_m))
        return changed
