"""
Converters in the EMT study: a VSC's average-value model and its controls.

The VSC's switches are averaged: each phase is a voltage the controller sets
every step, from its DC link's midpoint, behind its reactor (the leakage of an
isolating converter transformer, which keeps the AC ground and the DC negative
pole apart). Its DC side draws what its AC side delivers - a current
P / v_dc into its DC-link capacitor - losses in its reactor aside. Its diodes
stay in the circuit: they conduct only beyond the DC rails, as when it blocks.

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
integrators set to hold it.
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


class VscAverage:
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

        # Its steady state: the AC current it delivers, and the voltage behind its reactor.
        i_ph = ac._injection_phasor(ac_bus, -p, -q)        # its results use the load convention
        v_ph = ac._v_phasor(ac_bus)
        e_ph = v_ph + complex(self.r, self.w0 * self.l) * i_ph
        # Its DC current, as its load flow has it at its DC bus: what its AC side
        # takes, less its reactor's losses.
        p_dc = _f(net.res_vsc.at[vi, 'p_dc_mw']) if vi in net.res_vsc.index else p
        i_dc = -p_dc * 1e6 / max(v_bus, 1.0)
        v_link = v_bus + r_dc * i_dc

        # The circuit: DC link, its DC current, the isolating transformer, the averaged phases.
        self.p_node = ckt.node(f'{label} DC+', v_link)
        ckt.add_c(0, self.p_node, c_link, w0=-v_link)
        self.k_dc_out = ckt.add_rl(self.p_node, term, r_dc, 0.0, i0=i_dc)
        self.k_src = ckt.add_isrc(0, self.p_node, i0=i_dc)
        mid = ckt.node(f'{label} bridge star point', v_link / 2)
        ckt.add_r(mid, self.p_node, R_BIAS)
        ckt.add_r(mid, 0, R_BIAS)
        # Its isolating transformer's magnetising: 1000 pu of its rating, 0.1 % of its current.
        l_mag = 1000.0 * v_ll * v_ll / self.s_rated / self.w0
        r_m = np.diag([0.0, self.r])
        l_m = np.array([[l_mag, l_mag], [l_mag, l_mag + self.l]])
        amp, ph = ac._three(e_ph)
        self.k_e, self.k_block = [], []
        for k in range(3):
            x = ckt.node(f'{label} bridge {"abc"[k]}', v_link / 2)
            y = ckt.node(f'{label} switches {"abc"[k]}', v_link / 2)
            ckt.add_coupled([self.ac_nodes[k], x], [0, mid], r_m, l_m)
            self.k_e.append(ckt.add_rl(mid, y, 1e-6, 0.0, ac=(amp[k], self.w0, ph[k]), controlled=True))
            self.k_block.append(ckt.add_switch(y, x, closed=True))
            ckt.add_diode(x, self.p_node)
            ckt.add_diode(0, x)

        # Its controller, started in that steady state.
        self._start(v_ph, i_ph, e_ph, c_link, v_link, i_dc)

    @classmethod
    def standalone(cls, **kw):
        """
        A controller for a circuit built by hand (the benchmarks): the
        attributes control() uses - w0, ac_nodes, k_e, k_block, k_src,
        k_dc_out, p_node, r, l, i_max, v_nom_peak, block_v, block_i, mode_dc,
        mode_ac, label, s_rated - and the steady state it starts in: v_ph,
        i_ph, e_ph (phase a, amplitude), c_link, v_link, i_dc.
        """
        self = cls.__new__(cls)
        start = {k: kw.pop(k) for k in ('v_ph', 'i_ph', 'e_ph', 'c_link', 'v_link', 'i_dc')}
        for key, value in kw.items():
            setattr(self, key, value)
        self._start(**start)
        return self

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
        self.blocked_at = None
        self.limited_time = 0.0
        self.trace = []                                      # (t, P out, Q out, v_dc, |i|) for the results

    def control(self, t, state):
        dt = t - self.t_last
        self.t_last = t
        if dt <= 0 or self.blocked_at is not None:
            return False
        v = state.v[self.ac_nodes]
        i = np.array([state.i_rl[k] for k in self.k_e])
        v_dc = float(state.v[self.p_node])
        i_load = float(state.i_rl[self.k_dc_out])
        # Block: its DC voltage too low, or its current too high.
        if v_dc < self.block_v or float(np.max(np.abs(i))) > self.block_i:
            self.blocked_at = t
            for k in self.k_block:
                state.set_switch(k, False)
            state.set_source_value(self.k_src, 0.0)
            return True
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
        e = inverse_park(ed, eq, theta + 0.5 * self.w * dt)    # held over the next step: its midpoint's angle
        self.theta = theta
        for k in range(3):
            state.set_emf(self.k_e[k], float(e[k]))
        # Its DC side takes what its AC side delivers.
        p_e = float(np.dot(e, i))
        state.set_source_value(self.k_src, -p_e / max(v_dc, 1.0))
        p_out, q_out = 1.5 * (v_d * i_d + v_q * i_q), 1.5 * (v_q * i_d - v_d * i_q)
        self.trace.append((t, p_out, q_out, v_dc, math.hypot(i_d, i_q)))
        return False
