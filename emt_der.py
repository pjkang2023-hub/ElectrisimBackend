"""
The microgrid's sources and stores in the EMT study: each built into the
circuit at its terminal node, with its controller (``control(t, state)``)
and its trace.

  Battery         its open-circuit voltage by state of charge (a controlled
                  source, its state of charge counted from its current) behind
                  R0 and its inductance, then its RC pair R1 || C1:
                      v = OCV(SoC) - R0 i - v1,  C1 dv1/dt = i - v1 / R1.
  Supercapacitor  its capacitance, its leakage resistance across it, behind its
                  ESR and ESL.
  Flywheel        its DC link, a capacitor its machine converter holds at its
                  set voltage (a PI loop, its output current fed forward), the
                  rotor's power limited to P_rated min(1, w / w_base) and its
                  energy, 1/2 J w^2, following what it gives.
  SOFC system     the Padulles stack: the hydrogen, water and oxygen partial
                  pressures, each its own first-order lag, the fuel processor's
                  lag on the hydrogen fed, the Nernst voltage behind its ohmic
                  resistance; its balance of plant a constant-power load on it.
  PV array        its I-V curve, a nonlinear current source at its irradiance
                  and cell temperature (an event can step them).
"""
import math

import numpy as np

import der_electrisim as der

FARADAY = der.FARADAY
R_GAS = der.R_GAS

# The Padulles stack's time constants (s): hydrogen, water, oxygen, the fuel processor.
TAU_H2, TAU_H2O, TAU_O2, TAU_F = 26.1, 78.3, 2.91, 5.0


class _Model:
    kind = ''

    def __init__(self, obj, label):
        self.obj, self.label = obj, label
        self.trace = []           # (t, v terminal, i out, state)
        self.t_last = 0.0

    def _dt(self, t):
        dt = t - self.t_last
        self.t_last = t
        # A sliver of a step - the solver landing on an event or the run's end - is integrated, not traced:
        # its currents, a fraction of a step after the last, are not the steady values the trace reports.
        prev = getattr(self, 'dt_prev', None)
        self.sliver = prev is not None and 0 < dt < 0.1 * prev
        if dt > 0 and not self.sliver:
            self.dt_prev = dt
        return dt

    def _trace(self, row):
        if not getattr(self, 'sliver', False):
            self.trace.append(row)

    def soc_percent(self):
        return None


class BatteryEmt(_Model):
    kind = 'Battery'

    def __init__(self, ckt, obj, label, node, v0, i0):
        super().__init__(obj, label)
        ocv = obj.ocv()
        inner = ckt.node(f'{label} cells', v0 + obj.r1 * i0)
        self.k = ckt.add_rl(0, inner, obj.r0, obj.l, i0=i0, e0=ocv, controlled=True)
        ckt.add_r(inner, node, obj.r1)
        ckt.add_c(inner, node, obj.c1, w0=obj.r1 * i0)
        self.node, self.inner = node, inner
        self.trace.append((0.0, v0, i0, obj.soc0))

    def control(self, t, state):
        dt = self._dt(t)
        if dt <= 0:
            return False
        obj = self.obj
        i = float(state.i_rl[self.k])
        charge = i * dt if i >= 0 else i * obj.eta_charge * dt
        obj.soc0 = min(max(obj.soc0 - charge / (obj.ah * 3600.0), 0.0), 1.0)
        state.set_emf(self.k, obj.ocv())
        self._trace((t, float(state.v[self.node]), i, obj.soc0))
        return False

    def soc_percent(self):
        return 100.0 * self.obj.soc0


class SupercapacitorEmt(_Model):
    kind = 'Supercapacitor'

    def __init__(self, ckt, obj, label, node, v0, i0):
        super().__init__(obj, label)
        v_cap = v0 + obj.esr * i0
        self.cap = ckt.node(f'{label} capacitance', v_cap)
        ckt.add_c(0, self.cap, obj.c, w0=-v_cap)
        ckt.add_r(self.cap, 0, obj.r_leak)
        self.k = ckt.add_rl(self.cap, node, obj.esr, obj.esl, i0=i0)
        self.node = node
        obj.v0 = v_cap
        self.trace.append((0.0, v0, i0, v_cap))

    def control(self, t, state):
        if self._dt(t) <= 0:
            return False
        self.obj.v0 = float(state.v[self.cap])
        self._trace((t, float(state.v[self.node]), float(state.i_rl[self.k]), self.obj.v0))
        return False

    def soc_percent(self):
        o = self.obj
        span = o.v_rated ** 2 - o.v_min ** 2
        return 100.0 * (o.v0 ** 2 - o.v_min ** 2) / span if span > 0 else 0.0


class FlywheelEmt(_Model):
    """Its machine converter holds its DC link at its set voltage, within the rotor's power limit and energy."""
    kind = 'Flywheel'

    def __init__(self, ckt, obj, label, node, v0, i0, bandwidth_hz=100.0):
        super().__init__(obj, label)
        self.v_ref = obj.v_dc
        v_link = v0 + obj.r_dc * i0
        self.c_link = 4e-3 * obj.p_rated / obj.v_dc ** 2           # 2 ms of its rating stored
        self.link = ckt.node(f'{label} DC link', v_link)
        ckt.add_c(0, self.link, self.c_link, w0=-v_link)
        self.k_out = ckt.add_rl(self.link, node, max(obj.r_dc, 1e-6), 0.0, i0=i0)
        self.k_src = ckt.add_isrc(0, self.link, i0=i0)
        w = 2.0 * math.pi * bandwidth_hz
        self.kp, self.ki = 2.0 * 0.7 * w * self.c_link, w * w * self.c_link
        self.int_v = 0.0
        self.p_m = v_link * i0                     # the power its rotor gives the link
        self.limited_time = 0.0
        self.node = node
        self.trace.append((0.0, v0, i0, obj.s0))

    def p_max(self):
        o = self.obj
        return o.p_rated * min(1.0, o.s0 / o.s_base) if o.s0 > o.s_min + 1e-9 else 0.0

    def control(self, t, state):
        dt = self._dt(t)
        if dt <= 0:
            return False
        o = self.obj
        # Its rotor gave the link p_m over the step: its energy falls by that, its losses with it.
        e = o.e_max * o.s0 ** 2
        mech = self.p_m / o.eta if self.p_m >= 0 else self.p_m * o.eta
        e = min(max(e - (mech + o.standby * o.e_max / 3600.0) * dt, 0.0), o.e_max)
        o.s0 = math.sqrt(e / o.e_max)
        v = max(float(state.v[self.link]), 1.0)
        i_out = float(state.i_rl[self.k_out])
        err = self.v_ref - v
        i_ref = i_out + self.kp * err + self.int_v
        p = v * i_ref
        p_dis = self.p_max()
        p_ch = o.p_rated if o.s0 < 1.0 - 1e-9 else 0.0
        if p > p_dis or p < -p_ch:
            p = min(max(p, -p_ch), p_dis)
            self.limited_time += dt
        else:
            self.int_v += self.ki * err * dt
        self.p_m = p
        state.set_source_value(self.k_src, p / v)
        self._trace((t, float(state.v[self.node]), i_out, o.s0))
        return False

    def soc_percent(self):
        o = self.obj
        span = 1.0 - o.s_min ** 2
        return 100.0 * (o.s0 ** 2 - o.s_min ** 2) / span if span > 0 else 0.0


class SofcEmt(_Model):
    """
    The Padulles stack, scaled as der_electrisim.Sofc: its cells in series a
    multiple of the reference stack's voltage, its stacks in parallel of its
    current. The pressures and fuel flow are the reference stack's, at its
    current I = i / n_parallel; the fuel processor feeds 2 Kr I / U.
    """
    kind = 'SOFC'

    def __init__(self, ckt, obj, label, node, v0, i0):
        super().__init__(obj, label)
        self.kr = obj.N0 / (4.0 * FARADAY * 1e3)
        self.scale_v = obj.n_series / obj.N0
        self.aux_w = obj.aux_frac * obj.p_rated
        i_stack = i0 + self.aux_w / max(v0, 1.0)
        i_ref = i_stack / obj.n_parallel
        # Its steady state at that current.
        self.q_h2 = 2.0 * self.kr * i_ref / obj.u
        self.p_h2 = max(self.q_h2 - 2.0 * self.kr * i_ref, 1e-12) / obj.K_H2
        self.p_h2o = 2.0 * self.kr * i_ref / obj.K_H2O
        self.p_o2 = max(self.q_h2 / obj.R_HO - self.kr * i_ref, 1e-12) / obj.K_O2
        self.r = obj.R_STACK * self.scale_v / obj.n_parallel
        v_stack = v0
        inner = ckt.node(f'{label} stack', v_stack)
        self.k = ckt.add_rl(0, inner, self.r, 0.0, i0=i_stack, e0=self.nernst() * obj.n_series, controlled=True)
        self.k_out = ckt.add_rl(inner, node, 1e-6, 0.0, i0=i0)
        self.k_aux = ckt.add_isrc(inner, 0, i0=self.aux_w / max(v_stack, 1.0))
        self.inner, self.node = inner, node
        self.trace.append((0.0, v0, i0, self.p_h2))

    def nernst(self):
        o = self.obj
        return o.E0_CELL + R_GAS * o.T_K / (2.0 * FARADAY) * math.log(self.p_h2 * math.sqrt(self.p_o2) / self.p_h2o)

    def control(self, t, state):
        dt = self._dt(t)
        if dt <= 0:
            return False
        o = self.obj
        i_ref = float(state.i_rl[self.k]) / o.n_parallel
        a = lambda tau: 1.0 - math.exp(-dt / tau)
        self.q_h2 += (2.0 * self.kr * i_ref / o.u - self.q_h2) * a(TAU_F)
        self.p_h2 += ((self.q_h2 - 2.0 * self.kr * i_ref) / o.K_H2 - self.p_h2) * a(TAU_H2)
        self.p_h2o += (2.0 * self.kr * i_ref / o.K_H2O - self.p_h2o) * a(TAU_H2O)
        self.p_o2 += ((self.q_h2 / o.R_HO - self.kr * i_ref) / o.K_O2 - self.p_o2) * a(TAU_O2)
        self.p_h2, self.p_o2, self.p_h2o = max(self.p_h2, 1e-9), max(self.p_o2, 1e-9), max(self.p_h2o, 1e-9)
        state.set_emf(self.k, self.nernst() * o.n_series)
        v_stack = max(float(state.v[self.inner]), 1.0)
        state.set_source_value(self.k_aux, self.aux_w / v_stack)
        self._trace((t, float(state.v[self.node]), float(state.i_rl[self.k_out]), self.p_h2))
        return False


class PvEmt(_Model):
    """Its array's I-V curve: a nonlinear source into its node, at its irradiance and temperature."""
    kind = 'PV Array'

    def __init__(self, ckt, obj, label, node, v0, i0):
        super().__init__(obj, label)
        self.k = ckt.add_nonlinear(node, 0, self.current)
        self.node = node
        self.trace.append((0.0, v0, i0, obj.g))

    def current(self, v, t):
        """Into the circuit from its node: minus the array's current, and its derivative by v."""
        o = self.obj
        scale = o.n_p * (1.0 - o.loss)
        vm = v / o.n_s
        i = o._i_module(max(vm, 0.0), o.ipv, o.i0, o.vt) if vm > 0 else o._i_module(0.0, o.ipv, o.i0, o.vt)
        if vm <= 0:
            return -scale * i, 0.0
        # dI/dV of I = Ipv - I0 (exp((V + Rs I)/(a Vt)) - 1) - (V + Rs I)/Rp, implicitly.
        g = o.i0 / (o.a * o.vt) * math.exp(min((vm + o.rs * i) / (o.a * o.vt), 700.0)) + 1.0 / o.rp
        di = -g / (1.0 + o.rs * g)
        return -scale * max(i, 0.0), -scale * di / o.n_s if i > 0 else 0.0

    def step_irradiance(self, g, t_amb=None):
        o = self.obj
        o.g = max(g, 0.0)
        if t_amb is not None:
            o.t_amb = t_amb
        o._conditions(o.g, o.t_amb)

    def control(self, t, state):
        if self._dt(t) <= 0:
            return False
        v = float(state.v[self.node])
        self._trace((t, v, -self.current(v, t)[0], self.obj.g))
        return False


MODELS = {'Battery': BatteryEmt, 'Supercapacitor': SupercapacitorEmt, 'Flywheel': FlywheelEmt,
          'SOFC': SofcEmt, 'PV Array': PvEmt}


def build(ckt, obj, label, node, v0, i0):
    """An element's EMT model at ``node``, at its load-flow voltage ``v0`` (V) and current ``i0`` (A, delivering)."""
    return MODELS[obj.kind](ckt, obj, label, node, v0, i0)


# --- The DC/DC converter's modes, as its power or voltage reference ------------------------

class Mppt:
    """
    Perturb and observe on its PV array's voltage: its converter holds its
    input at a reference (a PI loop on the input current it draws); every
    period the reference moves a step, the same way if the power rose, back
    if it fell.
    """

    def __init__(self, pv, conv, v_ref, period_s=0.02, step_frac=0.005, bandwidth_hz=50.0, c_in=None):
        self.pv, self.conv = pv, conv
        self.v_ref = v_ref
        self.period, self.step = period_s, step_frac * pv.obj.n_s * pv.obj.v_oc_t
        self.dir = -1.0
        self.t_next = period_s
        self.p_last = None
        self.acc_p, self.acc_t = 0.0, 0.0
        c = c_in if c_in is not None else conv.c_in
        w = 2.0 * math.pi * bandwidth_hz
        self.kp, self.ki = 2.0 * 0.7 * w * c, w * w * c
        self.int_i = 0.0
        self.i_in = None
        self.t_last = 0.0
        self.trace = []         # (t, v reference, p in)

    def __call__(self, t, v_in, v_out, i_in):
        dt = t - self.t_last
        self.t_last = t
        if self.i_in is None:
            self.i_in = i_in
        self.acc_p += v_in * i_in * dt
        self.acc_t += dt
        if t >= self.t_next - 1e-12 and self.acc_t > 0:
            p = self.acc_p / self.acc_t
            if self.p_last is not None and p < self.p_last:
                self.dir = -self.dir
            self.p_last = p
            self.v_ref += self.dir * self.step
            self.acc_p = self.acc_t = 0.0
            self.t_next += self.period
            self.trace.append((t, self.v_ref, p))
        # The input current it draws: more when its input is above its reference.
        err = v_in - self.v_ref
        i_draw = i_in + self.kp * err + self.int_i
        self.int_i += self.ki * err * dt
        return v_in * i_draw            # the power its input draws (its stage loss taken by the converter)


class VscMppt:
    """
    Perturb and observe for a PV array on a grid-following PCS's DC link: the
    PCS holds its DC voltage at a reference, which moves a step every period -
    the same way if the array's power rose, back if it fell.
    """

    def __init__(self, pv, v_ref, period_s=0.02, step_frac=0.005):
        self.pv, self.v_ref = pv, v_ref
        self.period, self.step = period_s, step_frac * pv.obj.n_s * pv.obj.v_oc_t
        self.dir = -1.0
        self.t_next = period_s
        self.p_last = None
        self.acc_p, self.acc_t, self.t_last = 0.0, 0.0, 0.0
        self.trace = []

    def __call__(self, t, v_dc, i_load):
        dt = t - self.t_last
        self.t_last = t
        self.acc_p += -v_dc * i_load * max(dt, 0.0)       # the array feeds the link: its current from the terminals
        self.acc_t += max(dt, 0.0)
        if t >= self.t_next - 1e-12 and self.acc_t > 0:
            p = self.acc_p / self.acc_t
            if self.p_last is not None and p < self.p_last:
                self.dir = -self.dir
            self.p_last = p
            self.v_ref += self.dir * self.step
            self.acc_p = self.acc_t = 0.0
            self.t_next += self.period
            self.trace.append((t, self.v_ref, p))
        return self.v_ref


class Smoothing:
    """
    Its store supplies the fast part of its output bus's loads:
    P_store = share (P_rack - LPF_tau(P_rack)) + k (SoC - SoC_ref) P_rated, held
    to the store's and the converter's limits.
    """

    def __init__(self, store, loads, tau, soc_ref, gain, rated_w, share=1.0):
        self.store, self.loads = store, loads          # loads: [(node, func)]
        self.tau, self.soc_ref, self.k, self.rated, self.share = tau, soc_ref, gain, rated_w, share
        self.y = None
        self.t_last = 0.0
        self.trace = []           # (t, P_rack, P_store)
        self.limited_time = 0.0

    def rack_power(self, t, state):
        return sum(float(state.v[n]) * func(float(state.v[n]), t)[0] for n, func in self.loads)

    def __call__(self, t, state):
        dt = t - self.t_last
        self.t_last = t
        p_rack = self.rack_power(t, state)
        self.y = p_rack if self.y is None else self.y + (p_rack - self.y) * (1.0 - math.exp(-max(dt, 0.0) / self.tau))
        soc = self.store.soc_percent() or 0.0
        p = self.share * (p_rack - self.y) + self.k * (soc - self.soc_ref) / 100.0 * self.rated
        lim = self.rated
        dis, ch = _limits(self.store)
        hi, lo = min(lim, dis), -min(lim, ch)
        if not lo <= p <= hi:
            p = min(max(p, lo), hi)
            self.limited_time += dt
        self.trace.append((t, p_rack, p))
        return p


def _limits(store):
    """What a store can give and take now (W)."""
    o = store.obj
    if store.kind == 'Battery':
        v = o.ocv()
        return (o.c_rate_dis * o.ah * v if o.soc0 > o.soc_min + 1e-9 else 0.0,
                o.c_rate_ch * o.ah * v if o.soc0 < o.soc_max - 1e-9 else 0.0)
    if store.kind == 'Flywheel':
        return store.p_max(), (o.p_rated if o.s0 < 1.0 - 1e-9 else 0.0)
    return o.p_limits()


def waveform(trace, n=400):
    """A trace thinned to some ``n`` points for the results."""
    if not trace:
        return []
    k = max(1, len(trace) // n)
    rows = trace[::k]
    if rows[-1] is not trace[-1]:
        rows.append(trace[-1])
    return rows
