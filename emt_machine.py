# -*- coding: utf-8 -*-
"""
A synchronous machine in the EMT study: a gas turbine or engine with its
rotor, governor and exciter, so it shares an islanded load by droop with the
grid-forming PCS.

The classical model, per phase: its internal voltage E' behind its transient
reactance x'd (and armature resistance), on a star point of its own,

    e_k(t) = sqrt(2) E' cos(w0 t + delta + phi_k),

E' set by its exciter and delta by its rotor:

    2H d(w)/dt = (Pm - Pe) / S - D (w - 1)          (per unit of its rating S)
    d(delta)/dt = w0 (w - 1)

Pe = sum_k e_k i_k, the power at its internal voltage (the air-gap power).

Its governor (GAST's droop and lags, its load limit left out): a power order
Pset + (1 - w) / R, the valve following it with time constant T1 within
[VMIN, VMAX], the turbine's power the valve's with time constant T2.
Its exciter: a voltage regulator setting its field voltage, Efd = Efd0 +
K (V0 - V) through its lead-lag (SEXS's TA/TB and TB; none for the others)
and its lag TE, within [EMIN, EMAX] of Efd0; and E' following the field as
GENROU's d axis does,

    T'd0 d(E')/dt = Efd - E' - (xd - x'd) Id,

Id its current on the d axis, a quarter period behind E'. Without an
exciter, E' is held (the classical model). The regulator once set E'
itself, through 50 ms: in an island the turbines' voltages swung some 10 %
every 100 ms and the converters on them ran into their current limits.

It starts in the load flow's steady state: E' and delta from its terminal
voltage and current there, Pm and the valve at its power, at nominal speed.

The detailed model (ParkMachine, generator_model = detailed): GENROU's rotor,
a field and a damper on the d axis and two rotor circuits on the q axis, with
the stator's own transients, so a full Park model with x''q = x''d. In per
unit of its rating, on its rotor's axes with q leading d, id and iq the current
it delivers (id positive demagnetising):

    T'd0 d(psi'd)/dt  = Efd - psi'd - (xd - x'd) [gd1 id + gd2 (psi'd - psi1d)] - Se psi''d
    T''d0 d(psi1d)/dt = psi'd - psi1d - (x'd - xl) id
    psi''d = gd1 psi'd + (1 - gd1) psi1d,    gd1 = (x''d - xl) / (x'd - xl),
                                             gd2 = (x'd - x''d) / (x'd - xl)^2

and the same on q with Efd = 0 and Se scaled by (xq - xl) / (xd - xl). Those
are ANDES's GENROU (its e1q = psi'd, e2d = psi1d, e1d = -psi'q, e2q = -psi1q).
The stator, as a space vector on the rotor's frame:

    v = (1 / w0) d(psi)/dt + j w psi - ra i,    psi = psi'' - x'' i

so a voltage (j w psi'' + (1 / w0) d(psi'')/dt) behind ra + j x'', which is the
branch each phase has here; the branch's own L di/dt carries the DC offset
that the RMS model leaves out, and that offset reaches the rotor at the
fundamental, as the short-circuit torque does. The rotor sees
    2H d(w)/dt = Tm - Te - D (w - 1),    Te = psi''d iq - psi''q id.
Its governor and exciter are the classical model's, with Efd now GENROU's.
Left out: x''q != x''d, a zero sequence (its star floats, as the classical
model's does), and saturation's effect on anything but the field.
"""

import math

import numpy as np

A120 = 2.0 * math.pi / 3.0
SQ2 = math.sqrt(2.0)


class SynchronousMachine:
    """
    ``ckt``: the circuit; ``nodes``: its bus's three phase nodes; ``w0``: rad/s;
    ``v_ph``, ``i_ph``: its terminal voltage and the current it delivers, phase
    a peak phasors at the start; ``s_rated``: VA; ``vn``: line voltage (V).
    ``gov``: {R, T1, T2, VMAX, VMIN} or None; ``exc``: {K, TE, EMIN, EMAX} or None.
    """

    def __init__(self, ckt, label, nodes, w0, v_ph, i_ph, s_rated, vn, *, xd1_pu=0.3, ra_pu=0.0, h=3.0, d=0.0,
                 gov=None, exc=None, xd_pu=1.8, td0_s=8.0, r_bias=1e6):
        self.label, self.w0, self.s = label, w0, s_rated
        z_base = vn * vn / s_rated
        self.x = xd1_pu * z_base
        self.r = max(ra_pu * z_base, 0.01 * self.x)
        self.l = self.x / w0
        e_ph = v_ph + complex(self.r, self.x) * i_ph
        self.e_mag = abs(e_ph)                 # peak, per phase
        self.e0 = self.e_mag
        self.delta = float(np.angle(e_ph))     # phase a's angle at t = 0
        self.w = 1.0
        self.h, self.d = h, d
        p0 = float((1.5 * e_ph * np.conj(i_ph)).real)    # the power at E': its load-flow power plus its losses
        self.pm = p0 / s_rated
        self.p_set = self.pm
        self.gov = dict(gov) if gov else None
        if self.gov:
            self.valve = self.pm
        self.exc = dict(exc) if exc else None
        self.v0 = abs(v_ph)
        if self.exc:
            # Its field: E' and Efd per unit of its rated phase peak; Id per unit of its rated current's peak.
            self.e_base = SQ2 * vn / math.sqrt(3.0)
            self.i_base = SQ2 * s_rated / (math.sqrt(3.0) * vn)
            self.xdd, self.td0 = max(xd_pu - xd1_pu, 0.0), max(td0_s, 1e-3)
            self.efd0 = self.e_mag / self.e_base + self.xdd * self._i_d(i_ph, self.delta)
            self.efd, self.lead = self.efd0, 0.0
        star = ckt.node(f'{label} star point')
        ckt.add_r(star, 0, r_bias)
        self.k_e = [ckt.add_rl(star, nodes[k], self.r, self.l,
                               ac=(self.e_mag, w0, self.delta - k * A120), controlled=True) for k in range(3)]
        self.nodes = nodes
        self.t_last = 0.0
        self.e_now = None
        self.trace = []                       # (t, P MW, Q Mvar, f Hz, delta rad, E' pu, Pm MW)

    def _i_d(self, i_ph, angle):
        """A current phasor's d-axis component (per unit), the d axis a quarter period behind E' at ``angle``."""
        return -abs(i_ph) * math.sin(np.angle(i_ph) - angle) / self.i_base

    def _governor(self, dt):
        if self.gov:
            g = self.gov
            order = self.p_set + (1.0 - self.w) / g['R']
            self.valve += (min(max(order, g['VMIN']), g['VMAX']) - self.valve) * (1.0 - math.exp(-dt / g['T1']))
            self.valve = min(max(self.valve, g['VMIN']), g['VMAX'])
            self.pm += (self.valve - self.pm) * (1.0 - math.exp(-dt / max(g['T2'], 1e-6)))

    def _regulator(self, dt, v_mag):
        """Its field voltage from its terminal voltage's magnitude."""
        x = self.exc
        u = (self.v0 - v_mag) / self.e_base
        # Its lead-lag, (1 + s TA) / (1 + s TB): TA/TB at once, the rest through TB.
        ratio = x.get('TATB', 1.0)
        if ratio < 1.0 and x.get('TB', 0.0) > 0:
            self.lead += (u - self.lead) * (1.0 - math.exp(-dt / x['TB']))
            u = ratio * u + (1.0 - ratio) * self.lead
        target = self.efd0 + x['K'] * u
        self.efd += (target - self.efd) * (1.0 - math.exp(-dt / max(x['TE'], 1e-6)))
        self.efd = min(max(self.efd, x['EMIN'] * self.efd0), x['EMAX'] * self.efd0)

    def control(self, t, state):
        dt = t - self.t_last
        if dt <= 0:
            return False
        self.t_last = t
        i = np.array([state.i_rl[k] for k in self.k_e])
        v = state.v[self.nodes]
        th = self.w0 * t + self.delta
        e = self.e_mag * np.cos(th - np.arange(3) * A120)
        pe = float(np.dot(e, i))
        # Its governor, then its rotor.
        self._governor(dt)
        self.w += dt / (2.0 * self.h) * (self.pm - pe / self.s - self.d * (self.w - 1.0))
        self.delta += self.w0 * (self.w - 1.0) * dt
        # Its exciter, on its terminal voltage's magnitude (the space vector's).
        v_mag = math.sqrt(2.0 / 3.0 * float(np.dot(v, v)))
        if self.exc:
            self._regulator(dt, v_mag)
            # Its field: E' toward Efd less the armature's reaction, over T'd0.
            i_vec = (2.0 / 3.0) * (i[0] + i[1] * np.exp(1j * A120) + i[2] * np.exp(-1j * A120))
            i_d = self._i_d(i_vec, th)
            e_pu = self.e_mag / self.e_base
            e_pu += (self.efd - self.xdd * i_d - e_pu) * (1.0 - math.exp(-dt / self.td0))
            self.e_mag = e_pu * self.e_base
        # Its internal voltage over the next step, at the step's middle.
        th_next = self.w0 * (t + 0.5 * dt) + self.delta + 0.5 * self.w0 * (self.w - 1.0) * dt
        for k in range(3):
            state.set_emf(self.k_e[k], float(self.e_mag * math.cos(th_next - k * A120)))
        q = ((v[1] - v[2]) * i[0] + (v[2] - v[0]) * i[1] + (v[0] - v[1]) * i[2]) / math.sqrt(3.0)
        p_term = float(np.dot(v, i))
        self.trace.append((t, p_term / 1e6, q / 1e6, self.w * self.w0 / (2 * math.pi), self.delta,
                           self.e_mag / self.e0, self.pm * self.s / 1e6))
        return False


class ParkMachine(SynchronousMachine):
    """
    The detailed model; see the module's docstring. ``ckt`` to ``vn`` as the
    classical model's; reactances per unit of its rating, time constants in
    seconds. ``s10``, ``s12``: its saturation at 1.0 and 1.2 pu flux, GENROU's
    S(1.0) and S(1.2); none when ``s10`` is 0.
    """

    speed_voltage = True      # its voltage in proportion to its speed; an RMS GENROU such as ANDES's leaves that out

    def __init__(self, ckt, label, nodes, w0, v_ph, i_ph, s_rated, vn, *, xd_pu, xq_pu, xd1_pu, xq1_pu, xd2_pu, xl_pu,
                 td10_s, td20_s, tq10_s, tq20_s, ra_pu=0.0, s10=0.0, s12=0.0, h=3.0, d=0.0, gov=None, exc=None,
                 r_bias=1e6):
        self.label, self.w0, self.s, self.nodes = label, w0, s_rated, nodes
        self.h, self.d = h, d
        self.e_base = SQ2 * vn / math.sqrt(3.0)
        self.i_base = SQ2 * s_rated / (math.sqrt(3.0) * vn)
        z_base = vn * vn / s_rated
        self.xd, self.xq, self.xd1, self.xq1, self.x2, self.xl = xd_pu, xq_pu, xd1_pu, xq1_pu, xd2_pu, xl_pu
        self.td1, self.td2, self.tq1, self.tq2 = td10_s, td20_s, tq10_s, tq20_s
        self.ra = max(ra_pu, 0.01 * xd2_pu)
        self.gd1 = (xd2_pu - xl_pu) / (xd1_pu - xl_pu)
        self.gq1 = (xd2_pu - xl_pu) / (xq1_pu - xl_pu)
        self.gd2 = (xd1_pu - xd2_pu) / (xd1_pu - xl_pu) ** 2
        self.gq2 = (xq1_pu - xd2_pu) / (xq1_pu - xl_pu) ** 2
        self.gqd = (xq_pu - xl_pu) / (xd_pu - xl_pu)
        # Its saturation, GENROU's quadratic: Se psi = B (psi - A)^2 above A, through its two points.
        self.sat_a, self.sat_b = 1.0, 0.0
        if s10 > 0.0:
            s12 = max(s12, s10)
            ratio = math.sqrt(s10 / (1.2 * s12))
            self.sat_a = (1.0 - 1.2 * ratio) / (1.0 - ratio)
            self.sat_b = 1.2 * s12 / (1.2 - self.sat_a) ** 2
        # Its start, the load flow's steady state: delta the q axis, from psi''(1 + Se xq/xd) = -(xq - x'') iq on q.
        v, i = v_ph / self.e_base, i_ph / self.i_base
        e2 = v + complex(self.ra, xd2_pu) * i
        se = self._se(abs(e2))
        a, b = abs(e2) * (1.0 + se * self.gqd), (xq_pu - xd2_pu) * abs(i)
        beta, gamma = float(np.angle(e2)), float(np.angle(i))
        self.delta = math.atan2(a * math.sin(beta) + b * math.cos(gamma), a * math.cos(beta) - b * math.sin(gamma))
        rot = complex(math.cos(self.delta), -math.sin(self.delta))
        psi2, cur = e2 * rot, 1j * i * rot
        id0, iq0 = cur.real, cur.imag
        self.pd1 = psi2.real + (xd1_pu - xd2_pu) * id0     # psi'd
        self.pd2 = self.pd1 - (xd1_pu - xl_pu) * id0       # psi1d
        self.pq1 = psi2.imag + (xq1_pu - xd2_pu) * iq0     # psi'q
        self.pq2 = self.pq1 - (xq1_pu - xl_pu) * iq0       # psi1q
        self.efd0 = self.pd1 + (xd_pu - xd1_pu) * id0 + se * psi2.real
        self.efd, self.lead = self.efd0, 0.0
        self.w = 1.0
        self.pm = psi2.real * iq0 - psi2.imag * id0
        self.p_set = self.pm
        self.gov = dict(gov) if gov else None
        if self.gov:
            self.valve = self.pm
        self.exc = dict(exc) if exc else None
        self.v0 = abs(v_ph)
        self.e0 = abs(e2)
        self.r = self.ra * z_base
        self.l = xd2_pu * z_base / w0
        star = ckt.node(f'{label} star point')
        ckt.add_r(star, 0, r_bias)
        e_mag = abs(e2) * self.e_base
        self.k_e = [ckt.add_rl(star, nodes[k], self.r, self.l, ac=(e_mag, w0, beta - k * A120), controlled=True)
                    for k in range(3)]
        self.t_last = 0.0
        self.trace = []                       # (t, P MW, Q Mvar, f Hz, delta rad, psi'' pu, Pm MW)

    def _se(self, psi):
        return self.sat_b * (psi - self.sat_a) ** 2 / psi if psi > self.sat_a and self.sat_b > 0.0 else 0.0

    def _flux(self):
        """psi''d and psi''q."""
        return (self.gd1 * self.pd1 + (1.0 - self.gd1) * self.pd2,
                self.gq1 * self.pq1 + (1.0 - self.gq1) * self.pq2)

    def control(self, t, state):
        dt = t - self.t_last
        if dt <= 0:
            return False
        self.t_last = t
        i = np.array([state.i_rl[k] for k in self.k_e])
        v = state.v[self.nodes]
        # Its current on its rotor's axes: the d axis a quarter period behind delta.
        i_s = (2.0 / 3.0) * (i[0] + i[1] * np.exp(1j * A120) + i[2] * np.exp(-1j * A120)) / self.i_base
        th_d = self.w0 * t + self.delta - 0.5 * math.pi
        cur = i_s * complex(math.cos(th_d), -math.sin(th_d))
        i_d, i_q = cur.real, cur.imag
        psi_d, psi_q = self._flux()
        # Its governor, then its rotor.
        self._governor(dt)
        self.w += dt / (2.0 * self.h) * (self.pm - (psi_d * i_q - psi_q * i_d) - self.d * (self.w - 1.0))
        self.delta += self.w0 * (self.w - 1.0) * dt
        # Its exciter, on its terminal voltage's magnitude (the space vector's).
        v_mag = math.sqrt(2.0 / 3.0 * float(np.dot(v, v)))
        if self.exc:
            self._regulator(dt, v_mag)
        # Its rotor's circuits.
        se = self._se(math.hypot(psi_d, psi_q))
        d_pd1 = (self.efd - self.pd1 - (self.xd - self.xd1) * (self.gd1 * i_d + self.gd2 * (self.pd1 - self.pd2))
                 - se * psi_d) / self.td1
        d_pd2 = (self.pd1 - self.pd2 - (self.xd1 - self.xl) * i_d) / self.td2
        d_pq1 = (-self.pq1 - (self.xq - self.xq1) * (self.gq1 * i_q + self.gq2 * (self.pq1 - self.pq2))
                 - se * self.gqd * psi_q) / self.tq1
        d_pq2 = (self.pq1 - self.pq2 - (self.xq1 - self.xl) * i_q) / self.tq2
        self.pd1 += dt * d_pd1
        self.pd2 += dt * d_pd2
        self.pq1 += dt * d_pq1
        self.pq2 += dt * d_pq2
        psi_d, psi_q = self._flux()
        rate = complex(self.gd1 * d_pd1 + (1.0 - self.gd1) * d_pd2, self.gq1 * d_pq1 + (1.0 - self.gq1) * d_pq2)
        # Its voltage behind x'' over the next step, at the step's middle.
        mid = complex(psi_d, psi_q) + 0.5 * dt * rate
        th_next = self.w0 * (t + 0.5 * dt) + self.delta - 0.5 * math.pi + 0.5 * self.w0 * (self.w - 1.0) * dt
        e_s = (1j * (self.w if self.speed_voltage else 1.0) * mid + rate / self.w0) * self.e_base * complex(math.cos(th_next), math.sin(th_next))
        for k in range(3):
            state.set_emf(self.k_e[k], float((e_s * complex(math.cos(k * A120), -math.sin(k * A120))).real))
        q = ((v[1] - v[2]) * i[0] + (v[2] - v[0]) * i[1] + (v[0] - v[1]) * i[2]) / math.sqrt(3.0)
        self.trace.append((t, float(np.dot(v, i)) / 1e6, q / 1e6, self.w * self.w0 / (2 * math.pi), self.delta,
                           math.hypot(psi_d, psi_q) / self.e0, self.pm * self.s / 1e6))
        return False


def park_data(row, gov_defaults, exc_defaults, genrou):
    """
    The detailed model's data from its diagram row's dynamics fields, ANDES's
    GENROU's, ``genrou`` its defaults; and what had to be changed to make them
    a machine, x'' < x' <= x and xl < x''.
    """
    def f(key, default):
        v = (row or {}).get(key)
        try:
            v = float(v) if v not in (None, '', 'null', 'None') else 0.0
        except (TypeError, ValueError):
            v = 0.0
        return v if v > 0.0 else default

    data = machine_data(row, gov_defaults, exc_defaults, h_default=genrou['M'] / 2.0)
    for key in ('xd1_pu', 'xd_pu', 'td0_s'):
        data.pop(key)
    notes = []
    xd2 = f('dyn_xd2', f('xdss_pu', genrou['xd2']))
    xd, xq = f('dyn_xd', genrou['xd']), f('dyn_xq', genrou['xq'])
    xd1, xq1 = f('dyn_xd1', genrou['xd1']), f('dyn_xq1', genrou['xq1'])
    xl = f('dyn_xl', genrou['xl'])
    if xl >= xd2:
        notes.append(f"xl {xl:g} is not below x'' {xd2:g}: used {0.8 * xd2:g}")
        xl = 0.8 * xd2
    for axis, x, x1 in (('d', xd, xd1), ('q', xq, xq1)):
        if not xd2 < x1 <= x:
            notes.append(f"x{axis} {x:g}, x'{axis} {x1:g} and x'' {xd2:g} are not in order: used x'{axis} {min(max(x1, 1.01 * xd2), max(x, 1.01 * xd2)):g}")
    xd = max(xd, 1.01 * xd2)
    xq = max(xq, 1.01 * xd2)
    xd1 = min(max(xd1, 1.01 * xd2), xd)
    xq1 = min(max(xq1, 1.01 * xd2), xq)
    data.update(xd_pu=xd, xq_pu=xq, xd1_pu=xd1, xq1_pu=xq1, xd2_pu=xd2, xl_pu=xl,
                td10_s=f('dyn_Td10', genrou['Td10']), td20_s=f('dyn_Td20', genrou['Td20']),
                tq10_s=f('dyn_Tq10', genrou['Tq10']), tq20_s=f('dyn_Tq20', genrou['Tq20']),
                s10=f('dyn_S10', 0.0), s12=f('dyn_S12', 0.0))
    return data, notes


def machine_data(row, gov_defaults, exc_defaults, h_default=3.0):
    """
    Its rotor, governor and exciter from its diagram row's dynamics fields (the
    ANDES study's): H (or M = 2H), D, x'd, ra; its governor model's R, T1, T2,
    VMAX, VMIN (GAST's when it names none); its exciter's gain and lag, and
    SEXS's lead-lag (none when it names none or NONE); its field's xd and
    T'd0.
    """
    def f(key, default):
        v = row.get(key) if row else None
        try:
            return float(v) if v not in (None, '', 'null', 'None') else default
        except (TypeError, ValueError):
            return default

    h = f('dyn_H', f('dyn_M', 2.0 * h_default) / 2.0)
    gov_model = str((row or {}).get('dyn_governor_model') or 'GAST').strip().upper()
    gov = None
    if gov_model not in ('NONE', 'OFF'):
        base = gov_defaults.get(gov_model, gov_defaults['GAST'])
        gov = {k: f(f'dyn_gov_{k}', base.get(k, d)) for k, d in
               (('R', 0.05), ('T1', 0.4), ('T2', 0.1), ('VMAX', 1.2), ('VMIN', 0.0))}
        gov['R'] = max(gov['R'], 1e-3)
        gov['T1'] = max(gov['T1'], 1e-3)
    exc_model = str((row or {}).get('dyn_exciter_model') or 'NONE').strip().upper()
    exc = None
    if exc_model not in ('NONE', 'OFF'):
        base = exc_defaults.get(exc_model, {})
        k = f('dyn_exc_K', f('dyn_exc_KA', base.get('K', base.get('KA', 50.0))))
        exc = {'K': k, 'TE': f('dyn_exc_TE', f('dyn_exc_TA', base.get('TE', base.get('TA', 0.05)))),
               'EMIN': 0.0, 'EMAX': 2.0,
               # SEXS's lead-lag: its transient gain K TA/TB, the rest over TB.
               'TATB': base.get('TATB', 1.0) if exc_model == 'SEXS' else 1.0,
               'TB': base.get('TB', 0.0) if exc_model == 'SEXS' else 0.0}
    # Its field, as GENROU has it (ANDES's defaults): x'd to xd over T'd0.
    return dict(h=h, d=f('dyn_D', 0.0), xd1_pu=f('dyn_xd1', 0.3), ra_pu=f('dyn_ra', 0.0), gov=gov, exc=exc,
                xd_pu=f('dyn_xd', 1.8), td0_s=f('dyn_Td10', 8.0))
