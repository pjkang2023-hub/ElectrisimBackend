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
        if self.gov:
            g = self.gov
            order = self.p_set + (1.0 - self.w) / g['R']
            self.valve += (min(max(order, g['VMIN']), g['VMAX']) - self.valve) * (1.0 - math.exp(-dt / g['T1']))
            self.valve = min(max(self.valve, g['VMIN']), g['VMAX'])
            self.pm += (self.valve - self.pm) * (1.0 - math.exp(-dt / max(g['T2'], 1e-6)))
        self.w += dt / (2.0 * self.h) * (self.pm - pe / self.s - self.d * (self.w - 1.0))
        self.delta += self.w0 * (self.w - 1.0) * dt
        # Its exciter, on its terminal voltage's magnitude (the space vector's).
        v_mag = math.sqrt(2.0 / 3.0 * float(np.dot(v, v)))
        if self.exc:
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
