"""
Exciters and governors for the EMT study's detailed machine, as ANDES has them, so that a machine is the same
in both studies: the exciters EXDC2, SEXS, IEEEX1, ESDC2A, EXST1, ESST1A and AC8B, and the governors TGOV1,
GAST, IEEEG1, IEESGO and HYGOV - every one Electrisim offers. Their parameters come from where ANDES's study takes them:
ANDES's own defaults, then Electrisim's (andes_electrisim._EXCITER_DEFAULTS, _GOVERNOR_DEFAULTS), then the
generator's dyn_exc_ and dyn_gov_ fields.

Each block is ANDES's: a lag K / (1 + s T); a lag whose output is held within its limits (anti-windup);
a lead-lag (1 + s T1) / (1 + s T2), its output its input where T2 is 0; a washout K s / (1 + s T). Each steps
by the exact solution over the step for its input held. Exciters work in per unit of the machine's field
(Efd at its air-gap line), governors in per unit of its rating.
"""
import math
from functools import lru_cache

EXCITERS = ('EXDC2', 'SEXS', 'IEEEX1', 'ESDC2A', 'EXST1', 'ESST1A', 'AC8B')
GOVERNORS = ('TGOV1', 'GAST', 'IEEEG1', 'IEESGO', 'HYGOV')


def _quad_sat(e1, se1, e2, se2):
    """ANDES's quadratic saturation through (E1, SE1) and (E2, SE2): its A and B, both 0 where SE2 is 0."""
    if se2 == 0.0:
        e2, se2 = e2 + 2.0, se2 + 2.0
    a = math.sqrt(se1 * e1 / (se2 * e2)) if se2 * e2 != 0.0 and se1 * e1 >= 0.0 else 0.0
    sat_a = e2 - (e1 - e2) / (a - 1.0) if a != 1.0 else e2
    sat_b = se2 * e2 * (a - 1.0) ** 2 / (e1 - e2) ** 2 if a != 0.0 and e1 != e2 else 0.0
    return sat_a, sat_b


def _decay(dt, t):
    return 1.0 - math.exp(-dt / t) if t > 1e-9 else 1.0


class Lag:
    def __init__(self, y0, t, k=1.0, lower=None, upper=None):
        self.y, self.t, self.k, self.lower, self.upper = y0, t, k, lower, upper

    def step(self, dt, u):
        self.y += (self.k * u - self.y) * _decay(dt, self.t)
        if self.upper is not None:
            self.y = min(max(self.y, self.lower), self.upper)
        return self.y


class LeadLag:
    """(1 + s T1) / (1 + s T2): x follows u through T2, y = x + T1 / T2 (u - x)."""

    def __init__(self, y0, t1, t2):
        self.t1, self.t2 = t1, t2
        self.x = y0
        self.y = y0

    def step(self, dt, u):
        if self.t2 <= 1e-9:
            self.y = u
            return u
        self.x += (u - self.x) * _decay(dt, self.t2)
        self.y = self.x + self.t1 / self.t2 * (u - self.x)
        return self.y


class Washout:
    """K s / (1 + s T): x follows u through T, y = K (u - x) / T."""

    def __init__(self, u0, t, k):
        self.x, self.t, self.k = u0, max(t, 1e-6), k

    def step(self, dt, u):
        self.x += (u - self.x) * _decay(dt, self.t)
        return self.k * (u - self.x) / self.t


@lru_cache(maxsize=None)
def andes_defaults(model):
    """ANDES's own defaults for a model's numeric parameters."""
    import andes
    ss = andes.System()
    m = getattr(ss, model)
    out = {}
    for name, p in m.num_params.items():
        try:
            out[name] = float(p.default)       # some are given as text, SEXS's TE as '1'
        except (TypeError, ValueError):
            pass
    return out


def parameters(kind, row):
    """(model, its parameters) for kind 'exc', 'gov' or 'pss' as ANDES's study sets them up; None for none."""
    from andes_electrisim import _EXCITER_DEFAULTS, _GOVERNOR_DEFAULTS, _PSS_DEFAULTS, _model_kwargs, governor_kwargs
    if kind == 'pss':
        model = str((row or {}).get('dyn_pss_model') or 'NONE').strip().upper()
        if model not in _PSS_DEFAULTS:
            return None
        params = dict(andes_defaults(model))
        params.update(_model_kwargs(row or {}, 'dyn_pss_', _PSS_DEFAULTS[model]))
        params['MODE'] = int(round(params['MODE'])) if 1 <= round(params['MODE']) <= 6 else 1
        return model, params
    field, prefix, table, default = (('dyn_exciter_model', 'dyn_exc_', _EXCITER_DEFAULTS, 'EXDC2') if kind == 'exc'
                                     else ('dyn_governor_model', 'dyn_gov_', _GOVERNOR_DEFAULTS, 'TGOV1'))
    model = str((row or {}).get(field) or default).strip().upper()
    if model in ('', 'NONE', 'OFF'):
        return None
    if model not in table:
        model = default
    params = dict(andes_defaults(model))
    params.update(_model_kwargs(row or {}, prefix, table[model]) if kind == 'exc' else governor_kwargs(row or {}, model))
    known = andes_defaults(model)
    return model, {k: v for k, v in params.items() if k in known}


class Exdc2:
    """
    IEEE DC2A as ANDES's EXDC2: a sensing lag TR, the lead-lag TC/TB, the regulator KA/TA within VRMIN/VRMAX,
    the exciter TE with KE and its quadratic saturation through (E1, SE1) and (E2, SE2), the rate feedback
    KF1 s / (1 + s TF1). Its field voltage is the exciter's output times the speed.
    """

    def __init__(self, p, efd0, v0):
        self.p = p
        self.sat_a, self.sat_b = _quad_sat(p['E1'], p['SE1'], p['E2'], p['SE2'])
        self.vp = efd0
        vr0 = (p['KE'] + self._se(efd0)) * efd0
        vb0 = vr0 / p['KA']
        self.vref = v0 + vb0
        self.ls = Lag(v0, p['TR'])
        self.ll = LeadLag(vb0, p['TC'], p['TB'])
        self.la = Lag(vr0, p['TA'], p['KA'], p['VRMIN'], p['VRMAX'])
        self.w = Washout(efd0, p['TF1'], p['KF1'])
        self.w_y = 0.0

    def _se(self, vp):
        return self.sat_b * (vp - self.sat_a) ** 2 / vp if vp > self.sat_a and vp > 0.0 else 0.0

    def step(self, dt, v, omega, xadifd=None):
        p = self.p
        vi = self.vref - self.ls.step(dt, v) - self.w_y
        vr = self.la.step(dt, self.ll.step(dt, vi))
        self.vp += (vr - p['KE'] * self.vp - self._se(self.vp) * self.vp) * dt / max(p['TE'], 1e-6)
        self.w_y = self.w.step(dt, self.vp)
        return omega * self.vp


class Sexs:
    """SEXS: the lead-lag TA/TB (TA = TATB TB), then K / (1 + s TE) within EMIN/EMAX."""

    def __init__(self, p, efd0, v0):
        self.vref = v0 + efd0 / p['K']
        self.ll = LeadLag(efd0 / p['K'], p['TATB'] * p['TB'], p['TB'])
        self.law = Lag(efd0, p['TE'], p['K'], p['EMIN'], p['EMAX'])

    def step(self, dt, v, omega, xadifd=None):
        return self.law.step(dt, self.ll.step(dt, self.vref - v))


class Tgov1:
    """TGOV1: (pref - (w - 1)) / R through a lag T1 within VMIN/VMAX, then the lead-lag T2/T3, less Dt (w - 1)."""

    def __init__(self, p, tm0):
        self.p = p
        self.pref = tm0 * p['R']
        self.lag = Lag(tm0, p['T1'], 1.0, p['VMIN'], p['VMAX'])
        self.ll = LeadLag(tm0, p['T2'], p['T3'])

    def step(self, dt, omega):
        p = self.p
        wd = omega - 1.0
        y = self.ll.step(dt, self.lag.step(dt, (self.pref - wd) / p['R']))
        return y - p['Dt'] * wd


class Gast:
    """GAST: the speed order and the load limit AT + KT (AT - the exhaust lag's output) through a low-value gate,
    the fuel valve's lag T1 within VMIN/VMAX, the turbine's lag T2 and the exhaust's T3, less Dt (w - 1)."""

    def __init__(self, p, tm0):
        self.p = p
        self.pref = tm0 * p['R']
        self.lag = Lag(tm0, p['T1'], 1.0, p['VMIN'], p['VMAX'])
        self.lg2 = Lag(tm0, p['T2'])
        self.lg3 = Lag(tm0, p['T3'])

    def step(self, dt, omega):
        p = self.p
        wd = omega - 1.0
        limit = p['AT'] + p['KT'] * (p['AT'] - self.lg3.y)
        y = self.lg2.step(dt, self.lag.step(dt, min((self.pref - wd) / p['R'], limit)))
        self.lg3.step(dt, y)
        return y - p['Dt'] * wd


class Ieeex1(Exdc2):
    """IEEEX1: EXDC2 with its regulator's limits VRMAX and VRMIN times the terminal voltage, and no speed on its output."""

    def step(self, dt, v, omega, xadifd=None):
        self.la.lower, self.la.upper = self.p['VRMIN'] * v, self.p['VRMAX'] * v
        return super().step(dt, v, 1.0, xadifd)


class Esdc2a:
    """
    ESDC2A as ANDES has it: the error against the terminal voltage itself (its sensing lag computed but not used,
    as in ANDES), the lead-lag TC/TB, the regulator KA/TA within VRMIN and VRMAX (999 where 0) times the voltage,
    the exciter's integrator TE against KE and its saturation (not divided by the output), the rate feedback
    KF s / (1 + s TF1) on the output.
    """

    def __init__(self, p, efd0, v0):
        self.p = p
        self.sat_a, self.sat_b = _quad_sat(p['E1'], p['SE1'], p['E2'], p['SE2'])
        self.vrmax = p['VRMAX'] if p['VRMAX'] != 0.0 else 999.0
        se0 = self._se(efd0)
        vfe0 = efd0 * p['KE'] + se0
        self.vref = v0 + vfe0 / p['KA']
        self.ll = LeadLag(vfe0 / p['KA'], p['TC'], p['TB'])
        self.la = Lag(vfe0, p['TA'], p['KA'], p['VRMIN'] * v0, self.vrmax * v0)
        self.integral = efd0
        self.w = Washout(efd0, p['TF1'], p['KF'])
        self.w_y = 0.0

    def _se(self, x):
        return self.sat_b * (x - self.sat_a) ** 2 if x > self.sat_a else 0.0

    def step(self, dt, v, omega, xadifd=None):
        p = self.p
        self.la.lower, self.la.upper = p['VRMIN'] * v, self.vrmax * v
        vr = self.la.step(dt, self.ll.step(dt, self.vref - v - self.w_y))
        self.integral += (vr - (self.integral * p['KE'] + self._se(self.integral))) * dt / max(p['TE'], 1e-6)
        self.w_y = self.w.step(dt, self.integral)
        return self.integral


class Exst1:
    """
    EXST1: the sensing lag TR, the input within VIMIN and VIMAX, the lead-lag TC/TB, the regulator KA/TA, the rate
    feedback KF s / (1 + s TF) on its output; its ceiling VRMAX - KC XadIfd and floor VRMIN - KC XadIfd applied as
    ANDES applies them, by where the feedback's output lies.
    """

    def __init__(self, p, efd0, v0):
        self.p = p
        self.vref = v0 + efd0 / p['KA']
        self.ls = Lag(v0, p['TR'])
        self.ll = LeadLag(efd0 / p['KA'], p['TC'], p['TB'])
        self.lr = Lag(efd0, p['TA'], p['KA'])
        self.w = Washout(efd0, p['TF'], p['KF'])
        self.w_y = 0.0

    def step(self, dt, v, omega, xadifd=0.0):
        p = self.p
        vi = min(max(self.vref - self.ls.step(dt, v) - self.w_y, p['VIMIN']), p['VIMAX'])
        lr = self.lr.step(dt, self.ll.step(dt, vi))
        self.w_y = self.w.step(dt, lr)
        top, bottom = p['VRMAX'] - p['KC'] * xadifd, p['VRMIN'] - p['KC'] * xadifd
        return top if self.w_y > top else bottom if self.w_y < bottom else lr


class Esst1a:
    """
    ESST1A as ANDES runs it with no under- or over-excitation limiter or stabiliser: the sensing lag TR, the input
    within VIMIN and VIMAX, two lead-lags, the regulator KA/TA within VAMIN and VAMAX, less the field current limiter
    KLR (XadIfd - ILR) above zero; its output within the terminal voltage times VRMIN and VRMAX less KC XadIfd; the rate
    feedback KF s / (1 + s TF) on the regulator's output.
    """

    def __init__(self, p, efd0, v0, xadifd0):
        self.p = p
        lr0 = max(p['KLR'] * (xadifd0 - p['ILR']), 0.0)
        va0 = efd0 + lr0
        self.vref = v0 + va0 / p['KA']
        self.ls = Lag(v0, p['TR'])
        self.ll = LeadLag(va0 / p['KA'], p['TC'], p['TB'])
        self.ll1 = LeadLag(va0 / p['KA'], p['TC1'], p['TB1'])
        self.la = Lag(va0, p['TA'], p['KA'], p['VAMIN'], p['VAMAX'])
        self.w = Washout(efd0, p['TF'], p['KF'])
        self.w_y = 0.0

    def step(self, dt, v, omega, xadifd=0.0):
        p = self.p
        vi = min(max(self.vref - self.ls.step(dt, v) - self.w_y, p['VIMIN']), p['VIMAX'])
        va = self.la.step(dt, self.ll1.step(dt, self.ll.step(dt, vi)))
        vas = va - max(p['KLR'] * (xadifd - p['ILR']), 0.0)
        self.w_y = self.w.step(dt, vas)
        return min(max(vas, v * p['VRMIN']), v * p['VRMAX'] - p['KC'] * xadifd)


def _fex(i_n):
    """The rectifier's regulation, IEEE AC type."""
    if i_n <= 0.0:
        return 1.0
    if i_n <= 0.433:
        return 1.0 - 0.577 * i_n
    if i_n <= 0.75:
        return math.sqrt(0.75 - i_n * i_n)
    if i_n <= 1.0:
        return 1.732 * (1.0 - i_n)
    return 0.0


class Ac8b:
    """
    AC8B: the sensing lag TR; a PID (kP, kI, kD through a washout Td) within VPMIN and VPMAX, its integrator tracking
    the limit (gain 2); the regulator KA/TA within VRMIN and VRMAX; the exciter's integrator TE against
    KE, its saturation and KD XadIfd; its output times the rectifier's regulation at KC XadIfd.
    """

    KS = 2.0

    def __init__(self, p, efd0, v0, xadifd0):
        self.p = p
        self.sat_a, self.sat_b = _quad_sat(p['E1'], p['SE1'], p['E2'], p['SE2'])
        # Its exciter's output, from its field voltage: VE FEX(KC XadIfd / VE) = Efd.
        ve = max(efd0, 1e-3)
        for _ in range(60):
            g = ve * _fex(p['KC'] * xadifd0 / ve) - efd0
            h = 1e-6 * max(ve, 1e-3)
            dg = ((ve + h) * _fex(p['KC'] * xadifd0 / (ve + h)) - (ve - h) * _fex(p['KC'] * xadifd0 / (ve - h))) / (2.0 * h)
            if abs(g) < 1e-12 or dg == 0.0:
                break
            ve = max(ve - g / dg, 1e-6)
        self.ve = ve
        vfe0 = self._vfe(ve, xadifd0)
        self.vref = v0
        self.ls = Lag(v0, p['TR'])
        self.xi = vfe0 / p['KA']
        self.wo = Washout(0.0, p['Td'], p['kD'])
        self.la = Lag(vfe0, p['TA'], p['KA'], p['VRMIN'], p['VRMAX'])

    def _vfe(self, ve, xadifd):
        se = self.sat_b * (ve - self.sat_a) ** 2 if ve > self.sat_a else 0.0
        return ve * self.p['KE'] + se + xadifd * self.p['KD']

    def step(self, dt, v, omega, xadifd=0.0):
        p = self.p
        vi = self.vref - self.ls.step(dt, v)
        ys = p['kP'] * vi + self.xi + self.wo.step(dt, vi)
        y = min(max(ys, p['VPMIN']), p['VPMAX'])
        self.xi += p['kI'] * (vi - self.KS * (ys - y)) * dt
        va = self.la.step(dt, y)
        self.ve += (va - self._vfe(self.ve, xadifd)) * dt / max(p['TE'], 1e-6)
        return self.ve * _fex(p['KC'] * xadifd / max(self.ve, 1e-6))


class Ieeeg1:
    """
    IEEEG1: the speed error (1 - w) through K (1 + s T2) / (1 + s T1); the valve's rate (that, plus the set power, less
    its position) / T3 within UC and UO; its position, the integral, within PMIN and PMAX; four lags T4 to T7, the
    high-pressure shaft taking K1, K3, K5, K7 of them (the K's taken as fractions of their sum). A second shaft is
    not modelled.
    """

    def __init__(self, p, tm0):
        self.p = p
        total = sum(p[f'K{k}'] for k in range(1, 9)) or 1.0
        self.k_hp = [p[f'K{k}'] / total for k in (1, 3, 5, 7)]
        self.tm0 = tm0
        self.ll = LeadLag(0.0, p['T2'], p['T1'])
        self.position = tm0
        self.lags = [Lag(tm0, p[f'T{k}']) for k in (4, 5, 6, 7)]

    def step(self, dt, omega):
        p = self.p
        ll = p['K'] * self.ll.step(dt, 1.0 - omega)
        rate = min(max((ll + self.tm0 - self.position) / max(p['T3'], 1e-6), p['UC']), p['UO'])
        self.position = min(max(self.position + rate * dt, p['PMIN']), p['PMAX'])
        x, out = self.position, 0.0
        for k, lag in zip(self.k_hp, self.lags):
            x = lag.step(dt, x)
            out += k * x
        return out


class Ieesgo:
    """IEESGO: the speed through K1 / (1 + s T1) and the lead-lag T2/T3, off the set power within PMIN and PMAX; the
    turbine's lags T4, then K2 / (1 + s T5), then K3 / (1 + s T6), its power (1 - K2) F3 + (1 - K3) F4 + F5."""

    def __init__(self, p, tm0):
        self.p, self.tm0 = p, tm0
        self.f1 = Lag(0.0, p['T1'], p['K1'])
        self.f2 = LeadLag(0.0, p['T2'], p['T3'])
        self.f3 = Lag(tm0, p['T4'])
        self.f4 = Lag(p['K2'] * tm0, p['T5'], p['K2'])
        self.f5 = Lag(p['K3'] * p['K2'] * tm0, p['T6'], p['K3'])

    def step(self, dt, omega):
        p = self.p
        hl = min(max(self.tm0 - self.f2.step(dt, self.f1.step(dt, omega - 1.0)), p['PMIN']), p['PMAX'])
        f3 = self.f3.step(dt, hl)
        f4 = self.f4.step(dt, f3)
        f5 = self.f5.step(dt, f4)
        return (1.0 - p['K2']) * f3 + (1.0 - p['K3']) * f4 + f5


class Hygov:
    """
    HYGOV as ANDES has it: the speed error and the permanent droop R on the gate through the filter Tf; the gate's
    position its integral, at no more than VELM a second either way and within GMIN and GMAX, plus 1 / r of the
    filter's output; the servo Tg; the water column, Tw dq/dt = 1 - (q / gate)^2; its power At h (q - qNL), less
    Dt (w - 1) times the gate.
    """

    def __init__(self, p, tm0):
        self.p = p
        self.q = tm0 / p['At'] + p['qNL']
        self.pref = p['R'] * self.q
        self.lg = Lag(0.0, p['Tf'])
        self.gate = self.q
        self.dg = self.q
        self.lag = Lag(self.q, p['Tg'])

    def step(self, dt, omega):
        p = self.p
        wd = omega - 1.0
        gr = 1.0 / p['r']
        lg = self.lg.step(dt, -wd + self.pref - p['R'] * self.dg)
        rate = min(max(lg, -p['VELM'] - gr * lg), p['VELM'] - gr * lg)
        self.gate = min(max(self.gate + rate * dt, p['GMIN']), p['GMAX'])
        self.dg = self.gate + gr * lg
        g = max(self.lag.step(dt, self.dg), 1e-6)
        self.q += (1.0 - (self.q / g) ** 2) * dt / max(p['Tw'], 1e-6)
        h = (self.q / g) ** 2
        return p['At'] * h * (self.q - p['qNL']) - p['Dt'] * wd * g


class Linear2:
    """
    A second-order block, (1 + n1 s + n2 s^2) / (1 + d1 s + d2 s^2), stepped exactly for its input held (its
    matrices for each step length kept); first order where d2 is 0, its input where d1 is 0 too.
    """

    def __init__(self, u0, d1, d2, n1=0.0, n2=0.0):
        import numpy as np
        self.np = np
        self.u = u0
        if d2 > 1e-9:
            self.a = np.array([[0.0, 1.0], [-1.0 / d2, -d1 / d2]])
            self.b = np.array([0.0, 1.0 / d2])
            self.c = np.array([1.0 - n2 / d2, n1 - n2 * d1 / d2])
            self.d = n2 / d2
            self.x = np.array([u0, 0.0])
        elif d1 > 1e-9:
            self.a = np.array([[-1.0 / d1]])
            self.b = np.array([1.0 / d1])
            self.c = np.array([1.0 - n1 / d1])
            self.d = n1 / d1
            self.x = np.array([u0])
        else:
            self.a = None
        self.cache = {}

    def step(self, dt, u):
        if self.a is None:
            return u
        key = round(dt, 12)
        if key not in self.cache:
            from scipy.linalg import expm
            n = len(self.x)
            m = expm(self.np.block([[self.a, self.b[:, None]], [self.np.zeros((1, n + 1))]]) * dt)
            self.cache[key] = (m[:n, :n], m[:n, n])
        ad, bd = self.cache[key]
        self.x = ad @ self.x + bd * u
        return float(self.c @ self.x + self.d * u)


class Ieeest:
    """
    IEEEST as ANDES has it: its input by MODE (1 speed deviation, 2 bus frequency deviation, 3 electrical power,
    4 mechanical power's change, 5 bus voltage, 6 its rate of change), the filters 1 / (1 + A1 s + A2 s^2) and
    (1 + A5 s + A6 s^2) / (1 + A3 s + A4 s^2) (left out where A4 is 0, as ANDES does), two lead-lags T1/T2 and
    T3/T4, the gain KS, the washout T5 s / (1 + T6 s) (a lag T6 where T5 is 0), held within LSMIN/LSMAX, and out
    only while the bus voltage is within VCL..VCU (0 there meaning no bound). Its output adds to its exciter's
    input, as ANDES adds it to the exciter's vi.
    """

    def __init__(self, p, u0):
        self.p, self.mode = p, int(p['MODE'])
        self.f1 = Linear2(u0, p['A1'], p['A2'])
        self.f2 = Linear2(u0, p['A3'], p['A4'], p['A5'], p['A6']) if p['A4'] > 0 else None
        self.ll1 = LeadLag(u0, p['T1'], p['T2'])
        self.ll2 = LeadLag(u0, p['T3'], p['T4'])
        self.wo = Washout(p['KS'] * u0, p['T6'], p['T5']) if p['T5'] != 0 else Lag(p['KS'] * u0, p['T6'])
        self.vcu = p['VCU'] if p['VCU'] != 0 else 999.0
        self.vcl = p['VCL'] if p['VCL'] != 0 else -999.0
        self.vs = 0.0

    def step(self, dt, u, v):
        p = self.p
        x = self.f1.step(dt, u)
        if self.f2 is not None:
            x = self.f2.step(dt, x)
        x = self.wo.step(dt, p['KS'] * self.ll2.step(dt, self.ll1.step(dt, x)))
        x = min(max(x, p['LSMIN']), p['LSMAX'])
        self.vs = x if self.vcl < v < self.vcu else 0.0
        return self.vs


def _num(row, key, default=None):
    raw = (row or {}).get(key)
    try:
        return float(raw) if raw not in (None, '', 'null', 'None') else default
    except (TypeError, ValueError):
        return default


def limiter_parameters(row):
    """
    Its excitation limiters from its row, None where it has neither: the over-excitation limiter where
    dyn_oel_ifd_pu (its field's continuous rating, as XadIfd) is given, the under-excitation limiter where
    dyn_uel_q0_pu is.
    """
    p = {}
    ifd = _num(row, 'dyn_oel_ifd_pu')
    if ifd and ifd > 0:
        p['oel'] = {'ifd': ifd, 'k': max(_num(row, 'dyn_oel_k_s', 33.75), 1e-3),
                    'kp': _num(row, 'dyn_oel_kp', 0.5), 'ki': _num(row, 'dyn_oel_ki', 5.0)}
    q0 = _num(row, 'dyn_uel_q0_pu')
    if q0 is not None:
        p['uel'] = {'q0': q0, 'q1': _num(row, 'dyn_uel_q1_pu', q0), 'kp': _num(row, 'dyn_uel_kp', 0.5),
                    'ki': _num(row, 'dyn_uel_ki', 5.0)}
    return p or None


class Limiters:
    """
    Its exciter's limiters, each adding to its exciter's reference as ANDES's exciters take OEL and UEL:

    - over-excitation: its field current's heating, the integral of (Ifd / Ifd_rated)^2 - 1 (falling again when
      below 1, not under 0), reaching k (s; 33.75 by default, IEEE C50.13's 209 % for 10 s ... 113 % for 120 s)
      before it acts: then a PI, never positive, brings its field current to its rating, and lets go once its
      heating falls back below k;
    - under-excitation: a straight line in the P-Q plane, q0 at no power and q1 at rated, scaled by V^2 (as
      UEL2C with k 2): while its Q is below that, a PI, never negative, raises its excitation.
    """

    def __init__(self, p):
        self.oel, self.uel = p.get('oel'), p.get('uel')
        self.heat, self.i_oel, self.i_uel = 0.0, 0.0, 0.0
        self.oel_on, self.oel_picked_s, self.uel_acted_s = False, None, None
        self.t = 0.0

    def step(self, dt, ifd, p_pu, q_pu, v_pu):
        self.t += dt
        out = 0.0
        o = self.oel
        if o is not None:
            r = ifd / o['ifd']
            self.heat = max(self.heat + (r * r - 1.0) * dt, 0.0)
            if self.heat >= o['k'] - 1e-9:
                if not self.oel_on and self.oel_picked_s is None:
                    self.oel_picked_s = self.t
                self.oel_on = True
                self.heat = min(self.heat, o['k'])
            elif self.oel_on and self.i_oel >= 0.0:
                self.oel_on = False
            if self.oel_on:
                e = o['ifd'] - ifd
                self.i_oel = min(self.i_oel + o['ki'] * e * dt, 0.0)
                out += min(o['kp'] * e + self.i_oel, 0.0)
            else:
                self.i_oel = 0.0
        u = self.uel
        if u is not None:
            q_lim = v_pu * v_pu * (u['q0'] + (u['q1'] - u['q0']) * p_pu / max(v_pu * v_pu, 1e-3))
            e = q_lim - q_pu
            self.i_uel = max(self.i_uel + u['ki'] * e * dt, 0.0)
            y = max(u['kp'] * e + self.i_uel, 0.0)
            if y > 0.0 and self.uel_acted_s is None:
                self.uel_acted_s = self.t
            out += y
        return out


def exciter(model, p, efd0, v0, xadifd0):
    """Its exciter, started at field voltage efd0, terminal voltage v0 and field current xadifd0 (per unit)."""
    if model in ('ESST1A', 'AC8B'):
        return {'ESST1A': Esst1a, 'AC8B': Ac8b}[model](p, efd0, v0, xadifd0)
    return {'EXDC2': Exdc2, 'SEXS': Sexs, 'IEEEX1': Ieeex1, 'ESDC2A': Esdc2a, 'EXST1': Exst1}[model](p, efd0, v0)


def governor(model, p, tm0):
    return {'TGOV1': Tgov1, 'GAST': Gast, 'IEEEG1': Ieeeg1, 'IEESGO': Ieesgo, 'HYGOV': Hygov}[model](p, tm0)
