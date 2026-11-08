"""
Exciters and governors for the EMT study's detailed machine, as ANDES has them, so that a machine is the same
in both studies: EXDC2 and SEXS, TGOV1 and GAST. Their parameters come from where ANDES's study takes them:
ANDES's own defaults, then Electrisim's (andes_electrisim._EXCITER_DEFAULTS, _GOVERNOR_DEFAULTS), then the
generator's dyn_exc_ and dyn_gov_ fields.

Each block is ANDES's: a lag K / (1 + s T); a lag whose output is held within its limits (anti-windup);
a lead-lag (1 + s T1) / (1 + s T2), its output its input where T2 is 0; a washout K s / (1 + s T). Each steps
by the exact solution over the step for its input held. Exciters work in per unit of the machine's field
(Efd at its air-gap line), governors in per unit of its rating.
"""
import math
from functools import lru_cache

EXCITERS = ('EXDC2', 'SEXS')
GOVERNORS = ('TGOV1', 'GAST')


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
    """(model, its parameters) for kind 'exc' or 'gov' as ANDES's study sets them up; None for none."""
    from andes_electrisim import _EXCITER_DEFAULTS, _GOVERNOR_DEFAULTS, _model_kwargs
    field, prefix, table, default = (('dyn_exciter_model', 'dyn_exc_', _EXCITER_DEFAULTS, 'EXDC2') if kind == 'exc'
                                     else ('dyn_governor_model', 'dyn_gov_', _GOVERNOR_DEFAULTS, 'TGOV1'))
    model = str((row or {}).get(field) or default).strip().upper()
    if model in ('', 'NONE', 'OFF'):
        return None
    if model not in table:
        model = default
    params = dict(andes_defaults(model))
    params.update(_model_kwargs(row or {}, prefix, table[model]))
    return model, params


class Exdc2:
    """
    IEEE DC2A as ANDES's EXDC2: a sensing lag TR, the lead-lag TC/TB, the regulator KA/TA within VRMIN/VRMAX,
    the exciter TE with KE and its quadratic saturation through (E1, SE1) and (E2, SE2), the rate feedback
    KF1 s / (1 + s TF1). Its field voltage is the exciter's output times the speed.
    """

    def __init__(self, p, efd0, v0):
        self.p = p
        e1, se1, e2, se2 = p['E1'], p['SE1'], p['E2'], p['SE2']
        if se2 == 0.0:
            e2, se2 = e2 + 2.0, se2 + 2.0
        a = math.sqrt(se1 * e1 / (se2 * e2)) if se2 != 0.0 else 0.0
        self.sat_a = e2 - (e1 - e2) / (a - 1.0) if a != 1.0 else e2
        self.sat_b = se2 * e2 * (a - 1.0) ** 2 / (e1 - e2) ** 2 if a != 0.0 and e1 != e2 else 0.0
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

    def step(self, dt, v, omega):
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

    def step(self, dt, v, omega):
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


def exciter(model, p, efd0, v0):
    return {'EXDC2': Exdc2, 'SEXS': Sexs}[model](p, efd0, v0)


def governor(model, p, tm0):
    return {'TGOV1': Tgov1, 'GAST': Gast}[model](p, tm0)
