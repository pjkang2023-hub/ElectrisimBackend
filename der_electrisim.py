"""
Microgrid sources and stores on the DC side: battery, supercapacitor, flywheel,
SOFC system and PV array.

Each is generic in its sizing - its voltage, its capacity or energy and its
power entered in its dialog, for whatever bus it serves - either by its
ratings or from its building block (a cell or module, times modules in series
and strings in parallel). Each has one DC terminal on a DC bus, and gives:
- its terminal voltage at a current it delivers (its port's voltage, behind
  the converter that draws that current);
- its power limits at its present state;
- its state and what to report.

Its place in the load flow is _electrisim_build_ders in pandapower_electrisim:
behind a DC/DC converter it holds its own bus's voltage; directly on a network
bus it is what it physically is there (section 3 of the microgrid design note).
"""

import math

import numpy as np

FARADAY = 96485.33212
R_GAS = 8.314462618
K_BOLTZMANN = 1.380649e-23
Q_ELECTRON = 1.602176634e-19
LHV_H2_J_PER_MOL = 241.83e3         # hydrogen's lower heating value

KINDS = ('Battery', 'Supercapacitor', 'Flywheel', 'SOFC', 'PV Array')


def _f(value, default=0.0):
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _pos(value, default):
    v = _f(value, default)
    return v if v > 0 else default


def _flag(value, default=False):
    if value is None or value == '':
        return default
    if isinstance(value, str):
        return value.strip().lower() in ('true', '1', 'yes', 'on')
    return bool(value)


def kind_of(typ):
    """The element kind a payload type names, or None."""
    for kind in KINDS:
        if str(typ or '').startswith(kind):
            return kind
    return None


def build(el):
    """The element a payload row describes."""
    kind = kind_of(el.get('typ'))
    return {'Battery': Battery, 'Supercapacitor': Supercapacitor, 'Flywheel': Flywheel,
            'SOFC': Sofc, 'PV Array': PvArray}[kind](el)


class _Der:
    kind = ''
    stiff = True            # behind its converter, it holds its own bus's voltage

    def __init__(self, el):
        self.name = str(el.get('name'))
        self.label = str(el.get('userFriendlyName') or el.get('name'))
        self.id = el.get('id', '')
        self.bus_key = el.get('bus')
        self.in_service = _flag(el.get('in_service'), True)
        self.notes = []

    # What each kind gives.
    def v_terminal(self, i):
        """Its terminal voltage (V) delivering ``i`` (A; negative charges it)."""
        raise NotImplementedError

    def p_limits(self):
        """(most it can deliver, most it can take) in W, at its present state."""
        raise NotImplementedError

    def v_nominal(self):
        raise NotImplementedError

    def state(self, i, v):
        """What its results report at current ``i`` and terminal voltage ``v``."""
        return {}


# --- Battery ----------------------------------------------------------------------------

# A lithium iron phosphate cell's open-circuit voltage, per unit of its 3.2 V nominal, by state of charge.
LFP_OCV = ((0.0, 2.50), (0.05, 3.00), (0.10, 3.19), (0.20, 3.25), (0.30, 3.28), (0.50, 3.30),
           (0.70, 3.32), (0.90, 3.35), (0.95, 3.40), (1.00, 3.60))


def _ocv_table(text, nominal):
    """(SoC, OCV per unit) from 'soc:volts per cell' pairs or the LFP default."""
    pairs = []
    for part in str(text or '').replace(';', ',').replace('\n', ',').split(','):
        if ':' in part:
            a, b = part.split(':', 1)
            if math.isfinite(_f(a, np.nan)) and math.isfinite(_f(b, np.nan)):
                pairs.append((_f(a) / (100.0 if _f(a) > 1.0 else 1.0), _f(b)))
    if len(pairs) < 2:
        return np.array([s for s, _ in LFP_OCV]), np.array([v / 3.2 for _, v in LFP_OCV])
    pairs.sort()
    soc = np.array([s for s, _ in pairs])
    v = np.array([x for _, x in pairs])
    return soc, v / (nominal if nominal > 0 else 1.0)


class Battery(_Der):
    """
    Its open-circuit voltage by state of charge (a table, LFP by default)
    behind its series resistance and one RC pair:
        v = OCV(SoC) - R0 i - v1,   dv1/dt = i/C1 - v1/(R1 C1),
        dSoC/dt = -i eta / (3600 Q_Ah)    (eta its coulombic efficiency charging, 1 discharging).
    Sized by its ratings (nominal voltage, capacity, R0) or from its cell
    (voltage, capacity, resistance) times cells in series and strings in parallel.
    """
    kind = 'Battery'

    def __init__(self, el):
        super().__init__(el)
        if str(el.get('sizing') or '') == 'cells':
            n_s, n_p = max(1, int(_f(el.get('cells_series'), 250))), max(1, int(_f(el.get('strings_parallel'), 1)))
            v_cell, ah_cell = _pos(el.get('cell_v'), 3.2), _pos(el.get('cell_ah'), 280.0)
            self.vn = n_s * v_cell
            self.ah = n_p * ah_cell
            self.r0 = _f(el.get('cell_r_mohm'), 0.25) * 1e-3 * n_s / n_p
        else:
            self.vn = _pos(el.get('vn_v'), 800.0)
            kwh = _f(el.get('capacity_kwh'), 0.0)
            self.ah = kwh * 1e3 / self.vn if kwh > 0 else _pos(el.get('capacity_ah'), 280.0)
            self.r0 = _f(el.get('r0_mohm'), 62.5) * 1e-3
        # Its RC pair's resistance: in mOhm, or as a share of R0 (40 % by default).
        r1_mohm = _f(el.get('r1_mohm'), float('nan'))
        self.r1 = r1_mohm * 1e-3 if math.isfinite(r1_mohm) else max(_f(el.get('r1_percent'), 40.0), 0.0) / 100.0 * self.r0
        self.l = max(_f(el.get('l_uh'), 0.0), 0.0) * 1e-6        # its series inductance, for a DC fault
        tau1 = _pos(el.get('tau1_s'), 30.0)
        self.c1 = tau1 / self.r1 if self.r1 > 0 else 0.0
        self.soc0 = min(max(_f(el.get('soc_percent'), 50.0) / 100.0, 0.0), 1.0)
        self.soc_min = _f(el.get('soc_min_percent'), 10.0) / 100.0
        self.soc_max = _f(el.get('soc_max_percent'), 90.0) / 100.0
        self.c_rate_dis = _pos(el.get('c_rate_discharge'), 1.0)
        self.c_rate_ch = _pos(el.get('c_rate_charge'), 0.5)
        self.eta_charge = min(max(_f(el.get('coulombic_efficiency_percent'), 99.0) / 100.0, 0.5), 1.0)
        # Its OCV table is per cell: volts per cell over the cell's nominal (3.2 V by ratings).
        per_cell = str(el.get('sizing') or '') == 'cells'
        self.ocv_soc, self.ocv_pu = _ocv_table(el.get('ocv_table'), _pos(el.get('cell_v'), 3.2) if per_cell else 3.2)
        self.energy_kwh = self.vn * self.ah / 1e3

    def v_nominal(self):
        return self.vn

    def ocv(self, soc=None):
        return self.vn * float(np.interp(self.soc0 if soc is None else soc, self.ocv_soc, self.ocv_pu))

    def v_terminal(self, i):
        # In a steady state its RC pair has settled: v1 = R1 i.
        return self.ocv() - (self.r0 + self.r1) * i

    def p_limits(self):
        v = self.ocv()
        dis = self.c_rate_dis * self.ah * v if self.soc0 > self.soc_min + 1e-9 else 0.0
        ch = self.c_rate_ch * self.ah * v if self.soc0 < self.soc_max - 1e-9 else 0.0
        return dis, ch

    def state(self, i, v):
        c_rate = i / self.ah if self.ah > 0 else 0.0
        out = {'soc_percent': 100.0 * self.soc0, 'ocv_v': self.ocv(), 'c_rate': c_rate,
               'energy_kwh': self.energy_kwh, 'capacity_ah': self.ah}
        if c_rate > self.c_rate_dis + 1e-9:
            self.notes.append(f'discharging at {c_rate:.2f} C, above its {self.c_rate_dis:g} C')
        if -c_rate > self.c_rate_ch + 1e-9:
            self.notes.append(f'charging at {-c_rate:.2f} C, above its {self.c_rate_ch:g} C')
        if not self.soc_min - 1e-9 <= self.soc0 <= self.soc_max + 1e-9:
            self.notes.append(f'its state of charge ({100 * self.soc0:.0f} %) is outside its window '
                              f'({100 * self.soc_min:.0f}-{100 * self.soc_max:.0f} %)')
        return out


# --- Supercapacitor ---------------------------------------------------------------------

class Supercapacitor(_Der):
    """
    A capacitance behind its ESR, its leakage resistance across it; energy
    1/2 C (V^2 - V_min^2) usable down to V_min (half its rated voltage by
    default). Sized by its ratings (capacitance, rated voltage, ESR) or from
    its module times modules in series and strings in parallel. It connects
    through a DC/DC converter or directly on a bus (passive).
    """
    kind = 'Supercapacitor'

    def __init__(self, el):
        super().__init__(el)
        if str(el.get('sizing') or '') == 'modules':
            n_s, n_p = max(1, int(_f(el.get('modules_series'), 1))), max(1, int(_f(el.get('strings_parallel'), 1)))
            c_m, v_m, esr_m = _pos(el.get('module_c_f'), 130.0), _pos(el.get('module_v'), 54.0), _f(el.get('module_esr_mohm'), 4.0)
            self.c = c_m * n_p / n_s
            self.v_rated = v_m * n_s
            self.esr = esr_m * 1e-3 * n_s / n_p
            self.esl = max(_f(el.get('module_esl_uh'), 0.0), 0.0) * 1e-6 * n_s / n_p
        else:
            self.c = _pos(el.get('c_f'), 130.0)
            self.v_rated = _pos(el.get('v_rated'), 54.0)
            self.esr = _f(el.get('esr_mohm'), 4.0) * 1e-3
            self.esl = max(_f(el.get('esl_uh'), 0.0), 0.0) * 1e-6
        self.r_leak = _pos(el.get('r_leak_ohm'), 1e4)
        self.v_min = min(max(_f(el.get('v_min_percent'), 50.0), 0.0), 99.0) / 100.0 * self.v_rated
        self.v0 = min(max(_f(el.get('v0_percent'), 90.0), 0.0), 100.0) / 100.0 * self.v_rated
        self.direct = str(el.get('coupling') or 'converter') == 'direct'
        self.p_rated = _f(el.get('p_rated_kw'), 0.0) * 1e3

    def v_nominal(self):
        return self.v_rated

    def energy(self, v=None):
        v = self.v0 if v is None else v
        return 0.5 * self.c * v * v, 0.5 * self.c * max(v * v - self.v_min * self.v_min, 0.0)

    def v_terminal(self, i):
        return self.v0 - self.esr * i

    def p_limits(self):
        # Its ESR's matched-load power, or its rating if one is entered.
        p = self.p_rated if self.p_rated > 0 else self.v0 * self.v0 / (4.0 * self.esr) if self.esr > 0 else 1e12
        return (p if self.v0 > self.v_min + 1e-9 else 0.0), (p if self.v0 < self.v_rated - 1e-9 else 0.0)

    def state(self, i, v):
        total, usable = self.energy()
        return {'v_cap_v': self.v0, 'capacitance_f': self.c, 'energy_kj': total / 1e3, 'usable_kj': usable / 1e3,
                'soc_percent': 100.0 * usable / max(self.energy(self.v_rated)[1], 1e-12)}


# --- Flywheel ---------------------------------------------------------------------------

class Flywheel(_Der):
    """
    Its rotor's energy 1/2 J w^2 between its speed window's ends; its power
    limited to its rating above base speed and to its torque limit below,
    P_max(w) = P_rated min(1, w / w_base). Its machine and machine converter
    are averaged: the network sees its DC link, which the machine converter
    holds.
    """
    kind = 'Flywheel'

    def __init__(self, el):
        super().__init__(el)
        self.v_dc = _pos(el.get('v_dc'), 800.0)
        self.e_max = _pos(el.get('e_max_kwh'), 2.0) * 3.6e6
        self.p_rated = _pos(el.get('p_rated_kw'), 250.0) * 1e3
        self.s_min = min(max(_f(el.get('speed_min_percent'), 50.0), 0.0), 99.0) / 100.0
        self.s_base = min(max(_f(el.get('speed_base_percent'), 50.0), 1.0), 100.0) / 100.0
        self.s0 = min(max(_f(el.get('speed_percent'), 90.0), 0.0), 100.0) / 100.0
        self.eta = min(max(_f(el.get('efficiency_percent'), 95.0) / 100.0, 0.5), 1.0)
        self.standby = _f(el.get('standby_loss_percent_h'), 2.0) / 100.0
        self.r_dc = _f(el.get('r_dc_mohm'), 1.0) * 1e-3
        self.p_set = _f(el.get('p_set_kw'), 0.0) * 1e3     # directly on a bus: what its machine converter delivers

    def v_nominal(self):
        return self.v_dc

    def v_terminal(self, i):
        return self.v_dc - self.r_dc * i

    def p_limits(self):
        p = self.p_rated * min(1.0, self.s0 / self.s_base)
        return (p if self.s0 > self.s_min + 1e-9 else 0.0), (p if self.s0 < 1.0 - 1e-9 else 0.0)

    def state(self, i, v):
        energy = self.e_max * self.s0 ** 2
        usable = self.e_max * max(self.s0 ** 2 - self.s_min ** 2, 0.0)
        return {'speed_percent': 100.0 * self.s0, 'energy_kwh': energy / 3.6e6, 'usable_kwh': usable / 3.6e6,
                'p_max_kw': self.p_limits()[0] / 1e3, 'standby_loss_kw': self.standby * self.e_max / 3600.0 / 1e3}


# --- SOFC system ------------------------------------------------------------------------

class Sofc(_Der):
    """
    The Padulles SOFC stack (IEEE Trans. Energy Conversion 2000): in steady
    state, at fuel utilisation U,
        V = N0 (E0 + (RT/2F) ln(pH2 sqrt(pO2) / pH2O)) - r I,
        pH2 = 2 Kr I (1/U - 1) / K_H2,  pH2O = 2 Kr I / K_H2O,
        pO2 = (2 Kr I / (U r_HO) - Kr I) / K_O2,   Kr = N0 / 4F;
    its cells are its 384-cell, 100 kW stack's (a cell's voltage that stack's
    over N0 at the same current), cells in series setting its voltage and
    stacks in parallel its current. Its balance of plant draws its auxiliary
    load from the stack. It runs between its minimum load and its rating,
    changing no faster than its ramp rate.
    """
    kind = 'SOFC'
    N0, P0, E0_CELL, R_STACK, T_K = 384, 100e3, 1.18, 0.126, 1273.0
    K_H2, K_H2O, K_O2, R_HO = 8.43e-4, 2.81e-4, 2.52e-3, 1.145

    def __init__(self, el):
        super().__init__(el)
        self.p_rated = _pos(el.get('p_rated_kw'), 100.0) * 1e3
        self.v_rated = _pos(el.get('v_rated'), 800.0)
        self.u = min(max(_f(el.get('fuel_utilisation_percent'), 85.0) / 100.0, 0.5), 0.95)
        self.ramp = _pos(el.get('ramp_percent_s'), 1.0) / 100.0
        self.p_min_frac = min(max(_f(el.get('min_load_percent'), 30.0) / 100.0, 0.0), 1.0)
        self.aux_frac = min(max(_f(el.get('aux_load_percent'), 5.0) / 100.0, 0.0), 0.5)
        self.p_set = _f(el.get('p_set_kw'), 0.8 * self.p_rated / 1e3) * 1e3
        i_col = self._i_at(self.P0)
        v_cell = self._v_ref(i_col) / self.N0
        self.n_series = max(1, int(round(self.v_rated / v_cell)))
        self.n_parallel = self.p_rated / (self.n_series * v_cell * i_col)
        self.i_rated = self.n_parallel * i_col

    def _v_ref(self, i):
        """The 100 kW stack's voltage at its current ``i`` (A), in steady state."""
        i = max(i, 1e-6)
        kr = self.N0 / (4.0 * FARADAY * 1e3)                 # kmol/s per A
        q_h2 = 2.0 * kr * i / self.u                          # the fuel fed, at its utilisation
        p_h2 = max(q_h2 - 2.0 * kr * i, 1e-12) / self.K_H2
        p_h2o = 2.0 * kr * i / self.K_H2O
        p_o2 = max(q_h2 / self.R_HO - kr * i, 1e-12) / self.K_O2
        nernst = self.E0_CELL + R_GAS * self.T_K / (2.0 * FARADAY) * math.log(p_h2 * math.sqrt(p_o2) / p_h2o)
        return self.N0 * nernst - self.R_STACK * i

    def _i_at(self, p):
        """The 100 kW stack's current giving power ``p`` (W), on its rising branch."""
        lo, hi = 1e-6, 1000.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if mid * self._v_ref(mid) < p:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def v_nominal(self):
        return self.v_rated

    def v_stack(self, i):
        """Its stack's voltage at its current ``i`` (A)."""
        return self.n_series * self._v_ref(max(i, 1e-6) / max(self.n_parallel, 1e-12)) / self.N0

    def v_terminal(self, i):
        # Its auxiliary load comes from the stack: the stack's current is the port's plus the auxiliary's.
        v = self.v_stack(i)
        for _ in range(30):
            v = self.v_stack(i + self.aux_frac * self.p_rated / max(v, 1.0))
        return v

    def p_limits(self):
        return self.p_rated * (1.0 - self.aux_frac), 0.0

    def i_at_voltage(self, v):
        """
        The current (A) it delivers into a bus held at ``v`` (V), on its curve
        from its minimum load up (below it, at a fixed utilisation, the fuel
        fed vanishes and the model's voltage with it): (current, whether held
        at its minimum load).
        """
        i_min = max(self.p_min_frac * self.p_rated, 1e-3 * self.p_rated) / self.v_rated
        if v >= self.v_terminal(i_min):
            return i_min, True
        lo, hi = i_min, 3.0 * self.i_rated
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if self.v_terminal(mid) > v:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi), False

    def p_operating(self):
        """Its net set power, held between its minimum load and its rating less its auxiliary load."""
        p_max = self.p_rated * (1.0 - self.aux_frac)
        p_min = self.p_min_frac * self.p_rated
        p = min(max(self.p_set, p_min), p_max)
        if abs(p - self.p_set) > 1e-6:
            self.notes.append(f'its set power ({self.p_set / 1e3:g} kW) is held to {p / 1e3:g} kW, '
                              'between its minimum load and its rating less its auxiliary load')
        return p

    def state(self, i, v):
        i_stack = i + self.aux_frac * self.p_rated / max(v, 1.0)
        # Fuel fed: its stack's electrochemistry over its utilisation; every cell in series takes it.
        h2_mol_s = i_stack * self.n_series / (2.0 * FARADAY) / self.u
        fuel_w = h2_mol_s * LHV_H2_J_PER_MOL
        return {'stack_current_a': i_stack, 'stack_voltage_v': self.v_stack(i_stack),
                'aux_load_kw': self.aux_frac * self.p_rated / 1e3, 'fuel_utilisation_percent': 100.0 * self.u,
                'h2_kg_h': h2_mol_s * 2.016e-3 * 3600.0, 'fuel_power_kw': fuel_w / 1e3,
                'efficiency_percent': 100.0 * v * i / fuel_w if fuel_w > 0 else None,
                'cells_series': self.n_series, 'stacks_parallel': self.n_parallel}


# --- PV array ---------------------------------------------------------------------------

class PvArray(_Der):
    """
    Its module's single-diode model, fitted to its datasheet (Villalva, IEEE
    Trans. Power Electronics 2009: diode ideality 1.3, Rs raised until the
    model's maximum power meets the datasheet's at its Vmpp, Rp to match),
        I = Ipv - I0 (exp((V + Rs I) / (a Vt)) - 1) - (V + Rs I) / Rp,
    at its irradiance and cell temperature (ambient plus (NOCT - 20) G / 800);
    modules in series and strings in parallel, less its losses. Not stiff: it
    delivers the current its I-V curve gives at its terminal voltage.
    """
    kind = 'PV Array'
    stiff = False

    def __init__(self, el):
        super().__init__(el)
        self.p_mp = _pos(el.get('module_pmpp_w'), 550.0)
        self.v_mp = _pos(el.get('module_vmpp'), 41.9)
        self.i_mp = _pos(el.get('module_impp'), 13.13)
        self.v_oc = _pos(el.get('module_voc'), 49.9)
        self.i_sc = _pos(el.get('module_isc'), 14.0)
        self.n_cells = max(1, int(_f(el.get('module_cells_series'), 72)))      # a 144 half-cut-cell module: 72 in series
        self.k_i = _f(el.get('alpha_isc_percent_k'), 0.048) / 100.0 * self.i_sc
        self.k_v = _f(el.get('beta_voc_percent_k'), -0.27) / 100.0 * self.v_oc
        self.noct = _f(el.get('noct_c'), 45.0)
        self.n_s = max(1, int(_f(el.get('modules_series'), 18)))
        self.n_p = max(1, int(_f(el.get('strings_parallel'), 10)))
        self.loss = min(max(_f(el.get('loss_percent'), 3.0) / 100.0, 0.0), 0.5)
        self.g = max(_f(el.get('irradiance_wm2'), 1000.0), 0.0)
        self.t_amb = _f(el.get('ambient_c'), 25.0)
        # Its diode's ideality: 1.3 (Villalva), or the nearest that meets its datasheet.
        for a in (1.3, 1.2, 1.4, 1.1, 1.5, 1.0, 1.7, 2.0):
            self.a = a
            if self._fit():
                break
        else:
            raise ValueError(f"PV Array '{self.label}': no single-diode model meets its module's datasheet - "
                             'check Pmpp, Vmpp, Impp, Voc, Isc and its cells in series.')
        self._conditions(self.g, self.t_amb)

    # The fit, at standard test conditions.
    def _vt(self, t_c):
        return self.n_cells * K_BOLTZMANN * (t_c + 273.15) / Q_ELECTRON

    def _fit(self):
        """
        Its series and shunt resistances for its diode's ideality: Rp from the
        datasheet's power at Vmpp (Ipv updated with it), Rs raised until the
        model's maximum power is the datasheet's, at Vmpp. False when no
        finite Rp does it.
        """
        vt = self._vt(25.0)
        self.i0_stc = self.i_sc / (math.exp(self.v_oc / (self.a * vt)) - 1.0)
        # The maximum power point itself: a datasheet's rounded Pmpp can sit a little off Vmpp x Impp.
        p_mp = self.v_mp * self.i_mp

        def model(rs):
            """(Rp, Ipv, the model's maximum power) for series resistance ``rs``; None past its range."""
            ipv, rp = self.i_sc, None
            for _ in range(20):
                den = (self.v_mp * ipv - self.v_mp * self.i0_stc * math.exp((self.v_mp + self.i_mp * rs) / (vt * self.a))
                       + self.v_mp * self.i0_stc - p_mp)
                if den <= 0:
                    return None
                rp = self.v_mp * (self.v_mp + self.i_mp * rs) / den
                ipv = (rp + rs) / rp * self.i_sc
            self.rs, self.rp = rs, rp
            return rp, ipv, self._p_max_at(ipv, self.i0_stc, vt)

        # Its model's maximum power is never below the datasheet's (Rp makes them equal at Vmpp), and
        # is least - equal to it - where its maximum power point is Vmpp: the Rs to find, by golden section
        # over the range of Rs that leaves Rp finite.
        if model(0.0) is None:
            return False
        rs_hi, step = 0.0, 0.005
        while rs_hi < 5.0 and model(rs_hi + step) is not None:
            rs_hi += step
        lo, hi = 0.0, rs_hi
        for _ in range(60):
            m1, m2 = lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0
            if model(m1)[2] < model(m2)[2]:
                hi = m2
            else:
                lo = m1
        lo = 0.5 * (lo + hi)
        m = model(lo)
        if m is None or m[0] > 1e6 * self.v_oc / self.i_sc or abs(m[2] - p_mp) > 1e-3 * p_mp:
            return False
        self.rs, self.rp, self.ipv_stc = lo, m[0], m[1]
        return True

    def _i_module(self, v, ipv, i0, vt):
        """The module's current at its voltage ``v`` (Newton on the implicit equation)."""
        i = ipv
        for _ in range(60):
            e = math.exp(min((v + self.rs * i) / (self.a * vt), 700.0))
            f = ipv - i0 * (e - 1.0) - (v + self.rs * i) / self.rp - i
            df = -i0 * e * self.rs / (self.a * vt) - self.rs / self.rp - 1.0
            step = f / df
            i -= step
            if abs(step) < 1e-10:
                break
        return i

    def _p_max_at(self, ipv, i0, vt):
        lo, hi = 0.0, self.v_oc * 1.2
        for _ in range(80):
            m1, m2 = lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0
            if m1 * self._i_module(m1, ipv, i0, vt) < m2 * self._i_module(m2, ipv, i0, vt):
                lo = m1
            else:
                hi = m2
        v = 0.5 * (lo + hi)
        return v * self._i_module(v, ipv, i0, vt)

    # At its irradiance and temperature.
    def _conditions(self, g, t_amb):
        self.t_cell = t_amb + (self.noct - 20.0) * g / 800.0
        dt = self.t_cell - 25.0
        self.vt = self._vt(self.t_cell)
        self.ipv = (self.ipv_stc + self.k_i * dt) * g / 1000.0
        self.i0 = (self.i_sc + self.k_i * dt) / (math.exp((self.v_oc + self.k_v * dt) / (self.a * self.vt)) - 1.0)
        self.v_oc_t = self._v_at_module(0.0)

    def _v_at_module(self, i):
        """The module's voltage delivering ``i`` (bisection: its current falls with its voltage)."""
        if i >= self.ipv or self.ipv <= 0:
            return 0.0
        lo, hi = 0.0, 1.5 * self.v_oc
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if self._i_module(mid, self.ipv, self.i0, self.vt) > i:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def i_array(self, v):
        """The array's current (A) at its terminal voltage ``v``, its losses taken from its current."""
        if v <= 0:
            return self.n_p * (1.0 - self.loss) * self._i_module(0.0, self.ipv, self.i0, self.vt)
        return max(self.n_p * (1.0 - self.loss) * self._i_module(v / self.n_s, self.ipv, self.i0, self.vt), 0.0)

    def v_nominal(self):
        return self.n_s * self.v_mp

    def mpp(self):
        """(V, I, P) at its maximum power point at its irradiance and temperature."""
        if self.ipv <= 0:
            return 0.0, 0.0, 0.0
        lo, hi = 0.0, self.n_s * self.v_oc_t
        for _ in range(100):
            m1, m2 = lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0
            if m1 * self.i_array(m1) < m2 * self.i_array(m2):
                lo = m1
            else:
                hi = m2
        v = 0.5 * (lo + hi)
        i = self.i_array(v)
        return v, i, v * i

    def v_terminal(self, i):
        per_string = i / (self.n_p * (1.0 - self.loss))
        return self.n_s * self._v_at_module(per_string)

    def p_limits(self):
        return self.mpp()[2], 0.0

    def state(self, i, v):
        v_mp, i_mp, p_mp = self.mpp()
        return {'irradiance_wm2': self.g, 'cell_temperature_c': self.t_cell, 'p_mpp_kw': p_mp / 1e3,
                'v_mpp_v': v_mp, 'v_oc_v': self.n_s * self.v_oc_t, 'i_sc_a': self.i_array(0.0),
                'modules': self.n_s * self.n_p}
