"""
The microgrid through a time series: each source's and store's state from
step to step, the profiles they follow, the smoothing controllers, and the
rule-based dispatch.

Each step, before its load flow:
  - DC loads follow their library profiles (AC loads already do); PV arrays
    their irradiance and temperature profiles;
  - each SOFC moves toward its target - its set power, or under the dispatch
    the slow part of the demand - no faster than its ramp rate, between its
    minimum load and its rating;
  - each smoothing converter sets its store's power from its output bus's
    loads, P_store = P_rack - LPF_tau(P_rack) + k (SoC - SoC_ref) P_rated,
    within the store's and converter's limits (shared between the smoothing
    converters on one bus by their ratings);
  - each battery (and flywheel) dispatched or set is held within its window
    and its C-rate.
After it - under the dispatch - the batteries take what the grid would
supply, until the grid's exchange is zero or they reach their limits;
islanded, when the grid-forming PCS's batteries cannot supply or absorb what
the island asks, load is shed or PV curtailed. Then each store's state is
advanced by what it delivered over the step: a battery's charge, a
supercapacitor's and a flywheel's energy, an SOFC's fuel.

Signs: a source's or store's power is positive delivering.
"""
import math

import numpy as np

import der_electrisim
import load_profiles_electrisim as lp

DISPATCH_ITERATIONS = 20
SOFC_DISPATCH_TAU_S = 900.0     # the SOFC follows the demand averaged over this, under the dispatch
LONG_STEP_TAUS = 5.0            # a smoothing filter over a step this many time constants long holds its store


def _f(value, default=0.0):
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _real_rows(df):
    """A load table's rows that are the diagram's own: no converter-stage role, no stand-in."""
    keep = np.ones(len(df), dtype=bool)
    if 'electrisim_dcdc_role' in df.columns:
        keep &= df['electrisim_dcdc_role'].isna().values
    if 'electrisim_aux' in df.columns:
        keep &= (df['electrisim_aux'] != True).values
    return df.index[keep]


# --- Each store's state ---------------------------------------------------------------------

def soc_percent(obj):
    """A store's state of charge (%): a battery's own; a supercapacitor's or flywheel's usable energy left."""
    if obj.kind == 'Battery':
        return 100.0 * obj.soc0
    if obj.kind == 'Supercapacitor':
        span = obj.v_rated ** 2 - obj.v_min ** 2
        return 100.0 * (obj.v0 ** 2 - obj.v_min ** 2) / span if span > 0 else 0.0
    if obj.kind == 'Flywheel':
        span = 1.0 - obj.s_min ** 2
        return 100.0 * (obj.s0 ** 2 - obj.s_min ** 2) / span if span > 0 else 0.0
    return None


def usable_energy_j(obj):
    """A store's usable energy over its whole window (J)."""
    if obj.kind == 'Battery':
        return battery_energy_j(obj, obj.soc_max) - battery_energy_j(obj, obj.soc_min)
    if obj.kind == 'Supercapacitor':
        return 0.5 * obj.c * (obj.v_rated ** 2 - obj.v_min ** 2)
    if obj.kind == 'Flywheel':
        return obj.e_max * (1.0 - obj.s_min ** 2)
    return 0.0


def battery_energy_j(obj, soc):
    """A battery's energy from empty to ``soc`` (J): its open-circuit voltage integrated over its charge."""
    xs, ys = obj.ocv_soc, obj.ocv_pu
    pts = [0.0] + [x for x in xs if 0.0 < x < soc] + [soc]
    vals = [float(np.interp(x, xs, ys)) for x in pts]
    area = sum(0.5 * (vals[k] + vals[k + 1]) * (pts[k + 1] - pts[k]) for k in range(len(pts) - 1))
    return area * obj.vn * obj.ah * 3600.0


def energy_at_soc_j(obj, soc_pct):
    """What a store holds at a state of charge (%), as soc_percent counts it."""
    x = min(max(soc_pct / 100.0, 0.0), 1.0)
    if obj.kind == 'Battery':
        return battery_energy_j(obj, x)
    if obj.kind == 'Supercapacitor':
        return 0.5 * obj.c * (obj.v_min ** 2 + x * (obj.v_rated ** 2 - obj.v_min ** 2))
    if obj.kind == 'Flywheel':
        return obj.e_max * (obj.s_min ** 2 + x * (1.0 - obj.s_min ** 2))
    return 0.0


def stored_energy_j(obj):
    """What a store holds (J): a battery's energy over its OCV curve; a capacitor's or rotor's energy."""
    if obj.kind == 'Battery':
        return battery_energy_j(obj, obj.soc0)
    if obj.kind == 'Supercapacitor':
        return 0.5 * obj.c * obj.v0 ** 2
    if obj.kind == 'Flywheel':
        return obj.e_max * obj.s0 ** 2
    return 0.0


def available_w(obj, p, dt=None):
    """
    The power a store can deliver (p > 0) or take (p < 0) at its present state:
    ``p`` held to its limits and, over a step of ``dt`` (s), to what is left in
    its window - so a step ends at its window's edge, not past it.
    """
    if obj.kind == 'Battery':
        v = obj.ocv()
        if p > 0:
            lim = obj.c_rate_dis * obj.ah * v
            charge = (obj.soc0 - obj.soc_min) * obj.ah * 3600.0           # coulombs left to its window's edge
        else:
            lim = obj.c_rate_ch * obj.ah * v
            charge = (obj.soc_max - obj.soc0) * obj.ah * 3600.0 / obj.eta_charge
        left = charge * v
        if dt:
            # Its charge moves at p / v_terminal: the window's edge at its terminal voltage there.
            for _ in range(3):
                i = math.copysign(min(abs(p), lim, max(left, 0.0) / dt), p) / max(v, 1e-9)
                v = max(obj.v_terminal(i), 1e-9)
                left = charge * v
            lim = min(lim, max(left, 0.0) / dt)
        elif left <= 0:
            lim = 0.0
        return math.copysign(min(abs(p), lim), p)
    if obj.kind in ('Supercapacitor', 'Flywheel'):
        dis, ch = obj.p_limits()
        if dt:
            if obj.kind == 'Supercapacitor':
                stored, low, high = 0.5 * obj.c * obj.v0 ** 2, 0.5 * obj.c * obj.v_min ** 2, 0.5 * obj.c * obj.v_rated ** 2
            else:
                stored, low, high = obj.e_max * obj.s0 ** 2, obj.e_max * obj.s_min ** 2, obj.e_max
            dis = min(dis, max(stored - low, 0.0) / dt)
            ch = min(ch, max(high - stored, 0.0) / dt)
        return min(p, dis) if p > 0 else max(p, -ch)
    return p


def advance(obj, p_w, v, dt):
    """
    A store's or source's state after delivering ``p_w`` (W) at its terminal
    voltage ``v`` (V) for ``dt`` (s); returns the energy it drew from its
    store (J) - its terminal energy plus its own losses - for the bookkeeping.
    """
    i = p_w / max(v, 1e-9)
    if obj.kind == 'Battery':
        charge = i * dt if i >= 0 else i * obj.eta_charge * dt          # coulombs out of its cells
        before = battery_energy_j(obj, obj.soc0)
        obj.soc0 = min(max(obj.soc0 - charge / (obj.ah * 3600.0), 0.0), 1.0)
        return before - battery_energy_j(obj, obj.soc0)
    if obj.kind == 'Supercapacitor':
        e = 0.5 * obj.c * obj.v0 ** 2
        drawn = (p_w + obj.esr * i * i + obj.v0 ** 2 / obj.r_leak) * dt
        e = max(e - drawn, 0.0)
        obj.v0 = math.sqrt(2.0 * e / obj.c)
        return drawn
    if obj.kind == 'Flywheel':
        e = obj.e_max * obj.s0 ** 2
        mech = p_w / obj.eta if p_w >= 0 else p_w * obj.eta
        drawn = (mech + obj.standby * obj.e_max / 3600.0) * dt
        e = min(max(e - drawn, 0.0), obj.e_max)
        obj.s0 = math.sqrt(e / obj.e_max)
        return drawn
    if obj.kind == 'SOFC':
        st = obj.state(i, v)
        return (st.get('fuel_power_kw') or 0.0) * 1e3 * dt
    return p_w * dt


# --- The run --------------------------------------------------------------------------------

class MicrogridTs:
    """Created when a network has sources, stores, PCS or DC loads that follow profiles; None otherwise."""

    @classmethod
    def create(cls, net, params, step_s, time_steps):
        ders = getattr(net, 'electrisim_ders', None) or []
        pcs = getattr(net, 'electrisim_pcs', None) or []
        dc_profiles = params.get('dc_load_profile_assignments') or {}
        if not ders and not pcs and not dc_profiles:
            return None
        return cls(net, params, step_s, time_steps)

    def __init__(self, net, params, step_s, time_steps):
        import pandapower_electrisim as pe
        self.pe, self.net, self.dt, self.n = pe, net, float(step_s), int(time_steps)
        self.dispatch = params.get('microgrid_dispatch') in (True, 'true', 'True', 1, '1')
        self.sofc_tau = max(_f(params.get('sofc_tau_s'), SOFC_DISPATCH_TAU_S), 1e-6)
        self.notes = []
        self.library, problems = lp.library_from_params(params)
        self.notes += problems
        self.repeat = params.get('profile_repeat', True) not in (False, 'false', 'False', 0, '0')
        self.ders = list(getattr(net, 'electrisim_ders', None) or [])
        self.pcs = list(getattr(net, 'electrisim_pcs', None) or [])
        self.convs = list(getattr(net, 'electrisim_dc_dc_converters', None) or [])
        # Every element's record, with its object: DC-coupled ones, and those behind a PCS.
        self.items = [{'rec': r, 'obj': r['obj'], 'via': 'dc'} for r in self.ders]
        self.items += [{'rec': r, 'obj': r['source']['obj'], 'via': 'pcs'} for r in self.pcs]
        for it in self.items:
            it['e0'] = stored_energy_j(it['obj'])
            it['drawn_j'] = 0.0
            it['delivered_j'] = 0.0
            it['soc'] = []
        # DC loads following a library profile, and the loads' own powers.
        ld = net.load_dc
        real = ld.index if 'electrisim_dcdc_role' not in ld.columns else ld.index[ld['electrisim_dcdc_role'].isna()]
        self.dc_loads = {int(i): float(ld.at[i, 'p_dc_mw']) for i in real}
        # The diagram's own AC loads: not a converter stage's input (an SST's MV draw) or a stand-in.
        self.real_ac = [int(i) for i in _real_rows(net.load)]
        self.ac_loads = {i: (float(net.load.at[i, 'p_mw']), float(net.load.at[i, 'q_mvar'])) for i in self.real_ac}
        self.dc_profiles = {}
        for name, a in (params.get('dc_load_profile_assignments') or {}).items():
            idx = [i for i in self.dc_loads if str(ld.at[i, 'name']) == str(name)]
            values = self._profile_values(a['profile_id'], a['display_name'])
            if idx and values is not None:
                self.dc_profiles[idx[0]] = values
        # PV arrays' irradiance and temperature.
        for it in self.items:
            obj = it['obj']
            if obj.kind != 'PV Array':
                continue
            it['g'] = self._profile_values(obj.fields.get('irradiance_profile_id'), f'{obj.label} irradiance')
            it['t_amb'] = self._profile_values(obj.fields.get('temperature_profile_id'), f'{obj.label} temperature')
        # SOFC: its starting operating point is its target's.
        for it in self.items:
            if it['obj'].kind == 'SOFC':
                it['target_w'] = it['obj'].p_set
                it['obj'].p_set = it['obj'].p_operating()
        self.obj_notes = set()
        # Smoothing converters, by the bus they smooth.
        self.smoothing = []
        for c in self.convs:
            if c.get('control') != 'smoothing' or not c['in_service'] or c['output_load'] is None:
                continue
            store = next((it for it in self.items if it['via'] == 'dc' and it['rec']['bus'] == c['bus_in']), None)
            if store is None or store['obj'].kind not in ('Battery', 'Supercapacitor', 'Flywheel'):
                self.notes.append(f"DC/DC Converter '{c['label']}': smoothing needs a battery, supercapacitor or "
                                  "flywheel on its input; it delivers nothing.")
                continue
            sm = c['smoothing']
            self.smoothing.append({'conv': c, 'store': store, 'tau': max(sm['tau_s'], 1e-6),
                                   'soc_ref': sm['soc_ref_percent'], 'k': sm['soc_gain'], 'y': None,
                                   'rows': [], 'limited': 0})
        on_bus = {}
        for s in self.smoothing:
            on_bus.setdefault(s['conv']['bus_out'], []).append(s)
        for group in on_bus.values():
            total = sum(max(s['conv']['rated_mw'], 1e-9) for s in group)
            for s in group:
                s['share'] = max(s['conv']['rated_mw'], 1e-9) / total
        # The dispatch: batteries behind dispatch converters or grid-following PCS; SOFCs; the grid.
        self.batteries = []
        for it in self.items:
            if it['obj'].kind not in ('Battery', 'Flywheel'):
                continue
            rec = it['rec']
            if it['via'] == 'pcs' and rec['table'] == 'sgen':
                self.batteries.append({'item': it, 'kind': 'pcs', 'rated': rec['s_rated']})
            elif it['via'] == 'dc':
                conv = next((c for c in rec.get('converters') or [] if c['bus_in'] == rec['bus']
                             and c.get('control') in ('dispatch', 'power') and c['output_load'] is not None), None)
                if conv is not None:
                    self.batteries.append({'item': it, 'kind': 'conv', 'conv': conv,
                                           'rated': conv['rated_mw'] if conv['rated_mw'] > 0 else 1.0})
        for b in self.batteries:
            b['set_mw'] = b['conv']['p_set_mw'] if b['kind'] == 'conv' else b['item']['rec']['p_set_ac']
        self.demand_lpf = None
        self.rows, self.pcs_rows, self.balance = [], [], []
        self.unserved_j = self.curtailed_j = 0.0
        self.unserved_steps = []
        self.shed = 1.0
        self.curtail = 1.0

    def _profile_values(self, pid, what):
        """A library profile's value at each step (its mean over a step longer than its samples), or None."""
        if not pid:
            return None
        prof = self.library.get(str(pid))
        if prof is None:
            self.notes.append(f'{what}: its profile is not in the library, so it does not follow one.')
            return None
        t = prof['t'] - prof['t'][0]
        dt_prof = float(np.median(np.diff(t)))
        if self.dt > 2 * dt_prof:
            return [lp.average_profile(t, prof['p'], k * self.dt, (k + 1) * self.dt, self.repeat) for k in range(self.n)]
        return [float(v) for v in lp.sample_profile(t, prof['p'], np.arange(self.n) * self.dt, self.repeat)]

    # --- Before the load flow --------------------------------------------------------------

    def before_step(self, t):
        net, pe = self.net, self.pe
        for i, p in self.dc_loads.items():
            scale = self.dc_profiles[i][t] if i in self.dc_profiles else 1.0
            self._set_dc_load(i, p * scale)
        self.base_ac = {i: (float(net.load.at[i, 'p_mw']), float(net.load.at[i, 'q_mvar'])) for i in self.real_ac}
        self.base_dc = {i: float(net.load_dc.at[i, 'p_dc_mw']) for i in self.dc_loads}
        self.shed, self.curtail = 1.0, 1.0
        for it in self.items:
            obj = it['obj']
            if obj.kind == 'PV Array' and (it.get('g') or it.get('t_amb')):
                g = it['g'][t] if it.get('g') else obj.g
                ta = it['t_amb'][t] if it.get('t_amb') else obj.t_amb
                obj.g, obj.t_amb = max(g, 0.0), ta
                obj._conditions(obj.g, obj.t_amb)
                it['rec'].pop('mpp', None)
            if obj.kind == 'SOFC':
                target = it['target_w']
                if self.dispatch and self.demand_lpf is not None:
                    target = self.demand_lpf.get(id(it), target)
                step = obj.ramp * obj.p_rated * self.dt
                p = obj.p_set + max(min(target - obj.p_set, step), -step)
                obj.p_set = p
                obj.p_set = obj.p_operating()          # its minimum load and rating
                it['target_now_w'] = target
        self._apply_sources()
        self._apply_smoothing(t)
        self._hold_pcs_supercapacitors()
        self._pcs_step_windows()
        for b in self.batteries:
            self._set_battery(b, b['set_mw'])

    def _pcs_step_windows(self):
        """
        Each grid-forming PCS's window for this step: what its store can give
        and take over it, within its own. An island shares its power within
        them: the eSTATCOM's 7.5 MW-s is 0.13 MW over a minute.
        """
        pe = self.pe
        for it in self.items:
            rec, obj = it['rec'], it['obj']
            if it['via'] != 'pcs' or rec['table'] != 'gen' or not rec['in_service']:
                continue
            dis, ch = obj.p_limits()
            hi = pe._electrisim_stage_output(available_w(obj, max(dis, 0.0), self.dt) / 1e6, rec['eta'], rec['p_nl_mw'])
            lo = pe._electrisim_stage_output(available_w(obj, -max(ch, 0.0), self.dt) / 1e6, rec['eta'], rec['p_nl_mw'])
            rec['p_max_step'] = min(max(hi, 0.0), rec['p_max_mw'])
            rec['p_min_step'] = max(min(lo, 0.0), rec['p_min_mw'])

    def _hold_pcs_supercapacitors(self):
        """
        A grid-forming PCS on supercapacitors with no set power - an eSTATCOM -
        holds its store at its charge at the start: it draws from the network
        what its store leaks and what returns it there, as a smoothing
        converter does over a long step. At none, its supercapacitors leaked
        0.66 MJ an hour against 7.5 MJ and ended the day below their minimum.
        """
        net, pe = self.net, self.pe
        for it in self.items:
            rec, obj = it['rec'], it['obj']
            if (it['via'] != 'pcs' or rec['table'] != 'gen' or obj.kind != 'Supercapacitor' or not rec['in_service']
                    or abs(rec.get('p_set_ac') or 0.0) > 1e-12):
                continue
            e_ref = it.setdefault('e_ref_j', stored_energy_j(obj))
            p_dc = (stored_energy_j(obj) - e_ref) / self.dt / 1e6
            if getattr(obj, 'r_leak', 0) > 0:
                p_dc -= obj.v0 ** 2 / obj.r_leak / 1e6
            net.gen.at[rec['index'], 'p_mw'] = pe._electrisim_stage_output(p_dc, rec['eta'], rec['p_nl_mw'])

    def _apply_sources(self):
        """PV arrays and SOFCs at their present power: their converters' outputs, or their PCS's."""
        pe, net = self.pe, self.net
        for it in self.items:
            obj, rec = it['obj'], it['rec']
            if obj.kind not in ('PV Array', 'SOFC'):
                continue
            if it['via'] == 'dc':
                pe._electrisim_der_set_converters(net, rec)
                if obj.kind == 'PV Array' and self.curtail < 1.0:
                    for c in rec['converters']:
                        if c['bus_in'] == rec['bus'] and c['output_load'] is not None and c['control'] == 'mppt':
                            net.load_dc.at[c['output_load'], 'p_dc_mw'] *= self.curtail
            elif rec['table'] == 'sgen':
                p_dc = (obj.mpp()[2] if obj.kind == 'PV Array' else obj.p_operating()) / 1e6
                p_ac = min(pe._electrisim_stage_output(p_dc, rec['eta'], rec['p_nl_mw']), rec['s_rated'])
                if obj.kind == 'PV Array':
                    p_ac *= self.curtail
                net.sgen.at[rec['index'], 'p_mw'] = p_ac

    def _set_battery(self, b, p_mw):
        """A dispatched battery's output (AC side behind a PCS, its converter's output) held to what it can give."""
        pe, net = self.pe, self.net
        it = b['item']
        obj, rec = it['obj'], it['rec']
        p_mw = max(min(p_mw, b['rated']), -b['rated'])
        if b['kind'] == 'pcs':
            p_dc = pe._electrisim_dc_dc_input_power(p_mw, rec['eta'], rec['p_nl_mw'])
            avail = available_w(obj, p_dc * 1e6, self.dt) / 1e6
            if abs(avail - p_dc) > 1e-12:
                p_mw = pe._electrisim_stage_output(avail, rec['eta'], rec['p_nl_mw'])
                self._limited(it, 'at its window or C-rate, its PCS delivers less than asked')
            net.sgen.at[rec['index'], 'p_mw'] = p_mw
        else:
            c = b['conv']
            p_dc = pe._electrisim_dc_dc_input_power(p_mw, c['eta'], c['p_nl_mw'])
            avail = available_w(obj, p_dc * 1e6, self.dt) / 1e6
            if abs(avail - p_dc) > 1e-12:
                p_mw = pe._electrisim_stage_output(avail, c['eta'], c['p_nl_mw'])
                self._limited(it, 'at its window or C-rate, its converter delivers less than asked')
            c['p_set_mw'] = p_mw
            net.load_dc.at[c['output_load'], 'p_dc_mw'] = -p_mw
        b['applied_mw'] = p_mw
        return p_mw

    def _limited(self, it, why):
        key = (id(it), why)
        if key not in self.obj_notes:
            self.obj_notes.add(key)
            self.notes.append(f"{it['obj'].kind} '{it['obj'].label}': {why}.")

    def _apply_smoothing(self, t):
        net, pe = self.net, self.pe
        for s in self.smoothing:
            c = s['conv']
            p_rack = sum(float(net.load_dc.at[i, 'p_dc_mw']) for i in self.dc_loads
                         if int(net.load_dc.at[i, 'bus_dc']) == c['bus_out'])
            alpha = 1.0 - math.exp(-self.dt / s['tau'])
            s['y'] = p_rack if s['y'] is None else s['y'] + (p_rack - s['y']) * alpha
            obj = s['store']['obj']
            soc = soc_percent(obj)
            if self.dt > LONG_STEP_TAUS * s['tau']:
                # A step far longer than its filter: the swings it smooths are gone in the step's mean,
                # and its state-of-charge loop holds its store at its set point - its converter's
                # no-load loss taken from the bus. Its proportional term over a whole step would
                # empty the store: 0.04 MW for an hour is 130 MJ, against a supercapacitor's 2.
                p_dc = (stored_energy_j(obj) - energy_at_soc_j(obj, s['soc_ref'])) / self.dt / 1e6
                if obj.kind == 'Supercapacitor' and getattr(obj, 'r_leak', 0) > 0:
                    p_dc -= obj.v0 ** 2 / obj.r_leak / 1e6        # its own leakage, made up over the step
                want = pe._electrisim_stage_output(p_dc, c['eta'], c['p_nl_mw'])
            else:
                # Above its set point it discharges a little more, below it a little less.
                want = s['share'] * (p_rack - s['y']) + s['k'] * (soc - s['soc_ref']) / 100.0 * c['rated_mw']
            want = max(min(want, c['rated_mw']), -c['rated_mw']) if c['rated_mw'] > 0 else want
            p_dc = pe._electrisim_dc_dc_input_power(want, c['eta'], c['p_nl_mw'])
            avail = available_w(obj, p_dc * 1e6, self.dt) / 1e6
            p_out = want
            if abs(avail - p_dc) > 1e-12:
                p_out = pe._electrisim_stage_output(avail, c['eta'], c['p_nl_mw'])
                s['limited'] += 1
            c['p_set_mw'] = p_out
            net.load_dc.at[c['output_load'], 'p_dc_mw'] = -p_out
            s['rows'].append({'time_step': t, 'p_rack_mw': p_rack, 'p_store_mw': p_out, 'p_lpf_mw': s['y'],
                              'soc_percent': soc})

    # --- The load flow and the dispatch ------------------------------------------------------

    def solve(self, t, run_pf):
        """The step's load flow; under the dispatch, the batteries' powers and any shedding or curtailment settled."""
        converged = run_pf()
        if not self.dispatch or not converged:
            return converged
        net = self.net
        grid = net.ext_grid.index[net.ext_grid['in_service'] == True] if len(net.ext_grid) else []
        grid = [g for g in grid if not ('electrisim_aux' in net.ext_grid.columns and net.ext_grid.at[g, 'electrisim_aux'] == True)]
        for _ in range(DISPATCH_ITERATIONS):
            if grid:
                # The batteries take what the grid supplies (or absorbs), shared by their ratings.
                exchange = sum(float(net.res_ext_grid.at[g, 'p_mw']) for g in grid)
                if abs(exchange) < 1e-7 or not self.batteries:
                    break
                total = sum(b['rated'] for b in self.batteries)
                moved = False
                for b in self.batteries:
                    before = b['applied_mw']
                    got = self._set_battery(b, before + exchange * b['rated'] / total)
                    b['set_mw'] = got
                    moved |= abs(got - before) > 1e-10
                if not moved:
                    break
            else:
                if not self._island_limits():
                    break
            converged = run_pf()
            if not converged:
                break
        return converged

    def _island_limits(self):
        """
        Islanded: the grid-forming PCS's batteries can give (or take) only so
        much this step. Over that, load is shed - the island's loads scaled
        together by what the PCS can deliver over what they deliver, which
        converges as the losses fall with the load; under it, shed load comes
        back first, then PV is curtailed. True when something changed.
        """
        net, pe = self.net, self.pe
        actual = target = 0.0
        for rec in self.pcs:
            if rec['table'] != 'gen' or not rec['in_service'] or rec['index'] not in net.res_gen.index:
                continue
            obj = rec['source']['obj']
            p_ac = float(net.res_gen.at[rec['index'], 'p_mw'])
            p_dc = pe._electrisim_dc_dc_input_power(p_ac, rec['eta'], rec['p_nl_mw'])
            avail = available_w(obj, p_dc * 1e6, self.dt) / 1e6
            actual += p_ac
            target += pe._electrisim_stage_output(avail, rec['eta'], rec['p_nl_mw'])
        tol = 1e-7
        if actual > target + tol and actual > 0:
            # Short: shed load.
            self.shed = max(self.shed * max(target, 0.0) / actual, 0.0)
        elif actual < target - tol and self.shed < 1.0 and target > 0:
            self.shed = min(self.shed * target / max(actual, 1e-9), 1.0)
        elif actual < target - tol and target <= 0:
            # Surplus they cannot take: curtail PV.
            pv = sum(self._pv_output_mw())
            if pv <= 1e-12:
                return False
            self.curtail = max(min(self.curtail * (pv - (target - actual)) / pv, 1.0), 0.0)
            self._apply_sources()
            return True
        else:
            return False
        for i, (p, q) in self.base_ac.items():
            net.load.at[i, 'p_mw'], net.load.at[i, 'q_mvar'] = p * self.shed, q * self.shed
        for i, p in self.base_dc.items():
            self._set_dc_load(i, p * self.shed)
        return True

    def _set_dc_load(self, i, p_mw):
        """
        A DC load's power, and its rated power with it: a load with a model (or a
        minimum voltage) draws its rated power's share at its voltage each load
        flow, so its profile and any shedding were undone - the racks ran at
        their full 48 MW whatever their training cycle.
        """
        ld = self.net.load_dc
        ld.at[i, 'p_dc_mw'] = p_mw
        if 'electrisim_p_rated_mw' in ld.columns and np.isfinite(float(ld.at[i, 'electrisim_p_rated_mw'])):
            ld.at[i, 'electrisim_p_rated_mw'] = p_mw

    def _pv_output_mw(self):
        net = self.net
        for it in self.items:
            if it['obj'].kind != 'PV Array':
                continue
            rec = it['rec']
            if it['via'] == 'pcs' and rec['table'] == 'sgen':
                yield float(net.sgen.at[rec['index'], 'p_mw'])
            elif it['via'] == 'dc':
                for c in rec['converters']:
                    if c['bus_in'] == rec['bus'] and c['output_load'] is not None:
                        yield -float(net.load_dc.at[c['output_load'], 'p_dc_mw'])

    # --- After the load flow --------------------------------------------------------------

    def after_step(self, t, converged):
        net, pe, dt = self.net, self.pe, self.dt
        # The diagram's loads: an SST's MV draw counted too was its hall's load twice, and, moving as the
        # load flow settled it, read as load not served on a network on the grid.
        loads = sum(float(net.load.at[i, 'p_mw']) for i in self.real_ac if bool(net.load.at[i, 'in_service']))
        loads += sum(float(net.load_dc.at[i, 'p_dc_mw']) for i in self.dc_loads if bool(net.load_dc.at[i, 'in_service']))
        base = sum(p for p, _ in self.base_ac.values()) + sum(self.base_dc.values())
        unserved = max(base - loads, 0.0)
        self.unserved_j += unserved * 1e6 * dt
        if unserved > 1e-9:
            self.unserved_steps.append(t)
        pv_now = sum(self._pv_output_mw())
        curtailed = pv_now / self.curtail - pv_now if 0 < self.curtail < 1 else 0.0
        self.curtailed_j += curtailed * 1e6 * dt
        demand = 0.0
        for it in self.items:
            obj, rec = it['obj'], it['rec']
            if not obj.in_service:
                continue
            if it['via'] == 'dc':
                if rec['bus'] not in net.res_bus_dc.index:
                    continue
                v, p = pe._electrisim_der_power(net, rec)
                if obj.kind == 'Supercapacitor' and obj.direct:
                    obj.v0 = v
            else:
                res = net[f"res_{rec['table']}"]
                if rec['index'] not in res.index:
                    continue
                p_ac = float(res.at[rec['index'], 'p_mw'])
                p = pe._electrisim_dc_dc_input_power(p_ac, rec['eta'], rec['p_nl_mw']) * 1e6
                v, _ = pe._electrisim_der_dc_point(obj, p)
                self.pcs_rows.append({'time_step': t, 'name': rec['name'], 'id': rec['id'], 'label': rec['label'],
                                      'p_mw': p_ac, 'q_mvar': float(res.at[rec['index'], 'q_mvar']),
                                      'frequency_hz': float(getattr(net, 'f_hz', 50.0) or 50.0) * (1.0 + rec['df_pu'])})
            if not (math.isfinite(p) and math.isfinite(v)):
                continue        # a step whose load flow failed: its state stays as it was
            state = {k: (float(x) if isinstance(x, (int, float)) and x is not None else x)
                     for k, x in obj.state(p / max(v, 1e-9), v).items()}
            obj.notes = []
            if obj.kind in ('Battery', 'SOFC') and it.get('target_now_w') is not None:
                state['target_kw'] = it['target_now_w'] / 1e3
            if obj.kind in ('Battery', 'Flywheel', 'SOFC'):
                demand += p / 1e6          # the dispatchable sources' share of the demand
            drawn = advance(obj, p, v, dt)
            it['drawn_j'] += drawn
            it['delivered_j'] += p * dt
            row = {'time_step': t, 'name': it['rec']['name'] if it['via'] == 'dc' else rec['source']['name'],
                   'id': rec['id'] if it['via'] == 'dc' else rec['source']['id'],
                   'label': obj.label, 'kind': obj.kind, 'p_mw': p / 1e6, 'v_kv': v / 1e3,
                   'soc_percent_end': soc_percent(obj), **state}
            self.rows.append(row)
        # Under the dispatch the SOFCs follow the demand on the dispatchable sources, averaged.
        if self.dispatch:
            grid = sum(float(net.res_ext_grid.at[g, 'p_mw']) for g in net.ext_grid.index
                       if g in net.res_ext_grid.index and bool(net.ext_grid.at[g, 'in_service'])
                       and not ('electrisim_aux' in net.ext_grid.columns and net.ext_grid.at[g, 'electrisim_aux'] == True))
            d_total = grid + demand
            sofcs = [it for it in self.items if it['obj'].kind == 'SOFC']
            if sofcs:
                cap = sum(it['obj'].p_rated for it in sofcs)
                a = 1.0 - math.exp(-dt / self.sofc_tau)
                self.demand_lpf = self.demand_lpf or {}
                for it in sofcs:
                    target = d_total * 1e6 * it['obj'].p_rated / cap
                    prev = self.demand_lpf.get(id(it), it['obj'].p_set)
                    self.demand_lpf[id(it)] = prev + (target - prev) * a
        self.balance.append({'time_step': t, 'converged': bool(converged), 'unserved_mw': unserved,
                             'curtailed_mw': curtailed, 'load_mw': loads})

    # --- Results --------------------------------------------------------------------------

    def results(self):
        dt = self.dt
        stores = []
        for it in self.items:
            obj = it['obj']
            out = {'name': it['rec']['name'] if it['via'] == 'dc' else it['rec']['source']['name'],
                   'label': obj.label, 'kind': obj.kind,
                   'delivered_mwh': it['delivered_j'] / 3.6e9, 'drawn_mwh': it['drawn_j'] / 3.6e9}
            if obj.kind in ('Battery', 'Supercapacitor', 'Flywheel'):
                e1 = stored_energy_j(obj)
                usable = usable_energy_j(obj)
                socs = [r['soc_percent_end'] for r in self.rows if r['label'] == obj.label]
                throughput = sum(abs(r['p_mw']) for r in self.rows if r['label'] == obj.label) * 1e6 * dt
                out.update(stored_start_mwh=it['e0'] / 3.6e9, stored_end_mwh=e1 / 3.6e9,
                           soc_min_percent=min(socs) if socs else None, soc_max_percent=max(socs) if socs else None,
                           equivalent_full_cycles=throughput / (2.0 * usable) if usable > 0 else None)
            stores.append(out)
        smoothing = []
        for s in self.smoothing:
            rows = s['rows']
            rack = np.array([r['p_rack_mw'] for r in rows])
            store = np.array([r['p_store_mw'] for r in rows])
            feed = rack - store
            ramp = lambda x: float(np.max(np.abs(np.diff(x))) / dt) if len(x) > 1 else 0.0
            smoothing.append({'name': s['conv']['name'], 'id': s['conv']['id'], 'label': s['conv']['label'],
                              'store': s['store']['obj'].label, 'tau_s': s['tau'],
                              'rack_peak_mw': float(rack.max()) if len(rack) else 0.0,
                              'feed_peak_mw': float(feed.max()) if len(feed) else 0.0,
                              'rack_ramp_mw_s': ramp(rack), 'feed_ramp_mw_s': ramp(feed),
                              'store_peak_mw': float(np.max(np.abs(store))) if len(store) else 0.0,
                              'limited_steps': s['limited'],
                              'series': [{'time_step': r['time_step'], 'p_rack_mw': r['p_rack_mw'],
                                          'p_feed_mw': r['p_rack_mw'] - r['p_store_mw'], 'p_store_mw': r['p_store_mw'],
                                          'soc_percent': r['soc_percent']} for r in rows]})
            if s['limited']:
                self.notes.append(f"DC/DC Converter '{s['conv']['label']}': its store's limits cut in at "
                                  f"{s['limited']} of {len(rows)} steps.")
        if self.unserved_steps:
            self.notes.append(f'The island ran short at {len(self.unserved_steps)} steps: '
                              f'{self.unserved_j / 3.6e9:.4g} MWh of load was not served.')
        return {'dispatch': self.dispatch, 'time_step_s': dt, 'ders': self.rows, 'pcs': self.pcs_rows,
                'stores': stores, 'smoothing': smoothing, 'steps': self.balance,
                'unserved_mwh': self.unserved_j / 3.6e9, 'curtailed_mwh': self.curtailed_j / 3.6e9,
                'notes': self.notes}
