"""
DC fault study: the current a pole-to-pole fault on a DC bus draws.

The DC network is simulated in time from its load-flow state: DC-link
capacitors (with ESR and ESL), cables as pi sections (R, L, C), DC sources as
their voltage behind an internal R and L, DC load input filters, DC breakers'
current-limiting inductors, and VSCs - which block at the fault, so that
their DC-link capacitors discharge into it and their diodes rectify the AC
grid into it, behind the grid's short-circuit impedance at the converter's AC
bus and the converter's own reactor. DC/DC converters block too: their input
and output capacitors discharge into a fault on their buses. The converters'
capacitors are the EMT study's (emt_converters): as given, or its default,
behind the ESR and ESL their elements give.

The results are given in IEC 61660-1's terms - peak ip, time to peak tp,
quasi-steady current Ik, and the rise and decay time constants tau1, tau2 of
its standard approximation function, fitted to the simulated current - but
are not computed by its method: they are what this network's circuit does.

Simplifications, stated with the results:
- pole-to-pole faults only; a DC bus's voltage is pole to pole and a cable's
  R, L and C per km are its loop values, as in the load flow;
- DC loads leave the network at the fault (a fault current neglects them, as
  IEC 61660-1 does), their input filters stay; a constant-power load with
  none has the EMT study's default, 4 ms of its power at its voltage
  (emt_solver.default_input_capacitance): it is converter-fed;
- a VSC's pre-fault current stops as it blocks, and its AC source is the
  pre-fault voltage of its AC bus;
- the breakers do not open: each is checked on the prospective current
  through it at its opening time;
- a discharge faster than the time step is not resolved: a cable's own
  capacitance into a hard fault, which Simscape shows as a spike of some
  20 ns at t = 0 (tests/emt_reference); or a capacitor with no ESR or
  ESL given - a converter's by default - into a hard fault on its bus,
  which the study warns of.
"""

import copy
import json
import math
import warnings

import numpy as np
import pandapower as pp
from scipy.optimize import minimize_scalar

import pandapower_electrisim as pe
from emt_converters import dcdc_capacitors, dcdc_rating, vsc_dc_link, vsc_rating
from emt_solver import Circuit, R_BIAS, R_FLOOR, R_OFF, R_ON  # noqa: F401 - the tests use them
from emt_solver import DEFAULT_INPUT_WARNING, default_input_capacitance

# The solver: emt_solver.Circuit, shared with the EMT study.


# --- IEC 61660-1's terms -------------------------------------------------------------

def _rise(t, ip, tp, tau1):
    return ip * (1.0 - np.exp(-t / tau1)) / (1.0 - math.exp(-tp / tau1))


def _decay(t, ip, tp, ik, tau2):
    alpha = ik / ip
    return ip * ((1.0 - alpha) * np.exp(-(t - tp) / tau2) + alpha)


def _fit_tau(model, t, i, lo, hi):
    """The time constant (s) that best fits ``model``, and the fit's rms error."""
    if len(t) < 3 or hi <= lo:
        return None, None
    res = minimize_scalar(lambda x: float(np.mean((model(t, math.exp(x)) - i) ** 2)),
                          bounds=(math.log(lo), math.log(hi)), method='bounded',
                          options={'xatol': 1e-6})
    return math.exp(res.x), math.sqrt(res.fun)


def iec_terms(t, i, period=None):
    """
    A fault current in IEC 61660-1's terms: ip, tp, Ik, and tau1 (rise) and
    tau2 (decay) of its approximation function fitted to the current.
    ``period``: the AC period (s), when converters feed the fault - Ik is then
    the mean over the last one, the current's ripple aside.
    """
    t = np.asarray(t, dtype=float)
    i = np.asarray(i, dtype=float)
    T = t[-1]
    last = t >= T - (period if period else 0.05 * T)
    ik = float(np.mean(i[last])) if period else float(i[-1])
    k = int(np.argmax(i))
    ip, tp = float(i[k]), float(t[k])
    # No peak of its own: the current rises to its quasi-steady value.
    monotonic = bool(last[k]) or ip <= 1.0001 * ik
    if monotonic:
        ip = max(ip, ik)
        reached = np.nonzero(i >= 0.99 * ik)[0] if ik > 0 else []
        tp = float(t[reached[0]]) if len(reached) else T
    out = {'ip': ip, 'tp': tp, 'ik': ik, 'monotonic': monotonic,
           'tau1': None, 'tau2': None, 'rise_fit_rms': None, 'decay_fit_rms': None}
    if ip <= 0 or tp <= 0:
        return out
    rise = t <= tp
    out['tau1'], out['rise_fit_rms'] = _fit_tau(lambda tt, tau: _rise(tt, ip, tp, tau),
                                                t[rise], i[rise], tp * 1e-4, tp * 1e3)
    if not monotonic and ik < ip:
        decay = t >= tp
        out['tau2'], out['decay_fit_rms'] = _fit_tau(lambda tt, tau: _decay(tt, ip, tp, ik, tau),
                                                     t[decay], i[decay], 1e-7, 10 * T)
    return out


def _settled(t, i, period):
    """Whether the current has settled by the end: last window against the one before."""
    T = t[-1]
    w = period if period else 0.1 * T
    if T < 2 * w:
        return True
    a = np.mean(i[(t >= T - w)])
    b = np.mean(i[(t >= T - 2 * w) & (t < T - w)])
    return bool(abs(a - b) <= 0.02 * max(abs(a), 1e-9))


def _waveform(t, i, k_peak=None, t_fine=None, n_fine=300, n_rest=200):
    """A plotting copy: denser in the first milliseconds, with the peak kept."""
    T = t[-1]
    t_fine = min(T, t_fine or 0.01)
    marks = np.concatenate([np.linspace(0, t_fine, n_fine), np.linspace(t_fine, T, n_rest)])
    idx = set(np.searchsorted(t, marks).clip(0, len(t) - 1).tolist())
    if k_peak is not None:
        idx.add(int(k_peak))
    idx = sorted(idx)
    return {'t_ms': [round(float(t[j]) * 1e3, 6) for j in idx],
            'i_ka': [round(float(i[j]) * 1e-3, 6) for j in idx]}


# --- From the network to the circuit -------------------------------------------------

def _f(value, default=0.0):
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _label(net, table, idx):
    name = net[table].at[idx, 'name'] if 'name' in net[table].columns else None
    names = getattr(net, 'user_friendly_names', {}) or {}
    return names.get(name, name if name not in (None, '') else f'{table} {idx}')


def _row_id(net, table, idx):
    """Its diagram cell's id: none for an element of the model's own (NaN in the id column)."""
    if 'id' not in net[table].columns:
        return str(idx)
    value = net[table].at[idx, 'id']
    return '' if value is None or (isinstance(value, float) and not math.isfinite(value)) else value


def _ac_thevenin(net, warnings_out):
    """Short-circuit R and X (ohm) at each VSC's AC bus, from pandapower's AC short circuit."""
    try:
        import pandapower.shortcircuit as sc
        ac = copy.deepcopy(net)
        for table in ('vsc', 'b2b_vsc', 'line_dc', 'load_dc', 'source_dc', 'bus_dc'):
            if table in ac and len(ac[table]):
                ac[table].drop(ac[table].index, inplace=True)
        # A grid-forming PCS is a generator in the load flow, with no machine data: a NaN in
        # the Ybus, and every converter's AC source taken as stiff. In a fault it is a current
        # source, which leaves the impedance seen at a bus as it is.
        if len(ac.gen) and 'electrisim_pcs' in ac.gen.columns:
            ac.gen.drop(ac.gen.index[ac.gen['electrisim_pcs'] == True], inplace=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sc.calc_sc(ac, case='max', branch_results=False)
        return {int(b): (float(ac.res_bus_sc.at[b, 'rk_ohm']), float(ac.res_bus_sc.at[b, 'xk_ohm']))
                for b in ac.res_bus_sc.index}
    except Exception as exc:   # noqa: BLE001 - reported, the study goes on
        warnings_out.append(f"The AC short circuit at the converters' AC buses did not solve ({exc}): "
                            "each converter's AC source is taken as stiff behind its own reactor.")
        return {}


class _Builder:
    """The circuit of a DC network in its load-flow state, and what each branch is."""

    def __init__(self, net, params, warnings_out):
        self.net, self.params, self.warnings = net, params, warnings_out
        self.ckt = Circuit()
        self.f_hz = float(getattr(net, 'f_hz', 50.0) or 50.0)
        self.bus_node, self.v_bus = {}, {}
        self.contributions = []   # (kind, label, id, ('rl'|'c', index) or ('diff', port, port))
        self.breakers = []        # (record, rl index)
        self.cable_caps = []
        self.has_converter = False
        self.isolated = set()     # DC buses nothing in the fault circuit reaches
        self._terminals = {}      # (table, index, bus) -> its node beyond its breaker
        self.stiff_caps = []      # (bus, R to it, C, label): capacitors with no ESR or ESL, on a bus through R alone
        self.diodes = []          # (record, its RL branch): the DC diodes

    def _v(self, bus):
        return self.v_bus[bus]

    def build(self):
        net, ckt = self.net, self.ckt
        res = net.res_bus_dc
        for b in net.bus_dc.index:
            if not bool(net.bus_dc.at[b, 'in_service']) or b not in res.index or not np.isfinite(res.at[b, 'vm_pu']):
                continue
            if 'electrisim_aux' in net.bus_dc.columns and net.bus_dc.at[b, 'electrisim_aux'] == True:
                continue   # a solid-state transformer's internal DC link, inside its blocked stages
            if pe._electrisim_is_hidden(net.bus_dc, b):
                continue   # a battery's cells: modelled with it
            self.bus_node[b] = ckt.node(_label(net, 'bus_dc', b))
            self.v_bus[b] = float(res.at[b, 'vm_pu']) * float(net.bus_dc.at[b, 'vn_kv']) * 1e3
        breakers = {}
        for rec in getattr(net, 'electrisim_dc_breakers', None) or []:
            if rec['closed'] and rec['target'] is not None:
                breakers.setdefault(rec['target'], []).append(rec)
        self._breakers = breakers
        self._cables()
        self._capacitors()
        self._sources()
        self._loads()
        self._converters()
        self._dc_dc_capacitors()
        self._ders()
        self._tie_isolated()
        if len(net.b2b_vsc) if 'b2b_vsc' in net else False:
            self.warnings.append('The B2B VSCs are left out of the DC fault study: their two-pole DC side is not modelled yet.')
        return self

    def _terminal(self, table, idx, bus, i0):
        """The node an element connects at: its bus, or beyond a breaker in front of it (built once)."""
        recs = [r for r in self._breakers.get((table, idx), []) if r['bus_dc'] == bus]
        if not recs:
            return self.bus_node[bus], 0.0
        if (table, idx, bus) in self._terminals:
            return self._terminals[(table, idx, bus)], 0.0
        rec = recs[0]
        node = self.ckt.node(f"{rec['label']} terminal")
        k = self.ckt.add_rl(self.bus_node[bus], node, 0.0, rec['limiting_inductance_mh'] * 1e-3, i0=i0)
        self.breakers.append((rec, k))
        self._terminals[(table, idx, bus)] = node
        return node, 0.0

    def _cap(self, node, c, r, l, v, label, bus=None, r_to_bus=0.0):
        """
        A capacitor charged to ``v`` at ``node``, behind its ESR and ESL if it
        has them: the port of its current. ``bus``: the DC bus ``node`` is on
        through ``r_to_bus`` alone, when it is.
        """
        ckt = self.ckt
        if r > 0 or l > 0:
            inner = ckt.node(f'{label} capacitor')
            ckt.add_c(0, inner, c, w0=-v)
            return ('rl', ckt.add_rl(inner, node, r, l))
        if bus is not None:
            self.stiff_caps.append((bus, r_to_bus, c, label))
        return ('c', ckt.add_c(0, node, c, w0=-v))

    def _diode(self, li):
        """
        A DC diode: its forward voltage behind its on-resistance, conducting
        or blocking as the fault drives it. One blocking in the load flow is
        built too: a catcher's diode starts to conduct when the fault takes
        its shelf's own bus.
        """
        net, ckt = self.net, self.ckt
        rec = next((d for d in getattr(net, 'electrisim_dc_diodes', None) or [] if d.get('line') == li), None)
        fb, tb = int(net.line_dc.at[li, 'from_bus_dc']), int(net.line_dc.at[li, 'to_bus_dc'])
        if rec is None or not rec['in_service'] or fb not in self.bus_node or tb not in self.bus_node:
            return
        i0 = pe._electrisim_dc_diode_amps(net, rec)
        i0 = i0 if np.isfinite(i0) and i0 > 0 else 0.0
        # Through the breakers in front of it, as a cable is: a shelf feeder's breaker, then its diode.
        anode, _ = self._terminal('line_dc', li, fb, i0)
        cathode, _ = self._terminal('line_dc', li, tb, -i0)
        junction = ckt.node(f"{rec['label']} junction", self._v(tb) + i0 * rec['r_on_ohm'])
        ckt.add_diode(anode, junction, rec['v_f_v'])
        self.diodes.append((rec, ckt.add_rl(junction, cathode, rec['r_on_ohm'], 0.0, i0=i0)))

    def _cables(self):
        net, ckt = self.net, self.ckt
        for li in net.line_dc.index:
            if pe._electrisim_is_dc_diode(net, li):
                self._diode(li)
                continue
            if not bool(net.line_dc.at[li, 'in_service']):
                continue
            fb, tb = int(net.line_dc.at[li, 'from_bus_dc']), int(net.line_dc.at[li, 'to_bus_dc'])
            if fb not in self.bus_node or tb not in self.bus_node:
                continue
            i_from = _f(net.res_line_dc.at[li, 'i_from_ka']) * 1e3 if li in net.res_line_dc.index else 0.0
            i_to = _f(net.res_line_dc.at[li, 'i_to_ka']) * 1e3 if li in net.res_line_dc.index else -i_from
            if 'electrisim_dc_breaker' in net.line_dc.columns and net.line_dc.at[li, 'electrisim_dc_breaker'] == True:
                rec = next((r for recs in self._breakers.values() for r in recs if r['target'] == ('line_dc', li)), None)
                lim = rec['limiting_inductance_mh'] * 1e-3 if rec else 0.0
                k = ckt.add_rl(self.bus_node[fb], self.bus_node[tb], 0.0, lim, i0=i_from)
                if rec:
                    self.breakers.append((rec, k))
                continue
            km = _f(net.line_dc.at[li, 'length_km'], 1.0)
            r = _f(net.line_dc.at[li, 'r_ohm_per_km']) * km
            l = _f(net.line_dc.at[li, 'l_mh_per_km']) * 1e-3 * km if 'l_mh_per_km' in net.line_dc.columns else 0.0
            c = _f(net.line_dc.at[li, 'c_uf_per_km']) * 1e-6 * km if 'c_uf_per_km' in net.line_dc.columns else 0.0
            end_f, _ = self._terminal('line_dc', li, fb, i_from)
            end_t, _ = self._terminal('line_dc', li, tb, i_to)
            ckt.add_rl(end_f, end_t, r, l, i0=i_from)
            if c > 0:
                for end, bus in ((end_f, fb), (end_t, tb)):
                    self.cable_caps.append(ckt.add_c(0, end, c / 2.0, w0=-self._v(bus)))

    def _capacitors(self):
        for cap in getattr(self.net, 'electrisim_dc_capacitors', None) or []:
            if cap.get('electrisim_der'):
                continue   # a supercapacitor: with the sources and stores
            bus = cap['bus_dc']
            if not cap.get('in_service', True) or bus not in self.bus_node or cap['c_mf'] <= 0:
                continue
            label = (getattr(self.net, 'user_friendly_names', {}) or {}).get(cap['name'], cap['name'])
            port = self._cap(self.bus_node[bus], cap['c_mf'] * 1e-3, cap['esr_mohm'] * 1e-3, cap['esl_uh'] * 1e-6,
                             self._v(bus), label, bus)
            self.contributions.append(('DC capacitor', label, cap.get('id', ''), port))

    def _sources(self):
        net, ckt = self.net, self.ckt
        if 'source_dc' not in net:
            return
        for si in net.source_dc.index:
            bus = int(net.source_dc.at[si, 'bus_dc'])
            if not bool(net.source_dc.at[si, 'in_service']) or bus not in self.bus_node:
                continue
            label = _label(net, 'source_dc', si)
            # Injected: from what its bus draws (pandapower's res_source_dc does not give it).
            i0 = pe._electrisim_source_dc_currents_ka(net).get(si, 0.0) * 1e3
            r = _f(net.source_dc.at[si, 'electrisim_r_sc_mohm']) * 1e-3 if 'electrisim_r_sc_mohm' in net.source_dc.columns else 0.0
            l = _f(net.source_dc.at[si, 'electrisim_l_sc_uh']) * 1e-6 if 'electrisim_l_sc_uh' in net.source_dc.columns else 0.0
            if r <= 0 and l <= 0:
                self.warnings.append(f"Source DC '{label}' has no internal resistance or inductance: a fault near "
                                     "it draws a current only the network limits. Enter them on its Short Circuit tab.")
            term, _ = self._terminal('source_dc', si, bus, -i0)   # from the bus into the source
            e = self._v(bus) + r * i0
            k = ckt.add_rl(0, term, r, l, i0=i0, e0=e)
            self.contributions.append(('DC source', label, _row_id(net, 'source_dc', si), ('rl', k)))

    def _loads(self):
        net, ckt = self.net, self.ckt
        ld = net.load_dc
        # A converter stage's input is its converter's (its capacitors are _dc_dc_capacitors'), but
        # for an SST's grid-forming inverter, which the EMT study too keeps as a load.
        kept_stages = {sst_stage['input'][1] for rec in getattr(net, 'electrisim_ssts', None) or []
                       if rec.get('inverter_mode') == 'grid_forming'
                       for sst_stage in rec['stages'] if sst_stage['stage'] == 'inverter'}
        defaulted = []
        for li in ld.index:
            bus = int(ld.at[li, 'bus_dc'])
            if not bool(ld.at[li, 'in_service']) or bus not in self.bus_node:
                continue
            label = _label(net, 'load_dc', li)
            p = _f(net.res_load_dc.at[li, 'p_dc_mw']) if li in net.res_load_dc.index else 0.0
            c = _f(ld.at[li, 'filter_c_uf']) * 1e-6 if 'filter_c_uf' in ld.columns else 0.0
            aux = 'electrisim_aux' in ld.columns and ld.at[li, 'electrisim_aux'] == True
            if c <= 0 and (not aux or li in kept_stages):
                # A constant-power load is converter-fed: its DC link, as the EMT study gives it.
                share_p = _f(ld.at[li, 'electrisim_share_p'], 1.0) if 'electrisim_share_p' in ld.columns and not aux else 1.0
                c = default_input_capacitance(share_p, p * 1e6, self._v(bus))
                if c > 0:
                    defaulted.append(label)
            if c <= 0:
                continue   # it leaves the network at the fault, with nothing to discharge
            l = _f(ld.at[li, 'filter_l_mh']) * 1e-3 if 'filter_l_mh' in ld.columns else 0.0
            r = max(_f(ld.at[li, 'filter_r_mohm']), 0.0) * 1e-3 if 'filter_r_mohm' in ld.columns else 0.0
            i_load = p * 1e6 / self._v(bus)
            term, _ = self._terminal('load_dc', li, bus, i_load)
            inner = ckt.node(f'{label} filter')
            # Its capacitor at the drop the load's current leaves across the filter's R, as in EMT
            # (with no L to carry it on, that current stops at the fault).
            ckt.add_c(0, inner, c, w0=-(self._v(bus) - (r * i_load if l > 0 else 0.0)))
            # The filter inductor keeps the load's current at the fault, into its capacitor.
            k = ckt.add_rl(inner, term, r, l, i0=-i_load if l > 0 else 0.0)
            if l <= 0 and term == self.bus_node[bus]:
                self.stiff_caps.append((bus, max(r, R_FLOOR), c, f'{label} input capacitor'))
            self.contributions.append(('DC load input filter', label, _row_id(net, 'load_dc', li), ('rl', k)))
        if defaulted:
            self.warnings.append(DEFAULT_INPUT_WARNING.format(', '.join(defaulted)))

    def _converters(self):
        net, ckt = self.net, self.ckt
        if 'vsc' not in net or not len(net.vsc):
            return
        thevenin = None
        w = 2.0 * math.pi * self.f_hz
        for vi in net.vsc.index:
            bus_dc = int(net.vsc.at[vi, 'bus_dc'])
            if not bool(net.vsc.at[vi, 'in_service']) or bus_dc not in self.bus_node:
                continue
            if 'electrisim_aux' in net.vsc.columns and net.vsc.at[vi, 'electrisim_aux'] == True:
                # A DC/DC converter's output stage: it blocks; its capacitors are _dc_dc_capacitors'.
                note = ('The DC/DC converters block at a DC fault: only their input and output capacitors '
                        'discharge into it. Their current limits come with the EMT study.')
                if note not in self.warnings:
                    self.warnings.append(note)
                continue
            if thevenin is None:
                thevenin = _ac_thevenin(net, self.warnings)
            label = _label(net, 'vsc', vi)
            ac_bus = int(net.vsc.at[vi, 'bus'])
            rk, xk = thevenin.get(ac_bus, (0.0, 0.0))
            r_ac = rk + _f(net.vsc.at[vi, 'r_ohm'])
            x_ac = xk + _f(net.vsc.at[vi, 'x_ohm'])
            vm = _f(net.res_bus.at[ac_bus, 'vm_pu'], 1.0) if ac_bus in net.res_bus.index else 1.0
            v_ll = vm * float(net.bus.at[ac_bus, 'vn_kv']) * 1e3
            amp = math.sqrt(2.0 / 3.0) * v_ll
            v_dc = self._v(bus_dc)
            if v_dc < math.sqrt(2.0) * v_ll:
                self.warnings.append(f"VSC '{label}': its DC voltage ({v_dc / 1e3:.3g} kV) is below the peak of its AC "
                                     f"line voltage ({math.sqrt(2) * v_ll / 1e3:.3g} kV), so its diodes conduct before the fault.")
            phase0 = math.radians(_f(self.params.get('fault_angle_deg'), 0.0))
            p_node = ckt.node(f'{label} DC+')
            neutral = ckt.node(f'{label} AC neutral')
            ckt.add_r(neutral, p_node, R_BIAS)
            ckt.add_r(neutral, 0, R_BIAS)
            for k_ph in range(3):
                x = ckt.node(f'{label} phase {"abc"[k_ph]}')
                ckt.add_rl(neutral, x, r_ac, x_ac / w, ac=(amp, w, phase0 - k_ph * 2.0 * math.pi / 3.0))
                ckt.add_diode(x, p_node)
                ckt.add_diode(0, x)
            term, _ = self._terminal('vsc', vi, bus_dc, 0.0)
            col = lambda c: _f(net.vsc.at[vi, c]) if c in net.vsc.columns else 0.0
            # Its DC-link capacitor on its bridge's DC rail, its ESR and ESL with its DC resistance in
            # its DC terminals' branch, as the EMT study has them.
            r_out = _f(net.vsc.at[vi, 'r_dc_ohm']) + max(col('dc_link_esr_mohm'), 0.0) * 1e-3
            l_out = max(col('dc_link_esl_uh'), 0.0) * 1e-6
            k = ckt.add_rl(p_node, term, r_out, l_out)
            p, q = ((_f(net.res_vsc.at[vi, 'p_mw']), _f(net.res_vsc.at[vi, 'q_mvar']))
                    if vi in net.res_vsc.index else (0.0, 0.0))
            c_link = vsc_dc_link(col('dc_link_mf'), vsc_rating(col('rated_mva'), p, q),
                                 float(net.bus_dc.at[bus_dc, 'vn_kv']) * 1e3)
            stiff = term == self.bus_node[bus_dc] and l_out <= 0
            cap = self._cap(p_node, c_link, 0.0, 0.0, v_dc, f'{label} DC link', bus_dc if stiff else None,
                            max(r_out, R_FLOOR))
            row_id = _row_id(net, 'vsc', vi)
            self.contributions.append(('VSC (diodes, blocked)', label, row_id, ('diff', ('rl', k), cap)))
            self.contributions.append(('VSC DC-link capacitor', label, row_id, cap))
            self.has_converter = True

    def _dc_dc_capacitors(self):
        """
        Each DC/DC converter's input and output capacitors on its buses (an
        SST's DC/DC stage's output one: its input is on the SST's internal DC
        link), and an SST's grid-following inverter's DC link on its LV DC
        bus, as the EMT study has them: their bridges block at the fault, and
        they discharge into it. A source's own bus behind its converter - a
        battery's, a PV array's - is so reached, and faulted.
        """
        net = self.net

        def vn(bus):
            return float(net.bus_dc.at[bus, 'vn_kv']) * 1e3

        def add(kind, label, cid, table, idx, bus, c, esr_mohm=0.0, esl_uh=0.0):
            if bus in self.bus_node and c > 0:
                term, _ = self._terminal(table, idx, bus, 0.0)
                name = f"{label} {kind.split(' ', 1)[1]}"     # e.g. 'Shelf output capacitor'
                port = self._cap(term, c, max(esr_mohm, 0.0) * 1e-3, max(esl_uh, 0.0) * 1e-6, self._v(bus), name,
                                 bus if term == self.bus_node[bus] else None)
                self.contributions.append((kind, label, cid, port))

        for rec in getattr(net, 'electrisim_dc_dc_converters', None) or []:
            if not rec['in_service'] or rec['input'] is None or rec['input'] not in net.load_dc.index \
                    or not bool(net.load_dc.at[rec['input'], 'in_service']):
                continue
            if rec['vsc'] is not None:
                p_out = -_f(net.res_vsc.at[rec['vsc'], 'p_dc_mw']) if rec['vsc'] in net.res_vsc.index else 0.0
                out = ('vsc', rec['vsc'])
            else:
                p_out, out = rec['p_set_mw'], ('load_dc', rec['output_load'])
            b_in, b_out = rec['bus_in'], rec['bus_out']
            e = rec.get('emt') or {}
            c_in, c_out = dcdc_capacitors(e.get('c_out_mf', 0.0), dcdc_rating(rec['rated_mw'], p_out), vn(b_in), vn(b_out))
            add('DC/DC input capacitor', rec['label'], rec['id'], 'load_dc', rec['input'], b_in, c_in,
                e.get('c_in_esr_mohm', 0.0), e.get('c_in_esl_uh', 0.0))
            add('DC/DC output capacitor', rec['label'], rec['id'], out[0], out[1], b_out, c_out,
                e.get('c_out_esr_mohm', 0.0), e.get('c_out_esl_uh', 0.0))
        for rec in getattr(net, 'electrisim_ssts', None) or []:
            stages = {s['stage']: s for s in rec['stages']}
            rect, dcdc, inv = stages.get('rectifier'), stages.get('dcdc'), stages.get('inverter')
            lv = rec['bus_lvdc']
            if not rec['in_service'] or rect is None or dcdc is None or lv not in self.bus_node:
                continue
            d_vsc = dcdc['output'][1]
            p_out = -_f(net.res_vsc.at[d_vsc, 'p_dc_mw']) if d_vsc in net.res_vsc.index else 0.0
            _, c_out = dcdc_capacitors(0.0, dcdc_rating(dcdc['rated_mw'], p_out), vn(rect['link']), vn(lv))
            e = rec.get('emt') or {}
            add('DC/DC output capacitor', f"{rec['label']} DC/DC", rec['id'], 'vsc', d_vsc, lv, c_out,
                e.get('dcdc_c_out_esr_mohm', 0.0), e.get('dcdc_c_out_esl_uh', 0.0))
            lvac = rec['bus_lvac']
            if inv is None or rec['inverter_mode'] != 'grid_following' or lvac is None \
                    or lvac not in net.res_bus.index or not np.isfinite(net.res_bus.at[lvac, 'vm_pu']):
                continue
            sgen = inv['output'][1]
            p_s, q_s = ((_f(net.res_sgen.at[sgen, 'p_mw']), _f(net.res_sgen.at[sgen, 'q_mvar']))
                        if sgen in net.res_sgen.index else (0.0, 0.0))
            c_link = vsc_dc_link(0.0, vsc_rating(inv['rated_mw'], p_s, q_s), vn(lv))
            add('VSC DC-link capacitor', f"{rec['label']} inverter", rec['id'], 'load_dc', inv['input'][1], lv, c_link,
                e.get('inv_dc_link_esr_mohm', 0.0), e.get('inv_dc_link_esl_uh', 0.0))


    def _tie_isolated(self):
        """
        A DC bus nothing in the fault circuit reaches - a PV array's or SOFC's
        own bus behind its blocked converter, a converter's output bus with
        only loads on it - tied to the pole through a large resistance, so
        its node is defined; a fault there draws nothing, and it is not faulted.
        """
        ckt = self.ckt
        used = set()
        for branches in (ckt.rl, ckt.c, ckt.r, ckt.d, ckt.sw, ckt.arr, ckt.nl, ckt.isrc):
            for br in branches:
                used.update((int(br[0]), int(br[1])))
        for grp in ckt.cp:
            used.update(int(n) for n in grp[0])
            used.update(int(n) for n in grp[1])
        for bus, node in self.bus_node.items():
            if node not in used:
                ckt.add_r(node, 0, R_BIAS)
                self.isolated.add(bus)

    def _ders(self):
        """
        Batteries and supercapacitors on their buses, directly or behind their
        converters, from their load-flow state: a battery its open-circuit
        voltage behind R0 (its RC branch, tens of seconds, holding its voltage
        over the fault), a supercapacitor its capacitance behind its ESR, each
        with its series inductance. A converter between one and the network
        blocks, so it feeds a fault on its own bus only. PV arrays, SOFC
        systems and flywheels feed a fault through converters that limit it,
        and are left out.
        """
        net, ckt = self.net, self.ckt
        left_out = []
        for rec in getattr(net, 'electrisim_ders', None) or []:
            obj, bus = rec['obj'], rec['bus']
            if not obj.in_service or bus not in self.bus_node:
                continue
            if rec['kind'] not in ('Battery', 'Supercapacitor'):
                left_out.append(rec['label'])
                continue
            v, p = pe._electrisim_der_power(net, rec)
            i0 = p / max(v, 1e-6)
            if rec['kind'] == 'Battery':
                # Over a fault's milliseconds its RC branch (tens of seconds) holds its voltage: its
                # open-circuit voltage less that, behind R0.
                k = ckt.add_rl(0, self.bus_node[bus], obj.r0, obj.l, i0=i0, e0=v + obj.r0 * i0)
            else:
                inner = ckt.node(f"{rec['label']} capacitor")
                ckt.add_c(0, inner, obj.c, w0=-(v + obj.esr * i0))
                k = ckt.add_rl(inner, self.bus_node[bus], obj.esr, obj.esl, i0=i0)
            self.contributions.append((rec['kind'], rec['label'], rec['id'], ('rl', k)))
        if left_out:
            self.warnings.append(f"{', '.join(left_out)}: PV arrays, SOFC systems and flywheels feed a DC fault "
                                 "through converters that limit their current, and are left out of this study.")


def _current(sim, port):
    if port[0] == 'diff':
        return _current(sim, port[1]) - _current(sim, port[2])
    kind, k = port
    return sim['i_rl'][:, k] if kind == 'rl' else sim['i_c'][:, k]


# --- The study -----------------------------------------------------------------------

def _round(x, nd=6):
    return None if x is None or not math.isfinite(x) else round(float(x), nd)


def _terms_out(terms, scale=1e-3):
    return {
        'ip_ka': _round(terms['ip'] * scale), 'tp_ms': _round(terms['tp'] * 1e3),
        'ik_ka': _round(terms['ik'] * scale), 'monotonic': terms['monotonic'],
        'tau1_ms': _round(terms['tau1'] * 1e3) if terms['tau1'] else None,
        'tau2_ms': _round(terms['tau2'] * 1e3) if terms['tau2'] else None,
        'rise_fit_rms_percent': _round(100 * terms['rise_fit_rms'] / terms['ip'], 3)
        if terms['rise_fit_rms'] is not None and terms['ip'] > 0 else None,
        'decay_fit_rms_percent': _round(100 * terms['decay_fit_rms'] / terms['ip'], 3)
        if terms['decay_fit_rms'] is not None and terms['ip'] > 0 else None,
    }


def _fault_r(params):
    return max(_f(params.get('fault_resistance_mohm'), 0.0) * 1e-3, R_FLOOR)


def _time_step(params):
    return max(_f(params.get('time_step_us'), 1.0), 0.01) * 1e-6


def _unresolved(builder, bus, params):
    """
    The capacitors with no ESR or ESL that a fault on ``bus`` discharges
    within ten time steps, with their time constants: their peak, V / R at
    once, the time step does not resolve.
    """
    r_f, dt = _fault_r(params), _time_step(params)
    return [(label, (r + r_f) * c) for b, r, c, label in builder.stiff_caps if b == bus and (r + r_f) * c < 10 * dt]


def fault_at(builder, bus, params):
    """Simulate a pole-to-pole fault on DC bus ``bus``; its currents in IEC terms."""
    ckt = copy.deepcopy(builder.ckt)
    r_f = _fault_r(params)
    k_f = ckt.add_r(builder.bus_node[bus], 0, r_f)
    dt = _time_step(params)
    t_end = max(_f(params.get('duration_ms'), 200.0), 0.1) * 1e-3
    if builder.has_converter:
        # Ik is the mean over the last AC period: three of them at least.
        t_end = max(t_end, 3.0 / builder.f_hz)
    t_fine = min(t_end, 0.01)
    sim = ckt.simulate(t_end, dt, dt_coarse=min(10 * dt, 2e-5), t_fine=t_fine)
    t = sim['t']
    period = 1.0 / builder.f_hz if builder.has_converter else None
    i_f = sim['i_r'][:, k_f]
    terms = iec_terms(t, i_f, period)
    k_peak = int(np.argmax(i_f))
    out = {
        **_terms_out(terms),
        'didt_ka_per_ms': _round(float(np.max(np.diff(i_f[t <= min(t_end, 1e-4)])) /
                                       dt * 1e-6) if len(t[t <= 1e-4]) > 1 else None),
        'settled': _settled(t, i_f, period),
        'waveform': _waveform(t, i_f, k_peak, t_fine),
        'contributions': [],
        'breakers': [],
        'diodes': [],
    }
    for kind, label, cid, port in builder.contributions:
        i = _current(sim, port)
        c_terms = iec_terms(t, i, period)
        out['contributions'].append({
            'kind': kind, 'name': label, 'id': cid,
            'ip_ka': _round(c_terms['ip'] * 1e-3), 'tp_ms': _round(c_terms['tp'] * 1e3),
            'ik_ka': _round(c_terms['ik'] * 1e-3),
            'at_peak_ka': _round(float(i[k_peak]) * 1e-3),
            'waveform': _waveform(t, i, None, t_fine, 150, 100),
        })
    if builder.cable_caps:
        i = sum(sim['i_c'][:, k] for k in builder.cable_caps)
        c_terms = iec_terms(t, i, period)
        out['contributions'].append({
            'kind': 'Cable capacitance', 'name': 'DC cables', 'id': '',
            'ip_ka': _round(c_terms['ip'] * 1e-3), 'tp_ms': _round(c_terms['tp'] * 1e3),
            'ik_ka': _round(c_terms['ik'] * 1e-3), 'at_peak_ka': _round(float(i[k_peak]) * 1e-3),
            'waveform': _waveform(t, i, None, t_fine, 150, 100),
        })
    for rec, k in builder.breakers:
        i = np.abs(sim['i_rl'][:, k])
        t_open = rec['opening_time_ms'] * 1e-3
        i_open = float(np.interp(t_open, t, i)) if t_open <= t_end else float('nan')
        cap = rec['breaking_capacity_ka'] * 1e3
        over = np.nonzero(i > cap)[0] if cap > 0 else []
        out['breakers'].append({
            'name': rec['name'], 'id': rec['id'], 'label': rec['label'],
            'i_open_ka': _round(i_open * 1e-3), 'opening_time_ms': rec['opening_time_ms'],
            'ip_ka': _round(float(np.max(i)) * 1e-3), 'tp_ms': _round(float(t[int(np.argmax(i))]) * 1e3),
            'breaking_capacity_ka': rec['breaking_capacity_ka'],
            'exceeds': bool(cap > 0 and np.isfinite(i_open) and i_open > cap),
            't_reaches_capacity_ms': _round(float(t[over[0]]) * 1e3) if len(over) else None,
        })
    # Each diode's forward current: conducting before the fault, blocking backfeed into it,
    # or starting to conduct as its shelf's other supply fails.
    for rec, k in builder.diodes:
        i = sim['i_rl'][:, k]
        out['diodes'].append({
            'name': rec['name'], 'id': rec['id'], 'label': rec['label'],
            'i_prefault_ka': _round(float(i[0]) * 1e-3), 'ip_ka': _round(float(np.max(i)) * 1e-3),
            'tp_ms': _round(float(t[int(np.argmax(i))]) * 1e3), 'i_end_ka': _round(float(i[-1]) * 1e-3),
            'conducting_end': bool(i[-1] > 1e-3),
        })
    # A closed breaker in front of what feeds no fault current - a DC load without an
    # input filter, which leaves the network at the fault - carries none.
    modelled = {rec['name'] for rec, _ in builder.breakers}
    for rec in getattr(builder.net, 'electrisim_dc_breakers', None) or []:
        if rec['closed'] and rec['name'] not in modelled and rec['bus_dc'] in builder.bus_node:
            out['breakers'].append({
                'name': rec['name'], 'id': rec['id'], 'label': rec['label'], 'i_open_ka': 0.0,
                'opening_time_ms': rec['opening_time_ms'], 'ip_ka': 0.0, 'tp_ms': None,
                'breaking_capacity_ka': rec['breaking_capacity_ka'], 'exceeds': False,
                't_reaches_capacity_ms': None,
            })
    return out


METHOD = ("A time-domain simulation of the DC network from its load-flow state, reported in IEC 61660-1's "
          "terms (ip, tp, Ik, tau1, tau2) but not computed by its method. Pole-to-pole faults; DC loads leave "
          "at the fault, their input capacitors discharging (a constant-power load's, given none, 4 ms of its "
          "power, as in the EMT study); VSCs block, their DC-link capacitors discharging and their diodes feeding the fault "
          "from the AC grid; DC/DC converters block, their input and output capacitors discharging; breakers "
          "do not open, each is checked on the current through it at its opening time. A discharge faster than "
          "the time step - a cable's own capacitance into a hard fault, over nanoseconds, or a capacitor's with "
          "no ESR on the faulted bus, warned of - is not resolved.")


def dc_fault_study(net, params, in_data=None):
    params = params or {}
    warnings_out = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pe._electrisim_runpp(net, algorithm='nr', calculate_voltage_angles=True, init='auto')
    except Exception as exc:   # noqa: BLE001
        return json.dumps({'error': True, 'message': f'The load flow before the fault did not solve: {exc}'})
    warnings_out.extend(getattr(net, 'warnings', []) or [])
    if 'bus_dc' not in net or not len(net.bus_dc):
        return json.dumps({'error': True, 'message': 'The network has no DC bus to fault.',
                           'warnings': warnings_out})
    builder = _Builder(net, params, warnings_out).build()
    if not builder.bus_node:
        return json.dumps({'error': True, 'message': 'No DC bus has a load-flow result to start the fault from.',
                           'warnings': warnings_out})
    chosen = str(params.get('fault_bus') or 'all')
    buses = [b for b in builder.bus_node
             if (chosen in ('all', '') and b not in builder.isolated)
             or chosen in (str(net.bus_dc.at[b, 'name']), str(_row_id(net, 'bus_dc', b)))]
    if not buses:
        return json.dumps({'error': True, 'message': f'No DC bus {chosen} to fault.', 'warnings': warnings_out})
    duration = _f(params.get('duration_ms'), 200.0)
    if builder.has_converter and duration * 1e-3 < 3.0 / builder.f_hz:
        warnings_out.append(f"The run is {3e3 / builder.f_hz:.4g} ms rather than {duration:g} ms: with converters "
                            "feeding the fault, Ik is the mean over the last of three AC periods at least.")
    faults = []
    for b in buses:
        result = fault_at(builder, b, params)
        result.update({'bus': str(net.bus_dc.at[b, 'name']), 'id': _row_id(net, 'bus_dc', b),
                       'label': _label(net, 'bus_dc', b), 'vn_kv': float(net.bus_dc.at[b, 'vn_kv']),
                       'v_prefault_kv': _round(builder.v_bus[b] / 1e3)})
        if not result['settled']:
            warnings_out.append(f"A fault on {result['label']}: the current had not settled by the end of the run, "
                                "so Ik is the last value - lengthen the duration.")
        fast = _unresolved(builder, b, params)
        if fast:
            tau = min(t for _, t in fast)
            warnings_out.append(f"A fault on {result['label']}: {', '.join(n for n, _ in fast)}, with no ESR, "
                                f"discharge{'s' if len(fast) == 1 else ''} into it within ten time steps (time constant "
                                f"{tau * 1e6:.2g} us), so ip is the time step's, not the circuit's: enter its ESR "
                                "and ESL, a fault resistance, or a shorter time step.")
        faults.append(result)
    # Each breaker's worst case over the faults.
    worst = {}
    for fault in faults:
        for brk in fault['breakers']:
            prev = worst.get(brk['name'])
            if prev is None or (brk['i_open_ka'] or 0) > (prev['i_open_ka'] or 0):
                worst[brk['name']] = {**brk, 'fault_bus': fault['label']}
    for brk in worst.values():
        if brk['exceeds']:
            warnings_out.append(f"DC breaker {brk['label']}: a fault on {brk['fault_bus']} drives "
                                f"{brk['i_open_ka']:.3g} kA through it at its opening time, above its "
                                f"{brk['breaking_capacity_ka']:g} kA breaking capacity.")
    return json.dumps({
        'dcfault': {
            'method': METHOD,
            'settings': {'fault_resistance_mohm': _f(params.get('fault_resistance_mohm'), 0.0),
                         'time_step_us': _f(params.get('time_step_us'), 1.0),
                         'duration_ms': _f(params.get('duration_ms'), 200.0),
                         'fault_angle_deg': _f(params.get('fault_angle_deg'), 0.0)},
            'faults': faults,
            'breakers': list(worst.values()),
        },
        'warnings': warnings_out,
    })
