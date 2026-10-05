"""
DC fault study: the current a pole-to-pole fault on a DC bus draws.

The DC network is simulated in time from its load-flow state: DC-link
capacitors (with ESR and ESL), cables as pi sections (R, L, C), DC sources as
their voltage behind an internal R and L, DC load input filters, DC breakers'
current-limiting inductors, and VSCs - which block at the fault, so that
their diodes rectify the AC grid into it, behind the grid's short-circuit
impedance at the converter's AC bus and the converter's own reactor.

The results are given in IEC 61660-1's terms - peak ip, time to peak tp,
quasi-steady current Ik, and the rise and decay time constants tau1, tau2 of
its standard approximation function, fitted to the simulated current - but
are not computed by its method: they are what this network's circuit does.

Simplifications, stated with the results:
- pole-to-pole faults only; a DC bus's voltage is pole to pole and a cable's
  R, L and C per km are its loop values, as in the load flow;
- DC loads leave the network at the fault (a fault current neglects them, as
  IEC 61660-1 does), their input filters stay;
- a VSC's pre-fault current stops as it blocks, and its AC source is the
  pre-fault voltage of its AC bus;
- the breakers do not open: each is checked on the prospective current
  through it at its opening time;
- a discharge faster than the time step is not resolved: a cable's own
  capacitance into a hard fault, which Simscape shows as a spike of some
  20 ns at t = 0 (tests/emt_reference).
"""

import copy
import json
import math
import warnings

import numpy as np
import pandapower as pp
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import minimize_scalar

import pandapower_electrisim as pe

R_FLOOR = 1e-6          # ohm: a branch with no R or L
R_ON, R_OFF = 1e-5, 1e6  # ohm: a diode conducting / blocking
R_BIAS = 1e6            # ohm: holds a converter's AC side at its DC link's midpoint


# --- The circuit and its solver ------------------------------------------------------

class Circuit:
    """
    A circuit of two-terminal branches between nodes; node 0 is the DC
    negative pole. Each branch's current is from its node a to its node b:

    - RL: v_a - v_b + e(t) = R i + L di/dt, with e = e0 + E cos(w t + phi);
    - C:  i = C d(v_a - v_b)/dt;
    - R:  i = (v_a - v_b) / R;
    - diode, anode a and cathode b: a resistance of R_ON conducting, R_OFF not.

    Its state at t = 0 is each RL branch's current and each capacitor's
    voltage v_a - v_b.
    """

    def __init__(self):
        self.labels = ['DC negative pole']
        self.rl, self.c, self.r, self.d = [], [], [], []

    def node(self, label):
        self.labels.append(label)
        return len(self.labels) - 1

    def add_rl(self, a, b, r, l, i0=0.0, e0=0.0, ac=None):
        """``ac``: (amplitude, omega, phase) of an AC electromotive force."""
        if r <= 0 and l <= 0:
            r = R_FLOOR
        amp, om, ph = ac or (0.0, 0.0, 0.0)
        self.rl.append((a, b, max(r, 0.0), max(l, 0.0), i0, e0, amp, om, ph))
        return len(self.rl) - 1

    def add_c(self, a, b, c, w0=0.0):
        self.c.append((a, b, c, w0))
        return len(self.c) - 1

    def add_r(self, a, b, r):
        self.r.append((a, b, max(r, R_FLOOR)))
        return len(self.r) - 1

    def add_diode(self, anode, cathode):
        self.d.append((anode, cathode))
        return len(self.d) - 1

    def simulate(self, t_end, dt, dt_coarse=None, t_fine=None, max_switch_iter=40):
        """
        Trapezoidal integration, with backward Euler for the first step and
        for a step where a diode switches (which would otherwise ring).
        Returns times and, at each, node voltages and branch currents.
        """
        n = len(self.labels)
        rl = np.array(self.rl, dtype=float).reshape(-1, 9)
        cc = np.array(self.c, dtype=float).reshape(-1, 4)
        rr = np.array(self.r, dtype=float).reshape(-1, 3)
        dd = np.array(self.d, dtype=int).reshape(-1, 2)
        ra, rb = rl[:, 0].astype(int), rl[:, 1].astype(int)
        R, L = rl[:, 2], rl[:, 3]
        e0, eamp, eom, eph = rl[:, 5], rl[:, 6], rl[:, 7], rl[:, 8]
        ca, cb, C = cc[:, 0].astype(int), cc[:, 1].astype(int), cc[:, 2]
        xa, xb, Gr = rr[:, 0].astype(int), rr[:, 1].astype(int), 1.0 / rr[:, 2]
        da, db = dd[:, 0], dd[:, 1]

        steps = []
        t = 0.0
        t_fine = t_end if t_fine is None else t_fine
        while t < t_end - 1e-15:
            h = dt if t < t_fine - 1e-15 else (dt_coarse or dt)
            h = min(h, t_end - t)
            steps.append(h)
            t += h
        N = len(steps)
        times = np.concatenate([[0.0], np.cumsum(steps)])
        v_out = np.zeros((N + 1, n))
        i_rl = np.zeros((N + 1, len(R)))
        i_c = np.zeros((N + 1, len(C)))
        i_r = np.zeros((N + 1, len(Gr)))
        i_d = np.zeros((N + 1, len(da)))

        def emf(tt):
            return e0 + eamp * np.cos(eom * tt + eph)

        i_now = rl[:, 4].copy()
        w_now = cc[:, 3].copy()
        ic_now = np.zeros(len(C))
        v_now = None
        on = np.zeros(len(da), dtype=bool)
        i_rl[0] = i_now
        cache = {}

        def stamp(a, b, g, Y):
            np.add.at(Y, (a, a), g)
            np.add.at(Y, (b, b), g)
            np.add.at(Y, (a, b), -g)
            np.add.at(Y, (b, a), -g)

        def factor(method, h, state):
            key = (method, h, state.tobytes())
            if key not in cache:
                k = 2.0 if method == 'tr' else 1.0
                Y = np.zeros((n, n))
                stamp(ra, rb, 1.0 / (R + k * L / h), Y)
                stamp(ca, cb, k * C / h, Y)
                stamp(xa, xb, Gr, Y)
                stamp(da, db, np.where(state, 1.0 / R_ON, 1.0 / R_OFF), Y)
                if len(cache) > 256:
                    cache.clear()
                cache[key] = lu_factor(Y[1:, 1:])
            return cache[key]

        t = 0.0
        for s, h in enumerate(steps):
            t1 = t + h
            e_now, e_next = emf(t), emf(t1)
            method = 'be' if s == 0 else 'tr'
            for _ in range(2):
                state = on.copy()
                for _it in range(max_switch_iter):
                    k = 2.0 if method == 'tr' else 1.0
                    G = 1.0 / (R + k * L / h)
                    if method == 'tr':
                        u_now = v_now[ra] - v_now[rb] + e_now
                        J = G * e_next + G * (u_now + (2.0 * L / h - R) * i_now)
                        Gc = 2.0 * C / h
                        Jc = -Gc * w_now - ic_now
                    else:
                        J = G * e_next + G * (L / h) * i_now
                        Gc = C / h
                        Jc = -Gc * w_now
                    rhs = np.bincount(rb, J, n) - np.bincount(ra, J, n)
                    rhs += np.bincount(cb, Jc, n) - np.bincount(ca, Jc, n)
                    v = np.zeros(n)
                    v[1:] = lu_solve(factor(method, h, state), rhs[1:])
                    vd = v[da] - v[db]
                    new = np.where(state, vd > 0.0, vd > 1e-9)
                    if np.array_equal(new, state):
                        break
                    state = new
                if method == 'be' or np.array_equal(state, on):
                    break
                method = 'be'   # a diode switched: take this step by backward Euler
                on = state
            on = state
            i_now = G * (v[ra] - v[rb]) + J
            w_next = v[ca] - v[cb]
            ic_now = Gc * w_next + Jc
            w_now = w_next
            v_now = v
            t = t1
            v_out[s + 1] = v
            i_rl[s + 1] = i_now
            i_c[s + 1] = ic_now
            i_r[s + 1] = Gr * (v[xa] - v[xb])
            i_d[s + 1] = np.where(on, 1.0 / R_ON, 1.0 / R_OFF) * (v[da] - v[db])
        return {'t': times, 'v': v_out, 'i_rl': i_rl, 'i_c': i_c, 'i_r': i_r, 'i_d': i_d}


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
    return net[table].at[idx, 'id'] if 'id' in net[table].columns else str(idx)


def _ac_thevenin(net, warnings_out):
    """Short-circuit R and X (ohm) at each VSC's AC bus, from pandapower's AC short circuit."""
    try:
        import pandapower.shortcircuit as sc
        ac = copy.deepcopy(net)
        for table in ('vsc', 'b2b_vsc', 'line_dc', 'load_dc', 'source_dc', 'bus_dc'):
            if table in ac and len(ac[table]):
                ac[table].drop(ac[table].index, inplace=True)
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
        self.contributions = []   # (kind, label, id, ('rl'|'c', index))
        self.breakers = []        # (record, rl index)
        self.cable_caps = []
        self.has_converter = False

    def _v(self, bus):
        return self.v_bus[bus]

    def build(self):
        net, ckt = self.net, self.ckt
        res = net.res_bus_dc
        for b in net.bus_dc.index:
            if not bool(net.bus_dc.at[b, 'in_service']) or b not in res.index or not np.isfinite(res.at[b, 'vm_pu']):
                continue
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
        if len(net.b2b_vsc) if 'b2b_vsc' in net else False:
            self.warnings.append('The B2B VSCs are left out of the DC fault study: their two-pole DC side is not modelled yet.')
        return self

    def _terminal(self, table, idx, bus, i0):
        """The node an element connects at: its bus, or beyond a breaker in front of it."""
        recs = [r for r in self._breakers.get((table, idx), []) if r['bus_dc'] == bus]
        if not recs:
            return self.bus_node[bus], 0.0
        rec = recs[0]
        node = self.ckt.node(f"{rec['label']} terminal")
        k = self.ckt.add_rl(self.bus_node[bus], node, 0.0, rec['limiting_inductance_mh'] * 1e-3, i0=i0)
        self.breakers.append((rec, k))
        return node, 0.0

    def _cables(self):
        net, ckt = self.net, self.ckt
        for li in net.line_dc.index:
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
        ckt = self.ckt
        for cap in getattr(self.net, 'electrisim_dc_capacitors', None) or []:
            bus = cap['bus_dc']
            if not cap.get('in_service', True) or bus not in self.bus_node or cap['c_mf'] <= 0:
                continue
            label = (getattr(self.net, 'user_friendly_names', {}) or {}).get(cap['name'], cap['name'])
            c, r, l = cap['c_mf'] * 1e-3, cap['esr_mohm'] * 1e-3, cap['esl_uh'] * 1e-6
            if r > 0 or l > 0:
                inner = ckt.node(f'{label} capacitor')
                ckt.add_c(0, inner, c, w0=-self._v(bus))
                port = ('rl', ckt.add_rl(inner, self.bus_node[bus], r, l))
            else:
                port = ('c', ckt.add_c(0, self.bus_node[bus], c, w0=-self._v(bus)))
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
            p = _f(net.res_source_dc.at[si, 'p_dc_mw']) if si in net.res_source_dc.index else 0.0
            i0 = -p * 1e6 / self._v(bus)   # injected (the results use the load convention)
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
        for li in net.load_dc.index:
            bus = int(net.load_dc.at[li, 'bus_dc'])
            if not bool(net.load_dc.at[li, 'in_service']) or bus not in self.bus_node:
                continue
            c = _f(net.load_dc.at[li, 'filter_c_uf']) * 1e-6 if 'filter_c_uf' in net.load_dc.columns else 0.0
            if c <= 0:
                continue   # it leaves the network at the fault, with nothing to discharge
            label = _label(net, 'load_dc', li)
            l = _f(net.load_dc.at[li, 'filter_l_mh']) * 1e-3 if 'filter_l_mh' in net.load_dc.columns else 0.0
            p = _f(net.res_load_dc.at[li, 'p_dc_mw']) if li in net.res_load_dc.index else 0.0
            i_load = p * 1e6 / self._v(bus)
            term, _ = self._terminal('load_dc', li, bus, i_load)
            inner = ckt.node(f'{label} filter')
            ckt.add_c(0, inner, c, w0=-self._v(bus))
            # The filter inductor keeps the load's current at the fault, into its capacitor.
            k = ckt.add_rl(inner, term, 0.0, l, i0=-i_load if l > 0 else 0.0)
            self.contributions.append(('DC load input filter', label, _row_id(net, 'load_dc', li), ('rl', k)))

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
                # A DC/DC converter's output stage: it blocks and feeds no fault current.
                note = ('The DC/DC converters block at a DC fault and feed none of it; their current '
                        'limits come with the EMT study.')
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
            k = ckt.add_rl(p_node, term, _f(net.vsc.at[vi, 'r_dc_ohm']), 0.0)
            self.contributions.append(('VSC (diodes, blocked)', label, _row_id(net, 'vsc', vi), ('rl', k)))
            self.has_converter = True


def _current(sim, port):
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


def fault_at(builder, bus, params):
    """Simulate a pole-to-pole fault on DC bus ``bus``; its currents in IEC terms."""
    ckt = copy.deepcopy(builder.ckt)
    r_f = max(_f(params.get('fault_resistance_mohm'), 0.0) * 1e-3, R_FLOOR)
    k_f = ckt.add_r(builder.bus_node[bus], 0, r_f)
    dt = max(_f(params.get('time_step_us'), 1.0), 0.01) * 1e-6
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
          "at the fault; VSCs block and their diodes feed the fault from the AC grid; breakers do not open, "
          "each is checked on the current through it at its opening time. A discharge faster than the time step "
          "- a cable's own capacitance into a hard fault, over nanoseconds - is not resolved.")


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
             if chosen in ('all', '') or chosen in (str(net.bus_dc.at[b, 'name']), str(_row_id(net, 'bus_dc', b)))]
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
