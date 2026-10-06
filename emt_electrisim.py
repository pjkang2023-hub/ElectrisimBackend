"""
EMT study of the AC and DC networks: their voltages and currents in time
through a fault or a load step, from the load-flow state. The AC network is
built by emt_ac (three-phase, from the circuit's own AC steady state); this
module builds the DC networks and the converters between them.

The circuit (emt_solver.Circuit):
- DC cables as one or more pi sections (R, L, C), so a long cable carries its
  travelling waves as a ladder;
- DC-link capacitors with ESR and ESL; DC sources behind an internal R and L;
- DC loads by their load-flow model in time - shares of constant power,
  current and resistance, the constant-power part drawing constant current
  below its minimum voltage - behind their input filters, so a
  constant-power load's stability shows;
- DC breakers: their current-limiting inductor, and a switch in parallel
  with a surge arrester. A breaker trips when its current passes its trip
  current, opens after its opening time, and its arrester clamps the voltage
  that forces the current to zero, absorbing the energy;
- VSCs, until the converter models of the next phase: on the DC side a
  stiff source at their load-flow DC voltage, on the AC side a current at
  their load-flow power - both stopping when their DC voltage falls below a
  threshold and they block. Their diodes then rectify the AC network into
  the DC side, through an isolating converter transformer whose leakage is
  the VSC's reactor (the AC network's ground and the DC negative pole are the
  same reference node; the transformer keeps them apart). DC/DC converters'
  and SSTs' output stages are stiff sources that block the same way; their
  inputs, constant-power loads.

DC faults pole to pole; AC faults of any kind.
"""

import copy
import json
import math
import warnings

import numpy as np

import pandapower_electrisim as pe
from dc_fault_electrisim import _ac_thevenin, _f, _label, _row_id, _waveform
from emt_ac import AcBuilder
from emt_converters import VscAverage
from emt_solver import R_BIAS, Circuit, dc_load_current


# A constant-power load with no minimum voltage set draws constant current below this.
_DEFAULT_V_MIN = 0.8
# A constant-power load with no input capacitance is given 4 ms x P / V^2 (its stored energy, 2 ms of its power).
_DEFAULT_LINK_S = 4e-3


def _round(x, nd=6):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


class _EmtBuilder:
    def __init__(self, net, params, warnings_out):
        self.net, self.params, self.warnings = net, params, warnings_out
        self.ckt = Circuit()
        self.f_hz = float(getattr(net, 'f_hz', 50.0) or 50.0)
        self.bus_node, self.v_bus, self.vn = {}, {}, {}
        self.visible_buses = []
        self.series = []        # (kind, name, id, label, (array, index), sign)
        self.loads = []         # (label, id, bus, node, func-array, nominal V)
        self.breakers = []      # dict per breaker
        self.blockers = []      # (label, switch index, node, threshold V)
        self.vscs = []          # VSCs on the AC network: average-value models with their controls
        self._breakers_by_target = {}
        self.default_filters = []

    # --- the network ------------------------------------------------------------

    def build(self):
        net, ckt = self.net, self.ckt
        res = net.res_bus_dc
        for b in net.bus_dc.index:
            if not bool(net.bus_dc.at[b, 'in_service']) or b not in res.index or not np.isfinite(res.at[b, 'vm_pu']):
                continue
            self.vn[b] = float(net.bus_dc.at[b, 'vn_kv']) * 1e3
            self.v_bus[b] = float(res.at[b, 'vm_pu']) * self.vn[b]
            self.bus_node[b] = ckt.node(_label(net, 'bus_dc', b), self.v_bus[b])
            if not ('electrisim_aux' in net.bus_dc.columns and net.bus_dc.at[b, 'electrisim_aux'] == True):
                self.visible_buses.append(b)
        for rec in getattr(net, 'electrisim_dc_breakers', None) or []:
            if rec['closed'] and rec['target'] is not None:
                self._breakers_by_target.setdefault(rec['target'], []).append(rec)
        self.ac = AcBuilder(ckt, net, self.params, self.warnings, self.f_hz).build()
        self._cables()
        self._capacitors()
        self._sources()
        self._loads()
        self._converters()
        if 'b2b_vsc' in net and len(net.b2b_vsc):
            self.warnings.append('The B2B VSCs are left out of the EMT study: their two-pole DC side is not modelled yet.')
        return self

    def _terminal(self, table, idx, bus, i0):
        """The node an element connects at: its bus, or beyond a breaker in front of it."""
        recs = [r for r in self._breakers_by_target.get((table, idx), []) if r['bus_dc'] == bus]
        if not recs:
            return self.bus_node[bus]
        return self._breaker(recs[0], self.bus_node[bus], None, i0)

    def _breaker(self, rec, a, b, i0):
        """A breaker from node a (to node b, or a new node): its limiting inductor, then a switch and its arrester."""
        ckt = self.ckt
        v0 = self.v_bus.get(rec['bus_dc'], 0.0)
        mid = ckt.node(f"{rec['label']} contacts", v0)
        b = ckt.node(f"{rec['label']} terminal", v0) if b is None else b
        k_l = ckt.add_rl(a, mid, 0.0, rec['limiting_inductance_mh'] * 1e-3, i0=i0)
        k_sw = ckt.add_switch(mid, b, closed=True)
        vn = self.vn.get(rec['bus_dc'], 800.0)
        clamp = rec['arrester_clamp_kv'] * 1e3 if rec['arrester_clamp_kv'] > 0 else 1.5 * vn
        k_arr = ckt.add_arrester(mid, b, clamp)
        trip = rec.get('trip_current_ka') or 0.0
        if trip <= 0:
            trip = 2.0 * rec['rated_current_ka'] if rec['rated_current_ka'] > 0 else 0.0
        self.breakers.append({'rec': rec, 'rl': k_l, 'sw': k_sw, 'arr': k_arr, 'mid': mid, 'b': b,
                              'trip_a': trip * 1e3, 't_trip': None, 't_open': None})
        self.series.append(('DC breaker', rec['label'], rec['id'], rec['label'], ('i_rl', k_l), 1.0))
        return b

    def _cables(self):
        net, ckt = self.net, self.ckt
        max_km = max(_f(self.params.get('max_section_km'), 1.0), 1e-3)
        for li in net.line_dc.index:
            if not bool(net.line_dc.at[li, 'in_service']):
                continue
            fb, tb = int(net.line_dc.at[li, 'from_bus_dc']), int(net.line_dc.at[li, 'to_bus_dc'])
            if fb not in self.bus_node or tb not in self.bus_node:
                continue
            i_from = _f(net.res_line_dc.at[li, 'i_from_ka']) * 1e3 if li in net.res_line_dc.index else 0.0
            i_to = _f(net.res_line_dc.at[li, 'i_to_ka']) * 1e3 if li in net.res_line_dc.index else -i_from
            if 'electrisim_dc_breaker' in net.line_dc.columns and net.line_dc.at[li, 'electrisim_dc_breaker'] == True:
                rec = next((r for recs in self._breakers_by_target.values() for r in recs if r['target'] == ('line_dc', li)), None)
                if rec:
                    self._breaker(rec, self.bus_node[fb], self.bus_node[tb], i_from)
                else:
                    ckt.add_rl(self.bus_node[fb], self.bus_node[tb], 0.0, 0.0, i0=i_from)
                continue
            km = _f(net.line_dc.at[li, 'length_km'], 1.0)
            r = _f(net.line_dc.at[li, 'r_ohm_per_km']) * km
            l = _f(net.line_dc.at[li, 'l_mh_per_km']) * 1e-3 * km if 'l_mh_per_km' in net.line_dc.columns else 0.0
            c = _f(net.line_dc.at[li, 'c_uf_per_km']) * 1e-6 * km if 'c_uf_per_km' in net.line_dc.columns else 0.0
            end_f = self._terminal('line_dc', li, fb, i_from)
            end_t = self._terminal('line_dc', li, tb, i_to)
            n_sec = max(1, int(math.ceil(km / max_km - 1e-9)))
            label = _label(net, 'line_dc', li)
            v_f, v_t = self.v_bus[fb], self.v_bus[tb]
            nodes = [end_f] + [ckt.node(f'{label} section {k}', v_f + (v_t - v_f) * k / n_sec)
                               for k in range(1, n_sec)] + [end_t]
            first = None
            for k in range(n_sec):
                kk = ckt.add_rl(nodes[k], nodes[k + 1], r / n_sec, l / n_sec, i0=i_from)
                first = kk if first is None else first
            if c > 0:
                for k, node in enumerate(nodes):
                    share = 0.5 if k in (0, n_sec) else 1.0
                    v0 = v_f + (v_t - v_f) * k / n_sec
                    ckt.add_c(0, node, share * c / n_sec, w0=-v0)
            self.series.append(('DC cable', label, _row_id(net, 'line_dc', li), f'{label} (from end)', ('i_rl', first), 1.0))

    def _capacitors(self):
        ckt = self.ckt
        names = getattr(self.net, 'user_friendly_names', {}) or {}
        for cap in getattr(self.net, 'electrisim_dc_capacitors', None) or []:
            bus = cap['bus_dc']
            if not cap.get('in_service', True) or bus not in self.bus_node or cap['c_mf'] <= 0:
                continue
            label = names.get(cap['name'], cap['name'])
            inner = ckt.node(f'{label} capacitor', self.v_bus[bus])
            ckt.add_c(0, inner, cap['c_mf'] * 1e-3, w0=-self.v_bus[bus])
            k = ckt.add_rl(inner, self.bus_node[bus], cap['esr_mohm'] * 1e-3, cap['esl_uh'] * 1e-6)
            self.series.append(('DC capacitor', label, cap.get('id', ''), label, ('i_rl', k), 1.0))

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
            term = self._terminal('source_dc', si, bus, -i0)
            k = ckt.add_rl(0, term, r, l, i0=i0, e0=self.v_bus[bus] + r * i0)
            self.series.append(('DC source', label, _row_id(net, 'source_dc', si), label, ('i_rl', k), 1.0))

    def _loads(self):
        net, ckt = self.net, self.ckt
        ld = net.load_dc
        for li in ld.index:
            bus = int(ld.at[li, 'bus_dc'])
            if not bool(ld.at[li, 'in_service']) or bus not in self.bus_node:
                continue
            p = _f(net.res_load_dc.at[li, 'p_dc_mw']) if li in net.res_load_dc.index else _f(ld.at[li, 'p_dc_mw'])
            vn = self.vn[bus]
            role = ld.at[li, 'electrisim_dcdc_role'] if 'electrisim_dcdc_role' in ld.columns else None
            aux = 'electrisim_aux' in ld.columns and ld.at[li, 'electrisim_aux'] == True
            if aux:
                # A converter stage's input (a constant-power load) or a power-mode output (an injection).
                shares, v_min = (1.0, 0.0, 0.0), 0.8
                p_model = p
            else:
                shares = tuple(_f(ld.at[li, c], d) for c, d in (('electrisim_share_p', 1.0), ('electrisim_share_i', 0.0),
                                                               ('electrisim_share_r', 0.0))) if 'electrisim_share_p' in ld.columns else (1.0, 0.0, 0.0)
                v_min = _f(ld.at[li, 'electrisim_v_min_pu'], 0.0) if 'electrisim_v_min_pu' in ld.columns else 0.0
                if shares[0] > 0 and v_min <= 0:
                    v_min = _DEFAULT_V_MIN   # P / v grows without bound as a fault takes v to zero
                p_model = _f(ld.at[li, 'electrisim_p_rated_mw'], p) if 'electrisim_p_rated_mw' in ld.columns else p
            pw = [p_model]
            func = dc_load_current(pw, vn / 1e3, *shares, v_min)
            i_load = p * 1e6 / self.v_bus[bus]
            term = self._terminal('load_dc', li, bus, i_load)
            c = _f(ld.at[li, 'filter_c_uf']) * 1e-6 if 'filter_c_uf' in ld.columns else 0.0
            l = _f(ld.at[li, 'filter_l_mh']) * 1e-3 if 'filter_l_mh' in ld.columns else 0.0
            if shares[0] > 0 and c <= 0 and p > 0:
                # A converter-fed load has a DC-link capacitor; with none, a constant-power load fed
                # through any inductance is unstable within microseconds.
                c = _DEFAULT_LINK_S * p * 1e6 / self.v_bus[bus] ** 2
                self.default_filters.append(_label(net, 'load_dc', li))
            node = term
            if c > 0 or l > 0:
                node = ckt.node(f'{_label(net, "load_dc", li)} input', self.v_bus[bus])
                ckt.add_rl(term, node, 0.0, l, i0=i_load)
                if c > 0:
                    ckt.add_c(0, node, c, w0=-self.v_bus[bus])
            k = ckt.add_nonlinear(node, 0, func)
            label = _label(net, 'load_dc', li)
            if not aux:
                self.loads.append({'label': label, 'id': _row_id(net, 'load_dc', li), 'node': node, 'p': pw,
                                   'bus': bus, 'vn': vn, 'constant_power': shares[0] > 0})
                self.series.append(('DC load', label, _row_id(net, 'load_dc', li), label, ('i_nl', k), 1.0))
            elif role == 'input':
                self.series.append(('Converter input', label, '', label, ('i_nl', k), 1.0))

    def _converters(self):
        net, ckt = self.net, self.ckt
        if 'vsc' not in net or not len(net.vsc):
            return
        block = _f(self.params.get('vsc_block_pu'), 0.8)
        thevenin = None
        w = 2.0 * math.pi * self.f_hz
        for vi in net.vsc.index:
            bus = int(net.vsc.at[vi, 'bus_dc'])
            if not bool(net.vsc.at[vi, 'in_service']) or bus not in self.bus_node:
                continue
            aux = 'electrisim_aux' in net.vsc.columns and net.vsc.at[vi, 'electrisim_aux'] == True
            label = _label(net, 'vsc', vi)
            p = _f(net.res_vsc.at[vi, 'p_dc_mw']) if vi in net.res_vsc.index else 0.0
            i0 = -p * 1e6 / self.v_bus[bus]
            r_dc = max(_f(net.vsc.at[vi, 'r_dc_ohm']), 1e-6)
            term = self._terminal('vsc', vi, bus, -i0)
            if not aux and int(net.vsc.at[vi, 'bus']) in self.ac.nodes:
                # On the AC network: its average-value model, with its controls.
                conv = VscAverage(self, vi, label, bus, term, r_dc, block)
                conv.id = _row_id(net, 'vsc', vi)
                self.vscs.append(conv)
                self.series.append(('VSC', label, conv.id, f'{label} (DC side)', ('i_rl', conv.k_dc_out), -1.0))
                self.ac.series.append(('VSC', label, conv.id, f'{label} (AC side)',
                                       [[('i_rl', k, None, 1.0)] for k in conv.k_e]))
                continue
            # Its controlled stage: a stiff source at its DC voltage, until it blocks.
            src = ckt.node(f'{label} source', self.v_bus[bus])
            ckt.add_rl(0, src, r_dc, 0.0, e0=self.v_bus[bus] + r_dc * i0)
            k_sw = ckt.add_switch(src, term, closed=True)
            blocker = {'label': label, 'sw': k_sw, 'node': term, 'v_block': block * self.vn[bus], 't': None, 'sources': []}
            self.blockers.append(blocker)
            self.series.append(('Converter output' if aux else 'VSC', label, '' if aux else _row_id(net, 'vsc', vi),
                                label, ('i_sw', k_sw), 1.0))
            if aux:
                continue
            ac_bus = int(net.vsc.at[vi, 'bus'])
            if ac_bus in self.ac.nodes:
                self._vsc_on_ac_network(vi, label, bus, term, r_dc, blocker)
                continue
            # Blocked, its diodes rectify the AC grid into the DC side.
            if thevenin is None:
                thevenin = _ac_thevenin(net, self.warnings)
            ac_bus = int(net.vsc.at[vi, 'bus'])
            rk, xk = thevenin.get(ac_bus, (0.0, 0.0))
            r_ac, x_ac = rk + _f(net.vsc.at[vi, 'r_ohm']), xk + _f(net.vsc.at[vi, 'x_ohm'])
            vm = _f(net.res_bus.at[ac_bus, 'vm_pu'], 1.0) if ac_bus in net.res_bus.index else 1.0
            v_ll = vm * float(net.bus.at[ac_bus, 'vn_kv']) * 1e3
            p_node, neutral = ckt.node(f'{label} DC+', self.v_bus[bus]), ckt.node(f'{label} AC neutral', self.v_bus[bus] / 2)
            ckt.add_r(neutral, p_node, R_BIAS)
            ckt.add_r(neutral, 0, R_BIAS)
            for k_ph in range(3):
                x = ckt.node(f'{label} phase {"abc"[k_ph]}', self.v_bus[bus] / 2)
                ckt.add_rl(neutral, x, r_ac, x_ac / w, ac=(math.sqrt(2.0 / 3.0) * v_ll, w, -k_ph * 2.0 * math.pi / 3.0))
                ckt.add_diode(x, p_node)
                ckt.add_diode(0, x)
            ckt.add_rl(p_node, term, r_dc, 0.0)

    def _vsc_on_ac_network(self, vi, label, bus, term, r_dc, blocker):
        """
        A VSC whose AC bus is in the model: on the AC side, its load-flow power
        as a current; its diodes on that bus through an isolating converter
        transformer - 1:1, its leakage the VSC's reactor - the bridge's side
        floating about the DC link's midpoint.
        """
        net, ckt, ac = self.net, self.ckt, self.ac
        w = ac.w
        ac_bus = int(net.vsc.at[vi, 'bus'])
        p, q = (_f(net.res_vsc.at[vi, 'p_mw']), _f(net.res_vsc.at[vi, 'q_mvar'])) if vi in net.res_vsc.index else (0.0, 0.0)
        amp, ph = ac._three(ac._injection_phasor(ac_bus, -p, -q))     # its results use the load convention
        for k in range(3):
            blocker['sources'].append(ckt.add_isrc(0, ac.nodes[ac_bus][k], ac=(amp[k], w, ph[k])))
        r_ac, x_ac = max(_f(net.vsc.at[vi, 'r_ohm']), 1e-6), max(_f(net.vsc.at[vi, 'x_ohm']), 1e-6)
        l_leak = x_ac / w
        l_mag = 1000.0 * l_leak
        r_m = np.diag([0.0, r_ac])
        l_m = np.array([[l_mag, l_mag], [l_mag, l_mag + l_leak]])
        p_node = ckt.node(f'{label} DC+', self.v_bus[bus])
        mid = ckt.node(f'{label} bridge star point', self.v_bus[bus] / 2)
        ckt.add_r(mid, p_node, R_BIAS)
        ckt.add_r(mid, 0, R_BIAS)
        for k in range(3):
            x = ckt.node(f'{label} bridge {"abc"[k]}', self.v_bus[bus] / 2)
            ckt.add_coupled([ac.nodes[ac_bus][k], x], [0, mid], r_m, l_m)
            ckt.add_diode(x, p_node)
            ckt.add_diode(0, x)
        ckt.add_rl(p_node, term, r_dc, 0.0)

    # --- events and protection ----------------------------------------------------

    def controller(self, t, state):
        changed = False
        for conv in self.vscs:
            changed = conv.control(t, state) or changed
        for blk in self.blockers:
            if blk['t'] is None and state.v[blk['node']] < blk['v_block']:
                blk['t'] = t
                state.set_switch(blk['sw'], False)
                for k in blk.get('sources', []):
                    state.set_source(k, False)
                changed = True
        for brk in self.breakers:
            if brk['t_open'] is not None:
                continue
            if brk['t_trip'] is None and brk['trip_a'] > 0 and abs(state.i_rl[brk['rl']]) > brk['trip_a']:
                brk['t_trip'] = t
            if brk['t_trip'] is not None and t >= brk['t_trip'] + brk['rec']['opening_time_ms'] * 1e-3 - 1e-12:
                brk['t_open'] = t
                brk['i_open'] = abs(state.i_rl[brk['rl']])
                state.set_switch(brk['sw'], False)
                changed = True
        return changed


def _oscillation_verdict(t, v, t_from, vn, min_span=1e-3):
    """
    Whether a load's voltage settles after the last event, keeps swinging,
    swings ever wider - or is lost. ``min_span``: the run after the last event
    it needs to tell - a few periods of the slowest control loop there.
    """
    T = t[-1]
    if T - t_from < 1e-3:
        return None
    span = T - t_from
    early = (t >= t_from + 0.25 * span) & (t < t_from + 0.5 * span)
    late = t >= t_from + 0.75 * span
    if not early.any() or not late.any():
        return None
    if abs(float(v[-1])) < 0.5 * vn:
        return 'lost supply'      # cut off, or on the faulted bus: its swing says nothing of its stability
    if span < min_span:
        return 'too short to tell'
    a_early, a_late = float(np.ptp(v[early])), float(np.ptp(v[late]))
    if a_late < 1e-4 * vn or a_late < 0.9 * a_early:
        return 'settles'
    if a_late > 1.1 * a_early:
        return 'oscillates, growing'
    return 'oscillates'


METHOD = ("A time-domain simulation of the AC and DC networks from their load-flow state. AC: three-phase, grids behind their positive- and zero-sequence impedance, lines as coupled pi sections, transformers by their vector group, loads as constant impedance, inverter-based generation as constant current, generators behind their subtransient reactance, started in their own steady state; an AC fault clears at each phase's current zero. DC: cables as pi sections, "
          "DC loads by their load-flow model behind their input filters, breakers that trip on overcurrent and "
          "open into their surge arresters. A VSC on the AC network is an average-value model behind its reactor "
          "and isolating transformer, with its DC link, under its controls - PLL, dq current control, its DC "
          "voltage or power, its reactive power or AC voltage - limited to its current limit, and blocking, its "
          "diodes left, on DC undervoltage or overcurrent. DC/DC converter and SST outputs are stiff sources "
          "that block on undervoltage; converter inputs are constant-power loads. A constant-power load with no minimum voltage set "
          "draws constant current below 0.8 pu; one with no input capacitance is given 4 ms x P / V^2. "
          "Pole-to-pole faults.")


def emt_study(net, params, in_data=None):
    params = params or {}
    warnings_out = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pe._electrisim_runpp(net, algorithm='nr', calculate_voltage_angles=True, init='auto')
    except Exception as exc:   # noqa: BLE001
        return json.dumps({'error': True, 'message': f'The load flow the EMT study starts from did not solve: {exc}'})
    warnings_out.extend(getattr(net, 'warnings', []) or [])
    if not len(net.bus) and ('bus_dc' not in net or not len(net.bus_dc)):
        return json.dumps({'error': True, 'message': 'There is no network to study.', 'warnings': warnings_out})
    b = _EmtBuilder(net, params, warnings_out).build()
    ckt = b.ckt
    if b.ac.nodes:
        ckt.start_in_ac_steady_state(b.ac.w)
    dt = max(_f(params.get('time_step_us'), 1.0), 0.01) * 1e-6
    t_end = max(_f(params.get('duration_ms'), 50.0), 0.5) * 1e-3
    event_times = []

    # A fault on a DC bus.
    fault = None
    fault_bus = str(params.get('fault_bus') or 'none')
    if fault_bus not in ('none', ''):
        hit = [k for k in b.visible_buses if fault_bus in (str(net.bus_dc.at[k, 'name']), str(_row_id(net, 'bus_dc', k)))]
        if not hit:
            warnings_out.append(f'No DC bus {fault_bus} to fault: the study runs without a fault.')
        else:
            t_f = _f(params.get('fault_time_ms'), 5.0) * 1e-3
            r_f = max(_f(params.get('fault_resistance_mohm'), 0.0) * 1e-3, 1e-6)
            mid = ckt.node('fault')
            k_sw = ckt.add_switch(b.bus_node[hit[0]], mid, closed=False)
            ckt.add_r(mid, 0, r_f)
            ckt.at(t_f, lambda st, k=k_sw: st.set_switch(k, True))
            fault = {'bus': hit[0], 'sw': k_sw, 't': t_f, 'r': r_f}
            event_times.append(t_f)
            b.series.append(('Fault', 'Fault', '', f'Fault at {_label(net, "bus_dc", hit[0])}', ('i_sw', k_sw), 1.0))

    # A step in a DC load's power.
    step_load = str(params.get('step_load') or 'none')
    if step_load not in ('none', ''):
        load = next((l for l in b.loads if step_load in (l['id'], l['label'])), None)
        if load is None:
            hit = [i for i in net.load_dc.index if str(net.load_dc.at[i, 'name']) == step_load]
            load = next((l for l in b.loads if hit and l['id'] == _row_id(net, 'load_dc', hit[0])), None)
        if load is None:
            warnings_out.append(f'No DC load {step_load} to step: the study runs without a load step.')
        else:
            t_s = _f(params.get('step_time_ms'), 5.0) * 1e-3
            factor = 1.0 + _f(params.get('step_percent'), 10.0) / 100.0
            ckt.at(t_s, lambda st, pw=load['p'], f=factor: pw.__setitem__(0, pw[0] * f))
            event_times.append(t_s)

    # A fault on an AC bus.
    ac_fault = None
    ac_bus = str(params.get('ac_fault_bus') or 'none')
    if ac_bus not in ('none', ''):
        hit = [k for k in b.ac.nodes if ac_bus in (str(net.bus.at[k, 'name']), str(_row_id(net, 'bus', k)))]
        kind = str(params.get('ac_fault_type') or 'abcg').lower()
        if not hit:
            warnings_out.append(f'No AC bus {ac_bus} to fault: the study runs without an AC fault.')
        elif kind not in ('ag', 'bg', 'cg', 'ab', 'bc', 'ca', 'abg', 'bcg', 'cag', 'abc', 'abcg'):
            warnings_out.append(f'No AC fault of kind {kind}: the study runs without one.')
        else:
            t_on = _f(params.get('ac_fault_time_ms'), 20.0) * 1e-3
            dur = _f(params.get('ac_fault_duration_ms'), 0.0) * 1e-3
            r_f = max(_f(params.get('ac_fault_resistance_ohm'), 0.0), 1e-4)
            ac_fault = b.ac.fault(hit[0], kind, r_f, r_f, t_on, t_on + dur if dur > 0 else None)
            ac_fault.update(bus=hit[0], kind=kind, t=t_on, t_off=t_on + dur if dur > 0 else None)
            event_times.append(t_on + (dur if dur > 0 else 0.0))

    ckt.add_controller(b.controller)
    t_fine = min(t_end, (max(event_times) if event_times else 0.0) + 0.01)
    with np.errstate(all='ignore'):
        sim = ckt.simulate(t_end, dt, dt_coarse=min(10 * dt, 2e-5), t_fine=t_fine)
    t = sim['t']
    t_last = max(event_times) if event_times else 0.0

    buses = []
    for k in b.visible_buses:
        v = sim['v'][:, b.bus_node[k]]
        j = int(np.argmin(v))
        buses.append({'name': str(net.bus_dc.at[k, 'name']), 'id': _row_id(net, 'bus_dc', k),
                      'label': _label(net, 'bus_dc', k), 'vn_kv': b.vn[k] / 1e3,
                      'v_min_pu': _round(v[j] / b.vn[k]), 't_min_ms': _round(t[j] * 1e3),
                      'v_max_pu': _round(float(np.max(v)) / b.vn[k]), 'v_final_pu': _round(v[-1] / b.vn[k]),
                      'waveform': _volts(_waveform(t, v, j, t_fine))})
    branches = []
    for kind, name, cid, label, (arr, k), sign in b.series:
        i = sign * sim[arr][:, k]
        j = int(np.argmax(np.abs(i)))
        branches.append({'kind': kind, 'name': name, 'id': cid, 'label': label,
                         'i_peak_ka': _round(abs(i[j]) * 1e-3), 't_peak_ms': _round(t[j] * 1e3),
                         'waveform': _waveform(t, i, j, t_fine)})
    result = {'method': METHOD, 'buses': buses, 'branches': branches, 'breakers': [], 'loads': [], 'fault': None,
              'converters_blocked': [{'label': blk['label'], 't_ms': _round(blk['t'] * 1e3)} for blk in b.blockers if blk['t'] is not None]
              + [{'label': c.label, 't_ms': _round(c.blocked_at * 1e3)} for c in b.vscs if c.blocked_at is not None],
              'converters': [_converter_result(c) for c in b.vscs],
              'settings': {'time_step_us': dt * 1e6, 'duration_ms': t_end * 1e3,
                           'max_section_km': _f(params.get('max_section_km'), 1.0),
                           'vsc_block_pu': _f(params.get('vsc_block_pu'), 0.8)}}
    if fault:
        i = sim['i_sw'][:, fault['sw']]
        j = int(np.argmax(i))
        after = t >= fault['t']
        di = np.diff(i[after]) / np.maximum(np.diff(t[after]), 1e-15) if after.sum() > 1 else np.array([0.0])
        result['fault'] = {'bus': _label(net, 'bus_dc', fault['bus']), 'id': _row_id(net, 'bus_dc', fault['bus']),
                           't_ms': fault['t'] * 1e3, 'ip_ka': _round(i[j] * 1e-3), 'tp_ms': _round(t[j] * 1e3),
                           'didt_max_ka_per_ms': _round(float(np.max(di)) * 1e-6), 'i_final_ka': _round(i[-1] * 1e-3)}
    for brk in b.breakers:
        rec = brk['rec']
        va = sim['v'][:, brk['mid']] - sim['v'][:, brk['b']]
        ia = sim['i_arr'][:, brk['arr']]
        energy = float(np.trapezoid(va * ia, t))
        i = np.abs(sim['i_rl'][:, brk['rl']])
        cleared = None
        if brk['t_open'] is not None:
            done = np.nonzero((t > brk['t_open']) & (i <= max(1.0, 0.01 * brk.get('i_open', 0.0))))[0]
            cleared = float(t[done[0]]) if len(done) else None
        out = {'name': rec['name'], 'id': rec['id'], 'label': rec['label'], 'breaker_type': rec.get('breaker_type'),
               'trip_current_ka': _round(brk['trip_a'] / 1e3),
               'tripped_ms': _round(brk['t_trip'] * 1e3) if brk['t_trip'] is not None else None,
               'opened_ms': _round(brk['t_open'] * 1e3) if brk['t_open'] is not None else None,
               'i_open_ka': _round(brk.get('i_open', 0.0) * 1e-3) if brk['t_open'] is not None else None,
               'cleared_ms': _round(cleared * 1e3) if cleared is not None else None,
               'i_peak_ka': _round(float(np.max(i)) * 1e-3),
               'arrester_energy_kj': _round(energy / 1e3), 'arrester_v_peak_kv': _round(float(np.max(np.abs(va))) / 1e3),
               'breaking_capacity_ka': rec['breaking_capacity_ka'], 'arrester_energy_rating_kj': rec['arrester_energy_kj']}
        out['exceeds_capacity'] = bool(out['i_open_ka'] is not None and rec['breaking_capacity_ka'] > 0
                                       and out['i_open_ka'] > rec['breaking_capacity_ka'])
        out['exceeds_energy'] = bool(rec['arrester_energy_kj'] > 0 and energy / 1e3 > rec['arrester_energy_kj'])
        if out['exceeds_capacity']:
            warnings_out.append(f"DC breaker {rec['label']} interrupts {out['i_open_ka']:.3g} kA, above its "
                                f"{rec['breaking_capacity_ka']:g} kA breaking capacity.")
        if out['exceeds_energy']:
            warnings_out.append(f"DC breaker {rec['label']}: its arrester absorbs {energy / 1e3:.3g} kJ, above its "
                                f"{rec['arrester_energy_kj']:g} kJ rating.")
        if brk['t_open'] is not None and cleared is None:
            warnings_out.append(f"DC breaker {rec['label']} opened but its current had not reached zero by the end of the run.")
        result['breakers'].append(out)
    result['ac'] = _ac_results(sim, b.ac, net, t_fine, ac_fault)
    # A VSC's DC voltage loop, some 30 Hz, swings slower than the network: two of its periods to tell.
    min_span = VSC_VERDICT_SPAN if b.vscs else 1e-3
    for load in b.loads:
        v = sim['v'][:, load['node']]
        result['loads'].append({'label': load['label'], 'id': load['id'], 'constant_power': load['constant_power'],
                                'v_min_pu': _round(float(np.min(v)) / load['vn']),
                                'verdict': _oscillation_verdict(t, v, t_last, load['vn'], min_span) if load['constant_power'] else None})
        if load['constant_power'] and result['loads'][-1]['verdict'] == 'oscillates, growing':
            warnings_out.append(f"DC load {load['label']}: its voltage swings ever wider - a constant-power load "
                                "beyond its stability limit; more DC-link capacitance or a stiffer supply steadies it.")
    if any(l['verdict'] == 'too short to tell' for l in result['loads']):
        warnings_out.append(f'Constant-power loads: run at least {VSC_VERDICT_SPAN * 1e3:.0f} ms after the last event to tell '
                            "whether they settle - the VSCs' DC voltage loops swing at some 30 Hz.")
    if b.default_filters:
        warnings_out.append('Given an input capacitance of 4 ms x P / V^2, as they have none: '
                            + ', '.join(b.default_filters) + '. Enter their input filters for their own values.')
    if b.blockers and any(blk['t'] is not None for blk in b.blockers):
        names = ', '.join(blk['label'] for blk in b.blockers if blk['t'] is not None)
        warnings_out.append(f'Blocked on undervoltage: {names}.')
    return json.dumps({'emt': result, 'warnings': warnings_out})


def _rms(t, y, period):
    """The rms over the last period at each time (nan for the first)."""
    sq = np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] ** 2 + y[:-1] ** 2) * np.diff(t))])
    back = np.interp(t - period, t, sq, left=np.nan)
    return np.sqrt(np.maximum((sq - back) / period, 0.0))


def _ac_results(sim, ac, net, t_fine, fault):
    """Three-phase bus voltages with their lowest rms over a cycle, phase currents, the fault's."""
    if not ac.nodes:
        return None
    t = sim['t']
    period = 1.0 / ac.f_hz
    buses = []
    for b, nodes in ac.nodes.items():
        v_ph = ac.vn[b] / math.sqrt(3.0)
        waves, rms_min, t_min, rms_end = {}, np.inf, None, []
        for k, p in enumerate('abc'):
            v = sim['v'][:, nodes[k]]
            waves[f'v_{p}_kv'] = _waveform(t, v, None, t_fine)['i_ka']
            waves['t_ms'] = _waveform(t, v, None, t_fine)['t_ms']
            r = _rms(t, v, period)
            ok = np.isfinite(r)
            if ok.any():
                j = int(np.nanargmin(np.where(ok, r, np.nan)))
                if r[j] < rms_min:
                    rms_min, t_min = float(r[j]), float(t[j])
                rms_end.append(float(r[-1]))
        buses.append({'name': str(net.bus.at[b, 'name']), 'id': _row_id(net, 'bus', b), 'label': _label(net, 'bus', b),
                      'vn_kv': ac.vn[b] / 1e3,
                      'v_rms_min_pu': _round(rms_min / v_ph) if np.isfinite(rms_min) else None,
                      't_min_ms': _round(t_min * 1e3) if t_min is not None else None,
                      'v_rms_final_pu': _round(min(rms_end) / v_ph) if rms_end else None,
                      'waveform': waves})
    branches = []
    for kind, name, cid, label, phases in ac.series:
        out = {'kind': kind, 'name': name, 'id': cid, 'label': label}
        peak = 0.0
        for k, terms in enumerate(phases):
            terms = terms if isinstance(terms, list) else [terms]
            i = np.zeros(len(t))
            for arr, g, col, sign in terms:
                i = i + sign * (sim[arr][g][:, col] if arr == 'i_cp' else sim[arr][:, g])
            w = _waveform(t, i, None, t_fine)
            out['t_ms'] = w['t_ms']
            out[f'i_{"abc"[k]}_ka'] = w['i_ka']
            peak = max(peak, float(np.max(np.abs(i))))
        out['i_peak_ka'] = _round(peak * 1e-3)
        branches.append(out)
    fault_out = None
    if fault:
        fault_out = {'bus': _label(net, 'bus', fault['bus']), 'id': _row_id(net, 'bus', fault['bus']), 'kind': fault['kind'],
                     't_ms': fault['t'] * 1e3, 't_off_ms': fault['t_off'] * 1e3 if fault['t_off'] else None, 'phases': []}
        during = (t >= fault['t']) & (t <= (fault['t_off'] or t[-1]))
        for ph, k in zip(fault['phases'], fault['switches']):
            i = sim['i_sw'][:, k]
            r = _rms(t, i, period)
            sel = during & np.isfinite(r) & (t >= fault['t'] + period)
            # Its symmetrical current: the 50 Hz component over the fault's last cycle, the DC offset aside.
            last = during & (t >= (fault['t_off'] or t[-1]) - period)
            i_sym = None
            if last.sum() > 8 and t[last][-1] - t[last][0] > 0.9 * period:
                x = np.column_stack([np.cos(2 * math.pi * ac.f_hz * t[last]), np.sin(2 * math.pi * ac.f_hz * t[last]),
                                     np.ones(last.sum())])
                coef, *_ = np.linalg.lstsq(x, i[last], rcond=None)
                i_sym = math.hypot(coef[0], coef[1]) / math.sqrt(2.0)
            fault_out['phases'].append({'phase': 'abc'[ph], 'i_peak_ka': _round(float(np.max(np.abs(i[during]))) * 1e-3) if during.any() else None,
                                        'i_rms_ka': _round(float(r[sel][-1]) * 1e-3) if sel.any() else None,
                                        'i_sym_ka': _round(i_sym * 1e-3) if i_sym is not None else None,
                                        'waveform': _waveform(t, i, None, t_fine)})
    return {'buses': buses, 'branches': branches, 'fault': fault_out}


def _converter_result(conv):
    """A VSC's active and reactive power out, DC voltage and current through the run."""
    tr = np.array(conv.trace) if conv.trace else np.zeros((0, 5))
    out = {'label': conv.label, 'id': getattr(conv, 'id', ''), 'model': 'average',
           'rated_mva': _round(conv.s_rated / 1e6), 'current_limit_ka': _round(conv.i_max / SQ2 / 1e3),
           'blocked_ms': _round(conv.blocked_at * 1e3) if conv.blocked_at is not None else None,
           'limited_ms': _round(conv.limited_time * 1e3)}
    if len(tr):
        out.update(p_start_mw=_round(tr[0, 1] / 1e6), p_end_mw=_round(tr[-1, 1] / 1e6),
                   q_start_mvar=_round(tr[0, 2] / 1e6), q_end_mvar=_round(tr[-1, 2] / 1e6),
                   v_dc_min_kv=_round(float(np.min(tr[:, 3])) / 1e3), v_dc_end_kv=_round(tr[-1, 3] / 1e3),
                   i_peak_ka=_round(float(np.max(tr[:, 4])) / SQ2 / 1e3))
    return out


SQ2 = math.sqrt(2.0)
VSC_VERDICT_SPAN = 0.06    # s: two periods of a VSC's DC voltage loop


def _volts(w):
    """A voltage waveform in kV: the waveform helper scales by 1e-3, into 'i_ka'."""
    return {'t_ms': w['t_ms'], 'v_kv': w['i_ka']}
