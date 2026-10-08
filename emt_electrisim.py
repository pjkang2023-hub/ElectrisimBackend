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

import load_profiles_electrisim as lp
import pandapower_electrisim as pe
from dc_fault_electrisim import _ac_thevenin, _f, _label, _row_id, _waveform
import grid_voltage_profile as gvp
from emt_ac import AcBuilder
import emt_der
from emt_converters import DcDc, GridFormingVsc, Vsc
from emt_solver import DEFAULT_INPUT_WARNING, R_BIAS, Circuit, dc_load_current, default_input_capacitance


# A constant-power load with no minimum voltage set draws constant current below this.
_DEFAULT_V_MIN = 0.8


def _round(x, nd=6):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


class _EmtBuilder:
    in_data = None           # the diagram's rows: a generator's rotor, governor and exciter data

    def __init__(self, net, params, warnings_out, profiles=None):
        self.net, self.params, self.warnings = net, params, warnings_out
        self.profiles = profiles or {}   # (table, index) -> the load's profile plan (_plan_profiles)
        self.ckt = Circuit()
        self.f_hz = float(getattr(net, 'f_hz', 50.0) or 50.0)
        self.bus_node, self.v_bus, self.vn = {}, {}, {}
        self.visible_buses = []
        self.series = []        # (kind, name, id, label, (array, index), sign)
        self.loads = []         # (label, id, bus, node, func-array, nominal V)
        self.breakers = []      # dict per breaker
        self.blockers = []      # (label, switch index, node, threshold V)
        self.vscs = []          # VSCs on the AC network (an SST's rectifier and inverter among them), with their controls
        self.dcdcs = []         # DC/DC converters (an SST's DC/DC stage among them): dual active bridges
        self.skip = set()       # (table, index): load-flow elements a converter's EMT model stands for
        self.ders = []          # (load-flow record, its EMT model): batteries, supercapacitors, flywheels, SOFCs, PV
        self.smoothers = []     # (DC/DC converter, its smoothing controller)
        self._breakers_by_target = {}
        self.default_filters = []

    # --- the network ------------------------------------------------------------

    def build(self):
        net, ckt = self.net, self.ckt
        res = net.res_bus_dc
        for b in net.bus_dc.index:
            if not bool(net.bus_dc.at[b, 'in_service']) or b not in res.index or not np.isfinite(res.at[b, 'vm_pu']):
                continue
            if 'electrisim_hidden' in net.bus_dc.columns and net.bus_dc.at[b, 'electrisim_hidden'] == True:
                continue   # a battery's cells: its EMT model stands for them
            self.vn[b] = float(net.bus_dc.at[b, 'vn_kv']) * 1e3
            self.v_bus[b] = float(res.at[b, 'vm_pu']) * self.vn[b]
            self.bus_node[b] = ckt.node(_label(net, 'bus_dc', b), self.v_bus[b])
            if not ('electrisim_aux' in net.bus_dc.columns and net.bus_dc.at[b, 'electrisim_aux'] == True)                     and not ('electrisim_hidden' in net.bus_dc.columns and net.bus_dc.at[b, 'electrisim_hidden'] == True):
                self.visible_buses.append(b)
        for rec in getattr(net, 'electrisim_dc_breakers', None) or []:
            if rec['closed'] and rec['target'] is not None:
                self._breakers_by_target.setdefault(rec['target'], []).append(rec)
        self._plan_converters()
        ac = AcBuilder(ckt, net, self.params, self.warnings, self.f_hz)
        ac.skip = self.skip
        ac.gen_rows = {str(r.get('name')): r for r in (self.in_data or {}).values()
                       if isinstance(r, dict) and str(r.get('typ', '')).startswith('Generator')}
        ac.profiles = {i: plan for (table, i), plan in self.profiles.items() if table == 'load'}
        self.ac = ac.build()
        self._cables()
        self._capacitors()
        self._sources()
        self._loads()
        self._ders()
        self._converters()
        self._pcs()
        self._dc_dc_converters()
        self._ssts()
        if 'b2b_vsc' in net and len(net.b2b_vsc):
            self.warnings.append('The B2B VSCs are left out of the EMT study: their two-pole DC side is not modelled yet.')
        return self

    def _plan_converters(self):
        """
        The DC/DC converters and SSTs modelled as such, and the load-flow
        elements they stand for (their stages' inputs, auxiliary VSCs, an SST's
        MV load and its grid-following inverter's current), left out.
        """
        net = self.net
        self.dcdc_plan, self.sst_plan = [], []

        def dc_ok(b):
            return b is not None and b in self.bus_node

        def ac_ok(b):
            if b is None or b not in net.bus.index or not bool(net.bus.at[b, 'in_service']) or b not in net.res_bus.index:
                return False
            return np.isfinite(net.res_bus.at[b, 'vm_pu'])

        for rec in getattr(net, 'electrisim_dc_dc_converters', None) or []:
            parts = [('load_dc', rec['input']), ('vsc', rec['vsc']), ('load_dc', rec['output_load'])]
            if not rec['in_service'] or not dc_ok(rec['bus_in']) or not dc_ok(rec['bus_out']):
                continue
            if rec['input'] is None or rec['input'] not in net.load_dc.index or not bool(net.load_dc.at[rec['input'], 'in_service']):
                continue
            self.dcdc_plan.append(rec)
            self.skip.update(p for p in parts if p[1] is not None)
        # The PCS: their load-flow generators left out, their EMT models standing for them.
        for rec in getattr(net, 'electrisim_pcs', None) or []:
            if rec['in_service'] and rec['index'] is not None:
                self.skip.add((rec['table'], rec['index']))
        # The sources and stores: their load-flow stand-ins left out.
        for rec in getattr(net, 'electrisim_ders', None) or []:
            parts = rec['parts']
            for table, key in (('vsc', 'vsc'), ('vsc', 'cells'), ('line_dc', 'line'), ('load_dc', 'load')):
                if parts.get(key) is not None:
                    self.skip.add((table, parts[key]))
        for rec in getattr(net, 'electrisim_ssts', None) or []:
            stages = {s['stage']: s for s in rec['stages']}
            rect, dcdc = stages.get('rectifier'), stages.get('dcdc')
            if not rec['in_service'] or rect is None or dcdc is None:
                continue
            if not ac_ok(rec['bus_mv']) or not dc_ok(rect['link']) or not dc_ok(rec['bus_lvdc']):
                continue
            self.sst_plan.append(rec)
            self.skip.update([rect['input'], rect['output'], dcdc['input'], dcdc['output']])
            inv = stages.get('inverter')
            if inv is not None and rec['inverter_mode'] == 'grid_following' and ac_ok(rec['bus_lvac']):
                self.skip.update([inv['input'], inv['output']])

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

    def _diode(self, li):
        """
        A DC diode: its forward voltage behind its on-resistance. One blocking
        in the load flow is built too, to conduct when its anode rises v_f
        above its cathode - a catcher's diode, when its shelf's own bus fails.
        """
        net, ckt = self.net, self.ckt
        rec = next((d for d in getattr(net, 'electrisim_dc_diodes', None) or [] if d.get('line') == li), None)
        fb, tb = int(net.line_dc.at[li, 'from_bus_dc']), int(net.line_dc.at[li, 'to_bus_dc'])
        if rec is None or not rec['in_service'] or fb not in self.bus_node or tb not in self.bus_node:
            return
        i0 = pe._electrisim_dc_diode_amps(net, rec)
        i0 = i0 if np.isfinite(i0) and i0 > 0 else 0.0
        # Through the breakers in front of it, as a cable is: a shelf feeder's breaker trips, then its diode.
        anode = self._terminal('line_dc', li, fb, i0)
        cathode = self._terminal('line_dc', li, tb, -i0)
        junction = ckt.node(f"{rec['label']} junction", self.v_bus[tb] + i0 * rec['r_on_ohm'])
        ckt.add_diode(anode, junction, rec['v_f_v'])
        k = ckt.add_rl(junction, cathode, rec['r_on_ohm'], 0.0, i0=i0)
        self.series.append(('DC diode', rec['label'], rec['id'], rec['label'], ('i_rl', k), 1.0))

    def _cables(self):
        net, ckt = self.net, self.ckt
        max_km = max(_f(self.params.get('max_section_km'), 1.0), 1e-3)
        for li in net.line_dc.index:
            if pe._electrisim_is_dc_diode(net, li):
                self._diode(li)
                continue
            if not bool(net.line_dc.at[li, 'in_service']) or ('line_dc', li) in self.skip:
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
            if cap.get('electrisim_der'):
                continue   # a supercapacitor: its own model, with its leakage
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
            if not bool(ld.at[li, 'in_service']) or bus not in self.bus_node or ('load_dc', li) in self.skip:
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
            plan = self.profiles.get(('load_dc', li))
            scale = [1.0]                # a load step's factor on its profile
            power = None
            if plan is not None:
                # Its power follows its profile, at each solve's own time.
                def power(t, plan=plan, scale=scale):
                    return plan['p_set'] * plan['follower'](t) * scale[0]
            func = dc_load_current(pw, vn / 1e3, *shares, v_min, power=power)
            if plan is not None:
                plan['measure'] = ('dc', None, None, None)      # its node and current, below
            i_load = p * 1e6 / self.v_bus[bus]
            term = self._terminal('load_dc', li, bus, i_load)
            c = _f(ld.at[li, 'filter_c_uf']) * 1e-6 if 'filter_c_uf' in ld.columns else 0.0
            l = _f(ld.at[li, 'filter_l_mh']) * 1e-3 if 'filter_l_mh' in ld.columns else 0.0
            r = max(_f(ld.at[li, 'filter_r_mohm']), 0.0) * 1e-3 if 'filter_r_mohm' in ld.columns else 0.0
            if c <= 0:
                # A converter-fed load has a DC-link capacitor (emt_solver.default_input_capacitance).
                c = default_input_capacitance(shares[0], p * 1e6, self.v_bus[bus])
                if c > 0:
                    self.default_filters.append(_label(net, 'load_dc', li))
            node = term
            if c > 0 or l > 0:
                # Its filter: R and L to its input, the capacitor there, at the drop its current leaves.
                v_in = self.v_bus[bus] - r * i_load
                node = ckt.node(f'{_label(net, "load_dc", li)} input', v_in)
                ckt.add_rl(term, node, r, l, i0=i_load)
                if c > 0:
                    ckt.add_c(0, node, c, w0=-v_in)
            k = ckt.add_nonlinear(node, 0, func)
            if plan is not None:
                plan['measure'] = ('dc', node, k, None)
            label = _label(net, 'load_dc', li)
            if not aux:
                self.loads.append({'label': label, 'id': _row_id(net, 'load_dc', li), 'node': node, 'p': pw, 'func': func,
                                   'bus': bus, 'vn': vn, 'constant_power': shares[0] > 0,
                                   'scale': scale if plan is not None else None})
                self.series.append(('DC load', label, _row_id(net, 'load_dc', li), label, ('i_nl', k), 1.0))
            elif role == 'input':
                self.series.append(('Converter input', label, '', label, ('i_nl', k), 1.0))

    def _ders(self):
        """Each source and store, its EMT model at its bus, from its load-flow state."""
        net = self.net
        for rec in getattr(net, 'electrisim_ders', None) or []:
            bus = rec['bus']
            if not rec['obj'].in_service or bus not in self.bus_node:
                continue
            v, p = pe._electrisim_der_power(net, rec)
            model = emt_der.build(self.ckt, rec['obj'], rec['label'], self.bus_node[bus], v, p / max(v, 1e-6))
            model.id = rec['id']
            self.ders.append((rec, model))

    def _pcs(self):
        """
        Each PCS: its source's EMT model on its DC link, and the VSC between
        that link and its AC bus - grid-forming, or grid-following in power
        mode (a PV array's on its DC voltage, which its MPPT moves) - rated as
        the PCS, its reactor 0.15 pu, its current limit its own.
        """
        net = self.net
        block = _f(self.params.get('vsc_block_pu'), 0.8)
        for rec in getattr(net, 'electrisim_pcs', None) or []:
            if not rec['in_service'] or rec['bus'] not in self.ac.nodes:
                continue
            res = net[f"res_{rec['table']}"]
            if rec['index'] not in res.index:
                continue
            p_ac, q_ac = _f(res.at[rec['index'], 'p_mw']), _f(res.at[rec['index'], 'q_mvar'])
            src = rec['source']
            obj = src['obj']
            p_dc = pe._electrisim_dc_dc_input_power(p_ac, rec['eta'], rec['p_nl_mw'])
            v_dc, _ = pe._electrisim_der_dc_point(obj, p_dc * 1e6)
            key = ('pcs', rec['name'])
            self.vn[key], self.v_bus[key] = obj.v_nominal(), v_dc
            term = self.ckt.node(f"{rec['label']} DC terminals", v_dc)
            model = emt_der.build(self.ckt, obj, src['label'], term, v_dc, p_dc * 1e6 / max(v_dc, 1e-6))
            model.id = src['id']
            self.ders.append(({'id': src['id'], 'coupling': 'pcs', 'bus': None, 'label': src['label']}, model))
            z = self.ac.vn[rec['bus']] ** 2 / (rec['s_rated'] * 1e6)
            kw = dict(p=-p_ac, q=-q_ac, p_dc=p_dc, rated_mva=rec['s_rated'], limit_pu=rec['k'], r_ohm=0.01 * z,
                      x_ohm=0.15 * z, mode_ac='q_mvar', model='average', eta=rec['eta'], p_nl_mw=rec['p_nl_mw'],
                      input_side='dc', current_loop_hz=rec.get('current_loop_hz', 500.0))
            if rec['table'] == 'gen':
                conv = GridFormingVsc(self, rec['label'], rec['bus'], key, term, 1e-4, block, mode_dc='p_mw',
                                      droop_pf=rec['droop_pf'], droop_qv=rec['droop_qv'], **kw)
                conv.control_mode = 'grid_forming'
            else:
                pv = obj.kind == 'PV Array'
                conv = Vsc(self, rec['label'], rec['bus'], key, term, 1e-4, block,
                           mode_dc='vm_pu' if pv else 'p_mw', **kw)
                if pv:
                    conv.v_dc_ref_fn = emt_der.VscMppt(model, v_dc)
                    conv.mppt = conv.v_dc_ref_fn
                conv.control_mode = 'grid_following'
            conv.pcs = True
            self._add_vsc(conv, rec['id'])

    def _der_on(self, bus):
        return next((m for r, m in self.ders if r['bus'] == bus and r['coupling'] == 'converter'), None)

    def _converters(self):
        net, ckt = self.net, self.ckt
        if 'vsc' not in net or not len(net.vsc):
            return
        block = _f(self.params.get('vsc_block_pu'), 0.8)
        thevenin = None
        w = 2.0 * math.pi * self.f_hz
        for vi in net.vsc.index:
            bus = int(net.vsc.at[vi, 'bus_dc'])
            if not bool(net.vsc.at[vi, 'in_service']) or bus not in self.bus_node or ('vsc', vi) in self.skip:
                continue
            aux = 'electrisim_aux' in net.vsc.columns and net.vsc.at[vi, 'electrisim_aux'] == True
            label = _label(net, 'vsc', vi)
            p = _f(net.res_vsc.at[vi, 'p_dc_mw']) if vi in net.res_vsc.index else 0.0
            i0 = -p * 1e6 / self.v_bus[bus]
            r_dc = max(_f(net.vsc.at[vi, 'r_dc_ohm']), 1e-6)
            term = self._terminal('vsc', vi, bus, -i0)
            if not aux and int(net.vsc.at[vi, 'bus']) in self.ac.nodes:
                # On the AC network: its average-value model, with its controls.
                self._add_vsc(Vsc.from_net(self, vi, label, bus, term, r_dc, block), _row_id(net, 'vsc', vi))
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

    def _dc_dc_converters(self):
        """Each DC/DC converter: a dual active bridge between its buses, with its controls."""
        net = self.net
        block = _f(self.params.get('vsc_block_pu'), 0.8)
        for rec in self.dcdc_plan:
            res_in = net.res_load_dc
            p_in = _f(res_in.at[rec['input'], 'p_dc_mw']) if rec['input'] in res_in.index else 0.0
            if rec['vsc'] is not None:
                p_out = -_f(net.res_vsc.at[rec['vsc'], 'p_dc_mw']) if rec['vsc'] in net.res_vsc.index else 0.0
                out = ('vsc', rec['vsc'])
            else:
                p_out = rec['p_set_mw']
                out = ('load_dc', rec['output_load'])
            b_in, b_out = rec['bus_in'], rec['bus_out']
            term_in = self._terminal('load_dc', rec['input'], b_in, p_in * 1e6 / self.v_bus[b_in])
            term_out = self._terminal(out[0], out[1], b_out, -p_out * 1e6 / self.v_bus[b_out])
            e = rec.get('emt') or {}
            control = rec.get('control', rec['mode'])
            conv = DcDc(self, rec['label'], b_in, b_out, term_in, term_out, p_in_mw=p_in, p_out_mw=p_out,
                        mode=rec['mode'], vm_out_pu=rec['vm_out_pu'], p_set_mw=rec['p_set_mw'], rated_mw=rec['rated_mw'],
                        eta=rec['eta'], p_nl_mw=rec['p_nl_mw'], bidirectional=rec['bidirectional'] or control == 'smoothing',
                        limit_pu=e.get('current_limit_pu', 1.2), model=e.get('model', 'average'),
                        switching_khz=e.get('switching_khz', 20.0), c_out_mf=e.get('c_out_mf', 0.0),
                        c_out_esr_mohm=e.get('c_out_esr_mohm', 0.0), c_out_esl_uh=e.get('c_out_esl_uh', 0.0),
                        c_in_esr_mohm=e.get('c_in_esr_mohm', 0.0), c_in_esl_uh=e.get('c_in_esl_uh', 0.0), block_pu=block,
                        droop_pu=rec.get('droop_percent', 0.0) / 100.0 if control == 'droop' else 0.0)
            conv.control_mode = control
            conv.bus_out = b_out
            store = self._der_on(b_in)
            if control == 'mppt' and store is not None and store.kind == 'PV Array':
                # Perturb and observe on its PV array's voltage, from the load flow's maximum power point.
                conv.mppt = emt_der.Mppt(store, conv, self.v_bus[b_in])
                conv.p_in_ref = conv.mppt
            elif control == 'smoothing' and store is not None and store.kind in ('Battery', 'Supercapacitor', 'Flywheel'):
                sm = rec['smoothing']
                loads = [(ld['node'], ld['func']) for ld in self.loads if ld['bus'] == b_out]
                mates = [r for r in self.dcdc_plan if r.get('control') == 'smoothing' and r['bus_out'] == b_out]
                total = sum(max(r['rated_mw'], 1e-9) for r in mates)
                smoother = emt_der.Smoothing(store, loads, max(sm['tau_s'], 1e-6), sm['soc_ref_percent'], sm['soc_gain'],
                                             conv.rated, max(rec['rated_mw'], 1e-9) / total)
                conv.p_ref = lambda t, state, f=smoother: f(t, state)
                self.smoothers.append((conv, smoother))
            self._add_dcdc(conv, rec['id'])

    def _add_dcdc(self, conv, row_id):
        conv.id = row_id
        self.dcdcs.append(conv)
        self.series.append(('DC/DC converter', conv.label, row_id, f'{conv.label} (input)', ('i_rl', conv.k_in), 1.0))
        self.series.append(('DC/DC converter', conv.label, row_id, f'{conv.label} (output)', ('i_rl', conv.k_out), 1.0))

    def _ssts(self):
        """
        Each SST: its rectifier, a VSC on its MV bus holding its DC link; its
        DC/DC stage, a dual active bridge holding its LV DC port; and its
        grid-following inverter, a VSC delivering its set power to its LV AC
        port. A grid-forming inverter stays a source behind its impedance, its
        input a constant-power load.
        """
        net = self.net
        block = _f(self.params.get('vsc_block_pu'), 0.8)
        for rec in self.sst_plan:
            e = rec.get('emt') or {}
            stages = {s['stage']: s for s in rec['stages']}
            rect, dcdc, inv = stages['rectifier'], stages['dcdc'], stages.get('inverter')
            limit, model = e.get('current_limit_pu', 1.2), e.get('model', 'average')
            link = rect['link']

            def stage_loss(inp):
                table, i = inp
                df = net[table]
                return (_f(df.at[i, 'electrisim_dcdc_eta'], 1.0) if 'electrisim_dcdc_eta' in df.columns else 1.0,
                        _f(df.at[i, 'electrisim_dcdc_p_nl_mw'], 0.0) if 'electrisim_dcdc_p_nl_mw' in df.columns else 0.0)

            def reactor(bus, rated):
                z = (float(net.bus.at[bus, 'vn_kv']) * 1e3) ** 2 / (max(rated, 1e-3) * 1e6)
                return 0.005 * z, 0.15 * z

            # Its rectifier.
            mv_load, r_vsc = rect['input'][1], rect['output'][1]
            p, q = (_f(net.res_load.at[mv_load, 'p_mw']), _f(net.res_load.at[mv_load, 'q_mvar'])) \
                if mv_load in net.res_load.index else (0.0, 0.0)
            p_dc = _f(net.res_vsc.at[r_vsc, 'p_dc_mw']) if r_vsc in net.res_vsc.index else -p
            eta, p_nl = stage_loss(rect['input'])
            r_ohm, x_ohm = reactor(rec['bus_mv'], rect['rated_mw'])
            label = f"{rec['label']} rectifier"
            conv = Vsc(self, label, rec['bus_mv'], link, self.bus_node[link], 1e-6, block, p=p, q=q, p_dc=p_dc,
                       rated_mva=rect['rated_mw'], limit_pu=limit, r_ohm=r_ohm, x_ohm=x_ohm, mode_dc='vm_pu',
                       mode_ac='q_mvar', model=model, switching_khz=e.get('switching_khz', 5.0),
                       eta=eta, p_nl_mw=p_nl, input_side='ac', current_loop_hz=e.get('current_loop_hz', 500.0),
                       c_link_esr_mohm=e.get('rect_dc_link_esr_mohm', 0.0), c_link_esl_uh=e.get('rect_dc_link_esl_uh', 0.0))
            self._add_vsc(conv, rec['id'])

            # Its DC/DC stage.
            d_in, d_vsc = dcdc['input'][1], dcdc['output'][1]
            p_in = _f(net.res_load_dc.at[d_in, 'p_dc_mw']) if d_in in net.res_load_dc.index else 0.0
            p_out = -_f(net.res_vsc.at[d_vsc, 'p_dc_mw']) if d_vsc in net.res_vsc.index else 0.0
            eta, p_nl = stage_loss(dcdc['input'])
            lv = rec['bus_lvdc']
            term_out = self._terminal('vsc', d_vsc, lv, -p_out * 1e6 / self.v_bus[lv])
            vm_out = _f(net.vsc.at[d_vsc, 'control_value_dc'], 1.0) if 'control_value_dc' in net.vsc.columns else 1.0
            conv = DcDc(self, f"{rec['label']} DC/DC", link, lv, self.bus_node[link], term_out, p_in_mw=p_in,
                        p_out_mw=p_out, mode='voltage', vm_out_pu=vm_out, rated_mw=dcdc['rated_mw'], eta=eta,
                        p_nl_mw=p_nl, bidirectional=True, limit_pu=limit, model=model,
                        switching_khz=e.get('dcdc_switching_khz', 20.0), block_pu=block,
                        c_in_esr_mohm=e.get('dcdc_c_in_esr_mohm', 0.0), c_in_esl_uh=e.get('dcdc_c_in_esl_uh', 0.0),
                        c_out_esr_mohm=e.get('dcdc_c_out_esr_mohm', 0.0), c_out_esl_uh=e.get('dcdc_c_out_esl_uh', 0.0))
            self._add_dcdc(conv, rec['id'])

            # Its grid-following inverter.
            if inv is None or inv['input'] not in self.skip:
                continue
            i_in, sgen = inv['input'][1], inv['output'][1]
            p_s, q_s = (_f(net.res_sgen.at[sgen, 'p_mw']), _f(net.res_sgen.at[sgen, 'q_mvar'])) \
                if sgen in net.res_sgen.index else (0.0, 0.0)
            p_dc = _f(net.res_load_dc.at[i_in, 'p_dc_mw']) if i_in in net.res_load_dc.index else p_s
            eta, p_nl = stage_loss(inv['input'])
            r_ohm, x_ohm = reactor(rec['bus_lvac'], inv['rated_mw'])
            term = self._terminal('load_dc', i_in, lv, p_dc * 1e6 / self.v_bus[lv])
            conv = Vsc(self, f"{rec['label']} inverter", rec['bus_lvac'], lv, term, 1e-6, block, p=-p_s, q=-q_s,
                       p_dc=p_dc, rated_mva=inv['rated_mw'], limit_pu=limit, r_ohm=r_ohm, x_ohm=x_ohm, mode_dc='p_mw',
                       mode_ac='q_mvar', model=model, switching_khz=e.get('switching_khz', 5.0),
                       eta=eta, p_nl_mw=p_nl, input_side='dc', current_loop_hz=e.get('current_loop_hz', 500.0),
                       c_link_esr_mohm=e.get('inv_dc_link_esr_mohm', 0.0), c_link_esl_uh=e.get('inv_dc_link_esl_uh', 0.0))
            self._add_vsc(conv, rec['id'])

    def _add_vsc(self, conv, row_id):
        conv.id = row_id
        self.vscs.append(conv)
        self.series.append(('VSC', conv.label, row_id, f'{conv.label} (DC side)', ('i_rl', conv.k_dc_out), -1.0))
        self.ac.series.append(('VSC', conv.label, row_id, f'{conv.label} (AC side)',
                               [[('i_rl', k, None, 1.0)] for k in conv.k_e]))

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
        # AC loads following a profile: their impedance's change for the next step.
        if self.ac.profiled:
            self.ac.follow_profiles(t, state)
        changed = False
        if self.ac.island is not None and self.ac.island_control(t, state):
            changed = True
        for _, model in self.ders:
            model.control(t, state)
        for conv in self.vscs + self.dcdcs:
            r = conv.control(t, state)
            if r == 'soft':
                changed = changed or 'soft'
            elif r:
                changed = True
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
    if a_late < RIPPLE_PU * vn:
        # Held, and small: the AC network's ripple through its converters (six times its frequency from a
        # rectifier), not a swing of its own. The campus's islanded racks read 'oscillates' at 0.25 V on 800 V.
        return 'steady ripple'
    return 'oscillates'


METHOD = ("A time-domain simulation of the AC and DC networks from their load-flow state. AC: three-phase, grids behind their positive- and zero-sequence impedance, lines as coupled pi sections, transformers by their vector group, loads as constant impedance, inverter-based generation as constant current, generators behind their subtransient reactance, started in their own steady state; an AC fault clears at each phase's current zero. DC: cables as pi sections, "
          "DC loads by their load-flow model behind their input filters, breakers that trip on overcurrent and "
          "open into their surge arresters. A VSC on the AC network is an average-value model behind its reactor "
          "and isolating transformer, with its DC link, under its controls - PLL, dq current control, its DC "
          "voltage or power, its reactive power or AC voltage - limited to its current limit, and blocking, its "
          "diodes left, on DC undervoltage or overcurrent. A DC/DC converter is a dual active bridge holding its "
          "output voltage or delivering its set power within its current limit, blocking on undervoltage; an SST "
          "is a VSC rectifier holding its DC link, a dual active bridge holding its LV DC port, and a grid-following "
          "inverter VSC (a grid-forming one stays a source behind its impedance). Each converter is an average-value "
          "or a switching model; a switching one switches at exact instants, its controller sampling twice per "
          "period. A constant-power load with no minimum voltage set "
          "draws constant current below 0.8 pu; one with no input capacitance is given 4 ms x P / V^2. "
          "Pole-to-pole faults.")


def _plan_profiles(net, params, in_data, warnings_out):
    """
    The loads that follow a profile from the diagram's library, AC and DC:
    each set, for the load flow the run starts from, to its profile's value
    profile_start_s into it, so the run starts in steady state. Returns
    {(table, index): plan} - its set power, its profile as a function of the
    run's time, its name.
    """
    library, problems = lp.library_from_params(params)
    warnings_out.extend(problems)
    if not library:
        return {}
    t0 = max(_f(params.get('profile_start_s'), 0.0), 0.0)
    repeat = params.get('profile_repeat', True) not in (False, 'false', 'False', 0, '0')
    plan = {}
    for table, assignments in (('load', lp.load_assignments(in_data)), ('load_dc', lp.dc_load_assignments(in_data))):
        if table not in net or not len(net[table]):
            continue
        df = net[table]
        for name, a in assignments.items():
            prof = library.get(a['profile_id'])
            if prof is None:
                warnings_out.append(f"{a['display_name']}: its load profile is not in the library, so it does not follow one.")
                continue
            hit = df.index[df['name'].astype(str) == name]
            if not len(hit):
                continue
            i = int(hit[0])
            follower = lp.Follower(prof['t'], prof['p'], t0, repeat)
            f0 = follower(0.0)
            if table == 'load':
                p_set, q_set = _f(df.at[i, 'p_mw']), _f(df.at[i, 'q_mvar'])
                df.at[i, 'p_mw'] = p_set * f0
                df.at[i, 'q_mvar'] = q_set * f0 if a['q_mode'] == 'pf' else q_set
                item = {'p_set': p_set, 'q_set': q_set, 'q_mode': a['q_mode']}
            else:
                rated = 'electrisim_p_rated_mw' in df.columns and np.isfinite(_f(df.at[i, 'electrisim_p_rated_mw'], np.nan))
                p_set = _f(df.at[i, 'electrisim_p_rated_mw']) if rated else _f(df.at[i, 'p_dc_mw'])
                df.at[i, 'p_dc_mw'] = p_set * f0
                if rated:
                    df.at[i, 'electrisim_p_rated_mw'] = p_set * f0
                item = {'p_set': p_set}
            item.update(follower=follower, profile=prof['name'], label=a['display_name'], f0=f0)
            plan[(table, i)] = item
    if plan:
        warnings_out.append(f'Loads following a profile start {t0:g} s into it, at its value there; '
                            + ('it repeats after its end.' if repeat else 'it holds its last value after its end.'))
    return plan


def emt_study(net, params, in_data=None):
    params = params or {}
    warnings_out = []
    profiles = _plan_profiles(net, params, in_data, warnings_out)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pe._electrisim_runpp(net, algorithm='nr', calculate_voltage_angles=True, init='auto')
    except Exception as exc:   # noqa: BLE001
        return json.dumps({'error': True, 'message': f'The load flow the EMT study starts from did not solve: {exc}'})
    warnings_out.extend(getattr(net, 'warnings', []) or [])
    if not len(net.bus) and ('bus_dc' not in net or not len(net.bus_dc)):
        return json.dumps({'error': True, 'message': 'There is no network to study.', 'warnings': warnings_out})
    builder = _EmtBuilder(net, params, warnings_out, profiles)
    builder.in_data = in_data
    b = builder.build()
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
            if load.get('scale') is not None:
                # Following a profile: the step scales it.
                ckt.at(t_s, lambda st, sc=load['scale'], f=factor: sc.__setitem__(0, sc[0] * f))
            else:
                ckt.at(t_s, lambda st, pw=load['p'], f=factor: pw.__setitem__(0, pw[0] * f))
            event_times.append(t_s)

    # A step in the PV arrays' irradiance.
    if str(params.get('pv_step_wm2') or '').strip() not in ('', 'none'):
        pvs = [m for _, m in b.ders if m.kind == 'PV Array']
        if not pvs:
            warnings_out.append('There is no PV array for the irradiance step.')
        else:
            t_pv = _f(params.get('pv_step_time_ms'), 5.0) * 1e-3
            g = _f(params.get('pv_step_wm2'), 1000.0)
            ckt.at(t_pv, lambda st, ms=pvs, g=g: [m.step_irradiance(g) for m in ms])
            event_times.append(t_pv)

    # Islanding: the external grids' breakers open, each phase at its current's zero.
    if b.ac.island is not None:
        event_times.append(b.ac.island['t'])

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

    # The grid's voltage following a profile (IEEE 2800's ride-through envelope, or a table): its
    # source's electromotive force stepped, a share of its own, behind its short-circuit impedance.
    points, problems = gvp.profile_points(params)
    warnings_out.extend(problems)
    if points:
        target = str(params.get('grid_voltage_target') or '').strip()
        grids = [(gi, g, label) for gi, g, label in b.ac.grid_groups
                 if not target or target in (label, str(net.ext_grid.at[gi, 'name']))]
        if target and not grids:
            warnings_out.append(f"Grid voltage profile: no External Grid '{target}'; every one follows it.")
            grids = list(b.ac.grid_groups)
        t0 = _f(params.get('grid_voltage_start_ms'), 20.0) * 1e-3
        for _, g, _ in grids:
            for t_rel, v in points:
                ckt.at(t0 + t_rel, lambda st, g=g, v=v: st.scale_coupled(g, v))
        if grids:
            event_times.extend(t0 + t_rel for t_rel, _ in points if t0 + t_rel < t_end)
            warnings_out.append(gvp.describe(points, t0))
        else:
            warnings_out.append('Grid voltage profile: the network has no External Grid to apply it to.')

    ckt.add_controller(b.controller)
    t_fine = min(t_end, (max(event_times) if event_times else 0.0) + 0.01)
    with np.errstate(all='ignore'):
        # A switching converter keeps its fine step throughout: fifty to its switching period at most.
        t_sw = min((1.0 / c.f_sw for c in b.vscs + b.dcdcs if c.model == "switching"), default=None)
        dt_coarse = min(10 * dt, 2e-5) if t_sw is None else max(dt, min(10 * dt, 2e-5, t_sw / 50))
        sim = ckt.simulate(t_end, dt, dt_coarse=dt_coarse, t_fine=t_fine)
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
              + [{'label': c.label, 't_ms': _round(c.blocked_at * 1e3)} for c in b.vscs + b.dcdcs if c.blocked_at is not None],
              'profiled_loads': [_profile_result(plan, sim, net, key, b.ac.w) for key, plan in profiles.items()],
              'converters': [_converter_result(c, min(event_times, default=t_end)) for c in b.vscs]
              + [_dcdc_result(c, min(event_times, default=t_end)) for c in b.dcdcs],
              'settings': {'time_step_us': dt * 1e6, 'duration_ms': t_end * 1e3,
                           'max_section_km': _f(params.get('max_section_km'), 1.0),
                           'vsc_block_pu': _f(params.get('vsc_block_pu'), 0.8)}}
    if fault:
        i = sim['i_sw'][:, fault['sw']]
        j = int(np.argmax(i))
        after = t >= fault['t']
        di = np.diff(i[after]) / np.maximum(np.diff(t[after]), 1e-15) if after.sum() > 1 else np.array([0.0])
        # Ik as the DC fault study takes it: the mean over the run's last AC period (a VSC's diodes ripple at 6 f).
        last = t >= t[-1] - 1.0 / b.f_hz
        ik = float(np.trapezoid(i[last], t[last]) / max(t[last][-1] - t[last][0], 1e-12)) if last.sum() > 1 else float(i[-1])
        result['fault'] = {'bus': _label(net, 'bus_dc', fault['bus']), 'id': _row_id(net, 'bus_dc', fault['bus']),
                           't_ms': fault['t'] * 1e3, 'ip_ka': _round(i[j] * 1e-3), 'tp_ms': _round(t[j] * 1e3),
                           'didt_max_ka_per_ms': _round(float(np.max(di)) * 1e-6), 'i_final_ka': _round(i[-1] * 1e-3),
                           'ik_ka': _round(ik * 1e-3)}
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
    if b.ac.machines:
        result['machines'] = [_machine_result(m) for m in b.ac.machines]
    result['ders'] = [_der_result(rec, m) for rec, m in b.ders]
    if b.ac.island is not None:
        result['island'] = {'t_ms': _round(b.ac.island['t'] * 1e3),
                            'opened_ms': [_round(x * 1e3) if x is not None else None for x in b.ac.island['opened']]}
    for conv, sm in b.smoothers:
        tr = np.array(sm.trace) if sm.trace else np.zeros((0, 3))
        row = next(r for r in result['converters'] if r['label'] == conv.label)
        if len(tr) > 1:
            dt = np.maximum(np.diff(tr[:, 0]), 1e-12)
            # The feed: the racks less every smoothing store on their bus (they sample together).
            mates = [m for c, m in b.smoothers if c.bus_out == conv.bus_out and len(m.trace) == len(sm.trace)]
            feed = tr[:, 1] - sum(np.array(m.trace)[:, 2] for m in mates)
            row['smoothing'] = {'rack_peak_mw': _round(float(np.max(tr[:, 1])) / 1e6),
                                'feed_peak_mw': _round(float(np.max(feed)) / 1e6),
                                'rack_ramp_mw_s': _round(float(np.max(np.abs(np.diff(tr[:, 1])) / dt)) / 1e6),
                                'feed_ramp_mw_s': _round(float(np.max(np.abs(np.diff(feed)) / dt)) / 1e6),
                                'store_peak_mw': _round(float(np.max(np.abs(tr[:, 2]))) / 1e6),
                                'limited_ms': _round(sm.limited_time * 1e3)}
    # A VSC's DC voltage loop, some 30 Hz, swings slower than the network: two of its periods to tell.
    min_span = VSC_VERDICT_SPAN if b.vscs else 1e-3
    # Loads following a profile keep the voltages moving: whether a swing settles cannot be told from them.
    judge = not profiles
    if profiles and any(load['constant_power'] for load in b.loads):
        warnings_out.append('Constant-power loads: no stability verdict while loads follow a profile, which keeps '
                            'their voltages moving; run without profiles, with a load step, to see whether they settle.')
    for load in b.loads:
        v = sim['v'][:, load['node']]
        result['loads'].append({'label': load['label'], 'id': load['id'], 'constant_power': load['constant_power'],
                                'v_min_pu': _round(float(np.min(v)) / load['vn']),
                                'verdict': _oscillation_verdict(t, v, t_last, load['vn'], min_span)
                                if load['constant_power'] and judge else None})
        if load['constant_power'] and result['loads'][-1]['verdict'] == 'oscillates, growing':
            warnings_out.append(f"DC load {load['label']}: its voltage swings ever wider - a constant-power load "
                                "beyond its stability limit; more DC-link capacitance or a stiffer supply steadies it.")
    if any(l['verdict'] == 'too short to tell' for l in result['loads']):
        warnings_out.append(f'Constant-power loads: run at least {VSC_VERDICT_SPAN * 1e3:.0f} ms after the last event to tell '
                            "whether they settle - the VSCs' DC voltage loops swing at some 30 Hz.")
    if b.default_filters:
        warnings_out.append(DEFAULT_INPUT_WARNING.format(', '.join(b.default_filters)))
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


def _machine_result(m):
    """A synchronous machine: its power, frequency, rotor angle, internal voltage and mechanical power."""
    tr = np.array(m.trace) if m.trace else np.zeros((0, 7))
    out = {'label': m.label, 'id': m.row_id, 'governor': bool(m.gov), 'exciter': bool(m.exc)}
    if not len(tr):
        return out
    out.update(p_start_mw=_round(tr[0, 1]), p_end_mw=_round(tr[-1, 1]), q_end_mvar=_round(tr[-1, 2]),
               f_end_hz=_round(tr[-1, 3]), f_min_hz=_round(float(np.min(tr[:, 3]))), f_max_hz=_round(float(np.max(tr[:, 3]))),
               delta_end_deg=_round(math.degrees(tr[-1, 4] - tr[0, 4])), e_end_pu=_round(tr[-1, 5]),
               pm_start_mw=_round(tr[0, 6]), pm_end_mw=_round(tr[-1, 6]),
               trace={'t_ms': [_round(x * 1e3) for x in tr[::max(1, len(tr) // 400), 0]],
                      'p_mw': [_round(x) for x in tr[::max(1, len(tr) // 400), 1]],
                      'f_hz': [_round(x) for x in tr[::max(1, len(tr) // 400), 3]]})
    return out


def _converter_result(conv, t_event):
    """
    A VSC's active and reactive power out, DC voltage and current through the
    run: its P and Q over its first cycle (before the first event) and its last.
    """
    tr = np.array(conv.trace) if conv.trace else np.zeros((0, 5))
    out = {'label': conv.label, 'id': getattr(conv, 'id', ''), 'kind': 'PCS' if getattr(conv, 'pcs', False) else 'VSC',
           'control': getattr(conv, 'control_mode', None), 'model': conv.model,
           'switching_khz': _round(conv.f_sw / 1e3) if getattr(conv, 'f_sw', None) else None,
           'rated_mva': _round(conv.s_rated / 1e6), 'current_limit_ka': _round(conv.i_max / SQ2 / 1e3),
           'blocked_ms': _round(conv.blocked_at * 1e3) if conv.blocked_at is not None else None,
           'limited_ms': _round(conv.limited_time * 1e3)}
    if len(tr):
        # P and Q over its first and its last cycle: a switching model's samples carry its ripple.
        cycle = 2 * math.pi / conv.w0
        first = tr[:, 0] <= max(min(tr[0, 0] + cycle, t_event), tr[0, 0])
        last = tr[:, 0] >= tr[-1, 0] - cycle
        out.update(p_start_mw=_round(float(np.mean(tr[first, 1])) / 1e6),
                   p_end_mw=_round(float(np.mean(tr[last, 1])) / 1e6),
                   q_start_mvar=_round(float(np.mean(tr[first, 2])) / 1e6),
                   q_end_mvar=_round(float(np.mean(tr[last, 2])) / 1e6),
                   v_dc_min_kv=_round(float(np.min(tr[:, 3])) / 1e3), v_dc_end_kv=_round(tr[-1, 3] / 1e3),
                   i_peak_ka=_round(float(np.max(tr[:, 4])) / SQ2 / 1e3))
        # Its rms current over its last cycle: at a fault, its current limit.
        out['i_end_ka'] = _round(float(np.mean(tr[last, 4])) / SQ2 / 1e3)
    if getattr(conv, 'freq_trace', None):
        f = np.array(conv.freq_trace)
        out.update(f_min_hz=_round(float(np.min(f[:, 1]))), f_max_hz=_round(float(np.max(f[:, 1]))),
                   f_end_hz=_round(float(f[-1, 1])))
        k = max(1, len(f) // 400)
        out['frequency'] = {'t_ms': [_round(x * 1e3) for x in f[::k, 0]], 'f_hz': [_round(x) for x in f[::k, 1]]}
    return out


def _profile_result(plan, sim, net, key, w):
    """
    A load following a profile: its profile's power at the run's start, its
    least and most through the run, and over the run's last cycle both its
    profile's mean and the power it drew.
    """
    table, i = key
    t = sim['t']
    last = t >= t[-1] - 2 * math.pi / w
    times = np.linspace(0.0, t[-1], 401)
    p = plan['p_set'] * np.array([plan['follower'](x) for x in times])
    p_last = plan['p_set'] * np.array([plan['follower'](x) for x in t[last]])
    out = {'label': plan['label'], 'id': _row_id(net, table, i), 'kind': 'DC load' if table == 'load_dc' else 'Load',
           'profile': plan['profile'], 'p_set_mw': _round(plan['p_set']), 'p_start_mw': _round(p[0]),
           'p_min_mw': _round(float(p.min())), 'p_max_mw': _round(float(p.max())),
           'p_end_mw': _round(_mean(t[last], p_last)), 'p_drawn_end_mw': None}
    kind, node, k, parts = plan.get('measure', (None, None, None, None))
    if kind == 'dc' and node is not None:
        drawn = sim['v'][:, node] * sim['i_nl'][:, k]
    elif kind == 'ac':
        phase = sim['v'][:, node] - sim['v'][:, [k]]
        current = np.zeros_like(phase)
        for arr, ks in parts.items():
            for j, kk in enumerate(ks):
                current[:, j] += sim[arr][:, kk]
        drawn = np.sum(phase * current, axis=1)
    else:
        return out
    out['p_drawn_end_mw'] = _round(_mean(t[last], drawn[last]) / 1e6)
    return out


def _mean(t, y):
    """y's time-weighted mean over t (its value, for a single sample)."""
    return float(np.trapezoid(y, t) / (t[-1] - t[0])) if len(t) > 1 and t[-1] > t[0] else float(y[-1])


_DER_STATE = {'Battery': ('soc_percent', 100.0), 'Supercapacitor': ('v_cap_v', 1.0), 'Flywheel': ('speed_percent', 100.0),
              'SOFC': ('p_h2_atm', 1.0), 'PV Array': ('irradiance_wm2', 1.0)}


def _der_result(rec, model):
    """A source's or store's power, voltage and state through the run."""
    tr = np.array(model.trace) if model.trace else np.zeros((0, 4))
    key, scale = _DER_STATE[model.kind]
    out = {'label': model.label, 'id': rec['id'], 'kind': model.kind, 'coupling': rec['coupling']}
    if len(tr):
        p = tr[:, 1] * tr[:, 2]
        out.update(p_start_mw=_round(p[0] / 1e6), p_end_mw=_round(p[-1] / 1e6), v_end_kv=_round(tr[-1, 1] / 1e3),
                   v_min_kv=_round(float(np.min(tr[:, 1])) / 1e3), **{f'{key}_start': _round(tr[0, 3] * scale),
                                                                       f'{key}_end': _round(tr[-1, 3] * scale)})
        rows = emt_der.waveform(model.trace)
        out['waveform'] = {'t_ms': [_round(r[0] * 1e3) for r in rows], 'p_mw': [_round(r[1] * r[2] / 1e6) for r in rows],
                           'v_kv': [_round(r[1] / 1e3) for r in rows], key: [_round(r[3] * scale) for r in rows]}
    if getattr(model, 'limited_time', 0.0):
        out['limited_ms'] = _round(model.limited_time * 1e3)
    return out


def _dcdc_result(conv, t_event):
    """
    A DC/DC converter's power out over its first cycle (before the first
    event) and its last, its output voltage and current through the run.
    """
    tr = np.array(conv.trace) if conv.trace else np.zeros((0, 6))
    out = {'label': conv.label, 'id': getattr(conv, 'id', ''), 'kind': 'DC/DC', 'model': conv.model,
           'control': getattr(conv, 'control_mode', conv.mode),
           'switching_khz': _round(conv.f_sw / 1e3), 'rated_mw': _round(conv.rated / 1e6),
           'current_limit_ka': _round(conv.i_max / 1e3),
           'blocked_ms': _round(conv.blocked_at * 1e3) if conv.blocked_at is not None else None,
           'limited_ms': _round(conv.limited_time * 1e3)}
    if len(tr):
        cycle = 0.02
        first = tr[:, 0] <= max(min(tr[0, 0] + cycle, t_event), tr[0, 0])
        last = tr[:, 0] >= tr[-1, 0] - cycle
        out.update(p_start_mw=_round(float(np.mean(tr[first, 2])) / 1e6),
                   p_end_mw=_round(float(np.mean(tr[last, 2])) / 1e6),
                   v_dc_min_kv=_round(float(np.min(tr[:, 4])) / 1e3), v_dc_end_kv=_round(tr[-1, 4] / 1e3),
                   v_in_min_kv=_round(float(np.min(tr[:, 3])) / 1e3),
                   i_peak_ka=_round(float(np.max(np.abs(tr[:, 5]))) / 1e3))
    return out


SQ2 = math.sqrt(2.0)
VSC_VERDICT_SPAN = 0.06    # s: two periods of a VSC's DC voltage loop
RIPPLE_PU = 1e-3           # a held swing below this, peak to peak per unit of the load's voltage, is ripple


def _volts(w):
    """A voltage waveform in kV: the waveform helper scales by 1e-3, into 'i_ka'."""
    return {'t_ms': w['t_ms'], 'v_kv': w['i_ka']}
