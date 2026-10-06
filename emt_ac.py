"""
The three-phase AC network in the EMT study (emt_solver.Circuit), from the
load-flow state.

- External grids: their voltage behind their short-circuit impedance, positive
  and zero sequence (s_sc_max_mva, rx_max, x0x_max, r0x0_max), star grounded.
- Lines: pi sections with their phase-domain coupling, from positive- and
  zero-sequence R, X and C per km (self Z = (Z0 + 2 Z1) / 3, mutual
  (Z0 - Z1) / 3; C0 phase to ground, (C1 - C0) / 3 phase to phase).
- Two-winding transformers: three single-phase units on separate cores (no
  zero-sequence flux path between phases), each a coupled pair - leakage split
  between the windings, magnetising from i0, iron loss from pfe - connected by
  their vector group, the delta's phase shift included.
- Loads, motors and shunts: constant impedance at their load-flow power,
  ungrounded star. Static generators and storage: grid-following, a constant
  current at their load-flow power. Generators: their voltage behind their
  subtransient reactance.
- Bus couplers that are closed join their buses; an open switch on a line or
  transformer takes it out.

The run starts in the circuit's own AC steady state (emt_solver's phasor
solution at the grid frequency), the sources' voltages taken from the load
flow - rotated where a transformer's vector group shifts phase that the load
flow, given a shift of 0, did not.
"""

import math
import re

import numpy as np

from dc_fault_electrisim import _f, _label, _row_id
from emt_solver import R_BIAS

SQ2 = math.sqrt(2.0)
A120 = 2.0 * math.pi / 3.0
_VG = re.compile(r'^\s*(YN|Y|D|Z)(yn|y|d|z)(\d{1,2})?\s*$')
_SUPPORTED_CLOCKS = {('Y', 'Y'): (0, 6), ('D', 'D'): (0, 6), ('D', 'Y'): (1, 5, 7, 11), ('Y', 'D'): (1, 5, 7, 11)}


def parse_vector_group(vector_group, shift_degree=0.0):
    """
    (HV connection, HV neutral grounded, LV connection, LV neutral grounded,
    clock, assumed) from a vector group such as 'Dyn11' or 'YNd1'. Without a
    clock number, the load flow's shift (in 30-degree steps) gives it; a
    star-delta with neither is taken as clock 11.
    """
    m = _VG.match(str(vector_group or 'Dyn'))
    if not m:
        m = _VG.match('Dyn')
    hv, lv, clock = m.group(1), m.group(2), m.group(3)
    hv_c = 'D' if hv == 'D' else 'Y'
    lv_c = 'D' if lv == 'd' else 'Y'
    assumed = False
    if clock is None:
        shift = _f(shift_degree, 0.0)
        if abs(shift) > 1e-9:
            clock = int(round(shift / 30.0)) % 12
        else:
            clock = 0 if hv_c == lv_c else 11
            assumed = hv_c != lv_c
    clock = int(clock) % 12
    return hv_c, hv == 'YN', lv_c, lv == 'yn', clock, assumed


def winding_pairs(hv_c, lv_c, clock):
    """
    For each single-phase unit k: the HV winding's (from, to) phase indices and
    the LV's, and whether the LV winding is reversed - 'n' for the star point.
    Clock 11: LV leads HV by 30 degrees; clock 1: lags; 5 and 7 are 11 and 1
    reversed; 6 is 0 reversed.
    """
    base = {11: 11, 5: 11, 1: 1, 7: 1, 0: 0, 6: 0}.get(clock, 0 if hv_c == lv_c else 11)
    reverse = clock in (5, 7, 6)
    pairs = []
    for k in range(3):
        if hv_c == 'Y':
            hv_pair = (k, 'n')
        elif lv_c == 'Y' and base == 11:
            hv_pair = (k, (k + 1) % 3)
        elif lv_c == 'Y' and base == 1:
            hv_pair = (k, (k - 1) % 3)
        else:
            hv_pair = (k, (k + 1) % 3)
        if lv_c == 'Y':
            lv_pair = (k, 'n')
        elif hv_c == 'Y' and base == 11:
            lv_pair = (k, (k - 1) % 3)
        elif hv_c == 'Y' and base == 1:
            lv_pair = (k, (k + 1) % 3)
        else:
            lv_pair = (k, (k + 1) % 3)
        pairs.append((hv_pair, lv_pair, reverse))
    return pairs


def _sequence_to_phase(z1, z0):
    """Phase-domain self and mutual from positive- and zero-sequence values."""
    return (z0 + 2.0 * z1) / 3.0, (z0 - z1) / 3.0


def _matrix(self_, mutual):
    return np.full((3, 3), mutual) + np.eye(3) * (self_ - mutual)


class AcBuilder:
    """Builds the AC network into a circuit; knows its nodes and what to report."""

    def __init__(self, ckt, net, params, warnings_out, f_hz):
        self.ckt, self.net, self.params, self.warnings = ckt, net, params, warnings_out
        self.w = 2.0 * math.pi * f_hz
        self.f_hz = f_hz
        self.nodes = {}          # AC bus -> [phase a, b, c] nodes
        self.vn = {}             # AC bus -> nominal line voltage (V)
        self.offset = {}         # AC bus -> radians the EMT angle leads the load flow's
        self.series = []         # (kind, name, id, label, [(array, group or index, column or None, sign)] per phase)
        self.trafos = []
        self.left_out = set()
        self.skip = set()        # (table, index): elements a converter's EMT model stands for
        self.profiles = {}       # load index -> its profile's plan (emt_electrisim._plan_profiles)
        self.profiled = []       # loads following a profile: their nodes, correction sources and plan

    # --- buses and the phase shifts ------------------------------------------------

    def build(self):
        net, ckt = self.net, self.ckt
        aux = net.bus['electrisim_aux'] == True if 'electrisim_aux' in net.bus.columns else None
        for b in net.bus.index:
            if not bool(net.bus.at[b, 'in_service']) or b not in net.res_bus.index:
                continue
            if aux is not None and aux.at[b]:
                continue
            if not np.isfinite(net.res_bus.at[b, 'vm_pu']):
                continue
            label = _label(net, 'bus', b)
            self.nodes[b] = [ckt.node(f'{label} {p}') for p in 'abc']
            self.vn[b] = float(net.bus.at[b, 'vn_kv']) * 1e3
        if not self.nodes:
            return self
        self._open_switches = self._switches()
        self._phase_offsets()
        self._grids()
        self._lines()
        self._trafos()
        self._loads()
        self._generation()
        for table in ('trafo3w', 'impedance', 'ward', 'xward', 'dcline'):
            if table in net and len(net[table]) and net[table]['in_service'].astype(bool).any():
                self.left_out.add(table)
        if self.left_out:
            self.warnings.append('Left out of the EMT study so far: ' + ', '.join(sorted(self.left_out)) + '.')
        return self

    def _switches(self):
        """Close bus couplers; return the lines and transformers an open switch takes out."""
        net, ckt = self.net, self.ckt
        out = set()
        for si in net.switch.index:
            closed = bool(net.switch.at[si, 'closed'])
            et, el, bus = net.switch.at[si, 'et'], int(net.switch.at[si, 'element']), int(net.switch.at[si, 'bus'])
            if et == 'b':
                if closed and bus in self.nodes and el in self.nodes:
                    for k in range(3):
                        ckt.add_r(self.nodes[bus][k], self.nodes[el][k], 1e-4)
            elif not closed and et in ('l', 't'):
                out.add(('line' if et == 'l' else 'trafo', el))
        return out

    def _phase_offsets(self):
        """
        Each bus's angle in the EMT model less its load-flow angle: a
        transformer whose vector group shifts phase the load flow did not
        (its shift_degree) moves the buses beyond it.
        """
        net = self.net
        adj = {b: [] for b in self.nodes}
        for li in net.line.index:
            fb, tb = int(net.line.at[li, 'from_bus']), int(net.line.at[li, 'to_bus'])
            if fb in adj and tb in adj:
                adj[fb].append((tb, 0.0))
                adj[tb].append((fb, 0.0))
        for si in net.switch.index:
            if net.switch.at[si, 'et'] == 'b' and bool(net.switch.at[si, 'closed']):
                a, b = int(net.switch.at[si, 'bus']), int(net.switch.at[si, 'element'])
                if a in adj and b in adj:
                    adj[a].append((b, 0.0))
                    adj[b].append((a, 0.0))
        for ti in net.trafo.index:
            hv, lv = int(net.trafo.at[ti, 'hv_bus']), int(net.trafo.at[ti, 'lv_bus'])
            if hv not in adj or lv not in adj:
                continue
            hv_c, _, lv_c, _, clock, _ = parse_vector_group(net.trafo.at[ti, 'vector_group'] if 'vector_group' in net.trafo.columns else 'Dyn',
                                                           net.trafo.at[ti, 'shift_degree'] if 'shift_degree' in net.trafo.columns else 0.0)
            lf_shift = math.radians(_f(net.trafo.at[ti, 'shift_degree'], 0.0)) if 'shift_degree' in net.trafo.columns else 0.0
            # The LV side lags the HV by clock x 30 degrees; the load flow lagged it by shift_degree.
            d = -clock * math.pi / 6.0 + lf_shift
            adj[hv].append((lv, d))
            adj[lv].append((hv, -d))
        roots = [int(net.ext_grid.at[i, 'bus']) for i in net.ext_grid.index if int(net.ext_grid.at[i, 'bus']) in adj]
        roots += [b for b in adj if b not in roots]
        for r in roots:
            if r in self.offset:
                continue
            self.offset[r] = 0.0
            stack = [r]
            while stack:
                x = stack.pop()
                for y, d in adj[x]:
                    if y not in self.offset:
                        self.offset[y] = self.offset[x] + d
                        stack.append(y)

    def _v_phasor(self, b):
        """Bus b's phase-a voltage phasor (amplitude, V), in the EMT model's angles."""
        res = self.net.res_bus
        v_ph = float(res.at[b, 'vm_pu']) * self.vn[b] / math.sqrt(3.0)
        return SQ2 * v_ph * np.exp(1j * (math.radians(float(res.at[b, 'va_degree'])) + self.offset.get(b, 0.0)))

    def _injection_phasor(self, b, p_mw, q_mvar):
        """The phase-a current phasor (amplitude) of a three-phase injection of P + jQ into bus b."""
        v = self._v_phasor(b)
        return np.conj((p_mw + 1j * q_mvar) * 1e6 / 3.0 / (v / SQ2)) * SQ2

    @staticmethod
    def _three(ph):
        """Amplitudes and phases of a positive-sequence set from the phase-a phasor."""
        return np.full(3, abs(ph)), np.angle(ph) - np.arange(3) * A120

    # --- the elements ------------------------------------------------------------------

    def _grids(self):
        net, ckt, w = self.net, self.ckt, self.w
        for gi in net.ext_grid.index:
            b = int(net.ext_grid.at[gi, 'bus'])
            if not bool(net.ext_grid.at[gi, 'in_service']) or b not in self.nodes:
                continue
            vn = self.vn[b]
            s_sc = _f(net.ext_grid.at[gi, 's_sc_max_mva'], 0.0) if 's_sc_max_mva' in net.ext_grid.columns else 0.0
            if s_sc <= 0:
                s_sc = 1000.0
                self.warnings.append(f"External grid {_label(net, 'ext_grid', gi)} has no short-circuit power: "
                                     "1000 MVA is taken.")
            rx = _f(net.ext_grid.at[gi, 'rx_max'], 0.1) if 'rx_max' in net.ext_grid.columns else 0.1
            z1 = vn * vn / (s_sc * 1e6)
            x1 = z1 / math.sqrt(1 + rx * rx)
            r1 = rx * x1
            x0x = _f(net.ext_grid.at[gi, 'x0x_max'], 1.0) if 'x0x_max' in net.ext_grid.columns else 1.0
            r0x0 = _f(net.ext_grid.at[gi, 'r0x0_max'], rx) if 'r0x0_max' in net.ext_grid.columns else rx
            x0 = x0x * x1 if x0x > 0 else x1
            r0 = r0x0 * x0
            rs, rm = _sequence_to_phase(r1, r0)
            ls, lm = _sequence_to_phase(x1 / w, x0 / w)
            p = _f(net.res_ext_grid.at[gi, 'p_mw']) if gi in net.res_ext_grid.index else 0.0
            q = _f(net.res_ext_grid.at[gi, 'q_mvar']) if gi in net.res_ext_grid.index else 0.0
            i_ph = self._injection_phasor(b, p, q)
            e_ph = self._v_phasor(b) + complex(r1, x1) * i_ph
            amp, ph = self._three(e_ph)
            g = ckt.add_coupled([0, 0, 0], self.nodes[b], _matrix(rs, rm), _matrix(ls, lm), ac=(amp, w, ph))
            label = _label(net, 'ext_grid', gi)
            self.series.append(('External grid', label, _row_id(net, 'ext_grid', gi), label,
                                [('i_cp', g, k, 1.0) for k in range(3)]))

    def _lines(self):
        net, ckt, w = self.net, self.ckt, self.w
        max_km = max(_f(self.params.get('ac_max_section_km'), 50.0), 1e-3)
        for li in net.line.index:
            fb, tb = int(net.line.at[li, 'from_bus']), int(net.line.at[li, 'to_bus'])
            if not bool(net.line.at[li, 'in_service']) or fb not in self.nodes or tb not in self.nodes or ('line', li) in self._open_switches:
                continue
            km = _f(net.line.at[li, 'length_km'], 1.0)
            par = max(_f(net.line.at[li, 'parallel'], 1.0), 1.0) if 'parallel' in net.line.columns else 1.0
            col = lambda c, d: _f(net.line.at[li, c], d) if c in net.line.columns else d
            r1, x1, c1 = col('r_ohm_per_km', 0.0), col('x_ohm_per_km', 0.0), col('c_nf_per_km', 0.0)
            r0, x0, c0 = col('r0_ohm_per_km', r1), col('x0_ohm_per_km', x1), col('c0_nf_per_km', c1)
            r0 = r0 if r0 > 0 else r1
            x0 = x0 if x0 > 0 else x1
            c0 = c0 if c0 > 0 else c1
            n_sec = max(1, int(math.ceil(km / max_km - 1e-9)))
            sec = km / n_sec
            rs, rm = _sequence_to_phase(r1 * sec / par, r0 * sec / par)
            ls, lm = _sequence_to_phase(x1 * sec / par / w, x0 * sec / par / w)
            cg = c0 * 1e-9 * sec * par
            cl = max(c1 - c0, 0.0) / 3.0 * 1e-9 * sec * par
            label = _label(net, 'line', li)
            ends = [self.nodes[fb]] + [[ckt.node(f'{label} section {k} {p}') for p in 'abc'] for k in range(1, n_sec)] + [self.nodes[tb]]
            first = None
            for k in range(n_sec):
                g = ckt.add_coupled(ends[k], ends[k + 1], _matrix(rs, rm), _matrix(ls, lm))
                first = g if first is None else first
                for end in (ends[k], ends[k + 1]):
                    for ph in range(3):
                        if cg > 0:
                            ckt.add_c(0, end[ph], cg / 2.0)
                        if cl > 0:
                            ckt.add_c(end[ph], end[(ph + 1) % 3], cl / 2.0)
            self.series.append(('Line', label, _row_id(net, 'line', li), f'{label} (from end)',
                                [('i_cp', first, k, 1.0) for k in range(3)]))

    def _trafos(self):
        net, ckt, w = self.net, self.ckt, self.w
        for ti in net.trafo.index:
            hv, lv = int(net.trafo.at[ti, 'hv_bus']), int(net.trafo.at[ti, 'lv_bus'])
            if not bool(net.trafo.at[ti, 'in_service']) or hv not in self.nodes or lv not in self.nodes or ('trafo', ti) in self._open_switches:
                continue
            label = _label(net, 'trafo', ti)
            col = lambda c, d: _f(net.trafo.at[ti, c], d) if c in net.trafo.columns else d
            hv_c, hv_g, lv_c, lv_g, clock, assumed = parse_vector_group(
                net.trafo.at[ti, 'vector_group'] if 'vector_group' in net.trafo.columns else 'Dyn', col('shift_degree', 0.0))
            if assumed:
                self.warnings.append(f"Transformer {label}: its vector group gives no clock number and its load-flow "
                                     "shift is 0 - taken as clock 11.")
            s_ph = col('sn_mva', 1.0) * max(col('parallel', 1.0), 1.0) * 1e6 / 3.0
            v_hv, v_lv = col('vn_hv_kv', 1.0) * 1e3, col('vn_lv_kv', 1.0) * 1e3
            tap = (col('tap_pos', 0.0) - col('tap_neutral', 0.0)) * col('tap_step_percent', 0.0) / 100.0
            if str(net.trafo.at[ti, 'tap_side'] if 'tap_side' in net.trafo.columns else 'hv') == 'lv':
                v_lv *= 1.0 + tap
            else:
                v_hv *= 1.0 + tap
            vw1 = v_hv / math.sqrt(3.0) if hv_c == 'Y' else v_hv
            vw2 = v_lv / math.sqrt(3.0) if lv_c == 'Y' else v_lv
            zb1, zb2 = vw1 * vw1 / s_ph, vw2 * vw2 / s_ph
            vk, vkr = col('vk_percent', 6.0) / 100.0, col('vkr_percent', 0.5) / 100.0
            xk = math.sqrt(max(vk * vk - vkr * vkr, 1e-12))
            i0 = col('i0_percent', 0.0)
            xm = 100.0 / i0 if i0 > 0 else 1000.0
            n = vw2 / vw1
            lm1 = xm * zb1 / w
            r_m = np.diag([vkr / 2 * zb1, vkr / 2 * zb2])
            l_m = np.array([[lm1 + xk / 2 * zb1 / w, n * lm1], [n * lm1, n * n * lm1 + xk / 2 * zb2 / w]])
            neutral = {}
            for side, conn, grounded in (('hv', hv_c, hv_g), ('lv', lv_c, lv_g)):
                if conn == 'Y':
                    if grounded:
                        neutral[side] = 0
                    else:
                        neutral[side] = ckt.node(f'{label} {side} star point')
                        ckt.add_r(neutral[side], 0, R_BIAS)
            pfe = col('pfe_kw', 0.0) * 1e3 / 3.0
            groups = []
            pairs = winding_pairs(hv_c, lv_c, clock)
            for k, (hp, lp, rev) in enumerate(pairs):
                node = lambda bus, side, x: neutral[side] if x == 'n' else self.nodes[bus][x]
                a1, b1 = node(hv, 'hv', hp[0]), node(hv, 'hv', hp[1])
                a2, b2 = node(lv, 'lv', lp[0]), node(lv, 'lv', lp[1])
                if rev:
                    a2, b2 = b2, a2
                groups.append((ckt.add_coupled([a1, a2], [b1, b2], r_m, l_m), hp, a1, b1))
                if pfe > 0:
                    ckt.add_r(a1, b1, vw1 * vw1 / pfe)
            # Its HV line currents: what leaves each phase into the windings.
            phases = []
            for ph in range(3):
                terms = []
                for g, hp, _a1, _b1 in groups:
                    if hp[0] == ph:
                        terms.append(('i_cp', g, 0, 1.0))
                    elif hp[1] == ph:
                        terms.append(('i_cp', g, 0, -1.0))
                phases.append(terms)
            self.series.append(('Transformer', label, _row_id(net, 'trafo', ti), f'{label} (HV side)', phases))
            self.trafos.append({'label': label, 'clock': clock, 'groups': groups})

    def _impedance_load(self, b, p_mw, q_mvar, label, profile=None):
        """
        Constant impedance at its load-flow power, ungrounded star. Following a
        profile, its impedance changes with it: a current source per phase
        carries the change in its conductance and susceptance from their
        starting values, by its phase voltage and (for the susceptance) the
        line voltage a quarter cycle behind it.
        """
        ckt, w = self.ckt, self.w
        v_ll = float(self.net.res_bus.at[b, 'vm_pu']) * self.vn[b]
        star = ckt.node(f'{label} star point')
        ckt.add_r(star, 0, R_BIAS)
        parts = {'i_r': [], 'i_rl': [], 'i_c': []}       # its branches, phase by phase, for its measured power
        for k in range(3):
            node = self.nodes[b][k]
            if p_mw > 1e-12:
                parts['i_r'].append(ckt.add_r(node, star, v_ll * v_ll / (p_mw * 1e6)))
            if q_mvar > 1e-12:
                parts['i_rl'].append(ckt.add_rl(node, star, 0.0, v_ll * v_ll / (q_mvar * 1e6) / w))
            elif q_mvar < -1e-12:
                parts['i_c'].append(ckt.add_c(node, star, -q_mvar * 1e6 / (w * v_ll * v_ll)))
        if profile is not None:
            parts['i_src'] = [ckt.add_isrc(self.nodes[b][k], star) for k in range(3)]
            self.profiled.append({'nodes': self.nodes[b], 'star': star, 'v2': v_ll * v_ll,
                                  'p0': p_mw * 1e6, 'q0': q_mvar * 1e6, 'plan': profile,
                                  'ks': parts['i_src'], 'parts': parts})
            profile['measure'] = ('ac', self.nodes[b], star, parts)

    def follow_profiles(self, t, state):
        """Each profiled load's change in conductance and susceptance, as the currents its sources draw."""
        for ld in self.profiled:
            plan = ld['plan']
            f = plan['follower'](t)
            p = plan['p_set'] * f * 1e6
            q = (plan['q_set'] * f if plan['q_mode'] == 'pf' else plan['q_set']) * 1e6
            dg, db = (p - ld['p0']) / ld['v2'], (q - ld['q0']) / ld['v2']
            v = state.v[ld['nodes']]
            phase = v - state.v[ld['star']]
            behind = np.array([v[1] - v[2], v[2] - v[0], v[0] - v[1]]) / math.sqrt(3.0)
            i = dg * phase + db * behind
            for k in range(3):
                state.set_source_value(ld['ks'][k], float(i[k]))

    def _loads(self):
        net = self.net
        for table, res in (('load', 'res_load'), ('motor', 'res_motor'), ('shunt', 'res_shunt')):
            if table not in net or not len(net[table]):
                continue
            for i in net[table].index:
                b = int(net[table].at[i, 'bus'])
                if not bool(net[table].at[i, 'in_service']) or b not in self.nodes or i not in net[res].index:
                    continue
                if (table, i) in self.skip:
                    continue
                self._impedance_load(b, _f(net[res].at[i, 'p_mw']), _f(net[res].at[i, 'q_mvar']), _label(net, table, i),
                                     self.profiles.get(i) if table == 'load' else None)
        if 'asymmetric_load' in net and len(net.asymmetric_load):
            self.left_out.add('asymmetric loads')

    def _generation(self):
        net, ckt, w = self.net, self.ckt, self.w
        # Grid-following: a constant current at their load-flow power.
        for table, res in (('sgen', 'res_sgen'), ('storage', 'res_storage')):
            if table not in net or not len(net[table]):
                continue
            for i in net[table].index:
                b = int(net[table].at[i, 'bus'])
                if not bool(net[table].at[i, 'in_service']) or b not in self.nodes or i not in net[res].index:
                    continue
                if (table, i) in self.skip:
                    continue
                p, q = _f(net[res].at[i, 'p_mw']), _f(net[res].at[i, 'q_mvar'])
                if table == 'storage':
                    p, q = -p, -q          # storage results use the load convention
                amp, ph = self._three(self._injection_phasor(b, p, q))
                ks = [ckt.add_isrc(0, self.nodes[b][k], ac=(amp[k], w, ph[k])) for k in range(3)]
                label = _label(net, table, i)
                self.series.append(('Static generator' if table == 'sgen' else 'Storage', label,
                                    _row_id(net, table, i), label, [('i_src', k, None, 1.0) for k in ks]))
        # Synchronous: their voltage behind their subtransient reactance.
        for i in net.gen.index:
            b = int(net.gen.at[i, 'bus'])
            if not bool(net.gen.at[i, 'in_service']) or b not in self.nodes or i not in net.res_gen.index:
                continue
            vn = self.vn[b]
            sn = _f(net.gen.at[i, 'sn_mva'], 0.0) if 'sn_mva' in net.gen.columns else 0.0
            sn = sn if sn > 0 else max(abs(_f(net.res_gen.at[i, 'p_mw'])), 1.0) * 1.2
            xd = _f(net.gen.at[i, 'xdss_pu'], 0.2) if 'xdss_pu' in net.gen.columns else 0.2
            xd = xd if xd > 0 else 0.2
            x = xd * vn * vn / (sn * 1e6)
            r = _f(net.gen.at[i, 'rdss_ohm'], 0.0) if 'rdss_ohm' in net.gen.columns else 0.0
            r = r if r > 0 else 0.05 * x
            i_ph = self._injection_phasor(b, _f(net.res_gen.at[i, 'p_mw']), _f(net.res_gen.at[i, 'q_mvar']))
            amp, ph = self._three(self._v_phasor(b) + complex(r, x) * i_ph)
            star = ckt.node(f'{_label(net, "gen", i)} star point')
            ckt.add_r(star, 0, R_BIAS)
            g = ckt.add_coupled([star] * 3, self.nodes[b], np.eye(3) * r, np.eye(3) * x / w, ac=(amp, w, ph))
            label = _label(net, 'gen', i)
            self.series.append(('Generator', label, _row_id(net, 'gen', i), label, [('i_cp', g, k, 1.0) for k in range(3)]))

    # --- a fault -------------------------------------------------------------------------

    def fault(self, bus, kind, r_pn, r_ng, t_on, t_off):
        """
        A fault at ``bus`` from t_on: each faulted phase through r_pn to a star
        point, grounded through r_ng for a ground fault (as Simscape's
        three-phase fault block). ``kind``: ag, bg, cg, ab, bc, ca, abg, bcg,
        cag, abc, abcg. Cleared from t_off, if given, as an arc goes out: each
        phase at its current's next zero, the ground path with the last -
        opening at once would chop an inductive current into an overvoltage
        only the circuit's parasitics limit.
        """
        ckt = self.ckt
        phases = [k for k, p in enumerate('abc') if p in kind.replace('g', '')]
        star = ckt.node('AC fault star point')
        ckt.add_r(star, 0, R_BIAS)
        switches = []
        for k in phases:
            mid = ckt.node(f'AC fault {"abc"[k]}')
            switches.append(ckt.add_switch(self.nodes[bus][k], mid, closed=False))
            ckt.add_r(mid, star, r_pn)
        if kind.endswith('g'):
            mid = ckt.node('AC fault ground')
            switches.append(ckt.add_switch(star, mid, closed=False))
            ckt.add_r(mid, 0, r_ng)
        for k in switches:
            ckt.at(t_on, lambda st, k=k: st.set_switch(k, True))
        phase_sw, ground_sw = switches[:len(phases)], switches[len(phases):]
        if t_off is not None:
            last = {}

            def clear(t, state):
                if t < t_off - 1e-12:
                    return False
                changed = False
                for k in phase_sw:
                    if not state.sw_closed[k]:
                        continue
                    i = float(state.i_sw[k])
                    if k in last and (i == 0.0 or (i > 0) != (last[k] > 0)):
                        state.set_switch(k, False)
                        changed = True
                    last[k] = i
                if changed and not any(state.sw_closed[k] for k in phase_sw):
                    for k in ground_sw:
                        state.set_switch(k, False)
                return changed
            ckt.add_controller(clear)
        return {'phases': phases, 'switches': phase_sw}
