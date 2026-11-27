"""
A double-conversion UPS in the EMT study: a rectifier on the source side and a
grid-forming PCS on the critical bus, on one DC link with the PCS's store, run
together as one medium-voltage UPS with no static bypass.

The drawn network: the source and critical buses are not joined. A load at the
rectifier's bus stands for the rectifier in the load flow, and the PCS forms the
critical bus from its store; in the run, the rectifier replaces that load.

The rectifier takes from the source what the inverter delivers, at unity power factor:
- on_grid: its power set to the inverter's, the store idle. In a sag above
  v_transfer_pu it holds its current at its value before the sag, so its power
  falls with the voltage (the constant-current region of a large load's ride-through)
  and the store carries the rest.
- on_storage: the source below v_transfer_pu for t_detect_ms. It blocks: its
  diodes conduct only beyond the DC rails, so it backfeeds nothing. The store
  carries the load.
- on_grid again: the source above v_return_pu for t_return_ms. It resumes, its
  power ramped back over handback_s.
- lost: the inverter blocked. With no static bypass the critical bus goes dark:
  redundancy inside the product and the building's four-to-make-three are not modelled.

The inverter runs throughout: its power set point follows its power, so its
frequency returns to nominal, and a slow trim holds the critical bus's voltage.

What this does not model: the store's DC/DC stage (the store sits on the link, so the
link is stiff), redundancy among converter groups, and thermal limits. The inverter
has no fast voltage loop, only its droops, which will overstate its response to load steps.
"""
import math

import numpy as np

from emt_series_ups import RESTORE_TAU_S, SAMPLE_S, V_BACKFEED, SeriesUps, _HalfCycleRms, _space_vector, class1_check, settings

V_RIDE = 0.9              # below this the rectifier holds its current


class DoubleConversion(SeriesUps):
    """One UPS: its supervisor run each step, sharing the series-reactor UPS's assessment of the critical bus."""

    def __init__(self, label, rect, inv, source_nodes, critical_nodes, vn, ratio, f_hz, v_ref_pu, s):
        self.label, self.conv, self.rect = label, inv, rect
        self.source_nodes, self.critical_nodes = source_nodes, critical_nodes
        self.ratio = ratio                  # the rectifier's current to the source bus's side
        self.s, self.v_ref = s, v_ref_pu
        self.w0 = 2.0 * math.pi * f_hz
        self.cycle = 1.0 / f_hz
        self.peak = vn * math.sqrt(2.0 / 3.0)
        self.rms_s, self.rms_c = _HalfCycleRms(f_hz, self.peak), _HalfCycleRms(f_hz, self.peak)
        self.rms_i = _HalfCycleRms(f_hz, math.sqrt(2.0))
        self.mode = 'on_grid'
        self.modes = [(0.0, 'on_grid')]
        self.t_tick = 0.0
        self.below = self.above = None
        self.e_home = inv.e0
        self.limit_seen = inv.limited_time
        self.p_before = None                # its power when the sag began
        self.trace = []                     # (t, source lowest, critical lowest, critical highest, inverter MW, rectifier MW)
        self.th_ref = None
        self.series = []
        self.backfeed_peak = 0.0
        self.backfeed_rms = 0.0
        self.backfeed = []

    def control(self, t, state):
        h = t - self.t_tick
        if h < SAMPLE_S - 1e-12:
            return False
        self.t_tick = t
        conv, rect = self.conv, self.rect
        v_s, v_c = state.v[self.source_nodes], state.v[self.critical_nodes]
        rms_s, rms_c = self.rms_s.add(t, v_s, h), self.rms_c.add(t, v_c, h)
        i_r = np.array([float(state.i_rl[k]) for k in rect.k_e]) * self.ratio
        rms_i = self.rms_i.add(t, i_r, h)
        if rms_s is None or rms_c is None:
            return False
        low_s, low_c, high_c = float(rms_s.min()), float(rms_c.min()), float(rms_c.max())
        detect = min(low_s, abs(_space_vector(v_s)) / self.peak) if self.s['fast'] else low_s
        self._assess(t, h, v_c, rms_c)
        if not self.trace or t - self.trace[-1][0] >= 1e-3:
            self.trace.append((t, low_s, low_c, high_c, conv.p_f / 1e6, rect.p_ref / 1e6))
        if detect < V_BACKFEED:
            self.backfeed_peak = max(self.backfeed_peak, float(np.max(np.abs(i_r))))
            self.backfeed.append((t, i_r))
            if rms_i is not None:
                self.backfeed_rms = max(self.backfeed_rms, float(np.max(rms_i)))
        if self.mode == 'lost':
            return False
        if conv.blocked_at is not None:
            self._set_mode(t, 'lost')
            return False
        conv.p_set += (conv.p_f - conv.p_set) * min(h / RESTORE_TAU_S, 1.0)
        self._trim(float(rms_c.mean()), h)
        self.limit_seen = conv.limited_time
        if self.mode == 'on_grid':
            if rect.blocked_at is not None:        # its own protection
                self._set_mode(t, 'on_storage')
                return False
            if low_s < V_RIDE:
                self.p_before = rect.p_ref if self.p_before is None else self.p_before
                rect.p_ref = min(conv.p_f, self.p_before * low_s / V_RIDE)
            else:
                self.p_before = None
                rect.p_ref = conv.p_f
            if detect < self.s['v_transfer']:
                self.below = t if self.below is None else self.below
                if t - self.below >= self.s['t_detect'] - 1e-12:
                    self.below = None
                    rect._block(t, state)
                    self._set_mode(t, 'on_storage')
                    return True
            else:
                self.below = None
        elif self.mode == 'on_storage':
            if low_s >= self.s['v_return']:
                self.above = t if self.above is None else self.above
                if t - self.above >= self.s['t_return'] - 1e-12:
                    self.above = None
                    self.p_before = None
                    rect.p_ref = conv.p_f
                    if rect.resume(t, state, self.s['handback']):
                        self._set_mode(t, 'on_grid')
                        return True
            else:
                self.above = None
        return False

    def result(self):
        tr = np.array(self.trace) if self.trace else np.zeros((0, 6))
        outside = float(np.sum((tr[:, 2] < 0.9) | (tr[:, 3] > 1.1))) * 1e-3 if len(tr) else 0.0
        step = max(1, len(tr) // 400)
        sr = np.array(self.series) if self.series else np.zeros((0, 4))
        return {'label': self.label, 'mode_end': self.mode,
                'modes': [{'t_ms': round(t * 1e3, 3), 'mode': m} for t, m in self.modes],
                'v_source_min_pu': round(float(tr[:, 1].min()), 4) if len(tr) else None,
                'v_critical_min_pu': round(float(tr[:, 2].min()), 4) if len(tr) else None,
                'v_critical_max_pu': round(float(tr[:, 3].max()), 4) if len(tr) else None,
                'critical_outside_10pct_ms': round(outside * 1e3, 1),
                'backfeed_peak_ka': round(self.backfeed_peak * 1e-3, 3),
                'backfeed_rms_ka': round(self.backfeed_rms * 1e-3, 3),
                'backfeed_symmetrical_ka': self._backfeed_symmetrical(),
                'class1': class1_check(sr[:, 0], {'space vector': sr[:, 1], 'phase rms, low': sr[:, 2], 'phase rms, high': sr[:, 3]}),
                'trace': {'t_ms': [round(x * 1e3, 2) for x in tr[::step, 0]],
                          'v_source_pu': [round(x, 4) for x in tr[::step, 1]],
                          'v_critical_low_pu': [round(x, 4) for x in tr[::step, 2]],
                          'v_critical_high_pu': [round(x, 4) for x in tr[::step, 3]],
                          'p_inverter_mw': [round(x, 3) for x in tr[::step, 4]],
                          'p_rectifier_mw': [round(x, 3) for x in tr[::step, 5]]}}


def plan_load(net, params):
    """The load that stands for the rectifier in the load flow, by name; None if there is none."""
    name = str(params.get('double_conversion_rectifier_load') or '').strip()
    if not name or name == 'none' or not len(net.load):
        return None
    hit = [i for i in net.load.index if str(net.load.at[i, 'name']) == name]
    return hit[0] if hit else None


__all__ = ['DoubleConversion', 'plan_load', 'settings']
