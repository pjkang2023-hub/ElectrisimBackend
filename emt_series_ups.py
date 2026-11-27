"""
A series-reactor UPS in the EMT study: a drawn line (the reactor, from its source
bus to its critical bus) and a grid-forming PCS on the critical side, directly or
behind its transformer, run together as one reactor-coupled medium-voltage UPS.

Its power path: a static disconnect in series with the reactor, and a static
bypass, normally open, from the source bus to the critical bus around both. The
disconnect's phases open each at its current's zero, as a thyristor's would, or
with series_ups_opening = forced at once, a surge arrester clamping them, as a
turn-off switch's would.

Its modes:
- on_grid: the disconnect closed. The load's power comes through the reactor; the
  converter's droop takes a change of load at once and hands it to the source. A
  slow trim of its internal voltage holds the critical bus at its load-flow voltage.
- transfer: the source bus below v_transfer_pu for t_detect_ms. The disconnect's
  phases open; until then the converter feeds the sag through the reactor.
- on_storage: the converter forms the critical bus alone, its power set point
  following its power, so its frequency returns to nominal.
- sync: the source back above v_return_pu for t_return_ms. The converter's angle is
  brought to the source's, and at under close_angle_deg the disconnect closes.
- handback (on_grid again): its power set point ramped back over handback_s, so the
  source takes the load over at that rate.
- bypass: the converter blocked (by an event or its own protection), or held at its
  current limit for limit_to_bypass_s, on a healthy source. The bypass closes at
  once and the disconnect opens.
- retransfer: the converter back by a run's event, at the bus's voltage. The
  disconnect closes and the bypass opens at its zeros; then on_grid.
- lost: the converter blocked with no healthy source to bypass to.

The source's voltage is detected as each phase's half-cycle rms, or with
series_ups_detection = fast also as its space vector's magnitude, sampled every
100 us: the half-cycle window is most of the delay a transfer takes.

The critical bus is assessed against IEC 62040-3's dynamic output curve 1, by
analogy at medium voltage: see class1_check. What this does not model: the switches'
own voltage drop and losses, converter thermal limits, and the battery beyond its
DC model.
"""
import collections
import math

import numpy as np

SAMPLE_S = 1e-4
TRIM_TAU_S = 0.2          # the voltage trim's time constant
TRIM_MAX = 0.10           # its reach, as a share of the converter's internal voltage
RESTORE_TAU_S = 1.0       # on storage: the frequency's return to nominal
SYNC_RAD_S_PER_RAD = 20.0 # the angle loop's gain in sync
SYNC_MAX_HZ = 2.0         # and how far from the source's frequency it may go
V_BYPASS_MIN = 0.85       # the lowest source voltage the bypass closes on
V_BACKFEED = 0.5          # below this the source is taken as down and the reactor's current as backfeed
REF_TAU_S = 0.01          # the reference the critical bus's deviation is taken from follows its angle this fast
CURVE_SOURCE = 'IEC 62040-3 Ed. 2 FDIS (22H/129/FDIS) Figure 2, read from the figure; by analogy at medium voltage'
LEVELS = (0.10, 0.11, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30)


def curve_1_allowed_s(level):
    """How long curve 1 lets a deviation stay beyond a level (a share of nominal), in seconds."""
    if level >= 0.30:
        return 1e-4            # the curve's shortest time: anything shorter is not assessed
    if level >= 0.14:
        return 5e-3 * 4.0 ** ((0.30 - level) / 0.16)     # 30 % at 5 ms to 14 % at 20 ms, on its log time axis
    if level >= 0.12:
        return 40e-3
    if level >= 0.11:
        return 60e-3
    if level >= 0.10:
        return 100e-3
    return math.inf


def class1_check(t, measures):
    """
    Each measure's longest excursion beyond each of the curve's levels, against the
    time the curve allows. measures: {name: deviation, a share of nominal, at times t}.
    Returns whether all stay inside and the worst one, as its share of its allowed time.
    """
    worst = None
    t = np.asarray(t, dtype=float)
    for name, dev in measures.items():
        dev = np.asarray(dev, dtype=float)
        for level in LEVELS:
            beyond = dev > level
            if not beyond.any():
                continue
            edges = np.diff(np.concatenate(([0], beyond.astype(int), [0])))
            starts, stops = np.where(edges == 1)[0], np.where(edges == -1)[0]
            durations = t[np.minimum(stops, len(t) - 1)] - t[starts]
            k = int(np.argmax(durations))
            allowed = curve_1_allowed_s(level)
            share = float(durations[k]) / allowed
            if worst is None or share > worst['share']:
                worst = {'share': share, 'measure': name, 'level_pct': round(level * 100.0, 1),
                         'duration_ms': round(float(durations[k]) * 1e3, 2), 'allowed_ms': round(allowed * 1e3, 2),
                         't_start_ms': round(float(t[starts[k]]) * 1e3, 2), 'deviation_peak_pct': round(float(dev.max()) * 100.0, 1)}
    if worst is not None:
        worst['share'] = round(worst['share'], 3)
    return {'inside': worst is None or worst['share'] <= 1.0, 'worst': worst, 'curve': CURVE_SOURCE}


def _f(value, default):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def settings(params):
    """The study's series_ups_* settings, with their defaults."""
    p = params.get
    return {'v_transfer': _f(p('series_ups_v_transfer_pu'), 0.5),
            't_detect': _f(p('series_ups_t_detect_ms'), 2.0) * 1e-3,
            'v_return': _f(p('series_ups_v_return_pu'), 0.9),
            't_return': _f(p('series_ups_t_return_ms'), 50.0) * 1e-3,
            'close_angle': math.radians(_f(p('series_ups_close_angle_deg'), 10.0)),
            'handback': max(_f(p('series_ups_handback_s'), 0.5), 0.0),
            'limit_to_bypass': _f(p('series_ups_limit_to_bypass_s'), 1.0),
            'fast': str(p('series_ups_detection') or '').strip().lower() == 'fast',
            'forced': str(p('series_ups_opening') or '').strip().lower() == 'forced'}


def _space_vector(v):
    return complex((2.0 * v[0] - v[1] - v[2]) / 3.0, (v[1] - v[2]) / math.sqrt(3.0))


def _angle(v):
    vec = _space_vector(v)
    return math.atan2(vec.imag, vec.real)


def _wrap(x):
    return (x + math.pi) % (2.0 * math.pi) - math.pi


class _HalfCycleRms:
    """Each phase's rms over the last half cycle, per unit of a peak."""

    def __init__(self, f_hz, peak):
        self.window, self.peak = 0.5 / f_hz, peak
        self.hist = collections.deque()
        self.total = np.zeros(3)
        self.span = 0.0
        self.value = None

    def add(self, t, v, h):
        sq = np.asarray(v, dtype=float) ** 2 * h
        self.hist.append((t, sq, h))
        self.total += sq
        self.span += h
        while self.hist and self.hist[0][0] <= t - self.window:
            _, old, old_h = self.hist.popleft()
            self.total -= old
            self.span -= old_h
        if self.span >= 0.985 * self.window:
            self.value = np.sqrt(np.maximum(self.total, 0.0) / self.span * 2.0) / self.peak
        return self.value


class SeriesUps:
    """One UPS, its switches built into the circuit by emt_ac, its supervisor run each step."""

    def __init__(self, label, plan, source_nodes, critical_nodes, vn, conv, f_hz, v_ref_pu, s):
        self.label, self.conv = label, conv
        self.source_nodes, self.critical_nodes = source_nodes, critical_nodes
        self.disconnect, self.bypass = plan['disconnect'], plan['bypass']
        self.s, self.v_ref = s, v_ref_pu
        self.w0 = 2.0 * math.pi * f_hz
        self.cycle = 1.0 / f_hz
        self.peak = vn * math.sqrt(2.0 / 3.0)
        self.rms_s, self.rms_c = _HalfCycleRms(f_hz, self.peak), _HalfCycleRms(f_hz, self.peak)
        self.rms_i = _HalfCycleRms(f_hz, math.sqrt(2.0))
        self.mode = 'on_grid'
        self.modes = [(0.0, 'on_grid')]
        self.t_tick = 0.0
        self.below = self.above = None      # when the source went below v_transfer / above v_return
        self.opening = None                 # {'switches', 'until': the mode after, 'last': per phase}
        self.opened = []                    # per opening: what, its phases' times and currents
        self.p_home = conv.p_set            # its power set point on grid
        self.ramp = None                    # (t start, set point at the start)
        self.e_home = conv.e0
        self.limit_seen = conv.limited_time
        self.limit_since = None
        self.closed_at = []                 # when the disconnect closed again: (t, angle in degrees)
        self.trace = []                     # (t, source lowest, critical lowest, critical highest, converter MW)
        self.th_ref = None                  # the reference's angle for the critical bus's deviation
        self.series = []                    # every sample: (t, space vector's deviation, lowest rms's, highest rms's)
        self.backfeed_peak = 0.0            # the reactor's current while the source is down and the disconnect closed
        self.backfeed_rms = 0.0
        self.backfeed = []                  # (t, its three currents) in that time

    def _set_mode(self, t, mode):
        self.mode = mode
        self.modes.append((t, mode))

    def _open(self, t, state, switches, then, what):
        rec = {'what': what, 't_start': t, 't_phase': [None, None, None], 'i_phase': [None, None, None]}
        self.opened.append(rec)
        if then == 'on_storage':
            self._set_mode(t, 'transfer')
        if self.s['forced'] and what == 'disconnect':
            # A turn-off switch: every phase at once, its arrester taking the reactor's current.
            for n, k in enumerate(switches):
                rec['t_phase'][n], rec['i_phase'][n] = t, abs(float(state.i_sw[k]))
                state.set_switch(k, False)
            if then == 'on_storage':
                self._set_mode(t, 'on_storage')
            return True
        self.opening = {'switches': switches, 'until': then, 'last': [None, None, None]}
        return False

    def _opening_step(self, t, state):
        """Each phase opens at its current's zero; True if one did."""
        changed = False
        rec = self.opened[-1]
        for n, k in enumerate(self.opening['switches']):
            if rec['t_phase'][n] is not None:
                continue
            i = float(state.i_sw[k])
            last = self.opening['last'][n]
            self.opening['last'][n] = i
            if last is not None and (i == 0.0 or (i > 0) != (last > 0)):
                state.set_switch(k, False)
                rec['t_phase'][n], rec['i_phase'][n] = t, abs(i)
                changed = True
        if all(x is not None for x in rec['t_phase']):
            if self.opening['until'] != self.mode and self.opening['until'] in ('on_storage', 'on_grid'):
                self._set_mode(t, self.opening['until'])
            self.opening = None
        return changed

    def _to_bypass(self, t, state, source_low):
        if source_low < V_BYPASS_MIN:
            self._set_mode(t, 'lost')
            return False
        if self.opening is not None and self.opening['switches'] is self.bypass:
            self.opening = None
        for k in self.bypass:
            state.set_switch(k, True)
        self._set_mode(t, 'bypass')
        if self.conv.blocked_at is None:
            self.conv._block(t, state)
        if self.opening is None and all(bool(state.sw_closed[k]) for k in self.disconnect):
            self._open(t, state, self.disconnect, 'bypass', 'disconnect')
        return True

    def _retransfer(self, t, state):
        """The converter back, at its terminal's voltage and carrying nothing: the disconnect closes, the bypass opens."""
        conv = self.conv
        vec = _space_vector(state.v[conv.ac_nodes])
        conv.th_e, conv.w_gf = math.atan2(vec.imag, vec.real), conv.w0
        conv.p_f, conv.q_f = 0.0, 0.0
        conv.p_set = self.p_home
        conv.e0 = abs(vec) - conv.n_q * conv.q_set
        conv.k_ph = 1.0
        if conv.v_loop is not None:
            conv.v_loop.update(int_d=0.0, int_q=0.0)
        self.ramp = None
        for k in self.disconnect:
            state.set_switch(k, True)
        self._set_mode(t, 'retransfer')
        self._open(t, state, self.bypass, 'on_grid', 'bypass')
        return True

    def _trim(self, v_mean, h):
        """The converter's internal voltage trimmed to hold the critical bus, held while it is at its limit."""
        conv = self.conv
        if conv.limited_time > self.limit_seen:
            return
        e = conv.e0 + (self.v_ref - v_mean) * conv.v_nom_peak * h / TRIM_TAU_S
        conv.e0 = min(max(e, self.e_home * (1.0 - TRIM_MAX)), self.e_home * (1.0 + TRIM_MAX))

    def _assess(self, t, h, v_c, rms_c):
        """The critical bus's deviation this sample, from a reference that follows its angle but not its depth."""
        vec = _space_vector(v_c)
        angle = math.atan2(vec.imag, vec.real)
        if self.th_ref is None:
            self.th_ref = angle
        self.th_ref = self.th_ref + self.w0 * h
        self.th_ref += _wrap(angle - self.th_ref) * min(h / REF_TAU_S, 1.0)
        ref = self.v_ref * self.peak * complex(math.cos(self.th_ref), math.sin(self.th_ref))
        if t >= self.cycle:
            self.series.append((t, abs(vec - ref) / (self.v_ref * self.peak),
                                max(self.v_ref - float(rms_c.min()), 0.0) / self.v_ref,
                                max(float(rms_c.max()) - self.v_ref, 0.0) / self.v_ref))

    def control(self, t, state):
        changed = self.opening is not None and self._opening_step(t, state)
        h = t - self.t_tick
        if h < SAMPLE_S - 1e-12:
            return changed
        self.t_tick = t
        conv = self.conv
        v_s, v_c = state.v[self.source_nodes], state.v[self.critical_nodes]
        rms_s, rms_c = self.rms_s.add(t, v_s, h), self.rms_c.add(t, v_c, h)
        i_r = np.array([float(state.i_sw[k]) for k in self.disconnect])
        rms_i = self.rms_i.add(t, i_r, h)
        if rms_s is None or rms_c is None:
            return changed
        low_s, low_c, high_c = float(rms_s.min()), float(rms_c.min()), float(rms_c.max())
        detect = min(low_s, abs(_space_vector(v_s)) / self.peak) if self.s['fast'] else low_s
        self._assess(t, h, v_c, rms_c)
        if not self.trace or t - self.trace[-1][0] >= 1e-3:
            self.trace.append((t, low_s, low_c, high_c, conv.p_f / 1e6))
        if self.mode == 'transfer' or (self.mode == 'on_grid' and detect < V_BACKFEED):
            self.backfeed_peak = max(self.backfeed_peak, float(np.max(np.abs(i_r))))
            self.backfeed.append((t, i_r))
            if rms_i is not None:
                self.backfeed_rms = max(self.backfeed_rms, float(np.max(rms_i)))
        limited = conv.limited_time > self.limit_seen
        if self.mode == 'lost':
            self.limit_seen = conv.limited_time
            return changed
        if self.mode == 'bypass':
            self.limit_seen = conv.limited_time
            if conv.blocked_at is None:
                # Beside the closed bypass the reactor's current only decays, so the disconnect may still be opening.
                self.opening = None
                return self._retransfer(t, state)
            return changed
        if conv.blocked_at is not None:
            self.limit_seen = conv.limited_time
            return self._to_bypass(t, state, low_s) or changed
        if self.mode != 'transfer':
            self._trim(float(rms_c.mean()), h)
        self.limit_seen = conv.limited_time
        if self.mode == 'on_grid':
            if self.ramp is not None:
                t0, p0 = self.ramp
                share = 1.0 if self.s['handback'] <= 0 else min((t - t0) / self.s['handback'], 1.0)
                conv.p_set = p0 + (self.p_home - p0) * share
                if share >= 1.0:
                    self.ramp = None
            self.limit_since = (self.limit_since if self.limit_since is not None else t) if limited else None
            if detect < self.s['v_transfer']:
                self.below = t if self.below is None else self.below
                if t - self.below >= self.s['t_detect'] - 1e-12:
                    self.below, self.ramp, self.limit_since = None, None, None
                    changed = self._open(t, state, self.disconnect, 'on_storage', 'disconnect') or changed
            else:
                self.below = None
                if self.limit_since is not None and t - self.limit_since >= self.s['limit_to_bypass']:
                    self.limit_since = None
                    return self._to_bypass(t, state, low_s) or changed
        elif self.mode == 'on_storage':
            conv.p_set += (conv.p_f - conv.p_set) * min(h / RESTORE_TAU_S, 1.0)
            if low_s >= self.s['v_return']:
                self.above = t if self.above is None else self.above
                if t - self.above >= self.s['t_return'] - 1e-12:
                    self.above = None
                    self._set_mode(t, 'sync')
            else:
                self.above = None
        elif self.mode == 'sync':
            if low_s < self.s['v_return']:
                conv.p_set = conv.p_f
                self._set_mode(t, 'on_storage')
                return changed
            delta = _wrap(_angle(v_s) - _angle(v_c))
            dw = max(min(SYNC_RAD_S_PER_RAD * delta, 2.0 * math.pi * SYNC_MAX_HZ), -2.0 * math.pi * SYNC_MAX_HZ)
            conv.p_set = conv.p_f + dw / conv.m_p
            if abs(delta) <= self.s['close_angle']:
                for k in self.disconnect:
                    state.set_switch(k, True)
                conv.p_set = conv.p_f
                self.ramp = (t, conv.p_set)
                self.closed_at.append((t, math.degrees(delta)))
                self._set_mode(t, 'on_grid')
                changed = True
        return changed

    def _backfeed_symmetrical(self):
        """The backfeed's symmetrical rms, its DC offset fitted out over its last cycle, if it lasted one; kA."""
        if not self.backfeed or self.backfeed[-1][0] - self.backfeed[0][0] < self.cycle:
            return None
        t = np.array([x[0] for x in self.backfeed])
        i = np.array([x[1] for x in self.backfeed])
        last = t >= t[-1] - self.cycle
        x = np.column_stack([np.cos(self.w0 * t[last]), np.sin(self.w0 * t[last]), np.ones(last.sum())])
        coef, *_ = np.linalg.lstsq(x, i[last], rcond=None)
        return round(float(np.max(np.hypot(coef[0], coef[1]))) / math.sqrt(2.0) * 1e-3, 3)

    def result(self):
        tr = np.array(self.trace) if self.trace else np.zeros((0, 5))
        # The critical bus against 10 %: its time outside, by the half-cycle rms sampled each millisecond.
        outside = float(np.sum((tr[:, 2] < 0.9) | (tr[:, 3] > 1.1))) * 1e-3 if len(tr) else 0.0
        step = max(1, len(tr) // 400)
        sr = np.array(self.series) if self.series else np.zeros((0, 4))
        return {'label': self.label, 'mode_end': self.mode,
                'modes': [{'t_ms': round(t * 1e3, 3), 'mode': m} for t, m in self.modes],
                'openings': [{'what': o['what'], 't_start_ms': round(o['t_start'] * 1e3, 3),
                              't_phase_ms': [None if x is None else round(x * 1e3, 3) for x in o['t_phase']],
                              'i_phase_a': [None if x is None else round(x, 3) for x in o['i_phase']]} for o in self.opened],
                'closings': [{'t_ms': round(t * 1e3, 3), 'angle_deg': round(a, 2)} for t, a in self.closed_at],
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
                          'p_converter_mw': [round(x, 3) for x in tr[::step, 4]]}}
