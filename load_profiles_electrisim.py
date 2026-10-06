"""
Load power profiles: a load's active power over time, in per unit of its rating.

AI training loads swing between an idle level and full power every few
seconds, so a constant power misses what matters most about them. A profile
is parsed from an uploaded CSV or TXT file (time in s and power in p.u., or
power only at a given time step), reported on - so steps, overshoot and the
fastest content it can carry are visible before it is used - and sampled at
whatever times a study needs.
"""
import math
import re

import numpy as np

# A single-sample change above this (p.u.) is reported as a step.
DEFAULT_STEP_THRESHOLD_PU = 0.2


def parse_profile(text, time_step_s=None):
    """
    (t, p, notes) from CSV or TXT text.

    Two numeric columns are time (s) and power (p.u.); one column is power,
    sampled every ``time_step_s``. Delimiters may be commas, semicolons, tabs or
    spaces; header and comment lines (anything that is not numbers) are
    skipped. Raises ValueError with a message for the user when nothing usable
    is found.
    """
    rows, skipped = [], 0
    for line in str(text or '').splitlines():
        line = line.strip()
        if not line:
            continue
        cells = [c for c in re.split(r'[,;\t ]+', line) if c]
        try:
            values = [float(c) for c in cells]
        except ValueError:
            skipped += 1
            continue
        if values and all(math.isfinite(v) for v in values):
            rows.append(values)
        else:
            skipped += 1
    if not rows:
        raise ValueError('No numeric rows found: expected time (s) and power (p.u.) columns, or a power column.')

    notes = []
    widths = {len(r) for r in rows}
    if widths == {1}:
        if not time_step_s or time_step_s <= 0:
            raise ValueError('The file has a single column: give the time step between its samples.')
        p = np.array([r[0] for r in rows], dtype=float)
        t = np.arange(len(p)) * float(time_step_s)
    else:
        if min(widths) < 2:
            raise ValueError('Rows have different numbers of columns: expected time (s) and power (p.u.) on every row.')
        if max(widths) > 2:
            notes.append(f'Only the first two of the {max(widths)} columns are used: time and power.')
        t = np.array([r[0] for r in rows], dtype=float)
        p = np.array([r[1] for r in rows], dtype=float)
    if skipped:
        notes.append(f'{skipped} non-numeric line(s) skipped (headers or comments).')
    if len(t) < 2:
        raise ValueError('A profile needs at least two samples.')
    if np.any(np.diff(t) <= 0):
        raise ValueError('Time must increase from each row to the next.')
    return t, p, notes


def analyse_profile(t, p, step_threshold_pu=DEFAULT_STEP_THRESHOLD_PU):
    """The upload report: what the profile contains and what to watch for."""
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    dt = np.diff(t)
    dp = np.diff(p)
    rate = dp / dt
    dt_median = float(np.median(dt))
    regular = bool(np.allclose(dt, dt_median, rtol=1e-3, atol=1e-9))
    # Time-weighted mean (trapezoidal), right for irregular sampling too.
    mean = float(np.trapezoid(p, t) / (t[-1] - t[0]))

    steps = [{'time_s': float(t[k + 1]), 'from_pu': float(p[k]), 'to_pu': float(p[k + 1]),
              'change_pu': float(dp[k])}
             for k in np.flatnonzero(np.abs(dp) > step_threshold_pu)]
    flags = []
    over = p > 1.0 + 1e-9
    if over.any():
        flags.append(f'Exceeds 1.0 p.u. for {float(np.sum(dt[over[:-1]])):.2f} s, '
                     f'up to {float(p.max()):.3f} p.u.: check against the load\'s rating.')
    if (p < 0).any():
        flags.append(f'Negative values, down to {float(p.min()):.3f} p.u.: the load would generate.')
    if steps:
        biggest = max(steps, key=lambda s: abs(s['change_pu']))
        flags.append(f'{len(steps)} single-sample step(s) larger than {step_threshold_pu:g} p.u.; the largest is '
                     f'{biggest["change_pu"]:+.3f} p.u. at {biggest["time_s"]:.2f} s.')
    zero = np.abs(p) < 1e-6
    if zero.any():
        flags.append(f'At zero for {float(np.sum(dt[zero[:-1]])):.2f} s: the load is off, not idling.')
    if not regular:
        flags.append(f'Irregular sampling: steps from {float(dt.min()):.4g} to {float(dt.max()):.4g} s.')

    return {
        'samples': int(len(t)),
        'start_s': float(t[0]),
        'duration_s': float(t[-1] - t[0]),
        'time_step_s': dt_median,
        'regular_sampling': regular,
        # Half the sampling rate: nothing faster than this is in the profile.
        'fastest_content_hz': 0.5 / dt_median,
        'min_pu': float(p.min()),
        'mean_pu': mean,
        'max_pu': float(p.max()),
        'max_rise_pu_per_s': float(max(rate.max(), 0.0)),
        'max_fall_pu_per_s': float(min(rate.min(), 0.0)),
        'step_threshold_pu': step_threshold_pu,
        'steps': steps[:20],
        'step_count': len(steps),
        'flags': flags,
    }


def _one_period(t, p):
    """
    A repeating profile as one closed period: N samples dt apart last N x dt,
    so the cycle returns to its first value one step after its last sample.
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    dt = float(np.median(np.diff(t)))
    return np.append(t, t[-1] + dt), np.append(p, p[0])


def period_s(t):
    """How long a profile lasts when it repeats: its samples' span plus one step."""
    t = np.asarray(t, dtype=float)
    return float(t[-1] - t[0] + np.median(np.diff(t)))


def sample_profile(t, p, times, repeat=False):
    """
    The profile's value at ``times`` (s), linearly interpolated.

    Before the profile starts it holds its first value; after it ends it holds
    its last, or, with ``repeat``, starts again from the beginning.
    """
    if repeat:
        t, p = _one_period(t, p)
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    x = np.asarray(times, dtype=float)
    if repeat:
        span = t[-1] - t[0]
        x = np.where(x > t[-1], t[0] + np.mod(x - t[0], span), x)
    return np.interp(x, t, p)


def average_profile(t, p, start, end, repeat=False):
    """
    The profile's mean over [start, end] (s): what a load following it draws
    on average over a study step longer than the profile's own samples.
    """
    if end <= start:
        return float(sample_profile(t, p, [start], repeat)[0])
    if repeat:
        t, p = _one_period(t, p)
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    cum =np.concatenate(([0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(t))))
    span, total = t[-1] - t[0], cum[-1]

    def integral(x):
        """Integral of the profile from t[0] to x."""
        if x <= t[0]:
            return p[0] * (x - t[0])
        if repeat and x > t[-1]:
            periods, rest = divmod(x - t[0], span)
            return periods * total + integral(t[0] + rest)
        if x >= t[-1]:
            return total + p[-1] * (x - t[-1])
        k = int(np.searchsorted(t, x)) - 1
        px = np.interp(x, t, p)
        return cum[k] + 0.5 * (p[k] + px) * (x - t[k])

    return float((integral(end) - integral(start)) / (end - start))


def profile_arrays(spec):
    """
    (t, p) of a library entry: ``{'dt': s, 'p': [...]}`` (optionally ``'t0'``),
    or ``{'t': [...], 'p': [...]}``. Raises ValueError naming what is wrong.
    """
    if not isinstance(spec, dict) or not isinstance(spec.get('p'), (list, tuple)) or len(spec['p']) < 2:
        raise ValueError('a profile needs at least two power values')
    p = np.asarray([float(v) for v in spec['p']], dtype=float)
    if spec.get('t') is not None:
        t = np.asarray([float(v) for v in spec['t']], dtype=float)
        if len(t) != len(p):
            raise ValueError(f'{len(t)} times for {len(p)} power values')
    else:
        dt = float(spec.get('dt') or 0)
        if dt <= 0:
            raise ValueError('a profile without times needs a positive time step')
        t = float(spec.get('t0') or 0.0) + np.arange(len(p)) * dt
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(p))):
        raise ValueError('times and power values must be numbers')
    if np.any(np.diff(t) <= 0):
        raise ValueError('time must increase from each sample to the next')
    return t, p


def library_from_params(params):
    """{profile id: {'name', 't', 'p'}} from a study's parameters, and the problems found."""
    library, problems = {}, []
    for pid, spec in ((params or {}).get('load_profiles') or {}).items():
        name = str((spec or {}).get('name') or pid) if isinstance(spec, dict) else str(pid)
        try:
            t, p = profile_arrays(spec)
        except (TypeError, ValueError) as e:
            problems.append(f'Load profile "{name}" is not used: {e}.')
            continue
        library[str(pid)] = {'name': name, 't': t, 'p': p}
    return library, problems


def load_assignments(in_data):
    """
    {load name: {'profile_id', 'q_mode', 'display_name'}} for the AC loads that
    follow a library profile. ``q_mode`` 'pf' scales Q with P (constant power
    factor); 'constant' keeps the drawn Q.
    """
    out = {}
    for el in (in_data or {}).values():
        if not isinstance(el, dict):
            continue
        typ = str(el.get('typ', ''))
        if not typ.startswith('Load') or typ.startswith('Load DC'):
            continue
        pid = str(el.get('load_profile_id') or '').strip()
        if not pid:
            continue
        q_mode = 'constant' if str(el.get('load_profile_q_mode') or '').lower() == 'constant' else 'pf'
        out[str(el.get('name'))] = {
            'profile_id': pid,
            'q_mode': q_mode,
            'display_name': str(el.get('userFriendlyName') or el.get('name')),
        }
    return out


def dc_load_assignments(in_data):
    """{DC load name: {'profile_id', 'display_name'}} for the DC loads that follow a library profile."""
    out = {}
    for el in (in_data or {}).values():
        if not isinstance(el, dict) or not str(el.get('typ', '')).startswith('Load DC'):
            continue
        pid = str(el.get('load_profile_id') or '').strip()
        if pid:
            out[str(el.get('name'))] = {'profile_id': pid,
                                        'display_name': str(el.get('userFriendlyName') or el.get('name'))}
    return out


class Follower:
    """
    A profile as a function of a simulation's time: its value at t0 + t,
    linearly interpolated between its samples, repeating or holding its last
    value after its end - a load's power, per unit of its set power, through
    an EMT run that starts t0 into the profile.
    """

    def __init__(self, t, p, t0=0.0, repeat=True):
        t = np.asarray(t, dtype=float)
        self.t, self.p = (_one_period(t - t[0], p) if repeat else (t - t[0], np.asarray(p, dtype=float)))
        self.t0, self.repeat = float(t0), bool(repeat)
        self.span = float(self.t[-1] - self.t[0])

    def __call__(self, t):
        x = self.t0 + t
        if self.repeat and x > self.t[-1]:
            x = np.mod(x, self.span)
        return float(np.interp(x, self.t, self.p))
