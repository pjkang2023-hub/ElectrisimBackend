"""
A power converter's rating where it runs: derated by its inlet air temperature,
its altitude and its AC voltage, with a short-time overload above it and a cap
on its reactive power.

- Temperature and altitude: each a curve of rating multiplier against inlet air
  (deg C) or altitude (m), linear between its points and flat beyond its ends.
  The rating is the *minimum* of the factors, not their product (the String PCS
  derating rule): a cool inlet's 110 % does not lift a high site's 86 %. An
  altitude within its full range derates nothing and caps nothing.
- AC voltage: below its nominal voltage it is current-limited, its apparent
  power falling with the voltage (S x V / Vn); above it, power-capped at S.
  The PCS3750C-MV spec's 88 % at 444 V and 45 deg C (444/505 of 100 %) and 96 %
  at 35 deg C (444/505 of 110 %) are this, on the temperature-derated rating.
- Short-time overload: steps of (seconds, multiple of the derated continuous
  rating), '1:1.5, 10:1.2' - 150 % for 1 s, then 120 % up to 10 s, then the
  continuous rating.
- Reactive power: Q <= min(sqrt(S^2 - P^2), q_max S), its circle capped at
  q_max of its rating (0.6 for the String PCS).

Blank fields leave a converter at its rating: a curve with no inlet or altitude
given, an inlet or altitude with no curve.
"""

import math

# The altitude curve when an altitude is given without one: full rating to 2,000 m, 86 % at
# 4,000 m (the String PCS; IEC 62109-1's 1 % per 100 m above 2,000 m is close to it).
ALTITUDE_CURVE = '2000:1.0, 4000:0.86'


def _f(value, default=float('nan')):
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def parse_curve(text):
    """[(x, y), ...] sorted by x, from 'x:y' pairs separated by commas, semicolons or new lines."""
    pairs = []
    for part in str(text or '').replace(';', ',').replace('\n', ',').split(','):
        if ':' in part:
            a, b = (_f(s) for s in part.split(':', 1))
            if math.isfinite(a) and math.isfinite(b):
                pairs.append((a, b))
    return sorted(pairs)


def curve_at(curve, x):
    """A curve's value at x: linear between its points, flat beyond its ends (1.0 for no curve)."""
    if not curve:
        return 1.0
    if x <= curve[0][0]:
        return curve[0][1]
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else y1
    return curve[-1][1]


def derating(el):
    """
    A converter's derating from its element's fields: {'factor', 'temperature',
    'altitude', 'inlet_c', 'altitude_m'} - the factor the minimum of the two,
    each 1.0 when not given (with a note in 'notes' for a half-given one).
    """
    out = {'factor': 1.0, 'temperature': 1.0, 'altitude': 1.0, 'inlet_c': None, 'altitude_m': None, 'notes': []}
    inlet, t_curve = _f(el.get('inlet_c')), parse_curve(el.get('derate_temp_curve'))
    if math.isfinite(inlet):
        out['inlet_c'] = inlet
        if t_curve:
            out['temperature'] = max(curve_at(t_curve, inlet), 0.0)
        else:
            out['notes'].append(f'its inlet air ({inlet:g} deg C) is given without a temperature derating curve, '
                                'so it is not derated for it')
    alt = _f(el.get('altitude_m'))
    if math.isfinite(alt):
        out['altitude_m'] = alt
        out['altitude'] = max(curve_at(parse_curve(el.get('derate_altitude_curve') or ALTITUDE_CURVE), alt), 0.0)
    # The minimum of the factors that derate: within its full range an altitude derates nothing, and
    # does not cap a cool inlet's 110 % (the spec's 4,166 kVA at 35 deg C holds at a low site).
    out['factor'] = min(out['temperature'], out['altitude']) if out['altitude'] < 1.0 else out['temperature']
    return out


def voltage_factor(v_pu):
    """Its apparent power at an AC voltage of v_pu, per unit of its rating: current-limited below 1, capped above."""
    return min(max(v_pu, 0.0), 1.0) if math.isfinite(v_pu) else 1.0


def q_limit(s, p, q_max_pu=None):
    """The reactive power (same unit as s) it can give at active power p: its circle, capped at q_max_pu of s."""
    circle = math.sqrt(max(s * s - p * p, 0.0))
    if q_max_pu is None or not math.isfinite(q_max_pu) or q_max_pu <= 0:
        return circle
    return min(circle, q_max_pu * s)


def q_max_pu(el):
    """Its reactive-power cap, per unit of its rating, from 'q_max_percent' (None: its circle only)."""
    v = _f(el.get('q_max_percent'))
    return v / 100.0 if math.isfinite(v) and v > 0 else None


def overload_steps(text):
    """[(seconds, multiple), ...] by duration, each multiple above 1 (blank: none)."""
    return [(t, m) for t, m in parse_curve(text) if t > 0 and m > 1.0]


def overload_ceiling(steps, t_over):
    """The multiple of its continuous rating it may carry ``t_over`` seconds into an overload."""
    for t, m in steps:
        if t_over < t:
            return m
    return 1.0


def overload_time(steps, ratio):
    """How long (s) it may carry ``ratio`` times its continuous rating: inf at or under 1, 0 beyond its highest step."""
    if ratio <= 1.0 + 1e-9:
        return math.inf
    longest = 0.0
    for t, m in steps:
        if ratio <= m + 1e-9:
            longest = max(longest, t)
    return longest


class OverloadClock:
    """
    Its overload as time goes on: the time it has spent above its continuous
    rating sets the ceiling it may reach next (overload_ceiling). Back at or
    under its continuous rating for ``reset_s`` it has cooled and the clock
    starts again; a shorter dip leaves the clock where it was.
    """

    def __init__(self, steps, reset_s=60.0):
        self.steps = list(steps)
        self.reset_s = reset_s
        self.t_over = 0.0
        self.t_under = 0.0

    def ceiling(self):
        return overload_ceiling(self.steps, self.t_over)

    def step(self, ratio, dt):
        """Advance by dt at ``ratio`` times its continuous rating; returns its ceiling for the next step."""
        if ratio > 1.0 + 1e-9:
            self.t_over += dt
            self.t_under = 0.0
        else:
            self.t_under += dt
            if self.t_under >= self.reset_s:
                self.t_over = 0.0
        return self.ceiling()
