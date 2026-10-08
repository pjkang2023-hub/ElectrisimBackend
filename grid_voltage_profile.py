"""
A grid voltage that follows a profile, for the dynamic studies (ANDES and EMT).

The External Grid's voltage steps through a staircase from a start time: each
point (t, v) holds v - a share of the grid's own set voltage - from t after
the start until the next point. A preset gives IEEE 2800's low-voltage
ride-through envelope, as GE Vernova's 800 V AI factory paper draws it
(Figure 7): the lowest voltage a plant must ride through, for as long as it
must - under 0.25 pu for 0.32 s, then 0.25 pu to 1.2 s, 0.5 pu to 3 s,
0.7 pu to 6 s, then 0.9 pu, the edge of continuous operation. A custom
profile is a table, one "t_s, v_pu" per line.
"""
from typing import List, Tuple

IEEE_2800_LVRT = ((0.0, 0.0), (0.32, 0.25), (1.2, 0.5), (3.0, 0.7), (6.0, 0.9))
PRESETS = {'ieee2800': IEEE_2800_LVRT, 'ieee2800_lvrt': IEEE_2800_LVRT}


def profile_points(params) -> Tuple[List[Tuple[float, float]], List[str]]:
    """The profile's (t_s, v_pu) points, sorted, and any problems with it; none when there is no profile."""
    kind = str(params.get('grid_voltage_profile') or 'none').strip().lower()
    if kind in ('', 'none', 'off', 'false'):
        return [], []
    if kind in PRESETS:
        return list(PRESETS[kind]), []
    if kind != 'custom':
        return [], [f"Grid voltage profile '{kind}' is not known: none applied. "
                    "Choose none, IEEE 2800 LVRT or custom."]
    points, problems = [], []
    for n, line in enumerate(str(params.get('grid_voltage_table') or '').replace(';', '\n').splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = [p for p in line.replace('\t', ',').replace(' ', ',').split(',') if p]
        try:
            t, v = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            problems.append(f"Grid voltage profile, line {n} ('{line}'): give a time (s) and a voltage (pu).")
            continue
        if t < 0 or v < 0:
            problems.append(f"Grid voltage profile, line {n}: time and voltage cannot be negative.")
            continue
        points.append((t, v))
    if not points and not problems:
        problems.append('The custom grid voltage profile has no points: none applied.')
    return sorted(points), problems


def describe(points, start_s) -> str:
    steps = ', '.join(f'{v:g} pu at {start_s + t:g} s' for t, v in points)
    return f'The grid voltage follows its profile: {steps}.'
