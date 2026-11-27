"""
Dynamic studies built on the transient stability run (andes_electrisim.run_tds): the
critical clearing time of a fault, by search, and a batch of contingencies with one summary.

A case holds if its run reaches its end, no machine loses synchronism (run_tds's own test:
a rotor angle swinging by more than 180 degrees against its island) and the buses still fed
end above recover_pu. Each case is a full run on the same drawn network.

Critical clearing time: a coarse ladder of fault durations first, so that a case
holding again above the first failure is seen rather than missed, then bisection between the
last time that held and the first that did not. The time returned held.
"""
import copy
import json
import math

import andes_electrisim as ae

RECOVER_PU = 0.8
CUT_OFF_PU = 0.01       # below this a bus is taken as cut off, not as failing to recover


def _f(value, default):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _key(in_data):
    return next((k for k, v in in_data.items() if isinstance(v, dict) and 'Parameters' in str(v.get('typ', ''))), None)


def _run(in_data, params):
    """One run, reduced to what the two studies compare. A run that raises is a case that did not hold."""
    request = copy.deepcopy(in_data)
    key = _key(request)
    case = dict(params)
    if key is not None:
        request[key] = case
    try:
        out = json.loads(ae.run_tds(request, case).replace('-Infinity', 'null').replace('Infinity', 'null').replace('NaN', 'null'))
    except Exception as exc:   # noqa: BLE001
        return {'holds': False, 'reason': f'the run failed: {exc}', 'warnings': []}
    if out.get('error'):
        return {'holds': False, 'reason': str(out.get('message') or 'the run failed'), 'warnings': out.get('warnings') or []}
    warnings = out.get('warnings') or []
    lost = sorted({w.split("'")[1] for w in warnings if 'lost synchronism' in w and "'" in w})
    t = out.get('time') or []
    tf = _f(case.get('tf'), _f(out.get('tf'), 0.0))
    reached = bool(t) and t[-1] is not None and t[-1] >= tf - 1e-6
    ends = {}
    lows = {}
    for s in out.get('bus_voltage') or []:
        values = [x for x in (s.get('values') or []) if x is not None]
        if values:
            ends[s.get('name')] = values[-1]
            lows[s.get('name')] = min(values)
    fed = {k: v for k, v in ends.items() if v > CUT_OFF_PU}
    v_end = min(fed.values()) if fed else None
    recover = _f(case.get('recover_pu'), RECOVER_PU)
    failed_recovery = v_end is not None and v_end < recover
    holds = bool(out.get('converged')) and reached and not lost and not failed_recovery
    reason = None
    if not holds:
        reason = ('lost synchronism: ' + ', '.join(lost)) if lost else ('stopped at %.3f s of %.3f s' % (t[-1] if t else 0.0, tf)) if not reached \
            else 'did not converge' if not out.get('converged') else 'voltage %.3f pu at the end, under %.2f pu' % (v_end, recover)
    freq = [x for x in (out.get('frequency_hz') or []) if x is not None]
    return {'holds': holds, 'reason': reason, 'lost_synchronism': lost, 'run_end_s': t[-1] if t else None,
            'v_end_min_pu': v_end, 'v_min_pu': min(lows.values()) if lows else None, 'cut_off': sorted(str(k) for k, v in ends.items() if v <= CUT_OFF_PU),
            'f_min_hz': min(freq) if freq else None, 'f_max_hz': max(freq) if freq else None, 'warnings': warnings}


def critical_clearing(in_data, params):
    """The longest a bolted fault at params['fault_bus'] can last and the system hold, by search."""
    bus = str(params.get('fault_bus') or params.get('fault_bus_name') or '').strip()
    if not bus:
        return {'error': True, 'message': 'A critical clearing time needs a fault bus.'}
    t_on = _f(params.get('fault_tf'), 1.0)
    watch = _f(params.get('cct_watch_s'), 5.0)
    t_max = _f(params.get('cct_max_s'), 1.0)
    tol = max(_f(params.get('cct_tolerance_s'), 0.005), 1e-3)
    ladder = max(int(_f(params.get('cct_ladder'), 8)), 2)
    runs = []

    def holds(duration):
        case = {**params, 'fault_bus': bus, 'fault_enabled': 'true', 'fault_tf': str(t_on), 'fault_tc': str(t_on + duration), 'tf': str(t_on + duration + watch)}
        out = _run(in_data, case)
        runs.append({'duration_s': round(duration, 5), 'holds': out['holds'], 'reason': out.get('reason')})
        return out['holds']

    durations = [t_max * (k + 1) / ladder for k in range(ladder)]
    results = [holds(d) for d in durations]
    warnings = []
    if all(results):
        return {'error': False, 'fault_bus': bus, 'cct_s': None, 'at_least_s': t_max, 'runs': runs,
                'warnings': [f'It holds at every time tried, up to {t_max:g} s, so no critical clearing time was found under that.']}
    first_bad = results.index(False)
    if any(results[first_bad + 1:]):
        again = [durations[k] for k in range(first_bad + 1, ladder) if results[k]]
        warnings.append('It holds again at a longer fault than one that failed (%s s): this is not a single boundary, so read the runs, not only the time.' % ', '.join(f'{x:g}' for x in again))
    low = durations[first_bad - 1] if first_bad > 0 else 0.0
    high = durations[first_bad]
    if first_bad == 0 and not holds(tol):
        return {'error': False, 'fault_bus': bus, 'cct_s': 0.0, 'runs': runs, 'warnings': warnings + [f'It does not hold even at {tol:g} s.']}
    if first_bad == 0:
        low = tol
    while high - low > tol:
        mid = 0.5 * (low + high)
        if holds(mid):
            low = mid
        else:
            high = mid
    return {'error': False, 'fault_bus': bus, 'cct_s': round(low, 5), 'first_failure_s': round(high, 5), 'tolerance_s': tol, 'runs': runs, 'warnings': warnings}


def _auto_contingencies(in_data, params):
    """Every line out, every generator tripped, a bolted fault at every bus for the study's clearing time."""
    rows = [v for v in in_data.values() if isinstance(v, dict)]
    clear = _f(params.get('clear_s'), 0.1)
    out = []
    for r in rows:
        typ, name = str(r.get('typ', '')), r.get('name')
        label = r.get('userFriendlyName') or name
        if typ.startswith('Line') and not typ.startswith('Line DC'):
            out.append({'name': f'Line {label} out', 'type': 'line', 'element': name})
        elif typ.startswith('Generator'):
            out.append({'name': f'Generator {label} trip', 'type': 'gen', 'element': name})
        elif typ.startswith('Bus') and not typ.startswith('Busbar DC'):
            out.append({'name': f'Fault at {label}, {clear * 1e3:g} ms', 'type': 'fault', 'element': name, 'clear_s': clear})
    return out


def contingency_batch(in_data, params):
    """Each contingency run alone; one summary, worst first. contingencies: a list, or 'auto' (the default)."""
    given = params.get('contingencies') or 'auto'
    if isinstance(given, str) and given.strip().lower() != 'auto':
        try:
            given = json.loads(given)
        except ValueError:
            return {'error': True, 'message': 'contingencies is neither "auto" nor a JSON list.'}
    cases = _auto_contingencies(in_data, params) if isinstance(given, str) else list(given)
    t_on = _f(params.get('event_t_s'), 1.0)
    tf = t_on + _f(params.get('watch_s'), 5.0)
    rows = []
    base = {**params, 'fault_enabled': 'false', 'fault_bus': '', 'fault_bus_name': '', 'toggle_line': '', 'line_outage': '', 'toggle_gen': '', 'generator_trip': '', 'tf': str(tf)}
    for c in cases:
        kind = str(c.get('type', '')).lower()
        el = c.get('element')
        case = dict(base)
        if kind == 'fault':
            clear = _f(c.get('clear_s'), 0.1)
            case.update(fault_enabled='true', fault_bus=el, fault_tf=str(t_on), fault_tc=str(t_on + clear))
            if c.get('trip_line'):
                case.update(toggle_line=c['trip_line'], toggle_t=str(t_on + clear))
        elif kind == 'line':
            case.update(toggle_line=el, toggle_t=str(t_on))
        elif kind == 'gen':
            case.update(toggle_gen=el, toggle_gen_t=str(t_on))
        else:
            rows.append({'name': c.get('name') or str(el), 'holds': False, 'reason': f"no contingency type '{kind}'"})
            continue
        out = _run(in_data, case)
        rows.append({'name': c.get('name') or f'{kind} {el}', 'type': kind, 'element': el, **{k: v for k, v in out.items() if k != 'warnings'},
                     'warnings': [w for w in out['warnings'] if 'lost synchronism' not in w][:3]})
    rows.sort(key=lambda r: (r['holds'], r.get('v_end_min_pu') if r.get('v_end_min_pu') is not None else 0.0))
    return {'error': False, 'n': len(rows), 'n_failed': sum(1 for r in rows if not r['holds']), 'tf': tf, 'results': rows}
