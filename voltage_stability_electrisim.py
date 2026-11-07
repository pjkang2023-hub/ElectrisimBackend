"""
Voltage stability curves on the drawn network, by load flow.

PV curve: every load raised together at its own power factor, by a factor lam from the drawn
network, each step started from the last, until the load flow no longer solves; then bisection
to the last factor that does. That is the nose, to the tolerance set. Only the upper branch is
traced: past the nose there is no solution at higher load, and this is not a continuation load
flow, so the lower branch is not drawn. A solution dropping suddenly from the last (a step onto
the lower branch) is taken as past the nose. Generators stay at their set power unless
scale_gens, so the slack bus takes the rise.

QV curve: a fictitious generator at the bus holds its voltage, stepped down from v_max_pu; the
reactive power it supplies, against that voltage. Q at its lowest is the bus's reactive margin:
its negative is how much reactive load the bus can take before collapse, and a positive lowest
Q means the bus already needs that much support.

The nose ignores thermal limits, so the first line or transformer past its rating along the curve is
reported apart, first_thermal_limit, by interpolating between the steps: usually the margin that matters.

Reactive limits are not enforced on the drawn generators: a converter or machine at its limit would
bring the nose in. Each curve is at the network's own topology: run it again with an element out for N-1.
"""
import copy
import math

import numpy as np
import pandapower as pp

import pandapower_electrisim as pe

JUMP_PU = 0.15          # a fall between steps larger than this is off the upper branch


def _f(value, default):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _names(net):
    names = getattr(net, 'user_friendly_names', {}) or {}
    return {b: str(names.get(net.bus.at[b, 'name']) or net.bus.at[b, 'name']) for b in net.bus.index}


def _hidden(net):
    aux = net.bus['electrisim_aux'] == True if 'electrisim_aux' in net.bus.columns else None
    return {b for b in net.bus.index if aux is not None and aux.at[b]}


def _solve(net, lam, base, warm):
    """A copy of net at load factor lam, solved from net's results; None if it does not solve."""
    trial = copy.deepcopy(net)
    trial.load['scaling'] = base['load'] * lam
    if base['gen'] is not None:
        trial.gen['p_mw'] = base['gen'] * lam
    if base['sgen'] is not None:
        trial.sgen['scaling'] = base['sgen'] * lam
    try:
        pe._electrisim_runpp(trial, algorithm='nr', calculate_voltage_angles=True, init='results' if warm else 'auto', max_iteration=30)
    except pp.LoadflowNotConverged:
        return None
    vm = trial.res_bus['vm_pu'].to_numpy()
    if not np.isfinite(vm).any():
        return None
    return trial


def _heaviest(n):
    """The most loaded line or transformer, per cent of its rating, and its name."""
    names = getattr(n, 'user_friendly_names', {}) or {}
    best = (0.0, None)
    for table in ('line', 'trafo', 'trafo3w'):
        res = n[f'res_{table}'] if f'res_{table}' in n else None
        if res is None or not len(res) or 'loading_percent' not in res.columns:
            continue
        loading = res['loading_percent'].to_numpy(dtype=float)
        if not np.isfinite(loading).any():
            continue
        k = int(np.nanargmax(loading))
        if loading[k] > best[0]:
            raw = str(n[table].at[res.index[k], 'name'])
            best = (float(loading[k]), str(names.get(raw) or raw))
    return best


def _first_limit(points, heaviest):
    """Where the heaviest loading first passes 100 %, interpolated; its load factor, power and element."""
    if heaviest[0][0] > 100.0:
        return {'lambda': 1.0, 'p_total_mw': round(points[0][1], 4), 'element': heaviest[0][1], 'loading_percent': round(heaviest[0][0], 2), 'margin_mw': 0.0, 'already_over': True}
    for k in range(1, len(points)):
        if heaviest[k][0] > 100.0:
            a = (100.0 - heaviest[k - 1][0]) / max(heaviest[k][0] - heaviest[k - 1][0], 1e-9)
            lam = points[k - 1][0] + a * (points[k][0] - points[k - 1][0])
            p = points[k - 1][1] + a * (points[k][1] - points[k - 1][1])
            return {'lambda': round(lam, 5), 'p_total_mw': round(p, 4), 'element': heaviest[k][1], 'loading_percent': 100.0,
                    'margin_mw': round(p - points[0][1], 4), 'already_over': False}
    return None


def pv_curve(net, params):
    tol = max(_f(params.get('tolerance'), 0.002), 1e-4)
    step = max(_f(params.get('step'), 0.1), tol)
    lam_max = _f(params.get('lambda_max'), 20.0)
    scale_gens = str(params.get('scale_gens') or '').lower() in ('true', '1', 'yes')
    if not len(net.load):
        return {'error': True, 'message': 'A PV curve needs a load to raise.'}
    base = {'load': net.load['scaling'].to_numpy(dtype=float).copy(),
            'gen': net.gen['p_mw'].to_numpy(dtype=float).copy() if scale_gens and len(net.gen) else None,
            'sgen': net.sgen['scaling'].to_numpy(dtype=float).copy() if scale_gens and len(net.sgen) else None}
    last = _solve(net, 1.0, base, False)
    if last is None:
        return {'error': True, 'message': 'The load flow does not solve at the drawn load.'}
    names, hidden = _names(net), _hidden(net)
    shown = [b for b in net.bus.index if b not in hidden and np.isfinite(last.res_bus.at[b, 'vm_pu'])]

    def p_total(n):
        return float(n.res_load['p_mw'].sum())

    def voltages(n):
        return {b: float(n.res_bus.at[b, 'vm_pu']) for b in shown}

    def on_branch(n, prev):
        return n is not None and all(np.isfinite(n.res_bus.at[b, 'vm_pu']) and n.res_bus.at[b, 'vm_pu'] >= prev.res_bus.at[b, 'vm_pu'] - JUMP_PU for b in shown)

    points = [(1.0, p_total(last), voltages(last))]
    heavy = [_heaviest(last)]
    lam = 1.0
    bound = None
    while lam < lam_max:
        trial = _solve(last, min(lam + step, lam_max), base, True)
        if not on_branch(trial, last):
            bound = lam + step
            break
        lam = min(lam + step, lam_max)
        last = trial
        points.append((lam, p_total(last), voltages(last)))
        heavy.append(_heaviest(last))
    if bound is None:
        warnings = [f'No nose found up to {lam_max:g} times the drawn load.']
    else:
        warnings = []
        low, high = lam, bound
        while high - low > tol:
            mid = 0.5 * (low + high)
            trial = _solve(last, mid, base, True)
            if on_branch(trial, last):
                low, last = mid, trial
                points.append((mid, p_total(last), voltages(last)))
                heavy.append(_heaviest(last))
            else:
                high = mid
        lam = low
    p0 = points[0][1]
    nose_v = voltages(last)
    weakest = min(nose_v, key=lambda b: nose_v[b])
    return {'error': False, 'lambda': [round(p[0], 5) for p in points], 'p_total_mw': [round(p[1], 4) for p in points],
            'buses': [{'name': names[b], 'vm_pu': [round(p[2][b], 5) for p in points]} for b in shown],
            'nose': None if bound is None else {'lambda': round(lam, 5), 'p_total_mw': round(p_total(last), 4), 'margin_mw': round(p_total(last) - p0, 4),
                                               'margin_percent': round(100.0 * (lam - 1.0), 3), 'weakest_bus': names[weakest], 'weakest_vm_pu': round(nose_v[weakest], 5)},
            'first_thermal_limit': _first_limit(points, heavy), 'max_loading_percent': [round(x[0], 2) for x in heavy],
            'p_base_mw': round(p0, 4), 'warnings': warnings + list(getattr(last, 'warnings', []) or [])}


def qv_curve(net, params):
    name = str(params.get('qv_bus') or '').strip()
    hit = [b for b in net.bus.index if name and name in (str(net.bus.at[b, 'name']), _names(net)[b])]
    if not hit:
        return {'error': True, 'message': f"A QV curve needs a bus: no bus '{name}'."}
    bus = hit[0]
    v_max, v_min = _f(params.get('v_max_pu'), 1.1), _f(params.get('v_min_pu'), 0.4)
    step = max(_f(params.get('v_step_pu'), 0.02), 1e-3)
    work = copy.deepcopy(net)
    g = pp.create_gen(work, bus, p_mw=0.0, vm_pu=v_max, name='QV curve', slack=False)
    points = []
    warnings = []
    warm = False
    v = v_max
    while v >= v_min - 1e-9:
        work.gen.at[g, 'vm_pu'] = v
        try:
            pe._electrisim_runpp(work, algorithm='nr', calculate_voltage_angles=True, init='results' if warm else 'auto', max_iteration=30)
        except pp.LoadflowNotConverged:
            warnings.append(f'The load flow did not solve at {v:.3f} pu, so the curve stops there.')
            break
        points.append((round(v, 5), float(work.res_gen.at[g, 'q_mvar'])))
        warm = True
        v -= step
    if not points:
        return {'error': True, 'message': 'The load flow did not solve at any voltage tried.', 'warnings': warnings}
    k = int(np.argmin([q for _, q in points]))
    q_min, v_crit = points[k][1], points[k][0]
    if k == len(points) - 1:
        warnings.append('Q is lowest at the end of the range: widen it to find the true minimum.')
    return {'error': False, 'bus': _names(net)[bus], 'v_pu': [p[0] for p in points], 'q_mvar': [round(p[1], 4) for p in points],
            'q_min_mvar': round(q_min, 4), 'v_at_q_min_pu': v_crit, 'reactive_margin_mvar': round(-q_min, 4), 'warnings': warnings}


def study(net, params):
    kind = str(params.get('curve') or 'pv').strip().lower()
    return qv_curve(net, params) if kind == 'qv' else pv_curve(net, params)
