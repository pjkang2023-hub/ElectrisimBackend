"""
State estimation (pandapower) on the drawn network.

The measurement set is either simulated from a load flow of the drawn network -
meters placed by rule, each reading the load flow plus a random error of its
accuracy class, to judge a metering plan against the state it should recover -
or entered as a table of real readings. Loads and generation without a meter
can be added as pseudo-measurements of their drawn values with a wide error.

The weighted least squares (WLS) estimate is tested for bad data with the
chi-square test of its objective J, and bad readings are identified and removed
one at a time by the largest normalized residual test. The robust least
absolute value (LAV) estimator is offered as an alternative.
"""
import csv
import io
import json
import logging
import math
import warnings

import numpy as np
import pandapower as pp
from scipy.stats import chi2

# pandapower's estimation module still calls np.in1d and np.linalg.linalg,
# both removed in NumPy 2.
if not hasattr(np, 'in1d'):
    np.in1d = np.isin
if not hasattr(np.linalg, 'linalg'):
    np.linalg.linalg = np.linalg

from pandapower.estimation.state_estimation import StateEstimation  # noqa: E402

# The branch tables measured, the sides each has, and the result table of each.
_BRANCH_SIDES = {
    'line': ('from', 'to'),
    'trafo': ('hv', 'lv'),
    'trafo3w': ('hv', 'mv', 'lv'),
}
_ELEMENT_TYPE_ALIASES = {
    'bus': 'bus', 'busbar': 'bus',
    'line': 'line', 'cable': 'line',
    'trafo': 'trafo', 'transformer': 'trafo',
    'trafo3w': 'trafo3w', 'three_winding_transformer': 'trafo3w', '3w': 'trafo3w',
}
# Elements whose power is a bus injection a meter (or a pseudo-measurement) sees.
_INJECTION_TABLES = ('load', 'motor', 'storage', 'sgen', 'gen', 'ext_grid',
                     'asymmetric_load', 'asymmetric_sgen')
_SOURCE_TABLES = ('ext_grid', 'gen', 'sgen', 'storage')
_MIN_STD = {'v': 1e-5, 'p': 1e-4, 'q': 1e-4}
_MAX_BAD_DATA_REMOVALS = 10
_CSV_HEADER = ['type', 'element_type', 'element', 'side', 'value', 'std_dev']


def _clean(value):
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _f(value, default):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _flag(value, default=False):
    if value is None or value == '':
        return default
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


def _display_name(net, table, idx):
    """The dialog name of an element, else its internal name."""
    internal = net[table].at[idx, 'name'] if 'name' in net[table].columns else None
    internal = str(internal) if internal not in (None, '') and internal == internal else f'{table} {idx}'
    names = getattr(net, 'user_friendly_names', None) or {}
    for key in (internal, internal.replace('#', '_'), internal.replace('_', '#')):
        if names.get(key) not in (None, ''):
            return str(names[key])
    return internal


def _cell_id(net, table, idx):
    if 'id' in net[table].columns:
        value = net[table].at[idx, 'id']
        if value not in (None, '') and value == value:
            return str(value)
    return None


def _name_lookup(net):
    """{element_type: {name: index}} by dialog and by internal name."""
    lookup = {}
    for table in ('bus',) + tuple(_BRANCH_SIDES):
        names = {}
        for idx in net[table].index:
            names.setdefault(_display_name(net, table, idx), int(idx))
            internal = net[table].at[idx, 'name'] if 'name' in net[table].columns else None
            if internal not in (None, '') and internal == internal:
                names.setdefault(str(internal), int(idx))
        lookup[table] = names
    return lookup


def _branch_energized(net, table, idx):
    """In service, with no open breaker on it, and its ends solved by the load flow."""
    if not bool(net[table].at[idx, 'in_service']):
        return False
    et = {'line': 'l', 'trafo': 't', 'trafo3w': 't3'}[table]
    sw = net.switch
    if len(sw) and ((sw['et'] == et) & (sw['element'] == idx) & ~sw['closed'].astype(bool)).any():
        return False
    res = net[f'res_{table}'] if f'res_{table}' in net else None
    if res is not None and idx in res.index:
        first = 'p_from_mw' if table == 'line' else 'p_hv_mw'
        return bool(np.isfinite(res.at[idx, first]))
    return True


def _branch_rating_mva(net, table, idx, side):
    row = net[table].loc[idx]
    if table == 'line':
        vn = float(net.bus.at[int(row['from_bus']), 'vn_kv'])
        return math.sqrt(3) * vn * _f(row.get('max_i_ka'), 0) * _f(row.get('df'), 1) * _f(row.get('parallel'), 1)
    if table == 'trafo':
        return _f(row.get('sn_mva'), 0) * _f(row.get('parallel'), 1)
    return _f(row.get(f'sn_{side}_mva'), 0)


def _bus_injections(net):
    """
    Each bus's P and Q in the load convention as a meter sees them: the load
    flow's bus result less what its shunts draw - a shunt belongs to the
    network model, not to the measured injection.
    """
    p = net.res_bus['p_mw'].astype(float).copy()
    q = net.res_bus['q_mvar'].astype(float).copy()
    if len(net.shunt) and 'res_shunt' in net and len(net.res_shunt):
        for i in net.shunt.index:
            if i in net.res_shunt.index:
                b = int(net.shunt.at[i, 'bus'])
                p[b] -= _f(net.res_shunt.at[i, 'p_mw'], 0.0)
                q[b] -= _f(net.res_shunt.at[i, 'q_mvar'], 0.0)
    return p, q


def _buses_with(net, tables):
    buses = set()
    for table in tables:
        if table in net and len(net[table]):
            live = net[table]['in_service'].astype(bool) if 'in_service' in net[table].columns else True
            buses.update(int(b) for b in net[table].loc[live, 'bus'])
    return buses


def _simulated_measurements(net, params, warnings_out):
    """Meters placed by rule, reading the load flow plus their random error."""
    rng = np.random.default_rng(int(_f(params.get('seed'), 1)))
    noise = _flag(params.get('add_noise'), True)
    v_std = _f(params.get('v_std_percent'), 0.5) / 100.0
    pq_pct = _f(params.get('pq_std_percent'), 1.0) / 100.0
    v_rule = str(params.get('v_meas') or 'all')
    flow_rule = str(params.get('flow_meas') or 'all_ends')
    injections = _flag(params.get('injection_meas'), True)

    rows = []

    def add(mtype, etype, element, side, true_value, std):
        std = max(float(std), _MIN_STD[mtype])
        value = float(true_value) + (float(rng.normal(0.0, std)) if noise else 0.0)
        rows.append(dict(type=mtype, element_type=etype, element=int(element), side=side,
                         value=value, std_dev=std, source='meter', true_value=float(true_value)))

    live_bus = net.res_bus.index[np.isfinite(net.res_bus['vm_pu'].astype(float))]
    if v_rule != 'none':
        chosen = set(int(b) for b in live_bus)
        if v_rule == 'sources':
            chosen &= _buses_with(net, _SOURCE_TABLES)
        for b in sorted(chosen):
            add('v', 'bus', b, None, net.res_bus.at[b, 'vm_pu'], v_std)

    if flow_rule != 'none':
        for table, sides in _BRANCH_SIDES.items():
            res = net[f'res_{table}']
            for idx in net[table].index:
                if not _branch_energized(net, table, idx):
                    continue
                for side in (sides if flow_rule == 'all_ends' else sides[:1]):
                    rating = _branch_rating_mva(net, table, idx, side)
                    p, q = res.at[idx, f'p_{side}_mw'], res.at[idx, f'q_{side}_mvar']
                    std = pq_pct * (rating if rating > 0 else max(math.hypot(p, q), 1.0))
                    add('p', table, idx, side, p, std)
                    add('q', table, idx, side, q, std)

    if injections:
        p_inj, q_inj = _bus_injections(net)
        for b in sorted(_buses_with(net, _INJECTION_TABLES) & set(int(x) for x in live_bus)):
            std = pq_pct * max(math.hypot(p_inj[b], q_inj[b]), 0.1)
            add('p', 'bus', b, None, p_inj[b], std)
            add('q', 'bus', b, None, q_inj[b], std)
    if not rows:
        warnings_out.append('The placement rules put no meter on the network.')
    return rows


def _entered_measurements(net, text, errors_out):
    """Readings from the dialog's table: type, element_type, element, side, value, std_dev."""
    lookup = _name_lookup(net)
    rows = []
    reader = csv.reader(io.StringIO(text or ''))
    for line_no, cells in enumerate(reader, start=1):
        cells = [c.strip() for c in cells]
        if not any(cells) or cells[0].startswith('#'):
            continue
        if [c.lower() for c in cells[:len(_CSV_HEADER)]] == _CSV_HEADER:
            continue
        if len(cells) < 6:
            errors_out.append(f'Line {line_no}: expected {", ".join(_CSV_HEADER)}.')
            continue
        mtype, etype, element, side, value, std = cells[:6]
        mtype = mtype.lower()
        etype = _ELEMENT_TYPE_ALIASES.get(etype.lower().replace(' ', '_'))
        if mtype not in ('v', 'p', 'q'):
            errors_out.append(f'Line {line_no}: type "{cells[0]}" is not v, p or q.')
            continue
        if etype is None:
            errors_out.append(f'Line {line_no}: element type "{cells[1]}" is not bus, line, trafo or trafo3w.')
            continue
        if mtype == 'v' and etype != 'bus':
            errors_out.append(f'Line {line_no}: a voltage is measured at a bus.')
            continue
        idx = lookup[etype].get(element)
        if idx is None:
            errors_out.append(f'Line {line_no}: no {etype} named "{element}".')
            continue
        side = side.lower() or None
        if etype == 'bus':
            side = None
        elif side not in _BRANCH_SIDES[etype]:
            errors_out.append(f'Line {line_no}: the side of a {etype} is {" or ".join(_BRANCH_SIDES[etype])}.')
            continue
        value, std = _f(value, None), _f(std, None)
        if value is None or std is None or std <= 0:
            errors_out.append(f'Line {line_no}: the value must be a number and the standard deviation positive.')
            continue
        rows.append(dict(type=mtype, element_type=etype, element=int(idx), side=side,
                         value=value, std_dev=std, source='entered', true_value=None))
    return rows


def _pseudo_measurements(net, rows, params, lf_ok):
    """Drawn loads and generation as wide-error injections at buses without a meter on them."""
    if not lf_ok:
        return []
    pct = _f(params.get('pseudo_std_percent'), 30.0) / 100.0
    metered = {(r['type'], r['element']) for r in rows if r['element_type'] == 'bus'}
    p_inj, q_inj = _bus_injections(net)
    out = []
    for b in sorted(_buses_with(net, _INJECTION_TABLES)):
        if b not in net.res_bus.index or not np.isfinite(net.res_bus.at[b, 'vm_pu']):
            continue
        std = max(pct * math.hypot(p_inj[b], q_inj[b]), 0.01)
        for mtype, value in (('p', p_inj[b]), ('q', q_inj[b])):
            if (mtype, b) not in metered:
                out.append(dict(type=mtype, element_type='bus', element=b, side=None, value=float(value),
                                std_dev=std, source='pseudo', true_value=None))
    return out


def _estimated_value(net, row):
    """The estimate's value of what a measurement reads."""
    etype, idx, side, mtype = row['element_type'], row['element'], row['side'], row['type']
    if etype == 'bus':
        res = net.res_bus_est
        if idx not in res.index:
            return None
        if mtype == 'v':
            return _f(res.at[idx, 'vm_pu'], None)
        value = _f(res.at[idx, 'p_mw' if mtype == 'p' else 'q_mvar'], None)
        if value is None:
            return None
        # As _bus_injections: the shunts' draw at the estimated voltage is not
        # part of the measured injection.
        vm = _f(res.at[idx, 'vm_pu'], 1.0)
        for i in net.shunt.index[(net.shunt['bus'] == idx) & net.shunt['in_service'].astype(bool)]:
            step = _f(net.shunt.at[i, 'step'], 1.0) if 'step' in net.shunt.columns else 1.0
            col = 'p_mw' if mtype == 'p' else 'q_mvar'
            value -= _f(net.shunt.at[i, col], 0.0) * step * vm ** 2
        return value
    res = net[f'res_{etype}_est']
    if idx not in res.index:
        return None
    return _f(res.at[idx, f'{mtype}_{side}_{"mw" if mtype == "p" else "mvar"}'], None)


def _load_measurements(net, rows):
    net.measurement.drop(net.measurement.index, inplace=True)
    for i, row in enumerate(rows):
        pp.create_measurement(net, row['type'], row['element_type'], row['value'], row['std_dev'],
                              row['element'], side=row['side'], index=i)


def _normalized_residuals(se):
    """{measurement index: normalized residual} of the last WLS estimate, and J."""
    s = se.solver
    r = np.asarray(s.r, dtype=float).ravel()
    R = np.linalg.inv(s.R_inv)
    omega = np.diag(R - s.H @ np.linalg.inv(s.Gm) @ s.H.T)
    rn = np.abs(r) / np.sqrt(np.maximum(np.abs(omega), 1e-30))
    J = float(r @ s.R_inv @ r)
    known = set(int(i) for i in se.net.measurement.index)
    out = {}
    # Zero-injection virtual measurements have no pandapower index.
    for k, meas_idx in enumerate(np.asarray(s.pp_meas_indices, dtype=float).ravel()):
        if np.isfinite(meas_idx) and int(meas_idx) in known:
            out[int(meas_idx)] = float(rn[k])
    return out, J, len(r), int(s.H.shape[1])


def _no_injection_buses(net):
    """
    The buses with nothing drawing or injecting in service and no P or Q
    measurement - what pandapower's 'no_inj_bus' finds. Its own search pairs
    each element's in-service flag with its bus's by index, which misaligns
    once bus numbers are not among the element indices (an IndexError, read
    as an unobservable network): so the list is given instead.
    """
    busy = set()
    for table in ('load', 'motor', 'sgen', 'storage', 'ward', 'xward', 'gen', 'ext_grid', 'shunt',
                  'asymmetric_load', 'asymmetric_sgen'):
        df = net[table] if table in net else None
        if df is not None and len(df):
            busy.update(int(b) for b in df.loc[df['in_service'].astype(bool), 'bus'])
    m = net.measurement
    if len(m):
        busy.update(int(b) for b in m.loc[(m['element_type'] == 'bus') & m['measurement_type'].isin(['p', 'q']), 'element'])
    return [int(b) for b in net.bus.index[net.bus['in_service'].astype(bool)] if int(b) not in busy]


def _run_estimate(net, estimator, zero_injection, max_iterations):
    algorithm = 'lp' if estimator == 'lav' else 'wls'
    se = StateEstimation(net, tolerance=1e-6, maximum_iterations=max_iterations, algorithm=algorithm)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        outcome = se.estimate(v_start='flat', delta_start='flat',
                              zero_injection=_no_injection_buses(net) if zero_injection else 'aux_bus',
                              algorithm=algorithm)
    ok = outcome['success'] if isinstance(outcome, dict) else bool(outcome)
    iterations = outcome.get('num_iterations') if isinstance(outcome, dict) else None
    if ok and algorithm == 'wls':
        r = getattr(se.solver, 'r', None)
        if r is None or not np.all(np.isfinite(np.asarray(r, dtype=float))):
            ok = False
    if ok and not np.all(np.isfinite(net.res_bus_est['vm_pu'].astype(float).dropna())):
        ok = False
    return se, ok, iterations


def _unobservable_message(m, n):
    hint = ('Add voltage, flow or injection measurements, or pseudo-measurements of the drawn '
            'loads and generation, so every bus is reached.')
    if m < n:
        return (f'The network is not observable: {m} measurement{"s" if m != 1 else ""} for {n} '
                f'state variables (two per bus, less the reference angle). {hint}')
    return f'The estimate did not converge: the measurements do not make the network observable. {hint}'


def state_estimation(net, params, in_data=None):
    params = params or {}
    logging.getLogger('pandapower.estimation').setLevel(logging.ERROR)
    warnings_out, input_errors = [], []

    source = str(params.get('measurement_source') or 'simulated')
    estimator = 'lav' if str(params.get('estimator') or 'wls').lower() == 'lav' else 'wls'
    bad_data = _flag(params.get('bad_data'), True) and estimator == 'wls'
    rn_threshold = _f(params.get('rn_threshold'), 3.0)
    confidence = min(0.9999, max(0.5, _f(params.get('chi2_confidence'), 0.95)))
    zero_injection = _flag(params.get('zero_injection'), True)
    max_iterations = int(_f(params.get('max_iterations'), 50))

    import pandapower_electrisim as pe
    # pandapower's estimator has no VSC or DC network (an IndexError, reported as an
    # unobservable network): the network is the AC one, each converter the load it draws.
    pe._electrisim_set_aside_dc_network(
        net, 'State estimation', why="pandapower's state estimation does not model VSCs or DC networks",
        keep_ac_draw=True)
    warnings_out.extend(getattr(net, 'warnings', None) or [])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # Electrisim's load flow: it settles the PCS and the sources behind them,
            # which pandapower's alone does not. With the angles, as the estimator
            # takes the transformers' phase shifts.
            pe._electrisim_runpp(net, calculate_voltage_angles=True, init='auto')
        lf_ok = bool(net.converged)
    except Exception:
        lf_ok = False
    if source == 'simulated' and not lf_ok:
        return json.dumps({'error': True, 'message': (
            'The load flow of the drawn network did not converge, so no measurements can be '
            'simulated from it. Run the load flow and correct the network first.')})

    if source == 'simulated':
        rows = _simulated_measurements(net, params, warnings_out)
    else:
        rows = _entered_measurements(net, params.get('measurements_csv'), input_errors)
    if _flag(params.get('pseudo_measurements'), source == 'entered'):
        rows += _pseudo_measurements(net, rows, params, lf_ok)
    if not lf_ok:
        warnings_out.append('The load flow of the drawn network did not converge: no comparison with it.')
    if not rows:
        message = 'There are no measurements to estimate from.'
        if input_errors:
            message += ' ' + ' '.join(input_errors[:5])
        return json.dumps({'error': True, 'message': message})

    # Two states per bus - a three-winding transformer's star point is one too -
    # less the reference angle.
    live_buses = int(net.res_bus['vm_pu'].notna().sum()) if lf_ok else int(net.bus['in_service'].sum())
    n_states = 2 * (live_buses + int(net.trafo3w['in_service'].sum())) - 1
    removed = []
    active = list(range(len(rows)))
    rn_by_row, J, m, n = {}, None, len(rows), n_states
    while True:
        _load_measurements(net, [rows[i] for i in active])
        try:
            se, ok, iterations = _run_estimate(net, estimator, zero_injection, max_iterations)
        except Exception as e:  # singular gain matrix, an unsupported element, ...
            se, ok, iterations = None, False, None
            failure = f'{type(e).__name__}: {e}'
        else:
            failure = None
        if not ok:
            if removed:
                # The last removal left the network unobservable: put it back.
                last = removed.pop()
                active.append(last['row'])
                active.sort()
                warnings_out.append(
                    f"Removing {last['label']} would leave the network unobservable: it is kept, "
                    'although its normalized residual is over the threshold.')
                _load_measurements(net, [rows[i] for i in active])
                se, ok, iterations = _run_estimate(net, estimator, zero_injection, max_iterations)
                if ok and estimator == 'wls':
                    residuals, J, m, n = _normalized_residuals(se)
                    rn_by_row = {active[k]: v for k, v in residuals.items()}
                break
            message = _unobservable_message(len(active), n_states)
            if failure and not any(k in failure.lower() for k in ('singular', 'measurements required')):
                message += f' ({failure})'
            return json.dumps({'error': True, 'message': message, 'input_errors': input_errors})
        if estimator != 'wls':
            break
        residuals, J, m, n = _normalized_residuals(se)
        rn_by_row = {active[k]: v for k, v in residuals.items()}
        if not bad_data or m - n <= 0 or not rn_by_row or len(removed) >= _MAX_BAD_DATA_REMOVALS:
            break
        worst_row = max(rn_by_row, key=rn_by_row.get)
        if rn_by_row[worst_row] <= rn_threshold:
            break
        removed.append({'row': worst_row, 'rn': rn_by_row[worst_row],
                        'label': _measurement_label(net, rows[worst_row])})
        active.remove(worst_row)

    dof = (m - n) if J is not None else None
    threshold = float(chi2.ppf(confidence, dof)) if dof and dof > 0 else None

    measurements = []
    removed_rows = {r['row']: r for r in removed}
    for i, row in enumerate(rows):
        est = _estimated_value(net, row)
        status = 'removed' if i in removed_rows else (
            'suspect' if rn_by_row.get(i, 0.0) > rn_threshold else 'ok')
        rn = removed_rows[i]['rn'] if i in removed_rows else rn_by_row.get(i)
        measurements.append({
            'index': i,
            'type': row['type'],
            'element_type': row['element_type'],
            'element': _display_name(net, row['element_type'], row['element']),
            'id': _cell_id(net, row['element_type'], row['element']),
            'side': row['side'],
            'source': row['source'],
            'value': _clean(row['value']),
            'std_dev': _clean(row['std_dev']),
            'true_value': _clean(row['true_value']),
            'estimated': _clean(est),
            'residual': _clean(row['value'] - est) if est is not None else None,
            'normalized_residual': _clean(rn),
            'status': status,
        })

    buses = []
    for b in net.bus.index:
        if b not in net.res_bus_est.index or not np.isfinite(_f(net.res_bus_est.at[b, 'vm_pu'], np.nan)):
            continue
        lf_vm = _f(net.res_bus.at[b, 'vm_pu'], None) if lf_ok else None
        lf_va = _f(net.res_bus.at[b, 'va_degree'], None) if lf_ok else None
        vm, va = _f(net.res_bus_est.at[b, 'vm_pu'], None), _f(net.res_bus_est.at[b, 'va_degree'], None)
        buses.append({
            'name': _display_name(net, 'bus', b), 'id': _cell_id(net, 'bus', b),
            'vn_kv': _clean(net.bus.at[b, 'vn_kv']),
            'vm_pu': _clean(vm), 'va_degree': _clean(va),
            'p_mw': _clean(net.res_bus_est.at[b, 'p_mw']), 'q_mvar': _clean(net.res_bus_est.at[b, 'q_mvar']),
            'lf_vm_pu': _clean(lf_vm), 'lf_va_degree': _clean(lf_va),
            'vm_error_pu': _clean(vm - lf_vm) if lf_vm is not None and vm is not None else None,
            'va_error_degree': _clean(va - lf_va) if lf_va is not None and va is not None else None,
        })

    branches = []
    for table, sides in _BRANCH_SIDES.items():
        res = net[f'res_{table}_est'] if f'res_{table}_est' in net else None
        if res is None:
            continue
        first = sides[0]
        for idx in net[table].index:
            if idx not in res.index or not _branch_energized(net, table, idx):
                continue
            loading = _f(res.at[idx, 'loading_percent'], None) if 'loading_percent' in res.columns else None
            lf_loading = (_f(net[f'res_{table}'].at[idx, 'loading_percent'], None)
                          if lf_ok and idx in net[f'res_{table}'].index else None)
            branches.append({
                'element_type': table,
                'name': _display_name(net, table, idx), 'id': _cell_id(net, table, idx),
                'side': first,
                'p_mw': _clean(res.at[idx, f'p_{first}_mw']), 'q_mvar': _clean(res.at[idx, f'q_{first}_mvar']),
                'loading_percent': _clean(loading),
                'lf_p_mw': _clean(net[f'res_{table}'].at[idx, f'p_{first}_mw']) if lf_ok else None,
                'lf_loading_percent': _clean(lf_loading),
            })

    by_type = {}
    for i in (set(range(len(rows))) - set(removed_rows)):
        key = rows[i]['type'] if rows[i]['element_type'] == 'bus' and rows[i]['type'] == 'v' else (
            f"{rows[i]['type']}_{'injection' if rows[i]['element_type'] == 'bus' else 'flow'}")
        by_type[key] = by_type.get(key, 0) + 1
    vm_errors = [abs(b['vm_error_pu']) for b in buses if b['vm_error_pu'] is not None]
    va_errors = [abs(b['va_error_degree']) for b in buses if b['va_error_degree'] is not None]
    suspects = [m_ for m_ in measurements if m_['status'] == 'suspect']
    summary = {
        'converged': True,
        'estimator': 'WLS' if estimator == 'wls' else 'LAV',
        'iterations': _clean(iterations),
        'measurements': len(rows) - len(removed),
        'measurements_by_type': by_type,
        'pseudo_measurements': sum(1 for r in rows if r['source'] == 'pseudo'),
        'state_variables': n,
        'redundancy': _clean((len(rows) - len(removed)) / n) if n else None,
        'objective_j': _clean(J),
        'chi2_threshold': _clean(threshold),
        'chi2_confidence': confidence,
        'chi2_passed': (J <= threshold) if (J is not None and threshold is not None) else None,
        'max_normalized_residual': _clean(max(rn_by_row.values())) if rn_by_row else None,
        'rn_threshold': rn_threshold,
        'bad_data_removed': len(removed),
        'suspect_measurements': len(suspects),
        'max_vm_error_pu': _clean(max(vm_errors)) if vm_errors else None,
        'max_va_error_degree': _clean(max(va_errors)) if va_errors else None,
        'compared_with_load_flow': lf_ok,
    }
    if estimator == 'wls' and dof is not None and dof <= 0:
        warnings_out.append('No redundancy: as many measurements as states, so bad data cannot be detected.')
    if suspects and not bad_data:
        warnings_out.append(f'{len(suspects)} measurement(s) have a normalized residual over {rn_threshold:g}.')

    csv_out = io.StringIO()
    writer = csv.writer(csv_out, lineterminator='\n')
    writer.writerow(_CSV_HEADER)
    for row in rows:
        if row['source'] == 'pseudo':
            continue
        writer.writerow([row['type'], row['element_type'], _display_name(net, row['element_type'], row['element']),
                         row['side'] or '', f"{row['value']:.6g}", f"{row['std_dev']:.4g}"])

    result = {
        'state_estimation': {
            'summary': summary,
            'buses': buses,
            'branches': branches,
            'measurements': measurements,
            'removed': [{'label': r['label'], 'normalized_residual': _clean(r['rn'])} for r in removed],
            'measurements_csv': csv_out.getvalue(),
            'warnings': warnings_out,
            'input_errors': input_errors,
            'parameters': {
                'measurement_source': source, 'estimator': summary['estimator'], 'bad_data': bad_data,
                'rn_threshold': rn_threshold, 'chi2_confidence': confidence, 'zero_injection': zero_injection,
            },
        },
    }
    return json.dumps(result)


def _measurement_label(net, row):
    what = {'v': 'V', 'p': 'P', 'q': 'Q'}[row['type']]
    where = _display_name(net, row['element_type'], row['element'])
    if row['element_type'] == 'bus':
        return f"{what} {'at' if row['type'] == 'v' else 'injection at'} {where}"
    return f"{what} flow, {where} ({row['side']})"
