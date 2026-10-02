# -*- coding: utf-8 -*-
"""
Build a pandapower network from a declarative single-line-diagram spec.

The spec is plain JSON, written to be produced by a language model from a
description in prose, so it differs from pandapower's own API in three ways that
matter:

  * Elements are referred to by string id ("B1", "T1"), never by the integer
    position in a DataFrame. Nothing has to track index arithmetic, and ids
    survive reordering.
  * Electrical parameters have defaults chosen per voltage level, so "a 110/20 kV
    25 MVA transformer" is a complete statement and vk_percent need not be
    invented.
  * Validation collects every problem and reports them together, each naming the
    element and the fix. One malformed field should not hide the other six.

This module does not execute anything from the request. /import-pandapower takes
a Python script and exec()s it; this is the structured alternative, and the two
should not be confused.

Entry points:

    net, report = build_network(spec)
    results = solve(net)

`report` carries warnings worth showing even on success (an isolated bus, a
missing slack, transformer windings that look swapped). `solve` runs a balanced
AC power flow and reports it under the spec's own ids. The spec format is
documented for callers in electrisim_mcp/spec_format.md.
"""

import math

import pandapower as pp


# --- defaults ------------------------------------------------------------
#
# Chosen so a spec that names only the obvious quantities still solves. Values
# are typical for the voltage class rather than any specific product.

#: Short-circuit voltage (%) by transformer rating, used when vk_percent is absent.
_VK_PERCENT_BY_SN = ((2.5, 4.0), (10.0, 6.0), (40.0, 12.0), (200.0, 14.0))

#: Default line type by nominal voltage (kV), used when neither std_type nor
#: explicit impedances are given.
_LINE_STD_TYPE_BY_KV = (
    (1.0, 'NAYY 4x150 SE'),
    (30.0, 'NA2XS2Y 1x240 RM/25 12/20 kV'),
    (150.0, '149-AL1/24-ST1A 110.0'),
    (400.0, '490-AL1/64-ST1A 380.0'),
)

#: Accepted values for the spec's optional `layout` hint.
_LAYOUTS = ('transmission', 'radial', 'auto')

_ELEMENT_TABLES = (
    'buses', 'external_grids', 'transformers', 'lines', 'loads',
    'generators', 'static_generators', 'shunts', 'storage', 'switches',
)


class SpecError(ValueError):
    """Raised when a spec cannot be built. Carries every problem found, not the first."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__('; '.join(self.problems))


class _Report:
    def __init__(self):
        self.warnings = []
        self.counts = {}
        self.layout = None

    def warn(self, message):
        if message not in self.warnings:
            self.warnings.append(message)

    def as_dict(self):
        return {'warnings': list(self.warnings), 'counts': dict(self.counts),
                'layout': self.layout}


def _looks_like_wind(name):
    """Matches the test the radial layout itself applies to pick turbine symbols."""
    lowered = str(name).lower()
    return 'wind' in lowered or 'turbine' in lowered


def _vk_percent_for(sn_mva):
    for limit, vk in _VK_PERCENT_BY_SN:
        if sn_mva <= limit:
            return vk
    return _VK_PERCENT_BY_SN[-1][1]


def _line_std_type_for(vn_kv):
    for limit, std in _LINE_STD_TYPE_BY_KV:
        if vn_kv <= limit:
            return std
    return _LINE_STD_TYPE_BY_KV[-1][1]


def _num(value, field, where, problems, default=None, positive=False):
    """Read a numeric field, recording a problem rather than raising."""
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        problems.append(f'{where}: {field} must be a number, got {value!r}')
        return default
    if not math.isfinite(out):
        problems.append(f'{where}: {field} must be finite, got {value!r}')
        return default
    if positive and out <= 0:
        problems.append(f'{where}: {field} must be greater than 0, got {out}')
        return default
    return out


def _as_list(spec, key, problems):
    value = spec.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        problems.append(f'{key} must be a list, got {type(value).__name__}')
        return []
    for i, row in enumerate(value):
        if not isinstance(row, dict):
            problems.append(f'{key}[{i}] must be an object, got {type(row).__name__}')
            return []
    return value


def _ident(row, index, kind, problems, used):
    """Element id: explicit, or derived from the name, or positional."""
    raw = row.get('id') or row.get('name')
    ident = str(raw).strip() if raw not in (None, '') else f'{kind}{index + 1}'
    if ident in used:
        problems.append(
            f'duplicate id {ident!r}: ids must be unique across all elements, '
            f'because lines, switches and transformers refer to them by name'
        )
    used.add(ident)
    return ident


def build_network(spec):
    """
    Build a pandapower net from a spec dict.

    Returns (net, report_dict). Raises SpecError listing every problem when the
    spec cannot be built.
    """
    if not isinstance(spec, dict):
        raise SpecError([f'spec must be an object, got {type(spec).__name__}'])

    problems = []
    report = _Report()

    unknown = set(spec) - set(_ELEMENT_TABLES) - {'name', 'frequency_hz', 'layout'}
    for key in sorted(unknown):
        problems.append(
            f'unknown top-level key {key!r}; expected one of: '
            + ', '.join(sorted(_ELEMENT_TABLES + ('name', 'frequency_hz', 'layout')))
        )

    f_hz = _num(spec.get('frequency_hz'), 'frequency_hz', 'spec', problems,
                default=50.0, positive=True)
    net = pp.create_empty_network(name=str(spec.get('name') or 'Electrisim network'),
                                  f_hz=f_hz)

    layout = spec.get('layout')
    if layout is not None and str(layout).lower() not in _LAYOUTS:
        problems.append(f'layout={layout!r} must be one of: ' + ', '.join(_LAYOUTS))

    friendly = {}
    used_ids = set()

    # pandapower row index -> spec id, per table. Switches resolve their element
    # through it, and results are reported back under the ids the caller wrote.
    ids = {table: {} for table in
           ('bus', 'ext_grid', 'trafo', 'line', 'load', 'gen', 'sgen', 'shunt', 'storage', 'switch')}

    def record(table, idx, ident):
        ids[table][int(idx)] = ident

    # --- buses -----------------------------------------------------------
    bus_rows = _as_list(spec, 'buses', problems)
    bus_index = {}
    bus_kv = {}
    for i, row in enumerate(bus_rows):
        ident = _ident(row, i, 'B', problems, used_ids)
        where = f'buses[{i}] ({ident})'
        vn_kv = _num(row.get('vn_kv'), 'vn_kv', where, problems, positive=True)
        if vn_kv is None:
            problems.append(f'{where}: vn_kv is required - a bus has no default voltage')
            continue
        name = str(row.get('name') or ident)
        idx = pp.create_bus(net, vn_kv=vn_kv, name=name,
                            in_service=bool(row.get('in_service', True)))
        record('bus', idx, ident)
        bus_index[ident] = idx
        bus_kv[ident] = vn_kv

    if not bus_rows:
        problems.append('buses is empty: a diagram needs at least one bus')

    def bus_of(row, field, where, required=True):
        raw = row.get(field)
        if raw in (None, ''):
            if required:
                problems.append(f'{where}: {field} is required')
            return None
        key = str(raw).strip()
        if key not in bus_index:
            known = ', '.join(sorted(bus_index)) or '(none defined)'
            problems.append(f'{where}: {field}={key!r} is not a bus id. Defined buses: {known}')
            return None
        return key

    # --- external grids --------------------------------------------------
    for i, row in enumerate(_as_list(spec, 'external_grids', problems)):
        ident = _ident(row, i, 'Grid', problems, used_ids)
        where = f'external_grids[{i}] ({ident})'
        bus = bus_of(row, 'bus', where)
        if bus is None:
            continue
        idx = pp.create_ext_grid(
            net, bus=bus_index[bus], name=str(row.get('name') or ident),
            vm_pu=_num(row.get('vm_pu'), 'vm_pu', where, problems, default=1.0, positive=True),
            va_degree=_num(row.get('va_degree'), 'va_degree', where, problems, default=0.0),
            s_sc_max_mva=_num(row.get('s_sc_max_mva'), 's_sc_max_mva', where, problems, default=None),
            rx_max=_num(row.get('rx_max'), 'rx_max', where, problems, default=None),
        )
        record('ext_grid', idx, ident)

    # --- transformers ----------------------------------------------------
    for i, row in enumerate(_as_list(spec, 'transformers', problems)):
        ident = _ident(row, i, 'T', problems, used_ids)
        where = f'transformers[{i}] ({ident})'
        hv = bus_of(row, 'hv_bus', where)
        lv = bus_of(row, 'lv_bus', where)
        if hv is None or lv is None:
            continue
        if bus_kv[hv] < bus_kv[lv]:
            report.warn(
                f'{ident}: hv_bus {hv} ({bus_kv[hv]} kV) is lower voltage than '
                f'lv_bus {lv} ({bus_kv[lv]} kV) - the windings look swapped'
            )
        sn_mva = _num(row.get('sn_mva'), 'sn_mva', where, problems, default=25.0, positive=True)
        vk = _num(row.get('vk_percent'), 'vk_percent', where, problems,
                  default=_vk_percent_for(sn_mva), positive=True)
        idx = pp.create_transformer_from_parameters(
            net, hv_bus=bus_index[hv], lv_bus=bus_index[lv],
            name=str(row.get('name') or ident),
            sn_mva=sn_mva,
            vn_hv_kv=_num(row.get('vn_hv_kv'), 'vn_hv_kv', where, problems,
                          default=bus_kv[hv], positive=True),
            vn_lv_kv=_num(row.get('vn_lv_kv'), 'vn_lv_kv', where, problems,
                          default=bus_kv[lv], positive=True),
            vk_percent=vk,
            vkr_percent=_num(row.get('vkr_percent'), 'vkr_percent', where, problems,
                             default=round(vk / 25.0, 3)),
            pfe_kw=_num(row.get('pfe_kw'), 'pfe_kw', where, problems, default=sn_mva * 0.6),
            i0_percent=_num(row.get('i0_percent'), 'i0_percent', where, problems, default=0.1),
            shift_degree=_num(row.get('shift_degree'), 'shift_degree', where, problems, default=0.0),
            in_service=bool(row.get('in_service', True)),
        )
        record('trafo', idx, ident)

    # --- lines -----------------------------------------------------------
    for i, row in enumerate(_as_list(spec, 'lines', problems)):
        ident = _ident(row, i, 'L', problems, used_ids)
        where = f'lines[{i}] ({ident})'
        a = bus_of(row, 'from_bus', where)
        b = bus_of(row, 'to_bus', where)
        if a is None or b is None:
            continue
        if abs(bus_kv[a] - bus_kv[b]) > 1e-6:
            problems.append(
                f'{where}: from_bus {a} is {bus_kv[a]} kV but to_bus {b} is {bus_kv[b]} kV. '
                f'A line joins buses at one voltage - use a transformer instead'
            )
            continue
        length_km = _num(row.get('length_km'), 'length_km', where, problems,
                         default=1.0, positive=True)
        name = str(row.get('name') or ident)
        if row.get('r_ohm_per_km') is not None or row.get('x_ohm_per_km') is not None:
            idx = pp.create_line_from_parameters(
                net, from_bus=bus_index[a], to_bus=bus_index[b], name=name,
                length_km=length_km,
                r_ohm_per_km=_num(row.get('r_ohm_per_km'), 'r_ohm_per_km', where, problems, default=0.1),
                x_ohm_per_km=_num(row.get('x_ohm_per_km'), 'x_ohm_per_km', where, problems, default=0.1),
                c_nf_per_km=_num(row.get('c_nf_per_km'), 'c_nf_per_km', where, problems, default=0.0),
                max_i_ka=_num(row.get('max_i_ka'), 'max_i_ka', where, problems, default=0.4, positive=True),
                in_service=bool(row.get('in_service', True)),
            )
        else:
            std_type = str(row.get('std_type') or _line_std_type_for(bus_kv[a]))
            try:
                idx = pp.create_line(net, from_bus=bus_index[a], to_bus=bus_index[b], name=name,
                                     length_km=length_km, std_type=std_type,
                                     in_service=bool(row.get('in_service', True)))
            except (KeyError, UserWarning) as exc:
                problems.append(
                    f'{where}: std_type {std_type!r} is not in the pandapower library ({exc}). '
                    f'Give r_ohm_per_km and x_ohm_per_km instead, or omit std_type for the default'
                )
                continue
        record('line', idx, ident)

    # --- loads, machines, compensation -----------------------------------
    for i, row in enumerate(_as_list(spec, 'loads', problems)):
        ident = _ident(row, i, 'Load', problems, used_ids)
        where = f'loads[{i}] ({ident})'
        bus = bus_of(row, 'bus', where)
        if bus is None:
            continue
        p_mw = _num(row.get('p_mw'), 'p_mw', where, problems, default=0.0)
        idx = pp.create_load(
            net, bus=bus_index[bus], name=str(row.get('name') or ident), p_mw=p_mw,
            q_mvar=_num(row.get('q_mvar'), 'q_mvar', where, problems,
                        default=round(p_mw * 0.33, 6)),
            in_service=bool(row.get('in_service', True)),
        )
        record('load', idx, ident)

    for i, row in enumerate(_as_list(spec, 'generators', problems)):
        ident = _ident(row, i, 'Gen', problems, used_ids)
        where = f'generators[{i}] ({ident})'
        bus = bus_of(row, 'bus', where)
        if bus is None:
            continue
        p_mw = _num(row.get('p_mw'), 'p_mw', where, problems, default=0.0)
        idx = pp.create_gen(
            net, bus=bus_index[bus], name=str(row.get('name') or ident), p_mw=p_mw,
            vm_pu=_num(row.get('vm_pu'), 'vm_pu', where, problems, default=1.0, positive=True),
            sn_mva=_num(row.get('sn_mva'), 'sn_mva', where, problems,
                        default=max(p_mw * 1.2, 1.0), positive=True),
            slack=bool(row.get('slack', False)),
            in_service=bool(row.get('in_service', True)),
        )
        record('gen', idx, ident)

    for i, row in enumerate(_as_list(spec, 'static_generators', problems)):
        ident = _ident(row, i, 'SGen', problems, used_ids)
        where = f'static_generators[{i}] ({ident})'
        bus = bus_of(row, 'bus', where)
        if bus is None:
            continue
        name = str(row.get('name') or ident)
        idx = pp.create_sgen(
            net, bus=bus_index[bus], name=name,
            p_mw=_num(row.get('p_mw'), 'p_mw', where, problems, default=0.0),
            q_mvar=_num(row.get('q_mvar'), 'q_mvar', where, problems, default=0.0),
            type=str(row.get('type') or 'wye'),
            in_service=bool(row.get('in_service', True)),
        )
        record('sgen', idx, ident)
        # A radial import draws turbine symbols for a machine whose name says
        # "wind" or "turbine". Say so rather than editing the label behind the
        # user's back - the name is what the diagram shows.
        if row.get('kind') in ('wind', 'turbine') and not _looks_like_wind(name):
            report.warn(
                f'{ident}: kind={row["kind"]!r} but the name {name!r} does not mention '
                f'wind or turbine, so a radial import draws the generic static-generator '
                f'symbol. Rename it (e.g. "Wind {name}") to get turbine symbols'
            )

    for i, row in enumerate(_as_list(spec, 'shunts', problems)):
        ident = _ident(row, i, 'Shunt', problems, used_ids)
        where = f'shunts[{i}] ({ident})'
        bus = bus_of(row, 'bus', where)
        if bus is None:
            continue
        q_mvar = _num(row.get('q_mvar'), 'q_mvar', where, problems)
        if q_mvar is None:
            problems.append(f'{where}: q_mvar is required (negative for a capacitor bank)')
            continue
        idx = pp.create_shunt(net, bus=bus_index[bus], name=str(row.get('name') or ident),
                              q_mvar=q_mvar,
                              p_mw=_num(row.get('p_mw'), 'p_mw', where, problems, default=0.0),
                              in_service=bool(row.get('in_service', True)))
        record('shunt', idx, ident)

    for i, row in enumerate(_as_list(spec, 'storage', problems)):
        ident = _ident(row, i, 'Storage', problems, used_ids)
        where = f'storage[{i}] ({ident})'
        bus = bus_of(row, 'bus', where)
        if bus is None:
            continue
        idx = pp.create_storage(
            net, bus=bus_index[bus], name=str(row.get('name') or ident),
            p_mw=_num(row.get('p_mw'), 'p_mw', where, problems, default=0.0),
            max_e_mwh=_num(row.get('max_e_mwh'), 'max_e_mwh', where, problems,
                           default=1.0, positive=True),
            in_service=bool(row.get('in_service', True)),
        )
        record('storage', idx, ident)

    # --- switches --------------------------------------------------------
    # The element is named by its spec id, never by its display name - a line
    # with id "L1" and name "Feeder cable" is switched as "L1".
    by_id = {table: {ident: idx for idx, ident in ids[table].items()}
             for table in ('line', 'trafo')}
    for i, row in enumerate(_as_list(spec, 'switches', problems)):
        ident = _ident(row, i, 'Sw', problems, used_ids)
        where = f'switches[{i}] ({ident})'
        bus = bus_of(row, 'bus', where)
        et_raw = str(row.get('et') or row.get('element_type') or 'line').lower()
        et = {'line': 'l', 'l': 'l', 'trafo': 't', 'transformer': 't', 't': 't',
              'bus': 'b', 'b': 'b'}.get(et_raw)
        if et is None:
            problems.append(f'{where}: et={et_raw!r} must be one of line, transformer, bus')
            continue
        element = str(row.get('element') or '').strip()
        if bus is None:
            continue
        if et == 'b':
            if element not in bus_index:
                problems.append(f'{where}: element={element!r} must be a bus id when et=bus')
                continue
            if element == bus:
                problems.append(f'{where}: a bus coupler joins two different buses, '
                                f'but bus and element are both {bus!r}')
                continue
            target = bus_index[element]
        else:
            table = 'line' if et == 'l' else 'trafo'
            if element not in by_id[table]:
                known = ', '.join(sorted(by_id[table])) or '(none defined)'
                problems.append(
                    f'{where}: element={element!r} is not a {table} id. Defined: {known}')
                continue
            target = by_id[table][element]
            # pandapower accepts a switch at any bus; it only means something at
            # one of the element's own terminals.
            ends = ('from_bus', 'to_bus') if et == 'l' else ('hv_bus', 'lv_bus')
            df = net.line if et == 'l' else net.trafo
            terminals = {int(df.at[target, c]) for c in ends}
            if int(bus_index[bus]) not in terminals:
                names = ', '.join(ids['bus'][t] for t in sorted(terminals))
                problems.append(
                    f'{where}: bus {bus!r} is not a terminal of {table} {element!r}; '
                    f'a switch on it must sit at one of: {names}')
                continue
        idx = pp.create_switch(net, bus=bus_index[bus], element=target, et=et,
                               closed=bool(row.get('closed', True)),
                               name=str(row.get('name') or ident))
        record('switch', idx, ident)

    if problems:
        raise SpecError(problems)

    # --- post-build checks worth surfacing -------------------------------
    if net.ext_grid.empty and not (not net.gen.empty and net.gen['slack'].any()):
        report.warn(
            'no external grid and no slack generator: a power flow will not converge. '
            'Add an external_grids entry, or set slack=true on a generator'
        )

    connected = set()
    for table, cols in (('line', ('from_bus', 'to_bus')), ('trafo', ('hv_bus', 'lv_bus'))):
        df = getattr(net, table)
        for col in cols:
            connected.update(int(v) for v in df[col].tolist())
    # A bus coupler connects two bars as surely as a line does.
    couplers = net.switch[net.switch['et'] == 'b']
    connected.update(int(v) for v in couplers['bus'].tolist())
    connected.update(int(v) for v in couplers['element'].tolist())
    islanded = [ids['bus'][int(i)] for i in net.bus.index
                if int(i) not in connected and len(net.bus.index) > 1]
    if islanded:
        report.warn('buses with no line, transformer or coupler: ' + ', '.join(islanded))

    net.user_friendly_names = friendly
    # Read back by solve(); pandapower ignores keys it does not know.
    net['electrisim_ids'] = ids
    report.layout = str(layout).lower() if layout is not None else None
    report.counts = {
        'bus': len(net.bus), 'line': len(net.line), 'trafo': len(net.trafo),
        'load': len(net.load), 'gen': len(net.gen), 'sgen': len(net.sgen),
        'ext_grid': len(net.ext_grid), 'shunt': len(net.shunt),
        'storage': len(net.storage), 'switch': len(net.switch),
    }
    return net, report.as_dict()


def _r(value, digits):
    """Round a result cell, mapping NaN (an out-of-service element) to None."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return round(f, digits) if math.isfinite(f) else None


def solve(net, vm_min_pu=0.95, vm_max_pu=1.05, max_loading_percent=100.0):
    """
    Run a balanced AC power flow on a net from build_network() and report it
    under the spec's own ids.

    This is pandapower's runpp at its defaults - Newton-Raphson, the same engine
    behind Electrisim's load flow - so the numbers match what the app shows for
    a default load-flow run. The limits only decide what is flagged; nothing is
    enforced.
    """
    ids = net.get('electrisim_ids') or {}

    def ident(table, idx):
        return (ids.get(table) or {}).get(int(idx), f'{table}{int(idx)}')

    try:
        pp.runpp(net)
    except UserWarning as exc:
        # pandapower raises (not warns) when it cannot even start - most often
        # "No reference bus is available". Its own message is the best hint.
        return {'converged': False, 'hint': str(exc)}
    except pp.LoadflowNotConverged:
        return {
            'converged': False,
            'hint': ('The power flow did not converge. Usual causes: no external grid or '
                     'slack generator, an island with load but no source, a transformer '
                     'whose rated voltages do not match its buses, or load far beyond what '
                     'the lines and transformers can carry.'),
        }

    buses = []
    for idx, row in net.res_bus.iterrows():
        buses.append({
            'id': ident('bus', idx),
            'vn_kv': _r(net.bus.at[idx, 'vn_kv'], 3),
            'vm_pu': _r(row['vm_pu'], 4),
            'va_degree': _r(row['va_degree'], 3),
        })

    lines = [{
        'id': ident('line', idx),
        'loading_percent': _r(row['loading_percent'], 1),
        'p_from_mw': _r(row['p_from_mw'], 4),
        'q_from_mvar': _r(row['q_from_mvar'], 4),
        'i_ka': _r(row['i_ka'], 4),
        'losses_mw': _r(row['pl_mw'], 5),
    } for idx, row in net.res_line.iterrows()]

    trafos = [{
        'id': ident('trafo', idx),
        'loading_percent': _r(row['loading_percent'], 1),
        'p_hv_mw': _r(row['p_hv_mw'], 4),
        'q_hv_mvar': _r(row['q_hv_mvar'], 4),
        'losses_mw': _r(row['pl_mw'], 5),
    } for idx, row in net.res_trafo.iterrows()]

    grids = [{
        'id': ident('ext_grid', idx),
        'p_mw': _r(row['p_mw'], 4),
        'q_mvar': _r(row['q_mvar'], 4),
    } for idx, row in net.res_ext_grid.iterrows()]

    voltage_issues = [
        {'id': b['id'], 'vm_pu': b['vm_pu'],
         'issue': 'undervoltage' if b['vm_pu'] < vm_min_pu else 'overvoltage'}
        for b in buses
        if b['vm_pu'] is not None and not (vm_min_pu <= b['vm_pu'] <= vm_max_pu)
    ]
    overloads = [
        {'id': e['id'], 'kind': kind, 'loading_percent': e['loading_percent']}
        for kind, rows in (('line', lines), ('transformer', trafos))
        for e in rows
        if e['loading_percent'] is not None and e['loading_percent'] > max_loading_percent
    ]

    load_mw = float(net.res_load['p_mw'].sum()) if len(net.res_load) else 0.0
    losses_mw = float(net.res_line['pl_mw'].sum() + net.res_trafo['pl_mw'].sum())
    vms = [b['vm_pu'] for b in buses if b['vm_pu'] is not None]

    return {
        'converged': True,
        'summary': {
            'buses': len(buses),
            'vm_min_pu': min(vms) if vms else None,
            'vm_max_pu': max(vms) if vms else None,
            'load_mw': round(load_mw, 4),
            'losses_mw': round(losses_mw, 5),
            'limits': {'vm_min_pu': vm_min_pu, 'vm_max_pu': vm_max_pu,
                       'max_loading_percent': max_loading_percent},
            'voltage_issues': voltage_issues,
            'overloads': overloads,
        },
        'buses': buses,
        'lines': lines,
        'transformers': trafos,
        'external_grids': grids,
    }
