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

#: pandapower's three-winding field names -> the spec's pair-named equivalents.
_TRAFO3W_PANDAPOWER_NAMES = {
    'vk_hv_percent': 'vk_hv_mv_percent', 'vk_mv_percent': 'vk_mv_lv_percent',
    'vk_lv_percent': 'vk_hv_lv_percent', 'vkr_hv_percent': 'vkr_hv_mv_percent',
    'vkr_mv_percent': 'vkr_mv_lv_percent', 'vkr_lv_percent': 'vkr_hv_lv_percent',
}
_TRAFO3W_PAIR_MEANING = {
    'vk_hv_percent': 'HV-MV', 'vk_mv_percent': 'MV-LV', 'vk_lv_percent': 'HV-LV',
    'vkr_hv_percent': 'HV-MV', 'vkr_mv_percent': 'MV-LV', 'vkr_lv_percent': 'HV-LV',
}

#: Accepted values for the spec's optional `layout` hint.
_LAYOUTS = ('transmission', 'radial', 'auto')

_ELEMENT_TABLES = (
    'buses', 'external_grids', 'transformers', 'three_winding_transformers', 'lines', 'loads',
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
           ('bus', 'ext_grid', 'trafo', 'trafo3w', 'line', 'load', 'gen', 'sgen', 'shunt',
            'storage', 'switch')}

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

    # --- three-winding transformers --------------------------------------
    # Short-circuit voltages are named by winding pair. pandapower's own names
    # mislead: its vk_mv_percent is MV-LV and its vk_lv_percent is HV-LV. Each is
    # referred to the smaller rating of its pair, as pandapower does, so the
    # two-winding rating rule supplies a default per pair.
    for i, row in enumerate(_as_list(spec, 'three_winding_transformers', problems)):
        ident = _ident(row, i, 'T3W', problems, used_ids)
        where = f'three_winding_transformers[{i}] ({ident})'
        hv = bus_of(row, 'hv_bus', where)
        mv = bus_of(row, 'mv_bus', where)
        lv = bus_of(row, 'lv_bus', where)
        # pandapower's names would otherwise be ignored silently and the defaults
        # used - and they are the names someone who knows pandapower will write.
        for theirs, ours in _TRAFO3W_PANDAPOWER_NAMES.items():
            if theirs in row:
                problems.append(f'{where}: use {ours}, not pandapower\'s {theirs} '
                                f'(which means {_TRAFO3W_PAIR_MEANING[theirs]})')
        if 'shift_lv_degree' in row:
            problems.append(f'{where}: shift_lv_degree is not supported - the Electrisim '
                            f'canvas has no place to keep it, so the drawn transformer '
                            f'would differ from the one checked')
        if hv is None or mv is None or lv is None:
            continue
        if len({hv, mv, lv}) < 3:
            problems.append(f'{where}: hv_bus, mv_bus and lv_bus must be three different '
                            f'buses, got {hv}, {mv}, {lv}')
            continue
        if not bus_kv[hv] >= bus_kv[mv] >= bus_kv[lv]:
            report.warn(
                f'{ident}: winding voltages are not in HV >= MV >= LV order '
                f'({hv} {bus_kv[hv]} kV, {mv} {bus_kv[mv]} kV, {lv} {bus_kv[lv]} kV) - '
                f'the windings look swapped'
            )
        sn_hv = _num(row.get('sn_hv_mva'), 'sn_hv_mva', where, problems, default=40.0, positive=True)
        sn_mv = _num(row.get('sn_mv_mva'), 'sn_mv_mva', where, problems, default=sn_hv, positive=True)
        sn_lv = _num(row.get('sn_lv_mva'), 'sn_lv_mva', where, problems,
                     default=round(sn_hv / 3.0, 3), positive=True)

        def pair_vk(field, a, b):
            return _num(row.get(field), field, where, problems,
                        default=_vk_percent_for(min(a, b)), positive=True)

        vk_hm = pair_vk('vk_hv_mv_percent', sn_hv, sn_mv)
        vk_ml = pair_vk('vk_mv_lv_percent', sn_mv, sn_lv)
        vk_hl = pair_vk('vk_hv_lv_percent', sn_hv, sn_lv)
        idx = pp.create_transformer3w_from_parameters(
            net, hv_bus=bus_index[hv], mv_bus=bus_index[mv], lv_bus=bus_index[lv],
            name=str(row.get('name') or ident),
            vn_hv_kv=_num(row.get('vn_hv_kv'), 'vn_hv_kv', where, problems,
                          default=bus_kv[hv], positive=True),
            vn_mv_kv=_num(row.get('vn_mv_kv'), 'vn_mv_kv', where, problems,
                          default=bus_kv[mv], positive=True),
            vn_lv_kv=_num(row.get('vn_lv_kv'), 'vn_lv_kv', where, problems,
                          default=bus_kv[lv], positive=True),
            sn_hv_mva=sn_hv, sn_mv_mva=sn_mv, sn_lv_mva=sn_lv,
            vk_hv_percent=vk_hm, vk_mv_percent=vk_ml, vk_lv_percent=vk_hl,
            vkr_hv_percent=_num(row.get('vkr_hv_mv_percent'), 'vkr_hv_mv_percent', where,
                                problems, default=round(vk_hm / 25.0, 3)),
            vkr_mv_percent=_num(row.get('vkr_mv_lv_percent'), 'vkr_mv_lv_percent', where,
                                problems, default=round(vk_ml / 25.0, 3)),
            vkr_lv_percent=_num(row.get('vkr_hv_lv_percent'), 'vkr_hv_lv_percent', where,
                                problems, default=round(vk_hl / 25.0, 3)),
            pfe_kw=_num(row.get('pfe_kw'), 'pfe_kw', where, problems, default=sn_hv * 0.6),
            i0_percent=_num(row.get('i0_percent'), 'i0_percent', where, problems, default=0.1),
            shift_mv_degree=_num(row.get('shift_mv_degree'), 'shift_mv_degree', where,
                                 problems, default=0.0),
            in_service=bool(row.get('in_service', True)),
        )
        record('trafo3w', idx, ident)

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
        cos_phi = _num(row.get('cos_phi'), 'cos_phi', where, problems, default=0.85, positive=True)
        if cos_phi is not None and cos_phi > 1:
            problems.append(f'{where}: cos_phi={cos_phi} must be at most 1')
        # Short-circuit data: a short-circuit study fails on a generator without
        # it, so typical values stand in until real ones are given.
        idx = pp.create_gen(
            net, bus=bus_index[bus], name=str(row.get('name') or ident), p_mw=p_mw,
            vm_pu=_num(row.get('vm_pu'), 'vm_pu', where, problems, default=1.0, positive=True),
            sn_mva=_num(row.get('sn_mva'), 'sn_mva', where, problems,
                        default=max(p_mw * 1.2, 1.0), positive=True),
            vn_kv=_num(row.get('vn_kv'), 'vn_kv', where, problems,
                       default=bus_kv[bus], positive=True),
            xdss_pu=_num(row.get('xdss_pu'), 'xdss_pu', where, problems, default=0.2, positive=True),
            rdss_ohm=_num(row.get('rdss_ohm'), 'rdss_ohm', where, problems, default=0.0),
            cos_phi=cos_phi,
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
        p_mw = _num(row.get('p_mw'), 'p_mw', where, problems, default=0.0)
        q_mvar = _num(row.get('q_mvar'), 'q_mvar', where, problems, default=0.0)
        # A short-circuit study needs a rating and the short-circuit to rated
        # current ratio k; 1.1 is the value Electrisim has always assumed.
        idx = pp.create_sgen(
            net, bus=bus_index[bus], name=name, p_mw=p_mw, q_mvar=q_mvar,
            sn_mva=_num(row.get('sn_mva'), 'sn_mva', where, problems,
                        default=round(max(abs(p_mw or 0) * 1.1, abs(q_mvar or 0), 0.1), 6),
                        positive=True),
            k=_num(row.get('k'), 'k', where, problems, default=1.1, positive=True),
            # pandapower assumes this when it is unset, but the canvas defaults a
            # static generator to "async", which needs locked-rotor data instead.
            generator_type='current_source',
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
        soc = _num(row.get('soc_percent'), 'soc_percent', where, problems, default=50.0)
        if soc is not None and not 0 <= soc <= 100:
            problems.append(f'{where}: soc_percent={soc} must be between 0 and 100')
        # pandapower's load flow ignores the state of charge; OpenDSS does not,
        # and holds an empty battery idle whatever p_mw asks for.
        idx = pp.create_storage(
            net, bus=bus_index[bus], name=str(row.get('name') or ident),
            p_mw=_num(row.get('p_mw'), 'p_mw', where, problems, default=0.0),
            max_e_mwh=_num(row.get('max_e_mwh'), 'max_e_mwh', where, problems,
                           default=1.0, positive=True),
            soc_percent=soc,
            in_service=bool(row.get('in_service', True)),
        )
        record('storage', idx, ident)

    # --- switches --------------------------------------------------------
    # The element is named by its spec id, never by its display name - a line
    # with id "L1" and name "Feeder cable" is switched as "L1".
    # pandapower switch type -> (table, spec word, terminal columns)
    switched = {
        'l': ('line', 'line', ('from_bus', 'to_bus')),
        't': ('trafo', 'transformer', ('hv_bus', 'lv_bus')),
        't3': ('trafo3w', 'three-winding transformer', ('hv_bus', 'mv_bus', 'lv_bus')),
    }
    by_id = {table: {ident: idx for idx, ident in ids[table].items()}
             for table, _, _ in switched.values()}
    for i, row in enumerate(_as_list(spec, 'switches', problems)):
        ident = _ident(row, i, 'Sw', problems, used_ids)
        where = f'switches[{i}] ({ident})'
        bus = bus_of(row, 'bus', where)
        et_raw = str(row.get('et') or row.get('element_type') or 'line').lower()
        et = {'line': 'l', 'l': 'l', 'trafo': 't', 'transformer': 't', 't': 't',
              'three_winding_transformer': 't3', 'trafo3w': 't3', 't3': 't3',
              'bus': 'b', 'b': 'b'}.get(et_raw)
        if et is None:
            problems.append(f'{where}: et={et_raw!r} must be one of line, transformer, '
                            f'three_winding_transformer, bus')
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
            table, word, ends = switched[et]
            if element not in by_id[table]:
                known = ', '.join(sorted(by_id[table])) or '(none defined)'
                problems.append(
                    f'{where}: element={element!r} is not a {word} id. Defined: {known}')
                continue
            target = by_id[table][element]
            # pandapower accepts a switch at any bus; it only means something at
            # one of the element's own terminals.
            df = net[table]
            terminals = {int(df.at[target, c]) for c in ends}
            if int(bus_index[bus]) not in terminals:
                names = ', '.join(ids['bus'][t] for t in sorted(terminals))
                problems.append(
                    f'{where}: bus {bus!r} is not a terminal of {word} {element!r}; '
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
    for table, cols in (('line', ('from_bus', 'to_bus')), ('trafo', ('hv_bus', 'lv_bus')),
                        ('trafo3w', ('hv_bus', 'mv_bus', 'lv_bus'))):
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
        'trafo3w': len(net.trafo3w),
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

    trafos3w = [{
        'id': ident('trafo3w', idx),
        'loading_percent': _r(row['loading_percent'], 1),
        'p_hv_mw': _r(row['p_hv_mw'], 4),
        'p_mv_mw': _r(row['p_mv_mw'], 4),
        'p_lv_mw': _r(row['p_lv_mw'], 4),
        'losses_mw': _r(row['pl_mw'], 5),
    } for idx, row in net.res_trafo3w.iterrows()]

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
        for kind, rows in (('line', lines), ('transformer', trafos),
                           ('three-winding transformer', trafos3w))
        for e in rows
        if e['loading_percent'] is not None and e['loading_percent'] > max_loading_percent
    ]

    load_mw = float(net.res_load['p_mw'].sum()) if len(net.res_load) else 0.0
    losses_mw = float(net.res_line['pl_mw'].sum() + net.res_trafo['pl_mw'].sum()
                      + net.res_trafo3w['pl_mw'].sum())
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
        'three_winding_transformers': trafos3w,
        'external_grids': grids,
    }
