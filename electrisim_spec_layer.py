# -*- coding: utf-8 -*-
"""
The spec's DC and microgrid layer: DC buses, lines, loads, sources and
capacitors; VSCs, solid-state transformers, DC/DC converters and DC breakers;
the batteries, supercapacitors, flywheels, SOFC systems and PV arrays; the PCS;
zigzag grounding transformers; and the diagram's load-profile library.

These are Electrisim's own elements, not pandapower tables, so electrisim_sld
does not build them into the pandapower net it hands the canvas. They are
checked here, turned into the rows the diagram sends for them (the payload
dcPayload.js builds, each named by its spec id), and:

  * built by pandapower_electrisim's own builders - the code every study
    runs - onto a copy of the AC net, for solve();
  * handed to the canvas as `electrisim_elements`, which it draws.

Each element's fields are its dialog's, by the same names; a field the dialog
does not have is refused. tests/test_spec_layer.py keeps these lists in step
with utils/derParameters.js and utils/dcPayload.js.
"""

import copy
import math

# Each kind's fields, as its dialog names them (derParameters.js, dcPayload.js).
DER_FIELDS = {
    'Battery': ('sizing', 'vn_v', 'capacity_kwh', 'r0_mohm', 'cells_series', 'strings_parallel', 'cell_v', 'cell_ah',
                'cell_r_mohm', 'r1_percent', 'tau1_s', 'soc_percent', 'soc_min_percent', 'soc_max_percent',
                'c_rate_discharge', 'c_rate_charge', 'coulombic_efficiency_percent', 'ocv_table', 'l_uh'),
    'Supercapacitor': ('coupling', 'sizing', 'c_f', 'v_rated', 'esr_mohm', 'esl_uh', 'module_c_f', 'module_v',
                       'module_esr_mohm', 'module_esl_uh', 'modules_series', 'strings_parallel', 'v0_percent',
                       'v_min_percent', 'p_rated_kw', 'r_leak_ohm'),
    'Flywheel': ('v_dc', 'p_rated_kw', 'e_max_kwh', 'speed_percent', 'speed_min_percent', 'speed_base_percent',
                 'efficiency_percent', 'standby_loss_percent_h', 'r_dc_mohm', 'p_set_kw'),
    'SOFC': ('p_rated_kw', 'v_rated', 'p_set_kw', 'fuel_utilisation_percent', 'min_load_percent', 'aux_load_percent',
             'ramp_percent_s'),
    'PV Array': ('module_pmpp_w', 'module_vmpp', 'module_impp', 'module_voc', 'module_isc', 'module_cells_series',
                 'alpha_isc_percent_k', 'beta_voc_percent_k', 'noct_c', 'modules_series', 'strings_parallel',
                 'loss_percent', 'irradiance_wm2', 'ambient_c', 'irradiance_profile_id', 'temperature_profile_id'),
    'PCS': ('control', 's_rated_mva', 'vn_ac_kv', 'efficiency_percent', 'no_load_loss_kw', 'p_set_mw', 'q_mode',
            'q_set_mvar', 'pf', 'qv_droop_percent', 'vm_set_pu', 'droop_pf_percent', 'droop_qv_percent',
            'current_limit_pu', 'opf_marginal_cost_eur_per_mwh'),
    'Grounding Transformer': ('vn_kv', 'i_rated_a', 't_rated_s', 'r_n_ohm', 'x_n_ohm', 'x0_ohm', 'r0_ohm'),
}
DC_FIELDS = {
    'DC Bus': ('vn_kv',),
    'Load DC': ('p_mw', 'load_model', 'share_p_percent', 'share_i_percent', 'share_r_percent', 'v_min_pu',
                'filter_l_mh', 'filter_c_uf', 'load_profile_id'),
    'Source DC': ('vm_pu', 'r_sc_mohm', 'l_sc_uh'),
    'DC Capacitor': ('c_mf', 'esr_mohm', 'esl_uh'),
    'DC Line': ('length_km', 'r_ohm_per_km', 'max_i_ka', 'l_mh_per_km', 'c_uf_per_km'),
    'DC Breaker': ('closed', 'breaker_type', 'rated_voltage_kv', 'rated_current_ka', 'breaking_capacity_ka',
                   'trip_current_ka', 'opening_time_ms', 'limiting_inductance_mh', 'arrester_clamp_kv',
                   'arrester_energy_kj'),
    'Solid-State Transformer': ('vn_mv_kv', 'vn_lv_dc_kv', 'vn_lv_ac_kv', 'link_kv', 'q_mv_mvar', 'rect_rated_mw',
                                'rect_efficiency_percent', 'rect_no_load_kw', 'dcdc_rated_mw',
                                'dcdc_efficiency_percent', 'dcdc_no_load_kw', 'vm_lv_dc_pu', 'inverter_mode',
                                'inv_rated_mw', 'inv_efficiency_percent', 'inv_no_load_kw', 'p_ac_mw', 'q_ac_mvar',
                                'vm_lv_ac_pu', 'emt_model', 'switching_khz', 'dcdc_switching_khz', 'current_limit_pu'),
    'DC/DC Converter': ('control_mode', 'vm_out_pu', 'p_set_mw', 'rated_mw', 'vn_in_kv', 'vn_out_kv',
                        'efficiency_percent', 'no_load_loss_kw', 'bidirectional', 'droop_percent', 'smoothing_tau_s',
                        'soc_ref_percent', 'soc_gain', 'emt_model', 'switching_khz', 'current_limit_pu', 'c_out_mf',
                        'c_out_esr_mohm', 'c_out_esl_uh', 'c_in_esr_mohm', 'c_in_esl_uh'),
    'VSC': ('r_ohm', 'x_ohm', 'r_dc_ohm', 'control_mode_ac', 'control_value_ac', 'control_mode_dc',
            'control_value_dc', 'rated_mva', 'dc_link_mf', 'dc_link_esr_mohm', 'dc_link_esl_uh', 'current_limit_pu',
            'emt_model', 'switching_khz'),
}
FIELDS = {**DER_FIELDS, **DC_FIELDS}
# Each dialog's defaults (configureAttributes.js, derParameters.js): a field the spec leaves out
# takes its dialog's default, as a drawn element does on its drop.
DEFAULTS = {'Battery': {'vn_v': '800',
             'capacity_kwh': '224',
             'r0_mohm': '62.5',
             'cells_series': '250',
             'strings_parallel': '1',
             'cell_v': '3.2',
             'cell_ah': '280',
             'cell_r_mohm': '0.25',
             'r1_percent': '40',
             'tau1_s': '30',
             'soc_percent': '50',
             'soc_min_percent': '10',
             'soc_max_percent': '90',
             'c_rate_discharge': '1',
             'c_rate_charge': '0.5',
             'coulombic_efficiency_percent': '99',
             'l_uh': '0',
             'sizing': 'ratings',
             'ocv_table': ''},
 'Supercapacitor': {'c_f': '130',
                    'v_rated': '54',
                    'esr_mohm': '4',
                    'esl_uh': '0',
                    'module_c_f': '130',
                    'module_v': '54',
                    'module_esr_mohm': '4',
                    'module_esl_uh': '0',
                    'modules_series': '1',
                    'strings_parallel': '1',
                    'v0_percent': '90',
                    'v_min_percent': '50',
                    'p_rated_kw': '0',
                    'r_leak_ohm': '10000',
                    'coupling': 'converter',
                    'sizing': 'modules'},
 'Flywheel': {'v_dc': '800',
              'p_rated_kw': '250',
              'e_max_kwh': '2',
              'speed_percent': '90',
              'speed_min_percent': '50',
              'speed_base_percent': '50',
              'efficiency_percent': '95',
              'standby_loss_percent_h': '2',
              'r_dc_mohm': '1',
              'p_set_kw': '0'},
 'SOFC': {'p_rated_kw': '100',
          'v_rated': '800',
          'p_set_kw': '80',
          'fuel_utilisation_percent': '85',
          'min_load_percent': '30',
          'aux_load_percent': '5',
          'ramp_percent_s': '1'},
 'PV Array': {'module_pmpp_w': '550',
              'module_vmpp': '41.9',
              'module_impp': '13.13',
              'module_voc': '49.9',
              'module_isc': '14.0',
              'module_cells_series': '72',
              'alpha_isc_percent_k': '0.048',
              'beta_voc_percent_k': '-0.27',
              'noct_c': '45',
              'modules_series': '18',
              'strings_parallel': '10',
              'loss_percent': '3',
              'irradiance_wm2': '1000',
              'ambient_c': '25',
              'irradiance_profile_id': '',
              'temperature_profile_id': ''},
 'PCS': {'s_rated_mva': '1',
         'vn_ac_kv': '0',
         'efficiency_percent': '98',
         'no_load_loss_kw': '0',
         'p_set_mw': '0',
         'q_set_mvar': '0',
         'pf': '1',
         'qv_droop_percent': '5',
         'vm_set_pu': '1',
         'droop_pf_percent': '2',
         'droop_qv_percent': '5',
         'current_limit_pu': '1.2',
         'opf_marginal_cost_eur_per_mwh': '',
         'control': 'grid_following',
         'q_mode': 'q'},
 'Grounding Transformer': {'vn_kv': '0',
                           'i_rated_a': '400',
                           't_rated_s': '10',
                           'r_n_ohm': '',
                           'x_n_ohm': '0',
                           'x0_ohm': '',
                           'r0_ohm': ''},
 'DC Bus': {'vn_kv': '0'},
 'Load DC': {'p_mw': '0',
             'load_model': 'constant_power',
             'share_p_percent': '100',
             'share_i_percent': '0',
             'share_r_percent': '0',
             'v_min_pu': '0.8',
             'filter_l_mh': '0',
             'filter_c_uf': '0',
             'load_profile_id': ''},
 'Source DC': {'vm_pu': '1.0', 'r_sc_mohm': '20', 'l_sc_uh': '10'},
 'DC Capacitor': {'c_mf': '10', 'esr_mohm': '2', 'esl_uh': '0.1'},
 'DC Breaker': {'breaker_type': 'solid_state',
                'rated_voltage_kv': '1',
                'rated_current_ka': '1',
                'breaking_capacity_ka': '20',
                'trip_current_ka': '2',
                'opening_time_ms': '0.01',
                'limiting_inductance_mh': '0.01',
                'arrester_clamp_kv': '1.5',
                'arrester_energy_kj': '50'},
 'DC/DC Converter': {'control_mode': 'voltage',
                     'vm_out_pu': '1.0',
                     'p_set_mw': '0.1',
                     'rated_mw': '1',
                     'vn_in_kv': '0.8',
                     'vn_out_kv': '0.4',
                     'efficiency_percent': '98',
                     'no_load_loss_kw': '1',
                     'bidirectional': 'false',
                     'droop_percent': '5',
                     'smoothing_tau_s': '10',
                     'soc_ref_percent': '50',
                     'soc_gain': '0.1',
                     'emt_model': 'average',
                     'switching_khz': '20',
                     'current_limit_pu': '1.2',
                     'c_out_mf': '0',
                     'c_out_esr_mohm': '0',
                     'c_out_esl_uh': '0',
                     'c_in_esr_mohm': '0',
                     'c_in_esl_uh': '0'},
 'Solid-State Transformer': {'vn_mv_kv': '20',
                             'vn_lv_dc_kv': '0.8',
                             'vn_lv_ac_kv': '0.4',
                             'link_kv': '30',
                             'q_mv_mvar': '0',
                             'rect_rated_mw': '1',
                             'rect_efficiency_percent': '98.5',
                             'rect_no_load_kw': '2',
                             'dcdc_rated_mw': '1',
                             'dcdc_efficiency_percent': '98',
                             'dcdc_no_load_kw': '2',
                             'vm_lv_dc_pu': '1.0',
                             'inverter_mode': 'grid_following',
                             'inv_rated_mw': '0.5',
                             'inv_efficiency_percent': '97.5',
                             'inv_no_load_kw': '1',
                             'p_ac_mw': '0.1',
                             'q_ac_mvar': '0',
                             'vm_lv_ac_pu': '1.0',
                             'emt_model': 'average',
                             'switching_khz': '5',
                             'dcdc_switching_khz': '20',
                             'current_limit_pu': '1.2'},
 'VSC': {'r_ohm': '0.01',
         'x_ohm': '0.1',
         'r_dc_ohm': '0.01',
         'control_mode_ac': 'vm_pu',
         'control_value_ac': '1.0',
         'control_mode_dc': 'p_mw',
         'control_value_dc': '0.0',
         'rated_mva': '0',
         'dc_link_mf': '0',
         'dc_link_esr_mohm': '0',
         'dc_link_esl_uh': '0',
         'current_limit_pu': '1.2',
         'emt_model': 'average',
         'switching_khz': '5'},
 'DC Line': {'length_km': '0.1', 'r_ohm_per_km': '0.1', 'max_i_ka': '1', 'l_mh_per_km': '0.3', 'c_uf_per_km': '0.2'}}

# spec list -> (kind, {connection field: what it must name}); a '?' makes it optional.
LISTS = {
    'dc_buses': ('DC Bus', {}),
    'dc_lines': ('DC Line', {'from_bus': 'dc_bus', 'to_bus': 'dc_bus'}),
    'dc_loads': ('Load DC', {'bus': 'dc_bus'}),
    'dc_sources': ('Source DC', {'bus': 'dc_bus'}),
    'dc_capacitors': ('DC Capacitor', {'bus': 'dc_bus'}),
    'vscs': ('VSC', {'bus': 'ac_bus', 'bus_dc': 'dc_bus'}),
    'ssts': ('Solid-State Transformer', {'bus_mv': 'ac_bus', 'bus_lv_dc': 'dc_bus', 'bus_lv_ac': 'ac_bus?'}),
    'dc_dc_converters': ('DC/DC Converter', {'bus_in': 'dc_bus', 'bus_out': 'dc_bus'}),
    'dc_breakers': ('DC Breaker', {'bus': 'dc_bus', 'element': 'dc_element'}),
    'batteries': ('Battery', {'bus': 'dc_bus?'}),
    'supercapacitors': ('Supercapacitor', {'bus': 'dc_bus?'}),
    'flywheels': ('Flywheel', {'bus': 'dc_bus?'}),
    'sofcs': ('SOFC', {'bus': 'dc_bus?'}),
    'pv_arrays': ('PV Array', {'bus': 'dc_bus?'}),
    'pcs': ('PCS', {'bus': 'ac_bus', 'source': 'source?', 'bus_dc': 'dc_bus?'}),
    'grounding_transformers': ('Grounding Transformer', {'bus': 'ac_bus'}),
}
TOP_LEVEL = tuple(LISTS) + ('load_profiles',)
SOURCES = ('Battery', 'Supercapacitor', 'Flywheel', 'SOFC', 'PV Array')
# What a DC breaker switches, by the kind of element beyond it (dcPayload.js DC_BREAKER_TARGETS).
DC_BREAKER_TARGETS = {'DC Bus': 'bus_dc', 'DC Line': 'line_dc', 'VSC': 'vsc', 'Load DC': 'load_dc',
                      'Source DC': 'source_dc', 'DC/DC Converter': 'dc_dc_converter'}
# Row keys where the payload names a connection differently from the spec.
ROW_KEY = {('DC Line', 'from_bus'): 'busFrom', ('DC Line', 'to_bus'): 'busTo', ('PCS', 'source'): 'der'}
PROFILE_KINDS = ('power', 'irradiance', 'temperature')
PROFILE_FIELDS = {'load_profile_id': 'power', 'irradiance_profile_id': 'irradiance',
                  'temperature_profile_id': 'temperature'}


def _text(value):
    """A field as the diagram holds it: text, booleans as true / false."""
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return str(value)


def check(spec, ac_buses, used_ids, problems, num):
    """
    The layer's rows and its drawing, from ``spec``: ``ac_buses`` {id: (name,
    vn_kv)}; ``used_ids`` the ids taken so far (shared with the AC elements);
    ``num`` electrisim_sld._num. Problems are appended, as for the AC elements.
    Returns {'rows': [...], 'elements': [...], 'load_profiles': {...}, 'counts': {...}}.
    """
    layer = {'rows': [], 'elements': [], 'load_profiles': {}, 'counts': {}}
    # Load profiles first: elements refer to them.
    for i, row in enumerate(spec.get('load_profiles') or []):
        where = f'load_profiles[{i}]'
        if not isinstance(row, dict) or not str(row.get('id') or '').strip():
            problems.append(f'{where}: a profile needs an id')
            continue
        pid = str(row['id']).strip()
        kind = str(row.get('kind') or 'power')
        if kind not in PROFILE_KINDS:
            problems.append(f'{where} ({pid}): kind={kind!r} must be one of ' + ', '.join(PROFILE_KINDS))
        values = row.get('values')
        if not isinstance(values, list) or len(values) < 2 or not all(
                isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            problems.append(f'{where} ({pid}): values must be a list of at least two numbers')
            continue
        entry = {'name': str(row.get('name') or pid), 'kind': kind}
        if row.get('t_s') is not None:
            t = row['t_s']
            if not isinstance(t, list) or len(t) != len(values) or any(b <= a for a, b in zip(t, t[1:])):
                problems.append(f'{where} ({pid}): t_s must rise and have one time per value')
                continue
            entry.update(t=list(t), p=list(values))
        else:
            dt = num(row.get('dt_s'), 'dt_s', where, problems, positive=True)
            if dt is None:
                problems.append(f'{where} ({pid}): give dt_s (the step between values) or t_s (their times)')
                continue
            entry.update(dt=dt, t0=num(row.get('t0_s'), 't0_s', where, problems, default=0.0), p=list(values))
        layer['load_profiles'][pid] = entry

    # Every element's id and kind first, so connections can name elements listed later.
    kinds = {}
    dc_buses = {}
    for key, (kind, _) in LISTS.items():
        for i, row in enumerate(spec.get(key) or []):
            if isinstance(row, dict) and str(row.get('id') or '').strip():
                kinds[str(row['id']).strip()] = kind
                if kind == 'DC Bus':
                    dc_buses[str(row['id']).strip()] = row

    counters = {}
    for key, (kind, refs) in LISTS.items():
        rows = spec.get(key)
        if rows is None:
            continue
        if not isinstance(rows, list):
            problems.append(f'{key} must be a list')
            continue
        for i, row in enumerate(rows):
            where = f'{key}[{i}]'
            if not isinstance(row, dict):
                problems.append(f'{where} must be an object')
                continue
            ident = str(row.get('id') or '').strip()
            if not ident:
                problems.append(f'{where}: id is required')
                continue
            where = f'{where} ({ident})'
            if ident in used_ids:
                problems.append(f'{where}: id {ident!r} is used twice - ids are unique across the whole spec')
                continue
            used_ids.add(ident)
            allowed = set(FIELDS[kind]) | set(refs) | {'id', 'name', 'in_service'}
            unknown = sorted(set(row) - allowed)
            if unknown:
                problems.append(f'{where}: unknown field(s) ' + ', '.join(unknown) + f'. A {kind} takes: '
                                + ', '.join(sorted(allowed)))
                continue
            n = counters.get(kind, 0)
            counters[kind] = n + 1
            out = {'typ': f'{kind}{n}', 'name': ident, 'id': ident, 'userFriendlyName': str(row.get('name') or ident)}
            links = {}
            ok = True
            for field, need in refs.items():
                optional = need.endswith('?')
                need = need.rstrip('?')
                target = row.get(field)
                if target in (None, ''):
                    if not optional:
                        problems.append(f'{where}: {field} is required')
                        ok = False
                    continue
                target = str(target).strip()
                if need == 'ac_bus':
                    good = target in ac_buses
                    what = 'an AC bus id (buses)'
                elif need == 'dc_bus':
                    good = target in dc_buses
                    what = 'a DC bus id (dc_buses)'
                elif need == 'source':
                    good = kinds.get(target) in SOURCES
                    what = 'a battery, supercapacitor, flywheel, SOFC or PV array id'
                else:   # dc_element
                    good = kinds.get(target) in DC_BREAKER_TARGETS
                    what = 'a DC bus, line, load, source, VSC or DC/DC converter id'
                if not good:
                    problems.append(f'{where}: {field}={target!r} must be {what}')
                    ok = False
                    continue
                out[ROW_KEY.get((kind, field), field)] = target
                links[field] = target
            if kind == 'DC Breaker' and links.get('element'):
                out['et'] = DC_BREAKER_TARGETS[kinds[links['element']]]
            if kind == 'PCS' and not links.get('source') and not links.get('bus_dc'):
                problems.append(f'{where}: a PCS needs its source (a battery, flywheel, SOFC or PV array id) '
                                'or bus_dc (a DC bus with its source alone on it)')
                ok = False
            for field in FIELDS[kind]:
                if field not in row or row[field] is None:
                    if field in DEFAULTS.get(kind, {}):
                        out[field] = DEFAULTS[kind][field]
                    continue
                value = row[field]
                if isinstance(value, float) and not math.isfinite(value):
                    problems.append(f'{where}: {field} must be a finite number')
                    ok = False
                    continue
                if field in PROFILE_FIELDS and value not in ('', None):
                    entry = layer['load_profiles'].get(str(value))
                    if entry is None:
                        problems.append(f'{where}: {field}={value!r} is not a load_profiles id')
                        ok = False
                        continue
                    if entry['kind'] != PROFILE_FIELDS[field]:
                        problems.append(f'{where}: {field}={value!r} is a {entry["kind"]} profile; '
                                        f'it must be a {PROFILE_FIELDS[field]} profile')
                        ok = False
                        continue
                out[field] = _text(value)
            if 'in_service' in row:
                out['in_service'] = _text(bool(row['in_service']))
            if kind == 'DC Bus' and 'vn_kv' not in row:
                problems.append(f'{where}: vn_kv is required - a DC bus has no default voltage')
                ok = False
            if not ok:
                continue
            layer['rows'].append(out)
            # The canvas gets what the spec gave: the drop configures the rest with the same defaults.
            fields = {k: out[k] for k in FIELDS[kind] if k in row and row[k] is not None and k in out}
            if 'in_service' in out:
                fields['in_service'] = out['in_service']
            layer['elements'].append({
                'kind': kind, 'id': ident, 'name': out['userFriendlyName'], 'attributes': fields,
                # Each connection by spec id; an AC bus also by the name the canvas gives its bar.
                'connections': {f: {'id': t, 'ac_bus_name': ac_buses[t][0]} if refs[f].rstrip('?') == 'ac_bus'
                                else {'id': t} for f, t in links.items()},
            })
    layer['counts'] = dict(counters)
    return layer


def grounding_switch_rows(switches, ac_buses, layer, problems):
    """Switches whose element is a grounding transformer: rows of the layer (the AC net has no such element)."""
    gts = {e['id'] for e in layer['elements'] if e['kind'] == 'Grounding Transformer'}
    n = 0
    for i, row in switches:
        where = f'switches[{i}] ({row.get("id")})'
        element = str(row.get('element') or '').strip()
        if element not in gts:
            problems.append(f'{where}: element={element!r} is not a grounding_transformers id')
            continue
        bus = str(row.get('bus') or '').strip()
        if bus not in ac_buses:
            problems.append(f'{where}: bus={bus!r} is not a bus id')
            continue
        ident = str(row.get('id') or f'Sw_gt{n}')
        layer['rows'].append({'typ': f'Switch{n}', 'name': ident, 'id': ident,
                              'userFriendlyName': str(row.get('name') or ident), 'bus': bus, 'element': element,
                              'et': 't', 'closed': _text(bool(row.get('closed', True))), 'type': 'CB'})
        layer['elements'].append({'kind': 'Switch', 'id': ident, 'name': str(row.get('name') or ident),
                                  'attributes': {'closed': _text(bool(row.get('closed', True)))},
                                  'connections': {'bus': {'id': bus, 'ac_bus_name': ac_buses[bus][0]},
                                                  'element': {'id': element}}})
        n += 1


def with_layer(net, layer, bus_index, params=None):
    """
    A copy of the AC ``net`` with the layer built onto it by pandapower_electrisim's
    own builders, for the study ``params`` names (a load flow by default): some
    elements are built for their study, a grid-forming PCS a source in a load
    flow and a current source in a short circuit. Each bus carries its spec id,
    as a drawn bus carries its cell's.
    """
    import pandapower_electrisim as pe
    full = copy.deepcopy(net)
    ids = (net.get('electrisim_ids') or {}).get('bus') or {}
    if 'id' not in full.bus.columns:
        full.bus['id'] = [ids.get(int(i), '') for i in full.bus.index]
    rows = {'0': dict(params or {'typ': 'PowerFlowPandaPower Parameters'})}
    rows.update({str(k + 1): dict(r) for k, r in enumerate(layer['rows'])})
    dc = {k: r for k, r in rows.items() if str(r.get('typ', '')).startswith('DC Bus')}
    busbars = {ident: int(idx) for ident, idx in bus_index.items()}
    if dc:
        busbars.update(pe.create_busbars(dc, full))
    pe.create_other_elements(rows, full, '0', busbars)
    return full


def results(net, r):
    """The layer's load-flow results, each under its spec id (its row's name)."""
    import numpy as np
    import pandapower_electrisim as pe

    def clean(d):
        return {k: (r(v, 6) if isinstance(v, (float, np.floating)) else v) for k, v in d.items()
                if not k.startswith('_')}

    out = {}
    if len(getattr(net, 'bus_dc', [])):
        out['dc_buses'] = [{
            'id': str(net.bus_dc.at[i, 'name']), 'vn_kv': r(net.bus_dc.at[i, 'vn_kv'], 4),
            'vm_pu': r(net.res_bus_dc.at[i, 'vm_pu'], 5) if i in net.res_bus_dc.index else None,
            'p_mw': r(net.res_bus_dc.at[i, 'p_mw'], 5) if i in net.res_bus_dc.index else None,
        } for i in net.bus_dc.index if not pe._electrisim_is_aux(net.bus_dc, i) and not pe._electrisim_is_hidden(net.bus_dc, i)]
    if len(getattr(net, 'line_dc', [])):
        out['dc_lines'] = [{
            'id': str(net.line_dc.at[i, 'name']), 'p_from_mw': r(net.res_line_dc.at[i, 'p_from_mw'], 5),
            'i_ka': r(net.res_line_dc.at[i, 'i_ka'], 5), 'loading_percent': r(net.res_line_dc.at[i, 'loading_percent'], 2),
        } for i in net.line_dc.index if i in net.res_line_dc.index and not pe._electrisim_is_hidden(net.line_dc, i)]
    if len(getattr(net, 'vsc', [])):
        out['vscs'] = [{
            'id': str(net.vsc.at[i, 'name']), 'p_mw': r(net.res_vsc.at[i, 'p_mw'], 5),
            'q_mvar': r(net.res_vsc.at[i, 'q_mvar'], 5), 'p_dc_mw': r(net.res_vsc.at[i, 'p_dc_mw'], 5),
        } for i in net.vsc.index if i in net.res_vsc.index and not pe._electrisim_is_aux(net.vsc, i)]
    pcs = [pe._electrisim_pcs_result(net, rec) for rec in getattr(net, 'electrisim_pcs', None) or []]
    ders = [pe._electrisim_der_result(net, rec) for rec in getattr(net, 'electrisim_ders', None) or []]
    ders += [p.pop('_source') for p in pcs if p.get('_source')]
    if ders:
        out['sources_and_stores'] = [clean({**d, 'id': d['name']}) for d in ders]
    if pcs:
        out['pcs'] = [clean({**p, 'id': p['name']}) for p in pcs]
    if getattr(net, 'electrisim_dc_dc_converters', None):
        out['dc_dc_converters'] = [clean({**c, 'id': c['name']}) for c in
                                   (pe._electrisim_dc_dc_result(net, rec) for rec in net.electrisim_dc_dc_converters)]
    if getattr(net, 'electrisim_ssts', None):
        out['ssts'] = [clean({**s, 'id': s['name']}) for s in
                       (pe._electrisim_sst_result(net, rec) for rec in net.electrisim_ssts)]
    gts = pe._electrisim_grounding_results(net)
    if gts:
        out['grounding_transformers'] = [clean({**g, 'id': g['name']}) for g in gts]
    if getattr(net, 'warnings', None):
        out['warnings'] = list(net.warnings)
    return out
