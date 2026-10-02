# -*- coding: utf-8 -*-
"""
Build an Electrisim simulation payload from a pandapower network.

The frontend produces this shape in `prepareNetworkData` (networkDataPreparation.js);
this is the server-side equivalent used only by the tests, so a golden test can be
driven from a standard pandapower case without going through the browser.

Shape: a flat dict of {cell_id: {typ, name, ...}} plus one entry carrying the
simulation parameters. Bus references are by bus NAME, matching what the canvas
emits.

Supported elements are the ones the IEEE/pandapower test cases use: bus, ext_grid,
gen, sgen, load, line, trafo. That covers every case in
frontend/src/main/webapp/templates/power_system_test_cases.
"""

import math

import pandas as pd
import pandapower as pp


def _s(value, default=''):
    """Payload values arrive from the browser as strings."""
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    try:
        if value is pd.NA or pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return str(value)


def _i(value, default=''):
    """
    Integer-valued field, rendered without a decimal point.

    The backend parses tap fields with safe_int(), which is int(value) with a
    fallback of 1 - so '0.0' becomes 1 rather than 0. The browser sends '0' for
    these, so the tests must too. See test_safe_int_decimal_strings.
    """
    text = _s(value, '')
    if text == '':
        return default
    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        return default


def _bus_name(net, idx, prefix='Bus'):
    name = net.bus.at[idx, 'name']
    if name is None or (isinstance(name, float)) or str(name).strip() in ('', 'nan'):
        return f'{prefix}_{idx}'
    return str(name)


def build_payload(net, simulation_parameters=None, user_email='golden@test.local'):
    """Return the dict that would be POSTed to the simulation backend."""
    payload = {}
    n = [0]

    def cell(prefix):
        n[0] += 1
        return f'{prefix}-{n[0]}'

    names = {idx: _bus_name(net, idx) for idx in net.bus.index}

    # --- buses -------------------------------------------------------------
    for idx, row in net.bus.iterrows():
        cid = cell('bus')
        payload[cid] = {
            'typ': 'Bus',
            'name': names[idx],
            'userFriendlyName': names[idx],
            'id': cid,
            'vn_kv': _s(row['vn_kv']),
            'in_service': _s(bool(row.get('in_service', True))).lower(),
        }

    # --- external grids ----------------------------------------------------
    for idx, row in net.ext_grid.iterrows():
        cid = cell('extgrid')
        payload[cid] = {
            'typ': 'External Grid',
            'name': _s(row.get('name'), f'ExtGrid_{idx}'),
            'userFriendlyName': _s(row.get('name'), f'ExtGrid_{idx}'),
            'id': cid,
            'bus': names[row['bus']],
            'vm_pu': _s(row.get('vm_pu', 1.0)),
            'va_degree': _s(row.get('va_degree', 0.0)),
            'in_service': _s(bool(row.get('in_service', True))).lower(),
            'controllable': 'false',
            'min_q_mvar': '', 'max_q_mvar': '',
            's_sc_max_mva': '', 's_sc_min_mva': '',
            'rx_max': '', 'rx_min': '',
            'r0x0_max': '', 'x0x_max': '',
        }

    # --- generators --------------------------------------------------------
    for idx, row in net.gen.iterrows():
        cid = cell('gen')
        payload[cid] = {
            'typ': 'Generator',
            'name': _s(row.get('name'), f'Gen_{idx}'),
            'userFriendlyName': _s(row.get('name'), f'Gen_{idx}'),
            'id': cid,
            'bus': names[row['bus']],
            'p_mw': _s(row['p_mw']),
            'vm_pu': _s(row.get('vm_pu', 1.0)),
            'sn_mva': _s(row.get('sn_mva', '')),
            'scaling': _s(row.get('scaling', 1.0)),
            'slack': _s(bool(row.get('slack', False))).lower(),
            'in_service': _s(bool(row.get('in_service', True))).lower(),
            'controllable': 'true',
            'min_p_mw': _s(row.get('min_p_mw', '')),
            'max_p_mw': _s(row.get('max_p_mw', '')),
            'vn_kv': '', 'xdss_pu': '', 'rdss_ohm': '',
            'cos_phi': '', 'pg_percent': '', 'power_station_trafo': '',
        }

    # --- static generators -------------------------------------------------
    for idx, row in net.sgen.iterrows():
        cid = cell('sgen')
        payload[cid] = {
            'typ': 'Static Generator',
            'name': _s(row.get('name'), f'SGen_{idx}'),
            'userFriendlyName': _s(row.get('name'), f'SGen_{idx}'),
            'id': cid,
            'bus': names[row['bus']],
            'p_mw': _s(row['p_mw']),
            'q_mvar': _s(row.get('q_mvar', 0.0)),
            'sn_mva': _s(row.get('sn_mva', '')),
            'scaling': _s(row.get('scaling', 1.0)),
            'type': _s(row.get('type'), 'wye'),
            'in_service': _s(bool(row.get('in_service', True))).lower(),
        }

    # --- loads -------------------------------------------------------------
    for idx, row in net.load.iterrows():
        cid = cell('load')
        payload[cid] = {
            'typ': 'Load',
            'name': _s(row.get('name'), f'Load_{idx}'),
            'userFriendlyName': _s(row.get('name'), f'Load_{idx}'),
            'id': cid,
            'bus': names[row['bus']],
            'p_mw': _s(row['p_mw']),
            'q_mvar': _s(row.get('q_mvar', 0.0)),
            'const_z_percent': _s(row.get('const_z_percent', 0.0)),
            'const_i_percent': _s(row.get('const_i_percent', 0.0)),
            'sn_mva': _s(row.get('sn_mva', '')),
            'scaling': _s(row.get('scaling', 1.0)),
            'type': _s(row.get('type'), 'wye'),
            'in_service': _s(bool(row.get('in_service', True))).lower(),
            'controllable': 'false',
        }

    # --- lines -------------------------------------------------------------
    for idx, row in net.line.iterrows():
        cid = cell('line')
        payload[cid] = {
            'typ': 'Line',
            'name': _s(row.get('name'), f'Line_{idx}'),
            'userFriendlyName': _s(row.get('name'), f'Line_{idx}'),
            'id': cid,
            'busFrom': names[row['from_bus']],
            'busTo': names[row['to_bus']],
            'length_km': _s(row['length_km']),
            'r_ohm_per_km': _s(row['r_ohm_per_km']),
            'x_ohm_per_km': _s(row['x_ohm_per_km']),
            'c_nf_per_km': _s(row['c_nf_per_km']),
            'g_us_per_km': _s(row.get('g_us_per_km', 0.0)),
            'max_i_ka': _s(row['max_i_ka']),
            'type': _s(row.get('type'), 'ol'),
            'in_service': _s(bool(row.get('in_service', True))).lower(),
            'max_loading_percent': _s(row.get('max_loading_percent', 100)),
            'r0_ohm_per_km': '', 'x0_ohm_per_km': '', 'c0_nf_per_km': '',
            'endtemp_degree': '',
        }

    # --- shunts -----------------------------------------------------------
    # pandapower stores both reactors and capacitors in net.shunt with a signed
    # q_mvar. The backend's 'Shunt Reactor' branch passes p_mw/q_mvar straight to
    # pp.create_shunt, so the sign survives and one mapping covers both.
    for idx, row in net.shunt.iterrows():
        cid = cell('shunt')
        payload[cid] = {
            'typ': 'Shunt Reactor',
            'name': _s(row.get('name'), f'Shunt_{idx}'),
            'userFriendlyName': _s(row.get('name'), f'Shunt_{idx}'),
            'id': cid,
            'bus': names[row['bus']],
            'p_mw': _s(row.get('p_mw', 0.0), '0'),
            'q_mvar': _s(row['q_mvar']),
            'vn_kv': _s(row['vn_kv']),
            'step': _i(row.get('step', 1), '1'),
            'max_step': _i(row.get('max_step', 1), '1'),
            'in_service': _s(bool(row.get('in_service', True))).lower(),
            'vm_set_pu': '', 'discrete_shunt_control': 'false',
            'step_dependency_table': 'false',
        }

    # --- two-winding transformers -----------------------------------------
    for idx, row in net.trafo.iterrows():
        cid = cell('trafo')
        payload[cid] = {
            'typ': 'Transformer',
            'name': _s(row.get('name'), f'Trafo_{idx}'),
            'userFriendlyName': _s(row.get('name'), f'Trafo_{idx}'),
            'id': cid,
            'hv_bus': names[row['hv_bus']],
            'lv_bus': names[row['lv_bus']],
            'busFrom': names[row['hv_bus']],
            'busTo': names[row['lv_bus']],
            'sn_mva': _s(row['sn_mva']),
            'vn_hv_kv': _s(row['vn_hv_kv']),
            'vn_lv_kv': _s(row['vn_lv_kv']),
            'vk_percent': _s(row['vk_percent']),
            'vkr_percent': _s(row['vkr_percent']),
            'pfe_kw': _s(row['pfe_kw']),
            'i0_percent': _s(row['i0_percent']),
            'shift_degree': _s(row.get('shift_degree', 0.0)),
            'tap_side': _s(row.get('tap_side'), ''),
            'tap_pos': _i(row.get('tap_pos')),
            'tap_neutral': _i(row.get('tap_neutral')),
            'tap_min': _i(row.get('tap_min')),
            'tap_max': _i(row.get('tap_max')),
            'tap_step_percent': _s(row.get('tap_step_percent'), ''),
            'tap_step_degree': _s(row.get('tap_step_degree'), ''),
            'in_service': _s(bool(row.get('in_service', True))).lower(),
            'max_loading_percent': _s(row.get('max_loading_percent', 100)),
            'parallel': _i(row.get('parallel', 1), '1'),
            'df': _s(row.get('df', 1.0)),
        }

    params = {
        'typ': 'PowerFlowPandaPower Parameters',
        'frequency': _s(getattr(net, 'f_hz', 50)),
        'algorithm': 'nr',
        'calculate_voltage_angles': 'auto',
        'initialization': 'auto',
        'user_email': user_email,
    }
    params.update(simulation_parameters or {})
    payload['simulation-parameters'] = params
    return payload
