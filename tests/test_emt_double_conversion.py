"""
The double-conversion UPS in the EMT study: a 100 MW system at 33 kV, its source and
critical buses not joined, a rectifier and a grid-forming PCS on one 1.3 kV DC link
behind their transformers. The power balance, and the mode sequence through sags,
a source-bus fault and an inverter trip.
"""
import json
import math

import pytest

from test_pcs import _bus, _der, _pcs, GRID, LOAD

KV, MW, PF = 33.0, 100.0, 0.98
MVA = MW / PF


def _transformer(name, hv, lv, group):
    return {'typ': 'Transformer0', 'name': name, 'id': f'cell-{name}', 'userFriendlyName': name.upper(), 'hv_bus': hv, 'lv_bus': lv, 'sn_mva': '110',
            'vn_hv_kv': str(KV), 'vn_lv_kv': '0.69', 'vk_percent': '8', 'vkr_percent': '0.3', 'pfe_kw': '0', 'i0_percent': '0', 'vector_group': group,
            'vk0_percent': '8', 'vkr0_percent': '0.3', 'mag0_percent': '100', 'si0_hv_partial': '0.9', 'parallel': '1', 'shift_degree': '0', 'tap_side': 'hv',
            'tap_pos': '0', 'tap_neutral': '0', 'tap_max': '0', 'tap_min': '0', 'tap_step_percent': '0', 'tap_step_degree': '0', 'tap_phase_shifter': 'false', 'in_service': 'true'}


def _system(**params):
    q = MW * math.tan(math.acos(PF))
    return {'0': {'typ': 'EmtStudy Parameters', 'user_email': 't@t', 'double_conversion_pcs': 'inv', 'double_conversion_rectifier_load': 'rect',
                  'double_conversion_source_bus': 'src', 'double_conversion_critical_bus': 'crit', **{k: str(v) for k, v in params.items()}},
            '1': _bus('src', KV), '2': _bus('crit', KV), '3': _bus('rlv', 0.69), '4': _bus('ilv', 0.69),
            '5': {**GRID, 'bus': 'src', 's_sc_max_mva': '2000', 's_sc_min_mva': '2000', 'rx_max': '0.1', 'rx_min': '0.1'},
            '6': _transformer('tr', 'src', 'rlv', 'Dyn'), '7': _transformer('ti', 'crit', 'ilv', 'YNd'),
            '8': {**LOAD, 'name': 'rect', 'id': 'cell-rect', 'userFriendlyName': 'RECT', 'bus': 'rlv', 'p_mw': str(MW), 'q_mvar': '0'},
            '9': {**LOAD, 'bus': 'crit', 'p_mw': str(MW), 'q_mvar': str(q)},
            '10': _der('Battery', 'b', sizing='ratings', vn_v=1300, capacity_kwh=50000, r0_mohm=0.001, soc_percent=60, soc_min_percent=5,
                       soc_max_percent=95, c_rate_discharge=20, c_rate_charge=20, coulombic_efficiency_percent=99, in_service='true'),
            '11': _pcs('inv', 'ilv', 'b', control='grid_forming', s_rated_mva=round(MVA, 2), vn_ac_kv=0.69, p_set_mw=MW, q_mode='q',
                       q_set_mvar=0, pf=1, vm_set_pu=1.03, droop_pf_percent=1, droop_qv_percent=1, current_limit_pu=1.6, in_service='true')}


def _run(client, quiet, **params):
    params = {'time_step_us': 20, **params}
    with quiet():
        result = json.loads(client.post('/', json=_system(**params)).get_data(as_text=True))
    assert 'emt' in result, result.get('message')
    return result


def _modes(ups):
    return [m['mode'] for m in ups['modes']]


def _conv(emt, label):
    return next(c for c in emt['converters'] if c['label'] == label)


def test_steady_power_passes_through_and_the_store_rests(client, quiet):
    emt = _run(client, quiet, duration_ms=200)['emt']
    ups = emt['double_conversion'][0]
    assert _modes(ups) == ['on_grid']
    assert ups['class1']['inside']
    assert ups['v_critical_min_pu'] > 0.95
    rect, inv = _conv(emt, 'INV rectifier'), _conv(emt, 'INV')
    assert inv['p_end_mw'] == pytest.approx(MW, rel=0.1)
    # Its AC power, load convention: the rectifier draws what the inverter delivers.
    assert abs(rect['p_end_mw']) == pytest.approx(inv['p_end_mw'], rel=0.03)
    der = emt['ders'][0]
    assert abs(der['p_end_mw']) < 0.1 * MW


def test_a_sag_above_the_threshold_holds_the_rectifier_current_and_the_critical_bus(client, quiet):
    """To 0.7 pu: it stays on grid, its power falling with the voltage; the critical bus does not see it."""
    emt = _run(client, quiet, duration_ms=250, grid_voltage_profile='custom', grid_voltage_table='0, 0.7; 0.1, 1', grid_voltage_start_ms=60)['emt']
    ups = emt['double_conversion'][0]
    assert _modes(ups) == ['on_grid']
    assert ups['v_source_min_pu'] == pytest.approx(0.7, abs=0.03)
    assert ups['class1']['inside'] and ups['v_critical_min_pu'] > 0.97
    i = ups['trace']['t_ms'].index(next(t for t in ups['trace']['t_ms'] if t >= 150))
    assert ups['trace']['p_rectifier_mw'][i] == pytest.approx(MW * 0.7 / 0.9, rel=0.1)


def test_a_deep_sag_blocks_the_rectifier_and_it_comes_back(client, quiet):
    emt = _run(client, quiet, duration_ms=800, series_ups_handback_s=0.1, grid_voltage_profile='custom', grid_voltage_table='0, 0.3; 0.15, 1', grid_voltage_start_ms=60)['emt']
    ups = emt['double_conversion'][0]
    assert _modes(ups) == ['on_grid', 'on_storage', 'on_grid']
    assert ups['class1']['inside'] and ups['v_critical_min_pu'] > 0.97
    assert not [b for b in emt['converters_blocked'] if b['label'] == 'INV']
    assert abs(_conv(emt, 'INV rectifier')['p_end_mw']) == pytest.approx(MW, rel=0.15)


def test_a_source_bus_fault_does_not_reach_the_critical_bus(client, quiet):
    """A three-phase fault on the source bus: the critical bus holds, and the rectifier is blocked by the time it would feed it."""
    emt = _run(client, quiet, duration_ms=250, ac_fault_bus='src', ac_fault_type='abcg', ac_fault_time_ms=60, ac_fault_duration_ms=100)['emt']
    ups = emt['double_conversion'][0]
    assert _modes(ups)[:2] == ['on_grid', 'on_storage']
    assert ups['class1']['inside']
    assert ups['v_critical_min_pu'] > 0.97
    assert not [b for b in emt['converters_blocked'] if b['label'] == 'INV']


def test_an_inverter_trip_loses_the_critical_bus(client, quiet):
    """No static bypass: the inverter tripped, the critical bus goes dark and the study says so."""
    emt = _run(client, quiet, duration_ms=150, events=json.dumps([{'t': 0.06, 'action': 'trip', 'element': 'INV'}]))['emt']
    ups = emt['double_conversion'][0]
    assert ups['mode_end'] == 'lost' and not ups['class1']['inside']
    assert ups['trace']['v_critical_high_pu'][-1] < 0.3


def test_without_its_parts_it_says_so(client, quiet):
    request = _system(time_step_us=20, duration_ms=30)
    request['0']['double_conversion_pcs'] = 'nothing'
    with quiet():
        result = json.loads(client.post('/', json=request).get_data(as_text=True))
    assert result['emt']['double_conversion'] == []
    assert any('Double-conversion UPS' in w for w in result['warnings'])
