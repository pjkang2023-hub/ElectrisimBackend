"""
Running motors in the ANDES study: static loads by default (as before), or with motor_model "dynamic" ANDES's
Motor3, on the data the motor starting study derives - its slip in the run, rising through a fault at its bus and
coming back after it.
"""
import numpy as np
import pytest

from test_andes_pcs import TDS
from test_pcs import _post, _two_buses

MOTOR = {'typ': 'Motor0', 'name': 'mo', 'id': 'cell-mo', 'userFriendlyName': 'MO', 'bus': 'b', 'pn_mech_mw': '0.2', 'cos_phi': '0.85',
         'cos_phi_n': '0.85', 'efficiency_percent': '95', 'efficiency_n_percent': '95', 'loading_percent': '100', 'lrc_pu': '6',
         'rx': '0.15', 'vn_kv': '0.4', 'scaling': '1', 'in_service': 'true'}
GEN = {'typ': 'Generator0', 'name': 'g', 'id': 'cell-g', 'userFriendlyName': 'G', 'bus': 'a', 'p_mw': '0.5', 'vm_pu': '1', 'sn_mva': '2',
       'vn_kv': '0.4', 'in_service': 'true'}
PARAMS = {**TDS, 'tf': '3', 'fault_enabled': 'true', 'fault_bus': 'b', 'fault_tf': '1.0', 'fault_tc': '1.08'}


def test_motors_stay_static_loads_unless_asked(client, quiet):
    result = _post(client, quiet, _two_buses(dict(MOTOR), dict(GEN), params={**PARAMS, 'fault_enabled': 'false'}))
    assert result['motors'] == []
    assert any("'MO' has no ANDES dynamic model" in w for w in result['warnings'])


def test_a_running_motor_slows_in_a_fault_and_recovers(client, quiet):
    result = _post(client, quiet, _two_buses(dict(MOTOR), dict(GEN), params={**PARAMS, 'motor_model': 'dynamic'}))
    assert result['power_flow_converged'] and result['converged']
    (m,) = result['motors']
    t, s, p = np.asarray(result['time']), np.asarray(m['slip']), np.asarray(m['p_mw'])
    s0, p0 = s[0], p[0]
    assert 0.0 < s0 < 0.05
    assert 0.17 < p0 < 0.2 / 0.85
    assert np.all(np.abs(s[t < 0.99] - s0) < 1e-4) and np.all(np.abs(p[t < 0.99] - p0) < 1e-3)
    assert m['slip_max'] > s0 + 0.01 and not m['stalled']
    assert s[-1] == pytest.approx(s0, abs=2e-3)
    assert not any("'MO' has no ANDES dynamic model" in w for w in result['warnings'])


def test_a_long_fault_stalls_it(client, quiet):
    result = _post(client, quiet, _two_buses(dict(MOTOR), dict(GEN), params={**PARAMS, 'motor_model': 'dynamic', 'fault_tc': '2.5'}))
    (m,) = result['motors']
    assert m['stalled']
