"""
The Load Flow study on a constant-impedance load: 230 kV, 200 km to it. Its fallbacks solve what they can reach, at the
answer the same load gives as a shunt; beyond that its Newton-Raphson can converge on the bus at 0 V, drawing nothing,
which the study refuses rather than reports. A bus with no supply is another thing: dead, and still reported as such.
"""
import json

import pytest

from test_emt_start_load_flow import _as_shunt, _run, _study
from test_pcs import _bus

LF = {'typ': 'PowerFlowPandaPower Parameters', 'frequency': '50', 'algorithm': 'nr', 'calculate_voltage_angles': 'auto',
      'initialization': 'auto', 'user_email': 't@t'}


def _request(p_mw, const_z='100'):
    return {**_study(p_mw, const_z), '0': LF}


def _vm(out, name):
    return next(b for b in out['busbars'] if b['name'] == name)['vm_pu']


@pytest.mark.parametrize('p_mw', [100, 250])
def test_what_its_fallbacks_reach_is_the_true_voltage(client, quiet, p_mw):
    out = _run(client, quiet, _request(p_mw))
    assert _vm(out, 'b') == pytest.approx(_as_shunt(p_mw)[0], abs=1e-4)


def test_a_false_solution_at_0_v_is_refused(client, quiet):
    """400 MW came back as a solved load flow, its bus at 2e-21 pu; the true answer is 0.744 pu."""
    assert _as_shunt(400)[0] == pytest.approx(0.744, abs=1e-3)
    out = _run(client, quiet, _request(400))
    assert 'busbars' not in out and out['error'] is True
    assert 'bus B at 0 V' in out['exception']


def test_beyond_its_reach_it_does_not_converge_and_says_so(client, quiet):
    """300 MW, its answer 0.8125 pu: no plan reaches it from where the last left off, an error rather than a voltage."""
    out = _run(client, quiet, _request(300))
    assert 'busbars' not in out and out['error'] is True


def test_a_bus_with_no_supply_is_not_a_false_solution(client, quiet):
    request = {**_request(50), '6': _bus('c', 230)}
    out = _run(client, quiet, request)
    assert _vm(out, 'b') == pytest.approx(_as_shunt(50)[0], abs=1e-4)
    vm_c = _vm(out, 'c')
    assert vm_c is None or vm_c != vm_c or json.dumps(vm_c) == 'null'


def test_constant_power_is_as_it_was(client, quiet):
    assert _vm(_run(client, quiet, _request(100, const_z='0')), 'b') == pytest.approx(0.9508, abs=1e-3)
