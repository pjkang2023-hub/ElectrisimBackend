"""
The load flow an EMT study starts from, on a constant-impedance load: 230 kV, 200 km to the load. pandapower's
Newton-Raphson leaves the load's voltage dependence out of its Jacobian, so it needs more than its default 10
iterations from 100 MW, and heavier it reaches the other root of its equations, the bus at 0 V. The load is a
linear circuit, so its answer is the same load as a shunt, which sits in the admittance matrix.
"""
import json
import warnings

import pandapower as pp
import pytest

import emt_electrisim
from test_pcs import GRID, LINE, LOAD, _bus

LINE_230 = {'length_km': '200', 'r_ohm_per_km': '0.0613', 'x_ohm_per_km': '0.416', 'c_nf_per_km': '8.76', 'max_i_ka': '2'}


def _study(p_mw, const_z='100'):
    study = {'typ': 'EmtStudy Parameters', 'user_email': 't@t', 'frequency': '50', 'time_step_us': '20', 'duration_ms': '60', 'ac_max_section_km': '20'}
    return {'0': study, '1': _bus('a', 230), '2': _bus('b', 230), '3': {**LINE, **LINE_230},
            '4': {**GRID, 'bus': 'a', 's_sc_max_mva': '20000', 's_sc_min_mva': '20000'},
            '5': {**LOAD, 'bus': 'b', 'p_mw': str(p_mw), 'q_mvar': str(p_mw / 5), 'const_z_percent': const_z, 'const_i_percent': '0'}}


def _run(client, quiet, request):
    with quiet():
        return json.loads(client.post('/', json=request).get_data(as_text=True))


def _as_shunt(p_mw):
    """The far bus's voltage and the load drawn there, the load being a shunt."""
    net = pp.create_empty_network(f_hz=50)
    a, b = pp.create_bus(net, vn_kv=230), pp.create_bus(net, vn_kv=230)
    pp.create_ext_grid(net, a, vm_pu=1.0, s_sc_max_mva=20000, rx_max=0.1)
    pp.create_line_from_parameters(net, a, b, length_km=200, r_ohm_per_km=0.0613, x_ohm_per_km=0.416, c_nf_per_km=8.76, max_i_ka=2)
    pp.create_shunt(net, b, p_mw=p_mw, q_mvar=p_mw / 5)
    pp.runpp(net)
    return net.res_bus.vm_pu.at[b], net.res_shunt.p_mw.iat[0]


def _final_b(out):
    return next(x for x in out['emt']['ac']['buses'] if x['label'] == 'B')['v_rms_final_pu']


def test_the_default_iterations_are_not_enough_for_this_load():
    """What the study's fallbacks are for: pandapower alone, at its default, does not solve it."""
    net = pp.create_empty_network(f_hz=50)
    a, b = pp.create_bus(net, vn_kv=230), pp.create_bus(net, vn_kv=230)
    pp.create_ext_grid(net, a, vm_pu=1.0, s_sc_max_mva=20000, rx_max=0.1)
    pp.create_line_from_parameters(net, a, b, length_km=200, r_ohm_per_km=0.0613, x_ohm_per_km=0.416, c_nf_per_km=8.76, max_i_ka=2)
    pp.create_load(net, b, p_mw=100, q_mvar=20, const_z_p_percent=100, const_z_q_percent=100, const_i_p_percent=0, const_i_q_percent=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(pp.LoadflowNotConverged):
            pp.runpp(net)
        pp.runpp(net, max_iteration=50)
    assert net.res_bus.vm_pu.at[b] == pytest.approx(_as_shunt(100)[0], abs=1e-6)


@pytest.mark.parametrize('p_mw', [100, 200, 250])
def test_a_constant_impedance_load_starts_at_its_true_voltage(client, quiet, p_mw):
    """100 MW failed at 10 iterations; 250 MW takes the flat start's 80. The run holds the load flow's voltage."""
    out = _run(client, quiet, _study(p_mw))
    assert 'emt' in out, out.get('message')
    assert _final_b(out) == pytest.approx(_as_shunt(p_mw)[0], abs=0.01)


def test_the_load_it_starts_from_is_what_the_impedance_draws(client, quiet):
    """The load flow's own result behind the run: 100 MW at 1 pu is V squared of that below it."""
    import pandapower_electrisim as pe
    request = _study(100)
    net = pp.create_empty_network(f_hz=50)
    with quiet():
        busbars = pe.create_busbars(request, net)
        pe.create_other_elements(request, net, '0', busbars)
        solved, why = emt_electrisim._start_load_flow(net)
    assert why is None
    vm, p = _as_shunt(100)
    assert solved.res_bus.vm_pu.iat[1] == pytest.approx(vm, abs=1e-6)
    assert solved.res_load.p_mw.iat[0] == pytest.approx(p, abs=1e-4)
    assert solved.res_load.p_mw.iat[0] == pytest.approx(100.0 * vm ** 2, abs=1e-4)


@pytest.mark.parametrize('p_mw', [300, 400])
def test_a_false_solution_at_0_v_is_refused(client, quiet, p_mw):
    """Beyond what its Newton-Raphson can reach: the study says so, and does not run from a dead bus (400 MW did)."""
    assert _as_shunt(p_mw)[0] > 0.7
    out = _run(client, quiet, _study(p_mw))
    assert 'emt' not in out
    assert 'did not solve' in out['message'] and 'bus B at 0 V' in out['message']


def test_constant_power_is_as_it_was(client, quiet):
    out = _run(client, quiet, _study(100, const_z='0'))
    assert _final_b(out) == pytest.approx(0.951, abs=0.005)
