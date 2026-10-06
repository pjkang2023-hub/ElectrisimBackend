"""
A load flow with voltage angles on, a VSC and two phase-shifting transformers
in series, through the load flow route.

pandapower 3.3 starts a network with a VSC (or another FACTS device) flat even
when it calculates voltage angles, and from a flat start Newton-Raphson finds
a second, collapsed root here - 0.08 p.u. at the load - and calls it converged.
The backend starts it from pandapower's DC load flow of the network without the
VSC instead, which gives the answer pandapower gets without the VSC.
"""
import json

import pandapower as pp
import pytest

from electrisim_payload import build_payload

# The VSC feeds an 800 V DC load from its own transformer on the 35 kV slack
# bus, so it changes nothing at the other buses.
VSC = dict(r_ohm=0.00015, x_ohm=0.0046, r_dc_ohm=0.0005, control_mode_ac='q_mvar', control_value_ac=0.0,
           control_mode_dc='vm_pu', control_value_dc=1.0)


def _ac_network():
    net = pp.create_empty_network()
    a = pp.create_bus(net, 35, name='Grid 35 kV')
    b = pp.create_bus(net, 13.8, name='MV 13.8 kV')
    c = pp.create_bus(net, 0.48, name='Load 0.48 kV')
    d = pp.create_bus(net, 0.48, name='VSC 0.48 kV')
    pp.create_ext_grid(net, a, name='Grid')
    pp.create_transformer_from_parameters(net, a, b, 12, 35, 13.8, 0.6, 8, 0, 0, shift_degree=330, name='T1')
    pp.create_transformer_from_parameters(net, b, c, 4, 13.8, 0.48, 0.8, 6, 0, 0, shift_degree=330, name='T2')
    pp.create_transformer_from_parameters(net, a, d, 7.5, 35, 0.48, 0.7, 7, 0, 0, name='T3')
    pp.create_load(net, c, 3.5, 1.1, name='Load')
    return net


def _request(init, angles='true', facts='vsc'):
    """
    The network as the frontend sends it, angles as the dialog sends them, with
    the VSC and its DC side - or, for facts='ssc', a STATCOM on that bus instead,
    which pandapower starts flat the same way.
    """
    request = build_payload(_ac_network(), {'calculate_voltage_angles': angles, 'initialization': init})

    def el(key, **fields):
        return {'id': f'cell-{key}', 'name': key, 'userFriendlyName': fields.pop('label', key), **fields}
    if facts == 'ssc':
        request['ssc'] = el('ssc', typ='SSC0', label='STATCOM', bus='VSC 0.48 kV', in_service='true', r_ohm='0.0001',
                            x_ohm='0.005', set_vm_pu='1.0', vm_internal_pu='1.0', va_internal_degree='0.0',
                            controllable='true')
        return request
    request['dc'] = el('dc', typ='DC Bus0', label='DC bus', vn_kv='0.8')
    request['vsc'] = el('vsc', typ='VSC0', label='Rectifier', bus='VSC 0.48 kV', bus_dc='dc', in_service='true',
                        **{k: str(v) for k, v in VSC.items()})
    request['ld'] = el('ld', typ='Load DC0', label='DC load', bus='dc', p_mw='1.0')
    return request


def _solve(client, quiet, request):
    with quiet():
        response = client.post('/', json=request)
    result = json.loads(response.get_data(as_text=True))
    assert not result.get('error'), result.get('message') or result
    return result


def _reference():
    """pandapower's answer without the VSC, which leaves these buses as they are."""
    net = _ac_network()
    pp.runpp(net, algorithm='nr', calculate_voltage_angles=True, init='auto')
    return {net.bus.at[i, 'name']: (net.res_bus.at[i, 'vm_pu'], net.res_bus.at[i, 'va_degree'])
            for i in net.bus.index if net.bus.at[i, 'name'] != 'VSC 0.48 kV'}


def test_pandapower_alone_collapses_this_network():
    """The pandapower behaviour the backend works around: if this fails, pandapower has fixed it."""
    net = _ac_network()
    dc = pp.create_bus_dc(net, 0.8)
    pp.create_vsc(net, int(net.bus.index[net.bus.name == 'VSC 0.48 kV'][0]), dc, **VSC)
    pp.create_load_dc(net, dc, 1.0)
    pp.runpp(net, calculate_voltage_angles=True)
    load_bus = int(net.bus.index[net.bus.name == 'Load 0.48 kV'][0])
    assert net.res_bus.at[load_bus, 'vm_pu'] < 0.5


@pytest.mark.parametrize('init', ['auto', 'dc'])
def test_vsc_with_phase_shifting_transformers_in_series(client, quiet, init):
    result = _solve(client, quiet, _request(init))
    buses = {b['name']: b for b in result['busbars']}
    for name, (vm, va) in _reference().items():
        assert buses[name]['vm_pu'] == pytest.approx(vm, abs=1e-6), name
        assert buses[name]['va_degree'] == pytest.approx(va, abs=1e-4), name
    assert buses['Load 0.48 kV']['vm_pu'] == pytest.approx(0.9635, abs=1e-3)
    assert buses['Load 0.48 kV']['va_degree'] == pytest.approx(55.68, abs=1e-2)
    (vsc,) = result['vscs']
    assert vsc['p_dc_mw'] == pytest.approx(-1.0, abs=1e-6) or vsc['p_dc_mw'] == pytest.approx(1.0, abs=1e-6)
    assert not any('collapsed' in w for w in result.get('warnings', []))


@pytest.mark.parametrize('init', ['auto', 'dc', 'flat'])
def test_exported_script_starts_from_the_same_angles(client, quiet, init):
    """
    "Export Pandapower Python Code" gives a script with the backend's angle start,
    which gets the backend's answer; a flat start is exported as chosen, with a note.
    A STATCOM stands in for the VSC: the export does not yet write DC networks
    pandapower can build.
    """
    request = _request(init, facts='ssc')
    request['simulation-parameters']['exportPython'] = True
    result = _solve(client, quiet, request)
    code = result.get('pandapower_python')
    assert code, result.get('pandapower_python_error')
    if init == 'flat':
        assert 'facts_angle_start' not in code and 'collapsed solution' in code
        return
    assert 'init_va_degree=va_start' in code

    namespace = {}
    with quiet():
        exec(compile(code, 'vsc_phase_shift_export.py', 'exec'), namespace)
    net = namespace['net']
    got = dict(zip(net.bus.loc[net.res_bus.index, 'name'], net.res_bus['vm_pu']))
    for row in result['busbars']:
        assert got[row['name']] == pytest.approx(float(row['vm_pu']), abs=1e-6), row['name']
    assert got['Load 0.48 kV'] == pytest.approx(0.9635, abs=1e-3)


def test_flat_start_is_kept_and_warned_about(client, quiet):
    """A flat start the user chose is not replaced, but the result says what it risks."""
    result = _solve(client, quiet, _request('flat'))
    assert any('VSC' in w and 'phase-shifting transformers' in w and 'collapsed' in w
               for w in result.get('warnings', [])), result.get('warnings')
