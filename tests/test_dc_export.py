"""
"Export Pandapower Python Code" with a DC side: the script must build the DC
network pandapower 3.3 can solve - DC buses, VSCs, DC lines, DC loads - and
get the load flow route's own results.

It could export no network with a DC side: DC buses were written with
pp.create_dc_bus (pandapower names it create_bus_dc), DC loads and VSCs with
columns their tables do not have (KeyError: 'bus'), and DC lines not at all.
"""
import json

import pytest

from test_dc_dc_converter import BUS_48, LOAD_48, _conv
from test_dc_elements import _dc_element, _drawn_request, _with


def _with_load_model(request):
    """Server hall B a mixed DC load over a 2 km cable, so its power follows its voltage."""
    _dc_element(request, 'cable')['length_km'] = '2.0'
    _dc_element(request, 'ld_b').update(load_model='mixed', share_p_percent='50', share_i_percent='30',
                                        share_r_percent='20')
    return request


def _with_dc_dc(request):
    """K1 holds a 48 V bus from DC bus B; K2, in power mode, sends 5 kW from it back into DC bus A."""
    return _with(request, BUS_48, LOAD_48, _conv('k1', 'dc_b', 'dc_48'),
                 _conv('k2', 'dc_48', 'dc_a', control_mode='power', p_set_mw=0.005))


NETWORKS = {
    'dc_network': lambda: _drawn_request(),
    'dc_load_model': lambda: _with_load_model(_drawn_request()),
    'dc_dc_converters': lambda: _with_dc_dc(_drawn_request()),
}


def _export(client, quiet, request):
    params = next(v for v in request.values() if isinstance(v, dict) and 'Parameters' in str(v.get('typ')))
    params['exportPython'] = True
    with quiet():
        response = client.post('/', json=request)
    result = json.loads(response.get_data(as_text=True))
    assert not result.get('error'), result.get('message') or result
    code = result.get('pandapower_python')
    assert code, result.get('pandapower_python_error')
    return result, code


@pytest.mark.parametrize('network', list(NETWORKS))
def test_exported_dc_script_reproduces_the_load_flow(client, quiet, network):
    result, code = _export(client, quiet, NETWORKS[network]())
    for call in ('pp.create_bus_dc(', 'pp.create_vsc(', 'pp.create_line_dc_from_parameters(', 'pp.create_load_dc('):
        assert call in code, call
    assert 'create_dc_bus' not in code

    namespace = {}
    with quiet():
        exec(compile(code, f'{network}_export.py', 'exec'), namespace)
    net = namespace['net']

    # res_bus: every bus the route reports (the converters' auxiliary AC buses it leaves out).
    vm = dict(zip(net.bus.loc[net.res_bus.index, 'name'], net.res_bus['vm_pu']))
    va = dict(zip(net.bus.loc[net.res_bus.index, 'name'], net.res_bus['va_degree']))
    assert result['busbars']
    for row in result['busbars']:
        assert vm[row['name']] == pytest.approx(float(row['vm_pu']), abs=1e-6), row['name']
        assert va[row['name']] == pytest.approx(float(row['va_degree']), abs=1e-4), row['name']

    # res_bus_dc
    dc = net.res_bus_dc.assign(name=net.bus_dc.loc[net.res_bus_dc.index, 'name']).set_index('name')
    assert result['dcbuses']
    for row in result['dcbuses']:
        assert dc.at[row['name'], 'vm_pu'] == pytest.approx(row['vm_pu'], abs=1e-6), row['name']
        assert dc.at[row['name'], 'p_mw'] == pytest.approx(row['p_mw'], abs=1e-6), row['name']

    # res_vsc: the drawn VSC; the converters' output stages are VSCs the route reports as converters.
    vsc = net.res_vsc.assign(name=net.vsc.loc[net.res_vsc.index, 'name']).set_index('name')
    assert result['vscs']
    for row in result['vscs']:
        for col in ('p_mw', 'q_mvar', 'p_dc_mw', 'vm_pu', 'vm_dc_pu'):
            assert vsc.at[row['name'], col] == pytest.approx(row[col], abs=1e-6), (row['name'], col)

    # The DC loads draw what the route settled them at.
    loads = dict(zip(net.load_dc['name'], net.res_load_dc['p_dc_mw']))
    for row in result['loadsdc']:
        assert loads[row['name']] == pytest.approx(row['p_mw'], abs=1e-9), row['name']
    # A converter in voltage mode delivers through its output stage, a VSC the script builds as it is.
    for row in result.get('dcdcconverters') or []:
        if row['mode'] == 'voltage':
            assert -vsc.at[f"{row['name']} output", 'p_dc_mw'] == pytest.approx(row['p_out_mw'], abs=1e-6)
    if network == 'dc_dc_converters':
        assert [r['mode'] for r in result['dcdcconverters']] == ['voltage', 'power']


def test_exported_dc_script_says_what_it_cannot_settle(client, quiet):
    """A voltage-dependent DC load and the DC/DC converters are settled by repeated load flows: the script says so."""
    _, code = _export(client, quiet, _with_dc_dc(_with_load_model(_drawn_request())))
    assert 'cannot do' in code
    assert 'voltage-dependent DC loads at the power their voltage gives' in code
    assert "DC/DC converters' inputs at what their outputs deliver" in code
    assert '# ld_b: a voltage-dependent DC load, rated 0.1 MW (50 % constant power, 30 % constant current, ' \
           '20 % constant resistance).' in code
    assert "# k1 input: DC/DC converter k1's input" in code
    _, plain = _export(client, quiet, _drawn_request())
    assert 'cannot do' not in plain
