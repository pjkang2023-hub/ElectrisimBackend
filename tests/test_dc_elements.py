"""
DC elements drawn on a diagram: DC buses, a VSC, a DC cable, DC loads and a DC
source, through the load flow.
"""
import json
import os

import pandapower as pp
import pytest

import electrisim_sld as sld

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE_DIR = os.path.join(HERE, 'reference')

# An 800 V DC network fed from the radial grid's 0.4 kV bus by a VSC holding
# 1.0 p.u. on the DC side, a 100 m cable, a DC load on each DC bus, and - as
# the BESS plant builder draws them - a battery rack no converter connects.
VSC = dict(r_ohm=0.001, x_ohm=0.01, r_dc_ohm=0.0001, control_mode_ac='q_mvar', control_value_ac=0.0,
           control_mode_dc='vm_pu', control_value_dc=1.0)
CABLE = dict(length_km=0.1, r_ohm_per_km=0.1, max_i_ka=0.5)
DC_LOADS = {'Server hall A': ('dc_a', 0.05), 'Server hall B': ('dc_b', 0.1)}


def _payload(ac_bus_name):
    def el(key, **fields):
        return {'id': f'cell-{key}', 'name': key, 'userFriendlyName': fields.pop('label', key), **fields}
    return {
        'dc_a': el('dc_a', typ='DC Bus0', label='DC bus A', vn_kv='0.8'),
        'dc_b': el('dc_b', typ='DC Bus1', label='DC bus B', vn_kv='0.8'),
        'rack': el('rack', typ='DC Bus2', label='Battery rack', vn_kv='0.8'),
        'vsc': el('vsc1', typ='VSC0', label='Rectifier', bus=ac_bus_name, bus_dc='dc_a', in_service='true',
                  **{k: str(v) for k, v in VSC.items()}),
        'cable': el('cable', typ='DC Line0', label='DC cable', busFrom='dc_a', busTo='dc_b', in_service='true',
                    **{k: str(v) for k, v in CABLE.items()}),
        'ld_a': el('ld_a', typ='Load DC0', label='Server hall A', bus='dc_a', p_mw=str(DC_LOADS['Server hall A'][1])),
        'ld_b': el('ld_b', typ='Load DC1', label='Server hall B', bus='dc_b', p_mw=str(DC_LOADS['Server hall B'][1])),
        'batt': el('batt', typ='Source DC0', label='Battery', bus='rack', vm_pu='1.0'),
    }


def _with_dc(request):
    """The request with the DC network appended, under numeric keys as the frontend sends them."""
    lv = next(v for v in request.values() if isinstance(v, dict) and str(v.get('typ', '')).startswith('Bus')
              and v.get('userFriendlyName') == 'LV network A')
    start = max(int(k) for k in request if str(k).isdigit()) + 1
    for offset, element in enumerate(_payload(lv['name']).values()):
        request[str(start + offset)] = element
    return request


def _dc_element(request, name):
    return next(v for v in request.values() if isinstance(v, dict) and v.get('name') == name)


def _drawn_request(fixture='reference_radial.diagram_payload.json'):
    with open(os.path.join(REFERENCE_DIR, fixture), encoding='utf-8') as handle:
        return _with_dc(json.load(handle))


def _spec_with_dc():
    """pandapower's own model: the radial spec, with the same DC network added directly."""
    with open(os.path.join(REFERENCE_DIR, 'reference_radial.spec.json'), encoding='utf-8') as handle:
        net, _ = sld.build_network(json.load(handle))
    lv = int(net.bus.index[net.bus.name == 'LV network A'][0])
    a = pp.create_bus_dc(net, 0.8, name='DC bus A')
    b = pp.create_bus_dc(net, 0.8, name='DC bus B')
    pp.create_vsc(net, lv, a, **VSC)
    pp.create_line_dc_from_parameters(net, a, b, **CABLE)
    pp.create_load_dc(net, a, p_dc_mw=DC_LOADS['Server hall A'][1], index=0)
    pp.create_load_dc(net, b, p_dc_mw=DC_LOADS['Server hall B'][1], index=1)
    pp.runpp(net, algorithm='nr', calculate_voltage_angles='auto')
    return net


def test_drawn_dc_network_load_flow_matches_pandapower(client, quiet):
    """
    The DC elements reach pandapower as DC elements: DC buses kept apart from
    AC buses, the VSC on its AC and DC bus, the cable with its own length,
    resistance and rating, both DC loads (pandapower 3.3 overwrote one DC load
    with the next), and their results read from res_bus_dc and p_dc_mw (a DC
    load failed the whole load flow). The battery rack no converter connects
    is set aside with a warning, as pandapower cannot solve it.
    """
    with quiet():
        response = client.post('/', json=_drawn_request())
    result = json.loads(response.get_data(as_text=True))
    assert not result.get('error'), result.get('message') or result
    net = _spec_with_dc()

    dc_buses = {b['id']: b for b in result['dcbuses']}
    assert set(dc_buses) == {'cell-dc_a', 'cell-dc_b'}
    for cell, name in (('cell-dc_a', 'DC bus A'), ('cell-dc_b', 'DC bus B')):
        idx = int(net.bus_dc.index[net.bus_dc.name == name][0])
        assert dc_buses[cell]['vm_pu'] == pytest.approx(net.res_bus_dc.at[idx, 'vm_pu'], abs=1e-6), name
    assert dc_buses['cell-dc_a']['vm_pu'] == pytest.approx(1.0, abs=1e-9)   # the VSC holds it
    # The cable drops 0.1 MW over 10 mOhm at about 800 V: 1.6 V, 0.2 %.
    assert dc_buses['cell-dc_b']['vm_pu'] == pytest.approx(1 - 0.01 * 0.1e6 / 800 ** 2, abs=2e-5)

    loads = {l['id']: l['p_mw'] for l in result['loadsdc']}
    assert loads == pytest.approx({'cell-ld_a': 0.05, 'cell-ld_b': 0.1})
    (cable,) = result['linedcs']
    assert cable['p_from_mw'] == pytest.approx(float(net.res_line_dc.p_from_mw.iloc[0]), abs=1e-6)
    assert cable['loading_percent'] == pytest.approx(float(net.res_line_dc.loading_percent.iloc[0]), abs=1e-4)
    (vsc,) = result['vscs']
    assert vsc['p_dc_mw'] == pytest.approx(float(net.res_vsc.p_dc_mw.iloc[0]), abs=1e-6)
    assert vsc['p_mw'] == pytest.approx(float(net.res_vsc.p_mw.iloc[0]), abs=1e-6)

    vm = {b['name']: b['vm_pu'] for b in result['busbars']}
    for idx in net.bus.index:
        name = net.bus.at[idx, 'name']
        if name in vm:
            assert vm[name] == pytest.approx(net.res_bus.at[idx, 'vm_pu'], abs=1e-6), name
    assert any('Battery rack is not connected to the AC network through a VSC' in w for w in result['warnings'])
    assert 'sourcesdc' not in result


def test_dc_elements_on_the_wrong_buses_are_named(client, quiet):
    """A VSC without its DC bus, a DC load on an AC bus, a DC line from an AC to a DC bus: each is left out and said so."""
    request = _drawn_request()
    lv_name = _dc_element(request, 'vsc1')['bus']
    _dc_element(request, 'vsc1')['bus_dc'] = ''
    _dc_element(request, 'ld_b')['bus'] = lv_name
    _dc_element(request, 'cable')['busTo'] = lv_name
    with quiet():
        response = client.post('/', json=request)
    result = json.loads(response.get_data(as_text=True))
    assert not result.get('error'), result.get('message')
    warnings = result['warnings']
    assert any("VSC 'Rectifier' is not connected to a DC bus" in w for w in warnings), warnings
    assert any("Load DC 'Server hall B' is not connected to a DC bus" in w for w in warnings), warnings
    assert any("DC Line 'DC cable' is left out" in w for w in warnings), warnings
    # With no VSC left, the DC buses are set aside too, and the AC grid solves as drawn.
    assert any('DC buses' in w and 'DC bus A' in w for w in warnings), warnings


def test_optimal_power_flow_leaves_the_dc_network_out_and_says_so(client, quiet):
    """
    pandapower's OPF has no VSC or DC network model: it failed outright
    (a divide by zero on the DC branches) or, started flat, returned the AC
    result as if the DC loads were not there. The DC network is left out,
    the result is the AC grid's, and the warnings say what is missing.
    """
    def opf(request):
        with quiet():
            response = client.post('/', json=request)
        result = json.loads(response.get_data(as_text=True))
        assert not result.get('error'), result.get('exception') or result.get('message')
        return result
    with open(os.path.join(REFERENCE_DIR, 'reference_radial.diagram_opf_payload.json'), encoding='utf-8') as handle:
        plain = json.load(handle)
    without = opf(json.loads(json.dumps(plain)))
    with_dc = opf(_with_dc(plain))
    assert any('Optimal power flow leaves the DC network out (2 DC buses, 2 DC loads of 0.15 MW)' in w
               for w in with_dc['warnings']), with_dc.get('warnings')
    assert [g['p_mw'] for g in with_dc['externalgrids']] == pytest.approx([g['p_mw'] for g in without['externalgrids']])


@pytest.mark.parametrize('fixture, params', [
    ('reference_radial.diagram_sc_payload.json', None),
    ('reference_radial.diagram_timeseries_payload.json', {'time_steps': 2}),
    ('reference_radial.diagram_contingency_payload.json', None),
    ('reference_radial.diagram_harmonic_payload.json', None),
    ('reference_radial.diagram_sc_payload.json', {'typ': 'StateEstimationPandaPower Parameters'}),
    ('reference_radial.diagram_sc_payload.json', {'typ': 'ArcFlashPandaPower Parameters'}),
    ('reference_radial.diagram_sc_payload.json', {'typ': 'TransientStabilityAndes Parameters', 'frequency': '50',
                                                  'sn_mva': '100', 'tf': '1', 'fault_enabled': False}),
])
def test_other_studies_run_with_a_dc_network(client, quiet, fixture, params):
    """The DC elements now reach every study: none may fail because of them."""
    request = _drawn_request(fixture)
    if params:
        key = next(k for k, v in request.items() if 'Parameters' in str(v.get('typ', '')))
        request[key] = {**request[key], **params} if 'typ' not in params else {**params, 'user_email': 't@t'}
    with quiet():
        response = client.post('/', json=request)
    assert response.status_code == 200
    result = json.loads(response.get_data(as_text=True))
    if isinstance(result, dict):
        assert not result.get('error'), result.get('message') or result.get('exception')
