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
    # DC breakers too - on a cable, and coupling the battery rack - go with the DC network.
    with_dc = opf(_with(_with_dc(plain), _breaker('qa', 'dc_a', 'cable', 'line_dc'),
                        _breaker('qr', 'dc_b', 'rack', 'bus_dc')))
    assert not with_dc.get('dcbreakers')
    assert any('Optimal power flow leaves the DC network out (3 DC buses, 2 DC loads of 0.15 MW)' in w
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


# --- Phase 3: DC load models, DC capacitors, cable data for EMT -------------------

V_DC = 800.0          # V, the DC network's nominal voltage
P_B = 0.1e6           # W, Server hall B's rated power


def _bus_b_load_flow(client, quiet, cable_km=0.1, **load_fields):
    """The DC network with Server hall B given a load model; DC bus B's voltage and B's power."""
    request = _drawn_request()
    _dc_element(request, 'cable')['length_km'] = str(cable_km)
    _dc_element(request, 'ld_b').update({k: str(v) for k, v in load_fields.items()})
    with quiet():
        response = client.post('/', json=request)
    result = json.loads(response.get_data(as_text=True))
    assert not result.get('error'), result.get('message')
    v = next(b['vm_pu'] for b in result['dcbuses'] if b['id'] == 'cell-dc_b')
    p = next(l['p_mw'] for l in result['loadsdc'] if l['id'] == 'cell-ld_b')
    return v, p, result


def _exact_bus_b(cable_km, current_of_v):
    """DC bus A held at 1.0 p.u. by the VSC: V_B = V_A - R I(V_B), solved for V_B (p.u.)."""
    from scipy.optimize import brentq
    r = CABLE['r_ohm_per_km'] * cable_km
    return brentq(lambda v: v - (1.0 - r * current_of_v(v) / V_DC), 0.05, 1.0)


@pytest.mark.parametrize('model, shares, current', [
    ('constant_current', None, lambda v: P_B / V_DC),
    ('constant_resistance', None, lambda v: v * V_DC / (V_DC ** 2 / P_B)),
    ('mixed', (50, 30, 20), lambda v: (0.5 * P_B / (v * V_DC) + 0.3 * P_B / V_DC + 0.2 * v * P_B / V_DC)),
])
def test_dc_load_models_in_the_load_flow(client, quiet, model, shares, current):
    """
    A DC load's power follows its bus voltage: constant current P = P0 v,
    constant resistance P = P0 v^2, or a mix with constant power. The load
    flow repeats until the loads settle; each matches the cable's exact
    solution, over a 2 km cable so the voltage drop shows.
    """
    fields = {'load_model': model}
    if shares:
        fields.update(share_p_percent=shares[0], share_i_percent=shares[1], share_r_percent=shares[2])
    v, p, _ = _bus_b_load_flow(client, quiet, cable_km=2.0, **fields)
    exact = _exact_bus_b(2.0, current)
    assert v == pytest.approx(exact, abs=1e-6)
    assert p * 1e6 == pytest.approx(current(exact) * exact * V_DC, rel=1e-5)


def test_dc_constant_power_load_turns_constant_current_at_low_voltage(client, quiet):
    """
    Over 20 km (2 Ohm) a 0.1 MW constant-power load has no solution: V^2 - V +
    R P/V^2 = 0 needs R P/V^2 <= 1/4 and it is 0.31. Below 0.8 p.u. its
    converter draws constant current, P0/(0.8 V): the bus settles at
    1 - R P0/(0.8 V^2) = 0.609 p.u., drawing 0.076 MW.
    """
    v, p, _ = _bus_b_load_flow(client, quiet, cable_km=20.0, load_model='constant_power', v_min_pu=0.8)
    exact = _exact_bus_b(20.0, lambda v: P_B / (0.8 * V_DC) if v < 0.8 else P_B / (v * V_DC))
    assert exact == pytest.approx(1 - 2.0 * P_B / (0.8 * V_DC ** 2))
    assert v == pytest.approx(exact, abs=1e-6)
    assert p == pytest.approx(0.1 * exact / 0.8, rel=1e-5)


def test_dc_capacitor_holds_its_energy_and_cable_keeps_its_emt_data(client, quiet):
    """
    A DC-link capacitor draws nothing in steady state; the results give its
    stored energy, C V^2 / 2: 10 mF at 800 V holds 3.2 kJ. A DC cable's
    inductance and capacitance are kept for the EMT study.
    """
    import pandapower_electrisim as pe
    request = _drawn_request()
    start = max(int(k) for k in request if str(k).isdigit()) + 1
    request[str(start)] = {'typ': 'DC Capacitor0', 'name': 'cap', 'id': 'cell-cap', 'userFriendlyName': 'DC link',
                           'bus': 'dc_a', 'c_mf': '10', 'esr_mohm': '2', 'esl_uh': '0.5'}
    _dc_element(request, 'cable').update(l_mh_per_km='0.3', c_uf_per_km='0.2')
    with quiet():
        response = client.post('/', json=request)
    result = json.loads(response.get_data(as_text=True))
    assert not result.get('error'), result.get('message')
    (cap,) = result['dccapacitors']
    assert cap['vm_pu'] == pytest.approx(1.0) and cap['energy_kj'] == pytest.approx(3.2)
    # Same DC bus voltages as without it.
    v_b = next(b['vm_pu'] for b in result['dcbuses'] if b['id'] == 'cell-dc_b')
    assert v_b == pytest.approx(1 - 0.01 * P_B / V_DC ** 2, abs=2e-5)

    net = pp.create_empty_network()
    with quiet():
        busbars = pe.create_busbars(request, net)
        pe.create_other_elements(request, net, '0', busbars)
    assert net.line_dc.at[0, 'l_mh_per_km'] == pytest.approx(0.3) and net.line_dc.at[0, 'c_uf_per_km'] == pytest.approx(0.2)
    assert net.electrisim_dc_capacitors[0]['esr_mohm'] == pytest.approx(2.0)


# --- DC circuit breakers ------------------------------------------------------------

def _with(request, *elements):
    start = max(int(k) for k in request if str(k).isdigit()) + 1
    for k, el in enumerate(elements):
        request[str(start + k)] = el
    return request


def _breaker(name, bus, element, et, closed=True, **ratings):
    return {'typ': 'DC Breaker0', 'name': name, 'id': f'cell-{name}', 'userFriendlyName': name.upper(),
            'bus': bus, 'element': element, 'et': et, 'closed': 'true' if closed else 'false',
            **{k: str(v) for k, v in ratings.items()}}


def _run(client, quiet, request):
    with quiet():
        response = client.post('/', json=request)
    result = json.loads(response.get_data(as_text=True))
    assert not result.get('error'), result.get('message')
    return result


def test_dc_breaker_on_a_cable(client, quiet):
    """
    Closed, a breaker at DC bus A's end of the cable changes nothing and carries
    the cable's current there: 0.1 MW at DC bus B's voltage, 0.125 kA, 63 % of its 0.2 kA.
    Open, it takes the cable out: DC bus B has no converter left, so it and
    Server hall B are set aside, and the warnings say so.
    """
    closed = _run(client, quiet, _with(_drawn_request(), _breaker('q1', 'dc_a', 'cable', 'line_dc',
                                                                  rated_current_ka=0.2, rated_voltage_kv=1.0)))
    v_b = next(b['vm_pu'] for b in closed['dcbuses'] if b['id'] == 'cell-dc_b')
    assert v_b == pytest.approx(1 - 0.01 * P_B / V_DC ** 2, abs=2e-5)
    (q1,) = closed['dcbreakers']
    # The cable carries Server hall B's 0.1 MW at DC bus B's voltage.
    assert q1['closed'] is True and q1['i_ka'] == pytest.approx(0.1 / (0.8 * v_b), rel=1e-6)
    assert q1['loading_percent'] == pytest.approx(100 * q1['i_ka'] / 0.2)
    assert len(closed['linedcs']) == 1

    opened = _run(client, quiet, _with(_drawn_request(), _breaker('q1', 'dc_a', 'cable', 'line_dc', closed=False)))
    assert {b['id'] for b in opened['dcbuses']} == {'cell-dc_a'}
    assert any('DC bus B' in w and 'not connected to the AC network through a VSC' in w for w in opened['warnings'])
    (q1,) = opened['dcbreakers']
    assert q1['closed'] is False and q1['i_ka'] == 0.0


def test_dc_breaker_as_a_bus_coupler(client, quiet):
    """
    Between DC bus B and a new DC bus C, a closed breaker is a coupler: C's
    0.04 MW load is supplied at B's voltage and the breaker carries its 0.05 kA.
    Open, C has no converter and is set aside. The coupler is not a DC cable.
    """
    bus_c = {'typ': 'DC Bus3', 'name': 'dc_c', 'id': 'cell-dc_c', 'userFriendlyName': 'DC bus C', 'vn_kv': '0.8'}
    load_c = {'typ': 'Load DC2', 'name': 'ld_c', 'id': 'cell-ld_c', 'userFriendlyName': 'Server hall C',
              'bus': 'dc_c', 'p_mw': '0.04'}
    closed = _run(client, quiet, _with(_drawn_request(), bus_c, load_c,
                                       _breaker('qc', 'dc_b', 'dc_c', 'bus_dc', rated_current_ka=0.1)))
    vm = {b['id']: b['vm_pu'] for b in closed['dcbuses']}
    assert vm['cell-dc_c'] == pytest.approx(vm['cell-dc_b'], abs=1e-6)
    (qc,) = closed['dcbreakers']
    assert qc['i_ka'] == pytest.approx(0.04 / (0.8 * vm['cell-dc_c']), rel=1e-4)
    assert [l['id'] for l in closed['linedcs']] == ['cell-cable']

    opened = _run(client, quiet, _with(_drawn_request(), bus_c, load_c, _breaker('qc', 'dc_b', 'dc_c', 'bus_dc', closed=False)))
    assert 'cell-dc_c' not in {b['id'] for b in opened['dcbuses']}
    assert any('DC bus C' in w and 'not connected to the AC network' in w for w in opened['warnings'])


def test_dc_breaker_at_a_vsc_terminal_and_its_warnings(client, quiet):
    """
    Opening the breaker at the VSC's DC terminal takes the converter out: the
    DC network is left without one and set aside. A breaker rated below its
    bus's voltage, or not on a DC bus, is named.
    """
    lv = _dc_element(_drawn_request(), 'vsc1')['bus']
    result = _run(client, quiet, _with(_drawn_request(),
                                       _breaker('qv', 'dc_a', 'vsc1', 'vsc', closed=False, rated_voltage_kv=0.6),
                                       _breaker('qx', lv, 'cable', 'line_dc')))
    warnings = result['warnings']
    assert 'dcbuses' not in result or not result['dcbuses']
    assert any('DC buses' in w and 'DC bus A' in w for w in warnings), warnings
    assert any("DC Breaker 'QV' is rated 0.6 kV, below its bus's 0.8 kV" in w for w in warnings), warnings
    assert any("DC Breaker 'QX' is not connected to a DC bus" in w for w in warnings), warnings


def test_dc_source_power_from_what_its_bus_draws(client, quiet):
    """
    The battery rack, coupled to DC bus B, holds its bus while the VSC holds
    DC bus A: what the battery supplies leaves through the coupler. The load
    flow reports that power - pandapower's own res_source_dc gave 0 MW for a
    source supplying a load on its own bus.
    """
    result = _run(client, quiet, _with(_drawn_request(), _breaker('qr', 'dc_b', 'rack', 'bus_dc')))
    (batt,) = result['sourcesdc']
    (qr,) = result['dcbreakers']
    v_rack = next(b['vm_pu'] for b in result['dcbuses'] if b['id'] == 'cell-rack') * 0.8
    assert batt['p_mw'] == pytest.approx(qr['i_ka'] * v_rack, rel=1e-6)
    assert batt['p_mw'] > 0.01
