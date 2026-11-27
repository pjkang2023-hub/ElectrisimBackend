"""
The DC/DC converter: input = output / efficiency + no-load loss, in voltage
mode (it holds its output voltage) and power mode (it delivers a set power
into a network another element holds), alone, in a chain, backwards, behind
a breaker, and in every study.
"""
import json

import pytest

from test_dc_elements import _breaker, _drawn_request, _with

ETA, P_NL = 0.98, 0.002      # 98 %, 2 kW


def _conv(name, bus_in, bus_out, **fields):
    row = {'typ': 'DC/DC Converter0', 'name': name, 'id': f'cell-{name}', 'userFriendlyName': name.upper(),
           'bus_in': bus_in, 'bus_out': bus_out, 'rated_mw': '0.5', 'efficiency_percent': str(100 * ETA),
           'no_load_loss_kw': str(1e3 * P_NL), 'control_mode': 'voltage', 'vm_out_pu': '1.0'}
    row.update({k: str(v) for k, v in fields.items()})
    return row


BUS_48 = {'typ': 'DC Bus3', 'name': 'dc_48', 'id': 'cell-dc_48', 'userFriendlyName': 'DC 48 V', 'vn_kv': '0.048'}
LOAD_48 = {'typ': 'Load DC2', 'name': 'ld_48', 'id': 'cell-ld_48', 'userFriendlyName': 'Racks 48 V',
           'bus': 'dc_48', 'p_mw': '0.02'}


def _run(client, quiet, request):
    with quiet():
        response = client.post('/', json=request)
    result = json.loads(response.get_data(as_text=True))
    assert not result.get('error'), result.get('message')
    return result


def _p_in(p_out):
    return p_out / ETA + P_NL if p_out >= 0 else p_out * ETA + P_NL


def _by_id(rows):
    return {r['id']: r for r in rows}


def test_voltage_mode_holds_its_output_and_draws_output_over_efficiency(client, quiet):
    """
    800 V to 48 V, holding 0.99 pu: its input draws the 20 kW load / 0.98 +
    2 kW, and its auxiliary grid, bus and VSC are not in the results.
    """
    result = _run(client, quiet, _with(_drawn_request(), BUS_48, LOAD_48, _conv('k1', 'dc_b', 'dc_48', vm_out_pu=0.99)))
    (k1,) = result['dcdcconverters']
    assert k1['p_out_mw'] == pytest.approx(0.02, abs=1e-9)
    assert k1['p_in_mw'] == pytest.approx(_p_in(0.02), abs=1e-9)
    assert k1['loss_mw'] == pytest.approx(k1['p_in_mw'] - k1['p_out_mw'])
    assert k1['efficiency_percent'] == pytest.approx(100 * 0.02 / _p_in(0.02))
    assert k1['loading_percent'] == pytest.approx(100 * _p_in(0.02) / 0.5)
    buses = _by_id(result['dcbuses'])
    assert k1['vm_out_pu'] == pytest.approx(0.99) and buses['cell-dc_48']['vm_pu'] == pytest.approx(0.99)
    assert k1['vm_in_pu'] == pytest.approx(buses['cell-dc_b']['vm_pu'])
    assert {l['id'] for l in result['loadsdc']} == {'cell-ld_a', 'cell-ld_b', 'cell-ld_48'}
    assert [v['id'] for v in result['vscs']] == ['cell-vsc1']
    assert not any('auxiliary' in str(b.get('name')) for b in result['busbars'])
    assert len(result['externalgrids']) == 1
    # The rectifier supplies the 800 V loads, the cable and the converter's input.
    without = _run(client, quiet, _drawn_request())
    p_vsc = lambda r: -r['vscs'][0]['p_dc_mw']
    assert p_vsc(result) - p_vsc(without) == pytest.approx(_p_in(0.02), rel=0.01)


def test_power_mode_and_a_chain_of_converters(client, quiet):
    """
    K1 holds 48 V from DC bus B; K2, in power mode, sends 5 kW from the 48 V
    bus back into DC bus A. K1 then delivers the 48 V load and K2's input.
    """
    k2 = _conv('k2', 'dc_48', 'dc_a', control_mode='power', p_set_mw=0.005)
    result = _run(client, quiet, _with(_drawn_request(), BUS_48, LOAD_48, _conv('k1', 'dc_b', 'dc_48'), k2))
    conv = _by_id(result['dcdcconverters'])
    k1, k2 = conv['cell-k1'], conv['cell-k2']
    assert k2['mode'] == 'power' and k2['p_out_mw'] == pytest.approx(0.005)
    assert k2['p_in_mw'] == pytest.approx(_p_in(0.005), abs=1e-9)
    assert k1['p_out_mw'] == pytest.approx(0.02 + _p_in(0.005), abs=1e-8)
    assert k1['p_in_mw'] == pytest.approx(_p_in(0.02 + _p_in(0.005)), abs=1e-8)


def test_power_mode_needs_another_element_to_hold_its_output(client, quiet):
    """Into a 48 V bus nothing else holds, a power-mode converter is left out, and its bus with it."""
    result = _run(client, quiet, _with(_drawn_request(), BUS_48, LOAD_48,
                                       _conv('k2', 'dc_b', 'dc_48', control_mode='power', p_set_mw=0.01)))
    assert not result.get('dcdcconverters')
    assert 'cell-dc_48' not in _by_id(result['dcbuses'])
    assert any("DC/DC Converter 'K2' is left out: its output network" in w and 'power mode' in w
               for w in result['warnings'])


@pytest.mark.parametrize('bidirectional', [False, True])
def test_reverse_power_flow(client, quiet, bidirectional):
    """
    A set power of -10 kW sends power from the 48 V side back to DC bus A:
    the input then receives 10 kW x efficiency, less the no-load loss.
    """
    k1 = _conv('k1', 'dc_b', 'dc_48')
    k2 = _conv('k2', 'dc_a', 'dc_48', control_mode='power', p_set_mw=-0.01, bidirectional=bidirectional)
    result = _run(client, quiet, _with(_drawn_request(), BUS_48, LOAD_48, k1, k2))
    k2 = _by_id(result['dcdcconverters'])['cell-k2']
    assert k2['p_out_mw'] == pytest.approx(-0.01)
    assert k2['p_in_mw'] == pytest.approx(_p_in(-0.01))
    warned = any("'K2': power flows from its output to its input" in w for w in result.get('warnings', []))
    assert warned is not bidirectional


def test_breaker_at_a_converter_input(client, quiet):
    """
    Closed, a breaker in front of K1 carries its input current; open, K1 is
    out of service and the 48 V network it held is left out.
    """
    def request(closed):
        return _with(_drawn_request(), BUS_48, LOAD_48, _conv('k1', 'dc_b', 'dc_48'),
                     _breaker('qk', 'dc_b', 'k1', 'dc_dc_converter', closed=closed, rated_current_ka=0.1))
    closed = _run(client, quiet, request(True))
    (qk,) = closed['dcbreakers']
    v_b = _by_id(closed['dcbuses'])['cell-dc_b']['vm_pu'] * 0.8
    assert qk['i_ka'] == pytest.approx(_p_in(0.02) / v_b, rel=1e-6)

    opened = _run(client, quiet, request(False))
    (k1,) = opened['dcdcconverters']
    assert not k1['in_service'] and k1['p_in_mw'] == 0
    assert 'cell-dc_48' not in _by_id(opened['dcbuses'])
    assert any('DC 48 V' in w and 'not connected' in w for w in opened['warnings'])


@pytest.mark.parametrize('fixture, params', [
    ('reference_radial.diagram_sc_payload.json', None),
    ('reference_radial.diagram_timeseries_payload.json', {'time_steps': 2}),
    ('reference_radial.diagram_contingency_payload.json', None),
    ('reference_radial.diagram_harmonic_payload.json', None),
    ('reference_radial.diagram_opf_payload.json', None),
    ('reference_radial.diagram_sc_payload.json', {'typ': 'StateEstimationPandaPower Parameters'}),
    ('reference_radial.diagram_sc_payload.json', {'typ': 'ArcFlashPandaPower Parameters'}),
    ('reference_radial.diagram_sc_payload.json', {'typ': 'DcFaultStudy Parameters', 'duration_ms': '5',
                                                  'fault_bus': 'dc_48'}),
    ('reference_radial.diagram_sc_payload.json', {'typ': 'TransientStabilityAndes Parameters', 'frequency': '50',
                                                  'sn_mva': '100', 'tf': '1', 'fault_enabled': False}),
])
def test_other_studies_run_with_a_dc_dc_converter(client, quiet, fixture, params):
    """None may fail because of a converter, nor show its auxiliary elements."""
    request = _with(_drawn_request(fixture), BUS_48, LOAD_48, _conv('k1', 'dc_b', 'dc_48'))
    if params:
        key = next(k for k, v in request.items() if 'Parameters' in str(v.get('typ', '')))
        request[key] = {**request[key], **params} if 'typ' not in params else {**params, 'user_email': 't@t'}
    with quiet():
        response = client.post('/', json=request)
    assert response.status_code == 200
    text = response.get_data(as_text=True)
    result = json.loads(text)
    if isinstance(result, dict):
        assert not result.get('error'), result.get('message') or result.get('exception')
    assert 'auxiliary grid' not in text and 'auxiliary AC' not in text
    if params and params.get('typ') == 'DcFaultStudy Parameters':
        assert any('DC/DC converters block' in w for w in result['warnings'])
