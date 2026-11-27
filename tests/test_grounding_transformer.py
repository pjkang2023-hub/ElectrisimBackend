"""
The zigzag grounding transformer: a three-wire network's ground. It carries
no balanced current; a ground fault sees its zero-sequence impedance plus three
times its neutral resistor, Z0 + 3 R_N. Checked against the hand calculation in
the IEC short circuit (pandapower), OpenDSS and the EMT study, behind its breaker
open and closed, on an ungrounded and a grounded source.
"""
import json
import math

import pytest

from test_dc_elements import _drawn_request, _with
from test_pcs import LF, SC, _bus, _lv, _post

V = 35.0                    # kV
R_N, X0, R0 = 50.0, 6.0, 0.6
S_SC, RX = 600.0, 0.125


def _grid(grounded):
    x0x = 1.0 if grounded else 1e6
    return {'typ': 'External Grid0', 'name': 'g', 'id': 'cell-g', 'userFriendlyName': 'G', 'bus': 'a', 'vm_pu': '1',
            'va_degree': '0', 's_sc_max_mva': str(S_SC), 's_sc_min_mva': str(S_SC), 'rx_max': str(RX), 'rx_min': str(RX),
            'r0x0_max': '0.1', 'x0x_max': str(x0x), 'r0x0_min': '0.1', 'x0x_min': str(x0x), 'in_service': 'true'}


def _zigzag(name='zz', bus=None, **fields):
    row = {'typ': 'Grounding Transformer0', 'name': name, 'id': f'cell-{name}', 'userFriendlyName': name.upper(),
           'bus': bus, 'r_n_ohm': str(R_N), 'x0_ohm': str(X0), 'r0_ohm': str(R0), 'i_rated_a': '400'}
    row.update({k: str(v) for k, v in fields.items()})
    return row


def _breaker(element, closed=True, name='cb'):
    return {'typ': 'Switch0', 'name': name, 'id': f'cell-{name}', 'userFriendlyName': name.upper(), 'bus': 'a',
            'element': element, 'et': 't', 'closed': 'true' if closed else 'false', 'type': 'CB'}


def _network(params, grounded, *elements):
    request = {'0': dict(params), '1': _bus('a', V), '2': _grid(grounded)}
    return _with(request, *elements)


def _hand_ground_fault_ka(grounded, zigzags=1, c=1.1):
    """IEC 60909: c sqrt(3) U / |Z1 + Z2 + Z0|, the source's Z1 = c U^2 / S_sc, its Z0 = Z1 (X0/X1 1, R0/X0 0.1)."""
    z = c * V * V / S_SC
    x1 = z / math.sqrt(1 + RX * RX)
    z1 = complex(RX * x1, x1)
    zz = complex(R0 + 3 * R_N, X0) / zigzags
    z0 = 1 / (1 / zz + 1 / complex(0.1 * x1, x1)) if grounded else zz
    return c * math.sqrt(3) * V / abs(2 * z1 + z0)


def _ground_fault_ka(client, quiet, request):
    result = _post(client, quiet, request)
    return {r['name']: r['ikss_ka'] for r in result['busbars']}


SC1 = {**SC, 'fault_type': '1ph'}


@pytest.mark.parametrize('grounded', [False, True])
def test_iec_ground_fault_against_the_hand_calculation(client, quiet, grounded):
    """
    Ungrounded source: the zigzag alone, c sqrt(3) U / |Z0 + 3 R_N| ~ 440 A
    (c = 1.1); two in parallel, twice that. Grounded: the utility's and the
    zigzag's zero-sequence paths in parallel. Behind its breaker as drawn.
    """
    one = _ground_fault_ka(client, quiet, _network(SC1, grounded, _zigzag(), _breaker('zz')))
    assert one == {'a': pytest.approx(_hand_ground_fault_ka(grounded), rel=1e-4)}
    two = _ground_fault_ka(client, quiet, _network(SC1, grounded, _zigzag('z1', bus='a'), _zigzag('z2', bus='a')))
    assert two['a'] == pytest.approx(_hand_ground_fault_ka(grounded, zigzags=2), rel=1e-4)
    if not grounded:
        assert one['a'] == pytest.approx(0.440, abs=0.001)


def test_open_breaker_takes_it_out(client, quiet):
    """Its breaker open: an ungrounded network has no ground fault current; a grounded one, the utility's alone."""
    open_ungrounded = _ground_fault_ka(client, quiet, _network(SC1, False, _zigzag(), _breaker('zz', closed=False)))
    assert open_ungrounded['a'] < 1e-3
    open_grounded = _ground_fault_ka(client, quiet, _network(SC1, True, _zigzag(), _breaker('zz', closed=False)))
    z = 1.1 * V * V / S_SC
    x1 = z / math.sqrt(1 + RX * RX)
    utility = 1.1 * math.sqrt(3) * V / abs(2 * complex(RX * x1, x1) + complex(0.1 * x1, x1))
    assert open_grounded['a'] == pytest.approx(utility, rel=1e-4)


def test_default_neutral_resistor_passes_its_rated_current(client, quiet):
    """With no neutral resistor given, V_ph / I_rated: a bolted ground fault with it alone draws its rated current."""
    result = _post(client, quiet, _network(LF, True, _zigzag(bus='a', r_n_ohm='', x0_ohm='', r0_ohm='', i_rated_a=400)))
    (gt,) = result['groundingtransformers']
    assert gt['r_n_ohm'] == pytest.approx(V * 1e3 / math.sqrt(3) / 400)
    assert gt['i_ground_alone_a'] == pytest.approx(400, rel=0.01)


def test_load_flow_unchanged_and_kept_out_of_branch_results(client, quiet):
    """It carries no balanced current: the bus voltages are as without it; it is not a transformer or bus in the results."""
    without = _post(client, quiet, _network(LF, True))
    with_it = _post(client, quiet, _network(LF, True, _zigzag(), _breaker('zz')))
    assert [b['name'] for b in with_it['busbars']] == ['a']
    assert not with_it.get('transformers')
    assert with_it['busbars'][0]['vm_pu'] == pytest.approx(without['busbars'][0]['vm_pu'], abs=1e-9)
    (gt,) = with_it['groundingtransformers']
    assert gt['label'] == 'ZZ' and gt['r_n_ohm'] == R_N and gt['i_ground_alone_a'] == pytest.approx(
        math.sqrt(3) * V * 1e3 / abs(complex(R0 + 3 * R_N, X0)))


def test_opendss_command_gives_the_hand_ground_fault():
    """OpenDSS: its wye winding's neutral on a node of its own through rneut, its delta floating."""
    import opendssdirect as dss
    from opendss_electrisim import grounding_transformer_dss_command
    dss.Text.Command('clear')
    dss.Text.Command(f'New Circuit.t basekV={V} pu=1.0 phases=3 bus1=a MVAsc3={S_SC} MVAsc1=0.0001 x1r1=8 x0r0=10')
    dss.Text.Command(grounding_transformer_dss_command('gt', 'a', V, {'r0': R0, 'x0': X0, 'r_n': R_N, 'x_n': 0.0}))
    dss.Text.Command('New Fault.f bus1=a.1 phases=1 r=0.0001')
    dss.Text.Command(f'set voltagebases=[{V}]')
    dss.Text.Command('calcv')
    dss.Text.Command('solve mode=snap')
    dss.Circuit.SetActiveElement('Fault.f')
    c = dss.CktElement.Currents()
    assert abs(complex(c[0], c[1])) / 1e3 == pytest.approx(_hand_ground_fault_ka(False, c=1.0), rel=1e-3)


@pytest.mark.parametrize('grounded', [False, True])
def test_emt_ground_fault(client, quiet, grounded):
    """
    A bolted phase-a fault at 20 ms: its symmetrical current the hand
    calculation's (c = 1, the load flow's 1.0 pu before it) within 0.5 %;
    ungrounded, all of it through the zigzag's neutral.
    """
    emt = {'typ': 'EmtStudy Parameters', 'user_email': 't@t', 'time_step_us': '20', 'duration_ms': '120',
           'ac_fault_bus': 'a', 'ac_fault_type': 'ag', 'ac_fault_time_ms': '20', 'ac_fault_duration_ms': '0',
           'ac_fault_resistance_ohm': '0.001'}
    with quiet():
        result = json.loads(client.post('/', json=_network(emt, grounded, _zigzag(), _breaker('zz'))).get_data(as_text=True))
    ac = result['emt']['ac']
    (phase,) = ac['fault']['phases']
    assert phase['i_sym_ka'] == pytest.approx(_hand_ground_fault_ka(grounded, c=1.0), rel=0.005)
    (neutral,) = [b for b in ac['branches'] if b['kind'] == 'Grounding transformer']
    if not grounded:
        assert neutral['i_peak_ka'] == pytest.approx(phase['i_peak_ka'], rel=0.005)


@pytest.mark.parametrize('fixture, params', [
    ('reference_radial.diagram_payload.json', None),
    ('reference_radial.diagram_sc_payload.json', None),
    ('reference_radial.diagram_sc1ph_payload.json', None),
    ('reference_radial.diagram_sc1ph_payload.json', {'typ': 'ShortCircuitAnsiPandaPower Parameters'}),
    ('reference_radial.diagram_contingency_payload.json', None),
    ('reference_radial.diagram_opf_payload.json', None),
    ('reference_radial.diagram_harmonic_payload.json', None),
    ('reference_radial.diagram_timeseries_payload.json', {'time_steps': 2}),
    ('reference_radial.diagram_sc_payload.json', {'typ': 'StateEstimationPandaPower Parameters'}),
    ('reference_radial.diagram_sc_payload.json', {'typ': 'EmtStudy Parameters', 'time_step_us': '20',
                                                  'duration_ms': '10'}),
    ('reference_radial.diagram_sc_payload.json', {'typ': 'TransientStabilityAndes Parameters', 'frequency': '50',
                                                  'sn_mva': '100', 'tf': '1', 'fault_enabled': False}),
])
def test_other_studies_run_with_a_grounding_transformer(client, quiet, fixture, params):
    """None may fail because of a grounding transformer, one with its breaker open as well."""
    request = _drawn_request(fixture)
    lv = _lv(request)
    request = _with(request, _zigzag('z1', bus=lv), _zigzag('z2'), {**_breaker('z2', closed=False), 'bus': lv})
    if params:
        key = next(k for k, v in request.items() if 'Parameters' in str(v.get('typ', '')))
        request[key] = {**request[key], **params} if 'typ' not in params else {**params, 'user_email': 't@t'}
    with quiet():
        response = client.post('/', json=request)
    assert response.status_code == 200
    result = json.loads(response.get_data(as_text=True))
    if isinstance(result, dict):
        assert not result.get('error'), result.get('message') or result.get('exception')
