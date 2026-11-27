"""
The Load Flow dialog's "Calculate voltage angles" radio, through the load flow route.

The dialog sends the radio's value as text: 'auto', 'true' or 'false'.
pandapower only special-cases 'auto' and reads anything else by truthiness, so
the text 'false' calculated the angles - a 330 degree transformer moved the
load bus to 28.68 degrees instead of -1.32. The backend reads the text as
True, False or 'auto' before the load flow.
"""
import json

import pandapower as pp
import pytest

import pandapower_electrisim
from electrisim_payload import build_payload


def _network():
    net = pp.create_empty_network()
    a = pp.create_bus(net, 35, name='Grid 35 kV')
    b = pp.create_bus(net, 13.8, name='MV 13.8 kV')
    pp.create_ext_grid(net, a, name='Grid')
    pp.create_transformer_from_parameters(net, a, b, 12, 35, 13.8, 0.6, 8, 0, 0, shift_degree=330, name='T1')
    pp.create_load(net, b, 8, 2, name='Load')
    return net


def _pandapower_angle(calculate_voltage_angles):
    net = _network()
    pp.runpp(net, calculate_voltage_angles=calculate_voltage_angles)
    return net.res_bus.at[int(net.bus.index[net.bus.name == 'MV 13.8 kV'][0]), 'va_degree']


def _solve(client, quiet, angles, **params):
    request = build_payload(_network(), {'calculate_voltage_angles': angles, **params})
    with quiet():
        response = client.post('/', json=request)
    result = json.loads(response.get_data(as_text=True))
    assert not result.get('error'), result.get('message') or result
    return result


def _mv_angle(result):
    return next(float(b['va_degree']) for b in result['busbars'] if b['name'] == 'MV 13.8 kV')


def test_the_transformer_shifts_the_angle_only_when_angles_are_calculated():
    """The network tells the two apart: the shift shows only with angles calculated."""
    assert _pandapower_angle(True) - _pandapower_angle(False) == pytest.approx(30, abs=0.5)


@pytest.mark.parametrize('angles', ['false', 'False', 'FALSE', ' false ', False])
def test_false_does_not_calculate_the_angles(client, quiet, angles):
    va = _mv_angle(_solve(client, quiet, angles))
    assert va == pytest.approx(_pandapower_angle(False), abs=1e-4)
    assert va != pytest.approx(_pandapower_angle(True), abs=1)


@pytest.mark.parametrize('angles', ['true', 'True', True])
def test_true_calculates_the_angles(client, quiet, angles):
    assert _mv_angle(_solve(client, quiet, angles)) == pytest.approx(_pandapower_angle(True), abs=1e-4)


def test_auto_is_left_to_pandapower(client, quiet):
    """Below 70 kV pandapower's 'auto' does not calculate the angles."""
    assert _mv_angle(_solve(client, quiet, 'auto')) == pytest.approx(_pandapower_angle('auto'), abs=1e-4)


def test_exported_script_has_the_boolean(client, quiet):
    result = _solve(client, quiet, 'false', exportPython=True)
    code = result.get('pandapower_python')
    assert code, result.get('pandapower_python_error')
    assert 'calculate_voltage_angles=False' in code
    assert "calculate_voltage_angles='false'" not in code


@pytest.mark.parametrize('value, expected', [
    ('auto', 'auto'), ('AUTO', 'auto'), (None, 'auto'), ('', 'auto'),
    ('true', True), ('1', True), (True, True), (1, True),
    ('false', False), ('0', False), (False, False), (0, False),
])
def test_normalisation(value, expected):
    got = pandapower_electrisim._as_calculate_voltage_angles(value)
    assert got == expected and type(got) is type(expected)


def test_unknown_value_is_rejected():
    with pytest.raises(ValueError, match='Calculate voltage angles'):
        pandapower_electrisim._as_calculate_voltage_angles('sometimes')
