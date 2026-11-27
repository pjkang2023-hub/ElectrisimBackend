"""
Mode shapes against two classical machines on a lossless line, where the answer is exact: the
swing's frequency from the synchronising torque, and its speeds opposite in the ratio of inertia,
M1 dw1 + M2 dw2 = 0 because the power one loses the other gains. With a load, nodal analysis by hand.
"""
import cmath
import json
import math

import numpy as np
import pytest

from test_pcs import LINE, _bus

F = 50.0
KV, BASE = 11.0, 100.0
S1, M1, S2, M2, XD1 = 100.0, 6.0, 200.0, 8.0, 0.3
X_OHM = 0.242          # 0.2 pu on 100 MVA


def _gen(name, bus, mva, m, p_mw, slack):
    return {'typ': 'Generator0', 'name': name, 'id': f'cell-{name}', 'userFriendlyName': name.upper(), 'bus': bus, 'p_mw': str(p_mw), 'vm_pu': '1.0', 'sn_mva': str(mva), 'scaling': '1',
            'slack': 'true' if slack else 'false', 'in_service': 'true', 'controllable': 'true', 'min_p_mw': '', 'max_p_mw': '', 'vn_kv': str(KV), 'xdss_pu': '0.2', 'rdss_ohm': '', 'cos_phi': '0.8',
            'pg_percent': '', 'power_station_trafo': '', 'dyn_machine_model': 'GENCLS', 'dyn_M': str(m), 'dyn_D': '0', 'dyn_ra': '0', 'dyn_xd1': str(XD1), 'dyn_exciter_model': 'NONE', 'dyn_governor_model': 'NONE'}


def _request(p_b=0.0, load=0.0):
    line = {**LINE, 'length_km': '1', 'r_ohm_per_km': '0', 'x_ohm_per_km': str(X_OHM)}
    request = {'0': {'typ': 'EigenvalueAndes Parameters', 'frequency': str(F), 'sn_mva': str(BASE), 'user_email': 't@t', 'n_modes': '5'}, '1': _bus('a', KV), '2': _bus('b', KV), '3': line,
               '4': _gen('g1', 'a', S1, M1, 0.0, True), '5': _gen('g2', 'b', S2, M2, p_b, False)}
    if load:
        request['6'] = {'typ': 'Load0', 'name': 'ld', 'id': 'cell-ld', 'userFriendlyName': 'LD', 'bus': 'a', 'p_mw': str(load), 'q_mvar': '0', 'const_z_percent': '0', 'const_i_percent': '0',
                        'sn_mva': '0', 'scaling': '1', 'type': 'wye', 'in_service': 'true'}
    return request


def _post(client, quiet, request):
    with quiet():
        result = json.loads(client.post('/', json=request).get_data(as_text=True))
    assert not result.get('error'), result.get('message')
    return result


def _swing(result):
    mode = result['least_damped_modes'][0]
    return mode, next(s for s in result['mode_shapes'] if s['mode_index'] == mode['index'])


def test_the_swing_and_its_shape_are_the_exact_ones(client, quiet):
    """No power flowing: E = 1 pu, the machines in phase, K = 1 / X over the whole path."""
    mode, shape = _swing(_post(client, quiet, _request()))
    x = XD1 * BASE / S1 + 0.2 + XD1 * BASE / S2
    m1, m2 = M1 * S1 / BASE, M2 * S2 / BASE
    w = math.sqrt(2.0 * math.pi * F * (1.0 / x) * (1.0 / m1 + 1.0 / m2))
    assert mode['freq_hz'] == pytest.approx(w / (2.0 * math.pi), rel=1e-3)
    rows = {r['machine']: r for r in shape['machines']}
    big, small = (rows['G1'], rows['G2']) if rows['G1']['magnitude'] == 1 else (rows['G2'], rows['G1'])
    assert big is rows['G1']                                  # the lighter machine moves most
    assert small['magnitude'] == pytest.approx(m1 / m2, rel=1e-3)
    assert big['angle_deg'] == pytest.approx(0.0, abs=1e-6)
    assert abs(small['angle_deg']) == pytest.approx(180.0, abs=0.1)
    assert sorted(map(sorted, shape['groups'])) == [['G1'], ['G2']]
    assert shape['speed_participation'] == pytest.approx(0.5, abs=0.01)   # speed and angle share the swing


def _loaded_by_hand(p_pu):
    """
    The load at A as ANDES runs it after the load flow, a constant impedance: the machines' powers by nodal analysis
    of E1 - jx1 - A (load) - jX - B - jx2 - E2, and the 2 x 2 swing matrix from their derivatives.
    """
    x1, x2, x = XD1 * BASE / S1, XD1 * BASE / S2, 0.2
    m = np.array([M1 * S1 / BASE, M2 * S2 / BASE])
    vb = cmath.rect(1.0, math.asin(p_pu * x))
    i_line = (vb - 1.0) / (1j * x)
    i_a = p_pu - i_line                                          # machine 1 supplies what the line does not
    e0 = np.array([1.0 + 1j * x1 * i_a, vb + 1j * x2 * i_line])
    y = np.array([[1 / (1j * x1) + 1 / (1j * x) + p_pu, -1 / (1j * x)], [-1 / (1j * x), 1 / (1j * x) + 1 / (1j * x2)]])

    def powers(angles):
        e = e0 * np.exp(1j * np.array(angles))
        v = np.linalg.solve(y, np.array([e[0] / (1j * x1), e[1] / (1j * x2)]))
        return np.real(e * np.conj(np.array([(e[0] - v[0]) / (1j * x1), (e[1] - v[1]) / (1j * x2)])))

    h = 1e-6
    k = np.array([(powers([h, 0.0]) - powers([-h, 0.0])) / (2 * h), (powers([0.0, h]) - powers([0.0, -h])) / (2 * h)]).T
    lam, vec = np.linalg.eig(-2.0 * math.pi * F * k / m[:, None])
    swing = int(np.argmin(lam.real))
    return math.sqrt(-lam[swing].real) / (2.0 * math.pi), abs(vec[1, swing] / vec[0, swing])


def test_with_power_flowing_the_shape_is_the_networks(client, quiet):
    """150 MW over the line to a constant-impedance load at A, solved by hand: the load breaks M1 dw1 + M2 dw2 = 0."""
    mode, shape = _swing(_post(client, quiet, _request(p_b=150.0, load=150.0)))
    f_hand, ratio_hand = _loaded_by_hand(1.5)
    rows = {r['machine']: r for r in shape['machines']}
    assert mode['freq_hz'] == pytest.approx(f_hand, rel=0.005)
    assert rows['G2']['magnitude'] == pytest.approx(ratio_hand, rel=0.01)
    assert sorted(map(sorted, shape['groups'])) == [['G1'], ['G2']]
