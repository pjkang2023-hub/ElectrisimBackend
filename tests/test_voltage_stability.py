"""
PV and QV curves against the exact answers for a load behind a lossless line from a stiff source:
the nose at P = E^2 cos(phi) / (2 X (1 + sin(phi))), with E the source's voltage and X the line's,
and the reactive power a bus needs at any voltage from the power-flow equations themselves.
"""
import json
import math

import numpy as np
import pytest

from test_pcs import _bus, GRID, LINE, LOAD

KV, X = 33.0, 10.0


def _system(p, q, **params):
    return {'0': {'typ': 'VoltageStabilityPandaPower Parameters', 'user_email': 't@t', **{k: str(v) for k, v in params.items()}},
            '1': _bus('a', KV), '2': _bus('b', KV), '3': {**GRID, 'bus': 'a', 's_sc_max_mva': '100000', 's_sc_min_mva': '100000'},
            '4': {**LINE, 'busFrom': 'a', 'busTo': 'b', 'length_km': '1', 'r_ohm_per_km': '0.0001', 'x_ohm_per_km': str(X), 'max_i_ka': '50'},
            '5': {**LOAD, 'bus': 'b', 'p_mw': str(p), 'q_mvar': str(q)}}


def _post(client, quiet, request):
    with quiet():
        response = client.post('/', json=request)
    result = json.loads(response.get_data(as_text=True))
    assert 'voltage_stability' in result, str(result)[:300]
    return result['voltage_stability']


def _nose(tan_phi):
    phi = math.atan(tan_phi)
    return KV ** 2 * math.cos(phi) / (2.0 * X * (1.0 + math.sin(phi)))


@pytest.mark.parametrize('tan_phi', [0.0, 0.2, 0.5])
def test_the_nose_is_the_exact_one(client, quiet, tan_phi):
    r = _post(client, quiet, _system(20.0, 20.0 * tan_phi, curve='pv', step=0.2, tolerance=0.001))
    assert not r.get('error'), r
    assert r['nose']['p_total_mw'] == pytest.approx(_nose(tan_phi), rel=0.005)
    assert r['nose']['margin_mw'] == pytest.approx(_nose(tan_phi) - 20.0, rel=0.01)
    assert r['nose']['weakest_bus'] == 'B'


def test_the_voltage_at_a_unity_power_factor_nose(client, quiet):
    r = _post(client, quiet, _system(20.0, 0.0, curve='pv', step=0.2, tolerance=0.0005))
    assert r['nose']['weakest_vm_pu'] == pytest.approx(1.0 / math.sqrt(2.0), abs=0.02)


def test_the_upper_branch_is_the_exact_one(client, quiet):
    r = _post(client, quiet, _system(20.0, 4.0, curve='pv', step=0.25, tolerance=0.01))
    b = next(x for x in r['buses'] if x['name'] == 'B')
    for lam, p, v in zip(r['lambda'], r['p_total_mw'], b['vm_pu']):
        q = 0.2 * p
        # V^4 + (2 Q X - E^2) V^2 + X^2 (P^2 + Q^2) = 0, upper root, in kV, MW, Mvar, ohm
        c = 2.0 * q * X - KV ** 2
        v2 = (-c + math.sqrt(max(c * c - 4.0 * X * X * (p * p + q * q), 0.0))) / 2.0
        assert v * KV == pytest.approx(math.sqrt(v2), rel=0.01), lam


def test_no_nose_within_the_range_says_so(client, quiet):
    r = _post(client, quiet, _system(1.0, 0.0, curve='pv', step=0.5, lambda_max=2.0))
    assert r['nose'] is None and any('No nose' in w for w in r['warnings'])


@pytest.mark.parametrize('p, q', [(20.0, 4.0), (40.0, 8.0)])
def test_the_qv_curve_is_the_exact_one(client, quiet, p, q):
    r = _post(client, quiet, _system(p, q, curve='qv', qv_bus='b', v_max_pu=1.1, v_min_pu=0.45, v_step_pu=0.05))
    assert not r.get('error'), r
    for vpu, qgen in zip(r['v_pu'], r['q_mvar']):
        v = vpu * KV
        sin_d = p * X / (KV * v)
        if abs(sin_d) >= 1.0:
            continue
        exact = q + (v * v - KV * v * math.sqrt(1.0 - sin_d ** 2)) / X
        assert qgen == pytest.approx(exact, abs=0.02 * max(abs(exact), 1.0)), vpu


def test_the_qv_margin_falls_as_the_load_rises(client, quiet):
    light = _post(client, quiet, _system(10.0, 2.0, curve='qv', qv_bus='b', v_step_pu=0.02))
    heavy = _post(client, quiet, _system(40.0, 8.0, curve='qv', qv_bus='b', v_step_pu=0.02))
    assert light['reactive_margin_mvar'] > heavy['reactive_margin_mvar'] > 0.0


def test_a_qv_curve_needs_a_real_bus(client, quiet):
    r = _post(client, quiet, _system(10.0, 2.0, curve='qv', qv_bus='nowhere'))
    assert r['error'] is True
