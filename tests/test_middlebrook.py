"""
Middlebrook's criterion by a series scan (emt_impedance_scan.SeriesScan) against the exact answer on a passive
network: behind the bus, the grid's R + j w L; past it, the line in series with the load's R and L in parallel.
The ratio of the two is the minor-loop gain; a passive network keeps it clear of -1.
"""
import json
import math

import numpy as np
import pytest

from test_pcs import GRID, LINE, LOAD, _bus

KV, F1, S_SC, RX = 33.0, 50.0, 500.0, 0.1
MW, MVAR = 20.0, 4.0
R_KM, X_KM, KM = 0.1, 0.35, 5.0


def _request(**params):
    study = {'typ': 'EmtStudy Parameters', 'user_email': 't@t', 'frequency': str(F1), 'time_step_us': '10', 'duration_ms': '5', 'middlebrook_line': 'L1',
             'scan_f_min_hz': '10', 'scan_f_max_hz': '2000', 'scan_points_per_decade': '8', 'scan_settle_ms': '250', **params}
    line = {**LINE, 'length_km': str(KM), 'r_ohm_per_km': str(R_KM), 'x_ohm_per_km': str(X_KM), 'c_nf_per_km': '0', 'c0_nf_per_km': '0', 'max_i_ka': '1'}
    return {'0': {k: str(v) for k, v in study.items()}, '1': _bus('a', KV), '2': _bus('b', KV), '3': line,
            '4': {**GRID, 'bus': 'a', 's_sc_max_mva': str(S_SC), 's_sc_min_mva': str(S_SC), 'rx_max': str(RX), 'rx_min': str(RX)},
            '5': {**LOAD, 'bus': 'b', 'p_mw': str(MW), 'q_mvar': str(MVAR), 'const_z_percent': '100', 'const_i_percent': '0'}}


def _post(client, quiet, request):
    with quiet():
        out = json.loads(client.post('/', json=request).get_data(as_text=True))
    assert 'emt' in out, out.get('message')
    return out['emt'], out.get('warnings') or []


def test_both_sides_against_the_exact_impedances(client, quiet):
    emt, _ = _post(client, quiet, _request())
    mb = emt['middlebrook']
    assert mb['line'] == 'L1' and len(mb['f_hz']) > 10
    z = KV ** 2 / S_SC
    x = z / math.sqrt(1 + RX ** 2)
    v2 = (KV * 1e3) ** 2
    for k, f in enumerate(mb['f_hz']):
        source = complex(RX * x, x * f / F1)
        y_load = MW * 1e6 / v2 - 1j * MVAR * 1e6 / v2 * F1 / f
        load = complex(R_KM * KM, X_KM * KM * f / F1) + 1.0 / y_load
        assert complex(mb['r_source_ohm'][k], mb['x_source_ohm'][k]) == pytest.approx(source, rel=0.03), f
        assert complex(mb['r_load_ohm'][k], mb['x_load_ohm'][k]) == pytest.approx(load, rel=0.03), f
        assert mb['ratio'][k] == pytest.approx(abs(source / load), rel=0.05), f
    assert mb['distance_to_minus_one'] > 0.5


def test_a_line_not_drawn_says_so(client, quiet):
    emt, warnings = _post(client, quiet, _request(middlebrook_line='nowhere'))
    assert emt.get('middlebrook') is None and any("Middlebrook" in w for w in warnings)
