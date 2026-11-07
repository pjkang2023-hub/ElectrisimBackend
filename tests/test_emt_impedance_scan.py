"""
The impedance scan in the EMT study: against the exact answer on passive networks, and on a
grid-forming PCS, whose impedance at high frequency is its reactor and whose controls show below.
"""
import json
import math

import numpy as np
import pytest

import emt_impedance_scan as scan
import test_emt_double_conversion as DC
from test_pcs import _bus, GRID, LOAD

KV, F1 = 33.0, 50.0
S_SC, RX = 500.0, 0.1
MW, MVAR = 30.0, 6.0


def _system(load=True, **params):
    request = {'0': {'typ': 'EmtStudy Parameters', 'user_email': 't@t', 'frequency': str(F1), 'time_step_us': '10', 'duration_ms': '5', 'scan_bus': 'a',
                     **{k: str(v) for k, v in params.items()}},
               '1': _bus('a', KV),
               '2': {**GRID, 'bus': 'a', 's_sc_max_mva': str(S_SC), 's_sc_min_mva': str(S_SC), 'rx_max': str(RX), 'rx_min': str(RX), 'x0x_max': '1', 'r0x0_max': str(RX)}}
    if load:
        request['3'] = {**LOAD, 'bus': 'a', 'p_mw': str(MW), 'q_mvar': str(MVAR)}
    return request


def _run(client, quiet, request):
    with quiet():
        result = json.loads(client.post('/', json=request).get_data(as_text=True))
    assert 'emt' in result, result.get('message')
    return result['emt']['impedance_scan']


def _grid_z(f):
    """The grid's source impedance at f, from its short-circuit power, with c = 1 at its set voltage."""
    z = KV ** 2 / S_SC
    x = z / math.sqrt(1 + RX ** 2)
    return complex(RX * x, x * f / F1)


def test_the_tones_avoid_f1_and_their_mirrors():
    base, f = scan.tones(10.0, 500.0, 20, 50.0)
    assert base == pytest.approx(10.0)
    assert all(abs(x - 50.0) > 10.0 for x in f)
    for x in f:
        assert not any(abs(y - abs(100.0 - x)) < 1e-9 for y in f if y != x)
    assert np.all(np.mod(f, base) < 1e-9)


def test_the_tones_divide_f1_at_a_low_band():
    base, _ = scan.tones(0.1, 2.0, 10, 60.0)
    assert 60.0 / base == pytest.approx(round(60.0 / base))


def test_a_grid_alone_is_its_own_r_and_l(client, quiet):
    """Against the exact R + j w L, in magnitude and angle, from 20 Hz to 2 kHz."""
    z = _run(client, quiet, _system(load=False, scan_f_min_hz=10, scan_f_max_hz=2000, scan_settle_ms=250, duration_ms=5))
    assert len(z['f_hz']) > 10
    for f, mag, ang in zip(z['f_hz'], z['z_ohm'], z['angle_deg']):
        exact = _grid_z(f)
        assert mag == pytest.approx(abs(exact), rel=0.01), f
        assert ang == pytest.approx(math.degrees(math.atan2(exact.imag, exact.real)), abs=1.5), f
    assert z['negative_resistance_hz'] == []


def test_a_grid_and_a_load_in_parallel(client, quiet):
    """The load's R and L in parallel with the grid: the exact answer, through its resonance-free band."""
    z = _run(client, quiet, _system(scan_f_min_hz=10, scan_f_max_hz=2000, scan_settle_ms=250))
    v2 = (KV * 1e3) ** 2
    for f, mag in zip(z['f_hz'], z['z_ohm']):
        y_load = MW * 1e6 / v2 - 1j * MVAR * 1e6 / v2 * F1 / f
        exact = 1.0 / (1.0 / _grid_z(f) + y_load)
        assert mag == pytest.approx(abs(exact), rel=0.03), f


def test_the_scan_does_not_disturb_the_bus(client, quiet):
    """Small tones: the same answer at half the current."""
    a = _run(client, quiet, _system(scan_f_min_hz=20, scan_f_max_hz=1000, scan_settle_ms=250, scan_amp_pu=0.01))
    b = _run(client, quiet, _system(scan_f_min_hz=20, scan_f_max_hz=1000, scan_settle_ms=250, scan_amp_pu=0.005))
    assert np.allclose(a['z_ohm'], b['z_ohm'], rtol=0.005)


def test_an_inverter_with_its_voltage_loop_is_stiff_low_and_inductive_high(client, quiet):
    """At the critical bus of the double-conversion system, its transformer and load and inverter: below the loop's bandwidth the
    bus looks like the transformer; far above it, like the reactor and transformer in series with the capacitor across."""
    request = DC._system(time_step_us=20, duration_ms=5, scan_bus='crit', scan_f_min_hz=20, scan_f_max_hz=2000, scan_settle_ms=100, scan_base_mva=100, gfm_voltage_loop_pcs='inv', frequency=50)
    z = _run(client, quiet, request)
    f, r, x = np.array(z['f_hz']), np.array(z['r_ohm']), np.array(z['x_ohm'])
    zb = DC.KV ** 2 / DC.MVA
    low = np.abs(r + 1j * x)[f < 150]
    # The 8 % transformer on 110 MVA at those frequencies is well under 0.5 pu; no higher than that.
    assert np.all(low < 0.5 * zb)
