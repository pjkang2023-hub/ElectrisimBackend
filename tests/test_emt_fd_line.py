"""
The frequency-dependent line (emt_fd_line) against what has an answer on paper: its per-km values'
limits, its fit against the exact Zc and A, its driving-point impedance against Zc tanh and Zc coth of
gamma l through the resonances, its steady state against the exact two-port, and its wave's travel time.
"""
import math

import numpy as np
import pytest

import emt_fd_line as fd
import emt_impedance_scan as scan
import emt_solver as es

GEOMETRY = {'height_m': 20.0, 'spacing_m': 7.0, 'radius_m': 0.015, 'rdc_ohm_per_km': 0.06, 'rho_ohm_m': 100.0}
KM, F1, VLL = 200.0, 50.0, 230e3
W0 = 2.0 * math.pi * F1
C_KM_S = 299792.458


def _exact(f, mode=1):
    z1, y1, z0, y0 = fd.per_km(np.asarray(f, dtype=float), GEOMETRY)
    z, y = (z1, y1) if mode == 1 else (z0, y0)
    return np.sqrt(z / y), np.sqrt(z * y)


def test_its_earth_return_at_low_frequency_is_carsons():
    """Carson's limit: a self impedance's real part beyond the conductor is w u0 / 8 per metre."""
    f = np.array([1.0, 5.0])
    far = {**GEOMETRY, 'height_m': 5.0, 'rho_ohm_m': 1000.0}
    _, _, z0, _ = fd.per_km(f, far)
    _, _, z0b, _ = fd.per_km(f, {**far, 'rdc_ohm_per_km': 1e-6})
    w = 2.0 * math.pi * f
    # The ground mode carries 3 w u0 / 8 beyond the conductor's own resistance.
    assert (z0b.real * 1e-3)[0] == pytest.approx(3.0 * w[0] * fd.U0 / 8.0 + 1e-9, rel=0.03)
    assert np.all(z0.real > z0b.real)


def test_its_skin_effect_has_both_limits():
    w = 2.0 * math.pi * np.array([1e-2, 1e6])
    r, rdc = GEOMETRY['radius_m'], GEOMETRY['rdc_ohm_per_km'] * 1e-3
    z = fd._internal(w, r, rdc)
    assert z[0].real == pytest.approx(rdc, rel=1e-3)
    rho = rdc * math.pi * r * r
    # High frequency: the surface impedance over the circumference, R = X.
    surface = math.sqrt(w[1] * fd.U0 * rho / 2.0) / (2.0 * math.pi * r)
    assert z[1].real == pytest.approx(surface, rel=0.02) and z[1].imag == pytest.approx(surface, rel=0.02)


def test_its_aerial_wave_travels_near_light_and_its_surge_impedance_is_the_geometrys():
    zc, gamma = _exact([1e6])
    h, d, r = GEOMETRY['height_m'], GEOMETRY['spacing_m'], GEOMETRY['radius_m']
    assert 2.0 * math.pi * 1e6 / gamma[0].imag == pytest.approx(C_KM_S, rel=0.02)
    lossless = 60.0 * (math.log(2.0 * h / r) - sum(math.log(math.sqrt(4.0 * h * h + x * x) / x) for x in (d, d, 2.0 * d)) / 3.0)
    assert abs(zc[0]) == pytest.approx(lossless, rel=0.03)


@pytest.mark.parametrize('mode', [0, 1])
def test_the_fit_against_the_exact_zc_and_a(mode):
    fit0, fit1 = fd.fits(GEOMETRY, KM)
    fit = fit1 if mode == 1 else fit0
    f = np.logspace(0, 5, 101)
    zc, gamma = _exact(f, mode)
    s = 2j * math.pi * f
    assert np.max(np.abs(fit.zc(s) / zc - 1.0)) < 0.02
    a = np.exp(-gamma * KM)
    big = np.abs(a) > 0.2
    assert np.max(np.abs((fit.a(s)[big] - a[big]) / np.abs(a[big]))) < 0.03
    assert np.max(np.abs(fit.a(s) - a)) < 0.015
    assert np.all(np.abs(fit.a(s)) < 1.0 + 1e-3)
    assert np.all(fit.k >= 0.0)


def _run_scan(far_shorted, f_max=1500.0):
    dt = 2e-5
    ckt = es.Circuit()
    a = [ckt.node(f'a{p}') for p in 'abc']
    b = [ckt.node(f'b{p}') for p in 'abc']
    line = fd.FdLine(ckt, a, b, GEOMETRY, KM, dt, W0, 'L')
    ckt.add_controller(line.control)
    for n in b:
        ckt.add_r(n, 0, 1e-3 if far_shorted else 1e8)
    for n in a:
        ckt.add_r(n, 0, 1e8)
    params = {'scan_f_min_hz': 10, 'scan_f_max_hz': f_max, 'scan_points_per_decade': 30, 'scan_amp_pu': 1e-4, 'scan_settle_ms': 800, 'scan_base_mva': 100}
    sc = scan.ImpedanceScan(ckt, a, VLL, params, F1, dt, 'A')
    ckt.add_controller(sc.control)
    ckt.start_in_ac_steady_state(W0)
    ckt.simulate(sc.t_end + dt, dt, record_every=10 ** 6)
    return sc.result()


def test_its_driving_point_impedance_through_its_resonances():
    """Far end shorted, Zc tanh(gamma l); open, Zc coth(gamma l): the quarter-wave peaks and dips in place and in size."""
    for shorted, exact_of in ((True, np.tanh), (False, lambda x: 1.0 / np.tanh(x))):
        r = _run_scan(shorted)
        f, got = np.array(r['f_hz']), np.array(r['r_ohm']) + 1j * np.array(r['x_ohm'])
        zc, gamma = _exact(f)
        exact = zc * exact_of(gamma * KM)
        assert len(f) > 25
        # Worst on the sharp peaks and dips, where a small shift in a high Q is a large change.
        assert np.max(np.abs(got - exact) / np.abs(exact)) < 0.2, shorted
        assert np.median(np.abs(got - exact) / np.abs(exact)) < 0.03, shorted


def test_its_steady_state_is_the_exact_two_port():
    """Energised at the sending end, open at the far end: the Ferranti rise, 1 / cosh(gamma l), flat from the start."""
    dt = 2e-5
    ckt = es.Circuit()
    a = [ckt.node(f'a{p}') for p in 'abc']
    b = [ckt.node(f'b{p}') for p in 'abc']
    line = fd.FdLine(ckt, a, b, GEOMETRY, KM, dt, W0, 'L')
    ckt.add_controller(line.control)
    v_pk = math.sqrt(2.0 / 3.0) * VLL
    for k, n in enumerate(a):
        ckt.add_rl(0, n, 1e-3, 1e-6, ac=(v_pk, W0, -k * 2.0 * math.pi / 3.0))
    for n in b:
        ckt.add_r(n, 0, 1e8)
    ckt.start_in_ac_steady_state(W0)
    sim = ckt.simulate(0.2, dt)
    _, gamma = _exact([F1])
    rise = abs(1.0 / np.cosh(gamma[0] * KM))
    v = np.sqrt(2.0 / 3.0 * np.sum(sim['v'][:, b] ** 2, axis=1)) / v_pk
    assert rise > 1.01
    assert np.min(v) == pytest.approx(rise, rel=0.005) and np.max(v) == pytest.approx(rise, rel=0.005)


def test_a_wave_arrives_after_its_travel_time_and_doubles():
    """A voltage closed onto the sending end at a phase's peak: the far end stays still until l / v, then rises towards twice."""
    dt = 2e-5
    ckt = es.Circuit()
    a = [ckt.node(f'a{p}') for p in 'abc']
    b = [ckt.node(f'b{p}') for p in 'abc']
    line = fd.FdLine(ckt, a, b, GEOMETRY, KM, dt, W0, 'L')
    ckt.add_controller(line.control)
    v_pk = 1000.0
    for k, n in enumerate(a):
        mid = ckt.node(f's{k}')
        ckt.add_rl(0, mid, 1e-3, 1e-6, e0=v_pk * (1.0, -0.5, -0.5)[k])
        sw = ckt.add_switch(mid, n, closed=False)
        ckt.at(1e-3, lambda st, sw=sw: st.set_switch(sw, True))
    for n in b:
        ckt.add_r(n, 0, 1e8)
    ckt.start_in_ac_steady_state(W0)
    sim = ckt.simulate(1e-3 + 2.2e-3, dt)
    t, v = sim['t'] - 1e-3, sim['v'][:, b[0]]
    tau = line.tau_ms[1] * 1e-3
    assert tau == pytest.approx(KM / C_KM_S, rel=0.03)
    assert np.max(np.abs(v[t < tau * 0.97])) < 0.02 * v_pk
    # Aerial only (no zero sequence applied): near twice, less the line's attenuation.
    peak = float(np.max(v[(t > tau) & (t < 1.9 * tau)]))
    assert 1.6 * v_pk < peak < 2.05 * v_pk


def test_a_line_too_short_for_the_step_is_refused():
    ckt = es.Circuit()
    a = [ckt.node(f'a{p}') for p in 'abc']
    b = [ckt.node(f'b{p}') for p in 'abc']
    with pytest.raises(ValueError, match='travel time'):
        fd.FdLine(ckt, a, b, GEOMETRY, 1.0, 2e-5, W0, 'L')


# --- Through the application -----------------------------------------------------------------

def _request(km=KM, **params):
    from test_pcs import GRID, LINE, LOAD, _bus
    z1, y1, z0, y0 = fd.per_km(np.array([F1]), GEOMETRY)
    line = {**LINE, 'length_km': str(km), 'r_ohm_per_km': str(z1[0].real), 'x_ohm_per_km': str(z1[0].imag), 'c_nf_per_km': str(y1[0].imag / W0 * 1e9), 'r0_ohm_per_km': str(z0[0].real),
            'x0_ohm_per_km': str(z0[0].imag), 'c0_nf_per_km': str(y0[0].imag / W0 * 1e9), 'max_i_ka': '2', **{f'fd_{k}': str(v) for k, v in GEOMETRY.items()}}
    study = {'typ': 'EmtStudy Parameters', 'user_email': 't@t', 'frequency': str(F1), 'time_step_us': '20', 'duration_ms': '100', **params}
    return {'0': {k: str(v) for k, v in study.items()}, '1': _bus('a', VLL / 1e3), '2': _bus('b', VLL / 1e3), '3': line,
            '4': {**GRID, 'bus': 'a', 's_sc_max_mva': '20000', 's_sc_min_mva': '20000'}, '5': {**LOAD, 'bus': 'b', 'p_mw': '50', 'q_mvar': '10', 'const_z_percent': '100', 'const_i_percent': '0'}}


def _emt(client, quiet, request):
    import json
    with quiet():
        out = json.loads(client.post('/', json=request).get_data(as_text=True))
    assert 'emt' in out, out.get('message')
    return out['emt'], out.get('warnings') or []


def _rms_end(emt, label):
    return next(b for b in emt['ac']['buses'] if b['label'] == label)['v_rms_final_pu']


def test_in_a_study_it_agrees_with_pi_sections_at_the_fundamental(client, quiet):
    """A 200 km line with its load: the far bus's voltage from the two models, flat from the start, its travel time reported."""
    fine, w1 = _emt(client, quiet, _request(fd_line='true'))
    pi, _ = _emt(client, quiet, _request(ac_max_section_km=20))
    assert fine['fd_lines'][0]['travel_aerial_ms'] == pytest.approx(KM / C_KM_S * 1e3, rel=0.03)
    assert not [w for w in w1 if 'pi sections' in w or 'disagree' in w]
    assert _rms_end(fine, 'B') == pytest.approx(_rms_end(pi, 'B'), abs=0.005)


def test_a_line_it_cannot_model_stays_pi_and_says_why(client, quiet):
    emt, warnings = _emt(client, quiet, _request(km=1.0, fd_line='true'))
    assert 'fd_lines' not in emt
    assert any('pi sections' in w and 'travel time' in w for w in warnings)
    request = _request(fd_line='true')
    next(v for v in request.values() if v.get('typ') == 'Line0').pop('fd_rho_ohm_m')
    emt, warnings = _emt(client, quiet, request)
    assert 'fd_lines' not in emt and any('fd_rho_ohm_m' in w for w in warnings)


def test_the_studys_switch_reaches_a_line_saved_as_pi_sections(client, quiet):
    """The line dialog stores fd_line 'false' on a line left as pi sections; the study's 'true' still turns it on."""
    request = _request(fd_line='true')
    next(v for v in request.values() if v.get('typ') == 'Line0')['fd_line'] = 'false'
    emt, _ = _emt(client, quiet, request)
    assert emt['fd_lines'][0]['label']
