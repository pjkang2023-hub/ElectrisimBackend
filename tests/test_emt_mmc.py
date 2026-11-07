"""
The modular multilevel converter (emt_mmc) against what has an answer on paper: arm currents I_dc / 3 +- i / 2, its
power balance, each arm's energy ripple by Harnefors's formula, no second harmonic in its common current, its AC side's
impedance as the transformer plus half an arm, and a staircase's distortion from the size of its step.
"""
import cmath
import math

import numpy as np
import pytest

import emt_harmonics
import emt_impedance_scan as scan
import emt_mmc
import emt_solver as es

F1, VLL, S = 50.0, 33e3, 100e6
W0 = 2.0 * math.pi * F1
V_DC = 70e3         # a modulation index near 0.8: at 0.95 its arms run out of voltage on their capacitors' ripple
V_PK = math.sqrt(2.0 / 3.0) * VLL
I_PK = math.sqrt(2.0) * S / (math.sqrt(3.0) * VLL)
XT, XA = 0.15, 0.15
Z = VLL * VLL / S


def _build(p_pu=0.8, q_pu=0.2, dt=2e-5, s_grid=5000e6, **kw):
    """The MMC between a stiff DC source and a grid behind its short-circuit impedance, at its load-flow state."""
    ckt = es.Circuit()
    ac = [ckt.node(f'ac{p}') for p in 'abc']
    dc = ckt.node('dc', V_DC)
    ckt.add_rl(0, dc, 0.1, 1e-3, i0=p_pu * S / V_DC, e0=V_DC)
    i_ph = (complex(p_pu, -q_pu) * S) / (1.5 * V_PK)
    x_g = VLL * VLL / s_grid
    for k, n in enumerate(ac):
        e = complex(V_PK, 0.0) - complex(0.1 * x_g, x_g) * i_ph
        ckt.add_rl(n, 0, 0.1 * x_g, x_g / W0, ac=(abs(e), W0, cmath.phase(e) - k * 2.0 * math.pi / 3.0 + math.pi))
    m = emt_mmc.Mmc(ckt, 'M', ac, dc, W0, complex(V_PK, 0.0), i_ph, S, VLL, V_DC, x_t_pu=XT, x_arm_pu=XA, **kw)
    ckt.add_controller(m.control)
    return ckt, m, ac, dc, i_ph


def _run(t_end=0.2, dt=2e-5, **kw):
    ckt, m, ac, dc, i_ph = _build(dt=dt, **kw)
    ckt.start_in_ac_steady_state(W0)
    return ckt, m, ac, dc, i_ph, ckt.simulate(t_end, dt)


def test_it_starts_where_the_load_flow_left_it_and_stays():
    _, m, _, _, i_ph, sim = _run()
    tr = np.array(m.trace)
    assert tr[:, 1].mean() == pytest.approx(0.8 * S, rel=0.005)
    assert tr[:, 2].mean() == pytest.approx(0.2 * S, rel=0.02)
    assert np.ptp(tr[:, 1]) < 0.02 * S
    assert tr[-1, 5] == pytest.approx(m.w_nom, rel=0.005)


def test_its_arm_currents_are_a_third_of_dc_and_half_of_ac():
    _, m, _, _, i_ph, sim = _run()
    t = sim['t']
    keep = t > t[-1] - 0.02
    i_u, i_l = sim['i_rl'][keep, m.k_u[0]], sim['i_rl'][keep, m.k_l[0]]
    assert i_u.mean() == pytest.approx(m.i_dc0 / 3.0, rel=0.01)
    assert i_l.mean() == pytest.approx(m.i_dc0 / 3.0, rel=0.01)
    assert np.ptp(i_u) / 2.0 == pytest.approx(abs(i_ph) / 2.0, rel=0.02)


def test_its_power_balances_across_it():
    _, m, _, _, _, _ = _run()
    tr = np.array(m.trace)
    last = tr[tr[:, 0] > tr[-1, 0] - 0.02]
    drawn = float(np.mean(last[:, 3] * last[:, 4]))
    loss = drawn - float(np.mean(last[:, 1]))
    # Its resistances, by hand: the transformer and half an arm on the AC current, the arms on the common current.
    i_rms2 = (abs(_build()[4]) ** 2) / 2.0
    z = Z
    hand = 3.0 * i_rms2 * (0.005 * z + 0.005 * z / 2.0) + 2.0 * 0.005 * z * m.i_dc0 ** 2 / 3.0
    assert loss == pytest.approx(hand, rel=0.1)


def test_each_arms_energy_ripple_is_harneforss():
    """Its swing from the mean, S / (3 m w) (1 - (m cos phi / 2)^2)^(3/2), at m = 2 E / v_dc, cos phi at its internal voltage."""
    _, m, _, _, i_ph, _ = _run()
    captured = []
    ckt, m, _, _, i_ph = _build()
    ckt.add_controller(lambda t, st: captured.append((t, m.w_u[0])) or False)
    ckt.start_in_ac_steady_state(W0)
    ckt.simulate(0.2, 2e-5)
    t, w = np.array(captured).T
    w = w[t > 0.16]
    swing = (w.max() - w.min()) / 2.0
    e = abs(m.e_ph)
    s_mid = 1.5 * e * abs(i_ph)
    mod = 2.0 * e / V_DC
    cos_phi = math.cos(cmath.phase(m.e_ph) - cmath.phase(i_ph))
    hand = s_mid / (3.0 * mod * W0) * (1.0 - (mod * cos_phi / 2.0) ** 2) ** 1.5
    assert swing == pytest.approx(hand, rel=0.05)


def test_no_second_harmonic_in_its_common_current():
    """Its circulating-current control holds the common current at I_dc / 3. Only the two phases' sum would show less."""
    _, m, _, _, i_ph, sim = _run()
    t = sim['t']
    i_c = (sim['i_rl'][:, m.k_u[0]] + sim['i_rl'][:, m.k_l[0]]) / 2.0
    w = 2.0 * math.pi * 2.0 * F1
    sel = t > t[-1] - 0.04
    amp = 2.0 * abs(np.mean(i_c[sel] * np.exp(-1j * w * t[sel])))
    assert amp < 0.02 * abs(i_ph)


def test_a_step_in_power_is_followed_within_its_loop_and_the_dc_side_follows():
    ckt, m, _, _, _ = _build(p_pu=0.5, q_pu=0.0)
    ckt.at(0.05, lambda st: setattr(m, 'p_set_now', 0.9 * S))
    ckt.start_in_ac_steady_state(W0)
    ckt.simulate(0.25, 2e-5)
    tr = np.array(m.trace)
    assert np.interp(0.04, tr[:, 0], tr[:, 1]) == pytest.approx(0.5 * S, rel=0.02)
    assert np.interp(0.07, tr[:, 0], tr[:, 1]) == pytest.approx(0.9 * S, rel=0.03)
    last = tr[tr[:, 0] > 0.22]
    assert np.mean(last[:, 3] * last[:, 4]) == pytest.approx(0.9 * S, rel=0.03)
    assert last[-1, 5] == pytest.approx(m.w_nom, rel=0.01)


def test_its_ac_side_is_the_transformer_and_half_an_arm():
    """Well above its controls: L_t + L_arm / 2, its current loop's gain in series, in parallel with the grid's."""
    dt = 2e-5
    ckt, m, ac, _, _ = _build(p_pu=0.5, q_pu=0.0, s_grid=500e6)
    sc = scan.ImpedanceScan(ckt, ac, VLL, {'scan_f_min_hz': 500, 'scan_f_max_hz': 5000, 'scan_points_per_decade': 8, 'scan_amp_pu': 0.002, 'scan_settle_ms': 40, 'scan_base_mva': 100}, F1, dt, 'A')
    ckt.add_controller(sc.control)
    ckt.start_in_ac_steady_state(W0)
    ckt.simulate(sc.t_end + dt, dt, record_every=10 ** 6)
    r = sc.result()
    x_g = VLL * VLL / 500e6
    for f, x, r_ in zip(r['f_hz'], r['x_ohm'], r['r_ohm']):
        if f < 2000:
            continue
        ratio = f / F1
        y = 1.0 / complex(0.1 * x_g, x_g * ratio) + 1.0 / complex(0.0075 * Z + m.kp, (XT + XA / 2.0) * Z * ratio)
        assert abs(complex(r_, x)) == pytest.approx(abs(1.0 / y), rel=0.1), f


@pytest.mark.parametrize('levels', [20, 80])
def test_a_staircases_distortion_follows_its_step(levels):
    """Its arm voltages quantised to v_dc / N: an rms error of q / sqrt 12 per arm, so q / sqrt 24 on e, against e's rms."""
    ckt, m, ac, _, _ = _build(levels=levels)
    ckt.start_in_ac_steady_state(W0)
    sim = ckt.simulate(0.12, 1e-5)
    t = sim['t']
    v_mid = sim['v'][:, m.mid[0]] - (sim['v'][:, m.mid[0]] + sim['v'][:, m.mid[1]] + sim['v'][:, m.mid[2]]) / 3.0
    r = emt_harmonics.analyse(t, v_mid, F1, cycles=2, h_max=500)
    hand = (V_DC / levels / math.sqrt(24.0)) / (abs(m.e_ph) / math.sqrt(2.0)) * 100.0
    total = math.hypot(r['thd_percent'], r['interharmonic_percent'])
    assert 0.3 * hand < total < 1.5 * hand


# --- Through the application ---------------------------------------------------------------

def _mmc_request(**vsc):
    import test_emt_converters as C
    request = C._request(time_step_us='10', duration_ms='60')
    for el in request.values():
        if isinstance(el, dict) and str(el.get('typ', '')).startswith('VSC'):
            el.update(emt_model='mmc', mmc_energy_kj_per_mva=40, mmc_levels=0, **vsc)
    return request


def _post(client, quiet, request):
    import json
    with quiet():
        out = json.loads(client.post('/', json=request).get_data(as_text=True))
    assert not out.get('error') and 'emt' in out, out.get('message')
    return out


def test_as_a_vsc_row_it_holds_the_same_state_as_the_average_model(client, quiet):
    """The converter benchmark's rectifier holding its DC bus: its load-flow power, unity power factor, its DC voltage, and it says what it lacks."""
    out = _post(client, quiet, _mmc_request())
    (conv,) = out['emt']['converters']
    assert conv['model'] == 'mmc'
    assert conv['p_end_mw'] == pytest.approx(-0.1503, rel=0.02)
    assert abs(conv['q_end_mvar']) < 0.01
    assert conv['v_dc_end_kv'] == pytest.approx(0.8, rel=2e-3)
    assert any('no blocking' in w for w in out['warnings'])
