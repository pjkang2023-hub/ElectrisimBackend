"""
The wideband cable with its sheaths (emt_cable.WidebandCable, on the universal line model emt_ulm):

- its per-km impedances at power frequency are the Carson-based ones of the fundamental model;
- its fit's two-port against the exact six-conductor line, at 50 Hz and through the kHz, and passive;
- at power frequency, the sheath answers test_emt_cable checks for the fundamental model: a single-point bonded
  sheath's standing voltage, a solidly bonded one's circulating current, cross-bonding cancelling;
- a wave on a core arrives after the coaxial travel time, l sqrt(eps_r) / c;
- through the study, with fd_sheath_model 'wideband'.
"""
import math

import numpy as np
import pytest

import emt_cable
import emt_solver as es
import emt_ulm
from test_emt_cable import F1, G, I_RMS, KM, VLL, W0, _r_m

DT = 2e-6


def _exact_two_port(fr, km):
    z, y = emt_cable.per_km_wideband(np.array([fr]), G)
    z, y = z[0], y[0]
    lam, vec = np.linalg.eig(z @ y)
    g = np.sqrt(lam)
    g = np.where(g.real < 0, -g, g)
    vi, zi = np.linalg.inv(vec), np.linalg.inv(z)
    y11 = zi @ vec @ np.diag(g / np.tanh(g * km)) @ vi
    y12 = -zi @ vec @ np.diag(g / np.sinh(g * km)) @ vi
    return np.block([[y11, y12], [y12, y11]])


def test_at_power_frequency_its_impedances_are_carsons():
    z, _ = emt_cable.per_km_wideband(np.array([F1]), G)
    r, l, _, _ = emt_cable.matrices(G, F1)
    carson = (r + 1j * W0 * l) * 1e3
    assert np.max(np.abs(z[0] - carson) / np.abs(carson)) < 0.05


@pytest.mark.parametrize('km', [KM, KM / 3.0])
def test_its_fit_is_the_exact_line_and_passive(km):
    f = np.logspace(-2, 6, 241)
    z, y = emt_cable.per_km_wideband(f, G)
    fit = emt_ulm.UlmFit(z, y, km, f)
    ckt = es.Circuit()
    line = emt_ulm.UlmLine(ckt, [ckt.node(f'a{k}') for k in range(6)], [ckt.node(f'b{k}') for k in range(6)], fit, DT, W0, 'C')
    for fr, tol in ((F1, 0.01), (1e3, 0.01), (1e4, 0.01)):
        exact = _exact_two_port(fr, km)
        assert np.max(np.abs(line.admittance(2.0 * math.pi * fr) - exact)) / np.max(np.abs(exact)) < tol, fr
    assert line.passivity > -1e-3
    # Three coaxial modes travelling together, at the insulation's speed.
    coax = min(fit.groups, key=lambda gr: gr['tau'])
    assert len(coax['modes']) == 3
    assert coax['tau'] == pytest.approx(km * 1e3 * math.sqrt(G['eps_r']) / 299792458.0, rel=0.03)


def _run(bonding):
    ckt = es.Circuit()
    a = [ckt.node(f'a{p}') for p in 'abc']
    b = [ckt.node(f'b{p}') for p in 'abc']
    v_pk = math.sqrt(2.0 / 3.0) * VLL
    for k in range(3):
        ckt.add_rl(0, a[k], 0.01, 1e-4, ac=(v_pk, W0, -k * 2.0 * math.pi / 3.0))
        ckt.add_r(b[k], 0, VLL / math.sqrt(3.0) / I_RMS)
    cable = emt_cable.WidebandCable(ckt, a, b, G, KM, DT, W0, 'C', bonding=bonding)
    ckt.start_in_ac_steady_state(W0)
    sim = ckt.simulate(0.04, DT, record_every=5)
    i = np.array(cable.i_from)
    t_i = np.arange(1, len(i) + 1) * DT
    sel = t_i > t_i[-1] - 1.0 / F1
    currents = 2.0 * np.mean(i[sel] * np.exp(-1j * W0 * t_i[sel])[:, None], axis=0)
    return cable, sim, currents


def _phasor(t, y):
    sel = t > t[-1] - 1.0 / F1
    return 2.0 * np.mean(y[sel] * np.exp(-1j * W0 * t[sel]))


def test_at_power_frequency_its_sheaths_are_the_fundamental_models():
    single, sim, cur = _run('single')
    ia = cur[0]
    assert abs(ia) / math.sqrt(2.0) == pytest.approx(I_RMS, rel=0.05)
    per_m = abs(ia) * W0 * emt_cable.U0 / (2.0 * math.pi) * math.log(G['spacing_m'] / _r_m())
    for p in range(3):
        assert abs(_phasor(sim['t'], sim['v'][:, single.sheath_end[p]])) == pytest.approx(per_m * KM * 1e3, rel=0.03), p
    _, _, cur = _run('solid')
    # Against its own six-conductor impedances, the sheaths shorted at both ends; IEC's hand formula (the sheath's
    # DC resistance, no jacket or earth detail) is 3 % off that.
    z, _ = emt_cable.per_km_wideband(np.array([F1]), G)
    balanced = np.exp(-2j * math.pi / 3.0 * np.arange(3))
    exact = np.abs(np.linalg.solve(z[0][3:, 3:], z[0][3:, :3] @ balanced))
    xm = W0 * emt_cable.U0 / (2.0 * math.pi) * math.log(G['spacing_m'] / _r_m())
    hand = xm / math.hypot(G['sheath_rdc_ohm_per_km'] * 1e-3, xm)
    for p in range(3):
        assert abs(cur[3 + p]) / abs(cur[p]) == pytest.approx(exact[p], rel=0.01), p
        assert abs(cur[3 + p]) / abs(cur[p]) == pytest.approx(hand, rel=0.05), p
    i_solid = max(abs(x) for x in cur[3:])
    _, _, cur = _run('cross')
    assert max(abs(x) for x in cur[3:]) < 0.05 * i_solid


def test_a_wave_on_a_core_arrives_after_the_coaxial_travel_time():
    ckt = es.Circuit()
    a = [ckt.node(f'a{p}') for p in 'abc']
    b = [ckt.node(f'b{p}') for p in 'abc']
    mid = ckt.node('s')
    ckt.add_rl(0, mid, 1e-3, 1e-7, e0=1000.0)
    sw = ckt.add_switch(mid, a[0], closed=False)
    ckt.at(1e-4, lambda st: st.set_switch(sw, True))
    for n in a[1:] + b:
        ckt.add_r(n, 0, 1e8)
    cable = emt_cable.WidebandCable(ckt, a, b, G, KM, 1e-7, W0, 'C', bonding='solid')
    ckt.start_in_ac_steady_state(W0)
    sim = ckt.simulate(1e-4 + 60e-6, 1e-7)
    t, v = sim['t'] - 1e-4, sim['v'][:, b[0]]
    tau = KM * 1e3 * math.sqrt(G['eps_r']) / 299792458.0
    assert np.max(np.abs(v[t < 0.95 * tau])) < 0.02 * 1000.0
    assert np.max(v[(t > tau) & (t < 2.5 * tau)]) > 1.5 * 1000.0


def test_through_the_study_as_wideband(client, quiet):
    import json
    from test_pcs import GRID, LINE, LOAD, _bus
    r, l, c_ins, _ = emt_cable.matrices(G, F1)
    z1 = complex(r[0, 0] - r[0, 1], W0 * (l[0, 0] - l[0, 1])) * 1e3
    line = {**LINE, 'length_km': str(KM), 'r_ohm_per_km': str(z1.real), 'x_ohm_per_km': str(z1.imag), 'c_nf_per_km': str(c_ins * 1e12), 'type': 'cs',
            'max_i_ka': '1', 'fd_line': 'true', 'fd_kind': 'cable', 'fd_bonding': 'cross', 'fd_sheath_model': 'wideband',
            **{f'fd_{k}': str(v) for k, v in G.items()}}
    study = {'typ': 'EmtStudy Parameters', 'user_email': 't@t', 'frequency': str(F1), 'time_step_us': '2', 'duration_ms': '40'}
    request = {'0': study, '1': _bus('a', 33), '2': _bus('b', 33), '3': line, '4': {**GRID, 'bus': 'a', 's_sc_max_mva': '2000', 's_sc_min_mva': '2000'},
               '5': {**LOAD, 'bus': 'b', 'p_mw': '30', 'q_mvar': '3', 'const_z_percent': '100', 'const_i_percent': '0'}}
    with quiet():
        out = json.loads(client.post('/', json=request).get_data(as_text=True))
    assert 'emt' in out, out.get('message')
    (cable,) = out['emt']['cable_sheaths']
    assert cable['model'] == 'wideband' and cable['bonding'] == 'cross'
    assert min(cable['core_i_a']) > 400.0
    assert max(cable['sheath_i_a']) < 0.05 * min(cable['core_i_a'])
