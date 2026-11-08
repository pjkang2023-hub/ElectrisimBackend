"""
Cable sheaths (emt_cable) against IEC 60287's and the textbooks' answers at power frequency: the standing voltage
of a single-point bonded sheath, I w u0 / 2 pi ln(S / r_m) per metre in trefoil; the circulating current of a solidly
bonded one, X_m / sqrt(R_s^2 + X_m^2) of the core's; and a cross-bonded one's currents cancelling, its joints at a
minor section's standing voltage.
"""
import cmath
import math

import numpy as np
import pytest

import emt_cable
import emt_solver as es

F1, VLL, KM = 50.0, 33e3, 3.0      # 33 kV: the cable's own charging current, which the hand formulas leave out, stays small
W0 = 2.0 * math.pi * F1
G = {'core_radius_m': 0.0175, 'core_rdc_ohm_per_km': 0.0283, 'insulation_radius_m': 0.040, 'sheath_thickness_m': 0.0025,
     'sheath_rdc_ohm_per_km': 0.10, 'eps_r': 2.5, 'spacing_m': 0.10, 'depth_m': 1.2, 'jacket_radius_m': 0.048, 'rho_ohm_m': 100.0}
I_RMS = 600.0


def _run(bonding, sections=4):
    """A stiff 33 kV source through the cable into a resistive load drawing about 600 A."""
    ckt = es.Circuit()
    a = [ckt.node(f'a{p}') for p in 'abc']
    b = [ckt.node(f'b{p}') for p in 'abc']
    v_pk = math.sqrt(2.0 / 3.0) * VLL
    for k in range(3):
        ckt.add_rl(0, a[k], 0.01, 1e-4, ac=(v_pk, W0, -k * 2.0 * math.pi / 3.0))
        ckt.add_r(b[k], 0, VLL / math.sqrt(3.0) / I_RMS)
    cable = emt_cable.CableSheaths(ckt, a, b, G, KM, F1, 'C', bonding=bonding, sections=sections)
    ckt.start_in_ac_steady_state(W0)
    sim = ckt.simulate(0.06, 2e-5)
    return cable, sim


def _phasor(t, y):
    sel = t > t[-1] - 1.0 / F1
    return 2.0 * np.mean(y[sel] * np.exp(-1j * W0 * t[sel]))


def _core_and_sheath_currents(cable, sim, group=0):
    t = sim['t']
    i = sim['i_cp'][cable.groups[group]]
    return [_phasor(t, i[:, k]) for k in range(6)], t


def _r_m():
    return G['insulation_radius_m'] + G['sheath_thickness_m'] / 2.0


def test_a_single_point_bonded_sheath_stands_at_its_open_end():
    cable, sim = _run('single')
    t = sim['t']
    (ia, ib, ic, *_), _ = _core_and_sheath_currents(cable, sim)
    assert abs(ia) / math.sqrt(2.0) == pytest.approx(I_RMS, rel=0.05)
    # In trefoil each sheath sees the other two cores at S and its own at r_m: I w u0 / 2 pi ln(S / r_m) per metre.
    per_m = abs(ia) * W0 * emt_cable.U0 / (2.0 * math.pi) * math.log(G['spacing_m'] / _r_m())
    for p in range(3):
        v = abs(_phasor(t, sim['v'][:, cable.sheath_end[p]]))
        assert v == pytest.approx(per_m * KM * 1e3, rel=0.03), p


def test_a_solidly_bonded_sheath_carries_its_circulating_current():
    cable, sim = _run('solid')
    currents, _ = _core_and_sheath_currents(cable, sim)
    xm = W0 * emt_cable.U0 / (2.0 * math.pi) * math.log(G['spacing_m'] / _r_m())
    rs = G['sheath_rdc_ohm_per_km'] * 1e-3
    hand = xm / math.hypot(rs, xm)
    for p in range(3):
        assert abs(currents[3 + p]) / abs(currents[p]) == pytest.approx(hand, rel=0.03), p


def test_cross_bonding_cancels_the_sheath_currents():
    solid, sim_s = _run('solid')
    cross, sim_c = _run('cross', sections=2)
    i_solid = max(abs(x) for x in _core_and_sheath_currents(solid, sim_s)[0][3:])
    currents, t = _core_and_sheath_currents(cross, sim_c)
    assert max(abs(x) for x in currents[3:]) < 0.05 * i_solid
    # Its first joint stands at a minor section's standing voltage.
    per_m = abs(currents[0]) * W0 * emt_cable.U0 / (2.0 * math.pi) * math.log(G['spacing_m'] / _r_m())
    joint = max(abs(_phasor(t, sim_c['v'][:, n])) for n in cross.joints[1])
    assert joint == pytest.approx(per_m * KM * 1e3 / 3.0, rel=0.1)


def test_through_the_study_a_cross_bonded_cable_reports_its_sheaths(client, quiet):
    """A 33 kV cable row with its bonding: the study runs it with its sheaths and reports them, the sheath currents cancelled."""
    import json
    from test_pcs import GRID, LINE, LOAD, _bus
    r, l, c_ins, _ = emt_cable.matrices(G, F1)
    z1 = complex(r[0, 0] - r[0, 1], W0 * (l[0, 0] - l[0, 1])) * 1e3
    line = {**LINE, 'length_km': str(KM), 'r_ohm_per_km': str(z1.real), 'x_ohm_per_km': str(z1.imag), 'c_nf_per_km': str(c_ins * 1e12), 'type': 'cs',
            'max_i_ka': '1', 'fd_line': 'true', 'fd_kind': 'cable', 'fd_bonding': 'cross', 'fd_sections': '2', **{f'fd_{k}': str(v) for k, v in G.items()}}
    study = {'typ': 'EmtStudy Parameters', 'user_email': 't@t', 'frequency': str(F1), 'time_step_us': '20', 'duration_ms': '60'}
    request = {'0': study, '1': _bus('a', 33), '2': _bus('b', 33), '3': line, '4': {**GRID, 'bus': 'a', 's_sc_max_mva': '2000', 's_sc_min_mva': '2000'},
               '5': {**LOAD, 'bus': 'b', 'p_mw': '30', 'q_mvar': '3', 'const_z_percent': '100', 'const_i_percent': '0'}}
    with quiet():
        out = json.loads(client.post('/', json=request).get_data(as_text=True))
    assert 'emt' in out, out.get('message')
    (cable,) = out['emt']['cable_sheaths']
    assert cable['bonding'] == 'cross'
    assert min(cable['core_i_a']) > 400.0
    assert max(cable['sheath_i_a']) < 0.05 * min(cable['core_i_a'])
    assert 0.0 < cable['sheath_v_standing_kv'] < 0.2
