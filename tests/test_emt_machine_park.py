"""
The detailed synchronous machine (emt_machine.ParkMachine) against what has an
answer on paper: its three-phase short circuit's decay and DC offset, its
sudden load rejection on each axis (IEC 60034-4's tests for the same constants),
its field voltage from the phasor diagram, and the torque of the short circuit;
then ANDES's GENROU on a load step and a fault in a run through the application.
"""
import cmath
import json
import math

import numpy as np
import pytest
from scipy.linalg import expm

import emt_machine
import emt_solver as es

F = 50.0
W0 = 2.0 * math.pi * F
VLL, S = 11e3, 100e6
A120 = 2.0 * math.pi / 3.0
E_B = math.sqrt(2.0 / 3.0) * VLL
I_B = math.sqrt(2.0) * S / (math.sqrt(3.0) * VLL)
DATA = dict(xd_pu=1.8, xq_pu=1.7, xd1_pu=0.3, xq1_pu=0.55, xd2_pu=0.25, xl_pu=0.15, td10_s=2.0, td20_s=0.03,
            tq10_s=0.4, tq20_s=0.05, ra_pu=0.0025)


def _space(x):
    return (2.0 / 3.0) * (x[:, 0] + x[:, 1] * cmath.exp(1j * A120) + x[:, 2] * cmath.exp(-1j * A120))


def _cycle_mean(t, y):
    n = int(round(1.0 / F / (t[1] - t[0])))
    c = np.concatenate(([0.0], np.cumsum(y)))
    return t[n - 1:] - 0.5 / F, (c[n:] - c[:-n]) / n


def _short(dt=1e-4, t_end=2.0, h=1e4, **over):
    """The machine open-circuit at 1 pu, its terminals shorted at 20 ms."""
    ckt = es.Circuit()
    nodes = [ckt.node('abc'[k]) for k in range(3)]
    for n in nodes:
        ckt.add_r(n, 0, 1e6)
    m = emt_machine.ParkMachine(ckt, 'G', nodes, W0, complex(E_B, 0.0), 0j, S, VLL, h=h, **{**DATA, **over})
    ckt.add_controller(m.control)
    for n in nodes:
        sw = ckt.add_switch(n, 0, closed=False)
        ckt.at(0.02, lambda st, sw=sw: st.set_switch(sw, True))
    ckt.start_in_ac_steady_state(W0)
    sim = ckt.simulate(t_end, dt)
    i = np.array([sim['i_rl'][:, k] for k in m.k_e]).T
    return m, sim['t'], i / I_B


def _d_axis_after_short(t, x=DATA):
    """Its d axis with its terminals shorted, psi''d = x'' id, exactly: its field current from the matrix."""
    x2, x1, xd, xl = x['xd2_pu'], x['xd1_pu'], x['xd_pu'], x['xl_pu']
    g1, g2 = (x2 - xl) / (x1 - xl), (x1 - x2) / (x1 - xl) ** 2
    a = np.zeros((2, 2))
    # id = (g1 psi' + (1 - g1) psi1) / x''
    k1, k2 = g1 / x2, (1.0 - g1) / x2
    a[0] = [(-1.0 - (xd - x1) * (g1 * k1 + g2)) / x['td10_s'], (-(xd - x1) * (g1 * k2 - g2)) / x['td10_s']]
    a[1] = [(1.0 - (x1 - xl) * k1) / x['td20_s'], (-1.0 - (x1 - xl) * k2) / x['td20_s']]
    b = np.array([1.0 / x['td10_s'], 0.0])     # Efd = 1 pu, open-circuit
    steady = -np.linalg.solve(a, b)
    s0 = np.array([1.0, 1.0])
    states = [steady + expm(a * u) @ (s0 - steady) for u in t]
    return np.array([k1 * x_[0] + k2 * x_[1] for x_ in states])


def test_short_circuit_current_decays_as_the_textbook_says():
    """E/x'' + E/x' + E/xd with T'd and T''d: the fundamental, the DC offset's decay, and the current it ends at."""
    m, t, i = _short()
    i_s = _space(i)
    rot = i_s * np.exp(-1j * W0 * t)
    tm, fund = _cycle_mean(t, rot)
    fund = np.abs(fund)
    x2, x1, xd = DATA['xd2_pu'], DATA['xd1_pu'], DATA['xd_pu']
    t1, t2 = DATA['td10_s'] * x1 / xd, DATA['td20_s'] * x2 / x1
    paper = lambda u: 1.0 / xd + (1.0 / x1 - 1.0 / xd) * np.exp(-u / t1) + (1.0 / x2 - 1.0 / x1) * np.exp(-u / t2)
    # The same from the rotor's own matrix, exactly.
    g = np.arange(0.0, 2.0, 2e-3)
    et, em = _cycle_mean(g, _d_axis_after_short(g))
    for u in (0.04, 0.06, 0.1, 0.2, 0.5, 1.0, 1.9):
        got = float(np.interp(u + 0.02, tm, fund))
        # The cycle's average of the textbook's curve: its fast term shrinks through the window.
        window = np.linspace(u - 0.01, u + 0.01, 41)
        assert got == pytest.approx(float(np.mean(paper(window))), rel=0.03), u
        assert got == pytest.approx(float(np.interp(u, et, em)), rel=0.02), u
    # The offset: E/x'' at the start of its decay, at x'' / (w ra).
    tm, dc = _cycle_mean(t, i_s)
    dc = np.abs(dc)
    ta = x2 / (W0 * DATA['ra_pu'])
    a, b = float(np.interp(0.05, tm, dc)), float(np.interp(0.35, tm, dc))
    assert 0.3 / math.log(a / b) == pytest.approx(ta, rel=0.05)
    assert a * math.exp(0.03 / ta) == pytest.approx(1.0 / x2, rel=0.05)


def test_the_short_circuits_torque():
    """A pulsation at the fundamental of E^2 / x'': the rotor's speed swings by that over 2H w0."""
    m, t, _ = _short(h=1.0, t_end=0.3)
    tr = np.array(m.trace)
    w = tr[:, 3] / F
    sel = (tr[:, 0] > 0.02) & (tr[:, 0] < 0.05)
    swing = (w[sel].max() - w[sel].min())
    assert swing == pytest.approx(2.0 * (1.0 / DATA['xd2_pu']) / (2.0 * 1.0 * W0), rel=0.15)


@pytest.mark.parametrize('r_pu', [0.04, 0.08])
def test_the_dc_offsets_brake_on_the_rotor(r_pu):
    """
    Short through a resistance, the stator's flux stays put a while and the
    rotor's turns against it: Te = (psi'' psi_s / x'') sin(w0 t), decaying with
    the flux. An RMS model has only R I_ac^2. Its integral, E^2 / (x'' w0): half
    spent in the resistances as the offset dies, half stored in x'' as the AC
    current builds. The rotor pays that whatever R is, in a run the speed shows.
    """
    t_f, dur, h, dt = 0.02, 0.2, 3.0, 5e-5
    ckt = es.Circuit()
    nodes = [ckt.node('abc'[k]) for k in range(3)]
    for n in nodes:
        ckt.add_r(n, 0, 1e6)
    m = emt_machine.ParkMachine(ckt, 'G', nodes, W0, complex(E_B, 0.0), 0j, S, VLL, h=h, **{**DATA, 'td10_s': 8.0})
    ckt.add_controller(m.control)
    for n in nodes:
        mid = ckt.node('fault')
        ckt.add_r(mid, 0, r_pu * VLL ** 2 / S)
        sw = ckt.add_switch(n, mid, closed=False)
        ckt.at(t_f, lambda st, sw=sw: st.set_switch(sw, True))
    ckt.start_in_ac_steady_state(W0)
    sim = ckt.simulate(t_f + dur, dt)
    t = sim['t']
    i = np.array([sim['i_rl'][:, k] for k in m.k_e]).T / I_B
    tm, fund = _cycle_mean(t, _space(i) * np.exp(-1j * W0 * t))
    keep = (tm >= t_f + 0.01) & (tm <= t_f + dur - 0.01)
    r = r_pu + m.ra
    i_start = float(np.abs(fund[keep][0]))
    rms_cost = float(np.trapezoid(r * np.abs(fund[keep]) ** 2, tm[keep])) + r * i_start ** 2 * 0.01
    tr = np.array(m.trace)
    paid = (1.0 - float(np.mean(tr[tr[:, 0] > t_f + dur - 0.02, 3])) / F) * 2.0 * h
    assert paid - rms_cost == pytest.approx(1.0 / (DATA['xd2_pu'] * W0), rel=0.2)


def _reject(axis, dt=5e-5, t_end=0.6):
    """The machine carrying 0.5 pu all on one axis, its load removed at 20 ms; its voltage on its rotor's axes."""
    ckt = es.Circuit()
    nodes = [ckt.node('abc'[k]) for k in range(3)]
    r_stay = 1e3    # some 800 pu: its current after the rejection is negligible, and the solver's ring dies in steps
    for n in nodes:
        ckt.add_r(n, 0, r_stay)
    i_d, i_q = (0.5, 0.0) if axis == 'd' else (0.0, 0.5)
    if axis == 'd':
        v_d, v_q = 0.0, 1.0
    else:
        v_d, v_q = DATA['xq_pu'] * i_q, 0.6
    v_d, v_q = v_d - DATA['ra_pu'] * i_d, v_q - DATA['ra_pu'] * i_q
    # delta = 0: the q axis on phase a's reference, the d axis a quarter period behind.
    i_ph = complex(i_d, i_q) * (-1j) * I_B
    v_ph = complex(v_d, v_q) * (-1j) * E_B
    m = emt_machine.ParkMachine(ckt, 'G', nodes, W0, v_ph, i_ph, S, VLL, h=1e4, **DATA)
    assert m.delta == pytest.approx(0.0, abs=1e-9)
    ckt.add_controller(m.control)
    # Its load: a current source taking what it delivers.
    i_src = i_ph - v_ph / r_stay
    ks = [ckt.add_isrc(n, 0, ac=(abs(i_src), W0, float(np.angle(i_src)) - k * A120)) for k, n in enumerate(nodes)]
    for k in ks:
        ckt.at(0.02, lambda st, k=k: st.set_source(k, False))
    ckt.start_in_ac_steady_state(W0)
    sim = ckt.simulate(t_end, dt)
    v = _space(sim['v'][:, nodes]) / E_B
    t = sim['t']
    in_rotor = v * np.exp(-1j * (W0 * t - 0.5 * math.pi))
    # Two samples' mean: the trapezoid's ring at the step's own period, when the current stops at once.
    return m, t[:-1], 0.5 * (in_rotor[:-1] + in_rotor[1:])


def _open_circuit(axis):
    """The rotor's two circuits with no current, exactly: psi'' from the matrix."""
    x = DATA
    x1, xs, t1, t2 = (x['xd1_pu'], x['xd_pu'], x['td10_s'], x['td20_s']) if axis == 'd' else (x['xq1_pu'], x['xq_pu'], x['tq10_s'], x['tq20_s'])
    x2, xl = x['xd2_pu'], x['xl_pu']
    g1, g2 = (x2 - xl) / (x1 - xl), (x1 - x2) / (x1 - xl) ** 2
    a = np.array([[(-1.0 - (xs - x1) * g2) / t1, (xs - x1) * g2 / t1], [1.0 / t2, -1.0 / t2]])
    return a, g1


@pytest.mark.parametrize('axis', ['d', 'q'])
def test_a_sudden_load_rejection_shows_each_axis_time_constants(axis):
    """The load all on one axis, removed at once: the other's voltage steps and recovers over T'' and T'."""
    m, t, v = _reject(axis)
    i0 = 0.5
    x = DATA
    x1, xs, t1, t2 = (x['xd1_pu'], x['xd_pu'], x['td10_s'], x['td20_s']) if axis == 'd' else (x['xq1_pu'], x['xq_pu'], x['tq10_s'], x['tq20_s'])
    x2, xl = x['xd2_pu'], x['xl_pu']
    a, g1 = _open_circuit(axis)
    # Its fluxes before: psi' = psi'' + (x' - x'') i, psi1 = psi' - (x' - xl) i, measured from where they settle.
    p1 = -(xs - x1) * i0
    p2 = p1 - (x1 - xl) * i0
    s0 = np.array([p1, p2])
    for u in (0.005, 0.01, 0.03, 0.08, 0.2, 0.5):
        s = expm(a * u) @ s0
        dev = g1 * s[0] + (1.0 - g1) * s[1]
        got = float(np.interp(0.02 + u, t, v.imag if axis == 'd' else v.real))
        if axis == 'd':
            assert got - m.efd0 == pytest.approx(dev, abs=2e-3), u
        else:
            assert got == pytest.approx(-dev, abs=2e-3), u
        # The test's textbook reading, two exponentials with the open-circuit constants, needs them far apart: d only.
        if axis == 'd':
            paper = (xs - x1) * i0 * math.exp(-u / t1) + (x1 - x2) * i0 * math.exp(-u / t2)
            assert abs(dev) == pytest.approx(paper, abs=0.02), u


def test_its_field_voltage_and_angle_are_the_phasor_diagrams():
    """V + (ra + j xq) I on the q axis; Efd = |E_q| + (xd - xq) id. Salient: xq well under xd, 0.85 pf lagging."""
    ckt = es.Circuit()
    nodes = [ckt.node('abc'[k]) for k in range(3)]
    i_ph = cmath.rect(0.8, -math.acos(0.85)) * I_B
    over = {**DATA, 'xq_pu': 1.0, 'xq1_pu': 0.6}
    m = emt_machine.ParkMachine(ckt, 'G', nodes, W0, complex(E_B, 0.0), i_ph, S, VLL, h=3.0, **over)
    i = i_ph / I_B
    e_q = 1.0 + complex(over['ra_pu'], over['xq_pu']) * i
    assert m.delta == pytest.approx(cmath.phase(e_q), abs=1e-9)
    i_d = -(i * cmath.exp(-1j * m.delta)).imag
    assert m.efd0 == pytest.approx(abs(e_q) + (over['xd_pu'] - over['xq_pu']) * i_d, rel=1e-9)
    assert m.pm == pytest.approx((1.0 * np.conj(i)).real + over['ra_pu'] * abs(i) ** 2, rel=1e-9)


def test_saturation_raises_the_field_it_needs_as_genrou_does():
    """Efd = (1 + Se) psi''d + (xd - x'') id, Se through S(1.0) and S(1.2)."""
    ckt = es.Circuit()
    nodes = [ckt.node('abc'[k]) for k in range(3)]
    m = emt_machine.ParkMachine(ckt, 'G', nodes, W0, complex(1.05 * E_B, 0.0), 0j, S, VLL, s10=0.05, s12=0.3, **DATA)
    assert m._se(1.0) == pytest.approx(0.05, rel=1e-9) and m._se(1.2) == pytest.approx(0.3, rel=1e-9)
    assert m.efd0 == pytest.approx(1.05 * (1.0 + m._se(1.05)), rel=1e-9)


def test_a_smaller_step_does_not_move_it():
    a = _short(dt=2e-4, t_end=0.2)
    b = _short(dt=5e-5, t_end=0.2)
    x = float(np.max(np.abs(a[2][:, 0])))
    y = float(np.max(np.abs(b[2][:, 0])))
    assert x == pytest.approx(y, rel=0.01)


def test_its_data_are_put_in_order_and_say_so():
    from andes_electrisim import _DEFAULT_GENROU, _EXCITER_DEFAULTS, _GOVERNOR_DEFAULTS
    data, notes = emt_machine.park_data({'dyn_xd2': '0.4', 'dyn_xd1': '0.3', 'dyn_xl': '0.5'}, _GOVERNOR_DEFAULTS, _EXCITER_DEFAULTS, _DEFAULT_GENROU)
    assert data['xd2_pu'] < data['xd1_pu'] <= data['xd_pu'] and data['xl_pu'] < data['xd2_pu']
    assert len(notes) == 2
    data, notes = emt_machine.park_data({'xdss_pu': '0.2'}, _GOVERNOR_DEFAULTS, _EXCITER_DEFAULTS, _DEFAULT_GENROU)
    assert data['xd2_pu'] == 0.2 and not notes and data['h'] == _DEFAULT_GENROU['M'] / 2.0


# --- Through the application, against ANDES ------------------------------------------------

KV, MVA, P_MW, Q_MVAR = 11.0, 100.0, 50.0, 20.0
GEN = {'dyn_H': '3', 'dyn_D': '0', 'dyn_xd': '1.8', 'dyn_xq': '1.7', 'dyn_xd1': '0.3', 'dyn_xq1': '0.55', 'dyn_xd2': '0.25', 'dyn_xl': '0.15',
       'dyn_Td10': '8', 'dyn_Td20': '0.03', 'dyn_Tq10': '0.4', 'dyn_Tq20': '0.05', 'dyn_ra': '0.0025', 'dyn_governor_model': 'NONE', 'dyn_exciter_model': 'NONE'}


def _request(**params):
    from test_pcs import _bus
    gen = {'typ': 'Generator0', 'name': 'g', 'id': 'cell-g', 'userFriendlyName': 'G', 'bus': 'a', 'p_mw': str(P_MW), 'vm_pu': '1.0', 'sn_mva': str(MVA),
           'scaling': '1', 'slack': 'true', 'in_service': 'true', 'controllable': 'true', 'min_p_mw': '', 'max_p_mw': '', 'vn_kv': str(KV), 'xdss_pu': '0.25',
           'rdss_ohm': '', 'cos_phi': '0.8', 'pg_percent': '', 'power_station_trafo': '', **GEN}
    load = {'typ': 'Load0', 'name': 'ld', 'id': 'cell-ld', 'userFriendlyName': 'LD', 'bus': 'a', 'p_mw': str(P_MW), 'q_mvar': str(Q_MVAR), 'const_z_percent': '100',
            'const_i_percent': '0', 'sn_mva': '0', 'scaling': '1', 'type': 'wye', 'in_service': 'true'}
    study = {'typ': 'EmtStudy Parameters', 'user_email': 't@t', 'generator_model': 'detailed', 'time_step_us': '100', **params}
    return {'0': {k: str(v) for k, v in study.items()}, '1': _bus('a', KV), '2': gen, '3': load}


def _emt(client, quiet, request):
    with quiet():
        out = json.loads(client.post('/', json=request).get_data(as_text=True))
    assert 'emt' in out, out.get('message')
    return out['emt'], out.get('warnings', [])


def test_it_starts_where_the_load_flow_left_it(client, quiet):
    emt, _ = _emt(client, quiet, _request(duration_ms=200))
    (m,) = emt['machines']
    assert m['model'] == 'detailed'
    assert m['p_start_mw'] == pytest.approx(P_MW, rel=0.01) and m['p_end_mw'] == pytest.approx(P_MW, rel=1e-3)
    assert m['pm_end_mw'] == pytest.approx(P_MW + 0.0025 * (P_MW ** 2 + Q_MVAR ** 2) / MVA, rel=1e-3)
    assert m['f_end_hz'] == pytest.approx(F, abs=2e-3)
    assert m['efd_end_pu'] == pytest.approx(m['efd_start_pu'], rel=1e-9)


def test_the_classical_model_stays_the_default(client, quiet):
    request = _request(duration_ms=50)
    request['0']['generator_model'] = 'machine'
    emt, _ = _emt(client, quiet, request)
    assert 'model' not in emt['machines'][0]


def _andes(step_mw, t_step, t_end):
    import andes
    from andes_electrisim import tds_values
    andes.config_logger(stream_level=40)
    ss = andes.System()
    ss.config.freq, ss.config.mva = F, 100
    ss.files.no_output = True
    zb = KV * KV / 100
    ss.add('Bus', idx=1, name='A', Vn=KV)
    ss.add('Bus', idx=2, name='B', Vn=KV)
    ss.add('Line', idx='L', bus1=1, bus2=2, Vn1=KV, Vn2=KV, r=1e-4 / zb, x=1e-4 / zb, fn=F)
    ss.add('Slack', idx='GT', bus=1, Vn=KV, Sn=MVA, v0=1.0, a0=0, p0=P_MW / 100)
    ss.add('GENROU', idx='M', bus=1, gen='GT', Sn=MVA, Vn=KV, fn=F, M=6.0, D=0, ra=0.0025, xl=0.15, xd=1.8, xq=1.7, xd1=0.3, xq1=0.55, xd2=0.25, xq2=0.25,
           Td10=8.0, Td20=0.03, Tq10=0.4, Tq20=0.05)
    ss.add('PQ', idx='LD', bus=2, Vn=KV, p0=P_MW / 100, q0=Q_MVAR / 100)
    ss.add('Fault', idx='F', bus=2, tf=t_step, tc=t_end + 10.0, rf=(KV * KV / step_mw) / zb, xf=1e-8)
    ss.PQ.config.p2p, ss.PQ.config.p2i, ss.PQ.config.p2z = 0, 0, 1
    ss.PQ.config.q2q, ss.PQ.config.q2i, ss.PQ.config.q2z = 0, 0, 1
    ss.setup()
    ss.PFlow.run()
    ss.TDS.config.tf, ss.TDS.config.tstep, ss.TDS.config.no_tqdm = t_end, 0.002, 1
    ss.TDS.run()
    return np.asarray(ss.dae.ts.t), F * tds_values(ss, ss.GENROU.omega)[:, 0], tds_values(ss, ss.Bus.v)[:, 0]


def test_a_load_step_against_andess_genrou(client, quiet, monkeypatch):
    """
    10 MW more, resistive, with neither governor nor exciter. ANDES's GENROU
    leaves out its voltage's proportion to speed: without it here too, the
    speed stays within 5 mHz of ANDES's for 1.5 s and a 1.3 Hz fall. With it,
    the voltage falls with the speed, the load with the voltage, and the speed
    falls less: that is the difference between them.
    """
    t_step, t_end, step = 0.2, 1.7, 10.0
    request = _request(duration_ms=t_end * 1e3, ac_fault_bus='a', ac_fault_type='abc', ac_fault_time_ms=t_step * 1e3, ac_fault_duration_ms=0,
                       ac_fault_resistance_ohm=KV ** 2 / step)
    t_a, f_a, v_a = _andes(step, t_step, t_end)
    assert f_a[-1] < F - 1.0               # a real fall

    def speed(x, m):
        return float(np.interp(x, np.array(m['trace']['t_ms']) / 1e3, np.array(m['trace']['f_hz'])))

    (with_it,) = _emt(client, quiet, request)[0]['machines']
    monkeypatch.setattr(emt_machine.ParkMachine, 'speed_voltage', False)
    (without,) = _emt(client, quiet, request)[0]['machines']
    for x in (0.3, 0.5, 0.8, 1.2, 1.65):
        assert speed(x, without) == pytest.approx(np.interp(x, t_a, f_a), abs=0.005), x
    assert speed(1.65, with_it) - speed(1.65, without) > 0.05


def _on_a_grid(**params):
    from test_pcs import GRID, LINE, _bus
    request = _request(**params)
    gen = next(v for v in request.values() if v.get('typ') == 'Generator0')
    gen.update(slack='false')
    load = next(k for k, v in request.items() if v.get('typ') == 'Load0')
    del request[load]
    line = {**LINE, 'length_km': '1', 'r_ohm_per_km': '0.01', 'x_ohm_per_km': str(X_LINE)}
    grid = {**GRID, 'bus': 'b', 'vm_pu': '1.0', 's_sc_max_mva': '1e6', 'rx_max': '0.1'}
    request.update({'3': _bus('b', KV), '4': line, '5': grid})
    return request


X_LINE = 0.4


def _andes_fault(t_fault, t_clear, t_end):
    import andes
    from andes_electrisim import tds_values
    andes.config_logger(stream_level=40)
    ss = andes.System()
    ss.config.freq, ss.config.mva = F, 100
    ss.files.no_output = True
    zb = KV * KV / 100
    ss.add('Bus', idx=1, name='A', Vn=KV)
    ss.add('Bus', idx=2, name='B', Vn=KV)
    ss.add('Line', idx='L', bus1=1, bus2=2, Vn1=KV, Vn2=KV, r=0.01 / zb, x=X_LINE / zb, fn=F)
    ss.add('PV', idx='GT', bus=1, Vn=KV, Sn=MVA, v0=1.0, p0=P_MW / 100)
    ss.add('Slack', idx='S', bus=2, Vn=KV, v0=1.0, a0=0)
    ss.add('GENROU', idx='M', bus=1, gen='GT', Sn=MVA, Vn=KV, fn=F, M=6.0, D=0, ra=0.0025, xl=0.15, xd=1.8, xq=1.7, xd1=0.3, xq1=0.55, xd2=0.25, xq2=0.25,
           Td10=8.0, Td20=0.03, Tq10=0.4, Tq20=0.05)
    ss.add('Fault', idx='F', bus=1, tf=t_fault, tc=t_clear, rf=0.05 / zb, xf=1e-6)
    ss.setup()
    ss.PFlow.run()
    ss.TDS.config.tf, ss.TDS.config.tstep, ss.TDS.config.no_tqdm = t_end, 0.002, 1
    ss.TDS.run()
    return np.asarray(ss.dae.ts.t), F * tds_values(ss, ss.GENROU.omega)[:, 0]


def test_a_fault_and_its_swing_against_andess_genrou(client, quiet, monkeypatch):
    """
    100 ms at its terminals, then its first swings on a stiff grid. The swing's
    period is ANDES's. The rotor ends the fault slower than ANDES's by the
    offset's brake (the test above), less because the grid holds up some of its
    bus's voltage, so the stator's flux falls by less; and the swing is larger.
    """
    monkeypatch.setattr(emt_machine.ParkMachine, 'speed_voltage', False)
    t_f, t_c, t_end = 0.2, 0.3, 2.5
    emt, _ = _emt(client, quiet, _on_a_grid(duration_ms=t_end * 1e3, ac_fault_bus='a', ac_fault_type='abc', ac_fault_time_ms=t_f * 1e3, ac_fault_duration_ms=(t_c - t_f) * 1e3,
                                            ac_fault_resistance_ohm=0.05))
    (m,) = emt['machines']
    t_e, f_e = np.array(m['trace']['t_ms']) / 1e3, np.array(m['trace']['f_hz'])
    t_a, f_a = _andes_fault(t_f, t_c, t_end)
    window = lambda t, f: float(np.mean(f[(t > t_c - 0.02) & (t <= t_c)]))
    brake_hz = F * (1.0 / (0.25 * W0)) / (2.0 * 3.0)
    assert 0.5 * brake_hz < window(t_a, f_a) - window(t_e, f_e) < brake_hz
    assert t_e[np.argmax(f_e)] == pytest.approx(t_a[np.argmax(f_a)], abs=0.03)
    assert f_e.max() > f_a.max()
