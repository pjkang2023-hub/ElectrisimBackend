"""
The EMT study's three-phase AC network: coupled branches against the exact
single-line-to-ground current; each vector group's phase shift; an 800 V-free
20 kV benchmark against Simscape Electrical's own three-phase blocks
(tests/emt_reference: simscape_emt_ac_benchmark.m builds it from
emt_ac_benchmark_params.json, which these tests build Electrisim's circuit
from); and the study on the drawn radial grid.
"""
import json
import math
import os
import warnings

import numpy as np
import pandapower as pp
import pytest

import emt_ac
import emt_solver as es
from test_emt import _request

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'emt_reference')
with open(os.path.join(HERE, 'emt_ac_benchmark_params.json'), encoding='utf-8') as _h:
    P = json.load(_h)
W = 2 * math.pi * P['f_hz']


# --- Exact answers -------------------------------------------------------------------

def test_single_line_to_ground_through_a_coupled_source():
    """A grid whose Z0 differs from its Z1 drives 3 E / (2 Z1 + Z0) into a bolted phase-a fault."""
    e = 20e3 * math.sqrt(2 / 3)
    r, l, m = 0.5, 0.01, 0.004
    ckt = es.Circuit()
    ph = [ckt.node(f'p{k}') for k in range(3)]
    g = ckt.add_coupled([0, 0, 0], ph, np.eye(3) * r, np.full((3, 3), m) + np.eye(3) * (l - m),
                        ac=(np.full(3, e), W, -np.arange(3) * 2 * math.pi / 3))
    ckt.add_r(ph[0], 0, 1e-3)
    ckt.add_r(ph[1], 0, 1e6)
    ckt.add_r(ph[2], 0, 1e6)
    ckt.start_in_ac_steady_state(W)
    sim = ckt.simulate(0.04, 2e-5)
    i = sim['i_cp'][g][:, 0]
    z1, z0 = complex(r, W * (l - m)), complex(r, W * (l + 2 * m))
    exact = abs(3 * e / (z0 + 2 * z1 + 3e-3))
    assert np.max(np.abs(i)) == pytest.approx(exact, rel=1e-4)
    # Started in its steady state: the first cycle's peak is the last's.
    t = sim['t']
    assert np.max(np.abs(i[t < 0.02])) == pytest.approx(np.max(np.abs(i[t >= 0.02])), rel=1e-4)


@pytest.mark.parametrize('vector_group, lead_deg', [('Dyn11', 30), ('Dyn1', -30), ('YNd11', 30), ('YNd1', -30),
                                                    ('Dyn5', -150), ('Dyn7', 150), ('YNyn0', 0), ('Dd0', 0)])
def test_vector_group_phase_shift(vector_group, lead_deg):
    """The LV side leads the HV by the clock's angle (less the load's small drop); its voltage is the load flow's."""
    net = pp.create_empty_network(f_hz=50)
    hv, lv = pp.create_bus(net, 20), pp.create_bus(net, 0.4)
    pp.create_ext_grid(net, hv, s_sc_max_mva=500, rx_max=0.1, x0x_max=1.0, r0x0_max=0.1)
    pp.create_transformer_from_parameters(net, hv, lv, 1.0, 20, 0.4, 1.0, 6.0, 1.0, 0.1,
                                          vector_group=vector_group, shift_degree=0)
    pp.create_load(net, lv, 0.5, 0.1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pp.runpp(net)
    ckt = es.Circuit()
    ac = emt_ac.AcBuilder(ckt, net, {}, [], 50.0).build()
    v = ckt.start_in_ac_steady_state(W)
    lead = math.degrees(np.angle(v[ac.nodes[lv][0]]) - np.angle(v[ac.nodes[hv][0]]))
    drop = net.res_bus.va_degree[lv] - net.res_bus.va_degree[hv]
    assert (lead - drop - lead_deg + 180) % 360 - 180 == pytest.approx(0, abs=0.05)
    assert abs(v[ac.nodes[lv][0]]) / (math.sqrt(2 / 3) * 400) == pytest.approx(net.res_bus.vm_pu[lv], rel=1e-4)


# --- Against Simscape ---------------------------------------------------------------

def _benchmark(case):
    """The circuit of emt_ac_benchmark_params.json, from rest, as Simscape's blocks are."""
    c = P['cases'][case]
    ckt = es.Circuit()
    src = [ckt.node(f'source {x}') for x in 'abc']
    z = P['v_ll'] ** 2 / P['s_sc']
    r = z / math.sqrt(1 + P['xr'] ** 2)
    x = P['xr'] * r
    g_src = ckt.add_coupled([0, 0, 0], src, np.eye(3) * r, np.eye(3) * x / W,
                            ac=(np.full(3, math.sqrt(2 / 3) * P['v_ll']), W,
                                math.radians(P['phase_deg']) - np.arange(3) * 2 * math.pi / 3))
    q = P['line']
    n, sec = q['sections'], q['km'] / q['sections']
    ends = [src] + [[ckt.node(f'line {k} {x}') for x in 'abc'] for k in range(1, n + 1)]
    for k in range(n):
        ckt.add_coupled(ends[k], ends[k + 1], np.eye(3) * q['r_per_km'] * sec,
                        (np.full((3, 3), q['m_per_km']) + np.eye(3) * (q['l_per_km'] - q['m_per_km'])) * sec)
        for end in (ends[k], ends[k + 1]):
            for j in range(3):
                ckt.add_c(0, end[j], q['cg_per_km'] * sec / 2)
                ckt.add_c(end[j], end[(j + 1) % 3], q['cl_per_km'] * sec / 2)
    hv = ends[-1]
    t = P['trafo']
    lv = [ckt.node(f'lv {x}') for x in 'abc']
    vw1, vw2, s_ph = t['v1'], t['v2'] / math.sqrt(3), t['s'] / 3
    zb1, zb2 = vw1 ** 2 / s_ph, vw2 ** 2 / s_ph
    ratio, lm1 = vw2 / vw1, t['xm_pu'] * vw1 ** 2 / s_ph / W
    r_m = np.diag([t['rw_pu'] * zb1, t['rw_pu'] * zb2])
    l_m = np.array([[lm1 + t['xl_pu'] * zb1 / W, ratio * lm1],
                    [ratio * lm1, ratio * ratio * lm1 + t['xl_pu'] * zb2 / W]])
    for hp, lp, rev in emt_ac.winding_pairs('D', 'Y', t['clock']):
        a2, b2 = (0, lv[lp[0]]) if rev else (lv[lp[0]], 0)
        ckt.add_coupled([hv[hp[0]], a2], [hv[hp[1]], b2], r_m, l_m)
        ckt.add_r(hv[hp[0]], hv[hp[1]], t['rm_pu'] * zb1)
    for j in range(3):
        ckt.add_r(lv[j], 0, P['load_r'])
    bus = lv if c['bus'] == 'lv' else hv
    star = ckt.node('fault star point')
    ckt.add_r(star, 0, es.R_BIAS)
    switches = []
    for k, x in enumerate('abc'):
        if x in c['type'].replace('g', ''):
            mid = ckt.node(f'fault {x}')
            switches.append(ckt.add_switch(bus[k], mid, closed=False))
            ckt.add_r(mid, star, c['r'])
    if c['type'].endswith('g'):
        mid = ckt.node('fault ground')
        switches.append(ckt.add_switch(star, mid, closed=False))
        ckt.add_r(mid, 0, c['r'])
    for k in switches:
        ckt.at(c['t_on'], lambda st, k=k: st.set_switch(k, True))
    return ckt, g_src, lv, hv, c


def _fundamental(t, y):
    """Amplitude and phase (degrees) of y's 50 Hz component."""
    x = np.column_stack([np.cos(W * t), np.sin(W * t), np.ones_like(t)])
    c, *_ = np.linalg.lstsq(x, y, rcond=None)
    return math.hypot(c[0], c[1]), math.degrees(math.atan2(-c[1], c[0]))


@pytest.mark.parametrize('case', ['slg_lv', 'abcg_hv'])
def test_ac_benchmark_matches_simscape(case):
    """
    A single-line-to-ground fault at 0.4 kV and a three-phase-to-ground fault at
    20 kV, from rest, against Simscape Electrical's blocks: each signal's 50 Hz
    amplitude before the fault and late in it within 1 % (of its pre-fault
    amplitude, or of its own if larger), its phase within 0.5 degrees, and the source's peak currents at
    the fault within 2 %. The comparison stops at the fault's end: Simscape's
    fault block opens at once, chopping inductive current. (Simscape's
    transformer has a three- or five-limb core, whose outer limbs differ: its
    phases' magnetising currents, and so its source currents, differ by some
    0.5 %. Electrisim's is three single-phase units.)
    """
    ckt, g_src, lv, hv, c = _benchmark(case)
    sim = ckt.simulate(c['t_end'], 5e-6)
    ref = np.genfromtxt(os.path.join(HERE, f'emt_ac_{case}.csv'), delimiter=',', names=True)
    t_ref = ref['t']
    before = (t_ref >= c['t_on'] - 0.04) & (t_ref < c['t_on'])
    late = (t_ref >= c['t_on'] + 0.04) & (t_ref < c['t_on'] + c['duration'] - 0.001)
    for name, arr in (('i_src', sim['i_cp'][g_src]), ('v_lv', sim['v'][:, lv]), ('v_hv', sim['v'][:, hv])):
        for k, x in enumerate('abc'):
            theirs = ref[f'{name}_{x}']
            mine = np.interp(t_ref, sim['t'], arr[:, k])
            scale = _fundamental(t_ref[before], theirs[before])[0]
            for sel in (before, late):
                am, pm = _fundamental(t_ref[sel], mine[sel])
                ar, pr = _fundamental(t_ref[sel], theirs[sel])
                assert abs(am - ar) <= 0.01 * max(scale, ar), (name, x)
                if ar > 0.05 * scale:
                    assert abs((pm - pr + 180) % 360 - 180) <= 0.5, (name, x)
    onset = (t_ref >= c['t_on']) & (t_ref < c['t_on'] + 0.02)
    for k, x in enumerate('abc'):
        mine = np.interp(t_ref[onset], sim['t'], sim['i_cp'][g_src][:, k])
        assert np.max(np.abs(mine)) == pytest.approx(np.max(np.abs(ref[f'i_src_{x}'][onset])), rel=0.02)


# --- The study on the drawn radial grid ----------------------------------------------

def _run(client, quiet, **params):
    request = _request(**{'time_step_us': '10', 'duration_ms': '60', **{k: str(v) for k, v in params.items()}})
    with quiet():
        response = client.post('/', json=request)
    result = json.loads(response.get_data(as_text=True))
    assert not result.get('error'), result.get('message')
    return result


def _ac_bus(result, label):
    return next(b for b in result['emt']['ac']['buses'] if b['label'] == label)


def test_ac_network_starts_in_its_steady_state(client, quiet):
    """With nothing happening, every AC bus holds its rms voltage from the first cycle to the last."""
    result = _run(client, quiet, duration_ms=40)
    for bus in result['emt']['ac']['buses']:
        assert bus['v_rms_min_pu'] == pytest.approx(bus['v_rms_final_pu'], abs=1e-4)
    assert any('taken as clock 11' in w for w in result['warnings'])


def test_single_line_to_ground_fault_at_low_voltage(client, quiet):
    """
    A bolted fault on phase a of LV network A, behind its 0.8 MVA Dyn
    transformer: the fault's symmetrical current is pandapower's single-phase
    short-circuit current at that bus, scaled from its voltage factor c = 1.1 to the bus's
    load-flow voltage, within 5 % (pandapower takes its generation as current
    sources of their short-circuit ratio; here they hold their load-flow
    current). It clears at its current zero, and the bus recovers.
    """
    from test_dc_elements import _drawn_request
    request = _drawn_request()
    lv = next(v['name'] for v in request.values() if isinstance(v, dict) and v.get('userFriendlyName') == 'LV network A')
    result = _run(client, quiet, ac_fault_bus=lv, ac_fault_type='ag', ac_fault_time_ms=20, ac_fault_duration_ms=40,
                  ac_fault_resistance_ohm=0.0001, duration_ms=100)
    fault = result['emt']['ac']['fault']
    (phase_a,) = fault['phases']
    # pandapower's own single-phase short circuit at the bus.
    import pandapower.shortcircuit as sc
    import pandapower_electrisim as pe
    net = pp.create_empty_network()
    busbars = pe.create_busbars(request, net)
    key = next(k for k, v in request.items() if 'Parameters' in str(v.get('typ', '')))
    pe.create_other_elements(request, net, key, busbars)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pp.runpp(net)
        vm = float(net.res_bus.vm_pu[net.bus.name == lv].iloc[0])
        for table in ('vsc', 'line_dc', 'load_dc', 'source_dc', 'bus_dc'):
            net[table].drop(net[table].index, inplace=True)
        sc.calc_sc(net, case='max', fault='1ph', branch_results=False)
    ikss = float(net.res_bus_sc.ikss_ka[net.bus.name == lv].iloc[0]) * vm / 1.1
    assert phase_a['i_sym_ka'] == pytest.approx(ikss, rel=0.05)       # symmetrical: its DC offset aside
    assert phase_a['i_rms_ka'] > phase_a['i_sym_ka']                   # X/R 25: its offset lasts
    bus = _ac_bus(result, 'LV network A')
    assert bus['v_rms_min_pu'] < 0.05
    assert bus['v_rms_final_pu'] == pytest.approx(0.986, abs=0.02)       # recovered after clearing


def test_three_phase_fault_is_balanced(client, quiet):
    """Once its DC offsets have died away, a three-phase-to-ground fault draws the same current in each phase."""
    from test_dc_elements import _drawn_request
    request = _drawn_request()
    b1 = next(v['name'] for v in request.values() if isinstance(v, dict) and v.get('userFriendlyName') == 'B1')
    result = _run(client, quiet, ac_fault_bus=b1, ac_fault_type='abcg', ac_fault_time_ms=20, ac_fault_duration_ms=0,
                  duration_ms=150, time_step_us=20)
    phases = result['emt']['ac']['fault']['phases']
    rms = [p['i_rms_ka'] for p in phases]
    assert len(rms) == 3 and max(rms) == pytest.approx(min(rms), rel=0.02)
    assert _ac_bus(result, 'B1')['v_rms_final_pu'] < 0.01


def test_fault_along_a_line_cleared_by_its_breakers(client, quiet):
    """
    A three-phase fault halfway along LA1, its breakers at both ends told to
    open at 40 ms: each phase opens at its current's next zero, within a half
    cycle; the substation recovers. A1 and A2 beyond it are an island of their
    own: the rooftop PV, a current source at its load-flow output with no
    anti-islanding trip, drives their loads' impedance far from 1 pu.
    """
    from test_dc_elements import _drawn_request
    request = _drawn_request()
    la1 = next(v['id'] for v in request.values() if isinstance(v, dict) and v.get('userFriendlyName') == 'LA1')
    result = _run(client, quiet, ac_fault_line=la1, ac_fault_fraction=0.5, ac_fault_type='abc',
                  ac_fault_time_ms=20, ac_trip_time_ms=40, duration_ms=120, time_step_us=20)
    (trip,) = result['emt']['line_trips']
    assert trip['line'] == 'LA1' and len(trip['opened_ms']) == 6
    assert all(40 < t < 40 + 1e3 / 50 / 2 + 1 for t in trip['opened_ms'])
    fault = result['emt']['ac']['fault']
    assert fault['line'] == 'LA1, 50 %' and len(fault['phases']) == 3
    assert _ac_bus(result, '20 kV substation')['v_rms_final_pu'] > 0.95
    assert abs(_ac_bus(result, 'A2')['v_rms_final_pu'] - 1.0) > 0.1


# --- A capacitor bank energised during the run ---------------------------------------

def _bank_on_a_stiff_grid(as_filter, controlled, duration_ms='120'):
    import emt_electrisim
    import pandapower_electrisim as pe
    net = pp.create_empty_network(f_hz=50)
    b = pp.create_bus(net, 20)
    pp.create_ext_grid(net, b, s_sc_max_mva=200, rx_max=0.1, x0x_max=1.0, r0x0_max=0.1)
    if as_filter:
        hf = pe.harmonic_filter(2.0, 20, 5, 'high_pass', 2)
        i = pp.create_shunt(net, b, q_mvar=hf['q_mvar'], p_mw=hf['p_mw'], vn_kv=20, name='C1')
        for k in ('type', 'order', 'quality'):
            net.shunt.at[i, f'electrisim_filter_{k}'] = hf[k]
        net.shunt.at[i, 'electrisim_filter_rated_q_mvar'] = 2.0
    else:
        pp.create_shunt(net, b, q_mvar=-2.0, p_mw=0.0, vn_kv=20, name='C1')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = json.loads(emt_electrisim.emt_study(net, {
            'time_step_us': '5', 'duration_ms': duration_ms, 'ac_energise_shunt': 'C1', 'ac_energise_time_ms': '33',
            'ac_energise_controlled': 'true' if controlled else 'false'}))
    return out['emt']['energise']


def test_energising_a_capacitor_bank():
    """
    A 2 Mvar bank on a 20 kV bus of 200 MVA, out of the load flow the run
    starts from, its breaker's poles closing together: the classic inrush -
    the first two poles' charge ringing through the grid's inductance,
    sqrt(4 L / C) some 40 ohm, near 0.8 kA against its 82 A steady peak, at
    50 x sqrt(200 / 2) = 500 Hz - and the bus overshooting toward 2 pu. As a
    5th-harmonic filter its reactor holds the inrush to some 0.3 kA and the
    bus to 1.07 pu. Closed by a synchronous breaker - a and b at a zero of
    their line voltage, c a quarter cycle later at its own - the plain bank
    draws a fifth of it. Each settles to its rated 57.7 A.
    """
    i_ss = 2e6 / (math.sqrt(3) * 20e3)
    plain = _bank_on_a_stiff_grid(False, False, duration_ms='400')
    assert 5 * math.sqrt(2) * i_ss < plain['i_peak_ka'] * 1e3 < 11 * math.sqrt(2) * i_ss
    assert 1.5 < plain['v_peak_pu'] < 2.0
    assert plain['closed_ms'] == [33.0, 33.0, 33.0]
    for ph in plain['phases']:
        assert ph['i_rms_final_ka'] * 1e3 == pytest.approx(i_ss * 1.01, rel=0.05)
    sync = _bank_on_a_stiff_grid(False, True)
    assert sync['i_peak_ka'] < 0.25 * plain['i_peak_ka'] and sync['v_peak_pu'] < 1.15
    a, b_, c = sync['closed_ms']
    assert a == b_ and 33.0 <= a < 43.0 and c - a == pytest.approx(5.0, abs=0.05)
    filt = _bank_on_a_stiff_grid(True, False)
    assert 0.25 < filt['i_peak_ka'] < 0.4
    assert filt['v_peak_pu'] < 1.1
    for ph in filt['phases']:
        assert ph['i_rms_final_ka'] * 1e3 == pytest.approx(i_ss * 1.01, rel=0.03)


def test_a_bus_floating_behind_a_delta_is_held_to_earth():
    """
    A 20 kV bus reached only through a transformer's delta winding - its grid
    or generator out of service - has no path to earth: the network's matrix
    was singular and the solver fell back to the whole matrix, densely, at
    every step (an islanded AI factory took over ten minutes for 20 ms). Each
    such bus is now held to earth through R_BIAS a phase, as a star point is:
    the run is quick, the delta's voltage the transformer's.
    """
    import time
    import emt_electrisim
    net = pp.create_empty_network(f_hz=50)
    hv, lv = pp.create_bus(net, 20, name='HV'), pp.create_bus(net, 0.4, name='LV')
    pp.create_ext_grid(net, lv, s_sc_max_mva=50, rx_max=0.1, x0x_max=1.0, r0x0_max=0.1)
    pp.create_transformer_from_parameters(net, hv, lv, 1.0, 20, 0.4, 1.0, 6.0, 1.0, 0.1,
                                          vector_group='Dyn11', shift_degree=150)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t0 = time.time()
        out = json.loads(emt_electrisim.emt_study(net, {'time_step_us': '20', 'duration_ms': '40'}))
        took = time.time() - t0
    buses = {b['name']: b for b in out['emt']['ac']['buses']}
    assert took < 20
    assert buses['HV']['v_rms_final_pu'] == pytest.approx(1.0, abs=0.02)
    ckt = es.Circuit()
    builder = emt_ac.AcBuilder(ckt, net, {}, [], 50.0)
    builder.build()
    assert builder.ground_floating_buses() == ['HV']
