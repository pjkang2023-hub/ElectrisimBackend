"""
The EMT study of DC networks: the solver's switches, surge arresters and
constant-power loads against exact answers; five 800 V benchmarks against
Simscape (tests/emt_reference: simscape_emt_benchmarks.m builds each case from
emt_benchmark_params.json, which these tests build Electrisim's circuits from)
to the design note's acceptance limits - peak and time to peak within 2 %,
settled values within 0.5 %, the same stability verdict; and the study on the
drawn network.
"""
import json
import math
import os

import numpy as np
import pytest

import emt_solver as es
from test_dc_elements import _breaker, _drawn_request, _with

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'emt_reference')
with open(os.path.join(HERE, 'emt_benchmark_params.json'), encoding='utf-8') as _h:
    P = json.load(_h)


def _ref(name):
    return np.genfromtxt(os.path.join(HERE, f'emt_{name}.csv'), delimiter=',', names=True)


# --- The solver against exact answers -----------------------------------------------

def test_breaker_opening_into_its_arrester_matches_the_exponentials():
    """
    800 V behind 1 mH carries 2 kA into a 1 mOhm fault; the breaker opens at
    1 ms and its 1.5 kV arrester drives the current to zero along an
    exponential: the current at opening, the time it reaches zero and the
    arrester's energy, against the closed forms.
    """
    e, l, i0, rf, vc, t_open = 800.0, 1e-3, 2000.0, 1e-3, 1500.0, 1e-3
    ckt = es.Circuit()
    m, f = ckt.node('contacts'), ckt.node('fault')
    ckt.add_rl(0, m, 0.0, l, i0=i0, e0=e)
    k = ckt.add_switch(m, f, closed=True)
    ka = ckt.add_arrester(m, f, vc)
    ckt.add_r(f, 0, rf)
    ckt.at(t_open, lambda st: st.set_switch(k, False))
    sim = ckt.simulate(0.01, 1e-6)
    t, i = sim['t'], sim['i_rl'][:, 0]
    r1 = 1 / (1 / es.SW_R_ON + 1 / es.ARR_R_OFF) + rf
    i1 = e / r1 + (i0 - e / r1) * math.exp(-t_open * r1 / l)
    r2 = es.ARR_R_ON + rf
    i_inf, tau = (e - vc) / r2, l / r2
    t_zero = t_open + tau * math.log((i1 - i_inf) / -i_inf)
    tt = np.linspace(t_open, t_zero, 100001)
    ii = i_inf + (i1 - i_inf) * np.exp(-(tt - t_open) / tau)
    energy = np.trapezoid((vc + es.ARR_R_ON * ii) * ii, tt)
    assert np.interp(t_open, t, i) == pytest.approx(i1, rel=1e-5)
    assert t[np.nonzero((t > t_open) & (i <= 1.0))[0][0]] == pytest.approx(t_zero, abs=2e-6)
    v_arr = sim['v'][:, m] - sim['v'][:, f]
    assert np.trapezoid(v_arr * sim['i_arr'][:, ka], t) == pytest.approx(energy, rel=1e-3)


def test_dc_load_model_in_time():
    """Constant power above its minimum voltage, constant current below, nothing at reverse voltage."""
    f = es.dc_load_current([0.1], 0.8, 1.0, 0.0, 0.0, 0.8)
    assert f(800.0, 0)[0] == pytest.approx(125.0) and f(800.0, 0)[1] == pytest.approx(-125.0 / 800)
    assert f(600.0, 0)[0] == pytest.approx(0.1e6 / 640) and f(600.0, 0)[1] == 0
    assert f(-10.0, 0) == (0.0, 0.0)
    assert f(20.0, 0)[0] == pytest.approx(0.1e6 / 640 * 20 / 40)    # the ramp to zero below 5 %


def test_constant_power_loads_by_a_low_rank_update_as_by_refactoring(monkeypatch):
    """
    Constant-power loads enter each iteration as a low-rank update of the
    network's sparse factor (Woodbury), and a node only they reach by
    refactoring the whole matrix: the run is the one refactoring at every
    iteration gave, to round-off. That took 7 ms an iteration on the AI
    campus's 319 nodes - 560 s for 40 ms.
    """
    def run(behind=False):
        p = [0.1]
        ckt = es.Circuit()
        a, b, x = ckt.node('A', 800.0), ckt.node('B', 800.0), ckt.node('behind B', 800.0)
        ckt.add_rl(0, a, 0.01, 1e-4, i0=250.0, e0=800.0)
        ckt.add_rl(a, b, 0.005, 5e-5, i0=250.0)
        ckt.add_c(a, 0, 1e-3, w0=800.0)
        ckt.add_c(b, 0, 1e-3, w0=800.0)
        ckt.add_nonlinear(b, 0, es.dc_load_current(p, 0.8, 1.0, 0.0, 0.0, 0.8))
        if behind:
            ckt.add_nonlinear(b, x, lambda v, t: (v / 0.05, 1 / 0.05))      # its only path: nonlinear branches
            ckt.add_nonlinear(x, 0, es.dc_load_current([0.1], 0.8, 1.0, 0.0, 0.0, 0.8))
        else:
            ckt.add_r(x, 0, 1.0)
        ckt.at(2e-3, lambda st: p.__setitem__(0, 0.15))
        return ckt.simulate(6e-3, 5e-6)

    fast, fast_behind = run(), run(behind=True)

    def no_sparse(*_):
        raise RuntimeError('refactor every iteration')
    monkeypatch.setattr(es, 'splu', no_sparse)
    for mine, ref in ((fast, run()), (fast_behind, run(behind=True))):
        assert np.allclose(mine['v'], ref['v'], rtol=1e-9, atol=1e-6)
        assert np.allclose(mine['i_nl'], ref['i_nl'], rtol=1e-9, atol=1e-6)
        assert np.ptp(mine['v'][mine['t'] > 2e-3, 2]) > 1.0          # the step moved bus B


# --- Against Simscape -------------------------------------------------------------

def _compare(t_ref, ref, t, mine, peak_rel=0.02, rms_rel=0.01, events=()):
    """
    Peak and time to peak within 2 %, and the whole waveform close, against
    Simscape - leaving out the samples at a switching instant, where one tool
    records the value just before the switch and the other just after.
    """
    y = np.interp(t_ref, t, mine)
    keep = np.ones(len(t_ref), dtype=bool)
    for te in events:
        keep &= np.abs(t_ref - te) > 1.5e-6
    k_ref, k = int(np.argmax(np.abs(ref))), int(np.argmax(np.abs(y)))
    scale = abs(ref[k_ref])
    assert abs(y[k]) == pytest.approx(scale, rel=peak_rel)
    assert t_ref[k] == pytest.approx(t_ref[k_ref], rel=peak_rel, abs=2e-6)
    assert np.sqrt(np.mean((y[keep] - ref[keep]) ** 2)) <= rms_rel * scale


def test_long_cable_as_pi_sections_matches_simscape():
    q = P['long_cable']
    ckt = es.Circuit()
    src, a = ckt.node('source'), ckt.node('A')
    k_src = ckt.add_rl(0, a, q['Rs'], q['Ls'], e0=q['E'])
    n = q['sections']
    r, l, c = (q[k] * q['km'] / n for k in ('r_per_km', 'l_per_km', 'c_per_km'))
    nodes = [a] + [ckt.node(f'section {k}') for k in range(1, n + 1)]
    caps = [ckt.add_c(0, node, (0.5 if k in (0, n) else 1.0) * c) for k, node in enumerate(nodes)]
    for k in range(n):
        ckt.add_rl(nodes[k], nodes[k + 1], r, l)
    b = nodes[-1]
    ckt.add_r(b, 0, q['R_load'])
    f = ckt.node('fault')
    k_sw = ckt.add_switch(b, f, closed=False)
    k_f = ckt.add_r(f, 0, q['Rf'])
    ckt.at(q['t_fault'], lambda st: st.set_switch(k_sw, True))
    sim = ckt.simulate(q['t_end'], 1e-7)
    ref = _ref('long_cable')
    t = sim['t']
    _compare(ref['t'], ref['i_source'], t, sim['i_rl'][:, k_src])
    _compare(ref['t'], ref['v_end'], t, sim['v'][:, b])
    _compare(ref['t'], ref['i_fault'], t, sim['i_r'][:, k_f])
    # Settled before the fault: the load's current, to 0.5 %.
    before = (ref['t'] > 0.4e-3) & (ref['t'] < 0.5e-3)
    assert np.mean(np.interp(ref['t'][before], t, sim['i_rl'][:, k_src])) == pytest.approx(
        np.mean(ref['i_source'][before]), rel=5e-3)


@pytest.mark.parametrize('frac, verdict', [(0.95, 'decays'), (1.05, 'grows')])
def test_constant_power_load_stability_matches_simscape(frac, verdict):
    """At 95 % of its stability limit the 1 % step's oscillation decays; at 105 % it grows - in both tools."""
    q = P['cpl']
    v0 = q['E'] / (1 + frac * q['R'] ** 2 * q['C'] / q['L'])
    p = v0 * (q['E'] - v0) / q['R']
    pw = [p / 1e6]
    ckt = es.Circuit()
    n = ckt.node('load', v0)
    ckt.add_rl(0, n, q['R'], q['L'], i0=p / v0, e0=q['E'])
    ckt.add_c(0, n, q['C'], w0=-v0)
    ckt.add_nonlinear(n, 0, es.dc_load_current(pw, q['E'] / 1e3, 1.0, 0.0, 0.0, 0.0))
    ckt.at(q['t_step'], lambda st: pw.__setitem__(0, pw[0] * (1 + q['step'])))
    sim = ckt.simulate(q['t_end'], 2e-6, dt_coarse=1e-5, t_fine=0.01)
    ref = _ref(f'cpl_{round(100 * frac)}')
    t_ref, v_ref = ref['t'], ref['v_load']
    v = np.interp(t_ref, sim['t'], sim['v'][:, n])

    def swing(y, a, b):
        sel = (t_ref > a) & (t_ref < b)
        return np.ptp(y[sel])
    for y in (v_ref, v):
        grows = swing(y, 0.12, 0.15) > swing(y, 0.01, 0.04)
        assert ('grows' if grows else 'decays') == verdict
    # The swing itself, to 2 %, and the waveform close throughout.
    assert swing(v, 0.12, 0.15) == pytest.approx(swing(v_ref, 0.12, 0.15), rel=0.02)
    assert np.sqrt(np.mean((v - v_ref) ** 2)) <= 0.02 * swing(v_ref, 0.0, 0.15)


def test_constant_power_load_following_a_training_cycle_matches_simscape():
    """
    A 50 kW constant-power load following a training-cycle profile - from 0.9
    to 0.75 pu in its first 10 ms, up to 1.0 pu, down to a 0.3 pu checkpoint
    and back - on 2 mF behind 0.05 ohm and 1 mH, against Simscape's load drawing
    P(t) / v from a lookup table: the load's voltage within 0.05 % of 800 V
    throughout, its dips at the profile's rises and its rise at the
    checkpoint the same in both.
    """
    import load_profiles_electrisim as lp
    q = P['cpl_profile']
    f = lp.Follower(np.arange(len(q['profile'])) * q['dt_profile'], q['profile'], 0.0, repeat=False)
    p0 = q['p_set'] * f(0.0)
    v0 = (q['E'] + math.sqrt(q['E'] ** 2 - 4 * q['R'] * p0)) / 2
    ckt = es.Circuit()
    n = ckt.node('load', v0)
    ckt.add_rl(0, n, q['R'], q['L'], i0=p0 / v0, e0=q['E'])
    ckt.add_c(0, n, q['C'], w0=-v0)
    power = lambda t: q['p_set'] * f(t) / 1e6       # noqa: E731 - its power, at each solve's own time
    ckt.add_nonlinear(n, 0, es.dc_load_current([p0 / 1e6], q['E'] / 1e3, 1.0, 0.0, 0.0, 0.0, power=power))
    sim = ckt.simulate(q['t_end'], 1e-5)
    ref = _ref('cpl_profile')
    v = np.interp(ref['t'], sim['t'], sim['v'][:, n])
    assert np.max(np.abs(v - ref['v_load'])) < 5e-4 * q['E']
    assert np.ptp(ref['v_load']) > 0.01 * q['E']                    # the profile moves it
    assert np.argmin(v) == pytest.approx(np.argmin(ref['v_load']), abs=2)


@pytest.mark.parametrize('case', ['solid_state', 'mechanical'])
def test_breaker_clearing_a_fault_matches_simscape(case):
    q, c = P['breaker'], P['breaker']['cases'][case]
    ckt = es.Circuit()
    a, contacts, term, b = ckt.node('A'), ckt.node('contacts'), ckt.node('terminal'), ckt.node('B')
    ckt.add_rl(0, a, q['Rs'], q['Ls'], e0=q['E'])
    k_lim = ckt.add_rl(a, contacts, 0.0, q['L_lim'])
    k_sw = ckt.add_switch(contacts, term, closed=True)
    ckt.add_arrester(contacts, term, q['Vc'])
    ckt.add_rl(term, b, q['R_cable'], q['L_cable'])
    ckt.add_r(b, 0, q['R_load'])
    f = ckt.node('fault')
    k_fsw = ckt.add_switch(b, f, closed=False)
    k_f = ckt.add_r(f, 0, q['Rf'])
    ckt.at(q['t_fault'], lambda st: st.set_switch(k_fsw, True))
    ckt.at(c['t_open'], lambda st: st.set_switch(k_sw, False))
    sim = ckt.simulate(c['t_end'], 1e-6)
    ref = _ref(f'breaker_{case}')
    t = sim['t']
    events = (q['t_fault'], c['t_open'])
    _compare(ref['t'], ref['i_breaker'], t, sim['i_rl'][:, k_lim], events=events)
    _compare(ref['t'], ref['v_contacts'], t, sim['v'][:, contacts] - sim['v'][:, term], events=events)
    # Cleared: the current reaches zero at the same time, to 2 %.
    i_ref = ref['i_breaker']
    t_clear_ref = ref['t'][np.nonzero((ref['t'] > c['t_open']) & (i_ref <= 1.0))[0][0]]
    i = sim['i_rl'][:, k_lim]
    t_clear = t[np.nonzero((t > c['t_open']) & (i <= 1.0))[0][0]]
    assert t_clear - c['t_open'] == pytest.approx(t_clear_ref - c['t_open'], rel=0.02)


# --- The study on the drawn network -----------------------------------------------

def _request(**params):
    request = _drawn_request()
    for el in request.values():
        if isinstance(el, dict) and el.get('name') in ('ld_a', 'ld_b'):
            el.update(filter_c_uf='5000', filter_l_mh='0.001', v_min_pu='0.8')
    request = _with(request, {'typ': 'DC Capacitor0', 'name': 'clink', 'id': 'cell-clink', 'userFriendlyName': 'C link',
                              'bus': 'dc_a', 'c_mf': '10', 'esr_mohm': '2', 'esl_uh': '0.1'},
                    _breaker('qa', 'dc_a', 'cable', 'line_dc', breaking_capacity_ka=20, opening_time_ms=0.05,
                             limiting_inductance_mh=0.05, rated_current_ka=1, trip_current_ka=2,
                             arrester_clamp_kv=1.4, arrester_energy_kj=50))
    key = next(k for k, v in request.items() if 'Parameters' in str(v.get('typ', '')))
    request[key] = {'typ': 'EmtStudy Parameters', 'time_step_us': '1', 'duration_ms': '20', 'user_email': 't@t',
                    **{k: str(v) for k, v in params.items()}}
    return request


def _run(client, quiet, request):
    with quiet():
        response = client.post('/', json=request)
    result = json.loads(response.get_data(as_text=True))
    assert not result.get('error'), result.get('message')
    return result


def test_quiet_without_events(client, quiet):
    """From the load-flow state with nothing happening, the network stays at it."""
    result = _run(client, quiet, _request(duration_ms=5))
    for bus in result['emt']['buses']:
        assert bus['v_min_pu'] == pytest.approx(bus['v_final_pu'], abs=2e-4)
        assert bus['v_max_pu'] == pytest.approx(bus['v_final_pu'], abs=2e-4)
    assert all(b['opened_ms'] is None for b in result['emt']['breakers'])


def test_fault_cleared_by_a_breaker(client, quiet):
    """
    A fault on DC bus B at 5 ms: the breaker on the cable trips at 2 kA, opens
    0.05 ms later, its arrester clamps at 1.4 kV and absorbs the energy, and
    DC bus A rides through - dipping some 4 % while the rectifier's DC voltage
    loop catches up, and within 1 % of its set point by the end. The run is too short after the
    fault to tell whether the hall left on settles.
    """
    result = _run(client, quiet, _request(fault_bus='dc_b', fault_time_ms=5, fault_resistance_mohm=1))
    emt = result['emt']
    (qa,) = emt['breakers']
    assert 5.0 < qa['tripped_ms'] < qa['opened_ms'] < qa['cleared_ms']
    assert qa['opened_ms'] - qa['tripped_ms'] == pytest.approx(0.05, abs=0.002)
    assert qa['i_open_ka'] > 2 and qa['arrester_v_peak_kv'] == pytest.approx(1.4, rel=0.01)
    assert 0 < qa['arrester_energy_kj'] < 50 and not qa['exceeds_energy']
    buses = {b['id']: b for b in emt['buses']}
    assert 0.95 < buses['cell-dc_a']['v_min_pu'] < 0.99
    assert buses['cell-dc_a']['v_final_pu'] == pytest.approx(1.0, abs=1e-2)
    assert buses['cell-dc_b']['v_final_pu'] == pytest.approx(0.0, abs=1e-3)
    assert emt['fault']['ip_ka'] > qa['i_open_ka']    # the load's filter discharges into the fault too
    loads = {l['id']: l for l in emt['loads']}
    assert loads['cell-ld_b']['verdict'] == 'lost supply' and loads['cell-ld_a']['verdict'] == 'too short to tell'
    assert any('run at least 60 ms after the last event' in w for w in result['warnings'])


def test_load_step_settles(client, quiet):
    """Server hall B steps up 20 %: run long enough for the rectifier's DC voltage loop, both halls settle."""
    result = _run(client, quiet, _request(step_load='ld_b', step_percent=20, step_time_ms=2, duration_ms=100,
                                          time_step_us=5))
    loads = {l['id']: l for l in result['emt']['loads']}
    assert loads['cell-ld_b']['verdict'] == 'settles' and loads['cell-ld_a']['verdict'] == 'settles'


def test_loads_without_input_filters_are_given_one(client, quiet):
    request = _request(duration_ms=2)
    for el in request.values():
        if isinstance(el, dict) and el.get('name') == 'ld_b':
            el['filter_c_uf'] = '0'
    result = _run(client, quiet, request)
    assert any('input capacitance of 4 ms' in w and 'Server hall B' in w for w in result['warnings'])


# --- Loads following a profile -----------------------------------------------------

CYCLE = [0.9, 0.75, 0.75, 0.75, 0.85, 0.95, 1.0, 1.0, 1.0, 0.3, 0.3, 0.75]   # 10 ms apart


def _profiled(dc=True, ac=True, profile_id='profile_1', **params):
    """The drawn network, Server hall B (DC) and Village A (AC) following a training cycle, at a 10 us step."""
    request = _request(**{'time_step_us': 10, 'duration_ms': 85, **params})
    for el in request.values():
        if not isinstance(el, dict):
            continue
        if dc and el.get('name') == 'ld_b':
            el['load_profile_id'] = profile_id
        if ac and el.get('userFriendlyName') == 'Village A':
            el.update(load_profile_id=profile_id, load_profile_q_mode='pf')
    key = next(k for k, v in request.items() if isinstance(v, dict) and 'EmtStudy' in str(v.get('typ', '')))
    request[key]['load_profiles'] = {'profile_1': {'name': 'Training cycle', 'dt': 0.01, 'p': CYCLE}}
    return request


def _by_label(rows):
    return {r['label']: r for r in rows}


def test_loads_follow_their_profiles(client, quiet):
    """
    Server hall B (0.1 MW, DC) and Village A (1.5 MW, AC) on a training cycle:
    each starts at its profile's first value - the load flow too - and draws
    its profile through the run: over the last cycle, the DC load's power its
    profile's within 0.1 %, the AC load's - an impedance following it, its
    power with its voltage - within 0.5 %.
    """
    result = _run(client, quiet, _profiled())
    loads = _by_label(result['emt']['profiled_loads'])
    hall, village = loads['Server hall B'], loads['Village A']
    assert hall['kind'] == 'DC load' and village['kind'] == 'Load' and hall['profile'] == 'Training cycle'
    assert hall['p_start_mw'] == pytest.approx(0.09) and village['p_start_mw'] == pytest.approx(1.35)
    # Its least at the run's end, 85 ms, halfway down to the checkpoint; its most at 1.0 pu.
    assert hall['p_min_mw'] == pytest.approx(0.065, rel=1e-3) and hall['p_max_mw'] == pytest.approx(0.1, rel=1e-3)
    assert hall['p_drawn_end_mw'] == pytest.approx(hall['p_end_mw'], rel=1e-3)
    assert village['p_drawn_end_mw'] == pytest.approx(village['p_end_mw'], rel=5e-3)
    assert any('start 0 s into it' in w for w in result['warnings'])
    # The profiles keep the voltages moving: no stability verdict, and no warning of growing swings.
    assert all(l['verdict'] is None for l in result['emt']['loads'])
    assert not any('swings ever wider' in w for w in result['warnings'])


def test_loads_start_where_asked_in_their_profile(client, quiet):
    """
    Started 0.085 s into the cycle - halfway down to its checkpoint, at 0.65
    pu - the network starts in the load flow at that power and holds while
    the profile falls on (DC bus B within 0.2 % over the first 2 ms).
    """
    result = _run(client, quiet, _profiled(ac=False, profile_start_s=0.085, duration_ms=2))
    (hall,) = result['emt']['profiled_loads']
    assert hall['p_start_mw'] == pytest.approx(0.065, rel=1e-6)
    bus_b = next(b for b in result['emt']['buses'] if b['label'] == 'DC bus B')
    assert bus_b['v_max_pu'] - bus_b['v_min_pu'] < 2e-3


def test_load_step_scales_a_profile(client, quiet):
    """A 50 % step in Server hall B at 40 ms, while it follows its profile: the profile, scaled from then on."""
    result = _run(client, quiet, _profiled(ac=False, step_load='ld_b', step_percent=50, step_time_ms=40))
    (hall,) = result['emt']['profiled_loads']
    assert hall['p_drawn_end_mw'] == pytest.approx(1.5 * hall['p_end_mw'], rel=2e-3)


def test_a_profile_not_in_the_library_is_warned_about(client, quiet):
    result = _run(client, quiet, _profiled(ac=False, profile_id='profile_9', duration_ms=1))
    assert result['emt']['profiled_loads'] == []
    assert any('Server hall B' in w and 'not in the library' in w for w in result['warnings'])
