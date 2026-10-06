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
    loop catches up, and back by the end. The run is too short after the
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
    assert buses['cell-dc_a']['v_final_pu'] == pytest.approx(1.0, abs=5e-3)
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
