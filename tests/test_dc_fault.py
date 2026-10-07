"""
The DC fault study: its solver against exact solutions, IEC 61660-1's terms
fitted to known curves, the study on the drawn 800 V DC network, and an
800 V benchmark against Simscape (tests/emt_reference: the MATLAB script
there builds the same circuit from the same parameter file and regenerates
the reference waveforms).
"""
import json
import math
import os

import numpy as np
import pytest

import dc_fault_electrisim as dcf
from test_dc_elements import _breaker, _drawn_request, _with


# --- The solver against exact solutions ----------------------------------------------

def test_capacitor_discharge_matches_the_rlc_solution():
    """10 mF at 750 V discharging through 4 mOhm and 60 uH into a short: the series RLC answer."""
    c, v, r, l = 10e-3, 750.0, 4e-3, 60e-6
    ckt = dcf.Circuit()
    bus, inner = ckt.node('bus'), ckt.node('capacitor')
    ckt.add_c(0, inner, c, w0=-v)
    ckt.add_rl(inner, bus, r, l)
    k = ckt.add_r(bus, 0, dcf.R_FLOOR)
    sim = ckt.simulate(0.005, 1e-6)
    a = (r + dcf.R_FLOOR) / (2 * l)
    w = math.sqrt(1 / (l * c) - a * a)
    exact = v / (w * l) * np.exp(-a * sim['t']) * np.sin(w * sim['t'])
    tp = math.atan(w / a) / w
    i = sim['i_r'][:, k]
    assert np.max(np.abs(i - exact)) < 1e-5 * exact.max()
    assert sim['t'][np.argmax(i)] == pytest.approx(tp, abs=1e-6)


def test_source_behind_r_and_l_matches_the_exponential():
    """A battery behind 20 mOhm and 10 uH, carrying 125 A, into a 1 mOhm fault."""
    e, r, l, i0, rf = 800.0, 0.02, 10e-6, 125.0, 1e-3
    ckt = dcf.Circuit()
    bus = ckt.node('bus')
    ckt.add_rl(0, bus, r, l, i0=i0, e0=e)
    k = ckt.add_r(bus, 0, rf)
    sim = ckt.simulate(0.005, 1e-6)
    ik = e / (r + rf)
    exact = ik + (i0 - ik) * np.exp(-sim['t'] * (r + rf) / l)
    assert np.max(np.abs(sim['i_r'][1:, k] - exact[1:])) < 1e-5 * ik


def test_diode_bridge_into_a_dc_short_matches_the_rectified_three_phase_short():
    """
    Shorted at its DC terminals, a diode bridge carries the rectified three-
    phase short circuit: (|ia| + |ib| + |ic|) / 2, mean 3*sqrt(2)/pi * Ik3,
    between sqrt(3)/2 * sqrt(2) * Ik3 and sqrt(2) * Ik3.
    """
    v_ll, f, r, x = 400.0, 50.0, 0.002, 0.02
    w = 2 * math.pi * f
    ckt = dcf.Circuit()
    p, n = ckt.node('DC+'), ckt.node('neutral')
    ckt.add_r(n, p, dcf.R_BIAS)
    ckt.add_r(n, 0, dcf.R_BIAS)
    for k_ph in range(3):
        node = ckt.node('phase')
        ckt.add_rl(n, node, r, x / w, ac=(math.sqrt(2 / 3) * v_ll, w, -k_ph * 2 * math.pi / 3))
        ckt.add_diode(node, p)
        ckt.add_diode(0, node)
    k = ckt.add_r(p, 0, dcf.R_FLOOR)
    sim = ckt.simulate(0.2, 1e-6, dt_coarse=1e-5, t_fine=0.01)
    last = sim['t'] >= 0.2 - 1 / f
    i = sim['i_r'][last, k]
    peak = math.sqrt(2) * v_ll / (math.sqrt(3) * math.hypot(r, x))
    assert i.mean() == pytest.approx(3 * peak / math.pi, rel=2e-4)
    assert i.max() == pytest.approx(peak, rel=5e-3)
    assert i.min() == pytest.approx(math.sqrt(3) / 2 * peak, rel=5e-3)


def test_iec_terms_recover_the_approximation_function():
    """IEC 61660-1's rise and decay, generated, give back their ip, tp, Ik, tau1, tau2."""
    ip, tp, ik, tau1, tau2 = 10e3, 1.2e-3, 4e3, 0.4e-3, 3e-3
    t = np.linspace(0, 0.05, 50001)
    i = np.where(t <= tp, dcf._rise(t, ip, tp, tau1), dcf._decay(t, ip, tp, ik, tau2))
    terms = dcf.iec_terms(t, i)
    assert terms['ip'] == pytest.approx(ip) and terms['tp'] == pytest.approx(tp, abs=1e-6)
    assert terms['ik'] == pytest.approx(ik, rel=1e-3)
    assert terms['tau1'] == pytest.approx(tau1, rel=1e-3)
    assert terms['tau2'] == pytest.approx(tau2, rel=1e-3)
    assert not terms['monotonic']


def test_iec_terms_of_a_current_without_a_peak():
    """A current rising to Ik with no peak: ip = Ik, tp where it reaches 99 % of it."""
    t = np.linspace(0, 0.01, 10001)
    i = 5e3 * (1 - np.exp(-t / 1e-3))
    terms = dcf.iec_terms(t, i)
    assert terms['monotonic']
    assert terms['ip'] == pytest.approx(terms['ik'])
    assert terms['tp'] == pytest.approx(-1e-3 * math.log(1 - 0.99 * i[-1] / 5e3), abs=2e-6)
    assert terms['tau2'] is None


# --- The study on the drawn network --------------------------------------------------

STUDY = {'typ': 'DcFaultStudy Parameters', 'fault_resistance_mohm': '0', 'time_step_us': '1',
         'duration_ms': '200', 'fault_angle_deg': '0', 'user_email': 't@t'}


def _study(client, quiet, request, **params):
    key = next(k for k, v in request.items() if 'Parameters' in str(v.get('typ', '')))
    request[key] = {**STUDY, **{k: str(v) for k, v in params.items()}}
    with quiet():
        response = client.post('/', json=request)
    result = json.loads(response.get_data(as_text=True))
    assert not result.get('error'), result.get('message')
    return result


def _fault(result, bus_id):
    return next(f for f in result['dcfault']['faults'] if f['id'] == bus_id)


def _cap(name, bus, c_mf=10, esr_mohm=2, esl_uh=0.1):
    return {'typ': 'DC Capacitor0', 'name': name, 'id': f'cell-{name}', 'userFriendlyName': name,
            'bus': bus, 'c_mf': str(c_mf), 'esr_mohm': str(esr_mohm), 'esl_uh': str(esl_uh)}


def test_fault_on_each_dc_bus_fed_by_the_converter(client, quiet):
    """
    Without DC capacitors only the VSC feeds the fault: its DC-link
    capacitor at once - the peak - then its diodes, once the DC voltage falls
    below the AC peak - Ik. A fault at the far bus draws less than one at the
    converter's, through the cable. At the converter's own bus its DC link,
    with no ESR, discharges within a step of a bolted fault: the study warns
    that ip is not resolved there, and not at the far bus.
    """
    result = _study(client, quiet, _drawn_request())
    faults = {f['id']: f for f in result['dcfault']['faults']}
    assert set(faults) == {'cell-dc_a', 'cell-dc_b'}
    a, b = faults['cell-dc_a'], faults['cell-dc_b']
    assert a['ik_ka'] > b['ik_ka'] > 0
    for f in (a, b):
        diodes, link = f['contributions']
        assert (diodes['kind'], link['kind']) == ('VSC (diodes, blocked)', 'VSC DC-link capacitor')
        assert diodes['name'] == link['name'] == 'Rectifier'
        assert diodes['ik_ka'] == pytest.approx(f['ik_ka'], rel=1e-3)
        assert link['ip_ka'] == pytest.approx(f['ip_ka'], rel=1e-3) and abs(link['ik_ka']) < 1e-3
        assert f['tp_ms'] < 0.1 and f['settled']
    unresolved = [w for w in result['warnings'] if 'ip is the time step' in w]
    assert len(unresolved) == 1 and 'DC bus A' in unresolved[0] and 'Rectifier DC link' in unresolved[0]


def test_dc_link_capacitor_discharges_into_a_fault_at_its_bus(client, quiet):
    """
    A 10 mF capacitor (2 mOhm, 0.1 uH) on DC bus B: at the fault on B it sees
    a near-short, so its current is the series RLC discharge from B's
    pre-fault voltage - and it sets the peak, within the first millisecond,
    the VSC's DC link adding its own over the cable from A.
    """
    result = _study(client, quiet, _with(_drawn_request(), _cap('C link', 'dc_b')), duration_ms=20)
    fault = _fault(result, 'cell-dc_b')
    cap = next(c for c in fault['contributions'] if c['kind'] == 'DC capacitor')
    v0 = fault['v_prefault_kv'] * 1e3
    r, l, c = 2e-3 + dcf.R_FLOOR, 0.1e-6, 10e-3
    a = r / (2 * l)
    w = math.sqrt(1 / (l * c) - a * a)
    tp = math.atan(w / a) / w
    ip = v0 / (w * l) * math.exp(-a * tp) * math.sin(w * tp)
    assert cap['ip_ka'] == pytest.approx(ip / 1e3, rel=2e-3)
    link = next(c for c in fault['contributions'] if c['kind'] == 'VSC DC-link capacitor')
    assert fault['ip_ka'] == pytest.approx(cap['at_peak_ka'] + link['at_peak_ka'], rel=0.02)
    assert cap['at_peak_ka'] > 5 * link['at_peak_ka'] > 0
    assert fault['tp_ms'] < 1 and not fault['monotonic']
    assert fault['tau1_ms'] > 0 and fault['tau2_ms'] > 0
    # Every source's current at the peak adds up to the fault current.
    assert sum(c['at_peak_ka'] for c in fault['contributions']) == pytest.approx(fault['ip_ka'], rel=1e-3)


def test_battery_behind_its_internal_impedance(client, quiet):
    """
    The battery rack, coupled to DC bus B by a breaker, behind 20 mOhm and
    10 uH: at a fault on its own bus it settles at E / R, E its voltage
    behind the resistance before the fault.
    """
    request = _drawn_request()
    batt = next(v for v in request.values() if isinstance(v, dict) and v.get('name') == 'batt')
    batt.update(r_sc_mohm='20', l_sc_uh='10')
    result = _study(client, quiet, _with(request, _breaker('qr', 'dc_b', 'rack', 'bus_dc')), duration_ms=40)
    fault = _fault(result, 'cell-rack')
    src = next(c for c in fault['contributions'] if c['kind'] == 'DC source')
    assert src['name'] == 'Battery'
    v0 = fault['v_prefault_kv'] * 1e3
    # The rack carries little before the fault: E is close to its voltage.
    assert src['ik_ka'] == pytest.approx(v0 / 0.02 / 1e3, rel=0.01)
    assert not any('no internal resistance' in w for w in result['warnings'])


def test_breaker_checked_against_its_breaking_capacity(client, quiet):
    """
    A breaker on the cable at DC bus A, opening in 2 ms with a 1 kA breaking
    capacity: a fault on B drives more than that through it - the warning
    says so, and when it reached the capacity.
    """
    request = _with(_drawn_request(), _cap('C link', 'dc_a'),
                    _breaker('qa', 'dc_a', 'cable', 'line_dc', breaking_capacity_ka=1, opening_time_ms=2,
                             limiting_inductance_mh=0.01, rated_current_ka=1))
    result = _study(client, quiet, request, duration_ms=20)
    fault = _fault(result, 'cell-dc_b')
    (brk,) = fault['breakers']
    assert brk['i_open_ka'] > 1 and brk['exceeds']
    assert 0 < brk['t_reaches_capacity_ms'] < 2
    (worst,) = result['dcfault']['breakers']
    assert worst['fault_bus'] == 'DC bus B'
    assert any('QA' in w and 'breaking capacity' in w for w in result['warnings'])
    # At a fault on A the breaker is behind the fault: the cable feeds it back.
    assert _fault(result, 'cell-dc_a')['breakers'][0]['i_open_ka'] < brk['i_open_ka']


def test_source_without_impedance_is_named(client, quiet):
    request = _with(_drawn_request(), _breaker('qr', 'dc_b', 'rack', 'bus_dc'))
    result = _study(client, quiet, request, duration_ms=5, fault_bus='dc_a')
    assert [f['id'] for f in result['dcfault']['faults']] == ['cell-dc_a']
    assert any("Source DC 'Battery' has no internal resistance" in w for w in result['warnings'])


# --- Against Simscape ----------------------------------------------------------------

EMT_REFERENCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'emt_reference')


def _benchmark_circuit(p):
    """The circuit in dc_fault_benchmark_params.json, as simscape_dc_fault_benchmark.m builds it."""
    w = 2 * math.pi * p['f_hz']
    ckt = dcf.Circuit()
    bridge, neutral = ckt.node('DC+'), ckt.node('neutral')
    ckt.add_r(neutral, bridge, p['r_bias'])
    ckt.add_r(neutral, 0, p['r_bias'])
    for k in range(3):
        x = ckt.node('phase')
        ckt.add_rl(neutral, x, p['r_ac'], p['x_ac'] / w,
                   ac=(math.sqrt(2 / 3) * p['v_ll'], w, math.radians(p['phase_deg']) - k * 2 * math.pi / 3))
        ckt.add_diode(x, bridge)
        ckt.add_diode(0, x)
    a, b, inner = ckt.node('DC bus A'), ckt.node('DC bus B'), ckt.node('capacitor')
    ckt.add_c(0, inner, p['c_link'], w0=-p['v0'])
    ckt.add_c(0, a, p['cable_c'] / 2, w0=-p['v0'])
    ckt.add_c(0, b, p['cable_c'] / 2, w0=-p['v0'])
    branches = {
        'i_bridge': ('i_rl', ckt.add_rl(bridge, a, p['r_dc'], 0.0)),
        'i_cap': ('i_rl', ckt.add_rl(inner, a, p['esr'], p['esl'])),
        'i_cable': ('i_rl', ckt.add_rl(a, b, p['cable_r'], p['cable_l'])),
        'i_batt': ('i_rl', ckt.add_rl(0, b, p['batt_r'], p['batt_l'], e0=p['batt_e'])),
        'i_fault': ('i_r', ckt.add_r(b, 0, p['r_fault'])),
    }
    return ckt, branches


def test_benchmark_matches_simscape():
    """
    The 800 V benchmark - a diode bridge from the grid, a DC-link capacitor, a
    cable and a battery, faulted at the battery's bus - against Simscape, at
    the study's own time steps: each current within 0.5 % of its peak.

    At t = 0 Simscape has the cable's own 20 nF discharging into the 1 mOhm
    fault, 800 kA for some 20 ns; the study's microsecond steps do not resolve
    it (nor does IEC 61660-1, which neglects cable capacitance), so the
    comparison starts at the first sample after it.
    """
    with open(os.path.join(EMT_REFERENCE, 'dc_fault_benchmark_params.json'), encoding='utf-8') as handle:
        p = json.load(handle)
    ref = np.genfromtxt(os.path.join(EMT_REFERENCE, 'dc_fault_benchmark.csv'), delimiter=',', names=True)
    ckt, branches = _benchmark_circuit(p)
    sim = ckt.simulate(p['t_end'], 1e-6, dt_coarse=1e-5, t_fine=0.01)
    after = ref['t'] > 0
    for name, (kind, k) in branches.items():
        mine = np.interp(ref['t'][after], sim['t'], sim[kind][:, k])
        theirs = ref[name][after]
        scale = np.max(np.abs(theirs))
        assert np.max(np.abs(mine - theirs)) < 5e-3 * scale, name
    # And in IEC terms: the fault current rises, without a peak of its own, to the same Ik.
    mine = dcf.iec_terms(sim['t'], sim['i_r'][:, branches['i_fault'][1]], 1 / p['f_hz'])
    theirs = dcf.iec_terms(ref['t'][after], ref['i_fault'][after], 1 / p['f_hz'])
    assert mine['ik'] == pytest.approx(theirs['ik'], rel=2e-3)
    assert mine['ip'] == pytest.approx(theirs['ip'], rel=2e-3)


def test_breaker_in_front_of_a_load_carries_no_fault_current(client, quiet):
    """A DC load without an input filter leaves at the fault: its breaker is listed, at 0 kA."""
    request = _with(_drawn_request(), _breaker('ql', 'dc_b', 'ld_b', 'load_dc'))
    result = _study(client, quiet, request, duration_ms=5, fault_bus='dc_b')
    (brk,) = _fault(result, 'cell-dc_b')['breakers']
    assert brk['label'] == 'QL' and brk['i_open_ka'] == 0 and not brk['exceeds']
    assert any('three AC periods' in w for w in result['warnings'])
