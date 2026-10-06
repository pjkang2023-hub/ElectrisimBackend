"""
Converters in the EMT study: the VSC, average-value or switching, with its
controls - against Simscape through a DC load step in both models, the same VSC
in both models against each other, and on the drawn radial grid: holding its
load-flow state, holding its DC voltage through a DC load step, and limiting
its current, then blocking, through a fault on its AC bus.
"""
import json
import math
import os

import numpy as np
import pytest

import emt_solver as es
from emt_converters import A120, Vsc
from test_dc_elements import _drawn_request
from test_emt import _request

HERE = os.path.join(os.path.dirname(__file__), 'emt_reference')
with open(os.path.join(HERE, 'emt_vsc_benchmark_params.json')) as fh:
    P = json.load(fh)
W = 2 * math.pi * P['f_hz']


# --- Against Simscape ---------------------------------------------------------------

def _steady(p):
    """
    The benchmark's steady state, as simscape_emt_vsc_benchmark.m computes it:
    its DC load's current, then its AC bus voltage V and current Ic - in phase
    with V, Q = 0 - by fixed point; amplitude phasors of phase a.
    """
    g, q, ld = p['grid'], p['vsc'], p['load']
    z = g['v_ll'] ** 2 / g['s_sc']
    r_g = z / math.sqrt(1 + g['xr'] ** 2)
    x_g = g['xr'] * r_g
    zg = r_g + 1j * x_g * g['r_p'] / (g['r_p'] + 1j * x_g)
    e = math.sqrt(2 / 3) * g['v_ll']
    i_dc = q['v_dc'] / (q['r_dc'] + ld['r_load'])
    p_dc = q['v_dc'] * i_dc
    v = complex(e)
    for _ in range(200):
        a = abs(v)
        x = (-a + math.sqrt(a * a - 4 * q['r'] * p_dc / 1.5)) / (2 * q['r'])
        ic = x * v / a
        v = e - zg * (1j * W * p['c_f'] * v - ic)
    return dict(r_g=r_g, l_g=x_g / W, e=e, v=v, ic=ic, ec=v + complex(q['r'], W * q['l']) * ic, i_dc=i_dc,
                ig=(e - v) / zg)


def _vsc_benchmark():
    """The circuit of emt_vsc_benchmark_params.json, in its steady state, with Vsc's controller."""
    s = _steady(P)
    g, q, ld = P['grid'], P['vsc'], P['load']
    ckt = es.Circuit()
    ang = -np.arange(3) * A120
    pcc, k_e = [], []
    for k in range(3):
        a, b = ckt.node(f'grid {k}'), ckt.node(f'AC bus {k}')
        pcc.append(b)
        ckt.add_rl(0, a, s['r_g'], 0.0, ac=(s['e'], W, ang[k]))
        ckt.add_rl(a, b, 0.0, s['l_g'])
        ckt.add_r(a, b, g['r_p'])
        ckt.add_c(0, b, P['c_f'])
        k_e.append(ckt.add_rl(0, b, q['r'], q['l'], ac=(abs(s['ec']), W, np.angle(s['ec']) + ang[k]), controlled=True))
    v_dc = q['v_dc']
    dc = ckt.node('DC link', v_dc)
    c_link = ckt.add_c(0, dc, q['c_link'], w0=-v_dc)
    k_src = ckt.add_isrc(0, dc, i0=s['i_dc'])
    n = ckt.node('DC load', ld['r_load'] * s['i_dc'])
    k_dc = ckt.add_rl(dc, n, q['r_dc'], q['l_dc'], i0=s['i_dc'])
    ckt.add_r(n, 0, ld['r_load'])
    m = ckt.node('load step')
    sw = ckt.add_switch(n, m, closed=False, r_on=P['switch_r_on'], r_off=P['switch_r_off'])
    ckt.add_r(m, 0, ld['r_step'])
    ckt.at(ld['t_step'], lambda st: st.set_switch(sw, True))
    ckt.start_in_ac_steady_state(W)
    v_ll = g['v_ll']
    i_max = q['i_limit_pu'] * math.sqrt(2) * q['s_rated'] / (math.sqrt(3) * v_ll)
    ctrl = Vsc.standalone(
        label='VSC', w0=W, ac_nodes=pcc, k_e=k_e, k_block=[], k_src=k_src, k_dc_out=k_dc, p_node=dc,
        r=q['r'], l=q['l'], i_max=i_max, v_nom_peak=math.sqrt(2 / 3) * v_ll, block_v=q['block_pu'] * v_dc,
        block_i=2.5 * i_max, mode_dc='vm_pu', mode_ac='q_mvar', s_rated=q['s_rated'],
        v_ph=s['v'], i_ph=s['ic'], e_ph=s['ec'], c_link=q['c_link'], v_link=v_dc, i_dc=s['i_dc'])
    ckt.add_controller(ctrl.control)
    return ckt, ctrl, pcc, k_e, dc


def _vsc_floating(model):
    """
    The benchmark's circuit for the switching model, and for the same VSC
    averaged: the grid's and filter capacitors' star point floating, held by
    r_bias to the DC negative pole (the reference) - a grounded star would carry
    the bridge's common-mode voltage as current - the controller sampling at
    the carrier's peaks and valleys.
    """
    s = _steady(P)
    g, q, ld, sw_ = P['grid'], P['vsc'], P['load'], P['switching']
    ckt = es.Circuit()
    ang = -np.arange(3) * A120
    star = ckt.node('star point')
    ckt.add_r(star, 0, sw_['r_bias'])
    v_dc = q['v_dc']
    dc = ckt.node('DC link', v_dc)
    ph = lambda z, k: (z * np.exp(1j * ang[k])).real
    pcc, k_e, legs = [], [], []
    for k in range(3):
        a, b = ckt.node(f'grid {k}'), ckt.node(f'AC bus {k}')
        pcc.append(b)
        ckt.add_rl(star, a, s['r_g'], 0.0, ac=(s['e'], W, ang[k]))
        ckt.add_rl(a, b, 0.0, s['l_g'], i0=ph(s['ig'], k))
        ckt.add_r(a, b, g['r_p'])
        ckt.add_c(star, b, P['c_f'], w0=-ph(s['v'], k))
        if model == 'switching':
            leg = ckt.node(f'leg {k}', v_dc / 2)
            k_e.append(ckt.add_rl(leg, b, q['r'], q['l'], i0=ph(s['ic'], k)))
            legs.append((ckt.add_switch(leg, dc, closed=False, r_on=P['switch_r_on'], r_off=P['switch_r_off']),
                         ckt.add_switch(leg, 0, closed=False, r_on=P['switch_r_on'], r_off=P['switch_r_off'])))
        else:
            k_e.append(ckt.add_rl(star, b, q['r'], q['l'], i0=ph(s['ic'], k), controlled=True))
    ckt.add_c(0, dc, q['c_link'], w0=-v_dc)
    k_src = ckt.add_isrc(0, dc, i0=0.0 if model == 'switching' else s['i_dc'])
    n = ckt.node('DC load', ld['r_load'] * s['i_dc'])
    k_dc = ckt.add_rl(dc, n, q['r_dc'], q['l_dc'], i0=s['i_dc'])
    ckt.add_r(n, 0, ld['r_load'])
    m = ckt.node('load step')
    sw = ckt.add_switch(n, m, closed=False, r_on=P['switch_r_on'], r_off=P['switch_r_off'])
    ckt.add_r(m, 0, ld['r_step'])
    ckt.at(ld['t_step'], lambda st: st.set_switch(sw, True))
    v_ll = g['v_ll']
    i_max = q['i_limit_pu'] * math.sqrt(2) * q['s_rated'] / (math.sqrt(3) * v_ll)
    ctrl = Vsc.standalone(
        label='VSC', w0=W, ac_nodes=pcc, k_e=k_e, k_src=k_src, k_dc_out=k_dc, p_node=dc,
        r=q['r'], l=q['l'], i_max=i_max, v_nom_peak=math.sqrt(2 / 3) * v_ll, block_v=q['block_pu'] * v_dc,
        block_i=2.5 * i_max, mode_dc='vm_pu', mode_ac='q_mvar', s_rated=q['s_rated'],
        model=model, t_sample=0.5 / sw_['f_sw'], legs=legs,
        v_ph=s['v'], i_ph=s['ic'], e_ph=s['ec'], c_link=q['c_link'], v_link=v_dc, i_dc=s['i_dc'])
    ctrl.prime(ckt, s['ec'], v_dc)
    ckt.add_controller(ctrl.control)
    return ckt, ctrl, pcc, k_e, dc, star


def _i_rated_peak():
    q = P['vsc']
    return math.sqrt(2) * q['s_rated'] / (math.sqrt(3) * P['grid']['v_ll'])


def test_switching_vsc_matches_simscape():
    """
    The same rectifier and load step as a two-level switching bridge at 5 kHz,
    against Simscape's switches gated by the same carrier (exact switching
    instants, as Electrisim lands on them): its DC voltage within 0.4 V,
    ripple included, its phase current within 1 % of its rated peak, and the
    power into its AC bus, over each 10 ms, within 0.1 % of its rating.
    """
    sw_ = P['switching']
    ckt, ctrl, pcc, k_e, dc, star = _vsc_floating('switching')
    sim = ckt.simulate(sw_['t_end'], sw_['dt'])
    ref = np.genfromtxt(os.path.join(HERE, 'emt_vsc_switching.csv'), delimiter=',', names=True)
    keep = ref['t'] > 0                                   # t = 0: the solver's node voltages are its first guess
    t = ref['t'][keep]
    v_dc = np.interp(t, sim['t'], sim['v'][:, dc])
    i_a = np.interp(t, sim['t'], sim['i_rl'][:, k_e[0]])
    p_ac = np.interp(t, sim['t'], np.sum((sim['v'][:, pcc] - sim['v'][:, [star]]) * sim['i_rl'][:, k_e], axis=1))
    q = P['vsc']
    assert np.max(np.abs(v_dc - ref['v_dc'][keep])) < 0.4
    assert np.max(np.abs(i_a - ref['i_a'][keep])) < 0.01 * _i_rated_peak()
    for a in np.arange(0.0, sw_['t_end'] - 1e-9, 0.01):
        sel = (t > a) & (t <= a + 0.01)
        assert abs(np.mean(p_ac[sel]) - np.mean(ref['p_ac'][keep][sel])) < 1e-3 * q['s_rated']
    assert np.min(ref['v_dc']) < 0.98 * q['v_dc'] and ctrl.limited_time > 0 and ctrl.blocked_at is None


def test_vsc_the_same_in_both_models():
    """
    The same VSC, average-value and switching, through the load step: sampling
    alike, the average-value model is the switching model's average over each
    half carrier period, so their controllers' samples - power, reactive power,
    DC voltage (its mean since the last sample), current - agree within 1 % of
    its rating, 0.25 % of its DC voltage and 1 % of its rated current.
    """
    sw_ = P['switching']
    traces = {}
    for model in ('average', 'switching'):
        ckt, ctrl, *_ = _vsc_floating(model)
        ckt.simulate(sw_['t_end'], sw_['dt'])
        traces[model] = np.array(ctrl.trace)
    a, s = traces['average'], traces['switching']
    q = P['vsc']
    assert a.shape == s.shape and np.allclose(a[:, 0], s[:, 0], rtol=0, atol=1e-9)
    assert np.max(np.abs(a[:, 1] - s[:, 1])) < 0.01 * q['s_rated']
    assert np.max(np.abs(a[:, 2] - s[:, 2])) < 0.01 * q['s_rated']
    assert np.max(np.abs(a[:, 3] - s[:, 3])) < 2.5e-3 * q['v_dc']
    assert np.max(np.abs(a[:, 4] - s[:, 4])) < 0.01 * _i_rated_peak()


def test_vsc_load_step_matches_simscape():
    """
    The rectifier holding its DC link at 800 V while its DC load steps up 50 %,
    against the same circuit and controller in Simscape (the controller in a
    MATLAB Function block): it starts steady, its DC voltage dips to 0.97 pu
    while its current limit holds it for a moment, and recovers. The DC
    voltage within 0.05 % of its set point throughout, the power into its AC
    bus within 0.5 % of its rating.
    """
    ckt, ctrl, pcc, k_e, dc = _vsc_benchmark()
    sim = ckt.simulate(P['t_end'], P['dt'])
    ref = np.genfromtxt(os.path.join(HERE, 'emt_vsc_load_step.csv'), delimiter=',', names=True)
    t = ref['t']
    v_dc = np.interp(t, sim['t'], sim['v'][:, dc])
    p_ac = np.interp(t, sim['t'], np.sum(sim['v'][:, pcc] * sim['i_rl'][:, k_e], axis=1))
    q = P['vsc']
    assert np.max(np.abs(v_dc - ref['v_dc'])) < 5e-4 * q['v_dc']
    assert np.max(np.abs(p_ac - ref['p_ac'])) < 5e-3 * q['s_rated']
    assert np.min(ref['v_dc']) < 0.98 * q['v_dc']                   # the step is there to see
    assert ctrl.limited_time > 0 and ctrl.blocked_at is None


# --- The study on the drawn radial grid ----------------------------------------------


def _run(client, quiet, **params):
    request = _request(**{'time_step_us': '5', 'duration_ms': '60', **{k: str(v) for k, v in params.items()}})
    with quiet():
        response = client.post('/', json=request)
    result = json.loads(response.get_data(as_text=True))
    assert not result.get('error'), result.get('message')
    return result


def _vsc(result):
    (conv,) = result['emt']['converters']
    return conv


def _switching(request, **vsc):
    for el in request.values():
        if isinstance(el, dict) and str(el.get('typ', '')).startswith('VSC'):
            el.update(emt_model='switching', **vsc)
    return request


def _dc_bus(result, label):
    return next(b for b in result['emt']['buses'] if b['label'] == label)


def test_vsc_holds_its_load_flow_state(client, quiet):
    """
    With nothing happening, the rectifier holds DC bus A at its set point and
    draws its load-flow power - 0.150 MW for the DC loads and the cable - at
    unity power factor, as the load flow had it.
    """
    result = _run(client, quiet)
    conv = _vsc(result)
    assert conv['model'] == 'average' and conv['blocked_ms'] is None and conv['limited_ms'] == 0
    assert conv['p_end_mw'] == pytest.approx(-0.1503, rel=0.01)       # drawn from the AC grid
    assert abs(conv['q_end_mvar']) < 0.001 * conv['rated_mva'] * 10
    assert conv['v_dc_end_kv'] == pytest.approx(0.8, rel=1e-3)
    bus_a = _dc_bus(result, 'DC bus A')
    assert bus_a['v_min_pu'] > 0.999 and bus_a['v_max_pu'] < 1.001


def test_vsc_holds_its_dc_voltage_through_a_load_step(client, quiet):
    """
    Server hall B steps up 50 % at 10 ms: DC bus A dips while the rectifier's
    DC voltage loop catches up, and is back at its set point by the end; the
    rectifier draws the extra 50 kW.
    """
    result = _run(client, quiet, step_load='cell-ld_b', step_percent=50, step_time_ms=10, duration_ms=100)
    conv = _vsc(result)
    bus_a = _dc_bus(result, 'DC bus A')
    assert 0.9 < bus_a['v_min_pu'] < 0.999
    assert bus_a['v_final_pu'] == pytest.approx(1.0, abs=2e-3)
    assert -conv['p_end_mw'] == pytest.approx(0.1503 + 0.05, rel=0.02)
    assert conv['blocked_ms'] is None


def test_vsc_limits_its_current_then_blocks_through_an_ac_fault(client, quiet):
    """
    A bolted three-phase fault on LV network A, the rectifier's AC bus, at
    20 ms: it cannot draw power, so it holds its current at its limit while
    its DC link feeds the DC loads and sags - and blocks once its DC voltage
    falls below 0.8 pu. Its current overshoots its limit at the fault's onset,
    until its controller's next sample (every 100 us at 5 kHz), but stays
    below the 2.5 times its limit that would block it.
    """
    request = _drawn_request()
    lv = next(v['name'] for v in request.values() if isinstance(v, dict) and v.get('userFriendlyName') == 'LV network A')
    result = _run(client, quiet, ac_fault_bus=lv, ac_fault_type='abcg', ac_fault_time_ms=20, ac_fault_duration_ms=0,
                  ac_fault_resistance_ohm=0.001, duration_ms=60)
    conv = _vsc(result)
    assert conv['limited_ms'] > 0
    assert conv['i_peak_ka'] < 2.5 * conv['current_limit_ka']
    assert conv['blocked_ms'] is not None and conv['blocked_ms'] > 20
    assert _dc_bus(result, 'DC bus A')['v_min_pu'] < 0.8


def test_switching_vsc_holds_its_load_flow_state(client, quiet):
    """
    The rectifier as a switching bridge at 5 kHz, behind a 0.12 pu reactor:
    with nothing happening it holds DC bus A within 0.5 % - its switching
    ripple - and draws its load-flow power within 5 %.
    """
    request = _switching(_request(time_step_us='2', duration_ms='20'), x_ohm='0.1')
    with quiet():
        result = json.loads(client.post('/', json=request).get_data(as_text=True))
    assert not result.get('error'), result.get('message')
    conv = _vsc(result)
    assert conv['model'] == 'switching' and conv['switching_khz'] == 5 and conv['blocked_ms'] is None
    assert -conv['p_end_mw'] == pytest.approx(0.1503, rel=0.05)
    bus_a = _dc_bus(result, 'DC bus A')
    assert 0.995 < bus_a['v_min_pu'] and bus_a['v_max_pu'] < 1.005
    assert not any('current ripple' in w for w in result['warnings'])


def test_switching_vsc_with_too_small_a_reactor_is_warned_about(client, quiet):
    """The drawn grid's VSC reactor, 0.012 pu, leaves a ripple far above its rated current at 5 kHz."""
    request = _switching(_request(time_step_us='2', duration_ms='1'))
    with quiet():
        result = json.loads(client.post('/', json=request).get_data(as_text=True))
    assert any('current ripple' in w and 'Rectifier' in w for w in result['warnings'])
