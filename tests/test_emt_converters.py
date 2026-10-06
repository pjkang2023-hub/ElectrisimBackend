"""
Converters in the EMT study: the VSC's average-value model with its controls -
against Simscape through a DC load step, and on the drawn radial grid: holding
its load-flow state, holding its DC voltage through a DC load step, and
limiting its current, then blocking, through a fault on its AC bus.
"""
import json
import math
import os

import numpy as np
import pytest

import emt_solver as es
from emt_converters import A120, VscAverage
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
    return dict(r_g=r_g, l_g=x_g / W, e=e, v=v, ic=ic, ec=v + complex(q['r'], W * q['l']) * ic, i_dc=i_dc)


def _vsc_benchmark():
    """The circuit of emt_vsc_benchmark_params.json, in its steady state, with VscAverage's controller."""
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
    ctrl = VscAverage.standalone(
        label='VSC', w0=W, ac_nodes=pcc, k_e=k_e, k_block=[], k_src=k_src, k_dc_out=k_dc, p_node=dc,
        r=q['r'], l=q['l'], i_max=i_max, v_nom_peak=math.sqrt(2 / 3) * v_ll, block_v=q['block_pu'] * v_dc,
        block_i=2.5 * i_max, mode_dc='vm_pu', mode_ac='q_mvar', s_rated=q['s_rated'],
        v_ph=s['v'], i_ph=s['ic'], e_ph=s['ec'], c_link=q['c_link'], v_link=v_dc, i_dc=s['i_dc'])
    ckt.add_controller(ctrl.control)
    return ckt, ctrl, pcc, k_e, dc


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
    falls below 0.8 pu.
    """
    request = _drawn_request()
    lv = next(v['name'] for v in request.values() if isinstance(v, dict) and v.get('userFriendlyName') == 'LV network A')
    result = _run(client, quiet, ac_fault_bus=lv, ac_fault_type='abcg', ac_fault_time_ms=20, ac_fault_duration_ms=0,
                  ac_fault_resistance_ohm=0.001, duration_ms=60)
    conv = _vsc(result)
    assert conv['limited_ms'] > 0
    assert conv['i_peak_ka'] <= 1.5 * conv['current_limit_ka']
    assert conv['blocked_ms'] is not None and conv['blocked_ms'] > 20
    assert _dc_bus(result, 'DC bus A')['v_min_pu'] < 0.8
