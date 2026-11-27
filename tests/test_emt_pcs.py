"""
The PCS in the EMT study: two grid-forming PCS islanded from the grid,
sharing its load by their droops, against Simscape (simscape_emt_pcs_benchmark.m)
and the droops' closed form; on the drawn network, a grid-following and a
grid-forming PCS holding their load-flow state, islanding, and feeding an AC
fault at their current limit - the short-circuit studies' current source.
"""
import json
import math
import os

import numpy as np
import pytest

import emt_solver as es
from emt_converters import A120, GridFormingVsc
from test_pcs import _der, _pcs, _two_buses

HERE = os.path.join(os.path.dirname(__file__), 'emt_reference')
with open(os.path.join(HERE, 'emt_pcs_benchmark_params.json')) as fh:
    P = json.load(fh)
W = 2 * math.pi * P['f_hz']


def _steady():
    g = P['grid']
    z = g['v_ll'] ** 2 / g['s_sc']
    r_g = z / math.sqrt(1 + g['xr'] ** 2)
    l_g = g['xr'] * r_g / W
    v_ph = g['v_ll'] / math.sqrt(3)
    zl = abs(v_ph) ** 2 / np.conj((P['load']['p'] + 1j * P['load']['q']) / 3)
    e = math.sqrt(2 / 3) * g['v_ll']
    v = e * zl / (zl + complex(r_g, W * l_g))
    return dict(r_g=r_g, l_g=l_g, r_l=zl.real, l_l=zl.imag / W, e=e, v=v, i=v / zl)


def _island_benchmark():
    """The circuit of emt_pcs_benchmark_params.json with two GridFormingVsc controllers."""
    s = _steady()
    ckt = es.Circuit()
    ang = -np.arange(3) * A120
    pcc, k_g, sw = [], [], []
    for k in range(3):
        g, b = ckt.node(f'breaker {k}'), ckt.node(f'PCC {k}')
        pcc.append(b)
        k_g.append(ckt.add_rl(0, g, s['r_g'], s['l_g'], ac=(s['e'], W, ang[k])))
        sw.append(ckt.add_switch(g, b, closed=True, r_on=P['switch_r_on'], r_off=P['switch_r_off']))
        ckt.add_rl(b, 0, s['r_l'], s['l_l'])
    ctrls = []
    for j, q in enumerate(P['pcs']):
        zb = P['grid']['v_ll'] ** 2 / q['s_rated']
        r, l = q['r_pu'] * zb, q['x_pu'] * zb / W
        k_e = [ckt.add_rl(0, pcc[k], r, l, ac=(abs(s['v']), W, np.angle(s['v']) + ang[k]), controlled=True)
               for k in range(3)]
        # Its DC side, stiff: its link and a source behind a small resistance.
        dc = ckt.node(f'DC link {j}', P['v_dc'])
        ckt.add_c(0, dc, 1.0, w0=-P['v_dc'])
        k_src = ckt.add_isrc(0, dc, i0=0.0)
        k_dc = ckt.add_rl(0, dc, 1e-3, 0.0, e0=P['v_dc'])
        i_max = P['limit_pu'] * math.sqrt(2) * q['s_rated'] / (math.sqrt(3) * P['grid']['v_ll'])
        ctrl = GridFormingVsc.standalone(
            droop_pf=q['droop_pf'], droop_qv=q['droop_qv'], tau_f=P['tau_f'],
            label=f'PCS {j}', w0=W, ac_nodes=pcc, k_e=k_e, k_block=[], k_src=k_src, k_dc_out=k_dc, p_node=dc,
            r=r, l=l, i_max=i_max, v_nom_peak=math.sqrt(2 / 3) * P['grid']['v_ll'], block_v=0.0, block_i=1e12,
            mode_dc='p_mw', mode_ac='q_mvar', s_rated=q['s_rated'],
            v_ph=s['v'], i_ph=0j, e_ph=s['v'], c_link=1.0, v_link=P['v_dc'], i_dc=0.0)
        ctrls.append((ctrl, k_e))
    state = {'opened': [None] * 3, 'last': [None] * 3}

    def breaker(t, st):
        # Each phase opens as its current passes zero after the islanding time - as AcBuilder.island_control.
        if t < P['t_island'] - 1e-12:
            return False
        changed = False
        for k in range(3):
            if state['opened'][k] is not None:
                continue
            i = float(st.i_sw[sw[k]])
            last = state['last'][k]
            state['last'][k] = i
            if last is not None and (i == 0.0 or (i > 0) != (last > 0)):
                st.set_switch(sw[k], False)
                state['opened'][k] = t
                changed = True
        return changed

    ckt.add_controller(breaker)
    for ctrl, _ in ctrls:
        ckt.add_controller(ctrl.control)
    ckt.start_in_ac_steady_state(W)
    return ckt, ctrls, pcc, k_g, state


def test_grid_forming_pcs_islanded_against_simscape():
    """
    The grid's breaker opens at 20 ms, each phase at its current's zero; the
    two grid-forming PCS take the load. Their power through the run within 1 %
    of their rating of Simscape's, the PCC voltage within 0.5 %; at the end
    they share it 1 : 2 - their S_rated / droop - at one frequency,
    f0 (1 - droop P / S_rated).
    """
    ckt, ctrls, pcc, k_g, state = _island_benchmark()
    sim = ckt.simulate(P['t_end'], P['dt'])
    ref = np.genfromtxt(os.path.join(HERE, 'emt_pcs_island.csv'), delimiter=',', names=True)
    t = ref['t']
    for j, (ctrl, k_e) in enumerate(ctrls):
        p = np.sum(sim['v'][:, pcc] * sim['i_rl'][:, k_e], axis=1)
        got = np.interp(t, sim['t'], p)
        assert np.max(np.abs(got - ref[f'p{j + 1}'])) < 0.01 * P['pcs'][j]['s_rated'], j
    v_a = np.interp(t, sim['t'], sim['v'][:, pcc[0]])
    assert np.max(np.abs(v_a - ref['v_a'])) < 0.005 * math.sqrt(2 / 3) * P['grid']['v_ll']
    assert all(x is not None for x in state['opened'])
    # The droops' closed form at the end.
    (a, _), (b, _) = ctrls
    assert b.p_f == pytest.approx(2 * a.p_f, rel=1e-3)
    for ctrl, q in ((a, P['pcs'][0]), (b, P['pcs'][1])):
        f = ctrl.freq_trace[-1][1]
        assert f == pytest.approx(P['f_hz'] * (1 - q['droop_pf'] * ctrl.p_f / q['s_rated']), rel=1e-6)
    assert a.freq_trace[-1][1] == pytest.approx(b.freq_trace[-1][1], abs=1e-6)


# --- On the drawn network ------------------------------------------------------------------------

def _emt(**kw):
    return {'typ': 'EmtStudy Parameters', 'user_email': 't@t', **{k: str(v) for k, v in kw.items()}}


def _post(client, quiet, request):
    with quiet():
        result = json.loads(client.post('/', json=request).get_data(as_text=True))
    assert 'emt' in result, result.get('message')
    return result['emt']


def test_pcs_hold_their_load_flow_state(client, quiet):
    """A grid-following battery and PV array and a grid-forming battery start where the load flow left them."""
    emt = _post(client, quiet, _two_buses(
        _der('Battery', 'b1', capacity_kwh=500), _pcs('p1', 'b', 'b1', p_set_mw=0.2, q_mode='pf', pf=0.95),
        _der('PV Array', 'pv'), _pcs('p2', 'b', 'pv', s_rated_mva=0.2),
        _der('Battery', 'b2', capacity_kwh=1000), _pcs('p3', 'a', 'b2', control='grid_forming', s_rated_mva=1.0,
                                                       p_set_mw=0.1),
        params=_emt(time_step_us=20, duration_ms=60)))
    conv = {c['label']: c for c in emt['converters']}
    assert conv['P1']['control'] == 'grid_following' and conv['P3']['control'] == 'grid_forming'
    assert conv['P1']['p_end_mw'] == pytest.approx(0.2, rel=0.01)
    assert conv['P1']['q_end_mvar'] == pytest.approx(0.2 * math.tan(math.acos(0.95)), rel=0.02)
    assert conv['P3']['p_end_mw'] == pytest.approx(0.1, rel=0.01)
    assert conv['P3']['f_end_hz'] == pytest.approx(50.0, abs=0.01)
    assert not emt['converters_blocked']
    ders = {d['label']: d for d in emt['ders']}
    assert ders['B1']['coupling'] == 'pcs' and ders['B1']['p_end_mw'] == pytest.approx(0.2 / 0.98, rel=0.02)
    assert ders['PV']['p_end_mw'] == pytest.approx(ders['PV']['p_start_mw'], rel=0.01)


def test_islanding_on_the_drawn_network(client, quiet):
    """Islanded at 20 ms, two grid-forming PCS (0.5 and 1.0 MVA) share the load 1 : 2 at one frequency."""
    emt = _post(client, quiet, _two_buses(
        _der('Battery', 'ba', capacity_kwh=1000), _pcs('ga', 'a', 'ba', control='grid_forming', s_rated_mva=0.5),
        _der('Battery', 'bb', capacity_kwh=1000), _pcs('gb', 'b', 'bb', control='grid_forming', s_rated_mva=1.0),
        params=_emt(time_step_us=20, duration_ms=400, island_time_ms=20)))
    conv = {c['label']: c for c in emt['converters']}
    ga, gb = conv['GA'], conv['GB']
    assert ga['p_start_mw'] == pytest.approx(0, abs=1e-3)
    assert gb['p_end_mw'] == pytest.approx(2 * ga['p_end_mw'], rel=0.01)
    f = 50 * (1 - 0.02 * gb['p_end_mw'] / 1.0)
    assert ga['f_end_hz'] == pytest.approx(f, abs=0.005) and gb['f_end_hz'] == pytest.approx(f, abs=0.005)
    assert all(x is not None and x >= 20.0 for x in emt['island']['opened_ms'])
    assert not emt['converters_blocked']


@pytest.mark.parametrize('control', ['grid_following', 'grid_forming'])
def test_pcs_feeds_an_ac_fault_at_its_current_limit(client, quiet, control):
    """
    A three-phase fault at its bus: a 0.5 MVA PCS feeds 1.2 x its rated current,
    0.866 kA - the current source the IEC and ANSI studies take it as - after
    its controls catch it.
    """
    emt = _post(client, quiet, _two_buses(
        _der('Battery', 'b1', capacity_kwh=1000),
        _pcs('p1', 'b', 'b1', control=control, s_rated_mva=0.5, p_set_mw=0.2, current_limit_pu=1.2),
        params=_emt(time_step_us=10, duration_ms=100, ac_fault_bus='b', ac_fault_type='abcg', ac_fault_time_ms=20,
                    ac_fault_duration_ms=0, ac_fault_resistance_ohm=0.001)))
    (pcs,) = [c for c in emt['converters'] if c['label'] == 'P1']
    i_rated = 0.5 / (math.sqrt(3) * 0.4)
    assert pcs['i_end_ka'] == pytest.approx(1.2 * i_rated, rel=0.03)
    assert pcs['limited_ms'] > 50
