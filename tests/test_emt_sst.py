"""
Solid-state transformers and DC/DC converters in the EMT study: an SST through
a fault on its LV DC port against Simscape; its DC/DC stage - a dual active
bridge - in both models against each other; and on the drawn radial grid, an
SST holding its load-flow state, and riding a fault on its LV DC port.
"""
import json
import math
import os
import types

import numpy as np
import pytest

import emt_solver as es
from emt_converters import A120, DcDc, Vsc, stage_input
from test_dc_elements import _drawn_request
from test_sst import LV_DC, LV_DC_LOAD, LV_NETWORK_A, _with_sst

HERE = os.path.join(os.path.dirname(__file__), 'emt_reference')
with open(os.path.join(HERE, 'emt_sst_benchmark_params.json')) as fh:
    P = json.load(fh)
W = 2 * math.pi * P['f_hz']


# --- Against Simscape ---------------------------------------------------------------

def _steady(p):
    """
    The benchmark's steady state, as simscape_emt_sst_benchmark.m computes it:
    the LV load's current from the bridge's output capacitor at v_out; the
    bridge's input power by its efficiency; the link bus's voltage behind
    r_dc; then the rectifier's AC bus voltage V and current Ic - in phase with
    V, Q = 0 - by fixed point; amplitude phasors of phase a.
    """
    g, rc, d, ld = p['grid'], p['rectifier'], p['dab'], p['load']
    i_o = d['v_out'] / (ld['r_load'] + d['r_out'])
    p_out = d['v_out'] * i_o
    p_in = stage_input(p_out, d['eta'], 0.0)
    v_bus = rc['v_dc']
    for _ in range(50):
        i_dc = p_in / v_bus
        v_bus = rc['v_dc'] - rc['r_dc'] * i_dc
    z = g['v_ll'] ** 2 / g['s_sc']
    r_g = z / math.sqrt(1 + g['xr'] ** 2)
    x_g = g['xr'] * r_g
    zg = r_g + 1j * x_g * g['r_p'] / (g['r_p'] + 1j * x_g)
    e = math.sqrt(2 / 3) * g['v_ll']
    p_dc = rc['v_dc'] * i_dc
    v = complex(e)
    for _ in range(200):
        a = abs(v)
        x = (-a + math.sqrt(a * a - 4 * rc['r'] * p_dc / 1.5)) / (2 * rc['r'])
        ic = x * v / a
        v = e - zg * (1j * W * g['c_f'] * v - ic)
    return dict(r_g=r_g, l_g=x_g / W, e=e, v=v, ic=ic, ec=v + complex(rc['r'], W * rc['l']) * ic, i_dc=i_dc,
                v_link=v_bus, i_o=i_o, p_out=p_out, p_in=p_in, v_lv=d['v_out'] - d['r_out'] * i_o)


def _sst_benchmark(case, model='average', every_step=True):
    """
    The circuit of emt_sst_benchmark_params.json, in its steady state, with
    Electrisim's controllers: acting every step (as the Simscape benchmark's),
    or sampling twice per carrier period (as in the study), the bridge then
    average-value or switching.
    """
    s = _steady(P)
    g, rc, d, ld = P['grid'], P['rectifier'], P['dab'], P['load']
    c = P['cases'][case]
    ckt = es.Circuit()
    ang = -np.arange(3) * A120
    pcc, k_e = [], []
    for k in range(3):
        a, b = ckt.node(f'grid {k}'), ckt.node(f'AC bus {k}')
        pcc.append(b)
        ckt.add_rl(0, a, s['r_g'], 0.0, ac=(s['e'], W, ang[k]))
        ckt.add_rl(a, b, 0.0, s['l_g'])
        ckt.add_r(a, b, g['r_p'])
        ckt.add_c(0, b, g['c_f'])
        k_e.append(ckt.add_rl(0, b, rc['r'], rc['l'], ac=(abs(s['ec']), W, np.angle(s['ec']) + ang[k]), controlled=True))
    dc = ckt.node('rectifier DC+', rc['v_dc'])
    ckt.add_c(0, dc, rc['c_link'], w0=-rc['v_dc'])
    k_src = ckt.add_isrc(0, dc, i0=s['i_dc'])
    link = ckt.node('link bus', s['v_link'])
    k_dc = ckt.add_rl(dc, link, rc['r_dc'], 0.0, i0=s['i_dc'])
    lv = ckt.node('LV DC bus', s['v_lv'])
    ckt.add_r(lv, 0, ld['r_load'])
    f = ckt.node('fault')
    k_f = ckt.add_switch(lv, f, closed=False, r_on=P['switch_r_on'], r_off=P['switch_r_off'])
    k_rf = ckt.add_r(f, 0, c['r_fault'])
    ckt.at(c['t_on'], lambda st: st.set_switch(k_f, True))
    if c['t_off'] is not None:
        ckt.at(c['t_off'], lambda st: st.set_switch(k_f, False))
    builder = types.SimpleNamespace(ckt=ckt, vn={'link': rc['v_dc'], 'lv': d['v_out']},
                                    v_bus={'link': s['v_link'], 'lv': d['v_out']}, warnings=[])
    dab = DcDc(builder, 'DAB', 'link', 'lv', link, lv, p_in_mw=s['p_in'] / 1e6, p_out_mw=s['p_out'] / 1e6,
               mode='voltage', vm_out_pu=1.0, rated_mw=d['p_rated'] / 1e6, eta=d['eta'], bidirectional=True,
               limit_pu=d['i_limit_pu'], model=model, switching_khz=d['f_sw'] / 1e3, block_pu=d['block_pu'],
               r_in=d['r_in'], r_out=d['r_out'], every_step=every_step)
    ckt.start_in_ac_steady_state(W)
    v_ll = g['v_ll']
    i_max = rc['i_limit_pu'] * math.sqrt(2) * rc['s_rated'] / (math.sqrt(3) * v_ll)
    rect = Vsc.standalone(
        label='rectifier', w0=W, ac_nodes=pcc, k_e=k_e, k_block=[], k_src=k_src, k_dc_out=k_dc, p_node=dc,
        r=rc['r'], l=rc['l'], i_max=i_max, v_nom_peak=math.sqrt(2 / 3) * v_ll, block_v=rc['block_pu'] * rc['v_dc'],
        block_i=2.5 * i_max, mode_dc='vm_pu', mode_ac='q_mvar', s_rated=rc['s_rated'],
        v_ph=s['v'], i_ph=s['ic'], e_ph=s['ec'], c_link=rc['c_link'], v_link=rc['v_dc'], i_dc=s['i_dc'],
        t_sample=None if every_step else 0.5 / rc['f_sw'])
    rect.prime(ckt, s['ec'], rc['v_dc'])
    ckt.add_controller(rect.control)
    ckt.add_controller(dab.control)
    return ckt, rect, dab, dict(dc=dc, link=link, lv=lv, k_rf=k_rf)


def _keep(t, c):
    """Not the samples at the fault's switching: there the two tools' waveforms jump between neighbouring samples."""
    keep = np.ones_like(t, dtype=bool)
    for te in (c['t_on'], c['t_off']):
        if te is not None:
            keep &= np.abs(t - te) > 1.5 * P['dt']
    return keep


@pytest.mark.parametrize('case', ['limited', 'bolted'])
def test_sst_lv_dc_fault_matches_simscape(case):
    """
    The SST through a fault on its LV DC port, against the same circuit and
    controllers in Simscape. 'limited': a 0.4 ohm fault for 20 ms - its DC/DC
    stage holds its current at its limit, the port at some 0.92 pu, its
    rectifier at its own limit too, the link sagging 6 %; both recover when it
    clears. 'bolted': a 1 mohm fault - its DC/DC stage blocks within a step,
    its output capacitor emptying into the fault; the rectifier holds its link.
    The port's and link's voltages, the stage's and the fault's currents
    within 0.1 % of their scale ('bolted': 2 % of its port's voltage, and of
    the fault's peak, which a 5 us step resolves only in part - its output
    capacitor's 25 us discharge).
    """
    c = P['cases'][case]
    ckt, rect, dab, n = _sst_benchmark(case)
    sim = ckt.simulate(c['t_end'], P['dt'])
    ref = np.genfromtxt(os.path.join(HERE, f'emt_sst_{case}.csv'), delimiter=',', names=True)
    t = ref['t']
    keep = _keep(t, c)
    mine = {'v_out': sim['v'][:, dab.out_node], 'v_link': sim['v'][:, n['dc']], 'i_dab': sim['i_rl'][:, dab.k_out],
            'i_fault': sim['i_r'][:, n['k_rf']]}
    diff = {k: np.max(np.abs(np.interp(t, sim['t'], y) - ref[k])[keep]) for k, y in mine.items()}
    d, rc = P['dab'], P['rectifier']
    i_max = d['i_limit_pu'] * d['p_rated'] / d['v_out']
    assert diff['v_link'] < 5e-4 * rc['v_dc']
    if case == 'limited':
        assert diff['v_out'] < 1e-3 * d['v_out']
        assert diff['i_dab'] < 1e-3 * i_max and diff['i_fault'] < 1e-3 * i_max
        assert 0.9 * d['v_out'] < np.min(ref['v_out']) < 0.95 * d['v_out']
        assert dab.limited_time == pytest.approx(c['t_off'] - c['t_on'], abs=1e-3) and dab.blocked_at is None
        assert rect.limited_time > 0 and np.interp(c['t_end'], sim['t'], mine['v_out']) == pytest.approx(d['v_out'], abs=1.0)
    else:
        assert diff['v_out'] < 0.02 * d['v_out']
        assert diff['i_fault'] < 0.015 * np.max(ref['i_fault'])
        assert dab.blocked_at is not None and dab.blocked_at <= c['t_on'] + 2 * P['dt']
        assert rect.blocked_at is None


# --- The DC/DC stage in both models --------------------------------------------------

@pytest.fixture(scope='module')
def dab_runs():
    runs = {}
    for model in ('average', 'switching'):
        ckt, rect, dab, n = _sst_benchmark('limited', model=model, every_step=False)
        sim = ckt.simulate(0.04, 1e-6)
        runs[model] = (np.array(dab.trace), sim, dab)
    return runs


def test_dab_the_same_in_both_models(dab_runs):
    """
    The SST's DC/DC stage, average-value and switching (its two bridges at 20
    kHz), through the limited fault, its rectifier sampling as in the study:
    their samples - its port's voltage within 1 V, its link's within 0.5 %, its
    output bridge's current within 5 % of its limit (the gap at the fault's
    onset, the switching model's current reaching its limit within a few
    half periods).
    """
    a, s = dab_runs['average'][0], dab_runs['switching'][0]
    d = P['dab']
    i_max = d['i_limit_pu'] * d['p_rated'] / d['v_out']
    assert a.shape == s.shape and np.allclose(a[:, 0], s[:, 0], rtol=0, atol=1e-9)
    assert np.max(np.abs(a[:, 4] - s[:, 4])) < 1.0
    assert np.max(np.abs(a[:, 3] - s[:, 3])) < 5e-3 * P['rectifier']['v_dc']
    assert np.max(np.abs(a[:, 5] - s[:, 5])) < 0.05 * i_max


def test_switching_dab_passes_its_power_without_numerical_losses(dab_runs):
    """
    Its switching model's power at its terminals before the fault: what its
    input takes is what its output gives plus its losses (2 %), within 0.1 %
    of its rating - the solver's damped steps after each of its 80 000 edges a
    second would otherwise lose some per cent.
    """
    _, sim, dab = dab_runs['switching']
    t = sim['t']
    sel = (t >= 0.004) & (t < 0.0095)
    w = np.diff(t, prepend=t[0])[sel]

    def mean(y):
        return float(np.sum(y[sel] * w) / np.sum(w))

    v_in, v_out = sim['v'][:, dab.in_node], sim['v'][:, dab.out_node]
    p_in = mean(v_in * sim['i_rl'][:, dab.k_in])
    p_out = mean(v_out * sim['i_rl'][:, dab.k_out])
    p_loss = mean(v_in * sim['i_src'][:, dab.k_src_in])
    assert p_loss == pytest.approx(0.02 / 0.98 * p_out, rel=0.02)
    assert abs(p_in - p_loss - p_out) < 1e-3 * P['dab']['p_rated']


# --- The study on the drawn radial grid ----------------------------------------------

def _sst_request(model='average', **params):
    request = _with_sst(_drawn_request(), LV_DC, dict(LV_DC_LOAD, filter_c_uf='20000', filter_l_mh='0.001'),
                        lv_ac=LV_NETWORK_A, p_ac_mw=0.2, q_ac_mvar=0.0, emt_model=model)
    return _emt(request, **params)


def _emt(request, **params):
    key = next(k for k, v in request.items() if isinstance(v, dict) and 'Parameters' in str(v.get('typ', '')))
    request[key] = {'typ': 'EmtStudy Parameters', 'user_email': 't@t', 'time_step_us': '2', 'duration_ms': '20',
                    **{k: str(v) for k, v in params.items()}}
    return request


def _post(client, quiet, request):
    with quiet():
        result = json.loads(client.post('/', json=request).get_data(as_text=True))
    assert not result.get('error'), result.get('message')
    return result


def _converters(result):
    return {c['label']: c for c in result['emt']['converters']}


def test_sst_holds_its_load_flow_state(client, quiet):
    """
    The SST on bus B2 (20 kV), its 1 MW DC racks and its grid-following
    inverter's 0.2 MW into LV network A: its rectifier holds its 30 kV link
    and draws the load flow's power (its reactor's 0.5 % aside), its DC/DC
    stage holds its LV DC port and delivers the racks' power and the
    inverter's input, its inverter delivers its set power.
    """
    result = _post(client, quiet, _sst_request())
    conv = _converters(result)
    rect, dcdc, inv = conv['SST 1 rectifier'], conv['SST 1 DC/DC'], conv['SST 1 inverter']
    p_dcdc_out = 1.0 + 0.2 / 0.975
    assert rect['kind'] == 'VSC' and dcdc['kind'] == 'DC/DC' and inv['kind'] == 'VSC'
    assert -rect['p_end_mw'] == pytest.approx(p_dcdc_out / 0.98 / 0.985, rel=0.01)
    assert rect['v_dc_end_kv'] == pytest.approx(30.0, rel=1e-3)
    assert dcdc['p_end_mw'] == pytest.approx(p_dcdc_out, rel=2e-3)
    assert inv['p_end_mw'] == pytest.approx(0.2, rel=2e-3)
    lv = next(b for b in result['emt']['buses'] if b['label'] == 'SST LV DC')
    assert 0.999 < lv['v_min_pu'] and lv['v_max_pu'] < 1.001
    assert all(c['blocked_ms'] is None for c in conv.values())


def test_sst_lv_dc_fault(client, quiet):
    """
    A fault on the SST's LV DC port at 5 ms: its DC/DC stage and its inverter
    block on undervoltage within a step - its stage's current within its
    limit until then - and its rectifier holds its link.
    """
    result = _post(client, quiet, _sst_request(fault_bus='sst_dc', fault_time_ms=5, fault_resistance_mohm=5,
                                               duration_ms=15))
    conv = _converters(result)
    assert 5.0 < conv['SST 1 DC/DC']['blocked_ms'] < 5.1 and 5.0 < conv['SST 1 inverter']['blocked_ms'] < 5.1
    assert conv['SST 1 rectifier']['blocked_ms'] is None and conv['SST 1 rectifier']['v_dc_min_kv'] > 0.95 * 30
    assert 0 < conv['SST 1 DC/DC']['i_peak_ka'] <= 1.01 * conv['SST 1 DC/DC']['current_limit_ka']
    assert {c['label'] for c in result['emt']['converters_blocked']} == {'SST 1 DC/DC', 'SST 1 inverter'}


def test_sst_as_switching_models(client, quiet):
    """
    The SST's three stages as switching models - its rectifier and inverter
    at 5 kHz, its DC/DC stage's bridges at 20 kHz - with nothing happening:
    its LV DC port within 0.5 % (its ripple), its stages' powers within 2 % of
    their load-flow values.
    """
    result = _post(client, quiet, _sst_request('switching', time_step_us='1', duration_ms='10'))
    conv = _converters(result)
    assert {c['model'] for c in conv.values() if c['label'].startswith('SST')} == {'switching'}
    assert conv['SST 1 DC/DC']['switching_khz'] == 20 and conv['SST 1 rectifier']['switching_khz'] == 5
    assert conv['SST 1 DC/DC']['p_end_mw'] == pytest.approx(1.0 + 0.2 / 0.975, rel=0.02)
    assert conv['SST 1 inverter']['p_end_mw'] == pytest.approx(0.2, rel=0.02)
    lv = next(b for b in result['emt']['buses'] if b['label'] == 'SST LV DC')
    assert 0.995 < lv['v_min_pu'] and lv['v_max_pu'] < 1.005


@pytest.mark.parametrize('model', ['average', 'switching'])
def test_dc_dc_converter_holds_its_output(client, quiet, model):
    """
    A DC/DC converter from DC bus B (800 V) to a 48 V bus with 20 kW of racks,
    as a dual active bridge: it holds 48 V and delivers the racks' power.
    """
    from test_dc_dc_converter import BUS_48, LOAD_48, _conv
    from test_dc_elements import _with
    request = _emt(_with(_drawn_request(), BUS_48, dict(LOAD_48, filter_c_uf='50000'),
                         _conv('k1', 'dc_b', 'dc_48', emt_model=model)), time_step_us='1', duration_ms='10')
    result = _post(client, quiet, request)
    (k1,) = [c for c in result['emt']['converters'] if c['kind'] == 'DC/DC']
    assert k1['label'] == 'K1' and k1['model'] == model and k1['blocked_ms'] is None
    assert k1['p_end_mw'] == pytest.approx(0.02, rel=0.01)
    bus = next(b for b in result['emt']['buses'] if b['label'] == 'DC 48 V')
    assert 0.995 < bus['v_min_pu'] and bus['v_max_pu'] < 1.005


def test_sst_with_a_grid_forming_inverter(client, quiet):
    """
    The design note's case: the SST's grid-forming inverter holding an
    islanded 0.4 kV network - a source behind its impedance in the study, its
    input a constant-power load - while its rectifier and DC/DC stage are
    modelled; the inverter's AC currents named after it.
    """
    from test_sst import LV_AC, LV_AC_LOAD
    request = _emt(_with_sst(_drawn_request(), LV_DC, dict(LV_DC_LOAD, filter_c_uf='20000', filter_l_mh='0.001'),
                             LV_AC, LV_AC_LOAD, lv_ac='sst_ac', inverter_mode='grid_forming', link_kv=32),
                   duration_ms='10')
    result = _post(client, quiet, request)
    conv = _converters(result)
    assert set(conv) >= {'SST 1 rectifier', 'SST 1 DC/DC'} and 'SST 1 inverter' not in conv
    assert conv['SST 1 DC/DC']['p_end_mw'] == pytest.approx(1.0 + 0.5 / 0.975, rel=0.01)
    labels = {s['label'] for s in result['emt']['ac']['branches']}
    assert 'SST 1 inverter' in labels
    assert not any('sst ' in w for w in result['warnings'])      # its parts named after it, not its cell
