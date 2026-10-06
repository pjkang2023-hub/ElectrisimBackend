"""
The microgrid's sources and stores in the EMT study (emt_der.py), against
Simscape (simscape_emt_der_benchmark.m, from emt_der_benchmark_params.json)
and closed forms: a battery through a current step and its rest, a PV array's
I-V curves and its MPPT through an irradiance step, a supercapacitor smoothing
a rack's cycle on a 48 V and an 800 V bus, a flywheel driven into its power
limit, and the SOFC's partial pressures after a current step.
"""
import csv
import json
import math
import os
import types

import numpy as np
import pytest

import der_electrisim as der
import emt_der
from emt_converters import DcDc
from emt_solver import Circuit, dc_load_current

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, 'emt_reference')
with open(os.path.join(REF, 'emt_der_benchmark_params.json'), encoding='utf-8') as _h:
    P = json.load(_h)


def _csv(name):
    with open(os.path.join(REF, name), encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def _at(t_sim, y_sim, t):
    return np.interp(t, t_sim, y_sim)


# --- Battery ------------------------------------------------------------------------------------

def _battery():
    q = P['battery']
    table = ', '.join(f'{100 * s}:{v}' for s, v in zip(q['soc_table'], q['ocv_cell']))
    return der.build({'typ': 'Battery0', 'name': 'b', 'sizing': 'cells', 'cells_series': q['cells'], 'cell_v': 3.2,
                      'cell_ah': q['ah'], 'cell_r_mohm': 1e3 * q['r0'] / q['cells'],
                      'r1_mohm': 1e3 * q['r1'], 'tau1_s': q['tau1'], 'soc_percent': 100 * q['soc0'],
                      'ocv_table': table, 'soc_min_percent': 0, 'soc_max_percent': 100})


def test_battery_step_and_rest_against_simscape():
    """
    100 A from 1 s to 11 s, then rest: its terminal voltage - R0's step, the RC
    pair's charge and its relaxation after - and its state of charge, as
    Simscape Electrical's table-based battery gives them, and its charge
    counted in closed form.
    """
    q = P['battery']
    obj = _battery()
    assert obj.ocv() == pytest.approx(q['cells'] * 3.31)
    ckt = Circuit()
    node = ckt.node('battery', obj.ocv())
    model = emt_der.BatteryEmt(ckt, obj, 'B', node, obj.ocv(), 0.0)
    k_load = ckt.add_isrc(node, 0, i0=0.0)
    ckt.at(q['t_on'], lambda st: st.set_source_value(k_load, q['i_load']))
    ckt.at(q['t_off'], lambda st: st.set_source_value(k_load, 0.0))
    ckt.add_controller(lambda t, st: model.control(t, st))
    sim = ckt.simulate(q['t_end'], q['dt'])
    ref = _csv('emt_der_battery.csv')
    t = ref['t'][(ref['t'] % 1.0 > 0.02) & (ref['t'] % 1.0 < 0.98)]       # clear of the steps' instants
    v = _at(sim['t'], sim['v'][:, node], t)
    assert np.max(np.abs(v - np.interp(t, ref['t'], ref['v']))) < 2e-4 * q['cells'] * 3.2
    tr = np.array(model.trace)
    soc = _at(tr[:, 0], tr[:, 3], t)
    assert np.max(np.abs(soc - np.interp(t, ref['t'], ref['soc']))) < 1e-6
    # Its charge: 100 A for 10 s out of 100 Ah.
    assert obj.soc0 == pytest.approx(q['soc0'] - q['i_load'] * (q['t_off'] - q['t_on']) / (3600 * q['ah']), abs=1e-9)
    # Its rest: the RC pair's voltage relaxing with tau1 = 2 s.
    v_end = obj.ocv()
    rest = sim['t'] > q['t_off'] + 0.05
    k = np.argmin(np.abs(sim['t'] - (q['t_off'] + q['tau1'])))
    v1_0 = q['r1'] * q['i_load'] * (1 - math.exp(-(q['t_off'] - q['t_on']) / q['tau1']))
    assert v_end - sim['v'][k, node] == pytest.approx(v1_0 * math.exp(-1), rel=2e-3)
    assert rest.any()


# --- PV array -----------------------------------------------------------------------------------

def _pv(**fields):
    q = P['pv']
    return der.build({'typ': 'PV Array0', 'name': 'pv', 'modules_series': q['modules_series'],
                      'strings_parallel': q['strings_parallel'], 'loss_percent': 0, **fields})


@pytest.mark.parametrize('g', P['pv']['irradiances'])
def test_pv_iv_curve_against_simscape(g):
    """Its EMT source's current along its I-V curve, as Simscape's solar cell with the module's fitted values."""
    obj = _pv(irradiance_wm2=g, ambient_c=25 - 25 * g / 800)       # its cells at 25 C
    assert obj.t_cell == pytest.approx(25.0)
    model = emt_der.PvEmt(Circuit(), obj, 'PV', 1, 700.0, 0.0)
    ref = _csv(f'emt_der_pv_{g}.csv')
    for v, i in zip(ref['t'], ref['i']):
        got = -model.current(v, 0.0)[0]
        assert got == pytest.approx(max(i, 0.0), abs=1e-3), v


def test_pv_datasheet_points_at_two_temperatures():
    """At 25 C and 60 C (1000 W/m2): its Isc and Voc move by the datasheet's coefficients."""
    for t_cell in (25.0, 60.0):
        obj = _pv(ambient_c=t_cell - 25 * 1000 / 800)
        d = t_cell - 25
        isc = obj.i_array(0.0) / obj.n_p
        voc = obj.n_s * obj.v_oc_t / obj.n_s
        assert isc == pytest.approx(14.0 * (1 + 0.00048 * d), rel=1e-3)
        assert voc == pytest.approx(49.9 * (1 - 0.0027 * d), rel=5e-3)


def test_mppt_through_an_irradiance_step():
    """
    Behind its converter in MPPT (perturb and observe on its voltage), its
    power at 1000 W/m2, then - after a step to 500 W/m2 at 0.3 s - within 1 %
    of the curve's maximum at each.
    """
    obj = _pv(ambient_c=25.0)
    v_mp, _, p_mp = obj.mpp()
    ckt = Circuit()
    port = ckt.node('pv', v_mp)
    bus = ckt.node('bus', 800.0)
    ckt.add_rl(0, bus, 0.05, 0.0, i0=-p_mp / 800, e0=800.0 + 0.05 * (-p_mp / 800))
    model = emt_der.PvEmt(ckt, obj, 'PV', port, v_mp, p_mp / v_mp)
    builder = types.SimpleNamespace(ckt=ckt, vn={'pv': 800.0, 'bus': 800.0}, v_bus={'pv': v_mp, 'bus': 800.0})
    conv = DcDc(builder, 'K', 'pv', 'bus', port, bus, p_in_mw=p_mp / 1e6, p_out_mw=p_mp / 1e6, mode='power',
                p_set_mw=p_mp / 1e6, rated_mw=0.3, bidirectional=False)
    conv.p_in_ref = emt_der.Mppt(model, conv, v_mp)
    ckt.at(0.3, lambda st: model.step_irradiance(500.0))
    ckt.add_controller(lambda t, st: (model.control(t, st), conv.control(t, st))[1])
    sim = ckt.simulate(0.8, 2e-5)
    tr = np.array(model.trace)
    p = tr[:, 1] * tr[:, 2]
    before = (tr[:, 0] > 0.15) & (tr[:, 0] < 0.3)
    after = tr[:, 0] > 0.65
    obj2 = _pv(irradiance_wm2=500, ambient_c=25.0)
    assert p[before].mean() == pytest.approx(p_mp, rel=0.01)
    assert p[after].mean() == pytest.approx(obj2.mpp()[2], rel=0.01)
    assert len(sim['t'])


# --- Supercapacitor smoothing a rack's cycle ------------------------------------------------------

@pytest.mark.parametrize('name', sorted(P['smoothing']['cases']))
def test_supercapacitor_smoothing_against_simscape(name):
    """
    The store behind its smoothing converter (averaged): the feed's current,
    the rack bus's voltage and the store's terminal through three cycles, as
    Simscape's supercapacitor behind an averaged converter with the same law;
    the feed's power follows the filter's closed form.
    """
    s = P['smoothing']
    q = s['cases'][name]
    vn_in = q['module_v'] * q['modules_series']
    obj = der.build({'typ': 'Supercapacitor0', 'name': 'sc', 'sizing': 'modules', 'modules_series': q['modules_series'],
                     'strings_parallel': q['strings_parallel'], 'module_c_f': q['module_c'], 'module_v': q['module_v'],
                     'module_esr_mohm': 1e3 * q['module_esr'], 'v0_percent': q['v0_percent'], 'r_leak_ohm': q['r_leak']})
    v_store0 = obj.v0
    v_rack0 = (q['v_feed'] + math.sqrt(q['v_feed'] ** 2 - 4 * q['r_feed'] * q['p_high'])) / 2
    ckt = Circuit()
    rack = ckt.node('rack', v_rack0)
    port = ckt.node('port', v_store0)
    i_feed0 = q['p_high'] / v_rack0
    k_feed = ckt.add_rl(0, rack, q['r_feed'], 0.0, i0=i_feed0, e0=q['v_feed'])
    tb, pb = [], []
    for k in range(int(math.ceil(s['t_end'] / s['period'])) + 1):
        t0 = k * s['period']
        tb += [t0, t0 + s['high'], t0 + s['high'] + 1e-6, t0 + s['period'] - 1e-6]
        pb += [q['p_high'], q['p_high'], q['p_low'], q['p_low']]
    power = lambda t: float(np.interp(t, tb, pb)) / 1e6
    func = dc_load_current([q['p_high'] / 1e6], q['v_feed'] / 1e3, 1.0, 0.0, 0.0, 0.0, power=power)
    ckt.add_nonlinear(rack, 0, func)
    store = emt_der.SupercapacitorEmt(ckt, obj, 'SC', port, v_store0, 0.0)
    builder = types.SimpleNamespace(ckt=ckt, vn={'p': vn_in, 'r': q['v_feed']}, v_bus={'p': v_store0, 'r': v_rack0})
    conv = DcDc(builder, 'K', 'p', 'r', port, rack, p_in_mw=0.0, p_out_mw=0.0, mode='power', rated_mw=q['rated'] / 1e6,
                eta=q['eta'], bidirectional=True, every_step=True)
    smoother = emt_der.Smoothing(store, [(rack, func)], q['tau'], 50.0, 0.0, q['rated'])
    conv.p_ref = lambda t, st: smoother(t, st)
    ckt.add_controller(lambda t, st: (store.control(t, st), conv.control(t, st))[1])
    sim = ckt.simulate(s['t_end'], s['dt'])
    ref = _csv(f'emt_der_smoothing_{name}.csv')
    # Clear of each edge's first millisecond: its controller acts a step after what it measured, Simscape's at
    # once, and the difference dies away through R_feed and its output capacitor (0.125 ms at 800 V).
    def clear(times):
        out = np.ones_like(times, dtype=bool)
        for t_e in tb:
            out &= ~((times > t_e - 3 * s['dt']) & (times < t_e + 1e-3))
        return out
    t = ref['t'][clear(ref['t'])]
    i_feed = _at(sim['t'], sim['i_rl'][:, k_feed], t)
    assert np.max(np.abs(i_feed - np.interp(t, ref['t'], ref['i_feed']))) < 2e-3 * i_feed0
    for col, node in (('v_rack', rack), ('v_port', port)):
        got = _at(sim['t'], sim['v'][:, node], t)
        assert np.max(np.abs(got - np.interp(t, ref['t'], ref[col]))) < 1e-4 * q['v_feed'], col
    # The feed's power: the racks' through the filter (a first-order lag of tau), within the converter's losses.
    y, prev, err = q['p_high'], 0.0, []
    tt = sim['t']
    p_feed = sim['v'][:, rack] * sim['i_rl'][:, k_feed]
    ok = clear(tt)
    for k in range(1, len(tt)):
        y += (power(tt[k]) * 1e6 - y) * (1 - math.exp(-(tt[k] - tt[k - 1]) / q['tau']))
        if ok[k]:
            err.append(abs(p_feed[k] - y))
    assert max(err) < 0.03 * q['p_high']


# --- Flywheel ---------------------------------------------------------------------------------

def test_flywheel_into_its_power_limit_against_simscape():
    """
    Below its base speed its machine converter gives at most P_rated x w / w_base:
    a load asking more sags its DC link to sqrt(P_max R) as its rotor slows - as
    Simscape's inertia behind the same averaged converter.
    """
    q = P['flywheel']
    obj = der.build({'typ': 'Flywheel0', 'name': 'fw', 'v_dc': q['v_dc'], 'p_rated_kw': q['p_rated'] / 1e3,
                     'e_max_kwh': q['e_max_kwh'], 'speed_percent': 100 * q['speed'],
                     'speed_min_percent': 100 * q['speed_min'], 'speed_base_percent': 100 * q['speed_base'],
                     'efficiency_percent': 100 * q['efficiency'], 'standby_loss_percent_h': q['standby_percent_h'],
                     'r_dc_mohm': 1e3 * q['r_dc']})
    ckt = Circuit()
    i0 = q['v_dc'] / (q['r_load'] + q['r_dc'])
    term = ckt.node('term', q['v_dc'] - q['r_dc'] * i0)
    model = emt_der.FlywheelEmt(ckt, obj, 'FW', term, q['v_dc'] - q['r_dc'] * i0, i0)
    ckt.add_r(term, 0, q['r_load'])
    x = ckt.node('step')
    k_sw = ckt.add_switch(term, x, closed=False)
    ckt.add_r(x, 0, q['r_step'])
    ckt.at(q['t_step'], lambda st: st.set_switch(k_sw, True))
    ckt.add_controller(lambda t, st: model.control(t, st))
    sim = ckt.simulate(q['t_end'], q['dt'])
    ref = _csv('emt_der_flywheel.csv')
    t = ref['t'][np.abs(ref['t'] - q['t_step']) > 5e-4]
    v = _at(sim['t'], sim['v'][:, model.link], t)
    assert np.max(np.abs(v - np.interp(t, ref['t'], ref['v_link']))) < 2e-3 * q['v_dc']
    tr = np.array(model.trace)
    assert np.max(np.abs(_at(tr[:, 0], tr[:, 3], t) - np.interp(t, ref['t'], ref['speed']))) < 1e-5
    # Settled into its limit: its link at sqrt(P_max (r_dc + R_eq)), P_max at its speed then.
    r_eq = 1 / (1 / q['r_load'] + 1 / q['r_step'])
    p_max = q['p_rated'] * obj.s0 / q['speed_base']
    assert sim['v'][-1, model.link] == pytest.approx(math.sqrt(p_max * (r_eq + q['r_dc'])), rel=1e-3)
    assert model.limited_time > 0.15


# --- SOFC -------------------------------------------------------------------------------------

def test_sofc_partial_pressures_after_a_current_step():
    """
    Its current stepped from 60 kW's to 90 kW's: its ohmic drop at once, then
    the water's partial pressure relaxing with tau_H2O = 78.3 s and the oxygen's
    after the fuel processor's 5 s lag - each the Padulles model's closed form.
    """
    q = P['sofc']
    obj = der.build({'typ': 'SOFC0', 'name': 'fc', 'p_rated_kw': q['p_rated_kw'], 'v_rated': q['v_rated'],
                     'p_set_kw': q['p0_kw']})
    p0 = obj.p_operating()
    i0 = p0 / obj.v_terminal(1.0)
    for _ in range(50):
        i0 = p0 / obj.v_terminal(i0)
    v0 = obj.v_terminal(i0)
    ckt = Circuit()
    node = ckt.node('fc', v0)
    model = emt_der.SofcEmt(ckt, obj, 'FC', node, v0, i0)
    i1 = i0 * q['p1_kw'] / q['p0_kw']
    k_load = ckt.add_isrc(node, 0, i0=i0)
    ckt.at(q['t_step'], lambda st: st.set_source_value(k_load, i1))
    ckt.add_controller(lambda t, st: model.control(t, st))
    sim = ckt.simulate(q['t_end'], q['dt'])
    # Before the step it holds its load-flow voltage.
    k0 = np.argmin(np.abs(sim['t'] - 0.5 * q['t_step']))
    assert sim['v'][k0, node] == pytest.approx(v0, rel=1e-6)
    # The water's pressure: a first-order lag of tau_H2O toward 2 Kr I / K_H2O.
    kr = obj.N0 / (4 * der.FARADAY * 1e3)
    aux = obj.aux_frac * obj.p_rated
    stack0 = (i0 + aux / v0) / obj.n_parallel
    p_h2o_0 = 2 * kr * stack0 / obj.K_H2O
    # Recompute: pH2O's final value at the new stack current (its aux current at the new voltage, near enough).
    v_new = sim['v'][-1, node]
    stack1 = (i1 + aux / v_new) / obj.n_parallel
    p_h2o_1 = 2 * kr * stack1 / obj.K_H2O
    want = p_h2o_1 + (p_h2o_0 - p_h2o_1) * math.exp(-(q['t_end'] - q['t_step']) / emt_der.TAU_H2O)
    assert model.p_h2o == pytest.approx(want, rel=2e-3)
    # The ohmic drop at once: R times the stack's change of current - the load's, and its auxiliary
    # load's, a constant power drawing more at the lower voltage.
    ka = np.searchsorted(sim['t'], q['t_step'] + 0.02)
    kb = np.searchsorted(sim['t'], q['t_step'] - 0.02)
    va, vb = sim['v'][ka, node], sim['v'][kb, node]
    assert vb - va == pytest.approx(model.r * ((i1 - i0) + aux * (1 / va - 1 / vb)), rel=0.01)


# --- In the EMT study, on the drawn network --------------------------------------------------------

def test_emt_study_with_every_source_and_store(client, quiet):
    """
    The 800 V network with a battery behind a dispatch converter, a PV array
    behind MPPT, an SOFC behind a follower, a battery directly on a bus, and a
    supercapacitor and a flywheel smoothing a 48 V rack: each starts at its
    load-flow power and nothing blocks; after the racks step up 50 % the stores
    take the step and the feed ramps through the filter; after the irradiance
    halves, the PV array's power falls toward its new maximum.
    """
    from test_dc_elements import _drawn_request, _with
    from test_der import _bus, _conv, _der
    els = [_bus('bt_bus', 0.8), _der('Battery', 'bt', 'bt_bus', capacity_kwh=500),
           _conv('kbt', 'bt_bus', 'dc_a', control_mode='dispatch', p_set_mw=0.05),
           _bus('pv_bus', 0.8), _der('PV Array', 'pv', 'pv_bus'),
           _conv('kpv', 'pv_bus', 'dc_b', control_mode='mppt', rated_mw=0.15),
           _bus('fc_bus', 0.8), _der('SOFC', 'fc', 'fc_bus', p_rated_kw=100, p_set_kw=60),
           _conv('kfc', 'fc_bus', 'dc_b', control_mode='follower', rated_mw=0.15),
           _der('Battery', 'btd', 'dc_b', capacity_kwh=200),
           _bus('r48', 0.048), {'typ': 'Load DC9', 'name': 'ld48', 'id': 'cell-ld48', 'userFriendlyName': 'Racks',
                                'bus': 'r48', 'p_mw': '0.01'},
           _conv('k48', 'dc_b', 'r48', rated_mw=0.05),
           _bus('sc48', 0.054), _der('Supercapacitor', 'sc', 'sc48', sizing='modules', strings_parallel=2),
           _conv('ksc', 'sc48', 'r48', control_mode='smoothing', smoothing_tau_s=0.02, soc_gain=0, rated_mw=0.02),
           _bus('fw48', 0.054), _der('Flywheel', 'fw', 'fw48', v_dc=54, e_max_kwh=0.07, p_rated_kw=10),
           _conv('kfw', 'fw48', 'r48', control_mode='smoothing', smoothing_tau_s=0.02, soc_gain=0, rated_mw=0.02)]
    request = _with(_drawn_request(), *els)
    key = next(k for k, v in request.items() if 'Parameters' in str(v.get('typ', '')))
    request[key] = {'typ': 'EmtStudy Parameters', 'user_email': 't@t', 'time_step_us': '10', 'duration_ms': '60',
                    'step_load': 'ld48', 'step_percent': '50', 'step_time_ms': '20',
                    'pv_step_wm2': '500', 'pv_step_time_ms': '30'}
    with quiet():
        result = json.loads(client.post('/', json=request).get_data(as_text=True))
    assert 'emt' in result, result.get('message')
    emt = result['emt']
    assert not emt['converters_blocked']
    ders = {d['label']: d for d in emt['ders']}
    assert ders['BT']['p_start_mw'] == pytest.approx(0.05 / 0.98, abs=1e-6)          # results to six decimals
    assert ders['FC']['p_start_mw'] == pytest.approx(0.06, abs=1e-6)
    assert ders['FC']['p_end_mw'] == pytest.approx(0.06, rel=1e-3)
    assert ders['PV']['irradiance_wm2_end'] == 500 and ders['PV']['p_end_mw'] < 0.6 * ders['PV']['p_start_mw']
    assert ders['SC']['p_start_mw'] == pytest.approx(0, abs=1e-9)
    conv = {c['label']: c for c in emt['converters']}
    sm = conv['KSC']['smoothing']
    assert sm['rack_ramp_mw_s'] > 100 * sm['feed_ramp_mw_s']           # both stores took the step
    assert sm['store_peak_mw'] == pytest.approx(0.0025, rel=0.05)       # half of the 5 kW step each
    assert conv['KPV']['control'] == 'mppt'
