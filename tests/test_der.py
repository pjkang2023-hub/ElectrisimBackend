"""
The microgrid's sources and stores - battery, supercapacitor, flywheel, SOFC
system and PV array - each on a DC bus: their static curves against closed
forms, in the load flow behind their DC/DC converters (MPPT, follower,
dispatch, droop, smoothing) or directly on a bus, sized by ratings or by
building block, and what the batteries and supercapacitors feed into a DC
fault.
"""
import json
import math

import pytest

import der_electrisim as der
import dc_fault_electrisim as dcf
from test_dc_elements import _drawn_request, _with
from test_dc_fault import STUDY

ETA = 0.98


def _bus(name, kv):
    return {'typ': 'DC Bus9', 'name': name, 'id': f'cell-{name}', 'userFriendlyName': name.upper(), 'vn_kv': str(kv)}


def _conv(name, bus_in, bus_out, **fields):
    row = {'typ': 'DC/DC Converter0', 'name': name, 'id': f'cell-{name}', 'userFriendlyName': name.upper(),
           'bus_in': bus_in, 'bus_out': bus_out, 'rated_mw': '0.5', 'efficiency_percent': str(100 * ETA),
           'no_load_loss_kw': '0', 'control_mode': 'voltage', 'vm_out_pu': '1.0'}
    row.update({k: str(v) for k, v in fields.items()})
    return row


def _der(kind, name, bus, **fields):
    row = {'typ': f'{kind}0', 'name': name, 'id': f'cell-{name}', 'userFriendlyName': name.upper(), 'bus': bus}
    row.update({k: str(v) for k, v in fields.items()})
    return row


def _run(client, quiet, *elements, study=None):
    request = _with(_drawn_request(), *elements)
    if study is not None:
        key = next(k for k, v in request.items() if 'Parameters' in str(v.get('typ', '')))
        request[key] = {**STUDY, **{k: str(v) for k, v in study.items()}}
    with quiet():
        response = client.post('/', json=request)
    result = json.loads(response.get_data(as_text=True))
    assert not result.get('error'), result.get('message')
    return result


def _by_id(rows):
    return {r['id']: r for r in rows}


def _vm(result, bus):
    return _by_id(result['dcbuses'])[f'cell-{bus}']['vm_pu']


# --- Static curves ----------------------------------------------------------------------

def test_battery_ocv_and_terminal_voltage():
    """Its OCV is the LFP table's, scaled to its nominal voltage; behind it R0 + R1 in a steady state."""
    bt = der.build(_der('Battery', 'b', 'x', sizing='ratings', vn_v=800, capacity_kwh=500, soc_percent=25))
    lfp = dict(der.LFP_OCV)
    assert bt.ah == pytest.approx(625.0)
    assert bt.ocv() == pytest.approx(800 * (lfp[0.2] + 0.5 * (lfp[0.3] - lfp[0.2])) / 3.2)
    assert bt.r0 == pytest.approx(0.0625) and bt.r1 == pytest.approx(0.4 * 0.0625)
    assert bt.v_terminal(100.0) == pytest.approx(bt.ocv() - 100.0 * 0.0875)
    assert bt.p_limits() == pytest.approx((625 * bt.ocv(), 0.5 * 625 * bt.ocv()))
    # Its own table, per cell.
    own = der.build(_der('Battery', 'b', 'x', sizing='cells', cells_series=16, cell_v=3.2,
                         ocv_table='0:3.0, 100:3.4', soc_percent=50))
    assert own.ocv() == pytest.approx(16 * 3.2)


def test_supercapacitor_energy():
    """1/2 C V^2, usable down to half its rated voltage; its ESR's matched-load power without a rating."""
    sc = der.build(_der('Supercapacitor', 's', 'x', c_f=130, v_rated=54, esr_mohm=4, v0_percent=90))
    v0 = 0.9 * 54
    assert sc.energy() == pytest.approx((0.5 * 130 * v0 ** 2, 0.5 * 130 * (v0 ** 2 - 27 ** 2)))
    assert sc.p_limits()[0] == pytest.approx(v0 ** 2 / (4 * 0.004))
    assert sc.v_terminal(100.0) == pytest.approx(v0 - 0.4)


def test_flywheel_power_limit():
    """Its rating above base speed, its torque limit below: P_rated min(1, s / s_base); none at its floor."""
    def fw(speed):
        return der.build(_der('Flywheel', 'f', 'x', p_rated_kw=250, speed_base_percent=60, speed_min_percent=30,
                              speed_percent=speed, e_max_kwh=2))
    assert fw(90).p_limits() == pytest.approx((250e3, 250e3))
    assert fw(45).p_limits() == pytest.approx((250e3 * 45 / 60, 250e3 * 45 / 60))
    assert fw(30).p_limits()[0] == 0.0
    st = fw(80).state(0.0, 800.0)
    assert st['energy_kwh'] == pytest.approx(2 * 0.8 ** 2)
    assert st['usable_kwh'] == pytest.approx(2 * (0.8 ** 2 - 0.3 ** 2))


def _padulles_stack_v(i, u=0.85):
    """The 384-cell, 100 kW reference stack's steady-state voltage (Padulles 2000), written out here."""
    n0, e0, r, t = 384, 1.18, 0.126, 1273.0
    k_h2, k_h2o, k_o2, r_ho = 8.43e-4, 2.81e-4, 2.52e-3, 1.145
    kr = n0 / (4 * 96485.33212 * 1e3)
    p_h2 = 2 * kr * i * (1 / u - 1) / k_h2
    p_h2o = 2 * kr * i / k_h2o
    p_o2 = (2 * kr * i / (u * r_ho) - kr * i) / k_o2
    return n0 * (e0 + 8.314462618 * t / (2 * 96485.33212) * math.log(p_h2 * math.sqrt(p_o2) / p_h2o)) - r * i


@pytest.mark.parametrize('i', [50.0, 150.0, 300.0])
def test_sofc_polarisation_matches_padulles(i):
    fc = der.build(_der('SOFC', 'f', 'x', p_rated_kw=100, v_rated=800))
    assert fc._v_ref(i) == pytest.approx(_padulles_stack_v(i), rel=1e-12)


def test_sofc_scaled_to_its_rating_and_voltage():
    """Cells in series set its voltage, stacks in parallel its current: at its rated current, its rating."""
    fc = der.build(_der('SOFC', 'f', 'x', p_rated_kw=1000, v_rated=800, p_set_kw=990))
    i_ref = fc._i_at(100e3)
    assert i_ref * _padulles_stack_v(i_ref) == pytest.approx(100e3, rel=1e-9)
    assert fc.v_stack(fc.i_rated) * fc.i_rated == pytest.approx(1000e3, rel=1e-9)
    assert fc.v_stack(fc.i_rated) == pytest.approx(800, rel=_padulles_stack_v(i_ref) / 384 / 800)
    # Its set power is held below its rating less its auxiliary load (5 %).
    assert fc.p_operating() == pytest.approx(950e3) and fc.notes
    st = fc.state(1000.0, 800.0)
    assert st['h2_kg_h'] == pytest.approx(st['stack_current_a'] * fc.n_series / (2 * 96485.33212) / 0.85
                                          * 2.016e-3 * 3600)


def test_pv_array_meets_its_datasheet_at_stc():
    """At 1000 W/m2 and a 25 C cell: Vmpp x Impp per module at its MPP, its Isc, its Voc, less its losses."""
    pv = der.build(_der('PV Array', 'p', 'x', irradiance_wm2=1000, ambient_c=25 - 25 * 1000 / 800, loss_percent=0))
    assert pv.t_cell == pytest.approx(25.0)
    v, i, p = pv.mpp()
    assert p == pytest.approx(180 * 41.9 * 13.13, rel=1e-6)
    assert v == pytest.approx(18 * 41.9, rel=2e-3) and i == pytest.approx(10 * 13.13, rel=2e-3)
    assert pv.i_array(0.0) == pytest.approx(10 * 14.0, rel=1e-6)
    assert pv.n_s * pv.v_oc_t == pytest.approx(18 * 49.9, rel=2e-3)
    # Its I-V curve inverted: the voltage at a current gives that current back.
    assert pv.i_array(pv.v_terminal(100.0)) == pytest.approx(100.0, rel=1e-6)


def test_pv_array_at_half_irradiance_and_hot():
    """Less irradiance, less current; a hotter cell, less voltage."""
    stc = der.build(_der('PV Array', 'p', 'x', ambient_c=-6.25))
    hot = der.build(_der('PV Array', 'p', 'x', irradiance_wm2=500, ambient_c=40))
    assert hot.t_cell == pytest.approx(40 + 25 * 500 / 800)
    assert hot.mpp()[2] < 0.5 * stc.mpp()[2] and hot.mpp()[0] < stc.mpp()[0]


# --- Load flow ----------------------------------------------------------------------------

SOURCES_800 = [
    _bus('pv_bus', 0.8), _der('PV Array', 'pv', 'pv_bus'), _conv('kpv', 'pv_bus', 'dc_b', control_mode='mppt'),
    _bus('fc_bus', 0.8), _der('SOFC', 'fc', 'fc_bus', p_rated_kw=100, p_set_kw=60),
    _conv('kfc', 'fc_bus', 'dc_b', control_mode='follower'),
    _bus('bt_bus', 0.8), _der('Battery', 'bt', 'bt_bus', sizing='ratings', vn_v=800, capacity_kwh=500),
    _conv('kbt', 'bt_bus', 'dc_a', control_mode='dispatch', p_set_mw=0.05),
]
RACK_48 = [
    _bus('r48', 0.048), {'typ': 'Load DC9', 'name': 'ld48', 'id': 'cell-ld48', 'userFriendlyName': 'Racks',
                         'bus': 'r48', 'p_mw': '0.01'},
    _conv('k48', 'dc_b', 'r48'),
    _bus('sc48', 0.054), _der('Supercapacitor', 'sc', 'sc48'), _conv('ksc', 'sc48', 'r48', control_mode='smoothing'),
    _bus('fw48', 0.054), _der('Flywheel', 'fw', 'fw48', v_dc=54, e_max_kwh=0.07, p_rated_kw=5),
    _conv('kfw', 'fw48', 'r48', control_mode='smoothing'),
]


def test_each_behind_its_converter(client, quiet):
    """
    On the 800 V network: the PV array at its maximum power point (MPPT),
    the SOFC at its set power (follower), the battery at 50 kW (dispatch) -
    each holding its own bus at its terminal voltage at the current its
    converter draws. On the 48 V rack bus, a supercapacitor and a flywheel
    behind smoothing converters: no power in a load flow.
    """
    result = _run(client, quiet, *[dict(e) for e in SOURCES_800 + RACK_48])
    ders, convs = _by_id(result['ders']), _by_id(result['dcdcconverters'])
    pv = der.build(SOURCES_800[1])
    v_mp, _, p_mp = pv.mpp()
    assert ders['cell-pv']['p_mw'] == pytest.approx(p_mp / 1e6, rel=1e-6)
    assert convs['cell-kpv']['p_in_mw'] == pytest.approx(p_mp / 1e6, rel=1e-6)
    assert convs['cell-kpv']['p_out_mw'] == pytest.approx(ETA * p_mp / 1e6, rel=1e-6)
    assert convs['cell-kpv']['control'] == 'mppt'
    assert _vm(result, 'pv_bus') * 800 == pytest.approx(v_mp, rel=1e-6)

    fc = der.build(SOURCES_800[4])
    assert ders['cell-fc']['p_mw'] == pytest.approx(0.06, rel=1e-6)
    i_fc = 60e3 / (_vm(result, 'fc_bus') * 800)
    assert _vm(result, 'fc_bus') * 800 == pytest.approx(fc.v_terminal(i_fc), rel=1e-6)
    assert ders['cell-fc']['efficiency_percent'] == pytest.approx(
        100 * 60e3 / (ders['cell-fc']['fuel_power_kw'] * 1e3))

    bt = der.build(SOURCES_800[7])
    assert convs['cell-kbt']['p_out_mw'] == pytest.approx(0.05)
    assert ders['cell-bt']['p_mw'] == pytest.approx(0.05 / ETA, rel=1e-6)
    i_bt = ders['cell-bt']['i_ka'] * 1e3
    assert _vm(result, 'bt_bus') * 800 == pytest.approx(bt.ocv() - 0.0875 * i_bt, rel=1e-6)
    assert ders['cell-bt']['c_rate'] == pytest.approx(i_bt / 625)

    for key, v in (('cell-sc', 0.9 * 54), ('cell-fw', 54)):
        assert ders[key]['p_mw'] == pytest.approx(0, abs=1e-9) and ders[key]['coupling'] == 'converter'
        assert ders[key]['v_kv'] * 1e3 == pytest.approx(v, rel=1e-6)
    assert convs['cell-ksc']['control'] == 'smoothing' and convs['cell-ksc']['p_out_mw'] == 0
    # The DC/DC converters' and elements' own auxiliary parts stay out of the results.
    assert [v['id'] for v in result['vscs']] == ['cell-vsc1']
    assert len(result['externalgrids']) == 1


def test_battery_charging_behind_its_converter(client, quiet):
    """Dispatched at -40 kW it charges: its bus above its OCV, by (R0 + R1) times its charging current."""
    result = _run(client, quiet, _bus('bt_bus', 0.8), _der('Battery', 'bt', 'bt_bus', vn_v=800, capacity_kwh=500),
                  _conv('kbt', 'bt_bus', 'dc_a', control_mode='dispatch', p_set_mw=-0.04, bidirectional=True))
    (bt,) = result['ders']
    assert bt['p_mw'] == pytest.approx(-0.04 * ETA, rel=1e-6)
    assert bt['v_kv'] * 1e3 == pytest.approx(bt['ocv_v'] - 0.0875 * bt['i_ka'] * 1e3, rel=1e-6)
    assert bt['v_kv'] * 1e3 > bt['ocv_v']


def test_converter_in_droop(client, quiet):
    """In droop it holds its output at its set voltage less droop x its loading: 5 % at full power."""
    result = _run(client, quiet, *[dict(e) for e in RACK_48[:3]])
    flat = _vm(result, 'r48')
    result = _run(client, quiet, *[dict(e) for e in RACK_48[:2]],
                  _conv('k48', 'dc_b', 'r48', control_mode='droop', droop_percent=5, rated_mw=0.02))
    (k48,) = result['dcdcconverters']
    assert k48['control'] == 'droop' and flat == pytest.approx(1.0)
    assert _vm(result, 'r48') == pytest.approx(1.0 - 0.05 * k48['p_out_mw'] / 0.02, rel=1e-6)
    assert _vm(result, 'r48') == pytest.approx(0.975, abs=1e-3)


@pytest.mark.parametrize('kind, ratings, blocks', [
    ('Battery', dict(sizing='ratings', vn_v=800, capacity_ah=280, r0_mohm=62.5),
     dict(sizing='cells', cells_series=250, strings_parallel=1, cell_v=3.2, cell_ah=280, cell_r_mohm=0.25)),
    ('Battery', dict(sizing='ratings', vn_v=51.2, capacity_ah=560, r0_mohm=2),
     dict(sizing='cells', cells_series=16, strings_parallel=2, cell_v=3.2, cell_ah=280, cell_r_mohm=0.25)),
    ('Supercapacitor', dict(sizing='ratings', c_f=130 / 15, v_rated=810, esr_mohm=60),
     dict(sizing='modules', modules_series=15, strings_parallel=1, module_c_f=130, module_v=54, module_esr_mohm=4)),
])
def test_sizing_by_ratings_or_by_building_block(client, quiet, kind, ratings, blocks):
    """The same element either way: the same model, and the same load-flow result."""
    a, b = der.build(_der(kind, 'x', 'y', **ratings)), der.build(_der(kind, 'x', 'y', **blocks))
    for attr in ('vn', 'ah', 'r0', 'r1', 'c', 'v_rated', 'esr'):
        if hasattr(a, attr):
            assert getattr(a, attr) == pytest.approx(getattr(b, attr)), attr
    kv = a.v_nominal() / 1e3
    results = []
    for fields in (ratings, blocks):
        extra = dict(coupling='direct') if kind == 'Supercapacitor' else {}
        if kv > 0.1:
            els = [_der(kind, 'x', 'dc_b', **fields, **extra)]
        else:
            els = [_bus('b48', kv), _der(kind, 'x', 'b48', **fields), _conv('k', 'b48', 'dc_b', control_mode='dispatch',
                                                                            p_set_mw=0.005)]
        results.append(_run(client, quiet, *els)['ders'][0])
    for key, value in results[0].items():
        if isinstance(value, float):
            assert results[1][key] == pytest.approx(value, rel=1e-9), key


def test_supercapacitor_behind_its_converter_or_directly_on_the_bus(client, quiet):
    """
    Behind its converter it holds its own bus at its voltage; directly on DC
    bus B it is a DC-link capacitor at B's voltage - no load-flow current
    either way - and reported with the sources and stores, not the DC capacitors.
    """
    sc = dict(sizing='modules', modules_series=15)
    behind = _run(client, quiet, _bus('sc_bus', 0.8), _der('Supercapacitor', 'sc', 'sc_bus', **sc),
                  _conv('ksc', 'sc_bus', 'dc_b', control_mode='smoothing'))
    direct = _run(client, quiet, _der('Supercapacitor', 'sc', 'dc_b', coupling='direct', **sc))
    (b,), (d,) = behind['ders'], direct['ders']
    assert b['coupling'] == 'converter' and d['coupling'] == 'direct'
    assert b['v_kv'] == pytest.approx(0.9 * 810 / 1e3) and b['p_mw'] == pytest.approx(0, abs=1e-12)
    assert d['v_kv'] == pytest.approx(_vm(direct, 'dc_b') * 0.8) and d['p_mw'] == 0
    assert d['usable_kj'] == pytest.approx(0.5 * 130 / 15 * ((d['v_kv'] * 1e3) ** 2 - 405 ** 2) / 1e3)
    assert not direct.get('dccapacitors')
    # On the bus, without a load-flow current, it changes nothing there.
    assert _vm(direct, 'dc_b') == pytest.approx(_vm(_run(client, quiet), 'dc_b'), abs=1e-12)


def test_directly_connected_sources(client, quiet):
    """
    Directly on DC bus B: a battery at (OCV - V) / (R0 + R1), a PV array at
    its curve's current at B's voltage, an SOFC at the current its
    polarisation curve gives there, a flywheel at its set power. A cell bus
    for the battery stays out of the results.
    """
    els = [_der('Battery', 'bt', 'dc_b', vn_v=800, capacity_kwh=200, soc_percent=60),
           _der('PV Array', 'pv', 'dc_b', modules_series=22, irradiance_wm2=800),
           _der('SOFC', 'fc', 'dc_b', p_rated_kw=100, v_rated=780),
           _der('Flywheel', 'fw', 'dc_b', p_set_kw=-20)]
    result = _run(client, quiet, *els)
    ders = _by_id(result['ders'])
    v = _vm(result, 'dc_b') * 800
    bt, pv, fc = (der.build(e) for e in els[:3])
    assert ders['cell-bt']['i_ka'] * 1e3 == pytest.approx((bt.ocv() - v) / 0.0875, rel=1e-6)
    assert ders['cell-pv']['p_mw'] * 1e6 == pytest.approx(v * pv.i_array(v), rel=1e-6)
    i_fc = ders['cell-fc']['i_ka'] * 1e3
    assert i_fc > 0 and fc.v_terminal(i_fc) == pytest.approx(v, rel=1e-6)
    assert ders['cell-fw']['p_mw'] == pytest.approx(-0.02)
    assert all(d['coupling'] == 'direct' for d in ders.values())
    assert set(_by_id(result['dcbuses'])) == {'cell-dc_a', 'cell-dc_b'}
    assert {l['id'] for l in result['loadsdc']} == {'cell-ld_a', 'cell-ld_b'}
    assert len(result['linedcs']) == 1


def test_elements_left_out_with_a_reason(client, quiet):
    """Two sources on one converter's bus, a source on a converter's output, one on a bus nothing reaches."""
    result = _run(client, quiet,
                  _bus('p1', 0.8), _der('Battery', 'b1', 'p1'), _der('Battery', 'b2', 'p1'),
                  _conv('k1', 'p1', 'dc_a', control_mode='dispatch', p_set_mw=0.01),
                  _bus('p2', 0.8), _der('Battery', 'b3', 'p2'), _conv('k2', 'dc_a', 'p2'),
                  _bus('p3', 0.8), _der('Supercapacitor', 's1', 'p3', coupling='direct'))
    assert [d['id'] for d in result['ders']] == ['cell-b1']
    w = ' '.join(result['warnings'])
    assert "Battery 'B2' is left out: another source already holds its converter's bus" in w
    assert "Battery 'B3' is left out: it is on its DC/DC converter's output" in w
    assert "Supercapacitor 'S1' is on a DC bus nothing else reaches" in w


def test_mppt_without_a_pv_array(client, quiet):
    result = _run(client, quiet, _bus('p1', 0.8), _der('Battery', 'b1', 'p1'),
                  _conv('k1', 'p1', 'dc_a', control_mode='mppt', p_set_mw=0.01))
    assert any("mppt mode needs a PV array on its input" in w for w in result['warnings'])
    assert result['dcdcconverters'][0]['p_out_mw'] == pytest.approx(0.01)


# --- DC fault -----------------------------------------------------------------------------

def test_fault_fed_by_a_battery_and_a_supercapacitor_on_the_bus(client, quiet):
    """
    Directly on DC bus B, a battery and a supercapacitor feed a fault there
    with the rectifier's diodes; the supercapacitor is reported as one, not
    as a DC capacitor. Behind its converter, a battery feeds a fault on its
    own bus only.
    """
    result = _run(client, quiet,
                  _der('Battery', 'bt', 'dc_b', vn_v=800, capacity_kwh=200, l_uh=10),
                  _der('Supercapacitor', 'sc', 'dc_b', coupling='direct', sizing='modules', modules_series=15,
                       module_esl_uh=0.1),
                  _bus('p1', 0.8), _der('Battery', 'b2', 'p1', vn_v=800, capacity_kwh=500),
                  _conv('k1', 'p1', 'dc_a', control_mode='dispatch', p_set_mw=0.05),
                  study={'duration_ms': 50})
    faults = _by_id(result['dcfault']['faults'])
    assert set(faults) == {'cell-dc_a', 'cell-dc_b', 'cell-p1'}
    f = faults['cell-dc_b']
    kinds = {c['kind']: c for c in f['contributions'] if c['ik_ka'] > 1e-3}
    assert set(kinds) == {'VSC (diodes, blocked)', 'Battery', 'Supercapacitor'}
    assert kinds['Battery']['name'] == 'BT' and kinds['Supercapacitor']['name'] == 'SC'
    # Each settles near its voltage over its resistance: the battery's R0 (62.5 mOhm), the 15 modules' ESR (60 mOhm).
    v0 = f['v_prefault_kv'] * 1e3
    assert kinds['Battery']['ik_ka'] == pytest.approx(v0 / 0.0625 / 1e3, rel=0.03)
    assert kinds['Supercapacitor']['ip_ka'] == pytest.approx(v0 / 0.060 / 1e3, rel=0.03)
    assert sum(c['at_peak_ka'] for c in f['contributions']) == pytest.approx(f['ip_ka'], rel=1e-3)
    # Behind its blocked converter the second battery feeds only a fault on its own bus.
    assert next(c for c in f['contributions'] if c['name'] == 'B2')['ik_ka'] == pytest.approx(0, abs=1e-3)
    own = faults['cell-p1']
    b2 = next(c for c in own['contributions'] if c['name'] == 'B2')
    assert b2['ik_ka'] == pytest.approx(own['ik_ka'], rel=1e-6)


def test_battery_fault_current_closed_form(client, quiet):
    """
    A battery alone on its converter's bus: E' behind R0 and L, E' its OCV
    less its RC branch's voltage, settling at E' / (R0 + Rf).
    """
    result = _run(client, quiet, _bus('p1', 0.8), _der('Battery', 'b', 'p1', vn_v=800, capacity_kwh=500, l_uh=100),
                  _conv('k1', 'p1', 'dc_a', control_mode='dispatch', p_set_mw=0.1), study={'duration_ms': 20})
    lf = _run(client, quiet, _bus('p1', 0.8), _der('Battery', 'b', 'p1', vn_v=800, capacity_kwh=500, l_uh=100),
              _conv('k1', 'p1', 'dc_a', control_mode='dispatch', p_set_mw=0.1))
    (b,) = lf['ders']
    i0, v = b['i_ka'] * 1e3, b['v_kv'] * 1e3
    e = v + 0.0625 * i0
    r = 0.0625 + dcf.R_FLOOR
    f = _by_id(result['dcfault']['faults'])['cell-p1']
    # Its time constant L / R (1.6 ms) has long run out by 20 ms.
    assert f['ik_ka'] * 1e3 == pytest.approx(e / r, rel=2e-3)
    assert f['ip_ka'] == pytest.approx(f['ik_ka'], rel=1e-3)


# --- The other studies ----------------------------------------------------------------------

MIXED = [
    _bus('pv_bus', 0.8), _der('PV Array', 'pv', 'pv_bus'), _conv('kpv', 'pv_bus', 'dc_b', control_mode='mppt'),
    _bus('bt_bus', 0.8), _der('Battery', 'bt', 'bt_bus'), _conv('kbt', 'bt_bus', 'dc_a', control_mode='dispatch',
                                                                p_set_mw=0.02),
    _der('Battery', 'btd', 'dc_b', capacity_kwh=200), _der('Supercapacitor', 'scd', 'dc_b', coupling='direct',
                                                           sizing='modules', modules_series=15),
    _der('SOFC', 'fcd', 'dc_b', v_rated=780), _der('Flywheel', 'fwd', 'dc_a'),
]


@pytest.mark.parametrize('fixture, params', [
    ('reference_radial.diagram_sc_payload.json', None),
    ('reference_radial.diagram_timeseries_payload.json', {'time_steps': 2}),
    ('reference_radial.diagram_contingency_payload.json', None),
    ('reference_radial.diagram_harmonic_payload.json', None),
    ('reference_radial.diagram_opf_payload.json', None),
    ('reference_radial.diagram_sc_payload.json', {'typ': 'StateEstimationPandaPower Parameters'}),
    ('reference_radial.diagram_sc_payload.json', {'typ': 'ArcFlashPandaPower Parameters'}),
    ('reference_radial.diagram_sc_payload.json', {'typ': 'EmtStudy Parameters', 'time_step_us': '5',
                                                  'duration_ms': '5'}),
    ('reference_radial.diagram_sc_payload.json', {'typ': 'TransientStabilityAndes Parameters', 'frequency': '50',
                                                  'sn_mva': '100', 'tf': '1', 'fault_enabled': False}),
])
def test_other_studies_run_with_sources_and_stores(client, quiet, fixture, params):
    """None may fail because of them, nor show their own parts."""
    request = _with(_drawn_request(fixture), *[dict(e) for e in MIXED])
    if params:
        key = next(k for k, v in request.items() if 'Parameters' in str(v.get('typ', '')))
        request[key] = {**request[key], **params} if 'typ' not in params else {**params, 'user_email': 't@t'}
    with quiet():
        response = client.post('/', json=request)
    assert response.status_code == 200
    text = response.get_data(as_text=True)
    result = json.loads(text)
    if isinstance(result, dict):
        assert not result.get('error'), result.get('message') or result.get('exception')
    for part in ('auxiliary grid', 'auxiliary AC', 'BTD cells', 'btd cells', 'btd resistance'):
        assert part not in text, part


def test_fault_with_a_pv_array_behind_its_converter(client, quiet):
    """
    The PV array is left out of the DC fault and its converter blocks, so its
    own bus is reached by nothing: it is tied off (its node defined - before,
    every result was NaN) and not faulted; the network's buses are.
    """
    result = _run(client, quiet, _bus('pv_bus', 0.8), _der('PV Array', 'pv', 'pv_bus'),
                  _conv('kpv', 'pv_bus', 'dc_b', control_mode='mppt'), study={'duration_ms': 20})
    faults = _by_id(result['dcfault']['faults'])
    assert set(faults) == {'cell-dc_a', 'cell-dc_b'}
    assert all(math.isfinite(f['ik_ka']) and f['ik_ka'] > 0 for f in faults.values())
    assert any("PV arrays, SOFC systems and flywheels feed a DC fault" in w for w in result['warnings'])


def _droop_spec(vm_out_pu=1.0, sst=True, load_mw=6.0):
    """A 6 MW, 800 V row group held by an SST, a 1.5 MW battery on it through a DC/DC converter in droop."""
    import electrisim_sld as sld
    spec = {
        'name': 'droop', 'buses': [{'id': 'A', 'vn_kv': 35}],
        'external_grids': [{'id': 'G', 'bus': 'A', 's_sc_max_mva': 600, 'rx_max': 0.1}],
        'dc_buses': [{'id': 'RG', 'vn_kv': 0.8}, {'id': 'BD', 'vn_kv': 1.0}],
        'dc_loads': [{'id': 'LD', 'bus': 'RG', 'p_mw': load_mw}],
        'batteries': [{'id': 'B', 'bus': 'BD', 'vn_v': 1000, 'capacity_kwh': 750, 'c_rate_discharge': 2,
                       'c_rate_charge': 2}],
        'dc_dc_converters': [{'id': 'DD', 'bus_in': 'BD', 'bus_out': 'RG', 'control_mode': 'droop', 'rated_mw': 1.5,
                              'vm_out_pu': vm_out_pu, 'vn_in_kv': 1.0, 'vn_out_kv': 0.8, 'bidirectional': True,
                              'droop_percent': 5, 'no_load_loss_kw': 2}],
    }
    if sst:
        spec['ssts'] = [{'id': 'U1', 'bus_mv': 'A', 'bus_lv_dc': 'RG', 'vn_mv_kv': 35, 'vn_lv_dc_kv': 0.8,
                         'link_kv': 1.5, 'rect_rated_mw': 7, 'dcdc_rated_mw': 7, 'vm_lv_dc_pu': 1.0}]
    return sld.solve(sld.build_network(spec)[0])


@pytest.mark.parametrize('vm_out_pu', [1.0, 1.02])
def test_droop_on_a_bus_another_converter_holds(quiet, vm_out_pu):
    """
    A converter in droop on a bus a supply unit holds: two voltage sources in
    parallel had no load flow (it never converged). It delivers the power its
    droop gives at the bus's voltage - rated x (V_set - V) / (droop V_set) -
    and the supply unit the rest.
    """
    with quiet():
        res = _droop_spec(vm_out_pu)
    assert res['converged'], res.get('hint')
    (dd,) = res['dc_dc_converters']
    (rg,) = [b for b in res['dc_buses'] if b['id'] == 'RG']
    assert rg['vm_pu'] == pytest.approx(1.0, abs=1e-9)
    assert dd['p_out_mw'] == pytest.approx(1.5 * (vm_out_pu - 1.0) / (0.05 * vm_out_pu), abs=1e-6)
    (sst,) = res['ssts']
    assert sst['stages'][1]['p_out_mw'] == pytest.approx(6.0 - dd['p_out_mw'], abs=1e-5)


def test_droop_holds_its_bus_when_nothing_else_does(quiet):
    """With no supply unit the converter holds the bus again, lowered by its droop: 1 - 0.05 x P / 1.5 MW."""
    with quiet():
        res = _droop_spec(sst=False, load_mw=1.0)
    assert res['converged'], res.get('hint')
    (dd,) = res['dc_dc_converters']
    (rg,) = [b for b in res['dc_buses'] if b['id'] == 'RG']
    assert dd['p_out_mw'] == pytest.approx(1.0, abs=1e-6)
    assert rg['vm_pu'] == pytest.approx(1.0 - 0.05 * 1.0 / 1.5, abs=1e-5)
