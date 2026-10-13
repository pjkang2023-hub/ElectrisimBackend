"""
Phase 21 of the PCS3750C-MV reference: the parts of one double-conversion
block, checked against its spec sheet (12 Oct 2026).

- converter_rating: the String PCS's derating (minimum of the inlet-air and
  altitude factors; current-limited below 505 V), its 150 % / 1 s and
  120 % / 10 s overload, and its Q cap of 0.6 S;
- the GS1.1's sodium-ion (NFPP) open-circuit voltage, 750-1,500 V a unit;
- one block's load flow: a grid-side rectifier (a VSC following its DC bus's
  load), two GS1.1 directly on each of its four DC buses holding their
  voltage, and a grid-forming PCS drawing from each DC bus, the four sharing
  the 505 V load bus by droop;
- in the EMT study, the grid-forming converter holding its bus's negative-
  sequence voltage at zero against a line-to-line load, and the 2f ripple
  its DC side carries for it.
"""
import math

import numpy as np
import pytest

import converter_rating as cr
import der_electrisim as der
import electrisim_sld as sld
import emt_solver as es
from emt_converters import A120, GridFormingVsc
from test_pcs import _bus, _der, _pcs, _post, _with

# The String PCS's inlet-air derating, per unit of its 45 deg C rating (spec sheet, Derating rules).
T_CURVE = '35:1.10, 45:1.00, 50:0.95, 60:0'
OVERLOAD = '1:1.5, 10:1.2'
S_BUS = 0.947           # MVA per DC bus at 45 deg C: 6 x PCS157SE
BLOCK_45C = 3.788       # MVA per block at 45 deg C


# --- converter_rating ----------------------------------------------------------------------------

@pytest.mark.parametrize('inlet, block_kva', [(35, 4166), (40, 3977), (45, 3788), (50, 3598), (55, 1799), (60, 0)])
def test_block_rating_by_inlet_air(inlet, block_kva):
    """The spec's block ratings: 3,788 kVA at 45 deg C, ~1 %/deg C between 35 and 50, to zero at 60."""
    f = cr.derating({'inlet_c': inlet, 'derate_temp_curve': T_CURVE})['factor']
    assert BLOCK_45C * 1e3 * f == pytest.approx(block_kva, abs=1.0)


def test_minimum_of_the_factors_not_their_product():
    """A cool inlet's 110 % does not lift a 3,000 m site's 93 %; at 45 deg C the two give 93 %, not 0.93 x 1.0 x ..."""
    d = cr.derating({'inlet_c': 35, 'derate_temp_curve': T_CURVE, 'altitude_m': 3000})
    assert d['temperature'] == pytest.approx(1.10) and d['altitude'] == pytest.approx(0.93)
    assert d['factor'] == pytest.approx(0.93)
    hot = cr.derating({'inlet_c': 50, 'derate_temp_curve': T_CURVE, 'altitude_m': 3000})
    assert hot['factor'] == pytest.approx(0.93)
    assert cr.derating({'inlet_c': 50, 'derate_temp_curve': T_CURVE, 'altitude_m': 3500})['factor'] == pytest.approx(0.895)


def test_current_limited_below_nominal_voltage():
    """444 V on a 505 V basis: 88 % at 45 deg C, 96.7 % at 35 deg C (its 110 % x 444/505; the spec rounds it to 96)."""
    v = 444.0 / 505.0
    assert cr.voltage_factor(v) * 1.00 == pytest.approx(0.879, abs=0.001)
    assert cr.voltage_factor(v) * 1.10 == pytest.approx(0.967, abs=0.001)
    assert cr.voltage_factor(555.0 / 505.0) == 1.0      # power-capped above


def test_blank_fields_leave_the_nameplate():
    assert cr.derating({})['factor'] == 1.0
    noted = cr.derating({'inlet_c': 50})
    assert noted['factor'] == 1.0 and noted['notes']


def test_short_time_overload():
    """150 % for 1 s, then 120 % to 10 s, then the continuous rating: 5,681 and 4,545 kVA a block at 45 deg C."""
    steps = cr.overload_steps(OVERLOAD)
    assert BLOCK_45C * 1e3 * cr.overload_ceiling(steps, 0.5) == pytest.approx(5682, abs=1)
    assert BLOCK_45C * 1e3 * cr.overload_ceiling(steps, 5.0) == pytest.approx(4546, abs=1)
    assert cr.overload_ceiling(steps, 10.0) == 1.0
    assert cr.overload_time(steps, 1.3) == 1.0 and cr.overload_time(steps, 1.1) == 10.0
    assert cr.overload_time(steps, 1.6) == 0.0 and cr.overload_time(steps, 1.0) == math.inf
    clock = cr.OverloadClock(steps, reset_s=60.0)
    ceilings = [clock.step(1.4, 0.25) for _ in range(8)]          # 2 s at 140 %
    assert ceilings[:3] == [1.5] * 3 and ceilings[-1] == 1.2
    clock.step(0.9, 30.0)
    assert clock.ceiling() == 1.2                                  # not cooled yet
    clock.step(0.9, 30.0)
    assert clock.ceiling() == 1.5                                  # cooled: 60 s under


def test_reactive_cap():
    """Q = min(sqrt(S^2 - P^2), 0.6 S): 0.6 S at half power, the circle near full power, none at full."""
    assert cr.q_limit(1.0, 0.5, 0.6) == pytest.approx(0.6)
    assert cr.q_limit(1.0, 0.9, 0.6) == pytest.approx(math.sqrt(1 - 0.81))
    assert cr.q_limit(1.0, 1.0, 0.6) == pytest.approx(0.0)
    assert cr.q_limit(1.0, 0.5, None) == pytest.approx(math.sqrt(0.75))


# --- The GS1.1's sodium-ion cells ----------------------------------------------------------------

def test_nfpp_open_circuit_voltage():
    """A 1,200 V NFPP pack runs 750 V empty to under 1,500 V full, sloping - LFP's plateau is flat."""
    gs = der.build({'typ': 'Battery0', 'name': 'gs', 'vn_v': '1200', 'capacity_kwh': '3100', 'ocv_table': 'NFPP'})
    assert gs.ocv(0.0) == pytest.approx(754.8, abs=0.5)
    assert 1450 < gs.ocv(1.0) < 1500
    assert gs.ocv(0.2) < gs.ocv(0.5) < gs.ocv(0.8)
    assert gs.ocv(0.8) - gs.ocv(0.2) > 100                # the slope a state of charge shows in
    lfp = der.build({'typ': 'Battery0', 'name': 'l', 'vn_v': '1200', 'capacity_kwh': '3100', 'ocv_table': 'LFP'})
    assert lfp.ocv(0.8) - lfp.ocv(0.2) < 40


# --- One block's load flow -----------------------------------------------------------------------

def _block(inlet=45.0, load_mw=3.4, soc=50.0, at_505=False, q_set=None, out=()):
    """
    One PCS3750C-MV block: a 34.5 kV grid, T1, the 505 V grid-side bus, four
    DC buses (1.2 kV nominal) each with a rectifier following its load, two
    GS1.1 and a grid-forming PCS, the 505 V load bus, T2 and the load
    backbone - its load there, or on the 505 V load bus (``at_505``).
    """
    spec = {'name': 'pcs3750 block', 'frequency_hz': 60,
            'buses': [{'id': 'POI', 'vn_kv': 34.5}, {'id': 'GLV', 'vn_kv': 0.505}, {'id': 'LLV', 'vn_kv': 0.505},
                      {'id': 'BB', 'vn_kv': 34.5}],
            'external_grids': [{'id': 'G', 'bus': 'POI', 's_sc_max_mva': 500, 'rx_max': 0.1}],
            'transformers': [
                {'id': 'T1', 'hv_bus': 'POI', 'lv_bus': 'GLV', 'sn_mva': 3.75, 'vk_percent': 6, 'vkr_percent': 0.8,
                 'vector_group': 'YNd', 'shift_degree': 30},
                {'id': 'T2', 'hv_bus': 'BB', 'lv_bus': 'LLV', 'sn_mva': 3.75, 'vk_percent': 6, 'vkr_percent': 0.8,
                 'vector_group': 'Dyn', 'shift_degree': 30}],
            'loads': [{'id': 'L', 'bus': 'LLV' if at_505 else 'BB', 'p_mw': load_mw, 'q_mvar': 0}],
            'dc_buses': [], 'vscs': [], 'pcs': [], 'batteries': []}
    for k in range(1, 5):
        if k in out:
            continue
        spec['dc_buses'].append({'id': f'DC{k}', 'vn_kv': 1.2})
        spec['vscs'].append({'id': f'AFE{k}', 'bus': 'GLV', 'bus_dc': f'DC{k}', 'r_ohm': 0.0013, 'x_ohm': 0.0404,
                             'r_dc_ohm': 0.0001, 'control_mode_ac': 'q_mvar', 'control_value_ac': 0,
                             'control_mode_dc': 'follow_load', 'control_value_dc': 0, 'rated_mva': S_BUS,
                             'inlet_c': inlet, 'derate_temp_curve': T_CURVE, 'overload_curve': OVERLOAD})
        pcs = {'id': f'GFM{k}', 'bus': 'LLV', 'bus_dc': f'DC{k}', 'control': 'grid_forming', 's_rated_mva': S_BUS,
               'vn_ac_kv': 0.505, 'efficiency_percent': 98.5, 'droop_pf_percent': 2, 'droop_qv_percent': 5,
               'inlet_c': inlet, 'derate_temp_curve': T_CURVE, 'overload_curve': OVERLOAD, 'q_max_percent': 60}
        if q_set is not None:
            # Grid-following on the grid side's bus, the grid holding it: its set P and Q.
            pcs.update(bus='GLV', control='grid_following', q_mode='q', q_set_mvar=q_set, p_set_mw=0.3)
            spec['loads'] = []
        spec['pcs'].append(pcs)
        for u in 'ab':
            spec['batteries'].append({'id': f'B{k}{u}', 'bus': f'DC{k}', 'vn_v': 1200, 'capacity_kwh': 3100,
                                      'ocv_table': 'NFPP', 'soc_percent': soc, 'soc_min_percent': 0,
                                      'soc_max_percent': 100, 'c_rate_discharge': 0.5, 'c_rate_charge': 0.25})
    return spec


def _solve(spec, quiet):
    with quiet():
        res = sld.solve(sld.build_network(spec)[0])
    assert res['converged'], res.get('hint')
    return res


def _by(rows):
    return {r['id']: r for r in rows}


def test_block_rectifiers_follow_the_load_and_the_batteries_hold_the_dc_links(quiet):
    """
    3.4 MW on the backbone: the four grid-forming PCS share it equally by droop,
    each rectifier delivers what its PCS draws - so its batteries carry
    nothing and hold their DC bus at their open-circuit voltage - and the
    island runs at f0 (1 - droop P / S).
    """
    res = _solve(_block(), quiet)
    pcs, vscs, dc = _by(res['pcs']), _by(res['vscs']), _by(res['dc_buses'])
    p = [pcs[f'GFM{k}']['p_mw'] for k in range(1, 5)]
    assert max(p) - min(p) < 1e-6 and sum(p) == pytest.approx(3.4, rel=0.02)    # T2's losses on top
    ocv = der.build({'typ': 'Battery0', 'name': 'x', 'vn_v': '1200', 'capacity_kwh': '3100', 'ocv_table': 'NFPP',
                     'soc_percent': '50'}).ocv()
    for k in range(1, 5):
        g, a = pcs[f'GFM{k}'], vscs[f'AFE{k}']
        assert g['source_kind'] == 'DC bus' and g['islanded']
        assert -a['p_dc_mw'] == pytest.approx(g['p_dc_mw'], abs=1e-5)            # the rectifier follows
        assert dc[f'DC{k}']['vm_pu'] * 1200 == pytest.approx(ocv, abs=0.01)      # the batteries idle at OCV
        assert g['frequency_hz'] == pytest.approx(60 * (1 - 0.02 * g['p_mw'] / S_BUS), rel=1e-6)
    stores = [s for s in res['sources_and_stores'] if s['kind'] == 'Battery']
    assert len(stores) == 8 and all(abs(s['p_mw']) < 1e-5 for s in stores)


def test_per_bus_current_at_the_dc_floor(quiet):
    """
    The spec's 1,162 A per DC bus: 947 kW at the 815 V floor. The GS1.1 pair
    at 1.04 % charge (NFPP OCV 815 V), the load on the 505 V bus so each PCS
    delivers its 947 kW: its DC side draws 947 kW / 815 V, with its 98.5 %
    efficiency on top (T2's 9 kW no-load loss adds 0.2 %).
    """
    res = _solve(_block(load_mw=4 * S_BUS, soc=1.036, at_505=True), quiet)
    for k in range(1, 5):
        g = _by(res['pcs'])[f'GFM{k}']
        assert g['p_mw'] == pytest.approx(S_BUS, rel=0.003)
        assert g['v_dc_v'] == pytest.approx(815, abs=3)
        assert g['p_mw'] / g['v_dc_v'] * 1e3 == pytest.approx(1.162, abs=0.008)
        assert g['p_dc_mw'] / g['v_dc_v'] * 1e3 == pytest.approx(1.162 / 0.985, abs=0.008)


def test_derated_at_50c_the_batteries_carry_the_rest(quiet):
    """
    At 50 deg C each rectifier is 95 % of 947 kVA, 900 kVA: asked for 940 kW
    a bus it delivers its rating, said so, and the GS1.1 pair carries the rest;
    the four PCS, also 900 kVA, share the load equally by droop, each over its
    rating - within its 10 s overload.
    """
    res = _solve(_block(inlet=50.0, load_mw=4 * 0.94, at_505=True), quiet)
    vscs, pcs = _by(res['vscs']), _by(res['pcs'])
    stores = [s for s in res['sources_and_stores'] if s['kind'] == 'Battery']
    for k in range(1, 5):
        a, g = vscs[f'AFE{k}'], pcs[f'GFM{k}']
        assert a['rated_mva'] == pytest.approx(0.95 * S_BUS) and a['derating_factor'] == pytest.approx(0.95)
        assert a['loading_percent'] == pytest.approx(100.0, rel=1e-4)        # at its rating, at its bus's voltage
        assert a['p_mw'] == pytest.approx(0.95 * S_BUS * min(1.0, res_vm(res, 'GLV')), rel=1e-4)
        assert g['s_rated_mva'] == pytest.approx(0.95 * S_BUS) and g['s_nameplate_mva'] == pytest.approx(S_BUS)
        assert g['p_mw'] == pytest.approx(pcs['GFM1']['p_mw'], abs=1e-6) and g['loading_percent'] > 104
    assert sum(s['p_mw'] for s in stores) == pytest.approx(
        sum(g["p_dc_mw"] for g in pcs.values()) + sum(a["p_dc_mw"] for a in vscs.values()), abs=1e-4)   # its resistance's loss aside
    assert all(s['p_mw'] > 0.01 for s in stores)
    warnings = ' '.join(res['warnings'])
    assert "VSC 'AFE1' follows its DC bus's load to its rating" in warnings
    assert "within its short-time overload for 10 s, not continuously" in warnings


def res_vm(res, bus):
    return next(b['vm_pu'] for b in res['buses'] if b['id'] == bus)


def test_one_quadrant_out(quiet):
    """
    DC4 isolated - the spec's N-1 inside a block, 3/4 of its rating: the other
    three take its share by droop, each now over its continuous rating at 45 deg C.
    """
    res = _solve(_block(load_mw=3.0, at_505=True, out=(4,)), quiet)
    p = [g['p_mw'] for g in res['pcs']]
    assert len(p) == 3 and sum(p) == pytest.approx(3.0, rel=1e-3)
    assert max(p) - min(p) < 1e-6 and p[0] > S_BUS
    assert any("PCS 'GFM1' is loaded to" in w for w in res['warnings'])


def test_reactive_power_capped_at_60_percent(quiet):
    """0.8 Mvar asked of a 947 kVA PCS at 0.3 MW: its circle allows 0.898, its cap 0.568 (0.6 S)."""
    res = _solve(_block(q_set=0.8), quiet)
    for g in res['pcs']:
        assert g['p_mw'] == pytest.approx(0.3, rel=1e-6) and g['q_mvar'] == pytest.approx(0.6 * S_BUS, rel=1e-5)
        assert g['q_capability_mvar'] == pytest.approx(0.6 * S_BUS, rel=1e-5)
    assert any('within 60 % of it' in w for w in res['warnings'])


# --- Negative-sequence control in the EMT study --------------------------------------------------

F = 60.0
W = 2 * math.pi * F
V_LL, S = 505.0, S_BUS * 1e6
V_DC = 1000.0


def _unbalanced_island(control, ll_pu=0.3):
    """
    One grid-forming converter, its DC side stiff, feeding a 0.6 pu balanced
    star load and - from 20 ms - 0.3 pu line to line (a-b), on a three-wire bus.
    """
    ckt = es.Circuit()
    zb = V_LL ** 2 / S
    r, l = 0.01 * zb, 0.15 * zb / W
    zl = V_LL ** 2 / (0.6 * S)
    star = ckt.node('star')
    ckt.add_r(star, 0, 1e6)
    bus = [ckt.node(f'bus {k}') for k in range(3)]
    for k in range(3):
        ckt.add_r(bus[k], star, zl)
    mid = ckt.node('line-to-line load')
    sw = ckt.add_switch(bus[0], mid, closed=False, r_on=1e-4, r_off=1e9)
    ckt.add_r(mid, bus[1], V_LL ** 2 / (ll_pu * S))
    v_ph = math.sqrt(2 / 3) * V_LL
    i_ph = v_ph / zl
    e_ph = complex(v_ph + r * i_ph, W * l * i_ph)
    ang = -np.arange(3) * A120
    k_e = [ckt.add_rl(0, bus[k], r, l, ac=(abs(e_ph), W, np.angle(e_ph) + ang[k]), controlled=True) for k in range(3)]
    dc = ckt.node('DC link', V_DC)
    ckt.add_c(0, dc, 0.02, w0=-V_DC)
    k_src = ckt.add_isrc(0, dc, i0=0.0)
    k_dc = ckt.add_rl(0, dc, 1e-3, 0.0, e0=V_DC)
    i_max = 1.2 * math.sqrt(2) * S / (math.sqrt(3) * V_LL)
    ctrl = GridFormingVsc.standalone(
        unbalance_control=control, i2_limit_pu=0.3,
        label='GFM', w0=W, ac_nodes=bus, k_e=k_e, k_block=[], k_src=k_src, k_dc_out=k_dc, p_node=dc,
        r=r, l=l, i_max=i_max, v_nom_peak=v_ph, block_v=0.0, block_i=1e12, mode_dc='p_mw', mode_ac='q_mvar',
        s_rated=S, v_ph=complex(v_ph, 0), i_ph=complex(i_ph, 0), e_ph=e_ph, c_link=0.02, v_link=V_DC, i_dc=0.0)
    ckt.at(0.02, lambda st: st.set_switch(sw, True) or True)
    ckt.add_controller(ctrl.control)
    ckt.start_in_ac_steady_state(W)
    sim = ckt.simulate(0.4, 2e-5)
    return sim, ctrl, bus, k_dc


def _fundamental(t, y, h=1):
    """The h-th harmonic's complex amplitude of y over the run's last cycle, and its mean."""
    last = t >= t[-1] - 1 / F
    x = np.column_stack([np.cos(h * W * t[last]), np.sin(h * W * t[last]), np.ones(last.sum())])
    c, *_ = np.linalg.lstsq(x, y[last], rcond=None)
    return complex(c[0], -c[1]), c[2]


def _sequences(t, v):
    ph = [_fundamental(t, v[:, k])[0] for k in range(3)]
    a = complex(math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3))
    return abs(ph[0] + a * ph[1] + a * a * ph[2]) / 3, abs(ph[0] + a * a * ph[1] + a * ph[2]) / 3


def test_grid_forming_converter_holds_negative_sequence_voltage_at_zero():
    """
    A 0.3 pu line-to-line load on a three-wire bus: left alone the bus's
    unbalance is 4.6 %, over the spec's 2 %; with its unbalance control the
    converter carries the negative-sequence current (within its 0.3 pu
    limit) and the bus holds under 0.5 %. Its DC side carries the 2f ripple a
    line-to-line load always makes - 1.5 |V1| |I2|, some 280 kW here.
    """
    sim, ctrl, bus, k_dc = _unbalanced_island(False)
    v1, v2 = _sequences(sim['t'], sim['v'][:, bus])
    assert 100 * v2 / v1 == pytest.approx(4.6, abs=0.3)
    sim, ctrl, bus, k_dc = _unbalanced_island(True)
    t = sim['t']
    v1, v2 = _sequences(t, sim['v'][:, bus])
    assert 100 * v2 / v1 < 0.5
    i2 = ctrl.neg_trace[-1][2]
    assert 0 < i2 <= ctrl.i2_max * (1 + 1e-9)
    ripple, _ = _fundamental(t, sim['i_rl'][:, k_dc] * V_DC, h=2)
    assert abs(ripple) == pytest.approx(1.5 * v1 * i2, rel=0.1)


# --- On the drawn network: an asymmetric load, the converter's unbalance results -------------------

ASYM = {'typ': 'Asymmetric Load0', 'name': 'ul', 'id': 'cell-ul', 'userFriendlyName': 'UL', 'bus': 'b',
        'p_a_mw': '0.12', 'p_b_mw': '0', 'p_c_mw': '0', 'q_a_mvar': '0', 'q_b_mvar': '0', 'q_c_mvar': '0',
        'sn_mva': '0.2', 'scaling': '1', 'type': 'delta', 'in_service': 'true'}


def _emt_island(client, quiet, control):
    from test_emt_pcs import _emt, _post as _post_emt
    from test_pcs import _two_buses
    return _post_emt(client, quiet, _two_buses(
        _der('Battery', 'b1', capacity_kwh=2000), _pcs('p1', 'a', 'b1', control='grid_forming', s_rated_mva=1.0,
                                                       unbalance_control='true' if control else 'false'),
        dict(ASYM), grid=False, params=_emt(time_step_us=20, duration_ms=300)))


def test_drawn_asymmetric_load_and_unbalance_control(client, quiet):
    """
    An islanded grid-forming PCS, 0.6 MW balanced and 0.12 MW line to line (a-b)
    on 0.4 kV: the EMT study now builds the asymmetric load (it was left out).
    Its control holds its own bus near zero and the far one under 2 % - the
    line between them carries the negative-sequence current; off, the far bus is over.
    """
    off = _emt_island(client, quiet, False)
    on = _emt_island(client, quiet, True)
    vuf = lambda emt, b: next(x['vuf_end_percent'] for x in emt['ac']['buses'] if x['label'] == b)   # noqa: E731
    assert vuf(off, 'B') > 2.0
    assert vuf(on, 'A') < 0.5 and vuf(on, 'B') < 2.0
    (conv,) = [c for c in on['converters'] if c['label'] == 'P1']
    assert conv['unbalance_control'] and conv['i2_end_ka'] > 0 and conv['vuf_end_percent'] < 1.0
    assert not any('asymmetric loads' in w for w in on.get('warnings', []))
