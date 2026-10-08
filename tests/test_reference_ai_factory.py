# -*- coding: utf-8 -*-
"""
The 800 VDC AI factory reference: tests/reference/reference_ai_factory_800vdc.spec.json,
GE Vernova's "AI Factory 800 VDC Reference Designs" as a network (the design
note "800 VDC AI factory reference"). A 245 kV, 60 Hz grid through two 40 MVA
transformers, 400 A resistance-grounded at their 34.5 kV neutrals, onto two
buses with the tie open; a 35 MW gas turbine, a substation BESS of four DC
Stores behind grid-forming PCS and an eSTATCOM (supercapacitors behind a
grid-forming PCS); a 4 MW cooling plant. Hall A is the paper's Figure 4:
Lineups A and B and a Catcher B, each four 2 MW rectifiers sharing an 800 V
bus (N+1) with a DC Store, a DC/DC bus coupler between A and B, and twelve
1 MW shelves each fed through a breaker and a diode from its lineup and
through a diode from the catcher, whose bus sits 2 % low; one shelf drawn to
the GPU at 50 V and 12 V. Hall B is Figure 3: three 5 MW SSTs (2+1) on one
800 V bus, ten 1 MW racks through solid-state breakers, DC Stores and
supercapacitors. 12 + 10 MW of IT.

Its oracle, in three parts:

1. the conversion chain in closed form - each diode its forward drop and
   on-resistance, the higher bus carrying each shelf and the catcher's diodes
   blocking; each rectifier, SST and DC/DC converter its efficiency and losses;
   converters in parallel sharing equally; the grid the balance;
2. the AC network built by hand with pandapower, every default written out,
   each converter as its AC draw: the same voltages and flows;
3. with the PCS as current sources at their limits, the same IEC fault
   currents at every AC bus, three- and single-phase, max and min, the 400 A
   neutral resistors folded in.

Goldens pin the rest. Regenerate with:  pytest --regen-golden
"""
import contextlib
import io
import json
import math
import os

import numpy as np
import pandapower as pp
import pandapower.shortcircuit as sc
import pytest

import electrisim_sld as sld
import electrisim_spec_layer as layer
import pandapower_electrisim as pe

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = 'reference_ai_factory_800vdc'
SPEC_PATH = os.path.join(HERE, 'reference', f'{GRID}.spec.json')
GOLDEN_PATH = os.path.join(HERE, 'golden', f'{GRID}.json')

LF = {'typ': 'PowerFlowPandaPower Parameters'}
V0 = 0.8                                    # the 800 V buses, kV


def sc_params(fault='3ph', case='max'):
    return {'typ': 'ShortCircuitPandaPower Parameters', 'fault_type': fault, 'fault_location': case,
            'fault_bus_mode': 'all', 'fault_bus_ids': [], 'fault_bus_names': [], 'fault_impedance': '6',
            'topology': 'auto', 'tk_s': '1', 'r_fault_ohm': '0', 'x_fault_ohm': '0', 'inverse_y': 'True'}


@contextlib.contextmanager
def _silent():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield


def load_spec():
    with open(SPEC_PATH, encoding='utf-8') as handle:
        return json.load(handle)


@pytest.fixture(scope='module')
def factory():
    spec = load_spec()
    with _silent():
        net, report = sld.build_network(spec)
    return spec, net, report


def _layered(net, params=LF):
    return layer.with_layer(net, net['electrisim_layer'], net.get('electrisim_bus_index') or {}, params)


@pytest.fixture(scope='module')
def solved(factory):
    """The factory's load flow, as Electrisim runs it: the layer settled, voltage angles 'auto'."""
    _, net, _ = factory
    full = _layered(net)
    with _silent():
        pe._electrisim_runpp(full, calculate_voltage_angles='auto', init='auto')
    assert full.converged and not getattr(full, 'warnings', None), getattr(full, 'warnings', None)
    return full


def _results(full):
    """The layer's results under their spec ids, unrounded."""
    out = layer.results(full, lambda v, nd=6: None if v is None else float(v))
    return {key: {row['id']: row for row in rows} if isinstance(rows, list) and rows and 'id' in rows[0] else rows
            for key, rows in out.items()}


def _ac_ids(net):
    return {table: {ident: idx for idx, ident in mapping.items()} for table, mapping in net['electrisim_ids'].items()}


def _rows_by_id(spec):
    return {row['id']: row for key, rows in spec.items() if isinstance(rows, list) for row in rows if 'id' in row}


def _default(kind, field, row):
    return float(row.get(field, layer.DEFAULTS[kind][field]))


# --- what it is ------------------------------------------------------------------

def test_builds_cleanly_and_is_the_factory_of_the_note(factory):
    """No warnings; the halls, redundancy, storage and grounding the design note sets out."""
    spec, net, report = factory
    assert report['warnings'] == [], report['warnings']
    assert spec['frequency_hz'] == 60 and net.f_hz == 60
    rows = _rows_by_id(spec)
    hall_a = [r for r in spec['dc_loads'] if r['bus'] != 'HB']
    hall_b = [r for r in spec['dc_loads'] if r['bus'] == 'HB']
    assert sum(r['p_mw'] for r in hall_a) == pytest.approx(12.0) and sum(r['p_mw'] for r in hall_b) == pytest.approx(10.0)
    # Four 2 MW rectifiers on each lineup's bus, one more than its 6 MW needs; the catcher's 2 % low.
    for x in 'ABC':
        rect = [v for v in spec['vscs'] if v['bus_dc'] == f'DC{x}']
        assert len(rect) == 4 and {v['control_value_dc'] for v in rect} == {0.98 if x == 'C' else 1.0}
    # Every shelf: a diode from its lineup behind a breaker, and one from its catcher group.
    for g in 'AB':
        for k in range(1, 7):
            feeds = {d['from_bus'] for d in spec['dc_diodes'] if d['to_bus'] == f'S{g}{k}'}
            assert feeds == {f'DC{g}', f'CG{g}'}
            assert any(b['element'] == f'D_{g}{k}' and b['bus'] == f'DC{g}' for b in spec['dc_breakers'])
    # Diode drop 0.2 % at 800 V, as the paper's.
    assert {d['v_f_v'] for d in spec['dc_diodes']} == {1.6}
    # Hall B: three 5 MW SSTs for 10 MW (2+1), split over the two buses; every rack behind a breaker.
    assert sorted(s['bus_mv'] for s in spec['ssts']) == ['BUS1', 'BUS2', 'BUS2']
    assert {b['element'] for b in spec['dc_breakers'] if b['bus'] == 'HB'} == {r['id'] for r in hall_b}
    # The 34.5 kV grounded through 400 A at T1 and T2; no other 34.5 kV winding grounded.
    for t in spec['transformers']:
        if t['id'] in ('T1', 'T2'):
            assert 34.5e3 / math.sqrt(3) / t['rn_ohm'] == pytest.approx(400, rel=1e-4)
        elif rows[t['hv_bus']]['vn_kv'] == 34.5:
            assert t['vector_group'] in ('Yd', 'Dyn'), t['id']
    tie = rows['CB_TIE']
    assert tie['et'] == 'bus' and tie['closed'] is False
    # The eSTATCOM: 7.5 MW-s down to half its voltage behind a 15 MVA grid-forming PCS, no set power.
    est = rows['SC_EST']
    assert 0.5 * est['c_f'] * est['v_rated'] ** 2 * 0.75 == pytest.approx(7.5e6, rel=1e-4)
    assert (rows['PCS_EST']['control'], rows['PCS_EST']['p_set_mw']) == ('grid_forming', 0)
    # The back-up gensets off; the cooling plant 4 MW.
    assert all(g['in_service'] is False for g in spec['generators'] if g['id'] != 'GT')
    cooling = sum(m['pn_mech_mw'] / m['efficiency_percent'] * 100 for m in spec['motors']) + rows['COOL_AUX']['p_mw']
    assert cooling == pytest.approx(4.0)


# --- 1. the conversion chain in closed form ------------------------------------------

def _dcdc_input(p_out, eta, p_nl):
    return p_out / eta + p_nl if p_out >= 0 else p_out * eta + p_nl


def _diode_current(v_anode_kv, v_f_v, r_on_ohm, p_mw):
    """A diode delivering p_mw to its cathode from an anode at v_anode_kv: (v_a - v_f - I r_on) I = P, in kA."""
    v = v_anode_kv - v_f_v / 1e3
    return (v - math.sqrt(v * v - 4.0 * r_on_ohm * p_mw)) / (2.0 * r_on_ohm)


def test_conversion_chain_in_closed_form(factory, solved):
    """
    The higher bus carries each shelf through its diode, its forward drop and
    on-resistance; the catcher's diodes block, its rectifiers idle. Each
    converter draws what its efficiency and losses say; converters in
    parallel share equally; the grid supplies the balance.
    """
    spec, net, _ = factory
    rows = _rows_by_id(spec)
    res = _results(solved)
    dcdc, diodes, dc_bus = res['dc_dc_converters'], res['dc_diodes'], res['dc_buses']

    # Every DC/DC converter: its input its output over its efficiency, plus its no-load loss.
    for ident, conv in dcdc.items():
        row = rows[ident]
        eta = _default('DC/DC Converter', 'efficiency_percent', row) / 100
        p_nl = _default('DC/DC Converter', 'no_load_loss_kw', row) / 1e3
        assert conv['p_in_mw'] == pytest.approx(_dcdc_input(conv['p_out_mw'], eta, p_nl), abs=1e-9), ident
    # The stores idle (droop at their set voltage, smoothing nothing), the bus coupler at no power.
    for ident, conv in dcdc.items():
        if ident not in ('DD_PSU', 'DD_VRM'):
            assert conv['p_out_mw'] == pytest.approx(0.0, abs=1e-9), ident
    # The rack: its GPUs at 12 V, through the 50 / 12 V converters and the 800 / 50 V supply.
    assert dcdc['DD_VRM']['p_out_mw'] == pytest.approx(1.0, abs=1e-9)
    assert dc_bus['RK_50']['vm_pu'] == pytest.approx(1.0) and dc_bus['RK_12']['vm_pu'] == pytest.approx(1.0)
    assert dcdc['DD_PSU']['p_out_mw'] == pytest.approx(dcdc['DD_VRM']['p_in_mw'], abs=1e-9)

    # Each shelf through its lineup's diode: the anode at 800 V, the shelf a drop below.
    sent = {'A': 0.0, 'B': 0.0}
    for g in 'AB':
        for k in range(1, 7):
            d = rows[f'D_{g}{k}']
            p = dcdc['DD_PSU']['p_in_mw'] if (g, k) == ('A', 1) else 1.0
            i_ka = _diode_current(V0, d['v_f_v'], d['r_on_mohm'] / 1e3, p)
            got = diodes[f'D_{g}{k}']
            assert got['conducting'] and got['i_ka'] == pytest.approx(i_ka, abs=1e-6), (g, k)
            v_shelf = V0 - d['v_f_v'] / 1e3 - i_ka * d['r_on_mohm'] / 1e3
            assert dc_bus[f'S{g}{k}']['vm_pu'] == pytest.approx(v_shelf / V0, abs=1e-8), (g, k)
            # The catcher's diode blocks: its anode 784 V, below the shelf.
            back = diodes[f'D_C{g}{k}']
            assert not back['conducting'] and back['i_ka'] == 0.0
            assert back['v_ak_v'] == pytest.approx(0.98 * 800 - v_shelf * 1e3, abs=1e-3), (g, k)
            sent[g] += V0 * i_ka
    for g in 'AB':
        assert res['dc_lines'][f'L_CG{g}']['i_ka'] == pytest.approx(0.0, abs=1e-9)
    sent['A'] += dcdc['DD_TIE']['p_in_mw']                                  # the coupler's no-load loss
    sent['B'] -= dcdc['DD_TIE']['p_out_mw']

    # Each lineup's four rectifiers share its bus equally: DC power plus I^2 R on each side.
    ac_v = {ident: float(solved.res_bus.at[idx, 'vm_pu']) for ident, idx in _ac_ids(net)['bus'].items()}
    for x, total in (('A', sent['A']), ('B', sent['B']), ('C', 0.0)):
        for k in range(1, 5):
            ident = f'R_{x}{k}'
            row, got = rows[ident], res['vscs'][ident]
            p_dc = total / 4
            assert -got['p_dc_mw'] == pytest.approx(p_dc, abs=1e-6), ident
            i_dc = p_dc / (V0 * row['control_value_dc'])
            p_int = p_dc + i_dc * i_dc * row['r_dc_ohm']
            i_ac = got['p_mw'] / (math.sqrt(3) * ac_v[row['bus']] * 0.48)        # unity power factor
            assert got['p_mw'] == pytest.approx(p_int + 3 * i_ac * i_ac * row['r_ohm'], abs=2e-4), ident
            assert got['q_mvar'] == pytest.approx(0.0, abs=1e-6)

    # Hall B: three SSTs share its ten racks equally, each its DC/DC stage then its rectifier.
    def sst_mv(ident, p_dc):
        row = rows[ident]
        link = _dcdc_input(p_dc, row['dcdc_efficiency_percent'] / 100, row['dcdc_no_load_kw'] / 1e3)
        return _dcdc_input(link, row['rect_efficiency_percent'] / 100, row['rect_no_load_kw'] / 1e3)

    for ident in ('U1', 'U2', 'U3'):
        assert res['ssts'][ident]['p_mv_mw'] == pytest.approx(sst_mv(ident, 10.0 / 3), abs=1e-6), ident

    # Each grid-forming PCS: no active power, its bus on its Q-V droop line.
    for ident, p in res['pcs'].items():
        row = rows[ident]
        assert p['p_mw'] == pytest.approx(0.0, abs=1e-9), ident
        droop = row['droop_qv_percent'] / 100
        assert p['vm_pu'] == pytest.approx(1 - droop * p['q_mvar'] / row['s_rated_mva'], abs=1e-5), ident
    assert res['sources_and_stores']['SC_EST']['v_cap_v'] == pytest.approx(1350.0)

    # The grid supplies the balance.
    ids = _ac_ids(net)
    gen = np.nansum([float(solved.res_gen.at[i, 'p_mw']) for i in ids['gen'].values()])
    loads = (sum(float(solved.res_load.at[i, 'p_mw']) for i in ids['load'].values())
             + sum(float(solved.res_motor.at[i, 'p_mw']) for i in ids['motor'].values()))
    conv = sum(s['p_mv_mw'] for s in res['ssts'].values()) + sum(v['p_mw'] for v in res['vscs'].values())
    losses = (sum(float(solved.res_line.at[i, 'pl_mw']) for i in ids['line'].values())
              + float(np.nansum(solved.res_trafo['pl_mw'])))
    pcs_mw = sum(p['p_mw'] for p in res['pcs'].values())
    grid = float(solved.res_ext_grid.at[ids['ext_grid']['Grid'], 'p_mw'])
    assert grid == pytest.approx(loads + conv + losses - gen - pcs_mw, abs=1e-6)
    assert gen == pytest.approx(20.0) and 6 < grid < 12                      # the base case's import, about 8.6 MW


# --- 2. the AC network by hand ---------------------------------------------------------

def hand_built_ac(spec, equivalents=None, current_sources=None):
    """
    reference_ai_factory_800vdc's AC network at 60 Hz with the documented
    defaults written out, each converter as the load or source its AC side is.
    equivalents: {AC bus id: (p_mw, q_mvar)} loads standing for what the layer
    draws there (negative: delivers). current_sources: [(AC bus id, sn_mva, k)]
    for the PCS in a short circuit.
    """
    net = pp.create_empty_network(f_hz=60.0)
    kv = {r['id']: r['vn_kv'] for r in spec['buses']}
    b = {r['id']: pp.create_bus(net, vn_kv=r['vn_kv'], name=r['id']) for r in spec['buses']}
    ids = {'bus': dict(b), 'trafo': {}}
    g = spec['external_grids'][0]
    pp.create_ext_grid(net, b[g['bus']], vm_pu=1.0, va_degree=0.0, s_sc_max_mva=g['s_sc_max_mva'], rx_max=g['rx_max'],
                       s_sc_min_mva=g['s_sc_min_mva'], rx_min=g['rx_min'], x0x_max=1.0, r0x0_max=0.1, x0x_min=1.0,
                       r0x0_min=0.1)
    for r in spec['transformers']:
        i = pp.create_transformer_from_parameters(
            net, b[r['hv_bus']], b[r['lv_bus']], sn_mva=r['sn_mva'], vn_hv_kv=kv[r['hv_bus']], vn_lv_kv=kv[r['lv_bus']],
            vk_percent=r['vk_percent'], vkr_percent=r['vkr_percent'], pfe_kw=0.6 * r['sn_mva'], i0_percent=0.1,
            shift_degree=r['shift_degree'], vector_group=r['vector_group'], vk0_percent=r['vk_percent'],
            vkr0_percent=r['vkr_percent'], mag0_percent=100.0, mag0_rx=0.0, si0_hv_partial=0.9)
        net.trafo.loc[i, ['rn_ohm', 'xn_ohm']] = (r.get('rn_ohm', 0.0), 0.0)
        if 'tap_pos' in r:                       # the OLTC, at its neutral
            net.trafo.loc[i, ['tap_side', 'tap_neutral', 'tap_min', 'tap_max', 'tap_step_percent', 'tap_pos',
                              'tap_step_degree', 'tap_changer_type']] = (
                'hv', 0, -8, 8, r['tap_step_percent'], r['tap_pos'], 0.0, 'Ratio')
        ids['trafo'][r['id']] = i
    for r in spec['loads']:
        pp.create_load(net, b[r['bus']], p_mw=r['p_mw'], q_mvar=r['q_mvar'])
    for r in spec['motors']:
        pp.create_motor(net, b[r['bus']], pn_mech_mw=r['pn_mech_mw'], cos_phi=r['cos_phi'],
                        efficiency_percent=r['efficiency_percent'], lrc_pu=r['lrc_pu'], rx=r['rx'], vn_kv=4.16,
                        cos_phi_n=r['cos_phi'], efficiency_n_percent=r['efficiency_percent'])
    for r in spec['generators']:
        pp.create_gen(net, b[r['bus']], p_mw=r['p_mw'], vm_pu=1.0, sn_mva=r['sn_mva'], vn_kv=kv[r['bus']],
                      xdss_pu=r['xdss_pu'], rdss_ohm=r['rdss_ohm'], cos_phi=r['cos_phi'],
                      in_service=r.get('in_service', True))
    for r in spec['switches']:
        if r['et'] == 'transformer':
            pp.create_switch(net, b[r['bus']], ids['trafo'][r['element']], et='t', closed=r.get('closed', True))
        elif r['et'] == 'bus':
            pp.create_switch(net, b[r['bus']], b[r['element']], et='b', closed=r.get('closed', True))
    for bus, (p, q) in (equivalents or {}).items():
        pp.create_load(net, b[bus], p_mw=p, q_mvar=q)
    for bus, sn_mva, k in current_sources or ():
        pp.create_sgen(net, b[bus], p_mw=0.0, sn_mva=sn_mva, k=k, generator_type='current_source')
    return net, ids


def _equivalents(spec, res):
    """What the layer draws at each AC bus: SSTs and rectifiers draw, PCS deliver."""
    rows = _rows_by_id(spec)
    eq = {}

    def add(bus, p, q):
        eq[bus] = (eq.get(bus, (0.0, 0.0))[0] + p, eq.get(bus, (0.0, 0.0))[1] + q)

    for ident, s in res['ssts'].items():
        add(rows[ident]['bus_mv'], s['p_mv_mw'], s['q_mv_mvar'])
    for ident, v in res['vscs'].items():
        add(rows[ident]['bus'], v['p_mw'], v['q_mvar'])
    for ident, p in res['pcs'].items():
        add(rows[ident]['bus'], -p['p_mw'], -p['q_mvar'])
    return eq


def test_ac_network_matches_hand_built(factory, solved):
    """The AC network the spec builds, with the layer's draws at its buses, is the one written out by hand."""
    spec, net, _ = factory
    oracle, oids = hand_built_ac(spec, _equivalents(spec, _results(solved)))
    pp.runpp(oracle, calculate_voltage_angles='auto', init='auto')
    ids = _ac_ids(net)
    differ = []
    for ident, idx in ids['bus'].items():
        for col, tol in (('vm_pu', 1e-8), ('va_degree', 1e-6)):
            got, want = float(solved.res_bus.at[idx, col]), float(oracle.res_bus.at[oids['bus'][ident], col])
            if abs(got - want) > tol:
                differ.append(f'bus {ident} {col}: {want} by hand, {got} built')
    for ident, idx in ids['trafo'].items():
        for col in ('p_hv_mw', 'q_hv_mvar', 'loading_percent'):
            got, want = float(solved.res_trafo.at[idx, col]), float(oracle.res_trafo.at[oids['trafo'][ident], col])
            if abs(got - want) > 1e-6:
                differ.append(f'trafo {ident} {col}: {want} by hand, {got} built')
    for col in ('p_mw', 'q_mvar'):
        if abs(float(solved.res_ext_grid.at[ids['ext_grid']['Grid'], col]) - float(oracle.res_ext_grid.at[0, col])) > 1e-6:
            differ.append(f'grid {col}')
    assert not differ, '\n  '.join(differ[:20])


# --- 3. faults by hand -----------------------------------------------------------------

_SC = {}


def _study_sc(net, fault, case):
    """Electrisim's IEC short circuit on the factory with its layer, {bus id: row}; each case run once."""
    if (fault, case) not in _SC:
        full = _layered(net, sc_params(fault, case))
        with _silent():
            out = json.loads(pe.shortcircuit(full, sc_params(fault, case), None))
        assert not out.get('error'), out.get('message')
        names = {str(full.bus.at[i, 'name']): ident for i, ident in net['electrisim_ids']['bus'].items()}
        _SC[fault, case] = {names[r['name']]: r for r in out['busbars'] if r['name'] in names}
    return _SC[fault, case]


def _pcs_current_sources(spec):
    return [(p['bus'], p['s_rated_mva'], float(p.get('current_limit_pu', 1.2))) for p in spec['pcs']]


def _fold(net):
    """The documented IEC rule: 3 Z_N joins its grounded winding's zero sequence, uncorrected by K_T."""
    for i in net.trafo.index:
        rn = float(net.trafo.at[i, 'rn_ohm'])
        if not rn > 0:
            continue
        row = net.trafo.loc[i]
        vk, vkr, sn, vn = row.vk_percent, row.vkr_percent, row.sn_mva, row.vn_lv_kv      # Dyn: on the LV
        k_t = 0.95 * 1.1 / (1 + 0.6 * math.sqrt(vk ** 2 - vkr ** 2) / 100)
        z0 = complex(row.vkr0_percent, math.sqrt(row.vk0_percent ** 2 - row.vkr0_percent ** 2))
        z0 += 100 * 3 * rn * sn / vn ** 2 / k_t
        net.trafo.loc[i, ['vk0_percent', 'vkr0_percent']] = (abs(z0), z0.real)


@pytest.mark.parametrize('fault, case', [('3ph', 'max'), ('3ph', 'min'), ('1ph', 'max'), ('1ph', 'min')])
def test_short_circuit_matches_hand_built(factory, fault, case):
    """
    Every AC bus's IEC 60909 fault: the network by hand, the turbine and
    chillers as machines, each PCS a current source at its limit (1.2 x its
    rating) - the SSTs and rectifiers add nothing - and, earth faults, the
    34.5 kV neutral resistors folded in.
    """
    spec, net, _ = factory
    got = _study_sc(net, fault, case)
    oracle, oids = hand_built_ac(spec, current_sources=_pcs_current_sources(spec))
    if fault == '1ph':
        _fold(oracle)
    with _silent():
        pp.runpp(oracle, calculate_voltage_angles='auto', init='auto')
        sc.calc_sc(oracle, fault=fault, case=case, ip=True, ith=True, tk_s=1.0, kappa_method='C', r_fault_ohm=0.0,
                   x_fault_ohm=0.0, check_connectivity=True, branch_results=False, lv_tol_percent=6,
                   topology='auto', inverse_y=True)
    differ = []
    for ident, idx in oids['bus'].items():
        want = float(oracle.res_bus_sc.at[idx, 'ikss_ka'])
        if abs(float(got[ident]['ikss_ka']) - want) > 1e-6 * max(1.0, want):
            differ.append(f'{ident}: {want} by hand, {got[ident]["ikss_ka"]} by the study')
    assert not differ, '\n  '.join(differ)


def test_ground_faults_by_design(factory):
    """
    A 34.5 kV earth fault is held by its transformer's 400 A resistor: c x
    400 A through the neutral (0.438 kA by hand, the transformer's own
    impedance aside), the PCS on its bus adding about their current limit
    (within 3 %). A 245 kV earth fault is the grid's alone, near its three-phase.
    """
    spec, net, _ = factory
    one, three = _study_sc(net, '1ph', 'max'), _study_sc(net, '3ph', 'max')
    for bus, pcs_bus in (('BUS1', 'BESS_LV'), ('BUS2', 'EST_LV')):
        pcs_ka = sum(p['s_rated_mva'] * 1.2 for p in spec['pcs'] if p['bus'] == pcs_bus) / (math.sqrt(3) * 34.5)
        assert one[bus]['ikss_ka'] == pytest.approx(1.1 * 0.4 + pcs_ka, rel=0.03), bus
        assert one[bus]['ikss_ka'] < 0.15 * three[bus]['ikss_ka']                # some 6 kA
    assert 0.8 * three['GRID']['ikss_ka'] < one['GRID']['ikss_ka'] < 1.2 * three['GRID']['ikss_ka']
    # The grid's 5 GVA its fault level (c in it), the site's machines and PCS a few percent more.
    grid_ka = 5000 / (math.sqrt(3) * 245)
    assert grid_ka < three['GRID']['ikss_ka'] < 1.05 * grid_ka


# --- goldens and the canvas ---------------------------------------------------------------

def _summary(net, solved):
    res = _results(solved)
    ids = _ac_ids(net)
    on = {i: x for i, x in ids['gen'].items() if bool(solved.gen.at[x, 'in_service'])}
    out = {
        'bus': {i: {'vm_pu': float(solved.res_bus.at[x, 'vm_pu']), 'va_degree': float(solved.res_bus.at[x, 'va_degree'])}
                for i, x in ids['bus'].items()},
        'trafo': {i: {'p_hv_mw': float(solved.res_trafo.at[x, 'p_hv_mw']),
                      'loading_percent': float(solved.res_trafo.at[x, 'loading_percent'])} for i, x in ids['trafo'].items()},
        'gen': {i: {'q_mvar': float(solved.res_gen.at[x, 'q_mvar'])} for i, x in on.items()},
        'ext_grid': {i: {'p_mw': float(solved.res_ext_grid.at[x, 'p_mw']),
                         'q_mvar': float(solved.res_ext_grid.at[x, 'q_mvar'])} for i, x in ids['ext_grid'].items()},
        'dc_bus': {i: {'vm_pu': b['vm_pu']} for i, b in res['dc_buses'].items()},
        'dc_diode': {i: {'i_ka': d['i_ka'], 'v_ak_v': d['v_ak_v']} for i, d in res['dc_diodes'].items()},
        'sst': {i: {'p_mv_mw': s['p_mv_mw']} for i, s in res['ssts'].items()},
        'vsc': {i: {'p_mw': v['p_mw']} for i, v in res['vscs'].items()},
        'pcs': {i: {'p_mw': p['p_mw'], 'q_mvar': p['q_mvar']} for i, p in res['pcs'].items()},
        'dcdc': {i: {'p_in_mw': c['p_in_mw']} for i, c in res['dc_dc_converters'].items()},
    }
    for fault in ('3ph', '1ph'):
        for case in ('max', 'min'):
            rows = _study_sc(net, fault, case)
            out[f'bus_sc_{fault}_{case}'] = {i: {c: float(r[c]) for c in ('ikss_ka', 'ip_ka', 'ith_ka')}
                                             for i, r in rows.items()}
    return {t: {i: {c: round(v, 9) for c, v in cols.items()} for i, cols in rows.items()} for t, rows in out.items()}


def test_matches_golden(regen, factory, solved):
    actual = _summary(factory[1], solved)
    if regen:
        with open(GOLDEN_PATH, 'w', encoding='utf-8', newline='\n') as handle:
            json.dump(actual, handle, indent=2, sort_keys=True)
        pytest.skip(f'regenerated {os.path.basename(GOLDEN_PATH)}')
    assert os.path.exists(GOLDEN_PATH), 'missing golden - create it with: pytest --regen-golden'
    with open(GOLDEN_PATH, encoding='utf-8') as handle:
        expected = json.load(handle)
    drifted = []
    for table, rows in expected.items():
        assert set(actual[table]) == set(rows), f'{table}: element set changed'
        for ident, values in rows.items():
            for col, want in values.items():
                tol = 1e-8 if col == 'vm_pu' else 1e-6
                if abs(actual[table][ident][col] - want) > tol:
                    drifted.append(f'{table} {ident} {col}: {want} -> {actual[table][ident][col]}')
    assert not drifted, f'{len(drifted)} value(s) changed\n  ' + '\n  '.join(drifted[:20])


def test_build_model_route(client, factory):
    """/build-model hands the canvas every AC element and every DC and microgrid element, by id."""
    spec = factory[0]
    response = client.post('/build-model', json={'spec': spec, 'run_power_flow': False})
    assert response.status_code == 200, response.get_data(as_text=True)[:400]
    model = json.loads(response.get_json()['model'])['_object']
    table = lambda key: json.loads(model[key]['_object'])['data']              # noqa: E731
    assert len(table('bus')) == len(spec['buses'])
    assert len(table('trafo')) == len(spec['transformers'])
    assert len(table('motor')) == len(spec['motors'])
    assert len(table('gen')) == len(spec['generators'])
    drawn = json.loads(model['electrisim_elements']['_object'])
    elements = {e['id'] for e in drawn['elements']}
    wanted = {r['id'] for key in layer.LISTS for r in spec.get(key, [])}
    assert wanted <= elements, sorted(wanted - elements)
    assert set(drawn['load_profiles']) == {p['id'] for p in spec['load_profiles']}
