# -*- coding: utf-8 -*-
"""
The AI campus reference: tests/reference/reference_ai_campus.spec.json, a 65 MW
AI training campus on a 35 kV utility supply (the design note "AI campus
reference microgrid"). Its AC network - two feeders in parallel through a
closed bus tie, two gas turbines on Yd11 step-up units, zigzag grounding
transformers on their breakers, a 400 A resistance-grounded 13.8 kV campus on
each bus - carries a 20 MW grid-forming BESS, 15 MW of PV and 10 MW of SOFC
through Yd11 PCS transformers, and two 24 MW halls on 800 V DC: Hall 1 in
detail (two SSTs and two transformer-rectifiers, four row groups with DC
batteries in droop, clusters with supercapacitors, a 54 V rack), Hall 2 as
an equivalent.

Its DC and microgrid layer is Electrisim's own, so the oracle is in two parts:

1. the conversion chain in closed form - each SST, rectifier, DC/DC converter
   and PCS drawing what its efficiency and no-load loss say, each cable its
   voltage drop, the utility the balance;
2. the AC network built by hand with pandapower, every default written out,
   each converter as the load or source its AC side is: the same voltages and
   flows, and with the PCS as their current limits the same fault currents.

Goldens pin the rest. Regenerate with:  pytest --regen-golden
"""
import contextlib
import copy
import io
import json
import math
import os

import numpy as np
import pandapower as pp
import pandapower.shortcircuit as sc
import pytest

import der_electrisim
import electrisim_sld as sld
import electrisim_spec_layer as layer
import pandapower_electrisim as pe

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = 'reference_ai_campus'
SPEC_PATH = os.path.join(HERE, 'reference', f'{GRID}.spec.json')
GOLDEN_PATH = os.path.join(HERE, 'golden', f'{GRID}.json')

LF = {'typ': 'PowerFlowPandaPower Parameters'}


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
def campus():
    spec = load_spec()
    with _silent():
        net, report = sld.build_network(spec)
    return spec, net, report


def _layered(net, params=LF):
    return layer.with_layer(net, net['electrisim_layer'], net.get('electrisim_bus_index') or {}, params)


@pytest.fixture(scope='module')
def solved(campus):
    """The campus load flow, as Electrisim runs it: the layer settled, voltage angles 'auto'."""
    _, net, _ = campus
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

def test_builds_cleanly_and_is_the_campus_of_the_note(campus):
    """No warnings; the load groups, sources and grounding the design note sets out."""
    spec, net, report = campus
    assert report['warnings'] == [], report['warnings']
    rows = _rows_by_id(spec)
    it_dc = sum(r['p_mw'] for r in spec['dc_loads'])
    assert it_dc == pytest.approx(48.0)                                   # two 24 MW halls on 800 V DC
    assert sum(r['p_mw'] for r in spec['loads'] if r['id'].startswith('ACIT')) == pytest.approx(2.0)
    campus_mw = (sum(m['pn_mech_mw'] for m in spec['motors'])
                 + sum(r['p_mw'] for r in spec['loads'] if not r['id'].startswith('ACIT')))
    assert campus_mw == pytest.approx(17.0)                              # 12 cooling, 2 auxiliaries, 3 buildings
    assert sum(g['sn_mva'] * g['cos_phi'] for g in spec['generators']) == pytest.approx(50.0)
    assert sum(b['capacity_kwh'] for b in spec['batteries'] if 'bus' not in b) == pytest.approx(80000)
    assert {p['control'] for p in spec['pcs'] if p['source'].startswith('BESS')} == {'grid_forming'}
    # Every 35 kV winding but the zigzags' is ungrounded: they are the campus's only ground source islanded.
    for t in spec['transformers']:
        if rows[t['hv_bus']]['vn_kv'] == 35:
            assert t['vector_group'] in ('Yd', 'Dyn', 'Dd', 'Dy'), t['id']
    assert {s['element'] for s in spec['switches'] if s['et'] == 'grounding_transformer'} == {'ZA', 'ZB'}
    # The 13.8 kV campus networks grounded through 400 A.
    for t in ('T_CA', 'T_CB'):
        assert 13.8e3 / math.sqrt(3) / rows[t]['rn_ohm'] == pytest.approx(400, rel=1e-4)
    # Both MV-to-DC conversions, the rectifier transformers 30 degrees apart.
    assert {s['id'] for s in spec['ssts']} == {'U1', 'U2', 'U5'}
    assert (rows['T_U3']['vector_group'], rows['T_U4']['vector_group']) == ('Dd', 'Dy')
    assert rows['T_U4']['shift_degree'] - rows['T_U3']['shift_degree'] == 330


# --- 1. the conversion chain in closed form ------------------------------------------

def _dcdc_input(p_out, eta, p_nl):
    return p_out / eta + p_nl if p_out >= 0 else p_out * eta + p_nl


def _cable_send(v0_kv, r_ohm, p_mw):
    """A DC cable delivering p_mw at its far end from v0_kv: the power sent, its far end's voltage."""
    v_end = (v0_kv + math.sqrt(v0_kv * v0_kv - 4.0 * r_ohm * p_mw)) / 2.0
    return v0_kv * p_mw / v_end, v_end


def test_conversion_chain_in_closed_form(campus, solved):
    """
    Each converter draws what its efficiency and no-load loss say for what it
    delivers; each cluster busway drops I R; the utility supplies the balance.
    """
    spec, net, _ = campus
    rows = _rows_by_id(spec)
    res = _results(solved)
    dcdc = res['dc_dc_converters']

    # Every DC/DC converter: its input its output over its efficiency, plus its no-load loss.
    for ident, conv in dcdc.items():
        row = rows[ident]
        eta = _default('DC/DC Converter', 'efficiency_percent', row) / 100
        p_nl = _default('DC/DC Converter', 'no_load_loss_kw', row) / 1e3
        assert conv['p_in_mw'] == pytest.approx(_dcdc_input(conv['p_out_mw'], eta, p_nl), abs=1e-9), ident
    # Base case: the stores idle - the batteries' droop gives nothing at 1.0 pu, smoothing nothing in a load flow.
    for ident, conv in dcdc.items():
        if ident != 'DD_R10':
            assert conv['p_out_mw'] == pytest.approx(0.0, abs=1e-9), ident
    assert dcdc['DD_R10']['p_out_mw'] == pytest.approx(0.15, abs=1e-9)

    # RG1: four clusters 30 m away; cluster 1 its nine racks and rack 10's shelf.
    v0 = 0.8
    dc_bus = {i: b for i, b in res['dc_buses'].items()}
    sent = 0.0
    for c in range(1, 5):
        p = 1.5 if c > 1 else 9 * 0.15 + dcdc['DD_R10']['p_in_mw']
        line = rows[f'L_CL{c}']
        p_send, v_end = _cable_send(v0, line['length_km'] * line['r_ohm_per_km'], p)
        assert dc_bus[f'CL{c}']['vm_pu'] == pytest.approx(v_end / v0, abs=1e-9), c
        sent += p_send
    assert dc_bus['R10_54']['vm_pu'] == pytest.approx(1.0, abs=1e-9)

    # Each SST: its DC/DC stage, then its rectifier.
    def sst_mv(ident, p_dc):
        row = rows[ident]
        link = _dcdc_input(p_dc, row['dcdc_efficiency_percent'] / 100, row['dcdc_no_load_kw'] / 1e3)
        return _dcdc_input(link, row['rect_efficiency_percent'] / 100, row['rect_no_load_kw'] / 1e3)

    ssts = res['ssts']
    assert ssts['U1']['p_mv_mw'] == pytest.approx(sst_mv('U1', sent), abs=1e-8)
    assert ssts['U2']['p_mv_mw'] == pytest.approx(sst_mv('U2', 6.0), abs=1e-8)
    # Hall 2: the rectifier delivers its 12 MW, the SST the rest of the hall's 24.
    assert ssts['U5']['p_mv_mw'] == pytest.approx(sst_mv('U5', 24.0 - 12.0), abs=1e-8)

    # Each rectifier: its DC power plus I^2 R on each side.
    ac_v = {ident: float(solved.res_bus.at[idx, 'vm_pu']) for ident, idx in _ac_ids(net)['bus'].items()}
    for ident, p_dc in (('U3', 6.0), ('U4', 6.0), ('U6', 12.0)):
        row, got = rows[ident], res['vscs'][ident]
        assert -got['p_dc_mw'] == pytest.approx(p_dc, abs=1e-9)
        i_dc = p_dc / v0
        p_int = p_dc + i_dc * i_dc * row['r_dc_ohm']
        v_ac = ac_v[row['bus']] * rows[row['bus']]['vn_kv']
        i_ac = got['p_mw'] / (math.sqrt(3) * v_ac)                       # unity power factor
        assert got['p_mw'] == pytest.approx(p_int + 3 * i_ac * i_ac * row['r_ohm'], abs=2e-4), ident
        assert got['q_mvar'] == pytest.approx(0.0, abs=1e-6)

    # Each PCS: its source's power through its efficiency. SOFC at its set point, PV at its maximum power point.
    pcs, ders = res['pcs'], res['sources_and_stores']
    by_row = {r['name']: r for r in net['electrisim_layer']['rows']}
    for n in (1, 2):
        assert pcs[f'PCS_FC{n}']['p_mw'] == pytest.approx(4.5 * 0.98, abs=1e-9)
    for n in (1, 2, 3):
        p_mpp = der_electrisim.build(by_row[f'PV{n}']).mpp()[2] / 1e6
        assert 0.97 * 6.2535 * 0.8 * 0.85 < p_mpp < 6.2535 * 0.8                 # 800 W/m2, cells about 44 degC
        assert ders[f'PV{n}']['p_mw'] == pytest.approx(p_mpp, abs=1e-9)
        assert pcs[f'PCS_PV{n}']['p_mw'] == pytest.approx(p_mpp * 0.98, abs=1e-9)
    for n in range(1, 5):
        assert pcs[f'PCS_BESS{n}']['p_mw'] == pytest.approx(0.0, abs=1e-9)

    # The utility supplies the balance.
    ids = _ac_ids(net)
    gen = sum(float(solved.res_gen.at[i, 'p_mw']) for i in ids['gen'].values())
    loads = (sum(float(solved.res_load.at[i, 'p_mw']) for i in ids['load'].values())
             + sum(float(solved.res_motor.at[i, 'p_mw']) for i in ids['motor'].values()))
    conv = sum(s['p_mv_mw'] for s in ssts.values()) + sum(v['p_mw'] for v in res['vscs'].values())
    losses = (sum(float(solved.res_line.at[i, 'pl_mw']) for i in ids['line'].values())
              + float(solved.res_trafo['pl_mw'].sum()))
    pcs_mw = sum(p['p_mw'] for p in pcs.values())
    grid = float(solved.res_ext_grid.at[ids['ext_grid']['Utility'], 'p_mw'])
    assert grid == pytest.approx(loads + conv + losses - gen - pcs_mw, abs=1e-6)
    assert 20 < grid < 26                                                 # the base case's import, about 21 MW


# --- 2. the AC network by hand ---------------------------------------------------------

def hand_built_ac(spec, equivalents=None, current_sources=None):
    """
    reference_ai_campus's AC network with the documented defaults written out.
    equivalents: {AC bus id: (p_mw, q_mvar)} loads standing for what the layer
    draws there (negative: delivers). current_sources: [(AC bus id, sn_mva, k)]
    for the PCS in a short circuit. The zigzags as the grounding transformer's
    documented model: a YNd unit, Z0 + 3 Z_N its zero sequence, its delta
    unloaded, positive sequence 7.5 % on 1 MVA so IEC's K_T is 1.
    """
    net = pp.create_empty_network(f_hz=50.0)
    b = {r['id']: pp.create_bus(net, vn_kv=r['vn_kv'], name=r['id']) for r in spec['buses']}
    ids = {'bus': dict(b), 'line': {}, 'trafo': {}}
    pp.create_ext_grid(net, b['UTIL'], vm_pu=1.0, va_degree=0.0, s_sc_max_mva=600, rx_max=0.125, s_sc_min_mva=400,
                       rx_min=0.125, x0x_max=1.0, r0x0_max=0.1, x0x_min=1.0, r0x0_min=0.1)
    for r in spec['lines']:
        ids['line'][r['id']] = pp.create_line_from_parameters(
            net, b[r['from_bus']], b[r['to_bus']], r['length_km'], r_ohm_per_km=r['r_ohm_per_km'],
            x_ohm_per_km=r['x_ohm_per_km'], c_nf_per_km=r['c_nf_per_km'], max_i_ka=r['max_i_ka'],
            r0_ohm_per_km=r['r0_ohm_per_km'], x0_ohm_per_km=r['x0_ohm_per_km'], c0_nf_per_km=r['c0_nf_per_km'],
            endtemp_degree=80.0)
    for r in spec['transformers']:
        vn_hv = next(x['vn_kv'] for x in spec['buses'] if x['id'] == r['hv_bus'])
        vn_lv = next(x['vn_kv'] for x in spec['buses'] if x['id'] == r['lv_bus'])
        i = pp.create_transformer_from_parameters(
            net, b[r['hv_bus']], b[r['lv_bus']], sn_mva=r['sn_mva'], vn_hv_kv=vn_hv, vn_lv_kv=vn_lv,
            vk_percent=r['vk_percent'], vkr_percent=r['vkr_percent'], pfe_kw=0.6 * r['sn_mva'], i0_percent=0.1,
            shift_degree=r['shift_degree'], vector_group=r['vector_group'], vk0_percent=r['vk_percent'],
            vkr0_percent=r['vkr_percent'], mag0_percent=100.0, mag0_rx=0.0, si0_hv_partial=0.9)
        net.trafo.loc[i, ['rn_ohm', 'xn_ohm']] = (r.get('rn_ohm', 0.0), 0.0)
        if 'tap_pos' in r:                       # an off-load tap: HV side, +/- 2 steps about neutral
            net.trafo.loc[i, ['tap_side', 'tap_neutral', 'tap_min', 'tap_max', 'tap_step_percent', 'tap_pos',
                              'tap_step_degree', 'tap_changer_type']] = (
                'hv', 0, -2, 2, r['tap_step_percent'], r['tap_pos'], 0.0, 'Ratio')
        ids['trafo'][r['id']] = i
    for r in spec['loads']:
        pp.create_load(net, b[r['bus']], p_mw=r['p_mw'], q_mvar=r['q_mvar'])
    for r in spec['motors']:
        pp.create_motor(net, b[r['bus']], pn_mech_mw=r['pn_mech_mw'], cos_phi=r['cos_phi'],
                        efficiency_percent=r['efficiency_percent'], lrc_pu=r['lrc_pu'], rx=r['rx'], vn_kv=13.8,
                        cos_phi_n=r['cos_phi'], efficiency_n_percent=r['efficiency_percent'])
    for r in spec['generators']:
        pp.create_gen(net, b[r['bus']], p_mw=r['p_mw'], vm_pu=1.0, sn_mva=r['sn_mva'], vn_kv=13.8,
                      xdss_pu=r['xdss_pu'], rdss_ohm=r['rdss_ohm'], cos_phi=r['cos_phi'])
    for r in spec['switches']:
        if r['et'] == 'line':
            pp.create_switch(net, b[r['bus']], ids['line'][r['element']], et='l', closed=True)
        elif r['et'] == 'transformer':
            pp.create_switch(net, b[r['bus']], ids['trafo'][r['element']], et='t', closed=True)
        elif r['et'] == 'bus':
            pp.create_switch(net, b[r['bus']], b[r['element']], et='b', closed=True)
    v_ph, i_rated = 35e3 / math.sqrt(3), 400.0
    x0 = 0.12 * v_ph / i_rated
    r_tot, x_tot = 0.1 * x0 + 3 * 50.0, x0
    for z in spec['grounding_transformers']:
        delta = pp.create_bus(net, vn_kv=35, name=f"{z['id']} delta")
        pp.create_transformer_from_parameters(
            net, b[z['bus']], delta, sn_mva=1.0, vn_hv_kv=35, vn_lv_kv=35, vk_percent=7.5, vkr_percent=0.0,
            pfe_kw=0.0, i0_percent=0.0, vk0_percent=100 * math.hypot(r_tot, x_tot) / 35 ** 2,
            vkr0_percent=100 * r_tot / 35 ** 2, mag0_percent=1e6, mag0_rx=0.0, si0_hv_partial=0.9,
            vector_group='YNd', shift_degree=0.0)
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


def test_ac_network_matches_hand_built(campus, solved):
    """The AC network the spec builds, with the layer's draws at its buses, is the one written out by hand."""
    spec, net, _ = campus
    oracle, oids = hand_built_ac(spec, _equivalents(spec, _results(solved)))
    pp.runpp(oracle, calculate_voltage_angles='auto', init='auto')
    ids = _ac_ids(net)
    differ = []
    for ident, idx in ids['bus'].items():
        for col, tol in (('vm_pu', 1e-8), ('va_degree', 1e-6)):
            got, want = float(solved.res_bus.at[idx, col]), float(oracle.res_bus.at[oids['bus'][ident], col])
            if abs(got - want) > tol:
                differ.append(f'bus {ident} {col}: {want} by hand, {got} built')
    for table, cols in (('line', ('p_from_mw', 'q_from_mvar', 'loading_percent')),
                        ('trafo', ('p_hv_mw', 'q_hv_mvar', 'loading_percent'))):
        for ident, idx in ids[table].items():
            for col in cols:
                got = float(solved[f'res_{table}'].at[idx, col])
                want = float(oracle[f'res_{table}'].at[oids[table][ident], col])
                if abs(got - want) > 1e-6:
                    differ.append(f'{table} {ident} {col}: {want} by hand, {got} built')
    for col in ('p_mw', 'q_mvar'):
        if abs(float(solved.res_ext_grid.at[ids['ext_grid']['Utility'], col]) - float(oracle.res_ext_grid.at[0, col])) > 1e-6:
            differ.append(f'utility {col}')
    assert not differ, '\n  '.join(differ[:20])


def _study_sc(net, fault, case):
    """Electrisim's IEC short circuit on the campus with its layer, {bus id: row}."""
    full = _layered(net, sc_params(fault, case))
    with _silent():
        out = json.loads(pe.shortcircuit(full, sc_params(fault, case), None))
    assert not out.get('error'), out.get('message')
    ids = {v: k for k, v in net['electrisim_ids']['bus'].items()}
    names = {str(full.bus.at[i, 'name']): ident for i, ident in net['electrisim_ids']['bus'].items()}
    return {names[r['name']]: r for r in out['busbars'] if r['name'] in names}


def _pcs_current_sources(spec):
    rows = _rows_by_id(spec)
    return [(p['bus'], p['s_rated_mva'], float(p.get('current_limit_pu', 1.2))) for p in spec['pcs']
            if rows[p['source']]]


def _fold(net, lv_tol_percent=6):
    """The documented IEC rule: 3 Z_N joins its grounded winding's zero sequence, uncorrected by K_T."""
    for i in net.trafo.index:
        rn = float(net.trafo.at[i, 'rn_ohm'])
        if not rn > 0:                          # none, or a zigzag (its Z_N in its own vk0)
            continue
        row = net.trafo.loc[i]
        vk, vkr, sn, vn = row.vk_percent, row.vkr_percent, row.sn_mva, row.vn_lv_kv      # Dyn: on the LV
        k_t = 0.95 * 1.1 / (1 + 0.6 * math.sqrt(vk ** 2 - vkr ** 2) / 100)
        z0 = complex(row.vkr0_percent, math.sqrt(row.vk0_percent ** 2 - row.vkr0_percent ** 2))
        z0 += 100 * 3 * rn * sn / vn ** 2 / k_t
        net.trafo.loc[i, ['vk0_percent', 'vkr0_percent']] = (abs(z0), z0.real)


@pytest.mark.parametrize('fault, case', [('3ph', 'max'), ('3ph', 'min'), ('1ph', 'max'), ('1ph', 'min')])
def test_short_circuit_matches_hand_built(campus, fault, case):
    """
    Every AC bus's IEC 60909 fault: the network by hand, the turbines and
    chillers as machines, each PCS a current source at its limit (1.2 x its
    rating) - the SSTs and rectifiers add nothing - and, earth faults, the
    13.8 kV neutral resistors folded in.
    """
    spec, net, _ = campus
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


def test_ground_faults_by_design(campus):
    """
    A 35 kV earth fault: on the grid, mostly the utility's; the zigzags' path
    alone 3 V_ph / |Z0 + 3 R_N| each (about 400 A). A 13.8 kV earth fault is
    held by its 400 A resistor: c x 400 A through the neutral, the PCS's
    positive-sequence current on top.
    """
    spec, net, _ = campus
    one = _study_sc(net, '1ph', 'max')
    three = _study_sc(net, '3ph', 'max')
    assert 10 < one['BUS_A']['ikss_ka'] < three['BUS_A']['ikss_ka']
    assert one['CA_13']['ikss_ka'] > 1.1 * 0.4                          # c x 400 A plus the converters'
    assert one['CA_13']['ikss_ka'] < 1.1 * 0.4 + sum(p['s_rated_mva'] for p in spec['pcs']) * 1.2 / (
        math.sqrt(3) * 13.8)


# --- goldens and the canvas ---------------------------------------------------------------

def _summary(net, solved):
    res = _results(solved)
    ids = _ac_ids(net)
    out = {
        'bus': {i: {'vm_pu': float(solved.res_bus.at[x, 'vm_pu']), 'va_degree': float(solved.res_bus.at[x, 'va_degree'])}
                for i, x in ids['bus'].items()},
        'line': {i: {'p_from_mw': float(solved.res_line.at[x, 'p_from_mw']),
                     'q_from_mvar': float(solved.res_line.at[x, 'q_from_mvar'])} for i, x in ids['line'].items()},
        'trafo': {i: {'p_hv_mw': float(solved.res_trafo.at[x, 'p_hv_mw']),
                      'loading_percent': float(solved.res_trafo.at[x, 'loading_percent'])} for i, x in ids['trafo'].items()},
        'gen': {i: {'q_mvar': float(solved.res_gen.at[x, 'q_mvar'])} for i, x in ids['gen'].items()},
        'ext_grid': {i: {'p_mw': float(solved.res_ext_grid.at[x, 'p_mw']),
                         'q_mvar': float(solved.res_ext_grid.at[x, 'q_mvar'])} for i, x in ids['ext_grid'].items()},
        'dc_bus': {i: {'vm_pu': b['vm_pu']} for i, b in res['dc_buses'].items()},
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


def test_matches_golden(regen, campus, solved):
    actual = _summary(campus[1], solved)
    if regen:
        with open(GOLDEN_PATH, 'w', encoding='utf-8') as handle:
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


def test_build_model_route(client, campus):
    """/build-model hands the canvas every AC element and every DC and microgrid element, by id."""
    spec = campus[0]
    response = client.post('/build-model', json={'spec': spec, 'run_power_flow': False})
    assert response.status_code == 200, response.get_data(as_text=True)[:400]
    model = json.loads(response.get_json()['model'])['_object']
    table = lambda key: json.loads(model[key]['_object'])['data']
    assert len(table('bus')) == len(spec['buses'])
    assert len(table('trafo')) == len(spec['transformers'])
    assert len(table('motor')) == len(spec['motors'])
    drawn = json.loads(model['electrisim_elements']['_object'])
    elements = {e['id'] for e in drawn['elements']}
    wanted = {r['id'] for key in layer.LISTS for r in spec.get(key, [])}
    wanted |= {s['id'] for s in spec['switches'] if s['et'] == 'grounding_transformer'}
    assert wanted <= elements, sorted(wanted - elements)
    assert set(drawn['load_profiles']) == {p['id'] for p in spec['load_profiles']}


# --- the drawn diagram ----------------------------------------------------------------------
# Each tests/reference/reference_ai_campus.diagram_*payload.json is the request the
# browser sent after drawing the spec - /build-model's model through the canvas
# import, as the MCP server's draw_diagram does - and running the study from its
# dialog. Recapture after changing the import or a payload builder (see 16b in
# the design note): the tests say whether the drawing still computes the spec's
# answers.

def _drawn(name):
    with open(os.path.join(HERE, 'reference', f'{GRID}.diagram_{name}payload.json'), encoding='utf-8') as handle:
        return json.load(handle)


def _post_drawn(client, payload):
    with _silent():
        response = client.post('/', json=payload)
    assert response.status_code == 200, response.get_data(as_text=True)[:400]
    out = json.loads(response.get_data(as_text=True))
    assert not out.get('error'), out.get('message')
    return out


def _by_spec_id(spec, payload, rows):
    """A study's result rows under their spec ids: cell name -> its label -> the spec element so named."""
    label = {v['name']: v.get('userFriendlyName') for v in payload.values() if isinstance(v, dict) and 'name' in v}
    ident = {r.get('name', r['id']): r['id'] for key, lst in spec.items() if isinstance(lst, list)
             for r in lst if isinstance(r, dict) and 'id' in r}
    return {ident[label[r['name']]]: r for r in rows if label.get(r['name']) in ident}


def test_drawn_diagram_load_flow_matches_spec(client, campus, solved):
    """The drawn campus's load flow: every AC and DC bus, SST, rectifier, PCS and DC/DC converter as the spec's."""
    spec, net, _ = campus
    payload = _drawn('')
    out = _post_drawn(client, payload)
    assert not out.get('warnings'), out.get('warnings')
    res = _results(solved)
    ids = _ac_ids(net)
    differ = []

    def check(what, got, want, tol=1e-6):
        if got is None or abs(float(got) - float(want)) > tol:
            differ.append(f'{what}: spec {want}, drawn {got}')

    buses = _by_spec_id(spec, payload, out['busbars'])
    assert set(buses) == set(ids['bus'])
    for ident, idx in ids['bus'].items():
        check(f'bus {ident} vm_pu', buses[ident]['vm_pu'], solved.res_bus.at[idx, 'vm_pu'], 1e-8)
        check(f'bus {ident} va_degree', buses[ident]['va_degree'], solved.res_bus.at[idx, 'va_degree'])
    dc = _by_spec_id(spec, payload, out['dcbuses'])
    assert set(dc) == set(res['dc_buses'])
    for ident, row in dc.items():
        check(f'DC bus {ident}', row['vm_pu'], res['dc_buses'][ident]['vm_pu'], 1e-8)
    for key, spec_key, col in (('ssts', 'ssts', 'p_mv_mw'), ('vscs', 'vscs', 'p_mw'), ('pcs', 'pcs', 'p_mw'),
                               ('pcs', 'pcs', 'q_mvar'), ('dcdcconverters', 'dc_dc_converters', 'p_in_mw')):
        rows = _by_spec_id(spec, payload, out[key])
        assert set(rows) == set(res[spec_key]), key
        for ident, row in rows.items():
            check(f'{key} {ident} {col}', row[col], res[spec_key][ident][col])
    grid = out['externalgrids'][0]
    check('utility p_mw', grid['p_mw'], solved.res_ext_grid.at[ids['ext_grid']['Utility'], 'p_mw'])
    assert not differ, f'{len(differ)} differ\n  ' + '\n  '.join(differ[:20])


@pytest.mark.parametrize('fault, case, name', [
    ('3ph', 'max', 'sc_'), ('3ph', 'min', 'sc_min_'), ('2ph', 'max', 'sc2ph_'), ('2ph', 'min', 'sc2ph_min_'),
    ('1ph', 'max', 'sc1ph_'), ('1ph', 'min', 'sc1ph_min_')])
def test_drawn_diagram_short_circuit_matches_spec(client, campus, fault, case, name):
    """The drawn campus's IEC fault at every AC bus, as the spec's with its layer."""
    spec, net, _ = campus
    payload = _drawn(name)
    assert (payload['0']['fault_type'], payload['0']['fault_location']) == (fault, case)
    drawn = _by_spec_id(spec, payload, _post_drawn(client, payload)['busbars'])
    want = _study_sc(net, fault, case)
    assert set(drawn) == set(want)
    differ = [f'{i} {c}: spec {want[i][c]}, drawn {drawn[i][c]}' for i in want for c in ('ikss_ka', 'ip_ka', 'ith_ka')
              if abs(float(drawn[i][c]) - float(want[i][c])) > 1e-6 * max(1.0, float(want[i][c]))]
    assert not differ, '\n  '.join(differ)
