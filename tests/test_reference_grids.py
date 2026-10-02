# -*- coding: utf-8 -*-
"""
Reference grids: two networks that between them use every part of the spec.

tests/reference/reference_transmission.spec.json has every element type - a
three-winding transformer, two-winding transformers with given and default
impedance, lines by default type, named type and explicit impedance, a
voltage-controlled generator, PV and wind, a reactor and a capacitor bank,
charging storage - a meshed 20 kV ring, and a switch of every kind, one of
them open. tests/reference/reference_radial.spec.json is the feeder single-line
the radial layout is for. Draw either through the MCP server to test the
diagram by eye.

Each grid is checked three ways:

1. `test_spec_matches_hand_built_network` builds the same network directly
   with pandapower, writing out every default the spec reference documents.
   It is an oracle independent of electrisim_sld, so it catches a default that
   has drifted from the documentation as well as a wrong answer.
2. `test_matches_golden` pins the results, so any change in the numbers shows.
3. `test_build_model_route` drives the real /build-model route and checks the
   model handed to the canvas carries every element, under its name.
4. `test_drawn_diagram_matches_spec` closes the loop through the canvas. Each
   tests/reference/<grid>.diagram_payload.json is the load-flow request the
   frontend sent for that grid after drawing it from the spec - so it is what
   the diagram holds, not what the spec says. Posted to the load-flow route, it
   must reproduce the spec's own results. The short-circuit tests do the same
   for a maximum three-phase fault at every bus, from
   tests/reference/<grid>.diagram_sc_payload.json.

Regenerate goldens with:  pytest --regen-golden

Recapture the diagram payloads after changing the frontend import or a payload
builder: draw the spec through the MCP server into an empty diagram, run Load
Flow, Short Circuit (three-phase, two-phase, single-phase, each in the maximum
and the minimum case), Harmonic Analysis and
Load Flow again on the OpenDSS engine tab, and save the body of each POST to
the backend's "/" from the browser's network panel over the old files
(<grid>.diagram_payload.json, .diagram_sc_payload.json,
.diagram_sc2ph_payload.json, .diagram_sc1ph_payload.json, the same with _min
before _payload, .diagram_harmonic_payload.json and
.diagram_opendss_payload.json). The tests say whether the
new drawing still computes the spec's answers.
"""

import json
import os

import numpy as np
import pandapower as pp
import pandapower.shortcircuit as sc
import pytest

import electrisim_sld as sld

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE_DIR = os.path.join(HERE, 'reference')
GOLDEN_DIR = os.path.join(HERE, 'golden')

GRIDS = ('reference_transmission', 'reference_radial')

# The two builds run the same solver on the same numbers, so they must agree to
# rounding. The goldens are "did the number change at all".
ORACLE_TOL = 1e-9
VM_TOL = 1e-8
VA_TOL = 1e-6
FLOW_TOL = 1e-6

# Every element type the spec accepts, and every kind of switch.
SPEC_TABLES = ('bus', 'ext_grid', 'trafo', 'trafo3w', 'line', 'load', 'gen', 'sgen',
               'shunt', 'storage', 'switch')
SWITCH_KINDS = {'l', 't', 't3', 'b'}


def load_spec(grid):
    with open(os.path.join(REFERENCE_DIR, f'{grid}.spec.json'), encoding='utf-8') as handle:
        return json.load(handle)


def run(net):
    pp.runpp(net, algorithm='nr', calculate_voltage_angles='auto', init='auto')
    return net


def run_sc(net, fault='3ph', case='max'):
    """A fault at every bus, called as the backend calls it."""
    # Branch results need the state a power flow leaves; the backend's
    # pp.diagnostic() runs one before it calls calc_sc.
    run(net)
    sc.calc_sc(net, fault=fault, case=case, ip=True, ith=True, tk_s=1.0, kappa_method='C',
               r_fault_ohm=0.0, x_fault_ohm=0.0, check_connectivity=False, branch_results=True)
    return net


# --- the oracle: the same grids, built by hand -------------------------------

class _Builder:
    """Records pandapower indices under the spec's ids as elements are made."""

    def __init__(self, f_hz):
        self.net = pp.create_empty_network(f_hz=f_hz)
        self.ids = {table: {} for table in SPEC_TABLES}

    def bus(self, ident, vn_kv):
        self.ids['bus'][ident] = pp.create_bus(self.net, vn_kv=vn_kv)

    def b(self, ident):
        return self.ids['bus'][ident]

    def add(self, table, ident, idx):
        self.ids[table][ident] = idx


def _zero_sequence(net, trafos, trafo3w=None, lines=None, endtemps=None):
    """
    Zero-sequence data, as the spec reference documents it. trafos maps index
    -> vector group; vk0/vkr0 are the positive-sequence values, mag0 100 %,
    mag0_rx 0, si0_hv_partial 0.9. Lines not given explicitly get R0 = 4 R1,
    X0 = 3 X1, C0 = C1, and an end-of-fault temperature of 80 degC.
    """
    for idx, group in trafos.items():
        net.trafo.loc[idx, ['vector_group']] = group
        net.trafo.loc[idx, ['vk0_percent', 'vkr0_percent']] = (
            net.trafo.at[idx, 'vk_percent'], net.trafo.at[idx, 'vkr_percent'])
        net.trafo.loc[idx, ['mag0_percent', 'mag0_rx', 'si0_hv_partial']] = (100.0, 0.0, 0.9)
    for idx, group in (trafo3w or {}).items():
        net.trafo3w.loc[idx, ['vector_group']] = group
        for w in ('hv', 'mv', 'lv'):
            net.trafo3w.loc[idx, [f'vk0_{w}_percent', f'vkr0_{w}_percent']] = (
                net.trafo3w.at[idx, f'vk_{w}_percent'], net.trafo3w.at[idx, f'vkr_{w}_percent'])
    explicit = lines or {}
    for idx in net.line.index:
        r0, x0, c0 = explicit.get(idx, (round(4 * net.line.at[idx, 'r_ohm_per_km'], 6),
                                        round(3 * net.line.at[idx, 'x_ohm_per_km'], 6),
                                        round(net.line.at[idx, 'c_nf_per_km'], 6)))
        net.line.loc[idx, ['r0_ohm_per_km', 'x0_ohm_per_km', 'c0_nf_per_km']] = (r0, x0, c0)
        net.line.loc[idx, ['endtemp_degree']] = (endtemps or {}).get(idx, 80.0)


def hand_built_transmission():
    """reference_transmission.spec.json, with the documented defaults written out."""
    h = _Builder(50.0)
    for ident, kv in (('HV', 110), ('HV2', 110), ('MV1', 20), ('MV2', 20), ('F1', 20),
                      ('F2', 20), ('F3', 20), ('TERT', 10), ('LV1', 0.4), ('LV2', 0.4)):
        h.bus(ident, kv)
    net, b = h.net, h.b

    # The minimum case defaults to the maximum; zero sequence given.
    h.add('ext_grid', 'Grid', pp.create_ext_grid(net, b('HV'), vm_pu=1.02, va_degree=0.0,
                                                 s_sc_max_mva=5000, rx_max=0.1,
                                                 s_sc_min_mva=5000, rx_min=0.1,
                                                 x0x_max=2.5, r0x0_max=0.2,
                                                 x0x_min=2.5, r0x0_min=0.2))
    # vk by the smaller rating of each pair: MV-LV and HV-LV pair with the
    # 15 MVA tertiary, which is in the "up to 40 MVA" class -> 12 %.
    # vkr = vk / 25, pfe = 0.6 kW per MVA of HV rating, i0 = 0.1 %.
    h.add('trafo3w', 'T3W', pp.create_transformer3w_from_parameters(
        net, b('HV'), b('MV1'), b('TERT'), vn_hv_kv=110, vn_mv_kv=20, vn_lv_kv=10,
        sn_hv_mva=40, sn_mv_mva=40, sn_lv_mva=15,
        vk_hv_percent=12.5, vk_mv_percent=12.0, vk_lv_percent=12.0,
        vkr_hv_percent=0.5, vkr_mv_percent=0.48, vkr_lv_percent=0.48,
        pfe_kw=24.0, i0_percent=0.1, shift_mv_degree=0.0))
    h.add('trafo', 'T2', pp.create_transformer_from_parameters(
        net, b('HV2'), b('MV2'), sn_mva=25, vn_hv_kv=110, vn_lv_kv=20,
        vk_percent=12, vkr_percent=0.41, pfe_kw=15.0, i0_percent=0.1, shift_degree=0.0))
    # 0.63 MVA is in the "up to 2.5 MVA" class -> 4 %.
    h.add('trafo', 'T_LV1', pp.create_transformer_from_parameters(
        net, b('F1'), b('LV1'), sn_mva=0.63, vn_hv_kv=20, vn_lv_kv=0.4,
        vk_percent=4.0, vkr_percent=0.16, pfe_kw=0.378, i0_percent=0.1, shift_degree=0.0))
    h.add('trafo', 'T_LV2', pp.create_transformer_from_parameters(
        net, b('F2'), b('LV2'), sn_mva=1.0, vn_hv_kv=20, vn_lv_kv=0.4,
        vk_percent=6, vkr_percent=0.24, pfe_kw=0.6, i0_percent=0.1, shift_degree=0.0))

    # Default line types by voltage: 110 kV overhead, 20 kV cable.
    h.add('line', 'L_HV', pp.create_line(net, b('HV'), b('HV2'), 12,
                                         std_type='149-AL1/24-ST1A 110.0'))
    h.add('line', 'L1', pp.create_line(net, b('MV1'), b('F1'), 3,
                                       std_type='NA2XS2Y 1x240 RM/25 12/20 kV'))
    h.add('line', 'L2', pp.create_line_from_parameters(
        net, b('MV2'), b('F2'), 2.5, r_ohm_per_km=0.125, x_ohm_per_km=0.112,
        c_nf_per_km=300, max_i_ka=0.42))
    h.add('line', 'L3', pp.create_line(net, b('F1'), b('F3'), 1.8,
                                       std_type='NA2XS2Y 1x185 RM/25 12/20 kV'))
    h.add('line', 'L4', pp.create_line(net, b('F2'), b('F3'), 2.2,
                                       std_type='NA2XS2Y 1x240 RM/25 12/20 kV'))

    for ident, bus, p, q in (('LD_MV1', 'MV1', 6.0, 2.0), ('LD_F3', 'F3', 3.0, 1.0),
                             ('LD_LV1', 'LV1', 0.4, 0.4 * 0.33),  # unstated q: 0.33 x p
                             ('LD_LV2', 'LV2', 0.6, 0.15), ('LD_AUX', 'TERT', 0.5, 0.2)):
        h.add('load', ident, pp.create_load(net, b(bus), p_mw=p, q_mvar=q))
    h.add('gen', 'G1', pp.create_gen(net, b('MV2'), p_mw=4.0, vm_pu=1.01, sn_mva=6.0,
                                     vn_kv=20, xdss_pu=0.18, rdss_ohm=0.02, cos_phi=0.8))
    # Unstated sgen rating: 1.1 x p_mw, at least 0.1 MVA; k defaults to 1.1.
    h.add('sgen', 'PV', pp.create_sgen(net, b('LV2'), p_mw=0.3, q_mvar=0.0, sn_mva=0.33, k=1.1))
    h.add('sgen', 'WF', pp.create_sgen(net, b('F3'), p_mw=2.0, q_mvar=-0.2, sn_mva=2.5, k=1.2))
    h.add('shunt', 'SR', pp.create_shunt(net, b('TERT'), q_mvar=2.0, p_mw=0.0))
    h.add('shunt', 'CAP', pp.create_shunt(net, b('F1'), q_mvar=-1.5, p_mw=0.0))
    h.add('storage', 'BESS', pp.create_storage(net, b('LV1'), p_mw=0.1, max_e_mwh=0.5))

    ids = h.ids
    # T2 is given as YNd5; the rest take Dyn, the three-winding unit YNynd.
    _zero_sequence(net, {ids['trafo']['T2']: 'YNd', ids['trafo']['T_LV1']: 'Dyn',
                         ids['trafo']['T_LV2']: 'Dyn'},
                   trafo3w={ids['trafo3w']['T3W']: 'YNynd'},
                   lines={ids['line']['L_HV']: (0.35, 1.25, 5.0)},
                   endtemps={ids['line']['L_HV']: 70.0})
    for ident, bus, element, et, closed in (
            ('CB_LHV', 'HV', ids['line']['L_HV'], 'l', True),
            ('CB_T3W', 'HV', ids['trafo3w']['T3W'], 't3', True),
            ('CB_T2', 'HV2', ids['trafo']['T2'], 't', True),
            ('CB_L1', 'MV1', ids['line']['L1'], 'l', True),
            ('CB_L4', 'F3', ids['line']['L4'], 'l', True),
            ('TIE', 'MV1', b('MV2'), 'b', False)):
        h.add('switch', ident, pp.create_switch(net, b(bus), element, et=et, closed=closed))
    return h


def hand_built_radial():
    """reference_radial.spec.json, with the documented defaults written out."""
    h = _Builder(50.0)
    for ident, kv in (('GRID', 110), ('SUB', 20), ('A1', 20), ('A2', 20), ('LVA', 0.4),
                      ('B1', 20), ('B2', 20), ('C1', 20)):
        h.bus(ident, kv)
    net, b = h.net, h.b

    # Zero sequence by default: X0/X 1.0, R0/X0 0.1.
    h.add('ext_grid', 'Grid', pp.create_ext_grid(net, b('GRID'), vm_pu=1.0, va_degree=0.0,
                                                 s_sc_max_mva=3000, rx_max=0.1,
                                                 s_sc_min_mva=3000, rx_min=0.1,
                                                 x0x_max=1.0, r0x0_max=0.1,
                                                 x0x_min=1.0, r0x0_min=0.1))
    # 40 MVA -> 12 %, 0.8 MVA -> 4 %.
    h.add('trafo', 'T1', pp.create_transformer_from_parameters(
        net, b('GRID'), b('SUB'), sn_mva=40, vn_hv_kv=110, vn_lv_kv=20,
        vk_percent=12.0, vkr_percent=0.48, pfe_kw=24.0, i0_percent=0.1, shift_degree=0.0))
    h.add('trafo', 'TA', pp.create_transformer_from_parameters(
        net, b('A2'), b('LVA'), sn_mva=0.8, vn_hv_kv=20, vn_lv_kv=0.4,
        vk_percent=4.0, vkr_percent=0.16, pfe_kw=0.48, i0_percent=0.1, shift_degree=0.0))

    cable = 'NA2XS2Y 1x240 RM/25 12/20 kV'
    for ident, a, z, km in (('LA1', 'SUB', 'A1', 2.0), ('LA2', 'A1', 'A2', 1.5),
                            ('LB1', 'SUB', 'B1', 3.0), ('LB2', 'B1', 'B2', 2.0)):
        h.add('line', ident, pp.create_line(net, b(a), b(z), km, std_type=cable))
    h.add('line', 'LC1', pp.create_line_from_parameters(
        net, b('SUB'), b('C1'), 4.0, r_ohm_per_km=0.08, x_ohm_per_km=0.1,
        c_nf_per_km=350, max_i_ka=0.5))

    h.add('load', 'LD_A1', pp.create_load(net, b('A1'), p_mw=1.5, q_mvar=1.5 * 0.33))
    h.add('load', 'LD_LVA', pp.create_load(net, b('LVA'), p_mw=0.5, q_mvar=0.12))
    h.add('load', 'LD_B1', pp.create_load(net, b('B1'), p_mw=2.0, q_mvar=0.8))
    # sn_mva defaults to 1.2 x p_mw, at least 1. Short-circuit data defaults:
    # the bus voltage, xdss 0.2 pu, rdss 0 ohm, cos phi 0.85.
    h.add('gen', 'GE', pp.create_gen(net, b('B2'), p_mw=1.5, vm_pu=1.0, sn_mva=1.8,
                                     vn_kv=20, xdss_pu=0.2, rdss_ohm=0.0, cos_phi=0.85))
    h.add('sgen', 'WF', pp.create_sgen(net, b('C1'), p_mw=3.0, q_mvar=0.0, sn_mva=3.3, k=1.3))
    h.add('sgen', 'PV', pp.create_sgen(net, b('LVA'), p_mw=0.1, q_mvar=0.0, sn_mva=0.11, k=1.1))
    h.add('shunt', 'CAP', pp.create_shunt(net, b('B1'), q_mvar=-0.6, p_mw=0.0))
    h.add('storage', 'BESS', pp.create_storage(net, b('C1'), p_mw=-0.5, max_e_mwh=2.0))

    ids = h.ids
    _zero_sequence(net, {ids['trafo']['T1']: 'Dyn', ids['trafo']['TA']: 'Dyn'})
    for ident, element, et in (('CB_T1', ids['trafo']['T1'], 't'),
                               ('CB_A', ids['line']['LA1'], 'l'),
                               ('CB_B', ids['line']['LB1'], 'l'),
                               ('CB_C', ids['line']['LC1'], 'l')):
        h.add('switch', ident, pp.create_switch(net, b('SUB'), element, et=et, closed=True))
    return h


HAND_BUILT = {
    'reference_transmission': hand_built_transmission,
    'reference_radial': hand_built_radial,
}

# Result columns compared per table.
RESULT_COLUMNS = {
    'bus': ('vm_pu', 'va_degree', 'p_mw', 'q_mvar'),
    'line': ('p_from_mw', 'q_from_mvar', 'p_to_mw', 'q_to_mvar', 'loading_percent'),
    'trafo': ('p_hv_mw', 'q_hv_mvar', 'p_lv_mw', 'q_lv_mvar', 'loading_percent'),
    'trafo3w': ('p_hv_mw', 'q_hv_mvar', 'p_mv_mw', 'p_lv_mw', 'loading_percent'),
    'ext_grid': ('p_mw', 'q_mvar'),
    'gen': ('p_mw', 'q_mvar', 'vm_pu'),
}


def results_by_id(net, ids):
    """{table: {spec id: {column: value}}} from a solved net and an id -> index map."""
    out = {}
    for table, columns in RESULT_COLUMNS.items():
        res = net[f'res_{table}']
        out[table] = {ident: {c: float(res.at[idx, c]) for c in columns}
                      for ident, idx in ids[table].items()}
    return out


SC_COLUMNS = ('ikss_ka', 'ip_ka', 'ith_ka')
# A single-phase fault has no peak or thermal current in pandapower; its
# zero-sequence impedance at the bus is what tests the zero-sequence data.
SC_1PH_COLUMNS = ('ikss_ka', 'rk0_ohm', 'xk0_ohm')


def sc_by_id(net, ids, columns=SC_COLUMNS):
    """{spec bus id: {column: value}} from a net after run_sc."""
    return {ident: {c: float(net.res_bus_sc.at[idx, c]) for c in columns}
            for ident, idx in ids['bus'].items()}


def spec_ids(net):
    """The spec-built net's index -> id registry, turned round."""
    return {table: {ident: idx for idx, ident in mapping.items()}
            for table, mapping in net['electrisim_ids'].items()}


# --- tests ---------------------------------------------------------------------

@pytest.mark.parametrize('grid', GRIDS)
def test_builds_cleanly(grid):
    """No warnings: a reference grid that needs excusing is not a reference."""
    _, report = sld.build_network(load_spec(grid))
    assert report['warnings'] == []
    assert report['layout'] == load_spec(grid)['layout']


def test_transmission_grid_uses_every_element_and_switch_kind():
    net, report = sld.build_network(load_spec('reference_transmission'))
    missing = [t for t in SPEC_TABLES if report['counts'][t] == 0]
    assert not missing, f'reference grid has no {missing}'
    assert set(net.switch['et']) == SWITCH_KINDS
    # One switch open, so an open switch is drawn and honoured too.
    assert sorted(net.switch['closed'].tolist()) == [False, True, True, True, True, True]
    # Each line form appears: default type, named type, explicit impedance.
    assert net.line['std_type'].isna().sum() == 1
    assert net.line['std_type'].nunique() == 3


@pytest.mark.parametrize('grid', GRIDS)
def test_spec_matches_hand_built_network(grid):
    """The spec route must build the network its documentation describes."""
    net, _ = sld.build_network(load_spec(grid))
    built = results_by_id(run(net), spec_ids(net))

    oracle = HAND_BUILT[grid]()
    expected = results_by_id(run(oracle.net), oracle.ids)

    for table in RESULT_COLUMNS:
        assert set(built[table]) == set(expected[table]), f'{table}: element set differs'
    worst = []
    for table, rows in expected.items():
        for ident, values in rows.items():
            for column, want in values.items():
                got = built[table][ident][column]
                if abs(got - want) > ORACLE_TOL:
                    worst.append(f'{table} {ident} {column}: {want!r} by hand, {got!r} by spec')
    assert not worst, f'{grid}: {len(worst)} value(s) differ\n  ' + '\n  '.join(worst[:20])


def _summarise(grid):
    net, _ = sld.build_network(load_spec(grid))
    results = results_by_id(run(net), spec_ids(net))
    results['bus_sc'] = sc_by_id(run_sc(net), spec_ids(net))
    results['bus_sc_2ph'] = sc_by_id(run_sc(net, '2ph'), spec_ids(net))
    results['bus_sc_1ph'] = sc_by_id(run_sc(net, '1ph'), spec_ids(net), SC_1PH_COLUMNS)
    results['bus_sc_min'] = sc_by_id(run_sc(net, '3ph', 'min'), spec_ids(net))
    results['bus_sc_2ph_min'] = sc_by_id(run_sc(net, '2ph', 'min'), spec_ids(net))
    results['bus_sc_1ph_min'] = sc_by_id(run_sc(net, '1ph', 'min'), spec_ids(net), SC_1PH_COLUMNS)
    return {table: {ident: {c: round(v, 10) for c, v in values.items()}
                    for ident, values in rows.items()}
            for table, rows in results.items()}


def _tolerance(column):
    if column == 'vm_pu':
        return VM_TOL
    if column == 'va_degree':
        return VA_TOL
    return FLOW_TOL


@pytest.mark.parametrize('case', ('max', 'min'))
@pytest.mark.parametrize('fault', ('3ph', '2ph'))
@pytest.mark.parametrize('grid', GRIDS)
def test_short_circuit_matches_hand_built_network(grid, fault, case):
    """Short-circuit data, given or defaulted, must reach the network as documented."""
    net, _ = sld.build_network(load_spec(grid))
    built = sc_by_id(run_sc(net, fault, case), spec_ids(net))
    oracle = HAND_BUILT[grid]()
    expected = sc_by_id(run_sc(oracle.net, fault, case), oracle.ids)
    worst = [f'bus {ident} {c}: {want[c]!r} by hand, {built[ident][c]!r} by spec'
             for ident, want in expected.items() for c in SC_COLUMNS
             if abs(built[ident][c] - want[c]) > ORACLE_TOL]
    assert not worst, f'{grid}: {len(worst)} value(s) differ\n  ' + '\n  '.join(worst[:20])


@pytest.mark.parametrize('case', ('max', 'min'))
@pytest.mark.parametrize('grid', GRIDS)
def test_single_phase_short_circuit_matches_hand_built_network(grid, case):
    """Zero-sequence data, given or defaulted, reaches the network as documented."""
    net, _ = sld.build_network(load_spec(grid))
    built = sc_by_id(run_sc(net, '1ph', case), spec_ids(net), SC_1PH_COLUMNS)
    oracle = HAND_BUILT[grid]()
    expected = sc_by_id(run_sc(oracle.net, '1ph', case), oracle.ids, SC_1PH_COLUMNS)
    worst = [f'bus {ident} {c}: {want[c]!r} by hand, {built[ident][c]!r} by spec'
             for ident, want in expected.items() for c in SC_1PH_COLUMNS
             if abs(built[ident][c] - want[c]) > ORACLE_TOL]
    assert not worst, f'{grid}: {len(worst)} value(s) differ\n  ' + '\n  '.join(worst[:20])


@pytest.mark.parametrize('grid', GRIDS)
def test_matches_golden(regen, grid):
    actual = _summarise(grid)
    path = os.path.join(GOLDEN_DIR, f'{grid}.json')

    if regen:
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(actual, handle, indent=2, sort_keys=True)
        pytest.skip(f'regenerated {os.path.basename(path)}')

    assert os.path.exists(path), f'missing golden {path} - create it with: pytest --regen-golden'
    with open(path, encoding='utf-8') as handle:
        expected = json.load(handle)

    drifted = []
    for table, rows in expected.items():
        assert set(actual[table]) == set(rows), f'{table}: element set changed'
        for ident, values in rows.items():
            for column, want in values.items():
                got = actual[table][ident][column]
                if abs(got - want) > _tolerance(column):
                    drifted.append(f'{table} {ident} {column}: {want} -> {got}')
    assert not drifted, f'{grid}: {len(drifted)} value(s) changed\n  ' + '\n  '.join(drifted[:20])


# Tables of the canvas model, keyed as /build-model names them, against the spec
# lists they come from. The name is the first field of every row but a shunt's,
# which follows pandapower's order (bus, name, ...) as the importer expects.
MODEL_TABLES = {
    'bus': 'buses', 'ext_grid': 'external_grids', 'trafo': 'transformers',
    'trafo3w': 'three_winding_transformers', 'line': 'lines', 'load': 'loads',
    'gen': 'generators', 'sgen': 'static_generators', 'shunt': 'shunts',
    'storage': 'storage', 'switch': 'switches',
}
NAME_FIELD = {'shunt': 1}


@pytest.mark.parametrize('grid', GRIDS)
def test_build_model_route(client, grid):
    """The model the canvas draws holds every element, labelled with its name."""
    spec = load_spec(grid)
    response = client.post('/build-model', json={'spec': spec, 'run_power_flow': True,
                                                 'include_model': True})
    assert response.status_code == 200, response.get_data(as_text=True)[:400]
    data = response.get_json()
    assert data['power_flow']['converged']
    assert data['report']['warnings'] == []

    model = data['model']
    model = json.loads(model) if isinstance(model, str) else model
    for table, key in MODEL_TABLES.items():
        rows = json.loads(model['_object'][table]['_object'])['data']
        want = [str(e.get('name') or e['id']) for e in spec.get(key, [])]
        at = NAME_FIELD.get(table, 0)
        assert [str(r[at]) for r in rows] == want, f'{table}: names drawn differ from the spec'


# --- the drawn diagram ------------------------------------------------------------

# Load-flow response sections, against the result tables they report.
DRAWN_SECTIONS = {
    'bus': ('busbars', ('vm_pu', 'va_degree')),
    'trafo': ('transformers', ('p_hv_mw', 'q_hv_mvar', 'loading_percent')),
    'trafo3w': ('transformers3W', ('p_hv_mw', 'p_mv_mw', 'p_lv_mw', 'loading_percent')),
    'ext_grid': ('externalgrids', ('p_mw', 'q_mvar')),
    'gen': ('generators', ('p_mw', 'q_mvar')),
}
SPEC_LISTS = {
    'bus': 'buses', 'line': 'lines', 'trafo': 'transformers',
    'trafo3w': 'three_winding_transformers', 'ext_grid': 'external_grids', 'gen': 'generators',
}
# The diagram carries every value as text and the backend rebuilds the network
# from it, so this allows for rounding only.
DRAWN_TOL = 1e-8


@pytest.mark.parametrize('grid', GRIDS)
def test_drawn_diagram_matches_spec(client, quiet, grid):
    """What the canvas drew from a spec must compute the spec's answer."""
    with open(os.path.join(REFERENCE_DIR, f'{grid}.diagram_payload.json'), encoding='utf-8') as handle:
        payload = json.load(handle)
    with quiet():
        response = client.post('/', json=payload)
    assert response.status_code == 200, response.get_data(as_text=True)[:400]
    drawn = json.loads(response.get_data(as_text=True))
    assert not drawn.get('error'), drawn.get('message') or drawn.get('exception')

    spec = load_spec(grid)
    net, _ = sld.build_network(spec)
    want = results_by_id(run(net), spec_ids(net))

    # Results come back under canvas cell names; the request maps each to the
    # label the diagram shows, which is the spec name.
    label_of_cell = {v['name']: v.get('userFriendlyName') for v in payload.values()
                     if isinstance(v, dict) and 'name' in v}

    def drawn_rows(section):
        return {label_of_cell.get(r['name'], r['name']): r for r in drawn.get(section, [])}

    def label(table, ident):
        row = next(e for e in spec.get(SPEC_LISTS[table], []) if e['id'] == ident)
        return str(row.get('name') or ident)

    differ = []
    for table, (section, columns) in DRAWN_SECTIONS.items():
        rows = drawn_rows(section)
        for ident, values in want[table].items():
            got = rows.get(label(table, ident))
            if got is None:
                differ.append(f'{table} {ident}: not in the drawn load flow')
                continue
            for column in columns:
                if abs(float(got[column]) - values[column]) > DRAWN_TOL:
                    differ.append(f'{table} {ident} {column}: spec {values[column]!r}, '
                                  f'drawn {got[column]!r}')

    # A line drawn from its other end reports from/to swapped - the same flow.
    rows = drawn_rows('lines')
    for ident, values in want['line'].items():
        got = rows.get(label('line', ident))
        if got is None:
            differ.append(f'line {ident}: not in the drawn load flow')
            continue
        flow = (float(got['p_from_mw']), float(got['q_from_mvar']))
        ends = [(values['p_from_mw'], values['q_from_mvar']), (values['p_to_mw'], values['q_to_mvar'])]
        if not any(abs(flow[0] - p) <= DRAWN_TOL and abs(flow[1] - q) <= DRAWN_TOL for p, q in ends):
            differ.append(f'line {ident}: spec flow {ends[0]} (or {ends[1]} from the other end), '
                          f'drawn {flow}')
        if abs(float(got['loading_percent']) - values['loading_percent']) > DRAWN_TOL:
            differ.append(f'line {ident} loading_percent: spec {values["loading_percent"]!r}, '
                          f'drawn {got["loading_percent"]!r}')

    assert not differ, f'{grid}: the drawn diagram differs from the spec\n  ' + '\n  '.join(differ[:20])


# Each fault type's request, as the frontend sent it with that fault selected.
SC_FIXTURE = {('3ph', 'max'): 'diagram_sc_payload', ('2ph', 'max'): 'diagram_sc2ph_payload',
              ('1ph', 'max'): 'diagram_sc1ph_payload',
              ('3ph', 'min'): 'diagram_sc_min_payload', ('2ph', 'min'): 'diagram_sc2ph_min_payload',
              ('1ph', 'min'): 'diagram_sc1ph_min_payload'}


def _sc_fixture(grid, fault, case):
    """The drawn request for this fault and case, checked to be for them."""
    with open(os.path.join(REFERENCE_DIR, f'{grid}.{SC_FIXTURE[(fault, case)]}.json'),
              encoding='utf-8') as handle:
        payload = json.load(handle)
    params = next(v for v in payload.values() if 'Parameters' in str(v.get('typ')))
    assert (params['fault_type'], params['fault_location']) == (fault, case)
    return payload


@pytest.mark.parametrize('case', ('max', 'min'))
@pytest.mark.parametrize('fault', ('3ph', '2ph'))
@pytest.mark.parametrize('grid', GRIDS)
def test_drawn_diagram_short_circuit_matches_spec(client, quiet, grid, fault, case):
    """
    A maximum three-phase or two-phase fault at every bus of the drawn diagram
    must give the spec's currents - initial, peak and thermal, all of which
    pandapower computes for these faults. This needs the external grid's fault
    level, the machines' short-circuit data and the inverter model to have
    survived the drawing - none of which the load flow notices.
    """
    payload = _sc_fixture(grid, fault, case)
    with quiet():
        response = client.post('/', json=payload)
    assert response.status_code == 200, response.get_data(as_text=True)[:400]
    drawn = json.loads(response.get_data(as_text=True))
    assert not drawn.get('error'), drawn.get('message') or drawn.get('exception')

    spec = load_spec(grid)
    net, _ = sld.build_network(spec)
    want = sc_by_id(run_sc(net, fault, case), spec_ids(net))

    label_of_cell = {v['name']: v.get('userFriendlyName') for v in payload.values()
                     if isinstance(v, dict) and 'name' in v}
    rows = {label_of_cell.get(r['name'], r['name']): r for r in drawn.get('busbars', [])}
    differ = []
    for bus in spec['buses']:
        got = rows.get(str(bus.get('name') or bus['id']))
        if got is None:
            differ.append(f"bus {bus['id']}: not in the drawn short circuit")
            continue
        for column in SC_COLUMNS:
            if abs(float(got[column]) - want[bus['id']][column]) > DRAWN_TOL:
                differ.append(f"bus {bus['id']} {column}: spec {want[bus['id']][column]!r}, "
                              f"drawn {got[column]!r}")
    assert not differ, f'{grid} {fault}: the drawn short circuit differs\n  ' + '\n  '.join(differ[:20])


def test_short_circuit_names_missing_machine_data(client, quiet):
    """
    A diagram whose machines lack short-circuit data - every generator placed
    before the spec carried it - gets told which element needs which field, not
    pandas' "'DataFrame' object has no attribute 'vn_kv'".
    """
    with open(os.path.join(REFERENCE_DIR, 'reference_radial.diagram_sc_payload.json'),
              encoding='utf-8') as handle:
        payload = json.load(handle)
    for element in payload.values():
        if element.get('typ') == 'Generator':
            element.update(vn_kv='0', xdss_pu='0', rdss_ohm='0', cos_phi='0')
        if element.get('typ') in ('Static Generator', 'Wind Turbine'):
            element['sn_mva'] = 'null'
    with quiet():
        response = client.post('/', json=payload)
    # The frontend drops any other status without showing the message.
    assert response.status_code == 200
    message = json.loads(response.get_data(as_text=True)).get('message', '')
    assert "generator 'Gas engine' has no vn_kv" in message
    assert 'xdss_pu' in message and 'cos_phi' in message
    assert "static generator 'Wind farm C' has no sn_mva" in message
    assert "static generator 'Rooftop PV' has no sn_mva" in message
    assert 'DataFrame' not in message


# --- harmonics (OpenDSS) -------------------------------------------------------------

# OpenDSS models the grid behind its short-circuit impedance where pandapower's
# external grid is ideal, so fundamental voltages agree to this, not to rounding.
OPENDSS_VM_TOL = 0.01


@pytest.fixture
def opendss_scratch(tmp_path):
    """
    Point OpenDSS at a scratch directory. It saves solved voltages to a file in
    its data path - the backend directory by default - and a backend running
    from there holds that file open, failing the run with "Error opening/
    creating file to save voltages".
    """
    import opendssdirect as dss
    before = dss.Basic.DataPath()
    dss.Basic.DataPath(str(tmp_path))
    yield
    dss.Basic.DataPath(before)


def _post_harmonics(client, quiet, payload):
    with quiet():
        response = client.post('/', json=payload)
    assert response.status_code == 200, response.get_data(as_text=True)[:400]
    result = json.loads(response.get_data(as_text=True))
    assert not result.get('error'), result.get('error')
    return result


def _harmonic_payload(grid):
    with open(os.path.join(REFERENCE_DIR, f'{grid}.diagram_harmonic_payload.json'),
              encoding='utf-8') as handle:
        return json.load(handle)


@pytest.mark.parametrize('grid', GRIDS)
def test_drawn_diagram_harmonics_reach_every_bus(client, quiet, opendss_scratch, grid):
    """
    A harmonic analysis of the drawn diagram energises every bus at the spec's
    voltage and reports distortion at every bus. Lines behind a breaker used
    to reach OpenDSS with no bus at that end, leaving everything beyond them at
    0 pu, and a bus fed only through a three-winding transformer read 0 % THD.
    """
    payload = _harmonic_payload(grid)
    result = _post_harmonics(client, quiet, payload)

    spec = load_spec(grid)
    net, _ = sld.build_network(spec)
    want = results_by_id(run(net), spec_ids(net))['bus']
    label_of_cell = {v['name']: v.get('userFriendlyName') for v in payload.values()
                     if isinstance(v, dict) and 'name' in v}
    rows = {label_of_cell.get(r['name'], r['name']): r for r in result['busbars']}

    differ = []
    for bus in spec['buses']:
        got = rows.get(str(bus.get('name') or bus['id']))
        if got is None:
            differ.append(f"bus {bus['id']}: not in the harmonic results")
            continue
        if abs(float(got['vm_pu']) - want[bus['id']]['vm_pu']) > OPENDSS_VM_TOL:
            differ.append(f"bus {bus['id']}: {got['vm_pu']} pu in OpenDSS, "
                          f"{want[bus['id']]['vm_pu']:.4f} pu in the spec")
        if not float(got.get('vthd_percent') or 0) > 0:
            differ.append(f"bus {bus['id']}: no distortion reported")
    assert not differ, f'{grid}:\n  ' + '\n  '.join(differ)


def test_opendss_takes_defaults_for_null_text(client, quiet, opendss_scratch):
    """
    Diagrams imported before the frontend stopped writing pandapower's empty
    cells hold them as the text "null". OpenDSS stopped on the first one with
    "could not convert string to float: 'null'"; it now uses its defaults and
    gives the same answer as for a clean diagram.
    """
    clean = _post_harmonics(client, quiet, _harmonic_payload('reference_radial'))

    stale = _harmonic_payload('reference_radial')
    for element in stale.values():
        typ = str(element.get('typ', ''))
        if typ.startswith('Transformer'):
            element.update(tap_pos='null', tap_step_percent='null', tap_side='null',
                           tap_neutral='null')
        elif typ.startswith('Storage'):
            element.update(sn_mva='null', soc_percent='null', type='null')
        elif typ.startswith('Load'):
            element['sn_mva'] = 'null'
    result = _post_harmonics(client, quiet, stale)

    got = {r['name']: float(r['vm_pu']) for r in result['busbars']}
    for row in clean['busbars']:
        assert got[row["name"]] == pytest.approx(float(row["vm_pu"]), abs=1e-3)


# The OpenDSS load flow feeds the grid through its short-circuit impedance where
# pandapower's external grid is ideal; voltages, and so flows, differ by that.
OPENDSS_LF_VM_TOL = 0.002   # pu
OPENDSS_LF_P_TOL = 0.05     # MW at the external grid
OPENDSS_STORAGE_TOL = 1e-3  # MW


@pytest.mark.parametrize('grid', GRIDS)
def test_drawn_diagram_opendss_load_flow_matches_spec(client, quiet, opendss_scratch, grid):
    """
    The OpenDSS load flow of the drawn diagram gives the spec's answer: every
    bus at the spec's voltage - generators at their own setpoint, which OpenDSS
    used to ignore - the grid supplying the spec's active power, and each
    battery at its dispatch, which OpenDSS refused while the drawing gave it no
    state of charge.
    """
    with open(os.path.join(REFERENCE_DIR, f'{grid}.diagram_opendss_payload.json'),
              encoding='utf-8') as handle:
        payload = json.load(handle)
    result = _post_harmonics(client, quiet, payload)

    spec = load_spec(grid)
    net, _ = sld.build_network(spec)
    want = results_by_id(run(net), spec_ids(net))
    label_of_cell = {v['name']: v.get('userFriendlyName') for v in payload.values()
                     if isinstance(v, dict) and 'name' in v}
    rows = {label_of_cell.get(r['name'], r['name']): r for r in result['busbars']}

    differ = []
    for bus in spec['buses']:
        got = rows.get(str(bus.get('name') or bus['id']))
        if got is None:
            differ.append(f"bus {bus['id']}: not in the OpenDSS results")
        elif abs(float(got['vm_pu']) - want['bus'][bus['id']]['vm_pu']) > OPENDSS_LF_VM_TOL:
            differ.append(f"bus {bus['id']}: {got['vm_pu']} pu in OpenDSS, "
                          f"{want['bus'][bus['id']]['vm_pu']:.4f} pu in the spec")
    grid_p = float(result['externalgrids'][0]['p_mw'])
    if abs(grid_p - want['ext_grid']['Grid']['p_mw']) > OPENDSS_LF_P_TOL:
        differ.append(f"external grid: {grid_p:.4f} MW in OpenDSS, "
                      f"{want['ext_grid']['Grid']['p_mw']:.4f} MW in the spec")
    drawn_storage = sorted(float(s['p_mw']) for s in result.get('storages', []))
    spec_storage = sorted(float(s['p_mw']) for s in spec.get('storage', []))
    if len(drawn_storage) != len(spec_storage) or any(
            abs(a - b) > OPENDSS_STORAGE_TOL for a, b in zip(drawn_storage, spec_storage)):
        differ.append(f'storage dispatch {drawn_storage} MW in OpenDSS, {spec_storage} in the spec')
    assert not differ, f'{grid}:\n  ' + '\n  '.join(differ)


def test_opendss_warnings_name_elements_by_their_label(client, quiet, opendss_scratch):
    """
    OpenDSS warnings name the element the way the diagram does. The payload
    used to carry no label for most elements, so a warning read "Load
    'mxCell_214'" or "Storage 'mxCell_220'".
    """
    with open(os.path.join(REFERENCE_DIR, 'reference_radial.diagram_opendss_payload.json'),
              encoding='utf-8') as handle:
        payload = json.load(handle)
    # An empty battery is held idle, and without LA2 the LV network is cut off:
    # each draws a warning.
    payload = {k: v for k, v in payload.items() if v.get('userFriendlyName') != 'LA2'}
    for element in payload.values():
        if str(element.get('typ', '')).startswith('Storage'):
            element['soc_percent'] = '0'
    with quiet():
        response = client.post('/', json=payload)
    result = json.loads(response.get_data(as_text=True))
    warnings = ' | '.join(str(w) for w in (result.get('warnings') or []))
    assert "Storage 'Battery'" in warnings, warnings
    assert "Load 'LD_LVA'" in warnings, warnings
    assert 'mxCell' not in warnings, warnings


def _post_short_circuit(client, quiet, payload):
    with quiet():
        response = client.post('/', json=payload)
    assert response.status_code == 200, response.get_data(as_text=True)[:400]
    return json.loads(response.get_data(as_text=True))


@pytest.mark.parametrize('case', ('max', 'min'))
@pytest.mark.parametrize('grid', GRIDS)
def test_drawn_diagram_single_phase_short_circuit_matches_spec(client, quiet, grid, case):
    """
    An earth fault at every bus of the drawn diagram gives the spec's currents.
    The drawing used to carry no zero sequence - a zero-impedance grid (a NaN
    in the Ybus), 0.1 ohm/km placeholder lines, a YNyn0yn0 three-winding
    transformer pandapower cannot fault - so this checks it all arrived.
    """
    payload = _sc_fixture(grid, '1ph', case)
    drawn = _post_short_circuit(client, quiet, payload)
    assert not drawn.get('error'), drawn.get('message') or drawn.get('exception')

    spec = load_spec(grid)
    net, _ = sld.build_network(spec)
    want = sc_by_id(run_sc(net, '1ph', case), spec_ids(net), SC_1PH_COLUMNS)
    label_of_cell = {v['name']: v.get('userFriendlyName') for v in payload.values()
                     if isinstance(v, dict) and 'name' in v}
    rows = {label_of_cell.get(r['name'], r['name']): r for r in drawn.get('busbars', [])}

    # IEC 60909-0 peak and thermal current for an earth fault, from
    # pandapower's own three-phase kappa and m at each bus:
    # ip1 = kappa sqrt(2) Ik1'', ith1 = Ik1'' sqrt(m + 1).
    kappa, m = iec_kappa_and_m(spec, case)

    differ = []
    for bus in spec['buses']:
        got = rows.get(str(bus.get('name') or bus['id']))
        if got is None:
            differ.append(f"bus {bus['id']}: not in the drawn short circuit")
            continue
        ikss = want[bus['id']]['ikss_ka']
        for column, expected in (('ikss_ka', ikss),
                                 ('ip_ka', kappa[bus['id']] * np.sqrt(2) * ikss),
                                 ('ith_ka', ikss * np.sqrt(m[bus['id']] + 1))):
            if abs(float(got[column]) - expected) > DRAWN_TOL:
                differ.append(f"bus {bus['id']} {column}: expected {expected!r}, "
                              f"drawn {got[column]!r}")
    assert not differ, f'{grid}: the drawn earth fault differs\n  ' + '\n  '.join(differ)


def iec_kappa_and_m(spec, case='max'):
    """
    Each bus's peak factor kappa and thermal factor m for a maximum
    three-phase fault, read from pandapower's own result table - the backend
    computes m itself, and pandapower assumes 50 Hz, as these grids are.
    """
    from pandapower.pypower.idx_bus_sc import KAPPA, M
    net, _ = sld.build_network(spec)
    run_sc(net, '3ph', case)
    rows = net['_pd2ppc_lookups']['bus']
    table = net['_ppc']['bus']
    ids = spec_ids(net)['bus']
    return ({ident: float(table[rows[idx], KAPPA]) for ident, idx in ids.items()},
            {ident: float(table[rows[idx], M]) for ident, idx in ids.items()})


def test_single_phase_short_circuit_names_missing_zero_sequence(client, quiet):
    """
    A grid without zero-sequence data, or a three-winding transformer in a
    vector group pandapower cannot fault, is named - where the run used to
    stop on "nan value detected in Ybus matrix" and blame s_sc_min_mva.
    """
    with open(os.path.join(REFERENCE_DIR, 'reference_transmission.diagram_sc1ph_payload.json'),
              encoding='utf-8') as handle:
        payload = json.load(handle)
    for element in payload.values():
        typ = str(element.get('typ', ''))
        if typ.startswith('External Grid'):
            element.update(x0x_max='0', r0x0_max='0')
        if typ.startswith('Three Winding Transformer'):
            element['vector_group'] = 'YNyn0yn0'
    message = _post_short_circuit(client, quiet, payload).get('message', '')
    assert "external grid 'Utility 110 kV' has no x0x_max" in message, message
    assert "'Main transformer 110/20/10' has vector group 'YNynyn'" in message, message
    assert 'nan value' not in message.lower()


def test_minimum_case_names_missing_line_temperature(client, quiet):
    """
    A minimum-case study on lines without an end-of-fault temperature - the
    canvas stores 0 - is refused with the lines named. At 0 degC the lines came
    out 8 % less resistive than at 20 degC, so the "minimum" currents were too
    high, with no warning.
    """
    payload = _sc_fixture('reference_radial', '3ph', 'min')
    for element in payload.values():
        if str(element.get('typ', '')).startswith('Line'):
            element['endtemp_degree'] = '0'
        if str(element.get('typ', '')).startswith('External Grid'):
            element['s_sc_min_mva'] = '0'
    message = _post_short_circuit(client, quiet, payload).get('message', '')
    assert "line 'LA2' has no endtemp_degree" in message, message
    assert "external grid 'Grid' has no s_sc_min_mva" in message, message
