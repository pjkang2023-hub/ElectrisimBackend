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
import math
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
               'shunt', 'storage', 'motor', 'switch')
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


def _opf_data(net, prices):
    """
    What the spec reference documents for the optimal power flow, written out:
    buses held to 0.9-1.1 pu; the grid free to import or export; generators
    dispatchable from nothing to their rating, with the reactive power their
    rated power factor allows; priced static generators curtailable from their
    p_mw down, with reactive power up to power factor 0.9 at their rating;
    storage fixed. prices maps
    (table, index) -> cost per MWh.
    """
    import math
    net.bus['min_vm_pu'] = 0.9
    net.bus['max_vm_pu'] = 1.1
    net.ext_grid['min_p_mw'] = -1e6
    net.ext_grid['max_p_mw'] = 1e6
    for idx in net.gen.index:
        sn = float(net.gen.at[idx, 'sn_mva'])
        q = round(sn * math.sin(math.acos(float(net.gen.at[idx, 'cos_phi']))), 6)
        net.gen.loc[idx, ['controllable', 'min_p_mw', 'max_p_mw', 'min_q_mvar', 'max_q_mvar']] =             (True, 0.0, sn, -q, q)
    for idx in net.sgen.index:
        controllable = ('sgen', idx) in prices
        net.sgen.loc[idx, ['controllable']] = controllable
        if controllable:
            p = float(net.sgen.at[idx, 'p_mw'])
            q = round(float(net.sgen.at[idx, 'sn_mva']) * math.sin(math.acos(0.9)), 6)
            net.sgen.loc[idx, ['min_p_mw', 'max_p_mw', 'min_q_mvar', 'max_q_mvar']] = (0.0, p, -q, q)
    net.storage['controllable'] = False
    for (table, idx), cost in prices.items():
        pp.create_poly_cost(net, idx, table, cp1_eur_per_mw=cost)


def hand_built_transmission():
    """reference_transmission.spec.json, with the documented defaults written out."""
    h = _Builder(50.0)
    for ident, kv in (('HV', 110), ('HV2', 110), ('MV1', 20), ('MV2', 20), ('F1', 20),
                      ('F2', 20), ('F3', 20), ('TERT', 10), ('LV1', 0.4), ('LV2', 0.4),
                      ('WFC', 20)):
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
    # The wind farm's own connection, so grid-code studies have a plant PCC.
    h.add('line', 'L_WF', pp.create_line(net, b('F3'), b('WFC'), 1.5,
                                         std_type='NA2XS2Y 1x240 RM/25 12/20 kV'))

    for ident, bus, p, q in (('LD_MV1', 'MV1', 6.0, 2.0), ('LD_F3', 'F3', 3.0, 1.0),
                             ('LD_LV1', 'LV1', 0.4, 0.4 * 0.33),  # unstated q: 0.33 x p
                             ('LD_LV2', 'LV2', 0.6, 0.15), ('LD_AUX', 'TERT', 0.5, 0.2)):
        h.add('load', ident, pp.create_load(net, b(bus), p_mw=p, q_mvar=q))
    h.add('gen', 'G1', pp.create_gen(net, b('MV2'), p_mw=4.0, vm_pu=1.025, sn_mva=6.0,
                                     vn_kv=20, xdss_pu=0.18, rdss_ohm=0.02, cos_phi=0.8))
    # Unstated sgen rating: 1.1 x p_mw, at least 0.1 MVA; k defaults to 1.1.
    h.add('sgen', 'PV', pp.create_sgen(net, b('LV2'), p_mw=0.3, q_mvar=0.0, sn_mva=0.33, k=1.1))
    h.add('sgen', 'WF', pp.create_sgen(net, b('WFC'), p_mw=2.0, q_mvar=-0.2, sn_mva=2.5, k=1.2))
    h.add('shunt', 'SR', pp.create_shunt(net, b('TERT'), q_mvar=2.0, p_mw=0.0))
    h.add('shunt', 'CAP', pp.create_shunt(net, b('F1'), q_mvar=-1.5, p_mw=0.0))
    h.add('storage', 'BESS', pp.create_storage(net, b('LV1'), p_mw=0.1, max_e_mwh=0.5))
    # Rated values default to the operating ones; vn_kv to the bus.
    h.add('motor', 'M1', pp.create_motor(net, b('TERT'), pn_mech_mw=0.8, cos_phi=0.88,
                                         efficiency_percent=96, lrc_pu=6.5, rx=0.1, vn_kv=10,
                                         cos_phi_n=0.88, efficiency_n_percent=96))

    ids = h.ids
    # T2 is given as YNyn0 - it runs in parallel with the three-winding unit's
    # YNyn HV-MV path, so the phase shifts must match; the rest take Dyn, the
    # three-winding unit YNynd.
    _zero_sequence(net, {ids['trafo']['T2']: 'YNyn', ids['trafo']['T_LV1']: 'Dyn',
                         ids['trafo']['T_LV2']: 'Dyn'},
                   trafo3w={ids['trafo3w']['T3W']: 'YNynd'},
                   lines={ids['line']['L_HV']: (0.35, 1.25, 5.0)},
                   endtemps={ids['line']['L_HV']: 70.0})
    _opf_data(net, {('ext_grid', ids['ext_grid']['Grid']): 60.0, ('gen', ids['gen']['G1']): 45.0,
                    ('sgen', ids['sgen']['PV']): 0.0, ('sgen', ids['sgen']['WF']): 0.0})
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
    # R/X 0.15 by default.
    h.add('motor', 'M1', pp.create_motor(net, b('LVA'), pn_mech_mw=0.075, cos_phi=0.85,
                                         efficiency_percent=93, lrc_pu=7.2, rx=0.15, vn_kv=0.4,
                                         cos_phi_n=0.85, efficiency_n_percent=93))

    ids = h.ids
    _zero_sequence(net, {ids['trafo']['T1']: 'Dyn', ids['trafo']['TA']: 'Dyn'})
    _opf_data(net, {('ext_grid', ids['ext_grid']['Grid']): 50.0, ('gen', ids['gen']['GE']): 80.0,
                    ('sgen', ids['sgen']['WF']): 0.0, ('sgen', ids['sgen']['PV']): 0.0})
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
    'storage': 'storage', 'motor': 'motors', 'switch': 'switches',
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
# from it, so this allows for rounding and solver tolerance only: pandapower's
# load flow converges to a 1e-8 MVA mismatch, and a column the solver does not
# use - a generator's reactive limits - moves its last digits by about 5e-8.
DRAWN_TOL = 1e-6


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


# --- optimal power flow ----------------------------------------------------------

# The interior-point solver stops within its own tolerance; the backend and a
# plain runopp land this close on the same network.
OPF_P_TOL = 1e-3      # MW
OPF_COST_TOL = 1e-2   # per hour

# Spec lists whose elements the OPF request must carry, by the request's type.
OPF_ELEMENTS = {
    'lines': 'Line', 'transformers': 'Transformer',
    'three_winding_transformers': 'Three Winding Transformer', 'loads': 'Load',
    'generators': 'Generator', 'static_generators': 'Static Generator',
    'storage': 'Storage', 'switches': 'Switch', 'external_grids': 'External Grid',
    'motors': 'Motor',
}


@pytest.mark.parametrize('grid', GRIDS)
def test_drawn_diagram_optimal_power_flow(client, quiet, grid):
    """
    The minimum-cost OPF on the drawn network: the whole network is sent and
    every source is dispatched as pandapower dispatches the spec.

    Its builder read a line's buses from the line's own ends and sent no
    switches, so everything behind a breaker came back isolated and the OPF
    failed; it also left out wind turbines, shunt reactors and capacitor
    banks. And the spec's prices and limits had no way onto the drawing.
    """
    with open(os.path.join(REFERENCE_DIR, f'{grid}.diagram_opf_payload.json'),
              encoding='utf-8') as handle:
        payload = json.load(handle)
    spec = load_spec(grid)
    sent = list(payload.values())

    def sent_of(*types):
        # Request types carry a running number: "Line0", "Shunt Reactor1".
        return [e for e in sent if str(e.get('typ', '')).rstrip('0123456789') in types]

    for key, typ in OPF_ELEMENTS.items():
        assert len(sent_of(typ)) == len(spec.get(key, [])), f'{key}: not all sent'
    assert len(sent_of('Shunt Reactor', 'Capacitor')) == len(spec.get('shunts', [])), \
        'shunts: not all sent'
    assert all(e.get('busFrom') and e.get('busTo') for e in sent_of('Line')), 'a line has no bus'

    with quiet():
        response = client.post('/', json=payload)
    assert response.status_code == 200, response.get_data(as_text=True)[:400]
    result = json.loads(response.get_data(as_text=True))
    assert not result.get('error'), result.get('message') or result.get('error')
    assert result['opf_converged'] is True
    assert len(result['busbars']) == len(spec['buses'])
    assert all(0.9 < float(b['vm_pu']) < 1.1 for b in result['busbars']), 'a bus is not energised'
    reported = ' | '.join(str(r['name']) for key in ('generators', 'staticgenerators', 'externalgrids')
                          for r in result.get(key, []))
    for key in ('generators', 'static_generators', 'external_grids'):
        for element in spec.get(key, []):
            assert str(element.get('name') or element['id']) in reported, reported

    # The minimum-cost dispatch: the request carries the spec's prices, so each
    # source must be dispatched as pandapower dispatches the spec. Only active
    # power is priced, so reactive power has many equally cheap optima and is
    # not compared.
    params = next(v for v in sent if 'Parameters' in str(v.get('typ')))
    assert params['cost_function'] == 'polynomial'
    net, _ = sld.build_network(spec)
    pp.runopp(net, calculate_voltage_angles='auto', init='pf', delta=1e-16, trafo_model='t',
              trafo_loading='current', ac_line_model='pi')
    ids = spec_ids(net)
    drawn = {str(r['name']).rsplit(' (', 1)[0]: float(r['p_mw'])
             for key in ('generators', 'staticgenerators', 'externalgrids') for r in result.get(key, [])}
    differ = []
    for key, table in (('external_grids', 'ext_grid'), ('generators', 'gen'),
                       ('static_generators', 'sgen')):
        for element in spec.get(key, []):
            want = float(net[f'res_{table}'].at[ids[table][element['id']], 'p_mw'])
            got = drawn[str(element.get('name') or element['id'])]
            if abs(got - want) > OPF_P_TOL:
                differ.append(f"{key} {element['id']}: {got} MW drawn, {want} MW for the spec")
    if abs(float(result['total_cost']) - float(net.res_cost)) > OPF_COST_TOL:
        differ.append(f"total cost {result['total_cost']} drawn, {net.res_cost} for the spec")
    assert not differ, f'{grid}:\n  ' + '\n  '.join(differ)


# --- studies sent from the shared network builder ----------------------------
#
# Motor starting, the ANDES studies and DG screening are built by the same
# frontend code as the short circuit, so the drawn short-circuit request is
# their network too; each test swaps in its own study settings.

def _study_request(grid, params, key='0'):
    payload = _sc_fixture(grid, '3ph', 'max')
    network = {k: v for k, v in payload.items() if 'Parameters' not in str(v.get('typ'))}
    return {key: params, **network}


def _post_study(client, quiet, request):
    with quiet():
        response = client.post('/', json=request)
    assert response.status_code == 200, response.get_data(as_text=True)[:400]
    result = json.loads(response.get_data(as_text=True))
    assert not result.get('error'), result.get('message') or result.get('error')
    return result


def _label_of_cell(request):
    return {v['name']: v.get('userFriendlyName') for v in request.values()
            if isinstance(v, dict) and 'name' in v}


def _bus_label(spec, ident):
    row = next(b for b in spec['buses'] if b['id'] == ident)
    return str(row.get('name') or ident)


@pytest.mark.parametrize('grid', GRIDS)
def test_drawn_diagram_motor_starting(client, quiet, grid):
    """
    Direct-on-line start of the drawn motor: before, during (the motor as its
    locked-rotor load) and the dip at every bus, as pandapower gives them for
    the spec.

    The study's settings row ("MotorStartingPandaPower Parameters") matched
    the backend's Motor branch and failed on its missing bus, so no motor
    start ever ran.
    """
    request = _study_request(grid, {
        'typ': 'MotorStartingPandaPower Parameters', 'mode': 'steady', 'motor_ids': 'all',
        'starting_method': 'dol', 'voltage_limit_percent': '15', 'thermal_limit_percent': '100',
        'frequency': '50'})
    result = _post_study(client, quiet, request)

    spec = load_spec(grid)
    (motor,) = spec['motors']
    rx = motor.get('rx', 0.15)
    cos_n = motor.get('cos_phi_n', motor['cos_phi'])
    eff_n = motor.get('efficiency_n_percent', motor['efficiency_percent'])
    vn = next(b['vn_kv'] for b in spec['buses'] if b['id'] == motor['bus'])
    i_rated = motor['pn_mech_mw'] / (eff_n / 100) / cos_n / (np.sqrt(3) * vn)
    i_start = motor['lrc_pu'] * i_rated
    (started,) = result['motors']
    assert started['i_start_ka'] == pytest.approx(i_start, rel=1e-9)
    # Reported by the labels the diagram shows, not by cell id.
    assert started['name'] == motor['name']
    assert started['bus_name'] == _bus_label(spec, motor['bus'])

    # The oracle: the motor off, then its locked-rotor power at rated voltage.
    net, _ = sld.build_network(spec)
    ids = spec_ids(net)
    m = ids['motor'][motor['id']]
    net.motor.at[m, 'in_service'] = False
    before = run(net).res_bus['vm_pu'].copy()
    s_mva = np.sqrt(3) * vn * i_start
    pp.create_load(net, ids['bus'][motor['bus']], p_mw=s_mva * rx / np.sqrt(1 + rx * rx),
                   q_mvar=s_mva / np.sqrt(1 + rx * rx))
    during = run(net).res_bus['vm_pu']

    drawn = {b['name']: b for b in result['buses']}
    differ = []
    for bus in spec['buses']:
        got = drawn[_bus_label(spec, bus['id'])]
        idx = ids['bus'][bus['id']]
        for column, want in (('vm_before', before[idx]), ('vm_during', during[idx])):
            if abs(float(got[column]) - want) > 1e-4:  # reported to 4 decimals
                differ.append(f"{bus['id']} {column}: {got[column]} drawn, {want:.6f} spec")
    assert not differ, f'{grid}:\n  ' + '\n  '.join(differ)
    assert result['summary']['worst_dip_percent'] > 0.5, 'the start barely moved the voltage'


ANDES_PARAMS = {'frequency': '50', 'sn_mva': '100'}


def _assert_steady_start(result):
    assert not any('could not be initialised' in w for w in result.get('warnings', [])), \
        result['warnings']


@pytest.mark.parametrize('grid', GRIDS)
def test_drawn_diagram_transient_stability_starts_in_steady_state(client, quiet, grid):
    """
    With no event the ANDES run must stay where the power flow put it, and
    that power flow must be pandapower's.

    It crashed reading results (ANDES 2 has no plotter under a server), then
    read the wrong variables; it dropped the three-winding transformer,
    static generators without a plant model and storage; and xq'' defaulting
    apart from xd'' kept every generator with short-circuit data from
    initialising.
    """
    request = _study_request(grid, {
        'typ': 'TransientStabilityAndes Parameters', **ANDES_PARAMS, 'tf': '5', 'tstep': '0',
        'fault_enabled': 'false', 'fault_bus': '', 'toggle_line': '', 'toggle_gen': ''})
    result = _post_study(client, quiet, request)
    _assert_steady_start(result)
    assert result['converged'] is True
    freq = np.asarray(result['frequency_hz'], dtype=float)
    assert np.abs(freq - 50.0).max() < 1e-4, f'frequency drifted to {freq.min()}..{freq.max()} Hz'

    spec = load_spec(grid)
    net, _ = sld.build_network(spec)
    run(net)
    ids = spec_ids(net)
    drawn = {b['name']: b['values'][0] for b in result['bus_voltage']}
    assert len(drawn) == len(spec['buses']), sorted(drawn)
    differ = [f"{bus['id']}: {drawn[_bus_label(spec, bus['id'])]:.5f} in ANDES, "
              f"{net.res_bus.at[ids['bus'][bus['id']], 'vm_pu']:.5f} in pandapower"
              for bus in spec['buses']
              if abs(drawn[_bus_label(spec, bus['id'])]
                     - net.res_bus.at[ids['bus'][bus['id']], 'vm_pu']) > 1e-3]
    assert not differ, f'{grid}:\n  ' + '\n  '.join(differ)


@pytest.mark.parametrize('grid', GRIDS)
def test_drawn_diagram_eigenvalues(client, quiet, grid):
    """Small-signal analysis of the drawn network, linearised at a steady state."""
    request = _study_request(grid, {'typ': 'EigenvalueAndes Parameters', **ANDES_PARAMS,
                                    'n_modes': '10'})
    result = _post_study(client, quiet, request)
    _assert_steady_start(result)
    assert result['verdict'] == 'stable'
    assert result['n_positive'] == 0

    # One row per complex pair: the conjugate is the same mode.
    modes = result['least_damped_modes']
    assert modes and all(m['imag'] > 0 for m in modes)
    # Participation was always empty (a numpy array in `or`, swallowed), and
    # read ANDES's matrix the wrong way round: the machine's rotor swing must
    # come out as its speed and rotor angle, under the machine's name.
    machine = load_spec(grid)['generators'][0]['name']
    swing = max(modes, key=lambda m: m['freq_hz'])
    part = next(p for p in result['participation'] if p['mode_index'] == swing['index'])
    top = [s['state'] for s in part['states'][:2]]
    assert sorted(top) == sorted([f'{machine} (GENROU): speed ω', f'{machine} (GENROU): rotor angle δ']), part
    for p in result['participation']:
        factors = [s['factor'] for s in p['states']]
        assert all(0 <= f <= 1 for f in factors) and sum(factors) <= 1 + 1e-9, p


@pytest.mark.parametrize('grid', GRIDS)
def test_classical_machine_swing_matches_hand_calculation(client, quiet, grid):
    """
    The classical machine against the rest of the network, reduced to its
    Thevenin equivalent at the machine bus: f = sqrt(w0 Ks / 2H) / 2 pi with
    Ks = E' V cos(delta) / (X'd + Xe). The classical model took the
    subtransient xdss_pu as its X'd, so the machine was too stiff (1.94 Hz
    against 1.53 Hz on the transmission grid).
    """
    request = _study_request(grid, {'typ': 'EigenvalueAndes Parameters', **ANDES_PARAMS,
                                    'n_modes': '10'})
    machine_el = next(v for v in request.values() if str(v.get('typ', '')).startswith('Generator'))
    machine_el['dyn_machine_model'] = 'GENCLS'
    result = _post_study(client, quiet, request)
    swing = max(result['least_damped_modes'], key=lambda m: m['freq_hz'])

    spec = load_spec(grid)
    gen = spec['generators'][0]
    net, _ = sld.build_network(spec)
    run(net)
    ids = spec_ids(net)
    g = ids['gen'][gen['id']]
    bus = int(net.gen.at[g, 'bus'])
    sn, vn = float(machine_el['sn_mva']), float(net.bus.at[bus, 'vn_kv'])  # as drawn
    v = float(net.res_bus.at[bus, 'vm_pu'])
    i = complex(net.res_gen.at[g, 'p_mw'], -net.res_gen.at[g, 'q_mvar']) / sn / v
    net.gen.loc[g, 'in_service'] = False
    sc.calc_sc(net, bus=bus, case='max', ip=False)
    xe = 1.1 * vn / (math.sqrt(3) * net.res_bus_sc.at[bus, 'ikss_ka']) / (vn ** 2 / sn)
    xd1, h = 0.3, 6.0  # the defaults applied
    e, v_inf = v + 1j * xd1 * i, v - 1j * xe * i
    ks = abs(e) * abs(v_inf) * math.cos(np.angle(e) - np.angle(v_inf)) / (xd1 + xe)
    f_hand = math.sqrt(2 * math.pi * 50 * ks / (2 * h)) / (2 * math.pi)
    assert swing['freq_hz'] == pytest.approx(f_hand, rel=0.05), (swing, f_hand)


@pytest.mark.parametrize('grid', GRIDS)
def test_drawn_diagram_dg_screening(client, quiet, opendss_scratch, grid):
    """
    DG interconnection screening of the drawn rooftop PV raised to 500 kW: the
    grid's import must be pandapower's for the same change.

    The screening built its circuit from the raw External Grid element (no
    basekv) and failed; it never fell back from the solver that diverges here,
    and it read the grid's power from whichever element was active last, with
    the wrong sign - so every case showed reverse power.
    """
    spec = load_spec(grid)
    pv = next(g for g in spec['static_generators'] if g['id'] == 'PV')
    request = _study_request(grid, {}, key='dg_interconnection_params')
    cell = {v.get('userFriendlyName'): v['name'] for v in request.values()
            if isinstance(v, dict) and 'name' in v}
    request['dg_interconnection_params'] = {
        'typ': 'DgInterconnectionOpenDss', 'poc_bus_id': cell[_bus_label(spec, pv['bus'])],
        'der_id': cell[pv['name']], 'der_type': 'Generator', 'proposed_kw': 500,
        'vmin_pu': 0.95, 'vmax_pu': 1.05, 'max_loading_percent': 100,
        'run_hosting_capacity': False, 'compare_invcontrol': False, 'frequency': 50}
    result = _post_study(client, quiet, request)
    assert result['summary']['converged'] is True
    assert result['summary']['der_label'] == pv['name']
    checks = {c['id']: c for c in result['checks']}
    labels = {b.get('name') or b['id'] for b in spec['buses']}
    assert {checks['voltage_min']['location'], checks['voltage_max']['location']} <= labels,         'voltage checks must name buses as the diagram labels them'
    assert all(np.isfinite(float(c['value'])) for c in checks.values()), checks

    pv['p_mw'] = 0.5
    net, _ = sld.build_network(spec)
    grid_p = float(run(net).res_ext_grid['p_mw'].sum())
    drawn_p = checks['reverse_power']['value'] / 1000.0
    assert abs(drawn_p - grid_p) < OPENDSS_LF_P_TOL, \
        f'{grid}: grid supplies {drawn_p:.4f} MW in the screening, {grid_p:.4f} MW in pandapower'
    assert checks['reverse_power']['status'] == ('fail' if grid_p < -0.001 else 'pass')


# --- protection coordination ---------------------------------------------------
#
# reference_radial.diagram_protection_payload.json is the request the frontend
# sent for the drawn radial grid with a definite-time overcurrent relay
# (automatic pickup) on every breaker: CB_T1 on the transformer's 20 kV side,
# CB_A, CB_B and the wind feeder breaker on the feeders. Faults are placed at
# the middle of every line.
#
# reference_transmission.diagram_protection_payload.json has definite-time
# relays with pickups set by hand on every closed breaker: CB_LHV (110 kV
# line), CB_T3W and CB_T2 (110 kV side of both transformers), CB_L1 and CB_L4
# on the 20 kV ring, which both transformers feed.

def _protection_request(settings=None, grid='reference_radial'):
    with open(os.path.join(REFERENCE_DIR, f'{grid}.diagram_protection_payload.json'),
              encoding='utf-8') as handle:
        payload = json.load(handle)
    for element in payload.values():
        if str(element.get('typ', '')).startswith('Switch') and settings:
            element.update(settings(element.get('userFriendlyName')))
    return payload


def test_protection_automatic_pickup_it_cannot_build_is_not_computed(client, quiet):
    """
    pandapower's OCRelay grades only networks whose closed switches all sit on
    lines, and this one has a transformer breaker. Its fallback then read the
    dialog's unset pickups as 0 A, so every relay tripped instantly for every
    fault anywhere; the relays must be reported as not computed instead.
    """
    with quiet():
        response = client.post('/', json=_protection_request())
    result = json.loads(response.get_data(as_text=True))
    assert result['summary']['n_not_computed'] == 4
    assert not any(row.get('tripped') for s in result.get('scenarios', []) for row in s['trip'])
    reasons = [a['reason'] for a in result['attach_summaries']]
    assert all('pickup mode Manual' in r for r in reasons), reasons
    # The alert is all the user sees: it must say what to do, per relay.
    assert result['message'].count('pickup mode Manual') == 4, result['message']


def test_protection_manual_settings_grade(client, quiet):
    """
    With pickups set by hand, a fault on a feeder trips that feeder's breaker
    instantaneously and the transformer breaker as time-graded backup; the
    other feeders stay in. The transformer breaker sits on the 20 kV side, so
    it sees the 20 kV fault current - it was read from the 110 kV side - and a
    fault on the first line, whose index a transformer switch shares, must be
    evaluated too (pandapower re-pointed that switch at a line half).
    """
    def settings(name):
        if name == 'CB_T1':
            return dict(pickup_mode='manual', I_g_a='1500', I_gg_a='12000', t_g='0.8', t_gg='0.3')
        return dict(pickup_mode='manual', I_g_a='400', I_gg_a='3000', t_g='0.5', t_gg='0.07')

    request = _protection_request(settings)
    with quiet():
        response = client.post('/', json=request)
    result = json.loads(response.get_data(as_text=True))
    assert not result.get('error'), result.get('message')
    assert result['summary']['n_miscoordination'] == 0

    lines = [v for v in request.values() if str(v.get('typ', '')).startswith('Line')]
    feeder_breaker = {'LA1': 'CB_A', 'LA2': 'CB_A', 'LB1': 'CB_B', 'LB2': 'CB_B',
                      'Wind farm cable': 'Wind feeder breaker'}
    scenarios = result['scenarios']
    assert len(scenarios) == len(lines)
    for scenario in scenarios:
        assert not scenario.get('error'), scenario['error']
        line = lines[int(scenario['sc_line_id'])]['userFriendlyName']
        trips = {row['switch_name']: row for row in scenario['trip']}
        tripped = {name for name, row in trips.items() if row['tripped']}
        own = feeder_breaker[line]
        assert own in tripped and trips[own]['t_trip_s'] == pytest.approx(0.07), (line, trips)
        assert tripped <= {own, 'CB_T1'}, f'{line}: {sorted(tripped)} tripped'
        # The 20 kV-side breaker carries (nearly) the feeder's fault current.
        assert trips['CB_T1']['ikss_ka'] > 0.9 * trips[own]['ikss_ka'] - 0.3, (line, trips)
        if 'CB_T1' in tripped:
            assert trips['CB_T1']['t_trip_s'] == pytest.approx(0.8)
        # The feeder breaker clears its own fault; CB_T1 is its graded backup.
        assert scenario['fault_label'] == f'{line}, 50 %'
        assert scenario['primary_switches'] == [own]
        assert scenario['clearing_time_s'] == pytest.approx(0.07)
        # The gas engine on feeder B has no breaker of its own.
        assert scenario['unprotected_sources'] == (['Gas engine'] if own == 'CB_B' else [])
    assert result['unwanted_trips'] == []
    # Pickups set by hand need no note about pandapower's OCRelay.
    assert not any(a.get('reason') for a in result['attach_summaries'])


def test_protection_meshed_grading(client, quiet):
    """
    The 20 kV ring is fed by both transformers. Relays were graded by hop
    distance from the external grid, and a three-winding transformer was no
    path, so the two parallel transformer feeds (CB_T2, CB_T3W) were reported
    as primary and backup for five faults. Grading is by protection zone: the
    relays round the fault that lead to a source clear it, the next ones out
    back them up, and any other relay tripping before the fault is cleared is
    an unwanted trip - here the ring breakers for a 110 kV line fault.
    """
    with quiet():
        response = client.post('/', json=_protection_request(grid='reference_transmission'))
    result = json.loads(response.get_data(as_text=True))
    assert not result.get('error'), result.get('message')
    scenarios = {s['fault_label']: s for s in result['scenarios']}
    expected = {
        '110 kV overhead line, 50 %': ({'CB_LHV', 'CB_T2'}, 1.0, []),
        'L1, 50 %': ({'CB_L1', 'CB_L4'}, 0.7, []),
        'Cable with given impedance, 50 %': ({'CB_T2', 'CB_L4'}, 1.0, ['CHP plant']),
        'L3, 50 %': ({'CB_L1', 'CB_L4'}, 0.7, []),
        'L4, 50 %': ({'CB_L4', 'CB_T2'}, 1.0, ['CHP plant']),
        'Wind farm cable, 50 %': ({'CB_L1', 'CB_L4'}, 0.7, []),
    }
    assert set(scenarios) == set(expected)
    for label, (primaries, t_clear, unprotected) in expected.items():
        scenario = scenarios[label]
        assert set(scenario['primary_switches']) == primaries, label
        assert scenario['clearing_time_s'] == pytest.approx(t_clear), label
        assert scenario['unprotected_sources'] == unprotected, label

    # CB_L1 backs up CB_L4 from the far side of the ring, at the same 0.7 s.
    miscoord = {(m['fault_label'], m['primary_user_friendly_name'], m['backup_user_friendly_name'])
                for m in result['miscoordination']}
    assert miscoord == {('Cable with given impedance, 50 %', 'CB_L4', 'CB_L1'),
                        ('L4, 50 %', 'CB_L4', 'CB_L1')}
    unwanted = {(u['fault_label'], u['user_friendly_name']) for u in result['unwanted_trips']}
    assert unwanted == {('110 kV overhead line, 50 %', 'CB_L1'), ('110 kV overhead line, 50 %', 'CB_L4'),
                        ('Cable with given impedance, 50 %', 'CB_T3W'), ('L4, 50 %', 'CB_T3W')}
    assert result['summary']['n_miscoordination'] == 2
    assert result['summary']['n_unwanted_trips'] == 4


def test_site_screening_counts_lost_supply(client, quiet):
    """
    The radial grid hangs off one transformer, so N-1 can cut the data-centre
    site off. The islanded buses have no voltage (NaN), which no limit
    caught, so every site passed N-1; they must count as violations.
    """
    request = _study_request('reference_radial', {
        'typ': 'DataCenterSiteScreeningPandaPower Parameters', 'site_load_ids': 'Factory',
        'mw_sizes': '2', 'power_factor': '0.95', 'include_n11': 'false', 'element_type': 'all',
        'voltage_limits': 'true', 'thermal_limits': 'true', 'min_vm_pu': '0.95',
        'max_vm_pu': '1.05', 'max_loading_percent': '100'})
    with quiet():
        response = client.post('/', json=request)
    assert response.status_code == 200
    result = json.loads(response.get_data(as_text=True))
    (row,) = result['screening_results']
    assert row['base_violations'] == 0
    lost = {v['name'] for v in row['n1_violation_details'] if v['text'] == 'de-energised'}
    assert 'B1' in lost, row['n1_violation_details']  # the Factory's own bus
    assert row['upgrade_likely'] is True


@pytest.mark.parametrize('engine', ('opender', 'opendss'))
def test_drawn_diagram_bess_dispatch_reversal(client, quiet, opendss_scratch, engine):
    """
    The transmission grid's battery ramped from charging 0.1 MW to
    discharging 0.1 MW: the voltage at its bus must rise as pandapower says
    it does, and the result names the bus as the diagram labels it.
    """
    spec = load_spec('reference_transmission')
    (battery,) = spec['storage']
    request = _study_request('reference_transmission', {}, key='bess_dispatch_reversal_params')
    cell = {v.get('userFriendlyName'): v for v in request.values() if isinstance(v, dict) and 'name' in v}
    request['bess_dispatch_reversal_params'] = {
        'typ': 'BessDispatchReversalOpenDss', 'storage_id': cell[battery['name']]['id'],
        'poc_bus_id': cell[_bus_label(spec, battery['bus'])]['id'],
        'p_start_mw': 0.1, 'p_end_mw': -0.1, 'pre_hold_s': 1, 'ramp_s': 5, 'post_hold_s': 2,
        'dt': 0.1, 'vmin_pu': 0.98, 'vmax_pu': 1.02, 'olrt_s': 5, 'engine': engine,
        'q_source': 'inverter', 'frequency': 50}
    result = _post_study(client, quiet, request)
    assert result['converged'] is True and result['within_limits'] is True
    assert result['poc_bus_label'] == _bus_label(spec, battery['bus'])
    assert result['bus_voltage'][0]['name'] == result['poc_bus_label']

    def vm(p_mw):
        net, _ = sld.build_network(spec)
        ids = spec_ids(net)
        net.storage.at[ids['storage'][battery['id']], 'p_mw'] = p_mw
        return run(net).res_bus.at[ids['bus'][battery['bus']], 'vm_pu']

    v = result['bus_voltage'][0]['values']
    assert abs(v[0] - vm(0.1)) < OPENDSS_LF_VM_TOL
    assert abs((v[-1] - v[0]) - (vm(-0.1) - vm(0.1))) < 1e-4, 'the rise differs from pandapower'


@pytest.mark.parametrize('q_mode', ('from_sgen_curve', 'from_rating'))
def test_grid_code_pq_holds_other_generators_to_their_limits(client, quiet, q_mode):
    """
    The P-Q sweep drives the grid to 0.9-1.1 pu. The radial grid's gas engine
    (outside the plant) held its bus at 1.0 pu with no reactive limit, taking
    some 23 Mvar on a 1.8 MVA machine: feeders overloaded to 160 %, the PCC
    exchanged 20 Mvar with the plant off, and 44 points were "unphysical".
    The drawing never sent the limits; and with the plant's Q taken from its
    rating, no curve turned limit enforcement on.
    """
    with open(os.path.join(REFERENCE_DIR, 'reference_radial.diagram_grid_code_pq_payload.json'),
              encoding='utf-8') as handle:
        request = json.load(handle)
    request['0']['q_capability_mode'] = q_mode
    gas_engine = next(v for v in request.values() if v.get('typ') == 'Generator')
    assert float(gas_engine['max_q_mvar']) > float(gas_engine['min_q_mvar']), \
        "the drawn generator's reactive limits must reach the study"
    with quiet():
        response = client.post('/', json=request)
    assert response.status_code == 200
    lines = [json.loads(line) for line in response.get_data(as_text=True).splitlines() if line.strip()]
    progress = [line['message'] for line in lines if line.get('type') == 'progress']
    result = next(line for line in lines if line.get('type') == 'result')['data']['grid_code_pq_results']

    assert not [m for m in progress if 'unphysical' in m], 'unphysical points'
    assert result['summary']['maxloading_cbl'] < 100, result['summary']
    units_off = next(m for m in progress if m.startswith('Units off'))
    q_off = float(units_off.split('Q=')[1].split()[0])
    assert abs(q_off) < 1.0, units_off


def test_grid_code_pq_holds_the_pcc_at_each_voltage_level(client, quiet):
    """
    Each voltage level is set at the external grid. With the PCC at the wind
    farm's own bus, behind the 20 kV substation and the wind farm cable, it
    sat up to 1 % off the level it was reported under; the grid setpoint is
    now corrected until the PCC is at the level.
    """
    import re
    from grid_code_pq_electrisim import _PQ_PCC_V_TOL

    with open(os.path.join(REFERENCE_DIR, 'reference_radial.diagram_grid_code_pq_payload.json'),
              encoding='utf-8') as handle:
        request = json.load(handle)
    cell = {v.get('userFriendlyName'): v['name'] for v in request.values()
            if isinstance(v, dict) and 'name' in v}
    request['0']['pcc_bus_name'] = cell['Wind connection']
    request['0']['generator_names'] = [cell['Wind farm C']]
    with quiet():
        response = client.post('/', json=request)
    lines = [json.loads(line) for line in response.get_data(as_text=True).splitlines() if line.strip()]
    level, off = None, []
    for message in (line['message'] for line in lines if line.get('type') == 'progress'):
        found = re.search(r'Voltage ([\d.]+) pu \(applied', message)
        if found:
            level = float(found.group(1))
        at_pcc = re.search(r'U_pcc=([\d.]+) pu', message)
        if at_pcc and abs(float(at_pcc.group(1)) - level) > _PQ_PCC_V_TOL + 1e-4:  # 4-decimal log
            off.append(f'{level} pu level: PCC at {at_pcc.group(1)} pu')
    assert level is not None
    assert not off, off[:5]


def test_grid_code_vq_wind_farm_at_its_own_connection(client, quiet):
    """
    The transmission grid's wind farm has its own connection bus, so the V-Q
    study can be run on the plant alone: at Pmax it must cover the ENTSO-E
    U-Q/Pmax range at every voltage level, with the PCC held at the level.
    (On the Ring node, the Industrial park's load shared its bus and shifted
    the whole range by 1 Mvar.)
    """
    from grid_code_pq_electrisim import _PQ_PCC_V_TOL

    with open(os.path.join(REFERENCE_DIR, 'reference_transmission.diagram_grid_code_vq_payload.json'),
              encoding='utf-8') as handle:
        request = json.load(handle)
    with quiet():
        response = client.post('/', json=request)
    lines = [json.loads(line) for line in response.get_data(as_text=True).splitlines() if line.strip()]
    result = next(line for line in lines if line.get('type') == 'result')['data']['grid_code_vq_results']
    curve, need = result['uq_curve'], result['uq_requirements']
    assert result['p_pcc_mw'] == pytest.approx(2.0, abs=0.05)
    # Sized for the 2.0 MW wind farm chosen, not the 2.3 MW of PV and wind the
    # dialog first ticks: ENTSO-E's 0.3287 Q/Pmax x 2.0 MW.
    assert max(need['q_req_max_mvar']) == pytest.approx(0.3287 * 2.0, abs=0.005)
    for u, q_max, q_min, need_max, need_min in zip(curve['u_pu'], curve['q_max_mvar'], curve['q_min_mvar'],
                                                    need['q_req_max_mvar'], need['q_req_min_mvar']):
        assert q_max >= need_max and q_min <= need_min, f'{u} pu: {q_min}..{q_max} Mvar'
    assert result['uq_compliance'] is True
    at_pcc = [float(m.split('U_pcc=')[1].split()[0]) for m in
              (line['message'] for line in lines if line.get('type') == 'progress') if 'U_pcc=' in m]
    assert at_pcc and all(min(abs(v - u) for u in curve['u_pu']) <= _PQ_PCC_V_TOL + 1e-4 for v in at_pcc)


def test_grid_code_pmax_at_the_pcc_counts_export_only():
    """
    "Pmax at the PCC" sizes the grid-code requirement. It was the largest |P|,
    so a PCC that only imports - the transmission grid's 110 kV busbar, where
    the plant sits behind the town's load - reported its 7.5 MW import as the
    plant's Pmax and scaled the requirement to three times the plant.
    """
    from grid_code_pq_electrisim import _pq_pmax_pcc_for_req

    imports_only = {'p_mw': [-7.54, -6.39, -5.24]}
    assert _pq_pmax_pcc_for_req(imports_only) is None          # held at Pn instead
    mixed = {'p_max_mw': [-2.1, 0.4, 0.98], 'p_min_mw': [-2.1, 0.4, 0.97]}
    assert _pq_pmax_pcc_for_req(mixed) == pytest.approx(0.97)
    # Load orientation plots export as negative P.
    assert _pq_pmax_pcc_for_req({'p_mw': [0.0, -1.0, -2.0]}, sign_out=-1.0) == pytest.approx(2.0)


@pytest.mark.parametrize('grid', GRIDS)
def test_drawn_diagram_time_series_matches_spec(client, quiet, grid):
    """
    The time series on the drawn grid - 24 hours of the dialog's default
    profiles, in MW - must give each hour the power flow pandapower gives the
    spec with the same P, Q kept at each element's power factor. Every load and
    generator gets a profile: the dialog left out the radial grid's wind farm,
    drawn as a Wind Turbine, so it ran at its drawn 3 MW all day.
    """
    with open(os.path.join(REFERENCE_DIR, f'{grid}.diagram_timeseries_payload.json'),
              encoding='utf-8') as handle:
        request = json.load(handle)
    with quiet():
        response = client.post('/', json=request)
    assert response.status_code == 200
    result = json.loads(response.get_data(as_text=True))
    assert result['timeseries_converged'] is True
    steps = int(result['time_steps'])
    profiles = {p['display_name']: p['values'] for p in result['profiles_used'].values()}
    spec = load_spec(grid)
    expected = {str(e.get('name') or e['id'])
                for key in ('loads', 'generators', 'static_generators') for e in spec[key]}
    assert set(profiles) == expected
    assert all(len(v) == steps for v in profiles.values())

    vm = {(b['time_step'], b['name']): b['vm_pu'] for b in result['busbars']}
    loading = {(l['time_step'], l['name']): l['loading_percent'] for l in result['lines']}
    net, _ = sld.build_network(spec)
    base = {table: net[table].copy() for table in ('load', 'sgen', 'gen')}
    differ = []
    for t in range(steps):
        for table in ('load', 'sgen', 'gen'):
            for idx in net[table].index:
                name = net[table].at[idx, 'name']
                if name not in profiles:
                    continue
                p0, p = base[table].at[idx, 'p_mw'], profiles[name][t]
                net[table].at[idx, 'p_mw'] = p
                if 'q_mvar' in net[table] and p0:
                    net[table].at[idx, 'q_mvar'] = base[table].at[idx, 'q_mvar'] * p / p0
        pp.runpp(net, algorithm='nr', calculate_voltage_angles='auto')
        for idx in net.bus.index:
            name = net.bus.at[idx, 'name']
            if abs(vm[(t, name)] - net.res_bus.at[idx, 'vm_pu']) > DRAWN_TOL:
                differ.append(f'hour {t} bus {name}: {vm[(t, name)]} drawn, '
                              f'{net.res_bus.at[idx, "vm_pu"]} spec')
        for idx in net.line.index:
            name = net.line.at[idx, 'name']
            if abs(loading[(t, name)] - net.res_line.at[idx, 'loading_percent']) > 1e-4:
                differ.append(f'hour {t} line {name}: {loading[(t, name)]} % drawn')
    assert not differ, '\n  '.join(differ[:10])


@pytest.mark.parametrize('grid', GRIDS)
def test_drawn_diagram_contingency_analysis_matches_spec(client, quiet, grid):
    """
    N-1 on the drawn grid, every element: each outage must give pandapower's
    voltages for the spec with that element out, a bus it cuts off must count
    as lost supply - an islanded bus has no voltage, which no limit caught -
    and the transmission grid's three-winding main transformer is an outage
    too. On the radial grid nearly every outage cuts something off.
    """
    import math

    with open(os.path.join(REFERENCE_DIR, f'{grid}.diagram_contingency_payload.json'),
              encoding='utf-8') as handle:
        request = json.load(handle)
    with quiet():
        response = client.post('/', json=request)
    result = json.loads(response.get_data(as_text=True))
    spec = load_spec(grid)
    out_of = {'line': ('line',), 'transformer': ('trafo', 'trafo3w'), 'generator': ('gen',)}

    outages = []
    for case in result['contingency_results']:
        assert case['converged'], case['description']
        kind, name = case['description'].replace('Outage of ', '').split(' ', 1)
        net, _ = sld.build_network(spec)
        (table, idx), = [(t, i) for t in out_of[kind] for i in net[t].index if net[t].at[i, 'name'] == name]
        outages.append((table, name))
        net[table].at[idx, 'in_service'] = False
        pp.runpp(net, algorithm='nr', calculate_voltage_angles=True)
        drawn = {b['name']: b['vm_pu'] for b in case['bus_results']}
        cut_off = set()
        for i in net.bus.index:
            bus, want = net.bus.at[i, 'name'], net.res_bus.at[i, 'vm_pu']
            if math.isnan(want):
                cut_off.add(f'Bus_{bus}')
            else:
                assert drawn[bus] == pytest.approx(want, abs=DRAWN_TOL), (name, bus)
        flagged = {v['element'] for v in case['violations'] if v['type'] == 'supply'}
        assert flagged == cut_off, (name, flagged, cut_off)

    expected = ([('line', str(l.get('name') or l['id'])) for l in spec['lines']]
                + [('trafo', str(t.get('name') or t['id'])) for t in spec['transformers']]
                + [('trafo3w', str(t.get('name') or t['id'])) for t in spec.get('three_winding_transformers', [])]
                + [('gen', str(g.get('name') or g['id'])) for g in spec['generators']])
    assert sorted(outages) == sorted(expected)
    assert any(cut for case in result['contingency_results']
               for cut in case['violations'] if cut['type'] == 'supply'), 'no outage cut a bus off'


def test_site_screening_charges_the_site_only_what_it_adds(client, quiet):
    """
    The transmission grid's Industrial park as a data-centre site at 2, 5 and
    10 MW. Losing the wind farm cable cuts off the wind farm's bus whatever
    the site - that flagged every candidate as needing an upgrade. Only what
    the site adds counts now (and its own loss of supply). The three-winding
    main transformer is an N-1 outage too: at 10 MW losing it overloads two
    ring cables, as pandapower gives.
    """
    import math

    with open(os.path.join(REFERENCE_DIR, 'reference_transmission.diagram_site_screening_payload.json'),
              encoding='utf-8') as handle:
        request = json.load(handle)
    with quiet():
        response = client.post('/', json=request)
    lines = [json.loads(line) for line in response.get_data(as_text=True).splitlines() if line.strip()]
    result = next(line for line in lines if line.get('type') == 'result')['data']
    spec = load_spec('reference_transmission')
    n_elements = sum(len(spec[k]) for k in ('lines', 'transformers', 'three_winding_transformers', 'generators'))
    assert result['summary']['n1_cases'] == n_elements

    rows = {r['requested_mw']: r for r in result['screening_results']}
    for mw in (2.0, 5.0):
        assert rows[mw]['worst_n1_violations'] == 0, rows[mw]['n1_violation_details']
    big = rows[10.0]
    assert big['n1_worst_case'] == 'Trafo_Main transformer 110/20/10'
    drawn = {v['name']: float(v['text'].split('%')[0]) for v in big['n1_violation_details']}

    net, _ = sld.build_network(spec)
    site = net.load.index[net.load.name == 'Industrial park'][0]
    net.load.at[site, 'p_mw'] = 10.0
    net.load.at[site, 'q_mvar'] = 10.0 * math.tan(math.acos(0.95))
    net.trafo3w['in_service'] = False
    pp.runpp(net, algorithm='nr', calculate_voltage_angles=True)
    over = {net.line.at[i, 'name']: net.res_line.at[i, 'loading_percent']
            for i in net.line.index if net.res_line.at[i, 'loading_percent'] > 100}
    assert set(drawn) == set(over) and over
    for name, pct in over.items():
        assert drawn[name] == pytest.approx(pct, abs=0.05)


# --- arc flash -------------------------------------------------------------------

def _arc_flash(client, quiet, mode, grid):
    request = _study_request(grid, {
        'typ': 'ArcFlashPandaPower Parameters', 'electrode_config': 'VCB', 'equipment_mode': mode,
        'working_distance_mm': '455', 'conductor_gap_mm': '25', 'enclosure_height_mm': '508',
        'enclosure_width_mm': '508', 'enclosure_depth_mm': '508',
        'clearing_time_s': '0.2', 'clearing_time_min_s': '0.2'})
    result = _post_study(client, quiet, request)
    rows = {row['name']: row for row in result['arc_flash']}
    with open(os.path.join(REFERENCE_DIR, f'{grid}.spec.json'), encoding='utf-8') as handle:
        spec = json.load(handle)
    assert set(rows) == {_bus_label(spec, b['id']) for b in spec['buses']}, 'every bus is studied'
    return rows


def _lee(row, distance_mm, t_s=0.2):
    """IEEE 1584-2002 eq. 8, E in J/cm²; boundary at 5.0 J/cm²."""
    k = 2.142e6 * row['vn_kv'] * row['ikss_ka'] * t_s
    return k / distance_mm ** 2 / 4.184, math.sqrt(k / 5.0)


def _ieee1584(row, gap, distance, enclosure, t_s=0.2):
    from arcflash.ieee_1584.calculation import Calculation
    from arcflash.ieee_1584.cubicle import Cubicle
    from arcflash.ieee_1584.units import kA, kV, mm, ms, cal_per_sq_cm
    cubicle = Cubicle(V_oc=row['vn_kv'] * kV, EC='VCB', G=gap * mm, D=distance * mm,
                      height=enclosure[0] * mm, width=enclosure[1] * mm, depth=enclosure[2] * mm)
    energies = []
    for variation in ('full', 'reduced'):
        calc = Calculation(cubicle, row['ikss_ka'] * kA, variation)
        calc.calculate_I_arc()
        calc.calculate_E_AFB(t_s * 1000 * ms)
        energies.append(float(calc.E.to(cal_per_sq_cm).magnitude))
    return max(energies)


@pytest.mark.parametrize('grid', GRIDS)
def test_arc_flash_by_voltage_class(client, quiet, grid):
    """
    Every bus was studied as an LV panel (25 mm gap at 455 mm), 10 kV
    switchgear included, and Ralph Lee's J/cm² was reported as cal/cm² with
    its boundary solved for 1.2 J/cm²: 548 cal/cm² and 9.7 m on the 20 kV
    busbar instead of 33 cal/cm² and 4.8 m. Each bus now gets the typical
    equipment of its voltage class (IEEE 1584-2018 Table 8).
    """
    rows = _arc_flash(client, quiet, 'by_voltage', grid)
    assert {r['method'] for n, r in rows.items() if r['vn_kv'] > 15} == {'RalphLee'}
    for name, row in rows.items():
        if row['vn_kv'] > 15:
            energy, boundary = _lee(row, 910)
            assert row['working_distance_mm'] == 910 and row['conductor_gap_mm'] is None, name
        else:
            gap, distance, enclosure, equipment = (
                (32, 610, (508, 508, 508), 'LV switchgear') if row['vn_kv'] <= 0.6
                else (152, 910, (1143, 762, 762), '15 kV switchgear'))
            assert row['equipment_class'] == equipment, name
            assert (row['conductor_gap_mm'], row['working_distance_mm']) == (gap, distance), name
            energy, boundary = _ieee1584(row, gap, distance, enclosure), None
            assert row['method'] == 'IEEE1584-2018', name
        assert row['incident_energy_cal_cm2'] == pytest.approx(energy, rel=1e-6), name
        if boundary is not None:
            assert row['arc_flash_boundary_mm'] == pytest.approx(boundary, rel=1e-6), name
    if grid == 'reference_transmission':
        assert rows['20 kV busbar 1']['incident_energy_cal_cm2'] == pytest.approx(32.7, abs=0.1)
        assert rows['20 kV busbar 1']['ppe_category'] == '4'
        assert rows['10 kV station supply']['equipment_class'] == '15 kV switchgear'
    else:
        assert rows['20 kV substation']['incident_energy_cal_cm2'] == pytest.approx(24.93, abs=0.01)
        assert rows['20 kV substation']['ppe_category'] == '3'
        assert rows['LV network A']['incident_energy_cal_cm2'] == pytest.approx(6.50, abs=0.01)
        assert rows['110 kV supply']['ppe_category'] == 'Dangerous'


@pytest.mark.parametrize('grid', GRIDS)
def test_arc_flash_uniform_equipment(client, quiet, grid):
    """The values entered apply to every bus when asked for; Lee in cal/cm² still."""
    rows = _arc_flash(client, quiet, 'uniform', grid)
    for name, row in rows.items():
        assert row['working_distance_mm'] == 455, name
        if row['vn_kv'] > 15:
            energy, boundary = _lee(row, 455)
            assert row['incident_energy_cal_cm2'] == pytest.approx(energy, rel=1e-6), name
            assert row['arc_flash_boundary_mm'] == pytest.approx(boundary, rel=1e-6), name
        else:
            assert row['incident_energy_cal_cm2'] == pytest.approx(
                _ieee1584(row, 25, 455, (508, 508, 508)), rel=1e-6), name
    if grid == 'reference_transmission':
        assert rows['20 kV busbar 1']['incident_energy_cal_cm2'] == pytest.approx(130.9, abs=0.1)
