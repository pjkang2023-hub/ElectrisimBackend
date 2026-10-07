# -*- coding: utf-8 -*-
"""
The declarative spec behind /build-model and the Electrisim MCP server.

These pin the contract a language model writes against: what defaults fill in,
that every problem is reported at once, that elements are addressed by spec id,
and that results come back under those same ids.
"""

import json

import pytest

import electrisim_sld as sld


def substation():
    """A 110/20 kV substation with two feeders - the shape most prose describes."""
    return {
        'name': 'Test substation',
        'buses': [
            {'id': 'HV', 'vn_kv': 110, 'name': '110 kV bar'},
            {'id': 'MV', 'vn_kv': 20, 'name': '20 kV bar'},
            {'id': 'F1', 'vn_kv': 20},
            {'id': 'F2', 'vn_kv': 20},
        ],
        'external_grids': [{'bus': 'HV', 'vm_pu': 1.02}],
        'transformers': [{'id': 'T1', 'hv_bus': 'HV', 'lv_bus': 'MV', 'sn_mva': 25}],
        'lines': [
            {'id': 'L1', 'from_bus': 'MV', 'to_bus': 'F1', 'length_km': 1.2,
             'name': 'Feeder cable 1'},
            {'id': 'L2', 'from_bus': 'MV', 'to_bus': 'F2', 'length_km': 0.8},
        ],
        'loads': [{'id': 'Ld1', 'bus': 'F1', 'p_mw': 2.0}],
        'static_generators': [{'id': 'PV', 'bus': 'F2', 'p_mw': 1.5}],
        'switches': [{'bus': 'MV', 'element': 'L1', 'et': 'line'}],
    }


def problems_of(spec):
    with pytest.raises(sld.SpecError) as err:
        sld.build_network(spec)
    return err.value.problems


# --- defaults --------------------------------------------------------------

def test_minimal_statement_builds_with_defaults():
    net, report = sld.build_network(substation())
    assert report['counts'] == {
        'bus': 4, 'line': 2, 'trafo': 1, 'trafo3w': 0, 'load': 1, 'gen': 0, 'sgen': 1,
        'ext_grid': 1, 'shunt': 0, 'storage': 0, 'motor': 0, 'switch': 1,
    }
    assert report['warnings'] == []

    trafo = net.trafo.iloc[0]
    # Only sn_mva was given: vk by rating class, ratios from the buses.
    assert trafo['vk_percent'] == 12.0
    assert (trafo['vn_hv_kv'], trafo['vn_lv_kv']) == (110.0, 20.0)
    # No std_type given: chosen from the 20 kV bus voltage.
    assert set(net.line['std_type']) == {'NA2XS2Y 1x240 RM/25 12/20 kV'}
    # Unstated reactive load gets a typical power factor, not zero.
    assert net.load.iloc[0]['q_mvar'] == pytest.approx(0.66)


def test_explicit_line_impedance_bypasses_the_type_library():
    spec = substation()
    spec['lines'][1] = {'id': 'L2', 'from_bus': 'MV', 'to_bus': 'F2', 'length_km': 2,
                        'r_ohm_per_km': 0.2, 'x_ohm_per_km': 0.35}
    net, _ = sld.build_network(spec)
    row = net.line.iloc[1]
    assert (row['r_ohm_per_km'], row['x_ohm_per_km']) == (0.2, 0.35)


# --- validation ------------------------------------------------------------

def test_every_problem_is_reported_together():
    problems = problems_of({
        'buses': [{'id': 'B1', 'vn_kv': 110}, {'id': 'B2'}],
        'lines': [{'from_bus': 'B1', 'to_bus': 'NOPE'}],
        'transformers': [{'hv_bus': 'B1', 'lv_bus': 'B1', 'sn_mva': 'big'}],
        'nonsense': 1,
    })
    joined = '\n'.join(problems)
    assert len(problems) == 4, joined
    assert "unknown top-level key 'nonsense'" in joined
    assert 'buses[1] (B2): vn_kv is required' in joined
    assert "sn_mva must be a number, got 'big'" in joined
    assert "to_bus='NOPE' is not a bus id. Defined buses: B1" in joined


def test_line_across_voltage_levels_is_refused():
    spec = substation()
    spec['lines'].append({'id': 'Bad', 'from_bus': 'HV', 'to_bus': 'F1'})
    (problem,) = problems_of(spec)
    assert 'use a transformer instead' in problem


def test_duplicate_ids_are_refused():
    spec = substation()
    spec['loads'].append({'id': 'T1', 'bus': 'F2', 'p_mw': 1})
    (problem,) = problems_of(spec)
    assert "duplicate id 'T1'" in problem


def test_unknown_layout_is_refused():
    spec = substation()
    spec['layout'] = 'sideways'
    (problem,) = problems_of(spec)
    assert 'transmission, radial, auto' in problem


def test_non_finite_numbers_are_refused():
    spec = substation()
    spec['loads'][0]['p_mw'] = float('inf')
    (problem,) = problems_of(spec)
    assert 'must be finite' in problem


# --- switches --------------------------------------------------------------

def test_switch_finds_its_line_by_id_not_by_display_name():
    # L1 is named "Feeder cable 1". Resolving by name used to miss it.
    net, _ = sld.build_network(substation())
    sw = net.switch.iloc[0]
    assert net.line.at[int(sw['element']), 'name'] == 'Feeder cable 1'


def test_switch_must_sit_at_a_terminal_of_its_element():
    spec = substation()
    spec['switches'] = [{'bus': 'F2', 'element': 'L1', 'et': 'line'}]
    (problem,) = problems_of(spec)
    assert "bus 'F2' is not a terminal of line 'L1'" in problem
    assert 'MV, F1' in problem


def test_bus_coupler_counts_as_a_connection():
    spec = substation()
    spec['buses'].append({'id': 'MV2', 'vn_kv': 20})
    spec['switches'].append({'bus': 'MV', 'element': 'MV2', 'et': 'bus'})
    _, report = sld.build_network(spec)
    assert not any('coupler' in w for w in report['warnings']), report['warnings']


def test_isolated_bus_is_named_by_its_id():
    spec = substation()
    spec['buses'].append({'id': 'Lonely', 'vn_kv': 20, 'name': 'Spare bar'})
    _, report = sld.build_network(spec)
    assert 'buses with no line, transformer or coupler: Lonely' in report['warnings']


def test_missing_slack_is_warned_about():
    spec = substation()
    del spec['external_grids']
    _, report = sld.build_network(spec)
    assert any('no external grid and no slack generator' in w for w in report['warnings'])


# --- power flow ------------------------------------------------------------

def test_power_flow_reports_under_spec_ids():
    net, _ = sld.build_network(substation())
    result = sld.solve(net)
    assert result['converged'] is True
    assert [b['id'] for b in result['buses']] == ['HV', 'MV', 'F1', 'F2']
    assert [line['id'] for line in result['lines']] == ['L1', 'L2']
    assert result['transformers'][0]['id'] == 'T1'
    assert result['buses'][0]['vm_pu'] == pytest.approx(1.02)
    assert result['summary']['voltage_issues'] == []
    assert result['summary']['overloads'] == []


def test_weak_feeder_flags_undervoltage_and_overload():
    net, _ = sld.build_network({
        'buses': [{'id': 'S', 'vn_kv': 0.4}, {'id': 'E', 'vn_kv': 0.4}],
        'external_grids': [{'bus': 'S'}],
        'lines': [{'id': 'Long', 'from_bus': 'S', 'to_bus': 'E', 'length_km': 0.6}],
        'loads': [{'bus': 'E', 'p_mw': 0.15}],
    })
    summary = sld.solve(net)['summary']
    assert [(v['id'], v['issue']) for v in summary['voltage_issues']] == [('E', 'undervoltage')]
    assert [(o['id'], o['kind']) for o in summary['overloads']] == [('Long', 'line')]


def test_no_slack_reports_instead_of_raising():
    spec = substation()
    del spec['external_grids']
    net, _ = sld.build_network(spec)
    result = sld.solve(net)
    assert result == {'converged': False, 'hint': result['hint']}
    assert 'reference bus' in result['hint']


def test_divergent_case_reports_instead_of_raising():
    net, _ = sld.build_network({
        'buses': [{'id': 'S', 'vn_kv': 0.4}, {'id': 'E', 'vn_kv': 0.4}],
        'external_grids': [{'bus': 'S'}],
        'lines': [{'id': 'L', 'from_bus': 'S', 'to_bus': 'E', 'length_km': 0.6}],
        'loads': [{'bus': 'E', 'p_mw': 50}],
    })
    result = sld.solve(net)
    assert result['converged'] is False
    assert 'did not converge' in result['hint']


# --- three-winding transformers --------------------------------------------

def tertiary():
    """110/20/10 kV: a three-winding unit with a feeder on each lower winding."""
    return {
        'buses': [
            {'id': 'HV', 'vn_kv': 110}, {'id': 'MV', 'vn_kv': 20}, {'id': 'LV', 'vn_kv': 10},
            {'id': 'F1', 'vn_kv': 20}, {'id': 'F2', 'vn_kv': 10},
        ],
        'external_grids': [{'bus': 'HV'}],
        'three_winding_transformers': [
            {'id': 'T3', 'hv_bus': 'HV', 'mv_bus': 'MV', 'lv_bus': 'LV', 'sn_hv_mva': 40}],
        'lines': [{'id': 'L1', 'from_bus': 'MV', 'to_bus': 'F1', 'length_km': 3},
                  {'id': 'L2', 'from_bus': 'LV', 'to_bus': 'F2', 'length_km': 2}],
        'loads': [{'bus': 'F1', 'p_mw': 12}, {'bus': 'F2', 'p_mw': 5}],
    }


def test_three_winding_defaults_follow_the_pair_ratings():
    net, report = sld.build_network(tertiary())
    t = net.trafo3w.iloc[0]
    assert report['counts']['trafo3w'] == 1 and report['warnings'] == []
    # Tertiary at a third of the HV rating; ratios from the three buses.
    assert (t.sn_hv_mva, t.sn_mv_mva, t.sn_lv_mva) == (40.0, 40.0, pytest.approx(13.333))
    assert (t.vn_hv_kv, t.vn_mv_kv, t.vn_lv_kv) == (110.0, 20.0, 10.0)
    # Each pair defaults by the smaller rating of the pair, as pandapower refers it.
    assert (t.vk_hv_percent, t.vk_mv_percent, t.vk_lv_percent) == (12.0, 12.0, 12.0)


def test_three_winding_pair_names_map_onto_pandapowers():
    spec = tertiary()
    spec['three_winding_transformers'][0].update(
        vk_hv_mv_percent=11, vk_mv_lv_percent=7, vk_hv_lv_percent=18)
    net, _ = sld.build_network(spec)
    t = net.trafo3w.iloc[0]
    # pandapower: vk_hv = HV-MV, vk_mv = MV-LV, vk_lv = HV-LV.
    assert (t.vk_hv_percent, t.vk_mv_percent, t.vk_lv_percent) == (11.0, 7.0, 18.0)


@pytest.mark.parametrize('field, hint', [
    ('vk_mv_percent', "use vk_mv_lv_percent, not pandapower's vk_mv_percent (which means MV-LV)"),
    ('vk_lv_percent', "use vk_hv_lv_percent, not pandapower's vk_lv_percent (which means HV-LV)"),
    ('shift_lv_degree', 'shift_lv_degree is not supported'),
])
def test_three_winding_names_that_would_be_ignored_are_refused(field, hint):
    spec = tertiary()
    spec['three_winding_transformers'][0][field] = 8
    (problem,) = problems_of(spec)
    assert hint in problem


def test_three_winding_needs_three_different_buses():
    spec = tertiary()
    spec['three_winding_transformers'][0]['lv_bus'] = 'MV'
    (problem,) = problems_of(spec)
    assert 'must be three different buses' in problem


def test_three_winding_swapped_voltages_are_warned_about():
    spec = tertiary()
    spec['three_winding_transformers'][0].update(mv_bus='LV', lv_bus='MV')
    _, report = sld.build_network(spec)
    assert any('not in HV >= MV >= LV order' in w for w in report['warnings'])


def test_three_winding_switch_must_sit_at_one_of_its_windings():
    spec = tertiary()
    spec['switches'] = [{'bus': 'MV', 'element': 'T3', 'et': 'three_winding_transformer'}]
    net, _ = sld.build_network(spec)
    assert net.switch.iloc[0]['et'] == 't3'

    spec['switches'] = [{'bus': 'F1', 'element': 'T3', 'et': 't3'}]
    (problem,) = problems_of(spec)
    assert "bus 'F1' is not a terminal of three-winding transformer 'T3'" in problem
    assert 'HV, MV, LV' in problem


def test_three_winding_power_flow_balances_under_its_id():
    net, _ = sld.build_network(tertiary())
    result = sld.solve(net)
    (t3,) = result['three_winding_transformers']
    assert t3['id'] == 'T3'
    # What enters on HV leaves on MV and LV, less the transformer's losses.
    assert t3['p_hv_mw'] == pytest.approx(-(t3['p_mv_mw'] + t3['p_lv_mw']) + t3['losses_mw'],
                                          abs=1e-3)
    assert 0 < t3['loading_percent'] < 100



# --- two-winding transformers: tap changer and neutral resistor ---------------

def test_a_tap_and_a_neutral_resistor_reach_the_transformer():
    """
    An off-load tap at -2.5 % on the HV winding gives the LV what a 2.5 %
    lower HV rating would; its neutral resistor is kept for earth faults.
    Without tap fields a transformer has no tap changer.
    """
    spec = substation()
    spec['transformers'][0].update(tap_pos=-1, rn_ohm=20.0)
    net, _ = sld.build_network(spec)
    t = net.trafo.loc[0]
    assert (t.tap_side, t.tap_neutral, t.tap_min, t.tap_max, t.tap_step_percent, t.tap_pos) == ('hv', 0, -2, 2, 2.5, -1)
    assert t.rn_ohm == 20.0
    tapped = {b['id']: b['vm_pu'] for b in sld.solve(net)['buses']}
    spec = substation()
    spec['transformers'][0]['vn_hv_kv'] = 110 * 0.975
    rated = {b['id']: b['vm_pu'] for b in sld.solve(sld.build_network(spec)[0])['buses']}
    assert tapped['MV'] == pytest.approx(rated['MV'], abs=1e-9) and tapped['MV'] > 1.02
    assert pd_isna_or_none(sld.build_network(substation())[0].trafo.at[0, 'tap_pos'])


def pd_isna_or_none(value):
    return value is None or value != value


def test_a_generators_dynamics_reach_the_drawing(client):
    """
    A generator's Dynamics tab, given in the spec by its own field names,
    is kept on its row and handed to the canvas with its short-circuit
    data: the drawing kept the defaults, and the transient-stability and
    EMT studies each took their own.
    """
    spec = substation()
    dyn = {'dyn_machine_model': 'genrou', 'dyn_H': 3.5, 'dyn_governor_model': 'GAST', 'dyn_gov_R': 0.04,
           'dyn_exciter_model': 'SEXS'}
    spec['generators'] = [{'id': 'G1', 'bus': 'F2', 'p_mw': 1.0, 'name': 'Engine', **dyn}]
    net, _ = sld.build_network(spec)
    assert {k: net.gen.at[0, k] for k in dyn} == {**dyn, 'dyn_machine_model': 'GENROU'}
    body = client.post('/build-model', json={'spec': spec}).get_json()
    sidecar = json.loads(json.loads(body['model'])['_object']['electrisim_import_sidecar']['_object'])
    assert {k: sidecar['gen']['Engine'][k] for k in dyn} == {**dyn, 'dyn_machine_model': 'GENROU'}


@pytest.mark.parametrize('fields, problem', [
    ({'dyn_governor_model': 'GAS'}, "dyn_governor_model='GAS' must be one of NONE, TGOV1"),
    ({'dyn_H': 'heavy'}, 'dyn_H'),
    ({'dyn_inertia': 3.5}, "unknown dynamics field 'dyn_inertia'"),
])
def test_dynamics_problems_are_named(fields, problem):
    spec = substation()
    spec['generators'] = [{'id': 'G1', 'bus': 'F2', 'p_mw': 1.0, **fields}]
    assert any(problem in p for p in problems_of(spec)), problems_of(spec)


@pytest.mark.parametrize('fields, problem', [
    ({'tap_pos': 3}, 'tap_pos=3 is outside tap_min..tap_max (-2..2)'),
    ({'tap_side': 'mv', 'tap_pos': 0}, "tap_side='mv' must be 'hv' or 'lv'"),
])
def test_tap_problems_are_named(fields, problem):
    spec = substation()
    spec['transformers'][0].update(fields)
    assert any(problem in p for p in problems_of(spec)), problems_of(spec)


# --- the HTTP endpoint -----------------------------------------------------

#: Tables insertComponentsForData JSON.parse()s unconditionally. A model missing
#: any of these fails in the browser, not here - so check them here.
REQUIRED_TABLES = {
    'bus', 'line', 'trafo', 'trafo3w', 'gen', 'sgen', 'asymmetric_sgen', 'shunt',
    'load', 'asymmetric_load', 'impedance', 'ward', 'xward', 'motor', 'storage',
    'svc', 'tcsc', 'dcline',
}


@pytest.fixture(scope='module')
def client():
    import app
    return app.app.test_client()


def test_endpoint_returns_a_drawable_model(client):
    resp = client.post('/build-model', json={'spec': substation()})
    assert resp.status_code == 200
    body = resp.get_json()
    tables = json.loads(body['model'])['_object']
    missing = REQUIRED_TABLES - set(tables)
    assert not missing, f'model lacks tables the canvas requires: {sorted(missing)}'
    assert len(json.loads(tables['bus']['_object'])['data']) == 4
    assert 'power_flow' not in body


#: The positions insertComponentsForData destructures transformer rows into
#: (frontend supportingFunctions.js). The canvas reads by position, so a column
#: pandapower adds in the middle shifts every later field - three-winding rows
#: carried 40 fields against the importer's 29, and every tap setting landed one
#: place late.
IMPORTER_TRAFO = [
    'name', 'std_type', 'hv_bus', 'lv_bus', 'sn_mva', 'vn_hv_kv', 'vn_lv_kv', 'vk_percent',
    'vkr_percent', 'pfe_kw', 'i0_percent', 'shift_degree', 'tap_side', 'tap_neutral', 'tap_min',
    'tap_max', 'tap_step_percent', 'tap_step_degree', 'tap_pos', 'tap_phase_shifter', 'parallel',
    'df', 'in_service',
]
IMPORTER_TRAFO3W = [
    'name', 'std_type', 'hv_bus', 'mv_bus', 'lv_bus', 'sn_hv_mva', 'sn_mv_mva', 'sn_lv_mva',
    'vn_hv_kv', 'vn_mv_kv', 'vn_lv_kv', 'vk_hv_percent', 'vk_mv_percent', 'vk_lv_percent',
    'vkr_hv_percent', 'vkr_mv_percent', 'vkr_lv_percent', 'pfe_kw', 'i0_percent',
    'shift_mv_degree', 'tap_side', 'tap_neutral', 'tap_min', 'tap_max', 'tap_step_percent',
    'tap_step_degree', 'tap_pos', 'tap_at_star_point', 'in_service',
]
#: Static generators came through in pandapower's order, with min_q_mvar and
#: max_q_mvar where the importer reads sn_mva and scaling: every imported unit
#: drew with scaling "null" and type "1". Shunts read in_service from
#: pandapower's characteristic-table column.
IMPORTER_SGEN = [
    'name', 'bus', 'p_mw', 'q_mvar', 'sn_mva', 'scaling', 'in_service', 'type', 'current_source',
]
IMPORTER_SHUNT = ['bus', 'name', 'q_mvar', 'p_mw', 'vn_kv', 'step', 'max_step', 'in_service']
IMPORTER_STORAGE = [
    'name', 'bus', 'p_mw', 'q_mvar', 'sn_mva', 'soc_percent', 'min_e_mwh', 'max_e_mwh',
    'scaling', 'in_service', 'type',
]


def rows_as_the_importer_reads(model, table, positions):
    rows = json.loads(json.loads(model)['_object'][table]['_object'])['data']
    for row in rows:
        assert len(row) == len(positions), f'{table} row has {len(row)} fields, importer reads {len(positions)}'
    return [dict(zip(positions, row)) for row in rows]


def test_transformer_rows_line_up_with_the_importer(client):
    spec = tertiary()
    spec['buses'].append({'id': 'X', 'vn_kv': 20})
    spec['transformers'] = [{'id': 'T2', 'hv_bus': 'HV', 'lv_bus': 'X', 'sn_mva': 25}]
    model = client.post('/build-model', json={'spec': spec}).get_json()['model']

    (t2,) = rows_as_the_importer_reads(model, 'trafo', IMPORTER_TRAFO)
    assert (t2['name'], t2['vn_hv_kv'], t2['vk_percent'], t2['in_service']) == ('T2', 110.0, 12.0, True)

    (t3,) = rows_as_the_importer_reads(model, 'trafo3w', IMPORTER_TRAFO3W)
    assert (t3['name'], t3['hv_bus'], t3['mv_bus'], t3['lv_bus']) == ('T3', 0, 1, 2)
    assert (t3['vn_hv_kv'], t3['vn_mv_kv'], t3['vn_lv_kv']) == (110.0, 20.0, 10.0)
    assert t3['shift_mv_degree'] == 0.0
    assert t3['in_service'] is True


def test_injection_rows_line_up_with_the_importer(client):
    spec = substation()
    spec['shunts'] = [{'id': 'CAP', 'bus': 'F1', 'q_mvar': -1.5}]
    spec['storage'] = [{'id': 'BESS', 'bus': 'F2', 'p_mw': 0.2, 'max_e_mwh': 0.8}]
    model = client.post('/build-model', json={'spec': spec}).get_json()['model']

    (pv,) = rows_as_the_importer_reads(model, 'sgen', IMPORTER_SGEN)
    assert (pv['name'], pv['bus'], pv['p_mw'], pv['q_mvar']) == ('PV', 3, 1.5, 0.0)
    assert (pv['scaling'], pv['in_service'], pv['type']) == (1.0, True, 'wye')

    (cap,) = rows_as_the_importer_reads(model, 'shunt', IMPORTER_SHUNT)
    assert (cap['bus'], cap['name'], cap['q_mvar'], cap['in_service']) == (2, 'CAP', -1.5, True)

    (bess,) = rows_as_the_importer_reads(model, 'storage', IMPORTER_STORAGE)
    assert (bess['name'], bess['bus'], bess['p_mw'], bess['max_e_mwh']) == ('BESS', 3, 0.2, 0.8)
    assert (bess['scaling'], bess['in_service']) == (1.0, True)
    # OpenDSS holds an empty battery idle; the default keeps it dispatchable.
    assert bess['soc_percent'] == 50.0


def test_endpoint_accepts_a_bare_spec(client):
    resp = client.post('/build-model', json=substation())
    assert resp.status_code == 200
    assert resp.get_json()['report']['counts']['bus'] == 4


def test_endpoint_runs_power_flow_on_request(client):
    resp = client.post('/build-model', json={
        'spec': substation(), 'run_power_flow': True, 'include_model': False,
    })
    body = resp.get_json()
    assert 'model' not in body
    assert body['power_flow']['converged'] is True
    assert body['power_flow']['summary']['limits']['vm_min_pu'] == 0.95


def test_endpoint_applies_caller_limits(client):
    resp = client.post('/build-model', json={
        'spec': substation(), 'run_power_flow': True, 'include_model': False,
        'vm_max_pu': 1.01,
    })
    issues = resp.get_json()['power_flow']['summary']['voltage_issues']
    # The grid is held at 1.02 pu, so a 1.01 ceiling flags its bus.
    assert {'id': 'HV', 'vm_pu': 1.02, 'issue': 'overvoltage'} in issues


def test_endpoint_returns_every_problem(client):
    resp = client.post('/build-model', json={'spec': {'buses': [{'id': 'B1'}]}})
    assert resp.status_code == 400
    body = resp.get_json()
    assert body['error'] == 'The spec could not be built.'
    assert body['problems'] == ['buses[0] (B1): vn_kv is required - a bus has no default voltage']


def test_endpoint_rejects_a_non_object_body(client):
    resp = client.post('/build-model', data='[1, 2]', content_type='application/json')
    assert resp.status_code == 400
