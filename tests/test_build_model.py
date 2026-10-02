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
        'bus': 4, 'line': 2, 'trafo': 1, 'load': 1, 'gen': 0, 'sgen': 1,
        'ext_grid': 1, 'shunt': 0, 'storage': 0, 'switch': 1,
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
