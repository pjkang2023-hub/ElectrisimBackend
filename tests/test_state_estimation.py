"""
State estimation study on the drawn reference grids.
"""
import json
import os

import pandapower as pp
import pytest

import electrisim_sld as sld

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE_DIR = os.path.join(HERE, 'reference')
GRIDS = ('reference_radial', 'reference_transmission')


def _study(client, quiet, grid, **params):
    with open(os.path.join(REFERENCE_DIR, f'{grid}.diagram_payload.json'), encoding='utf-8') as handle:
        request = json.load(handle)
    key = next(k for k, v in request.items() if isinstance(v, dict) and 'Parameters' in str(v.get('typ', '')))
    request[key] = {'typ': 'StateEstimationPandaPower Parameters', 'user_email': 'test@example.com', **params}
    with quiet():
        response = client.post('/', json=request)
    assert response.status_code == 200
    return json.loads(response.get_data(as_text=True))


def _load_flow_vm(grid):
    """pandapower's load flow of the reference grid, by bus name."""
    with open(os.path.join(REFERENCE_DIR, f'{grid}.spec.json'), encoding='utf-8') as handle:
        spec = json.load(handle)
    net, _ = sld.build_network(spec)
    pp.runpp(net, calculate_voltage_angles=True)
    return {str(net.bus.at[b, 'name']): float(net.res_bus.at[b, 'vm_pu']) for b in net.bus.index}


@pytest.mark.parametrize('grid', GRIDS)
def test_state_estimation_recovers_the_load_flow_from_exact_meters(client, quiet, grid):
    """
    Meters at every bus voltage, both ends of every branch and every injection,
    reading the load flow without error: the estimate is the load flow's state,
    every measurement is reproduced, and J is nil. The bus injections a meter
    reads exclude the capacitor banks, which are part of the network model.
    """
    result = _study(client, quiet, grid, add_noise=False)
    assert not result.get('error'), result.get('message')
    se = result['state_estimation']
    assert se['summary']['converged'] and se['summary']['objective_j'] < 1e-3
    assert se['summary']['max_vm_error_pu'] < 1e-6

    reference = _load_flow_vm(grid)
    for bus in se['buses']:
        if bus['name'] in reference:
            assert bus['vm_pu'] == pytest.approx(reference[bus['name']], abs=1e-6), bus['name']
    for meas in se['measurements']:
        assert meas['estimated'] == pytest.approx(meas['true_value'], abs=1e-5), meas
        assert meas['status'] == 'ok'


@pytest.mark.parametrize('grid', GRIDS)
def test_state_estimation_with_metering_errors(client, quiet, grid):
    """
    Meters of their accuracy class: J lies under the chi-square threshold for
    its m - n degrees of freedom, no reading is flagged, and the voltages come
    back within the meters' 0.5 %. The states are each bus's voltage and angle,
    less the reference angle - a three-winding transformer's star point too.
    """
    se = _study(client, quiet, grid, seed=7)['state_estimation']
    s = se['summary']
    star_points = {'reference_radial': 0, 'reference_transmission': 1}[grid]
    assert s['state_variables'] == 2 * (len(se['buses']) + star_points) - 1
    assert s['measurements'] == sum(s['measurements_by_type'].values()) == len(se['measurements'])
    assert s['chi2_passed'] and s['objective_j'] <= s['chi2_threshold']
    assert s['bad_data_removed'] == 0 and s['max_normalized_residual'] < s['rn_threshold']
    assert s['max_vm_error_pu'] < 0.005


@pytest.mark.parametrize('grid', GRIDS)
def test_state_estimation_removes_a_gross_error(client, quiet, grid):
    """
    The simulated readings entered as a table with one line flow 20 standard
    deviations off: the chi-square test fails on it, the largest normalized
    residual names it, and once it is removed the estimate passes again. (With
    seed 7 the readings themselves flag nothing: about one in 370 good readings
    has a normalized residual over 3.)
    """
    simulated = _study(client, quiet, grid, seed=7)['state_estimation']
    assert simulated['summary']['bad_data_removed'] == 0
    lines = simulated['measurements_csv'].strip().splitlines()
    target = next(i for i, row in enumerate(lines) if row.startswith('p,line,'))
    cells = lines[target].split(',')
    cells[4] = repr(float(cells[4]) + 20 * float(cells[5]))
    lines[target] = ','.join(cells)
    bad_label = f'P flow, {cells[2]} ({cells[3]})'

    without = _study(client, quiet, grid, measurement_source='entered', pseudo_measurements=False,
                     bad_data=False, measurements_csv='\n'.join(lines))['state_estimation']
    assert without['summary']['chi2_passed'] is False
    worst = max(without['measurements'], key=lambda m: m['normalized_residual'] or 0)
    assert (worst['type'], worst['element_type'], worst['element'], worst['side']) == ('p', 'line', cells[2], cells[3])

    se = _study(client, quiet, grid, measurement_source='entered', pseudo_measurements=False,
                measurements_csv='\n'.join(lines))['state_estimation']
    assert [r['label'] for r in se['removed']] == [bad_label]
    assert se['summary']['chi2_passed'] and se['summary']['bad_data_removed'] == 1
    assert se['summary']['max_vm_error_pu'] < 0.005
    removed = [m for m in se['measurements'] if m['status'] == 'removed']
    assert len(removed) == 1 and removed[0]['residual'] == pytest.approx(20 * float(cells[5]), rel=0.2)


@pytest.mark.parametrize('grid, bus, states', [('reference_radial', 'B1', 15),
                                              ('reference_transmission', 'F2', 23)])
def test_state_estimation_reports_an_unobservable_network(client, quiet, grid, bus, states):
    """
    One voltage reading cannot fix the states: two per bus, a three-winding
    transformer's star point included, less the reference angle.
    """
    result = _study(client, quiet, grid, measurement_source='entered',
                    pseudo_measurements=False, measurements_csv=f'v,bus,{bus},,1.0,0.005')
    assert result.get('error'), result
    assert f'not observable: 1 measurement for {states} state variables' in result['message']


def test_state_estimation_pseudo_measurements_make_it_observable(client, quiet):
    """
    Voltages and the transformer flow alone leave the feeders unobservable;
    the drawn loads and generation as pseudo-measurements of 30 % accuracy
    complete it.
    """
    simulated = _study(client, quiet, 'reference_radial', seed=5)['state_estimation']
    keep = [row for row in simulated['measurements_csv'].strip().splitlines()
            if row.startswith(('type,', 'v,')) or ',trafo,' in row]
    csv_text = '\n'.join(keep)
    assert _study(client, quiet, 'reference_radial', measurement_source='entered',
                  pseudo_measurements=False, measurements_csv=csv_text).get('error')
    se = _study(client, quiet, 'reference_radial', measurement_source='entered',
                measurements_csv=csv_text)['state_estimation']
    assert se['summary']['pseudo_measurements'] > 0
    assert se['summary']['max_vm_error_pu'] < 0.01


def test_state_estimation_entered_table_errors(client, quiet):
    """Rows naming nothing on the diagram are reported and skipped, not fatal."""
    simulated = _study(client, quiet, 'reference_radial', add_noise=False)['state_estimation']
    csv_text = simulated['measurements_csv'] + 'v,bus,No such bus,,1.0,0.005\np,line,LA1,middle,1,0.1\n'
    se = _study(client, quiet, 'reference_radial', measurement_source='entered',
                measurements_csv=csv_text)['state_estimation']
    assert len(se['input_errors']) == 2
    assert 'No such bus' in se['input_errors'][0] and 'side' in se['input_errors'][1]
    assert se['summary']['max_vm_error_pu'] < 1e-6


def test_state_estimation_lav_estimator(client, quiet):
    """The least absolute value estimator, robust to bad data by its own norm."""
    se = _study(client, quiet, 'reference_transmission', estimator='lav', seed=2)['state_estimation']
    assert se['summary']['estimator'] == 'LAV'
    assert se['summary']['max_vm_error_pu'] < 0.005
