"""
The critical clearing time and contingency batch studies: the search against the equal-area
criterion on the classical machines the transient stability test uses, and the batch on the
radial reference grid. The hand figures are that test's.
"""
import json

import pytest

import test_reference_grids as G

CLASSICAL = dict(dyn_machine_model='GENCLS', dyn_exciter_model='NONE', dyn_governor_model='NONE')


def _request(grid, study, **params):
    request = G._study_request(grid, {'typ': f'{study} Parameters', **G.ANDES_PARAMS, **{k: str(v) for k, v in params.items()}})
    next(v for v in request.values() if str(v.get('typ', '')).startswith('Generator')).update(CLASSICAL)
    return request


def _bus(request, label):
    return next(v['name'] for v in request.values() if isinstance(v, dict) and str(v.get('typ', '')).startswith('Bus') and (v.get('userFriendlyName') or v['name']) == label)


@pytest.mark.parametrize('grid, t_cr_hand', [('reference_transmission', 0.463), ('reference_radial', 0.389)])
def test_the_search_finds_the_equal_area_time(client, quiet, grid, t_cr_hand):
    spec = G.load_spec(grid)
    gen = spec['generators'][0]
    request = _request(grid, 'CriticalClearingAndes', cct_tolerance_s=0.01, cct_watch_s=3)
    request['0']['fault_bus'] = _bus(request, G._bus_label(spec, gen['bus']))
    result = G._post_study(client, quiet, request)['critical_clearing']
    assert result['cct_s'] == pytest.approx(t_cr_hand, abs=0.02)
    assert result['first_failure_s'] - result['cct_s'] <= 0.01
    # The time it returns held in a run; the next one up did not.
    by_time = {r['duration_s']: r['holds'] for r in result['runs']}
    assert by_time[result['cct_s']] is True and by_time[result['first_failure_s']] is False


def test_a_case_that_always_holds_says_so(client, quiet):
    spec = G.load_spec('reference_radial')
    gen = spec['generators'][0]
    request = _request('reference_radial', 'CriticalClearingAndes', cct_max_s=0.1, cct_ladder=2, cct_watch_s=2)
    request['0']['fault_bus'] = _bus(request, G._bus_label(spec, gen['bus']))
    result = G._post_study(client, quiet, request)['critical_clearing']
    assert result['cct_s'] is None and result['at_least_s'] == 0.1


def test_the_search_needs_a_bus(client, quiet):
    with quiet():
        response = client.post('/', json=_request('reference_radial', 'CriticalClearingAndes'))
    assert json.loads(response.get_data(as_text=True))['critical_clearing']['error'] is True


def test_the_batch_runs_its_list_and_puts_the_failures_first(client, quiet):
    spec = G.load_spec('reference_radial')
    gen = spec['generators'][0]
    request = _request('reference_radial', 'ContingencyBatchAndes', watch_s=3)
    bus = _bus(request, G._bus_label(spec, gen['bus']))
    gen_name = next(v['name'] for v in request.values() if str(v.get('typ', '')).startswith('Generator'))
    request['0']['contingencies'] = json.dumps([
        {'name': 'Short fault', 'type': 'fault', 'element': bus, 'clear_s': 0.1},
        {'name': 'Long fault', 'type': 'fault', 'element': bus, 'clear_s': 0.8},
        {'name': 'Machine off', 'type': 'gen', 'element': gen_name},
        {'name': 'Nonsense', 'type': 'meteor', 'element': bus}])
    result = G._post_study(client, quiet, request)['contingency_batch']
    rows = {r['name']: r for r in result['results']}
    assert rows['Short fault']['holds'] is True
    assert rows['Long fault']['holds'] is False and rows['Long fault']['lost_synchronism']
    assert rows['Machine off']['holds'] is True
    assert rows['Nonsense']['holds'] is False
    assert result['n_failed'] == 2
    assert [r['holds'] for r in result['results']] == sorted(r['holds'] for r in result['results'])


def test_auto_covers_every_line_generator_and_bus(client, quiet):
    request = _request('reference_radial', 'ContingencyBatchAndes', watch_s=1, clear_s=0.05)
    rows = [v for v in request.values() if isinstance(v, dict)]
    expected = sum(1 for r in rows if str(r.get('typ', '')).startswith(('Line', 'Generator', 'Bus')) and not str(r.get('typ', '')).startswith(('Line DC', 'Busbar DC')))
    result = G._post_study(client, quiet, request)['contingency_batch']
    assert result['n'] == expected
    assert all('holds' in r for r in result['results'])
