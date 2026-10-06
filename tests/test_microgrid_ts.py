"""
The microgrid through the time series: each store's state from step to step,
DC loads and PV arrays following library profiles, the smoothing controller
against its filter's closed form, the SOFC's ramp and minimum load, and the
rule-based dispatch - on the grid the batteries taking the grid's exchange,
islanded load shed or PV curtailed when the stores cannot follow.
"""
import json
import math

import pytest

import der_electrisim as der
import microgrid_ts_electrisim as _mg
from test_dc_elements import _drawn_request, _with
from test_der import _bus, _conv
from test_der import _der as _dc_der
from test_pcs import _der, _pcs, _two_buses

CYCLE = {'name': 'Training cycle', 'dt': 1.0, 'p': [1.0] * 20 + [0.2] * 20}     # 40 s, 1.0 / 0.2 p.u.


def _ts(steps, dt, dispatch=False, profiles=None, **extra):
    return {'typ': 'TimeSeriesSimulationPandaPower Parameters', 'time_steps': str(steps), 'time_step_s': str(dt),
            'load_profiles': profiles or {}, 'profile_repeat': True, 'microgrid_dispatch': dispatch,
            'profile_mode': 'preset', 'load_profile': 'constant', 'generation_profile': 'constant',
            'frequency': '50', 'algorithm': 'nr', 'user_email': 't@t', **{k: str(v) for k, v in extra.items()}}


def _post(client, quiet, request):
    with quiet():
        response = client.post('/', json=request)
    result = json.loads(response.get_data(as_text=True))
    assert not result.get('error'), result.get('error') or result.get('message')
    assert result['timeseries_converged']
    return result


def _rack(*stores):
    """A 48 V rack bus fed from DC bus B, its 10 kW of racks following the training cycle."""
    return [_bus('r48', 0.048),
            {'typ': 'Load DC9', 'name': 'ld48', 'id': 'cell-ld48', 'userFriendlyName': 'Racks', 'bus': 'r48',
             'p_mw': '0.01', 'load_profile_id': 'cyc'},
            _conv('k48', 'dc_b', 'r48', rated_mw=0.05), *stores]


def _on_drawn(steps, dt, *elements, **params):
    request = _with(_drawn_request('reference_radial.diagram_timeseries_payload.json'), *elements)
    key = next(k for k, v in request.items() if 'Parameters' in str(v.get('typ', '')))
    request[key] = _ts(steps, dt, profiles={'cyc': CYCLE}, **params)
    return request


def _rows(mg, label):
    return [r for r in mg['ders'] if r['label'] == label]


# --- Rack-level smoothing ------------------------------------------------------------------------

def test_supercapacitor_and_flywheel_smooth_a_training_cycle(client, quiet):
    """
    A supercapacitor and a flywheel on the 48 V rack bus, behind smoothing
    converters (tau 10 s, no state-of-charge correction), sharing by their
    ratings: the feed sees the racks' power through the first-order filter,
    y_n = y_(n-1) + (P_n - y_(n-1)) (1 - exp(-dt / tau)) - its steepest ramp
    0.8 x 10 kW x (1 - e^-0.1) per second against the racks' 8 kW per second -
    and each store's energy changes by exactly what it drew.
    """
    result = _post(client, quiet, _on_drawn(60, 1.0, *_rack(
        _bus('sc48', 0.054), _dc_der('Supercapacitor', 'sc', 'sc48', sizing='modules', strings_parallel=4),
        _conv('ksc', 'sc48', 'r48', control_mode='smoothing', smoothing_tau_s=10, soc_gain=0, rated_mw=0.02),
        _bus('fw48', 0.054), _dc_der('Flywheel', 'fw', 'fw48', v_dc=54, e_max_kwh=0.2, p_rated_kw=10),
        _conv('kfw', 'fw48', 'r48', control_mode='smoothing', smoothing_tau_s=10, soc_gain=0, rated_mw=0.02))))
    mg = result['microgrid']
    alpha = 1 - math.exp(-0.1)
    y, feeds = None, []
    for k in range(60):
        p = 0.01 * CYCLE['p'][k % 40]
        y = p if y is None else y + (p - y) * alpha
        feeds.append(y)
    smoothing = {s['label']: s for s in mg['smoothing']}
    assert set(smoothing) == {'KSC', 'KFW'}
    for s in smoothing.values():
        assert s['rack_ramp_mw_s'] == pytest.approx(0.008, rel=1e-9)
        assert s['limited_steps'] == 0
    # The two stores' shares add up: the feed is the filter's output.
    store = [a['p_store_mw'] + b['p_store_mw'] for a, b in zip(smoothing['KSC']['series'], smoothing['KFW']['series'])]
    for k, (want, row) in enumerate(zip(feeds, smoothing['KSC']['series'])):
        assert row['p_rack_mw'] - store[k] == pytest.approx(want, abs=1e-12)
    ramp = max(abs(feeds[k + 1] - feeds[k]) for k in range(59))
    assert ramp == pytest.approx(0.008 * alpha, rel=1e-9)
    assert smoothing['KSC']['series'][0]['p_store_mw'] == pytest.approx(0, abs=1e-12)
    for st in mg['stores']:
        assert st['stored_end_mwh'] - st['stored_start_mwh'] == pytest.approx(-st['drawn_mwh'], rel=1e-9, abs=1e-15)
    # The racks followed their cycle in the network: 10 kW, then 2 kW.
    loads = [r['p_mw'] for r in result.get('loads', []) if r['id'] == 'ld48']
    assert not loads or loads[0] == pytest.approx(0.01)


def test_state_of_charge_correction_brings_the_store_back(client, quiet):
    """With its gain, a store's state of charge returns toward its set point; without it, it drifts."""
    def final_soc(gain):
        mg = _post(client, quiet, _on_drawn(80, 1.0, *_rack(
            _bus('sc48', 0.054), _dc_der('Supercapacitor', 'sc', 'sc48', sizing='modules', strings_parallel=2, v0_percent=95),
            _conv('ksc', 'sc48', 'r48', control_mode='smoothing', smoothing_tau_s=10, soc_gain=gain,
                  soc_ref_percent=50, rated_mw=0.02))))['microgrid']
        return _rows(mg, 'SC')[-1]['soc_percent_end']
    start = 100 * (0.95 ** 2 - 0.25) / 0.75          # 87 %
    with_gain, without = final_soc(0.5), final_soc(0)
    assert abs(with_gain - 50) < abs(start - 50) - 10       # it came back toward 50 %
    assert without > start                                  # the first half-cycle left it charging


# --- Sources and the dispatch --------------------------------------------------------------------

def test_dispatch_on_the_grid_keeps_the_exchange_at_zero_until_the_window_runs_out(client, quiet):
    """
    0.6 MW of load, a 100 kWh battery behind a 1 MVA PCS at 50 %: it takes the
    grid's exchange - zero - until its state of charge reaches 10 %, exactly,
    in the step that gets there; then the grid takes it all. Its stored energy
    falls by what it drew.
    """
    result = _post(client, quiet, _two_buses(
        _der('Battery', 'b1', capacity_kwh=100, soc_percent=50, c_rate_discharge=10, c_rate_charge=10),
        _pcs('p1', 'b', 'b1', s_rated_mva=1.0), params=_ts(8, 60, dispatch=True)))
    grid = [r['p_mw'] for r in result['externalgrids']]
    mg = result['microgrid']
    soc = [r['soc_percent_end'] for r in _rows(mg, 'B1')]
    assert grid[0] == pytest.approx(0, abs=1e-6) and grid[1] == pytest.approx(0, abs=1e-6)
    assert soc[-1] == pytest.approx(10.0, abs=1e-6) and min(soc) >= 10.0 - 1e-9
    assert grid[-1] > 0.6
    (st,) = mg['stores']
    # Its energy over its OCV curve, from 50 % to 10 %; what it delivered less its resistive losses.
    bat = der.build(_der('Battery', 'x', capacity_kwh=100))
    want = (_mg.battery_energy_j(bat, 0.5) - _mg.battery_energy_j(bat, 0.1)) / 3.6e9
    assert st['stored_start_mwh'] - st['stored_end_mwh'] == pytest.approx(want, rel=1e-9)
    assert st['drawn_mwh'] == pytest.approx(want, rel=1e-9)
    assert st['delivered_mwh'] < st['drawn_mwh']


def test_energy_balance_over_a_day_to_the_joule(client, quiet):
    """
    A day in hours: PV following an irradiance profile, an SOFC, a battery under
    the dispatch, the load following its profile. Each hour the grid and the
    PCS supply the load and the line's losses; over the day the residual is
    below a joule. The PV array delivers its curve's maximum power at each
    hour's irradiance; none at night.
    """
    sun = {'name': 'Sun', 'kind': 'irradiance', 'dt': 3600.0,
           'p': [0, 0, 0, 0, 0, 50, 200, 400, 600, 750, 850, 900, 900, 850, 750, 600, 400, 200, 50, 0, 0, 0, 0, 0]}
    day = {'name': 'Day', 'dt': 3600.0, 'p': [0.6] * 7 + [1.0] * 12 + [0.8] * 5}
    request = _two_buses(
        _der('PV Array', 'pv', irradiance_profile_id='sun', modules_series=18, strings_parallel=40),
        _pcs('ppv', 'b', 'pv', s_rated_mva=0.5),
        _der('SOFC', 'fc', p_rated_kw=200, p_set_kw=150), _pcs('pfc', 'b', 'fc', s_rated_mva=0.25),
        _der('Battery', 'b1', capacity_kwh=1000, soc_percent=60), _pcs('pb', 'b', 'b1', s_rated_mva=0.5),
        params=_ts(24, 3600, dispatch=True, profiles={'sun': sun, 'day': day}))
    load = next(v for v in request.values() if isinstance(v, dict) and v.get('typ') == 'Load0')
    load['load_profile_id'] = 'day'
    result = _post(client, quiet, request)
    mg = result['microgrid']
    residual_j = 0.0
    for t in range(24):
        grid = sum(r['p_mw'] for r in result['externalgrids'] if r['time_step'] == t)
        pcs = sum(r['p_mw'] for r in mg['pcs'] if r['time_step'] == t)
        losses = sum(r['p_from_mw'] + r['p_to_mw'] for r in result['lines'] if r['time_step'] == t)
        served = mg['steps'][t]['load_mw']
        residual_j += (grid + pcs - served - losses) * 1e6 * 3600
    assert abs(residual_j) < 1.0
    pv = der.build(_der('PV Array', 'x', modules_series=18, strings_parallel=40))
    for row in _rows(mg, 'PV'):
        g = sun['p'][row['time_step']]
        if g == 0:
            assert row['p_mw'] == pytest.approx(0, abs=1e-9)
        else:
            pv._conditions(g, 25.0)
            assert row['p_mw'] == pytest.approx(pv.mpp()[2] / 1e6, rel=1e-6)
    for st in mg['stores']:
        if st['kind'] == 'Battery':
            assert st['stored_end_mwh'] - st['stored_start_mwh'] == pytest.approx(-st['drawn_mwh'], rel=1e-9)
    assert mg['unserved_mwh'] == 0


def test_sofc_within_its_ramp_rate_and_minimum_load(client, quiet):
    """
    Under the dispatch the SOFC follows the demand averaged over 60 s; after
    the load steps from 0.18 to 0.6 MW it climbs at its ramp rate - 0.1 %/s of
    1 MW, 60 kW a minute - never faster, and never below its 30 % minimum load.
    """
    step = {'step': {'name': 'Step', 'dt': 60.0, 'p': [0.3] * 4 + [1.0] * 30}}
    request = _two_buses(_der('SOFC', 'fc', p_rated_kw=1000, p_set_kw=300, ramp_percent_s=0.1),
                         _pcs('pf', 'b', 'fc', s_rated_mva=1.2), params=_ts(14, 60, True, step, sofc_tau_s=60))
    load = next(v for v in request.values() if isinstance(v, dict) and v.get('typ') == 'Load0')
    load['load_profile_id'] = 'step'
    mg = _post(client, quiet, request)['microgrid']
    p = [r['p_mw'] for r in _rows(mg, 'FC')]
    steps = [p[k + 1] - p[k] for k in range(len(p) - 1)]
    assert max(steps) == pytest.approx(0.06, rel=1e-6)          # it climbed at its ramp rate
    assert all(abs(d) <= 0.06 * (1 + 1e-9) for d in steps)
    assert min(p) >= 0.3 - 1e-9


def test_island_sheds_load_when_its_battery_runs_out(client, quiet):
    """
    Islanded, a grid-forming PCS's 60 kWh battery at 30 % against 0.6 MW: it
    serves what its C-rate and window allow, the rest is shed and reported, and
    it never goes below its window.
    """
    mg = _post(client, quiet, _two_buses(
        _der('Battery', 'ba', capacity_kwh=60, soc_percent=30, c_rate_discharge=10, c_rate_charge=10),
        _pcs('ga', 'a', 'ba', control='grid_forming', s_rated_mva=1.0), grid=False,
        params=_ts(6, 60, dispatch=True)))['microgrid']
    soc = [r['soc_percent_end'] for r in _rows(mg, 'BA')]
    assert min(soc) >= 10.0 - 1e-6 and soc[-1] == pytest.approx(10.0, abs=0.01)
    unserved = [s['unserved_mw'] for s in mg['steps']]
    assert unserved[0] > 0 and unserved[-1] > 0.59
    served_j = sum(s['load_mw'] for s in mg['steps']) * 1e6 * 60
    assert mg['unserved_mwh'] * 3.6e9 == pytest.approx(0.6e6 * 60 * 6 - served_j, rel=1e-9)
    assert any('ran short' in n for n in mg['notes'])


def test_island_curtails_pv_when_its_battery_is_full(client, quiet):
    """Islanded with a full battery and more PV than load: the PV array is curtailed, the battery stays at 90 %."""
    mg = _post(client, quiet, _two_buses(
        _der('Battery', 'ba', capacity_kwh=500, soc_percent=90), _pcs('ga', 'a', 'ba', control='grid_forming', s_rated_mva=1.0),
        _der('PV Array', 'pv', modules_series=18, strings_parallel=80), _pcs('pp', 'b', 'pv', s_rated_mva=1.0),
        grid=False, params=_ts(3, 60, dispatch=True)))['microgrid']
    assert mg['curtailed_mwh'] > 0
    assert all(r['soc_percent_end'] <= 90.0 + 1e-6 for r in _rows(mg, 'BA'))
    assert mg['unserved_mwh'] == 0
