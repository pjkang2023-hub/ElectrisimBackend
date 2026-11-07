"""
The PCS in the ANDES transient stability study: each source behind its PCS
as an ANDES model - a battery or flywheel ESD1, a PV array or an SOFC
system PVD1, a grid-forming PCS a virtual machine (GENCLS) -
starting where the load flow left it and riding through a fault.
"""
import json
import math

import numpy as np
import pytest

import andes_electrisim
from test_dc_elements import _drawn_request, _with
from test_pcs import LF, LINE, LOAD, _bus, _der, _lv, _pcs, _post, _two_buses

TDS = {'typ': 'TransientStabilityAndes Parameters', 'frequency': '50', 'sn_mva': '100', 'user_email': 't@t'}


def _radial(params):
    """The radial grid with, on its LV network A, a battery, a PV array and an SOFC system grid-following
    and a battery grid-forming."""
    request = _drawn_request('reference_radial.diagram_sc_payload.json')
    lv = _lv(request)
    request = _with(request, _der('Battery', 'b1', capacity_kwh=500), _pcs('p1', lv, 'b1', p_set_mw=0.1),
                    _der('PV Array', 'pv'), _pcs('p2', lv, 'pv'),
                    _der('SOFC', 'fc'), _pcs('p3', lv, 'fc', s_rated_mva=0.2),
                    _der('Battery', 'b2', capacity_kwh=1000),
                    _pcs('p4', lv, 'b2', control='grid_forming', p_set_mw=0.05))
    key = next(k for k, v in request.items() if 'Parameters' in str(v.get('typ', '')))
    request[key] = dict(params)
    return request, lv


def test_battery_and_pv_ride_through_a_fault(client, quiet):
    """
    A bolted fault on the LV bus they feed, 1.0 to 1.1 s. Each PCS starts at
    the load flow's power and gives nothing into the fault (PVD1 and ESD1
    cease below 0.88 pu). The grid-following PCS are back at it 0.4 s after
    the fault clears; the grid-forming PCS by the run's end, at 50 Hz. The battery's state of charge falls by
    the energy it delivered.
    """
    request, lv = _radial({**TDS, 'tf': '6', 'fault_enabled': 'true', 'fault_bus': 'x', 'fault_tf': '1.0',
                           'fault_tc': '1.1'})
    params = next(v for v in request.values() if 'Parameters' in str(v.get('typ', '')))
    params['fault_bus'] = lv
    flow, _ = _radial(LF)
    flow = {r['id']: r for r in _post(client, quiet, flow)['pcs']}

    result = _post(client, quiet, request)
    assert result['converged'] is True
    assert not any('could not be initialised' in w for w in result['warnings']), result['warnings']
    t = np.asarray(result['time'])
    at = lambda series, x: float(np.asarray(series, dtype=float)[np.searchsorted(t, x)])
    pcs = {r['id']: r for r in result['pcs']}
    assert {r['model'] for r in pcs.values()} == {'ESD1', 'PVD1', 'GENCLS'}
    for cell, row in pcs.items():
        p0 = row['p_mw'][0]
        assert p0 == pytest.approx(flow[cell]['p_mw'], abs=1e-5), cell
        # Q to 1 % of its rating: a grid-forming PCS's through its droop from a voltage ANDES
        # and pandapower agree on to some 1e-4 pu.
        assert row['q_mvar'][0] == pytest.approx(flow[cell]['q_mvar'], abs=0.01 * row['s_rated_mva']), cell
        assert abs(at(row['p_mw'], 1.05)) < 0.01 * p0, cell
        if row['control'] == 'grid_following':
            assert at(row['p_mw'], 1.5) == pytest.approx(p0, rel=0.01), cell
        assert row['p_mw'][-1] == pytest.approx(p0, rel=0.005), cell

    # The battery's state of charge, by the energy it delivered (EtaD = 1).
    b1 = pcs['cell-p1']
    delivered_mwh = np.trapezoid(b1['p_mw'], t) / 3600.0
    assert b1['soc_percent'][0] == pytest.approx(50.0, abs=1e-3)
    assert b1['soc_percent'][0] - b1['soc_percent'][-1] == pytest.approx(100.0 * delivered_mwh / 0.5, rel=0.01)
    assert pcs['cell-p4']['frequency_hz'][-1] == pytest.approx(50.0, abs=1e-3)
    v = next(s['values'] for s in result['bus_voltage'] if s['name'] == 'LV network A')
    assert v[-1] == pytest.approx(v[0], abs=1e-3)


def test_each_pcs_model_carries_its_source(quiet):
    """
    ESD1 a battery's window and energy, a flywheel's as its speed squared;
    PVD1 a PV array's MPP and an SOFC system's rating (its ramp and minimum
    load never act on a run's constant power order: REGCA1 + REECA1 gave
    them, and stalled at REGCA1's low-voltage breakpoint); a grid-forming
    PCS's virtual machine (GENCLS) its droop as its
    damping, its power filter as its inertia. ESD1's and PVD1's frequency
    trip points IEEE 1547-2018 Category III's, scaled to 50 Hz.
    """
    request = _two_buses(
        _der('Battery', 'b1', capacity_kwh=500, soc_percent=40, soc_min_percent=15, soc_max_percent=95),
        _pcs('p1', 'b', 'b1', p_set_mw=0.1),
        _der('Flywheel', 'fw', e_max_kwh=5, speed_percent=80, speed_min_percent=40),
        _pcs('p2', 'b', 'fw', s_rated_mva=0.25),
        _der('PV Array', 'pv'), _pcs('p3', 'b', 'pv'),
        _der('SOFC', 'fc', p_rated_kw=100, ramp_percent_s=2, min_load_percent=30, aux_load_percent=5),
        _pcs('p4', 'b', 'fc', s_rated_mva=0.2),
        _pcs('p5', 'a', 'b2', control='grid_forming', s_rated_mva=1.0, droop_pf_percent=4),
        _der('Battery', 'b2', capacity_kwh=1000), params=TDS)
    with quiet():
        ss, meta = andes_electrisim.build_system(request, TDS)
    vin = lambda model, name, idx: float(getattr(getattr(ss, model), name).vin[list(getattr(ss, model).idx.v).index(idx)])
    g = meta['gen_map']
    b1, fw = g['p1']['model_idx'], g['p2']['model_idx']
    assert (vin('ESD1', 'En', b1), vin('ESD1', 'SOCinit', b1), vin('ESD1', 'SOCmin', b1), vin('ESD1', 'SOCmax', b1)) \
        == pytest.approx((0.5, 0.40, 0.15, 0.95))
    assert (vin('ESD1', 'En', fw), vin('ESD1', 'SOCinit', fw), vin('ESD1', 'SOCmin', fw)) \
        == pytest.approx((5e-3, 0.64, 0.16))
    assert (vin('ESD1', 'ft0', b1), vin('ESD1', 'ft1', b1)) == pytest.approx((56.5 * 50 / 60, 58.5 * 50 / 60))
    pv = g['p3']
    assert vin('PVD1', 'pmx', pv['model_idx']) == pytest.approx(pv['p_mw'] / 0.5)
    fc = g['p4']
    assert fc['model'] == 'PVD1' and not ss.REGCA1.n
    assert vin('PVD1', 'pmx', fc['model_idx']) == pytest.approx(0.98 * 0.095 / 0.2)
    gf = g['p5']['model_idx']
    assert vin('GENCLS', 'D', gf) == pytest.approx(25.0) and vin('GENCLS', 'M', gf) == pytest.approx(0.5)


def _sofc_island(ramp, tf, set_kw=120, **params):
    """An island: a 1 MVA grid-forming PCS, a running 400 kW SOFC group and one that may stand by; the 0.6 MW load."""
    fc = dict(p_rated_kw=400, ramp_percent_s=ramp, min_load_percent=30, aux_load_percent=0)
    return _two_buses(_der('Battery', 'ba', capacity_kwh=1000), _pcs('ga', 'a', 'ba', control='grid_forming', s_rated_mva=1.0),
                      _der('SOFC', 'f1', p_set_kw=set_kw, **fc), _pcs('p1', 'b', 'f1', q_mode='pf', pf=0.96, efficiency_percent=100),
                      _der('SOFC', 'f2', p_set_kw=set_kw, **fc), _pcs('p2', 'b', 'f2', q_mode='pf', pf=0.96, efficiency_percent=100),
                      grid=False, params={**TDS, 'tf': str(tf), 'fault_enabled': 'false', **params})


def _series(result, name, key='p_mw'):
    t = np.asarray(result['time'], dtype=float)
    return t, np.asarray(next(r for r in result['pcs'] if r['name'] == name)[key], dtype=float)


def test_sofc_groups_hold_their_power_without_the_ramp(client, quiet):
    """Unasked for, nothing moves: ANDES holds a grid-following inverter's power."""
    result = _post(client, quiet, _sofc_island(2, 4))
    _, p1 = _series(result, 'p1')
    assert p1[-1] == pytest.approx(p1[0], abs=1e-4) and p1[0] == pytest.approx(0.12, abs=1e-3)


def test_sofc_ramp_moves_at_its_rate_and_standby_returns_late(client, quiet):
    """
    The stations carry the load's 0.36 MW, more than the groups can take up in the run: the running group climbs at
    its 2 %/s of 400 kW, 8 kW/s, at its 0.96 power factor. The one in hot standby starts at nothing, below its minimum
    load, and climbs at the same rate from 3 s.
    """
    result = _post(client, quiet, _sofc_island(2, 8, sofc_ramp='true', sofc_standby='p2', sofc_return_s='3'))
    t, p1 = _series(result, 'p1')
    _, p2 = _series(result, 'p2')
    _, q1 = _series(result, 'p1', 'q_mvar')
    assert p1[-1] - p1[0] == pytest.approx(0.008 * 8, abs=0.002)
    assert q1[-1] / p1[-1] == pytest.approx(math.tan(math.acos(0.96)), abs=0.01)
    assert np.abs(p2[t < 2.9]).max() < 1e-4
    assert p2[-1] == pytest.approx(0.008 * 5, abs=0.002)


def test_sofc_ramp_stops_when_it_has_caught_the_load(client, quiet):
    """At 20 %/s the groups take the whole load and the stations end at nothing: their droop is measured from what they started carrying, so the island ends that far above 50 Hz (2 %)."""
    result = _post(client, quiet, _sofc_island(20, 20, sofc_ramp='true'))
    _, ga = _series(result, 'ga')
    _, f = _series(result, 'ga', 'frequency_hz')
    assert ga[0] > 0.3 and abs(ga[-1]) < 2e-3 and f[-1] == pytest.approx(50.0 + 0.02 * 50.0 * ga[0], abs=5e-3)
    assert _series(result, 'p1')[1][-1] == pytest.approx(_series(result, 'p2')[1][-1], abs=1e-3)


def test_sofc_ramp_keeps_a_group_at_its_minimum_load(client, quiet):
    """Above the load the groups stay at their 30 % minimum, 240 kW between them against a 100 kW load: the stations charge."""
    request = _sofc_island(20, 5, sofc_ramp='true')
    next(v for v in request.values() if isinstance(v, dict) and v.get('typ', '').startswith('Load'))['p_mw'] = '0.1'
    result = _post(client, quiet, request)
    _, p1 = _series(result, 'p1')
    _, ga = _series(result, 'ga')
    assert p1.min() == pytest.approx(0.12, abs=1e-3) and ga[-1] < -0.1


def test_sofc_standby_without_the_ramp_says_it_was_not_applied(client, quiet):
    """The dialog takes the names whatever the switch says: with the ramp off the group runs, and the result says why."""
    result = _post(client, quiet, _sofc_island(2, 4, sofc_standby='p2'))
    assert _series(result, 'p2')[1][0] == pytest.approx(0.12, abs=1e-3)
    assert any('hot standby was not applied' in w for w in result['warnings'])


def test_sofc_ramp_with_nothing_to_follow_says_so(client, quiet):
    """A group with no grid-forming PCS in its island holds, and a switch left on does not pass for a result."""
    request = _two_buses(_der('SOFC', 'f1', p_set_kw=120, p_rated_kw=400, ramp_percent_s=2, min_load_percent=30, aux_load_percent=0),
                         _pcs('p1', 'b', 'f1', q_mode='pf', pf=0.96, efficiency_percent=100),
                         params={**TDS, 'tf': '2', 'fault_enabled': 'false', 'sofc_ramp': 'true'})
    result = _post(client, quiet, request)
    assert any('nothing moved' in w for w in result['warnings'])


def _one_bus_island(tied, **params):
    """A battery's grid-forming PCS and a 0.6 MW load following a half-then-full profile: on one bus, or the PCS on its own bus through a short line."""
    profile = {'P': {'name': 'P', 'kind': 'power', 'dt': 0.1, 't0': 0, 'p': [0.5] * 20 + [1.0] * 21}}
    request = {'0': {**TDS, 'tf': '4', 'fault_enabled': 'false', 'load_model': 'power', 'profile_repeat': 'false', 'load_profiles': profile, **params},
               '1': _bus('a', 0.4), '2': {**LOAD, 'bus': 'a', 'load_profile_id': 'P', 'load_profile_q_mode': 'pf'}}
    pcs = 'c' if tied else 'a'
    elements = [_der('Battery', 'ba', capacity_kwh=1000), _pcs('ga', pcs, 'ba', control='grid_forming', s_rated_mva=1.0)]
    if tied:
        elements += [_bus('c', 0.4), {**LINE, 'name': 'tie', 'id': 'cell-tie', 'userFriendlyName': 'TIE', 'busFrom': 'c', 'busTo': 'a', 'length_km': '0.001'}]
    return _with(request, *elements)


def test_a_one_bus_island_runs_as_the_same_island_with_a_tie(client, quiet):
    """ANDES pins a bus without a branch: this one ran flat at the load's set power and 50 Hz, with no error."""
    lone, tied = _post(client, quiet, _one_bus_island(False)), _post(client, quiet, _one_bus_island(True))
    t, p = _series(lone, 'ga')
    _, f = _series(lone, 'ga', 'frequency_hz')
    assert p[t < 1.9].max() == pytest.approx(0.3, abs=2e-3) and p[-1] == pytest.approx(0.6, abs=2e-3)
    assert f[t < 1.9].min() == pytest.approx(50.0, abs=1e-3) and f[-1] < 49.8      # its droop, from where it started
    assert np.interp(t, *_series(tied, 'ga')) == pytest.approx(p, abs=5e-3)
    assert [b['name'] for b in lone['bus_voltage']] == ['A']


def test_an_outage_leaving_one_bus_keeps_its_load_on_the_pcs(client, quiet):
    """The grid's line out: the load's bus, with the PCS on it and no branch left, stays an island, and the PCS takes the load."""
    request = _two_buses(_der('Battery', 'bb', capacity_kwh=1000), _pcs('gb', 'b', 'bb', control='grid_forming', s_rated_mva=1.0),
                         params={**TDS, 'tf': '3', 'fault_enabled': 'false', 'load_model': 'power', 'toggle_line': 'l1', 'toggle_t': '1'})
    result = _post(client, quiet, request)
    t, p = _series(result, 'gb')
    assert abs(p[t < 0.9]).max() < 0.1 and p[-1] == pytest.approx(0.6, abs=5e-3)
    assert any('as an island' in w for w in result['warnings'])


@pytest.mark.parametrize('typ', ['Load0', 'Storage0', 'Motor0', 'Capacitor0', 'Static Generator0', 'External Grid0'])
def test_an_element_with_no_bus_is_named(client, quiet, typ):
    """It is left out, so the network reads smaller: it says so, on the grid as on an island. Out of service, it has nothing to say."""
    missing = {'typ': typ, 'name': 'lost', 'id': 'cell-lost', 'userFriendlyName': 'LOST', 'bus': 'nowhere', 'p_mw': '0.5', 'in_service': 'true'}
    params = {**TDS, 'tf': '0.5', 'fault_enabled': 'false', 'load_model': 'power'}
    source = _der('Battery', 'bb', capacity_kwh=1000), _pcs('gb', 'b', 'bb', control='grid_forming', s_rated_mva=1.0)
    result = _post(client, quiet, _two_buses(missing, *source, params=params))
    assert any("'LOST'" in w and 'not connected to a bus' in w for w in result['warnings']), result['warnings']
    result = _post(client, quiet, _two_buses({**missing, 'in_service': 'false'}, *source, params=params))
    assert not any("'LOST'" in w for w in result['warnings']), result['warnings']


def test_the_frequency_result_is_the_island_with_the_load(client, quiet):
    """The line between two PCS out: the one left unloaded rises on its droop, and the mean of the two is neither island's."""
    request = _two_buses(_der('Battery', 'ba', capacity_kwh=1000), _pcs('ga', 'a', 'ba', control='grid_forming', s_rated_mva=1.0),
                         _der('Battery', 'bb', capacity_kwh=1000), _pcs('gb', 'b', 'bb', control='grid_forming', s_rated_mva=1.0),
                         grid=False, params={**TDS, 'tf': '3', 'fault_enabled': 'false', 'load_model': 'power', 'toggle_line': 'l1', 'toggle_t': '1'})
    result = _post(client, quiet, request)
    _, f_load = _series(result, 'gb', 'frequency_hz')
    _, f_apart = _series(result, 'ga', 'frequency_hz')
    assert f_apart[-1] > 50 > f_load[-1]
    assert result['frequency_hz'][-1] == pytest.approx(f_load[-1], abs=1e-6)
    assert any('island with the most load' in w and 'GA' in w for w in result['warnings']), result['warnings']


def test_a_lone_machine_switched_off_later_says_it_ran_flat(client, quiet):
    """It gets no stub, which its trip would leave floating; it says what that means."""
    machine = {'typ': 'Generator0', 'name': 'm', 'id': 'cell-m', 'userFriendlyName': 'M', 'bus': 'b', 'p_mw': '0.6', 'vm_pu': '1', 'sn_mva': '1', 'vn_kv': '0.4', 'in_service': 'true'}
    request = _two_buses(machine, grid=False, params={**TDS, 'tf': '2', 'fault_enabled': 'false', 'load_model': 'power', 'toggle_gen': 'm', 'toggle_gen_t': '1'})
    request = {k: v for k, v in request.items() if v.get('name') not in ('l1', 'a')}
    result = _post(client, quiet, request)
    assert any("'M' has no branch" in w for w in result['warnings']), result['warnings']


def test_an_island_held_only_by_pcs_rides_through_a_fault(client, quiet):
    """
    An island its grid-forming PCS alone holds: its virtual machine is the
    island's slack, starting at the load flow's power (the load and the
    line's losses); a fault at the load clears and it returns to that power
    at 50 Hz. ANDES's REGCV1 could not hold it - unstable for a small load
    step - and the study sent it to the EMT study.
    """
    request = _two_buses(_der('Battery', 'ba', capacity_kwh=1000),
                         _pcs('ga', 'a', 'ba', control='grid_forming', s_rated_mva=1.0),
                         grid=False, params={**TDS, 'tf': '5', 'fault_enabled': 'true', 'fault_bus': 'b',
                                             'fault_tf': '1.0', 'fault_tc': '1.1'})
    with quiet():
        result = json.loads(client.post('/', json=request).get_data(as_text=True))
    assert not result.get('error') and result['converged'] is True, result.get('message')
    # The fault takes it to its current limit, which its virtual impedance holds it at.
    warns = [w for w in result['warnings'] if not w.startswith('Loads are ')]
    assert result['time'][-1] == pytest.approx(5.0) and len(warns) == 1, result['warnings']
    assert warns[0].startswith("PCS 'GA' reached its current limit (1.2 pu of its rating) at t = 1.0")
    (pcs,) = result['pcs']
    t, i = np.asarray(result['time']), np.asarray(pcs['current_pu'])
    assert i[(t > 1.07) & (t < 1.1)].max() <= 1.2 * 1.03
    flow = _post(client, quiet, _two_buses(_der('Battery', 'ba', capacity_kwh=1000),
                                           _pcs('ga', 'a', 'ba', control='grid_forming', s_rated_mva=1.0),
                                           grid=False, params=LF))
    # To its line's losses: as the slack its voltage is its set point's, not on its Q-V droop.
    assert pcs['model'] == 'GENCLS' and pcs['p_mw'][0] == pytest.approx(flow['pcs'][0]['p_mw'], rel=2e-3)
    assert pcs['p_mw'][-1] == pytest.approx(pcs['p_mw'][0], rel=1e-4)
    assert min(pcs['frequency_hz']) < 49.9 and pcs['frequency_hz'][-1] == pytest.approx(50.0, abs=1e-3)


@pytest.mark.parametrize('model, weights', [(None, (0, 0, 1)), ('impedance', (0, 0, 1)), ('current', (0, 1, 0)),
                                            ('power', (1, 0, 0)), ('zip', (0, 0, 1))])
def test_load_model_through_the_run(quiet, model, weights):
    """
    ANDES holds a load at its power in the load flow only; through the run it is what the PQ weights say - by
    default all constant impedance. The study's load_model sets them, and the result says which it was; an unknown
    one falls back to constant impedance and says so.
    """
    params = {**TDS, **({'load_model': model} if model else {})}
    request, _ = _radial(params)
    with quiet():
        ss, meta = andes_electrisim.build_system(request, params)
    c = ss.PQ.config
    assert (c.p2p, c.p2i, c.p2z) == weights and (c.q2q, c.q2i, c.q2z) == weights
    name = {(0, 0, 1): 'impedance', (0, 1, 0): 'current', (1, 0, 0): 'power'}[weights]
    assert meta['load_model'] == name
    assert any(w.startswith(f'Loads are constant {name} through the run') for w in meta['warnings'])
    assert (model == 'zip') == any("Load model 'zip'" in w for w in meta['warnings'])
