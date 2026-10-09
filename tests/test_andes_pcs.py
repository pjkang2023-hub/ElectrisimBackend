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
from test_pcs import LF, _der, _lv, _pcs, _post, _two_buses

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
    assert result['time'][-1] == pytest.approx(5.0) and len(result['warnings']) == 1, result['warnings']
    assert result['warnings'][0].startswith("PCS 'GA' reached its current limit (1.2 pu of its rating) at t = 1.0")
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
