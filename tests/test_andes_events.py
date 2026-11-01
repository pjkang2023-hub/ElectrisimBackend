"""
The ANDES study's timed events, secondary voltage control and loads behind a UPS (C003 Phase 3): a PCS tripped
at its time and its share picked up by the grid-forming PCS; the grid-forming PCS's set point moved to bring a
pilot bus to 1.0 pu; a load going onto its UPS at a fault and walking back in.
"""
import json

import numpy as np
import pytest

from test_andes_pcs import TDS
from test_pcs import _der, _pcs, _two_buses


def _island(**params):
    """An island: a 1 MVA grid-forming battery on A; on B a 0.5 MVA grid-following SOFC at 0.2 MW and the 0.6 MW
    load LD."""
    return _two_buses(_der('Battery', 'ba', capacity_kwh=1000), _pcs('ga', 'a', 'ba', control='grid_forming', s_rated_mva=1.0),
                      _der('SOFC', 'fc', p_rated_kw=500, p_set_kw=200), _pcs('p2', 'b', 'fc', s_rated_mva=0.5),
                      grid=False, params={**TDS, 'tf': '4', **params})


def _run(client, quiet, request):
    with quiet():
        result = json.loads(client.post('/', json=request).get_data(as_text=True))
    assert not result.get('error') and result['converged'] is True, result.get('message')
    return result


def _bus(result, name):
    return np.asarray(next(b for b in result['bus_voltage'] if b['name'] == name)['values'])


def test_pcs_tripped_by_a_timed_event(client, quiet):
    """P2 tripped at 1 s: its ANDES model toggled out, its share taken up by the grid-forming PCS."""
    result = _run(client, quiet, _island(events=[{'t': 1.0, 'action': 'trip', 'element': 'P2'}]))
    assert result['events']['timed'] == [{'t': 1.0, 'action': 'trip', 'element': 'P2', 'model': 'PVD1'}]
    pcs = {p['label']: p for p in result['pcs']}
    assert pcs['P2']['p_mw'][-1] == pytest.approx(0.0, abs=1e-3)
    # the loads are impedances through the run: what GA takes up is P2's share, to the voltage's small change
    assert pcs['GA']['p_mw'][-1] - pcs['GA']['p_mw'][0] == pytest.approx(pcs['P2']['p_mw'][0], rel=0.03)


def test_event_on_an_unknown_element_is_reported_not_applied(client, quiet):
    result = _run(client, quiet, _island(events=[{'t': 1.0, 'action': 'trip', 'element': 'nowhere'}]))
    assert result['events']['timed'] == []
    assert any(w.startswith("Event at 1 s: no PCS, generator, branch or load 'nowhere'") for w in result['warnings'])


@pytest.mark.parametrize('secondary', [False, True])
def test_secondary_voltage_control_brings_its_pilot_to_1_pu(client, quiet, secondary):
    """
    On its Q-V droop alone the island's load bus B sits where the droop leaves it, below 1.0 pu, and falls further
    as P2 trips. With secondary voltage control on B (over 1 s) the grid-forming PCS's set point moves until B is
    back at 1.0 pu.
    """
    result = _run(client, quiet, _island(events=[{'t': 1.0, 'action': 'trip', 'element': 'P2'}],
                                         secondary_voltage=str(secondary).lower(), secondary_voltage_pilots='B',
                                         secondary_voltage_t_s='1'))
    v_b = _bus(result, 'B')
    if secondary:
        assert v_b[-1] == pytest.approx(1.0, abs=2e-3)
    else:
        assert v_b[-1] < 0.995


def test_load_behind_a_ups_rides_off_a_fault_and_walks_back_in(client, quiet):
    """A fault at B from 1.0 to 1.1 s: LD (0.85 pu, floor 0, back over 1 s) goes onto its UPS - all of it, and
    from zero it can walk back in - and the grid-forming PCS is back at its power by 2.5 s."""
    request = _two_buses(_der('Battery', 'ba', capacity_kwh=1000), _pcs('ga', 'a', 'ba', control='grid_forming', s_rated_mva=1.0),
                         grid=False, params={**TDS, 'tf': '4', 'fault_enabled': 'true', 'fault_bus': 'b',
                                             'fault_tf': '1.0', 'fault_tc': '1.1'})
    request['4'].update({'uv_ease_pu': '0.85', 'uv_floor_pu': '0', 'uv_restore_s': '1'})
    result = _run(client, quiet, request)
    (ups,) = [u for u in result['rectifier_easing'] if u['kind'] == 'UPS load']
    assert ups['label'] == 'LD' and 1.0 <= ups['eased_from_s'] < 1.05 and ups['min_factor'] < 1e-3
    assert ups['factor'][-1] == pytest.approx(1.0)
    t, p = np.asarray(result['time']), np.asarray(result['pcs'][0]['p_mw'])
    assert p[np.searchsorted(t, 1.15)] < 0.1 * p[0]
    assert p[-1] == pytest.approx(p[0], rel=1e-3)
    assert any(w.startswith("Load 'LD' went onto its UPS at t = 1.0") and 'down to 0 of its 0.6 MW' in w
               for w in result['warnings'])
