"""
Unit protection on breakers drawn between two buses (C003 Rev C.2: every breaker is): the 87L / 87T of a breaker
between its board and a node is the line or transformer beyond the node; its 87B is its own current, a bus section's
at both boards; a cable ending on a node of its own is measured at its far breaker. The draft settings
(c003_revC2_site_settings_draft.json, the C003 protection settings draft) are carried onto the drawing by
mgtwin_site_io.apply_protection, and Electrisim's protection coordination evaluates them.
"""
import json
import os

import pytest

from test_reference_c003 import _payload, _post

HERE = os.path.dirname(os.path.abspath(__file__))


def _draft():
    with open(os.path.join(HERE, 'reference', 'c003_revC2_site_settings_draft.json'), encoding='utf-8') as fh:
        return json.load(fh)


def _trips(client, site, payload, bus, fault_type, state='NORMAL'):
    """{breaker: (t, how)} for a bolted fault on `bus`, with the scenario's clearing time and unwanted trips."""
    from mgtwin_site_io import apply_protection, apply_state
    group = next(st for st in site['islands'] if st['id'] == state).get('settings_group', 'A')
    prot, _ = apply_protection(payload, site, group)
    p = apply_state(prot, site, state)
    cell = {v.get('userFriendlyName'): v['name'] for v in p.values() if isinstance(v, dict) and 'name' in v}
    p['0'] = {'typ': 'ProtectionCoordinationPandaPower', 'user_email': 'test@electrisim', 'fault_type': fault_type,
              'case': 'max', 'fault_location_mode': 'bus', 'fault_bus_id': cell[bus], 'sc_fraction': 0.5,
              'grading_mode': 'manual', 't_diff': 0.3, 'frequency': '50'}
    out = _post(client, p)
    sc = out['scenarios'][0]
    trips = {t['user_friendly_name']: (t['t_trip_s'], 'B' if (t.get('bus_differential') or {}).get('tripped_first')
                                       else t.get('device')) for t in sc['trip'] if t.get('tripped')}
    return trips, sc.get('clearing_time_s'), out.get('unwanted_trips') or []


@pytest.mark.parametrize('fault_type', ['3ph', '1ph'])
def test_each_unit_element_trips_for_its_own_zone_only(client, fault_type):
    """
    NORMAL, IEC max. A fault on feeder F_S1A's node: its 87L alone, both ends in 0.03 s - no board's 87B, though the
    board feeds it (a meshed board drawing out through two feeders summed by magnitude used to). On PCS1's transformer
    node: its 87T alone. On board SOFC1A: the board's 87B trips every breaker on it - the bus section CB_BS1, which
    carries nothing but the 87B, among them - and nothing on SOFC1B beyond it.
    """
    site, payload = _draft(), _payload('revc2')
    trips, _, unwanted = _trips(client, site, payload, 'F_S1A_N', fault_type)
    assert trips == {'CB_F_S1A': (0.03, 'Differential (87)'), 'CB_F_S1A_FAR': (0.03, 'Differential (87)')} and not unwanted
    trips, clearing, unwanted = _trips(client, site, payload, 'PCS1_HV', fault_type)
    assert trips == {'CB_PCS1': (0.03, 'Differential (87)')} and clearing == pytest.approx(0.03) and not unwanted
    trips, clearing, unwanted = _trips(client, site, payload, 'SOFC1A', fault_type)
    board = {'CB_BS1', 'CB_F_S1A'} | {f'CB_SOFC{n:02d}_1' for n in range(1, 6)}
    assert set(trips) == board and all(t == 0.05 for t, _ in trips.values())
    assert clearing == pytest.approx(0.05) and not unwanted


def test_a_cable_ending_on_its_own_node_is_measured_at_its_far_breaker(client):
    """
    F_S1A with its breaker at the MVA end (Site Corrections item 9): its 87L is two-ended, each end's breaker against
    the other's current. A fault on either node is inside - both ends trip in 0.03 s and the fault is cleared from
    both boards; a fault on MVA, beyond the far breaker, leaves both 87L ends still and is cleared by MVA's 87B, the
    far breaker with it.
    """
    site, payload = _draft(), _payload('revc2')
    for node in ('F_S1A_N', 'F_S1A_FAR_N'):
        trips, clearing, unwanted = _trips(client, site, payload, node, '3ph')
        assert trips == {'CB_F_S1A': (0.03, 'Differential (87)'), 'CB_F_S1A_FAR': (0.03, 'Differential (87)')}, node
        assert clearing == pytest.approx(0.03) and not unwanted
    trips, _, unwanted = _trips(client, site, payload, 'MVA', '3ph')
    assert 'CB_F_S1A' not in trips and trips['CB_F_S1A_FAR'] == (0.05, 'B') and not unwanted


def test_a_bus_breakers_angle_is_read_from_its_own_branch_row():
    """
    pandapower keeps a closed bus-bus breaker given an impedance as a ppc branch with its current's angle at each end,
    but its switch rows do not follow net.switch's order: the row is found by its two buses. Into the breaker at
    its upstream bus the fault current's angle; at its downstream bus the opposite.
    """
    import numpy as np
    import pandapower as pp
    import pandapower.shortcircuit as sc
    from pandapower_electrisim import _prot_bus_switch_phasor
    net = pp.create_empty_network(sn_mva=10)
    b = [pp.create_bus(net, 33) for _ in range(5)]
    pp.create_ext_grid(net, b[0], s_sc_max_mva=100, rx_max=0.1)
    pp.create_line_from_parameters(net, b[0], b[1], 1, 0.1, 0.3, 10, 0.5)
    pp.create_sgen(net, b[4], p_mw=1, sn_mva=1, k=1.2)
    pp.create_switch(net, b[3], b[4], et='b', z_ohm=1e-3)
    sw = pp.create_switch(net, b[2], b[1], et='b', z_ohm=1e-3)          # drawn downstream -> upstream
    pp.create_line_from_parameters(net, b[2], b[3], 0.1, 0.1, 0.3, 10, 0.5)
    sc.calc_sc(net, bus=b[3], branch_results=True, fault='3ph', case='max')
    line_angle = float(net.res_line_sc.at[0, 'ikss_from_degree'])
    up, down = _prot_bus_switch_phasor(net, sw, b[1]), _prot_bus_switch_phasor(net, sw, b[2])
    assert up[0] == pytest.approx(down[0]) and up[0] > 1.0
    assert np.cos(np.radians(up[1] - line_angle)) == pytest.approx(1.0, abs=1e-3)
    assert np.cos(np.radians(down[1] - line_angle)) == pytest.approx(-1.0, abs=1e-3)
