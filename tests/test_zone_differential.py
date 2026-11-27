"""
Zone differential (87 over a zone of buses), C003 six topologies, design A: a ring bus whose every element is fed
through two ring breakers. The element's zone is its ring node and its tap; the breakers on its edge are its CTs and
trip for it; a fault anywhere else balances. The draft relays (specs/c003six/c003six_protection.py) are carried onto
the model by mgtwin_site_io.apply_protection with zones=True.
"""
import contextlib
import io
import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    with open(os.path.join(HERE, 'reference', name), encoding='utf-8') as fh:
        return json.load(fh)


def _trips(client, bus, fault_type, state='NORMAL'):
    from mgtwin_site_io import apply_protection, apply_state
    site, payload = _load('c003six_A_protection_site.json'), _load('c003six_A.payload.json')
    p, _ = apply_protection(payload, site, 'A', fault_type, zones=True)
    p = apply_state(p, site, state)
    p['0'] = {'typ': 'ProtectionCoordinationPandaPower', 'user_email': 'test@electrisim', 'fault_type': fault_type,
              'case': 'max', 'fault_location_mode': 'bus', 'fault_bus_id': bus, 'sc_fraction': 0.5,
              'grading_mode': 'manual', 't_diff': 0.3, 'frequency': '50'}
    with contextlib.redirect_stdout(io.StringIO()):
        out = json.loads(client.post('/', json=p).get_data(as_text=True))
    sc = out['scenarios'][0]
    return ({t['user_friendly_name']: t['t_trip_s'] for t in sc['trip'] if t.get('tripped') and t['t_trip_s'] < 0.1},
            sc.get('clearing_time_s'), [u['user_friendly_name'] for u in out.get('unwanted_trips') or []])


@pytest.mark.parametrize('fault_type', ['3ph', '1ph'])
def test_a_ring_element_is_cleared_by_its_two_ring_breakers(client, fault_type):
    """A fault on group 01's ring node, or at its transformer's terminal beyond the tap cable: its two ring breakers in
    0.03 s, nothing else - the ring's other zones see the fault's current pass through and balance."""
    for bus in ('R_SOFC01', 'TR_SOFC01_HV'):
        trips, clearing, unwanted = _trips(client, bus, fault_type)
        assert trips == {'RB01': 0.03, 'RB02': 0.03}, bus
        assert clearing == pytest.approx(0.03) and not unwanted


def test_a_feeder_zone_reaches_its_board_end_breaker(client):
    """The feeder's zone runs from its ring node to the breaker at the load board: both ring breakers and that one."""
    trips, clearing, unwanted = _trips(client, 'R_F_MVA', '3ph')
    assert trips == {'RB01': 0.03, 'RB27': 0.03, 'CB_LN_A_MVA_FAR': 0.03} and not unwanted


def test_a_black_start_cell_fault_on_one_station_is_seen(client):
    """Cell A on one station (about 100 A of fault current): a ring element's zone still operates."""
    trips, clearing, unwanted = _trips(client, 'R_SOFC03', '3ph', 'BS_CELL_A_RAMP')
    assert set(trips) == {'RB03', 'RB04'} and not unwanted
