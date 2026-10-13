"""
Phase 27 (design note "PCS3750C-MV reference"): a differential relay that
compares the currents into its protected element, and a DC breaker in front
of a battery.
"""
import copy

import pandapower as pp
import pandapower.shortcircuit as sc
import pytest

import pandapower_electrisim as pe
from test_dc_elements import _breaker, _drawn_request, _run, _with
from test_dc_fault import _fault, _study


# --- The differential relay's zone (87L / 87T) -------------------------------------------------

def _net():
    """A 20 kV grid at A, cables A-B and B-C, an inverter at C (1.2 x its 5 MVA), a 20/0.4 kV Dyn at B."""
    net = pp.create_empty_network()
    a, b, c = (pp.create_bus(net, 20, name=n) for n in 'ABC')
    lv = pp.create_bus(net, 0.4, name='LV')
    pp.create_ext_grid(net, a, s_sc_max_mva=500, rx_max=0.1, x0x_max=1, r0x0_max=0.1)
    for f, t in ((a, b), (b, c)):
        pp.create_line_from_parameters(net, f, t, 2, 0.1, 0.1, 10, 0.5, r0_ohm_per_km=0.3, x0_ohm_per_km=0.3,
                                       c0_nf_per_km=10, endtemp_degree=80)
    pp.create_sgen(net, c, 5, sn_mva=5, k=1.2)
    pp.create_transformer_from_parameters(net, b, lv, 1, 20, 0.4, 1, 6, 0, 0, shift_degree=150, vector_group='Dyn',
                                          vk0_percent=6, vkr0_percent=1, mag0_percent=100, mag0_rx=0,
                                          si0_hv_partial=0.9)
    pp.create_switch(net, a, 0, 'l')
    pp.create_switch(net, b, 0, 'l')
    pp.create_switch(net, b, 1, 'l')
    pp.create_switch(net, b, 0, 't')
    return net


def _faulted(fault, line=None, bus=None):
    net = _net()
    pe._prot_ensure_bus_geodata(net)
    if line is not None:
        net = pe._prot_split_line(net, line, 0.5)
        bus = int(max(net.bus.index))
        not_line = _net().switch.index[_net().switch['et'] != 'l']
        net.switch.loc[not_line, 'element'] = _net().switch.loc[not_line, 'element']
    pe._prot_earth_fault_net(net, fault)
    sc.calc_sc(net, bus=bus, branch_results=True, fault=fault, case='max')
    return net, bus


@pytest.mark.parametrize('fault', ['3ph', '1ph'])
def test_differential_operates_for_a_fault_on_its_cable(fault):
    """
    A fault halfway along A-B, fed from both ends: the grid through A, the
    inverter through B. All of it enters the zone, so the operating current is
    the fault current; an earth fault's branch results have no angles, and the
    magnitudes are added. Cable B-C carries the inverter's current through: its
    relay sees none. (Taking the branch current alone, it tripped too.)
    """
    net, bus = _faulted(fault, line=0)
    ik = float(net.res_bus_sc.at[bus, 'ikss_ka']) * 1e3
    for sw in (0, 1):
        zone = pe._prot_diff_zone_currents(net, sw, bus)
        assert zone['phasors'] == (fault == '3ph')
        assert zone['operate_a'] == pytest.approx(ik, rel=1e-3)
        assert len(zone['terminals_a']) == 2 and min(zone['terminals_a']) > 0
    through = pe._prot_diff_zone_currents(net, 2, bus)
    assert through['operate_a'] < 1e-6 * through['restraint_a']
    assert through['restraint_a'] == pytest.approx(5 * 1.2 / (3 ** 0.5 * 20) * 1e3, rel=0.02)


@pytest.mark.parametrize('fault', ['3ph', '1ph'])
def test_differential_restrains_for_a_fault_beyond_its_transformer(fault):
    """
    A fault on the LV side of the 20/0.4 kV Dyn: the transformer's HV and LV
    currents, the LV referred to the switch's 20 kV by the ratio (pandapower's
    short-circuit angles leave the vector group's shift out), cancel.
    """
    net, bus = _faulted(fault, bus=3)
    zone = pe._prot_diff_zone_currents(net, 3, bus)
    hv, lv = zone['terminals_a']
    assert hv > 100 and lv == pytest.approx(hv, rel=1e-3)
    assert zone['operate_a'] < 1e-3 * hv


def test_differential_trips_only_its_own_cable_in_the_study():
    """Both relays of the faulted cable trip at t_g; the healthy cable's, through-fault, does not."""
    net, bus = _faulted('3ph', line=0)
    summaries = [{'sw_idx': i, 'kind': 'differential', 'custom_evaluator': 'differential',
                  'settings': {'I_diff_a': 100.0, 'diff_slope': 0.3, 't_g': 0.04}} for i in (0, 1, 2)]
    rows = {r['switch_idx']: r for r in pe._prot_custom_trip_rows(net, summaries, '3ph', bus)}
    assert rows[0]['tripped'] and rows[1]['tripped'] and rows[0]['t_trip_s'] == 0.04
    assert not rows[2]['tripped']


# --- A DC breaker in front of a battery ---------------------------------------------------------

def _battery(name='gs', bus='dc_b', **fields):
    return {'typ': 'Battery0', 'name': name, 'id': f'cell-{name}', 'userFriendlyName': name.upper(), 'bus': bus,
            'vn_v': '800', 'capacity_kwh': '500', 'r0_mohm': '50', 'soc_percent': '80',
            **{k: str(v) for k, v in fields.items()}}


def _request(closed=True, opening_time_ms=10):
    # The payload names the shape beyond the breaker: et 'Battery'.
    return _with(_drawn_request(), _battery(),
                 _breaker('qb', 'dc_b', 'gs', 'Battery', closed=closed, rated_current_ka=1, breaking_capacity_ka=30,
                          opening_time_ms=opening_time_ms))


def test_breaker_in_front_of_a_battery_in_the_load_flow(client, quiet):
    """Closed, it carries the battery's current; open, the battery is out of the network."""
    closed = _run(client, quiet, _request())
    (brk,) = closed['dcbreakers']
    bat = next(d for d in closed['ders'] if d['name'] == 'gs')
    assert bat['i_ka'] is not None and abs(bat['i_ka']) > 1e-3
    assert brk['i_ka'] == pytest.approx(abs(bat['i_ka']), rel=1e-3)
    assert not any('QB' in w and 'left out' in w for w in closed.get('warnings', []))
    opened = _run(client, quiet, _request(closed=False))
    bat = next(d for d in opened['ders'] if d['name'] == 'gs')
    assert bat['in_service'] is False and bat['p_mw'] is None
    assert opened['dcbreakers'][0]['i_ka'] == 0


def test_breaker_in_front_of_a_battery_in_the_dc_fault_study(client, quiet):
    """
    A fault on the battery's bus: its breaker carries the battery's own
    current. A fault between the breaker and the battery is the same bolted
    fault, the breaker then carrying everything else the bus feeds it - the
    fault's current less the battery's (opening as the fault settles, to
    compare with Ik).
    """
    result = _study(client, quiet, _request(opening_time_ms=38), duration_ms=40)
    fault = _fault(result, 'cell-dc_b')
    bat = next(c for c in fault['contributions'] if c['kind'] == 'Battery')
    (brk,) = fault['breakers']
    assert brk['i_open_ka'] == pytest.approx(bat['ik_ka'], rel=0.02)
    assert brk['i_open_far_side_ka'] == pytest.approx(fault['ik_ka'] - bat['ik_ka'], rel=0.05)
    assert brk['ip_far_side_ka'] > brk['i_open_far_side_ka']
    assert brk['exceeds'] is False
    # On the other bus the breaker is not beside the fault: no far side.
    assert 'i_open_far_side_ka' not in _fault(result, 'cell-dc_a')['breakers'][0]


# --- The undervoltage relay (27) and breaker failure (50BF) -----------------------------------

def test_undervoltage_relay_reads_its_buses_fault_voltage():
    """
    A 27 element on cable B-C's breaker at B: the fault halfway along A-B
    pulls B down, read from the cable's branch results at B. An earth fault's
    have no voltages: not evaluated, and said so.
    """
    net, bus = _faulted('3ph', line=0)
    vm = float(net.res_line_sc.at[1, 'vm_from_pu'])
    assert 0 < vm < 0.9
    summaries = [{'sw_idx': 2, 'kind': 'undervoltage', 'custom_evaluator': 'undervoltage',
                  'settings': {'U_pu': u, 't_u': 0.4}} for u in (vm + 0.05, vm - 0.05)]
    rows = pe._prot_custom_trip_rows(net, summaries[:1], '3ph', bus) + pe._prot_custom_trip_rows(net, summaries[1:], '3ph', bus)
    assert rows[0]['tripped'] and rows[0]['t_trip_s'] == 0.4 and rows[0]['evaluation']['vm_pu'] == pytest.approx(vm)
    assert not rows[1]['tripped']
    net, bus = _faulted('1ph', line=0)
    (row,) = pe._prot_custom_trip_rows(net, summaries[:1], '1ph', bus)
    assert not row['tripped'] and row['evaluation']['evaluated'] is False


def test_breaker_failure_on_the_radial_grid(client, quiet):
    """
    Each feeder breaker failing to open on its feeder's fault: with 50BF
    (0.15 s) the substation's other breakers trip and the fault is off 0.22 s
    after it began - unless a source lies beyond the failed breaker: feeder
    B's gas engine keeps feeding its fault, nothing clearing it. The relays
    alone back up none (T1's pickup is above the feeders' fault current).
    """
    import json
    from test_protection_inverters import MANUAL, REFERENCE_DIR
    import os
    with open(os.path.join(REFERENCE_DIR, 'reference_radial.diagram_protection_payload.json'), encoding='utf-8') as fh:
        p = json.load(fh)
    for v in p.values():
        if isinstance(v, dict) and str(v.get('typ', '')).startswith('Switch') and v.get('protection_type') == 'ocr':
            v.update(pickup_mode='manual', **MANUAL['reference_radial'].get(v['userFriendlyName'], MANUAL['reference_radial'][None]))
            if v['userFriendlyName'] != 'CB_T1':
                v['bf_time_s'] = '0.15'
    with quiet():
        out = json.loads(client.post('/', json=p).get_data(as_text=True))
    assert out['summary']['n_breaker_failure_uncleared'] == 2
    for sc in out['scenarios']:
        (r,) = sc['breaker_failure']
        assert r['t_clear_relays_s'] is None
        others = {'CB_T1', 'CB_A', 'CB_B', 'Wind feeder breaker'} - {r['failed_switch']}
        assert set(r['opened_by_50bf']) == others
        if r['failed_switch'] == 'CB_B':
            assert r['t_clear_s'] is None and sc['unprotected_sources'] == ['Gas engine']
        else:
            assert r['t_clear_s'] == pytest.approx(0.22)
