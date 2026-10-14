# -*- coding: utf-8 -*-
"""
The PCS3750C-MV reference: tests/reference/reference_pcs3750_mv.spec.json, three
double-conversion blocks (design note "PCS3750C-MV reference", Phase 23). A
34.5 kV utility feeds each block's T1 and four rectifiers following their DC
buses' load; two GS1.1 sodium-ion units hold each DC bus; four grid-forming PCS
per block rebuild a separate 34.5 kV backbone through T2, a ring of three
switchboards with a zigzag; three SSTs carry 10.8 MW of DGX H100 racks.

Sizing (load flow):
- 40 deg C: the twelve PCS share the island at 94 % of their rating, the
  rectifiers carry it all, the batteries idle; 92.6 % from the grid to the racks
  (each AFE's reactor resistance set for the spec's 98.5 % at its rating);
- 45 deg C: the rectifiers, current-limited on their 0.99 pu bus, reach their
  rating first and the batteries drain 0.30 MW;
- 50 deg C: the PCS are over their rating (104 %): the load does not fit;
- a block out: the other eight PCS at 142 %, inside the 150 % / 1 s overload
  only - load must be shed within a second; at 45 deg C, 149 %: on their sagging
  load bus, past even that;
- the utility at 0.95 / 1.05 pu: the 505 V grid-side buses inside 444-555 V;
  at 0.95 the rectifiers are current-limited and the batteries make up 0.36 MW;
- the GS1.1 full (1,490 V) or at the 815 V floor: the same load flow.

Protection (Phase 27): the breakers on the drawn diagram - each transformer's,
each backbone cable's two ends, each hall feeder's, a DC breaker per GS1.1 -
with a relay scheme set per study.

And the drawn diagram: tests/reference/reference_pcs3750_mv.diagram_payload.json
is the request the page built (prepareNetworkData) from the spec drawn through
/build-model - the spec's answers.
"""
import contextlib
import copy
import io
import json
import os

import pytest

import electrisim_sld as sld

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = 'reference_pcs3750_mv'
SPEC_PATH = os.path.join(HERE, 'reference', f'{GRID}.spec.json')
RACKS_MW = 3 * 3.6


@pytest.fixture(scope='module')
def spec():
    with open(SPEC_PATH, encoding='utf-8') as fh:
        return json.load(fh)


def _case(spec, inlet=40, out_block=None, vm_grid=1.0, soc=None):
    s = copy.deepcopy(spec)
    for r in s['vscs'] + s['pcs']:
        r['inlet_c'] = inlet
    if out_block:
        for r in s['pcs']:
            if r['id'].startswith(f'B{out_block}_'):
                r['in_service'] = False
    s['external_grids'][0]['vm_pu'] = vm_grid
    if soc is not None:
        for b in s['batteries']:
            b['soc_percent'] = soc
    with contextlib.redirect_stdout(io.StringIO()):
        res = sld.solve(sld.build_network(s)[0])
    assert res['converged'], res.get('hint')
    pcs = [g for g in res['pcs'] if g.get('p_mw') is not None]
    return {
        'res': res, 'pcs': pcs, 'vsc': res['vscs'],
        'bat_mw': sum(x['p_mw'] for x in res['sources_and_stores'] if x['kind'] == 'Battery'),
        'grid_mw': res['external_grids'][0]['p_mw'],
        'vm': {b['id']: b['vm_pu'] for b in res['buses']},
        'vdc': [d['vm_pu'] * 1200 for d in res['dc_buses'] if d['id'].startswith('B')],
    }


def _equal(pcs):
    p = [g['p_mw'] for g in pcs]
    return max(p) - min(p) < 1e-6


def test_base_case_40c(spec):
    c = _case(spec)
    assert len(c['pcs']) == 12 and _equal(c['pcs'])
    assert max(g['loading_percent'] for g in c['pcs']) == pytest.approx(94.1, abs=0.2)
    assert max(a['loading_percent'] for a in c['vsc']) < 100
    assert abs(c['bat_mw']) < 1e-4
    assert RACKS_MW / c['grid_mw'] == pytest.approx(0.926, abs=0.002)      # the spec's ~93 %
    assert c['pcs'][0]['frequency_hz'] == pytest.approx(60.0, abs=0.02)


def test_at_45c_the_rectifiers_limit_first(spec):
    """
    3,788 kVA a block, 947 kVA a rectifier - 938 kVA on its 0.99 pu bus, current-
    limited - against 949 kW its PCS draws: the GS1.1 pairs make up 0.30 MW,
    every hour it runs. The PCS, on their 505 V load bus at 1.0, at 98.8 %.
    """
    c = _case(spec, inlet=45)
    assert all(a['loading_percent'] == pytest.approx(100, abs=0.01) for a in c['vsc'])
    assert c['bat_mw'] == pytest.approx(0.30, abs=0.01)
    assert max(g['loading_percent'] for g in c['pcs']) == pytest.approx(98.8, abs=0.2)
    assert any("follows its DC bus's load to its rating" in w for w in c['res']['warnings'])


def test_at_50c_the_load_does_not_fit(spec):
    """At 50 deg C, 3,598 kVA a block: the PCS at 104 %, over their continuous rating."""
    c = _case(spec, inlet=50)
    assert min(g['loading_percent'] for g in c['pcs']) > 103
    assert c['bat_mw'] > 0.7


@pytest.mark.parametrize('inlet, loading, verdict', [(40, 142.0, 'within its short-time overload for 1 s'),
                                                     (45, 149.1, 'beyond its short-time overload (150 %)')])
def test_a_block_out(spec, inlet, loading, verdict):
    """
    Block 3's PCS out: the other eight share 11.26 MW by droop, 1.41 MW each -
    inside the 150 % / 1 s overload, not the 120 % / 10 s: within a second some
    3.3 MW of racks must go (eight PCS give 7.95 MW at 40 deg C). The island
    runs at 59.4 Hz and the backbone at 0.979 pu; the batteries carry what the
    rectifiers cannot. At 45 deg C, 149 % of its rating is 152 % of what it can
    carry on its 505 V load bus at 0.98 pu, current-limited: past even 1 s.
    """
    c = _case(spec, inlet=inlet, out_block=3)
    assert len(c['pcs']) == 8 and _equal(c['pcs'])
    assert c['pcs'][0]['loading_percent'] == pytest.approx(loading, abs=0.3)
    assert c['pcs'][0]['loading_percent'] < 150
    assert c['pcs'][0]['frequency_hz'] == pytest.approx(59.4, abs=0.05)
    assert c['vm']['BB1'] == pytest.approx(0.979, abs=0.002)
    assert c['bat_mw'] > 3.5
    assert any(verdict in w for w in c['res']['warnings'] if 'GFM' in w)


@pytest.mark.parametrize('vm_grid', [0.95, 1.05])
def test_utility_voltage_window(spec, vm_grid):
    """The 505 V grid-side buses stay inside 444-555 V; at 0.95 pu the rectifiers are current-limited."""
    c = _case(spec, vm_grid=vm_grid)
    for k, v in c['vm'].items():
        if k.endswith('_G') or k.endswith('_L'):
            assert 444 / 505 <= v <= 555 / 505, k
    if vm_grid < 1:
        assert all(a['loading_percent'] == pytest.approx(100, abs=0.01) for a in c['vsc'])
        assert c['bat_mw'] == pytest.approx(0.36, abs=0.01)
    else:
        assert abs(c['bat_mw']) < 1e-4


@pytest.mark.parametrize('soc, vdc', [(100, 1490), (1.036, 815)])
def test_dc_link_full_and_at_its_floor(spec, soc, vdc):
    """The GS1.1 full or at the 815 V floor: the DC links there, inside 815-1,500 V, the load flow unchanged."""
    c = _case(spec, soc=soc)
    assert min(c['vdc']) == pytest.approx(vdc, abs=2) and max(c['vdc']) == pytest.approx(vdc, abs=2)
    assert 814 <= min(c['vdc']) and max(c['vdc']) < 1500
    assert max(g['loading_percent'] for g in c['pcs']) == pytest.approx(94.1, abs=0.2)


def test_drawn_diagram_gives_the_specs_answers(client, quiet, spec):
    """The page's own request for the drawn spec: the same PCS power, frequency and grid power."""
    with open(os.path.join(HERE, 'reference', f'{GRID}.diagram_payload.json'), encoding='utf-8') as fh:
        payload = json.load(fh)
    assert payload['0']['frequency'] == '60'
    with quiet():
        out = json.loads(client.post('/', json=payload).get_data(as_text=True))
    assert not out.get('error'), out.get('message')
    c = _case(spec)
    assert len(out['pcs']) == 12
    for g in out['pcs']:
        assert g['p_mw'] == pytest.approx(c['pcs'][0]['p_mw'], abs=1e-5)
    assert out['externalgrids'][0]['p_mw'] == pytest.approx(c['grid_mw'], abs=1e-4)


# --- Unbalance (EMT) ---------------------------------------------------------------------------------
# The backbone is three-wire: a line-to-line load is the only unbalance. Added to the drawn diagram's
# request on backbone 2, 200 ms from the balanced load flow.

def _unbalanced(client, quiet, control, rack_pu, ll_mw=1.0):
    with open(os.path.join(HERE, 'reference', f'{GRID}.diagram_payload.json'), encoding='utf-8') as fh:
        p = json.load(fh)
    p['0'] = {'typ': 'EmtStudy Parameters', 'frequency': '60', 'time_step_us': '20', 'duration_ms': '200',
              'user_email': 't@t'}
    bb2 = next(k for k, v in p.items() if isinstance(v, dict) and v.get('userFriendlyName') == 'Load backbone 2')
    for v in p.values():
        if isinstance(v, dict) and str(v.get('typ', '')).startswith('PCS'):
            v['unbalance_control'] = 'true' if control else 'false'
            v['p_set_mw'] = str(float(v['p_set_mw']) * (rack_pu * RACKS_MW / 0.975 + ll_mw) / (RACKS_MW / 0.975))
        if isinstance(v, dict) and str(v.get('typ', '')).startswith('Load DC'):
            v['p_mw'] = str(float(v['p_mw']) * rack_pu)
    p['ul'] = {'typ': 'Asymmetric Load0', 'name': 'ul', 'id': 'cell-ul', 'userFriendlyName': 'UL', 'bus': p[bb2]['name'],
               'p_a_mw': str(ll_mw), 'p_b_mw': '0', 'p_c_mw': '0', 'q_a_mvar': '0', 'q_b_mvar': '0', 'q_c_mvar': '0',
               'sn_mva': str(ll_mw), 'scaling': '1', 'type': 'delta', 'in_service': 'true'}
    with quiet():
        out = json.loads(client.post('/', json=p).get_data(as_text=True))
    emt = out['emt']
    vuf = {b['label']: b['vuf_end_percent'] for b in emt['ac']['buses']}
    return emt, vuf


def test_unbalance_over_two_percent_without_the_control(client, quiet):
    """At the racks' cycle average (0.73 pu), 1 MW line to line on backbone 2: 2.06 % on the backbone - over 2 %."""
    emt, vuf = _unbalanced(client, quiet, control=False, rack_pu=0.73)
    assert vuf['Load backbone 2'] == pytest.approx(2.06, abs=0.05)
    assert vuf['Block 1 505 V load side'] == pytest.approx(1.44, abs=0.05)
    assert not emt['converters_blocked']


def test_unbalance_control_at_full_load(client, quiet):
    """
    The same at full load, the control on: the PCS hold their own 505 V buses
    at 0.01 % and the backbone, beyond T2, at 0.57 % - their negative sequence
    within the headroom their positive sequence leaves. Nothing blocks: given
    the negative sequence's priority, the backbone sagged to 0.67 pu and the
    SSTs, drawing constant power, blocked.
    """
    emt, vuf = _unbalanced(client, quiet, control=True, rack_pu=1.0)
    assert vuf['Load backbone 2'] < 0.7
    assert all(vuf[f'Block {k} 505 V load side'] < 0.05 for k in (1, 2, 3))
    assert not emt['converters_blocked']
    gfm = [c for c in emt['converters'] if 'GFM' in c['label']]
    assert len(gfm) == 12 and all(c['i2_end_ka'] > 0.08 for c in gfm)


# --- Phase 24: faults and earthing ----------------------------------------------------------------
# IEC 60909 on the drawn diagram: the islanded backbone's PCS as current sources at their limit (one of
# them, in the study, a source with no zero sequence); ANSI on the same request; the DC fault study on a
# quadrant's DC bus; arc flash by voltage class.

IEC = {'typ': 'ShortCircuitPandaPower Parameters', 'fault_location': 'max', 'fault_bus_mode': 'all',
       'fault_bus_ids': [], 'fault_bus_names': [], 'fault_impedance': '6', 'frequency': '60', 'topology': 'auto',
       'tk_s': '1', 'r_fault_ohm': '0', 'x_fault_ohm': '0', 'inverse_y': 'True', 'user_email': 't@t'}
I_LOAD_KA = 11.1 / (3 ** 0.5 * 34.5)          # the backbone's load current: 0.186 kA


def _drawn_study(client, quiet, params, k_pcs=None, x_afe=None):
    with open(os.path.join(HERE, 'reference', f'{GRID}.diagram_payload.json'), encoding='utf-8') as fh:
        p = json.load(fh)
    for v in p.values():
        if isinstance(v, dict) and k_pcs and str(v.get('typ', '')).startswith('PCS'):
            v['current_limit_pu'] = str(k_pcs)
        if isinstance(v, dict) and x_afe and str(v.get('typ', '')).startswith('VSC'):
            v['x_ohm'] = str(x_afe)
    label = {v['name']: v.get('userFriendlyName') for v in p.values() if isinstance(v, dict) and 'name' in v}
    p['0'] = params
    with quiet():
        out = json.loads(client.post('/', json=p).get_data(as_text=True))
    assert not out.get('error'), out.get('message')
    return out, label, p


def _bus_ka(out, label):
    return {label[r['name']]: r.get('ikss_ka', r.get('i_first_sym_ka')) for r in out['busbars']}


@pytest.mark.parametrize('k, backbone_a', [(1.2, 239), (1.5, 299)])
def test_three_phase_faults(client, quiet, k, backbone_a):
    """
    The 505 V grid side: 68 kA through T1 at 6 % (the spec's ~70 kA). The POI:
    8.37 kA, the 500 MVA utility. The backbone: the twelve PCS at their limit,
    k x 994 kVA each - 239 A at 1.2 pu, 299 A at 1.5 pu (the spec's ~95 A a
    block for 1 s) - only 1.3-1.6 times the backbone's 186 A of load: no
    overcurrent setting tells a fault from the load (differential, directional
    and undervoltage elements, as the spec says).
    """
    out, label, _ = _drawn_study(client, quiet, {**IEC, 'fault_type': '3ph'}, k_pcs=k)
    ka = _bus_ka(out, label)
    for b in (1, 2, 3):
        assert ka[f'Block {b} 505 V grid side'] == pytest.approx(68.2, abs=0.3)
        assert ka[f'Load backbone {b}'] * 1e3 == pytest.approx(backbone_a, abs=2)
    assert ka['Utility 34.5 kV POI'] == pytest.approx(8.37, abs=0.02)
    assert 1.25 < ka['Load backbone 1'] / I_LOAD_KA < 1.65


@pytest.mark.parametrize('k, backbone_a', [(1.2, 312), (1.5, 369), (None, 7.6)])
def test_earth_faults(client, quiet, k, backbone_a):
    """
    The backbone's earth faults by IEC 60909: 312 A at the inverters' 1.2 pu,
    369 A at 1.5 pu - the two zigzags' (EMT, which builds their windings:
    427 A at 1.2 pu). With both zigzags out, 7.6 A: the cables' capacitance
    alone (EMT: 6.5 A). Before Phase 31 IEC read 249-311 A with them and 228 A
    without: pandapower added the inverters' current-source share whatever the
    zero sequence, and the island's eleven other PCS as current sources
    through the twelfth's impedance. Now each grid-forming PCS is a source
    behind its limit's impedance for an earth fault, and a current source's
    share is IEC's 3 Z(1) / (2 Z(1) + Z(0)) (sc_current_sources). The 505 V
    buses none: the grid side a delta, the load side T2's star left floating,
    as the spec asks (solidly grounded it gave 17 kA). The POI: the utility's own.
    """
    if k is None:
        with open(os.path.join(HERE, 'reference', f'{GRID}.diagram_payload.json'), encoding='utf-8') as fh:
            p = json.load(fh)
        for v in p.values():
            if isinstance(v, dict) and v.get('userFriendlyName') in ('Zigzag breaker', 'Zigzag 2 breaker'):
                v['closed'] = 'false'
        label = {v['name']: v.get('userFriendlyName') for v in p.values() if isinstance(v, dict) and 'name' in v}
        p['0'] = {**IEC, 'fault_type': '1ph'}
        with quiet():
            out = json.loads(client.post('/', json=p).get_data(as_text=True))
    else:
        out, label, _ = _drawn_study(client, quiet, {**IEC, 'fault_type': '1ph'}, k_pcs=k)
    ka = _bus_ka(out, label)
    for b in (1, 2, 3):
        assert ka[f'Load backbone {b}'] * 1e3 == pytest.approx(backbone_a, rel=0.02)
        assert ka[f'Load backbone {b}'] < 0.4
        assert ka[f'Block {b} 505 V grid side'] == pytest.approx(0, abs=1e-6)
        # Its PCS's own source's zero sequence, 1e6 times its X(1): a trace, some 0.02 A.
        assert ka[f'Block {b} 505 V load side'] == pytest.approx(0, abs=1e-4)
    assert ka['Utility 34.5 kV POI'] == pytest.approx(8.40, abs=0.02)


def test_ansi_earth_fault_on_the_backbone(client, quiet):
    from test_reference_grids import ANSI_PARAMS
    out, label, _ = _drawn_study(client, quiet, {**ANSI_PARAMS, 'fault_type': '1ph', 'frequency_hz': 60, 'user_email': 't@t'})
    ka = _bus_ka(out, label)
    assert 0.2 < ka['Load backbone 2'] < 0.4
    assert ka['Block 1 505 V load side'] == pytest.approx(0, abs=1e-4)


@pytest.mark.parametrize('x_afe, afe_ka, total_ka', [(None, 8.68, 67.3), (0.00943, 26.3, 84.9)])
def test_dc_bus_fault_on_a_quadrant(client, quiet, x_afe, afe_ka, total_ka):
    """
    A bolted fault on Block 1's DC1. The two quadrant DC links discharge first,
    some 400 kA each behind their 0.1 mOhm / 0.05 uH. Then, sustained: the GS1.1
    pair 51 kA (25.7 kA a unit at the assumed 50 mOhm - its fuses must break
    that at up to 1,500 V); the rectifier's diodes from the grid 8.7 kA behind
    its 0.15 pu reactor - 26 kA behind the spec's illustrative 150 uH a module;
    the inverter's diodes 7.2 kA from the 505 V load bus, which the other eleven
    PCS hold at their limits. The spec's ~70 kA counts T1 alone: the batteries,
    not the grid, carry most of it.
    """
    from test_dc_fault import STUDY
    with open(os.path.join(HERE, 'reference', f'{GRID}.diagram_payload.json'), encoding='utf-8') as fh:
        cells = {v.get('userFriendlyName'): v['name'] for v in json.load(fh).values() if isinstance(v, dict) and 'name' in v}
    out, _, _ = _drawn_study(client, quiet, {**STUDY, 'fault_bus': cells['Block 1 DC1'], 'duration_ms': '20',
                                             'frequency': '60'}, x_afe=x_afe)
    (f,) = out['dcfault']['faults']
    c = {(x['kind'], x['name']): x for x in f['contributions']}
    bat = [x for (k, n), x in c.items() if k == 'Battery' and n.startswith('Block 1 DC1')]
    assert len(bat) == 2 and all(x['ik_ka'] == pytest.approx(25.70, abs=0.05) for x in bat)
    assert c[('VSC (diodes, blocked)', 'Block 1 AFE 1 (6 x PCS157SE)')]['ik_ka'] == pytest.approx(afe_ka, abs=0.1)
    assert c[('PCS (diodes, blocked)', 'Block 1 GFM 1 (6 x PCS157SE)')]['ik_ka'] == pytest.approx(7.18, abs=0.1)
    for kind, name in (('VSC DC-link capacitor', 'Block 1 AFE 1 (6 x PCS157SE)'),
                       ('PCS DC-link capacitor', 'Block 1 GFM 1 (6 x PCS157SE)')):
        assert 350 < c[(kind, name)]['ip_ka'] < 450
    assert f['ik_ka'] == pytest.approx(total_ka, abs=0.3)
    for (k, n), x in c.items():
        if n.startswith(('Block 2', 'Block 3')) or ('AFE' in n and not n.startswith('Block 1 AFE 1')):
            assert abs(x['ik_ka']) < 1e-3, n


def test_arc_flash_by_voltage_class(client, quiet):
    """
    0.2 s clearing: the 505 V grid side, 68 kA bolted, 12.8 cal/cm2 (PPE 3); the
    505 V load side, inverter-fed, 3.9 cal/cm2; the backbone, by Ralph Lee at its
    0.24 kA, about 1 cal/cm2.
    """
    out, _, _ = _drawn_study(client, quiet, {
        'typ': 'ArcFlashPandaPower Parameters', 'electrode_config': 'VCB', 'equipment_mode': 'by_voltage',
        'working_distance_mm': '455', 'conductor_gap_mm': '25', 'enclosure_height_mm': '508',
        'enclosure_width_mm': '508', 'enclosure_depth_mm': '508', 'clearing_time_s': '0.2',
        'clearing_time_min_s': '0.2', 'frequency': '60', 'user_email': 't@t'})
    rows = {r['name']: r for r in out['arc_flash']}
    for b in (1, 2, 3):
        g = rows[f'Block {b} 505 V grid side']
        assert g['incident_energy_cal_cm2'] == pytest.approx(12.8, abs=0.2) and str(g['ppe_category']) == '3'
        assert 3.5 < rows[f'Block {b} 505 V load side']['incident_energy_cal_cm2'] < 4.5
        assert rows[f'Load backbone {b}']['incident_energy_cal_cm2'] < 1.5


# --- Phase 25: dynamics (ANDES) -----------------------------------------------------------------------
# ANDES on the drawn diagram: the backbone an island of twelve grid-forming PCS (virtual machines), the
# SSTs' MV draw following the DGX H100 profile - a 0.1 -> 1.0 pu restart at 7 s - the PCS's DC side
# (rectifier and GS1.1) holding through the run.

def _andes(client, quiet, tf, line=''):
    with open(os.path.join(HERE, 'reference', f'{GRID}.diagram_payload.json'), encoding='utf-8') as fh:
        p = json.load(fh)
    with open(SPEC_PATH, encoding='utf-8') as fh:
        lp = {x['id']: {'name': x['name'], 'kind': x['kind'], 'dt': x['dt_s'], 't0': 0, 'p': x['values']}
              for x in json.load(fh)['load_profiles']}
    cell = {v.get('userFriendlyName'): v['name'] for v in p.values() if isinstance(v, dict) and 'name' in v}
    p['0'] = {'typ': 'TransientStabilityAndes Parameters', 'frequency': '60', 'sn_mva': '100', 'tf': str(tf),
              'tstep': '0', 'fault_bus': '', 'fault_enabled': 'false', 'toggle_line': cell.get(line, ''),
              'toggle_t': '8.0', 'toggle_gen': '', 'poi_bus': cell['Utility 34.5 kV POI'],
              'grid_voltage_profile': 'none', 'load_profiles': lp, 'user_email': 't@t'}
    with quiet():
        out = json.loads(client.post('/', json=p).get_data(as_text=True))
    assert not out.get('error'), out.get('message')
    return out


def _at(g, key, t, times):
    import numpy as np
    return float(np.interp(t, np.array(times), np.array(g[key])))


def test_andes_restart_step_shared_by_droop(client, quiet):
    """
    The island starts at the profile's 0.8 pu, its load shared by droop (each PCS
    0.746 MW at 60 Hz - the holding one too: left alone it absorbed 1.2 MW and
    sat at its limit); at 0.1 pu before the restart, 60.6 Hz; after it, 0.91 MW
    each at 59.9 Hz, well within their current - 0.92 pu at most.
    """
    out = _andes(client, quiet, 10)
    pcs = out['pcs']
    assert len(pcs) == 12
    for g in pcs:
        assert _at(g, 'p_mw', 0.0, out['time']) == pytest.approx(0.746, abs=0.005)
        assert _at(g, 'frequency_hz', 0.0, out['time']) == pytest.approx(60.0, abs=0.01)
        assert 60.35 < _at(g, 'frequency_hz', 6.95, out['time']) < 60.7      # the load at 0.1 pu, over its set point
        assert _at(g, 'p_mw', 7.05, out['time']) == pytest.approx(0.89, abs=0.03)
        assert _at(g, 'frequency_hz', 7.05, out['time']) == pytest.approx(59.88, abs=0.04)
        assert g['peak_current_pu'] < 0.95 and g['current_limited_from_s'] is None


def test_andes_a_block_tripped_at_8_s(client, quiet):
    """
    Block 3's T2 out at 8 s, the racks at 0.85 pu: the other eight PCS take
    1.16 MW each, 117 % - inside their 120 % for 10 s - at 59.5 Hz; past 10 s
    over their rating their ceiling falls to it and the island sags further:
    load must come off within 10 s of losing a block.
    """
    out = _andes(client, quiet, 20, line='Block 3 T2 0.505/34.5 kV Dy1 (505 V star floating)')
    left = [g for g in out['pcs'] if not g['label'].startswith('Block 3')]
    assert len(left) == 8
    for g in left:
        assert _at(g, 'p_mw', 8.5, out['time']) == pytest.approx(1.16, abs=0.02)
        assert _at(g, 'frequency_hz', 8.5, out['time']) == pytest.approx(59.5, abs=0.03)
        assert 1.15 < _at(g, 'current_pu', 8.5, out['time']) < 1.25
        assert max(g['current_limit_series_pu']) == pytest.approx(1.5) and min(g['current_limit_series_pu']) == pytest.approx(1.0)
    assert any("reached its short-time overload's ceiling" in w for w in out['warnings'])


# --- Phase 25: the SSTs' constant power against the blocks' output impedance ---------------------

def _backbone_source(blocks):
    """The blocks' output impedance at the backbone: each PCS's 0.15 pu / 0.01 pu reactor and T2, in parallel."""
    import math
    zb_pcs = 34.5e3 ** 2 / (4 * 0.994e6)                  # a block's four PCS at 40 deg C, on 34.5 kV
    zb_t2 = 34.5e3 ** 2 / 3.75e6
    r = (0.01 * zb_pcs + 0.008 * zb_t2) / blocks
    x = (0.15 * zb_pcs + 0.06 * zb_t2) / blocks
    return r, x / (2 * math.pi * 60)


@pytest.mark.parametrize('blocks, f_c_max', [(3, 151), (2, 101)])
def test_constant_power_ssts_against_the_blocks(blocks, f_c_max):
    """
    The SSTs, 11.1 MW at 34.5 kV, are a negative conductance P / V^2 out to the
    bandwidth of what sets their input power - their rectifiers' 30 Hz DC-link
    voltage loop. Against the blocks' reactors and T2s the minor loop peaks at
    0.10 (20 dB of margin; 0.15 with a block out). The loop reaches -1 if the
    SSTs' power control is faster than ~300 Hz: under 151 Hz for 6 dB (101 Hz
    with a block out). The spec's Nyquist check passes, with a bandwidth to keep.
    """
    import impedance_check as ic
    r, l = _backbone_source(blocks)
    v = ic.check(r, l, 11.1e6, 34.5e3, f_c_hz=30.0)
    assert v['stable'] and v['peak_gain'] == pytest.approx(0.10 * 3 / blocks, rel=0.08)
    assert v['f_c_max_hz'] == pytest.approx(f_c_max, rel=0.03)
    fast = ic.check(r, l, 11.1e6, 34.5e3, f_c_hz=4 * f_c_max)
    assert not fast['stable']


# --- Phase 25: dynamics (EMT) ---------------------------------------------------------------------------

def _emt(client, quiet, **params):
    with open(os.path.join(HERE, 'reference', f'{GRID}.diagram_payload.json'), encoding='utf-8') as fh:
        p = json.load(fh)
    cell = {v.get('userFriendlyName'): v['name'] for v in p.values() if isinstance(v, dict) and 'name' in v}
    params = {k: (cell[v] if isinstance(v, str) and v in cell else v) for k, v in params.items()}
    p['0'] = {'typ': 'EmtStudy Parameters', 'frequency': '60', 'time_step_us': '20', 'user_email': 't@t', **params}
    with quiet():
        out = json.loads(client.post('/', json=p).get_data(as_text=True))
    emt = out['emt']
    return emt, {b['label']: b for b in emt['ac']['buses']}, {c['label']: c for c in emt['converters']}


def test_emt_restart_step(client, quiet):
    """
    The profile from 6.8 s: the racks' 0.1 -> 1.0 pu restart in 100 ms. The PCS
    carry it - 0.70 -> 0.88 MW, 82 % of their current limit - the island dips to
    59.8 Hz and the backbone to 0.97 pu; the GS1.1 take the step on the DC links
    (the rectifiers, ramp-limited, hold their power over the run). Nothing blocks.
    """
    with open(SPEC_PATH, encoding='utf-8') as fh:
        lp = {x['id']: {'name': x['name'], 'kind': x['kind'], 'dt': x['dt_s'], 't0': 0, 'p': x['values']}
              for x in json.load(fh)['load_profiles']}
    emt, bus, conv = _emt(client, quiet, duration_ms='400', load_profiles=lp, profile_start_s='6.8', profile_repeat='true')
    assert not emt['converters_blocked']
    g = conv['Block 1 GFM 1 (6 x PCS157SE)']
    assert g['p_start_mw'] == pytest.approx(0.70, abs=0.02) and g['p_end_mw'] == pytest.approx(0.88, abs=0.02)
    assert 59.7 < g['f_min_hz'] < 59.9 and g['i_peak_ka'] < g['current_limit_ka']
    assert bus['Load backbone 2']['v_rms_min_pu'] > 0.96
    a = conv['Block 1 AFE 1 (6 x PCS157SE)']
    assert a['p_end_mw'] == pytest.approx(a['p_start_mw'], abs=1e-3)
    ders = {d['label']: d for d in emt['ders']}
    assert ders['Block 1 DC1 GS1.1 A']['p_end_mw'] > 0.05


def test_emt_backbone_fault_drops_the_ssts(client, quiet):
    """
    A bolted three-phase fault on backbone 2 for 100 ms: the PCS hold at their
    current limit through it and the backbone recovers after; but the SSTs,
    their MV side at zero, discharge their 50 kV links past their 80 % block in
    2 ms and stay out - the racks lose supply. Ride-through of a backbone fault
    is the SSTs' to give (their links, their restart), not the blocks'.
    """
    emt, bus, conv = _emt(client, quiet, duration_ms='300', ac_fault_bus='Load backbone 2', ac_fault_type='abcg',
                          ac_fault_resistance_ohm='0.001', ac_fault_time_ms='20', ac_fault_duration_ms='100')
    blocked = {b['label'] for b in emt['converters_blocked']}
    assert {f'Hall {k} SST rectifier' for k in (1, 2, 3)} <= blocked
    assert min(b['t_ms'] for b in emt['converters_blocked']) < 25
    assert not any('GFM' in b or 'AFE' in b for b in blocked)
    assert conv['Block 1 GFM 1 (6 x PCS157SE)']['limited_ms'] > 90
    assert bus['Load backbone 2']['v_rms_final_pu'] > 0.95


def test_emt_utility_single_phase_fault_ridden_through(client, quiet):
    """
    Phase a to ground at the utility POI for 100 ms: the 505 V grid side dips to
    0.48 pu, the rectifiers current-limit some 70 ms and ride through, the DC
    links sag 1 %; across the double conversion the backbone keeps 0.98 pu and
    0.01 % unbalance - the racks never see the utility's fault.
    """
    emt, bus, conv = _emt(client, quiet, duration_ms='300', ac_fault_bus='Utility 34.5 kV POI', ac_fault_type='ag',
                          ac_fault_resistance_ohm='0.001', ac_fault_time_ms='20', ac_fault_duration_ms='100')
    assert not emt['converters_blocked']
    assert bus['Block 1 505 V grid side']['v_rms_min_pu'] == pytest.approx(0.48, abs=0.03)
    a = conv['Block 1 AFE 1 (6 x PCS157SE)']
    assert 50 < a['limited_ms'] < 100 and a['v_dc_min_kv'] > 1.26
    assert bus['Load backbone 2']['v_rms_min_pu'] > 0.97 and bus['Load backbone 2']['vuf_end_percent'] < 0.1


# --- Phase 26: time series and energy ---------------------------------------------------------------

def test_usable_energy_above_the_815_v_floor(spec):
    """
    The spec's open question: how much of the GS1.1's 750-1,500 V window is
    usable above the PCS's 815 V floor. At 947 kW a bus - 581 A a unit at 815 V
    - its 50 + 20 mOhm (R0 and its settled RC pair) drop 41 V: the floor is
    reached at an OCV of 856 V, 1.7 % charge on the generic NFPP curve. Some
    98 % of the 3.1 MWh is usable at full power; Peak's own curve decides it.
    """
    import der_electrisim as der
    import numpy as np
    row = next(b for b in spec['batteries'] if b['id'] == 'B1_GS1a')
    gs = der.build({'typ': 'Battery0', 'name': 'gs', **{k: str(v) for k, v in row.items() if k not in ('id', 'name', 'bus')}})
    i = 0.947e6 / 2 / 815.0
    socs = np.linspace(0, 0.2, 20001)
    v = np.array([gs.ocv(s) - (gs.r0 + gs.r1) * i for s in socs])
    soc_floor = float(socs[np.argmax(v >= 815.0)])
    assert gs.r0 + gs.r1 == pytest.approx(0.070)
    assert gs.ocv(soc_floor) == pytest.approx(815 + 0.070 * i, abs=0.5)
    assert soc_floor == pytest.approx(0.0174, abs=0.001)
    assert 1 - soc_floor > 0.98


def _rack_buffers(p):
    """The spec's Option B at each hall: 93 Wh a rack (225 modules of 130 F, 15 in series), tau 10 s, a smoothing DC/DC."""
    cell = {v.get('userFriendlyName'): v['name'] for v in p.values() if isinstance(v, dict) and 'name' in v}
    for n in (1, 2, 3):
        p[f'scb{n}'] = {'typ': f'DC Bus9{n}', 'name': f'scb{n}', 'id': f'scb{n}', 'vn_kv': '0.81', 'in_service': 'true',
                        'userFriendlyName': f'Hall {n} buffer 0.81 kV'}
        p[f'sc{n}'] = {'typ': f'Supercapacitor9{n}', 'name': f'sc{n}', 'id': f'sc{n}', 'userFriendlyName': f'Hall {n} rack buffer',
                       'bus': f'scb{n}', 'coupling': 'converter', 'sizing': 'modules', 'module_c_f': '130',
                       'module_v': '54', 'module_esr_mohm': '4', 'module_esl_uh': '0', 'modules_series': '15',
                       'strings_parallel': '15', 'v0_percent': '79', 'v_min_percent': '50', 'p_rated_kw': '0',
                       'r_leak_ohm': '10000', 'in_service': 'true'}
        p[f'scc{n}'] = {'typ': f'DC/DC Converter9{n}', 'name': f'scc{n}', 'id': f'scc{n}',
                        'userFriendlyName': f'Hall {n} buffer converter', 'bus_in': f'scb{n}',
                        'bus_out': cell[f'Hall {n} 800 V DC'], 'control_mode': 'smoothing', 'rated_mw': '3.6',
                        'vn_in_kv': '0.81', 'vn_out_kv': '0.8', 'efficiency_percent': '98', 'no_load_loss_kw': '1',
                        'bidirectional': 'true', 'smoothing_tau_s': '10', 'soc_ref_percent': '50', 'soc_gain': '0.1',
                        'in_service': 'true'}


def _timeseries(client, quiet, steps, dt_s, outage=False, buffers=False):
    with open(os.path.join(HERE, 'reference', f'{GRID}.diagram_payload.json'), encoding='utf-8') as fh:
        p = json.load(fh)
    if buffers:
        _rack_buffers(p)
    with open(SPEC_PATH, encoding='utf-8') as fh:
        lp = {x['id']: {'name': x['name'], 'kind': x['kind'], 'dt': x['dt_s'], 't0': 0, 'p': x['values']}
              for x in json.load(fh)['load_profiles']}
    if outage:
        for v in p.values():
            if isinstance(v, dict) and str(v.get('typ', '')).startswith('External Grid'):
                v['in_service'] = 'false'
    p['0'] = {'typ': 'TimeSeriesSimulationPandaPower Parameters', 'time_steps': str(steps), 'time_step_s': str(dt_s),
              'load_profiles': lp, 'load_profile': 'constant', 'generation_profile': 'constant',
              'profile_mode': 'custom', 'element_profiles': {}, 'frequency': '60', 'algorithm': 'nr',
              'calculate_voltage_angles': 'auto', 'init': 'auto', 'microgrid_dispatch': 'True', 'user_email': 't@t'}
    with quiet():
        out = json.loads(client.post('/', json=p).get_data(as_text=True))
    assert not out.get('error'), out.get('message')
    return out['microgrid']


def test_training_cycle_the_rectifiers_hold_the_average(client, quiet):
    """
    30 s at 0.5 s steps: the restart at 7 s and a full 21 s training cycle.
    Each quadrant's demand swings 0.14-0.88 MW (the steps' means) about its
    0.72 MW average, ramping 1.4 MW/s; its rectifier, through its 60 s low-pass,
    stays at 0.74-0.77 MW, ramping 0.01 MW/s - the grid sees a flat load. The
    GS1.1 pair takes the swing, +0.15 / -0.61 MW (+0.21 / -0.66 MW at 0.1 s
    steps; the spec's +256 / -597 kW about the cycle's mean).
    """
    mg = _timeseries(client, quiet, 60, 0.5)
    rects = mg['rectifiers']
    assert len(rects) == 12
    for r in rects:
        assert r['tau_s'] == 60
        assert r['demand_mean_mw'] == pytest.approx(0.72, abs=0.01)
        assert r['demand_max_mw'] - r['demand_min_mw'] > 0.7
        assert r['rectifier_max_mw'] - r['rectifier_min_mw'] < 0.04
        assert r['rectifier_ramp_mw_s'] < 0.02 < 1.0 < r['demand_ramp_mw_s']
        assert r['batteries_max_mw'] == pytest.approx(0.145, abs=0.02)
        assert r['batteries_min_mw'] == pytest.approx(-0.608, abs=0.02)
    assert mg['unserved_mwh'] == pytest.approx(0.0, abs=1e-9)


def test_utility_outage_ride_through(client, quiet):
    """
    The utility out, 30 min steps, the racks at their cycle's average: the
    rectifiers have no supply and the GS1.1 carry every quadrant, 0.32 MW a
    unit. From 80 % they last 7-7.5 h - the spec's 6.5 h is at full block load
    from full - their DC links falling from 1,267 V as they empty; then the
    island sheds its load. Their energy adds up: delivered at their terminals
    what their cells gave, less their losses.
    """
    mg = _timeseries(client, quiet, 20, 1800, outage=True)
    gs = [s for s in mg['stores'] if s['kind'] == 'Battery']
    assert len(gs) == 24
    for s in gs:
        assert s['stored_start_mwh'] == pytest.approx(0.8 * 3.1 * 0.981, abs=0.05)
        assert s['soc_min_percent'] < 1.0
        assert 0.99 * s['drawn_mwh'] <= s['delivered_mwh'] <= s['drawn_mwh']
    rows = [r for r in mg['ders'] if r['label'] == 'Block 1 DC1 GS1.1 A']
    carried = [r['time_step'] for r in rows if r['p_mw'] > 0.3]
    assert 14 <= len(carried) <= 15                 # 7-7.5 h at 30 min a step
    assert rows[1]['v_kv'] < rows[0]['v_kv'] - 0.01
    short = [s for s in mg['steps'] if s['unserved_mw'] > 1.0]
    assert short and min(s['time_step'] for s in short) >= 15


def test_rack_buffers_take_the_swing_from_the_gs11(client, quiet):
    """
    The spec's Option B on each hall's 800 V bus. Each quadrant's demand now
    swings some 0.2 pu about its average - the spec's 0.61-0.80 pu upstream -
    not 0.1-0.9; the GS1.1 pair's swing falls about fourfold (to some
    +0.03 / -0.16 MW at 0.1 s steps) and with it their swing wear, the spec's
    0.36 -> 0.08 equivalent cycles a day. The supercapacitors' converters
    never reach their 3.6 MW, nor their stores their window.
    """
    mg = _timeseries(client, quiet, 60, 0.5, buffers=True)
    for r in mg['rectifiers']:
        assert r['demand_max_mw'] - r['demand_min_mw'] < 0.25
        assert r['batteries_min_mw'] > -0.25 and r['batteries_max_mw'] < 0.08
    sm = mg['smoothing']
    assert len(sm) == 3
    for s in sm:
        assert s['limited_steps'] == 0 and s['store_peak_mw'] < 3.6
        assert s['feed_ramp_mw_s'] < 0.1 * s['rack_ramp_mw_s']


# --- Protection (Phase 27) ---------------------------------------------------------------------
# The drawn diagram's breakers: each transformer's both sides, each backbone cable's two ends, each
# hall feeder at the backbone, the zigzag's; a DC breaker in front of each GS1.1. The spec carries the
# relays (87L on the backbone cables, 87T on T2, definite time on the hall feeders, overcurrent set
# automatically on the rest), and the page's protection study request
# (reference_pcs3750_mv.diagram_protection_payload.json) is the drawing's, with them. Other schemes
# replace them, as a protection engineer would on the diagram.

def _protection(client, quiet, scheme=None, fault='3ph', mode='line', bus=None, k_pcs=None):
    with open(os.path.join(HERE, 'reference', f'{GRID}.diagram_protection_payload.json'), encoding='utf-8') as fh:
        p = json.load(fh)
    cells = {v.get('userFriendlyName'): v['name'] for v in p.values() if isinstance(v, dict) and 'name' in v}
    for v in p.values():
        if isinstance(v, dict) and scheme and str(v.get('typ', '')).startswith('Switch'):
            v.update(scheme(v['userFriendlyName']) or {'protection_type': 'none'})
        if isinstance(v, dict) and k_pcs and str(v.get('typ', '')).startswith('PCS'):
            v['current_limit_pu'] = str(k_pcs)
    p['0'] = {**p['0'], 'fault_type': fault, 'fault_location_mode': mode, 'sc_line_id': 'all',
              'fault_bus_id': cells.get(bus, ''), 'user_email': 't@t'}
    with quiet():
        out = json.loads(client.post('/', json=p).get_data(as_text=True))
    assert not out.get('error'), out.get('message')
    return out


OCR_AUTO = {'protection_type': 'ocr', 'curve_type': 'standard_inverse', 'tms': '1.0', 't_grade': '0.3',
            't_gg': '0.1', 't_g': '0.5', 't_diff': '0.3', 'pickup_mode': 'auto'}


def test_the_drawing_carries_the_specs_relays(spec):
    """Each switch's protection in the spec is on the drawn diagram's switch, as its Protection tab holds it."""
    with open(os.path.join(HERE, 'reference', f'{GRID}.diagram_protection_payload.json'), encoding='utf-8') as fh:
        drawn = {v['userFriendlyName']: v for v in json.load(fh).values()
                 if isinstance(v, dict) and str(v.get('typ', '')).startswith('Switch')}
    assert len(drawn) == len(spec['switches']) == 23
    for row in spec['switches']:
        sw = drawn[row['name']]
        assert sw['protection_type'] == row.get('protection_type', 'none'), row['name']
        for key, value in row.items():
            if key not in ('id', 'bus', 'element', 'et', 'name'):
                assert sw[key] == value or float(sw[key]) == pytest.approx(value), (row['name'], key)
    assert drawn['Cable BB1-BB2 breaker at BB1']['I_diff_a'] == '60'
    assert drawn['Hall 2 feeder breaker']['pickup_mode'] == 'manual'


def _trips(scenario):
    return {t['user_friendly_name']: t for t in scenario['trip']}


def test_overcurrent_cannot_see_a_backbone_fault(client, quiet):
    """
    Overcurrent on every breaker, set automatically (rated current x 1.2 x
    1.25): a fault on any backbone cable, 238 A from the twelve PCS at their
    limit, trips nothing - each T2 brings 79 A against its 75 A pickup, so
    nearly at its rating, and the cables 119 A against 720 A. Nothing clears
    it; nor any hall feeder's.
    """
    out = _protection(client, quiet, lambda n: None if 'Zigzag' in n else OCR_AUTO)
    assert out['summary']['n_scenarios'] == 6 and out['summary']['n_tripped'] == 0
    for sc in out['scenarios']:
        assert sc['short_circuit']['ikss_ka'] == pytest.approx(0.238, abs=0.002)
        assert sc['clearing_time_s'] is None
        t = _trips(sc)
        for b in (1, 2, 3):
            hv = t[f'Block {b} T2 HV breaker']
            assert 1.0 < hv['ikss_ka'] * 1e3 / hv['evaluation']['pickup_a'] < 1.1


@pytest.mark.parametrize('fault, k, ik_a', [('3ph', 1.2, 238), ('3ph', 1.5, 298), ('1ph', 1.2, 300)])
def test_line_differential_clears_each_backbone_cable(client, quiet, fault, k, ik_a):
    """
    The spec's scheme. 87L at each end of each backbone cable (60 A, 30 %
    slope, 50 ms with its signalling): a fault on a cable, fed from both ends, puts its whole
    current in the zone - both ends trip; the other cables and each T2's 87T
    carry it through and restrain. A fault on a hall feeder: its breaker's
    overcurrent, 0.1 s. Nothing miscoordinates or trips needlessly. The earth
    faults, the zigzag's zero sequence, likewise (pandapower gives their branch
    currents no angles: magnitudes, added inside the zone). The inverters'
    own undervoltage trip on each T2's 505 V breaker (Phase 28: 0.5 pu for
    0.5 s) sees its bus at 0.08 pu in a three-phase fault, at either limit: the
    backup, which takes the whole island off. An earth fault it does not see
    (nor can the study: pandapower gives no phase voltages for one).
    """
    out = _protection(client, quiet, fault=fault, k_pcs=k)
    s = out['summary']
    assert s['n_scenarios'] == 6 and s['n_miscoordination'] == 0 and s['n_unwanted_trips'] == 0
    for sc in out['scenarios']:
        assert sc['short_circuit']['ikss_ka'] * 1e3 == pytest.approx(ik_a, abs=2)
        t = _trips(sc)
        tripped = {n for n, r in t.items() if r['tripped'] and r['t_trip_s'] < 0.3}
        backup = {n for n, r in t.items() if r['tripped'] and r['t_trip_s'] >= 0.3}
        assert backup == ({f'Block {b} T2 LV breaker' for b in (1, 2, 3)} if fault == '3ph' else set())
        for n in backup:
            assert t[n]['t_trip_s'] == 0.5 and t[n]['evaluation']['vm_pu'] < 0.1
        line = sc['fault_label'].split(',')[0]
        if line.startswith('Backbone cable'):
            a, b = line.split()[-1].split('-')
            assert tripped == {f'Cable {a}-{b} breaker at {a}', f'Cable {a}-{b} breaker at {b}'}
            assert sc['clearing_time_s'] == 0.05
            for n in tripped:
                assert t[n]['evaluation']['i_operate_a'] == pytest.approx(ik_a, abs=2)
                assert t[n]['evaluation']['phasors'] == (fault == '3ph')
        else:
            assert tripped == {f'{line} breaker'} and sc['clearing_time_s'] == 0.1
        for n, r in t.items():
            if n not in tripped and (r.get('evaluation') or {}).get('i_operate_a') is not None:
                assert r['evaluation']['i_operate_a'] < 1.0, n


def test_definite_time_earth_fault_cannot_tell_the_ring_cables_apart(client, quiet):
    """
    51N alone: 40 A, the feeders at 0.1 s and the ring at 0.4 s. A feeder's
    earth fault is cleared and graded; but a fault on a backbone cable sends
    the zigzag's current round both sides of the ring - over 40 A in the
    healthy cables too - and their relays trip with the faulted cable's.
    """
    def scheme(name):
        if 'feeder' in name:
            return {'protection_type': 'earth_fault', 'I_e_a': '40', 't_e': '0.1'}
        if name.startswith('Cable'):
            return {'protection_type': 'earth_fault', 'I_e_a': '40', 't_e': '0.4'}
        return None
    out = _protection(client, quiet, scheme, fault='1ph')
    unwanted = {(u['fault_label'].split(',')[0], u['user_friendly_name']) for u in out['unwanted_trips']}
    assert out['summary']['n_unwanted_trips'] == 12
    assert all(f.startswith('Backbone cable') for f, _ in unwanted)
    for sc in out['scenarios']:
        if 'feeder' in sc['fault_label']:
            assert sc['clearing_time_s'] == 0.1


def test_utility_side_overcurrent_grades(client, quiet):
    """
    A fault on Block 1's 505 V grid side, 68.6 kA from the utility through T1:
    the LV breaker's overcurrent clears it at 0.5 s and the HV breaker's backs
    it up at 0.8 s, the 0.3 s margin; the other blocks' T1s see nothing (their
    rectifiers do not feed an AC fault). 0.5 s at 68.6 kA is the arc-flash
    case: an instantaneous element would shorten it.
    """
    out = _protection(client, quiet, lambda n: OCR_AUTO if ' T1 ' in n else None, mode='bus',
                      bus='Block 1 505 V grid side')
    (sc,) = out['scenarios']
    assert sc['short_circuit']['ikss_ka'] == pytest.approx(68.56, abs=0.1)
    t = _trips(sc)
    assert sc['primary_switches'] == ['Block 1 T1 LV breaker']
    assert t['Block 1 T1 LV breaker']['t_trip_s'] == pytest.approx(0.5)
    assert t['Block 1 T1 HV breaker']['t_trip_s'] == pytest.approx(0.8)
    assert out['summary']['n_miscoordination'] == 0
    assert not any(r['tripped'] for n, r in t.items() if not n.startswith('Block 1'))


def test_gs11_breakers_against_a_quadrant_bus_fault(client, quiet):
    """
    The DC fault on Block 1's DC1 with each GS1.1 behind its own breaker
    (mechanical, 50 kA, opening in 20 ms - assumed). Each breaks its own unit's
    25.7 kA. A fault between a breaker and its unit is fed by everything else
    on the bus: 40.6 kA at its opening, inside 50 kA - but the quadrant's DC
    links discharge through it first, some 830 kA peak behind their 0.05 uH:
    the battery cable's own inductance, or a reactor, must hold that down.
    """
    from test_dc_fault import STUDY
    with open(os.path.join(HERE, 'reference', f'{GRID}.diagram_payload.json'), encoding='utf-8') as fh:
        cells = {v.get('userFriendlyName'): v['name'] for v in json.load(fh).values() if isinstance(v, dict) and 'name' in v}
    out, _, _ = _drawn_study(client, quiet, {**STUDY, 'fault_bus': cells['Block 1 DC1'], 'duration_ms': '50',
                                             'frequency': '60'})
    (f,) = out['dcfault']['faults']
    assert f['ik_ka'] == pytest.approx(67.3, abs=0.3)
    brk = {b['label']: b for b in f['breakers']}
    for u in 'AB':
        b = brk[f'Block 1 DC1 GS1.1 {u} breaker']
        assert b['i_open_ka'] == pytest.approx(25.70, abs=0.05) and not b['exceeds']
        assert b['i_open_far_side_ka'] == pytest.approx(40.6, abs=0.3) and not b['exceeds_far_side']
        assert 700 < b['ip_far_side_ka'] < 950
    # The other quadrants' breakers carry nothing of it: each quadrant's DC bus is its own.
    assert all(b['ip_ka'] < 2.0 for n, b in brk.items() if not n.startswith('Block 1 DC1'))


# --- Backup protection (Phase 28) ---------------------------------------------------------------

@pytest.mark.parametrize('fault, by_relays', [('3ph', 0.5), ('1ph', None)])
def test_a_failed_backbone_breaker(client, quiet, fault, by_relays):
    """
    Each primary breaker failing to open in turn. Breaker-failure protection
    (50BF, 0.15 s) on the backbone's breakers then trips the rest of its
    switchboard: the fault is off 0.15 s after the primary's trip, and that
    switchboard's block, hall and other cable with it - at BB1 the zigzag's
    breaker too, which leaves the backbone ungrounded. Without it, the backup
    is the inverters' undervoltage trip at 0.5 s for a three-phase fault, which
    takes the whole island off - and nothing for an earth fault.
    """
    out = _protection(client, quiet, fault=fault)
    assert out['summary']['n_breaker_failure_uncleared'] == 0
    for sc in out['scenarios']:
        line = sc['fault_label'].split(',')[0]
        rows = {r['failed_switch']: r for r in sc['breaker_failure']}
        assert len(rows) == (2 if line.startswith('Backbone cable') else 1)
        for name, r in rows.items():
            assert r['t_clear_50bf_s'] == pytest.approx(r['t_primary_s'] + 0.15)
            assert r['t_clear_s'] == r['t_clear_50bf_s'] and r['t_clear_relays_s'] == by_relays
            bus = 'BB' + name[-1] if name.startswith('Cable') else f'BB{name.split()[1]}'
            n = bus[-1]
            # Every other breaker at its switchboard: the block's, the hall's, the cables'.
            at_bus = {f'Block {n} T2 HV breaker', f'Hall {n} feeder breaker'} | {
                f'Cable {a}-{b} breaker at {bus}' for a, b in (('BB1', 'BB2'), ('BB2', 'BB3'), ('BB3', 'BB1'))
                if bus in (a, b)} | {'BB1': {'Zigzag breaker'}, 'BB3': {'Zigzag 2 breaker'}}.get(bus, set())
            assert set(r['opened_by_50bf']) == at_bus - {name}

    bare = _protection(client, quiet, fault=fault, scheme=lambda name: {'bf_time_s': '0'})
    left = bare['summary']['n_breaker_failure_uncleared']
    assert left == (0 if fault == '3ph' else 9)


def _emt_line_fault(client, quiet, buffers=False, ride_through=None, hold_up=None, **params):
    with open(os.path.join(HERE, 'reference', f'{GRID}.diagram_payload.json'), encoding='utf-8') as fh:
        p = json.load(fh)
    if buffers:
        _rack_buffers(p)
    for v in p.values():
        if isinstance(v, dict) and ride_through and str(v.get('typ', '')).startswith('Solid-State Transformer'):
            v['ride_through_pu'] = str(ride_through)
        if isinstance(v, dict) and hold_up and str(v.get('typ', '')).startswith('DC/DC Converter'):
            v['hold_up_pu'] = str(hold_up)
    line = next(v['id'] for v in p.values() if isinstance(v, dict) and v.get('userFriendlyName') == 'Backbone cable BB1-BB2')
    p['0'] = {'typ': 'EmtStudy Parameters', 'frequency': '60', 'time_step_us': '20', 'user_email': 't@t',
              'duration_ms': '300', 'ac_fault_line': line, 'ac_fault_fraction': '0.5', 'ac_fault_type': 'abc',
              'ac_fault_resistance_ohm': '0.001', 'ac_fault_time_ms': '20', 'ac_trip_time_ms': '70', **params}
    with quiet():
        out = json.loads(client.post('/', json=p).get_data(as_text=True))
    return out['emt']


def test_emt_backbone_cable_fault_cleared_by_its_87l(client, quiet):
    """
    A bolted fault halfway along backbone cable BB1-BB2 at 20 ms, its 87L
    breakers told to open at 70 ms: each phase opens at its current's zero,
    71.7-76 ms; the PCS hold at their limit through it (56 ms) and the
    backbone, an open ring now, comes back. The SSTs do not: their 50 kV links
    fall past their block in 2.2 ms; let them sag to 5 % and they block at
    27-30 ms - the links' and output capacitors' energy carries the racks
    7-10 ms, not the fault's 56. The rack buffers as modelled smooth the
    racks' swing about the SST's supply and do not hold the 800 V bus without
    it: they block when it collapses. Riding through a backbone fault needs
    the SSTs' own ride-through, or rack storage that holds the bus.
    """
    emt = _emt_line_fault(client, quiet)
    (trip,) = emt['line_trips']
    assert trip['t_trip_ms'] == 70 and all(70 < t < 80 for t in trip['opened_ms'])
    bus = {b['label']: b for b in emt['ac']['buses']}
    for b in (1, 2, 3):
        assert bus[f'Load backbone {b}']['v_rms_min_pu'] < 0.01
        assert bus[f'Load backbone {b}']['v_rms_final_pu'] > 0.95
    conv = {c['label']: c for c in emt['converters']}
    assert 50 < conv['Block 1 GFM 1 (6 x PCS157SE)']['limited_ms'] < 60
    blocked = {b['label']: b['t_ms'] for b in emt['converters_blocked']}
    assert not any('GFM' in n or 'AFE' in n for n in blocked)
    for h in (1, 2, 3):
        assert 22 < blocked[f'Hall {h} SST rectifier'] < 23
    deep = _emt_line_fault(client, quiet, vsc_block_pu='0.05')
    assert all(26 < b['t_ms'] < 31 for b in deep['converters_blocked'] if 'SST' in b['label'])
    buffered = _emt_line_fault(client, quiet, buffers=True)
    held = {b['label']: b['t_ms'] for b in buffered['converters_blocked']}
    for h in (1, 2, 3):
        assert 22 < held[f'Hall {h} buffer converter'] < 30         # not at its first step (its store at 79 %)


# --- Ride-through (Phase 29) ---------------------------------------------------------------------

@pytest.mark.parametrize('ride_through, hold_up', [(0.9, None), (None, 0.95), (0.9, 0.95)])
def test_emt_backbone_cable_fault_ridden_through(client, quiet, ride_through, hold_up):
    """
    The BB1-BB2 cable fault cleared by its 87L at 70 ms again, the SSTs and
    the racks' buffers given what the vendors would have to give:
    - the SSTs riding through (their DC/DC stages pausing below 0.9 pu of
      their links, not blocking) alone: nothing holds the racks' 800 V bus,
      and the racks are lost at 24 ms;
    - the racks' buffers holding their bus (at 0.95 pu, to 1.2 times their
      3.6 MW) alone: the racks ride through at 0.92 pu, but the SSTs block
      and stay out - the buffers carry the halls until they empty;
    - both: the SSTs pause from 21 to 74 ms, resume as their links recharge
      and ramp back over 0.2 s; the buffers hold the racks meanwhile and hand
      back; the inverters come off their limit and every hall settles.
    """
    emt = _emt_line_fault(client, quiet, buffers=hold_up is not None, ride_through=ride_through, hold_up=hold_up,
                          duration_ms='300')
    loads = {ld['label']: ld for ld in emt['loads']}
    conv = {c['label']: c for c in emt['converters']}
    blocked = {b['label'] for b in emt['converters_blocked']}
    for h in (1, 2, 3):
        racks = loads[f'Hall {h}: 90 DGX H100 racks']
        if hold_up is None:
            assert racks['t_lost_ms'] == pytest.approx(24.4, abs=0.5)
            continue
        assert racks['t_lost_ms'] is None and racks['v_min_pu'] > 0.9
        buffer = conv[f'Hall {h} buffer converter']['smoothing']
        assert buffer['held_up_ms'] > 150 and buffer['store_peak_mw'] < 1.2 * 3.6
        if ride_through is None:
            assert f'Hall {h} SST rectifier' in blocked
        else:
            assert not any(n.startswith(f'Hall {h}') for n in blocked)
            ((start, end),) = conv[f'Hall {h} SST DC/DC']['paused_ms']
            assert 20 < start < 23 and 70 < end < 80
            assert racks['verdict'] == 'settles'
            assert conv[f'Hall {h} SST DC/DC']['p_end_mw'] == pytest.approx(3.6, rel=0.03)
    if ride_through and hold_up:
        assert conv['Block 1 GFM 1 (6 x PCS157SE)']['p_end_mw'] == pytest.approx(0.95, abs=0.03)


# --- A second earthing source (Phase 30) -----------------------------------------------------------

@pytest.mark.parametrize('zigzags, i_a', [(('Zigzag breaker', 'Zigzag 2 breaker'), 427), (('Zigzag 2 breaker',), 367),
                                          ((), 6.5)])
def test_emt_backbone_earth_fault_needs_a_zigzag(client, quiet, zigzags, i_a):
    """
    Phase a to ground on backbone 2 for 100 ms, in EMT. Both zigzags: 427 A,
    shared. ZZ2 alone - after a failed breaker at BB1 has opened ZZ1's with the
    rest of its switchboard - 367 A, inside its 400 A rating: a second
    earthing source keeps the backbone grounded, and its relays seeing its
    earth faults. Neither: 6.5 A, the cables' capacitance, which nothing
    detects, and the backbone left displaced after the fault clears.
    """
    with open(os.path.join(HERE, 'reference', f'{GRID}.diagram_payload.json'), encoding='utf-8') as fh:
        p = json.load(fh)
    cell = {v.get('userFriendlyName'): v['name'] for v in p.values() if isinstance(v, dict) and 'name' in v}
    for v in p.values():
        if isinstance(v, dict) and v.get('userFriendlyName') in ('Zigzag breaker', 'Zigzag 2 breaker'):
            v['closed'] = 'true' if v['userFriendlyName'] in zigzags else 'false'
    p['0'] = {'typ': 'EmtStudy Parameters', 'frequency': '60', 'time_step_us': '20', 'user_email': 't@t',
              'duration_ms': '150', 'ac_fault_bus': cell['Load backbone 2'], 'ac_fault_type': 'ag',
              'ac_fault_resistance_ohm': '0.001', 'ac_fault_time_ms': '20', 'ac_fault_duration_ms': '100'}
    with quiet():
        emt = json.loads(client.post('/', json=p).get_data(as_text=True))['emt']
    (phase_a,) = emt['ac']['fault']['phases']
    assert phase_a['i_sym_ka'] * 1e3 == pytest.approx(i_a, rel=0.03)
    backbone = next(b for b in emt['ac']['buses'] if b['label'] == 'Load backbone 2')
    if zigzags:
        assert phase_a['i_sym_ka'] / len(zigzags) < 0.4             # within each zigzag's 400 A (they share it)
        assert backbone['v_rms_final_pu'] < 1.05
    else:
        assert backbone['v_rms_final_pu'] > 1.4


# --- Charge balancing across quadrants (Phase 32) ----------------------------------------------------

def _ts_quadrants(client, quiet, gain, low=80, afe_out=None, steps=20):
    """30 min steps, the racks at their cycle's average; Block 1 DC1's GS1.1 at ``low`` %, the rest at 80 %."""
    with open(os.path.join(HERE, 'reference', f'{GRID}.diagram_payload.json'), encoding='utf-8') as fh:
        p = json.load(fh)
    with open(SPEC_PATH, encoding='utf-8') as fh:
        lp = {x['id']: {'name': x['name'], 'kind': x['kind'], 'dt': x['dt_s'], 't0': 0, 'p': x['values']}
              for x in json.load(fh)['load_profiles']}
    for v in p.values():
        if not isinstance(v, dict):
            continue
        name = str(v.get('userFriendlyName', ''))
        if str(v.get('typ', '')).startswith('VSC') and 'AFE' in name:
            v['soc_balance_gain'] = str(gain)
            if afe_out and name.startswith(afe_out):
                v['in_service'] = 'false'
        if str(v.get('typ', '')).startswith('Battery') and name.startswith('Block 1 DC1'):
            v['soc_percent'] = str(low)
    p['0'] = {'typ': 'TimeSeriesSimulationPandaPower Parameters', 'time_steps': str(steps), 'time_step_s': '1800',
              'load_profiles': lp, 'load_profile': 'constant', 'generation_profile': 'constant',
              'profile_mode': 'custom', 'element_profiles': {}, 'frequency': '60', 'algorithm': 'nr',
              'calculate_voltage_angles': 'auto', 'init': 'auto', 'microgrid_dispatch': 'True', 'user_email': 't@t'}
    with quiet():
        out = json.loads(client.post('/', json=p).get_data(as_text=True))
    assert not out.get('error'), out.get('message')
    mg = out['microgrid']
    soc = {}
    for r in mg['ders']:
        if r['kind'] == 'Battery' and r['label'].endswith(' A'):
            soc.setdefault(r['label'][:-len(' GS1.1 A')], []).append(r['soc_percent_end'])
    return mg, soc


@pytest.mark.parametrize('gain', [0, 2])
def test_charge_balancing_evens_the_quadrants(client, quiet, gain):
    """
    Block 1 DC1's GS1.1 at 60 %, the other eleven quadrants' at 80 %, for
    10 h. Each rectifier following its bus's load leaves each quadrant where
    it started. Balancing (soc_balance_gain 2: 2 x its rating per unit of its
    quadrant's charge below the quadrants' average) feeds DC1 up to 0.37 MW
    more and the others some 33 kW less each - the grid's draw unchanged -
    and the twelve meet near their 78.3 % average within the 10 h.
    """
    mg, soc = _ts_quadrants(client, quiet, gain, low=60)
    assert len(soc) == 12 and mg['unserved_mwh'] == 0
    low, others = soc['Block 1 DC1'], [v for k, v in soc.items() if k != 'Block 1 DC1']
    if gain == 0:
        assert low[-1] == pytest.approx(60.0, abs=0.2)
        assert all(v[-1] == pytest.approx(80.0, abs=0.2) for v in others)
        return
    rect = {r['label']: r for r in mg['rectifiers']}
    assert rect['Block 1 AFE 1 (6 x PCS157SE)']['balance_max_mw'] == pytest.approx(0.36, abs=0.03)
    assert all(r['balance_max_mw'] < 0.04 for n, r in rect.items() if not n.startswith('Block 1 AFE 1'))
    assert 75 < low[-1] < 78.5
    assert all(78 < v[-1] < 79 for v in others)
    assert max(v[-1] for v in others) - low[-1] < 2.0
    assert low[9] > 72                                  # most of the way in 5 h


def test_a_lost_rectifier_drains_its_quadrant(client, quiet):
    """
    Block 1's AFE 1 out, every GS1.1 at 80 %, for 12 h. Its quadrant's
    inverter goes on taking its share from DC1's batteries alone, which empty
    in 7.5-8 h; the other eleven inverters then take its share, their
    rectifiers up to 0.71 MW - nothing unserved. Balancing through the
    rectifiers cannot keep DC1's charge (it has none) - only the inverters'
    sharing could, and DC1 is left with no reserve for a utility outage.
    """
    mg, soc = _ts_quadrants(client, quiet, 2, low=80, afe_out='Block 1 AFE 1', steps=24)
    assert mg['unserved_mwh'] == 0
    dc1 = soc['Block 1 DC1']
    empty = next(k for k, v in enumerate(dc1) if v < 1.0)
    assert 14 <= empty <= 16                          # 7-8 h at 30 min a step
    assert all(v[-1] > 79 for k, v in soc.items() if k != 'Block 1 DC1')
    rect = {r['label']: r for r in mg['rectifiers']}
    assert 'Block 1 AFE 1 (6 x PCS157SE)' not in rect
    assert max(r['rectifier_max_mw'] for r in rect.values()) == pytest.approx(0.71, abs=0.02)

