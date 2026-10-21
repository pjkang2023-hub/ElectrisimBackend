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
import math
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


@pytest.mark.parametrize('k, backbone_a', [(1.2, 415), (1.5, 519), (None, 7.6)])
def test_earth_faults(client, quiet, k, backbone_a):
    """
    The backbone's earth faults by IEC 60909, with both zigzags: 415 A at the
    inverters' 1.2 pu, 519 A at 1.5 pu - where their worst phase reaches its
    limit (EMT, which builds their windings and controls: 367 A at 1.2 pu,
    the SSTs drawing their load through it - Phase 41). An earth fault draws I1 = I2 = I0; through T2's delta
    each inverter's worst phase carries sqrt(3) of it, so the fault is
    3 x 1.2 x 190 A / sqrt(3). The zigzags alone would pass 750 A. With both
    out, 7.6 A: the cables' capacitance alone (EMT: 6.5 A). Before Phase 37
    each grid-forming PCS stood behind its limit's impedance, 1/k pu, in both
    sequences - 312 A and 369 A, far below the limit yet held by it twice;
    before Phase 31, 249-311 A and 228 A without a zigzag. Each is now a
    source behind its own reactor (0.15 pu, EMT's), the fault capped at the
    limit (sc_current_sources), and a current source's share IEC's
    3 Z(1) / (2 Z(1) + Z(0)). The 505 V buses none: the grid side a delta,
    the load side T2's star left floating, as the spec asks (solidly grounded
    it gave 17 kA). The POI: the utility's own.
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
        assert ka[f'Load backbone {b}'] / (2 if k else 1) < 0.4          # within each zigzag's 400 A
        assert ka[f'Block {b} 505 V grid side'] == pytest.approx(0, abs=1e-6)
        # Its PCS's own source's zero sequence, 1e6 times its X(1): a trace, some 0.02 A.
        assert ka[f'Block {b} 505 V load side'] == pytest.approx(0, abs=1e-4)
    assert ka['Utility 34.5 kV POI'] == pytest.approx(8.40, abs=0.02)


def test_ansi_earth_fault_on_the_backbone(client, quiet):
    """ANSI C37 as IEC (Phase 37): the inverters' current limit holds the earth fault, 415 A, the two zigzags sharing it."""
    from test_reference_grids import ANSI_PARAMS
    out, label, _ = _drawn_study(client, quiet, {**ANSI_PARAMS, 'fault_type': '1ph', 'frequency_hz': 60, 'user_email': 't@t'})
    ka = _bus_ka(out, label)
    assert ka['Load backbone 2'] == pytest.approx(0.415, rel=0.01)
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
    1.22 MW each, 123 % - inside their 150 % for 1 s, past their 120 % for
    10 s - at 59.43 Hz, their voltage control holding the backbone (without
    it the island sagged and the racks drew 1.16 MW); past 10 s over their
    rating their ceiling falls to it and the island sags further: load must
    come off within 10 s of losing a block.
    """
    out = _andes(client, quiet, 20, line='Block 3 T2 0.505/34.5 kV Dy1 (505 V star floating)')
    left = [g for g in out['pcs'] if not g['label'].startswith('Block 3')]
    assert len(left) == 8
    for g in left:
        assert _at(g, 'p_mw', 8.5, out['time']) == pytest.approx(1.22, abs=0.02)
        assert _at(g, 'frequency_hz', 8.5, out['time']) == pytest.approx(59.43, abs=0.03)
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


@pytest.mark.parametrize('fault, k, ik_a', [('3ph', 1.2, 238), ('3ph', 1.5, 298), ('1ph', 1.2, 415)])
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
    (nor can the study: pandapower gives no phase voltages for one); its
    backup is the zigzags' neutral overcurrent (Phase 36), 2 s, tripping their
    switchboards.
    """
    out = _protection(client, quiet, fault=fault, k_pcs=k)
    s = out['summary']
    assert s['n_scenarios'] == 6 and s['n_miscoordination'] == 0 and s['n_unwanted_trips'] == 0
    for sc in out['scenarios']:
        assert sc['short_circuit']['ikss_ka'] * 1e3 == pytest.approx(ik_a, abs=2)
        t = _trips(sc)
        tripped = {n for n, r in t.items() if r['tripped'] and r['t_trip_s'] < 0.3}
        backup = {n for n, r in t.items() if r['tripped'] and r['t_trip_s'] >= 0.3}
        if fault == '3ph':
            assert backup == {f'Block {b} T2 LV breaker' for b in (1, 2, 3)}
            for n in backup:
                assert t[n]['t_trip_s'] == 0.5 and t[n]['evaluation']['vm_pu'] < 0.1
        else:
            assert {'Zigzag breaker', 'Zigzag 2 breaker'} <= backup
            for n in backup:
                assert t[n]['t_trip_s'] == 2.0 and (n.startswith('Zigzag') or t[n].get('intertrip_from', '').startswith('Zigzag'))
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
        off = {'bdiff_i_a': '0', 'intertrip_bus': 'false'}
        if 'feeder' in name:
            return {'protection_type': 'earth_fault', 'I_e_a': '40', 't_e': '0.1', **off}
        if name.startswith('Cable'):
            return {'protection_type': 'earth_fault', 'I_e_a': '40', 't_e': '0.4', **off}
        return {'protection_type': 'none', **off}
    out = _protection(client, quiet, scheme, fault='1ph')
    unwanted = {(u['fault_label'].split(',')[0], u['user_friendly_name']) for u in out['unwanted_trips']}
    assert out['summary']['n_unwanted_trips'] == 12
    assert all(f.startswith('Backbone cable') for f, _ in unwanted)
    for sc in out['scenarios']:
        if 'feeder' in sc['fault_label']:
            assert sc['clearing_time_s'] == 0.1


def test_t1_lv_instantaneous_grades(client, quiet):
    """
    The spec's T1 LV breaker (the arc-flash item): I>> 20 kA at 0.05 s takes
    the 505 V grid side's 68.6 kA fault at once; the HV breaker's backup
    stays at 0.8 s, graded.
    """
    out = _protection(client, quiet, mode='bus', bus='Block 1 505 V grid side')
    (sc,) = out['scenarios']
    t = _trips(sc)
    assert sc['primary_switches'] == ['Block 1 T1 LV breaker'] and sc['clearing_time_s'] == pytest.approx(0.05)
    assert t['Block 1 T1 LV breaker']['t_trip_s'] == pytest.approx(0.05)
    assert t['Block 1 T1 HV breaker']['t_trip_s'] == pytest.approx(0.8)
    assert out['summary']['n_miscoordination'] == 0


def _arc_by_relays(client, quiet, scheme=None):
    with open(os.path.join(HERE, 'reference', f'{GRID}.diagram_protection_payload.json'), encoding='utf-8') as fh:
        p = json.load(fh)
    for v in p.values():
        if isinstance(v, dict) and scheme and str(v.get('typ', '')).startswith('Switch'):
            v.update(scheme(v['userFriendlyName']) or {})
    p['0'] = {'typ': 'ArcFlashPandaPower Parameters', 'electrode_config': 'VCB', 'equipment_mode': 'by_voltage',
              'clearing_time_mode': 'relays', 'frequency': '60', 'user_email': 't@t'}
    with quiet():
        out = json.loads(client.post('/', json=p).get_data(as_text=True))
    assert not out.get('error'), out.get('message')
    return {r['name']: r for r in out['arc_flash']}, out.get('warnings') or []


def test_arc_flash_cleared_by_the_relays(client, quiet):
    """
    Each bus's arc cleared by the spec's relays at its arcing current, plus
    the breaker's 0.05 s. The 505 V grid side: T1 LV's I>> sees the arc's
    31 kA (27 kA reduced) and takes it in 0.1 s - 6.4 cal/cm2, PPE 2. A
    backbone arc: the inverters' 27s on every T2 LV breaker part it from
    them at 0.55 s (Ralph Lee, 2.8 cal/cm2); a hall's, its feeder breaker at
    0.15 s - and since Phase 36 the backbone's arcs are its switchboards'
    bus differential's, 0.05 s. The 505 V load side: its own grid-forming inverters feed it, and
    no breaker parts them from the bus; the T2 LV breaker's arc-flash
    detection (Phase 35) sees the arc's light and the backbone's 10.9 kA,
    trips in 0.01 s and blocks them - 0.06 s, 1.2 cal/cm2, PPE 0. The
    utility's breaker at the POI is not on the diagram.
    """
    rows, warnings = _arc_by_relays(client, quiet)
    for b in (1, 2, 3):
        g = rows[f'Block {b} 505 V grid side']
        assert g['clearing_relays'] == [f'Block {b} T1 LV breaker']
        assert g['relay_time_s'] == pytest.approx(0.05) and g['relay_time_min_s'] == pytest.approx(0.05)
        assert g['clearing_time_used_s'] == pytest.approx(0.1)
        assert g['incident_energy_cal_cm2'] == pytest.approx(6.4, abs=0.1) and str(g['ppe_category']) == '2'
        assert 20 < g['relay_fault_min_ka'] < g['relay_fault_ka'] < g['ikss_ka']
        bb = rows[f'Load backbone {b}']
        assert f'Block {b} T2 HV breaker' in bb['clearing_relays'] and bb['relay_time_s'] == pytest.approx(0.05)
        assert bb['clearing_time_used_s'] == pytest.approx(0.1)
        assert bb['incident_energy_cal_cm2'] == pytest.approx(2.8 * 0.1 / 0.55, abs=0.05)
        h = rows[f'Hall {b} 34.5 kV']
        assert h['clearing_relays'] == [f'Hall {b} feeder breaker'] and h['clearing_time_used_s'] == pytest.approx(0.15)
        ld = rows[f'Block {b} 505 V load side']
        assert ld['clearing_relays'] == [f'Block {b} T2 LV breaker'] and ld['inverters_blocked']
        assert ld['relay_time_s'] == pytest.approx(0.01) and ld['relay_time_min_s'] == pytest.approx(0.01)
        assert ld['clearing_time_used_s'] == pytest.approx(0.06)
        assert ld['incident_energy_cal_cm2'] == pytest.approx(1.2, abs=0.05) and str(ld['ppe_category']) == '0'
    assert rows['Utility 34.5 kV POI']['grid_on_bus'] and rows['Utility 34.5 kV POI']['clearing_time_used_s'] == 2.0
    assert [w for w in warnings if 'feeds its arc' in w] == [
        'Bus Utility 34.5 kV POI: a grid, machine or grid-forming inverter on the bus feeds its arc, no relay '
        'parting it from the bus; 2 s used.']


def test_without_arc_detection_the_load_sides_arc_burns_on(client, quiet):
    """
    The T2 LV breakers without their arc-flash detection: the load side's own
    inverters feed its arc and no relay parts them from the bus - IEEE 1584's
    2 s, 39 cal/cm2, PPE 4. Their 27 does not see it either: the arc holds the
    bus near 0.7 pu.
    """
    rows, warnings = _arc_by_relays(client, quiet, lambda n: {'arc_t_s': '0'} if ' T2 LV ' in n else None)
    for b in (1, 2, 3):
        ld = rows[f'Block {b} 505 V load side']
        assert ld['grid_on_bus'] and ld['relay_time_s'] is None and ld['clearing_time_used_s'] == 2.0
        assert len(ld['inverter_sources']) == 4 and not ld['inverters_blocked']
        assert 38.5 < ld['incident_energy_cal_cm2'] < 40 and str(ld['ppe_category']) == '4'
    assert sum('feeds its arc, no relay parting it' in w for w in warnings) == 4


def test_arc_detection_sees_only_its_own_switchboard(client, quiet):
    """
    A bolted fault on Block 1's 505 V load side: the T2 LV breaker's arc-flash
    detection sees its light and the backbone's 10.9 kA, over its 6.43 kA,
    and trips in 0.01 s, before its 27; the other blocks' T2 LV breakers are
    dark and their 27s take 0.5 s. A backbone fault is dark to all of them.
    """
    out = _protection(client, quiet, mode='bus', bus='Block 1 505 V load side')
    (sc,) = out['scenarios']
    t = _trips(sc)
    own = t['Block 1 T2 LV breaker']
    assert own['t_trip_s'] == pytest.approx(0.01) and own['arc_detection']['tripped_first']
    assert own['arc_detection']['light'] and own['arc_detection']['current_a'] == pytest.approx(10913, abs=20)
    assert sc['clearing_time_s'] == pytest.approx(0.01)
    for k in (2, 3):
        other = t[f'Block {k} T2 LV breaker']
        assert not other['arc_detection']['light'] and other['t_trip_s'] == pytest.approx(0.5)
    assert out['summary']['n_miscoordination'] == 0 and out['summary']['n_unwanted_trips'] == 0
    out = _protection(client, quiet, mode='bus', bus='Load backbone 1')
    (sc,) = out['scenarios']
    for k in (1, 2, 3):
        row = _trips(sc)[f'Block {k} T2 LV breaker']
        assert not row['arc_detection']['tripped'] and row['t_trip_s'] == pytest.approx(0.5)


def test_without_the_instantaneous_the_grid_side_arc_lasts_half_a_second(client, quiet):
    """
    T1 LV's overcurrent without I>>: the grid side's arc lasts the I> stage's
    0.5 s plus the breaker's - 35 cal/cm2, PPE 4. The 0.2 s the voltage-class
    study assumed (12.8 cal/cm2) was not the relays'.
    """
    rows, _ = _arc_by_relays(client, quiet, lambda n: {'I_gg_a': '0'} if ' T1 LV ' in n else None)
    for b in (1, 2, 3):
        g = rows[f'Block {b} 505 V grid side']
        assert g['relay_time_s'] == pytest.approx(0.5) and g['clearing_time_used_s'] == pytest.approx(0.55)
        assert g['incident_energy_cal_cm2'] == pytest.approx(35.2, abs=0.2) and str(g['ppe_category']) == '4'


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

@pytest.mark.parametrize('fault, by_relays', [('3ph', 0.5), ('1ph', {'BB1': 2.0, 'BB3': 2.0, 'BB2': None})])
def test_a_failed_backbone_breaker(client, quiet, fault, by_relays):
    """
    Each primary breaker failing to open in turn. Breaker-failure protection
    (50BF, 0.15 s) on the backbone's breakers then trips the rest of its
    switchboard: the fault is off 0.15 s after the primary's trip, and that
    switchboard's block, hall and other cable with it - at BB1 the zigzag's
    breaker too, which leaves the backbone ungrounded. Without it, the backup
    is the inverters' undervoltage trip at 0.5 s for a three-phase fault, which
    takes the whole island off; for an earth fault, at the zigzags'
    switchboards, their neutral overcurrent (Phase 36) tripping the
    switchboard at 2 s - at BB2, with no zigzag, nothing.
    """
    out = _protection(client, quiet, fault=fault)
    assert out['summary']['n_breaker_failure_uncleared'] == 0
    for sc in out['scenarios']:
        line = sc['fault_label'].split(',')[0]
        rows = {r['failed_switch']: r for r in sc['breaker_failure']}
        assert len(rows) == (2 if line.startswith('Backbone cable') else 1)
        for name, r in rows.items():
            assert r['t_clear_50bf_s'] == pytest.approx(r['t_primary_s'] + 0.15)
            bus = 'BB' + name[-1] if name.startswith('Cable') else f'BB{name.split()[1]}'
            want = by_relays[bus] if isinstance(by_relays, dict) else by_relays
            assert r['t_clear_s'] == r['t_clear_50bf_s'] and r['t_clear_relays_s'] == want
            n = bus[-1]
            # Every other breaker at its switchboard: the block's, the hall's, the cables'.
            at_bus = {f'Block {n} T2 HV breaker', f'Hall {n} feeder breaker'} | {
                f'Cable {a}-{b} breaker at {bus}' for a, b in (('BB1', 'BB2'), ('BB2', 'BB3'), ('BB3', 'BB1'))
                if bus in (a, b)} | {'BB1': {'Zigzag breaker'}, 'BB3': {'Zigzag 2 breaker'}}.get(bus, set())
            assert set(r['opened_by_50bf']) == at_bus - {name}

    bare = _protection(client, quiet, fault=fault, scheme=lambda name: {'bf_time_s': '0'})
    left = bare['summary']['n_breaker_failure_uncleared']
    assert left == (0 if fault == '3ph' else 3)


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

@pytest.mark.parametrize('zigzags, i_a', [(('Zigzag breaker', 'Zigzag 2 breaker'), 367), (('Zigzag 2 breaker',), 367),
                                          ((), 6.5)])
def test_emt_backbone_earth_fault_needs_a_zigzag(client, quiet, zigzags, i_a):
    """
    Phase a to ground on backbone 2 for 100 ms, in EMT. Both zigzags: 367 A,
    shared - the inverters' worst phase at its limit, the SSTs drawing their
    load through them all along (427 A before Phase 41, the positive sequence
    alone limited). ZZ2 alone - after a failed breaker at BB1 has opened ZZ1's
    with the rest of its switchboard - 367 A too, inside its 400 A rating: a second
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


@pytest.mark.parametrize('case, backbone_a', [('max', 406), ('min', 372)])
def test_iec_earth_fault_with_one_zigzag_against_emt(client, quiet, case, backbone_a):
    """
    ZZ2 alone, by IEC 60909 (Phase 37): 406 A at c = 1.1, 372 A at c = 1.0,
    under the inverters' limit (415 A) - the zigzag's impedance and the
    inverters' reactors hold it. EMT, its sources at 1.0 pu: 367 A, within
    1.5 % of the minimum case.
    """
    with open(os.path.join(HERE, 'reference', f'{GRID}.diagram_payload.json'), encoding='utf-8') as fh:
        p = json.load(fh)
    for v in p.values():
        if isinstance(v, dict) and v.get('userFriendlyName') == 'Zigzag breaker':
            v['closed'] = 'false'
    label = {v['name']: v.get('userFriendlyName') for v in p.values() if isinstance(v, dict) and 'name' in v}
    p['0'] = {**IEC, 'fault_type': '1ph', 'fault_location': case}
    with quiet():
        out = json.loads(client.post('/', json=p).get_data(as_text=True))
    ka = _bus_ka(out, label)
    for b in (1, 2, 3):
        assert ka[f'Load backbone {b}'] * 1e3 == pytest.approx(backbone_a, rel=0.01)
    if case == 'min':
        assert ka['Load backbone 2'] * 1e3 == pytest.approx(367, rel=0.015)


# --- Charge balancing across quadrants (Phase 32) ----------------------------------------------------

def _ts_quadrants(client, quiet, gain, low=80, afe_out=None, steps=20, share=0, outage=False):
    """
    30 min steps, the racks at their cycle's average; Block 1 DC1's GS1.1 at
    ``low`` %, the rest at 80 %; the rectifiers' soc_balance_gain ``gain``,
    the inverters' soc_share_gain ``share``; the utility out if ``outage``.
    """
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
        if str(v.get('typ', '')).startswith('PCS'):
            v['soc_share_gain'] = str(share)
        if outage and str(v.get('typ', '')).startswith('External Grid'):
            v['in_service'] = 'false'
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


# --- Inverters sharing by charge (Phase 33) ----------------------------------------------------------

def test_inverters_share_by_charge_after_a_lost_rectifier(client, quiet):
    """
    Block 1's AFE 1 out again, now with the inverters sharing by charge
    (soc_share_gain 5: half an inverter's rating for 10 points of its
    quadrant's charge off the sharing inverters' average). As DC1 drains its
    inverter hands its share to the other eleven - some 58 kW more each - and
    DC1 settles near 66 %, its inverter at next to nothing, instead of running
    to empty. Nothing unserved.
    """
    mg, soc = _ts_quadrants(client, quiet, 2, low=80, afe_out='Block 1 AFE 1', steps=24, share=5)
    assert mg['unserved_mwh'] == 0
    dc1 = soc['Block 1 DC1']
    assert 64 < dc1[-1] < 68 and abs(dc1[-1] - dc1[-5]) < 0.5
    pcs = {}
    for r in mg['pcs']:
        pcs.setdefault(r['label'], []).append(r)
    assert abs(pcs['Block 1 GFM 1 (6 x PCS157SE)'][-1]['p_mw']) < 0.05
    for name, rows in pcs.items():
        if not name.startswith('Block 1 GFM 1'):
            assert rows[-1]['soc_share_mw'] == pytest.approx(0.058, abs=0.01), name


@pytest.mark.parametrize('dc1, short_step, unserved', [(65.6, 14, 19.2), (0.0, 13, 22.9)])
def test_a_utility_outage_after_the_lost_rectifier(client, quiet, dc1, short_step, unserved):
    """
    The utility then out, all twelve inverters sharing by charge: from DC1 at
    the 65.6 % sharing kept, the island carries its racks 7 h; from DC1 empty,
    as it was without sharing, 6.5 h, and 3.7 MWh more goes unserved over the
    10 h (every quadrant at 80 %: 7.5 h). An empty quadrant's inverter gives
    nothing and takes nothing: it is not charged from the others, which only
    shuffled energy at a loss.
    """
    mg, soc = _ts_quadrants(client, quiet, 0, low=dc1, steps=20, share=5, outage=True)
    first = next(x['time_step'] for x in mg['steps'] if x['unserved_mw'] > 0.5)
    assert first == short_step
    assert mg['unserved_mwh'] == pytest.approx(unserved, abs=0.2)
    if dc1 == 0.0:
        assert max(soc['Block 1 DC1']) < 0.5


# --- Phase 36: bus differential on the backbone's switchboards; the zigzags' neutral overcurrent ---------

@pytest.mark.parametrize('fault', ['1ph', '3ph'])
@pytest.mark.parametrize('n', [1, 2, 3])
def test_bus_differential_clears_a_switchboard_fault(client, quiet, fault, n):
    """
    A fault on a backbone switchboard itself: the 87L and 87T see their own
    cables and transformers, the feeders theirs, the 27s three-phase faults
    only - an earth fault there stood, the zigzags carrying it inside their
    10 s. The switchboard's bus differential (87B: every breaker at it a CT,
    60 A, 30 %) sees the fault's whole current and opens them all in 0.05 s;
    the other switchboards' sum nothing. No miscoordination, no needless trip.
    """
    out = _protection(client, quiet, mode='bus', bus=f'Load backbone {n}', fault=fault)
    (sc,) = out['scenarios']
    t = _trips(sc)
    bus = f'BB{n}'
    at_bus = {f'Block {n} T2 HV breaker', f'Hall {n} feeder breaker'} | {
        f'Cable {a}-{b} breaker at {bus}' for a, b in (('BB1', 'BB2'), ('BB2', 'BB3'), ('BB3', 'BB1'))
        if bus in (a, b)} | {'BB1': {'Zigzag breaker'}, 'BB3': {'Zigzag 2 breaker'}}.get(bus, set())
    fast = {k for k, r in t.items() if r['tripped'] and r['t_trip_s'] < 0.3}
    assert fast == at_bus and sc['clearing_time_s'] == pytest.approx(0.05)
    ik_a = sc['short_circuit']['ikss_ka'] * 1e3
    for k in at_bus:
        assert t[k]['bus_differential']['tripped'] and t[k]['bus_differential']['i_operate_a'] > 0.9 * ik_a
    for k, r in t.items():
        if r.get('bus_differential') and k not in at_bus:
            assert r['bus_differential']['i_operate_a'] < 1.0 and not r['bus_differential']['tripped']
    assert out['summary']['n_miscoordination'] == 0 and out['summary']['n_unwanted_trips'] == 0


def test_the_zigzags_neutral_overcurrent_is_the_backup(client, quiet):
    """
    The zigzags' neutral current: the fault's 3 I0 shared by their zero-sequence
    admittance - 208 A each for a backbone earth fault (415 A), 406 A in one with
    the other's breaker open (pandapower's single-phase branch results carry no
    zero sequence: a zigzag read 0 A). Their 51N, 100 A at 2 s, trips its
    switchboard. Without 87B a fault on BB1 is cleared that way at 2 s - BB3's
    switchboard opened needlessly with it - and a fault on BB2, with no zigzag,
    is not: opening BB1 and BB3 leaves it on Block 2, ungrounded. 87B is the
    switchboards' protection; the 51N keeps the zigzags inside their rating.
    """
    off = lambda name: {'bdiff_i_a': '0'}
    out = _protection(client, quiet, off, mode='bus', bus='Load backbone 2', fault='1ph')
    (sc,) = out['scenarios']
    t = _trips(sc)
    for z in ('Zigzag breaker', 'Zigzag 2 breaker'):
        assert t[z]['evaluation']['residual_current_a'] == pytest.approx(sc['short_circuit']['ikss_ka'] * 500, rel=1e-6)
        assert t[z]['evaluation']['residual_current_a'] == pytest.approx(208, abs=2) and t[z]['t_trip_s'] == 2.0
    assert sc['clearing_time_s'] is None
    out = _protection(client, quiet, off, mode='bus', bus='Load backbone 1', fault='1ph')
    (sc,) = out['scenarios']
    t = _trips(sc)
    assert sc['clearing_time_s'] == 2.0
    assert t['Block 1 T2 HV breaker']['intertrip_from'] == 'Zigzag breaker'
    assert {u['user_friendly_name'] for u in out['unwanted_trips']} == {
        'Zigzag 2 breaker', 'Cable BB2-BB3 breaker at BB3', 'Block 3 T2 HV breaker', 'Hall 3 feeder breaker'}
    # One zigzag alone carries it all.
    with open(os.path.join(HERE, 'reference', f'{GRID}.diagram_protection_payload.json'), encoding='utf-8') as fh:
        p = json.load(fh)
    cells = {v.get('userFriendlyName'): v['name'] for v in p.values() if isinstance(v, dict) and 'name' in v}
    for v in p.values():
        if isinstance(v, dict) and v.get('userFriendlyName') == 'Zigzag breaker':
            v['closed'] = 'false'
    p['0'] = {**p['0'], 'fault_type': '1ph', 'fault_location_mode': 'bus', 'fault_bus_id': cells['Load backbone 2'],
              'user_email': 't@t'}
    with quiet():
        out = json.loads(client.post('/', json=p).get_data(as_text=True))
    (sc,) = out['scenarios']
    t = _trips(sc)
    assert t['Zigzag 2 breaker']['evaluation']['residual_current_a'] == pytest.approx(
        sc['short_circuit']['ikss_ka'] * 1e3, rel=1e-6)
    assert sc['clearing_time_s'] == pytest.approx(0.05)


# --- Phase 38: the zigzags' duty --------------------------------------------------------------------

def _earth_fault_on_bb2(client, quiet, r_fault=0, scheme=None):
    with open(os.path.join(HERE, 'reference', f'{GRID}.diagram_protection_payload.json'), encoding='utf-8') as fh:
        p = json.load(fh)
    cells = {v.get('userFriendlyName'): v['name'] for v in p.values() if isinstance(v, dict) and 'name' in v}
    for v in p.values():
        if scheme and isinstance(v, dict) and str(v.get('typ', '')).startswith('Switch'):
            v.update(scheme(v['userFriendlyName']) or {})
    p['0'] = {**p['0'], 'fault_type': '1ph', 'fault_location_mode': 'bus', 'fault_bus_id': cells['Load backbone 2'],
              'r_fault_ohm': r_fault, 'user_email': 't@t'}
    with quiet():
        out = json.loads(client.post('/', json=p).get_data(as_text=True))
    assert not out.get('error'), out.get('message')
    (sc,) = out['scenarios']
    return out, sc, {d['label']: d for d in sc['grounding_duty']}


def test_the_zigzags_short_time_duty(client, quiet):
    """
    A zigzag's duty: its neutral current for as long as it flows - until the
    fault is cleared, or its own breaker opens - against its 400 A, 10 s
    rating, I^2 t. A bolted earth fault on BB2: 208 A each, 87B clearing it in
    0.05 s, 0.13 %; with a primary breaker stuck, until their 51N opens them at
    2 s, 5.4 %. One zigzag alone, the other's breaker open: 404 A, 20 % behind
    the 51N - the worst short-time duty, a fifth of the rating.
    """
    out, sc, duty = _earth_fault_on_bb2(client, quiet)
    assert len(duty) == 2
    for d in duty.values():
        assert d['neutral_a'] == pytest.approx(207.5, abs=1) and d['t_flowing_s'] == pytest.approx(0.05)
        assert d['duty_percent'] == pytest.approx(100 * (207.5 / 400) ** 2 * 0.05 / 10, rel=0.01)
        assert d['t_flowing_backup_s'] == pytest.approx(2.0) and d['duty_backup_percent'] == pytest.approx(5.38, abs=0.05)
    assert out['summary']['n_grounding_continuous'] == 0
    out, sc, duty = _earth_fault_on_bb2(client, quiet, scheme=lambda n: {'closed': 'false'} if n == 'Zigzag breaker' else None)
    (d,) = duty.values()
    assert d['neutral_a'] == pytest.approx(sc['short_circuit']['ikss_ka'] * 1e3, rel=1e-6)
    assert 400 < d['neutral_a'] < 410 and d['duty_backup_percent'] == pytest.approx(20.4, abs=0.3)


@pytest.mark.parametrize('r_fault, cleared, backed_up', [(150, True, False), (300, False, False)])
def test_a_high_resistance_earth_fault(client, quiet, r_fault, cleared, backed_up):
    """
    Through 150 ohm a BB2 earth fault draws 124 A: 87B, at 60 A plus 30 % of
    its restraint, still sees it; the zigzags' 51N (100 A each, 62 A here)
    does not, so a stuck breaker leaves it on. Through 300 ohm, 67 A - under
    87B's threshold, about 71 A: nothing trips, and each zigzag carries 34 A
    indefinitely (one alone, up to some 70 A). That is a continuous duty, which
    their short-time rating does not cover: a zigzag's continuous rating, or a
    sensitive alarm on its neutral, is the designer's.
    """
    out, sc, duty = _earth_fault_on_bb2(client, quiet, r_fault)
    assert out['summary']['r_fault_ohm'] == r_fault
    assert (sc['clearing_time_s'] == pytest.approx(0.05)) if cleared else sc['clearing_time_s'] is None
    for d in duty.values():
        assert d['neutral_a'] == pytest.approx(sc['short_circuit']['ikss_ka'] * 500, rel=1e-6)
        assert d['continuous'] == (not cleared) and d['continuous_backup'] == (not backed_up)
    assert out['summary']['n_grounding_continuous'] == (0 if cleared else 2)
    if not cleared:
        assert all(30 < d['neutral_a'] < 40 for d in duty.values())


# --- Phase 39: an alarm on the zigzags' neutrals ----------------------------------------------------

@pytest.mark.parametrize('r_fault, alone, alarmed', [(0, False, False), (300, False, True), (500, False, True),
                                                     (700, False, False), (1000, True, True), (1500, True, False)])
def test_the_zigzags_neutral_alarm(client, quiet, r_fault, alone, alarmed):
    """
    Each zigzag's neutral has an alarm stage, 20 A for 10 s, which trips
    nothing. A bolted fault 87B clears in 0.05 s: no alarm. Under the relays'
    ~71 A nothing trips, and the alarm flags it down to 20 A a zigzag - a BB2
    earth fault through 500 ohm with both in (21 A each), through 1000 ohm with
    one alone (21 A). What it does not flag is at most its 20 A: 5 % of the
    400 A, within a zigzag's common continuous rating.
    """
    scheme = (lambda n: {'closed': 'false'} if n == 'Zigzag breaker' else None) if alone else None
    out, sc, duty = _earth_fault_on_bb2(client, quiet, r_fault, scheme)
    s = out['summary']
    assert len(duty) == (1 if alone else 2)
    for d in duty.values():
        assert d['alarmed'] == alarmed
        if d['continuous'] and not alarmed:
            assert d['neutral_a'] < 20
    if r_fault == 0:
        assert sc['alarms'] == [] and s['n_grounding_continuous'] == 0
    else:
        assert s['n_grounding_continuous'] == len(duty)
        assert s['n_grounding_continuous_unalarmed'] == (0 if alarmed else len(duty))
        zigzags = {'Zigzag 2 breaker'} if alone else {'Zigzag breaker', 'Zigzag 2 breaker'}
        assert set(sc['alarms']) == (zigzags if alarmed else set())


# --- Phase 40: the backbone's phase faults, IEC against EMT ------------------------------------------

def _emt_bb2_fault(client, quiet, kind, k=None):
    with open(os.path.join(HERE, 'reference', f'{GRID}.diagram_payload.json'), encoding='utf-8') as fh:
        p = json.load(fh)
    for v in p.values():
        if k and isinstance(v, dict) and str(v.get('typ', '')).startswith('PCS'):
            v['current_limit_pu'] = str(k)
    cell = {v.get('userFriendlyName'): v['name'] for v in p.values() if isinstance(v, dict) and 'name' in v}
    p['0'] = {'typ': 'EmtStudy Parameters', 'frequency': '60', 'time_step_us': '20', 'user_email': 't@t',
              'duration_ms': '150', 'ac_fault_bus': cell['Load backbone 2'], 'ac_fault_type': kind,
              'ac_fault_resistance_ohm': '0.001', 'ac_fault_time_ms': '20', 'ac_fault_duration_ms': '100'}
    with quiet():
        out = json.loads(client.post('/', json=p).get_data(as_text=True))
    assert 'emt' in out, out.get('message') or out.get('error') or str(out)[:500]
    return out['emt']


@pytest.mark.parametrize('k, iec_a, emt_a', [(1.2, 239, 239), (1.5, 299, 299)])
def test_three_phase_backbone_fault_iec_against_emt(client, quiet, k, iec_a, emt_a):
    """
    A bolted three-phase fault on BB2 (Phase 40). Every inverter at its
    current limit - each block's T2 1.26 pu of its 63 A - with next to no
    negative sequence: IEC's current sources at k I_rated are right, EMT within
    1 % (241 A and 301 A before Phase 41 held each phase at its limit). The
    relays' settings rest on it.
    """
    out, label, _ = _drawn_study(client, quiet, {**IEC, 'fault_type': '3ph'}, k_pcs=k)
    iec = _bus_ka(out, label)['Load backbone 2'] * 1e3
    emt = _emt_bb2_fault(client, quiet, 'abc', k)
    assert iec == pytest.approx(iec_a, abs=2)
    for ph in emt['ac']['fault']['phases']:
        assert ph['i_sym_ka'] * 1e3 == pytest.approx(emt_a, rel=0.01)
        assert ph['i_sym_ka'] * 1e3 == pytest.approx(iec, rel=0.02)


def test_line_to_line_backbone_fault_at_the_phase_limit(client, quiet):
    """
    A bolted line-to-line fault on BB2 draws I1 = -I2; through T2's delta an
    inverter's worst phase carries twice either. Held at k I_r there (Phase
    41), IEC caps the fault at sqrt(3) x 239.5 A / 2 = 207.5 A - it read 236 A
    with the positive sequence alone limited. EMT, each phase now held at its
    limit too (its scaling down to 0.7), gives 185 A: the SSTs, their voltage
    only halved, keep drawing their load through the inverters, which IEC
    leaves out. Before, EMT's 262 A took the worst phase to some 1.6 pu. All
    are over three times the 87L and 87B's 60 A: the settings stand.
    """
    out, label, _ = _drawn_study(client, quiet, {**IEC, 'fault_type': '2ph'}, k_pcs=1.2)
    iec = _bus_ka(out, label)['Load backbone 2'] * 1e3
    emt = _emt_bb2_fault(client, quiet, 'bc')
    i_emt = max(ph['i_sym_ka'] for ph in emt['ac']['fault']['phases']) * 1e3
    assert iec == pytest.approx(math.sqrt(3) * 239.5 / 2, rel=0.005)
    assert i_emt == pytest.approx(185, rel=0.02) and i_emt < iec
    gfm = [c for c in emt['converters'] if 'GFM' in c['label']]
    assert all(c['phase_limit_min'] < 0.8 and c['phase_limited_ms'] > 50 for c in gfm)
    assert i_emt > 3 * 60


# --- Phase 42: harmonics on the inverter island -----------------------------------------------------

def _harmonic_study(client, quiet, tmp_path, spectrum=None, neglect_load_y=False, orders='5,7,11,13,17,19,23,25',
                    variant=None, extra=None):
    import opendssdirect as dss
    with open(os.path.join(HERE, 'reference', 'reference_ai_factory_800vdc.diagram_harmonic_payload.json'),
              encoding='utf-8') as fh:
        params = json.load(fh)['0']
    with open(os.path.join(HERE, 'reference', f'{GRID}.diagram_payload.json'), encoding='utf-8') as fh:
        p = json.load(fh)
    for v in p.values():
        if spectrum and isinstance(v, dict) and str(v.get('typ', '')).startswith('Solid-State'):
            v['spectrum'] = spectrum
        if variant and isinstance(v, dict):
            variant(v)
    if extra:
        cell = {v.get('userFriendlyName'): v['name'] for v in p.values() if isinstance(v, dict) and 'name' in v}
        for k, v in extra.items():
            p[k] = {**v, 'bus': cell[v['bus']]}
    p['0'] = {**params, 'harmonics': orders, 'neglectLoadY': neglect_load_y}
    label = {v['name']: v.get('userFriendlyName') for v in p.values() if isinstance(v, dict) and 'name' in v}
    before = dss.Basic.DataPath()
    dss.Basic.DataPath(str(tmp_path))
    try:
        with quiet():
            out = json.loads(client.post('/', json=p).get_data(as_text=True))
    finally:
        dss.Basic.DataPath(before)
    assert out['harmonic_analysis']['executed']
    return {label.get(b['name'], b['name']): b for b in out['busbars']}, out


def test_harmonics_reach_the_inverter_island(client, quiet, tmp_path):
    """
    The harmonic study (OpenDSS) on the reference, the SSTs and rectifiers
    injecting active front ends' spectra. Each grid-forming PCS of the
    backbone's island is a source behind its 0.15 pu reactor - EMT's - its
    EMF the load flow's terminal voltage plus its reactor's drop: before, a
    constant power source, the island had no voltage reference, OpenDSS left
    it unsolved and its buses read 0 % THD. Its fundamental is Electrisim's
    own load flow's, and every bus passes IEEE 519: the backbone 2.3 % THD
    against 5 %, the 505 V load sides 1.6 % against 8 %, the POI 0.4 %. With
    six-pulse SSTs the weak island fails at once - 17 % on the backbone, 12 %
    on the load sides - while the utility side, stiff, does not notice.
    """
    import opendss_electrisim as od
    buses, out = _harmonic_study(client, quiet, tmp_path)
    assert any("holds its island's voltage behind its 0.15 pu reactor" in w for w in out['warnings'])
    with open(os.path.join(HERE, 'reference', f'{GRID}.diagram_payload.json'), encoding='utf-8') as fh:
        p = json.load(fh)
    label = {v['name']: v.get('userFriendlyName') for v in p.values() if isinstance(v, dict) and 'name' in v}
    with quiet():
        lf = {label.get(b['name']): b['vm_pu'] for b in od._opendss_electrisim_load_flow(p, 60.0)['busbars']}
    for n in (1, 2, 3):
        bb = buses[f'Load backbone {n}']
        assert bb['vm_pu'] == pytest.approx(lf[f'Load backbone {n}'], abs=0.01)
        assert bb['vthd_percent'] == pytest.approx(2.26, abs=0.1) and bb['ieee519']['ok']
        ld = buses[f'Block {n} 505 V load side']
        assert ld['vthd_percent'] == pytest.approx(1.58, abs=0.1) and ld['ieee519']['ok']
        assert buses[f'Hall {n} 34.5 kV']['vthd_percent'] > 2.0
    assert buses['Utility 34.5 kV POI']['vthd_percent'] < 0.5
    six, _ = _harmonic_study(client, quiet, tmp_path, spectrum='six_pulse')
    for n in (1, 2, 3):
        assert six[f'Load backbone {n}']['vthd_percent'] > 15 and not six[f'Load backbone {n}']['ieee519']['ok']
        assert not six[f'Block {n} 505 V load side']['ieee519']['ok']
    assert six['Utility 34.5 kV POI']['ieee519']['ok']


def test_the_inverter_islands_resonance(client, quiet, tmp_path):
    """
    A frequency scan: Hall 2's SST injecting 1 % at every order to the 40th,
    the rest none. Undamped by the loads (neglectLoadY: an active front end, a
    controlled current source, is no resistance at a harmonic), the backbone's
    impedance peaks at the 20th, 1.2 kHz - the inverters' reactors and T2s,
    about 56 mH, against the cables' 0.3 uF - some 100 times its flat value;
    damped by the SSTs taken as their fundamental resistance, OpenDSS's
    default, it stays flat. Real damping - the SSTs' input filters, the
    inverters' own - lies between, and decides it: with the active front
    ends' own spectra, undamped, the 19th reaches 8.7 % on the backbone
    (IEEE 519: 3 %), 11.7 % THD.
    """
    orders = list(range(2, 41))
    flat = '\n'.join(f'{n},1.0,0' for n in orders)

    def one_source(v):
        if str(v.get('typ', '')).startswith('Solid-State'):
            v['spectrum'] = 'custom' if v.get('userFriendlyName') == 'Hall 2 SST' else 'none'
            v['spectrum_csv'] = flat
        if str(v.get('typ', '')).startswith('VSC'):
            v['spectrum'] = 'none'

    scans = {}
    for neglect in (False, True):
        buses, _ = _harmonic_study(client, quiet, tmp_path, neglect_load_y=neglect,
                                   orders=','.join(str(n) for n in orders), variant=one_source)
        scans[neglect] = {int(k): float(v) for k, v in buses['Load backbone 2']['harmonic_voltages_kv'].items() if float(v) > 0}
    peak = max(scans[True], key=scans[True].get)
    assert peak == 20 and scans[True][20] > 50 * scans[True][5]
    assert max(scans[False].values()) < 2 * min(v for k, v in scans[False].items() if k >= 5)
    buses, _ = _harmonic_study(client, quiet, tmp_path, neglect_load_y=True)
    bb = buses['Load backbone 2']
    nineteenth = 100 * float(bb['harmonic_voltages_kv']['19']) / bb['fundamental_voltage_kv']
    assert nineteenth == pytest.approx(8.7, abs=0.3) and not bb['ieee519']['ok']
    assert bb['vthd_percent'] == pytest.approx(11.7, abs=0.4)
    assert buses['Utility 34.5 kV POI']['ieee519']['ok']


# --- Phase 43: a damped harmonic filter on the backbone ---------------------------------------------

BACKBONE_FILTER = {'typ': 'Capacitor0', 'name': 'HF1', 'id': 'hf1', 'userFriendlyName': 'Backbone harmonic filter',
                   'bus': 'Load backbone 2', 'q_mvar': '1.0', 'loss_factor': '0', 'vn_kv': '34.5', 'step': '1',
                   'max_step': '1', 'filter_order': '5', 'filter_type': 'high_pass', 'filter_q': '2'}


def test_a_damped_filter_holds_the_island_within_ieee_519(client, quiet, tmp_path):
    """
    The backbone island resonates at the 20th; undamped by the SSTs (an active
    front end is no resistance at a harmonic) their 19th reaches 8.7 % there.
    A 1 Mvar high-pass filter on BB2 tuned to the 5th, its 496 ohm resistor
    across its reactor - about the island's characteristic impedance at its
    resonance - damps it: above the 5th the filter is that resistance. Tuned
    near the resonance, at the 17th, its capacitance shifts the resonance down
    onto the SSTs' 11th-13th and makes it worse, 13 %; smaller, 0.5 Mvar, it
    does not pass. With it,
    undamped, the backbone is 3.7 % THD, 2.0 % at the most at any order
    (IEEE 519: 5 % and 3 %), the 505 V load sides 2.6 %; damped by the loads,
    1.6 %. Its fundamental is the 1 Mvar it supplies, the bus voltages
    unchanged within 0.01 pu.
    """
    extra = {'hf': BACKBONE_FILTER}
    bare, _ = _harmonic_study(client, quiet, tmp_path, neglect_load_y=True)
    buses, out = _harmonic_study(client, quiet, tmp_path, neglect_load_y=True, extra=extra)
    for n in (1, 2, 3):
        bb = buses[f'Load backbone {n}']
        assert bb['vthd_percent'] == pytest.approx(3.73, abs=0.2) and bb['ieee519']['ok']
        assert bb['ieee519']['max_individual_percent'] < 2.5
        assert bb['vm_pu'] == pytest.approx(bare[f'Load backbone {n}']['vm_pu'], abs=0.01)
        assert buses[f'Hall {n} 34.5 kV']['ieee519']['ok']
        ld = buses[f'Block {n} 505 V load side']
        assert ld['vthd_percent'] < 3.0 and ld['ieee519']['ok']
    assert buses['Utility 34.5 kV POI']['ieee519']['ok']
    damped, _ = _harmonic_study(client, quiet, tmp_path, extra=extra)
    assert damped['Load backbone 2']['vthd_percent'] == pytest.approx(1.58, abs=0.2)
    tuned_high, _ = _harmonic_study(client, quiet, tmp_path, neglect_load_y=True,
                                    extra={'hf': {**BACKBONE_FILTER, 'filter_order': '17'}})
    assert tuned_high['Load backbone 2']['vthd_percent'] > bare['Load backbone 2']['vthd_percent']
    assert not tuned_high['Load backbone 2']['ieee519']['ok']


# --- Phase 44: the inverters' active damping in place of the filter -----------------------------

def test_the_inverters_active_damping_in_place_of_the_filter(client, quiet, tmp_path):
    """
    The filter's alternative: the grid-forming PCS damping the island
    themselves, their control drawing harmonic current in phase with their
    terminals' harmonic voltage - a virtual resistance across them at the
    harmonics (harmonic_r_pu), none at the fundamental. Undamped by the SSTs,
    1 pu on each of the twelve holds the backbone to 3.2 % THD, 1.7 % at any
    order - each inverter carrying some 2 % of its rating in harmonic
    current; 2.5 pu still passes (4.8 %), 3 pu does not (5.2 %, THD the
    limit, any one order within 3 % to some 6 pu). Its series counterpart, a
    virtual resistance in line with the reactor, does little - 1 pu leaves
    the 19th at 4.6 %: the island's resonance is the cables against the
    reactors and the T2s, as the SSTs' bus sees them. The fundamental is
    untouched.
    """
    def damped(r):
        def v(e):
            if str(e.get('typ', '')).startswith('PCS'):
                e['harmonic_r_pu'] = str(r)
        return v

    bare, _ = _harmonic_study(client, quiet, tmp_path, neglect_load_y=True)
    one, out = _harmonic_study(client, quiet, tmp_path, neglect_load_y=True, variant=damped(1.0))
    for n in (1, 2, 3):
        bb = one[f'Load backbone {n}']
        assert bb['vthd_percent'] == pytest.approx(3.21, abs=0.2) and bb['ieee519']['ok']
        assert bb['ieee519']['max_individual_percent'] == pytest.approx(1.74, abs=0.15)
        assert bb['vm_pu'] == pytest.approx(bare[f'Load backbone {n}']['vm_pu'], abs=1e-3)
        ld = one[f'Block {n} 505 V load side']
        assert ld['vthd_percent'] < 2.5 and ld['ieee519']['ok']
        assert ld['vm_pu'] == pytest.approx(bare[f'Block {n} 505 V load side']['vm_pu'], abs=1e-3)
    assert one['Utility 34.5 kV POI']['ieee519']['ok']
    edge, _ = _harmonic_study(client, quiet, tmp_path, neglect_load_y=True, variant=damped(2.5))
    assert edge['Load backbone 2']['ieee519']['ok'] and edge['Load backbone 2']['vthd_percent'] > 4.5
    weak, _ = _harmonic_study(client, quiet, tmp_path, neglect_load_y=True, variant=damped(3.0))
    assert not weak['Load backbone 2']['ieee519']['ok']
    assert weak['Load backbone 2']['ieee519']['max_individual_percent'] < 3.0


# --- Phase 45: energising the backbone filter on the island ---------------------------------------

def _energise_filter(client, quiet, controlled=False, load=1.0):
    with open(os.path.join(HERE, 'reference', f'{GRID}.diagram_payload.json'), encoding='utf-8') as fh:
        p = json.load(fh)
    cell = {v.get('userFriendlyName'): v['name'] for v in p.values() if isinstance(v, dict) and 'name' in v}
    for v in p.values():
        if isinstance(v, dict) and str(v.get('typ', '')).startswith('Load DC'):
            v['p_mw'] = str(float(v['p_mw']) * load)
    p['hf'] = {**BACKBONE_FILTER, 'bus': cell[BACKBONE_FILTER['bus']]}
    p['0'] = {'typ': 'EmtStudy Parameters', 'frequency': '60', 'time_step_us': '20', 'user_email': 't@t',
              'duration_ms': '250', 'ac_energise_shunt': 'HF1', 'ac_energise_time_ms': '40',
              'ac_energise_controlled': 'true' if controlled else 'false'}
    with quiet():
        out = json.loads(client.post('/', json=p).get_data(as_text=True))
    assert 'emt' in out, out.get('message') or out.get('error') or str(out)[:500]
    emt = out['emt']
    return emt, [c for c in emt['converters'] if c['kind'] == 'PCS']


def test_energising_the_filter_on_the_island(client, quiet):
    """
    The backbone filter (Section 6w) switched on with the island at full
    load, its breaker's three poles together. Its inrush, some 72 A, is
    small - its reactor holds it - but the inverters run at 0.95 of their
    rating with a 1.2 limit: the inrush and the bus's ringing (near 460 Hz,
    the filter against the inverters' reactors and T2s) take them to their
    current limit for 27 ms. Held at it they are current sources, and the
    SSTs - constant power - pull the backbone down to 0.71 pu: every SST
    blocks below its 0.8 pu, the racks are lost, the island's frequency
    rises to 61 Hz. Closed by a synchronous breaker - a and b at a zero of
    their line voltage, c a quarter cycle later - the inrush halves, the
    backbone stays within 0.98-1.02 pu and nothing blocks. Or the filter
    goes in before the load: at 85 % its poles together still ride through
    (the backbone at 0.93 pu), at 93 % only just (0.83 pu, the inverters
    limited 44 ms).
    """
    emt, pcs = _energise_filter(client, quiet)
    en = emt['energise']
    assert en['closed_ms'] == [40.0, 40.0, 40.0] and 0.05 < en['i_peak_ka'] < 0.1
    assert en['v_rms_min_pu'] < 0.8
    assert max(c['limited_ms'] for c in pcs) > 20
    assert {b['label'] for b in emt['converters_blocked']} >= {f'Hall {n} SST rectifier' for n in (1, 2, 3)}
    assert max(c['f_max_hz'] for c in pcs) > 60.5
    emt, pcs = _energise_filter(client, quiet, controlled=True)
    en = emt['energise']
    assert en['controlled'] and en['i_peak_ka'] < 0.04
    assert en['v_rms_min_pu'] > 0.97 and en['v_rms_max_pu'] < 1.03
    assert not emt['converters_blocked'] and max(c['limited_ms'] for c in pcs) < 10
    assert sum(c['p_end_mw'] for c in pcs) > 11
    emt, pcs = _energise_filter(client, quiet, load=0.85)
    assert not emt['converters_blocked'] and emt['energise']['v_rms_min_pu'] > 0.9


# --- The specification's own example: 30 MW of racks on nine blocks, N+1 ---------------------------

@pytest.fixture(scope='module')
def spec9():
    with open(os.path.join(HERE, 'reference', f'{GRID}_9blocks.spec.json'), encoding='utf-8') as fh:
        return json.load(fh)


@pytest.mark.parametrize('inlet, out_block, pcs_loading, battery_mw', [
    (40, None, 87.1, 0.0), (45, None, 91.4, 0.0), (50, None, 96.2, 0.0),
    (40, 9, 98.1, 0.65), (45, 9, 103.0, 2.08)])
def test_nine_blocks_carry_30_mw(spec9, inlet, out_block, pcs_loading, battery_mw):
    """
    The specification's example: 30 MW of racks (3.33 MW a hall) on nine
    blocks, a ring of nine backbone switchboards, one block and one hall on
    each - the 3-block reference scaled (its generator reproduces the 3-block
    spec exactly). With all nine in service the inverters carry it at 87 %
    of their 40 °C rating, 91 % at 45 °C and 96 % even at 50 °C; 92.8 % from
    the grid to the racks. N+1 at 40 °C holds: a block out, the other 32
    inverters run at 98 % - no shedding, where the 3-block reference's eight
    reach 142 % - and only the AFEs, current-limited on their 0.99 pu bus,
    reach their rating, the batteries making up 0.65 MW. At 45 °C it does
    not: 103 %, inside the 120 % ten-second rating but over the continuous,
    some 1 MW of racks to shed within 10 s (the hand check's 0.5 MW left out
    the T2, backbone and SST losses the inverters carry). Each case solves in
    seconds; nine blocks once took half an hour.
    """
    c = _case(spec9, inlet=inlet, out_block=out_block)
    assert len(c['pcs']) == (32 if out_block else 36) and _equal(c['pcs'])
    assert c['pcs'][0]['loading_percent'] == pytest.approx(pcs_loading, abs=0.3)
    assert c['bat_mw'] == pytest.approx(battery_mw, abs=0.05)
    if not out_block:
        assert 30.0 / c['grid_mw'] == pytest.approx(0.928, abs=0.002)
        assert c['pcs'][0]['frequency_hz'] == pytest.approx(60.0, abs=0.02)
    else:
        assert max(a['loading_percent'] for a in c['vsc']) == pytest.approx(100, abs=0.05)


# --- Nine blocks: Block 9 lost at full load, in time ---------------------------------------------------
# tests/reference/reference_pcs3750_mv_9blocks.diagram_payload.json: the nine-block spec drawn through the
# MCP server (875 cells) and its load-flow request captured from the page (prepareNetworkData).

def _payload9(inlet):
    with open(os.path.join(HERE, 'reference', f'{GRID}_9blocks.diagram_payload.json'), encoding='utf-8') as fh:
        p = json.load(fh)
    for v in p.values():
        if isinstance(v, dict) and 'inlet_c' in v:
            v['inlet_c'] = str(inlet)
    return p


def test_nine_blocks_drawn_give_the_specs_answer(client, quiet):
    """The drawn nine blocks' load flow: the 36 PCS at 87.1 %, as the spec's."""
    with quiet():
        out = json.loads(client.post('/', json=_payload9(40)).get_data(as_text=True))
    pcs = [g for g in out['pcs'] if g.get('p_mw') is not None]
    assert len(pcs) == 36 and _equal(pcs)
    assert pcs[0]['loading_percent'] == pytest.approx(87.1, abs=0.1)


@pytest.mark.parametrize('inlet, profile, i_after, i_end, bb_end', [
    (40, False, 0.9795, 0.9795, 0.9886), (45, False, 1.0285, 1.0009, 0.9626), (45, True, None, None, None)])
def test_andes_nine_blocks_lose_one(client, quiet, inlet, profile, i_after, i_end, bb_end):
    """
    Block 9's T2 out at 8 s, the racks at full load: the other 32 PCS take
    0.971 MW each in one step, no overshoot, at 59.87 Hz, their voltage
    control holding the backbone. At 40 deg C that is 98 % of their current
    and holds. At 45 deg C, 103 %: inside the 120 % ten-second rating; at 18 s
    their ceiling falls to 1 pu, they hold at it, and the backbone sags to
    0.963 pu, the racks' load falling with it - 0.92 MW each, some 1.6 MW short
    of what they need: the load to shed within 10 s.
    On the DGX H100 training profile, though, 45 deg C is no trouble: the
    cycle averages under the 32 PCS's rating and they never reach a ceiling.
    """
    p = _payload9(inlet)
    lp = {}
    if profile:
        with open(os.path.join(HERE, 'reference', f'{GRID}_9blocks.spec.json'), encoding='utf-8') as fh:
            lp = {x['id']: {'name': x['name'], 'kind': x['kind'], 'dt': x['dt_s'], 't0': 0, 'p': x['values']}
                  for x in json.load(fh)['load_profiles']}
    cell = {v.get('userFriendlyName'): v['name'] for v in p.values() if isinstance(v, dict) and 'name' in v}
    p['0'] = {'typ': 'TransientStabilityAndes Parameters', 'frequency': '60', 'sn_mva': '100', 'tf': '25',
              'tstep': '0', 'fault_bus': '', 'fault_enabled': 'false',
              'toggle_line': cell['Block 9 T2 0.505/34.5 kV Dy1 (505 V star floating)'], 'toggle_t': '8.0',
              'toggle_gen': '', 'poi_bus': cell['Utility 34.5 kV POI'], 'grid_voltage_profile': 'none',
              'load_profiles': lp, 'user_email': 't@t'}
    with quiet():
        out = json.loads(client.post('/', json=p).get_data(as_text=True))
    assert not out.get('error'), out.get('message')
    tm = out['time']
    left = [g for g in out['pcs'] if not g['label'].startswith('Block 9')]
    assert len(left) == 32
    if profile:
        for g in left:
            assert g['current_limited_from_s'] is None and g['peak_current_pu'] < 1.01
        return
    bb = next(b['values'] for b in out['bus_voltage'] if b['name'] == 'Load backbone 5')
    for g in left:
        assert _at(g, 'p_mw', 7.9, tm) == pytest.approx(0.865, abs=0.003)
        assert _at(g, 'p_mw', 12, tm) == pytest.approx(0.971, abs=0.003)
        assert _at(g, 'frequency_hz', 12, tm) == pytest.approx(59.87, abs=0.01)
        assert _at(g, 'current_pu', 12, tm) == pytest.approx(i_after, abs=0.003)
        assert g['peak_current_pu'] == pytest.approx(i_after, abs=0.003)          # no overshoot on the step
        assert _at(g, 'current_pu', 24.9, tm) == pytest.approx(i_end, abs=0.003)
    assert _at({'v': bb}, 'v', 24.9, tm) == pytest.approx(bb_end, abs=0.002)
    if inlet == 45:
        g = left[0]
        assert g['current_limited_from_s'] == pytest.approx(18.0, abs=0.1)
        assert _at(g, 'p_mw', 24.9, tm) == pytest.approx(0.92, abs=0.003)


def test_emt_nine_blocks_lose_one_at_45c(client, quiet):
    """
    The same loss in EMT, at 45 deg C: Block 9's T2 behind a 1 m feeder whose
    breakers open at 30 ms (each phase at its current's zero, by 35 ms). The
    32 PCS step from 0.88 to 0.99 MW, at their current limit some 9 ms of
    the first cycles (48 ms over 400 ms), the island to 59.85 Hz and the
    backbone to 0.973 pu; no SST blocks, the racks carried. The rectifiers,
    ramp-limited, do not move - the GS1.1 take the step. The transition is
    not where the trouble is; the ten seconds after it are (ANDES above).
    """
    p = _payload9(45)
    row = lambda name: next(v for v in p.values() if isinstance(v, dict) and v.get('userFriendlyName') == name)
    bb9, t2 = row('Load backbone 9'), row('Block 9 T2 0.505/34.5 kV Dy1 (505 V star floating)')
    k = max(int(x) for x in p if x.isdigit()) + 1
    p[str(k)] = dict(bb9, typ='Bus99', name='t2hv9', id='t2hv9', userFriendlyName='Block 9 T2 HV terminal')
    p[str(k + 1)] = dict(row('Backbone cable BB8-BB9'), typ='Line99', name='t2feed9', id='t2feed9',
                         userFriendlyName='Block 9 T2 feeder', busFrom=bb9['name'], busTo='t2hv9', length_km='0.001')
    t2['hv_bus'] = row('Block 9 T2 HV breaker')['bus'] = 't2hv9'
    p['0'] = {'typ': 'EmtStudy Parameters', 'frequency': '60', 'time_step_us': '20', 'user_email': 't@t',
              'duration_ms': '100', 'ac_trip_lines': 'Block 9 T2 feeder', 'ac_trip_time_ms': '30'}
    with quiet():
        emt = json.loads(client.post('/', json=p).get_data(as_text=True))['emt']
    (trip,) = emt['line_trips']
    assert all(30 <= t < 36 for t in trip['opened_ms'])
    assert not emt['converters_blocked']
    conv = {c['label']: c for c in emt['converters']}
    gfm = [c for n, c in conv.items() if 'GFM' in n and not n.startswith('Block 9')]
    assert len(gfm) == 32
    for c in gfm:
        assert c['p_start_mw'] == pytest.approx(0.876, abs=0.005) and c['p_end_mw'] == pytest.approx(0.987, abs=0.005)
        assert 5 < c['limited_ms'] < 15 and c['f_min_hz'] == pytest.approx(59.85, abs=0.01)
    for n, c in conv.items():
        if 'AFE' in n and not n.startswith('Block 9'):
            assert c['p_end_mw'] == pytest.approx(c['p_start_mw'], abs=0.001)
    bus = {b['label']: b for b in emt['ac']['buses']}
    for b in range(1, 10):
        assert bus[f'Load backbone {b}']['v_rms_min_pu'] == pytest.approx(0.973, abs=0.002)


# --- Phase 47: load shedding after a block's loss (ANDES) ----------------------------------------------

def _andes9_shed(client, quiet, inlet, trigger, profile=False, pickup='', step='5', delay='2', tf='25'):
    p = _payload9(inlet)
    lp = {}
    if profile:
        with open(os.path.join(HERE, 'reference', f'{GRID}_9blocks.spec.json'), encoding='utf-8') as fh:
            lp = {x['id']: {'name': x['name'], 'kind': x['kind'], 'dt': x['dt_s'], 't0': 0, 'p': x['values']}
                  for x in json.load(fh)['load_profiles']}
    cell = {v.get('userFriendlyName'): v['name'] for v in p.values() if isinstance(v, dict) and 'name' in v}
    p['0'] = {'typ': 'TransientStabilityAndes Parameters', 'frequency': '60', 'sn_mva': '100', 'tf': tf,
              'tstep': '0', 'fault_bus': '', 'fault_enabled': 'false',
              'toggle_line': cell['Block 9 T2 0.505/34.5 kV Dy1 (505 V star floating)'], 'toggle_t': '8.0',
              'toggle_gen': '', 'poi_bus': cell['Utility 34.5 kV POI'], 'grid_voltage_profile': 'none',
              'load_profiles': lp, 'user_email': 't@t', 'shed_trigger': trigger, 'shed_pickup': pickup,
              'shed_delay_s': delay, 'shed_step_percent': step}
    with quiet():
        out = json.loads(client.post('/', json=p).get_data(as_text=True))
    assert not out.get('error'), out.get('message')
    left = [g for g in out['pcs'] if not g['label'].startswith('Block 9')]
    return out, out['load_shedding'], left


def test_shedding_on_inverter_loading_at_45c(client, quiet):
    """
    Block 9 lost at full load and 45 deg C, a scheme watching the PCS's
    loading (pickup 1.0 pu of their rating, 2 s): one 5 % step of the nine
    halls' SSTs - 1.54 MW - at 10.03 s takes the 32 PCS from 102.9 % to
    97.8 % at 59.93 Hz, and they never reach their ten-second ceiling. A 3 %
    step (0.93 MW) only just does it (99.8 %); a delay of 9.5 s still sheds
    before the ceiling falls at 18 s.
    """
    out, shed, left = _andes9_shed(client, quiet, 45, 'loading')
    assert shed['enabled'] and len(shed['loads']) == 9 and all('SST' in n for n in shed['loads'])
    (s,) = shed['steps']
    assert s['t_s'] == pytest.approx(10.03, abs=0.03) and s['shed_mw'] == pytest.approx(1.54, abs=0.02)
    assert s['kept_percent'] == pytest.approx(95.0)
    for g in left:
        assert _at(g, 'current_pu', 20, out['time']) == pytest.approx(0.978, abs=0.003)
        assert _at(g, 'frequency_hz', 20, out['time']) == pytest.approx(59.926, abs=0.005)
        assert g['current_limited_from_s'] is None
    out3, small, left = _andes9_shed(client, quiet, 45, 'loading', step='3')
    assert small['shed_mw'] == pytest.approx(0.93, abs=0.02)
    assert _at(left[0], 'current_pu', 20, out3['time']) == pytest.approx(0.998, abs=0.003)
    _, late, left = _andes9_shed(client, quiet, 45, 'loading', delay='9.5')
    assert late['steps'][0]['t_s'] == pytest.approx(17.53, abs=0.03) and left[0]['current_limited_from_s'] is None


@pytest.mark.parametrize('inlet, trigger, profile, shed_mw', [
    (40, 'loading', False, 0.0), (45, 'loading', True, 0.0),
    (40, 'frequency', False, 1.54), (45, 'frequency', True, 2.74)])
def test_shedding_by_loading_not_by_frequency(client, quiet, inlet, trigger, profile, shed_mw):
    """
    What tells the scheme the PCS are over their rating. Their loading does:
    at 40 deg C after the loss (97 %) and on the DGX H100 training cycle at
    45 deg C (peaks of 1.00 pu, never 2 s long) it sheds nothing. The
    island's frequency does not: droop sets it by power, not by rating, so a
    block lost at 40 deg C (59.885 Hz) looks as one at 45 deg C (59.88), and
    a 59.9 Hz pickup sheds 1.54 MW there for nothing - and 2.74 MW of the
    training cycle, whose restarts take the island to 59.80 Hz.
    """
    pickup = '59.9' if trigger == 'frequency' else ''
    _, shed, left = _andes9_shed(client, quiet, inlet, trigger, profile=profile, pickup=pickup,
                                 tf='40' if profile else '25')
    assert shed['shed_mw'] == pytest.approx(shed_mw, abs=0.03)
    assert all(g['current_limited_from_s'] is None for g in left)


# --- Phase 48: the nine blocks' faults, protection and harmonics -----------------------------------------

def _study9(client, quiet, params, k_pcs=None, open_breakers=(), variant=None, extra=None, payload=None):
    p = payload or _payload9(40)
    for v in p.values():
        if not isinstance(v, dict):
            continue
        if k_pcs and str(v.get('typ', '')).startswith('PCS'):
            v['current_limit_pu'] = str(k_pcs)
        if v.get('userFriendlyName') in open_breakers:
            v['closed'] = 'false'
        if variant:
            variant(v)
    cell = {v.get('userFriendlyName'): v['name'] for v in p.values() if isinstance(v, dict) and 'name' in v}
    for k, v in (extra or {}).items():
        p[k] = {**v, 'bus': cell[v['bus']]}
    label = {v['name']: v.get('userFriendlyName') for v in p.values() if isinstance(v, dict) and 'name' in v}
    p['0'] = params
    with quiet():
        out = json.loads(client.post('/', json=p).get_data(as_text=True))
    assert not out.get('error'), out.get('message')
    return out, label, cell


I_LOAD9_KA = 30.8 / (3 ** 0.5 * 34.5)          # the nine blocks' backbone load: 0.515 kA


@pytest.mark.parametrize('fault, k, open_breakers, backbone_a', [
    ('3ph', 1.2, (), 718), ('3ph', 1.5, (), 898), ('2ph', 1.2, (), 623),
    ('1ph', 1.2, (), 838), ('1ph', 1.5, (), 838), ('1ph', 1.2, ('Zigzag 2 breaker',), 431),
    ('1ph', 1.2, ('Zigzag breaker', 'Zigzag 2 breaker'), 22.4)])
def test_nine_blocks_fault_levels(client, quiet, fault, k, open_breakers, backbone_a):
    """
    IEC 60909 on the nine blocks. A backbone phase fault is three times the
    three blocks' - 718 A three-phase at 1.2 pu, 898 A at 1.5, 623 A line to
    line - and still only 1.2-1.7 times the 515 A load: overcurrent cannot
    see it, differential protection must. An earth fault is no longer the
    inverters' to limit but the zigzags': 838 A at 1.2 pu and 1.5 alike, 419 A
    through each - a little over their 400 A short-time rating, a small
    fraction of its I2t at the relays' times; with one, 431 A; with none, the
    nine cables' charging current, 22 A. The 505 V load side, fed by its own
    PCS and the other 32 through its T2: 49 kA, three times the three blocks'
    16 kA - its switchgear's rating and arc flash rise with it. The grid side
    and the POI are as the reference's.
    """
    out, label, _ = _study9(client, quiet, {**IEC, 'fault_type': fault}, k_pcs=k, open_breakers=open_breakers)
    ka = {label[r['name']]: r.get('ikss_ka', r.get('i_first_sym_ka')) for r in out['busbars']}
    for n in range(1, 10):
        assert ka[f'Load backbone {n}'] * 1e3 == pytest.approx(backbone_a, rel=0.01)
    if fault == '3ph' and k == 1.2:
        assert ka['Load backbone 1'] / I_LOAD9_KA == pytest.approx(1.39, abs=0.02)
        assert ka['Block 5 505 V load side'] == pytest.approx(49.1, abs=1.5)
        assert ka['Block 1 505 V grid side'] == pytest.approx(68.2, abs=0.3)
        assert ka['Utility 34.5 kV POI'] == pytest.approx(8.37, abs=0.02)


def _protection_payload9():
    """
    The nine blocks' protection request: the drawn nine blocks' cells, each
    breaker's relay as the three-block drawing's protection request carries
    its counterpart's (Block n's as Block 1's, each ring cable's ends as
    BB1-BB2's), checked against the nine-block spec's settings.
    """
    import re
    p = _payload9(40)
    with open(os.path.join(HERE, 'reference', f'{GRID}.diagram_protection_payload.json'), encoding='utf-8') as fh:
        p3 = json.load(fh)
    with open(os.path.join(HERE, 'reference', f'{GRID}.diagram_payload.json'), encoding='utf-8') as fh:
        base = set(next(v for v in json.load(fh).values() if isinstance(v, dict) and str(v.get('typ', '')).startswith('Switch')))
    with open(os.path.join(HERE, 'reference', f'{GRID}_9blocks.spec.json'), encoding='utf-8') as fh:
        spec = {w['name']: w for w in json.load(fh)['switches']}
    tmpl = {v['userFriendlyName']: v for v in p3.values() if isinstance(v, dict) and str(v.get('typ', '')).startswith('Switch')}

    def role(name):
        m = re.match(r'Cable BB(\d+)-BB(\d+) breaker at BB(\d+)', name)
        if m:
            return 'Cable BB1-BB2 breaker at BB1' if m.group(3) == m.group(1) else 'Cable BB1-BB2 breaker at BB2'
        return re.sub(r'^(Block|Hall) \d+', r'\1 1', name)

    for v in p.values():
        if not (isinstance(v, dict) and str(v.get('typ', '')).startswith('Switch')):
            continue
        v.update({k: x for k, x in tmpl[role(v['userFriendlyName'])].items() if k not in base})
        for k, x in spec[v['userFriendlyName']].items():
            if k not in ('id', 'bus', 'element', 'et', 'name'):
                assert str(v[k]) == str(x) or float(v[k]) == pytest.approx(float(x)), (v['userFriendlyName'], k)
    p['0'] = dict(p3['0'])
    return p


@pytest.mark.parametrize('fault', ['3ph', '1ph'])
def test_nine_blocks_protection_holds(client, quiet, fault):
    """
    The reference's relays on the nine blocks, every one of the nine ring
    cables and the hall feeders faulted in turn: each cleared by its own
    87L in 0.05 s, no miscoordination, no unwanted trip, no breaker failure
    left uncleared. The settings were chosen on fault currents a third of
    these and keep their margins; the zigzags' duty, both in service and the
    87L clearing, is about 1 % of their I2t (3 % on the 51N's backup).
    """
    p = _protection_payload9()
    p['0'] = {**p['0'], 'fault_type': fault, 'fault_location_mode': 'line', 'sc_line_id': 'all',
              'fault_bus_id': '', 'user_email': 't@t'}
    with quiet():
        out = json.loads(client.post('/', json=p).get_data(as_text=True))
    assert not out.get('error'), out.get('message')
    s = out['summary']
    assert s['n_scenarios'] == 18 and s['n_devices'] == 65 and s['n_not_computed'] == 0
    assert s['n_miscoordination'] == 0 and s['n_unwanted_trips'] == 0 and s['n_breaker_failure_uncleared'] == 0
    first = {t['user_friendly_name']: t['t_trip_s'] for t in out['scenarios'][0]['trip']}
    assert first['Cable BB1-BB2 breaker at BB1'] == pytest.approx(0.05) and first['Cable BB1-BB2 breaker at BB2'] == pytest.approx(0.05)
    if fault == '1ph':
        assert s['max_grounding_duty_percent'] == pytest.approx(1.1, abs=0.2)
        assert s['max_grounding_duty_backup_percent'] == pytest.approx(2.7, abs=0.3)


def test_nine_blocks_arc_flash_by_the_relays(client, quiet):
    """
    Arc flash with the relays clearing, on the nine blocks. The 505 V grid
    side is the reference's (T1 LV's I>>, 6.4 cal/cm2, PPE 2). The 505 V load
    side's 48 kA arc, three times the reference's, is still parted by its
    arc-flash detection in 0.06 s: 3.2 cal/cm2, PPE 1 (the reference's 1.2,
    PPE 0). The backbone's arcs, 0.72 kA, cleared by 87B in 0.1 s - 1.5
    cal/cm2; a hall's by its feeder in 0.15 s - 2.3 cal/cm2. Every bus stays
    within PPE 2.
    """
    p = _protection_payload9()
    p['0'] = {'typ': 'ArcFlashPandaPower Parameters', 'electrode_config': 'VCB', 'equipment_mode': 'by_voltage',
              'clearing_time_mode': 'relays', 'frequency': '60', 'user_email': 't@t'}
    with quiet():
        out = json.loads(client.post('/', json=p).get_data(as_text=True))
    assert not out.get('error'), out.get('message')
    rows = {r['name']: r for r in out['arc_flash']}
    for n in range(1, 10):
        g = rows[f'Block {n} 505 V grid side']
        assert g['incident_energy_cal_cm2'] == pytest.approx(6.4, abs=0.1) and str(g['ppe_category']) == '2'
        ld = rows[f'Block {n} 505 V load side']
        assert ld['clearing_relays'] == [f'Block {n} T2 LV breaker'] and ld['clearing_time_used_s'] == pytest.approx(0.06)
        assert ld['ikss_ka'] == pytest.approx(48.3, abs=1.0)
        assert ld['incident_energy_cal_cm2'] == pytest.approx(3.17, abs=0.1) and str(ld['ppe_category']) == '1'
        bb = rows[f'Load backbone {n}']
        assert bb['clearing_time_used_s'] == pytest.approx(0.1) and bb['incident_energy_cal_cm2'] == pytest.approx(1.53, abs=0.05)
        h = rows[f'Hall {n} 34.5 kV']
        assert h['clearing_time_used_s'] == pytest.approx(0.15) and h['incident_energy_cal_cm2'] == pytest.approx(2.30, abs=0.05)


def _harmonics9(client, quiet, tmp_path, neglect=False, orders='5,7,11,13,17,19,23,25', variant=None, extra=None):
    import opendssdirect as dss
    with open(os.path.join(HERE, 'reference', 'reference_ai_factory_800vdc.diagram_harmonic_payload.json'),
              encoding='utf-8') as fh:
        params = json.load(fh)['0']
    before = dss.Basic.DataPath()
    dss.Basic.DataPath(str(tmp_path))
    try:
        out, label, _ = _study9(client, quiet, {**params, 'harmonics': orders, 'neglectLoadY': neglect},
                                variant=variant, extra=extra)
    finally:
        dss.Basic.DataPath(before)
    assert out['harmonic_analysis']['executed']
    return {label.get(b['name'], b['name']): b for b in out['busbars']}


def test_nine_blocks_harmonics(client, quiet, tmp_path):
    """
    The nine blocks' island resonates where the three blocks' does, at the
    20th: three times the inverters' reactors and T2s in parallel against
    three times the cables. Damped by the SSTs every bus passes (backbone
    2.2 % THD); undamped, the 19th reaches 8.1 %, 10.8 % THD - as before. But
    the island's impedance is a third of the reference's, and the remedy
    must scale with it: the reference's 1 Mvar filter leaves 6.1 % THD and
    fails; 3 Mvar - on BB5, or 1 Mvar on each of BB2, BB5 and BB8 - passes
    (3.4 %). The inverters' own damping, a resistance in each PCS's own per
    unit, scales by itself: 1 pu holds 3.0 %, and even 3 pu passes (4.8 %),
    which on three blocks did not.
    """
    def show(buses):
        return [buses[f'Load backbone {n}'] for n in range(1, 10)]

    damped = show(_harmonics9(client, quiet, tmp_path))
    assert all(b['vthd_percent'] == pytest.approx(2.20, abs=0.1) and b['ieee519']['ok'] for b in damped)
    bare = _harmonics9(client, quiet, tmp_path, neglect=True)
    b2 = bare['Load backbone 2']
    assert b2['vthd_percent'] == pytest.approx(10.8, abs=0.3) and not b2['ieee519']['ok']
    assert 100 * float(b2['harmonic_voltages_kv']['19']) / b2['fundamental_voltage_kv'] == pytest.approx(8.07, abs=0.2)
    orders = list(range(2, 41))
    flat = '\n'.join(f'{n},1.0,0' for n in orders)

    def one_source(v):
        if str(v.get('typ', '')).startswith('Solid-State'):
            v['spectrum'] = 'custom' if v.get('userFriendlyName') == 'Hall 2 SST' else 'none'
            v['spectrum_csv'] = flat
        if str(v.get('typ', '')).startswith('VSC'):
            v['spectrum'] = 'none'

    scan = _harmonics9(client, quiet, tmp_path, neglect=True, orders=','.join(map(str, orders)), variant=one_source)
    scan = {int(k): float(v) for k, v in scan['Load backbone 2']['harmonic_voltages_kv'].items() if float(v) > 0}
    assert max(scan, key=scan.get) == 20 and scan[20] > 50 * scan[5]

    def filters(q, where):
        return {f'hf{i}': {**BACKBONE_FILTER, 'name': f'HF{i}', 'id': f'hf{i}', 'userFriendlyName': f'Backbone filter {i}',
                           'bus': b, 'q_mvar': str(q)} for i, b in enumerate(where)}

    one = show(_harmonics9(client, quiet, tmp_path, neglect=True, extra=filters(1.0, ['Load backbone 2'])))
    assert one[1]['vthd_percent'] == pytest.approx(6.07, abs=0.2) and not one[1]['ieee519']['ok']
    for extra in (filters(3.0, ['Load backbone 5']), filters(1.0, ['Load backbone 2', 'Load backbone 5', 'Load backbone 8'])):
        held = show(_harmonics9(client, quiet, tmp_path, neglect=True, extra=extra))
        assert all(b['vthd_percent'] == pytest.approx(3.44, abs=0.15) and b['ieee519']['ok'] for b in held)
    for r, thd in ((1.0, 2.97), (3.0, 4.83)):
        def damp(v, r=r):
            if str(v.get('typ', '')).startswith('PCS'):
                v['harmonic_r_pu'] = str(r)
        held = show(_harmonics9(client, quiet, tmp_path, neglect=True, variant=damp))
        assert all(b['vthd_percent'] == pytest.approx(thd, abs=0.15) and b['ieee519']['ok'] for b in held)


# --- Phase 49: the nine blocks' DC fault, filter energising and a shed step -------------------------------

def test_nine_blocks_dc_bus_fault_on_a_quadrant(client, quiet):
    """
    Block 5's DC1 bolted on the nine blocks: as the reference's - the GS1.1
    pair 51.4 kA (25.7 kA a unit), the rectifier's diodes 8.7 kA from its
    T1 - but the inverter's diodes 9.1 kA from a 505 V load bus the other 35
    PCS hold stiffer (7.2 kA on three blocks): 69.2 kA in all (67.3). Only
    the faulted quadrant's sources feed it.
    """
    from test_dc_fault import STUDY
    p = _payload9(40)
    cell = {v.get('userFriendlyName'): v['name'] for v in p.values() if isinstance(v, dict) and 'name' in v}
    out, _, _ = _study9(client, quiet, {**STUDY, 'fault_bus': cell['Block 5 DC1'], 'duration_ms': '20', 'frequency': '60'},
                        payload=p)
    (f,) = out['dcfault']['faults']
    c = {(x['kind'], x['name']): x for x in f['contributions']}
    assert c[('Battery', 'Block 5 DC1 GS1.1 A')]['ik_ka'] == pytest.approx(25.70, abs=0.05)
    assert c[('VSC (diodes, blocked)', 'Block 5 AFE 1 (6 x PCS157SE)')]['ik_ka'] == pytest.approx(8.68, abs=0.1)
    assert c[('PCS (diodes, blocked)', 'Block 5 GFM 1 (6 x PCS157SE)')]['ik_ka'] == pytest.approx(9.13, abs=0.1)
    assert f['ik_ka'] == pytest.approx(69.2, abs=0.3)
    assert all(abs(x['ik_ka']) < 1e-3 for (k, n), x in c.items() if not n.startswith('Block 5'))


@pytest.mark.parametrize('controlled, i_peak_a, limited_ms', [(False, 81, 2.0), (True, 32, 0.0)])
def test_nine_blocks_energising_one_of_three_filters(client, quiet, controlled, i_peak_a, limited_ms):
    """
    The nine blocks' filtering as Phase 48 sized it - 1 Mvar on each of BB2,
    BB5 and BB8 - with BB5's switched on at 40 ms at full load. On three
    blocks the 1 Mvar filter closed with its poles together took the
    inverters to their limit for 27 ms and every SST blocked; here each unit
    is a third of the island's share, and the 36 PCS carry 87 % of their
    rating: an 81 A inrush, the backbone within 0.99-1.01 pu, the inverters
    at their limit 2 ms, no SST blocking. A synchronous breaker brings it to
    32 A and never touches the limit.
    """
    p = _payload9(40)
    cell = {v.get('userFriendlyName'): v['name'] for v in p.values() if isinstance(v, dict) and 'name' in v}
    for i, bb in enumerate(('Load backbone 2', 'Load backbone 5', 'Load backbone 8')):
        p[f'hf{i}'] = {**BACKBONE_FILTER, 'name': f'HF{i}', 'id': f'hf{i}', 'userFriendlyName': f'Backbone filter {i}',
                       'bus': cell[bb]}
    p['0'] = {'typ': 'EmtStudy Parameters', 'frequency': '60', 'time_step_us': '20', 'user_email': 't@t',
              'duration_ms': '200', 'ac_energise_shunt': 'HF1', 'ac_energise_time_ms': '40',
              'ac_energise_controlled': 'true' if controlled else 'false'}
    with quiet():
        emt = json.loads(client.post('/', json=p).get_data(as_text=True))['emt']
    e = emt['energise']
    assert e['bus'] == 'Load backbone 5' and e['i_peak_ka'] * 1e3 == pytest.approx(i_peak_a, abs=3)
    assert 0.99 < e['v_rms_min_pu'] and e['v_rms_max_pu'] < 1.01
    assert not emt['converters_blocked']
    pcs = [c for c in emt['converters'] if c['kind'] == 'PCS']
    assert len(pcs) == 36 and all(c['limited_ms'] == pytest.approx(limited_ms, abs=0.5) for c in pcs)


def test_emt_nine_blocks_a_shed_step_at_45c(client, quiet):
    """
    Phase 47's shedding step in EMT: Block 9 out, 45 deg C, the 32 PCS at
    103 % of their rating; at 30 ms 1.5 MW of racks - one 5 % step of the
    nine halls, here 45 % of Hall 5's - comes off. The PCS fall from 0.99 to
    0.94 MW each, the island's frequency rises by 0.06 Hz, the backbone
    holds above 0.98 pu and nothing blocks: the step is benign.
    """
    p = _payload9(45)
    for v in p.values():
        if isinstance(v, dict) and str(v.get('typ', '')).startswith('PCS') and v.get('userFriendlyName', '').startswith('Block 9'):
            v['in_service'] = 'false'
    p['0'] = {'typ': 'EmtStudy Parameters', 'frequency': '60', 'time_step_us': '20', 'user_email': 't@t',
              'duration_ms': '200', 'step_load': 'Hall 5: 90 DGX H100 racks', 'step_percent': '-45', 'step_time_ms': '30'}
    with quiet():
        emt = json.loads(client.post('/', json=p).get_data(as_text=True))['emt']
    assert not emt['converters_blocked']
    pcs = [c for c in emt['converters'] if c['kind'] == 'PCS' and not c['label'].startswith('Block 9')]
    assert len(pcs) == 32
    for c in pcs:
        assert c['p_start_mw'] == pytest.approx(0.987, abs=0.005) and c['p_end_mw'] == pytest.approx(0.937, abs=0.005)
        assert c['i_peak_ka'] < c['current_limit_ka']
    bus = {b['label']: b for b in emt['ac']['buses']}
    for n in range(1, 9):
        assert bus[f'Load backbone {n}']['v_rms_min_pu'] > 0.98


# --- The OPF on the inverter island ---------------------------------------------------------------

@pytest.mark.parametrize('nine, inlet', [(False, 40), (False, 45), (True, 40), (True, 45)])
def test_opf_on_the_inverter_island(client, quiet, nine, inlet):
    """
    The AC OPF on the drawn reference, three blocks and nine. A PCS on a DC
    network had a power window of its whole rating - its short-time overload's
    too - and its Q box was taken at that window's ends, where its rating
    leaves none: the backbone island, whose only sources they are, had no
    reactive power and the OPF never converged. Now its P is held to its
    continuous rating and its Q to its cap (60 %); every case converges, each
    PCS carrying its share within its rating (no warning of a dispatch past
    its circle), the backbone at the OPF's 1.10 pu bound (equal costs: the
    least losses).
    """
    name = f'{GRID}_9blocks' if nine else GRID
    with open(os.path.join(HERE, 'reference', f'{name}.diagram_payload.json'), encoding='utf-8') as fh:
        p = json.load(fh)
    for v in p.values():
        if isinstance(v, dict) and 'inlet_c' in v:
            v['inlet_c'] = str(inlet)
    p['0'] = {'typ': 'OptimalPowerFlowPandaPower Parameters', 'opf_type': 'ac', 'frequency': '60',
              'ac_algorithm': 'pypower', 'dc_algorithm': 'pypower', 'calculate_voltage_angles': 'auto', 'init': 'pf',
              'delta': '1e-16', 'trafo_model': 't', 'trafo_loading': 'current', 'ac_line_model': 'pi', 'numba': True,
              'suppress_warnings': True, 'cost_function': 'polynomial', 'user_email': 't@t',
              **{f'{e}_cost_{c}': {} for e in ('generator', 'ext_grid', 'storage', 'sgen', 'load', 'dcline')
                 for c in ('cp1', 'cp2')}}
    with quiet():
        out = json.loads(client.post('/', json=p).get_data(as_text=True))
    assert not out.get('error'), out.get('exception')
    pcs = out['generators']
    assert len(pcs) == (36 if nine else 12)
    rating = 0.994 if inlet == 40 else 0.947
    for g in pcs:
        assert 0.85 < g['p_mw'] < rating and abs(g['q_mvar']) < 0.1
        assert math.hypot(g['p_mw'], g['q_mvar']) < rating
    assert not [w for w in out.get('warnings', []) if 'OPF dispatched' in w]
    assert max(b['vm_pu'] for b in out['busbars']) == pytest.approx(1.10, abs=0.002)


def test_opf_says_when_the_island_cannot_be_supplied(client, quiet):
    """
    Block 3 out: its eight PCS at their continuous rating, 7.95 MW, against
    the three blocks' 11.1 MW. The load flow carries it on their short-time
    overload; no steady-state dispatch exists, and the OPF says so - the
    island, its load and its sources' limit - in the error details the dialog
    shows, where it had said only that it did not converge.
    """
    with open(os.path.join(HERE, 'reference', f'{GRID}.diagram_payload.json'), encoding='utf-8') as fh:
        p = json.load(fh)
    for v in p.values():
        if isinstance(v, dict) and str(v.get('typ', '')).startswith('PCS') and v['userFriendlyName'].startswith('Block 3'):
            v['in_service'] = 'false'
    p['0'] = {'typ': 'OptimalPowerFlowPandaPower Parameters', 'opf_type': 'ac', 'frequency': '60',
              'ac_algorithm': 'pypower', 'dc_algorithm': 'pypower', 'calculate_voltage_angles': 'auto', 'init': 'pf',
              'delta': '1e-16', 'trafo_model': 't', 'trafo_loading': 'current', 'ac_line_model': 'pi', 'numba': True,
              'suppress_warnings': True, 'cost_function': 'polynomial', 'user_email': 't@t',
              **{f'{e}_cost_{c}': {} for e in ('generator', 'ext_grid', 'storage', 'sgen', 'load', 'dcline')
                 for c in ('cp1', 'cp2')}}
    with quiet():
        out = json.loads(client.post('/', json=p).get_data(as_text=True))
    assert out['error']
    msg = out['diagnostic']['general_error']          # the dialog's error details (the exception is suppressed)
    assert msg.startswith('The island with ') and 'Hall 1 34.5 kV' in msg
    assert 'draws 11.1 MW, more than its 8 source(s) can give at their limits (7.95 MW' in msg
    assert 'no dispatch exists' in msg


def test_load_flow_names_transformers_past_their_rating(client, quiet):
    """
    At 40 C full load each block's 3.75 MVA T1 carries the AFEs' 3.85 MW and
    its own loss: 103.9 % of its nameplate (the block is rated 3,977 kVA). The
    result boxes coloured it; the load flow now says so, as it does for a
    converter past its rating. Each T2, at 100.0 %, is not named.
    """
    with open(os.path.join(HERE, 'reference', f'{GRID}.diagram_payload.json'), encoding='utf-8') as fh:
        p = json.load(fh)
    with quiet():
        out = json.loads(client.post('/', json=p).get_data(as_text=True))
    over = [w for w in out['warnings'] if ' is loaded to ' in w and w.startswith(('Line', 'Transformer'))]
    assert len(over) == 3, over
    for k in (1, 2, 3):
        assert any(w.startswith(f"Transformer 'Block {k} T1") and 'loaded to 103.9 % of its rating' in w for w in over)
    assert not any('T2' in w for w in over)
