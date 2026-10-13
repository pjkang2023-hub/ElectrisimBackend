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
