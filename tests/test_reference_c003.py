"""
C003 off-grid microgrid (CAMPAIGN-003), the sixth reference: SOFC 39.6 MW (21 groups x 2.2 MW, grid-following)
and PCST5500 grid-forming storage on a 33 kV network, 31.6 MW of data halls and 4 MW of auxiliaries, 50 Hz.

Two designs from the owner's mgtwin Site artifacts (specs/make_c003_spec.py):
  ring - recommended design v1.3, Site v9: closed ring, five PCS, two zig-zag grounding transformers (NER 38 ohm);
  revc - Rev C.1, Site revC1: six-board dual-radial, six PCS, four zig-zag grounding transformers (NER 76 ohm).
The design case is end of life: each SOFC group 1.91 MW at its inverter terminals at 44 000 h (workbook
SOFC_EOL_inv), dispatched at the load plus 1 % over 21 groups (1.7122 MW each).

Checked against hand calculations and the owner's own figures: the designs' earth-fault currents (V_ph / R_N,
the workbook's 250.69 A), the power balance, the spec build and the drawn payload agreeing.
"""
import contextlib
import io
import json
import math
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
GROUPS = 21
P_GROUP = 35.6 * 1.01 / GROUPS          # 1.7122 MW


def _spec(design):
    with open(os.path.join(HERE, 'reference', f'reference_c003_{design}.spec.json'), encoding='utf-8') as fh:
        return json.load(fh)


def _payload(design):
    with open(os.path.join(HERE, 'reference', f'reference_c003_{design}.diagram_payload.json'), encoding='utf-8') as fh:
        return json.load(fh)


def _post(client, payload):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        response = client.post('/', json=payload)
    assert response.status_code == 200, response.get_data(as_text=True)[:400]
    out = json.loads(response.get_data(as_text=True))
    assert not out.get('error'), out.get('message')
    return out


@pytest.mark.parametrize('design, n_pcs', [('ring', 5), ('revc', 6)])
def test_spec_solves_off_grid(quiet, design, n_pcs):
    """
    Built from its spec: no external grid, the grid-forming PCS form the island at 50 Hz. The 21 SOFC groups carry
    the load at 1.7122 MW each; the PCS share the rest equally - absorbing the 1 % dispatched for losses less the
    network's (placeholder cables, 0.05 % transformer resistance: under 0.1 MW) - and supply the halls' reactive power.
    """
    import electrisim_sld as sld
    with quiet():
        net, _ = sld.build_network(_spec(design))
        res = sld.solve(net)
    assert res['converged'], res.get('hint')
    pcs = res['pcs']
    gfm = [p for p in pcs if not str(p['id']).startswith('SOFC')]
    gfl = [p for p in pcs if str(p['id']).startswith('SOFC')]
    assert len(gfm) == n_pcs and len(gfl) == GROUPS
    assert sum(p['p_mw'] for p in gfl) == pytest.approx(GROUPS * P_GROUP, abs=2e-3)
    p = [g['p_mw'] for g in gfm]
    assert max(p) - min(p) < 1e-3 and all(x < 0 for x in p)
    losses = GROUPS * P_GROUP - 35.6 + sum(p)
    assert 0 < losses < 0.1
    assert all(abs(g['frequency_hz'] - 50.0) < 0.05 for g in gfm)


@pytest.mark.parametrize('design, n_pcs, v_min', [('ring', 5, 0.9500), ('revc', 6, 0.9590)])
def test_drawn_load_flow(client, design, n_pcs, v_min):
    """
    The drawn network, as the app sends it: no warnings, 35.6 MW served, each SOFC inverter at 1.7122 MW (77.8 % of
    its 2.2 MVA), the PCS sharing the reactive power equally. With only the PCS's Q-V droop holding the voltage
    (the SOFC inverters at unity power factor), the lowest bus sits at the bottom of the band: 0.950 pu on the ring
    (a SOFC group's 400 V bus), 0.959 pu on Rev C - a finding: the EMS's secondary voltage control must restore it.
    """
    out = _post(client, _payload(design))
    assert not out.get('warnings'), out.get('warnings')
    assert sum(ld['p_mw'] for ld in out['loads']) == pytest.approx(35.6, abs=1e-6)
    inv = [p for p in out['pcs'] if p['source_kind'] == 'SOFC']
    gfm = [p for p in out['pcs'] if p['source_kind'] == 'Battery']
    assert len(inv) == GROUPS and len(gfm) == n_pcs
    assert all(p['p_mw'] == pytest.approx(P_GROUP, abs=1e-4) and p['loading_percent'] == pytest.approx(77.83, abs=0.05)
               for p in inv)
    q = [p['q_mvar'] for p in gfm]
    assert max(q) - min(q) < 0.01 and min(q) > 1.5
    assert min(b['vm_pu'] for b in out['busbars']) == pytest.approx(v_min, abs=5e-4)


@pytest.mark.parametrize('design, n_gt, r_n, i_hand', [('ring', 2, 38, 501.4), ('revc', 4, 76, 250.7)])
def test_grounding_transformers_give_the_designs_earth_fault(client, design, n_gt, r_n, i_hand):
    """
    Each zig-zag with its NER passes V_ph / R_N: 501.4 A for the ring's 38 ohm (the design note's ~500 A), 250.7 A for
    Rev C's 76 ohm (the workbook's 250.69 A); with the zig-zag's own zero-sequence impedance, a little less.
    """
    assert 33e3 / math.sqrt(3) / r_n == pytest.approx(i_hand, abs=0.1)
    out = _post(client, _payload(design))
    gts = out['groundingtransformers']
    assert len(gts) == n_gt and all(g['r_n_ohm'] == r_n for g in gts)
    assert all(0.99 * i_hand < g['i_ground_alone_a'] < i_hand for g in gts)


# --- Phase 2: the owner's operating states ----------------------------------------------------------------------
SITES = {'ring': 'c003_site_v9.json', 'revc': 'c003_revC1_site.json',
         'ring10': 'c003_site_v10.json', 'revc2': 'c003_revC2_site.json'}     # the owner's data of 20 October: test_reference_c003_v2


def _site(design):
    with open(os.path.join(HERE, 'reference', SITES[design]), encoding='utf-8') as fh:
        return json.load(fh)


def _state(design, state_id, typ='PowerFlowPandaPower Parameters', extra=None, load_scale=None, also_off=()):
    """
    The drawn payload in one of the Site's operating states: its switches opened, its sources out of service. As
    mgtwin (v0.6.5) reads a state: the payload draws normally open switches open; the state opens its
    open_switches and closes its closed_switches (a transfer's second legs, a tie); its load_factors scale its
    loads (v0.6.10: the dual-corded halls moving to the other cord).
    """
    import copy
    site = _site(design)
    state = next(s for s in site['islands'] if s['id'] == state_id)
    sources = [c['id'] for c in site['converters']] + [m['id'] for m in site['machines']]
    on = set(state['energized_source_ids']) - set(also_off)
    opened = set(state.get('open_switches') or [])
    closes = set(state.get('closed_switches') or [])
    # the state's load transfer (mgtwin v0.6.10 load_factors); a load_scale given here replaces its factor
    load_scale = {**(state.get('load_factors') or {}), **(load_scale or {})}
    p = copy.deepcopy(_payload(design))
    p['0'] = {**p['0'], 'typ': typ, **(extra or {})}

    def label(s):
        return f'{s} inverter' if s.startswith('SOFC') else (f'{s} PCST5500' if s.startswith('PCS') else f'{s} genset')

    for v in p.values():
        if not isinstance(v, dict):
            continue
        name, t = str(v.get('userFriendlyName', '')), str(v.get('typ', ''))
        if t.startswith('Switch') and name in opened:
            v['closed'] = 'false'
        if t.startswith('Switch') and name in closes:
            v['closed'] = 'true'
        for s in sources:
            if name.startswith(label(s)):
                v['in_service'] = 'true' if s in on else 'false'
        if t.startswith('Generator') and name.startswith('GEN_BS') and 'GEN_BS' in on:
            v['slack'] = 'true'
        for lid, k in (load_scale or {}).items():
            if t.startswith('Load') and name.split(' ')[0] == lid:
                v['p_mw'], v['q_mvar'] = str(float(v['p_mw']) * k), str(float(v['q_mvar']) * k)
    return p


OPF = {'frequency': '50', 'objective': 'shed', 'fuel_cells': 'true', 'batteries': 'false', 'branch_limits': 'true',
       'shed_ac_loads': 'true'}


def _margin(out):
    """The sustained margin (SOFC capability less load served and network losses) and the batteries' discharge."""
    src = [s for s in out['sources'] if s['kind'] == 'SOFC']
    served = sum(l['p_full_mw'] - l['shed_mw'] for l in out['loads_ac'])
    gfm = [x['p_mw'] for x in out['pcs'] if 'PCST' in str(x['name'])]
    losses = sum(s['p_mw'] for s in src) + sum(gfm) - served
    return sum(s['p_max_mw'] for s in src) - served - losses, sum(max(x, 0.0) for x in gfm)


@pytest.mark.parametrize('design', ['ring', 'revc'])
def test_n_minus_1_sofc_at_end_of_life(client, design):
    """
    One SOFC group out at 44 000 h: 20 x 1.91 MW against 35.6 MW and the network's losses - +2.52 MW, nothing shed,
    no battery needed. The owner's +1.5 to +2.0 MW counts 1.5 % in each group transformer (1.91 -> 1.881 MW) and
    1-1.5 % in the network; the Site's own impedances lose under 0.1 MW (cables placeholders, A18).
    """
    out = _post(client, _state(design, 'N1_SOFC', 'AcDcOpfPandaPower Parameters', OPF))
    assert out['solved'] and out['shed_mw'] == 0
    margin, battery = _margin(out)
    assert margin == pytest.approx(2.52, abs=0.03) and battery < 0.01


def test_split_ring_half_a_short_at_n_minus_1(client):
    """
    The ring split (CB-SEG-AB1 and the tie open), one group out of half A: no load shed, but only because half A's
    three PCS discharge their batteries at 0.22 MW each - 9 x 1.91 MW against the 17.8 MW of switchboard MVA leaves
    half A 0.65 MW short, sustained (the owner's -0.87 MW, on 1.881 MW). Half B, eleven groups, is not.
    """
    out = _post(client, _state('ring', 'RING_SPLIT', 'AcDcOpfPandaPower Parameters', OPF, also_off=('SOFC01',)))
    assert out['solved'] and out['shed_mw'] == 0
    assert _margin(out)[1] == pytest.approx(0.65, abs=0.03)
    out = _post(client, _state('ring', 'RING_SPLIT', 'AcDcOpfPandaPower Parameters', OPF, also_off=('SOFC11',)))
    assert out['solved'] and out['shed_mw'] == 0 and _margin(out)[1] < 0.01


def test_revc_section_fault_carried_by_the_batteries(client):
    """
    Rev C, board section 1A isolated before the transfer (its five groups off): 16 x 1.91 = 30.6 MW against 35.7 MW -
    the PCS carry 5.1 MW from their batteries until the groups are back (0.26 MWh over a 3 min slow transfer, inside
    the 2.6 MWh usable; the owner's 0.54 MWh is the conservative figure).
    """
    out = _post(client, _state('revc', 'SOFC1A_SECTION_FAULT_PRE', 'AcDcOpfPandaPower Parameters', OPF))
    assert out['solved'] and out['shed_mw'] == 0
    margin, battery = _margin(out)
    assert battery == pytest.approx(5.1, abs=0.05) and margin == pytest.approx(-5.1, abs=0.05)


@pytest.mark.parametrize('design, state, scale', [('ring', 'BUS_MVB_FAULT', {'DH_A': 2.0}),
                                                  ('revc', 'MVA_BUS_FAULT', {'DH3_cordB': 2.0, 'DH4_cordB': 2.0})])
def test_bus_fault_with_dual_corded_halls(client, design, state, scale):
    """
    A switchboard bus fault: as the Sites draw the halls (each cord a fixed load), 17.8 MW is lost; with the halls
    dual-corded (design note v1.2) their whole 31.6 MW moves to the other cord and only the 2 MW auxiliary is lost -
    33.6 MW served with +6.4 MW to spare.
    """
    lf = _post(client, _state(design, state))
    assert sum(l['p_mw'] for l in lf['loads']) == pytest.approx(17.8, abs=1e-6)
    out = _post(client, _state(design, state, 'AcDcOpfPandaPower Parameters', OPF, load_scale=scale))
    assert out['solved'] and out['shed_mw'] == 0
    margin, battery = _margin(out)
    assert margin == pytest.approx(6.41, abs=0.05) and battery < 0.01


def test_revc_genset_black_start_as_declared_overloads_the_machine(client):
    """
    Rev C's BLACKSTART_GEN state opens the A-side SOFC and BESS feeders but leaves the cord-A hall feeders closed:
    17.8 MW on the 3 MVA genset - its transformer at 854 %, the board at 0.72 pu. The twin's gates (grounding,
    forming) pass it; a load flow does not. The later states (first PCS transformer, PCS soft energisation) open the
    load breakers and carry Aux A alone: 2.0 MW, within the machine.
    """
    lf = _post(client, _state('revc', 'BLACKSTART_GEN'))
    assert any("'TR_GEN' is loaded to 85" in w for w in lf.get('warnings') or [])
    assert min(b['vm_pu'] for b in lf['busbars'] if b['vm_pu'] and b['vm_pu'] > 0) < 0.75
    ok = _post(client, _state('revc', 'BLACKSTART_GEN_FIRST_PCS_TR'))
    assert sum(l['p_mw'] for l in ok['loads']) == pytest.approx(2.0, abs=1e-6)
    assert not any('TR_GEN' in w for w in ok.get('warnings') or [])


def test_pcs_black_start_carrying_aux_a_runs_on_batteries(client):
    """
    Rev C, PCS 1-2 forming BESS-1 and the mesh with Aux A closed and no SOFC running: the two batteries carry the
    2.05 MW - their 1.27 MWh (70 % usable) lasts some 26 min against the SOFC's 5-6 h start-up (A11): the genset is
    needed for the start-up, not only for black start.
    """
    out = _post(client, _state('revc', 'BLACKSTART_BESS1_SOFC_PAIR', 'AcDcOpfPandaPower Parameters', OPF))
    assert out['solved'] and out['shed_mw'] == 0
    battery = _margin(out)[1]
    assert battery == pytest.approx(2.05, abs=0.03)
    assert 2 * 0.635 * 0.7 / battery * 60 == pytest.approx(26, abs=1)


def test_battery_asked_beyond_its_power_says_so(client):
    """
    Ring v9 has no load breakers: its PCS black-start state with the halls connected asks the PCS for 35.6 MW. The
    reference PCS's battery cannot deliver that at any terminal voltage - said so, at its maximum-power point, not
    as a C-rate of billions.
    """
    p = _state('ring', 'BLACKSTART_PCS')
    for v in p.values():
        if isinstance(v, dict) and str(v.get('typ', '')).startswith('Battery'):
            v['r0_mohm'] = '200'       # (at the reference's 20 mOhm it can deliver it: 31 C, said so as a C-rate)
    lf = _post(client, p)
    w = [x for x in lf.get('warnings') or [] if 'LTO' in x]
    assert any('more than it can deliver at any terminal voltage' in x for x in w), w
    rates = [float(x.split('discharging at ')[1].split(' C')[0]) for x in w if 'discharging at' in x]
    assert all(r < 1e3 for r in rates), rates


# --- Phase 3: short circuits and earth faults -------------------------------------------------------------------
SC = {'fault_bus_mode': 'all', 'fault_bus_ids': [], 'fault_bus_names': [], 'fault_impedance': '6', 'frequency': '50',
      'topology': 'auto', 'tk_s': '1', 'r_fault_ohm': '0', 'x_fault_ohm': '0', 'inverse_y': 'True',
      'exportPython': False, 'exportPandapowerResults': False, 'exportPdfReport': False}


def _bus_ka(design, out):
    names = {v['name']: v.get('userFriendlyName') for v in _payload(design).values()
             if isinstance(v, dict) and str(v.get('typ', '')).startswith('Bus')}
    return {names.get(b['name']): b.get('ikss_ka') for b in out['busbars']}


def _ka(mva):
    return mva / (math.sqrt(3) * 33.0)


@pytest.mark.parametrize('design, n_pcs', [('ring', 5), ('revc', 6)])
def test_three_phase_fault_level_is_the_converters_limits(client, design, n_pcs):
    """
    IEC 60909 maximum, three-phase, every 33 kV bus: the fleet at its current limits - 21 SOFC inverters x 2.2 MVA x
    1.2 and the PCS x 5.5 MVA x 1.1 - 1.499 kA on the ring, 1.605 kA on Rev C, by hand to the ampere. EMT, the
    converters' controls acting, gives 1.11-1.14 kA: the owner's ~1.1 kA.
    """
    out = _post(client, _state(design, 'NORMAL', 'ShortCircuitPandaPower Parameters',
                               {**SC, 'fault_type': '3ph', 'fault_location': 'max'}))
    ka = _bus_ka(design, out)
    hand = _ka(21 * 2.2 * 1.2 + n_pcs * 5.5 * 1.1)
    assert ka['MVA'] == pytest.approx(hand, abs=2e-3) and ka['MVB'] == pytest.approx(hand, abs=2e-3)


def test_genset_adds_its_subtransient_current(client):
    """
    With the 3 MVA genset running (x''d 0.15 behind its 6 % transformer, ~14 MVA): +0.29 kA at maximum, 1.78 kA - the
    owner's ~1.7 kA, though from a converter base of 1.5 kA, not 1.1; at IEC's minimum (converters neglected) the
    genset alone, 0.26 kA.
    """
    p = _state('ring', 'NORMAL', 'ShortCircuitPandaPower Parameters', {**SC, 'fault_type': '3ph', 'fault_location': 'max'})
    for v in p.values():
        if isinstance(v, dict) and str(v.get('userFriendlyName', '')).startswith('GEN_BS'):
            v['in_service'] = 'true'
    assert _bus_ka('ring', _post(client, p))['MVA'] == pytest.approx(1.784, abs=5e-3)
    p['0']['fault_location'] = 'min'
    assert _bus_ka('ring', _post(client, p))['MVA'] == pytest.approx(0.259, abs=5e-3)


def test_black_start_minimum_fault_level(client):
    """
    Rev C, PCS 1-2 forming BESS-1 alone: the rest of the network de-energised - left out of the study and named, not
    refused - and the fault level two PCS at their limit, 0.21 kA (2 x 5.5 x 1.1 MVA): a 7.6 : 1 range against the
    1.60 kA maximum, which phase overcurrent alone cannot span. The dead buses' results are null - a strict JSON
    parser (the browser's) reads the reply.
    """
    p = _state('revc', 'BLACKSTART_BESS1', 'ShortCircuitPandaPower Parameters', {**SC, 'fault_type': '3ph', 'fault_location': 'max'})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        text = client.post('/', json=p).get_data(as_text=True)

    def strict(x):
        raise ValueError(x)

    out = json.loads(text, parse_constant=strict)
    assert any('no source energises are left out of the short circuit' in w for w in out.get('warnings') or [])
    ka = _bus_ka('revc', out)
    assert ka['BESS1'] == pytest.approx(_ka(2 * 5.5 * 1.1), abs=2e-3)
    assert ka['MVA'] is None


def _emt(client, design, **params):
    p = _payload(design)
    cell = {v.get('userFriendlyName'): v['name'] for v in p.values() if isinstance(v, dict) and 'name' in v}
    p['0'] = {'typ': 'EmtStudy Parameters', 'frequency': '50', 'time_step_us': '20', 'user_email': 't@t',
              'duration_ms': '120', 'ac_fault_time_ms': '20', 'ac_fault_duration_ms': '80',
              **{k: (cell[v] if k == 'ac_fault_bus' else v) for k, v in params.items()}}
    return _post(client, p)['emt']


def test_emt_three_phase_fault_ridden_through(client):
    """
    EMT, a bolted three-phase fault on switchboard MVA for 80 ms: the converters current-limit and nothing blocks;
    1.11 kA symmetrical - the owner's ~1.1 kA - against IEC's 1.50 kA with every converter at its limit.
    """
    emt = _emt(client, 'ring', ac_fault_bus='MVA', ac_fault_type='abcg', ac_fault_resistance_ohm='0.001')
    assert not emt['converters_blocked']
    assert emt['ac']['fault']['phases'][0]['i_sym_ka'] == pytest.approx(1.107, abs=0.02)


def test_emt_earth_fault_through_resistance_ridden_through(client):
    """
    EMT, phase a to earth on MVA through 50 ohm: nothing blocks; 0.150 kA, half each zig-zag neutral. (A near-bolted
    earth fault - under some 10 ohm - blocks every PCS on overcurrent within 2-12 ms in EMT on both designs, where the
    sequence networks say it is benign behind the delta windings: an open item, not asserted here.)
    """
    emt = _emt(client, 'ring', ac_fault_bus='MVA', ac_fault_type='ag', ac_fault_resistance_ohm='50')
    assert not emt['converters_blocked']
    assert emt['ac']['fault']['phases'][0]['i_sym_ka'] == pytest.approx(0.150, abs=0.01)


# --- Phase 4: protection setting windows --------------------------------------------------------------------------
def _feeder_ka(client, design, state, fault_bus):
    """Each cable's three-phase fault current (A), IEC maximum, for a fault on one bus."""
    p = _payload(design)
    cell = {v.get('userFriendlyName'): v['name'] for v in p.values() if isinstance(v, dict) and 'name' in v}
    lname = {v['name']: v.get('userFriendlyName') for v in p.values() if isinstance(v, dict) and str(v.get('typ', '')).startswith('Line')}
    out = _post(client, _state(design, state, 'ShortCircuitPandaPower Parameters',
                               {**SC, 'fault_type': '3ph', 'fault_location': 'max', 'fault_bus_mode': 'selected',
                                'fault_bus_ids': [cell[fault_bus]], 'fault_bus_names': [fault_bus]}))
    return {lname[l['name']]: 1000 * l['ikss_ka'] for l in out['lines_sc']}


def _feeder_load_a(client, design, state, load_scale=None):
    p = _payload(design)
    lname = {v['name']: v.get('userFriendlyName') for v in p.values() if isinstance(v, dict) and str(v.get('typ', '')).startswith('Line')}
    out = _post(client, _state(design, state, load_scale=load_scale))
    return {lname[l['name']]: 1000 * max(l['i_from_ka'] or 0, l['i_to_ka'] or 0) for l in out['lines']}


def test_revc_sofc_feeders_cannot_be_protected_by_overcurrent(client):
    """
    Rev C: a SOFC feeder carries its board's groups, whose fault current is their 1.2 pu current limit. A fault on MVA
    puts 431 A through F-S1A; the same kind of feeder carries 346 A of load when MVA is lost and the halls move to cord
    B. Pickup at 1.25 x load would be 433 A - above the fault: no overcurrent setting separates fault from load. The
    feeders need their 87L (as rev C has it), not 50/51.
    """
    fault = _feeder_ka(client, 'revc', 'NORMAL', 'MVA')
    load = _feeder_load_a(client, 'revc', 'MVA_BUS_FAULT', load_scale={'DH3_cordB': 2.0, 'DH4_cordB': 2.0})
    assert fault['F_S1A'] == pytest.approx(431, abs=5) and load['F_S2B'] == pytest.approx(346, abs=5)
    assert fault['F_S1A'] / max(load.values()) < 1.3


def test_revc_bess_feeders_need_direction(client):
    """
    Rev C's BESS feeders: 306 A for a fault on MVA against 89 A of load at most - room for a 110 A pickup (2.8 x) -
    but 105 A flows through them for a fault on a SOFC board: a non-directional pickup that low trips on a remote
    fault. 67 elements (or 87L) make them selective.
    """
    near = _feeder_ka(client, 'revc', 'NORMAL', 'MVA')
    remote = _feeder_ka(client, 'revc', 'NORMAL', 'SOFC1A')
    load = _feeder_load_a(client, 'revc', 'MVA_BUS_FAULT', load_scale={'DH3_cordB': 2.0, 'DH4_cordB': 2.0})
    assert near['F_E1A'] == pytest.approx(306, abs=5) and load['F_E1B'] == pytest.approx(89, abs=2)
    assert near['F_E1A'] / (1.25 * load['F_E1B']) > 2.5
    assert remote['F_E1A'] > 0.9 * 1.25 * load['F_E1B']


def test_ring_carries_through_fault_both_ways(client):
    """
    The ring, tie closed: a fault on MVA draws 943 A through LN-A-MVA and 554 A through the tie; a fault on MVB, 569 A
    through the tie the other way. Overcurrent on the ring circuits is not selective without direction; the note's
    87 unit protection and 67N are what make it so.
    """
    a = _feeder_ka(client, 'ring', 'NORMAL', 'MVA')
    b = _feeder_ka(client, 'ring', 'NORMAL', 'MVB')
    assert a['LN_A_MVA'] == pytest.approx(943, abs=10) and a['TIE_AB'] == pytest.approx(554, abs=10)
    assert b['TIE_AB'] == pytest.approx(569, abs=10)


# --- Phase 5: ramp bridging -------------------------------------------------------------------------------------
GAP = 15.58          # the bridging requirement: the workload ramp (1.64 MW/s) against the SOFC fleet's (0.396 MW/s)


def _bridging(client, design, state, inlet_c, sofc_pf=None):
    """
    The bridging peak as a load-flow snapshot: the halls at full load, the SOFC groups lagging by GAP, the PCS at
    the inlet temperature carrying the gap and (unless the SOFC inverters run at a power factor) the halls' Q.
    """
    p = _state(design, state)
    site = _site(design)
    on = next(s for s in site['islands'] if s['id'] == state)['energized_source_ids']
    n_on = len([s for s in on if s.startswith('SOFC')])
    load = sum(float(v['p_mw']) for v in p.values() if isinstance(v, dict) and str(v.get('typ', '')).startswith('Load'))
    for v in p.values():
        if not isinstance(v, dict):
            continue
        name = str(v.get('userFriendlyName', ''))
        if 'PCST5500' in name:
            v['inlet_c'] = str(inlet_c)
        if name.endswith(' SOFC'):
            v['p_set_kw'] = str(round((load - GAP) / n_on * 1e3, 3))
        if name.endswith(' inverter') and sofc_pf:
            v['q_mode'], v['pf'] = 'pf', str(sofc_pf)
    out = _post(client, p)
    gfm = [x for x in out['pcs'] if x['source_kind'] == 'Battery' and x['in_service']]
    return out, gfm


def test_ring_bridging_at_45c_with_a_pcs_out_is_reactive_power_bound(client):
    """
    The ring, one PCS out, 45 C (5.0 MVA each): at the bridging peak each of the four carries 3.92 MW of the gap and,
    with the SOFC inverters at unity power factor, 3.02 Mvar of the halls' reactive power - 102 % of its rating at its
    bus's 0.97 pu, where the owner's +4.42 MW counts active power only. With the SOFC inverters at 0.96 power factor
    (their 2.2 MVA has the room) the halls' Q moves to them: 83.7 %, and the switchboards at 0.97 pu.
    """
    out, gfm = _bridging(client, 'ring', 'N1_PCS', 45)
    assert len(gfm) == 4 and all(x['p_mw'] == pytest.approx(3.92, abs=0.02) for x in gfm)
    assert all(x['loading_percent'] > 98 for x in gfm)
    assert any('loaded to 10' in w for w in out.get('warnings') or [])
    out, gfm = _bridging(client, 'ring', 'N1_PCS', 45, sofc_pf=0.96)
    assert all(x['loading_percent'] == pytest.approx(83.7, abs=1.0) for x in gfm)
    assert not out.get('warnings')


def test_revc_bridging_at_45c_with_a_pcs_out(client):
    """Rev C, one of six PCS out, 45 C: the five carry the gap at 78 % - the sixth PCS is its margin."""
    _, gfm = _bridging(client, 'revc', 'N1_PCS', 45)
    assert len(gfm) == 5 and all(x['loading_percent'] == pytest.approx(78.4, abs=1.0) for x in gfm)


def test_bridging_frequency_is_droop_bound(client):
    """
    The PCS's 4 % droop holds the island's frequency at f0 (1 - droop P / S): at the bridging peak 48.43 Hz with four
    PCS, 48.75 Hz with five - below the 49 Hz an under-frequency stage commonly starts at. The EMS's secondary control
    (design note s2) must act within the 12.5 s ramp, not after it; ANDES gives the same path, smooth and monotone.
    """
    _, four = _bridging(client, 'ring', 'N1_PCS', 45, sofc_pf=0.96)
    _, five = _bridging(client, 'ring', 'NORMAL', 45, sofc_pf=0.96)
    assert four[0]['frequency_hz'] == pytest.approx(48.43, abs=0.03)
    assert five[0]['frequency_hz'] == pytest.approx(48.75, abs=0.03)


# --- Phase 6: EMT - earth faults, a board fault, and the recovery after it -------------------------------------
def _emt_with(client, design, pcs_limit=None, **params):
    p = _payload(design)
    cell = {v.get('userFriendlyName'): v['name'] for v in p.values() if isinstance(v, dict) and 'name' in v}
    if pcs_limit is not None:
        for v in p.values():
            if isinstance(v, dict) and 'PCST5500' in str(v.get('userFriendlyName', '')):
                v['current_limit_pu'] = str(pcs_limit)
    p['0'] = {'typ': 'EmtStudy Parameters', 'frequency': '50', 'time_step_us': '20', 'user_email': 't@t',
              'duration_ms': '120', 'ac_fault_time_ms': '20', 'ac_fault_duration_ms': '80',
              **{k: (cell[v] if k == 'ac_fault_bus' else v) for k, v in params.items()}}
    emt = _post(client, p)['emt']
    return emt, {b['label']: b for b in emt['ac']['buses']}, {c['label']: c for c in emt['converters']}


@pytest.mark.parametrize('design, i_ka, i_hand', [('ring', 0.835, 1.003), ('revc', 0.892, 1.003)])
def test_emt_bolted_earth_fault_ridden_through(client, design, i_ka, i_hand):
    """
    EMT, phase a solidly to earth on MVA: every converter rides through, the earth fault near the NERs' V_ph / R_N -
    0.835 kA on the ring (two 38 ohm: 1.0 kA by hand), 0.892 kA on Rev C (four 76 ohm: 1.0 kA), the rest in the
    converters' and zig-zags' own impedance. With the batteries at 1,000 V every PCS tripped on overcurrent here: a
    690 V inverter needs a DC link of ~1.13 kV and more to make its voltage in an unbalanced fault - the PCST5500's
    DC range is a vendor question.
    """
    emt, _, _ = _emt_with(client, design, ac_fault_bus='MVA', ac_fault_type='ag', ac_fault_resistance_ohm='0.001')
    assert not emt['converters_blocked']
    i = emt['ac']['fault']['phases'][0]['i_sym_ka']
    assert i == pytest.approx(i_ka, abs=0.03) and i < i_hand


def test_emt_board_fault_collapses_the_whole_island(client):
    """
    Rev C, a bolted three-phase fault on SOFC board section 1A for 100 ms (87B's clearing time): every 33 kV board
    goes to zero, not only 1A - a converter-limited island cannot hold any voltage through a bolted fault. Every
    SOFC inverter on the site, not only the five to be transferred, must ride through the clearing time (A16).
    """
    emt, bus, _ = _emt_with(client, 'revc', ac_fault_bus='SOFC1A', ac_fault_type='abcg', ac_fault_resistance_ohm='0.001',
                            ac_fault_duration_ms='100', duration_ms='200')
    assert all(bus[b]['v_rms_min_pu'] < 0.01 for b in ('SOFC1A', 'SOFC2B', 'MVA', 'MVB', 'BESS1', 'BESS2'))


@pytest.mark.parametrize('limit, recovers', [(1.2, False), (1.3, True)])
def test_emt_recovery_needs_pcs_short_time_current(client, limit, recovers):
    """
    After that board fault clears, the island recovers only if the PCS can give some 1.3 pu: at 1.2 pu they stay in
    current limit (the boards at 0.935 pu, 3 % unbalance, 400 ms after), at 1.3 pu they leave it with the fault and
    the boards return to 0.96 pu balanced. The workbook's 1.1 pu (ASSUMED) latches too. Model-dependent (Electrisim's
    averaged grid-forming PCS, generic grid-following SOFC inverters): the PCST5500's short-time current and its
    recovery from current limit are vendor questions.
    """
    emt, bus, conv = _emt_with(client, 'revc', pcs_limit=limit, ac_fault_bus='SOFC1A', ac_fault_type='abcg',
                               ac_fault_resistance_ohm='0.001', ac_fault_duration_ms='100', duration_ms='500')
    assert not emt['converters_blocked']
    pcs = conv['PCS1 PCST5500']
    if recovers:
        assert pcs['limited_ms'] < 130 and bus['MVA']['vuf_end_percent'] < 0.1 and bus['MVA']['v_rms_final_pu'] > 0.95
    else:
        assert pcs['limited_ms'] > 400 and bus['MVA']['vuf_end_percent'] > 1.0


def test_revc_feeder_relays_see_their_own_current(client):
    """
    Rev C draws its feeder breakers as bus-bus switches (board to feeder bus). The protection study gave a relay on
    such a breaker the fault's total current - 1.60 kA on every feeder for a fault on MVA, and 16 miscoordinations that
    were artefacts. Each relay now sees its own feeder's current, the short-circuit study's cable current: 431 A on
    F-S1A, 306 A on F-E1A.
    """
    p = _state('revc', 'NORMAL', 'ProtectionCoordinationPandaPower')
    cell = {v.get('userFriendlyName'): v['name'] for v in p.values() if isinstance(v, dict) and 'name' in v}
    for v in p.values():
        if isinstance(v, dict) and str(v.get('userFriendlyName', '')).startswith('CB_F_'):
            v.update({'protection_type': 'ocr', 'oc_relay_type': 'DTOC', 'pickup_mode': 'manual', 'I_s_a': '440',
                      'I_g_a': '440', 't_g': '0.4', 'I_gg_a': '0'})
    p['0'] = {'typ': 'ProtectionCoordinationPandaPower', 'user_email': 't@t', 'fault_type': '3ph', 'case': 'max',
              'fault_location_mode': 'bus', 'fault_bus_id': cell['MVA'], 'sc_fraction': 0.5, 'grading_mode': 'manual',
              't_diff': 0.3, 'frequency': '50'}
    out = _post(client, p)
    trip = {t['user_friendly_name']: 1000 * t['ikss_ka'] for t in out['scenarios'][0]['trip']}
    cable = _feeder_ka(client, 'revc', 'NORMAL', 'MVA')
    total = 1000 * out['scenarios'][0]['short_circuit']['ikss_ka']
    assert trip['CB_F_S1A'] == pytest.approx(cable['F_S1A'], rel=0.01) and trip['CB_F_E1A'] == pytest.approx(cable['F_E1A'], rel=0.01)
    assert max(trip.values()) < 0.5 * total


def _blackstart_gen_corrected(design='revc', typ='PowerFlowPandaPower Parameters', extra=None):
    """Rev C's BLACKSTART_GEN with the cord-A hall feeders open too - as its later black-start states have them."""
    p = _state(design, 'BLACKSTART_GEN', typ, extra)
    for v in p.values():
        if isinstance(v, dict) and str(v.get('typ', '')).startswith('Switch') \
                and v.get('userFriendlyName') in ('CB_L_DH1_cordA', 'CB_L_DH2_cordA'):
            v['closed'] = 'false'
    return p


def test_revc_genset_black_start_corrected_carries_aux_a(client):
    """
    BLACKSTART_GEN corrected - the cord-A hall feeders open, as the later black-start states have them: the genset
    carries Aux A alone, 2.0 MW at 67 % of its 3 MVA, the board at 1.0 pu, TR_GEN within its rating.
    """
    lf = _post(client, _blackstart_gen_corrected())
    (gen,) = [g for g in lf['generators'] if g['p_mw']]
    assert gen['p_mw'] == pytest.approx(2.0, abs=0.01) and sum(l['p_mw'] for l in lf['loads']) == pytest.approx(2.0)
    assert min(b['vm_pu'] for b in lf['busbars'] if b['vm_pu'] and b['vm_pu'] > 0) > 0.99
    assert not any('TR_GEN' in w for w in lf.get('warnings') or [])


def test_revc_genset_picks_up_aux_a(client):
    """
    ANDES, the corrected state: Aux A's 2.0 MW switched onto the genset in one step at 1 s - the frequency's nadir
    48.0 Hz at 2.4 s, then its 5 % droop's 48.37 Hz (by hand 48.33 Hz, 50 (1 - 0.05 x 2.0 / 3.0)) until something restores it; the
    loads at constant power. The rest of the site has no source in this state: ANDES failed on it ("list index out of
    range"); its buses are set aside and reported at 0 V. The voltage swings slowly (0.76-1.08 pu) on the generic
    exciter data - the genset's AVR is a vendor question.
    """
    import numpy as np
    dt, tf = 0.05, 8.0
    p = _blackstart_gen_corrected('revc', 'TransientStabilityAndes Parameters', {
        'sn_mva': '100', 'tf': str(tf), 'tstep': '0', 'fault_enabled': 'false', 'load_model': 'power',
        'load_profiles': {'pick': {'name': 'Aux A pick-up', 'kind': 'power', 'dt': dt, 't0': 0,
                                   'p': [0.02 if k * dt < 1.0 else 1.0 for k in range(int(tf / dt) + 2)]}}})
    for v in p.values():
        if isinstance(v, dict) and str(v.get('typ', '')).startswith('Load') and str(v.get('userFriendlyName', '')).startswith('AUX_A'):
            v['load_profile_id'], v['load_profile_q_mode'] = 'pick', 'pf'
    out = _post(client, p)
    assert any(w.startswith('No source reaches 70 bus(es)') for w in out['warnings'])
    t, f = np.array(out['time']), out['frequency_hz']
    f = np.array(f[0]['values'] if isinstance(f, list) and isinstance(f[0], dict) else f)
    assert f.min() == pytest.approx(48.0, abs=0.05) and f[-1] == pytest.approx(48.37, abs=0.02)
    v = {b['name']: b['values'] for b in out['bus_voltage']}
    assert max(v['MVB']) == 0.0 and min(v['MVA']) > 0.7


@pytest.mark.parametrize('voltage_control, v_end, f_end', [(True, 0.974, 48.61), (False, 0.939, 48.85)])
def test_andes_bridging_ramp_with_the_pcs_voltage_control(client, voltage_control, v_end, f_end):
    """
    ANDES through the ring's bridging ramp (one PCS out, 45 C, SOFC at 0.96 pf), the loads constant impedance: the
    workload ramps 1.244 MW/s net of the SOFC for 12.5 s. A grid-forming PCS in ANDES is a virtual machine (GENCLS)
    whose internal voltage was held through a run: its terminal sagged as it loaded up - RING_A at 0.939 pu, the
    loads shrinking with it, 48.85 Hz. With its voltage control (Electrisim's, every 20 ms, its Q-V droop through its
    operating point) RING_A holds 0.974 pu and the frequency is the droop's, 48.61 Hz - what ANDES's WECC REGF1 gives.
    """
    import numpy as np
    gap, rate, dt, tf = 15.58, 1.64 - 0.396, 0.25, 16.0
    p = _state('ring', 'N1_PCS', 'TransientStabilityAndes Parameters', {
        'sn_mva': '100', 'tf': str(tf), 'tstep': '0', 'fault_enabled': 'false', 'load_model': 'impedance',
        'pcs_voltage_control': 'true' if voltage_control else 'false'})
    load = sum(float(v['p_mw']) for v in p.values() if isinstance(v, dict) and str(v.get('typ', '')).startswith('Load'))
    n_on = len([x for x in next(st for st in _site('ring')['islands'] if st['id'] == 'N1_PCS')['energized_source_ids']
                if x.startswith('SOFC')])
    f0 = (load - gap) / load
    values = [min(f0 + (1 - f0) * (k * dt) / (gap / rate), 1.0) for k in range(int(tf / dt) + 2)]
    p['0']['load_profiles'] = {'ramp': {'name': 'ramp', 'kind': 'power', 'dt': dt, 't0': 0, 'p': values}}
    for v in p.values():
        if not isinstance(v, dict):
            continue
        name, t = str(v.get('userFriendlyName', '')), str(v.get('typ', ''))
        if 'PCST5500' in name:
            v['inlet_c'] = '45'
        if name.endswith(' SOFC'):
            v['p_set_kw'] = str(round((load - gap) / n_on * 1e3, 3))
        if name.endswith(' inverter'):
            v['q_mode'], v['pf'] = 'pf', '0.96'
        if t.startswith('Load') and not t.startswith('Load DC'):
            v['load_profile_id'], v['load_profile_q_mode'] = 'ramp', 'pf'
    out = _post(client, p)
    ring_a = next(b['values'] for b in out['bus_voltage'] if b['name'] == 'RING_A')
    gfm = [x for x in out['pcs'] if x['control'] == 'grid_forming']
    assert out['time'][-1] == pytest.approx(tf)
    assert ring_a[-1] == pytest.approx(v_end, abs=0.004)
    assert np.asarray(gfm[0]['frequency_hz'])[-1] == pytest.approx(f_end, abs=0.03)
