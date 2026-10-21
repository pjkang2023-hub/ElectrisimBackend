"""
C003 off-grid microgrid on the owner's data of 20 October 2026: IT 25 MW at PF 0.95 with PUE 1.2 - 30.0 MW and
9.86 Mvar, 31.58 MVA - nineteen SOFC groups (18 + 1) at 40 000 h, PCST5500 grid-forming storage, 33 kV, 50 Hz.

Two designs from the owner's mgtwin Sites (specs/make_c003_spec.py --p-group 1.936):
  ring10 - recommended design v1.4, Site v10 (design report v2.0): closed ring, five PCS (three on half A, two on
           half B, on the ring itself), two zig-zags with 38 ohm NERs, load breakers, black-start steps BS1-BS5;
  revc2  - Rev C.2, Site revC2: six-board dual-radial, six PCS (2 + 2 + 1 + 1), four zig-zags with 76 ohm NERs.
Each SOFC group gives 1.936 MW at its inverter at 40 000 h - 0.44 x 5 x (1 - 0.003 x 40); the owner's 1.907 MW is
after its 98.5 % transformer - and is dispatched at the load plus 1 % over nineteen groups (1.5947 MW each).

Site v10 draws IT cord A behind TR_IT_A, one 3.5 MVA unit of the cord's bank, with x 0.08 on its own rating (every
other transformer's r/x are on the 50 MVA base): drawn as a bank of five (4 + 1, 17.5 MVA, ASSUMED) at 8 %.
"""
import math

import pytest

from test_reference_c003 import _payload, _post, _spec

GROUPS = 19
P_GROUP = 30.0 * 1.01 / GROUPS          # 1.5947 MW
LOAD_MW, LOAD_MVAR = 30.0, 9.86


@pytest.mark.parametrize('design, n_pcs', [('ring10', 5), ('revc2', 6)])
def test_spec_solves_off_grid(quiet, design, n_pcs):
    """
    Built from its spec: no external grid, the PCS form the island at 50 Hz; the nineteen groups carry the load at
    1.5947 MW each and the PCS share the rest equally - absorbing the 1 % dispatched for losses less the network's.
    """
    import electrisim_sld as sld
    with quiet():
        net, _ = sld.build_network(_spec(design))
        res = sld.solve(net)
    assert res['converged'], res.get('hint')
    gfm = [p for p in res['pcs'] if not str(p['id']).startswith('SOFC')]
    gfl = [p for p in res['pcs'] if str(p['id']).startswith('SOFC')]
    assert len(gfm) == n_pcs and len(gfl) == GROUPS
    assert sum(p['p_mw'] for p in gfl) == pytest.approx(GROUPS * P_GROUP, abs=2e-3)
    p = [g['p_mw'] for g in gfm]
    assert max(p) - min(p) < 1e-3 and all(x < 0 for x in p)
    assert 0 < GROUPS * P_GROUP - LOAD_MW + sum(p) < 0.2          # the ring 0.15 MW: its IT bank's losses
    assert all(abs(g['frequency_hz'] - 50.0) < 0.05 for g in gfm)


@pytest.mark.parametrize('design, n_pcs, v_min, q_pcs', [('ring10', 5, 0.9227, 2.46), ('revc2', 6, 0.9620, 1.869)])
def test_drawn_load_flow(client, design, n_pcs, v_min, q_pcs):
    """
    The drawn network as the app sends it: 30.0 MW and 9.86 Mvar served without a warning, each SOFC inverter at
    1.5947 MW (72.5 % of its 2.2 MVA), the PCS sharing the reactive power equally. The ring's lowest bus is IT cord
    A's 400 V behind its bank (0.923 pu: 8 % at 0.95 power factor - the bank needs its taps); Rev C.2's 0.962 pu.
    """
    out = _post(client, _payload(design))
    assert not out.get('warnings'), out.get('warnings')
    assert sum(ld['p_mw'] for ld in out['loads']) == pytest.approx(LOAD_MW, abs=1e-6)
    assert sum(ld['q_mvar'] for ld in out['loads']) == pytest.approx(LOAD_MVAR, abs=0.01)
    inv = [p for p in out['pcs'] if p['source_kind'] == 'SOFC']
    gfm = [p for p in out['pcs'] if p['source_kind'] == 'Battery']
    assert len(inv) == GROUPS and len(gfm) == n_pcs
    assert all(p['p_mw'] == pytest.approx(P_GROUP, abs=1e-4) and p['loading_percent'] == pytest.approx(72.49, abs=0.05)
               for p in inv)
    assert all(g['q_mvar'] == pytest.approx(q_pcs, abs=0.01) for g in gfm)
    assert min(b['vm_pu'] for b in out['busbars'] if b['vm_pu']) == pytest.approx(v_min, abs=5e-4)


@pytest.mark.parametrize('design, n_gt, r_n, i_hand', [('ring10', 2, 38, 501.4), ('revc2', 4, 76, 250.7)])
def test_grounding_unchanged(client, design, n_gt, r_n, i_hand):
    """The grounding is as before: 501.4 A per 38 ohm NER on the ring, 250.7 A per 76 ohm on Rev C.2 (a little less
    through the zig-zags' own impedance)."""
    assert 33e3 / math.sqrt(3) / r_n == pytest.approx(i_hand, abs=0.1)
    gts = _post(client, _payload(design))['groundingtransformers']
    assert len(gts) == n_gt and all(g['r_n_ohm'] == r_n for g in gts)
    assert all(0.99 * i_hand < g['i_ground_alone_a'] < i_hand for g in gts)


# --- Phase 2: margins, states and the owner's new operating points -----------------------------------------------
from test_reference_c003 import OPF, _margin, _state      # noqa: E402

IDLE = {'IT_cordA': 11.05 / 30, 'IT_cordB': 11.05 / 30, 'OTHER_A': 11.05 / 30, 'OTHER_B': 11.05 / 30}


def _edit(p, open_sw=(), sofc_kw=None, pcs_inlet=None, bank_mva=None):
    for v in p.values():
        if not isinstance(v, dict):
            continue
        n, t = str(v.get('userFriendlyName', '')), str(v.get('typ', ''))
        if t.startswith('Switch') and n in open_sw:
            v['closed'] = 'false'
        if sofc_kw is not None and n.endswith(' SOFC'):
            v['p_set_kw'] = str(sofc_kw)
        if pcs_inlet is not None and 'PCST5500' in n:
            v['inlet_c'] = str(pcs_inlet)
        if bank_mva is not None and n.startswith('TR_IT_A'):
            v['sn_mva'] = str(bank_mva)
    return p


@pytest.mark.parametrize('design, n, n1, n2', [('ring10', 6.64, 4.705, 2.768), ('revc2', 6.711, 4.786, 2.849)])
def test_sofc_margins_at_40000_h(client, design, n, n1, n2):
    """
    Nineteen groups at 1.936 MW against 30.0 MW and the network's losses: +6.6 MW, N-1 +4.7, N-2 +2.8 - nothing shed,
    no battery. The owner's N-1 +2.27 MW reads 31.58 MVA as MW and adds 1.5 % losses; on real power, as here, the
    nineteenth group is spare twice over (18 groups: N-1 +2.8 MW).
    """
    for state, also_off, want in (('NORMAL', (), n), ('N1_SOFC', (), n1), ('N1_SOFC', ('SOFC11',), n2)):
        out = _post(client, _state(design, state, 'AcDcOpfPandaPower Parameters', OPF, also_off=also_off))
        assert out['solved'] and out['shed_mw'] == 0
        margin, battery = _margin(out)
        assert margin == pytest.approx(want, abs=0.03) and battery < 0.01


def test_ring_split_holds_on_either_half(client):
    """The ring split with a group out of either half: nothing shed and no battery - half B's -0.5 MW in the
    workbook is the MVA-as-MW reading; on real power its eight groups carry its 15 MW."""
    for g in ('SOFC01', 'SOFC11'):
        out = _post(client, _state('ring10', 'RING_SPLIT', 'AcDcOpfPandaPower Parameters', OPF, also_off=(g,)))
        assert out['solved'] and out['shed_mw'] == 0 and _margin(out)[1] < 0.01


@pytest.mark.parametrize('units, loading, shed', [(5, 171.6, 7.39), (8, 103.2, 0.0)])
def test_ring_cord_bank_must_carry_the_whole_it_load(client, units, loading, shed):
    """
    Board MVB lost, the halls dual-corded: all 25 MW of IT moves to cord A, through its 400 V bank. Drawn as five
    3.5 MVA units (17.5 MVA) it is loaded to 172 %, its bus at 0.88 pu, and 7.4 MW must be shed; eight (28 MVA) are
    still 103 %. Each cord's bank must carry the whole IT load, 26.3 MVA - nine units, not the 12.5 MW of one cord.
    """
    lf = _post(client, _edit(_state('ring10', 'BUS_MVB_FAULT', load_scale={'IT_cordA': 2.0}), bank_mva=3.5 * units))
    (tr,) = [w for w in lf['warnings'] if "'TR_IT_A' is loaded" in w]
    assert float(tr.split('loaded to ')[1].split(' %')[0]) == pytest.approx(loading, abs=0.5)
    out = _post(client, _edit(_state('ring10', 'BUS_MVB_FAULT', 'AcDcOpfPandaPower Parameters', OPF,
                                     load_scale={'IT_cordA': 2.0}), bank_mva=3.5 * units))
    assert out['solved'] and out['shed_mw'] == pytest.approx(shed, abs=0.05)


@pytest.mark.parametrize('design, extra_open, gen_mw', [('ring10', ('CB_L_IT_cordA', 'CB_L_IT_cordB', 'CB_L_OTHER_B'), 2.55),
                                                       ('revc2', ('CB_L_IT_cordA',), 2.50)])
def test_genset_black_start_as_declared_and_corrected(client, design, extra_open, gen_mw):
    """
    BLACKSTART_GEN as both Sites declare it leaves IT feeders closed: the ring's opens only CB-B-MVB, and with its
    tie closed MVB stays on MVA - its load flow does not solve (17.5 MW with cord A opened: TR_GEN 894 %); Rev C.2's
    puts 15 MW on the 3 MVA genset (TR_GEN 657 %, 0.80 pu) - as in Rev C.1. With the IT feeders open (the ring's
    BS1 set, Rev C.2's cord A) the genset carries Other-A alone: 2.5 MW and the ring's charging, within its rating.
    """
    lf = _post(client, _edit(_state(design, 'BLACKSTART_GEN'), open_sw=extra_open))
    (gen,) = [g for g in lf['generators'] if g['p_mw']]
    assert gen['p_mw'] == pytest.approx(gen_mw, abs=0.02) and math.hypot(gen['p_mw'], gen['q_mvar']) < 3.0
    assert not any('TR_GEN' in w for w in lf.get('warnings') or [])
    if design == 'revc2':
        lf = _post(client, _state(design, 'BLACKSTART_GEN'))
        assert any("'TR_GEN' is loaded to 65" in w for w in lf['warnings'])


@pytest.mark.parametrize('design', ['ring10', 'revc2'])
def test_idle_below_the_sofc_minimum_load(client, design):
    """
    Idle, 11.05 MW: nineteen groups dispatched to it run at 587 kW, 26.7 % - under the 30 % minimum load the study
    assumes (A20) - and Electrisim says so for each. Sixteen groups would hold 31.7 %.
    """
    lf = _post(client, _edit(_state(design, 'NORMAL', load_scale=IDLE), sofc_kw=round(11.05 * 1.01 / 19 * 1e3, 1)))
    low = [w for w in lf['warnings'] if 'below its minimum load' in w]
    assert len(low) == GROUPS and all('587.4 kW' in w for w in low)
    assert 11.05 * 1.01 / 16 / 2.2 > 0.30


@pytest.mark.parametrize('design, n_pcs, each, loading', [('ring10', 4, 3.117, 67.1), ('revc2', 5, 2.496, 53.4)])
def test_ramp_down_charge_peak_at_45c_with_a_pcs_out(client, design, n_pcs, each, loading):
    """
    The workload falls 0.82 MW/s, the SOFC 0.319 MW/s: at the end of the ramp-down the PCS absorb 12.55 MW. At 45 C
    with one PCS out they take it within rating - the ring's four at 67 %, Rev C.2's five at 53 % - each battery at
    about 4-5 C.
    """
    lf = _post(client, _edit(_state(design, 'N1_PCS', load_scale=IDLE), sofc_kw=round((11.05 + 12.55) / 19 * 1e3, 1),
                             pcs_inlet=45))
    gfm = [x for x in lf['pcs'] if x['source_kind'] == 'Battery' and x['in_service']]
    assert len(gfm) == n_pcs and all(x['p_mw'] == pytest.approx(-each, abs=0.01) for x in gfm)
    assert all(x['loading_percent'] == pytest.approx(loading, abs=0.5) for x in gfm) and not lf.get('warnings')


def test_revc2_section_fault_and_bess_board_fault(client):
    """
    Rev C.2: section 1A isolated before the transfer - fourteen groups give 27.1 MW, the PCS carry 2.95 MW from
    their batteries (0.15 MWh over a 3 min slow transfer). BESS board 1 lost and PCS3 out at 45 C: the three left
    carry the halls' reactive power at 77 % each.
    """
    out = _post(client, _state('revc2', 'SOFC1A_SECTION_FAULT_PRE', 'AcDcOpfPandaPower Parameters', OPF))
    assert out['solved'] and out['shed_mw'] == 0 and _margin(out)[1] == pytest.approx(2.95, abs=0.03)
    lf = _post(client, _edit(_state('revc2', 'BESS1_BUS_FAULT', also_off=('PCS3',)), pcs_inlet=45))
    gfm = [x for x in lf['pcs'] if x['source_kind'] == 'Battery' and x['in_service']]
    assert len(gfm) == 3 and all(x['loading_percent'] == pytest.approx(77.4, abs=0.5) for x in gfm)


# --- Phase 3: fault levels ------------------------------------------------------------------------------------------
from test_reference_c003 import SC, _bridging, _bus_ka, _emt, _feeder_ka, _feeder_load_a, _ka      # noqa: E402


@pytest.mark.parametrize('design, n_pcs, with_gen', [('ring10', 5, 1.692), ('revc2', 6, 1.798)])
def test_iec_fault_levels(client, design, n_pcs, with_gen):
    """
    IEC 60909 maximum, three-phase: nineteen inverters x 2.2 MVA x 1.2 and the PCS x 5.5 MVA x 1.1 - 1.405 kA on the
    ring, 1.511 kA on Rev C.2, by hand to the ampere (two groups fewer than Rev C.1: -0.09 kA); with the genset 1.69
    and 1.80 kA - the design report's 'about 1 kA' is the converters at 1.1 pu alone, 0.53 kA, plus the genset.
    """
    p = _state(design, 'NORMAL', 'ShortCircuitPandaPower Parameters', {**SC, 'fault_type': '3ph', 'fault_location': 'max'})
    assert _bus_ka(design, _post(client, p))['MVA'] == pytest.approx(_ka(GROUPS * 2.2 * 1.2 + n_pcs * 5.5 * 1.1), abs=2e-3)
    for v in p.values():
        if isinstance(v, dict) and str(v.get('userFriendlyName', '')).startswith('GEN_BS'):
            v['in_service'] = 'true'
    assert _bus_ka(design, _post(client, p))['MVA'] == pytest.approx(with_gen, abs=5e-3)


@pytest.mark.parametrize('design, state, bus, i_ka', [('ring10', 'BS2_PCS_SOFT_ENERGISE', 'MVA', 0.391),
                                                     ('revc2', 'BLACKSTART_BESS1', 'BESS1', 0.210)])
def test_black_start_fault_levels(client, design, state, bus, i_ka):
    """The ring's black start from the genset with PCS1 synchronised (BS2): 0.39 kA; Rev C.2's from PCS 1-2 alone:
    0.21 kA - a 4-7 : 1 range under the normal maximum that phase overcurrent alone cannot span."""
    out = _post(client, _state(design, state, 'ShortCircuitPandaPower Parameters', {**SC, 'fault_type': '3ph', 'fault_location': 'max'}))
    assert _bus_ka(design, out)[bus] == pytest.approx(i_ka, abs=3e-3)


@pytest.mark.parametrize('design, i3, iag0', [('ring10', 1.025, 0.826), ('revc2', 1.063, 0.897)])
def test_emt_faults_ridden_through(client, design, i3, iag0):
    """
    EMT on MVA, 80 ms: three-phase bolted 1.03 kA (ring) and 1.06 kA (Rev C.2) in phase a; phase a to earth through
    50 ohm 0.15 kA; bolted to earth 0.83 / 0.90 kA, the NERs' 1.0 kA less the converters' and zig-zags' impedance.
    Nothing blocks - the batteries at 1,300 V (Q1).
    """
    for params, want in ((dict(ac_fault_type='abcg', ac_fault_resistance_ohm='0.001'), i3),
                         (dict(ac_fault_type='ag', ac_fault_resistance_ohm='50'), 0.152),
                         (dict(ac_fault_type='ag', ac_fault_resistance_ohm='0.001'), iag0)):
        emt = _emt(client, design, ac_fault_bus='MVA', **params)
        assert not emt['converters_blocked']
        assert emt['ac']['fault']['phases'][0]['i_sym_ka'] == pytest.approx(want, abs=0.02)


# --- Phase 4: protection setting windows ----------------------------------------------------------------------------
def test_revc2_feeder_windows(client):
    """
    Rev C.2: a SOFC feeder carries 420 A into a fault on MVA (F-S1A) and up to 291 A of load when MVA is lost and
    the halls move to cord B (F-S1B) - a pickup at 1.25 x load (364 A) sits 13 % under the fault: overcurrent is
    still no protection for these feeders (87L). The BESS feeders: 294 A against 77 A of load, but 105 A for a
    fault on a SOFC board - direction (67) still needed.
    """
    fault = _feeder_ka(client, 'revc2', 'NORMAL', 'MVA')
    remote = _feeder_ka(client, 'revc2', 'NORMAL', 'SOFC1A')
    load = _feeder_load_a(client, 'revc2', 'MVA_BUS_FAULT', load_scale={'IT_cordB': 2.0})
    assert fault['F_S1A'] == pytest.approx(420, abs=5) and load['F_S1B'] == pytest.approx(291, abs=5)
    assert fault['F_S1A'] / (1.25 * load['F_S1B']) < 1.2
    assert fault['F_E1A'] == pytest.approx(294, abs=5) and load['F_E1B'] == pytest.approx(77, abs=2)
    assert remote['F_E1A'] > 1.25 * load['F_E1B']


def test_ring_feeders_through_fault_both_ways(client):
    """The ring: 897 A through LN-A-MVA for a fault on MVA, 574 A of load when MVB is lost (ratio 1.56); the tie
    508 A one way, 546 A the other - unit protection and direction on the ring circuits, as designed."""
    a = _feeder_ka(client, 'ring10', 'NORMAL', 'MVA')
    b = _feeder_ka(client, 'ring10', 'NORMAL', 'MVB')
    load = _feeder_load_a(client, 'ring10', 'BUS_MVB_FAULT', load_scale={'IT_cordA': 2.0})
    assert a['LN_A_MVA'] == pytest.approx(897, abs=10) and load['LN_A_MVA'] == pytest.approx(574, abs=10)
    assert a['TIE_AB'] == pytest.approx(508, abs=10) and b['TIE_AB'] == pytest.approx(546, abs=10)


# --- Phase 5: the bridging peak at 45 C -----------------------------------------------------------------------------
@pytest.mark.parametrize('design, state, pf, loading, f_hz', [
    ('ring10', 'N1_PCS', None, 100.0, 48.43), ('ring10', 'N1_PCS', 0.96, 87.8, 48.43),
    ('revc2', 'N1_PCS', None, 76.5, 48.75), ('revc2', 'BESS1_BUS_FAULT', None, 96.3, 48.44)])
def test_bridging_peak_at_45c(client, design, state, pf, loading, f_hz):
    """
    15.58 MW of bridging at 45 C (5.0 MVA each). The ring with one PCS out: four carry 3.94 MW and, the SOFC at unity
    power factor, 3.08 Mvar each - over their rating (103 % at their bus's 0.969 pu); at 0.96 on the SOFC, 88 %. Rev
    C.2 with one out: 77 %; with a BESS board lost, its four at 96 %. The droop's frequency: 48.43 Hz on four PCS.
    """
    out, gfm = _bridging(client, design, state, 45, sofc_pf=pf)
    assert all(g['loading_percent'] == pytest.approx(loading, abs=0.6) for g in gfm)
    assert gfm[0]['frequency_hz'] == pytest.approx(f_hz, abs=0.03)
    assert any('loaded to 103' in w for w in out.get('warnings') or []) == (design == 'ring10' and pf is None)


def test_revc2_mva_lost_halls_on_cord_b(client):
    """Rev C.2, board MVA lost, the dual-corded halls on cord B: Other-A alone is lost, 27.5 MW served, +9.2 MW."""
    out = _post(client, _state('revc2', 'MVA_BUS_FAULT', 'AcDcOpfPandaPower Parameters', OPF, load_scale={'IT_cordB': 2.0}))
    assert out['solved'] and out['shed_mw'] == 0 and _margin(out)[0] == pytest.approx(9.21, abs=0.05)


# --- Phase 6: EMT - a bolted board fault and the recovery after it ------------------------------------------------
from test_reference_c003 import _emt_with      # noqa: E402


@pytest.mark.parametrize('design, bus, limit, recovers', [('revc2', 'SOFC1A', 1.1, True), ('ring10', 'MVA', 1.3, False),
                                                         ('ring10', 'MVA', 1.5, True)])
def test_emt_board_fault_recovery(client, design, bus, limit, recovers):
    """
    A bolted three-phase fault for 100 ms on a board: every board goes to zero, as before. Rev C.2 recovers even at
    the workbook's 1.1 pu - the PCS leave current limit 15 ms after the fault and MVA returns to 0.965 pu, balanced
    (Rev C.1 needed 1.3 pu at 35.6 MW). The ring does not at 1.1-1.3 pu: its PCS stay in limit (at 1.3 pu still after
    1 s), MVA at 0.88 pu and unbalanced; at 1.5 pu they recover (0.953 pu). Model-dependent (averaged grid-forming
    PCS, generic grid-following inverters): the PCST5500's short-time current and recovery are vendor questions.
    """
    emt, bus_res, conv = _emt_with(client, design, pcs_limit=limit, ac_fault_bus=bus, ac_fault_type='abcg',
                                   ac_fault_resistance_ohm='0.001', ac_fault_duration_ms='100', duration_ms='500')
    assert not emt['converters_blocked']
    assert all(bus_res[b]['v_rms_min_pu'] < 0.01 for b in ('MVA', 'MVB'))
    pcs = conv['PCS1 PCST5500']
    if recovers:
        assert pcs['limited_ms'] < 120 and bus_res['MVA']['vuf_end_percent'] < 0.1 and bus_res['MVA']['v_rms_final_pu'] > 0.95
    else:
        assert pcs['limited_ms'] > 400 and bus_res['MVA']['vuf_end_percent'] > 1.0 and bus_res['MVA']['v_rms_final_pu'] < 0.9


def test_emt_ring_recovers_with_the_sofc_carrying_reactive_power(client, monkeypatch):
    """
    The ring with its SOFC inverters at 0.96 power factor (R1): its PCS carry less of the IT's reactive power and
    recover from the same board fault even at 1.1 pu - out of current limit 7 ms after the fault, MVA at 0.99 pu and
    balanced. R1 settles the ring's bridging overload and its fault recovery together.
    """
    import test_reference_c003 as base
    payload = base._payload

    def with_pf(design):
        p = payload(design)
        for v in p.values():
            if isinstance(v, dict) and str(v.get('userFriendlyName', '')).endswith(' inverter'):
                v['q_mode'], v['pf'] = 'pf', '0.96'
        return p

    monkeypatch.setattr(base, '_payload', with_pf)
    emt, bus_res, conv = base._emt_with(client, 'ring10', pcs_limit=1.1, ac_fault_bus='MVA', ac_fault_type='abcg',
                                        ac_fault_resistance_ohm='0.001', ac_fault_duration_ms='100', duration_ms='500')
    assert not emt['converters_blocked'] and conv['PCS1 PCST5500']['limited_ms'] < 120
    assert bus_res['MVA']['v_rms_final_pu'] == pytest.approx(0.99, abs=0.01) and bus_res['MVA']['vuf_end_percent'] < 0.1


# --- Phase 5b: the bridging ramp in ANDES, the PCS holding their Q-V droop ----------------------------------------
from test_reference_c003 import _site      # noqa: E402


def _ramp(client, design, load_model, tf=20.0):
    """The bridging ramp in ANDES: one PCS out at 45 C, the SOFC at 0.96 pf and at load - 15.58 MW, the IT ramping
    1.244 MW/s net of the SOFC for 12.5 s."""
    import numpy as np
    gap, rate, dt = 15.58, 1.64 - 0.396, 0.25
    p = _state(design, 'N1_PCS', 'TransientStabilityAndes Parameters', {
        'sn_mva': '100', 'tf': str(tf), 'tstep': '0', 'fault_enabled': 'false', 'load_model': load_model})
    n_on = len([x for x in next(st for st in _site(design)['islands'] if st['id'] == 'N1_PCS')['energized_source_ids']
                if x.startswith('SOFC')])
    f0 = (LOAD_MW - gap) / LOAD_MW
    p['0']['load_profiles'] = {'ramp': {'name': 'ramp', 'kind': 'power', 'dt': dt, 't0': 0,
                                        'p': [min(f0 + (1 - f0) * k * dt / (gap / rate), 1.0) for k in range(int(tf / dt) + 2)]}}
    for v in p.values():
        if not isinstance(v, dict):
            continue
        name, t = str(v.get('userFriendlyName', '')), str(v.get('typ', ''))
        if 'PCST5500' in name:
            v['inlet_c'] = '45'
        if name.endswith(' SOFC'):
            v['p_set_kw'] = str(round((LOAD_MW - gap) / n_on * 1e3, 3))
        if name.endswith(' inverter'):
            v['q_mode'], v['pf'] = 'pf', '0.96'
        if t.startswith('Load') and not t.startswith('Load DC'):
            v['load_profile_id'], v['load_profile_q_mode'] = 'ramp', 'pf'
    out = _post(client, p)
    gfm = [x for x in out['pcs'] if x['control'] == 'grid_forming']
    sofc = sum(np.asarray(x['p_mw'], dtype=float) for x in out['pcs'] if x['control'] != 'grid_forming')
    return out, np.asarray(gfm[0]['frequency_hz'], dtype=float), sofc


@pytest.mark.parametrize('design, f_min, sofc_lost', [('ring10', 48.26, 1.71), ('revc2', 48.75, 0.0)])
def test_andes_bridging_ramp_reaches_the_sofc_under_frequency_trip(client, design, f_min, sofc_lost):
    """
    The bridging ramp in ANDES, the IT at constant power, each PCS holding its Q-V droop: the island's frequency is
    the droop's, 48.4 Hz at the ring's peak with four PCS - past the SOFC inverters' under-frequency trip (48.75 Hz,
    IEEE 1547 Category III at 50 Hz, ASSUMED): some 1.7 MW of SOFC trips and the ring settles at 48.26 Hz, its PCS at
    their 45 C rating. Rev C.2's five PCS reach 48.75 Hz, the trip point itself. Without secondary control acting
    within the ramp (R2) the SOFC's frequency ride-through must reach below the droop's excursion.
    """
    _, f, sofc = _ramp(client, design, 'power')
    assert f.min() == pytest.approx(f_min, abs=0.03)
    assert sofc[0] - sofc[-1] == pytest.approx(sofc_lost, abs=0.1)


def _step(client, design, state, tf=10.0, dt=0.05):
    """The owner's 65 % step at t = 1 s: the IT from idle (11.05 MW) to 30 MW at constant power, the SOFC held at their
    idle dispatch (0.96 pf), the PCS at 45 C holding their Q-V droop."""
    import numpy as np
    p = _state(design, state, 'TransientStabilityAndes Parameters', {
        'sn_mva': '100', 'tf': str(tf), 'tstep': '0', 'fault_enabled': 'false', 'load_model': 'power'})
    n_on = len([x for x in next(st for st in _site(design)['islands'] if st['id'] == state)['energized_source_ids']
                if x.startswith('SOFC')])
    p['0']['load_profiles'] = {'step': {'name': 'step', 'kind': 'power', 'dt': dt, 't0': 0,
                                        'p': [11.05 / LOAD_MW if k * dt < 1.0 else 1.0 for k in range(int(tf / dt) + 2)]}}
    for v in p.values():
        if not isinstance(v, dict):
            continue
        name, t = str(v.get('userFriendlyName', '')), str(v.get('typ', ''))
        if 'PCST5500' in name:
            v['inlet_c'] = '45'
        if name.endswith(' SOFC'):
            v['p_set_kw'] = str(round(11.05 * 1.01 / n_on * 1e3, 1))
        if name.endswith(' inverter'):
            v['q_mode'], v['pf'] = 'pf', '0.96'
        if t.startswith('Load') and not t.startswith('Load DC'):
            v['load_profile_id'], v['load_profile_q_mode'] = 'step', 'pf'
    out = _post(client, p)
    gfm = [x for x in out['pcs'] if x['control'] == 'grid_forming']
    f = np.asarray(gfm[0]['frequency_hz'], dtype=float)
    sofc = sum(np.asarray(x['p_mw'], dtype=float) for x in out['pcs'] if x['control'] != 'grid_forming')
    cur = np.max([np.asarray(x['current_pu'], dtype=float) for x in gfm], axis=0)
    return out, f, sofc, cur


@pytest.mark.parametrize('design, state, f_end, i_peak', [('ring10', 'NORMAL', 48.46, 0.92), ('revc2', 'NORMAL', 48.74, 0.72),
                                                         ('revc2', 'N1_PCS', 48.47, 0.89)])
def test_andes_65_percent_step_carried(client, design, state, f_end, i_peak):
    """
    The owner's 65 % step taken at once - 18.95 MW on the PCS, the SOFC still at idle: the ring with its five PCS and
    Rev C.2 with five or six carry it at 45 C, the frequency on their droop (48.46, 48.74 and 48.47 Hz), the PCS's
    current at most 0.92 pu, the batteries giving some 1.5 % of their charge in 9 s. (The workload's own ramp, 8 %/s,
    is the bridging case.)
    """
    out, f, sofc, cur = _step(client, design, state)
    assert out['time'][-1] == pytest.approx(10.0)
    assert f[-1] == pytest.approx(f_end, abs=0.03) and cur.max() == pytest.approx(i_peak, abs=0.03)
    assert sofc[-1] > sofc[0] - 0.3


def test_andes_65_percent_step_ring_with_a_pcs_out_collapses(client):
    """
    The ring with a PCS out at 45 C: four PCS take the 19 MW step through their virtual machines' 0.2 pu, and the
    switchboards dip below 0.88 pu within 10 ms - where the SOFC inverters cease (IEEE 1547 momentary cessation,
    ASSUMED). The PCS then carry the SOFC's 11 MW too, pass their limit and the island collapses within 50 ms - at
    1.5 pu as at 1.1. Some 16 MW at once is the edge. Model-dependent: a real PCS's voltage loop acts within
    milliseconds; the requirement is that the SOFC ride through such a dip (R5) and the workload step no faster
    than its 8 %/s ramp.
    """
    out, f, sofc, cur = _step(client, 'ring10', 'N1_PCS', tf=4.0)
    assert out['time'][-1] < 1.1
    assert sofc[-1] < 0.5 * sofc[0] and cur[-1] > 1.1
