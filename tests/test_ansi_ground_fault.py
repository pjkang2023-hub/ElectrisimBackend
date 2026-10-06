# -*- coding: utf-8 -*-
"""
ANSI line-to-ground faults: I = 3 E / |Z1 + Z2 + Z0 + 3 Zf| on the zero-sequence
network built from the model - lines as series branches with R0 / X0 / C0,
external grids with their X0/X1 and R0/X0, two- and three-winding transformers
by vector group with 3 Z_N for a neutral impedance, grounding transformers.

Checked against hand calculations and against pandapower's IEC 60909
single-phase currents with its voltage factor c and transformer correction K_T
both set to 1, which is then the same E / Z calculation as ANSI's at 1.0 pu
prefault voltage on a network of external grids alone.
"""
import copy
import json
import math

import numpy as np
import pandapower as pp
import pandapower.shortcircuit as sc
import pytest

import ansi_shortcircuit_electrisim as ansi
import electrisim_sld as sld
from test_grounding_transformer import R0, R_N, RX, S_SC, V, X0, _breaker, _network, _zigzag
from test_pcs import _post
from test_reference_grids import ANSI_PARAMS, load_spec

ANSI_1PH = {**ANSI_PARAMS, 'fault_type': '1ph', 'user_email': 't@t'}


def _ansi(net, **params):
    return json.loads(ansi.shortcircuit_ansi(net, {'fault_type': '1ph', 'frequency_hz': 60, 'prefault_v_pu': 1.0,
                                                   'contact_parting_cycles': 3, **params}))


def _ansi_by_name(net, **params):
    return {r['name']: r for r in _ansi(net, **params)['busbars']}


@pytest.fixture
def iec_c1(monkeypatch):
    """pandapower's IEC 60909 with c = 1 and K_T = 1."""
    import pandapower.build_bus as build_bus
    import pandapower.build_branch as build_branch
    import pandapower.pd2ppc_zero as pd2ppc_zero
    import pandapower.shortcircuit.ppc_conversion as ppc_conversion
    from pandapower.pypower.idx_bus_sc import C_MAX, C_MIN

    add_c = build_bus._add_c_to_ppc

    def c_one(net, ppc):
        add_c(net, ppc)
        ppc['bus'][:, C_MAX] = 1.0
        ppc['bus'][:, C_MIN] = 1.0

    monkeypatch.setattr(build_bus, '_add_c_to_ppc', c_one)
    for module in (build_branch, pd2ppc_zero, ppc_conversion):
        monkeypatch.setattr(module, '_transformer_correction_factor',
                            lambda trafo_df, vk, vkr, sn, cmax: np.ones_like(np.asarray(vk, dtype=float)))

    def run(net):
        work = copy.deepcopy(net)
        sc.calc_sc(work, fault='1ph', case='max', branch_results=False)
        return dict(zip(work.bus.loc[work.res_bus_sc.index, 'name'], work.res_bus_sc['ikss_ka']))
    return run


# --- Grounding transformer on 35 kV --------------------------------------------------------------

def _hand_zigzag_ka(grounded, zigzags=1):
    """sqrt(3) U / |2 Z1 + Z0| at 1.0 pu: Z1 = U^2 / S_sc; Z0 the zigzag's Z0 + 3 R_N,
    in parallel with the source's (X0/X1 1, R0/X0 0.1) or, ungrounded, X0/X1 1e6."""
    z = V * V / S_SC
    x1 = z / math.sqrt(1 + RX * RX)
    z1 = complex(RX * x1, x1)
    x0x = 1.0 if grounded else 1e6
    y0 = zigzags / complex(R0 + 3 * R_N, X0) + 1 / complex(0.1 * x0x * x1, x0x * x1)
    return math.sqrt(3) * V / abs(2 * z1 + 1 / y0)


@pytest.mark.parametrize('grounded', [False, True])
def test_grounding_transformer_against_the_hand_calculation(client, quiet, grounded):
    """
    Ungrounded source: the zigzag alone, sqrt(3) U / |2 Z1 + Z0 + 3 R_N| ~ 400 A
    at 1.0 pu; two in parallel, about twice that. Grounded: the utility's and the
    zigzag's zero-sequence paths in parallel. The same in all three networks.
    """
    rows = _post(client, quiet, _network(ANSI_1PH, grounded, _zigzag(), _breaker('zz')))['busbars']
    (a,) = [r for r in rows if r['name'] == 'a']
    for key in ('i_first_sym_ka', 'i_steady_ka'):
        assert a[key] == pytest.approx(_hand_zigzag_ka(grounded), rel=1e-6)
    if not grounded:
        assert a['i_first_sym_ka'] == pytest.approx(0.4003, abs=1e-4)
    two = _post(client, quiet, _network(ANSI_1PH, grounded, _zigzag('z1', bus='a'), _zigzag('z2', bus='a')))
    (a2,) = [r for r in two['busbars'] if r['name'] == 'a']
    assert a2['i_first_sym_ka'] == pytest.approx(_hand_zigzag_ka(grounded, zigzags=2), rel=1e-6)


def test_open_breaker_leaves_an_ungrounded_network_without_ground_fault_current(client, quiet):
    """Before, the source counted as solidly grounded whatever its X0/X1: ~15 kA here."""
    rows = _post(client, quiet, _network(ANSI_1PH, False, _zigzag(), _breaker('zz', closed=False)))['busbars']
    (a,) = [r for r in rows if r['name'] == 'a']
    assert a['i_first_sym_ka'] < 1e-3


@pytest.mark.parametrize('grounded', [False, True])
def test_grounding_transformer_matches_iec_at_c_1(iec_c1, grounded):
    import pandapower_electrisim as pe
    net = pp.create_empty_network(f_hz=60)
    a = pp.create_bus(net, V, name='a')
    x0x = 1.0 if grounded else 1e6
    pp.create_ext_grid(net, a, s_sc_max_mva=S_SC, s_sc_min_mva=S_SC, rx_max=RX, rx_min=RX,
                       x0x_max=x0x, r0x0_max=0.1, x0x_min=x0x, r0x0_min=0.1)
    pe._electrisim_build_grounding_transformer(
        net, {'name': 'zz', 'bus': 'a', 'r_n_ohm': R_N, 'x0_ohm': X0, 'r0_ohm': R0, 'i_rated_a': 400},
        {'a': a}, {}, {})
    assert _ansi_by_name(net)['a']['i_first_sym_ka'] == pytest.approx(iec_c1(net)['a'], rel=1e-9)


# --- Two-winding transformers ---------------------------------------------------------------------

S_HV, VK, VKR, VK0, VKR0 = 2000.0, 10.0, 0.5, 9.0, 0.45
LINE_Z1, LINE_Z0 = complex(0.6, 1.05), complex(1.8, 3.6)        # 3 km


def _substation(vector_group, rn_ohm=0.0, x0x=1.0, shift=0.0):
    """69 kV source - 20 MVA 69/13.8 kV transformer - 3 km feeder."""
    net = pp.create_empty_network(f_hz=60)
    hv = pp.create_bus(net, 69.0, name='hv')
    lv = pp.create_bus(net, 13.8, name='lv')
    end = pp.create_bus(net, 13.8, name='end')
    pp.create_ext_grid(net, hv, s_sc_max_mva=S_HV, s_sc_min_mva=S_HV, rx_max=0.1, rx_min=0.1,
                       x0x_max=x0x, r0x0_max=0.1, x0x_min=x0x, r0x0_min=0.1)
    t = pp.create_transformer_from_parameters(
        net, hv, lv, sn_mva=20, vn_hv_kv=69, vn_lv_kv=13.8, vk_percent=VK, vkr_percent=VKR, pfe_kw=0,
        i0_percent=0, vector_group=vector_group, vk0_percent=VK0, vkr0_percent=VKR0, mag0_percent=100,
        mag0_rx=0, si0_hv_partial=0.9, shift_degree=shift)
    net.trafo['xn_ohm'] = 0.0
    net.trafo.at[t, 'rn_ohm'] = rn_ohm
    pp.create_line_from_parameters(net, lv, end, length_km=3, r_ohm_per_km=0.2, x_ohm_per_km=0.35, c_nf_per_km=0,
                                   max_i_ka=1, r0_ohm_per_km=0.6, x0_ohm_per_km=1.2, c0_nf_per_km=0)
    return net


def _zpct(vk, vkr, kv, sn=20.0):
    return complex(vkr, math.sqrt(vk * vk - vkr * vkr)) / 100 * kv * kv / sn


def _source_z1_lv():
    z = 69.0 ** 2 / S_HV
    x = z / math.sqrt(1.01)
    return complex(0.1 * x, x) * (13.8 / 69.0) ** 2


def test_dyn11_with_a_400_a_neutral_resistor():
    """
    Dyn11 at 13.8 kV grounded through R_N = 13.8 kV / sqrt(3) / 400 A: at its
    bus 3 E / |2 Z1 + Z0T + 3 R_N| a little under 400 A, its X/R that of
    2 Z1 + Z0 (resistive, so the peak barely above the symmetrical current);
    down the feeder the line's Z1 and Z0 added; at 69 kV the source alone.
    """
    r_n = 13.8e3 / math.sqrt(3) / 400
    rows = _ansi_by_name(_substation('Dyn', rn_ohm=r_n, shift=330))
    z1 = _source_z1_lv() + _zpct(VK, VKR, 13.8)
    z0 = _zpct(VK0, VKR0, 13.8) + 3 * r_n
    want_lv = math.sqrt(3) * 13.8 / abs(2 * z1 + z0)
    assert rows['lv']['i_first_sym_ka'] == pytest.approx(want_lv, rel=1e-9)
    assert 0.39 < want_lv < 0.40
    assert rows['lv']['xr_first'] == pytest.approx((2 * z1 + z0).imag / (2 * z1 + z0).real, rel=1e-9)
    assert rows['lv']['i_first_peak_ka'] < 1.45 * want_lv
    want_end = math.sqrt(3) * 13.8 / abs(2 * (z1 + LINE_Z1) + z0 + LINE_Z0)
    assert rows['end']['i_first_sym_ka'] == pytest.approx(want_end, rel=1e-9)
    z_src = 69.0 ** 2 / S_HV / math.sqrt(1.01) * complex(0.1, 1)
    assert rows['hv']['i_first_sym_ka'] == pytest.approx(math.sqrt(3) * 69 / abs(3 * z_src), rel=1e-9)


@pytest.mark.parametrize('vector_group', ['Dyn', 'YNyn'])
def test_poi_neutral_resistor_fold_matches_ansi(iec_c1, vector_group):
    """
    The POI study's IEC rows fold rn_ohm into vk0 / vkr0 (pandapower does not
    read it). On the grounded winding's base - the LV for Dyn, where the HV base
    made a 13.8 kV resistor 25x too small - and as a resistance, the IEC current
    at c = 1 is ANSI's, which reads rn_ohm itself. YNyn only nearly: folded, 3 Z_N
    is spread over the T model with Z0 instead of sitting in the LV leg.
    """
    from poi_fault_study_electrisim import apply_ngr_to_net
    net = _substation(vector_group, rn_ohm=13.8e3 / math.sqrt(3) / 400)
    rows = _ansi_by_name(net)
    folded = copy.deepcopy(net)
    apply_ngr_to_net(folded)
    got = iec_c1(folded)
    buses, rel = (('hv', 'lv', 'end'), 1e-9) if vector_group == 'Dyn' else (('lv', 'end'), 1e-3)
    for bus in buses:
        assert got[bus] == pytest.approx(rows[bus]['i_first_sym_ka'], rel=rel), bus
    if vector_group == 'Dyn':
        assert 0.39 < got['lv'] < 0.40


def test_fault_resistance_counts_three_times():
    """I = 3 E / |Z1 + Z2 + Z0 + 3 Zf|."""
    z1 = _source_z1_lv() + _zpct(VK, VKR, 13.8)
    z0 = _zpct(VK0, VKR0, 13.8)
    rows = _ansi_by_name(_substation('Dyn'), r_fault_ohm=2.0)
    assert rows['lv']['i_first_sym_ka'] == pytest.approx(math.sqrt(3) * 13.8 / abs(2 * z1 + z0 + 6.0), rel=1e-9)


@pytest.mark.parametrize('vector_group', ['Dyn', 'YNyn', 'Yyn'])
def test_vector_groups_match_iec_at_c_1(iec_c1, vector_group):
    """Dyn: shunt on the LV side; YNyn: series, with its magnetizing branch; Yyn: through Z_m0 alone."""
    net = _substation(vector_group)
    rows = _ansi_by_name(net)
    for bus, ikss in iec_c1(net).items():
        assert rows[bus]['i_first_sym_ka'] == pytest.approx(ikss, rel=1e-9), bus


@pytest.mark.parametrize('vector_group', ['YNd', 'Dd', 'Yd', 'Dy'])
def test_no_ground_path_through_the_transformer(vector_group):
    """
    No grounded winding on the LV side: no ground-fault current there (pandapower
    finds the zero-sequence matrix singular). YNd adds the transformer's Z0 + 3 R_N
    at 69 kV in parallel with the source.
    """
    rows = _ansi_by_name(_substation(vector_group, rn_ohm=5.0))
    assert rows['lv']['i_first_sym_ka'] == 0.0 and rows['end']['i_first_sym_ka'] == 0.0
    z_src = 69.0 ** 2 / S_HV / math.sqrt(1.01) * complex(0.1, 1)
    z0 = z_src
    if vector_group == 'YNd':
        z0 = 1 / (1 / z_src + 1 / (_zpct(VK0, VKR0, 69.0) + 15.0))
    assert rows['hv']['i_first_sym_ka'] == pytest.approx(math.sqrt(3) * 69 / abs(2 * z_src + z0), rel=1e-9)


def test_ungrounded_source_feeding_a_dyn_transformer():
    """The source's X0/X1 1e6 takes the 69 kV ground fault away; the Dyn LV side keeps its own."""
    rows = _ansi_by_name(_substation('Dyn', x0x=1e6))
    assert rows['hv']['i_first_sym_ka'] < 1e-3
    grounded = _ansi_by_name(_substation('Dyn'))
    assert rows['lv']['i_first_sym_ka'] == pytest.approx(grounded['lv']['i_first_sym_ka'], rel=1e-9)


def test_source_x0x_zero_is_taken_as_not_given():
    """The diagram's default X0/X1 of 0 (which IEC rejects) counts as 1, not as a zero-impedance ground."""
    zero = _ansi_by_name(_substation('Dyn', x0x=0.0))
    one = _ansi_by_name(_substation('Dyn', x0x=1.0))
    assert zero['hv']['i_first_sym_ka'] == pytest.approx(one['hv']['i_first_sym_ka'], rel=1e-12)


def test_generator_with_zero_sequence_data():
    """A generator given x0_pu is a grounded source of Z0 + 3 Z_N at its bus; without it, none."""
    net = _substation('Dyn')
    pp.create_gen(net, 1, p_mw=5, vm_pu=1.0, sn_mva=10, vn_kv=13.8, xdss_pu=0.15, rdss_ohm=0.0, cos_phi=0.9)
    without = _ansi_by_name(net)['lv']
    net.gen['x0_pu'] = 0.08
    net.gen['rn_ohm'] = 4.0
    with_z0 = _ansi_by_name(net)['lv']
    z1_sys = _source_z1_lv() + _zpct(VK, VKR, 13.8)
    x_gen = 0.15 * 13.8 ** 2 / 10
    z1 = 1 / (1 / z1_sys + 1 / complex(0.05 * x_gen, x_gen))
    z0_gen = complex(12.0, 0.08 * 13.8 ** 2 / 10)
    z0 = 1 / (1 / _zpct(VK0, VKR0, 13.8) + 1 / z0_gen)
    assert with_z0['i_first_sym_ka'] == pytest.approx(math.sqrt(3) * 13.8 / abs(2 * z1 + z0), rel=1e-9)
    assert without['i_first_sym_ka'] == pytest.approx(
        math.sqrt(3) * 13.8 / abs(2 * z1 + _zpct(VK0, VKR0, 13.8)), rel=1e-9)


# --- Reference grids --------------------------------------------------------------------------------

def _ansi_z0(net):
    """Z0 at each bus; None where no zero-sequence path reaches ground."""
    work = copy.deepcopy(net)
    ppci0 = ansi._build_zero_sequence_y(work, 1.0)
    diag = np.diag(ppci0['internal']['Zbus'])
    floating = ppci0['internal']['z0_floating']
    return {int(b): None if floating[ansi._ppc_bus(work, int(b))] else diag[ansi._ppc_bus(work, int(b))]
            for b in work.bus.index}


def _pandapower_z0(net):
    import pandapower.pd2ppc_zero as pd2ppc_zero
    from pandapower.shortcircuit.impedance import _calc_ybus, _calc_zbus
    work = copy.deepcopy(net)
    ansi._init_ansi_ppc(work, 1.0)
    _, ppci0 = pd2ppc_zero._pd2ppc_zero(work, None)
    _calc_ybus(ppci0)
    _calc_zbus(work, ppci0)
    diag = np.diag(ppci0['internal']['Zbus'])
    return {int(b): diag[work._pd2ppc_lookups['bus'][b]] for b in work.bus.index}


@pytest.mark.parametrize('grid', ['reference_radial', 'reference_transmission'])
def test_reference_grid_zero_sequence_matches_pandapower(iec_c1, grid):
    """
    Every bus's Z0 as pandapower's own zero-sequence network has it, K_T 1: Dyn,
    YNyn and the YNynd three-winding transformer, lines with their R0 / X0 / C0.
    pandapower's generators keep a token 1000 pu shunt the ANSI network leaves out.
    """
    net, _ = sld.build_network(load_spec(grid))
    got, want = _ansi_z0(net), _pandapower_z0(net)
    for bus, z in want.items():
        if got[bus] is None:        # behind a delta: pandapower's open branches are 1e20 pu
            assert abs(z) > 1e10, net.bus.at[bus, 'name']
        else:
            assert got[bus] == pytest.approx(z, rel=1e-4), net.bus.at[bus, 'name']
    assert any(z is None for z in got.values()) == (grid == 'reference_transmission')


def _trafo_branches(net, build):
    work = copy.deepcopy(net)
    _, ppci = build(work)
    rows = {('trafo', i): r for i, r in ansi._ppci_branch_rows(work, ppci, 'trafo').items()}
    for i, legs in ansi._ppci_branch_rows(work, ppci, 'trafo3w', n_sides=3).items():
        rows.update({('trafo3w', i, k): r for k, r in enumerate(legs)})
    return {key: complex(ppci['branch'][r, ansi.BR_R].real, ppci['branch'][r, ansi.BR_X].real)
            for key, r in rows.items()}


@pytest.mark.parametrize('grid', ['reference_radial', 'reference_transmission'])
def test_positive_sequence_transformers_carry_no_iec_kt(monkeypatch, grid):
    """
    pandapower builds sc-mode transformer impedances already multiplied by IEC
    K_T (~0.986 for a 10 % unit), so ANSI's 3ph currents behind them were ~1.4 %
    high. Each two-winding branch and three-winding leg must be pandapower's
    built with K_T = 1.
    """
    import pandapower.build_branch as build_branch
    net, _ = sld.build_network(load_spec(grid))
    got = _trafo_branches(net, lambda work: ansi._init_ansi_ppc(work, 1.0))
    monkeypatch.setattr(build_branch, '_transformer_correction_factor',
                        lambda trafo_df, vk, vkr, sn, cmax: np.ones_like(np.asarray(vk, dtype=float)))
    want = _trafo_branches(net, lambda work: ansi._init_ansi_ppc(work, 1.0))
    assert got.keys() == want.keys() and len(got) >= 2
    assert any(key[0] == 'trafo3w' for key in got) == (grid == 'reference_transmission')
    for key, z in want.items():
        assert got[key] == pytest.approx(z, rel=1e-9, abs=1e-15), key
