# -*- coding: utf-8 -*-
"""
ANSI/IEEE C37 short-circuit analysis for Electrisim.

Implements first-cycle (1/2-cycle), interrupting (1.5-4 cycle), and 30-cycle
symmetrical networks with IEEE C37.010 / C37.13 / UL 489 / C37.013 duty
factors. Uses the pandapower network model but does NOT call IEC 60909 calc_sc.

Beta: engineering implementation for review. Verify against utility
requirements before using results for equipment ratings.
"""
from __future__ import annotations

import copy
import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandapower as pp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from pandapower.auxiliary import _add_auxiliary_elements
from pandapower.pd2ppc import _pd2ppc, _ppc2ppci
from pandapower.pypower.idx_brch import (
    BR_B, BR_B_ASYM, BR_G, BR_G_ASYM, BR_R, BR_R_ASYM, BR_STATUS, BR_X, BR_X_ASYM,
    F_BUS, SHIFT, T_BUS, TAP,
)
from pandapower.pypower.idx_bus import BASE_KV, BS, GS
from pandapower.pypower.idx_bus_sc import C_MAX, C_MIN
from pandapower.pypower.idx_brch_sc import K_T, K_ST
from pandapower.shortcircuit.impedance import _calc_ybus, _calc_zbus

from sc_fault_location import (
    collect_fault_bus_refs,
    normalize_fault_bus_mode,
    resolve_pp_fault_bus_indices,
)

try:
    from pandapower.pf.makeYbus_numba import makeYbus
except ImportError:
    from pandapower.pypower.makeYbus import makeYbus

NETWORK_FIRST = "first_cycle"
NETWORK_INT = "interrupting"
NETWORK_30 = "steady_state"

ANSI_MACHINE_TYPES = ("turbo", "hydro_amortisseur", "hydro", "sync_motor")
DEVICE_CLASSES = ("auto", "hv_c37_010", "lv_c37_013", "mccb_ul_489", "generator_c37_013")


def _f(v, default=0.0) -> float:
    try:
        if v is None or v == "" or str(v).lower() in ("null", "none", "nan"):
            return float(default)
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _clean(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        if math.isnan(float(v)) or math.isinf(float(v)):
            return None
        return float(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    return v


def _display_name(net, internal_name: str, row_id: Any = None) -> str:
    uf = getattr(net, "user_friendly_names", None) or {}
    for key in (internal_name, str(internal_name).replace("#", "_"), str(internal_name).replace("_", "#")):
        if key in uf and uf[key]:
            return str(uf[key])
    return str(internal_name) if internal_name else (str(row_id) if row_id is not None else "")


# ---------------------------------------------------------------------------
# IEEE C37.010 multiplying factors (60 Hz primary; 50 Hz adjusted)
# ---------------------------------------------------------------------------

def _dc_decay(xr: float, t_s: float, freq_hz: float) -> float:
    """DC component as a fraction of the initial value after t_s seconds.

    The DC time constant is tau = (X/R) / (2*pi*f), so the decay depends only on
    X/R and the elapsed time expressed in cycles.
    """
    xr = max(float(xr), 1e-6)
    return math.exp(-2.0 * math.pi * float(freq_hz) * float(t_s) / xr)


def _peak_multiplier_remote(xr: float, freq_hz: float = 60.0) -> float:
    """First-cycle peak (crest) current as a multiple of the symmetrical rms.

    ip / Isym = sqrt(2) * (1 + exp(-pi / (X/R))), the first peak of a fully
    offset wave half a cycle after fault inception. Bounded by 2*sqrt(2) as
    X/R -> infinity. Frequency independent, because both the DC time constant
    and the half-cycle time scale with 1/f.
    """
    dc = _dc_decay(xr, 1.0 / (2.0 * float(freq_hz)), freq_hz)
    return math.sqrt(2.0) * (1.0 + dc)


def _interrupting_multiplier_remote(xr: float, cp_cycles: float, freq_hz: float = 60.0) -> float:
    """Total asymmetrical rms at contact parting, as a multiple of symmetrical.

    Itotal = sqrt(Iac^2 + Idc^2) with Idc = sqrt(2) * Isym * decay, giving
    sqrt(1 + 2 * decay^2).
    """
    dc = _dc_decay(xr, float(cp_cycles) / float(freq_hz), freq_hz)
    return math.sqrt(1.0 + 2.0 * dc * dc)


def _symmetry_factor_s(cp_cycles: float, freq_hz: float = 60.0) -> float:
    """Asymmetrical rms already covered by a symmetrically rated breaker.

    IEEE C37.010 rates breakers on a test circuit with X/R = 17, so the duty a
    symmetrically rated device tolerates is the asymmetrical rms of that circuit
    at contact parting. The system duty is divided by this value.
    """
    dc = _dc_decay(17.0, float(cp_cycles) / float(freq_hz), freq_hz)
    return math.sqrt(1.0 + 2.0 * dc * dc)


def _lv_test_xr(device_class: str) -> float:
    if device_class == "mccb_ul_489":
        return 4.9
    return 6.6


def _lv_extra_multiplier(xr: float, device_class: str, freq_hz: float = 60.0) -> float:
    """Derating when the system X/R exceeds the LV device's test X/R.

    LV devices are tested on a circuit with a specified X/R (6.6 for C37.13
    power circuit breakers, 4.9 for UL 489 moulded case), so the duty is scaled
    by the ratio of first-cycle peak asymmetry factors. Never below 1.0.
    """
    test_xr = _lv_test_xr(device_class)
    if xr <= test_xr:
        return 1.0
    return _peak_multiplier_remote(xr, freq_hz) / _peak_multiplier_remote(test_xr, freq_hz)


def _machine_x_multiplier(network: str, machine_kind: str, hp: float = 0.0) -> float:
    """ANSI reactance multiplier on subtransient base."""
    if network == NETWORK_FIRST:
        return 1.0
    if network == NETWORK_INT:
        if machine_kind == "motor":
            if hp > 0 and hp < 50.0:
                return 0.0
            return 4.5
        return 1.5
    if network == NETWORK_30:
        if machine_kind == "motor":
            return 0.0
        return 1.5
    return 1.0


def _motor_hp(pn_mech_mw: float) -> float:
    if pn_mech_mw <= 0:
        return 0.0
    return pn_mech_mw * 1000.0 / 0.746


# ---------------------------------------------------------------------------
# PPC / Ybus for ANSI (no IEC c-factor, no K_G, no K_T)
# ---------------------------------------------------------------------------

def _init_ansi_ppc(net, prefault_v: float = 1.0):
    """Build ppc/ppci for ANSI study without IEC 60909 corrections."""
    from pandapower.auxiliary import _add_ppc_options, _add_sc_options
    net["_options"] = {}
    _add_ppc_options(
        net,
        calculate_voltage_angles=False,
        trafo_model="pi",
        check_connectivity=False,
        mode="sc",
        switch_rx_ratio=2,
        init_vm_pu="flat",
        init_va_degree="flat",
        enforce_q_lims=False,
        recycle=None,
    )
    _add_sc_options(
        net,
        fault="3ph",
        case="max",
        lv_tol_percent=10,
        tk_s=1.0,
        topology="auto",
        r_fault_ohm=0.0,
        x_fault_ohm=0.0,
        kappa=False,
        ip=False,
        ith=False,
        branch_results=False,
        kappa_method="C",
        return_all_currents=False,
        inverse_y=True,
        use_pre_fault_voltage=False,
    )
    _add_auxiliary_elements(net)
    # A second build on the same net (the zero sequence after the positive) found the
    # last one's auxiliary-bus lookups and read its VSC table, built empty in sc mode:
    # with a VSC on the network every ground fault failed. Start as the first build does.
    lookups = net.get('_pd2ppc_lookups')
    if isinstance(lookups, dict):
        lookups.pop('aux', None)
    ppc, _ = _pd2ppc(net)
    ppci = _ppc2ppci(ppc, net)
    # In sc mode _pd2ppc pre-builds the ext_grid and motor short-circuit impedances
    # into the bus shunt columns using IEC voltage factors. ANSI needs to own the
    # source model so it can apply per-network machine multipliers (first-cycle,
    # interrupting, 30-cycle), so clear those shunts and let the _add_*_ansi
    # helpers rebuild them. Nothing legitimate is discarded: sc mode never calls
    # _calc_shunts_and_add_on_ppc, so capacitors, reactors and wards are absent,
    # and load impedances are only added when use_pre_fault_voltage is set.
    ppci["bus"][:, GS] = 0.0
    ppci["bus"][:, BS] = 0.0
    _remove_iec_kt(net, ppci)
    ppci["bus"][:, C_MAX] = prefault_v
    ppci["bus"][:, C_MIN] = prefault_v
    if ppci["branch"].shape[1] > K_T:
        ppci["branch"][:, K_T] = 1.0
    if ppci["branch"].shape[1] > K_ST:
        ppci["branch"][:, K_ST] = 1.0
    # _calc_ybus / _calc_zbus repopulate "internal", but branch_is comes from
    # _ppc2ppci and is what maps pandapower branches onto ppci rows once
    # out-of-service branches have been dropped, so carry it across.
    branch_is = ppci["internal"].get("branch_is") if ppci.get("internal") else None
    ppci["internal"] = {} if branch_is is None else {"branch_is": branch_is}
    return ppc, ppci


def _remove_iec_kt(net, ppci):
    """Divide IEC 60909's transformer correction K_T back out of the branches.

    In sc mode pandapower multiplies two- and three-winding transformer R and X
    by K_T while building the ppc (the K_T column is only a record), so
    resetting the column does not undo it. ANSI has no such correction. Runs
    before C_MAX is overwritten: two-winding K_T uses the LV bus's c_max.
    """
    from pandapower.build_branch import (
        _transformer_correction_factor, wye_delta_vector, z_br_to_bus_vector,
    )
    branch = ppci["branch"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    rows = _ppci_branch_rows(net, ppci, "trafo")
    if rows:
        # Only transformers in ppci: an out-of-service one's LV bus may not be.
        tr = net.trafo.loc[list(rows)]
        cmax = ppci["bus"][bus_lookup[tr["lv_bus"].values.astype(np.int64)], C_MAX]
        kt = np.asarray(_transformer_correction_factor(
            tr, tr["vk_percent"], tr["vkr_percent"], tr["sn_mva"], cmax), dtype=float)
        for idx, k in zip(tr.index, np.broadcast_to(kt, (len(tr),))):
            if k > 0:
                branch[rows[idx], BR_R] /= k
                branch[rows[idx], BR_X] /= k
    legs = _ppci_branch_rows(net, ppci, "trafo3w", n_sides=3) if hasattr(net, "trafo3w") else {}
    if legs:
        # K_T scales each winding pair's vk / vkr before the star conversion,
        # so each leg's R and X go back by the ratio of the uncorrected to the
        # corrected star value.
        t3 = net.trafo3w
        sides = ("hv", "mv", "lv")
        vk = np.stack([t3[f"vk_{s}_percent"].values for s in sides]).astype(float)
        vkr = np.stack([t3[f"vkr_{s}_percent"].values for s in sides]).astype(float)
        sn = np.stack([t3[f"sn_{s}_mva"].values for s in sides]).astype(float)
        kt = np.asarray(_transformer_correction_factor(t3, vk, vkr, sn, 1.1), dtype=float)
        vk_d, vkr_d = z_br_to_bus_vector(vk, sn), z_br_to_bus_vector(vkr, sn)
        vki_d = np.sqrt(vk_d ** 2 - vkr_d ** 2)
        r_ratio = wye_delta_vector(vkr_d, sn), wye_delta_vector(kt * vkr_d, sn)
        x_ratio = wye_delta_vector(vki_d, sn), wye_delta_vector(kt * vki_d, sn)
        for pos, idx in enumerate(t3.index):
            for k, row in enumerate(legs.get(int(idx), ())):
                if row is None:
                    continue
                for col, (true, built) in ((BR_R, r_ratio), (BR_X, x_ratio)):
                    if abs(built[k, pos]) > 1e-12:
                        branch[row, col] *= true[k, pos] / built[k, pos]


def _y_add_shunt(ppci, bus_ppc: int, y_pu: complex):
    """Inject a per-unit shunt admittance at a ppc bus.

    makeYbus rebuilds the shunt as Ysh = (GS + j*BS) / baseMVA, so the values
    stored in those columns must carry the admittance scaled by baseMVA.
    """
    ppci["bus"][bus_ppc, GS] += y_pu.real * ppci["baseMVA"]
    ppci["bus"][bus_ppc, BS] += y_pu.imag * ppci["baseMVA"]


def _z_to_y_pu(r_ohm: float, x_ohm: float, base_z_ohm: float) -> complex:
    """Per-unit admittance of an ohmic impedance: y_pu = 1 / (z_ohm / base_z)."""
    z = complex(r_ohm, x_ohm)
    if abs(z) < 1e-12:
        z = 1e-6 + 1e-6j
    return base_z_ohm / z


def _ppc_bus(net, pp_bus_idx) -> Optional[int]:
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    try:
        return int(bus_lookup[pp_bus_idx])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _ext_grid_z1_ohm(net, row) -> Tuple[float, float, float]:
    """(vn_kv, R1, X1) of an external grid: Z1 = Un^2 / S_sc, no voltage factor."""
    vn = float(net.bus.at[int(row["bus"]), "vn_kv"])
    s_sc = _f(row.get("s_sc_max_mva"), 10000.0)
    if s_sc <= 0:
        s_sc = 10000.0
    rx = _f(row.get("rx_max"), 0.1)
    z_abs = vn * vn / s_sc
    # rx_max is R/X, so X = |Z|/sqrt(1+(R/X)^2) and R = (R/X)*X.
    x = z_abs / math.sqrt(1.0 + rx * rx)
    return vn, rx * x, x


def _add_ext_grid_ansi(net, ppci, prefault_v: float):
    if net.ext_grid.empty:
        return
    for _, row in net.ext_grid[net.ext_grid.in_service].iterrows():
        bus_ppc = _ppc_bus(net, int(row["bus"]))
        if bus_ppc is None:
            continue
        vn, r, x = _ext_grid_z1_ohm(net, row)
        base_z = vn * vn / ppci["baseMVA"]
        _y_add_shunt(ppci, bus_ppc, _z_to_y_pu(r, x, base_z))


def _add_gen_ansi(net, ppci, network: str, gen_meta: Dict[int, dict]):
    if net.gen.empty:
        return
    for idx, row in net.gen[net.gen.in_service].iterrows():
        bus = int(row["bus"])
        bus_ppc = _ppc_bus(net, bus)
        if bus_ppc is None:
            continue
        vn_bus = float(net.bus.at[bus, "vn_kv"])
        vn_gen = _f(row.get("vn_kv"), vn_bus) or vn_bus
        sn = _f(row.get("sn_mva"), 100.0) or 100.0
        xdss = _f(row.get("xdss_pu"), 0.2) or 0.2
        rdss = _f(row.get("rdss_ohm"), 0.0)
        mtype = str(row.get("ansi_machine_type", "turbo") or "turbo").lower()
        if mtype not in ANSI_MACHINE_TYPES:
            mtype = "turbo"
        mult = _machine_x_multiplier(network, "gen")
        if mult <= 0:
            continue
        x_ohm = xdss * mult * (vn_gen * vn_gen) / sn
        r_ohm = rdss if rdss > 0 else 0.05 * x_ohm
        base_z = vn_bus * vn_bus / ppci["baseMVA"]
        _y_add_shunt(ppci, bus_ppc, _z_to_y_pu(r_ohm, x_ohm, base_z))
        gen_meta[idx] = {"bus_ppc": bus_ppc, "x_ohm": x_ohm, "bus": bus}


def _add_motor_ansi(net, ppci, network: str):
    if not hasattr(net, "motor") or net.motor.empty:
        return
    for _, row in net.motor[net.motor.in_service].iterrows():
        bus = int(row["bus"])
        bus_ppc = _ppc_bus(net, bus)
        if bus_ppc is None:
            continue
        pn = _f(row.get("pn_mech_mw"), 0.0)
        hp = _motor_hp(pn)
        mult = _machine_x_multiplier(network, "motor", hp)
        if mult <= 0:
            continue
        vn = _f(row.get("vn_kv"), float(net.bus.at[bus, "vn_kv"])) or float(net.bus.at[bus, "vn_kv"])
        lrc = _f(row.get("lrc_pu"), 0.0)
        if lrc <= 0:
            continue
        rx = _f(row.get("rx"), 0.0)
        cos_phi = _f(row.get("cos_phi"), 0.85) or 0.85
        eff = _f(row.get("efficiency_n_percent"), 90.0) or 90.0
        sn = pn / (cos_phi * eff / 100.0) if cos_phi > 0 and eff > 0 else pn
        if sn <= 0:
            sn = pn if pn > 0 else 1.0
        z_base = vn * vn / sn
        z_lr = z_base / lrc
        x_ohm = mult * z_lr / math.sqrt(1.0 + rx * rx)
        r_ohm = x_ohm * rx
        base_z = vn * vn / ppci["baseMVA"]
        _y_add_shunt(ppci, bus_ppc, _z_to_y_pu(r_ohm, x_ohm, base_z))


def _add_sgen_ansi(net, ppci, network: str):
    if net.sgen.empty or "generator_type" not in net.sgen.columns:
        return
    for _, row in net.sgen[net.sgen.in_service].iterrows():
        gtype = str(row.get("generator_type", "") or "").lower()
        bus = int(row["bus"])
        bus_ppc = _ppc_bus(net, bus)
        if bus_ppc is None:
            continue
        vn = float(net.bus.at[bus, "vn_kv"])
        base_z = vn * vn / ppci["baseMVA"]
        if network == NETWORK_30:
            continue
        if gtype in ("async", "async_doubly_fed"):
            if network == NETWORK_INT:
                continue
            sn = _f(row.get("sn_mva"), 0.0)
            lrc = _f(row.get("lrc_pu"), 0.0)
            if sn <= 0 or lrc <= 0:
                continue
            rx = _f(row.get("rx"), 0.0)
            z_lr = (vn * vn / sn) / lrc
            x_ohm = z_lr / math.sqrt(1.0 + rx * rx)
            r_ohm = x_ohm * rx
            _y_add_shunt(ppci, bus_ppc, _z_to_y_pu(r_ohm, x_ohm, base_z))
        elif row.get("current_source") or gtype == "current_source":
            max_ik = _f(row.get("max_ik_ka"), 0.0)
            if max_ik <= 0:
                continue
            rx = _f(row.get("rx"), 0.1) or 0.1
            z_ohm = vn / (math.sqrt(3) * max_ik)
            x_ohm = z_ohm / math.sqrt(1.0 + rx * rx)
            r_ohm = x_ohm * rx
            _y_add_shunt(ppci, bus_ppc, _z_to_y_pu(r_ohm, x_ohm, base_z))


def _add_storage_ansi(net, ppci, network: str):
    if not hasattr(net, "storage") or net.storage.empty:
        return
    if network == NETWORK_30:
        return
    for _, row in net.storage[net.storage.in_service].iterrows():
        max_ik = _f(row.get("max_ik_ka"), 0.0)
        cs = row.get("current_source")
        if not (cs in (True, "true", "True", 1, "1") or max_ik > 0):
            continue
        if max_ik <= 0:
            continue
        bus = int(row["bus"])
        bus_ppc = _ppc_bus(net, bus)
        if bus_ppc is None:
            continue
        vn = float(net.bus.at[bus, "vn_kv"])
        base_z = vn * vn / ppci["baseMVA"]
        rx = _f(row.get("rx"), 0.1) or 0.1
        z_ohm = vn / (math.sqrt(3) * max_ik)
        x_ohm = z_ohm / math.sqrt(1.0 + rx * rx)
        r_ohm = x_ohm * rx
        _y_add_shunt(ppci, bus_ppc, _z_to_y_pu(r_ohm, x_ohm, base_z))


# ---------------------------------------------------------------------------
# Zero-sequence network for line-to-ground faults
# ---------------------------------------------------------------------------

_GROUNDED_WINDINGS = ("yn", "zn")


def _finite(v) -> Optional[float]:
    """The value as a float when it is a finite number, else None."""
    x = _f(v, float("nan"))
    return x if math.isfinite(x) else None


def _windings(vector_group, count: int) -> Optional[List[str]]:
    """Winding connections of a vector group, HV first: 'YNd11' -> ['yn', 'd'].

    None when the group does not spell exactly `count` windings, so the
    transformer is left open in the zero-sequence network.
    """
    text = re.sub(r"[^a-z]", "", str(vector_group or "").lower())
    parts = re.findall(r"yn|zn|y|z|d", text)
    if len(parts) != count or "".join(parts) != text:
        return None
    return parts


def _z_percent(vk: float, vkr: float) -> complex:
    """Complex short-circuit voltage in percent from |vk| and its real part."""
    vk = abs(vk)
    vkr = min(abs(vkr), vk)
    return complex(vkr, math.sqrt(vk * vk - vkr * vkr))


def _zero_seq_vk(row, vk_col: str, vkr_col: str, vk0_col: str, vkr0_col: str) -> complex:
    """Zero-sequence vk in percent, each of vk0 / vkr0 falling back to its
    positive-sequence value when not given (zero or blank), as pandapower does."""
    vk0 = _finite(row.get(vk0_col))
    vkr0 = _finite(row.get(vkr0_col))
    vk = vk0 if vk0 is not None and vk0 > 0 else _f(row.get(vk_col), 0.0)
    vkr = vkr0 if vkr0 is not None and vkr0 > 0 else _f(row.get(vkr_col), 0.0)
    return _z_percent(vk, vkr)


def _neutral_ohm(row) -> complex:
    """Neutral grounding impedance R_N + jX_N in ohm (rn_ohm / xn_ohm)."""
    return complex(max(_f(row.get("rn_ohm"), 0.0), 0.0), max(_f(row.get("xn_ohm"), 0.0), 0.0))


def _y_pu_from_ohm(z_ohm: complex, base_kv: float, base_mva: float) -> complex:
    return _z_to_y_pu(z_ohm.real, z_ohm.imag, base_kv * base_kv / base_mva)


def _open_branch(ppci0, row: int):
    branch = ppci0["branch"]
    branch[row, BR_STATUS] = 0
    # makeYbus divides the status by R + jX under errstate(raise).
    if branch[row, BR_R] == 0 and branch[row, BR_X] == 0:
        branch[row, BR_X] = 1.0


def _set_series_branch(ppci0, row: int, z_pu: complex):
    branch = ppci0["branch"]
    branch[row, BR_R] = z_pu.real
    branch[row, BR_X] = z_pu.imag
    branch[row, [BR_G, BR_B, BR_R_ASYM, BR_X_ASYM, BR_G_ASYM, BR_B_ASYM, SHIFT]] = 0.0
    branch[row, BR_STATUS] = 1


def _zero_seq_lines(net, ppci0):
    """Lines as series branches with R0 / X0 / C0. R0 or X0 not given falls
    back to its positive-sequence value; C0 not given is zero."""
    if net.line.empty:
        return
    base_mva = ppci0["baseMVA"]
    omega = 2.0 * math.pi * _f(net.get("f_hz"), 50.0)
    for idx, row in _ppci_branch_rows(net, ppci0, "line").items():
        line = net.line.loc[idx]
        length = _f(line.get("length_km"), 1.0)
        parallel = _f(line.get("parallel"), 1.0) or 1.0
        r1 = _f(line.get("r_ohm_per_km"), 0.0)
        x1 = _f(line.get("x_ohm_per_km"), 0.0)
        r0 = _finite(line.get("r0_ohm_per_km"))
        x0 = _finite(line.get("x0_ohm_per_km"))
        r0 = r1 if r0 is None else r0
        x0 = x1 if x0 is None else x0
        if r0 == 0 and x0 == 0:
            r0, x0 = r1, x1
        c0 = _finite(line.get("c0_nf_per_km")) or 0.0
        base_kv = float(np.real(ppci0["bus"][int(np.real(ppci0["branch"][row, F_BUS])), BASE_KV]))
        base_z = base_kv * base_kv / base_mva
        _set_series_branch(ppci0, row, complex(r0, x0) * length / parallel / base_z)
        ppci0["branch"][row, BR_B] = omega * c0 * 1e-9 * length * parallel * base_z


def _zero_seq_ext_grids(net, ppci0):
    """External grids: X0 = (X0/X1) X1, R0 = (R0/X0) X0 from x0x_max / r0x0_max.
    An ungrounded source (X0/X1 very large) then adds next to nothing. X0/X1
    blank or not positive (the diagram's default is 0, which IEC rejects) is
    taken as 1, R0/X0 blank as 0.1 - ensure_ext_grid_zero_sequence_min's
    defaults."""
    if net.ext_grid.empty:
        return
    for _, row in net.ext_grid[net.ext_grid.in_service].iterrows():
        bus_ppc = _ppc_bus(net, int(row["bus"]))
        if bus_ppc is None:
            continue
        vn, _, x1 = _ext_grid_z1_ohm(net, row)
        x0x = _finite(row.get("x0x_max"))
        r0x0 = _finite(row.get("r0x0_max"))
        x0 = (x0x if x0x is not None and x0x > 0 else 1.0) * x1
        r0 = abs(0.1 if r0x0 is None else r0x0) * x0
        _y_add_shunt(ppci0, bus_ppc, _y_pu_from_ohm(complex(r0, x0), vn, ppci0["baseMVA"]))


def _zero_seq_gens(net, ppci0):
    """Generators with zero-sequence data (x0_pu, optional r0_pu, on the
    machine's own rating) as a grounded-wye source through 3 Z_N
    (rn_ohm / xn_ohm). Without x0_pu the machine is taken as ungrounded or
    behind a delta winding and adds nothing, as in pandapower."""
    if net.gen.empty or "x0_pu" not in net.gen.columns:
        return
    for _, row in net.gen[net.gen.in_service].iterrows():
        x0 = _f(row.get("x0_pu"), 0.0)
        if x0 <= 0:
            continue
        bus = int(row["bus"])
        bus_ppc = _ppc_bus(net, bus)
        if bus_ppc is None:
            continue
        vn_bus = float(net.bus.at[bus, "vn_kv"])
        vn_gen = _f(row.get("vn_kv"), vn_bus) or vn_bus
        sn = _f(row.get("sn_mva"), 100.0) or 100.0
        r0 = _f(row.get("r0_pu"), 0.0)
        z_ohm = complex(r0, x0) * vn_gen * vn_gen / sn + 3.0 * _neutral_ohm(row)
        _y_add_shunt(ppci0, bus_ppc, _y_pu_from_ohm(z_ohm, vn_bus, ppci0["baseMVA"]))


def _zero_seq_trafos(net, ppci0):
    """Two-winding transformers by vector group.

    The neutral impedance Z_N (rn_ohm / xn_ohm, ohm at the voltage of the
    grounded winding) adds 3 Z_N to that winding's zero-sequence path:
      YN-d, ZN-any        shunt Z0 + 3 Z_N at the HV bus
      D-yn, any-zn        shunt Z0 + 3 Z_N at the LV bus
      YN-yn               series Z0 + 3 Z_N (Z_N taken on the LV neutral); with
                          mag0_percent the T model with its magnetizing branch
      YN-y, Y-yn          shunt Z0 + Z_m0 at the grounded side when mag0_percent
                          is given, else open
      Y-d, D-y, D-d, Y-y  open
    Grounding transformers (electrisim_grounding) are a shunt Z0 + 3 Z_N at
    their bus from their own data.
    """
    if net.trafo.empty:
        return
    base_mva = ppci0["baseMVA"]
    grounding = net.trafo["electrisim_grounding"] == True if "electrisim_grounding" in net.trafo.columns else None
    for idx, row in _ppci_branch_rows(net, ppci0, "trafo").items():
        tr = net.trafo.loc[idx]
        _open_branch(ppci0, row)
        hv_bus, lv_bus = int(tr["hv_bus"]), int(tr["lv_bus"])
        hv_ppc, lv_ppc = _ppc_bus(net, hv_bus), _ppc_bus(net, lv_bus)
        vn_hv_bus = float(net.bus.at[hv_bus, "vn_kv"])
        vn_lv_bus = float(net.bus.at[lv_bus, "vn_kv"])
        if grounding is not None and bool(grounding.at[idx]):
            z = complex(_f(tr.get("electrisim_gt_r0"), 0.0) + 3.0 * _f(tr.get("electrisim_gt_r_n"), 0.0),
                        _f(tr.get("electrisim_gt_x0"), 0.0) + 3.0 * _f(tr.get("electrisim_gt_x_n"), 0.0))
            if hv_ppc is not None:
                _y_add_shunt(ppci0, hv_ppc, _y_pu_from_ohm(z, vn_hv_bus, base_mva))
            continue
        groups = _windings(tr.get("vector_group"), 2)
        sn = _f(tr.get("sn_mva"), 0.0)
        if groups is None or sn <= 0 or hv_ppc is None or lv_ppc is None:
            continue
        hv, lv = groups
        parallel = _f(tr.get("parallel"), 1.0) or 1.0
        z_pct = _zero_seq_vk(tr, "vk_percent", "vkr_percent", "vk0_percent", "vkr0_percent")
        z_n = 3.0 * _neutral_ohm(tr)
        mag0 = _f(tr.get("mag0_percent"), 0.0)
        mag0_rx = _f(tr.get("mag0_rx"), 0.0)

        def z0_ohm(vn_kv):
            return z_pct / 100.0 * vn_kv * vn_kv / sn

        def zm0_ohm(vn_kv):
            # As pandapower: |Z_m0| = mag0_percent x |Z0| (a ratio, despite
            # the name) at the angle of its R/X, mag0_rx.
            return abs(z0_ohm(vn_kv)) * mag0 * complex(mag0_rx, 1.0) / math.sqrt(1.0 + mag0_rx * mag0_rx)

        vn_hv_t = _f(tr.get("vn_hv_kv"), vn_hv_bus) or vn_hv_bus
        vn_lv_t = _f(tr.get("vn_lv_kv"), vn_lv_bus) or vn_lv_bus
        hv_grounded, lv_grounded = hv in _GROUNDED_WINDINGS, lv in _GROUNDED_WINDINGS
        if hv == "zn" or (hv_grounded and lv == "d"):
            z = (z0_ohm(vn_hv_t) + z_n) / parallel
            _y_add_shunt(ppci0, hv_ppc, _y_pu_from_ohm(z, vn_hv_bus, base_mva))
        elif lv == "zn" or (lv_grounded and hv == "d"):
            z = (z0_ohm(vn_lv_t) + z_n) / parallel
            _y_add_shunt(ppci0, lv_ppc, _y_pu_from_ohm(z, vn_lv_bus, base_mva))
        elif hv_grounded and lv_grounded:
            # Referred to the LV side, where pypower puts a branch's series
            # impedance (the off-nominal ratio stays in TAP).
            z0 = z0_ohm(vn_lv_t)
            base_z = vn_lv_bus * vn_lv_bus / base_mva
            if mag0 > 0:
                # pandapower's T model: Z0 split si0_hv_partial : 1 - si0 about
                # a magnetizing branch Z_m0, star-delta to a pi.
                si0 = min(max(_f(tr.get("si0_hv_partial"), 0.5), 0.0), 1.0)
                z_a, z_b, z_m = si0 * z0, (1.0 - si0) * z0 + z_n, zm0_ohm(vn_lv_t)
                z_t = z_a * z_b + z_b * z_m + z_a * z_m
                z_series = z_t / z_m
                y_hv, y_lv = z_b / z_t, z_a / z_t
            else:
                z_series, y_hv, y_lv = z0 + z_n, 0j, 0j
            _set_series_branch(ppci0, row, z_series / parallel / base_z)
            tap = float(np.real(ppci0["branch"][row, TAP])) or 1.0
            # A shunt at the internal (LV-side) node of the ideal transformer
            # appears at the HV bus divided by tap^2.
            _y_add_shunt(ppci0, hv_ppc, y_hv * parallel * base_z / (tap * tap))
            _y_add_shunt(ppci0, lv_ppc, y_lv * parallel * base_z)
        elif (hv_grounded or lv_grounded) and mag0 > 0:
            vn_t, vn_b, bus_ppc = (vn_hv_t, vn_hv_bus, hv_ppc) if hv_grounded else (vn_lv_t, vn_lv_bus, lv_ppc)
            z = (z0_ohm(vn_t) + zm0_ohm(vn_t) + z_n) / parallel
            _y_add_shunt(ppci0, bus_ppc, _y_pu_from_ohm(z, vn_b, base_mva))


def _zero_seq_trafo3w(net, ppci0):
    """Three-winding transformers: the star equivalent of vk0_*/vkr0_* about
    the internal star bus, each leg by its winding: a grounded star (yn/zn)
    joins its bus to the star point, a delta grounds the star point through
    its leg, an ungrounded star is open."""
    if not hasattr(net, "trafo3w") or net.trafo3w.empty:
        return
    base_mva = ppci0["baseMVA"]
    legs = _ppci_branch_rows(net, ppci0, "trafo3w", n_sides=3)
    for idx, rows in legs.items():
        t3 = net.trafo3w.loc[idx]
        for row in rows:
            if row is not None:
                _open_branch(ppci0, row)
        groups = _windings(t3.get("vector_group"), 3)
        if groups is None or rows[0] is None:
            continue
        sn = [_f(t3.get(f"sn_{s}_mva"), 0.0) for s in ("hv", "mv", "lv")]
        if min(sn) <= 0:
            continue
        # Pairwise HV-MV, MV-LV, HV-LV on the smaller rating, to the HV rating.
        pairs = [
            _zero_seq_vk(t3, f"vk_{s}_percent", f"vkr_{s}_percent", f"vk0_{s}_percent", f"vkr0_{s}_percent")
            for s in ("hv", "mv", "lv")
        ]
        z_hm = pairs[0] * sn[0] / min(sn[0], sn[1])
        z_ml = pairs[1] * sn[0] / min(sn[1], sn[2])
        z_hl = pairs[2] * sn[0] / min(sn[0], sn[2])
        star = (0.5 * (z_hm + z_hl - z_ml), 0.5 * (z_hm + z_ml - z_hl), 0.5 * (z_hl + z_ml - z_hm))
        vn_t = [_f(t3.get(f"vn_{s}_kv"), 0.0) for s in ("hv", "mv", "lv")]
        star_ppc = int(np.real(ppci0["branch"][rows[0], T_BUS]))
        star_kv = float(np.real(ppci0["bus"][star_ppc, BASE_KV]))
        for k, (winding, row) in enumerate(zip(groups, rows)):
            if row is None:
                continue
            if winding in _GROUNDED_WINDINGS:
                # The leg's impedance sits on its T side: the star bus for the
                # HV leg, the MV / LV bus for the others.
                t_kv = float(np.real(ppci0["bus"][int(np.real(ppci0["branch"][row, T_BUS])), BASE_KV]))
                z_ohm = star[k] / 100.0 * vn_t[k] * vn_t[k] / sn[0]
                _set_series_branch(ppci0, row, z_ohm / (t_kv * t_kv / base_mva))
            elif winding == "d":
                z_ohm = star[k] / 100.0 * vn_t[0] * vn_t[0] / sn[0]
                _y_add_shunt(ppci0, star_ppc, _y_pu_from_ohm(z_ohm, star_kv, base_mva))


def _ungrounded_buses(ppci0) -> np.ndarray:
    """Mask of buses in zero-sequence islands with no path to ground.

    Their Ybus is singular. A ground fault there draws no current (no line
    capacitance given), so they get a token shunt to make Ybus invertible and
    are reported with zero current.
    """
    n = ppci0["bus"].shape[0]
    branch = ppci0["branch"]
    live = np.real(branch[:, BR_STATUS]) > 0
    f = np.real(branch[live, F_BUS]).astype(np.int64)
    t = np.real(branch[live, T_BUS]).astype(np.int64)
    shunt = np.abs(ppci0["bus"][:, GS] + 1j * ppci0["bus"][:, BS]) > 0
    charging = np.abs(branch[live, BR_G] + 1j * branch[live, BR_B]) > 0
    shunt[f[charging]] = True
    shunt[t[charging]] = True
    graph = csr_matrix((np.ones(len(f)), (f, t)), shape=(n, n))
    n_comp, labels = connected_components(graph, directed=False)
    grounded = np.zeros(n_comp, dtype=bool)
    grounded[np.unique(labels[shunt])] = True
    return ~grounded[labels]


def _build_zero_sequence_y(net, prefault_v: float):
    """Zero-sequence Ybus / Zbus on the same bus indexing as the positive
    sequence: lines, transformers by vector group with their neutral
    impedance, external grids, grounded generators and grounding
    transformers. Motors, static generators and storage (ungrounded or
    behind a delta) add no zero-sequence path. Other branch kinds keep their
    positive-sequence impedance."""
    _, ppci0 = _init_ansi_ppc(net, prefault_v)
    _zero_seq_lines(net, ppci0)
    _zero_seq_trafos(net, ppci0)
    _zero_seq_trafo3w(net, ppci0)
    _zero_seq_ext_grids(net, ppci0)
    _zero_seq_gens(net, ppci0)
    floating = _ungrounded_buses(ppci0)
    if floating.any():
        ppci0["bus"][floating, GS] += ppci0["baseMVA"]
    _calc_ybus(ppci0)
    _calc_zbus(net, ppci0)
    ppci0["internal"]["z0_floating"] = floating
    return ppci0


def _fault_currents_ka(
    ppci,
    ppci0,
    fault: str,
    prefault_v: float,
    r_fault_ohm: float,
    x_fault_ohm: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (i_sym_ka, r_ohm, x_ohm) per ppci bus index."""
    n = ppci["bus"].shape[0]
    z_diag = np.diag(ppci["internal"]["Zbus"]).astype(np.complex128)
    base_kv = ppci["bus"][:, BASE_KV]
    base_mva = ppci["baseMVA"]
    base_z = base_kv ** 2 / base_mva
    # Base current in kA. pandapower stores the reciprocal of this as "baseI"
    # (BASE_KV * sqrt(3) / baseMVA) and divides per-unit currents by it.
    base_i = base_mva / (math.sqrt(3) * base_kv)

    i_sym = np.zeros(n, dtype=float)
    r_eq = np.zeros(n, dtype=float)
    x_eq = np.zeros(n, dtype=float)
    z0_diag = None
    floating = np.zeros(n, dtype=bool)
    if fault == "1ph" and ppci0 is not None:
        z0_diag = np.diag(ppci0["internal"]["Zbus"]).astype(np.complex128)
        floating = ppci0["internal"].get("z0_floating", floating)

    for b in range(n):
        z_f = complex(r_fault_ohm, x_fault_ohm) / base_z[b]
        z_th = z_diag[b] + z_f
        if z0_diag is not None and not floating[b]:
            # Line-to-ground: I = 3 E / |Z1 + Z2 + Z0 + 3 Zf| with Z2 = Z1. The
            # equivalent per-phase impedance (2 Z1 + Z0) / 3 + Zf gives that
            # current as E / |Z| and the C37.010 ground-fault X/R,
            # (2 X1 + X0) / (2 R1 + R0).
            z_th = (2.0 * z_diag[b] + z0_diag[b]) / 3.0 + z_f
        r_eq[b] = z_th.real * base_z[b]
        x_eq[b] = z_th.imag * base_z[b]
        z_abs = abs(z_th)
        if z_abs < 1e-12:
            z_abs = 1e-12
        v = prefault_v
        if fault == "2ph":
            # Line-to-line fault: I2 = sqrt(3)/2 * I3ph in this per-unit system,
            # since I2 = Un/|Z1+Z2| on the line-to-line voltage.
            i_pu = math.sqrt(3.0) * v / (2.0 * z_abs)
        elif fault == "1ph" and floating[b]:
            # No zero-sequence path to ground: no ground-fault current. X/R
            # stays the positive sequence's.
            i_pu = 0.0
        else:
            i_pu = v / z_abs
        i_sym[b] = abs(i_pu) * base_i[b]
    return i_sym, r_eq, x_eq


def _branch_currents_ka(
    ppci,
    fault: str,
    prefault_v: float,
    r_fault_ohm: float,
    x_fault_ohm: float,
    fault_ppc_indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Worst-case branch end currents in kA, per ppci branch row.

    Superposition on the Zbus: a bolted fault at bus f draws I_f = V0 / Z_ff and
    leaves the bus voltages at V = V0 - Zbus[:, f] * I_f. The branch end currents
    then follow from the Yf / Yt incidence admittance matrices, which already
    carry transformer tap ratios and phase shifts. Each branch keeps the largest
    magnitude over all fault locations, matching the convention pandapower uses
    for branch_results with return_all_currents=False.

    Only the positive-sequence network is used, so this is meaningful for 3ph and
    2ph faults. Single-phase branch flows would need the full zero-sequence
    branch model, which this module only approximates, so the caller skips them.
    """
    branch = ppci["branch"]
    n_br = branch.shape[0]
    if n_br == 0:
        return np.zeros(0), np.zeros(0)

    zbus = ppci["internal"]["Zbus"]
    yf = ppci["internal"]["Yf"]
    yt = ppci["internal"]["Yt"]
    n_bus = ppci["bus"].shape[0]
    base_kv = ppci["bus"][:, BASE_KV]
    base_mva = ppci["baseMVA"]
    base_z = base_kv ** 2 / base_mva
    base_i = base_mva / (math.sqrt(3) * base_kv)

    fb = np.real(branch[:, 0]).astype(np.int64)
    tb = np.real(branch[:, 1]).astype(np.int64)
    z_extra = complex(r_fault_ohm, x_fault_ohm)
    # Same driving-voltage basis as _fault_currents_ka so bus and branch duties
    # stay comparable.
    scale = math.sqrt(3.0) / 2.0 if fault == "2ph" else 1.0

    max_f = np.zeros(n_br)
    max_t = np.zeros(n_br)
    v0 = np.full(n_bus, prefault_v, dtype=np.complex128)

    if fault_ppc_indices is None:
        fault_iter = range(n_bus)
    else:
        fault_iter = [int(i) for i in fault_ppc_indices if 0 <= int(i) < n_bus]

    for f in fault_iter:
        z_ff = zbus[f, f]
        if z_extra != 0:
            z_ff = z_ff + z_extra / base_z[f]
        if abs(z_ff) < 1e-12:
            continue
        i_f = (prefault_v * scale) / z_ff
        v = v0 - zbus[:, f] * i_f
        np.maximum(max_f, np.abs(yf.dot(v)) * base_i[fb], out=max_f)
        np.maximum(max_t, np.abs(yt.dot(v)) * base_i[tb], out=max_t)
    return max_f, max_t


def _ppci_branch_rows(net, ppci, element: str, n_sides: int = 1) -> Dict[int, Any]:
    """Map pandapower element index -> ppci branch row for a branch element.

    The pd2ppc branch lookup gives a contiguous [start, stop) range of *ppc*
    rows, while ppci keeps only the in-service ones. branch_is is the mask that
    relates the two.

    With n_sides = 3 (trafo3w: its HV legs, then its MV legs, then its LV
    legs) each element maps to a tuple of its three rows, None where a leg
    is not in ppci.
    """
    try:
        start, stop = net["_pd2ppc_lookups"]["branch"][element]
    except (KeyError, TypeError, ValueError):
        return {}
    table = net.get(element)
    if table is None or table.empty:
        return {}

    branch_is = ppci["internal"].get("branch_is")
    if branch_is is None or len(branch_is) == 0:
        ppc_to_ppci = {i: i for i in range(int(start), int(stop))}
    else:
        ppci_row = np.cumsum(branch_is) - 1
        ppc_to_ppci = {
            i: int(ppci_row[i])
            for i in range(int(start), min(int(stop), len(branch_is)))
            if branch_is[i]
        }

    out: Dict[int, Any] = {}
    n = len(table.index)
    for pos, elm_idx in enumerate(table.index):
        rows = [ppc_to_ppci.get(int(start) + side * n + pos) for side in range(n_sides)]
        rows = [r if r is not None and r >= 0 else None for r in rows]
        if n_sides == 1:
            if rows[0] is not None:
                out[int(elm_idx)] = rows[0]
        elif any(r is not None for r in rows):
            out[int(elm_idx)] = tuple(rows)
    return out


def _is_grounding(table, idx) -> bool:
    """A grounding transformer's own row: its transformer, or the delta bus
    behind it. Like the IEC study, the results leave both out."""
    return ("electrisim_grounding" in table.columns and idx in table.index
            and table.at[idx, "electrisim_grounding"] == True)


def _xr_ratio(r_ohm: float, x_ohm: float) -> float:
    r = max(abs(r_ohm), 1e-6)
    x = max(abs(x_ohm), 1e-6)
    return x / r


def _resolve_device_class(vn_kv: float, switch_row: pd.Series) -> str:
    dc = str(switch_row.get("ansi_device_class", "auto") or "auto").lower()
    if switch_row.get("generator_cb") in (True, "true", "True", 1, "1"):
        return "generator_c37_013"
    if dc != "auto":
        return dc
    if vn_kv < 1.0:
        sw_type = str(switch_row.get("type", "CB") or "CB").upper()
        if sw_type in ("CB", "LS", "LBS"):
            return "lv_c37_013"
        return "mccb_ul_489"
    return "hv_c37_010"


def _compute_device_duties(
    net,
    bus_results: Dict[int, dict],
    freq_hz: float,
    cp_cycles: float,
) -> List[dict]:
    duties = []
    if not hasattr(net, "switch") or net.switch.empty:
        return duties

    for idx, sw in net.switch.iterrows():
        if not sw.get("closed", True):
            continue
        bus = int(sw["bus"])
        br = bus_results.get(bus)
        if not br:
            continue
        vn = float(net.bus.at[bus, "vn_kv"])
        dc = _resolve_device_class(vn, sw)
        int_rating = _f(sw.get("interrupting_rating_ka"), 0.0)
        mom_rating = _f(sw.get("momentary_rating_ka"), 0.0)
        if int_rating <= 0:
            int_rating = _f(sw.get("in_ka"), 0.0)
        if mom_rating <= 0:
            mom_rating = int_rating

        xr_fc = br.get("xr_first", 15.0) or 15.0
        xr_int = br.get("xr_interrupting", xr_fc) or xr_fc
        i_fc_sym = br.get("i_first_sym_ka", 0.0) or 0.0
        i_fc_peak = br.get("i_first_peak_ka", 0.0) or 0.0
        i_int = br.get("i_interrupting_ka", 0.0) or 0.0
        i_30 = br.get("i_steady_ka", 0.0) or 0.0

        duty_int = i_int
        duty_mom = i_fc_peak
        std = "IEEE C37.010"
        if dc == "lv_c37_013":
            std = "IEEE C37.13"
            duty_int = i_fc_sym * _lv_extra_multiplier(xr_fc, dc, freq_hz)
            duty_mom = i_fc_peak
        elif dc == "mccb_ul_489":
            std = "UL 489"
            duty_int = i_fc_sym * _lv_extra_multiplier(xr_fc, dc, freq_hz)
            duty_mom = i_fc_peak
        elif dc == "generator_c37_013":
            std = "IEEE C37.013"
            duty_int = i_int
            duty_mom = i_fc_peak

        int_pass = int_rating <= 0 or duty_int <= int_rating * 1.0001
        mom_pass = mom_rating <= 0 or duty_mom <= mom_rating * 1.0001
        margin_int = (int_rating - duty_int) if int_rating > 0 else None
        margin_mom = (mom_rating - duty_mom) if mom_rating > 0 else None

        sw_id = sw.get("id", sw.get("name", str(idx)))
        duties.append({
            "name": _clean(sw.get("name", str(idx))),
            "id": _clean(sw_id),
            "bus": _clean(net.bus.at[bus, "name"] if bus in net.bus.index else bus),
            "vn_kv": _clean(vn),
            "device_class": dc,
            "standard": std,
            "duty_interrupting_ka": _clean(duty_int),
            "duty_momentary_ka": _clean(duty_mom),
            "duty_steady_ka": _clean(i_30),
            "interrupting_rating_ka": _clean(int_rating) if int_rating > 0 else None,
            "momentary_rating_ka": _clean(mom_rating) if mom_rating > 0 else None,
            "interrupting_pass": int_pass if int_rating > 0 else None,
            "momentary_pass": mom_pass if mom_rating > 0 else None,
            "margin_interrupting_ka": _clean(margin_int),
            "margin_momentary_ka": _clean(margin_mom),
            "xr_interrupting": _clean(xr_int),
            "contact_parting_cycles": _clean(_f(sw.get("contact_parting_cycles"), cp_cycles)),
        })
    return duties


def _solve_network(
    net,
    network: str,
    fault: str,
    prefault_v: float,
    r_fault_ohm: float,
    x_fault_ohm: float,
    gen_meta: Dict[int, dict],
    fault_pp_buses: Optional[List[int]] = None,
) -> Dict[str, dict]:
    """Solve one ANSI network.

    Returns {"bus": {pp_bus: (i_sym_ka, r_ohm, x_ohm)},
             "line": {pp_line: (i_from_ka, i_to_ka)},
             "trafo": {pp_trafo: (i_hv_ka, i_lv_ka)}}.
    """
    work = copy.deepcopy(net)
    if hasattr(work, "gen") and not work.gen.empty and "ansi_machine_type" not in work.gen.columns:
        work.gen["ansi_machine_type"] = "turbo"
    ppc, ppci = _init_ansi_ppc(work, prefault_v)
    _add_ext_grid_ansi(work, ppci, prefault_v)
    _add_gen_ansi(work, ppci, network, gen_meta)
    _add_motor_ansi(work, ppci, network)
    _add_sgen_ansi(work, ppci, network)
    _add_storage_ansi(work, ppci, network)
    _calc_ybus(ppci)
    _calc_zbus(work, ppci)
    ppci0 = _build_zero_sequence_y(work, prefault_v) if fault == "1ph" else None
    i_sym, r_eq, x_eq = _fault_currents_ka(ppci, ppci0, fault, prefault_v, r_fault_ohm, x_fault_ohm)

    buses: Dict[int, Tuple[float, float, float]] = {}
    for pp_bus in work.bus.index:
        ppc_b = _ppc_bus(work, int(pp_bus))
        if ppc_b is None or ppc_b >= len(i_sym):
            continue
        buses[int(pp_bus)] = (float(i_sym[ppc_b]), float(r_eq[ppc_b]), float(x_eq[ppc_b]))

    lines: Dict[int, Tuple[float, float]] = {}
    trafos: Dict[int, Tuple[float, float]] = {}
    if fault != "1ph":
        fault_ppc = None
        if fault_pp_buses is not None:
            fault_ppc = []
            for b in fault_pp_buses:
                ppc_b = _ppc_bus(work, int(b))
                if ppc_b is not None:
                    fault_ppc.append(ppc_b)
        i_f, i_t = _branch_currents_ka(
            ppci, fault, prefault_v, r_fault_ohm, x_fault_ohm, fault_ppc_indices=fault_ppc
        )
        for element, target in (("line", lines), ("trafo", trafos)):
            for elm_idx, row in _ppci_branch_rows(work, ppci, element).items():
                if row < len(i_f):
                    target[elm_idx] = (float(i_f[row]), float(i_t[row]))

    return {"bus": buses, "line": lines, "trafo": trafos}


def _branch_result_rows(
    net,
    element: str,
    bus_cols: Tuple[str, str],
    end_keys: Tuple[str, str],
    sol_first: Dict[str, dict],
    sol_int: Dict[str, dict],
    sol_30: Dict[str, dict],
    bus_results: Dict[int, dict],
) -> List[dict]:
    """Assemble per-branch ANSI duties for one branch element table.

    Each branch reports the larger of its two end currents for the first-cycle,
    interrupting and 30-cycle networks. Asymmetry multipliers are taken from the
    worse of the branch's two terminal buses, since the worst-case end current
    can come from a fault at either end.
    """
    table = net.get(element)
    if table is None or table.empty:
        return []

    first = sol_first.get(element, {})
    interrupting = sol_int.get(element, {})
    steady = sol_30.get(element, {})
    if not first:
        return []

    rows: List[dict] = []
    for idx in table.index:
        i = int(idx)
        if i not in first or _is_grounding(table, idx):
            continue
        i_f, i_t = first[i]
        i_first = max(i_f, i_t)
        i_int_sym = max(interrupting.get(i, (0.0, 0.0)))
        i_steady = max(steady.get(i, (0.0, 0.0)))

        mf_peak = 0.0
        mf_int = 0.0
        for col in bus_cols:
            try:
                b = int(table.at[idx, col])
            except (KeyError, TypeError, ValueError):
                continue
            br = bus_results.get(b)
            if not br:
                continue
            mf_peak = max(mf_peak, _f(br.get("mf_peak"), 0.0))
            mf_int = max(mf_int, _f(br.get("mf_interrupting"), 0.0))

        internal_name = table.at[idx, "name"] if "name" in table.columns else str(idx)
        cell_id = table.at[idx, "id"] if "id" in table.columns else internal_name
        rows.append({
            "name": _clean(internal_name),
            "id": _clean(cell_id),
            "userFriendlyName": _display_name(net, internal_name, cell_id),
            end_keys[0]: _clean(i_f),
            end_keys[1]: _clean(i_t),
            "i_first_sym_ka": _clean(i_first),
            "i_first_peak_ka": _clean(i_first * mf_peak if mf_peak > 0 else i_first),
            "i_interrupting_ka": _clean(i_int_sym * mf_int if mf_int > 0 else i_int_sym),
            "i_steady_ka": _clean(i_steady),
        })
    return rows


def shortcircuit_ansi(net, in_data, in_data_full=None) -> str:
    """Run ANSI/IEEE C37 short-circuit study; returns compact JSON string."""
    fault = str(in_data.get("fault_type", in_data.get("fault", "3ph")) or "3ph").lower()
    if fault not in ("3ph", "2ph", "1ph"):
        fault = "3ph"
    freq_hz = _f(in_data.get("frequency_hz", in_data.get("frequency", 60)), 60.0)
    if freq_hz <= 0:
        freq_hz = 60.0
    prefault_v = _f(in_data.get("prefault_v_pu", 1.0), 1.0)
    if prefault_v <= 0:
        prefault_v = 1.0
    cp_cycles = _f(in_data.get("contact_parting_cycles", 3), 3.0)
    r_fault = _f(in_data.get("r_fault_ohm", 0), 0.0)
    x_fault = _f(in_data.get("x_fault_ohm", 0), 0.0)
    fault_bus_mode = normalize_fault_bus_mode(in_data)
    fault_pp_buses = resolve_pp_fault_bus_indices(net, in_data)

    gen_meta: Dict[int, dict] = {}
    sol_first = _solve_network(net, NETWORK_FIRST, fault, prefault_v, r_fault, x_fault, gen_meta, fault_pp_buses)
    sol_int = _solve_network(net, NETWORK_INT, fault, prefault_v, r_fault, x_fault, gen_meta, fault_pp_buses)
    sol_30 = _solve_network(net, NETWORK_30, fault, prefault_v, r_fault, x_fault, gen_meta, fault_pp_buses)
    rx_first, rx_int, rx_30 = sol_first["bus"], sol_int["bus"], sol_30["bus"]

    busbar_list = []
    bus_results: Dict[int, dict] = {}

    for bus_idx in net.bus[net.bus.in_service].index:
        bi = int(bus_idx)
        if fault_pp_buses is not None and bi not in fault_pp_buses:
            continue
        if _is_grounding(net.bus, bus_idx):
            continue   # a grounding transformer's delta
        if bi not in rx_first:
            continue
        i_fc_sym, r_ohm, x_ohm = rx_first[bi]
        i_fc_int_sym = rx_int.get(bi, (0.0, 0.0, 0.0))[0]
        i_steady = rx_30.get(bi, (0.0, 0.0, 0.0))[0]
        r_i, x_i = rx_int.get(bi, (r_ohm, x_ohm))[1], rx_int.get(bi, (r_ohm, x_ohm))[2]

        xr_fc = _xr_ratio(r_ohm, x_ohm)
        xr_int = _xr_ratio(r_i, x_i)

        mf_peak = _peak_multiplier_remote(xr_fc, freq_hz)
        mf_int = _interrupting_multiplier_remote(xr_int, cp_cycles, freq_hz)
        s_factor = _symmetry_factor_s(cp_cycles, freq_hz)
        # A symmetrically rated breaker already withstands the X/R = 17 test
        # asymmetry, so only the excess counts; never credit below 1.0.
        mf_int_symm = max(1.0, mf_int / s_factor) if s_factor > 0 else mf_int

        i_fc_peak = i_fc_sym * mf_peak
        i_interrupting = i_fc_int_sym * mf_int_symm

        vn = float(net.bus.at[bus_idx, "vn_kv"])
        br = {
            "i_first_sym_ka": i_fc_sym,
            "i_first_peak_ka": i_fc_peak,
            "i_interrupting_ka": i_interrupting,
            "i_steady_ka": i_steady,
            "xr_first": xr_fc,
            "xr_interrupting": xr_int,
            "mf_peak": mf_peak,
            "mf_interrupting": mf_int_symm,
        }
        bus_results[int(bus_idx)] = br

        internal_name = net.bus.at[bus_idx, "name"]
        cell_id = net.bus.at[bus_idx, "id"] if "id" in net.bus.columns else internal_name
        busbar_list.append({
            "name": _clean(internal_name),
            "id": _clean(cell_id),
            "userFriendlyName": _display_name(net, internal_name, cell_id),
            "vn_kv": _clean(vn),
            "i_first_sym_ka": _clean(i_fc_sym),
            "i_first_peak_ka": _clean(i_fc_peak),
            "i_interrupting_ka": _clean(i_interrupting),
            "i_steady_ka": _clean(i_steady),
            "xr_first": _clean(xr_fc),
            "xr_interrupting": _clean(xr_int),
            "rk_ohm": _clean(r_ohm),
            "xk_ohm": _clean(x_ohm),
            "mf_peak": _clean(mf_peak),
            "mf_interrupting": _clean(mf_int_symm),
        })

    device_duties = _compute_device_duties(net, bus_results, freq_hz, cp_cycles)

    lines_sc = _branch_result_rows(
        net, "line", ("from_bus", "to_bus"), ("i_from_ka", "i_to_ka"),
        sol_first, sol_int, sol_30, bus_results)
    trafos_sc = _branch_result_rows(
        net, "trafo", ("hv_bus", "lv_bus"), ("i_hv_ka", "i_lv_ka"),
        sol_first, sol_int, sol_30, bus_results)

    result = {
        "standard": "ANSI/IEEE C37",
        "beta": True,
        "fault_type": fault,
        "frequency_hz": freq_hz,
        "prefault_v_pu": prefault_v,
        "contact_parting_cycles": cp_cycles,
        "disclaimer": (
            "ANSI/IEEE C37 is in beta. Engineering implementation for North American "
            "studies — verify against utility requirements before using for equipment "
            "ratings. Do not mix with IEC 60909-rated equipment without verification."
        ),
        "busbars": busbar_list,
        "device_duties": device_duties,
        "lines_sc": lines_sc,
        "trafos_sc": trafos_sc,
        "study": "shortcircuit",
        "engine": "ansi",
        "study_params": {
            "fault_type": fault,
            "fault_bus_mode": fault_bus_mode,
            "fault_bus_ids": collect_fault_bus_refs(in_data) if fault_bus_mode == "selection" else [],
            "fault_bus_names": list(in_data.get("fault_bus_names") or []) if fault_bus_mode == "selection" else [],
            "frequency_hz": freq_hz,
            "prefault_v_pu": prefault_v,
            "contact_parting_cycles": cp_cycles,
            "standard": "ansi_c37",
        },
    }
    return json.dumps(result, separators=(",", ":"))


def _parse_id_list(text) -> set:
    if not text:
        return set()
    if isinstance(text, list):
        return {str(x).strip() for x in text if str(x).strip()}
    return {p.strip() for p in str(text).replace(";", ",").split(",") if p.strip()}


def _set_project_out_of_service(net, element_ids) -> None:
    """Disable loads/gens/sgen/storage matching diagram ids (pre-project case)."""
    want = _parse_id_list(element_ids)
    if not want:
        return
    for attr in ("load", "gen", "sgen", "storage"):
        if not hasattr(net, attr):
            continue
        df = getattr(net, attr)
        if df.empty:
            continue
        for idx in df.index:
            lid = str(df.loc[idx, "id"]) if "id" in df.columns else ""
            name = str(df.loc[idx, "name"])
            if lid in want or name in want:
                df.loc[idx, "in_service"] = False


def _merge_pre_post_ansi(pre: dict, post: dict) -> dict:
    pre_by = {str(b.get("id") or b.get("name")): b for b in pre.get("busbars", [])}
    comparison = []
    for bus in post.get("busbars", []):
        key = str(bus.get("id") or bus.get("name"))
        pb = pre_by.get(key, {})
        row = {
            "name": bus.get("userFriendlyName") or bus.get("name"),
            "id": bus.get("id"),
            "vn_kv": bus.get("vn_kv"),
            "fault_type": post.get("fault_type"),
            "i_first_sym_pre_ka": pb.get("i_first_sym_ka"),
            "i_first_sym_post_ka": bus.get("i_first_sym_ka"),
            "i_first_peak_pre_ka": pb.get("i_first_peak_ka"),
            "i_first_peak_post_ka": bus.get("i_first_peak_ka"),
            "i_interrupting_pre_ka": pb.get("i_interrupting_ka"),
            "i_interrupting_post_ka": bus.get("i_interrupting_ka"),
            "slg_note": "Use i_first_sym for 1ph SLG / NGR sizing at POI" if post.get("fault_type") == "1ph" else "",
        }
        comparison.append(row)

    duties_pre = {d.get("name"): d for d in pre.get("device_duties", [])}
    duty_comparison = []
    for d in post.get("device_duties", []):
        pd = duties_pre.get(d.get("name"), {})
        duty_comparison.append({**d, "pre_duty_interrupting_ka": pd.get("duty_interrupting_ka"), "pre_interrupting_pass": pd.get("interrupting_pass")})

    merged = dict(post)
    merged["pre_post_comparison"] = True
    merged["bus_comparison"] = comparison
    merged["device_duties"] = duty_comparison
    merged["pre_project"] = {"busbars": pre.get("busbars", []), "device_duties": pre.get("device_duties", [])}
    return merged


def shortcircuit_with_optional_pre_post(net, in_data, in_data_full=None) -> str:
    # As the IEC study does: a bus no source supplies made Ybus singular
    # ("singular matrix"), or was solved from local machines alone.
    from pandapower_electrisim import isolated_buses_message

    isolated = isolated_buses_message(net, machines=True)
    if isolated:
        return json.dumps({"error": True, "message": isolated})
    compare = str(in_data.get("compare_pre_post", "false")).lower() in ("true", "1", "yes")
    project_ids = in_data.get("project_element_ids", "")
    if compare and _parse_id_list(project_ids):
        from copy import deepcopy

        net_pre = deepcopy(net)
        _set_project_out_of_service(net_pre, project_ids)
        pre = json.loads(shortcircuit_ansi(net_pre, {**in_data, "compare_pre_post": "false"}, in_data_full))
        post = json.loads(shortcircuit_ansi(net, {**in_data, "compare_pre_post": "false"}, in_data_full))
        return json.dumps(_merge_pre_post_ansi(pre, post), separators=(",", ":"))
    return shortcircuit_ansi(net, in_data, in_data_full)


def shortcircuit(net, in_data, in_data_full=None) -> str:
    """Alias used by app.py routing."""
    return shortcircuit_with_optional_pre_post(net, in_data, in_data_full)
