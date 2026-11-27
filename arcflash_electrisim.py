# -*- coding: utf-8 -*-
"""
Arc Flash analysis for Electrisim (IEEE 1584-2018).

Runs a 3-phase maximum short-circuit study with pandapower, then applies the
IEEE 1584-2018 empirical model (via arcflash-calc) at each bus. Voltages above
15 kV use the Ralph Lee method and are flagged in the response.
"""
from __future__ import annotations

import copy
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandapower as pp
import pandapower.shortcircuit as sc

import sc_current_sources
sc_current_sources.install()     # IEC 60909's single-phase share of a current source
try:
    from arcflash.ieee_1584.cubicle import Cubicle
    from arcflash.ieee_1584.calculation import Calculation
    from arcflash.ieee_1584.units import kV, kA, mm, ms, cal_per_sq_cm
    _HAS_ARCFLASH = True
except ImportError:
    Cubicle = None  # type: ignore
    Calculation = None  # type: ignore
    _HAS_ARCFLASH = False


VALID_ELECTRODE_CONFIGS = ("VCB", "VCBB", "HCB", "VOA", "HOA")

# Typical equipment by bus voltage: IEEE 1584-2018 Table 8 (bus gap,
# enclosure H x W x D) and the customary working distances. One set of LV
# panel values (25 mm gap at 455 mm) was applied to every bus, 10 kV
# switchgear included.
_EQUIPMENT_BY_VOLTAGE = (
    # (up to kV, class, gap mm, working distance mm, enclosure H, W, D mm)
    (0.6, "LV switchgear", 32.0, 610.0, (508.0, 508.0, 508.0)),
    (5.0, "5 kV switchgear", 104.0, 910.0, (914.4, 914.4, 914.4)),
    (15.0, "15 kV switchgear", 152.0, 910.0, (1143.0, 762.0, 762.0)),
)
# Above 15 kV only Ralph Lee applies, which needs a working distance alone.
_ABOVE_15KV = ("Above 15 kV", None, 910.0, (None, None, None))

_J_PER_CAL = 4.184

# Clearing by the relays: a breaker's opening time after its relay trips (an LV air
# circuit breaker's 3 cycles at 60 Hz), and IEEE 1584-2018's 2 s - by then a worker
# has moved away from the arc - for an arc nothing clears, or clears later.
_BREAKER_OPENING_S = 0.05
_MAX_CLEARING_S = 2.0


def _ppe_category(ie_cal: float) -> str:
    """Map incident energy (cal/cm²) to NFPA 70E PPE category label."""
    if ie_cal is None or (isinstance(ie_cal, float) and math.isnan(ie_cal)):
        return "N/A"
    if ie_cal < 1.2:
        return "0"
    if ie_cal < 4.0:
        return "1"
    if ie_cal < 8.0:
        return "2"
    if ie_cal < 25.0:
        return "3"
    if ie_cal < 40.0:
        return "4"
    return "Dangerous"


def _bus_display_name(net, bus_row, bus_idx) -> str:
    """Prefer Electrisim dialog / user-friendly name over mxCell object id."""
    internal = bus_row["name"] if "name" in bus_row.index and bus_row["name"] is not None else str(bus_idx)
    internal_s = str(internal)
    uf = getattr(net, "user_friendly_names", None) or {}
    for key in (
        internal_s,
        internal_s.replace("#", "_"),
        internal_s.replace("_", "#"),
    ):
        friendly = uf.get(key)
        if friendly not in (None, ""):
            return str(friendly)
    return internal_s


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


def _ralph_lee(ibf_ka: float, voc_kv: float, working_distance_mm: float, clearing_time_s: float) -> Tuple[float, float]:
    """
    Ralph Lee method for systems above 15 kV (IEEE 1584-2002, eq. 8):
    E (J/cm²) = 2.142e6 * V(kV) * Ibf(kA) * t(s) / D(mm)^2.

    Returns (E in cal/cm², arc-flash boundary in mm), the boundary being
    where E falls to 5.0 J/cm² (1.2 cal/cm²). E was reported in J/cm² as
    if cal/cm² (4.18 times too high), and the boundary solved for 1.2 J/cm².
    """
    if working_distance_mm <= 0 or clearing_time_s <= 0 or ibf_ka <= 0 or voc_kv <= 0:
        return float("nan"), float("nan")
    k = 2.142e6 * voc_kv * ibf_ka * clearing_time_s
    ie = k / (working_distance_mm ** 2) / _J_PER_CAL
    afb = math.sqrt(k / 5.0)
    return ie, afb


def _equipment_for(voc_kv: float):
    """(class, gap mm, working distance mm, (H, W, D) mm) typical for the bus voltage."""
    for top_kv, *equipment in _EQUIPMENT_BY_VOLTAGE:
        if voc_kv <= top_kv:
            return tuple(equipment)
    return _ABOVE_15KV


def _ieee1584_bus(
    ibf_ka: float,
    voc_kv: float,
    electrode: str,
    gap_mm: float,
    working_distance_mm: float,
    height_mm: float,
    width_mm: float,
    depth_mm: float,
    clearing_time_s: float,
    clearing_time_min_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute IEEE 1584-2018 results for one bus. Returns dict of result fields."""
    if not _HAS_ARCFLASH:
        raise RuntimeError(
            "arcflash-calc is not installed. Install with: pip install arcflash-calc"
        )

    t_full_ms = clearing_time_s * 1000.0
    t_min_ms = (clearing_time_min_s if clearing_time_min_s is not None else clearing_time_s) * 1000.0

    cubicle = Cubicle(
        V_oc=voc_kv * kV,
        EC=electrode,
        G=gap_mm * mm,
        D=working_distance_mm * mm,
        height=height_mm * mm,
        width=width_mm * mm,
        depth=depth_mm * mm,
    )

    full = Calculation(cubicle, ibf_ka * kA, "full")
    full.calculate_I_arc()
    full.calculate_E_AFB(t_full_ms * ms)

    reduced = Calculation(cubicle, ibf_ka * kA, "reduced")
    reduced.calculate_I_arc()
    reduced.calculate_E_AFB(t_min_ms * ms)

    e_full = float(full.E.to(cal_per_sq_cm).magnitude)
    e_red = float(reduced.E.to(cal_per_sq_cm).magnitude)
    afb_full = float(full.AFB.magnitude)
    afb_red = float(reduced.AFB.magnitude)

    # Worst-case (max) incident energy and corresponding AFB / arcing current
    if e_red >= e_full:
        ie = e_red
        afb = afb_red
        ia = float(reduced.I_arc.magnitude)
        t_used = t_min_ms / 1000.0
    else:
        ie = e_full
        afb = afb_full
        ia = float(full.I_arc.magnitude)
        t_used = t_full_ms / 1000.0

    return {
        "ia_ka": ia,
        "ia_full_ka": float(full.I_arc.magnitude),
        "ia_min_ka": float(reduced.I_arc.magnitude),
        "incident_energy_cal_cm2": ie,
        "arc_flash_boundary_mm": afb,
        "ppe_category": _ppe_category(ie),
        "method": "IEEE1584-2018",
        "clearing_time_used_s": t_used,
        "cf": float(cubicle.CF.magnitude) if cubicle.CF is not None else None,
    }


def _clamped_ibf(ibf: float, voc_kv: float) -> Tuple[float, Optional[str]]:
    """The bolted current inside the IEEE 1584-2018 model's range, and why it was moved."""
    lo, hi, kind = (0.5, 106.0, "LV") if voc_kv <= 0.6 else (0.2, 65.0, "MV")
    if ibf < lo:
        return lo, f"Ibf={ibf:.3f} kA below {kind} model min {lo:g} kA; clamped."
    if ibf > hi:
        return hi, f"Ibf={ibf:.3f} kA above {kind} model max {hi:g} kA; clamped."
    return ibf, None


def _relay_clearing_times(net, buses, in_data, in_data_full, opening_s, max_s):
    """
    bus_idx -> (t_full_s, t_min_s, info): the relays' clearing time at the bus's
    full and reduced arcing currents (bolted, for Ralph Lee), plus the breaker's
    opening, capped at max_s. buses: [(bus_idx, ibf_ka, i_arc_full_ka, i_arc_min_ka)].
    """
    from pandapower_electrisim import protection_arc_clearing_times

    prot_params = {k: in_data.get(k, d) for k, d in (
        ('grading_mode', 'auto'), ('curve_type', 'standard_inverse'), ('overload_factor', 1.25),
        ('ct_current_factor', 1.2), ('safety_factor', 1.0), ('t_diff', 0.3), ('t_g', 0.5),
        ('t_gg', 0.07), ('tms', 1.0), ('t_grade', 0.5))}
    faults = [(b, i / ibf if i else None) for b, ibf, i_full, i_min in buses for i in (i_full, i_min)]
    found = protection_arc_clearing_times(net, in_data_full or {}, prot_params, faults)
    out = {}
    for n, (b, _, _, _) in enumerate(buses):
        full, red = found[2 * n], found[2 * n + 1]
        times = []
        for r in (full, red):
            t = r.get("t_relay_s")
            times.append(min(t + opening_s, max_s) if t is not None else max_s)
        out[b] = (times[0], times[1], {
            "relay_time_s": full.get("t_relay_s"),
            "relay_time_min_s": red.get("t_relay_s"),
            "clearing_relays": full.get("clearing_switches") or [],
            "tripped_full": full.get("tripped") or [],
            "tripped_min": red.get("tripped") or [],
            "relay_fault_ka": full.get("ikss_ka"),
            "relay_fault_min_ka": red.get("ikss_ka"),
            "inverter_sources": full.get("inverters_on_bus") or [],
            "grid_on_bus": bool(full.get("grid_on_bus")),
            "inverters_blocked": bool(full.get("inverters_blocked")),
            "error": full.get("error") or red.get("error"),
        })
    return out


def arcflash(net, in_data, in_data_full=None):
    """
    Run short-circuit then IEEE 1584-2018 (or Ralph Lee) per bus.

    Expected keys in in_data:
      electrode_config: VCB|VCBB|HCB|VOA|HOA (default VCB)
      working_distance_mm (default 455)
      conductor_gap_mm (default 25)
      enclosure_height_mm / enclosure_width_mm / enclosure_depth_mm (default 508)
      clearing_time_s (default 0.2)
      clearing_time_min_s (optional; defaults to clearing_time_s)
      clearing_time_mode: fixed (default) - the times above at every bus - or
        relays: each bus's arc cleared by the diagram's relays at its full and
        reduced arcing currents, plus breaker_opening_s (default 0.05), and
        at most max_clearing_time_s (default 2, IEEE 1584-2018's limit)
      equipment_mode: by_voltage (default) - typical gap, enclosure and working
        distance for each bus voltage (IEEE 1584-2018 Table 8) - or uniform,
        the values above for every bus
      lv_tol_percent: 6 (default) or 10 - the LV voltage tolerance that sets
        c max (1.05 or 1.10) for the bolted fault currents, as in the
        short-circuit study
    """
    if not _HAS_ARCFLASH:
        return json.dumps({
            "error": True,
            "message": "Arc flash library not available on the server. Install arcflash-calc.",
            "exception": "ImportError: arcflash-calc",
        })

    electrode = str(in_data.get("electrode_config", "VCB") or "VCB").upper().strip()
    if electrode not in VALID_ELECTRODE_CONFIGS:
        electrode = "VCB"

    working_distance_mm = float(in_data.get("working_distance_mm", 455) or 455)
    conductor_gap_mm = float(in_data.get("conductor_gap_mm", 25) or 25)
    height_mm = float(in_data.get("enclosure_height_mm", 508) or 508)
    width_mm = float(in_data.get("enclosure_width_mm", 508) or 508)
    depth_mm = float(in_data.get("enclosure_depth_mm", 508) or 508)
    clearing_time_s = float(in_data.get("clearing_time_s", 0.2) or 0.2)
    clearing_time_min_s = in_data.get("clearing_time_min_s")
    if clearing_time_min_s is None or clearing_time_min_s == "":
        clearing_time_min_s = clearing_time_s
    else:
        clearing_time_min_s = float(clearing_time_min_s)

    if working_distance_mm < 305:
        working_distance_mm = 305.0
    if clearing_time_s <= 0:
        clearing_time_s = 0.2
    if clearing_time_min_s <= 0:
        clearing_time_min_s = clearing_time_s
    # The short-circuit study's LV tolerance, defaulting as its dialog does:
    # pandapower's default 10 % (c max 1.10) gave LV bolted currents 0.15-0.5 %
    # above the short-circuit study's for the same network.
    try:
        lv_tol_percent = int(float(in_data.get("lv_tol_percent", 6) or 6))
    except (TypeError, ValueError):
        lv_tol_percent = 6
    if lv_tol_percent not in (6, 10):
        lv_tol_percent = 6
    equipment_mode = str(in_data.get("equipment_mode", "by_voltage") or "by_voltage").strip().lower()
    if equipment_mode not in ("by_voltage", "uniform"):
        equipment_mode = "by_voltage"
    clearing_mode = str(in_data.get("clearing_time_mode", "fixed") or "fixed").strip().lower()
    if clearing_mode not in ("fixed", "relays"):
        clearing_mode = "fixed"
    opening_s = float(in_data.get("breaker_opening_s", _BREAKER_OPENING_S) or 0.0)
    max_clearing_s = float(in_data.get("max_clearing_time_s", _MAX_CLEARING_S) or _MAX_CLEARING_S)

    # The short-circuit study's rule: 1.1 where none is given, so a k of 0
    # no longer drops the sgen from the arcing current.
    from pandapower_electrisim import ensure_sgen_k

    ensure_sgen_k(net)
    # The relays study their own copy: one that has run a short circuit carries its stale arrays.
    net_relays = copy.deepcopy(net) if clearing_mode == "relays" else None

    try:
        sc.calc_sc(
            net,
            fault="3ph",
            case="max",
            lv_tol_percent=lv_tol_percent,
            ip=False,
            ith=False,
            kappa_method="C",
            branch_results=False,
            return_all_currents=False,
        )
    except Exception as e:
        return json.dumps({
            "error": True,
            "message": "Short-circuit calculation failed (required for arc flash).",
            "exception": str(e),
        })

    if not hasattr(net, "res_bus_sc") or net.res_bus_sc is None or net.res_bus_sc.empty:
        return json.dumps({
            "error": True,
            "message": "No short-circuit bus results available for arc flash.",
            "exception": "Empty res_bus_sc",
        })

    results: List[Dict[str, Any]] = []
    warnings: List[str] = []

    def equipment_of(voc_kv):
        if equipment_mode == "by_voltage":
            equipment, gap_b, distance_b, dims = _equipment_for(voc_kv)
            return equipment, gap_b, distance_b, dims
        return "As entered", conductor_gap_mm, working_distance_mm, (height_mm, width_mm, depth_mm)

    relay_times: Dict[Any, Any] = {}
    if clearing_mode == "relays":
        arcs = []
        for bus_idx, bus_row in net.bus.iterrows():
            ikss = net.res_bus_sc.at[bus_idx, "ikss_ka"] if bus_idx in net.res_bus_sc.index else None
            if ikss is None or not (float(ikss) > 0):
                continue
            voc_kv = float(bus_row["vn_kv"])
            if voc_kv > 15.0:
                arcs.append((bus_idx, float(ikss), None, None))
            elif voc_kv >= 0.208:
                _, gap_b, distance_b, (height_b, width_b, depth_b) = equipment_of(voc_kv)
                ibf = _clamped_ibf(float(ikss), voc_kv)[0]
                try:
                    af = _ieee1584_bus(ibf, voc_kv, electrode, gap_b, distance_b, height_b, width_b, depth_b, 0.1, 0.1)
                except Exception:
                    continue
                arcs.append((bus_idx, ibf, af["ia_full_ka"], af["ia_min_ka"]))
        try:
            relay_times = _relay_clearing_times(net_relays, arcs, in_data, in_data_full, opening_s, max_clearing_s)
        except Exception as e:
            warnings.append(f"The relays' clearing times could not be found ({e}); "
                            f"the entered clearing time used at every bus.")

    for bus_idx, bus_row in net.bus.iterrows():
        if bus_idx not in net.res_bus_sc.index:
            continue

        ikss = net.res_bus_sc.at[bus_idx, "ikss_ka"]
        if ikss is None or (isinstance(ikss, float) and (math.isnan(ikss) or ikss <= 0)):
            continue

        voc_kv = float(bus_row["vn_kv"])
        internal_name = bus_row["name"] if "name" in bus_row.index else str(bus_idx)
        name = _bus_display_name(net, bus_row, bus_idx)
        bus_id = bus_row["id"] if "id" in bus_row.index else str(bus_idx)

        equipment, gap_b, distance_b, (height_b, width_b, depth_b) = equipment_of(voc_kv)
        t_bus, t_min_bus, relays = clearing_time_s, clearing_time_min_s, None
        if bus_idx in relay_times:
            t_bus, t_min_bus, relays = relay_times[bus_idx]

        entry: Dict[str, Any] = {
            "name": _clean(name),
            "id": _clean(bus_id),
            "object_id": _clean(internal_name),
            "vn_kv": _clean(voc_kv),
            "ikss_ka": _clean(float(ikss)),
            "electrode_config": electrode,
            "equipment_class": equipment,
            "working_distance_mm": distance_b,
            "conductor_gap_mm": gap_b,
            "enclosure_height_mm": height_b,
            "enclosure_width_mm": width_b,
            "enclosure_depth_mm": depth_b,
            "clearing_time_s": t_bus,
        }
        if relays is not None:
            entry["clearing_time_min_s"] = t_min_bus
            entry.update({k: _clean(v) if not isinstance(v, list) else v for k, v in relays.items() if k != "error"})
            if relays.get("error"):
                warnings.append(f"Bus {name}: the relays could not be evaluated ({relays['error']}); "
                                f"{max_clearing_s:g} s used.")
            elif relays.get("grid_on_bus"):
                warnings.append(f"Bus {name}: a grid, machine or grid-forming inverter on the bus feeds its arc, "
                                f"no relay parting it from the bus; {max_clearing_s:g} s used.")
            elif relays.get("relay_time_s") is None or relays.get("relay_time_min_s") is None:
                warnings.append(f"Bus {name}: no relay clears its arc; {max_clearing_s:g} s used.")
            if relays.get("inverter_sources") and not relays.get("inverters_blocked"):
                warnings.append(f"Bus {name}: inverters on the bus feed its arc until their own controls stop them "
                                f"- the relays' time does not cover them.")

        # Outside IEEE 1584 range: Ralph Lee (>15 kV) or skip/warn (<0.208 kV)
        if voc_kv > 15.0:
            # Lee has no gap or enclosure.
            entry.update({k: None for k in ("conductor_gap_mm", "enclosure_height_mm",
                                            "enclosure_width_mm", "enclosure_depth_mm")})
            ie, afb = _ralph_lee(float(ikss), voc_kv, distance_b, t_bus)
            entry.update({
                "ia_ka": _clean(float(ikss)),  # Ralph Lee uses bolted current
                "incident_energy_cal_cm2": _clean(ie),
                "arc_flash_boundary_mm": _clean(afb),
                "ppe_category": _ppe_category(ie),
                "method": "RalphLee",
                "clearing_time_used_s": t_bus,
                "note": "Voltage > 15 kV: IEEE 1584-2018 not applicable; Ralph Lee method used.",
            })
            warnings.append(f"Bus {name}: Ralph Lee method used (V={voc_kv} kV > 15 kV).")
            results.append(entry)
            continue

        if voc_kv < 0.208:
            entry.update({
                "ia_ka": None,
                "incident_energy_cal_cm2": None,
                "arc_flash_boundary_mm": None,
                "ppe_category": "N/A",
                "method": "skipped",
                "note": "Voltage < 0.208 kV: below IEEE 1584-2018 applicability range.",
            })
            warnings.append(f"Bus {name}: skipped (V={voc_kv} kV < 0.208 kV).")
            results.append(entry)
            continue

        # Clamp Ibf to model ranges with a soft warning (library raises otherwise)
        ibf, clamp = _clamped_ibf(float(ikss), voc_kv)
        if clamp:
            warnings.append(f"Bus {name}: {clamp}")

        try:
            af = _ieee1584_bus(
                ibf_ka=ibf,
                voc_kv=voc_kv,
                electrode=electrode,
                gap_mm=gap_b,
                working_distance_mm=distance_b,
                height_mm=height_b,
                width_mm=width_b,
                depth_mm=depth_b,
                clearing_time_s=t_bus,
                clearing_time_min_s=t_min_bus,
            )
            entry.update({k: _clean(v) for k, v in af.items()})
        except Exception as e:
            entry.update({
                "ia_ka": None,
                "incident_energy_cal_cm2": None,
                "arc_flash_boundary_mm": None,
                "ppe_category": "N/A",
                "method": "error",
                "note": str(e),
            })
            warnings.append(f"Bus {name}: arc flash calculation failed: {e}")

        results.append(entry)

    payload = {
        "arc_flash": results,
        "parameters": {
            "equipment_mode": equipment_mode,
            "electrode_config": electrode,
            "working_distance_mm": working_distance_mm,
            "conductor_gap_mm": conductor_gap_mm,
            "enclosure_height_mm": height_mm,
            "enclosure_width_mm": width_mm,
            "enclosure_depth_mm": depth_mm,
            "clearing_time_s": clearing_time_s,
            "clearing_time_min_s": clearing_time_min_s,
            "clearing_time_mode": clearing_mode,
            **({"breaker_opening_s": opening_s, "max_clearing_time_s": max_clearing_s}
               if clearing_mode == "relays" else {}),
            "fault": "3ph",
            "case": "max",
            "lv_tol_percent": lv_tol_percent,
        },
    }
    if warnings:
        payload["warnings"] = warnings

    return json.dumps(payload, allow_nan=False)
