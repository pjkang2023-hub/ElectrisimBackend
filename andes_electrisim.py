# -*- coding: utf-8 -*-
"""
ANDES transient stability (TDS) and eigenvalue (EIG) analysis for Electrisim.

Builds an ANDES System directly from Electrisim JSON (no pandapower→ANDES converter).
Applies default GENROU + EXDC2 + TGOV1 dynamics when generator fields are missing.
"""
from __future__ import annotations

import json
import math
import traceback
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import grid_voltage_profile as gvp

GRID_PROFILE_V_FLOOR = 0.05     # pu: the lowest grid voltage ANDES applies

try:
    import andes

    _HAS_ANDES = True
except ImportError:
    andes = None  # type: ignore
    _HAS_ANDES = False


# Textbook GENROU defaults (device base, Kundur-like)
_DEFAULT_GENROU = {
    "M": 12.0,  # 2H
    "D": 0.0,
    "ra": 0.0,
    "xl": 0.15,
    "xd": 1.8,
    "xq": 1.7,
    "xd1": 0.3,
    "xq1": 0.55,
    "xd2": 0.25,
    "xq2": 0.25,
    "Td10": 8.0,
    "Td20": 0.03,
    "Tq10": 0.4,
    "Tq20": 0.05,
}

_EXCITER_DEFAULTS: Dict[str, Dict[str, float]] = {
    "EXDC2": {
    "TR": 0.01,
    "TA": 0.2,
    "TC": 1.0,
    "TB": 10.0,
    "TE": 0.314,
    "TF1": 1.0,
    "KF1": 0.063,
    "KA": 20.0,
    "KE": 1.0,
    "VRMAX": 5.0,
    "VRMIN": -5.0,
    "E1": 3.1,
    "SE1": 0.33,
    "E2": 2.3,
    "SE2": 0.1,
    },
    "SEXS": {"TATB": 0.1, "TB": 10.0, "K": 100.0, "TE": 0.05, "EMIN": -4.0, "EMAX": 4.0},
    # The remaining parameters deliberately use ANDES model defaults. These common
    # parameters provide useful, conservative starting values when supplied by UI.
    "IEEEX1": {"TR": 0.01, "KA": 50.0, "TA": 0.05, "VRMAX": 5.0, "VRMIN": -5.0},
    "ESDC2A": {"TR": 0.01, "KA": 20.0, "TA": 0.2, "VRMAX": 5.0, "VRMIN": -5.0},
    "EXST1": {"TR": 0.01, "KA": 100.0, "TA": 0.05, "VRMAX": 5.0, "VRMIN": -5.0},
    "ESST1A": {"TR": 0.01, "KA": 100.0, "TA": 0.05, "VRMAX": 5.0, "VRMIN": -5.0},
    "AC8B": {"TR": 0.01, "KA": 40.0, "TA": 0.05, "VRMAX": 5.0, "VRMIN": -5.0},
}

_GOVERNOR_DEFAULTS: Dict[str, Dict[str, float]] = {
    "TGOV1": {
    "R": 0.05,
    "T1": 0.5,
    "T2": 1.0,
    "T3": 1.0,
    "VMAX": 1.2,
    "VMIN": 0.0,
    "Dt": 0.0,
    },
    "IEEEG1": {"R": 0.05, "T1": 0.5, "T2": 1.0, "T3": 1.0, "VMAX": 1.2, "VMIN": 0.0},
    "IEESGO": {"T1": 0.1, "T2": 0.1, "T3": 0.1, "T4": 0.1, "T5": 0.1, "T6": 0.1},
    "GAST": {"R": 0.05, "T1": 0.4, "T2": 0.1, "T3": 0.1, "VMAX": 1.2, "VMIN": 0.0},
    "HYGOV": {"R": 0.05, "T1": 0.5, "T2": 1.0, "T3": 1.0, "VMAX": 1.2, "VMIN": 0.0},
}

_PSS_DEFAULTS = {"IEEEST": {"A1": 0.1, "A2": 0.1, "A3": 0.1, "A4": 0.1, "A5": 0.1, "A6": 0.1}}

_RENEWABLE_DEFAULTS: Dict[str, Dict[str, float]] = {
    "REGCA1": {"Tg": 0.02, "Lvplsw": 1.0, "Volim": 1.2, "Lvpnt0": 0.4, "Iolim": -1.5},
    "REECA1": {"Vref0": 1.0, "dbd1": -0.02, "dbd2": 0.02},
    "REPCA1": {"dbd1": -0.02, "dbd2": 0.02, "Kp": 1.0},
    "WTDTA1": {"H": 3.0, "DAMP": 0.0, "Htfrac": 0.5, "Freq1": 1.0, "Dshaft": 1.0},
    "WTARA1": {},
    "WTPTA1": {},
    "WTTQA1": {},
    "PVD1": {},
    "ESD1": {},
}

# Mode flags ANDES 2.0 makes mandatory (no default), so a device added without
# them is rejected. Values follow ANDES' own ieee14_wt3 case: the electrical
# controller takes its Q from the plant controller, which regulates voltage with
# no frequency response; Q priority at the current limit; WTTQA1 on speed error.
# A wind plant's power order is speed dependent (REECA1 PFLAG = 1).
_RENEWABLE_FLAGS: Dict[str, Dict[str, int]] = {
    "REECA1": {"PFFLAG": 0, "VFLAG": 0, "QFLAG": 0, "PFLAG": 0, "PQFLAG": 0},
    "REPCA1": {"VCFlag": 1, "RefFlag": 1, "Fflag": 0, "PLflag": 0},
    "WTTQA1": {"Tflag": 0},
    "PVD1": {"pqflag": 0},
    "ESD1": {"pqflag": 0},
}


def _sf(value: Any, default: float = 0.0) -> float:
    if value is None or value == "" or str(value).lower() in ("none", "null", "nan"):
        return default
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _sb(value: Any, default: bool = True) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes")


def _dyn_value(el: Dict[str, Any], key: str, default: float) -> float:
    """Read a Dynamics attribute while preserving blank → model default behavior."""
    value = el.get(key)
    if value is None or str(value).strip().lower() in ("", "none", "null"):
        return default
    return _sf(value, default)


def _add_model_safe(ss: Any, model: str, defaults_applied: List[str], label: str, **kwargs: Any) -> Optional[str]:
    """
    Add an optional ANDES model without making a diagram unusable on another ANDES
    release. ANDES validates both model availability and parameter names in ss.add().
    A rejected device is taken back out, so the system is as it was before.
    """
    mdl = getattr(ss, model, None)
    n0 = getattr(mdl, "n", 0)
    try:
        idx = ss.add(model, **kwargs)
    except Exception as exc:
        if mdl is not None and mdl.n != n0:
            _discard_partial_add(mdl, n0)
        defaults_applied.append(
            f"{label}: could not add {model} ({exc}); continuing without that optional dynamic model."
        )
        return None
    if idx is None:  # ANDES only logs an unknown model
        defaults_applied.append(
            f"{label}: this ANDES release has no {model}; continuing without that optional dynamic model."
        )
        return None
    return str(idx)


def _discard_partial_add(mdl: Any, n0: int) -> None:
    """
    Undo a device add that raised part-way. ANDES appends each parameter's
    value in turn and registers the device with its group only at the end, so
    a failure leaves the model counting a device whose later parameters are
    missing; TDS initialisation then fails on the mismatched array lengths.
    """
    for param in mdl.params.values():
        v = getattr(param, "v", None)
        if isinstance(v, list):
            del v[n0:]
        elif isinstance(v, np.ndarray) and v.size > n0:
            param.v = v[:n0]
    dropped = [k for k, u in mdl.uid.items() if u >= n0]
    for k in dropped:
        del mdl.uid[k]
    for idxes in getattr(mdl, "_param_corrections", {}).values():
        idxes[:] = [i for i in idxes if i not in dropped]
    mdl.n = n0


def _model_kwargs(
    el: Dict[str, Any],
    attr_prefix: str,
    defaults: Dict[str, float],
) -> Dict[str, float]:
    """Build only the documented key parameters; unset fields use per-model defaults."""
    return {
        key: _dyn_value(el, f"{attr_prefix}{key}", default)
        for key, default in defaults.items()
    }


def tds_values(ss, var) -> np.ndarray:
    """Recorded time-domain values of an ANDES variable, one column per element.

    States and algebraic variables are numbered separately, so their addresses
    index ``dae.ts.x`` and ``dae.ts.y`` respectively. (``TDS.plt`` is None
    under a server in ANDES 2, and its columns are offset by time and by the
    states, so raw addresses read the wrong variables there.)
    """
    data = ss.dae.ts.x if var.v_code == "x" else ss.dae.ts.y
    return np.asarray(data, dtype=float)[:, list(var.a)]


def _check_init(ss, warnings: List[str]) -> None:
    """Warn when the dynamic models do not start in steady state.

    ANDES marks the system initialised even when its own residual check
    fails, so results then begin with a spurious transient.
    """
    if not ss.TDS.initialized:
        ss.TDS.init()
    fg = np.asarray(ss.dae.fg, dtype=float)
    mismatch = float(np.max(np.abs(fg))) if fg.size else 0.0
    if mismatch > 1e-3:
        warnings.append(
            f"The dynamic models could not be initialised from the power flow "
            f"(largest mismatch {mismatch:.3g} pu), so the results start with a "
            "transient that is not caused by any event. Check generator setpoints "
            "and reactive limits, and the machine and exciter data."
        )


def _clean_num(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            return None
        return fv
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, complex):
        return {"re": _clean_num(v.real), "im": _clean_num(v.imag)}
    return v


def _downsample(arr: np.ndarray, max_points: int = 800) -> np.ndarray:
    n = len(arr)
    if n <= max_points:
        return arr
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return arr[idx]


def _z_base_ohm(vn_kv: float, sn_mva: float) -> float:
    if vn_kv <= 0 or sn_mva <= 0:
        return 1.0
    return (vn_kv ** 2) / sn_mva


def _iter_elements(in_data: Dict[str, Any]):
    for key in sorted(in_data.keys(), key=lambda k: int(k) if str(k).isdigit() else str(k)):
        el = in_data[key]
        if not isinstance(el, dict):
            continue
        typ = el.get("typ") or ""
        if "Parameters" in typ:
            continue
        yield key, el, typ


def _islands(ss, out_lines=()) -> List[set]:
    """The buses as islands: each set joined by in-service lines and transformers, out_lines left out."""
    adjacency: Dict[Any, set] = {b: set() for b in ss.Bus.idx.v}
    for idx, a, b, u in zip(ss.Line.idx.v, ss.Line.bus1.v, ss.Line.bus2.v, ss.Line.u.v):
        if u and idx not in out_lines:
            adjacency[a].add(b)
            adjacency[b].add(a)
    seen, out = set(), []
    for start in ss.Bus.idx.v:
        if start in seen:
            continue
        island, queue = {start}, [start]
        for bus in queue:
            for nxt in adjacency[bus] - island:
                island.add(nxt)
                queue.append(nxt)
        seen |= island
        out.append(island)
    return out


def _deenergise_island(ss, out_lines, t: float, out_slacks=(), grid_forming=()) -> Tuple[List[Any], List[str], List[set]]:
    """
    Switch off, at time t, everything an outage - lines, or an External
    Grid's slack - cuts off from the grid, unless it can run as an island.

    On a radial network every line outage islands what lies beyond it. ANDES
    cannot simulate a dead island - its loads at 0 V leave the bus angles
    undetermined - and the run stopped at the outage. A generator cut off
    from the grid is tripped by its loss-of-mains protection, so the island's
    generators are tripped with the line, and its loads, shunts, static
    generators and inner lines switched off; its buses keep no devices and
    are reported at 0 V - as pandapower reports an island without an External
    Grid.

    A part cut off with a grid-forming PCS is a microgrid built to island:
    it runs on, its machines' governors and the PCS's droops sharing its load.
    Returns the de-energised bus idx, the generators tripped, and the islands
    that run on.
    """
    sources = {b for idx, b in zip(ss.Slack.idx.v, ss.Slack.bus.v) if idx not in out_slacks}
    live = set().union(*(p for p in _islands(ss) if p & set(ss.Slack.bus.v)))
    running, island = [], []
    for part in _islands(ss, set(out_lines)):
        part = part & live
        if not part or part & sources:
            continue
        if part & set(grid_forming):
            running.append(part)
        else:
            island += [b for b in ss.Bus.idx.v if b in part]
    lines = list(zip(ss.Line.idx.v, ss.Line.bus1.v, ss.Line.bus2.v, ss.Line.u.v))
    if not island:
        return [], [], running
    n = 0
    tripped, machine_buses = [], set()
    for model in ("GENROU", "GENCLS"):
        mdl = getattr(ss, model)
        for k in range(mdl.n):
            if mdl.bus.v[k] in island:
                n += 1
                ss.add("Toggle", idx=f"Toggle_Island_{n}", model="SynGen", dev=mdl.idx.v[k], t=t)
                tripped.append(str(mdl.name.v[k]))
                machine_buses.add(mdl.bus.v[k])
    for model in ("PQ", "Shunt", "PV"):
        mdl = getattr(ss, model)
        for k in range(mdl.n):
            # A PV a machine replaces goes with the machine.
            if mdl.bus.v[k] in island and not (model == "PV" and mdl.bus.v[k] in machine_buses):
                n += 1
                ss.add("Toggle", idx=f"Toggle_Island_{n}", model=model, dev=mdl.idx.v[k], t=t)
    for idx, a, b, u in lines:
        if idx not in out_lines and u and a in island and b in island:
            n += 1
            ss.add("Toggle", idx=f"Toggle_Island_{n}", model="Line", dev=idx, t=t)
    return island, tripped, running


# --- PCS: a battery, flywheel, SOFC system or PV array on an AC bus through its inverter -----------

# A grid-forming PCS's power filter (s), as the EMT study's GridFormingVsc: its droop through that
# filter is a virtual machine, M = tau D, D = 1 / droop.
PCS_TAU_F = 0.02
# A closed bus tie's reactance (p.u. on the system base): small against any branch, not so small
# that the network matrix loses its conditioning.
BUS_TIE_X_PU = 1e-4
# A grid-forming PCS's coupling reactance (p.u. on its rating): REGCV1's xs.
PCS_XS_PU = 0.2


def _pcs_plants(in_data: Dict[str, Any], warnings: List[str]) -> List[Dict[str, Any]]:
    """
    Each PCS with its source, and the AC power and Q the load flow asks of it
    (pandapower_electrisim._electrisim_pcs_set_point): its source found as
    the load flow finds it - wired to its DC side, or alone on its DC bus.
    """
    import types
    import der_electrisim
    from pandapower_electrisim import _electrisim_pcs_set_point, _electrisim_pcs_window

    rows = [el for _, el, typ in _iter_elements(in_data) if typ.startswith("PCS")]
    if not rows:
        return []
    ders = [el for _, el, typ in _iter_elements(in_data) if der_electrisim.kind_of(typ)]
    out = []
    for el in rows:
        label = str(el.get("userFriendlyName") or el.get("name"))
        src = next((d for d in ders if el.get("der") and d.get("name") == el.get("der")), None)
        if src is None and el.get("bus_dc"):
            on_bus = [d for d in ders if d.get("bus") == el.get("bus_dc")]
            src = on_bus[0] if len(on_bus) == 1 else None
        if src is None:
            warnings.append(f"PCS '{label}' has no battery, supercapacitor, flywheel, SOFC system or PV array "
                            "on its DC side, so it is left out.")
            continue
        kind = der_electrisim.kind_of(src.get("typ"))
        try:
            obj = der_electrisim.build(src)
        except ValueError as e:
            warnings.append(f"PCS '{label}' is left out: {e}")
            continue
        if not (_sb(el.get("in_service"), True) and obj.in_service):
            continue
        s_rated = _sf(el.get("s_rated_mva"), 1.0)
        s_rated = s_rated if s_rated > 0 else 1.0
        eta = _sf(el.get("efficiency_percent"), 98.0) / 100.0
        rec = {
            "name": el.get("name"), "id": el.get("id", ""), "label": label, "bus": el.get("bus"),
            "control": "grid_forming" if str(el.get("control") or "").strip().lower() == "grid_forming"
            else "grid_following",
            "s_rated": s_rated, "eta": eta if 0 < eta <= 1 else 1.0,
            "p_nl_mw": _sf(el.get("no_load_loss_kw"), 0.0) / 1e3, "p_set_mw": _sf(el.get("p_set_mw"), 0.0),
            "q_mode": str(el.get("q_mode") or "q"), "q_set_mvar": _sf(el.get("q_set_mvar"), 0.0),
            "pf": _sf(el.get("pf"), 1.0), "vm_set_pu": _sf(el.get("vm_set_pu"), 1.0),
            "droop_pf": max(_sf(el.get("droop_pf_percent"), 2.0), 1e-3) / 100.0,
            "droop_qv": max(_sf(el.get("droop_qv_percent"), 5.0), 0.0) / 100.0,
            "qv_droop": _sf(el.get("qv_droop_percent"), 5.0) / 100.0,
            "k": _sf(el.get("current_limit_pu"), 1.2),
            "source": {"obj": obj, "kind": kind, "name": src.get("name"),
                       "label": str(src.get("userFriendlyName") or src.get("name"))},
        }
        net = types.SimpleNamespace(warnings=[])
        rec["p_ac"], rec["q"] = _electrisim_pcs_set_point(net, rec)
        warnings.extend(net.warnings)
        if rec["q_mode"] == "qv" and rec["control"] == "grid_following":
            warnings.append(f"PCS '{label}': it starts at its Q(V) droop's point, and holds that Q through the run.")
        rec["p_max_ac"], rec["p_min_ac"] = _electrisim_pcs_window(rec)
        out.append(rec)
    return out


def _add_pcs_dynamics(ss: Any, rec: Dict[str, Any], n: int, bus: Any, static_idx: str, freq: float,
                      sn_base: float, defaults_applied: List[str]) -> Dict[str, Any]:
    """
    The PCS's ANDES model. Grid-forming: GENCLS, a virtual machine whose
    damping is its P-f droop and whose inertia its power filter's. Grid-
    following: a battery or flywheel ESD1 (its state of charge between its
    window's ends, the energy it can store), a PV array or an SOFC system
    PVD1 up to its rating.
    """
    s, label = rec["s_rated"], f"PCS '{rec['label']}'"
    obj, kind = rec["source"]["obj"], rec["source"]["kind"]
    if rec["control"] == "grid_forming":
        # A virtual machine: GENCLS's swing equation is the grid-forming control's, M dw/dt = P0 - P
        # - D (w - 1), its damping D the P-f droop's 1 / droop and its inertia M = tau D its power
        # filter's; its voltage behind REGCV1's coupling reactance, held through the run (its Q-V
        # droop sets where it starts; through a run it is the EMT study's). REGCV1 itself answered
        # the angle through slow voltage loops: in an island it slipped poles against the turbines.
        d = 1.0 / rec["droop_pf"]
        vn = ss.Bus.Vn.v[list(ss.Bus.idx.v).index(bus)]
        idx = _add_model_safe(ss, "GENCLS", defaults_applied, label, idx=f"GENCLS_PCS_{n}",
                              name=rec["label"], bus=bus, gen=static_idx, Sn=s, Vn=vn, fn=freq,
                              D=d, M=PCS_TAU_F * d, ra=0.0, xd1=PCS_XS_PU)
        return {"model": "GENCLS", "model_idx": idx}
    # Frequency trip points, as a 60 Hz system's scaled to this one: IEEE 1547-2018 Category III's
    # (UF2 56.5, UF1 58.5, OF1 61.8, OF2 62.0 Hz), between which ESD1 and PVD1 ride through. ANDES's
    # own, 59.5-59.7 Hz, are 1547-2003's: an island at 49.3 Hz cut every PV array and battery out.
    trips = {k: v * freq / 60.0 for k, v in (("ft0", 56.5), ("ft1", 58.5), ("ft2", 61.8), ("ft3", 62.0))}
    common = dict(bus=bus, gen=static_idx, Sn=s, fn=freq, pqflag=0, ialim=max(rec["k"], 0.1),
                  qmx=1.0, qmn=-1.0, **trips)
    if kind in ("Battery", "Flywheel", "Supercapacitor"):
        if kind == "Battery":
            en, soc0, soc_min, soc_max = obj.energy_kwh / 1e3, obj.soc0, obj.soc_min, obj.soc_max
            eta_c, eta_d = obj.eta_charge, 1.0
        elif kind == "Supercapacitor":
            # Its usable energy, 1/2 C (V^2 - V_min^2), as a share of its energy at its rated voltage.
            usable = obj.energy(obj.v_rated)[1]
            en, soc0, soc_min, soc_max = usable / 3.6e9, obj.energy()[1] / max(usable, 1e-12), 0.0, 1.0
            eta_c = eta_d = 1.0
        else:
            # Its energy, 1/2 J w^2, as a share of its energy at full speed: speed squared.
            en, soc0, soc_min, soc_max = obj.e_max / 3.6e9, obj.s0 ** 2, obj.s_min ** 2, 1.0
            eta_c = eta_d = obj.eta
        idx = _add_model_safe(ss, "ESD1", defaults_applied, label, idx=f"ESD1_PCS_{n}",
                              name=f"ESD1_{rec['label']}", pmx=max(rec["p_max_ac"], -rec["p_min_ac"], 1e-6) / s,
                              En=max(en, 1e-9), SOCinit=soc0, SOCmin=soc_min, SOCmax=soc_max,
                              EtaC=eta_c, EtaD=eta_d, **common)
        return {"model": "ESD1", "model_idx": idx}
    # A PV array, and an SOFC system: its power order is constant through a run, so the ramp limit
    # and minimum load REGCA1 + REECA1 gave it never acted, while REGCA1's low-voltage gain stalled
    # the solver - at its 0.8 pu breakpoint the campus's SOFC inverters switched state almost every
    # step through IEEE 2800's ride-through envelope. Its ramp is the time series' to apply.
    idx = _add_model_safe(ss, "PVD1", defaults_applied, label, idx=f"PVD1_PCS_{n}",
                          name=f"PVD1_{rec['label']}", pmx=max(rec["p_max_ac"], 1e-6) / s, **common)
    return {"model": "PVD1", "model_idx": idx}


def _settle_pcs_set_points(ss, meta: Dict[str, Any], rounds: int = 30) -> bool:
    """
    The load flow's droops in ANDES's power flow, rerun until they hold: a
    grid-forming PCS at v = v_set - droop Q / S (its static generator's
    voltage), a grid-following PCS in Q(V) mode at Q = -(v - v_set) / droop S
    within its capability (the Q its static generator is held at). Each by
    secant steps, as pandapower_electrisim._electrisim_settle_pcs.
    """
    sn = meta["sn_mva"]
    plants = [g for g in meta["gen_map"].values() if g.get("pcs") and (
        g["control"] == "grid_forming" and g["droop_qv"] > 0 and not g["on_slack_bus"]
        or g["control"] == "grid_following" and g["q_mode"] == "qv")]
    if not plants:
        return True
    hist: Dict[str, Tuple[float, float]] = {}

    def secant(key, x, g, lo, hi, max_step):
        last = hist.get(key)
        hist[key] = (x, g)
        x_new = x - 0.5 * g
        if last is not None and abs(x - last[0]) > 1e-12 and abs(g - last[1]) > 1e-15:
            slope = (g - last[1]) / (x - last[0])
            if slope > 0:
                x_new = x - g / slope
        return min(max(x + max(min(x_new - x, max_step), -max_step), lo), hi)

    pv_idx = list(ss.PV.idx.v)
    for _ in range(rounds):
        worst = 0.0
        for g in plants:
            k = pv_idx.index(g["static_idx"])
            q = float(ss.PV.q.v[k]) * sn
            v = float(ss.Bus.v.v[ss.Bus.idx2uid(g["bus"])])
            if g["control"] == "grid_forming":
                v_set = float(ss.PV.v0.v[k])
                err = v_set - (g["vm_set_pu"] - g["droop_qv"] * q / g["s_rated_mva"])
                worst = max(worst, abs(err))
                if abs(err) > 1e-7:
                    ss.PV.set("v0", g["static_idx"], secant(g["static_idx"], v_set, err, 0.8, 1.2, 0.05), base="device")
            else:
                target = -(v - g["vm_set_pu"]) / max(g["qv_droop"], 1e-3) * g["s_rated_mva"]
                target = max(min(target, g["q_max_mvar"]), -g["q_max_mvar"])
                err = (q - target) / g["s_rated_mva"]
                worst = max(worst, abs(err))
                if abs(err) > 1e-7:
                    q_new = q - 0.5 * (q - target)
                    ss.PV.set("qmax", g["static_idx"], q_new / sn, base="device")
                    ss.PV.set("qmin", g["static_idx"], q_new / sn, base="device")
        if worst <= 1e-7:
            return True
        if not ss.PFlow.run():
            return False
    meta["warnings"].append("The PCS's voltage droops did not settle in ANDES's power flow; it starts near them.")
    return True


# --- The DC network, through its converters ----------------------------------------------------

_DC_CONVERTERS = ("VSC", "B2B VSC", "Solid-State Transformer")


def _electrisim_flow(in_data: Dict[str, Any], freq: float) -> Tuple[Any, Optional[str]]:
    """
    Electrisim's own load flow of the diagram - the DC network solved, its
    sources and stores settled, an island's machines and grid-forming PCS
    sharing its load by their droops - with the Load Flow study's fallbacks
    (more iterations, then a flat start): (the solved network or None, why
    not). Without them the islanded campus did not solve, and its data halls
    vanished from the study.
    """
    import pandapower as pp
    import pandapower_electrisim as pe
    rows = {k: v for k, v in in_data.items() if isinstance(v, dict) and "Parameters" not in str(v.get("typ", ""))}
    rows["__andes_lf"] = {"typ": "PowerFlowPandaPower Parameters"}
    why = "it did not converge"
    try:
        net = pp.create_empty_network(f_hz=freq)
        busbars = pe.create_busbars(rows, net)
        pe.create_other_elements(rows, net, "__andes_lf", busbars)
    except Exception as exc:
        return None, str(exc)
    for plan in ({}, {"max_iteration": 50}, {"init": "flat", "max_iteration": 100}):
        # Each plan from the network as built: a failed one leaves its DC loads where it gave up.
        attempt = deepcopy(net)
        try:
            pe._electrisim_runpp(attempt, algorithm="nr", calculate_voltage_angles="auto", **plan)
            if attempt.converged:
                return attempt, None
        except Exception as exc:
            why = str(exc)
    return None, why


def _converter_ac_loads(net: Any) -> List[Tuple[str, float, float, str]]:
    """
    Each converter joining the DC network to an AC bus, as the load it is on
    that bus: (bus name, P MW, Q Mvar, label) - what Electrisim's own load
    flow gives at its AC side. A VSC or back-to-back VSC draws P and Q; a
    solid-state transformer draws its MV power and, from its LV AC port,
    gives its inverter's (a negative load). ANDES has no DC network: without
    this a data hall's load vanished from the study.
    """
    import pandapower_electrisim as pe
    out = []
    name = lambda b: str(net.bus.at[int(b), "name"])
    label = lambda table, i: str(getattr(net, "user_friendly_names", {}).get(net[table].at[i, "name"], net[table].at[i, "name"]))
    for table in ("vsc", "b2b_vsc"):
        df, res = net.get(table), net.get(f"res_{table}")
        if df is None or not len(df) or res is None:
            continue
        for i in df.index:
            if pe._electrisim_is_aux(df, i) or i not in res.index or not bool(df.at[i, "in_service"]):
                continue     # a DC/DC converter's own, on its hidden bus
            p, q = float(res.at[i, "p_mw"]), float(res.at[i, "q_mvar"])
            if np.isfinite(p) and np.isfinite(q):
                out.append((name(df.at[i, "bus"]), p, q, label(table, i)))
    for rec in getattr(net, "electrisim_ssts", None) or []:
        r = pe._electrisim_sst_result(net, rec)
        if not rec.get("in_service", True):
            continue
        out.append((name(rec["bus_mv"]), r["p_mv_mw"], r["q_mv_mvar"], f"{rec['label']} (MV)"))
        if rec.get("bus_lvac") is not None and r.get("p_lv_ac_mw") is not None:
            out.append((name(rec["bus_lvac"]), -r["p_lv_ac_mw"], -(r.get("q_lv_ac_mvar") or 0.0), f"{rec['label']} (LV AC)"))
    return out


def build_system(
    in_data: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None,
    exclude_motors: Optional[set] = None,
    setup: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Build an ANDES System from Electrisim JSON.

    Motors are static loads; exclude_motors names the ones a caller models
    itself (dynamic motor starting), so they are not counted twice. With
    setup=False the caller can add devices before calling ss.setup().

    Returns (ss, meta) where meta includes bus_map, line_map, gen_map, defaults_applied, warnings.
    """
    if not _HAS_ANDES:
        raise RuntimeError("ANDES is not installed. Install with: pip install andes")

    params = params or {}
    freq = _sf(params.get("frequency"), 50.0)
    sn_base = _sf(params.get("sn_mva"), 100.0)
    if sn_base <= 0:
        sn_base = 100.0

    defaults_applied: List[str] = []
    warnings: List[str] = []
    bus_map: Dict[str, Any] = {}  # electrisim name -> andes bus idx
    bus_vn: Dict[Any, float] = {}
    bus_name_by_idx: Dict[Any, str] = {}
    line_map: Dict[str, Any] = {}
    gen_map: Dict[str, Dict[str, Any]] = {}  # electrisim gen name -> {static_idx, syn_idx, ...}
    friendly: Dict[str, str] = {}

    ss = andes.System()
    ss.config.freq = freq
    ss.config.mva = sn_base
    # Avoid writing report files into the server cwd
    try:
        ss.files.no_output = True
    except Exception:
        pass

    # --- Buses ---
    bus_counter = 1
    for _, el, typ in _iter_elements(in_data):
        if "DC Bus" in typ:
            continue          # the DC network enters through its converters (_converter_ac_loads)
        if "Bus" not in typ:
            continue
        name = el.get("name")
        if not name:
            continue
        vn = _sf(el.get("vn_kv"), 0.0)
        if vn <= 0:
            vn = 110.0
            defaults_applied.append(f"Bus '{el.get('userFriendlyName', name)}': vn_kv missing, used 110 kV.")
        idx = bus_counter
        bus_counter += 1
        u = 1 if _sb(el.get("in_service"), True) else 0
        ss.add(
            "Bus",
            idx=idx,
            name=str(el.get("userFriendlyName") or name),
            Vn=vn,
            u=u,
            v0=1.0,
            a0=0.0,
        )
        bus_map[name] = idx
        ufn = el.get("userFriendlyName")
        if ufn and str(ufn) != str(name):
            bus_map[str(ufn)] = idx
        bus_vn[idx] = vn
        bus_name_by_idx[idx] = str(el.get("userFriendlyName") or name)
        friendly[name] = str(el.get("userFriendlyName") or name)

    if not bus_map:
        raise ValueError("No buses found in the diagram. Place Bus elements before running stability analysis.")

    # --- Switches ---
    # ANDES has none, and they were ignored: a closed bus tie left its buses apart (the campus's two
    # 35 kV buses each on their own feeder), an open breaker left its line or transformer in.
    open_elements, bus_ties, breakers = set(), [], {}
    for _, el, typ in _iter_elements(in_data):
        if not typ.startswith("Switch"):
            continue
        closed = _sb(el.get("closed"), True)
        if str(el.get("et") or "").lower() == "b":
            if closed:
                bus_ties.append(el)
        elif el.get("element"):
            if not closed:
                open_elements.add(el.get("element"))
            breakers[el.get("name")] = el

    # --- Lines ---
    for _, el, typ in _iter_elements(in_data):
        if not typ.startswith("Line") or typ.startswith("Load"):
            continue
        if "DC" in typ:
            continue          # the DC network enters through its converters
            continue
        name = el.get("name")
        bus1 = bus_map.get(el.get("busFrom"))
        bus2 = bus_map.get(el.get("busTo"))
        if bus1 is None or bus2 is None:
            warnings.append(f"Skipped Line '{el.get('userFriendlyName', name)}': missing bus connection.")
            continue
        length = _sf(el.get("length_km"), 1.0)
        if length <= 0:
            length = 1.0
        vn1 = bus_vn.get(bus1, 110.0)
        vn2 = bus_vn.get(bus2, vn1)
        zb = _z_base_ohm(vn1, sn_base)
        r_ohm = _sf(el.get("r_ohm_per_km")) * length
        x_ohm = _sf(el.get("x_ohm_per_km")) * length
        c_nf = _sf(el.get("c_nf_per_km")) * length
        g_us = _sf(el.get("g_us_per_km")) * length
        r_pu = r_ohm / zb if zb else 0.0
        x_pu = x_ohm / zb if zb else 0.01
        # B (S) = 2*pi*f*C; C in F = c_nf * 1e-9; b_pu = B * Zb
        b_siemens = 2.0 * math.pi * freq * (c_nf * 1e-9)
        b_pu = b_siemens * zb if zb else 0.0
        g_pu = (g_us * 1e-6) * zb if zb else 0.0
        if x_pu == 0 and r_pu == 0:
            x_pu = 0.01
            defaults_applied.append(f"Line '{el.get('userFriendlyName', name)}': zero impedance, used x=0.01 pu.")
        line_idx = f"Line_{name}"
        u = 1 if _sb(el.get("in_service"), True) and name not in open_elements else 0
        ss.add(
            "Line",
            idx=line_idx,
            name=str(el.get("userFriendlyName") or name),
            bus1=bus1,
            bus2=bus2,
            r=r_pu,
            x=x_pu,
            b=b_pu,
            g=g_pu,
            Vn1=vn1,
            Vn2=vn2,
            Sn=sn_base,
            fn=freq,
            u=u,
        )
        line_map[name] = line_idx
        ufn = el.get("userFriendlyName")
        if ufn:
            line_map[str(ufn)] = line_idx

    # --- Two-winding transformers as ANDES Line with trans=1 ---
    for _, el, typ in _iter_elements(in_data):
        if not (
            (typ.startswith("Transformer") or typ.startswith("Two Winding Transformer"))
            and not typ.startswith("Three Winding Transformer")
        ):
            continue
        name = el.get("name")
        bus1 = bus_map.get(el.get("hv_bus"))
        bus2 = bus_map.get(el.get("lv_bus"))
        if bus1 is None or bus2 is None:
            warnings.append(f"Skipped Transformer '{el.get('userFriendlyName', name)}': missing HV/LV bus.")
            continue
        sn_t = _sf(el.get("sn_mva"), sn_base)
        if sn_t <= 0:
            sn_t = sn_base
        vk = _sf(el.get("vk_percent"), 6.0)
        vkr = _sf(el.get("vkr_percent"), 1.0)
        # Impedance on transformer base → system base
        z_t = vk / 100.0
        r_t = vkr / 100.0
        x_t = math.sqrt(max(z_t ** 2 - r_t ** 2, 0.0)) if z_t >= r_t else z_t
        scale = sn_base / sn_t
        r_pu = r_t * scale
        x_pu = x_t * scale if x_t > 0 else 0.06 * scale
        vn1 = _sf(el.get("vn_hv_kv"), bus_vn.get(bus1, 110.0))
        vn2 = _sf(el.get("vn_lv_kv"), bus_vn.get(bus2, 11.0))
        tap = 1.0
        # Approximate tap from tap_pos * tap_step_percent
        tap_pos = _sf(el.get("tap_pos"), 0.0)
        tap_step = _sf(el.get("tap_step_percent"), 0.0)
        if tap_step != 0:
            tap = 1.0 + (tap_pos * tap_step / 100.0)
        line_idx = f"Trafo_{name}"
        u = 1 if _sb(el.get("in_service"), True) and name not in open_elements else 0
        ss.add(
            "Line",
            idx=line_idx,
            name=str(el.get("userFriendlyName") or name),
            bus1=bus1,
            bus2=bus2,
            r=r_pu,
            x=x_pu,
            b=0.0,
            g=0.0,
            Vn1=vn1,
            Vn2=vn2,
            Sn=sn_base,
            fn=freq,
            trans=1,
            tap=tap,
            phi=0.0,
            u=u,
        )
        line_map[name] = line_idx

    # --- Three-winding transformers as a star of three trans=1 Lines ---
    # Same equivalent as pandapower: a star-point bus at HV voltage, the pair
    # short-circuit impedances (each on the smaller rating of its pair) split
    # into branch impedances, resistive and reactive parts separately.
    for _, el, typ in _iter_elements(in_data):
        if not typ.startswith("Three Winding Transformer"):
            continue
        name = el.get("name")
        label = str(el.get("userFriendlyName") or name)
        ends = [bus_map.get(el.get(k)) for k in ("hv_bus", "mv_bus", "lv_bus")]
        if any(b is None for b in ends):
            warnings.append(f"Skipped Three Winding Transformer '{label}': missing HV/MV/LV bus.")
            continue
        sn = [_sf(el.get(f"sn_{w}_mva"), 0.0) for w in ("hv", "mv", "lv")]
        if min(sn) <= 0:
            warnings.append(f"Skipped Three Winding Transformer '{label}': missing winding ratings.")
            continue
        vn = [_sf(el.get(f"vn_{w}_kv"), bus_vn.get(b, 0.0)) for w, b in zip(("hv", "mv", "lv"), ends)]

        def _pair(w: str, a: int, b: int) -> complex:
            vk = _sf(el.get(f"vk_{w}_percent"), 0.0)
            vkr = _sf(el.get(f"vkr_{w}_percent"), 0.0)
            scale = sn_base / min(sn[a], sn[b]) / 100.0
            return complex(vkr, math.sqrt(max(vk ** 2 - vkr ** 2, 0.0))) * scale

        # pandapower naming: vk_hv = HV-MV, vk_mv = MV-LV, vk_lv = HV-LV.
        z_hm, z_ml, z_hl = _pair("hv", 0, 1), _pair("mv", 1, 2), _pair("lv", 0, 2)
        z_star = [0.5 * (z_hm + z_hl - z_ml), 0.5 * (z_hm + z_ml - z_hl), 0.5 * (z_hl + z_ml - z_hm)]
        if any(abs(z) == 0 for z in z_star):
            defaults_applied.append(f"Three Winding Transformer '{label}': zero branch impedance, used x=0.001 pu.")

        star = bus_counter
        bus_counter += 1
        u = 1 if _sb(el.get("in_service"), True) and name not in open_elements else 0
        # Kept out of bus_name_by_idx, so results never list it.
        ss.add("Bus", idx=star, name=f"{label} star point", Vn=vn[0], u=u, v0=1.0, a0=0.0)
        bus_vn[star] = vn[0]

        tap = {"hv": 1.0, "mv": 1.0, "lv": 1.0}
        tap_step = _sf(el.get("tap_step_percent"), 0.0)
        side = str(el.get("tap_side") or "hv").lower()
        if tap_step != 0 and side in tap:
            tap[side] = 1.0 + (_sf(el.get("tap_pos"), 0.0) - _sf(el.get("tap_neutral"), 0.0)) * tap_step / 100.0

        for w, bus, v, z in zip(("hv", "mv", "lv"), ends, vn, z_star):
            # The HV branch runs bus -> star; MV and LV run star -> bus.
            bus1, bus2, vn1, vn2 = (bus, star, v, vn[0]) if w == "hv" else (star, bus, vn[0], v)
            ss.add(
                "Line",
                idx=f"Trafo3w_{name}_{w}",
                name=f"{label} ({w.upper()})",
                bus1=bus1,
                bus2=bus2,
                r=z.real,
                x=z.imag if abs(z) > 0 else 0.001,
                b=0.0,
                g=0.0,
                Vn1=vn1,
                Vn2=vn2,
                Sn=sn_base,
                fn=freq,
                trans=1,
                tap=tap[w],
                phi=0.0,
                u=u,
            )
        line_map[name] = f"Trafo3w_{name}_hv"

    # A closed bus tie: its buses joined through a negligible impedance. A breaker is its line or
    # transformer: an outage of either opens it.
    for el in bus_ties:
        a, b = bus_map.get(el.get("bus")), bus_map.get(el.get("element"))
        if a is None or b is None or a == b:
            continue
        name = el.get("name")
        ss.add("Line", idx=f"Switch_{name}", name=str(el.get("userFriendlyName") or name), bus1=a, bus2=b,
               r=0.0, x=BUS_TIE_X_PU, b=0.0, g=0.0, Vn1=bus_vn[a], Vn2=bus_vn[b], Sn=sn_base, fn=freq, u=1)
        line_map[name] = f"Switch_{name}"
        if el.get("userFriendlyName"):
            line_map.setdefault(str(el["userFriendlyName"]), f"Switch_{name}")
    for name, el in breakers.items():
        target = line_map.get(el.get("element"))
        if target is not None:
            line_map.setdefault(name, target)
            if el.get("userFriendlyName"):
                line_map.setdefault(str(el["userFriendlyName"]), target)

    # --- Loads (PQ) ---
    # Loads following a profile from the diagram's library start at its first
    # value; run_tds scales them through the run (1.0 p.u. = the drawn P).
    import load_profiles_electrisim as _lp
    profile_library, profile_problems = _lp.library_from_params(params)
    warnings.extend(profile_problems)
    profile_assignments = _lp.load_assignments(in_data) if profile_library else {}
    profile_repeat = params.get("profile_repeat", True) not in (False, "false", "False", 0, "0")
    profiled_loads: List[Dict[str, Any]] = []
    pq_i = 0
    for _, el, typ in _iter_elements(in_data):
        if not typ.startswith("Load") or typ.startswith("Load DC"):
            continue
        if "Asymmetric" in typ:
            warnings.append(f"Skipped Asymmetric Load '{el.get('userFriendlyName', el.get('name'))}'.")
            continue
        bus = bus_map.get(el.get("bus"))
        if bus is None:
            continue
        pq_i += 1
        scaling = _sf(el.get("scaling"), 1.0)
        p_mw = _sf(el.get("p_mw")) * scaling
        q_mvar = _sf(el.get("q_mvar")) * scaling
        u = 1 if _sb(el.get("in_service"), True) else 0
        load_name = str(el.get("userFriendlyName") or el.get("name") or f"PQ_{pq_i}")
        f0, q_f0 = 1.0, 1.0
        assignment = profile_assignments.get(str(el.get("name")))
        profile = profile_library.get(assignment["profile_id"]) if assignment else None
        if assignment and profile is None:
            warnings.append(f"{load_name}: its load profile is not in the library, so it does not follow one.")
        if profile is not None:
            rel_t = profile["t"] - profile["t"][0]
            f0 = float(_lp.sample_profile(rel_t, profile["p"], [0.0], profile_repeat)[0])
            q_f0 = f0 if assignment["q_mode"] == "pf" else 1.0
            profiled_loads.append({
                "idx": f"PQ_{pq_i}", "name": load_name, "profile": profile["name"],
                "t": rel_t, "p": profile["p"], "q_mode": assignment["q_mode"],
                "p_rated": p_mw / sn_base, "q_rated": q_mvar / sn_base,
            })
        ss.add(
            "PQ",
            idx=f"PQ_{pq_i}",
            name=load_name,
            bus=bus,
            p0=p_mw * f0 / sn_base,
            q0=q_mvar * q_f0 / sn_base,
            Vn=bus_vn.get(bus, 110.0),
            u=u,
        )

    # --- The DC network: each converter the load its AC side is in the load flow ---
    flow, flow_problem = None, None
    if any(typ.startswith(k) for _, _, typ in _iter_elements(in_data) for k in _DC_CONVERTERS):
        flow, flow_problem = _electrisim_flow(in_data, freq)
        if flow is None:
            warnings.append(f"The DC network's converters are left out: its load flow failed ({flow_problem}).")
    converter_loads = _converter_ac_loads(flow) if flow is not None else []
    for bus_name, p_mw, q_mvar, label in converter_loads:
        bus = bus_map.get(bus_name)
        if bus is None:
            continue
        pq_i += 1
        ss.add("PQ", idx=f"PQ_{pq_i}", name=f"{label} (DC network)", bus=bus, Vn=bus_vn.get(bus, 110.0),
               p0=p_mw / sn_base, q0=q_mvar / sn_base)
    if converter_loads:
        warnings.append(
            "The DC network is in this study as its converters' AC power from the load flow, held through the run "
            "(as loads are): " + ", ".join(f"{label} {p:.4g} MW" for _, p, _, label in converter_loads)
            + ". The DC network's own dynamics are the EMT study's.")

    # --- Storage: a fixed P/Q, positive while charging (pandapower's sign) ---
    for _, el, typ in _iter_elements(in_data):
        if not typ.startswith("Storage"):
            continue
        bus = bus_map.get(el.get("bus"))
        if bus is None or not _sb(el.get("in_service"), True):
            continue
        pq_i += 1
        scaling = _sf(el.get("scaling"), 1.0)
        ss.add(
            "PQ", idx=f"PQ_{pq_i}", name=str(el.get("userFriendlyName") or el.get("name")),
            bus=bus, Vn=bus_vn.get(bus, 110.0),
            p0=_sf(el.get("p_mw")) * scaling / sn_base,
            q0=_sf(el.get("q_mvar")) * scaling / sn_base,
        )

    # --- Motors: the load pandapower gives them in a power flow ---
    for _, el, typ in _iter_elements(in_data):
        if not typ.startswith("Motor") or "Parameters" in typ:
            continue
        if exclude_motors and str(el.get("name")) in exclude_motors:
            continue
        bus = bus_map.get(el.get("bus"))
        if bus is None or not _sb(el.get("in_service"), True):
            continue
        eff = _sf(el.get("efficiency_percent"), 100.0) or 100.0
        p_mw = (_sf(el.get("pn_mech_mw")) * _sf(el.get("loading_percent"), 100.0) / eff
                * _sf(el.get("scaling"), 1.0))
        cos_phi = min(max(_sf(el.get("cos_phi"), 0.85), 1e-3), 1.0)
        pq_i += 1
        ss.add(
            "PQ", idx=f"PQ_{pq_i}", name=str(el.get("userFriendlyName") or el.get("name")),
            bus=bus, Vn=bus_vn.get(bus, 110.0),
            p0=p_mw / sn_base, q0=p_mw * math.tan(math.acos(cos_phi)) / sn_base,
        )
        # Like a static generator without a plant model, the user should know
        # the motor's own dynamics are not in the study.
        warnings.append(
            f"Motor '{el.get('userFriendlyName') or el.get('name')}' has no ANDES dynamic model here: "
            "modelled as a constant P/Q load."
        )

    # --- Shunts / capacitors ---
    sh_i = 0
    for _, el, typ in _iter_elements(in_data):
        is_shunt = typ.startswith("Shunt") or typ.startswith("Capacitor") or "Shunt Reactor" in typ
        if not is_shunt:
            continue
        bus = bus_map.get(el.get("bus"))
        if bus is None:
            continue
        sh_i += 1
        q_mvar = _sf(el.get("q_mvar"))
        # ANDES Shunt: g, b in pu (positive b = capacitive)
        b_pu = q_mvar / sn_base
        if "Reactor" in typ:
            b_pu = -abs(b_pu)
        u = 1 if _sb(el.get("in_service"), True) else 0
        ss.add(
            "Shunt",
            idx=f"Sh_{sh_i}",
            name=str(el.get("userFriendlyName") or el.get("name") or f"Sh_{sh_i}"),
            bus=bus,
            g=0.0,
            b=b_pu,
            Vn=bus_vn.get(bus, 110.0),
            u=u,
        )

    # --- External grids → Slack (no SynGen) ---
    slack_count = 0
    slack_v0: Dict[Any, float] = {}
    grid_slacks: Dict[str, str] = {}   # an External Grid's name and label -> its slack, for its outage
    grid_v0: Dict[str, float] = {}      # each External Grid's slack -> its set voltage
    for _, el, typ in _iter_elements(in_data):
        if not (typ.startswith("External Grid") or typ.startswith("ExternalGrid")):
            continue
        bus = bus_map.get(el.get("bus"))
        if bus is None:
            continue
        if not _sb(el.get("in_service"), True):
            continue
        slack_count += 1
        idx = f"Slack_EG_{slack_count}"
        vm = _sf(el.get("vm_pu"), 1.0)
        if vm <= 0:
            vm = 1.0
        if vm > 1.5:
            vn = bus_vn.get(bus, 0)
            if vn > 0:
                vm = vm / vn
        va = _sf(el.get("va_degree"), 0.0)
        slack_v0[bus] = vm
        grid_v0[idx] = vm
        for key in (el.get("name"), el.get("userFriendlyName")):
            if key:
                grid_slacks.setdefault(str(key), idx)
        ss.add(
            "Slack",
            idx=idx,
            name=str(el.get("userFriendlyName") or el.get("name") or idx),
            bus=bus,
            Vn=bus_vn.get(bus, 110.0),
            Sn=sn_base,
            v0=vm,
            a0=va * math.pi / 180.0,
            p0=0.0,
        )

    # An island no External Grid holds - its feeders' breakers open - is held, as in the load flow, by
    # its largest machine; with none it had no slack and the study refused it ("No slack bus found").
    # Its machines and grid-forming PCS start at the load flow's dispatch, each at its droop's share of
    # the island's load (their references reset to the rated frequency), not the slack with all of it.
    held = set(slack_v0)
    island_slacks, islanded = set(), set()
    for island in _islands(ss):
        if island & held:
            continue
        islanded |= island
        machines = [(_sf(el.get("sn_mva"), 0.0), el.get("name")) for _, el, typ in _iter_elements(in_data)
                    if typ.startswith("Generator") and "Asymmetric" not in typ and "1ph" not in typ
                    and _sb(el.get("in_service"), True) and bus_map.get(el.get("bus")) in island]
        if any(_sb(el.get("slack"), False) for _, el, typ in _iter_elements(in_data)
               if typ.startswith("Generator") and bus_map.get(el.get("bus")) in island):
            continue
        if not machines:
            # Without one, its largest grid-forming PCS by the power its droop takes (S / droop).
            machines = [(_sf(el.get("s_rated_mva"), 1.0) / max(_sf(el.get("droop_pf_percent"), 2.0), 1e-3),
                         el.get("name")) for _, el, typ in _iter_elements(in_data)
                        if typ.startswith("PCS") and str(el.get("control") or "").strip().lower() == "grid_forming"
                        and _sb(el.get("in_service"), True) and bus_map.get(el.get("bus")) in island]
        if machines:
            island_slacks.add(max(machines, key=lambda m: m[0])[1])
    dispatch: Dict[str, float] = {}
    if island_slacks:
        if flow is None and flow_problem is None:
            flow, flow_problem = _electrisim_flow(in_data, freq)
        if flow is not None:
            dispatch = {str(n): float(p) for n, p, b in zip(flow.gen["name"], flow.res_gen["p_mw"], flow.gen["bus"])
                        if np.isfinite(p) and bus_map.get(str(flow.bus.at[int(b), "name"])) in islanded}
        else:
            warnings.append(f"The island's machines start at their set points: its load flow failed ({flow_problem}).")

    # --- Generators → PV/Slack + SynGen + Exciter + Governor ---
    gen_count = 0
    for _, el, typ in _iter_elements(in_data):
        if not typ.startswith("Generator") or typ.startswith("Static Generator") or typ.startswith("Wind Turbine"):
            continue
        if "Asymmetric" in typ or "1ph" in typ:
            continue
        bus = bus_map.get(el.get("bus"))
        if bus is None:
            warnings.append(f"Skipped Generator '{el.get('userFriendlyName', el.get('name'))}': not connected.")
            continue
        if not _sb(el.get("in_service"), True):
            continue

        gen_count += 1
        name = el.get("name")
        ufname = str(el.get("userFriendlyName") or name or f"Gen_{gen_count}")
        p_mw = dispatch.get(str(name), _sf(el.get("p_mw")) * _sf(el.get("scaling"), 1.0))
        vm = _sf(el.get("vm_pu"), 1.0)
        if vm <= 0:
            vm = 1.0
        if vm > 1.5:
            vn_bus = bus_vn.get(bus, 0)
            if vn_bus > 0:
                vm = vm / vn_bus
        sn_mva = _sf(el.get("sn_mva"), 0.0)
        if sn_mva <= 0:
            sn_mva = max(abs(p_mw) * 1.25, 100.0)
            defaults_applied.append(f"Generator '{ufname}': sn_mva missing/zero, used {sn_mva:.3g} MVA.")
        vn = _sf(el.get("vn_kv"), 0.0)
        bus_v = bus_vn.get(bus, 20.0)
        # Prefer bus nominal voltage when machine vn is missing or clearly mismatched
        # (common Electrisim diagrams leave vn_kv=0 or set generator terminal kV without a unit trafo).
        if vn <= 0:
            vn = bus_v
        elif bus_v > 0 and abs(vn - bus_v) / bus_v > 0.5:
            defaults_applied.append(
                f"Generator '{ufname}': vn_kv={vn} differs from bus {bus_v} kV; "
                f"using bus voltage for ANDES SynGen (add an explicit transformer for step-up)."
            )
            vn = bus_v

        is_slack = _sb(el.get("slack"), False) or name in island_slacks
        static_idx = f"{'Slack' if is_slack else 'PV'}_G_{gen_count}"
        static_model = "Slack" if is_slack else "PV"
        static_kw = dict(
            idx=static_idx,
            name=ufname,
            bus=bus,
            Vn=vn,
            Sn=sn_mva,
            p0=p_mw / sn_base,
            v0=vm,
            q0=0.0,
        )
        if is_slack:
            static_kw["a0"] = 0.0
            if slack_count == 0 and gen_count == 1:
                pass
        ss.add(static_model, **static_kw)
        if is_slack:
            slack_count += 1

        # Machine model
        machine = (el.get("dyn_machine_model") or el.get("machine_model") or "GENROU").strip().upper()
        if machine not in ("GENROU", "GENCLS"):
            machine = "GENROU"
            defaults_applied.append(f"Generator '{ufname}': unknown machine model, used GENROU.")

        syn_idx = f"{machine}_{gen_count}"
        used_defaults = False

        def _dyn(key: str, default: float, aliases: Tuple[str, ...] = ()) -> float:
            nonlocal used_defaults
            for k in (key,) + aliases:
                if el.get(k) is not None and str(el.get(k)).strip() not in ("", "none", "null"):
                    return _sf(el.get(k), default)
            used_defaults = True
            return default

        if machine == "GENCLS":
            M = _dyn("dyn_M", _DEFAULT_GENROU["M"], ("M", "H"))
            # If user provided H instead of M and M missing, M=2H
            if el.get("dyn_H") is not None and el.get("dyn_M") in (None, "", "null"):
                M = 2.0 * _sf(el.get("dyn_H"), M / 2.0)
                used_defaults = False
            D = _dyn("dyn_D", 0.0, ("D",))
            ra = _dyn("dyn_ra", 0.0, ("ra",))
            # The classical model's reactance is the transient X'd; xdss_pu is
            # the subtransient X''d of the short-circuit data, and made the
            # machine too stiff (1.94 Hz against 1.53 Hz on the reference grid).
            xd1 = _dyn("dyn_xd1", 0.3, ("xd1",))
            ss.add(
                "GENCLS",
                idx=syn_idx,
                name=ufname,
                bus=bus,
                gen=static_idx,
                Sn=sn_mva,
                Vn=vn,
                fn=freq,
                M=M,
                D=D,
                ra=ra,
                xd1=xd1,
            )
        else:
            M = _dyn("dyn_M", _DEFAULT_GENROU["M"], ("M",))
            if el.get("dyn_H") is not None and el.get("dyn_M") in (None, "", "null"):
                M = 2.0 * _sf(el.get("dyn_H"), M / 2.0)
                used_defaults = False
            kw = dict(
                idx=syn_idx,
                name=ufname,
                bus=bus,
                gen=static_idx,
                Sn=sn_mva,
                Vn=vn,
                fn=freq,
                M=M,
                D=_dyn("dyn_D", _DEFAULT_GENROU["D"], ("D",)),
                ra=_dyn("dyn_ra", _DEFAULT_GENROU["ra"], ("ra",)),
                xl=_dyn("dyn_xl", _DEFAULT_GENROU["xl"], ("xl",)),
                xd=_dyn("dyn_xd", _DEFAULT_GENROU["xd"], ("xd",)),
                xq=_dyn("dyn_xq", _DEFAULT_GENROU["xq"], ("xq",)),
                xd1=_dyn("dyn_xd1", _DEFAULT_GENROU["xd1"], ("xd1",)),
                xq1=_dyn("dyn_xq1", _DEFAULT_GENROU["xq1"], ("xq1",)),
                xd2=_dyn("dyn_xd2", _DEFAULT_GENROU["xd2"], ("xd2", "xdss_pu")),
                Td10=_dyn("dyn_Td10", _DEFAULT_GENROU["Td10"], ("Td10",)),
                Td20=_dyn("dyn_Td20", _DEFAULT_GENROU["Td20"], ("Td20",)),
                Tq10=_dyn("dyn_Tq10", _DEFAULT_GENROU["Tq10"], ("Tq10",)),
                Tq20=_dyn("dyn_Tq20", _DEFAULT_GENROU["Tq20"], ("Tq20",)),
            )
            # ANDES's GENROU has no subtransient saliency: with xq2 != xd2 it
            # cannot initialise. xd2 can come from the short-circuit xdss_pu,
            # so a fixed xq2 default broke every generator that had one.
            xq2_given = _sf(el.get("dyn_xq2", el.get("xq2")), 0.0)
            if xq2_given > 0 and abs(xq2_given - kw["xd2"]) > 1e-9:
                defaults_applied.append(
                    f"Generator '{ufname}': GENROU needs xq'' = xd'', so xq2={xq2_given:g} "
                    f"was replaced by xd2={kw['xd2']:g}."
                )
            kw["xq2"] = kw["xd2"]
            ss.add("GENROU", **kw)

        if used_defaults:
            defaults_applied.append(f"Generator '{ufname}': applied default {machine} parameters.")

        # Exciter. The model-specific dictionaries intentionally expose only key
        # parameters; all other ANDES parameters retain their library defaults.
        exc_model = (el.get("dyn_exciter_model") or "EXDC2").strip().upper()
        if exc_model in ("", "NONE", "OFF"):
            defaults_applied.append(f"Generator '{ufname}': no exciter.")
            exc_idx = None
        else:
            if exc_model not in _EXCITER_DEFAULTS:
                exc_model = "EXDC2"
                defaults_applied.append(f"Generator '{ufname}': unknown exciter, used EXDC2.")
            exc_idx = f"{exc_model}_{gen_count}"
            exc_idx = _add_model_safe(
                ss, exc_model, defaults_applied, f"Generator '{ufname}'",
                idx=exc_idx, name=f"{exc_model}_{ufname}", syn=syn_idx,
                **_model_kwargs(el, "dyn_exc_", _EXCITER_DEFAULTS[exc_model]),
            )
            if all(el.get(k) in (None, "", "null") for k in ("dyn_exciter_model", "dyn_exc_KA", "dyn_exc_K")):
                defaults_applied.append(f"Generator '{ufname}': applied default {exc_model} exciter.")

        # Governor
        gov_model = (el.get("dyn_governor_model") or "TGOV1").strip().upper()
        if gov_model in ("", "NONE", "OFF"):
            gov_idx = None
            defaults_applied.append(f"Generator '{ufname}': no governor.")
        else:
            if gov_model not in _GOVERNOR_DEFAULTS:
                gov_model = "TGOV1"
                defaults_applied.append(f"Generator '{ufname}': unknown governor, used TGOV1.")
            gov_idx = _add_model_safe(
                ss, gov_model, defaults_applied, f"Generator '{ufname}'",
                idx=f"{gov_model}_{gen_count}", name=f"{gov_model}_{ufname}", syn=syn_idx,
                **_model_kwargs(el, "dyn_gov_", _GOVERNOR_DEFAULTS[gov_model]),
            )
            if all(el.get(k) in (None, "", "null") for k in ("dyn_governor_model", "dyn_gov_R")):
                defaults_applied.append(f"Generator '{ufname}': applied default {gov_model} governor.")

        pss_model = (el.get("dyn_pss_model") or "NONE").strip().upper()
        pss_idx = None
        if pss_model == "IEEEST":
            pss_idx = _add_model_safe(
                ss, "IEEEST", defaults_applied, f"Generator '{ufname}'",
                idx=f"IEEEST_{gen_count}", name=f"IEEEST_{ufname}", syn=syn_idx,
                **_model_kwargs(el, "dyn_pss_", _PSS_DEFAULTS["IEEEST"]),
            )
        elif pss_model not in ("", "NONE", "OFF"):
            defaults_applied.append(f"Generator '{ufname}': unknown PSS '{pss_model}', omitted.")

        gen_map[name] = {
            "static_idx": static_idx,
            "syn_idx": syn_idx,
            "machine": machine,
            "exc_idx": exc_idx,
            "gov_idx": gov_idx,
            "pss_idx": pss_idx,
            "name": ufname,
            "bus": bus,
        }

    # Static generators can supply renewable / inverter dynamics. They remain
    # ANDES PV devices: unlike synchronous machines they deliberately have no SynGen.
    renewable_count = 0
    static_count = 0
    for _, el, typ in _iter_elements(in_data):
        if not (typ.startswith("Static Generator") or typ.startswith("Wind Turbine")):
            continue
        bus = bus_map.get(el.get("bus"))
        if bus is None or not _sb(el.get("in_service"), True):
            continue
        # Wind Turbine: derive p_mw from power curve when present
        if typ.startswith("Wind Turbine"):
            raw = el.get("wind_power_curve_json")
            if raw is not None and (not isinstance(raw, str) or str(raw).strip()):
                try:
                    points = json.loads(raw) if isinstance(raw, str) else raw
                    if isinstance(points, list) and len(points) >= 2:
                        v = float(el.get("wind_speed_ms"))
                        knots = sorted(
                            ((float(pt["v_ms"]), float(pt["p_mw"])) for pt in points if isinstance(pt, dict)),
                            key=lambda x: x[0],
                        )
                        if len(knots) >= 2:
                            if v <= knots[0][0]:
                                el["p_mw"] = knots[0][1]
                            elif v >= knots[-1][0]:
                                el["p_mw"] = knots[-1][1]
                            else:
                                for i in range(len(knots) - 1):
                                    v0, p0 = knots[i]
                                    v1, p1 = knots[i + 1]
                                    if v0 <= v <= v1:
                                        span = v1 - v0
                                        el["p_mw"] = p0 if abs(span) < 1e-12 else p0 + (v - v0) / span * (p1 - p0)
                                        break
                except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                    pass
        plant_kind = (el.get("dyn_plant_kind") or "NONE").strip().upper()
        if plant_kind in ("", "NONE", "OFF"):
            # Still part of the power flow, as pandapower has it: a fixed P/Q
            # injection. Dropping it took its output out of the network.
            ufname = str(el.get("userFriendlyName") or el.get("name"))
            scaling = _sf(el.get("scaling"), 1.0)
            pq_i += 1
            ss.add(
                "PQ", idx=f"PQ_{pq_i}", name=ufname, bus=bus, Vn=bus_vn.get(bus, 110.0),
                p0=-_sf(el.get("p_mw")) * scaling / sn_base,
                q0=-_sf(el.get("q_mvar")) * scaling / sn_base,
            )
            warnings.append(
                f"Static Generator '{ufname}' has no ANDES dynamic plant model: "
                "modelled as a fixed P/Q injection."
            )
            continue
        if plant_kind not in ("IBR", "WIND", "PVD1", "ESD1"):
            defaults_applied.append(
                f"Static Generator '{el.get('userFriendlyName', el.get('name'))}': "
                f"unknown plant kind '{plant_kind}', omitted."
            )
            continue

        static_count += 1
        name = el.get("name") or f"SGen_{static_count}"
        ufname = str(el.get("userFriendlyName") or name)
        sn_mva = _sf(el.get("dyn_Sn", el.get("sn_mva")), 0.0)
        if sn_mva <= 0:
            sn_mva = max(abs(_sf(el.get("p_mw")) * _sf(el.get("scaling"), 1.0)) * 1.25, 100.0)
            defaults_applied.append(f"Static Generator '{ufname}': dyn_Sn/sn_mva missing, used {sn_mva:.3g} MVA.")
        static_idx = f"PV_SG_{static_count}"
        ss.add(
            "PV", idx=static_idx, name=ufname, bus=bus, Vn=bus_vn.get(bus, 110.0),
            Sn=sn_mva, p0=_sf(el.get("p_mw")) * _sf(el.get("scaling"), 1.0) / sn_base,
            q0=_sf(el.get("q_mvar")) * _sf(el.get("scaling"), 1.0) / sn_base, v0=1.0,
        )

        model_ids: Dict[str, Optional[str]] = {}
        if plant_kind in ("IBR", "WIND"):
            reg_idx = _add_model_safe(
                ss, "REGCA1", defaults_applied, f"Static Generator '{ufname}'",
                idx=f"REGCA1_{static_count}", name=f"REGCA1_{ufname}", bus=bus, gen=static_idx, Sn=sn_mva,
                **_model_kwargs(el, "dyn_reg_", _RENEWABLE_DEFAULTS["REGCA1"]),
            )
            model_ids["reg_idx"] = reg_idx
            if reg_idx:
                ree_idx = _add_model_safe(
                    ss, "REECA1", defaults_applied, f"Static Generator '{ufname}'",
                    idx=f"REECA1_{static_count}", name=f"REECA1_{ufname}", reg=reg_idx,
                    **_model_kwargs(el, "dyn_ree_", _RENEWABLE_DEFAULTS["REECA1"]),
                    **dict(_RENEWABLE_FLAGS["REECA1"], PFLAG=int(plant_kind == "WIND")),
                )
                model_ids["ree_idx"] = ree_idx
                if ree_idx:
                    model_ids["repca_idx"] = _add_model_safe(
                        ss, "REPCA1", defaults_applied, f"Static Generator '{ufname}'",
                        idx=f"REPCA1_{static_count}", name=f"REPCA1_{ufname}", ree=ree_idx,
                        **_model_kwargs(el, "dyn_repca_", _RENEWABLE_DEFAULTS["REPCA1"]), **_RENEWABLE_FLAGS["REPCA1"],
                    )
                    if plant_kind == "WIND":
                        wt_idx = _add_model_safe(
                            ss, "WTDTA1", defaults_applied, f"Static Generator '{ufname}'",
                            idx=f"WTDTA1_{static_count}", name=f"WTDTA1_{ufname}", ree=ree_idx,
                            **_model_kwargs(el, "dyn_wt_", _RENEWABLE_DEFAULTS["WTDTA1"]),
                        )
                        model_ids["wtdta_idx"] = wt_idx
                        if wt_idx:
                            rea_idx = _add_model_safe(
                                ss, "WTARA1", defaults_applied, f"Static Generator '{ufname}'",
                                idx=f"WTARA1_{static_count}", name=f"WTARA1_{ufname}", rego=wt_idx,
                                **_model_kwargs(el, "dyn_wta_", _RENEWABLE_DEFAULTS["WTARA1"]),
                            )
                            model_ids["wtara_idx"] = rea_idx
                            if rea_idx:
                                pitch_idx = _add_model_safe(
                                    ss, "WTPTA1", defaults_applied, f"Static Generator '{ufname}'",
                                    idx=f"WTPTA1_{static_count}", name=f"WTPTA1_{ufname}", rea=rea_idx,
                                    **_model_kwargs(el, "dyn_wtp_", _RENEWABLE_DEFAULTS["WTPTA1"]),
                                )
                                model_ids["wtpta_idx"] = pitch_idx
                                if pitch_idx:
                                    model_ids["wttqa_idx"] = _add_model_safe(
                                        ss, "WTTQA1", defaults_applied, f"Static Generator '{ufname}'",
                                        idx=f"WTTQA1_{static_count}", name=f"WTTQA1_{ufname}", rep=pitch_idx,
                                        **_model_kwargs(el, "dyn_wtt_", _RENEWABLE_DEFAULTS["WTTQA1"]), **_RENEWABLE_FLAGS["WTTQA1"],
                                    )
        else:
            dg_model = plant_kind
            # The dialog's DG Tg is the converter's current lag, active (tip) and reactive (tiq)
            # alike. A lag of zero would leave its state undefined, so the model's own stands.
            tg = _dyn_value(el, "dyn_dg_Tg", 0.0)
            lag = {"tip": tg, "tiq": tg} if tg > 0 else {}
            if tg <= 0 and str(el.get("dyn_dg_Tg") or "").strip().lower() not in ("", "none", "null"):
                defaults_applied.append(
                    f"Static Generator '{ufname}': DG Tg must be positive; used the {dg_model} default."
                )
            model_ids["dg_idx"] = _add_model_safe(
                ss, dg_model, defaults_applied, f"Static Generator '{ufname}'",
                idx=f"{dg_model}_{static_count}", name=f"{dg_model}_{ufname}", bus=bus, gen=static_idx, Sn=sn_mva,
                **_model_kwargs(el, "dyn_dg_", _RENEWABLE_DEFAULTS[dg_model]), **_RENEWABLE_FLAGS[dg_model],
                **lag,
            )

        if any(model_ids.values()):
            renewable_count += 1
        gen_map[name] = {
            "static_idx": static_idx, "syn_idx": None, "plant_kind": plant_kind,
            "name": ufname, "bus": bus, **model_ids,
        }

    # --- PCS ---
    pcs_count = 0
    pcs_plants = _pcs_plants(in_data, warnings)
    if any(r["control"] == "grid_following" for r in pcs_plants):
        # Its Q as set: its static generator a PV bus held at that Q, which the power flow turns to PQ.
        ss.PV.config.pv2pq = 1
        ss.PV.qlim.enable = True
    for rec in pcs_plants:
        bus = bus_map.get(rec["bus"])
        if bus is None:
            warnings.append(f"PCS '{rec['label']}' needs an AC bus on its AC side, so it is left out.")
            continue
        rec["p_ac"] = dispatch.get(str(rec["name"]), rec["p_ac"])
        pcs_count += 1
        # An island's slack when it holds the island alone (its virtual machine then holds it).
        holds = str(rec["name"]) in island_slacks
        static_idx = f"{'Slack' if holds else 'PV'}_PCS_{pcs_count}"
        q0 = rec["q"] / sn_base
        kw = dict(idx=static_idx, name=rec["label"], bus=bus, Vn=bus_vn.get(bus, 110.0), Sn=rec["s_rated"],
                  p0=rec["p_ac"] / sn_base, q0=q0)
        if rec["control"] == "grid_forming":
            # On an External Grid's bus it takes the grid's voltage, as in the load flow.
            kw["v0"] = slack_v0.get(bus, rec["vm_set_pu"])
        else:
            kw.update(v0=1.0, qmax=q0, qmin=q0)
        if holds:
            kw["a0"] = 0.0
            slack_count += 1
        ss.add("Slack" if holds else "PV", **kw)
        ids = _add_pcs_dynamics(ss, rec, pcs_count, bus, static_idx, freq, sn_base, defaults_applied)
        if ids.get("model_idx"):
            renewable_count += 1
        gen_map[str(rec["name"])] = {
            "static_idx": static_idx, "syn_idx": None, "plant_kind": "PCS", "name": rec["label"], "bus": bus,
            "pcs": True, "id": rec["id"], "control": rec["control"], "source_kind": rec["source"]["kind"],
            "source": rec["source"]["label"], "s_rated_mva": rec["s_rated"], "p_mw": rec["p_ac"],
            "q_mvar": rec["q"], "p_max_mw": rec["p_max_ac"], "p_min_mw": rec["p_min_ac"],
            "vm_set_pu": rec["vm_set_pu"], "droop_qv": rec["droop_qv"], "q_mode": rec["q_mode"],
            "qv_droop": rec["qv_droop"], "q_max_mvar": rec["q_max"], "on_slack_bus": bus in slack_v0 or holds, **ids,
        }

    if gen_count + renewable_count == 0:
        raise ValueError(
            "Transient / eigenvalue analysis requires at least one synchronous Generator "
            "or renewable dynamic plant. External Grid alone is not sufficient."
        )

    if slack_count == 0:
        # Promote first PV to Slack for power flow
        # Find first PV and convert by adding a Slack with same setpoints is hard post-add;
        # instead require a slack — auto-fix first generator as Slack was already handled if slack flag set.
        # If still no slack, rebuild is too costly; raise clear error.
        raise ValueError(
            "No slack bus found. Mark one Generator as slack=true or add an External Grid."
        )

    # --- Disturbances (TDS) ---
    fault_bus_name = params.get("fault_bus") or params.get("fault_bus_name") or ""
    fault_enabled = _sb(params.get("fault_enabled"), True) if params.get("fault_enabled") is not None else bool(fault_bus_name)
    if fault_enabled and fault_bus_name:
        fbus = bus_map.get(fault_bus_name)
        if fbus is None:
            # try friendly name match
            for bname, bidx in bus_map.items():
                if str(friendly.get(bname, bname)) == str(fault_bus_name) or str(bname) == str(fault_bus_name):
                    fbus = bidx
                    break
        if fbus is not None:
            slack_buses = {str(b) for b in getattr(ss.Slack.bus, "v", [])}
            if str(fbus) in slack_buses:
                warnings.append(
                    f"The fault is at '{friendly.get(fault_bus_name, fault_bus_name)}', which holds the "
                    "External Grid: ANDES keeps that bus at its set voltage, so the fault has no effect. "
                    "Choose another bus.")
            ss.add(
                "Fault",
                idx="Fault_1",
                bus=fbus,
                tf=_sf(params.get("fault_tf"), 1.0),
                tc=_sf(params.get("fault_tc"), 1.1),
                xf=_sf(params.get("fault_xf"), 0.0001),
                rf=_sf(params.get("fault_rf"), 0.0),
            )
        else:
            warnings.append(f"Fault bus '{fault_bus_name}' not found; no Fault applied.")

    islanded_after = None
    running_islands: List[set] = []
    toggle_line = params.get("toggle_line") or params.get("line_outage") or ""
    toggle_t = _sf(params.get("toggle_t"), 2.0)
    if toggle_line:
        lidx = line_map.get(toggle_line)
        if lidx is None:
            for lname, lid in line_map.items():
                if str(lname) == str(toggle_line):
                    lidx = lid
                    break
        # The outage of a line, a transformer or a breaker's element - or of an External Grid:
        # the utility lost, its slack switched off.
        grid = grid_slacks.get(toggle_line) if lidx is None else None
        if lidx is not None and not ss.Line.u.v[list(ss.Line.idx.v).index(lidx)]:
            warnings.append(f"'{toggle_line}' is already out of service or its breaker open; no outage applied.")
        elif lidx is not None or grid is not None:
            if grid is not None:
                ss.add("Toggle", idx="Toggle_1", model="Slack", dev=grid, t=toggle_t)
                label = str(ss.Slack.name.v[list(ss.Slack.idx.v).index(grid)])
            else:
                ss.add("Toggle", idx="Toggle_1", model="Line", dev=lidx, t=toggle_t)
                label = str(ss.Line.name.v[list(ss.Line.idx.v).index(lidx)])
            grid_forming = {g["bus"] for g in gen_map.values() if g.get("pcs") and g.get("control") == "grid_forming"}
            island, tripped, running = _deenergise_island(
                ss, {lidx} - {None}, toggle_t, out_slacks={grid} - {None}, grid_forming=grid_forming)
            if island:
                names = [str(bus_name_by_idx.get(b, b)) for b in island if b in bus_name_by_idx]
                warnings.append(
                    f"Taking '{label}' out at {toggle_t:g} s cuts {', '.join(names)} off from the "
                    "External Grid: de-energised from then on"
                    + (f", {', '.join(tripped)} tripped (loss of mains)." if tripped else "."))
                syn_total = sum(getattr(ss, m).n for m in ("GENROU", "GENCLS"))
                islanded_after = (toggle_t, [str(b) for b in island], len(tripped) == syn_total)
            running_islands += running
            for part in running:
                names = [str(bus_name_by_idx[b]) for b in ss.Bus.idx.v if b in part and b in bus_name_by_idx]
                warnings.append(
                    f"Taking '{label}' out at {toggle_t:g} s leaves {len(names)} buses as an island "
                    f"({', '.join(names[:4])}{', ...' if len(names) > 4 else ''}): its machines' governors "
                    "and its grid-forming PCS share its load by their droops.")
        else:
            warnings.append(f"Line outage target '{toggle_line}' not found; no Toggle applied.")

    toggle_gen = params.get("toggle_gen") or params.get("generator_trip") or ""
    toggle_gen_t = _sf(params.get("toggle_gen_t"), toggle_t)
    if toggle_gen:
        syn_dev = None
        gm = gen_map.get(toggle_gen)
        if gm and gm.get("syn_idx"):
            syn_dev = gm["syn_idx"]
        else:
            for gname, ginfo in gen_map.items():
                if str(gname) == str(toggle_gen) or str(ginfo.get("name")) == str(toggle_gen):
                    syn_dev = ginfo.get("syn_idx")
                    break
        if syn_dev:
            ss.add("Toggle", idx="Toggle_Gen_1", model="SynGen", dev=syn_dev, t=toggle_gen_t)
        else:
            warnings.append(f"Generator trip target '{toggle_gen}' not found; no Toggle applied.")

    # The grid's voltage following a profile (IEEE 2800's ride-through envelope, or a table):
    # each step an Alter of its slack's set voltage, which its reactive power holds the bus at.
    points, problems = gvp.profile_points(params)
    warnings.extend(problems)
    if points:
        target = str(params.get("grid_voltage_target") or "").strip()
        slacks = [grid_slacks[target]] if target in grid_slacks else list(grid_v0)
        if target and target not in grid_slacks:
            warnings.append(f"Grid voltage profile: no External Grid '{target}'; every one follows it.")
        start = _sf(params.get("grid_voltage_start_s"), 1.0)
        if any(v < GRID_PROFILE_V_FLOOR for _, v in points):
            # A phasor model has no solution at no voltage: its converters draw P / V.
            warnings.append(f"Grid voltage profile: ANDES takes its voltages below {GRID_PROFILE_V_FLOOR:g} pu "
                            f"as {GRID_PROFILE_V_FLOOR:g} pu (the EMT study applies them as given).")
            points = [(t, max(v, GRID_PROFILE_V_FLOOR)) for t, v in points]
        n_alter = 0
        for slack in slacks:
            for t_rel, v in points:
                n_alter += 1
                ss.add("Alter", idx=f"Alter_GV_{n_alter}", t=start + t_rel, model="Slack", dev=slack,
                       src="v0", attr="v", method="=", amount=v * grid_v0[slack])
        if slacks:
            warnings.append(gvp.describe(points, start))
        else:
            warnings.append("Grid voltage profile: the network has no External Grid to apply it to.")

    if setup:
        ss.setup()

    meta = {
        "bus_map": bus_map,
        "bus_name_by_idx": {str(k): v for k, v in bus_name_by_idx.items()},
        "line_map": line_map,
        "gen_map": gen_map,
        "defaults_applied": defaults_applied,
        "warnings": warnings,
        "frequency": freq,
        "sn_mva": sn_base,
        "n_generators": gen_count,
        "n_renewable_plants": renewable_count,
        "n_pcs": pcs_count,
        "n_buses": len(bus_name_by_idx),
        "islanded_after": islanded_after,
        "free_islands": bool(island_slacks or running_islands),
        "profiled_loads": profiled_loads,
        "profile_repeat": profile_repeat,
    }
    return ss, meta


def _pcs_series(ss, meta: Dict[str, Any], idx: np.ndarray) -> List[Dict[str, Any]]:
    """
    Each PCS through the run: its power and Q (MW, Mvar) at its AC side; a
    battery's state of charge, a flywheel's speed; a grid-forming PCS's
    frequency. ESD1 and PVD1 inject their currents into their bus's voltage.
    """
    sn, out = meta["sn_mva"], []
    v_bus = tds_values(ss, ss.Bus.v) if ss.Bus.n else None
    for name, g in meta["gen_map"].items():
        if not g.get("pcs") or not g.get("model_idx"):
            continue
        model = getattr(ss, g["model"])
        try:
            k = list(model.idx.v).index(g["model_idx"])
        except ValueError:
            continue
        col = lambda var: tds_values(ss, var)[:, k]
        row: Dict[str, Any] = {
            "name": name, "id": g.get("id", ""), "label": g["name"], "control": g["control"], "model": g["model"],
            "source": g["source"], "source_kind": g["source_kind"], "s_rated_mva": g["s_rated_mva"],
            "p_max_mw": g["p_max_mw"], "p_min_mw": g["p_min_mw"],
        }
        if g["model"] in ("ESD1", "PVD1"):
            v = v_bus[:, ss.Bus.idx2uid(g["bus"])]
            p, q = col(model.Ipout_y) * v, col(model.Iqout_y) * v
        else:
            p, q = col(model.Pe), col(model.Qe)
        row["p_mw"] = [_clean_num(float(x) * sn) for x in p[idx]]
        row["q_mvar"] = [_clean_num(float(x) * sn) for x in q[idx]]
        if g["model"] == "ESD1":
            soc = col(model.pIG_y)[idx]
            if g["source_kind"] == "Flywheel":
                row["speed_percent"] = [_clean_num(100.0 * math.sqrt(max(float(x), 0.0))) for x in soc]
            else:
                row["soc_percent"] = [_clean_num(100.0 * float(x)) for x in soc]
        if g["model"] == "GENCLS":
            row["frequency_hz"] = [_clean_num(float(x) * meta["frequency"]) for x in col(model.omega)[idx]]
        out.append(row)
    return out


def _extract_syn_series(ss, model_name: str, var_name: str, names: List[str]) -> List[Dict[str, Any]]:
    if not hasattr(ss, model_name):
        return []
    model = getattr(ss, model_name)
    if model.n == 0 or not hasattr(model, var_name):
        return []
    var = getattr(model, var_name)
    addrs = list(var.a)
    if not addrs:
        return []
    try:
        values = tds_values(ss, var)
    except Exception:
        return []
    series = []
    for i, addr in enumerate(addrs):
        col = values[:, i] if values.ndim == 2 else values
        label = names[i] if i < len(names) else f"{model_name}_{i}"
        try:
            label = str(model.name.v[i]) if hasattr(model, "name") else label
        except Exception:
            pass
        series.append({"id": str(model.idx.v[i]), "name": label, "values": [_clean_num(x) for x in col.tolist()]})
    return series


def _angle_references(ss, idx: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Each machine's angle reference through the run, by its idx: none (0) in
    the part an External Grid holds - its slack bus keeps its angle - else
    the mean angle of the machines and grid-forming PCS in its island, the
    islands as they are at the run's end; None for a machine tripped.
    """
    attached = {g for m in ("GENROU", "GENCLS") for g in getattr(ss, m).gen.v}
    fixed = {b for i, b, u in zip(ss.Slack.idx.v, ss.Slack.bus.v, ss.Slack.u.v) if u and i not in attached}
    machines, tripped = [], set()
    for m in ("GENROU", "GENCLS"):
        mdl = getattr(ss, m)
        if mdl.n:
            values = tds_values(ss, mdl.delta)
            values = values if values.ndim == 2 else values[:, None]
            machines += [(str(mdl.idx.v[k]), mdl.bus.v[k], values[idx, k]) for k in range(mdl.n) if mdl.u.v[k]]
            tripped |= {str(mdl.idx.v[k]) for k in range(mdl.n) if not mdl.u.v[k]}
    out = dict.fromkeys(tripped, None)
    for part in _islands(ss):
        group = [(i, d) for i, b, d in machines if b in part]
        if not group or part & fixed:
            continue
        mean = np.mean([d for _, d in group], axis=0)
        out.update({i: mean for i, _ in group})
    return out


def _syn_names(ss) -> Tuple[List[str], List[str]]:
    """Return (model_list for each machine instance flattened is hard) — names per GENROU then GENCLS."""
    names = []
    models = []
    for mname in ("GENROU", "GENCLS"):
        if hasattr(ss, mname) and getattr(ss, mname).n > 0:
            m = getattr(ss, mname)
            for i in range(m.n):
                try:
                    names.append(str(m.name.v[i]))
                except Exception:
                    names.append(f"{mname}_{i}")
                models.append(mname)
    return names, models


def _restore_prefault_state_on_clearing(ss) -> None:
    """
    At fault clearance, restart the algebraic solution from the full pre-fault
    state - bus angles included.

    ANDES (2.0) restores the pre-fault voltages and other algebraic variables
    but not the bus angles. During a bolted fault the buses it collapses sit
    near 0 V, where an angle means nothing, and drift (F1 and its ring to
    -30 deg on the transmission reference grid); restarted at full voltage on
    those angles, the post-fault solve fell back to 0 V and stayed there, as
    if the fault were never cleared, or failed outright ("time step reduced
    to zero") for a fault at the generator's bus. Restored whole, a cleared
    fault recovers, and the critical clearing time of a classical machine
    agrees with the equal-area criterion (0.467 s against 0.463 s).

    The callbacks are bound when the Fault model is built, so they are
    replaced on the timer parameters, not the methods.
    """
    fault = getattr(ss, "Fault", None)
    if fault is None or not getattr(fault, "n", 0):
        return
    apply_fault, clear_fault = fault.apply_fault, fault.clear_fault

    def apply(is_time):
        acted = apply_fault(is_time)
        if acted:
            fault._electrisim_prefault_y = np.array(ss.dae.y)
        return acted

    def clear(is_time):
        acted = clear_fault(is_time)
        stored = getattr(fault, "_electrisim_prefault_y", None)
        if acted and stored is not None and len(stored) == len(ss.dae.y):
            ss.dae.y[:] = stored
        return acted

    fault.tf.callback = apply
    fault.tc.callback = clear


def _lp_sample(load: Dict[str, Any], times, repeat: bool):
    import load_profiles_electrisim as _lp
    return _lp.sample_profile(load["t"], load["p"], times, repeat)


# At most this many pieces when loads follow a profile: each restarts the solver.
_PROFILE_MAX_PIECES = 2000


def _run_tds_following_profiles(ss, profiled: List[Dict[str, Any]], tf: float, repeat: bool,
                                warnings: List[str]) -> bool:
    """
    Run the time-domain simulation in short pieces, setting each profiled
    load's power before each piece.

    ANDES fixes a PQ load's power when the simulation starts: Ppf, and the
    current and impedance it converts to, Ipeq and Req (Qpf, Iqeq, Xeq for Q).
    Altering p0 during the run does nothing. A load following a profile has
    all three rescaled to the profile's mean over each piece, so it keeps the
    voltage dependence ANDES gives every load and only its size follows the
    profile.
    """
    import load_profiles_electrisim as _lp
    try:
        ss.TDS.config.no_tqdm = True
    except Exception:
        pass
    position = {idx: i for i, idx in enumerate(ss.PQ.idx.v)}
    for load in profiled:
        load["i"] = position[load["idx"]]
        # The bus voltage the services were computed at.
        load["v0"] = float(ss.PQ.v.v[load["i"]]) or 1.0
    finest = min(float(np.median(np.diff(load["t"]))) for load in profiled)
    step = max(finest, tf / _PROFILE_MAX_PIECES)
    if step > finest * (1 + 1e-9):
        warnings.append(f"Load profiles applied in {step:.4g} s pieces, coarser than their "
                        f"{finest:g} s samples, to keep the run to {_PROFILE_MAX_PIECES} pieces; "
                        "each piece uses the profile's mean over it.")
    t0 = 0.0
    while t0 < tf - 1e-12:
        t1 = min(t0 + step, tf)
        for load in profiled:
            f = _lp.average_profile(load["t"], load["p"], t0, t1, repeat)
            i, v0 = load["i"], load["v0"]
            p = load["p_rated"] * f
            ss.PQ.Ppf.v[i], ss.PQ.Ipeq.v[i], ss.PQ.Req.v[i] = p, p / v0, p / v0 ** 2
            if load["q_mode"] == "pf":
                q = load["q_rated"] * f
                ss.PQ.Qpf.v[i], ss.PQ.Iqeq.v[i], ss.PQ.Xeq.v[i] = q, q / v0, q / v0 ** 2
        ss.TDS.config.tf = t1
        if not ss.TDS.run():
            return False
        t0 = t1
    return True


def run_tds(in_data: Dict[str, Any], params: Dict[str, Any]) -> str:
    """Run power flow + time-domain simulation; return JSON string."""
    try:
        if not _HAS_ANDES:
            return json.dumps({
                "error": True,
                "message": "ANDES is not installed on the backend.",
                "exception": "pip install andes",
            })

        andes.config_logger(stream_level=40)
        ss, meta = build_system(in_data, params)

        # Suppress file outputs
        try:
            ss.TDS.config.noprint = True
        except Exception:
            pass

        pf_ok = bool(ss.PFlow.run()) and _settle_pcs_set_points(ss, meta)
        if not pf_ok:
            return json.dumps({
                "error": True,
                "message": "Power flow did not converge. Check network data and generator setpoints.",
                "defaults_applied": meta["defaults_applied"],
                "warnings": meta["warnings"],
            })

        tf = _sf(params.get("tf"), 10.0)
        tstep = _sf(params.get("tstep"), 0.0)
        ss.TDS.config.tf = tf
        if tstep > 0:
            try:
                ss.TDS.config.tstep = tstep
            except Exception:
                pass

        islanded = meta.get("islanded_after")
        if islanded and islanded[2] or meta.get("free_islands"):
            # ANDES's rotor-angle stop criterion looks for machines in the
            # largest island and fails when none is left there; and it takes
            # angles against the rated frequency's, which an island off it
            # turns - with a machine tripped, its frozen angle against the
            # rest stopped the run. Set before the initialisation, which
            # reads it. Losing synchronism is reported below instead, each
            # angle against its island's.
            ss.TDS.config.criteria = 0
        _check_init(ss, meta["warnings"])
        _restore_prefault_state_on_clearing(ss)
        profiled = meta.get("profiled_loads") or []
        if profiled:
            tds_ok = _run_tds_following_profiles(ss, profiled, tf, meta.get("profile_repeat", True),
                                                 meta["warnings"])
        else:
            tds_ok = bool(ss.TDS.run())
        t = np.asarray(ss.dae.ts.t, dtype=float)
        if not tds_ok and len(t):
            # The run used to end with converged=false and no word of why;
            # the plots just stopped.
            meta["warnings"].append(
                f"The simulation stopped at t = {t[-1]:.3f} s of {tf:g} s: the solver did not "
                "converge there (often a loss of synchronism or voltage collapse).")
        max_pts = int(_sf(params.get("max_points"), 800))
        t_ds = _downsample(t, max_pts)
        # indices for downsample of series
        if len(t) > max_pts:
            idx = np.linspace(0, len(t) - 1, max_pts).astype(int)
        else:
            idx = np.arange(len(t))

        def _series_for(model_name: str, var_name: str) -> List[Dict[str, Any]]:
            if not hasattr(ss, model_name):
                return []
            model = getattr(ss, model_name)
            if model.n == 0 or not hasattr(model, var_name):
                return []
            addrs = list(getattr(model, var_name).a)
            if not addrs:
                return []
            values = tds_values(ss, getattr(model, var_name))
            out = []
            for i in range(len(addrs)):
                if str(model.idx.v[i]).startswith("GENCLS_PCS_"):
                    continue          # a grid-forming PCS's virtual machine: the PCS series carry it
                col = values[:, i] if values.ndim == 2 else values
                col = col[idx]
                try:
                    label = str(model.name.v[i])
                    mid = str(model.idx.v[i])
                except Exception:
                    label = f"{model_name}_{i}"
                    mid = label
                out.append({
                    "id": mid,
                    "name": label,
                    "model": model_name,
                    "values": [_clean_num(float(x)) for x in col.tolist()],
                })
            return out

        omega = _series_for("GENROU", "omega") + _series_for("GENCLS", "omega")
        delta = _series_for("GENROU", "delta") + _series_for("GENCLS", "delta")
        # Each machine's electrical power (MW): its share of a disturbance, its governor's droop.
        power = _series_for("GENROU", "Pe") + _series_for("GENCLS", "Pe")
        for series in power:
            series["values"] = [_clean_num(x * meta["sn_mva"]) if x is not None else None for x in series["values"]]

        # Bus voltages
        bus_v = []
        if ss.Bus.n > 0 and hasattr(ss.Bus, "v"):
            addrs = list(ss.Bus.v.a)
            values = tds_values(ss, ss.Bus.v)
            for i in range(len(addrs)):
                col = values[:, i] if values.ndim == 2 else values
                col = col[idx]
                bidx = ss.Bus.idx.v[i]
                if str(bidx) not in meta["bus_name_by_idx"]:
                    continue  # a three-winding transformer's star point
                islanded = meta.get("islanded_after")
                if islanded and str(bidx) in islanded[1]:
                    # A bus with nothing left on it keeps its last voltage in
                    # ANDES; it is dead.
                    col = np.where(t[idx] >= islanded[0], 0.0, col)
                bus_v.append({
                    "id": str(bidx),
                    "name": meta["bus_name_by_idx"].get(str(bidx), str(ss.Bus.name.v[i])),
                    "values": [_clean_num(float(x)) for x in col.tolist()],
                })

        pcs = _pcs_series(ss, meta, idx)

        # Frequency estimate from mean omega (pu → Hz); without a machine, the grid-forming PCS's
        freq_hz = None
        # The machines still running: a tripped one keeps its last speed, and pulled the mean to it.
        running = {str(i) for m in ("GENROU", "GENCLS") for i, u in zip(getattr(ss, m).idx.v, getattr(ss, m).u.v) if u}
        spinning = [s for s in omega if s["id"] in running] or omega
        if spinning:
            mean_w = np.mean([np.asarray(s["values"], dtype=float) for s in spinning], axis=0)
            freq_hz = (mean_w * meta["frequency"]).tolist()
        elif any(r.get("frequency_hz") for r in pcs):
            freq_hz = np.mean([np.asarray(r["frequency_hz"], dtype=float) for r in pcs if r.get("frequency_hz")],
                              axis=0).tolist()

        # Losing synchronism did not show in the result, only in the plots. A
        # pole slip runs the rotor angle more than 180 degrees from where it
        # started; an undamped but stable swing can span more than that from
        # its peak to its back-swing. Each angle is measured against its
        # island's: a grid's held by its External Grid, an island's the mean of
        # its machines' (its frequency off 50 Hz turned every angle).
        reference = _angle_references(ss, idx)
        for s in delta:
            if s["id"] in reference and reference[s["id"]] is None:
                continue
            swing = np.asarray(s["values"], dtype=float) - reference.get(s["id"], 0.0)
            if swing.size and np.nanmax(np.abs(swing - swing[0])) > math.pi:
                meta["warnings"].append(
                    f"'{s['name']}' lost synchronism: its rotor angle swung by more than 180 degrees.")

        poi_bus_key = params.get("poi_bus") or params.get("poi_bus_name") or ""
        # The dialog sends the bus's diagram name; the series carry the ANDES
        # bus idx and the display name, so the POI was never found.
        poi_idx = str(meta.get("bus_map", {}).get(poi_bus_key, "")) if poi_bus_key else ""
        poi_v_min = None
        poi_v_series = None
        poi_label = None
        if poi_bus_key and bus_v:
            for s in bus_v:
                if s.get("id") in (poi_bus_key, poi_idx) or s.get("name") == poi_bus_key:
                    poi_label = s.get("name")
                    poi_v_series = s.get("values") or []
                    if poi_v_series:
                        poi_v_min = min(poi_v_series)
                    break

        freq_nadir = None
        freq_settling = None
        if freq_hz:
            arr = np.asarray(freq_hz, dtype=float)
            freq_nadir = float(np.min(arr))
            if len(arr) > 10:
                freq_settling = float(arr[-1])

        ride_csv = params.get("ride_through_csv") or ""
        if not ride_csv:
            for _, el, typ in _iter_elements(in_data):
                if not typ.startswith("Load") or typ.startswith("Load DC"):
                    continue
                if str(el.get("dc_computational_enabled", "")).lower() in ("true", "1", "yes"):
                    ride_csv = el.get("dc_ride_through_csv") or "0,0.9\n10,0.9"
                    break
        ride_pts = []
        for line in str(ride_csv).strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.replace(";", ",").split(",")
            if len(parts) >= 2:
                ride_pts.append((_sf(parts[0]), _sf(parts[1])))
        ride_pts.sort(key=lambda p: p[0])

        def _vmin_curve(t_val: float) -> float:
            if not ride_pts:
                return 0.0
            if t_val <= ride_pts[0][0]:
                return ride_pts[0][1]
            for i in range(len(ride_pts) - 1):
                t0, v0 = ride_pts[i]
                t1, v1 = ride_pts[i + 1]
                if t0 <= t_val <= t1:
                    if t1 <= t0:
                        return v0
                    return v0 + (v1 - v0) * (t_val - t0) / (t1 - t0)
            return ride_pts[-1][1]

        ride_pass = None
        ride_fail_t = None
        ride_fail_v = None
        if poi_v_series and ride_pts and t_ds is not None:
            ride_pass = True
            for ti, vi in zip(t_ds.tolist(), poi_v_series):
                vmin = _vmin_curve(float(ti))
                if float(vi) < vmin - 1e-5:
                    ride_pass = False
                    ride_fail_t = float(ti)
                    ride_fail_v = float(vi)
                    break

        result = {
            "error": False,
            "routine": "tds",
            "converged": tds_ok,
            "power_flow_converged": pf_ok,
            "time": [_clean_num(float(x)) for x in t_ds.tolist()],
            "omega": omega,
            "delta": delta,
            "generator_p_mw": power,
            "bus_voltage": bus_v,
            "pcs": pcs,
            "frequency_hz": [_clean_num(float(x)) for x in freq_hz] if freq_hz is not None else None,
            "tf": tf,
            "n_points": int(len(t_ds)),
            "defaults_applied": meta["defaults_applied"],
            "warnings": meta["warnings"],
            "n_generators": meta["n_generators"],
            "n_buses": meta["n_buses"],
            "frequency_base_hz": meta["frequency"],
            "sn_mva": meta["sn_mva"],
            "events": {
                "fault_bus": params.get("fault_bus") or params.get("fault_bus_name"),
                "fault_tf": params.get("fault_tf"),
                "fault_tc": params.get("fault_tc"),
                "toggle_line": params.get("toggle_line") or params.get("line_outage"),
                "toggle_t": params.get("toggle_t"),
                "toggle_gen": params.get("toggle_gen") or params.get("generator_trip"),
                "toggle_gen_t": params.get("toggle_gen_t"),
            },
            "poi_metrics": {
                "poi_bus": poi_bus_key or None,
                "poi_bus_label": poi_label,
                "v_min_pu": _clean_num(poi_v_min) if poi_v_min is not None else None,
                "frequency_nadir_hz": _clean_num(freq_nadir) if freq_nadir is not None else None,
                "frequency_final_hz": _clean_num(freq_settling) if freq_settling is not None else None,
            },
            "load_profiles": [{
                "load": load["name"],
                "profile": load["profile"],
                "q_mode": load["q_mode"],
                # The power it was set to through the run (MW), at the plotted times.
                "p_mw": [_clean_num(float(load["p_rated"] * meta["sn_mva"] * v)) for v in
                         _lp_sample(load, t_ds, meta.get("profile_repeat", True))],
            } for load in (meta.get("profiled_loads") or [])],
            "ride_through": {
                "enabled": bool(ride_pts),
                "pass": ride_pass,
                "fail_time_s": _clean_num(ride_fail_t) if ride_fail_t is not None else None,
                "fail_voltage_pu": _clean_num(ride_fail_v) if ride_fail_v is not None else None,
                "curve_points": [{"t_s": a, "v_min_pu": b} for a, b in ride_pts],
            },
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({
            "error": True,
            "message": str(e),
            "exception": traceback.format_exc(),
        })


_STATE_MEANING = {
    "delta": "rotor angle δ",
    "omega": "speed ω",
    "e1q": "E'q",
    "e1d": "E'd",
    "e2d": "ψ''d",
    "e2q": "ψ''q",
}


def _state_labels(ss) -> Dict[str, str]:
    """ANDES state name ('omega GENROU 1') -> 'CHP plant (GENROU): speed ω'."""
    labels = {}
    names = list(ss.dae.x_name)
    for model_name, model in ss.models.items():
        if not getattr(model, "n", 0) or not getattr(model, "states", None):
            continue
        devices = [str(v) for v in model.name.v]
        for var_name, var in model.states.items():
            for j, addr in enumerate(np.asarray(var.a).ravel()):
                if int(addr) >= len(names) or j >= len(devices):
                    continue
                device = devices[j]
                if device.startswith(f"{model_name}_"):
                    device = device[len(model_name) + 1:]
                meaning = _STATE_MEANING.get(var_name, var_name)
                labels[str(names[int(addr)])] = f"{device} ({model_name}): {meaning}"
    return labels


def _participation(As, mu, x_names, mode_indices, labels, top=5):
    """
    Participation of each state in each mode (Kundur 12.2.4): p_ki = |l_ki r_ik|
    for left and right eigenvectors l_k, r_k of the state matrix, normalised so
    each mode's factors add up to 1.

    ANDES's EIG.pfactors has modes as rows, not states as its docstring says,
    and is not normalised per mode; read by column it credited the CHP
    plant's 1.33 Hz rotor swing to E'd.
    """
    As = As.toarray() if hasattr(As, "toarray") else np.asarray(As, dtype=float)
    lam, right = np.linalg.eig(As)
    left = np.linalg.inv(right)
    pf = np.abs(left * right.T)
    pf = pf / pf.sum(axis=1, keepdims=True)
    out = []
    for mi in mode_indices:
        k = int(np.argmin(np.abs(lam - mu[mi])))
        order = np.argsort(pf[k])[::-1][:top]
        out.append({
            "mode_index": int(mi),
            "states": [
                {
                    "state": labels.get(str(x_names[j]), str(x_names[j])),
                    "factor": _clean_num(float(pf[k, j])),
                }
                for j in order if j < len(x_names)
            ],
        })
    return out


def run_eig(in_data: Dict[str, Any], params: Dict[str, Any]) -> str:
    """Run power flow + eigenvalue analysis; return JSON string."""
    try:
        if not _HAS_ANDES:
            return json.dumps({
                "error": True,
                "message": "ANDES is not installed on the backend.",
                "exception": "pip install andes",
            })

        andes.config_logger(stream_level=40)
        # Fresh system without TDS disturbances for clean linearization
        eig_params = dict(params or {})
        eig_params["fault_enabled"] = False
        eig_params["fault_bus"] = ""
        eig_params["fault_bus_name"] = ""
        eig_params["toggle_line"] = ""
        eig_params["line_outage"] = ""

        ss, meta = build_system(in_data, eig_params)
        pf_ok = bool(ss.PFlow.run()) and _settle_pcs_set_points(ss, meta)
        if not pf_ok:
            return json.dumps({
                "error": True,
                "message": "Power flow did not converge. Check network data and generator setpoints.",
                "defaults_applied": meta["defaults_applied"],
                "warnings": meta["warnings"],
            })

        _check_init(ss, meta["warnings"])
        eig_ok = bool(ss.EIG.run())
        mu = np.asarray(ss.EIG.mu, dtype=complex)

        eigenvalues = []
        for i, lam in enumerate(mu):
            re = float(lam.real)
            im = float(lam.imag)
            f_hz = abs(im) / (2.0 * math.pi) if im != 0 else 0.0
            damp = None
            if abs(lam) > 1e-12:
                damp = -re / abs(lam)
            eigenvalues.append({
                "index": i,
                "real": _clean_num(re),
                "imag": _clean_num(im),
                "freq_hz": _clean_num(f_hz),
                "damping_ratio": _clean_num(damp),
            })

        # Sort oscillatory modes by least damping (ascending damping ratio).
        # One row per complex pair: its conjugate is the same mode.
        osc = [e for e in eigenvalues if (e["imag"] or 0) > 1e-6]
        osc_sorted = sorted(
            osc,
            key=lambda e: (e["damping_ratio"] if e["damping_ratio"] is not None else -1e9),
        )
        n_highlight = int(_sf(params.get("n_modes"), 10))
        least_damped = osc_sorted[: max(n_highlight, 0)]

        n_pos = int(getattr(ss.EIG, "n_positive", sum(1 for e in eigenvalues if (e["real"] or 0) > 1e-6)))
        n_zero = int(getattr(ss.EIG, "n_zeros", sum(1 for e in eigenvalues if abs(e["real"] or 0) <= 1e-6 and abs(e["imag"] or 0) <= 1e-6)))
        n_neg = int(getattr(ss.EIG, "n_negative", len(eigenvalues) - n_pos - n_zero))

        if n_pos > 0:
            verdict = "unstable"
        elif any((e["damping_ratio"] is not None and e["damping_ratio"] < 0.03 and abs(e["imag"] or 0) > 1e-6) for e in eigenvalues):
            verdict = "marginally_stable"
        else:
            verdict = "stable"

        # Participation factors for the least-damped modes. `x_name or []` on
        # ANDES's numpy array raised, and the bare except left this empty.
        try:
            participation = _participation(
                ss.EIG.As, mu, list(ss.EIG.x_name), [m["index"] for m in least_damped[:5]],
                _state_labels(ss))
        except Exception as e:
            participation = []
            meta["warnings"].append(f"Participation factors could not be computed: {e}")

        result = {
            "error": False,
            "routine": "eig",
            "converged": eig_ok,
            "power_flow_converged": pf_ok,
            "verdict": verdict,
            "n_positive": n_pos,
            "n_zeros": n_zero,
            "n_negative": n_neg,
            "eigenvalues": eigenvalues,
            "least_damped_modes": least_damped,
            "participation": participation,
            "defaults_applied": meta["defaults_applied"],
            "warnings": meta["warnings"],
            "n_generators": meta["n_generators"],
            "n_buses": meta["n_buses"],
            "frequency_base_hz": meta["frequency"],
            "sn_mva": meta["sn_mva"],
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({
            "error": True,
            "message": str(e),
            "exception": traceback.format_exc(),
        })
