import sys
import io
import contextlib
import pandapower as pp
import pandapower.contingency as contingency
import pandapower.shortcircuit as sc
import pandapower.plotting as plt
from pandapower.diagnostic import diagnostic
import pandapower.topology as top
from typing import List
import math
import re
import json
import warnings
import numpy as np
import pandas as pd
import pandapower.control as control
import pandapower.timeseries as ts
from pandapower.timeseries import DFData
from copy import deepcopy
import weakref
import der_electrisim

from storage_q_capability import resolve_storage_pq
from sc_fault_location import (
    collect_fault_bus_refs,
    normalize_fault_bus_mode,
    resolve_pp_fault_bus_indices,
)


Busbars = {}


def _pf_res_row_for_element(net_element_df, res_df, element_index):
    """
    Map a net element row (trafo / trafo3w) to its power-flow result row.
    Prefer matching index; if lengths match, fall back to same position (handles rare index dtype mismatches).
    """
    if res_df is None or getattr(res_df, 'empty', True):
        return None
    try:
        if element_index in res_df.index:
            return res_df.loc[element_index]
    except (TypeError, KeyError, ValueError):
        pass
    if len(res_df) == len(net_element_df):
        try:
            pos = list(net_element_df.index).index(element_index)
            return res_df.iloc[pos]
        except (ValueError, IndexError, TypeError):
            pass
    return None


def _trafo_out_id(raw_id, name, index_fallback):
    """Stable string id for JSON (avoid NaN/null from pandas)."""
    if raw_id is not None and not pd.isna(raw_id):
        return str(raw_id)
    if name is not None and not pd.isna(name):
        return str(name)
    return str(index_fallback)


def _json_serialize_default(obj):
    """Handle numpy/pandas types for json.dumps without walking object internals.

    Dumping ``obj.__dict__`` is unsafe: pandas DataFrames/Flags and pandapower
    protection devices store ``weakref.ReferenceType``, which raises
    ``Object of type ReferenceType is not JSON serializable``.
    """
    if isinstance(obj, weakref.ReferenceType):
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.replace({np.nan: None}).to_dict(orient='records')
    if isinstance(obj, pd.Series):
        return obj.replace({np.nan: None}).tolist()
    if isinstance(obj, pd.Timestamp):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    # numpy scalars (bool_, int64, float64, etc.) have .item() -> native Python type
    if hasattr(obj, 'item') and callable(getattr(obj, 'item')):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, '__dict__') and not isinstance(obj, type):
        out = {}
        for k, v in obj.__dict__.items():
            if str(k).startswith('_') or isinstance(v, weakref.ReferenceType):
                continue
            if callable(v):
                continue
            out[k] = v
        return out
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _jsonify_safe(obj):
    """Deep-convert numpy/pandas scalars and arrays so Flask jsonify succeeds."""
    return json.loads(json.dumps(obj, default=_json_serialize_default))


def _blank_numeric(value):
    """True when a frontend field is missing and should inherit another ratio."""
    if value is None:
        return True
    if isinstance(value, float) and (value != value):  # NaN
        return True
    text = str(value).strip().lower()
    return text in ('', 'null', 'none', 'nan')


def _ext_grid_zero_seq_min(data, min_key, max_value):
    """Return the min zero-sequence ratio, falling back to the matching max ratio."""
    raw = data.get(min_key) if isinstance(data, dict) else None
    if _blank_numeric(raw):
        return max_value
    return safe_float(raw, max_value)


def ensure_sgen_k(net):
    """Each static generator's short-circuit ratio k: 1.1 where the diagram
    gives none, a value set on the element kept."""
    if 'k' not in net.sgen.columns:
        net.sgen['k'] = 1.1
    else:
        net.sgen['k'] = net.sgen['k'].where(net.sgen['k'] > 0, 1.1)


def isolated_buses_message(net, advice="Check your network connectivity."):
    """The studies' refusal of buses no source supplies, naming them as the
    diagram does, then the advice; None when every bus is supplied."""
    isolated_buses = top.unsupplied_buses(net)
    if len(isolated_buses) == 0:
        return None
    isolated_refs = resolve_element_refs(net, 'bus', isolated_buses)
    isolated_names = [r.get('name') or r.get('id') or str(r.get('index')) for r in isolated_refs]
    return f"Isolated buses found: {', '.join(isolated_names)}. {advice}"


def ensure_ext_grid_zero_sequence_min(net):
    """
    pandapower single-phase min short-circuit reads ext_grid['x0x_min'] and
    ['r0x0_min']. Older ElectriSim models only stored the max ratios, so copy
    those when the min columns are missing or NaN.
    """
    eg = getattr(net, 'ext_grid', None)
    if eg is None or getattr(eg, 'empty', True):
        return
    if 'x0x_max' not in eg.columns:
        net.ext_grid['x0x_max'] = 1.0
        eg = net.ext_grid
    if 'r0x0_max' not in eg.columns:
        net.ext_grid['r0x0_max'] = 0.1
        eg = net.ext_grid
    if 'x0x_min' not in eg.columns:
        net.ext_grid['x0x_min'] = eg['x0x_max']
        eg = net.ext_grid
    else:
        net.ext_grid['x0x_min'] = eg['x0x_min'].where(eg['x0x_min'].notna(), eg['x0x_max'])
        eg = net.ext_grid
    if 'r0x0_min' not in eg.columns:
        net.ext_grid['r0x0_min'] = eg['r0x0_max']
    else:
        net.ext_grid['r0x0_min'] = eg['r0x0_min'].where(eg['r0x0_min'].notna(), eg['r0x0_max'])


def _sanitize_for_strict_json(obj):
    """
    Recursively replace NaN/Inf with None so payloads are valid RFC 8259 JSON.
    Browser Response.json() / JSON.parse reject unquoted NaN/Infinity tokens.
    """
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    if isinstance(obj, str):
        return obj
    if isinstance(obj, np.ndarray):
        return _sanitize_for_strict_json(obj.tolist())
    if isinstance(obj, weakref.ReferenceType):
        return None
    if isinstance(obj, pd.DataFrame):
        return _sanitize_for_strict_json(obj.replace({np.nan: None}).to_dict(orient='records'))
    if isinstance(obj, pd.Series):
        return _sanitize_for_strict_json(obj.replace({np.nan: None}).tolist())
    if isinstance(obj, dict):
        return {k: _sanitize_for_strict_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_strict_json(v) for v in obj]
    if hasattr(obj, 'item') and callable(getattr(obj, 'item')):
        try:
            return _sanitize_for_strict_json(obj.item())
        except Exception:
            return None
    if hasattr(obj, '__dict__') and not isinstance(obj, type):
        return _sanitize_for_strict_json({
            k: v for k, v in vars(obj).items()
            if not str(k).startswith('_') and not isinstance(v, weakref.ReferenceType) and not callable(v)
        })
    return obj


def _electrisim_switch_res_for_output(net, sw_idx, row):
    """
    Fill ``net.res_switch``-like quantities for JSON / SwitchOut.

    pandapower 3.x often leaves ``p_from_mw`` / ``q_*`` as NaN for switches (it copies ``i_ka`` only
    for line/trafo switches). Ideal ``z_ohm=0`` bus–bus ties also yield NaN. The frontend maps null/NaN
    to \"N/A\", so we repair rows here.

    Bus–Switch–injecting element exports ``et='b'`` between the diagram bus and ``_electrisim_aux_*``; we detect
    the aux on **either** terminal and derive P/Q/I from ``res_sgen``, ``res_storage``, ``res_load``,
    ``res_asymmetric_load``, or ``res_shunt`` on that bus. Line / 2W-trafo
    switches copy branch P/Q (and current if missing) from ``res_line`` / ``res_trafo``.
    """
    def _as_float(x, default=float('nan')):
        try:
            v = float(x)
            if math.isnan(v) or math.isinf(v):
                return default
            return v
        except (TypeError, ValueError):
            return default

    def _is_nan(v):
        try:
            v = float(v)
            return not (v == v)
        except (TypeError, ValueError):
            return True

    def _fin(v):
        try:
            x = float(v)
            if math.isnan(x) or math.isinf(x):
                return 0.0
            return x
        except (TypeError, ValueError):
            return 0.0

    i_ka = _as_float(row.get('i_ka'), float('nan'))
    p_from = _as_float(row.get('p_from_mw'), float('nan'))
    q_from = _as_float(row.get('q_from_mvar'), float('nan'))
    p_to = _as_float(row.get('p_to_mw'), float('nan'))
    q_to = _as_float(row.get('q_to_mvar'), float('nan'))
    loading = _as_float(row.get('loading_percent'), float('nan'))

    try:
        if getattr(net, 'switch', None) is None or net.switch.empty or sw_idx not in net.switch.index:
            return _fin(i_ka), _fin(p_from), _fin(q_from), _fin(p_to), _fin(q_to), _fin(loading)

        sw = net.switch.loc[sw_idx]
        et = str(sw['et'])

        # --- Bus–bus: Electrisim aux stub + sgen / storage / load / shunt / … on aux ---
        if et == 'b':
            bus_a = int(sw['bus'])
            bus_b = int(sw['element'])
            aux_bus = None
            for cand in (bus_a, bus_b):
                try:
                    nm = str(net.bus.at[cand, 'name'])
                except (KeyError, TypeError, ValueError):
                    continue
                if nm.startswith('_electrisim_aux_'):
                    aux_bus = cand
                    break
            if aux_bus is not None:
                try:
                    vn = float(net.bus.at[aux_bus, 'vn_kv'])
                    vm = float(net.res_bus.at[aux_bus, 'vm_pu'])
                except (KeyError, TypeError, ValueError):
                    vn, vm = 0.0, 0.0

                def _i_switch_loading(pm, qm):
                    smva = math.hypot(float(pm), float(qm))
                    i_new = smva / (math.sqrt(3) * vn * vm) if vn > 0 and vm > 0 and smva > 1e-12 else 0.0
                    ink = float('nan')
                    if 'in_ka' in net.switch.columns:
                        try:
                            ink = float(net.switch.at[sw_idx, 'in_ka'])
                        except (TypeError, ValueError):
                            ink = float('nan')
                    load_pct = 0.0
                    if ink == ink and ink > 0:
                        load_pct = abs(i_new) / ink * 100.0
                    return i_new, load_pct

                if not net.sgen.empty and hasattr(net, 'res_sgen') and net.res_sgen is not None and not net.res_sgen.empty:
                    sel = net.sgen['bus'] == aux_bus
                    if sel.any():
                        si = int(net.sgen.index[sel][0])
                        pm = float(net.res_sgen.at[si, 'p_mw'])
                        qm = float(net.res_sgen.at[si, 'q_mvar'])
                        i_new, load_pct = _i_switch_loading(pm, qm)
                        return _fin(i_new), _fin(-pm), _fin(-qm), _fin(pm), _fin(qm), _fin(load_pct)

                if hasattr(net, 'storage') and net.storage is not None and not net.storage.empty \
                        and hasattr(net, 'res_storage') and net.res_storage is not None and not net.res_storage.empty:
                    sel = net.storage['bus'] == aux_bus
                    if sel.any():
                        si = int(net.storage.index[sel][0])
                        pm = float(net.res_storage.at[si, 'p_mw'])
                        qm = float(net.res_storage.at[si, 'q_mvar'])
                        i_new, load_pct = _i_switch_loading(pm, qm)
                        return _fin(i_new), _fin(-pm), _fin(-qm), _fin(pm), _fin(qm), _fin(load_pct)

                if not net.load.empty and hasattr(net, 'res_load') and net.res_load is not None and not net.res_load.empty:
                    sel = net.load['bus'] == aux_bus
                    if sel.any():
                        li = int(net.load.index[sel][0])
                        pl = float(net.res_load.at[li, 'p_mw'])
                        ql = float(net.res_load.at[li, 'q_mvar'])
                        i_new, load_pct = _i_switch_loading(pl, ql)
                        return _fin(i_new), _fin(pl), _fin(ql), _fin(-pl), _fin(-ql), _fin(load_pct)

                if hasattr(net, 'asymmetric_load') and net.asymmetric_load is not None and not net.asymmetric_load.empty \
                        and hasattr(net, 'res_asymmetric_load') and net.res_asymmetric_load is not None \
                        and not net.res_asymmetric_load.empty:
                    sel = net.asymmetric_load['bus'] == aux_bus
                    if sel.any():
                        ai = int(net.asymmetric_load.index[sel][0])
                        ra = net.res_asymmetric_load.loc[ai]
                        pl = float(ra.get('p_mw', 0.0))
                        ql = float(ra.get('q_mvar', 0.0))
                        i_new, load_pct = _i_switch_loading(pl, ql)
                        return _fin(i_new), _fin(pl), _fin(ql), _fin(-pl), _fin(-ql), _fin(load_pct)

                if not net.shunt.empty and hasattr(net, 'res_shunt') and net.res_shunt is not None and not net.res_shunt.empty:
                    sel = net.shunt['bus'] == aux_bus
                    if sel.any():
                        hi = int(net.shunt.index[sel][0])
                        pm = float(net.res_shunt.at[hi, 'p_mw'])
                        qm = float(net.res_shunt.at[hi, 'q_mvar'])
                        i_new, load_pct = _i_switch_loading(pm, qm)
                        return _fin(i_new), _fin(pm), _fin(qm), _fin(-pm), _fin(-qm), _fin(load_pct)

            # Plain bus–tie (no aux, or aux had no mapped injection): a single AC line between the
            # same two buses — copy ``res_line`` like ``et='l'`` so the switch result box is not all zeros
            # when the frontend exported a parallel ``et='b'`` next to that line.
            pair = {bus_a, bus_b}
            par_line_idx = None
            if not net.line.empty:
                for li in net.line.index:
                    lf = int(net.line.at[li, 'from_bus'])
                    lt = int(net.line.at[li, 'to_bus'])
                    if {lf, lt} == pair:
                        if par_line_idx is not None:
                            par_line_idx = None
                            break
                        par_line_idx = li
            if (
                par_line_idx is not None
                and par_line_idx in net.line.index
                and hasattr(net, 'res_line')
                and par_line_idx in net.res_line.index
            ):
                li = int(par_line_idx)
                bsw = bus_a
                lf = int(net.line.at[li, 'from_bus'])
                lt = int(net.line.at[li, 'to_bus'])
                rl = net.res_line.loc[li]
                if bsw == lf:
                    if _is_nan(p_from):
                        p_from = _as_float(rl.get('p_from_mw'), p_from)
                    if _is_nan(q_from):
                        q_from = _as_float(rl.get('q_from_mvar'), q_from)
                    if _is_nan(p_to):
                        p_to = _as_float(rl.get('p_to_mw'), p_to)
                    if _is_nan(q_to):
                        q_to = _as_float(rl.get('q_to_mvar'), q_to)
                    if _is_nan(i_ka):
                        i_ka = _as_float(rl.get('i_from_ka'), i_ka)
                elif bsw == lt:
                    if _is_nan(p_from):
                        p_from = _as_float(rl.get('p_to_mw'), p_from)
                    if _is_nan(q_from):
                        q_from = _as_float(rl.get('q_to_mvar'), q_from)
                    if _is_nan(p_to):
                        p_to = _as_float(rl.get('p_from_mw'), p_to)
                    if _is_nan(q_to):
                        q_to = _as_float(rl.get('q_from_mvar'), q_to)
                    if _is_nan(i_ka):
                        i_ka = _as_float(rl.get('i_to_ka'), i_ka)

        # --- Line switch: copy P/Q from res_line (same idea as pandapower i_ka copy) ---
        if et == 'l':
            li = int(sw['element'])
            bsw = int(sw['bus'])
            if not net.line.empty and li in net.line.index and hasattr(net, 'res_line') and li in net.res_line.index:
                lf = int(net.line.at[li, 'from_bus'])
                lt = int(net.line.at[li, 'to_bus'])
                rl = net.res_line.loc[li]
                if bsw == lf:
                    if _is_nan(p_from):
                        p_from = _as_float(rl.get('p_from_mw'), p_from)
                    if _is_nan(q_from):
                        q_from = _as_float(rl.get('q_from_mvar'), q_from)
                    if _is_nan(p_to):
                        p_to = _as_float(rl.get('p_to_mw'), p_to)
                    if _is_nan(q_to):
                        q_to = _as_float(rl.get('q_to_mvar'), q_to)
                    if _is_nan(i_ka):
                        i_ka = _as_float(rl.get('i_from_ka'), i_ka)
                elif bsw == lt:
                    if _is_nan(p_from):
                        p_from = _as_float(rl.get('p_to_mw'), p_from)
                    if _is_nan(q_from):
                        q_from = _as_float(rl.get('q_to_mvar'), q_from)
                    if _is_nan(p_to):
                        p_to = _as_float(rl.get('p_from_mw'), p_to)
                    if _is_nan(q_to):
                        q_to = _as_float(rl.get('q_from_mvar'), q_to)
                    if _is_nan(i_ka):
                        i_ka = _as_float(rl.get('i_to_ka'), i_ka)

        # --- 2W transformer switch ---
        if et == 't':
            ti = int(sw['element'])
            bsw = int(sw['bus'])
            if not net.trafo.empty and ti in net.trafo.index and hasattr(net, 'res_trafo') and ti in net.res_trafo.index:
                hv = int(net.trafo.at[ti, 'hv_bus'])
                lv = int(net.trafo.at[ti, 'lv_bus'])
                rt = net.res_trafo.loc[ti]
                if bsw == hv:
                    if _is_nan(p_from):
                        p_from = _as_float(rt.get('p_hv_mw'), p_from)
                    if _is_nan(q_from):
                        q_from = _as_float(rt.get('q_hv_mvar'), q_from)
                    if _is_nan(p_to):
                        p_to = _as_float(rt.get('p_lv_mw'), p_to)
                    if _is_nan(q_to):
                        q_to = _as_float(rt.get('q_lv_mvar'), q_to)
                    if _is_nan(i_ka):
                        i_ka = _as_float(rt.get('i_hv_ka'), i_ka)
                elif bsw == lv:
                    if _is_nan(p_from):
                        p_from = _as_float(rt.get('p_lv_mw'), p_from)
                    if _is_nan(q_from):
                        q_from = _as_float(rt.get('q_lv_mvar'), q_from)
                    if _is_nan(p_to):
                        p_to = _as_float(rt.get('p_hv_mw'), p_to)
                    if _is_nan(q_to):
                        q_to = _as_float(rt.get('q_hv_mvar'), q_to)
                    if _is_nan(i_ka):
                        i_ka = _as_float(rt.get('i_lv_ka'), i_ka)

        # Loading % from i_ka and switch rating
        ink = float('nan')
        if 'in_ka' in net.switch.columns:
            try:
                ink = float(net.switch.at[sw_idx, 'in_ka'])
            except (TypeError, ValueError):
                ink = float('nan')
        if _is_nan(loading) and not _is_nan(i_ka) and ink == ink and ink > 0:
            loading = abs(float(i_ka)) / ink * 100.0

        return _fin(i_ka), _fin(p_from), _fin(q_from), _fin(p_to), _fin(q_to), _fin(loading)
    except Exception:
        return _fin(i_ka), _fin(p_from), _fin(q_from), _fin(p_to), _fin(q_to), _fin(loading)


def _electrisim_bus_branch_p_q_sum(net, bus_idx):
    """
    Compute "through power" for ``bus_idx`` from AC branch results (lines, 2W/3W trafos, impedances).

    pandapower ``res_bus.p_mw`` / ``q_mvar`` are *lumped* injections (load, sgen, ext_grid, …) at that bus.
    On a pass-through bus (e.g. LV node with trafo + gen behind a bus–bus switch on an aux bus) net
    ``res_bus`` injection can be **zero** while line/trafo terminals still carry power.

    Terminal convention: ``p_from_mw`` / ``p_to_mw`` / ``p_hv_mw`` / ``p_lv_mw`` is the active power
    **from the bus into the branch** (positive ⇒ leaves the bus). We accumulate:

    - **inflow** ``(p_in, q_in)``: contribution when ``p < 0`` (or ``q < 0``) — power/var flowing into the bus.
    - **outflow** ``(p_out, q_out)``: contribution when ``p > 0`` (or ``q > 0``) — leaving the bus.

    At a pass-through node, inflow and outflow magnitudes match (Kirchhoff). Earlier code only kept
    *inflow*, so buses that **only** export into a trafo (``p_lv > 0``, gen behind an aux switch) got
    ``p_branch_mw = 0``. We return the pair from whichever side has larger |P| so the UI shows the
    correct through-power **magnitude**.

    **Sign:** pandapower ``res_bus.p_mw`` on a bus with local generation is typically **negative** (same
    sign as ``P_from`` on an intervening bus–bus switch). Branch outflow uses terminal ``p > 0`` (power
    *leaving* the bus into the branch), so it comes out **positive**. When outflow dominates
    (``p_out > p_in``), we **negate** P and Q so the bus label matches the normal ``res_bus`` convention
    and stays consistent with the “no aux / no switch” case.
    """
    p_in = 0.0
    p_out = 0.0
    q_in_s = 0.0  # signed Σ Q at terminals with P flowing into the bus (p < 0)
    q_out_s = 0.0  # signed Σ Q at terminals with P flowing out of the bus (p > 0)

    def _f(v, default=0.0):
        try:
            x = float(v)
            if math.isnan(x) or math.isinf(x):
                return default
            return x
        except (TypeError, ValueError):
            return default

    def _add_in_out(p_term, q_term):
        nonlocal p_in, p_out, q_in_s, q_out_s
        pt = _f(p_term)
        qt = _f(q_term)
        if pt < 0:
            p_in += -pt
            q_in_s += qt
        elif pt > 0:
            p_out += pt
            q_out_s += qt

    try:
        if hasattr(net, "line") and net.line is not None and not net.line.empty \
                and hasattr(net, "res_line") and net.res_line is not None and not net.res_line.empty:
            for i in net.line.index:
                if i not in net.res_line.index:
                    continue
                fr = net.line.at[i, "from_bus"]
                to = net.line.at[i, "to_bus"]
                rl = net.res_line.loc[i]
                if fr == bus_idx:
                    _add_in_out(rl.get("p_from_mw", 0.0), rl.get("q_from_mvar", 0.0))
                if to == bus_idx:
                    _add_in_out(rl.get("p_to_mw", 0.0), rl.get("q_to_mvar", 0.0))
    except Exception:
        pass

    try:
        if hasattr(net, "trafo") and net.trafo is not None and not net.trafo.empty \
                and hasattr(net, "res_trafo") and net.res_trafo is not None and not net.res_trafo.empty:
            for i in net.trafo.index:
                if i not in net.res_trafo.index:
                    continue
                hv = net.trafo.at[i, "hv_bus"]
                lv = net.trafo.at[i, "lv_bus"]
                rt = net.res_trafo.loc[i]
                if hv == bus_idx:
                    _add_in_out(rt.get("p_hv_mw", 0.0), rt.get("q_hv_mvar", 0.0))
                if lv == bus_idx:
                    _add_in_out(rt.get("p_lv_mw", 0.0), rt.get("q_lv_mvar", 0.0))
    except Exception:
        pass

    try:
        if hasattr(net, "trafo3w") and net.trafo3w is not None and not net.trafo3w.empty \
                and hasattr(net, "res_trafo3w") and net.res_trafo3w is not None and not net.res_trafo3w.empty:
            for i in net.trafo3w.index:
                if i not in net.res_trafo3w.index:
                    continue
                row = net.trafo3w.loc[i]
                r3 = net.res_trafo3w.loc[i]
                if row["hv_bus"] == bus_idx:
                    _add_in_out(r3.get("p_hv_mw", 0.0), r3.get("q_hv_mvar", 0.0))
                if row["mv_bus"] == bus_idx:
                    _add_in_out(r3.get("p_mv_mw", 0.0), r3.get("q_mv_mvar", 0.0))
                if row["lv_bus"] == bus_idx:
                    _add_in_out(r3.get("p_lv_mw", 0.0), r3.get("q_lv_mvar", 0.0))
    except Exception:
        pass

    try:
        if hasattr(net, "impedance") and net.impedance is not None and not net.impedance.empty \
                and hasattr(net, "res_impedance") and net.res_impedance is not None and not net.res_impedance.empty:
            for i in net.impedance.index:
                if i not in net.res_impedance.index:
                    continue
                fr = net.impedance.at[i, "from_bus"]
                to = net.impedance.at[i, "to_bus"]
                r = net.res_impedance.loc[i]
                if fr == bus_idx:
                    _add_in_out(r.get("p_from_mw", 0.0), r.get("q_from_mvar", 0.0))
                if to == bus_idx:
                    _add_in_out(r.get("p_to_mw", 0.0), r.get("q_to_mvar", 0.0))
    except Exception:
        pass

    if p_in >= p_out:
        return p_in, q_in_s
    return -p_out, -q_out_s


def _electrisim_bus_linked_aux_buses(net, bus_idx):
    """
    ``_electrisim_aux_*`` bus indices joined to ``bus_idx`` by closed, in-service bus–bus
    switches (Electrisim's Bus–Switch–element export pattern).
    """
    aux = []
    try:
        if getattr(net, 'switch', None) is None or net.switch.empty:
            return aux
        for sw_idx in net.switch.index:
            sw = net.switch.loc[sw_idx]
            if str(sw.get('et', '')) != 'b':
                continue
            closed = sw['closed'] if 'closed' in sw.index else True
            if closed is False or (isinstance(closed, (int, float)) and float(closed) == 0):
                continue
            in_service = sw['in_service'] if 'in_service' in sw.index else True
            if in_service is False or (isinstance(in_service, (int, float)) and float(in_service) == 0):
                continue
            bus_a = int(sw['bus'])
            bus_b = int(sw['element'])
            if bus_a == bus_idx:
                other = bus_b
            elif bus_b == bus_idx:
                other = bus_a
            else:
                continue
            try:
                nm = str(net.bus.at[other, 'name'])
            except (KeyError, TypeError, ValueError):
                continue
            if nm.startswith('_electrisim_aux_'):
                aux.append(other)
    except Exception:
        pass
    return aux


def _electrisim_bus_aux_injection_sum(net, bus_idx):
    """
    Sum ``res_bus`` injections on ``_electrisim_aux_*`` buses linked to ``bus_idx`` via closed
    bus–bus switches (generator / load / shunt / storage behind a diagram switch).
    """
    def _f(v, default=0.0):
        try:
            x = float(v)
            if math.isnan(x) or math.isinf(x):
                return default
            return x
        except (TypeError, ValueError):
            return default

    p = 0.0
    q = 0.0
    try:
        if not hasattr(net, 'res_bus') or net.res_bus is None or net.res_bus.empty:
            return p, q
        for other in _electrisim_bus_linked_aux_buses(net, bus_idx):
            if other not in net.res_bus.index:
                continue
            p += _f(net.res_bus.at[other, 'p_mw'])
            q += _f(net.res_bus.at[other, 'q_mvar'])
    except Exception:
        pass
    return p, q


def _electrisim_bus_has_slack(net, bus_idx):
    """
    True when an in-service external grid (or slack generator) sits on ``bus_idx`` — directly
    or on a linked ``_electrisim_aux_*`` bus behind a closed bus–bus switch.
    """
    buses = {bus_idx}
    buses.update(_electrisim_bus_linked_aux_buses(net, bus_idx))
    try:
        if hasattr(net, 'ext_grid') and net.ext_grid is not None and not net.ext_grid.empty:
            for i in net.ext_grid.index:
                if int(net.ext_grid.at[i, 'bus']) not in buses:
                    continue
                ins = net.ext_grid.at[i, 'in_service'] if 'in_service' in net.ext_grid.columns else True
                if ins is False or (isinstance(ins, (int, float)) and float(ins) == 0):
                    continue
                return True
        if hasattr(net, 'gen') and net.gen is not None and not net.gen.empty and 'slack' in net.gen.columns:
            for i in net.gen.index:
                if not bool(net.gen.at[i, 'slack']):
                    continue
                if int(net.gen.at[i, 'bus']) not in buses:
                    continue
                ins = net.gen.at[i, 'in_service'] if 'in_service' in net.gen.columns else True
                if ins is False or (isinstance(ins, (int, float)) and float(ins) == 0):
                    continue
                return True
    except Exception:
        pass
    return False


def _electrisim_bus_branch_terminal_count(net, bus_idx):
    """
    Number of energized AC branch terminals (line / 2W trafo / 3W trafo / impedance) incident
    to ``bus_idx``. A line connected to the same bus on both ends counts twice. Only branches
    present in the matching ``res_*`` table are counted (consistent with the power summation),
    so de-energized branches do not flip a radial bus into a junction.
    """
    n = 0
    try:
        if hasattr(net, "line") and net.line is not None and not net.line.empty \
                and hasattr(net, "res_line") and net.res_line is not None and not net.res_line.empty:
            for i in net.line.index:
                if i not in net.res_line.index:
                    continue
                if net.line.at[i, "from_bus"] == bus_idx:
                    n += 1
                if net.line.at[i, "to_bus"] == bus_idx:
                    n += 1
    except Exception:
        pass
    try:
        if hasattr(net, "trafo") and net.trafo is not None and not net.trafo.empty \
                and hasattr(net, "res_trafo") and net.res_trafo is not None and not net.res_trafo.empty:
            for i in net.trafo.index:
                if i not in net.res_trafo.index:
                    continue
                if net.trafo.at[i, "hv_bus"] == bus_idx:
                    n += 1
                if net.trafo.at[i, "lv_bus"] == bus_idx:
                    n += 1
    except Exception:
        pass
    try:
        if hasattr(net, "trafo3w") and net.trafo3w is not None and not net.trafo3w.empty \
                and hasattr(net, "res_trafo3w") and net.res_trafo3w is not None and not net.res_trafo3w.empty:
            for i in net.trafo3w.index:
                if i not in net.res_trafo3w.index:
                    continue
                row = net.trafo3w.loc[i]
                if row["hv_bus"] == bus_idx:
                    n += 1
                if row["mv_bus"] == bus_idx:
                    n += 1
                if row["lv_bus"] == bus_idx:
                    n += 1
    except Exception:
        pass
    try:
        if hasattr(net, "impedance") and net.impedance is not None and not net.impedance.empty \
                and hasattr(net, "res_impedance") and net.res_impedance is not None and not net.res_impedance.empty:
            for i in net.impedance.index:
                if i not in net.res_impedance.index:
                    continue
                if net.impedance.at[i, "from_bus"] == bus_idx:
                    n += 1
                if net.impedance.at[i, "to_bus"] == bus_idx:
                    n += 1
    except Exception:
        pass
    return n


def _electrisim_bus_nodal_p_q_sum(net, bus_idx):
    """
    P/Q shown on the bus result label.

    **Net local injection**:

        = this bus ``res_bus`` (load − generation of elements on the bus)
        + injections on ``_electrisim_aux_*`` buses behind closed bus–bus switches.

    Kirchhoff already sends that injection out through the incident branches, so
    adding the dominant branch term (``_electrisim_bus_branch_p_q_sum``) double-counts
    it. Never add the two.

    Choice when both are non-zero:

    - If local injection accounts for most of the branch magnitude (generation or
      load collection bus, including a junction LV busbar with BESS plus
      transformer *and* another feeder) → show **injection**. Two 1.75 MW BESS
      units must read −3.5 MW, not −7 MW. The solved load flow (transformer /
      ext. grid) is already ~3.5 MW — only the bus label was wrong.
    - If through-power dominates a small local leftover (transit hub, shunt vs
      cable charging) → show **through-power**, not injection + branch.
    - Pass-through (no local injection) → through-power so the label is not 0.

    Slack buses report net local injection only (``res_bus`` already nets the
    slack infeed). Adding the branch would double-count the reference-bus infeed.
    """
    def _f(v, default=0.0):
        try:
            x = float(v)
            if math.isnan(x) or math.isinf(x):
                return default
            return x
        except (TypeError, ValueError):
            return default

    p_inj = _f(net.res_bus.at[bus_idx, 'p_mw'])
    q_inj = _f(net.res_bus.at[bus_idx, 'q_mvar'])
    p_aux, q_aux = _electrisim_bus_aux_injection_sum(net, bus_idx)

    p_local = p_inj + p_aux
    q_local = q_inj + q_aux

    # Slack / reference bus: show the net local injection only (the branch re-exports
    # that same power). res_bus already nets all elements.
    if _electrisim_bus_has_slack(net, bus_idx):
        return p_local, q_local

    p_br, q_br = _electrisim_bus_branch_p_q_sum(net, bus_idx)
    mag_local = math.hypot(p_local, q_local)
    mag_br = math.hypot(p_br, q_br)

    if mag_local < 1e-6:
        return p_br, q_br

    # Collection bus: branches merely re-export the local devices (radial sgen,
    # two BESS on a junction LV busbar, …).
    if mag_br < 1e-6 or mag_local >= 0.5 * mag_br:
        return p_local, q_local

    # Transit hub: large through-flow, small local leftover — through-power only.
    return p_br, q_br


def _export_py_literal(obj):
    """Render a Python literal for generated export scripts (no numpy import required)."""
    try:
        import numpy as np
        if isinstance(obj, (np.bool_, bool)):
            return 'True' if bool(obj) else 'False'
        if isinstance(obj, np.integer):
            return str(int(obj))
        if isinstance(obj, np.floating):
            v = float(obj)
            return 'None' if v != v else repr(v)
    except ImportError:
        if isinstance(obj, bool):
            return 'True' if obj else 'False'
    if obj is None:
        return 'None'
    if isinstance(obj, bool):
        return 'True' if obj else 'False'
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, float):
        return 'None' if obj != obj else repr(obj)
    if isinstance(obj, str):
        return repr(obj)
    if isinstance(obj, (list, tuple)):
        return '[' + ', '.join(_export_py_literal(x) for x in obj) + ']'
    return repr(obj)


def _normalize_tap_side(val):
    """Normalize tap_side for pandapower (diagram may send 'null')."""
    if val is None:
        return 'hv'
    s = str(val).strip().lower()
    if s in ('', 'null', 'none', 'nan'):
        return 'hv'
    return str(val)


def _append_electrisim_sgen_setup_python(lines, net):
    """Export Q capability curves, sgen ids, and initial Q setpoints (post curve, pre park)."""
    import pandas as pd

    q_init = getattr(net, '_electrisim_export_sgen_q_init', None) or {}
    ufn = getattr(net, 'user_friendly_names', None) or {}
    qtbl = net.get('q_capability_curve_table')
    has_qtbl = qtbl is not None and hasattr(qtbl, 'empty') and not qtbl.empty
    has_q_init = bool(q_init)
    has_minmax = 'min_q_mvar' in net.sgen.columns and 'max_q_mvar' in net.sgen.columns
    has_ids = 'id' in net.sgen.columns
    has_sn = 'sn_mva' in net.sgen.columns

    if not (has_qtbl or has_q_init or ufn or has_ids or has_sn or has_minmax):
        return False

    lines.append("# --- Electrisim sgen setup (Q capability + initial Q setpoints) ---")
    if ufn:
        lines.append(f"net.user_friendly_names = {_export_py_literal(dict(ufn))}")

    if has_ids or has_sn:
        for idx, row in net.sgen.iterrows():
            if has_ids:
                sid = row.get('id')
                if sid is not None and str(sid).strip():
                    lines.append(f"net.sgen.at[{idx}, 'id'] = {sid!r}")
            if has_sn:
                sn = row.get('sn_mva')
                if sn is not None and sn == sn:
                    lines.append(f"net.sgen.at[{idx}, 'sn_mva'] = {float(sn)}")

    if has_qtbl:
        lines.append("import pandas as pd")
        lines.append("from pandapower.control.util.auxiliary import create_q_capability_characteristics_object")
        records = []
        for _, r in qtbl.iterrows():
            records.append({
                'id_q_capability_curve': int(r['id_q_capability_curve']),
                'p_mw': float(r['p_mw']),
                'q_min_mvar': float(r['q_min_mvar']),
                'q_max_mvar': float(r['q_max_mvar']),
            })
        lines.append(f"net['q_capability_curve_table'] = pd.DataFrame({_export_py_literal(records)})")
        if 'id_q_capability_characteristic' in net.sgen.columns:
            for idx, row in net.sgen.iterrows():
                cid = row.get('id_q_capability_characteristic')
                if cid is None or (isinstance(cid, float) and pd.isna(cid)):
                    continue
                lines.append(f"net.sgen.at[{idx}, 'id_q_capability_characteristic'] = {int(cid)}")
                if 'curve_style' in net.sgen.columns:
                    cs = net.sgen.at[idx, 'curve_style']
                    if cs is not None and not (isinstance(cs, float) and pd.isna(cs)):
                        lines.append(f"net.sgen.at[{idx}, 'curve_style'] = {str(cs)!r}")
        lines.append("create_q_capability_characteristics_object(net)")

    if has_minmax:
        for idx, row in net.sgen.iterrows():
            for col in ('min_q_mvar', 'max_q_mvar'):
                v = net.sgen.at[idx, col]
                if v is not None and v == v:
                    lines.append(f"net.sgen.at[{idx}, '{col}'] = {float(v)}")

    if has_q_init:
        lines.append("# Q setpoints from capability curve (capacitive_max / inductive_max / manual)")
        for idx, q in sorted(q_init.items(), key=lambda x: int(x[0])):
            lines.append(f"net.sgen.at[{int(idx)}, 'q_mvar'] = {float(q)}")
    lines.append("")
    return True


# IEC 60909 ext_grid impedance inputs (needed so exported SC scripts match Electrisim).
_EXT_GRID_SC_EXPORT_COLS = (
    's_sc_max_mva', 's_sc_min_mva',
    'rx_max', 'rx_min',
    'r0x0_max', 'x0x_max',
    'r0x0_min', 'x0x_min',
)

# Columns the exported create_* calls carry beyond their fixed arguments:
# whatever pandapower's short circuit reads (motor cos_phi_n, generator
# subtransient data, sgen k / kappa, line end temperature and zero sequence,
# trafo neutral and power-station data) and in_service. Each was left out, so
# the script either failed in calc_sc or ran a different network.
_EXPORT_EXTRA_COLS = {
    'motor': ('pn_mech_mw', 'cos_phi', 'cos_phi_n', 'efficiency_n_percent', 'lrc_pu', 'rx', 'vn_kv',
              'efficiency_percent', 'loading_percent', 'scaling', 'in_service'),
    'line': ('in_service', 'r0_ohm_per_km', 'x0_ohm_per_km', 'c0_nf_per_km', 'g0_us_per_km',
             'endtemp_degree'),
    'trafo': ('in_service', 'xn_ohm', 'rn_ohm', 'pt_percent', 'oltc', 'power_station_unit'),
    'trafo3w': ('in_service',),
    'gen': ('in_service', 'sn_mva', 'vn_kv', 'xdss_pu', 'rdss_ohm', 'cos_phi', 'pg_percent',
            'power_station_trafo'),
    'sgen': ('in_service', 'k', 'rx', 'generator_type', 'kappa', 'lrc_pu', 'max_ik_ka',
             'current_source'),
    'load': ('in_service',),
    'shunt': ('in_service', 'vn_kv', 'step', 'max_step'),
    'ext_grid': ('in_service',),
}


def _export_extra_kwargs(table, row):
    """', col=value' for each of the table's extra columns the live row sets.
    A null column is skipped so pandapower keeps its own NaN default."""
    return ''.join(
        f", {col}={_export_py_literal(row[col])}" for col in _EXPORT_EXTRA_COLS[table]
        if col in row.index and not pd.isnull(row[col]))


def _export_float_from_payload(val):
    if val is None or val == '' or str(val).lower() in ('null', 'none'):
        return None
    try:
        f = float(val)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


def _export_sc_param_value(row, in_data_elem, col):
    """Prefer net.ext_grid row; fall back to diagram payload (in_data) for IEC SC fields."""
    index = getattr(row, 'index', None)
    if index is not None and col in index:
        try:
            val = row[col]
            if val is not None and not pd.isna(val):
                return float(val)
        except (TypeError, ValueError):
            pass
    if isinstance(in_data_elem, dict):
        return _export_float_from_payload(in_data_elem.get(col))
    return None


def _ext_grid_sc_kwargs_for_export(row, in_data_elem=None):
    parts = []
    for col in _EXT_GRID_SC_EXPORT_COLS:
        v = _export_sc_param_value(row, in_data_elem, col)
        if v is not None:
            parts.append(f", {col}={_export_py_literal(v)}")
    return ''.join(parts)


def _electrisim_find_in_data_by_ext_grid(in_data, ext_name, ext_id=None):
    if not isinstance(in_data, dict):
        return None
    for _k, el in in_data.items():
        if not isinstance(el, dict):
            continue
        typ = str(el.get('typ') or '')
        if not (typ.startswith('External Grid') or typ.startswith('ExternalGrid')):
            continue
        if ext_id is not None and el.get('id') is not None and str(el.get('id')) == str(ext_id):
            return el
        if el.get('name') == ext_name or el.get('userFriendlyName') == ext_name:
            return el
    return None


def generate_pandapower_python_code(net, in_data, Busbars, algorithm, calculate_voltage_angles, init,
                                    study='powerflow', sc_in_data=None):
    """Generate Python code to recreate the pandapower network (load flow or short circuit)."""
    lines = []
    is_sc = study == 'shortcircuit'
    
    # Add header
    lines.append("# Pandapower Network Model")
    lines.append("# Auto-generated code to recreate the network")
    if is_sc:
        lines.append("# Short-circuit study (IEC 60909 via pandapower.shortcircuit.calc_sc)")
    lines.append("")
    lines.append("import pandapower as pp")
    if is_sc:
        lines.append("import pandapower.shortcircuit as sc")
    lines.append("")
    
    # Get frequency from network
    frequency = net.f_hz
    lines.append(f"# Create empty network with {frequency} Hz")
    lines.append(f"net = pp.create_empty_network(f_hz={frequency})")
    lines.append("")
    
    # Create buses
    lines.append("# Create buses")
    for idx, row in net.bus.iterrows():
        name = row['name'] if 'name' in row else f"Bus_{idx}"
        vn_kv = row['vn_kv']
        lines.append(f"bus_{idx} = pp.create_bus(net, vn_kv={vn_kv}, name='{name}')")
    lines.append("")
    
    # Create external grids
    if not net.ext_grid.empty:
        lines.append("# Create external grids")
        for idx, row in net.ext_grid.iterrows():
            bus = row['bus']
            vm_pu = row['vm_pu']
            va_degree = row['va_degree']
            name = row['name'] if 'name' in row else f"ExtGrid_{idx}"
            ext_id = row.get('id') if hasattr(row, 'get') else (row['id'] if 'id' in row.index else None)
            in_elem = _electrisim_find_in_data_by_ext_grid(in_data, name, ext_id)
            sc_kwargs = _ext_grid_sc_kwargs_for_export(row, in_elem)
            lines.append(
                f"pp.create_ext_grid(net, bus=bus_{bus}, vm_pu={vm_pu}, va_degree={va_degree}, "
                f"name='{name}'{sc_kwargs}{_export_extra_kwargs('ext_grid', row)})"
            )
        lines.append("")
    
    # Create lines
    if not net.line.empty:
        lines.append("# Create lines")
        for idx, row in net.line.iterrows():
            from_bus = row['from_bus']
            to_bus = row['to_bus']
            name = row['name'] if 'name' in row else f"Line_{idx}"
            
            # Get all line parameters for create_line_from_parameters
            line_id = row.get('id', name)  # Use name as fallback if id not stored
            r_ohm_per_km = row.get('r_ohm_per_km', 0.122)
            x_ohm_per_km = row.get('x_ohm_per_km', 0.112)
            c_nf_per_km = row.get('c_nf_per_km', 304.0)
            g_us_per_km = row.get('g_us_per_km', 0.0)
            max_i_ka = row.get('max_i_ka', 1.0)
            line_type = row.get('type', 'ol')
            length_km = row['length_km']
            
            # Get optional parameters if they exist
            parallel = row.get('parallel', 1)
            df = row.get('df', 1.0)
            
            # Build the create_line_from_parameters call with all parameters
            # Note: 'id' is included as a comment since pandapower doesn't store it natively
            lines.append(f"# Line ID: {line_id}")
            line_code = (f"pp.create_line_from_parameters(net, from_bus=bus_{from_bus}, to_bus=bus_{to_bus}, "
                        f"length_km={length_km}, r_ohm_per_km={r_ohm_per_km}, x_ohm_per_km={x_ohm_per_km}, "
                        f"c_nf_per_km={c_nf_per_km}, g_us_per_km={g_us_per_km}, max_i_ka={max_i_ka}, "
                        f"type='{line_type}', parallel={parallel}, df={df}, name='{name}'"
                        f"{_export_extra_kwargs('line', row)})")
            lines.append(line_code)
        lines.append("")
    
    # Create transformers
    if not net.trafo.empty:
        lines.append("# Create transformers (2-winding)")
        for idx, row in net.trafo.iterrows():
            hv_bus = row['hv_bus']
            lv_bus = row['lv_bus']
            name = row['name'] if 'name' in row else f"Trafo_{idx}"
            
            # Get transformer ID
            trafo_id = row.get('id', name)
            
            # Get all transformer parameters for create_transformer_from_parameters
            sn_mva = row.get('sn_mva', 1.0)
            vn_hv_kv = row.get('vn_hv_kv', 110.0)
            vn_lv_kv = row.get('vn_lv_kv', 20.0)
            vk_percent = row.get('vk_percent', 6.0)
            vkr_percent = row.get('vkr_percent', 1.0)
            pfe_kw = row.get('pfe_kw', 0.0)
            i0_percent = row.get('i0_percent', 0.0)
            
            # Get optional parameters
            parallel = row.get('parallel', 1)
            shift_degree = row.get('shift_degree', 0.0)
            tap_side = _export_tap_side(row.get('tap_side', 'hv'))
            tap_pos = row.get('tap_pos', 0)
            tap_neutral = row.get('tap_neutral', 0)
            tap_max = row.get('tap_max', 0)
            tap_min = row.get('tap_min', 0)
            tap_step_percent = row.get('tap_step_percent', 0.0)
            tap_step_degree = row.get('tap_step_degree', 0.0)
            vector_group = row.get('vector_group', 'Dyn')
            
            # Get zero sequence parameters if available
            vk0_percent = row.get('vk0_percent', vk_percent)
            vkr0_percent = row.get('vkr0_percent', vkr_percent)
            mag0_percent = row.get('mag0_percent', 0.0)
            mag0_rx = row.get('mag0_rx', 0.0)
            si0_hv_partial = row.get('si0_hv_partial', 0.0)
            
            # Get tap_changer_type (pandapower 3.0+)
            tap_changer_type = row.get('tap_changer_type', 'Ratio')
            
            # Build the create_transformer_from_parameters call
            lines.append(f"# Transformer ID: {trafo_id}")
            trafo_code = (f"pp.create_transformer_from_parameters(net, hv_bus=bus_{hv_bus}, lv_bus=bus_{lv_bus}, "
                         f"sn_mva={sn_mva}, vn_hv_kv={vn_hv_kv}, vn_lv_kv={vn_lv_kv}, "
                         f"vkr_percent={vkr_percent}, vk_percent={vk_percent}, "
                         f"pfe_kw={pfe_kw}, i0_percent={i0_percent}, "
                         f"parallel={parallel}, shift_degree={shift_degree}, "
                         f"tap_side='{tap_side}', tap_pos={tap_pos}, tap_neutral={tap_neutral}, "
                         f"tap_max={tap_max}, tap_min={tap_min}, "
                         f"tap_step_percent={tap_step_percent}, tap_step_degree={tap_step_degree}, "
                         f"tap_changer_type='{tap_changer_type}', "
                         f"vector_group='{vector_group}', "
                         f"vk0_percent={vk0_percent}, vkr0_percent={vkr0_percent}, "
                         f"mag0_percent={mag0_percent}, mag0_rx={mag0_rx}, "
                         f"si0_hv_partial={si0_hv_partial}, name='{name}'"
                         f"{_export_extra_kwargs('trafo', row)})")
            lines.append(trafo_code)
        lines.append("")
    
    # Create three-winding transformers
    if hasattr(net, 'trafo3w') and not net.trafo3w.empty:
        lines.append("# Create transformers (3-winding)")
        for idx, row in net.trafo3w.iterrows():
            hv_bus = row['hv_bus']
            mv_bus = row['mv_bus']
            lv_bus = row['lv_bus']
            name = row['name'] if 'name' in row else f"Trafo3W_{idx}"
            
            # Get transformer ID
            trafo3w_id = row.get('id', name)
            
            # Get all 3-winding transformer parameters
            sn_hv_mva = row.get('sn_hv_mva', 1.0)
            sn_mv_mva = row.get('sn_mv_mva', 1.0)
            sn_lv_mva = row.get('sn_lv_mva', 1.0)
            vn_hv_kv = row.get('vn_hv_kv', 110.0)
            vn_mv_kv = row.get('vn_mv_kv', 30.0)
            vn_lv_kv = row.get('vn_lv_kv', 10.0)
            vk_hv_percent = row.get('vk_hv_percent', 10.0)
            vk_mv_percent = row.get('vk_mv_percent', 10.0)
            vk_lv_percent = row.get('vk_lv_percent', 10.0)
            vkr_hv_percent = row.get('vkr_hv_percent', 0.5)
            vkr_mv_percent = row.get('vkr_mv_percent', 0.5)
            vkr_lv_percent = row.get('vkr_lv_percent', 0.5)
            pfe_kw = row.get('pfe_kw', 0.0)
            i0_percent = row.get('i0_percent', 0.0)
            
            # Get optional parameters
            shift_mv_degree = row.get('shift_mv_degree', 0.0)
            shift_lv_degree = row.get('shift_lv_degree', 0.0)
            tap_side = _export_tap_side(row.get('tap_side', 'hv'))
            tap_pos = row.get('tap_pos', 0)
            tap_neutral = row.get('tap_neutral', 0)
            tap_min = row.get('tap_min', 0)
            tap_max = row.get('tap_max', 0)
            tap_step_percent = row.get('tap_step_percent', 0.0)
            tap_step_degree = row.get('tap_step_degree', 0.0)
            vector_group = row.get('vector_group', 'YNyn')
            
            # Get zero sequence parameters if available
            vk0_hv_percent = row.get('vk0_hv_percent', vk_hv_percent)
            vk0_mv_percent = row.get('vk0_mv_percent', vk_mv_percent)
            vk0_lv_percent = row.get('vk0_lv_percent', vk_lv_percent)
            vkr0_hv_percent = row.get('vkr0_hv_percent', vkr_hv_percent)
            vkr0_mv_percent = row.get('vkr0_mv_percent', vkr_mv_percent)
            vkr0_lv_percent = row.get('vkr0_lv_percent', vkr_lv_percent)
            
            # Get tap_changer_type (pandapower 3.0+)
            tap_changer_type_3w = row.get('tap_changer_type', 'Ratio')
            
            # Build the create_transformer3w_from_parameters call
            lines.append(f"# Three-Winding Transformer ID: {trafo3w_id}")
            trafo3w_code = (f"pp.create_transformer3w_from_parameters(net, "
                           f"hv_bus=bus_{hv_bus}, mv_bus=bus_{mv_bus}, lv_bus=bus_{lv_bus}, "
                           f"sn_hv_mva={sn_hv_mva}, sn_mv_mva={sn_mv_mva}, sn_lv_mva={sn_lv_mva}, "
                           f"vn_hv_kv={vn_hv_kv}, vn_mv_kv={vn_mv_kv}, vn_lv_kv={vn_lv_kv}, "
                           f"vk_hv_percent={vk_hv_percent}, vk_mv_percent={vk_mv_percent}, vk_lv_percent={vk_lv_percent}, "
                           f"vkr_hv_percent={vkr_hv_percent}, vkr_mv_percent={vkr_mv_percent}, vkr_lv_percent={vkr_lv_percent}, "
                           f"pfe_kw={pfe_kw}, i0_percent={i0_percent}, "
                           f"shift_mv_degree={shift_mv_degree}, shift_lv_degree={shift_lv_degree}, "
                           f"tap_side='{tap_side}', tap_neutral={tap_neutral}, tap_pos={tap_pos}, tap_min={tap_min}, tap_max={tap_max}, "
                           f"tap_step_percent={tap_step_percent}, tap_step_degree={tap_step_degree}, tap_changer_type='{tap_changer_type_3w}', "
                           f"vector_group='{vector_group}', "
                           f"vk0_hv_percent={vk0_hv_percent}, vk0_mv_percent={vk0_mv_percent}, vk0_lv_percent={vk0_lv_percent}, "
                           f"vkr0_hv_percent={vkr0_hv_percent}, vkr0_mv_percent={vkr0_mv_percent}, vkr0_lv_percent={vkr0_lv_percent}, "
                           f"name='{name}'{_export_extra_kwargs('trafo3w', row)})")
            lines.append(trafo3w_code)
        lines.append("")
    
    # Create loads
    if not net.load.empty:
        lines.append("# Create loads")
        for idx, row in net.load.iterrows():
            bus = row['bus']
            p_mw = row['p_mw']
            q_mvar = row['q_mvar']
            name = row['name'] if 'name' in row else f"Load_{idx}"
            lines.append(f"pp.create_load(net, bus=bus_{bus}, p_mw={p_mw}, q_mvar={q_mvar}, name='{name}'"
                         f"{_export_extra_kwargs('load', row)})")
        lines.append("")
    
    # Create static generators
    if not net.sgen.empty:
        lines.append("# Create static generators")
        for idx, row in net.sgen.iterrows():
            bus = row['bus']
            p_mw = row['p_mw']
            q_mvar = row['q_mvar']
            name = row['name'] if 'name' in row else f"SGen_{idx}"
            sgen_id = row.get('id', None) if hasattr(row, 'get') else (row['id'] if 'id' in row.index else None)
            src = _electrisim_find_in_data_by_sgen(in_data, name, sgen_id)
            if src is not None and str(src.get('typ') or '').startswith('Wind Turbine'):
                lines.append(
                    f"# Wind Turbine '{name}': wind_speed_ms={src.get('wind_speed_ms')}, "
                    f"p_mw from curve/Pref"
                    + (f" (controller={src.get('_wind_controller')!r})" if src.get('_wind_controller') else "")
                )
            sn_part = ''
            if 'sn_mva' in row.index:
                sn = row.get('sn_mva')
                if sn is not None and sn == sn:
                    sn_part = f", sn_mva={float(sn)}"
            scale_part = ''
            if 'scaling' in row.index:
                sc = row.get('scaling')
                if sc is not None and sc == sc:
                    scale_part = f", scaling={float(sc)}"
            type_part = ''
            if 'type' in row.index:
                st = row.get('type')
                if st is not None and str(st).strip():
                    type_part = f", type={str(st)!r}"
            lines.append(
                f"pp.create_sgen(net, bus=bus_{bus}, p_mw={p_mw}, q_mvar={q_mvar}, name='{name}'"
                f"{sn_part}{scale_part}{type_part}{_export_extra_kwargs('sgen', row)})"
            )
        lines.append("")
    
    # Create generators
    if not net.gen.empty:
        lines.append("# Create generators")
        for idx, row in net.gen.iterrows():
            bus = row['bus']
            p_mw = row['p_mw']
            vm_pu = row['vm_pu']
            name = row['name'] if 'name' in row else f"Gen_{idx}"
            lines.append(f"pp.create_gen(net, bus=bus_{bus}, p_mw={p_mw}, vm_pu={vm_pu}, name='{name}'"
                         f"{_export_extra_kwargs('gen', row)})")
        lines.append("")
    
    # Create shunts
    if not net.shunt.empty:
        lines.append("# Create shunts")
        for idx, row in net.shunt.iterrows():
            bus = row['bus']
            q_mvar = row['q_mvar']
            p_mw = row['p_mw']
            name = row['name'] if 'name' in row else f"Shunt_{idx}"
            lines.append(f"pp.create_shunt(net, bus=bus_{bus}, q_mvar={q_mvar}, p_mw={p_mw}, name='{name}'"
                         f"{_export_extra_kwargs('shunt', row)})")
        lines.append("")
    
    # Create storage elements
    if hasattr(net, 'storage') and not net.storage.empty:
        lines.append("# Create storage elements")
        for idx, row in net.storage.iterrows():
            bus = row['bus']
            name = row['name'] if 'name' in row else f"Storage_{idx}"
            p_mw = row['p_mw']
            q_mvar = row.get('q_mvar', 0.0)
            sn_mva = row.get('sn_mva', 1.0)
            scaling = row.get('scaling', 1.0)
            storage_type = row.get('type', '')
            max_e_mwh = row.get('max_e_mwh', 1.0)
            min_e_mwh = row.get('min_e_mwh', 0.0)
            soc_percent = row.get('soc_percent', 50.0)
            in_service = row.get('in_service', True)
            controllable = row.get('controllable', False)
            max_p_mw = row.get('max_p_mw')
            min_p_mw = row.get('min_p_mw')
            max_q_mvar = row.get('max_q_mvar')
            min_q_mvar = row.get('min_q_mvar')
            
            # Build the create_storage call with all parameters
            storage_code = (f"pp.create_storage(net, bus=bus_{bus}, name='{name}', "
                          f"p_mw={p_mw}, q_mvar={q_mvar}, sn_mva={sn_mva}, "
                          f"scaling={scaling}, type='{storage_type}', "
                          f"max_e_mwh={max_e_mwh}, min_e_mwh={min_e_mwh}, "
                          f"soc_percent={soc_percent}, in_service={in_service}, "
                          f"controllable={controllable}")
            import pandas as pd
            if max_p_mw is not None and not (isinstance(max_p_mw, float) and pd.isna(max_p_mw)):
                storage_code += f", max_p_mw={max_p_mw}"
            if min_p_mw is not None and not (isinstance(min_p_mw, float) and pd.isna(min_p_mw)):
                storage_code += f", min_p_mw={min_p_mw}"
            if max_q_mvar is not None and not (isinstance(max_q_mvar, float) and pd.isna(max_q_mvar)):
                storage_code += f", max_q_mvar={max_q_mvar}"
            if min_q_mvar is not None and not (isinstance(min_q_mvar, float) and pd.isna(min_q_mvar)):
                storage_code += f", min_q_mvar={min_q_mvar}"
            storage_code += ")"
            lines.append(storage_code)
        lines.append("")
    
    # Create DC buses
    if hasattr(net, 'bus_dc') and not net.bus_dc.empty:
        lines.append("# Create DC buses")
        for idx, row in net.bus_dc.iterrows():
            name = row['name'] if 'name' in row else f"BusDC_{idx}"
            vn_kv = row.get('vn_kv', 0.0)
            in_service = row.get('in_service', True)
            lines.append(f"bus_dc_{idx} = pp.create_dc_bus(net, vn_kv={vn_kv}, name='{name}', in_service={in_service})")
        lines.append("")
    
    # Create asymmetric static generators
    if hasattr(net, 'asymmetric_sgen') and not net.asymmetric_sgen.empty:
        lines.append("# Create asymmetric static generators")
        for idx, row in net.asymmetric_sgen.iterrows():
            bus = row['bus']
            name = row['name'] if 'name' in row else f"AsymSGen_{idx}"
            p_a_mw = row.get('p_a_mw', 0.0)
            p_b_mw = row.get('p_b_mw', 0.0)
            p_c_mw = row.get('p_c_mw', 0.0)
            q_a_mvar = row.get('q_a_mvar', 0.0)
            q_b_mvar = row.get('q_b_mvar', 0.0)
            q_c_mvar = row.get('q_c_mvar', 0.0)
            sn_mva = row.get('sn_mva', 1.0)
            scaling = row.get('scaling', 1.0)
            sgen_type = row.get('type', 'current_source')
            in_service = row.get('in_service', True)
            lines.append(f"pp.create_asymmetric_sgen(net, bus=bus_{bus}, name='{name}', "
                        f"p_a_mw={p_a_mw}, p_b_mw={p_b_mw}, p_c_mw={p_c_mw}, "
                        f"q_a_mvar={q_a_mvar}, q_b_mvar={q_b_mvar}, q_c_mvar={q_c_mvar}, "
                        f"sn_mva={sn_mva}, scaling={scaling}, type='{sgen_type}', in_service={in_service})")
        lines.append("")
    
    # Create asymmetric loads
    if hasattr(net, 'asymmetric_load') and not net.asymmetric_load.empty:
        lines.append("# Create asymmetric loads")
        for idx, row in net.asymmetric_load.iterrows():
            bus = row['bus']
            name = row['name'] if 'name' in row else f"AsymLoad_{idx}"
            p_a_mw = row.get('p_a_mw', 0.0)
            p_b_mw = row.get('p_b_mw', 0.0)
            p_c_mw = row.get('p_c_mw', 0.0)
            q_a_mvar = row.get('q_a_mvar', 0.0)
            q_b_mvar = row.get('q_b_mvar', 0.0)
            q_c_mvar = row.get('q_c_mvar', 0.0)
            sn_mva = row.get('sn_mva', 1.0)
            scaling = row.get('scaling', 1.0)
            load_type = row.get('type', 'wye')
            in_service = row.get('in_service', True)
            lines.append(f"pp.create_asymmetric_load(net, bus=bus_{bus}, name='{name}', "
                        f"p_a_mw={p_a_mw}, p_b_mw={p_b_mw}, p_c_mw={p_c_mw}, "
                        f"q_a_mvar={q_a_mvar}, q_b_mvar={q_b_mvar}, q_c_mvar={q_c_mvar}, "
                        f"sn_mva={sn_mva}, scaling={scaling}, type='{load_type}', in_service={in_service})")
        lines.append("")
    
    # Create impedance elements
    if hasattr(net, 'impedance') and not net.impedance.empty:
        lines.append("# Create impedance elements")
        for idx, row in net.impedance.iterrows():
            from_bus = row['from_bus']
            to_bus = row['to_bus']
            name = row['name'] if 'name' in row else f"Impedance_{idx}"
            rft_pu = row.get('rft_pu', 0.0)
            xft_pu = row.get('xft_pu', 0.0)
            sn_mva = row.get('sn_mva', 1.0)
            in_service = row.get('in_service', True)
            lines.append(f"pp.create_impedance(net, from_bus=bus_{from_bus}, to_bus=bus_{to_bus}, "
                        f"name='{name}', rft_pu={rft_pu}, xft_pu={xft_pu}, sn_mva={sn_mva}, in_service={in_service})")
        lines.append("")
    
    # Create ward elements
    if hasattr(net, 'ward') and not net.ward.empty:
        lines.append("# Create ward elements")
        for idx, row in net.ward.iterrows():
            bus = row['bus']
            name = row['name'] if 'name' in row else f"Ward_{idx}"
            ps_mw = row.get('ps_mw', 0.0)
            qs_mvar = row.get('qs_mvar', 0.0)
            pz_mw = row.get('pz_mw', 0.0)
            qz_mvar = row.get('qz_mvar', 0.0)
            in_service = row.get('in_service', True)
            lines.append(f"pp.create_ward(net, bus=bus_{bus}, name='{name}', "
                        f"ps_mw={ps_mw}, qs_mvar={qs_mvar}, pz_mw={pz_mw}, qz_mvar={qz_mvar}, in_service={in_service})")
        lines.append("")
    
    # Create extended ward elements
    if hasattr(net, 'xward') and not net.xward.empty:
        lines.append("# Create extended ward elements")
        for idx, row in net.xward.iterrows():
            bus = row['bus']
            name = row['name'] if 'name' in row else f"XWard_{idx}"
            ps_mw = row.get('ps_mw', 0.0)
            qs_mvar = row.get('qs_mvar', 0.0)
            pz_mw = row.get('pz_mw', 0.0)
            qz_mvar = row.get('qz_mvar', 0.0)
            r_ohm = row.get('r_ohm', 0.0)
            x_ohm = row.get('x_ohm', 0.0)
            vm_pu = row.get('vm_pu', 1.0)
            in_service = row.get('in_service', True)
            lines.append(f"pp.create_xward(net, bus=bus_{bus}, name='{name}', "
                        f"ps_mw={ps_mw}, qs_mvar={qs_mvar}, pz_mw={pz_mw}, qz_mvar={qz_mvar}, "
                        f"r_ohm={r_ohm}, x_ohm={x_ohm}, vm_pu={vm_pu}, in_service={in_service})")
        lines.append("")
    
    # Create motor elements
    if hasattr(net, 'motor') and not net.motor.empty:
        lines.append("# Create motor elements")
        for idx, row in net.motor.iterrows():
            bus = row['bus']
            name = row['name'] if 'name' in row else f"Motor_{idx}"
            lines.append(f"pp.create_motor(net, bus=bus_{bus}, name='{name}'"
                         f"{_export_extra_kwargs('motor', row)})")
        lines.append("")
    
    # Create SVC elements
    if hasattr(net, 'svc') and not net.svc.empty:
        lines.append("# Create SVC (Static Var Compensator) elements")
        for idx, row in net.svc.iterrows():
            bus = row['bus']
            name = row['name'] if 'name' in row else f"SVC_{idx}"
            x_l_ohm = row.get('x_l_ohm', 0.0)
            x_cvar_ohm = row.get('x_cvar_ohm', 0.0)
            set_vm_pu = row.get('set_vm_pu', 1.0)
            thyristor_firing_angle_degree = row.get('thyristor_firing_angle_degree', 90.0)
            controllable = row.get('controllable', True)
            min_angle_degree = row.get('min_angle_degree', 90.0)
            max_angle_degree = row.get('max_angle_degree', 180.0)
            in_service = row.get('in_service', True)
            lines.append(f"pp.create_svc(net, bus=bus_{bus}, name='{name}', "
                        f"x_l_ohm={x_l_ohm}, x_cvar_ohm={x_cvar_ohm}, set_vm_pu={set_vm_pu}, "
                        f"thyristor_firing_angle_degree={thyristor_firing_angle_degree}, controllable={controllable}, "
                        f"min_angle_degree={min_angle_degree}, max_angle_degree={max_angle_degree}, in_service={in_service})")
        lines.append("")
    
    # Create TCSC elements
    if hasattr(net, 'tcsc') and not net.tcsc.empty:
        lines.append("# Create TCSC (Thyristor-Controlled Series Capacitor) elements")
        for idx, row in net.tcsc.iterrows():
            from_bus = row['from_bus']
            to_bus = row['to_bus']
            name = row['name'] if 'name' in row else f"TCSC_{idx}"
            x_l_ohm = row.get('x_l_ohm', 0.0)
            x_cvar_ohm = row.get('x_cvar_ohm', 0.0)
            set_p_to_mw = row.get('set_p_to_mw', 0.0)
            thyristor_firing_angle_degree = row.get('thyristor_firing_angle_degree', 90.0)
            controllable = row.get('controllable', True)
            min_angle_degree = row.get('min_angle_degree', 90.0)
            max_angle_degree = row.get('max_angle_degree', 180.0)
            in_service = row.get('in_service', True)
            lines.append(f"pp.create_tcsc(net, from_bus=bus_{from_bus}, to_bus=bus_{to_bus}, name='{name}', "
                        f"x_l_ohm={x_l_ohm}, x_cvar_ohm={x_cvar_ohm}, set_p_to_mw={set_p_to_mw}, "
                        f"thyristor_firing_angle_degree={thyristor_firing_angle_degree}, controllable={controllable}, "
                        f"min_angle_degree={min_angle_degree}, max_angle_degree={max_angle_degree}, in_service={in_service})")
        lines.append("")
    
    # Create SSC elements
    if hasattr(net, 'ssc') and not net.ssc.empty:
        lines.append("# Create SSC (Static Synchronous Compensator) elements")
        for idx, row in net.ssc.iterrows():
            bus = row['bus']
            name = row['name'] if 'name' in row else f"SSC_{idx}"
            r_ohm = row.get('r_ohm', 0.0)
            x_ohm = row.get('x_ohm', 0.0)
            set_vm_pu = row.get('set_vm_pu', 1.0)
            vm_internal_pu = row.get('vm_internal_pu', 1.0)
            va_internal_degree = row.get('va_internal_degree', 0.0)
            controllable = row.get('controllable', True)
            in_service = row.get('in_service', True)
            lines.append(f"pp.create_ssc(net, bus=bus_{bus}, name='{name}', "
                        f"r_ohm={r_ohm}, x_ohm={x_ohm}, set_vm_pu={set_vm_pu}, "
                        f"vm_internal_pu={vm_internal_pu}, va_internal_degree={va_internal_degree}, "
                        f"controllable={controllable}, in_service={in_service})")
        lines.append("")
    
    # Create load DC elements
    if hasattr(net, 'load_dc') and not net.load_dc.empty:
        lines.append("# Create DC load elements")
        for idx, row in net.load_dc.iterrows():
            bus = row['bus']
            name = row['name'] if 'name' in row else f"LoadDC_{idx}"
            p_mw = row.get('p_mw', 0.0)
            in_service = row.get('in_service', True)
            lines.append(f"pp.create_load_dc(net, bus=bus_{bus}, name='{name}', p_mw={p_mw}, in_service={in_service})")
        lines.append("")
    
    # Create source DC elements
    if hasattr(net, 'source_dc') and not net.source_dc.empty:
        lines.append("# Create DC source elements")
        for idx, row in net.source_dc.iterrows():
            bus = row['bus']
            name = row['name'] if 'name' in row else f"SourceDC_{idx}"
            vm_pu = row.get('vm_pu', 1.0)
            in_service = row.get('in_service', True)
            lines.append(f"pp.create_source_dc(net, bus=bus_{bus}, name='{name}', vm_pu={vm_pu}, in_service={in_service})")
        lines.append("")
    
    # Create switch elements (no in_service - Switch is always in service)
    if hasattr(net, 'switch') and not net.switch.empty:
        lines.append("# Create switch elements")
        for idx, row in net.switch.iterrows():
            bus = row['bus']
            element = row['element']
            et = row.get('et', 'l')
            name = row['name'] if 'name' in row else f"Switch_{idx}"
            closed = row.get('closed', True)
            switch_type = row.get('type', 'CB')
            z_ohm = row.get('z_ohm', 0.0)
            in_ka = row.get('in_ka', 0.0)
            lines.append(f"pp.create_switch(net, bus=bus_{bus}, element={element}, et='{et}', name='{name}', "
                        f"closed={closed}, type='{switch_type}', z_ohm={z_ohm}, in_ka={in_ka})")
        lines.append("")
    
    # Create VSC elements
    if hasattr(net, 'vsc') and not net.vsc.empty:
        lines.append("# Create VSC (Voltage Source Converter) elements")
        for idx, row in net.vsc.iterrows():
            bus = row['bus']
            bus_dc = row.get('bus_dc')
            name = row['name'] if 'name' in row else f"VSC_{idx}"
            p_mw = row.get('p_mw', 0.0)
            vm_pu = row.get('vm_pu', 1.0)
            sn_mva = row.get('sn_mva', 0.0)
            rx = row.get('rx', 0.1)
            max_ik_ka = row.get('max_ik_ka', 0.0)
            in_service = row.get('in_service', True)
            # bus_dc refers to a DC bus index, so use bus_dc_{bus_dc} variable name
            bus_dc_var = f"bus_dc_{bus_dc}" if bus_dc is not None else "None"
            lines.append(f"pp.create_vsc(net, bus=bus_{bus}, bus_dc={bus_dc_var}, name='{name}', "
                        f"p_mw={p_mw}, vm_pu={vm_pu}, sn_mva={sn_mva}, rx={rx}, max_ik_ka={max_ik_ka}, in_service={in_service})")
        lines.append("")
    
    # Create B2B VSC elements
    if hasattr(net, 'b2b_vsc') and not net.b2b_vsc.empty:
        lines.append("# Create B2B VSC (Back-to-Back Voltage Source Converter) elements")
        for idx, row in net.b2b_vsc.iterrows():
            bus1 = row['bus1']
            bus2 = row['bus2']
            name = row['name'] if 'name' in row else f"B2BVSC_{idx}"
            p_mw = row.get('p_mw', 0.0)
            vm1_pu = row.get('vm1_pu', 1.0)
            vm2_pu = row.get('vm2_pu', 1.0)
            sn_mva = row.get('sn_mva', 0.0)
            rx = row.get('rx', 0.1)
            max_ik_ka = row.get('max_ik_ka', 0.0)
            in_service = row.get('in_service', True)
            lines.append(f"pp.create_b2b_vsc(net, bus1=bus_{bus1}, bus2=bus_{bus2}, name='{name}', "
                        f"p_mw={p_mw}, vm1_pu={vm1_pu}, vm2_pu={vm2_pu}, sn_mva={sn_mva}, "
                        f"rx={rx}, max_ik_ka={max_ik_ka}, in_service={in_service})")
        lines.append("")
    
    # Create DC line elements
    if hasattr(net, 'dcline') and not net.dcline.empty:
        lines.append("# Create DC line elements")
        for idx, row in net.dcline.iterrows():
            from_bus = row['from_bus']
            to_bus = row['to_bus']
            name = row['name'] if 'name' in row else f"DCLine_{idx}"
            p_mw = row.get('p_mw', 0.0)
            loss_percent = row.get('loss_percent', 0.0)
            loss_mw = row.get('loss_mw', 0.0)
            vm_from_pu = row.get('vm_from_pu', 1.0)
            vm_to_pu = row.get('vm_to_pu', 1.0)
            in_service = row.get('in_service', True)
            lines.append(f"pp.create_dcline(net, from_bus=bus_{from_bus}, to_bus=bus_{to_bus}, name='{name}', "
                        f"p_mw={p_mw}, loss_percent={loss_percent}, loss_mw={loss_mw}, "
                        f"vm_from_pu={vm_from_pu}, vm_to_pu={vm_to_pu}, in_service={in_service})")
        lines.append("")
    
    # Electrisim Q capability + initial Q (before controllers)
    _append_electrisim_sgen_setup_python(lines, net)

    if is_sc:
        sc = sc_in_data or {}
        fault_type = sc.get('fault_type', '3ph')
        if fault_type not in ('3ph', '2ph', '1ph'):
            fault_type = '3ph'
        fault_location = sc.get('fault_location', 'max')
        if fault_location not in ('max', 'min'):
            fault_location = 'max'
        tk_s = float(sc.get('tk_s', 1.0))
        r_fault_ohm = float(sc.get('r_fault_ohm', 0.0))
        x_fault_ohm = float(sc.get('x_fault_ohm', 0.0))
        sc_bus = resolve_pp_fault_bus_indices(net, sc, Busbars)
        if sc_bus is not None and len(sc_bus) == 1:
            sc_bus = sc_bus[0]
        lines.append("# Short-circuit calculation (matches Electrisim IEC 60909 run)")
        # Each sgen's k is written on its create_sgen line: the 1.1 set here
        # for all of them overrode a ratio set on the element.
        lines.append("")
        lines.append("sc.calc_sc(")
        lines.append("    net,")
        lines.append(f"    fault={fault_type!r},")
        lines.append(f"    case={fault_location!r},")
        lines.append(f"    bus={_export_py_literal(sc_bus)},")
        lines.append("    ip=True,")
        lines.append("    ith=True,")
        lines.append(f"    tk_s={tk_s},")
        lines.append("    kappa_method='C',")
        lines.append(f"    r_fault_ohm={r_fault_ohm},")
        lines.append(f"    x_fault_ohm={x_fault_ohm},")
        # As the backend runs it: with False a fresh net fails in pandapower's
        # gen lookup (net._is_elements_final is never set).
        lines.append("    check_connectivity=True,")
        lines.append("    branch_results=True,")
        lines.append("    return_all_currents=False,")
        for key, value in _sc_iec_options(sc).items():
            lines.append(f"    {key}={value!r},")
        lines.append(")")
        lines.append("")
        lines.append("# Print short-circuit results")
        lines.append("print('\\nBus SC Results:')")
        lines.append("print(net.res_bus_sc)")
        lines.append("if hasattr(net, 'res_line_sc') and net.res_line_sc is not None and not net.res_line_sc.empty:")
        lines.append("    print('\\nLine SC Results:')")
        lines.append("    print(net.res_line_sc)")
        lines.append("if hasattr(net, 'res_trafo_sc') and net.res_trafo_sc is not None and not net.res_trafo_sc.empty:")
        lines.append("    print('\\nTransformer SC Results:')")
        lines.append("    print(net.res_trafo_sc)")
        return '\n'.join(lines)

    # Electrisim Park / Wind Turbine controllers (must run before runpp)
    run_control = _append_electrisim_controllers_to_python(
        lines, net, in_data, algorithm, calculate_voltage_angles, init
    )

    # Run power flow (seed LF for cosphi(P)/Q(V) parks, then final LF with controllers)
    cva_str = repr(calculate_voltage_angles)
    need_seed = bool(getattr(net, '_electrisim_export_park_need_seed', False))
    enforce_q = bool(getattr(net, '_electrisim_enforce_q_lims', False))

    if need_seed:
        lines.append("# Seed load flow (park cosphi(P)/Q(V) measurements)")
        lines.append(
            f"pp.runpp(net, algorithm='{algorithm}', calculate_voltage_angles={cva_str}, "
            f"init='{init}', run_control=False)"
        )
        lines.append("")

    lines.append("# Run power flow")
    run_kwargs = [
        f"algorithm='{algorithm}'",
        f"calculate_voltage_angles={cva_str}",
        f"init='{init}'",
    ]
    if run_control:
        run_kwargs.append("run_control=True")
    if enforce_q:
        run_kwargs.append("enforce_q_lims=True")
    lines.append(f"pp.runpp(net, {', '.join(run_kwargs)})")
    lines.append("")
    
    # Add results printing
    lines.append("# Print results")
    lines.append("print('\\nBus Results:')")
    lines.append("print(net.res_bus)")
    lines.append("print('\\nLine Results:')")
    lines.append("print(net.res_line)")
    lines.append("if hasattr(net, 'controller') and net.controller is not None and not net.controller.empty:")
    lines.append("    print('\\nControllers:')")
    lines.append("    print(net.controller)")
    
    return '\n'.join(lines)

# --- Voltage-dependent DC loads ---------------------------------------------------------
#
# pandapower's DC load draws a fixed power. A DC load can also be constant
# current, constant resistance, or a mix; its constant-power part changes to
# constant current below v_min_pu (a converter's input current limit). The
# model is kept as columns on net.load_dc, so it survives network copies, and
# _electrisim_runpp repeats the load flow until those loads settle.

_DC_LOAD_MODEL_COLUMNS = ('electrisim_p_rated_mw', 'electrisim_share_p', 'electrisim_share_i',
                          'electrisim_share_r', 'electrisim_v_min_pu')
_DC_LOAD_SHARES = {
    'constant_power': (1.0, 0.0, 0.0),
    'constant_current': (0.0, 1.0, 0.0),
    'constant_resistance': (0.0, 0.0, 1.0),
}


def _electrisim_dc_load_shares(el):
    """(constant power, constant current, constant resistance) shares of a DC load's rated power."""
    model = str(el.get('load_model') or 'constant_power').strip().lower()
    if model in _DC_LOAD_SHARES:
        return _DC_LOAD_SHARES[model]
    shares = [max(0.0, safe_float(el.get(k), d)) for k, d in
              (('share_p_percent', 100.0), ('share_i_percent', 0.0), ('share_r_percent', 0.0))]
    total = sum(shares)
    return tuple(x / total for x in shares) if total > 0 else (1.0, 0.0, 0.0)


def _electrisim_dc_load_power(p_rated, share_p, share_i, share_r, v_min, v):
    """A DC load's power (MW) at its bus voltage v (p.u.)."""
    v = max(float(v), 0.0)
    p_part = 1.0 if v >= v_min else (v / v_min if v_min > 0 else 1.0)
    return p_rated * (share_p * p_part + share_i * v + share_r * v * v)


def _electrisim_has_dc_load_models(net):
    ld = getattr(net, 'load_dc', None)
    return (ld is not None and len(ld) and 'electrisim_share_p' in ld.columns
            and bool((ld['electrisim_share_p'].fillna(1.0) < 1.0 - 1e-12).any()
                     | (ld['electrisim_v_min_pu'].fillna(0.0) > 0).any()))


def _electrisim_runpp(net, max_rounds=60, tolerance_mw=1e-9, **kwargs):
    """
    The load flow, with the microgrid sources and stores settled: each one's
    terminal voltage at the current it delivers, each directly connected
    PV array or SOFC at the power its curve gives at its bus's voltage, each
    DC/DC converter in droop at the voltage its power gives - repeating until
    they agree. Without them, _electrisim_runpp_converters alone.
    """
    droop = any(c.get('control') == 'droop' for c in getattr(net, 'electrisim_dc_dc_converters', None) or [])
    pcs = _electrisim_pcs_to_settle(net)
    if not getattr(net, 'electrisim_ders', None) and not droop and not pcs:
        return _electrisim_runpp_converters(net, max_rounds, tolerance_mw, **kwargs)
    for _ in range(60):
        _electrisim_runpp_converters(net, max_rounds, tolerance_mw, **kwargs)
        if max(_electrisim_settle_ders(net), _electrisim_settle_pcs(net)) <= 1e-8:
            return None
        kwargs = {**kwargs, 'init': 'results'}
    _electrisim_warn(net, 'The sources, stores and PCS had not settled after 60 load flows: '
                          'their voltages and powers are those of the last one.')
    return None


def _electrisim_runpp_converters(net, max_rounds=60, tolerance_mw=1e-9, **kwargs):
    """
    pp.runpp with the DC/DC converters' inputs matched to their outputs, and
    voltage-dependent DC loads settled: plain runpp when there is neither.

    Each round solves the load flow, then sets each converter's input to what
    its output delivered, divided by its efficiency, plus its no-load loss -
    until they agree. A chain of converters settles in a round per converter.
    """
    rows = _electrisim_dc_dc_rows(net)
    if not rows:
        return _electrisim_runpp_dc_loads(net, max_rounds, tolerance_mw, **kwargs)
    for _ in range(max_rounds):
        _electrisim_runpp_dc_loads(net, max_rounds, tolerance_mw, **kwargs)
        worst = 0.0
        for table, i in rows:
            df, col = net[table], _STAGE_P_COLUMN[table]
            p_out = _electrisim_dc_dc_output_mw(net, i, table)
            if not np.isfinite(p_out):
                continue
            target = _electrisim_dc_dc_input_power(p_out, float(df.at[i, 'electrisim_dcdc_eta']),
                                                   float(df.at[i, 'electrisim_dcdc_p_nl_mw']))
            worst = max(worst, abs(target - float(df.at[i, col])))
            df.at[i, col] = target
        if worst <= tolerance_mw:
            return None
        kwargs = {**kwargs, 'init': 'results'}
    _electrisim_warn(net, f"The DC/DC converters' inputs had not settled after {max_rounds} load flows.")
    return None


def _electrisim_runpp_dc_loads(net, max_rounds=60, tolerance_mw=1e-9, **kwargs):
    """
    pp.runpp with voltage-dependent DC loads settled: plain runpp when there
    is none.

    Each such load is solved for the power it draws at the voltage that power
    gives, g(p) = p - P(v(p)) = 0. A round sets each load to the power its
    voltage gave in the last one - which settles normal cases in a few rounds -
    while each load keeps a bracket on g's sign change and bisects when the
    next power would leave it. A load flow that does not solve counts as too
    much load. Near the voltage-collapse point, where repeating alone swings
    back and forth, the bracket still closes on the answer.
    """
    if not _electrisim_has_dc_load_models(net):
        return pp.runpp(net, **kwargs)
    ld = net.load_dc
    rows = [i for i in ld.index if pd.notna(ld.at[i, 'electrisim_share_p'])]
    lo = {i: 0.0 for i in rows}       # powers known to be too low (g < 0)
    hi = {i: None for i in rows}      # powers known to be too high (g > 0), once one is
    for _ in range(max_rounds):
        try:
            pp.runpp(net, **kwargs)
        except pp.LoadflowNotConverged:
            for i in rows:
                p = float(ld.at[i, 'p_dc_mw'])
                hi[i] = p if hi[i] is None else min(hi[i], p)
                ld.at[i, 'p_dc_mw'] = 0.5 * (lo[i] + hi[i])
            kwargs = {**kwargs, 'init': 'auto'}
            continue
        worst = 0.0
        proposals = {}
        for i in rows:
            bus = ld.at[i, 'bus_dc']
            v = net.res_bus_dc.at[bus, 'vm_pu'] if bus in net.res_bus_dc.index else np.nan
            if not np.isfinite(v):
                continue
            p = float(ld.at[i, 'p_dc_mw'])
            target = _electrisim_dc_load_power(*(float(ld.at[i, c]) for c in _DC_LOAD_MODEL_COLUMNS), v)
            worst = max(worst, abs(target - p))
            if target > p:
                lo[i] = max(lo[i], p)
            elif target < p:
                hi[i] = p if hi[i] is None else min(hi[i], p)
            inside = target > lo[i] and (hi[i] is None or target < hi[i])
            proposals[i] = target if inside else 0.5 * (lo[i] + (hi[i] if hi[i] is not None else 2 * p))
        if worst <= tolerance_mw:
            # The powers this round used are those its voltages give.
            return None
        for i, p_next in proposals.items():
            ld.at[i, 'p_dc_mw'] = p_next
        if kwargs.get('init') != 'results':
            kwargs = {**kwargs, 'init': 'results'}
    _electrisim_warn(net, f"Voltage-dependent DC loads had not settled after {max_rounds} load flows: "
                          "their powers are those of the last one.")
    return None


def _electrisim_warn(net, message):
    """A warning returned with the results (net.warnings), and printed."""
    if not hasattr(net, 'warnings'):
        net.warnings = []
    net.warnings.append(message)
    print(f"WARNING: {message}")


# --- DC/DC converters --------------------------------------------------------------------
#
# pandapower has no DC/DC converter, so one is built from parts it has. Its
# input is a DC load drawing what the output delivers divided by the
# efficiency, plus the no-load loss. Its output, in voltage mode, is a
# near-lossless VSC from an auxiliary AC grid holding the output voltage (a DC
# network held by a DC source alone does not converge); in power mode, an
# injection of the set power into an output network another element holds.
# _electrisim_runpp repeats the load flow until each input matches its
# output. The converter's data is kept as columns on its input load, so it
# survives network copies; the auxiliary elements are marked electrisim_aux
# and left out of the results.

_DCDC_AUX_R_PU, _DCDC_AUX_X_PU, _DCDC_AUX_RDC_PU = 1e-4, 1e-3, 1e-6
_DCDC_CONTROLS = ('voltage', 'power', 'droop', 'dispatch', 'mppt', 'follower', 'smoothing')


def _electrisim_is_pcs(df, index):
    """Whether a generator or static generator row is a PCS, reported with the PCS."""
    return 'electrisim_pcs' in df.columns and index in df.index and df.at[index, 'electrisim_pcs'] == True


def _electrisim_is_hidden(df, index):
    """Whether a row is a source's or store's own part, reported with it."""
    return 'electrisim_hidden' in df.columns and index in df.index and df.at[index, 'electrisim_hidden'] == True


def _electrisim_is_aux(df, index):
    """Whether a row is a DC/DC converter's auxiliary element, kept out of the results."""
    return 'electrisim_aux' in df.columns and index in df.index and df.at[index, 'electrisim_aux'] == True


def _electrisim_without_aux(df):
    """A results table without the DC/DC converters' auxiliary rows."""
    if df is None or 'electrisim_aux' not in getattr(df, 'columns', ()):
        return df
    return df[df['electrisim_aux'] != True]


def _electrisim_res_without_aux(net, table):
    """A copy of ``table``'s results without the converters' auxiliary rows."""
    res = net[f'res_{table}']
    df = net[table]
    if 'electrisim_aux' in df.columns:
        aux = df.index[df['electrisim_aux'] == True]
        res = res.drop(index=res.index.intersection(aux))
    return res.copy()


def _electrisim_dc_dc_input_power(p_out, eta, p_nl):
    """What a DC/DC converter draws (MW) for ``p_out`` delivered; losses either way."""
    return p_out / eta + p_nl if p_out >= 0 else p_out * eta + p_nl


def _electrisim_stage_columns(name, eta, p_nl_mw, output):
    """
    The columns that make a load a converter stage's input: _electrisim_runpp
    sets its power from what ``output`` (table, index) delivers.
    """
    return dict(electrisim_aux=True, electrisim_dcdc_role='input', electrisim_dcdc_name=name,
                electrisim_dcdc_eta=eta, electrisim_dcdc_p_nl_mw=p_nl_mw,
                electrisim_dcdc_out_table=output[0], electrisim_dcdc_out_idx=float(output[1]))


def _electrisim_aux_dc_source(net, bus_dc, vm_pu, rated_mw, name, in_service):
    """
    A near-lossless VSC from an auxiliary AC grid, holding ``bus_dc`` at
    ``vm_pu``: a converter's output stage. Returns (vsc, AC bus, grid).
    """
    vn = float(net.bus_dc.at[bus_dc, 'vn_kv'])
    z_base = vn ** 2 / max(rated_mw, 1.0)
    aux_bus = pp.create_bus(net, vn_kv=vn, name=f'{name} auxiliary AC', in_service=in_service)
    net.bus.at[aux_bus, 'electrisim_aux'] = True
    aux_grid = pp.create_ext_grid(net, aux_bus, vm_pu=1.0, name=f'{name} auxiliary grid', in_service=in_service,
                                  s_sc_max_mva=1000.0, s_sc_min_mva=1000.0, rx_max=0.1, rx_min=0.1)
    net.ext_grid.at[aux_grid, 'electrisim_aux'] = True
    vsc = pp.create_vsc(net, aux_bus, bus_dc, r_ohm=_DCDC_AUX_R_PU * z_base, x_ohm=_DCDC_AUX_X_PU * z_base,
                        r_dc_ohm=_DCDC_AUX_RDC_PU * z_base, control_mode_ac='q_mvar', control_value_ac=0.0,
                        control_mode_dc='vm_pu', control_value_dc=vm_pu, name=f'{name} output',
                        in_service=in_service)
    net.vsc.at[vsc, 'electrisim_aux'] = True
    return int(vsc), int(aux_bus), int(aux_grid)


def _electrisim_new_load_dc(net, bus, p_mw, name, in_service, **columns):
    idx = int(net.load_dc.index.max()) + 1 if len(net.load_dc) else 0
    pp.create_load_dc(net, bus_dc=bus, p_dc_mw=p_mw, name=name, in_service=in_service, index=idx)
    for key, value in columns.items():
        net.load_dc.at[idx, key] = value
    return idx


def _electrisim_build_dc_dc_converters(net):
    pending = getattr(net, '_electrisim_pending_dc_dc', None) or []
    net.electrisim_dc_dc_converters = []
    for el in pending:
        name = el.get('name')
        label = el.get('userFriendlyName') or name
        b_in, b_out = _electrisim_dc_bus(net, el.get('bus_in')), _electrisim_dc_bus(net, el.get('bus_out'))
        if b_in is None or b_out is None or b_in == b_out:
            _electrisim_warn(net, f"DC/DC Converter '{label}' needs a DC bus at its input and another at its output, "
                                  "so it is left out.")
            continue
        # Its control: voltage and power, and the modes its sources need - droop (voltage, lowered
        # with its power), dispatch (power), MPPT and follower (power: its PV array's maximum
        # power, its SOFC's set power) and smoothing (power: none in a load flow).
        control = str(el.get('control_mode') or 'voltage').strip().lower()
        if control not in _DCDC_CONTROLS:
            control = 'power' if control.startswith('p') else 'voltage'
        mode = 'voltage' if control in ('voltage', 'droop') else 'power'
        eta = safe_float(el.get('efficiency_percent'), 98.0) / 100.0
        if not 0 < eta <= 1:
            _electrisim_warn(net, f"DC/DC Converter '{label}': an efficiency of {100 * eta:g} % is taken as 100 %.")
            eta = 1.0
        rec = {
            'name': name, 'id': el.get('id', ''), 'label': label, 'mode': mode, 'bus_in': int(b_in), 'bus_out': int(b_out),
            'rated_mw': safe_float(el.get('rated_mw'), 0.0), 'eta': eta,
            'p_nl_mw': safe_float(el.get('no_load_loss_kw'), 0.0) / 1e3,
            'vm_out_pu': safe_float(el.get('vm_out_pu'), 1.0), 'p_set_mw': safe_float(el.get('p_set_mw'), 0.0),
            'bidirectional': _electrisim_flag(el.get('bidirectional'), False),
            'in_service': _electrisim_in_service(el),
            'control': control, 'droop_percent': safe_float(el.get('droop_percent'), 5.0),
            'smoothing': {'tau_s': safe_float(el.get('smoothing_tau_s'), 10.0),
                          'soc_ref_percent': safe_float(el.get('soc_ref_percent'), 50.0),
                          'soc_gain': safe_float(el.get('soc_gain'), 0.1)},
            'input': None, 'vsc': None, 'output_load': None, 'aux_bus': None, 'aux_ext_grid': None,
            # For the EMT study: its model (a dual active bridge), switching frequency, current limit, output capacitor.
            'emt': {'model': 'switching' if el.get('emt_model') == 'switching' else 'average',
                    'switching_khz': safe_float(el.get('switching_khz'), 20.0),
                    'current_limit_pu': safe_float(el.get('current_limit_pu'), 1.2),
                    'c_out_mf': safe_float(el.get('c_out_mf'), 0.0)},
        }
        for side, bus, key in (('input', b_in, 'vn_in_kv'), ('output', b_out, 'vn_out_kv')):
            rated_kv, vn = safe_float(el.get(key), 0.0), float(net.bus_dc.at[bus, 'vn_kv'])
            if rated_kv > 0 and abs(rated_kv - vn) > 0.1 * vn:
                _electrisim_warn(net, f"DC/DC Converter '{label}': its {side} is rated {rated_kv:g} kV, "
                                      f"its {side} bus is {vn:g} kV.")
        on = rec['in_service']
        if control == 'smoothing':
            rec['p_set_mw'] = 0.0      # its store's power follows its loads' swings: none in a load flow
        if mode == 'voltage':
            vsc, aux_bus, aux_grid = _electrisim_aux_dc_source(net, b_out, rec['vm_out_pu'], rec['rated_mw'], name, on)
            rec.update(vsc=vsc, aux_bus=aux_bus, aux_ext_grid=aux_grid)
            p_in0 = rec['p_nl_mw']
        else:
            out = _electrisim_new_load_dc(net, b_out, -rec['p_set_mw'], f'{name} output', on,
                                          electrisim_aux=True, electrisim_dcdc_role='output', electrisim_dcdc_name=name)
            rec['output_load'] = int(out)
            p_in0 = _electrisim_dc_dc_input_power(rec['p_set_mw'], eta, rec['p_nl_mw'])
        output = ('vsc', rec['vsc']) if rec['vsc'] is not None else ('load_dc', rec['output_load'])
        rec['input'] = int(_electrisim_new_load_dc(
            net, b_in, p_in0, f'{name} input', on, **_electrisim_stage_columns(name, eta, rec['p_nl_mw'], output)))
        net.electrisim_dc_dc_converters.append(rec)
        if not hasattr(net, 'user_friendly_names'):
            net.user_friendly_names = {}
        net.user_friendly_names[name] = label


def _electrisim_dc_dc_parts(rec):
    return [('load_dc', rec['input']), ('load_dc', rec['output_load']), ('vsc', rec['vsc']),
            ('ext_grid', rec['aux_ext_grid']), ('bus', rec['aux_bus'])]


def _electrisim_set_dc_dc_in_service(net, rec, on):
    rec['in_service'] = on
    for table, idx in _electrisim_dc_dc_parts(rec):
        if idx is not None and idx in net[table].index:
            net[table].at[idx, 'in_service'] = on


def _electrisim_finish_dc_dc(net):
    """
    Leave out a DC/DC converter one of whose networks was set aside: its
    input network must be held by a converter, and in power mode so must its
    output network. Its output network may then be set aside in turn.
    """
    removed = False
    kept = []
    for rec in getattr(net, 'electrisim_dc_dc_converters', None) or []:
        has_in = rec['input'] in net.load_dc.index
        has_out = (rec['vsc'] in net.vsc.index) if rec['vsc'] is not None else (rec['output_load'] in net.load_dc.index)
        if (has_in and has_out) or (has_in and not rec['in_service']):
            # Out of service (a breaker opened it), it stays to be reported so.
            kept.append(rec)
            continue
        for table, idx in _electrisim_dc_dc_parts(rec):
            if idx is not None and idx in net[table].index:
                net[table].drop(idx, inplace=True)
        side = 'input' if not has_in else 'output'
        _electrisim_warn(net, f"DC/DC Converter '{rec['label']}' is left out: its {side} network is held by no "
                              "converter" + (", and in power mode another element must hold its output voltage."
                                             if side == 'output' and rec['mode'] == 'power' else "."))
        removed = True
    net.electrisim_dc_dc_converters = kept
    if removed:
        _electrisim_drop_uncoupled_dc(net)
        for b in getattr(net, 'electrisim_dc_breakers', None) or []:
            if b['target'] is not None and b['target'][1] not in net[b['target'][0]].index:
                b['target'] = None


# The studies that model DC/DC converters; the others see each as the power its input draws.
_DCDC_STUDIES = ('PowerFlowPandaPower', 'TimeSeriesSimulationPandaPower', 'ContingencyAnalysisPandaPower', 'EmtStudy',
                 'DcFaultStudy')


def _electrisim_freeze_dc_dc(net):
    """
    For a study that does not model DC/DC converters: a load flow settles each
    converter's input, which then stays at that power, and the auxiliary grids
    and the DC networks they held are left out.
    """
    convs = getattr(net, 'electrisim_dc_dc_converters', None) or []
    ssts = getattr(net, 'electrisim_ssts', None) or []
    ders = getattr(net, 'electrisim_ders', None) or []
    if not convs and not ssts and not ders:
        return
    solved = None
    try:
        # On a copy: the study starts from a network without load-flow results.
        solved = deepcopy(net)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _electrisim_runpp(solved, algorithm='nr', calculate_voltage_angles=True, init='auto')
        for table, i in _electrisim_dc_dc_rows(solved):
            col = _STAGE_P_COLUMN[table]
            if i in net[table].index:
                net[table].at[i, col] = float(solved[table].at[i, col])
        for k, rec in enumerate(ders):
            if 'load' in rec['parts'] and rec['parts']['load'] in net.load_dc.index:
                net.load_dc.at[rec['parts']['load'], 'p_dc_mw'] = -_electrisim_der_power(solved, solved.electrisim_ders[k])[1] / 1e6
    except Exception:   # noqa: BLE001 - the inputs keep their first estimate
        _electrisim_warn(net, "The load flow that settles the DC/DC converters' inputs did not solve: "
                              "they draw their no-load loss, or their set power over their efficiency.")
    for table, i in _electrisim_dc_dc_rows(net):
        net[table].at[i, 'electrisim_dcdc_role'] = 'frozen'
    held = []
    for rec in ssts:
        # Its rectifier and DC/DC stage are the power the MV side draws; a
        # grid-forming inverter stays, as the source of its LV AC network.
        held.append({'bus_out': rec['bus_lvdc']})
        for table, idx in rec['aux']:
            if idx is not None and idx in net[table].index:
                net[table].drop(idx, inplace=True)
    for rec in convs:
        if rec['mode'] == 'voltage':
            held.append(rec)
            for table, idx in (('vsc', rec['vsc']), ('ext_grid', rec['aux_ext_grid']), ('bus', rec['aux_bus'])):
                if idx is not None and idx in net[table].index:
                    net[table].drop(idx, inplace=True)
    if held:
        names = getattr(net, 'user_friendly_names', {}) or {}
        buses = sorted({str(names.get(net.bus_dc.at[r['bus_out'], 'name'], net.bus_dc.at[r['bus_out'], 'name']))
                        for r in held if r['bus_out'] in net.bus_dc.index})
        _electrisim_warn(net, "This study does not model DC/DC converters or solid-state transformers: each is "
                              "the power its input draws in the load flow, and the DC networks they hold "
                              f"({', '.join(buses)}) are left out.")
        _electrisim_drop_uncoupled_dc(net, quiet=True)
        for b in getattr(net, 'electrisim_dc_breakers', None) or []:
            if b['target'] is not None and b['target'][1] not in net[b['target'][0]].index:
                b['target'] = None
    # The sources and stores: behind their converters, left out with the networks those held; a battery
    # directly on a bus, the power it delivered in the load flow.
    for k, rec in enumerate(ders):
        parts = rec['parts']
        if 'line' in parts:
            p = 0.0
            if solved is not None:
                try:
                    p = _electrisim_der_power(solved, solved.electrisim_ders[k])[1]
                except Exception:   # noqa: BLE001 - it then delivers nothing
                    p = 0.0
            _electrisim_new_load_dc(net, rec['bus'], -p / 1e6, f"{rec['name']} output", rec['obj'].in_service,
                                    electrisim_aux=True, electrisim_dcdc_role='frozen')
            if parts['line'] in net.line_dc.index:
                net.line_dc.drop(parts['line'], inplace=True)
        for table, key in (('vsc', 'vsc'), ('ext_grid', 'aux_grid'), ('bus', 'aux_bus'),
                           ('vsc', 'cells'), ('bus_dc', 'inner')):
            idx = parts.get(key)
            if idx is not None and idx in net[table].index:
                net[table].drop(idx, inplace=True)
    if ders:
        _electrisim_drop_uncoupled_dc(net, quiet=True)
    net.electrisim_ders = []
    net.electrisim_dc_dc_converters = []
    net.electrisim_ssts = []


_STAGE_P_COLUMN = {'load_dc': 'p_dc_mw', 'load': 'p_mw'}


def _electrisim_dc_dc_rows(net):
    """Each converter stage's input, as (table, index): a DC load, or an AC load (an SST's rectifier)."""
    rows = []
    for table in ('load_dc', 'load'):
        df = getattr(net, table, None)
        if df is None or not len(df) or 'electrisim_dcdc_role' not in df.columns:
            continue
        rows += [(table, i) for i in df.index
                 if df.at[i, 'electrisim_dcdc_role'] == 'input' and bool(df.at[i, 'in_service'])]
    return rows


def _electrisim_stage_output_mw(net, table, idx):
    """What a stage's output delivers (MW), from the last load flow."""
    if idx is None or not np.isfinite(idx):
        return np.nan
    idx = int(idx)
    if table == 'vsc':
        res = getattr(net, 'res_vsc', None)
        return -float(res.at[idx, 'p_dc_mw']) if res is not None and idx in res.index else np.nan
    if table == 'load_dc':
        return -float(net.load_dc.at[idx, 'p_dc_mw']) if idx in net.load_dc.index else np.nan
    if table == 'ext_grid':
        res = getattr(net, 'res_ext_grid', None)
        return float(res.at[idx, 'p_mw']) if res is not None and idx in res.index else np.nan
    if table == 'sgen':
        return float(net.sgen.at[idx, 'p_mw']) if idx in net.sgen.index else np.nan
    return np.nan


def _electrisim_dc_dc_output_mw(net, i, table='load_dc'):
    """What the stage whose input is row ``i`` of ``table`` delivers (MW)."""
    df = net[table]
    return _electrisim_stage_output_mw(net, df.at[i, 'electrisim_dcdc_out_table'], df.at[i, 'electrisim_dcdc_out_idx'])


def _electrisim_stage_result(net, stage):
    """A converter stage's input and output power, loss, efficiency and loading."""
    table, i = stage['input']
    out = {'stage': stage['stage'], 'rated_mw': stage['rated_mw'], 'p_in_mw': 0.0, 'p_out_mw': 0.0,
           'loss_mw': 0.0, 'efficiency_percent': None, 'loading_percent': 0.0}
    if i is None or i not in net[table].index or not bool(net[table].at[i, 'in_service']):
        return out
    p_in = float(net[table].at[i, _STAGE_P_COLUMN[table]])
    p_out = _electrisim_dc_dc_output_mw(net, i, table)
    if not np.isfinite(p_out):
        return out
    out.update(p_in_mw=p_in, p_out_mw=p_out, loss_mw=p_in - p_out,
               efficiency_percent=100.0 * p_out / p_in if p_in > 1e-12 and p_out >= 0 else None,
               loading_percent=100.0 * max(abs(p_in), abs(p_out)) / stage['rated_mw'] if stage['rated_mw'] > 0 else None)
    return out


def _electrisim_dc_dc_result(net, rec):
    out = {'name': rec['name'], 'id': rec['id'], 'mode': rec['mode'], 'control': rec.get('control', rec['mode']),
           'in_service': rec['in_service'],
           'rated_mw': rec['rated_mw'], 'p_in_mw': 0.0, 'p_out_mw': 0.0, 'loss_mw': 0.0,
           'efficiency_percent': None, 'loading_percent': 0.0, 'vm_in_pu': None, 'vm_out_pu': None}
    res_bus = getattr(net, 'res_bus_dc', None)
    for key, bus in (('vm_in_pu', rec['bus_in']), ('vm_out_pu', rec['bus_out'])):
        if res_bus is not None and bus in res_bus.index and np.isfinite(res_bus.at[bus, 'vm_pu']):
            out[key] = float(res_bus.at[bus, 'vm_pu'])
    if not rec['in_service'] or rec['input'] not in net.load_dc.index:
        return out
    p_in = float(net.load_dc.at[rec['input'], 'p_dc_mw'])
    p_out = _electrisim_dc_dc_output_mw(net, rec['input'])
    if not np.isfinite(p_out):
        return out
    out.update(p_in_mw=p_in, p_out_mw=p_out, loss_mw=p_in - p_out,
               efficiency_percent=100.0 * p_out / p_in if p_in > 1e-12 and p_out >= 0 else None,
               loading_percent=100.0 * max(abs(p_in), abs(p_out)) / rec['rated_mw'] if rec['rated_mw'] > 0 else None)
    if p_out < -1e-9 and not rec['bidirectional']:
        _electrisim_warn(net, f"DC/DC Converter '{rec['label']}': power flows from its output to its input "
                              f"({-p_out:.4g} MW), but it is not bidirectional.")
    if rec['rated_mw'] > 0 and out['loading_percent'] and out['loading_percent'] > 100.0:
        _electrisim_warn(net, f"DC/DC Converter '{rec['label']}' is loaded to {out['loading_percent']:.1f} % "
                              "of its rated power.")
    return out


# --- Microgrid sources and stores ----------------------------------------------------------
#
# A battery, supercapacitor, flywheel, SOFC system or PV array (der_electrisim) sits on a DC bus:
# - behind its DC/DC converter, alone on a bus that only converters' terminals reach (its port):
#   it holds that bus at its terminal voltage, at the current the converter draws - an auxiliary
#   source as a converter's output is, its voltage settled with the load flow;
# - directly on a network bus: a battery as its open-circuit voltage behind its resistance (a DC
#   source on a hidden bus, a cable of that resistance); a supercapacitor as a DC-link capacitor
#   (no current in a load flow); a flywheel at its set power, its machine converter's; a PV array
#   or SOFC at the power its curve gives at the bus's voltage.

def _electrisim_build_ders(net):
    pending = getattr(net, '_electrisim_pending_der', None) or []
    net.electrisim_ders = []
    if not pending:
        return
    convs = getattr(net, 'electrisim_dc_dc_converters', None) or []
    held = set()
    for el in pending:
        kind = der_electrisim.kind_of(el.get('typ'))
        label = el.get('userFriendlyName') or el.get('name')
        bus = _electrisim_dc_bus(net, el.get('bus'))
        if bus is None:
            _electrisim_warn(net, f"{kind} '{label}' is not connected to a DC bus, so it is left out.")
            continue
        try:
            obj = der_electrisim.build(el)
        except ValueError as e:
            _electrisim_warn(net, f"{kind} '{label}' is left out: {e}")
            continue
        on = obj.in_service
        vn = float(net.bus_dc.at[bus, 'vn_kv']) * 1e3
        rec = {'obj': obj, 'kind': kind, 'name': el.get('name'), 'label': label, 'id': el.get('id', ''),
               'bus': int(bus), 'vn': vn, 'coupling': 'direct', 'parts': {}, 'converters': []}
        terminals = [c for c in convs if bus in (c['bus_in'], c['bus_out'])]
        rec['converters'] = terminals
        port = bool(terminals) and not _electrisim_dc_bus_has_network(net, bus)
        if kind == 'Supercapacitor' and obj.direct:
            port = False
        if port and bus in held:
            _electrisim_warn(net, f"{kind} '{label}' is left out: another source already holds its converter's bus. "
                                  "Give each its own bus and converter.")
            continue
        if not port and not _electrisim_dc_bus_has_network(net, bus) and not terminals:
            _electrisim_warn(net, f"{kind} '{label}' is on a DC bus nothing else reaches, so it is left out.")
            continue
        if abs(obj.v_nominal() - vn) > 0.25 * vn and not (kind == 'Supercapacitor' and not obj.direct):
            _electrisim_warn(net, f"{kind} '{label}': its voltage ({obj.v_nominal():.4g} V) is far from its bus's "
                                  f"({vn:.4g} V).")
        if port and any(c['bus_out'] == bus for c in terminals):
            _electrisim_warn(net, f"{kind} '{label}' is left out: it is on its DC/DC converter's output. "
                                  "Draw the converter from the source's bus (its input) to the network (its output).")
            continue
        if port:
            held.add(int(bus))
            rec['coupling'] = 'converter'
            rated = max(obj.p_limits()[0], obj.p_limits()[1], 1e3) / 1e6
            vsc, aux_bus, aux_grid = _electrisim_aux_dc_source(net, bus, obj.v_terminal(0.0) / vn, rated,
                                                               f"{el.get('name')} terminal", on)
            rec['parts'] = {'vsc': vsc, 'aux_bus': aux_bus, 'aux_grid': aux_grid}
            _electrisim_der_set_converters(net, rec)
        elif kind == 'Supercapacitor':
            # Directly on its bus: a DC-link capacitor, which the DC fault and EMT studies model.
            if not hasattr(net, 'electrisim_dc_capacitors'):
                net.electrisim_dc_capacitors = []
            net.electrisim_dc_capacitors.append({
                'name': el.get('name'), 'id': el.get('id', ''), 'bus_dc': int(bus), 'c_mf': obj.c * 1e3,
                'esr_mohm': obj.esr * 1e3, 'esl_uh': obj.esl * 1e6, 'in_service': on, 'electrisim_der': True})
        elif kind == 'Battery':
            # Directly on its bus: its open-circuit voltage on a hidden bus, behind its resistance.
            # (pandapower holds a DC voltage reliably only through a VSC: an auxiliary one, as a converter's output.)
            inner = pp.create_bus_dc(net, vn_kv=vn / 1e3, name=f"{el.get('name')} cells", in_service=on)
            net.bus_dc.at[inner, 'electrisim_hidden'] = True
            rated = max(obj.p_limits()[0], obj.p_limits()[1], 1e3) / 1e6
            vsc, aux_bus, aux_grid = _electrisim_aux_dc_source(net, inner, obj.ocv() / vn, rated,
                                                               f"{el.get('name')} cells", on)
            line = pp.create_line_dc_from_parameters(net, from_bus_dc=inner, to_bus_dc=bus, length_km=1.0,
                                                     r_ohm_per_km=max(obj.r0 + obj.r1, 1e-6), max_i_ka=1e3,
                                                     name=f"{el.get('name')} resistance", in_service=on)
            net.line_dc.at[line, 'electrisim_hidden'] = True
            rec['parts'] = {'inner': int(inner), 'cells': vsc, 'aux_bus': aux_bus, 'aux_grid': aux_grid, 'line': int(line)}
        else:
            # A flywheel at its set power; a PV array or SOFC on its curve (settled with the load flow).
            p0 = _f_der_direct_power(obj, vn)
            ld = _electrisim_new_load_dc(net, bus, -p0 / 1e6, f"{el.get('name')} output", on,
                                         electrisim_aux=True, electrisim_dcdc_role='der')
            rec['parts'] = {'load': int(ld)}
        net.electrisim_ders.append(rec)
        if not hasattr(net, 'user_friendly_names'):
            net.user_friendly_names = {}
        net.user_friendly_names[el.get('name')] = label
        for part in ('cells', 'resistance', 'output', 'terminal', 'cells output', 'terminal output'):
            net.user_friendly_names[f"{el.get('name')} {part}"] = f'{label} ({part.split()[0]})'


def _electrisim_dc_bus_has_network(net, bus):
    """Whether something other than DC/DC converters' terminals reaches a DC bus: a cable, a VSC, a source."""
    for table, cols in (('line_dc', ('from_bus_dc', 'to_bus_dc')), ('source_dc', ('bus_dc',)), ('b2b_vsc', ('bus_dc_plus', 'bus_dc_minus'))):
        df = net[table] if table in net else None
        if df is not None and len(df) and df[list(cols)].isin([bus]).any(axis=None):
            return True
    if len(net.vsc):
        real = net.vsc if 'electrisim_aux' not in net.vsc.columns else net.vsc[net.vsc['electrisim_aux'] != True]
        if (real['bus_dc'] == bus).any():
            return True
    ld = net.load_dc
    if len(ld):
        real = ld if 'electrisim_dcdc_role' not in ld.columns else ld[ld['electrisim_dcdc_role'].isna()]
        if (real['bus_dc'] == bus).any():
            return True
    return False


def _f_der_direct_power(obj, v):
    """What a directly connected flywheel, PV array or SOFC delivers (W) at its bus voltage ``v`` (V)."""
    if obj.kind == 'Flywheel':
        return min(max(obj.p_set, -obj.p_limits()[1]), obj.p_limits()[0])
    if obj.kind == 'PV Array':
        return v * obj.i_array(v)
    if obj.kind == 'SOFC':
        return v * obj.i_at_voltage(v)[0]
    return 0.0


def _electrisim_der_set_converters(net, rec):
    """A converter in MPPT or follower mode delivers what its source gives: its PV array's maximum power, its SOFC's set power."""
    obj = rec['obj']
    for conv in rec['converters']:
        if conv['bus_in'] != rec['bus'] or conv['output_load'] is None:
            continue
        p_src = None
        if conv['control'] == 'mppt':
            p_src = obj.mpp()[2] if obj.kind == 'PV Array' else None
        elif conv['control'] == 'follower':
            p_src = obj.p_operating() if obj.kind == 'SOFC' else None
        if conv['control'] in ('mppt', 'follower') and p_src is None:
            _electrisim_warn(net, f"DC/DC Converter '{conv['label']}': {conv['control']} mode needs a "
                                  f"{'PV array' if conv['control'] == 'mppt' else 'SOFC system'} on its input; "
                                  "it delivers its set power.")
            continue
        if p_src is None:
            continue
        p_out = _electrisim_stage_output(p_src / 1e6, conv['eta'], conv['p_nl_mw'])
        conv['p_set_mw'] = p_out
        net.load_dc.at[conv['output_load'], 'p_dc_mw'] = -p_out


def _electrisim_stage_output(p_in, eta, p_nl):
    """What a stage delivers (MW) for ``p_in`` drawn at its input: _electrisim_dc_dc_input_power's inverse."""
    p = p_in - p_nl
    return p * eta if p >= 0 else p / eta


def _electrisim_settle_ders(net):
    """One round of settling the sources and stores after a load flow; the largest change (p.u. or MW)."""
    worst = 0.0
    res_bus = net.res_bus_dc
    for rec in getattr(net, 'electrisim_ders', None) or []:
        obj, parts, bus = rec['obj'], rec['parts'], rec['bus']
        if not obj.in_service or bus not in res_bus.index or not np.isfinite(res_bus.at[bus, 'vm_pu']):
            continue
        v = float(res_bus.at[bus, 'vm_pu']) * rec['vn']
        if rec['coupling'] == 'converter':
            vsc = parts['vsc']
            if vsc not in net.res_vsc.index:
                continue
            i = -float(net.res_vsc.at[vsc, 'p_dc_mw']) * 1e6 / max(v, 1e-6)
            if obj.kind == 'PV Array':
                # Its converter takes it to its maximum power point at most: at it in MPPT, where the
                # array's voltage would be tangent to the power drawn; past it, it cannot give more.
                if 'mpp' not in rec:
                    rec['mpp'] = obj.mpp()
                v_mp, i_mp, p_mp = rec['mpp']
                rec['beyond_mpp'] = i * v > p_mp * (1.0 + 1e-6)
                mppt = any(c['control'] == 'mppt' and c['bus_in'] == bus for c in rec['converters'])
                at_mpp = mppt or i >= i_mp * (1.0 - 1e-9)
                vm_new = (v_mp if at_mpp else max(obj.v_terminal(i), v_mp)) / rec['vn']
            else:
                vm_new = obj.v_terminal(i) / rec['vn']
            worst = max(worst, abs(vm_new - float(net.vsc.at[vsc, 'control_value_dc'])))
            net.vsc.at[vsc, 'control_value_dc'] = vm_new
        elif 'load' in parts and obj.kind in ('PV Array', 'SOFC'):
            p_new = -_f_der_direct_power(obj, v) / 1e6
            worst = max(worst, abs(p_new - float(net.load_dc.at[parts['load'], 'p_dc_mw'])))
            net.load_dc.at[parts['load'], 'p_dc_mw'] = p_new
    # Converters in droop: their voltage set point lowered with the power they deliver.
    for conv in getattr(net, 'electrisim_dc_dc_converters', None) or []:
        if conv.get('control') != 'droop' or conv['vsc'] is None or conv['vsc'] not in net.res_vsc.index:
            continue
        p_out = -float(net.res_vsc.at[conv['vsc'], 'p_dc_mw'])
        rated = conv['rated_mw'] if conv['rated_mw'] > 0 else 1.0
        vm_new = conv['vm_out_pu'] * (1.0 - conv['droop_percent'] / 100.0 * p_out / rated)
        worst = max(worst, abs(vm_new - float(net.vsc.at[conv['vsc'], 'control_value_dc'])))
        net.vsc.at[conv['vsc'], 'control_value_dc'] = vm_new
    return worst


def _electrisim_der_power(net, rec):
    """A source's or store's bus voltage (V) and the power it delivers into its bus (W), from the last load flow."""
    parts, bus = rec['parts'], rec['bus']
    v = float(net.res_bus_dc.at[bus, 'vm_pu']) * rec['vn']
    p = 0.0
    if rec['coupling'] == 'converter' and parts['vsc'] in net.res_vsc.index:
        p = -float(net.res_vsc.at[parts['vsc'], 'p_dc_mw']) * 1e6
    elif 'line' in parts and parts['line'] in net.res_line_dc.index:
        # From its cells through its resistance into its bus.
        p = -float(net.res_line_dc.at[parts['line'], 'p_to_mw']) * 1e6
    elif 'load' in parts and parts['load'] in net.load_dc.index:
        p = -float(net.load_dc.at[parts['load'], 'p_dc_mw']) * 1e6
    return v, p


def _electrisim_der_result(net, rec):
    """A source's or store's power, voltage, current and state; its warnings."""
    obj, parts, bus = rec['obj'], rec['parts'], rec['bus']
    out = {'name': rec['name'], 'id': rec['id'], 'label': rec['label'], 'kind': rec['kind'],
           'coupling': rec['coupling'], 'in_service': obj.in_service, 'p_mw': None, 'v_kv': None, 'i_ka': None}
    res_bus = getattr(net, 'res_bus_dc', None)
    if not obj.in_service or res_bus is None or bus not in res_bus.index or not np.isfinite(res_bus.at[bus, 'vm_pu']):
        return out
    v, p = _electrisim_der_power(net, rec)
    return _electrisim_der_report(net, rec, out, v, p)


def _electrisim_der_report(net, rec, out, v, p):
    """``out`` with its power ``p`` (W) at voltage ``v`` (V), its state, and its notes warned about."""
    obj = rec['obj']
    i = p / max(v, 1e-6)
    out.update(p_mw=p / 1e6 + 0.0, v_kv=v / 1e3, i_ka=i / 1e3 + 0.0)
    if rec['kind'] == 'Supercapacitor' and obj.direct:
        obj.v0 = v                                   # a capacitor on its bus: at its bus's voltage
    out.update({k: (float(x) if isinstance(x, (int, float, np.floating)) and x is not None else x)
                for k, x in obj.state(i, v).items()})
    if rec['kind'] == 'SOFC' and rec['coupling'] == 'direct' and obj.i_at_voltage(v)[1]:
        obj.notes.append(f"its bus ({v:.4g} V) is above its voltage at its minimum load: it runs at its minimum load")
    if rec.get('beyond_mpp'):
        obj.notes.append(f"its converter draws {p / 1e3:.4g} kW, more than its maximum power "
                         f"({rec['mpp'][2] / 1e3:.4g} kW) at its irradiance and temperature")
    for note in obj.notes:
        _electrisim_warn(net, f"{rec['kind']} '{rec['label']}': {note}.")
    obj.notes = []
    return out


# --- Power conversion systems (PCS) --------------------------------------------------------
#
# A PCS joins one source or store (der_electrisim) to an AC bus: the source wired straight to
# its DC side, or alone on a DC bus its DC side reaches (that bus is then the PCS's own, and
# not reported). Grid-following, it is a current source (pandapower sgen) at the power its
# source gives - a PV array's maximum power, an SOFC's set power, a battery's or flywheel's set
# power - and a Q by set point, power factor or Q(V). Grid-forming, it holds its bus's voltage
# (pandapower gen); on the grid it delivers its set power, islanded the grid-forming PCS of an
# island share its imbalance by their P-f droops - each takes S_rated / droop of it - one of
# them the island's reference (slack), its frequency f_n (1 - droop (P - P_set) / S_rated). A
# Q-V droop lowers its voltage set point with its Q. Its DC side is its source at the power
# it draws: the AC power over its efficiency, plus its no-load loss.
#
# In the short-circuit studies each is a current source at its current limit (k x I_rated),
# grid-forming or grid-following; an island no grid or machine feeds has its largest
# grid-forming PCS as a source giving that current at its own bus.

_PCS_SC_STUDIES = ('ShortCircuit', 'ArcFlash', 'ProtectionCoordination', 'PoiFault', 'FuseCharacteristic')


def _electrisim_build_pcs(net, Busbars, study):
    pending = getattr(net, '_electrisim_pending_pcs', None) or []
    net.electrisim_pcs = []
    if not pending:
        return
    ders = getattr(net, '_electrisim_pending_der', None) or []
    convs = getattr(net, 'electrisim_dc_dc_converters', None) or []
    sc_study = any(k in str(study) for k in _PCS_SC_STUDIES)
    for el in pending:
        name = el.get('name')
        label = el.get('userFriendlyName') or name
        bus = Busbars.get(el.get('bus')) if el.get('bus') is not None else None
        if bus is None:
            _electrisim_warn(net, f"PCS '{label}' needs an AC bus on its AC side, so it is left out.")
            continue
        # Its source: wired to its DC side, or alone on the DC bus its DC side reaches.
        src = next((d for d in ders if el.get('der') and d.get('name') == el.get('der')), None)
        bus_dc = _electrisim_dc_bus(net, el.get('bus_dc')) if el.get('bus_dc') else None
        if src is None and bus_dc is not None:
            on_bus = [d for d in ders if _electrisim_dc_bus(net, d.get('bus')) == bus_dc]
            alone = (len(on_bus) == 1 and not _electrisim_dc_bus_has_network(net, bus_dc)
                     and not any(bus_dc in (c['bus_in'], c['bus_out']) for c in convs))
            if not alone:
                _electrisim_warn(net, f"PCS '{label}' is left out: its DC side must be one source or store, alone on "
                                      "its DC bus. A DC network joins an AC bus through a VSC.")
                continue
            src = on_bus[0]
        if src is None:
            _electrisim_warn(net, f"PCS '{label}' has no battery, flywheel, SOFC system or PV array on its DC side, "
                                  "so it is left out.")
            continue
        ders.remove(src)
        kind = der_electrisim.kind_of(src.get('typ'))
        src_label = src.get('userFriendlyName') or src.get('name')
        if bus_dc is not None and bus_dc in net.bus_dc.index:
            net.bus_dc.drop(bus_dc, inplace=True)       # the PCS's own DC link: its source's terminals
        if kind == 'Supercapacitor':
            _electrisim_warn(net, f"PCS '{label}' is left out: a supercapacitor connects to a DC bus, "
                                  f"directly or through a DC/DC converter ('{src_label}').")
            continue
        try:
            obj = der_electrisim.build(src)
        except ValueError as e:
            _electrisim_warn(net, f"PCS '{label}' is left out: {e}")
            continue
        on = _electrisim_in_service(el) and obj.in_service
        s_rated = safe_float(el.get('s_rated_mva'), 1.0)
        if s_rated <= 0:
            s_rated = 1.0
        eta = safe_float(el.get('efficiency_percent'), 98.0) / 100.0
        if not 0 < eta <= 1:
            eta = 1.0
        control = 'grid_forming' if str(el.get('control') or '').strip().lower() == 'grid_forming' else 'grid_following'
        vn = float(net.bus.at[bus, 'vn_kv'])
        rec = {
            'name': name, 'id': el.get('id', ''), 'label': label, 'bus': int(bus), 'vn_kv': vn,
            'control': control, 'in_service': on, 's_rated': s_rated, 'eta': eta,
            'p_nl_mw': safe_float(el.get('no_load_loss_kw'), 0.0) / 1e3,
            'p_set_mw': safe_float(el.get('p_set_mw'), 0.0),
            'q_mode': str(el.get('q_mode') or 'q'), 'q_set_mvar': safe_float(el.get('q_set_mvar'), 0.0),
            'pf': safe_float(el.get('pf'), 1.0), 'qv_droop': safe_float(el.get('qv_droop_percent'), 5.0) / 100.0,
            'vm_set_pu': safe_float(el.get('vm_set_pu'), 1.0),
            'droop_pf': max(safe_float(el.get('droop_pf_percent'), 2.0), 1e-3) / 100.0,
            'droop_qv': max(safe_float(el.get('droop_qv_percent'), 5.0), 0.0) / 100.0,
            'k': safe_float(el.get('current_limit_pu'), 1.2),
            'source': {'obj': obj, 'kind': kind, 'name': src.get('name'), 'label': src_label, 'id': src.get('id', ''),
                       'coupling': 'pcs', 'parts': {}, 'bus': None},
            'table': None, 'index': None, 'island': None, 'reference': False, 'df_pu': 0.0, 'qv_hist': None,
        }
        vn_ac = safe_float(el.get('vn_ac_kv'), 0.0)
        if vn_ac > 0 and abs(vn_ac - vn) > 0.1 * vn:
            _electrisim_warn(net, f"PCS '{label}': its AC side is rated {vn_ac:g} kV, its bus is {vn:g} kV - "
                                  "draw its transformer between them.")
        p_ac, q = _electrisim_pcs_set_point(net, rec)
        rec['p_set_ac'] = p_ac
        i_rated_ka = s_rated / (math.sqrt(3.0) * vn)
        common = dict(name=name, id=el.get('id', ''), in_service=on, sn_mva=s_rated, controllable=False)
        if sc_study or control == 'grid_following':
            idx = pp.create_sgen(net, bus, p_mw=p_ac, q_mvar=q, k=rec['k'], rx=0.1, generator_type='current_source',
                                 current_source=True, type='PCS', **common)
            # Its short-circuit data, as the IEC (k, sn_mva) and ANSI (max_ik_ka) studies read them.
            for col, value in (('k', rec['k']), ('rx', 0.1), ('current_source', True),
                               ('generator_type', 'current_source'), ('max_ik_ka', rec['k'] * i_rated_ka)):
                net.sgen.at[idx, col] = value
            rec.update(table='sgen', index=int(idx))
        else:
            idx = pp.create_gen(net, bus, p_mw=p_ac, vm_pu=rec['vm_set_pu'], max_q_mvar=s_rated, min_q_mvar=-s_rated,
                                max_p_mw=s_rated, min_p_mw=-s_rated, **common)
            rec.update(table='gen', index=int(idx))
        net[rec['table']].at[rec['index'], 'electrisim_pcs'] = True
        net.electrisim_pcs.append(rec)
        if not hasattr(net, 'user_friendly_names'):
            net.user_friendly_names = {}
        net.user_friendly_names[name] = label
    if sc_study:
        _electrisim_pcs_sc_islands(net)
    else:
        _electrisim_pcs_islands(net)


def _electrisim_pcs_set_point(net, rec):
    """The AC power (MW) and Q (Mvar) a PCS is asked for, within its rating: its source's, or its set power."""
    src, obj = rec['source'], rec['source']['obj']
    if src['kind'] == 'PV Array':
        p_dc = obj.mpp()[2] / 1e6
        p_ac = _electrisim_stage_output(p_dc, rec['eta'], rec['p_nl_mw'])
    elif src['kind'] == 'SOFC':
        p_ac = _electrisim_stage_output(obj.p_operating() / 1e6, rec['eta'], rec['p_nl_mw'])
    else:
        p_ac = rec['p_set_mw']
    s = rec['s_rated']
    if abs(p_ac) > s:
        _electrisim_warn(net, f"PCS '{rec['label']}': {p_ac:.4g} MW is more than its rating ({s:g} MVA), "
                              f"so it delivers {math.copysign(s, p_ac):.4g} MW.")
        p_ac = math.copysign(s, p_ac)
    q_max = math.sqrt(max(s * s - p_ac * p_ac, 0.0))
    if rec['q_mode'] == 'pf':
        pf = min(max(abs(rec['pf']), 1e-3), 1.0)
        q = math.copysign(abs(p_ac) * math.tan(math.acos(pf)), rec['pf'])
    elif rec['q_mode'] == 'qv':
        q = 0.0
    else:
        q = rec['q_set_mvar']
    if abs(q) > q_max + 1e-12:
        _electrisim_warn(net, f"PCS '{rec['label']}': its Q ({q:.4g} Mvar) is held to {math.copysign(q_max, q):.4g} "
                              f"Mvar by its rating, its active power first.")
        q = math.copysign(q_max, q)
    rec['q_max'] = q_max
    return p_ac, q


def _electrisim_pcs_weight(rec):
    return rec['s_rated'] / rec['droop_pf']


def _electrisim_pcs_islands(net):
    """
    Each grid-forming PCS's island: with an external grid it delivers its set
    power; without one, the island's grid-forming PCS share its imbalance by
    droop, the one with the largest S_rated / droop its reference (slack).
    """
    gf = [r for r in net.electrisim_pcs if r['table'] == 'gen' and r['in_service']]
    if not gf:
        return
    graph = top.create_nxgraph(net, respect_switches=True)
    grid_buses = set(int(b) for b in net.ext_grid.loc[net.ext_grid['in_service'] == True, 'bus']) \
        if len(net.ext_grid) else set()
    slack_gens = set(int(b) for b in net.gen.loc[(net.gen['slack'] == True) & (net.gen['in_service'] == True), 'bus']) \
        if len(net.gen) and 'slack' in net.gen.columns else set()
    islands = {}
    for rec in gf:
        comp = frozenset(top.connected_component(graph, rec['bus']))
        if comp & (grid_buses | slack_gens):
            continue
        islands.setdefault(comp, []).append(rec)
    for n, (comp, recs) in enumerate(islands.items()):
        ref = max(recs, key=_electrisim_pcs_weight)
        for rec in recs:
            rec['island'] = n
        ref['reference'] = True
        net.gen.at[ref['index'], 'slack'] = True
        net.gen.at[ref['index'], 'slack_weight'] = 1.0


def _electrisim_pcs_sc_islands(net):
    """
    A short-circuit study needs a voltage source in each island: one no grid or
    machine feeds gets its largest grid-forming PCS as an external grid that
    gives k x I_rated at its bus - a converter at its current limit.
    """
    pcs = [r for r in net.electrisim_pcs if r['in_service']]
    if not pcs:
        return
    graph = top.create_nxgraph(net, respect_switches=True)
    sources = set(int(b) for b in net.ext_grid.loc[net.ext_grid['in_service'] == True, 'bus']) if len(net.ext_grid) else set()
    if len(net.gen):
        real = net.gen[(net.gen['in_service'] == True)]
        if 'electrisim_pcs' in real.columns:
            real = real[real['electrisim_pcs'] != True]
        sources |= set(int(b) for b in real['bus'])
    done = set()
    for rec in pcs:
        comp = frozenset(top.connected_component(graph, rec['bus']))
        if comp in done or comp & sources:
            continue
        done.add(comp)
        mates = [r for r in pcs if r['bus'] in comp]
        forming = [r for r in mates if r['control'] == 'grid_forming'] or mates
        ref = max(forming, key=lambda r: r['k'] * r['s_rated'])
        s_sc = ref['k'] * ref['s_rated']
        g = pp.create_ext_grid(net, ref['bus'], vm_pu=1.0, name=f"{ref['name']} (as a source)", s_sc_max_mva=s_sc,
                               s_sc_min_mva=s_sc, rx_max=0.1, rx_min=0.1, x0x_max=1.0, r0x0_max=0.1,
                               x0x_min=1.0, r0x0_min=0.1)
        net.ext_grid.at[g, 'id'] = ref['id']
        net.sgen.at[ref['index'], 'in_service'] = False
        ref['as_source'] = int(g)
        _electrisim_warn(net, f"PCS '{ref['label']}' forms an island no grid or machine feeds: in the short-circuit "
                              f"study it is a source giving its current limit ({ref['k']:g} x its rated current) at its "
                              "own bus, the other PCS current sources at theirs.")


def _electrisim_pcs_to_settle(net):
    return [r for r in getattr(net, 'electrisim_pcs', None) or []
            if r['in_service'] and r['table'] in ('gen', 'sgen')
            and (r['table'] == 'gen' and (r['island'] is not None or r['droop_qv'] > 0) or r['q_mode'] == 'qv')]


def _electrisim_secant(rec, key, x, g, x_min, x_max):
    """One secant step on g(x) = 0 for a PCS's set point, from the last; bounded."""
    hist = rec.get(key)
    rec[key] = (x, g)
    if hist is not None and abs(x - hist[0]) > 1e-12 and abs(g - hist[1]) > 1e-15:
        slope = (g - hist[1]) / (x - hist[0])
        x_new = x - g / slope if slope > 0 else x - 0.5 * g
    else:
        x_new = x - 0.5 * g
    step = max(min(x_new - x, 0.05), -0.05)
    return min(max(x + step, x_min), x_max)


def _electrisim_settle_pcs(net):
    """One round of settling the PCS after a load flow; the largest change (MW or p.u.)."""
    worst = 0.0
    recs = _electrisim_pcs_to_settle(net)
    if not recs or not hasattr(net, 'res_bus'):
        return 0.0
    # Islanded grid-forming PCS: their island's imbalance shared by droop.
    islands = {}
    for rec in recs:
        if rec['table'] == 'gen' and rec['island'] is not None and rec['index'] in net.res_gen.index:
            islands.setdefault(rec['island'], []).append(rec)
    for group in islands.values():
        dp = sum(float(net.res_gen.at[r['index'], 'p_mw']) - r['p_set_ac'] for r in group)
        w_sum = sum(_electrisim_pcs_weight(r) for r in group)
        for rec in group:
            share = _electrisim_pcs_weight(rec) / w_sum * dp
            rec['df_pu'] = -rec['droop_pf'] * share / rec['s_rated']
            if rec['reference']:
                continue
            target = rec['p_set_ac'] + share
            worst = max(worst, abs(target - float(net.gen.at[rec['index'], 'p_mw'])))
            net.gen.at[rec['index'], 'p_mw'] = target
    # Q-V droop of the grid-forming PCS: its voltage set point lowered by its Q.
    for rec in recs:
        if rec['table'] == 'gen' and rec['droop_qv'] > 0 and rec['index'] in net.res_gen.index:
            q = float(net.res_gen.at[rec['index'], 'q_mvar'])
            v_set = float(net.gen.at[rec['index'], 'vm_pu'])
            g = v_set - (rec['vm_set_pu'] - rec['droop_qv'] * q / rec['s_rated'])
            worst = max(worst, abs(g))
            if abs(g) > 1e-9:
                net.gen.at[rec['index'], 'vm_pu'] = _electrisim_secant(rec, 'qv_hist', v_set, g, 0.8, 1.2)
        elif rec['table'] == 'sgen' and rec['q_mode'] == 'qv' and rec['index'] in net.res_sgen.index:
            # Grid-following Q(V): Q = -(V - V_set) / droop x S_rated, within its capability.
            v = float(net.res_bus.at[rec['bus'], 'vm_pu'])
            q_now = float(net.sgen.at[rec['index'], 'q_mvar'])
            target = -(v - rec['vm_set_pu']) / max(rec['qv_droop'], 1e-3) * rec['s_rated']
            target = max(min(target, rec['q_max']), -rec['q_max'])
            new = q_now + 0.5 * (target - q_now)
            worst = max(worst, abs(target - q_now))
            net.sgen.at[rec['index'], 'q_mvar'] = new
    return worst


def _electrisim_der_dc_point(obj, p):
    """
    A source's or store's terminal voltage (V) delivering power ``p`` (W) into
    its PCS: (voltage, note or None).
    """
    if obj.kind == 'PV Array':
        v_mp, _, p_mp = obj.mpp()
        if p >= p_mp * (1.0 - 1e-9):
            note = (f"its PCS draws {p / 1e3:.4g} kW, more than its maximum power ({p_mp / 1e3:.4g} kW)"
                    if p > p_mp * (1.0 + 1e-6) else None)
            return v_mp, note
        lo, hi = v_mp, obj.n_s * obj.v_oc_t
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if mid * obj.i_array(mid) > p:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi), None
    if obj.kind == 'SOFC':
        i_min = max(obj.p_min_frac * obj.p_rated, 1e-3 * obj.p_rated) / obj.v_rated
        if p <= i_min * obj.v_terminal(i_min):
            return obj.v_terminal(i_min), (f"its PCS draws {p / 1e3:.4g} kW, below its minimum load"
                                           if p < i_min * obj.v_terminal(i_min) * (1 - 1e-6) else None)
        lo, hi = i_min, 3.0 * obj.i_rated
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if mid * obj.v_terminal(mid) < p:
                lo = mid
            else:
                hi = mid
        i = 0.5 * (lo + hi)
        note = (f"its PCS draws {p / 1e3:.4g} kW, more than its rating less its auxiliary load"
                if p > obj.p_limits()[0] * (1 + 1e-6) else None)
        return obj.v_terminal(i), note
    v = obj.v_nominal()
    for _ in range(100):
        v_new = obj.v_terminal(p / max(v, 1e-6))
        if abs(v_new - v) < 1e-9 * max(abs(v), 1.0):
            v = v_new
            break
        v = v_new
    note = None
    if obj.kind == 'Flywheel' and not -obj.p_limits()[1] - 1e-6 <= p <= obj.p_limits()[0] + 1e-6:
        note = f"its PCS asks {p / 1e3:.4g} kW, beyond its power limit at its speed ({obj.p_limits()[0] / 1e3:.4g} kW)"
    return v, note


def _electrisim_pcs_result(net, rec):
    """A PCS's AC and DC power, loading, losses and frequency; its source's report, under '_source'."""
    src = rec['source']
    out = {'name': rec['name'], 'id': rec['id'], 'label': rec['label'], 'control': rec['control'],
           'source': src['label'], 'source_kind': src['kind'], 'in_service': rec['in_service'],
           'islanded': rec['island'] is not None, 'p_mw': None, 'q_mvar': None, 's_mva': None,
           'loading_percent': None, 'vm_pu': None, 'p_dc_mw': None, 'loss_mw': None, 'frequency_hz': None}
    res = getattr(net, f"res_{rec['table']}", None)
    if not rec['in_service'] or res is None or rec['index'] not in res.index:
        return out
    p, q = float(res.at[rec['index'], 'p_mw']), float(res.at[rec['index'], 'q_mvar'])
    if not (np.isfinite(p) and np.isfinite(q)):
        return out
    s = math.hypot(p, q)
    p_dc = _electrisim_dc_dc_input_power(p, rec['eta'], rec['p_nl_mw'])
    f_n = float(getattr(net, 'f_hz', 50.0) or 50.0)
    out.update(p_mw=p + 0.0, q_mvar=q + 0.0, s_mva=s, loading_percent=100.0 * s / rec['s_rated'],
               vm_pu=float(net.res_bus.at[rec['bus'], 'vm_pu']) if rec['bus'] in net.res_bus.index else None,
               p_dc_mw=p_dc, loss_mw=p_dc - p, frequency_hz=f_n * (1.0 + rec['df_pu']))
    if s > rec['s_rated'] * 1.0001:
        _electrisim_warn(net, f"PCS '{rec['label']}' is loaded to {out['loading_percent']:.1f} % of its rating.")
    v, note = _electrisim_der_dc_point(src['obj'], p_dc * 1e6)
    if note:
        src['obj'].notes.append(note)
    report = {'name': src['name'], 'id': src['id'], 'label': src['label'], 'kind': src['kind'], 'coupling': 'pcs',
              'pcs': rec['label'], 'in_service': rec['in_service'], 'p_mw': None, 'v_kv': None, 'i_ka': None}
    out['_source'] = _electrisim_der_report(net, src, report, v, p_dc * 1e6)
    out['v_dc_v'] = v
    return out


# --- Solid-state transformers -------------------------------------------------------------
#
# One element, three ports: MV AC, LV DC, and an optional LV AC. It is built
# as three stages on the DC/DC converter's parts, each stage's input drawing
# its output divided by its efficiency plus its no-load loss:
#   rectifier   the power the MV bus supplies, with a set Q, into an internal
#               DC link held at the voltage the user sets;
#   DC/DC       from the link to the LV DC port, holding its voltage;
#   inverter    from the LV DC port to the LV AC port: grid-following, a set
#               P and Q into an LV AC network another source holds, or
#               grid-forming, holding an islanded LV AC network's voltage.
# The rectifier is the P and Q it draws at the MV bus rather than a lossy VSC,
# so each stage's efficiency holds exactly, as the DC/DC converter's does.

def _electrisim_build_ssts(net, Busbars):
    pending = getattr(net, '_electrisim_pending_sst', None) or []
    net.electrisim_ssts = []
    for el in pending:
        name = el.get('name')
        label = el.get('userFriendlyName') or name
        b_mv, b_lvac = Busbars.get(el.get('bus_mv')), Busbars.get(el.get('bus_lv_ac'))
        b_lvdc = _electrisim_dc_bus(net, el.get('bus_lv_dc'))
        if b_mv is not None and b_lvac is not None and float(net.bus.at[b_lvac, 'vn_kv']) > float(net.bus.at[b_mv, 'vn_kv']):
            b_mv, b_lvac = b_lvac, b_mv   # the MV port is the higher voltage, whichever pin it is on
        if b_mv is None or b_lvdc is None:
            _electrisim_warn(net, f"Solid-State Transformer '{label}' needs an AC bus at its MV port and a DC bus at "
                                  "its LV DC port, so it is left out.")
            continue
        on = _electrisim_in_service(el)

        def stage_params(prefix, eta_default, rated_default):
            eta = safe_float(el.get(f'{prefix}_efficiency_percent'), eta_default) / 100.0
            if not 0 < eta <= 1:
                _electrisim_warn(net, f"Solid-State Transformer '{label}': a {prefix} efficiency of {100 * eta:g} % "
                                      "is taken as 100 %.")
                eta = 1.0
            return eta, safe_float(el.get(f'{prefix}_no_load_kw'), 0.0) / 1e3, safe_float(el.get(f'{prefix}_rated_mw'), rated_default)

        rec = {'name': name, 'id': el.get('id', ''), 'label': label, 'in_service': on, 'bus_mv': int(b_mv),
               'bus_lvdc': int(b_lvdc), 'bus_lvac': int(b_lvac) if b_lvac is not None else None,
               'link_kv': safe_float(el.get('link_kv'), 0.0), 'stages': [], 'aux': [], 'inverter_mode': None,
               # For the EMT study: its stages' model, switching frequencies and current limit.
               'emt': {'model': 'switching' if el.get('emt_model') == 'switching' else 'average',
                       'switching_khz': safe_float(el.get('switching_khz'), 5.0),
                       'dcdc_switching_khz': safe_float(el.get('dcdc_switching_khz'), 20.0),
                       'current_limit_pu': safe_float(el.get('current_limit_pu'), 1.2)}}
        if rec['link_kv'] <= 0:
            rec['link_kv'] = float(net.bus.at[b_mv, 'vn_kv']) * math.sqrt(2) * 1.1
        for key, bus, vn in (('vn_mv_kv', b_mv, float(net.bus.at[b_mv, 'vn_kv'])),
                             ('vn_lv_dc_kv', b_lvdc, float(net.bus_dc.at[b_lvdc, 'vn_kv'])),
                             ('vn_lv_ac_kv', b_lvac, float(net.bus.at[b_lvac, 'vn_kv']) if b_lvac is not None else None)):
            rated_kv = safe_float(el.get(key), 0.0)
            if vn and rated_kv > 0 and abs(rated_kv - vn) > 0.1 * vn:
                port = {'vn_mv_kv': 'MV', 'vn_lv_dc_kv': 'LV DC', 'vn_lv_ac_kv': 'LV AC'}[key]
                _electrisim_warn(net, f"Solid-State Transformer '{label}': its {port} port is rated {rated_kv:g} kV, "
                                      f"its bus is {vn:g} kV.")

        # The internal DC link, held by the rectifier.
        link = pp.create_bus_dc(net, vn_kv=rec['link_kv'], name=f'{name} DC link', in_service=on)
        net.bus_dc.at[link, 'electrisim_aux'] = True
        r_eta, r_nl, r_rated = stage_params('rect', 98.5, 1.0)
        r_vsc, r_bus, r_grid = _electrisim_aux_dc_source(net, link, 1.0, r_rated, f'{name} rectifier', on)
        mv_load = pp.create_load(net, b_mv, p_mw=r_nl, q_mvar=safe_float(el.get('q_mv_mvar'), 0.0), name=name,
                                 in_service=on)
        for key, value in _electrisim_stage_columns(name, r_eta, r_nl, ('vsc', r_vsc)).items():
            net.load.at[mv_load, key] = value
        if 'id' in net.load.columns:
            net.load.at[mv_load, 'id'] = el.get('id', '')
        rec['stages'].append({'stage': 'rectifier', 'input': ('load', int(mv_load)), 'rated_mw': r_rated,
                              'output': ('vsc', int(r_vsc)), 'link': int(link)})

        # The DC/DC stage, holding the LV DC port.
        d_eta, d_nl, d_rated = stage_params('dcdc', 98.0, 1.0)
        d_vsc, d_bus, d_grid = _electrisim_aux_dc_source(net, b_lvdc, safe_float(el.get('vm_lv_dc_pu'), 1.0),
                                                         d_rated, f'{name} DC/DC', on)
        d_in = _electrisim_new_load_dc(net, link, d_nl, f'{name} DC/DC input', on,
                                       **_electrisim_stage_columns(name, d_eta, d_nl, ('vsc', d_vsc)))
        rec['stages'].append({'stage': 'dcdc', 'input': ('load_dc', int(d_in)), 'rated_mw': d_rated,
                              'output': ('vsc', int(d_vsc))})
        rec['aux'] = [('vsc', r_vsc), ('ext_grid', r_grid), ('bus', r_bus), ('vsc', d_vsc), ('ext_grid', d_grid),
                      ('bus', d_bus), ('load_dc', int(d_in)), ('bus_dc', int(link))]

        # The inverter, when the LV AC port is connected.
        if b_lvac is not None:
            i_eta, i_nl, i_rated = stage_params('inv', 97.5, 0.5)
            mode = 'grid_forming' if str(el.get('inverter_mode') or '').strip().lower() == 'grid_forming' else 'grid_following'
            rec['inverter_mode'] = mode
            if mode == 'grid_following':
                out = pp.create_sgen(net, b_lvac, p_mw=safe_float(el.get('p_ac_mw'), 0.0),
                                     q_mvar=safe_float(el.get('q_ac_mvar'), 0.0), name=f'{name} inverter',
                                     in_service=on, k=1.2, rx=0.1)
                net.sgen.at[out, 'electrisim_aux'] = True
                output = ('sgen', int(out))
            else:
                # Its short-circuit power: an inverter's current limit, some 1.2 times its rating.
                s_sc = max(1.2 * i_rated, 0.01)
                out = pp.create_ext_grid(net, b_lvac, vm_pu=safe_float(el.get('vm_lv_ac_pu'), 1.0), name=f'{name} inverter',
                                         in_service=on, s_sc_max_mva=s_sc, s_sc_min_mva=s_sc, rx_max=0.1, rx_min=0.1)
                net.ext_grid.at[out, 'electrisim_aux'] = True
                output = ('ext_grid', int(out))
                _electrisim_sst_check_island(net, rec, int(out))
            p0 = _electrisim_dc_dc_input_power(safe_float(el.get('p_ac_mw'), 0.0), i_eta, i_nl) if mode == 'grid_following' else i_nl
            i_in = _electrisim_new_load_dc(net, b_lvdc, p0, f'{name} inverter input', on,
                                           **_electrisim_stage_columns(name, i_eta, i_nl, output))
            rec['stages'].append({'stage': 'inverter', 'input': ('load_dc', int(i_in)), 'rated_mw': i_rated,
                                  'output': output})
            rec['aux'].append(('load_dc', int(i_in)))
        net.electrisim_ssts.append(rec)
        if not hasattr(net, 'user_friendly_names'):
            net.user_friendly_names = {}
        net.user_friendly_names[name] = label
        for part in ('inverter', 'inverter input', 'DC/DC input', 'DC link'):
            net.user_friendly_names[f'{name} {part}'] = f'{label} {part}'


def _electrisim_sst_check_island(net, rec, grid):
    """A grid-forming inverter holds an LV AC network nothing else holds."""
    try:
        mg = top.create_nxgraph(net, respect_switches=True)
        island = set(top.connected_component(mg, rec['bus_lvac']))
    except Exception:   # noqa: BLE001 - the check is advisory
        return
    others = [i for i in net.ext_grid.index if i != grid and bool(net.ext_grid.at[i, 'in_service'])
              and int(net.ext_grid.at[i, 'bus']) in island]
    others += [('gen', i) for i in net.gen.index if bool(net.gen.at[i, 'in_service']) and int(net.gen.at[i, 'bus']) in island]
    if others or rec['bus_mv'] in island:
        _electrisim_warn(net, f"Solid-State Transformer '{rec['label']}': its grid-forming inverter holds an LV AC "
                              "network another source also holds; a grid-following inverter suits a network that "
                              "is not an island.")


def _electrisim_sst_result(net, rec):
    out = {'name': rec['name'], 'id': rec['id'], 'in_service': rec['in_service'], 'inverter_mode': rec['inverter_mode'],
           'link_kv': rec['link_kv'], 'stages': [_electrisim_stage_result(net, st) for st in rec['stages']],
           'p_mv_mw': 0.0, 'q_mv_mvar': 0.0, 'vm_mv_pu': None, 'vm_lv_dc_pu': None, 'vm_lv_ac_pu': None,
           'p_lv_ac_mw': None, 'q_lv_ac_mvar': None}
    table, i = rec['stages'][0]['input']
    if rec['in_service'] and i in net.load.index:
        out['p_mv_mw'] = float(net.load.at[i, 'p_mw'])
        out['q_mv_mvar'] = float(net.load.at[i, 'q_mvar'])
    for key, res, bus in (('vm_mv_pu', 'res_bus', rec['bus_mv']), ('vm_lv_dc_pu', 'res_bus_dc', rec['bus_lvdc']),
                          ('vm_lv_ac_pu', 'res_bus', rec['bus_lvac'])):
        df = getattr(net, res, None)
        if bus is not None and df is not None and bus in df.index and np.isfinite(df.at[bus, 'vm_pu']):
            out[key] = float(df.at[bus, 'vm_pu'])
    inverter = next((st for st in rec['stages'] if st['stage'] == 'inverter'), None)
    if inverter and rec['in_service']:
        table, idx = inverter['output']
        if table == 'sgen':
            out['p_lv_ac_mw'], out['q_lv_ac_mvar'] = float(net.sgen.at[idx, 'p_mw']), float(net.sgen.at[idx, 'q_mvar'])
        elif idx in getattr(net, 'res_ext_grid', pd.DataFrame()).index:
            out['p_lv_ac_mw'] = float(net.res_ext_grid.at[idx, 'p_mw'])
            out['q_lv_ac_mvar'] = float(net.res_ext_grid.at[idx, 'q_mvar'])
    for st in out['stages']:
        if st['loading_percent'] and st['loading_percent'] > 100.0:
            _electrisim_warn(net, f"Solid-State Transformer '{rec['label']}': its {st['stage']} stage is loaded to "
                                  f"{st['loading_percent']:.1f} % of its rated power.")
    return out


# --- DC circuit breakers ----------------------------------------------------------------
#
# pandapower has no DC switch. A DC breaker sits between a DC bus and what it
# switches: a DC cable, a VSC's DC terminal, a DC load or source - which is
# taken out of service while the breaker is open - or a second DC bus, joined
# by a near-zero-resistance DC line (a coupler) in service while it is closed.
# Breakers are applied once every element exists, as the payload may list a
# breaker before what it switches.

_DC_BREAKER_TARGETS = ('line_dc', 'vsc', 'b2b_vsc', 'load_dc', 'source_dc')
_DC_COUPLER_R_OHM = 1e-6


def _electrisim_flag(value, default=True):
    if value is None or value == '':
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ('false', '0', 'no', 'off', 'open')


def _electrisim_apply_dc_breakers(net):
    pending = getattr(net, '_electrisim_pending_dc_breakers', None) or []
    net.electrisim_dc_breakers = []
    for el in pending:
        label = el.get('userFriendlyName') or el.get('name')
        bus = _electrisim_dc_bus(net, el.get('bus'))
        if bus is None:
            _electrisim_warn(net, f"DC Breaker '{label}' is not connected to a DC bus, so it is left out.")
            continue
        closed = _electrisim_flag(el.get('closed'), True)
        rated_kv = safe_float(el.get('rated_voltage_kv'), 0.0)
        rec = {
            'name': el.get('name'), 'id': el.get('id', ''), 'label': label, 'closed': closed, 'bus_dc': int(bus),
            'et': el.get('et'), 'target': None, 'breaker_type': el.get('breaker_type') or 'solid_state',
            'rated_voltage_kv': rated_kv,
            'rated_current_ka': safe_float(el.get('rated_current_ka'), 0.0),
            'breaking_capacity_ka': safe_float(el.get('breaking_capacity_ka'), 0.0),
            'opening_time_ms': safe_float(el.get('opening_time_ms'), 0.0),
            'limiting_inductance_mh': safe_float(el.get('limiting_inductance_mh'), 0.0),
            'arrester_clamp_kv': safe_float(el.get('arrester_clamp_kv'), 0.0),
            'arrester_energy_kj': safe_float(el.get('arrester_energy_kj'), 0.0),
            # The EMT study trips it above this (0: twice its rated current).
            'trip_current_ka': safe_float(el.get('trip_current_ka'), 0.0),
        }
        vn = float(net.bus_dc.at[bus, 'vn_kv'])
        if 0 < rated_kv < vn:
            _electrisim_warn(net, f"DC Breaker '{label}' is rated {rated_kv:g} kV, below its bus's {vn:g} kV.")
        et, element = el.get('et'), el.get('element')
        if et == 'bus_dc':
            other = _electrisim_dc_bus(net, element)
            if other is None:
                _electrisim_warn(net, f"DC Breaker '{label}' joins a DC bus to something that is not a DC bus, so it is left out.")
                continue
            idx = pp.create_line_dc_from_parameters(
                net, from_bus_dc=bus, to_bus_dc=other, length_km=1.0, r_ohm_per_km=_DC_COUPLER_R_OHM,
                max_i_ka=rec['rated_current_ka'] or 10.0, name=el.get('name'), in_service=closed)
            net.line_dc.at[idx, 'electrisim_dc_breaker'] = True
            if 'id' in net.line_dc.columns:
                net.line_dc.at[idx, 'id'] = el.get('id', '')
            rec['target'] = ('line_dc', int(idx))
        elif et == 'dc_dc_converter':
            conv = next((c for c in getattr(net, 'electrisim_dc_dc_converters', None) or [] if c['name'] == element), None)
            if conv is None:
                _electrisim_warn(net, f"DC Breaker '{label}': the DC/DC converter it switches was not built, so it is left out.")
                continue
            if int(bus) == conv['bus_in']:
                rec['target'] = ('load_dc', conv['input'])
            elif int(bus) == conv['bus_out']:
                rec['target'] = ('vsc', conv['vsc']) if conv['vsc'] is not None else ('load_dc', conv['output_load'])
            else:
                _electrisim_warn(net, f"DC Breaker '{label}' is not on either of its DC/DC converter's buses, so it is left out.")
                continue
            if not closed:
                _electrisim_set_dc_dc_in_service(net, conv, False)
        elif et in _DC_BREAKER_TARGETS:
            table = net[et]
            hit = table.index[table['name'] == element] if len(table) else []
            if not len(hit):
                _electrisim_warn(net, f"DC Breaker '{label}': what it switches was not built, so it is left out.")
                continue
            idx = int(hit[0])
            if not closed:
                table.at[idx, 'in_service'] = False
            rec['target'] = (et, idx)
        else:
            _electrisim_warn(net, f"DC Breaker '{label}' is left out: it goes between a DC bus and a DC cable, "
                                  "a VSC, a DC load or source, or a second DC bus.")
            continue
        net.electrisim_dc_breakers.append(rec)
        if not hasattr(net, 'user_friendly_names'):
            net.user_friendly_names = {}
        net.user_friendly_names[el.get('name')] = label


def _electrisim_dc_breaker_current_ka(net, rec):
    """The current through a DC breaker (kA), from what it switches; 0 while it is open."""
    if not rec['closed'] or rec['target'] is None:
        return 0.0
    table, idx = rec['target']
    bus = rec['bus_dc']
    res_bus = getattr(net, 'res_bus_dc', None)
    vm = float(res_bus.at[bus, 'vm_pu']) if res_bus is not None and bus in res_bus.index else np.nan
    v_kv = vm * float(net.bus_dc.at[bus, 'vn_kv'])
    if table == 'line_dc':
        res = net.res_line_dc
        if idx not in res.index:
            return np.nan
        if 'electrisim_dc_breaker' in net.line_dc.columns and net.line_dc.at[idx, 'electrisim_dc_breaker'] == True:
            return abs(float(res.at[idx, 'i_ka']))
        side = 'from' if int(net.line_dc.at[idx, 'from_bus_dc']) == bus else 'to'
        return abs(float(res.at[idx, f'i_{side}_ka']))
    if table == 'vsc':
        p = net.res_vsc.at[idx, 'p_dc_mw'] if idx in net.res_vsc.index else np.nan
    elif table == 'b2b_vsc':
        row = net.res_b2b_vsc.loc[idx] if idx in net.res_b2b_vsc.index else None
        p = np.nan if row is None else (row['p_dc_mw_p'] if int(net.b2b_vsc.at[idx, 'bus_dc_plus']) == bus else row['p_dc_mw_m'])
    elif table in ('load_dc', 'source_dc'):
        res = net[f'res_{table}']
        p = res.at[idx, 'p_dc_mw'] if idx in res.index else np.nan
    else:
        return np.nan
    return abs(float(p)) / v_kv if v_kv and np.isfinite(v_kv) else np.nan


def _electrisim_dc_bus(net, name):
    """Index of the DC bus drawn as ``name``, or None: an AC bus is not a DC bus."""
    return (getattr(net, 'dc_buses', None) or {}).get(name)


def _electrisim_in_service(el):
    value = el.get('in_service', True)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ('false', '0', 'no', 'off')


def _electrisim_drop_uncoupled_dc(net, quiet=False):
    """
    Set aside the DC buses no converter ties to the AC network, with what is on them.

    pandapower solves a DC network only through a VSC: one held by a DC source
    alone does not converge, and an uncoupled DC island upsets the AC
    Jacobian too (bess_preliminary_electrisim._strip_dc_for_ac_lf). The BESS
    plant builder draws battery racks this way, for the drawing only.
    """
    bus_dc = getattr(net, 'bus_dc', None)
    if bus_dc is None or not len(bus_dc):
        return
    parent = {int(b): int(b) for b in bus_dc.index}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, row in net.line_dc.iterrows():
        if bool(row['in_service']):
            parent[find(int(row['from_bus_dc']))] = find(int(row['to_bus_dc']))
    coupled = set()
    for _, row in net.vsc.iterrows():
        if bool(row['in_service']):
            coupled.add(find(int(row['bus_dc'])))
    for _, row in net.b2b_vsc.iterrows():
        if bool(row['in_service']):
            coupled.update({find(int(row['bus_dc_plus'])), find(int(row['bus_dc_minus']))})
    drop = {int(b) for b in bus_dc.index if find(int(b)) not in coupled}
    if not drop:
        return
    names = getattr(net, 'user_friendly_names', {}) or {}
    labels = sorted(str(names.get(bus_dc.at[b, 'name'], bus_dc.at[b, 'name'])) for b in drop)
    for table, cols in (('line_dc', ('from_bus_dc', 'to_bus_dc')), ('load_dc', ('bus_dc',)),
                        ('source_dc', ('bus_dc',)), ('vsc', ('bus_dc',)),
                        ('b2b_vsc', ('bus_dc_plus', 'bus_dc_minus'))):
        df = net[table]
        if len(df):
            hit = df[list(cols)].isin(drop).any(axis=1)
            df.drop(df.index[hit], inplace=True)
    bus_dc.drop(sorted(drop), inplace=True)
    if getattr(net, 'electrisim_dc_capacitors', None):
        net.electrisim_dc_capacitors = [c for c in net.electrisim_dc_capacitors if c['bus_dc'] not in drop]
    if getattr(net, 'electrisim_ders', None):
        net.electrisim_ders = [r for r in net.electrisim_ders if r['bus'] not in drop]
    if getattr(net, 'electrisim_dc_breakers', None):
        # A breaker stays while its own bus does; what it switched may be gone.
        net.electrisim_dc_breakers = [b for b in net.electrisim_dc_breakers if b['bus_dc'] not in drop]
        for b in net.electrisim_dc_breakers:
            if b['target'] is not None and b['target'][1] not in net[b['target'][0]].index:
                b['target'] = None
    if quiet:
        return
    _electrisim_warn(net, f"DC bus{'es' if len(labels) > 1 else ''} {', '.join(labels)} "
                          f"{'are' if len(labels) > 1 else 'is'} not connected to the AC network through a VSC, "
                          f"so {'they and what is on them are' if len(labels) > 1 else 'it and what is on it are'} left out: "
                          "pandapower solves a DC network only "
                          "through a converter.")


def _electrisim_set_aside_dc_network(net, study):
    """
    Leave a DC network out of a study that cannot model it, and say so.

    pandapower's optimal power flow has no VSC or DC network model: with one it
    failed (with init='pf', a divide by zero on the DC branches) or, started
    flat, returned the AC result as if the DC loads were not there.
    """
    bus_dc = getattr(net, 'bus_dc', None)
    if bus_dc is None or not len(bus_dc):
        return
    n_bus, n_load = len(bus_dc), len(net.load_dc)
    p_load = float(net.load_dc['p_dc_mw'].sum()) if n_load else 0.0
    for table in ('line_dc', 'load_dc', 'source_dc', 'vsc', 'b2b_vsc', 'bus_dc'):
        df = net[table]
        if len(df):
            df.drop(df.index, inplace=True)
    # The converters' auxiliary AC grids go with them; a grid-forming
    # inverter, on a real bus, stays as the source of its LV AC network.
    if 'electrisim_aux' in net.bus.columns:
        aux_buses = net.bus.index[net.bus['electrisim_aux'] == True]
        net.ext_grid.drop(net.ext_grid.index[net.ext_grid['bus'].isin(aux_buses)], inplace=True)
        net.bus.drop(aux_buses, inplace=True)
        # pandapower's OPF takes the external grids as numbered 0, 1, ...: renumber
        # them, and the cost rows that name them.
        renumber = {old: new for new, old in enumerate(net.ext_grid.index)}
        if any(old != new for old, new in renumber.items()):
            net.ext_grid.index = [renumber[i] for i in net.ext_grid.index]
            for cost in ('poly_cost', 'pwl_cost'):
                df = net[cost] if cost in net else None
                if df is not None and len(df):
                    hit = df['et'] == 'ext_grid'
                    df.loc[hit, 'element'] = df.loc[hit, 'element'].map(lambda e: renumber.get(int(e), e))
    if getattr(net, 'electrisim_ssts', None):
        net.electrisim_ssts = []
    if getattr(net, 'electrisim_dc_dc_converters', None):
        net.electrisim_dc_dc_converters = []
    for records in ('electrisim_dc_breakers', 'electrisim_dc_capacitors', 'electrisim_ders'):
        if getattr(net, records, None):
            setattr(net, records, [])
    _electrisim_warn(net, f"{study} leaves the DC network out ({n_bus} DC bus{'es' if n_bus != 1 else ''}"
                          + (f", {n_load} DC load{'s' if n_load != 1 else ''} of {p_load:.3g} MW" if n_load else '')
                          + "): pandapower's optimal power flow does not model VSCs or DC networks, "
                          "so this result does not include them.")


def _electrisim_row_id(df, index):
    return df.at[index, 'id'] if 'id' in df.columns else str(index)


def _electrisim_source_dc_currents_ka(net):
    """
    Each DC source's current into its bus (kA), by Kirchhoff's current law
    there: what its loads draw and its cables carry away, less what its VSCs
    inject, shared among the bus's sources. pandapower 3.3's res_source_dc
    does not give it: a source holding a bus with a 0.15 MW load reports 0 MW.
    """
    out = {}
    res_bus = getattr(net, 'res_bus_dc', None)
    if res_bus is None or 'source_dc' not in net or not len(net.source_dc):
        return out
    by_bus = {}
    for si in net.source_dc.index:
        bus = int(net.source_dc.at[si, 'bus_dc'])
        if bool(net.source_dc.at[si, 'in_service']) and bus in res_bus.index and np.isfinite(res_bus.at[bus, 'vm_pu']):
            by_bus.setdefault(bus, []).append(si)
    for bus, sources in by_bus.items():
        v_kv = float(res_bus.at[bus, 'vm_pu']) * float(net.bus_dc.at[bus, 'vn_kv'])
        if v_kv <= 0:
            continue
        i_out = 0.0
        ld = net.load_dc
        for li in ld.index[(ld['bus_dc'] == bus) & ld['in_service'].astype(bool)]:
            i_out += float(ld.at[li, 'p_dc_mw']) / v_kv
        res_line = getattr(net, 'res_line_dc', None)
        for li in net.line_dc.index:
            if not bool(net.line_dc.at[li, 'in_service']) or res_line is None or li not in res_line.index:
                continue
            if int(net.line_dc.at[li, 'from_bus_dc']) == bus:
                i_out += float(res_line.at[li, 'i_from_ka'])
            if int(net.line_dc.at[li, 'to_bus_dc']) == bus:
                i_out += float(res_line.at[li, 'i_to_ka'])
        res_vsc = getattr(net, 'res_vsc', None)
        for vi in net.vsc.index:
            if int(net.vsc.at[vi, 'bus_dc']) == bus and bool(net.vsc.at[vi, 'in_service']) and res_vsc is not None and vi in res_vsc.index:
                i_out += float(res_vsc.at[vi, 'p_dc_mw']) / v_kv     # negative when the VSC injects
        for si in sources:
            out[si] = i_out / len(sources)
    return out


def _electrisim_source_dc_vm(net, index):
    """res_source_dc has only its power: its voltage is its DC bus's."""
    bus = net.source_dc.at[index, 'bus_dc']
    return net.res_bus_dc.at[bus, 'vm_pu'] if bus in net.res_bus_dc.index else None


def create_busbars(in_data, net):
    Busbars = {}
    # Store user-friendly names mapping for later use
    net.user_friendly_names = {}
    
    # Create a separate dictionary for DC buses
    DcBuses = {}
    
    # Check if DC bus functionality is available in this pandapower version
    has_dc_bus_support = hasattr(pp, 'create_bus_dc')
    if not has_dc_bus_support:
        print(f"Note: pandapower version {pp.__version__} does not support DC buses (create_bus_dc).")
        print("   DC Bus, VSC, and B2B VSC elements will be skipped.")
        print("   You can still use 'DC Line' which connects two AC buses directly.")
        print("   Upgrade to pandapower 3.1+ for full DC grid support.")
    
    for x in in_data:
        if not isinstance(in_data[x], dict) or not isinstance(in_data[x].get('typ'), str):
            continue
        if "DC Bus" in in_data[x]['typ']:
            # Handle DC Bus separately - requires pandapower 3.1+
            if not has_dc_bus_support:
                user_friendly_name = in_data[x].get('userFriendlyName', in_data[x].get('name', 'Unknown'))
                print(f"   Skipping DC Bus '{user_friendly_name}' - DC bus not supported in pandapower {pp.__version__}")
                continue
                
            dc_bus_name = in_data[x]['name']
            user_friendly_name = in_data[x].get('userFriendlyName', dc_bus_name)
            
            # Get in_service parameter (default to True if not specified)
            in_service = True
            if 'in_service' in in_data[x]:
                in_service = bool(in_data[x]['in_service']) if isinstance(in_data[x]['in_service'], bool) else (in_data[x]['in_service'] == 'true' or in_data[x]['in_service'] == True)
            
            dc_kw = dict(
                vn_kv=float(in_data[x].get('vn_kv', 0.0) or 0.0),
                name=dc_bus_name,
                in_service=in_service,
            )
            if in_data[x].get('id') is not None:
                dc_kw['id'] = in_data[x]['id']
            DcBuses[dc_bus_name] = pp.create_bus_dc(net, **dc_kw)
            
            # Store the user-friendly name mapping
            net.user_friendly_names[dc_bus_name] = user_friendly_name
        elif "Bus" in in_data[x]['typ']:
            bus_name = in_data[x]['name']
            user_friendly_name = in_data[x].get('userFriendlyName', bus_name)
            
            # Get in_service parameter (default to True if not specified)
            in_service = True
            if 'in_service' in in_data[x]:
                in_service = bool(in_data[x]['in_service']) if isinstance(in_data[x]['in_service'], bool) else (in_data[x]['in_service'] == 'true' or in_data[x]['in_service'] == True)
            
            bus_kw = dict(
                name=bus_name,
                id=in_data[x]['id'],
                vn_kv=float(in_data[x]['vn_kv']),
                type='b',
                in_service=in_service,
            )
            for vm_key in ('min_vm_pu', 'max_vm_pu'):
                raw = in_data[x].get(vm_key)
                if raw is None or str(raw).strip().lower() in ('', 'none', 'null'):
                    continue
                try:
                    v = float(raw)
                    if v == v and math.isfinite(v) and v >= 0.0:
                        bus_kw[vm_key] = v
                except (TypeError, ValueError):
                    pass
            bus_idx = pp.create_bus(net, **bus_kw)
            Busbars[bus_name] = bus_idx
            # Diagram XML often stores the pandapower semantic name as userFriendlyName while `name`
            # stays as mxObjectId — duplicate the mapping so Switch payloads resolve either key.
            _ufn_b = in_data[x].get('userFriendlyName')
            if _ufn_b not in (None, '') and str(_ufn_b) != str(bus_name):
                Busbars[str(_ufn_b)] = bus_idx
            
            # Store the user-friendly name mapping
            net.user_friendly_names[bus_name] = user_friendly_name
    
    # DC buses keep their own map: in Busbars, a DC bus index read as an AC bus
    # put an AC element drawn on a DC bus on whichever AC bus had that index.
    net.dc_buses = dict(DcBuses)
    
    # Store DC bus names in net object for later use (to distinguish DC vs AC buses)
    net.dc_bus_names = set(DcBuses.keys())
    
    return Busbars


def _is_static_generator_like(typ):
    """True for Static Generator and Wind Turbine (both map to pandapower sgen)."""
    t = typ or ''
    return t.startswith('Static Generator') or t.startswith('Wind Turbine')


def interp_wind_power_at_v(points, v_ms, approx='linear'):
    """Linear or constant (step) interpolate P [MW] at wind speed v_ms from [{v_ms, p_mw}, ...]. Clamp to endpoints."""
    if not isinstance(points, list) or len(points) < 2:
        return None
    try:
        v = float(v_ms)
    except (TypeError, ValueError):
        return None
    knots = []
    for pt in points:
        if not isinstance(pt, dict):
            return None
        try:
            knots.append((float(pt['v_ms']), float(pt['p_mw'])))
        except (KeyError, TypeError, ValueError):
            return None
    knots.sort(key=lambda x: x[0])
    if v <= knots[0][0]:
        return knots[0][1]
    if v >= knots[-1][0]:
        return knots[-1][1]
    style = 'constant' if str(approx or '').strip().lower() == 'constant' else 'linear'
    if style == 'constant':
        for i in range(len(knots) - 1):
            v0, p0 = knots[i]
            v1, _p1 = knots[i + 1]
            if v0 <= v < v1:
                return p0
        return knots[-1][1]
    for i in range(len(knots) - 1):
        v0, p0 = knots[i]
        v1, p1 = knots[i + 1]
        if v0 <= v <= v1:
            span = v1 - v0
            if abs(span) < 1e-12:
                return p0
            t = (v - v0) / span
            return p0 + t * (p1 - p0)
    return knots[-1][1]


def apply_wind_turbine_p_from_curve(elem):
    """
    For Wind Turbine elements, overwrite p_mw from wind_speed_ms + wind_power_curve_json.
    Returns the computed p_mw, or None if not applicable / invalid.
    """
    if not isinstance(elem, dict):
        return None
    typ = elem.get('typ') or ''
    if not typ.startswith('Wind Turbine'):
        return None
    raw = elem.get('wind_power_curve_json')
    if raw is None or (isinstance(raw, str) and not str(raw).strip()):
        return None
    try:
        points = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        print(f"Warning: Wind Turbine '{elem.get('name')}': invalid wind_power_curve_json, keeping p_mw.")
        return None
    approx = elem.get('wind_curve_approx') or 'linear'
    p = interp_wind_power_at_v(points, elem.get('wind_speed_ms'), approx)
    if p is None:
        return None
    elem['p_mw'] = p
    return p


def apply_sgen_q_capability_curves(net, in_data, rpc_use_diagram_curves=False):
    """
    Build pandapower net.q_capability_curve_table and characteristic objects for static generators
    from Electrisim diagram data (reactive_capability_curve, curve_style, q_capability_curve_json).
    Sets net._electrisim_enforce_q_lims True when any curve is applied (required for runpp to use Q limits).

    If rpc_use_diagram_curves is True (RPC with q_capability_mode=from_sgen_curve), also apply curves
    when valid q_capability_curve_json exists even if reactive_capability_curve is false, so RPC can
    use manufacturer P–Q data without requiring the PF checkbox.
    """
    store_sgen_q_cap_2d(net, in_data)
    if in_data is None or not hasattr(net, 'sgen') or net.sgen.empty:
        return
    if 'id' not in net.sgen.columns:
        return

    def _truthy(val):
        if val is None:
            return False
        if isinstance(val, bool):
            return val
        s = str(val).strip().lower()
        return s in ('true', '1', 'yes', 'on')

    rows = []
    updates = []  # (sgen_row_index, curve_id, curve_style_str)
    next_curve_id = 0

    for key, elem in in_data.items():
        if not isinstance(elem, dict):
            continue
        typ = elem.get('typ') or ''
        if not _is_static_generator_like(typ):
            continue
        curve_requested = _truthy(elem.get('reactive_capability_curve'))
        if not curve_requested and rpc_use_diagram_curves:
            raw_try = elem.get('q_capability_curve_json') or elem.get('q_capability_curve_points')
            if raw_try is not None and (not isinstance(raw_try, str) or str(raw_try).strip()):
                curve_requested = True
        if not curve_requested:
            continue
        raw_json = elem.get('q_capability_curve_json') or elem.get('q_capability_curve_points')
        cell_id = elem.get('id')
        cap2d = getattr(net, '_electrisim_q_cap_2d', None) or {}
        rec2d = None
        if cell_id is not None:
            rec2d = cap2d.get(cell_id) or cap2d.get(str(cell_id))
        points = None
        if rec2d:
            points = _flatten_q_cap_2d_mvar_points(rec2d, u_pu=1.0)
        if not points:
            if raw_json is None or (isinstance(raw_json, str) and not raw_json.strip()):
                continue
            try:
                if isinstance(raw_json, str):
                    points = json.loads(raw_json)
                else:
                    points = raw_json
            except (json.JSONDecodeError, TypeError):
                print(f"Warning: Static Generator '{elem.get('name', key)}': invalid q_capability_curve_json, skipping Q curve.")
                continue
            if _is_wind_turbine_typ(typ) and _is_legacy_park_scaled_q_curve(points, elem.get('sn_mva')):
                rec2d = _default_frc_wtg_q_cap_rec(elem)
                points = _flatten_q_cap_2d_mvar_points(rec2d, u_pu=1.0)
                if cell_id is not None and rec2d:
                    cap2d[cell_id] = rec2d
                    cap2d[str(cell_id)] = rec2d
                    net._electrisim_q_cap_2d = cap2d
        if not isinstance(points, list) or len(points) < 2:
            print(f"Warning: Static Generator '{elem.get('name', key)}': Q capability curve needs at least 2 points, skipping.")
            continue
        style = elem.get('curve_style') or 'straightLineYValues'
        if style not in ('straightLineYValues', 'constantYValue'):
            style = 'straightLineYValues'
        try:
            mask = net.sgen['id'] == cell_id
            if not mask.any():
                print(f"Warning: No sgen with id={cell_id!r} for Q capability curve, skipping.")
                continue
            sgen_idx = net.sgen.index[mask][0]
        except Exception as ex:
            print(f"Warning: Could not match sgen for Q curve (id={cell_id!r}): {ex}")
            continue
        curve_id = next_curve_id
        next_curve_id += 1
        try:
            parsed_pts = []
            for pt in points:
                if not isinstance(pt, dict):
                    continue
                parsed_pts.append({
                    'id_q_capability_curve': curve_id,
                    'p_mw': float(pt['p_mw']),
                    'q_min_mvar': float(pt['q_min_mvar']),
                    'q_max_mvar': float(pt['q_max_mvar']),
                })
            if len(parsed_pts) < 2:
                print(f"Warning: Static Generator '{elem.get('name', key)}': fewer than 2 valid curve points, skipping.")
                next_curve_id -= 1
                continue
            parsed_pts.sort(key=lambda r: r['p_mw'])
            rows.extend(parsed_pts)
            updates.append((sgen_idx, curve_id, style))
        except (KeyError, TypeError, ValueError) as ex:
            print(f"Warning: Static Generator '{elem.get('name', key)}': bad Q curve point data: {ex}")
            next_curve_id -= 1
            continue

    if not rows:
        return

    try:
        from pandapower.control.util.auxiliary import create_q_capability_characteristics_object
    except ImportError:
        print('Warning: pandapower.control.util.auxiliary.create_q_capability_characteristics_object not available; Q curves skipped.')
        return

    net['q_capability_curve_table'] = pd.DataFrame(rows)
    for sgen_idx, curve_id, style in updates:
        net.sgen.at[sgen_idx, 'id_q_capability_characteristic'] = curve_id
        net.sgen.at[sgen_idx, 'curve_style'] = style
    create_q_capability_characteristics_object(net)
    net._electrisim_enforce_q_lims = True
    print(f"Applied {len(updates)} static generator Q capability curve(s); power flow will use enforce_q_lims=True.")


def _interp_q_capability_at_p(p_vals, q_vals, p_target, curve_style='straightLineYValues'):
    """
    Interpolate Q at p_target from sorted knot arrays. Matches pandapower curve styles:
    straightLineYValues (linear segments) and constantYValue (Q holds until next P).
    """
    if p_vals is None or q_vals is None or len(p_vals) < 2 or len(q_vals) < 2:
        return None
    p = np.asarray(p_vals, dtype=float)
    q = np.asarray(q_vals, dtype=float)
    p_m = float(p_target)
    if p_m <= p[0]:
        return float(q[0])
    if p_m >= p[-1]:
        return float(q[-1])
    style = curve_style if curve_style in ('straightLineYValues', 'constantYValue') else 'straightLineYValues'
    if style == 'constantYValue':
        for i in range(len(p) - 1):
            if p[i] <= p_m < p[i + 1]:
                return float(q[i])
        return float(q[-1])
    return float(np.interp(p_m, p, q))


def _parse_q_cap_float_list(raw, min_len=1):
    if raw is None:
        return None
    try:
        v = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(v, list) or len(v) < min_len:
            return None
        return [float(x) for x in v]
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _parse_q_cap_float_matrix(raw, n_u, n_p):
    if raw is None:
        return None
    try:
        v = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(v, list) or not v:
        return None
    try:
        if isinstance(v[0], list):
            if len(v) != n_u:
                return None
            out = []
            for row in v:
                if not isinstance(row, list) or len(row) != n_p:
                    return None
                out.append([float(x) for x in row])
            return out
        if len(v) != n_u * n_p:
            return None
        return [[float(v[i * n_p + j]) for j in range(n_p)] for i in range(n_u)]
    except (TypeError, ValueError):
        return None


# Default FRC WTG Q(P,U) — PowerFactory "Fully Rated Converter WTG 2.5MW 50Hz"
# (matches frontend qCapabilityVoltageDependent.js). Rows = U [pu], columns = P [pu].
_FRC_WTG_QCAP_U_PU = [0.9, 0.95, 1.0, 1.05, 1.08, 1.09, 1.095]
_FRC_WTG_QCAP_P_PU = [0, 0.2, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0]
_FRC_WTG_QCAP_QMAX_PU = [
    [0, 0.41, 0.41, 0.41, 0.41, 0.41, 0.39, 0.3, 0, 0, 0],
    [0, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.37, 0.22, 0, 0],
    [0, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.31, 0.2, 0],
    [0, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.38, 0.3, 0.22],
    [0, 0.44, 0.44, 0.44, 0.41, 0.38, 0.36, 0.34, 0.32, 0.3, 0.28],
    [0, 0.4, 0.4, 0.4, 0.37, 0.34, 0.32, 0.3, 0.28, 0.2, 0.17],
    [0, 0.3, 0.3, 0.3, 0.27, 0.24, 0.22, 0.2, 0.18, 0.12, 0.09],
]
_FRC_WTG_QCAP_QMIN_PU = [
    [0, -0.41, -0.41, -0.41, -0.41, -0.41, -0.39, -0.3, 0, 0, 0],
    [0, -0.44, -0.44, -0.44, -0.44, -0.44, -0.44, -0.37, -0.22, 0, 0],
    [0, -0.44, -0.44, -0.44, -0.44, -0.44, -0.44, -0.44, -0.31, -0.2, 0],
    [0, -0.44, -0.44, -0.44, -0.44, -0.44, -0.44, -0.44, -0.38, -0.3, -0.22],
    [0, -0.44, -0.44, -0.44, -0.44, -0.44, -0.44, -0.44, -0.42, -0.35, -0.28],
    [0, -0.44, -0.44, -0.44, -0.44, -0.44, -0.44, -0.44, -0.43, -0.36, -0.3],
    [0, -0.44, -0.44, -0.44, -0.44, -0.44, -0.44, -0.44, -0.44, -0.37, -0.31],
]


def _is_wind_turbine_typ(typ):
    return str(typ or '').strip() == 'Wind Turbine'


def _parse_q_capability_points(raw_json):
    if raw_json is None:
        return None
    try:
        pts = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
    except (json.JSONDecodeError, TypeError):
        return None
    return pts if isinstance(pts, list) else None


def _is_legacy_park_scaled_q_curve(points, sn_mva):
    """Detect old 15 MW park-level 1D curve stored on individual ~2.5 MW WTGs."""
    if not points or len(points) < 2:
        return False
    try:
        sn = float(sn_mva or 0)
    except (TypeError, ValueError):
        sn = 0.0
    p_vals = []
    for pt in points:
        if not isinstance(pt, dict):
            continue
        try:
            p_vals.append(float(pt.get('p_mw')))
        except (TypeError, ValueError):
            pass
    if not p_vals:
        return False
    max_p = max(p_vals)
    if sn > 0 and max_p > sn * 1.5:
        return True
    legacy_p = [0, 3.75, 7.5, 11.25, 15]
    if len(p_vals) == len(legacy_p) and all(abs(p_vals[i] - legacy_p[i]) < 0.01 for i in range(len(legacy_p))):
        return True
    return False


def _default_frc_wtg_q_cap_rec(elem):
    try:
        sn = float(elem.get('sn_mva') or 2.5)
    except (TypeError, ValueError):
        sn = 2.5
    if sn <= 0:
        sn = 2.5
    try:
        smin = float(elem.get('q_cap_scale_min_percent') if elem.get('q_cap_scale_min_percent') not in (None, '') else 100)
    except (TypeError, ValueError):
        smin = 100.0
    try:
        smax = float(elem.get('q_cap_scale_max_percent') if elem.get('q_cap_scale_max_percent') not in (None, '') else 100)
    except (TypeError, ValueError):
        smax = 100.0
    vd_raw = elem.get('q_cap_voltage_dependent')
    if vd_raw is None or vd_raw == '':
        voltage_dependent = True
    else:
        voltage_dependent = str(vd_raw).strip().lower() in ('true', '1', 'yes', 'on')
    return {
        'u': list(_FRC_WTG_QCAP_U_PU),
        'p': list(_FRC_WTG_QCAP_P_PU),
        'qmax': [row[:] for row in _FRC_WTG_QCAP_QMAX_PU],
        'qmin': [row[:] for row in _FRC_WTG_QCAP_QMIN_PU],
        'sn_mva': sn,
        'scale_min': smin,
        'scale_max': smax,
        'voltage_dependent': voltage_dependent,
    }


def _flatten_q_cap_2d_mvar_points(rec, u_pu=1.0):
    """Flatten voltage-dependent Q(P,U) table to pandapower 1D P–Q points at fixed U."""
    if not rec:
        return []
    try:
        sn = float(rec.get('sn_mva') or 0)
    except (TypeError, ValueError):
        sn = 0.0
    if sn <= 0:
        return []
    smin = float(rec.get('scale_min', 100) or 100) / 100.0
    smax = float(rec.get('scale_max', 100) or 100) / 100.0
    u = float(u_pu)
    out = []
    for p_pu in rec.get('p') or []:
        qmin_pu = _bilinear_q_cap_pu(rec['u'], rec['p'], rec['qmin'], u, p_pu)
        qmax_pu = _bilinear_q_cap_pu(rec['u'], rec['p'], rec['qmax'], u, p_pu)
        if qmin_pu is None or qmax_pu is None:
            continue
        out.append({
            'p_mw': float(p_pu) * sn,
            'q_min_mvar': qmin_pu * sn * smin,
            'q_max_mvar': qmax_pu * sn * smax,
        })
    return out


def store_sgen_q_cap_2d(net, in_data):
    """Store voltage-dependent Q(P,U) tables (p.u. of Sn) keyed by diagram cell id."""
    store = {}
    if in_data is None:
        net._electrisim_q_cap_2d = store
        return
    for _key, elem in in_data.items():
        if not isinstance(elem, dict):
            continue
        typ = (elem.get('typ') or '')
        if not _is_static_generator_like(typ):
            continue
        u_axis = _parse_q_cap_float_list(elem.get('q_cap_u_json'), min_len=1)
        p_axis = _parse_q_cap_float_list(elem.get('q_cap_p_json'), min_len=2)
        cell_id = elem.get('id')
        if not u_axis or not p_axis:
            use_default = _is_wind_turbine_typ(typ)
            if not use_default:
                pts = _parse_q_capability_points(elem.get('q_capability_curve_json'))
                use_default = _is_legacy_park_scaled_q_curve(pts, elem.get('sn_mva'))
            if use_default and cell_id is not None:
                rec = _default_frc_wtg_q_cap_rec(elem)
                store[cell_id] = rec
                store[str(cell_id)] = rec
            continue
        n_u, n_p = len(u_axis), len(p_axis)
        qmax = _parse_q_cap_float_matrix(elem.get('q_cap_qmax_json'), n_u, n_p)
        qmin = _parse_q_cap_float_matrix(elem.get('q_cap_qmin_json'), n_u, n_p)
        if qmax is None or qmin is None:
            continue
        u_order = sorted(range(n_u), key=lambda i: u_axis[i])
        p_order = sorted(range(n_p), key=lambda j: p_axis[j])
        u_axis = [u_axis[i] for i in u_order]
        p_axis = [p_axis[j] for j in p_order]
        qmax = [[qmax[i][j] for j in p_order] for i in u_order]
        qmin = [[qmin[i][j] for j in p_order] for i in u_order]
        try:
            sn = float(elem.get('sn_mva') or 0)
        except (TypeError, ValueError):
            sn = 0.0
        if sn <= 0:
            continue
        try:
            smin = float(elem.get('q_cap_scale_min_percent') if elem.get('q_cap_scale_min_percent') not in (None, '') else 100)
        except (TypeError, ValueError):
            smin = 100.0
        try:
            smax = float(elem.get('q_cap_scale_max_percent') if elem.get('q_cap_scale_max_percent') not in (None, '') else 100)
        except (TypeError, ValueError):
            smax = 100.0
        vd_raw = elem.get('q_cap_voltage_dependent')
        if vd_raw is None or vd_raw == '':
            voltage_dependent = len(u_axis) > 1
        else:
            voltage_dependent = str(vd_raw).strip().lower() in ('true', '1', 'yes', 'on')
        rec = {
            'u': u_axis,
            'p': p_axis,
            'qmax': qmax,
            'qmin': qmin,
            'sn_mva': sn,
            'scale_min': smin,
            'scale_max': smax,
            'voltage_dependent': voltage_dependent,
        }
        store[cell_id] = rec
        if cell_id is not None:
            store[str(cell_id)] = rec
    net._electrisim_q_cap_2d = store


def _bilinear_q_cap_pu(u_axis, p_axis, qmat, u_pu, p_pu):
    """Bilinear interpolate Q [p.u.] from voltage rows × P columns. Axes must be sorted."""
    u = float(u_pu)
    p = float(p_pu)
    nu = len(u_axis)
    np_ = len(p_axis)
    if nu < 1 or np_ < 1:
        return None
    if u <= u_axis[0]:
        iu0, iu1, tu = 0, 0, 0.0
    elif u >= u_axis[-1]:
        iu0, iu1, tu = nu - 1, nu - 1, 0.0
    else:
        iu1 = 1
        while iu1 < nu and u_axis[iu1] < u:
            iu1 += 1
        iu0 = iu1 - 1
        du = u_axis[iu1] - u_axis[iu0]
        tu = 0.0 if du == 0 else (u - u_axis[iu0]) / du
    if p <= p_axis[0]:
        ip0, ip1, tp = 0, 0, 0.0
    elif p >= p_axis[-1]:
        ip0, ip1, tp = np_ - 1, np_ - 1, 0.0
    else:
        ip1 = 1
        while ip1 < np_ and p_axis[ip1] < p:
            ip1 += 1
        ip0 = ip1 - 1
        dp = p_axis[ip1] - p_axis[ip0]
        tp = 0.0 if dp == 0 else (p - p_axis[ip0]) / dp
    q00 = float(qmat[iu0][ip0])
    q01 = float(qmat[iu0][ip1])
    q10 = float(qmat[iu1][ip0])
    q11 = float(qmat[iu1][ip1])
    return (1 - tu) * ((1 - tp) * q00 + tp * q01) + tu * ((1 - tp) * q10 + tp * q11)


def _sgen_bus_vm_pu(net, sgen_idx):
    try:
        bus = int(net.sgen.at[sgen_idx, 'bus'])
        if hasattr(net, 'res_bus') and bus in net.res_bus.index:
            vm = net.res_bus.at[bus, 'vm_pu']
            if vm is not None and not (isinstance(vm, float) and pd.isna(vm)):
                return float(vm)
    except Exception:
        pass
    return None


def _interp_sgen_pq_limits(net, sgen_idx, p_mw, vm_pu=None):
    """
    Interpolate (q_min_mvar, q_max_mvar) at p_mw.

    Prefers Electrisim voltage-dependent Q(P,U) tables when present (bilinear in P and U).
    Falls back to net.q_capability_curve_table (1D P–Q) for the characteristic linked
    to net.sgen row sgen_idx. Returns None if unavailable.
    """
    try:
        if not hasattr(net, 'sgen') or net.sgen.empty:
            return None
        cap2d = getattr(net, '_electrisim_q_cap_2d', None) or {}
        cell_id = net.sgen.at[sgen_idx, 'id'] if 'id' in net.sgen.columns else None
        rec = None
        if cell_id is not None:
            rec = cap2d.get(cell_id)
            if rec is None:
                rec = cap2d.get(str(cell_id))
        if rec:
            sn = float(rec.get('sn_mva') or 0)
            if sn > 0:
                if rec.get('voltage_dependent'):
                    u = vm_pu if vm_pu is not None else _sgen_bus_vm_pu(net, sgen_idx)
                    if u is None:
                        u = 1.0
                else:
                    u = 1.0
                p_pu = float(p_mw) / sn
                qmin_pu = _bilinear_q_cap_pu(rec['u'], rec['p'], rec['qmin'], u, p_pu)
                qmax_pu = _bilinear_q_cap_pu(rec['u'], rec['p'], rec['qmax'], u, p_pu)
                if qmin_pu is not None and qmax_pu is not None:
                    smin = float(rec.get('scale_min', 100) or 100) / 100.0
                    smax = float(rec.get('scale_max', 100) or 100) / 100.0
                    return (qmin_pu * sn * smin, qmax_pu * sn * smax)
        if 'id_q_capability_characteristic' not in net.sgen.columns:
            return None
        cid = net.sgen.at[sgen_idx, 'id_q_capability_characteristic']
        if cid is None or (isinstance(cid, float) and pd.isna(cid)):
            return None
        qtbl = net.get('q_capability_curve_table', None)
        if qtbl is None or (hasattr(qtbl, 'empty') and qtbl.empty):
            return None
        if 'id_q_capability_curve' not in qtbl.columns:
            return None
        sub = _rpc_q_curve_table_subset(qtbl, cid)
        if len(sub) < 2:
            return None
        sub = sub.sort_values('p_mw')
        p = sub['p_mw'].astype(float).values
        qmin = sub['q_min_mvar'].astype(float).values
        qmax = sub['q_max_mvar'].astype(float).values
        style = net.sgen.at[sgen_idx, 'curve_style'] if 'curve_style' in net.sgen.columns else 'straightLineYValues'
        if style is None or (isinstance(style, float) and pd.isna(style)):
            style = 'straightLineYValues'
        q_mi = _interp_q_capability_at_p(p, qmin, p_mw, style)
        q_ma = _interp_q_capability_at_p(p, qmax, p_mw, style)
        if q_mi is None or q_ma is None:
            return None
        return (q_mi, q_ma)
    except Exception:
        return None


def _electrisim_truthy(val):
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in ('true', '1', 'yes', 'on')


def apply_sgen_q_setpoint_from_curve(net, in_data):
    """
    Set net.sgen.q_mvar from the Q capability curve when reactive_capability_curve is enabled
    and q_setpoint_mode is capacitive_max or inductive_max. Manual mode keeps diagram q_mvar.
    """
    if in_data is None or not hasattr(net, 'sgen') or net.sgen.empty:
        return
    if 'id' not in net.sgen.columns:
        return

    applied = 0

    for key, elem in in_data.items():
        if not isinstance(elem, dict):
            continue
        typ = elem.get('typ') or ''
        if not _is_static_generator_like(typ):
            continue
        if not _electrisim_truthy(elem.get('reactive_capability_curve')):
            continue
        mode = str(elem.get('q_setpoint_mode') or 'manual').strip().lower()
        if mode not in ('capacitive_max', 'inductive_max'):
            continue
        cell_id = elem.get('id')
        try:
            mask = net.sgen['id'] == cell_id
            if not mask.any():
                continue
            sgen_idx = net.sgen.index[mask][0]
        except Exception:
            continue

        p_mw = float(net.sgen.at[sgen_idx, 'p_mw'])

        lim = _interp_sgen_pq_limits(net, sgen_idx, p_mw, vm_pu=_sgen_bus_vm_pu(net, sgen_idx))
        if lim is None:
            print(f"Warning: Static Generator '{elem.get('name', key)}': Q setpoint from curve skipped (no valid curve).")
            continue
        q_mi, q_ma = lim

        if mode == 'capacitive_max':
            q_effective = q_ma
        else:
            q_effective = q_mi

        net.sgen.at[sgen_idx, 'q_mvar'] = float(q_effective)
        applied += 1
        print(
            f"Static Generator '{elem.get('name', key)}': Q setpoint {q_effective:.4f} MVar "
            f"(mode={mode}, P={p_mw:.4f} MW, curve q_min={q_mi:.4f}, q_max={q_ma:.4f})"
        )

    if applied:
        print(f"Applied Q setpoint from capability curve for {applied} static generator(s).")


def _electrisim_enforce_q_lims_kw(net):
    """Keyword args for pp.runpp when static generator Q capability curves are present."""
    return {'enforce_q_lims': bool(getattr(net, '_electrisim_enforce_q_lims', False))}


def _electrisim_net_has_facts(net):
    """True when an in-service FACTS device is present.

    Pandapower solves SSC (STATCOM), SVC, TCSC, and VSC only with algorithm='nr'.
    """
    for table in ('svc', 'tcsc', 'ssc', 'vsc', 'b2b_vsc'):
        df = getattr(net, table, None)
        if df is None or getattr(df, 'empty', True):
            continue
        if 'in_service' in getattr(df, 'columns', []):
            try:
                if bool(df.in_service.fillna(False).any()):
                    return True
                continue
            except Exception:
                return True
        return True
    return False


def _electrisim_validate_ssc(net):
    """Reject STATCOM data that pandapower cannot solve, and log the values that were sent.

    The element dialog used to default r, x, voltage setpoint and internal voltage to 0.
    A coupling reactance of 0 makes the STATCOM admittance infinite, and a voltage setpoint
    of 0 pu asks Newton-Raphson to collapse the bus. The same network converges with the
    STATCOM removed because those equations are not in the model.
    """
    ssc = getattr(net, 'ssc', None)
    if ssc is None or ssc.empty:
        return
    ext_buses = set()
    gen_buses = set()
    try:
        if hasattr(net, 'ext_grid') and not net.ext_grid.empty and 'bus' in net.ext_grid.columns:
            ext_buses = set(int(b) for b in net.ext_grid.bus.values)
    except Exception:
        ext_buses = set()
    try:
        if hasattr(net, 'gen') and not net.gen.empty and 'bus' in net.gen.columns:
            gens = net.gen
            if 'in_service' in gens.columns:
                gens = gens[gens.in_service.fillna(True).astype(bool)]
            gen_buses = set(int(b) for b in gens.bus.values)
    except Exception:
        gen_buses = set()

    problems = []
    for idx, row in ssc.iterrows():
        in_service = True
        if 'in_service' in ssc.columns:
            try:
                in_service = bool(row['in_service'])
            except Exception:
                in_service = True
        name = row['name'] if 'name' in ssc.columns else idx
        try:
            r_ohm = float(row['r_ohm'])
            x_ohm = float(row['x_ohm'])
            set_vm = float(row['set_vm_pu'])
            vm_int = float(row['vm_internal_pu'])
        except (TypeError, ValueError, KeyError) as ex:
            problems.append(f"SSC '{name}': parameters are not numeric ({ex}).")
            continue
        bus = None
        try:
            bus = int(row['bus'])
        except Exception:
            bus = None
        bus_name = ''
        if bus is not None and hasattr(net, 'bus') and bus in net.bus.index and 'name' in net.bus.columns:
            bus_name = str(net.bus.at[bus, 'name'])
        vn_kv = None
        x_pu = None
        try:
            if bus is not None and bus in net.bus.index:
                vn_kv = float(net.bus.at[bus, 'vn_kv'])
                sn_mva = float(getattr(net, 'sn_mva', 1.0) or 1.0)
                base_z = (vn_kv ** 2) / sn_mva
                if base_z > 0:
                    x_pu = x_ohm / base_z
        except Exception:
            vn_kv = None
            x_pu = None
        # powerflow() redirects stdout/stderr into a buffer, so write to the real console.
        try:
            sys.__stderr__.write(
                f"SSC '{name}' bus={bus_name or bus} vn_kv={vn_kv}: r_ohm={r_ohm}, x_ohm={x_ohm}, "
                f"x_pu={None if x_pu is None else round(x_pu, 6)}, set_vm_pu={set_vm}, "
                f"vm_internal_pu={vm_int}, in_service={in_service}\n"
            )
            sys.__stderr__.flush()
        except Exception:
            pass
        if not in_service:
            continue
        if abs(x_ohm) < 1e-9 and abs(r_ohm) < 1e-9:
            problems.append(
                f"SSC '{name}': coupling impedance is 0 Ω (r_ohm={r_ohm}, x_ohm={x_ohm}). "
                "Set the coupling reactance x_ohm above 0. Pandapower divides by r + jx, "
                "so 0 Ω makes the STATCOM admittance infinite and Newton-Raphson cannot converge."
            )
        elif abs(x_ohm) < 1e-9:
            problems.append(
                f"SSC '{name}': coupling reactance x_ohm is 0. Set x_ohm above 0 Ω."
            )
        if set_vm <= 0:
            problems.append(
                f"SSC '{name}': voltage setpoint set_vm_pu is {set_vm}. "
                "Use a setpoint near 1.0 pu. A value of 0 asks the STATCOM to hold the bus at 0 pu, "
                "which cannot be solved at normal generation levels."
            )
        if vm_int <= 0:
            replacement = set_vm if set_vm > 0 else 1.0
            net.ssc.at[idx, 'vm_internal_pu'] = replacement
            print(
                f"SSC '{name}': internal voltage was {vm_int} pu; "
                f"using {replacement} pu as the Newton-Raphson starting value. The voltage setpoint is unchanged."
            )
        if bus is not None and bus in ext_buses:
            problems.append(
                f"SSC '{name}' is connected to external-grid bus '{bus_name or bus}'. "
                "Pandapower cannot voltage-control a slack bus. Connect the STATCOM to a load or collector bus."
            )
        elif bus is not None and bus in gen_buses:
            problems.append(
                f"SSC '{name}' is connected to generator bus '{bus_name or bus}'. "
                "Pandapower cannot put a STATCOM on a bus that already has a generator voltage setpoint. "
                "Connect it to a PQ bus."
            )
    if problems:
        raise ValueError("STATCOM (SSC) cannot be solved. " + " ".join(problems))


def _electrisim_diagnose_ssc_failure(net, calculate_voltage_angles=True):
    """After an SSC net fails Newton-Raphson, re-solve with the STATCOM out of service.

    The comparison separates "the STATCOM setpoint is unreachable" from "the network itself
    does not solve", and reports the voltage each STATCOM would have to move.
    """
    lines = []
    ssc = getattr(net, 'ssc', None)
    if ssc is None or ssc.empty:
        return ''
    try:
        probe = deepcopy(net)
        probe.ssc['in_service'] = False
        pp.runpp(probe, algorithm='nr', calculate_voltage_angles=calculate_voltage_angles,
                 init='auto', max_iteration=100)
    except Exception as ex:
        lines.append(
            f"Without the STATCOM the network also fails Newton-Raphson ({type(ex).__name__}), "
            "so the STATCOM is not the only problem."
        )
        probe = None
    if probe is not None:
        lines.append("Without the STATCOM the network converges.")
        for idx, row in ssc.iterrows():
            try:
                bus = int(row['bus'])
                vm = float(probe.res_bus.at[bus, 'vm_pu'])
                set_vm = float(row['set_vm_pu'])
                bus_label = str(net.bus.at[bus, 'name']) if 'name' in net.bus.columns else bus
                lines.append(
                    f"STATCOM '{row.get('name')}' at bus '{bus_label}': the bus settles at "
                    f"{vm:.4f} pu on its own, and the STATCOM is set to hold {set_vm:.4f} pu "
                    f"(gap {vm - set_vm:+.4f} pu)."
                )
            except Exception:
                continue
    msg = ' '.join(lines)
    try:
        sys.__stderr__.write('[SSC diagnosis] ' + msg + '\n')
        sys.__stderr__.flush()
    except Exception:
        pass
    return msg


def _ensure_shunt_characteristic_table(net):
    """Ensure pandapower net has net.shunt_characteristic_table DataFrame for step-dependent shunts."""
    if "shunt_characteristic_table" not in net or net["shunt_characteristic_table"] is None:
        net["shunt_characteristic_table"] = pd.DataFrame(columns=["id_characteristic", "step", "q_mvar", "p_mw"])


def _electrisim_parse_shunt_characteristic_table_json(raw_json):
    """
    Parse Electrisim diagram JSON: [{'step','p_mw','q_mvar'}, ...] → rows for pandapower.
    Duplicate step keys use the last occurrence. Returns sorted list of dicts or [].
    """
    if raw_json is None:
        return []
    try:
        if isinstance(raw_json, list):
            data = raw_json
        elif isinstance(raw_json, str):
            data = json.loads(raw_json.strip() or "[]")
        else:
            return []
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    by_step = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        try:
            st = int(round(float(row.get("step", 0))))
            pm = float(row.get("p_mw", 0.0))
            qv = float(row.get("q_mvar", 0.0))
        except (TypeError, ValueError):
            continue
        by_step[st] = {"step": st, "p_mw": pm, "q_mvar": qv}
    return sorted(by_step.values(), key=lambda r: r["step"])


def _electrisim_append_shunt_characteristic_table(net, rows):
    """
    Append one characteristic family (multiple steps) under a new id_characteristic index.
    Returns id_characteristic (int), or None if rows empty.
    """
    if not rows:
        return None
    _ensure_shunt_characteristic_table(net)
    df_existing = net["shunt_characteristic_table"]
    if df_existing.empty:
        nid = 0
    else:
        try:
            mx = pd.to_numeric(df_existing["id_characteristic"], errors="coerce")
            nid = int(mx.fillna(-1).max()) + 1
        except Exception:
            nid = int(len(df_existing))
    add_rows = []
    for r in rows:
        add_rows.append({
            "id_characteristic": nid,
            "step": int(r["step"]),
            "q_mvar": float(r["q_mvar"]),
            "p_mw": float(r["p_mw"]),
        })
    new_df = pd.DataFrame(add_rows, columns=["id_characteristic", "step", "q_mvar", "p_mw"])
    if df_existing.empty:
        net["shunt_characteristic_table"] = new_df
    else:
        net["shunt_characteristic_table"] = pd.concat([df_existing, new_df], ignore_index=True)
    return nid


def _electrisim_shunt_nominals_for_step(rows, step_val, p_fallback, q_fallback):
    """Nominal p_mw, q_mvar at v = 1.0 pu for the discrete step matching step_val."""
    try:
        si = int(round(float(step_val)))
    except (TypeError, ValueError):
        si = 1
    for r in rows:
        if int(r["step"]) == si:
            return float(r["p_mw"]), float(r["q_mvar"])
    try:
        return float(p_fallback), float(q_fallback)
    except (TypeError, ValueError):
        return 0.0, 0.0


def _electrisim_shunt_uses_zero_based(electrisim_step, characteristic_rows, lf_bands):
    """
    Electrisim tap indices are 0..max_step; pandapower uses 1..max_step and treats step=0 as off.
    When step 0 appears in the diagram payload, shift +1 for pandapower and map back on export.
    """
    try:
        if int(round(float(electrisim_step))) == 0:
            return True
    except (TypeError, ValueError):
        pass
    for r in characteristic_rows or []:
        try:
            if int(r.get("step", -1)) == 0:
                return True
        except (TypeError, ValueError):
            continue
    for b in lf_bands or []:
        try:
            if int(b.get("step", -1)) == 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _electrisim_shunt_step_to_pp(step_val, zero_based):
    try:
        s = int(round(float(step_val)))
    except (TypeError, ValueError):
        s = 1
    if zero_based:
        return max(1, s + 1)
    return max(1, s)


def _electrisim_shunt_step_from_pp(step_val, zero_based):
    try:
        s = float(step_val)
        if math.isnan(s) or math.isinf(s):
            return None
        s = int(round(s))
    except (TypeError, ValueError):
        return None
    if zero_based:
        return max(0, s - 1)
    return s


def _electrisim_shunt_max_step_to_pp(max_step, zero_based):
    try:
        m = int(round(float(max_step)))
    except (TypeError, ValueError):
        m = 1
    return max(1, m + 1) if zero_based else max(1, m)


def _electrisim_shift_shunt_characteristic_rows_pp(rows, zero_based):
    if not zero_based or not rows:
        return rows
    return [{"step": int(r["step"]) + 1, "p_mw": r["p_mw"], "q_mvar": r["q_mvar"]} for r in rows]


def _electrisim_shunt_is_zero_based(net, shunt_index):
    zb = getattr(net, "_electrisim_shunt_zero_based", None) or {}
    return bool(zb.get(int(shunt_index)))


def _electrisim_shunt_res_for_output(net, shunt_index, row):
    """Repair NaN p_mw/q_mvar from pandapower when step=0 with step_dependency_table (divide-by-zero)."""
    def _is_nan(v):
        try:
            v = float(v)
            return not (v == v)
        except (TypeError, ValueError):
            return True

    def _as_float(v, default=float("nan")):
        try:
            x = float(v)
            if math.isnan(x) or math.isinf(x):
                return default
            return x
        except (TypeError, ValueError):
            return default

    p_mw = _as_float(row.get("p_mw"))
    q_mvar = _as_float(row.get("q_mvar"))
    vm_pu = _as_float(row.get("vm_pu"), 1.0)
    if not _is_nan(p_mw) and not _is_nan(q_mvar):
        return p_mw, q_mvar, vm_pu

    if getattr(net, "shunt", None) is None or net.shunt.empty or shunt_index not in net.shunt.index:
        return (0.0 if _is_nan(p_mw) else p_mw), (0.0 if _is_nan(q_mvar) else q_mvar), vm_pu

    sh = net.shunt.loc[shunt_index]
    use_table = bool(sh.get("step_dependency_table", False))
    if not use_table:
        return (0.0 if _is_nan(p_mw) else p_mw), (0.0 if _is_nan(q_mvar) else q_mvar), vm_pu

    try:
        step_pp = float(sh["step"])
        id_char = sh.get("id_characteristic_table")
        bus_i = int(sh["bus"])
        vn_sh = float(sh["vn_kv"])
        vn_bus = float(net.bus.at[bus_i, "vn_kv"])
        vm = float(vm_pu) if not _is_nan(vm_pu) else float(net.res_bus.at[bus_i, "vm_pu"])
        v_ratio = (vn_bus / vn_sh) ** 2 if vn_sh > 0 else 1.0
        char_df = getattr(net, "shunt_characteristic_table", None)
        if char_df is None or char_df.empty or id_char is None or pd.isna(id_char):
            return (0.0 if _is_nan(p_mw) else p_mw), (0.0 if _is_nan(q_mvar) else q_mvar), vm

        sel = (char_df["id_characteristic"] == id_char) & (char_df["step"] == step_pp)
        if not sel.any():
            return (0.0 if _is_nan(p_mw) else p_mw), (0.0 if _is_nan(q_mvar) else q_mvar), vm

        crow = char_df.loc[sel].iloc[0]
        p_char = float(crow["p_mw"])
        q_char = float(crow["q_mvar"])
        if step_pp == 0:
            p_out = 0.0
            q_out = 0.0
        else:
            p_out = (vm ** 2) * p_char * v_ratio
            q_out = (vm ** 2) * q_char * v_ratio
        return (
            p_out if _is_nan(p_mw) else p_mw,
            q_out if _is_nan(q_mvar) else q_mvar,
            vm if not _is_nan(vm_pu) else vm,
        )
    except Exception:
        return (0.0 if _is_nan(p_mw) else p_mw), (0.0 if _is_nan(q_mvar) else q_mvar), vm_pu


def _electrisim_find_line_index_by_cell_id(net, cell_id):
    """Resolve diagram line cell id → pandapower net.line row index.

    Exported JSON uses mxGraph ``cell.id``. Legacy UI sometimes stored ``getId()`` strings
    (e.g. ``layerPrefix-14``). Match exact string first; then compare to the last hyphen
    suffix so ``*-14`` still resolves when ``net.line.id`` holds ``\"14\"``.
    """
    if cell_id is None or str(cell_id).strip() == "":
        return None
    if getattr(net, "line", None) is None or net.line.empty or "id" not in net.line.columns:
        return None
    try:
        cid = str(cell_id).strip()
        col = net.line["id"]

        # 1) Exact string match (pandas / pandapower may store numeric or text ids).
        ids_str = col.astype(str)
        m = net.line.index[ids_str == cid]
        if len(m):
            return int(m[0])

        # 2) Hyphen suffix: hierarchical mxGraph ids vs plain row id stored on line.
        if "-" in cid:
            suff = cid.split("-")[-1].strip()
            if suff != cid:
                m = net.line.index[ids_str == suff]
                if len(m):
                    return int(m[0])
                # Rare: id column is numeric, suffix parses as integer part only.
                if suff.isdigit():
                    mv = net.line.index[col == int(suff)]
                    if len(mv):
                        return int(mv[0])

        return None
    except Exception:
        return None


def _electrisim_finalize_pending_line_flow_shunts(net):
    """
    Line P→shunt step must be registered after all ``pp.create_line_*`` calls: ``in_data`` key order
    is not guaranteed, so shunt entries can be processed before Line entries and ``net.line`` was empty.
    Pending specs are queued on each shunt; this runs once at end of create_other_elements().
    """
    pending = getattr(net, "_electrisim_pending_line_flow_shunts", None) or []
    if not pending:
        return
    if not hasattr(net, "line_flow_shunt_controllers"):
        net.line_flow_shunt_controllers = []
    for item in pending:
        ref = item.get("ref_line")
        line_pp_idx = _electrisim_find_line_index_by_cell_id(net, ref)
        lf_rows = item.get("bands") or []
        name = item.get("name", "?")
        if line_pp_idx is None:
            print(
                f"Warning: Shunt reactor '{name}': line_flow_step_control enabled "
                f"but reference line id {ref!r} does not match any net.line.id "
                f"(pick the diagram Line again in the shunt dialog, or ensure the element is an AC Line)."
            )
            continue
        if not lf_rows:
            print(
                f"Warning: Shunt reactor '{name}': line_flow_step_control enabled "
                f"but line_flow_step_table_json is empty or invalid."
            )
            continue
        net.line_flow_shunt_controllers.append({
            "shunt_index": int(item["shunt_index"]),
            "line_index": int(line_pp_idx),
            "p_col": item.get("p_col") or "p_from_mw",
            "use_abs": bool(item.get("use_abs", True)),
            "bands": lf_rows,
            "zero_based_steps": bool(item.get("zero_based_steps", False)),
        })
    try:
        del net._electrisim_pending_line_flow_shunts
    except Exception:
        pass


def _electrisim_parse_line_flow_step_table_json(raw_json):
    """
    [{ p_mw_min, p_mw_max, step }, ...] from Electrisim — sorted by band start.
    """
    if raw_json is None:
        return []
    try:
        if isinstance(raw_json, list):
            data = raw_json
        elif isinstance(raw_json, str):
            data = json.loads(raw_json.strip() or "[]")
        else:
            return []
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    out = []
    for row in data:
        if not isinstance(row, dict):
            continue
        try:
            lo = float(row.get("p_mw_min", 0))
            hi = float(row.get("p_mw_max", 0))
            st = int(row.get("step", 0))
        except (TypeError, ValueError):
            continue
        out.append({"p_mw_min": lo, "p_mw_max": hi, "step": st})
    out.sort(key=lambda r: (r["p_mw_min"], r["p_mw_max"]))
    return out


def _electrisim_line_flow_pick_step(bands, p_mw):
    """
    Map active power [MW] to integer step using band table.
    UI: [min, max) semi-open intervals; **last band** includes **both** endpoints.
    Fallback outside all bands: nearest edge step.
    """
    if not bands:
        return 0
    try:
        pv = float(p_mw)
    except (TypeError, ValueError):
        pv = 0.0
    if pv < float(bands[0]["p_mw_min"]):
        return int(bands[0]["step"])
    if pv > float(bands[-1]["p_mw_max"]):
        return int(bands[-1]["step"])
    n = len(bands)
    for i, b in enumerate(bands):
        lo = float(b["p_mw_min"])
        hi = float(b["p_mw_max"])
        st = int(b["step"])
        if i == n - 1:
            if lo <= pv <= hi:
                return st
        else:
            if lo <= pv < hi:
                return st
    return int(bands[-1]["step"])


def _electrisim_attach_line_flow_shunt_controllers(net):
    """
    pandapower CharacteristicControl: res_line.{p_from_mw|p_to_mw} → net.shunt.step
    Uses dense piecewise-linear Characteristic sampled at 1 MW for tabulated step bands.
    """
    lst = getattr(net, "line_flow_shunt_controllers", None) or []
    if not lst:
        return
    try:
        from pandapower.control.util.characteristic import Characteristic
    except ImportError:
        print("Warning: pandapower Characteristic unavailable; Line P→shunt step controllers skipped.")
        return
    for spec in lst:
        try:
            bands = spec.get("bands") or []
            if not bands:
                continue
            ln_i = int(spec["line_index"])
            sh_i = int(spec["shunt_index"])
            use_abs = bool(spec.get("use_abs", True))
            p_col = str(spec.get("p_col") or "p_from_mw")
            if p_col not in ("p_from_mw", "p_to_mw"):
                p_col = "p_from_mw"
            hi_max = max(float(b["p_mw_max"]) for b in bands)
            # Dense sample up to plausible |P|; cap for memory. If use_abs is enabled,
            # include negative x-values because CharacteristicControl passes raw res_line P.
            span = int(min(max(hi_max + 500.0, 512.0), 2e6))

            def _interp_y(p_scalar):
                try:
                    p0 = abs(float(p_scalar)) if use_abs else float(p_scalar)
                except (TypeError, ValueError):
                    p0 = 0.0
                return float(_electrisim_line_flow_pick_step(bands, p0))

            if use_abs:
                xs = [float(k) for k in range(-span, span + 1)]
            else:
                lo_min = min(float(b["p_mw_min"]) for b in bands)
                lo_span = int(max(abs(lo_min) + 500.0, 512.0)) if lo_min < 0 else 0
                xs = [float(k) for k in range(-lo_span, span + 1)]
            ys = [_interp_y(float(x)) for x in xs]
            if bool(spec.get("zero_based_steps")):
                ys = [float(y) + 1.0 for y in ys]

            ch = Characteristic(net, x_values=xs, y_values=ys)
            control.CharacteristicControl(
                net,
                output_element="shunt",
                output_variable="step",
                output_element_index=sh_i,
                input_element="res_line",
                input_variable=p_col,
                input_element_index=ln_i,
                characteristic_index=ch.index,
                tol=1e-6,
            )
        except Exception as ex:
            print(f"Warning: Line-flow shunt controller registration failed for spec {spec!r}: {ex}")


def create_other_elements(in_data,net,x, Busbars):
    # The study's own parameters (x is reused below for each element).
    study = str(in_data[x].get('typ', '')) if isinstance(in_data.get(x), dict) else ''

    #tworzymy zmienne ktorych nazwa odpowiada modelowi z js - np.Hwap0ntfbV98zYtkLMVm-8

    # Helper function for safe type conversion (local version with different default)
    def safe_float_local(value, default=None):
        if value is None or value == 'None' or value == '':
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    # Maps for Switch element lookup: line/trafo name -> pandapower index
    LinesDict = {}
    TrafoDict = {}
    Trafo3wDict = {}

    def _map_et_to_pandapower(et):
        """Map frontend et values to pandapower single-letter codes."""
        if et is None or et == '':
            return 'l'
        et_lower = str(et).lower()
        if et_lower in ('l', 'line'):
            return 'l'
        if et_lower in ('t', 'trafo', 'transformer'):
            return 't'
        if et_lower in ('t3', 'trafo3w', 'trafo3winding'):
            return 't3'
        if et_lower == 'b':
            return 'b'
        return et_lower[0] if et_lower else 'l'

    _payload_keys = list(in_data.keys())
    _key_pos = {k: i for i, k in enumerate(_payload_keys)}

    def _creation_order_key(k):
        row = in_data.get(k)
        if not isinstance(row, dict):
            return (1, _key_pos.get(k, 0))
        typ = str(row.get('typ', ''))
        if typ.startswith('Switch'):
            return (2, _key_pos.get(k, 0))
        if typ.startswith('Line'):
            return (0, _key_pos.get(k, 0))
        return (1, _key_pos.get(k, 0))

    _ordered_keys = sorted(_payload_keys, key=_creation_order_key)

    for name,value in Busbars.items():
        globals()[name] = value    
       
    for x in _ordered_keys:
        if not isinstance(in_data[x], dict) or not isinstance(in_data[x].get('typ'), str):
            continue
        # Study settings ride in the same payload ("MotorStartingPandaPower
        # Parameters" matched the Motor branch below and failed on 'bus').
        if in_data[x]['typ'].endswith(' Parameters'):
            continue

        #eval - rozwiazuje problem z wartosciami NaN
        if (in_data[x]['typ'].startswith("Line")):
            try:
                # Lines have busFrom and busTo fields directly
                bus_from = in_data[x].get('busFrom')
                bus_to = in_data[x].get('busTo')                
             
                from_bus_idx = Busbars.get(bus_from)
                to_bus_idx = Busbars.get(bus_to)
                
                if from_bus_idx is None:
                    element_name = in_data[x].get('userFriendlyName', in_data[x].get('name', 'Unknown'))
                    raise ValueError(
                        f"CONNECTION ERROR: Line '{element_name}' is trying to connect from bus '{bus_from}', "
                        f"but this bus does not exist in your diagram.\n\n"
                        f"SOLUTION: Please ensure that:\n"
                        f"1. You have placed Bus/Busbar elements at both ends of the line\n"
                        f"2. The line is properly connected to these Bus elements\n"
                        f"3. Lines must connect two Bus elements (from_bus and to_bus)"
                    )
                if to_bus_idx is None:
                    element_name = in_data[x].get('userFriendlyName', in_data[x].get('name', 'Unknown'))
                    raise ValueError(
                        f"CONNECTION ERROR: Line '{element_name}' is trying to connect to bus '{bus_to}', "
                        f"but this bus does not exist in your diagram.\n\n"
                        f"SOLUTION: Please ensure that:\n"
                        f"1. You have placed Bus/Busbar elements at both ends of the line\n"
                        f"2. The line is properly connected to these Bus elements\n"
                        f"3. Lines must connect two Bus elements (from_bus and to_bus)"
                    )
                    
         
            except Exception as e:
                continue
                
            # Create a base parameters dict with required parameters
            line_params = {
                "from_bus": from_bus_idx,
                "to_bus": to_bus_idx,
                "name": in_data[x]['name'],
                "id": in_data[x]['id'],
                "r_ohm_per_km": float(in_data[x]['r_ohm_per_km']),
                "x_ohm_per_km": float(in_data[x]['x_ohm_per_km']),
                "c_nf_per_km": float(in_data[x]['c_nf_per_km']),
                "g_us_per_km": float(in_data[x]['g_us_per_km']),
                "max_i_ka": float(in_data[x]['max_i_ka']),
                "type": in_data[x]['type'],
                "length_km": float(in_data[x]['length_km'])
            }

            # Handle optional parameters - include if they have valid values
            optional_params = ['parallel', 'df']
            
            for param in optional_params:
                value = in_data[x].get(param)
                if value is not None and value not in ('None', '', 'null'):
                    try:
                        if param == 'parallel':
                            line_params[param] = int(value)
                        else:  # df
                            line_params[param] = float(value)
                    except (ValueError, TypeError):
                        # If conversion fails, use default values
                        if param == 'parallel':
                            line_params[param] = 1
                        else:  # df
                            line_params[param] = 1.0
                else:
                    # Use default values when parameter is None or empty
                    if param == 'parallel':
                        line_params[param] = 1
                    else:  # df
                        line_params[param] = 1.0

            # Make sure zero sequence parameters are explicitly converted to float
            # This should solve the isnan() issue
            try:
                if 'r0_ohm_per_km' in in_data[x] and in_data[x]['r0_ohm_per_km'] is not None:
                    line_params["r0_ohm_per_km"] = float(in_data[x]['r0_ohm_per_km'])
                else:
                    line_params["r0_ohm_per_km"] = 1.0  # Default as float
                    
                if 'x0_ohm_per_km' in in_data[x] and in_data[x]['x0_ohm_per_km'] is not None:
                    line_params["x0_ohm_per_km"] = float(in_data[x]['x0_ohm_per_km'])
                else:
                    line_params["x0_ohm_per_km"] = 1.0  # Default as float
                    
                if 'c0_nf_per_km' in in_data[x] and in_data[x]['c0_nf_per_km'] is not None:
                    line_params["c0_nf_per_km"] = float(in_data[x]['c0_nf_per_km'])
                else:
                    line_params["c0_nf_per_km"] = 0.0  # Default as float
            except (ValueError, TypeError) as e:
                # Set to defaults if conversion fails
                line_params["r0_ohm_per_km"] = 1.0
                line_params["x0_ohm_per_km"] = 1.0
                line_params["c0_nf_per_km"] = 0.0

            # Handle endtemp_degree separately as it's truly optional
            if 'endtemp_degree' in in_data[x] and in_data[x]['endtemp_degree'] is not None:
                try:
                    line_params["endtemp_degree"] = float(in_data[x]['endtemp_degree'])
                except (ValueError, TypeError):
                    # Skip adding this parameter if conversion fails
                    pass

            # Add in_service parameter (default to True if not specified)
            if 'in_service' in in_data[x]:
                line_params["in_service"] = bool(in_data[x]['in_service']) if isinstance(in_data[x]['in_service'], bool) else (in_data[x]['in_service'] == 'true' or in_data[x]['in_service'] == True)
            else:
                line_params["in_service"] = True

            _mlp_line = _electrisim_optional_max_loading_percent(in_data[x].get('max_loading_percent'))
            if _mlp_line is not None:
                line_params['max_loading_percent'] = _mlp_line

            # Call the function with the prepared parameters
            line_idx = pp.create_line_from_parameters(net, **line_params)
            LinesDict[in_data[x]['name']] = line_idx
            _ufn_ln = in_data[x].get('userFriendlyName')
            if _ufn_ln not in (None, '') and str(_ufn_ln) != str(in_data[x]['name']):
                LinesDict[str(_ufn_ln)] = line_idx
            
            # Store user-friendly name for line
            line_name = in_data[x]['name']
            user_friendly_name = in_data[x].get('userFriendlyName', line_name)
            if not hasattr(net, 'user_friendly_names'):
                net.user_friendly_names = {}
            net.user_friendly_names[line_name] = user_friendly_name  
            
            
            #pp.create_line_from_parameters(net,  from_bus=eval(in_data[x]['busFrom']), to_bus=eval(in_data[x]['busTo']), name=in_data[x]['name'], id=in_data[x]['id'], r_ohm_per_km=in_data[x]['r_ohm_per_km'], x_ohm_per_km=in_data[x]['x_ohm_per_km'], c_nf_per_km= in_data[x]['c_nf_per_km'], g_us_per_km= in_data[x]['g_us_per_km'], 
            #                               r0_ohm_per_km=1, x0_ohm_per_km=1, c0_nf_per_km=0, endtemp_degree=in_data[x]['endtemp_degree'],
            #                               max_i_ka= in_data[x]['max_i_ka'],type= in_data[x]['type'], length_km=in_data[x]['length_km'], parallel=in_data[x]['parallel'], df=in_data[x]['df'])
            #w specyfikacji zapisano, że poniższe parametry są typu nan. Wartosci składowych zerowych mogą być wprowadzone przez funkcję create line.
            #r0_ohm_per_km= in_data[x]['r0_ohm_per_km'], x0_ohm_per_km= in_data[x]['x0_ohm_per_km'], c0_nf_per_km= in_data[x]['c0_nf_per_km'], max_loading_percent=in_data[x]['max_loading_percent'], endtemp_degree=in_data[x]['endtemp_degree'],
        
        if (in_data[x]['typ'].startswith("External Grid") or in_data[x]['typ'].startswith("ExternalGrid")):
            bus_idx = Busbars.get(in_data[x]['bus'])
            if bus_idx is None:
                element_name = in_data[x].get('userFriendlyName', in_data[x].get('name', 'Unknown'))
                bus_name = in_data[x]['bus']
                
                # Check if bus is None (not connected) or references non-existent bus
                if bus_name is None:
                    raise ValueError(
                        f"CONNECTION ERROR: External Grid '{element_name}' (ID: {in_data[x].get('id', 'Unknown')}) is NOT CONNECTED to any bus.\n\n"
                        f"SOLUTION: Please ensure that:\n"
                        f"1. You have placed a Bus/Busbar element in your diagram\n"
                        f"2. Draw a connection line from the External Grid to a Bus element\n"
                        f"3. Verify the connection line is properly attached at both ends\n\n"
                        f"IMPORTANT: Every electrical component must be connected to at least one Bus element."
                    )
                else:
                    raise ValueError(
                        f"CONNECTION ERROR: External Grid '{element_name}' is trying to connect to bus '{bus_name}', "
                        f"but this bus does not exist in your diagram.\n\n"
                        f"SOLUTION: Please ensure that:\n"
                        f"1. You have placed a Bus/Busbar element in your diagram\n"
                        f"2. The External Grid is connected to this Bus element with a connection line\n"
                        f"3. All electrical elements (External Grids, Generators, Loads, Transformers, etc.) "
                        f"are properly connected to Bus elements\n\n"
                        f"IMPORTANT: Each electrical component must be connected to at least one Bus element."
                    )

            # Get in_service parameter (default to True if not specified)
            in_service = True
            if 'in_service' in in_data[x]:
                in_service = bool(in_data[x]['in_service']) if isinstance(in_data[x]['in_service'], bool) else (in_data[x]['in_service'] == 'true' or in_data[x]['in_service'] == True)
            
            # Validate and auto-correct vm_pu for external grid
            ext_grid_vm_pu = safe_float(in_data[x]['vm_pu'])
            bus_vn_kv = net.bus.loc[bus_idx, 'vn_kv'] if bus_idx is not None else None
            
            if ext_grid_vm_pu == 0:
                ext_grid_vm_pu = 1.0
                print(f"WARNING: External Grid '{in_data[x].get('userFriendlyName', in_data[x]['name'])}' had vm_pu=0, auto-corrected to 1.0 p.u.")
            elif ext_grid_vm_pu > 1.5 and bus_vn_kv is not None and bus_vn_kv > 0:
                # User likely entered voltage in kV instead of per unit
                corrected_vm_pu = ext_grid_vm_pu / bus_vn_kv
                print(f"WARNING: External Grid '{in_data[x].get('userFriendlyName', in_data[x]['name'])}' has vm_pu={ext_grid_vm_pu}, "
                      f"which is unreasonably high. vm_pu should be close to 1.0 (per unit). "
                      f"Bus nominal voltage is {bus_vn_kv} kV. Auto-correcting vm_pu from {ext_grid_vm_pu} to {corrected_vm_pu:.4f} p.u. "
                      f"(assuming user entered kV instead of p.u.)")
                if not hasattr(net, 'warnings'):
                    net.warnings = []
                net.warnings.append(
                    f"External Grid '{in_data[x].get('userFriendlyName', in_data[x]['name'])}': "
                    f"vm_pu was set to {ext_grid_vm_pu}, which appears to be a voltage in kV, not per unit. "
                    f"Auto-corrected to {corrected_vm_pu:.4f} p.u. "
                    f"(vm_pu should be close to 1.0, e.g. 0.95-1.05 for normal operation). "
                    f"The bus nominal voltage ({bus_vn_kv} kV) is used as the base."
                )
                ext_grid_vm_pu = corrected_vm_pu
            
            ext_grid_kw = dict(
                vm_pu=ext_grid_vm_pu,
                va_degree=safe_float(in_data[x]['va_degree']),
                s_sc_max_mva=safe_float(in_data[x]['s_sc_max_mva']),
                s_sc_min_mva=safe_float(in_data[x]['s_sc_min_mva']),
                rx_max=safe_float(in_data[x]['rx_max']),
                rx_min=safe_float(in_data[x]['rx_min']),
                r0x0_max=safe_float(in_data[x].get('r0x0_max')),
                x0x_max=safe_float(in_data[x].get('x0x_max')),
                r0x0_min=_ext_grid_zero_seq_min(in_data[x], 'r0x0_min', safe_float(in_data[x].get('r0x0_max'))),
                x0x_min=_ext_grid_zero_seq_min(in_data[x], 'x0x_min', safe_float(in_data[x].get('x0x_max'))),
                in_service=in_service,
            )
            for fld in ('max_p_mw', 'min_p_mw'):
                raw = in_data[x].get(fld)
                if raw is None or str(raw).lower() in ('null', 'none', ''):
                    continue
                try:
                    ext_grid_kw[fld] = float(raw)
                except (TypeError, ValueError):
                    pass
            # OPF: leaving min_q=max_q=0 (diagram defaults) fixes slack Q to exactly 0 MVar — AC OPF then often
            # fails to converge because the reference cannot exchange reactive power with PV machines / loads.
            qmin_raw = in_data[x].get('min_q_mvar')
            qmax_raw = in_data[x].get('max_q_mvar')
            qmin_parsed = None
            qmax_parsed = None
            if qmin_raw is not None and str(qmin_raw).lower() not in ('null', 'none', ''):
                try:
                    qmin_parsed = float(qmin_raw)
                except (TypeError, ValueError):
                    qmin_parsed = None
            if qmax_raw is not None and str(qmax_raw).lower() not in ('null', 'none', ''):
                try:
                    qmax_parsed = float(qmax_raw)
                except (TypeError, ValueError):
                    qmax_parsed = None
            if qmin_parsed is not None and qmax_parsed is not None and qmin_parsed == 0.0 and qmax_parsed == 0.0:
                pass
            else:
                if qmin_parsed is not None:
                    ext_grid_kw['min_q_mvar'] = qmin_parsed
                if qmax_parsed is not None:
                    ext_grid_kw['max_q_mvar'] = qmax_parsed
            cont_raw = in_data[x].get('controllable')
            if cont_raw is not None:
                ext_grid_kw['controllable'] = (
                    bool(cont_raw) if isinstance(cont_raw, bool)
                    else str(cont_raw).lower() in ('true', '1')
                )
            pp.create_ext_grid(
                net,
                bus=bus_idx,
                name=in_data[x]['name'],
                id=in_data[x]['id'],
                **ext_grid_kw
            )
            
            # Store user-friendly name for external grid
            ext_grid_name = in_data[x]['name']
            user_friendly_name = in_data[x].get('userFriendlyName', ext_grid_name)
            if not hasattr(net, 'user_friendly_names'):
                net.user_friendly_names = {}
            net.user_friendly_names[ext_grid_name] = user_friendly_name
       
        if (in_data[x]['typ'].startswith("Generator")):
            bus_idx = Busbars.get(in_data[x]['bus'])
            if bus_idx is None:
                element_name = in_data[x].get('userFriendlyName', in_data[x].get('name', 'Unknown'))
                bus_name = in_data[x]['bus']
                
                # Check if bus is None (not connected) or references non-existent bus
                if bus_name is None:
                    raise ValueError(
                        f"CONNECTION ERROR: Generator '{element_name}' (ID: {in_data[x].get('id', 'Unknown')}) is NOT CONNECTED to any bus.\n\n"
                        f"SOLUTION: Please ensure that:\n"
                        f"1. You have placed a Bus/Busbar element in your diagram\n"
                        f"2. Draw a connection line from the Generator to a Bus element\n"
                        f"3. Verify the connection line is properly attached at both ends\n\n"
                        f"IMPORTANT: Every electrical component must be connected to at least one Bus element."
                    )
                else:
                    raise ValueError(
                        f"CONNECTION ERROR: Generator '{element_name}' is trying to connect to bus '{bus_name}', "
                        f"but this bus does not exist in your diagram.\n\n"
                        f"SOLUTION: Please ensure that:\n"
                        f"1. You have placed a Bus/Busbar element in your diagram\n"
                        f"2. The Generator is connected to this Bus element with a connection line\n"
                        f"3. All electrical elements must be properly connected to Bus elements"
                    )
    
            # Get in_service parameter (default to True if not specified)
            in_service = True
            if 'in_service' in in_data[x]:
                in_service = bool(in_data[x]['in_service']) if isinstance(in_data[x]['in_service'], bool) else (in_data[x]['in_service'] == 'true' or in_data[x]['in_service'] == True)
            
            # Get slack parameter (default to False if not specified)
            slack = False
            if 'slack' in in_data[x]:
                slack = bool(in_data[x]['slack']) if isinstance(in_data[x]['slack'], bool) else (str(in_data[x]['slack']).lower() == 'true')
            
            # Validate and auto-correct vm_pu for generator
            gen_vm_pu = safe_float(in_data[x]['vm_pu'])
            gen_bus_vn_kv = net.bus.loc[bus_idx, 'vn_kv'] if bus_idx is not None else None
            
            if gen_vm_pu == 0:
                gen_vm_pu = 1.0
                print(f"WARNING: Generator '{in_data[x].get('userFriendlyName', in_data[x]['name'])}' had vm_pu=0, auto-corrected to 1.0 p.u.")
            elif gen_vm_pu > 1.5 and gen_bus_vn_kv is not None and gen_bus_vn_kv > 0:
                corrected_vm_pu = gen_vm_pu / gen_bus_vn_kv
                print(f"WARNING: Generator '{in_data[x].get('userFriendlyName', in_data[x]['name'])}' has vm_pu={gen_vm_pu}, "
                      f"which is unreasonably high. Auto-correcting to {corrected_vm_pu:.4f} p.u.")
                if not hasattr(net, 'warnings'):
                    net.warnings = []
                net.warnings.append(
                    f"Generator '{in_data[x].get('userFriendlyName', in_data[x]['name'])}': "
                    f"vm_pu was set to {gen_vm_pu}, which appears to be a voltage in kV, not per unit. "
                    f"Auto-corrected to {corrected_vm_pu:.4f} p.u. "
                    f"(vm_pu should be close to 1.0, e.g. 0.95-1.05 for normal operation)."
                )
                gen_vm_pu = corrected_vm_pu
            elif gen_vm_pu > 0 and gen_vm_pu < 0.5:
                print(f"WARNING: Generator '{in_data[x].get('userFriendlyName', in_data[x]['name'])}' has unusually low vm_pu={gen_vm_pu}. "
                      f"vm_pu should be close to 1.0 (per unit). Please verify this value.")
                if not hasattr(net, 'warnings'):
                    net.warnings = []
                net.warnings.append(
                    f"Generator '{in_data[x].get('userFriendlyName', in_data[x]['name'])}': "
                    f"vm_pu is set to {gen_vm_pu}, which is unusually low. "
                    f"vm_pu is the voltage setpoint in per unit and should be close to 1.0 (e.g. 0.95-1.05). "
                    f"A very low value will cause the generator to try to regulate bus voltage to near zero, "
                    f"leading to unrealistic results."
                )
            
            gen_kw = dict(
                bus=bus_idx,
                name=in_data[x]['name'],
                id=in_data[x]['id'],
                p_mw=safe_float(in_data[x]['p_mw']),
                vm_pu=gen_vm_pu,
                scaling=safe_float(in_data[x].get('scaling'), 1.0),
                in_service=in_service,
                slack=slack,
            )
            # Only pass short-circuit / rating fields when they are physical (diagram often sends zeros).
            # Passing sn_mva=0, cos_phi=0, etc. overrides pandapower defaults and shifts reactive dispatch vs tutorial networks.
            _sn = safe_float(in_data[x]['sn_mva'])
            if _sn > 0:
                gen_kw['sn_mva'] = _sn
            _vn = safe_float(in_data[x]['vn_kv'])
            if _vn > 0:
                gen_kw['vn_kv'] = _vn
            _xd = safe_float(in_data[x]['xdss_pu'])
            if _xd > 0:
                gen_kw['xdss_pu'] = _xd
            _rd = safe_float(in_data[x]['rdss_ohm'])
            # A machine given its reactance may well have zero resistance, and the
            # short-circuit calculation needs the column either way.
            if _rd > 0 or _xd > 0:
                gen_kw['rdss_ohm'] = max(_rd, 0.0)
            _cos = safe_float(in_data[x]['cos_phi'])
            if 0 < _cos <= 1:
                gen_kw['cos_phi'] = _cos
            _pg = safe_float(in_data[x]['pg_percent'])
            if _pg > 0:
                gen_kw['pg_percent'] = _pg
            pst_raw = in_data[x].get('power_station_trafo')
            if pst_raw is not None and str(pst_raw).strip() not in ('', 'None', 'null', 'none'):
                try:
                    pst_f = float(safe_float(pst_raw, 0.0))
                except (TypeError, ValueError):
                    pst_f = 0.0
                if pst_f > 0:
                    gen_kw['power_station_trafo'] = int(pst_f)

            # OPF: pandapower tutorial (opf_basic) uses controllable=True; missing attr must not imply "fixed P".
            gen_kw['controllable'] = _electrisim_boolish(in_data[x].get('controllable'), True)

            # OPF payloads send explicit P limits — pass through when present (otherwise pandapower uses NaN / internal defaults).
            if in_data[x].get('min_p_mw') is not None and str(in_data[x].get('min_p_mw')).lower() not in ('null', 'none', ''):
                gen_kw['min_p_mw'] = safe_float(in_data[x]['min_p_mw'], 0.0)
            if in_data[x].get('max_p_mw') is not None and str(in_data[x].get('max_p_mw')).lower() not in ('null', 'none', ''):
                gen_kw['max_p_mw'] = safe_float(in_data[x]['max_p_mw'], safe_float(in_data[x]['p_mw']) * 1.2)
            # Reactive limits for the OPF, when they form a range: the canvas
            # stores 0 / 0 for a generator nobody gave limits, and passing that
            # on would pin it at zero reactive power instead of leaving it free.
            _q = _electrisim_opf_optional_fields_from_payload(
                in_data[x], float_keys=('min_q_mvar', 'max_q_mvar'))
            if 'min_q_mvar' in _q and 'max_q_mvar' in _q and _q['max_q_mvar'] > _q['min_q_mvar']:
                gen_kw['min_q_mvar'] = _q['min_q_mvar']
                gen_kw['max_q_mvar'] = _q['max_q_mvar']
            pp.create_gen(net, **gen_kw)
            gen_idx = net.gen.index[-1]
            ansi_mt = in_data[x].get('ansi_machine_type')
            if ansi_mt and str(ansi_mt).strip().lower() not in ('', 'none', 'null'):
                if 'ansi_machine_type' not in net.gen.columns:
                    net.gen['ansi_machine_type'] = 'turbo'
                net.gen.at[gen_idx, 'ansi_machine_type'] = str(ansi_mt).strip().lower()
            
            # Store user-friendly name for generator
            gen_name = in_data[x]['name']
            user_friendly_name = in_data[x].get('userFriendlyName', gen_name)
            if not hasattr(net, 'user_friendly_names'):
                net.user_friendly_names = {}
            net.user_friendly_names[gen_name] = user_friendly_name
        
        if _is_static_generator_like(in_data[x]['typ']):
            apply_wind_turbine_p_from_curve(in_data[x])
            bus_idx = Busbars.get(in_data[x]['bus'])
            if bus_idx is None:
                element_name = in_data[x].get('userFriendlyName', in_data[x].get('name', 'Unknown'))
                bus_name = in_data[x]['bus']
                _el_label = 'Wind Turbine' if str(in_data[x]['typ']).startswith('Wind Turbine') else 'Static Generator'
                
                # Check if bus is None (not connected) or references non-existent bus
                if bus_name is None:
                    raise ValueError(
                        f"CONNECTION ERROR: {_el_label} '{element_name}' (ID: {in_data[x].get('id', 'Unknown')}) is NOT CONNECTED to any bus.\n\n"
                        f"SOLUTION: Please ensure that:\n"
                        f"1. You have placed a Bus/Busbar element in your diagram\n"
                        f"2. Draw a connection line from the {_el_label} to a Bus element\n"
                        f"3. Verify the connection line is properly attached at both ends\n\n"
                        f"IMPORTANT: Every electrical component must be connected to at least one Bus element."
                    )
                else:
                    raise ValueError(
                        f"CONNECTION ERROR: {_el_label} '{element_name}' is trying to connect to bus '{bus_name}', "
                        f"but this bus does not exist in your diagram.\n\n"
                        f"SOLUTION: Please ensure that:\n"
                        f"1. You have placed a Bus/Busbar element in your diagram\n"
                        f"2. The {_el_label} is connected to this Bus element with a connection line\n"
                        f"3. All electrical elements must be properly connected to Bus elements"
                    )
           
            # Get in_service parameter (default to True if not specified)
            in_service = True
            if 'in_service' in in_data[x]:
                in_service = bool(in_data[x]['in_service']) if isinstance(in_data[x]['in_service'], bool) else (in_data[x]['in_service'] == 'true' or in_data[x]['in_service'] == True)
            
            _sgen_opf = _electrisim_opf_optional_fields_from_payload(
                in_data[x],
                float_keys=('min_p_mw', 'max_p_mw', 'min_q_mvar', 'max_q_mvar'),
                bool_keys=('controllable',),
            )
            pp.create_sgen(net, bus=bus_idx, name=in_data[x]['name'], id=in_data[x]['id'], p_mw=safe_float(in_data[x]['p_mw']), q_mvar=safe_float(in_data[x]['q_mvar']), sn_mva=safe_float(in_data[x]['sn_mva']), scaling=safe_float(in_data[x].get('scaling'), 1.0), type=in_data[x]['type'],
                           k=safe_float(in_data[x].get('k'), 0.0) if safe_float(in_data[x].get('k'), 0.0) > 0 else 1.1,
                           rx=safe_float(in_data[x]['rx']), generator_type=in_data[x]['generator_type'], lrc_pu=safe_float(in_data[x]['lrc_pu']), max_ik_ka=safe_float(in_data[x]['max_ik_ka']), current_source=in_data[x]['current_source'], kappa = 1.5, in_service=in_service,
                           **_sgen_opf)
            
            # Store user-friendly name for static generator
            sgen_name = in_data[x]['name']
            user_friendly_name = in_data[x].get('userFriendlyName', sgen_name)
            if not hasattr(net, 'user_friendly_names'):
                net.user_friendly_names = {}
            net.user_friendly_names[sgen_name] = user_friendly_name
        
        if (in_data[x]['typ'].startswith("Asymmetric Static Generator")):
            bus_idx = Busbars.get(in_data[x]['bus'])
            if bus_idx is None:
                continue
            # Get in_service parameter (default to True if not specified)
            in_service = True
            if 'in_service' in in_data[x]:
                in_service = bool(in_data[x]['in_service']) if isinstance(in_data[x]['in_service'], bool) else (in_data[x]['in_service'] == 'true' or in_data[x]['in_service'] == True)
            pp.create_asymmetric_sgen(net, bus=bus_idx, name=in_data[x]['name'], id=in_data[x]['id'], p_a_mw=safe_float(in_data[x]['p_a_mw']), p_b_mw=safe_float(in_data[x]['p_b_mw']), p_c_mw=safe_float(in_data[x]['p_c_mw']), q_a_mvar=safe_float(in_data[x]['q_a_mvar']), q_b_mvar=safe_float(in_data[x]['q_b_mvar']), q_c_mvar=safe_float(in_data[x]['q_c_mvar']), sn_mva=safe_float(in_data[x]['sn_mva']), scaling=safe_float(in_data[x].get('scaling'), 1.0), type=in_data[x]['type'], in_service=in_service)   
        #Zero sequence parameters** (Added through std_type For Three phase load flow) :
            #vk0_percent** - zero sequence relative short-circuit voltage
            #vkr0_percent** - real part of zero sequence relative short-circuit voltage
            #mag0_percent** - ratio between magnetizing and short circuit impedance (zero sequence)                                
            #mag0_rx**  - zero sequence magnetizing r/x  ratio
            #si0_hv_partial** - zero sequence short circuit impedance  distribution in hv side
            #vk0_percent=in_data[x]['vk0_percent'], vkr0_percent=in_data[x]['vkr0_percent'], mag0_percent=in_data[x]['mag0_percent'], si0_hv_partial=in_data[x]['si0_hv_partial'],
        _typ = in_data[x].get('typ') or ''
        if (_typ.startswith("Transformer") or _typ.startswith("Two Winding Transformer")) and not _typ.startswith("Three Winding Transformer"):
            # Get values with default fallbacks and proper type conversion
            parallel_value = safe_int(in_data[x].get('parallel', 1), 1)
            vector_group_raw = in_data[x].get('vector_group', None)
            vk0_percent = safe_float(in_data[x].get('vk0_percent', None))
            vkr0_percent = safe_float(in_data[x].get('vkr0_percent', None))
            mag0_percent = safe_float(in_data[x].get('mag0_percent', None))
            mag0_rx = safe_float(in_data[x].get('mag0_rx', None))
            si0_hv_partial = safe_float(in_data[x].get('si0_hv_partial', None))
            
            # Parse vector group to separate base group from phase shift
            vector_group, phase_shift_from_group = parse_vector_group(vector_group_raw)
            
            # Get bus indices with error checking
            hv_bus_name = in_data[x].get('hv_bus')
            lv_bus_name = in_data[x].get('lv_bus')
            if not hv_bus_name or not lv_bus_name:
                label = in_data[x].get('userFriendlyName') or in_data[x].get('name') or in_data[x].get('id')
                raise ValueError(
                    f"Transformer '{label}' is not connected to two busbars "
                    f"(HV '{hv_bus_name or '—'}', LV '{lv_bus_name or '—'}')."
                )
            hv_bus_idx = Busbars.get(hv_bus_name)
            lv_bus_idx = Busbars.get(lv_bus_name)
            
            if hv_bus_idx is None:
                continue
            if lv_bus_idx is None:
                continue
            
            # Prepare parameters dict for transformer creation with proper type conversion
            # CRITICAL: tap_step_percent must be non-zero for tap control to work
            # If it's 0, all tap positions are identical (no voltage change)
            tap_step_value = safe_float(in_data[x].get('tap_step_percent', None))
            if tap_step_value is None or tap_step_value == 0.0:
                # Only use default if tap_max != tap_min (indicating tap control is intended)
                tap_max_val = safe_int(in_data[x].get('tap_max', 0))
                tap_min_val = safe_int(in_data[x].get('tap_min', 0))
                if tap_max_val != tap_min_val and tap_max_val != 0:
                    tap_step_value = 1.5  # Standard default for distribution transformers
                else:
                    tap_step_value = 0.0  # No tap control
            
            # Get tap_changer_type (pandapower 3.0+): "Ratio", "Symmetrical", or "Ideal"
            tap_changer_type = in_data[x].get('tap_changer_type', 'Ratio')
            if tap_changer_type not in ['Ratio', 'Symmetrical', 'Ideal']:
                tap_changer_type = 'Ratio'  # Default value
            
            transformer_params = {
                'hv_bus': hv_bus_idx,
                'lv_bus': lv_bus_idx,
                'name': in_data[x]['name'],
                'id': in_data[x]['id'],
                'sn_mva': safe_float(in_data[x]['sn_mva']),
                'vn_hv_kv': safe_float(in_data[x]['vn_hv_kv']),
                'vn_lv_kv': safe_float(in_data[x]['vn_lv_kv']),
                'vkr_percent': safe_float(in_data[x].get('vkr_percent', 1.0)),
                'vk_percent': safe_float(in_data[x].get('vk_percent', 6.0)),
                'pfe_kw': safe_float(in_data[x].get('pfe_kw', 0.0)),
                'i0_percent': safe_float(in_data[x].get('i0_percent', 0.0)),
                'parallel': float(parallel_value),
                'shift_degree': safe_float(in_data[x].get('shift_degree', 0)) + phase_shift_from_group,
                'tap_side': _normalize_tap_side(in_data[x].get('tap_side', 'hv')),
                'tap_pos': float(safe_int(in_data[x].get('tap_pos', 0))),
                'tap_neutral': float(safe_int(in_data[x].get('tap_neutral', 0))),
                'tap_max': float(safe_int(in_data[x].get('tap_max', 0))),
                'tap_min': float(safe_int(in_data[x].get('tap_min', 0))),
                'tap_step_percent': tap_step_value,
                'tap_step_degree': safe_float(in_data[x].get('tap_step_degree', 0)),
                'tap_changer_type': tap_changer_type  # pandapower 3.0+
            }
            
            # Add optional parameters with proper defaults
            if vector_group is not None and vector_group != 'None' and vector_group != '':
                transformer_params['vector_group'] = vector_group
            else:
                transformer_params['vector_group'] = 'Dyn'  # Default vector group
            
            # For zero sequence parameters, use provided values or default to main sequence values
            if vk0_percent is not None and vk0_percent != 0.0:
                transformer_params['vk0_percent'] = vk0_percent
            else:
                transformer_params['vk0_percent'] = transformer_params['vk_percent']  # Default to main sequence
            
            if vkr0_percent is not None and vkr0_percent != 0.0:
                transformer_params['vkr0_percent'] = vkr0_percent
            else:
                transformer_params['vkr0_percent'] = transformer_params['vkr_percent']  # Default to main sequence
            
            if mag0_percent is not None:
                transformer_params['mag0_percent'] = mag0_percent
            else:
                transformer_params['mag0_percent'] = 0.0  # Default zero sequence magnetizing current
            
            if si0_hv_partial is not None:
                transformer_params['si0_hv_partial'] = si0_hv_partial
            else:
                transformer_params['si0_hv_partial'] = 0.0  # Default zero sequence partial current
            
            if mag0_rx is not None:
                transformer_params['mag0_rx'] = mag0_rx
            else:
                transformer_params['mag0_rx'] = 0.0  # Default zero sequence magnetizing r/x ratio
            
            # Add in_service parameter (default to True if not specified)
            if 'in_service' in in_data[x]:
                transformer_params['in_service'] = bool(in_data[x]['in_service']) if isinstance(in_data[x]['in_service'], bool) else (in_data[x]['in_service'] == 'true' or in_data[x]['in_service'] == True)
            else:
                transformer_params['in_service'] = True

            _mlp_t2 = _electrisim_optional_max_loading_percent(in_data[x].get('max_loading_percent'))
            if _mlp_t2 is not None:
                transformer_params['max_loading_percent'] = _mlp_t2

            trafo_idx = pp.create_transformer_from_parameters(net, **transformer_params)
            for _ngr_col in ('rn_ohm', 'xn_ohm'):
                if _ngr_col not in net.trafo.columns:
                    net.trafo[_ngr_col] = 0.0
                _ngr_val = safe_float(in_data[x].get(_ngr_col))
                if _ngr_val is not None and _ngr_val != 0.0:
                    net.trafo.at[trafo_idx, _ngr_col] = _ngr_val
            TrafoDict[in_data[x]['name']] = trafo_idx
            _ufn_tr = in_data[x].get('userFriendlyName')
            if _ufn_tr not in (None, '') and str(_ufn_tr) != str(in_data[x]['name']):
                TrafoDict[str(_ufn_tr)] = trafo_idx
            
            # Store user-friendly name for transformer
            trafo_name = in_data[x]['name']
            user_friendly_name = in_data[x].get('userFriendlyName', trafo_name)
            if not hasattr(net, 'user_friendly_names'):
                net.user_friendly_names = {}
            net.user_friendly_names[trafo_name] = user_friendly_name
            
            # Discrete Tap Control: collect (trafo_idx, control_side, vm_lower_pu, vm_upper_pu) for controllers
            discrete_tap = in_data[x].get('discrete_tap_control')
            if discrete_tap in (True, 'true', 'True', '1'):
                vm_lo = safe_float_local(in_data[x].get('vm_lower_pu'), 0.99)
                vm_hi = safe_float_local(in_data[x].get('vm_upper_pu'), 1.01)
                vm_lo = 0.99 if vm_lo is None else float(vm_lo)
                vm_hi = 1.01 if vm_hi is None else float(vm_hi)
                control_side = in_data[x].get('control_side', 'lv')  # Default to 'lv' if not specified
                trafo_idx = int(net.trafo.index[-1])
                if not hasattr(net, 'trafo_discrete_tap_controllers'):
                    net.trafo_discrete_tap_controllers = []
                net.trafo_discrete_tap_controllers.append((trafo_idx, control_side, vm_lo, vm_hi))
       
        if (in_data[x]['typ'].startswith("Three Winding Transformer")):  
            # Parse vector group to separate base group from phase shift
            vector_group_raw = in_data[x].get('vector_group', None)
            vector_group, phase_shift_from_group = parse_vector_group(vector_group_raw)
            
            # Get bus indices with error checking
            hv_bus_name = in_data[x]['hv_bus']
            mv_bus_name = in_data[x]['mv_bus']
            lv_bus_name = in_data[x]['lv_bus']
            hv_bus_idx = Busbars.get(hv_bus_name)
            mv_bus_idx = Busbars.get(mv_bus_name)
            lv_bus_idx = Busbars.get(lv_bus_name)
            
            if hv_bus_idx is None:
                continue
            if mv_bus_idx is None:
                continue
            if lv_bus_idx is None:
                continue
            
            # Get tap_changer_type (pandapower 3.0+): "Ratio", "Symmetrical", or "Ideal"
            tap_changer_type_3w = in_data[x].get('tap_changer_type', 'Ratio')
            if tap_changer_type_3w not in ['Ratio', 'Symmetrical', 'Ideal']:
                tap_changer_type_3w = 'Ratio'  # Default value
            
            # Prepare optional parameters - only include if they are not None
            transformer_params = {
                'hv_bus': hv_bus_idx,
                'mv_bus': mv_bus_idx,
                'lv_bus': lv_bus_idx,
                'name': in_data[x]['name'],
                'id': in_data[x]['id'],
                'sn_hv_mva': safe_float(in_data[x]['sn_hv_mva']),
                'sn_mv_mva': safe_float(in_data[x]['sn_mv_mva']),
                'sn_lv_mva': safe_float(in_data[x]['sn_lv_mva']),
                'vn_hv_kv': safe_float(in_data[x]['vn_hv_kv']),
                'vn_mv_kv': safe_float(in_data[x]['vn_mv_kv']),
                'vn_lv_kv': safe_float(in_data[x]['vn_lv_kv']),
                'vk_hv_percent': safe_float(in_data[x]['vk_hv_percent']),
                'vk_mv_percent': safe_float(in_data[x]['vk_mv_percent']),
                'vk_lv_percent': safe_float(in_data[x]['vk_lv_percent']),
                'vkr_hv_percent': safe_float(in_data[x]['vkr_hv_percent']),
                'vkr_mv_percent': safe_float(in_data[x]['vkr_mv_percent']),
                'vkr_lv_percent': safe_float(in_data[x]['vkr_lv_percent']),
                'pfe_kw': safe_float(in_data[x]['pfe_kw']),
                'i0_percent': safe_float(in_data[x]['i0_percent']),
                'shift_mv_degree': safe_float(in_data[x]['shift_mv_degree']) + phase_shift_from_group,
                'shift_lv_degree': safe_float(in_data[x]['shift_lv_degree']) + phase_shift_from_group,
                'tap_step_percent': safe_float(in_data[x]['tap_step_percent']),
                'tap_step_degree': safe_float(in_data[x].get('tap_step_degree', 0)),
                'tap_side': _normalize_tap_side(in_data[x]['tap_side']),
                'tap_neutral': float(safe_int(in_data[x].get('tap_neutral', 0))),
                'tap_min': float(safe_int(in_data[x]['tap_min'])),
                'tap_max': float(safe_int(in_data[x]['tap_max'])),
                'tap_pos': float(safe_int(in_data[x]['tap_pos'])),
                'tap_changer_type': tap_changer_type_3w  # pandapower 3.0+
            }
            
            # Add optional parameters only if they are not None
            optional_params = ['vector_group', 'vk0_hv_percent', 'vk0_mv_percent', 'vk0_lv_percent', 
                             'vkr0_hv_percent', 'vkr0_mv_percent', 'vkr0_lv_percent']
            
            for param in optional_params:
                value = in_data[x].get(param)
                if value is not None and value not in ('None', '', 'null'):
                    if param == 'vector_group':
                        # pandapower names three-winding groups without clock
                        # numbers ("YNynd"); parse_vector_group only strips a
                        # trailing one, so "YNyn0d5" reached it unrecognised.
                        transformer_params[param] = re.sub(r'\d+', '', str(value))
                    else:
                        transformer_params[param] = safe_float(value)  # Convert to float
            # A zero zero-sequence voltage is the canvas placeholder, and a
            # zero impedance; fall back to the positive-sequence value, the rule
            # pandapower itself applies to two-winding transformers.
            for winding in ('hv', 'mv', 'lv'):
                for kind in ('vk', 'vkr'):
                    zero_key = f'{kind}0_{winding}_percent'
                    if not (safe_float(transformer_params.get(zero_key), 0.0) > 0):
                        transformer_params[zero_key] = transformer_params[f'{kind}_{winding}_percent']
            
            # Add in_service parameter (default to True if not specified)
            if 'in_service' in in_data[x]:
                transformer_params['in_service'] = bool(in_data[x]['in_service']) if isinstance(in_data[x]['in_service'], bool) else (in_data[x]['in_service'] == 'true' or in_data[x]['in_service'] == True)
            else:
                transformer_params['in_service'] = True

            _mlp_t3 = _electrisim_optional_max_loading_percent(in_data[x].get('max_loading_percent'))
            if _mlp_t3 is not None:
                transformer_params['max_loading_percent'] = _mlp_t3
            
            trafo3w_id = transformer_params.pop('id', None)
            trafo3w_idx = pp.create_transformer3w_from_parameters(net, **transformer_params)
            if trafo3w_id is not None:
                net.trafo3w.at[trafo3w_idx, 'id'] = trafo3w_id
            Trafo3wDict[in_data[x]['name']] = trafo3w_idx
            _ufn_t3 = in_data[x].get('userFriendlyName')
            if _ufn_t3 not in (None, '') and str(_ufn_t3) != str(in_data[x]['name']):
                Trafo3wDict[str(_ufn_t3)] = trafo3w_idx
            
            # Store user-friendly name for three-winding transformer
            trafo3w_name = in_data[x]['name']
            user_friendly_name = in_data[x].get('userFriendlyName', trafo3w_name)
            if not hasattr(net, 'user_friendly_names'):
                net.user_friendly_names = {}
            net.user_friendly_names[trafo3w_name] = user_friendly_name

            # DiscreteTapControl for 3-winding transformers (pandapower: element="trafo3w", side hv|mv|lv)
            discrete_tap_3w = in_data[x].get('discrete_tap_control')
            if discrete_tap_3w in (True, 'true', 'True', '1'):
                vm_lo = safe_float_local(in_data[x].get('vm_lower_pu'), 0.99)
                vm_hi = safe_float_local(in_data[x].get('vm_upper_pu'), 1.01)
                vm_lo = 0.99 if vm_lo is None else float(vm_lo)
                vm_hi = 1.01 if vm_hi is None else float(vm_hi)
                control_side_3w = in_data[x].get('control_side', 'lv')
                if control_side_3w not in ('hv', 'mv', 'lv'):
                    control_side_3w = 'lv'
                t3_idx = int(trafo3w_idx)
                if not hasattr(net, 'trafo3w_discrete_tap_controllers'):
                    net.trafo3w_discrete_tap_controllers = []
                net.trafo3w_discrete_tap_controllers.append((t3_idx, control_side_3w, vm_lo, vm_hi))
        
        if (in_data[x]['typ'].startswith("Shunt Reactor")):
            bus_idx = Busbars.get(in_data[x]['bus'])
            if bus_idx is None:
                continue
            # Get in_service parameter (default to True if not specified)
            in_service = True
            if 'in_service' in in_data[x]:
                in_service = bool(in_data[x]['in_service']) if isinstance(in_data[x]['in_service'], bool) else (in_data[x]['in_service'] == 'true' or in_data[x]['in_service'] == True)
            p_mw_use = safe_float(in_data[x]['p_mw'])
            q_mvar_use = safe_float(in_data[x]['q_mvar'])
            sdt = in_data[x].get('step_dependency_table')
            use_characteristic = sdt in (True, 'true', 'True', '1')
            raw_char_json = in_data[x].get('shunt_characteristic_table_json')
            characteristic_rows = _electrisim_parse_shunt_characteristic_table_json(raw_char_json)
            lf_rows_preview = _electrisim_parse_line_flow_step_table_json(in_data[x].get('line_flow_step_table_json'))
            st_raw = safe_float(in_data[x].get('step', 1))
            step_electrisim = st_raw if st_raw is not None else in_data[x].get('step', 1)
            max_step_electrisim = safe_float(in_data[x].get('max_step', 1)) or 1
            zero_based_steps = _electrisim_shunt_uses_zero_based(step_electrisim, characteristic_rows, lf_rows_preview)
            id_characteristic_table = None
            step_dependency_table = False
            if use_characteristic and characteristic_rows:
                char_rows_pp = _electrisim_shift_shunt_characteristic_rows_pp(characteristic_rows, zero_based_steps)
                id_characteristic_table = _electrisim_append_shunt_characteristic_table(net, char_rows_pp)
                step_dependency_table = id_characteristic_table is not None
                if step_dependency_table:
                    p_mw_use, q_mvar_use = _electrisim_shunt_nominals_for_step(
                        characteristic_rows,
                        step_electrisim,
                        p_mw_use,
                        q_mvar_use,
                    )
            elif use_characteristic and not characteristic_rows:
                print(f"Warning: Shunt reactor '{in_data[x].get('name', '?')}': step_dependency_table is enabled "
                      f"but shunt_characteristic_table_json is missing or invalid; using nominal p_mw/q_mvar only.")

            step_pp = _electrisim_shunt_step_to_pp(step_electrisim, zero_based_steps)
            max_step_pp = _electrisim_shunt_max_step_to_pp(max_step_electrisim, zero_based_steps)

            shunt_idx = pp.create_shunt(net, typ="shuntreactor", bus=bus_idx, name=in_data[x]['name'], id=in_data[x]['id'], p_mw=p_mw_use, q_mvar=q_mvar_use, vn_kv=safe_float(in_data[x]['vn_kv']), step=float(step_pp), max_step=float(max_step_pp), in_service=in_service, step_dependency_table=step_dependency_table, id_characteristic_table=id_characteristic_table if step_dependency_table else None)
            if zero_based_steps:
                if not hasattr(net, "_electrisim_shunt_zero_based"):
                    net._electrisim_shunt_zero_based = {}
                net._electrisim_shunt_zero_based[int(shunt_idx)] = True
            # DiscreteShuntController (pandapower): step shunt to regulate vm at shunt bus toward vm_set_pu
            dsc = in_data[x].get('discrete_shunt_control')
            if dsc in (True, 'true', 'True', '1'):
                vm_set = safe_float_local(in_data[x].get('vm_set_pu'), 1.0)
                vm_set = 1.0 if vm_set is None else float(vm_set)
                try:
                    incr = int(safe_int(in_data[x].get('shunt_control_increment', 1), 1))
                except Exception:
                    incr = 1
                if incr < 1:
                    incr = 1
                tol = safe_float_local(in_data[x].get('shunt_control_tol'), 1e-3)
                tol = 1e-3 if tol is None else float(tol)
                reset_init = in_data[x].get('shunt_reset_at_init', False) in (True, 'true', 'True', '1')
                if not hasattr(net, 'shunt_discrete_controllers'):
                    net.shunt_discrete_controllers = []
                net.shunt_discrete_controllers.append({
                    'shunt_index': int(shunt_idx),
                    'vm_set_pu': vm_set,
                    'bus_index': None,
                    'increment': incr,
                    'tol': tol,
                    'reset_at_init': bool(reset_init),
                })
            # Active power on reference line → shunt step (CharacteristicControl); exclusive with DiscreteShuntController
            dsc_on = dsc in (True, 'true', 'True', '1')
            lfc = in_data[x].get('line_flow_step_control')
            if lfc in (True, 'true', 'True', '1') and not dsc_on:
                ref_line = in_data[x].get('line_flow_reference_line_id')
                lf_rows = _electrisim_parse_line_flow_step_table_json(in_data[x].get('line_flow_step_table_json'))
                pref = in_data[x].get('line_flow_p_reference') or 'p_from_mw'
                if pref not in ('p_from_mw', 'p_to_mw'):
                    pref = 'p_from_mw'
                lf_abs = in_data[x].get('line_flow_p_use_abs', True) in (True, 'true', 'True', '1')
                if not hasattr(net, '_electrisim_pending_line_flow_shunts'):
                    net._electrisim_pending_line_flow_shunts = []
                net._electrisim_pending_line_flow_shunts.append({
                    'shunt_index': int(shunt_idx),
                    'ref_line': ref_line,
                    'bands': lf_rows,
                    'p_col': pref,
                    'use_abs': lf_abs,
                    'name': in_data[x].get('name', '?'),
                    'zero_based_steps': bool(zero_based_steps),
                })
        
        if (in_data[x]['typ'].startswith("Capacitor")):
            bus_idx = Busbars.get(in_data[x]['bus'])
            if bus_idx is None:
                continue
            # Get in_service parameter (default to True if not specified)
            in_service = True
            if 'in_service' in in_data[x]:
                in_service = bool(in_data[x]['in_service']) if isinstance(in_data[x]['in_service'], bool) else (in_data[x]['in_service'] == 'true' or in_data[x]['in_service'] == True)
            pp.create_shunt_as_capacitor(net, typ="capacitor", bus=bus_idx, name=in_data[x]['name'], id=in_data[x]['id'], q_mvar=safe_float(in_data[x]['q_mvar']), loss_factor=safe_float(in_data[x]['loss_factor']), vn_kv=safe_float(in_data[x]['vn_kv']), step=float(safe_float(in_data[x].get('step', 1)) or 1), max_step=float(safe_float(in_data[x].get('max_step', 1)) or 1), in_service=in_service)        
        
        if (in_data[x]['typ'].startswith("Load") and not in_data[x]['typ'].startswith("Load DC")):
            bus_idx = Busbars.get(in_data[x]['bus'])
            if bus_idx is None:
                element_name = in_data[x].get('userFriendlyName', in_data[x].get('name', 'Unknown'))
                bus_name = in_data[x]['bus']
                
                # Check if bus is None (not connected) or references non-existent bus
                if bus_name is None:
                    raise ValueError(
                        f"CONNECTION ERROR: Load '{element_name}' (ID: {in_data[x].get('id', 'Unknown')}) is NOT CONNECTED to any bus.\n\n"
                        f"SOLUTION: Please ensure that:\n"
                        f"1. You have placed a Bus/Busbar element in your diagram\n"
                        f"2. Draw a connection line from the Load to a Bus element\n"
                        f"3. Verify the connection line is properly attached at both ends\n\n"
                        f"IMPORTANT: Every electrical component must be connected to at least one Bus element."
                    )
                else:
                    raise ValueError(
                        f"CONNECTION ERROR: Load '{element_name}' is trying to connect to bus '{bus_name}', "
                        f"but this bus does not exist in your diagram.\n\n"
                        f"SOLUTION: Please ensure that:\n"
                        f"1. You have placed a Bus/Busbar element in your diagram\n"
                        f"2. The Load is connected to this Bus element with a connection line\n"
                        f"3. All electrical elements must be properly connected to Bus elements"
                    )
          
            # Get in_service parameter (default to True if not specified)
            in_service = True
            if 'in_service' in in_data[x]:
                in_service = bool(in_data[x]['in_service']) if isinstance(in_data[x]['in_service'], bool) else (in_data[x]['in_service'] == 'true' or in_data[x]['in_service'] == True)
            
            _load_opf = _electrisim_opf_optional_fields_from_payload(
                in_data[x],
                float_keys=('min_p_mw', 'max_p_mw', 'min_q_mvar', 'max_q_mvar'),
                bool_keys=('controllable',),
            )
            load_ctrl = _electrisim_boolish(_load_opf.get('controllable', in_data[x].get('controllable')), False)
            if not load_ctrl:
                for _lk in ('min_p_mw', 'max_p_mw', 'min_q_mvar', 'max_q_mvar'):
                    _load_opf.pop(_lk, None)
            pp.create_load(net, bus=bus_idx, name=in_data[x]['name'], id=in_data[x]['id'], p_mw=safe_float(in_data[x]['p_mw']),q_mvar=safe_float(in_data[x]['q_mvar']),const_z_percent=safe_float(in_data[x]['const_z_percent']),const_i_percent=safe_float(in_data[x]['const_i_percent']), sn_mva=safe_float(in_data[x]['sn_mva']),scaling=safe_float(in_data[x].get('scaling'), 1.0),type=in_data[x]['type'], in_service=in_service,
                           **_load_opf)
            
            # Store user-friendly name for load
            load_name = in_data[x]['name']
            user_friendly_name = in_data[x].get('userFriendlyName', load_name)
            if not hasattr(net, 'user_friendly_names'):
                net.user_friendly_names = {}
            net.user_friendly_names[load_name] = user_friendly_name
      
        if (in_data[x]['typ'].startswith("Asymmetric Load")):
            bus_idx = Busbars.get(in_data[x]['bus'])
            if bus_idx is None:
                continue
            # Get in_service parameter (default to True if not specified)
            in_service = True
            if 'in_service' in in_data[x]:
                in_service = bool(in_data[x]['in_service']) if isinstance(in_data[x]['in_service'], bool) else (in_data[x]['in_service'] == 'true' or in_data[x]['in_service'] == True)
            pp.create_asymmetric_load(net, bus=bus_idx, name=in_data[x]['name'], id=in_data[x]['id'], p_a_mw=in_data[x]['p_a_mw'],p_b_mw=in_data[x]['p_b_mw'],p_c_mw=in_data[x]['p_c_mw'],q_a_mvar=in_data[x]['q_a_mvar'], q_b_mvar=in_data[x]['q_b_mvar'], q_c_mvar=in_data[x]['q_c_mvar'], sn_mva=in_data[x]['sn_mva'], scaling=in_data[x]['scaling'],type=in_data[x]['type'], in_service=in_service)         
   
        if (in_data[x]['typ'].startswith("Impedance")):
            from_bus_idx = Busbars.get(in_data[x]['busFrom'])
            to_bus_idx = Busbars.get(in_data[x]['busTo'])
            if from_bus_idx is None:
                continue
            if to_bus_idx is None:
                continue
            # Get in_service parameter (default to True if not specified)
            in_service = True
            if 'in_service' in in_data[x]:
                in_service = bool(in_data[x]['in_service']) if isinstance(in_data[x]['in_service'], bool) else (in_data[x]['in_service'] == 'true' or in_data[x]['in_service'] == True)
            pp.create_impedance(net, from_bus=from_bus_idx, to_bus=to_bus_idx, name=in_data[x]['name'], id=in_data[x]['id'], rft_pu=in_data[x]['rft_pu'],xft_pu=in_data[x]['xft_pu'],sn_mva=in_data[x]['sn_mva'], in_service=in_service)         
         
        if (in_data[x]['typ'].startswith("Ward")):
            bus_idx = Busbars.get(in_data[x]['bus'])
            if bus_idx is None:
                continue
            # Get in_service parameter (default to True if not specified)
            in_service = True
            if 'in_service' in in_data[x]:
                in_service = bool(in_data[x]['in_service']) if isinstance(in_data[x]['in_service'], bool) else (in_data[x]['in_service'] == 'true' or in_data[x]['in_service'] == True)
            pp.create_ward(net, bus=bus_idx, name=in_data[x]['name'], id=in_data[x]['id'], ps_mw=in_data[x]['ps_mw'],qs_mvar=in_data[x]['qs_mvar'], pz_mw=in_data[x]['pz_mw'], qz_mvar=in_data[x]['qz_mvar'], in_service=in_service)         
   
        if (in_data[x]['typ'].startswith("Extended Ward")):
            bus_idx = Busbars.get(in_data[x]['bus'])
            if bus_idx is None:
                continue
            # Get in_service parameter (default to True if not specified)
            in_service = True
            if 'in_service' in in_data[x]:
                in_service = bool(in_data[x]['in_service']) if isinstance(in_data[x]['in_service'], bool) else (in_data[x]['in_service'] == 'true' or in_data[x]['in_service'] == True)
            pp.create_xward(net, bus=bus_idx, name=in_data[x]['name'], id=in_data[x]['id'], ps_mw=in_data[x]['ps_mw'], qs_mvar=in_data[x]['qs_mvar'], pz_mw=in_data[x]['pz_mw'], qz_mvar=in_data[x]['qz_mvar'], r_ohm =in_data[x]['r_ohm'], x_ohm=in_data[x]['x_ohm'],vm_pu=in_data[x]['vm_pu'], in_service=in_service)         
   
        if (in_data[x]['typ'].startswith("Motor")):
            bus_idx = Busbars.get(in_data[x]['bus'])
            if bus_idx is None:
                continue
            # Get in_service parameter (default to True if not specified)
            in_service = True
            if 'in_service' in in_data[x]:
                in_service = bool(in_data[x]['in_service']) if isinstance(in_data[x]['in_service'], bool) else (in_data[x]['in_service'] == 'true' or in_data[x]['in_service'] == True)
            
            # The diagram's attribute is lrc_pu; older payloads spelt it Irc_pu.
            lrc_pu_value = in_data[x].get('lrc_pu') or in_data[x].get('Irc_pu')
            if lrc_pu_value is None or lrc_pu_value == 'None' or lrc_pu_value == '':
                lrc_pu_value = None
            else:
                lrc_pu_value = safe_float_local(lrc_pu_value, None)
            
            pp.create_motor(net, bus=bus_idx, name=in_data[x]['name'], id=in_data[x]['id'], 
                            pn_mech_mw=safe_float_local(in_data[x].get('pn_mech_mw'), 0.0),
                            cos_phi=safe_float_local(in_data[x].get('cos_phi'), 0.85),
                            # The short circuit needs the rated power factor; it
                            # was never passed, so any motor failed every max case.
                            cos_phi_n=safe_float_local(in_data[x].get('cos_phi_n'), None)
                            or safe_float_local(in_data[x].get('cos_phi'), 0.85),
                            efficiency_n_percent=safe_float_local(in_data[x].get('efficiency_n_percent'), 90.0),
                            lrc_pu=lrc_pu_value,
                            rx=safe_float_local(in_data[x].get('rx'), 0.0),
                            vn_kv=safe_float_local(in_data[x].get('vn_kv'), 0.4),
                            efficiency_percent=safe_float_local(in_data[x].get('efficiency_percent'), 90.0),
                            loading_percent=safe_float_local(in_data[x].get('loading_percent'), 100.0),
                            scaling=safe_float_local(in_data[x].get('scaling'), 1.0),
                            in_service=in_service)
            if in_data[x].get('userFriendlyName'):
                if not hasattr(net, 'user_friendly_names'):
                    net.user_friendly_names = {}
                net.user_friendly_names[in_data[x]['name']] = in_data[x]['userFriendlyName']

        if (in_data[x]['typ'].startswith("SVC")):
            bus_idx = Busbars.get(in_data[x]['bus'])
            if bus_idx is None:
                continue
            # Get in_service parameter (default to True if not specified)
            in_service = True
            if 'in_service' in in_data[x]:
                in_service = bool(in_data[x]['in_service']) if isinstance(in_data[x]['in_service'], bool) else (in_data[x]['in_service'] == 'true' or in_data[x]['in_service'] == True)
            pp.create_svc(net, bus=bus_idx, name=in_data[x]['name'], id=in_data[x]['id'], x_l_ohm=in_data[x]['x_l_ohm'], x_cvar_ohm=in_data[x]['x_cvar_ohm'], set_vm_pu=in_data[x]['set_vm_pu'], thyristor_firing_angle_degree=in_data[x]['thyristor_firing_angle_degree'], controllable=in_data[x]['controllable'], min_angle_degree=in_data[x]['min_angle_degree'], max_angle_degree=in_data[x]['max_angle_degree'], in_service=in_service)
         
        if (in_data[x]['typ'].startswith("TCSC")):
            from_bus_idx = Busbars.get(in_data[x]['busFrom'])
            to_bus_idx = Busbars.get(in_data[x]['busTo'])
            if from_bus_idx is None:
                continue
            if to_bus_idx is None:
                continue
            # Get in_service parameter (default to True if not specified)
            in_service = True
            if 'in_service' in in_data[x]:
                in_service = bool(in_data[x]['in_service']) if isinstance(in_data[x]['in_service'], bool) else (in_data[x]['in_service'] == 'true' or in_data[x]['in_service'] == True)
            pp.create_tcsc(net, from_bus=from_bus_idx, to_bus=to_bus_idx, name=in_data[x]['name'], id=in_data[x]['id'], x_l_ohm=in_data[x]['x_l_ohm'], x_cvar_ohm=in_data[x]['x_cvar_ohm'], set_p_to_mw=in_data[x]['set_p_to_mw'], thyristor_firing_angle_degree=in_data[x]['thyristor_firing_angle_degree'], controllable=in_data[x]['controllable'], min_angle_degree=in_data[x]['min_angle_degree'], max_angle_degree=in_data[x]['max_angle_degree'], in_service=in_service)
                   
        if (in_data[x]['typ'].startswith("SSC")):
            bus_idx = Busbars.get(in_data[x]['bus'])
            if bus_idx is None:
                continue
            # Get in_service parameter (default to True if not specified)
            in_service = True
            if 'in_service' in in_data[x]:
                in_service = bool(in_data[x]['in_service']) if isinstance(in_data[x]['in_service'], bool) else (in_data[x]['in_service'] == 'true' or in_data[x]['in_service'] == True)
            pp.create_ssc(net, bus=bus_idx, name=in_data[x]['name'], id=in_data[x]['id'], r_ohm=in_data[x]['r_ohm'], x_ohm=in_data[x]['x_ohm'], set_vm_pu=in_data[x]['set_vm_pu'], vm_internal_pu=in_data[x]['vm_internal_pu'], va_internal_degree=in_data[x]['va_internal_degree'], controllable=in_data[x]['controllable'], in_service=in_service)
        

        if (in_data[x]['typ'].startswith("Storage")):
            bus_idx = Busbars.get(in_data[x]['bus'])
            if bus_idx is None:
                continue
            # Get in_service parameter (default to True if not specified)
            in_service = True
            if 'in_service' in in_data[x]:
                in_service = bool(in_data[x]['in_service']) if isinstance(in_data[x]['in_service'], bool) else (in_data[x]['in_service'] == 'true' or in_data[x]['in_service'] == True)
            # Imported cases (for example CIGRE batteries) leave energy and state of
            # charge empty. The diagram then sends the text "null", which float() rejects.
            p_mw = safe_float(in_data[x].get('p_mw'), 0.0)
            q_mvar = safe_float(in_data[x].get('q_mvar'), 0.0)
            sn_mva = safe_float(in_data[x].get('sn_mva'), 0.0)
            max_e_mwh = safe_float(in_data[x].get('max_e_mwh'), 0.0)
            min_e_mwh = safe_float(in_data[x].get('min_e_mwh'), 0.0)
            soc_percent = safe_float(in_data[x].get('soc_percent'), 50.0)
            scaling = safe_float(in_data[x].get('scaling'), 1.0)
            if max_e_mwh <= 0:
                max_e_mwh = sn_mva if sn_mva > 0 else max(abs(p_mw), 0.001)
            # No MVA rating stays 0 = unrated, as every reader of sn_mva
            # treats it. It was max(|P|, MWh): the energy capacity posing as
            # a power rating, which capped BESS sizing at 2 MVA for a 2 MWh
            # battery.
            if sn_mva <= 0:
                sn_mva = 0.0
            storage_type = str(in_data[x].get('type', ''))
            # OPF parameters (optional)
            controllable_raw = in_data[x].get('controllable', False)
            controllable = bool(controllable_raw) if isinstance(controllable_raw, bool) else (str(controllable_raw).lower() in ('true', '1'))
            max_p_mw_raw = in_data[x].get('max_p_mw')
            min_p_mw_raw = in_data[x].get('min_p_mw')
            max_q_mvar_raw = in_data[x].get('max_q_mvar')
            min_q_mvar_raw = in_data[x].get('min_q_mvar')
            storage_kwargs = dict(
                p_mw=p_mw, q_mvar=q_mvar, sn_mva=sn_mva,
                max_e_mwh=max_e_mwh, min_e_mwh=min_e_mwh,
                soc_percent=soc_percent, scaling=scaling,
                type=storage_type, in_service=in_service,
                controllable=controllable
            )
            if max_p_mw_raw is not None:
                try:
                    storage_kwargs['max_p_mw'] = float(max_p_mw_raw)
                except (TypeError, ValueError):
                    pass
            if min_p_mw_raw is not None:
                try:
                    storage_kwargs['min_p_mw'] = float(min_p_mw_raw)
                except (TypeError, ValueError):
                    pass
            if max_q_mvar_raw is not None:
                try:
                    storage_kwargs['max_q_mvar'] = float(max_q_mvar_raw)
                except (TypeError, ValueError):
                    pass
            if min_q_mvar_raw is not None:
                try:
                    storage_kwargs['min_q_mvar'] = float(min_q_mvar_raw)
                except (TypeError, ValueError):
                    pass
            stor_idx = pp.create_storage(net, bus=bus_idx, name=in_data[x]['name'], id=in_data[x]['id'], **storage_kwargs)
            for _sc_col, _sc_key, _sc_default in (
                ('max_ik_ka', 'max_ik_ka', 0.0),
                ('rx', 'rx', 0.1),
                ('current_source', 'current_source', False),
            ):
                if _sc_col not in net.storage.columns:
                    net.storage[_sc_col] = _sc_default
                raw_sc = in_data[x].get(_sc_key)
                if raw_sc is None or str(raw_sc).strip().lower() in ('', 'none', 'null', 'nan'):
                    continue
                if _sc_col == 'current_source':
                    net.storage.at[stor_idx, _sc_col] = bool(raw_sc) if isinstance(raw_sc, bool) else str(raw_sc).lower() in ('true', '1', 'yes')
                else:
                    try:
                        net.storage.at[stor_idx, _sc_col] = float(raw_sc)
                    except (TypeError, ValueError):
                        pass
            stor_nm = in_data[x]['name']
            stor_cell_id = in_data[x]['id']
            try:
                stor_mask = net.storage['id'] == stor_cell_id
                if stor_mask.any():
                    stor_idx = net.storage.index[stor_mask][0]
                    p_res, q_res, q_min_lim, q_max_lim = resolve_storage_pq(in_data[x])
                    net.storage.at[stor_idx, 'p_mw'] = p_res
                    net.storage.at[stor_idx, 'q_mvar'] = q_res
                    if q_min_lim is not None:
                        net.storage.at[stor_idx, 'min_q_mvar'] = q_min_lim
                    if q_max_lim is not None:
                        net.storage.at[stor_idx, 'max_q_mvar'] = q_max_lim
            except Exception as stor_q_err:
                print(f"Warning: Storage '{stor_nm}' Q capability apply failed: {stor_q_err}")
            uf_storage = in_data[x].get('userFriendlyName', stor_nm)
            if not hasattr(net, 'user_friendly_names'):
                net.user_friendly_names = {}
            net.user_friendly_names[stor_nm] = uf_storage
   
        if (in_data[x]['typ'].startswith("Load DC")):
            bus_idx = _electrisim_dc_bus(net, in_data[x].get('bus'))
            if bus_idx is None:
                _electrisim_warn(net, f"Load DC '{in_data[x].get('userFriendlyName', in_data[x].get('name'))}' "
                                      "is not connected to a DC bus, so it is left out.")
                continue
            # Get in_service parameter (default to True if not specified)
            in_service = True
            if 'in_service' in in_data[x]:
                in_service = bool(in_data[x]['in_service']) if isinstance(in_data[x]['in_service'], bool) else (in_data[x]['in_service'] == 'true' or in_data[x]['in_service'] == True)
            p_dc = safe_float(in_data[x].get('p_dc_mw', in_data[x].get('p_mw', 0.0)))
            dc_extra = {}
            if in_data[x].get('id') is not None:
                dc_extra['id'] = in_data[x]['id']
            # An explicit index: pandapower 3.3 takes create_load_dc's next index
            # from the DC source table, so each DC load overwrote the last.
            dc_extra['index'] = int(net.load_dc.index.max()) + 1 if len(net.load_dc) else 0
            # Its model: shares of constant power, current and resistance, and
            # the voltage below which the constant-power part draws constant
            # current; the input filter is for the EMT study.
            share_p, share_i, share_r = _electrisim_dc_load_shares(in_data[x])
            dc_extra.update(electrisim_p_rated_mw=p_dc, electrisim_share_p=share_p, electrisim_share_i=share_i,
                            electrisim_share_r=share_r,
                            electrisim_v_min_pu=safe_float(in_data[x].get('v_min_pu'), 0.0),
                            filter_l_mh=safe_float(in_data[x].get('filter_l_mh'), 0.0),
                            filter_c_uf=safe_float(in_data[x].get('filter_c_uf'), 0.0))
            try:
                pp.create_load_dc(net, bus_dc=bus_idx, name=in_data[x]['name'],
                                  p_dc_mw=p_dc, in_service=in_service, **dc_extra)
            except TypeError:
                try:
                    pp.create_load_dc(net, bus=bus_idx, name=in_data[x]['name'],
                                      p_mw=p_dc, in_service=in_service, **dc_extra)
                except Exception as load_dc_err:
                    print(f"Warning: Load DC '{in_data[x].get('name')}' not created: {load_dc_err}")
                    continue
            except Exception as load_dc_err:
                print(f"Warning: Load DC '{in_data[x].get('name')}' not created: {load_dc_err}")
                continue
            
            # Store user-friendly name for load DC
            load_dc_name = in_data[x]['name']
            user_friendly_name = in_data[x].get('userFriendlyName', load_dc_name)
            if not hasattr(net, 'user_friendly_names'):
                net.user_friendly_names = {}
            net.user_friendly_names[load_dc_name] = user_friendly_name
        
        if (in_data[x]['typ'].startswith("Solid-State Transformer")):
            # Built once every element exists: see _electrisim_build_ssts.
            if not hasattr(net, '_electrisim_pending_sst'):
                net._electrisim_pending_sst = []
            net._electrisim_pending_sst.append(in_data[x])
            continue

        if str(in_data[x]['typ']).startswith("PCS"):
            # A power conversion system: built once its source and the buses exist, see _electrisim_build_pcs.
            if not hasattr(net, '_electrisim_pending_pcs'):
                net._electrisim_pending_pcs = []
            net._electrisim_pending_pcs.append(in_data[x])
            continue

        if der_electrisim.kind_of(in_data[x]['typ']):
            # A battery, supercapacitor, flywheel, SOFC system or PV array: built once every
            # converter exists, see _electrisim_build_ders.
            if not hasattr(net, '_electrisim_pending_der'):
                net._electrisim_pending_der = []
            net._electrisim_pending_der.append(in_data[x])
            continue

        if (in_data[x]['typ'].startswith("DC/DC Converter")):
            # Built once every DC bus exists: see _electrisim_build_dc_dc_converters.
            if not hasattr(net, '_electrisim_pending_dc_dc'):
                net._electrisim_pending_dc_dc = []
            net._electrisim_pending_dc_dc.append(in_data[x])
            continue

        if (in_data[x]['typ'].startswith("DC Breaker")):
            # Applied once every element exists: see _electrisim_apply_dc_breakers.
            if not hasattr(net, '_electrisim_pending_dc_breakers'):
                net._electrisim_pending_dc_breakers = []
            net._electrisim_pending_dc_breakers.append(in_data[x])
            continue

        if (in_data[x]['typ'].startswith("DC Capacitor")):
            cap_name = in_data[x].get('name')
            cap_label = in_data[x].get('userFriendlyName', cap_name)
            bus_idx = _electrisim_dc_bus(net, in_data[x].get('bus'))
            if bus_idx is None:
                _electrisim_warn(net, f"DC Capacitor '{cap_label}' is not connected to a DC bus, so it is left out.")
                continue
            # A DC-link capacitor draws no current in steady state: it is kept
            # for the DC fault and EMT studies, and its stored energy reported.
            if not hasattr(net, 'electrisim_dc_capacitors'):
                net.electrisim_dc_capacitors = []
            net.electrisim_dc_capacitors.append({
                'name': cap_name, 'id': in_data[x].get('id', ''), 'bus_dc': int(bus_idx),
                'c_mf': safe_float(in_data[x].get('c_mf'), 0.0),
                'esr_mohm': safe_float(in_data[x].get('esr_mohm'), 0.0),
                'esl_uh': safe_float(in_data[x].get('esl_uh'), 0.0),
                'in_service': _electrisim_in_service(in_data[x]),
            })
            if not hasattr(net, 'user_friendly_names'):
                net.user_friendly_names = {}
            net.user_friendly_names[cap_name] = cap_label

        if (in_data[x]['typ'].startswith("Source DC")):
            bus_idx = _electrisim_dc_bus(net, in_data[x].get('bus'))
            if bus_idx is None:
                _electrisim_warn(net, f"Source DC '{in_data[x].get('userFriendlyName', in_data[x].get('name'))}' "
                                      "is not connected to a DC bus, so it is left out.")
                continue
            # Get in_service parameter (default to True if not specified)
            in_service = True
            if 'in_service' in in_data[x]:
                in_service = bool(in_data[x]['in_service']) if isinstance(in_data[x]['in_service'], bool) else (in_data[x]['in_service'] == 'true' or in_data[x]['in_service'] == True)
            src_extra = {}
            if in_data[x].get('id') is not None:
                src_extra['id'] = in_data[x]['id']
            try:
                src_idx = pp.create_source_dc(net, bus_dc=bus_idx, name=in_data[x]['name'],
                                    vm_pu=safe_float(in_data[x].get('vm_pu', 1.0)),
                                    in_service=in_service, **src_extra)
            except TypeError:
                try:
                    src_idx = pp.create_source_dc(net, bus=bus_idx, name=in_data[x]['name'],
                                        vm_pu=safe_float(in_data[x].get('vm_pu', 1.0)),
                                        in_service=in_service, **src_extra)
                except Exception as src_dc_err:
                    print(f"Warning: Source DC '{in_data[x].get('name')}' not created: {src_dc_err}")
                    continue
            except Exception as src_dc_err:
                print(f"Warning: Source DC '{in_data[x].get('name')}' not created: {src_dc_err}")
                continue
            
            # Its internal resistance and inductance, for the DC fault study: the load flow holds its voltage.
            net.source_dc.at[src_idx, 'electrisim_r_sc_mohm'] = safe_float(in_data[x].get('r_sc_mohm'), 0.0)
            net.source_dc.at[src_idx, 'electrisim_l_sc_uh'] = safe_float(in_data[x].get('l_sc_uh'), 0.0)

            # Store user-friendly name for source DC
            source_dc_name = in_data[x]['name']
            user_friendly_name = in_data[x].get('userFriendlyName', source_dc_name)
            if not hasattr(net, 'user_friendly_names'):
                net.user_friendly_names = {}
            net.user_friendly_names[source_dc_name] = user_friendly_name
        
        if (in_data[x]['typ'].startswith("Switch")):
            bus_idx = Busbars.get(in_data[x]['bus'])
            if bus_idx is None:
                element_name = in_data[x].get('userFriendlyName', in_data[x].get('name', 'Unknown'))
                bus_name = in_data[x].get('bus')
                if bus_name is None:
                    print(f"Warning: Switch '{element_name}' has no bus connection - skipped")
                else:
                    print(f"Warning: Switch '{element_name}' bus '{bus_name}' not found - skipped")
                continue
            # Get element: line/transformer name (frontend) -> lookup pandapower index
            element_name = in_data[x].get('element')
            if element_name is None:
                print(f"Warning: Switch '{in_data[x].get('userFriendlyName', in_data[x].get('name', 'Unknown'))}' has no element connection - skipped")
                continue
            # Map et to pandapower codes: 'l', 't', 't3', 'b'
            et_raw = in_data[x].get('et', 'l')
            et = _map_et_to_pandapower(et_raw)
            # Look up element index from name
            if et == 'l':
                element_idx = LinesDict.get(element_name)
            elif et == 't':
                element_idx = TrafoDict.get(element_name)
            elif et == 't3':
                element_idx = Trafo3wDict.get(element_name)
            elif et == 'b':
                element_idx = Busbars.get(element_name)
            else:
                element_idx = LinesDict.get(element_name) or TrafoDict.get(element_name)
            if element_idx is None:
                print(f"Warning: Switch '{in_data[x].get('userFriendlyName', in_data[x].get('name', 'Unknown'))}' element '{element_name}' (et={et}) not found - skipped")
                continue
            # Switch does not use in_service parameter - always in service
            closed = in_data[x].get('closed', True)
            if isinstance(closed, str):
                closed = closed.lower() == 'true'
            switch_type = in_data[x].get('type', 'CB')
            z_ohm = safe_float(in_data[x].get('z_ohm', 0.0))
            in_ka_val = in_data[x].get('in_ka')
            in_ka = safe_float(in_ka_val) if in_ka_val not in (None, '', 'nan') else float('nan')
            switch_name = in_data[x].get('name', in_data[x].get('userFriendlyName', f'Switch_{x}'))
            in_service = True

            if et == 'l':
                line_from = int(net.line.at[int(element_idx), 'from_bus'])
                line_to = int(net.line.at[int(element_idx), 'to_bus'])
                if int(bus_idx) not in (line_from, line_to):
                    label = in_data[x].get('userFriendlyName') or switch_name
                    print(
                        f"Warning: Switch '{label}' bus index {bus_idx} is not an end of "
                        f"line {element_idx} (ends {line_from}, {line_to}). Using bus {line_from}."
                    )
                    bus_idx = line_from

            sw_idx = pp.create_switch(net, bus=bus_idx, element=int(element_idx), et=et, name=switch_name,
                           closed=closed, type=switch_type, z_ohm=z_ohm, in_ka=in_ka, in_service=in_service)
            # Store frontend cell id for result matching
            if 'id' not in net.switch.columns:
                net.switch['id'] = None
            net.switch.at[sw_idx, 'id'] = in_data[x].get('id', in_data[x].get('name', str(sw_idx)))
            for col, key, default in (
                ('ansi_device_class', 'ansi_device_class', 'auto'),
                ('interrupting_rating_ka', 'interrupting_rating_ka', float('nan')),
                ('momentary_rating_ka', 'momentary_rating_ka', float('nan')),
                ('rated_voltage_kv', 'rated_voltage_kv', float('nan')),
                ('contact_parting_cycles', 'contact_parting_cycles', float('nan')),
                ('generator_cb', 'generator_cb', False),
            ):
                if col not in net.switch.columns:
                    net.switch[col] = default
                raw = in_data[x].get(key)
                if raw is None or str(raw).strip().lower() in ('', 'none', 'null', 'nan'):
                    continue
                if col == 'generator_cb':
                    net.switch.at[sw_idx, col] = bool(raw) if isinstance(raw, bool) else str(raw).lower() in ('true', '1', 'yes')
                elif col in ('ansi_device_class',):
                    net.switch.at[sw_idx, col] = str(raw).strip().lower()
                else:
                    try:
                        net.switch.at[sw_idx, col] = float(raw)
                    except (TypeError, ValueError):
                        pass
            
            # Store user-friendly name for switch
            switch_name = in_data[x]['name']
            user_friendly_name = in_data[x].get('userFriendlyName', switch_name)
            if not hasattr(net, 'user_friendly_names'):
                net.user_friendly_names = {}
            net.user_friendly_names[switch_name] = user_friendly_name
        
        if (in_data[x]['typ'].startswith("VSC")):
            # VSC requires pandapower 3.1+ with DC grid support
            if not hasattr(pp, 'create_vsc'):
                element_name = in_data[x].get('userFriendlyName', in_data[x].get('name', 'Unknown'))
                print(f"Warning: VSC '{element_name}' skipped - VSC not supported in pandapower {pp.__version__}. Upgrade to pandapower 3.1+")
                continue
                
            element_name = in_data[x].get('userFriendlyName', in_data[x].get('name', 'Unknown'))
            bus_idx = Busbars.get(in_data[x].get('bus', ''))
            bus_dc_idx = _electrisim_dc_bus(net, in_data[x].get('bus_dc', ''))
            if bus_idx is None or bus_dc_idx is None:
                missing = ' and '.join(k for k, v in (('an AC bus', bus_idx), ('a DC bus', bus_dc_idx)) if v is None)
                _electrisim_warn(net, f"VSC '{element_name}' is not connected to {missing}, so it is left out: "
                                      "a VSC joins one AC bus and one DC bus.")
                continue
            # Get in_service parameter (default to True if not specified)
            in_service = True
            if 'in_service' in in_data[x]:
                in_service = bool(in_data[x]['in_service']) if isinstance(in_data[x]['in_service'], bool) else (in_data[x]['in_service'] == 'true' or in_data[x]['in_service'] == True)
            
            # VSC parameters according to pandapower API
            # Required: r_ohm, x_ohm, r_dc_ohm (coupling transformer and DC resistance)
            r_ohm = safe_float(in_data[x].get('r_ohm', 0.01))  # Coupling transformer resistance
            x_ohm = safe_float(in_data[x].get('x_ohm', 0.1))   # Coupling transformer reactance
            r_dc_ohm = safe_float(in_data[x].get('r_dc_ohm', 0.01))  # Internal DC resistance
            
            # Control parameters
            control_mode_ac = in_data[x].get('control_mode_ac', 'vm_pu')  # 'vm_pu' or 'q_mvar'
            control_value_ac = safe_float(in_data[x].get('control_value_ac', in_data[x].get('vm_pu', 1.0)))
            control_mode_dc = in_data[x].get('control_mode_dc', 'p_mw')  # 'vm_pu' or 'p_mw'
            control_value_dc = safe_float(in_data[x].get('control_value_dc', in_data[x].get('p_mw', 0.0)))
            
            vsc_idx = pp.create_vsc(net, bus=bus_idx, bus_dc=bus_dc_idx, 
                         r_ohm=r_ohm, x_ohm=x_ohm, r_dc_ohm=r_dc_ohm,
                         control_mode_ac=control_mode_ac, control_value_ac=control_value_ac,
                         control_mode_dc=control_mode_dc, control_value_dc=control_value_dc,
                         name=in_data[x]['name'], in_service=in_service)
            
            # Store custom 'id' field in the VSC dataframe
            if 'id' not in net.vsc.columns:
                net.vsc['id'] = ''
            net.vsc.at[vsc_idx, 'id'] = in_data[x].get('id', '')
            # For the EMT study: its rating, DC link and current limit (0: from its load flow), its model
            for col, default in (('rated_mva', 0.0), ('dc_link_mf', 0.0), ('current_limit_pu', 1.2),
                                 ('switching_khz', 5.0)):
                if col not in net.vsc.columns:
                    net.vsc[col] = default
                net.vsc.at[vsc_idx, col] = safe_float(in_data[x].get(col, default), default)
            if 'emt_model' not in net.vsc.columns:
                net.vsc['emt_model'] = 'average'
            net.vsc.at[vsc_idx, 'emt_model'] = 'switching' if in_data[x].get('emt_model') == 'switching' else 'average'
            
            # Store user-friendly name for VSC
            vsc_name = in_data[x]['name']
            user_friendly_name = in_data[x].get('userFriendlyName', vsc_name)
            if not hasattr(net, 'user_friendly_names'):
                net.user_friendly_names = {}
            net.user_friendly_names[vsc_name] = user_friendly_name
        
        if (in_data[x]['typ'].startswith("B2B VSC")):
            element_name = in_data[x].get('userFriendlyName', in_data[x].get('name', 'Unknown'))
            
            # B2B VSC can work in two modes:
            # 1. Simple mode: AC bus to DC bus (uses create_vsc internally)
            # 2. Full mode: AC bus to DC bus pair (bus_dc_plus/bus_dc_minus)
            
            bus_idx = Busbars.get(in_data[x].get('bus', ''))
            bus_dc_idx = _electrisim_dc_bus(net, in_data[x].get('bus_dc', ''))
            
            # Check for simple VSC mode (AC bus to single DC bus)
            if bus_idx is not None and bus_dc_idx is not None:
                # Use create_vsc for simple AC-to-DC connection
                if not hasattr(pp, 'create_vsc'):
                    print(f"Warning: B2B VSC '{element_name}' skipped - VSC not supported in pandapower {pp.__version__}. Upgrade to pandapower 3.1+")
                    continue
                
                # Get in_service parameter
                in_service = True
                if 'in_service' in in_data[x]:
                    in_service = bool(in_data[x]['in_service']) if isinstance(in_data[x]['in_service'], bool) else (in_data[x]['in_service'] == 'true' or in_data[x]['in_service'] == True)
                
                # VSC parameters
                r_ohm = safe_float(in_data[x].get('r_ohm', 0.01))
                x_ohm = safe_float(in_data[x].get('x_ohm', 0.1))
                r_dc_ohm = safe_float(in_data[x].get('r_dc_ohm', 0.01))
                
                # Control parameters
                control_mode_ac = in_data[x].get('control_mode_ac', 'vm_pu')
                control_value_ac = safe_float(in_data[x].get('control_value_ac', in_data[x].get('vm_pu', 1.0)))
                control_mode_dc = in_data[x].get('control_mode_dc', 'p_mw')
                control_value_dc = safe_float(in_data[x].get('control_value_dc', in_data[x].get('p_mw', 0.0)))
                
                print(f"Creating VSC for B2B VSC '{element_name}': AC bus={bus_idx}, DC bus={bus_dc_idx}")
                vsc_b2b_idx = pp.create_vsc(net, bus=bus_idx, bus_dc=bus_dc_idx,
                             r_ohm=r_ohm, x_ohm=x_ohm, r_dc_ohm=r_dc_ohm,
                             control_mode_ac=control_mode_ac, control_value_ac=control_value_ac,
                             control_mode_dc=control_mode_dc, control_value_dc=control_value_dc,
                             name=in_data[x]['name'], in_service=in_service)
                if 'id' not in net.vsc.columns:
                    net.vsc['id'] = ''
                net.vsc.at[vsc_b2b_idx, 'id'] = in_data[x].get('id', '')
            else:
                # Try full B2B VSC mode with bus_dc_plus/bus_dc_minus
                if not hasattr(pp, 'create_b2b_vsc'):
                    print(f"Warning: B2B VSC '{element_name}' skipped - B2B VSC not supported in pandapower {pp.__version__}. Upgrade to pandapower 3.1+")
                    continue
                    
                bus_dc_plus_idx = _electrisim_dc_bus(net, in_data[x].get('bus_dc_plus', ''))
                bus_dc_minus_idx = _electrisim_dc_bus(net, in_data[x].get('bus_dc_minus', ''))
                
                if bus_idx is None or bus_dc_plus_idx is None or bus_dc_minus_idx is None:
                    _electrisim_warn(net, f"B2B VSC '{element_name}' is left out: it joins one AC bus to one DC bus, "
                                          "or to two DC buses (plus and minus poles).")
                    continue
                    
                # Get in_service parameter
                in_service = True
                if 'in_service' in in_data[x]:
                    in_service = bool(in_data[x]['in_service']) if isinstance(in_data[x]['in_service'], bool) else (in_data[x]['in_service'] == 'true' or in_data[x]['in_service'] == True)
                
                # B2B VSC parameters
                r_ohm = safe_float(in_data[x].get('r_ohm', 0.01))
                x_ohm = safe_float(in_data[x].get('x_ohm', 0.1))
                r_dc_ohm = safe_float(in_data[x].get('r_dc_ohm', 0.01))
                
                # Control parameters
                control_mode_ac = in_data[x].get('control_mode_ac', 'vm_pu')
                control_value_ac = safe_float(in_data[x].get('control_value_ac', in_data[x].get('vm1_pu', 1.0)))
                control_mode_dc = in_data[x].get('control_mode_dc', 'p_mw')
                control_value_dc = safe_float(in_data[x].get('control_value_dc', in_data[x].get('p_mw', 0.0)))
                
                pp.create_b2b_vsc(net, bus=bus_idx, bus_dc_plus=bus_dc_plus_idx, bus_dc_minus=bus_dc_minus_idx,
                                r_ohm=r_ohm, x_ohm=x_ohm, r_dc_ohm=r_dc_ohm,
                                control_mode_ac=control_mode_ac, control_value_ac=control_value_ac,
                                control_mode_dc=control_mode_dc, control_value_dc=control_value_dc,
                                name=in_data[x]['name'], id=in_data[x].get('id', ''), in_service=in_service)
            
            # Store user-friendly name
            b2b_vsc_name = in_data[x]['name']
            user_friendly_name = in_data[x].get('userFriendlyName', b2b_vsc_name)
            if not hasattr(net, 'user_friendly_names'):
                net.user_friendly_names = {}
            net.user_friendly_names[b2b_vsc_name] = user_friendly_name
        
        if (in_data[x]['typ'].startswith("DC Line")):
            element_name = in_data[x].get('userFriendlyName', in_data[x].get('name', 'Unknown'))
            bus_from = in_data[x].get('busFrom')
            bus_to = in_data[x].get('busTo')
            
            if bus_from is None:
                print(f"Warning: DC Line '{element_name}' skipped - missing 'busFrom' connection. "
                      f"DC Lines must be connected between two buses (draw it as a line connecting buses).")
                continue
            if bus_to is None:
                print(f"Warning: DC Line '{element_name}' skipped - missing 'busTo' connection. "
                      f"DC Lines must be connected between two buses (draw it as a line connecting buses).")
                continue
            
            from_dc, to_dc = _electrisim_dc_bus(net, bus_from), _electrisim_dc_bus(net, bus_to)
            is_dc_to_dc = from_dc is not None and to_dc is not None
            from_bus_idx = from_dc if is_dc_to_dc else Busbars.get(bus_from)
            to_bus_idx = to_dc if is_dc_to_dc else Busbars.get(bus_to)
            if from_bus_idx is None or to_bus_idx is None:
                _electrisim_warn(net, f"DC Line '{element_name}' is left out: it joins two DC buses (a DC cable) "
                                      "or two AC buses (an HVDC link), not one of each.")
                continue

            # Get in_service parameter (default to True if not specified)
            in_service = _electrisim_in_service(in_data[x])

            if is_dc_to_dc:
                # A DC cable between two DC buses, with its own length,
                # resistance and rating. (One std type built from the first
                # line used to give every DC line that line's resistance and 2 kA.)
                line_dc_idx = pp.create_line_dc_from_parameters(
                    net, from_bus_dc=from_bus_idx, to_bus_dc=to_bus_idx,
                    length_km=safe_float(in_data[x].get('length_km'), 1.0),
                    r_ohm_per_km=safe_float(in_data[x].get('r_ohm_per_km'), 0.1),
                    max_i_ka=safe_float(in_data[x].get('max_i_ka'), 1.0),
                    name=in_data[x]['name'], in_service=in_service,
                    # For the EMT study; a steady-state DC load flow does not use them.
                    l_mh_per_km=safe_float(in_data[x].get('l_mh_per_km'), 0.0),
                    c_uf_per_km=safe_float(in_data[x].get('c_uf_per_km'), 0.0))
                if 'id' not in net.line_dc.columns:
                    net.line_dc['id'] = ''
                net.line_dc.at[line_dc_idx, 'id'] = in_data[x].get('id', '')
            else:
                # DC Line connecting two AC buses - use create_dcline (simplified HVDC model)
                print(f"Creating DC line (dcline) '{element_name}': AC bus {bus_from} -> AC bus {bus_to}")
                _dcl_opf = _electrisim_opf_optional_fields_from_payload(
                    in_data[x],
                    float_keys=('max_p_mw', 'min_q_from_mvar', 'max_q_from_mvar', 'min_q_to_mvar', 'max_q_to_mvar'),
                    bool_keys=(),
                )
                pp.create_dcline(net, from_bus=from_bus_idx, to_bus=to_bus_idx, 
                               name=in_data[x]['name'], 
                               p_mw=safe_float(in_data[x].get('p_mw', 0.0)), 
                               loss_percent=safe_float(in_data[x].get('loss_percent', 0.0)), 
                               loss_mw=safe_float(in_data[x].get('loss_mw', 0.0)), 
                               vm_from_pu=safe_float(in_data[x].get('vm_from_pu', 1.0)), 
                               vm_to_pu=safe_float(in_data[x].get('vm_to_pu', 1.0)), 
                               in_service=in_service,
                               **_dcl_opf)
            
            # Store user-friendly name for DC Line
            dcline_name = in_data[x]['name']
            if not hasattr(net, 'user_friendly_names'):
                net.user_friendly_names = {}
            net.user_friendly_names[dcline_name] = element_name

    _electrisim_build_dc_dc_converters(net)
    _electrisim_build_ssts(net, Busbars)
    _electrisim_build_pcs(net, Busbars, study)
    _electrisim_build_ders(net)
    _electrisim_apply_dc_breakers(net)
    _electrisim_drop_uncoupled_dc(net)
    _electrisim_finish_dc_dc(net)
    _electrisim_finalize_pending_line_flow_shunts(net)
    apply_sgen_q_capability_curves(net, in_data)
    apply_sgen_q_setpoint_from_curve(net, in_data)
    try:
        net._electrisim_export_sgen_q_init = {
            int(i): float(net.sgen.at[i, 'q_mvar'])
            for i in net.sgen.index
        }
    except Exception:
        net._electrisim_export_sgen_q_init = {}
    # A study that does not model DC/DC converters sees each as its input's power.
    if (getattr(net, 'electrisim_dc_dc_converters', None) or getattr(net, 'electrisim_ssts', None)
            or getattr(net, 'electrisim_ders', None)) and (
            'OptimalPowerFlow' in study or not any(k in study for k in _DCDC_STUDIES)):
        _electrisim_freeze_dc_dc(net)


def _electrisim_boolish(v, default=False):
    if v is None:
        return default
    if isinstance(v, str):
        return v.strip().lower() in ('true', '1', 'yes', 'on')
    return bool(v)


def _electrisim_opf_optional_fields_from_payload(row, float_keys=(), bool_keys=()):
    """Parse optional OPF-related keys from a frontend element dict (skip empty / NaN)."""
    out = {}
    if not isinstance(row, dict):
        return out
    for k in bool_keys:
        if k not in row:
            continue
        out[k] = _electrisim_boolish(row.get(k), False)
    for k in float_keys:
        if k not in row:
            continue
        raw = row[k]
        if raw is None or raw == '' or str(raw).strip() == '':
            continue
        v = safe_float(raw, float('nan'))
        if v == v:
            out[k] = v
    return out


def _resolve_controller_family_flags(payload, legacy_key='run_control'):
    """
    Returns (rc2, rc3, rcs): whether to run DiscreteTapControl on 2w trafos, 3w trafos,
    and DiscreteShuntController on shunts. If any granular key is present on payload,
    those values are used (missing granular keys default to False). Otherwise all three
    follow legacy run_control (single boolean).
    """
    if not isinstance(payload, dict):
        return False, False, False
    k2, k3, ks = 'run_control_trafo2w', 'run_control_trafo3w', 'run_control_shunt'
    if any(k in payload for k in (k2, k3, ks)):
        return (
            _electrisim_boolish(payload.get(k2), False),
            _electrisim_boolish(payload.get(k3), False),
            _electrisim_boolish(payload.get(ks), False),
        )
    u = _electrisim_boolish(payload.get(legacy_key), False)
    return u, u, u


def _park_truthy(v, default=True):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ('1', 'true', 'yes', 'on')


def _park_parse_json_list(raw):
    if raw is None or raw == '':
        return []
    if isinstance(raw, list):
        return raw
    try:
        v = json.loads(raw)
        return v if isinstance(v, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _park_interp_xy(points, x_key, y_key, x):
    """Piecewise-linear interpolate y at x from list of dicts; clamp outside range."""
    pts = []
    for p in points or []:
        try:
            xv = float(p.get(x_key))
            yv = float(p.get(y_key))
        except (TypeError, ValueError):
            continue
        if xv == xv and yv == yv:
            pts.append((xv, yv))
    if not pts or x is None or x != x:
        return None
    pts.sort(key=lambda t: t[0])
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        if x0 <= x <= x1:
            if abs(x1 - x0) < 1e-12:
                return y0
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return pts[-1][1]


def _park_find_by_name(df, name, user_friendly_names=None):
    """Return first index in df whose name or friendly name matches."""
    if df is None or getattr(df, 'empty', True) or not name:
        return None
    target = str(name).strip()
    friendly = user_friendly_names or {}
    # reverse friendly map: friendly -> technical
    rev = {str(v): str(k) for k, v in friendly.items()}
    tech = rev.get(target, target)
    if 'name' not in df.columns:
        return None
    for idx in df.index:
        n = df.at[idx, 'name']
        if n is None or (isinstance(n, float) and np.isnan(n)):
            continue
        ns = str(n)
        if ns == target or ns == tech or friendly.get(ns) == target:
            return idx
    return None


def _park_q_from_cosphi(p_mw, cos_phi):
    """Q = P * tan(acos(|cosφ|)); magnitude for OE/UE branch signing."""
    try:
        p = float(p_mw)
        c = float(cos_phi)
    except (TypeError, ValueError):
        return 0.0
    c = max(-1.0, min(1.0, c))
    if abs(c) < 1e-9:
        return 0.0
    import math as _math
    return p * _math.tan(_math.acos(abs(c)))


def _park_pick_cosphi_p_branch(park, q_mvar_meas):
    """
    Return 'oe' or 'ue' for cosφ(P) characteristic selection.
    Auto uses measured Q at the control point (Q >= 0 → overexcited).
    """
    mode = str(park.get('cosphi_p_excitation') or 'Overexcited').strip().lower()
    if mode.startswith('under'):
        return 'ue'
    if mode.startswith('auto'):
        try:
            q = float(q_mvar_meas) if q_mvar_meas is not None else 0.0
        except (TypeError, ValueError):
            q = 0.0
        return 'oe' if q >= 0.0 else 'ue'
    return 'oe'


def _park_cosphi_p_points(park, branch):
    """Resolve OE/UE cosφ(P) table; fall back to legacy single curve."""
    if branch == 'ue':
        raw = park.get('cosphi_p_ue_characteristic_json')
    else:
        raw = park.get('cosphi_p_oe_characteristic_json')
    pts = _park_parse_json_list(raw)
    if pts:
        return pts
    return _park_parse_json_list(park.get('cosphi_p_characteristic_json'))


def _park_measure_p_q_at(net, element_name):
    """Return (p_mw, q_mvar) at a named bus, line (from-side), or trafo (HV), or (None, None)."""
    ufn = getattr(net, 'user_friendly_names', {}) or {}
    bi = _park_find_by_name(net.bus, element_name, ufn)
    if bi is not None and hasattr(net, 'res_bus') and not net.res_bus.empty and bi in net.res_bus.index:
        return float(net.res_bus.at[bi, 'p_mw']), float(net.res_bus.at[bi, 'q_mvar'])
    li = _park_find_by_name(net.line, element_name, ufn)
    if li is not None and hasattr(net, 'res_line') and not net.res_line.empty and li in net.res_line.index:
        return float(net.res_line.at[li, 'p_from_mw']), float(net.res_line.at[li, 'q_from_mvar'])
    ti = _park_find_by_name(net.trafo, element_name, ufn)
    if ti is not None and hasattr(net, 'res_trafo') and not net.res_trafo.empty and ti in net.res_trafo.index:
        return float(net.res_trafo.at[ti, 'p_hv_mw']), float(net.res_trafo.at[ti, 'q_hv_mvar'])
    return None, None


def _park_resolve_control_q_at(net, boundary):
    """Resolve Control Q at name to (input_element, input_variable, index) or (None, None, None)."""
    ufn = getattr(net, 'user_friendly_names', {}) or {}
    bi = _park_find_by_name(net.bus, boundary, ufn)
    if bi is not None:
        return 'res_bus', 'q_mvar', bi
    li = _park_find_by_name(net.line, boundary, ufn)
    if li is not None:
        return 'res_line', 'q_from_mvar', li
    ti = _park_find_by_name(net.trafo, boundary, ufn)
    if ti is not None:
        return 'res_trafo', 'q_hv_mvar', ti
    return None, None, None


def _park_machine_q_limits(net, sgen_idx):
    """
    (q_min_mvar, q_max_mvar) for a park machine at its current P and terminal voltage.
    Prefers P–Q / P–U capability (Wind Turbine / SGen Q capability tab); falls back to
    min_q_mvar / max_q_mvar columns.
    """
    try:
        p_mw = float(net.sgen.at[sgen_idx, 'p_mw']) if 'p_mw' in net.sgen.columns else 0.0
    except Exception:
        p_mw = 0.0
    lim = _interp_sgen_pq_limits(net, sgen_idx, p_mw, vm_pu=_sgen_bus_vm_pu(net, sgen_idx))
    if lim is not None:
        return float(lim[0]), float(lim[1])
    q_min = None
    q_max = None
    try:
        if 'min_q_mvar' in net.sgen.columns:
            v = net.sgen.at[sgen_idx, 'min_q_mvar']
            if v == v and v is not None:
                q_min = float(v)
        if 'max_q_mvar' in net.sgen.columns:
            v = net.sgen.at[sgen_idx, 'max_q_mvar']
            if v == v and v is not None:
                q_max = float(v)
    except Exception:
        pass
    return q_min, q_max


def _park_apply_machine_q_capability_limits(net, sgen_indices):
    """
    Write interpolated P–Q capability limits onto net.sgen min/max_q_mvar for park machines
    and enable enforce_q_lims so BinarySearchControl stays inside the curve.
    Returns number of machines updated from a curve.
    """
    if not sgen_indices:
        return 0
    updated = 0
    for si in sgen_indices:
        lim = _interp_sgen_pq_limits(
            net,
            si,
            float(net.sgen.at[si, 'p_mw']) if 'p_mw' in net.sgen.columns else 0.0,
            vm_pu=_sgen_bus_vm_pu(net, si),
        )
        if lim is None:
            continue
        q_mi, q_ma = float(lim[0]), float(lim[1])
        if 'min_q_mvar' not in net.sgen.columns:
            net.sgen['min_q_mvar'] = float('nan')
        if 'max_q_mvar' not in net.sgen.columns:
            net.sgen['max_q_mvar'] = float('nan')
        net.sgen.at[si, 'min_q_mvar'] = q_mi
        net.sgen.at[si, 'max_q_mvar'] = q_ma
        updated += 1
    if updated:
        net._electrisim_enforce_q_lims = True
    return updated


def _park_resolve_sgen_indices(net, machines, distribution_method=None):
    """machines: list of {name, connected, q_percent}. Returns (indices, in_service, distribution, gen_Q_response).

    distribution_method (optional) recomputes weights from the live network for non-Individual modes.
    """
    ufn = getattr(net, 'user_friendly_names', {}) or {}
    idxs, insvc, user_pct, resp = [], [], [], []
    for m in machines or []:
        if not m or not _park_truthy(m.get('connected', True), True):
            continue
        name = m.get('name')
        si = _park_find_by_name(net.sgen, name, ufn)
        if si is None:
            # Generators (sync) are net.gen — BinarySearchControl Q supports sgen primarily
            continue
        idxs.append(si)
        insvc.append(bool(net.sgen.at[si, 'in_service']) if 'in_service' in net.sgen.columns else True)
        user_pct.append(max(0.0, float(m.get('q_percent') or 0.0)))
        resp.append(1)
    if not idxs:
        return [], [], [], []

    method = str(distribution_method or 'Individual Reactive Power')
    if method == 'Individual Reactive Power':
        dist = user_pct
    elif method == 'According to Dispatched Active Power':
        dist = [abs(float(net.sgen.at[i, 'p_mw'])) for i in idxs]
    elif method == 'According to Rated Power':
        dist = []
        for i in idxs:
            sn = float(net.sgen.at[i, 'sn_mva']) if 'sn_mva' in net.sgen.columns and net.sgen.at[i, 'sn_mva'] == net.sgen.at[i, 'sn_mva'] else 0.0
            dist.append(abs(sn) if sn else 0.0)
    elif method == 'According to Q Capability':
        # Share proportional to Q band width at current P (from P–Q curve when available)
        dist = []
        for i in idxs:
            q_mi, q_ma = _park_machine_q_limits(net, i)
            if q_mi is not None and q_ma is not None:
                dist.append(max(0.0, float(q_ma) - float(q_mi)))
            else:
                dist.append(1.0)
    elif method == 'Maximise Reactive Reserve':
        dist = []
        for i in idxs:
            q = float(net.sgen.at[i, 'q_mvar']) if 'q_mvar' in net.sgen.columns else 0.0
            q_mi, q_ma = _park_machine_q_limits(net, i)
            if q_ma is not None and q_mi is not None:
                # Remaining headroom toward both capacitive and inductive limits
                dist.append(max(0.0, float(q_ma) - q) + max(0.0, q - float(q_mi)))
            elif q_ma is not None:
                dist.append(max(0.0, float(q_ma) - q))
            else:
                dist.append(1.0)
    else:
        # Voltage Setpoint Adaption / unknown → equal shares
        dist = [1.0] * len(idxs)

    if sum(dist) <= 0:
        dist = [1.0] * len(idxs)
    return idxs, insvc, dist, resp


def _electrisim_collect_park_payloads(in_data):
    parks = []
    if not isinstance(in_data, dict):
        return parks
    for _k, el in in_data.items():
        if not isinstance(el, dict):
            continue
        typ = str(el.get('typ') or '')
        if typ == 'ParkController' or typ.startswith('ParkController'):
            if _park_truthy(el.get('enabled', True), True):
                parks.append(el)
    return parks


def _electrisim_collect_wtc_ss_payloads(in_data):
    """Steady-state Wind Turbine Controllers from LF payload."""
    out = []
    if not isinstance(in_data, dict):
        return out
    for _k, el in in_data.items():
        if not isinstance(el, dict):
            continue
        typ = str(el.get('typ') or '')
        if typ == 'WindTurbineDynamicController':
            continue
        if typ == 'WindTurbineController' or typ.startswith('WindTurbineController'):
            if _park_truthy(el.get('enabled', True), True):
                out.append(el)
    return out


def _electrisim_collect_wtc_dyn_payloads(in_data):
    """Dynamic Wind Turbine Controllers from LF payload (documented only for snapshot LF)."""
    out = []
    if not isinstance(in_data, dict):
        return out
    for _k, el in in_data.items():
        if not isinstance(el, dict):
            continue
        typ = str(el.get('typ') or '')
        if typ == 'WindTurbineDynamicController' or typ.startswith('WindTurbineDynamicController'):
            if _park_truthy(el.get('enabled', True), True):
                out.append(el)
    return out


def _electrisim_find_in_data_by_sgen(in_data, sgen_name, sgen_id=None):
    if not isinstance(in_data, dict):
        return None
    for _k, el in in_data.items():
        if not isinstance(el, dict):
            continue
        typ = str(el.get('typ') or '')
        if not (typ.startswith('Wind Turbine') or typ.startswith('Static Generator') or typ == 'Static Generator'):
            continue
        if sgen_id is not None and el.get('id') is not None and str(el.get('id')) == str(sgen_id):
            return el
        if el.get('name') == sgen_name or el.get('userFriendlyName') == sgen_name:
            return el
    return None


def _electrisim_build_park_controller_results(net, in_data):
    """Summaries for results .txt / UI after load flow."""
    results = []
    parks = _electrisim_collect_park_payloads(in_data)
    if not parks:
        return results
    bsc_by_name = {}
    if hasattr(net, 'controller') and net.controller is not None and not net.controller.empty:
        for _ci, row in net.controller.iterrows():
            obj = row['object']
            if type(obj).__name__ == 'BinarySearchControl':
                bsc_by_name[str(getattr(obj, 'name', '') or '')] = obj
    for park in parks:
        name = str(park.get('name') or 'ParkController')
        machines = _park_parse_json_list(park.get('machines_json'))
        idxs, _insvc, _dist, _resp = _park_resolve_sgen_indices(
            net, machines, park.get('distribution_method'))
        sgen_names = []
        sgen_q = []
        for i in idxs:
            try:
                sgen_names.append(str(net.sgen.at[i, 'name']) if 'name' in net.sgen.columns else str(i))
            except Exception:
                sgen_names.append(str(i))
            try:
                if hasattr(net, 'res_sgen') and net.res_sgen is not None and not net.res_sgen.empty and i in net.res_sgen.index:
                    sgen_q.append(float(net.res_sgen.at[i, 'q_mvar']))
                else:
                    sgen_q.append(None)
            except Exception:
                sgen_q.append(None)
        bsc = bsc_by_name.get(name)
        mode = str(park.get('control_mode') or 'Voltage Control')
        results.append({
            'name': name,
            'control_mode': mode,
            'q_control_type': park.get('q_control_type'),
            'pf_control_type': park.get('pf_control_type'),
            'cosphi_p_excitation': park.get('cosphi_p_excitation'),
            'cosphi_p_oe_characteristic_json': (
                park.get('cosphi_p_oe_characteristic_json') or park.get('cosphi_p_characteristic_json')
            ),
            'cosphi_p_ue_characteristic_json': park.get('cosphi_p_ue_characteristic_json'),
            'controlled_bus': park.get('controlled_bus') or park.get('target_bus'),
            'control_q_at': park.get('control_q_at'),
            'vm_set_pu': park.get('vm_set_pu'),
            'q_set_mvar': park.get('q_set_mvar'),
            'cos_phi': park.get('cos_phi'),
            'tan_phi': park.get('tan_phi'),
            'enable_droop': _park_truthy(park.get('enable_droop'), False),
            'droop_percent': park.get('droop_percent'),
            'q_rated_mvar': park.get('q_rated_mvar'),
            'distribution_method': park.get('distribution_method'),
            'machines': sgen_names,
            'sgen_q_mvar': sgen_q,
            'set_point': float(bsc.set_point) if bsc is not None else None,
            'voltage_ctrl': bool(getattr(bsc, 'voltage_ctrl', mode == 'Voltage Control')) if bsc is not None else (mode == 'Voltage Control'),
            'attached': bsc is not None,
        })
    return results


def _electrisim_build_wtc_controller_results(in_data):
    """Steady-state + dynamic Wind Turbine Controller summaries for results export."""
    ss_out = []
    dyn_out = []
    if not isinstance(in_data, dict):
        return ss_out, dyn_out
    # Pref applied onto turbines (frontend apply)
    pref_by_ctrl = {}
    for _k, el in in_data.items():
        if not isinstance(el, dict):
            continue
        typ = str(el.get('typ') or '')
        if typ.startswith('Wind Turbine') and el.get('_wind_controller'):
            cname = str(el.get('_wind_controller'))
            pref_by_ctrl.setdefault(cname, []).append({
                'turbine': el.get('userFriendlyName') or el.get('name'),
                'p_mw': el.get('p_mw'),
                'wind_speed_ms': el.get('wind_speed_ms'),
            })
    for park_like in _electrisim_collect_wtc_ss_payloads(in_data):
        cname = str(park_like.get('name') or 'WindTurbineController')
        applied = pref_by_ctrl.get(cname) or []
        target = park_like.get('wind_turbine') or (applied[0]['turbine'] if applied else '')
        pref = applied[0]['p_mw'] if applied else None
        v_ms = applied[0]['wind_speed_ms'] if applied else park_like.get('wind_speed_ms')
        ss_out.append({
            'name': cname,
            'kind': 'steady-state',
            'wind_turbine': target,
            'power_curve_type': park_like.get('power_curve_type'),
            'use_turbine_wind_speed': park_like.get('use_turbine_wind_speed'),
            'wind_speed_ms': v_ms,
            'pref_mw': pref,
            'wind_curve_approx': park_like.get('wind_curve_approx'),
        })
    # Orphan Pref notes (controller applied but payload object missing)
    named_ss = {r['name'] for r in ss_out}
    for cname, apps in pref_by_ctrl.items():
        if cname in named_ss:
            continue
        for a in apps:
            ss_out.append({
                'name': cname,
                'kind': 'steady-state',
                'wind_turbine': a.get('turbine'),
                'power_curve_type': None,
                'use_turbine_wind_speed': None,
                'wind_speed_ms': a.get('wind_speed_ms'),
                'pref_mw': a.get('p_mw'),
                'wind_curve_approx': None,
            })
    for dyn in _electrisim_collect_wtc_dyn_payloads(in_data):
        dyn_out.append({
            'name': str(dyn.get('name') or 'WindTurbineController (dynamic)'),
            'kind': 'dynamic',
            'wind_turbine': dyn.get('wind_turbine'),
            'wind_avg_T': dyn.get('wind_avg_T'),
            'wind_avg_Tavg': dyn.get('wind_avg_Tavg'),
            'power_avg_T': dyn.get('power_avg_T'),
            'power_avg_Tavg': dyn.get('power_avg_Tavg'),
            'gradient_T': dyn.get('gradient_T'),
            'gradient_max': dyn.get('gradient_max'),
            'note': 'Not applied to snapshot load flow (time-domain / dynamic studies only).',
        })
    return ss_out, dyn_out


def _export_tap_side(val):
    """Normalize tap_side for exported pandapower scripts."""
    return _normalize_tap_side(val)


def _append_electrisim_park_specs_python(lines, net, parks):
    """Emit BinarySearchControl (+ Droop) from post-LF export snapshot."""
    park_specs = getattr(net, '_electrisim_export_park_specs', None) or []
    if not park_specs:
        return False

    lines.append("from pandapower.control.controller.station_control import BinarySearchControl, DroopControl")
    lines.append("")
    for spec in park_specs:
        cname = str(spec.get('name') or 'ParkController')
        park_meta = next((p for p in parks if str(p.get('name') or '') == cname), None)
        if park_meta:
            lines.append(
                f"# ParkController '{cname}': mode={spec.get('control_mode')!r}, "
                f"q_type={park_meta.get('q_control_type')!r}, "
                f"pf_type={park_meta.get('pf_control_type')!r}, "
                f"cosphi_p_excitation={spec.get('cosphi_p_excitation')!r}, "
                f"distribution={spec.get('distribution_method')!r}"
            )
            oe = park_meta.get('cosphi_p_oe_characteristic_json') or park_meta.get('cosphi_p_characteristic_json')
            ue = park_meta.get('cosphi_p_ue_characteristic_json')
            if oe:
                lines.append(f"#   cosphi(P) OE characteristic: {oe}")
            if ue:
                lines.append(f"#   cosphi(P) UE characteristic: {ue}")
        else:
            lines.append(
                f"# ParkController '{cname}': mode={spec.get('control_mode')!r}, "
                f"pf_type={spec.get('pf_control_type')!r}, set_point={spec.get('set_point')}"
            )

        idxs = spec.get('idxs') or []
        insvc = spec.get('insvc') or []
        dist = spec.get('dist') or []
        in_el = spec.get('input_element') or 'res_bus'
        in_var = spec.get('input_variable') or 'q_mvar'
        in_idx = spec.get('input_element_index')
        set_point = float(spec.get('set_point') or 0.0)
        voltage_ctrl = bool(spec.get('voltage_ctrl', False))
        gen_q = spec.get('gen_Q_response') or [1] * len(idxs)
        bus_idx = spec.get('bus_idx')
        ci = spec.get('controller_idx')
        if ci is None:
            ci = 0

        lines.append(
            f"_park_ctrl_{ci} = BinarySearchControl(\n"
            f"    net, True, 'sgen', 'q_mvar',\n"
            f"    {_export_py_literal(idxs)}, {_export_py_literal(insvc)}, {_export_py_literal(dist)},\n"
            f"    {in_el!r}, {in_var!r}, {_export_py_literal(in_idx)}, {set_point}, {voltage_ctrl},\n"
            f"    name={cname!r}, gen_Q_response={_export_py_literal(gen_q)},\n"
            f"    bus_idx={bus_idx!r}, tol=0.001, in_service=True, order=0, level=0)"
        )
        lines.append(f"_park_ctrl_idx_{ci} = net.controller.index[-1]")

        droop = spec.get('droop')
        if isinstance(droop, dict) and droop.get('q_droop_mvar') is not None:
            lines.append(
                f"DroopControl(\n"
                f"    net, {float(droop['q_droop_mvar'])}, "
                f"{droop.get('bus_idx')!r}, _park_ctrl_idx_{ci}, True,\n"
                f"    tol=1e-6, vm_set_pu_bsc={droop.get('vm_set_pu_bsc')!r},\n"
                f"    name={str(droop.get('name') or cname + '_droop')!r}, "
                f"in_service=True, order=-1, level=0)"
            )
        lines.append("")
    return True


def _append_electrisim_controllers_to_python(lines, net, in_data, algorithm, calculate_voltage_angles, init):
    """
    Append ParkController (BinarySearchControl/DroopControl) and Wind Turbine Controller
    blocks to exported pandapower Python. Returns True if runpp should use run_control=True.
    """
    parks = _electrisim_collect_park_payloads(in_data)
    wtc_ss = _electrisim_collect_wtc_ss_payloads(in_data)
    wtc_dyn = _electrisim_collect_wtc_dyn_payloads(in_data)
    run_control = False

    # Annotate Pref applied on wind turbines even without controller payload objects
    pref_notes = []
    if isinstance(in_data, dict):
        for _k, el in in_data.items():
            if not isinstance(el, dict):
                continue
            if str(el.get('typ') or '').startswith('Wind Turbine') and el.get('_wind_controller'):
                pref_notes.append(el)

    if not parks and not wtc_ss and not wtc_dyn and not pref_notes:
        return False

    lines.append("# --- Electrisim controllers (Park / Wind Turbine) ---")
    lines.append("")

    if wtc_ss or pref_notes:
        lines.append("# Wind Turbine Controller (steady-state): Pref applied to linked turbines before LF")
        for ctrl in wtc_ss:
            cname = ctrl.get('name') or 'WindTurbineController'
            lines.append(
                f"#   {cname}: turbine={ctrl.get('wind_turbine')!r}, "
                f"curve={ctrl.get('power_curve_type')!r}, "
                f"use_turbine_wind_speed={ctrl.get('use_turbine_wind_speed')}, "
                f"wind_speed_ms={ctrl.get('wind_speed_ms')}, approx={ctrl.get('wind_curve_approx')!r}"
            )
        for el in pref_notes:
            lines.append(
                f"#   Pref applied: turbine={el.get('userFriendlyName') or el.get('name')!r} "
                f"p_mw={el.get('p_mw')} wind_speed_ms={el.get('wind_speed_ms')} "
                f"via controller={el.get('_wind_controller')!r}"
            )
        lines.append("")

    if wtc_dyn:
        lines.append("# Wind Turbine Controller (dynamic) — parameters for time-domain studies (not used in snapshot LF)")
        for ctrl in wtc_dyn:
            lines.append(
                f"#   {ctrl.get('name')}: turbine={ctrl.get('wind_turbine')!r}, "
                f"wind_avg T={ctrl.get('wind_avg_T')} Tavg={ctrl.get('wind_avg_Tavg')}, "
                f"gradient T={ctrl.get('gradient_T')} max={ctrl.get('gradient_max')}, "
                f"power_avg T={ctrl.get('power_avg_T')} Tavg={ctrl.get('power_avg_Tavg')}"
            )
        lines.append("")

    if parks:
        emitted_names = set()
        if _append_electrisim_park_specs_python(lines, net, parks):
            run_control = True
            for spec in (getattr(net, '_electrisim_export_park_specs', None) or []):
                emitted_names.add(str(spec.get('name') or 'ParkController'))
        else:
            lines.append("from pandapower.control.controller.station_control import BinarySearchControl, DroopControl")
            lines.append("")
            # Prefer emitting controllers already attached on net (post-LF) so setpoints match Electrisim
            if hasattr(net, 'controller') and net.controller is not None and not net.controller.empty:
                for ci, row in net.controller.iterrows():
                    obj = row['object']
                    cls = type(obj).__name__
                    if cls != 'BinarySearchControl':
                        continue
                    cname = str(getattr(obj, 'name', '') or f'ParkController_{ci}')
                    # Match Electrisim park metadata for comments
                    park_meta = next((p for p in parks if str(p.get('name') or '') == cname), None)
                    if park_meta:
                        lines.append(
                            f"# ParkController '{cname}': mode={park_meta.get('control_mode')!r}, "
                            f"q_type={park_meta.get('q_control_type')!r}, "
                            f"pf_type={park_meta.get('pf_control_type')!r}, "
                            f"cosphi_p_excitation={park_meta.get('cosphi_p_excitation')!r}, "
                            f"distribution={park_meta.get('distribution_method')!r}"
                        )
                        oe = park_meta.get('cosphi_p_oe_characteristic_json') or park_meta.get('cosphi_p_characteristic_json')
                        ue = park_meta.get('cosphi_p_ue_characteristic_json')
                        if oe:
                            lines.append(f"#   cosphi(P) OE characteristic: {oe}")
                        if ue:
                            lines.append(f"#   cosphi(P) UE characteristic: {ue}")
                        if _park_truthy(park_meta.get('enable_droop'), False):
                            lines.append(
                                f"#   droop enabled: droop_percent={park_meta.get('droop_percent')}, "
                                f"q_rated_mvar={park_meta.get('q_rated_mvar')}"
                            )
                    out_idx = list(getattr(obj, 'output_element_index', []) or [])
                    out_insvc = list(getattr(obj, 'output_element_in_service', []) or [])
                    dist = getattr(obj, 'output_values_distribution', None)
                    if dist is None:
                        dist_list = [1.0] * len(out_idx)
                    else:
                        dist_list = [float(x) for x in list(dist)]
                    # pandapower may store normalized weights as zeros in the object; rebuild from diagram
                    if park_meta and (not dist_list or sum(abs(x) for x in dist_list) <= 0):
                        machines = _park_parse_json_list(park_meta.get('machines_json'))
                        _, _, dist_list, _ = _park_resolve_sgen_indices(
                            net, machines, park_meta.get('distribution_method'))
                    if not dist_list or sum(abs(x) for x in dist_list) <= 0:
                        dist_list = [1.0] * len(out_idx)
                    gen_q = list(getattr(obj, 'gen_Q_response', []) or [])
                    if park_meta and (not gen_q or len(gen_q) != len(out_idx)):
                        q_resp = str(park_meta.get('q_change_response') or 'same').lower()
                        gen_q = [1 if q_resp != 'opposite' else -1] * len(out_idx)
                    in_el = getattr(obj, 'input_element', 'res_bus')
                    in_var_list = getattr(obj, 'input_variable', None)
                    voltage_ctrl = bool(getattr(obj, 'voltage_ctrl', False))
                    if voltage_ctrl:
                        in_var = 'vm_pu'
                    elif isinstance(in_var_list, list) and in_var_list:
                        v0 = in_var_list[0]
                        in_var = v0 if isinstance(v0, str) else str(v0)
                        # Common Q measurements when flag objects stringify poorly
                        if in_var and ('object' in in_var.lower() or len(in_var) > 40):
                            if in_el == 'res_bus':
                                in_var = 'q_mvar'
                            elif in_el == 'res_line':
                                in_var = 'q_from_mvar'
                            elif in_el in ('res_trafo', 'res_trafo3w'):
                                in_var = 'q_hv_mvar'
                    else:
                        in_var = 'q_mvar' if in_el == 'res_bus' else 'q_from_mvar'
                    in_idx_list = list(getattr(obj, 'input_element_index', []) or [])
                    in_idx = in_idx_list[0] if len(in_idx_list) == 1 else in_idx_list
                    set_point = float(getattr(obj, 'set_point', 0.0))
                    bus_idx = getattr(obj, 'bus_idx', None)
                    out_var = getattr(obj, 'output_variable', 'q_mvar')
                    if isinstance(out_var, list):
                        out_var = out_var[0] if out_var else 'q_mvar'
                    lines.append(
                        f"_park_ctrl_{ci} = BinarySearchControl(\n"
                        f"    net, True, {getattr(obj, 'output_element', 'sgen')!r}, {out_var!r},\n"
                        f"    {_export_py_literal(out_idx)}, {_export_py_literal(out_insvc)}, {_export_py_literal(dist_list)},\n"
                        f"    {in_el!r}, {in_var!r}, {_export_py_literal(in_idx)}, {set_point}, {voltage_ctrl},\n"
                        f"    name={cname!r}, gen_Q_response={_export_py_literal(gen_q)},\n"
                        f"    bus_idx={bus_idx!r}, tol=0.001, in_service=True, order=0, level=0)"
                    )
                    lines.append(f"_park_ctrl_idx_{ci} = net.controller.index[-1]")
                    # Droop linked to this BSC
                    for dj, drow in net.controller.iterrows():
                        dobj = drow['object']
                        if type(dobj).__name__ != 'DroopControl':
                            continue
                        if getattr(dobj, 'controller_idx', None) != ci:
                            continue
                        lines.append(
                            f"DroopControl(\n"
                            f"    net, {float(getattr(dobj, 'q_droop_mvar', 0.0))}, "
                            f"{getattr(dobj, 'bus_idx', None)!r}, _park_ctrl_idx_{ci}, True,\n"
                            f"    tol=1e-6, vm_set_pu_bsc={getattr(dobj, 'vm_set_pu_bsc', None)!r},\n"
                            f"    name={str(getattr(dobj, 'name', cname + '_droop'))!r}, "
                            f"in_service=True, order=-1, level=0)"
                        )
                    lines.append("")
                    emitted_names.add(cname)
                    run_control = True

        # Parks that failed to attach: still document configuration
        for park in parks:
            pname = str(park.get('name') or 'ParkController')
            if pname in emitted_names:
                continue
            lines.append(
                f"# ParkController '{pname}' was not attached during Electrisim LF "
                f"(mode={park.get('control_mode')!r}, machines={park.get('machines_json')}). "
                f"Check machine / Control Q at names."
            )
            lines.append(
                f"#   cosphi_p_excitation={park.get('cosphi_p_excitation')!r}, "
                f"OE={park.get('cosphi_p_oe_characteristic_json') or park.get('cosphi_p_characteristic_json')}, "
                f"UE={park.get('cosphi_p_ue_characteristic_json')}"
            )
            lines.append("")

    return run_control


def _electrisim_attach_park_controllers(net, in_data, algorithm='nr', calculate_voltage_angles=True, init='auto', quiet=False):
    """
    Attach pandapower BinarySearchControl (+ optional DroopControl) for Electrisim ParkController
    payloads (Electrisim ParkController steady-state plant control).
    Returns number of park controllers successfully attached.
    """
    parks = _electrisim_collect_park_payloads(in_data)
    if not parks:
        return 0

    def _plog(msg):
        if not quiet:
            print(msg)

    try:
        from pandapower.control.controller.station_control import BinarySearchControl, DroopControl
    except ImportError:
        print('[ParkController] pandapower station_control not available; skipping')
        return 0

    # Ensure Wind Turbine / SGen P–Q capability curves are on the net when parks use them
    if any(_park_truthy(p.get('use_q_capability', True), True) for p in parks):
        try:
            apply_sgen_q_capability_curves(net, in_data, rpc_use_diagram_curves=True)
        except Exception as ex:
            print(f'[ParkController] Q capability curve apply skipped: {ex}')

    # One uncontrolled solve so characteristics can use P/V measurements
    need_seed = any(
        (p.get('control_mode') in ('Reactive Power Control', 'Power Factor Control') and
         (str(p.get('q_control_type') or '').startswith('Q(') or
          str(p.get('pf_control_type') or '').startswith('cosphi(')))
        or p.get('control_mode') == 'Power Factor Control'
        or p.get('control_mode') == 'tan(phi) Control'
        for p in parks
    )
    if need_seed:
        try:
            pp.runpp(net, algorithm=algorithm, calculate_voltage_angles=calculate_voltage_angles,
                     init=init, run_control=False)
        except Exception as ex:
            print(f'[ParkController] seed load flow failed (continuing with setpoints only): {ex}')
    net._electrisim_export_park_need_seed = need_seed

    attached = 0
    ufn = getattr(net, 'user_friendly_names', {}) or {}

    for park in parks:
        machines = _park_parse_json_list(park.get('machines_json'))
        idxs, insvc, dist, resp = _park_resolve_sgen_indices(
            net, machines, park.get('distribution_method'))
        if not idxs:
            print(f"[ParkController] '{park.get('name')}': no connected sgen machines resolved; skipped")
            continue

        use_q_cap = _park_truthy(park.get('use_q_capability', True), True)
        if use_q_cap:
            n_lim = _park_apply_machine_q_capability_limits(net, idxs)
            if n_lim:
                _plog(
                    f"[ParkController] '{park.get('name')}': applied P–Q capability limits "
                    f"to {n_lim}/{len(idxs)} machines (enforce_q_lims)"
                )
            # Recompute distribution after limits are on the sgen table
            idxs, insvc, dist, resp = _park_resolve_sgen_indices(
                net, machines, park.get('distribution_method'))

        mode = str(park.get('control_mode') or 'Voltage Control')
        q_resp_same = str(park.get('q_change_response') or 'same').lower() != 'opposite'
        gen_Q_response = [1 if q_resp_same else -1] * len(idxs)

        voltage_ctrl = False
        set_point = 0.0
        input_element = 'res_bus'
        input_variable = 'vm_pu'
        input_element_index = None
        bus_idx = None

        if mode == 'Voltage Control':
            voltage_ctrl = True
            bus_name = park.get('controlled_bus') or park.get('target_bus')
            bus_idx = _park_find_by_name(net.bus, bus_name, ufn)
            if bus_idx is None and str(park.get('node_selection') or '') == 'Automatic Selection':
                # nearest high-voltage bus among machine buses
                try:
                    buses = [int(net.sgen.at[i, 'bus']) for i in idxs]
                    bus_idx = max(buses, key=lambda b: float(net.bus.at[b, 'vn_kv']))
                except Exception:
                    bus_idx = idxs and int(net.sgen.at[idxs[0], 'bus'])
            if bus_idx is None:
                print(f"[ParkController] '{park.get('name')}': controlled bus not found; skipped")
                continue
            input_element = 'res_bus'
            input_variable = 'vm_pu'
            input_element_index = bus_idx
            set_point = safe_float(park.get('vm_set_pu'), 1.0)
            if str(park.get('uset_mode') or '') == 'bus target voltage':
                # Prefer bus.vn or existing vm if column present — keep explicit vm_set_pu
                pass

        elif mode == 'Reactive Power Control':
            voltage_ctrl = False
            boundary = park.get('control_q_at') or ''
            input_element, input_variable, input_element_index = _park_resolve_control_q_at(net, boundary)
            if input_element is None:
                print(f"[ParkController] '{park.get('name')}': Control Q at '{boundary}' not found; skipped")
                continue
            q_type = str(park.get('q_control_type') or 'Const. Q')
            if q_type == 'Const. Q':
                set_point = safe_float(park.get('q_set_mvar'), 0.0)
            elif q_type.startswith('Q(V)'):
                vm = 1.0
                cb = _park_find_by_name(net.bus, park.get('controlled_bus') or park.get('target_bus') or boundary, ufn)
                if cb is not None and hasattr(net, 'res_bus') and not net.res_bus.empty and cb in net.res_bus.index:
                    vm = float(net.res_bus.at[cb, 'vm_pu'])
                set_point = _park_interp_xy(_park_parse_json_list(park.get('qv_characteristic_json')),
                                           'vm_pu', 'q_mvar', vm)
                if set_point is None:
                    set_point = 0.0
            else:  # Q(P)
                p_m, _q_m = _park_measure_p_q_at(net, boundary)
                set_point = _park_interp_xy(_park_parse_json_list(park.get('qp_characteristic_json')),
                                           'p_mw', 'q_mvar', p_m if p_m is not None else 0.0)
                if set_point is None:
                    set_point = 0.0

        elif mode == 'Power Factor Control':
            voltage_ctrl = False
            boundary = park.get('control_q_at') or ''
            input_element, input_variable, input_element_index = _park_resolve_control_q_at(net, boundary)
            if input_element is None:
                print(f"[ParkController] '{park.get('name')}': Control Q at '{boundary}' not found; skipped")
                continue
            p_m, q_m = _park_measure_p_q_at(net, boundary)
            p_use = p_m if p_m is not None else sum(float(net.sgen.at[i, 'p_mw']) for i in idxs)
            pf_type = str(park.get('pf_control_type') or 'Const. cosphi')
            if pf_type == 'Const. cosphi':
                cos_phi = safe_float(park.get('cos_phi'), 1.0)
                set_point = _park_q_from_cosphi(p_use, cos_phi)
            elif pf_type.startswith('cosphi(P)'):
                branch = _park_pick_cosphi_p_branch(park, q_m)
                cos_phi = _park_interp_xy(_park_cosphi_p_points(park, branch),
                                          'p_mw', 'cos_phi', abs(p_use) if p_use is not None else 0.0)
                if cos_phi is None:
                    cos_phi = 1.0
                q_mag = abs(_park_q_from_cosphi(abs(p_use) if p_use is not None else 0.0, cos_phi))
                set_point = q_mag if branch == 'oe' else -q_mag
            else:
                # Prefer V at Control Q bus when boundary is a busbar
                cb = _park_find_by_name(
                    net.bus,
                    park.get('controlled_bus') or park.get('target_bus') or boundary,
                    ufn,
                )
                vm = 1.0
                if cb is not None and hasattr(net, 'res_bus') and cb in net.res_bus.index:
                    vm = float(net.res_bus.at[cb, 'vm_pu'])
                cos_phi = _park_interp_xy(_park_parse_json_list(park.get('cosphi_v_characteristic_json')),
                                          'vm_pu', 'cos_phi', vm)
                if cos_phi is None:
                    cos_phi = 1.0
                set_point = _park_q_from_cosphi(p_use, cos_phi)

        elif mode == 'tan(phi) Control':
            voltage_ctrl = False
            boundary = park.get('control_q_at') or ''
            input_element, input_variable, input_element_index = _park_resolve_control_q_at(net, boundary)
            if input_element is None:
                print(f"[ParkController] '{park.get('name')}': Control Q at '{boundary}' not found; skipped")
                continue
            p_m, _q_m = _park_measure_p_q_at(net, boundary)
            p_use = p_m if p_m is not None else sum(float(net.sgen.at[i, 'p_mw']) for i in idxs)
            set_point = float(p_use) * safe_float(park.get('tan_phi'), 0.0)
        else:
            print(f"[ParkController] '{park.get('name')}': unknown mode {mode!r}; skipped")
            continue

        # Clamp plant Q setpoint to sum of machine P–Q capability at current P
        if use_q_cap and not voltage_ctrl:
            qmin_sum = 0.0
            qmax_sum = 0.0
            have_lim = False
            for i in idxs:
                q_mi, q_ma = _park_machine_q_limits(net, i)
                if q_mi is None or q_ma is None:
                    continue
                have_lim = True
                qmin_sum += float(q_mi)
                qmax_sum += float(q_ma)
            if have_lim:
                sp0 = float(set_point)
                set_point = min(max(sp0, qmin_sum), qmax_sum)
                if abs(set_point - sp0) > 1e-6:
                    print(
                        f"[ParkController] '{park.get('name')}': Q set_point {sp0:.4f} → {set_point:.4f} Mvar "
                        f"(clamped to plant capability [{qmin_sum:.4f}, {qmax_sum:.4f}])"
                    )

        try:
            bsc = BinarySearchControl(
                net,
                True,
                'sgen',
                'q_mvar',
                idxs,
                insvc,
                dist,
                input_element,
                input_variable,
                input_element_index,
                float(set_point),
                voltage_ctrl,
                name=str(park.get('name') or 'ParkController'),
                gen_Q_response=gen_Q_response,
                bus_idx=bus_idx if voltage_ctrl else None,
                tol=0.001,
                in_service=True,
                order=0,
                level=0,
            )
            # Locate controller index for droop chaining
            ctrl_idx = None
            droop_spec = None
            if hasattr(net, 'controller') and not net.controller.empty:
                for ci in net.controller.index:
                    if net.controller.at[ci, 'object'] is bsc:
                        ctrl_idx = ci
                        break

            if voltage_ctrl and _park_truthy(park.get('enable_droop'), False) and ctrl_idx is not None:
                droop_pct = safe_float(park.get('droop_percent'), 0.0)
                q_rated = safe_float(park.get('q_rated_mvar'), 0.0)
                if droop_pct > 1e-9 and q_rated > 1e-9:
                    q_droop_mvar = q_rated * 100.0 / droop_pct  # Mvar / p.u.
                    droop_spec = {
                        'q_droop_mvar': float(q_droop_mvar),
                        'bus_idx': bus_idx,
                        'vm_set_pu_bsc': float(set_point),
                        'name': str(park.get('name') or 'ParkController') + '_droop',
                    }
                    DroopControl(
                        net,
                        q_droop_mvar,
                        bus_idx,
                        ctrl_idx,
                        True,
                        tol=1e-6,
                        vm_set_pu_bsc=float(set_point),
                        name=str(park.get('name') or 'ParkController') + '_droop',
                        in_service=True,
                        order=-1,
                        level=0,
                    )
            if not hasattr(net, '_electrisim_export_park_specs'):
                net._electrisim_export_park_specs = []
            net._electrisim_export_park_specs.append({
                'name': str(park.get('name') or 'ParkController'),
                'control_mode': mode,
                'pf_control_type': str(park.get('pf_control_type') or ''),
                'cosphi_p_excitation': str(park.get('cosphi_p_excitation') or ''),
                'distribution_method': str(park.get('distribution_method') or ''),
                'idxs': [int(x) for x in idxs],
                'insvc': [bool(x) for x in insvc],
                'dist': [float(x) for x in dist],
                'input_element': str(input_element),
                'input_variable': str(input_variable),
                'input_element_index': input_element_index,
                'set_point': float(set_point),
                'voltage_ctrl': bool(voltage_ctrl),
                'gen_Q_response': [int(x) for x in gen_Q_response],
                'bus_idx': bus_idx,
                'controller_idx': int(ctrl_idx) if ctrl_idx is not None else None,
                'droop': droop_spec,
            })
            attached += 1
            _plog(f"[ParkController] attached '{park.get('name')}' mode={mode} setpoint={set_point} sgens={idxs}")
        except Exception as ex:
            print(f"[ParkController] attach failed for '{park.get('name')}': {ex}")

    return attached



def powerflow(net, algorithm, calculate_voltage_angles, init, export_python=False, in_data=None, Busbars=None,
              run_control_trafo2w=False, run_control_trafo3w=False, run_control_shunt=False):
            #pandapower - rozpływ mocy
            # Initialize tap_control_results before try block so it's accessible in else block
            tap_control_results = []
            shunt_control_results = []
            controller_fallback_warning = None

            # Redirect stdout/stderr to a safe UTF-8 buffer during power flow
            # to prevent UnicodeEncodeError on Windows (cp1252 can't handle emoji/Unicode
            # characters that pandapower or our code may print)
            import io as _io
            _orig_stdout = sys.stdout
            _orig_stderr = sys.stderr
            _safe_buf = _io.StringIO()
            sys.stdout = _safe_buf
            sys.stderr = _safe_buf
            
            try:
                # Check for isolated buses before running power flow
                isolated = isolated_buses_message(net)
                if isolated:
                    raise ValueError(isolated)
                
                # DiscreteTapControl + DiscreteShuntController (per-family flags from UI)
                rc2 = bool(run_control_trafo2w)
                rc3 = bool(run_control_trafo3w)
                rcs = bool(run_control_shunt)
                tc2_list = getattr(net, 'trafo_discrete_tap_controllers', None) or []
                tc3_list = getattr(net, 'trafo3w_discrete_tap_controllers', None) or []
                shunt_ctrl_list = getattr(net, 'shunt_discrete_controllers', None) or []
                lf_shunt_list = getattr(net, 'line_flow_shunt_controllers', None) or []
                attach_2w = rc2 and bool(tc2_list)
                attach_3w = rc3 and bool(tc3_list)
                attach_sh_disc = rcs and bool(shunt_ctrl_list)
                attach_lf_sh = rcs and bool(lf_shunt_list)
                # BinarySearchControl writes straight into net.sgen.q_mvar, and attaching the park
                # controllers already runs a power flow. Snapshot the diagram/capability-curve Q
                # first: pandapower's enforce_q_lims only bounds net.gen, so a controller that fails
                # to settle can leave net.sgen.q_mvar far outside the machine capability curve and
                # every later solve attempt would inherit those values.
                initial_sgen_q = {}
                try:
                    if hasattr(net, 'sgen') and not net.sgen.empty and 'q_mvar' in net.sgen.columns:
                        initial_sgen_q = {int(i): float(net.sgen.at[i, 'q_mvar']) for i in net.sgen.index}
                except Exception:
                    initial_sgen_q = {}

                # ParkController is an explicit diagram element with its own enable toggle, so it is
                # not gated on the run_control_* checkboxes (those cover tap/shunt control only).
                park_attached = 0
                try:
                    park_attached = _electrisim_attach_park_controllers(
                        net, in_data, algorithm=algorithm,
                        calculate_voltage_angles=calculate_voltage_angles, init=init
                    )
                except Exception as park_ex:
                    print(f"[ParkController] attach error: {park_ex}")
                    park_attached = 0
                run_pp_control = attach_2w or attach_3w or attach_sh_disc or attach_lf_sh or bool(park_attached)
                if run_pp_control:
                    print(
                        f"Controllers active: 2w_tap={attach_2w} ({len(tc2_list)} configured), "
                        f"3w_tap={attach_3w} ({len(tc3_list)} configured), "
                        f"shunt_DiscreteShunt={attach_sh_disc} ({len(shunt_ctrl_list)} configured), "
                        f"shunt_line_P={attach_lf_sh} ({len(lf_shunt_list)} configured), "
                        f"park={park_attached}"
                    )
                    if attach_2w:
                        for (trafo_idx, control_side, vm_lower_pu, vm_upper_pu) in tc2_list:
                            try:
                                trafo_name = net.trafo.at[trafo_idx, 'name']
                                tap_side = net.trafo.at[trafo_idx, 'tap_side'] if 'tap_side' in net.trafo.columns else 'hv'
                                tap_min = net.trafo.at[trafo_idx, 'tap_min']
                                tap_max = net.trafo.at[trafo_idx, 'tap_max']
                                tap_step_percent = net.trafo.at[trafo_idx, 'tap_step_percent']
                                tap_pos = net.trafo.at[trafo_idx, 'tap_pos']
                                print(f"  Transformer {trafo_idx} ({trafo_name}): tap_side={tap_side}, control_side={control_side}, range=[{tap_min}, {tap_max}], step={tap_step_percent}%, pos={tap_pos}")
                                print(f"    Control limits: vm_lower={vm_lower_pu} pu, vm_upper={vm_upper_pu} pu")
                                if tap_min >= tap_max:
                                    print(f"    WARNING: tap_min ({tap_min}) >= tap_max ({tap_max}) - controller will not work!")
                                if tap_step_percent == 0:
                                    print(f"    WARNING: tap_step_percent is 0 - controller will not work!")
                                if tap_pos == tap_max:
                                    print(f"    WARNING: Initial tap_pos ({tap_pos}) is at MAXIMUM - controller cannot increase tap further!")
                                elif tap_pos == tap_min:
                                    print(f"    WARNING: Initial tap_pos ({tap_pos}) is at MINIMUM - controller cannot decrease tap further!")
                            except Exception:
                                pass
                    if attach_3w:
                        for (t3_idx, control_side, vm_lower_pu, vm_upper_pu) in tc3_list:
                            try:
                                t3_name = net.trafo3w.at[t3_idx, 'name']
                                tap_side = net.trafo3w.at[t3_idx, 'tap_side'] if 'tap_side' in net.trafo3w.columns else 'hv'
                                tap_min = net.trafo3w.at[t3_idx, 'tap_min']
                                tap_max = net.trafo3w.at[t3_idx, 'tap_max']
                                tap_step_percent = net.trafo3w.at[t3_idx, 'tap_step_percent']
                                tap_pos = net.trafo3w.at[t3_idx, 'tap_pos']
                                print(f"  Trafo3w {t3_idx} ({t3_name}): tap_side={tap_side}, control_side={control_side}, range=[{tap_min}, {tap_max}], step={tap_step_percent}%, pos={tap_pos}")
                                print(f"    Control limits: vm_lower={vm_lower_pu} pu, vm_upper={vm_upper_pu} pu")
                                if tap_min >= tap_max:
                                    print(f"    WARNING: tap_min ({tap_min}) >= tap_max ({tap_max}) - controller will not work!")
                                if tap_step_percent == 0:
                                    print(f"    WARNING: tap_step_percent is 0 - controller will not work!")
                                if tap_pos == tap_max:
                                    print(f"    WARNING: Initial tap_pos ({tap_pos}) is at MAXIMUM - controller cannot increase tap further!")
                                elif tap_pos == tap_min:
                                    print(f"    WARNING: Initial tap_pos ({tap_pos}) is at MINIMUM - controller cannot decrease tap further!")
                            except Exception:
                                pass
                    if attach_sh_disc:
                        for spec in shunt_ctrl_list:
                            try:
                                si = int(spec['shunt_index'])
                                sname = net.shunt.at[si, 'name']
                                st = net.shunt.at[si, 'step']
                                mx = net.shunt.at[si, 'max_step']
                                print(f"  Shunt {si} ({sname}): step={st}, max_step={mx}, vm_set_pu={spec.get('vm_set_pu')}, increment={spec.get('increment', 1)}")
                                if mx is not None and float(mx) < 1:
                                    print(f"    WARNING: max_step ({mx}) < 1 — controller cannot change reactive capability")
                            except Exception:
                                pass
                    if attach_lf_sh:
                        for spec in lf_shunt_list:
                            try:
                                si = int(spec['shunt_index'])
                                ln = int(spec['line_index'])
                                sname = net.shunt.at[si, 'name']
                                lname = net.line.at[ln, 'name'] if 'name' in net.line.columns else ln
                                print(f"  Line P→shunt step: shunt {si} ({sname}) ← line {ln} ({lname}), "
                                      f"use {spec.get('p_col')}, use_abs={spec.get('use_abs')}")
                            except Exception:
                                pass
                    if attach_2w or attach_3w:
                        _electrisim_attach_discrete_tap_controllers(net, attach_trafo=attach_2w, attach_trafo3w=attach_3w)
                    if attach_sh_disc:
                        _electrisim_attach_discrete_shunt_controllers(net)
                    if attach_lf_sh:
                        _electrisim_attach_line_flow_shunt_controllers(net)
                
                # Snapshot tap positions before runpp whenever controllers may run (for UI / diagnostics)
                initial_tap_positions = {}
                initial_tap3w_positions = {}
                initial_shunt_steps = {}
                if run_pp_control:
                    if attach_2w:
                        for (trafo_idx, _, _, _) in tc2_list:
                            try:
                                initial_tap_positions[trafo_idx] = float(net.trafo.at[trafo_idx, 'tap_pos'])
                            except Exception:
                                pass
                    if attach_3w:
                        for (t3_idx, _, _, _) in tc3_list:
                            try:
                                initial_tap3w_positions[t3_idx] = float(net.trafo3w.at[t3_idx, 'tap_pos'])
                            except Exception:
                                pass
                    if attach_sh_disc:
                        for spec in shunt_ctrl_list:
                            try:
                                si = int(spec['shunt_index'])
                                initial_shunt_steps[si] = float(net.shunt.at[si, 'step'])
                            except Exception:
                                pass
                    if attach_lf_sh:
                        for spec in lf_shunt_list:
                            try:
                                si = int(spec['shunt_index'])
                                initial_shunt_steps[si] = float(net.shunt.at[si, 'step'])
                            except Exception:
                                pass

                # Log controller status before power flow
                if run_pp_control and hasattr(net, 'controller') and not net.controller.empty:
                    print(f"Running power flow WITH controllers (run_control=True)")
                    print(f"   Controllers in net: {len(net.controller)}")
                    print(net.controller[['object', 'in_service']])
                    print(f"   Initial tap positions (2w): {initial_tap_positions}")
                    print(f"   Initial tap positions (3w): {initial_tap3w_positions}")
                    print(f"   Initial shunt steps: {initial_shunt_steps}")
                else:
                    print(f"Running power flow WITHOUT controllers (run_pp_control={run_pp_control})")
                
                def _restore_controller_setpoints():
                    for idx, pos in initial_tap_positions.items():
                        try:
                            net.trafo.at[idx, 'tap_pos'] = pos
                        except Exception:
                            pass
                    for idx, pos in initial_tap3w_positions.items():
                        try:
                            net.trafo3w.at[idx, 'tap_pos'] = pos
                        except Exception:
                            pass
                    for si, s0 in initial_shunt_steps.items():
                        try:
                            net.shunt.at[si, 'step'] = s0
                        except Exception:
                            pass
                    for si, q0 in initial_sgen_q.items():
                        try:
                            net.sgen.at[si, 'q_mvar'] = q0
                        except Exception:
                            pass

                pf_kwargs = _electrisim_enforce_q_lims_kw(net)
                facts_present = _electrisim_net_has_facts(net)
                if facts_present and str(algorithm) != 'nr':
                    raise NotImplementedError(
                        "The STATCOM (SSC) and other FACTS devices require the Newton-Raphson algorithm. "
                        f"Selected algorithm: '{algorithm}'."
                    )
                if facts_present:
                    _electrisim_validate_ssc(net)

                # A heavily compensated cable network needs more than the 10 Newton iterations
                # pandapower allows by default, so work through progressively more robust solver
                # settings rather than reporting failure after the first divergence. Later plans
                # drop the controllers: DiscreteTapControl on this farm often cannot reach 0.99 pu
                # on 0.69 kV buses (tap range too small) and then Newton-Raphson diverges.
                requested_label = f"{algorithm}, init={init}"
                solve_plans = [(requested_label, {'algorithm': algorithm, 'init': init}, run_pp_control)]
                if run_pp_control:
                    solve_plans.append(
                        (f"{requested_label}, no controllers", {'algorithm': algorithm, 'init': init}, False)
                    )
                solve_plans.append(
                    (f"{algorithm}, init={init}, max_iteration=50",
                     {'algorithm': algorithm, 'init': init, 'max_iteration': 50}, False)
                )
                solve_plans.append(
                    (f"{algorithm}, init=flat, max_iteration=100",
                     {'algorithm': algorithm, 'init': 'flat', 'max_iteration': 100}, False)
                )
                # Iwamoto's step-size multiplier is built for ill-conditioned cases that plain
                # Newton-Raphson overshoots. Pandapower implements FACTS only for algorithm='nr',
                # so an in-service STATCOM/SSC stays on Newton-Raphson.
                if not facts_present:
                    solve_plans.append(
                        ("iwamoto_nr, init=flat, max_iteration=100",
                         {'algorithm': 'iwamoto_nr', 'init': 'flat', 'max_iteration': 100}, False)
                    )

                pf_plan_used = None
                pf_last_error = None
                pf_attempt_log = []
                for plan_label, plan_kwargs, plan_run_control in solve_plans:
                    if not plan_run_control and run_pp_control:
                        _restore_controller_setpoints()
                        if hasattr(net, 'controller') and not net.controller.empty:
                            try:
                                net.controller['in_service'] = False
                            except Exception:
                                pass
                    try:
                        _electrisim_runpp(net, calculate_voltage_angles=calculate_voltage_angles,
                                          run_control=plan_run_control, **plan_kwargs, **pf_kwargs)
                        pf_plan_used = plan_label
                        break
                    except Exception as plan_err:
                        pf_last_error = plan_err
                        detail = f"{plan_label}: {type(plan_err).__name__}: {plan_err}"
                        pf_attempt_log.append(detail)
                        print(f"Power flow attempt failed [{detail}]")

                if pf_plan_used is None:
                    if facts_present and pf_attempt_log:
                        ssc_desc = ""
                        ssc_df = getattr(net, 'ssc', None)
                        if ssc_df is not None and not ssc_df.empty:
                            bits = []
                            for _, srow in ssc_df.iterrows():
                                bits.append(
                                    f"{srow.get('name')}: r_ohm={srow.get('r_ohm')} x_ohm={srow.get('x_ohm')} "
                                    f"set_vm_pu={srow.get('set_vm_pu')} vm_internal_pu={srow.get('vm_internal_pu')} "
                                    f"bus={srow.get('bus')}"
                                )
                            ssc_desc = " SSC data: " + "; ".join(bits) + "."
                        ssc_diag = _electrisim_diagnose_ssc_failure(net, calculate_voltage_angles)
                        raise RuntimeError(
                            "Load flow with the STATCOM (SSC) did not converge with Newton-Raphson. "
                            "FACTS devices stay on Newton-Raphson. "
                            "Attempts: " + " | ".join(pf_attempt_log) + "." + ssc_desc +
                            (" " + ssc_diag if ssc_diag else "")
                        ) from pf_last_error
                    raise pf_last_error

                # A park controller that never settled leaves machines outside their capability
                # curve; report it instead of returning those values as a valid operating point.
                park_q_violations = []
                try:
                    if park_attached and hasattr(net, 'sgen') and not net.sgen.empty:
                        for si in net.sgen.index:
                            q = float(net.sgen.at[si, 'q_mvar'])
                            q_mi = net.sgen.at[si, 'min_q_mvar'] if 'min_q_mvar' in net.sgen.columns else None
                            q_ma = net.sgen.at[si, 'max_q_mvar'] if 'max_q_mvar' in net.sgen.columns else None
                            if q_mi is None or q_ma is None or q_mi != q_mi or q_ma != q_ma:
                                continue
                            # Voltage-dependent curves shift by a few kVAr between the setpoint
                            # evaluation and the solved voltage. Only flag a real excursion.
                            q_tol = max(0.05, 0.02 * max(abs(float(q_mi)), abs(float(q_ma)), 1.0))
                            if q < float(q_mi) - q_tol or q > float(q_ma) + q_tol:
                                sname = net.sgen.at[si, 'name'] if 'name' in net.sgen.columns else si
                                ufn = getattr(net, 'user_friendly_names', {}) or {}
                                park_q_violations.append(
                                    f"{ufn.get(sname, sname)}: {q:.3f} Mvar "
                                    f"(curve allows {float(q_mi):.3f} to {float(q_ma):.3f})"
                                )
                except Exception:
                    park_q_violations = []
                if park_q_violations:
                    msg = (
                        "The Park controller drove static generators outside their reactive capability "
                        "curve: " + "; ".join(park_q_violations) + ". The controller could not reach its "
                        "setpoint, most often because another voltage-controlling element (STATCOM/SSC, "
                        "generator, or tap changer) regulates the same bus. Results are not a valid "
                        "operating point until the controller setpoint or the machine capability is changed."
                    )
                    print(msg)
                    controller_fallback_warning = (
                        msg if not controller_fallback_warning
                        else controller_fallback_warning + " " + msg
                    )

                if pf_plan_used != solve_plans[0][0]:
                    print(f"Power flow converged with fallback settings [{pf_plan_used}]")
                    fallback_msg = (
                        f"Load flow did not converge with the requested settings ({requested_label}) "
                        f"and succeeded with [{pf_plan_used}]. Failed attempts: "
                        + " | ".join(pf_attempt_log)
                    )
                    controller_fallback_warning = (
                        fallback_msg if not controller_fallback_warning
                        else controller_fallback_warning + " " + fallback_msg
                    )
                
                # Check if tap positions changed
                if run_pp_control and (initial_tap_positions or initial_tap3w_positions):
                    changed = False
                    for idx, initial_pos in initial_tap_positions.items():
                        final_pos = net.trafo.at[idx, 'tap_pos']
                        if initial_pos != final_pos:
                            print(f"   Tap changed: Trafo {idx}: {initial_pos} -> {final_pos}")
                            changed = True
                    for idx, initial_pos in initial_tap3w_positions.items():
                        final_pos = net.trafo3w.at[idx, 'tap_pos']
                        if initial_pos != final_pos:
                            print(f"   Tap changed: Trafo3w {idx}: {initial_pos} -> {final_pos}")
                            changed = True
                    if not changed and ((attach_2w and tc2_list) or (attach_3w and tc3_list)):
                        print(f"   WARNING: No tap positions changed during controlled power flow!")
                if (attach_sh_disc or attach_lf_sh) and initial_shunt_steps:
                    sh_changed = False
                    for si, s0 in initial_shunt_steps.items():
                        s1 = float(net.shunt.at[si, 'step'])
                        if s0 != s1:
                            print(f"   Shunt step changed: Shunt {si}: {s0} -> {s1}")
                            sh_changed = True
                    if not sh_changed and (shunt_ctrl_list or lf_shunt_list):
                        print(f"   WARNING: No shunt step positions changed during controlled power flow!")
                
                # Log transformer tap positions after power flow (only for controlled transformers)
                # Also build tap_control_results for frontend display
                
                if attach_2w and not net.trafo.empty and getattr(net, 'trafo_discrete_tap_controllers', None):
                    # Get list of transformer indices that have controllers
                    controlled_trafo_indices = set(ctrl_data[0] for ctrl_data in net.trafo_discrete_tap_controllers)
                    
                    print(f"Controlled transformer tap positions after power flow:")
                    for idx in net.trafo.index:
                        if idx not in controlled_trafo_indices:
                            continue  # Skip transformers without controllers
                        
                        name = net.trafo.at[idx, 'name']
                        tap_pos = net.trafo.at[idx, 'tap_pos']
                        tap_min = net.trafo.at[idx, 'tap_min']
                        tap_max = net.trafo.at[idx, 'tap_max']
                        tap_step = net.trafo.at[idx, 'tap_step_percent']
                        hv_bus_idx = net.trafo.at[idx, 'hv_bus']
                        lv_bus_idx = net.trafo.at[idx, 'lv_bus']
                        hv_vm_pu = net.res_bus.at[hv_bus_idx, 'vm_pu']
                        lv_vm_pu = net.res_bus.at[lv_bus_idx, 'vm_pu']
                        
                        # Get user-friendly name if available
                        user_friendly_name = net.user_friendly_names.get(name, name) if hasattr(net, 'user_friendly_names') else name
                        
                        # Get control limits for this transformer
                        ctrl_data = next((c for c in net.trafo_discrete_tap_controllers if c[0] == idx), None)
                        if ctrl_data:
                            ctrl_side, vm_lower, vm_upper = ctrl_data[1], ctrl_data[2], ctrl_data[3]
                            controlled_vm = lv_vm_pu if ctrl_side == 'lv' else hv_vm_pu
                            
                            # Check if voltage is within limits
                            in_limits = vm_lower <= controlled_vm <= vm_upper
                            status = "IN LIMITS" if in_limits else "OUT OF LIMITS"
                            
                            # Check if tap is at limit
                            at_limit = ""
                            at_limit_type = None
                            if tap_pos == tap_max:
                                at_limit = " (AT MAX LIMIT - cannot increase further)"
                                at_limit_type = "max"
                            elif tap_pos == tap_min:
                                at_limit = " (AT MIN LIMIT - cannot decrease further)"
                                at_limit_type = "min"
                            
                            print(f"   Trafo {idx} ({name}):")
                            print(f"      Tap: {tap_pos} [{tap_min}, {tap_max}] step={tap_step}%{at_limit}")
                            print(f"      Control side ({ctrl_side}): {controlled_vm:.4f} pu  Target: [{vm_lower}, {vm_upper}] pu  {status}")
                            print(f"      HV bus: {hv_vm_pu:.4f} pu, LV bus: {lv_vm_pu:.4f} pu")
                            
                            # Calculate how much more tap range would be needed
                            taps_needed = None
                            if not in_limits and (tap_pos == tap_max or tap_pos == tap_min):
                                voltage_gap = abs(controlled_vm - vm_upper) if controlled_vm > vm_upper else abs(vm_lower - controlled_vm)
                                taps_needed = int(voltage_gap / (tap_step / 100) / controlled_vm) + 1
                                print(f"      Need ~{taps_needed} more tap positions OR increase tap_step_percent to reach target")
                            
                            # Build result object for frontend
                            # Convert numpy types to native Python types for JSON serialization
                            trafo_cell_id = None
                            if 'id' in net.trafo.columns:
                                try:
                                    trafo_cell_id = str(net.trafo.at[idx, 'id'])
                                except Exception:
                                    trafo_cell_id = None
                            tap_pos_initial = float(initial_tap_positions[idx]) if idx in initial_tap_positions else float(tap_pos)
                            tap_control_results.append({
                                'name': str(user_friendly_name),
                                'id': str(name),
                                'cell_id': trafo_cell_id,
                                'tap_pos_initial': tap_pos_initial,
                                'tap_pos': float(tap_pos),
                                'tap_min': float(tap_min),
                                'tap_max': float(tap_max),
                                'tap_step_percent': float(tap_step),
                                'control_side': str(ctrl_side),
                                'controlled_vm_pu': round(float(controlled_vm), 4),
                                'vm_lower_pu': float(vm_lower),
                                'vm_upper_pu': float(vm_upper),
                                'hv_vm_pu': round(float(hv_vm_pu), 4),
                                'lv_vm_pu': round(float(lv_vm_pu), 4),
                                'in_limits': bool(in_limits),  # Convert numpy.bool_ to Python bool
                                'at_limit': at_limit_type,
                                'taps_needed': int(taps_needed) if taps_needed is not None else None
                            })

                if attach_3w and hasattr(net, 'trafo3w') and not net.trafo3w.empty and getattr(net, 'trafo3w_discrete_tap_controllers', None):
                    controlled_t3_indices = set(c[0] for c in net.trafo3w_discrete_tap_controllers)
                    print(f"Controlled three-winding transformer tap positions after power flow:")
                    for idx in net.trafo3w.index:
                        if idx not in controlled_t3_indices:
                            continue
                        name = net.trafo3w.at[idx, 'name']
                        tap_pos = net.trafo3w.at[idx, 'tap_pos']
                        tap_min = net.trafo3w.at[idx, 'tap_min']
                        tap_max = net.trafo3w.at[idx, 'tap_max']
                        tap_step = net.trafo3w.at[idx, 'tap_step_percent']
                        hv_bus_idx = int(net.trafo3w.at[idx, 'hv_bus'])
                        mv_bus_idx = int(net.trafo3w.at[idx, 'mv_bus'])
                        lv_bus_idx = int(net.trafo3w.at[idx, 'lv_bus'])
                        hv_vm_pu = net.res_bus.at[hv_bus_idx, 'vm_pu']
                        mv_vm_pu = net.res_bus.at[mv_bus_idx, 'vm_pu']
                        lv_vm_pu = net.res_bus.at[lv_bus_idx, 'vm_pu']
                        user_friendly_name = net.user_friendly_names.get(name, name) if hasattr(net, 'user_friendly_names') else name
                        ctrl_data = next((c for c in net.trafo3w_discrete_tap_controllers if c[0] == idx), None)
                        if not ctrl_data:
                            continue
                        ctrl_side, vm_lower, vm_upper = ctrl_data[1], ctrl_data[2], ctrl_data[3]
                        side_bus = {'hv': hv_bus_idx, 'mv': mv_bus_idx, 'lv': lv_bus_idx}.get(str(ctrl_side), lv_bus_idx)
                        controlled_vm = net.res_bus.at[side_bus, 'vm_pu']
                        in_limits = vm_lower <= controlled_vm <= vm_upper
                        status = "IN LIMITS" if in_limits else "OUT OF LIMITS"
                        at_limit_type = None
                        if tap_pos == tap_max:
                            at_limit_type = "max"
                        elif tap_pos == tap_min:
                            at_limit_type = "min"
                        print(f"   Trafo3w {idx} ({name}):")
                        print(f"      Tap: {tap_pos} [{tap_min}, {tap_max}] step={tap_step}%")
                        print(f"      Control side ({ctrl_side}): {controlled_vm:.4f} pu  Target: [{vm_lower}, {vm_upper}] pu  {status}")
                        print(f"      HV/MV/LV bus vm_pu: {hv_vm_pu:.4f} / {mv_vm_pu:.4f} / {lv_vm_pu:.4f} pu")
                        taps_needed = None
                        if not in_limits and (tap_pos == tap_max or tap_pos == tap_min):
                            voltage_gap = abs(controlled_vm - vm_upper) if controlled_vm > vm_upper else abs(vm_lower - controlled_vm)
                            try:
                                taps_needed = int(voltage_gap / (tap_step / 100) / controlled_vm) + 1
                            except Exception:
                                taps_needed = None
                        t3_cell_id = None
                        if 'id' in net.trafo3w.columns:
                            try:
                                t3_cell_id = str(net.trafo3w.at[idx, 'id'])
                            except Exception:
                                t3_cell_id = None
                        tap_pos_initial = float(initial_tap3w_positions[idx]) if idx in initial_tap3w_positions else float(tap_pos)
                        tap_control_results.append({
                            'element': 'trafo3w',
                            'name': str(user_friendly_name),
                            'id': str(name),
                            'cell_id': t3_cell_id,
                            'tap_pos_initial': tap_pos_initial,
                            'tap_pos': float(tap_pos),
                            'tap_min': float(tap_min),
                            'tap_max': float(tap_max),
                            'tap_step_percent': float(tap_step),
                            'control_side': str(ctrl_side),
                            'controlled_vm_pu': round(float(controlled_vm), 4),
                            'vm_lower_pu': float(vm_lower),
                            'vm_upper_pu': float(vm_upper),
                            'hv_vm_pu': round(float(hv_vm_pu), 4),
                            'mv_vm_pu': round(float(mv_vm_pu), 4),
                            'lv_vm_pu': round(float(lv_vm_pu), 4),
                            'in_limits': bool(in_limits),
                            'at_limit': at_limit_type,
                            'taps_needed': int(taps_needed) if taps_needed is not None else None
                        })

                if attach_sh_disc and shunt_ctrl_list and hasattr(net, 'shunt') and not net.shunt.empty:
                    for spec in shunt_ctrl_list:
                        try:
                            si = int(spec['shunt_index'])
                            name = net.shunt.at[si, 'name']
                            bus_i = int(net.shunt.at[si, 'bus'])
                            vm_pu = float(net.res_bus.at[bus_i, 'vm_pu'])
                            step_f = float(net.shunt.at[si, 'step'])
                            max_st = float(net.shunt.at[si, 'max_step'])
                            user_friendly_name = net.user_friendly_names.get(name, name) if hasattr(net, 'user_friendly_names') else name
                            cell_id = None
                            if 'id' in net.shunt.columns:
                                try:
                                    cell_id = str(net.shunt.at[si, 'id'])
                                except Exception:
                                    cell_id = None
                            s0 = float(initial_shunt_steps[si]) if si in initial_shunt_steps else step_f
                            zb = _electrisim_shunt_is_zero_based(net, si)
                            shunt_control_results.append({
                                'element': 'shunt',
                                'control_type': 'discrete_voltage',
                                'name': str(user_friendly_name),
                                'id': str(name),
                                'cell_id': cell_id,
                                'step_initial': _electrisim_shunt_step_from_pp(s0, zb),
                                'step': _electrisim_shunt_step_from_pp(step_f, zb),
                                'max_step': _electrisim_shunt_step_from_pp(max_st, zb) if zb else max_st,
                                'vm_set_pu': float(spec.get('vm_set_pu', 1.0)),
                                'vm_pu': round(vm_pu, 4),
                                'tol': float(spec.get('tol', 1e-3)),
                                'increment': int(spec.get('increment', 1)),
                            })
                        except Exception:
                            pass
                if attach_lf_sh and lf_shunt_list and hasattr(net, 'shunt') and not net.shunt.empty and hasattr(net, 'res_line'):
                    for spec in lf_shunt_list:
                        try:
                            si = int(spec['shunt_index'])
                            ln_i = int(spec['line_index'])
                            name = net.shunt.at[si, 'name']
                            bus_i = int(net.shunt.at[si, 'bus'])
                            vm_pu = float(net.res_bus.at[bus_i, 'vm_pu'])
                            step_f = float(net.shunt.at[si, 'step'])
                            max_st = float(net.shunt.at[si, 'max_step'])
                            user_friendly_name = net.user_friendly_names.get(name, name) if hasattr(net, 'user_friendly_names') else name
                            cell_id = None
                            if 'id' in net.shunt.columns:
                                try:
                                    cell_id = str(net.shunt.at[si, 'id'])
                                except Exception:
                                    cell_id = None
                            s0 = float(initial_shunt_steps[si]) if si in initial_shunt_steps else step_f
                            pcol = spec.get('p_col') or 'p_from_mw'
                            try:
                                raw_p = float(net.res_line.at[ln_i, pcol])
                            except Exception:
                                raw_p = float('nan')
                            use_abs_pf = bool(spec.get('use_abs', True))
                            display_p = abs(raw_p) if use_abs_pf and raw_p == raw_p else raw_p
                            zb = _electrisim_shunt_is_zero_based(net, si)
                            shunt_control_results.append({
                                'element': 'shunt',
                                'control_type': 'line_flow',
                                'name': str(user_friendly_name),
                                'id': str(name),
                                'cell_id': cell_id,
                                'step_initial': _electrisim_shunt_step_from_pp(s0, zb),
                                'step': _electrisim_shunt_step_from_pp(step_f, zb),
                                'max_step': _electrisim_shunt_step_from_pp(max_st, zb) if zb else max_st,
                                'vm_pu': round(vm_pu, 4),
                                'line_p_mw_used': round(float(display_p), 6) if display_p == display_p else None,
                                'line_p_column': str(pcol),
                            })
                        except Exception:
                            pass
                
            except Exception as e:
                # Restore stdout/stderr before handling error
                sys.stdout = _orig_stdout
                sys.stderr = _orig_stderr

                # stdout was captured during the solve, so without this the server log shows only
                # the diagnostic dump and the real cause is visible in the HTTP response alone.
                import traceback
                print(f"[pandapower] Power flow failed: {type(e).__name__}: {e}")
                traceback.print_exc()

                # Initialize diagnostic response
                diagnostic_response = {
                    "error": True,
                    "message": "Power flow calculation failed",
                    "exception": str(e),
                    "diagnostic": {}
                }               
                
                # Check for disconnected sections by finding unsupplied buses (buses not connected to ext_grid)
                # This is the same as isolated buses - buses without connection to external grid
                try:
                    # Get all buses that are not supplied (disconnected from external grid)
                    unsupplied_buses_set = pp.topology.unsupplied_buses(net)
                    if len(unsupplied_buses_set) > 0:
                        # Convert set to list and ensure all values are native Python int (not numpy int64)
                        if isinstance(unsupplied_buses_set, set):
                            unsupplied_buses_list = [int(x) for x in unsupplied_buses_set]
                        elif hasattr(unsupplied_buses_set, 'tolist'):
                            unsupplied_buses_list = [int(x) for x in unsupplied_buses_set.tolist()]
                        else:
                            unsupplied_buses_list = [int(x) for x in list(unsupplied_buses_set)]

                        trafo_idxs = []
                        line_idxs = []
                        sgen_idxs = []
                        load_idxs = []
                        gen_idxs = []

                        # Find transformers connected to unsupplied buses
                        if hasattr(net, 'trafo') and not net.trafo.empty:
                            for trafo_idx in net.trafo.index:
                                hv_bus = net.trafo.loc[trafo_idx, 'hv_bus']
                                lv_bus = net.trafo.loc[trafo_idx, 'lv_bus']
                                if hv_bus in unsupplied_buses_set or lv_bus in unsupplied_buses_set:
                                    trafo_idxs.append(int(trafo_idx))

                        # Find lines connected to unsupplied buses
                        if hasattr(net, 'line') and not net.line.empty:
                            for line_idx in net.line.index:
                                from_bus = net.line.loc[line_idx, 'from_bus']
                                to_bus = net.line.loc[line_idx, 'to_bus']
                                if from_bus in unsupplied_buses_set or to_bus in unsupplied_buses_set:
                                    line_idxs.append(int(line_idx))

                        # Find static generators connected to unsupplied buses
                        if hasattr(net, 'sgen') and not net.sgen.empty:
                            for sgen_idx in net.sgen.index:
                                if net.sgen.loc[sgen_idx, 'bus'] in unsupplied_buses_set:
                                    sgen_idxs.append(int(sgen_idx))

                        # Find loads connected to unsupplied buses
                        if hasattr(net, 'load') and not net.load.empty:
                            for load_idx in net.load.index:
                                if net.load.loc[load_idx, 'bus'] in unsupplied_buses_set:
                                    load_idxs.append(int(load_idx))

                        # Find generators connected to unsupplied buses
                        if hasattr(net, 'gen') and not net.gen.empty:
                            for gen_idx in net.gen.index:
                                if net.gen.loc[gen_idx, 'bus'] in unsupplied_buses_set:
                                    gen_idxs.append(int(gen_idx))

                        # Resolve indices to frontend names/ids for dialog + canvas highlight
                        disconnected_elements = {
                            "buses": resolve_element_refs(net, 'bus', unsupplied_buses_list),
                            "lines": resolve_element_refs(net, 'line', line_idxs),
                            "trafos": resolve_element_refs(net, 'trafo', trafo_idxs),
                            "sgens": resolve_element_refs(net, 'sgen', sgen_idxs),
                            "loads": resolve_element_refs(net, 'load', load_idxs),
                            "generators": resolve_element_refs(net, 'gen', gen_idxs),
                        }

                        total_disconnected = (
                            len(disconnected_elements["buses"])
                            + len(disconnected_elements["lines"])
                            + len(disconnected_elements["trafos"])
                            + len(disconnected_elements["sgens"])
                            + len(disconnected_elements["loads"])
                            + len(disconnected_elements["generators"])
                        )

                        diagnostic_response["diagnostic"]["disconnected_elements"] = disconnected_elements
                        diagnostic_response["diagnostic"]["total_disconnected_elements"] = int(total_disconnected)
                except Exception as disconn_error:
                    # If detection fails, continue with other diagnostics
                    pass
                
                # Check for isolated buses (buses without connection to external grid)
                try:
                    isolated_buses = pp.topology.unsupplied_buses(net)
                    if len(isolated_buses) > 0:
                        isolated_refs = resolve_element_refs(net, 'bus', isolated_buses)
                        diagnostic_response["diagnostic"]["isolated_buses"] = isolated_refs
                        diagnostic_response["diagnostic"]["num_isolated_buses"] = len(isolated_refs)
                except Exception as isolated_error:
                    pass
                
                # Check if external grid exists
                if not hasattr(net, 'ext_grid') or net.ext_grid.empty:
                    diagnostic_response["diagnostic"]["no_external_grid"] = True
                    diagnostic_response["message"] = "No external grid found. At least one External Grid element is required for power flow simulation."
                
                # Access initial voltage magnitudes and angles  
                try:
                    # Capture the diagnostic output from stdout
                    import io
                    
                    captured_output = io.StringIO()
                    old_stdout = sys.stdout
                    old_stderr = sys.stderr
                    sys.stdout = captured_output
                    sys.stderr = captured_output  # Also capture stderr
                    
                    diag_result_dict = {}
                    try:
                        # Call diagnostic without report_style to get full text output
                        # The default call prints the detailed diagnostic tool output
                        diag_result_dict = pp.diagnostic(net)
                    except Exception as diag_ex:
                        captured_output.write(f"\nDiagnostic error: {str(diag_ex)}\n")
                    
                    # Restore stdout/stderr
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
                    diagnostic_text_output = captured_output.getvalue()
                    
                    # Debug: print captured output to server console
                    print(f"[DEBUG] Captured diagnostic output length: {len(diagnostic_text_output)}")
                    if diagnostic_text_output:
                        print(f"[DEBUG] Diagnostic output preview: {diagnostic_text_output[:500]}...")
                    
                    # Include the text output in the diagnostic response
                    if diagnostic_text_output and len(diagnostic_text_output.strip()) > 0:
                        diagnostic_response["diagnostic"]["diagnostic_output"] = diagnostic_text_output
                    else:
                        # If no output captured, add a note
                        diagnostic_response["diagnostic"]["diagnostic_note"] = "No detailed diagnostic output available"
                    
                    # Process diagnostic data to convert element indices to user-friendly names
                    if diag_result_dict and isinstance(diag_result_dict, dict):
                        processed_diagnostic = process_diagnostic_data(net, diag_result_dict)
                        # Merge processed diagnostic (don't overwrite disconnected_sections or isolated_buses)
                        for key, value in processed_diagnostic.items():
                            if key not in diagnostic_response["diagnostic"]:
                                diagnostic_response["diagnostic"][key] = value
                except Exception as diag_error:
                    # If diagnostic fails, continue with what we have
                    diagnostic_response["diagnostic"]["diagnostic_error"] = str(diag_error)
                    import traceback
                    diagnostic_response["diagnostic"]["diagnostic_traceback"] = traceback.format_exc()
                
                # If no specific diagnostic was found, include the original exception
                if not diagnostic_response["diagnostic"]:
                   diagnostic_response["diagnostic"]["general_error"] = str(e)
                else:
                    # Enhance the message with diagnostic summary
                    if "no_external_grid" in diagnostic_response["diagnostic"]:
                        diagnostic_response["message"] = "No external grid found. At least one External Grid element is required for power flow simulation."
                    elif "disconnected_elements" in diagnostic_response["diagnostic"]:
                        disconnected = diagnostic_response["diagnostic"]["disconnected_elements"]
                        num_buses = len(disconnected.get("buses", []))
                        num_lines = len(disconnected.get("lines", []))
                        num_trafos = len(disconnected.get("trafos", []))
                        num_sgens = len(disconnected.get("sgens", []))
                        num_loads = len(disconnected.get("loads", []))
                        total_elements = diagnostic_response["diagnostic"].get("total_disconnected_elements", 0)
                        
                        elements_summary = []
                        if num_buses > 0:
                            elements_summary.append(f"{num_buses} bus(es)")
                        if num_lines > 0:
                            elements_summary.append(f"{num_lines} line(s)")
                        if num_trafos > 0:
                            elements_summary.append(f"{num_trafos} transformer(s)")
                        if num_sgens > 0:
                            elements_summary.append(f"{num_sgens} static generator(s)")
                        if num_loads > 0:
                            elements_summary.append(f"{num_loads} load(s)")
                        
                        summary_text = ", ".join(elements_summary) if elements_summary else "elements"
                        diagnostic_response["message"] = f"Network connectivity issue: {total_elements} disconnected {summary_text} found. All network sections must be connected to an External Grid element."
                    elif "isolated_buses" in diagnostic_response["diagnostic"]:
                        num_isolated = diagnostic_response["diagnostic"].get("num_isolated_buses", 0)
                        diagnostic_response["message"] = f"Network connectivity issue: {num_isolated} isolated bus(es) found. All buses must be connected to an External Grid."
                
                # Convert diagnostic response to JSON string (same format as successful response)
                # Use a custom encoder to handle numpy types
                def convert_numpy_types(obj):
                    """Recursively convert numpy types to native Python types for JSON serialization"""
                    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                        return int(obj)
                    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {key: convert_numpy_types(value) for key, value in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [convert_numpy_types(item) for item in obj]
                    elif isinstance(obj, set):
                        return [convert_numpy_types(item) for item in obj]
                    return obj
                
                # Convert any remaining numpy types
                diagnostic_response = convert_numpy_types(diagnostic_response)
                return json.dumps(
                    _sanitize_for_strict_json(diagnostic_response),
                    separators=(',', ':'),
                    allow_nan=False,
                )
            else:
                # Restore stdout/stderr after successful power flow
                sys.stdout = _orig_stdout
                sys.stderr = _orig_stderr
                
                class BusbarOut(object):
                    def __init__(self, name: str, id: str, vm_pu: float, va_degree: float, p_mw: float, q_mvar: float, pf: float, q_p: float,
                                 p_branch_mw: float = 0.0, q_branch_mvar: float = 0.0,
                                 p_nodal_mw: float = 0.0, q_nodal_mvar: float = 0.0,
                                 pf_nodal: float = 0.0, q_p_nodal: float = 0.0,
                                 vm_kv: float = None):
                        self.name = name
                        self.id = id
                        self.vm_pu = vm_pu
                        self.va_degree = va_degree   
                        self.p_mw = p_mw
                        self.q_mvar = q_mvar  
                        self.pf = pf #p_mw/math.sqrt(math.pow(p_mw,2)+math.pow(q_mvar,2))  
                        self.q_p = q_p
                        self.p_branch_mw = p_branch_mw
                        self.q_branch_mvar = q_branch_mvar
                        self.p_nodal_mw = p_nodal_mw
                        self.q_nodal_mvar = q_nodal_mvar
                        self.pf_nodal = pf_nodal
                        self.q_p_nodal = q_p_nodal
                        self.vm_kv = vm_kv
                        
                class BusbarsOut(object):
                    def __init__(self, busbars: List[BusbarOut]):
                        self.busbars = busbars                
                
                busbarList = list() 
                
                class LineOut(object):
                    def __init__(self, name: str, id: str, p_from_mw: float, q_from_mvar: float, p_to_mw: float, q_to_mvar: float, i_from_ka: float, i_to_ka: float, loading_percent: float):          
                        self.name = name 
                        self.id = id                      
                        self.p_from_mw = p_from_mw
                        self.q_from_mvar = q_from_mvar 
                        self.p_to_mw = p_to_mw 
                        self.q_to_mvar = q_to_mvar            
                        self.i_from_ka = i_from_ka 
                        self.i_to_ka = i_to_ka               
                        self.loading_percent = loading_percent 
                       
                class LinesOut(object):
                    def __init__(self, lines: List[BusbarOut]):
                        self.lines = lines
                linesList = list()                              
                         
                class ExternalGridOut(object):
                    def __init__(self,  name: str, id: str, p_mw: float, q_mvar: float, pf: float, q_p:float):        
                        self.name = name
                        self.id = id
                        self.p_mw = p_mw 
                        self.q_mvar = q_mvar  
                        self.pf = pf            
                        self.q_p=q_p              
                       
                class ExternalGridsOut(object):
                    def __init__(self, externalgrids: List[ExternalGridOut]):
                        self.externalgrids = externalgrids              
                externalgridsList = list() 
                
                class GeneratorOut(object):
                    def __init__(self, name: str, id: str, p_mw: float, q_mvar: float, va_degree: float, vm_pu: float):          
                        self.name = name
                        self.id = id
                        self.p_mw = p_mw 
                        self.q_mvar = q_mvar  
                        self.va_degree = va_degree 
                        self.vm_pu = vm_pu                         
                       
                class GeneratorsOut(object):
                    def __init__(self, generators: List[GeneratorOut]):
                        self.generators = generators             
                generatorsList = list()
                
                
                class StaticGeneratorOut(object):
                    def __init__(self, name: str, id: str, p_mw: float, q_mvar: float):          
                        self.name = name
                        self.id = id
                        self.p_mw = p_mw 
                        self.q_mvar = q_mvar                                       
                       
                class StaticGeneratorsOut(object):
                    def __init__(self, staticgenerators: List[StaticGeneratorOut]):
                        self.staticgenerators = staticgenerators             
                staticgeneratorsList = list()
                
                
                class AsymmetricStaticGeneratorOut(object):
                    def __init__(self, name: str, id: str,  p_a_mw: float, q_a_mvar: float, p_b_mw: float, q_b_mvar: float, p_c_mw: float, q_c_mvar: float):          
                        self.name = name
                        self.id = id
                        self.p_a_mw = p_a_mw 
                        self.q_a_mvar = q_a_mvar    
                        self.p_b_mw = p_b_mw 
                        self.q_b_mvar = q_b_mvar  
                        self.p_c_mw = p_c_mw 
                        self.q_c_mvar = q_c_mvar                                     
                       
                class AsymmetricStaticGeneratorsOut(object):
                    def __init__(self, asymmetricstaticgenerators: List[AsymmetricStaticGeneratorOut]):
                        self.asymmetricstaticgenerators = asymmetricstaticgenerators             
                asymmetricstaticgeneratorsList = list()  
                
                
                class TransformerOut(object):
                    def __init__(self, name: str, id: str, p_hv_mw: float, q_hv_mvar: float, p_lv_mw: float, q_lv_mvar: float, pl_mw: float, ql_mvar: float, i_hv_ka: float, i_lv_ka: float, vm_hv_pu: float, vm_lv_pu: float, va_hv_degree: float, va_lv_degree: float, loading_percent: float):          
                        self.name = name
                        self.id = id
                        self.p_hv_mw = p_hv_mw 
                        self.q_hv_mvar = q_hv_mvar
                        self.p_lv_mw = p_lv_mw                            
                        self.q_lv_mvar = q_lv_mvar 
                        self.pl_mw = pl_mw  
                        self.ql_mvar = ql_mvar                         
                        self.i_hv_ka = i_hv_ka 
                        self.i_lv_ka = i_lv_ka
                        self.vm_hv_pu = vm_hv_pu
                        self.vm_lv_pu = vm_lv_pu
                        self.va_hv_degree = va_hv_degree
                        self.va_lv_degree = va_lv_degree
                        self.loading_percent = loading_percent
                                                             
                       
                class TransformersOut(object):
                    def __init__(self, transformers: List[TransformerOut]):
                        self.transformers = transformers             
                transformersList = list() 
                
                
                class Transformer3WOut(object):
                    def __init__(self, name: str, id: str, p_hv_mw: float, q_hv_mvar: float, p_mv_mw: float, q_mv_mvar: float, 
                                 p_lv_mw: float, q_lv_mvar: float, pl_mw: float, ql_mvar: float, i_hv_ka: float, 
                                 i_mv_ka: float, i_lv_ka: float, vm_hv_pu: float, vm_mv_pu: float, 
                                 vm_lv_pu: float, va_hv_degree: float, va_mv_degree: float, va_lv_degree: float, loading_percent: float):          
                        self.name = name
                        self.id = id
                        self.p_hv_mw = p_hv_mw 
                        self.q_hv_mvar = q_hv_mvar
                        self.p_mv_mw = p_mv_mw                            
                        self.q_mv_mvar = q_mv_mvar 
                        self.p_lv_mw = p_lv_mw  
                        self.q_lv_mvar = q_lv_mvar                         
                        self.pl_mw = pl_mw 
                        self.ql_mvar = ql_mvar
                        self.i_hv_ka = i_hv_ka
                        self.i_mv_ka = i_mv_ka
                        self.i_lv_ka = i_lv_ka
                        self.vm_hv_pu = vm_hv_pu
                        self.vm_mv_pu = vm_mv_pu
                        self.vm_lv_pu = vm_lv_pu
                        self.va_hv_degree = va_hv_degree
                        self.va_mv_degree = va_mv_degree
                        self.va_lv_degree = va_lv_degree
                        self.loading_percent = loading_percent                                                             
                       
                class Transformers3WOut(object):
                    def __init__(self, transformers3W: List[Transformer3WOut]):
                        self.transformers3W = transformers3W             
                transformers3WList = list()                 
                
                
                class ShuntOut(object):
                    def __init__(self, name: str, id: str, p_mw: float, q_mvar: float, vm_pu: float,
                                step=None, max_step=None):
                        self.name = name
                        self.id = id
                        self.p_mw = p_mw
                        self.q_mvar = q_mvar
                        self.vm_pu = vm_pu
                        # Discrete tap / CharacteristicControl / fixed: final step after PF (for UI Result Box)
                        self.step = step
                        self.max_step = max_step
                       
                class ShuntsOut(object):
                    def __init__(self, shunts: List[ShuntOut]):
                        self.shunts = shunts              
                shuntsList = list() 
                
                
                class CapacitorOut(object):
                    def __init__(self, name: str, id:str, p_mw: float, q_mvar: float, vm_pu: float):          
                        self.name = name
                        self.id = id
                        self.p_mw = p_mw 
                        self.q_mvar = q_mvar  
                        self.vm_pu = vm_pu                          
                       
                class CapacitorsOut(object):
                    def __init__(self, capacitors: List[CapacitorOut]):
                        self.capacitors = capacitors              
                capacitorsList = list()                 
                
                
                class LoadOut(object):
                    def __init__(self, name: str, id:str, p_mw: float, q_mvar: float):          
                        self.name = name
                        self.id = id
                        self.p_mw = p_mw 
                        self.q_mvar = q_mvar 
                                                                    
                class LoadsOut(object):
                    def __init__(self, loads: List[LoadOut]):
                        self.loads = loads              
                loadsList = list() 
                
                
                class AsymmetricLoadOut(object):
                    def __init__(self, name: str, id:str, p_a_mw: float, q_a_mvar: float, p_b_mw: float, q_b_mvar: float, p_c_mw: float, q_c_mvar: float):          
                        self.name = name
                        self.id = id
                        self.p_a_mw = p_a_mw 
                        self.q_a_mvar = q_a_mvar 
                        self.p_b_mw = p_b_mw 
                        self.q_b_mvar = q_b_mvar
                        self.p_c_mw = p_c_mw 
                        self.q_c_mvar = q_c_mvar
                                                                    
                class AsymmetricLoadsOut(object):
                    def __init__(self, asymmetricloads: List[AsymmetricLoadOut]):
                        self.asymmetricloads = asymmetricloads              
                asymmetricloadsList = list() 
                
                
                class ImpedanceOut(object):
                    def __init__(self, name: str, id:str, p_from_mw: float, q_from_mvar: float, p_to_mw: float, q_to_mvar: float, pl_mw: float, ql_mvar: float, i_from_ka: float, i_to_ka: float ):          
                        self.name = name
                        self.id = id
                        self.p_from_mw = p_from_mw 
                        self.q_from_mvar = q_from_mvar 
                        self.p_to_mw = p_to_mw 
                        self.q_to_mvar = q_to_mvar
                        self.pl_mw = pl_mw 
                        self.ql_mvar = ql_mvar
                        self.i_from_ka = i_from_ka 
                        self.i_to_ka = i_to_ka                        
                                                                    
                class ImpedancesOut(object):
                    def __init__(self, impedances: List[ImpedanceOut]):
                        self.impedances = impedances              
                impedancesList = list() 
                
                
                class WardOut(object):
                    def __init__(self, name: str, id:str, p_mw: float, q_mvar: float, vm_pu: float):          
                        self.name = name
                        self.id = id
                        self.p_mw = p_mw 
                        self.q_mvar = q_mvar 
                        self.vm_pu = vm_pu 
                       
                class WardsOut(object):
                    def __init__(self, wards: List[WardOut]):
                        self.wards = wards              
                wardsList = list() 
                
                
                class ExtendedWardOut(object):
                    def __init__(self, name: str, id:str, p_mw: float, q_mvar: float, vm_pu: float):          
                        self.name = name                        
                        self.id = id
                        self.p_mw = p_mw 
                        self.q_mvar = q_mvar 
                        self.vm_pu = vm_pu 
                       
                class ExtendedWardsOut(object):
                    def __init__(self, extendedwards: List[ExtendedWardOut]):
                        self.extendedwards = extendedwards              
                extendedwardsList = list() 
                
                
                class MotorOut(object):
                    def __init__(self, name: str, id:str, p_mw: float, q_mvar: float):          
                        self.name = name
                        self.id = id
                        self.p_mw = p_mw 
                        self.q_mvar = q_mvar 
                       
                       
                class MotorsOut(object):
                    def __init__(self, motors: List[MotorOut]):
                        self.motors = motors              
                motorsList = list()

                class SVCOut(object):
                    def __init__(self, name: str, id:str, thyristor_firing_angle_degree: float, x_ohm: float, q_mvar: float, vm_pu: float, va_degree: float):          
                        self.name = name
                        self.id = id
                        self.thyristor_firing_angle_degree = thyristor_firing_angle_degree 
                        self.x_ohm = x_ohm   
                        self.q_mvar = q_mvar
                        self.vm_pu = vm_pu
                        self.va_degree = va_degree      
                       
                class SVCsOut(object):
                    def __init__(self, svcs: List[SVCOut]):
                        self.svcs = svcs              
                SVCsList = list()

                class TCSCOut(object):
                    def __init__(self, name: str, id:str, thyristor_firing_angle_degree: float, x_ohm: float, p_from_mw: float, q_from_mvar: float, p_to_mw: float, q_to_mvar: float, p_l_mw: float, q_l_mvar: float, vm_from_pu: float, va_from_degree: float, vm_to_pu: float, va_to_degree: float ):          
                        self.name = name
                        self.id = id
                        self.thyristor_firing_angle_degree = thyristor_firing_angle_degree 
                        self.x_ohm = x_ohm
                        self.p_from_mw = p_from_mw 
                        self.q_from_mvar = q_from_mvar
                        self.p_to_mw = p_to_mw 
                        self.q_to_mvar = q_to_mvar
                        self.p_l_mw = p_l_mw 
                        self.q_l_mvar = q_l_mvar
                        self.vm_from_pu = vm_from_pu 
                        self.va_from_degree = va_from_degree
                        self.vm_to_pu = vm_to_pu 
                        self.va_to_degree = va_to_degree                        
                       
                class TCSCsOut(object):
                    def __init__(self, tcscs: List[TCSCOut]):
                        self.tcscs = tcscs              
                TCSCsList = list()

                
                class SSCOut(object):
                    def __init__(self, name: str, id:str, q_mvar: float, vm_internal_pu: float, va_internal_degree: float, vm_pu: float, va_degree: float):          
                        self.name = name
                        self.id = id
                        self.q_mvar = q_mvar 
                        self.vm_internal_pu = vm_internal_pu
                        self.va_internal_degree = va_internal_degree
                        self.vm_pu = vm_pu
                        self.va_degree = va_degree                       
                       
                class SSCsOut(object):
                    def __init__(self, sscs: List[SSCOut]):
                        self.sscs = sscs              
                sscsList = list()
                
                
                
                class StorageOut(object):
                    def __init__(self, name: str, id:str, p_mw: float, q_mvar: float):          
                        self.name = name
                        self.id = id
                        self.p_mw = p_mw 
                        self.q_mvar = q_mvar 
                       
                       
                class StoragesOut(object):
                    def __init__(self, storages: List[StorageOut]):
                        self.storages = storages              
                storagesList = list() 
                
                
                class DClineOut(object):
                    def __init__(self, name: str, id:str, p_from_mw: float, q_from_mvar: float, p_to_mw: float, q_to_mvar: float, pl_mw: float, vm_from_pu: float, va_from_degree: float, vm_to_pu: float, va_to_degree: float):          
                        self.name = name
                        self.id = id
                        self.p_from_mw = p_from_mw 
                        self.q_from_mvar = q_from_mvar 
                        self.p_to_mw = p_to_mw 
                        self.q_to_mvar = q_to_mvar
                        self.pl_mw = pl_mw                       
                        self.vm_from_pu = vm_from_pu 
                        self.va_from_degree = va_from_degree 
                        self.vm_to_pu = vm_to_pu 
                        self.va_to_degree = va_to_degree                           
                                                                    
                class DClinesOut(object):
                    def __init__(self, dclines: List[DClineOut]):
                        self.dclines = dclines              
                dclinesList = list()
                
                class DcBusOut(object):
                    def __init__(self, name: str, id: str, vm_pu: float, p_mw: float):          
                        self.name = name
                        self.id = id
                        self.vm_pu = vm_pu
                        self.p_mw = p_mw
                       
                class DcBusesOut(object):
                    def __init__(self, dcbuses: List[DcBusOut]):
                        self.dcbuses = dcbuses              
                dcbusesList = list()
                
                class LoadDcOut(object):
                    def __init__(self, name: str, id: str, p_mw: float):          
                        self.name = name
                        self.id = id
                        self.p_mw = p_mw
                       
                class LoadsDcOut(object):
                    def __init__(self, loadsdc: List[LoadDcOut]):
                        self.loadsdc = loadsdc              
                loadsdcList = list()
                
                class SourceDcOut(object):
                    def __init__(self, name: str, id: str, vm_pu: float, p_mw: float):          
                        self.name = name
                        self.id = id
                        self.vm_pu = vm_pu
                        self.p_mw = p_mw
                       
                class SourcesDcOut(object):
                    def __init__(self, sourcesdc: List[SourceDcOut]):
                        self.sourcesdc = sourcesdc              
                sourcesdcList = list()
                
                class SwitchOut(object):
                    def __init__(self, name: str, id: str, closed: bool, i_ka: float,
                                 p_from_mw: float = 0.0, q_from_mvar: float = 0.0,
                                 p_to_mw: float = 0.0, q_to_mvar: float = 0.0,
                                 loading_percent: float = 0.0):
                        self.name = name
                        self.id = id
                        self.closed = closed
                        self.i_ka = i_ka
                        self.p_from_mw = p_from_mw
                        self.q_from_mvar = q_from_mvar
                        self.p_to_mw = p_to_mw
                        self.q_to_mvar = q_to_mvar
                        self.loading_percent = loading_percent
                       
                class SwitchesOut(object):
                    def __init__(self, switches: List[SwitchOut]):
                        self.switches = switches              
                switchesList = list()
                
                class VSCOut(object):
                    def __init__(self, name: str, id: str, p_mw: float, vm_pu: float, q_mvar: float = None,
                                 p_dc_mw: float = None, vm_dc_pu: float = None):
                        self.name = name
                        self.id = id
                        self.p_mw = p_mw
                        self.vm_pu = vm_pu
                        self.q_mvar = q_mvar
                        self.p_dc_mw = p_dc_mw
                        self.vm_dc_pu = vm_dc_pu
                       
                class VSCsOut(object):
                    def __init__(self, vscs: List[VSCOut]):
                        self.vscs = vscs              
                vscsList = list()
                
                class B2bVSCOut(object):
                    def __init__(self, name: str, id: str, p_mw: float, q_mvar: float, vm_pu: float,
                                 p_dc_mw_p: float, p_dc_mw_m: float, vm_dc_pu_p: float, vm_dc_pu_m: float):
                        self.name = name
                        self.id = id
                        self.p_mw = p_mw
                        self.q_mvar = q_mvar
                        self.vm_pu = vm_pu
                        self.p_dc_mw_p = p_dc_mw_p
                        self.p_dc_mw_m = p_dc_mw_m
                        self.vm_dc_pu_p = vm_dc_pu_p
                        self.vm_dc_pu_m = vm_dc_pu_m
                       
                class B2bVSCsOut(object):
                    def __init__(self, b2bvscs: List[B2bVSCOut]):
                        self.b2bvscs = b2bvscs              
                b2bvscsList = list() 
                
                
                #Bus
                for index, row in net.res_bus.iterrows():
                    if _electrisim_is_aux(net.bus, index):
                        continue   # a DC/DC converter's auxiliary AC bus
                    p_mw = row['p_mw']
                    q_mvar = row['q_mvar']
                    denom_pf = math.sqrt(math.pow(p_mw, 2) + math.pow(q_mvar, 2))
                    if denom_pf != 0 and not math.isnan(denom_pf):
                        pf = p_mw / denom_pf
                    else:
                        pf = 0.0
                    if math.isnan(pf):
                        pf = 0.0
                    if p_mw != 0 and not math.isnan(p_mw):
                        q_p = q_mvar / p_mw
                    else:
                        q_p = 0.0
                    if math.isnan(q_p) or math.isinf(q_p):
                        q_p = 0.0
                    p_br, q_br = _electrisim_bus_branch_p_q_sum(net, index)
                    p_nodal, q_nodal = _electrisim_bus_nodal_p_q_sum(net, index)
                    denom_pf_nodal = math.sqrt(math.pow(p_nodal, 2) + math.pow(q_nodal, 2))
                    if denom_pf_nodal != 0 and not math.isnan(denom_pf_nodal):
                        pf_nodal = p_nodal / denom_pf_nodal
                    else:
                        pf_nodal = 0.0
                    if math.isnan(pf_nodal):
                        pf_nodal = 0.0
                    if p_nodal != 0 and not math.isnan(p_nodal):
                        q_p_nodal = q_nodal / p_nodal
                    else:
                        q_p_nodal = 0.0
                    if math.isnan(q_p_nodal) or math.isinf(q_p_nodal):
                        q_p_nodal = 0.0
                    _vm_pu = row['vm_pu']
                    _vn_kv = float(net.bus.at[index, 'vn_kv'])
                    _vm_kv = None
                    if _vn_kv > 0 and _vm_pu == _vm_pu and not math.isnan(_vm_pu):
                        _vm_kv = float(_vm_pu) * _vn_kv
                    busbar = BusbarOut(
                        name=net.bus._get_value(index, 'name'), id=net.bus._get_value(index, 'id'),
                        vm_pu=_vm_pu, va_degree=row['va_degree'],
                        p_mw=p_mw, q_mvar=q_mvar, pf=pf, q_p=q_p,
                        p_branch_mw=p_br, q_branch_mvar=q_br,
                        p_nodal_mw=p_nodal, q_nodal_mvar=q_nodal,
                        pf_nodal=pf_nodal, q_p_nodal=q_p_nodal,
                        vm_kv=_vm_kv,
                    )
                    busbarList.append(busbar) 
                    busbars = BusbarsOut(busbars = busbarList)
                
                #Line
                if(net.res_line.empty):
                        result = {**busbars.__dict__}                  
                else:                    
                        for index, row in net.res_line.iterrows():    
                            line = LineOut(name=net.line._get_value(index, 'name'), id = net.line._get_value(index, 'id'), p_from_mw=row['p_from_mw'], q_from_mvar=row['q_from_mvar'], p_to_mw=row['p_to_mw'], q_to_mvar=row['q_to_mvar'], i_from_ka=row['i_from_ka'], i_to_ka=row['i_to_ka'], loading_percent=row['loading_percent'])        
                            linesList.append(line) 
                            lines = LinesOut(lines = linesList)
                            
                            result = {**busbars.__dict__, **lines.__dict__} #łączenie dwóch dictionaries                        
                     
                
                #External Grid
                if(net.res_ext_grid.empty):
                    pass
                else:                    
                        for index, row in net.res_ext_grid.iterrows():    
                            if _electrisim_is_aux(net.ext_grid, index):
                                continue   # a DC/DC converter's auxiliary grid
                            externalgrid = ExternalGridOut(name=net.ext_grid._get_value(index, 'name'), id = net.ext_grid._get_value(index, 'id'), p_mw=row['p_mw'], q_mvar=row['q_mvar'], pf = row['p_mw']/math.sqrt(math.pow(row['p_mw'],2)+math.pow(row['q_mvar'],2)), q_p=row['q_mvar']/row['p_mw'])        
                            externalgridsList.append(externalgrid) 
                            externalgrids = ExternalGridsOut(externalgrids = externalgridsList) 
                        result = {**result, **ExternalGridsOut(externalgrids = externalgridsList).__dict__}          
                             
                #Generator         
                if(net.res_gen.empty):
                    pass
                else:                    
                        for index, row in net.res_gen.iterrows():    
                            if _electrisim_is_pcs(net.gen, index):
                                continue   # a PCS: reported with the PCS
                            generator = GeneratorOut(name=net.gen._get_value(index, 'name'), id = net.gen._get_value(index, 'id'), p_mw=row['p_mw'], q_mvar=row['q_mvar'], va_degree=row['va_degree'], vm_pu=row['vm_pu'])        
                            generatorsList.append(generator) 
                            generators = GeneratorsOut(generators = generatorsList)
                        
                        result = {**result, **GeneratorsOut(generators = generatorsList).__dict__}
                        
                #Static Generator                     
                if(net.res_sgen.empty):
                    pass
                else:                    
                        for index, row in net.res_sgen.iterrows():    
                            if _electrisim_is_aux(net.sgen, index) or _electrisim_is_pcs(net.sgen, index):
                                continue   # a converter stage's own element, or a PCS
                            staticgenerator = StaticGeneratorOut(name=net.sgen._get_value(index, 'name'), id = net.sgen._get_value(index, 'id'), p_mw=row['p_mw'], q_mvar=row['q_mvar'])        
                            staticgeneratorsList.append(staticgenerator) 
                            staticgenerators = StaticGeneratorsOut(staticgenerators = staticgeneratorsList)
                        
                        result = {**result, **StaticGeneratorsOut(staticgenerators = staticgeneratorsList).__dict__}
                        
                        
                
                #Asymmetric Static Generator                     
                if(net.res_asymmetric_sgen.empty):
                    pass
                else:
                        available_columns = net.res_asymmetric_sgen.columns.tolist()
                        
                        # Check if phase-specific columns exist
                        has_phase_specific = all(col in available_columns for col in ['p_a_mw', 'q_a_mvar', 'p_b_mw', 'q_b_mvar', 'p_c_mw', 'q_c_mvar'])
                        has_aggregate = all(col in available_columns for col in ['p_mw', 'q_mvar'])
                        
                        if has_phase_specific:
                            # Use phase-specific results if available
                            for index, row in net.res_asymmetric_sgen.iterrows():    
                                asymmetricstaticgenerator = AsymmetricStaticGeneratorOut(
                                    name=net.asymmetric_sgen._get_value(index, 'name'), 
                                    id=net.asymmetric_sgen._get_value(index, 'id'), 
                                    p_a_mw=row['p_a_mw'], 
                                    q_a_mvar=row['q_a_mvar'], 
                                    p_b_mw=row['p_b_mw'], 
                                    q_b_mvar=row['q_b_mvar'], 
                                    p_c_mw=row['p_c_mw'], 
                                    q_c_mvar=row['q_c_mvar']
                                )        
                                asymmetricstaticgeneratorsList.append(asymmetricstaticgenerator)
                        elif has_aggregate:
                            # If only aggregate results available, distribute based on input phase distribution
                            for index, row in net.res_asymmetric_sgen.iterrows():
                                # Get input phase values to determine distribution
                                p_a_input = float(net.asymmetric_sgen._get_value(index, 'p_a_mw'))
                                p_b_input = float(net.asymmetric_sgen._get_value(index, 'p_b_mw'))
                                p_c_input = float(net.asymmetric_sgen._get_value(index, 'p_c_mw'))
                                q_a_input = float(net.asymmetric_sgen._get_value(index, 'q_a_mvar'))
                                q_b_input = float(net.asymmetric_sgen._get_value(index, 'q_b_mvar'))
                                q_c_input = float(net.asymmetric_sgen._get_value(index, 'q_c_mvar'))
                                
                                # Calculate total input power for distribution
                                p_total_input = abs(p_a_input) + abs(p_b_input) + abs(p_c_input)
                                q_total_input = abs(q_a_input) + abs(q_b_input) + abs(q_c_input)
                                
                                # Get aggregate results
                                p_total = float(row['p_mw'])
                                q_total = float(row['q_mvar'])
                                
                                # Distribute results proportionally based on input phase distribution
                                if p_total_input > 0:
                                    p_a_mw = p_total * (abs(p_a_input) / p_total_input)
                                    p_b_mw = p_total * (abs(p_b_input) / p_total_input)
                                    p_c_mw = p_total * (abs(p_c_input) / p_total_input)
                                else:
                                    # If no input power, distribute equally
                                    p_a_mw = p_total / 3.0
                                    p_b_mw = p_total / 3.0
                                    p_c_mw = p_total / 3.0
                                
                                if q_total_input > 0:
                                    q_a_mvar = q_total * (abs(q_a_input) / q_total_input)
                                    q_b_mvar = q_total * (abs(q_b_input) / q_total_input)
                                    q_c_mvar = q_total * (abs(q_c_input) / q_total_input)
                                else:
                                    # If no input reactive power, distribute equally
                                    q_a_mvar = q_total / 3.0
                                    q_b_mvar = q_total / 3.0
                                    q_c_mvar = q_total / 3.0
                                
                                asymmetricstaticgenerator = AsymmetricStaticGeneratorOut(
                                    name=net.asymmetric_sgen._get_value(index, 'name'), 
                                    id=net.asymmetric_sgen._get_value(index, 'id'), 
                                    p_a_mw=p_a_mw, 
                                    q_a_mvar=q_a_mvar, 
                                    p_b_mw=p_b_mw, 
                                    q_b_mvar=q_b_mvar, 
                                    p_c_mw=p_c_mw, 
                                    q_c_mvar=q_c_mvar
                                )        
                                asymmetricstaticgeneratorsList.append(asymmetricstaticgenerator)
                        else:
                            print(f"Warning: res_asymmetric_sgen has unexpected column structure. Available columns: {available_columns}")
                            pass
                        
                        if asymmetricstaticgeneratorsList:
                            asymmetricstaticgenerators = AsymmetricStaticGeneratorsOut(asymmetricstaticgenerators = asymmetricstaticgeneratorsList)
                            result = {**result, **asymmetricstaticgenerators.__dict__}
                        
               
                # Transformer — one JSON row per net.trafo (frontend updates every 2W trafo on the diagram).
                # net.res_trafo can omit rows (out-of-service, unsupplied, etc.); iterating only res_trafo left those shapes without results.
                if not net.trafo.empty:
                    res_tf = getattr(net, 'res_trafo', None)
                    for trafo_index in net.trafo.index:
                        t_name = net.trafo._get_value(trafo_index, 'name')
                        t_raw_id = net.trafo._get_value(trafo_index, 'id')
                        t_name_s = str(t_name) if t_name is not None and not pd.isna(t_name) else str(trafo_index)
                        t_id = _trafo_out_id(t_raw_id, t_name, trafo_index)
                        row = _pf_res_row_for_element(net.trafo, res_tf, trafo_index)
                        if row is not None:
                            transformer = TransformerOut(
                                name=t_name_s,
                                id=t_id,
                                p_hv_mw=row['p_hv_mw'],
                                q_hv_mvar=row['q_hv_mvar'],
                                p_lv_mw=row['p_lv_mw'],
                                q_lv_mvar=row['q_lv_mvar'],
                                pl_mw=row['pl_mw'],
                                ql_mvar=row['ql_mvar'],
                                i_hv_ka=row['i_hv_ka'],
                                i_lv_ka=row['i_lv_ka'],
                                vm_hv_pu=row['vm_hv_pu'],
                                vm_lv_pu=row['vm_lv_pu'],
                                va_hv_degree=row['va_hv_degree'],
                                va_lv_degree=row['va_lv_degree'],
                                loading_percent=row['loading_percent'],
                            )
                        else:
                            transformer = TransformerOut(
                                name=t_name_s,
                                id=t_id,
                                p_hv_mw=0.0,
                                q_hv_mvar=0.0,
                                p_lv_mw=0.0,
                                q_lv_mvar=0.0,
                                pl_mw=0.0,
                                ql_mvar=0.0,
                                i_hv_ka=0.0,
                                i_lv_ka=0.0,
                                vm_hv_pu=1.0,
                                vm_lv_pu=1.0,
                                va_hv_degree=0.0,
                                va_lv_degree=0.0,
                                loading_percent=0.0,
                            )
                        transformersList.append(transformer)
                    if transformersList:
                        transformers = TransformersOut(transformers=transformersList)
                        result = {**result, **transformers.__dict__}

                # Three-winding transformer — same pattern as 2W
                if hasattr(net, 'trafo3w') and not net.trafo3w.empty:
                    res_t3 = getattr(net, 'res_trafo3w', None)
                    for t3_index in net.trafo3w.index:
                        t_name = net.trafo3w._get_value(t3_index, 'name')
                        t_raw_id = net.trafo3w._get_value(t3_index, 'id')
                        t_name_s = str(t_name) if t_name is not None and not pd.isna(t_name) else str(t3_index)
                        t_id = _trafo_out_id(t_raw_id, t_name, t3_index)
                        row = _pf_res_row_for_element(net.trafo3w, res_t3, t3_index)
                        if row is not None:
                            transformer3W = Transformer3WOut(
                                name=t_name_s,
                                id=t_id,
                                p_hv_mw=row['p_hv_mw'],
                                q_hv_mvar=row['q_hv_mvar'],
                                p_mv_mw=row['p_mv_mw'],
                                q_mv_mvar=row['q_mv_mvar'],
                                p_lv_mw=row['p_lv_mw'],
                                q_lv_mvar=row['q_lv_mvar'],
                                pl_mw=row['pl_mw'],
                                ql_mvar=row['ql_mvar'],
                                i_hv_ka=row['i_hv_ka'],
                                i_mv_ka=row['i_mv_ka'],
                                i_lv_ka=row['i_lv_ka'],
                                vm_hv_pu=row['vm_hv_pu'],
                                vm_mv_pu=row['vm_mv_pu'],
                                vm_lv_pu=row['vm_lv_pu'],
                                va_hv_degree=row['va_hv_degree'],
                                va_mv_degree=row['va_mv_degree'],
                                va_lv_degree=row['va_lv_degree'],
                                loading_percent=row['loading_percent'],
                            )
                        else:
                            transformer3W = Transformer3WOut(
                                name=t_name_s,
                                id=t_id,
                                p_hv_mw=0.0,
                                q_hv_mvar=0.0,
                                p_mv_mw=0.0,
                                q_mv_mvar=0.0,
                                p_lv_mw=0.0,
                                q_lv_mvar=0.0,
                                pl_mw=0.0,
                                ql_mvar=0.0,
                                i_hv_ka=0.0,
                                i_mv_ka=0.0,
                                i_lv_ka=0.0,
                                vm_hv_pu=1.0,
                                vm_mv_pu=1.0,
                                vm_lv_pu=1.0,
                                va_hv_degree=0.0,
                                va_mv_degree=0.0,
                                va_lv_degree=0.0,
                                loading_percent=0.0,
                            )
                        transformers3WList.append(transformer3W)
                    if transformers3WList:
                        transformers3W = Transformers3WOut(transformers3W=transformers3WList)
                        result = {**result, **transformers3W.__dict__}
               
               

                
                #Shunt reactor
                if(net.res_shunt.empty):
                    pass
                else:                    
                        for index, row in net.res_shunt.iterrows():
                            #if (row['q_mvar'] >= 0):
                            if (net.shunt._get_value(index, 'typ') == 'shuntreactor'):
                                try:
                                    sw = net.shunt._get_value(index, 'step')
                                    smx = net.shunt._get_value(index, 'max_step')
                                    zb = _electrisim_shunt_is_zero_based(net, index)
                                    step_pf = _electrisim_shunt_step_from_pp(sw, zb) if sw is not None and not pd.isna(sw) else None
                                    max_pf = _electrisim_shunt_step_from_pp(smx, zb) if zb and smx is not None and not pd.isna(smx) else (float(smx) if smx is not None and not pd.isna(smx) else None)
                                except Exception:
                                    step_pf = max_pf = None
                                p_out, q_out, vm_out = _electrisim_shunt_res_for_output(net, index, row)
                                shunt = ShuntOut(
                                    name=net.shunt._get_value(index, 'name'), id = net.shunt._get_value(index, 'id'),
                                    p_mw=p_out, q_mvar=q_out, vm_pu = vm_out,
                                    step=step_pf, max_step=max_pf,
                                )
                                shuntsList.append(shunt)
                                shunts = ShuntsOut(shunts = shuntsList) 
                                result = {**result, **shunts.__dict__}  
                        
                #Capacitor
                if(net.res_shunt.empty):
                    pass
                else:                    
                        for index, row in net.res_shunt.iterrows(): 
                            if (net.shunt._get_value(index, 'typ') == 'capacitor'):  # q is always negative for capacitor
                                capacitor = CapacitorOut(name=net.shunt._get_value(index, 'name'), id = net.shunt._get_value(index, 'id'), p_mw=row['p_mw'], q_mvar=row['q_mvar'], vm_pu = row['vm_pu'])        
                                capacitorsList.append(capacitor) 
                                capacitors = CapacitorsOut(capacitors = capacitorsList) 
                                result = {**result, **capacitors.__dict__}  
                
               
                #Load
                if(net.res_load.empty):
                    pass
                else:                    
                        for index, row in net.res_load.iterrows():    
                            if _electrisim_is_aux(net.load, index):
                                continue   # a converter stage's own element
                            load = LoadOut(name=net.load._get_value(index, 'name'), id = net.load._get_value(index, 'id'), p_mw=row['p_mw'], q_mvar=row['q_mvar'])        
                            # print(load)  # Comment out this debug line
                            loadsList.append(load) 
                            # print(loadsList)  # Comment out this debug line
                            loads = LoadsOut(loads = loadsList) 
                        result = {**result, **LoadsOut(loads = loadsList).__dict__}        
                        
                #Asymmetric Load
                if(net.res_asymmetric_load.empty):
                    pass
                else:
                        available_columns = net.res_asymmetric_load.columns.tolist()
                        
                        # Check if phase-specific columns exist
                        has_phase_specific = all(col in available_columns for col in ['p_a_mw', 'q_a_mvar', 'p_b_mw', 'q_b_mvar', 'p_c_mw', 'q_c_mvar'])
                        has_aggregate = all(col in available_columns for col in ['p_mw', 'q_mvar'])
                        
                        if has_phase_specific:
                            # Use phase-specific results if available
                            for index, row in net.res_asymmetric_load.iterrows():    
                                asymmetricload = AsymmetricLoadOut(
                                    name=net.asymmetric_load._get_value(index, 'name'), 
                                    id=net.asymmetric_load._get_value(index, 'id'), 
                                    p_a_mw=row['p_a_mw'], 
                                    q_a_mvar=row['q_a_mvar'], 
                                    p_b_mw=row['p_b_mw'], 
                                    q_b_mvar=row['q_b_mvar'], 
                                    p_c_mw=row['p_c_mw'], 
                                    q_c_mvar=row['q_c_mvar']
                                )        
                                asymmetricloadsList.append(asymmetricload)
                        elif has_aggregate:
                            # If only aggregate results available, distribute based on input phase distribution
                            for index, row in net.res_asymmetric_load.iterrows():
                                # Get input phase values to determine distribution
                                p_a_input = float(net.asymmetric_load._get_value(index, 'p_a_mw'))
                                p_b_input = float(net.asymmetric_load._get_value(index, 'p_b_mw'))
                                p_c_input = float(net.asymmetric_load._get_value(index, 'p_c_mw'))
                                q_a_input = float(net.asymmetric_load._get_value(index, 'q_a_mvar'))
                                q_b_input = float(net.asymmetric_load._get_value(index, 'q_b_mvar'))
                                q_c_input = float(net.asymmetric_load._get_value(index, 'q_c_mvar'))
                                
                                # Calculate total input power for distribution
                                p_total_input = abs(p_a_input) + abs(p_b_input) + abs(p_c_input)
                                q_total_input = abs(q_a_input) + abs(q_b_input) + abs(q_c_input)
                                
                                # Get aggregate results
                                p_total = float(row['p_mw'])
                                q_total = float(row['q_mvar'])
                                
                                # Distribute results proportionally based on input phase distribution
                                if p_total_input > 0:
                                    p_a_mw = p_total * (abs(p_a_input) / p_total_input)
                                    p_b_mw = p_total * (abs(p_b_input) / p_total_input)
                                    p_c_mw = p_total * (abs(p_c_input) / p_total_input)
                                else:
                                    # If no input power, distribute equally
                                    p_a_mw = p_total / 3.0
                                    p_b_mw = p_total / 3.0
                                    p_c_mw = p_total / 3.0
                                
                                if q_total_input > 0:
                                    q_a_mvar = q_total * (abs(q_a_input) / q_total_input)
                                    q_b_mvar = q_total * (abs(q_b_input) / q_total_input)
                                    q_c_mvar = q_total * (abs(q_c_input) / q_total_input)
                                else:
                                    # If no input reactive power, distribute equally
                                    q_a_mvar = q_total / 3.0
                                    q_b_mvar = q_total / 3.0
                                    q_c_mvar = q_total / 3.0
                                
                                asymmetricload = AsymmetricLoadOut(
                                    name=net.asymmetric_load._get_value(index, 'name'), 
                                    id=net.asymmetric_load._get_value(index, 'id'), 
                                    p_a_mw=p_a_mw, 
                                    q_a_mvar=q_a_mvar, 
                                    p_b_mw=p_b_mw, 
                                    q_b_mvar=q_b_mvar, 
                                    p_c_mw=p_c_mw, 
                                    q_c_mvar=q_c_mvar
                                )        
                                asymmetricloadsList.append(asymmetricload)
                        else:
                            print(f"Warning: res_asymmetric_load has unexpected column structure. Available columns: {available_columns}")
                            pass
                        
                        if asymmetricloadsList:
                            asymmetricloads = AsymmetricLoadsOut(asymmetricloads = asymmetricloadsList) 
                            result = {**result, **asymmetricloads.__dict__}    
                        
                        
                #Impedance
                if(net.res_impedance.empty):
                    pass
                else:                    
                        for index, row in net.res_impedance.iterrows():    
                            impedance = ImpedanceOut(name=net.impedance._get_value(index, 'name'), id = net.impedance._get_value(index, 'id'), p_from_mw=row['p_from_mw'], q_from_mvar=row['q_from_mvar'], p_to_mw=row['p_to_mw'], q_to_mvar=row['q_to_mvar'], pl_mw=row['pl_mw'], ql_mvar=row['ql_mvar'], i_from_ka=row['i_from_ka'], i_to_ka=row['i_to_ka'])        
                            impedancesList.append(impedance) 
                            impedances = ImpedancesOut(impedances = impedancesList) 
                        result = {**result, **impedances.__dict__} 
                        
                
                #Ward
                if(net.res_ward.empty):
                    pass
                else:                    
                        for index, row in net.res_ward.iterrows():    
                            ward = WardOut(name=net.ward._get_value(index, 'name'), id = net.ward._get_value(index, 'id'), p_mw=row['p_mw'], q_mvar=row['q_mvar'], vm_pu=row['vm_pu'])        
                            wardsList.append(ward) 
                            wards = WardsOut(wards = wardsList) 
                        result = {**result, **wards.__dict__} 
                        
                        
                #Extended Ward
                if(net.res_xward.empty):
                    pass
                else:                    
                        for index, row in net.res_xward.iterrows():    
                            extendedward = ExtendedWardOut(name=net.xward._get_value(index, 'name'), id = net.xward._get_value(index, 'id'), p_mw=row['p_mw'], q_mvar=row['q_mvar'], vm_pu=row['vm_pu'])        
                            extendedwardsList.append(extendedward) 
                            extendedwards = ExtendedWardsOut(extendedwards = extendedwardsList) 
                        result = {**result, **extendedwards.__dict__} 
                        
                        
                #Motor
                if(net.res_motor.empty):
                    pass
                else:                    
                        for index, row in net.res_motor.iterrows():    
                            motor = MotorOut(name=net.motor._get_value(index, 'name'), id = net.motor._get_value(index, 'id'), p_mw=row['p_mw'], q_mvar=row['q_mvar'])        
                            motorsList.append(motor) 
                            motors = MotorsOut(motors = motorsList) 
                        result = {**result, **motors.__dict__} 
                        
                #Storage
                if(net.res_storage.empty):
                    pass
                else:                    
                        for index, row in net.res_storage.iterrows():    
                            storage = StorageOut(name=net.storage._get_value(index, 'name'), id = net.storage._get_value(index, 'id'), p_mw=row['p_mw'], q_mvar=row['q_mvar'])        
                            storagesList.append(storage) 
                            storages = StoragesOut(storages = storagesList) 
                        result = {**result, **storages.__dict__}


                #SVC
                try:                    
                    for index, row in net.res_svc.iterrows():    
                        svc = SVCOut(name=net.svc._get_value(index, 'name'), id = net.svc._get_value(index, 'id'), thyristor_firing_angle_degree=row['thyristor_firing_angle_degree'], x_ohm=row['x_ohm'], q_mvar=row['q_mvar'], vm_pu=row['vm_pu'], va_degree=row['va_degree'] )        
                        SVCsList.append(svc) 
                        svcs = SVCsOut(svcs = SVCsList) 
                    result = {**result, **svcs.__dict__}
                       
                except AttributeError:  
                    pass
                except UnboundLocalError:
                    pass
                        
                #TCSC   
                try:
                    for index, row in net.res_tcsc.iterrows():    
                            tcsc = TCSCOut(name=net.tcsc._get_value(index, 'name'), id = net.tcsc._get_value(index, 'id'), thyristor_firing_angle_degree=row['thyristor_firing_angle_degree'], x_ohm=row['x_ohm'], p_from_mw=row['p_from_mw'], q_from_mvar=row['q_from_mvar'], p_to_mw=row['p_to_mw'], q_to_mvar=row['q_to_mvar'], p_l_mw=row['p_l_mw'], q_l_mvar=row['q_l_mvar'], vm_from_pu=row['vm_from_pu'], va_from_degree=row['va_from_degree'], vm_to_pu=row['vm_to_pu'], va_to_degree=row['va_to_degree']  )        
                            TCSCsList.append(tcsc) 
                            tcscs = TCSCsOut(tcscs = TCSCsList) 
                    result = {**result, **tcscs.__dict__} 
                     
                except AttributeError:  
                    pass
                except UnboundLocalError:
                    pass

                                               
                #SSC
                
                #try:
                #    for index, row in net.res_ssc.iterrows():    
                #            ssc = SSCOut(name=net.ssc._get_value(index, 'name'), id = net.ssc._get_value(index, 'id'), q_mvar=row['q_mvar'], vm_internal_pu=row['vm_internal_pu'], va_internal_degree=row['va_internal_degree'], vm_pu=row['vm_pu'], va_degree=row['va_degree'])        
                #            sscsList.append(ssc) 
                #            sscs = SSCsOut(sscs = sscsList) 
                #    result = {**result, **sscs.__dict__}
                #except AttributeError:  
                #SSC    print("no SSC in the model")  
                     
                #SSC
                if(net.res_ssc.empty):
                    pass
                else:                    
                    for index, row in net.res_ssc.iterrows():    
                        ssc = SSCOut(name=net.ssc._get_value(index, 'name'), id = net.ssc._get_value(index, 'id'), q_mvar=row['q_mvar'], vm_internal_pu=row['vm_internal_pu'], va_internal_degree=row['va_internal_degree'], vm_pu=row['vm_pu'], va_degree=row['va_degree'])        
                        sscsList.append(ssc) 
                        sscs = SSCsOut(sscs = sscsList) 
                    result = {**result, **sscs.__dict__}                    
                       
                                        
                #DC Bus
                if(hasattr(net, 'res_bus_dc') and not net.res_bus_dc.empty):
                    for index, row in net.res_bus_dc.iterrows():    
                        if _electrisim_is_aux(net.bus_dc, index) or _electrisim_is_hidden(net.bus_dc, index):
                            continue   # a converter stage's own element, or a battery's cells
                        dcbus = DcBusOut(name=net.bus_dc.at[index, 'name'], id=_electrisim_row_id(net.bus_dc, index), vm_pu=row['vm_pu'], p_mw=row['p_mw'])        
                        dcbusesList.append(dcbus) 
                        dcbuses = DcBusesOut(dcbuses = dcbusesList) 
                    result = {**result, **DcBusesOut(dcbuses = dcbusesList).__dict__}
                
                #Load DC
                if(hasattr(net, 'res_load_dc') and not net.res_load_dc.empty):
                    for index, row in net.res_load_dc.iterrows():    
                        if _electrisim_is_aux(net.load_dc, index):
                            continue   # a DC/DC converter's input or output
                        loaddc = LoadDcOut(name=net.load_dc.at[index, 'name'], id=_electrisim_row_id(net.load_dc, index), p_mw=row['p_dc_mw'])        
                        loadsdcList.append(loaddc) 
                        loadsdc = LoadsDcOut(loadsdc = loadsdcList) 
                    result = {**result, **LoadsDcOut(loadsdc = loadsdcList).__dict__}
                
                #Source DC
                if(hasattr(net, 'res_source_dc') and not net.res_source_dc.empty):
                    # Its power from what its bus draws: res_source_dc does not give it.
                    src_i = _electrisim_source_dc_currents_ka(net)
                    for index, row in net.res_source_dc.iterrows():    
                        if _electrisim_is_hidden(net.source_dc, index):
                            continue   # a battery's cells: reported with it
                        vm_src = _electrisim_source_dc_vm(net, index)
                        p_src = (src_i[index] * vm_src * float(net.bus_dc.at[int(net.source_dc.at[index, 'bus_dc']), 'vn_kv'])
                                 if index in src_i and vm_src is not None else row['p_dc_mw'])
                        sourcedc = SourceDcOut(name=net.source_dc.at[index, 'name'], id=_electrisim_row_id(net.source_dc, index), vm_pu=vm_src, p_mw=p_src)        
                        sourcesdcList.append(sourcedc) 
                        sourcesdc = SourcesDcOut(sourcesdc = sourcesdcList) 
                    result = {**result, **sourcesdc.__dict__}
                
                # DC capacitors: their bus voltage and the energy they hold, E = C V^2 / 2.
                if getattr(net, 'electrisim_dc_capacitors', None) and hasattr(net, 'res_bus_dc'):
                    dccaps = []
                    for cap in net.electrisim_dc_capacitors:
                        if cap.get('electrisim_der'):
                            continue   # a supercapacitor: reported with the sources and stores
                        bus = cap['bus_dc']
                        vm = net.res_bus_dc.at[bus, 'vm_pu'] if bus in net.res_bus_dc.index else None
                        v_kv = vm * float(net.bus_dc.at[bus, 'vn_kv']) if vm is not None and np.isfinite(vm) else None
                        dccaps.append({
                            'name': cap['name'], 'id': cap['id'], 'vm_pu': vm,
                            'energy_kj': 0.5 * cap['c_mf'] * 1e-3 * (v_kv * 1e3) ** 2 / 1e3 if v_kv is not None and cap['in_service'] else None,
                        })
                    result = {**result, 'dccapacitors': dccaps}

                # DC breakers: open or closed, the current through each and its loading.
                if getattr(net, 'electrisim_dc_breakers', None):
                    breakers = []
                    for rec in net.electrisim_dc_breakers:
                        i_ka = _electrisim_dc_breaker_current_ka(net, rec)
                        rated = rec['rated_current_ka']
                        breakers.append({
                            'name': rec['name'], 'id': rec['id'], 'closed': rec['closed'],
                            'i_ka': i_ka if np.isfinite(i_ka) else None,
                            'loading_percent': 100.0 * i_ka / rated if rated > 0 and np.isfinite(i_ka) else None,
                            'rated_current_ka': rated, 'breaking_capacity_ka': rec['breaking_capacity_ka'],
                        })
                    result = {**result, 'dcbreakers': breakers}

                # Solid-state transformers: each stage's power, losses and loading; MV P and Q; port voltages.
                if getattr(net, 'electrisim_ssts', None):
                    result = {**result, 'ssts': [_electrisim_sst_result(net, r) for r in net.electrisim_ssts]}

                # Sources and stores: power, voltage, current and state; the PCS, and those behind them.
                pcs_out = [_electrisim_pcs_result(net, r) for r in getattr(net, 'electrisim_pcs', None) or []]
                ders_out = [_electrisim_der_result(net, r) for r in getattr(net, 'electrisim_ders', None) or []]
                ders_out += [r.pop('_source') for r in pcs_out if r.get('_source')]
                for r in pcs_out:
                    r.pop('_source', None)
                if ders_out:
                    result = {**result, 'ders': ders_out}
                if pcs_out:
                    result = {**result, 'pcs': pcs_out}

                # DC/DC converters: input and output power, losses, loading, both port voltages.
                if getattr(net, 'electrisim_dc_dc_converters', None):
                    result = {**result, 'dcdcconverters': [_electrisim_dc_dc_result(net, c)
                                                           for c in net.electrisim_dc_dc_converters]}

                #Switch (net.res_switch: p_from_mw, q_from_mvar, p_to_mw, q_to_mvar, i_ka, loading_percent)
                if(hasattr(net, 'res_switch') and not net.res_switch.empty):
                    for index, row in net.res_switch.iterrows():    
                        sw_name = net.switch._get_value(index, 'name')
                        sw_id = net.switch._get_value(index, 'id') if 'id' in net.switch.columns else str(index)
                        sw_closed = net.switch._get_value(index, 'closed') if 'closed' in net.switch.columns else True
                        fk, fpf, fqff, fpt, fqtf, fld = _electrisim_switch_res_for_output(net, index, row)
                        switch = SwitchOut(
                            name=sw_name, id=sw_id, closed=sw_closed,
                            i_ka=fk,
                            p_from_mw=fpf, q_from_mvar=fqff,
                            p_to_mw=fpt, q_to_mvar=fqtf,
                            loading_percent=fld
                        )
                        switchesList.append(switch) 
                        switches = SwitchesOut(switches = switchesList) 
                    result = {**result, **switches.__dict__}
                
                #VSC
                if(hasattr(net, 'res_vsc') and not net.res_vsc.empty):
                    for index, row in net.res_vsc.iterrows():
                        if _electrisim_is_aux(net.vsc, index):
                            continue   # a DC/DC converter's output stage
                        vsc_name = net.vsc.at[index, 'name'] if 'name' in net.vsc.columns else f'VSC_{index}'
                        vsc_id = net.vsc.at[index, 'id'] if 'id' in net.vsc.columns else str(index)
                        vsc = VSCOut(name=vsc_name, id=vsc_id, p_mw=row['p_mw'], vm_pu=row.get('vm_pu', 0.0), q_mvar=row.get('q_mvar'), p_dc_mw=row.get('p_dc_mw'), vm_dc_pu=row.get('vm_dc_pu'))        
                        vscsList.append(vsc) 
                        vscs = VSCsOut(vscs = vscsList) 
                    result = {**result, **VSCsOut(vscs = vscsList).__dict__}
                
                #B2B VSC
                if(hasattr(net, 'res_b2b_vsc') and not net.res_b2b_vsc.empty):
                    for index, row in net.res_b2b_vsc.iterrows():    
                        b2b_name = net.b2b_vsc.at[index, 'name'] if 'name' in net.b2b_vsc.columns else f'B2B_VSC_{index}'
                        b2b_id = net.b2b_vsc.at[index, 'id'] if 'id' in net.b2b_vsc.columns else str(index)
                        b2bvsc = B2bVSCOut(name=b2b_name, id=b2b_id, p_mw=row['p_mw'], q_mvar=row.get('q_mvar'), vm_pu=row.get('vm_pu'), p_dc_mw_p=row.get('p_dc_mw_p'), p_dc_mw_m=row.get('p_dc_mw_m'), vm_dc_pu_p=row.get('vm_dc_pu_p'), vm_dc_pu_m=row.get('vm_dc_pu_m'))        
                        b2bvscsList.append(b2bvsc) 
                        b2bvscs = B2bVSCsOut(b2bvscs = b2bvscsList) 
                    result = {**result, **b2bvscs.__dict__}
                
                #DCLine (old HVDC link element - pp.create_dcline)
                if(net.res_dcline.empty):
                    pass
                else:                    
                        for index, row in net.res_dcline.iterrows():    
                            dcline = ImpedanceOut(name=net.dcline._get_value(index, 'name'), id = net.dcline._get_value(index, 'id'), p_from_mw=row['p_from_mw'], q_from_mvar=row['q_from_mvar'], p_to_mw=row['p_to_mw'], q_to_mvar=row['q_to_mvar'], pl_mw=row['pl_mw'], vm_from_pu=row['vm_from_pu'], va_from_degree=row['va_from_degree'], vm_to_pu=row['vm_to_pu'], va_to_degree=row['va_to_degree'] )        
                            dclinesList.append(dcline) 
                            dclines = ImpedancesOut(dclines = dclinesList) 
                        result = {**result, **dclines.__dict__}         
                
                # DC Grid Line (line_dc element - pp.create_line_dc)
                # This is part of DC grid modeling (bus_dc + line_dc + vsc)
                if hasattr(net, 'res_line_dc') and not net.res_line_dc.empty:
                    print(f"Processing res_line_dc results: {len(net.res_line_dc)} items")
                    
                    class LineDcOut(object):
                        def __init__(self, name: str, id: str, p_from_mw: float, p_to_mw: float, pl_mw: float, 
                                     vm_from_pu: float, vm_to_pu: float, i_from_ka: float, i_to_ka: float, loading_percent: float):
                            self.name = name
                            self.id = id
                            self.p_from_mw = p_from_mw
                            self.p_to_mw = p_to_mw
                            self.pl_mw = pl_mw
                            self.vm_from_pu = vm_from_pu
                            self.vm_to_pu = vm_to_pu
                            self.i_from_ka = i_from_ka
                            self.i_to_ka = i_to_ka
                            self.loading_percent = loading_percent
                    
                    class LineDcsOut(object):
                        def __init__(self, linedcs: List[LineDcOut]):
                            self.linedcs = linedcs
                    
                    linedcsList = []
                    for index, row in net.res_line_dc.iterrows():
                        if 'electrisim_dc_breaker' in net.line_dc.columns and net.line_dc.at[index, 'electrisim_dc_breaker'] == True:
                            continue   # a DC breaker's coupler, reported with the breakers
                        if _electrisim_is_hidden(net.line_dc, index):
                            continue   # a battery's resistance, reported with it
                        line_dc_name = net.line_dc.at[index, 'name'] if 'name' in net.line_dc.columns else f'LineDC_{index}'
                        line_dc_id = net.line_dc.at[index, 'id'] if 'id' in net.line_dc.columns else str(index)
                        
                        linedc = LineDcOut(
                            name=line_dc_name,
                            id=line_dc_id,
                            p_from_mw=row.get('p_from_mw', 0.0),
                            p_to_mw=row.get('p_to_mw', 0.0),
                            pl_mw=row.get('pl_mw', 0.0),
                            vm_from_pu=row.get('vm_from_pu', 0.0),
                            vm_to_pu=row.get('vm_to_pu', 0.0),
                            i_from_ka=row.get('i_from_ka', 0.0),
                            i_to_ka=row.get('i_to_ka', 0.0),
                            loading_percent=row.get('loading_percent', 0.0)
                        )
                        linedcsList.append(linedc)
                    
                    linedcs = LineDcsOut(linedcs=linedcsList)
                    result = {**result, **linedcs.__dict__}
                    print(f"Added {len(linedcsList)} line_dc results to response")
                           
                # Generate Python code if export is requested
                if export_python and in_data and Busbars:
                    python_code = generate_pandapower_python_code(net, in_data, Busbars, algorithm, calculate_voltage_angles, init)
                    result['pandapower_python'] = python_code
                
                # Add tap control results to the response for frontend display
                if tap_control_results:
                    result['tap_control_results'] = tap_control_results
                if shunt_control_results:
                    result['shunt_control_results'] = shunt_control_results
                if controller_fallback_warning:
                    result['controller_fallback_warning'] = controller_fallback_warning
                    print(f"[WARN] {controller_fallback_warning}")

                # Park / Wind Turbine controller summaries for results export
                try:
                    park_ctrl_results = _electrisim_build_park_controller_results(net, in_data)
                    if park_ctrl_results:
                        result['park_controller_results'] = park_ctrl_results
                    wtc_ss_results, wtc_dyn_results = _electrisim_build_wtc_controller_results(in_data)
                    if wtc_ss_results:
                        result['wind_turbine_controller_results'] = wtc_ss_results
                    if wtc_dyn_results:
                        result['wind_turbine_dynamic_controller_results'] = wtc_dyn_results
                except Exception as ctrl_ex:
                    print(f"[controllers export] summary build error: {ctrl_ex}")
                
                # Add any vm_pu validation warnings to the response
                if hasattr(net, 'warnings') and net.warnings:
                    result['warnings'] = net.warnings
                
                #json.dumps - convert a subset of Python objects into a json string
                #default: If specified, default should be a function that gets called for objects that can't otherwise be serialized. It should return a JSON encodable version of the object or raise a TypeError. If not specified, TypeError is raised. 
                # OPTIMIZED: Removed indent=4, using compact separators for ~40% size reduction
                # Sanitize NaN/Inf so the body is strict JSON (browser JSON.parse rejects NaN tokens).
                response = json.dumps(
                    _sanitize_for_strict_json(result),
                    default=_json_serialize_default,
                    allow_nan=False,
                    separators=(',', ':'),
                ) 
            
                print("Response to FRONTEND CORRECT")   
                   
                return response  


def analyze_shortcircuit_input_data(in_data):
    """
    Analyze input data for invalid short-circuit parameters that cause Ybus NaN.
    Returns a list of specific recommendations: [{element_type, name, param, message}, ...]
    """
    recommendations = []
    if not in_data or not isinstance(in_data, dict):
        return recommendations

    def _safe_float(val, default=None):
        if val is None or val == '' or val == 'null' or val == 'None':
            return default
        try:
            f = float(val)
            return f if f == f else default  # NaN check
        except (ValueError, TypeError):
            return default

    for key, elem in in_data.items():
        if not isinstance(elem, dict) or 'typ' not in elem:
            continue
        typ = elem.get('typ', '')
        name = elem.get('name', elem.get('id', str(key)))

        # External Grid: s_sc_max_mva must be > 0.1
        if typ and 'External Grid' in typ:
            s_sc_max = _safe_float(elem.get('s_sc_max_mva'))
            if s_sc_max is None or s_sc_max <= 0.1:
                recommendations.append({
                    'element_type': 'External Grid',
                    'name': name,
                    'param': 's_sc_max_mva',
                    'message': 'Set to a positive value (e.g. 10000 or 1000000 MVA). Right-click → Edit data → Short circuit parameters.'
                })
            s_sc_min = _safe_float(elem.get('s_sc_min_mva'))
            if s_sc_min is not None and s_sc_min <= 0:
                recommendations.append({
                    'element_type': 'External Grid',
                    'name': name,
                    'param': 's_sc_min_mva',
                    'message': 'Set to a positive value. Right-click → Edit data → Short circuit parameters.'
                })

        # Static Generator / Wind Turbine with async/async_doubly_fed: sn_mva must be > 0
        if typ and ('Static Generator' in typ or typ.startswith('Wind Turbine')):
            gen_type = (elem.get('generator_type') or '').lower()
            if gen_type in ('async', 'async_doubly_fed'):
                sn_mva = _safe_float(elem.get('sn_mva'))
                if sn_mva is None or sn_mva <= 0:
                    recommendations.append({
                        'element_type': 'Static Generator' if 'Static Generator' in typ else 'Wind Turbine',
                        'name': name,
                        'param': 'sn_mva',
                        'message': f'Set sn_mva (rated power) to a positive value, or change generator_type to "current_source" for PV inverters. Right-click → Edit data.'
                    })

    return recommendations


def _sc_iec_options(in_data):
    """
    The dialog's IEC 60909 settings as calc_sc takes them. They were read and
    dropped, so pandapower's defaults always applied: the 6 % LV tolerance the
    dialog defaults to (c max 1.05) ran as 10 % (c max 1.10), and Radial,
    Meshed and the inverse-Y choice did nothing.
    """
    try:
        lv_tol = int(float(in_data.get('lv_tol_percent', in_data.get('fault_impedance', 6))))
    except (TypeError, ValueError):
        lv_tol = 6
    if lv_tol not in (6, 10):
        lv_tol = 6
    topology = str(in_data.get('topology') or 'auto').strip().lower()
    if topology not in ('auto', 'radial', 'meshed'):
        topology = 'auto'
    inverse_y = in_data.get('inverse_y', True)
    if not isinstance(inverse_y, bool):
        inverse_y = str(inverse_y).strip().lower() not in ('false', '0', 'no', 'off')
    return {'lv_tol_percent': lv_tol, 'topology': topology, 'inverse_y': inverse_y}


def _three_phase_kappa(net, case, bus, tk_s, r_fault_ohm, x_fault_ohm, iec_options=None):
    """
    Each bus's peak factor kappa for a three-phase fault, as pandapower
    computes it (method C), indexed like net.res_bus_sc.

    A single-phase run computes kappa too but clears it before returning, and
    ip / (sqrt(2) Ik'') is not kappa: pandapower adds the current-source part
    (static generators, storage) without it. So run the three-phase fault on a
    copy and read kappa from the result table pandapower leaves behind.
    """
    import copy
    from pandapower.pypower.idx_bus_sc import KAPPA
    net3 = copy.deepcopy(net)
    sc.calc_sc(net3, fault='3ph', case=case, bus=bus, ip=True, ith=False, tk_s=tk_s,
               kappa_method='C', r_fault_ohm=r_fault_ohm, x_fault_ohm=x_fault_ohm,
               check_connectivity=True, branch_results=False, **(iec_options or {}))
    index = net.res_bus_sc.index
    rows = net3['_pd2ppc_lookups']['bus'][index.values]
    return pd.Series(net3['_ppc']['bus'][rows, KAPPA], index=index)


def _iec_thermal_m(kappa, tk_s, f_hz):
    """
    IEC 60909-0 factor m, the heat effect of the DC component, for a fault of
    tk_s seconds: m = (exp(4 f tk ln(kappa - 1)) - 1) / (2 f tk ln(kappa - 1)).
    pandapower's own formula, with the network frequency where it assumes
    50 Hz, and 0 as it gives for kappa > 1.99.
    """
    ln = np.log(kappa - 1.0)
    m = (np.exp(4.0 * f_hz * tk_s * ln) - 1.0) / (2.0 * f_hz * tk_s * ln)
    return m.where(kappa <= 1.99, 0.0)


def _sc_missing_machine_data(net):
    """
    In-service machines that lack what pandapower's short-circuit calculation
    reads, one line each. Without this check the calculation stops on the first
    missing column with "'DataFrame' object has no attribute 'vn_kv'", naming
    neither the element nor the field.
    """
    out = []

    def named(table, idx):
        return get_element_display_name(net, table, idx)

    def given(df, idx, column):
        if column not in df.columns:
            return False
        try:
            value = float(df.at[idx, column])
        except (TypeError, ValueError):
            return False
        return value == value and value > 0

    gen_fields = (('vn_kv', 'vn_kv (rated voltage)'),
                  ('xdss_pu', 'xdss_pu (subtransient reactance, typically 0.15-0.25)'),
                  ('cos_phi', 'cos_phi (rated power factor)'))
    for idx in net.gen.index:
        if not bool(net.gen.at[idx, 'in_service']):
            continue
        missing = [label for column, label in gen_fields if not given(net.gen, idx, column)]
        if missing:
            out.append(f"generator '{named('gen', idx)}' has no {', '.join(missing)}")
    for idx in net.sgen.index:
        if bool(net.sgen.at[idx, 'in_service']) and not given(net.sgen, idx, 'sn_mva'):
            out.append(f"static generator '{named('sgen', idx)}' has no sn_mva (rated power)")
    return out


# Three-winding vector groups pandapower can fault single-phase (pd2ppc_zero).
_SC_1PH_TRAFO3W_GROUPS = (
    {a + b + c for a in 'dy' for b in 'dy' for c in 'dy'}
    | {'ynyd', 'yndy', 'yynd', 'ydyn', 'ynynd', 'yndyn', 'yndd', 'ynyy', 'dynyn'}
)


def _sc_missing_zero_sequence_data(net):
    """
    What a single-phase fault needs and the network lacks, one line each:
    an external grid with no zero-sequence impedance (a NaN in the Ybus, and
    a diagnostic that blamed s_sc_min_mva), or a three-winding transformer in
    a vector group pandapower cannot fault.
    """
    out = []
    for idx in net.ext_grid.index:
        if not bool(net.ext_grid.at[idx, 'in_service']):
            continue
        x0x = net.ext_grid.at[idx, 'x0x_max'] if 'x0x_max' in net.ext_grid.columns else None
        try:
            ok = float(x0x) > 0
        except (TypeError, ValueError):
            ok = False
        if not ok:
            out.append(f"external grid '{get_element_display_name(net, 'ext_grid', idx)}' has no "
                       f"x0x_max (zero-sequence X0/X, typically 1-3) - set it with r0x0_max")
    if 'vector_group' in net.trafo3w.columns:
        for idx in net.trafo3w.index:
            group = str(net.trafo3w.at[idx, 'vector_group'] or '')
            if bool(net.trafo3w.at[idx, 'in_service']) and group.lower() not in _SC_1PH_TRAFO3W_GROUPS:
                out.append(f"three-winding transformer '{get_element_display_name(net, 'trafo3w', idx)}' "
                           f"has vector group {group!r}, which pandapower cannot fault single-phase; "
                           f"use one of YNynd, YNdyn, YNdd, YNyy, Dynyn")
    return out


def _sc_missing_min_case_data(net):
    """
    What a minimum-case short circuit needs and the network lacks: each line's
    end-of-fault temperature, to which IEC 60909-0 raises its resistance, and
    each grid's minimum fault level. The canvas stores 0 for both; at 0 degC
    the lines come out 8 % less resistive than at 20 degC, so the "minimum"
    currents were higher than they should be.
    """
    def positive(df, idx, column):
        try:
            value = float(df.at[idx, column])
        except (TypeError, ValueError, KeyError):
            return False
        return value == value and value > 0

    out = []
    for idx in net.line.index:
        if bool(net.line.at[idx, 'in_service']) and not positive(net.line, idx, 'endtemp_degree'):
            out.append(f"line '{get_element_display_name(net, 'line', idx)}' has no endtemp_degree "
                       f"(conductor temperature at the end of the fault, typically 80-160 degC)")
    for idx in net.ext_grid.index:
        if bool(net.ext_grid.at[idx, 'in_service']) and not positive(net.ext_grid, idx, 's_sc_min_mva'):
            out.append(f"external grid '{get_element_display_name(net, 'ext_grid', idx)}' has no "
                       f"s_sc_min_mva (minimum fault level)")
    return out


def _electrisim_sc_storage_as_current_sources(net):
    """
    pandapower's IEC 60909 calculation leaves the storage table out: a Storage
    element with a maximum short-circuit current (its converter's limit) is
    added as a current source giving it - as the ANSI study already counts it.
    """
    st = getattr(net, 'storage', None)
    if st is None or not len(st) or 'max_ik_ka' not in st.columns:
        return
    for i in st.index:
        max_ik = safe_float(st.at[i, 'max_ik_ka'], 0.0)
        if not bool(st.at[i, 'in_service']) or max_ik <= 0:
            continue
        bus = int(st.at[i, 'bus'])
        sn = math.sqrt(3.0) * float(net.bus.at[bus, 'vn_kv']) * max_ik
        idx = pp.create_sgen(net, bus, p_mw=0.0, sn_mva=sn, k=1.0, name=f"{st.at[i, 'name']} (short circuit)",
                             current_source=True, generator_type='current_source')
        for col, value in (('k', 1.0), ('current_source', True), ('generator_type', 'current_source'),
                           ('electrisim_aux', True)):
            net.sgen.at[idx, col] = value


def shortcircuit(net, in_data, in_data_full=None, export_python=False, Busbars=None):
    
    # Add diagnostic prints
    # Print key parameters
    # print("\nBus Data:")
    # print(net.bus)
        
    ensure_sgen_k(net)
    #print(net.sgen["k"])
    
    
    # print("\nShunt reactor Data:")
    # print(net.shunt)
        
    # print("\nTransformer Data:")
    # print(net.trafo)
    
    #print("\nThree-winding transformer Data:")
    # print(net.trafo3w)
        
    #print("\nExternal Grid Data:")
    #print(net.ext_grid)
    
    #print("\nLine Data:")
    #print(net.line)
    
    #print(net.bus.isna().sum())          # Check buses
    #print(net.line.isna().sum())         # Check lines
    #print(net.trafo.isna().sum())        # Check transformers
    #print(net.load.isna().sum())         # Check loads
   # print(net.sgen.isna().sum())         # Check static generators
  
    #print(net.line[net.line.isna().any(axis=1)])
    
    isolated = isolated_buses_message(net)
    if isolated:
        raise ValueError(isolated)

    pp.diagnostic(net)
    
    
    # Validate network before running calculations
    #pp.runpp(net, calculate_voltage_angles=True)    

    
    # Extract short circuit parameters with defaults
    # Frontend sends: fault_type, fault_location, fault_impedance
    # According to Pandapower docs: fault=fault_type, case=calculation_case
    fault_type = in_data.get('fault_type', '3ph')  # Frontend 'fault_type' becomes pandapower 'fault'
    fault_location = in_data.get('fault_location', 'max')  # Frontend 'fault_location' becomes pandapower 'case'
    fault_bus_mode = normalize_fault_bus_mode(in_data)
    bus = resolve_pp_fault_bus_indices(net, in_data, Busbars)
    if bus is not None and len(bus) == 1:
        bus = bus[0]
    
    # Get other parameters
    # The frontend sends the LV tolerance as 'fault_impedance'.
    iec_options = _sc_iec_options(in_data)
    ip = True
    ith = True
    tk_s = float(in_data.get('tk_s', 1.0))
    r_fault_ohm = float(in_data.get('r_fault_ohm', 0.0))
    x_fault_ohm = float(in_data.get('x_fault_ohm', 0.0))
    
    # Debug print to see what parameters are being passed
    
    # Print Pandapower version for debugging
    
    # Validate fault_type parameter - Pandapower expects specific values
    valid_fault_types = ['3ph', '2ph', '1ph']
    if fault_type not in valid_fault_types:
        fault_type = '3ph'  # Default to 3ph if invalid
    
    # Validate case parameter
    valid_cases = ['max', 'min']
    if fault_location not in valid_cases:
        fault_location = 'max'  # Default to max if invalid
 

    try:
        # Use correct parameter mapping according to Pandapower documentation
        
        # Call short circuit calculation with correct parameters including ip and ith.
        # IMPORTANT: branch_results=True is required to populate res_line_sc / res_trafo_sc / res_trafo3w_sc.
        # NOTE: return_all_currents=False (default) gives max/min per branch (simple index).
        #       return_all_currents=True gives results per (branch, fault_bus) combination (MultiIndex).
        #       For UI display, we want max/min per branch, so keep return_all_currents=False.
        # Raised inside the try so it reaches the diagnostic dialog - the
        # frontend drops a non-200 response without showing its message.
        # Long-standing defaults for grids whose zero-sequence columns are
        # absent come first, so only an explicit zero is reported below.
        ensure_ext_grid_zero_sequence_min(net)
        _electrisim_sc_storage_as_current_sources(net)
        missing_machine_data = _sc_missing_machine_data(net)
        if fault_type == '1ph':
            missing_machine_data += _sc_missing_zero_sequence_data(net)
        if fault_location == 'min':
            missing_machine_data += _sc_missing_min_case_data(net)
        if missing_machine_data:
            raise ValueError(
                'data is missing - '
                + '; '.join(missing_machine_data)
                + '. Enter it under the element\'s Short circuit parameters, '
                  'or take the element out of service'
            )
        sc.calc_sc(
            net,
            fault=fault_type,
            case=fault_location,
            bus=bus,
            ip=ip,
            ith=ith,
            tk_s=tk_s,
            kappa_method='C',
            r_fault_ohm=r_fault_ohm,
            x_fault_ohm=x_fault_ohm,
            # False ran only because pp.diagnostic above had left
            # net._is_elements_final behind; without it pandapower's gen
            # lookup fails. Isolated buses are refused above, so the check
            # changes nothing else.
            check_connectivity=True,
            branch_results=True,
            return_all_currents=False,  # Changed: False gives max/min per branch with simple index
            **iec_options,
        )
        
        # Check if ip_ka and ith_ka calculations failed (all NaN) for single-phase faults
        if fault_type == '1ph' and net.res_bus_sc['ip_ka'].isna().all() and net.res_bus_sc['ith_ka'].isna().all():
            # pandapower computes no peak or thermal current for an earth
            # fault. IEC 60909-0 allows the three-phase kappa at the same bus:
            #   ip1 = kappa * sqrt(2) * Ik1''     ith1 = Ik1'' * sqrt(m + n)
            # with n = 1 (far from generator) and m from kappa, f and tk.
            kappa = _three_phase_kappa(net, fault_location, bus, tk_s, r_fault_ohm, x_fault_ohm,
                                       iec_options)
            ikss = net.res_bus_sc['ikss_ka']
            net.res_bus_sc['ip_ka'] = kappa * np.sqrt(2) * ikss
            net.res_bus_sc['ith_ka'] = ikss * np.sqrt(
                _iec_thermal_m(kappa, tk_s, float(getattr(net, 'f_hz', 50.0) or 50.0)) + 1.0)

    except Exception as e:
        
        # Capture the diagnostic output and process it
        import io
        import sys
        
        # Capture stdout to get the diagnostic output
        captured_output = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured_output
        
        try:
            # Run diagnostic again to capture the output
            pp.diagnostic(net)
        except:
            pass
        
        # Restore stdout
        sys.stdout = old_stdout
        diagnostic_output = captured_output.getvalue()
        
        # Process the diagnostic output to extract structured information
        processed_diagnostic = process_short_circuit_diagnostic(diagnostic_output, net)

        # Analyze input data for invalid parameters when error suggests Ybus/NaN
        invalid_params = []
        err_str = str(e).lower()
        if ('nan' in err_str and ('ybus' in err_str or 'calculation parameters' in err_str)) and in_data_full:
            invalid_params = analyze_shortcircuit_input_data(in_data_full)
        
        # Return a diagnostic response with both raw and processed data
        diagnostic_response = {
            "error": True,
            "message": f"Short circuit calculation failed: {str(e)}",
            "exception": str(e),
            "diagnostic": {
                "raw_output": diagnostic_output,
                "processed": processed_diagnostic,
                "invalid_parameters": invalid_params,
                "fault_type": fault_type,
                "calculation_case": fault_location,
                "bus_index": bus,
                "network_elements": {
                    "buses": len(net.bus),
                    "lines": len(net.line),
                    "transformers": len(net.trafo),
                    "generators": len(net.gen),
                    "loads": len(net.load)
                }
            }
        }
        # OPTIMIZED: Compact JSON for faster transfer
        return json.dumps(diagnostic_response, separators=(',', ':'))
    #print(net.res_line_sc) # nie uwzględniam ze względu na: Branch results are in beta mode and might not always be reliable, especially for transformers
                
    #wyrzuciłem skss_mw bo wyskakiwał błąd przy zwarciu jednofazowym
    class BusbarOut(object):
        def __init__(self, name: str, id: str, ikss_ka: float, ip_ka: float, ith_ka: float, rk_ohm: float, xk_ohm: float):
            self.name = name
            self.id = id
            self.ikss_ka = ikss_ka
            self.ip_ka = ip_ka
            self.ith_ka = ith_ka
            self.rk_ohm = rk_ohm
            self.xk_ohm = xk_ohm

    class BusbarsOut(object):
        def __init__(self, busbars: List[BusbarOut]):
            self.busbars = busbars

    busbarList: List[BusbarOut] = []

    for index, row in net.res_bus_sc.iterrows():

        # Handle ip_ka column (might not exist if ip=False)
        if 'ip_ka' in row and not math.isnan(row['ip_ka']):
            ip_ka = row['ip_ka']
        else:
            ip_ka = None

        # Handle ith_ka column (might not exist if ith=False)
        if 'ith_ka' in row and not math.isnan(row['ith_ka']):
            ith_ka = row['ith_ka']
        else:
            ith_ka = None

        busbar = BusbarOut(
            name=net.bus._get_value(index, 'name'),
            id=net.bus._get_value(index, 'id'),
            ikss_ka=row['ikss_ka'],
            ip_ka=ip_ka,
            ith_ka=ith_ka,
            rk_ohm=row['rk_ohm'],
            xk_ohm=row['xk_ohm'],
        )

        busbarList.append(busbar)

    busbars = BusbarsOut(busbars=busbarList)

    def _clean_value(v):
        """Convert NaN to None and numpy scalars to python scalars for JSON."""
        if isinstance(v, (float, np.floating)):
            if math.isnan(v):
                return None
            return float(v)
        return v

    # Start result payload with busbar data (existing behaviour)
    result = {**busbars.__dict__}

    # ------------------------------------------------------------------
    # External grid short-circuit results (from bus results at ext_grid bus)
    # ------------------------------------------------------------------
    if hasattr(net, "ext_grid") and not net.ext_grid.empty and not net.res_bus_sc.empty:
        ext_grid_sc_list = []
        for idx, ext_row in net.ext_grid.iterrows():
            bus_idx = ext_row['bus']
            if bus_idx not in net.res_bus_sc.index:
                continue
            row = net.res_bus_sc.loc[bus_idx]
            ip_ka = row['ip_ka'] if 'ip_ka' in row and not math.isnan(row['ip_ka']) else None
            ith_ka = row['ith_ka'] if 'ith_ka' in row and not math.isnan(row['ith_ka']) else None
            ext_entry = {
                "name": _clean_value(ext_row['name']) if 'name' in ext_row.index else str(idx),
                "id": _clean_value(ext_row['id']) if 'id' in ext_row.index else str(idx),
                "ikss_ka": _clean_value(row['ikss_ka']),
                "ip_ka": ip_ka,
                "ith_ka": ith_ka,
                "rk_ohm": _clean_value(row['rk_ohm']),
                "xk_ohm": _clean_value(row['xk_ohm']),
            }
            ext_grid_sc_list.append(ext_entry)
        if ext_grid_sc_list:
            result["ext_grid_sc"] = ext_grid_sc_list
            print(f"Short Circuit: Added {len(ext_grid_sc_list)} external grid SC results")

    # ------------------------------------------------------------------
    # NEW: expose short-circuit results for lines and transformers
    # ------------------------------------------------------------------
    print(f"Short Circuit: Processing branch results...")
    print(f"Short Circuit: net.res_line_sc exists: {hasattr(net, 'res_line_sc')}")
    if hasattr(net, "res_line_sc"):
        print(f"Short Circuit: net.res_line_sc shape: {net.res_line_sc.shape}")
        print(f"Short Circuit: net.res_line_sc columns: {list(net.res_line_sc.columns)}")
    print(f"Short Circuit: net.res_trafo_sc exists: {hasattr(net, 'res_trafo_sc')}")
    if hasattr(net, "res_trafo_sc"):
        print(f"Short Circuit: net.res_trafo_sc shape: {net.res_trafo_sc.shape}")
    print(f"Short Circuit: net.res_trafo3w_sc exists: {hasattr(net, 'res_trafo3w_sc')}")
    if hasattr(net, "res_trafo3w_sc"):
        print(f"Short Circuit: net.res_trafo3w_sc shape: {net.res_trafo3w_sc.shape}")

    # Lines short-circuit results (net.res_line_sc)
    if hasattr(net, "res_line_sc") and not net.res_line_sc.empty:
        print(f"Short Circuit: Processing {len(net.res_line_sc)} line SC results...")
        print(f"Short Circuit: net.line shape: {net.line.shape}")
        print(f"Short Circuit: net.res_line_sc index type: {type(net.res_line_sc.index)}")
        
        # Check if we have a MultiIndex (happens with return_all_currents=True)
        if hasattr(net.res_line_sc.index, 'levels'):
            print(f"Short Circuit: WARNING - res_line_sc has MultiIndex, will group by branch")
            # Group by first level (branch index) and take max values
            res_line_sc_grouped = net.res_line_sc.groupby(level=0).max()
        else:
            res_line_sc_grouped = net.res_line_sc
        
        lines_sc_list = []
        for idx, row in res_line_sc_grouped.iterrows():
            line_entry = {}

            # Map back to original line name and id used by the frontend
            if idx in net.line.index:
                line_entry["name"] = _clean_value(net.line.at[idx, "name"]) if "name" in net.line.columns else str(idx)
                if "id" in net.line.columns:
                    line_entry["id"] = _clean_value(net.line.at[idx, "id"])
                else:
                    # Fallback: use pandapower index if custom id is missing
                    line_entry["id"] = str(idx)
            else:
                print(f"Short Circuit: Warning - line index {idx} not found in net.line")
                continue

            # Copy all numeric result columns
            for col, val in row.items():
                line_entry[col] = _clean_value(val)

            lines_sc_list.append(line_entry)
            
            # Log first line entry
            if len(lines_sc_list) == 1:
                print(f"Short Circuit: First line SC entry with name/id: {line_entry}")

        result["lines_sc"] = lines_sc_list
        print(f"Short Circuit: Successfully processed {len(lines_sc_list)} line SC results")

    # Two-winding transformer short-circuit results (net.res_trafo_sc)
    if hasattr(net, "res_trafo_sc") and not net.res_trafo_sc.empty:
        print(f"Short Circuit: Processing {len(net.res_trafo_sc)} transformer SC results...")
        print(f"Short Circuit: net.trafo shape: {net.trafo.shape}")
        print(f"Short Circuit: net.res_trafo_sc index type: {type(net.res_trafo_sc.index)}")
        
        # Check if we have a MultiIndex (happens with return_all_currents=True)
        if hasattr(net.res_trafo_sc.index, 'levels'):
            print(f"Short Circuit: WARNING - res_trafo_sc has MultiIndex, will group by branch")
            # Group by first level (branch index) and take max values
            res_trafo_sc_grouped = net.res_trafo_sc.groupby(level=0).max()
        else:
            res_trafo_sc_grouped = net.res_trafo_sc
        
        trafos_sc_list = []
        for idx, row in res_trafo_sc_grouped.iterrows():
            trafo_entry = {}

            if idx in net.trafo.index:
                trafo_entry["name"] = _clean_value(net.trafo.at[idx, "name"]) if "name" in net.trafo.columns else str(idx)
                if "id" in net.trafo.columns:
                    trafo_entry["id"] = _clean_value(net.trafo.at[idx, "id"])
                else:
                    trafo_entry["id"] = str(idx)
            else:
                print(f"Short Circuit: Warning - trafo index {idx} not found in net.trafo")
                continue

            for col, val in row.items():
                trafo_entry[col] = _clean_value(val)

            trafos_sc_list.append(trafo_entry)
            
            # Log first trafo entry
            if len(trafos_sc_list) == 1:
                print(f"Short Circuit: First trafo SC entry: {trafo_entry}")

        result["trafos_sc"] = trafos_sc_list
        print(f"Short Circuit: Successfully processed {len(trafos_sc_list)} trafo SC results")

    # Three-winding transformer short-circuit results (net.res_trafo3w_sc)
    if hasattr(net, "res_trafo3w_sc") and not net.res_trafo3w_sc.empty:
        print(f"Short Circuit: Processing {len(net.res_trafo3w_sc)} 3-winding transformer SC results...")
        
        # Check if we have a MultiIndex (happens with return_all_currents=True)
        if hasattr(net.res_trafo3w_sc.index, 'levels'):
            print(f"Short Circuit: WARNING - res_trafo3w_sc has MultiIndex, will group by branch")
            res_trafo3w_sc_grouped = net.res_trafo3w_sc.groupby(level=0).max()
        else:
            res_trafo3w_sc_grouped = net.res_trafo3w_sc
        
        trafos3w_sc_list = []
        for idx, row in res_trafo3w_sc_grouped.iterrows():
            trafo_entry = {}

            if idx in net.trafo3w.index:
                trafo_entry["name"] = _clean_value(net.trafo3w.at[idx, "name"]) if "name" in net.trafo3w.columns else str(idx)
                if "id" in net.trafo3w.columns:
                    trafo_entry["id"] = _clean_value(net.trafo3w.at[idx, "id"])
                else:
                    trafo_entry["id"] = str(idx)
            else:
                print(f"Short Circuit: Warning - trafo3w index {idx} not found in net.trafo3w")
                continue

            for col, val in row.items():
                trafo_entry[col] = _clean_value(val)

            trafos3w_sc_list.append(trafo_entry)

        result["trafos3w_sc"] = trafos3w_sc_list
        print(f"Short Circuit: Successfully processed {len(trafos3w_sc_list)} 3-winding trafo SC results")

    # Log what we're sending back
    print(f"Short Circuit: Final result keys: {list(result.keys())}")
    if "lines_sc" in result:
        print(f"Short Circuit: Sending {len(result['lines_sc'])} line SC results")
        if len(result['lines_sc']) > 0:
            print(f"Short Circuit: First line SC result: {result['lines_sc'][0]}")
    if "trafos_sc" in result:
        print(f"Short Circuit: Sending {len(result['trafos_sc'])} trafo SC results")
    if "trafos3w_sc" in result:
        print(f"Short Circuit: Sending {len(result['trafos3w_sc'])} trafo3w SC results")

    result['study'] = 'shortcircuit'
    result['engine'] = 'pandapower'
    result['study_params'] = {
        'fault_type': fault_type,
        'fault_location': fault_location,
        'fault_bus_mode': fault_bus_mode,
        'fault_bus_ids': collect_fault_bus_refs(in_data) if fault_bus_mode == 'selection' else [],
        'fault_bus_names': list(in_data.get('fault_bus_names') or []) if fault_bus_mode == 'selection' else [],
        'tk_s': tk_s,
        'r_fault_ohm': r_fault_ohm,
        'x_fault_ohm': x_fault_ohm,
        'lv_tol_percent': iec_options['lv_tol_percent'],
        'topology': iec_options['topology'],
        'inverse_y': iec_options['inverse_y'],
        'standard': 'iec60909',
    }

    if export_python and in_data_full is not None and Busbars is not None:
        try:
            python_code = generate_pandapower_python_code(
                net, in_data_full, Busbars,
                algorithm='nr', calculate_voltage_angles=True, init='auto',
                study='shortcircuit', sc_in_data=in_data,
            )
            if python_code:
                result['pandapower_python'] = python_code
        except Exception as py_err:
            print(f"Short Circuit: pandapower Python export failed: {py_err}")
            result['pandapower_python_error'] = str(py_err)

    # OPTIMIZED: Compact JSON for faster transfer
    response = json.dumps(result, default=_json_serialize_default, separators=(",", ":"))
    return response


def process_short_circuit_diagnostic(diagnostic_output, net):
    """
    Process the raw diagnostic output and extract structured information
    """
    processed = {
        "invalid_values": {},
        "overload": {},
        "nominal_voltages_dont_match": {},
        "isolated_buses": [],
        "convergence": {},
        "summary": {}
    }
    
    lines = diagnostic_output.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        
        # Detect sections
        if 'Checking for invalid_values' in line:
            current_section = 'invalid_values'
        elif 'Checking for overload' in line:
            current_section = 'overload'
        elif 'Checking for nominal_voltages_dont_match' in line:
            current_section = 'nominal_voltages_dont_match'
        elif 'Checking for isolated_buses' in line:
            current_section = 'isolated_buses'
        elif 'SUMMARY:' in line:
            current_section = 'summary'
        
        # Process invalid values
        elif current_section == 'invalid_values' and ':' in line and '=' in line:
            # Parse lines like: "Invalid value found: 'trafo 0' with attribute 'vk_percent' = 0.0"
            if "Invalid value found:" in line:
                try:
                    # Extract element type and name
                    parts = line.split("'")
                    if len(parts) >= 3:
                        element_info = parts[1]  # e.g., "trafo 0"
                        element_parts = element_info.split()
                        element_type = element_parts[0]  # e.g., "trafo"
                        
                        # Extract attribute and value
                        attr_part = line.split("attribute '")[1].split("'")[0]
                        value_part = line.split("= ")[1].split(" (")[0]
                        
                        if element_type not in processed["invalid_values"]:
                            processed["invalid_values"][element_type] = []
                        
                        # Get user-friendly name if available
                        element_index = int(element_parts[1]) if len(element_parts) > 1 else 0
                        display_name = get_element_display_name(net, element_type, element_index)
                        
                        processed["invalid_values"][element_type].append(
                            f"{display_name}: {attr_part} = {value_part}"
                        )
                except:
                    # If parsing fails, add the raw line
                    if "invalid_values" not in processed:
                        processed["invalid_values"] = {}
                    if "general" not in processed["invalid_values"]:
                        processed["invalid_values"]["general"] = []
                    processed["invalid_values"]["general"].append(line)
        
        # Process summary
        elif current_section == 'summary' and 'invalid values found' in line:
            processed["summary"]["invalid_values_count"] = line
        
        # Process other error messages
        elif 'failed' in line.lower() or 'error' in line.lower():
            if "convergence" not in processed:
                processed["convergence"] = {}
            if "errors" not in processed["convergence"]:
                processed["convergence"]["errors"] = []
            processed["convergence"]["errors"].append(line)
    
    return processed 


def _contingency_friendly_name(net, raw_name):
    """Resolve diagram userFriendlyName for contingency result labels."""
    if raw_name is None:
        return 'Unknown'
    name = str(raw_name)
    ufn_map = getattr(net, 'user_friendly_names', None)
    if isinstance(ufn_map, dict):
        friendly = ufn_map.get(name)
        if friendly not in (None, '') and str(friendly).strip():
            return str(friendly).strip()
    return name


def _contingency_bus_power(net):
    """
    Base-case load and generation at each bus, MW: what an outage cuts off
    with the bus. Storage counts as load while charging, generation while
    discharging.
    """
    load, gen = {}, {}

    def add(target, buses, values):
        for bus, value in zip(buses, values):
            if pd.notna(value) and value:
                target[bus] = target.get(bus, 0.0) + float(value)

    for table, target in (('load', load), ('motor', load), ('asymmetric_load', load),
                          ('gen', gen), ('sgen', gen), ('asymmetric_sgen', gen)):
        res = net.get('res_' + table) if hasattr(net, 'get') else getattr(net, 'res_' + table, None)
        if table in net and len(net[table]) and res is not None and len(res):
            on = net[table].index[net[table].in_service]
            p = res['p_mw'] if 'p_mw' in res else res.filter(like='p_').sum(axis=1)
            add(target, net[table].loc[on, 'bus'], p.reindex(on))
    if 'storage' in net and len(net.storage) and len(net.res_storage):
        on = net.storage.index[net.storage.in_service]
        p = net.res_storage.p_mw.reindex(on)
        add(load, net.storage.loc[on, 'bus'], p.clip(lower=0))
        add(gen, net.storage.loc[on, 'bus'], (-p).clip(lower=0))
    return load, gen


def _contingency_worst_by_element(net, contingency_results):
    """
    Each element's worst over the N-1 cases, with the outage that causes it.

    The diagram showed one case - the one with most violations - labelled
    "worst-case N-1" on every element: on the transmission grid the Wind
    farm cable's outage, with L3 at 16.9 % where the main transformer's
    outage takes it to 52 %.
    """
    buses, lines, trafos = {}, {}, {}
    for case in contingency_results:
        if not case.get('converged'):
            continue
        outage = case.get('outage') or case.get('description')
        for row in case.get('line_results', []):
            value = row.get('loading_percent')
            if value is None or pd.isna(value):
                continue
            best = lines.get(row['name'])
            if best is None or value > best['loading_percent']:
                lines[row['name']] = dict(row, worst_outage=outage)
        for row in case.get('trafo_results', []):
            value = row.get('loading_percent')
            if value is None or pd.isna(value):
                continue
            best = trafos.get(row['name'])
            if best is None or value > best['loading_percent']:
                trafos[row['name']] = dict(row, worst_outage=outage)
        for row in case.get('bus_results', []):
            entry = buses.setdefault(row['name'], {
                'bus_id': row['bus_id'], 'name': row['name'], 'vm_pu': None,
                'vm_max_pu': None, 'worst_outage': None, 'worst_outage_max': None,
                'deenergised_by': [],
            })
            value = row.get('vm_pu')
            if value is None or pd.isna(value):
                entry['deenergised_by'].append(outage)
                continue
            if entry['vm_pu'] is None or value < entry['vm_pu']:
                entry.update(vm_pu=value, va_degree=row.get('va_degree'), p_mw=row.get('p_mw'),
                             q_mvar=row.get('q_mvar'), worst_outage=outage)
            if entry['vm_max_pu'] is None or value > entry['vm_max_pu']:
                entry.update(vm_max_pu=value, worst_outage_max=outage)
    return {'bus': list(buses.values()), 'line': list(lines.values()), 'transformer': list(trafos.values())}


def contingency_analysis(net, contingency_params):
    """
    Perform contingency analysis on the network.
    
    Parameters:
    net: pandapower network
    contingency_params: dictionary containing contingency analysis parameters
    """
    try:
        # Extract parameters
        contingency_type = contingency_params.get('contingency_type', 'N-1')
        element_type = contingency_params.get('element_type', 'line')
        elements_to_analyze = contingency_params.get('elements_to_analyze', 'all')
        voltage_limits = contingency_params.get('voltage_limits', 'true') == 'true'
        thermal_limits = contingency_params.get('thermal_limits', 'true') == 'true'
        min_vm_pu = float(contingency_params.get('min_vm_pu', 0.95))
        max_vm_pu = float(contingency_params.get('max_vm_pu', 1.05))
        max_loading_percent = float(contingency_params.get('max_loading_percent', 100))
        
        # Validate network connectivity, naming the buses as the diagram does:
        # the raw set read "{np.int64(5), np.int64(6), np.int64(7)}".
        isolated = isolated_buses_message(net)
        if isolated:
            raise ValueError(isolated)

        # Check if network has elements
        
        # Run base case power flow
        _electrisim_runpp(net, algorithm='nr', calculate_voltage_angles=True)
        
        # Define contingency cases based on element type
        contingency_cases = []
        
        if element_type == 'line' or element_type == 'all':
            # Add line contingencies
            for line_idx in net.line.index:
                if net.line.loc[line_idx, 'in_service']:
                    line_name = _contingency_friendly_name(net, net.line.loc[line_idx, 'name'])
                    contingency_cases.append({
                        'name': f"Line_{line_name}",
                        'type': 'line',
                        'element_idx': line_idx,
                        'description': f"Outage of line {line_name}"
                    })
        
        if element_type == 'transformer' or element_type == 'all':
            # Add transformer contingencies
            for trafo_idx in net.trafo.index:
                if net.trafo.loc[trafo_idx, 'in_service']:
                    trafo_name = _contingency_friendly_name(net, net.trafo.loc[trafo_idx, 'name'])
                    contingency_cases.append({
                        'name': f"Trafo_{trafo_name}",
                        'type': 'trafo',
                        'element_idx': trafo_idx,
                        'description': f"Outage of transformer {trafo_name}"
                    })
            # Three-winding units too: they were never taken out, though one
            # may be a bus's only supply.
            for trafo_idx in net.trafo3w.index:
                if net.trafo3w.loc[trafo_idx, 'in_service']:
                    trafo_name = _contingency_friendly_name(net, net.trafo3w.loc[trafo_idx, 'name'])
                    contingency_cases.append({
                        'name': f"Trafo3w_{trafo_name}",
                        'type': 'trafo3w',
                        'element_idx': trafo_idx,
                        'description': f"Outage of transformer {trafo_name}"
                    })
        
        if element_type == 'generator' or element_type == 'all':
            # Add generator contingencies
            for gen_idx in net.gen.index:
                if net.gen.loc[gen_idx, 'in_service']:
                    gen_name = _contingency_friendly_name(net, net.gen.loc[gen_idx, 'name'])
                    contingency_cases.append({
                        'name': f"Gen_{gen_name}",
                        'type': 'gen',
                        'element_idx': gen_idx,
                        'description': f"Outage of generator {gen_name}"
                    })
            # Static generators too - PV, wind - which were never taken out:
            # the transmission grid's 2 MW wind farm among them.
            for gen_idx in net.sgen.index:
                if net.sgen.loc[gen_idx, 'in_service']:
                    gen_name = _contingency_friendly_name(net, net.sgen.loc[gen_idx, 'name'])
                    contingency_cases.append({
                        'name': f"Sgen_{gen_name}",
                        'type': 'sgen',
                        'element_idx': gen_idx,
                        'description': f"Outage of generator {gen_name}"
                    })
        
        # Results storage
        contingency_results = []
        violations = []
        critical_contingencies = []
        
        bus_load_mw, bus_gen_mw = _contingency_bus_power(net)

        # Store base case results
        base_case_results = {
            'bus_vm_pu': net.res_bus.vm_pu.copy(),
            'bus_va_degree': net.res_bus.va_degree.copy(),
            'line_loading_percent': net.res_line.loading_percent.copy() if not net.res_line.empty else pd.Series(),
            'trafo_loading_percent': net.res_trafo.loading_percent.copy() if not net.res_trafo.empty else pd.Series()
        }
        
        # Run contingency analysis
        for i, contingency_case in enumerate(contingency_cases):
            try:
                # Deep copy required — net.copy() returns a plain dict without .line / .bus accessors
                net_cont = deepcopy(net)
                
                # Apply contingency
                if contingency_case['type'] == 'line':
                    net_cont.line.loc[contingency_case['element_idx'], 'in_service'] = False
                elif contingency_case['type'] == 'trafo':
                    net_cont.trafo.loc[contingency_case['element_idx'], 'in_service'] = False
                elif contingency_case['type'] == 'trafo3w':
                    net_cont.trafo3w.loc[contingency_case['element_idx'], 'in_service'] = False
                elif contingency_case['type'] == 'gen':
                    net_cont.gen.loc[contingency_case['element_idx'], 'in_service'] = False
                elif contingency_case['type'] == 'sgen':
                    net_cont.sgen.loc[contingency_case['element_idx'], 'in_service'] = False
                
                # Run power flow for contingency case
                _electrisim_runpp(net_cont, algorithm='nr', calculate_voltage_angles=True)
                
                # Check for violations
                case_violations = []

                # Lost supply: buses the outage islands have no voltage (NaN),
                # which no limit catches - cutting off a bus counted as no
                # violation at all.
                in_service = net_cont.bus.index[net_cont.bus.in_service]
                dead = net_cont.res_bus.loc[net_cont.res_bus.index.intersection(in_service)]
                # How much each cut-off bus carried: every one counted the
                # same, so an outage cutting off only a wind farm ranked with
                # one dropping 1.3 MW of load.
                lost_load = lost_gen = 0.0
                for bus_idx in dead.index[dead.vm_pu.isna()]:
                    bus_name = _contingency_friendly_name(net, net_cont.bus.loc[bus_idx, 'name'])
                    load_mw, gen_mw = bus_load_mw.get(bus_idx, 0.0), bus_gen_mw.get(bus_idx, 0.0)
                    lost_load += load_mw
                    lost_gen += gen_mw
                    case_violations.append({
                        'type': 'supply',
                        'element': f"Bus_{bus_name}",
                        'description': (f'Loss of supply: bus de-energised, {load_mw:.3f} MW load '
                                        f'and {gen_mw:.3f} MW generation cut off'),
                        'severity': 'high',
                        'lost_load_mw': load_mw,
                        'lost_generation_mw': gen_mw,
                    })
                
                # Check voltage violations
                if voltage_limits:
                    voltage_violations = net_cont.res_bus[
                        (net_cont.res_bus.vm_pu < min_vm_pu) | 
                        (net_cont.res_bus.vm_pu > max_vm_pu)
                    ]
                    for bus_idx, bus_data in voltage_violations.iterrows():
                        bus_name = _contingency_friendly_name(net, net_cont.bus.loc[bus_idx, 'name'])
                        case_violations.append({
                            'type': 'voltage',
                            'element': f"Bus_{bus_name}",
                            'description': f"Voltage violation: {bus_data.vm_pu:.3f} p.u.",
                            'severity': 'high' if bus_data.vm_pu < 0.9 or bus_data.vm_pu > 1.1 else 'medium'
                        })
                
                # Check thermal violations
                if thermal_limits:
                    # Check line loading
                    if not net_cont.res_line.empty:
                        line_overloads = net_cont.res_line[
                            net_cont.res_line.loading_percent > max_loading_percent
                        ]
                        for line_idx, line_data in line_overloads.iterrows():
                            line_name = _contingency_friendly_name(net, net_cont.line.loc[line_idx, 'name'])
                            case_violations.append({
                                'type': 'thermal',
                                'element': f"Line_{line_name}",
                                'description': f"Line overload: {line_data.loading_percent:.1f}%",
                                'severity': 'high' if line_data.loading_percent > 120 else 'medium'
                            })
                    
                    # Check transformer loading
                    if not net_cont.res_trafo.empty:
                        trafo_overloads = net_cont.res_trafo[
                            net_cont.res_trafo.loading_percent > max_loading_percent
                        ]
                        for trafo_idx, trafo_data in trafo_overloads.iterrows():
                            trafo_name = _contingency_friendly_name(net, net_cont.trafo.loc[trafo_idx, 'name'])
                            case_violations.append({
                                'type': 'thermal',
                                'element': f"Trafo_{trafo_name}",
                                'description': f"Transformer overload: {trafo_data.loading_percent:.1f}%",
                                'severity': 'high' if trafo_data.loading_percent > 120 else 'medium'
                            })

                    if not net_cont.res_trafo3w.empty:
                        trafo3w_overloads = net_cont.res_trafo3w[
                            net_cont.res_trafo3w.loading_percent > max_loading_percent
                        ]
                        for trafo_idx, trafo_data in trafo3w_overloads.iterrows():
                            trafo_name = _contingency_friendly_name(net, net_cont.trafo3w.loc[trafo_idx, 'name'])
                            case_violations.append({
                                'type': 'thermal',
                                'element': f"Trafo3w_{trafo_name}",
                                'description': f"Transformer overload: {trafo_data.loading_percent:.1f}%",
                                'severity': 'high' if trafo_data.loading_percent > 120 else 'medium'
                            })
                
                # Store results for this contingency
                loadings = [v for v in list(net_cont.res_line.loading_percent)
                            + list(net_cont.res_trafo.loading_percent)
                            + list(net_cont.res_trafo3w.loading_percent) if pd.notna(v)]
                contingency_result = {
                    'name': contingency_case['name'],
                    'description': contingency_case['description'],
                    # The element taken out, by name ("Outage of line L4" -> "L4").
                    'outage': contingency_case['description'].split(' ', 3)[-1],
                    'converged': True,
                    'violations': case_violations,
                    'lost_load_mw': lost_load,
                    'lost_generation_mw': lost_gen,
                    'max_loading_percent': max(loadings) if loadings else 0.0,
                    'bus_results': [],
                    'line_results': [],
                    'trafo_results': []
                }
                
                # Store bus results
                for bus_idx, bus_data in net_cont.res_bus.iterrows():
                    if _electrisim_is_aux(net_cont.bus, bus_idx):
                        continue   # a DC/DC converter's auxiliary AC bus
                    contingency_result['bus_results'].append({
                        'bus_id': net_cont.bus.loc[bus_idx, 'id'],
                        'name': _contingency_friendly_name(net, net_cont.bus.loc[bus_idx, 'name']),
                        'vm_pu': bus_data.vm_pu,
                        'va_degree': bus_data.va_degree,
                        'p_mw': bus_data.p_mw,
                        'q_mvar': bus_data.q_mvar
                    })
                
                # Store line results
                for line_idx, line_data in net_cont.res_line.iterrows():
                    contingency_result['line_results'].append({
                        'line_id': net_cont.line.loc[line_idx, 'id'],
                        'name': _contingency_friendly_name(net, net_cont.line.loc[line_idx, 'name']),
                        'loading_percent': line_data.loading_percent,
                        'p_from_mw': line_data.p_from_mw,
                        'q_from_mvar': line_data.q_from_mvar,
                        'p_to_mw': line_data.p_to_mw,
                        'q_to_mvar': line_data.q_to_mvar
                    })
                
                # Store transformer results
                for trafo_idx, trafo_data in net_cont.res_trafo.iterrows():
                    contingency_result['trafo_results'].append({
                        'trafo_id': net_cont.trafo.loc[trafo_idx, 'id'],
                        'name': _contingency_friendly_name(net, net_cont.trafo.loc[trafo_idx, 'name']),
                        'loading_percent': trafo_data.loading_percent,
                        'p_hv_mw': trafo_data.p_hv_mw,
                        'q_hv_mvar': trafo_data.q_hv_mvar,
                        'p_lv_mw': trafo_data.p_lv_mw,
                        'q_lv_mvar': trafo_data.q_lv_mvar
                    })
                for trafo_idx, trafo_data in net_cont.res_trafo3w.iterrows():
                    contingency_result['trafo_results'].append({
                        'trafo_id': net_cont.trafo3w.loc[trafo_idx, 'id'] if 'id' in net_cont.trafo3w.columns else str(trafo_idx),
                        'name': _contingency_friendly_name(net, net_cont.trafo3w.loc[trafo_idx, 'name']),
                        'loading_percent': trafo_data.loading_percent,
                        'p_hv_mw': trafo_data.p_hv_mw,
                        'q_hv_mvar': trafo_data.q_hv_mvar,
                        'p_lv_mw': trafo_data.p_lv_mw,
                        'q_lv_mvar': trafo_data.q_lv_mvar
                    })
                
                # Add to violations list if any violations found
                if case_violations:
                    violations.extend(case_violations)
                    if any(v['severity'] == 'high' for v in case_violations):
                        critical_contingencies.append({
                            'name': contingency_case['name'],
                            'description': contingency_case['description'],
                            'violations': len(case_violations)
                        })
                
                contingency_results.append(contingency_result)
                
            except Exception as e:
                err_text = str(e)
                err_lower = err_text.lower()
                is_convergence = 'did not converge' in err_lower
                try:
                    from pandapower.auxiliary import LoadFlowNotConverged
                    is_convergence = is_convergence or isinstance(e, LoadFlowNotConverged)
                except ImportError:
                    pass

                if is_convergence:
                    failure_desc = 'Non-convergent case'
                    violation_desc = 'Power flow did not converge'
                    violation_type = 'convergence'
                else:
                    failure_desc = err_text
                    violation_desc = err_text
                    violation_type = 'error'

                contingency_result = {
                    'name': contingency_case['name'],
                    'description': contingency_case['description'],
                    'converged': False,
                    'error': err_text,
                    'violations': [{
                        'type': violation_type,
                        'element': 'System',
                        'description': violation_desc,
                        'severity': 'high'
                    }]
                }
                contingency_results.append(contingency_result)
                critical_contingencies.append({
                    'name': contingency_case['name'],
                    'description': failure_desc,
                    'violations': 1
                })
        
        # Prepare summary
        summary = {
            'contingencies_analyzed': len(contingency_cases),
            'violations': violations,
            'critical_contingencies': critical_contingencies,
            'total_violations': len(violations),
            'total_critical': len(critical_contingencies)
        }
        
        # Prepare output classes for consistent formatting
        class ContingencyBusOut(object):
            def __init__(self, bus_id: str, name: str, vm_pu: float, va_degree: float, p_mw: float, q_mvar: float):
                self.bus_id = bus_id
                self.name = name
                self.vm_pu = vm_pu
                self.va_degree = va_degree
                self.p_mw = p_mw
                self.q_mvar = q_mvar
        
        class ContingencyLineOut(object):
            def __init__(self, line_id: str, name: str, loading_percent: float, p_from_mw: float, q_from_mvar: float, p_to_mw: float, q_to_mvar: float):
                self.line_id = line_id
                self.name = name
                self.loading_percent = loading_percent
                self.p_from_mw = p_from_mw
                self.q_from_mvar = q_from_mvar
                self.p_to_mw = p_to_mw
                self.q_to_mvar = q_to_mvar
        
        class ContingencyTransformerOut(object):
            def __init__(self, trafo_id: str, name: str, loading_percent: float, p_hv_mw: float, q_hv_mvar: float, p_lv_mw: float, q_lv_mvar: float):
                self.trafo_id = trafo_id
                self.name = name
                self.loading_percent = loading_percent
                self.p_hv_mw = p_hv_mw
                self.q_hv_mvar = q_hv_mvar
                self.p_lv_mw = p_lv_mw
                self.q_lv_mvar = q_lv_mvar
        
        # Convert results to output format
        bus_out_list = []
        line_out_list = []
        trafo_out_list = []
        
        # Check if we have contingency results
        if not contingency_cases:
            error_message = f"No contingency cases found. Network has {len(net.line)} lines, {len(net.trafo)} transformers, {len(net.gen)} generators."
            return json.dumps({'error': error_message}, separators=(',', ':'))
        
        if not contingency_results:
            error_message = f"No contingency results generated. All {len(contingency_cases)} cases failed to converge."
            return json.dumps({'error': error_message}, separators=(',', ':'))
        
        # The worst case: most violations, then most load cut off, then the
        # highest loading. Violations alone tied every cut-off bus, and the
        # first such case won - one cutting off no load at all.
        worst_case = max(contingency_results, key=lambda x: (
            len(x.get('violations', [])), x.get('lost_load_mw', 0.0), x.get('max_loading_percent', 0.0)))
        
        for bus_result in worst_case.get('bus_results', []):
            bus_out = ContingencyBusOut(
                bus_id=bus_result['bus_id'],
                name=bus_result['name'],
                vm_pu=bus_result['vm_pu'],
                va_degree=bus_result['va_degree'],
                p_mw=bus_result['p_mw'],
                q_mvar=bus_result['q_mvar']
            )
            bus_out_list.append(bus_out)
        
        for line_result in worst_case.get('line_results', []):
            line_out = ContingencyLineOut(
                line_id=line_result['line_id'],
                name=line_result['name'],
                loading_percent=line_result['loading_percent'],
                p_from_mw=line_result['p_from_mw'],
                q_from_mvar=line_result['q_from_mvar'],
                p_to_mw=line_result['p_to_mw'],
                q_to_mvar=line_result['q_to_mvar']
            )
            line_out_list.append(line_out)
        
        for trafo_result in worst_case.get('trafo_results', []):
            trafo_out = ContingencyTransformerOut(
                trafo_id=trafo_result['trafo_id'],
                name=trafo_result['name'],
                loading_percent=trafo_result['loading_percent'],
                p_hv_mw=trafo_result['p_hv_mw'],
                q_hv_mvar=trafo_result['q_hv_mvar'],
                p_lv_mw=trafo_result['p_lv_mw'],
                q_lv_mvar=trafo_result['q_lv_mvar']
            )
            trafo_out_list.append(trafo_out)
        
        # Prepare result dictionary
        result = {
            'bus': [bus.__dict__ for bus in bus_out_list],
            'line': [line.__dict__ for line in line_out_list],
            'transformer': [trafo.__dict__ for trafo in trafo_out_list],
            'summary': summary,
            'contingency_results': contingency_results,
            'worst_case': worst_case.get('name'),
            'worst_by_element': _contingency_worst_by_element(net, contingency_results),
        }
        
        # Sanitize NaN/Inf so the body is strict JSON (browser JSON.parse rejects NaN tokens).
        response = json.dumps(
            _sanitize_for_strict_json(result),
            default=_json_serialize_default,
            allow_nan=False,
            separators=(',', ':'),
        )
        
        return response
        
    except Exception as e:
        error_message = f"Contingency analysis failed: {str(e)}"
        return json.dumps({'error': error_message}) 


def _truncate_solver_verbose_log(text: str, max_chars: int = 750_000) -> str:
    """Avoid oversized JSON payloads from verbose PyPower printpf output."""
    if not text or not isinstance(text, str):
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n… [output truncated — reduce diagram size or keep Suppress warnings enabled]\n"


def optimalPowerFlow(net, opf_params):
    """
    Run optimal power flow using pandapower.runopp (AC) or pandapower.rundcopp (DC)
    
    Args:
        net: pandapower network
        opf_params: dictionary containing OPF parameters from frontend
    
    Returns:
        JSON response with optimal power flow results or error message
    """
    
    solver_verbose_text = ""

    try:
        # Extract OPF parameters
        opf_type = opf_params.get('opf_type', 'ac')
        algorithm = opf_params.get('ac_algorithm', 'pypower') if opf_type == 'ac' else opf_params.get('dc_algorithm', 'pypower')
        calculate_voltage_angles = opf_params.get('calculate_voltage_angles', 'auto')
        init = opf_params.get('init', 'pf')
        _electrisim_set_aside_dc_network(net, 'Optimal power flow')
        delta = float(opf_params.get('delta', 1e-8))
        trafo_model = opf_params.get('trafo_model', 't')
        trafo_loading = opf_params.get('trafo_loading', 'current')
        ac_line_model = opf_params.get('ac_line_model', 'pi')
        numba = opf_params.get('numba', True)
        suppress_warnings = opf_params.get('suppress_warnings', True)
        cost_function = opf_params.get('cost_function', 'none')
        generator_cost_cp1 = opf_params.get('generator_cost_cp1') or {}
        if not isinstance(generator_cost_cp1, dict):
            generator_cost_cp1 = {}
        generator_cost_cp2 = opf_params.get('generator_cost_cp2') or {}
        if not isinstance(generator_cost_cp2, dict):
            generator_cost_cp2 = {}
        ext_grid_cost_cp1 = opf_params.get('ext_grid_cost_cp1') or {}
        if not isinstance(ext_grid_cost_cp1, dict):
            ext_grid_cost_cp1 = {}
        ext_grid_cost_cp2 = opf_params.get('ext_grid_cost_cp2') or {}
        if not isinstance(ext_grid_cost_cp2, dict):
            ext_grid_cost_cp2 = {}
        storage_cost_cp1 = opf_params.get('storage_cost_cp1') or {}
        if not isinstance(storage_cost_cp1, dict):
            storage_cost_cp1 = {}
        storage_cost_cp2 = opf_params.get('storage_cost_cp2') or {}
        if not isinstance(storage_cost_cp2, dict):
            storage_cost_cp2 = {}
        sgen_cost_cp1 = opf_params.get('sgen_cost_cp1') or {}
        if not isinstance(sgen_cost_cp1, dict):
            sgen_cost_cp1 = {}
        sgen_cost_cp2 = opf_params.get('sgen_cost_cp2') or {}
        if not isinstance(sgen_cost_cp2, dict):
            sgen_cost_cp2 = {}
        load_cost_cp1 = opf_params.get('load_cost_cp1') or {}
        if not isinstance(load_cost_cp1, dict):
            load_cost_cp1 = {}
        load_cost_cp2 = opf_params.get('load_cost_cp2') or {}
        if not isinstance(load_cost_cp2, dict):
            load_cost_cp2 = {}
        dcline_cost_cp1 = opf_params.get('dcline_cost_cp1') or {}
        if not isinstance(dcline_cost_cp1, dict):
            dcline_cost_cp1 = {}
        dcline_cost_cp2 = opf_params.get('dcline_cost_cp2') or {}
        if not isinstance(dcline_cost_cp2, dict):
            dcline_cost_cp2 = {}
        
        # Check for isolated buses, named as the diagram does: the raw set
        # read "{np.int64(5), np.int64(6), np.int64(7)}".
        isolated = isolated_buses_message(net)
        if isolated:
            raise ValueError(isolated)

        # Ensure min/max p_mw columns exist before default cost setup (PWL uses these breakpoints)
        if 'min_p_mw' not in net.gen.columns:
            net.gen['min_p_mw'] = 0.0
        else:
            net.gen['min_p_mw'] = net.gen['min_p_mw'].fillna(0.0)
        if 'max_p_mw' not in net.gen.columns:
            net.gen['max_p_mw'] = net.gen['p_mw'] * 1.2
        else:
            net.gen['max_p_mw'] = net.gen['max_p_mw'].where(
                net.gen['max_p_mw'].notna(), net.gen['p_mw'] * 1.2
            )
        net.gen.loc[net.gen['max_p_mw'] <= net.gen['min_p_mw'], 'max_p_mw'] = (
            net.gen.loc[net.gen['max_p_mw'] <= net.gen['min_p_mw'], 'min_p_mw'] + 1e-3
        )

        # OPF limits for external grids / storage (needed for PWL breakpoints)
        if not net.ext_grid.empty:
            if 'min_p_mw' not in net.ext_grid.columns:
                net.ext_grid['min_p_mw'] = 0.0
            else:
                net.ext_grid['min_p_mw'] = pd.to_numeric(net.ext_grid['min_p_mw'], errors='coerce').fillna(0.0)
            if 'max_p_mw' not in net.ext_grid.columns:
                net.ext_grid['max_p_mw'] = 1e6
            else:
                net.ext_grid['max_p_mw'] = pd.to_numeric(net.ext_grid['max_p_mw'], errors='coerce')
                net.ext_grid['max_p_mw'] = net.ext_grid['max_p_mw'].where(net.ext_grid['max_p_mw'].notna(), 1e6)
            slack_need_wide_cap = net.ext_grid['max_p_mw'] <= net.ext_grid['min_p_mw']
            net.ext_grid.loc[slack_need_wide_cap, 'max_p_mw'] = net.ext_grid.loc[slack_need_wide_cap, 'min_p_mw'] + 1e6

        if hasattr(net, 'storage') and not net.storage.empty:
            if 'min_p_mw' not in net.storage.columns:
                net.storage['min_p_mw'] = 0.0
            else:
                net.storage['min_p_mw'] = pd.to_numeric(net.storage['min_p_mw'], errors='coerce').fillna(0.0)
            if 'max_p_mw' not in net.storage.columns:
                net.storage['max_p_mw'] = net.storage['p_mw'].fillna(0.0).abs() * 1.2 + 1e-3
            else:
                net.storage['max_p_mw'] = pd.to_numeric(net.storage['max_p_mw'], errors='coerce')
                fb = net.storage['p_mw'].fillna(0.0).abs() * 1.2 + 1e-3
                net.storage['max_p_mw'] = net.storage['max_p_mw'].where(net.storage['max_p_mw'].notna(), fb)
            bad = net.storage['max_p_mw'] <= net.storage['min_p_mw']
            net.storage.loc[bad, 'max_p_mw'] = net.storage.loc[bad, 'min_p_mw'] + 1e-3

        # Static generators: P/Q limits for OPF (PWL breakpoints)
        if hasattr(net, 'sgen') and not net.sgen.empty:
            if 'min_p_mw' not in net.sgen.columns:
                net.sgen['min_p_mw'] = 0.0
            else:
                net.sgen['min_p_mw'] = pd.to_numeric(net.sgen['min_p_mw'], errors='coerce').fillna(0.0)
            if 'max_p_mw' not in net.sgen.columns:
                net.sgen['max_p_mw'] = net.sgen['p_mw'].fillna(0.0).abs() * 1.2 + 1e-3
            else:
                net.sgen['max_p_mw'] = pd.to_numeric(net.sgen['max_p_mw'], errors='coerce')
                fb = net.sgen['p_mw'].fillna(0.0).abs() * 1.2 + 1e-3
                net.sgen['max_p_mw'] = net.sgen['max_p_mw'].where(net.sgen['max_p_mw'].notna(), fb)
            bad = net.sgen['max_p_mw'] <= net.sgen['min_p_mw']
            net.sgen.loc[bad, 'max_p_mw'] = net.sgen.loc[bad, 'min_p_mw'] + 1e-3

        # Controllable loads / DC lines: ensure finite P (and Q) bounds before cost setup
        if not net.load.empty:
            if 'controllable' not in net.load.columns:
                net.load['controllable'] = False
            else:
                net.load['controllable'] = net.load['controllable'].fillna(value=False)

            for li in net.load.index:
                try:
                    cflag = bool(net.load.loc[li, 'controllable'])
                except (TypeError, ValueError):
                    cflag = False
                if not cflag:
                    continue
                try:
                    p0 = float(net.load.loc[li, 'p_mw'])
                except (TypeError, ValueError):
                    p0 = 0.0
                mn = net.load.loc[li, 'min_p_mw'] if 'min_p_mw' in net.load.columns else float('nan')
                mx = net.load.loc[li, 'max_p_mw'] if 'max_p_mw' in net.load.columns else float('nan')
                mn = pd.to_numeric(mn, errors='coerce')
                mx = pd.to_numeric(mx, errors='coerce')
                if pd.isna(mn):
                    mn = 0.0
                if pd.isna(mx):
                    mx = max(p0 * 1.2, p0 + 1e-3, 1e-3)
                if float(mx) <= float(mn):
                    mx = float(mn) + 1e-3
                net.load.loc[li, 'min_p_mw'] = float(mn)
                net.load.loc[li, 'max_p_mw'] = float(mx)

        if hasattr(net, 'dcline') and not net.dcline.empty:
            BIG_Q = 999999.0
            for qc in ('min_q_from_mvar', 'max_q_from_mvar', 'min_q_to_mvar', 'max_q_to_mvar'):
                if qc not in net.dcline.columns:
                    net.dcline[qc] = float('nan')
            for di in net.dcline.index:
                try:
                    p0 = float(net.dcline.loc[di, 'p_mw'])
                except (TypeError, ValueError):
                    p0 = 0.0
                mxp = net.dcline.loc[di, 'max_p_mw'] if 'max_p_mw' in net.dcline.columns else float('nan')
                mxp = pd.to_numeric(mxp, errors='coerce')
                if pd.isna(mxp) or float(mxp) <= 0:
                    net.dcline.loc[di, 'max_p_mw'] = max(abs(p0) * 1.2, 1e-3)
                for qcol, qdef in (
                    ('min_q_from_mvar', -BIG_Q),
                    ('max_q_from_mvar', BIG_Q),
                    ('min_q_to_mvar', -BIG_Q),
                    ('max_q_to_mvar', BIG_Q),
                ):
                    v = pd.to_numeric(net.dcline.loc[di, qcol], errors='coerce')
                    if pd.isna(v):
                        net.dcline.loc[di, qcol] = qdef
        
        # Set up cost functions if specified and not already present
        if cost_function != 'none':
            setup_default_cost_functions(
                net, cost_function,
                generator_cost_cp1, generator_cost_cp2,
                ext_grid_cost_cp1, ext_grid_cost_cp2,
                storage_cost_cp1, storage_cost_cp2,
                sgen_cost_cp1, sgen_cost_cp2,
                load_cost_cp1, load_cost_cp2,
                dcline_cost_cp1, dcline_cost_cp2,
            )
        
        # If an economic model was requested but no cost rows exist (edge case), fall back to polynomial gens.
        # Do NOT add costs when the user explicitly chose "none" — they expect no poly_cost / pwl_cost objective.
        if cost_function != 'none' and len(net.poly_cost) == 0 and len(net.pwl_cost) == 0:
            setup_default_cost_functions(
                net, 'polynomial',
                generator_cost_cp1, generator_cost_cp2,
                ext_grid_cost_cp1, ext_grid_cost_cp2,
                storage_cost_cp1, storage_cost_cp2,
                sgen_cost_cp1, sgen_cost_cp2,
                load_cost_cp1, load_cost_cp2,
                dcline_cost_cp1, dcline_cost_cp2,
            )
        # Optionally ensure min_q_mvar/max_q_mvar as well
        if 'min_q_mvar' not in net.gen.columns:
            net.gen['min_q_mvar'] = -9999.0
        if 'max_q_mvar' not in net.gen.columns:
            net.gen['max_q_mvar'] = 9999.0

        # AC OPF: pandapower needs finite sn_mva for some paths. Using ~1.05×P made Q limits unrealistically tight
        # vs opf_basic.ipynb (unset sn_mva), shifting voltages (~1.12 pu) while P stayed correct. Use a loose rating
        # when the diagram omits sn_mva (NaN/0) so Q does not bind before active dispatch.
        if not net.gen.empty and 'sn_mva' in net.gen.columns:
            for gi in net.gen.index:
                raw_sn = net.gen.loc[gi, 'sn_mva']
                try:
                    if pd.notna(raw_sn):
                        sn = float(raw_sn)
                        if sn > 0:
                            continue
                except (TypeError, ValueError):
                    pass
                try:
                    p0 = float(net.gen.loc[gi, 'p_mw'])
                except (TypeError, ValueError):
                    p0 = 0.0
                try:
                    pmax = float(net.gen.loc[gi, 'max_p_mw'])
                except (TypeError, ValueError):
                    pmax = abs(p0) * 1.2 if p0 != 0 else 10.0
                base_mw = max(abs(p0), abs(pmax), 1.0)
                net.gen.loc[gi, 'sn_mva'] = max(base_mw * 500.0, 5e4)

        # External grid OPF prep:
        # 1. Diagram defaults leave Q limits degenerate (0|0, NaN, empty span, inverted span) which traps slack
        #    reactive power; widen to ±BIG_Q.
        # 2. Diagram default max_p_mw=1e6 wrecks IPOPT scaling when the slack is controllable. Cap P bounds only
        #    for ext_grid rows that are already controllable (user/diagram), to something proportional to demand.
        # 3. Do **not** force controllable=True just because a poly_cost row exists: pandapower's opf_basic.ipynb
        #    keeps the slack non-controllable while still minimizing slack + gen cost; forcing True changes the
        #    OPF (notably coupled with trafo-from-parameters + manual lines) and can inflate voltages (~1.12 pu).
        if not net.ext_grid.empty:
            BIG_Q = 999999.0
            if 'min_q_mvar' not in net.ext_grid.columns:
                net.ext_grid['min_q_mvar'] = -BIG_Q
            if 'max_q_mvar' not in net.ext_grid.columns:
                net.ext_grid['max_q_mvar'] = BIG_Q
            net.ext_grid['min_q_mvar'] = pd.to_numeric(net.ext_grid['min_q_mvar'], errors='coerce')
            net.ext_grid['max_q_mvar'] = pd.to_numeric(net.ext_grid['max_q_mvar'], errors='coerce')
            for ei in net.ext_grid.index:
                qmn = net.ext_grid.loc[ei, 'min_q_mvar']
                qmx = net.ext_grid.loc[ei, 'max_q_mvar']
                widen = False
                if pd.isna(qmn) or pd.isna(qmx):
                    widen = True
                else:
                    try:
                        qmn_f = float(qmn)
                        qmx_f = float(qmx)
                    except (TypeError, ValueError):
                        widen = True
                    else:
                        if qmx_f <= qmn_f or abs(qmx_f - qmn_f) < 1e-12:
                            widen = True
                if widen:
                    net.ext_grid.loc[ei, 'min_q_mvar'] = -BIG_Q
                    net.ext_grid.loc[ei, 'max_q_mvar'] = BIG_Q

            # Reasonable slack capacity = max(loads + gens + 2× headroom, 1000 MW).
            try:
                load_total = float(net.load['p_mw'].fillna(0).abs().sum()) if not net.load.empty else 0.0
            except Exception:
                load_total = 0.0
            try:
                gen_total = float(net.gen['max_p_mw'].fillna(0).abs().sum()) if not net.gen.empty else 0.0
            except Exception:
                gen_total = 0.0
            slack_cap = max((load_total + gen_total) * 2.0, 1000.0)

            if 'controllable' in net.ext_grid.columns:
                for ei in net.ext_grid.index:
                    if not bool(net.ext_grid.loc[ei, 'controllable']):
                        continue
                    try:
                        mxp = float(net.ext_grid.loc[ei, 'max_p_mw'])
                    except (TypeError, ValueError):
                        mxp = float('inf')
                    try:
                        mnp = float(net.ext_grid.loc[ei, 'min_p_mw'])
                    except (TypeError, ValueError):
                        mnp = 0.0
                    if not (mxp == mxp) or mxp > slack_cap:
                        net.ext_grid.loc[ei, 'max_p_mw'] = slack_cap
                    if not (mnp == mnp) or mnp < -slack_cap:
                        net.ext_grid.loc[ei, 'min_p_mw'] = -slack_cap

        def _opf_element_indices_with_cost(et):
            idx = set()
            try:
                if not net.poly_cost.empty and 'et' in net.poly_cost.columns:
                    idx.update(int(i) for i in net.poly_cost.loc[net.poly_cost['et'] == et, 'element'].tolist())
                if not net.pwl_cost.empty and 'et' in net.pwl_cost.columns:
                    idx.update(int(i) for i in net.pwl_cost.loc[net.pwl_cost['et'] == et, 'element'].tolist())
            except Exception:
                pass
            return idx

        BIG_Q_GEN = 999999.0
        if hasattr(net, 'sgen') and not net.sgen.empty:
            if 'controllable' not in net.sgen.columns:
                net.sgen['controllable'] = False
            else:
                net.sgen['controllable'] = net.sgen['controllable'].fillna(value=False)
            sgen_c = _opf_element_indices_with_cost('sgen')
            for si in net.sgen.index:
                if int(si) in sgen_c:
                    net.sgen.loc[si, 'controllable'] = True
                if not bool(net.sgen.loc[si, 'controllable']):
                    continue
                if 'min_q_mvar' not in net.sgen.columns:
                    net.sgen['min_q_mvar'] = -BIG_Q_GEN
                if 'max_q_mvar' not in net.sgen.columns:
                    net.sgen['max_q_mvar'] = BIG_Q_GEN
                qmn = net.sgen.loc[si, 'min_q_mvar']
                qmx = net.sgen.loc[si, 'max_q_mvar']
                widen = False
                try:
                    qmn_f = float(pd.to_numeric(qmn, errors='coerce'))
                    qmx_f = float(pd.to_numeric(qmx, errors='coerce'))
                    if not (qmn_f == qmn_f and qmx_f == qmx_f) or qmx_f <= qmn_f or abs(qmx_f - qmn_f) < 1e-12:
                        widen = True
                except (TypeError, ValueError):
                    widen = True
                if widen:
                    net.sgen.loc[si, 'min_q_mvar'] = -BIG_Q_GEN
                    net.sgen.loc[si, 'max_q_mvar'] = BIG_Q_GEN

        if not net.load.empty:
            if 'controllable' not in net.load.columns:
                net.load['controllable'] = False
            else:
                net.load['controllable'] = net.load['controllable'].fillna(value=False)
            load_c = _opf_element_indices_with_cost('load')
            for li in net.load.index:
                if int(li) in load_c:
                    net.load.loc[li, 'controllable'] = True
                if not bool(net.load.loc[li, 'controllable']):
                    continue
                if 'min_q_mvar' not in net.load.columns:
                    net.load['min_q_mvar'] = -BIG_Q_GEN
                if 'max_q_mvar' not in net.load.columns:
                    net.load['max_q_mvar'] = BIG_Q_GEN
                qmn = net.load.loc[li, 'min_q_mvar']
                qmx = net.load.loc[li, 'max_q_mvar']
                widen = False
                try:
                    qmn_f = float(pd.to_numeric(qmn, errors='coerce'))
                    qmx_f = float(pd.to_numeric(qmx, errors='coerce'))
                    if not (qmn_f == qmn_f and qmx_f == qmx_f) or qmx_f <= qmn_f or abs(qmx_f - qmn_f) < 1e-12:
                        widen = True
                except (TypeError, ValueError):
                    widen = True
                if widen:
                    net.load.loc[li, 'min_q_mvar'] = -BIG_Q_GEN
                    net.load.loc[li, 'max_q_mvar'] = BIG_Q_GEN

        # Bus voltage limits for AC OPF: honor per-bus values from the diagram (create_bus). Where missing or
        # invalid, use loose defaults 0.8–1.2 pu so small networks still converge (see comment below).
        if not net.bus.empty:
            DEFAULT_OPF_VM_MIN = 0.8
            DEFAULT_OPF_VM_MAX = 1.2
            if 'min_vm_pu' not in net.bus.columns:
                net.bus['min_vm_pu'] = np.nan
            else:
                net.bus['min_vm_pu'] = pd.to_numeric(net.bus['min_vm_pu'], errors='coerce')
            if 'max_vm_pu' not in net.bus.columns:
                net.bus['max_vm_pu'] = np.nan
            else:
                net.bus['max_vm_pu'] = pd.to_numeric(net.bus['max_vm_pu'], errors='coerce')
            for bi in net.bus.index:
                try:
                    mn = net.bus.at[bi, 'min_vm_pu']
                    mx = net.bus.at[bi, 'max_vm_pu']
                except (KeyError, TypeError):
                    mn = mx = float('nan')
                if pd.isna(mn) or not (mn == mn) or not math.isfinite(float(mn)):
                    mn = DEFAULT_OPF_VM_MIN
                else:
                    mn = float(mn)
                if pd.isna(mx) or not (mx == mx) or not math.isfinite(float(mx)):
                    mx = DEFAULT_OPF_VM_MAX
                else:
                    mx = float(mx)
                if mx <= mn:
                    mn, mx = DEFAULT_OPF_VM_MIN, DEFAULT_OPF_VM_MAX
                net.bus.at[bi, 'min_vm_pu'] = mn
                net.bus.at[bi, 'max_vm_pu'] = mx
        
        # Run optimal power flow based on type
        from pandapower.auxiliary import OPFNotConverged as _OPFNotConverged

        def _run_opf_once(use_init, use_numba, verbose_solver):
            if opf_type == 'ac':
                pp.runopp(
                    net,
                    verbose=verbose_solver,
                    suppress_warnings=not verbose_solver,
                    delta=delta,
                    trafo_model=trafo_model,
                    trafo_loading=trafo_loading,
                    ac_line_model=ac_line_model,
                    calculate_voltage_angles=calculate_voltage_angles,
                    init=use_init,
                    numba=use_numba,
                )
            else:
                pp.rundcopp(
                    net,
                    verbose=verbose_solver,
                    suppress_warnings=not verbose_solver,
                    delta=delta,
                    trafo_model=trafo_model,
                    trafo_loading=trafo_loading,
                    calculate_voltage_angles=calculate_voltage_angles,
                    init=use_init,
                    numba=use_numba,
                )

        _solver_attempts = []
        if opf_type != 'ac':
            _solver_attempts = [(init, numba)]
        else:
            def _push(ii, nn):
                if not _solver_attempts or _solver_attempts[-1] != (ii, nn):
                    _solver_attempts.append((ii, nn))

            _push(init, numba)
            if str(init).lower() == 'pf':
                _push('flat', numba)
            if numba:
                _push('flat', False)

        buf = None
        if not suppress_warnings:
            buf = io.StringIO()

        try:
            if buf is not None:
                # pandapower.optimal_powerflow does `from sys import stdout` and passes that handle to printpf()
                # as fd=stdout. redirect_stdout() only replaces sys.stdout; the module-local `stdout` still points at
                # the process console, so verbose tables went to Flask logs but not our buffer. Point it at the
                # redirected sys.stdout while OPF runs, then restore (see pandapower optimal_powerflow.py).
                import pandapower.optimal_powerflow as _pp_opf_core

                _saved_printpf_fd = _pp_opf_core.stdout
                try:
                    for attempt_idx, (use_init, use_numba) in enumerate(_solver_attempts):
                        if attempt_idx > 0:
                            buf.write(
                                f"\n\n===== OPF retry {attempt_idx + 1}/{len(_solver_attempts)} "
                                f"(init={use_init!r}, numba={use_numba}) =====\n"
                            )
                        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                            _pp_opf_core.stdout = sys.stdout
                            try:
                                _run_opf_once(use_init, use_numba, verbose_solver=True)
                                break
                            except _OPFNotConverged:
                                if attempt_idx == len(_solver_attempts) - 1:
                                    raise
                finally:
                    _pp_opf_core.stdout = _saved_printpf_fd
                solver_verbose_text = buf.getvalue()
            else:
                for attempt_idx, (use_init, use_numba) in enumerate(_solver_attempts):
                    try:
                        _run_opf_once(use_init, use_numba, verbose_solver=False)
                        break
                    except _OPFNotConverged:
                        if attempt_idx == len(_solver_attempts) - 1:
                            raise
        except Exception:
            if buf is not None:
                solver_verbose_text = buf.getvalue()
            raise
        
        
    except Exception as e:
        from pandapower.auxiliary import OPFNotConverged
        
        # Initialize diagnostic response
        exc_text = str(e)
        if isinstance(e, OPFNotConverged) or "did not converge" in exc_text.lower():
            exc_text += (
                "\n\nTips:\n"
                "• The backend already retries AC OPF with **init=flat** (and **numba off** if needed) after a "
                "failed **pf** start — if you still see this, try **Polynomial** costs if you use piecewise linear.\n"
                "• Manually set **Initialization = flat** in the OPF dialog if you want to skip the first attempt.\n"
                "• Ensure generators have sensible **min/max active power** (per-unit voltage setpoints near 1.0)."
            )
        
        diagnostic_response = {
            "error": True,
            "message": "Optimal Power Flow calculation failed",
            "exception": exc_text,
            "diagnostic": {}
        }

        # Try to get diagnostic information
        try:
            diag_result_dict = pp.diagnostic(net, report_style='detailed')

            # Isolated buses as the load flow reports them, {index, id, name}:
            # bare indices were listed as "5, 6, 7" and could not be located.
            isolated_buses = pp.topology.unsupplied_buses(net)
            if len(isolated_buses) > 0:
                diagnostic_response["diagnostic"]["isolated_buses"] = resolve_element_refs(
                    net, 'bus', sorted(isolated_buses))

            # Process diagnostic data to convert element indices to user-friendly names
            processed_diagnostic = process_diagnostic_data(net, diag_result_dict)
            # Merge processed diagnostic with isolated_buses (don't overwrite)
            diagnostic_response["diagnostic"].update(processed_diagnostic)
                    
        except Exception as diag_error:
            pass
        
        # If no specific diagnostic was found, include the original exception
        if not diagnostic_response["diagnostic"]:
            diagnostic_response["diagnostic"]["general_error"] = str(e)

        if solver_verbose_text.strip():
            diagnostic_response["solver_verbose_log"] = _truncate_solver_verbose_log(solver_verbose_text)
        
        return _jsonify_safe(diagnostic_response)
    
    # Build response with OPF results
    else:
        # Define output classes (similar to powerflow function)
        class BusbarOut(object):
            def __init__(self, name: str, id: str, vm_pu: float, va_degree: float, p_mw: float, q_mvar: float, 
                        pf: float, q_p: float, lam_p: float = 0.0, lam_q: float = 0.0):          
                self.name = name
                self.id = id
                self.vm_pu = vm_pu
                self.va_degree = va_degree   
                self.p_mw = p_mw
                self.q_mvar = q_mvar  
                self.pf = pf
                self.q_p = q_p
                self.lam_p = lam_p  # Lagrange multiplier for active power
                self.lam_q = lam_q  # Lagrange multiplier for reactive power
        
        class LineOut(object):
            def __init__(self, name: str, id: str, p_from_mw: float, q_from_mvar: float, p_to_mw: float, q_to_mvar: float, 
                        i_from_ka: float, i_to_ka: float, loading_percent: float, mu_sf: float = 0.0, mu_st: float = 0.0):          
                self.name = name 
                self.id = id                      
                self.p_from_mw = p_from_mw
                self.q_from_mvar = q_from_mvar 
                self.p_to_mw = p_to_mw 
                self.q_to_mvar = q_to_mvar            
                self.i_from_ka = i_from_ka 
                self.i_to_ka = i_to_ka               
                self.loading_percent = loading_percent
                self.mu_sf = mu_sf  # Shadow price for from-side flow limit
                self.mu_st = mu_st  # Shadow price for to-side flow limit
        
        class GeneratorOut(object):
            def __init__(self, name: str, id: str, p_mw: float, q_mvar: float, va_degree: float, vm_pu: float,
                        gen_cost: float = 0.0, marginal_cost: float = 0.0):          
                self.name = name
                self.id = id
                self.p_mw = p_mw 
                self.q_mvar = q_mvar  
                self.va_degree = va_degree 
                self.vm_pu = vm_pu
                self.gen_cost = gen_cost        # Total generation cost
                self.marginal_cost = marginal_cost  # Marginal cost
        
        # Similar classes for other components (simplified for space)
        class ExternalGridOut(object):
            def __init__(self, name: str, id: str, p_mw: float, q_mvar: float, pf: float, q_p: float,
                        gen_cost: float = 0.0, marginal_cost: float = 0.0):
                self.name = name
                self.id = id
                self.p_mw = p_mw
                self.q_mvar = q_mvar
                self.pf = pf
                self.q_p = q_p
                self.gen_cost = gen_cost
                self.marginal_cost = marginal_cost
        
        class LoadOut(object):
            def __init__(self, name: str, id:str, p_mw: float, q_mvar: float,
                         gen_cost: float = 0.0, marginal_cost: float = 0.0):
                self.name = name
                self.id = id
                self.p_mw = p_mw
                self.q_mvar = q_mvar
                self.gen_cost = gen_cost
                self.marginal_cost = marginal_cost

        class StorageOpfOut(object):
            def __init__(self, name: str, id: str, p_mw: float, q_mvar: float,
                         gen_cost: float = 0.0, marginal_cost: float = 0.0):
                self.name = name
                self.id = id
                self.p_mw = p_mw
                self.q_mvar = q_mvar
                self.gen_cost = gen_cost
                self.marginal_cost = marginal_cost

        class SgenOpfOut(object):
            def __init__(self, name: str, id: str, p_mw: float, q_mvar: float,
                         gen_cost: float = 0.0, marginal_cost: float = 0.0):
                self.name = name
                self.id = id
                self.p_mw = p_mw
                self.q_mvar = q_mvar
                self.gen_cost = gen_cost
                self.marginal_cost = marginal_cost

        class DclineOpfOut(object):
            def __init__(self, name: str, id: str, p_mw: float,
                         p_from_mw: float = 0.0, p_to_mw: float = 0.0,
                         pl_mw: float = 0.0,
                         gen_cost: float = 0.0, marginal_cost: float = 0.0):
                self.name = name
                self.id = id
                self.p_mw = p_mw
                self.p_from_mw = p_from_mw
                self.p_to_mw = p_to_mw
                self.pl_mw = pl_mw
                self.gen_cost = gen_cost
                self.marginal_cost = marginal_cost

        # Initialize result lists
        busbarList = list()
        linesList = list()
        generatorsList = list()
        externalgridsList = list()
        storagesList = list()
        loadsList = list()
        staticgeneratorsList = list()
        dclinesList = list()
        
        # Process bus results with OPF-specific data
        for index, row in net.res_bus.iterrows():
            try:
                bus_name = net.bus.loc[index, 'name']
                bus_id = net.bus.loc[index, 'id'] if 'id' in net.bus.columns else str(index)
                
                # Get user-friendly name from stored mapping
                user_friendly_name = getattr(net, 'user_friendly_names', {}).get(bus_name, bus_name)
                
                # Calculate power values
                p_mw = row['p_mw'] if 'p_mw' in row else 0.0
                q_mvar = row['q_mvar'] if 'q_mvar' in row else 0.0
                
                # Calculate power factor
                s_mva = (p_mw**2 + q_mvar**2)**0.5
                pf = p_mw / s_mva if s_mva > 0 else 0.0
                q_p = q_mvar / p_mw if p_mw > 0 else 0.0
                
                # Get Lagrange multipliers if available
                lam_p = 0.0
                lam_q = 0.0
                if hasattr(net, 'res_bus_opf') and not net.res_bus_opf.empty:
                    if index in net.res_bus_opf.index:
                        lam_p = net.res_bus_opf.loc[index, 'lam_p'] if 'lam_p' in net.res_bus_opf.columns else 0.0
                        lam_q = net.res_bus_opf.loc[index, 'lam_q'] if 'lam_q' in net.res_bus_opf.columns else 0.0
                
                busbar = BusbarOut(
                    name=get_display_name(user_friendly_name, bus_name, 'Bus', index),
                    id=bus_id,
                    vm_pu=row['vm_pu'],
                    va_degree=row['va_degree'],
                    p_mw=p_mw,
                    q_mvar=q_mvar,
                    pf=pf,
                    q_p=q_p,
                    lam_p=lam_p,
                    lam_q=lam_q
                )
                busbarList.append(busbar)
                
            except Exception as e:
                continue
        
        # Process line results with OPF-specific data
        for index, row in net.res_line.iterrows():
            try:
                line_name = net.line.loc[index, 'name']
                line_id = net.line.loc[index, 'id'] if 'id' in net.line.columns else str(index)
                
                # Get user-friendly name from stored mapping
                user_friendly_name = getattr(net, 'user_friendly_names', {}).get(line_name, line_name)
                
                # Get shadow prices if available
                mu_sf = 0.0
                mu_st = 0.0
                if hasattr(net, 'res_line_opf') and not net.res_line_opf.empty:
                    if index in net.res_line_opf.index:
                        mu_sf = net.res_line_opf.loc[index, 'mu_sf'] if 'mu_sf' in net.res_line_opf.columns else 0.0
                        mu_st = net.res_line_opf.loc[index, 'mu_st'] if 'mu_st' in net.res_line_opf.columns else 0.0
                
                line = LineOut(
                    name=get_display_name(user_friendly_name, line_name, 'Line', index),
                    id=line_id,
                    p_from_mw=row['p_from_mw'],
                    q_from_mvar=row['q_from_mvar'],
                    p_to_mw=row['p_to_mw'],
                    q_to_mvar=row['q_to_mvar'],
                    i_from_ka=row['i_from_ka'],
                    i_to_ka=row['i_to_ka'],
                    loading_percent=row['loading_percent'],
                    mu_sf=mu_sf,
                    mu_st=mu_st
                )
                linesList.append(line)
                
            except Exception as e:
                continue
        
        # Process generator results with OPF-specific data
        for index, row in net.res_gen.iterrows():
            try:
                gen_name = net.gen.loc[index, 'name']
                gen_id = net.gen.loc[index, 'id'] if 'id' in net.gen.columns else str(index)
                
                # Get user-friendly name from stored mapping
                user_friendly_name = getattr(net, 'user_friendly_names', {}).get(gen_name, gen_name)
                
                # Calculate generation costs if available
                gen_cost = 0.0
                marginal_cost = 0.0
                
                # Get costs from poly_cost table
                pc = net.poly_cost
                if pc.empty:
                    poly_costs = pc
                elif 'et' in pc.columns:
                    poly_costs = pc[(pc['element'] == index) & (pc['et'] == 'gen')]
                else:
                    poly_costs = pc[pc['element'] == index]
                got_poly = False
                if not poly_costs.empty:
                    poly_cost_row = poly_costs.iloc[0]
                    p_gen = row['p_mw']
                    # Assuming quadratic cost: cost = c2*P^2 + c1*P + c0
                    if 'cp2_eur_per_mw2' in poly_cost_row and 'cp1_eur_per_mw' in poly_cost_row and 'cp0_eur' in poly_cost_row:
                        c2 = poly_cost_row['cp2_eur_per_mw2']
                        c1 = poly_cost_row['cp1_eur_per_mw']
                        c0 = poly_cost_row['cp0_eur']
                        gen_cost = c2 * p_gen**2 + c1 * p_gen + c0
                        marginal_cost = 2 * c2 * p_gen + c1
                        got_poly = True
                if not got_poly:
                    pw_c, pw_m = _opf_cost_from_pwl_net(net, index, 'gen', row['p_mw'])
                    gen_cost = pw_c
                    marginal_cost = pw_m
                
                generator = GeneratorOut(
                    name=get_display_name(user_friendly_name, gen_name, 'Generator', index),
                    id=gen_id,
                    p_mw=row['p_mw'],
                    q_mvar=row['q_mvar'],
                    va_degree=row['va_degree'],
                    vm_pu=row['vm_pu'],
                    gen_cost=gen_cost,
                    marginal_cost=marginal_cost
                )
                generatorsList.append(generator)
                
            except Exception as e:
                continue
        
        # Process external grid results
        for index, row in net.res_ext_grid.iterrows():
            try:
                ext_grid_name = net.ext_grid.loc[index, 'name']
                ext_grid_id = net.ext_grid.loc[index, 'id'] if 'id' in net.ext_grid.columns else str(index)
                
                # Get user-friendly name from stored mapping
                user_friendly_name = getattr(net, 'user_friendly_names', {}).get(ext_grid_name, ext_grid_name)
                
                p_mw = row['p_mw'] if 'p_mw' in row else 0.0
                q_mvar = row['q_mvar'] if 'q_mvar' in row else 0.0
                
                # Calculate power factor
                s_mva = (p_mw**2 + q_mvar**2)**0.5
                pf = p_mw / s_mva if s_mva > 0 else 0.0
                q_p = q_mvar / p_mw if p_mw > 0 else 0.0

                gen_cost = 0.0
                marginal_cost = 0.0
                pc_eg = net.poly_cost
                if pc_eg.empty:
                    poly_eg = pc_eg
                elif 'et' in pc_eg.columns:
                    poly_eg = pc_eg[(pc_eg['element'] == index) & (pc_eg['et'] == 'ext_grid')]
                else:
                    poly_eg = pc_eg[pc_eg['element'] == index]
                got_poly = False
                if not poly_eg.empty:
                    poly_cost_row = poly_eg.iloc[0]
                    p_ext = p_mw
                    if 'cp2_eur_per_mw2' in poly_cost_row and 'cp1_eur_per_mw' in poly_cost_row and 'cp0_eur' in poly_cost_row:
                        c2 = poly_cost_row['cp2_eur_per_mw2']
                        c1 = poly_cost_row['cp1_eur_per_mw']
                        c0 = poly_cost_row['cp0_eur']
                        gen_cost = c2 * p_ext**2 + c1 * p_ext + c0
                        marginal_cost = 2 * c2 * p_ext + c1
                        got_poly = True
                if not got_poly:
                    pw_c, pw_m = _opf_cost_from_pwl_net(net, index, 'ext_grid', p_mw)
                    gen_cost = pw_c
                    marginal_cost = pw_m
                
                ext_grid = ExternalGridOut(
                    name=get_display_name(user_friendly_name, ext_grid_name, 'External Grid', index),
                    id=ext_grid_id,
                    p_mw=p_mw,
                    q_mvar=q_mvar,
                    pf=pf,
                    q_p=q_p,
                    gen_cost=gen_cost,
                    marginal_cost=marginal_cost,
                )
                externalgridsList.append(ext_grid)
                
            except Exception as e:
                continue

        # Storage results (OPF dispatch when present)
        if hasattr(net, 'storage') and not net.storage.empty and hasattr(net, 'res_storage') and not net.res_storage.empty:
            for index, row in net.res_storage.iterrows():
                try:
                    stor_name = net.storage.loc[index, 'name']
                    stor_id = net.storage.loc[index, 'id'] if 'id' in net.storage.columns else str(index)
                    user_friendly_name = getattr(net, 'user_friendly_names', {}).get(stor_name, stor_name)

                    p_mw = row['p_mw'] if 'p_mw' in row else 0.0
                    q_mvar = row['q_mvar'] if 'q_mvar' in row else 0.0

                    gen_cost = 0.0
                    marginal_cost = 0.0
                    pc_st = net.poly_cost
                    if pc_st.empty:
                        poly_st = pc_st
                    elif 'et' in pc_st.columns:
                        poly_st = pc_st[(pc_st['element'] == index) & (pc_st['et'] == 'storage')]
                    else:
                        poly_st = pc_st[pc_st['element'] == index]
                    got_poly = False
                    if not poly_st.empty:
                        poly_cost_row = poly_st.iloc[0]
                        if 'cp2_eur_per_mw2' in poly_cost_row and 'cp1_eur_per_mw' in poly_cost_row and 'cp0_eur' in poly_cost_row:
                            c2 = poly_cost_row['cp2_eur_per_mw2']
                            c1 = poly_cost_row['cp1_eur_per_mw']
                            c0 = poly_cost_row['cp0_eur']
                            gen_cost = c2 * p_mw**2 + c1 * p_mw + c0
                            marginal_cost = 2 * c2 * p_mw + c1
                            got_poly = True
                    if not got_poly:
                        pw_c, pw_m = _opf_cost_from_pwl_net(net, index, 'storage', p_mw)
                        gen_cost = pw_c
                        marginal_cost = pw_m

                    storagesList.append(StorageOpfOut(
                        name=get_display_name(user_friendly_name, stor_name, 'Storage', index),
                        id=stor_id,
                        p_mw=p_mw,
                        q_mvar=q_mvar,
                        gen_cost=gen_cost,
                        marginal_cost=marginal_cost,
                    ))
                except Exception:
                    continue
        
        # Static generator OPF results
        if hasattr(net, 'sgen') and not net.sgen.empty and hasattr(net, 'res_sgen') and not net.res_sgen.empty:
            for index, row in net.res_sgen.iterrows():
                try:
                    sg_name = net.sgen.loc[index, 'name']
                    sg_id = net.sgen.loc[index, 'id'] if 'id' in net.sgen.columns else str(index)
                    user_friendly_name = getattr(net, 'user_friendly_names', {}).get(sg_name, sg_name)
                    p_mw = row['p_mw'] if 'p_mw' in row else 0.0
                    q_mvar = row['q_mvar'] if 'q_mvar' in row else 0.0
                    gen_cost = 0.0
                    marginal_cost = 0.0
                    pc_sg = net.poly_cost
                    if pc_sg.empty:
                        poly_sg = pc_sg
                    elif 'et' in pc_sg.columns:
                        poly_sg = pc_sg[(pc_sg['element'] == index) & (pc_sg['et'] == 'sgen')]
                    else:
                        poly_sg = pc_sg[pc_sg['element'] == index]
                    got_poly = False
                    if not poly_sg.empty:
                        poly_cost_row = poly_sg.iloc[0]
                        if 'cp2_eur_per_mw2' in poly_cost_row and 'cp1_eur_per_mw' in poly_cost_row and 'cp0_eur' in poly_cost_row:
                            c2 = poly_cost_row['cp2_eur_per_mw2']
                            c1 = poly_cost_row['cp1_eur_per_mw']
                            c0 = poly_cost_row['cp0_eur']
                            gen_cost = c2 * p_mw**2 + c1 * p_mw + c0
                            marginal_cost = 2 * c2 * p_mw + c1
                            got_poly = True
                    if not got_poly:
                        pw_c, pw_m = _opf_cost_from_pwl_net(net, index, 'sgen', p_mw)
                        gen_cost = pw_c
                        marginal_cost = pw_m
                    staticgeneratorsList.append(SgenOpfOut(
                        name=get_display_name(user_friendly_name, sg_name, 'Static Generator', index),
                        id=sg_id,
                        p_mw=p_mw,
                        q_mvar=q_mvar,
                        gen_cost=gen_cost,
                        marginal_cost=marginal_cost,
                    ))
                except Exception:
                    continue

        # Process load results
        for index, row in net.res_load.iterrows():
            try:
                load_name = net.load.loc[index, 'name']
                load_id = net.load.loc[index, 'id'] if 'id' in net.load.columns else str(index)
                
                # Get user-friendly name from stored mapping
                user_friendly_name = getattr(net, 'user_friendly_names', {}).get(load_name, load_name)
                
                p_mw = row['p_mw'] if 'p_mw' in row else 0.0
                q_mvar = row['q_mvar'] if 'q_mvar' in row else 0.0
                gen_cost = 0.0
                marginal_cost = 0.0
                pc_ld = net.poly_cost
                if pc_ld.empty:
                    poly_ld = pc_ld
                elif 'et' in pc_ld.columns:
                    poly_ld = pc_ld[(pc_ld['element'] == index) & (pc_ld['et'] == 'load')]
                else:
                    poly_ld = pc_ld[pc_ld['element'] == index]
                got_poly = False
                if not poly_ld.empty:
                    poly_cost_row = poly_ld.iloc[0]
                    if 'cp2_eur_per_mw2' in poly_cost_row and 'cp1_eur_per_mw' in poly_cost_row and 'cp0_eur' in poly_cost_row:
                        c2 = poly_cost_row['cp2_eur_per_mw2']
                        c1 = poly_cost_row['cp1_eur_per_mw']
                        c0 = poly_cost_row['cp0_eur']
                        gen_cost = c2 * p_mw**2 + c1 * p_mw + c0
                        marginal_cost = 2 * c2 * p_mw + c1
                        got_poly = True
                if not got_poly:
                    pw_c, pw_m = _opf_cost_from_pwl_net(net, index, 'load', p_mw)
                    gen_cost = pw_c
                    marginal_cost = pw_m

                loadsList.append(LoadOut(
                    name=get_display_name(user_friendly_name, load_name, 'Load', index),
                    id=load_id,
                    p_mw=p_mw,
                    q_mvar=q_mvar,
                    gen_cost=gen_cost,
                    marginal_cost=marginal_cost,
                ))
                
            except Exception as e:
                continue

        # DC line (AC-model HVDC) OPF results
        if hasattr(net, 'dcline') and not net.dcline.empty and hasattr(net, 'res_dcline') and not net.res_dcline.empty:
            for index, row in net.res_dcline.iterrows():
                try:
                    dc_name = net.dcline.loc[index, 'name']
                    dc_id = net.dcline.loc[index, 'id'] if 'id' in net.dcline.columns else str(index)
                    user_friendly_name = getattr(net, 'user_friendly_names', {}).get(dc_name, dc_name)
                    try:
                        p_sched = float(net.dcline.loc[index, 'p_mw'])
                    except (TypeError, ValueError):
                        p_sched = float(row['p_from_mw']) if 'p_from_mw' in row else 0.0
                    p_from = float(row['p_from_mw']) if 'p_from_mw' in row else 0.0
                    p_to = float(row['p_to_mw']) if 'p_to_mw' in row else 0.0
                    pl = float(row['pl_mw']) if 'pl_mw' in row else 0.0
                    gen_cost = 0.0
                    marginal_cost = 0.0
                    pc_dc = net.poly_cost
                    if pc_dc.empty:
                        poly_dc = pc_dc
                    elif 'et' in pc_dc.columns:
                        poly_dc = pc_dc[(pc_dc['element'] == index) & (pc_dc['et'] == 'dcline')]
                    else:
                        poly_dc = pc_dc[pc_dc['element'] == index]
                    got_poly = False
                    if not poly_dc.empty:
                        poly_cost_row = poly_dc.iloc[0]
                        if 'cp2_eur_per_mw2' in poly_cost_row and 'cp1_eur_per_mw' in poly_cost_row and 'cp0_eur' in poly_cost_row:
                            c2 = poly_cost_row['cp2_eur_per_mw2']
                            c1 = poly_cost_row['cp1_eur_per_mw']
                            c0 = poly_cost_row['cp0_eur']
                            gen_cost = c2 * p_sched**2 + c1 * p_sched + c0
                            marginal_cost = 2 * c2 * p_sched + c1
                            got_poly = True
                    if not got_poly:
                        pw_c, pw_m = _opf_cost_from_pwl_net(net, index, 'dcline', p_sched)
                        gen_cost = pw_c
                        marginal_cost = pw_m
                    dclinesList.append(DclineOpfOut(
                        name=get_display_name(user_friendly_name, dc_name, 'DC Line', index),
                        id=dc_id,
                        p_mw=p_sched,
                        p_from_mw=p_from,
                        p_to_mw=p_to,
                        pl_mw=pl,
                        gen_cost=gen_cost,
                        marginal_cost=marginal_cost,
                    ))
                except Exception:
                    continue
        
        # Create response dictionary
        response_data = {
            'busbars': [busbar.__dict__ for busbar in busbarList],
            'lines': [line.__dict__ for line in linesList],
            'generators': [gen.__dict__ for gen in generatorsList],
            'externalgrids': [ext_grid.__dict__ for ext_grid in externalgridsList],
            'storages': [s.__dict__ for s in storagesList],
            'loads': [load.__dict__ for load in loadsList],
            'staticgenerators': [sg.__dict__ for sg in staticgeneratorsList],
            'dclines': [dc.__dict__ for dc in dclinesList],
        }
        
        # Add optimization results summary if available
        # With no cost rows pandapower minimises the total generation, and
        # res_cost is that sum in MW: it was reported as "11.4985 EUR". Report
        # it as what it is, and a cost - per hour, MW times price per MWh -
        # only when prices were used.
        priced = len(net.poly_cost) > 0 or len(net.pwl_cost) > 0
        response_data['cost_function'] = cost_function if priced else 'none'
        if hasattr(net, 'OPF_converged') and net.OPF_converged:
            response_data['opf_converged'] = True
            if hasattr(net, 'res_cost'):
                if priced:
                    response_data['total_cost'] = float(net.res_cost)
                    response_data['cost_per'] = 'h'
                else:
                    response_data['total_cost'] = None
                    response_data['objective'] = 'total_generation'
                    response_data['total_generation_mw'] = float(net.res_cost)
        else:
            response_data['opf_converged'] = False
        
        if solver_verbose_text.strip():
            response_data['solver_verbose_log'] = _truncate_solver_verbose_log(solver_verbose_text)

        # Label for UI: study currency from OPF payload (coefficient column names stay EUR-style in pandapower).
        response_data['cost_currency'] = str(opf_params.get('cost_currency') or 'EUR')
        if getattr(net, 'warnings', None):
            response_data['warnings'] = list(net.warnings)

        return _jsonify_safe(response_data)


def _lookup_generator_cost_scalar(cost_by_id, gen_row_id, default):
    """Resolve a per-generator OPF cost scalar from frontend dict keyed by diagram cell id."""
    if cost_by_id is None or len(cost_by_id) == 0:
        return default
    gid = gen_row_id
    if gid is None or (isinstance(gid, float) and pd.isna(gid)):
        return default
    candidates = [gid, str(gid)]
    try:
        if isinstance(gid, str) and gid.strip().isdigit():
            candidates.append(int(gid))
        elif isinstance(gid, (int, float)) and not isinstance(gid, bool):
            candidates.append(int(gid))
            candidates.append(str(int(gid)))
    except (ValueError, TypeError, OverflowError):
        pass
    for key in candidates:
        if key in cost_by_id:
            return safe_float(cost_by_id[key], default)
    return default


def _lookup_cp2_from_payload(cost_by_id, row_id):
    """
    Polynomial cp2 only if the frontend included that cell id in *_cost_cp2.
    Empty dict or missing id → 0 (linear-only poly_cost, as in pandapower opf_basic tutorial).
    """
    if cost_by_id is None or len(cost_by_id) == 0:
        return 0.0
    gid = row_id
    if gid is None or (isinstance(gid, float) and pd.isna(gid)):
        return 0.0
    candidates = [gid, str(gid)]
    try:
        if isinstance(gid, str) and gid.strip().isdigit():
            candidates.append(int(gid))
        elif isinstance(gid, (int, float)) and not isinstance(gid, bool):
            candidates.append(int(gid))
            candidates.append(str(int(gid)))
    except (ValueError, TypeError, OverflowError):
        pass
    for key in candidates:
        if key in cost_by_id:
            v = safe_float(cost_by_id[key], float('nan'))
            if v == v:
                return max(0.0, v)
            return 0.0
    return 0.0


def _opf_pwl_dispatch_cost(points, p_mw):
    """
    Map pandapower piecewise-linear cost *points* to (variable_cost_eur, marginal_eur_per_mw) for reporting.

    create_pwl_cost format: [[p_lo, p_hi, slope], ...] with *slope* = ∂C/∂P in €/MW on that interval
    (see pandapower.create.create_pwl_cost docstring).
    """
    p = float(p_mw)
    if points is None:
        return 0.0, 0.0
    if hasattr(points, 'tolist'):
        points = points.tolist()
    if not isinstance(points, (list, tuple)) or len(points) == 0:
        return 0.0, 0.0
    segments = []
    for seg in points:
        try:
            if seg is None or not hasattr(seg, '__len__') or len(seg) < 3:
                continue
            p1, p2, m = float(seg[0]), float(seg[1]), float(seg[2])
            if not (p1 == p1 and p2 == p2 and m == m):
                continue
            if p2 < p1:
                p1, p2 = p2, p1
            segments.append((p1, p2, m))
        except (TypeError, ValueError):
            continue
    if not segments:
        return 0.0, 0.0
    segments.sort(key=lambda s: s[0])
    tol = 1e-6
    if p < segments[0][0] - tol:
        m0 = segments[0][2]
        return m0 * p, m0
    total = 0.0
    marginal = segments[-1][2]
    for i, (p1, p2, m) in enumerate(segments):
        if p <= p2 + tol:
            for j in range(i):
                sj1, sj2, mj = segments[j]
                total += mj * (sj2 - sj1)
            total += m * (p - p1)
            marginal = m
            return total, marginal
        marginal = m
    for sj1, sj2, mj in segments:
        total += mj * (sj2 - sj1)
    _p1, p2_last, m_last = segments[-1]
    total += m_last * (p - p2_last)
    return total, m_last


def _opf_cost_from_pwl_net(net, element_idx, et, p_mw, power_type='p'):
    """Return (variable_cost, marginal) from net.pwl_cost for one element, or (0, 0) if none."""
    pw = getattr(net, 'pwl_cost', None)
    if pw is None or pw.empty:
        return 0.0, 0.0
    rows = pw
    if 'power_type' in rows.columns:
        sub = rows[rows['power_type'] == power_type]
        if not sub.empty:
            rows = sub
    if 'et' in rows.columns:
        rows = rows[(rows['element'] == element_idx) & (rows['et'] == et)]
    else:
        rows = rows[rows['element'] == element_idx]
    if rows.empty:
        return 0.0, 0.0
    try:
        return _opf_pwl_dispatch_cost(rows.iloc[0]['points'], p_mw)
    except Exception:
        return 0.0, 0.0


def _lookup_explicit_cp1(cost_by_id, row_id):
    """Finite marginal cp1 only when an entry exists for row_id (no default). Used for ext_grid/storage."""
    if not cost_by_id:
        return None
    gid = row_id
    if gid is None or (isinstance(gid, float) and pd.isna(gid)):
        return None
    candidates = [gid, str(gid)]
    try:
        if isinstance(gid, str) and gid.strip().isdigit():
            candidates.append(int(gid))
        elif isinstance(gid, (int, float)) and not isinstance(gid, bool):
            candidates.append(int(gid))
            candidates.append(str(int(gid)))
    except (ValueError, TypeError, OverflowError):
        pass
    for key in candidates:
        if key in cost_by_id:
            v = safe_float(cost_by_id[key], float('nan'))
            if v == v:
                return v
    return None


def setup_default_cost_functions(
    net,
    cost_type='polynomial',
    cp1_by_gen_id=None,
    cp2_by_gen_id=None,
    ext_cp1_by_id=None,
    ext_cp2_by_id=None,
    stor_cp1_by_id=None,
    stor_cp2_by_id=None,
    sgen_cp1_by_id=None,
    sgen_cp2_by_id=None,
    load_cp1_by_id=None,
    load_cp2_by_id=None,
    dcline_cp1_by_id=None,
    dcline_cp2_by_id=None,
):
    """
    Set up polynomial / piecewise-linear OPF costs on generators, static generators, loads, DC lines,
    and optionally on external grids and storage.

    Args:
        net: pandapower network
        cost_type: 'polynomial' or 'piecewise_linear'
        cp1_by_gen_id / cp2_by_gen_id: per synchronous generator (defaults applied when polynomial/PWL)
        sgen_cp1_by_id / load_cp1_by_id / dcline_cp1_by_id: same pattern for sgen, controllable load, dcline
        ext_cp1_by_id / stor_cp1_by_id: marginal slopes — rows are created only when a finite value is supplied
        ext_cp2_by_id / stor_cp2_by_id: optional quadratic term for polynomial mode (0 if omitted from payload)
    """
    cp1_by_gen_id = cp1_by_gen_id if isinstance(cp1_by_gen_id, dict) else {}
    cp2_by_gen_id = cp2_by_gen_id if isinstance(cp2_by_gen_id, dict) else {}
    ext_cp1_by_id = ext_cp1_by_id if isinstance(ext_cp1_by_id, dict) else {}
    ext_cp2_by_id = ext_cp2_by_id if isinstance(ext_cp2_by_id, dict) else {}
    stor_cp1_by_id = stor_cp1_by_id if isinstance(stor_cp1_by_id, dict) else {}
    stor_cp2_by_id = stor_cp2_by_id if isinstance(stor_cp2_by_id, dict) else {}
    sgen_cp1_by_id = sgen_cp1_by_id if isinstance(sgen_cp1_by_id, dict) else {}
    sgen_cp2_by_id = sgen_cp2_by_id if isinstance(sgen_cp2_by_id, dict) else {}
    load_cp1_by_id = load_cp1_by_id if isinstance(load_cp1_by_id, dict) else {}
    load_cp2_by_id = load_cp2_by_id if isinstance(load_cp2_by_id, dict) else {}
    dcline_cp1_by_id = dcline_cp1_by_id if isinstance(dcline_cp1_by_id, dict) else {}
    dcline_cp2_by_id = dcline_cp2_by_id if isinstance(dcline_cp2_by_id, dict) else {}

    def _poly_row_exists(idx, et):
        pc = net.poly_cost
        if pc.empty or 'et' not in pc.columns:
            return not pc[pc['element'] == idx].empty
        return not pc[(pc['element'] == idx) & (pc['et'] == et)].empty

    def _pwl_row_exists(idx, et):
        pw = net.pwl_cost
        if pw.empty or 'et' not in pw.columns:
            return not pw[pw['element'] == idx].empty
        return not pw[(pw['element'] == idx) & (pw['et'] == et)].empty

    if cost_type == 'polynomial':
        for gen_idx in net.gen.index:
            if not _poly_row_exists(gen_idx, 'gen'):
                row_id = net.gen.loc[gen_idx, 'id'] if 'id' in net.gen.columns else None
                cp1 = _lookup_generator_cost_scalar(cp1_by_gen_id, row_id, 20.0)
                cp2 = _lookup_cp2_from_payload(cp2_by_gen_id, row_id)
                pp.create_poly_cost(net, element=gen_idx, et='gen',
                                   cp2_eur_per_mw2=cp2,
                                   cp1_eur_per_mw=cp1,
                                   cp0_eur=0)

        if not net.ext_grid.empty:
            for eg_idx in net.ext_grid.index:
                if _poly_row_exists(eg_idx, 'ext_grid'):
                    continue
                row_id = net.ext_grid.loc[eg_idx, 'id'] if 'id' in net.ext_grid.columns else None
                cp1 = _lookup_explicit_cp1(ext_cp1_by_id, row_id)
                if cp1 is None:
                    continue
                cp2 = _lookup_cp2_from_payload(ext_cp2_by_id, row_id)
                pp.create_poly_cost(net, element=eg_idx, et='ext_grid',
                                   cp2_eur_per_mw2=cp2,
                                   cp1_eur_per_mw=cp1,
                                   cp0_eur=0)

        if hasattr(net, 'storage') and not net.storage.empty:
            for sto_idx in net.storage.index:
                if _poly_row_exists(sto_idx, 'storage'):
                    continue
                row_id = net.storage.loc[sto_idx, 'id'] if 'id' in net.storage.columns else None
                cp1 = _lookup_explicit_cp1(stor_cp1_by_id, row_id)
                if cp1 is None:
                    continue
                cp2 = _lookup_cp2_from_payload(stor_cp2_by_id, row_id)
                pp.create_poly_cost(net, element=sto_idx, et='storage',
                                   cp2_eur_per_mw2=cp2,
                                   cp1_eur_per_mw=cp1,
                                   cp0_eur=0)

        if hasattr(net, 'sgen') and not net.sgen.empty:
            for sg_idx in net.sgen.index:
                if _poly_row_exists(sg_idx, 'sgen'):
                    continue
                row_id = net.sgen.loc[sg_idx, 'id'] if 'id' in net.sgen.columns else None
                cp1 = _lookup_generator_cost_scalar(sgen_cp1_by_id, row_id, 20.0)
                cp2 = _lookup_cp2_from_payload(sgen_cp2_by_id, row_id)
                pp.create_poly_cost(net, element=sg_idx, et='sgen',
                                   cp2_eur_per_mw2=cp2,
                                   cp1_eur_per_mw=cp1,
                                   cp0_eur=0)

        if not net.load.empty:
            for ld_idx in net.load.index:
                if _poly_row_exists(ld_idx, 'load'):
                    continue
                row_id = net.load.loc[ld_idx, 'id'] if 'id' in net.load.columns else None
                cp1 = _lookup_explicit_cp1(load_cp1_by_id, row_id)
                if cp1 is None:
                    continue
                cp2 = _lookup_cp2_from_payload(load_cp2_by_id, row_id)
                pp.create_poly_cost(net, element=ld_idx, et='load',
                                   cp2_eur_per_mw2=cp2,
                                   cp1_eur_per_mw=cp1,
                                   cp0_eur=0)

        if hasattr(net, 'dcline') and not net.dcline.empty:
            for dc_idx in net.dcline.index:
                if _poly_row_exists(dc_idx, 'dcline'):
                    continue
                row_id = net.dcline.loc[dc_idx, 'id'] if 'id' in net.dcline.columns else None
                cp1 = _lookup_explicit_cp1(dcline_cp1_by_id, row_id)
                if cp1 is None:
                    continue
                cp2 = _lookup_cp2_from_payload(dcline_cp2_by_id, row_id)
                pp.create_poly_cost(net, element=dc_idx, et='dcline',
                                   cp2_eur_per_mw2=cp2,
                                   cp1_eur_per_mw=cp1,
                                   cp0_eur=0)

    elif cost_type == 'piecewise_linear':
        for gen_idx in net.gen.index:
            if not _pwl_row_exists(gen_idx, 'gen'):
                gen_max_p = float(net.gen.loc[gen_idx, 'max_p_mw']) if 'max_p_mw' in net.gen.columns else 100.0
                gen_min_p = float(net.gen.loc[gen_idx, 'min_p_mw']) if 'min_p_mw' in net.gen.columns else 0.0
                span = gen_max_p - gen_min_p
                if span <= 0:
                    gen_max_p = gen_min_p + 1.0
                    span = 1.0
                row_id = net.gen.loc[gen_idx, 'id'] if 'id' in net.gen.columns else None
                slope = _lookup_generator_cost_scalar(cp1_by_gen_id, row_id, 20.0)
                points = [[float(gen_min_p), float(gen_max_p), slope]]
                pp.create_pwl_cost(net, element=gen_idx, et='gen', points=points)

        if not net.ext_grid.empty:
            for eg_idx in net.ext_grid.index:
                if _pwl_row_exists(eg_idx, 'ext_grid'):
                    continue
                row_id = net.ext_grid.loc[eg_idx, 'id'] if 'id' in net.ext_grid.columns else None
                slope = _lookup_explicit_cp1(ext_cp1_by_id, row_id)
                if slope is None:
                    continue
                gen_max_p = float(net.ext_grid.loc[eg_idx, 'max_p_mw']) if 'max_p_mw' in net.ext_grid.columns else 1e6
                gen_min_p = float(net.ext_grid.loc[eg_idx, 'min_p_mw']) if 'min_p_mw' in net.ext_grid.columns else 0.0
                span = gen_max_p - gen_min_p
                if span <= 0:
                    gen_max_p = gen_min_p + 1.0
                    span = 1.0
                points = [[float(gen_min_p), float(gen_max_p), slope]]
                pp.create_pwl_cost(net, element=eg_idx, et='ext_grid', points=points)

        if hasattr(net, 'storage') and not net.storage.empty:
            for sto_idx in net.storage.index:
                if _pwl_row_exists(sto_idx, 'storage'):
                    continue
                row_id = net.storage.loc[sto_idx, 'id'] if 'id' in net.storage.columns else None
                slope = _lookup_explicit_cp1(stor_cp1_by_id, row_id)
                if slope is None:
                    continue
                gen_max_p = float(net.storage.loc[sto_idx, 'max_p_mw']) if 'max_p_mw' in net.storage.columns else 100.0
                gen_min_p = float(net.storage.loc[sto_idx, 'min_p_mw']) if 'min_p_mw' in net.storage.columns else 0.0
                span = gen_max_p - gen_min_p
                if span <= 0:
                    gen_max_p = gen_min_p + 1.0
                    span = 1.0
                points = [[float(gen_min_p), float(gen_max_p), slope]]
                pp.create_pwl_cost(net, element=sto_idx, et='storage', points=points)

        if hasattr(net, 'sgen') and not net.sgen.empty:
            for sg_idx in net.sgen.index:
                if _pwl_row_exists(sg_idx, 'sgen'):
                    continue
                row_id = net.sgen.loc[sg_idx, 'id'] if 'id' in net.sgen.columns else None
                slope = _lookup_generator_cost_scalar(sgen_cp1_by_id, row_id, 20.0)
                gen_max_p = float(net.sgen.loc[sg_idx, 'max_p_mw']) if 'max_p_mw' in net.sgen.columns else 100.0
                gen_min_p = float(net.sgen.loc[sg_idx, 'min_p_mw']) if 'min_p_mw' in net.sgen.columns else 0.0
                span = gen_max_p - gen_min_p
                if span <= 0:
                    gen_max_p = gen_min_p + 1.0
                    span = 1.0
                points = [[float(gen_min_p), float(gen_max_p), slope]]
                pp.create_pwl_cost(net, element=sg_idx, et='sgen', points=points)

        if not net.load.empty:
            for ld_idx in net.load.index:
                if _pwl_row_exists(ld_idx, 'load'):
                    continue
                row_id = net.load.loc[ld_idx, 'id'] if 'id' in net.load.columns else None
                slope = _lookup_explicit_cp1(load_cp1_by_id, row_id)
                if slope is None:
                    continue
                gen_max_p = float(net.load.loc[ld_idx, 'max_p_mw']) if 'max_p_mw' in net.load.columns else 1e3
                gen_min_p = float(net.load.loc[ld_idx, 'min_p_mw']) if 'min_p_mw' in net.load.columns else 0.0
                span = gen_max_p - gen_min_p
                if span <= 0:
                    gen_max_p = gen_min_p + 1.0
                    span = 1.0
                points = [[float(gen_min_p), float(gen_max_p), slope]]
                pp.create_pwl_cost(net, element=ld_idx, et='load', points=points)

        if hasattr(net, 'dcline') and not net.dcline.empty:
            for dc_idx in net.dcline.index:
                if _pwl_row_exists(dc_idx, 'dcline'):
                    continue
                row_id = net.dcline.loc[dc_idx, 'id'] if 'id' in net.dcline.columns else None
                slope = _lookup_explicit_cp1(dcline_cp1_by_id, row_id)
                if slope is None:
                    continue
                try:
                    p0 = float(net.dcline.loc[dc_idx, 'p_mw'])
                except (TypeError, ValueError):
                    p0 = 0.0
                gen_max_p = float(net.dcline.loc[dc_idx, 'max_p_mw']) if 'max_p_mw' in net.dcline.columns else float('nan')
                if not (gen_max_p == gen_max_p) or gen_max_p <= 0:
                    gen_max_p = max(abs(p0) * 1.2, 1e-3)
                gen_min_p = 0.0
                span = gen_max_p - gen_min_p
                if span <= 0:
                    gen_max_p = gen_min_p + 1.0
                points = [[float(gen_min_p), float(gen_max_p), slope]]
                pp.create_pwl_cost(net, element=dc_idx, et='dcline', points=points)


def _electrisim_optional_max_loading_percent(raw):
    """pandapower line/trafo/trafo3w: max_loading_percent constrains OPF loading when > 0; omit otherwise."""
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in ('', 'none', 'null', 'nan'):
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if v != v or v <= 0:
        return None
    return v


def safe_float(value, default=0.0):
    """Convert value to float. Returns default for None, 'null', 'None', empty string, or invalid values."""
    import math
    if value is None or value == 'null' or value == 'None' or value == '':
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    """Convert value to int with default fallback. Handles 'null', 'None', empty string.

    A decimal-formatted integer ('0.0', '2.0') is parsed as the number it is.
    int() rejects those outright, so they used to fall through to the default -
    and the default was 1, which silently turned a tap position of 0 into 1 and
    shifted the transformer ratio. Values that reach the payload that way come
    from imported models, grid cells and JSON round-trips of floats.

    int() is still tried first so anything that already converted exactly keeps
    doing so, rather than losing precision through float.
    """
    if value is None or value == 'null' or value == 'None' or value == '':
        return default
    try:
        return int(value)
    except (ValueError, TypeError, OverflowError):
        pass
    try:
        return int(float(value))
    except (ValueError, TypeError, OverflowError):
        return default

def parse_vector_group(vector_group):
    """Parse vector group to extract base group and phase shift number.
    
    Args:
        vector_group: String like 'Dyn11', 'Yd5', etc.
        
    Returns:
        tuple: (base_group, phase_shift_degrees)
    """
    if not vector_group or vector_group == 'None' or vector_group == '':
        return 'Dyn', 0
    
    # Common vector group patterns
    import re
    
    # Pattern to match vector groups with numbers (e.g., Dyn11, Yd5, Yy0)
    pattern = r'^([A-Za-z]+)(\d+)$'
    match = re.match(pattern, vector_group)
    
    if match:
        base_group = match.group(1)
        phase_shift_number = int(match.group(2))
        
        # Convert phase shift number to degrees (multiply by 30°)
        phase_shift_degrees = phase_shift_number * 30
        
        return base_group, phase_shift_degrees
    else:
        # No number found, return as-is with 0 phase shift
        return vector_group, 0

def get_display_name(user_friendly_name, technical_id, element_type, element_index, simulation_type='opf'):
    """
    Create a display name that combines user-friendly name with technical ID for uniqueness
    For controller and time series simulations, only return user-friendly name
    For other simulations (OPF), combine both for uniqueness
    """
    if simulation_type in ['controller', 'timeseries']:
        # For controller and time series simulations, only use user-friendly name
        if user_friendly_name and user_friendly_name != technical_id:
            return user_friendly_name
        else:
            # Fallback to type + index if no user-friendly name
            return f"{element_type} no. {element_index + 1}"
    else:
        # For other simulations (OPF), combine both for uniqueness
        if user_friendly_name and user_friendly_name != technical_id:
            # Use user-friendly name with technical ID in parentheses for uniqueness
            return f"{user_friendly_name} ({technical_id})"
        else:
            # Fallback to type + index if no user-friendly name
            return f"{element_type} no. {element_index + 1} ({technical_id})"

def process_diagnostic_data(net, diag_result_dict):
    """
    Process diagnostic data and convert element indices to user-friendly names
    """
    processed_diagnostic = {}
    
    # Process invalid values
    if 'invalid_values' in diag_result_dict:
        processed_invalid = {}
        invalid_data = diag_result_dict['invalid_values']
        
        # Handle different possible formats
        if isinstance(invalid_data, dict):
            for element_type, invalid_items in invalid_data.items():
                processed_items = []
                if isinstance(invalid_items, (list, tuple)):
                    for item in invalid_items:
                        if isinstance(item, (list, tuple)) and len(item) >= 4:
                            # Format: [element_index, parameter_name, current_value, constraint]
                            element_index = item[0]
                            parameter_name = item[1]
                            current_value = item[2]
                            constraint = item[3]
                            
                            # Get user-friendly name based on element type
                            element_id = get_element_display_name(net, element_type, element_index)
                            
                            # Create formatted message with element type and ID
                            element_type_display = element_type.capitalize()
                            if element_type == 'trafo':
                                element_type_display = 'Transformer'
                            elif element_type == 'trafo3w':
                                element_type_display = 'Three-Winding Transformer'
                            elif element_type == 'ext_grid':
                                element_type_display = 'External Grid'
                            elif element_type == 'gen':
                                element_type_display = 'Generator'
                            
                            formatted_item = f"{element_type_display} {element_id}: {parameter_name} = {current_value} (constraint: {constraint})"
                            processed_items.append(formatted_item)
                        else:
                            # Keep original format if not in expected format
                            processed_items.append(str(item))
                else:
                    processed_items.append(str(invalid_items))
                
                processed_invalid[element_type] = processed_items
        elif isinstance(invalid_data, (list, tuple)):
            processed_items = []
            for item in invalid_data:
                if isinstance(item, (list, tuple)) and len(item) >= 4:
                    element_index = item[0]
                    parameter_name = item[1]
                    current_value = item[2]
                    constraint = item[3]
                    element_id = get_element_display_name(net, 'unknown', element_index)
                    formatted_item = f"Element {element_id}: {parameter_name} = {current_value} (constraint: {constraint})"
                    processed_items.append(formatted_item)
                else:
                    processed_items.append(str(item))
            processed_invalid['general'] = processed_items
        else:
            processed_invalid['status'] = str(invalid_data)
        
        processed_diagnostic['invalid_values'] = processed_invalid
    
    # Process overload data
    if 'overload' in diag_result_dict:
        processed_overload = {}
        overload_data = diag_result_dict['overload']

        # Handle different possible formats of overload data
        if isinstance(overload_data, dict):
            # Expected format: dictionary with element types as keys
            for element_type, overload_items in overload_data.items():
                processed_items = []
                if isinstance(overload_items, (list, tuple)):
                    for item in overload_items:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            # Format: [element_index, loading_percent]
                            element_index = item[0]
                            loading_percent = item[1]

                            # Get user-friendly name
                            element_id = get_element_display_name(net, element_type, element_index)

                            # Create formatted message with element type and ID
                            element_type_display = element_type.capitalize()
                            if element_type == 'trafo':
                                element_type_display = 'Transformer'
                            elif element_type == 'trafo3w':
                                element_type_display = 'Three-Winding Transformer'
                            elif element_type == 'ext_grid':
                                element_type_display = 'External Grid'
                            elif element_type == 'gen':
                                element_type_display = 'Generator'

                            formatted_item = f"{element_type_display} {element_id}: Loading = {loading_percent}%"
                            processed_items.append(formatted_item)
                        else:
                            processed_items.append(str(item))
                else:
                    # If overload_items is not a list/tuple, just convert to string
                    processed_items.append(str(overload_items))

                processed_overload[element_type] = processed_items
        elif isinstance(overload_data, (list, tuple)):
            # Handle case where overload is a list directly
            processed_items = []
            for item in overload_data:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    element_index = item[0]
                    loading_percent = item[1]
                    element_id = get_element_display_name(net, 'unknown', element_index)
                    formatted_item = f"Element {element_id}: Loading = {loading_percent}%"
                    processed_items.append(formatted_item)
                else:
                    processed_items.append(str(item))
            processed_overload['general'] = processed_items
        else:
            # Handle boolean or other single values
            processed_overload['status'] = str(overload_data)

        processed_diagnostic['overload'] = processed_overload
    
    # Process nominal voltage mismatches
    if 'nominal_voltages_dont_match' in diag_result_dict:
        processed_voltage = {}
        voltage_data = diag_result_dict['nominal_voltages_dont_match']

        # Handle different possible formats
        if isinstance(voltage_data, dict):
            for element_type, voltage_items in voltage_data.items():
                processed_items = []
                if isinstance(voltage_items, (list, tuple)):
                    for item in voltage_items:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            # Format: [element_index, voltage_info]
                            element_index = item[0]
                            voltage_info = item[1]

                            # Get user-friendly name
                            element_id = get_element_display_name(net, element_type, element_index)

                            # Create formatted message with element type and ID
                            element_type_display = element_type.capitalize()
                            if element_type == 'trafo':
                                element_type_display = 'Transformer'
                            elif element_type == 'trafo3w':
                                element_type_display = 'Three-Winding Transformer'
                            elif element_type == 'ext_grid':
                                element_type_display = 'External Grid'
                            elif element_type == 'gen':
                                element_type_display = 'Generator'

                            formatted_item = f"{element_type_display} {element_id}: {voltage_info}"
                            processed_items.append(formatted_item)
                        else:
                            processed_items.append(str(item))
                else:
                    processed_items.append(str(voltage_items))

                processed_voltage[element_type] = processed_items
        elif isinstance(voltage_data, (list, tuple)):
            processed_items = []
            for item in voltage_data:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    element_index = item[0]
                    voltage_info = item[1]
                    element_id = get_element_display_name(net, 'unknown', element_index)
                    formatted_item = f"Element {element_id}: {voltage_info}"
                    processed_items.append(formatted_item)
                else:
                    processed_items.append(str(item))
            processed_voltage['general'] = processed_items
        else:
            processed_voltage['status'] = str(voltage_data)

        processed_diagnostic['nominal_voltages_dont_match'] = processed_voltage
    
    # Add other diagnostic data as-is
    for key, value in diag_result_dict.items():
        if key not in ['invalid_values', 'overload', 'nominal_voltages_dont_match']:
            processed_diagnostic[key] = value
    
    return processed_diagnostic

def _pp_element_table(net, element_type):
    """Return pandapower element DataFrame for a diagnostic element type key."""
    key = str(element_type or '').lower().rstrip('s')
    # Normalize plural / alias keys from diagnostics
    aliases = {
        'buses': 'bus', 'bus': 'bus',
        'lines': 'line', 'line': 'line',
        'trafos': 'trafo', 'trafo': 'trafo',
        'trafo3w': 'trafo3w', 'trafos3w': 'trafo3w',
        'sgens': 'sgen', 'sgen': 'sgen',
        'loads': 'load', 'load': 'load',
        'generators': 'gen', 'gens': 'gen', 'gen': 'gen',
        'ext_grid': 'ext_grid', 'ext_grids': 'ext_grid',
        'storage': 'storage', 'storages': 'storage',
        'shunt': 'shunt', 'shunts': 'shunt',
    }
    table_name = aliases.get(str(element_type or '').lower(), aliases.get(key, key))
    if not hasattr(net, table_name):
        return None
    df = getattr(net, table_name)
    if df is None or getattr(df, 'empty', True):
        return None
    return df


def get_element_ref(net, element_type, element_index):
    """
    Resolve a pandapower element index to frontend identifiers.
    Returns {index, id, name} where id is the technical mxCell_* name and
    name is the user-friendly diagram name used in the UI.
    """
    try:
        idx = int(element_index)
    except (TypeError, ValueError):
        return {
            "index": element_index,
            "id": str(element_index),
            "name": str(element_type),
        }

    technical_id = None
    try:
        df = _pp_element_table(net, element_type)
        if df is not None:
            if idx in df.index and 'name' in df.columns:
                technical_id = df.at[idx, 'name']
            elif 0 <= idx < len(df) and 'name' in df.columns:
                # Fallback for contiguous positional indices
                technical_id = df.iloc[idx]['name']
    except Exception:
        technical_id = None

    if technical_id is None or (isinstance(technical_id, float) and np.isnan(technical_id)):
        technical_id = f"{element_type}_{idx}"
    else:
        technical_id = str(technical_id)

    ufn_map = getattr(net, 'user_friendly_names', None) or {}
    friendly = ufn_map.get(technical_id, technical_id)
    if friendly is None or str(friendly).strip() == '':
        friendly = technical_id

    return {
        "index": idx,
        "id": technical_id,
        "name": str(friendly),
    }


def resolve_element_refs(net, element_type, indices):
    """Convert a list of pandapower indices to element ref dicts."""
    refs = []
    for raw in list(indices or []):
        try:
            refs.append(get_element_ref(net, element_type, raw))
        except Exception:
            refs.append({"index": raw, "id": str(raw), "name": str(raw)})
    return refs


def get_element_display_name(net, element_type, element_index):
    """
    Get user-friendly display name for an element based on its type and index
    """
    try:
        ref = get_element_ref(net, element_type, element_index)
        return ref.get('name') or ref.get('id') or f"{element_type} {element_index}"
    except Exception:
        return f"{str(element_type).capitalize()} no. {element_index}"

def controller_simulation(net, controller_params):
    """
    Run controller simulation using pandapower control module
    Based on: https://pandapower.readthedocs.io/en/latest/control/run.html#pandapower.control.run_control
    """
    
    # Try to import control module, but don't fail if not available
 
    from pandapower.control import run_control
    
    try:
        # Clear any existing controllers
        if hasattr(net, 'controller') and len(net.controller) > 0:
            net.controller = net.controller.drop(net.controller.index)
        
        # Create controllers based on parameters
        controllers = []
        
        # Use proper pandapower control module
        
        # Voltage control using generator voltage setpoints
        if controller_params.get('voltage_control', False):
            for idx, gen in net.gen.iterrows():
                if 'vm_pu' in gen and gen['vm_pu'] != 1.0:
                    # Create a simple voltage controller
                    # Note: This is a simplified controller - in a full implementation,
                    # you would use specific controller classes like VoltageController
                    pass
        
        # Tap control using transformer tap positions
        if controller_params.get('tap_control', False):
            if len(net.trafo) > 0:
                for idx, trafo in net.trafo.iterrows():
                    # Create a simple tap controller
                    # Note: This is a simplified controller - in a full implementation,
                    # you would use specific controller classes like TapController
                    pass
        
        # Run controller simulation using the proper run_control function
        run_control(net, 
                   max_iter=30,
                   continue_on_divergence=False,
                   check_each_level=True)
        
        
        # Prepare results
        class ControllerBusOut(object):
            def __init__(self, name: str, id: str, vm_pu: float, va_degree: float, p_mw: float, q_mvar: float):
                self.name = name
                self.id = id
                self.vm_pu = vm_pu
                self.va_degree = va_degree
                self.p_mw = p_mw
                self.q_mvar = q_mvar
        
        class ControllerLineOut(object):
            def __init__(self, name: str, id: str, p_from_mw: float, q_from_mvar: float, p_to_mw: float, q_to_mvar: float, 
                         i_from_ka: float, i_to_ka: float, loading_percent: float):
                self.name = name
                self.id = id
                self.p_from_mw = p_from_mw
                self.q_from_mvar = q_from_mvar
                self.p_to_mw = p_to_mw
                self.q_to_mvar = q_to_mvar
                self.i_from_ka = i_from_ka
                self.i_to_ka = i_to_ka
                self.loading_percent = loading_percent
        
        class ControllerGeneratorOut(object):
            def __init__(self, name: str, id: str, p_mw: float, q_mvar: float, va_degree: float, vm_pu: float):
                self.name = name
                self.id = id
                self.p_mw = p_mw
                self.q_mvar = q_mvar
                self.va_degree = va_degree
                self.vm_pu = vm_pu
        
        class ControllerLoadOut(object):
            def __init__(self, name: str, id: str, p_mw: float, q_mvar: float):
                self.name = name
                self.id = id
                self.p_mw = p_mw
                self.q_mvar = q_mvar
        
        # Collect results with display names (user-friendly + technical ID)
        busbars = []
        for idx, bus in net.res_bus.iterrows():
            bus_name = net.bus.loc[idx, 'name']
            # Get user-friendly name from stored mapping
            user_friendly_name = getattr(net, 'user_friendly_names', {}).get(bus_name, bus_name)
            
            busbars.append(ControllerBusOut(
                name=get_display_name(user_friendly_name, bus_name, 'Bus', idx, 'controller'),
                id=str(bus_name),
                vm_pu=safe_float(bus['vm_pu']),
                va_degree=safe_float(bus['va_degree']),
                p_mw=safe_float(bus['p_mw']),
                q_mvar=safe_float(bus['q_mvar'])
            ))
        
        lines = []
        for idx, line in net.res_line.iterrows():
            line_name = net.line.loc[idx, 'name']
            # Get user-friendly name from stored mapping
            user_friendly_name = getattr(net, 'user_friendly_names', {}).get(line_name, line_name)
            
            lines.append(ControllerLineOut(
                name=get_display_name(user_friendly_name, line_name, 'Line', idx, 'controller'),
                id=str(line_name),
                p_from_mw=safe_float(line['p_from_mw']),
                q_from_mvar=safe_float(line['q_from_mvar']),
                p_to_mw=safe_float(line['p_to_mw']),
                q_to_mvar=safe_float(line['q_to_mvar']),
                i_from_ka=safe_float(line['i_from_ka']),
                i_to_ka=safe_float(line['i_to_ka']),
                loading_percent=safe_float(line['loading_percent'])
            ))
        
        generators = []
        for idx, gen in net.res_gen.iterrows():
            gen_name = net.gen.loc[idx, 'name']
            # Get user-friendly name from stored mapping
            user_friendly_name = getattr(net, 'user_friendly_names', {}).get(gen_name, gen_name)
            
            generators.append(ControllerGeneratorOut(
                name=get_display_name(user_friendly_name, gen_name, 'Generator', idx, 'controller'),
                id=str(gen_name),
                p_mw=safe_float(gen['p_mw']),
                q_mvar=safe_float(gen['q_mvar']),
                va_degree=safe_float(gen['va_degree']),
                vm_pu=safe_float(gen['vm_pu'])
            ))
        
        loads = []
        for idx, load in net.res_load.iterrows():
            load_name = net.load.loc[idx, 'name']
            # Get user-friendly name from stored mapping
            user_friendly_name = getattr(net, 'user_friendly_names', {}).get(load_name, load_name)
            
            loads.append(ControllerLoadOut(
                name=get_display_name(user_friendly_name, load_name, 'Load', idx, 'controller'),
                id=str(load_name),
                p_mw=safe_float(load['p_mw']),
                q_mvar=safe_float(load['q_mvar'])
            ))
        
        # Controller status for pandapower.control simulation
        controller_status = []
        if controller_params.get('voltage_control', False):
            controller_status.append({
                'controller_id': 0,
                'controller_type': 'VoltageControl',
                'active': True,
                'description': 'Generator voltage control using pandapower.control.run_control',
                'method': 'pandapower.control.run_control',
                'max_iterations': 30
            })
        if controller_params.get('tap_control', False):
            controller_status.append({
                'controller_id': 1,
                'controller_type': 'TapControl',
                'active': True,
                'description': 'Transformer tap control using pandapower.control.run_control',
                'method': 'pandapower.control.run_control',
                'max_iterations': 30
            })
        
        return {
            'controller_converged': net.converged,
            'controller_status': controller_status,
            'busbars': [vars(bus) for bus in busbars],
            'lines': [vars(line) for line in lines],
            'generators': [vars(gen) for gen in generators],
            'loads': [vars(load) for load in loads]
        }
        
    except Exception as e:
        
        # Initialize diagnostic response
        diagnostic_response = {
            "error": True,
            "message": "Controller simulation failed",
            "exception": str(e),
            "diagnostic": {}
        }
        
        # Try to get diagnostic information
        try:
            diag_result_dict = pp.diagnostic(net, report_style='detailed')
            
            # Isolated buses as the load flow reports them, {index, id, name}:
            # bare indices were listed as "5, 6, 7" and could not be located.
            isolated_buses = pp.topology.unsupplied_buses(net)
            if len(isolated_buses) > 0:
                diagnostic_response["diagnostic"]["isolated_buses"] = resolve_element_refs(
                    net, 'bus', sorted(isolated_buses))
            
            # Process diagnostic data to convert element indices to user-friendly names
            processed_diagnostic = process_diagnostic_data(net, diag_result_dict)
            # Merge processed diagnostic with isolated_buses (don't overwrite)
            diagnostic_response["diagnostic"].update(processed_diagnostic)
                    
        except Exception as diag_error:
            pass
        
        # If no specific diagnostic was found, include the original exception
        if not diagnostic_response["diagnostic"]:
            diagnostic_response["diagnostic"]["general_error"] = str(e)
        
        return diagnostic_response


def _ts_normalize_profile(values, time_steps, default=1.0):
    """Repeat or truncate profile list to match time_steps."""
    if not values:
        return [float(default)] * time_steps
    vals = [float(v) for v in values]
    if len(vals) >= time_steps:
        return vals[:time_steps]
    return [vals[i % len(vals)] for i in range(time_steps)]


def _ts_build_load_preset(preset, time_steps):
    import random
    import math
    if preset == 'daily':
        base_profile = [0.3, 0.25, 0.2, 0.15, 0.2, 0.4, 0.7, 0.9, 1.0, 1.1, 1.05, 1.0,
                        0.95, 1.0, 1.05, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35]
        out = []
        for i, val in enumerate(base_profile):
            random.seed(42 + i)
            out.append(max(0.1, min(1.3, val + random.uniform(-0.2, 0.2))))
        return _ts_normalize_profile(out, time_steps)
    if preset == 'industrial':
        base_profile = [0.1, 0.05, 0.05, 0.05, 0.1, 0.2, 0.6, 0.9, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05]
        out = []
        for i, val in enumerate(base_profile):
            random.seed(42 + i)
            out.append(max(0.02, min(1.1, val + random.uniform(-0.1, 0.1))))
        return _ts_normalize_profile(out, time_steps)
    if preset == 'variable':
        out = []
        for hour in range(time_steps):
            random.seed(42 + hour)
            base_load = 0.4 + 0.6 * math.sin(2 * math.pi * hour / 8)
            spike = random.uniform(0.6, 1.6) if random.random() < 0.4 else 1.0
            out.append(max(0.05, min(1.8, base_load * spike)))
        return out
    # constant: unity scale — same as load flow snapshot (no artificial jitter)
    return [1.0] * time_steps


def _ts_build_gen_preset(preset, time_steps):
    import random
    import math
    if preset == 'solar':
        base_profile = [0, 0, 0, 0, 0, 0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0,
                        1.0, 0.95, 0.8, 0.5, 0.2, 0.05, 0, 0, 0, 0, 0, 0]
        out = []
        for i, val in enumerate(base_profile):
            random.seed(42 + i)
            if val > 0.3:
                out.append(max(0, min(1.2, val * random.uniform(0.7, 1.1))))
            else:
                out.append(val)
        return _ts_normalize_profile(out, time_steps)
    if preset == 'wind':
        out = []
        for hour in range(time_steps):
            random.seed(42 + hour)
            base_wind = 0.5 + 0.5 * math.sin(2 * math.pi * hour / 24)
            out.append(max(0.1, min(1.1, base_wind * random.uniform(0.6, 1.4))))
        return out
    if preset == 'variable':
        out = []
        for hour in range(time_steps):
            random.seed(42 + hour)
            base_gen = 0.5 + 0.5 * math.cos(2 * math.pi * hour / 6)
            fluctuation = random.uniform(0.5, 1.5) if random.random() < 0.5 else 1.0
            out.append(max(0.1, min(1.4, base_gen * fluctuation)))
        return out
    # constant: unity scale — same as load flow snapshot (no artificial jitter)
    return [1.0] * time_steps


def _ts_orig_p_for_element(net, elem_name, element_type):
    """Baseline P (MW) for profile mode detection."""
    elem_name = str(elem_name)
    if element_type == 'load' and len(net.load) > 0:
        for idx in net.load.index:
            if str(net.load.loc[idx, 'name']) == elem_name:
                return float(net.load.loc[idx, 'p_mw'])
    if element_type in ('sgen', 'static_generator') and len(net.sgen) > 0:
        for idx in net.sgen.index:
            if str(net.sgen.loc[idx, 'name']) == elem_name:
                return float(net.sgen.loc[idx, 'p_mw'])
    if element_type == 'gen' and len(net.gen) > 0:
        for idx in net.gen.index:
            if str(net.gen.loc[idx, 'name']) == elem_name:
                return float(net.gen.loc[idx, 'p_mw'])
    return None


def _ts_resolve_profile_mode(mode, values, orig_p):
    """
    The profile's mode as the user set it. Scale factors peaking between 20 %
    and 105 % of an element's base P used to be taken for MW: the dialog's own
    Constant preset (1.0) ran a 3 MW load at 1 MW.
    """
    return mode if mode in ('scale', 'absolute') else 'scale'


def _ts_set_pq(table, idx, orig_p, orig_q, value, mode):
    op = float(orig_p.loc[idx])
    oq = float(orig_q.loc[idx]) if orig_q is not None else 0.0
    if mode == 'absolute':
        new_p = float(value)
        if op != 0:
            net_q = oq * (new_p / op)
        elif oq != 0:
            net_q = oq
        else:
            net_q = new_p * 0.66
    else:
        scale = float(value)
        new_p = op * scale
        net_q = oq * scale
    table.loc[idx, 'p_mw'] = new_p
    if 'q_mvar' in table.columns:
        table.loc[idx, 'q_mvar'] = net_q


def _ts_storage_state(net):
    """
    Each battery's drawn power and stored energy, for tracking hour by hour.

    A battery held its drawn power every hour: 0.5 MW for 24 h from the
    radial grid's 2 MWh at 80 %, 12 MWh out of 1.6. One without an energy
    rating or a state of charge cannot be tracked and keeps its power.
    """
    state = {}
    if not hasattr(net, 'storage') or net.storage.empty:
        return state
    for idx in net.storage.index:
        row = net.storage.loc[idx]
        scaling = float(row['scaling']) if 'scaling' in row and pd.notna(row['scaling']) else 1.0
        max_e = float(row['max_e_mwh']) if 'max_e_mwh' in row and pd.notna(row['max_e_mwh']) else float('nan')
        min_e = float(row['min_e_mwh']) if 'min_e_mwh' in row and pd.notna(row['min_e_mwh']) else 0.0
        soc = float(row['soc_percent']) if 'soc_percent' in row and pd.notna(row['soc_percent']) else float('nan')
        trackable = bool(row.get('in_service', True)) and max_e > 0 and not math.isnan(soc)
        state[idx] = {
            'p_mw': float(row['p_mw']) * scaling,   # + charging, as pandapower
            'scaling': scaling or 1.0,
            'max_e': max_e, 'min_e': min_e,
            'energy': max_e * soc / 100.0 if trackable else float('nan'),
            'trackable': trackable,
            'limited_from': None,
        }
    return state


def _ts_dispatch_storage(net, state, t, hours=1.0):
    """Set each battery's power for this hour within the energy it has room for or holds."""
    for idx, st in state.items():
        if not st['trackable']:
            continue
        want = st['p_mw']
        if want > 0:
            p = min(want, max(0.0, st['max_e'] - st['energy']) / hours)
        else:
            p = -min(-want, max(0.0, st['energy'] - st['min_e']) / hours)
        if abs(p) < 1e-12:
            p = 0.0
        if abs(p - want) > 1e-9 and st['limited_from'] is None:
            st['limited_from'], st['limited_p'] = t, p
        net.storage.loc[idx, 'p_mw'] = p / st['scaling']
        st['p_now'] = p


def _ts_advance_storage(state, hours=1.0):
    for st in state.values():
        if st['trackable']:
            st['energy'] += st.get('p_now', 0.0) * hours


def _ts_run_powerflow(net, timeseries_params, time_index, prev_converged):
    """Run PF for one time step; warm-start from previous step when possible (pandapower timeseries style)."""
    algorithm = timeseries_params.get('algorithm', 'nr')
    cva = timeseries_params.get('calculate_voltage_angles', 'auto')
    init_param = timeseries_params.get('init') or timeseries_params.get('initialization') or 'auto'
    pf_kwargs = _electrisim_enforce_q_lims_kw(net)

    if time_index == 0 or not prev_converged or init_param in ('dc', 'flat', 'pf'):
        pf_init = init_param if init_param != 'results' else 'auto'
    else:
        pf_init = 'results'

    try:
        _electrisim_runpp(net, algorithm=algorithm, calculate_voltage_angles=cva, init=pf_init, **pf_kwargs)
    except Exception:
        if pf_init == 'results':
            _electrisim_runpp(net, algorithm=algorithm, calculate_voltage_angles=cva, init='auto', **pf_kwargs)
        else:
            raise
    return bool(net.converged)


def time_series_simulation(net, timeseries_params):
    """
    Sequential AC power flow over time steps with scaled loads and generators.

    Electrisim uses repeated pandapower ``runpp`` calls with profile shapes (similar in spirit to
    `pandapower.timeseries <https://pandapower.readthedocs.io/en/v3.4.0/timeseries.html>`_ but
    without ``run_timeseries`` / OutputWriter). Imports of ``pandapower.timeseries`` are optional
    and do not change this execution path.
    """
    try:
        time_steps = int(timeseries_params.get('time_steps', 24))
        load_profile = timeseries_params.get('load_profile', 'constant')
        generation_profile = timeseries_params.get('generation_profile', 'constant')
        profile_mode = timeseries_params.get('profile_mode', 'preset')
        element_profiles = timeseries_params.get('element_profiles') or {}
        # Hourly unless the study sets its own step: AI training loads swing
        # within seconds.
        step_s = float(safe_float(timeseries_params.get('time_step_s')) or 3600.0)
        if not step_s > 0:
            step_s = 3600.0
        hourly = abs(step_s - 3600.0) < 1e-9
        step_hours = step_s / 3600.0

        import datetime
        # Offsets, not hour=h: datetime(2024, 1, 1, hour=24) raised, so no run
        # could go past 24 steps.
        time_stamps = [datetime.datetime(2024, 1, 1) + datetime.timedelta(seconds=step_s * h)
                       for h in range(time_steps)]

        orig_load_p = net.load['p_mw'].copy() if len(net.load) > 0 else None
        orig_load_q = net.load['q_mvar'].copy() if len(net.load) > 0 else None
        orig_gen_p = net.gen['p_mw'].copy() if len(net.gen) > 0 else None
        orig_gen_q = net.gen['q_mvar'].copy() if len(net.gen) > 0 and 'q_mvar' in net.gen.columns else None
        orig_sgen_p = net.sgen['p_mw'].copy() if len(net.sgen) > 0 else None
        orig_sgen_q = net.sgen['q_mvar'].copy() if len(net.sgen) > 0 else None

        load_profile_values = _ts_build_load_preset(load_profile, time_steps)
        gen_profile_values = _ts_build_gen_preset(generation_profile, time_steps)

        # Resolve per-element profiles for custom mode
        resolved_profiles = {}
        profiles_used = {}
        if profile_mode == 'custom' and element_profiles:
            for elem_name, spec in element_profiles.items():
                if not isinstance(spec, dict):
                    continue
                values = _ts_normalize_profile(spec.get('values', []), time_steps)
                declared_mode = spec.get('mode', 'scale')
                element_type = spec.get('element_type')
                orig_p = _ts_orig_p_for_element(net, elem_name, element_type)
                mode = _ts_resolve_profile_mode(declared_mode, values, orig_p)
                ufn = getattr(net, 'user_friendly_names', {}) or {}
                display_name = (
                    spec.get('display_name')
                    or spec.get('userFriendlyName')
                    or ufn.get(str(elem_name), str(elem_name))
                )
                resolved_profiles[str(elem_name)] = {'values': values, 'mode': mode, 'element_type': element_type}
                profiles_used[str(elem_name)] = {
                    'mode': mode,
                    'declared_mode': declared_mode,
                    'values': values,
                    'element_type': element_type,
                    'display_name': display_name,
                    'id': str(elem_name),
                }

        # Loads following a profile from the diagram's library: its value at
        # each step, or its mean over the step when the step is longer than
        # the profile's own samples.
        import load_profiles_electrisim as _lp
        library, library_notes = _lp.library_from_params(timeseries_params)
        repeat = timeseries_params.get('profile_repeat', True) not in (False, 'false', 'False', 0, '0')
        library_loads = {}
        for load_name, assignment in (timeseries_params.get('load_profile_assignments') or {}).items():
            prof = library.get(assignment['profile_id'])
            if prof is None:
                library_notes.append(f"{assignment['display_name']}: its load profile is not in the library, "
                                     "so it does not follow one.")
                continue
            rel_t = prof['t'] - prof['t'][0]
            prof_dt = float(np.median(np.diff(rel_t)))
            averaged = step_s > 2 * prof_dt
            if averaged:
                values = [_lp.average_profile(rel_t, prof['p'], k * step_s, (k + 1) * step_s, repeat)
                          for k in range(time_steps)]
            else:
                values = [float(v) for v in _lp.sample_profile(rel_t, prof['p'], np.arange(time_steps) * step_s, repeat)]
            library_loads[load_name] = {'values': values, 'q_mode': assignment['q_mode']}
            profiles_used[load_name] = {
                'mode': 'scale',
                'declared_mode': 'library',
                'values': values,
                'element_type': 'load',
                'display_name': assignment['display_name'],
                'id': load_name,
                'library_profile': prof['name'],
                'q_mode': assignment['q_mode'],
                'sampling': 'average' if averaged else 'instant',
            }
            if averaged:
                library_notes.append(f"{assignment['display_name']} follows \"{prof['name']}\" averaged over each "
                                     f"{step_s:g} s step: the profile's samples are {prof_dt:g} s apart.")
            lasts = _lp.period_s(rel_t) if repeat else float(rel_t[-1])
            if time_steps * step_s > lasts + 1e-9:
                library_notes.append(f"{assignment['display_name']}: \"{prof['name']}\" lasts {lasts:g} s, "
                                     + ("so it repeats through the run." if repeat else "and holds its last value after."))

        preset_used = set()

        def _element_profile(name, element_type, t, global_values):
            spec = resolved_profiles.get(str(name))
            if spec and spec.get('element_type') == element_type:
                return spec['values'][t], spec.get('mode', 'scale')
            preset_used.add('load' if element_type == 'load' else 'gen')
            return global_values[t % len(global_values)], 'scale'

        all_results = []
        prev_converged = False
        storage_state = _ts_storage_state(net)
        all_storages = []

        for t in range(time_steps):
            if orig_load_p is not None:
                for idx in net.load.index:
                    elem_name = str(net.load.loc[idx, 'name'])
                    lib = library_loads.get(elem_name)
                    if lib is not None:
                        # 1.0 p.u. is the load's drawn P; Q follows at constant
                        # power factor or stays as drawn.
                        v = lib['values'][t]
                        net.load.loc[idx, 'p_mw'] = float(orig_load_p.loc[idx]) * v
                        net.load.loc[idx, 'q_mvar'] = float(orig_load_q.loc[idx]) * (v if lib['q_mode'] == 'pf' else 1.0)
                        continue
                    val, mode = _element_profile(elem_name, 'load', t, load_profile_values)
                    _ts_set_pq(net.load, idx, orig_load_p, orig_load_q, val, mode)

            if orig_sgen_p is not None:
                for idx in net.sgen.index:
                    elem_name = str(net.sgen.loc[idx, 'name'])
                    val, mode = _element_profile(elem_name, 'sgen', t, gen_profile_values)
                    _ts_set_pq(net.sgen, idx, orig_sgen_p, orig_sgen_q, val, mode)

            if orig_gen_p is not None:
                for idx in net.gen.index:
                    elem_name = str(net.gen.loc[idx, 'name'])
                    val, mode = _element_profile(elem_name, 'gen', t, gen_profile_values)
                    _ts_set_pq(net.gen, idx, orig_gen_p, orig_gen_q, val, mode)

            _ts_dispatch_storage(net, storage_state, t, hours=step_hours)
            prev_converged = _ts_run_powerflow(net, timeseries_params, t, prev_converged)
            _ts_advance_storage(storage_state, hours=step_hours)
            ufn = getattr(net, 'user_friendly_names', {}) or {}
            for idx, st in storage_state.items():
                technical = net.storage.loc[idx, 'name']
                all_storages.append({
                    'name': get_display_name(ufn.get(technical, technical), technical, 'Storage', idx, 'timeseries'),
                    'id': str(technical),
                    'time_step': t,
                    'p_mw': safe_float(net.res_storage.loc[idx, 'p_mw']),
                    'q_mvar': safe_float(net.res_storage.loc[idx, 'q_mvar']),
                    # State of charge at the end of the step.
                    'soc_percent': safe_float(100.0 * st['energy'] / st['max_e']) if st['trackable'] else None,
                    'energy_mwh': safe_float(st['energy']) if st['trackable'] else None,
                })

            all_results.append({
                'time_step': t,
                'converged': net.converged,
                'bus_results': _electrisim_res_without_aux(net, 'bus'),
                'line_results': net.res_line.copy(),
                'gen_results': net.res_gen.copy() if len(net.gen) > 0 else None,
                'sgen_results': _electrisim_res_without_aux(net, 'sgen') if len(net.sgen) > 0 else None,
                'load_results': _electrisim_res_without_aux(net, 'load') if len(net.load) > 0 else None,
                'trafo_results': net.res_trafo.copy() if len(net.trafo) > 0 else None,
                'trafo3w_results': net.res_trafo3w.copy() if len(net.trafo3w) > 0 else None,
                'ext_grid_results': _electrisim_res_without_aux(net, 'ext_grid') if len(net.ext_grid) > 0 else None,
            })

        # Prepare results
        class TimeSeriesBusOut(object):
            def __init__(self, name: str, id: str, time_step: int, vm_pu: float, va_degree: float, p_mw: float, q_mvar: float):
                self.name = name
                self.id = id
                self.time_step = time_step
                self.vm_pu = vm_pu
                self.va_degree = va_degree
                self.p_mw = p_mw
                self.q_mvar = q_mvar
        
        class TimeSeriesLineOut(object):
            def __init__(self, name: str, id: str, time_step: int, loading_percent: float, p_from_mw: float, p_to_mw: float):
                self.name = name
                self.id = id
                self.time_step = time_step
                self.loading_percent = loading_percent
                self.p_from_mw = p_from_mw
                self.p_to_mw = p_to_mw

        class TimeSeriesLoadOut(object):
            def __init__(self, name: str, id: str, time_step: int, p_mw: float, q_mvar: float):
                self.name = name
                self.id = id
                self.time_step = time_step
                self.p_mw = p_mw
                self.q_mvar = q_mvar

        class TimeSeriesSgenOut(object):
            def __init__(self, name: str, id: str, time_step: int, p_mw: float, q_mvar: float):
                self.name = name
                self.id = id
                self.time_step = time_step
                self.p_mw = p_mw
                self.q_mvar = q_mvar
        
        # Collect results for each time step (from sequential runpp snapshots)
        all_busbars = []
        all_lines = []
        all_loads = []
        all_sgens = []
        # Generators were run but not returned, transformers and the external
        # grid not even kept: the most loaded element (a 79 % transformer on
        # the transmission grid) did not show.
        all_gens = []
        all_transformers = []
        all_ext_grids = []

        def _element_rows(table, res, element_type, t, fields):
            ufn = getattr(net, 'user_friendly_names', {}) or {}
            rows = []
            for idx, row in res.iterrows():
                technical = net[table].loc[idx, 'name']
                item = {
                    'name': get_display_name(ufn.get(technical, technical), technical, element_type, idx, 'timeseries'),
                    'id': str(technical),
                    'time_step': t,
                }
                for key, column in fields:
                    item[key] = safe_float(row[column])
                rows.append(item)
            return rows

        for result in all_results:
            t = result['time_step']
            bus_results = result['bus_results']
            line_results = result['line_results']
            load_results = result.get('load_results')
            sgen_results = result.get('sgen_results')

            for idx, bus in bus_results.iterrows():
                bus_name = net.bus.loc[idx, 'name']
                user_friendly_name = getattr(net, 'user_friendly_names', {}).get(bus_name, bus_name)
                all_busbars.append(TimeSeriesBusOut(
                    name=get_display_name(user_friendly_name, bus_name, 'Bus', idx, 'timeseries'),
                    id=str(bus_name),
                    time_step=t,
                    vm_pu=safe_float(bus['vm_pu']),
                    va_degree=safe_float(bus['va_degree']),
                    p_mw=safe_float(bus['p_mw']),
                    q_mvar=safe_float(bus['q_mvar'])
                ))

            for idx, line in line_results.iterrows():
                line_name = net.line.loc[idx, 'name']
                user_friendly_name = getattr(net, 'user_friendly_names', {}).get(line_name, line_name)
                all_lines.append(TimeSeriesLineOut(
                    name=get_display_name(user_friendly_name, line_name, 'Line', idx, 'timeseries'),
                    id=str(line_name),
                    time_step=t,
                    loading_percent=safe_float(line['loading_percent']),
                    p_from_mw=safe_float(line['p_from_mw']),
                    p_to_mw=safe_float(line['p_to_mw'])
                ))

            if load_results is not None:
                for idx, load in load_results.iterrows():
                    load_name = net.load.loc[idx, 'name']
                    user_friendly_name = getattr(net, 'user_friendly_names', {}).get(load_name, load_name)
                    all_loads.append(TimeSeriesLoadOut(
                        name=get_display_name(user_friendly_name, load_name, 'Load', idx, 'timeseries'),
                        id=str(load_name),
                        time_step=t,
                        p_mw=safe_float(load['p_mw']),
                        q_mvar=safe_float(load['q_mvar'])
                    ))

            if result.get('gen_results') is not None:
                all_gens.extend(_element_rows('gen', result['gen_results'], 'Generator', t, (
                    ('p_mw', 'p_mw'), ('q_mvar', 'q_mvar'), ('vm_pu', 'vm_pu'))))
            if result.get('trafo_results') is not None:
                all_transformers.extend(_element_rows('trafo', result['trafo_results'], 'Transformer', t, (
                    ('loading_percent', 'loading_percent'), ('p_hv_mw', 'p_hv_mw'), ('q_hv_mvar', 'q_hv_mvar'))))
            if result.get('trafo3w_results') is not None:
                all_transformers.extend(_element_rows('trafo3w', result['trafo3w_results'], 'Transformer3W', t, (
                    ('loading_percent', 'loading_percent'), ('p_hv_mw', 'p_hv_mw'), ('q_hv_mvar', 'q_hv_mvar'))))
            if result.get('ext_grid_results') is not None:
                all_ext_grids.extend(_element_rows('ext_grid', result['ext_grid_results'], 'External Grid', t, (
                    ('p_mw', 'p_mw'), ('q_mvar', 'q_mvar'))))

            if sgen_results is not None:
                for idx, sgen in sgen_results.iterrows():
                    sgen_name = net.sgen.loc[idx, 'name']
                    user_friendly_name = getattr(net, 'user_friendly_names', {}).get(sgen_name, sgen_name)
                    all_sgens.append(TimeSeriesSgenOut(
                        name=get_display_name(user_friendly_name, sgen_name, 'Static Generator', idx, 'timeseries'),
                        id=str(sgen_name),
                        time_step=t,
                        p_mw=safe_float(sgen['p_mw']),
                        q_mvar=safe_float(sgen['q_mvar'])
                    ))

        # Summary statistics with display names (user-friendly + technical ID)
        vm_stats = {}
        for idx, bus in net.bus.iterrows():
            bus_name = bus['name']
            user_friendly_name = getattr(net, 'user_friendly_names', {}).get(bus_name, bus_name)
            display_name = get_display_name(user_friendly_name, bus_name, 'Bus', idx, 'timeseries')
            vm_values = [all_busbars[i].vm_pu for i in range(len(all_busbars)) 
                        if all_busbars[i].name == display_name]
            if vm_values:  # Check if list is not empty
                vm_stats[display_name] = {
                    'min_vm_pu': min(vm_values),
                    'max_vm_pu': max(vm_values),
                    'avg_vm_pu': sum(vm_values) / len(vm_values)
                }
            else:
                vm_stats[display_name] = {
                    'min_vm_pu': 0.0,
                    'max_vm_pu': 0.0,
                    'avg_vm_pu': 0.0
                }
        
        loading_stats = {}
        for idx, line in net.line.iterrows():
            line_name = line['name']
            user_friendly_name = getattr(net, 'user_friendly_names', {}).get(line_name, line_name)
            display_name = get_display_name(user_friendly_name, line_name, 'Line', idx, 'timeseries')
            loading_values = [all_lines[i].loading_percent for i in range(len(all_lines)) 
                             if all_lines[i].name == display_name]
            if loading_values:  # Check if list is not empty
                loading_stats[display_name] = {
                    'min_loading_percent': min(loading_values),
                    'max_loading_percent': max(loading_values),
                    'avg_loading_percent': sum(loading_values) / len(loading_values)
                }
            else:
                loading_stats[display_name] = {
                    'min_loading_percent': 0.0,
                    'max_loading_percent': 0.0,
                    'avg_loading_percent': 0.0
                }
        
        transformer_stats = {}
        for row in all_transformers:
            transformer_stats.setdefault(row['name'], []).append(row['loading_percent'])
        transformer_stats = {
            name: {'min_loading_percent': min(values), 'max_loading_percent': max(values),
                   'avg_loading_percent': sum(values) / len(values)}
            for name, values in transformer_stats.items()
        }

        timeseries_converged = all(result['converged'] for result in all_results)

        notes = list(library_notes)
        for idx, st in storage_state.items():
            technical = net.storage.loc[idx, 'name']
            ufn = getattr(net, 'user_friendly_names', {}) or {}
            label = ufn.get(technical, technical)
            if not st['trackable']:
                notes.append(f"{label}: no energy rating or state of charge, so it held "
                             f"{abs(st['p_mw']):.3g} MW every {'hour' if hourly else 'step'}.")
            elif st['limited_from'] is not None:
                when = (f"in hour {st['limited_from']} ({abs(st['limited_p']):.3g} MW that hour)" if hourly else
                        f"at step {st['limited_from']}, t = {st['limited_from'] * step_s:g} s "
                        f"({abs(st['limited_p']):.3g} MW that step)")
                notes.append(f"{label}: {'charging' if st['p_mw'] > 0 else 'discharging'} "
                             f"{abs(st['p_mw']):.3g} MW, it runs {'full' if st['p_mw'] > 0 else 'empty'} "
                             f"{when} and is idle after.")

        # A preset only reads as the run's when some element followed it:
        # with a profile of its own for every element, "constant" described
        # nothing that ran.
        return {
            'timeseries_converged': timeseries_converged,
            'time_steps': time_steps,
            'time_step_s': step_s,
            'profile_mode': profile_mode,
            'load_profile': load_profile if 'load' in preset_used else None,
            'generation_profile': generation_profile if 'gen' in preset_used else None,
            'load_profile_values': load_profile_values if 'load' in preset_used else [],
            'generation_profile_values': gen_profile_values if 'gen' in preset_used else [],
            'profiles_used': profiles_used,
            'busbars': [vars(bus) for bus in all_busbars],
            'lines': [vars(line) for line in all_lines],
            'loads': [vars(ld) for ld in all_loads],
            'sgens': [vars(sg) for sg in all_sgens],
            'gens': all_gens,
            'transformers': all_transformers,
            'externalgrids': all_ext_grids,
            'storages': all_storages,
            'notes': notes,
            'voltage_statistics': vm_stats,
            'loading_statistics': loading_stats,
            'transformer_loading_statistics': transformer_stats,
            'time_stamps': [str(ts) for ts in time_stamps]
        }
        
    except Exception as e:
        
        # Initialize diagnostic response
        diagnostic_response = {
            "error": True,
            "message": "Time series simulation failed",
            "exception": str(e),
            "diagnostic": {}
        }
        
        # Try to get diagnostic information
        try:
            diag_result_dict = pp.diagnostic(net, report_style='detailed')
            
            # Isolated buses as the load flow reports them, {index, id, name}:
            # bare indices were listed as "5, 6, 7" and could not be located.
            isolated_buses = pp.topology.unsupplied_buses(net)
            if len(isolated_buses) > 0:
                diagnostic_response["diagnostic"]["isolated_buses"] = resolve_element_refs(
                    net, 'bus', sorted(isolated_buses))
            
            # Process diagnostic data to convert element indices to user-friendly names
            processed_diagnostic = process_diagnostic_data(net, diag_result_dict)
            # Merge processed diagnostic with isolated_buses (don't overwrite)
            diagnostic_response["diagnostic"].update(processed_diagnostic)
                    
        except Exception as diag_error:
            pass
        
        # If no specific diagnostic was found, include the original exception
        if not diagnostic_response["diagnostic"]:
            diagnostic_response["diagnostic"]["general_error"] = str(e)
        
        return diagnostic_response


def _bess_sizing_failure(message):
    return json.dumps({
        'error': message,
        'bess_p_mw': None, 'bess_q_mvar': None, 'bess_s_mva': None,
        'achieved_p_mw': None, 'achieved_q_mvar': None,
        'error_p_mw': None, 'error_q_mvar': None,
        'converged': False, 'iterations': 0,
    })


def _bess_sizing_violations(net, vmin_pu, vmax_pu, max_loading_percent=100.0):
    """Branches above their rating and buses outside the voltage band, by name."""
    out = []
    for table, results, kind in (('line', 'res_line', 'Line'), ('trafo', 'res_trafo', 'Transformer'),
                                 ('trafo3w', 'res_trafo3w', 'Transformer')):
        if table not in net or net[table].empty or results not in net or 'loading_percent' not in net[results]:
            continue
        for idx, loading in net[results]['loading_percent'].items():
            if pd.notna(loading) and loading > max_loading_percent:
                out.append({'kind': kind, 'name': _contingency_friendly_name(net, net[table].at[idx, 'name']),
                            'value': round(float(loading), 1), 'unit': '%',
                            'limit': max_loading_percent})
    for idx, vm in net.res_bus['vm_pu'].items():
        if pd.notna(vm) and not (vmin_pu <= vm <= vmax_pu):
            out.append({'kind': 'Bus', 'name': _contingency_friendly_name(net, net.bus.at[idx, 'name']),
                        'value': round(float(vm), 4), 'unit': 'pu',
                        'limit': vmin_pu if vm < vmin_pu else vmax_pu})
    return out


def bess_sizing(net, bess_params):
    """
    The battery P and Q that put the target exchange at the POC, found by a
    Newton solve on the load flow, and what that operating point does to the
    battery's own rating and to the network.

    The drawn battery's rating does not cap the answer - finding the rating is
    the point. A proportional controller limited P and Q each to the drawn
    sn_mva (which a battery without one took from its MWh), and reported the
    clipped values as the required size: 2.8 MVA where the radial reference
    grid needs 21.1 MVA, and a feeder cable at 126 % unmentioned.

    bess_params: storageId, pocBusbarId (diagram cell ids), targetP / targetQ
    (MW / Mvar exported at the POC), tolerance, maxIterations, algorithm,
    vmin_pu / vmax_pu (default 0.9 / 1.1).

    Returns a JSON string.
    """
    try:
        storage_id = bess_params.get('storageId')
        poc_busbar_id = bess_params.get('pocBusbarId')
        target = np.array([float(bess_params.get('targetP', 0.0)), float(bess_params.get('targetQ', 0.0))])
        tolerance = float(bess_params.get('tolerance', 0.001))
        max_iterations = int(bess_params.get('maxIterations', 50))
        algorithm = bess_params.get('algorithm', 'nr')
        vmin_pu = float(bess_params.get('vmin_pu', 0.9))
        vmax_pu = float(bess_params.get('vmax_pu', 1.1))

        if net.storage.empty:
            return _bess_sizing_failure('No storage element found in network')
        # The dialog sends the diagram cell id; name and bus index are older forms.
        storage_idx = next((idx for idx in net.storage.index
                            if str(storage_id) in (str(net.storage.at[idx, 'id']) if 'id' in net.storage else '',
                                                   str(net.storage.at[idx, 'name']),
                                                   str(net.storage.at[idx, 'bus']))),
                           net.storage.index[0])
        poc_bus_idx = next((idx for idx in net.bus.index
                            if str(poc_busbar_id) in (str(net.bus.at[idx, 'id']) if 'id' in net.bus else '',
                                                      str(net.bus.at[idx, 'name']), str(idx))), None)
        if poc_bus_idx is None:
            return _bess_sizing_failure('No POC bus found in network')

        # P and Q are measured at the External Grid on the POC bus (it was
        # the first External Grid in the network, wherever it was).
        ext_at_poc = [idx for idx in net.ext_grid.index if net.ext_grid.at[idx, 'bus'] == poc_bus_idx]
        if not ext_at_poc:
            grid_buses = [net.bus.at[int(b), 'name'] for b in net.ext_grid['bus']]
            labels = getattr(net, 'user_friendly_names', None) or {}
            return _bess_sizing_failure(
                'BESS sizing sets P and Q where the network meets the External Grid. '
                "Choose the External Grid's bus as the POC ("
                + ', '.join(str(labels.get(n, n)) for n in grid_buses) + ').')
        ext_idx = ext_at_poc[0]

        net_s = deepcopy(net)
        solved = [False]

        def poc_exchange(x):
            """Export-positive P, Q at the POC with the battery at x (pandapower
            storage sign: + charge)."""
            net_s.storage.at[storage_idx, 'p_mw'] = float(x[0])
            net_s.storage.at[storage_idx, 'q_mvar'] = float(x[1])
            pp.runpp(net_s, algorithm=algorithm, calculate_voltage_angles=True,
                     init='results' if solved[0] else 'auto')
            solved[0] = True
            return np.array([-float(net_s.res_ext_grid.at[ext_idx, 'p_mw']),
                             -float(net_s.res_ext_grid.at[ext_idx, 'q_mvar'])])

        warnings_out = []
        # Start from the battery as drawn. Starting from minus the whole POC
        # target put 4.9 MW of charge on a 0.4 kV battery behind a 0.63 MVA
        # transformer when the target was the grid's own 5 MW import less
        # 0.3 MW; that load flow failed before any point had solved.
        x = np.array([float(net_s.storage.at[storage_idx, 'p_mw']),
                      float(net_s.storage.at[storage_idx, 'q_mvar'])])
        try:
            y = poc_exchange(x)
        except pp.LoadflowNotConverged:
            return _bess_sizing_failure('The load flow does not converge with the battery as drawn.')
        converged, iterations = False, 0
        while iterations < max_iterations:
            err = y - target
            if np.abs(err).max() < tolerance:
                converged = True
                break
            iterations += 1
            try:
                h = 1e-3
                jac = np.column_stack([(poc_exchange(x + h * e) - y) / h for e in np.eye(2)])
                step = np.linalg.solve(jac, err)
            except (pp.LoadflowNotConverged, np.linalg.LinAlgError) as e:
                poc_exchange(x)
                warnings_out.append(
                    f'The solve stopped at the last operating point that converged ({e.__class__.__name__}).')
                break
            # Halve a step whose load flow fails; the network may not carry it.
            for _ in range(8):
                try:
                    x_try = x - step
                    y_try = poc_exchange(x_try)
                    x, y = x_try, y_try
                    break
                except pp.LoadflowNotConverged:
                    step = step / 2
            else:
                poc_exchange(x)  # leave the results at the last operating point that solved
                warnings_out.append(
                    'The load flow did not converge on the way to this target: the network '
                    'cannot carry the power it needs.')
                break
        if not converged and not warnings_out:
            warnings_out.append(f'The target was not reached within {max_iterations} iterations.')

        bess_s_mva = float(np.hypot(*x))
        rating = float(net.storage.at[storage_idx, 'sn_mva']) if 'sn_mva' in net.storage else 0.0
        rating = rating if pd.notna(rating) and rating > 0 else None
        within_rating = None if rating is None else bool(bess_s_mva <= rating + 1e-6)
        storage_name = _contingency_friendly_name(net, net.storage.at[storage_idx, 'name'])
        if rating is None:
            warnings_out.append(f"'{storage_name}' has no MVA rating; it needs {bess_s_mva:.3f} MVA.")
        elif not within_rating:
            warnings_out.append(
                f"'{storage_name}' is rated {rating:.3f} MVA; this target needs {bess_s_mva:.3f} MVA.")
        violations = _bess_sizing_violations(net_s, vmin_pu, vmax_pu)
        for v in violations:
            warnings_out.append(
                f"{v['kind']} {v['name']}: {v['value']} {v['unit']} (limit {v['limit']} {v['unit']}) "
                'at this operating point.')

        return json.dumps({
            'bess_p_mw': float(x[0]),
            'bess_q_mvar': float(x[1]),
            'bess_s_mva': bess_s_mva,
            'achieved_p_mw': float(y[0]),
            'achieved_q_mvar': float(y[1]),
            'error_p_mw': float(y[0] - target[0]),
            'error_q_mvar': float(y[1] - target[1]),
            'converged': bool(converged),
            'iterations': int(iterations),
            'storage_name': storage_name,
            'storage_rating_mva': rating,
            'within_rating': within_rating,
            'violations': violations,
            'warnings': warnings_out,
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return _bess_sizing_failure(f'BESS sizing calculation failed: {e}')


def _economic_get_month_index(time_steps):
    """Return month index (0=Jan .. 11=Dec) for each hour. Assumes hour 0 = Jan 1 00:00."""
    days_per_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    cumulative_days = np.cumsum(np.concatenate([[0], days_per_month]))
    t_arr = np.arange(time_steps)
    day_of_year = t_arr // 24
    day_of_year = np.minimum(day_of_year, 365)
    month_idx = np.searchsorted(cumulative_days[1:], day_of_year, side='right')
    return np.minimum(month_idx, 11)


def _economic_get_load_profile(load_profile, time_steps):
    """Return load scaling factors for each hour (0..time_steps-1). Yearly profiles with daily + monthly variation."""
    base_24 = {
        'daily': np.array([0.3, 0.25, 0.2, 0.15, 0.2, 0.4, 0.7, 0.9, 1.0, 1.1, 1.05, 1.0,
                          0.95, 1.0, 1.05, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35]),
        'industrial': np.array([0.1, 0.05, 0.05, 0.05, 0.1, 0.2, 0.6, 0.9, 1.0, 1.0, 1.0, 1.0,
                               1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05]),
    }
    if load_profile == 'constant':
        # As the generation profile has it: the diagram's load every hour.
        # The random variation below used to apply here too.
        return np.ones(time_steps, dtype=float)
    t_arr = np.arange(time_steps)
    hour_arr = t_arr % 24
    month_idx = _economic_get_month_index(time_steps)
    if load_profile == 'daily':
        base = base_24['daily']
        month_factor = np.array([1.15, 1.1, 1.0, 0.85, 0.8, 0.9, 1.0, 0.95, 0.9, 0.95, 1.05, 1.1])
    elif load_profile == 'industrial':
        base = base_24['industrial']
        month_factor = np.array([1.02, 0.98, 1.0, 1.0, 1.0, 0.98, 0.95, 0.96, 1.0, 1.02, 1.0, 0.99])
    else:
        base = np.ones(24)
        month_factor = np.ones(12)
    daily_val = base[hour_arr]
    monthly_val = month_factor[month_idx]
    val = daily_val * monthly_val
    rng = np.random.RandomState(42)
    variation = rng.uniform(-0.1, 0.1, size=time_steps)
    out = np.clip(val + variation, 0.1, 1.3)
    return out


def _economic_get_generation_profile(generation_profile, time_steps):
    """Return generation scaling factors. Yearly profiles: daily pattern × monthly seasonal factor."""
    t_arr = np.arange(time_steps)
    hour_arr = t_arr % 24
    month_idx = _economic_get_month_index(time_steps)
    rng = np.random.RandomState(42)

    if generation_profile == 'solar':
        base_24 = np.array([0, 0, 0, 0, 0, 0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0,
                            1.0, 0.95, 0.8, 0.5, 0.2, 0.05, 0, 0, 0, 0, 0, 0])
        daily_val = base_24[hour_arr]
        month_factor = np.array([0.25, 0.35, 0.55, 0.75, 0.9, 1.0, 0.95, 0.85, 0.65, 0.45, 0.3, 0.22])
        monthly_val = month_factor[month_idx]
        val = daily_val * monthly_val
        mult = rng.uniform(0.85, 1.0, size=time_steps)
        val = np.where(val > 0.2, val * mult, val)
        out = np.clip(val, 0, 1.0)
    elif generation_profile == 'onshore_wind':
        base_24 = 0.5 + 0.4 * np.sin(2 * np.pi * (hour_arr - 6) / 24)
        month_factor = np.array([1.15, 1.1, 1.0, 0.85, 0.7, 0.6, 0.55, 0.6, 0.75, 0.95, 1.05, 1.15])
        val = base_24 * month_factor[month_idx]
        variation = rng.uniform(0.7, 1.25, size=time_steps)
        out = np.clip(val * variation, 0.05, 1.15)
    elif generation_profile == 'offshore_wind':
        base_24 = 0.7 + 0.2 * np.sin(2 * np.pi * (hour_arr - 4) / 24)
        month_factor = np.array([1.0, 0.96, 0.88, 0.75, 0.63, 0.54, 0.5, 0.54, 0.67, 0.83, 0.92, 1.0])
        val = base_24 * month_factor[month_idx]
        variation = rng.uniform(0.9, 1.0, size=time_steps)
        out = np.clip(val * variation, 0.25, 1.0)
    elif generation_profile == 'wind':
        base_24 = 0.55 + 0.4 * np.sin(2 * np.pi * (hour_arr - 5) / 24)
        month_factor = np.array([1.12, 1.08, 1.0, 0.88, 0.72, 0.62, 0.58, 0.62, 0.78, 0.98, 1.05, 1.12])
        val = base_24 * month_factor[month_idx]
        variation = rng.uniform(0.7, 1.3, size=time_steps)
        out = np.clip(val * variation, 0.08, 1.12)
    elif generation_profile == 'constant':
        out = np.ones(time_steps, dtype=float)
    else:
        variation = rng.uniform(-0.15, 0.15, size=time_steps)
        out = np.clip(1.0 + variation, 0.8, 1.0)
    return out


def _economic_get_loss_mw(net):
    """Return total active power loss (MW) from pandapower results."""
    loss = 0.0
    if hasattr(net, 'res_line') and not net.res_line.empty:
        loss += float(net.res_line['pl_mw'].sum())
    if hasattr(net, 'res_trafo') and not net.res_trafo.empty:
        loss += float(net.res_trafo['pl_mw'].sum())
    if hasattr(net, 'res_trafo3w') and not net.res_trafo3w.empty:
        loss += float(net.res_trafo3w['pl_mw'].sum())
    if hasattr(net, 'res_impedance') and not net.res_impedance.empty:
        loss += float(net.res_impedance['pl_mw'].sum())
    if hasattr(net, 'res_dcline') and not net.res_dcline.empty:
        loss += float(net.res_dcline['pl_mw'].sum())
    return loss


def economic_analysis(net, in_data, params):
    """
    Calculate CAPEX (total capital expenditure), power losses, and electrical energy losses.
    
    Parameters:
    -----------
    net : pandapower network
        The network object (after create_busbars and create_other_elements)
    in_data : dict
        Full request data with all elements (keys 0, 1, 2, ...)
    params : dict
        Economic analysis parameters: frequency, currency, algorithm, init,
        include_energy_loss, use_generation_profile, time_steps, load_profile,
        generation_profile, energy_price_per_mwh.
        
    Returns:
    --------
    dict : Results with total_capex, total_power_losses_mw, total_energy_losses_mwh (optional lifetime total),
           total_energy_losses_period_mwh (optional), energy_loss_cost (optional), capex_breakdown, power_losses_breakdown, currency
    """
    try:
        currency = params.get('currency', 'EUR').upper()
        
        algorithm = params.get('algorithm', 'nr')
        calculate_voltage_angles = params.get('calculate_voltage_angles', 'auto')
        init = params.get('init', 'dc')
        use_generation_profile = params.get('use_generation_profile', False)
        time_steps = max(1, min(8760, int(params.get('time_steps', 8760))))
        calculation_mode = params.get('calculation_mode', 'full')
        load_profile = params.get('load_profile', 'constant')
        generation_profile = params.get('generation_profile', 'constant')
        energy_price = params.get('energy_price_per_mwh')
        energy_price_currency = params.get('energy_price_currency', 'EUR')
        if energy_price is not None:
            try:
                energy_price = float(energy_price)
            except (TypeError, ValueError):
                energy_price = None
        
        try:
            pp.runpp(net, algorithm=algorithm, calculate_voltage_angles=calculate_voltage_angles, init=init,
                     **_electrisim_enforce_q_lims_kw(net))
        except Exception as pf_err:
            return {
                'error': f'Power flow failed: {str(pf_err)}',
                'total_capex': 0,
                'total_power_losses_mw': 0,
                'total_energy_losses_mwh': None,
                'energy_loss_cost': None,
                'capex_breakdown': [],
                'power_losses_breakdown': [],
                'currency': currency
            }
        
        # Collect CAPEX from in_data - iterate over all elements
        capex_breakdown = []
        total_capex = 0.0
        for x in in_data:
            elem = in_data[x]
            if not isinstance(elem, dict):
                continue
            typ = elem.get('typ', '')
            if 'Parameters' in typ or typ == 'simulationParameters':
                continue
            
            cost_per_unit = 0.0
            if elem.get('cost_per_unit_by_currency'):
                try:
                    import json
                    by_curr = elem.get('cost_per_unit_by_currency')
                    if isinstance(by_curr, str):
                        by_curr = json.loads(by_curr)
                    if isinstance(by_curr, dict) and currency in by_curr:
                        cost_per_unit = float(by_curr.get(currency, 0) or 0)
                    elif isinstance(by_curr, (int, float)) and not isinstance(by_curr, bool):
                        cost_per_unit = float(by_curr)
                except (ValueError, TypeError, Exception):
                    pass
            
            if cost_per_unit > 0:
                # Multiply by quantity for elements where cost is per unit (e.g. per km for lines)
                quantity = 1.0
                if 'Line' in typ or 'line' in typ.lower():
                    length_km = float(elem.get('length_km', 1) or 1)
                    parallel = float(elem.get('parallel', 1) or 1)
                    quantity = length_km * parallel
                elif 'Transformer' in typ or 'trafo' in typ.lower():
                    parallel = float(elem.get('parallel', 1) or 1)
                    quantity = parallel
                cost_val = cost_per_unit * quantity
                name = elem.get('userFriendlyName', elem.get('name', elem.get('id', str(x))))
                capex_breakdown.append({
                    'element_type': typ,
                    'name': name,
                    'id': elem.get('id', str(x)),
                    'cost': round(cost_val, 2),
                    'currency': currency
                })
                total_capex += cost_val
        
        # Helper to resolve user-friendly name (same as CAPEX breakdown)
        def _friendly_name(net, internal_name):
            return getattr(net, 'user_friendly_names', {}).get(internal_name, internal_name)

        # Collect power losses from pandapower results
        power_losses_breakdown = []
        total_power_losses_mw = 0.0

        if hasattr(net, 'res_line') and not net.res_line.empty:
            for index, row in net.res_line.iterrows():
                pl_mw = float(row.get('pl_mw', 0) or 0)
                if pl_mw > 0:
                    internal = net.line.at[index, 'name'] if 'name' in net.line.columns else str(index)
                    name = _friendly_name(net, internal)
                    power_losses_breakdown.append({'element_type': 'Line', 'name': name, 'id': str(index), 'pl_mw': pl_mw})
                    total_power_losses_mw += pl_mw

        if hasattr(net, 'res_trafo') and not net.res_trafo.empty:
            for index, row in net.res_trafo.iterrows():
                pl_mw = float(row.get('pl_mw', 0) or 0)
                if pl_mw > 0:
                    internal = net.trafo.at[index, 'name'] if 'name' in net.trafo.columns else str(index)
                    name = _friendly_name(net, internal)
                    power_losses_breakdown.append({'element_type': 'Transformer', 'name': name, 'id': str(index), 'pl_mw': pl_mw})
                    total_power_losses_mw += pl_mw

        if hasattr(net, 'res_trafo3w') and not net.res_trafo3w.empty:
            for index, row in net.res_trafo3w.iterrows():
                pl_mw = float(row.get('pl_mw', 0) or 0)
                if pl_mw > 0:
                    internal = net.trafo3w.at[index, 'name'] if 'name' in net.trafo3w.columns else str(index)
                    name = _friendly_name(net, internal)
                    power_losses_breakdown.append({'element_type': 'Three Winding Transformer', 'name': name, 'id': str(index), 'pl_mw': pl_mw})
                    total_power_losses_mw += pl_mw

        if hasattr(net, 'res_impedance') and not net.res_impedance.empty:
            for index, row in net.res_impedance.iterrows():
                pl_mw = float(row.get('pl_mw', 0) or 0)
                if pl_mw > 0:
                    internal = net.impedance.at[index, 'name'] if 'name' in net.impedance.columns else str(index)
                    name = _friendly_name(net, internal)
                    power_losses_breakdown.append({'element_type': 'Impedance', 'name': name, 'id': str(index), 'pl_mw': pl_mw})
                    total_power_losses_mw += pl_mw

        if hasattr(net, 'res_dcline') and not net.res_dcline.empty:
            for index, row in net.res_dcline.iterrows():
                pl_mw = float(row.get('pl_mw', 0) or 0)
                if pl_mw > 0:
                    internal = net.dcline.at[index, 'name'] if 'name' in net.dcline.columns else str(index)
                    name = _friendly_name(net, internal)
                    power_losses_breakdown.append({'element_type': 'DC Line', 'name': name, 'id': str(index), 'pl_mw': pl_mw})
                    total_power_losses_mw += pl_mw
        
        if hasattr(net, 'res_line_dc') and not net.res_line_dc.empty:
            for index, row in net.res_line_dc.iterrows():
                pl_mw = float(row.get('pl_mw', 0) or 0)
                if _electrisim_is_hidden(net.line_dc, index):
                    continue   # a battery's own resistance
                if pl_mw > 0:
                    try:
                        internal = net.line_dc.at[index, 'name'] if hasattr(net, 'line_dc') and index in net.line_dc.index and 'name' in net.line_dc.columns else str(index)
                    except Exception:
                        internal = str(index)
                    name = _friendly_name(net, internal)
                    power_losses_breakdown.append({'element_type': 'DC Line', 'name': name, 'id': str(index), 'pl_mw': pl_mw})
                    total_power_losses_mw += pl_mw
        
        total_energy_losses_mwh = None
        total_energy_losses_period_mwh = None
        total_energy_losses_annual_mwh = None
        energy_warnings = []
        energy_loss_cost = None
        energy_loss_period_hours = None
        
        if use_generation_profile:
            load_scale = _economic_get_load_profile(load_profile, time_steps)
            gen_scale = _economic_get_generation_profile(generation_profile, time_steps)
            orig_load_p = net.load['p_mw'].copy() if len(net.load) > 0 else None
            orig_load_q = net.load['q_mvar'].copy() if len(net.load) > 0 else None
            # The generation profile (solar, wind) is for static generators and
            # wind turbines. Synchronous generators keep their drawn P: a 4 MW
            # CHP plant followed the onshore-wind profile, averaging 1.7 MW, and
            # the year's losses came out 12 % low.
            orig_sgen_p = net.sgen['p_mw'].copy() if len(net.sgen) > 0 else None
            orig_sgen_q = net.sgen['q_mvar'].copy() if len(net.sgen) > 0 and 'q_mvar' in net.sgen.columns else None

            # Lookup table: power flows on a grid of (load, generation) scales,
            # interpolated per hour. The grid spans what the profiles reach:
            # it stopped at load 1.2 and generation 1.0 while the residential
            # profile reaches 1.3 and the wind ones 1.15, and every hour
            # beyond read 0 MW - the heaviest hours, 7.6 % of a year's losses.
            def _axis(values, n):
                lo, hi = float(np.min(values)), float(np.max(values))
                return np.array([lo]) if hi - lo < 1e-9 else np.linspace(lo, hi, n)

            load_vals = _axis(load_scale, 11)
            gen_vals = _axis(gen_scale, 21 if len(load_vals) == 1 else 11)
            loss_grid = np.full((len(load_vals), len(gen_vals)), np.nan)
            for i, ls in enumerate(load_vals):
                for j, gs in enumerate(gen_vals):
                    if orig_load_p is not None:
                        net.load['p_mw'] = orig_load_p * ls
                        net.load['q_mvar'] = orig_load_q * ls
                    if orig_sgen_p is not None:
                        net.sgen['p_mw'] = orig_sgen_p * gs
                        if orig_sgen_q is not None:
                            net.sgen['q_mvar'] = orig_sgen_q * gs
                    try:
                        init_this = init if (i == 0 and j == 0) else "results"
                        pp.runpp(net, algorithm=algorithm, calculate_voltage_angles=calculate_voltage_angles, init=init_this,
                                 **_electrisim_enforce_q_lims_kw(net))
                        if net.converged:
                            loss_grid[i, j] = _economic_get_loss_mw(net)
                    except Exception:
                        pass  # left NaN: not counted, and said so below
            if len(load_vals) == 1 and len(gen_vals) == 1:
                losses_per_hour = np.full(time_steps, loss_grid[0, 0])
            elif len(load_vals) == 1:
                losses_per_hour = np.interp(gen_scale, gen_vals, loss_grid[0])
            elif len(gen_vals) == 1:
                losses_per_hour = np.interp(load_scale, load_vals, loss_grid[:, 0])
            else:
                from scipy.interpolate import RegularGridInterpolator
                interp = RegularGridInterpolator((load_vals, gen_vals), loss_grid, method='linear')
                losses_per_hour = interp(np.column_stack((load_scale, gen_scale)))
            uncounted = int(np.count_nonzero(~np.isfinite(losses_per_hour)))
            if uncounted:
                energy_warnings.append(
                    f'{uncounted} of {time_steps} hours fall on operating points where the power flow '
                    'did not converge; their losses are not counted.'
                )
            total_energy_period_mwh = float(np.nansum(np.maximum(losses_per_hour, 0)))

            if orig_load_p is not None:
                net.load['p_mw'] = orig_load_p
                net.load['q_mvar'] = orig_load_q
            if orig_sgen_p is not None:
                net.sgen['p_mw'] = orig_sgen_p
                if orig_sgen_q is not None:
                    net.sgen['q_mvar'] = orig_sgen_q
            lifetime_years_econ = max(1, min(100, int(params.get('lifetime_years', 30))))
            total_energy_losses_period_mwh = round(total_energy_period_mwh, 4)
            # The lifetime is in years, so the period is scaled to one first:
            # a 24-hour period gave "30 years" of losses equal to 30 days.
            annual_mwh = total_energy_period_mwh * 8760.0 / time_steps
            if time_steps < 8760:
                energy_warnings.append(
                    f'Losses over the first {time_steps} hours of the year (from 1 January) '
                    'were scaled to a full year; simulate 8760 hours for the seasons.'
                )
            total_energy_losses_annual_mwh = round(annual_mwh, 4)
            total_energy_losses_mwh = round(annual_mwh * lifetime_years_econ, 4)
            energy_loss_period_hours = time_steps
            if energy_price is not None and energy_price > 0:
                energy_loss_cost = round(total_energy_losses_mwh * energy_price, 2)
        
        result = {
            'total_capex': round(total_capex, 2),
            'total_power_losses_mw': round(total_power_losses_mw, 6),
            'capex_breakdown': capex_breakdown,
            'power_losses_breakdown': power_losses_breakdown,
            'currency': currency
        }
        if total_energy_losses_mwh is not None:
            lifetime_years = max(1, min(100, int(params.get('lifetime_years', 30))))
            result['total_energy_losses_period_mwh'] = total_energy_losses_period_mwh
            result['total_energy_losses_annual_mwh'] = total_energy_losses_annual_mwh
            result['total_energy_losses_mwh'] = total_energy_losses_mwh
            result['energy_loss_cost'] = energy_loss_cost
            result['energy_loss_cost_currency'] = energy_price_currency
            result['energy_loss_period_hours'] = energy_loss_period_hours
            result['time_steps'] = energy_loss_period_hours
            result['lifetime_years'] = lifetime_years
            result['generation_profile'] = generation_profile
            result['load_profile'] = load_profile
            result['calculation_mode'] = calculation_mode
            result['load_profile_values'] = load_scale.tolist()
            result['generation_profile_values'] = gen_scale.tolist()
            if energy_warnings:
                result['warnings'] = energy_warnings
        return result
        
    except Exception as e:
        import traceback
        error_msg = f"Economic analysis failed: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return {
            'error': error_msg,
            'total_capex': 0,
            'total_power_losses_mw': 0,
            'total_energy_losses_mwh': None,
            'energy_loss_cost': None,
            'capex_breakdown': [],
            'power_losses_breakdown': [],
            'currency': params.get('currency', 'EUR')
        }


def _rpc_q_curve_table_subset(qtbl, cid):
    """Rows matching id_q_capability_curve == cid (tolerant int/float/numpy types)."""
    try:
        cnum = int(np.floor(float(cid)))
    except (TypeError, ValueError):
        return qtbl.iloc[0:0]
    col = pd.to_numeric(qtbl['id_q_capability_curve'], errors='coerce')
    return qtbl.loc[col == cnum]


def _rpc_interp_sgen_pq_limits(net, sgen_idx, p_mw):
    """RPC alias for shared P–Q limit interpolation."""
    return _interp_sgen_pq_limits(net, sgen_idx, p_mw)


def _rpc_sgen_has_q_curve(net, sgen_idx):
    """True if this sgen row is linked to at least two points in q_capability_curve_table."""
    try:
        if not hasattr(net, 'sgen') or net.sgen.empty:
            return False
        if 'id_q_capability_characteristic' not in net.sgen.columns:
            return False
        cid = net.sgen.at[sgen_idx, 'id_q_capability_characteristic']
        if cid is None or (isinstance(cid, float) and pd.isna(cid)):
            return False
        qtbl = net.get('q_capability_curve_table', None)
        if qtbl is None or (hasattr(qtbl, 'empty') and qtbl.empty):
            return False
        if 'id_q_capability_curve' not in qtbl.columns:
            return False
        sub = _rpc_q_curve_table_subset(qtbl, cid)
        return len(sub) >= 2
    except Exception:
        return False


def _rpc_sgen_q_caps(net, sgen_idx, p_gen, sn, mode):
    """
    (q_pos_cap, q_neg_cap) for RPC sweeps: positive Q uses q_pos_cap * fraction;
    negative Q uses -q_neg_cap * fraction. For symmetric modes both are equal.
    """
    if mode == 'from_sgen_curve':
        lim = _rpc_interp_sgen_pq_limits(net, sgen_idx, p_gen)
        if lim is not None:
            q_mi, q_ma = lim
            q_pos = max(0.0, float(q_ma))
            q_neg = max(0.0, float(-q_mi))
            return q_pos, q_neg
        if sn > 0:
            fr = math.sqrt(max(sn ** 2 - p_gen ** 2, 0))
            return fr, fr
        return 0.0, 0.0
    if mode == 'from_rating':
        if sn <= 0:
            return 0.0, 0.0
        fr = math.sqrt(max(sn ** 2 - p_gen ** 2, 0))
        return fr, fr
    half = sn * 0.5 if sn > 0 else 0.0
    return half, half


def _rpc_masked_q_interp_clip(p_target, p_list, q_list):
    """
    Interpolate Q at p_target using only finite (p, q) pairs (skip None q from failed PF).
    Outside the span of valid P, clip to endpoint values (matches line chart behavior).
    Returns None if there are no valid pairs.
    """
    pairs = []
    for p, q in zip(p_list, q_list):
        if q is None:
            continue
        try:
            pairs.append((float(p), float(q)))
        except (TypeError, ValueError):
            continue
    if not pairs:
        return None
    pairs.sort(key=lambda t: t[0])
    px = np.array([t[0] for t in pairs], dtype=float)
    qy = np.array([t[1] for t in pairs], dtype=float)
    pt = float(p_target)
    if len(px) == 1:
        return float(qy[0])
    return float(np.interp(pt, px, qy, left=float(qy[0]), right=float(qy[-1])))


def _rpc_merge_uq_voltages(voltage_levels, uq_requirements, requirements=None):
    """
    U-Q/Pmax needs a voltage sweep. If the client sent U-Q requirement voltages
    (or only one P-Q voltage such as 1.0 pu), union those U points into voltage_levels
    so the U-Q chart is a curve rather than two dots at a single U.
    Replicates the P-Q requirement envelope onto any newly added voltage keys.
    """
    extra = []
    if isinstance(uq_requirements, dict):
        extra = uq_requirements.get('u_pu') or []
    merged = []
    seen = set()
    for v in list(voltage_levels or []) + list(extra or []):
        try:
            fv = round(float(v), 4)
        except (TypeError, ValueError):
            continue
        if fv in seen:
            continue
        seen.add(fv)
        merged.append(fv)
    merged.sort()
    if not merged:
        merged = [1.0]
    if isinstance(requirements, dict) and requirements:
        proto = None
        for val in requirements.values():
            if isinstance(val, dict) and val.get('p_mw'):
                proto = val
                break
        if proto:
            for v in merged:
                vk = f"{float(v):.4f}"
                if vk not in requirements:
                    requirements[vk] = proto
    return merged


def _rpc_build_uq_curve(voltage_levels, curves, p_target_mw):
    """
    Extract Q_min/Q_max at P ≈ p_target from per-voltage P-Q curves for U-Q/Pmax chart.
    """
    u_pu = []
    q_max_mvar = []
    q_min_mvar = []
    p_target = float(p_target_mw)

    for v in voltage_levels:
        v_key = f"{float(v):.4f}"
        curve = curves.get(v_key, {})
        p_arr = curve.get('p_mw', [])
        q_max_arr = curve.get('q_max_mvar', [])
        q_min_arr = curve.get('q_min_mvar', [])

        q_max_v = _rpc_masked_q_interp_clip(p_target, p_arr, q_max_arr)
        q_min_v = _rpc_masked_q_interp_clip(p_target, p_arr, q_min_arr)

        u_pu.append(round(float(v), 4))
        q_max_mvar.append(round(q_max_v, 4) if q_max_v is not None else None)
        q_min_mvar.append(round(q_min_v, 4) if q_min_v is not None else None)

    return {
        'u_pu': u_pu,
        'q_max_mvar': q_max_mvar,
        'q_min_mvar': q_min_mvar,
        'p_mw': round(p_target, 4),
    }


def _rpc_check_uq_compliance(uq_curve, uq_requirements, tol_mvar=1e-4):
    """
    Compare achieved Q at Pmax vs U-Q requirement envelope over voltage.
    Returns True, False, or None if requirements missing / insufficient data.
    """
    if not uq_requirements or not uq_curve:
        return None

    req_u = uq_requirements.get('u_pu', [])
    req_q_max = uq_requirements.get('q_req_max_mvar', [])
    req_q_min = uq_requirements.get('q_req_min_mvar', [])
    n = min(len(req_u), len(req_q_max), len(req_q_min))
    if n < 1:
        return None

    try:
        order = np.argsort([float(req_u[i]) for i in range(n)])
        ru = np.array([float(req_u[i]) for i in order], dtype=float)
        rmax = np.array([float(req_q_max[i]) for i in order], dtype=float)
        rmin = np.array([float(req_q_min[i]) for i in order], dtype=float)
    except (TypeError, ValueError):
        return None

    cap_u = uq_curve.get('u_pu', [])
    cap_q_max = uq_curve.get('q_max_mvar', [])
    cap_q_min = uq_curve.get('q_min_mvar', [])

    u_lo = float(ru[0])
    u_hi = float(ru[-1])
    u_check = set(float(x) for x in ru.tolist())
    for u in cap_u:
        try:
            uf = float(u)
        except (TypeError, ValueError):
            continue
        if u_lo <= uf <= u_hi:
            u_check.add(uf)

    for u_s in sorted(u_check):
        req_max_v = float(np.interp(u_s, ru, rmax, left=float(rmax[0]), right=float(rmax[-1])))
        req_min_v = float(np.interp(u_s, ru, rmin, left=float(rmin[0]), right=float(rmin[-1])))
        cap_max_v = _rpc_masked_q_interp_clip(u_s, cap_u, cap_q_max)
        cap_min_v = _rpc_masked_q_interp_clip(u_s, cap_u, cap_q_min)
        if cap_max_v is None or cap_min_v is None:
            return False
        if cap_max_v < req_max_v - tol_mvar or cap_min_v > req_min_v + tol_mvar:
            return False

    return True


def _electrisim_tap_control_snapshot(net):
    """DiscreteTapControl results from a solved net (for diagram overlay / RPC snapshots)."""
    rows = []
    if net is None or not hasattr(net, 'res_bus') or net.res_bus is None or net.res_bus.empty:
        return rows
    ufn = getattr(net, 'user_friendly_names', None) or {}

    def _side_vm(element, idx, side):
        try:
            if element == 'trafo3w':
                bus_col = {'hv': 'hv_bus', 'mv': 'mv_bus', 'lv': 'lv_bus'}.get(str(side), 'lv_bus')
                bus = int(net.trafo3w.at[idx, bus_col])
            else:
                bus_col = 'lv_bus' if str(side) == 'lv' else 'hv_bus'
                bus = int(net.trafo.at[idx, bus_col])
            return float(net.res_bus.at[bus, 'vm_pu'])
        except Exception:
            return None

    # The Grid Code Compliance (P-Q) study rewrites the band to bias taps toward a
    # worst-case setpoint. Report the band the user configured, and keep the pinned
    # one alongside it so the diagram never shows limits the user did not enter.
    configured_bands = getattr(net, '_pq_tap_bands_original', None) or {}

    def _one(element, table, idx, side, vm_lo, vm_hi):
        study_lo, study_hi = float(vm_lo), float(vm_hi)
        vm_lo, vm_hi = configured_bands.get(f'{element}:{idx}', (study_lo, study_hi))
        if vm_hi < vm_lo:
            vm_lo, vm_hi = vm_hi, vm_lo
        try:
            name = str(table.at[idx, 'name'])
            tap_pos = float(table.at[idx, 'tap_pos'])
            tap_min = float(table.at[idx, 'tap_min'])
            tap_max = float(table.at[idx, 'tap_max'])
            tap_step = float(table.at[idx, 'tap_step_percent']) if 'tap_step_percent' in table.columns else 0.0
            hv_bus = int(table.at[idx, 'hv_bus'])
            lv_bus = int(table.at[idx, 'lv_bus'])
            hv_vm = float(net.res_bus.at[hv_bus, 'vm_pu'])
            lv_vm = float(net.res_bus.at[lv_bus, 'vm_pu'])
            controlled_vm = _side_vm(element, idx, side)
            if controlled_vm is None:
                controlled_vm = lv_vm if str(side) == 'lv' else hv_vm
            in_limits = float(vm_lo) - 1e-6 <= float(controlled_vm) <= float(vm_hi) + 1e-6
            at_limit_type = None
            if tap_pos >= tap_max - 1e-9:
                at_limit_type = 'max'
            elif tap_pos <= tap_min + 1e-9:
                at_limit_type = 'min'
            cell_id = None
            if 'id' in table.columns:
                try:
                    raw = table.at[idx, 'id']
                    if raw is not None and not pd.isna(raw):
                        cell_id = str(raw)
                except Exception:
                    cell_id = None
            rec = {
                'element': element,
                'name': str(ufn.get(name, name)),
                'id': str(name),
                'cell_id': cell_id,
                'tap_pos': tap_pos,
                'tap_min': tap_min,
                'tap_max': tap_max,
                'tap_step_percent': tap_step,
                'control_side': str(side),
                'controlled_vm_pu': round(float(controlled_vm), 4),
                'vm_lower_pu': float(vm_lo),
                'vm_upper_pu': float(vm_hi),
                'hv_vm_pu': round(hv_vm, 4),
                'lv_vm_pu': round(lv_vm, 4),
                'in_limits': bool(in_limits),
                'at_limit': at_limit_type,
            }
            if abs(study_lo - vm_lo) > 1e-9 or abs(study_hi - vm_hi) > 1e-9:
                rec['study_band_lower_pu'] = study_lo
                rec['study_band_upper_pu'] = study_hi
            if element == 'trafo3w' and 'mv_bus' in table.columns:
                try:
                    rec['mv_vm_pu'] = round(float(net.res_bus.at[int(table.at[idx, 'mv_bus']), 'vm_pu']), 4)
                except Exception:
                    pass
            rows.append(rec)
        except Exception:
            return

    for spec in getattr(net, 'trafo_discrete_tap_controllers', None) or []:
        if not spec or len(spec) < 4:
            continue
        _one('trafo', net.trafo, spec[0], spec[1], spec[2], spec[3])
    for spec in getattr(net, 'trafo3w_discrete_tap_controllers', None) or []:
        if not spec or len(spec) < 4:
            continue
        if hasattr(net, 'trafo3w') and not net.trafo3w.empty:
            _one('trafo3w', net.trafo3w, spec[0], spec[1], spec[2], spec[3])
    return rows


def _rpc_clean_pf_val(v):
    if isinstance(v, (float, np.floating)):
        if math.isnan(v) or math.isinf(v):
            return None
        return float(v)
    return v


def _rpc_serialize_solved_net(net):
    """
    Build the same load-flow result shape the frontend expects, from an already-solved net
    (used for RPC PQ point snapshots — no second powerflow run).
    """
    if not hasattr(net, 'res_bus') or net.res_bus is None or net.res_bus.empty:
        return None
    try:
        result = {}

        busbar_list = []
        for index, row in net.res_bus.iterrows():
            p_mw = float(row['p_mw'])
            q_mvar = float(row['q_mvar'])
            denom_pf = math.sqrt(p_mw ** 2 + q_mvar ** 2)
            pf = (p_mw / denom_pf) if denom_pf > 0 and not math.isnan(denom_pf) else 0.0
            q_p = (q_mvar / p_mw) if p_mw != 0 and not math.isnan(p_mw) else 0.0
            if math.isnan(q_p) or math.isinf(q_p):
                q_p = 0.0
            p_br, q_br = _electrisim_bus_branch_p_q_sum(net, index)
            p_nodal, q_nodal = _electrisim_bus_nodal_p_q_sum(net, index)
            denom_pf_nodal = math.sqrt(p_nodal ** 2 + q_nodal ** 2)
            pf_nodal = (p_nodal / denom_pf_nodal) if denom_pf_nodal > 0 and not math.isnan(denom_pf_nodal) else 0.0
            q_p_nodal = (q_nodal / p_nodal) if p_nodal != 0 and not math.isnan(p_nodal) else 0.0
            if math.isnan(q_p_nodal) or math.isinf(q_p_nodal):
                q_p_nodal = 0.0
            _vm_pu = float(row['vm_pu'])
            _vn_kv = float(net.bus.at[index, 'vn_kv'])
            _vm_kv = float(_vm_pu) * _vn_kv if _vn_kv > 0 and _vm_pu == _vm_pu else None
            busbar_list.append({
                'name': str(net.bus.at[index, 'name']),
                'id': str(net.bus.at[index, 'id']) if 'id' in net.bus.columns else str(index),
                'vm_pu': _rpc_clean_pf_val(_vm_pu),
                'va_degree': _rpc_clean_pf_val(row['va_degree']),
                'p_mw': _rpc_clean_pf_val(p_mw),
                'q_mvar': _rpc_clean_pf_val(q_mvar),
                'pf': _rpc_clean_pf_val(pf),
                'q_p': _rpc_clean_pf_val(q_p),
                'p_branch_mw': _rpc_clean_pf_val(p_br),
                'q_branch_mvar': _rpc_clean_pf_val(q_br),
                'p_nodal_mw': _rpc_clean_pf_val(p_nodal),
                'q_nodal_mvar': _rpc_clean_pf_val(q_nodal),
                'pf_nodal': _rpc_clean_pf_val(pf_nodal),
                'q_p_nodal': _rpc_clean_pf_val(q_p_nodal),
                'vm_kv': _rpc_clean_pf_val(_vm_kv),
            })
        result['busbars'] = busbar_list

        if hasattr(net, 'res_line') and not net.res_line.empty:
            lines_list = []
            for index, row in net.res_line.iterrows():
                lines_list.append({
                    'name': str(net.line.at[index, 'name']),
                    'id': str(net.line.at[index, 'id']) if 'id' in net.line.columns else str(index),
                    'p_from_mw': _rpc_clean_pf_val(row['p_from_mw']),
                    'q_from_mvar': _rpc_clean_pf_val(row['q_from_mvar']),
                    'p_to_mw': _rpc_clean_pf_val(row['p_to_mw']),
                    'q_to_mvar': _rpc_clean_pf_val(row['q_to_mvar']),
                    'i_from_ka': _rpc_clean_pf_val(row['i_from_ka']),
                    'i_to_ka': _rpc_clean_pf_val(row['i_to_ka']),
                    'loading_percent': _rpc_clean_pf_val(row['loading_percent']),
                })
            result['lines'] = lines_list

        if hasattr(net, 'res_ext_grid') and not net.res_ext_grid.empty:
            ext_list = []
            for index, row in net.res_ext_grid.iterrows():
                p_mw = float(row['p_mw'])
                q_mvar = float(row['q_mvar'])
                denom = math.sqrt(p_mw ** 2 + q_mvar ** 2)
                ext_list.append({
                    'name': str(net.ext_grid.at[index, 'name']),
                    'id': str(net.ext_grid.at[index, 'id']) if 'id' in net.ext_grid.columns else str(index),
                    'p_mw': _rpc_clean_pf_val(p_mw),
                    'q_mvar': _rpc_clean_pf_val(q_mvar),
                    'pf': _rpc_clean_pf_val(p_mw / denom if denom > 0 else 0.0),
                    'q_p': _rpc_clean_pf_val(q_mvar / p_mw if p_mw != 0 else 0.0),
                })
            result['externalgrids'] = ext_list

        if hasattr(net, 'res_sgen') and not net.res_sgen.empty:
            sgen_list = []
            for index, row in net.res_sgen.iterrows():
                sgen_list.append({
                    'name': str(net.sgen.at[index, 'name']),
                    'id': str(net.sgen.at[index, 'id']) if 'id' in net.sgen.columns else str(index),
                    'p_mw': _rpc_clean_pf_val(row['p_mw']),
                    'q_mvar': _rpc_clean_pf_val(row['q_mvar']),
                })
            result['staticgenerators'] = sgen_list

        if hasattr(net, 'res_gen') and not net.res_gen.empty:
            gen_list = []
            for index, row in net.res_gen.iterrows():
                gen_list.append({
                    'name': str(net.gen.at[index, 'name']),
                    'id': str(net.gen.at[index, 'id']) if 'id' in net.gen.columns else str(index),
                    'p_mw': _rpc_clean_pf_val(row['p_mw']),
                    'q_mvar': _rpc_clean_pf_val(row['q_mvar']),
                    'va_degree': _rpc_clean_pf_val(row['va_degree']),
                    'vm_pu': _rpc_clean_pf_val(row['vm_pu']),
                })
            result['generators'] = gen_list

        if hasattr(net, 'trafo') and not net.trafo.empty:
            res_tf = getattr(net, 'res_trafo', None)
            trafo_list = []
            for trafo_index in net.trafo.index:
                t_name = net.trafo.at[trafo_index, 'name']
                t_raw_id = net.trafo.at[trafo_index, 'id'] if 'id' in net.trafo.columns else trafo_index
                t_id = _trafo_out_id(t_raw_id, t_name, trafo_index)
                row = _pf_res_row_for_element(net.trafo, res_tf, trafo_index)
                if row is None:
                    row = {k: 0.0 for k in (
                        'p_hv_mw', 'q_hv_mvar', 'p_lv_mw', 'q_lv_mvar', 'pl_mw', 'ql_mvar',
                        'i_hv_ka', 'i_lv_ka', 'vm_hv_pu', 'vm_lv_pu', 'va_hv_degree', 'va_lv_degree',
                        'loading_percent')}
                trafo_list.append({
                    'name': str(t_name),
                    'id': str(t_id),
                    'p_hv_mw': _rpc_clean_pf_val(row['p_hv_mw']),
                    'q_hv_mvar': _rpc_clean_pf_val(row['q_hv_mvar']),
                    'p_lv_mw': _rpc_clean_pf_val(row['p_lv_mw']),
                    'q_lv_mvar': _rpc_clean_pf_val(row['q_lv_mvar']),
                    'pl_mw': _rpc_clean_pf_val(row.get('pl_mw', 0.0)),
                    'ql_mvar': _rpc_clean_pf_val(row.get('ql_mvar', 0.0)),
                    'i_hv_ka': _rpc_clean_pf_val(row['i_hv_ka']),
                    'i_lv_ka': _rpc_clean_pf_val(row['i_lv_ka']),
                    'vm_hv_pu': _rpc_clean_pf_val(row.get('vm_hv_pu', 1.0)),
                    'vm_lv_pu': _rpc_clean_pf_val(row.get('vm_lv_pu', 1.0)),
                    'va_hv_degree': _rpc_clean_pf_val(row.get('va_hv_degree', 0.0)),
                    'va_lv_degree': _rpc_clean_pf_val(row.get('va_lv_degree', 0.0)),
                    'loading_percent': _rpc_clean_pf_val(row['loading_percent']),
                    'tap_pos': _rpc_clean_pf_val(
                        net.trafo.at[trafo_index, 'tap_pos'] if 'tap_pos' in net.trafo.columns else None),
                })
            if trafo_list:
                result['transformers'] = trafo_list

        if hasattr(net, 'trafo3w') and not net.trafo3w.empty:
            res_t3 = getattr(net, 'res_trafo3w', None)
            t3_list = []
            for t3_index in net.trafo3w.index:
                t_name = net.trafo3w.at[t3_index, 'name']
                t_raw_id = net.trafo3w.at[t3_index, 'id'] if 'id' in net.trafo3w.columns else t3_index
                t_id = _trafo_out_id(t_raw_id, t_name, t3_index)
                row = _pf_res_row_for_element(net.trafo3w, res_t3, t3_index)
                if row is None:
                    row = {k: 0.0 for k in (
                        'p_hv_mw', 'q_hv_mvar', 'p_mv_mw', 'q_mv_mvar', 'p_lv_mw', 'q_lv_mvar',
                        'pl_mw', 'ql_mvar', 'i_hv_ka', 'i_mv_ka', 'i_lv_ka',
                        'vm_hv_pu', 'vm_mv_pu', 'vm_lv_pu',
                        'va_hv_degree', 'va_mv_degree', 'va_lv_degree', 'loading_percent')}
                t3_list.append({
                    'name': str(t_name),
                    'id': str(t_id),
                    'p_hv_mw': _rpc_clean_pf_val(row['p_hv_mw']),
                    'q_hv_mvar': _rpc_clean_pf_val(row['q_hv_mvar']),
                    'p_mv_mw': _rpc_clean_pf_val(row['p_mv_mw']),
                    'q_mv_mvar': _rpc_clean_pf_val(row['q_mv_mvar']),
                    'p_lv_mw': _rpc_clean_pf_val(row['p_lv_mw']),
                    'q_lv_mvar': _rpc_clean_pf_val(row['q_lv_mvar']),
                    'loading_percent': _rpc_clean_pf_val(row['loading_percent']),
                    'i_hv_ka': _rpc_clean_pf_val(row.get('i_hv_ka', 0.0)),
                    'i_mv_ka': _rpc_clean_pf_val(row.get('i_mv_ka', 0.0)),
                    'i_lv_ka': _rpc_clean_pf_val(row.get('i_lv_ka', 0.0)),
                })
            if t3_list:
                result['transformers3W'] = t3_list

        if hasattr(net, 'res_shunt') and not net.res_shunt.empty:
            shunts_list = []
            caps_list = []
            for index, row in net.res_shunt.iterrows():
                typ = net.shunt.at[index, 'typ'] if 'typ' in net.shunt.columns else 'shuntreactor'
                p_out, q_out, vm_out = _electrisim_shunt_res_for_output(net, index, row)
                entry = {
                    'name': str(net.shunt.at[index, 'name']),
                    'id': str(net.shunt.at[index, 'id']) if 'id' in net.shunt.columns else str(index),
                    'p_mw': _rpc_clean_pf_val(p_out),
                    'q_mvar': _rpc_clean_pf_val(q_out),
                    'vm_pu': _rpc_clean_pf_val(vm_out),
                }
                if typ == 'capacitor':
                    caps_list.append(entry)
                else:
                    try:
                        sw = net.shunt.at[index, 'step']
                        smx = net.shunt.at[index, 'max_step']
                        zb = _electrisim_shunt_is_zero_based(net, index)
                        entry['step'] = _rpc_clean_pf_val(_electrisim_shunt_step_from_pp(sw, zb) if sw is not None and not pd.isna(sw) else None)
                        entry['max_step'] = _rpc_clean_pf_val(_electrisim_shunt_step_from_pp(smx, zb) if zb and smx is not None and not pd.isna(smx) else (float(smx) if smx is not None and not pd.isna(smx) else None))
                    except Exception:
                        pass
                    shunts_list.append(entry)
            if shunts_list:
                result['shunts'] = shunts_list
            if caps_list:
                result['capacitors'] = caps_list

        if hasattr(net, 'res_load') and not net.res_load.empty:
            loads_list = []
            for index, row in net.res_load.iterrows():
                loads_list.append({
                    'name': str(net.load.at[index, 'name']),
                    'id': str(net.load.at[index, 'id']) if 'id' in net.load.columns else str(index),
                    'p_mw': _rpc_clean_pf_val(row['p_mw']),
                    'q_mvar': _rpc_clean_pf_val(row['q_mvar']),
                })
            result['loads'] = loads_list

        if hasattr(net, 'res_switch') and not net.res_switch.empty:
            sw_list = []
            for index, row in net.res_switch.iterrows():
                sw_list.append({
                    'name': str(net.switch.at[index, 'name']),
                    'id': str(net.switch.at[index, 'id']) if 'id' in net.switch.columns else str(index),
                    'closed': bool(net.switch.at[index, 'closed']) if 'closed' in net.switch.columns else True,
                    'i_ka': _rpc_clean_pf_val(row.get('i_ka', 0.0)),
                    'p_from_mw': _rpc_clean_pf_val(row.get('p_from_mw', 0.0)),
                    'q_from_mvar': _rpc_clean_pf_val(row.get('q_from_mvar', 0.0)),
                    'p_to_mw': _rpc_clean_pf_val(row.get('p_to_mw', 0.0)),
                    'q_to_mvar': _rpc_clean_pf_val(row.get('q_to_mvar', 0.0)),
                    'loading_percent': _rpc_clean_pf_val(row.get('loading_percent', 0.0)),
                })
            result['switches'] = sw_list

        tap_rows = _electrisim_tap_control_snapshot(net)
        if tap_rows:
            result['tap_control_results'] = tap_rows

        return _sanitize_for_strict_json(result)
    except Exception:
        traceback.print_exc()
        return None


def _rpc_store_point_snapshot(point_loadflows, v_key, side, p_val, net_pf):
    snap = _rpc_serialize_solved_net(net_pf)
    if not snap:
        return
    if v_key not in point_loadflows:
        point_loadflows[v_key] = {'q_max': {}, 'q_min': {}}
    p_key = f"{float(p_val):.4f}"
    point_loadflows[v_key][side][p_key] = snap


def _rpc_build_and_run_point_net(net, ext_grid_idx, v_pu, gen_info, total_installed_mw, p_val,
                                 q_capability_mode, direction, q_frac, verbose_iwamoto,
                                 run_control_trafo2w, run_control_trafo3w, run_control_shunt):
    """Rebuild generator dispatch for one RPC point and run power flow; returns solved net or None."""
    net_pt = deepcopy(net)
    net_pt.ext_grid.at[ext_grid_idx, 'vm_pu'] = float(v_pu)
    for g in gen_info:
        share = g['p_rated_mw'] / total_installed_mw
        p_gen = float(p_val) * share
        q_pos_cap, q_neg_cap = _rpc_sgen_q_caps(net, g['idx'], p_gen, g['sn_mva'], q_capability_mode)
        if direction == 'max':
            net_pt.sgen.at[g['idx'], 'q_mvar'] = q_pos_cap * q_frac
        else:
            net_pt.sgen.at[g['idx'], 'q_mvar'] = -q_neg_cap * q_frac
        net_pt.sgen.at[g['idx'], 'p_mw'] = p_gen
    if not _rpc_run_pf_robust(
            net_pt, verbose_iwamoto, run_control_trafo2w, run_control_trafo3w, run_control_shunt):
        return None
    return net_pt


def _rpc_eval_q_frac(net, ext_grid_idx, v_pu, gen_info, total_installed_mw, p_val,
                     q_capability_mode, direction, frac, limit_overloads, max_loading_percent,
                     pcc_bus_idx, verbose_iwamoto, rc2, rc3, rcs):
    """
    Dispatch every selected static generator at `frac` of its Q-capability in the given
    direction ('max' = overexcited/+Q, 'min' = underexcited/-Q), run power flow, and report
    feasibility. Returns dict with keys: converged (bool), overloaded (bool), q_pcc (float|None).
    """
    net_try = deepcopy(net)
    net_try.ext_grid.at[ext_grid_idx, 'vm_pu'] = float(v_pu)
    sign = 1.0 if direction == 'max' else -1.0
    for g in gen_info:
        share = g['p_rated_mw'] / total_installed_mw
        p_gen = float(p_val) * share
        q_pos_cap, q_neg_cap = _rpc_sgen_q_caps(net, g['idx'], p_gen, g['sn_mva'], q_capability_mode)
        q_full = q_pos_cap if direction == 'max' else q_neg_cap
        net_try.sgen.at[g['idx'], 'p_mw'] = p_gen
        net_try.sgen.at[g['idx'], 'q_mvar'] = sign * q_full * float(frac)

    if not _rpc_run_pf_robust(net_try, verbose_iwamoto, rc2, rc3, rcs):
        return {'converged': False, 'overloaded': False, 'q_pcc': None}

    overloaded = False
    if limit_overloads:
        if not net_try.res_trafo.empty and net_try.res_trafo.loading_percent.max() > max_loading_percent:
            overloaded = True
        if not net_try.res_line.empty and net_try.res_line.loading_percent.max() > max_loading_percent:
            overloaded = True

    return {
        'converged': True,
        'overloaded': overloaded,
        'q_pcc': _rpc_pcc_q_for_chart(net_try, pcc_bus_idx, ext_grid_idx),
    }


def _rpc_max_feasible_q(net, ext_grid_idx, v_pu, gen_info, total_installed_mw, p_val,
                        q_capability_mode, direction, limit_overloads, max_loading_percent,
                        pcc_bus_idx, verbose_iwamoto, rc2, rc3, rcs,
                        max_iterations=16, frac_tol=2.5e-3):
    """
    Resolve the reactive-power boundary in one direction by finding the maximum fraction
    (0..1) of each unit's Q-capability whose power flow converges and, when
    limit_overloads=True, stays within max_loading_percent.

    Instead of snapping to a coarse fraction ladder (which under-reports the boundary when
    full capability fails to converge), this bisects the feasible fraction so the reported
    Q at the PCC is the true feasible edge.

    Feasibility is assumed monotonic in the fraction (larger |Q| -> harder to converge and
    higher branch loading), consistent with the physical voltage-collapse / loading trend.

    Returns (q_pcc, frac_used, limit_reason):
      - limit_reason is None when full capability (frac=1.0) is feasible,
        'overload' when the binding limit is branch loading, or
        'divergence' when the binding limit is power-flow non-convergence.
      - Returns (None, None, None) if even unity-Q (frac=0.0) fails to converge.
    """
    def feas(frac):
        return _rpc_eval_q_frac(
            net, ext_grid_idx, v_pu, gen_info, total_installed_mw, p_val,
            q_capability_mode, direction, frac, limit_overloads, max_loading_percent,
            pcc_bus_idx, verbose_iwamoto, rc2, rc3, rcs)

    top = feas(1.0)
    if top['converged'] and not top['overloaded']:
        return top['q_pcc'], 1.0, None

    base = feas(0.0)
    if not base['converged']:
        return None, None, None

    best_q = base['q_pcc']
    best_frac = 0.0
    limit_reason = 'overload' if (top['converged'] and top['overloaded']) else 'divergence'

    lo, hi = 0.0, 1.0
    for _ in range(max_iterations):
        if (hi - lo) < frac_tol:
            break
        mid = 0.5 * (lo + hi)
        r = feas(mid)
        if r['converged'] and not r['overloaded']:
            lo = mid
            best_q = r['q_pcc']
            best_frac = mid
        else:
            hi = mid
            limit_reason = 'overload' if (r['converged'] and r['overloaded']) else 'divergence'

    return best_q, best_frac, limit_reason


def reactive_power_capability(net, rpc_params):
    """
    Perform Reactive Power Capability (RPC) analysis for a plant
    (static generators and/or wind turbines; both map to pandapower sgen).
    Sweeps active power and determines Q_min/Q_max at the PCC bus for
    each requested voltage level. Compares against grid code requirements.

    rpc_params['q_capability_mode']:
      - 'from_rating': circular limit sqrt(S_n^2 - P^2) per unit
      - 'fixed_fraction': 0.5 * S_n
      - 'from_sgen_curve': interpolate q_min/q_max vs P from net.q_capability_curve_table
        (after apply_sgen_q_capability_curves); units without a curve fall back to from_rating

    rpc_params may include verbose_iwamoto (default False): when True, pandapower's
    per-iteration Iwamoto multiplier lines are printed; otherwise they are suppressed.
    """
    import traceback

    try:
        pcc_bus_name = rpc_params.get('pcc_bus_name')
        ext_grid_name = rpc_params.get('ext_grid_name')
        generator_names = rpc_params.get('generator_names', [])
        voltage_levels = list(rpc_params.get('voltage_levels', [1.0]) or [1.0])
        p_min_mw = float(rpc_params.get('p_min_mw', 0))
        p_max_mw = float(rpc_params.get('p_max_mw', 0))
        p_steps = int(rpc_params.get('p_steps', 10))
        q_capability_mode = rpc_params.get('q_capability_mode', 'from_rating')
        limit_overloads = rpc_params.get('limit_overloads', False)
        max_loading_percent = float(rpc_params.get('max_loading_percent', 100))
        requirements = rpc_params.get('requirements', None)
        uq_requirements = rpc_params.get('uq_requirements', None)
        grid_code_template_name = rpc_params.get('grid_code_template_name')
        uq_grid_code_template_name = rpc_params.get('uq_grid_code_template_name')
        voltage_levels = _rpc_merge_uq_voltages(voltage_levels, uq_requirements, requirements)
        verbose_iwamoto = bool(rpc_params.get('verbose_iwamoto', False))
        progress_cb = rpc_params.get('_progress_callback')
        rc2, rc3, rcs = _resolve_controller_family_flags(rpc_params)
        run_control_any = rc2 or rc3 or rcs

        print(f"=== RPC Analysis ===")
        print(f"  PCC bus: {pcc_bus_name}")
        print(f"  Ext grid: {ext_grid_name}")
        print(f"  Generators: {generator_names}")
        print(f"  Voltage levels: {voltage_levels}")
        print(f"  P range: {p_min_mw} - {p_max_mw} MW, {p_steps} steps")
        print(f"  Q capability mode: {q_capability_mode}")
        print(f"  Limit overloads: {limit_overloads}")
        print(
            f"  controllers: 2w_tap={rc2}, 3w_tap={rc3}, shunt={rcs} "
            f"(any={run_control_any})"
        )
        if verbose_iwamoto:
            print(f"  verbose_iwamoto: True (pandapower will print each Iwamoto multiplier line)")

        # Resolve PCC bus index
        pcc_bus_idx = None
        for idx in net.bus.index:
            if net.bus.at[idx, 'name'] == pcc_bus_name:
                pcc_bus_idx = idx
                break
        if pcc_bus_idx is None:
            return json.dumps({'error': f'PCC bus "{pcc_bus_name}" not found in network'}, separators=(',', ':'))

        # Resolve external grid index
        ext_grid_idx = None
        for idx in net.ext_grid.index:
            if net.ext_grid.at[idx, 'name'] == ext_grid_name:
                ext_grid_idx = idx
                break
        if ext_grid_idx is None:
            return json.dumps({'error': f'External grid "{ext_grid_name}" not found in network'}, separators=(',', ':'))

        # Resolve generator (sgen) indices
        sgen_indices = []
        for idx in net.sgen.index:
            if net.sgen.at[idx, 'name'] in generator_names:
                sgen_indices.append(idx)

        # Also check gen table
        gen_indices = []
        if hasattr(net, 'gen') and not net.gen.empty:
            for idx in net.gen.index:
                if net.gen.at[idx, 'name'] in generator_names:
                    gen_indices.append(idx)

        if not sgen_indices and not gen_indices:
            return json.dumps({'error': 'No matching generators found in the network'}, separators=(',', ':'))

        print(f"  Resolved: PCC bus idx={pcc_bus_idx}, ext_grid idx={ext_grid_idx}")
        print(f"  sgen indices: {sgen_indices}, gen indices: {gen_indices}")

        # Resolve user-friendly PCC bus name for display
        pcc_bus_friendly = pcc_bus_name
        if hasattr(net, 'user_friendly_names') and pcc_bus_name in net.user_friendly_names:
            pcc_bus_friendly = net.user_friendly_names[pcc_bus_name]

        # Compute installed active-power capacity (P_rated) for P-axis / dispatch shares.
        # Use p_mw, not sn_mva — apparent power overstates wind-farm MW (e.g. 30×16 MVA vs 30×15 MW).
        gen_info = []
        for idx in sgen_indices:
            sn = net.sgen.at[idx, 'sn_mva'] if 'sn_mva' in net.sgen.columns and not pd.isna(net.sgen.at[idx, 'sn_mva']) else 0
            p_mw = float(net.sgen.at[idx, 'p_mw']) if not pd.isna(net.sgen.at[idx, 'p_mw']) else 0.0
            sn_f = float(sn) if sn > 0 else 0.0
            p_rated = p_mw if p_mw > 0 else sn_f
            name = net.sgen.at[idx, 'name']
            gen_info.append({
                'type': 'sgen',
                'idx': idx,
                'name': name,
                'p_rated_mw': p_rated,
                'sn_mva': sn_f if sn_f > 0 else p_rated,
            })

        total_installed_mw = sum(g['p_rated_mw'] for g in gen_info)
        if total_installed_mw <= 0:
            return json.dumps({'error': 'Total installed capacity is zero. Set p_mw on static generators or wind turbines.'}, separators=(',', ':'))

        if p_max_mw <= 0:
            p_max_mw = total_installed_mw

        p_points = np.linspace(p_min_mw, p_max_mw, max(p_steps + 1, 2))

        curves = {}
        point_loadflows = {}
        warnings_list = []
        try:
            eg_bus = int(net.ext_grid.at[ext_grid_idx, 'bus'])
            if int(pcc_bus_idx) != eg_bus:
                warnings_list.append(
                    'Named PCC bus differs from External Grid bus — red curves show net Q at the named PCC '
                    '(res_bus after PF); ensure grid-code blue curves refer to the same node.'
                )
        except Exception:
            pass
        if q_capability_mode == 'from_sgen_curve':
            if not any(_rpc_sgen_has_q_curve(net, g['idx']) for g in gen_info):
                warnings_list.append(
                    'Q mode "from_sgen_curve": no selected static generator or wind turbine has an active P–Q curve '
                    '(enable reactive capability on the unit). Using circular √(S_n²−P²) fallback for all.'
                )
        tc2_list = getattr(net, 'trafo_discrete_tap_controllers', None) or []
        tc3_list = getattr(net, 'trafo3w_discrete_tap_controllers', None) or []
        tc_names = []
        for row in tc2_list:
            try:
                ti = row[0]
                tc_names.append('2w:' + str(net.trafo.at[ti, 'name']))
            except Exception:
                tc_names.append(str(row[0]) if row else '?')
        for row in tc3_list:
            try:
                ti = row[0]
                tc_names.append('3w:' + str(net.trafo3w.at[ti, 'name']))
            except Exception:
                tc_names.append('3w:' + str(row[0]) if row else '?')
        shc_list = getattr(net, 'shunt_discrete_controllers', None) or []
        has_applicable = (rc2 and tc2_list) or (rc3 and tc3_list) or (rcs and shc_list)
        if run_control_any and not has_applicable:
            warnings_list.append(
                'Include controller is on, but no elements have discrete controllers configured '
                'for the selected controller types '
                '(enable discrete tap on transformers or discrete shunt control on shunt reactors). RPC runs as unconstrained PF.'
            )
        compliance = {}

        for v_pu in voltage_levels:
            v_key = f"{float(v_pu):.4f}"
            _vl_msg = f"\n  --- Voltage level: {v_pu} pu ---"
            print(_vl_msg)
            if progress_cb:
                progress_cb(_vl_msg)

            p_result = []
            q_max_result = []
            q_min_result = []

            for p_total in p_points:
                p_val = float(p_total)

                # Emit progress per P step so the NDJSON stream keeps producing bytes during the
                # (per-point) convergence bisection; long silent gaps otherwise let proxies/dev
                # tunnels drop the HTTP/2 connection mid-computation.
                if progress_cb:
                    progress_cb(f"    P = {p_val:.1f} MW: resolving Q_max / Q_min boundary ...")

                # --- Q_max (overexcited, +Q) & Q_min (underexcited, -Q) ---
                # Resolve each boundary via convergence-based bisection so the reported limit
                # is the true feasible fraction of unit capability, instead of snapping to a
                # coarse ladder step when full capability fails to converge.
                q_max_pcc, q_max_frac_used, q_max_reason = _rpc_max_feasible_q(
                    net, ext_grid_idx, float(v_pu), gen_info, total_installed_mw, p_val,
                    q_capability_mode, 'max', limit_overloads, max_loading_percent, pcc_bus_idx,
                    verbose_iwamoto, rc2, rc3, rcs)

                if q_max_pcc is None:
                    print(f"    Q_max PF failed at P={p_val:.1f}MW, V={v_pu}pu (all strategies)")
                else:
                    if q_max_frac_used is not None and q_max_frac_used < 0.999:
                        _reason = ('overload' if q_max_reason == 'overload'
                                   else 'power flow non-convergence at higher Q')
                        warnings_list.append(
                            f"V={v_pu}pu, P={p_val:.1f}MW: Q_max limited to "
                            f"{q_max_frac_used*100:.0f}% capability ({_reason})"
                        )
                    net_qmax_snap = _rpc_build_and_run_point_net(
                        net, ext_grid_idx, float(v_pu), gen_info, total_installed_mw, p_val,
                        q_capability_mode, 'max', q_max_frac_used, verbose_iwamoto, rc2, rc3, rcs)
                    if net_qmax_snap is not None:
                        _rpc_store_point_snapshot(point_loadflows, v_key, 'q_max', p_val, net_qmax_snap)

                q_min_pcc, q_min_frac_used, q_min_reason = _rpc_max_feasible_q(
                    net, ext_grid_idx, float(v_pu), gen_info, total_installed_mw, p_val,
                    q_capability_mode, 'min', limit_overloads, max_loading_percent, pcc_bus_idx,
                    verbose_iwamoto, rc2, rc3, rcs)

                if q_min_pcc is None:
                    print(f"    Q_min PF failed at P={p_val:.1f}MW, V={v_pu}pu (all strategies)")
                else:
                    if q_min_frac_used is not None and q_min_frac_used < 0.999:
                        _reason = ('overload' if q_min_reason == 'overload'
                                   else 'power flow non-convergence at higher Q')
                        warnings_list.append(
                            f"V={v_pu}pu, P={p_val:.1f}MW: Q_min limited to "
                            f"{q_min_frac_used*100:.0f}% capability ({_reason})"
                        )
                    net_qmin_snap = _rpc_build_and_run_point_net(
                        net, ext_grid_idx, float(v_pu), gen_info, total_installed_mw, p_val,
                        q_capability_mode, 'min', q_min_frac_used, verbose_iwamoto, rc2, rc3, rcs)
                    if net_qmin_snap is not None:
                        _rpc_store_point_snapshot(point_loadflows, v_key, 'q_min', p_val, net_qmin_snap)

                if progress_cb:
                    _qmx = f"{q_max_pcc:.1f}" if q_max_pcc is not None else "n/a"
                    _qmn = f"{q_min_pcc:.1f}" if q_min_pcc is not None else "n/a"
                    progress_cb(f"    P = {p_val:.1f} MW: Q_max={_qmx} Mvar, Q_min={_qmn} Mvar")

                p_result.append(round(p_val, 4))
                q_max_result.append(round(q_max_pcc, 4) if q_max_pcc is not None else None)
                q_min_result.append(round(q_min_pcc, 4) if q_min_pcc is not None else None)

            curves[v_key] = {
                'p_mw': p_result,
                'q_max_mvar': q_max_result,
                'q_min_mvar': q_min_result
            }

            # Check compliance against requirements for this voltage level
            if requirements:
                v_req = requirements.get(v_key, None)
                if v_req:
                    req_p = v_req.get('p_mw', [])
                    req_q_max = v_req.get('q_req_max_mvar', [])
                    req_q_min = v_req.get('q_req_min_mvar', [])
                    is_compliant = True
                    tol_mvar = 1e-4

                    n = min(len(req_p), len(req_q_max), len(req_q_min))
                    if n < 1:
                        is_compliant = False
                    else:
                        try:
                            order = np.argsort([float(req_p[i]) for i in range(n)])
                            rp = np.array([float(req_p[i]) for i in order], dtype=float)
                            rmax = np.array([float(req_q_max[i]) for i in order], dtype=float)
                            rmin = np.array([float(req_q_min[i]) for i in order], dtype=float)
                        except (TypeError, ValueError):
                            is_compliant = False
                            rp = rmax = rmin = None

                        if rp is not None:
                            p_lo = float(rp[0])
                            p_hi = float(rp[-1])
                            p_check = set(float(x) for x in rp.tolist())
                            for p_val in p_result:
                                try:
                                    pf = float(p_val)
                                except (TypeError, ValueError):
                                    continue
                                if p_lo <= pf <= p_hi:
                                    p_check.add(pf)
                            for p_s in sorted(p_check):
                                req_max_v = float(np.interp(p_s, rp, rmax, left=float(rmax[0]), right=float(rmax[-1])))
                                req_min_v = float(np.interp(p_s, rp, rmin, left=float(rmin[0]), right=float(rmin[-1])))
                                cap_max_v = _rpc_masked_q_interp_clip(p_s, p_result, q_max_result)
                                cap_min_v = _rpc_masked_q_interp_clip(p_s, p_result, q_min_result)
                                if cap_max_v is None or cap_min_v is None:
                                    is_compliant = False
                                    break
                                if cap_max_v < req_max_v - tol_mvar or cap_min_v > req_min_v + tol_mvar:
                                    is_compliant = False
                                    break

                    compliance[v_key] = is_compliant
                else:
                    compliance[v_key] = None
            else:
                compliance[v_key] = None

        p_target_uq = float(p_max_mw)
        uq_curve = _rpc_build_uq_curve(voltage_levels, curves, p_target_uq)
        uq_compliance = _rpc_check_uq_compliance(uq_curve, uq_requirements)

        result = {
            'rpc_results': {
                'voltage_levels': [round(float(v), 4) for v in voltage_levels],
                'curves': curves,
                'uq_curve': uq_curve,
                'point_loadflows': point_loadflows,
                'requirements': requirements if requirements else {},
                'uq_requirements': uq_requirements if uq_requirements else {},
                'compliance': compliance,
                'uq_compliance': uq_compliance,
                'warnings': warnings_list,
                'total_installed_mw': round(total_installed_mw, 4),
                'pcc_bus_name': pcc_bus_friendly,
                'generator_count': len(gen_info),
                'grid_code_template_name': grid_code_template_name,
                'grid_code_template_key': rpc_params.get('grid_code_template_key'),
                'uq_grid_code_template_name': uq_grid_code_template_name,
                'uq_grid_code_template_key': rpc_params.get('uq_grid_code_template_key'),
                'q_capability_mode': q_capability_mode,
                'tap_changer_control': {
                    'run_control_requested': run_control_any,
                    'controllers_applied': bool(has_applicable),
                    'transformer_count': len(tc2_list) + len(tc3_list),
                    'shunt_controller_count': len(shc_list),
                    'transformer_names': tc_names,
                },
                'pcc_q_convention': (
                    'Red curves: net reactive power at the selected PCC bus after power flow '
                    '(res_bus.q_mvar), including all shunts, lines, transformers, and injections at that bus. '
                    'Sign for the chart: if PCC is the External Grid bus, use res_bus.q_mvar as plotted; '
                    'if PCC is another bus, use −res_bus.q_mvar so +Q matches overexcited / −Q underexcited '
                    'relative to the usual grid-connection frame. Blue curves are grid-code PCC requirements.'
                ),
            }
        }

        return json.dumps(result, default=_json_serialize_default, separators=(',', ':'))

    except Exception as e:
        traceback.print_exc()
        return json.dumps({'error': f'RPC analysis failed: {str(e)}'}, separators=(',', ':'))


def _electrisim_attach_discrete_tap_controllers(net, attach_trafo=True, attach_trafo3w=True):
    """
    Register pandapower DiscreteTapControl for 2-winding (element trafo) and 3-winding (trafo3w)
    entries from create_other_elements.
    Safe on a fresh or deep-copied net before runpp(..., run_control=True).
    """
    def _attach_one(element, element_index, control_side, vm_lower_pu, vm_upper_pu):
        try:
            try:
                control.DiscreteTapControl(
                    net=net,
                    tid=element_index,
                    side=control_side,
                    vm_lower_pu=vm_lower_pu,
                    vm_upper_pu=vm_upper_pu,
                    element=element,
                )
            except TypeError:
                try:
                    control.DiscreteTapControl(
                        net=net,
                        element_index=element_index,
                        side=control_side,
                        vm_lower_pu=vm_lower_pu,
                        vm_upper_pu=vm_upper_pu,
                        element=element,
                    )
                except TypeError:
                    if element != 'trafo':
                        raise
                    control.DiscreteTapControl(
                        net=net,
                        element_index=element_index,
                        side=control_side,
                        vm_lower_pu=vm_lower_pu,
                        vm_upper_pu=vm_upper_pu,
                    )
        except Exception:
            pass

    if attach_trafo:
        for row in getattr(net, 'trafo_discrete_tap_controllers', None) or []:
            _attach_one('trafo', row[0], row[1], row[2], row[3])
    if attach_trafo3w:
        for row in getattr(net, 'trafo3w_discrete_tap_controllers', None) or []:
            _attach_one('trafo3w', row[0], row[1], row[2], row[3])


def _electrisim_attach_discrete_shunt_controllers(net):
    """
    Register pandapower DiscreteShuntController for specs in net.shunt_discrete_controllers
    (populated during create_other_elements for Shunt Reactor).
    See https://pandapower.readthedocs.io/en/latest/control/controller.html#discrete-shunt-control
    """
    lst = getattr(net, 'shunt_discrete_controllers', None) or []
    if not lst:
        return
    for spec in lst:
        try:
            bi = spec.get('bus_index')
            kwargs = dict(
                net=net,
                shunt_index=int(spec['shunt_index']),
                vm_set_pu=float(spec['vm_set_pu']),
                tol=float(spec.get('tol', 1e-3)),
                increment=int(spec.get('increment', 1)),
                reset_at_init=bool(spec.get('reset_at_init', False)),
            )
            if bi is not None:
                kwargs['bus_index'] = bi
            try:
                control.DiscreteShuntController(**kwargs)
            except TypeError:
                args = [net, kwargs['shunt_index'], kwargs['vm_set_pu']]
                k2 = dict(
                    tol=kwargs['tol'],
                    increment=kwargs['increment'],
                    reset_at_init=kwargs['reset_at_init'],
                )
                if bi is not None:
                    k2['bus_index'] = bi
                control.DiscreteShuntController(*args, **k2)
        except Exception:
            pass


def _rpc_pcc_q_for_chart(net_pf, pcc_bus_idx, ext_grid_idx):
    """
    Net reactive power (Mvar) at the PCC for RPC red curves: always res_bus.q_mvar at pcc_bus_idx
    after runpp, so shunts, lines, trafos, and all elements at that bus are included.

    Sign vs chart axes (overexcited +Q, underexcited −Q): when PCC is the External Grid bus, return
    q_mvar as stored in res_bus; when PCC is a different bus, return −q_mvar so the plotted sign
    matches the connection-side convention used alongside grid-code blue bands.
    """
    q_raw = float(net_pf.res_bus.at[pcc_bus_idx, 'q_mvar'])
    try:
        if ext_grid_idx is not None and not net_pf.ext_grid.empty:
            eg_bus = int(net_pf.ext_grid.at[ext_grid_idx, 'bus'])
            if int(pcc_bus_idx) == eg_bus:
                return q_raw
    except Exception:
        pass
    return -q_raw


def _rpc_run_pf_robust(net_pf, verbose_iwamoto=False, run_control_trafo2w=False, run_control_trafo3w=False, run_control_shunt=False):
    """
    Run power flow for RPC with multiple solver fallbacks (nr first, then iwamoto_nr).
    Pandapower's iwamoto_nr prints one line per iteration ("iwamoto muliplier: ...").
    By default those prints are captured and discarded so RPC does not flood server logs;
    pass verbose_iwamoto=True to forward them to stdout (for debugging).

    When any of run_control_trafo2w / run_control_trafo3w / run_control_shunt is True and the net
    lists matching controller specs, registers DiscreteTapControl / DiscreteShuntController /
    line-P CharacteristicControl for shunt step and runs pp.runpp(..., run_control=True) with a single NR
    strategy (controller state is not reliable across solver fallbacks on the same net).
    """
    import io

    q_kw = _electrisim_enforce_q_lims_kw(net_pf)
    tc2 = getattr(net_pf, 'trafo_discrete_tap_controllers', None) or []
    tc3 = getattr(net_pf, 'trafo3w_discrete_tap_controllers', None) or []
    shc = getattr(net_pf, 'shunt_discrete_controllers', None) or []
    lfc = getattr(net_pf, 'line_flow_shunt_controllers', None) or []
    attach_2w = bool(run_control_trafo2w) and bool(tc2)
    attach_3w = bool(run_control_trafo3w) and bool(tc3)
    attach_sh = bool(run_control_shunt) and bool(shc)
    attach_lf = bool(run_control_shunt) and bool(lfc)
    rc = attach_2w or attach_3w or attach_sh or attach_lf
    if rc:
        if attach_2w or attach_3w:
            _electrisim_attach_discrete_tap_controllers(net_pf, attach_trafo=attach_2w, attach_trafo3w=attach_3w)
        if attach_sh:
            _electrisim_attach_discrete_shunt_controllers(net_pf)
        if attach_lf:
            _electrisim_attach_line_flow_shunt_controllers(net_pf)
        strategies = [{'algorithm': 'nr', 'init': 'auto', 'max_iteration': 100}]
    else:
        strategies = [
            {'algorithm': 'nr', 'init': 'auto', 'max_iteration': 50},
            {'algorithm': 'nr', 'init': 'dc', 'max_iteration': 80},
            {'algorithm': 'nr', 'init': 'flat', 'max_iteration': 80},
            {'algorithm': 'iwamoto_nr', 'init': 'dc', 'max_iteration': 80},
        ]
    if _electrisim_net_has_facts(net_pf):
        strategies = [s for s in strategies if s['algorithm'] == 'nr']
    for s in strategies:
        algo = s['algorithm']
        try:
            if algo == 'iwamoto_nr' and not verbose_iwamoto:
                buf = io.StringIO()
                old_out, old_err = sys.stdout, sys.stderr
                sys.stdout = sys.stderr = buf
                try:
                    pp.runpp(net_pf,
                             algorithm=algo,
                             calculate_voltage_angles=True,
                             init=s['init'],
                             max_iteration=s['max_iteration'],
                             run_control=rc,
                             **q_kw)
                finally:
                    sys.stdout = old_out
                    sys.stderr = old_err
            else:
                pp.runpp(net_pf,
                         algorithm=algo,
                         calculate_voltage_angles=True,
                         init=s['init'],
                         max_iteration=s['max_iteration'],
                         run_control=rc,
                         **q_kw)
            return True
        except Exception:
            continue
    return False


# NOTE: The former _rpc_binary_search_q (overload-only bisection, invoked from a coarse
# fraction ladder) has been superseded by _rpc_max_feasible_q, which bisects the feasible
# Q fraction on both convergence and (optionally) branch loading for every RPC point.


# ============================================================================
# Protection Coordination
# ----------------------------------------------------------------------------
# End-to-end implementation built on pandapower.protection:
#   - OCRelay (DTOC / IDMT / IDTOC) with IEC 60255 curves
#   - Fuse with the pandapower fuse standard library (16-1000 A)
#   - calculate_protection_times() for tripping table
#   - device.create_characteristic() for time-current grading curves
# Differential (87) and Distance (21) devices are accepted as configuration but
# returned with `not_computed=true` because pandapower has no native 87/21 model.
# ============================================================================


# Map switch frontend `protection_type` (UI) -> internal kind used by the engine.
# OCR is one logical kind (DTOC/IDMT/IDTOC drives the subtype).
_PROTECTION_KIND_OCR = "ocr"
_PROTECTION_KIND_FUSE = "fuse"
_PROTECTION_KIND_EARTH_FAULT = "earth_fault"
_PROTECTION_KIND_DIRECTIONAL = "directional"
_PROTECTION_KIND_DIFF = "differential"
_PROTECTION_KIND_DIST = "distance"
_PROTECTION_KIND_NONE = "none"

# OC relay subtype -> pandapower switch.type marker used internally by the library.
_OC_SUBTYPE_TO_SWITCH_TYPE = {
    "DTOC": "CB_DTOC",
    "IDMT": "CB_IDMT",
    "IDTOC": "CB_IDTOC",
}

# IEC 60255 curve aliases the dialog might send.
_VALID_OC_CURVE_TYPES = {
    "standard_inverse",
    "very_inverse",
    "extremely_inverse",
    "long_inverse",
}

# IEEE / ANSI curves are evaluated by Electrisim because pandapower's OCRelay
# currently implements IEC 60255 curves only. Values use IEEE C37.112:
# t = TMS * (A / (M ** p - 1) + B).
_IEEE_OC_CURVES = {
    "ieee_moderately_inverse": (0.0515, 0.114, 0.02),
    "ieee_very_inverse": (19.61, 0.491, 2.0),
    "ieee_extremely_inverse": (28.2, 0.1217, 2.0),
}

# IEC 60255-151 k / alpha used by pandapower OCRelay._select_k_alpha.
_IEC_K_ALPHA = {
    "standard_inverse": (0.14, 0.02),
    "very_inverse": (13.5, 1.0),
    "extremely_inverse": (80.0, 2.0),
    "long_inverse": (120.0, 1.0),
}


def _prot_iec_k_alpha(curve_type):
    return _IEC_K_ALPHA.get(str(curve_type or "").lower(), _IEC_K_ALPHA["standard_inverse"])


def _prot_has_native_protection(net):
    prot = getattr(net, "protection", None)
    return prot is not None and len(prot) > 0


def _prot_protection_index(net):
    """Snapshot of ``net.protection`` row labels before a device is created."""
    prot = getattr(net, "protection", None)
    if prot is None:
        return set()
    return set(prot.index)


def _prot_drop_new_protection_rows(net, existing_index):
    """Remove protection rows added since ``existing_index`` was taken.

    pandapower protection devices register themselves in ``net.protection``
    inside ``__init__``, before their pickup currents and trip times are
    computed. A failure in that computation therefore leaves a half-initialized
    device behind, which would both duplicate the ElectriSim-evaluated relay in
    the results and break ``calculate_protection_times``.
    """
    prot = getattr(net, "protection", None)
    if prot is None or len(prot) == 0:
        return
    stale = [idx for idx in prot.index if idx not in existing_index]
    if stale:
        net.protection.drop(index=stale, inplace=True)


# Protection-tab fields that each device actually uses. The raw spec carries
# every field on the tab, so reporting all of them mixes in settings the chosen
# device ignores (e.g. an instantaneous pickup on a pure IDMT relay).
_PROT_OC_SUBTYPE_SETTING_KEYS = {
    "DTOC": ("oc_relay_type", "I_g_a", "I_gg_a", "t_g", "t_gg", "t_diff"),
    "IDMT": ("oc_relay_type", "curve_type", "I_s_a", "tms", "t_grade", "t_diff"),
    "IDTOC": ("oc_relay_type", "curve_type", "I_s_a", "I_g_a", "I_gg_a",
              "tms", "t_grade", "t_g", "t_gg", "t_diff"),
}

_PROT_KIND_SETTING_KEYS = {
    _PROTECTION_KIND_FUSE: ("fuse_type", "rated_i_a"),
    _PROTECTION_KIND_EARTH_FAULT: ("I_e_a", "t_e"),
    _PROTECTION_KIND_DIRECTIONAL: ("directional_mode", "I_g_a", "I_s_a", "t_g"),
    _PROTECTION_KIND_DIFF: ("I_diff_a", "diff_slope", "t_g"),
    _PROTECTION_KIND_DIST: ("z1_r_ohm", "z1_x_ohm", "t_z1", "z2_r_ohm", "z2_x_ohm",
                            "t_z2", "z3_r_ohm", "z3_x_ohm", "t_z3"),
}


def _prot_spec_settings_for_ui(spec, subtype=None):
    """Subtype-aware settings payload for an ElectriSim-evaluated device."""
    kind = spec.get("protection_type")
    if kind == _PROTECTION_KIND_OCR:
        subtype = str(subtype or spec.get("oc_relay_type") or "DTOC").upper()
        keys = list(_PROT_OC_SUBTYPE_SETTING_KEYS.get(
            subtype, _PROT_OC_SUBTYPE_SETTING_KEYS["DTOC"]))
        keys.append("pickup_mode")
        if spec.get("pickup_mode") != "manual":
            keys += ["overload_factor", "ct_current_factor", "safety_factor"]
    else:
        keys = list(_PROT_KIND_SETTING_KEYS.get(kind, ()))
    out = {}
    for key in keys:
        value = spec.get(key)
        if value is None or value == "":
            continue
        out[key] = value
    return out


def _prot_to_jsonable(obj, _depth=0):
    """Convert a protection-coordination payload to JSON-safe primitives.

    Never serializes pandapower protection device objects or pandas internals
    (those contain ``weakref.ReferenceType``).
    """
    if _depth > 40:
        return str(obj)
    if obj is None:
        return None
    if isinstance(obj, weakref.ReferenceType):
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    if isinstance(obj, str):
        return obj
    if isinstance(obj, np.ndarray):
        return _prot_to_jsonable(obj.tolist(), _depth + 1)
    if isinstance(obj, pd.DataFrame):
        return _prot_to_jsonable(obj.replace({np.nan: None}).to_dict(orient="records"), _depth + 1)
    if isinstance(obj, pd.Series):
        return _prot_to_jsonable(obj.replace({np.nan: None}).tolist(), _depth + 1)
    if isinstance(obj, pd.Timestamp):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    if isinstance(obj, dict):
        return {str(k): _prot_to_jsonable(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_prot_to_jsonable(v, _depth + 1) for v in obj]
    module = getattr(type(obj), "__module__", "") or ""
    name = type(obj).__name__
    if name in ("OCRelay", "Fuse", "ProtectionDevice", "BlockManager", "Flags") or "pandapower" in module:
        return str(obj)
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return _prot_to_jsonable(obj.item(), _depth + 1)
        except Exception:
            return str(obj)
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return {
            str(k): _prot_to_jsonable(v, _depth + 1)
            for k, v in vars(obj).items()
            if not str(k).startswith("_") and not isinstance(v, weakref.ReferenceType) and not callable(v)
        }
    return str(obj)


def _prot_apply_ocrelay_overrides(device, spec, grading_mode):
    """Force Switch-dialog pickups/times onto a native OCRelay.

    pandapower IDTOC indexes ``time_settings`` as a 5-element list (DataFrame
    manual grading is ignored / mistyped in time_grading). Manual pickup also
    copies I_g into I_gg. Override after construction so the device matches the
    dialog and ``net.protection`` still exists for calculate_protection_times.
    """
    if device is None:
        return device
    subtype = str(getattr(device, "oc_relay_type", None) or spec.get("oc_relay_type") or "").upper()
    if spec.get("pickup_mode") == "manual":
        if subtype in ("DTOC", "IDTOC") and spec.get("I_g_a") is not None:
            device.I_g = float(spec["I_g_a"]) / 1000.0
        if subtype in ("DTOC", "IDTOC") and spec.get("I_gg_a") is not None:
            device.I_gg = float(spec["I_gg_a"]) / 1000.0
        if subtype in ("IDMT", "IDTOC") and spec.get("I_s_a") is not None:
            device.I_s = float(spec["I_s_a"]) / 1000.0
    if grading_mode == "manual":
        if subtype in ("DTOC", "IDTOC"):
            if spec.get("t_g") is not None:
                device.t_g = float(spec["t_g"])
            if spec.get("t_gg") is not None:
                device.t_gg = float(spec["t_gg"])
        if subtype in ("IDMT", "IDTOC"):
            if spec.get("tms") is not None:
                device.tms = float(spec["tms"])
            if spec.get("t_grade") is not None:
                device.t_grade = float(spec["t_grade"])
    if getattr(device, "k", None) is None or getattr(device, "alpha", None) is None:
        k, alpha = _prot_iec_k_alpha(spec.get("curve_type") or getattr(device, "curve_type", None))
        device.k = k
        device.alpha = alpha
    return device


def _prot_eval_oc_trip(spec, subtype, current_a, evaluator="oc_electrisim"):
    """Trip time for an ElectriSim-side OCR (IEEE curves or native-OCRelay fallback)."""
    # A pickup of 0 A is unset, not "trip on any current".
    I_s, I_g, I_gg = (v if v is not None and v > 0 else None
                      for v in (spec.get("I_s_a"), spec.get("I_g_a"), spec.get("I_gg_a")))
    tms = spec.get("tms", 1.0)
    t_grade = spec.get("t_grade", 0.5)
    t_g = spec.get("t_g", 0.5)
    t_gg = spec.get("t_gg", 0.07)
    curve = str(spec.get("curve_type") or "standard_inverse").lower()
    subtype = str(subtype or spec.get("oc_relay_type") or "DTOC").upper()
    if current_a is None:
        return False, None, {}

    def inverse_time(pickup):
        if pickup is None or pickup <= 0 or current_a <= pickup:
            return None
        multiple = current_a / pickup
        if evaluator == "ieee_oc" or curve in _IEEE_OC_CURVES:
            a, b, p = _IEEE_OC_CURVES.get(curve, _IEEE_OC_CURVES["ieee_moderately_inverse"])
            t = float(tms or 1.0) * (a / (multiple ** p - 1.0) + b)
        else:
            k, alpha = _prot_iec_k_alpha(curve)
            t = float(tms or 1.0) * k / (multiple ** alpha - 1.0) + float(t_grade or 0.0)
        return t if math.isfinite(t) and t >= 0 else None

    if subtype == "DTOC":
        if I_gg is not None and current_a >= I_gg:
            return True, t_gg, {"element": "I>>"}
        if I_g is not None and current_a >= I_g:
            return True, t_g, {"element": "I>"}
        return False, None, {}
    if subtype == "IDMT":
        t = inverse_time(I_s if I_s else I_g)
        return (t is not None), t, {"element": "IDMT"}
    if I_gg is not None and current_a >= I_gg:
        return True, t_gg, {"element": "I>>"}
    if I_g is not None and current_a >= I_g:
        return True, t_g, {"element": "I>"}
    t = inverse_time(I_s if I_s else I_g)
    return (t is not None), t, {"element": "IDMT"}


def _prot_sample_spec_oc_characteristic(spec, subtype=None, curve_type=None, n_points=100):
    """Sample a complete OCR I–t curve from Switch-dialog settings (Amperes, seconds)."""
    subtype = str(subtype or spec.get("oc_relay_type") or "DTOC").upper()
    curve_type = str(curve_type or spec.get("curve_type") or "standard_inverse").lower()
    I_s = spec.get("I_s_a")
    I_g = spec.get("I_g_a")
    I_gg = spec.get("I_gg_a")
    pickups = [p for p in (I_s, I_g, I_gg) if p is not None and p > 0]
    x_min = max(1.0, min(pickups) * 0.5) if pickups else 10.0
    x_max = max(pickups) * 50.0 if pickups else 1.0e5
    x_max = max(x_max, 1.0e4)
    x = list(np.logspace(math.log10(x_min), math.log10(x_max), int(n_points)))
    for p in pickups:
        x.extend([p * 0.999, p, p * 1.001])
    x = sorted(set(float(v) for v in x if v > 0))
    currents, times = [], []
    evaluator = "ieee_oc" if curve_type in _IEEE_OC_CURVES else "oc_electrisim"
    for i_a in x:
        tripped, t_trip, _ = _prot_eval_oc_trip(spec, subtype, i_a, evaluator)
        currents.append(float(i_a))
        if tripped and t_trip is not None and math.isfinite(float(t_trip)) and float(t_trip) > 0:
            times.append(float(t_trip))
        else:
            times.append(None)
    return currents, times


def _prot_safe_float(value, default=None):
    """safe_float that tolerates None / empty / 'nan' and returns `default` instead of 0.0."""
    if value is None:
        return default
    if isinstance(value, str):
        v = value.strip()
        if v == "" or v.lower() == "nan" or v.lower() == "none":
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default
    try:
        f = float(value)
        if math.isnan(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


# --- Automatic OCR settings -------------------------------------------------------
#
# pandapower's OCRelay sets pickups only for relays on lines, and grades them only
# when every closed switch sits on a line: on any grid with a transformer breaker
# no relay set to "automatic" could be evaluated. These rules are pandapower's,
# carried over to transformer breakers, and evaluated by Electrisim.

_PROT_AUTO_SC_FRACTION = 0.95          # I>> of a line relay: a fault 95 % along its line
_PROT_AUTO_TRAFO_INST_MARGIN = 1.2     # I>> of a transformer relay: 120 % of its worst through-fault
_PROT_AUTO_INVERSE_OVERLOAD = 1.2      # IDMT pickup: 120 % of the rated current, as pandapower


def _prot_auto_grading_depths(net, sw_indices):
    """
    Relays crossed between the External Grid and each relay (itself included),
    the depth pandapower grades line relays by: the deepest trips first.
    """
    from collections import deque
    adjacency, element_node = _prot_element_graph(net)
    relay_edges = {}
    for sw_idx in sw_indices:
        sw = net.switch.loc[int(sw_idx)]
        if not bool(sw['closed']) or sw['et'] == 'b':
            continue
        relay_edges[frozenset((('bus', int(sw['bus'])), element_node(sw['et'], sw['element'])))] = int(sw_idx)
    dist, queue = {}, deque()
    for _, row in net.ext_grid.iterrows():
        if bool(row.get('in_service', True)):
            node = ('bus', int(row['bus']))
            dist[node] = 0
            queue.append(node)
    while queue:
        node = queue.popleft()
        for nxt in adjacency.get(node, ()):
            weight = 1 if frozenset((node, nxt)) in relay_edges else 0
            if dist[node] + weight < dist.get(nxt, math.inf):
                dist[nxt] = dist[node] + weight
                (queue.appendleft if weight == 0 else queue.append)(nxt)
    depths = {}
    for edge, sw_idx in relay_edges.items():
        known = [dist[n] for n in edge if n in dist]
        if known:
            depths[sw_idx] = min(known) + 1
    return depths


def _prot_auto_rated_ka(net, sw):
    """Rated current of what the relay protects, on its own side."""
    et, element, bus = str(sw['et']), int(sw['element']), int(sw['bus'])

    def num(row, key, default=1.0):
        value = row.get(key, default)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        return value if value > 0 and math.isfinite(value) else default

    if et == 'l' and element in net.line.index:
        row = net.line.loc[element]
        return num(row, 'max_i_ka', 0.0) * num(row, 'df') * num(row, 'parallel'), 'its line rating'
    if et == 't' and element in net.trafo.index:
        row = net.trafo.loc[element]
        side = 'hv' if int(row['hv_bus']) == bus else 'lv'
        return (num(row, 'sn_mva', 0.0) * num(row, 'parallel')
                / (math.sqrt(3) * num(row, f'vn_{side}_kv'))), f'the transformer {side.upper()} rating'
    if et == 't3' and element in net.trafo3w.index:
        row = net.trafo3w.loc[element]
        side = next((s for s in ('hv', 'mv', 'lv') if int(row[f'{s}_bus']) == bus), 'hv')
        return (num(row, f'sn_{side}_mva', 0.0)
                / (math.sqrt(3) * num(row, f'vn_{side}_kv'))), f'the transformer {side.upper()} rating'
    return 0.0, None


def _prot_auto_fault_current_ka(net, sw_idx, fault_bus=None, line_fraction=None):
    """Maximum three-phase fault current the relay sees, for a fault at a bus or along its line."""
    net_sc = deepcopy(net)
    if line_fraction is not None:
        from pandapower.protection.utility_functions import create_sc_bus
        element = int(net.switch.at[sw_idx, 'element'])
        net_sc = create_sc_bus(net_sc, sc_line_id=element, sc_fraction=float(line_fraction))
        # As in the line-fault scenarios: create_sc_bus moves non-line switches too.
        not_line = net.switch.index[net.switch['et'] != 'l']
        net_sc.switch.loc[not_line, 'element'] = net.switch.loc[not_line, 'element']
        fault_bus = int(max(net_sc.bus.index))
    ensure_ext_grid_zero_sequence_min(net_sc)
    sc.calc_sc(net_sc, bus=int(fault_bus), branch_results=True, fault='3ph', case='max')
    return _prot_switch_current_ka(net_sc, sw_idx)


def _prot_auto_oc_settings(net, spec, sw_idx, depths, grading_mode):
    """
    (settings, note): the relay's automatic pickups and grading, or (None, why).

    I>: the rated current x overload factor x CT factor. I>>: for a line relay
    the current it sees for a fault 95 % along its line (x safety factor); for a
    transformer relay on the grid side 120 % of the worst through-fault at its
    other terminals, so only an internal fault trips it at once, and none on
    the load side (an incomer cannot tell its busbar from a feeder fault). t>:
    the base t> for the deepest relay, plus t_diff for each relay nearer the
    grid.
    """
    sw = net.switch.loc[int(sw_idx)]
    et = str(sw['et'])
    rated_ka, rated_basis = _prot_auto_rated_ka(net, sw)
    if not rated_ka > 0:
        return None, 'the protected element has no current rating'
    out = dict(spec)
    overload = float(spec.get('overload_factor') or 1.25)
    ct = float(spec.get('ct_current_factor') or 1.2)
    safety = float(spec.get('safety_factor') or 1.0)
    out['I_g_a'] = rated_ka * overload * ct * 1000.0
    out['I_s_a'] = rated_ka * _PROT_AUTO_INVERSE_OVERLOAD * 1000.0

    inst_ka, inst_basis = None, None
    try:
        if et == 'l':
            line = net.line.loc[int(sw['element'])]
            at_from = int(line['from_bus']) == int(sw['bus'])
            fraction = _PROT_AUTO_SC_FRACTION if at_from else 1.0 - _PROT_AUTO_SC_FRACTION
            seen = _prot_auto_fault_current_ka(net, sw_idx, line_fraction=fraction)
            if seen:
                inst_ka = seen * safety
                inst_basis = f'a fault {_PROT_AUTO_SC_FRACTION:.0%} along its line'
        elif et in ('t', 't3'):
            adjacency, element_node = _prot_element_graph(net)
            grid = {('bus', int(b)) for b, live in zip(net.ext_grid['bus'], net.ext_grid.get('in_service', [True] * len(net.ext_grid))) if bool(live)}
            if _prot_reaches_source(adjacency, ('bus', int(sw['bus'])), grid,
                                    {element_node(et, sw['element'])}):
                table = net.trafo if et == 't' else net.trafo3w
                cols = ('hv_bus', 'lv_bus') if et == 't' else ('hv_bus', 'mv_bus', 'lv_bus')
                others = [int(table.at[int(sw['element']), c]) for c in cols
                          if int(table.at[int(sw['element']), c]) != int(sw['bus'])]
                through = [x for x in (_prot_auto_fault_current_ka(net, sw_idx, fault_bus=b) for b in others) if x]
                if through:
                    inst_ka = max(through) * _PROT_AUTO_TRAFO_INST_MARGIN * safety
                    inst_basis = f'{_PROT_AUTO_TRAFO_INST_MARGIN:.0%} of the worst through-fault'
            else:
                inst_basis = 'none on the load side of a transformer'
    except Exception as e:
        inst_basis = f'none: its fault current could not be computed ({type(e).__name__})'
    if inst_ka is not None and inst_ka * 1000.0 <= out['I_g_a']:
        inst_ka, inst_basis = None, 'none: the fault current along it is below I>'
    out['I_gg_a'] = inst_ka * 1000.0 if inst_ka is not None else None

    depth = depths.get(int(sw_idx))
    if grading_mode != 'manual' and depth is not None and depths:
        steps = max(depths.values()) - depth
        out['t_g'] = float(spec.get('t_g') or 0.5) + steps * float(spec.get('t_diff') or 0.3)

    parts = [f"I> {out['I_g_a']:.0f} A ({rated_basis} {rated_ka * 1000:.0f} A x {overload:g} x {ct:g})"]
    parts.append(f"I>> {out['I_gg_a']:.0f} A ({inst_basis})" if out['I_gg_a'] else f'I>> {inst_basis or "none"}')
    if depth is not None and grading_mode != 'manual':
        parts.append(f"t> {out['t_g']:.2f} s ({depth} relay{'s' if depth != 1 else ''} from the grid)")
    return out, 'Automatic settings by Electrisim: ' + '; '.join(parts) + '.'


def _prot_collect_switch_protection_specs(in_data):
    """
    Walk the raw frontend payload and collect the protection spec for every Switch
    component that has `protection_type` set. Returns a dict keyed by the frontend
    switch identifier (id, falling back to name) so we can match it back to the
    pandapower switch row after `pp.create_switch`.
    """
    specs = {}
    if not isinstance(in_data, dict):
        return specs
    for key, row in in_data.items():
        if not isinstance(row, dict):
            continue
        typ = row.get('typ') or row.get('type')
        if not isinstance(typ, str) or not typ.startswith('Switch'):
            continue
        protection_type = str(row.get('protection_type', _PROTECTION_KIND_NONE) or _PROTECTION_KIND_NONE).strip().lower()
        if protection_type in ('', _PROTECTION_KIND_NONE):
            continue
        sw_id = row.get('id') or row.get('name')
        if sw_id is None:
            continue
        specs[str(sw_id)] = {
            'protection_type': protection_type,
            'oc_relay_type': str(row.get('oc_relay_type', 'DTOC') or 'DTOC').upper(),
            'curve_type': str(row.get('curve_type') or '').lower(),
            'tms': _prot_safe_float(row.get('tms')),
            't_grade': _prot_safe_float(row.get('t_grade')),
            't_gg': _prot_safe_float(row.get('t_gg')),
            't_g': _prot_safe_float(row.get('t_g')),
            't_diff': _prot_safe_float(row.get('t_diff')),
            'pickup_mode': str(row.get('pickup_mode', 'auto') or 'auto').lower(),
            'I_s_a': _prot_safe_float(row.get('I_s_a')),
            'I_g_a': _prot_safe_float(row.get('I_g_a')),
            'I_gg_a': _prot_safe_float(row.get('I_gg_a')),
            'fuse_type': row.get('fuse_type'),
            'fuse_mode': str(row.get('fuse_mode', 'library') or 'library').strip().lower(),
            'fuse_custom_std_json': row.get('fuse_custom_std_json'),
            'rated_i_a': _prot_safe_float(row.get('rated_i_a')),
            'overload_factor': _prot_safe_float(row.get('overload_factor')),
            'ct_current_factor': _prot_safe_float(row.get('ct_current_factor')),
            'safety_factor': _prot_safe_float(row.get('safety_factor')),
            'I_e_a': _prot_safe_float(row.get('I_e_a')),
            't_e': _prot_safe_float(row.get('t_e')),
            'directional_mode': str(row.get('directional_mode', 'forward') or 'forward').lower(),
            'I_diff_a': _prot_safe_float(row.get('I_diff_a')),
            'diff_slope': _prot_safe_float(row.get('diff_slope')),
            'z1_r_ohm': _prot_safe_float(row.get('z1_r_ohm')),
            'z1_x_ohm': _prot_safe_float(row.get('z1_x_ohm')),
            'z2_r_ohm': _prot_safe_float(row.get('z2_r_ohm')),
            'z2_x_ohm': _prot_safe_float(row.get('z2_x_ohm')),
            'z3_r_ohm': _prot_safe_float(row.get('z3_r_ohm')),
            'z3_x_ohm': _prot_safe_float(row.get('z3_x_ohm')),
            't_z1': _prot_safe_float(row.get('t_z1')),
            't_z2': _prot_safe_float(row.get('t_z2')),
            't_z3': _prot_safe_float(row.get('t_z3')),
            'sw_id': str(sw_id),
            'sw_name': str(row.get('name', sw_id)),
            'user_friendly_name': str(row.get('userFriendlyName', row.get('name', sw_id))),
        }
    return specs


def _prot_merge_study_defaults(specs, prot_params):
    """Fill omitted / blank switch settings from the study Grading tab."""
    defaults = {
        'curve_type': str(prot_params.get('curve_type', 'standard_inverse') or 'standard_inverse').lower(),
        'tms': _prot_safe_float(prot_params.get('tms'), 1.0),
        't_grade': _prot_safe_float(prot_params.get('t_grade'), 0.5),
        't_gg': _prot_safe_float(prot_params.get('t_gg'), 0.07),
        't_g': _prot_safe_float(prot_params.get('t_g'), 0.5),
        't_diff': _prot_safe_float(prot_params.get('t_diff'), 0.3),
        'overload_factor': _prot_safe_float(prot_params.get('overload_factor'), 1.25),
        'ct_current_factor': _prot_safe_float(prot_params.get('ct_current_factor'), 1.2),
        'safety_factor': _prot_safe_float(prot_params.get('safety_factor'), 1.0),
    }
    for spec in specs.values():
        for key, value in defaults.items():
            if spec.get(key) is None or spec.get(key) == '':
                spec[key] = value
    return specs


def _prot_resolve_sw_idx_for_id(net, frontend_id):
    """Locate pandapower switch row index given the frontend cell id we wrote into net.switch['id']."""
    if 'id' not in net.switch.columns:
        return None
    matches = net.switch.index[net.switch['id'].astype(str) == str(frontend_id)].tolist()
    if matches:
        return int(matches[0])
    # Fallback: try matching by name
    if 'name' in net.switch.columns:
        name_matches = net.switch.index[net.switch['name'].astype(str) == str(frontend_id)].tolist()
        if name_matches:
            return int(name_matches[0])
    return None


def _prot_build_pickup_current_manual_df(spec):
    """Build a manual pickup dataframe for OCRelay when pickup_mode == 'manual'.

    Switch-dialog pickups are in A; pandapower OCRelay stores and compares kA.
    """
    if spec.get('pickup_mode') != 'manual':
        return None
    subtype = spec.get('oc_relay_type', 'DTOC')
    cols = {}
    if subtype == 'DTOC':
        if spec.get('I_gg_a') is None or spec.get('I_g_a') is None:
            return None
        cols = {
            'switch_id': [0],
            'I_gg': [float(spec['I_gg_a']) / 1000.0],
            'I_g': [float(spec['I_g_a']) / 1000.0],
        }
    elif subtype == 'IDMT':
        if spec.get('I_s_a') is None:
            return None
        cols = {'switch_id': [0], 'I_s': [float(spec['I_s_a']) / 1000.0]}
    elif subtype == 'IDTOC':
        if spec.get('I_gg_a') is None or spec.get('I_g_a') is None or spec.get('I_s_a') is None:
            return None
        cols = {
            'switch_id': [0],
            'I_gg': [float(spec['I_gg_a']) / 1000.0],
            'I_g': [float(spec['I_g_a']) / 1000.0],
            'I_s': [float(spec['I_s_a']) / 1000.0],
        }
    if not cols:
        return None
    return pd.DataFrame(cols)


def _prot_build_oc_relay_time_settings(spec, grading_mode='auto', manual_time_settings=None):
    """Build the `time_settings` list expected by OCRelay for the chosen subtype.

    IDTOC always uses a 5-element list: pandapower indexes ``time_settings[0:5]``
    and its DataFrame column check for IDTOC is mistyped (``t_gg`` twice).
    Manual times are applied afterwards via ``_prot_apply_ocrelay_overrides``.
    """
    subtype = spec.get('oc_relay_type', 'DTOC')
    t_gg = spec.get('t_gg', 0.07)
    t_g = spec.get('t_g', 0.5)
    t_diff = spec.get('t_diff', 0.3)
    tms = spec.get('tms', 1.0)
    t_grade = spec.get('t_grade', 0.5)
    if subtype == 'IDTOC':
        return [t_gg, t_g, t_diff, tms, t_grade]
    if grading_mode == 'manual' and manual_time_settings is not None:
        return manual_time_settings
    if subtype == 'DTOC':
        return [t_gg, t_g, t_diff]
    if subtype == 'IDMT':
        return [tms, t_grade]
    return [t_gg, t_g, t_diff]


def _prot_build_manual_time_settings(net, specs, subtype):
    """Return the per-switch DataFrame format expected by pandapower time_grading.

    The frame is indexed by switch index because OCRelay reads its times with
    ``time_grading.t_g[switch_index]`` (a label lookup), which would otherwise
    miss whenever switch indices are not a contiguous 0..n-1 range.
    """
    rows = []
    for sw_idx in net.switch.index:
        matching = next(
            (s for s in specs.values()
             if _prot_resolve_sw_idx_for_id(net, s.get('sw_id')) == int(sw_idx)
             and s.get('protection_type') == _PROTECTION_KIND_OCR
             and s.get('oc_relay_type') == subtype),
            None
        )
        matching = matching or {}
        if subtype == 'IDMT':
            rows.append({'switch_id': int(sw_idx), 'tms': matching.get('tms', 1.0),
                         't_grade': matching.get('t_grade', 0.5)})
        else:
            rows.append({'switch_id': int(sw_idx), 't_gg': matching.get('t_gg', 0.07),
                         't_g': matching.get('t_g', 0.5)})
    return pd.DataFrame(rows, index=[int(i) for i in net.switch.index])


def _prot_ensure_bus_geodata(net):
    """
    pandapower.protection.utility_functions.create_sc_bus reads bus coordinates from
    `net.bus.geo` (pandapower 3.x) or `net.bus_geodata` (older), so we synthesize
    sequential placeholder coordinates for any bus that is missing them. The
    coordinates themselves do not affect the calculation, they only satisfy the
    geodata lookup.
    """
    # pandapower 3.x stores geo on the bus DataFrame as JSON in column 'geo'
    try:
        if 'geo' in net.bus.columns:
            for idx, bus_row in net.bus.iterrows():
                geo_val = bus_row['geo']
                if geo_val is None or (isinstance(geo_val, float) and math.isnan(geo_val)) or geo_val == '' or geo_val == 'null':
                    net.bus.at[idx, 'geo'] = json.dumps({
                        "type": "Point",
                        "coordinates": [float(idx) * 1.0, 0.0]
                    })
        else:
            # Add the column so pandapower.protection can read it consistently.
            net.bus['geo'] = [
                json.dumps({"type": "Point", "coordinates": [float(i) * 1.0, 0.0]})
                for i in net.bus.index
            ]
    except Exception as e:
        print(f"Protection: failed to ensure net.bus.geo - {e}")

    # Legacy net.bus_geodata table (pre-3.x)
    try:
        if hasattr(net, 'bus_geodata'):
            for idx in net.bus.index:
                if idx not in net.bus_geodata.index:
                    net.bus_geodata.loc[idx] = [float(idx) * 1.0, 0.0]
    except Exception:
        pass


def _prot_register_custom_fuse_std_types(net, specs):
    """
    For each switch with protection_type=fuse and fuse_mode=custom, register
    net.std_types['fuse'][name] via pp.create_std_type so Fuse() can load curves.
    """
    errors = []
    if not hasattr(pp, 'create_std_type'):
        return errors
    for sw_id, spec in specs.items():
        if spec.get('protection_type') != _PROTECTION_KIND_FUSE:
            continue
        mode = str(spec.get('fuse_mode', 'library') or 'library').strip().lower()
        if mode != 'custom':
            continue
        fuse_name = str(spec.get('fuse_type') or '').strip()
        if not fuse_name:
            errors.append({
                'switch_id': sw_id,
                'switch_name': spec.get('sw_name'),
                'user_friendly_name': spec.get('user_friendly_name'),
                'kind': 'Fuse',
                'fuse_register': True,
                'attached': False,
                'not_computed': False,
                'reason': 'Custom fuse: missing fuse name (fuse_type).',
            })
            continue
        rated = spec.get('rated_i_a')
        if rated is None or float(rated) <= 0:
            errors.append({
                'switch_id': sw_id,
                'switch_name': spec.get('sw_name'),
                'user_friendly_name': spec.get('user_friendly_name'),
                'kind': 'Fuse',
                'fuse_register': True,
                'attached': False,
                'not_computed': False,
                'reason': 'Custom fuse: rated_i_a must be positive.',
            })
            continue
        raw_json = spec.get('fuse_custom_std_json')
        extra = {}
        if raw_json is not None and str(raw_json).strip():
            try:
                if isinstance(raw_json, dict):
                    extra = raw_json
                else:
                    extra = json.loads(str(raw_json))
                if not isinstance(extra, dict):
                    raise ValueError('curve JSON must be a JSON object')
            except Exception as e:
                errors.append({
                    'switch_id': sw_id,
                    'switch_name': spec.get('sw_name'),
                    'user_friendly_name': spec.get('user_friendly_name'),
                    'kind': 'Fuse',
                    'fuse_register': True,
                    'attached': False,
                    'not_computed': False,
                    'reason': f'Custom fuse curve JSON: {e}',
                })
                continue
        try:
            data = {'fuse_type': fuse_name, 'i_rated_a': float(rated)}
            for k in ('t_avg', 't_min', 't_total', 'x_avg', 'x_min', 'x_total'):
                if k in extra:
                    data[k] = extra[k]
            pp.create_std_type(net, data, name=fuse_name, element='fuse', overwrite=True)
        except Exception as e:
            errors.append({
                'switch_id': sw_id,
                'switch_name': spec.get('sw_name'),
                'user_friendly_name': spec.get('user_friendly_name'),
                'kind': 'Fuse',
                'fuse_register': True,
                'attached': False,
                'not_computed': False,
                'reason': f'create_std_type failed: {e}',
            })
    return errors


def _attach_protection_devices(net, specs, grading_mode='auto'):
    """
    Instantiate pandapower protection devices for every collected spec. Returns a list of
    summaries (one per device) describing what was attached (or why it was skipped).
    """
    summaries = []
    if not specs:
        return summaries

    # Ensure bus coordinates are present (required by pandapower.protection.create_sc_bus).
    _prot_ensure_bus_geodata(net)

    summaries.extend(_prot_register_custom_fuse_std_types(net, specs))

    # Lazy imports: OCRelay pulls matplotlib via pandapower; Fuse does not. Import separately so
    # fuse-only studies still run on minimal environments (pip install matplotlib for OCR).
    OCRelay = None
    _ocrelay_import_err = None
    try:
        from pandapower.protection.protection_devices.ocrelay import OCRelay as _OCRelay
        OCRelay = _OCRelay
    except ImportError as e:
        _ocrelay_import_err = str(e)

    Fuse = None
    try:
        from pandapower.protection.protection_devices.fuse import Fuse as _Fuse
        Fuse = _Fuse
    except ImportError:
        Fuse = None

    manual_times = {
        subtype: _prot_build_manual_time_settings(net, specs, subtype)
        for subtype in _OC_SUBTYPE_TO_SWITCH_TYPE
    } if grading_mode == 'manual' else {}
    auto_depths = None  # relays' grading depths, found when first needed

    for sw_id, spec in specs.items():
        sw_idx = _prot_resolve_sw_idx_for_id(net, sw_id)
        if sw_idx is None:
            summaries.append({
                'switch_id': sw_id,
                'switch_name': spec.get('sw_name'),
                'user_friendly_name': spec.get('user_friendly_name'),
                'kind': spec.get('protection_type'),
                'attached': False,
                'not_computed': False,
                'reason': 'Switch index not found in pandapower net (switch may have been skipped).',
            })
            continue

        kind = spec.get('protection_type', _PROTECTION_KIND_NONE)

        # --- OCR --------------------------------------------------------------------
        if kind == _PROTECTION_KIND_OCR:
            if OCRelay is None:
                summaries.append({
                    'switch_id': sw_id,
                    'switch_name': spec.get('sw_name'),
                    'user_friendly_name': spec.get('user_friendly_name'),
                    'kind': kind,
                    'attached': False,
                    'not_computed': True,
                    'reason': (
                        f'pandapower.protection.OCRelay unavailable ({_ocrelay_import_err or "unknown"}). '
                        'Install matplotlib in the backend environment for OCR relays (e.g. pip install matplotlib).'
                    ),
                })
                continue
            subtype = spec.get('oc_relay_type', 'DTOC')
            if subtype not in _OC_SUBTYPE_TO_SWITCH_TYPE:
                summaries.append({
                    'switch_id': sw_id, 'switch_name': spec.get('sw_name'),
                    'user_friendly_name': spec.get('user_friendly_name'),
                    'kind': kind, 'subtype': subtype, 'attached': False, 'not_computed': False,
                    'reason': f'Unknown OCR subtype "{subtype}". Expected DTOC / IDMT / IDTOC.',
                })
                continue
            try:
                net.switch.at[sw_idx, 'type'] = _OC_SUBTYPE_TO_SWITCH_TYPE[subtype]
            except Exception:
                pass
            curve_type = spec.get('curve_type', 'standard_inverse')
            # IEEE curves are not implemented by pandapower OCRelay (IEC 60255 only).
            if curve_type in _IEEE_OC_CURVES:
                summaries.append({
                    'switch_id': sw_id, 'switch_name': spec.get('sw_name'),
                    'user_friendly_name': spec.get('user_friendly_name'), 'sw_idx': int(sw_idx),
                    'kind': 'OCR', 'subtype': subtype, 'curve_type': curve_type,
                    'pickup_mode': spec.get('pickup_mode'), 'engine': 'electrisim',
                    'attached': True, 'custom_evaluator': 'ieee_oc',
                    'not_computed': False, 'settings': dict(spec),
                })
                continue
            if curve_type not in _VALID_OC_CURVE_TYPES:
                curve_type = 'standard_inverse'
            time_settings = _prot_build_oc_relay_time_settings(
                spec, grading_mode, manual_times.get(subtype)
            )
            pickup_df = _prot_build_pickup_current_manual_df(spec)
            protection_index_before = _prot_protection_index(net)
            try:
                kwargs = dict(
                    switch_index=int(sw_idx),
                    oc_relay_type=subtype,
                    time_settings=time_settings,
                    curve_type=curve_type,
                )
                if spec.get('overload_factor') is not None:
                    kwargs['overload_factor'] = float(spec['overload_factor'])
                if spec.get('ct_current_factor') is not None:
                    kwargs['ct_current_factor'] = float(spec['ct_current_factor'])
                if spec.get('safety_factor') is not None:
                    kwargs['safety_factor'] = float(spec['safety_factor'])
                if pickup_df is not None:
                    # OCRelay reads manual values using ``iloc[switch_index]``.
                    # Supply every switch row (with this relay's settings) so a
                    # relay on a non-zero switch index is addressed correctly.
                    pickup_df = pd.concat([pickup_df] * len(net.switch), ignore_index=True)
                    pickup_df['switch_id'] = list(net.switch.index)
                    kwargs['pickup_current_manual'] = pickup_df
                device = OCRelay(net, **kwargs)
                _prot_apply_ocrelay_overrides(device, spec, grading_mode)
                summaries.append({
                    'switch_id': sw_id,
                    'switch_name': spec.get('sw_name'),
                    'user_friendly_name': spec.get('user_friendly_name'),
                    'sw_idx': int(sw_idx),
                    'kind': 'OCR',
                    'subtype': subtype,
                    'curve_type': curve_type,
                    'time_settings': _prot_to_jsonable(time_settings),
                    'pickup_mode': spec.get('pickup_mode'),
                    'engine': 'pandapower',
                    'attached': True,
                    'not_computed': False,
                })
            except Exception as e:
                import traceback as _tb
                tb_text = _tb.format_exc(limit=10)
                # OCRelay registered itself before it failed; unregister it so the
                # relay is reported once, by the ElectriSim evaluator below.
                _prot_drop_new_protection_rows(net, protection_index_before)
                if spec.get('pickup_mode') != 'manual':
                    # pandapower's OCRelay sets and grades relays on lines only:
                    # Electrisim sets this one by the same rules, extended to
                    # transformer breakers. (Evaluating the dialog's unset, zero,
                    # pickups tripped every relay instantly for every fault.)
                    if auto_depths is None:
                        auto_depths = _prot_auto_grading_depths(net, [
                            i for i in (_prot_resolve_sw_idx_for_id(net, k) for k, v in specs.items()
                                        if v.get('protection_type') == _PROTECTION_KIND_OCR)
                            if i is not None])
                    auto_spec, auto_note = _prot_auto_oc_settings(net, spec, int(sw_idx), auto_depths, grading_mode)
                    if auto_spec is not None:
                        summaries.append({
                            'switch_id': sw_id,
                            'switch_name': spec.get('sw_name'),
                            'user_friendly_name': spec.get('user_friendly_name'),
                            'sw_idx': int(sw_idx),
                            'kind': 'OCR', 'subtype': subtype, 'curve_type': curve_type,
                            'pickup_mode': spec.get('pickup_mode'),
                            'engine': 'electrisim',
                            'attached': True,
                            'custom_evaluator': 'oc_electrisim',
                            'not_computed': False,
                            'settings': auto_spec,
                            'reason': auto_note,
                            'traceback': tb_text,
                        })
                        continue
                    summaries.append({
                        'switch_id': sw_id,
                        'switch_name': spec.get('sw_name'),
                        'user_friendly_name': spec.get('user_friendly_name'),
                        'sw_idx': int(sw_idx),
                        'kind': 'OCR', 'subtype': subtype, 'curve_type': curve_type,
                        'pickup_mode': spec.get('pickup_mode'),
                        'attached': False, 'not_computed': True,
                        'reason': (
                            f'Automatic pickup could not be set: {auto_note}. '
                            "Set this relay's pickup currents by hand (pickup mode Manual) to evaluate it."
                        ),
                        'traceback': tb_text,
                    })
                    continue
                summaries.append({
                    'switch_id': sw_id,
                    'switch_name': spec.get('sw_name'),
                    'user_friendly_name': spec.get('user_friendly_name'),
                    'sw_idx': int(sw_idx),
                    'kind': 'OCR',
                    'subtype': subtype,
                    'curve_type': curve_type,
                    'pickup_mode': spec.get('pickup_mode'),
                    'engine': 'electrisim',
                    'attached': True,
                    'custom_evaluator': 'oc_electrisim',
                    'not_computed': False,
                    'settings': dict(spec),
                    # No 'reason': pandapower's OCRelay grades only networks of
                    # lines, and with pickups set by hand it is not needed. Its
                    # exception (kept in the traceback) filled the evaluation
                    # notes for every relay; the settings table already names
                    # the ElectriSim evaluator.
                    'traceback': tb_text,
                })

        # --- Fuse -------------------------------------------------------------------
        elif kind == _PROTECTION_KIND_FUSE:
            if Fuse is None:
                summaries.append({
                    'switch_id': sw_id, 'switch_name': spec.get('sw_name'),
                    'user_friendly_name': spec.get('user_friendly_name'),
                    'kind': 'Fuse', 'attached': False, 'not_computed': True,
                    'reason': 'pandapower.protection.Fuse not available in this pandapower version.',
                })
                continue
            protection_index_before = _prot_protection_index(net)
            try:
                kwargs = {'switch_index': int(sw_idx)}
                if spec.get('fuse_type'):
                    kwargs['fuse_type'] = str(spec['fuse_type'])
                if spec.get('rated_i_a') is not None:
                    kwargs['rated_i_a'] = float(spec['rated_i_a'])
                Fuse(net, **kwargs)
                summaries.append({
                    'switch_id': sw_id,
                    'switch_name': spec.get('sw_name'),
                    'user_friendly_name': spec.get('user_friendly_name'),
                    'sw_idx': int(sw_idx),
                    'kind': 'Fuse',
                    'fuse_type': spec.get('fuse_type'),
                    'rated_i_a': spec.get('rated_i_a'),
                    'engine': 'pandapower',
                    'attached': True,
                    'not_computed': False,
                })
            except Exception as e:
                _prot_drop_new_protection_rows(net, protection_index_before)
                summaries.append({
                    'switch_id': sw_id, 'switch_name': spec.get('sw_name'),
                    'user_friendly_name': spec.get('user_friendly_name'),
                    'kind': 'Fuse', 'attached': False, 'not_computed': False,
                    'reason': f'Fuse instantiation failed: {e}',
                })

        # --- Electrisim-side devices -----------------------------------------------
        elif kind in (_PROTECTION_KIND_EARTH_FAULT, _PROTECTION_KIND_DIRECTIONAL,
                      _PROTECTION_KIND_DIFF, _PROTECTION_KIND_DIST):
            summaries.append({
                'switch_id': sw_id,
                'switch_name': spec.get('sw_name'),
                'user_friendly_name': spec.get('user_friendly_name'),
                'sw_idx': int(sw_idx),
                'kind': {
                    _PROTECTION_KIND_EARTH_FAULT: 'Earth-fault OCR',
                    _PROTECTION_KIND_DIRECTIONAL: 'Directional OCR',
                    _PROTECTION_KIND_DIFF: 'Differential (87)',
                    _PROTECTION_KIND_DIST: 'Distance (21)',
                }[kind],
                'engine': 'electrisim',
                'attached': True,
                'custom_evaluator': kind,
                'not_computed': False,
                'settings': dict(spec),
            })
        else:
            summaries.append({
                'switch_id': sw_id, 'switch_name': spec.get('sw_name'),
                'user_friendly_name': spec.get('user_friendly_name'),
                'kind': kind, 'attached': False, 'not_computed': False,
                'reason': f'Unknown protection_type "{kind}".',
            })

    return summaries


def _prot_sample_oc_relay_characteristic(device, x_min=10.0, x_max=1.0e5, n_points=80):
    """
    Sample an OCRelay (DTOC / IDMT / IDTOC) time-current characteristic into
    (I [A], t [s]) arrays. Mirrors pandapower's plot_protection_characteristic
    but writes raw arrays instead of drawing a matplotlib figure.

    OCRelay stores currents in kA, so we convert to A for the UI.
    """
    try:
        subtype = getattr(device, 'oc_relay_type', None)
        if subtype is None:
            return [], []
        x = list(np.logspace(math.log10(max(x_min, 1.0)), math.log10(max(x_max, x_min * 10.0)), int(n_points)))
        I_g = getattr(device, 'I_g', None)
        I_gg = getattr(device, 'I_gg', None)
        I_s = getattr(device, 'I_s', None)
        for pickup_ka in (I_s, I_g, I_gg):
            if pickup_ka is None:
                continue
            try:
                p_a = float(pickup_ka) * 1000.0
            except (TypeError, ValueError):
                continue
            if p_a > 0:
                x.extend([p_a * 0.999, p_a, p_a * 1.001])
        x = sorted(set(float(v) for v in x if v > 0))
        t_g = getattr(device, 't_g', None)
        t_gg = getattr(device, 't_gg', None)
        t_grade = getattr(device, 't_grade', None)
        tms = getattr(device, 'tms', None)
        k = getattr(device, 'k', None)
        alpha = getattr(device, 'alpha', None)
        if k is None or alpha is None:
            k, alpha = _prot_iec_k_alpha(getattr(device, 'curve_type', None))

        currents = []
        times = []
        for i_a in x:
            i_ka = float(i_a) / 1000.0
            t = float('inf')
            if subtype == 'DTOC':
                if I_gg is not None and i_ka >= I_gg:
                    t = float(t_gg) if t_gg is not None else float('inf')
                elif I_g is not None and i_ka >= I_g:
                    t = float(t_g) if t_g is not None else float('inf')
            elif subtype == 'IDMT':
                if I_s is not None and i_ka > I_s and tms is not None and k is not None and alpha is not None:
                    denom = ((i_ka / float(I_s)) ** float(alpha)) - 1.0
                    if denom > 0:
                        t = (float(tms) * float(k)) / denom + (float(t_grade) if t_grade is not None else 0.0)
            elif subtype == 'IDTOC':
                if I_gg is not None and i_ka >= I_gg:
                    t = float(t_gg) if t_gg is not None else float('inf')
                elif I_g is not None and i_ka >= I_g:
                    t = float(t_g) if t_g is not None else float('inf')
                elif I_s is not None and i_ka > I_s and tms is not None and k is not None and alpha is not None:
                    denom = ((i_ka / float(I_s)) ** float(alpha)) - 1.0
                    if denom > 0:
                        t = (float(tms) * float(k)) / denom + (float(t_grade) if t_grade is not None else 0.0)
            currents.append(float(i_a))
            times.append(t if math.isfinite(t) else None)
        return currents, times
    except Exception as e:
        print(f"Protection: failed to sample OCRelay characteristic - {e}")
        return [], []


def fuse_characteristic_preview(row):
    """
    Sample a fuse I–t melting curve for the Switch dialog (library or custom std_types).
    Mirrors what ``Fuse.plot_protection_characteristic(net)`` visualizes, as JSON for Chart.js.

    Expected keys on ``row`` (same request object shape as other *PandaPower types):
    fuse_mode, fuse_type, rated_i_a, fuse_custom_std_json (optional).
    """
    try:
        from pandapower.protection.protection_devices.fuse import Fuse
    except ImportError as e:
        return json.dumps({
            'error': True,
            'message': f'pandapower Fuse unavailable: {e}',
            'i_a': [], 't_s': [],
        }, separators=(',', ':'))

    fuse_mode = str(row.get('fuse_mode', 'library') or 'library').strip().lower()
    fuse_type = str(row.get('fuse_type') or '').strip()
    rated = _prot_safe_float(row.get('rated_i_a'))

    if fuse_mode != 'custom' and not fuse_type:
        return json.dumps({
            'error': True, 'message': 'No fuse type selected.',
            'i_a': [], 't_s': [],
        }, separators=(',', ':'))
    if fuse_mode == 'custom' and not fuse_type:
        return json.dumps({
            'error': True, 'message': 'Enter a custom fuse name.',
            'i_a': [], 't_s': [],
        }, separators=(',', ':'))
    if fuse_mode == 'custom' and (rated is None or float(rated) <= 0):
        return json.dumps({
            'error': True, 'message': 'Rated current must be positive for a custom fuse preview.',
            'i_a': [], 't_s': [],
        }, separators=(',', ':'))

    try:
        net = pp.create_empty_network()
        pp.create_bus(net, vn_kv=20.0, name='fuse_preview_a')
        pp.create_bus(net, vn_kv=20.0, name='fuse_preview_b')
        sw_idx = pp.create_switch(net, bus=0, element=1, et='b', closed=True,
                                  name='fuse_preview_sw', type='CB')
        _prot_ensure_bus_geodata(net)

        if fuse_mode == 'custom':
            spec = {
                'protection_type': _PROTECTION_KIND_FUSE,
                'fuse_type': fuse_type,
                'fuse_mode': 'custom',
                'rated_i_a': rated,
                'fuse_custom_std_json': row.get('fuse_custom_std_json'),
            }
            reg_errs = _prot_register_custom_fuse_std_types(net, {'preview': spec})
            if reg_errs:
                return json.dumps({
                    'error': True,
                    'message': str(reg_errs[0].get('reason', 'Custom fuse registration failed')),
                    'i_a': [], 't_s': [],
                }, separators=(',', ':'))

        kwargs = {'switch_index': int(sw_idx)}
        if fuse_type:
            kwargs['fuse_type'] = fuse_type
        if rated is not None:
            kwargs['rated_i_a'] = float(rated)
        device = Fuse(net, **kwargs)
        i_a, t_s = _prot_sample_fuse_characteristic(device, net, n_points=120)

        out_i, out_t = [], []
        for a, t in zip(i_a, t_s):
            if t is None:
                continue
            try:
                tf = float(t)
                af = float(a)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(tf) or tf <= 0 or not math.isfinite(af) or af <= 0:
                continue
            out_i.append(af)
            out_t.append(tf)

        axis_payload = {}
        try:
            char_obj = net.characteristic.at[int(device.characteristic_index), 'object']
            xv = np.asarray(char_obj.x_vals, dtype=float)
            yv = np.asarray(char_obj.y_vals, dtype=float)
            if xv.size and yv.size:
                t_hi = max(float(np.max(out_t)) if out_t else 0.0, float(10 ** np.ceil(np.max(yv))))
                axis_payload = {
                    'i_a_min': float(10 ** np.floor(np.min(xv))),
                    'i_a_max': float(10 ** np.ceil(np.max(xv))),
                    't_s_min': float(10 ** np.floor(np.min(yv))),
                    't_s_max': float(max(t_hi * 1.05, 10 ** np.ceil(np.max(yv)))),
                }
        except Exception:
            pass

        return json.dumps({
            'error': False,
            'fuse_type': fuse_type,
            'fuse_mode': fuse_mode,
            'i_a': out_i,
            't_s': out_t,
            'axis': axis_payload,
        }, separators=(',', ':'))
    except Exception as e:
        import traceback
        traceback.print_exc(limit=8)
        return json.dumps({
            'error': True,
            'message': str(e),
            'i_a': [], 't_s': [],
            'axis': {},
        }, separators=(',', ':'))


def _prot_sample_fuse_characteristic(device, net, n_points=80):
    """Sample the fuse melting curve from net.characteristic[char_idx]."""
    try:
        char_idx = getattr(device, 'characteristic_index', None)
        if char_idx is None or not hasattr(net, 'characteristic'):
            return [], []
        char_obj = net.characteristic.at[char_idx, 'object']
        x_vals_arr = getattr(char_obj, 'x_vals', None)
        if x_vals_arr is None:
            return [], []
        x_vals = list(np.asarray(x_vals_arr).flatten())
        if len(x_vals) == 0:
            return [], []
        # pandapower's LogSplineCharacteristic stores log10 values in x_vals.
        x_min_log = float(min(x_vals))
        x_max_log = float(max(x_vals))
        x_logspaced = np.logspace(x_min_log, x_max_log, int(n_points))
        try:
            y_logspaced = np.asarray(char_obj(x_logspaced)).flatten()
        except Exception:
            y_logspaced = np.asarray([char_obj(float(v)) for v in x_logspaced]).flatten()
        currents = [float(v) for v in np.asarray(x_logspaced).flatten()]
        times = []
        for v in y_logspaced:
            try:
                fv = float(v)
                times.append(fv if math.isfinite(fv) and fv > 0 else None)
            except (TypeError, ValueError):
                times.append(None)
        pairs = [(c, t) for c, t in zip(currents, times) if t is not None]
        pairs.sort(key=lambda p: p[0])
        return [p[0] for p in pairs], [p[1] for p in pairs]
    except Exception as e:
        print(f"Protection: failed to sample Fuse characteristic - {e}")
        return [], []


def _prot_device_settings_for_ui(device, attach_summary):
    """
    Extract user-visible settings from a pandapower protection device object.

    OCRelay stores pickup currents (I_g, I_gg, I_s) in kA; we expose those plus
    Ampere-scaled aliases (I_g_a, I_gg_a, I_s_a) so the UI can pick the unit it
    wants without re-deriving conversions.
    """
    out = {}
    for attr in ('oc_relay_type', 'curve_type', 'tms', 't_grade', 't_gg', 't_g', 't_diff',
                 'pickup_current', 'I_s', 'I_g', 'I_gg',
                 'rated_i_a', 'fuse_type', 'overload_factor', 'ct_current_factor', 'safety_factor'):
        v = getattr(device, attr, None)
        if v is None:
            continue
        if isinstance(v, (int, float, np.integer, np.floating)):
            fv = float(v)
            if not math.isnan(fv) and math.isfinite(fv):
                out[attr] = fv
        elif isinstance(v, str):
            out[attr] = v

    # Add Ampere-scaled aliases (OCRelay internal currents are kA).
    if 'I_g' in out and 'I_g_a' not in out:
        out['I_g_a'] = out['I_g'] * 1000.0
    if 'I_gg' in out and 'I_gg_a' not in out:
        out['I_gg_a'] = out['I_gg'] * 1000.0
    if 'I_s' in out and 'I_s_a' not in out:
        out['I_s_a'] = out['I_s'] * 1000.0
    return out


def _prot_extract_devices_for_ui(net, attach_summaries):
    """
    Iterate pandapower's net.protection table and produce per-device payloads:
    settings + sampled I-t characteristic. attach_summaries provides the
    mapping back to the frontend switch id.
    """
    devices = []
    if not hasattr(net, 'protection'):
        return devices
    prot_df = getattr(net, 'protection', None)
    if prot_df is None or len(prot_df) == 0:
        return devices

    summary_by_sw_idx = {}
    for s in attach_summaries:
        sw_idx = s.get('sw_idx')
        if sw_idx is not None:
            summary_by_sw_idx[int(sw_idx)] = s

    for idx, prow in prot_df.iterrows():
        device = prow.get('object') if 'object' in prow.index else None
        if device is None:
            continue
        sw_idx = getattr(device, 'switch_index', None)
        if sw_idx is None and 'switch_index' in prow.index:
            sw_idx = prow['switch_index']
        sw_idx = int(sw_idx) if sw_idx is not None and not (isinstance(sw_idx, float) and math.isnan(sw_idx)) else None
        summary = summary_by_sw_idx.get(sw_idx, {}) if sw_idx is not None else {}
        if summary.get('custom_evaluator'):
            # ElectriSim evaluates this switch (see _prot_append_custom_devices),
            # so a pandapower object left on it must not be reported as well.
            continue

        device_class = type(device).__name__  # 'OCRelay' / 'Fuse' / ...
        kind = summary.get('kind') or device_class
        if device_class == 'OCRelay':
            subtype = summary.get('subtype') or getattr(device, 'oc_relay_type', None)
            curve_type = summary.get('curve_type') or getattr(device, 'curve_type', None)
        elif device_class == 'Fuse':
            subtype = getattr(device, 'fuse_type', None) or summary.get('fuse_type')
            curve_type = 'fuse_melting_curve'
        else:
            subtype = summary.get('subtype')
            curve_type = summary.get('curve_type')

        settings = _prot_device_settings_for_ui(device, summary)

        if device_class == 'OCRelay':
            i_a, t_s = _prot_sample_oc_relay_characteristic(device)
        elif device_class == 'Fuse':
            # Dense sampling helps Chart.js draw smooth fuse melting curves on log-log axes.
            i_a, t_s = _prot_sample_fuse_characteristic(device, net, n_points=180)
        else:
            i_a, t_s = [], []

        devices.append({
            'protection_idx': int(idx),
            'switch_idx': sw_idx,
            'switch_id': summary.get('switch_id'),
            'switch_name': summary.get('switch_name'),
            'user_friendly_name': summary.get('user_friendly_name'),
            'type': kind,
            'subtype': subtype,
            'curve_type': curve_type,
            'engine': 'pandapower',
            'settings': settings,
            'characteristic': {'i_a': i_a, 't_s': t_s},
        })

    return devices


def _prot_fault_bus_label(net, fault_bus_idx):
    """Best-effort label for the fault bus (id, name, or index)."""
    try:
        if fault_bus_idx in net.bus.index:
            row = net.bus.loc[fault_bus_idx]
            label_id = row['id'] if 'id' in row.index and pd.notna(row['id']) else None
            label_name = row['name'] if 'name' in row.index and pd.notna(row['name']) else None
            return str(label_id or label_name or fault_bus_idx)
    except Exception:
        pass
    return str(fault_bus_idx)


def _prot_fault_label(net, scenario):
    """'L1, 50 %' for a line fault, the bus name for a busbar fault."""
    try:
        if scenario.get('sc_line_id') is not None and scenario.get('fault_location_mode') != 'bus':
            idx = int(scenario['sc_line_id'])
            name = _contingency_friendly_name(net, net.line.at[idx, 'name']) if idx in net.line.index else f'line {idx}'
            return f"{name}, {100 * float(scenario.get('sc_fraction') or 0):.0f} %"
        idx = int(scenario['fault_bus_idx'])
        return _contingency_friendly_name(net, net.bus.at[idx, 'name']) if idx in net.bus.index else str(idx)
    except (KeyError, TypeError, ValueError):
        return scenario.get('fault_bus')


def _prot_clean_scalar(v):
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            return None
        return fv
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    return str(v)


def _prot_parse_prot_results(prot_results, summary_by_sw_idx):
    """Turn pandapower calculate_protection_times dataframe into UI trip rows."""
    trip_rows = []
    if prot_results is None or getattr(prot_results, 'empty', True):
        return trip_rows

    for _, row in prot_results.iterrows():
        sw_idx_val = row.get('switch_id') if 'switch_id' in row.index else None
        try:
            sw_idx_int = int(sw_idx_val) if sw_idx_val is not None else None
        except (TypeError, ValueError):
            sw_idx_int = None
        summary = summary_by_sw_idx.get(sw_idx_int, {}) if sw_idx_int is not None else {}

        t_trip = None
        trip_melt_time_s = None
        for col in ('trip_melt_time_s', 'trip_time', 'trip time [s]', 'trip_time_s'):
            if col in row.index:
                val = row[col]
                if col == 'trip_melt_time_s':
                    trip_melt_time_s = val
                if t_trip is None:
                    t_trip = val

        activation_parameter_value = None
        if 'activation_parameter_value' in row.index:
            activation_parameter_value = row['activation_parameter_value']

        ikss = None
        for col in ('activation_parameter_value', 'ikss_ka', 'ikss'):
            if col in row.index:
                ikss = row[col]
                break

        tripped = None
        for col in ('trip_melt', 'tripped'):
            if col in row.index:
                tripped = row[col]
                break
        if tripped is None and t_trip is not None:
            try:
                tripped = math.isfinite(float(t_trip))
            except (TypeError, ValueError):
                tripped = False

        device_kind = summary.get('kind') or row.get('protection_type', 'Unknown')
        device_subtype = summary.get('subtype') or summary.get('fuse_type')
        device_label = f"{device_kind}-{device_subtype}".strip('-') if device_subtype else str(device_kind)
        is_fuse = str(device_kind).lower() == 'fuse' or str(row.get('protection_type', '')).lower() == 'fuse'
        t_trip_clean = _prot_clean_scalar(t_trip)
        trip_melt_clean = _prot_clean_scalar(trip_melt_time_s if trip_melt_time_s is not None else t_trip)
        act_param_clean = _prot_clean_scalar(
            activation_parameter_value if activation_parameter_value is not None else ikss
        )

        trip_rows.append({
            'switch_idx': sw_idx_int,
            'switch_id': summary.get('switch_id'),
            'switch_name': summary.get('switch_name'),
            'user_friendly_name': summary.get('user_friendly_name'),
            'device': device_label,
            'device_kind': str(device_kind),
            'is_fuse': is_fuse,
            'tripped': bool(_prot_clean_scalar(tripped)) if tripped is not None else False,
            't_trip_s': t_trip_clean,
            't_melt_s': trip_melt_clean if is_fuse else None,
            'trip_melt_time_s': trip_melt_clean,
            'ikss_ka': _prot_clean_scalar(ikss),
            'activation_parameter_value': act_param_clean,
        })
    return trip_rows


def _prot_extract_short_circuit_at_bus(net_sc, bus_idx):
    """Read Ikss / Ip / Ith at the fault bus from res_bus_sc after calc_sc."""
    out = {
        'bus_idx': int(bus_idx),
        'ikss_ka': None,
        'ip_ka': None,
        'ith_ka': None,
        'skss_mva': None,
    }
    res = getattr(net_sc, 'res_bus_sc', None)
    if res is None or getattr(res, 'empty', True) or bus_idx not in res.index:
        return out
    row = res.loc[bus_idx]
    for key, col in (
        ('ikss_ka', 'ikss_ka'),
        ('ip_ka', 'ip_ka'),
        ('ith_ka', 'ith_ka'),
        ('skss_mva', 'skss_mva'),
    ):
        if col in row.index:
            out[key] = _prot_clean_scalar(row[col])
    return out


def _prot_switch_current_ka(net_sc, sw_idx, fault_bus_idx=None):
    """Best available branch current at a switch for Electrisim-side relays."""
    try:
        sw = net_sc.switch.loc[int(sw_idx)]
        et, element = str(sw.get('et')), int(sw.get('element'))
        table = {'l': 'res_line_sc', 't': 'res_trafo_sc', 't3': 'res_trafo3w_sc'}.get(et)
        if table and hasattr(net_sc, table):
            result = getattr(net_sc, table)
            if element in result.index:
                row = result.loc[element]
                # The current on the switch's own side: a transformer's HV
                # current is its LV current scaled by the ratio, so taking
                # whichever column came first made an LV-side breaker see
                # 20/110 of its fault current.
                sides = {'l': (('from_bus', 'ikss_from_ka'), ('to_bus', 'ikss_to_ka')),
                         't': (('hv_bus', 'ikss_hv_ka'), ('lv_bus', 'ikss_lv_ka')),
                         't3': (('hv_bus', 'ikss_hv_ka'), ('mv_bus', 'ikss_mv_ka'),
                                ('lv_bus', 'ikss_lv_ka'))}[et]
                elements = {'l': net_sc.line, 't': net_sc.trafo, 't3': net_sc.trafo3w}[et]
                own = [col for bus_col, col in sides
                       if element in elements.index
                       and int(elements.at[element, bus_col]) == int(sw.get('bus'))]
                for col in own + ['ikss_ka'] + [col for _, col in sides]:
                    if col in row.index and pd.notna(row[col]):
                        return abs(float(row[col]))
    except Exception:
        pass
    if fault_bus_idx is not None:
        return _prot_extract_short_circuit_at_bus(net_sc, fault_bus_idx).get('ikss_ka')
    return None


def _prot_custom_trip_rows(net_sc, attach_summaries, fault_type, fault_bus_idx=None):
    """Evaluate IEEE, earth-fault, directional, 87 and 21 devices from SC results."""
    rows = []
    for summary in attach_summaries:
        evaluator = summary.get('custom_evaluator')
        if not evaluator:
            continue
        spec = summary.get('settings', {})
        current_ka = _prot_switch_current_ka(net_sc, summary.get('sw_idx'), fault_bus_idx)
        current_a = (float(current_ka) * 1000.0) if current_ka is not None else None
        tripped, t_trip, detail = False, None, {}

        if evaluator in ('ieee_oc', 'oc_electrisim'):
            subtype = summary.get('subtype') or spec.get('oc_relay_type') or 'DTOC'
            tripped, t_trip, detail = _prot_eval_oc_trip(spec, subtype, current_a, evaluator)
            pickup = spec.get('I_s_a') or spec.get('I_g_a')
            if pickup:
                detail = dict(detail or {})
                detail['pickup_a'] = pickup
                if current_a and pickup:
                    detail['multiple'] = current_a / pickup
        elif evaluator == _PROTECTION_KIND_EARTH_FAULT:
            pickup = spec.get('I_e_a') or 1.0
            # For a 1ph SC, Ikss is the fault-loop current and is the available
            # residual-current estimate. Other fault types have no residual trip.
            residual_a = current_a if fault_type == '1ph' else 0.0
            tripped = residual_a >= pickup
            t_trip = spec.get('t_e', 0.2) if tripped else None
            detail = {'residual_current_a': residual_a, 'pickup_a': pickup}
        elif evaluator == _PROTECTION_KIND_DIRECTIONAL:
            # SC branch results contain magnitude but no reliable phasor angle in
            # every pandapower version. Relay orientation is therefore the switch
            # bus → protected element direction; it is reported with every trip.
            pickup = spec.get('I_g_a') or spec.get('I_s_a') or 1.0
            forward = True
            requested = spec.get('directional_mode', 'forward')
            tripped = current_a is not None and current_a >= pickup and (requested == 'forward') == forward
            t_trip = spec.get('t_g', 0.5) if tripped else None
            detail = {'orientation': 'switch bus -> protected element', 'direction': 'forward',
                      'pickup_a': pickup}
        elif evaluator == _PROTECTION_KIND_DIFF:
            pickup = spec.get('I_diff_a') or 1.0
            slope = spec.get('diff_slope') or 0.0
            # A switch defines the protected boundary. Branch fault current is the
            # operating current; half is a conservative restraint-current proxy.
            operate = current_a or 0.0
            restraint = operate / 2.0
            tripped = operate >= pickup + slope * restraint
            t_trip = spec.get('t_g', 0.03) if tripped else None
            detail = {'i_operate_a': operate, 'i_restraint_a': restraint,
                      'pickup_a': pickup, 'slope': slope}
        elif evaluator == _PROTECTION_KIND_DIST:
            z_base = None
            try:
                sw = net_sc.switch.loc[int(summary.get('sw_idx'))]
                vn_kv = float(net_sc.bus.at[int(sw['bus']), 'vn_kv'])
                z_base = (vn_kv / math.sqrt(3.0)) / current_ka if current_ka and current_ka > 0 else None
            except Exception:
                pass
            zones = [
                (1, spec.get('z1_r_ohm'), spec.get('z1_x_ohm'), spec.get('t_z1', 0.0)),
                (2, spec.get('z2_r_ohm'), spec.get('z2_x_ohm'), spec.get('t_z2', 0.3)),
                (3, spec.get('z3_r_ohm'), spec.get('z3_x_ohm'), spec.get('t_z3', 0.6)),
            ]
            reached = next(((n, r, x, delay) for n, r, x, delay in zones
                            if z_base is not None and r is not None and x is not None
                            and z_base <= math.hypot(r, x)), None)
            tripped = reached is not None
            t_trip = reached[3] if reached else None
            detail = {'z_apparent_ohm': z_base, 'zone': reached[0] if reached else None}

        rows.append({
            'switch_idx': summary.get('sw_idx'), 'switch_id': summary.get('switch_id'),
            'switch_name': summary.get('switch_name'),
            'user_friendly_name': summary.get('user_friendly_name'),
            'device': summary.get('kind'), 'device_kind': summary.get('kind'),
            'is_fuse': False, 'tripped': bool(tripped), 't_trip_s': _prot_clean_scalar(t_trip),
            't_melt_s': None, 'trip_melt_time_s': None, 'ikss_ka': _prot_clean_scalar(current_ka),
            'activation_parameter_value': _prot_clean_scalar(current_ka),
            'evaluation': detail,
        })
    return rows


def _prot_append_custom_devices(devices, attach_summaries):
    """Add the ElectriSim-evaluated devices, one entry per switch."""
    already_reported = {d.get('switch_idx') for d in devices if d.get('switch_idx') is not None}
    for summary in attach_summaries:
        if not summary.get('custom_evaluator'):
            continue
        if summary.get('sw_idx') in already_reported:
            continue
        spec = summary.get('settings', {}) or {}
        evaluator = summary.get('custom_evaluator')
        subtype = summary.get('subtype') or spec.get('oc_relay_type')
        curve_type = summary.get('curve_type') or spec.get('curve_type')
        i_a, t_s = [], []
        if evaluator in ('ieee_oc', 'oc_electrisim') or summary.get('kind') == 'OCR':
            i_a, t_s = _prot_sample_spec_oc_characteristic(spec, subtype, curve_type)
        devices.append({
            'switch_idx': summary.get('sw_idx'), 'switch_id': summary.get('switch_id'),
            'switch_name': summary.get('switch_name'),
            'user_friendly_name': summary.get('user_friendly_name'),
            'type': summary.get('kind'), 'subtype': subtype,
            'curve_type': curve_type, 'engine': 'electrisim',
            'settings': _prot_spec_settings_for_ui(spec, subtype),
            'characteristic': {'i_a': i_a, 't_s': t_s},
        })
        already_reported.add(summary.get('sw_idx'))
    return devices


def _prot_resolve_fault_bus_idx(in_data, net, fault_bus_cell_id):
    """Map frontend bus cell id (diagram id) to pandapower bus index."""
    if fault_bus_cell_id in (None, ''):
        return None
    target = str(fault_bus_cell_id).strip()
    bus_name = None
    user_friendly = None
    for x in in_data:
        row = in_data.get(x)
        if not isinstance(row, dict):
            continue
        typ = str(row.get('typ', ''))
        if 'Bus' not in typ or 'DC Bus' in typ:
            continue
        rid = str(row.get('id', '')).strip()
        rname = str(row.get('name', '')).strip()
        if rid == target or rname == target:
            bus_name = row.get('name')
            user_friendly = row.get('userFriendlyName')
            break
    if bus_name is None and 'name' in net.bus.columns:
        for idx in net.bus.index:
            if str(net.bus.at[idx, 'name']).strip() == target:
                return int(idx)
    if bus_name is None:
        try:
            idx = int(target)
            if idx in net.bus.index:
                return int(idx)
        except (TypeError, ValueError):
            return None
        return None
    if 'name' in net.bus.columns:
        for candidate in (bus_name, user_friendly):
            if candidate in (None, ''):
                continue
            matches = net.bus.index[net.bus['name'].astype(str) == str(candidate)].tolist()
            if matches:
                return int(matches[0])
    return None


def _prot_native_and_custom_trips(net_sc, attach_summaries, summary_by_sw_idx, fault_type, fault_bus_idx):
    """Trip rows from pandapower (when net.protection exists) plus ElectriSim devices.

    Rows are keyed by switch so one relay can never appear twice in the table.
    """
    warning = None
    trip = []
    if _prot_has_native_protection(net_sc):
        try:
            from pandapower.protection.run_protection import calculate_protection_times
            prot_results = calculate_protection_times(net_sc, scenario='sc')
            trip.extend(_prot_parse_prot_results(prot_results, summary_by_sw_idx))
        except Exception as e:
            warning = f'calculate_protection_times failed: {e}'
    reported = {row.get('switch_idx') for row in trip if row.get('switch_idx') is not None}
    for row in _prot_custom_trip_rows(net_sc, attach_summaries, fault_type, fault_bus_idx):
        if row.get('switch_idx') in reported:
            continue
        trip.append(row)
        reported.add(row.get('switch_idx'))
    return trip, warning


def _prot_run_bus_scenario(base_net, fault_bus_idx, fault_type, case, attach_summaries):
    """Short-circuit and protection times for a fault placed directly on a bus."""
    summary_by_sw_idx = {int(s['sw_idx']): s for s in attach_summaries if s.get('sw_idx') is not None}
    net_sc = deepcopy(base_net)
    try:
        ensure_ext_grid_zero_sequence_min(net_sc)
        sc.calc_sc(net_sc, bus=int(fault_bus_idx), branch_results=True, fault=fault_type, case=case)
    except Exception as e:
        return {
            'fault_location_mode': 'bus',
            'fault_bus_idx': int(fault_bus_idx),
            'error': f'calc_sc failed: {e}',
            'fault_bus': _prot_fault_bus_label(net_sc, fault_bus_idx),
            'trip': [],
            'short_circuit': {},
        }

    sc_info = _prot_extract_short_circuit_at_bus(net_sc, fault_bus_idx)
    trip, prot_warning = _prot_native_and_custom_trips(
        net_sc, attach_summaries, summary_by_sw_idx, fault_type, int(fault_bus_idx)
    )
    out = {
        'fault_location_mode': 'bus',
        'fault_bus_idx': int(fault_bus_idx),
        'sc_line_id': None,
        'sc_fraction': None,
        'fault_bus': _prot_fault_bus_label(net_sc, fault_bus_idx),
        'fault_type': fault_type,
        'case': case,
        'short_circuit': sc_info,
        'trip': trip,
    }
    if prot_warning:
        out['warning'] = prot_warning
    return out


def _prot_run_scenario(base_net, sc_line_id, sc_fraction, fault_type, case, attach_summaries):
    """
    Run a single fault scenario on a copy of the network and return:
    - per-device trip rows (switch_id, tripped, t_trip_s, ikss_ka)
    - the fault bus label
    Uses pandapower.protection.utility_functions.create_sc_bus + sc.calc_sc +
    calculate_protection_times.
    """
    try:
        from pandapower.protection.utility_functions import create_sc_bus
    except ImportError as e:
        return {
            'sc_line_id': int(sc_line_id),
            'sc_fraction': float(sc_fraction),
            'error': f'pandapower.protection is unavailable: {e}',
            'fault_bus': None,
            'trip': [],
        }

    # Map sw_idx -> frontend summary so the trip rows show user-friendly labels.
    summary_by_sw_idx = {int(s['sw_idx']): s for s in attach_summaries if s.get('sw_idx') is not None}

    try:
        net_sc = create_sc_bus(deepcopy(base_net), sc_line_id=int(sc_line_id), sc_fraction=float(sc_fraction))
        # pandapower (3.3) moves every switch whose element number is the
        # line's onto the new line half - transformer and bus switches too, so
        # a transformer switch then named a transformer that does not exist.
        not_line = base_net.switch.index[base_net.switch['et'] != 'l']
        net_sc.switch.loc[not_line, 'element'] = base_net.switch.loc[not_line, 'element']
    except Exception as e:
        return {
            'sc_line_id': int(sc_line_id),
            'sc_fraction': float(sc_fraction),
            'error': f'create_sc_bus failed: {e}',
            'fault_bus': None,
            'trip': [],
        }

    try:
        fault_bus = int(max(net_sc.bus.index))
        ensure_ext_grid_zero_sequence_min(net_sc)
        sc.calc_sc(net_sc, bus=fault_bus, branch_results=True, fault=fault_type, case=case)
    except Exception as e:
        return {
            'sc_line_id': int(sc_line_id),
            'sc_fraction': float(sc_fraction),
            'error': f'calc_sc failed: {e}',
            'fault_bus': None,
            'trip': [],
        }

    trip, prot_warning = _prot_native_and_custom_trips(
        net_sc, attach_summaries, summary_by_sw_idx, fault_type, fault_bus
    )

    out = {
        'fault_location_mode': 'line',
        'sc_line_id': int(sc_line_id),
        'sc_fraction': float(sc_fraction),
        'fault_bus': _prot_fault_bus_label(net_sc, fault_bus),
        'fault_bus_idx': int(fault_bus),
        'fault_type': fault_type,
        'case': case,
        'short_circuit': _prot_extract_short_circuit_at_bus(net_sc, fault_bus),
        'trip': trip,
    }
    if prot_warning:
        out['warning'] = prot_warning
    return out


def _prot_element_graph(net, fault_line_id=None):
    """
    Buses and branch elements as graph nodes. A branch is a node of its own,
    joined to each of its buses, so a relay - a switch at one bus of one
    element - is the edge between the two. A faulted line becomes the fault
    node 'F'. Open switches cut their edge.
    """
    adjacency = {}

    def connect(a, b):
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    for b in net.bus.index:
        adjacency.setdefault(('bus', int(b)), set())
    tables = (('line', 'l', ('from_bus', 'to_bus')), ('trafo', 't', ('hv_bus', 'lv_bus')),
              ('trafo3w', 't3', ('hv_bus', 'mv_bus', 'lv_bus')), ('impedance', 'i', ('from_bus', 'to_bus')))
    for table, et, cols in tables:
        data = getattr(net, table, None)
        if data is None or data.empty:
            continue
        for idx, row in data.iterrows():
            if not bool(row.get('in_service', True)):
                continue
            node = 'F' if (et == 'l' and fault_line_id is not None and int(idx) == int(fault_line_id)) else (et, int(idx))
            for col in cols:
                connect(('bus', int(row[col])), node)
    if fault_line_id is not None:
        adjacency.setdefault('F', set())

    def element_node(et, element):
        if et == 'l' and fault_line_id is not None and int(element) == int(fault_line_id):
            return 'F'
        return ('bus', int(element)) if et == 'b' else (et, int(element))

    for _, sw in net.switch.iterrows():
        a, b = ('bus', int(sw['bus'])), element_node(sw['et'], sw['element'])
        if sw['et'] == 'b':
            if bool(sw['closed']):
                connect(a, b)
        elif not bool(sw['closed']) and b in adjacency.get(a, ()):
            adjacency[a].discard(b)
            adjacency[b].discard(a)
    return adjacency, element_node


def _prot_zone(adjacency, starts, relay_at_edge, blocked=()):
    """Nodes reachable from starts without crossing a relay, and the relays met (relay -> far node)."""
    seen = set(starts) | set(blocked)
    zone, boundary, queue = set(starts), {}, list(starts)
    for node in queue:
        for nxt in adjacency.get(node, ()):
            relay = relay_at_edge.get(frozenset((node, nxt)))
            if relay is not None:
                boundary.setdefault(relay, nxt)
                continue
            if nxt not in seen:
                seen.add(nxt)
                zone.add(nxt)
                queue.append(nxt)
    return zone, boundary


def _prot_reaches_source(adjacency, start, sources, blocked):
    seen, queue = set(blocked) | {start}, [start]
    for node in queue:
        if node in sources:
            return True
        for nxt in adjacency.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def _prot_check_miscoordination(net, scenarios, t_diff):
    """
    Grade the relays of each fault by protection zone. The zone is what the
    fault reaches without passing a relay; the relays on its edge that lead
    to a source must clear it (primaries). Behind each primary, the next
    relays towards a source are its backups and must be t_diff slower. Any
    other relay that trips before the fault is cleared (plus t_diff) trips
    needlessly - on a meshed network, fault current flows round the healthy
    side too.

    Ordering relays by their hop distance from the external grid paired
    unrelated relays: a three-winding transformer was not a path, so the
    parallel transformer feeds were graded as primary and backup.

    Returns (miscoordination, unwanted_trips); each scenario gets
    primary_switches, clearing_time_s and unprotected_sources.
    """
    miscoord, unwanted = [], []
    sources = set()
    for table in ('ext_grid', 'gen'):
        data = getattr(net, table, None)
        if data is None or data.empty:
            continue
        for _, row in data.iterrows():
            if bool(row.get('in_service', True)):
                sources.add(('bus', int(row['bus'])))
    source_names = {}
    for table in ('gen', 'ext_grid'):
        data = getattr(net, table, None)
        if data is not None and not data.empty:
            for idx, row in data.iterrows():
                source_names.setdefault(('bus', int(row['bus'])), []).append(
                    _contingency_friendly_name(net, row.get('name') if pd.notna(row.get('name')) else f'{table} {idx}'))
    # Inverter-based sources (static generators, wind turbines, storage) feed a
    # fault too - about their rated current - until their own protection
    # disconnects them. Too little to operate an overcurrent relay, they do not
    # make a relay a primary, but a fault zone holding one is not cleared of
    # it by the breakers: the wind farm inside the transmission ring's L1-L4
    # zone, or behind the radial grid's wind feeder breaker, went unmentioned.
    inverter_names = {}
    for table in ('sgen', 'storage'):
        data = getattr(net, table, None)
        if data is None or data.empty:
            continue
        for idx, row in data.iterrows():
            if bool(row.get('in_service', True)):
                inverter_names.setdefault(('bus', int(row['bus'])), []).append(
                    _contingency_friendly_name(net, row.get('name') if pd.notna(row.get('name')) else f'{table} {idx}'))

    for scenario in scenarios:
        if scenario.get('error'):
            continue
        line_id = scenario.get('sc_line_id')
        if scenario.get('fault_location_mode') == 'bus' or line_id is None:
            fault_line = None
            fault_node = ('bus', int(scenario['fault_bus_idx']))
        else:
            fault_line = int(line_id)
            fault_node = 'F'
        adjacency, element_node = _prot_element_graph(net, fault_line)
        rows = {}
        relay_at_edge = {}
        for row in scenario.get('trip', []):
            try:
                sw = net.switch.loc[int(row['switch_idx'])]
            except (KeyError, TypeError, ValueError):
                continue
            if not bool(sw['closed']):
                continue
            key = int(row['switch_idx'])
            rows[key] = row
            relay_at_edge[frozenset((('bus', int(sw['bus'])), element_node(sw['et'], sw['element'])))] = key

        def trip_time(key):
            row = rows.get(key) or {}
            return float(row['t_trip_s']) if row.get('tripped') and row.get('t_trip_s') is not None else None

        def label(key):
            row = rows[key]
            return row.get('user_friendly_name') or row.get('switch_name') or str(key)

        zone, boundary = _prot_zone(adjacency, [fault_node], relay_at_edge)
        primaries = {k: far for k, far in boundary.items()
                     if _prot_reaches_source(adjacency, far, sources, zone)}
        scenario['unprotected_sources'] = sorted({n for b in zone & sources for n in source_names.get(b, [])})
        scenario['unprotected_inverter_sources'] = sorted(
            {n for b in zone for n in inverter_names.get(b, [])})
        scenario['primary_switches'] = [label(k) for k in primaries]

        backups = {}
        for key, far in primaries.items():
            zone2, boundary2 = _prot_zone(adjacency, [far], relay_at_edge, blocked=zone)
            if zone2 & sources:
                continue  # a source right behind the relay: nothing can back it up
            backups[key] = [b for b, far2 in boundary2.items()
                            if b != key and b not in primaries
                            and _prot_reaches_source(adjacency, far2, sources, zone | zone2)]

        clearing = []
        for key in primaries:
            t = trip_time(key)
            if t is None:
                t_backup = [trip_time(b) for b in backups.get(key, []) if trip_time(b) is not None]
                t = min(t_backup) if t_backup else None
            clearing.append(t)
        t_clear = max(clearing) if clearing and None not in clearing else None
        scenario['clearing_time_s'] = t_clear

        common = {
            'sc_line_id': scenario.get('sc_line_id'),
            'sc_fraction': scenario.get('sc_fraction'),
            'fault_bus': scenario.get('fault_bus'),
            'fault_label': scenario.get('fault_label'),
        }
        for key in primaries:
            t_p = trip_time(key)
            if t_p is None:
                continue
            for b in backups.get(key, []):
                t_b = trip_time(b)
                if t_b is None or t_b - t_p >= t_diff - 1e-9:
                    continue
                miscoord.append({
                    **common,
                    'primary_switch_id': rows[key].get('switch_id'),
                    'primary_user_friendly_name': label(key),
                    'primary_t_s': t_p,
                    'backup_switch_id': rows[b].get('switch_id'),
                    'backup_user_friendly_name': label(b),
                    'backup_t_s': t_b,
                    'delta_t_s': t_b - t_p,
                    'required_t_diff_s': float(t_diff),
                    'topology_path': True,
                })
        graded = set(primaries) | {b for bs in backups.values() for b in bs}
        if t_clear is not None:
            for key in rows:
                t = trip_time(key)
                if key in graded or t is None or t >= t_clear + t_diff - 1e-9:
                    continue
                unwanted.append({
                    **common,
                    'switch_id': rows[key].get('switch_id'),
                    'user_friendly_name': label(key),
                    't_trip_s': t,
                    'clearing_time_s': t_clear,
                    'required_t_diff_s': float(t_diff),
                    'primary_switches': scenario['primary_switches'],
                })
    return miscoord, unwanted


def protection_coordination(net, prot_params, in_data):
    """
    Run a complete protection coordination study.

    Returns a JSON string containing scenarios (trip tables), per-device settings +
    sampled time-current characteristics for the UI to plot, and a miscoordination
    summary. The function never raises into Flask: any failure is wrapped in an
    error JSON so the frontend can show a meaningful message.
    """
    try:
        fault_type = prot_params.get('fault_type', '3ph')
        if fault_type not in ('3ph', '2ph', '1ph'):
            fault_type = '3ph'
        case = prot_params.get('case', 'max')
        if case not in ('max', 'min'):
            case = 'max'
        sc_line_id_raw = prot_params.get('sc_line_id')
        sc_fraction = float(prot_params.get('sc_fraction', 0.5))
        if not (0.0 < sc_fraction < 1.0):
            sc_fraction = 0.5
        t_diff = float(prot_params.get('t_diff', 0.3))
        grading_mode = str(prot_params.get('grading_mode', 'auto') or 'auto').lower()
        if grading_mode not in ('auto', 'manual'):
            grading_mode = 'auto'

        # Validate connectivity early so we surface a clear message before
        # sc.calc_sc - naming the buses: the indices read
        # "[np.int64(5), np.int64(6), np.int64(7)]".
        isolated = isolated_buses_message(
            net, 'Connect every component to a supplied bus before running protection coordination.')
        if isolated:
            return json.dumps({
                'error': True,
                'message': isolated,
                'scenarios': [],
                'devices': [],
                'summary': {'converged': False},
            }, separators=(',', ':'))

        if net.switch.empty:
            return json.dumps({
                'error': True,
                'message': 'No switches found. Protection coordination requires Switch components on the diagram (one per protected element).',
                'scenarios': [],
                'devices': [],
                'summary': {'converged': False},
            }, separators=(',', ':'))

        # Attach protection devices based on the frontend Switch attributes.
        specs = _prot_collect_switch_protection_specs(in_data)
        _prot_merge_study_defaults(specs, prot_params)
        if not specs:
            return json.dumps({
                'error': True,
                'message': 'No protection device assigned. Open at least one Switch dialog, go to the Protection tab and pick a protection type (OCR / Fuse).',
                'scenarios': [],
                'devices': [],
                'summary': {'converged': False},
            }, separators=(',', ':'))

        # Ensure bus geodata exists before any pandapower.protection call so
        # create_sc_bus and time_grading do not raise on missing coordinates.
        _prot_ensure_bus_geodata(net)

        attach_summaries = _attach_protection_devices(net, specs, grading_mode)
        attached_count = sum(1 for s in attach_summaries if s.get('attached'))
        not_computed_count = sum(1 for s in attach_summaries if s.get('not_computed'))

        if attached_count == 0:
            return json.dumps({
                'error': True,
                # The alert is all the user sees, so it carries each reason.
                'message': 'No protection device could be evaluated:\n' + '\n'.join(
                    f"- {s.get('user_friendly_name') or s.get('switch_name') or s.get('switch_id')}: "
                    f"{s.get('reason') or 'not attached'}"
                    for s in attach_summaries),
                'attach_summaries': attach_summaries,
                'scenarios': [],
                'devices': [],
                'summary': {'converged': False, 'n_devices': 0, 'n_attached': 0, 'n_not_computed': not_computed_count},
            }, separators=(',', ':'))

        fault_location_mode = str(prot_params.get('fault_location_mode', 'line') or 'line').strip().lower()
        if fault_location_mode not in ('line', 'bus'):
            fault_location_mode = 'line'
        fault_bus_cell_id = prot_params.get('fault_bus_id')

        # Build the list of fault scenarios.
        scenarios = []
        if fault_location_mode == 'bus':
            fault_bus_idx = _prot_resolve_fault_bus_idx(in_data, net, fault_bus_cell_id)
            if fault_bus_idx is None:
                return json.dumps({
                    'error': True,
                    'message': 'Could not resolve the selected fault busbar. Place a Bus on the diagram and pick it in the Fault tab.',
                    'scenarios': [],
                    'devices': [],
                    'summary': {'converged': False},
                }, separators=(',', ':'))
            scenarios.append(_prot_run_bus_scenario(net, fault_bus_idx, fault_type, case, attach_summaries))
        else:
            if sc_line_id_raw in (None, '', 'all'):
                line_ids = [int(idx) for idx in net.line.index if net.line.at[idx, 'in_service']]
            else:
                try:
                    line_ids = [int(sc_line_id_raw)]
                except (TypeError, ValueError):
                    line_ids = [int(idx) for idx in net.line.index if net.line.at[idx, 'in_service']]
            for line_id in line_ids:
                scenarios.append(_prot_run_scenario(net, line_id, sc_fraction, fault_type, case, attach_summaries))

        scenario_warning = None
        if not scenarios:
            if fault_location_mode == 'line' and (not hasattr(net, 'line') or net.line.empty):
                scenario_warning = (
                    'No fault scenarios were run: fault location is "line" but the model has no in-service lines. '
                    'Use "At selected busbar" in the Fault tab, or add lines to the diagram.'
                )
            elif fault_location_mode == 'line':
                scenario_warning = 'No fault scenarios were run: no matching in-service lines for the selected line fault.'
            else:
                scenario_warning = 'No fault scenarios were run.'

        # Sample characteristics on the unmodified net so curves do not include the sc_bus.
        devices = _prot_extract_devices_for_ui(net, attach_summaries)
        _prot_append_custom_devices(devices, attach_summaries)

        for scenario in scenarios:
            scenario['fault_label'] = _prot_fault_label(net, scenario)
        miscoord, unwanted = _prot_check_miscoordination(net, scenarios, t_diff)

        n_tripped = sum(1 for sc_res in scenarios for trip in sc_res.get('trip', []) if trip.get('tripped'))
        response = {
            'error': False,
            'scenarios': scenarios,
            'devices': devices,
            'attach_summaries': attach_summaries,
            'summary': {
                'converged': True,
                'n_devices': len(devices),
                'n_attached': attached_count,
                'n_not_computed': not_computed_count,
                'n_scenarios': len(scenarios),
                'n_tripped': n_tripped,
                'n_miscoordination': len(miscoord),
                'n_unwanted_trips': len(unwanted),
                'fault_type': fault_type,
                'case': case,
                'grading_mode': grading_mode,
                'fault_location_mode': fault_location_mode,
                't_diff_s': t_diff,
                **({'scenario_warning': scenario_warning} if scenario_warning else {}),
            },
            'miscoordination': miscoord,
            'unwanted_trips': unwanted,
            'output': {
                'show_curves': bool(prot_params.get('show_curves', True)),
                'show_table': bool(prot_params.get('show_table', True)),
                'show_miscoordination': bool(prot_params.get('show_miscoordination', True)),
            },
        }
        if scenario_warning:
            response['warning'] = scenario_warning
        return json.dumps(
            _prot_to_jsonable(response),
            default=_json_serialize_default,
            separators=(',', ':'),
        )
    except Exception as e:
        import traceback
        return json.dumps({
            'error': True,
            'message': f'Protection coordination failed: {e}',
            'traceback': traceback.format_exc(limit=20),
            'scenarios': [],
            'devices': [],
            'summary': {'converged': False},
        }, separators=(',', ':'))
