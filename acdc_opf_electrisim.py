"""
AC/DC optimal power flow with least shed (Pyomo and Ipopt) on the network Electrisim builds.

pandapower's optimal power flow leaves the DC network out, and on an island no dispatch can supply
it says only that it did not converge. This study keeps the DC side and answers how much load must
go:

  AC     pandapower's own admittance matrix, from Electrisim's load flow of the same network - lines,
         transformers (two- and three-winding, at their taps), shunts, capacitors and switches as the
         load flow has them; polar power flow, every voltage within its bus limits.
  Sources each external grid (its voltage held, P and Q free); generators within their P and Q
         limits; static generators, AC storage, motors and loads fixed at the load flow's power.
  AFEs   each VSC that is not one of Electrisim's internal stages, at the internal bus pandapower puts
         behind its reactor (the reactor and its loss in the admittance matrix); its MVA within its
         rating times min(V, 1).
  PCS    each PCS: its efficiency both ways, within its rating times min(V, 1), |Q| within its cap;
         fed from its DC bus, or from the battery behind it.
  SSTs   each SST: its rectifier's and DC/DC stage's efficiency and no-load loss, each stage within
         its rating; its MV reactive power as the load flow has it.
  DC     the DC buses joined by in-service DC lines and closed DC breakers, each group a node (their
         resistance left out); its batteries within their C-rates (priced, so they discharge only
         where needed); its DC loads.
  DC/DC  each DC/DC converter a link between the groups at its terminals: its efficiency and no-load
         loss, within its rating, one way or both as drawn - whatever its control in the load flow
         (droop, voltage, power, smoothing), the optimiser sets its power.
  Diodes each DC diode a one-way link from its anode's group to its cathode's, within its rated
         current at its bus voltage, its forward drop a loss. The groups have no voltage, so what each
         of two diode-ORed feeds carries is the optimiser's choice, not set by their voltages.
  Sources a PV array up to its maximum power point (curtailed only if it must be); an SOFC system at
         its set power, or up to its rating if asked (priced like stored energy); supercapacitors and
         flywheels hold no steady power, so at none (a PCS fed from one gives Q only).
  Ratings each line (max_i_ka) and two-winding transformer (sn_mva) within its rating times its
         voltage, if asked.

Objectives: least losses (all loads served), or least shed - the most of the sheddable loads served,
then the least losses. A PCS fed from a source this study does not know is named and the study
refused, rather than left out silently.
"""
from __future__ import annotations

import json
import math
import os
import shutil
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import pandapower_electrisim as pe

BAT_PRICE = 2.0          # stored energy against the grid's, in the objective
SHED_WEIGHT = 100.0      # a MW shed against a MW of loss


def _ipopt_path() -> Optional[str]:
    """Ipopt: ELECTRISIM_IPOPT, then ../tools/ipopt beside the backend, then the PATH."""
    env = os.environ.get('ELECTRISIM_IPOPT')
    if env and os.path.exists(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    tools = os.path.join(os.path.dirname(here), 'tools', 'ipopt')
    for root, _, files in os.walk(tools) if os.path.isdir(tools) else []:
        for f in files:
            if f.lower() in ('ipopt.exe', 'ipopt'):
                return os.path.join(root, f)
    return shutil.which('ipopt')


def _cid(df, i) -> str:
    """An element's drawing id (the diagram cell its result box hangs on), or ''."""
    return str(df.at[i, 'id']) if 'id' in df.columns and pd.notna(df.at[i, 'id']) else ''


def _label(net, name) -> str:
    return str((getattr(net, 'user_friendly_names', None) or {}).get(name, name))


def _flag(df, col) -> pd.Series:
    if df is None or col not in df.columns:
        return pd.Series(False, index=getattr(df, 'index', []))
    return df[col].fillna(False).astype(bool)


SOURCE_KINDS = ('DC bus', 'Battery', 'PV Array', 'SOFC', 'Supercapacitor', 'Flywheel', None)
DER_KINDS = ('Battery', 'PV Array', 'SOFC', 'Supercapacitor', 'Flywheel')


def _unsupported(net) -> List[str]:
    out = []
    for d in getattr(net, 'electrisim_ders', None) or []:
        if d.get('kind') not in DER_KINDS:
            out.append(f"{d.get('kind')} '{d.get('label')}'")
    for r in getattr(net, 'electrisim_pcs', None) or []:
        src = r.get('source') or {}
        if src.get('kind') not in SOURCE_KINDS:
            out.append(f"PCS '{r.get('label')}' fed from a {src.get('kind')}")
    return out


def _dc_groups(net) -> Dict[int, int]:
    """Each DC bus's group: joined by in-service DC lines (and closed DC breakers, which are DC lines), not by diodes."""
    parent = {int(b): int(b) for b in net.bus_dc.index} if len(getattr(net, 'bus_dc', [])) else {}

    def root(b):
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        return b

    diode = _flag(getattr(net, 'line_dc', None), 'electrisim_dc_diode')
    for i in getattr(net, 'line_dc', pd.DataFrame()).index:
        if bool(net.line_dc.at[i, 'in_service']) and not bool(diode.get(i, False)):
            a, b = int(net.line_dc.at[i, 'from_bus_dc']), int(net.line_dc.at[i, 'to_bus_dc'])
            if a in parent and b in parent:
                parent[root(a)] = root(b)
    return {b: root(b) for b in parent}


def _battery_limits(obj) -> tuple:
    """(discharge MW, charge MW) of a battery DER: its C-rates on its energy, nothing past its charge window."""
    f = getattr(obj, 'fields', {}) or {}
    energy = float(getattr(obj, 'energy_kwh', 0.0) or float(f.get('capacity_kwh', 0) or 0)) / 1000.0
    dis = energy * float(f.get('c_rate_discharge', 1.0) or 1.0)
    ch = energy * float(f.get('c_rate_charge', 1.0) or 1.0)
    soc, lo, hi = (float(getattr(obj, k, d)) for k, d in (('soc0', 0.5), ('soc_min', 0.0), ('soc_max', 1.0)))
    return (dis if soc > lo + 1e-6 else 0.0), (ch if soc < hi - 1e-6 else 0.0)


def _aux_rows(net, kind: str, n: int) -> Optional[np.ndarray]:
    """
    The internal rows (ppci) of pandapower's auxiliary buses of ``kind`` ('vsc', 'trafo3w'), or -1 for one out of
    service. pandapower updates its bus lookup to the internal numbering but leaves its auxiliary lookup in the
    order it built them, before the rows out of service (an open bus, or one fused to another by a closed switch)
    went to the end: each auxiliary row moves up by the rows out of service before it.
    """
    aux = net._pd2ppc_lookups.get('aux') or {}
    if aux.get(kind) is None:
        return None
    lookup = net._pd2ppc_lookups['bus']
    drawn = np.asarray([int(lookup[b]) for b in net.bus.index])
    oos_before_aux = len(net.bus) - len({int(i) for i in drawn if 0 <= i < n})
    # The auxiliary rows out of service: those of elements out of service, in build order.
    oos_aux = []
    for k, rows_k in aux.items():
        table = net.get(k)
        rows_k = np.asarray(rows_k if rows_k is not None else []).ravel()
        if table is None or len(table) != len(rows_k):
            continue
        oos_aux.extend(int(r) for r, on in zip(rows_k, table['in_service'].to_numpy(dtype=bool)) if not on)
    oos_set = set(oos_aux)
    oos_aux = np.sort(np.asarray(oos_aux, dtype=np.int64))
    out = []
    for r in np.asarray(aux[kind]).ravel():
        r = int(r)
        if r in oos_set:
            out.append(-1)
            continue
        i = r - oos_before_aux - int(np.searchsorted(oos_aux, r))
        out.append(i if 0 <= i < n else -1)
    return np.asarray(out, dtype=np.int64)


def _stage_eta(net, table, idx) -> tuple:
    df = net[table]
    eta = float(df.at[idx, 'electrisim_dcdc_eta']) if 'electrisim_dcdc_eta' in df.columns and pd.notna(df.at[idx, 'electrisim_dcdc_eta']) else 1.0
    nl = float(df.at[idx, 'electrisim_dcdc_p_nl_mw']) if 'electrisim_dcdc_p_nl_mw' in df.columns and pd.notna(df.at[idx, 'electrisim_dcdc_p_nl_mw']) else 0.0
    return (eta if 0 < eta <= 1 else 1.0), nl


def build_and_solve(net, params: Dict[str, Any]) -> Dict[str, Any]:
    """The study on a network Electrisim has built and solved (its load flow's results in place)."""
    import pyomo.environ as pyo
    from pandapower.pypower.makeYbus import branch_vectors

    mode = str(params.get('objective', 'shed'))
    shed_dc = str(params.get('shed_dc_loads', 'true')).lower() == 'true'
    shed_ac = str(params.get('shed_ac_loads', 'false')).lower() == 'true'
    batteries = str(params.get('batteries', 'false')).lower() == 'true'
    fuel_cells = str(params.get('fuel_cells', 'false')).lower() == 'true'
    limits = str(params.get('branch_limits', 'true')).lower() == 'true'
    tf = float(params.get('transformer_factor', 1.0) or 1.0)
    warnings: List[str] = []

    ppc = net._ppc
    ppci = ppc['internal']
    base = float(ppci['baseMVA'])
    ybus = ppci['Ybus'].tocsr()
    n = ybus.shape[0]
    lookup = net._pd2ppc_lookups['bus']
    vm0 = np.abs(ppci['V'])
    va0 = np.angle(ppci['V'])
    bus_vn = ppci['bus'][:, 9]                       # BASE_KV

    def ix(pp_bus):
        i = int(lookup[int(pp_bus)])
        return i if 0 <= i < n else None

    # The rows modelled: every row pandapower built - the drawn buses, a three-winding transformer's star
    # point, the internal bus behind each drawn VSC's reactor (its AC injection is there; the reactor is
    # a branch) - but Electrisim's own aids: its auxiliary buses (an SST's stages, a battery's inner bus)
    # and the internal buses of its auxiliary VSCs.
    drop = set()
    for b in net.bus.index:
        if pe._electrisim_is_aux(net.bus, b):
            i = ix(b)
            if i is not None:
                drop.add(i)
    vsc_row = {}
    aux_rows = _aux_rows(net, 'vsc', n)
    if aux_rows is not None and len(getattr(net, 'vsc', [])):
        for k, v in enumerate(net.vsc.index):
            if k >= len(aux_rows) or not 0 <= int(aux_rows[k]) < n or not bool(net.vsc.at[v, 'in_service']):
                continue
            if pe._electrisim_is_aux(net.vsc, v):
                drop.add(int(aux_rows[k]))
            else:
                vsc_row[v] = int(aux_rows[k])
    keep_ac = set(range(n)) - drop
    rows = sorted(keep_ac)
    coupling = ybus[rows][:, [j for j in range(n) if j not in keep_ac]] if len(rows) < n else None
    if coupling is not None and coupling.nnz:
        warnings.append('Some of Electrisim\'s internal buses are coupled to the drawn network through branches; '
                        'they are kept at the load flow\'s voltage.')

    m = pyo.ConcreteModel()
    m.N = pyo.Set(initialize=rows)
    # Voltage limits from the drawn buses (a load flow leaves pandapower's own at 0-2 pu); a VSC's internal
    # bus none of its own (0.5-1.5).
    vmin = {i: 0.5 for i in rows}
    vmax = {i: 1.5 for i in rows}
    seen = set()
    for b in net.bus.index:
        i = ix(b)
        if i not in vmin or pe._electrisim_is_aux(net.bus, b):
            continue
        lo_b = float(net.bus.at[b, 'min_vm_pu']) if 'min_vm_pu' in net.bus.columns and pd.notna(net.bus.at[b, 'min_vm_pu']) else 0.9
        hi_b = float(net.bus.at[b, 'max_vm_pu']) if 'max_vm_pu' in net.bus.columns and pd.notna(net.bus.at[b, 'max_vm_pu']) else 1.1
        vmin[i] = lo_b if i not in seen else max(vmin[i], lo_b)
        vmax[i] = hi_b if i not in seen else min(vmax[i], hi_b)
        seen.add(i)
    lo = {i: min(max(vmin[i], 0.5), float(vm0[i])) for i in rows}
    hi = {i: max(min(vmax[i] if vmax[i] > 0 else 1.5, 1.5), float(vm0[i])) for i in rows}
    m.V = pyo.Var(m.N, bounds=lambda m, i: (lo[i], hi[i]), initialize=lambda m, i: float(vm0[i]))
    m.th = pyo.Var(m.N, initialize=lambda m, i: float(va0[i]))
    for i in rows:
        # Limits Electrisim leaves wide open (0-2 pu: a zigzag's floating delta) mean "no limit": the 0.5-1.5 pu
        # of a bus with none of its own (it follows its neighbours).
        if i in seen and (vmin[i] <= 0.0 or vmax[i] >= 1.9):
            m.V[i].setlb(0.5)
            m.V[i].setub(1.5)


    p_inj = {i: 0 for i in rows}
    q_inj = {i: 0 for i in rows}

    dropped = []

    def add(i, p, q, what=''):
        if i in p_inj:
            p_inj[i] = p_inj[i] + p
            q_inj[i] = q_inj[i] + q
        else:
            dropped.append((what, i))

    refs = set()
    # External grids (not Electrisim's aids): voltage held, P and Q free.
    m.Pg = pyo.Var(list(net.ext_grid.index), initialize=0.0)
    m.Qg = pyo.Var(list(net.ext_grid.index), initialize=0.0)
    grids = []
    for g in net.ext_grid.index:
        if not bool(net.ext_grid.at[g, 'in_service']) or pe._electrisim_is_aux(net.ext_grid, g):
            m.Pg[g].fix(0.0)
            m.Qg[g].fix(0.0)
            continue
        i = ix(net.ext_grid.at[g, 'bus'])
        if i is None or i not in p_inj:
            m.Pg[g].fix(0.0)
            m.Qg[g].fix(0.0)
            continue
        m.V[i].fix(float(net.ext_grid.at[g, 'vm_pu']))
        m.Pg[g].set_value(float(net.res_ext_grid.at[g, 'p_mw']) / base)
        add(i, m.Pg[g], m.Qg[g])
        refs.add(i)
        grids.append(g)

    # PCS: their generator rows are theirs, not generators of their own.
    pcs = [r for r in getattr(net, 'electrisim_pcs', None) or [] if r.get('in_service', True)]
    pcs_names = {r.get('name') for r in getattr(net, 'electrisim_pcs', None) or []}
    # (a PCS on a PV array or SOFC is a static generator in the load flow, on a battery a generator)
    pcs_rows = {(r.get('table'), r.get('index')) for r in getattr(net, 'electrisim_pcs', None) or []}

    # Generators (not PCS): P within its limits if controllable, else fixed; Q within its limits.
    gens = [g for g in net.gen.index if bool(net.gen.at[g, 'in_service']) and net.gen.at[g, 'name'] not in pcs_names
            and not pe._electrisim_is_aux(net.gen, g)]
    m.Pgen = pyo.Var(gens, initialize=lambda m, g: float(net.res_gen.at[g, 'p_mw']) / base)
    m.Qgen = pyo.Var(gens, initialize=lambda m, g: float(net.res_gen.at[g, 'q_mvar']) / base)
    for g in gens:
        col = lambda c, d: float(net.gen.at[g, c]) if c in net.gen.columns and pd.notna(net.gen.at[g, c]) else d
        has_limits = all(c in net.gen.columns and pd.notna(net.gen.at[g, c]) for c in ('min_p_mw', 'max_p_mw'))
        controllable = bool(net.gen.at[g, 'controllable']) if 'controllable' in net.gen.columns and pd.notna(net.gen.at[g, 'controllable']) else False
        slack = bool(net.gen.at[g, 'slack']) if 'slack' in net.gen.columns and pd.notna(net.gen.at[g, 'slack']) else False
        if slack and not (controllable and has_limits):
            # An island's slack (the grid gone): free as a grid is, within its rating if it has one.
            hi_g = col('max_p_mw', float('nan'))
            if not math.isfinite(hi_g):
                hi_g = col('sn_mva', float('nan'))
            m.Pgen[g].setlb(max(col('min_p_mw', 0.0), 0.0) / base)
            m.Pgen[g].setub(hi_g / base if math.isfinite(hi_g) and hi_g > 0 else None)
        elif controllable and has_limits:
            # Dispatched within its limits, as pandapower's OPF would.
            m.Pgen[g].setlb(col('min_p_mw', 0.0) / base)
            m.Pgen[g].setub(col('max_p_mw', 0.0) / base)
        else:
            # No limits of its own: held at the load flow's output, not switched off for being as dear as the grid.
            m.Pgen[g].fix(float(net.res_gen.at[g, 'p_mw']) / base)
        m.Qgen[g].setlb(col('min_q_mvar', -1e6) / base)
        m.Qgen[g].setub(col('max_q_mvar', 1e6) / base)
        i = ix(net.gen.at[g, 'bus'])
        add(i, m.Pgen[g], m.Qgen[g])
        if bool(net.gen.at[g, 'slack']) if 'slack' in net.gen.columns else False:
            refs.add(i)

    # Fixed injections at the load flow's power: static generators, AC storage, motors, and the loads.
    def fixed(table, sign):
        df, res = net.get(table), net.get(f'res_{table}')
        if df is None or res is None:
            return
        for e in df.index:
            if e in res.index and bool(df.at[e, 'in_service']) and not pe._electrisim_is_aux(df, e) \
                    and (table, e) not in pcs_rows and df.at[e, 'name'] not in pcs_names:
                add(ix(df.at[e, 'bus']), sign * float(res.at[e, 'p_mw']) / base, sign * float(res.at[e, 'q_mvar']) / base, f'{table} {e}')

    fixed('sgen', 1)
    fixed('storage', -1)
    fixed('motor', -1)
    # Loads: Electrisim's own (an SST's MV draw) are the SSTs'; the rest fixed, or sheddable.
    ac_loads = [e for e in net.load.index if bool(net.load.at[e, 'in_service']) and not pe._electrisim_is_aux(net.load, e)
                and e in net.res_load.index]
    m.La = pyo.Var(ac_loads, bounds=(0.0, 1.0), initialize=1.0)       # share of each AC load served
    for e in ac_loads:
        if not shed_ac or mode != 'shed':
            m.La[e].fix(1.0)
        add(ix(net.load.at[e, 'bus']), -m.La[e] * float(net.res_load.at[e, 'p_mw']) / base,
            -m.La[e] * float(net.res_load.at[e, 'q_mvar']) / base)

    # The DC side: groups of DC buses.
    groups = _dc_groups(net)
    dc_bal: Dict[int, Any] = {}

    def dc(b, expr):
        g = groups.get(int(b))
        if g is not None:
            dc_bal[g] = dc_bal.get(g, 0) + expr

    # AFEs: the VSCs that are not Electrisim's internal stages.
    afe = [v for v in net.vsc.index if v in vsc_row] if len(getattr(net, 'vsc', [])) else []
    m.Pa = pyo.Var(afe, initialize=lambda m, v: float(net.res_vsc.at[v, 'p_mw']) / base)
    m.Qa = pyo.Var(afe, initialize=lambda m, v: float(net.res_vsc.at[v, 'q_mvar']) / base)
    m.ua = pyo.Var(afe, bounds=(0.0, 1.0), initialize=1.0)
    m.conv = pyo.ConstraintList()
    afe_rating = {}
    for v in afe:
        i = vsc_row[v]                                   # its internal bus, behind its reactor
        s = float(net.vsc.at[v, 'rated_mva']) if 'rated_mva' in net.vsc.columns and pd.notna(net.vsc.at[v, 'rated_mva']) and float(net.vsc.at[v, 'rated_mva']) > 0 else None
        if s is None:
            s = math.hypot(float(net.res_vsc.at[v, 'p_mw']), float(net.res_vsc.at[v, 'q_mvar'])) * 1.2 or 1.0
            warnings.append(f"VSC '{_label(net, net.vsc.at[v, 'name'])}' has no rating: taken as 1.2 times its load flow's.")
        afe_rating[v] = s
        m.conv.add(m.ua[v] <= m.V[i])
        m.conv.add(m.Pa[v] ** 2 + m.Qa[v] ** 2 <= (s / base * m.ua[v]) ** 2)
        add(i, -m.Pa[v], -m.Qa[v])
        dc(net.vsc.at[v, 'bus_dc'], m.Pa[v])           # its reactor's loss is in the admittance matrix

    # Batteries on DC buses (coupled directly) and behind PCS.
    bats = [d for d in getattr(net, 'electrisim_ders', None) or [] if d.get('kind') == 'Battery'
            and getattr(d.get('obj'), 'in_service', True)]
    # A battery drawn only as a PCS's source is not one of the DERs: its own entry, behind its PCS.
    known = {d.get('name') for d in bats}
    for r in pcs:
        src = r.get('source') or {}
        name = getattr(src.get('obj'), 'name', None)
        if src.get('kind') == 'Battery' and name is not None and name not in known and getattr(src['obj'], 'in_service', True):
            known.add(name)
            bats.append({'kind': 'Battery', 'obj': src['obj'], 'coupling': 'pcs', 'name': name, 'label': src.get('label')})
    bat_ids = list(range(len(bats)))
    lim = {k: _battery_limits(bats[k]['obj']) for k in bat_ids}
    m.Bd = pyo.Var(bat_ids, bounds=lambda m, k: (0.0, (lim[k][0] if batteries else 0.0) / base), initialize=0.0)
    m.Bc = pyo.Var(bat_ids, bounds=lambda m, k: (0.0, (lim[k][1] if batteries else 0.0) / base), initialize=0.0)
    pcs_battery = {}
    for k, d in enumerate(bats):
        if d.get('coupling') in ('direct', 'converter') and d.get('bus') is not None:
            dc(d['bus'], m.Bd[k] - m.Bc[k])          # on a converter's input: the converter is a link of its own
        else:
            pcs_battery[d.get('name')] = k

    # PV arrays and SOFC systems: on a DC bus (directly, or behind a DC/DC converter) and behind PCS.
    # Each source's DC power within (lo, hi) MW; an SOFC's rise above its set power priced like stored energy.
    srcs = []                                        # (record or PCS source, kind, lo, hi, set)
    zero_stores = []

    def source_window(obj, kind, p_now):
        if kind == 'PV Array':
            return 0.0, max(p_now, 0.0), max(p_now, 0.0)
        p_max = obj.p_rated * (1.0 - obj.aux_frac) / 1e6
        return p_now, (max(p_max, p_now) if fuel_cells else p_now), p_now

    for d in getattr(net, 'electrisim_ders', None) or []:
        obj = d.get('obj')
        if not getattr(obj, 'in_service', True) or d.get('coupling') == 'pcs':
            continue
        if d['kind'] in ('Supercapacitor', 'Flywheel'):
            zero_stores.append(d.get('label'))
            continue
        if d['kind'] not in ('PV Array', 'SOFC'):
            continue
        if d.get('coupling') == 'direct' and 'load' in (d.get('parts') or {}):
            ld = d['parts']['load']
            p_now = -float(net.res_load_dc.at[ld, 'p_dc_mw']) if ld in net.res_load_dc.index else -float(net.load_dc.at[ld, 'p_dc_mw'])
        else:
            p_now = (obj.mpp()[2] if d['kind'] == 'PV Array' else obj.p_operating()) / 1e6
        srcs.append((d, d['kind']) + source_window(obj, d['kind'], p_now))
    pcs_src = {}                                     # a PCS (by id) -> its source's variable
    for r in pcs:
        src = r.get('source') or {}
        if src.get('kind') in ('PV Array', 'SOFC') and src.get('obj') is not None:
            obj = src['obj']
            p_now = (obj.mpp()[2] if src['kind'] == 'PV Array' else obj.p_operating()) / 1e6
            pcs_src[id(r)] = len(srcs)
            srcs.append(({'label': r.get('label')}, src['kind']) + source_window(obj, src['kind'], p_now))
        elif src.get('kind') in ('Supercapacitor', 'Flywheel'):
            zero_stores.append(getattr(src.get('obj'), 'name', None) and _label(net, src['obj'].name) or r.get('label'))
    m.Pv = pyo.Var(list(range(len(srcs))), bounds=lambda m, k: (srcs[k][2] / base, srcs[k][3] / base),
                   initialize=lambda m, k: srcs[k][4] / base)
    for k, (d, kind, lo_s, hi_s, _) in enumerate(srcs):
        if 'bus' in d:
            dc(d['bus'], m.Pv[k])
    if zero_stores:
        warnings.append(f"{len(zero_stores)} supercapacitor or flywheel store(s) hold no steady power: at none in this study"
                        f" ({', '.join(str(z) for z in zero_stores[:4])}{', ...' if len(zero_stores) > 4 else ''}).")

    # DC/DC converters: links between the groups at their terminals, forward and (if drawn so) back.
    convs = [c for c in getattr(net, 'electrisim_dc_dc_converters', None) or [] if c.get('in_service', True)
             and groups.get(int(c['bus_in'])) is not None and groups.get(int(c['bus_out'])) is not None]
    cids = list(range(len(convs)))
    rated = lambda c: (float(c['rated_mw']) if float(c.get('rated_mw') or 0) > 0 else None)
    m.Df = pyo.Var(cids, bounds=lambda m, k: (0.0, rated(convs[k]) / base if rated(convs[k]) else None), initialize=0.0)
    m.Dr = pyo.Var(cids, bounds=lambda m, k: (0.0, (rated(convs[k]) / base if rated(convs[k]) else None)
                                               if convs[k].get('bidirectional') else 0.0), initialize=0.0)
    for k, c in enumerate(convs):
        eta = float(c.get('eta') or 1.0)
        if rated(c):
            m.conv.add(m.Df[k] + m.Dr[k] <= rated(c) / base)
        dc(c['bus_out'], m.Df[k] - m.Dr[k])
        dc(c['bus_in'], -(m.Df[k] / eta - m.Dr[k] * eta) - float(c.get('p_nl_mw') or 0.0) / base)

    # DC diodes: one way, anode to cathode, within their rated current; the forward drop a loss.
    diodes = [d for d in getattr(net, 'electrisim_dc_diodes', None) or [] if d.get('in_service', True)
              and groups.get(int(d['bus_from'])) is not None and groups.get(int(d['bus_to'])) is not None]
    did = list(range(len(diodes)))
    vn_dc = lambda b: float(net.bus_dc.at[b, 'vn_kv'])
    m.Dd = pyo.Var(did, bounds=lambda m, k: (0.0, (float(diodes[k]['rated_current_ka']) * vn_dc(diodes[k]['bus_from']) / base)
                                             if float(diodes[k].get('rated_current_ka') or 0) > 0 else None), initialize=0.0)
    for k, d in enumerate(diodes):
        drop_ = min(float(d.get('v_f_v') or 0.0) / (vn_dc(d['bus_from']) * 1e3), 0.5)
        dc(d['bus_from'], -m.Dd[k])
        dc(d['bus_to'], m.Dd[k] * (1.0 - drop_))
    if diodes:
        warnings.append('The DC buses are nodes without voltage here: what each diode carries is the optimiser\'s '
                        'choice, not set by its feeds\' voltages.')

    # PCS: discharging and charging apart, each with its efficiency.
    pids = list(range(len(pcs)))
    m.Po = pyo.Var(pids, bounds=(0.0, None), initialize=lambda m, k: max(float(pcs[k].get('p_set_mw') or 0.0), 0.0) / base)
    m.Pc = pyo.Var(pids, bounds=(0.0, None), initialize=0.0)
    m.Qi = pyo.Var(pids, initialize=0.0)
    m.ui = pyo.Var(pids, bounds=(0.0, 1.0), initialize=1.0)
    for k, r in enumerate(pcs):
        i = ix(r['bus'])
        eta = float(r.get('eta') or 1.0)
        s = float(r.get('s_rated') or 0.0) / base
        qcap = float(r.get('q_max_pu') or 1.0) * s
        m.Qi[k].setlb(-qcap)
        m.Qi[k].setub(qcap)
        m.conv.add(m.ui[k] <= m.V[i])
        m.conv.add((m.Po[k] - m.Pc[k]) ** 2 + m.Qi[k] ** 2 <= (s * m.ui[k]) ** 2)
        add(i, m.Po[k] - m.Pc[k], m.Qi[k])
        draw = m.Po[k] / eta - m.Pc[k] * eta
        src = r.get('source') or {}
        if src.get('kind') == 'DC bus' and r.get('dc_bus') is not None:
            dc(r['dc_bus'], -draw)
        elif id(r) in pcs_src:
            m.Pc[k].fix(0.0)                             # a PV array or SOFC takes nothing back
            m.conv.add(m.Pv[pcs_src[id(r)]] == draw)
        elif src.get('kind') in ('Supercapacitor', 'Flywheel'):
            m.Po[k].fix(0.0)                             # Q only
            m.Pc[k].fix(0.0)
        else:
            kb = pcs_battery.get(getattr(src.get('obj'), 'name', None)) if src.get('obj') is not None else None
            if kb is None:
                warnings.append(f"PCS '{r.get('label')}' has no source this study models: held at none.")
                m.Po[k].fix(0.0)
                m.Pc[k].fix(0.0)
            else:
                m.conv.add(m.Bd[kb] - m.Bc[kb] == draw)
                if not batteries:
                    m.Bd[kb].setub(_battery_limits(bats[kb]['obj'])[0] / base)
                    m.Bc[kb].setub(_battery_limits(bats[kb]['obj'])[1] / base)
        if r.get('reference') or r.get('island') is not None:
            refs.add(i)

    # SSTs: the MV draw for what they deliver at their LV DC port.
    ssts = [s for s in getattr(net, 'electrisim_ssts', None) or [] if s.get('in_service', True)]
    sids = list(range(len(ssts)))
    m.Ps = pyo.Var(sids, bounds=(0.0, None), initialize=0.0)       # delivered at the LV DC port
    sst_info = []
    for k, s in enumerate(ssts):
        st = {x['stage']: x for x in s['stages']}
        rect, dcdc = st['rectifier'], st['dcdc']
        eta_r, nl_r = _stage_eta(net, *rect['input'])
        eta_d, nl_d = _stage_eta(net, *dcdc['input'])
        link_in = m.Ps[k] / eta_d + nl_d / base
        mv = link_in / eta_r + nl_r / base
        mv_load = rect['input'][1]
        q = float(net.res_load.at[mv_load, 'q_mvar']) / base if mv_load in net.res_load.index else 0.0
        add(ix(s['bus_mv']), -mv, -q)
        m.conv.add(m.Ps[k] <= float(dcdc['rated_mw']) / base)
        m.conv.add(link_in <= float(rect['rated_mw']) / base)
        dc(s['bus_lvdc'], m.Ps[k])
        sst_info.append((s, mv))

    # DC loads: fixed, or sheddable.
    dloads = [e for e in net.load_dc.index if bool(net.load_dc.at[e, 'in_service']) and not pe._electrisim_is_aux(net.load_dc, e)] \
        if len(getattr(net, 'load_dc', [])) else []
    m.Ld = pyo.Var(dloads, bounds=(0.0, 1.0), initialize=1.0)
    for e in dloads:
        if not shed_dc or mode != 'shed':
            m.Ld[e].fix(1.0)
        p0 = float(net.res_load_dc.at[e, 'p_dc_mw']) if e in net.res_load_dc.index else float(net.load_dc.at[e, 'p_dc_mw'])
        dc(net.load_dc.at[e, 'bus_dc'], -m.Ld[e] * p0 / base)

    m.dcbal = pyo.ConstraintList()
    for g, expr in dc_bal.items():
        if not isinstance(expr, (int, float)):
            m.dcbal.add(expr == 0)

    # The AC power flow on the kept rows.
    m.bal = pyo.ConstraintList()
    for i in rows:
        start, end = ybus.indptr[i], ybus.indptr[i + 1]
        pe_, qe_ = 0, 0
        for jj in range(start, end):
            j = int(ybus.indices[jj])
            g, b = float(ybus.data[jj].real), float(ybus.data[jj].imag)
            vj = m.V[j] if j in keep_ac else float(vm0[j])
            tj = m.th[j] if j in keep_ac else float(va0[j])
            d = m.th[i] - tj
            pe_ = pe_ + m.V[i] * vj * (g * pyo.cos(d) + b * pyo.sin(d))
            qe_ = qe_ + m.V[i] * vj * (g * pyo.sin(d) - b * pyo.cos(d))
        m.bal.add(p_inj[i] == pe_)
        m.bal.add(q_inj[i] == qe_)

    # Angle references: one in each AC island.
    import scipy.sparse.csgraph as csg
    ncomp, comp = csg.connected_components(ybus[rows][:, rows], directed=False)
    for c in range(ncomp):
        members = [rows[k] for k in range(len(rows)) if comp[k] == c]
        r = next((i for i in members if i in refs), members[0])
        m.th[r].fix(float(va0[r]))

    # Branch ratings: lines and two-winding transformers.
    branch_rows = []
    look = net._pd2ppc_lookups.get('branch', {})
    br = ppc['branch']
    Ytt, Yff, Yft, Ytf = branch_vectors(br, br.shape[0])
    for table, rating in (('line', lambda e: math.sqrt(3) * float(net.line.at[e, 'max_i_ka']) * float(net.bus.at[net.line.at[e, 'from_bus'], 'vn_kv'])
                                   * float(net.line.at[e, 'df']) * float(net.line.at[e, 'parallel'])),
                          ('trafo', lambda e: float(net.trafo.at[e, 'sn_mva']) * float(net.trafo.at[e, 'parallel']) * tf)):
        if table not in look:
            continue
        a, _ = look[table]
        for k, e in enumerate(net[table].index):
            row = a + k
            if not bool(net[table].at[e, 'in_service']) or pe._electrisim_is_aux(net[table], e) or br[row, 10] == 0:
                continue
            f, t = int(br[row, 0].real), int(br[row, 1].real)
            if f not in keep_ac or t not in keep_ac:
                continue
            branch_rows.append((table, e, row, f, t, rating(e)))
    m.rate = pyo.ConstraintList()

    def flow(row, f, t, end):
        if end == 'f':
            yii, yij, a, b = Yff[row], Yft[row], f, t
        else:
            yii, yij, a, b = Ytt[row], Ytf[row], t, f
        d = m.th[a] - m.th[b]
        p = yii.real * m.V[a] ** 2 + m.V[a] * m.V[b] * (yij.real * pyo.cos(d) + yij.imag * pyo.sin(d))
        q = -yii.imag * m.V[a] ** 2 + m.V[a] * m.V[b] * (yij.real * pyo.sin(d) - yij.imag * pyo.cos(d))
        return p, q

    if limits:
        for table, e, row, f, t, s in branch_rows:
            if not (s > 0 and math.isfinite(s)):
                continue
            for end, bus in (('f', f), ('t', t)):
                p, q = flow(row, f, t, end)
                m.rate.add(p ** 2 + q ** 2 <= (s / base * m.V[bus]) ** 2)

    # Objective.
    grid_p = sum(m.Pg[g] for g in grids) + sum(m.Pgen[g] for g in gens)
    stored = BAT_PRICE * (sum(m.Bd[k] for k in bat_ids)
                          + sum(m.Pv[k] - srcs[k][4] / base for k in range(len(srcs)) if srcs[k][1] == 'SOFC'))
    shed = sum((1 - m.La[e]) * float(net.res_load.at[e, 'p_mw']) / base for e in ac_loads) + \
        sum((1 - m.Ld[e]) * (float(net.res_load_dc.at[e, 'p_dc_mw']) if e in net.res_load_dc.index else float(net.load_dc.at[e, 'p_dc_mw'])) / base
            for e in dloads)
    spread = sum((1 - m.Ld[e]) ** 2 for e in dloads) + sum((1 - m.La[e]) ** 2 for e in ac_loads)
    m.obj = pyo.Objective(expr=grid_p + stored + (SHED_WEIGHT * shed + 0.01 * spread if mode == 'shed' else 0))

    if dropped:
        warnings.append(f'{len(dropped)} injection(s) on buses outside the modelled network were left out: '
                        + ', '.join(str(w) for w, _ in dropped[:5]) + '.')
    ipopt = _ipopt_path()
    if not ipopt:
        raise RuntimeError('Ipopt was not found: set ELECTRISIM_IPOPT to ipopt.exe, or unzip it into tools/ipopt beside the backend.')
    res = pyo.SolverFactory('ipopt', executable=ipopt).solve(m, options={'max_iter': 3000, 'tol': 1e-7})
    status = str(res.solver.termination_condition)
    ok = pyo.check_optimal_termination(res)
    v = lambda x: float(pyo.value(x))

    out: Dict[str, Any] = {'status': status, 'solved': bool(ok), 'objective': mode, 'warnings': warnings,
                           'branch_limits': limits, 'batteries': batteries, 'transformer_factor': tf}
    if not ok:
        out['message'] = ('No dispatch meets every constraint' + (' with the branch ratings enforced' if limits else '')
                          + (': try least shed, letting the batteries discharge, or a higher transformer factor.' if mode != 'shed' else '.'))
        return out
    lf_grid = float(net.res_ext_grid.loc[grids, 'p_mw'].sum()) if grids else 0.0
    out['grid'] = {'p_mw': round(sum(v(m.Pg[g]) for g in grids) * base, 4), 'q_mvar': round(sum(v(m.Qg[g]) for g in grids) * base, 4),
                   'load_flow_p_mw': round(lf_grid, 4)}
    out['generators'] = [{'id': _cid(net.gen, g), 'name': _label(net, net.gen.at[g, 'name']), 'p_mw': round(v(m.Pgen[g]) * base, 4),
                          'q_mvar': round(v(m.Qgen[g]) * base, 4)} for g in gens]
    out['loads_dc'] = [{'id': _cid(net.load_dc, e), 'name': _label(net, net.load_dc.at[e, 'name']), 'served_share': round(v(m.Ld[e]), 4),
                        'p_full_mw': round((float(net.res_load_dc.at[e, 'p_dc_mw']) if e in net.res_load_dc.index else float(net.load_dc.at[e, 'p_dc_mw'])), 4)}
                       for e in dloads]
    for d in out['loads_dc']:
        d['served_mw'] = round(d['p_full_mw'] * d['served_share'], 4)
        d['shed_mw'] = round(d['p_full_mw'] - d['served_mw'], 4)
    out['loads_ac'] = [{'id': _cid(net.load, e), 'name': _label(net, net.load.at[e, 'name']), 'served_share': round(v(m.La[e]), 4),
                        'p_full_mw': round(float(net.res_load.at[e, 'p_mw']), 4)} for e in ac_loads]
    for d in out['loads_ac']:
        d['shed_mw'] = round(d['p_full_mw'] * (1 - d['served_share']), 4)
    out['shed_mw'] = round(sum(d['shed_mw'] for d in out['loads_dc']) + sum(d['shed_mw'] for d in out['loads_ac']), 4)
    out['pcs'] = [{'id': str(r.get('id') or ''), 'name': r.get('label'), 'p_mw': round((v(m.Po[k]) - v(m.Pc[k])) * base, 4), 'q_mvar': round(v(m.Qi[k]) * base, 4),
                   'loading_percent': round(math.hypot(v(m.Po[k]) - v(m.Pc[k]), v(m.Qi[k])) * base / float(r.get('s_rated') or 1) * 100, 2)}
                  for k, r in enumerate(pcs)]
    out['afes'] = [{'id': _cid(net.vsc, a), 'name': _label(net, net.vsc.at[a, 'name']), 'p_mw': round(v(m.Pa[a]) * base, 4), 'q_mvar': round(v(m.Qa[a]) * base, 4),
                    'loading_percent': round(math.hypot(v(m.Pa[a]), v(m.Qa[a])) * base / afe_rating[a] * 100, 2)} for a in afe]
    out['ssts'] = [{'id': str(s.get('id') or ''), 'name': s.get('label'), 'p_lvdc_mw': round(v(m.Ps[k]) * base, 4), 'p_mv_mw': round(v(mv) * base, 4)}
                   for k, (s, mv) in enumerate(sst_info)]
    out['batteries_mw'] = round(sum(v(m.Bd[k]) - v(m.Bc[k]) for k in bat_ids) * base, 4)
    out['dcdc'] = [{'id': str(c.get('id') or ''), 'name': c.get('label'), 'p_mw': round((v(m.Df[k]) - v(m.Dr[k])) * base, 4),
                    'loading_percent': round(abs(v(m.Df[k]) - v(m.Dr[k])) * base / rated(c) * 100, 2) if rated(c) else None}
                   for k, c in enumerate(convs)]
    out['diodes'] = [{'id': str(d.get('id') or ''), 'name': d.get('label'), 'p_mw': round(v(m.Dd[k]) * base, 4),
                      'i_ka': round(v(m.Dd[k]) * base / vn_dc(d['bus_from']), 4)} for k, d in enumerate(diodes)]
    out['sources'] = [{'name': d.get('label'), 'kind': kind, 'p_mw': round(v(m.Pv[k]) * base, 4), 'p_max_mw': round(hi_s, 4)}
                      for k, (d, kind, lo_s, hi_s, _) in enumerate(srcs)]
    out['fuel_cells'] = fuel_cells
    vmap = {i: v(m.V[i]) for i in rows}
    out['buses'] = [{'id': _cid(net.bus, b), 'name': _label(net, net.bus.at[b, 'name']), 'vm_pu': round(vmap[ix(b)], 4)}
                    for b in net.bus.index if ix(b) in vmap and not pe._electrisim_is_aux(net.bus, b)]
    branches = []
    for table, e, row, f, t, s in branch_rows:
        sf = math.hypot(*(v(x) for x in flow(row, f, t, 'f'))) / vmap[f]
        st_ = math.hypot(*(v(x) for x in flow(row, f, t, 't'))) / vmap[t]
        branches.append({'kind': 'Line' if table == 'line' else 'Transformer', 'id': _cid(net[table], e), 'name': _label(net, net[table].at[e, 'name']),
                         'loading_percent': round(max(sf, st_) * base / s * 100, 2) if s > 0 else None})
    out['branches'] = branches
    return out


def acdc_opf_study(net, params: Dict[str, Any], in_data=None) -> str:
    """The study's JSON reply: Electrisim's load flow first, then the AC/DC optimal power flow on it."""
    try:
        pe._electrisim_runpp(net, algorithm='nr', calculate_voltage_angles=True, init='auto')
    except Exception as e:  # noqa: BLE001 - said, not raised
        return json.dumps({'error': True, 'message': f'The load flow it starts from did not solve: {e}'})
    bad = _unsupported(net)
    if bad:
        return json.dumps({'error': True, 'message': 'This study does not model yet: ' + ', '.join(bad[:8])
                           + (f' and {len(bad) - 8} more' if len(bad) > 8 else '') + '.'})
    try:
        out = build_and_solve(net, params)
    except Exception as e:  # noqa: BLE001
        return json.dumps({'error': True, 'message': f'The AC/DC optimal power flow could not be built: {e}'})
    out['warnings'] = list(getattr(net, 'warnings', []) or []) + out.get('warnings', [])
    return json.dumps(out, default=float)
