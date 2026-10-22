"""
Electrisim <-> mgtwin Site: the interchange between a drawn network and the twin's Site record.

* site_to_spec(site, profile=...)   a Site -> an Electrisim network spec (what electrisim_sld / the MCP server draw)
* payload_to_site(payload, ...)     a drawn diagram (the app's payload) -> a Site, as mgtwin reads it
* apply_state(payload, site, id)    a Site's operating state read onto the drawn payload: its switches (open and
                                    closed, normally open ones open), its sources in or out of service, its loads
                                    scaled by its load factors - as mgtwin v0.6.10 reads a state

A Site carries what the twin reasons on (topology, ratings, own-rating impedances, sources and their control,
grounding, states, relays); a spec carries what a solver needs on top of it (a battery's DC voltage, a fuel cell's
ramp, a dispatch). What the Site does not say, the importer's profile supplies, each assumption listed in its notes:

* 'generic'  storage converters (energy_mwh) as grid-forming PCS behind a battery of that energy; converters
             without storage as static generators (current sources, k = their transient current limit) sharing
             the load; machines as generators out of service (a state brings them in).
* 'c003'     the C003 microgrid's mapping (specs/make_c003_spec.py): PCST5500 grid-forming PCS behind LTO packs,
             the SOFC groups as fuel cells behind grid-following inverters.

Transformer impedances: mgtwin's x_pu / r_pu are on the transformer's own rating; a Site that gives an x above
0.25 pu (mgtwin's PLAUSIBLE["tx_x_pu"]) gave it on the system base, and it is rebased.
"""
from __future__ import annotations

import copy
import math

TX_X_OWN_MAX = 0.25            # mgtwin's PLAUSIBLE["tx_x_pu"] upper bound: an x above it is on the system base
LOSS_ALLOWANCE = 0.01          # dispatch only; the load flow computes the real losses


# --- Site -> spec -------------------------------------------------------------------------------------------------
def site_to_spec(site, profile='generic', life='eol', p_group_eol=1.91, it_bank=1):
    """A Site (dict, as mgtwin writes it) -> (spec, notes). `life`, `p_group_eol`, `it_bank`: the c003 profile's."""
    if profile not in ('generic', 'c003'):
        raise ValueError(f"unknown profile {profile!r} (generic, c003)")
    base = site.get('base') or {}
    s_base = float(base.get('s_base_mva', 50.0))
    f_hz = float(base.get('f_base_hz', 50.0))
    buses = {b['id']: float(b['v_nom_kv']) for b in site['buses']}
    notes = []
    spec = {'name': f"{'C003 ' if profile == 'c003' else ''}{site.get('id', '')} - {site.get('name', '')}"[:160],
            'frequency_hz': f_hz, 'layout': 'transmission',
            'buses': [{'id': b, 'vn_kv': v, 'name': b} for b, v in buses.items()]}

    # Branches -> lines (the whole branch impedance on 1 km).
    lines, branch_ends = [], {}
    for br in site.get('branches', []):
        sb, vb = float(br.get('s_base_mva', s_base)), float(br['v_base_kv'])
        zb = vb * vb / sb
        lines.append({'id': br['id'], 'name': br['id'], 'from_bus': br['from_bus'], 'to_bus': br['to_bus'],
                      'length_km': 1.0, 'r_ohm_per_km': round(float(br['r_pu']) * zb, 6),
                      'x_ohm_per_km': round(float(br['x_pu']) * zb, 6), 'c_nf_per_km': 0.0,
                      'max_i_ka': round(float(br['rating_mva']) / (math.sqrt(3) * vb), 4)})
        branch_ends[frozenset((br['from_bus'], br['to_bus']))] = br
    spec['lines'] = lines

    # Transformers: on their own rating; an x above the band is on the system base.
    trafos, rebased = [], 0
    for t in site.get('transformers', []):
        sn = float(t['s_rated_mva'])
        r_own, x_own = float(t['r_pu']), float(t['x_pu'])
        if x_own > TX_X_OWN_MAX:
            r_own, x_own = r_own * sn / s_base, x_own * sn / s_base
            rebased += 1
        if profile == 'c003' and str(t['id']).startswith('TR_IT') and it_bank > 1:
            notes.append(f"{t['id']}: the Site's one {sn:g} MVA unit drawn as a bank of {it_bank} ({sn * it_bank:g} MVA)")
            sn *= it_bank
        lv = str(t.get('lv_config', 'wye_grounded'))
        vg = 'Dyn' if lv == 'wye_grounded' else 'Dy'
        trafos.append({'id': t['id'], 'name': t['id'], 'hv_bus': t['hv_bus'], 'lv_bus': t['lv_bus'], 'sn_mva': sn,
                       'vk_percent': round(100 * math.hypot(r_own, x_own), 4), 'vkr_percent': round(100 * r_own, 4),
                       'vector_group': vg, 'shift_degree': 330})
    if trafos:
        notes.append(f"{len(trafos)} transformers, {rebased} given on the {s_base:g} MVA system base and rebased; "
                     f"{trafos[0]['id']}: vk {trafos[0]['vk_percent']:.2f} %, vkr {trafos[0]['vkr_percent']:.3f} % on its own rating")
    spec['transformers'] = trafos

    # Switches: a switch between the same two buses as a branch is that branch's breaker; normally open drawn open.
    switches = []
    for sw in site.get('switches', []):
        pair = frozenset((sw['from_bus'], sw['to_bus']))
        closed = bool(sw.get('normally_closed', True))
        if pair in branch_ends:
            br = branch_ends[pair]
            switches.append({'id': sw['id'], 'name': sw['id'], 'bus': br['from_bus'], 'element': br['id'], 'et': 'line',
                             'closed': closed})
        else:
            switches.append({'id': sw['id'], 'name': sw['id'], 'bus': sw['from_bus'], 'element': sw['to_bus'], 'et': 'bus',
                             'closed': closed})
    spec['switches'] = switches

    # Loads (constant power, as the Site's ZIP [0, 0, 1]; the critical flag in the name).
    spec['loads'] = [{'id': ld['id'], 'name': ld['id'] + (' (critical)' if ld.get('critical') else ''), 'bus': ld['bus'],
                      'p_mw': float(ld['p_mw']), 'q_mvar': float(ld.get('q_mvar') or 0.0)} for ld in site.get('loads', [])]
    p_load = sum(ld['p_mw'] for ld in spec['loads'])

    # Converters.
    pcs, batteries, sofcs, sgens = [], [], [], []
    groups = [c for c in site.get('converters', []) if not c.get('energy_mwh')]
    if profile == 'c003':
        p_group = p_group_eol if life == 'eol' else 2.2
        p_share = min(p_load * (1 + LOSS_ALLOWANCE) / max(len(groups), 1), p_group)
    else:
        p_share = None
    for c in site.get('converters', []):
        prm = c.get('params') or {}
        if c.get('energy_mwh'):
            bat = f"{c['id']}_BAT"
            e_kwh = float(c['energy_mwh']) * 1e3
            if profile == 'c003':
                batteries.append({'id': bat, 'name': f"{c['id']} LTO {e_kwh:g} kWh", 'vn_v': 1300, 'capacity_kwh': e_kwh,
                                  'r0_mohm': 20, 'soc_percent': 60, 'c_rate_discharge': 10, 'c_rate_charge': 10})
                pcs.append({'id': c['id'], 'name': f"{c['id']} PCST5500", 'bus': c['bus'], 'source': bat,
                            'control': 'grid_forming', 's_rated_mva': float(c['s_rated_mva']), 'vn_ac_kv': buses[c['bus']],
                            'p_set_mw': 0, 'droop_pf_percent': 100 * float(prm.get('p_f_droop_pu') or 0.04),
                            'droop_qv_percent': 100 * float(prm.get('q_v_droop_pu') or 0.05),
                            'inlet_c': 40, 'derate_temp_curve': '40:1.00, 45:0.909',
                            'current_limit_pu': float(prm.get('i_limit_transient_pu') or 1.1)})
            else:
                batteries.append({'id': bat, 'name': f"{c['id']} battery {e_kwh:g} kWh", 'capacity_kwh': e_kwh})
                pcs.append({'id': c['id'], 'name': f"{c['id']} PCS", 'bus': c['bus'], 'source': bat,
                            'control': 'grid_following' if c.get('control') == 'grid_following' else 'grid_forming',
                            's_rated_mva': float(c['s_rated_mva']), 'vn_ac_kv': buses[c['bus']], 'p_set_mw': 0,
                            'droop_pf_percent': 100 * float(prm.get('p_f_droop_pu') or 0.04),
                            'droop_qv_percent': 100 * float(prm.get('q_v_droop_pu') or 0.05),
                            'current_limit_pu': float(prm.get('i_limit_transient_pu') or 1.1)})
        elif profile == 'c003':
            fc = f"{c['id']}_FC"
            sofcs.append({'id': fc, 'name': f"{c['id']} SOFC", 'p_rated_kw': round(p_group * 1e3, 1), 'v_rated': 800,
                          'p_set_kw': round(p_share * 1e3, 1), 'min_load_percent': 30, 'ramp_percent_s': 1,
                          'aux_load_percent': 0})
            pcs.append({'id': c['id'], 'name': f"{c['id']} inverter", 'bus': c['bus'], 'source': fc,
                        'control': 'grid_following', 's_rated_mva': float(c['s_rated_mva']), 'vn_ac_kv': buses[c['bus']],
                        'efficiency_percent': 100, 'current_limit_pu': float(prm.get('i_limit_transient_pu') or 1.2)})
        else:
            sgens.append({'id': c['id'], 'name': f"{c['id']} converter", 'bus': c['bus'], 'sn_mva': float(c['s_rated_mva']),
                          'p_mw': 0.0, 'q_mvar': 0.0, 'k': float(prm.get('i_limit_transient_pu') or 1.2)})
    if profile == 'c003':
        notes.append(f"{len(groups)} SOFC groups at {p_group} MW ({life}), each dispatched at {p_share:.4f} MW for {p_load:g} MW of load")
        spec['pcs'], spec['batteries'], spec['sofcs'] = pcs, batteries, sofcs
    else:
        if sgens:
            share = min(p_load * (1 + LOSS_ALLOWANCE) / sum(s['sn_mva'] for s in sgens), 1.0)
            for s in sgens:
                s['p_mw'] = round(s['sn_mva'] * share, 6)
            notes.append(f"{len(sgens)} converters without storage as static generators, {100 * share:.1f} % of their "
                         f"rating each (ASSUMED: the Site gives no dispatch)")
        if batteries:
            notes.append(f"{len(batteries)} storage converters behind batteries of their energy_mwh (DC data ASSUMED: "
                         f"Electrisim's defaults)")
        spec['pcs'], spec['batteries'], spec['static_generators'] = pcs, batteries, sgens

    # Machines: out of service, as an islanded Site's NORMAL state commonly has them; a state brings them in.
    spec['generators'] = [{'id': m['id'], 'name': f"{m['id']} genset {m['s_rated_mva']:g} MVA", 'bus': m['bus'], 'p_mw': 0.0,
                           'vm_pu': 1.0, 'sn_mva': float(m['s_rated_mva']), 'xdss_pu': float(m.get('xd_subtransient_pu') or 0.15),
                           'cos_phi': 0.8, 'dyn_machine_model': 'GENROU', 'dyn_H': float(m.get('h_sec') or 1.0),
                           'dyn_gov_R': float(m.get('droop_pu') or 0.05), 'dyn_exciter_model': 'SEXS', 'dyn_governor_model': 'TGOV1',
                           'in_service': False} for m in site.get('machines', [])]

    # Grounding: a zig-zag + NER per low-resistance-grounded bus.
    gts = []
    for g in site.get('grounding', []):
        if g.get('kind') != 'low_resistance_grounded':
            continue
        r = float(g['z0_ohm'])
        i_a = buses[g['bus']] * 1e3 / math.sqrt(3) / r
        gts.append({'id': g['id'], 'name': f"{g['id']} zig-zag NER {r:g} ohm", 'bus': g['bus'], 'r_n_ohm': r,
                    'i_rated_a': round(i_a), 't_rated_s': 10})
        notes.append(f"{g['id']} at {g['bus']}: NER {r:g} ohm, {i_a:.1f} A")
    spec['grounding_transformers'] = gts
    return spec, notes


# --- payload -> Site ----------------------------------------------------------------------------------------------
def _kind(cell):
    t = str(cell.get('typ', ''))
    for k in ('Grounding Transformer', 'Transformer', 'Bus', 'Line', 'Switch', 'Load', 'Generator', 'PCS', 'Battery',
              'Static Generator', 'External Grid'):
        if t.startswith(k):
            return k
    return None


def _first(name):
    return str(name or '').split(' ')[0]


def payload_to_site(payload, site_id='DRAWN', s_base_mva=50.0, template=None):
    """
    A drawn diagram -> a Site (dict, as mgtwin reads it), in the diagram's normal state: switches as drawn (normally
    closed when drawn closed), every element named by the first word of its name. Transformer impedances on their own
    rating; branches on `s_base_mva`.

    A diagram holds no operating states or relays, and less about a converter than a Site does (grid forming, not
    droop or VSM; no inertia, ride-through or firmware). `template` - a Site of the same network - lends them: its
    islands and relays (those whose sources and switches the diagram has), and for each converter or machine it
    shares, its class and parameters, the drawn values taking precedence where the diagram has them. Without one the
    Site carries the drawn state alone (AS_DRAWN) and converters as droop grid-forming or grid-following.
    """
    cells = {k: v for k, v in payload.items() if isinstance(v, dict) and _kind(v)}
    by_cell = {v.get('name'): v for v in cells.values()}
    bus_id = {v['name']: str(v.get('userFriendlyName') or v['name']) for v in cells.values() if _kind(v) == 'Bus'}
    kv = {bus_id[n]: float(v['vn_kv']) for n, v in by_cell.items() if n in bus_id}
    f_hz = float((payload.get('0') or {}).get('frequency') or 50.0)
    site = {'id': site_id, 'name': f'drawn in Electrisim ({site_id})',
            'base': {'s_base_mva': s_base_mva, 'v_base_kv': max(kv.values()) if kv else 1.0, 'f_base_hz': f_hz},
            'buses': [{'id': b, 'v_nom_kv': v} for b, v in kv.items()],
            'branches': [], 'transformers': [], 'switches': [], 'relays': [], 'converters': [], 'machines': [],
            'loads': [], 'grounding': [], 'islands': []}
    lines = {}
    for v in cells.values():
        if _kind(v) != 'Line':
            continue
        f, t = bus_id[v['busFrom']], bus_id[v['busTo']]
        vb = kv[f]
        zb = vb * vb / s_base_mva
        L = float(v.get('length_km') or 1.0)
        br = {'id': str(v.get('userFriendlyName') or v['name']), 'from_bus': f, 'to_bus': t,
              'r_pu': round(float(v['r_ohm_per_km']) * L / zb, 8), 'x_pu': round(float(v['x_ohm_per_km']) * L / zb, 8),
              'b_pu': 0.0, 's_base_mva': s_base_mva, 'v_base_kv': vb,
              'rating_mva': round(float(v.get('max_i_ka') or 0) * math.sqrt(3) * vb, 4)}
        lines[v['name']] = br
        site['branches'].append(br)
    for v in cells.values():
        k = _kind(v)
        if k == 'Transformer':
            vk, vkr = float(v['vk_percent']) / 100, float(v['vkr_percent']) / 100
            vg = str(v.get('vector_group') or 'Dyn')
            hv, lv = bus_id[v['hv_bus']], bus_id[v['lv_bus']]
            site['transformers'].append({
                'id': str(v.get('userFriendlyName') or v['name']), 'hv_bus': hv, 'lv_bus': lv,
                's_rated_mva': float(v['sn_mva']), 'hv_kv': kv[hv], 'lv_kv': kv[lv],
                'x_pu': round(math.sqrt(max(vk * vk - vkr * vkr, 0.0)), 6), 'r_pu': round(vkr, 8),
                'hv_config': 'delta' if vg.upper().startswith('D') else 'wye_grounded',
                'lv_config': 'wye_grounded' if vg.lower().endswith('n') else ('delta' if vg[1:2].lower() == 'd' else 'wye_ungrounded')})
        elif k == 'Switch':
            f = bus_id.get(v['bus'])
            if str(v.get('et')) in ('l', 'line'):
                br = lines.get(v['element'])
                t = (br['to_bus'] if br['from_bus'] == f else br['from_bus']) if br else None
            else:
                t = bus_id.get(v['element'])
            if f and t:
                site['switches'].append({'id': str(v.get('userFriendlyName') or v.get('name')), 'from_bus': f, 'to_bus': t,
                                         'normally_closed': str(v.get('closed', 'true')).lower() == 'true'})
        elif k == 'Load':
            site['loads'].append({'id': _first(v.get('userFriendlyName')), 'bus': bus_id[v['bus']],
                                  'p_mw': float(v['p_mw']), 'q_mvar': float(v.get('q_mvar') or 0.0),
                                  'critical': '(critical)' in str(v.get('userFriendlyName'))})
        elif k == 'Generator':
            site['machines'].append({'id': _first(v.get('userFriendlyName')), 'bus': bus_id[v['bus']],
                                     's_rated_mva': float(v['sn_mva']), 'h_sec': float(v.get('dyn_H') or 1.0),
                                     'xd_subtransient_pu': float(v.get('xdss_pu') or 0.15),
                                     'droop_pu': float(v.get('dyn_gov_R') or 0.05)})
        elif k == 'PCS':
            src = by_cell.get(v.get('der')) or {}
            gfm = str(v.get('control')) == 'grid_forming'
            conv = {'id': _first(v.get('userFriendlyName')), 'bus': bus_id[v['bus']], 's_rated_mva': float(v['s_rated_mva']),
                    'control': 'grid_forming_droop' if gfm else 'grid_following',
                    'params': {'i_limit_transient_pu': float(v.get('current_limit_pu') or 1.2),
                               **({'p_f_droop_pu': float(v.get('droop_pf_percent') or 4) / 100,
                                   'q_v_droop_pu': float(v.get('droop_qv_percent') or 5) / 100} if gfm else {})}}
            if _kind(src) == 'Battery':
                conv['energy_mwh'] = float(src.get('capacity_kwh') or 0) / 1e3
            site['converters'].append(conv)
        elif k == 'Grounding Transformer':
            site['grounding'].append({'id': _first(v.get('userFriendlyName')), 'bus': bus_id[v['bus']],
                                      'kind': 'low_resistance_grounded', 'z0_ohm': float(v.get('r_n_ohm') or 0.0),
                                      'available_when_islanded': True})
    # in id order: a diagram's cells come in whatever order its keys were written
    for key in ('buses', 'branches', 'transformers', 'switches', 'converters', 'machines', 'loads', 'grounding'):
        site[key].sort(key=lambda x: str(x['id']))
    if template is not None:
        ids = {c['id'] for c in site['converters']} | {m['id'] for m in site['machines']}
        sw_ids = {s['id'] for s in site['switches']}
        site['islands'] = [copy.deepcopy(i) for i in template.get('islands', [])
                           if set(i.get('energized_source_ids', [])) <= ids
                           and set(i.get('open_switches', [])) | set(i.get('closed_switches', [])) <= sw_ids]
        site['relays'] = [copy.deepcopy(r) for r in template.get('relays', []) if r['switch_id'] in sw_ids]
        gfm = lambda c: str(c.get('control', '')).startswith('grid_forming')  # noqa: E731
        conv_t = {c['id']: c for c in template.get('converters', [])}
        for c in site['converters']:
            t = conv_t.get(c['id'])
            if t is None:
                continue
            for k, v in t.items():
                if k not in c:
                    c[k] = copy.deepcopy(v)              # vendor, firmware, sign, ...
            c['params'] = {**copy.deepcopy(t.get('params') or {}), **c['params']}
            if gfm(t) == gfm(c):
                c['control'] = t['control']             # droop or VSM: the diagram says only grid forming
        # earthing a diagram does not draw - a declared ungrounded (IT) pocket, a solid neutral - lent by the template
        drawn = {g['id'] for g in site['grounding']}
        buses = {b['id'] for b in site['buses']}
        site['grounding'] += [copy.deepcopy(g) for g in template.get('grounding', [])
                              if g['id'] not in drawn and g['bus'] in buses]
        site['grounding'].sort(key=lambda x: str(x['id']))
        mach_t = {m['id']: m for m in template.get('machines', [])}
        for m in site['machines']:
            for k, v in (mach_t.get(m['id']) or {}).items():
                m.setdefault(k, copy.deepcopy(v))
    else:
        on = [_first(v.get('userFriendlyName')) for v in cells.values() if _kind(v) in ('PCS', 'Generator')
              and str(v.get('in_service', 'true')).lower() == 'true']
        has_grid = any(_kind(v) == 'External Grid' for v in cells.values())
        site['islands'] = [{'id': 'AS_DRAWN', 'description': 'the diagram as drawn', 'grid_connected': has_grid,
                            'energized_source_ids': on}]
    return site


# --- a Site state on the drawn payload ----------------------------------------------------------------------------
def apply_state(payload, site, state_id, typ=None, extra=None, load_scale=None, also_off=()):
    """
    The drawn payload in one of the Site's operating states: as mgtwin reads a state, the payload draws normally open
    switches open; the state opens its open_switches and closes its closed_switches (a transfer's second legs, a tie);
    its sources are in or out of service (a converter's or machine's cell named by its id's first word); its
    load_factors scale its loads - a load_scale given here replaces the factor for that load. `typ` / `extra` set the
    study (cell '0'), `also_off` takes further sources out.
    """
    state = next(s for s in site['islands'] if s['id'] == state_id)
    sources = {c['id'] for c in site.get('converters', [])} | {m['id'] for m in site.get('machines', [])}
    on = set(state['energized_source_ids']) - set(also_off)
    opened = set(state.get('open_switches') or [])
    closes = set(state.get('closed_switches') or [])
    scale = {**(state.get('load_factors') or {}), **(load_scale or {})}
    p = copy.deepcopy(payload)
    if typ is not None or extra:
        p['0'] = {**p['0'], **({'typ': typ} if typ is not None else {}), **(extra or {})}
    for v in p.values():
        if not isinstance(v, dict):
            continue
        name, t = str(v.get('userFriendlyName', '')), str(v.get('typ', ''))
        if t.startswith('Switch') and name in opened:
            v['closed'] = 'false'
        if t.startswith('Switch') and name in closes:
            v['closed'] = 'true'
        if (t.startswith('PCS') or t.startswith('Generator')) and _first(name) in sources:
            v['in_service'] = 'true' if _first(name) in on else 'false'
        if t.startswith('Generator') and _first(name) in on:
            v['slack'] = 'true'
        if t.startswith('Load') and _first(name) in scale:
            k = scale[_first(name)]
            v['p_mw'], v['q_mvar'] = str(float(v['p_mw']) * k), str(float(v['q_mvar']) * k)
    return p
