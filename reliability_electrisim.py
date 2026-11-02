"""
Reliability of an islanded network by analytical contingency enumeration, from its mgtwin Site.

The network is cut into protection zones by its breakers (each connected set of buses at or above zone_kv, joined
by branches and by switches that carry no CT, a source transformer's LV terminal with it). Each zone fails at the
sum of its elements' rates - its bus section, its transformers (each with its tap), its cables per km - and every
zone fault is followed through its clearing:

  zone      - cleared by its own breakers: the zone is lost for its repair
  stuck     - one of its breakers fails to open (p_bf): breaker failure opens the zone beyond it too, until the
              operators isolate the stuck breaker and re-close it (restore_min)
  relay     - its main protection fails (p_rf; duplicated mains p_rf (beta + (1 - beta) p_rf)): the backup clears
              it after the grid-following sources' ride-through, so they trip (backup_trips_gfl); the surviving
              forming sources recover the voltage only up to recoverable_factor x their MVA, the rest shed until
              the fleet returns from hot standby (t_rt_min) or cold-restarts (p_cold, t_cold_h)
  maint     - it faults while a breaker is out for maintenance (t_maint_h per breaker per year): both open
  blackout  - a total blackout (lam_bo) whose black start of black_start_steps steps fails (p_step): all IT for t_man_h

Each event's consequence is an island balance after the switching: an island with no forming source is dead; one
with forming sources serves its load up to its sources' power - grid-following sources run only beside a forming
one; transfer groups (dual-corded IT) move a dead cord's load to its partner when the partner's island has room. A
deficit lasts the event's duration, less what the island's storage (storage_mwh per forming source) carries; IT and
auxiliary unserved energy are priced at their values of lost load.

The rates are IEEE 493-class planning values, every one a parameter. This is the first-order enumeration a planning
study uses, not a sequential Monte Carlo: it does not follow storage state of charge through a year.
"""
import math

DEFAULTS = {
    'zone_kv': 10.0, 'no_ct_prefixes': ('RS_',),
    'lam_tx': 0.010, 'lam_zone': 0.001, 'lam_cab_km': 0.020, 'p_bf': 0.003, 'p_rf': 0.01, 'dup_87': True,
    'beta_cc': 0.10, 't_rep_h': 48.0, 'restore_min': 5.0, 't_maint_h': 8.0, 'lam_bo': 0.05, 'p_step': 0.02,
    'black_start_steps': 3, 't_man_h': 2.0, 'p_cold': 0.10, 't_cold_h': 9.0, 't_rt_min': 2.0, 'p_vrt': 1.0,
    'backup_trips_gfl': True, 'recoverable_factor': 0.84, 'ups_min': 0.0, 'storage_mwh': 0.254,
    'forming_storage_backed': True, 'aux_shed_max_h': 4.0, 'maint_restore_h': 4.0,
    'p_forming_mw': None, 'p_following_mw': None, 'voll_it': 50000.0, 'voll_aux': 500.0,
    'it_loads': None, 'transfer_groups': (), 'cable_km': None, 'grid_buses': (), 'include_maintenance': True,
    # per element, by its Site id: {'rate': per year for the whole element, 'repair_h': h, 'p_bf': per demand}
    'element_data': None,
}


def _uf(items):
    parent = {x: x for x in items}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def join(a, b):
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    return find, join


def zones(site, p):
    """The protection zones: [{'id', 'buses', 'kind' 87T|87L|87B, 'members' (breakers), 'tx', 'branches'}]."""
    kv = {b['id']: b['v_nom_kv'] for b in site['buses']}
    mv = {b for b, v in kv.items() if v >= p['zone_kv'] and not b.endswith('_STUB')}
    find, join = _uf(mv)
    no_ct = tuple(p['no_ct_prefixes'])
    for br in site['branches']:
        join(br['from_bus'], br['to_bus'])
    for w in site['switches']:
        if w['id'].startswith(no_ct):
            join(w['from_bus'], w['to_bus'])
    comps = {}
    for b in mv:
        comps.setdefault(find(b), []).append(b)
    out = []
    for buses in comps.values():
        buses = buses + [t['lv_bus'] for t in site['transformers'] if t['hv_bus'] in set(buses) and t['lv_bus'].endswith('_LV_T')]
        bs = set(buses)
        members = sorted(w['id'] for w in site['switches'] if not w['id'].startswith(no_ct)
                         and (w['from_bus'] in bs) != (w['to_bus'] in bs))
        if not members:
            continue
        tx = sorted(t['id'] for t in site['transformers'] if t['hv_bus'] in bs)
        branches = sorted(br['id'] for br in site['branches'] if br['from_bus'] in bs and br['to_bus'] in bs)
        cable = bool(branches) or any(w['id'].startswith(no_ct) for w in site['switches'] if w['from_bus'] in bs)
        out.append({'id': '+'.join(sorted(buses))[:60], 'buses': sorted(buses), 'members': members, 'tx': tx,
                    'branches': branches, 'kind': '87T' if tx and not cable else '87L' if cable else '87B'})
    return sorted(out, key=lambda z: z['buses'])


class Network:
    """The Site in an operating state, as islands after any set of switches opened and buses lost."""

    def __init__(self, site, state, p):
        self.site, self.p = site, p
        self.open0 = set(state.get('open_switches') or [])
        self.closed0 = set(state.get('closed_switches') or [])
        self.on = set(state.get('energized_source_ids') or [])
        self.forming = {c['id'] for c in site['converters'] if 'forming' in str(c.get('control'))} | \
                       {m['id'] for m in site.get('machines', [])}
        self.loads = {l['id']: l for l in site['loads']}
        it = p['it_loads']
        self.it = set(it) if it is not None else {l['id'] for l in site['loads'] if l.get('critical') and l['p_mw'] > 5}
        self.source_bus = {c['id']: c['bus'] for c in site['converters']} | {m['id']: m['bus'] for m in site.get('machines', [])}
        self.converter_ids = {c['id'] for c in site['converters']}
        self.grid = set(p['grid_buses']) | {b['id'] for b in site['buses'] if b.get('is_poi')} \
            if state.get('grid_connected') else set()

    def _p(self, sid):
        conv = next((c for c in self.site['converters'] if c['id'] == sid), None)
        if sid in self.forming:
            if self.p['p_forming_mw'] is not None and conv is not None:
                return self.p['p_forming_mw']
            return (conv or next(m for m in self.site['machines'] if m['id'] == sid))['s_rated_mva']
        return self.p['p_following_mw'] if self.p['p_following_mw'] is not None else conv['s_rated_mva']

    def islands(self, opened=(), dead_buses=(), tripped=()):
        """[{'buses', 'forming': [ids], 'following': [ids], 'loads': [ids]}] after opening switches, losing buses
        (a cleared zone) and tripping sources."""
        dead = set(dead_buses)
        buses = [b['id'] for b in self.site['buses'] if b['id'] not in dead]
        find, join = _uf(buses)
        opened = set(opened) | self.open0
        for br in self.site['branches']:
            join(br['from_bus'], br['to_bus'])
        for t in self.site['transformers']:
            join(t['hv_bus'], t['lv_bus'])
        for w in self.site['switches']:
            closed = (w.get('normally_closed', True) or w['id'] in self.closed0) and w['id'] not in opened
            if closed:
                join(w['from_bus'], w['to_bus'])
        isl = {}
        for b in buses:
            isl.setdefault(find(b), {'buses': [], 'forming': [], 'following': [], 'loads': [], 'grid': False})['buses'].append(b)
            if b in self.grid:
                isl[find(b)]['grid'] = True
        alive = set(buses)
        for sid in self.on - set(tripped):
            b = self.source_bus.get(sid)
            if b in alive:
                isl[find(b)]['forming' if sid in self.forming else 'following'].append(sid)
        for lid, l in self.loads.items():
            if l['bus'] in alive:
                isl[find(l['bus'])]['loads'].append(lid)
        return [i for i in isl.values() if i['loads'] or i['forming'] or i['following']]

    def balance(self, islands, hours, recoverable=None, storage=True):
        """
        What the islands do not serve over ``hours``: {'it_mwh', 'aux_mwh', 'it_mw', 'aux_mw'} (the energy with
        the islands' storage bridging; the MW the gap left, for a cold restart). An island with no forming source
        is dead. With forming_storage_backed (battery PCS), only the grid-following sources and the machines
        sustain an island; its forming units bridge the gap up to their power from their storage
        (storage_mwh each). recoverable: after a fleet trip, the forming units hold that share of their MVA and
        nothing bridges (the shed is power-limited). storage=False: the storage is already spent (the later part
        of an event). The auxiliaries are shed first when the interruption is
        no longer than aux_shed_max_h; transfer groups move a dead member's load to a live one.
        """
        p = self.p
        load_mw = {lid: l['p_mw'] for lid, l in self.loads.items()}
        isl_of = {lid: i for i in islands for lid in i['loads']}
        for group in p['transfer_groups']:
            live = [l for l in group if l in isl_of and (isl_of[l]['forming'] or isl_of[l].get('grid'))]
            dead_m = [l for l in group if l in load_mw and l not in live]
            if live and dead_m:
                moved = sum(load_mw[l] for l in dead_m)
                for l in dead_m:
                    load_mw[l] = 0.0
                load_mw[live[0]] += moved
        shed_aux = hours <= p['aux_shed_max_h']
        out = {'it_mwh': 0.0, 'aux_mwh': 0.0, 'it_mw': 0.0, 'aux_mw': 0.0}
        # Each island's sustained supply, bridging power and storage.
        supply = {}
        for i in islands:
            if i.get('grid'):
                supply[id(i)] = (math.inf, 0.0, 0.0)
                continue
            if not i['forming']:
                continue
            machines = [x for x in i['forming'] if x not in self.converter_ids]
            units = [x for x in i['forming'] if x in self.converter_ids]
            following = sum(self._p(x) for x in i['following'])
            if recoverable is not None:
                supply[id(i)] = (recoverable * sum(self._p(x) for x in i['forming']) + following, 0.0, 0.0)
            elif p['forming_storage_backed']:
                supply[id(i)] = (following + sum(self._p(x) for x in machines), sum(self._p(x) for x in units),
                                 p['storage_mwh'] * len(units))
            else:
                supply[id(i)] = (following + sum(self._p(x) for x in i['forming']), 0.0, 0.0)
        # A dual-corded load also shifts from a short island to its partner's island while that one has room.
        for group in p['transfer_groups']:
            live = [l for l in group if l in isl_of and id(isl_of[l]) in supply and load_mw[l] > 0]
            for short_l in live:
                isl_s = isl_of[short_l]
                need = sum(load_mw[l] for l in isl_s['loads']) - supply[id(isl_s)][0]
                if need <= 1e-9:
                    continue
                for other in live:
                    isl_o = isl_of[other]
                    if isl_o is isl_s:
                        continue
                    room = supply[id(isl_o)][0] - sum(load_mw[l] for l in isl_o['loads'])
                    move = max(0.0, min(need, room, load_mw[short_l]))
                    if move > 1e-9:
                        load_mw[short_l] -= move
                        load_mw[other] += move
                        need -= move
        for i in islands:
            demand_it = sum(load_mw[l] for l in i['loads'] if l in self.it)
            demand_aux = sum(load_mw[l] for l in i['loads'] if l not in self.it)
            if id(i) not in supply:
                out['it_mw'] += demand_it
                out['aux_mw'] += demand_aux
                out['it_mwh'] += demand_it * hours
                out['aux_mwh'] += demand_aux * hours
                continue
            sustained, bridge, energy = supply[id(i)]
            if not storage:
                energy = 0.0
            gap = max(0.0, demand_it + demand_aux - sustained)
            if gap <= 1e-9:
                continue
            # storage first: if it carries the whole gap for the whole time, nothing is lost
            if gap <= bridge + 1e-9 and gap * hours <= energy + 1e-9:
                continue
            a = min(gap, demand_aux) if shed_aux else 0.0
            g = gap - a
            out['aux_mw'] += a
            out['aux_mwh'] += a * hours
            out['it_mw'] += g
            out['it_mwh'] += max(0.0, min(g, bridge) * hours - energy) + max(0.0, g - bridge) * hours
        # Loads whose bus is lost (in the cleared zone itself) and not moved to a partner are not served.
        for lid in self.loads:
            if lid not in isl_of and load_mw[lid] > 0:
                key = 'it' if lid in self.it else 'aux'
                out[f'{key}_mw'] += load_mw[lid]
                out[f'{key}_mwh'] += load_mw[lid] * hours
        return out


def enumerate_events(site, params=None, state_id='NORMAL'):
    """
    Every event: {'class', 'zone', 'detail', 'freq', 'it_hot', 'it_cold', 'aux', 'it_mw', 'aux_mw'} - its IT
    energy not served with the sources returning hot and if they cold-restart, its auxiliary energy, and the gap
    it opens. A zone fault: what the clearing cuts off is restored after restore_min (normally-open ties closed,
    load-break switches re-closed around the faulted zone), the zone itself after its repair (t_rep_h). A stuck
    breaker or a maintenance overlap counts only what it adds to its zone fault, until restored.
    """
    p = dict(DEFAULTS, **(params or {}))
    state = next(i for i in site['islands'] if i['id'] == state_id)
    net = Network(site, state, p)
    zs = zones(site, p)
    km = p['cable_km'] or {}
    breakers = sorted({m for z in zs for m in z['members']})
    normally_open = {w['id'] for w in site['switches'] if not w.get('normally_closed', True)} - net.closed0
    p_main = p['p_rf'] * ((p['beta_cc'] + (1 - p['beta_cc']) * p['p_rf']) if p['dup_87'] else 1.0)
    h_rest = p['restore_min'] / 60
    events = []

    data = p['element_data'] or {}

    def own(x, key):
        v = (data.get(x) or {}).get(key)
        return None if v is None else float(v)

    def parts(z):
        """A zone's rate in parts: what its elements' own rates give, and what is left at the defaults - its bus
        section (unless one of its buses has its own rate), its transformers and its cables' km without one."""
        own_rate = sum(own(x, 'rate') for x in z['buses'] + z['tx'] + z['branches'] if own(x, 'rate') is not None)
        bus_default = 0 if any(own(b, 'rate') is not None for b in z['buses']) else 1
        tx_default = sum(1 for t in z['tx'] if own(t, 'rate') is None)
        km_default = sum(km.get(b, 0.0) for b in z['branches'] if own(b, 'rate') is None)
        return own_rate, bus_default, tx_default, km_default

    def rate(z):
        own_rate, bus_d, tx_d, km_d = parts(z)
        return own_rate + p['lam_zone'] * bus_d + p['lam_tx'] * tx_d + p['lam_cab_km'] * km_d

    def repair_h(z):
        given = [own(x, 'repair_h') for x in z['buses'] + z['tx'] + z['branches'] if own(x, 'repair_h') is not None]
        return max(given) if given else p['t_rep_h']

    def p_bf(m):
        v = own(m, 'p_bf')
        return p['p_bf'] if v is None else v

    def loss(opened, dead, hours, **kw):
        return net.balance(net.islands(opened=opened, dead_buses=dead, tripped=kw.get('tripped', ())), hours,
                           recoverable=kw.get('recoverable'))

    def extra(a, b):
        return {k: max(0.0, a[k] - b[k]) for k in a}

    by_member = {}
    for z in zs:
        for m in z['members']:
            by_member.setdefault(m, []).append(z)
    for z in zs:
        lam = rate(z)
        h_rep = repair_h(z)
        opened = set(z['members'])
        # what the drawing highlights and what the workbook rebuilds the rate from
        own_rate, bus_d, tx_d, km_d = parts(z)
        ref = {'buses': z['buses'], 'breakers': z['members'], 'n_tx': len(z['tx']),
               'cable_km': round(sum(km.get(b, 0.0) for b in z['branches']), 4), 'repair_h': h_rep,
               'own_rate': own_rate, 'bus_default': bus_d, 'tx_default': tx_d, 'km_default': round(km_d, 4)}
        # zone: cut off until restored, the zone itself until repaired
        first = loss(opened, z['buses'], h_rest)
        after = net.balance(net.islands(opened=opened, dead_buses=z['buses']) if not normally_open else
                            _with_ties(net, opened, z['buses'], normally_open), max(h_rep - h_rest, 0.0), storage=False)
        cold = loss(opened, z['buses'], p['t_cold_h'])
        events.append({**ref, 'also': [], 'class': 'zone', 'zone': z['id'], 'detail': z['kind'], 'freq': lam,
                       'it_mw': first['it_mw'], 'aux_mw': first['aux_mw'], 'it_hot': first['it_mwh'] + after['it_mwh'],
                       'it_cold': cold['it_mwh'] + after['it_mwh'], 'aux': first['aux_mwh'] + after['aux_mwh']})
        # stuck: breaker failure opens the zone beyond each member until it is isolated and re-closed
        for m in z['members']:
            beyond = {x for y in by_member[m] if y is not z for x in y['members']}
            hot = extra(loss(opened | beyond, z['buses'], h_rest), first)
            cold_x = extra(loss(opened | beyond, z['buses'], p['t_cold_h']), cold)
            events.append({**ref, 'also': sorted(beyond | {m}), 'class': 'stuck', 'zone': z['id'], 'detail': m,
                           'freq': lam * p_bf(m), 'p_bf_own': own(m, 'p_bf'),
                           'it_mw': hot['it_mw'], 'aux_mw': hot['aux_mw'], 'it_hot': hot['it_mwh'],
                           'it_cold': cold_x['it_mwh'], 'aux': hot['aux_mwh']})
        # relay: the backup clears it after the ride-through; the grid-following fleet trips; the forming units
        # hold recoverable x their MVA until the fleet is back (hot standby, less any UPS ride-through) or cold
        if p['backup_trips_gfl']:
            following = [x for x in net.on if x not in net.forming]
            h_rt = max(0.0, p['t_rt_min'] - p['ups_min']) / 60
            hot = extra(loss(opened, z['buses'], h_rt, tripped=following, recoverable=p['recoverable_factor']),
                        loss(opened, z['buses'], h_rt))
            cold_x = extra(loss(opened, z['buses'], p['t_cold_h'], tripped=following, recoverable=p['recoverable_factor']), cold)
            aux_rt = extra(loss(opened, z['buses'], p['t_rt_min'] / 60, tripped=following, recoverable=p['recoverable_factor']),
                           loss(opened, z['buses'], p['t_rt_min'] / 60))
            events.append({**ref, 'also': [], 'class': 'relay', 'zone': z['id'], 'detail': 'backup clears, fleet trips',
                           'freq': lam * p_main * p['p_vrt'], 'it_mw': hot['it_mw'], 'aux_mw': aux_rt['aux_mw'],
                           'it_hot': hot['it_mwh'], 'it_cold': cold_x['it_mwh'], 'aux': aux_rt['aux_mwh']})
        # maint: the zone faults while a breaker elsewhere is out for maintenance, until that breaker is back
        h_m = p['maint_restore_h']
        base_m = loss(opened, z['buses'], h_m) if p['include_maintenance'] else None
        for b in (breakers if p['include_maintenance'] else ()):
            if b in z['members']:
                continue
            hot = extra(loss(opened | {b}, z['buses'], h_m), base_m)
            if hot['it_mwh'] > 1e-9 or hot['aux_mwh'] > 1e-9:
                cold_x = extra(loss(opened | {b}, z['buses'], p['t_cold_h']), cold)
                events.append({**ref, 'also': [b], 'class': 'maint', 'zone': z['id'], 'detail': b,
                               'freq': lam * p['t_maint_h'] / 8760,
                               'it_mw': hot['it_mw'], 'aux_mw': hot['aux_mw'], 'it_hot': hot['it_mwh'],
                               'it_cold': cold_x['it_mwh'], 'aux': hot['aux_mwh']})
    it_total = sum(l['p_mw'] for l in site['loads'] if l['id'] in net.it)
    events.append({'buses': [], 'breakers': [], 'also': [], 'n_tx': 0, 'cable_km': 0.0, 'own_rate': 0.0,
                   'bus_default': 0, 'tx_default': 0, 'km_default': 0.0, 'repair_h': None,
                   'class': 'blackout', 'zone': '-', 'detail': f"black start of {p['black_start_steps']} steps fails",
                   'freq': p['lam_bo'] * (1 - (1 - p['p_step']) ** p['black_start_steps']), 'it_mw': it_total, 'aux_mw': 0.0,
                   'it_hot': it_total * p['t_man_h'], 'it_cold': it_total * p['t_man_h'], 'aux': 0.0})
    return events, zs, p


def _with_ties(net, opened, dead, normally_open):
    """The islands once the normally-open ties are closed around a cleared zone (a transfer)."""
    keep = net.closed0
    net.closed0 = keep | normally_open
    try:
        return net.islands(opened=opened, dead_buses=dead)
    finally:
        net.closed0 = keep


def evaluate(site, params=None, state_id='NORMAL'):
    """Expected IT and auxiliary energy not served (MWh/yr), its cost, and the events by class."""
    events, zs, p = enumerate_events(site, params, state_id)
    for e in events:
        e['it_mwh_yr'] = e['freq'] * ((1 - p['p_cold']) * e['it_hot'] + p['p_cold'] * e['it_cold'])
        e['aux_mwh_yr'] = e['freq'] * e['aux']
    by_class = {}
    for e in events:
        c = by_class.setdefault(e['class'], {'events': 0, 'freq': 0.0, 'it_mwh_yr': 0.0, 'aux_mwh_yr': 0.0})
        c['events'] += 1
        c['freq'] += e['freq']
        c['it_mwh_yr'] += e['it_mwh_yr']
        c['aux_mwh_yr'] += e['aux_mwh_yr']
    it = sum(e['it_mwh_yr'] for e in events)
    aux = sum(e['aux_mwh_yr'] for e in events)
    return {'it_mwh_yr': it, 'aux_mwh_yr': aux, 'cost_keur_yr': (it * p['voll_it'] + aux * p['voll_aux']) / 1000,
            'zones': len(zs), 'events_enumerated': len(events), 'by_class': by_class,
            'inputs': {k: v for k, v in p.items() if k in WORKBOOK_INPUTS},
            'events': sorted((e for e in events if e['it_mwh_yr'] > 1e-9 or e['aux_mwh_yr'] > 1e-9),
                             key=lambda e: -(e['it_mwh_yr'] * p['voll_it'] + e['aux_mwh_yr'] * p['voll_aux']))}


#: the inputs the workbook carries as editable cells, with their units and what they are
WORKBOOK_INPUTS = {
    'lam_zone': ('/yr', 'bus section of a zone'), 'lam_tx': ('/yr', 'transformer with its tap'),
    'lam_cab_km': ('/km-yr', 'cable'), 'p_bf': ('per demand', 'breaker fails to open'),
    'p_rf': ('per demand', 'main protection fails to operate'), 'dup_87': ('1 / 0', 'duplicated mains'),
    'beta_cc': ('', 'common-cause share of main-protection failures'),
    'p_vrt': ('', 'share of backup clearances in which the grid-following fleet trips'),
    't_maint_h': ('h/yr', 'maintenance per breaker'), 'lam_bo': ('/yr', 'total blackout'),
    'p_step': ('per step', 'black-start step fails'), 'black_start_steps': ('', 'black-start steps'),
    'p_cold': ('', 'share of events whose sources cold-restart'), 'voll_it': ('EUR/MWh', 'value of lost IT load'),
    'voll_aux': ('EUR/MWh', 'value of lost auxiliary load'),
}


def workbook(result, title='Reliability'):
    """
    The study as an Excel workbook of formulas (bytes): Inputs (the rates and probabilities, editable), Events (each
    event's frequency rebuilt from the inputs - its zone's bus section, transformers and cable km - and its
    consequences as the enumeration found them), Summary (by class, and the cost). Edit a rate and the workbook
    recalculates, as the hand-built models do.
    """
    import io
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    head = Font(bold=True, color='FFFFFF')
    fill = PatternFill('solid', fgColor='1F6F5C')
    edit = Font(color='0000FF')
    wb = Workbook()
    ws = wb.active
    ws.title = 'Inputs'
    ws['A1'] = f'{title} - inputs (blue cells are editable; the other sheets are formulas on them)'
    ws['A1'].font = Font(bold=True, size=12)
    for j, h in enumerate(['Input', 'Value', 'Unit', 'What it is'], 1):
        c = ws.cell(row=3, column=j, value=h)
        c.font, c.fill = head, fill
    ref = {}
    for r, (name, (unit, what)) in enumerate(WORKBOOK_INPUTS.items(), 4):
        v = result['inputs'].get(name)
        ws.cell(row=r, column=1, value=name)
        c = ws.cell(row=r, column=2, value=(1 if v else 0) if name == 'dup_87' else v)
        c.font = edit
        ws.cell(row=r, column=3, value=unit)
        ws.cell(row=r, column=4, value=what)
        ref[name] = f'Inputs!$B${r}'
    for j, w in enumerate([18, 12, 12, 52], 1):
        ws.column_dimensions[get_column_letter(j)].width = w

    ev = wb.create_sheet('Events')
    cols = ['Class', 'Zone', 'Detail', 'Own rates /yr', 'Bus section at default', 'Transformers at default',
            'Cable km at default', 'Breaker p_bf (own)', 'Frequency /yr', 'IT if hot MWh', 'IT if cold MWh', 'Aux MWh',
            'Expected IT MWh/yr', 'Expected aux MWh/yr', 'Zone breakers', 'Also opened']
    for j, h in enumerate(cols, 1):
        c = ev.cell(row=1, column=j, value=h)
        c.font, c.fill = head, fill
        c.alignment = Alignment(wrap_text=True)
    R = ref
    for r, e in enumerate(result['events'], 2):
        rate = f"(D{r}+{R['lam_zone']}*E{r}+{R['lam_tx']}*F{r}+{R['lam_cab_km']}*G{r})"
        freq = {'zone': f'={rate}', 'stuck': f"={rate}*IF(ISNUMBER(H{r}),H{r},{R['p_bf']})",
                'relay': f"={rate}*{R['p_rf']}*IF({R['dup_87']}=1,{R['beta_cc']}+(1-{R['beta_cc']})*{R['p_rf']},1)*{R['p_vrt']}",
                'maint': f"={rate}*{R['t_maint_h']}/8760",
                'blackout': f"={R['lam_bo']}*(1-(1-{R['p_step']})^{R['black_start_steps']})"}[e['class']]
        row = [e['class'], e['zone'], e['detail'], e.get('own_rate', 0.0), e.get('bus_default', 1),
               e.get('tx_default', e['n_tx']), e.get('km_default', e['cable_km']), e.get('p_bf_own'), freq,
               round(e['it_hot'], 6), round(e['it_cold'], 6), round(e['aux'], 6),
               f"=I{r}*((1-{R['p_cold']})*J{r}+{R['p_cold']}*K{r})", f'=I{r}*L{r}',
               ' '.join(e['breakers']), ' '.join(e['also'])]
        for j, v in enumerate(row, 1):
            ev.cell(row=r, column=j, value=v)
    last = len(result['events']) + 1
    for j, w in enumerate([10, 34, 26, 12, 12, 12, 12, 12, 14, 13, 13, 11, 16, 16, 40, 30], 1):
        ev.column_dimensions[get_column_letter(j)].width = w
    ev.freeze_panes = 'D2'
    ev.cell(row=last + 2, column=1, value='Own rates: the elements\' own failure rates (per year for the whole element) where '
            'the drawing gives them; the defaults on the Inputs sheet cover the parts counted at default.')

    sm = wb.create_sheet('Summary')
    sm['A1'] = f'{title} - expected energy not served and its cost'
    sm['A1'].font = Font(bold=True, size=12)
    for j, h in enumerate(['Class', 'Events', 'Expected IT MWh/yr', 'Expected aux MWh/yr', 'Cost kEUR/yr'], 1):
        c = sm.cell(row=3, column=j, value=h)
        c.font, c.fill = head, fill
    classes = ['zone', 'stuck', 'relay', 'maint', 'blackout']
    for i, cl in enumerate(classes, 4):
        sm.cell(row=i, column=1, value=cl)
        sm.cell(row=i, column=2, value=f'=COUNTIF(Events!$A$2:$A${last},"{cl}")')
        sm.cell(row=i, column=3, value=f'=SUMIF(Events!$A$2:$A${last},"{cl}",Events!$M$2:$M${last})')
        sm.cell(row=i, column=4, value=f'=SUMIF(Events!$A$2:$A${last},"{cl}",Events!$N$2:$N${last})')
        sm.cell(row=i, column=5, value=f"=(C{i}*{R['voll_it']}+D{i}*{R['voll_aux']})/1000")
    t = 4 + len(classes)
    sm.cell(row=t, column=1, value='Total').font = Font(bold=True)
    for j in (2, 3, 4, 5):
        col = get_column_letter(j)
        sm.cell(row=t, column=j, value=f'=SUM({col}4:{col}{t - 1})').font = Font(bold=True)
    sm.cell(row=t + 2, column=1, value='Events with no energy not served are left out: their rates move nothing.')
    for j, w in enumerate([12, 10, 20, 20, 16], 1):
        sm.column_dimensions[get_column_letter(j)].width = w
    if result.get('comparison'):
        _comparison_sheets(wb, result, R, t, head, fill, edit)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _comparison_sheets(wb, result, R, total_row, head, fill, edit):
    """The comparison (Comparison: the base's energy from Summary, live; its cost, annualised capex, total, rank and
    break-even capex as formulas on the capital recovery factor and the capexes, both editable) and the sweep
    (Sweep: the totals, ranking and tie-points as the study found them)."""
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    cmp_ = result['comparison']
    ws = wb.create_sheet('Comparison')
    ws['A1'] = 'Designs on one table - lowest total first (blue cells are editable)'
    ws['A1'].font = Font(bold=True, size=12)
    ws['A2'] = 'Capital recovery factor /yr'
    ws['B2'] = cmp_['crf']
    ws['B2'].font = edit
    cols = ['Design', 'State', 'IT MWh/yr', 'Aux MWh/yr', 'Cost kEUR/yr', 'Capex MEUR', 'Annualised capex kEUR/yr',
            'Total kEUR/yr', 'Total vs base kEUR/yr', 'Rank', 'Break-even capex MEUR']
    for j, h in enumerate(cols, 1):
        c = ws.cell(row=4, column=j, value=h)
        c.font, c.fill = head, fill
    n = len(cmp_['designs'])
    last = 4 + n
    for r, d in enumerate(cmp_['designs'], 5):
        base = r == 5
        row = [d['name'], d['state'],
               f'=Summary!$C${total_row}' if base else round(d['it_mwh_yr'], 6),
               f'=Summary!$D${total_row}' if base else round(d['aux_mwh_yr'], 6),
               f"=(C{r}*{R['voll_it']}+D{r}*{R['voll_aux']})/1000", d['capex_meur'], f'=F{r}*$B$2*1000',
               f'=E{r}+G{r}', f'=H{r}-$H$5', f'=RANK(H{r},$H$5:$H${last},1)',
               f'=IF($B$2>0,($H$5-E{r})/($B$2*1000),"")']
        for j, v in enumerate(row, 1):
            c = ws.cell(row=r, column=j, value=v)
            if j == 6:
                c.font = edit
    ws.cell(row=last + 2, column=1, value='The base (first row) takes its energy not served from the Summary sheet, so it '
            'follows the Inputs; the other designs\' energy is as the study found it at these inputs.')
    ws.cell(row=last + 3, column=1, value='Break-even capex: the capex at which the design\'s total equals the base\'s.')
    for j, w in enumerate([22, 12, 12, 12, 14, 12, 16, 14, 16, 8, 18], 1):
        ws.column_dimensions[get_column_letter(j)].width = w

    sw = cmp_.get('sweep')
    if not sw:
        return
    names = [d['name'] for d in cmp_['designs']]
    ss = wb.create_sheet('Sweep')
    ss['A1'] = f"Sensitivity: {sw['param']} - {sw['label']} (base value {sw['base_value']})"
    ss['A1'].font = Font(bold=True, size=12)
    cols = [sw['param']] + [f'{n} total vs base kEUR/yr' for n in names] + ['Ranking (lowest total first)'] + \
           [f'{n} IT MWh/yr' for n in names]
    for j, h in enumerate(cols, 1):
        c = ss.cell(row=3, column=j, value=h)
        c.font, c.fill = head, fill
    for r, row in enumerate(sw['rows'], 4):
        vals = [row['value']] + [round(row['designs'][n]['total_vs_base_keur_yr'], 3) for n in names] + \
               [' < '.join(row['ranking'])] + [round(row['designs'][n]['it_mwh_yr'], 6) for n in names]
        for j, v in enumerate(vals, 1):
            ss.cell(row=r, column=j, value=v)
    t = 4 + len(sw['rows']) + 1
    ss.cell(row=t, column=1, value='Tie-points (where two designs\' totals cross; linear between the swept values)').font = Font(bold=True)
    if not sw['tie_points']:
        ss.cell(row=t + 1, column=1, value='None: the ranking holds over the swept range.')
    for i, tp in enumerate(sw['tie_points'], t + 1):
        ss.cell(row=i, column=1, value=round(tp['value'], 6))
        ss.cell(row=i, column=2, value=f"{tp['below']} is cheaper below, {tp['above']} above")
    ss.column_dimensions['A'].width = 16
    for j in range(2, len(cols) + 1):
        ss.column_dimensions[get_column_letter(j)].width = 18


#: the inputs a sensitivity sweep may move, with what they are; the pricing ones (values of lost load, the capital
#: recovery factor) only re-price the events, every other one re-runs the enumeration
SWEEPABLE = {
    'lam_zone': 'bus section of a zone (/yr)', 'lam_tx': 'transformer with its tap (/yr)',
    'lam_cab_km': 'cable (/km-yr)', 't_rep_h': 'repair time (h)', 'restore_min': 'restoration by switching (min)',
    'p_bf': 'breaker fails to open (per demand)', 'p_rf': 'main protection fails (per demand)',
    'dup_87': 'duplicated main protection (0 / 1)', 'beta_cc': 'common-cause share of main-protection failures',
    't_maint_h': 'maintenance per breaker (h/yr)', 'maint_restore_h': 'breaker back from maintenance (h)',
    'include_maintenance': 'maintenance overlap counted (0 / 1)', 'storage_mwh': 'storage per grid-forming unit (MWh)',
    'forming_storage_backed': 'grid-forming units storage-backed (0 / 1)', 'p_vrt': 'share of backup clearances that trip the fleet',
    'recoverable_factor': 'load the forming units recover (x MVA)', 't_rt_min': 'fleet back from hot standby (min)',
    'p_cold': 'share of events with a cold restart', 't_cold_h': 'cold restart (h)', 'ups_min': 'UPS ride-through (min)',
    'aux_shed_max_h': 'auxiliaries shed up to (h)', 'lam_bo': 'total blackouts (/yr)', 'p_step': 'black-start step fails',
    'black_start_steps': 'black-start steps', 't_man_h': 'manual recovery (h)',
    'voll_it': 'value of lost IT load (EUR/MWh)', 'voll_aux': 'value of lost auxiliary load (EUR/MWh)',
    'crf': 'capital recovery factor (/yr)',
}
PRICING = ('voll_it', 'voll_aux', 'crf')
BOOLEAN = ('dup_87', 'include_maintenance', 'forming_storage_backed', 'backup_trips_gfl')


def sweep_values(name, text, base):
    """The values a sweep runs: numbers, or multiples of the base value written x0.5 (blank: x0.5, x1, x2; a
    switch: off and on)."""
    if name in BOOLEAN:
        return [0.0, 1.0]
    tokens = [t.strip() for t in str(text or '').replace(';', ',').split(',') if t.strip()] or ['x0.5', 'x1', 'x2']
    out = []
    for t in tokens:
        out.append(float(t[1:]) * float(base) if t[0] in 'x*' else float(t))
    if name == 'black_start_steps':
        out = [float(round(v)) for v in out]
    return sorted(set(out))


def _priced(r, p, capex_meur):
    cost = (r['it_mwh_yr'] * p['voll_it'] + r['aux_mwh_yr'] * p['voll_aux']) / 1000
    capex_keur = (capex_meur or 0.0) * p['crf'] * 1000
    return {'it_mwh_yr': r['it_mwh_yr'], 'aux_mwh_yr': r['aux_mwh_yr'], 'cost_keur_yr': cost,
            'capex_keur_yr': capex_keur, 'total_keur_yr': cost + capex_keur}


def compare(designs, params=None, sweep=None, crf=0.08):
    """
    Several designs on one table, and one input swept across them. designs: [{'name', 'site', 'state',
    'capex_meur', 'params'}], the first the base. Each design's expected IT and auxiliary energy not served, their
    cost, its annualised capex (capex x crf), its total against the base and its rank (lowest total first), and the
    capex at which it would tie with the base; then, with sweep ({'param', 'values'}), the same totals and ranking at
    every value and the tie-points - where two designs' totals cross between two values (linear between them).
    """
    params = dict(params or {})
    base_p = dict(DEFAULTS, **params, crf=crf)
    names = [d['name'] for d in designs]

    def run(over):
        p = dict(base_p, **over)
        p_eval = {k: v for k, v in p.items() if k != 'crf'}
        out = {}
        for d in designs:
            if all(k in PRICING for k in over):
                r = first[d['name']]
            else:
                # the swept input holds for every design, over the inputs a design has of its own
                r = evaluate(d['site'], {**p_eval, **d.get('params', {}), **over}, d['state'])
            out[d['name']] = dict(_priced(r, p, d.get('capex_meur')), zones=r['zones'],
                                  events_enumerated=r['events_enumerated'])
        base_total = out[names[0]]['total_keur_yr']
        order = sorted(names, key=lambda n: out[n]['total_keur_yr'])
        for n in names:
            out[n]['total_vs_base_keur_yr'] = out[n]['total_keur_yr'] - base_total
            out[n]['rank'] = order.index(n) + 1
        return out, order

    first = {}
    for d in designs:
        first[d['name']] = evaluate(d['site'], dict({k: v for k, v in base_p.items() if k != 'crf'},
                                                    **d.get('params', {})), d['state'])
    table, order = run({})
    base = table[names[0]]
    rows = []
    for d in designs:
        t = table[d['name']]
        # the capex at which this design's total equals the base's
        t['break_even_capex_meur'] = ((base['total_keur_yr'] - t['cost_keur_yr']) / (crf * 1000)) if crf > 0 else None
        rows.append(dict(t, name=d['name'], state=d['state'], capex_meur=d.get('capex_meur') or 0.0,
                         it_loads=d.get('params', {}).get('it_loads'), notes=d.get('notes', [])))
    out = {'crf': crf, 'base': names[0], 'designs': rows, 'ranking': order}
    if sweep and sweep.get('param'):
        name = sweep['param']
        values = sweep['values']
        srows = []
        for v in values:
            val = bool(v) if name in BOOLEAN else (int(v) if name == 'black_start_steps' else v)
            res, rk = run({name: val})
            srows.append({'value': v, 'designs': res, 'ranking': rk})
        ties = []
        for a_i, a in enumerate(names):
            for b in names[a_i + 1:]:
                for r0, r1 in zip(srows, srows[1:]):
                    d0 = r0['designs'][a]['total_keur_yr'] - r0['designs'][b]['total_keur_yr']
                    d1 = r1['designs'][a]['total_keur_yr'] - r1['designs'][b]['total_keur_yr']
                    if d0 * d1 < 0:
                        x = r0['value'] + (r1['value'] - r0['value']) * d0 / (d0 - d1)
                        ties.append({'a': a, 'b': b, 'value': x,
                                     'below': a if d0 < 0 else b, 'above': b if d0 < 0 else a})
        out['sweep'] = {'param': name, 'label': SWEEPABLE.get(name, name), 'base_value': base_p.get(name),
                        'rows': srows, 'tie_points': ties}
    return out


def _design_notes(site, p):
    """What a design's inputs say that its drawing does not bear out: IT loads or dual-cord pairs it does not have."""
    have = {l['id'] for l in site['loads']}
    notes = []
    if p.get('it_loads') and not set(p['it_loads']) & have:
        notes.append('none of the IT loads named is on it')
    groups = p.get('transfer_groups') or ()
    if groups and not any(len(set(g) & have) > 1 for g in groups):
        notes.append('its dual-cord pairs name none of its loads: no cord moves to its partner')
    return notes


def _fit_it_loads(site, it_loads):
    """A design's IT loads: the names given that it has; none of them - its critical loads above 5 MW."""
    if not it_loads:
        return None
    have = {l['id'] for l in site['loads']}
    mine = [x for x in it_loads if x in have]
    return mine or None


def _drawn(payload, template=None, state_id=None):
    """A drawn network as a Site - its template's states and converter classes when one is given, else the state as
    drawn - with what the drawing itself says: its cables' lengths, its grid's buses, each element's own failure
    data (rel_failure_rate per year for the whole element, rel_repair_h, and a breaker's rel_p_fail_open per
    demand). Returns (site, state, those params, the payload's element cells, the cell naming)."""
    from mgtwin_site_io import _kind, payload_to_site

    site = payload_to_site(payload, 'DRAWN', template=template)
    state = state_id if template is not None and any(i['id'] == state_id for i in site['islands']) else 'AS_DRAWN'
    cells = [v for v in payload.values() if isinstance(v, dict) and _kind(v)]
    name = lambda v: str(v.get('userFriendlyName') or v.get('name'))  # noqa: E731 - the Site's own naming
    bus_id = {v['name']: name(v) for v in cells if _kind(v) == 'Bus'}
    drawn = {'cable_km': {name(v): float(v.get('length_km') or 0) for v in cells if _kind(v) == 'Line'},
             'grid_buses': [bus_id[v['bus']] for v in cells if _kind(v) == 'External Grid' and v.get('bus') in bus_id]}
    data = {}
    for v in cells:
        if _kind(v) not in ('Bus', 'Switch', 'Transformer', 'Line'):
            continue
        d = {}
        for attr, key in (('rel_failure_rate', 'rate'), ('rel_repair_h', 'repair_h'), ('rel_p_fail_open', 'p_bf')):
            try:
                if str(v.get(attr, '')).strip() != '':
                    d[key] = float(v[attr])
            except (TypeError, ValueError):
                pass
        if d:
            data[name(v)] = d
    if data:
        drawn['element_data'] = data
    elements = [v for v in cells if _kind(v) in ('Bus', 'Switch', 'Transformer', 'Line')]
    return site, state, drawn, elements, name


def study(payload, params=None, template=None, state_id=None, designs=None, sweep=None, crf=0.08,
          capex_meur=0.0, title='This page'):
    """
    The study on a drawn network: the payload (already in its operating state) evaluated; each event's buses and
    breakers as the drawing's cell ids; the live workbook as base64. With designs ([{'name', 'payload',
    'capex_meur', 'params'}], other drawings, each as drawn, params the inputs it has of its own) or a sweep
    ({'param', 'values'}), the comparison too - this drawing the base (its capex capex_meur), the others against it.
    Returns a dict for JSON.
    """
    import base64

    params = dict(params or {})
    if not params.get('it_loads'):
        params.pop('it_loads', None)
    site, state, drawn, elements, name = _drawn(payload, template, state_id)
    result = evaluate(site, dict(drawn, **params), state)
    cell_of = {name(v): v.get('id') for v in elements}
    for e in result['events']:
        e['cells'] = sorted({cell_of[x] for x in e['buses'] + e['breakers'] + e['also'] if cell_of.get(x)})
        e['also_cells'] = sorted({cell_of[x] for x in e['also'] if cell_of.get(x)})
    result['state'] = state
    if designs or (sweep and sweep.get('param')):
        user = {k: v for k, v in params.items() if k not in ('cable_km', 'grid_buses', 'element_data')}
        entries = [{'name': title, 'site': site, 'state': state, 'capex_meur': capex_meur, 'params': drawn}]
        for i, d in enumerate(designs or []):
            s, st, dr, _, _ = _drawn(d['payload'])
            fitted = _fit_it_loads(s, params.get('it_loads'))
            entries.append({'name': str(d.get('name') or f'Design {i + 2}'), 'site': s, 'state': st,
                            'capex_meur': float(d.get('capex_meur') or 0.0),
                            'params': {**dr, 'it_loads': fitted, **(d.get('params') or {})}})
        for e in entries:
            e['notes'] = _design_notes(e['site'], {**user, **e['params']})
        if sweep and sweep.get('param'):
            base = dict(DEFAULTS, crf=crf, **user).get(sweep['param'])
            sweep = {'param': sweep['param'], 'values': sweep_values(sweep['param'], sweep.get('values'), base)}
        result['comparison'] = compare(entries, user, sweep, crf)
    result['workbook_xlsx_b64'] = base64.b64encode(workbook(result, f'Reliability - {state}')).decode('ascii')
    result['error'] = False
    return result
