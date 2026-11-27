"""
Convert a microgrid site model (the detroit_site_v4.json schema) into an
Electrisim network spec.

    python convert_site_json.py <site.json> [out.spec.json]

Judgment calls, each checkable against the source:

* Transformer r_pu / x_pu are on the site's system base (base.s_base_mva), not
  each transformer's rating. On their own ratings they would be 28.75 % for a
  2 MVA unit; converted from the system base every transformer lands on a
  textbook 5.75 % or 6.0 %. The conversion prints both so this stays visible.
* A switch between the same two buses as a branch is that branch's breaker, in
  series with it at the from-bus end - not a coupler in parallel, which would
  short the cable out. Switches with no branch beside them are bus couplers.
* Operating points the source does not give (converter and machine dispatch)
  are 0 MW, so the power flow is the loads fed from the grid.
* Not carried, because the spec has no place for them: transformer vector
  groups and winding grounding, zig-zag grounding transformers, converter
  control parameters, ZIP load models (all [0,0,1] - constant power - here),
  and the island scenarios (the drawing is the all-closed GRID case).
"""

import json
import math
import sys
from pathlib import Path


def convert(site):
    base = site.get('base') or {}
    s_base = float(base.get('s_base_mva', 10.0))
    notes = []

    spec = {
        'name': site.get('name') or site.get('id') or 'Site',
        'frequency_hz': float(base.get('f_base_hz', 50.0)),
        # Bus couplers are invisible to the radial layout's walk, and this site
        # reaches its first switchgear through one.
        'layout': 'transmission',
        'buses': [{'id': b['id'], 'vn_kv': float(b['v_nom_kv'])} for b in site['buses']],
    }

    # --- the grid --------------------------------------------------------
    grid_buses = [g['bus'] for g in site.get('grounding', [])
                  if not g.get('available_when_islanded', True)]
    if not grid_buses:
        raise SystemExit('cannot tell which bus is the utility connection')
    spec['external_grids'] = [{'id': 'Grid', 'bus': grid_buses[0], 'vm_pu': 1.0,
                               'name': f'Utility at {grid_buses[0]}'}]

    # --- branches -> lines (per unit -> ohms) ----------------------------
    lines = []
    branch_ends = {}
    for br in site.get('branches', []):
        sb = float(br.get('s_base_mva', s_base))
        vb = float(br['v_base_kv'])
        z_base = vb * vb / sb
        lines.append({
            'id': br['id'], 'from_bus': br['from_bus'], 'to_bus': br['to_bus'],
            'length_km': 1.0,  # impedances below are the whole branch
            'r_ohm_per_km': round(float(br['r_pu']) * z_base, 6),
            'x_ohm_per_km': round(float(br['x_pu']) * z_base, 6),
            'c_nf_per_km': 0.0,
            'max_i_ka': round(float(br['rating_mva']) / (math.sqrt(3) * vb), 4),
        })
        branch_ends[frozenset((br['from_bus'], br['to_bus']))] = br
    spec['lines'] = lines

    # --- transformers (system-base per unit -> own-rating percent) --------
    trafos = []
    for t in site.get('transformers', []):
        sn = float(t['s_rated_mva'])
        r_own = float(t['r_pu']) * sn / s_base
        x_own = float(t['x_pu']) * sn / s_base
        vk = 100 * math.hypot(r_own, x_own)
        notes.append(f"{t['id']}: x {float(t['x_pu']):.4f} pu on {s_base:g} MVA -> "
                     f"{100 * x_own:.2f} % on its own {sn:g} MVA "
                     f"(vk {vk:.3f} %, vkr {100 * r_own:.3f} %)")
        trafos.append({
            'id': t['id'], 'hv_bus': t['hv_bus'], 'lv_bus': t['lv_bus'], 'sn_mva': sn,
            'vn_hv_kv': float(t['hv_kv']), 'vn_lv_kv': float(t['lv_kv']),
            'vk_percent': round(vk, 4), 'vkr_percent': round(100 * r_own, 4),
            # No-load data is not in the source; leave it out rather than invent it.
            'pfe_kw': 0.0, 'i0_percent': 0.0,
        })
    spec['transformers'] = trafos

    # --- switches --------------------------------------------------------
    switches = []
    for sw in site.get('switches', []):
        pair = frozenset((sw['from_bus'], sw['to_bus']))
        closed = bool(sw.get('normally_closed', True))
        if pair in branch_ends:
            br = branch_ends[pair]
            switches.append({'id': sw['id'], 'bus': br['from_bus'], 'element': br['id'],
                             'et': 'line', 'closed': closed})
            notes.append(f"{sw['id']}: breaker on {br['id']} at {br['from_bus']} "
                         f"(same buses as the branch - series, not parallel)")
        else:
            switches.append({'id': sw['id'], 'bus': sw['from_bus'], 'element': sw['to_bus'],
                             'et': 'bus', 'closed': closed})
    spec['switches'] = switches

    # --- loads -----------------------------------------------------------
    spec['loads'] = [{'id': ld['id'], 'bus': ld['bus'], 'p_mw': float(ld['p_mw']),
                      'q_mvar': float(ld.get('q_mvar') or 0.0)}
                     for ld in site.get('loads', [])]

    # --- sources (no dispatch in the source: 0 MW) ------------------------
    storage, sgens = [], []
    for c in site.get('converters', []):
        if c.get('energy_mwh'):
            storage.append({'id': c['id'], 'bus': c['bus'], 'p_mw': 0.0,
                            'max_e_mwh': float(c['energy_mwh']),
                            'name': f"{c['id']} {c['s_rated_mva']:g} MVA / {c['energy_mwh']:g} MWh"})
        else:
            sgens.append({'id': c['id'], 'bus': c['bus'], 'p_mw': 0.0, 'q_mvar': 0.0,
                          'name': f"{c['id']} {c['s_rated_mva']:g} MVA"})
    spec['storage'] = storage
    spec['static_generators'] = sgens
    spec['generators'] = [{'id': m['id'], 'bus': m['bus'], 'p_mw': 0.0, 'vm_pu': 1.0,
                           'sn_mva': float(m['s_rated_mva']),
                           'name': f"{m['id']} {m['s_rated_mva']:g} MVA"}
                          for m in site.get('machines', [])]
    return spec, notes


def main():
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix('.spec.json')
    spec, notes = convert(json.loads(src.read_text(encoding='utf-8')))
    out.write_text(json.dumps(spec, indent=2), encoding='utf-8')
    for n in notes:
        print(n)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
