# Electrisim network spec

A network is one JSON object. Every element has an `id`; elements refer to each
other by id, never by position. Ids must be unique across the whole spec, not
just within a list. `name` is the label drawn on the diagram (defaults to the id).

Units: kV, MW, MVAr, MVA, km, per-unit (`_pu`), percent (`_percent`), degrees.

## Top level

| key | meaning |
|---|---|
| `name` | network name |
| `frequency_hz` | 50 (default) or 60 |
| `layout` | `transmission`, `radial` or `auto` - see Layout below |
| `buses`, `external_grids`, `transformers`, `lines`, `loads`, `generators`, `static_generators`, `shunts`, `storage`, `switches` | lists of elements |

Any other top-level key is rejected.

## Elements

Required fields are **bold**. Everything else has the default shown.

**buses** - `id`, **`vn_kv`**, `name`, `in_service` (true).
A bus has no default voltage; it must always be given.

**external_grids** - the connection to the wider grid, and the power-flow slack.
`id`, **`bus`**, `vm_pu` (1.0), `va_degree` (0), `s_sc_max_mva` (none - give it
for short-circuit studies), `rx_max`.

**transformers** - two-winding.
`id`, **`hv_bus`**, **`lv_bus`**, `sn_mva` (25), `vn_hv_kv` / `vn_lv_kv` (taken
from the two buses), `vk_percent` (by rating: up to 2.5 MVA -> 4, up to 10 -> 6,
up to 40 -> 12, above -> 14), `vkr_percent` (vk / 25), `pfe_kw` (0.6 x sn_mva),
`i0_percent` (0.1), `shift_degree` (0), `name`, `in_service`.
A warning is raised if `hv_bus` is the lower voltage.

**lines** - joins two buses at the **same** voltage; use a transformer between
voltage levels.
`id`, **`from_bus`**, **`to_bus`**, `length_km` (1.0), `name`, `in_service`, and
either
- `std_type`, a pandapower line type. Default by voltage: up to 1 kV
  `NAYY 4x150 SE`; up to 30 kV `NA2XS2Y 1x240 RM/25 12/20 kV`; up to 150 kV
  `149-AL1/24-ST1A 110.0`; above `490-AL1/64-ST1A 380.0`. Or
- explicit impedance: `r_ohm_per_km` (0.1), `x_ohm_per_km` (0.1),
  `c_nf_per_km` (0), `max_i_ka` (0.4). Giving `r` or `x` selects this form.

**loads** - `id`, **`bus`**, `p_mw` (0), `q_mvar` (0.33 x p_mw, about pf 0.95),
`name`, `in_service`.

**generators** - voltage-controlled synchronous machines.
`id`, **`bus`**, `p_mw` (0), `vm_pu` (1.0), `sn_mva` (1.2 x p_mw, at least 1),
`slack` (false - set true on one generator if there is no external grid),
`name`, `in_service`.

**static_generators** - inverter-connected PV, wind, batteries in PQ mode.
`id`, **`bus`**, `p_mw` (0), `q_mvar` (0), `name`, `in_service`.
On a radial layout a static generator whose `name` contains "wind" or
"turbine" is drawn as a wind turbine. `kind: "wind"` with a name that does not
say so raises a warning rather than renaming it.

**shunts** - `id`, **`bus`**, **`q_mvar`** (positive absorbs - a reactor;
negative injects - a capacitor bank), `p_mw` (0), `name`, `in_service`.

**storage** - `id`, **`bus`**, `p_mw` (0; positive charges, negative discharges),
`max_e_mwh` (1.0), `name`, `in_service`.

**switches** - `id`, **`bus`**, **`element`**, `et` (`line` default,
`transformer`, or `bus`), `closed` (true), `name`.
- `et: line` / `transformer`: `element` is that line's or transformer's id, and
  `bus` must be one of its two terminals.
- `et: bus`: a bus coupler; `element` is the other bus's id.

## Layout

- `transmission` - meshed networks, IEEE-style test cases, anything with loops.
  Buses are banded by voltage.
- `radial` - a plant or feeder single-line: grid at the top, feeders hanging in
  columns below, machines at the feeder ends.
- `auto` (default) - Electrisim guesses the way the import dialog's "Other"
  button does: more than 60 buses or many loops means transmission, else radial.

## Example

A 110/20 kV substation feeding two 20 kV feeders, one with an industrial load,
one with a wind farm.

```json
{
  "name": "Example substation",
  "layout": "radial",
  "buses": [
    {"id": "HV", "vn_kv": 110, "name": "110 kV busbar"},
    {"id": "MV", "vn_kv": 20, "name": "20 kV busbar"},
    {"id": "F1", "vn_kv": 20, "name": "Feeder 1"},
    {"id": "F2", "vn_kv": 20, "name": "Feeder 2"}
  ],
  "external_grids": [{"id": "Grid", "bus": "HV", "vm_pu": 1.02, "s_sc_max_mva": 2000}],
  "transformers": [{"id": "T1", "hv_bus": "HV", "lv_bus": "MV", "sn_mva": 25}],
  "lines": [
    {"id": "L1", "from_bus": "MV", "to_bus": "F1", "length_km": 1.2},
    {"id": "L2", "from_bus": "MV", "to_bus": "F2", "length_km": 4.5}
  ],
  "loads": [{"id": "Plant", "bus": "F1", "p_mw": 2.0, "name": "Industrial load"}],
  "static_generators": [{"id": "WF", "bus": "F2", "p_mw": 5.0, "name": "Wind farm"}],
  "switches": [
    {"id": "CB1", "bus": "MV", "element": "L1", "et": "line"},
    {"id": "CB2", "bus": "MV", "element": "L2", "et": "line"}
  ]
}
```

## Errors

A spec that cannot be built is rejected with every problem listed at once, each
naming the element and the fix, e.g.

```
lines[0] (L1): to_bus='F9' is not a bus id. Defined buses: HV, MV, F1, F2
```

Fix them all and resubmit; there is no need to resubmit after each one.
