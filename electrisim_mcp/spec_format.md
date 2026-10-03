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
| `buses`, `external_grids`, `transformers`, `three_winding_transformers`, `lines`, `loads`, `generators`, `static_generators`, `shunts`, `storage`, `switches` | lists of elements |

Any other top-level key is rejected.

## Elements

Required fields are **bold**. Everything else has the default shown.

**buses** - `id`, **`vn_kv`**, `name`, `in_service` (true).
A bus has no default voltage; it must always be given.

**external_grids** - the connection to the wider grid, and the power-flow slack.
`id`, **`bus`**, `vm_pu` (1.0), `va_degree` (0), `s_sc_max_mva` (none - give it
for short-circuit studies; without it Electrisim treats the grid as an
infinite 1,000,000 MVA source), `rx_max`, `s_sc_min_mva` and `rx_min` (for a
minimum-case study; default to the maximum values), and zero sequence for
earth faults: `x0x_max` (X0/X, 1.0) and `r0x0_max` (R0/X0, 0.1), with `x0x_min`
and `r0x0_min` for the minimum case (default: the maximum values).

**transformers** - two-winding.
`id`, **`hv_bus`**, **`lv_bus`**, `sn_mva` (25), `vn_hv_kv` / `vn_lv_kv` (taken
from the two buses), `vk_percent` (by rating: up to 2.5 MVA -> 4, up to 10 -> 6,
up to 40 -> 12, above -> 14), `vkr_percent` (vk / 25), `pfe_kw` (0.6 x sn_mva),
`i0_percent` (0.1), `shift_degree` (0), `name`, `in_service`, and zero sequence
for earth faults: `vector_group` (`Dyn`), `vk0_percent` / `vkr0_percent` (the
positive-sequence values), `mag0_percent` (100), `mag0_rx` (0),
`si0_hv_partial` (0.9).
A warning is raised if `hv_bus` is the lower voltage.

**three_winding_transformers** - one unit joining three voltage levels, e.g.
110/20/10 kV with a tertiary for station supply.
`id`, **`hv_bus`**, **`mv_bus`**, **`lv_bus`** (three different buses),
`sn_hv_mva` (40), `sn_mv_mva` (= sn_hv_mva), `sn_lv_mva` (sn_hv_mva / 3),
`vn_hv_kv` / `vn_mv_kv` / `vn_lv_kv` (from the three buses), `pfe_kw`
(0.6 x sn_hv_mva), `i0_percent` (0.1), `shift_mv_degree` (0), `name`,
`in_service`, `vector_group` (`YNynd`: star-star with a delta tertiary; a
single-phase fault also accepts YNdyn, YNdd, YNyy, Dynyn - not the canvas
default YNyn0yn0), and short-circuit voltages **named by winding pair**:

| field | between | default |
|---|---|---|
| `vk_hv_mv_percent` | HV and MV | by the smaller rating of the pair, as for a two-winding transformer |
| `vk_mv_lv_percent` | MV and LV | likewise |
| `vk_hv_lv_percent` | HV and LV | likewise |
| `vkr_hv_mv_percent`, `vkr_mv_lv_percent`, `vkr_hv_lv_percent` | the same pairs | vk / 25 |
| `vk0_hv_mv_percent` ... `vkr0_hv_lv_percent` | zero sequence, the same pairs | the positive-sequence value |

Each percentage is referred to the smaller rated power of its pair.
pandapower's own names (`vk_hv_percent`, `vk_mv_percent`, `vk_lv_percent`)
are refused, because they mislead: its `vk_mv_percent` is MV-LV and its
`vk_lv_percent` is HV-LV. `shift_lv_degree` is refused too - the canvas cannot
keep it, so the drawn transformer would differ from the one checked.
A warning is raised unless the voltages run HV >= MV >= LV.

**lines** - joins two buses at the **same** voltage; use a transformer between
voltage levels.
`id`, **`from_bus`**, **`to_bus`**, `length_km` (1.0), `name`, `in_service`, and
either
- `std_type`, a pandapower line type. Default by voltage: up to 1 kV
  `NAYY 4x150 SE`; up to 30 kV `NA2XS2Y 1x240 RM/25 12/20 kV`; up to 150 kV
  `149-AL1/24-ST1A 110.0`; above `490-AL1/64-ST1A 380.0`. Or
- explicit impedance: `r_ohm_per_km` (0.1), `x_ohm_per_km` (0.1),
  `c_nf_per_km` (0), `max_i_ka` (0.4). Giving `r` or `x` selects this form.

Either way, zero sequence for earth faults: `r0_ohm_per_km` (4 x r),
`x0_ohm_per_km` (3 x x), `c0_nf_per_km` (= c) - a rule of thumb, since
pandapower's line types carry none; give the cable's own for a real study.
And `endtemp_degree` (80): the conductor temperature at the end of a fault,
to which a minimum-case short circuit raises the line's resistance.

**loads** - `id`, **`bus`**, `p_mw` (0), `q_mvar` (0.33 x p_mw, about pf 0.95),
`name`, `in_service`.

**generators** - voltage-controlled synchronous machines.
`id`, **`bus`**, `p_mw` (0), `vm_pu` (1.0), `sn_mva` (1.2 x p_mw, at least 1),
`slack` (false - set true on one generator if there is no external grid),
`name`, `in_service`, and short-circuit data: `vn_kv` (the bus voltage),
`xdss_pu` (subtransient reactance, 0.2), `rdss_ohm` (0), `cos_phi` (0.85).
The defaults are typical values so a short-circuit study runs; give the
machine's own for a study you will rely on.

**static_generators** - inverter-connected PV, wind, batteries in PQ mode.
`id`, **`bus`**, `p_mw` (0), `q_mvar` (0), `name`, `in_service`, and for
short circuit `sn_mva` (rated power: 1.1 x |p_mw|, at least |q_mvar| and
0.1) and `k` (short-circuit to rated current, 1.1). Modelled as an inverter
(current source).
On a radial layout a static generator whose `name` contains "wind" or
"turbine" is drawn as a wind turbine. `kind: "wind"` with a name that does not
say so raises a warning rather than renaming it.

**shunts** - `id`, **`bus`**, **`q_mvar`** (positive absorbs - a reactor;
negative injects - a capacitor bank), `p_mw` (0), `name`, `in_service`.

**storage** - `id`, **`bus`**, `p_mw` (0; positive charges, negative discharges),
`max_e_mwh` (1.0), `soc_percent` (state of charge, 50), `name`, `in_service`.
An OpenDSS load flow holds a battery at 0 % idle however much it is asked to
discharge; pandapower's ignores the state of charge.

**switches** - `id`, **`bus`**, **`element`**, `et` (`line` default,
`transformer`, `three_winding_transformer`, or `bus`), `closed` (true), `name`.
- `et: line` / `transformer` / `three_winding_transformer`: `element` is that
  element's id, and `bus` must be one of its terminals.
- `et: bus`: a bus coupler; `element` is the other bus's id.

## Optimal power flow

For Electrisim's OPF with the polynomial cost function, give each source a
price, `cost_per_mwh` (marginal cost per MWh of active power), and the OPF
minimises total cost:

| element | dispatch limits (defaults) |
|---|---|
| `external_grids` | `min_p_mw` (-1e6: may export) and `max_p_mw` (1e6) |
| `generators` | `min_p_mw` (0) to `max_p_mw` (`sn_mva`); `min_q_mvar` / `max_q_mvar` (what the rated power factor allows); `controllable` (true) |
| `static_generators` | priced ones are curtailable: `min_p_mw` (0) to `max_p_mw` (`p_mw`, the output available), reactive power up to power factor 0.9 at `sn_mva`; unpriced ones run at `p_mw` |
| `storage` | runs at `p_mw` |
| `buses` | voltage held to `min_vm_pu` (0.9) - `max_vm_pu` (1.1) |

Without a price Electrisim assumes 20 per MWh for generators and static
generators and nothing for the grid - which makes grid power free.

## Layout

- `transmission` - meshed networks, IEEE-style test cases, anything with loops.
  Buses are banded by voltage.
- `radial` - a plant or feeder single-line: grid at the top, feeders hanging in
  columns below, machines at the feeder ends.
- `auto` (default) - Electrisim guesses the way the import dialog's "Other"
  button does: more than 60 buses or many loops means transmission, else radial.
  A network with a three-winding transformer always gets transmission.

The radial layout cannot place three-winding transformers yet. Asked for
explicitly, it draws them, but stacks the buses below them in one column, and
`draw_diagram` warns and names those buses.

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
