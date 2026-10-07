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
| `buses`, `external_grids`, `transformers`, `three_winding_transformers`, `lines`, `loads`, `generators`, `static_generators`, `shunts`, `storage`, `motors`, `switches` | lists of AC elements |
| `dc_buses`, `dc_lines`, `dc_loads`, `dc_sources`, `dc_capacitors`, `vscs`, `ssts`, `dc_dc_converters`, `dc_breakers`, `batteries`, `supercapacitors`, `flywheels`, `sofcs`, `pv_arrays`, `pcs`, `grounding_transformers` | lists of DC and microgrid elements - see below |
| `load_profiles` | the diagram's load-profile library - see below |

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
`si0_hv_partial` (0.9), and the grounded winding's neutral resistor and reactor
`rn_ohm` / `xn_ohm` (0: solidly grounded) - on the LV winding of a Dyn, Yyn or
YNyn unit, the HV of a YNd or YNy.
A tap changer, when any of its fields is given: `tap_side` (`hv`), `tap_pos`
(= `tap_neutral`), `tap_neutral` (0), `tap_min` / `tap_max` (neutral -/+ 2),
`tap_step_percent` (2.5) - e.g. an off-load tap at -2.5 % on the HV winding:
`tap_pos` -1. Without them the transformer has none.
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
Dynamics, for the transient-stability, eigenvalue and EMT studies, as the
generator's Dynamics tab names them: `dyn_machine_model` (GENROU or GENCLS),
`dyn_H` (inertia constant, s) or `dyn_M` (2H), `dyn_D`, `dyn_ra`, `dyn_xl`,
`dyn_xd`, `dyn_xq`, `dyn_xd1`, `dyn_xq1`, `dyn_xd2`, `dyn_xq2`, `dyn_Td10`,
`dyn_Td20`, `dyn_Tq10`, `dyn_Tq20` (per unit on `sn_mva`, seconds);
`dyn_exciter_model` (NONE, EXDC2, SEXS, IEEEX1, ESDC2A, EXST1, ESST1A, AC8B)
with `dyn_exc_KA`, `dyn_exc_TR`, `dyn_exc_TA`, `dyn_exc_TE`, `dyn_exc_K`;
`dyn_governor_model` (NONE, TGOV1, IEEEG1, IEESGO, GAST, HYGOV) with
`dyn_gov_R` (droop, per unit), `dyn_gov_T1`, `dyn_gov_T2`, `dyn_gov_T3`;
`dyn_pss_model` (NONE, IEEEST) with `dyn_pss_A1`, `dyn_pss_A2`. Left out,
each study takes its own defaults.

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

**motors** - `id`, **`bus`**, **`pn_mech_mw`** (rated shaft power), `cos_phi`
(0.86), `efficiency_percent` (95), `loading_percent` (100), `lrc_pu`
(locked-rotor to rated current, 6), `rx` (locked-rotor R/X, 0.15), `vn_kv`
(the bus voltage), `cos_phi_n` / `efficiency_n_percent` (rated values, default
the operating ones), `name`, `in_service`. A load in the power flow, drawing
`pn_mech_mw` x `loading_percent` / `efficiency_percent`; short circuit adds its
locked-rotor contribution, and motor starting starts it.

**switches** - `id`, **`bus`**, **`element`**, `et` (`line` default,
`transformer`, `three_winding_transformer`, `bus`, or `grounding_transformer`),
`closed` (true), `name`.
- `et: line` / `transformer` / `three_winding_transformer`: `element` is that
  element's id, and `bus` must be one of its terminals.
- `et: bus`: a bus coupler; `element` is the other bus's id.
- `et: grounding_transformer`: the breaker of a grounding transformer;
  `element` is its id, `bus` its bus.

## DC and microgrid elements

These are Electrisim's own elements. Each takes its dialog's fields, by the
dialog's names (the symbol in brackets after each field label); a field the
dialog does not have is refused, listing the ones it does. Anything not given
takes the dialog's default. Connections name other elements by id: an AC bus
from `buses`, a DC bus from `dc_buses`. They are drawn below the AC network,
each converter, PCS and grounding transformer under its AC bus with what it
feeds beneath it. `check_network` reports them under their ids: `dc_buses`,
`dc_lines`, `vscs`, `ssts`, `dc_dc_converters`, `sources_and_stores`, `pcs`,
`grounding_transformers`.

**dc_buses** - `id`, **`vn_kv`** (DC, e.g. 0.8 for an 800 V bus), `name`, `in_service`.

**dc_lines** - a DC cable. `id`, **`from_bus`**, **`to_bus`** (DC buses),
`length_km`, `r_ohm_per_km`, `max_i_ka`, `l_mh_per_km`, `c_uf_per_km`.

**dc_loads** - `id`, **`bus`** (DC), `p_mw`, `load_model` and its shares
(`share_p_percent`, `share_i_percent`, `share_r_percent`), `v_min_pu`,
`filter_l_mh`, `filter_c_uf`, `load_profile_id` (a power profile).

**dc_sources** - an ideal DC source. `id`, **`bus`** (DC), `vm_pu`, `r_sc_mohm`, `l_sc_uh`.

**dc_capacitors** - `id`, **`bus`** (DC), `c_mf`, `esr_mohm`, `esl_uh`.

**vscs** - an AC/DC converter. `id`, **`bus`** (AC), **`bus_dc`** (DC),
`r_ohm`, `x_ohm`, `r_dc_ohm`, `control_mode_ac` (`vm_pu` or `q_mvar`) with
`control_value_ac`, `control_mode_dc` (`vm_pu` or `p_mw`) with
`control_value_dc`, `rated_mva`, `dc_link_mf`, `current_limit_pu`, `emt_model`,
`switching_khz`. A VSC needs at least sqrt(2) times its AC line voltage on its
DC side: 0.48 kV AC for an 800 V bus.

**ssts** - a solid-state transformer, MV AC to LV DC (and optionally LV AC).
`id`, **`bus_mv`** (AC), **`bus_lv_dc`** (DC), `bus_lv_ac` (AC), and its
stages: `vn_mv_kv`, `vn_lv_dc_kv`, `vn_lv_ac_kv`, `link_kv`, `q_mv_mvar`,
`rect_rated_mw`, `rect_efficiency_percent`, `rect_no_load_kw`,
`dcdc_rated_mw`, `dcdc_efficiency_percent`, `dcdc_no_load_kw`, `vm_lv_dc_pu`,
`inverter_mode`, `inv_rated_mw`, `inv_efficiency_percent`, `inv_no_load_kw`,
`p_ac_mw`, `q_ac_mvar`, `vm_lv_ac_pu`, `emt_model`, `switching_khz`,
`dcdc_switching_khz`, `current_limit_pu`.

**dc_dc_converters** - `id`, **`bus_in`**, **`bus_out`** (DC), `control_mode`
(`voltage`, `power`, `droop`, `dispatch`, `mppt`, `follower` or `smoothing`),
`vm_out_pu`, `p_set_mw`, `rated_mw`, `vn_in_kv`, `vn_out_kv`,
`efficiency_percent`, `no_load_loss_kw`, `bidirectional`, `droop_percent`,
`smoothing_tau_s`, `soc_ref_percent`, `soc_gain`, `emt_model`,
`switching_khz`, `current_limit_pu`, `c_out_mf`.

**dc_breakers** - `id`, **`bus`** (DC), **`element`** (the DC bus, line, load,
source, VSC or DC/DC converter it switches), `closed`, `breaker_type`,
`rated_voltage_kv`, `rated_current_ka`, `breaking_capacity_ka`,
`trip_current_ka`, `opening_time_ms`, `limiting_inductance_mh`,
`arrester_clamp_kv`, `arrester_energy_kj`.

**batteries** - `id`, `bus` (a DC bus, or none when behind a PCS), `sizing`
(`ratings` or `cells`), `vn_v`, `capacity_kwh`, `r0_mohm`, `cells_series`,
`strings_parallel`, `cell_v`, `cell_ah`, `cell_r_mohm`, `r1_percent`, `tau1_s`,
`soc_percent`, `soc_min_percent`, `soc_max_percent`, `c_rate_discharge`,
`c_rate_charge`, `coulombic_efficiency_percent`, `ocv_table`, `l_uh`.

**supercapacitors** - `id`, `bus`, `coupling`, `sizing`, `c_f`, `v_rated`,
`esr_mohm`, `esl_uh`, `module_c_f`, `module_v`, `module_esr_mohm`,
`module_esl_uh`, `modules_series`, `strings_parallel`, `v0_percent`,
`v_min_percent`, `p_rated_kw`, `r_leak_ohm`.

**flywheels** - `id`, `bus`, `v_dc`, `p_rated_kw`, `e_max_kwh`, `speed_percent`,
`speed_min_percent`, `speed_base_percent`, `efficiency_percent`,
`standby_loss_percent_h`, `r_dc_mohm`, `p_set_kw`.

**sofcs** - `id`, `bus`, `p_rated_kw`, `v_rated`, `p_set_kw`,
`fuel_utilisation_percent`, `min_load_percent`, `aux_load_percent`, `ramp_percent_s`.

**pv_arrays** - `id`, `bus`, `module_pmpp_w`, `module_vmpp`, `module_impp`,
`module_voc`, `module_isc`, `module_cells_series`, `alpha_isc_percent_k`,
`beta_voc_percent_k`, `noct_c`, `modules_series`, `strings_parallel`,
`loss_percent`, `irradiance_wm2`, `ambient_c`, `irradiance_profile_id` (an
irradiance profile), `temperature_profile_id` (a temperature profile).

A source or store goes on a DC bus (behind a DC/DC converter on a bus of its
own, or directly on a network bus), or behind a PCS - then give it no `bus`.

**pcs** - a power conversion system joining one source or store to an AC bus.
`id`, **`bus`** (AC), and its DC side: `source` (a battery, supercapacitor,
flywheel, SOFC or PV array id) or `bus_dc` (a DC bus with its source alone on
it). `control` (`grid_following` or `grid_forming`), `s_rated_mva`,
`vn_ac_kv`, `efficiency_percent`, `no_load_loss_kw`, `p_set_mw`, `q_mode`
(`q`, `pf` or `qv`), `q_set_mvar`, `pf`, `qv_droop_percent`, `vm_set_pu`,
`droop_pf_percent`, `droop_qv_percent`, `current_limit_pu`,
`opf_marginal_cost_eur_per_mwh`. Draw its transformer to the network as an
ordinary transformer between its own LV bus and the network bus.

**grounding_transformers** - a zigzag grounding transformer. `id`, **`bus`**
(AC), `vn_kv`, `i_rated_a` (400), `t_rated_s` (10), `r_n_ohm` (neutral
resistor; blank: V_ph / `i_rated_a`), `x_n_ohm`, `x0_ohm`, `r0_ohm`. Its
breaker is a switch with `et: grounding_transformer`.

**load_profiles** - the library elements follow: a list of `id`, `name`,
`kind` (`power` - per unit of the load's P -, `irradiance` in W/m², or
`temperature` in °C), **`values`**, and either `dt_s` (the step between
values, with `t0_s`, 0) or `t_s` (each value's time). A `load_profile_id`,
`irradiance_profile_id` or `temperature_profile_id` must name one of the
right kind.

## Optimal power flow

For Electrisim's OPF with the polynomial cost function, give each source a
price, `cost_per_mwh` (marginal cost per MWh of active power), and the OPF
minimises total cost:

| element | dispatch limits (defaults) |
|---|---|
| `external_grids` | `min_p_mw` (-1e6: may export) and `max_p_mw` (1e6) |
| `generators` | `min_p_mw` (0) to `max_p_mw` (rated active power, `sn_mva` × `cos_phi`); `min_q_mvar` / `max_q_mvar` (what the rated power factor allows), so the rated point is within `sn_mva`; `controllable` (true) |
| `static_generators` | priced ones are curtailable: `min_p_mw` (0) to `max_p_mw` (`p_mw`, the output available), reactive power up to power factor 0.9 at `sn_mva`; unpriced ones run at `p_mw` |
| `storage` | runs at `p_mw` |
| `motors` | run at their load, like `loads` |
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
