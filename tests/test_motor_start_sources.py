"""
The steady-state motor start against ETAP's static motor starting (Example-ANSI, 9 Nov 2026):

- a motor rated below its bus's voltage (4 kV on 4.16 kV) is its locked-rotor impedance (4 kV)^2 / S_lr, so it draws
  its locked-rotor current times the voltage it sees over its own rating; taken at the bus's voltage it drew 7.5 %
  less than ETAP's;
- with source_model 'behind_impedance' each grid and generator is its internal voltage behind its impedance (grid
  1 / S_sc at its R/X, generator Ra + jX'd), set by the load flow before the start, as ETAP and IEEE 399 have it: the
  bus the grid or a generator holds dips too. With it Electrisim matched ETAP's during-start voltages to 0.05 %.
"""
import math

import pandapower as pp
import pytest

import motor_starting_electrisim as ms

HP = 0.7457e-3


def _net(motor_kv=4.16, s_sc=100.0, rx=0.1):
    net = pp.create_empty_network(f_hz=60.0)
    a = pp.create_bus(net, vn_kv=4.16, name="a")
    pp.create_ext_grid(net, a, vm_pu=1.0, name="grid", s_sc_max_mva=s_sc, s_sc_min_mva=s_sc, rx_max=rx, rx_min=rx)
    pp.create_motor(net, a, pn_mech_mw=500 * HP, cos_phi=0.85, efficiency_percent=94.0, loading_percent=80.0, name="pump",
                    lrc_pu=5.0, rx=0.25, vn_kv=motor_kv, cos_phi_n=0.85, efficiency_n_percent=94.0)
    return net


def _start(net, model, in_data=None):
    out = ms._steady_state_start(net, {"starting_method": "dol", "motor_ids": "all", "source_model": model}, in_data or {})
    assert not out.get("error"), out
    return out


@pytest.mark.parametrize("motor_kv", [4.16, 4.0])
def test_a_motor_draws_its_locked_rotor_current_at_the_voltage_it_sees(motor_kv):
    out = _start(_net(motor_kv), "fixed_voltage")
    (m,) = out["motors"]
    i_n = 500 * HP / 0.94 / 0.85 / (math.sqrt(3) * motor_kv)
    assert m["i_start_nominal_ka"] == pytest.approx(5 * i_n, rel=1e-9)
    v = out["buses"][0]["vm_during"] * 4.16                      # kV at the motor, the grid holding it at 1 pu
    assert m["i_start_ka"] == pytest.approx(5 * i_n * v / motor_kv, rel=1e-6)
    assert m["v_motor_percent"] == pytest.approx(100 * v / motor_kv, rel=1e-9)


def test_behind_impedance_the_grids_bus_dips_by_the_divider():
    net = _net(4.0, s_sc=20.0, rx=0.1)
    fixed = _start(net, "fixed_voltage")
    assert fixed["buses"][0]["vm_during"] == pytest.approx(1.0)  # the load flow holds the grid's bus
    out = _start(_net(4.0, s_sc=20.0, rx=0.1), "behind_impedance")
    assert out["summary"]["source_model"] == "behind_impedance"
    # Nothing else on the bus: the grid's E is 1 pu, the motor its locked-rotor impedance behind 1 / S_sc.
    i_n = 500 * HP / 0.94 / 0.85 / (math.sqrt(3) * 4.0)
    s_lr = math.sqrt(3) * 4.0 * 5 * i_n
    z_m = 4.0 ** 2 / s_lr * complex(0.25, 1) / math.hypot(0.25, 1)
    z_s = 4.16 ** 2 / 20.0 * complex(0.1, 1) / math.hypot(0.1, 1)
    (bus,) = out["buses"]
    assert bus["vm_before"] == pytest.approx(1.0)
    assert bus["vm_during"] == pytest.approx(abs(z_m / (z_m + z_s)), rel=1e-6)
    assert bus["vm_after"] == pytest.approx(fixed["buses"][0]["vm_after"], rel=1e-9)   # sources restored
    assert len(net.bus) == 1 and len(net.ext_grid) == 1 and net.impedance.empty


def test_behind_impedance_starts_from_the_load_flow_with_a_generator():
    """E = V + Z I from the before load flow: with nothing started, the network solves to the same voltages."""
    net = pp.create_empty_network(f_hz=60.0)
    a, b, c = (pp.create_bus(net, vn_kv=13.8, name=n) for n in "abc")
    pp.create_ext_grid(net, a, vm_pu=1.0, s_sc_max_mva=500.0, s_sc_min_mva=300.0, rx_max=0.1, rx_min=0.15)
    pp.create_line_from_parameters(net, a, b, 1.0, 0.1, 0.4, 0.0, 1.0)
    pp.create_line_from_parameters(net, b, c, 1.0, 0.1, 0.4, 0.0, 1.0)
    pp.create_gen(net, c, p_mw=6.0, vm_pu=1.01, sn_mva=8.0, name="g1")
    pp.create_load(net, b, p_mw=9.0, q_mvar=4.0)
    pp.runpp(net)
    before = net.res_bus.vm_pu.copy(), net.res_bus.va_degree.copy()
    warnings = []
    added = ms._sources_behind_impedance(net, {"x": {"typ": "Generator", "name": "g1", "dyn_xd1": "0.28", "dyn_ra": "0.01"}}, warnings)
    assert not warnings and len(added["ext_grid"]) == 2
    pp.runpp(net)
    for k in (a, b, c):
        assert net.res_bus.at[k, "vm_pu"] == pytest.approx(before[0][k], abs=1e-9)
        assert net.res_bus.at[k, "va_degree"] == pytest.approx(before[1][k], abs=1e-7)
    # The grid behind its minimum short-circuit power, at its R/X; the generator behind Ra + jX'd.
    zg = net.impedance.iloc[0]
    assert (zg.sn_mva, zg.rft_pu / zg.xft_pu) == pytest.approx((300.0, 0.15))
    zm = net.impedance.iloc[1]
    assert (zm.sn_mva, zm.rft_pu, zm.xft_pu) == pytest.approx((8.0, 0.01, 0.28))
    ms._undo_sources(net, added)
    assert len(net.bus) == 3 and len(net.ext_grid) == 1 and net.impedance.empty
    assert bool(net.ext_grid.in_service.all()) and bool(net.gen.in_service.all())


def test_a_generator_without_its_transient_reactance_falls_back_with_a_warning():
    net = pp.create_empty_network(f_hz=60.0)
    a = pp.create_bus(net, vn_kv=13.8)
    pp.create_ext_grid(net, a, vm_pu=1.0)                       # no short-circuit power: held, with a warning
    pp.create_gen(net, a, p_mw=1.0, vm_pu=1.0, sn_mva=2.0, name="g")
    pp.runpp(net)
    warnings = []
    added = ms._sources_behind_impedance(net, {}, warnings)
    assert added["ext_grid"] == [] and len(warnings) == 2       # nor a generator without any reactance
    net.gen["xdss_pu"], net.gen["rdss_ohm"] = 0.2, 0.0
    warnings = []
    added = ms._sources_behind_impedance(net, {}, warnings)
    assert len(added["ext_grid"]) == 1 and any("subtransient" in w for w in warnings)
