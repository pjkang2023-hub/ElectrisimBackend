"""
IEC 60909's single-phase short circuit with full-converter current sources
(sc_current_sources): their share scaled by 3 Z(1) / (2 Z(1) + Z(0)) at the
fault, as pandapower 3.3 does not.
"""
import pandapower as pp
import pandapower.shortcircuit as sc
import pytest

import sc_current_sources

sc_current_sources.install()


def _grid(sgen=True, earthed=True):
    """A 110 kV grid, a 110/20 kV YNyn transformer (or YNd: no zero sequence at 20 kV), 2 km of cable, a 5 MVA inverter."""
    net = pp.create_empty_network()
    hv, mv, far = pp.create_bus(net, 110), pp.create_bus(net, 20), pp.create_bus(net, 20)
    pp.create_ext_grid(net, hv, s_sc_max_mva=2000, rx_max=0.1, x0x_max=1.0, r0x0_max=0.1)
    pp.create_transformer_from_parameters(net, hv, mv, 40, 110, 20, 0.5, 12, 0, 0,
                                          vector_group='YNyn' if earthed else 'YNd', vk0_percent=12,
                                          vkr0_percent=0.5, mag0_percent=100, mag0_rx=0, si0_hv_partial=0.9)
    pp.create_line_from_parameters(net, mv, far, 2, 0.1, 0.12, 250, 0.4, r0_ohm_per_km=0.4, x0_ohm_per_km=0.36,
                                   c0_nf_per_km=250, endtemp_degree=80)
    pp.create_sgen(net, far, 4, sn_mva=5, k=1.2, in_service=sgen)
    return net, far


def _ik(net, bus, fault):
    sc.calc_sc(net, fault=fault, case='max', bus=bus, branch_results=False)
    return float(net.res_bus_sc.at[bus, 'ikss_ka'])


def test_the_current_source_share_of_an_earth_fault():
    """
    With the inverter, the earth fault grows by its three-phase share times
    3 |Z(1)| / |2 Z(1) + Z(0)| at the fault (pandapower 3.3 added it whole).
    """
    net, far = _grid()
    ik1, ik3 = _ik(net, far, '1ph'), _ik(net, far, '3ph')
    net0, _ = _grid(sgen=False)
    ik3_0, ik1_0 = _ik(net0, far, '3ph'), _ik(net0, far, '1ph')
    z1 = complex(net0.res_bus_sc.at[far, 'rk_ohm'], net0.res_bus_sc.at[far, 'xk_ohm'])
    z0 = complex(net0.res_bus_sc.at[far, 'rk0_ohm'], net0.res_bus_sc.at[far, 'xk0_ohm'])
    f = 3 * abs(z1) / abs(2 * z1 + z0)
    assert 0 < f < 1.5 and f != pytest.approx(1.0, abs=0.05)
    assert ik1 - ik1_0 == pytest.approx(f * (ik3 - ik3_0), rel=1e-3)


def test_no_earth_fault_share_without_a_zero_sequence_path():
    """
    Behind a YNd transformer the 20 kV side has no ground but its cable's
    capacitance: an earth fault there draws its 6 A, and the inverter adds
    next to nothing to it (pandapower 3.3 added its whole 0.17 kA).
    """
    net, far = _grid(earthed=False)
    with_inverter = _ik(net, far, '1ph')
    net.sgen.at[0, 'in_service'] = False
    without = _ik(net, far, '1ph')
    assert 0.003 < without < 0.01
    assert with_inverter == pytest.approx(without, abs=0.002)


def test_install_is_idempotent():
    import importlib
    calc = importlib.import_module('pandapower.shortcircuit.calc_sc')
    first = calc._calc_ikss_1ph
    sc_current_sources.install()
    assert calc._calc_ikss_1ph is first and getattr(first, '_electrisim_iec_current_sources', False)
