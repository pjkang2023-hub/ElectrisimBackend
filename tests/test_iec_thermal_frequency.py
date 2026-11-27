"""
IEC 60909 thermal current at the network's frequency. pandapower takes m - the
DC component's heat - at 50 Hz whatever the network's (currents._calc_ith:
f = 50); Electrisim puts it at the network's. By hand on a grid and a
transformer, no current sources, so kappa is ip / (sqrt(2) Ik''):

    m = (exp(4 f tk ln(kappa - 1)) - 1) / (2 f tk ln(kappa - 1)),  ith = Ik'' sqrt(m + 1)
"""
import contextlib
import io
import json
import math

import pytest

import electrisim_sld as sld
import electrisim_spec_layer as layer
import pandapower_electrisim as pe


def _spec(f_hz):
    return {
        'name': 'iec thermal', 'frequency_hz': f_hz,
        'buses': [{'id': 'HV', 'vn_kv': 110}, {'id': 'MV', 'vn_kv': 20}],
        'external_grids': [{'id': 'Grid', 'bus': 'HV', 'vm_pu': 1.0, 's_sc_max_mva': 2000, 'rx_max': 0.1,
                            'x0x_max': 1.0, 'r0x0_max': 0.1}],
        'transformers': [{'id': 'T', 'hv_bus': 'HV', 'lv_bus': 'MV', 'sn_mva': 40, 'vk_percent': 12,
                          'vkr_percent': 0.4, 'vector_group': 'YNyn', 'shift_degree': 0}],
        'loads': [{'id': 'L', 'bus': 'MV', 'p_mw': 10, 'q_mvar': 2}],
    }


def _sc(f_hz, fault, tk_s):
    params = {'typ': 'ShortCircuitPandaPower Parameters', 'fault_type': fault, 'fault_location': 'max',
              'fault_bus_mode': 'all', 'fault_bus_ids': [], 'fault_bus_names': [], 'fault_impedance': '6',
              'topology': 'auto', 'tk_s': str(tk_s), 'r_fault_ohm': '0', 'x_fault_ohm': '0', 'inverse_y': 'True'}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        net, _ = sld.build_network(_spec(f_hz))
        full = layer.with_layer(net, net.get('electrisim_layer') or {}, net.get('electrisim_bus_index') or {}, params)
        out = json.loads(pe.shortcircuit(full, params, None))
    assert full.f_hz == f_hz and not out.get('error'), out.get('message')
    return {r['name']: r for r in out['busbars']}


def _m(kappa, f_hz, tk_s):
    x = f_hz * tk_s * math.log(kappa - 1.0)
    return (math.exp(4.0 * x) - 1.0) / (2.0 * x)


@pytest.mark.parametrize('f_hz', [50, 60])
@pytest.mark.parametrize('fault', ['3ph', '2ph'])
@pytest.mark.parametrize('tk_s', [0.1, 1.0])
def test_thermal_current_at_the_network_frequency(f_hz, fault, tk_s):
    """Each bus's ith, Ik'' sqrt(m + 1) with m at the network's frequency, to 1e-9."""
    rows = _sc(f_hz, fault, tk_s)
    assert set(rows) == {'HV', 'MV'}
    for name, r in rows.items():
        kappa = r['ip_ka'] / (math.sqrt(2) * r['ikss_ka'])
        assert 1.0 < kappa < 1.99, name
        want = r['ikss_ka'] * math.sqrt(_m(kappa, f_hz, tk_s) + 1.0)
        assert r['ith_ka'] == pytest.approx(want, rel=1e-9), (name, f_hz, fault, tk_s)


def test_sixty_hertz_lower_than_fifty():
    """The DC component decays in fewer seconds at 60 Hz: ith lower, Ik'' and ip the same."""
    at50, at60 = _sc(50, '3ph', 0.1), _sc(60, '3ph', 0.1)
    for name in at50:
        assert at60[name]['ikss_ka'] == pytest.approx(at50[name]['ikss_ka'], rel=1e-12)
        assert at60[name]['ip_ka'] == pytest.approx(at50[name]['ip_ka'], rel=1e-12)
        assert at60[name]['ith_ka'] < at50[name]['ith_ka']
