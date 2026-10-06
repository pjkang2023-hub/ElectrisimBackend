"""
A transformer's neutral resistor (rn_ohm / xn_ohm) in the IEC 60909 earth
fault. pandapower's zero sequence does not read it, so a resistance-grounded
winding was solidly grounded: the campus's 13.8 kV, grounded through 400 A,
would have shown several kA. 3 Z_N now joins the grounded winding's zero
sequence, corrected by K_T no more than IEC corrects it - not at all.
"""
import json
import math

import pytest

C_MAX = 1.1
S_SC, RX = 600.0, 0.125


def _params(lv_tol='6'):
    return {'typ': 'ShortCircuitPandaPower Parameters', 'fault_type': '1ph', 'fault_location': 'max',
            'fault_bus_mode': 'all', 'fault_bus_ids': [], 'fault_bus_names': [], 'fault_impedance': lv_tol,
            'topology': 'auto', 'tk_s': '1', 'r_fault_ohm': '0', 'x_fault_ohm': '0', 'inverse_y': 'True',
            'exportPython': False, 'exportPandapowerResults': False, 'exportPdfReport': False, 'user_email': 't@t'}


def _bus(name, vn):
    return {'typ': 'Bus0', 'name': name, 'id': f'cell-{name}', 'userFriendlyName': name.upper(), 'vn_kv': str(vn)}


def _grid(bus, x0x='1'):
    return {'typ': 'External Grid0', 'name': 'grid', 'id': 'cell-grid', 'userFriendlyName': 'Grid', 'bus': bus,
            'vm_pu': '1', 'va_degree': '0', 's_sc_max_mva': str(S_SC), 's_sc_min_mva': '400', 'rx_max': str(RX),
            'rx_min': str(RX), 'r0x0_max': '0.1', 'x0x_max': x0x, 'r0x0_min': '0.1', 'x0x_min': x0x,
            'in_service': 'true'}


def _trafo(hv, lv, vn_hv, vn_lv, sn, vk, vkr, group, rn=0.0, xn=0.0):
    return {'typ': 'Transformer0', 'name': 'tx', 'id': 'cell-tx', 'userFriendlyName': 'TX', 'hv_bus': hv,
            'lv_bus': lv, 'sn_mva': str(sn), 'vn_hv_kv': str(vn_hv), 'vn_lv_kv': str(vn_lv), 'vkr_percent': str(vkr),
            'vk_percent': str(vk), 'pfe_kw': '0', 'i0_percent': '0', 'vector_group': group, 'vk0_percent': str(vk),
            'vkr0_percent': str(vkr), 'mag0_percent': '100', 'si0_hv_partial': '0.9', 'rn_ohm': str(rn),
            'xn_ohm': str(xn), 'parallel': '1', 'shift_degree': '0', 'tap_side': 'hv', 'tap_pos': '0',
            'tap_neutral': '0', 'tap_max': '0', 'tap_min': '0', 'tap_step_percent': '0', 'tap_step_degree': '0',
            'tap_phase_shifter': 'false', 'tap_changer_type': 'Ratio', 'in_service': 'true'}


def _ikss(client, quiet, rows, lv_tol='6'):
    payload = {'0': _params(lv_tol), **{str(k + 1): r for k, r in enumerate(rows)}}
    with quiet():
        out = json.loads(client.post('/', json=payload).get_data(as_text=True))
    assert not out.get('error'), out.get('message')
    return {r['name']: float(r['ikss_ka']) for r in out['busbars']}


def _z(vk, vkr, vn, sn):
    return complex(vkr, math.sqrt(vk * vk - vkr * vkr)) / 100 * vn * vn / sn


def _k_t(vk, vkr, c_max):
    return 0.95 * c_max / (1 + 0.6 * math.sqrt(vk * vk - vkr * vkr) / 100)


def _z_q(vn):
    z = C_MAX * vn * vn / S_SC
    x = z / math.sqrt(1 + RX * RX)
    return complex(RX * x, x)


R_N_13 = 13.8e3 / math.sqrt(3) / 400


@pytest.mark.parametrize('rn,xn', [(R_N_13, 0.0), (0.0, 5.0), (0.0, 0.0)])
def test_dyn_grounded_through_its_neutral(client, quiet, rn, xn):
    """
    35/13.8 kV Dyn, 12 MVA, 8 %: an earth fault on its LV bus is
    c sqrt(3) U / |2 Z1 + K_T Z0T + 3 Z_N|, Z1 the source referred to 13.8 kV
    and K_T Z_T; through 400 A of resistance, a little under c x 400 A.
    """
    got = _ikss(client, quiet, [_bus('hv', 35), _bus('lv', 13.8), _grid('hv'),
                                 _trafo('hv', 'lv', 35, 13.8, 12, 8, 0.6, 'Dyn', rn, xn)])
    k_t = _k_t(8, 0.6, C_MAX)
    z_t = _z(8, 0.6, 13.8, 12)
    z1 = _z_q(35) * (13.8 / 35) ** 2 + k_t * z_t
    z0 = k_t * z_t + 3 * complex(rn, xn)
    want = C_MAX * math.sqrt(3) * 13.8 / abs(2 * z1 + z0)
    assert got['lv'] == pytest.approx(want, rel=1e-6)
    if rn:
        assert 0.98 * C_MAX * 0.4 < got['lv'] < C_MAX * 0.4      # c x 400 A, a little under


def test_ynd_grounded_on_its_hv_with_an_lv_delta_below_1_kv(client, quiet):
    """
    A PCS-style 35/0.69 kV YNd unit, its star grounded through 50 ohm, on an
    ungrounded 35 kV source: the HV earth fault is the star point's alone. Its
    K_T takes its LV bus's c_max, 1.05 under a 6 % LV tolerance.
    """
    got = _ikss(client, quiet, [_bus('hv', 35), _bus('lv', 0.69), _grid('hv', x0x='1000000'),
                                 _trafo('hv', 'lv', 35, 0.69, 5.5, 6, 0.5, 'YNd', 50.0)])
    k_t = _k_t(6, 0.5, 1.05)
    z0 = k_t * _z(6, 0.5, 35, 5.5) + 150.0
    z0_grid = 1e6 * _z_q(35).imag * complex(0.1, 1)
    z0 = z0 * z0_grid / (z0 + z0_grid)
    want = C_MAX * math.sqrt(3) * 35 / abs(2 * _z_q(35) + z0)
    assert got['hv'] == pytest.approx(want, rel=1e-5)
    assert 0.98 * C_MAX * 0.4 < got['hv'] < 1.01 * C_MAX * 0.4


def test_a_delta_hv_resistor_does_nothing(client, quiet):
    """A neutral resistor on a winding that is not grounded (Dyn's HV) carries no current: the LV fault as solid."""
    rows = [_bus('hv', 35), _bus('lv', 13.8), _grid('hv'), _trafo('hv', 'lv', 35, 13.8, 12, 8, 0.6, 'Yd', 20.0)]
    assert _ikss(client, quiet, rows)['hv'] == pytest.approx(
        _ikss(client, quiet, rows[:3] + [_trafo('hv', 'lv', 35, 13.8, 12, 8, 0.6, 'Yd')])['hv'], rel=1e-9)
