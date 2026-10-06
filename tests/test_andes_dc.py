"""
The DC network in the ANDES transient stability study: each VSC and
solid-state transformer is the load its AC side is in Electrisim's load flow,
so a data hall's power stays in the study. Before, ANDES skipped the DC
network and the hall's load vanished: the voltages it started from were not
the load flow's.
"""
import json

import numpy as np
import pytest

from test_dc_elements import _drawn_request, _with
from test_pcs import LF, _bus, _der, _pcs, _post, _two_buses

TDS = {'typ': 'TransientStabilityAndes Parameters', 'frequency': '50', 'sn_mva': '100', 'tf': '1',
       'fault_enabled': 'false', 'fault_bus': '', 'toggle_line': '', 'toggle_gen': '', 'user_email': 't@t'}


def _with_params(request, params):
    key = next(k for k, v in request.items() if 'Parameters' in str(v.get('typ', '')))
    request[key] = dict(params)
    return request


def _start_voltages(result):
    return {s['name']: s['values'][0] for s in result['bus_voltage']}


def test_dc_network_enters_through_its_rectifier(client, quiet):
    """
    The radial grid with its DC network (a rectifier on LV network A feeding
    two server halls): the study starts where the load flow does, at every
    bus, and says what stands for the DC network.
    """
    flow = _post(client, quiet, _with_params(_drawn_request(), LF))
    tds = _post(client, quiet, _with_params(_drawn_request(), TDS))
    assert not any('could not be initialised' in w for w in tds['warnings']), tds['warnings']
    names = {b['name']: b.get('userFriendlyName') or b['name'] for b in flow['busbars']}
    lf_v = {}
    for b in flow['busbars']:
        lf_v[b['name']] = b['vm_pu']
    req = _drawn_request()
    label = {v['name']: v.get('userFriendlyName') or v['name'] for v in req.values()
             if isinstance(v, dict) and str(v.get('typ', '')).startswith('Bus')}
    start = _start_voltages(tds)
    for name, vm in lf_v.items():
        if label.get(name) in start:
            assert start[label[name]] == pytest.approx(vm, abs=1e-3), label[name]
    note = next(w for w in tds['warnings'] if 'DC network is in this study' in w)
    vsc = next(v for v in flow['vscs'] if v.get('name') == 'vsc1' or 'Rectifier' in str(v.get('label', v.get('name'))))
    assert 'Rectifier' in note and f"{vsc['p_mw']:.4g} MW" in note
    assert not any('Skipped DC' in w for w in tds['warnings'])
    v = np.asarray(next(s['values'] for s in tds['bus_voltage'] if s['name'] == 'LV network A'))
    assert np.abs(v - v[0]).max() < 1e-4, 'steady'


def test_sst_draws_its_mv_power(client, quiet):
    """A 20 kV bus feeding an SST to an 800 V hall: the study's MV voltage is the load flow's, the hall's load in it."""
    def network(params):
        request = {'0': dict(params), '1': _bus('mv', 20), '2': {
            'typ': 'External Grid0', 'name': 'g', 'id': 'cell-g', 'userFriendlyName': 'G', 'bus': 'mv', 'vm_pu': '1',
            'va_degree': '0', 's_sc_max_mva': '50', 's_sc_min_mva': '50', 'rx_max': '0.1', 'rx_min': '0.1',
            'r0x0_max': '0.1', 'x0x_max': '1', 'r0x0_min': '0.1', 'x0x_min': '1', 'in_service': 'true'},
            '3': {'typ': 'Bus0', 'name': 'far', 'id': 'cell-far', 'userFriendlyName': 'FAR', 'vn_kv': '20'},
            '4': {'typ': 'Line0', 'name': 'l', 'id': 'cell-l', 'userFriendlyName': 'L', 'busFrom': 'mv', 'busTo': 'far',
                  'length_km': '5', 'parallel': '1', 'df': '1', 'in_service': 'true', 'r_ohm_per_km': '0.2',
                  'x_ohm_per_km': '0.12', 'c_nf_per_km': '0', 'g_us_per_km': '0', 'max_i_ka': '0.5', 'type': 'cs',
                  'r0_ohm_per_km': '0.8', 'x0_ohm_per_km': '0.4', 'c0_nf_per_km': '0', 'endtemp_degree': '80'},
            '5': {'typ': 'DC Bus0', 'name': 'hall', 'id': 'cell-hall', 'userFriendlyName': 'HALL', 'vn_kv': '0.8'},
            '6': {'typ': 'Solid-State Transformer0', 'name': 'sst', 'id': 'cell-sst', 'userFriendlyName': 'SST',
                  'bus_mv': 'far', 'bus_lv_dc': 'hall', 'vn_mv_kv': '20', 'rect_rated_mw': '5', 'dcdc_rated_mw': '5'},
            '7': {'typ': 'Load DC0', 'name': 'racks', 'id': 'cell-racks', 'userFriendlyName': 'RACKS', 'bus': 'hall',
                  'p_mw': '4'},
            # ANDES needs a machine with dynamics: a battery behind a PCS.
            '8': _der('Battery', 'b1', capacity_kwh=2000), '9': _pcs('p1', 'far', 'b1', p_set_mw=0.5, s_rated_mva=1.0)}
        return request
    flow = _post(client, quiet, network(LF))
    tds = _post(client, quiet, network(TDS))
    sst = flow['ssts'][0]
    assert sst['p_mv_mw'] == pytest.approx(4.0 / 0.98 / 0.985 + 0.004, rel=0.01)
    start = _start_voltages(tds)
    far = next(b for b in flow['busbars'] if b['name'] == 'far')
    assert start['FAR'] == pytest.approx(far['vm_pu'], abs=1e-3)
    assert far['vm_pu'] < 0.995         # the hall's 4 MW drops the far end: the study sees it
    assert any('SST (MV)' in w for w in tds['warnings'])
