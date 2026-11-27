"""
The three-winding transformer in the EMT study against the load flow it starts from, on the reference transmission
drawing, whose station supply hangs off one: every bus's voltage as the load flow has it, the station supply alive,
and its CHP plant, run as a machine, starting at its load-flow power and holding it. Before, the study left
three-winding transformers out: that bus sat at 0 V, every flow shifted, and the machine swung from the start.
"""
import json

import numpy as np
import pandapower as pp
import pytest

import pandapower_electrisim
import pandapower_electrisim as pe
import test_reference_grids as G


def _load_flow(request):
    key = '0'
    net = pp.create_empty_network(f_hz=50.0)
    buses = pandapower_electrisim.create_busbars(request, net)
    pandapower_electrisim.create_other_elements(request, net, key, buses)
    pe._electrisim_runpp(net, algorithm='nr', calculate_voltage_angles=True)
    return {str(net.bus.at[b, 'name']): float(net.res_bus.at[b, 'vm_pu']) for b in net.bus.index}


def _emt(client, quiet, **params):
    request = G._study_request('reference_transmission', {'typ': 'EmtStudy Parameters', 'user_email': 't@t', 'frequency': '50', 'time_step_us': '20',
                                                          'duration_ms': '300', **params})
    with quiet():
        out = json.loads(client.post('/', json=request).get_data(as_text=True))
    assert 'emt' in out, out.get('message')
    return request, out


def test_every_bus_starts_where_the_load_flow_left_it(client, quiet):
    request, out = _emt(client, quiet)
    assert not any('Left out' in w and 'trafo3w' in w for w in out.get('warnings', []))
    lf = _load_flow(request)
    for bus in out['emt']['ac']['buses']:
        assert bus['v_rms_final_pu'] == pytest.approx(lf[bus['name']], abs=0.003), bus['label']
        assert bus['v_rms_min_pu'] == pytest.approx(lf[bus['name']], abs=0.003), bus['label']


def test_the_plant_as_a_machine_holds_its_load_flow_power(client, quiet):
    _, out = _emt(client, quiet, generator_model='machine')
    (m,) = out['emt']['machines']
    assert m['p_start_mw'] == pytest.approx(4.0, rel=0.01) and m['p_end_mw'] == pytest.approx(4.0, rel=0.01)
    assert m['f_max_hz'] - m['f_min_hz'] < 0.005
