"""
The OpenDSS load flow on the drawn transmission grid: its three-winding
transformer's loading against pandapower's.
"""
import json
import math
import os

import pandapower as pp
import pytest

import electrisim_sld as sld

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE_DIR = os.path.join(HERE, 'reference')


def test_opendss_three_winding_loading_is_the_worst_winding(client, quiet):
    """
    The loading was the HV side's apparent power over the HV rating: the main
    transformer read 14.6 % (5.8 MVA of 40) while its 15 MVA tertiary carried
    19.7 % - pandapower's figure. It is each winding against its own rating,
    the worst of them.
    """
    with open(os.path.join(REFERENCE_DIR, 'reference_transmission.diagram_opendss_payload.json'),
              encoding='utf-8') as handle:
        payload = json.load(handle)
    with quiet():
        response = client.post('/', json=payload)
    result = json.loads(response.get_data(as_text=True).splitlines()[-1])
    (t3w,) = result['transformers3w']
    rated = {'hv': 40.0, 'mv': 40.0, 'lv': 15.0}
    windings = {side: math.hypot(t3w[f'p_{side}_mw'], t3w[f'q_{side}_mvar']) / rated[side] * 100
                for side in rated}
    assert max(windings, key=windings.get) == 'lv'
    assert t3w['loading_percent'] == pytest.approx(windings['lv'], rel=1e-9)

    with open(os.path.join(REFERENCE_DIR, 'reference_transmission.spec.json'), encoding='utf-8') as handle:
        net, _ = sld.build_network(json.load(handle))
    pp.runpp(net)
    # The engines differ a little (the grid behind its source impedance in
    # OpenDSS); the loading must not be a third low.
    assert t3w['loading_percent'] == pytest.approx(float(net.res_trafo3w.loading_percent.iloc[0]), rel=0.02)


def _radial(client, quiet, engine, change=None):
    """The drawn radial grid through one engine's load flow, labelled by name."""
    fixture = 'diagram_opendss_payload' if engine == 'opendss' else 'diagram_payload'
    with open(os.path.join(REFERENCE_DIR, f'reference_radial.{fixture}.json'), encoding='utf-8') as handle:
        payload = json.load(handle)
    if change:
        for element in payload.values():
            if isinstance(element, dict) and element.get('userFriendlyName') in change:
                element.update(change[element['userFriendlyName']])
    with quiet():
        response = client.post('/', json=payload)
    result = json.loads(response.get_data(as_text=True).splitlines()[-1])
    label = {v['name']: v.get('userFriendlyName') for v in payload.values()
             if isinstance(v, dict) and 'name' in v}
    return {key: {label.get(row['name'], row['name']): row for row in rows}
            for key, rows in result.items() if isinstance(rows, list)}


def test_opendss_line_loading_is_the_worse_end(client, quiet):
    """
    The loading was the from end's current: on LA2, which the cable's
    charging current leaves higher at the far end, 3.42 % where pandapower
    reads 3.54 %. It is the worse end, against max_i_ka x df x parallel.
    """
    dss = _radial(client, quiet, 'opendss')
    ppw = _radial(client, quiet, 'pandapower')
    for name in ('LA1', 'LA2', 'LB1', 'LB2', 'Wind farm cable'):
        row = dss['lines'][name]
        assert row['loading_percent'] == pytest.approx(
            max(row['i_from_ka'], row['i_to_ka']) / 0.5 * 100 if name == 'Wind farm cable'
            else max(row['i_from_ka'], row['i_to_ka']) / 0.421 * 100, rel=1e-9), name
        assert row['loading_percent'] == pytest.approx(ppw['lines'][name]['loading_percent'], rel=0.03), name


@pytest.mark.parametrize('element, buses, branch_key', (
    ('LA1', ('A1', 'A2', 'LV network A'), 'lines'),
    ('TA', ('LV network A',), 'transformers'),
))
def test_opendss_parallel_circuits(client, quiet, element, buses, branch_key):
    """
    "parallel" was never read for OpenDSS: two circuits of LA1, or two units
    of TA, went in as one - twice the impedance, half the rating - so its
    loading and the voltage beyond stayed as for one, where pandapower halves
    the loading and raises the voltage.
    """
    change = {element: {'parallel': '2'}}
    dss_one, dss_two = _radial(client, quiet, 'opendss'), _radial(client, quiet, 'opendss', change)
    pp_two = _radial(client, quiet, 'pandapower', change)

    one = dss_one[branch_key][element]['loading_percent']
    two = dss_two[branch_key][element]['loading_percent']
    assert two == pytest.approx(one / 2, rel=0.02)
    assert two == pytest.approx(pp_two[branch_key][element]['loading_percent'], rel=0.03)
    for bus in buses:
        assert dss_two['busbars'][bus]['vm_pu'] > dss_one['busbars'][bus]['vm_pu'], bus
        assert dss_two['busbars'][bus]['vm_pu'] == pytest.approx(pp_two['busbars'][bus]['vm_pu'], abs=2e-4), bus
