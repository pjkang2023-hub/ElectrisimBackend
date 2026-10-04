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
