"""
BESS preliminary design: the wizard's auto-sized ratings against the study.

The ratings come from the wizard's own computeSuggestedRatings (frontend
bessPlantBuilder.js, run with Node), are written into the reference plants,
and the study then has to pass every case without leaving the equipment
idle in its worst one.
"""
import json
import os
import re
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE_DIR = os.path.join(HERE, 'reference')
BUILDER = os.path.join(HERE, '..', '..', 'frontend', 'src', 'main', 'webapp', 'js', 'electrisim',
                       'bessPlantBuilder.js')

# The wizard settings each reference plant was generated with.
WIZARD = {
    'reference_radial': dict(pocP_MW=5, powerFactor=0.95, numUnits=2, umin_pu=0.95, auxP_MW=0.5,
                             auxQ_Mvar=0.1, hvTrafoEnabled=False, mvVoltage_kV=20, hvVoltage_kV=110),
    'reference_transmission': dict(pocP_MW=50, powerFactor=0.95, numUnits=4, umin_pu=0.95, auxP_MW=0.5,
                                   auxQ_Mvar=0.1, hvTrafoEnabled=True, mvVoltage_kV=33, hvVoltage_kV=110),
}


def _suggested_ratings(params):
    """Run the wizard's computeSuggestedRatings, as written, with Node."""
    node = shutil.which('node')
    if not node or not os.path.exists(BUILDER):
        pytest.skip('needs Node and the frontend checkout beside the backend')
    with open(BUILDER, encoding='utf-8') as handle:
        source = handle.read()
    start = source.index('export function computeSuggestedRatings(')
    depth, end = 0, source.index('{', start)
    for end in range(end, len(source)):
        depth += {'{': 1, '}': -1}.get(source[end], 0)
        if depth == 0:
            break
    constants = '\n'.join(re.findall(r'^const BESS_SIZING_\w+ = [^;]+;', source, re.M))
    script = (constants + '\n' + source[start:end + 1].replace('export ', '', 1)
              + f'\nconsole.log(JSON.stringify(computeSuggestedRatings({json.dumps(params)})));')
    out = subprocess.run([node, '-e', script], capture_output=True, text=True, timeout=60, check=True)
    return json.loads(out.stdout)


def _with_ratings(request, ratings):
    """The wizard's plant with its auto-sized ratings."""
    pmax = ratings['storagePMaxMw']
    for cell in request.values():
        if not isinstance(cell, dict):
            continue
        name = str(cell.get('userFriendlyName', ''))
        if name.startswith('PCS_'):
            cell.update(sn_mva=str(ratings['storageSnMva']), max_p_mw=str(pmax), min_p_mw=str(-pmax))
        elif name.startswith('MV_LV_Trafo_'):
            cell['sn_mva'] = str(ratings['stringTrafoSnMva'])
        elif name == 'POC_Transformer':
            cell['sn_mva'] = str(ratings['hvTrafoSnMva'])
        elif name.startswith('MV_Cable_'):
            cell['max_i_ka'] = str(ratings['cableMaxIKa'])
    request['bess_preliminary_params'].update(
        storageSnMva=ratings['storageSnMva'], pMaxDischarge_MW=pmax, pMaxCharge_MW=pmax, batteryPmax_MW=pmax)
    return request


@pytest.mark.parametrize('grid', sorted(WIZARD))
def test_bess_auto_sized_plant_passes_without_idle_headroom(client, quiet, grid):
    """
    The suggested ratings stacked their margins - x 1.12 on P, / 0.82 and
    x 1.30 on Q, / U, then x 1.18 or x 1.22 - so the default 5 MW plant got
    4.6 MVA PCS units and string transformers its worst case loaded to about
    65 %. Sized to the estimated worst case with one 5 % margin, every case
    still passes and the worst loads them past 80 %.
    """
    with open(os.path.join(REFERENCE_DIR, f'{grid}.diagram_bess_preliminary_payload.json'),
              encoding='utf-8') as handle:
        request = _with_ratings(json.load(handle), _suggested_ratings(WIZARD[grid]))
    with quiet():
        response = client.post('/', json=request)
    result = json.loads(response.get_data(as_text=True))
    assert not result.get('error'), result.get('message')
    result = result['bess_preliminary_results']

    assert result['summary']['passed_cases'] == result['summary']['total_cases'] == 18
    assert result['summary']['target_met_cases'] == result['summary']['target_cases'] == 12

    worst = {}
    for case in result['named_cases']:
        for row in case.get('elements') or []:
            name = str(row.get('name', ''))
            kind = next((k for k in ('PCS_', 'MV_LV_Trafo_', 'MV_Cable_', 'POC_Transformer')
                         if name.startswith(k)), None)
            if kind and row.get('loading_percent') is not None:
                worst[kind] = max(worst.get(kind, 0.0), float(row['loading_percent']))
    expected = {'PCS_', 'MV_LV_Trafo_', 'MV_Cable_'} | ({'POC_Transformer'} if WIZARD[grid]['hvTrafoEnabled'] else set())
    assert set(worst) == expected, worst
    for kind, loading in worst.items():
        assert 80.0 <= loading <= 100.0, (kind, worst)
