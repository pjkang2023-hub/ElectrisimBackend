"""
A capacitor bank energised in the EMT study (ac_energise_shunt): its inrush current is among the study's branches
however its breaker closes. Closed the ordinary way (its poles together), it was left out - only a controlled closing
listed it - so the inrush could be read only from its peaks, not seen. Found benchmarking against PSCAD 4.6, where the
bank's current had to be taken from the grid's instead.
"""
import json

import numpy as np
import pytest


def _request(controlled):
    rows = [
        {'typ': 'Bus0', 'name': 'hv', 'id': 'cell-hv', 'userFriendlyName': 'HV', 'vn_kv': '230'},
        {'typ': 'External Grid0', 'name': 'g', 'id': 'cell-g', 'userFriendlyName': 'G', 'bus': 'hv', 'vm_pu': '1',
         'va_degree': '0', 's_sc_max_mva': '5000', 's_sc_min_mva': '5000', 'rx_max': '0.1', 'rx_min': '0.1',
         'r0x0_max': '0.1', 'x0x_max': '1', 'r0x0_min': '0.1', 'x0x_min': '1', 'in_service': 'true'},
        {'typ': 'Capacitor0', 'name': 'c1', 'id': 'cell-c1', 'userFriendlyName': 'C1', 'bus': 'hv', 'q_mvar': '50',
         'loss_factor': '0', 'vn_kv': '230', 'step': '1', 'max_step': '1', 'in_service': 'true'},
    ]
    study = {'typ': 'EmtStudy Parameters', 'user_email': 't@t', 'frequency': '60', 'time_step_us': '20',
             'duration_ms': '60', 'ac_energise_shunt': 'c1', 'ac_energise_time_ms': '20',
             'ac_energise_controlled': 'true' if controlled else 'false'}
    return {'0': study, **{str(k + 1): r for k, r in enumerate(rows)}}


@pytest.mark.parametrize('controlled', [False, True])
def test_the_banks_inrush_is_a_branch_either_way(client, quiet, controlled):
    with quiet():
        out = json.loads(client.post('/', json=_request(controlled)).get_data(as_text=True))
    assert 'emt' in out, out.get('message')
    emt = out['emt']
    bank = [b for b in emt['ac']['branches'] if b['kind'] == 'Shunt' and 'energised' in b['label']]
    assert len(bank) == 1
    i = np.array([bank[0][f'i_{p}_ka'] for p in 'abc'])
    t = np.array(bank[0]['t_ms'])
    assert np.max(np.abs(i[:, t < 19.9])) < 1e-3                       # under 1 A (its open poles' leakage) before it closes
    # its peak is the energising result's, and it is what the grid delivers (nothing else on the bus)
    assert np.max(np.abs(i)) == pytest.approx(emt['energise']['i_peak_ka'], rel=0.02)
    grid = next(b for b in emt['ac']['branches'] if b['kind'] == 'External grid')
    assert np.max(np.abs(i)) == pytest.approx(max(np.max(np.abs(grid[f'i_{p}_ka'])) for p in 'abc'), rel=0.02)
