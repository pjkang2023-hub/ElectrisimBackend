"""
A transformer whose load-flow shift is a clock its windings cannot make: the EMT study says so.

In Electrisim a vector group's number is the transformer's shift (YNd1: 30 degrees) and shift_degree adds to it, so
YNd1 with shift_degree 30 makes the load flow's 60 degrees - clock 2, which no star-delta connection makes. The EMT
wound it at clock 11 without a word: a generator behind it then started some 90 degrees from the grid it shared a
bus with, with tens of kA between them. Found benchmarking against PSCAD 4.6's sync_machines/sync_SCtest, where the
bench's own row had that shift.
"""
import json

import numpy as np


def _request(shift_degree, vector_group='YNd1'):
    bus = lambda name, kv: {'typ': 'Bus0', 'name': name, 'id': f'cell-{name}', 'userFriendlyName': name.upper(), 'vn_kv': str(kv)}
    gen = {'typ': 'Generator0', 'name': 'g', 'id': 'cell-g', 'userFriendlyName': 'G', 'bus': 'lv', 'p_mw': '0',
           'vm_pu': '1', 'sn_mva': '120', 'vn_kv': '13.8', 'in_service': 'true', 'xdss_pu': '0.28', 'rdss_ohm': '',
           'cos_phi': '0.9', 'scaling': '1', 'slack': 'false', 'controllable': 'true', 'min_p_mw': '', 'max_p_mw': '',
           'pg_percent': '', 'power_station_trafo': ''}
    tx = {'typ': 'Transformer0', 'name': 'tx', 'id': 'cell-tx', 'userFriendlyName': 'TX', 'hv_bus': 'hv', 'lv_bus': 'lv',
          'sn_mva': '120', 'vn_hv_kv': '230', 'vn_lv_kv': '13.8', 'vkr_percent': '0.1', 'vk_percent': '10', 'pfe_kw': '0',
          'i0_percent': '0', 'vector_group': vector_group, 'vk0_percent': '10', 'vkr0_percent': '0.1',
          'mag0_percent': '100', 'si0_hv_partial': '0.9', 'rn_ohm': '0', 'xn_ohm': '0', 'parallel': '1',
          'shift_degree': str(shift_degree), 'tap_side': 'hv', 'tap_pos': '0', 'tap_neutral': '0', 'tap_max': '0',
          'tap_min': '0', 'tap_step_percent': '0', 'tap_step_degree': '0', 'tap_phase_shifter': 'false',
          'tap_changer_type': 'Ratio', 'in_service': 'true'}
    grid = {'typ': 'External Grid0', 'name': 'grid', 'id': 'cell-grid', 'userFriendlyName': 'GRID', 'bus': 'hv',
            'vm_pu': '1', 'va_degree': '0', 's_sc_max_mva': '5000', 's_sc_min_mva': '5000', 'rx_max': '0.1',
            'rx_min': '0.1', 'r0x0_max': '0.1', 'x0x_max': '1', 'r0x0_min': '0.1', 'x0x_min': '1', 'in_service': 'true'}
    study = {'typ': 'EmtStudy Parameters', 'user_email': 't@t', 'frequency': '60', 'time_step_us': '50',
             'duration_ms': '40'}
    return {'0': study, '1': bus('hv', 230), '2': bus('lv', 13.8), '3': gen, '4': tx, '5': grid}


def _run(client, quiet, request):
    with quiet():
        out = json.loads(client.post('/', json=request).get_data(as_text=True))
    assert 'emt' in out, out.get('message')
    gen = next(b for b in out['emt']['ac']['branches'] if b['kind'] == 'Generator')
    peak = max(float(np.max(np.abs(gen[f'i_{p}_ka']))) for p in 'abc')
    return peak, [w for w in out.get('warnings') or [] if 'not one a' in w]


def test_the_vector_groups_own_shift_starts_in_the_load_flow(client, quiet):
    """YNd1 as drawn (shift_degree 0): no warning, and the no-load generator carries next to nothing."""
    peak, warned = _run(client, quiet, _request(0))
    assert not warned
    assert peak < 0.05                                 # kA; a 120 MVA machine's rated current is 7.1 kA peak


def test_a_shift_its_windings_cannot_make_is_warned_about(client, quiet):
    peak, warned = _run(client, quiet, _request(30))
    assert len(warned) == 1
    assert '60 degrees (clock 2)' in warned[0] and 'clock 11' in warned[0] and 'shift_degree adds' in warned[0]
    assert peak > 5.0                                  # what the warning is about: the run starts far from the load flow


def test_a_shift_on_top_that_still_makes_a_clock_is_not_warned_about(client, quiet):
    """YNd1 with shift_degree 120 is clock 5: a winding that exists, so nothing to say."""
    _, warned = _run(client, quiet, _request(120))
    assert not warned
