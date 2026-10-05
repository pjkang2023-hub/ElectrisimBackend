"""
Protection coordination: inverter-based sources inside a fault's zone.
"""
import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE_DIR = os.path.join(HERE, 'reference')

# The relays the reference fixtures carry, with pickups set by hand so the
# study runs on its own evaluator.
MANUAL = {
    'reference_radial': {
        'CB_T1': dict(I_g_a='1500', I_gg_a='12000', t_g='0.8', t_gg='0.3'),
        None: dict(I_g_a='400', I_gg_a='3000', t_g='0.5', t_gg='0.07'),
    },
    'reference_transmission': {None: {}},
}

# Each line fault's zone - what it reaches without passing a relay - and the
# sources in it, by the drawn topology. Radial: feeder A's zone runs through
# the unswitched transformer TA to the rooftop PV; the wind feeder's to the
# wind farm and the battery beside it; feeder B's holds the gas engine only.
# Transmission: the CB_L1 / CB_L4 zone reaches the wind farm and, through the
# unswitched T_LV1, the battery on LV1; the CB_T2 / CB_L4 zone reaches the
# CHP plant and, through Substation LV2, the rooftop PV.
EXPECTED = {
    'reference_radial': {
        'LA1': ([], ['Rooftop PV']),
        'LA2': ([], ['Rooftop PV']),
        'LB1': (['Gas engine'], []),
        'LB2': (['Gas engine'], []),
        'Wind farm cable': ([], ['Battery', 'Wind farm C']),
    },
    'reference_transmission': {
        '110 kV overhead line': ([], []),
        'L1': ([], ['Battery', 'Wind farm']),
        'Cable with given impedance': (['CHP plant'], ['Rooftop PV']),
        'L3': ([], ['Battery', 'Wind farm']),
        'L4': (['CHP plant'], ['Rooftop PV']),
        'Wind farm cable': ([], ['Battery', 'Wind farm']),
    },
}


@pytest.mark.parametrize('grid', sorted(EXPECTED))
def test_protection_names_inverter_sources_in_the_fault_zone(client, quiet, grid):
    """
    Only External Grids and synchronous generators counted as sources, so a
    wind farm, PV or battery feeding a fault inside a zone - until its own
    protection disconnects it - went unmentioned. They are named apart, and do
    not make a relay a primary: their current is about their rating, too
    little to operate an overcurrent relay.
    """
    with open(os.path.join(REFERENCE_DIR, f'{grid}.diagram_protection_payload.json'),
              encoding='utf-8') as handle:
        payload = json.load(handle)
    for element in payload.values():
        if (isinstance(element, dict) and str(element.get('typ', '')).startswith('Switch')
                and element.get('protection_type') == 'ocr'):
            settings = MANUAL[grid].get(element['userFriendlyName'], MANUAL[grid][None])
            element.update(pickup_mode='manual', **settings)
    with quiet():
        response = client.post('/', json=payload)
    result = json.loads(response.get_data(as_text=True))
    assert not result.get('error'), result.get('message')

    lines = [v['userFriendlyName'] for v in payload.values()
             if isinstance(v, dict) and str(v.get('typ', '')).startswith('Line')]
    for scenario in result['scenarios']:
        line = lines[int(scenario['sc_line_id'])]
        synchronous, inverter = EXPECTED[grid][line]
        assert scenario['unprotected_sources'] == synchronous, line
        assert scenario['unprotected_inverter_sources'] == inverter, line
