"""
Large load steps in the EMT study: a load's impedance scaled by an event, from a stiff
grid, against its constant-impedance power at the bus's voltage. A change as large as
the load's own conductance, taken from the last step's voltage, blew up the run.
"""
import json
import math

import pytest

from test_pcs import _bus, GRID, LOAD

KV, MW, MVAR = 33.0, 30.0, 6.0


def _system(value, **params):
    return {'0': {'typ': 'EmtStudy Parameters', 'user_email': 't@t', 'time_step_us': '20', 'duration_ms': '200',
                  'events': json.dumps([{'t': 0.05, 'action': 'scale', 'element': 'LD', 'value': value}]), **{k: str(v) for k, v in params.items()}},
            '1': _bus('a', KV),
            '2': {**GRID, 'bus': 'a', 's_sc_max_mva': '100000', 's_sc_min_mva': '100000', 'rx_max': '0.1', 'rx_min': '0.1'},
            '3': {**LOAD, 'bus': 'a', 'p_mw': str(MW), 'q_mvar': str(MVAR)}}


def _run(client, quiet, value, **params):
    with quiet():
        result = json.loads(client.post('/', json=_system(value, **params)).get_data(as_text=True))
    assert 'emt' in result, result.get('message')
    return result['emt']


def _phase_a_peak(emt):
    bus = emt['ac']['buses'][0]
    return max(abs(x) for x in bus['waveform']['v_a_kv'])


@pytest.mark.parametrize('value', [0.0, 0.001, 0.5, 1.33, 2.0, 3.33, 10.0])
def test_a_load_step_of_any_size_stays_on_a_stiff_bus(client, quiet, value):
    """The bus holds its voltage through the step, and the load's share ends where the event put it."""
    emt = _run(client, quiet, value)
    assert _phase_a_peak(emt) == pytest.approx(KV * math.sqrt(2.0 / 3.0), rel=0.03)
    (ld,) = [x for x in emt['changing_loads'] if x['label'].startswith('LD')]
    assert ld['share_end'] == pytest.approx(value, rel=0.01, abs=1e-6)
    assert ld['v_end_pu'] == pytest.approx(1.0, abs=0.03)


def test_a_load_behind_a_ups_falling_to_nothing_stays_finite(client, quiet):
    """A fault, the load's UPS taking it to nothing and walking it back: every voltage stays a voltage to the run's end."""
    request = _system(1.0, ac_fault_bus='a', ac_fault_type='abcg', ac_fault_time_ms=30, ac_fault_duration_ms=50, ac_fault_resistance_ohm=0.1, duration_ms=250)
    request['2'].update({'s_sc_max_mva': '1000', 's_sc_min_mva': '1000'})
    request['3'].update({'uv_ease_pu': '0.85', 'uv_floor_pu': '0', 'uv_restore_s': '0.2'})
    with quiet():
        emt = json.loads(client.post('/', json=request).get_data(as_text=True))['emt']
    assert _phase_a_peak(emt) < 1.5 * KV * math.sqrt(2.0 / 3.0)
    (ld,) = [x for x in emt['changing_loads'] if x['label'].startswith('LD')]
    assert ld['ups_min'] == 0 and ld['v_end_pu'] == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize('value', [2.0, 3.33])
def test_the_power_a_large_step_draws(client, quiet, value):
    """The grid's current after the step is the scaled load's, at the bus's own voltage: the source's phase current."""
    emt = _run(client, quiet, value)
    expected = value * math.hypot(MW, MVAR) * 1e3 / (math.sqrt(3.0) * KV) * math.sqrt(2.0)   # A peak
    grid = next(x for x in emt['ac']['branches'] if 'G' in str(x.get('label')))
    assert grid['i_peak_ka'] * 1e3 == pytest.approx(expected, rel=0.1)
