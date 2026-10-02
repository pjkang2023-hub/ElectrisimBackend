# -*- coding: utf-8 -*-
"""
Golden-file regression tests for the load-flow path.

Each case is driven through the real HTTP route, so the whole chain is covered:
payload parsing -> create_busbars -> create_other_elements -> powerflow() ->
JSON serialisation.

Two independent checks per case:

1. `test_matches_pandapower` compares the backend's bus voltages and angles
   against `pp.runpp` on the same network. pandapower is an *external* oracle, so
   this catches a wrong answer even if the golden file was recorded wrong.

2. `test_matches_golden` compares against a committed golden file. This catches
   any change in the numbers, including ones pandapower would also make (a solver
   upgrade, a different default), which check 1 cannot see.

Regenerate goldens with:  pytest --regen-golden
Always read the resulting diff before committing it - that diff IS the review.
"""

import json
import os

import pandapower as pp
import pytest
from pandapower.networks import (
    case9, case14, case30, case39, case57, case_ieee30,
)

from electrisim_payload import build_payload, _bus_name

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden')

# Tolerances. The solver is deterministic, so these are tight on purpose: they
# are "did the number change at all", not "is it roughly right".
VM_TOL = 1e-8       # per unit
VA_TOL = 1e-6       # degrees
LOADING_TOL = 1e-6  # percent

CASES = {
    'case9': case9,
    'case14': case14,
    'case30': case30,
    'case39': case39,
    'case57': case57,
    'case_ieee30': case_ieee30,
}


def _solve(client, quiet, factory):
    """POST the case through the real route and return the parsed response."""
    with quiet():
        response = client.post('/', json=build_payload(factory()))
    assert response.status_code == 200, response.get_data(as_text=True)[:400]
    data = json.loads(response.get_data(as_text=True))
    assert 'error' not in data, data.get('error')
    return data


def _summarise(data):
    """
    Reduce a response to the numbers worth pinning, keyed by element name.

    The route returns busbars ordered lexicographically by name ('1', '10', '11',
    ...), not by bus index, so anything positional here would be comparing
    different buses. Keyed and sorted means a pure reordering is not a failure
    while a changed value still is.
    """
    return {
        'busbars': {
            str(b['name']): {
                'vm_pu': round(float(b['vm_pu']), 10),
                'va_degree': round(float(b['va_degree']), 8),
            }
            for b in data.get('busbars', [])
        },
        'lines': {
            str(l['name']): {
                'loading_percent': round(float(l['loading_percent']), 8),
                'p_from_mw': round(float(l['p_from_mw']), 8),
                'q_from_mvar': round(float(l['q_from_mvar']), 8),
            }
            for l in data.get('lines', [])
        },
    }


@pytest.mark.parametrize('case_name', sorted(CASES))
def test_matches_pandapower(client, quiet, case_name):
    """The backend must reproduce pandapower's own solution for the same network."""
    factory = CASES[case_name]
    data = _solve(client, quiet, factory)

    reference = factory()
    pp.runpp(reference, algorithm='nr', calculate_voltage_angles='auto', init='auto')

    got = {str(b['name']): b for b in data['busbars']}
    want = {
        _bus_name(reference, i): reference.res_bus.loc[i]
        for i in reference.bus.index
    }
    assert set(got) == set(want), (
        f'bus set changed: missing {sorted(set(want) - set(got))[:5]}, '
        f'unexpected {sorted(set(got) - set(want))[:5]}'
    )

    worst_vm = worst_va = 0.0
    for name, row in want.items():
        worst_vm = max(worst_vm, abs(float(got[name]['vm_pu']) - row['vm_pu']))
        worst_va = max(worst_va, abs(float(got[name]['va_degree']) - row['va_degree']))

    assert worst_vm < VM_TOL, f'{case_name}: worst |dV| = {worst_vm:.3e} pu'
    assert worst_va < VA_TOL, f'{case_name}: worst |dAngle| = {worst_va:.3e} deg'


@pytest.mark.parametrize('case_name', sorted(CASES))
def test_matches_golden(client, quiet, regen, case_name):
    """The numbers must not drift from what is committed."""
    data = _solve(client, quiet, CASES[case_name])
    actual = _summarise(data)
    path = os.path.join(GOLDEN_DIR, f'powerflow_{case_name}.json')

    if regen:
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(actual, handle, indent=2, sort_keys=True)
        pytest.skip(f'regenerated {os.path.basename(path)}')

    assert os.path.exists(path), (
        f'missing golden {path} - create it with: pytest --regen-golden'
    )
    with open(path, encoding='utf-8') as handle:
        expected = json.load(handle)

    assert set(actual['busbars']) == set(expected['busbars']), 'bus set changed'
    assert set(actual['lines']) == set(expected['lines']), 'line set changed'

    drifted = []
    for name, want in expected['busbars'].items():
        got = actual['busbars'][name]
        if abs(got['vm_pu'] - want['vm_pu']) > VM_TOL:
            drifted.append(f"bus {name} vm_pu {want['vm_pu']} -> {got['vm_pu']}")
        if abs(got['va_degree'] - want['va_degree']) > VA_TOL:
            drifted.append(f"bus {name} va_degree {want['va_degree']} -> {got['va_degree']}")

    for name, want in expected['lines'].items():
        got = actual['lines'][name]
        if abs(got['loading_percent'] - want['loading_percent']) > LOADING_TOL:
            drifted.append(
                f"line {name} loading_percent {want['loading_percent']} -> {got['loading_percent']}"
            )

    assert not drifted, (
        f'{case_name}: {len(drifted)} value(s) changed\n  '
        + '\n  '.join(drifted[:20])
    )


def test_payload_carries_case_frequency():
    """
    Regression guard for a real bug in this harness.

    Every IEEE case in pandapower is 60 Hz. Sending 50 Hz still converges and
    still looks plausible - it shifted bus voltages by up to 0.8% - which is
    exactly the kind of silent numerical error these tests exist to catch.
    """
    for name, factory in CASES.items():
        net = factory()
        payload = build_payload(net)
        sent = float(payload['simulation-parameters']['frequency'])
        assert sent == float(net.f_hz), f'{name}: sent {sent} Hz for a {net.f_hz} Hz network'


def test_safe_int_accepts_decimal_strings():
    """
    Regression test. Found by this suite: case39 transformers have
    tap_neutral = 0, which came back as 1 and moved every bus voltage.

    The browser happens to send '0' for these today, so this is latent rather
    than live - but any value that reaches the payload as '2.0' (an imported
    model, a grid cell, a JSON round-trip of a float) is silently wrong.
    """
    import pandapower_electrisim as pe

    assert pe.safe_int('0.0') == 0
    assert pe.safe_int('2.0') == 2
    assert pe.safe_int('-1.0') == -1

    # The other half of the same bug: an absent tap field fell back to 1.
    # Every tap call site relies on this default, so it has to be 0.
    assert pe.safe_int('') == 0
    assert pe.safe_int(None) == 0
    assert pe.safe_int('null') == 0
    assert pe.safe_int('None') == 0

    # An explicit default still wins - 'parallel' and the shunt control
    # increment pass 1 and must keep getting it.
    assert pe.safe_int('', 1) == 1
    assert pe.safe_int('rubbish', 1) == 1

    # Values that already converted keep converting, exactly.
    assert pe.safe_int(3) == 3
    assert pe.safe_int('-4') == -4
    assert pe.safe_int(10 ** 20) == 10 ** 20

    # Nothing numeric to read still falls back rather than raising.
    assert pe.safe_int('rubbish') == 0
    assert pe.safe_int(float('nan')) == 0
    assert pe.safe_int(float('inf')) == 0
