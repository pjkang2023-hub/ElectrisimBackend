# -*- coding: utf-8 -*-
"""
Golden-file tests for `POST /import-pandapower`.

This route runs a pandapower script and serialises the resulting network into the
structure the canvas consumes (`pandapower_net_to_json`, ~500 lines in app.py).
Nothing else exercises that serialiser, and a regression there corrupts every
imported model while the power flow itself stays correct.

The scripts come from the repository's own template library, so these also fail
if a template is deleted or edited.

Regenerate with:  pytest --regen-golden
"""

import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_DIR = os.path.join(HERE, 'golden')
TEMPLATES = os.path.normpath(os.path.join(
    HERE, '..', '..', 'frontend', 'src', 'main', 'webapp',
    'templates', 'power_system_test_cases'
))

# Small, fast, structurally varied: no trafos / trafos / larger mesh.
TEMPLATE_CASES = ['case9.py', 'case14.py', 'case30.py', 'case39.py']


def _structure(model):
    """
    Element counts and per-table column counts.

    Deliberately not the full payload: bus coordinates are layout, and pinning
    them would make every cosmetic template tweak a test failure.
    """
    out = {}
    tables = model.get('_object', model)
    for table_name, table in sorted(tables.items()):
        raw = table.get('_object') if isinstance(table, dict) else None
        if raw is None:
            continue
        rows = json.loads(raw).get('data', []) if isinstance(raw, str) else raw.get('data', [])
        out[table_name] = {
            'rows': len(rows),
            'columns': len(rows[0]) if rows else 0,
        }
    return out


@pytest.mark.skipif(not os.path.isdir(TEMPLATES),
                    reason=f'template library not found at {TEMPLATES}')
@pytest.mark.parametrize('template', TEMPLATE_CASES)
def test_import_structure_golden(client, quiet, regen, template):
    path = os.path.join(TEMPLATES, template)
    if not os.path.exists(path):
        pytest.skip(f'{template} not in template library')

    with open(path, encoding='utf-8') as handle:
        source = handle.read()

    with quiet():
        response = client.post('/import-pandapower', json={'content': source})

    assert response.status_code == 200, response.get_data(as_text=True)[:400]
    model = json.loads(response.get_data(as_text=True))
    actual = _structure(model)
    assert actual, f'{template}: import returned no tables'

    golden_path = os.path.join(GOLDEN_DIR, f'import_{template.replace(".py", "")}.json')
    if regen:
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        with open(golden_path, 'w', encoding='utf-8') as handle:
            json.dump(actual, handle, indent=2, sort_keys=True)
        pytest.skip(f'regenerated {os.path.basename(golden_path)}')

    assert os.path.exists(golden_path), (
        f'missing golden {golden_path} - create it with: pytest --regen-golden'
    )
    with open(golden_path, encoding='utf-8') as handle:
        expected = json.load(handle)

    assert actual == expected, (
        f'{template}: import structure changed\n'
        f'  expected: {json.dumps(expected, sort_keys=True)}\n'
        f'  actual  : {json.dumps(actual, sort_keys=True)}'
    )


def test_import_is_gated_by_default(client, quiet, monkeypatch):
    """The exec() route must stay closed unless explicitly enabled."""
    monkeypatch.delenv('ELECTRISIM_ALLOW_SCRIPT_IMPORT', raising=False)
    with quiet():
        response = client.post('/import-pandapower', json={'content': 'net = 1'})
    assert response.status_code == 403, 'script import must be disabled without the env var'
