# -*- coding: utf-8 -*-
"""The MCP tools, called over a real in-process MCP session."""

import pytest
from _support import FakePage
from mcp import Client

from electrisim_mcp.server import create_server

pytestmark = pytest.mark.anyio

SPEC = {'name': 'Two-bus', 'buses': [{'id': 'A', 'vn_kv': 20}, {'id': 'B', 'vn_kv': 20}]}


def text_of(result):
    return '\n'.join(getattr(c, 'text', '') for c in result.content)


async def call(backend, bridge, name, args=None):
    async with Client(create_server(backend=backend, bridge=bridge)) as client:
        return await client.call_tool(name, args or {})


async def test_tools_and_their_hints(backend, bridge):
    async with Client(create_server(backend=backend, bridge=bridge)) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == {'get_spec_format', 'check_network', 'draw_diagram', 'bridge_status'}
    # Only drawing changes anything, and it never destroys existing work.
    assert tools['draw_diagram'].annotations.read_only_hint is False
    assert tools['draw_diagram'].annotations.destructive_hint is False
    for name in ('get_spec_format', 'check_network', 'bridge_status'):
        assert tools[name].annotations.read_only_hint is True
    # A bad layout is a schema error the client can see before calling.
    layout = tools['draw_diagram'].input_schema['properties']['layout']
    assert 'transmission' in str(layout) and 'radial' in str(layout)


async def test_spec_format_is_the_reference(backend, bridge):
    result = await call(backend, bridge, 'get_spec_format')
    text = text_of(result)
    assert '## Example' in text and 'switches' in text
    assert backend.calls == []


async def test_check_network_runs_power_flow_by_default(backend, bridge):
    result = await call(backend, bridge, 'check_network', {'spec': SPEC})
    assert not result.is_error
    data = result.structured_content
    assert data['valid'] is True
    assert data['power_flow']['converged'] is True
    (sent,) = backend.calls
    assert sent['run_power_flow'] is True and sent['include_model'] is False
    assert sent['limits'] == {'vm_min_pu': 0.95, 'vm_max_pu': 1.05, 'max_loading_percent': 100.0}


async def test_spec_problems_reach_the_model_verbatim(backend, bridge):
    backend.problems = ["buses[1] (B): vn_kv is required", "lines[0] (L1): to_bus='X' is not a bus id"]
    result = await call(backend, bridge, 'check_network', {'spec': SPEC})
    assert result.is_error
    text = text_of(result)
    assert '2 problem(s)' in text
    assert "- lines[0] (L1): to_bus='X' is not a bus id" in text


async def test_backend_outage_is_explained_not_hidden(backend, bridge):
    backend.error = 'Cannot reach the Electrisim backend at http://x (refused). Start it ...'
    result = await call(backend, bridge, 'check_network', {'spec': SPEC})
    assert result.is_error
    # The SDK withholds the text of unexpected exceptions; ours must survive.
    assert 'Cannot reach the Electrisim backend' in text_of(result)


async def test_draw_reports_what_the_page_drew(backend, bridge):
    with FakePage(bridge) as page:
        result = await call(backend, bridge, 'draw_diagram', {'spec': SPEC, 'layout': 'radial'})
    assert not result.is_error, text_of(result)
    data = result.structured_content
    assert (data['status'], data['cells_added'], data['layout']) == ('drawn', 19, 'vertical')
    assert page.drawn[0]['layout'] == 'radial'
    assert page.drawn[0]['title'] == 'Two-bus'


async def test_draw_warns_when_radial_could_not_place_buses(backend, bridge):
    # Drawn, but not faithfully - the model cannot see the canvas, so say so.
    outcome = {'ok': True, 'cellsAdded': 30, 'layout': 'radial', 'unplaced': ['S1', 'S2']}
    with FakePage(bridge, outcome):
        result = await call(backend, bridge, 'draw_diagram', {'spec': SPEC, 'layout': 'radial'})
    data = result.structured_content
    assert data['status'] == 'drawn'
    assert data['unplaced_buses'] == ['S1', 'S2']
    (warning,) = data['warnings']
    assert 'could not place 2 bus(es)' in warning and 'layout="transmission"' in warning


async def test_clean_draw_has_no_unplaced_warning(backend, bridge):
    with FakePage(bridge):
        result = await call(backend, bridge, 'draw_diagram', {'spec': SPEC})
    data = result.structured_content
    assert 'unplaced_buses' not in data
    assert data['warnings'] == []


async def test_draw_without_a_page_queues_and_says_where_to_go(backend, bridge):
    result = await call(backend, bridge, 'draw_diagram', {'spec': SPEC, 'wait_seconds': 0.2})
    data = result.structured_content
    assert data['status'] == 'queued'
    assert 'No Electrisim page is connected. Open http://127.0.0.1:5501' in data['hint']
    assert bridge.status()['pending'] == [data['id']]


async def test_draw_with_no_diagram_open_queues_and_says_so(backend, bridge):
    with FakePage(bridge, ready=False) as page:
        result = await call(backend, bridge, 'draw_diagram', {'spec': SPEC, 'wait_seconds': 0.5})
    data = result.structured_content
    assert data['status'] == 'queued'
    assert 'Electrisim is open but no diagram is in view' in data['hint']
    assert page.drawn == []
    assert bridge.status()['pending'] == [data['id']]


async def test_draw_failure_on_the_page_is_an_error(backend, bridge):
    outcome = {'ok': False, 'error': 'The Electrisim editor is not ready.', 'errors': ['boom']}
    with FakePage(bridge, outcome):
        result = await call(backend, bridge, 'draw_diagram', {'spec': SPEC})
    assert result.is_error
    text = text_of(result)
    assert 'The Electrisim editor is not ready.' in text and 'boom' in text


async def test_layout_falls_back_to_the_spec_then_auto(backend, bridge):
    backend.layout = 'transmission'
    with FakePage(bridge) as page:
        await call(backend, bridge, 'draw_diagram', {'spec': SPEC})
    backend.layout = None
    with FakePage(bridge) as page2:
        await call(backend, bridge, 'draw_diagram', {'spec': SPEC})
    assert page.drawn[0]['layout'] == 'transmission'
    assert page2.drawn[0]['layout'] == 'auto'


async def test_auto_layout_avoids_radial_for_three_winding_transformers(backend, bridge):
    backend.counts = {'bus': 3, 'trafo3w': 1}
    with FakePage(bridge) as page:
        await call(backend, bridge, 'draw_diagram', {'spec': SPEC})
    assert page.drawn[0]['layout'] == 'transmission'
    # An explicit choice is still honoured; the unplaced warning covers it.
    with FakePage(bridge) as page2:
        await call(backend, bridge, 'draw_diagram', {'spec': SPEC, 'layout': 'radial'})
    assert page2.drawn[0]['layout'] == 'radial'


async def test_unknown_layout_is_rejected_before_anything_runs(backend, bridge):
    result = await call(backend, bridge, 'draw_diagram', {'spec': SPEC, 'layout': 'sideways'})
    assert result.is_error
    assert backend.calls == []


async def test_bridge_status_names_both_ends(backend, bridge):
    result = await call(backend, bridge, 'bridge_status')
    data = result.structured_content
    assert data['running'] is True
    assert data['backend_url'] == 'http://fake-backend'
    assert data['frontend_url'] == 'http://127.0.0.1:5501'
