# -*- coding: utf-8 -*-
"""
Electrisim MCP server: draw single-line diagrams, and check them, from a JSON spec.

    MCP client  --tools-->  this server  --/build-model-->  Electrisim backend
                                 |
                                 +--> local bridge  <--polls--  Electrisim tab

Run it the way an MCP client launches it, over stdio:

    python -m electrisim_mcp            (from the backend directory)
    python path/to/electrisim_mcp/server.py

Configuration (environment):

    ELECTRISIM_BACKEND_URL    backend base URL            (http://127.0.0.1:5000)
    ELECTRISIM_TOKEN          bearer token, only if the backend enforces auth
    ELECTRISIM_BRIDGE_PORT    local bridge port           (5503)
    ELECTRISIM_BRIDGE_ORIGINS comma-separated page origins allowed to use the bridge
                              (http://127.0.0.1:5501,http://localhost:5501)
    ELECTRISIM_FRONTEND_URL   where to tell the user to open Electrisim
                              (http://127.0.0.1:5501)

stdout carries the MCP protocol. Nothing here may print to it; logs go to stderr.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Literal

if __package__ in (None, ''):
    # Launched as a script: make the package importable by its absolute name.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

from electrisim_mcp.backend_client import BackendClient, BackendError, SpecProblems
from electrisim_mcp.bridge import DEFAULT_ORIGINS, DEFAULT_PORT, Bridge

log = logging.getLogger('electrisim.mcp')

SPEC_FORMAT = (Path(__file__).resolve().parent / 'spec_format.md').read_text(encoding='utf-8')

INSTRUCTIONS = """\
Builds power-system single-line diagrams in Electrisim from a JSON network spec.

Call get_spec_format once before writing a spec - it has every field, default and
an example. check_network validates a spec and can run a power flow without
touching the diagram; draw_diagram puts it on the Electrisim canvas. Drawing adds
to what is already there, so draw a design once it is settled rather than after
every edit. Elements refer to each other by id, and results come back under the
same ids.
"""

Layout = Literal['auto', 'transmission', 'radial']


def _origins():
    raw = os.getenv('ELECTRISIM_BRIDGE_ORIGINS')
    return tuple(o.strip() for o in raw.split(',')) if raw else DEFAULT_ORIGINS


def _frontend_url():
    return os.getenv('ELECTRISIM_FRONTEND_URL', 'http://127.0.0.1:5501').rstrip('/')


def _spec_error(problems):
    lines = '\n'.join(f'- {p}' for p in problems)
    return ToolError(
        f'The spec has {len(problems)} problem(s). Fix all of them and call again:\n{lines}'
    )


def create_server(backend=None, bridge=None):
    """Build the server. Tests pass their own backend client and bridge."""
    backend = backend or BackendClient()
    if bridge is None:
        bridge = Bridge(port=int(os.getenv('ELECTRISIM_BRIDGE_PORT', DEFAULT_PORT)),
                        allowed_origins=_origins())
        bridge.start()

    server = MCPServer(name='electrisim', title='Electrisim',
                       instructions=INSTRUCTIONS, version='0.1.0')

    def build(spec, **kw):
        try:
            return backend.build_model(spec, **kw)
        except SpecProblems as exc:
            raise _spec_error(exc.problems) from None
        except BackendError as exc:
            raise ToolError(str(exc)) from None

    @server.tool(
        title='Network spec format',
        description='Reference for the JSON network spec: every element, field, '
                    'unit and default, the layouts, and a worked example. '
                    'Read it before writing a spec.',
        annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True,
                                    open_world_hint=False),
    )
    def get_spec_format() -> str:
        return SPEC_FORMAT

    @server.tool(
        title='Check network',
        description='Validate a network spec without drawing it, and optionally run '
                    'a balanced AC power flow. Returns element counts, warnings '
                    '(missing slack, isolated buses, swapped transformer windings) '
                    'and, with run_power_flow, bus voltages, line and transformer '
                    'loadings and any limit violations, all keyed by spec id.',
        annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True,
                                    open_world_hint=False),
    )
    def check_network(
        spec: dict[str, Any],
        run_power_flow: bool = True,
        vm_min_pu: float = 0.95,
        vm_max_pu: float = 1.05,
        max_loading_percent: float = 100.0,
    ) -> dict[str, Any]:
        body = build(spec, run_power_flow=run_power_flow, include_model=False,
                     limits={'vm_min_pu': vm_min_pu, 'vm_max_pu': vm_max_pu,
                             'max_loading_percent': max_loading_percent})
        out = {'valid': True, **body['report']}
        if run_power_flow:
            out['power_flow'] = body.get('power_flow')
        return out

    @server.tool(
        title='Draw diagram',
        description='Draw a network spec as a single-line diagram on the open '
                    'Electrisim canvas. Adds to the current diagram; does not '
                    'replace it. layout: transmission (meshed), radial (plant or '
                    'feeder single-line) or auto. Waits up to wait_seconds for the '
                    'page to finish; if Electrisim is not open, the diagram is '
                    'queued and drawn when it is.',
        annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False,
                                    idempotent_hint=False, open_world_hint=False),
    )
    def draw_diagram(
        spec: dict[str, Any],
        layout: Layout | None = None,
        title: str | None = None,
        wait_seconds: float = 20.0,
    ) -> dict[str, Any]:
        if not bridge.running:
            raise ToolError(f'The local bridge is not running: {bridge.start_error}')
        body = build(spec, include_model=True)
        # The report's `layout` is the spec's hint. Take it out before the report
        # is merged into the result, where it would overwrite the layout drawn.
        report = dict(body['report'])
        spec_layout = report.pop('layout', None)
        chosen = layout or spec_layout or 'auto'
        if chosen == 'auto' and (report.get('counts') or {}).get('trafo3w'):
            # The radial layout cannot place three-winding transformers yet, and
            # auto picks radial for small networks. Don't let it guess wrong.
            chosen = 'transmission'
        job = bridge.submit(body['model'], chosen, title or spec.get('name'))

        if not job.wait(max(0.0, min(float(wait_seconds), 120.0))):
            status = bridge.status()
            where = _frontend_url()
            if not status['page_connected']:
                hint = (f'No Electrisim page is connected. Open {where} and open or create a '
                        f'diagram - the diagram is queued and will be drawn then.')
            elif status['diagram_open'] is False:
                hint = ('Electrisim is open but no diagram is. Open or create one (File > New) '
                        '- the diagram is queued and will be drawn as soon as one is open.')
            else:
                hint = 'The page is connected but has not finished drawing yet.'
            return {'status': 'queued', 'id': job.id, 'layout': chosen,
                    'hint': hint, **report}

        if job.status == 'failed':
            result = job.result or {}
            detail = result.get('error') or 'the page reported a failure'
            extra = result.get('console_errors') or []
            raise ToolError(f'Electrisim could not draw the diagram: {detail}'
                            + (f'\nConsole errors: {extra}' if extra else ''))
        if job.status != 'done':
            raise ToolError(f'The diagram was {job.status} before it could be drawn.')

        result = job.result or {}
        out = {'status': 'drawn', 'id': job.id,
               'layout': result.get('layout') or chosen,
               'cells_added': result.get('cells_added'), **report}
        unplaced = result.get('unplaced') or []
        if unplaced:
            # Drawn, but not faithfully: the radial layout stacked these in one
            # column, so the picture does not show how they connect.
            out['warnings'] = list(out.get('warnings') or []) + [
                f'The radial layout could not place {len(unplaced)} bus(es) and stacked them '
                f'in a column, so the diagram misrepresents how they connect: '
                f'{", ".join(unplaced)}. Radial expects one path from the grid to a single '
                f'collector bus (lines, or one main transformer) with feeders below it. '
                f'Redraw with layout="transmission".'
            ]
            out['unplaced_buses'] = unplaced
        if result.get('console_errors'):
            out['console_errors'] = result['console_errors']
        return out

    @server.tool(
        title='Bridge status',
        description='Whether an Electrisim page is connected to receive diagrams, '
                    'which URL to open if not, and any diagrams waiting to be drawn.',
        annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True,
                                    open_world_hint=False),
    )
    def bridge_status() -> dict[str, Any]:
        return {**bridge.status(), 'frontend_url': _frontend_url(),
                'backend_url': backend.base_url}

    return server


def main():
    logging.basicConfig(level=os.getenv('ELECTRISIM_MCP_LOG', 'INFO'), stream=sys.stderr,
                        format='%(asctime)s %(name)s %(levelname)s %(message)s')
    create_server().run('stdio')


if __name__ == '__main__':
    main()
