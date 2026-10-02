# -*- coding: utf-8 -*-
"""
Local bridge between the MCP server and an open Electrisim tab.

The MCP server runs as a child of the MCP client and cannot reach into a
browser. This holds diagrams it wants drawn; a small script in the page
(js/electrisim/utils/mcpBridge.js) polls for them, draws each one, and reports
back. Nothing here pushes into the browser - the page pulls.

HTTP surface, all on 127.0.0.1:

    GET  /health   whether the bridge is up, and when the page last polled
    GET  /next     the oldest pending diagram (200), or nothing (204)
    POST /ack      {id, ok, cellsAdded, layout, error, errors}

Guard rails, because any web page the user has open can send requests to
localhost:

  * Bound to loopback only.
  * The Host header must name loopback, which defeats DNS rebinding (a public
    hostname pointed at 127.0.0.1 after the page has loaded).
  * A request carrying an Origin must come from an allowed origin - by default
    the local Electrisim frontend - or it is refused. Without that, any site
    could read the diagram being drawn or acknowledge it falsely.

Only the newest undelivered diagram is kept: if the page is closed while
several are submitted, opening it draws the latest design, not every draft.
"""

import itertools
import json
import logging
import os
import socket
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger('electrisim.mcp.bridge')

DEFAULT_PORT = 5503
DEFAULT_ORIGINS = ('http://127.0.0.1:5501', 'http://localhost:5501')

#: A page that polled this recently counts as connected.
CONNECTED_WITHIN_S = 6.0
#: An undelivered diagram older than this is dropped rather than drawn late.
PENDING_TTL_S = 15 * 60.0
#: A delivered diagram not acknowledged within this is given up on. It is not
#: re-offered: the page may have drawn it and died before acknowledging, and a
#: duplicate diagram is worse than a missing one the caller already knows about.
DELIVERED_TTL_S = 120.0

_MAX_ACK_BYTES = 64 * 1024
#: An oversized body is read and discarded up to this much before answering 413.
#: Closing a socket with unread data makes Windows reset the connection, and the
#: client would never see the status. Beyond this the client is not worth waiting for.
_MAX_DRAIN_BYTES = 1024 * 1024


class _ExclusiveServer(ThreadingHTTPServer):
    """
    Refuses to share its port.

    The stock server sets SO_REUSEADDR, and on Windows that lets a second process
    bind a port already in use. A second MCP server - another Claude session -
    would then start without complaint, and the page's polls would be split
    between the two, so diagrams from either would go missing at random.
    """

    allow_reuse_address = os.name != 'nt'

    def server_bind(self):
        if os.name == 'nt' and hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class Job:
    """One diagram on its way to the page."""

    def __init__(self, job_id, model, layout, title):
        self.id = job_id
        self.model = model
        self.layout = layout
        self.title = title
        self.created = time.monotonic()
        self.delivered_at = None
        self.status = 'pending'  # pending | delivered | done | failed | superseded | expired
        self.result = None
        self._done = threading.Event()

    def finish(self, status, result=None):
        self.status = status
        self.result = result
        self._done.set()

    def wait(self, timeout):
        """Block until the page acknowledges, or timeout. True if acknowledged."""
        return self._done.wait(timeout)

    def describe(self):
        return {'id': self.id, 'status': self.status, 'title': self.title,
                'layout': self.layout, 'result': self.result}


class Bridge:
    def __init__(self, port=DEFAULT_PORT, allowed_origins=DEFAULT_ORIGINS, host='127.0.0.1'):
        self.host = host
        self.port = int(port)
        self.allowed_origins = frozenset(o.rstrip('/') for o in allowed_origins if o)
        self._jobs = OrderedDict()
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._last_poll = None
        # Whether the polling page has a diagram open; None until it first polls.
        self._page_ready = None
        self._server = None
        self._thread = None
        self.start_error = None

    # --- lifecycle -------------------------------------------------------

    def start(self):
        """Start serving in a daemon thread. Returns False if the port is taken."""
        try:
            self._server = _ExclusiveServer((self.host, self.port), self._handler_class())
        except OSError as exc:
            self.start_error = (
                f'could not listen on {self.host}:{self.port} ({exc}). Another Electrisim '
                f'MCP server is probably running; set ELECTRISIM_BRIDGE_PORT to use another port '
                f'and point the page at it.'
            )
            log.warning(self.start_error)
            return False
        self._server.daemon_threads = True
        if self.port == 0:  # tests ask for an ephemeral port
            self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        name='electrisim-bridge', daemon=True)
        self._thread.start()
        log.info('bridge listening on http://%s:%s', self.host, self.port)
        return True

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    @property
    def running(self):
        return self._server is not None

    @property
    def url(self):
        return f'http://{self.host}:{self.port}'

    # --- jobs ------------------------------------------------------------

    def submit(self, model, layout, title=None):
        """Queue a diagram. Supersedes any diagram still waiting to be picked up."""
        with self._lock:
            self._expire_locked()
            for job in self._jobs.values():
                if job.status == 'pending':
                    job.finish('superseded')
            job = Job(f'd{next(self._ids)}', model, layout, title)
            self._jobs[job.id] = job
            # Keep the history short; only recent jobs are ever looked up.
            while len(self._jobs) > 50:
                self._jobs.popitem(last=False)
            return job

    def _next_locked(self, ready=True):
        self._last_poll = time.monotonic()
        self._page_ready = ready
        self._expire_locked()
        if not ready:
            # The page is here but has no diagram open. Leave the job queued:
            # drawing it now would land in a placeholder graph and be discarded.
            return None
        for job in self._jobs.values():
            if job.status == 'pending':
                job.status = 'delivered'
                job.delivered_at = time.monotonic()
                return job
        return None

    def _ack_locked(self, payload):
        job = self._jobs.get(str(payload.get('id', '')))
        if job is None or job.status != 'delivered':
            return False
        ok = bool(payload.get('ok'))
        result = {
            'cells_added': payload.get('cellsAdded'),
            'layout': payload.get('layout'),
            'unplaced': [str(b)[:200] for b in (payload.get('unplaced') or [])][:200],
            'error': payload.get('error'),
            'console_errors': [str(e)[:500] for e in (payload.get('errors') or [])][:20],
        }
        job.finish('done' if ok else 'failed', result)
        return True

    def _expire_locked(self):
        now = time.monotonic()
        for job in self._jobs.values():
            if job.status == 'pending' and now - job.created > PENDING_TTL_S:
                job.finish('expired')
            elif job.status == 'delivered' and now - job.delivered_at > DELIVERED_TTL_S:
                job.finish('expired')

    def status(self):
        with self._lock:
            self._expire_locked()
            age = None if self._last_poll is None else time.monotonic() - self._last_poll
            pending = [j.id for j in self._jobs.values() if j.status == 'pending']
            ready = self._page_ready
        connected = age is not None and age <= CONNECTED_WITHIN_S
        return {
            'running': self.running,
            'url': self.url,
            'error': self.start_error,
            'page_connected': connected,
            # Meaningful only while connected: False means Electrisim is open
            # but no diagram is, so queued diagrams wait.
            'diagram_open': ready if connected else None,
            'seconds_since_last_poll': None if age is None else round(age, 1),
            'pending': pending,
        }

    # --- HTTP ------------------------------------------------------------

    def _handler_class(self):
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            server_version = 'ElectrisimBridge/1'

            def log_message(self, fmt, *args):  # route through logging, not stderr
                log.debug('%s - %s', self.address_string(), fmt % args)

            # -- guards --

            def _origin_ok(self):
                origin = self.headers.get('Origin')
                return origin is None or origin.rstrip('/') in bridge.allowed_origins

            def _host_ok(self):
                # Read the port per request: a test binds port 0 and learns the
                # real one only after the server exists.
                host = (self.headers.get('Host') or '').lower()
                return host in {f'127.0.0.1:{bridge.port}', f'localhost:{bridge.port}',
                                f'[::1]:{bridge.port}'}

            def _guard(self):
                if not self._host_ok():
                    self._send(421, {'error': 'Host must be loopback.'})
                    return False
                if not self._origin_ok():
                    self._send(403, {'error': 'Origin not allowed. Add it to '
                                              'ELECTRISIM_BRIDGE_ORIGINS.'})
                    return False
                return True

            def _cors(self):
                origin = self.headers.get('Origin')
                if origin and origin.rstrip('/') in bridge.allowed_origins:
                    self.send_header('Access-Control-Allow-Origin', origin)
                    self.send_header('Vary', 'Origin')

            def _drain(self, n):
                while n > 0:
                    chunk = self.rfile.read(min(n, 65536))
                    if not chunk:
                        break
                    n -= len(chunk)

            def _send(self, code, body=None):
                data = b'' if body is None else json.dumps(body).encode('utf-8')
                self.send_response(code)
                self._cors()
                self.send_header('Cache-Control', 'no-store')
                if body is not None:
                    self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                if data:
                    self.wfile.write(data)

            # -- verbs --

            def do_OPTIONS(self):
                if not self._guard():
                    return
                self.send_response(204)
                self._cors()
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                self.send_header('Access-Control-Max-Age', '600')
                self.send_header('Content-Length', '0')
                self.end_headers()

            def do_GET(self):
                if not self._guard():
                    return
                path = self.path.split('?', 1)[0]
                if path == '/health':
                    self._send(200, {'ok': True, **bridge.status()})
                elif path == '/next':
                    query = self.path.split('?', 1)[1] if '?' in self.path else ''
                    ready = 'ready=0' not in query.split('&')
                    with bridge._lock:
                        job = bridge._next_locked(ready)
                    if job is None:
                        self._send(204)
                    else:
                        self._send(200, {'id': job.id, 'model': job.model,
                                         'layout': job.layout, 'title': job.title})
                else:
                    self._send(404, {'error': 'Not found.'})

            def do_POST(self):
                if not self._guard():
                    return
                if self.path.split('?', 1)[0] != '/ack':
                    self._send(404, {'error': 'Not found.'})
                    return
                try:
                    length = int(self.headers.get('Content-Length') or 0)
                except ValueError:
                    length = -1
                if not 0 < length <= _MAX_ACK_BYTES:
                    if length > _MAX_ACK_BYTES:
                        self._drain(min(length, _MAX_DRAIN_BYTES))
                    self._send(413 if length > _MAX_ACK_BYTES else 400,
                               {'error': 'Body must be a JSON object under 64 KB.'})
                    return
                try:
                    payload = json.loads(self.rfile.read(length).decode('utf-8'))
                except (ValueError, UnicodeDecodeError):
                    payload = None
                if not isinstance(payload, dict):
                    self._send(400, {'error': 'Body must be a JSON object.'})
                    return
                with bridge._lock:
                    accepted = bridge._ack_locked(payload)
                self._send(200 if accepted else 409,
                           {'accepted': accepted} if accepted else
                           {'error': 'No delivered diagram with that id.'})

        return Handler
