# -*- coding: utf-8 -*-
"""Fakes for the MCP suite: a backend that answers like /build-model, and a page
that polls the bridge like mcpBridge.js."""

import json
import threading
import time
import urllib.error
import urllib.request

from electrisim_mcp.backend_client import BackendError, SpecProblems

PAGE_ORIGIN = 'http://127.0.0.1:5501'

MODEL = json.dumps({'_object': {'bus': {'_object': '{"data": []}'}}})


class FakeBackend:
    """Answers build_model() the way /build-model would."""

    base_url = 'http://fake-backend'

    def __init__(self):
        self.calls = []
        self.problems = None
        self.error = None
        self.layout = None

    def build_model(self, spec, run_power_flow=False, include_model=True, limits=None):
        self.calls.append({'spec': spec, 'run_power_flow': run_power_flow,
                           'include_model': include_model, 'limits': limits})
        if self.problems:
            raise SpecProblems(self.problems)
        if self.error:
            raise BackendError(self.error)
        body = {'report': {'warnings': [], 'counts': {'bus': 2}, 'layout': self.layout}}
        if include_model:
            body['model'] = MODEL
        if run_power_flow:
            body['power_flow'] = {'converged': True, 'summary': {'voltage_issues': []}}
        return body




def request(bridge, method, path, body=None, origin=PAGE_ORIGIN, host=None, headers=None):
    """Talk to the bridge the way a page would. Returns (status, headers, json-or-None)."""
    data = None if body is None else (body if isinstance(body, bytes)
                                      else json.dumps(body).encode('utf-8'))
    req = urllib.request.Request(bridge.url + path, data=data, method=method)
    if origin:
        req.add_header('Origin', origin)
    if host:
        req.add_header('Host', host)
    if data is not None:
        req.add_header('Content-Type', 'application/json')
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            raw = resp.read()
            return resp.status, resp.headers, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        return exc.code, exc.headers, (json.loads(raw) if raw else None)


class FakePage:
    """Polls the bridge like mcpBridge.js and acknowledges with a chosen outcome."""

    def __init__(self, bridge, outcome=None):
        self.bridge = bridge
        self.outcome = outcome or {'ok': True, 'cellsAdded': 19, 'layout': 'vertical'}
        self.drawn = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            status, _, job = request(self.bridge, 'GET', '/next')
            if status == 200:
                self.drawn.append(job)
                request(self.bridge, 'POST', '/ack', {'id': job['id'], **self.outcome})
            time.sleep(0.05)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(2)
