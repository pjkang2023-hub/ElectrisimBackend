# -*- coding: utf-8 -*-
"""
Calls the Electrisim backend's /build-model route.

The backend owns pandapower; this side only ships the spec over and reads back
the drawable model, the build report and, when asked, power-flow results.
Standard library only, so the MCP server needs nothing from the backend's
environment.
"""

import json
import os
import urllib.error
import urllib.request

DEFAULT_BACKEND_URL = 'http://127.0.0.1:5000'


class BackendError(Exception):
    """A failure worth showing to the caller verbatim."""


class SpecProblems(BackendError):
    """The backend refused the spec. `problems` lists every reason."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__('; '.join(self.problems))


class BackendClient:
    def __init__(self, base_url=None, token=None, timeout=60.0):
        self.base_url = (base_url or os.getenv('ELECTRISIM_BACKEND_URL')
                         or DEFAULT_BACKEND_URL).rstrip('/')
        # Needed only when the backend runs with ELECTRISIM_AUTH_MODE=enforce.
        self.token = token if token is not None else os.getenv('ELECTRISIM_TOKEN')
        self.timeout = timeout

    def build_model(self, spec, run_power_flow=False, include_model=True, limits=None):
        body = {'spec': spec, 'run_power_flow': bool(run_power_flow),
                'include_model': bool(include_model)}
        body.update({k: v for k, v in (limits or {}).items() if v is not None})
        return self._post('/build-model', body)

    def _post(self, path, body):
        data = json.dumps(body).encode('utf-8')
        req = urllib.request.Request(self.base_url + path, data=data, method='POST',
                                     headers={'Content-Type': 'application/json'})
        if self.token:
            req.add_header('Authorization', f'Bearer {self.token}')
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except urllib.error.HTTPError as exc:
            payload = _json_or_none(exc)
            if exc.code == 400 and payload and payload.get('problems'):
                raise SpecProblems(payload['problems']) from None
            if exc.code == 401:
                raise BackendError(
                    'The backend requires authentication (ELECTRISIM_AUTH_MODE=enforce). '
                    'Set ELECTRISIM_TOKEN for the MCP server to a valid api.electrisim.com token.'
                ) from None
            if exc.code == 404:
                raise BackendError(
                    f'{self.base_url} has no {path} route. The backend is older than '
                    f'this MCP server - restart it from a checkout that includes electrisim_sld.py.'
                ) from None
            detail = (payload or {}).get('error') or exc.reason
            raise BackendError(f'Backend returned HTTP {exc.code}: {detail}') from None
        except urllib.error.URLError as exc:
            raise BackendError(
                f'Cannot reach the Electrisim backend at {self.base_url} ({exc.reason}). '
                f'Start it (start-backend.cmd, or start-all.cmd), or set ELECTRISIM_BACKEND_URL.'
            ) from None
        except TimeoutError:
            raise BackendError(
                f'The backend at {self.base_url} did not answer within {self.timeout:.0f} s.'
            ) from None


def _json_or_none(http_error):
    try:
        return json.loads(http_error.read().decode('utf-8'))
    except Exception:
        return None
