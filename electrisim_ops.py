# -*- coding: utf-8 -*-
"""
Operational hardening for the Electrisim simulation API: request size limit and
error-detail scrubbing.

Two problems this addresses.

1. No request size limit. Flask will buffer a body of any size into memory before
   a view ever runs, so one large POST can exhaust the container.

2. Tracebacks are returned to callers. `app.py` returns
   `{'error': ..., 'details': traceback.format_exc()}` on failure, and several
   analysis modules embed a `traceback` key in their result. That leaks absolute
   filesystem paths, library versions, and code structure.

   The frontend uses these deliberately - ProtectionCoordinationResultsDialog
   renders `results.traceback` in a collapsible block - so they are not removed.
   They are suppressed unless ELECTRISIM_DEBUG_ERRORS=1, and replaced with a
   short reference id that is logged in full server-side. Set the variable in
   development and leave it unset in production.

Configuration (environment):

    ELECTRISIM_MAX_CONTENT_MB   request body cap in MB   (default: 32)
    ELECTRISIM_DEBUG_ERRORS     "1" returns tracebacks to the caller (default: off)

Known limitation: responses already gzipped by the view (app.py compresses
successful simulation results over 1 KB) are passed through untouched. Error
responses are not compressed, so the error paths this targets are covered. If you
later compress error bodies too, scrub before compressing.
"""

import json
import logging
import os
import uuid

from flask import jsonify, request

log = logging.getLogger('electrisim.ops')

# Keys that may carry a traceback or internal diagnostic text.
_SENSITIVE_KEYS = ('traceback', 'details', 'exception', 'diagnostic_traceback')

# Cheap pre-check so we only parse bodies that could contain something.
_MARKERS = tuple(b'"%s"' % k.encode('ascii') for k in _SENSITIVE_KEYS)

# Do not parse absurdly large bodies looking for markers.
_MAX_SCRUB_BYTES = 8 * 1024 * 1024


def _debug_errors():
    return os.getenv('ELECTRISIM_DEBUG_ERRORS', '').strip() == '1'


def _max_content_length():
    try:
        mb = float(os.getenv('ELECTRISIM_MAX_CONTENT_MB', '32'))
    except ValueError:
        mb = 32.0
    return int(mb * 1024 * 1024)


def _scrub(node, ref):
    """Recursively replace sensitive values. Returns True if anything changed."""
    changed = False
    if isinstance(node, dict):
        for key in list(node.keys()):
            if key in _SENSITIVE_KEYS and isinstance(node[key], str):
                log.warning('ops: suppressed %s (ref %s): %s',
                            key, ref, node[key].replace('\n', ' | ')[:2000])
                node[key] = (f'Suppressed. Server reference {ref} - the full text '
                             f'is in the server log. Set ELECTRISIM_DEBUG_ERRORS=1 '
                             f'to return it.')
                changed = True
            else:
                changed = _scrub(node[key], ref) or changed
    elif isinstance(node, list):
        for item in node:
            changed = _scrub(item, ref) or changed
    return changed


def install(app):
    """Register the body cap, the 413 handler, and the error scrubber."""

    app.config['MAX_CONTENT_LENGTH'] = _max_content_length()

    @app.before_request
    def _reject_oversized():
        """
        Reject before the view runs. The simulation view wraps its body in a broad
        `except Exception`, which would otherwise swallow Werkzeug's
        RequestEntityTooLarge and report it as a generic 500.
        """
        limit = app.config.get('MAX_CONTENT_LENGTH') or 0
        length = request.content_length
        if limit and length and length > limit:
            return _too_large(None)
        return None

    @app.errorhandler(413)
    def _too_large(_err):
        limit_mb = app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
        log.warning('ops: rejected oversized body on %s from %s',
                    request.path, request.remote_addr)
        return jsonify({
            'error': f'Request body too large. The limit is {limit_mb:.0f} MB.',
            'detail': 'Reduce the network size, or raise ELECTRISIM_MAX_CONTENT_MB.'
        }), 413

    @app.after_request
    def _scrub_error_details(response):
        if _debug_errors():
            return response
        if response.direct_passthrough or response.is_streamed:
            return response
        # Views compress successful results; error bodies are not compressed.
        if response.headers.get('Content-Encoding'):
            return response
        if 'json' not in (response.content_type or ''):
            return response

        try:
            data = response.get_data()
        except RuntimeError:
            return response

        if not data or len(data) > _MAX_SCRUB_BYTES:
            return response
        if not any(marker in data for marker in _MARKERS):
            return response

        try:
            payload = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            return response

        ref = uuid.uuid4().hex[:12]
        if _scrub(payload, ref):
            response.set_data(json.dumps(payload))
            response.headers['X-Electrisim-Error-Ref'] = ref

        return response

    return app


def startup_report(app):
    mb = app.config.get('MAX_CONTENT_LENGTH', 0) / (1024 * 1024)
    mode = 'RETURNED to callers' if _debug_errors() else 'suppressed (logged server-side)'
    return f'[ops] max request body {mb:.0f} MB; error tracebacks {mode}'
