# -*- coding: utf-8 -*-
"""
Bearer-token authentication for the Electrisim simulation API.

The simulation service has no authentication: every route is open, and the only
caller identity is a `user_email` string in the request body that the client sets
freely. This module verifies the JWT that api.electrisim.com already issues and
that the frontend already stores, so identity comes from a signed claim instead.

Rollout is staged with ELECTRISIM_AUTH_MODE, because turning verification on for
every caller at once will lock out any client that is not yet sending the header:

    off      (default) No checks. Identical to current behaviour.
    monitor  Verify when a token is present, log the outcome, always allow the
             request through. Use this to confirm tokens are arriving and valid
             before you start rejecting anything.
    enforce  Reject requests with a missing, expired, or invalid token (401).

Configuration (environment):

    ELECTRISIM_AUTH_MODE        off | monitor | enforce      (default: off)
    ELECTRISIM_JWT_SECRET       shared secret, for HS256/HS384/HS512
    ELECTRISIM_JWT_PUBLIC_KEY   PEM public key, for RS*/ES*  (alternative)
    ELECTRISIM_JWT_ALGS         comma-separated              (default: HS256)
    ELECTRISIM_JWT_ISSUER       expected `iss`, optional
    ELECTRISIM_JWT_AUDIENCE     expected `aud`, optional
    ELECTRISIM_JWT_LEEWAY       clock-skew seconds           (default: 30)

The key must match whatever api.electrisim.com signs with. If you do not know,
decode a real token's header: the "alg" field tells you which family to use.

Requires PyJWT:  pip install "PyJWT[crypto]>=2.8.0"
"""

import functools
import json
import logging
import os

from flask import g, jsonify, request

try:
    import jwt as _pyjwt
    from jwt import InvalidTokenError
except ImportError:  # pragma: no cover - surfaced at startup instead
    _pyjwt = None

    class InvalidTokenError(Exception):
        pass


log = logging.getLogger('electrisim.auth')

_VALID_MODES = ('off', 'monitor', 'enforce')


def _mode():
    mode = os.getenv('ELECTRISIM_AUTH_MODE', 'off').strip().lower()
    return mode if mode in _VALID_MODES else 'off'


def _algorithms():
    raw = os.getenv('ELECTRISIM_JWT_ALGS', 'HS256')
    return [a.strip() for a in raw.split(',') if a.strip()]


def _key():
    """Signing key: PEM public key wins over shared secret when both are set."""
    pub = os.getenv('ELECTRISIM_JWT_PUBLIC_KEY')
    if pub:
        # Railway and similar strip newlines from multi-line env vars.
        return pub.replace('\\n', '\n')
    return os.getenv('ELECTRISIM_JWT_SECRET')


def _bearer_token():
    header = request.headers.get('Authorization', '')
    if header[:7].lower() == 'bearer ':
        token = header[7:].strip()
        return token or None
    return None


def verify_token(token):
    """Return the decoded claims. Raises InvalidTokenError on any failure."""
    if _pyjwt is None:
        raise InvalidTokenError('PyJWT is not installed on this server')

    key = _key()
    if not key:
        raise InvalidTokenError(
            'No ELECTRISIM_JWT_SECRET or ELECTRISIM_JWT_PUBLIC_KEY configured'
        )

    options = {
        'require': ['exp'],
        'verify_exp': True,
        'verify_signature': True,
    }
    kwargs = {
        'algorithms': _algorithms(),
        'options': options,
        'leeway': int(os.getenv('ELECTRISIM_JWT_LEEWAY', '30')),
    }
    issuer = os.getenv('ELECTRISIM_JWT_ISSUER')
    audience = os.getenv('ELECTRISIM_JWT_AUDIENCE')
    if issuer:
        kwargs['issuer'] = issuer
    if audience:
        kwargs['audience'] = audience
    else:
        # Tokens issued without an `aud` claim must not fail audience checks.
        options['verify_aud'] = False

    return _pyjwt.decode(token, key, **kwargs)


def authenticated_email(fallback=None):
    """
    Verified caller email, or `fallback` when unauthenticated.

    Prefer this over in_data[...]['user_email'] anywhere identity matters:
    the body field is client-controlled, this comes from a signed claim.
    """
    claims = getattr(g, 'electrisim_claims', None)
    if isinstance(claims, dict):
        email = claims.get('email') or claims.get('sub')
        if email:
            return email
    return fallback


def require_auth(view):
    """Gate a Flask view according to ELECTRISIM_AUTH_MODE."""

    @functools.wraps(view)
    def wrapper(*args, **kwargs):
        g.electrisim_claims = None
        mode = _mode()

        if mode == 'off':
            return view(*args, **kwargs)

        token = _bearer_token()

        if token is None:
            if mode == 'enforce':
                log.warning('auth: no bearer token on %s', request.path)
                return jsonify({
                    'error': 'Authentication required.',
                    'detail': 'Send the api.electrisim.com token as '
                              'Authorization: Bearer <token>.'
                }), 401
            log.info('auth[monitor]: no bearer token on %s', request.path)
            return view(*args, **kwargs)

        try:
            g.electrisim_claims = verify_token(token)
        except InvalidTokenError as exc:
            # Never echo the token or the exception text to the caller.
            log.warning('auth: rejected token on %s: %s', request.path, exc)
            if mode == 'enforce':
                return jsonify({'error': 'Invalid or expired token.'}), 401
            return view(*args, **kwargs)

        if mode == 'monitor':
            log.info('auth[monitor]: accepted %s on %s',
                     authenticated_email('<no email claim>'), request.path)

        return view(*args, **kwargs)

    return wrapper


def startup_report():
    """One line at boot so the active mode is never a surprise."""
    mode = _mode()
    if mode == 'off':
        return ('[auth] ELECTRISIM_AUTH_MODE=off - all simulation routes are '
                'OPEN to unauthenticated callers.')
    if _pyjwt is None:
        return (f'[auth] ELECTRISIM_AUTH_MODE={mode} but PyJWT is NOT installed '
                '- every token will be rejected. pip install "PyJWT[crypto]"')
    if not _key():
        return (f'[auth] ELECTRISIM_AUTH_MODE={mode} but no ELECTRISIM_JWT_SECRET '
                'or ELECTRISIM_JWT_PUBLIC_KEY is set - every token will be rejected.')
    return (f'[auth] ELECTRISIM_AUTH_MODE={mode}, algorithms='
            f'{",".join(_algorithms())}')
