# -*- coding: utf-8 -*-
"""Shared fixtures. Keeps the backend quiet and the app importable from tests/."""

import io
import os
import sys
import contextlib

import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# The simulation routes must stay open for tests; auth is covered separately.
os.environ.setdefault('ELECTRISIM_AUTH_MODE', 'off')
# Script import is gated by default; the import goldens need it.
os.environ.setdefault('ELECTRISIM_ALLOW_SCRIPT_IMPORT', '1')


def pytest_addoption(parser):
    parser.addoption(
        '--regen-golden', action='store_true', default=False,
        help='Rewrite golden files from the current implementation. '
             'Review the resulting diff before committing it.'
    )


@pytest.fixture(scope='session')
def regen(pytestconfig):
    return pytestconfig.getoption('--regen-golden')


@pytest.fixture(scope='session')
def client():
    import app as flask_app
    flask_app.app.config.update(TESTING=True)
    return flask_app.app.test_client()


@pytest.fixture
def quiet():
    """
    Swallow the backend's very chatty stdout/stderr.

    app.py writes progress directly to the streams (`_console`) rather than
    through logging, so pytest's capture alone does not keep output readable.
    """
    @contextlib.contextmanager
    def _quiet():
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            yield buf
    return _quiet
