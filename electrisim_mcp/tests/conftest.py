# -*- coding: utf-8 -*-
r"""
Fixtures for the MCP server's own suite. Run it in the MCP environment, from
this directory:

    ..\..\.venv-mcp\Scripts\python.exe -m pytest

A fake backend stands in for /build-model and a fake page for the browser, so
these exercise the server and bridge without pandapower or a browser.
"""

import pytest
from _support import PAGE_ORIGIN, FakeBackend

from electrisim_mcp.bridge import Bridge


@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.fixture
def backend():
    return FakeBackend()


@pytest.fixture
def bridge():
    b = Bridge(port=0, allowed_origins=(PAGE_ORIGIN,))
    assert b.start()
    yield b
    b.stop()
