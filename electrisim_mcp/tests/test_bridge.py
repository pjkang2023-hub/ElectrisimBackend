# -*- coding: utf-8 -*-
"""The local bridge: job hand-off, and the guards that keep other pages out."""

from _support import PAGE_ORIGIN, request

from electrisim_mcp.bridge import Bridge


def test_a_job_is_handed_over_once(bridge):
    assert request(bridge, 'GET', '/next')[0] == 204
    job = bridge.submit('{"m": 1}', 'radial', 'Plant')

    status, _, body = request(bridge, 'GET', '/next')
    assert status == 200
    assert body == {'id': job.id, 'model': '{"m": 1}', 'layout': 'radial', 'title': 'Plant'}
    assert job.status == 'delivered'
    # Not offered a second time, even before it is acknowledged.
    assert request(bridge, 'GET', '/next')[0] == 204


def test_acknowledgement_completes_the_job(bridge):
    job = bridge.submit('{}', 'auto')
    request(bridge, 'GET', '/next')
    status, _, body = request(bridge, 'POST', '/ack', {
        'id': job.id, 'ok': True, 'cellsAdded': 12, 'layout': 'vertical',
        'errors': ['x' * 900],
    })
    assert (status, body) == (200, {'accepted': True})
    assert job.wait(1)
    assert job.status == 'done'
    assert job.result['cells_added'] == 12
    # Console noise is truncated before it reaches the model.
    assert len(job.result['console_errors'][0]) == 500


def test_failed_drawing_is_recorded_as_failed(bridge):
    job = bridge.submit('{}', 'auto')
    request(bridge, 'GET', '/next')
    request(bridge, 'POST', '/ack', {'id': job.id, 'ok': False, 'error': 'editor not ready'})
    assert job.wait(1)
    assert (job.status, job.result['error']) == ('failed', 'editor not ready')


def test_ack_for_a_job_never_delivered_is_refused(bridge):
    job = bridge.submit('{}', 'auto')
    assert request(bridge, 'POST', '/ack', {'id': job.id, 'ok': True})[0] == 409
    assert request(bridge, 'POST', '/ack', {'id': 'd999', 'ok': True})[0] == 409
    assert job.status == 'pending'


def test_newer_design_supersedes_one_still_waiting(bridge):
    first = bridge.submit('{"v": 1}', 'auto')
    second = bridge.submit('{"v": 2}', 'auto')
    assert first.wait(0) and first.status == 'superseded'
    assert request(bridge, 'GET', '/next')[2]['id'] == second.id


def test_page_without_a_diagram_open_leaves_the_job_queued(bridge):
    # Drawing into the placeholder graph behind the start dialog loses the
    # diagram, so a not-ready poll must not take the job.
    job = bridge.submit('{}', 'auto')
    assert request(bridge, 'GET', '/next?ready=0')[0] == 204
    assert job.status == 'pending'
    status = bridge.status()
    assert status['page_connected'] is True and status['diagram_open'] is False
    # Once a diagram is open, the same job is handed over.
    assert request(bridge, 'GET', '/next')[2]['id'] == job.id
    assert bridge.status()['diagram_open'] is True


def test_poll_marks_the_page_connected(bridge):
    assert bridge.status()['page_connected'] is False
    request(bridge, 'GET', '/next')
    status = bridge.status()
    assert status['page_connected'] is True
    assert status['seconds_since_last_poll'] < 2


# --- guards ----------------------------------------------------------------

def test_other_origins_are_refused(bridge):
    bridge.submit('{"secret": true}', 'auto')
    status, headers, body = request(bridge, 'GET', '/next', origin='https://evil.example')
    assert status == 403
    assert 'ELECTRISIM_BRIDGE_ORIGINS' in body['error']
    assert headers.get('Access-Control-Allow-Origin') is None
    # And the refused poll did not consume the job.
    assert request(bridge, 'GET', '/next')[0] == 200


def test_allowed_origin_gets_cors_headers(bridge):
    _, headers, _ = request(bridge, 'GET', '/health')
    assert headers['Access-Control-Allow-Origin'] == PAGE_ORIGIN


def test_requests_without_origin_are_allowed(bridge):
    # curl and other local tools send no Origin; only browsers do.
    assert request(bridge, 'GET', '/health', origin=None)[0] == 200


def test_non_loopback_host_is_refused(bridge):
    # A DNS-rebinding page reaches 127.0.0.1 under its own hostname.
    status, _, _ = request(bridge, 'GET', '/next', host=f'attacker.example:{bridge.port}')
    assert status == 421


def test_preflight_allows_the_page_to_post(bridge):
    status, headers, _ = request(bridge, 'OPTIONS', '/ack',
                                 headers={'Access-Control-Request-Method': 'POST'})
    assert status == 204
    assert 'POST' in headers['Access-Control-Allow-Methods']
    assert headers['Access-Control-Allow-Origin'] == PAGE_ORIGIN


def test_oversized_ack_is_refused(bridge):
    status, _, _ = request(bridge, 'POST', '/ack', body=b'{' + b' ' * 70000 + b'}')
    assert status == 413


def test_malformed_ack_is_refused(bridge):
    assert request(bridge, 'POST', '/ack', body=b'[1, 2]')[0] == 400
    assert request(bridge, 'POST', '/ack', body=b'not json')[0] == 400


def test_second_bridge_on_a_taken_port_reports_why(bridge):
    other = Bridge(port=bridge.port)
    assert other.start() is False
    assert 'Another Electrisim MCP server is probably running' in other.start_error
    assert other.running is False
