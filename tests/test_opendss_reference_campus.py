# -*- coding: utf-8 -*-
"""
The OpenDSS load flow and harmonic analysis of the AI campus reference
(tests/reference/reference_ai_campus.diagram_payload.json, its study
parameters swapped for the OpenDSS ones of the drawn transmission grid).

Its gas turbines sit on the delta side of YNd11 step-up units, and OpenDSS
did not converge: the PV generators, connected wye, regulated the phase-to-
ground voltage of a bus with no other ground and drove its zero sequence to
52 pu. The 200th iterate was reported as a result, the DC buses among the AC
ones at 0 pu, and the harmonic analysis failed with a server error. Nor did
OpenDSS have the converters - 49 MW of data halls behind SSTs and rectifiers.

The oracle is pandapower's load flow of the same payload with its external
grid behind the short-circuit impedance OpenDSS feeds the grid through,
worked out here from the grid's s_sc_max_mva and rx_max.
"""
import copy
import json
import math
import os

import pytest

import opendss_electrisim

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE_DIR = os.path.join(HERE, 'reference')

VM_TOL = 0.005   # pu, ~0.5 %


@pytest.fixture
def opendss_scratch(tmp_path):
    """OpenDSS saves solved voltages in its data path: keep it out of the backend directory."""
    import opendssdirect as dss
    before = dss.Basic.DataPath()
    dss.Basic.DataPath(str(tmp_path))
    yield
    dss.Basic.DataPath(before)


def _reference(name):
    with open(os.path.join(REFERENCE_DIR, name), encoding='utf-8') as handle:
        return json.load(handle)


def _campus(study=None):
    """The campus payload; with study 'opendss' or 'harmonic', those OpenDSS parameters."""
    payload = _reference('reference_ai_campus.diagram_payload.json')
    if study:
        payload['0'] = _reference(f'reference_transmission.diagram_{study}_payload.json')['0']
    return payload


def _post(client, quiet, payload):
    with quiet():
        response = client.post('/', json=payload)
    assert response.status_code == 200, response.get_data(as_text=True)[:400]
    result = json.loads(response.get_data(as_text=True).splitlines()[-1])
    return result


def _elements(payload, prefix):
    return [v for v in payload.values() if isinstance(v, dict) and str(v.get('typ', '')).startswith(prefix)]


def _behind_its_impedance(payload):
    """The payload with its external grid behind its Thevenin impedance, as OpenDSS has it."""
    payload = copy.deepcopy(payload)
    (grid,) = _elements(payload, 'External Grid')
    (bus,) = [v for v in payload.values() if isinstance(v, dict) and v.get('name') == grid['bus']]
    vn, s_sc, rx = float(bus['vn_kv']), float(grid['s_sc_max_mva']), float(grid['rx_max'])
    z = vn ** 2 / s_sc
    x = z / math.hypot(1.0, rx)
    z_base = vn ** 2 / 100.0
    payload['oracle_source'] = {'typ': 'Bus', 'name': 'oracle_source', 'id': 'oracle_source', 'vn_kv': str(vn)}
    payload['oracle_z'] = {'typ': 'Impedance', 'name': 'oracle_z', 'id': 'oracle_z', 'busFrom': 'oracle_source',
                           'busTo': grid['bus'], 'rft_pu': rx * x / z_base, 'xft_pu': x / z_base,
                           'sn_mva': 100.0, 'in_service': True}
    grid['bus'] = 'oracle_source'
    return payload


@pytest.fixture(scope='module')
def opendss_result(client):
    import contextlib
    import io
    import opendssdirect as dss
    import tempfile
    before = dss.Basic.DataPath()
    with tempfile.TemporaryDirectory() as scratch:
        dss.Basic.DataPath(scratch)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                response = client.post('/', json=_campus('opendss'))
        finally:
            dss.Basic.DataPath(before)
    assert response.status_code == 200, response.get_data(as_text=True)[:400]
    return json.loads(response.get_data(as_text=True).splitlines()[-1])


def test_opendss_load_flow_converges_with_the_gas_turbines(opendss_result):
    """
    Converged, each gas turbine at its 12 MW and holding its 13.8 kV bus at
    its 1.0 pu setpoint. It was 11.2 MW absorbing 17 Mvar at 0.98 pu after
    200 iterations - and every generator read 1.0 pu, whatever its bus held.
    """
    result = opendss_result
    assert not result.get('error'), result.get('error')
    warnings = result.get('warnings') or []
    assert not [w for w in warnings if 'did not converge' in w], warnings
    payload = _campus()
    for gen in _elements(payload, 'Generator'):
        (row,) = [g for g in result['generators'] if g['name'] == gen['name']]
        assert row['p_mw'] == pytest.approx(float(gen['p_mw']), abs=0.01), gen['userFriendlyName']
        assert row['vm_pu'] == pytest.approx(float(gen['vm_pu']), abs=1e-3), gen['userFriendlyName']
        (bus,) = [b for b in result['busbars'] if b['name'] == gen['bus']]
        assert bus['vm_pu'] == pytest.approx(row['vm_pu'], abs=1e-6)


def test_opendss_leaves_the_dc_buses_out_of_the_ac_results(opendss_result):
    """
    No DC bus among the AC busbars - they were there at 0 pu - and every AC
    bus is; the warning says the DC network is left out and how each
    converter stands in for it.
    """
    payload = _campus()
    rows = {b['name'] for b in opendss_result['busbars']}
    dc = {b['name'] for b in _elements(payload, 'DC Bus')}
    ac = {b['name'] for b in payload.values()
          if isinstance(b, dict) and str(b.get('typ', '')).startswith('Bus')}
    assert not rows & dc, sorted(rows & dc)
    assert rows == ac
    assert all(b['vm_pu'] > 0.9 for b in opendss_result['busbars'])
    (warning,) = [w for w in opendss_result.get('warnings') or [] if 'DC network is left out' in w]
    assert '23 DC buses' in warning
    for converter in ('Hall 1 SST U1', 'Hall 2 rectifier equivalent', 'PV 1 PCS', 'BESS 1 PCS'):
        assert converter in warning


def test_opendss_matches_pandapower_with_the_grid_behind_its_impedance(client, quiet, opendss_result):
    """
    Every bus within 0.5 % of pandapower's load flow of the same payload, the
    grid behind the impedance OpenDSS feeds it through, and the utility
    supplying the same power. The data halls' SSTs and rectifiers and the PV,
    SOFC and BESS PCS are in OpenDSS as the loads and sources their AC sides
    are; without them OpenDSS exported 4 MW where pandapower imports 23.5.
    """
    want = _post(client, quiet, _behind_its_impedance(_campus()))
    assert not want.get('error'), want.get('error')
    want_vm = {b['name']: float(b['vm_pu']) for b in want['busbars']}
    differ = [f"{b['name']}: {b['vm_pu']:.5f} pu in OpenDSS, {want_vm[b['name']]:.5f} in pandapower"
              for b in opendss_result['busbars'] if abs(b['vm_pu'] - want_vm[b['name']]) > VM_TOL]
    assert not differ, '\n'.join(differ)
    # pandapower's grid row is at the source, behind the impedance's 0.12 MW of losses.
    (got_grid,), (want_grid,) = opendss_result['externalgrids'], want['externalgrids']
    assert got_grid['p_mw'] == pytest.approx(want_grid['p_mw'], abs=0.25)


def test_opendss_harmonics_run_on_the_campus(client, quiet, opendss_scratch, opendss_result):
    """
    The harmonic analysis runs - it failed with "#487 Circuit must be solved
    in a fundamental frequency power flow" and a server error - and reports
    distortion at every AC bus. Every harmonic solve was NaN (0 % THD) while
    the delta windings with nothing grounded beyond them floated, and no bus
    behind a transformer drawn with hv_bus / lv_bus had a monitor.
    """
    result = _post(client, quiet, _campus('harmonic'))
    assert not result.get('error'), result.get('error')
    assert result['harmonic_analysis']['executed']
    load_flow = {b['name']: b['vm_pu'] for b in opendss_result['busbars']}
    assert {b['name'] for b in result['busbars']} == set(load_flow)
    for bus in result['busbars']:
        assert bus['vm_pu'] == pytest.approx(load_flow[bus['name']], abs=1e-4), bus['name']
        assert bus['vthd_percent'] is not None and bus['vthd_percent'] > 0, bus['name']
    # The distortion is the campus loads': worst at their own 0.48 kV buses.
    worst = max(result['busbars'], key=lambda b: b['vthd_percent'])
    assert worst['name'] in {'mxCell_194', 'mxCell_196'}


def _stuck(algorithm, max_iterations):
    """One solve plan of two Newton iterations: finite voltages, not converged."""
    return [{'label': 'Newton, MaxIterations=2', 'algorithm': 'Newton', 'max_iterations': 2, 'gen_vminpu': None}]


def test_unconverged_load_flow_says_so(client, quiet, opendss_scratch, monkeypatch):
    """
    A load flow that stops short of convergence reports OpenDSS's last
    iterate with a warning saying so; it said nothing.
    """
    monkeypatch.setattr(opendss_electrisim, '_opendss_snapshot_solve_plans', _stuck)
    result = _post(client, quiet, _campus('opendss'))
    assert not result.get('error'), result.get('error')
    assert [w for w in result.get('warnings') or [] if 'did not converge' in w]


def test_unconverged_fundamental_stops_the_harmonics_with_a_message(client, quiet, opendss_scratch, monkeypatch):
    """With no converged fundamental the harmonic analysis says so, not a server error."""
    monkeypatch.setattr(opendss_electrisim, '_opendss_snapshot_solve_plans', _stuck)
    result = _post(client, quiet, _campus('harmonic'))
    assert 'did not converge' in (result.get('error') or ''), result
