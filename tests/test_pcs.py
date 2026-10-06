"""
The PCS: a battery, flywheel, SOFC system or PV array on an AC bus, through
a grid-following or grid-forming inverter - in the load flow on the grid and
islanded (droop sharing), and in the IEC and ANSI short-circuit studies as a
current source at its limit.
"""
import json
import math

import pytest

import der_electrisim as der
from test_dc_elements import _drawn_request, _with
from test_reference_grids import ANSI_PARAMS

LF = {'typ': 'PowerFlowPandaPower Parameters', 'frequency': '50', 'algorithm': 'nr', 'calculate_voltage_angles': 'auto',
      'initialization': 'auto', 'user_email': 't@t'}
SC = {'typ': 'ShortCircuitPandaPower Parameters', 'fault_type': '3ph', 'fault_location': 'max', 'fault_bus_mode': 'all',
      'fault_bus_ids': [], 'fault_bus_names': [], 'fault_impedance': '6', 'topology': 'auto', 'tk_s': '1',
      'r_fault_ohm': '0', 'x_fault_ohm': '0', 'inverse_y': 'True', 'user_email': 't@t'}
ETA = 0.98


def _pcs(name, bus, source=None, **fields):
    row = {'typ': 'PCS0', 'name': name, 'id': f'cell-{name}', 'userFriendlyName': name.upper(), 'bus': bus,
           'der': source, 's_rated_mva': '0.5', 'efficiency_percent': str(100 * ETA), 'control': 'grid_following'}
    row.update({k: str(v) for k, v in fields.items()})
    return row


def _der(kind, name, bus=None, **fields):
    row = {'typ': f'{kind}0', 'name': name, 'id': f'cell-{name}', 'userFriendlyName': name.upper(), 'bus': bus}
    row.update({k: str(v) for k, v in fields.items()})
    return row


def _bus(name, kv, dc=False):
    return {'typ': 'DC Bus9' if dc else 'Bus0', 'name': name, 'id': f'cell-{name}', 'userFriendlyName': name.upper(),
            'vn_kv': str(kv)}


LINE = {'typ': 'Line0', 'name': 'l1', 'id': 'cell-l1', 'userFriendlyName': 'L1', 'busFrom': 'a', 'busTo': 'b',
        'length_km': '0.1', 'parallel': '1', 'df': '1', 'in_service': 'true', 'r_ohm_per_km': '0.1',
        'x_ohm_per_km': '0.08', 'c_nf_per_km': '0', 'g_us_per_km': '0', 'max_i_ka': '2', 'type': 'cs',
        'r0_ohm_per_km': '0.4', 'x0_ohm_per_km': '0.3', 'c0_nf_per_km': '0', 'endtemp_degree': '80'}
GRID = {'typ': 'External Grid0', 'name': 'g', 'id': 'cell-g', 'userFriendlyName': 'G', 'bus': 'a', 'vm_pu': '1',
        'va_degree': '0', 's_sc_max_mva': '10', 's_sc_min_mva': '10', 'rx_max': '0.1', 'rx_min': '0.1',
        'r0x0_max': '0.1', 'x0x_max': '1', 'r0x0_min': '0.1', 'x0x_min': '1', 'in_service': 'true'}
LOAD = {'typ': 'Load0', 'name': 'ld', 'id': 'cell-ld', 'userFriendlyName': 'LD', 'bus': 'b', 'p_mw': '0.6',
        'q_mvar': '0.1', 'const_z_percent': '0', 'const_i_percent': '0', 'sn_mva': '0', 'scaling': '1',
        'type': 'wye', 'in_service': 'true'}


def _two_buses(*elements, grid=True, params=LF):
    """0.4 kV buses A and B, 100 m apart; an external grid on A unless islanded; a 0.6 MW load on B."""
    request = {'0': dict(params), '1': _bus('a', 0.4), '2': _bus('b', 0.4), '3': dict(LINE), '4': dict(LOAD)}
    if grid:
        request['5'] = dict(GRID)
    return _with(request, *elements)


def _post(client, quiet, request):
    with quiet():
        response = client.post('/', json=request)
    result = json.loads(response.get_data(as_text=True))
    assert not result.get('error'), result.get('message') or result.get('exception')
    return result


def _by_id(rows):
    return {r['id']: r for r in rows}


def _lv(request):
    return next(v for v in request.values() if isinstance(v, dict) and v.get('userFriendlyName') == 'LV network A')['name']


# --- Load flow on the grid ---------------------------------------------------------------------

def test_grid_following_battery_pv_and_sofc(client, quiet):
    """
    On the grid, grid-following: a battery at its set 0.2 MW and power factor
    0.95, drawing 0.2 / 0.98 MW from its cells at OCV - (R0 + R1) i; a PV array
    at 0.98 x its maximum power, its DC side at its maximum power point; an SOFC
    at 0.98 x its set power. The PCS's own sgen is not listed with the static
    generators.
    """
    request = _drawn_request()
    lv = _lv(request)
    battery = _der('Battery', 'b1', capacity_kwh=500)
    pv = _der('PV Array', 'pv')
    fc = _der('SOFC', 'fc', p_rated_kw=100, p_set_kw=60)
    result = _post(client, quiet, _with(request, battery, _pcs('p1', lv, 'b1', p_set_mw=0.2, q_mode='pf', pf=0.95),
                                        pv, _pcs('p2', lv, 'pv'), fc, _pcs('p3', lv, 'fc')))
    pcs, ders = _by_id(result['pcs']), _by_id(result['ders'])
    p1 = pcs['cell-p1']
    assert p1['control'] == 'grid_following' and not p1['islanded']
    assert p1['p_mw'] == pytest.approx(0.2) and p1['q_mvar'] == pytest.approx(0.2 * math.tan(math.acos(0.95)))
    assert p1['p_dc_mw'] == pytest.approx(0.2 / ETA) and p1['loss_mw'] == pytest.approx(0.2 / ETA - 0.2)
    b = der.build(battery)
    i = 0.2 / ETA * 1e6 / p1['v_dc_v']
    assert p1['v_dc_v'] == pytest.approx(b.ocv() - 0.0875 * i, rel=1e-9)
    assert ders['cell-b1']['coupling'] == 'pcs' and ders['cell-b1']['p_mw'] == pytest.approx(0.2 / ETA)
    v_mp, _, p_mp = der.build(pv).mpp()
    assert pcs['cell-p2']['p_mw'] == pytest.approx(ETA * p_mp / 1e6, rel=1e-9)
    assert pcs['cell-p2']['v_dc_v'] == pytest.approx(v_mp, rel=1e-9)
    assert pcs['cell-p3']['p_mw'] == pytest.approx(ETA * 0.06, rel=1e-9)
    assert ders['cell-fc']['p_mw'] == pytest.approx(0.06, rel=1e-9)
    assert not {'cell-p1', 'cell-p2', 'cell-p3'} & {g['id'] for g in result.get('staticgenerators', [])}


def test_grid_forming_on_the_grid_holds_its_voltage_by_q_v_droop(client, quiet):
    """On the grid it delivers its set power; its voltage is its set point less 5 % x Q / S_rated."""
    result = _post(client, quiet, _two_buses(_der('Battery', 'bb', capacity_kwh=1000),
                                             _pcs('gb', 'b', 'bb', control='grid_forming', s_rated_mva=1.0,
                                                  p_set_mw=0.3, vm_set_pu=1.01)))
    (gb,) = result['pcs']
    assert gb['p_mw'] == pytest.approx(0.3) and gb['frequency_hz'] == 50 and not gb['islanded']
    assert gb['vm_pu'] == pytest.approx(1.01 - 0.05 * gb['q_mvar'] / 1.0, abs=1e-7)
    assert not result.get('generators')


def test_islanded_grid_forming_pcs_share_by_droop(client, quiet):
    """
    Islanded, two grid-forming PCS - 0.5 and 1.0 MVA, both at 2 % droop and
    0 MW set - share the load and losses in proportion to S_rated / droop,
    1 : 2, at one frequency f_n (1 - droop x P / S_rated); each voltage its
    set point less its Q-V droop. A grid-following PV array on the island
    takes part of the load off them.
    """
    def run(*extra):
        return _post(client, quiet, _two_buses(
            _der('Battery', 'ba', capacity_kwh=1000), _pcs('ga', 'a', 'ba', control='grid_forming', s_rated_mva=0.5),
            _der('Battery', 'bb', capacity_kwh=1000), _pcs('gb', 'b', 'bb', control='grid_forming', s_rated_mva=1.0),
            *extra, grid=False))
    result = run()
    pcs = _by_id(result['pcs'])
    ga, gb = pcs['cell-ga'], pcs['cell-gb']
    assert ga['islanded'] and gb['islanded']
    assert gb['p_mw'] == pytest.approx(2 * ga['p_mw'], rel=1e-6)
    assert ga['p_mw'] + gb['p_mw'] == pytest.approx(0.6, rel=0.01)       # the load and the line's losses
    f = 50 * (1 - 0.02 * gb['p_mw'] / 1.0)
    assert ga['frequency_hz'] == pytest.approx(f, rel=1e-6) and gb['frequency_hz'] == pytest.approx(f, rel=1e-6)
    for row, s in ((ga, 0.5), (gb, 1.0)):
        assert row['vm_pu'] == pytest.approx(1.0 - 0.05 * row['q_mvar'] / s, abs=1e-7)
    with_pv = _by_id(run(_der('PV Array', 'pv'), _pcs('pp', 'b', 'pv'))['pcs'])
    p_pv = with_pv['cell-pp']['p_mw']
    assert with_pv['cell-ga']['p_mw'] + with_pv['cell-gb']['p_mw'] == pytest.approx(ga['p_mw'] + gb['p_mw'] - p_pv, rel=0.01)
    assert with_pv['cell-gb']['p_mw'] == pytest.approx(2 * with_pv['cell-ga']['p_mw'], rel=1e-6)


def test_source_on_a_dc_bus_behind_the_pcs(client, quiet):
    """
    The battery alone on a DC bus the PCS's DC side reaches is the same as one
    wired straight to it, and that DC bus is the PCS's own, not reported. A
    DC bus with a network on it is a VSC's: the PCS is left out.
    """
    request = _drawn_request()
    lv = _lv(request)
    straight = _post(client, quiet, _with(_drawn_request(), _der('Battery', 'b1'), _pcs('p1', lv, 'b1', p_set_mw=0.1)))
    on_bus = _post(client, quiet, _with(request, _bus('link', 0.8, dc=True), _der('Battery', 'b1', 'link'),
                                        _pcs('p1', lv, bus_dc='link', p_set_mw=0.1)))
    for key in ('p_mw', 'q_mvar', 'p_dc_mw', 'v_dc_v'):
        assert on_bus['pcs'][0][key] == pytest.approx(straight['pcs'][0][key], rel=1e-9), key
    assert 'cell-link' not in _by_id(on_bus['dcbuses'])
    assert not any('LINK' in w for w in on_bus.get('warnings', []))
    network = _post(client, quiet, _with(_drawn_request(), _pcs('p9', lv, bus_dc='dc_b')))
    assert not network.get('pcs')
    assert any("PCS 'P9' is left out: its DC side must be one source or store" in w for w in network['warnings'])


def test_rating_and_capability(client, quiet):
    """0.8 MW asked of a 0.5 MVA PCS: it delivers 0.5 MW, and Q held to what is left of its circle."""
    request = _drawn_request()
    lv = _lv(request)
    result = _post(client, quiet, _with(request, _der('Battery', 'b1', capacity_kwh=1000),
                                        _pcs('p1', lv, 'b1', p_set_mw=0.8, q_set_mvar=0.2),
                                        _der('Battery', 'b2', capacity_kwh=1000),
                                        _pcs('p2', lv, 'b2', p_set_mw=0.4, q_set_mvar=0.4)))
    pcs = _by_id(result['pcs'])
    assert pcs['cell-p1']['p_mw'] == pytest.approx(0.5) and pcs['cell-p1']['q_mvar'] == pytest.approx(0, abs=1e-9)
    assert pcs['cell-p2']['q_mvar'] == pytest.approx(0.3) and pcs['cell-p2']['loading_percent'] == pytest.approx(100)
    w = ' '.join(result['warnings'])
    assert "PCS 'P1': 0.8 MW is more than its rating" in w and "PCS 'P2': its Q (0.4 Mvar) is held to 0.3" in w


def test_left_out_with_a_reason(client, quiet):
    request = _drawn_request()
    lv = _lv(request)
    result = _post(client, quiet, _with(request, _der('Supercapacitor', 'sc'), _pcs('p1', lv, 'sc'), _pcs('p2', lv)))
    assert not result.get('pcs')
    w = ' '.join(result['warnings'])
    assert "PCS 'P1' is left out: a supercapacitor connects to a DC bus" in w
    assert "PCS 'P2' has no battery, flywheel, SOFC system or PV array on its DC side" in w


# --- Short circuit -----------------------------------------------------------------------------

K, S_RATED = 1.2, 0.5
I_LIMIT = K * S_RATED / (math.sqrt(3) * 0.4)        # kA


def _sc(client, quiet, params, *elements, grid=True):
    rows = _post(client, quiet, _two_buses(*elements, grid=grid, params={**params, 'user_email': 't@t'}))['busbars']
    return {r['name']: r for r in rows}


@pytest.mark.parametrize('control', ['grid_following', 'grid_forming'])
def test_iec_short_circuit_adds_its_current_limit(client, quiet, control):
    """IEC 60909: a current source at k x I_rated, grid-forming or grid-following: + 0.866 kA at each bus."""
    base = _sc(client, quiet, SC)
    with_pcs = _sc(client, quiet, SC, _der('Battery', 'b1'),
                   _pcs('p1', 'b', 'b1', control=control, s_rated_mva=S_RATED, current_limit_pu=K))
    for bus in ('a', 'b'):
        assert with_pcs[bus]['ikss_ka'] - base[bus]['ikss_ka'] == pytest.approx(I_LIMIT, rel=1e-6), bus


def test_ansi_short_circuit_counts_it_as_its_current_limit(client, quiet):
    """
    ANSI: a shunt drawing its current limit at rated voltage (R/X 0.1) in the
    first-cycle and interrupting networks, none in the 30-cycle one:
    I = (V / sqrt 3) |1 / Z_th + 1 / z|, z = V / (sqrt 3 x I_limit).
    """
    base = _sc(client, quiet, ANSI_PARAMS)
    with_pcs = _sc(client, quiet, ANSI_PARAMS, _der('Battery', 'b1'),
                   _pcs('p1', 'b', 'b1', s_rated_mva=S_RATED, current_limit_pu=K))
    b0, b1 = base['b'], with_pcs['b']
    z_th = complex(b0['rk_ohm'], b0['xk_ohm'])
    z_mag = 0.4 / (math.sqrt(3) * I_LIMIT)
    x = z_mag / math.sqrt(1.01)
    z_cs = complex(0.1 * x, x)
    want = 0.4 / math.sqrt(3) * abs(1 / z_th + 1 / z_cs)
    assert b1['i_first_sym_ka'] == pytest.approx(want, rel=1e-6)
    assert b1['i_interrupting_ka'] == pytest.approx(want, rel=1e-6)
    assert b1['i_steady_ka'] == pytest.approx(b0['i_steady_ka'], rel=1e-9)


def test_islanded_short_circuit(client, quiet):
    """
    An island two grid-forming PCS feed: the larger is a source giving its
    current limit at its own bus, the other a current source - at the larger's
    bus, the two limits added.
    """
    rows = _sc(client, quiet, SC, _der('Battery', 'ba'), _pcs('ga', 'a', 'ba', control='grid_forming', s_rated_mva=0.5),
               _der('Battery', 'bb'), _pcs('gb', 'b', 'bb', control='grid_forming', s_rated_mva=1.0), grid=False)
    assert rows['b']['ikss_ka'] == pytest.approx(1.2 * 1.5 / (math.sqrt(3) * 0.4), rel=1e-6)
    assert 0 < rows['a']['ikss_ka'] < rows['b']['ikss_ka']


def test_iec_short_circuit_counts_storage(client, quiet):
    """A Storage element with a maximum short-circuit current: in the IEC study as a current source giving it."""
    storage = {'typ': 'Storage0', 'name': 'st', 'id': 'cell-st', 'userFriendlyName': 'ST', 'bus': 'b', 'p_mw': '0',
               'max_e_mwh': '1', 'q_mvar': '0', 'sn_mva': '0.5', 'soc_percent': '50', 'min_e_mwh': '0', 'scaling': '1',
               'type': '', 'in_service': 'true', 'max_ik_ka': '0.6', 'current_source': 'true'}
    base = _sc(client, quiet, SC)
    with_st = _sc(client, quiet, SC, storage)
    assert with_st['b']['ikss_ka'] - base['b']['ikss_ka'] == pytest.approx(0.6, rel=1e-6)


# --- The other studies -------------------------------------------------------------------------

@pytest.mark.parametrize('fixture, params', [
    ('reference_radial.diagram_sc_payload.json', None),
    ('reference_radial.diagram_sc_payload.json', ANSI_PARAMS),
    ('reference_radial.diagram_timeseries_payload.json', {'time_steps': 2}),
    ('reference_radial.diagram_contingency_payload.json', None),
    ('reference_radial.diagram_opf_payload.json', None),
    ('reference_radial.diagram_sc_payload.json', {'typ': 'StateEstimationPandaPower Parameters'}),
    ('reference_radial.diagram_sc_payload.json', {'typ': 'ArcFlashPandaPower Parameters'}),
    ('reference_radial.diagram_sc_payload.json', {'typ': 'EmtStudy Parameters', 'time_step_us': '20',
                                                  'duration_ms': '10'}),
    ('reference_radial.diagram_sc_payload.json', {'typ': 'TransientStabilityAndes Parameters', 'frequency': '50',
                                                  'sn_mva': '100', 'tf': '1', 'fault_enabled': False}),
])
def test_other_studies_run_with_a_pcs(client, quiet, fixture, params):
    """None may fail because of a PCS, grid-forming or grid-following."""
    request = _drawn_request(fixture)
    lv = _lv(request)
    request = _with(request, _der('Battery', 'b1'), _pcs('p1', lv, 'b1', p_set_mw=0.1),
                    _der('PV Array', 'pv'), _pcs('p2', lv, 'pv'),
                    _der('Battery', 'b2'), _pcs('p3', lv, 'b2', control='grid_forming', p_set_mw=0.05))
    if params:
        key = next(k for k, v in request.items() if 'Parameters' in str(v.get('typ', '')))
        request[key] = {**request[key], **params} if 'typ' not in params else {**params, 'user_email': 't@t'}
    with quiet():
        response = client.post('/', json=request)
    assert response.status_code == 200
    result = json.loads(response.get_data(as_text=True))
    if isinstance(result, dict):
        assert not result.get('error'), result.get('message') or result.get('exception')


# --- The BESS builder's pair ---------------------------------------------------------------------

def _as_pair(request):
    """Each PCS_n Storage element drawn as the builder's pair: a PCS and a Battery wired to it."""
    for key in [k for k, v in request.items() if isinstance(v, dict)
                and str(v.get('typ', '')).startswith('Storage') and str(v.get('userFriendlyName', '')).startswith('PCS_')]:
        st = request[key]
        n = st['userFriendlyName'].split('_')[-1]
        e_mwh, p_max = float(st['max_e_mwh']), float(st['max_p_mw'])
        request[key] = _pcs(f'pcs{n}', st['bus'], f'bat{n}', s_rated_mva=st['sn_mva'], p_set_mw=0,
                            userFriendlyName=st['userFriendlyName'], id=st['id'])
        request[key]['name'] = st['name']
        request[f'bat{n}'] = _der('Battery', f'bat{n}', vn_v=1500, capacity_kwh=1000 * e_mwh, soc_percent=50,
                                  c_rate_discharge=p_max / e_mwh, c_rate_charge=p_max / e_mwh)
        request[key]['der'] = f'bat{n}'
    return request


def test_bess_study_runs_on_the_pcs_and_battery_pair(client, quiet):
    """
    The BESS preliminary design study on the builder's pair: each PCS and its
    battery are the Storage element it dispatches - the same cases, passed and
    failed alike, with the same loadings.
    """
    import os
    from test_bess_sizing_headroom import REFERENCE_DIR, WIZARD, _suggested_ratings, _with_ratings

    def study(pair):
        with open(os.path.join(REFERENCE_DIR, 'reference_radial.diagram_bess_preliminary_payload.json'),
                  encoding='utf-8') as handle:
            request = _with_ratings(json.load(handle), _suggested_ratings(WIZARD['reference_radial']))
        if pair:
            request = _as_pair(request)
        result = _post(client, quiet, request)
        return result['bess_preliminary_results']

    storage, pair = study(False), study(True)
    assert pair['summary'] == storage['summary']
    for a, b in zip(storage['named_cases'], pair['named_cases']):
        rows_a = {r['name']: r for r in a.get('elements') or [] if str(r.get('name', '')).startswith('PCS_')}
        rows_b = {r['name']: r for r in b.get('elements') or [] if str(r.get('name', '')).startswith('PCS_')}
        assert rows_a.keys() == rows_b.keys() and rows_a, a.get('name')
        for name in rows_a:
            assert rows_b[name]['loading_percent'] == pytest.approx(rows_a[name]['loading_percent'], rel=1e-6)
