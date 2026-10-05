"""
The solid-state transformer: MV AC to an internal DC link (rectifier), to the
LV DC port (DC/DC stage), and optionally to the LV AC port (inverter), each
stage drawing its output / efficiency + no-load loss - the design note's
prototype case, stage by stage; grid-following and grid-forming inverters;
no LV AC port; and every study with an SST present.
"""
import json

import pytest

from test_dc_elements import _drawn_request, _with

MV_BUS = 'B2'                  # 20 kV
LV_NETWORK_A = 'LV network A'  # 0.4 kV, fed from the grid through transformer TA
ETA_R, ETA_D, ETA_I = 0.985, 0.98, 0.975

LV_DC = {'typ': 'DC Bus5', 'name': 'sst_dc', 'id': 'cell-sst_dc', 'userFriendlyName': 'SST LV DC', 'vn_kv': '0.8'}
LV_DC_LOAD = {'typ': 'Load DC5', 'name': 'sst_dc_load', 'id': 'cell-sst_dc_load', 'userFriendlyName': 'DC racks',
              'bus': 'sst_dc', 'p_mw': '1.0'}
LV_AC = {'typ': 'Bus9', 'name': 'sst_ac', 'id': 'cell-sst_ac', 'userFriendlyName': 'SST LV AC', 'vn_kv': '0.4'}
LV_AC_LOAD = {'typ': 'Load9', 'name': 'sst_ac_load', 'id': 'cell-sst_ac_load', 'userFriendlyName': 'AC hall',
              'bus': 'sst_ac', 'p_mw': '0.5', 'q_mvar': '0', 'const_z_percent': '0', 'const_i_percent': '0',
              'sn_mva': '0', 'scaling': '1', 'type': 'wye', 'in_service': 'true'}


def _bus(request, label):
    """An AC bus's cell name, by its label on the diagram (each fixture names its cells afresh)."""
    return next(v['name'] for v in request.values() if isinstance(v, dict)
                and str(v.get('typ', '')).startswith('Bus') and v.get('userFriendlyName') == label)


def _with_sst(request, *elements, **sst_fields):
    """The request with these elements and an SST, its ports named by label."""
    lv_ac = sst_fields.pop('lv_ac', None)
    if lv_ac and lv_ac != 'sst_ac':
        lv_ac = _bus(request, lv_ac)
    return _with(request, *elements, _sst(lv_ac, bus_mv=_bus(request, MV_BUS), **sst_fields))


def _sst(bus_lv_ac=None, **fields):
    row = {'typ': 'Solid-State Transformer0', 'name': 'sst', 'id': 'cell-sst', 'userFriendlyName': 'SST 1',
           'bus_mv': MV_BUS, 'bus_lv_dc': 'sst_dc', 'link_kv': '30', 'q_mv_mvar': '0',
           'rect_efficiency_percent': str(100 * ETA_R), 'rect_no_load_kw': '0', 'rect_rated_mw': '2',
           'dcdc_efficiency_percent': str(100 * ETA_D), 'dcdc_no_load_kw': '0', 'dcdc_rated_mw': '2',
           'inv_efficiency_percent': str(100 * ETA_I), 'inv_no_load_kw': '0', 'inv_rated_mw': '1',
           'vm_lv_dc_pu': '1.0', 'vm_lv_ac_pu': '1.0', 'inverter_mode': 'grid_following', 'p_ac_mw': '0', 'q_ac_mvar': '0'}
    if bus_lv_ac:
        row['bus_lv_ac'] = bus_lv_ac
    row.update({k: str(v) for k, v in fields.items()})
    return row


def _run(client, quiet, request):
    with quiet():
        response = client.post('/', json=request)
    result = json.loads(response.get_data(as_text=True))
    assert not result.get('error'), result.get('message')
    return result


def _stages(sst):
    return {st['stage']: st for st in sst['stages']}


def test_prototype_case_stage_by_stage(client, quiet):
    """
    The design note's case: a 1.0 MW LV DC load and a 0.5 MW LV AC load on an
    island the grid-forming inverter holds, 98 % DC/DC and 97.5 % inverter.
    """
    request = _with_sst(_drawn_request(), LV_DC, LV_DC_LOAD, LV_AC, LV_AC_LOAD,
                        lv_ac='sst_ac', inverter_mode='grid_forming', link_kv=32)
    result = _run(client, quiet, request)
    (sst,) = result['ssts']
    st = _stages(sst)
    p_inv_in = 0.5 / ETA_I
    p_dcdc_out = 1.0 + p_inv_in
    p_dcdc_in = p_dcdc_out / ETA_D
    assert st['inverter']['p_out_mw'] == pytest.approx(0.5, abs=1e-6)
    assert st['inverter']['p_in_mw'] == pytest.approx(p_inv_in, abs=1e-6)          # 0.5128
    assert st['dcdc']['p_out_mw'] == pytest.approx(p_dcdc_out, abs=1e-6)           # 1.5128
    assert st['dcdc']['p_in_mw'] == pytest.approx(p_dcdc_in, abs=1e-6)             # 1.5437
    assert st['rectifier']['p_out_mw'] == pytest.approx(p_dcdc_in, abs=1e-6)
    assert sst['p_mv_mw'] == pytest.approx(p_dcdc_in / ETA_R, abs=1e-6)
    assert sst['vm_lv_dc_pu'] == pytest.approx(1.0) and sst['vm_lv_ac_pu'] == pytest.approx(1.0)
    assert sst['link_kv'] == 32 and sst['inverter_mode'] == 'grid_forming'
    assert st['dcdc']['loading_percent'] == pytest.approx(100 * p_dcdc_in / 2)
    # Its own elements are not in the results; the island's load is.
    assert {b['id'] for b in result['dcbuses']} >= {'cell-sst_dc'}
    assert not any('DC link' in str(b.get('name')) for b in result['dcbuses'])
    assert 'cell-sst' not in {l['id'] for l in result['loads']}
    assert len(result['externalgrids']) == 1
    assert not any('grid-forming inverter holds' in w for w in result.get('warnings', []))


def test_grid_following_inverter_and_mv_reactive_power(client, quiet):
    """
    Injecting 0.2 MW and 0.05 Mvar into LV network A, which the grid holds;
    0.3 Mvar drawn at the MV bus. The rectifier supplies the DC racks and the
    inverter, through the DC/DC stage.
    """
    request = _with_sst(_drawn_request(), LV_DC, LV_DC_LOAD,
                        lv_ac=LV_NETWORK_A, p_ac_mw=0.2, q_ac_mvar=0.05, q_mv_mvar=0.3, inv_no_load_kw=1)
    result = _run(client, quiet, request)
    (sst,) = result['ssts']
    st = _stages(sst)
    p_inv_in = 0.2 / ETA_I + 0.001
    assert sst['inverter_mode'] == 'grid_following'
    assert sst['p_lv_ac_mw'] == pytest.approx(0.2) and sst['q_lv_ac_mvar'] == pytest.approx(0.05)
    assert st['inverter']['p_in_mw'] == pytest.approx(p_inv_in, abs=1e-6)
    assert sst['p_mv_mw'] == pytest.approx((1.0 + p_inv_in) / ETA_D / ETA_R, abs=1e-6)
    assert sst['q_mv_mvar'] == pytest.approx(0.3)
    assert not any(s.get('name', '').endswith('inverter') for s in result.get('sgens', []) or [])


def test_without_an_lv_ac_port(client, quiet):
    """Two stages only: the rectifier and the DC/DC stage."""
    result = _run(client, quiet, _with_sst(_drawn_request(), LV_DC, LV_DC_LOAD, dcdc_no_load_kw=5))
    (sst,) = result['ssts']
    assert [s['stage'] for s in sst['stages']] == ['rectifier', 'dcdc']
    assert sst['inverter_mode'] is None and sst['vm_lv_ac_pu'] is None
    assert sst['p_mv_mw'] == pytest.approx((1.0 / ETA_D + 0.005) / ETA_R, abs=1e-6)


def test_grid_forming_inverter_on_a_grid_connected_network_is_warned(client, quiet):
    result = _run(client, quiet, _with_sst(_drawn_request(), LV_DC, LV_DC_LOAD,
                                           lv_ac=LV_NETWORK_A, inverter_mode='grid_forming'))
    assert any("Solid-State Transformer 'SST 1': its grid-forming inverter holds an LV AC network another source"
               in w for w in result['warnings'])


@pytest.mark.parametrize('fixture, params', [
    ('reference_radial.diagram_sc_payload.json', None),
    ('reference_radial.diagram_timeseries_payload.json', {'time_steps': 2}),
    ('reference_radial.diagram_contingency_payload.json', None),
    ('reference_radial.diagram_harmonic_payload.json', None),
    ('reference_radial.diagram_opf_payload.json', None),
    ('reference_radial.diagram_sc_payload.json', {'typ': 'StateEstimationPandaPower Parameters'}),
    ('reference_radial.diagram_sc_payload.json', {'typ': 'ArcFlashPandaPower Parameters'}),
    ('reference_radial.diagram_sc_payload.json', {'typ': 'DcFaultStudy Parameters', 'duration_ms': '5',
                                                  'fault_bus': 'sst_dc'}),
    ('reference_radial.diagram_sc_payload.json', {'typ': 'TransientStabilityAndes Parameters', 'frequency': '50',
                                                  'sn_mva': '100', 'tf': '1', 'fault_enabled': False}),
])
def test_other_studies_run_with_an_sst(client, quiet, fixture, params):
    """None may fail because of an SST, nor show its auxiliary elements."""
    request = _with_sst(_drawn_request(fixture), LV_DC, LV_DC_LOAD, LV_AC, LV_AC_LOAD,
                        lv_ac='sst_ac', inverter_mode='grid_forming')
    if params:
        key = next(k for k, v in request.items() if 'Parameters' in str(v.get('typ', '')))
        request[key] = {**request[key], **params} if 'typ' not in params else {**params, 'user_email': 't@t'}
    with quiet():
        response = client.post('/', json=request)
    assert response.status_code == 200
    text = response.get_data(as_text=True)
    result = json.loads(text)
    if isinstance(result, dict):
        assert not result.get('error'), result.get('message') or result.get('exception')
        assert not any("Solid-State Transformer 'SST 1' needs" in w for w in result.get('warnings', []))
    assert 'auxiliary' not in text and 'DC link' not in text


def test_an_sst_alone_on_a_grid_without_other_dc_elements(client, quiet):
    """
    Every VSC, DC load and grid-forming source the SST builds is its own: the
    results list none of them, and still give the drawn grid, DC bus and loads.
    """
    import os
    from test_dc_elements import REFERENCE_DIR
    with open(os.path.join(REFERENCE_DIR, 'reference_radial.diagram_payload.json'), encoding='utf-8') as handle:
        plain = json.load(handle)
    result = _run(client, quiet, _with_sst(plain, LV_DC, LV_DC_LOAD, LV_AC, LV_AC_LOAD,
                                           lv_ac='sst_ac', inverter_mode='grid_forming'))
    assert result.get('vscs', []) == []
    assert [b['id'] for b in result['dcbuses']] == ['cell-sst_dc']
    assert [l['id'] for l in result['loadsdc']] == ['cell-sst_dc_load']
    assert len(result['externalgrids']) == 1
    assert result['ssts'][0]['p_mv_mw'] == pytest.approx((1.0 + 0.5 / ETA_I) / ETA_D / ETA_R, abs=1e-6)
