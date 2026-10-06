"""
The spec's DC and microgrid layer (electrisim_spec_layer): DC buses, lines,
loads, sources and capacitors, VSCs, SSTs, DC/DC converters, DC breakers, the
five sources and stores, the PCS, grounding transformers and load profiles.
Checked against the same network posted as the diagram's rows, refused with a
reason when wrong, handed to the canvas by /build-model, and documented for
the MCP server.
"""
import json
import os
import re

import pytest

import electrisim_sld
import electrisim_spec_layer as layer
from test_pcs import LF, _der, _pcs, _post, _two_buses

HERE = os.path.dirname(os.path.abspath(__file__))
JS = os.path.join(HERE, '..', '..', 'frontend', 'src', 'main', 'webapp', 'js', 'electrisim')


def spec(**extra):
    """0.4 kV buses A and B, 100 m apart, a grid on A and a 0.6 MW load on B, and the layer."""
    base = {
        'name': 'layer',
        'buses': [{'id': 'A', 'vn_kv': 0.4}, {'id': 'B', 'vn_kv': 0.4}],
        'external_grids': [{'id': 'G', 'bus': 'A', 's_sc_max_mva': 10, 'rx_max': 0.1}],
        'lines': [{'id': 'L', 'from_bus': 'A', 'to_bus': 'B', 'length_km': 0.1, 'r_ohm_per_km': 0.1,
                   'x_ohm_per_km': 0.08, 'max_i_ka': 2}],
        'loads': [{'id': 'LD', 'bus': 'B', 'p_mw': 0.6, 'q_mvar': 0.1}],
    }
    base.update(extra)
    return base


MICROGRID = dict(
    batteries=[{'id': 'B1', 'capacity_kwh': 500}],
    pv_arrays=[{'id': 'PV'}],
    pcs=[{'id': 'P1', 'bus': 'B', 'source': 'B1', 'p_set_mw': 0.2, 's_rated_mva': 0.5, 'efficiency_percent': 98},
         {'id': 'P2', 'bus': 'B', 'source': 'PV', 's_rated_mva': 0.5, 'efficiency_percent': 98}],
    dc_buses=[{'id': 'D1', 'vn_kv': 0.8}, {'id': 'D2', 'vn_kv': 0.8}],
    vscs=[{'id': 'V1', 'bus': 'B', 'bus_dc': 'D1', 'control_mode_ac': 'q_mvar', 'control_value_ac': 0,
           'control_mode_dc': 'vm_pu', 'control_value_dc': 1.0}],
    dc_loads=[{'id': 'R1', 'bus': 'D2', 'p_mw': 0.1}],
    dc_dc_converters=[{'id': 'C1', 'bus_in': 'D1', 'bus_out': 'D2', 'control_mode': 'voltage', 'vm_out_pu': 1.0,
                       'rated_mw': 0.3}],
    supercapacitors=[{'id': 'SC', 'bus': 'D2'}],
    grounding_transformers=[{'id': 'ZA', 'bus': 'A'}],
)


def _js_ids(path, start, stop):
    s = open(os.path.join(JS, path), encoding='utf-8').read()
    return s[s.index(start):s.index(stop, s.index(start))]


def test_fields_follow_the_dialogs():
    """Each kind's fields are its dialog's (derParameters.js) and its payload's (dcPayload.js)."""
    src = open(os.path.join(JS, 'utils', 'derParameters.js'), encoding='utf-8').read()
    body = src[src.index('export const DER_PARAMETERS = {'):]
    for kind in ('Battery', 'Supercapacitor', 'Flywheel', 'SOFC', 'PV Array'):
        key = kind if ' ' not in kind else f"'{kind}'"
        block = body[body.index(f'\n    {key}: ['):]
        block = block[:block.index('\n    ]')]
        ids = [a or b for a, b in re.findall(r"(?:num|sel)\('(\w+)'|\bid: '(\w+)'", block)]
        assert tuple(i for i in ids if i not in ('name', 'in_service')) == layer.DER_FIELDS[kind], kind
    for kind, start in (('PCS', 'DER_PARAMETERS.PCS = ['), ('Grounding Transformer', "DER_PARAMETERS['Grounding Transformer'] = [")):
        block = src[src.index(start):src.index('\n];', src.index(start))]
        assert tuple(re.findall(r"(?:num|sel)\('(\w+)'", block)) == layer.DER_FIELDS[kind], kind
    dc = open(os.path.join(JS, 'utils', 'dcPayload.js'), encoding='utf-8').read()
    for kind in ('DC Breaker', 'Solid-State Transformer', 'DC/DC Converter', 'DC Line'):
        case = dc[dc.index(f"case '{kind}'"):]
        listed = re.findall(r"'(\w+)'", case[case.index('withOptional'):case.index(']', case.index('withOptional'))])
        listed = [f for f in listed if f not in ('in_service', 'cost_per_unit_by_currency')]
        assert set(layer.DC_FIELDS[kind]) >= set(listed) - {'p_mw', 'loss_percent', 'loss_mw', 'vm_from_pu', 'vm_to_pu',
                                                             'max_p_mw', 'min_q_from_mvar', 'max_q_from_mvar',
                                                             'min_q_to_mvar', 'max_q_to_mvar',
                                                             'opf_marginal_cost_eur_per_mwh', 'opf_cp2_eur_per_mw2',
                                                             'opf_cost_currency'}, kind


def test_defaults_follow_the_dialogs():
    """A field the spec leaves out takes its dialog's default, as a dropped element does."""
    src = open(os.path.join(JS, 'utils', 'derParameters.js'), encoding='utf-8').read()
    num = re.compile(r"num\('(\w+)',\s*'[^']*',\s*'[^']*',\s*('[^']*'|[0-9.e+-]+)")
    sel = re.compile(r"sel\('(\w+)',\s*'[^']*',\s*'([^']*)'")
    for kind in layer.DER_FIELDS:
        if kind in ('PCS', 'Grounding Transformer'):
            start = 'DER_PARAMETERS.PCS = [' if kind == 'PCS' else "DER_PARAMETERS['Grounding Transformer'] = ["
            block = src[src.index(start):src.index('\n];', src.index(start))]
        else:
            body = src[src.index('export const DER_PARAMETERS = {'):]
            key = kind if ' ' not in kind else f"'{kind}'"
            block = body[body.index(f'\n    {key}: ['):]
            block = block[:block.index('\n    ]')]
        js = {m.group(1): m.group(2).strip("'") for m in num.finditer(block)}
        js.update({m.group(1): m.group(2) for m in sel.finditer(block)})
        for field, value in js.items():
            assert layer.DEFAULTS[kind][field] == value, (kind, field)
    cfg = open(os.path.join(JS, 'configureAttributes.js'), encoding='utf-8').read()
    fns = {'Load DC': 'configureLoadDcAttributes', 'Source DC': 'configureSourceDcAttributes',
           'DC Capacitor': 'configureDcCapacitorAttributes', 'DC Breaker': 'configureDcBreakerAttributes',
           'DC/DC Converter': 'configureDcDcConverterAttributes', 'VSC': 'configureVscAttributes',
           'DC Line': 'configureDCLineAttributes', 'Solid-State Transformer': 'configureSstAttributes'}
    for kind, fn in fns.items():
        i = cfg.index(f'export function {fn}')
        body = cfg[i:cfg.index('\nexport function', i + 10)]
        js = {m.group(1): m.group(2) for m in re.finditer(
            r'setAttribute\("(\w+)",\s*(?:String\()?options\.\w+\s*(?:\?\?|\|\|)\s*"([^"]*)"', body)}
        js.update({m.group(1): m.group(2) for m in re.finditer(r'(\w+): "([^"]*)"', body)})
        for field in layer.FIELDS[kind]:
            if field in js and field != 'closed':
                assert layer.DEFAULTS[kind][field] == js[field], (kind, field)


def test_spec_solves_as_the_diagram_does(client, quiet):
    """
    The layer built from the spec gives the load flow the same network,
    posted as the diagram's rows, gives: each PCS, source and store, DC bus,
    converter and grounding transformer.
    """
    net, _ = electrisim_sld.build_network(spec(**MICROGRID))
    res = electrisim_sld.solve(net)
    assert res['converged']

    drawn = _two_buses(
        _der('Battery', 'b1', capacity_kwh=500), _pcs('p1', 'b', 'b1', p_set_mw=0.2),
        _der('PV Array', 'pv'), _pcs('p2', 'b', 'pv'),
        {'typ': 'DC Bus0', 'name': 'd1', 'id': 'cell-d1', 'userFriendlyName': 'D1', 'vn_kv': '0.8'},
        {'typ': 'DC Bus1', 'name': 'd2', 'id': 'cell-d2', 'userFriendlyName': 'D2', 'vn_kv': '0.8'},
        {'typ': 'VSC0', 'name': 'v1', 'id': 'cell-v1', 'userFriendlyName': 'V1', 'bus': 'b', 'bus_dc': 'd1',
         'r_ohm': '0.01', 'x_ohm': '0.1', 'r_dc_ohm': '0.01', 'control_mode_ac': 'q_mvar', 'control_value_ac': '0',
         'control_mode_dc': 'vm_pu', 'control_value_dc': '1.0'},
        {'typ': 'Load DC0', 'name': 'r1', 'id': 'cell-r1', 'userFriendlyName': 'R1', 'bus': 'd2', 'p_mw': '0.1'},
        {'typ': 'DC/DC Converter0', 'name': 'c1', 'id': 'cell-c1', 'userFriendlyName': 'C1', 'bus_in': 'd1',
         'bus_out': 'd2', 'control_mode': 'voltage', 'vm_out_pu': '1.0', 'rated_mw': '0.3', 'no_load_loss_kw': '1'},
        _der('Supercapacitor', 'sc', bus='d2'),
        {'typ': 'Grounding Transformer0', 'name': 'za', 'id': 'cell-za', 'userFriendlyName': 'ZA', 'bus': 'a'})
    # The spec's grid is 10 MVA at R/X 0.1 with no other data; the drawn one carries the same.
    flow = _post(client, quiet, drawn)
    by = lambda rows, key='id': {r[key]: r for r in rows}
    assert {b['id']: b['vm_pu'] for b in res['buses']}['B'] == pytest.approx(
        by(flow['busbars'], 'name')['b']['vm_pu'], abs=1e-4)
    pcs, drawn_pcs = by(res['pcs']), by(flow['pcs'])
    for s_id, d_id in (('P1', 'cell-p1'), ('P2', 'cell-p2')):
        assert pcs[s_id]['p_mw'] == pytest.approx(drawn_pcs[d_id]['p_mw'], abs=1e-6)
        assert pcs[s_id]['q_mvar'] == pytest.approx(drawn_pcs[d_id]['q_mvar'], abs=1e-6)
    ders, drawn_ders = by(res['sources_and_stores']), by(flow['ders'])
    for s_id, d_id in (('B1', 'cell-b1'), ('PV', 'cell-pv'), ('SC', 'cell-sc')):
        assert ders[s_id]['p_mw'] == pytest.approx(drawn_ders[d_id]['p_mw'], abs=1e-6), s_id
    conv = res['dc_dc_converters'][0]
    assert conv['p_out_mw'] == pytest.approx(0.1) and conv['p_in_mw'] == pytest.approx(
        flow['dcdcconverters'][0]['p_in_mw'], abs=1e-6)
    assert [b['vm_pu'] for b in res['dc_buses']] == pytest.approx([1.0, 1.0], abs=1e-6)
    assert res['grounding_transformers'][0]['i_ground_alone_a'] == pytest.approx(
        flow['groundingtransformers'][0]['i_ground_alone_a'])
    assert res['vscs'][0]['id'] == 'V1'
    # Only the spec's own elements in the AC results: the converters' internal parts stay out.
    assert [b['id'] for b in res['buses']] == ['A', 'B']


@pytest.mark.parametrize('extra, problem', [
    ({'batteries': [{'id': 'B1', 'capacity_kwhh': 500}]}, 'unknown field(s) capacity_kwhh'),
    ({'pcs': [{'id': 'P1', 'bus': 'B', 'source': 'LD'}]}, "source='LD' must be a battery"),
    ({'dc_buses': [{'id': 'D1', 'vn_kv': 0.8}], 'dc_loads': [{'id': 'R1', 'bus': 'B', 'p_mw': 0.1}]},
     "bus='B' must be a DC bus id"),
    ({'pcs': [{'id': 'P1', 'bus': 'B'}]}, 'a PCS needs its source'),
    ({'dc_buses': [{'id': 'A', 'vn_kv': 0.8}]}, "id 'A' is used twice"),
    ({'dc_buses': [{'id': 'D1'}]}, 'vn_kv is required'),
    ({'load_profiles': [{'id': 'sun', 'kind': 'irradiance', 'dt_s': 3600, 'values': [0, 500, 1000]}],
      'dc_buses': [{'id': 'D1', 'vn_kv': 0.8}],
      'dc_loads': [{'id': 'R1', 'bus': 'D1', 'p_mw': 0.1, 'load_profile_id': 'sun'}]},
     "is a irradiance profile; it must be a power profile"),
    ({'switches': [{'id': 'S', 'bus': 'A', 'element': 'L', 'et': 'grounding_transformer'}]},
     "element='L' is not a grounding_transformers id"),
])
def test_problems_are_named(extra, problem):
    with pytest.raises(electrisim_sld.SpecError) as err:
        electrisim_sld.build_network(spec(**extra))
    assert any(problem in p for p in err.value.problems), err.value.problems


def test_build_model_hands_the_layer_to_the_canvas(client):
    """/build-model's model carries electrisim_elements: each element, its fields, its connections."""
    body = {**spec(**MICROGRID, switches=[{'id': 'CB', 'bus': 'A', 'element': 'ZA', 'et': 'grounding_transformer'}],
                   load_profiles=[{'id': 'day', 'name': 'Day', 'dt_s': 3600, 'values': [0.6, 1.0, 0.8]}])}
    response = client.post('/build-model', json={'spec': body, 'run_power_flow': True})
    assert response.status_code == 200, response.get_data(as_text=True)
    out = response.get_json()
    model = json.loads(out['model'])
    drawn = json.loads(model['_object']['electrisim_elements']['_object'])
    elements = {e['id']: e for e in drawn['elements']}
    assert elements['P1']['connections'] == {'bus': {'id': 'B', 'ac_bus_name': 'B'}, 'source': {'id': 'B1'}}
    assert elements['C1']['connections'] == {'bus_in': {'id': 'D1'}, 'bus_out': {'id': 'D2'}}
    assert elements['CB']['kind'] == 'Switch' and elements['CB']['connections']['element'] == {'id': 'ZA'}
    assert elements['B1']['attributes'] == {'capacity_kwh': '500'}
    assert drawn['load_profiles']['day'] == {'name': 'Day', 'kind': 'power', 'dt': 3600.0, 't0': 0.0, 'p': [0.6, 1.0, 0.8]}
    # The pandapower model is the AC network alone.
    assert len(json.loads(model['_object']['bus']['_object'])['data']) == 2
    assert out['power_flow']['converged'] and out['power_flow']['pcs']


def test_spec_format_documents_every_list():
    doc = open(os.path.join(HERE, '..', 'electrisim_mcp', 'spec_format.md'), encoding='utf-8').read()
    for key in layer.TOP_LEVEL:
        assert f'**{key}**' in doc, key
