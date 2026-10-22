"""
The Electrisim <-> mgtwin Site interchange (mgtwin_site_io): a Site imported as a spec, a drawn diagram exported as
a Site, and a Site's operating state read onto the diagram - each checked round trip on the C003 references.
"""
import contextlib
import io
import json
import math
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, 'reference')
SPECS = REF       # the corrected Sites, copied from the root repo's specs/
sys.path.insert(0, os.path.join(HERE, '..'))

from mgtwin_site_io import apply_state, payload_to_site, site_to_spec  # noqa: E402


def _json(*parts):
    with open(os.path.join(*parts), encoding='utf-8') as fh:
        return json.load(fh)


@pytest.mark.parametrize('design, site, p_group, bank', [
    ('ring', 'c003_site_v9.json', 1.91, 1), ('revc', 'c003_revC1_site.json', 1.91, 1),
    ('ring10', 'c003_site_v10.json', 1.936, 5), ('revc2', 'c003_revC2_site.json', 1.936, 1)])
def test_c003_profile_reproduces_the_reference_specs(design, site, p_group, bank):
    """The importer's c003 profile is the C003 mapping: each reference spec, as drawn, from its Site."""
    spec, notes = site_to_spec(_json(REF, site), 'c003', 'eol', p_group, bank)
    assert spec == _json(REF, f'reference_c003_{design}.spec.json')
    assert any('given on the 50 MVA system base and rebased' in n for n in notes)


def test_generic_profile_builds_and_solves():
    """Any Site, no C003 assumptions: storage converters as grid-forming PCS behind batteries, the rest as static
    generators sharing the load; Electrisim builds the network and the load flow converges."""
    import electrisim_sld as sld
    spec, notes = site_to_spec(_json(SPECS, 'c003_revC2_site_corrected.json'))
    assert len(spec['static_generators']) == 19 and len(spec['pcs']) == len(spec['batteries']) == 6
    assert 'sofcs' not in spec and any('static generators' in n for n in notes)
    with contextlib.redirect_stdout(io.StringIO()):
        net, _ = sld.build_network(spec)
        res = sld.solve(net)
    assert res['converged'] and len(net.sgen) == 19


@pytest.mark.parametrize('design, site', [('ring10', 'c003_site_v10_corrected.json'),
                                          ('revc2', 'c003_revC2_site_corrected.json')])
def test_drawn_diagram_exports_as_its_site(design, site):
    """
    The drawn C003 diagram exported as a Site recovers the corrected Site element by element - buses, cables,
    transformers on their own rating, switches as drawn (normally open ones open), loads, converters, the genset,
    the earthing - and, the corrected Site lent as its template, every state and relay and each converter's class.
    The one difference is drawn on purpose: the ring's TR_IT_A is the Site's 3.5 MVA unit drawn as a bank of five.
    """
    ref = _json(SPECS, site)
    got = payload_to_site(_json(REF, f'reference_c003_{design}.diagram_payload.json'), ref['id'],
                          ref['base']['s_base_mva'], template=ref)
    fields = {'buses': ('v_nom_kv',), 'branches': ('from_bus', 'to_bus', 'r_pu', 'x_pu', 'rating_mva'),
              'transformers': ('hv_bus', 'lv_bus', 's_rated_mva', 'x_pu', 'r_pu', 'hv_config', 'lv_config'),
              'switches': ('from_bus', 'to_bus', 'normally_closed'), 'loads': ('bus', 'p_mw', 'q_mvar'),
              'converters': ('bus', 's_rated_mva', 'control', 'energy_mwh'), 'machines': ('bus', 's_rated_mva'),
              'grounding': ('bus', 'z0_ohm')}
    diffs = []
    for key, keys in fields.items():
        a = {x['id']: x for x in ref[key] if key != 'grounding' or x.get('kind') == 'low_resistance_grounded'}
        b = {x['id']: x for x in got[key]}
        assert set(a) == set(b), key
        for i in a:
            for f in keys:
                va, vb = a[i].get(f), b[i].get(f)
                same = (math.isclose(va, vb, rel_tol=2e-3, abs_tol=1e-6)
                        if isinstance(va, (int, float)) and not isinstance(va, bool) and vb is not None else va == vb)
                if not same:
                    diffs.append((i, f, va, vb))
    assert diffs == ([('TR_IT_A', 's_rated_mva', 3.5, 17.5)] if design == 'ring10' else [])
    assert [i['id'] for i in got['islands']] == [i['id'] for i in ref['islands']]
    assert len(got['relays']) == len(ref['relays'])
    mgtwin_repo = os.path.join(HERE, '..', '..', '..', 'mgtwin')
    if os.path.isdir(mgtwin_repo):
        sys.path.insert(0, mgtwin_repo)
        from mgtwin.ontology import Site
        Site.model_validate(got)


def test_a_diagram_without_a_template_is_its_drawn_state():
    """No template: the Site carries the drawn state alone (AS_DRAWN, the in-service sources), and reading that state
    back onto the diagram changes nothing."""
    payload = _json(REF, 'reference_c003_revc2.diagram_payload.json')
    site = payload_to_site(payload)
    (state,) = site['islands']
    assert state['id'] == 'AS_DRAWN' and not state['grid_connected'] and 'GEN_BS' not in state['energized_source_ids']
    assert {c['control'] for c in site['converters']} == {'grid_forming_droop', 'grid_following'}
    assert apply_state(payload, site, 'AS_DRAWN') == payload


def test_a_state_read_onto_the_diagram():
    """A state as mgtwin reads it: Rev C.2's fast transfer closes CB_SOFC01_2, opens CB_SOFC01_1, takes SOFC01 out;
    a load factor scales its load, and a load_scale given replaces it."""
    payload, site = _json(REF, 'reference_c003_revc2.diagram_payload.json'), _json(SPECS, 'c003_revC2_site_corrected.json')
    p = apply_state(payload, site, 'FAST_TRANSFER_WHOLE_SECTION')
    cell = {v.get('userFriendlyName'): v for v in p.values() if isinstance(v, dict)}
    assert cell['CB_SOFC01_2']['closed'] == 'true' and cell['CB_SOFC01_1']['closed'] == 'false'
    assert cell['SOFC01 inverter']['in_service'] == 'false' and cell['SOFC06 inverter']['in_service'] == 'true'
    p = apply_state(payload, site, 'MVA_BUS_FAULT')
    load = next(v for v in p.values() if isinstance(v, dict) and str(v.get('typ', '')).startswith('Load')
                and str(v.get('userFriendlyName', '')).startswith('IT_cordB'))
    assert float(load['p_mw']) == pytest.approx(25.0)
    p = apply_state(payload, site, 'MVA_BUS_FAULT', load_scale={'IT_cordB': 1.0})
    load = next(v for v in p.values() if isinstance(v, dict) and str(v.get('typ', '')).startswith('Load')
                and str(v.get('userFriendlyName', '')).startswith('IT_cordB'))
    assert float(load['p_mw']) == pytest.approx(12.5)
