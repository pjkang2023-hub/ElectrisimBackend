"""
Operating states in the API: a study run in one of the network's mgtwin Site states (mgtwin_state), the states
listed for a selector (/mgtwin/states), and the drawn diagram exported as a Site (/mgtwin/site).
"""
import contextlib
import io
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, 'reference')
sys.path.insert(0, os.path.join(HERE, '..'))

from mgtwin_site_io import apply_state, payload_to_site  # noqa: E402


def _json(name):
    with open(os.path.join(REF, name), encoding='utf-8') as fh:
        return json.load(fh)


def _post(client, path, body, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        r = client.post(path, json=body, **kw)
    return r, json.loads(r.get_data(as_text=True))


def test_a_study_runs_in_the_state_asked_for(client):
    """The load flow of the drawn diagram with mgtwin_state MVA_BUS_FAULT is the load flow of that state read onto
    it - board MVA lost, the halls on cord B (IT_cordB at 25 MW) - and the response names the state it ran in."""
    payload, site = _json('reference_c003_revc2.diagram_payload.json'), _json('c003_revC2_site_corrected.json')
    r, out = _post(client, '/', {**payload, 'mgtwin_state': {'site': site, 'state_id': 'MVA_BUS_FAULT'}},
                   headers={'Origin': 'http://127.0.0.1:5501'})
    assert r.status_code == 200 and r.headers['X-Electrisim-State'] == 'MVA_BUS_FAULT'
    assert 'X-Electrisim-State' in r.headers.get('Access-Control-Expose-Headers', '')
    _, by_hand = _post(client, '/', apply_state(payload, site, 'MVA_BUS_FAULT'))
    assert out['loads'] == by_hand['loads'] and out['busbars'] == by_hand['busbars']
    assert sum(ld['p_mw'] for ld in out['loads']) == pytest.approx(25.0 + 2.5)       # cord B doubled, Other B
    _, normal = _post(client, '/', payload)
    assert sum(ld['p_mw'] for ld in normal['loads']) == pytest.approx(30.0)
    assert 'X-Electrisim-State' not in _post(client, '/', payload)[0].headers


def test_an_unknown_state_is_refused_clearly(client):
    site = _json('c003_revC2_site_corrected.json')
    r, out = _post(client, '/', {**_json('reference_c003_revc2.diagram_payload.json'),
                                 'mgtwin_state': {'site': site, 'state_id': 'NOT_A_STATE'}})
    assert r.status_code == 400 and "NOT_A_STATE" in out['message']


def test_the_states_listed_for_a_selector(client):
    r, out = _post(client, '/mgtwin/states', {'site': _json('c003_revC2_site_corrected.json')})
    assert r.status_code == 200 and out['site_id'] == 'CAMPAIGN-003-revC2'
    states = {s['id']: s for s in out['states']}
    assert len(states) == 14 and states['MVA_BUS_FAULT']['load_factors'] == {'IT_cordB': 2.0}
    assert states['FAST_TRANSFER_WHOLE_SECTION']['closed_switches'][0] == 'CB_SOFC01_2'
    assert _post(client, '/mgtwin/states', {'site': {}})[0].status_code == 400


def test_the_drawn_diagram_exported_as_a_site(client):
    payload, site = _json('reference_c003_revc2.diagram_payload.json'), _json('c003_revC2_site_corrected.json')
    r, out = _post(client, '/mgtwin/site', {'diagram': payload, 'site_id': site['id'], 'template': site})
    assert r.status_code == 200
    assert out == json.loads(json.dumps(payload_to_site(payload, site['id'], 50.0, template=site)))
    assert len(out['islands']) == 14 and len(out['relays']) == 59
    r, out = _post(client, '/mgtwin/site', {'diagram': payload})
    assert [i['id'] for i in out['islands']] == ['AS_DRAWN']
    assert _post(client, '/mgtwin/site', {})[0].status_code == 400
