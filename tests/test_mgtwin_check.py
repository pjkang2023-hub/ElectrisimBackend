"""
mgtwin's screening beside Electrisim's studies: Check design (/mgtwin/check) runs the twin's gates on the drawn
network, and a study run in a Site state carries the twin's validity envelope as a label (X-Electrisim-Envelope).
"""
import contextlib
import io
import json
import os

import pytest

pytest.importorskip('mgtwin')

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, 'reference')


def _json(name):
    with open(os.path.join(REF, name), encoding='utf-8') as fh:
        return json.load(fh)


def _post(client, path, body):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        r = client.post(path, json=body)
    return r, json.loads(r.get_data(as_text=True))


def test_check_design_runs_the_gates_on_the_drawn_network(client):
    """Rev C.2 as drawn, its Site lent as template, through mgtwin's seven gates: the Site's own result - 3 FAIL,
    all three the energisation gate's (a whole section re-closed at once, a genset energising a PCS transformer,
    two groups on two PCS)."""
    from mgtwin.gates import Severity, run_all
    from mgtwin.ontology import Site
    site = _json('c003_revC2_site_corrected.json')
    r, out = _post(client, '/mgtwin/check', {'diagram': _json('reference_c003_revc2.diagram_payload.json'),
                                             'site_id': site['id'], 'template': site})
    assert r.status_code == 200
    direct = run_all(Site.model_validate(site))
    assert out['counts'] == {s.value: len(direct.by_severity(s)) for s in Severity}
    fails = [f for f in out['findings'] if f['severity'] == 'fail']
    assert len(fails) == 3 and {f['gate'] for f in fails} == {'energisation'}
    assert len(out['states']) == 14
    assert _post(client, '/mgtwin/check', {})[0].status_code == 400


def test_the_ring_as_drawn_differs_from_its_site_only_by_the_drawn_bank(client):
    """The ring's diagram draws TR_IT_A as a bank of five (17.5 MVA): the loading gate FAILs it only where both
    cords' 26.3 MVA reach it (BUS_MVB_FAULT), and the declared-ungrounded IT pockets the template lends keep the
    grounding gate clean."""
    site = _json('c003_site_v10_corrected.json')
    r, out = _post(client, '/mgtwin/check', {'diagram': _json('reference_c003_ring10.diagram_payload.json'),
                                             'site_id': site['id'], 'template': site})
    loading = [f for f in out['findings'] if f['gate'] == 'loading' and f['severity'] == 'fail']
    assert [f['scope'] for f in loading] == ['BUS_MVB_FAULT/TR_IT_A']
    assert out['by_gate']['grounding']['fail'] == 0


@pytest.mark.parametrize('design, site, state, typ, verdict, reason', [
    ('ring10', 'c003_site_v10_corrected.json', 'NORMAL', 'PowerFlowPandaPower Parameters', 'outside', 'TR_IT_A'),
    ('revc2', 'c003_revC2_site_corrected.json', 'NORMAL', 'PowerFlowPandaPower Parameters', 'inside', None),
    ('revc2', 'c003_revC2_site_corrected.json', 'NORMAL', 'ShortCircuitPandaPower Parameters', 'provisional',
     'PCS1, PCS2, PCS3, PCS4, PCS5, PCS6: dynamic parameters assumed, not identified'),
])
def test_a_study_in_a_state_carries_the_envelope(client, design, site, state, typ, verdict, reason):
    """A study run in a Site state carries the twin's verdict on it - about the Site's data, not mgtwin's tier: the
    ring's load flow is outside (its gates fail TR_IT_A in that state), Rev C.2's inside, a short circuit
    provisional on the converters' assumed current limits and ride-through."""
    payload, site = _json(f'reference_c003_{design}.diagram_payload.json'), _json(site)
    body = {**payload, '0': {**payload['0'], 'typ': typ}, 'mgtwin_state': {'site': site, 'state_id': state}}
    if 'ShortCircuit' in typ:
        body['0'].update({'fault_type': '3ph', 'fault_location': 'max', 'fault_bus_mode': 'all', 'fault_bus_ids': [],
                          'fault_bus_names': [], 'fault_impedance': '6', 'frequency': '50', 'topology': 'auto',
                          'tk_s': '1', 'r_fault_ohm': '0', 'x_fault_ohm': '0', 'inverse_y': 'True'})
    r, _ = _post(client, '/', body)
    assert r.status_code == 200 and r.headers['X-Electrisim-Envelope'] == verdict
    why = r.headers.get('X-Electrisim-Envelope-Why', '')
    assert (reason in why) if reason else not why
    assert 'X-Electrisim-Envelope' not in _post(client, '/', payload)[0].headers


def test_each_study_reads_its_question():
    """Every study the backend dispatches on maps to the envelope question its result rests on - or to none, and
    then carries no label (the names as the frontend sends them in cell '0')."""
    from mgtwin_check import phenomenon_for
    expect = {
        'loading': ['PowerFlowPandaPower', 'PowerFlowOpenDss', 'OptimalPowerFlowPandaPower', 'AcDcOpfPandaPower',
                    'StateEstimationPandaPower', 'ContingencyAnalysisPandaPower'],
        'protection': ['ShortCircuitPandaPower', 'ShortCircuitOpenDss', 'ShortCircuitAnsi', 'ProtectionCoordinationPandaPower',
                       'ArcFlashPandaPower', 'PoiFaultStudy', 'DcFaultStudy', 'EmtStudy'],
        'frequency': ['TransientStabilityAndes', 'EigenvalueAndes', 'MotorStartingPandaPower'],
        None: ['GridCodePqPandaPower', 'GridCodeVqPandaPower', 'RPCAnalysisPandaPower', 'EconomicAnalysisPandaPower',
               'DataCenterSiteScreeningPandaPower', 'DgInterconnectionOpenDss', 'FuseCharacteristicPreviewPandaPower'],
    }
    got = {ph: [t for t in typs if phenomenon_for(f'{t} Parameters') != ph] for ph, typs in expect.items()}
    assert all(not wrong for wrong in got.values()), got
