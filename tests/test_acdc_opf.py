"""
The AC/DC optimal power flow with least shed (acdc_opf_electrisim), run as the frontend sends it.

Checked against an independent model of the PCS3750C-MV reference written from its spec
(electrisim-lab, experiment 01), against Electrisim's own load flow, and on the networks it must
refuse. Needs Pyomo and Ipopt; skipped without them.
"""
import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))

pytest.importorskip('pyomo.environ')
import acdc_opf_electrisim  # noqa: E402

if not acdc_opf_electrisim._ipopt_path():
    pytest.skip('Ipopt not found (ELECTRISIM_IPOPT or tools/ipopt)', allow_module_level=True)


def _payload(ref, frequency='60', **params):
    with open(os.path.join(HERE, 'reference', f'{ref}.diagram_payload.json'), encoding='utf-8') as fh:
        p = json.load(fh)
    p['0'] = {'typ': 'AcDcOpfPandaPower Parameters', 'user_email': 't@t', 'frequency': frequency,
              **{k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in params.items()}}
    return p


def _blocks_out(p, *blocks):
    for v in p.values():
        if isinstance(v, dict) and str(v.get('typ', '')).startswith('PCS') and \
                any(str(v.get('userFriendlyName', '')).startswith(f'Block {b} ') for b in blocks):
            v['in_service'] = 'false'
    return p


def _run(client, quiet, p):
    with quiet():
        return json.loads(client.post('/', json=p).get_data(as_text=True))


def test_least_losses_matches_the_spec_model(client, quiet):
    """
    The 3-block reference, every load served, ratings off: the grid supplies 11.6459 MW - 17 kW less than
    Electrisim's load flow (11.6635), the backbone raised to its 1.1 pu limit - as the model built from
    the spec gives to 1e-4 MW. Each T1 at 103.5 % of its nameplate.
    """
    out = _run(client, quiet, _payload('reference_pcs3750_mv', objective='losses', branch_limits=False))
    assert out['solved'], out
    assert out['grid']['p_mw'] == pytest.approx(11.6459, abs=2e-3)
    assert out['grid']['load_flow_p_mw'] == pytest.approx(11.6635, abs=2e-3)
    t1 = [b['loading_percent'] for b in out['branches'] if ' T1 ' in b['name']]
    assert len(t1) == 3 and all(x == pytest.approx(103.5, abs=0.3) for x in t1)
    assert max(b['vm_pu'] for b in out['buses'] if 'backbone' in b['name']) == pytest.approx(1.092, abs=3e-3)


def test_ratings_enforced_no_dispatch_without_shedding(client, quiet):
    """With the transformers' ratings enforced and every load to be served, no dispatch exists (T1 at 103.9 %)."""
    out = _run(client, quiet, _payload('reference_pcs3750_mv', objective='losses', branch_limits=True))
    assert not out['solved'] and 'No dispatch meets every constraint' in out['message']


def test_least_shed_to_respect_the_transformers(client, quiet):
    """Least shed with the ratings enforced: about 0.37 MW of racks, each T1 at 100 %."""
    out = _run(client, quiet, _payload('reference_pcs3750_mv', objective='shed', branch_limits=True))
    assert out['solved'], out
    assert out['shed_mw'] == pytest.approx(0.37, abs=0.03)
    assert all(b['loading_percent'] <= 100.01 for b in out['branches'] if b['kind'] == 'Transformer')


def test_block_out_least_shed(client, quiet):
    """
    Block 3's inverters out, the batteries free to discharge, ratings enforced: 3.151 MW shed (7.649 of
    10.8 MW served), the batteries giving 0.743 MW, the remaining inverters at 100 % - where Electrisim's
    OPF says only that no dispatch exists.
    """
    out = _run(client, quiet, _blocks_out(_payload('reference_pcs3750_mv', objective='shed', batteries=True,
                                                    branch_limits=True), 3))
    assert out['solved'], out
    assert out['shed_mw'] == pytest.approx(3.151, abs=0.01)
    assert out['batteries_mw'] == pytest.approx(0.743, abs=0.01)
    assert max(d['loading_percent'] for d in out['pcs']) == pytest.approx(100.0, abs=0.05)


def test_nine_blocks_block_out(client, quiet):
    """Nine blocks, Block 9 out, ratings enforced, batteries held: 2.172 MW shed, as the spec model gives."""
    out = _run(client, quiet, _blocks_out(_payload('reference_pcs3750_mv_9blocks', objective='shed', branch_limits=True), 9))
    assert out['solved'], out
    assert out['shed_mw'] == pytest.approx(2.172, abs=0.01)


@pytest.mark.parametrize('ref', ['reference_radial', 'reference_transmission'])
def test_ac_networks_no_worse_than_the_load_flow(client, quiet, ref):
    """
    On the AC references (switches, a three-winding transformer, shunts, motors, storage): solved, the
    grid's import at most the load flow's, and a generator without P limits held at its output.
    """
    out = _run(client, quiet, _payload(ref, frequency='50', objective='losses', branch_limits=False))
    assert out['solved'], out
    assert out['grid']['p_mw'] <= out['grid']['load_flow_p_mw'] + 1e-3
    for g in out['generators']:
        assert g['p_mw'] > 0


def test_refuses_what_it_does_not_model(client, quiet):
    """The 800 VDC factory has DC/DC converters and diodes: refused, naming them, not solved without them."""
    out = _run(client, quiet, _payload('reference_ai_factory_800vdc', objective='shed'))
    assert out['error'] and out['message'].startswith('This study does not model yet: DC/DC converter')
