"""
Data-center site screening on the drawn transmission grid: the site's
headroom, checked against pandapower's own search on the spec network.
"""
import copy
import json
import math
import os

import pandapower as pp
import pytest

import electrisim_sld as sld

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE_DIR = os.path.join(HERE, 'reference')


def _screen(client, quiet, sizes):
    with open(os.path.join(REFERENCE_DIR, 'reference_transmission.diagram_site_screening_payload.json'),
              encoding='utf-8') as handle:
        payload = json.load(handle)
    payload['0'].update(mw_sizes=sizes, include_n11='false')
    with quiet():
        text = client.post('/', json=payload).get_data(as_text=True)
    last = [line for line in text.splitlines() if line.strip()][-1]
    return json.loads(last)['data']['screening_results']


def _pandapower_headroom(site='Industrial park', power_factor=0.95):
    """The largest load at the site with every bus in 0.95-1.05 pu and no
    line or transformer above 100 %, by bisection on the spec network."""
    with open(os.path.join(REFERENCE_DIR, 'reference_transmission.spec.json'), encoding='utf-8') as handle:
        spec = json.load(handle)
    net0, _ = sld.build_network(spec)
    load = int(net0.load.index[net0.load.name == site][0])

    def within_limits(p_mw):
        net = copy.deepcopy(net0)
        net.load.loc[load, 'p_mw'] = p_mw
        net.load.loc[load, 'q_mvar'] = p_mw * math.tan(math.acos(power_factor))
        try:
            pp.runpp(net, algorithm='nr', calculate_voltage_angles=True)
        except Exception:
            return False
        if not net.res_bus.vm_pu.between(0.95, 1.05).all():
            return False
        return all((net['res_' + table].loading_percent <= 100).all() for table in ('line', 'trafo', 'trafo3w'))

    lo, hi = 0.0, 5000.0
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if within_limits(mid) else (lo, mid)
    return lo


def test_site_headroom_does_not_depend_on_the_size_asked_about(client, quiet):
    """
    The headroom is the site's: the largest load it takes within the limits,
    whatever size is asked about. A size whose load flow failed reported 0 MW,
    so every row of the dialog's default 300 / 500 / 1000 MW at the
    transmission grid's 20 kV Industrial park said it could take nothing. It
    takes 21.65 MW, up to the cable's rating.
    """
    rows = _screen(client, quiet, '2,300')
    by_size = {row['requested_mw']: row for row in rows}
    want = _pandapower_headroom()
    assert want == pytest.approx(21.65, abs=0.01)
    for size, row in by_size.items():
        assert row['headroom_mw'] == pytest.approx(want, abs=0.01), size

    small, large = by_size[2.0], by_size[300.0]
    assert small['upgrade_likely'] is False
    assert large['upgrade_likely'] is True
    assert 'did not converge' in large['notes'] and '21.65 MW' in large['notes']
