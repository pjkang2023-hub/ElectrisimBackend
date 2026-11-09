"""
An ANDES study starts where Electrisim's load flow is: a load drawn as constant impedance (or current) draws its
rated power times V^2 (or V) there, and ANDES's power flow, which holds a PQ load at its p0, is given that power.
Holding the drawn power instead started the WSCC 9-bus case's rotor angles 0.4 deg off ETAP's (9 Nov 2026).
"""
import pytest

import andes_electrisim
from test_andes_pcs import TDS
from test_pcs import _two_buses

GEN = {'typ': 'Generator0', 'name': 'g', 'id': 'cell-g', 'userFriendlyName': 'G', 'bus': 'a', 'p_mw': '0.3', 'vm_pu': '1', 'sn_mva': '2',
       'vn_kv': '0.4', 'in_service': 'true', 'xdss_pu': '0.2', 'rdss_ohm': '', 'cos_phi': '0.9', 'scaling': '1', 'slack': 'false',
       'controllable': 'true', 'min_p_mw': '', 'max_p_mw': '', 'pg_percent': '', 'power_station_trafo': ''}


def _request(z, i):
    request = _two_buses(dict(GEN), params={**TDS, 'tf': '1', 'fault_enabled': 'false'})
    load = next(v for v in request.values() if v.get('typ') == 'Load0')
    load.update(const_z_percent=str(z), const_i_percent=str(i))
    return request, load


@pytest.mark.parametrize('z, i', [(100, 0), (0, 100), (50, 25)])
def test_its_power_flow_starts_at_the_load_flows_load(quiet, z, i):
    request, load = _request(z, i)
    flow, why = andes_electrisim._electrisim_flow(request, 50.0)
    assert flow is not None, why
    vm = float(flow.res_bus.at[flow.bus.index[flow.bus['name'] == load['bus']][0], 'vm_pu'])
    assert vm < 0.995                      # the line drops it, so the test means something
    want = float(load['p_mw']) * (z / 100 * vm * vm + i / 100 * vm + (1 - (z + i) / 100))
    assert float(flow.res_load['p_mw'].sum()) == pytest.approx(want, rel=1e-6)
    with quiet():
        ss, meta = andes_electrisim.build_system(request, request['0'])
    pq = ss.PQ
    k = list(pq.name.v).index(load['userFriendlyName'])
    assert float(pq.p0.v[k]) * float(ss.config.mva) == pytest.approx(want, rel=1e-6)


def test_a_constant_power_load_is_left_at_its_drawn_power(quiet):
    request, load = _request(0, 0)
    with quiet():
        ss, _ = andes_electrisim.build_system(request, request['0'])
    k = list(ss.PQ.name.v).index(load['userFriendlyName'])
    assert float(ss.PQ.p0.v[k]) * float(ss.config.mva) == pytest.approx(float(load['p_mw']))
