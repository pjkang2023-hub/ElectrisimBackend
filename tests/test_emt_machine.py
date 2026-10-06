"""
The synchronous machine in the EMT study (emt_machine): its rotor, governor
and exciter. A machine alone in an island takes a load step as ANDES's
classical machine (GENCLS) and GAST governor do on the same case; a machine
and a grid-forming PCS islanded share the load at one frequency, each by its
own droop.
"""
import json
import os

import numpy as np
import pytest

import emt_machine
from andes_electrisim import _EXCITER_DEFAULTS, _GOVERNOR_DEFAULTS
from test_pcs import _bus, _der, _pcs, _two_buses

HERE = os.path.join(os.path.dirname(__file__), 'emt_reference')
with open(os.path.join(HERE, 'emt_machine_benchmark_params.json')) as fh:
    P = json.load(fh)
M, G = P['machine'], P['governor']


def _gen(bus, p_mw, slack, **dyn):
    row = {'typ': 'Generator0', 'name': 'gt', 'id': 'cell-gt', 'userFriendlyName': 'GT', 'bus': bus,
           'p_mw': str(p_mw), 'vm_pu': '1.0', 'sn_mva': str(M['s_rated_mva']), 'scaling': '1',
           'slack': 'true' if slack else 'false', 'in_service': 'true', 'controllable': 'true', 'min_p_mw': '',
           'max_p_mw': '', 'vn_kv': str(M['v_ll_kv']), 'xdss_pu': '0.15', 'rdss_ohm': '', 'cos_phi': '0.8',
           'pg_percent': '', 'power_station_trafo': '', 'dyn_H': str(M['h_s']), 'dyn_xd1': str(M['xd1_pu']),
           'dyn_governor_model': 'GAST', 'dyn_gov_R': str(G['r']), 'dyn_gov_T1': str(G['t1']),
           'dyn_gov_T2': str(G['t2']), 'dyn_exciter_model': 'NONE'}
    row.update({k: str(v) for k, v in dyn.items()})
    return row


def _emt(client, quiet, request):
    with quiet():
        out = json.loads(client.post('/', json=request).get_data(as_text=True))
    assert 'emt' in out, out.get('message')
    return out['emt']


def _alone(params):
    v = M['v_ll_kv']
    load = {'typ': 'Load0', 'name': 'ld', 'id': 'cell-ld', 'userFriendlyName': 'LD', 'bus': 'a',
            'p_mw': str(P['load']['p_mw']), 'q_mvar': str(P['load']['q_mvar']), 'const_z_percent': '100',
            'const_i_percent': '0', 'sn_mva': '0', 'scaling': '1', 'type': 'wye', 'in_service': 'true'}
    return {'0': {k: str(x) for k, x in params.items()}, '1': _bus('a', v), '2': _gen('a', P['load']['p_mw'], True),
            '3': load}


def _andes_load_step():
    """The same case in ANDES: GENCLS and GAST, the step a balanced resistance to ground (its Fault)."""
    import andes
    from andes_electrisim import tds_values
    andes.config_logger(stream_level=40)
    v, s = M['v_ll_kv'], M['s_rated_mva']
    ss = andes.System()
    ss.config.freq, ss.config.mva = P['f_hz'], 100
    ss.files.no_output = True
    zb = v * v / 100
    ss.add('Bus', idx=1, name='A', Vn=v)
    ss.add('Bus', idx=2, name='B', Vn=v)       # a lone machine's own bus takes no fault in ANDES: a negligible line
    ss.add('Line', idx='L', bus1=1, bus2=2, Vn1=v, Vn2=v, r=1e-4 / zb, x=1e-4 / zb, fn=P['f_hz'])
    ss.add('Slack', idx='GT', bus=1, Vn=v, Sn=s, v0=1.0, a0=0, p0=P['load']['p_mw'] / 100)
    ss.add('GENCLS', idx='M', bus=1, gen='GT', Sn=s, Vn=v, fn=P['f_hz'], M=2 * M['h_s'], D=0,
           ra=0.01 * M['xd1_pu'], xd1=M['xd1_pu'])
    ss.add('GAST', idx='G', syn='M', R=G['r'], T1=G['t1'], T2=G['t2'], T3=G['t3'], VMAX=G['vmax'], VMIN=G['vmin'],
           AT=1.2, KT=2.0)
    ss.add('PQ', idx='LD', bus=2, Vn=v, p0=P['load']['p_mw'] / 100, q0=P['load']['q_mvar'] / 100)
    ss.add('Fault', idx='F', bus=2, tf=P['step']['t_s'], tc=P['t_end_s'] + 10.0,
           rf=(v * v / P['step']['p_mw']) / zb, xf=1e-8)
    ss.setup()
    ss.PFlow.run()
    ss.TDS.config.tf = P['t_end_s']
    ss.TDS.config.tstep = 0.005
    ss.TDS.config.no_tqdm = 1
    ss.TDS.run()
    return np.asarray(ss.dae.ts.t), P['f_hz'] * tds_values(ss, ss.GENCLS.omega)[:, 0]


def test_load_step_against_andes(client, quiet):
    """
    A 0.1 MW resistive step on a machine alone in an island: its frequency
    through the run within 5 mHz of ANDES's classical machine and GAST
    governor on the same case, its nadir within 5 mHz and 10 ms.
    """
    emt = _emt(client, quiet, _alone({
        'typ': 'EmtStudy Parameters', 'user_email': 't@t', 'time_step_us': P['dt_us'],
        'duration_ms': P['t_end_s'] * 1e3, 'generator_model': 'machine', 'ac_fault_bus': 'a', 'ac_fault_type': 'abc',
        'ac_fault_time_ms': P['step']['t_s'] * 1e3, 'ac_fault_duration_ms': 0,
        'ac_fault_resistance_ohm': M['v_ll_kv'] ** 2 / P['step']['p_mw']}))
    (m,) = emt['machines']
    t_e, f_e = np.array(m['trace']['t_ms']) / 1e3, np.array(m['trace']['f_hz'])
    t_a, f_a = _andes_load_step()
    for x in (0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 3.0, P['t_end_s'] - 0.01):
        assert np.interp(x, t_e, f_e) == pytest.approx(np.interp(x, t_a, f_a), abs=0.005), x
    assert f_e.min() == pytest.approx(f_a.min(), abs=0.005)
    assert t_e[np.argmin(f_e)] == pytest.approx(t_a[np.argmin(f_a)], abs=0.01)
    assert f_e.min() < P['f_hz'] - 0.5         # a real dip: 0.4 pu of load on a 0.5 MVA machine


def test_machine_and_grid_forming_pcs_share_an_island(client, quiet):
    """
    Islanded at 20 ms, a gas turbine (0.5 MVA, 4 % droop) and a grid-forming
    battery PCS (1 MVA, 2 %) take the grid's share at one frequency: the
    turbine's governor f0 (1 - R dPm / S), the PCS's f0 (1 - droop dP / S).
    """
    emt = _emt(client, quiet, _two_buses(
        _gen('b', 0.2, False), _der('Battery', 'bb', capacity_kwh=1000),
        _pcs('gb', 'a', 'bb', control='grid_forming', s_rated_mva=1.0),
        params={'typ': 'EmtStudy Parameters', 'user_email': 't@t', 'time_step_us': '100', 'duration_ms': '3000',
                'island_time_ms': '20'}))
    (m,) = emt['machines']
    g = next(c for c in emt['converters'] if c['label'] == 'GB')
    f0 = P['f_hz']
    assert m['f_end_hz'] == pytest.approx(g['f_end_hz'], abs=1e-4)
    assert m['f_end_hz'] == pytest.approx(f0 * (1 - G['r'] * (m['pm_end_mw'] - m['pm_start_mw']) / M['s_rated_mva']),
                                          abs=2e-4)
    assert g['f_end_hz'] == pytest.approx(f0 * (1 - 0.02 * (g['p_end_mw'] - g['p_start_mw']) / 1.0), abs=5e-4)
    assert m['p_end_mw'] > m['p_start_mw'] + 0.05 and g['p_end_mw'] > 0.3
    assert all(x is not None for x in emt['island']['opened_ms'])


def test_generator_model_by_setting(client, quiet):
    """
    By default a generator stays a source behind x" unless the study islands;
    as a machine it starts where the load flow left it, steady.
    """
    base = {'typ': 'EmtStudy Parameters', 'user_email': 't@t', 'time_step_us': '100', 'duration_ms': '200'}
    assert 'machines' not in _emt(client, quiet, _alone(base))
    emt = _emt(client, quiet, _alone({**base, 'generator_model': 'machine'}))
    (m,) = emt['machines']
    assert m['p_start_mw'] == pytest.approx(P['load']['p_mw'], rel=0.01)
    assert m['f_end_hz'] == pytest.approx(P['f_hz'], abs=1e-3)
    assert m['p_end_mw'] == pytest.approx(P['load']['p_mw'], rel=1e-3)
    assert m['pm_end_mw'] == pytest.approx(m['pm_start_mw'], rel=1e-3)


def test_machine_data_from_its_dynamics_fields():
    """H from M = 2H when only M is given; its governor model's defaults; no exciter unless named."""
    d = emt_machine.machine_data({'dyn_M': '9', 'dyn_governor_model': 'TGOV1', 'dyn_gov_R': '0.05'},
                                 _GOVERNOR_DEFAULTS, _EXCITER_DEFAULTS)
    assert d['h'] == 4.5 and d['gov']['R'] == 0.05 and d['gov']['T1'] == _GOVERNOR_DEFAULTS['TGOV1']['T1']
    assert d['exc'] is None
    d = emt_machine.machine_data({'dyn_exciter_model': 'SEXS', 'dyn_governor_model': 'NONE'},
                                 _GOVERNOR_DEFAULTS, _EXCITER_DEFAULTS)
    assert d['gov'] is None and d['exc']['K'] == _EXCITER_DEFAULTS['SEXS']['K']
    assert emt_machine.machine_data(None, _GOVERNOR_DEFAULTS, _EXCITER_DEFAULTS)['gov']['R'] == 0.05
