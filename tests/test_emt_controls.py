"""
The detailed machine's exciters and governors (emt_controls) against ANDES's own on the same case: a machine
alone in an island with its exciter and governor, a 10 MW resistive step on its 50 MW. Each ANDES case gets the
parameters the EMT study resolved, so that both run the same numbers; the EMT's voltage in proportion to speed
is left out, as ANDES's GENROU leaves it out.
"""
import json

import numpy as np
import pytest

import emt_controls
import emt_machine
import test_emt_machine_park as T

STEP_MW, T_STEP, T_END = 10.0, 0.2, 3.0


def _andes(exc, gov):
    import andes
    from andes_electrisim import tds_values
    andes.config_logger(stream_level=40)
    ss = andes.System()
    ss.config.freq, ss.config.mva = T.F, 100
    ss.files.no_output = True
    kv = T.KV
    zb = kv * kv / 100
    ss.add('Bus', idx=1, name='A', Vn=kv)
    ss.add('Bus', idx=2, name='B', Vn=kv)
    ss.add('Line', idx='L', bus1=1, bus2=2, Vn1=kv, Vn2=kv, r=1e-4 / zb, x=1e-4 / zb, fn=T.F)
    ss.add('Slack', idx='GT', bus=1, Vn=kv, Sn=T.MVA, v0=1.0, a0=0, p0=T.P_MW / 100)
    ss.add('GENROU', idx='M', bus=1, gen='GT', Sn=T.MVA, Vn=kv, fn=T.F, M=6.0, D=0, ra=0.0025, xl=0.15, xd=1.8, xq=1.7, xd1=0.3, xq1=0.55,
           xd2=0.25, xq2=0.25, Td10=8.0, Td20=0.03, Tq10=0.4, Tq20=0.05)
    ss.add(exc[0], idx='E', syn='M', **{k: v for k, v in exc[1].items() if k not in ('u', 'ue', 'Sn', 'Vn', 'bus')})
    ss.add(gov[0], idx='G', syn='M', **{k: v for k, v in gov[1].items() if k not in ('u', 'ue', 'Sg', 'Vn', 'Tn', 'wref0')})
    ss.add('PQ', idx='LD', bus=2, Vn=kv, p0=T.P_MW / 100, q0=T.Q_MVAR / 100)
    ss.add('Fault', idx='F', bus=2, tf=T_STEP, tc=T_END + 10.0, rf=(kv * kv / STEP_MW) / zb, xf=1e-8)
    ss.PQ.config.p2p, ss.PQ.config.p2i, ss.PQ.config.p2z = 0, 0, 1
    ss.PQ.config.q2q, ss.PQ.config.q2i, ss.PQ.config.q2z = 0, 0, 1
    ss.setup()
    ss.PFlow.run()
    ss.TDS.config.tf, ss.TDS.config.tstep, ss.TDS.config.no_tqdm = T_END, 0.002, 1
    ss.TDS.run()
    t = np.asarray(ss.dae.ts.t)
    return t, T.F * tds_values(ss, ss.GENROU.omega)[:, 0], tds_values(ss, ss.Bus.v)[:, 0]


@pytest.mark.parametrize('exc, gov', [('EXDC2', 'TGOV1'), ('SEXS', 'GAST')])
def test_a_load_step_against_andes(client, quiet, monkeypatch, exc, gov):
    monkeypatch.setattr(emt_machine.ParkMachine, 'speed_voltage', False)
    request = T._request(duration_ms=T_END * 1e3, ac_fault_bus='a', ac_fault_type='abc', ac_fault_time_ms=T_STEP * 1e3, ac_fault_duration_ms=0,
                         ac_fault_resistance_ohm=T.KV ** 2 / STEP_MW)
    gen = next(v for v in request.values() if v.get('typ') == 'Generator0')
    gen.update(dyn_exciter_model=exc, dyn_governor_model=gov)
    emt, warnings = T._emt(client, quiet, request)
    (m,) = emt['machines']
    assert (m['exciter_model'], m['governor_model']) == (exc, gov)
    t_a, f_a, v_a = _andes(emt_controls.parameters('exc', gen), emt_controls.parameters('gov', gen))
    t_e, f_e = np.array(m['trace']['t_ms']) / 1e3, np.array(m['trace']['f_hz'])
    for x in (0.3, 0.6, 1.0, 1.5, 2.0, 2.9):
        assert np.interp(x, t_e, f_e) == pytest.approx(np.interp(x, t_a, f_a), abs=0.003), x
    bus = next(b for b in emt['ac']['buses'] if b['label'] == 'A')
    assert bus['v_rms_final_pu'] == pytest.approx(v_a[-1], abs=0.01)


def test_it_starts_steady_with_its_default_exciter_and_governor(client, quiet):
    """No dyn_ fields: EXDC2 and TGOV1, as ANDES's study takes them, holding the load flow's state."""
    request = T._request(duration_ms=300)
    gen = next(v for v in request.values() if v.get('typ') == 'Generator0')
    for k in ('dyn_exciter_model', 'dyn_governor_model'):
        gen.pop(k)
    emt, _ = T._emt(client, quiet, request)
    (m,) = emt['machines']
    assert (m['exciter_model'], m['governor_model']) == ('EXDC2', 'TGOV1')
    assert m['p_end_mw'] == pytest.approx(T.P_MW, rel=2e-3) and m['f_end_hz'] == pytest.approx(T.F, abs=2e-3)
    assert m['efd_end_pu'] == pytest.approx(m['efd_start_pu'], rel=2e-3)


def test_a_model_the_emt_lacks_runs_simplified_and_says_so(client, quiet):
    request = T._request(duration_ms=50)
    next(v for v in request.values() if v.get('typ') == 'Generator0').update(dyn_exciter_model='IEEEX1', dyn_governor_model='TGOV1')
    emt, warnings = T._emt(client, quiet, request)
    assert emt['machines'][0]['exciter_model'] is None and emt['machines'][0]['governor_model'] == 'TGOV1'
    assert any('IEEEX1' in w and 'simplified' in w for w in warnings)
