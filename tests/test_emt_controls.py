"""
The detailed machine's exciters and governors (emt_controls) against ANDES's own on the same case: a machine
alone in an island with its exciter and governor, a 10 MW resistive step on its 50 MW. Each ANDES case gets the
parameters the EMT study resolved, so that both run the same numbers; the EMT's voltage in proportion to speed
is left out, as ANDES's GENROU leaves it out.

Compared through the whole run, the slow hydro and steam governors (HYGOV, IEESGO) taking it more than 2 Hz off
nominal: the frequencies within 3 mHz plus 0.5 % of how far it is off (measured: 7.5 mHz at 2.2 Hz off, where the
EMT's reactances follow the frequency and ANDES's stay at nominal), the terminal voltages within 0.002 pu. AC8B at its
default KA 4: at the earlier default of 40 its loop limit-cycled in ANDES as well, and two limit cycles part company.
"""
import json

import numpy as np
import pytest

import emt_controls
import emt_machine
import test_emt_machine_park as T

STEP_MW, T_STEP, T_END = 10.0, 0.2, 3.0


def _andes(exc, gov, pss=None, genrou=None):
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
           xd2=0.25, xq2=0.25, Td10=8.0, Td20=0.03, Tq10=0.4, Tq20=0.05, **(genrou or {}))
    ss.add(exc[0], idx='E', syn='M', **{k: v for k, v in exc[1].items() if k not in ('u', 'ue', 'Sn', 'Vn', 'bus')})
    ss.add(gov[0], idx='G', syn='M', **{k: v for k, v in gov[1].items() if k not in ('u', 'ue', 'Sg', 'Vn', 'Tn', 'wref0')})
    if pss is not None:
        ss.add(pss[0], idx='S', avr='E', **{k: v for k, v in pss[1].items() if k not in ('u', 'ue')})
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


@pytest.mark.parametrize('exc, gov, extra', [('EXDC2', 'TGOV1', {}), ('SEXS', 'GAST', {}), ('IEEEX1', 'IEEEG1', {}), ('ESDC2A', 'IEESGO', {}),
                                             ('EXST1', 'HYGOV', {}), ('ESST1A', 'TGOV1', {}), ('AC8B', 'GAST', {})])
def test_a_load_step_against_andes(client, quiet, monkeypatch, exc, gov, extra):
    monkeypatch.setattr(emt_machine.ParkMachine, 'speed_voltage', False)
    request = T._request(duration_ms=T_END * 1e3, ac_fault_bus='a', ac_fault_type='abc', ac_fault_time_ms=T_STEP * 1e3, ac_fault_duration_ms=0,
                         ac_fault_resistance_ohm=T.KV ** 2 / STEP_MW)
    gen = next(v for v in request.values() if v.get('typ') == 'Generator0')
    gen.update(dyn_exciter_model=exc, dyn_governor_model=gov, **extra)
    emt, warnings = T._emt(client, quiet, request)
    (m,) = emt['machines']
    assert (m['exciter_model'], m['governor_model']) == (exc, gov)
    t_a, f_a, v_a = _andes(emt_controls.parameters('exc', gen), emt_controls.parameters('gov', gen))
    t_e, f_e = np.array(m['trace']['t_ms']) / 1e3, np.array(m['trace']['f_hz'])
    v_e = np.array(m['trace']['v_pu'])
    for x in (0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 2.9):
        off = abs(np.interp(x, t_a, f_a) - T.F)
        assert np.interp(x, t_e, f_e) == pytest.approx(np.interp(x, t_a, f_a), abs=0.003 + 0.005 * off), x
        assert np.interp(x, t_e, v_e) == pytest.approx(np.interp(x, t_a, v_a), abs=0.002), x
    bus = next(b for b in emt['ac']['buses'] if b['label'] == 'A')
    assert bus['v_rms_final_pu'] == pytest.approx(v_a[-1], abs=0.005)


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


def test_every_model_the_dialog_offers_is_modelled():
    from andes_electrisim import _EXCITER_DEFAULTS, _GOVERNOR_DEFAULTS
    assert set(emt_controls.EXCITERS) == set(_EXCITER_DEFAULTS) and set(emt_controls.GOVERNORS) == set(_GOVERNOR_DEFAULTS)


def test_ieeeg1_takes_its_gain_from_the_dialogs_droop():
    """The dialog offers R; IEEEG1's gain is 1 / R. Before, its R was refused by ANDES and the machine ran with no governor."""
    model, params = emt_controls.parameters('gov', {'dyn_governor_model': 'IEEEG1', 'dyn_gov_R': '0.04'})
    assert model == 'IEEEG1' and params['K'] == pytest.approx(25.0) and 'R' not in params


def test_a_saturated_machine_against_andes(client, quiet, monkeypatch):
    """dyn_S10 and dyn_S12 in both: the load step with EXDC2 and TGOV1 on a saturated machine."""
    monkeypatch.setattr(emt_machine.ParkMachine, 'speed_voltage', False)
    request = T._request(duration_ms=T_END * 1e3, ac_fault_bus='a', ac_fault_type='abc', ac_fault_time_ms=T_STEP * 1e3, ac_fault_duration_ms=0,
                         ac_fault_resistance_ohm=T.KV ** 2 / STEP_MW)
    gen = next(v for v in request.values() if v.get('typ') == 'Generator0')
    gen.update(dyn_exciter_model='EXDC2', dyn_governor_model='TGOV1', dyn_S10='0.1', dyn_S12='0.4')
    emt, _ = T._emt(client, quiet, request)
    (m,) = emt['machines']
    t_a, f_a, v_a = _andes(emt_controls.parameters('exc', gen), emt_controls.parameters('gov', gen), genrou={'S10': 0.1, 'S12': 0.4})
    t_e, f_e = np.array(m['trace']['t_ms']) / 1e3, np.array(m['trace']['f_hz'])
    v_e = np.array(m['trace']['v_pu'])
    for x in (0.3, 0.6, 1.0, 2.0, 2.9):
        assert np.interp(x, t_e, f_e) == pytest.approx(np.interp(x, t_a, f_a), abs=0.003), x
        assert np.interp(x, t_e, v_e) == pytest.approx(np.interp(x, t_a, v_a), abs=0.004), x


def test_andes_study_takes_saturation_and_the_stabiliser(quiet):
    import andes_electrisim
    request = T._request()
    gen = next(v for v in request.values() if v.get('typ') == 'Generator0')
    gen.update(dyn_exciter_model='EXDC2', dyn_S10='0.1', dyn_S12='0.4', dyn_pss_model='IEEEST')
    request = {k: v for k, v in request.items() if v.get('typ') != 'EmtStudy Parameters'}
    with quiet():
        ss, meta = andes_electrisim.build_system(request, {})
    assert float(ss.GENROU.S10.v[0]) == pytest.approx(0.1) and float(ss.GENROU.S12.v[0]) == pytest.approx(0.4)
    assert ss.IEEEST.n == 1 and str(ss.IEEEST.avr.v[0]) == str(ss.EXDC2.idx.v[0])
    assert not any('IEEEST' in str(w) for w in meta.get('defaults_applied', []))
