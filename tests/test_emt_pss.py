"""
The detailed machine's stabiliser (IEEEST, emt_controls.Ieeest): its blocks against their transfer function on
paper, then the machine with it against ANDES's own on test_emt_controls's load step, and ANDES's study now
taking it (it was refused before for want of its exciter's link).
"""
import cmath
import math

import numpy as np
import pytest

import emt_controls
import emt_machine
import test_emt_controls as C
import test_emt_machine_park as T

P = dict(MODE=1, A1=0.01, A2=0.0004, A3=0.02, A4=0.0003, A5=0.05, A6=0.0002, T1=0.15, T2=0.05, T3=0.3, T4=0.03,
         T5=2.0, T6=2.0, KS=0.5, LSMAX=9.0, LSMIN=-9.0, VCU=0.0, VCL=0.0)


def _h(f):
    s = 2j * math.pi * f
    p = P
    return (1.0 / (1 + p['A1'] * s + p['A2'] * s * s) * (1 + p['A5'] * s + p['A6'] * s * s) / (1 + p['A3'] * s + p['A4'] * s * s)
            * (1 + p['T1'] * s) / (1 + p['T2'] * s) * (1 + p['T3'] * s) / (1 + p['T4'] * s) * p['KS'] * p['T5'] * s / (1 + p['T6'] * s))


@pytest.mark.parametrize('f', [0.3, 1.0, 3.0, 8.0])
def test_its_blocks_follow_their_transfer_function(f):
    dt = 1e-4
    pss = emt_controls.Ieeest(P, 0.0)
    t = np.arange(0.0, 12.0, dt)
    u = 1e-3 * np.sin(2 * math.pi * f * t)
    y = np.array([pss.step(dt, x, 1.0) for x in u])
    sel = t >= 12.0 - math.ceil(2.0 * f) / f
    ref = np.exp(-2j * math.pi * f * t[sel])
    gain = np.mean(y[sel] * ref) / np.mean(u[sel] * ref)
    expect = _h(f)
    assert abs(gain) == pytest.approx(abs(expect), rel=0.02)
    assert math.degrees(cmath.phase(gain / expect)) == pytest.approx(0.0, abs=2.0)


def test_its_output_held_and_switched_off_outside_its_voltage_window():
    pss = emt_controls.Ieeest({**P, 'LSMAX': 0.05, 'LSMIN': -0.05, 'VCU': 1.1, 'VCL': 0.9}, 0.0)
    assert pss.step(1e-3, 100.0, 1.0) == pytest.approx(0.05)
    assert pss.step(1e-3, 100.0, 1.2) == 0.0


def test_load_step_with_a_stabiliser_against_andes(client, quiet, monkeypatch):
    """EXDC2, TGOV1 and IEEEST on speed: the EMT's frequency and voltage as ANDES has them, and the stabiliser
    changing them (so that the comparison sees it)."""
    monkeypatch.setattr(emt_machine.ParkMachine, 'speed_voltage', False)
    runs = {}
    for pss in ('NONE', 'IEEEST'):
        request = T._request(duration_ms=C.T_END * 1e3, ac_fault_bus='a', ac_fault_type='abc', ac_fault_time_ms=C.T_STEP * 1e3,
                             ac_fault_duration_ms=0, ac_fault_resistance_ohm=T.KV ** 2 / C.STEP_MW)
        gen = next(v for v in request.values() if v.get('typ') == 'Generator0')
        gen.update(dyn_exciter_model='EXDC2', dyn_governor_model='TGOV1', dyn_pss_model=pss, dyn_pss_KS='20', dyn_pss_LSMAX='0.2',
                   dyn_pss_LSMIN='-0.2')
        emt, warnings = T._emt(client, quiet, request)
        (m,) = emt['machines']
        runs[pss] = (m, gen)
    m, gen = runs['IEEEST']
    assert m['pss_model'] == 'IEEEST'
    t_a, f_a, v_a = C._andes(emt_controls.parameters('exc', gen), emt_controls.parameters('gov', gen), emt_controls.parameters('pss', gen))
    t_e, f_e = np.array(m['trace']['t_ms']) / 1e3, np.array(m['trace']['f_hz'])
    v_e = np.array(m['trace']['v_pu'])
    t_0, v_0 = np.array(runs['NONE'][0]['trace']['t_ms']) / 1e3, np.array(runs['NONE'][0]['trace']['v_pu'])
    changed = max(abs(np.interp(x, t_e, v_e) - np.interp(x, t_0, v_0)) for x in (0.4, 0.6, 0.8, 1.0))
    assert changed > 0.005, changed
    for x in (0.3, 0.6, 1.0, 1.5, 2.0, 2.9):
        assert np.interp(x, t_e, f_e) == pytest.approx(np.interp(x, t_a, f_a), abs=0.003), x
        assert np.interp(x, t_e, v_e) == pytest.approx(np.interp(x, t_a, v_a), abs=0.004), x


def test_andes_study_takes_the_stabiliser():
    import andes_electrisim as ae
    assert {'MODE', 'KS', 'T5', 'T6', 'LSMAX'} <= set(ae._PSS_DEFAULTS['IEEEST'])
    model, params = emt_controls.parameters('pss', {'dyn_pss_model': 'IEEEST', 'dyn_pss_MODE': '7'})
    assert model == 'IEEEST' and params['MODE'] == 1
    assert emt_controls.parameters('pss', {'dyn_pss_model': 'NONE'}) is None
    data, notes = emt_machine.park_data({'dyn_pss_model': 'IEEEST', 'dyn_exciter_model': 'NONE'}, *_defaults())
    assert 'pss' not in data['controls'] and any('PSS' in n for n in notes)


def _defaults():
    from andes_electrisim import _DEFAULT_GENROU, _EXCITER_DEFAULTS, _GOVERNOR_DEFAULTS
    return _GOVERNOR_DEFAULTS, _EXCITER_DEFAULTS, _DEFAULT_GENROU
