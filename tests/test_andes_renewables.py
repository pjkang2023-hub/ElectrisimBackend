"""
Static Generators in the ANDES transient stability study: an inverter-based
plant (IBR) as REGCA1 + REECA1 + REPCA1, a wind plant (WIND) as those plus
WTDTA1, WTARA1, WTPTA1 and WTTQA1, each given the mode flags ANDES 2.0
requires, starting where the power flow left it.
"""
import numpy as np
import pytest

import andes_electrisim
from test_dc_elements import _with
from test_pcs import LINE, _bus, _post, _two_buses

TDS = {'typ': 'TransientStabilityAndes Parameters', 'frequency': '50', 'sn_mva': '100', 'user_email': 't@t'}
IBR_MODELS = ('REGCA1', 'REECA1', 'REPCA1')
WIND_MODELS = IBR_MODELS + ('WTDTA1', 'WTARA1', 'WTPTA1', 'WTTQA1')


def _sgen(name, bus, kind, sn_mva=0.5, **fields):
    return {'typ': 'Static Generator0', 'name': name, 'id': f'cell-{name}', 'userFriendlyName': name.upper(),
            'bus': bus, 'p_mw': '0.2', 'q_mvar': '0', 'sn_mva': str(sn_mva), 'scaling': '1', 'in_service': 'true',
            'dyn_plant_kind': kind, **{k: str(v) for k, v in fields.items()}}


def _param(ss, model, name, idx):
    mdl = getattr(ss, model)
    return float(getattr(mdl, name).v[list(mdl.idx.v).index(idx)])


def _ibr_and_wind(params):
    """The two 0.4 kV buses and a third, C, 100 m on from B; an IBR plant on B, a wind plant on C."""
    request = _two_buses(_bus('c', 0.4), dict(LINE, name='l2', id='cell-l2', userFriendlyName='L2',
                                              busFrom='b', busTo='c'), params=params)
    return _with(request, _sgen('ibr', 'b', 'IBR'), _sgen('wind', 'c', 'WIND'))


def test_ibr_and_wind_plants_have_their_models(quiet):
    with quiet():
        ss, meta = andes_electrisim.build_system(_ibr_and_wind(TDS), TDS)
    assert not any('could not add' in d for d in meta['defaults_applied']), meta['defaults_applied']
    for model in WIND_MODELS:
        assert getattr(ss, model).n == (2 if model in IBR_MODELS else 1), model
    g = meta['gen_map']
    assert all(g['ibr'][k] for k in ('reg_idx', 'ree_idx', 'repca_idx'))
    assert all(g['wind'][k] for k in ('reg_idx', 'ree_idx', 'repca_idx', 'wtdta_idx', 'wtara_idx', 'wtpta_idx',
                                      'wttqa_idx'))
    pflag = {idx: int(v) for idx, v in zip(ss.REECA1.idx.v, ss.REECA1.PFLAG.v)}
    # A wind plant's power order follows its rotor speed; an IBR plant's does not.
    assert pflag == {g['ibr']['ree_idx']: 0, g['wind']['ree_idx']: 1}


def test_ibr_and_wind_plants_start_steady(client, quiet):
    """No event: the plants start from the power flow and the bus voltages stay where they began."""
    result = _post(client, quiet, _ibr_and_wind({**TDS, 'tf': '2'}))
    assert result['converged'] is True
    assert not any('could not be initialised' in w for w in result['warnings']), result['warnings']
    assert not any('could not add' in d for d in result['defaults_applied']), result['defaults_applied']
    for series in result['bus_voltage']:
        v = np.asarray(series['values'], dtype=float)
        assert np.ptp(v) < 1e-4, series['name']


def test_a_rejected_model_leaves_no_partial_device(quiet):
    """REECA1 without its mandatory flags is refused and taken back out, so the study can still start."""
    request = _two_buses(_sgen('ibr', 'b', 'IBR'), params=TDS)
    with quiet():
        ss, meta = andes_electrisim.build_system(request, TDS, setup=False)
    n = ss.REECA1.n
    lengths = {k: len(p.v) for k, p in ss.REECA1.params.items()}
    notes = []
    assert andes_electrisim._add_model_safe(ss, 'REECA1', notes, 'test', idx='REECA1_bad',
                                            reg=meta['gen_map']['ibr']['reg_idx']) is None
    assert 'could not add REECA1' in notes[0]
    assert ss.REECA1.n == n and 'REECA1_bad' not in ss.REECA1.uid
    assert {k: len(p.v) for k, p in ss.REECA1.params.items()} == lengths
    assert andes_electrisim._add_model_safe(ss, 'NoSuchModel', notes, 'test', idx='x') is None
    assert 'no NoSuchModel' in notes[1]


def test_the_dialogs_plant_controller_gain_and_dg_lag_reach_the_models(quiet):
    """
    REPCA1 Kp from the dialog; DG Tg as both of PVD1's and ESD1's current lags,
    tip and tiq. Left blank, each keeps the ANDES model's default.
    """
    request = _two_buses(_bus('c', 0.4), dict(LINE, name='l2', id='cell-l2', userFriendlyName='L2',
                                              busFrom='b', busTo='c'), params=TDS)
    request = _with(request, _sgen('ibr', 'b', 'IBR', dyn_repca_Kp=2.5), _sgen('pv', 'c', 'PVD1', 1.5, dyn_dg_Tg=0.05),
                    _sgen('es', 'a', 'ESD1', 1.5, dyn_dg_Tg=0.08), _sgen('bare', 'c', 'PVD1', 1.5))
    with quiet():
        ss, meta = andes_electrisim.build_system(request, TDS)
    g = meta['gen_map']
    assert _param(ss, 'REPCA1', 'Kp', g['ibr']['repca_idx']) == pytest.approx(2.5)
    for name, tg in (('pv', 0.05), ('es', 0.08), ('bare', 0.02)):
        model = g[name]['plant_kind']
        assert _param(ss, model, 'tip', g[name]['dg_idx']) == pytest.approx(tg), name
        assert _param(ss, model, 'tiq', g[name]['dg_idx']) == pytest.approx(tg), name


def test_a_dg_lag_of_zero_keeps_the_model_default(quiet):
    request = _two_buses(_sgen('pv', 'b', 'PVD1', 1.5, dyn_dg_Tg=0), params=TDS)
    with quiet():
        ss, meta = andes_electrisim.build_system(request, TDS)
    assert _param(ss, 'PVD1', 'tip', meta['gen_map']['pv']['dg_idx']) == pytest.approx(0.02)
    assert any('DG Tg must be positive' in d for d in meta['defaults_applied']), meta['defaults_applied']


def test_a_pv_plant_with_its_lag_set_starts_steady(client, quiet):
    result = _post(client, quiet, _two_buses(_sgen('pv', 'b', 'PVD1', 1.5, dyn_dg_Tg=0.05), params={**TDS, 'tf': '1'}))
    assert result['converged'] is True
    assert not any('could not be initialised' in w for w in result['warnings']), result['warnings']
