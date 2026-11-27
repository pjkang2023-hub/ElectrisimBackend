"""
The eSTATCOM: a supercapacitor behind a grid-forming PCS, a STATCOM with an
energy store (GE Vernova's FACTSFLEX, for the 800 V AI factory reference).
A PCS refused a supercapacitor - one joined a DC bus only.

Two 0.4 kV buses, the grid on A and a 0.6 MW load on B; on B a 0.5 MVA
grid-forming PCS at no active power with Q-V droop (5 %), its supercapacitor
50 F at 800 V, charged to 90 %, usable down to half, at 60 Hz.
"""
import json

import numpy as np
import pytest

import andes_electrisim
from test_microgrid_ts import _ts
from test_pcs import LF, _der, _pcs, _two_buses

SC = dict(sizing='ratings', c_f=50, v_rated=800, p_rated_kw=500, v0_percent=90)
USABLE = (0.9 ** 2 - 0.5 ** 2) / (1 - 0.5 ** 2)     # its usable energy now, a share of it at its rated voltage


def _estatcom(control='grid_forming'):
    return [_der('Supercapacitor', 'sc', **SC),
            _pcs('est', 'b', 'sc', control=control, s_rated_mva=0.5, p_set_mw=0, q_mode='qv')]


def _post(client, quiet, request):
    with quiet():
        out = json.loads(client.post('/', json=request).get_data(as_text=True))
    assert not out.get('error'), out.get('message')
    return out


def _tds(**extra):
    return {'typ': 'TransientStabilityAndes Parameters', 'frequency': '60', 'sn_mva': '100', 'tf': '3', 'tstep': '0',
            'fault_enabled': 'false', 'fault_bus': '', 'toggle_line': '', 'toggle_gen': '', 'user_email': 't@t',
            **{k: str(v) for k, v in extra.items()}}


def test_the_load_flow_holds_its_bus_on_its_droop(client, quiet):
    """
    No active power; the Q that holds bus B on its Q-V droop line,
    V = 1 - 0.05 Q / 0.5; its supercapacitor at 720 V, 74.7 % of its usable
    energy - (0.9^2 - 0.5^2) / (1 - 0.5^2).
    """
    out = _post(client, quiet, _two_buses(*_estatcom(), params=dict(LF, frequency='60')))
    (pcs,) = out['pcs']
    assert pcs['p_mw'] == pytest.approx(0.0, abs=1e-6) and pcs['q_mvar'] > 0.1
    vb = next(b['vm_pu'] for b in out['busbars'] if b['name'] == 'b')
    assert vb == pytest.approx(1 - 0.05 * pcs['q_mvar'] / 0.5, abs=1e-4)
    (sc,) = out['ders']
    assert sc['v_cap_v'] == pytest.approx(720.0) and sc['soc_percent'] == pytest.approx(100 * USABLE)
    assert not any('left out' in w for w in out.get('warnings', []))


def test_andes_it_answers_a_dip_and_returns(client, quiet):
    """
    A 30 % grid dip for 0.5 s: its reactive power rises to its current limit,
    1.2 pu of its rating, which its virtual impedance holds it at from the
    next look (each step) - before, ANDES's virtual machine had none and went
    past it - and returns after. Its active power swings as the dip steps in
    and out (its store answering, as a grid-forming converter's does) and
    returns to none.
    """
    out = _post(client, quiet, _two_buses(*_estatcom(), params=_tds(
        grid_voltage_profile='custom', grid_voltage_table='0, 0.7; 0.5, 1.0', grid_voltage_start_s=1)))
    assert out['converged'] is True
    t = np.asarray(out['time'])
    (pcs,) = out['pcs']
    assert pcs['model'] == 'GENCLS'
    q, p, i = np.asarray(pcs['q_mvar']), np.asarray(pcs['p_mw']), np.asarray(pcs['current_pu'])
    q0 = np.interp(0.9, t, q)
    assert np.interp(1.3, t, q) > q0 + 0.1
    assert pcs['current_limit_pu'] == 1.2 and 1.0 < pcs['current_limited_from_s'] < 1.05
    held = (t > 1.1) & (t < 1.5)
    assert i[held].max() <= 1.2 * 1.03 and i[held].min() >= 1.2 * 0.97
    assert any("PCS 'EST' reached its current limit (1.2 pu of its rating)" in w for w in out['warnings'])
    assert np.interp(2.5, t, q) == pytest.approx(q0, abs=1e-3)
    assert np.interp(2.5, t, pcs['virtual_impedance_x']) == 1.0
    assert np.abs(p).max() < 0.5 and abs(np.interp(2.5, t, p)) < 1e-3


def test_andes_its_store_empties_and_it_stops(client, quiet):
    """
    Holding an island alone, it gives the load from its supercapacitor -
    0.5 x 50 F x (720^2 - 400^2) V^2, 8.96 MJ usable - until it is empty, then
    stops: before, its store never emptied. It empties when the DC power it
    gave, its AC power over its efficiency and its no-load loss, reaches its
    energy.
    """
    request = _two_buses(_der('Supercapacitor', 'sc', **SC),
                         _pcs('est', 'a', 'sc', control='grid_forming', s_rated_mva=1.0), grid=False,
                         params=_tds(tf=20))
    out = _post(client, quiet, request)
    (pcs,) = out['pcs']
    t = np.asarray(out['time'])
    stop = pcs['stopped_s']
    assert pcs['stopped_because'] == 'emptied' and stop is not None
    p = np.asarray(pcs['p_mw'])
    before = t <= stop
    p_dc = p[before] / 0.98                                    # its efficiency, 98 %; no no-load loss
    given = float(np.sum(0.5 * (p_dc[1:] + p_dc[:-1]) * np.diff(t[before]))) + p_dc[0] * t[0]
    assert given == pytest.approx(0.5 * 50 * (720 ** 2 - 400 ** 2) / 1e6, rel=0.01)
    assert pcs['soc_percent'][0] == pytest.approx(100 * USABLE, abs=0.5) and min(pcs['soc_percent']) == 0.0
    assert any(f"PCS 'EST': its supercapacitors emptied at t = {stop:.3f} s, and it stopped" in w
               for w in out['warnings'])


def test_andes_grid_following_is_a_store(quiet):
    """Grid-following, its PCS is ESD1 with its usable energy and its share of it now, as a flywheel's."""
    request = _two_buses(*_estatcom('grid_following'), params=_tds())
    with quiet():
        ss, meta = andes_electrisim.build_system(request, _tds())
    g = meta['gen_map']['est']
    k = list(ss.ESD1.idx.v).index(g['model_idx'])
    usable_mwh = 0.5 * 50 * (800 ** 2 - 400 ** 2) / 3.6e9
    assert float(ss.ESD1.En.vin[k]) == pytest.approx(usable_mwh)
    assert float(ss.ESD1.SOCinit.vin[k]) == pytest.approx(USABLE)


def test_emt_it_holds_the_bus_up_in_a_dip(client, quiet):
    """
    A 30 % grid dip for 50 ms at 60 Hz: bus B falls to 0.67 pu without it,
    some 0.73 pu with it, and it ends at the load flow's Q within its current limit.
    """
    emt = dict(typ='EmtStudy Parameters', user_email='t@t', frequency='60', time_step_us='20', duration_ms='150',
               grid_voltage_profile='custom', grid_voltage_table='0, 0.7; 0.05, 1.0', grid_voltage_start_ms='30')
    bare = _post(client, quiet, _two_buses(params=emt))['emt']
    held = _post(client, quiet, _two_buses(*_estatcom(), params=emt))
    v = lambda r: next(b['v_rms_min_pu'] for b in r['ac']['buses'] if b['label'] == 'B')
    assert v(held['emt']) > v(bare) + 0.04
    (conv,) = held['emt']['converters']
    lf = _post(client, quiet, _two_buses(*_estatcom(), params=dict(LF, frequency='60')))['pcs'][0]
    assert conv['control'] == 'grid_forming'
    assert conv['q_end_mvar'] == pytest.approx(lf['q_mvar'], rel=0.02)


def test_time_series_carries_it(client, quiet):
    """A short time series with it: it solves, its store reported."""
    out = _post(client, quiet, _two_buses(*_estatcom(), params=_ts(5, 1.0)))
    assert out['timeseries_converged']
