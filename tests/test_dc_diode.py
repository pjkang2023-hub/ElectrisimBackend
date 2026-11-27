"""
The DC diode: a server shelf fed from two 800 V buses through a diode from
each takes its power from the higher - the GE Vernova / NVIDIA 800 V designs'
redundancy, a lineup and a catcher.

The radial grid's DC network (test_dc_elements) with a second rectifier
holding a catcher bus 2 % low, and a shelf bus with an 80 kW load fed through
D1 from DC bus A and D2 from the catcher, each 1.6 V forward and 0.1 mOhm.
"""
import json
import math

import pytest

import electrisim_sld as sld
from test_dc_elements import VSC, _breaker, _dc_element, _drawn_request, _with

V_F, R_ON, P_SHELF = 1.6, 1e-4, 0.08e6
V_CATCH = 0.98 * 800.0


def _el(key, **fields):
    return {'id': f'cell-{key}', 'name': key, 'userFriendlyName': fields.pop('label', key), **fields}


def _shelf(lineup_rectifier=True, d1=True, extra=(), load=None):
    req = _drawn_request()
    lv = _dc_element(req, 'vsc1')['bus']
    if not lineup_rectifier:
        _dc_element(req, 'vsc1')['in_service'] = 'false'
    diode = dict(v_f_v=str(V_F), r_on_mohm=str(R_ON * 1e3), rated_current_ka='0.2')
    return _with(req,
                 _el('catch', typ='DC Bus3', label='Catcher bus', vn_kv='0.8'),
                 _el('shelf', typ='DC Bus4', label='Shelf', vn_kv='0.8'),
                 _el('vsc2', typ='VSC1', label='Catcher rectifier', bus=lv, bus_dc='catch', in_service='true',
                     **{k: str(v) for k, v in dict(VSC, control_value_dc=0.98).items()}),
                 _el('ld_s', typ='Load DC2', label='Shelf load', bus='shelf', p_mw=str(P_SHELF / 1e6), **(load or {})),
                 _el('d1', typ='DC Diode0', label='D1', busFrom='dc_a', busTo='shelf',
                     in_service='true' if d1 else 'false', **diode),
                 _el('d2', typ='DC Diode1', label='D2', busFrom='catch', busTo='shelf', **diode),
                 *extra)


def _post(client, quiet, request):
    with quiet():
        out = json.loads(client.post('/', json=request).get_data(as_text=True))
    assert not out.get('error'), out.get('message')
    return out


def _shelf_volts(v_source):
    """The shelf's voltage behind a diode from v_source: V = v_source - v_f - r_on P / V."""
    a = v_source - V_F
    return (a + math.sqrt(a * a - 4 * R_ON * P_SHELF)) / 2


def _volts(out):
    return {b['id']: b['vm_pu'] * 800.0 if b.get('vm_pu') is not None else None for b in out['dcbuses']}


def test_the_higher_bus_carries_the_shelf(client, quiet):
    """
    Both rectifiers on: D1 carries the shelf from DC bus A's 800 V, its drop
    exactly v_f + r_on I at the current the load draws there; D2 blocks,
    the catcher bus 2 % lower. Neither diode is a DC cable in the results.
    """
    out = _post(client, quiet, _shelf())
    v = _volts(out)
    want = _shelf_volts(v['cell-dc_a'])
    assert v['cell-shelf'] == pytest.approx(want, abs=1e-3)
    d1, d2 = sorted(out['dcdiodes'], key=lambda d: d['name'])
    amps = P_SHELF / want
    assert d1['conducting'] and d1['i_ka'] == pytest.approx(amps / 1e3, rel=1e-5)
    assert d1['v_ak_v'] == pytest.approx(V_F + R_ON * amps, abs=1e-3)
    assert d1['loss_kw'] == pytest.approx((V_F + R_ON * amps) * amps / 1e3, rel=1e-4)
    assert d1['loading_percent'] == pytest.approx(100 * amps / 200.0, rel=1e-5)
    assert not d2['conducting'] and d2['i_ka'] == 0.0
    assert d2['v_ak_v'] == pytest.approx(V_CATCH - want, abs=1e-3)
    assert [l['id'] for l in out['linedcs']] == ['cell-cable']


def test_the_catcher_picks_up_a_lost_lineup(client, quiet):
    """
    The lineup's rectifier out: DC bus A has no supply, D1 blocks, and D2
    carries the shelf from the catcher's 784 V. DC buses A and B, which only a
    diode would have fed, are named as unsupplied.
    """
    out = _post(client, quiet, _shelf(lineup_rectifier=False))
    v = _volts(out)
    assert v['cell-shelf'] == pytest.approx(_shelf_volts(V_CATCH), abs=1e-3)
    d1, d2 = sorted(out['dcdiodes'], key=lambda d: d['name'])
    assert not d1['conducting'] and d2['conducting']
    assert d2['i_ka'] == pytest.approx(P_SHELF / _shelf_volts(V_CATCH) / 1e3, rel=1e-5)
    assert any('DC bus A' in w and 'no supply' in w for w in out['warnings'])


def test_a_diode_behind_an_open_breaker(client, quiet):
    """
    The shelf feeder's breaker in front of D1 open: D1 is out, though DC bus A
    is live and 15 V above the catcher, and the catcher carries the shelf.
    Closed, the breaker carries D1's current.
    """
    opened = _post(client, quiet, _shelf(extra=[_breaker('qf', 'dc_a', 'd1', 'line_dc', closed=False)]))
    d1, d2 = sorted(opened['dcdiodes'], key=lambda d: d['name'])
    assert not d1['in_service'] and not d1['conducting'] and d2['conducting']
    assert _volts(opened)['cell-shelf'] == pytest.approx(_shelf_volts(V_CATCH), abs=1e-3)

    closed = _post(client, quiet, _shelf(extra=[_breaker('qf', 'dc_a', 'd1', 'line_dc', rated_current_ka=0.2)]))
    (qf,) = closed['dcbreakers']
    d1 = next(d for d in closed['dcdiodes'] if d['name'] == 'd1')
    assert d1['conducting'] and qf['i_ka'] == pytest.approx(d1['i_ka'], rel=1e-6)


def test_a_diode_fitted_backwards_blocks(client, quiet):
    """A shelf whose only diode points into DC bus A has no supply."""
    req = _drawn_request()
    req = _with(req, _el('shelf', typ='DC Bus3', label='Shelf', vn_kv='0.8'),
                _el('ld_s', typ='Load DC2', label='Shelf load', bus='shelf', p_mw='0.08'),
                _el('d1', typ='DC Diode0', label='D1', busFrom='shelf', busTo='dc_a'))
    out = _post(client, quiet, req)
    (d1,) = out['dcdiodes']
    assert not d1['conducting'] and _volts(out)['cell-shelf'] is None
    assert any('Shelf' in w and 'no supply' in w for w in out['warnings'])


def test_a_fault_on_the_lineup_bus(client, quiet):
    """
    A 10 mOhm fault on DC bus A. In the DC fault study (loads leave at the
    fault) D1 stops at once - no backfeed from the shelf into the fault - and
    D2, its shelf drawing nothing, stays off. In EMT the shelf's load stays:
    D1 stops and D2 takes the shelf from the catcher, one forward drop below
    it. Here both rectifiers hang on the one 0.4 kV bus, which the faulted
    rectifier's diodes drag down: the catcher's rectifier blocks and its bus
    sags to some 380 V (the paper gives the catcher a transformer of its own).
    """
    fault = _shelf(load={'filter_c_uf': '2000'})
    fault['0'] = {'typ': 'DcFaultStudy Parameters', 'fault_bus': 'dc_a', 'duration_ms': '60',
                  'fault_resistance_mohm': '10', 'user_email': 't@t'}
    (f,) = _post(client, quiet, fault)['dcfault']['faults']
    d1, d2 = sorted(f['diodes'], key=lambda d: d['name'])
    assert d1['i_prefault_ka'] == pytest.approx(P_SHELF / _shelf_volts(800.0) / 1e3, rel=1e-3)
    assert abs(d1['i_end_ka']) < 1e-4 and not d1['conducting_end']
    assert d2['ip_ka'] < 1e-4

    emt = _shelf(load={'filter_c_uf': '2000'})
    emt['0'] = {'typ': 'EmtStudy Parameters', 'time_step_us': '20', 'duration_ms': '120', 'fault_bus': 'dc_a',
                'fault_time_ms': '10', 'fault_resistance_mohm': '10', 'user_email': 't@t'}
    result = _post(client, quiet, emt)['emt']
    branches = {b['label']: b for b in result['branches'] if b['kind'] == 'DC diode'}
    v_end = {b['label']: b['waveform']['v_kv'][-1] * 1e3 for b in result['buses']}
    i1, i2 = branches['D1']['waveform']['i_ka'], branches['D2']['waveform']['i_ka']
    assert i1[0] == pytest.approx(P_SHELF / _shelf_volts(800.0) / 1e3, rel=1e-2)
    assert abs(i1[-1]) < 1e-3 and i2[0] == 0.0 and i2[-1] > 0.05
    assert v_end['Catcher bus'] - v_end['Shelf'] == pytest.approx(V_F + R_ON * i2[-1] * 1e3, abs=0.02)


def test_a_spec_with_diodes(client):
    """A spec's dc_diodes: built, solved and reported under their ids; one between AC buses refused."""
    spec = {
        'name': 'diode OR-ing', 'frequency_hz': 60,
        'buses': [{'id': 'MV', 'vn_kv': 34.5}, {'id': 'LV', 'vn_kv': 0.48}],
        'external_grids': [{'id': 'G', 'bus': 'MV', 's_sc_max_mva': 500, 'rx_max': 0.1}],
        'transformers': [{'id': 'T', 'hv_bus': 'MV', 'lv_bus': 'LV', 'sn_mva': 2.0, 'vn_hv_kv': 34.5,
                          'vn_lv_kv': 0.48, 'vk_percent': 6, 'vkr_percent': 1}],
        'dc_buses': [{'id': 'A', 'vn_kv': 0.8}, {'id': 'C', 'vn_kv': 0.8}, {'id': 'S', 'vn_kv': 0.8}],
        'vscs': [{'id': 'RA', 'bus': 'LV', 'bus_dc': 'A', 'control_mode_ac': 'q_mvar', 'control_value_ac': 0,
                  'control_mode_dc': 'vm_pu', 'control_value_dc': 1.0},
                 {'id': 'RC', 'bus': 'LV', 'bus_dc': 'C', 'control_mode_ac': 'q_mvar', 'control_value_ac': 0,
                  'control_mode_dc': 'vm_pu', 'control_value_dc': 0.98}],
        'dc_loads': [{'id': 'L', 'bus': 'S', 'p_mw': 0.08}],
        'dc_diodes': [{'id': 'D1', 'from_bus': 'A', 'to_bus': 'S', 'v_f_v': 1.6, 'r_on_mohm': 0.1},
                      {'id': 'D2', 'from_bus': 'C', 'to_bus': 'S', 'v_f_v': 1.6, 'r_on_mohm': 0.1}],
    }
    net, report = sld.build_network(spec)
    flow = sld.solve(net)
    diodes = {d['id']: d for d in flow['dc_diodes']}
    assert diodes['D1']['conducting'] and not diodes['D2']['conducting']
    assert diodes['D1']['v_ak_v'] == pytest.approx(1.6 + 1e-4 * diodes['D1']['i_ka'] * 1e3, abs=1e-3)
    assert {l['id'] for l in flow.get('dc_lines', [])} == set()

    bad = {**spec, 'dc_diodes': [{'id': 'D1', 'from_bus': 'MV', 'to_bus': 'S'}]}
    with pytest.raises(sld.SpecError) as err:
        sld.build_network(bad)
    assert any('D1' in p for p in err.value.problems)
