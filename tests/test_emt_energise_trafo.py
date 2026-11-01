"""
A transformer energised in the EMT study (ac_energise_trafo): closed onto a stiff 33 kV grid with its LV
winding open, its cores saturating past their knee from their remanence. The flux each core reaches is the
textbook's - its remanence plus twice its rated peak, closed at its voltage's zero; its rated peak alone, closed
at the voltage's crest - and only past the knee does the inrush come.
"""
import json

import pytest

from test_iec_neutral_resistor import _bus, _grid, _trafo


def _energise(client, quiet, **kw):
    params = {'typ': 'EmtStudy Parameters', 'user_email': 't@t', 'frequency': '50', 'time_step_us': '20',
              'duration_ms': '120', 'ac_energise_trafo': 'tx', 'ac_energise_time_ms': '20',
              **{k: str(v) for k, v in kw.items()}}
    rows = [_bus('hv', 33), _bus('lv', 0.69), _grid('hv'), _trafo('hv', 'lv', 33, 0.69, 2.5, 6, 0.5, 'Dyn11')]
    with quiet():
        out = json.loads(client.post('/', json={'0': params, **{str(k + 1): r for k, r in enumerate(rows)}})
                         .get_data(as_text=True))
    assert 'emt' in out, out.get('message')
    return out['emt']['energise_trafo']


def test_closed_at_one_windings_crest_the_others_are_60_degrees_off_theirs(client, quiet):
    """
    The three poles close together, at the crest of the first winding's voltage: its flux swings +-1 pu, no
    offset; the other two windings' are 60 degrees from their crests, and theirs reach 1 + sin 60 = 1.87 pu -
    past the knee, so even the best single closing instant leaves two cores saturating.
    """
    en = _energise(client, quiet, ac_energise_angle_deg=90, trafo_remanence_pu='0,0,0')
    assert en['psi_max_pu'][0] == pytest.approx(1.0, abs=0.03)
    assert en['psi_max_pu'][1] == pytest.approx(1.866, abs=0.05)
    assert en['psi_max_pu'][2] == pytest.approx(1.866, abs=0.05)
    assert en['saturated_ms'] > 0


def test_closed_at_the_zero_the_flux_doubles_and_remanence_adds(client, quiet):
    bare = _energise(client, quiet, ac_energise_angle_deg=0, trafo_remanence_pu='0,0,0')
    rem = _energise(client, quiet, ac_energise_angle_deg=0, trafo_remanence_pu='0.8,-0.4,-0.4')
    assert bare['psi_max_pu'][0] == pytest.approx(2.0, abs=0.08)
    assert rem['psi_max_pu'][0] == pytest.approx(2.8, abs=0.1)
    assert rem['saturated_ms'] > 50
    peak = lambda en: max(p['i_peak_ka'] for p in en['phases'])
    assert peak(rem) > 1.5 * peak(bare) > 0
    # an inrush is rich in the second harmonic: on its worst phase over 10 %, on some phase far more
    for en in (bare, rem):
        assert all(p['h2_ratio_min'] is None or p['h2_ratio_min'] > 0.1 for p in en['phases'])
        assert en['h2_ratio_cross_min'] > 0.3


def test_point_on_wave_sets_the_closing(client, quiet):
    """90 degrees later on the wave is a quarter cycle later."""
    at0 = _energise(client, quiet, ac_energise_angle_deg=0)
    at90 = _energise(client, quiet, ac_energise_angle_deg=90)
    assert at90['closed_ms'] - at0['closed_ms'] == pytest.approx(5.0, abs=0.1)
