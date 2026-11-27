"""
Load power profiles: parsing and reporting, and loads following them in the
time series and transient stability (ANDES) studies.
"""
import json
import os

import numpy as np
import pandapower as pp
import pytest

import electrisim_sld as sld
import load_profiles_electrisim as lp

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE_DIR = os.path.join(HERE, 'reference')

# A 5 s training cycle sampled every 0.1 s: 1 s idle at 0.3 p.u., 4 s at 1.0.
CYCLE_DT = 0.1
CYCLE = [0.3] * 10 + [1.0] * 40


def _fine_mean(t, p, start, end, repeat):
    """The profile's mean over [start, end] from a 0.1 ms sampling of it."""
    x = np.linspace(start, end, int(round((end - start) / 1e-4)) + 1)
    return float(np.trapezoid(lp.sample_profile(t, p, x, repeat), x) / (end - start))


# --- parsing and the upload report ------------------------------------------------

def test_parse_two_columns_with_header_and_any_delimiter():
    for sep in (',', ';', '\t', ' '):
        t, p, notes = lp.parse_profile(f'time_s{sep}p_pu\n0{sep}0.5\n0.1{sep}0.9\n0.2{sep}1.0\n')
        assert list(t) == [0, 0.1, 0.2] and list(p) == [0.5, 0.9, 1.0]
        assert notes == ['1 non-numeric line(s) skipped (headers or comments).']


def test_parse_one_column_needs_its_time_step():
    with pytest.raises(ValueError, match='time step'):
        lp.parse_profile('0.5\n0.9\n1.0')
    t, p, _ = lp.parse_profile('0.5\n0.9\n1.0', time_step_s=0.01)
    assert t == pytest.approx([0, 0.01, 0.02]) and list(p) == [0.5, 0.9, 1.0]


@pytest.mark.parametrize('text, message', [
    ('a,b\nc,d', 'No numeric rows'),
    ('0,1\n0,1', 'Time must increase'),
    ('0,1', 'at least two samples'),
])
def test_parse_rejects_what_cannot_be_a_profile(text, message):
    with pytest.raises(ValueError, match=message):
        lp.parse_profile(text)


def test_profile_report():
    """
    0.9 p.u. for 2 s, a 0.6 p.u. drop to idle in one sample, a 0.25 s ramp to
    1.05 and back to 0.9, and 0.5 s at zero: each shows in the report.
    """
    t = np.round(np.arange(0, 6.0001, 0.01), 2)
    p = np.where(t < 2, 0.9, 0.3)
    ramp = (t >= 3) & (t < 3.25)
    p[ramp] = 0.3 + 0.75 * (t[ramp] - 3) / 0.25
    p[(t >= 3.25) & (t < 5)] = 1.05
    p[(t >= 5) & (t < 5.5)] = 0.0
    p[t >= 5.5] = 0.9
    r = lp.analyse_profile(t, p)
    assert r['samples'] == 601 and r['duration_s'] == pytest.approx(6.0)
    assert r['time_step_s'] == pytest.approx(0.01) and r['fastest_content_hz'] == pytest.approx(50.0)
    assert (r['min_pu'], r['max_pu']) == (0.0, 1.05)
    assert r['mean_pu'] == pytest.approx(np.trapezoid(p, t) / 6.0)
    assert r['max_rise_pu_per_s'] == pytest.approx(90.0)     # 0 -> 0.9 in one 10 ms sample
    assert r['max_fall_pu_per_s'] == pytest.approx(-105.0)   # 1.05 -> 0
    # The ramp moves 0.03 p.u. a sample: below the 0.2 p.u. step threshold.
    assert [(s['time_s'], round(s['change_pu'], 6)) for s in r['steps']] == [(2.0, -0.6), (5.0, -1.05), (5.5, 0.9)]
    assert any('Exceeds 1.0 p.u.' in f and '1.050' in f for f in r['flags'])
    assert any('At zero for 0.50 s' in f for f in r['flags'])
    assert any('3 single-sample step(s)' in f and '-1.050 p.u. at 5.00 s' in f for f in r['flags'])


def test_repeating_profile_lasts_its_samples_times_the_step():
    """N samples dt apart repeat every N x dt, wrapping back to the first one step after the last."""
    t = np.arange(len(CYCLE)) * CYCLE_DT
    assert lp.period_s(t) == pytest.approx(5.0)
    assert lp.sample_profile(t, CYCLE, [5.0, 5.5, 7.2, 10.95], repeat=True) == pytest.approx([0.3, 0.3, 1.0, 0.65])
    assert lp.sample_profile(t, CYCLE, [6.0], repeat=False) == pytest.approx([1.0])


@pytest.mark.parametrize('start, end', [(0.0, 5.0), (5.0, 10.0), (0.0, 15.0), (0.5, 1.0), (4.5, 5.0), (3.3, 11.7)])
def test_profile_mean_over_a_step(start, end):
    """The mean over a study step equals the integral of the interpolated profile."""
    t = np.arange(len(CYCLE)) * CYCLE_DT
    assert lp.average_profile(t, CYCLE, start, end, repeat=True) == pytest.approx(
        _fine_mean(t, CYCLE, start, end, True), abs=1e-6)
    if start == 0.0 and end == 5.0:
        # By hand: 0.9 s at 0.3, two 0.1 s ramps averaging 0.65, 3.9 s at 1.0.
        assert lp.average_profile(t, CYCLE, start, end, repeat=True) == pytest.approx(4.3 / 5)


# --- the time series study ------------------------------------------------------------

TS_BATTERY_MW = {'reference_radial': -0.5, 'reference_transmission': 0.1}
PROFILED_LOAD = {'reference_radial': 'Factory', 'reference_transmission': None}


def _time_series(client, quiet, grid, steps, step_s, q_mode='pf', profile=None):
    with open(os.path.join(REFERENCE_DIR, f'{grid}.diagram_timeseries_payload.json'), encoding='utf-8') as handle:
        request = json.load(handle)
    loads = [v for v in request.values() if isinstance(v, dict) and str(v.get('typ', '')).startswith('Load')]
    load = next(v for v in loads if v['userFriendlyName'] == (PROFILED_LOAD[grid] or loads[0]['userFriendlyName']))
    load.update(load_profile_id='cycle', load_profile_q_mode=q_mode)
    request['0'].update(time_steps=steps, time_step_s=step_s,
                        load_profiles={'cycle': profile or {'name': 'Training cycle', 'dt': CYCLE_DT, 'p': CYCLE}})
    with quiet():
        response = client.post('/', json=request)
    assert response.status_code == 200
    result = json.loads(response.get_data(as_text=True))
    assert not result.get('error'), result.get('message')
    return result, load


@pytest.mark.parametrize('grid', ['reference_radial', 'reference_transmission'])
@pytest.mark.parametrize('step_s, q_mode', [(0.1, 'pf'), (0.7, 'pf'), (0.7, 'constant')])
def test_time_series_load_follows_a_library_profile(client, quiet, grid, step_s, q_mode):
    """
    A load following the training cycle draws its drawn P times the profile:
    at each step's instant when the steps are as fine as the profile, its mean
    over the step when they are coarser. Every step gives the power flow
    pandapower gives the spec with the same powers; Q follows at constant
    power factor or stays as drawn.
    """
    steps = 16
    result, load = _time_series(client, quiet, grid, steps, step_s, q_mode)
    assert result['timeseries_converged'] is True and result['time_step_s'] == step_s
    used = result['profiles_used'][load['name']]
    assert (used['library_profile'], used['q_mode']) == ('Training cycle', q_mode)
    t = np.arange(len(CYCLE)) * CYCLE_DT
    if step_s > 2 * CYCLE_DT:
        assert used['sampling'] == 'average'
        expected = [_fine_mean(t, CYCLE, k * step_s, (k + 1) * step_s, True) for k in range(steps)]
    else:
        assert used['sampling'] == 'instant'
        expected = list(lp.sample_profile(t, CYCLE, np.arange(steps) * step_s, True))
    assert used['values'] == pytest.approx(expected, abs=1e-6)
    assert any('repeats through the run' in n for n in result['notes']) == (steps * step_s > 5.0)

    profiles = {p['display_name']: p for p in result['profiles_used'].values()}
    vm = {(b['time_step'], b['name']): b['vm_pu'] for b in result['busbars']}
    net, _ = sld.build_network(sld_spec(grid))
    base = {table: net[table].copy() for table in ('load', 'sgen', 'gen')}
    for k in range(steps):
        for table in ('load', 'sgen', 'gen'):
            for idx in net[table].index:
                name = net[table].at[idx, 'name']
                if name not in profiles:
                    continue
                prof, p0 = profiles[name], base[table].at[idx, 'p_mw']
                q0 = base[table].at[idx, 'q_mvar'] if 'q_mvar' in net[table] else 0.0
                v = prof['values'][k]
                if prof.get('library_profile'):
                    net[table].at[idx, 'p_mw'] = p0 * v
                    net[table].at[idx, 'q_mvar'] = q0 * (v if q_mode == 'pf' else 1.0)
                else:                                  # the fixture's MW profiles
                    net[table].at[idx, 'p_mw'] = v
                    if 'q_mvar' in net[table] and p0:
                        net[table].at[idx, 'q_mvar'] = q0 * v / p0
        # Seconds-long steps hardly move the battery's charge: it holds its power.
        net.storage['p_mw'] = TS_BATTERY_MW[grid]
        pp.runpp(net, algorithm='nr', calculate_voltage_angles='auto')
        for idx in net.bus.index:
            assert vm[(k, net.bus.at[idx, 'name'])] == pytest.approx(net.res_bus.at[idx, 'vm_pu'], abs=1e-6), \
                (k, net.bus.at[idx, 'name'])


def sld_spec(grid):
    with open(os.path.join(REFERENCE_DIR, f'{grid}.spec.json'), encoding='utf-8') as handle:
        return json.load(handle)


def test_time_series_runs_past_24_steps(client, quiet):
    """Hourly steps were time-stamped with datetime(2024, 1, 1, hour=h), which raised from hour 24 on."""
    with open(os.path.join(REFERENCE_DIR, 'reference_radial.diagram_timeseries_payload.json'), encoding='utf-8') as handle:
        request = json.load(handle)
    request['0']['time_steps'] = 30
    with quiet():
        response = client.post('/', json=request)
    result = json.loads(response.get_data(as_text=True))
    assert not result.get('error'), result.get('message')
    assert result['time_steps'] == 30 and result['time_stamps'][24] == '2024-01-02 00:00:00'


def test_time_series_reports_a_missing_library_profile(client, quiet):
    with open(os.path.join(REFERENCE_DIR, 'reference_radial.diagram_timeseries_payload.json'), encoding='utf-8') as handle:
        request = json.load(handle)
    load = next(v for v in request.values() if isinstance(v, dict) and v.get('userFriendlyName') == 'Factory')
    load['load_profile_id'] = 'gone'
    request['0'].update(time_steps=3, load_profiles={'other': {'name': 'Other', 'dt': 1, 'p': [1, 1]}})
    with quiet():
        response = client.post('/', json=request)
    result = json.loads(response.get_data(as_text=True))
    assert 'Factory: its load profile is not in the library, so it does not follow one.' in result['notes']


# --- the transient stability study (ANDES) ---------------------------------------------

STEP_PROFILE = {'name': 'Half from 1 s to 3 s', 'dt': 0.5,
                'p': [1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]}


def _andes_request(profile=True, tf=4.0):
    with open(os.path.join(REFERENCE_DIR, 'reference_radial.diagram_sc_payload.json'), encoding='utf-8') as handle:
        payload = json.load(handle)
    network = {k: v for k, v in payload.items() if 'Parameters' not in str(v.get('typ'))}
    params = {'typ': 'TransientStabilityAndes Parameters', 'frequency': '50', 'sn_mva': '100',
              'tf': str(tf), 'fault_enabled': False, 'max_points': 100000}
    if profile:
        load = next(v for v in network.values() if v.get('userFriendlyName') == 'Factory')
        load.update(load_profile_id='step', load_profile_q_mode='pf')
        params['load_profiles'] = {'step': STEP_PROFILE}
    return {'0': params, **network}


def test_transient_stability_load_follows_a_library_profile(client, quiet):
    """
    The Factory load follows its profile through an ANDES run - run in pieces
    with its power reset before each - exactly as ANDES's own Alter events
    setting the same powers at the same times make it.
    """
    andes = pytest.importorskip('andes')
    import andes_electrisim as ae
    request = _andes_request()
    with quiet():
        response = client.post('/', json=request)
    result = json.loads(response.get_data(as_text=True))
    assert not result.get('error') and result['converged'], result.get('message')
    (followed,) = result['load_profiles']
    assert (followed['load'], followed['profile'], followed['q_mode']) == ('Factory', STEP_PROFILE['name'], 'pf')
    t = np.asarray(result['time'])
    prof_t = np.arange(len(STEP_PROFILE['p'])) * STEP_PROFILE['dt']
    assert followed['p_mw'] == pytest.approx(2.0 * lp.sample_profile(prof_t, STEP_PROFILE['p'], t, True), abs=1e-9)

    # The same powers through Alter events, piece by piece (0.5 s pieces, each at its mean).
    network = {k: v for k, v in _andes_request(profile=False).items() if k != '0'}
    with quiet():
        ss, meta = ae.build_system(network, {'frequency': '50', 'sn_mva': '100'}, setup=False)
    factory = next(i for i, name in zip(ss.PQ.idx.v, ss.PQ.name.v) if name == 'Factory')
    pieces = [(t0, lp.average_profile(prof_t, STEP_PROFILE['p'], t0, t0 + 0.5, True)) for t0 in np.arange(0, 4.0, 0.5)]
    pieces = [(t0, f) for k, (t0, f) in enumerate(pieces) if k and abs(f - pieces[k - 1][1]) > 1e-12]
    for t0, _ in pieces:
        for src in ('Ppf', 'Ipeq', 'Req', 'Qpf', 'Iqeq', 'Xeq'):
            ss.add('Alter', dict(t=float(t0), model='PQ', dev=factory, src=src, attr='v', method='=', amount=0.0))
    ss.setup()
    ss.TDS.config.noprint = True
    ss.TDS.config.no_tqdm = True
    assert ss.PFlow.run()
    ss.TDS.config.tf = 4.0
    ss.TDS.init()
    i = list(ss.PQ.idx.v).index(factory)
    v0, p_rated, q_rated = float(ss.PQ.v.v[i]), float(ss.PQ.p0.v[i]), float(ss.PQ.q0.v[i])
    k = 0
    for _, f in pieces:
        for src, base, power in (('Ppf', p_rated, 0), ('Ipeq', p_rated, 1), ('Req', p_rated, 2),
                                 ('Qpf', q_rated, 0), ('Iqeq', q_rated, 1), ('Xeq', q_rated, 2)):
            ss.Alter.amount.v[k] = base * f / v0 ** power
            k += 1
    assert ss.TDS.run()
    ref_t = np.asarray(ss.dae.ts.t)
    names = meta['bus_name_by_idx']
    midpoints = np.arange(0.25, 4.0, 0.5)
    for bus in result['bus_voltage']:
        a = next(j for j, idx in enumerate(ss.Bus.idx.v) if names.get(str(idx)) == bus['name'])
        ref = np.interp(midpoints, ref_t, ss.dae.ts.y[:, ss.Bus.v.a[a]])
        got = np.interp(midpoints, t, np.asarray(bus['values'], dtype=float))
        assert got == pytest.approx(ref, abs=1e-5), bus['name']
    # And the load's drop shows: B1 rises while the Factory draws half.
    b1 = np.asarray(next(b for b in result['bus_voltage'] if b['name'] == 'B1')['values'], dtype=float)
    assert np.interp(2.0, t, b1) > np.interp(0.5, t, b1) + 1e-3


def test_transient_stability_without_profiles_is_unchanged(client, quiet):
    """A library no load uses leaves the run as it was."""
    pytest.importorskip('andes')
    plain = _andes_request(profile=False)
    unused = _andes_request(profile=False)
    unused['0']['load_profiles'] = {'step': STEP_PROFILE}
    out = []
    for request in (plain, unused):
        with quiet():
            response = client.post('/', json=request)
        out.append(json.loads(response.get_data(as_text=True)))
    assert out[1]['load_profiles'] == []
    for a, b in zip(out[0]['bus_voltage'], out[1]['bus_voltage']):
        assert a['values'] == pytest.approx(b['values'], abs=1e-12)


# --- the browser's copy of the parser and report ---------------------------------------

FRONTEND_LIBRARY = os.path.join(HERE, '..', '..', 'frontend', 'src', 'main', 'webapp', 'js', 'electrisim',
                                'utils', 'loadProfileLibrary.js')


def _run_frontend(script_body, tmp_path):
    """Run Node on the browser's loadProfileLibrary.js (as an ES module) with a script appended."""
    import shutil
    import subprocess
    node = shutil.which('node')
    if not node or not os.path.exists(FRONTEND_LIBRARY):
        pytest.skip('needs Node and the frontend checkout beside the backend')
    with open(FRONTEND_LIBRARY, encoding='utf-8') as handle:
        source = handle.read()
    module = tmp_path / 'library.mjs'
    module.write_text(source + '\n' + script_body, encoding='utf-8')
    out = subprocess.run([node, str(module)], capture_output=True, text=True, timeout=60, check=True)
    return json.loads(out.stdout)


_STEPPED = [0.9] * 200 + [0.3] * 100 + [1.05] * 150 + [0.0] * 50 + [0.9] * 101
_STEPPED_CSV = 'time,p\n' + '\n'.join(f'{k * 0.01:.2f},{v}' for k, v in enumerate(_STEPPED))


@pytest.mark.parametrize('text, step', [
    (_STEPPED_CSV, None),
    ('0;0.2\n0.5;0.6\n0.75;1.2\n2.0;-0.1\n2.1;0.5', None),
    ('# one column\n' + '\n'.join(str(v) for v in CYCLE), 0.1),
])
def test_browser_report_matches_the_backend(text, step, tmp_path):
    """The library dialog's upload report is the one the backend would give."""
    browser = _run_frontend(
        f'const r = parseProfile({json.dumps(text)}, {json.dumps(step)});\n'
        'console.log(JSON.stringify({ t: r.t, p: r.p, notes: r.notes, report: analyseProfile(r.t, r.p) }));', tmp_path)
    t, p, notes = lp.parse_profile(text, step)
    assert browser['t'] == pytest.approx(list(t)) and browser['p'] == pytest.approx(list(p))
    assert browser['notes'] == notes
    report = lp.analyse_profile(t, p)
    for key in ('samples', 'regular_sampling', 'step_count'):
        assert browser['report'][key] == report[key], key
    for key in ('duration_s', 'time_step_s', 'fastest_content_hz', 'min_pu', 'mean_pu', 'max_pu',
                'max_rise_pu_per_s', 'max_fall_pu_per_s'):
        assert browser['report'][key] == pytest.approx(report[key], rel=1e-12), key
    assert [s['time_s'] for s in browser['report']['steps']] == pytest.approx([s['time_s'] for s in report['steps']])
    # Same warnings; numbers may print differently (Python repr vs JS), so compare their gist.
    assert [f.split(':')[0].split(' ')[0] for f in browser['report']['flags']] == \
        [f.split(':')[0].split(' ')[0] for f in report['flags']]


def test_browser_stores_regular_profiles_by_time_step(tmp_path):
    """A regular profile is saved as a time step and its powers; the backend reads it back to the same times."""
    browser = _run_frontend(
        'const t = [0, 0.1, 0.2, 0.3]; const p = [0.3, 1, 1, 0.3];\n'
        'console.log(JSON.stringify({ entry: profileEntry("Cycle", t, p, "c.csv"), irregular: profileEntry("Odd", [0, 0.1, 0.5], [1, 0.5, 1]) }));',
        tmp_path)
    assert browser['entry'] == {'name': 'Cycle', 'source': 'c.csv', 'p': [0.3, 1, 1, 0.3], 't0': 0, 'dt': 0.1}
    t, p = lp.profile_arrays(browser['entry'])
    assert list(t) == pytest.approx([0, 0.1, 0.2, 0.3]) and list(p) == [0.3, 1, 1, 0.3]
    assert browser['irregular']['t'] == [0, 0.1, 0.5]
    assert list(lp.profile_arrays(browser['irregular'])[0]) == [0, 0.1, 0.5]
