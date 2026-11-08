"""
The MMC's full-bridge submodules and its submodule-by-submodule model (emt_mmc with cell 'full', submodules N):

- full-bridge, blocked on a DC fault: each arm stands at its capacitors' voltage against the current either way,
  above the AC side's line-to-line peak, so no current reaches the fault: the AC side's falls to nothing and the
  capacitors keep (or gain) their charge - where half-bridges feed it through their bypass diodes;
- submodule by submodule: the arm-averaged staircase's state at as many levels (its nearest-level steps shift
  the state the same way), its capacitors held together by sorting from a 10 % spread, and drifting apart with
  balancing off.
"""
import numpy as np
import pytest

import test_emt_mmc as M

W0 = M.W0


def test_full_bridge_cells_run_as_half_bridge_ones_in_the_steady_state():
    _, half, *_ = M._run(t_end=0.1)
    _, full, *_ = M._run(t_end=0.1, cell='full')
    th, tf = np.array(half.trace), np.array(full.trace)
    assert tf[:, 1].mean() == pytest.approx(th[:, 1].mean(), rel=2e-3)
    assert tf[-1, 5] == pytest.approx(th[-1, 5], rel=2e-3)


def test_full_bridge_cells_block_a_dc_fault():
    ckt, m, ac, dc, i_ph = M._build(p_pu=0.8, q_pu=0.2, s_grid=500e6, cell='full')
    sw = ckt.add_switch(dc, 0, closed=False, r_on=0.001)
    ckt.at(0.02, lambda st: st.set_switch(sw, True))
    ckt.start_in_ac_steady_state(W0)
    sim = ckt.simulate(0.15, 2e-5)
    t = sim['t']
    assert m.blocked_at is not None and 0.02 < m.blocked_at < 0.03
    arms = np.array([sim['i_rl'][:, k] for k in m.k_u + m.k_l]).T
    late = t > m.blocked_at + 0.02
    assert np.max(np.abs(arms[late])) < 0.01 * M.I_PK
    i_ac = np.array([sim['i_rl'][:, u] - sim['i_rl'][:, d] for u, d in zip(m.k_u, m.k_l)]).T
    assert np.max(np.abs(i_ac[late])) < 0.01 * M.I_PK
    tr = np.array(m.trace)
    w_block = float(np.interp(m.blocked_at + 1e-4, tr[:, 0], tr[:, 5]))
    assert tr[-1, 5] >= 0.999 * w_block
    # The half-bridge converter, the same fault: still fed, at the AC side's short-circuit current.
    ckt, h, ac, dc, i_ph = M._build(p_pu=0.8, q_pu=0.2, s_grid=500e6)
    sw = ckt.add_switch(dc, 0, closed=False, r_on=0.001)
    ckt.at(0.02, lambda st: st.set_switch(sw, True))
    ckt.start_in_ac_steady_state(W0)
    sim = ckt.simulate(0.15, 2e-5)
    i_ac = np.array([sim['i_rl'][:, u] - sim['i_rl'][:, d] for u, d in zip(h.k_u, h.k_l)]).T
    assert np.max(np.abs(i_ac[sim['t'] > h.blocked_at + 0.02])) > 0.5 * M.I_PK


def test_submodule_by_submodule_it_holds_the_averaged_state_and_balances():
    _, avg, *_ = M._run(t_end=0.12, levels=20)
    _, sm, *_ = M._run(t_end=0.12, submodules=20, sm_spread=0.1)
    ta, ts = np.array(avg.trace), np.array(sm.trace)
    late_s, late_a = ts[:, 0] > 0.06, ta[:, 0] > 0.06
    assert ts[late_s, 1].mean() == pytest.approx(ta[late_a, 1].mean(), rel=2e-3)
    assert ts[late_s, 5].mean() == pytest.approx(ta[late_a, 5].mean(), rel=3e-3)
    r = sm.sm_result(0.12)
    assert r['spread_end'] < 0.05, r
    assert r['switching_hz'] > 50.0
    # Balancing off: the same submodules always inserted first, their charge left to drift.
    _, drift, *_ = M._run(t_end=0.12, submodules=20, sm_spread=0.1, balancing=False)
    assert drift.sm_result(0.12)['spread_end'] > 2.0 * r['spread_end']


def test_submodules_through_the_study(client, quiet):
    out = M._post(client, quiet, M._mmc_request(mmc_submodules='10', mmc_cell='full'))
    (conv,) = out['emt']['converters']
    assert conv['cell'] == 'full' and conv['submodules']['submodules'] == 10
    request = M._mmc_request()
    next(v for v in request.values() if isinstance(v, dict) and str(v.get('typ', '')).startswith('VSC'))['mmc_levels'] = 10
    stair = M._post(client, quiet, request)['emt']['converters'][0]
    assert conv['v_dc_end_kv'] == pytest.approx(stair['v_dc_end_kv'], rel=3e-3)
    assert conv['p_end_mw'] == pytest.approx(stair['p_end_mw'], rel=0.01)
    assert any('full-bridge' in w for w in out['warnings'])
