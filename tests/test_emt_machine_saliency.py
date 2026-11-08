"""
The detailed machine's subtransient saliency (x''q apart from x''d) and its star point's grounding, against what
has an answer on paper:

- a sudden three-phase short's trapped flux, read in the stator, has a part at twice the frequency beside its DC:
  their ratio is (x''q - x''d) / (x''q + x''d);
- its steady state, started loaded, stays put;
- a single-phase fault at its terminals takes 3E / |x1 + x2 + x0 + 3 Zn|, and none with its star point floating.
"""
import cmath
import math

import numpy as np

import emt_machine
import emt_solver as es

F = 50.0
W0 = 2.0 * math.pi * F
VLL, S = 11e3, 100e6
A120 = 2.0 * math.pi / 3.0
E_B = math.sqrt(2.0 / 3.0) * VLL
I_B = math.sqrt(2.0) * S / (math.sqrt(3.0) * VLL)
Z_B = VLL * VLL / S
# Slow rotor circuits: what the tests read stays at its subtransient value over the cycles they read.
SLOW = dict(xd_pu=1.8, xq_pu=1.7, xd1_pu=0.3, xq1_pu=0.55, xd2_pu=0.25, xl_pu=0.15, td10_s=50.0, td20_s=5.0,
            tq10_s=50.0, tq20_s=5.0, ra_pu=0.0005)


def _space(x):
    return (2.0 / 3.0) * (x[:, 0] + x[:, 1] * cmath.exp(1j * A120) + x[:, 2] * cmath.exp(-1j * A120))


def _open_machine(t_end, close, dt=5e-5, **over):
    """The machine open-circuit at 1 pu; ``close`` the phases shorted to ground at 20 ms."""
    ckt = es.Circuit()
    nodes = [ckt.node('abc'[k]) for k in range(3)]
    for n in nodes:
        ckt.add_r(n, 0, 1e6)
    m = emt_machine.ParkMachine(ckt, 'G', nodes, W0, complex(E_B, 0.0), 0j, S, VLL, h=1e4, **{**SLOW, **over})
    ckt.add_controller(m.control)
    for k in close:
        sw = ckt.add_switch(nodes[k], 0, closed=False)
        ckt.at(0.02, lambda st, sw=sw: st.set_switch(sw, True))
    ckt.start_in_ac_steady_state(W0)
    sim = ckt.simulate(t_end, dt)
    i = np.array([sim['i_rl'][:, k] for k in m.k_e]).T
    return m, sim['t'], i / I_B


def _cycle(t, y, t0, harmonic):
    """y's component turning at harmonic x the fundamental over the cycle from t0 (its complex amplitude)."""
    sel = (t >= t0) & (t < t0 + 1.0 / F - 1e-9)
    return np.mean(y[sel] * np.exp(-1j * harmonic * W0 * t[sel]))


def test_short_circuit_second_harmonic_shows_the_subtransient_saliency():
    xq2 = 0.40
    _, t, i = _open_machine(0.1, (0, 1, 2), xq2_pu=xq2)
    i_s = _space(i)
    dc, second = abs(_cycle(t, i_s, 0.04, 0)), abs(_cycle(t, i_s, 0.04, 2))
    expect = (xq2 - SLOW['xd2_pu']) / (xq2 + SLOW['xd2_pu'])
    assert dc > 1.0
    assert abs(second / dc - expect) < 0.08 * expect, (second / dc, expect)
    # Without saliency, no such part.
    _, t, i = _open_machine(0.1, (0, 1, 2))
    i_s = _space(i)
    assert abs(_cycle(t, i_s, 0.04, 2)) < 0.01 * abs(_cycle(t, i_s, 0.04, 0))


def test_salient_machine_started_loaded_stays_put():
    """On a resistive and inductive load at its rated current, nothing moves (read as two-step means: the trapezoid
    rule's step-period ring on the load is there with or without saliency)."""
    ckt = es.Circuit()
    nodes = [ckt.node('abc'[k]) for k in range(3)]
    z = complex(0.8, 0.6) * Z_B
    for n in nodes:
        ckt.add_rl(n, 0, z.real, z.imag / W0)
    v = complex(E_B, 0.0)
    m = emt_machine.ParkMachine(ckt, 'G', nodes, W0, v, v / z, S, VLL, h=3.0, d=0.0,
                                **{**SLOW, 'td10_s': 6.0, 'td20_s': 0.03, 'tq10_s': 0.6, 'tq20_s': 0.05}, xq2_pu=0.40)
    ckt.add_controller(m.control)
    ckt.start_in_ac_steady_state(W0)
    sim = ckt.simulate(0.5, 5e-5)
    tr = np.array(m.trace)
    p = 0.5 * (tr[1:, 1] + tr[:-1, 1])
    assert abs(p[0] - 80.0) < 0.2
    assert np.max(np.abs(p - 80.0)) < 0.002 * 80.0, np.max(np.abs(p - 80.0))
    assert np.max(np.abs(tr[:, 3] - F)) < 1e-3


def _one_phase(neutral_ohm=None, x0_pu=None):
    over = {} if neutral_ohm is None else dict(neutral_ohm=neutral_ohm, x0_pu=x0_pu)
    _, t, i = _open_machine(0.1, (0,), **over)
    return abs(_cycle(t, i[:, 0], 0.04, 1)) * 2.0


def test_single_phase_fault_through_the_grounded_star_point():
    x2, x0 = SLOW['xd2_pu'], 0.40
    assert abs(_one_phase(0.0, x0) - 3.0 / (2.0 * x2 + x0)) < 0.04 * 3.0 / (2.0 * x2 + x0)
    rn = 0.3 * Z_B
    expect = 3.0 / abs(complex(3.0 * 0.3, 2.0 * x2 + x0))
    assert abs(_one_phase(rn, x0) - expect) < 0.04 * expect
    # x0 below x'' cannot be had through the star point: x0 = x'' then.
    assert abs(_one_phase(0.0, 0.1) - 1.0 / x2) < 0.04 / x2
    # Floating: nothing flows.
    assert _one_phase() < 0.01


def test_park_data_reads_the_saliency_and_grounding_fields():
    row = {'dyn_xd2': '0.25', 'dyn_xq2': '0.4', 'dyn_xq1': '0.3', 'dyn_neutral_ohm': '0', 'dyn_x0': '0.12'}
    data, notes = emt_machine.park_data(row, *_defaults())
    assert data['xq2_pu'] == 0.4 and data['neutral_ohm'] == 0.0 and data['x0_pu'] == 0.12
    assert data['xq1_pu'] > 0.4 and any("x''q" in n for n in notes)
    data, _ = emt_machine.park_data({'dyn_xd2': '0.25'}, *_defaults())
    assert data['xq2_pu'] == 0.25 and 'neutral_ohm' not in data


def _defaults():
    from andes_electrisim import _DEFAULT_GENROU, _EXCITER_DEFAULTS, _GOVERNOR_DEFAULTS
    return _GOVERNOR_DEFAULTS, _EXCITER_DEFAULTS, _DEFAULT_GENROU
