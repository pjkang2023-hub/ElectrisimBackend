"""
The detailed machine's excitation limiters (emt_controls.Limiters), through its exciter (SEXS here):

- over-excitation: with its field at 125 % of the rating it is given and k = 1 s, it waits (1.25^2 - 1)^-1 s, as
  its heating integral says, then brings its field current down to that rating;
- under-excitation: on a stiff grid, absorbing more reactive power than its line allows, it raises its excitation
  until its Q sits on the line; with the line below its operating point, it does nothing.
"""
import math

import numpy as np
import pytest

import emt_controls
import emt_machine
import emt_solver as es

F = 50.0
W0 = 2.0 * math.pi * F
VLL, S = 11e3, 100e6
A120 = 2.0 * math.pi / 3.0
E_B = math.sqrt(2.0 / 3.0) * VLL
Z_B = VLL * VLL / S
DATA = dict(xd_pu=1.8, xq_pu=1.7, xd1_pu=0.3, xq1_pu=0.55, xd2_pu=0.25, xl_pu=0.15, td10_s=6.0, td20_s=0.03,
            tq10_s=0.4, tq20_s=0.05, ra_pu=0.0025)
EXC = emt_controls.parameters('exc', {'dyn_exciter_model': 'SEXS'})


def _machine(ckt, nodes, v, i, lim):
    m = emt_machine.ParkMachine(ckt, 'G', nodes, W0, v, i, S, VLL, h=1e4, d=0.0, **DATA,
                                controls={'exc': EXC, 'lim': lim} if lim else {'exc': EXC})
    ckt.add_controller(m.control)
    return m


def test_over_excitation_waits_its_heating_time_then_holds_its_rating():
    v = complex(E_B, 0.0)
    z = complex(0.8, 0.6) * Z_B
    probe = emt_machine.ParkMachine(es.Circuit(), 'P', [0, 0, 0], W0, v, v / z, S, VLL, **DATA, controls={'exc': EXC})
    rating = probe.efd0 / 1.25
    ckt = es.Circuit()
    nodes = [ckt.node('abc'[k]) for k in range(3)]
    for n in nodes:
        ckt.add_rl(n, 0, z.real, z.imag / W0)
    m = _machine(ckt, nodes, v, v / z, emt_controls.limiter_parameters({'dyn_oel_ifd_pu': str(rating), 'dyn_oel_k_s': '1'}))
    ifd = []
    ckt.add_controller(lambda t, st: ifd.append((t, getattr(m, 'ifd', m.efd0))) and False)
    ckt.start_in_ac_steady_state(W0)
    ckt.simulate(6.0, 1e-4)
    t, x = np.array(ifd).T
    assert m.limiters.oel_picked_s == pytest.approx(1.0 / (1.25 ** 2 - 1.0), rel=0.02)
    assert np.interp(1.7, t, x) == pytest.approx(probe.efd0, rel=0.01)
    assert x[-1] == pytest.approx(rating, rel=0.01)


def _grid_run(q0):
    """At 0.5 pu power absorbing 0.3 pu reactive, behind 0.1 pu of a stiff grid."""
    v = complex(E_B, 0.0)
    s_pu = complex(0.5, -0.3)
    i = (s_pu * S / (1.5 * v)).conjugate()
    zg = complex(0.0, 0.1) * Z_B
    eg = v - zg * i
    ckt = es.Circuit()
    nodes = [ckt.node('abc'[k]) for k in range(3)]
    for k, n in enumerate(nodes):
        ckt.add_rl(0, n, zg.real + 1e-3, zg.imag / W0, ac=(abs(eg), W0, float(np.angle(eg)) - k * A120))
    m = _machine(ckt, nodes, v, i, emt_controls.limiter_parameters({'dyn_uel_q0_pu': str(q0)}))
    ckt.start_in_ac_steady_state(W0)
    ckt.simulate(4.0, 1e-4)
    tr = np.array(m.trace)
    return m, tr[:, 0], tr[:, 2] * 1e6 / S, tr[:, 8]


def test_under_excitation_raises_its_excitation_onto_its_line():
    m, t, q, v = _grid_run(-0.2)
    q_end, v_end = np.mean(q[t > 3.8]), np.mean(v[t > 3.8])
    assert m.limiters.uel_acted_s is not None and m.limiters.uel_acted_s < 0.01
    assert q_end == pytest.approx(-0.2 * v_end * v_end, abs=0.01)
    assert m.efd > m.efd0


def test_under_excitation_idle_inside_its_line():
    m, t, q, _ = _grid_run(-0.5)
    assert m.limiters.uel_acted_s is None
    assert np.mean(q[t > 3.8]) == pytest.approx(-0.3, abs=0.005)
