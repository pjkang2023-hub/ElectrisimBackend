"""
The frequency-dependent line untransposed, and as two circuits on one tower (emt_fd_line with conductors),
against the exact multi-conductor line at the fundamental: its two-port from Z Y's own complex eigenvectors,
Y11 = Yc coth(Gamma l), Y12 = -Yc csch(Gamma l), Yc = Z^-1 Gamma. The fixed real transformation is J. Marti's
approximation; what it leaves out is under a per cent here. Then each mode's wave at its own speed, and through the
study: a line's switch to untransposed, two circuits, and a fault along it.
"""
import math

import numpy as np
import pytest

import emt_fd_line as fd
import emt_solver as es

GEOMETRY = {'height_m': 20.0, 'spacing_m': 7.0, 'radius_m': 0.015, 'rdc_ohm_per_km': 0.06, 'rho_ohm_m': 100.0}
FLAT = {**GEOMETRY, 'transposed': False}
DOUBLE = {**GEOMETRY, 'conductors': [(-4.0, 20.0), (-4.5, 27.0), (-4.0, 34.0), (4.0, 20.0), (4.5, 27.0), (4.0, 34.0)]}
KM, F1, VLL = 200.0, 50.0, 230e3
W0 = 2.0 * math.pi * F1
A120 = 2.0 * math.pi / 3.0


def _exact_far_end(geometry, n_circuits, va):
    """The far end's voltages, open there, the near end held at ``va`` (phases), exactly."""
    z, y = fd.per_km_matrix(np.array([F1]), geometry)
    z, y = z[0], y[0]
    lam, vec = np.linalg.eig(z @ y)
    g = np.sqrt(lam)
    g = np.where(g.real < 0, -g, g)
    vi = np.linalg.inv(vec)
    zi = np.linalg.inv(z)
    y11 = zi @ vec @ np.diag(g / np.tanh(g * KM)) @ vi
    y12 = -zi @ vec @ np.diag(g / np.sinh(g * KM)) @ vi
    pm = np.kron(np.ones((n_circuits, 1)), np.eye(3))
    return -np.linalg.solve(pm.T @ y11 @ pm, pm.T @ y12 @ pm @ va)


def _energise(geometry, n_circuits, dt=2e-5, t_end=0.1):
    ckt = es.Circuit()
    a = [ckt.node(f'a{p}') for p in 'abc']
    b = [ckt.node(f'b{p}') for p in 'abc']
    line = fd.FdLine(ckt, a * n_circuits, b * n_circuits, geometry, KM, dt, W0, 'L')
    ckt.add_controller(line.control)
    v_pk = math.sqrt(2.0 / 3.0) * VLL
    for k, n in enumerate(a):
        ckt.add_rl(0, n, 1e-3, 1e-6, ac=(v_pk, W0, -k * A120))
    for n in b:
        ckt.add_r(n, 0, 1e8)
    ckt.start_in_ac_steady_state(W0)
    sim = ckt.simulate(t_end, dt)
    va = v_pk * np.exp(-1j * A120 * np.arange(3))
    return line, sim, b, va


@pytest.mark.parametrize('geometry, n', [(FLAT, 1), (DOUBLE, 2)], ids=['flat untransposed', 'double circuit'])
def test_its_steady_state_is_the_exact_multiconductor_line(geometry, n):
    line, sim, b, va = _energise(geometry, n)
    exact = _exact_far_end(geometry, n, va)
    t = sim['t']
    sel = t >= t[-1] - 1.0 / F1
    for k in range(3):
        got = 2.0 * np.mean(sim['v'][sel, b[k]] * np.exp(-1j * W0 * t[sel]))
        assert abs(got) == pytest.approx(abs(exact[k]), rel=0.01), k
        assert math.degrees(abs(np.angle(got / exact[k]))) < 0.5, k
    # Flat from the start: no transient beyond what the fixed transformation leaves.
    first = np.max(np.abs(sim['v'][t < 1.0 / F1, b[0]]))
    assert first == pytest.approx(abs(exact[0]), rel=0.015)


def _two_port(geometry):
    z, y = fd.per_km_matrix(np.array([F1]), geometry)
    z, y = z[0], y[0]
    lam, vec = np.linalg.eig(z @ y)
    g = np.sqrt(lam)
    g = np.where(g.real < 0, -g, g)
    vi, zi = np.linalg.inv(vec), np.linalg.inv(z)
    y11 = zi @ vec @ np.diag(g / np.tanh(g * KM)) @ vi
    y12 = -zi @ vec @ np.diag(g / np.sinh(g * KM)) @ vi
    return np.block([[y11, y12], [y12, y11]])


def test_one_phase_energised_couples_unequally_into_the_others():
    """Phase a alone energised, b and c open at both ends: the middle phase, nearer, takes more than the far outer
    one - the coupling a transposed line averages away - each as the exact line has it."""
    dt = 2e-5
    ckt = es.Circuit()
    a = [ckt.node(f'a{p}') for p in 'abc']
    b = [ckt.node(f'b{p}') for p in 'abc']
    line = fd.FdLine(ckt, a, b, FLAT, KM, dt, W0, 'L')
    ckt.add_controller(line.control)
    v_pk = 100e3
    ckt.add_rl(0, a[0], 1e-3, 1e-6, ac=(v_pk, W0, 0.0))
    for n in a[1:] + b:
        ckt.add_r(n, 0, 1e9)
    ckt.start_in_ac_steady_state(W0)
    sim = ckt.simulate(0.06, dt)
    y = _two_port(FLAT)
    # Unknowns: near b, c and far a, b, c (indices 1, 2, 3, 4, 5), no current into them; near a held.
    free = [1, 2, 3, 4, 5]
    v = np.zeros(6, dtype=complex)
    v[0] = v_pk
    v[free] = -np.linalg.solve(y[np.ix_(free, free)], y[free, 0] * v_pk)
    t = sim['t']
    sel = t >= t[-1] - 1.0 / F1
    nodes = a + b
    for k in free:
        got = 2.0 * np.mean(sim['v'][sel, nodes[k]] * np.exp(-1j * W0 * t[sel]))
        assert abs(got - v[k]) < 0.02 * abs(v[k]), (k, got, v[k])
    assert abs(v[1]) > 1.3 * abs(v[2]) and abs(v[4]) > 1.3 * abs(v[5])


def test_each_mode_travels_at_its_own_speed():
    line, *_ = _energise(FLAT, 1, t_end=1e-4)
    ground, aerial = line.tau_ms
    assert len(line.tau_modes_ms) == 3
    assert aerial == pytest.approx(KM / 299792.458 * 1e3, rel=0.03)
    assert ground > 1.03 * aerial
    # Six conductors, six modes.
    line6, *_ = _energise(DOUBLE, 2, t_end=1e-4)
    assert len(line6.tau_modes_ms) == 6


def test_its_modes_are_the_flat_lines_own():
    """Q Z Q^T is diagonal at the transformation's frequency, and the transposed modes are its average."""
    f = np.array([1000.0])
    z, y = fd.per_km_matrix(f, FLAT)
    q, zm, ym = fd.modal(z, y, f, 1000.0)
    full = q @ z[0] @ q.T
    assert np.max(np.abs(full - np.diag(np.diag(full)))) < 1e-3 * np.max(np.abs(np.diag(full)))
    z1, _, z0, _ = fd.per_km(f, GEOMETRY)
    # Its phase impedances' mean is the transposed line's: (z0 + 2 z1) / 3 on the diagonal.
    assert np.mean(np.diag(z[0])) == pytest.approx(complex((z0[0] + 2.0 * z1[0]) / 3.0), rel=0.02)


# --- Through the study -------------------------------------------------------------------------

def _request(**line_fields):
    import test_emt_fd_line as T
    request = T._request(fd_line='true')
    next(v for v in request.values() if v.get('typ') == 'Line0').update({k: str(v) for k, v in line_fields.items()})
    return request


def test_the_study_runs_it_untransposed_and_as_two_circuits(client, quiet):
    import test_emt_fd_line as T
    flat, w1 = T._emt(client, quiet, _request(fd_transposed='false'))
    assert flat['fd_lines'] and not [w for w in w1 if 'pi sections' in w]
    xy = '; '.join(f'{x},{y}' for x, y in DOUBLE['conductors'])
    double, w2 = T._emt(client, quiet, _request(fd_conductors=xy, parallel='2'))
    assert double['fd_lines'] and not [w for w in w2 if 'pi sections' in w], w2
    bad, w3 = T._emt(client, quiet, _request(fd_conductors=xy))
    assert 'fd_lines' not in bad and any('not three for each' in w for w in w3)


def test_a_fault_along_it_splits_it_at_the_fault(client, quiet):
    import test_emt_fd_line as T
    request = T._request(fd_line='true', ac_fault_line=next(v for v in T._request().values() if v.get('typ') == 'Line0')['name'],
                         ac_fault_fraction='0.25', ac_fault_type='abc', ac_fault_time_ms='40', ac_fault_duration_ms='40')
    emt, warnings = T._emt(client, quiet, request)
    assert len(emt['fd_lines']) == 2 and not [w for w in warnings if 'pi sections' in w], warnings
    taus = sorted(x['travel_aerial_ms'] for x in emt['fd_lines'])
    assert taus[1] == pytest.approx(3.0 * taus[0], rel=0.03)
