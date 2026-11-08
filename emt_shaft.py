"""
A turbine-generator's shaft as masses and springs, for the detailed machine (emt_machine.ParkMachine): the
torsional modes a single rotor leaves out, which converters' and series capacitors' controls can excite.

Masses 0 .. n-1 along the shaft, the generator one of them (``gen``), each with its inertia H (s, on the machine's
rating) and damping D to its own speed; springs K between neighbours (per unit torque per electrical radian).
The turbine's mechanical torque is shared among the masses by fractions F (summing to 1; the generator's and an
exciter's 0); the electrical torque acts on the generator's mass. Per unit, angles in electrical radians:

    2 H_i dw_i/dt = F_i Tm - K_(i-1,i) (d_i - d_(i-1)) - K_(i,i+1) (d_i - d_(i+1)) - D_i (w_i - 1)  [- Te at the generator]
    d d_i / dt = w0 (w_i - 1)

It starts twisted to carry its steady torques at nominal speed. Each step updates the speeds from the torques,
then the angles from the new speeds (semi-implicit Euler, which keeps an undamped spring's energy bounded).
"""
import math

import numpy as np


def parse(row, key):
    """A comma-separated list of numbers from a row's field, or None."""
    raw = str((row or {}).get(key) or '').strip()
    if not raw:
        return None
    try:
        return [float(x) for x in raw.replace(';', ',').split(',') if x.strip()]
    except ValueError:
        return None


class Shaft:
    def __init__(self, h, k, f, gen, w0, tm0, d=None):
        n = len(h)
        if len(k) != n - 1 or len(f) != n or not 0 <= gen < n:
            raise ValueError('a shaft needs one spring fewer than masses, a share of the torque for each mass, and its generator among them')
        if min(h) <= 0 or min(k) <= 0:
            raise ValueError('a shaft needs positive inertias and springs')
        total = sum(f)
        self.h = np.array(h, dtype=float)
        self.k = np.array(k, dtype=float)
        self.f = np.array(f, dtype=float) / (total if total > 0 else 1.0)
        self.d = np.array(d if d is not None and len(d) == n else [0.0] * n, dtype=float)
        self.gen, self.w0, self.n = gen, w0, n
        self.w = np.ones(n)
        # Twisted to carry the steady torques: the torque in each spring is what the masses beyond it put in.
        self.angle = np.zeros(n)
        # Before the generator, spring i carries towards it what masses 0 .. i put in; beyond it, spring i carries
        # back towards it what masses i + 1 .. n - 1 put in.
        for i in range(gen):
            self.angle[i + 1] = self.angle[i] - self.f[:i + 1].sum() * tm0 / self.k[i]
        for i in range(gen, n - 1):
            self.angle[i + 1] = self.angle[i] + self.f[i + 1:].sum() * tm0 / self.k[i]
        self.angle -= self.angle[gen]
        self.torque0 = self._spring_torques()
        self.torque_max = np.abs(self.torque0)

    def _spring_torques(self):
        """Each spring's torque, from mass i to i + 1."""
        return self.k * (self.angle[:-1] - self.angle[1:])

    def frequencies(self):
        """Its torsional modes' frequencies (Hz) with the generator free: the rigid-body mode left out."""
        m = np.diag(2.0 * self.h)
        stiffness = np.zeros((self.n, self.n))
        for i, kk in enumerate(self.k):
            stiffness[i, i] += kk
            stiffness[i + 1, i + 1] += kk
            stiffness[i, i + 1] -= kk
            stiffness[i + 1, i] -= kk
        lam = np.linalg.eigvals(np.linalg.solve(m, stiffness) * self.w0)
        return sorted(math.sqrt(x.real) / (2.0 * math.pi) for x in lam if x.real > 1e-6)

    def step(self, dt, tm, te):
        """Speeds then angles over dt; returns the generator's speed and the change in its angle."""
        torque = self.f * tm - self.d * (self.w - 1.0)
        torque[self.gen] -= te
        spring = self._spring_torques()
        torque[:-1] -= spring
        torque[1:] += spring
        self.w += dt * torque / (2.0 * self.h)
        before = self.angle[self.gen]
        self.angle += self.w0 * (self.w - 1.0) * dt
        np.maximum(self.torque_max, np.abs(self._spring_torques()), out=self.torque_max)
        return self.w[self.gen], self.angle[self.gen] - before
