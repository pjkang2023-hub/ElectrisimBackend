"""
A three-phase run of single-core cables in the EMT study with their sheaths as conductors: the sheath voltages
and currents that bonding sets, solid, single-point or cross-bonded.

Six conductors - three cores, three sheaths - coupled to each other and returning through the earth, by
Carson's equations in their usual approximation (depth of return De = 658.5 sqrt(rho / f) m):

    Z_ii = R_i + w u0 / 8 + j w u0 / 2 pi  ln(De / GMR_i),    Z_ij = w u0 / 8 + j w u0 / 2 pi  ln(De / d_ij)

a core's GMR r e^(-1/4), a sheath's its mean radius, a core and its own sheath d = the sheath's mean radius,
otherwise the cables' centre spacing. Capacitance from each core to its sheath through the insulation, and from
each sheath to the earth through its jacket. R and L are taken at the fundamental (``f_hz``): this model is for
the sheaths at power frequency and the slow transients on it, not for travelling waves, which emt_fd_line's
coaxial model carries. Pi sections, ``sections`` per minor section.

Bonding:
- solid: each sheath grounded at both ends;
- single: grounded at the from end, open at the to end, where its standing voltage appears;
- cross: three equal minor sections per major section, the sheaths transposed at each joint (a to b to c), each
  major section's ends grounded; the sheath currents cancel and the voltage at the joints is a minor section's.

Cable positions: trefoil (an equilateral triangle of side ``spacing_m``) or flat (a row, ``spacing_m`` apart),
``depth_m`` down.

Wideband (WidebandCable): the same six conductors with their impedances at every frequency, for travelling
waves as well - each core's skin effect (Bessel functions), each sheath a thin tube (its inner and outer
surfaces' impedances and the mutual one between them, rho m / 2 pi r coth or csch of m t), and the earth's
return by Saad, Gaba and Giroux's closed form of Pollaczek's integral,

    Z_e ij = j w u0 / 2 pi  (K0(m d_ij) + 2 / (4 + m^2 x_ij^2) exp(-(h_i + h_j) m)),   m = sqrt(j w u0 / rho)

(d_ij the cables' distance, a cable's own its outer radius; x_ij their horizontal distance; h their depths),
which at power frequency is Carson's above. Each cable:

    Z_cc = Z_core + Z_ins + Z_sheath in + Z_sheath out - 2 Z_sheath mutual + Z_jacket + Z_e,   Z_cs = Z_sheath out - Z_sheath mutual + Z_jacket + Z_e,
    Z_ss = Z_sheath out + Z_jacket + Z_e

and between cables Z_e ij in all four. Its admittance through the insulation and the jacket, the cables
screened from each other by their sheaths. Run by the universal line model (emt_ulm), in the phase domain: a
constant modal transformation is far off for cores and sheaths together (its two-port 57 % off at 50 Hz on
the test cable). One line per minor section, bonded as above.
"""
import math

import numpy as np

U0 = 4e-7 * math.pi
E0 = 8.854187817e-12
FIELDS = ('core_radius_m', 'core_rdc_ohm_per_km', 'insulation_radius_m', 'sheath_thickness_m', 'sheath_rdc_ohm_per_km', 'eps_r',
          'spacing_m', 'depth_m', 'jacket_radius_m', 'rho_ohm_m')
BONDINGS = ('solid', 'single', 'cross')


def positions(formation, spacing, depth):
    if formation == 'flat':
        return [(-spacing, depth), (0.0, depth), (spacing, depth)]
    h = spacing * math.sqrt(3.0) / 2.0
    return [(-spacing / 2.0, depth + h / 3.0), (spacing / 2.0, depth + h / 3.0), (0.0, depth - 2.0 * h / 3.0)]


def matrices(g, f_hz, formation='trefoil'):
    """R and L (per m) of the six conductors, cores then sheaths, and the per-m capacitances core to sheath and sheath to earth."""
    w = 2.0 * math.pi * f_hz
    xy = positions(formation, g['spacing_m'], g['depth_m'])
    r_sm = g['insulation_radius_m'] + g['sheath_thickness_m'] / 2.0
    de = 658.5 * math.sqrt(g['rho_ohm_m'] / f_hz)
    re = w * U0 / 8.0
    gmr = [g['core_radius_m'] * math.exp(-0.25)] * 3 + [r_sm] * 3
    rdc = [g['core_rdc_ohm_per_km'] * 1e-3] * 3 + [g['sheath_rdc_ohm_per_km'] * 1e-3] * 3
    z = np.zeros((6, 6), dtype=complex)
    for i in range(6):
        for j in range(6):
            if i == j:
                z[i, j] = rdc[i] + re + 1j * w * U0 / (2.0 * math.pi) * math.log(de / gmr[i])
            else:
                ci, cj = i % 3, j % 3
                d = r_sm if ci == cj else math.dist(xy[ci], xy[cj])
                z[i, j] = re + 1j * w * U0 / (2.0 * math.pi) * math.log(de / d)
    c_ins = 2.0 * math.pi * E0 * g['eps_r'] / math.log(g['insulation_radius_m'] / g['core_radius_m'])
    outer = g['insulation_radius_m'] + g['sheath_thickness_m']
    c_jacket = 2.0 * math.pi * E0 * g.get('jacket_eps_r', 2.5) / math.log(max(g['jacket_radius_m'], outer * 1.01) / outer)
    return z.real, z.imag / w, c_ins, c_jacket


class CableSheaths:
    """``a``, ``b``: its ends' core nodes (three each). Its sheath nodes and section groups are kept for the results."""

    def __init__(self, ckt, a, b, g, km, f_hz, label, bonding='cross', formation='trefoil', major_sections=1, sections=4, r_ground=0.1):
        if bonding not in BONDINGS:
            raise ValueError(f"its bonding must be one of {', '.join(BONDINGS)}")
        self.label, self.bonding = label, bonding
        r_m, l_m, c_ins, c_jacket = matrices(g, f_hz, formation)
        minors = 3 * max(int(major_sections), 1) if bonding == 'cross' else 1
        n_sec = max(int(sections), 1)
        length = km * 1e3 / (minors * n_sec)
        r, l = r_m * length, l_m * length
        self.groups = []
        self.sheath_nodes = []          # every sheath node along it, per sheath: for its highest voltage
        self.joints = []                # the sheath nodes at each cross-bond joint and at the ends
        cores = list(a)
        sheath_start = [ckt.node(f'{label} sheath {p} start') for p in 'abc']
        self._bond(ckt, sheath_start, r_ground, 'start')
        sheaths = sheath_start
        self.joints.append(list(sheaths))
        for minor in range(minors):
            for k in range(n_sec):
                last = minor == minors - 1 and k == n_sec - 1
                next_cores = list(b) if last else [ckt.node(f'{label} core {p} {minor}.{k}') for p in 'abc']
                next_sheaths = [ckt.node(f'{label} sheath {p} {minor}.{k}') for p in 'abc']
                self.groups.append(ckt.add_coupled(cores + sheaths, next_cores + next_sheaths, r, l))
                for end_cores, end_sheaths in ((cores, sheaths), (next_cores, next_sheaths)):
                    for p in range(3):
                        ckt.add_c(end_cores[p], end_sheaths[p], c_ins * length / 2.0)
                        ckt.add_c(end_sheaths[p], 0, c_jacket * length / 2.0)
                cores, sheaths = next_cores, next_sheaths
                self.sheath_nodes.append(list(sheaths))
            self.joints.append(list(sheaths))
            if bonding == 'cross' and minor < minors - 1:
                if (minor + 1) % 3 == 0:
                    # A major section's end: its sheaths bonded and grounded, and on into the next.
                    self._bond(ckt, sheaths, r_ground, f'major {minor // 3}')
                else:
                    # A cross-bond joint: sheath a goes on as sheath b's continuation, b as c, c as a.
                    sheaths = [sheaths[2], sheaths[0], sheaths[1]]
        if bonding in ('solid', 'cross'):
            self._bond(ckt, sheaths, r_ground, 'end')
        self.sheath_end = list(sheaths)

    def _bond(self, ckt, sheaths, r_ground, where):
        """The three sheaths joined together and the join grounded once: a circulating current meets no ground resistance."""
        bond = ckt.node(f'{self.label} sheath bond {where}')
        for node in sheaths:
            ckt.add_r(node, bond, 1e-6)
        ckt.add_r(bond, 0, r_ground)


def _tube(w, a, b, rdc_per_m):
    """A thin tube's inner surface, outer surface and mutual impedances (ohm/m), its inner and outer radii a, b."""
    rho = rdc_per_m * math.pi * (b * b - a * a)
    m = np.sqrt(1j * w * U0 / rho)
    t = b - a
    coth = 1.0 / np.tanh(m * t)
    return (rho * m / (2.0 * math.pi * a) * coth, rho * m / (2.0 * math.pi * b) * coth,
            rho * m / (math.pi * (a + b)) / np.sinh(m * t))


def _earth(w, d, x, h_sum, rho):
    from scipy.special import kv
    m = np.sqrt(1j * w * U0 / rho)
    return 1j * w * U0 / (2.0 * math.pi) * (kv(0, m * d) + 2.0 / (4.0 + m * m * x * x) * np.exp(-h_sum * m))


def per_km_wideband(f, g, formation='trefoil'):
    """Z and Y per km of the six conductors (cores, then sheaths) at frequencies ``f``: arrays (len(f), 6, 6)."""
    import emt_fd_line
    w = 2.0 * math.pi * np.asarray(f, dtype=float)
    rc, a = g['core_radius_m'], g['insulation_radius_m']
    b = a + g['sheath_thickness_m']
    rj = max(g['jacket_radius_m'], b * 1.01)
    xy = positions(formation, g['spacing_m'], g['depth_m'])
    z_core = emt_fd_line._internal(w, rc, g['core_rdc_ohm_per_km'] * 1e-3)
    z_ins = 1j * w * U0 / (2.0 * math.pi) * math.log(a / rc)
    z_in, z_out, z_mut = _tube(w, a, b, g['sheath_rdc_ohm_per_km'] * 1e-3)
    z_jkt = 1j * w * U0 / (2.0 * math.pi) * math.log(rj / b)
    z = np.zeros((len(w), 6, 6), dtype=complex)
    for i in range(3):
        for j in range(3):
            if i == j:
                ze = _earth(w, rj, rj, 2.0 * xy[i][1], g['rho_ohm_m'])
                z[:, i, i] = z_core + z_ins + z_in + z_out - 2.0 * z_mut + z_jkt + ze
                z[:, i, 3 + i] = z[:, 3 + i, i] = z_out - z_mut + z_jkt + ze
                z[:, 3 + i, 3 + i] = z_out + z_jkt + ze
            else:
                ze = _earth(w, math.dist(xy[i], xy[j]), abs(xy[i][0] - xy[j][0]), xy[i][1] + xy[j][1], g['rho_ohm_m'])
                for p in (i, 3 + i):
                    for q in (j, 3 + j):
                        z[:, p, q] = ze
    c_ins = 2.0 * math.pi * E0 * g['eps_r'] / math.log(a / rc)
    c_jkt = 2.0 * math.pi * E0 * g.get('jacket_eps_r', 2.5) / math.log(rj / b)
    pot = np.zeros((6, 6))
    for i in range(3):
        pot[i, i] = 1.0 / c_ins + 1.0 / c_jkt
        pot[i, 3 + i] = pot[3 + i, i] = pot[3 + i, 3 + i] = 1.0 / c_jkt
    y = 1j * w[:, None, None] * np.linalg.inv(pot)[None, :, :]
    return z * 1e3, y * 1e3


class WidebandCable:
    """
    The six-conductor cable as emt_fd_line lines, one per minor section, bonded as CableSheaths is. ``a``, ``b``:
    its ends' core nodes. Its from end's core and sheath currents are kept for the results.
    """

    def __init__(self, ckt, a, b, g, km, dt, w0, label, bonding='cross', formation='trefoil', major_sections=1, r_ground=0.1):
        import emt_ulm
        if bonding not in BONDINGS:
            raise ValueError(f"its bonding must be one of {', '.join(BONDINGS)}")
        self.label, self.bonding, self.formation = label, bonding, formation
        minors = 3 * max(int(major_sections), 1) if bonding == 'cross' else 1
        f = np.logspace(-2, 6, 241)
        z, y = per_km_wideband(f, g, formation)
        self.fit = emt_ulm.UlmFit(z, y, km / minors, f)
        self.lines, self.sheath_nodes, self.joints = [], [], []
        cores = list(a)
        sheaths = [ckt.node(f'{label} sheath {p} start') for p in 'abc']
        self._ground = r_ground
        CableSheaths._bond(self, ckt, sheaths, r_ground, 'start')
        self.joints.append(list(sheaths))
        for minor in range(minors):
            last = minor == minors - 1
            next_cores = list(b) if last else [ckt.node(f'{label} core {p} {minor}') for p in 'abc']
            next_sheaths = [ckt.node(f'{label} sheath {p} {minor}') for p in 'abc']
            line = emt_ulm.UlmLine(ckt, cores + sheaths, next_cores + next_sheaths, self.fit, dt, w0, f'{label} minor {minor}')
            ckt.add_controller(line.control)
            self.lines.append(line)
            cores, sheaths = next_cores, next_sheaths
            self.sheath_nodes.append(list(sheaths))
            self.joints.append(list(sheaths))
            if bonding == 'cross' and not last:
                if (minor + 1) % 3 == 0:
                    CableSheaths._bond(self, ckt, sheaths, r_ground, f'major {minor // 3}')
                else:
                    sheaths = [sheaths[2], sheaths[0], sheaths[1]]
        if bonding in ('solid', 'cross'):
            CableSheaths._bond(self, ckt, sheaths, r_ground, 'end')
        self.sheath_end = list(sheaths)
        self.i_from = []
        ckt.add_controller(self._record)

    def _record(self, t, state):
        self.i_from.append(self.lines[0].currents()[0].copy())
        return False

    @property
    def tau_ms(self):
        return self.lines[0].tau_ms
