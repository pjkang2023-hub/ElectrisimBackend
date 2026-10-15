"""
IEC 60909's single-phase short circuit with full-converter current sources,
as pandapower 3.3 does not do it.

pandapower adds the current sources' contribution to a single-phase fault as
it does to a three-phase one - from the positive-sequence network alone - and
whatever the zero sequence. IEC 60909-0:2016 scales it by
3 Z(1) / (2 Z(1) + Z(0)) at the fault: the sources drive the positive
sequence only, and the earth fault's current is the three sequences'
together. On an island of grid-forming inverters with no ground behind them
the difference is all of it: a 34.5 kV backbone with no zigzag gave 228 A of
"earth fault" (the EMT study, which builds the windings, gives 6.5 A of
cable capacitance); with its two zigzags, 249 A where EMT gives 427 A.

``install()`` wraps pandapower's ``_calc_ikss_1ph`` where ``calc_sc`` calls
it: after it, each faulted bus's current-source current is scaled by its own
factor; with one faulted bus, the sources' injections too, so the branch
currents carry the same share. Once, process-wide; idempotent.
"""
import importlib

import numpy as np

from pandapower.pypower.idx_bus_sc import IKCV, IKSS1, IKSS2, R_EQUIV, X_EQUIV

# The module, not the function the package exports under its name.
_calc_sc = importlib.import_module('pandapower.shortcircuit.calc_sc')


def current_source_factor(ppci, ppci_0, buses):
    """3 |Z(1)| / |2 Z(1) + Z(0)| at each bus (0 where its zero sequence has no path)."""
    z1 = ppci['bus'][buses, R_EQUIV] + 1j * ppci['bus'][buses, X_EQUIV]
    z0 = ppci_0['bus'][buses, R_EQUIV] + 1j * ppci_0['bus'][buses, X_EQUIV]
    with np.errstate(all='ignore'):
        f = 3.0 * np.abs(z1) / np.abs(2.0 * z1 + z0)
    return np.where(np.isfinite(f), np.clip(f, 0.0, 1.5), 0.0)


def _cap_at_converter_limits(net, ppci, ppci_0, buses, fault='1ph'):
    """
    An island of grid-forming PCS: each faulted bus's current no more than its
    inverters' current limit allows (pandapower_electrisim's
    _electrisim_pcs_1ph_limits, by bus) - the voltage sources' and the
    current sources' scaled alike, and the zero sequence's with them.
    """
    key = '_electrisim_pcs_1ph_islands' if fault == '1ph' else '_electrisim_pcs_islands'
    if not len(buses) or not (hasattr(net, 'get') and net.get(key)):
        return
    from pandapower_electrisim import electrisim_pcs_1ph_limits
    cap = {}
    for b, lim in electrisim_pcs_1ph_limits(net, fault).items():
        try:
            (i,) = np.atleast_1d(_calc_sc._get_is_ppci_bus(net, np.array([int(b)], dtype=np.int64)))
        except Exception:
            continue                                    # out of service, or not in this calculation
        cap[int(i)] = min(lim, cap.get(int(i), lim))
    for i in buses:
        lim = cap.get(int(i))
        if lim is None:
            continue
        total = np.nan_to_num(ppci['bus'][i, IKSS1]) + np.nan_to_num(ppci['bus'][i, IKSS2])
        if total > lim > 0:
            f = lim / total
            ppci['bus'][i, IKSS1] *= f
            ppci['bus'][i, IKSS2] *= f
            if ppci_0 is not None:
                ppci_0['bus'][i, IKSS1] *= f


def install():
    original = _calc_sc._calc_ikss_1ph
    if getattr(original, '_electrisim_iec_current_sources', False):
        return

    def _calc_ikss_1ph(net, ppci, ppci_0, bus_idx):
        original(net, ppci, ppci_0, bus_idx)
        buses = np.atleast_1d(bus_idx)
        _cap_at_converter_limits(net, ppci, ppci_0, buses)
        if not len(buses) or not np.any(np.nan_to_num(ppci['bus'][:, IKCV])):
            return
        f = current_source_factor(ppci, ppci_0, buses)
        ppci['bus'][buses, IKSS2] = ppci['bus'][buses, IKSS2] * f
        if len(buses) == 1:
            # The sources' injections, which the branch currents are built from: one fault, one factor.
            ppci['bus'][:, IKCV] = ppci['bus'][:, IKCV] * f[0]

    _calc_ikss_1ph._electrisim_iec_current_sources = True
    _calc_sc._calc_ikss_1ph = _calc_ikss_1ph

    # A line-to-line fault on a PCS island: capped where its inverters' worst phase reaches their limit.
    original_ikss = _calc_sc._calc_ikss

    def _calc_ikss(net, ppci, bus_idx):
        original_ikss(net, ppci, bus_idx)
        if net._options.get('fault') == '2ph':
            _cap_at_converter_limits(net, ppci, None, np.atleast_1d(bus_idx), '2ph')

    _calc_sc._calc_ikss = _calc_ikss
