"""
mgtwin's screening beside Electrisim's studies: the twin's gates on the drawn network ("Check design"), and the
twin's validity envelope as a label on a study run in one of the Site's operating states.

mgtwin is an optional dependency (requirements-mgtwin.txt): without it the check answers that it is not installed
and studies carry no label.

The label is about the Site's data and the state, not about mgtwin's own tier: mgtwin has no network solver and
marks voltages and EMT as outside its envelope, but here Electrisim is the solver. So every phenomenon is taken as
supported by the tier, and what remains is what the Site says - a state its gates fail for the study's phenomenon,
a site-wide data error, load beyond a state's sources, converter dynamics or current limits assumed rather than
identified, a protection study's verdict.
"""
from __future__ import annotations

import hashlib
import json

#: a study (the request's cell '0' typ) -> the envelope phenomenon its result rests on
_PHENOMENON = (
    (('PowerFlowPandaPower', 'PowerFlowOpenDss', 'OptimalPowerFlowPandaPower', 'AcDcOpfPandaPower',
      'StateEstimationPandaPower', 'ContingencyAnalysisPandaPower'), 'loading'),
    (('ShortCircuitPandaPower', 'ShortCircuitOpenDss', 'ShortCircuitAnsi', 'ProtectionCoordinationPandaPower',
      'ArcFlashPandaPower', 'PoiFaultStudy', 'DcFaultStudy', 'EmtStudy'), 'protection'),
    (('TransientStabilityAndes', 'EigenvalueAndes', 'MotorStartingPandaPower'), 'frequency'),
)

_envelopes = {}


def available():
    try:
        import mgtwin  # noqa: F401
        return True
    except ImportError:
        return False


def phenomenon_for(typ):
    for keys, ph in _PHENOMENON:
        if any(k in str(typ or '') for k in keys):
            return ph
    return None


def _site_model(site):
    from mgtwin.ontology import Site
    return Site.model_validate(site)


def check_design(diagram, site_id='DRAWN', s_base_mva=50.0, template=None):
    """The drawn network as a Site (mgtwin_site_io.payload_to_site) through mgtwin's seven gates."""
    import mgtwin
    from mgtwin.gates import Severity, run_all
    from mgtwin_site_io import payload_to_site
    site = payload_to_site(diagram, site_id, s_base_mva, template=template)
    report = run_all(_site_model(site))
    findings = [{'gate': f.gate, 'severity': f.severity.value, 'scope': f.scope, 'message': f.message,
                 'remedy': f.remedy} for f in report.findings if f.severity != Severity.PASS]
    by_gate = {}
    for f in report.findings:
        by_gate.setdefault(f.gate, {'fail': 0, 'warn': 0, 'pass': 0})[f.severity.value] += 1
    return {'site_id': site['id'], 'mgtwin_version': mgtwin.__version__,
            'states': [i['id'] for i in site['islands']],
            'counts': {s.value: len(report.by_severity(s)) for s in Severity},
            'by_gate': by_gate, 'findings': findings}


def _envelope(site):
    from mgtwin.envelope import Support, build_envelope
    key = hashlib.sha1(json.dumps(site, sort_keys=True).encode()).hexdigest()
    env = _envelopes.get(key)
    if env is None:
        env = build_envelope(_site_model(site))
        # Electrisim is the solver: no phenomenon is outside the tier, and the tier's own notes are not the Site's
        env.phenomena = {p: (Support.SUPPORTED, '') for p in env.phenomena}
        if len(_envelopes) > 16:
            _envelopes.clear()
        _envelopes[key] = env
    return env


def study_label(site, state_id, typ):
    """(verdict, reasons) for a study of `typ` run in `state_id` of `site`; None when the study maps to no
    phenomenon or mgtwin is not installed."""
    ph = phenomenon_for(typ)
    if ph is None or not available():
        return None
    verdict, why = _envelope(site).check(state_id, ph)
    return verdict.value, _condense([w for w in why if w])


def _condense(reasons):
    """One line for the sources sharing a reason ("PCS1: ... assumed", "PCS2: ... assumed" -> "PCS1, PCS2: ...")."""
    out, by_tail = [], {}
    for r in reasons:
        head, sep, tail = r.partition(': ')
        if sep and ' ' not in head and tail in ('dynamic parameters assumed, not identified',):
            if tail not in by_tail:
                by_tail[tail] = []
                out.append(tail)
            by_tail[tail].append(head)
        else:
            out.append(r)
    return [f"{', '.join(by_tail[r])}: {r}" if r in by_tail else r for r in out]
