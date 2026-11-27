"""
A parameter sweep of any study: the same request run for each combination of values, one table back.

The sweep's parameter row (typ 'ParameterSweep Parameters'):
- study_typ: the study's parameter row's typ, say 'EmtStudy Parameters';
- study_params: that row's settings (a dict or JSON);
- axes: a list of {'element': a drawn element's name, or '@study' for the study's settings, 'field': its field,
  'values': [...]} (a dict or JSON); every combination runs;
- metrics: dotted paths into each run's result to report, say 'emt.series_ups.0.v_critical_min_pu';
- max_runs: refused above this (default 200): a sweep costs a run per combination.

Each run goes through the application's own entry point, so it is that study exactly, with its checks. A run that
fails is reported in its row and the others go on. Runs are in series.
"""
import copy
import itertools
import json


def _parsed(value, default):
    if value in (None, ''):
        return default
    return json.loads(value) if isinstance(value, str) else value


def _path(result, path):
    node = result
    for part in str(path).split('.'):
        try:
            node = node[int(part)] if isinstance(node, list) else node[part]
        except (KeyError, IndexError, ValueError, TypeError):
            return None
    return node


def run(app, in_data, key):
    params = in_data[key]
    try:
        axes = _parsed(params.get('axes'), [])
        metrics = _parsed(params.get('metrics'), [])
        study_params = _parsed(params.get('study_params'), {})
    except ValueError as exc:
        return {'error': True, 'message': f'A sweep setting is not valid JSON: {exc}'}
    study_typ = str(params.get('study_typ') or '').strip()
    if not study_typ:
        return {'error': True, 'message': 'A sweep needs study_typ: the typ of the study to run.'}
    if 'ParameterSweep' in study_typ:
        return {'error': True, 'message': 'A sweep cannot run a sweep.'}
    if not axes or not metrics:
        return {'error': True, 'message': 'A sweep needs axes and metrics.'}
    names = {v.get('name') for v in in_data.values() if isinstance(v, dict)}
    for axis in axes:
        if axis.get('element') != '@study' and axis.get('element') not in names:
            return {'error': True, 'message': f"No element '{axis.get('element')}' to sweep."}
        if not axis.get('field') or not axis.get('values'):
            return {'error': True, 'message': 'Each axis needs its field and its values.'}
    total = 1
    for axis in axes:
        total *= len(axis['values'])
    limit = int(params.get('max_runs') or 200)
    if total > limit:
        return {'error': True, 'message': f'{total} runs is over the limit of {limit}: raise max_runs if that is meant.'}
    client = app.test_client()
    rows = []
    for combination in itertools.product(*[axis['values'] for axis in axes]):
        request = copy.deepcopy(in_data)
        request[key] = {'typ': study_typ, 'user_email': params.get('user_email', ''), **{k: v for k, v in study_params.items()}}
        for axis, value in zip(axes, combination):
            if axis['element'] == '@study':
                request[key][axis['field']] = str(value)
            else:
                next(v for v in request.values() if isinstance(v, dict) and v.get('name') == axis['element'])[axis['field']] = str(value)
        row = {'values': {f"{a['element']}.{a['field']}": v for a, v in zip(axes, combination)}}
        try:
            response = client.post('/', json=request)
            text = response.get_data(as_text=True).replace('-Infinity', 'null').replace('Infinity', 'null').replace('NaN', 'null')
            result = json.loads(text)
            if response.status_code != 200 or result.get('error'):
                row['error'] = str(result.get('message') or result.get('error') or response.status_code)[:300]
            else:
                row['metrics'] = {m: _path(result, m) for m in metrics}
                if result.get('warnings'):
                    row['warnings'] = len(result['warnings'])
        except Exception as exc:   # noqa: BLE001
            row['error'] = str(exc)[:300]
        rows.append(row)
    summary = {}
    for m in metrics:
        values = [r['metrics'][m] for r in rows if isinstance(r.get('metrics', {}).get(m), (int, float)) and not isinstance(r['metrics'][m], bool)]
        if values:
            summary[m] = {'min': min(values), 'max': max(values)}
    return {'error': False, 'n': len(rows), 'n_failed': sum(1 for r in rows if 'error' in r), 'rows': rows, 'summary': summary}
