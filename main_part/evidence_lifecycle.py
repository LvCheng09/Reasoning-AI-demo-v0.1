"""Versioned derivation evidence; decisions and refresh scheduling belong to Control."""
from copy import deepcopy
import hashlib
import json

from .alignment_contract import alignment_is_current
from .control_state import control_state_span_is_active


class EvidenceSpanIndex(dict):
    """Matcher-local cache, never fields injected into source evidence."""
    def __init__(self, spans, validity):
        super().__init__(spans)
        self.validity = validity


def evidence_fingerprint(span):
    # Audit timestamps/proof bookkeeping do not change the proposition itself.
    value = deepcopy(span)
    for key in ('computed_at_step_id', 'derived_at_step_id', 'computed_by_rule_id', 'derived_by_rule_id',
                'computation', 'runtime_value_resolution', 'expression', 'calculation', 'value_policy', 'result_unit', '_evidence_checked',
                '_invalid_derivation', '_invalid_alignment', '_inactive_control_state'):
        value.pop(key, None)
    metadata = dict(value.get('metadata') or {})
    for key in ('derivation_supports', 'derived_from_span_ids', 'evidence_active',
                'invalidated_by_control_rule_id', 'invalidated_reason', 'rebound_by_control_rule_id', 'measurement_basis'):
        metadata.pop(key, None)
    value['metadata'] = metadata
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def evidence_validity(spans, relations=None):
    """All premises (including predicates), recursively, not just numeric operands."""
    memo, fingerprints = {}, {}
    def valid(sid, trail=()):
        if sid in memo:
            return memo[sid]
        if sid in trail or sid not in spans:
            return False
        span = spans[sid]
        metadata = span.get('metadata') or {}
        alignment = metadata.get('semantic_alignment') or {}
        if alignment and any(not valid(parent, (*trail, sid)) for parent in alignment.get('source_fingerprints', {})):
            memo[sid] = False
            return False
        if (metadata.get('evidence_active') is False or span.get('status') in {'invalid', 'retracted', 'retired'}
                or not control_state_span_is_active(span) or not alignment_is_current(span, spans, relations)):
            memo[sid] = False
            return False
        supports = metadata.get('derivation_supports')
        if supports is None:
            # Observations/alignment have their own identity contract. A current
            # Executor derivation always carries a complete support record.
            memo[sid] = True
            return True
        for support in supports:
            sources = support.get('source_fingerprints') or {}
            if not sources or sid in sources:
                continue
            okay = True
            for parent, expected in sources.items():
                if not valid(parent, (*trail, sid)):
                    okay = False
                    break
                if parent not in fingerprints:
                    fingerprints[parent] = evidence_fingerprint(spans[parent])
                if fingerprints[parent] != expected:
                    okay = False
                    break
            if okay and (relations is None or all(r in relations for r in support.get('source_relations', []))):
                memo[sid] = True
                return True
        memo[sid] = False
        return False
    for sid in spans:
        valid(sid)
    return memo


def span_evidence_is_current(span, spans, relations=None):
    if span.get('_invalid_derivation'):
        return False
    if span.get('span_id') in getattr(spans, 'validity', {}):
        return spans.validity[span['span_id']]
    return evidence_validity(spans, relations).get(span.get('span_id'), False)


def attach_derivation(item, *, workspace, bindings, rule_id, step_id):
    from .matcher.routing import build_span_index
    index = build_span_index(workspace)
    sources = sorted(set(bindings.values()))
    if not sources or any(sid not in index for sid in sources):
        raise ValueError('A derived conclusion requires all actual matched premises')
    validity = evidence_validity(index, workspace.get('relations', []))
    if any(not validity[sid] for sid in sources):
        raise ValueError('Cannot derive a conclusion from stale evidence')
    relations = [deepcopy(r) for r in workspace.get('relations', [])
                 if r.get('source_span_id') in sources and r.get('target_span_id') in sources]
    proof = {'schema': 'derivation_support.v1', 'rule_id': rule_id, 'step_id': step_id,
             'slot_bindings': deepcopy(bindings),
             'source_fingerprints': {sid: evidence_fingerprint(index[sid]) for sid in sources},
             'source_relations': relations}
    metadata = dict(item.get('metadata') or {})
    item.update(status='active', _invalid_derivation=False)
    previous = index.get(item['span_id'])
    supports = []
    if previous and validity.get(item['span_id']) and evidence_fingerprint(previous) == evidence_fingerprint(item):
        supports = deepcopy((previous.get('metadata') or {}).get('derivation_supports') or [])
    supports.append(proof)
    # Preserve distinct justified alternatives, not duplicate executions.
    unique = {}
    for support in supports:
        key = json.dumps({k: v for k, v in support.items() if k != 'step_id'}, sort_keys=True)
        unique[key] = support
    metadata.update(derivation_supports=list(unique.values()), derived_from_span_ids=sources, evidence_active=True)
    item.update(metadata=metadata, status='active', _invalid_derivation=False)


def review_event_evidence(memory, event_ids, *, control_rule_id, policy):
    if policy.get('invalidate_stale') is not True or policy.get('retain_history') is not True:
        raise ValueError('Evidence review requires explicit invalidation and history policies')
    invalidated, rebound, stagnant = [], [], []
    for event_id in event_ids:
        workspace = memory.get_event_workspace(event_id)
        index = {s['span_id']: s for s in workspace.get('spans', [])}
        validity = evidence_validity(index, workspace.get('relations', []))
        metadata = workspace.setdefault('metadata', {})
        history = metadata.setdefault('derivation_history', [])
        for sid, span in index.items():
            smeta = span.get('metadata') or {}
            if (smeta.get('derivation_supports') is not None or smeta.get('semantic_alignment')) and not validity[sid] and smeta.get('evidence_active') is not False:
                history.append({'span': deepcopy(span), 'workspace_version': workspace.get('version', 0),
                                'control_rule_id': control_rule_id, 'reason': 'premise_changed_or_unavailable'})
                span['status'] = 'retired'
                span['metadata'] = {**smeta, 'evidence_active': False,
                                    'invalidated_by_control_rule_id': control_rule_id,
                                    'invalidated_reason': 'premise_changed_or_unavailable'}
                invalidated.append({'event_id': event_id, 'span_id': sid})
            elif smeta.get('derivation_supports') and validity[sid]:
                usable = next((p for p in smeta['derivation_supports']
                               if all(parent in index and validity[parent] and evidence_fingerprint(index[parent]) == expected
                                      for parent, expected in p['source_fingerprints'].items())
                               and all(r in workspace.get('relations', []) for r in p.get('source_relations', []))), None)
                if usable and set(smeta.get('derived_from_span_ids', [])) != set(usable['source_fingerprints']):
                    history.append({'span': deepcopy(span), 'workspace_version': workspace.get('version', 0),
                                    'control_rule_id': control_rule_id, 'reason': 'retain_independently_supported_conclusion'})
                    smeta.update(derived_from_span_ids=sorted(usable['source_fingerprints']), rebound_by_control_rule_id=control_rule_id)
                    span['metadata'] = smeta
                    rebound.append({'event_id': event_id, 'span_id': sid})
        current = evidence_validity(index, workspace.get('relations', []))
        signature = hashlib.sha256(json.dumps(sorted((sid, evidence_fingerprint(s)) for sid, s in index.items()
                     if current[sid] and s.get('span_role', s.get('role')) != 'control_state'), sort_keys=True).encode()).hexdigest()
        prior = metadata.get('evidence_review') or {}
        count = int(prior.get('stagnant_rounds', 0)) + 1 if prior.get('signature') == signature else 0
        metadata['evidence_review'] = {'signature': signature, 'stagnant_rounds': count,
                                      'control_rule_id': control_rule_id}
        if any(i['event_id'] == event_id for i in invalidated + rebound):
            workspace['version'] = int(workspace.get('version', 0)) + 1
        stagnant.append(count)
    return {'invalidated': invalidated, 'rebound': rebound, 'stagnant_rounds': min(stagnant, default=0),
            'control_rule_id': control_rule_id, 'history_retained': True}


def inspect_rule_frontier(rules, memory, event_ids, *, max_rules):
    """Read actual applicability without simulating values or deciding to stop."""
    if max_rules < 1:
        raise ValueError('Rule inspection budget must be positive')
    from .matcher.routing import graph_bundle_from_spec, required_bundle_from_rule
    from .matcher.matching import match_required_bundle_against_graph
    candidates = [r for r in rules if r.get('rule_layer') not in {'control', 'matcher_rule', 'auto_learning'}
                  and r.get('status', 'active') == 'active']
    graph = graph_bundle_from_spec({'type': 'working_memory', 'event_ids': event_ids,
                                   'zones': ['event_workspace']}, {}, memory)
    facts = set()
    for event_id in event_ids:
        w = memory.get_event_workspace(event_id)
        index = {s['span_id']: s for s in w.get('spans', [])}
        validity = evidence_validity(index, w.get('relations', []))
        facts.update(s.get('feature_type') for sid, s in index.items() if validity[sid])
    executable, evidence, unsupported = [], [], []
    for r in candidates[:max_rules]:
        actions = r.get('produced') or []
        outputs = {s.get('feature_type') for a in actions if a.get('operation') == 'add' for s in a.get('conclusion', [])}
        supported = bool(outputs) and all(a.get('operation') == 'add' for a in actions)
        if not supported:
            if all(a.get('operation') == 'output' for a in actions):
                continue
            unsupported.append(r['rule_id'])
            continue
        match = match_required_bundle_against_graph(required_bundle_from_rule(r, {}), graph, {})
        if match['ok'] and outputs - facts:
            executable.append(r['rule_id'])
        evidence.append({'rule_id': r['rule_id'], 'strict_match': bool(match['ok']),
                         'new_feature_types': sorted(outputs - facts),
                         'match_evidence': match.get('metadata', {}).get('per_required', [])})
    kind = ('progress_available' if executable else 'unexamined_routes'
            if len(candidates) > max_rules or unsupported else 'frontier_exhausted')
    return {'kind': kind, 'executable_rule_ids': executable, 'inspected_count': len(evidence),
            'unsupported_rule_ids': unsupported, 'evidence': evidence,
            'scope': 'current_strict_evidence_and_declared_rule_inventory_not_global_unsolvability'}
