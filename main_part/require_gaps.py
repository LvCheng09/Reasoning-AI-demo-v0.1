"""Read-only Require diagnostics. Missing evidence is never a false predicate.

Repair routes belong to the target rule's metadata.require_recovery contract;
this inspector cannot invent source facts or executable knowledge.
"""
from copy import deepcopy
from .rule_execution_contract import digest
from .matcher.matching import match_required_bundle_against_graph
from .matcher.routing import graph_bundle_from_spec, required_bundle_from_rule


def inspect_require_gaps(rules, memory, event_ids, *, target_ids, source_text='', attempts=None):
    if not target_ids:
        return {'kind': 'none', 'gaps': []}
    graph = graph_bundle_from_spec({'type': 'event_workspace', 'event_ids': event_ids}, {}, memory)
    attempts = attempts if attempts is not None else {}
    gaps = []
    for rule in rules:
        if rule['rule_id'] not in target_ids or rule.get('rule_layer') == 'control':
            continue
        match = match_required_bundle_against_graph(required_bundle_from_rule(rule, {}), graph, {})
        bound_ids = {s.get('candidate_span_id')
            for outcome in match.get('metadata', {}).get('per_required', [])
            for s in outcome.get('details', {}).get('slot_matches', []) if s.get('matched')}
        conflicts = [deepcopy(c) for eid in event_ids
            for span in memory.get_event_workspace(eid).get('spans', []) if span.get('span_id') in bound_ids
            for c in span.get('metadata', {}).get('require_conflicts', []) if not c.get('resolved')]
        if conflicts:
            fingerprint = digest([rule['rule_id'], conflicts])
            gaps.append({'gap_id': fingerprint, 'rule_id': rule['rule_id'], 'required_id': 'source_conflict',
                'kind': 'conflict', 'question': '适用信息源给出了相互冲突的值，请核实来源及适用范围；当前证据保留，不直接覆盖。',
                'match_evidence': {'source_conflicts': conflicts}, 'repair_rule_ids': [], 'attempts': 0})
            continue
        if match['ok']:
            return {'kind': 'none', 'gaps': [], 'executable_alternative': rule['rule_id']}
        for index, outcome in enumerate(match.get('metadata', {}).get('per_required', [])):
            if outcome.get('matched'):
                continue
            detail = outcome.get('details', {})
            required = rule['required'][index]
            rid = required.get('required_id', str(index))
            kind = 'unknown'
            shared = [s for s in detail.get('shared_semantic_relation_constraint_matches', []) if not s.get('matched')]
            slots = detail.get('slot_matches', [])
            bound = bool(slots) and all(s.get('matched') for s in slots)
            if shared:
                kind = 'conflict' if any(s.get('candidate_counts') and all(n > 0 for n in s['candidate_counts'].values()) for s in shared) else 'missing_relation'
            elif bound and any(p.get('status') != 'error' and p.get('matched') is False for p in detail.get('predicate_matches', [])):
                kind = 'condition_false'
            recipe = rule.get('metadata', {}).get('require_recovery', {}).get(rid, {})
            routes = recipe.get('repair_rule_ids', [])
            if not isinstance(routes, list) or any(not isinstance(r, str) or not r for r in routes):
                raise ValueError('Invalid recovery rule IDs')
            if routes and any(rid not in {r['rule_id'] for r in rules} for rid in routes):
                raise ValueError('Recovery references an unknown rule')
            if routes and kind not in {'conflict', 'condition_false'}:
                kind = recipe.get('kind')
                if kind not in {'missing_relation', 'derivation'}:
                    raise ValueError('Recovery routes must declare missing_relation or derivation')
            elif kind == 'unknown' and source_text:
                # An extraction request is a check, not a claim that the text contains the answer.
                kind = 'reextract'
            evidence = deepcopy(detail)
            fingerprint = digest([rule['rule_id'], rid, evidence, source_text])
            tried = attempts.get(fingerprint, 0)
            if tried and kind in {'missing_relation', 'derivation', 'reextract'}:
                kind = 'unknown'
                routes = []
            question = recipe.get('question') or (
                f"请核实规则 {rule['rule_id']} 的条件 {rid}："
                f"{required.get('condition_text') or [s.get('feature_type', s.get('span_id')) for s in required.get('condition', [])]}。"
                '请给出来源及适用对象；缺少的信息请回答未知，不要推测。')
            gaps.append({'gap_id': fingerprint, 'rule_id': rule['rule_id'], 'required_id': rid,
                'kind': kind, 'question': question, 'match_evidence': evidence,
                'repair_rule_ids': list(routes), 'attempts': tried})
    # A false requirement blocks its rule, not another viable alternative.
    blocked = {g['rule_id'] for g in gaps if g['kind'] in {'conflict', 'condition_false'}}
    viable = [g for g in gaps if g['rule_id'] not in blocked]
    order = {'conflict': 0, 'condition_false': 1, 'missing_relation': 2, 'derivation': 3, 'reextract': 4, 'unknown': 5}
    gaps.sort(key=lambda g: order[g['kind']])
    if not gaps:
        return {'kind': 'none', 'gaps': []}
    selected = min(viable, key=lambda g: order[g['kind']]) if viable else gaps[0]
    kind = selected['kind']
    if kind == 'missing_relation' and not selected['repair_rule_ids']:
        kind = 'unknown'
    return {**selected, 'kind': kind, 'gaps': gaps,
        'source_text': source_text, 'target_rule_ids': list(target_ids)}


def install_recovery_branches(entry):
    """Extend an ordinary Control coordinator; keep existing strategy branches."""
    entry = deepcopy(entry)
    decision = entry['produced'][0]
    decision['before_decide'] = [op for op in decision.get('before_decide', [])
        if op.get('operation') != 'inspect_require_gaps']
    decision['before_decide'].append({'operation': 'inspect_require_gaps', 'policy': {'max_attempts_per_evidence': 1}})
    decision['branches'] = [b for b in decision['branches']
        if not b.get('policy_id', '').startswith('require_recovery_')]
    branches = [{'policy_id': 'require_recovery_resume',
        'when': {'path': 'workflow.require_gaps.resume_rule_ids', 'operator': 'truthy'},
        'then': {'operation': 'run_rule', 'rule_ids': '$context.workflow.require_gaps.resume_rule_ids'}}]
    for kind in ('missing_relation', 'derivation'):
        branches.append({'policy_id': 'require_recovery_' + kind,
            'when': {'path': 'workflow.require_gaps.kind', 'operator': 'eq', 'value': kind},
            'then': {'operation': 'run_rule', 'rule_ids': '$context.workflow.require_gaps.repair_rule_ids'}})
    branches.append({'policy_id': 'require_recovery_reextract',
        'when': {'path': 'workflow.require_gaps.kind', 'operator': 'eq', 'value': 'reextract'},
        'then': {'operation': 'request_frontend_patch', 'request_id': '$context.workflow.require_gaps.gap_id',
            'question': '$context.workflow.require_gaps.question', 'source_text': '$context.workflow.require_gaps.source_text',
            'missing_span_query': '$context.workflow.require_gaps.question',
            'missing_requirements': '$context.workflow.require_gaps.gaps'}})
    for kind in ('unknown', 'conflict', 'condition_false'):
        branches.append({'policy_id': 'require_recovery_' + kind,
            'when': {'path': 'workflow.require_gaps.kind', 'operator': 'eq', 'value': kind},
            'then': {'operation': 'stop_path', 'status': 'require_' + kind,
                'reason': '$context.workflow.require_gaps.question',
                'task_result': {'status': 'blocked' if kind == 'condition_false' else 'pending',
                    'reason': kind, 'request_id': '$context.workflow.require_gaps.gap_id',
                    'question': '$context.workflow.require_gaps.question',
                    'gaps': '$context.workflow.require_gaps.gaps'}}})
    decision['branches'] = branches + decision['branches']
    return entry


def bind_knowledge_context(memory, event_ids, knowledge, *, control_rule_id):
    """Instantiate Knowledge's scoped inheritance rule using observed contexts.

    Knowledge supplies relation types, never missing currencies/period values.
    Only an explicitly scoped source context may supply a relation's nature.
    """
    from .vector_contracts import rescale_cosine_strict
    contract = knowledge.get('metadata', {}).get('context_binding')
    if not isinstance(contract, dict) or contract.get('scope') != 'explicit_applies_to':
        raise ValueError('Knowledge must declare explicit context inheritance scope')
    approved = contract.get('relation_types', [])
    if not approved:
        raise ValueError('Knowledge must declare allowed relation types')
    changed, conflicts = [], []
    staged = {eid: deepcopy(memory.event_workspaces[eid]) for eid in event_ids}
    for event_id in event_ids:
        workspace = staged[event_id]
        index = {s['span_id']: s for s in workspace.get('spans', [])}
        for source in list(index.values()):
            targets = source.get('metadata', {}).get('applies_to', [])
            if not targets:
                continue
            if not isinstance(targets, list) or any(t not in index for t in targets):
                raise ValueError('Observed context references an unknown target')
            for relation in source.get('semantic_relations', []):
                def same_type(a, b):
                    return a.get('embedding_model') == b.get('embedding_model') and a.get('vector_version') == b.get('vector_version') and all(
                        (result := rescale_cosine_strict(a.get(field), b.get(field))).get('ok') and result['score'] >= .995
                        for field in ('core_vector', 'relation_vector'))
                if not any(same_type(relation, spec) for spec in approved):
                    continue
                if not relation.get('nature_vector'):
                    raise ValueError('Observed context needs an explicit vectorized value')
                for target_id in targets:
                    target = index[target_id]
                    existing = [r for r in target.get('semantic_relations', []) if same_type(r, relation)]
                    if existing:
                        if any(not (v := rescale_cosine_strict(r.get('nature_vector'), relation['nature_vector'])).get('ok') or v['score'] < .995 for r in existing):
                            conflict = {'target_span_id': target_id, 'source_span_id': source['span_id'],
                                'knowledge_id': knowledge.get('knowledge_id', knowledge.get('id')),
                                'candidate': deepcopy(relation), 'existing': deepcopy(existing)}
                            conflicts.append(conflict)
                            retained = target.setdefault('metadata', {}).setdefault('require_conflicts', [])
                            if conflict not in retained:
                                retained.append(deepcopy(conflict))
                        continue
                    result = deepcopy(relation)
                    result['provenance'] = {'knowledge_id': knowledge.get('knowledge_id', knowledge.get('id')),
                        'source_span_id': source['span_id'], 'control_rule_id': control_rule_id}
                    target.setdefault('semantic_relations', []).append(result)
                    changed.append({'target_span_id': target_id, 'source_span_id': source['span_id']})
    # Validate every scope first; a late invalid source cannot leave half a repair.
    for eid, workspace in staged.items():
        if workspace != memory.event_workspaces[eid]:
            workspace['version'] = int(workspace.get('version', 0)) + 1
            memory.event_workspaces[eid] = workspace
    return {'changed': changed, 'conflicts': conflicts}


def recovery_control_network(entry):
    """Build a callable three-level network from the declared recovery policy.

    The entry only inspects and delegates. Route rules select a capability;
    capability rules recheck the context before performing their one action.
    Existing run_rule dispatch, matching and execution enforce each hop.
    """
    expanded = install_recovery_branches(entry)
    decision = expanded['produced'][0]
    recovery = [b for b in decision['branches'] if b.get('policy_id', '').startswith('require_recovery_')]
    originals = [b for b in decision['branches'] if b not in recovery]
    children, router_branches = [], []

    def node(rid, level, branches):
        return {'rule_id': rid, 'rule_layer': 'control', 'domain': 'general', 'status': 'active',
            'required': [], 'scope_tags': ['require_recovery_network', 'control_' + level],
            'metadata': {'control_level': level, 'runtime_control_rule': True},
            'produced': [{'operation': 'control_decide', 'branches': branches,
                'default': {'operation': 'stop_path', 'status': 'inapplicable_control',
                    'reason': 'Recovery context does not satisfy this capability precondition'}}]}

    for branch in recovery:
        name = branch['policy_id'].removeprefix('require_recovery_')
        leaf_id, route_id = 'control_require_action_' + name, 'control_require_route_' + name
        children.append(node(leaf_id, 'low', [deepcopy(branch)]))
        middle = {**deepcopy(branch), 'then': {'operation': 'run_rule', 'rule_ids': [leaf_id]}}
        children.append(node(route_id, 'middle', [middle]))
        router_branches.append({**deepcopy(branch), 'then': {'operation': 'run_rule', 'rule_ids': [route_id]}})
    router_id = 'control_require_recovery_router'
    children.append(node(router_id, 'high', router_branches))
    decision['branches'] = [{'policy_id': 'require_recovery_network',
        'when': {'any': [deepcopy(b['when']) for b in recovery]},
        'then': {'operation': 'run_rule', 'rule_ids': [router_id]}}] + originals
    return expanded, children
