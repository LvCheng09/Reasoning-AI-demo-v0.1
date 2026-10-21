from copy import deepcopy
import pytest

from main_part.data_structure import WorkingMemoryState
from main_part.evidence_lifecycle import attach_derivation, evidence_validity, review_event_evidence, evidence_fingerprint
from main_part.executor_new import execute_rule_chain
from main_part.rule_selector import child_rule_selectors
from main_part.scheduler_new import _runtime_failure_points

POLICY = {'invalidate_stale': True, 'retain_history': True}


def workspace():
    wm = WorkingMemoryState()
    wm.event_workspaces['e'] = {'event_id': 'e', 'version': 0, 'spans': [
        {'span_id': 'x', 'value': 10, 'unit': 'USD', 'feature_type': 'amount'},
        {'span_id': 'approval', 'value': 1, 'feature_type': 'approval'}], 'relations': []}
    return wm


def derive(wm, sid, sources, value=10):
    w = wm.get_event_workspace('e')
    item = {'span_id': sid, 'feature_type': sid, 'value': value, 'unit': 'USD'}
    attach_derivation(item, workspace=w, bindings={s: s for s in sources}, rule_id='r_' + sid, step_id='step')
    wm.apply_event_workspace_operation('e', {'operation': 'update', 'id': sid, 'item': item})
    return item


def test_change_propagates_and_preserves_native_history():
    wm = workspace()
    derive(wm, 'middle', ['x'])
    derive(wm, 'answer', ['middle'])
    wm.apply_event_workspace_operation('e', {'operation': 'update', 'id': 'x', 'item': {'value': 20}})
    report = review_event_evidence(wm, ['e'], control_rule_id='control', policy=POLICY)
    assert {r['span_id'] for r in report['invalidated']} == {'middle', 'answer'}
    assert len(wm.event_workspaces['e']['metadata']['derivation_history']) == 2
    derive(wm, 'middle', ['x'], 20)
    derive(wm, 'answer', ['middle'], 20)
    spans = {s['span_id']: s for s in wm.event_workspaces['e']['spans']}
    assert evidence_validity(spans)['answer']
    assert spans['answer']['value'] == 20


def test_predicate_premises_are_recorded_by_real_executor():
    wm = workspace()
    rule = {'rule_id': 'cost', 'rule_layer': 'domain', 'produced': [{
        'operation': 'add', 'event_id': 'e', 'conclusion': [{'span_id': 'answer',
            'expression': {'operator': 'mean', 'operands': ['x']}, 'result_unit': 'inherit:x'}]}]}
    match = {'rule': rule, 'match': {'ok': True, 'slot_bindings': {'x': 'x', 'approval': 'approval'}}}
    result = execute_rule_chain(matched_rules=[match], working_memory=wm, session_id='test')
    assert result['ok']
    answer = next(s for s in wm.event_workspaces['e']['spans'] if s['span_id'] == 'answer')
    assert set(answer['metadata']['derived_from_span_ids']) == {'x', 'approval'}
    wm.apply_event_workspace_operation('e', {'operation': 'update', 'id': 'approval', 'item': {'value': 0}})
    index = {s['span_id']: s for s in wm.event_workspaces['e']['spans']}
    assert not evidence_validity(index)['answer']


def test_independent_support_survives_and_rebinds_without_fake_recomputation():
    wm = workspace()
    wm.apply_event_workspace_operation('e', {'operation': 'add', 'item': {'span_id': 'y', 'value': 10, 'unit': 'USD', 'feature_type': 'amount'}})
    derive(wm, 'answer', ['x'])
    derive(wm, 'answer', ['y'])
    answer = next(s for s in wm.event_workspaces['e']['spans'] if s['span_id'] == 'answer')
    assert len(answer['metadata']['derivation_supports']) == 2
    wm.apply_event_workspace_operation('e', {'operation': 'delete', 'id': 'y'})
    report = review_event_evidence(wm, ['e'], control_rule_id='control', policy=POLICY)
    assert not report['invalidated'] and report['rebound']
    assert answer['value'] == 10 and answer['metadata']['derived_from_span_ids'] == ['x']


def test_unrelated_fact_change_does_not_invalidate_answer():
    wm = workspace()
    derive(wm, 'answer', ['x'])
    wm.apply_event_workspace_operation('e', {'operation': 'update', 'id': 'approval', 'item': {'value': 0}})
    assert not review_event_evidence(wm, ['e'], control_rule_id='control', policy=POLICY)['invalidated']


def test_subject_period_and_relations_are_real_evidence():
    wm = workspace()
    w = wm.event_workspaces['e']
    w['relations'] = [{'relation_id': 'rel', 'source_span_id': 'x', 'target_span_id': 'approval'}]
    derive(wm, 'answer', ['x', 'approval'])
    index = {s['span_id']: s for s in w['spans']}
    assert not evidence_validity(index, [])['answer']
    for field in ('entity_id', 'period'):
        changed = deepcopy(index)
        changed['x'][field] = 'different'
        assert not evidence_validity(changed, w['relations'])['answer']


def test_noop_reexecution_is_not_progress():
    wm = workspace()
    derive(wm, 'answer', ['x'])
    a = review_event_evidence(wm, ['e'], control_rule_id='control', policy=POLICY)
    derive(wm, 'answer', ['x'])
    b = review_event_evidence(wm, ['e'], control_rule_id='control', policy=POLICY)
    assert a['stagnant_rounds'] == 0 and b['stagnant_rounds'] == 1


def test_intermediate_fact_is_not_an_executor_failure():
    result = {'steps': [{'operation': 'add', 'status': 'completed', 'rule_id': 'r'}], 'status': 'completed'}
    assert not _runtime_failure_points(None, [result])
    result['steps'][0]['status'] = 'error'
    assert _runtime_failure_points(None, [result])


def test_hierarchy_trace_is_bounded_and_contains_ancestors():
    first = child_rule_selectors({'rule_selector': {'rule_layer': 'control'}}, rule_match={
        'rule': {'rule_id': 'parent', 'rule_layer': 'control'}})[0]
    second = child_rule_selectors({'rule_selector': {'rule_layer': 'domain'}}, rule_match={
        'rule': {'rule_id': 'child', 'rule_layer': 'control'}, 'selector_context': first})[0]
    assert second['search_trace']['hop'] == 2
    assert second['search_trace']['control_rule_ids'] == ['parent', 'child']


def test_policy_must_explicitly_authorize_maintenance():
    with pytest.raises(ValueError):
        review_event_evidence(workspace(), ['e'], control_rule_id='control', policy={})


def test_audit_timestamps_do_not_change_semantic_fingerprint():
    a = {'span_id': 'x', 'value': 10}
    b = {**a, 'computed_at_step_id': 'another', 'metadata': {'derived_from_span_ids': ['z']}}
    assert evidence_fingerprint(a) == evidence_fingerprint(b)


def test_changing_control_observations_are_not_business_progress():
    wm = workspace()
    w = wm.get_event_workspace('e')
    w['spans'].append({'span_id': 'voc', 'role': 'control_state', 'value': 1})
    review_event_evidence(wm, ['e'], control_rule_id='c', policy=POLICY)
    w['spans'][-1]['value'] = 2
    assert review_event_evidence(wm, ['e'], control_rule_id='c', policy=POLICY)['stagnant_rounds'] == 1


def test_control_preparation_refreshes_context_before_branch_selection():
    wm = workspace()
    calls = []
    def handler(op):
        calls.append(op)
        return {'workflow': {'reviewed': True}}
    rule = {'rule_id': 'c', 'rule_layer': 'control', 'produced': [{
        'operation': 'control_decide', 'before_decide': [{'operation': 'review_event_evidence', 'policy': POLICY}],
        'branches': [{'when': {'path': 'workflow.reviewed', 'operator': 'eq', 'value': True},
                      'then': {'operation': 'stop_path', 'status': 'reviewed'}}],
        'default': {'operation': 'stop_path', 'status': 'not_reviewed'}}]}
    result = execute_rule_chain(matched_rules=[{'rule': rule, 'match': {'ok': True}}], working_memory=wm,
        session_id='preparation', control_operation_handler=handler)
    assert result['ok'] and result['stopped_paths'][0]['status'] == 'reviewed'
    assert calls[0]['rule_id'] == 'c'
    rejected = execute_rule_chain(matched_rules=[{'rule': rule, 'match': {'ok': True}}], working_memory=workspace(),
        session_id='missing_preparation')
    assert not rejected['ok']
    assert 'operation handler' in str(rejected)


def test_output_cannot_render_a_retired_conclusion():
    wm = workspace()
    derive(wm, 'answer', ['x'])
    wm.apply_event_workspace_operation('e', {'operation': 'delete', 'id': 'x'})
    review_event_evidence(wm, ['e'], control_rule_id='c', policy=POLICY)
    rule = {'rule_id': 'release', 'rule_layer': 'domain', 'produced': [{
        'operation': 'output', 'answer_mode': 'event_span', 'event_id': 'e', 'conclusion': [{'span_id': 'answer'}]}]}
    result = execute_rule_chain(matched_rules=[{'rule': rule, 'match': {'ok': True}}], working_memory=wm, session_id='release')
    assert not result['ok'] and not result['outputs']
