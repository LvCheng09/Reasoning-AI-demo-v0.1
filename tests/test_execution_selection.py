"""Mode-specific A/Q decisions exercised through actual Control and Executor."""
from copy import deepcopy
import json

import pytest

from main_part.activation_policies import execution_selection_control_rule
from main_part.activation_semantics import ActivationError
from main_part.calculator import Calculator
from main_part.data_structure import WorkingMemoryState
from main_part.executor_new import execute_rule_chain
from main_part.execution_selection import select_execution
from main_part.persistence import rule_db
from main_part.rule_execution_runtime import RuleExecutionRuntime
from main_part.scheduler_new import SchedulerNew
from tests.test_rule_activation import graph_memory, typed_rule, actual_value, signal, match_operation


def policy(**changes):
    return {'strategy': 'deterministic', 'allow_random': False, 'comparison_key': 'signed_action.v1', **changes}


def choice_rule():
    rule = typed_rule()
    rule['require_policy'] = {'partial_mode': 'control', 'execution_selection': 'control'}
    rule['produced'] = [deepcopy(rule['produced'][0]), deepcopy(rule['produced'][0])]
    for produced, factor in zip(rule['produced'], (1, -1)):
        produced['conclusion'][0]['expression']['operands'][1] = factor
    rule['io_contract']['partial_modes'] = {name: {'required_ids': ['input'], 'required_slots': ['x'],
        'outputs': ['answer'], 'produced_indices': [i], 'result_scope': 'partial'} for i, name in enumerate(('positive', 'negative'))}
    rule['activation_spec'] = {'primary': 'response', 'signals': {'response': signal(-10)}, 'by_mode': {}}
    for name, factor in (('positive', 1), ('negative', -1)):
        rule['activation_spec']['by_mode'][name] = {'primary': 'response', 'signals': {
            'x': {'domain': 'real', 'source': {'kind': 'input', 'slot': 'x'}},
            'response': {'domain': 'real', 'source': {'kind': 'operation', 'op': 'multiply',
                'inputs': [{'signal': 'x'}, {'value': factor}]}}}}
    rule['q_semantics'] = {'selection': [{'id': 'action_score', 'type': 'score',
        'source': {'signal': 'response'}, 'comparison_key': 'signed_action.v1',
        'score_definition': 'preference for a positive action result; not correctness probability',
        'mapping': {'weights': [1.], 'bias': 0., 'normalization': 'sigmoid', 'parameter_version': 'manual.v1'}}]}
    return rule


def make_case(tmp_path, value=3, selection_policy=None, rule=None):
    memory, rule = graph_memory(value), rule or choice_rule()
    owner = execution_selection_control_rule('choose_execution', policy=selection_policy or policy())
    db = tmp_path / 'rules.sqlite'
    rule_db.upsert_rule_records(db, [rule], rule_kind='content')
    rule_db.upsert_rule_records(db, [owner], rule_kind='control')
    scheduler = SchedulerNew(memory, session_id='selection', db_paths={'rule': str(db)})
    return memory, rule, owner, scheduler


def run_control(memory, rule, owner, scheduler):
    scheduler._apply_produced_operations([match_operation(memory, rule)])
    assert 'answer' not in {s['span_id'] for s in memory.event_workspaces['e']['spans']}
    action = scheduler._default_executor({'session_id': 'selection'}, memory, {'operations': [match_operation(memory, owner)]})
    assert action['pending_operations'], action
    resolved = scheduler._default_executor({'session_id': 'selection'}, memory, {'operations': action['pending_operations']})
    scheduler._apply_produced_operations(resolved['pending_operations'])
    return resolved


@pytest.mark.parametrize('value,mode', [(3, 'positive'), (-3, 'negative')])
def test_input_changes_actual_mode_control_and_output(tmp_path, value, mode):
    memory, rule, owner, scheduler = make_case(tmp_path, value)
    run_control(memory, rule, owner, scheduler)
    output = actual_value(memory)
    assert output['value'] == abs(value)
    assert output['rule_execution']['execution_mode'] == mode
    assert output['rule_execution']['require_degree'] == 1
    trace = next(iter(scheduler.rule_runtime.state['selections'].values()))
    assert {r['execution_mode'] for r in trace['candidates']} == {'full', 'positive', 'negative'}
    assert trace['execution_outcome']['status'] == 'completed'
    assert trace['policy']['comparison_key'] == 'signed_action.v1'


def test_new_rule_cannot_execute_without_actual_control(tmp_path):
    memory, rule, _, scheduler = make_case(tmp_path)
    result = execute_rule_chain(matched_rules=[{'rule': rule}], working_memory=memory,
        session_id='selection', rule_runtime=scheduler.rule_runtime, calculator=scheduler.calculator)
    assert not result['ok']
    assert result['errors'][0]['error_code'] == 'q_control_decision_required'
    assert len(memory.event_workspaces['e']['spans']) == 1


def test_modes_with_missing_inputs_never_receive_activation(tmp_path):
    rule = choice_rule()
    rule['required'].append({'required_id': 'missing', 'condition': [{'span_id': 'y', 'feature_type': 'y',
        'core_vector': [-1., 0.], 'embedding_model': 'numerical-test', 'vector_version': 'v1'}]})
    mode = rule['io_contract']['partial_modes']['negative']
    mode['required_ids'].append('missing'); mode['required_slots'].append('y')
    rule['activation_spec']['by_mode']['negative']['signals']['x']['source']['slot'] = 'y'
    memory, rule, owner, scheduler = make_case(tmp_path, rule=rule)
    candidates = scheduler.rule_runtime.execution_candidates({'rule': rule})
    assert [c['candidate_execution_mode'] for c in candidates] == ['positive']
    assert candidates[0]['require_evaluation']['require_degree'] < 1
    run_control(memory, rule, owner, scheduler)
    assert actual_value(memory)['value'] == 3


def test_probability_switch_and_persisted_seed(tmp_path):
    _, rule, _, scheduler = make_case(tmp_path)
    candidates = scheduler.rule_runtime.execution_candidates({'rule': rule})
    p = policy(strategy='probabilistic', allow_random=True, temperature=.4)
    first, trace = select_execution(candidates, p, seed='repeatable')
    restored = json.loads(json.dumps(candidates))
    second, again = select_execution(restored, p, seed='repeatable')
    assert first['candidate_id'] == second['candidate_id'] and trace == again
    results = {select_execution(candidates, p, seed=str(i))[0]['candidate_execution_mode'] for i in range(80)}
    assert len(results) > 1
    p['allow_random'] = False
    for i in range(10):
        selected, trace = select_execution(candidates, p, seed=str(i))
        assert selected['candidate_execution_mode'] == 'positive'
        assert trace['effective_strategy'] == 'deterministic' and 'sample' not in trace


def test_incomparable_scores_are_not_ranked(tmp_path):
    _, rule, _, scheduler = make_case(tmp_path)
    candidates = scheduler.rule_runtime.execution_candidates({'rule': rule})
    candidates[0]['resolved_q']['selection'][0]['comparison_key'] = 'different_meaning'
    with pytest.raises(ActivationError, match='') as error:
        select_execution(candidates, policy(), seed='s')
    assert error.value.code == 'incomparable_execution_scores'


def test_no_eligible_score_rejects_without_writing(tmp_path):
    memory, rule, owner, scheduler = make_case(tmp_path, selection_policy=policy(minimum_score=1.))
    run_control(memory, rule, owner, scheduler)
    assert len(memory.event_workspaces['e']['spans']) == 1
    assert next(iter(scheduler.rule_runtime.state['selections'].values()))['status'] == 'no_eligible_candidate'


def test_mode_authorization_cannot_be_reused_for_another_mode(tmp_path):
    _, rule, owner, scheduler = make_case(tmp_path)
    candidate = scheduler.rule_runtime.execution_candidates({'rule': rule})[0]
    with pytest.raises(ActivationError) as error:
        scheduler.rule_runtime.authorize(candidate['candidate_id'], mode='positive', control_rule=owner)
    assert error.value.code == 'candidate_mode_mismatch'


def test_q_profile_is_part_of_candidate_and_changes_choice(tmp_path):
    rule = choice_rule()
    q = rule['q_semantics']
    inverse = deepcopy(q)
    inverse['selection'][0]['mapping']['weights'] = [-1.]
    rule['q_semantics'] = {'default_profile': 'normal', 'profiles': {'normal': q, 'inverse': inverse}}
    memory, rule, owner, scheduler = make_case(tmp_path, rule=rule)
    run_control(memory, rule, owner, scheduler)
    output = actual_value(memory)
    # Inverting the full graph's -10 response makes full the strongest candidate.
    assert output['rule_execution']['execution_mode'] == 'full'
    trace = next(iter(scheduler.rule_runtime.state['selections'].values()))
    chosen = next(r for r in trace['candidates'] if r['candidate_id'] == trace['selected_candidate_id'])
    assert chosen['q_profile'] == 'inverse' and len(trace['candidates']) == 6


@pytest.mark.parametrize('field,value,code', [
    ('allow_random', 'false', 'invalid_random_selection_switch'),
    ('temperature', 0, 'invalid_selection_parameter'),
    ('cost_weight', -1, 'invalid_selection_parameter'),
    ('minimum_score', float('nan'), 'invalid_activation_number'),
    ('comparison_key', '', 'missing_comparison_key'),
    ('unused', 1, 'invalid_selection_policy'),
])
def test_invalid_control_configuration_is_not_silently_ignored(field, value, code):
    with pytest.raises(ActivationError) as error:
        execution_selection_control_rule('bad', policy=policy(**{field: value}))
    assert error.value.code == code


def test_vector_q_mapping_and_cost_budget_affect_actual_choice(tmp_path):
    rule = choice_rule()
    graph = rule['activation_spec']['by_mode']['positive']
    graph['signals']['response'] = {'domain': 'vector', 'size': 2, 'source': {
        'kind': 'operation', 'op': 'multiply', 'inputs': [{'signal': 'x'}, {'value': [1., 2.]}]}}
    # All mode graphs must support the same projection dimensions.
    for name in ('negative',):
        rule['activation_spec']['by_mode'][name]['signals']['response'] = {
            'domain': 'vector', 'size': 2, 'source': {'kind': 'operation', 'op': 'multiply',
                'inputs': [{'signal': 'x'}, {'value': [-1., -2.]}]}}
    rule['activation_spec']['signals']['response'] = {'domain': 'vector', 'source': {'kind': 'constant', 'value': [-10., -10.]}}
    entry = rule['q_semantics']['selection'][0]
    entry['mapping']['weights'] = [0., 1.]
    expensive = deepcopy(rule['q_semantics'])
    expensive['selection'][0]['cost'] = 10
    expensive['selection'][0]['mapping']['bias'] = 20
    rule['q_semantics'] = {'default_profile': 'cheap', 'profiles': {'cheap': rule['q_semantics'], 'expensive': expensive}}
    memory, rule, owner, scheduler = make_case(tmp_path, rule=rule, selection_policy=policy(max_cost=1))
    run_control(memory, rule, owner, scheduler)
    assert actual_value(memory)['value'] == 3
    trace = next(iter(scheduler.rule_runtime.state['selections'].values()))
    assert sum(row['excluded_reason'] == 'over_cost_budget' for row in trace['candidates']) == 3
    chosen = next(row for row in trace['candidates'] if row['candidate_id'] == trace['selected_candidate_id'])
    assert chosen['q_profile'] == 'cheap'
    assert chosen['q']['selection'][0]['raw_response'] == [3., 6.]


def test_changed_input_invalidates_retained_selection_candidates(tmp_path):
    memory, rule, owner, scheduler = make_case(tmp_path)
    scheduler._apply_produced_operations([match_operation(memory, rule)])
    action = scheduler._default_executor({'session_id': 'selection'}, memory, {'operations': [match_operation(memory, owner)]})
    memory.event_workspaces['e']['spans'][0]['value'] = -3
    with pytest.raises(ActivationError) as error:
        scheduler._default_executor({'session_id': 'selection'}, memory, {'operations': action['pending_operations']})
    assert error.value.code == 'stale_activation_candidate'
    assert len(memory.event_workspaces['e']['spans']) == 1


def test_restore_pending_control_decision_preserves_choice_and_trace(tmp_path):
    memory, rule, owner, scheduler = make_case(tmp_path, selection_policy=policy(
        strategy='probabilistic', allow_random=True, temperature=.2))
    scheduler._apply_produced_operations([match_operation(memory, rule)])
    action = scheduler._default_executor({'session_id': 'selection'}, memory, {'operations': [match_operation(memory, owner)]})
    data = json.loads(json.dumps(memory.to_dict()))
    restored_memory = WorkingMemoryState(event_workspaces=data['event_workspaces'], rule_runtime=data['rule_runtime'])
    restored = SchedulerNew(restored_memory, session_id='selection', db_paths=scheduler.state.db_paths)
    restored.state.control_context = deepcopy(scheduler.state.control_context)
    for current, mem in ((scheduler, memory), (restored, restored_memory)):
        result = current._default_executor({'session_id': 'selection'}, mem, {'operations': deepcopy(action['pending_operations'])})
        current._apply_produced_operations(result['pending_operations'])
    assert actual_value(memory)['value'] == actual_value(restored_memory)['value']
    assert scheduler.rule_runtime.state['selections'] == restored.rule_runtime.state['selections']


def test_random_disabled_also_blocks_random_dispatch(tmp_path):
    rule = choice_rule()
    rule['activation_spec']['signals']['chance'] = signal(.5, 'unit_interval')
    for graph in rule['activation_spec']['by_mode'].values():
        graph['signals']['chance'] = signal(.5, 'unit_interval')
    rule['q_semantics']['dispatch'] = [{'id': 'chance', 'type': 'probability', 'source': {'signal': 'chance'}}]
    memory, rule, owner, scheduler = make_case(tmp_path, rule=rule)
    run_control(memory, rule, owner, scheduler)
    trace = next(iter(scheduler.rule_runtime.state['selections'].values()))
    assert trace['status'] == 'no_eligible_candidate'
    assert all(row['excluded_reason'] == 'random_dispatch_disabled' for row in trace['candidates'])
    assert len(memory.event_workspaces['e']['spans']) == 1


def test_partial_low_score_cannot_make_missing_full_mode_eligible(tmp_path):
    rule = choice_rule()
    rule['required'][0]['condition'][0]['core_vector'] = [-1., 0.]
    memory, rule, _, scheduler = make_case(tmp_path, rule=rule)
    with pytest.raises(ActivationError) as error:
        scheduler.rule_runtime.execution_candidates({'rule': rule})
    assert error.value.code == 'no_feasible_execution_mode'
    assert len(memory.event_workspaces['e']['spans']) == 1


def test_control_does_not_select_high_score_with_disabled_gate(tmp_path):
    rule = choice_rule()
    rule['activation_spec']['signals']['enabled'] = signal(True, 'boolean')
    for name, graph in rule['activation_spec']['by_mode'].items():
        graph['signals']['enabled'] = signal(name != 'positive', 'boolean')
    rule['q_semantics']['dispatch'] = [{'id': 'gate', 'type': 'boolean', 'source': {'signal': 'enabled'}}]
    memory, rule, owner, scheduler = make_case(tmp_path, rule=rule)
    run_control(memory, rule, owner, scheduler)
    assert actual_value(memory)['rule_execution']['execution_mode'] == 'negative'
    trace = next(iter(scheduler.rule_runtime.state['selections'].values()))
    assert next(r for r in trace['candidates'] if r['execution_mode'] == 'positive')['excluded_reason'] == 'boolean_gate_disabled'


def test_profile_output_scope_limits_candidate_pairs(tmp_path):
    rule = choice_rule()
    rule['produced'][1]['conclusion'][0]['span_id'] = 'other'
    rule['io_contract']['outputs'].append('other')
    rule['io_contract']['partial_modes']['negative']['outputs'] = ['other']
    plain = rule['q_semantics']
    scoped = deepcopy(plain)
    scoped['transform'] = [{'id': 'effect', 'type': 'signed_effect', 'source': {'signal': 'response'},
                           'target': {'span_id': 'other', 'field': 'value'}}]
    rule['q_semantics'] = {'default_profile': 'plain', 'profiles': {'plain': plain, 'other_only': scoped}}
    _, rule, _, scheduler = make_case(tmp_path, rule=rule)
    candidates = scheduler.rule_runtime.execution_candidates({'rule': rule})
    pairs = {(c['candidate_execution_mode'], c['candidate_q_profile']) for c in candidates}
    assert len(pairs) == 5
    assert ('positive', 'other_only') not in pairs
    assert ('negative', 'other_only') in pairs


def test_selection_control_recall_is_scoped_by_score_meaning(tmp_path):
    from main_part.activation_policies import selection_scope_tag
    memory, rule, owner, scheduler = make_case(tmp_path)
    candidates = scheduler.rule_runtime.execution_candidates({'rule': rule})
    request = scheduler._request_activation_control(candidates)
    tags = request['pending_rule_selectors'][0]['required_any_rule_tags']
    assert tags == [selection_scope_tag('signed_action.v1')]
    assert tags[0] in owner['scope_tags']
    other = execution_selection_control_rule('other_domain', policy=policy(comparison_key='other_meaning.v1'))
    assert not set(tags).intersection(other['scope_tags'])
    run_control(memory, rule, owner, scheduler)
    assert actual_value(memory)['value'] == 3


def test_mixed_score_meanings_fail_before_requesting_wrong_control(tmp_path):
    _, rule, _, scheduler = make_case(tmp_path)
    other = deepcopy(rule)
    other['rule_id'] = 'other'
    other['q_semantics']['selection'][0]['comparison_key'] = 'incompatible.v1'
    candidates = scheduler.rule_runtime.execution_candidates({'rule': rule})
    candidates += scheduler.rule_runtime.execution_candidates({'rule': other})
    with pytest.raises(ActivationError, match='common score meaning'):
        scheduler._request_activation_control(candidates)


def test_repeated_full_memory_clear_keeps_runtime_bound_and_allows_next_task(tmp_path):
    memory, rule, owner, scheduler = make_case(tmp_path)
    run_control(memory, rule, owner, scheduler)
    seed = scheduler.rule_runtime.state['seed']
    for _ in range(2):
        scheduler.clear_transient_state(clear_working_memory=True, clear_errors=True)
        assert scheduler.rule_runtime.state['seed'] == seed
        assert not scheduler.rule_runtime.state['authorizations']
        assert not scheduler.rule_runtime.state.get('selections')
    memory.event_workspaces.update(graph_memory(-4).event_workspaces)
    run_control(memory, rule, owner, scheduler)
    assert actual_value(memory)['value'] == 4
