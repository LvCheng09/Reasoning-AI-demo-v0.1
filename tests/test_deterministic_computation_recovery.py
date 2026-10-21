"""Deterministic failure replay must not recompute, and must recover on change."""
from copy import deepcopy
import json

import pytest

from main_part.calculator import Calculator
from main_part.structured_arithmetic import StructuredComputationError
from main_part.executor_new import execute_rule_chain
from main_part.rule_execution_runtime import RuleExecutionRuntime
from main_part.data_structure import WorkingMemoryState
from main_part.scheduler_new import SchedulerNew, _runtime_failure_points
from tests.test_rule_activation import graph_memory, typed_rule


def request():
    return {'kind': 'expression', 'expression': {'operator': 'divide', 'numerator': 60, 'denominator': 'x'}}


def test_failure_survives_json_restore_and_ignores_unrelated_changes(monkeypatch):
    import main_part.calculator.service as service
    actual = service.evaluate_structured_expression
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return actual(*args, **kwargs)
    monkeypatch.setattr(service, 'evaluate_structured_expression', counted)
    memory = graph_memory(0)
    runtime = RuleExecutionRuntime(memory, Calculator(), session_id='test')
    with pytest.raises(StructuredComputationError) as first:
        runtime.calculator.calculate(request(), workspace=memory.get_event_workspace('e'), slot_bindings={'x': 'input'})
    assert first.value.code == 'division_by_zero'
    data = json.loads(json.dumps(memory.to_dict()))
    restored = WorkingMemoryState(event_workspaces=data['event_workspaces'], rule_runtime=data['rule_runtime'])
    runtime = RuleExecutionRuntime(restored, Calculator(), session_id='test')
    restored.event_workspaces['e']['spans'][0]['vector_dimension'] = 2
    restored.event_workspaces['e']['spans'].append({'span_id': 'unrelated', 'value': 5})
    for _ in range(3):
        with pytest.raises(StructuredComputationError) as error:
            runtime.calculator.calculate(request(), workspace=restored.get_event_workspace('e'), slot_bindings={'x': 'input'})
        assert error.value.code == 'unchanged_computation_failure'
    assert len(calls) == 1
    restored.event_workspaces['e']['spans'][0]['value'] = 2
    assert runtime.calculator.calculate(request(), workspace=restored.get_event_workspace('e'), slot_bindings={'x': 'input'})['value'] == 30
    assert len(calls) == 2
    assert not runtime.state['computation_failures']


def test_formula_change_and_new_task_are_not_blocked():
    memory = graph_memory(0)
    runtime = RuleExecutionRuntime(memory, Calculator(), session_id='test')
    args = {'workspace': memory.get_event_workspace('e'), 'slot_bindings': {'x': 'input'}}
    with pytest.raises(StructuredComputationError):
        runtime.calculator.calculate(request(), **args)
    changed = request()
    changed['expression']['denominator'] = 2
    assert runtime.calculator.calculate(changed, **args)['value'] == 30
    runtime.reset_for_task()
    with pytest.raises(StructuredComputationError) as error:
        runtime.calculator.calculate(request(), **args)
    assert error.value.code == 'division_by_zero'


def test_executor_and_scheduler_stop_unchanged_recovery_but_allow_repair():
    memory, rule = graph_memory(0), typed_rule()
    rule['produced'][0]['conclusion'][0]['expression'] = request()['expression']
    scheduler = SchedulerNew(memory, session_id='test', db_paths={})
    def execute():
        return execute_rule_chain(matched_rules=[{'rule': rule}], working_memory=memory,
            session_id='test', calculator=scheduler.calculator, rule_runtime=scheduler.rule_runtime)
    first = execute()
    assert not first['ok']
    args = {'failure_type': 'execution_failure', 'status': 'error', 'reason': 'test'}
    assert scheduler._request_control_recovery(**args, executor_results=[first]) is not None
    second = execute()
    assert second['steps'][0]['result']['error_code'] == 'unchanged_computation_failure'
    assert scheduler._request_control_recovery(**args, executor_results=[second]) is None
    points = _runtime_failure_points(None, [second])
    assert points[0]['kind'] == 'deterministic_computation_failure'
    assert points[0]['original_error_code'] == 'division_by_zero'
    assert len(memory.event_workspaces['e']['spans']) == 1
    memory.event_workspaces['e']['spans'][0]['value'] = 2
    assert execute()['ok']
    assert memory.event_workspaces['e']['spans'][-1]['value'] == 30


def test_other_calculator_errors_are_not_cached():
    calculator = Calculator()
    bad = request(); bad['expression']['operator'] = 'not_an_operator'
    for _ in range(2):
        with pytest.raises(StructuredComputationError) as error:
            calculator.calculate(bad, workspace={'spans': []}, slot_bindings={})
        assert error.value.code != 'unchanged_computation_failure'
    assert not calculator._deterministic_failures
