from copy import deepcopy
import pytest
from main_part.require_gaps import inspect_require_gaps, install_recovery_branches
from main_part.persistence import rule_db
from main_part.scheduler_new import SchedulerNew
from tests.test_rule_activation import typed_rule, graph_memory


def inspect(rule, memory, **kw):
    return inspect_require_gaps([rule], memory, ['e'], target_ids=[rule['rule_id']], **kw)


def missing():
    m, r = graph_memory(), typed_rule()
    r['required'][0]['condition'][0]['feature_type'] = 'missing_x'
    return m, r


def test_unknown_reextract_and_no_progress_budget():
    m,r = missing()
    assert inspect(r,m)['kind']=='unknown'
    a=inspect(r,m,source_text='The source document is available for verification.')
    assert a['kind']=='reextract' and a['question']
    b=inspect(r,m,source_text=a['source_text'],attempts={a['gap_id']:1})
    assert b['kind']=='unknown' and b['gap_id']==a['gap_id']


def test_false_predicate_is_not_missing_information():
    m,r=graph_memory(),typed_rule()
    r['required'][0]['predicates']=[{'operator':'gt','left':'x','right':100}]
    assert inspect(r,m)['kind']=='condition_false'


def test_repair_routes_require_actual_declared_rules():
    m,r=missing()
    r['metadata']={'require_recovery':{'input':{'kind':'derivation','repair_rule_ids':['absent']}}}
    with pytest.raises(ValueError,match='unknown rule'):
        inspect(r,m)
    repair=typed_rule();repair['rule_id']='derive_input'
    for kind in ('derivation','missing_relation'):
        r['metadata']['require_recovery']['input'].update(kind=kind,repair_rule_ids=[repair['rule_id']])
        result=inspect_require_gaps([r,repair],m,['e'],target_ids=[r['rule_id']])
        assert result['kind']==kind and result['repair_rule_ids']==['derive_input']


def test_actual_control_reports_unknown_instead_of_retrying(tmp_path):
    m,r=missing()
    owner=install_recovery_branches({'rule_id':'recover','rule_layer':'control','required':[],
        'produced':[{'operation':'control_decide','branches':[],
            'default':{'operation':'stop_path','reason':'no repair requested'}}]})
    db=tmp_path/'rules.sqlite'
    rule_db.upsert_rule_records(db,[r],rule_kind='content')
    rule_db.upsert_rule_records(db,[owner],rule_kind='control')
    scheduler=SchedulerNew(m,session_id='gaps',db_paths={'rule':str(db)})
    scheduler.state.active_event_ids=['e']
    scheduler.state.control_context=scheduler._build_control_context(trigger={
        'kind':'runtime_failure','pending_rule_ids':[r['rule_id']]})
    result=scheduler._execute_rule_chain(matched_rules=[{'rule':owner}])
    assert result['ok'],result
    stop=result['stopped_paths'][0]
    assert stop['task_result']['status']=='pending'
    assert stop['task_result']['gaps'][0]['rule_id']==r['rule_id']
    assert not result['next_rule_ids'] and not result['frontend_requests']


def test_scoped_knowledge_binding_preserves_conflicts_and_unscoped_facts():
    from main_part.require_gaps import bind_knowledge_context
    from main_part.data_structure import WorkingMemoryState
    relation = {'core':'measurement context','relation':'denominated in','nature':'USD',
        'core_vector':[1.,0.], 'relation_vector':[1.,0.], 'nature_vector':[1.,0.],
        'embedding_model':'numerical-test','vector_version':'v1'}
    opposite = {**relation, 'nature':'EUR','nature_vector':[-1.,0.]}
    memory = WorkingMemoryState(event_workspaces={'e':{'event_id':'e','spans':[
        {'span_id':'a'}, {'span_id':'b','semantic_relations':[opposite]}, {'span_id':'c'},
        {'span_id':'scope','metadata':{'applies_to':['a','b']},'semantic_relations':[relation]}]}})
    knowledge={'knowledge_id':'context_policy','metadata':{'context_binding':{
        'scope':'explicit_applies_to','relation_types':[relation]}}}
    report=bind_knowledge_context(memory,['e'],knowledge,control_rule_id='control')
    assert len(report['changed'])==1 and len(report['conflicts'])==1
    spans=memory.event_workspaces['e']['spans']
    assert spans[0]['semantic_relations'][0]['provenance']['knowledge_id']=='context_policy'
    assert spans[1]['semantic_relations']==[opposite]
    assert 'semantic_relations' not in spans[2]
    assert not bind_knowledge_context(memory,['e'],knowledge,control_rule_id='control')['changed']


def test_control_resumes_target_after_actual_calculator_repair(tmp_path):
    from main_part.executor_new import execute_rule_chain
    m,r=missing()
    repair=typed_rule(); repair['rule_id']='repair'
    repair['produced'][0]['conclusion'][0].update(span_id='new_x',feature_type='missing_x',
        core_vector=[1.,0.],relation_vector=[1.,0.],nature_vector=[1.,0.],
        embedding_model='numerical-test',vector_version='v1')
    repair['io_contract']['outputs']=['new_x']
    r['metadata']={'require_recovery':{'input':{'kind':'derivation','repair_rule_ids':['repair']}}}
    owner=install_recovery_branches({'rule_id':'recover','rule_layer':'control','required':[],
        'produced':[{'operation':'control_decide','branches':[], 'default':{'operation':'stop_path','reason':'done'}}]})
    db=tmp_path/'r.sqlite'
    rule_db.upsert_rule_records(db,[r,repair],rule_kind='content')
    rule_db.upsert_rule_records(db,[owner],rule_kind='control')
    s=SchedulerNew(m,session_id='repair',db_paths={'rule':str(db)})
    s.state.active_event_ids=['e']
    s.state.control_context=s._build_control_context(trigger={'kind':'runtime_failure','pending_rule_ids':['compute']})
    result=s._execute_rule_chain(matched_rules=[{'rule':owner}])
    assert result['next_rule_ids']==['repair']
    computed=s._execute_rule_chain(matched_rules=[{'rule':repair}])
    assert computed['ok']
    s._apply_produced_operations(computed['pending_operations'])
    assert next(x for x in m.event_workspaces['e']['spans'] if x['span_id']=='new_x')['value']==6
    s.state.control_context=s._build_control_context(trigger={'kind':'path_segment_completed'})
    resumed=s._execute_rule_chain(matched_rules=[{'rule':owner}])
    assert resumed['next_rule_ids']==['compute']
    assert not s.state.control_state.get('require_repair_targets')


def test_layered_network_executes_each_control_hop_and_rechecks_conditions(tmp_path):
    from main_part.require_gaps import recovery_control_network
    m,r=missing()
    entry,children=recovery_control_network({'rule_id':'entry','rule_layer':'control','required':[],
        'produced':[{'operation':'control_decide','branches':[], 'default':{'operation':'stop_path','reason':'done'}}]})
    registry={x['rule_id']:x for x in [entry,*children]}
    db=tmp_path/'network.sqlite'
    rule_db.upsert_rule_records(db,[r],rule_kind='content')
    rule_db.upsert_rule_records(db,list(registry.values()),rule_kind='control')
    s=SchedulerNew(m,session_id='network',db_paths={'rule':str(db)})
    s.state.active_event_ids=['e']
    s.state.control_context=s._build_control_context(trigger={'kind':'runtime_failure','pending_rule_ids':['compute']})
    rid='entry'; visited=[]
    for _ in range(4):
        visited.append(rid)
        result=s._execute_rule_chain(matched_rules=[{'rule':registry[rid]}])
        assert result['ok'],result
        if result['next_rule_ids']: rid=result['next_rule_ids'][0]
    assert visited==['entry','control_require_recovery_router','control_require_route_unknown','control_require_action_unknown']
    assert result['stopped_paths'][0]['task_result']['status']=='pending'
    s.state.control_state['require_gaps']={'kind':'none'}
    s.state.control_context=s._build_control_context(trigger={})
    wrong=s._execute_rule_chain(matched_rules=[{'rule':registry['control_require_action_reextract']}])
    assert not wrong['frontend_requests']
    assert wrong['stopped_paths'][0]['status']=='inapplicable_control'


@pytest.mark.parametrize('kind', ['reextract', 'missing_relation', 'derivation',
    'unknown', 'condition_false', 'conflict'])
def test_six_routes_through_actual_scheduler(tmp_path, kind):
    from tools.validate_require_recovery_network import run_case
    result = run_case(tmp_path, kind)
    assert result['loops'] < 80


def test_false_alternative_does_not_block_repairable_rule():
    m, r = missing()
    false = typed_rule()
    false['rule_id'] = 'false_path'
    false['required'][0]['predicates'] = [{'operator': 'gt', 'left': 'x', 'right': 100}]
    report = inspect_require_gaps([false, r], m, ['e'], target_ids=['false_path', 'compute'])
    assert report['kind'] == 'unknown'
    assert len(report['gaps']) == 2


def test_public_clarification_resumes_real_scheduler(tmp_path, monkeypatch):
    from tools.validate_require_recovery_network import scheduler_for, span
    from main_part.model_launcher import MainPartModelLauncher
    monkeypatch.setenv('MAIN_PART_FRONTEND_VECTORIZE', '0')
    scheduler = scheduler_for(tmp_path, 'unknown')
    scheduler.run()
    request_id = scheduler.state.control_state['task_result']['request_id']
    launcher = MainPartModelLauncher(scheduler=scheduler, db_paths=scheduler.state.db_paths)
    launcher.submit_require_answer(request_id, span_updates=[span('answer_from_user', 'missing', 7)],
        source='user clarification')
    answers = [s['value'] for w in scheduler.working_memory.event_workspaces.values()
        for s in w['spans'] if s['span_id'] == 'answer']
    assert answers == [14]
    assert scheduler.state.control_state['task_result']['status'] == 'solved'


def test_knowledge_repair_is_atomic_and_retains_source_conflict():
    from tools.validate_require_recovery_network import scenario, span, relation
    from main_part.require_gaps import bind_knowledge_context
    m, target, _, knowledge = scenario('missing_relation')
    original = deepcopy(m.event_workspaces)
    m.event_workspaces['e']['spans'].append({
        **span('bad_scope', 'scope'), 'metadata': {'applies_to': ['nonexistent']},
        'semantic_relations': [relation()]})
    before = deepcopy(m.event_workspaces)
    with pytest.raises(ValueError, match='unknown target'):
        bind_knowledge_context(m, ['e'], knowledge[0], control_rule_id='bind_context')
    assert m.event_workspaces == before
    m.event_workspaces = original
    # Both amounts agree, but the explicitly applicable source disagrees.
    for fact in m.event_workspaces['e']['spans'][:2]:
        fact['semantic_relations'] = [relation('EUR')]
    bind_knowledge_context(m, ['e'], knowledge[0], control_rule_id='bind_context')
    report = inspect_require_gaps([target], m, ['e'], target_ids=['compute'])
    assert report['kind'] == 'conflict'
    assert report['match_evidence']['source_conflicts'][0]['source_span_id'] == 'scope'


def test_network_reinstallation_is_idempotent():
    from main_part.require_gaps import recovery_control_network
    entry = {'rule_id': 'entry', 'rule_layer': 'control', 'required': [],
        'produced': [{'operation': 'control_decide', 'branches': [],
            'default': {'operation': 'stop_path'}}]}
    first, children = recovery_control_network(entry)
    second, again = recovery_control_network(first)
    assert first == second and children == again
