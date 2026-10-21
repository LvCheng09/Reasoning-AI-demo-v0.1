from copy import deepcopy
import pytest
from tests.test_semantic_alignment import _memory, _rule, _match, _schema_context, CORE, PROJECTED, MODEL, VERSION
from main_part.semantic_alignment import align_event_spans
from main_part.scheduler_new import _alignment_requirements
from main_part.alignment_contract import alignment_is_current
from main_part.knowledge_schema_contract import validate_span_schema
from main_part.rule_path_candidates import choose_candidate_rule_path
from main_part.executor_new import execute_rule_chain
from main_part.data_structure import WorkingMemoryState
from tests.test_semantic_alignment import _ambiguous_turnover_memory


def multi_memory():
    memory=_memory(); workspace=memory.event_workspaces['e']
    workspace['spans'][0].update(entity_id='A',period='current')
    workspace['spans'].append({'span_id':'support','core_vector':PROJECTED,'relation_vector':PROJECTED,'nature_vector':PROJECTED,
                              'embedding_model':MODEL,'vector_version':VERSION,'entity_id':'A','period':'current','value':999})
    workspace['context_items']=[_schema_context(activation_mode='all_cores',edge_role='activation_core',core=CORE),
                                _schema_context(activation_mode='all_cores',edge_role='activation_core',core=PROJECTED,component_role='support')]
    return memory


def align(memory):
    return align_event_spans(memory,_alignment_requirements(_match(memory,_rule())),None,control_rule_id='generic_proof')


def test_subject_is_not_retrieval_order_and_support_is_fully_recorded():
    memory=multi_memory(); memory.event_workspaces['e']['context_items'].reverse()
    assert align(memory)['accepted']
    span=memory.event_workspaces['e']['spans'][-1]; proof=span['metadata']['semantic_alignment']
    assert span['value']==120 and proof['source_span_id']=='s'
    assert proof['supporting_span_ids']==['support']
    assert set(proof['source_fingerprints'])=={'s','support'}


@pytest.mark.parametrize('field,value',[('entity_id','B'),('period','past'),('period',None)])
def test_cross_scope_or_missing_scope_cannot_prove_completion(field,value):
    memory=multi_memory(); memory.event_workspaces['e']['spans'][1][field]=value
    assert not align(memory)['accepted']


def test_negative_support_cannot_authorize_a_positive_completion():
    memory=multi_memory(); memory.event_workspaces['e']['spans'][1]['polarity']='negative'
    assert not align(memory)['accepted']


def test_invalid_first_support_does_not_hide_valid_bound_support():
    memory=multi_memory(); invalid=deepcopy(memory.event_workspaces['e']['spans'][1])
    invalid.update(span_id='first_bad',entity_id='B')
    memory.event_workspaces['e']['spans'].insert(1,invalid)
    assert align(memory)['accepted']


def test_support_mutation_invalidates_existing_completion():
    memory=multi_memory(); assert align(memory)['accepted']
    spans=memory.event_workspaces['e']['spans']; derived=spans[-1]
    assert alignment_is_current(derived,{s['span_id']:s for s in spans},[])
    spans[1]['period']='past'
    assert not alignment_is_current(derived,{s['span_id']:s for s in spans},[])
    assert not _match(memory,_rule())['ok']


def test_derived_value_cannot_be_changed_while_sources_stay_the_same():
    memory=multi_memory(); assert align(memory)['accepted']
    spans=memory.event_workspaces['e']['spans']; spans[-1]['value']=999
    assert not alignment_is_current(spans[-1],{s['span_id']:s for s in spans},[])


def test_old_implicit_subject_is_rejected_instead_of_compatibility_guess():
    schema={'activation_mode':'all_cores','edge_role':'activation_core'}
    with pytest.raises(ValueError,match='explicit subject'): validate_span_schema(schema)


def candidate(name,quality=1,valid=True):
    return {'candidate_path_id':name,'rule_ids':[name],
            'matched_rules':[{'rule':{'rule_id':name},'match':{'ok':valid}}],
            'selection_features':{'match_quality':quality}}


def test_memory_preference_is_effective_only_among_strict_candidate_paths():
    paths=[candidate('normal'),candidate('history',.1)]
    assert choose_candidate_rule_path(paths)['rule_ids']==['normal']
    selected=choose_candidate_rule_path(paths,memory_preferred_rule_ids=['history'])
    assert selected['rule_ids']==['history'] and selected['normal_fallback_candidates'][0]['rule_ids']==['normal']
    paths[1]['matched_rules'][0]['match']['ok']=False
    assert choose_candidate_rule_path(paths,memory_preferred_rule_ids=['history'])['rule_ids']==['normal']


def test_executed_control_inherits_declared_memory_policy_for_path_conflict():
    rule={'rule_id':'generic_control','rule_layer':'control','produced':[{'operation':'control_decide','default':{
        'operation':'select_candidate_path','selection_policy':{'inherit_memory_path_preference':True,'memory_failure_fallback':'normal_candidates'}}}]}
    context={'trigger':{'candidate_rule_paths':[candidate('normal'),candidate('history',.1)],
                        'source_rule_selector':{'memory_path_policy':'prefer_then_fallback','memory_preferred_rule_ids':['history']}}}
    result=execute_rule_chain(matched_rules=[{'rule':rule,'match':{'ok':True}}],working_memory=WorkingMemoryState(),
                              session_id='conflict_policy',control_context=context)
    assert result['pending_operations'][0]['candidate_path']['rule_ids']==['history']
    context['trigger']['source_rule_selector']['memory_path_policy']='reference_only'
    normal=execute_rule_chain(matched_rules=[{'rule':rule,'match':{'ok':True}}],working_memory=WorkingMemoryState(),
                              session_id='no_preference',control_context=context)
    assert normal['pending_operations'][0]['candidate_path']['rule_ids']==['normal']


def linked_context_memory():
    memory = _ambiguous_turnover_memory([0.2, 0.98])
    workspace = memory.event_workspaces['e']
    source = workspace['spans'][0]
    source.update(entity_id='A', period='current', value=7, unit='turns')
    context = source['semantic_relations'].pop()
    workspace['spans'].append({
        'span_id': 'context', **context, 'entity_id': 'A', 'period': 'current',
    })
    workspace['relations'] = [{'source_span_id': 'turnover', 'target_span_id': 'context'}]
    rule = {'rule_id': 'inventory_rule', 'rule_layer': 'domain', 'required': [{'condition': [{
        'span_id': 'inventory_turnover', 'feature_type': 'inventory_turnover',
        'core_vector': PROJECTED, 'relation_vector': PROJECTED,
        'nature_vector': PROJECTED, 'match_threshold': .99,
    }]}]}
    return memory, rule


@pytest.mark.parametrize('field,value', [
    ('entity_id', 'B'), ('period', 'past'), ('period', None), ('polarity', 'negative'),
])
def test_adjacent_relation_context_still_requires_current_scope(field, value):
    memory, rule = linked_context_memory()
    memory.event_workspaces['e']['spans'][1][field] = value
    result = align_event_spans(memory, _alignment_requirements(_match(memory, rule)), None,
                              control_rule_id='generic_proof')
    assert not result['accepted']
    assert not _match(memory, rule)['ok']


def test_linked_context_has_full_provenance_and_expires_when_link_removed():
    memory, rule = linked_context_memory()
    result = align_event_spans(memory, _alignment_requirements(_match(memory, rule)), None,
                              control_rule_id='generic_proof')
    assert result['accepted'] and _match(memory, rule)['ok']
    workspace = memory.event_workspaces['e']
    derived = workspace['spans'][-1]
    proof = derived['metadata']['semantic_alignment']
    assert derived['value'] == 7 and proof['source_span_id'] == 'turnover'
    assert set(proof['source_fingerprints']) == {'turnover', 'context'}
    workspace['relations'].clear()
    assert not _match(memory, rule)['ok']


def test_unlinked_context_cannot_be_used_despite_matching_scope():
    memory, rule = linked_context_memory()
    memory.event_workspaces['e']['relations'].clear()
    result = align_event_spans(memory, _alignment_requirements(_match(memory, rule)), None,
                              control_rule_id='generic_proof')
    assert not result['accepted']
