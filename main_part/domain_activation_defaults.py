"""Explicit baseline contracts installed into writable bundled runtime copies.

These are authored cold-start policies, not calibrated probabilities or learned
parameters. Existing custom databases and unknown domains are not rewritten.
"""
from copy import deepcopy
import math

from .activation_policies import execution_selection_control_rule
from .persistence import rule_db
from .rule_execution_contract import validate_rule_contract

COMPARISON = 'dso_numeric_input_evidence_quality.v1'
DSO_RULE_IDS = ('dso_calc_direct', 'dso_calc_from_turnover', 'dso_calc_reverse_sales')
SUPPLY_COMPARISON = 'supply_authored_rule_preference.v1'

def migrate_rule(original):
    rule = deepcopy(original)
    continuations = [p for p in rule['produced'] if p.get('operation') == 'run_rule']
    if any(p.get('rule_ids') not in (['dso_output_numeric'], ['dso_output_reverse']) or p.get('rule_selector') for p in continuations):
        raise ValueError('pilot migration only moves the existing numeric output continuation')
    rule['produced'] = [p for p in rule['produced'] if p.get('operation') != 'run_rule']
    for i, group in enumerate(rule['required']):
        group['required_id'] = group.get('required_id') or f'inputs_{i}'
    slots = list(dict.fromkeys(s['span_id'] for group in rule['required'] for s in group['condition']))
    signals = {f'quality_{i}': {'domain': 'unit_interval', 'source': {'kind': 'input', 'slot': slot, 'field': 'confidence'}}
               for i, slot in enumerate(slots)}
    signals['input_quality'] = {'domain': 'unit_interval', 'source': {'kind': 'operation', 'op': 'min',
        'inputs': [{'signal': key} for key in signals]}}
    rule.update(execution_contract_version=2, require_policy={'partial_mode': 'disabled', 'execution_selection': 'control'},
        activation_spec={'primary': 'input_quality', 'signals': signals},
        q_semantics={'selection': [{'id': 'input_quality_preference', 'type': 'score',
            'source': {'signal': 'input_quality'}, 'comparison_key': COMPARISON,
            'score_definition': 'minimum declared input evidence quality; not calibrated answer correctness',
            'mapping': {'weights': [1.], 'bias': 0., 'normalization': 'identity', 'parameter_version': 'manual_quality.v1'}}]},
        io_contract={'required_slots': slots, 'outputs': [s['span_id'] for p in rule['produced'] for s in p.get('conclusion', [])]},
        effect_contract={'kind': 'graph', 'write_targets': ['spans']})
    validate_rule_contract(rule)
    return rule



def supply_rule(original):
    """Preserve qualitative inference; use an explicitly authored rule prior.

    Existing supply observations need not carry numeric confidence. We do not
    invent one: this baseline responds to the stored rule prior only. Learning
    or evidence-sensitive response graphs can replace it under the same API.
    """
    rule = deepcopy(original)
    if any(p.get('operation') != 'add' or not p.get('conclusion') for p in rule['produced']):
        raise ValueError('supply baseline supports pure conclusion rules only')
    prior = rule.get('confidence')
    if isinstance(prior, bool) or not isinstance(prior, (int, float)) or not 0 < prior < 1:
        raise ValueError('supply baseline requires an explicitly declared interior prior')
    for i, group in enumerate(rule['required']):
        group['required_id'] = group.get('required_id') or f'inputs_{i}'
    slots = list(dict.fromkeys(s['span_id'] for g in rule['required'] for s in g['condition']))
    rule.update(execution_contract_version=2,
        require_policy={'partial_mode': 'disabled', 'execution_selection': 'control'},
        activation_spec={'primary': 'response', 'signals': {
            'prior_logit': {'domain': 'real', 'source': {'kind': 'constant', 'value': math.log(prior / (1-prior))}},
            'response': {'domain': 'real', 'source': {'kind': 'operation', 'op': 'multiply',
                'inputs': [{'signal': 'prior_logit'}, {'value': 1.0}]}}}},
        q_semantics={'selection': [{'id': 'authored_preference', 'type': 'score',
            'source': {'signal': 'response'}, 'comparison_key': SUPPLY_COMPARISON,
            'score_definition': 'authored rule preference; not evidence confidence or calibrated correctness',
            'mapping': {'weights': [1.0], 'bias': 0.0, 'normalization': 'sigmoid', 'parameter_version': 'authored_prior.v1'}}]},
        io_contract={'required_slots': slots, 'outputs': [s['span_id'] for p in rule['produced'] for s in p['conclusion']]},
        effect_contract={'kind': 'graph', 'write_targets': ['spans']})
    validate_rule_contract(rule)
    return rule


def install_bundled_activation(db):
    """Install baseline policies in a freshly copied bundled registry only."""
    originals = rule_db.load_rule_records(db, limit=None)
    by_id = {r['rule_id']: r for r in originals}
    required = {*DSO_RULE_IDS, 'control_runtime_entry_001'}
    if not required.issubset(by_id):
        raise ValueError('bundled activation installation requires the bundled DSO registry')
    changed = [migrate_rule(by_id[rid]) for rid in DSO_RULE_IDS]
    changed.extend(supply_rule(r) for r in originals if r.get('domain') == 'supply_chain' and r.get('rule_layer') == 'domain')
    controls = []
    for rid, domain, key in (
        ('dso_execution_selection_control', 'finance_dso', COMPARISON),
        ('supply_execution_selection_control', 'supply_chain', SUPPLY_COMPARISON)):
        owner = execution_selection_control_rule(rid, policy={
            'strategy': 'deterministic', 'allow_random': False, 'comparison_key': key})
        owner['domain'] = domain
        controls.append(owner)
    entry = deepcopy(by_id['control_runtime_entry_001'])
    branches = [b for b in entry['produced'][0]['branches']
        if not b.get('policy_id', '').startswith('activation_completion_')]
    entry['produced'][0]['branches'] = branches
    # Continue only calculations migrated here. Do not intercept other domains
    # or send an already pending output through the numeric output rule again.
    for rid in DSO_RULE_IDS:
        output = 'dso_output_reverse' if rid == 'dso_calc_reverse_sales' else 'dso_output_numeric'
        branches.insert(0, {'policy_id': 'activation_completion_' + rid,
            'when': {'all': [
                {'path': 'trigger.kind', 'operator': 'eq', 'value': 'path_segment_completed'},
                {'path': 'path.visited_rule_ids.*', 'operator': 'eq', 'value': rid},
                {'not': {'path': 'path.visited_rule_ids.*', 'operator': 'eq', 'value': output}},
                {'path': 'output.pending_candidates', 'operator': 'falsy'}]},
            'then': {'operation': 'run_rule', 'rule_ids': [output]}})
    from .require_gaps import recovery_control_network
    entry, recovery_controls = recovery_control_network(entry)
    controls.extend(recovery_controls)
    rule_db.upsert_rule_records(db, changed, rule_kind='content', preserve_stats=True)
    rule_db.upsert_rule_records(db, [*controls, entry], rule_kind='control', preserve_stats=True)
    return {'version': 1, 'migrated_rules': [r['rule_id'] for r in changed],
            'selection_controls': [r['rule_id'] for r in controls if 'rule_activation_decision' in r.get('scope_tags', [])],
            'recovery_controls': [r['rule_id'] for r in recovery_controls],
            'adjusted_control': entry['rule_id'], 'trains_parameters': False,
            'parameter_origin': 'authored, not calibrated'}
