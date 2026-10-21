"""Control selection over comparable, mode-specific Q scores.

No graph mutation or model training happens here. A persisted seed and the
complete candidate snapshot make probability draws repeatable after restore.
"""
from __future__ import annotations

from copy import deepcopy
import math

from .activation_semantics import ActivationError, number
from .rule_execution_contract import digest


def validate_selection_policy(policy: dict) -> None:
    allowed = {'strategy', 'allow_random', 'comparison_key', 'minimum_score',
               'cost_weight', 'max_cost', 'temperature'}
    if not isinstance(policy, dict) or set(policy) - allowed:
        raise ActivationError('invalid_selection_policy', 'unknown selection field')
    if policy.get('strategy') not in {'deterministic', 'probabilistic'}:
        raise ActivationError('invalid_selection_strategy', str(policy.get('strategy')))
    if not isinstance(policy.get('allow_random'), bool):
        raise ActivationError('invalid_random_selection_switch', 'allow_random must be boolean')
    if not isinstance(policy.get('comparison_key'), str) or not policy['comparison_key']:
        raise ActivationError('missing_comparison_key', 'Control must name a common score meaning and scale')
    for key in ('minimum_score', 'cost_weight', 'max_cost', 'temperature'):
        if key in policy:
            value = number(policy[key], key)
            if key in {'cost_weight', 'max_cost'} and value < 0 or key == 'temperature' and value <= 0:
                raise ActivationError('invalid_selection_parameter', key)
    if policy['strategy'] == 'probabilistic' and 'temperature' not in policy:
        raise ActivationError('missing_selection_temperature', 'probabilistic selection needs temperature')
    if policy['strategy'] == 'deterministic' and 'temperature' in policy:
        raise ActivationError('unused_selection_temperature', 'deterministic selection has no temperature')


def select_execution(candidates: list[dict], policy: dict, *, seed: str) -> tuple[dict | None, dict]:
    validate_selection_policy(policy)
    ordered = sorted({c['candidate_id']: c for c in candidates}.values(), key=lambda c: c['candidate_id'])
    rows, eligible = [], []
    for candidate in ordered:
        entries = [e for e in candidate['resolved_q'].get('selection', [])
                   if e['comparison_key'] == policy['comparison_key']]
        if len(entries) != 1:
            raise ActivationError('incomparable_execution_scores', candidate['rule']['rule_id'])
        entry = entries[0]
        score, cost = number(entry['value']), number(entry.get('cost', 0))
        reason = None
        for gate in candidate['resolved_q'].get('dispatch', []):
            value = gate['value']
            if gate['type'] == 'boolean':
                enabled = value if isinstance(value, bool) else number(value) >= number(gate['threshold'])
                if not enabled:
                    reason = 'boolean_gate_disabled'
            elif gate['type'] == 'probability':
                if not 0 <= number(value) <= 1:
                    raise ActivationError('invalid_execution_probability', gate['id'])
                if value == 0:
                    reason = 'zero_execution_probability'
            elif gate['type'] == 'frequency':
                if number(value) < 0:
                    raise ActivationError('invalid_execution_frequency', gate['id'])
                if value == 0:
                    reason = 'zero_execution_frequency'
        if 'minimum_score' in policy and score < policy['minimum_score']:
            reason = 'below_minimum_score'
        if 'max_cost' in policy and cost > policy['max_cost']:
            reason = 'over_cost_budget'
        if not policy['allow_random'] and any(e['type'] == 'probability' for e in candidate['resolved_q'].get('dispatch', [])):
            reason = 'random_dispatch_disabled'
        row = {'candidate_id': candidate['candidate_id'], 'rule_id': candidate['rule']['rule_id'],
               'execution_mode': candidate['candidate_execution_mode'], 'q_profile': candidate.get('candidate_q_profile'),
               'require_evaluation': deepcopy(candidate['require_evaluation']),
               'activation': deepcopy(candidate['activation']), 'q': deepcopy(candidate['resolved_q']),
               'score': score, 'cost': cost, 'utility': number(score - policy.get('cost_weight', 0) * cost),
               'excluded_reason': reason}
        rows.append(row)
        if reason is None:
            eligible.append((candidate, row))
    strategy = policy['strategy'] if policy['allow_random'] else 'deterministic'
    trace = {'schema': 'execution_selection.v1', 'policy': deepcopy(policy), 'effective_strategy': strategy,
             'candidates': rows, 'selected_candidate_id': None, 'status': 'no_eligible_candidate'}
    trace['selection_id'] = digest([trace, seed])
    if not eligible:
        return None, trace
    if strategy == 'deterministic':
        selected, _ = max(eligible, key=lambda pair: pair[1]['utility'])
    else:
        peak = max(row['utility'] for _, row in eligible)
        weights = [math.exp((row['utility'] - peak) / policy['temperature']) for _, row in eligible]
        total = sum(weights)
        draw = int(digest([seed, trace['selection_id']])[:13], 16) / 16 ** 13
        cumulative = 0.0
        selected = eligible[-1][0]
        chosen = False
        for (candidate, row), weight in zip(eligible, weights):
            probability = weight / total
            row['selection_probability'] = probability
            cumulative += probability
            if not chosen and draw < cumulative:
                selected, chosen = candidate, True
        trace['sample'] = draw
    trace.update(selected_candidate_id=selected['candidate_id'], status='selected')
    return selected, trace
