"""Validate declarative within-span definitions; no business vocabulary here."""
from copy import deepcopy


def binding_policy(policy):
    if not isinstance(policy, dict) or set(policy) != {'equal_fields', 'require_fields', 'support_scope'}:
        raise ValueError('Knowledge binding_policy requires equal_fields, require_fields, support_scope')
    fields = policy['equal_fields']
    if not isinstance(fields, list) or not fields or any(not isinstance(f, str) or not f or any(not p for p in f.split('.')) for f in fields):
        raise ValueError('Knowledge equal_fields must declare nonempty field paths')
    if len(set(fields)) != len(fields) or not isinstance(policy['require_fields'], bool):
        raise ValueError('Knowledge binding fields must be unique and require_fields must be boolean')
    if policy['support_scope'] not in {'same_event', 'same_span_or_linked'}:
        raise ValueError('Unsupported Knowledge support_scope')
    return deepcopy(policy)


def validate_span_schema(schema):
    mode = schema.get('activation_mode', 'explicit_components')
    role = schema.get('edge_role', 'evidence')
    allowed = {'explicit_components': {'evidence'}, 'single_core': {'activation_core'},
               'all_cores': {'activation_core'}, 'relation_context': {'activation_relation', 'context_core'}}
    if mode not in allowed or role not in allowed[mode]:
        raise ValueError('Invalid Knowledge activation mode or edge role')
    if mode in {'single_core', 'all_cores'}:
        if schema.get('component_role') not in {'subject', 'support'}:
            raise ValueError('Core activation requires an explicit subject/support role; migrate old schemas')
        if mode == 'single_core' and schema['component_role'] != 'subject':
            raise ValueError('Single-core definition must declare its subject')
    if mode in {'all_cores', 'relation_context'}:
        binding_policy(schema.get('binding_policy'))


def field_value(value, path):
    for key in path.split('.'):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value
