# Architecture and Runtime Interfaces

[中文技术报告](TECH_REPORT.CHINESE.md) · [English Tech Report](TECH_REPORT.ENGLISH.md)

This is the code map for the inference implementation updated on 2026-10-21. It separates semantic translation, current state, Require, Activation, Q interpretation, execution and Control. Bounded child calls and recorded-support revision are released mechanisms; automatic hierarchy generation and arbitrary dependency maintenance are not implied.

## 1 Package boundaries

```text
demo.py / main_part.cli              configuration and presentation
          |
MainPartModelLauncher               composition and stable graph boundary
          |
semantic_frontend/                  text -> versioned semantic document
          |
normalization + vectors + Contract
          |
frontend_working_memory_bridge      document -> Event workspaces
          |
RuntimeLauncher                     reliable input delivery
          |
SchedulerNew                        stage loop and Control invocation
          |
retrieval + matcher/                Require, bindings and feasible modes
          |
RuleExecutionRuntime                new-contract Activation / Q / authorization
          |                         revalidation and dispatch before effects
executor_new                        accepted graph/control operations
   |             |
calculator/   RuleCompositionRuntime -> actual child matching and execution
   |             |
   +--- checked results, supports and Event writeback ---+
          |
Control review                      continue / recover / wait / stop / output
          |                         Knowledge definitions, Memory hints,
          |                         source refinement and evidence review
          +---- current Event state -> next matching round
          |
Output Gate -> output_service -> output_decoder -> answer_verbalizer
```

The frontend/backend interface is in-process Python data. Only provider clients use HTTP. `submit_event_graph` bypasses the LLM but not validation, workspace delivery, or reasoning.

The diagram expands the new-contract branch; Rules without `execution_contract_version=2` retain full-Require binary execution through the shared runtime. The bilingual reports include the complete flowchart and recovery branches. Partial internal execution does not automatically enable partial-answer presentation.

The released seed has 20 new-contract Rules and 190 binary Rules. All have partial execution `disabled`; partial-mode runtime support must not be confused with an enabled default business configuration.

## 2 Components and contracts

| Component | Receives | Returns or owns | Implementation |
| --- | --- | --- | --- |
| Frontend | Source text or a scoped refinement request | Versioned document with events, mainlines, explicit edges, and references | [frontend.py](../semantic_frontend/frontend.py), [prompts.py](../semantic_frontend/prompts.py), [validation.py](../semantic_frontend/validation.py) |
| Adapter | Semantic document | Backend span/Event dictionaries | [adapter.py](../semantic_frontend/adapter.py) |
| Graph boundary | Graph dictionaries and source input | Normalized vectors, contract result, frontend-ready input | [model_launcher.py](../main_part/model_launcher.py), [frontend_schema.py](../main_part/frontend_schema.py), [event_graph_contract.py](../main_part/event_graph_contract.py), [bridge](../main_part/frontend_working_memory_bridge.py) |
| Working Memory | Valid graph and operation updates | Event workspaces, contexts, claims, and runtime entries | [data_structure.py](../main_part/data_structure.py) |
| Runtime host | Frontend-ready input entries | Delivered output or visible retry/dead-letter failure | [runtime_launcher.py](../main_part/runtime_launcher.py) |
| Scheduler | Current state and Rule feedback | Pending stages, selectors, operations, active strategy, output candidates | [scheduler_new.py](../main_part/scheduler_new.py) |
| Retrieval | Current graph/query and selectors | Knowledge definitions, Memory hints, candidate Rules | [retrieval.py](../main_part/retrieval.py), [matcher/routing.py](../main_part/matcher/routing.py) |
| Matcher | Required patterns and current evidence | Slot bindings, admissible matches, rejected requirements | [matcher/](../main_part/matcher/) |
| Rule contract | Rule declarations and selected mode | Valid Require policy, input/output/effect boundaries | [rule_execution_contract.py](../main_part/rule_execution_contract.py) |
| Activation and Q | Current bound inputs, context and declared signal graph | Typed responses and explicit dispatch/selection/transform/assessment interpretations | [activation_semantics.py](../main_part/activation_semantics.py) |
| Execution runtime | Candidate, current Event, authorization and function versions | Revalidated decision, wait/reject/ready state; no direct graph mutation | [rule_execution_runtime.py](../main_part/rule_execution_runtime.py) |
| Execution selection | Feasible mode/profile candidates and Control policy | Comparable-score decision and selection trace | [execution_selection.py](../main_part/execution_selection.py), [activation_policies.py](../main_part/activation_policies.py) |
| Knowledge alignment | Missing requirements, definitions, current facts | Provenance-bearing completion and Rule IDs for rematching | [semantic_alignment.py](../main_part/semantic_alignment.py) |
| Require recovery | Failed premises, current source and declared repair references | Gap classification, repair/clarification targets; no invented facts | [require_gaps.py](../main_part/require_gaps.py) |
| Path organization | Matched Rules, selectors, conflicts, claims | Bounded continuations and compatible alternatives | [rule_selector.py](../main_part/rule_selector.py), [rule_path_candidates.py](../main_part/rule_path_candidates.py) |
| Sequential composition | Declared child calls and parent return bindings | Bounded frames, checked returns and resumable waiting position | [rule_composition.py](../main_part/rule_composition.py) |
| Executor | Matched Rules and produced operations | Event changes, calculation results, continuations, failures | [executor_new.py](../main_part/executor_new.py) |
| Evidence lifecycle | Recorded premises, fingerprints and relations | Recursive validity, history-preserving retirement, alternative-support rebinding | [evidence_lifecycle.py](../main_part/evidence_lifecycle.py) |
| Calculator | Copied values/workspace and bindings | Contract-checked numeric output; no writable Working Memory | [calculator/service.py](../main_part/calculator/service.py) |
| Presentation | Control-approved runtime result | Scoped decoded answer or unresolved result; guarded wording | [output_service.py](../main_part/output_service.py), [output_decoder.py](../main_part/output_decoder.py), [answer_verbalizer.py](../main_part/answer_verbalizer.py) |

Control Rule declarations are stored in the Rule library. Scheduler executes their decisions through supported operations; it also contains fixed stage and resource orchestration. `RuleStrategyCompiler` compiles feedback programs, not business Rules into a PRS instruction set. The separate `RuleCompositionRuntime` executes declared child calls rather than compiling a PRS.

## 3 Data ownership

| Data | Storage | Meaning |
| --- | --- | --- |
| Knowledge | Portable `data/demo_v2/seed.json.gz`; writable runtime SQLite | Common core–relation–nature definitions and span schemas |
| Rules | Same portable seed; writable runtime SQLite | Reusable domain operations and Control policies |
| Event state | `WorkingMemoryState.event_workspaces` | Current task facts, supported completions, derived results, relations |
| Control state | `SchedulerNewState.control_state/control_context` | Active strategy and decision context |
| Execution runtime state | `WorkingMemoryState.rule_runtime[session_id]` | Candidates, authorizations, decisions, waiting, versions and composition frames |
| Runtime Memory | Runtime workspace/database; no historical seed distributed | Case-specific context, paths, and outcomes |
| Execution records | Working-memory entries and Scheduler histories | Actual attempts, decisions, errors, and outputs |

Semantic vectors do not replace exact IDs, values, units, or scope fields. Retrieval results enter context first; they are not automatically present-task facts. Historical paths can be preferred only with fresh matching.

[persistence/](../main_part/persistence/) provides SQLite, serialization, and indexing. [runtime_databases.py](../main_part/runtime_databases.py) calls [demo_seed.py](../main_part/demo_seed.py) to checksum-verify and restore the portable seed into fresh writable SQLite stores. The seed contains 210 inference Rules and 106 Knowledge records, with no historical Memory or learning Rules. Old `data/demo/db/` fixtures are not the default; aliases to those SQLite seeds are rejected at the inference entry. This is not a universal access policy for every utility.

The release retains and updates the previous 191 Rules, adding 19 generic inference Control Rules. It excludes 13 new broad financial-analysis business Rules whose coexistence interfered with supply-chain selection, as well as 36 learning Rules. This curated inventory must not be described as validation of the complete research registry.

## 4 One control/execution cycle

1. A frontend-ready input starts the Event-build stage and a `runtime_control_entry` selector.
2. Control Rules produce a strategy, more selectors, retrieval requests, or graph operations.
3. Retrieval/pre-match prepares candidates; Matcher checks current facts and establishes bindings.
4. For new-contract Rules, enumerate feasible execution modes and Q profiles, compute typed Activation signals and resolve their declared Q meanings. A partial mode needs an explicit restricted input/output contract, not merely a high coverage score.
5. When `execution_selection=control`, Scheduler requests an ordinary activation-selection Control Rule. Candidates must share the declared score meaning and scale (`comparison_key`); arbitrary confidence values are not interchangeable selection scores.
6. The execution runtime revalidates current bindings, authorization, Activation and Q before dispatch. Boolean gates, probability draws and frequency deadlines have different semantics. Legacy binary Rules still require full matching and do not receive inferred Q semantics.
7. Executor checks declared inputs/outputs/effects and applies graph operations or Calculator results. Sequential compositions invoke actual child Rules through the same matching, admission and execution chain, with depth and shared step bounds.
8. Results and recorded supports return to the Event. Control can continue, review evidence, inspect a bounded producer frontier, or classify a Require gap. Knowledge can prove completion, Memory can suggest paths, and source refinement can request missing information; every continuation must rematch.
9. A produced output remains a candidate until the applicable gate or strategy-completion check allows it. Presentation independently checks Event scope, approved targets and current support.

Knowledge alignment preserves the original value, unit, entity, and period and creates a derived semantic description. Its activation modes are `explicit_components`, `single_core`, `all_cores`, and `relation_context`. An unsupported completion is rejected.

[knowledge_schema_contract.py](../main_part/knowledge_schema_contract.py) requires subject/support roles and explicit scope binding where applicable. Knowledge activation modes describe definition applicability; they are not Rule Activation computation. [require_gaps.py](../main_part/require_gaps.py) distinguishes missing relations, declared derivations, source rechecks, unknown evidence, conflicts and false predicates. `MainPartModelLauncher.submit_require_answer` accepts an explicitly sourced clarification and resumes checking; a request is not evidence that its answer exists.

Q categories are `dispatch` (Boolean/probability/frequency), `selection` (score), `transform` (intensity/signed effect/weight), and `assessment` (confidence). No category is implicit and Q does not automatically mean a reinforcement-learning action value.

Matcher also supports explicit temporary require revision and removal of declared soft conditions. That mechanism is separate from alignment and must not be described as proof that a missing hard premise exists.

## 5 Execution, failure, and acceptance

Delivery retry is not reasoning recovery. Runtime retry restores pre-delivery in-memory state after a runtime exception; Control recovery chooses another reasoning continuation after a reported match/execution failure. Persistent and tool side effects are not guaranteed globally exactly-once.

`max_loops`, `max_rule_chain_steps`, selector hop limits, and bounded recovery attempts constrain execution. These are distinct budgets, not a complete implementation of the paper's abstract resource policy or global no-solution proof.

`evidence_lifecycle.attach_derivation` records actual matched premises and relevant relations. `evidence_validity` recursively requires all premises within a support and accepts any recorded valid alternative support. A Control-authorized review retires stale conclusions while preserving history or rebinds provenance to an existing valid support. Changed numerical conclusions still require recalculation. This is not arbitrary external-tool dependency maintenance or a complete ATMS.

`output_service` requires a solved task, an approved output, one identifiable Event scope, valid approved targets and supports, and consistent snapshots. It rejects ambiguous IDs and partial target decoding. Internal `execution_scope=partial` results cannot directly mark a task solved; a full user-facing partial-answer delivery path remains incomplete. The Decoder uses retained semantic fields and values, not inverse embedding reconstruction.

Waiting decisions and composition frames are serializable and can continue from restored runtime state; this does not provide a complete automatic disk-checkpoint CLI. Deterministic Calculator failures are cached by relevant input/expression identity, remain visible, and can be retried after those premises change.

Only unit-bearing numeric answers may use the fixed `Qwen/Qwen3-8B` wording client. Boolean, categorical, unitless, and unresolved results retain deterministic text. Numeric guard failures are recorded and fall back.

[runtime_status.py](../main_part/runtime_status.py) distinguishes solved and classified blocked outcomes from runtime failure, incomplete execution, and unresolved results. A completed delivery is not proof that the task was solved.

## 6 Theory-to-release boundaries

| Theory concept | Released mechanism | Boundary |
| --- | --- | --- |
| Semantic nomination and discrete admission | Vector retrieval plus required-pattern checks | Declared constraints and input grounding determine validity |
| Rebinding and path composition | Slot bindings, graph writeback, Rule selectors | Coverage and finite budgets limit what can be found |
| Recursive control organization | Bounded selectors, declared sequential child frames and checked returns | Not automatic intermediate-strategy generation or optimal dynamic depth selection |
| Activation and Q | Explicit signal graphs, interpretations, authorization and dispatch | Only declared contracts; scores are not automatically calibrated probabilities |
| Knowledge completion and Memory guidance | Span-schema alignment and opt-in historical preferences | No invented present facts or copied historical answers |
| Dependency-preserving revision | Recorded supports, recursive validity, Control review, retained history and alternative-support rebinding | Integrated Event operations only; not full ATMS or arbitrary external dependencies |
| Goal/evidence acceptance | Control output gate and scoped runtime projection | Not a proof of real-world factual truth or complete domain coverage |
| Capability-library learning | Not part of this inference demo | Auto-Learn, training, and Rule reconstruction excluded |
| Multimodal semantic input | Text frontend only | Visual grounding not implemented by this release |

Chronological history, selector ancestry, composition frames and derivation dependencies are different structures. None alone proves arbitrary dynamic hierarchy or complete truth maintenance. Internal partial results and formally delivered partial answers are also different capabilities.

## 7 Verification entry points

The linked tests provide executable checks, not benchmark results or broad capability claims:

- [test_rule_activation.py](../tests/test_rule_activation.py): signals, Q, execution permissions, waiting, partial modes and nested composition.
- [test_execution_selection.py](../tests/test_execution_selection.py): score comparability, selection and current-input checks.
- [test_require_gaps.py](../tests/test_require_gaps.py): actual Scheduler recovery and clarification/rematching.
- [test_evidence_lifecycle.py](../tests/test_evidence_lifecycle.py): support validity, retirement, alternatives and history.
- [test_context_contract_fixes.py](../tests/test_context_contract_fixes.py): scoped Knowledge support and invalidation.
- [test_deterministic_computation_recovery.py](../tests/test_deterministic_computation_recovery.py): unchanged computation failures and recovery.
- [test_portable_seed.py](../tests/test_portable_seed.py): checksum, independent restore and inference-only registry.

For the rationale and a DSO walkthrough, see the bilingual tech reports. For runnable commands and test scope, see the repository README. No paper figures, experiment reports, or cost comparisons are distributed by this documentation update.
