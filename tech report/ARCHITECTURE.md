# Architecture and Runtime Interfaces

[中文技术报告](TECH_REPORT.CHINESE.md) · [English Tech Report](TECH_REPORT.ENGLISH.md)

This is the code map for the released inference demo. The ERS theory separates semantic candidate discovery, explicit state, local execution, and task-dependent control. Those roles inform this implementation, but research-only dynamic hierarchy and dependency revision are not automatically part of this repository.

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
   |         |           |
retrieval   matcher/    executor_new -> calculator/
   |         |           |
Knowledge  bindings    graph updates, feedback, selectors
Memory hints            |
          +-------------+
          |
Output Gate -> output_service -> output_decoder -> answer_verbalizer
```

The frontend/backend interface is in-process Python data. Only provider clients use HTTP. `submit_event_graph` bypasses the LLM but not validation, workspace delivery, or reasoning.

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
| Knowledge alignment | Missing requirements, definitions, current facts | Provenance-bearing completion and Rule IDs for rematching | [semantic_alignment.py](../main_part/semantic_alignment.py) |
| Path organization | Matched Rules, selectors, conflicts, claims | Bounded continuations and compatible alternatives | [rule_selector.py](../main_part/rule_selector.py), [rule_path_candidates.py](../main_part/rule_path_candidates.py) |
| Executor | Matched Rules and produced operations | Event changes, calculation results, continuations, failures | [executor_new.py](../main_part/executor_new.py) |
| Calculator | Copied values/workspace and bindings | Contract-checked numeric output; no writable Working Memory | [calculator/service.py](../main_part/calculator/service.py) |
| Presentation | Control-approved runtime result | Scoped decoded answer or unresolved result; guarded wording | [output_service.py](../main_part/output_service.py), [output_decoder.py](../main_part/output_decoder.py), [answer_verbalizer.py](../main_part/answer_verbalizer.py) |

Control Rule declarations are stored in the Rule library. Scheduler executes their decisions through supported operations; it also contains fixed stage and resource orchestration. `SchedulerInstructionCompiler` compiles feedback programs, not business Rules into a PRS instruction set.

## 3 Data ownership

| Data | Storage | Meaning |
| --- | --- | --- |
| Knowledge | Portable `data/demo_v2/seed.json.gz`; writable runtime SQLite | Common core–relation–nature definitions and span schemas |
| Rules | Same portable seed; writable runtime SQLite | Reusable domain operations and Control policies |
| Event state | `WorkingMemoryState.event_workspaces` | Current task facts, supported completions, derived results, relations |
| Control state | `SchedulerNewState.control_state/control_context` | Active strategy and decision context |
| Runtime Memory | Runtime workspace/database; no historical seed distributed | Case-specific context, paths, and outcomes |
| Execution records | Working-memory entries and Scheduler histories | Actual attempts, decisions, errors, and outputs |

Semantic vectors do not replace exact IDs, values, units, or scope fields. Retrieval results enter context first; they are not automatically present-task facts. Historical paths can be preferred only with fresh matching.

[persistence/](../main_part/persistence/) provides SQLite, serialization, and indexing services. [runtime_databases.py](../main_part/runtime_databases.py) copies released seeds before normal inference and rejects seed aliases through symbolic or hard links. Retrieval counters, indexes, and runtime Memory are written to the copy. This protection applies at the inference entry, not as a universal access policy for every utility.

## 4 One control/execution cycle

1. A frontend-ready input starts the Event-build stage and a `runtime_control_entry` selector.
2. Control Rules produce a strategy, more selectors, retrieval requests, or graph operations.
3. Retrieval/pre-match prepares candidates; Matcher checks current facts and establishes bindings.
4. Conflicting matches become bounded alternatives requiring Control selection; valid operations proceed to Executor.
5. Executor writes accepted operation results, which can become the next Rule's inputs.
6. Unmet requirements or execution errors become Control feedback. Knowledge may prove an alignment, Memory may guide another path, or a scoped frontend patch may clarify source text.
7. New facts and bindings are checked again. A produced output remains a candidate until the applicable gate or strategy-completion check allows it.

Knowledge alignment preserves the original value, unit, entity, and period and creates a derived semantic description. Its activation modes are `explicit_components`, `single_core`, `all_cores`, and `relation_context`. An unsupported completion is rejected.

Matcher also supports explicit temporary require revision and removal of declared soft conditions. That mechanism is separate from alignment and must not be described as proof that a missing hard premise exists.

## 5 Execution, failure, and acceptance

Delivery retry is not reasoning recovery. Runtime retry restores pre-delivery in-memory state after a runtime exception; Control recovery chooses another reasoning continuation after a reported match/execution failure. Persistent and tool side effects are not guaranteed globally exactly-once.

`max_loops`, `max_rule_chain_steps`, selector hop limits, and bounded recovery attempts constrain execution. These are distinct budgets, not a complete implementation of the paper's abstract resource policy or global no-solution proof.

`output_service` requires a solved task, an approved output, one identifiable Event scope, valid approved targets, and consistent snapshots. It rejects ambiguous IDs and partial target decoding. The Decoder uses retained semantic fields and values; it does not reconstruct language by inverting vectors.

Only unit-bearing numeric answers may use the fixed `Qwen/Qwen3-8B` wording client. Boolean, categorical, unitless, and unresolved results retain deterministic text. Numeric guard failures are recorded and fall back.

[runtime_status.py](../main_part/runtime_status.py) distinguishes solved and classified blocked outcomes from runtime failure, incomplete execution, and unresolved results. A completed delivery is not proof that the task was solved.

## 6 Theory-to-release boundaries

| Theory concept | Released mechanism | Boundary |
| --- | --- | --- |
| Semantic nomination and discrete admission | Vector retrieval plus required-pattern checks | Declared constraints and input grounding determine validity |
| Rebinding and path composition | Slot bindings, graph writeback, Rule selectors | Coverage and finite budgets limit what can be found |
| Recursive control organization | Bounded selector delegation and recovery | Not a full subgoal invocation/return stack or research dynamic intermediate planner |
| Knowledge completion and Memory guidance | Span-schema alignment and opt-in historical preferences | No invented present facts or copied historical answers |
| Dependency-preserving revision | Alignment fingerprints, computation sources, strategy history, output snapshots | No released general invalidation/rebinding ledger with independent-support maintenance |
| Goal/evidence acceptance | Control output gate and scoped runtime projection | Not a proof of real-world factual truth or complete domain coverage |
| Capability-library learning | Not part of this inference demo | Auto-Learn, training, and Rule reconstruction excluded |
| Multimodal semantic input | Text frontend only | Visual grounding not implemented by this release |

Chronological history, selector ancestry, and derivation dependencies are different structures. None alone proves the complete dynamic hierarchy or truth-maintenance mechanism described in the paper.

For the rationale and a DSO walkthrough, see the bilingual tech reports. For runnable commands and test scope, see the repository README. No paper figures, experiment reports, or cost comparisons are distributed by this documentation update.
