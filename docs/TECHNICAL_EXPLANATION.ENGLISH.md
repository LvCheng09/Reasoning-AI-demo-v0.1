# Technical Overview: Explicit Event-Graph Reasoning

This document explains how the inference portion of Demo v0.1 works, why its boundaries are designed this way, and what it deliberately does not claim. For installation and commands, see the repository [README](../README.md).

## 1. The problem being addressed

Natural language has many surface forms, while a verifiable answer must rest on explicit conditions, relations, and computations. The system therefore separates a task into two layers:

```text
language understanding: source text -> normalized Event Graph
explicit reasoning: Event Graph -> Knowledge / Rule Path -> result
```

The frontend LLM handles the first layer: it understands alternative wording, identifies stated facts, and organizes them into structure. The backend does not treat that text generation as an answer. It strictly matches Knowledge and Rules against the graph, then derives a result step by step.

This is not an attempt to replace LLMs with rules. The two components solve different problems. LLMs are useful for translating diverse language into a common representation; rules are useful for deterministic, inspectable reasoning once their conditions hold.

## 2. From language to features

An object may be described by one or many features. A more detailed description contains more features, while a name is a high-level abstraction over a feature set. For example, a screen may be described only as an object, or more specifically as black, luminous, and able to display an image.

People communicate with names, but reasoning relies on explicit properties and relations: whether a feature is present, whether several features hold together, and whether one property entails another. The Event Graph preserves both language-level names and the more specific features that become the state of reasoning.

The primary semantics of a span have three parts:

```text
core      : what it is, for example accounts_receivable, sales, inventory
relation  : how it occurs, for example temporal_role, measurement, scope
nature    : the property of that relation, for example beginning, annual, net_credit
```

For example, “beginning accounts receivable” can be represented as:

```text
core = accounts_receivable
relation = temporal_role
nature = beginning
```

The same core may have different meanings under different relations and natures. The system therefore does not rely on equal keywords, nor does it treat every similar phrase as the same concept.

## 3. Vectors, text, and the role of the LLM

Text fields are used for debugging, logging, and user-facing explanations. The vectors for `core`, `relation`, and `nature` are the main semantic representation used for retrieval and strict matching. The current demo uses Qwen embeddings to map normalized semantics into vectors.

An embedding is not proof that a fact is true. It allows different wording to be compared in one semantic space. A condition may enter a reasoning path only after it also satisfies the Rule's structural, relational, unit, temporal, entity, and threshold constraints.

The frontend LLM acts more like an encoder for real-world language. Its prompt defines what information to extract, how detailed the graph should be, and how to distinguish stated facts from unstated inferences. If the backend determines that the source text is genuinely incomplete, it can make a targeted frontend request for a missing text fact. The frontend still does not make a business conclusion.

The same boundary—first encode an input into structure, then reason explicitly—could be extended to visual input. This release validates text input only.

## 4. Event Graph and Contract

The frontend produces an Event Graph containing spans, inter-span relations, source evidence, values, units, entities, time, and confidence. The backend first applies the Event Graph Contract, which checks:

- JSON shape and required fields;
- IDs, relation endpoints, and evidence references;
- normalizable confidence, list, and object fields;
- vector-field compatibility with the declared contract.

The Contract checks structure and provenance. It does not pretend to be a second subjective judge of whether the LLM “understood” a sentence. The frontend prompt prohibits inventing facts; the backend Knowledge and Matcher decide whether a graph item may actually be used as a reasoning condition.

## 5. Knowledge: proving that a missing condition is present

Knowledge stores definitions of concepts and local semantic structure: how cores, relations, natures, and related features form a recognizable span. It does not store a direct business transformation from input to final answer.

When a Rule's `require` does not match, the system does not lower the matching bar. Control can query a Knowledge schema for the missing slot:

```text
Does the current graph already contain every premise required by this definition?
```

If the answer can be proven, the system adds a normalized span or relation with provenance and **strictly rematches the original Rule**. Otherwise, it rejects that Rule and either tries another path or requests missing source information.

Knowledge schemas support several activation modes:

- **single core**: the core alone proves a declared property;
- **multiple cores**: every declared core must be present before a property is added;
- **relation context**: the meaning of one core is resolved by its relation and adjacent cores.

For example, observing `sales` never automatically creates `total_sales`. That completion is allowed only when every premise in the `total_sales` definition is present in the current graph. This is completion after proof, not similarity-based admission.

## 6. Rules: explicit bridges from conditions to results

A Rule bridges Knowledge spans with the following form:

```text
require  ->  produced
```

- `require`: required spans, relations, units, periods, entities, or prior results;
- `produced`: a new span/relation, a calculation call, a next-path choice, or an output candidate.

One possible local DSO path is:

```text
beginning AR + ending AR -> average AR
average AR + net credit sales + days -> DSO
DSO + unit/evidence checks -> output candidate
```

Complex questions are assembled from small Rules. Each step exposes its input, output, and basis rather than hiding all logic in one large, opaque rule.

## 7. Boundaries of Control, Matcher, Executor, and Calculator

```text
Control    : decides the current goal, gap, and next strategy
Matcher    : strictly determines whether a Rule's require is actually true
Executor   : carries out the produced operation of an already matched Rule
Calculator : performs the deterministic numerical operation declared by a Rule
```

Control provides the broad strategy: identify critical conditions, choose viable paths, and when none is executable, determine what is missing, whether Knowledge can prove it, or whether another path is needed. It does not replace domain computation.

Matcher verifies every required slot binding, whether supposedly distinct slots incorrectly share a span, relation endpoints, vector contracts, thresholds, and limits such as unit, period, and entity. A failed Rule leaves a structured gap for Control but is never treated as a successful match merely because it is “close enough.”

Executor does not interpret domain meaning or guess a next step. It converts matched Rules into Event Graph updates, calculator calls, next selectors, or output candidates. Domain-level Rule changes therefore do not require Executor to understand every domain.

Calculator is separate from Executor so that deterministic formulas, possible future controlled small models, and graph writes remain independent. Current demo conclusions rely on deterministic, auditable calculations; a general neural network is not used as a replacement for business reasoning.

## 8. The role of Memory

Knowledge, Rules, and Memory store different layers of information:

```text
Knowledge : common concepts and their local semantic structures
Rule      : reusable state transformations
Memory    : how one specific prior task chose a path and what happened
```

A historical Memory can let Control try a similar successful path first, but it cannot bypass Matcher. The current Event Graph must match again; if that path fails, the system returns to ordinary candidate paths. This prevents a past special case from being copied to a new task with different conditions.

Demo v0.1 neither ships historical Memory nor writes learning results.

## 9. Multi-step execution and the output gate

Each Rule's `produced` operation is written back to Working Memory/Event Graph and becomes input for the next iteration:

```text
Control -> Match -> Execute -> updated Event Graph -> Control
```

This supports intermediate results and multi-step reasoning. A final result is still only an output candidate at first. The Output Gate checks the active control strategy's completion conditions, result granularity, path, and necessary constraints. A rejected candidate is revised or blocked.

Only an allowed result is passed to the Output Decoder. The decoder extracts the final result span and its supporting facts. The later LLM Verbalizer only restates that small structured payload; it cannot access Knowledge/Rule databases or run reasoning again. If the text layer changes a verified number or unit, the system falls back to a deterministic draft.

## 10. A DSO example

```text
Input: average accounts receivable 120, net credit sales 960,
       and 360 days in the year. Calculate DSO.

1. The frontend LLM builds evidence-linked AR, net-credit-sales, and days spans.
2. The Contract checks graph and evidence structure.
3. Control recognizes a DSO goal and selects the finance strategy.
4. Matcher strictly binds the three input conditions.
5. Calculator executes: 120 / 960 × 360 = 45.
6. Executor writes dso_result = 45 days and preserves inputs and Rule path.
7. After the Output Gate allows it, the output layer renders the answer.
```

If net credit sales are absent, ordinary sales are not silently treated as net credit sales. The system can complete that condition only when Knowledge proves the relevant scope from the current graph; otherwise it switches path, requests information, or reports the missing condition.

## 11. Current capability and boundaries

The following components are integrated and have engineering-test support: Event Graph Contract, Knowledge-driven completion followed by strict rematching, stepwise Rule execution, Calculator, output gating, and the independent text-output layer. DSO and inventory-capital-tied-up have both completed live API end-to-end demonstrations.

The following boundaries remain important:

- This is a research demo for two connected domains, not an open-ended general question-answering product.
- The frontend LLM can still translate text incorrectly; the Contract does not guarantee perfect natural-language understanding.
- Missing Knowledge/Rule coverage cannot be solved by weakening a match.
- Control can compose and switch paths incrementally, but this release does not claim globally optimal multi-step planning.
- Auto-Learn, training, Rule reconstruction, and a PRS compiler are outside this release.

## 12. Code map

| Component | Implementation |
| --- | --- |
| Frontend and main entry | `main_part/model_launcher.py`, `semantic_frontend/` |
| Event Graph Contract | `main_part/event_graph_contract.py` |
| Scheduler, Control, Output Gate | `main_part/scheduler_new.py` |
| Knowledge-driven completion | `main_part/semantic_alignment.py`, `main_part/retrieval.py` |
| Strict matching | `main_part/matcher/matching.py` |
| Rule paths | `main_part/rule_selector.py`, `main_part/rule_path_candidates.py` |
| Execution and calculation | `main_part/executor_new.py`, `main_part/calculator/` |
| Result decoding and verbalization | `main_part/output_service.py`, `main_part/output_decoder.py`, `main_part/answer_verbalizer.py` |

For a shorter module-boundary reference, see [ARCHITECTURE.md](ARCHITECTURE.md).
