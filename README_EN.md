<p align="center">
  <a href="assets/reasoning-overview.png"><img src="assets/reasoning-overview.png" alt="Reasoning-AI concept: attributes and relations form event states; composable rules reason over those explicit states." width="800"></a>
</p>

<h1 align="center">Reasoning-AI</h1>

<p align="center">
  <strong>Explicit event states · Composable rules · Traceable reasoning</strong><br>
  Translate language into an Event Graph. Compose a Rule Network to solve it.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-111827?style=flat-square" alt="License: Apache 2.0"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python 3.11+"></a>
  <a href="#scope-and-limitations"><img src="https://img.shields.io/badge/Release-v0.1-111827?style=flat-square" alt="Demo v0.1"></a>
  <a href="#scope-and-limitations"><img src="https://img.shields.io/badge/Scope-Inference_only-6B7280?style=flat-square" alt="Inference only"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> ·
  <a href="#demo-tasks">Demo Tasks</a> ·
  <a href="tech%20report/TECH_REPORT.ENGLISH.md">Tech Report</a> ·
  <a href="#scope-and-limitations">Scope</a> ·
  <a href="README.md">中文</a>
</p>

---

## Overview

Reasoning-AI is a runnable inference demo of the ERS approach. A frontend LLM translates natural language into a structured Event Graph. The backend matches, composes, and executes explicit rules over current evidence, then presents only output-gated results.

The central distinction: **semantic similarity nominates candidates; declared conditions and current evidence determine whether they may execute.**

This is a research demo for finance and inventory / supply chain, not a general-purpose question-answering product. The release includes inference only, not Auto-Learn, training, or Rule reconstruction.

## Why an explicit Rule Network?

- **Expose the problem state.** Represent facts, properties, goals, and intermediate results as spans, relations, and Event Graphs.
- **Compose reusable rules.** Control organizes strategies and subsequent Rules; Executor performs operations; Calculator handles computation.
- **Separate supported completion from guessing.** Knowledge supplies grounded concept/property completion; Memory supplies prior experience. Neither should substitute for missing current facts.
- **Connect conclusions to evidence.** Final output requires gate approval and valid references to results in the current Event, rather than treating every generated item as an answer.

Explicit checks do not guarantee that frontend interpretation or library content is always correct. Capabilities still depend on input quality and Knowledge / Rule coverage; see [scope and limitations](#scope-and-limitations).

## Demo tasks

| Domain | Built-in task | Command |
|---|---|---|
| Finance | Days Sales Outstanding (DSO) | `python demo.py --preset dso` |
| Inventory | Slower turnover and capital tied up | `python demo.py --preset inventory` |
| Supply chain | Demand volatility, safety stock, and stockout exposure | `python demo.py --preset stockout` |
| Cross-domain | Inventory effects on operating cash flow | `python demo.py --preset cash-flow` |
| Boundary diagnostic | Avoiding unsupported conclusions when conditions are missing | `python demo.py --preset missing-condition` |

The DSO example has the following business calculation. This illustrates the result, not complete terminal output or a performance benchmark:

```text
Average accounts receivable: 120 million USD
Net credit sales:            960 million USD
Period convention:           360 days

DSO = 120 / 960 * 360 = 45 days
```

> Numeric results with units may use LLM wording. Non-numeric judgments retain the deterministic Decoder draft. The missing-condition preset is diagnostic; this release does not always produce an explicit blocked classification.

## Quick start

### 1. Install

Requirements: Python 3.11+, a working PyTorch environment, and network access plus an API key for SiliconFlow. The following commands target macOS / Linux shells:

```bash
git clone https://github.com/LvCheng09/Reasoning-AI-demo-v0.1.git
cd Reasoning-AI-demo-v0.1

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[demo,dev]"
```

The first run may download the local embedding model, `Qwen/Qwen3-Embedding-0.6B`. The generation model is fixed to `Qwen/Qwen3-8B`; these models serve different roles.

### 2. Configure

```bash
source .env.example
export LLM_API_KEY="your-real-key"
```

`.env.example` already sets the SiliconFlow endpoint and required model. Never put a real key in a README, source file, or commit. Live API calls may incur provider charges.

### 3. Run

```bash
python demo.py --preset dso
```

Ask your own question or save a run record:

```bash
python demo.py --question "某公司库存周转速度明显变慢。请判断是否会形成库存资金占用。" --json
python demo.py --preset dso --output demo-results/dso.json
```

This custom-input example is deliberately Chinese: English grounding and refinement remain known limitations. Custom questions are constrained by the current Knowledge and Rule libraries, not unrestricted question answering.

<details>
<summary><strong>Optional configuration: local model, timeouts, and retries</strong></summary>

For an existing local embedding snapshot, set `MAIN_PART_EMBEDDING_MODEL` to its directory and `MAIN_PART_EMBEDDING_LOCAL_FILES_ONLY=true`.

| Environment variable | Purpose |
|---|---|
| `LLM_READ_TIMEOUT_SECONDS` | Generation-request read timeout |
| `LLM_MAX_OUTPUT_TOKENS` | Frontend output-token limit |
| `LLM_MAX_RETRIES` | Frontend temporary-request retry limit |
| `MAIN_PART_EMBEDDING_MODEL` | Embedding model name or local directory |
| `MAIN_PART_EMBEDDING_LOCAL_FILES_ONLY` | Load an already cached local model only |

These settings do not change the generation model. Text entry supports only `semantic_v1` and `Qwen/Qwen3-8B`. See [.env.example](.env.example) for defaults.

</details>

## How it works and technical reports

The frontend translates meaning; it does not calculate business answers. The backend receives an Event Graph through a Python data interface, uses local vectors to discover candidates, and checks and executes Rule conditions. Presentation receives only approved results.

| Component | Responsibility |
|---|---|
| Semantic Frontend | Language normalization, Event Graph construction, and targeted refinement |
| Knowledge / Memory | Concept/property support / specific historical experience |
| Control / Matcher | Strategy organization, candidate nomination, current-condition checks, and binding |
| Executor / Calculator | Rule execution, computation calls, and updates to Event state |
| Output Gate / Decoder | Final result-reference checks and decoding to structured output and text |

- [English Tech Report](tech%20report/TECH_REPORT.ENGLISH.md): rationale, data structures, module interfaces, and implementation.
- [中文 Tech Report](tech%20report/TECH_REPORT.CHINESE.md): the corresponding Chinese report.
- [Architecture and runtime interfaces](tech%20report/ARCHITECTURE.md): component boundaries and source entry points.
- [Release notes](tech%20report/RELEASE_NOTES.md): changes to the released demo.

Terminology follows ERS theory, but capability statements refer to the code in this repository. Mechanisms from the paper or isolated research experiments are not automatically released capabilities.

## Verification

### Offline engineering tests

No real LLM API calls or model downloads:

```bash
MAIN_PART_FRONTEND_VECTORIZE=0 python -m pytest
```

[GitHub Actions](https://github.com/LvCheng09/Reasoning-AI-demo-v0.1/actions/workflows/tests.yml) uses the same offline test entry. An engineering test count is not a count of reasoning tasks.

### Real-backend regression

Requires a locally cached embedding model. Uses authored Event Graphs, real vectors, the actual Launcher / Scheduler, Matcher, Executor, and output gates. It makes no external LLM calls or model downloads:

```bash
python -m tools.validate_release_runtime --output demo-results/release-check.json
```

Use a new output filename each time; old records are not overwritten. This regression is not a live frontend evaluation, blind test, or generalization proof. Text-to-answer runs also depend on extraction, refinement contracts, provider availability, and the network.

<details>
<summary><strong>Reading runtime status and output</strong></summary>

JSON reports include `execution_outcome`, `task_result`, frontend and derived spans, executed Rules, answers, and fallback information. Interpret the states separately:

| State | Meaning |
|---|---|
| `solved` | Control explicitly confirms a solution |
| `blocked` | Control explicitly confirms a block under current conditions; not a solved question |
| `incomplete` / `unresolved` | Execution is incomplete or the result is unconfirmed |
| `runtime_failed` | A runtime failure occurred |

Exit codes: `0` for solved or blocked; `1` for runtime failure; `2` for incomplete / unconfirmed results. Process exit or a Scheduler status of completed does not automatically mean solved.

Reports separate initialization, frontend / input context, graph boundary, backend wall / process-CPU time, output presentation, and provider-reported token usage. Missing usage remains unknown, not free. These records are not FLOPs, energy, or price estimates.

Numeric results with units may use the same fixed model for wording. Request failures or changes to protected numbers / units trigger a reported fallback to the Decoder draft. Textual, boolean, and unitless judgments do not receive free-form LLM rewriting. Numeric checks do not prove complete semantic fidelity.

</details>

<details>
<summary><strong>Database isolation and runtime files</strong></summary>

`data/demo/db/knowledge.sqlite` holds the core–relation–nature Knowledge network; `data/demo/db/rule.sqlite` holds the Rule network.

These are release seeds. The demo and default Python entry create independent writable copies under `runtime/inference_*`; retrieval counters, indexes, and runtime Memory are written only to those copies. Historical Memory is not shipped, and demo runs do not perform learning.

`MAIN_PART_DB_DIR`, if set, must point to existing writable runtime databases, never the released seeds. Explicit `db_paths` also reject seed paths, including symlinks and hardlinks. Graph input requires no API configuration, but must satisfy graph and vector contracts.

Final output requires gate approval, an unambiguous current Event, and valid result references. Missing, conflicting, or stale references must not cause arbitrary graph results to be selected as answers.

`runtime/`, `demo-results/`, credentials, and model weights are not release content. Do not commit them.

</details>

## Scope and limitations

- **Research demo, not a production system.** Covers finance and inventory / supply chain; it does not promise arbitrary-domain or arbitrary-question support.
- **Frontend errors remain possible.** Exact grounding of English quotations and refinement format / polarity still need improvement. Do not assume stable English end-to-end behavior.
- **Missing conditions do not always yield explicit blocks.** Some tasks may remain unknown / unresolved or reach a loop limit. These are not solved cases or explicit blocked outcomes.
- **Non-numeric presentation remains structured.** The default may display an English Decoder draft rather than natural Chinese prose.
- **Training is not released.** Auto-Learn, training, and Rule reconstruction are excluded. Dynamic intermediate strategies, full subgoal invocation / return, and general evidence invalidation / rebinding are not released default capabilities either.
- **No superiority claim.** Single-case time, Rule-step counts, or isolated experiments do not establish broadly better performance than other models. Results depend on Knowledge, Rules, inputs, and environment.

## Contributing

Use [Issues](https://github.com/LvCheng09/Reasoning-AI-demo-v0.1/issues) for reproducible problems, domain cases, and suggestions, or open a Pull Request.

Include the question, expected behavior, runtime status, and relevant sanitized logs. Do not upload API keys, customer-private data, model weights, or full private runtime databases.

Released under [Apache License 2.0](LICENSE). Third-party models and services remain subject to their respective licenses and terms.
