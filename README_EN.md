# Reasoning-AI-demo-v0.1

[中文 README](README.md)

This is a runnable Event Graph reasoning demonstration. The frontend LLM only translates natural-language input into a normalized Event Graph. The backend uses Knowledge to prove conditions that can be completed, applies the Rule network step by step, and renders only Control-approved results as a user-facing answer.

The project is released under the [Apache License 2.0](LICENSE).

The demo currently covers two connected domains:

- Finance: Days Sales Outstanding (DSO) calculation.
- Inventory and supply chain: inventory capital tied up, stockout exposure, and operating-cash-flow pressure.

The frontend performs semantic normalization and Event Graph construction only; it does not perform business reasoning. This release contains the inference system only. It excludes Auto-Learn, training, and Rule reconstruction, and a demo run never changes the released Rule or Knowledge databases.

## Technical overview

For the design rationale, feature representation, Knowledge/Rule separation, strict matching, Control, Calculator, Memory, the Output Gate, and the current boundaries, see:

- [Technical overview (English)](docs/TECHNICAL_EXPLANATION.ENGLISH.md)
- [技术原理说明（中文）](docs/TECHNICAL_EXPLANATION.CHINESE.md)
- [Architecture boundaries](docs/ARCHITECTURE.md)

## Quick start

Requirements: Python 3.11+, a working PyTorch environment, and a SiliconFlow API key. The frontend model is fixed to `Qwen/Qwen3-8B`.

```bash
git clone git@github.com:LvCheng09/Reasoning-AI-demo-v0.1.git
cd Reasoning-AI-demo-v0.1

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[demo,dev]"

# Load this only in the current terminal. Never upload a real key to GitHub.
source .env.example
export LLM_API_KEY="your-real-key"

python demo.py --preset dso
```

On the first run, the default local embedding model, `Qwen/Qwen3-Embedding-0.6B`, may be downloaded. If you already have a local model snapshot, set `MAIN_PART_EMBEDDING_MODEL` to that directory and set `MAIN_PART_EMBEDDING_LOCAL_FILES_ONLY=true`.

Available built-in cases:

```bash
python demo.py --preset inventory
python demo.py --preset stockout
python demo.py --preset cash-flow
python demo.py --preset missing-condition
python demo.py --question "Inventory turnover is slowing and safety stock is low. What could happen?" --json
```

Save a complete run record as JSON:

```bash
python demo.py --preset dso --output demo-results/dso.json
```

`data/demo/db/knowledge.sqlite` contains the core–relation–nature Knowledge network. `data/demo/db/rule.sqlite` contains the Rule network. Memory is a runtime experience layer: it is not shipped as historical data and demo runs do not write learning results.

## Verification

The following tests make no real API calls and do not download models:

```bash
python -m pytest tests/test_contracts.py tests/test_semantic_alignment.py tests/test_semantic_frontend.py tests/test_output_presentation.py tests/output_decoder/test_output_decoder.py
```

Live API demonstrations require your own key and depend on network access, provider availability, and model downloads. A normal demonstration uses the same `Qwen/Qwen3-8B` model twice: once to build the Event Graph and once only to verbalize the validated result. If the second call fails or changes a verified number or unit, the deterministic Decoder draft is shown instead. A single-case runtime is not a fixed performance benchmark.

## Current boundaries

- This is a two-domain research demo, not a general-purpose question-answering product.
- Rule and Knowledge quality and coverage define the capability ceiling. If a condition is not proven, the system must reject completion or change path rather than weaken a match.
- Validated presentation paths include DSO, inventory capital tied up, stockout exposure, cross-domain inventory-to-cash-flow effects, and a missing-condition rejection case. A live LLM service can still fail transiently; a retry may recover.
- This repository contains no API keys, model weights, old run records, training logs, or original project Git history.

## Contributing or publishing later changes

This repository is already connected to GitHub through SSH. After making a change:

```bash
git add .
git commit -m "Describe the change"
git push
```

Before pushing, confirm that `git status` does not contain `.env`, model directories, or `demo-results/`.
