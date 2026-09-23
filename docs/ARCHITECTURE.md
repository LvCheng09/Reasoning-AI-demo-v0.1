# Architecture

## Boundaries

`semantic_frontend/` and `main_part/` are separate modules.

- `semantic_frontend/` receives text and returns a structured Event Graph. It uses one configured LLM endpoint and does not choose business Rules or produce a business conclusion.
- `main_part/` validates that graph, retrieves Knowledge and Rules, runs the Scheduler, and writes derived results back into the graph.
- `demo.py` is composition code only: it configures the two modules, opens the released databases, and prints the result. It does not contain domain reasoning.

## Data layers

| Layer | Released artifact | Role |
| --- | --- | --- |
| Knowledge | `data/demo/db/knowledge.sqlite` | Vector-based core–relation–nature definitions used to prove whether a missing required condition is present in the current graph. |
| Rule | `data/demo/db/rule.sqlite` | Rules that transform strictly matched input spans into derived spans or numeric outputs. |
| Memory | Runtime only | Historical cases and execution choices. It is not shipped; 本版本不包含学习模块。 |

## Execution rule

When a Rule requirement does not match, the system first asks whether the current Event Graph already satisfies the complete Knowledge definition. If yes, it adds a provenance-bearing completion and strictly matches again. If not, it does not weaken the requirement; Control selects another Rule or strategy. Only a missing text fact can request targeted frontend refinement.
