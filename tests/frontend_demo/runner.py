"""Manual frontend test runner. Calls real services only when explicitly run."""
import argparse
import json
import os
from pathlib import Path
from dataclasses import asdict
from semantic_frontend import SimpleSemanticFrontend, SingleSmallLLM
from semantic_frontend.embedding import SentenceTransformerEmbedder
from .checks import run_case

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cases",
        default=str(Path(__file__).resolve().parent / "sample_cases.jsonl"),
    )

    parser.add_argument(
        "--base-url",
        default=os.getenv(
            "LLM_BASE_URL",
            "http://localhost:8000/v1",
        ),
    )

    parser.add_argument(
        "--llm-model",
        default=os.getenv(
            "LLM_MODEL",
            "Qwen/Qwen3-8B",
        ),
    )

    parser.add_argument(
        "--api-key",
        default=os.getenv(
            "LLM_API_KEY",
            "local",
        ),
    )

    parser.add_argument(
        "--embedding-model",
        default=os.getenv(
            "EMBEDDING_MODEL",
            "Qwen/Qwen3-Embedding-0.6B",
        ),
    )

    parser.add_argument(
        "--device",
        default=os.getenv("EMBEDDING_DEVICE") or None,
    )

    parser.add_argument(
        "--output",
        default="results.json",
    )

    args = parser.parse_args()

    llm = SingleSmallLLM(
        base_url=args.base_url,
        model=args.llm_model,
        api_key=args.api_key,
    )

    embedder = SentenceTransformerEmbedder(
        model_name_or_path=args.embedding_model,
        device=args.device,
    )

    frontend = SimpleSemanticFrontend(
        llm=llm,
        embedder=embedder,
    )

    cases: list[dict[str, Any]] = []
    with open(args.cases, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    results = []

    for case in cases:
        print(f"[RUN] {case['id']}")
        results.append(run_case(frontend, case))

    payload = {
        "configuration": {
            "frontend": "single_fixed_small_llm",
            "llm_model": args.llm_model,
            "llm_base_url": args.base_url,
            "embedding_model": args.embedding_model,
            "model_router": False,
        },
        "results": results,
        "total_llm_usage": asdict(llm.usage),
    }

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        json.dumps(
            {
                "cases": len(results),
                "total_llm_usage": asdict(llm.usage),
                "output": args.output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
