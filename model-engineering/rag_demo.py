"""RAG demonstration with Gemini Flash and LangChain.

This script follows the same artifact pattern as llm_demo.py while adding
retrieval over local markdown reports.

Artifacts per run:
- response.txt
- metadata.json
- params.json
- bundle_info.json
- retrieved_chunks.json
- runs/summary.csv (append row)

Usage:
    python rag_demo.py --config configs/llm/gemini.yaml \
                       --prompt-config configs/prompts/rag_grounded_answer.yaml \
                       --query "How does this repo explain RAG?"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.common import ensure_dir, load_yaml


BUDGET_PROFILES = {
    "low": {
        "top_k": 2,
        "chunk_size": 700,
        "chunk_overlap": 80,
        "max_tokens": 500,
    },
    "balanced": {
        "top_k": 3,
        "chunk_size": 900,
        "chunk_overlap": 100,
        "max_tokens": 700,
    },
    "detailed": {
        "top_k": 4,
        "chunk_size": 1200,
        "chunk_overlap": 150,
        "max_tokens": 1200,
    },
}


def load_api_key_from_dotenv(env_name: str) -> str | None:
    """Load API key from project-level .env file when available."""
    dotenv_path = Path(__file__).resolve().parent / ".env"
    if not dotenv_path.exists():
        return None

    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(f"{env_name}="):
            value = stripped.split("=", 1)[1].strip().strip('"').strip("'")
            return value or None
    return None


def load_markdown_documents(docs_dir: Path) -> list[Document]:
    """Read markdown files recursively and convert to LangChain documents."""
    docs: list[Document] = []
    for path in sorted(docs_dir.rglob("*.md")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={"source": str(path).replace("\\", "/")},
            )
        )
    return docs


def resolve_docs_dir(user_value: str, fallback_parent: str = "reports") -> Path:
    """Resolve docs directory, supporting shorthand names like 'demo_data'.

    Resolution order:
    1) user_value
    2) fallback_parent / user_value
    """
    direct = Path(user_value)
    fallback = Path(fallback_parent) / user_value

    if direct.exists() and direct.is_dir():
        direct_docs = load_markdown_documents(direct)
        if direct_docs:
            return direct
        if fallback.exists() and fallback.is_dir() and load_markdown_documents(fallback):
            return fallback
        return direct

    if fallback.exists() and fallback.is_dir():
        return fallback

    return direct


def resolve_path(user_value: str, fallback_dir: str) -> Path:
    """Resolve a user-supplied path, supporting shorthand filenames.

    If the provided value does not exist as-is, this checks fallback_dir/value.
    """
    direct = Path(user_value)
    if direct.exists():
        return direct

    fallback = Path(fallback_dir) / user_value
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Could not find file '{user_value}'. Tried '{direct}' and '{fallback}'."
    )


def chunk_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split documents into retrieval-friendly chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def get_effective_rag_settings(
    budget_mode: str,
    cli_top_k: int | None,
    cli_chunk_size: int | None,
    cli_chunk_overlap: int | None,
    configured_max_tokens: int | None,
) -> dict[str, int | None]:
    """Resolve runtime settings from budget mode and optional CLI overrides."""
    defaults = BUDGET_PROFILES[budget_mode]
    budget_cap = defaults["max_tokens"]
    resolved_max_tokens = budget_cap
    if configured_max_tokens is not None:
        resolved_max_tokens = min(int(configured_max_tokens), budget_cap)

    return {
        "top_k": cli_top_k if cli_top_k is not None else defaults["top_k"],
        "chunk_size": cli_chunk_size if cli_chunk_size is not None else defaults["chunk_size"],
        "chunk_overlap": (
            cli_chunk_overlap if cli_chunk_overlap is not None else defaults["chunk_overlap"]
        ),
        "max_tokens": resolved_max_tokens,
    }


def infer_rag_budget_mode(query: str) -> dict[str, str]:
    """Infer retrieval budget mode from query complexity and intent."""
    text = query.strip().lower()

    broad_patterns = [
        r"\bcompare|contrast|trade[- ]?off|pros and cons|summarize all|comprehensive\b",
        r"\barchitecture|pipeline|end[- ]to[- ]end|deep dive|explain in detail\b",
        r"\bmultiple|across documents|from all reports\b",
    ]
    precise_patterns = [
        r"\bwhat is|define|when|where|who\b",
        r"\bbrief|short|one line|quick\b",
    ]

    if any(re.search(pattern, text) for pattern in broad_patterns) or len(text) > 180:
        return {
            "budget_mode": "detailed",
            "reason": "broad or multi-part query detected",
        }

    if any(re.search(pattern, text) for pattern in precise_patterns) and len(text) <= 120:
        return {
            "budget_mode": "low",
            "reason": "narrow factual query detected",
        }

    return {
        "budget_mode": "balanced",
        "reason": "default complexity",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini Flash RAG demo with LangChain")
    parser.add_argument("--config", required=True, help="Path to configs/llm/*.yaml")
    parser.add_argument("--prompt-config", required=True, help="Path to configs/prompts/*.yaml")
    parser.add_argument("--query", required=True, help="User question for grounded answering")
    parser.add_argument("--docs-dir", default="reports", help="Knowledge base directory")
    parser.add_argument(
        "--budget-mode",
        choices=["auto", "low", "balanced", "detailed"],
        default="auto",
        help="Token budget profile for retrieval + generation (auto by query)",
    )
    parser.add_argument(
        "--disable-auto-route",
        action="store_true",
        help="Disable automatic query-based budget routing",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieved chunk count")
    parser.add_argument("--chunk-size", type=int, default=None, help="Override chunk size in characters")
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Override chunk overlap in characters",
    )
    parser.add_argument("--run-name", help="Custom run name")
    args = parser.parse_args()

    llm_cfg_path = resolve_path(args.config, "configs/llm")
    prompt_cfg_path = resolve_path(args.prompt_config, "configs/prompts")
    llm_cfg = load_yaml(llm_cfg_path)
    prompt_cfg = load_yaml(prompt_cfg_path)

    task_name = prompt_cfg.get("task_name", "rag_demo")
    if args.run_name:
        run_name = args.run_name
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_name = f"{task_name}_{timestamp}"

    env_name = llm_cfg["paths"]["api_key_env"]
    api_key = os.getenv(env_name) or load_api_key_from_dotenv(env_name)
    if not api_key:
        print(
            f"ERROR: {env_name} environment variable not set.\n"
            f"Tip: set {env_name} in your shell or add it to a local .env file.\n"
            "Get free key at: https://ai.google.dev/"
        )
        sys.exit(1)

    docs_dir = resolve_docs_dir(args.docs_dir, fallback_parent="reports")
    if not docs_dir.exists():
        print(f"ERROR: docs directory not found: {docs_dir}")
        sys.exit(1)

    source_docs = load_markdown_documents(docs_dir)
    if not source_docs:
        print(f"ERROR: no markdown files found in {docs_dir}")
        sys.exit(1)

    gen_params = llm_cfg.get("generation_params", {})
    if "generation_override" in prompt_cfg:
        gen_params.update(prompt_cfg["generation_override"])

    auto_route_enabled = not args.disable_auto_route
    route_info = {
        "budget_mode": "balanced",
        "reason": "auto-route disabled",
    }
    if auto_route_enabled:
        route_info = infer_rag_budget_mode(args.query)

    if args.budget_mode == "auto":
        resolved_budget_mode = route_info["budget_mode"]
    else:
        resolved_budget_mode = args.budget_mode

    effective = get_effective_rag_settings(
        budget_mode=resolved_budget_mode,
        cli_top_k=args.top_k,
        cli_chunk_size=args.chunk_size,
        cli_chunk_overlap=args.chunk_overlap,
        configured_max_tokens=gen_params.get("max_tokens"),
    )

    chunks = chunk_documents(
        source_docs,
        chunk_size=int(effective["chunk_size"]),
        chunk_overlap=int(effective["chunk_overlap"]),
    )

    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = int(effective["top_k"])
    retrieved_docs = retriever.invoke(args.query)

    context_sections: list[str] = []
    retrieved_manifest: list[dict] = []
    for idx, doc in enumerate(retrieved_docs, start=1):
        source = doc.metadata.get("source", "unknown")
        snippet = doc.page_content.strip()
        context_sections.append(f"[Source {idx}: {source}]\n{snippet}")
        retrieved_manifest.append(
            {
                "rank": idx,
                "source": source,
                "content_preview": snippet[:300],
                "content_length": len(snippet),
            }
        )

    context_text = "\n\n".join(context_sections)

    system_prompt = prompt_cfg.get("system_prompt", "")
    user_prompt_template = prompt_cfg.get(
        "user_prompt_template",
        # rag_guide.md
        # How does this repo explain RAG?
        "Context:\n{context}\n\nQuestion:\n{question}",
    )
    user_prompt = user_prompt_template.format(context=context_text, question=args.query)

    llm = ChatGoogleGenerativeAI(
        model=llm_cfg.get("model_name", "gemini-2.5-flash"),
        google_api_key=api_key,
        temperature=gen_params.get("temperature", 0.2),
        max_output_tokens=int(effective["max_tokens"]),
    )

    print(f"[{run_name}] Running RAG with Gemini...")
    print(f"  Model: {llm_cfg.get('model_name')}")
    print(f"  Docs dir: {docs_dir}")
    print(f"  Budget mode: {resolved_budget_mode}")
    print(f"  Auto route: {auto_route_enabled}")
    print(f"  Routing reason: {route_info['reason']}")
    print(
        "  Effective settings: "
        f"top_k={effective['top_k']}, "
        f"chunk_size={effective['chunk_size']}, "
        f"chunk_overlap={effective['chunk_overlap']}, "
        f"max_tokens={effective['max_tokens']}"
    )
    print(f"  Retrieved chunks: {len(retrieved_docs)}")
    print()

    response = llm.invoke(f"{system_prompt}\n\n{user_prompt}".strip())
    response_text = response.content if isinstance(response.content, str) else str(response.content)

    usage = getattr(response, "usage_metadata", {}) or {}
    prompt_tokens = usage.get("input_tokens")
    response_tokens = usage.get("output_tokens")
    total_tokens = usage.get("total_tokens")

    run_dir = ensure_dir(Path(llm_cfg["paths"]["runs_dir"]) / run_name)
    artifacts_cfg = llm_cfg["artifacts"]

    response_file = artifacts_cfg.get("response_file", "response.txt")
    metadata_file = artifacts_cfg.get("metadata_file", "metadata.json")
    params_file = artifacts_cfg.get("params_file", "params.json")
    bundle_info_file = artifacts_cfg.get("bundle_info_file", "bundle_info.json")
    retrieved_file = "retrieved_chunks.json"

    (run_dir / response_file).write_text(response_text, encoding="utf-8")
    (run_dir / retrieved_file).write_text(json.dumps(retrieved_manifest, indent=2), encoding="utf-8")

    metadata = {
        "model": llm_cfg.get("model_name"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "total_tokens": total_tokens,
        "task": task_name,
        "query": args.query,
        "docs_dir": str(docs_dir).replace("\\", "/"),
        "budget_mode": resolved_budget_mode,
        "auto_route": auto_route_enabled,
        "routing_reason": route_info["reason"],
        "top_k": effective["top_k"],
        "chunk_size": effective["chunk_size"],
        "chunk_overlap": effective["chunk_overlap"],
        "max_tokens": effective["max_tokens"],
        "retrieved_count": len(retrieved_docs),
    }
    (run_dir / metadata_file).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    params = {
        "llm_config": str(llm_cfg_path).replace("\\", "/"),
        "prompt_config": str(prompt_cfg_path).replace("\\", "/"),
        "model_name": llm_cfg.get("model_name"),
        "temperature": gen_params.get("temperature", 0.2),
        "budget_mode": resolved_budget_mode,
        "auto_route": auto_route_enabled,
        "routing_reason": route_info["reason"],
        "max_tokens": effective["max_tokens"],
        "docs_dir": str(docs_dir).replace("\\", "/"),
        "chunk_size": effective["chunk_size"],
        "chunk_overlap": effective["chunk_overlap"],
        "top_k": effective["top_k"],
    }
    (run_dir / params_file).write_text(json.dumps(params, indent=2), encoding="utf-8")

    bundle_info = {
        "run_name": run_name,
        "task_name": task_name,
        "model": llm_cfg.get("model_name"),
        "prompt_config": str(prompt_cfg_path).replace("\\", "/"),
        "llm_config": str(llm_cfg_path).replace("\\", "/"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": {
            "response": response_file,
            "metadata": metadata_file,
            "params": params_file,
            "retrieved_chunks": retrieved_file,
        },
    }
    (run_dir / bundle_info_file).write_text(json.dumps(bundle_info, indent=2), encoding="utf-8")

    summary_path = Path(llm_cfg["paths"]["runs_dir"]) / "summary.csv"
    summary_row = {
        "run_name": run_name,
        "task": task_name,
        "model": llm_cfg.get("model_name"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latency_ms": None,
        "status": "success" if "Error" not in response_text else "error",
    }
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        df = pd.DataFrame([summary_row])
    df.to_csv(summary_path, index=False)

    print("=" * 70)
    print("RAG Response:")
    print("=" * 70)
    print(response_text)
    print("=" * 70)
    print(f"\nArtifacts saved to: {run_dir}")
    print(f"  - {response_file}")
    print(f"  - {metadata_file}")
    print(f"  - {params_file}")
    print(f"  - {bundle_info_file}")
    print(f"  - {retrieved_file}")


if __name__ == "__main__":
    main()
