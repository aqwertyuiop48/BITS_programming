# RAG Guide for This Repo

This guide explains how Retrieval-Augmented Generation (RAG) is implemented in this project and how to run it.

## What RAG Means Here

RAG combines two steps:
1. Retrieve relevant text chunks from local documents.
2. Generate an answer with Gemini using the retrieved context.

In this repo, the default knowledge base is markdown files under reports.

## Files Involved

- rag_demo.py: main RAG script
- configs/llm/gemini.yaml: Gemini model and generation settings
- configs/prompts/rag_grounded_answer.yaml: prompt template for grounded answers (if present)
- runs/<run_name>/: run artifacts

## RAG Pipeline in This Project

1. Load LLM and prompt configs.
2. Load API key from environment variable or .env.
3. Read markdown files from docs directory (default: reports).
4. Chunk documents for retrieval.
5. Build BM25 retriever.
6. Retrieve top-k chunks for the user query.
7. Inject retrieved context into prompt.
8. Call Gemini and save artifacts.

## Run Command

Use the virtual environment Python:

python rag_demo.py --config configs/llm/gemini.yaml --prompt-config configs/prompts/rag_grounded_answer.yaml --query "How does this repo explain RAG?"

You can also use shorthand names:

python rag_demo.py --config gemini.yaml --prompt-config rag_grounded_answer.yaml --query "How does this repo explain RAG?"

Optional flags:

- --docs-dir reports
- --top-k 4
- --chunk-size 1200
- --chunk-overlap 150
- --run-name rag_demo_custom

## Artifacts Produced

Each run stores:

- response.txt: final answer
- metadata.json: model and token metadata
- params.json: parameters used
- bundle_info.json: artifact manifest
- retrieved_chunks.json: retrieved context snippets and sources

A summary row is also appended to runs/summary.csv.

## Tuning Tips

- Increase top-k to include more evidence when answers are incomplete.
- Reduce chunk size if retrieval returns broad, noisy context.
- Increase chunk overlap for better continuity across split sections.
- Keep prompt instructions explicit about grounding and source use.

## Common Failure Modes

1. Empty retrieval:
- Cause: docs directory has no markdown or query mismatch.
- Fix: verify docs directory and query phrasing.

2. Generic answers:
- Cause: weak grounding instructions.
- Fix: strengthen prompt to require context-based answers.

3. Missing API key:
- Cause: GEMINI_API_KEY not set.
- Fix: set environment variable or add GEMINI_API_KEY in local .env.

## Suggested Evaluation Checklist

- Does the answer reflect retrieved context?
- Are key claims supported by retrieved sources?
- Does changing top-k materially improve quality?
- Are responses consistent across repeated runs?
