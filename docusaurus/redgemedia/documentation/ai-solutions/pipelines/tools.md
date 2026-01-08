---
sidebar_position: 10
title: Pipeline Tools (Helpers)
---

This page documents helper building blocks located under `chat-ai-deployment/pipelines/libs/*` used to compose production pipelines.

## Overview
- Retrievers: Provide context for retrieval (by name, from messages, etc.).
- Postprocessors: Filter, rank, and enrich retrieved nodes (e.g., cutoff, citations).
- Observability: Timing and event containers.
- Repositories: Thread-safe object cache (local and global singleton).
- Utilities: Query expansion and general-purpose helpers.

## Retrievers

### NameRetriever
- Purpose: Thin wrapper that assigns a human-readable name to a retriever (useful for logging/debugging).
- Init:
  - NameRetriever(name: str, retriever: BaseRetriever)
- Behavior:
  - Retrieve delegates to the underlying retriever.

### MultiCollectionRetriever
- Purpose: Runs multiple `NameRetriever`s, concatenates results, and then applies node postprocessors.
- Init:
  - retrievers: List[NameRetriever]
  - node_postprocessors: Optional[List[BaseNodePostprocessor]]
  - observer: Optional[Observer]
- Behavior:
  - For each underlying retriever: retrieve(query_bundle) and collect results.
  - For each postprocessor: `postprocess_nodes(results)` is called in order.
- Typical usage (from prod pipelines):
  - Build one retriever per Qdrant collection, wrap with `NameRetriever`, then pass the list here.


### MessagesRetriever
- Purpose: Exposes previous assistant messages as retrieval context (e.g., include the last answer).
- Init:
  - MessagesRetriever(messages: List[dict]) where each message has at least {"role", "content"}.
- Behavior:
  - Emits a high-score Node for each assistant message after cleaning via `normalize_message()`.

## Postprocessors

### SimilarityCutoffPostprocessor
- Purpose: Remove nodes below a similarity threshold and optionally limit the total number of chunks.
- Params:
  - similarity_cutoff: Optional[float]
  - max_chunks: Optional[int]
- Behavior:
  - Filter by score >= similarity_cutoff.
  - If max_chunks is set, keep top-N by score.

### BuildReference
- Purpose: Add a human-readable citation to node metadata under a chosen key (e.g., "reference").
- Init:
  - BuildReference(reference_key: str)
- Behavior:
  - Inspects node metadata and creates references for:
    - PDF (file_path, section, sentence)
    - Web pages (title/url)
    - Docusaurus docs (maps file_path to docs URL)
    - Plain text files (file_path)
    - Websearch results (title/source stored under metadata.metadata)
- Notes:
  - `class_name()` returns "Citations" for UI friendliness.
  - Typically used as the last step before presenting nodes to the model so citations are available.

## Observability

### Observer
- Purpose: Container for timing information across named sections.
- Init:
  - Observer()
- Methods:
  - start(key: str, name: str) -> str  // returns unique key
  - stop(key: str) -> None
  - summary() -> str  // string representation of all timers
  - clear() -> None

### Event
- Purpose: Lightweight container for timing/event information with structured data.
- Init:
  - Event(name: str, **data)  // init data as kwargs
- Behavior:
  - __str__() -> str  // string representation of the event

## Repositories

### ObjectRepository
- Purpose: Thread-safe registry to cache expensive/shared objects (e.g., rerankers, sparse encoders) across requests.
- Methods:
  - get_or_create(name: str, factory: Callable, **kwargs) -> Any
    - Creates the object with factory(**kwargs) if not present or if kwargs hash changed; otherwise returns cached.
  - delete(name: str) -> None
- Usage:
  - Create a repository instance in your pipeline.
  - Use get_or_create to cache expensive objects.
  - Use delete to remove objects from the cache.
- Notes:
  - Thread-safe.
  - Not shared across requests.

### GlobalRepository
- Purpose: Singleton instance of ObjectRepository.
- Usage:
  - GlobalRepository.get_or_create(name: str, factory: Callable, **kwargs) -> Any
  - GlobalRepository.delete(name: str) -> None

## Query Expansion

### ExpandQueries
- Purpose: Expand the user query into diverse, validated variations to improve retrieval recall across collections.
- Init:
  - ExpandQueries(
      llm: Any,
      num_expansions: int = 3,
      expansion_strategy: str = "diversify",
      min_query_length: int = 30,
      max_query_length: int = 200,
      similarity_threshold: float = 0.8,
      fallback_strategies: List[str] = ["synonyms", "keywords"]
    )
- Behavior:
  - Builds an LLM prompt based on selected strategy and asks for `num_expansions` follow-up questions.
  - Parses LLM output robustly.
  - Validates each candidate (min/max length, must contain letters, at least 2 words).
  - Deduplicates via token overlap against the original query and among candidates (controlled by `similarity_threshold`).
  - Applies fallbacks (`synonyms`/`keywords`) if the LLM response is empty/invalid or yields too few items; also used on errors.
  - Returns List[QueryBundle] (LlamaIndex) with at most `num_expansions` items.
- Typical use:
  - Pass an instance to `MultiCollectionRetriever` so each expanded query is retrieved alongside the original and results are merged.

#### Strategies:
  - `diversify`: Generate varied, related technical questions that span architectures, protocols, performance, edge cases, and interoperability to maximize semantic coverage of the topic.
    - Use when: The initial query is broad/ambiguous or early retrieval looks too narrow/biased and you want to surface diverse chunks from different parts of the corpus.

  - `specify`: Create detailed follow-up questions that drill into concrete configs, flags, parameters, API calls, and hardware-acceleration behaviors for high-precision retrieval.
    - Use when: You need exact, implementation-level chunks (e.g., FFmpeg/CUDA flags, CDN settings) or over-retrieval is noisy and you want sharper precision.

  - `broaden`: Form higher-level questions that connect to adjacent domains, related standards, scalability, compatibility, and industry trends to expand contextual recall.
    - Use when: Relevant knowledge might live in neighboring topics or cross-cutting docs (e.g., new codecs/protocols) or when initial searches return few/no hits.

  - `rephrase`: Restate the same question using alternate terminology, synonyms, and documentation-style wording to match varied authoring styles.
    - Use when: There’s a vocabulary mismatch between the query and stored documents (e.g., “NVENC” vs “NVIDIA hardware encoder”) or to improve lexical/semantic match without changing intent.

  - `diagnose`: Pose symptom-driven, root-cause-focused questions aimed at misconfigurations, bottlenecks, and compatibility issues to retrieve troubleshooting content.
    - Use when: You’re investigating failures or performance regressions and want logs, metrics, playbooks, and known-issue chunks prioritized.
    These map to prompt templates in `pipelines/libs/template.py` under `QUERIES_TEMPLATES` and tailor how questions are generated.

#### Example (pseudocode):
  ```text
  llm = Ollama(...)
  expand = ExpandQueries(llm=llm, num_expansions=3, expansion_strategy="diversify")
  mcr = MultiCollectionRetriever(retrievers=[...], expand_queries=expand, observer=Observer())
  nodes = mcr.retrieve(QueryBundle(query_str="..."))
  ```

## Utilities

- setup_logger: Sets up a logger with a default handler that writes to stdout.
- wrap_text_in_box: Wraps text in a markdown box.
- metadata_to_string: Converts metadata to a string.
- normalize_message: Removes references/times sections and converts markdown to HTML.
- parse_nodes_to_markdown: Renders a diagnostic view of retrieved nodes and timings.
- parse_code_nodes_to_markdown: Like above but with file/position info for code.
- strip_whitespace: Compact helper used to normalize LLM outputs (e.g., in detectors).

## Notes
- Hybrid search: Enabled by providing a sparse encoder (e.g., `fastembed_sparse_encoder`) to Qdrant vector store and setting `enable_hybrid=True`.
- Postprocessor order matters: Run `SimilarityCutoffPostprocessor` before rerankers if you want to reduce work.
- `BuildReference` should typically be the last step before presenting nodes to the model so citations are available.
- `Observer.summary()` is designed to be rendered at the end of a markdown response; for API use you may collect timings separately.
- All retrievers return a list of `NodeWithScore` objects (LlamaIndex type).
