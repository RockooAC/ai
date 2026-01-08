"""
title: Code Assistant Pipeline
author: Karol Siegieda
date: 2025-02-11
version: 1.0
license: MIT
description: A pipeline for analyzing source code using LlamaIndex and Deepseek Coder.
"""

import asyncio
import os
import time
import hashlib
import re

from typing import List, Union, Generator, Iterator, Optional, cast, Dict, Any

import qdrant_client
from llama_index.core import VectorStoreIndex, StorageContext, Settings, PromptTemplate
from llama_index.core.callbacks import CallbackManager
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.indices.list.base import ListRetrieverMode
from llama_index.core.prompts import PromptType
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.bridge.pydantic import Field, BaseModel
from llama_index.core.schema import NodeWithScore, QueryBundle, QueryType
from llama_index.llms.ollama import Ollama
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate

from typing import List, Union, Generator, Iterator, Optional
from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama

CODE_QA_PROMPT_STR = (
    "You are a Redge Code Assistant, specialized in understanding and explaining code across different programming languages (mainly C++ and Python).\n"
    "A specific code-related query has been provided, along with relevant code snippets and their metadata.\n\n"
    "Your Task:\n"
    "1. Analyze the provided code snippets\n"
    "2. Provide **concise** explanations of:\n"
    "   - Core functionality\n"
    "   - Key implementation details\n"
    "   - Important patterns or techniques used\n"
    "3. When dealing with C++ code:\n"
    "   - Suggest modern C++20 features where relevant\n"
    "   - Keep suggestions **practical and maintainable**\n\n"
    "Guidelines:\n"
    "- Be brief but precise\n"
    "- Reference key parts of the code only\n"
    "- Keep explanations under 3-5 sentences\n"
    "- If improvements are needed, suggest only the necessary changes**\n"
    "- If a query cannot be fully answered, explain why in a short sentence\n"
    "- DO NOT provide full original code if it's too long—only show essential refactored or optimized sections with explanations.\n\n"
    "Context information is below:\n"
    "{context_str}\n"
    "Question: {query_str}\n\n"
    "Answer:"
)


DEFAULT_CHOICE_SELECT_PROMPT = PromptTemplate(CODE_QA_PROMPT_STR, prompt_type=PromptType.CHOICE_SELECT)


def parse_nodes_to_markdown(user_message: str, nodes: list, retrieval_time: float) -> str:
    output = f"**Code Analysis Query**: _{user_message}_\n\n"
    output += f"**Total Relevant Snippets**: {len(nodes)}\n"
    output += f"**Retrieval Time**: {retrieval_time:.2f} seconds\n\n"

    # Sort nodes by relevance score
    sorted_nodes = sorted(nodes, key=lambda x: x.score or 0, reverse=True)

    for node in sorted_nodes:
        # More detailed metadata display
        metadata_details = [
            f"**Relevance**: {node.score:.2f}%",
            f"**File**: `{node.metadata.get('file_path', 'N/A')}`",
            f"**Method**: `{node.metadata.get('method_name', 'N/A')}`",
            f"**Class**: `{node.metadata.get('class_name', 'N/A')}`",
        ]

        output += "\n".join(metadata_details) + "\n\n"
        output += "**Code**:\n```cpp\n" + node.text + "\n```\n\n"

    return output


class SamplePostprocessor(BaseNodePostprocessor):
    """Similarity-based Node processor."""

    similarity_cutoff: float = Field(default=None)

    @classmethod
    def class_name(cls) -> str:
        return "SimilarityPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        sim_cutoff_exists = self.similarity_cutoff is not None
        new_nodes = []
        for node in nodes:
            should_use_node = True
            if sim_cutoff_exists:
                similarity = node.score
                if similarity is None:
                    should_use_node = False
                elif cast(float, similarity) < cast(float, self.similarity_cutoff):
                    should_use_node = False
            if should_use_node:
                new_nodes.append(node)
        return new_nodes


class MultiCollectionRetriever(BaseRetriever):
    """A retriever that combines multiple retrievers."""

    retrievers: List[BaseRetriever] = Field(description="A list of retrievers to combine.")
    node_postprocessors: Optional[List[BaseNodePostprocessor]] = Field(description="A list of node postprocessors.")

    def __init__(
        self,
        retrievers: List[BaseRetriever],
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        callback_manager: Optional[CallbackManager] = None,
        object_map: Optional[dict] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        self.retrievers = retrievers or []
        self.node_postprocessors = node_postprocessors or []
        super().__init__(callback_manager=callback_manager, object_map=object_map, verbose=verbose, **kwargs)

    def retrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
        return super().retrieve(str_or_query_bundle)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes with more robust error handling."""
        results = []
        try:
            for retriever in self.retrievers:
                try:
                    retriever_results = retriever.retrieve(query_bundle)
                    results.extend(retriever_results)
                except Exception as retriever_error:
                    print(f"Error in retriever: {retriever_error}")
                    continue

            for postprocessor in self.node_postprocessors:
                try:
                    results = postprocessor.postprocess_nodes(results, query_bundle=query_bundle)
                except Exception as postprocessor_error:
                    print(f"Error in postprocessor: {postprocessor_error}")
                    continue

            return results
        except Exception as e:
            print(f"Critical error in retrieval: {e}")
            return []


class Pipeline:
    class Valves(BaseModel):
        OLLAMA_MODEL_BASE_URL: str
        OLLAMA_MODEL_NAME: str
        OLLAMA_CONTEXT_WINDOW: int
        OLLAMA_REQUEST_TIMEOUT: float
        OLLAMA_TEMPERATURE: float
        OLLAMA_SIMILARITY_TOP_K: int
        OLLAMA_EMBEDDING_BASE_URL: str
        OLLAMA_EMBEDDING_MODEL_NAME: str
        OLLAMA_CHUNK_SIZE: int
        OLLAMA_CHUNK_OVERLAP: int
        OLLAMA_RERANK_BASE_URL: str
        OLLAMA_RERANK_MODEL_NAME: str
        OLLAMA_RERANK_TEMPERATURE: float
        OLLAMA_RERANK_TOP_N: int
        OLLAMA_RERANK_CHOICE_BATCH_SIZE: int
        QDRANT_SIMILARITY_TOP_K: int
        QDRANT_BASE_URL: str
        QDRANT_COLLECTION_NAME: str
        QDRANT_VECTOR_STORE_PARALLEL: int
        QDRANT_HYBRID_SEARCH: bool
        QDRANT_SIMILARITY_CUTOFF: float

        def __init__(self, **data):
            print(f"Initializing Code Assistant Valves with data: {data}")
            super().__init__(**data)

    def __init__(self):
        print("Initializing Code Assistant")
        self.name = "Code Assistant Fast (unsafe)"
        self.index = None
        self.qdrant = None
        self.ollama_rerank = None
        self.query_engine = None
        self.llm = None
        self.llm_rerank = None
        self.retrievers = []
        self.query_engine_postprocessors = []

        self.valves = self.Valves(
            **{
                "OLLAMA_MODEL_BASE_URL": os.getenv("OLLAMA_MODEL_BASE_URL", "http://10.255.240.156:11434"),
                "OLLAMA_MODEL_NAME": os.getenv("OLLAMA_MODEL_NAME", "deepseek-coder-v2-fixed:latest"),
                "OLLAMA_CONTEXT_WINDOW": os.getenv("OLLAMA_CONTEXT_WINDOW", "96000"),
                "OLLAMA_REQUEST_TIMEOUT": os.getenv("OLLAMA_REQUEST_TIMEOUT", "120"),
                "OLLAMA_TEMPERATURE": os.getenv("OLLAMA_TEMPERATURE", "0.3"),
                "OLLAMA_SIMILARITY_TOP_K": os.getenv("OLLAMA_SIMILARITY_TOP_K", "5"),
                "OLLAMA_EMBEDDING_BASE_URL": os.getenv("OLLAMA_EMBEDDING_BASE_URL", "http://10.255.240.161:11434"),
                "OLLAMA_EMBEDDING_MODEL_NAME": os.getenv(
                    "OLLAMA_EMBEDDING_MODEL_NAME", "jina/jina-embeddings-v2-base-en:latest"
                ),
                "OLLAMA_CHUNK_SIZE": os.getenv("OLLAMA_CHUNK_SIZE", "1024"),
                "OLLAMA_CHUNK_OVERLAP": os.getenv("OLLAMA_CHUNK_OVERLAP", "128"),
                "OLLAMA_REQUEST_TIMEOUT": os.getenv("OLLAMA_REQUEST_TIMEOUT", "120"),
                "OLLAMA_RERANK_BASE_URL": os.getenv("OLLAMA_RERANK_BASE_URL", "http://10.255.240.161:11434"),
                "OLLAMA_RERANK_MODEL_NAME": os.getenv("OLLAMA_RERANK_MODEL_NAME", "llama3.2:3b"),
                "OLLAMA_RERANK_TEMPERATURE": os.getenv("OLLAMA_RERANK_TEMPERATURE", "0.5"),
                "OLLAMA_RERANK_TOP_N": os.getenv("OLLAMA_RERANK_TOP_N", "4"),
                "OLLAMA_RERANK_CHOICE_BATCH_SIZE": os.getenv(
                    "OLLAMA_RERANK_CHOICE_BATCH_SIZE", "3"
                ),  # MORE CAN CAUSE OOM on OLLAMA
                "QDRANT_SIMILARITY_TOP_K": os.getenv("QDRANT_SIMILARITY_TOP_K", "8"),
                "QDRANT_BASE_URL": os.getenv("QDRANT_BASE_URL", "http://10.255.240.18:6333"),
                "QDRANT_COLLECTION_NAME": os.getenv("QDRANT_COLLECTION_NAME", "jina-code-embed2"),
                "QDRANT_VECTOR_STORE_PARALLEL": os.getenv("QDRANT_VECTOR_STORE_PARALLEL", "4"),
                "QDRANT_HYBRID_SEARCH": os.getenv("QDRANT_HYBRID_SEARCH", "False").lower() == "true",
                "QDRANT_SIMILARITY_CUTOFF": os.getenv("QDRANT_SIMILARITY_CUTOFF", "0.5"),
            }
        )

    def load_config(self):
        """Load configuration and initialize components."""
        print("Loading Code Retrieval configuration")

        # Configure LlamaIndex settings
        Settings.chunk_size = self.valves.OLLAMA_CHUNK_SIZE
        Settings.chunk_overlap = self.valves.OLLAMA_CHUNK_OVERLAP
        Settings.text_splitter = SentenceSplitter(
            chunk_size=self.valves.OLLAMA_CHUNK_SIZE,
            chunk_overlap=int(self.valves.OLLAMA_CHUNK_SIZE * 0.1),  # 20% overlap
        )

        Settings.context_window = self.valves.OLLAMA_CONTEXT_WINDOW

        # Initialize embedding model
        Settings.embed_model = OllamaEmbedding(
            model_name=self.valves.OLLAMA_EMBEDDING_MODEL_NAME,
            base_url=self.valves.OLLAMA_EMBEDDING_BASE_URL,
        )

        # Initialize reranking model
        self.ollama_rerank = Ollama(
            base_url=self.valves.OLLAMA_RERANK_BASE_URL,
            model=self.valves.OLLAMA_RERANK_MODEL_NAME,
            temperature=self.valves.OLLAMA_RERANK_TEMPERATURE,
            context_window=self.valves.OLLAMA_RERANK_CHOICE_BATCH_SIZE * self.valves.OLLAMA_CHUNK_SIZE + 500,
            request_timeout=self.valves.OLLAMA_REQUEST_TIMEOUT,
        )

        Settings.llm = Ollama(
            model=self.valves.OLLAMA_MODEL_NAME,
            base_url=self.valves.OLLAMA_MODEL_BASE_URL,
            temperature=self.valves.OLLAMA_TEMPERATURE,
            request_timeout=self.valves.OLLAMA_REQUEST_TIMEOUT,
            # context_window=Settings.context_window, # In fast mode we do not provide context_window (does not alloc > 40GiB in runtime)
        )

        # Initialize Qdrant client
        self.qdrant = qdrant_client.QdrantClient(url=self.valves.QDRANT_BASE_URL)

        query_engine_postprocessors = [
            SamplePostprocessor(similarity_cutoff=self.valves.QDRANT_SIMILARITY_CUTOFF),
            LLMRerank(
                llm=self.ollama_rerank,
                top_n=self.valves.OLLAMA_RERANK_TOP_N,
                choice_batch_size=self.valves.OLLAMA_RERANK_CHOICE_BATCH_SIZE,
                choice_select_prompt=DEFAULT_CHOICE_SELECT_PROMPT,
            ),
        ]

        retrievers = []
        collections = self.valves.QDRANT_COLLECTION_NAME.split(",")
        for collection_name in collections:
            collection_retriever = self._build_retriever(collection_name)
            if collection_retriever is not None:
                print(f"Loaded collection: {collection_name}")
                self.retrievers.append(collection_retriever)
        pass

    def _build_retriever(self, collection_name: str):
        """
        Build a retriever for the specified collection name.
        """
        if not self.qdrant.collection_exists(collection_name):
            print(f"Collection {collection_name} does not exist. Creating collection.")
            return None

        # Connect to the Qdrant vector store collection with parallelism and collection name
        vector_store = QdrantVectorStore(
            client=self.qdrant,
            parallel=self.valves.QDRANT_VECTOR_STORE_PARALLEL,
            collection_name=collection_name,
            enable_hybrid=self.valves.QDRANT_HYBRID_SEARCH,
        )

        # Create a storage context from the vector store
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
        )

        return VectorStoreIndex.from_vector_store(
            storage_context=storage_context,
            vector_store=vector_store,
        ).as_retriever(
            retriever_mode=ListRetrieverMode.DEFAULT,
            similarity_top_k=self.valves.OLLAMA_SIMILARITY_TOP_K,
        )

    def _extract_keywords(self, query: str) -> List[str]:
        """
        Dynamically extract keywords from the query with improved context extraction
        """
        query = query.lower()

        # Stop words that do not contribute meaningfully
        stop_words = {"show", "me", "find", "list", "all", "of", "the", "in"}

        # Tokenize while preserving useful symbols (e.g., file paths, underscores)
        tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b|\/[\w\/.]+", query)

        # Remove stop words
        tokens = [word for word in tokens if word not in stop_words]

        # Identify function-like patterns
        function_patterns = re.compile(
            r"\b(get|set|fetch|retrieve|process|compute|handle|convert|parse|validate|handle)_[a-zA-Z_]+\b"
        )
        potential_methods = [word for word in tokens if function_patterns.match(word)]

        # Identify class/file names based on capitalization or path structure
        potential_names = [word for word in tokens if word[0].isupper() or "/" in word]

        # Collect all relevant keywords
        context_keywords = list(dict.fromkeys(potential_methods + potential_names))

        return context_keywords

    def _filter_nodes(self, query: str, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """
        Efficiently filter nodes based on query relevance using metadata prioritization.
        """
        keywords = set(self._extract_keywords(query))  # Use set for faster lookups

        if not keywords:
            return nodes  # No filtering needed if no keywords

        filtered_nodes = []

        for node in nodes:
            node_score = 0

            # Convert metadata values to lowercase for case-insensitive matching
            metadata_values = {k: str(v).lower() for k, v in node.metadata.items() if v}

            # Strong matches (highest weight)
            if metadata_values.get("fully_qualified_name") and any(
                kw in metadata_values["fully_qualified_name"] for kw in keywords
            ):
                node_score += 3

            elif metadata_values.get("method_name") and any(kw in metadata_values["method_name"] for kw in keywords):
                node_score += 2

            elif metadata_values.get("class_name") and any(kw in metadata_values["class_name"] for kw in keywords):
                node_score += 2

            # Medium matches
            if any(kw in value for kw in keywords for value in metadata_values.values()):
                node_score += 1

            # Low-priority: Check in raw text content
            if any(kw in node.text.lower() for kw in keywords):
                node_score += 1

            if node_score > 0:
                filtered_nodes.append((node, node_score))

    def _targeted_retrieve(self, query: str):
        """
        Targeted retrieval with dynamic filtering
        """
        # Extract specific search terms
        keywords = self._extract_keywords(query)

        # Directly use existing retrievers
        nodes = []
        for retriever in self.multi_retriever.retrievers:
            try:
                # Directly retrieve without modifying the retriever
                retriever_nodes = retriever.retrieve(query)
                nodes.extend(retriever_nodes)
            except Exception as e:
                print(f"Error retrieving from retriever: {e}")

        # Comprehensive filtering based on keywords and metadata
        filtered_nodes = [
            node
            for node in nodes
            if (
                # Check metadata for keywords
                any(keyword.lower() in str(value).lower() for keyword in keywords for value in node.metadata.values())
                or
                # Check full text for keywords
                any(keyword.lower() in node.text.lower() for keyword in keywords)
                or
                # Check for method-related metadata with expanded code types
                any(
                    [
                        node.metadata.get("code_type") in ["method", "global_code", "function"],
                        node.metadata.get("class_name") is not None,
                        node.metadata.get("namespace") is not None,
                        node.metadata.get("fully_qualified_name") is not None,
                        node.metadata.get("method_name") is not None,
                    ]
                )
            )
        ]

        return filtered_nodes

    async def on_startup(self):
        """Initialize the pipeline on startup."""
        print("Loading Code Assistant Pipeline")
        self.load_config()

    async def on_valves_updated(self):
        """Handle configuration updates."""
        print("Valves updated")
        if self.qdrant:
            self.qdrant.close()
        self.load_config()

    async def on_shutdown(self):
        """Clean up resources on shutdown."""
        if self.qdrant:
            self.qdrant.close()

    async def inlet(self, body: dict, user: dict) -> dict:
        """Modifies form data before the OpenAI API request."""
        print("Processing inlet request")
        if "files" in body:
            for file in body["files"]:
                if "collection_name" in file:
                    if not "external_collections" in body:
                        body["external_collections"] = []
                    body["external_collections"].append(f"open-webui_{file['collection_name']}")
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        try:
            if not body or "messages" not in body or not body["messages"]:
                return {"messages": [{"content": "Invalid response format"}]}

            if body["messages"][-1]["content"] == "Empty Response":
                body["messages"][-1]["content"] = (
                    "Sorry, I couldn't find any relevant information. " "Please try asking in a different way."
                )
            return body
        except Exception as e:
            print(f"Error in outlet: {str(e)}")
            return {"messages": [{"content": "Error processing response"}]}

    def _validate_query(self, user_message: str) -> str:
        """Validate and clean user query"""
        if not user_message or not isinstance(user_message, str):
            raise ValueError("Invalid query format")
        # Remove any template/XML tags from the query
        clean_message = re.sub(r"<[^>]+>", "", user_message)
        return clean_message.strip()

    def _preprocess_query(self, user_message: str) -> str:
        """Preprocess query to fit context window"""
        target_size = self.valves.OLLAMA_CONTEXT_WINDOW - 1000
        if len(user_message) > target_size:
            return user_message[:target_size]
        return user_message

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        try:
            user_message = self._preprocess_query(user_message)
            retrievers = self.retrievers.copy()

            if "external_collections" in body:
                for external_collection in body["external_collections"]:
                    collection_retriever = self._build_retriever(external_collection)
                    if collection_retriever is not None:
                        retrievers.append(collection_retriever)

            if not retrievers:
                return "No retrievers available to process the query."

            query_engine = RetrieverQueryEngine.from_args(
                retriever=MultiCollectionRetriever(
                    retrievers=retrievers, node_postprocessors=self.query_engine_postprocessors
                ),
                llm=Settings.llm,
                text_qa_template=PromptTemplate(CODE_QA_PROMPT_STR),
                streaming=True,
            )

            response = query_engine.query(user_message)
            return response.response_gen if response.response_gen else response.response

        except RuntimeError as e:
            if "context does not support K-shift" in str(e):
                # Retry with smaller context
                return self._fallback_query(user_message)
        except Exception as e:
            print(f"Fatal error in pipe: {e}")
            return "Error processing request. Please try with a shorter query."
