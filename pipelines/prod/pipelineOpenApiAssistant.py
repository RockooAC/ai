"""
title: OpenAPI Ollama Pipeline
description: A pipeline for retrieving relevant information from the qdrant collection with the OpenAPI entries using the Llama Index library with Ollama embeddings.
"""
from os import getenv
from typing import List, Optional, Union, Generator, Iterator

from qdrant_client import QdrantClient
from llama_index.core.base.response.schema import StreamingResponse
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.base.response.schema import Response
from llama_index.vector_stores.qdrant.utils import fastembed_sparse_encoder

from libs.tools import Observer, setup_logger
from libs.openapi_engine import OpenAPIPromptGenerator
from libs.variables import GlobalRepository, DEFAULT_REQUEST_TIMEOUT

# Logger Configuration
logger = setup_logger(name="OpenAPI Assistant", debug=False)
ENV_SUFFIX = "_OPENAPI_ASSISTANT"

class Pipeline:
    from libs.valves import Valves

    def __init__(self):
        self.name = "OpenAPI Assistant"
        logger.info(f"Initializing {self.name}")
        self.qdrant = None
        self.ollama_llm = None
        self.ollama_embedder = None
        self.text_splitter = None
        self.retrievers = []
        self.llm_rerank = None
        self.openapi_prompt_generator = None
        self.valves = self.Valves(
            **{k: getenv(f"{k}{ENV_SUFFIX}", v.default) for k, v in self.Valves.model_fields.items()}
        )

    def load_config(self):
        """
        Load the configuration. This function is called when the server is started and when the valves are updated.
        It sets various settings for the Ollama model and Qdrant client, and initializes the vector store index.
        """

        logger.info("Loading configuration...")
        try:
            # Initialize the text splitter with chunk size and overlap
            self.text_splitter = SentenceSplitter(
                chunk_size=self.valves.OLLAMA_CHUNK_SIZE,
                chunk_overlap=self.valves.OLLAMA_CHUNK_OVERLAP,
            )

            # Initialize the embedding model with the specified model name and base URL
            self.ollama_embedder = OllamaEmbedding(
                model_name=self.valves.OLLAMA_EMBEDDING_MODEL_NAME,
                base_url=self.valves.OLLAMA_EMBEDDING_BASE_URL,
            )

            # Set the LLM settings with model name, base URL, temperature, and request timeout
            self.ollama_llm = Ollama(
                model=self.valves.OLLAMA_MODEL_NAME,
                base_url=self.valves.OLLAMA_MODEL_BASE_URL,
                temperature=self.valves.OLLAMA_TEMPERATURE,
                context_window=self.valves.OLLAMA_CONTEXT_WINDOW,
                request_timeout=DEFAULT_REQUEST_TIMEOUT,
            )

            # Connect to the Qdrant client using the base URL
            self.qdrant = QdrantClient(url=self.valves.QDRANT_BASE_URL)

            self.sparse_text_embedding = None
            if self.valves.QDRANT_HYBRID_SEARCH:
                self.sparse_text_embedding = GlobalRepository.get_or_create(
                    name="fastembed_sparse_encoder",
                    factory=fastembed_sparse_encoder,
                    model_name=self.valves.SPARSE_TEXT_EMBEDDING_MODEL,
                )

            self.openapi_prompt_generator = OpenAPIPromptGenerator(
                self.ollama_embedder,
                self.qdrant,
                self.valves.QDRANT_OPENAPI_COLLECTION_NAME,
                self.valves.QDRANT_VECTOR_STORE_PARALLEL,
                self.valves.QDRANT_SIMILARITY_TOP_K,
                self.sparse_text_embedding,
                logger,
            )

            logger.info("Loading configuration success")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")

    async def on_startup(self):
        """
        This function is called when the server is started.
        It loads the configuration for the OpenAPI Assistant Pipeline.
        """
        logger.info(f"Loading {self.name} Pipeline")
        self.load_config()

    async def on_valves_updated(self):
        """
        This function is called when the valves are updated.
        It reloads the configuration and closes the existing Qdrant client connection.
        """
        logger.info("Valves updated. Reloading configuration.")
        self.load_config()

    async def on_shutdown(self):
        """
        This function is called when the server is stopped.
        Currently, it does not perform any actions.
        """
        logger.info(f"Shutting down {self.name} Pipeline")
        self.qdrant.close()

    async def inlet(self, body: dict, user: dict) -> dict:
        """Modifies form data before the OpenAI API request."""
        logger.info("Processing inlet request")
        if 'files' in body:
            for file in body['files']:
                if 'collection_name' in file:
                    if 'external_collections' not in body:
                        body['external_collections'] = []
                    body['external_collections'].append(f"open-webui_{file['collection_name']}")
        elif 'metadata' in body and body['metadata'].get('files', None) is not None:
            for file in body['metadata']['files']:
                if 'collection_name' in file:
                    if 'external_collections' not in body:
                        body['external_collections'] = []
                    body['external_collections'].append(f"open-webui_{file['collection_name']}")

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Modifies the response body before sending it to the user."""
        # Check if the last message in body is an 'Empty Response' and replace it with a 'No Results Found' message
        if body.get("messages"):
            # "Empty Response"
            content = body["messages"][-1].get("content", "")
            if content.startswith("Empty Response"):
                content = content.replace("Empty Response", "Sorry, I couldn't find any relevant information. Please try asking in a different way.")
                body["messages"][-1]["content"] = content
        return body

    @staticmethod
    def _prepare_references(source_nodes: list[NodeWithScore]) -> str:
        """Prepare the references for the response."""
        references_info = f"\n\n***References:***\n"
        # Add unique references to the response
        references = set()
        # Sort nodes by score in place
        source_nodes.sort(key=lambda x: x.score, reverse=True)
        for node in source_nodes:
            if 'reference' in node.metadata and node.metadata['reference'] not in references:
                references.add(node.metadata['reference'])
                references_info += f"\n- {node.metadata['reference']}  (Score: {node.score:.2f})"
        return references_info

    def _extend_stream(self, response, event_key: str, observer: Observer) -> Generator:
        """
        Extend the response stream by yielding chunks from the response generator,
        stopping the observer, and appending references and summary.

        Args:
            response: The response object containing the response generator.
            event_key (str): The event key for the observer.
            observer (Observer): The observer instance to track the processing.

        Yields:
            Generator: Chunks of the response, followed by references and summary.
        """
        generator = response.response_gen() if callable(response.response_gen) else response.response_gen
        for chunk in generator:
            yield chunk
        observer.stop(key=event_key)
        logger.info(f"Finished processing response with event key: {event_key}")
        yield self._prepare_references(response.source_nodes)
        yield observer.summary()

    def _process_response(self, response, event_key, observer) -> Union[str, Generator, Iterator]:
        """
        Process the response from the query engine.

        Args:
            response: The response object from the query engine.
            event_key: The event key for the observer.
            observer: The observer instance to track the processing.

        Returns:
            Union[str, Generator, Iterator]: The processed response, which can be a string, generator, or iterator.
        """
        if isinstance(response, StreamingResponse):
            return self._extend_stream(response, event_key, observer)
        else:
            content = response.response
            observer.stop(key=event_key)
            #content += self._prepare_references(response.source_nodes)
            content += observer.summary()
            logger.info(f"Finished processing response with event key: {event_key}")
            return content

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) \
            -> Union[str, Generator, Iterator]:
        """
        Process the user message through the pipeline.

        Args:
            user_message (str): The message from the user.
            model_id (str): The ID of the model to use.
            messages (List[dict]): A list of message dictionaries.
            body (dict): Additional data for processing.

        Returns:
            Union[str, Generator, Iterator]: The response from the query engine, which can be a string, generator, or iterator.
        """
        logger.info(f"Processing message: '{user_message}'")
        observer = Observer()
        event_key = observer.start(name="LLM Query Engine")
        logger.info(f"Send message: '{user_message}' to query engine with event key: {event_key}")
        generated_prompt = self.openapi_prompt_generator.generate_prompt(user_message, False)
        filled_prompt = generated_prompt.format(query_str=user_message)
        response = Response(response=str(self.ollama_llm.complete(filled_prompt)))

        return self._process_response(response, event_key, observer)