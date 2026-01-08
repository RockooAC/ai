from libs.openapi_prompt_builder import prepare_openapi_prompt

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.bridge.pydantic import BaseModel, ConfigDict
from llama_index.core.indices.list.base import ListRetrieverMode
from llama_index.core.retrievers import BaseRetriever
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.vector_stores.qdrant.utils import SparseEncoderCallable

import yaml
from logging import Logger
from typing import Optional


OPENAPI_FIELD_TYPE = "openAPI_type"
OPENAPI_FIELD_ENTRY = "openAPI_entry"
OPENAPI_FIELD_KEY = "openAPI_key"
OPENAPI_FIELD_PARAMETERS = "parameters"

OPENAPI_TYPE_PARAMETER = "parameter"
OPENAPI_TYPE_PATH = "path"

OPENAPI_REQUIRED_FIELDS = [OPENAPI_FIELD_TYPE, OPENAPI_FIELD_ENTRY, OPENAPI_FIELD_KEY]

def validate_openapi_node(metadata):
    """Validate that the metadata contains all required OpenAPI fields.

    This function checks whether the provided metadata dictionary contains
    all the required OpenAPI fields defined in OPENAPI_REQUIRED_FIELDS.

    Args:
        metadata (dict): A dictionary containing metadata to validate.

    Returns:
        bool: True if all required fields are present in metadata, False otherwise.
    """
    return all(field in metadata for field in OPENAPI_REQUIRED_FIELDS)


def retrieve_chunks_by_custom_attr(qdrant_client, collection_name, custom_attr_value):
    """Retrieve chunks from Qdrant by custom attribute value.

    This function searches for a specific OpenAPI Parameter record in the Qdrant database
    based on the provided OpenAPI key.

    Args:
        qdrant_client: The Qdrant client instance to use for the search.
        collection_name (str): The name of the Qdrant collection to search in.
        custom_attr_value (str): The value of the custom attribute to filter by.

    Returns:
        dict or None: The payload of the found record, or None if no record is found.
    """
    # Create a filter condition for OpenAPI attributes in qdrant database
    filter_condition = qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key=OPENAPI_FIELD_KEY,
                match=qdrant_models.MatchValue(value=custom_attr_value)
            ),
            qdrant_models.FieldCondition(
                key=OPENAPI_FIELD_TYPE,
                match=qdrant_models.MatchValue(value=OPENAPI_TYPE_PARAMETER)
            )
        ]
    )
    # Search for points with the filtered attributes
    response = qdrant_client.scroll(
        collection_name=collection_name,
        scroll_filter=filter_condition,
        limit=1  # We need to extract a specific record with the given openAPI key (there is exacly one such record in the correct database)
    )
    return response[0][0].payload if response[0] else None



class OpenAPIPromptGenerator(BaseModel):
    """Tool to generate prompts with the OpenAPI entries (retireved from the qdrant database)"""

    retriever: BaseRetriever
    qdrant_client: QdrantClient
    qdrant_openapi_collection_name: str
    embedder: OllamaEmbedding
    logger: Logger


    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        embedder: OllamaEmbedding,
        qdrant_client: QdrantClient,
        qdrant_openapi_collection_name: str,
        qdrant_vector_store_parallel: int,
        qdrant_similarity_top_k: int,
        sparse_text_embedding: Optional[SparseEncoderCallable],
        logger: Logger
    ):
        retriever = self._build_retriever(qdrant_client,
                                          embedder,
                                          sparse_text_embedding,
                                          qdrant_openapi_collection_name,
                                          qdrant_vector_store_parallel,
                                          qdrant_similarity_top_k,
                                          )

        super().__init__(embedder=embedder,
                         retriever=retriever,
                         qdrant_client=qdrant_client,
                         qdrant_openapi_collection_name=qdrant_openapi_collection_name,
                         logger=logger,
                         )

    def _build_retriever(self, qdrant_client: QdrantClient,
                         embedder: OllamaEmbedding,
                         sparse_text_embedding: Optional[SparseEncoderCallable],
                         collection_name: str,
                         qdrant_vector_store_parallel: int,
                         qdrant_similarity_top_k: int,
                         ):
        """
        Build a retriever for the specified collection name.
        Args:
            qdrant_client (QdrantClient): The Qdrant client instance.
            embedder (OllamaEmbedding): The embedding model to use.
            collection_name (str): The name of the collection with the OpenAPI entries.
            qdrant_vector_store_parallel (int): The parallelism setting for Qdrant vector store.
            qdrant_similarity_top_k (int): The number of top-k similar items to retrieve.
        Returns:
            VectorStoreIndex: The retriever built from the vector store.
        """

        # Connect to the Qdrant vector store collection with parallelism and collection name
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            parallel=qdrant_vector_store_parallel,
            collection_name=collection_name,
            sparse_doc_fn=sparse_text_embedding,
            sparse_query_fn=sparse_text_embedding,
            enable_hybrid=sparse_text_embedding is not None,
        )
        # Create a storage context from the vector store
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
        )
        retriever = VectorStoreIndex.from_vector_store(
            storage_context=storage_context,
            vector_store=vector_store,
            embed_model=embedder,
        ).as_retriever(
            retriever_mode=ListRetrieverMode.DEFAULT,
            similarity_top_k=qdrant_similarity_top_k,
        )
        return retriever

    def _handle_path_node(self, paths: dict, parameters: dict, metadata: dict) -> None:
        """
        Handle a path node from OpenAPI metadata.

        This method processes path metadata and updates the paths and parameters dictionaries.
        It retrieves parameter information from Qdrant database if not already present.

        Args:
            paths (dict): Dictionary to store path entries
            parameters (dict): Dictionary to store parameter entries
            metadata (dict): Metadata containing path information

        Returns:
            None: Modifies the paths and parameters dictionaries in-place
        """
        paths[metadata[OPENAPI_FIELD_KEY]] = metadata[OPENAPI_FIELD_ENTRY]
        for parameter_key in metadata[OPENAPI_FIELD_PARAMETERS]:
            if parameter_key not in parameters:
                find_parameters_result = retrieve_chunks_by_custom_attr(self.qdrant_client, self.qdrant_openapi_collection_name, parameter_key)
                if find_parameters_result:
                    parameters[parameter_key] = find_parameters_result[OPENAPI_FIELD_ENTRY]

    def _handle_parameter_node(self, parameters: dict, metadata: dict) -> None:
        """
        Handle a parameter node from OpenAPI metadata.

        This method processes parameter metadata and updates the parameters dictionaries.

        Args:
            parameters (dict): Dictionary to store parameter entries
            metadata (dict): Metadata containing paramterer information

        Returns:
            None: Modifies the parameters dictionaries in-place
        """
        parameters[metadata[OPENAPI_FIELD_KEY]] = metadata[OPENAPI_FIELD_ENTRY]

    def generate_yaml(self, query_str: str) -> str:
        """
        Build a yaml file with the selected retrieved entries from the qdrant database
        Args:
            query_str (str): The user query (used to retrieve the revelant entries)
        Returns:
            The string with the generated yaml configuration.
        """

        nodes = self.retriever.retrieve(query_str)
        if not nodes:
            self.logger.debug("No nodes with the OpenAPI entries found for this query: ". query_str)
            return ""
        paths = {}
        parameters = {}
        for n in nodes:
            if not validate_openapi_node(n.metadata):
                self.logger.error("Incorrect chunk, not all required metadata are set: ", n.metadata.keys())
                continue
            field_type = n.metadata[OPENAPI_FIELD_TYPE]
            if field_type == OPENAPI_TYPE_PATH:
                self._handle_path_node(paths, parameters, n.metadata)
            elif field_type == OPENAPI_TYPE_PARAMETER:
                self._handle_parameter_node(parameters, n.metadata)
            else:
                self.logger.error("Incorrect openAPI type: ", field_type)
        return yaml.dump({"paths": paths, "components":{"parameters": parameters}})

    def generate_prompt(self, query_str: str, with_embedded_materials: bool):
        """
        Build a prompt template with the selected retrieved entries from the qdrant database
        Args:
            query_str (str): The user query (used to retrieve the revelant entries)
            with_embedded_materials (bool): Specify if the additional context will be added to prompt
        Returns:
            The string with the generated prompt template
        """
        yaml_file = self.generate_yaml(query_str)
        prompt = prepare_openapi_prompt(yaml_file, with_embedded_materials)
        return prompt
