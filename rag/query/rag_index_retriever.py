"""
Example:
      python rag_index_retriever.py 'documents' 'How encoding process works?' --local
      python rag_index_retriever.py 'documents' 'How encoding process works?' --remote
"""

import qdrant_client
import argparse
import time
import sys
import os

from typing import List, Union, Generator, Iterator
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Globals import *
from Common import *

class Pipeline:
    def __init__(self):
        self.documents = None
        self.index = None
        self.qdrant = None
        
        Settings.chunk_size = CHUNK_SIZE
        Settings.chunk_overlap = CHUNK_OVERLAP
        Settings.num_output = LLM_NUM_OUTPUT
        Settings.context_window = LLM_CONTEXT_WINDOW
        
        # Initialize the embedding model with the specified model name and base URL
        Settings.embed_model = OllamaEmbedding(
            model_name=EMBEDDER_MODEL_STELLA_EN_V5,
            base_url=EMBEDDER_KS_BASE_URL,
        )

    async def on_startup(self, collection_name: str, local: bool):
        try:
            # Connect to the Qdrant client using the base URL
            qdrant_base_url = QDRANT_LOCAL_BASE_URL if local else QDRANT_BASE_URL
            client = qdrant_client.QdrantClient(url=qdrant_base_url)
            logging.info("Qdrant connected: {qdrant_base_url}")

            # Access the existing Qdrant vector store
            qdrant_vector_store = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                parallel=4
            )
            logging.info(f"Qdrant client accessed {collection_name}.")

            storage_context = StorageContext.from_defaults(
                vector_store=qdrant_vector_store,
            )
            logging.info("Storage context loaded.")

            # A vector store index only needs an embed model
            self.index = VectorStoreIndex.from_vector_store(
                storage_context=storage_context,
                vector_store=qdrant_vector_store,
                parallel=4,
            )
            logging.info("Vector store index loaded.")
            
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            self.index = None

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        
        # Check if the index is properly loaded
        if not self.index:
            raise ValueError("The index is not properly loaded or is None.")

        # Measure the time taken for Qdrant retrieval
        qdrant_start_time = time.time()

        # Retrieve relevant information using the index retriever
        retriever = self.index.as_retriever(
            similarity_top_k=4,
        )

        # Retrieve relevant document chunks
        retrieved_chunks = retriever.retrieve(user_message)

        # Calculate Qdrant retrieval time
        qdrant_end_time = time.time()
        qdrant_time_taken = qdrant_end_time - qdrant_start_time
        
        logging.info(f"Time taken for Qdrant retrieval: {qdrant_time_taken:.2f} seconds")

        # Print the retrieved document chunks
        logging.info("Retrieved document chunks:")
        logging.info("========================================================")
        for chunk in retrieved_chunks:
            logging.info(f"Chunk: {chunk.text}")
            logging.info(f"Metadata: {chunk.metadata}")
            logging.info("========================================================")

        return ""
    
# Guard to ensure the multiprocessing code runs correctly
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process query without uploading documents.')
    parser.add_argument('collection', type=str, help='The name of the collection in Qdrant to access.')
    parser.add_argument('query', type=str, help='The query message to be sent to the chat engine.')
    
    # Use an action for a boolean flag
    parser.add_argument('--local', action='store_true', help='Use local Qdrant database')
    parser.add_argument('--remote', dest='local', action='store_false', help='Use hosted Qdrant database')

    args = parser.parse_args()
    
    import asyncio

    pipeline = Pipeline()

    asyncio.run(pipeline.on_startup(args.collection, args.local))
    
    try:
        response = pipeline.pipe(
            user_message=args.query, 
            model_id="", 
            messages=[], 
            body={}
        )
        print(response)
    except ValueError as e:
        print(e)

    asyncio.run(pipeline.on_shutdown())