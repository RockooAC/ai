## Usage:
##      python rag.py "documents-llama-splitter" "~/docs" True "How low-latency work in DASH?"
##

import qdrant_client
import argparse
import asyncio
import sys
import os

from typing import List, Union, Generator, Iterator
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Globals import *

class Pipeline:
    def __init__(self):
        self.index = None

        Settings.chunk_size = CHUNK_SIZE
        Settings.chunk_overlap = CHUNK_OVERLAP
        Settings.num_output = LLM_NUM_OUTPUT
        Settings.context_window = LLM_CONTEXT_WINDOW

        Settings.embed_model = OllamaEmbedding(
            model_name=EMBEDDER_MODEL_MXBAI,
            base_url=EMBEDDER_BASE_URL,
        )

        Settings.llm = Ollama(
            model=LLM_MODEL_NAME,
            base_url=LLM_BASE_URL,
            request_timeout=90.0,
        )

    async def on_startup(self, collection_name: str, directory: str, local: bool):
        try:
            # Load documents
            print("Loading data...")
            directory_reader = SimpleDirectoryReader(directory, recursive=True, required_exts=[".pdf"])
            documents = directory_reader.load_data(show_progress=True, num_workers=8)
            print("Data loaded")

            # Connect to the Qdrant client
            qdrant_base_url = QDRANT_LOCAL_BASE_URL if local else QDRANT_BASE_URL
            client = qdrant_client.QdrantClient(url=qdrant_base_url)
            print("Qdrant client connected")

            # Create Qdrant vector store
            qdrant_vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
            print(f"Qdrant vector store created for collection: {collection_name}")

            # Create storage context
            storage_context = StorageContext.from_defaults(vector_store=qdrant_vector_store)
            print("Storage context created")

            # a vector store index only needs an embed model
            self.index = VectorStoreIndex.from_documents(
                documents=documents,
                embed_model=Settings.embed_model,
                storage_context=storage_context,
            )
            print("Index created and documents uploaded")

        except Exception as e:
            print(f"Error loading data: {e}")
            self.index = None

    async def on_shutdown(self):
        # Placeholder for shutdown tasks if needed
        pass

    def pipe(self, user_message: str, model_id: str, chat_mode: str = "condense_question") -> Union[str, Generator, Iterator]:
        # Check if the index is properly loaded
        if not self.index:
            raise ValueError("The index is not properly loaded or is None.")

        # Create a query engine from the index
        query_engine = self.index.as_query_engine(llm=Settings.llm)
        
        # Get response from the query engine
        response = query_engine.query(user_message)

        return response.response

# Guard to ensure the multiprocessing code runs correctly
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process and query documents using a chat engine.')
    parser.add_argument('collection', type=str, help='The name of the collection in Qdrant to access.')
    parser.add_argument('directory', type=str, help='Directory to access documents')
    parser.add_argument('local', type=bool, help='Whether to use local Qdrant database or hosted')
    parser.add_argument('query', type=str, help='The query message to be sent to the chat engine.')
    
    args = parser.parse_args()

    print(f"\nCollection: {args.collection}\nDirectory: {args.directory}\nQuery: {args.query}")


    pipeline = Pipeline()

    asyncio.run(pipeline.on_startup(args.collection, args.directory, args.local))
    try:
        response = pipeline.pipe(
            user_message=args.query,
            model_id=""
        )
        print(response)
    except ValueError as e:
        print(e)

    asyncio.run(pipeline.on_shutdown())