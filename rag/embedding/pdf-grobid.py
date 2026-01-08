"""
Example:
      python pdf-grobid.py '~/docs' "documents" --local
      python pdf-grobid.py '~/docs' "documents" --remote
"""

import argparse
import logging
import sys
import os

from llama_index.core import  Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import Settings
from qdrant_client import QdrantClient, models

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Globals import *
from Common import *
from Grobid import *

def process_documents_in_batches(documents, batch_size=1):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        # Validate that each document in the batch has metadata
        for doc in batch:
            if not hasattr(doc, "metadata"):
                logging.warning(f"Document {doc} is missing metadata.")
        yield batch

def main(directory_path, collection_name, local):

    # Expand the tilde (~) to the user's home directory
    directory_path = os.path.expanduser(directory_path)
    logging.info(f"Loading PDF documents from directory: {directory_path}")
 
     # Debug: Check if the directory path is correct
    if not os.path.exists(directory_path):
        logging.error(f"Directory {directory_path} does not exist.")
        return 1

    # Create QdrantVectorStore instance
    qdrant_base_url = QDRANT_LOCAL_BASE_URL if local else QDRANT_BASE_URL
    client = QdrantClient(qdrant_base_url)

    # Create Ollama Embedding instance
    Settings.embed_model = OllamaEmbedding(
        base_url=EMBEDDER_KS_BASE_URL,  # embedder on Konrad's server
        model_name=EMBEDDER_MODEL_STELLA_EN_V5
    )
    logging.info("Ollama Embedding loaded.")

    # Get the vector length from the embedding model
    vector_length = Settings.embed_model.get_text_embedding("test").__len__()

    # Set the chunk size and overlap for SentenceSplitter
    # 85% of the vector length is used as the chunk size
    # 15% of the chunk size is used as the overlap
    Settings.chunk_size = int(vector_length * 0.8)
    logging.info(f"Chunk size: {Settings.chunk_size}")
    Settings.chunk_overlap = int(Settings.chunk_size * 0.15)
    logging.info(f"Chunk overlap: {Settings.chunk_overlap}")

    Settings.text_splitter = SentenceSplitter(
        chunk_size=Settings.chunk_size,
        chunk_overlap=Settings.chunk_overlap,
    )

    # Load PDFs with detected encoding
    directory_reader = SimpleDirectoryReader(
        input_dir=directory_path,
        file_extractor={
            ".pdf": GrobidPDFReader(GROBID_BASE_URL, split_sentence=False),
        }
    )
    documents = directory_reader.load_data(show_progress=True, num_workers=8)

    logging.info(f"Embedding to local Qdrant? {local}.")
    logging.info(f"Qdrant base URL: {qdrant_base_url}")
    logging.info(f"Loaded: {len(documents)} pages.")

    if not documents:
        logging.error(f"No data to embedd.")
        return 1

    # Create Qdrant Vector Store
    qdrant_vector_store = QdrantVectorStore(client=client, collection_name=collection_name)


    # Check if collection exists in Qdrant and delete it
    if client.collection_exists(collection_name):
        logging.info(f"Deleting existing collection '{collection_name}'...")
        client.delete_collection(collection_name)

    # Create a new collection in Qdrant with the specified vector length
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_length, distance=models.Distance.DOT),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.99,
                always_ram=True,
            ),
        ),
    )
    logging.info(f"Collection '{collection_name}' created successfully.")

    # Create StorageContext instance
    storage_context = StorageContext.from_defaults(vector_store=qdrant_vector_store)
    logging.info("Storage Context loaded.")

    # Create VectorStoreIndex instance
    VectorStoreIndex.from_documents(
        documents,
        embed_model=Settings.embed_model,
        storage_context=storage_context,
        show_progress=True
    )

    logging.info("Indexing completed successfully.")
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process PDF documents for embedding and indexing.')
    parser.add_argument('directory', type=str, help='The directory containing PDF files to process.')
    parser.add_argument('collection', type=str, help='The name of the collection in Qdrant.')
    
    # Use an action for a boolean flag
    parser.add_argument('--local', action='store_true', help='Use local Qdrant database')
    parser.add_argument('--remote', dest='local', action='store_false', help='Use hosted Qdrant database')
    
    args = parser.parse_args()
    
    exit_code = main(args.directory, args.collection, args.local)
    exit(exit_code)