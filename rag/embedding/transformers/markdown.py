import argparse
import logging
import os
import torch
import sys
from llama_index.core import SimpleDirectoryReader, Document, StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient, models

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Globals import *
from Common import *

import gc

# Load the local model and tokenizer
def load_model(model_path):
    logging.info("Loading model and tokenizer...")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=HUGGINGFACE_TOKEN)

    # Load the model with quantization and device map
    model = AutoModel.from_pretrained(
        model_path,
        use_auth_token=HUGGINGFACE_TOKEN,
        device_map="auto",  # Automatically allocate layers to available devices (CPU/GPU)
        torch_dtype=torch.float16
        #quantization_config=quantization_config,  # Apply 8-bit quantization
    )

    logging.info("Model and tokenizer loaded successfully!")
    return tokenizer, model

def main(directory_path, collection_name, local, debug):
    # Load the model and tokenizer
    model_path = TRANSFORMERS_EMBEDDER_QWEN2
    tokenizer, model = load_model(model_path)

    # Create the embedding model using HuggingFaceEmbeddings
    embed_model = HuggingFaceEmbeddings(model_name=model_path)

    # Create QdrantVectorStore instance
    qdrant_base_url = QDRANT_LOCAL_BASE_URL if local else QDRANT_BASE_URL
    client = QdrantClient(qdrant_base_url)

    # Load Markdown documents from a directory
    markdown_reader = SimpleDirectoryReader(input_dir=directory_path, required_exts=[".md"], recursive=True)
    markdown_files = markdown_reader.load_data(show_progress=True, num_workers=8)

    if not markdown_files:
        logging.error("No markdown data to embed.")
        return 1

    logging.info(f"Markdown files: {len(markdown_files)}.")
    logging.info(f"Directory path: {directory_path}.")
    logging.info(f"Embedding to local Qdrant: {local}.")
    logging.info(f"Qdrant base URL: {qdrant_base_url}")
    logging.info(f"Loaded {len(markdown_files)} markdown files.")

    # Extract text and metadata from Markdown files
    documents = []
    logging.info("Parsing and splitting content...")

    for md_file in markdown_files:
        content = md_file.get_content()
        metadata = md_file.metadata

        if debug:
            logging.debug(f"------------------- Content -------------------: \n {content}")

        document = Document(text=content, metadata=metadata)
        documents.append(document)

    logging.info(f"Processed {len(documents)} documents.")

    # Get the vector length from the embedding model
    vector_length = embed_model.embed_query("test").__len__()

    # Check if collection exists in Qdrant and delete it
    if collection_exists(client, collection_name):
        logging.info(f"Deleting existing collection '{collection_name}'...")
        client.delete_collection(collection_name)

    # Recreate collection with custom settings
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_length, distance=models.Distance.COSINE),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.95,
                always_ram=True,
            ),
        ),
    )
    
    qdrant_vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    
    gc.collect()
    torch.cuda.empty_cache()    
    
    # Create StorageContext instance
    storage_context = StorageContext.from_defaults(vector_store=qdrant_vector_store)
    logging.info("Storage Context loaded.")
    
    # Create VectorStoreIndex instance
    VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        storage_context=storage_context,
        show_progress=True
    )
    
    logging.info("Embedding completed successfully.")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process Markdown documents for embedding and indexing.')
    parser.add_argument('directory', type=str, help='The directory containing Markdown files to process.')
    parser.add_argument('collection', type=str, help='The name of the collection in Qdrant.')

    # Use an action for a boolean flag
    parser.add_argument('--local', action='store_true', help='Use local Qdrant database')
    parser.add_argument('--remote', dest='local', action='store_false', help='Use hosted Qdrant database')
    parser.add_argument('--debug', dest='debug', action='store_true', help='Enable debug mode for markdown splitting')

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    exit_code = main(args.directory, args.collection, args.local, args.debug)
    exit(exit_code)
