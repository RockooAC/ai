"""
Example:
      python pdf-grobid.py '~/docs' "documents" --local
      python pdf-grobid.py '~/docs' "documents" --remote
"""

import argparse
import logging
import sys
import os
import torch

from llama_index.core import  Document, StorageContext, VectorStoreIndex
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from llama_index.vector_stores.qdrant import QdrantVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.core import Settings
from qdrant_client import QdrantClient, models

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Globals import *
from Common import *
from Grobid import *

import gc

def load_model(model_path):
    
    gc.collect()
    torch.cuda.empty_cache()
    
    logging.info("Loading model and tokenizer...")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=HUGGINGFACE_TOKEN)

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,  # Enable 4/8-bit quantization
    )

    # Load the model with quantization and device map
    model = AutoModel.from_pretrained(
        model_path,
        use_auth_token=HUGGINGFACE_TOKEN,
        #torch_dtype="auto",
        device_map="auto",  # Automatically allocate layers to available devices (CPU/GPU)
        #quantization_config=quantization_config
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    logging.info("Model and tokenizer loaded successfully!")
    return model


def process_documents_in_batches(documents, batch_size=1):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        # Validate that each document in the batch has metadata
        for doc in batch:
            if not hasattr(doc, "metadata"):
                logging.warning(f"Document {doc} is missing metadata.")
        yield batch


def main(directory_path, collection_name, local):
    directory_path = os.path.expanduser(directory_path)
    logging.info(f"Loading PDF documents from directory: {directory_path}")

    if not os.path.exists(directory_path):
        logging.error(f"Directory {directory_path} does not exist.")
        return 1

    qdrant_base_url = QDRANT_LOCAL_BASE_URL if local else QDRANT_BASE_URL
    client = QdrantClient(qdrant_base_url)

    model_path = TRANSFORMERS_EMBEDDER_QWEN2_3B
    model = load_model(model_path)

    Settings.embed_model = HuggingFaceEmbeddings(model_name=model_path)
    logging.info("HuggingFace Embedding loaded.")

    vector_length = Settings.embed_model.get_text_embedding("test").__len__()

    pdf_reader = GrobidPDFReader(grobid_server=GROBID_BASE_URL, split_sentence=False)

    documents = []
    for pdf_file in Path(directory_path).glob("*.pdf"):
        documents.extend(pdf_reader.load_data(file=pdf_file))

    logging.info(f"Embedding to local Qdrant? {local}.")
    logging.info(f"Qdrant base URL: {qdrant_base_url}")
    logging.info(f"Loaded: {len(documents)} documents.")

    if not documents:
        logging.error("No data to embed.")
        return 1

    qdrant_vector_store = QdrantVectorStore(client=client, collection_name=collection_name)

    if client.collection_exists(collection_name):
        logging.info(f"Deleting existing collection '{collection_name}'...")
        client.delete_collection(collection_name)

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

    storage_context = StorageContext.from_defaults(vector_store=qdrant_vector_store)
    logging.info("Storage Context loaded.")

    gc.collect()
    torch.cuda.empty_cache()

    for doc_batch in process_documents_in_batches(documents, batch_size=1):
        with torch.no_grad():
            for doc in doc_batch:
                logging.info(f"Processing Document: {doc.text[:100]}")
                logging.info(f"Metadata: {doc.metadata}")
            VectorStoreIndex.from_documents(
                documents=doc_batch,
                embed_model=Settings.embed_model,
                storage_context=storage_context,
                show_progress=True,
                # Pass metadata if supported
                metadata=[doc.metadata for doc in doc_batch]  # Change this line as needed
            )
            torch.cuda.empty_cache()
            gc.collect()

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