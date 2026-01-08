# Function to detect encoding of a file
import logging
import os
from typing import List

import chardet
from llama_index.readers.file import PDFReader

from qdrant_client import QdrantClient
from llama_index.core import SimpleDirectoryReader, Document
from bs4 import UnicodeDammit
import re
import spacy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the spaCy model
# Install the spaCy model with: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

# Can handle encoding detection more effectively in some cases, especially when the content is in HTML or XML-like formats
def detect_encoding_with_unicodedammit(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    suggestion = UnicodeDammit(raw_data)
    return suggestion.original_encoding

# Function for encoding detection
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read(4096)  # Increase byte sample size

    # Check for BOM (Byte Order Mark)
    if raw_data.startswith(b'\xff\xfe') or raw_data.startswith(b'\xfe\xff'):
        return 'utf-16'
    elif raw_data.startswith(b'\xef\xbb\xbf'):
        return 'utf-8'
    
    # Use chardet for encoding detection
    result = chardet.detect(raw_data)
    
    # If chardet fails, use UnicodeDammit as a backup
    if not result['encoding']:
        suggestion = UnicodeDammit(raw_data)
        return suggestion.original_encoding or 'utf-8'  # Fallback to utf-8

    return result['encoding']


# Check if a collection exists in Qdrant
def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    try:
        # Check if the collection exists
        exists = client.collection_exists(collection_name)
        return exists
    except Exception as e:
        logging.error(f"An error occurred while checking for collection existence: {e}")
        return False


# Function to load PDFs from a directory with detected encoding
def load_pdfs_with_detected_encoding(directory_path: str, return_full_document: bool = False) -> List[Document]:
    documents = []

    # Walk through the directory to detect encoding and load files
    for root, _, files in os.walk(directory_path):
        logging.info(f"Checking directory: {root}")  # Debug statement
        for file in files:
            logging.info(f"Found file: {file}")  # Debug statement
            if file.lower().endswith('.pdf'):  # Case-insensitive check
                file_path = os.path.join(root, file)
                logging.info(f"Processing PDF file: {file_path}")  # Debug statement
                try:
                    encoding = detect_encoding(file_path)
                    logging.info(f"Detected encoding: {encoding}")

                    # Create a SimpleDirectoryReader instance for the specific file
                    directory_reader = SimpleDirectoryReader(
                        input_files=[file_path],
                        encoding=encoding,
                        file_extractor={
                            ".pdf": PDFReader(return_full_document=return_full_document)
                        },
                    )
                    pdf_data = directory_reader.load_data(show_progress=True, num_workers=1)
                    
                    # Append loaded PDF data to the documents list
                    for doc in pdf_data:
                        if doc.text:
                            doc.text = normalize_text(doc.text)
                            documents.extend(pdf_data)
                    
                except Exception as e:
                    logging.error(f"Failed to process {file_path}: {e}")

    return documents

def normalize_text(text: str) -> str:
    """Normalize text."""
    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text).strip()

    # TODO - Add more text normalization steps here
    # ...

    # convert to UTF-8 encoding
    text = text.encode("utf-8", errors="ignore").decode("utf-8")

    return preprocess_text(text)

def preprocess_text(text):
    doc = nlp(text)
    lemmatized = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(lemmatized)