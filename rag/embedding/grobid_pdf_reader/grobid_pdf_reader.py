import logging
from pathlib import Path
from typing import Dict, List, Optional

from fsspec import AbstractFileSystem
from grobid_client import Client
from grobid_client.api.pdf import process_fulltext_document
from grobid_client.models import Article, ProcessForm
from grobid_client.types import TEI, File
from llama_index.core.readers.base import BaseReader
from llama_index.core.readers.file.base import get_default_fs
from llama_index.core.schema import Document
import spacy
import re

logger = logging.getLogger(__name__)
nlp = spacy.load("en_core_web_sm")
MIN_SENTENCE_LEN = 10


def normalize_text(text: str) -> str:
    """
    Normalize text by removing newlines, extra spaces, and converting to UTF-8 encoding.

    Args:
        text (str): The input text to be normalized.

    Returns:
        str: The normalized text.
    """
    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text).strip()
    # convert to UTF-8 encoding
    text = text.encode("utf-8", errors="ignore").decode("utf-8")

    return preprocess_text(text)

def preprocess_text(text) -> str:
    """
    Preprocess the input text by lemmatizing and removing stop words.

    Args:
        text (str): The input text to be preprocessed.

    Returns:
        str: The preprocessed text with lemmatized tokens and without stop words.
    """
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])


def append_docs(documents: List[Document], text: str, metadata: Optional[Dict] = None) -> List[Document]:
    """
    Append a new Document to the list of documents if the text meets the minimum sentence length.

    Args:
        documents (List[Document]): The list of Document objects to append to.
        text (str): The text content of the new Document.
        metadata (Optional[Dict]): Optional metadata for the new Document.

    Returns:
        List[Document]: The updated list of Document objects.
    """
    text = normalize_text(text)
    if text.__len__() < MIN_SENTENCE_LEN:
        return documents

    documents.append(Document(text=text, metadata=metadata))

class GrobidPDFReader(BaseReader):
    """Grobid PDF parser."""

    def __init__(self, grobid_server: str, split_sentence: Optional[bool] = False, load_figures: Optional[bool] = False) -> None:
        """
        Initialize GrobidPDFReader.

        Args:
            grobid_server (str): The URL of the Grobid server.
            split_sentence (Optional[bool]): Whether to split sentences into separate documents.
        """
        self.client = Client(base_url=grobid_server+"/api", verify_ssl=False, timeout=300)
        self.split_sentence = split_sentence
        self.load_figures = load_figures

    def load_data(
            self,
            file: Path,
            extra_info: Optional[Dict] = None,
            fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """
        Parse a PDF file and extract its content into a list of Document objects.

        Args:
            file (Path): The path to the PDF file to be processed.
            extra_info (Optional[Dict]): Additional metadata to include in the documents.
            fs (Optional[AbstractFileSystem]): The filesystem to use for opening the file.

        Returns:
            List[Document]: A list of Document objects containing the extracted content.
        """
        logger.info(f"Processing {file.name}...")
        file = Path(file) if not isinstance(file, Path) else file

        fs = fs or get_default_fs()
        with fs.open(str(file), "rb") as fin:

            resp = process_fulltext_document.sync_detailed(
                client=self.client,
                multipart_data=ProcessForm(
                    generate_ids="0",
                    consolidate_header="0",
                    consolidate_citations="0",
                    include_raw_citations="0",
                    include_raw_affiliations="0",
                    tei_coordinates="0",
                    segment_sentences="1",
                    input_=File(
                        file_name=file.name,
                        payload=fin,
                        mime_type=extra_info['file_type'] if extra_info and 'file_type' in extra_info else "application/pdf",
                    ),
                ),
            )
            if not resp.is_success:
                logger.error(f"Failed to process {file.name}: {resp.content}")
                return []

            # Parse the TEI XML response
            article: Article = TEI.parse(resp.content, figures=self.load_figures)

            # Initialize metadata
            metadata = {
                "file_name": file.name,
                "file_title": article.title.title(),
                "file_md5": article.identifier,
            }
            if extra_info:
                metadata.update(extra_info)

            documents = []
            # Iterate over sections and paragraphs
            for section_idx, section in enumerate(article.sections):
                if not section.paragraphs:
                    continue

                # Add section metadata
                metadata.update({
                    "section_idx": section_idx+1, # For more human-readable section numbering
                    "section_name": section.name,
                    "section_num": section.num if section.num else "",
                    "section_sentences_len": len(section.paragraphs),
                })
                doc_text = ""
                for sentence_idx, sentence in enumerate(section.paragraphs):
                    if not sentence:
                        continue

                    # Split sentences into separate documents
                    if self.split_sentence:
                        metadata["sentence_idx"] = sentence_idx + 1 # For more human-readable section numbering
                        doc_text = ""

                    # Join phrases in a sentence
                    doc_text += " ".join(phrase.text for phrase in sentence) + " "
                    if self.split_sentence and len(doc_text) > MIN_SENTENCE_LEN:
                        append_docs(documents=documents, text=doc_text, metadata=metadata)

                # Append the full section text as a single document
                if not self.split_sentence:
                    append_docs(documents=documents, text=doc_text, metadata=metadata)
            logger.info(f"Extracted {len(documents)} documents from {file.name}.")
            return documents