#! /usr/bin/python3

import argparse
from uuid import uuid4
from typing import NoReturn
from ingestionPipeline import ZosIngestion
from llama_index.core.schema import Document


def parser():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Gerrit repository path"
    )
    args.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory."
    )
    args.add_argument(
        "--debug",
        type=bool,
        required=False,
        default=False,
        help="Enable debug mode with verbose logging."
    )
    args.add_argument(
        "--collection",
        type=str,
        required=True,
        help="Qdrant collection to embed data into"
    )
    args.add_argument(
        "--cache_collection",
        type=str,
        required=False,
        default="cache",
        help="Redis collection to cache data into"
    )
    return args.parse_args()


class CodebaseEmbedder(ZosIngestion):
    def __init__(self):
        self.args = parser()
        super().__init__(
            collection_name=self.args.collection,
            cache_collection_name=self.args.cache_collection
        )
        self.documents = dict()

    def main(self) -> NoReturn:
        self.logger.info("Running codebase embedding ingestion pipeline")
        self.pipeline = self.__ingestion_pipeline__(
            transformations=[
                self.__doc_splitter__,
                self.__embedder_jina__,
            ],
            vector_store=self.__bm42_hybrid_vector_store__,
            docstore=None,
            cache=None,
        )

        self._digest_creator()
        self._read_codebase()

        for file, documents in self.documents.items():
            for i in range(0, len(documents), 500):
                nodes = self.pipeline.run(
                    show_progress=True,
                    documents=documents[i:i + 500]
                )
                self.logger.info(f"Ingested {len(nodes)} nodes for {file} (chunk {i // 500 + 1})")

    def _digest_creator(self) -> NoReturn:
        from src.rag.config import REPOSITORY_CONFIG

        self.logger.info(f"Creating digest files from codebase in: {self.args.directory}")
        self.digest_files = self.__codebase_parser__(
            source_dir=self.args.directory,
            output_dir=self.args.output,
            debug=self.args.debug
        ).process_repo(exclude_patterns=REPOSITORY_CONFIG["RG_EXCLUDE_PATTERNS"])
        self.documents = {file: [] for file in self.digest_files.values()}

    def _read_codebase(self) -> NoReturn:
        self.logger.info(f"Running digest files reading")
        for file in self.digest_files.values():
            data = self.__cpp_code_reader__(debug=self.args.debug).load_data(file)
            for name, info in data.items():
                for chunk in info["chunks"]:
                    doc = chunk.to_document(info["source"], info.get("extra_info"))
                    if doc is not None:
                        doc.id_ = str(uuid4())
                        if self.filter_valid_document(doc):
                            self.documents[file].append(doc)

    def filter_valid_document(self, doc: Document) -> bool:
        doc_id = getattr(doc, "id_", "unknown")
        text = doc.text if hasattr(doc, "text") else None
        if not text or not text.strip():
            self.logger.info(f"Empty text in document {doc_id}, type: {type(doc)}")
            return False
        if len(text.split()) > 10 and all(c in "0123456789abcdefABCDEF, \n" for c in text):
            self.logger.info(f"Binary-like content in document {doc_id}")
            return False
        return True


if __name__ == "__main__":
    CodebaseEmbedder().main()
