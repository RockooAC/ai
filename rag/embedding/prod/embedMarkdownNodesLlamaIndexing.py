#! /usr/bin/python3

import argparse
import logging
from GlobalsProd import *
from typing import NoReturn, List

from qdrant_client import QdrantClient
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.core.ingestion import IngestionPipeline, IngestionCache, DocstoreStrategy
from llama_index.core import SimpleDirectoryReader, Document, StorageContext, VectorStoreIndex


def parser():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--docs_path",
        type=str,
        required=True,
        help="Provide path to markdown documentation"
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
        help="Qdrant collection to embed data into"
    )
    return args.parse_args()


class MarkdownEmbedder:
    def __init__(self):
        self.args = parser()
        self.pipeline = None
        self.insert_strategy = DocstoreStrategy.UPSERTS
        self.__setup_logger__()

    def __setup_logger__(self) -> logging:
        self.logger = logging
        self.logger.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )

    @property
    def __embedder__(self) -> OllamaEmbedding:
        return OllamaEmbedding(
            base_url=EMBEDDER_BASE_URL,
            model_name=EMBEDDER_MODEL_GWEN
        )

    @property
    def __doc_splitter__(self) -> SentenceSplitter:
        return SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

    @property
    def __vector_store__(self) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=QdrantClient(QDRANT_BASE_URL),
            collection_name=self.args.collection
        )

    @property
    def __cache__(self) -> IngestionCache:
        return IngestionCache(
            cache=RedisCache.from_host_and_port(
                host=REDIS_HOST, 
                port=REDIS_PORT
            ),
            collection=self.args.cache_collection,
        )

    @property
    def __docstore__(self) -> RedisDocumentStore:
        return RedisDocumentStore.from_host_and_port(
            host=REDIS_HOST, 
            port=REDIS_PORT, 
            namespace=self.args.cache_collection
        )

    @property
    def __markdown_parser__(self) -> MarkdownNodeParser:
        return MarkdownNodeParser.from_defaults()

    def main(self) -> NoReturn:
        self.logger.info("Running docs embedding ingestion pipeline")
        self.pipeline = self.create_ingestion_pipeline()
        nodes = self.pipeline.run(documents=self.process_documents())
        self.logger.info(f"Ingested {len(nodes)} nodes")


    def create_ingestion_pipeline(self) -> IngestionPipeline:
        return IngestionPipeline(
            transformations=[
                # self.__doc_splitter__,
                self.__embedder__,
            ],
            vector_store=self.__vector_store__,
            docstore=self.__docstore__,
            docstore_strategy=self.insert_strategy,
            cache=self.__cache__,
        )

    def read_documents(self) -> List:
        self.logger.info(f"Reading docs from: {self.args.docs_path}")
        documents = SimpleDirectoryReader(
            self.args.docs_path,
            required_exts=[".md"],
            recursive=True,
            filename_as_id=True
        ).load_data(
            show_progress=True,
            num_workers=8
        )
        self.logger.info(f"Loaded {len(documents)} for embedding")
        return documents

    def process_documents(self) -> List:
        self.logger.info(f"Extracting nodes from markdown documents")
        processed_documents = []
        for md_file in self.read_documents():
            for node in self._create_nodes_from_file(file=md_file):
                sentences = self.__doc_splitter__.split_text(node.get_content())
                for sentence in sentences:
                    document = Document(
                        text=sentence,
                        metadata=node.metadata
                    )
                    processed_documents.append(document)
        self.logger.info(f"Processed {len(processed_documents)} documents.")
        return processed_documents

    def _create_nodes_from_file(self, file) -> List:
        self.logger.debug(f"Creating nodes from: {file}")
        base_node = Document(
            text=file.get_content(),
            metadata=file.metadata
        )
        return self.__markdown_parser__.get_nodes_from_node(base_node)



if __name__ == "__main__":
    MarkdownEmbedder().main()
