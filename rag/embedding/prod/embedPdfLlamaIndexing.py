#! /usr/bin/python3


import argparse
from typing import NoReturn
from ingestionPipeline import ZosIngestion


def parser():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--pdfs_path",
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
        help="Redis collection to cache data into"
    )
    return args.parse_args()


class PdfEmbedder(ZosIngestion):
    def __init__(self):
        self.args = parser()
        super().__init__(
            collection_name=self.args.collection,
            cache_collection_name=self.args.cache_collection
        )
        self.documents = list()

    def main(self) -> NoReturn:
        self.logger.info("Running pdfs embedding ingestion pipeline")
        self.pipeline = self.__ingestion_pipeline__(
            transformations=[
                self.__doc_splitter__,
                self.__embedder__,
            ],
            vector_store=self.__bm42_hybrid_vector_store__,
            docstore=self.__docstore__,
            docstore_strategy=self.insert_strategy,
            cache=self.__cache__,
        )

        self._read_documents()
        self._adjust_dynamic_metadata(self.documents)

        for i in range(0, len(self.documents), 500):
            nodes = self.pipeline.run(
                show_progress=True,
                documents=self.documents[i:i + 500]
            )
            self.logger.info(f"Ingested {len(nodes)} nodes for (chunk {i // 500 + 1})")

    def _read_documents(self) -> NoReturn:
        self.logger.info(f"Reading pdfs from: {self.args.pdfs_path}")
        self.documents = self.__pdf_reader__(
            pdf_path=self.args.pdfs_path
        ).load_data(
            show_progress=True,
            num_workers=16
        )


if __name__ == "__main__":
    PdfEmbedder().main()
