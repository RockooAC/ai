#!/usr/bin/python3

import argparse
from typing import NoReturn
from ingestionPipeline import ZosIngestion


def parser():
    args = argparse.ArgumentParser()
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
    args.add_argument(
        "--url",
        required=False,
        type=str,
        metavar="Website url to scrape and embed"
    )
    return args.parse_args()


class WebContentEmbedder(ZosIngestion):
    def __init__(self):
        self.args = parser()
        super().__init__(
            collection_name=self.args.collection,
            cache_collection_name=self.args.cache_collection
        )
        self.documents = list()

    def main(self) -> NoReturn:
        self.logger.info("Running Website embedding ingestion pipeline")
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
        self.documents = self._read_webpage()

        nodes = self.pipeline.run(
            documents=self.documents
        )

        self.logger.info(f"Ingested {len(nodes)} nodes")

    def _read_webpage(self) -> ZosIngestion.__confluence_reader__:
        self.logger.info("Running webpage pages reading")
        return self.__webpage_reader__().load_data(
            urls=[self.args.url],
        )


if __name__ == "__main__":
    WebContentEmbedder().main()
