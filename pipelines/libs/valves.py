from llama_index.core.bridge.pydantic import BaseModel, Field

from libs.variables import (
    DEFAULT_SPARSE_TEXT_EMBEDDING_MODEL,
    DEFAULT_SENTENCE_TRANSFORMER_RERANK_MODEL,
)

class Valves(BaseModel):
        # Ollama settings
        OLLAMA_MODEL_BASE_URL: str = Field(default="http://10.255.240.156:11434")
        OLLAMA_MODEL_NAME: str = Field(default="qwen3:30b")
        OLLAMA_CONTEXT_WINDOW: int = Field(default=61440)
        OLLAMA_EMBEDDING_BASE_URL: str = Field(default="http://10.255.240.149:11434")
        OLLAMA_EMBEDDING_MODEL_NAME: str = Field(default="gte-qwen2.5-instruct-q5")
        OLLAMA_CHUNK_SIZE: int = Field(default=1024)
        OLLAMA_CHUNK_OVERLAP: int = Field(default=256)
        OLLAMA_TEMPERATURE: float = Field(default=0.5)
        # Ollama rerank settings
        OLLAMA_RERANK_ACTIVE: bool = Field(default=True)
        OLLAMA_RERANK_BASE_URL: str = Field(default="http://10.255.240.156:11434")
        OLLAMA_RERANK_MODEL_NAME: str = Field(default="llama3.2:3b")
        OLLAMA_RERANK_TEMPERATURE: float = Field(default=0.5)
        OLLAMA_RERANK_TOP_N: int = Field(default=10)
        OLLAMA_RERANK_CHOICE_BATCH_SIZE: int = Field(default=5)
        # Qdrant settings
        QDRANT_BASE_URL: str = Field(default="http://10.255.240.18:6333")
        QDRANT_COLLECTION_NAME: str = Field(default="confluence_prod_bm42,docusaurus_prod_bm42,pdf_prod_bm42")
        QDRANT_OPENAPI_COLLECTION_NAME: str = Field(default="openapi_prod_bm42")
        QDRANT_VECTOR_STORE_PARALLEL: int = Field(default=4)
        QDRANT_SIMILARITY_TOP_K: int = Field(default=20)
        QDRANT_HYBRID_SEARCH: bool = Field(default=True)
        QDRANT_SIMILARITY_CUTOFF_ACTIVE: bool = Field(default=True)
        QDRANT_SIMILARITY_CUTOFF: float = Field(default=0.3)
        # Sentence transformer rerank settings
        SENTENCE_TRANSFORMER_RERANK_ACTIVE: bool = Field(default=True)
        SENTENCE_TRANSFORMER_RERANK_MODEL: str = Field(default=DEFAULT_SENTENCE_TRANSFORMER_RERANK_MODEL)
        SENTENCE_TRANSFORMER_RERANK_TOP_N: int = Field(default=20)
        # Pipeline settings
        PIPELINE_TIME_RECORDING: bool = Field(default=True)
        CHAT_HISTORY_ACTIVE: bool = Field(default=True)
        FORCE_ONLY_EXTERNAL_SOURCES: bool = Field(default=True)
        SPARSE_TEXT_EMBEDDING_MODEL: str = Field(default=DEFAULT_SPARSE_TEXT_EMBEDDING_MODEL)

        def __init__(self, **data):
            super().__init__(**data)
