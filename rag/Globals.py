
import os

# URL and model name for the Ollama Embedding service
EMBEDDER_BASE_URL = "http://10.255.246.131:11434"  # embedder on c24 server
EMBEDDER_KS_BASE_URL = "http://10.255.240.161:11434"  # embedder on ks server
EMBEDDER_MODEL_MXBAI = "mxbai-embed-large:latest"
EMBEDDER_MODEL_STELLA = "Losspost/stella_en_1.5b_v5:latest"
EMBEDDER_MODEL_GWEN = "rjmalagon/gte-qwen2-1.5b-instruct-embed-f16:latest"
EMBEDDER_MODEL_GWEN2_3B = "qwen2.5:3b"
EMBEDDER_MODEL_QWEN2_7B_INSTRUCT = "gte-Qwen2-7B-instruct"
EMBEDDER_MODEL_GWEN_3B_INSTRUCT = "qwen2.5-3b-instruct:latest"
EMBEDDER_MODEL_SFR_EMBEDDING_Q4 = "ggml-sfr-embedding-mistral-q4" # https://huggingface.co/cminja/sfr-embedding-mistral-GGUF
EMBEDDER_MODEL_STELLA_EN_V5 = "stella-en-1-5-v5" # https://huggingface.co/abhishekbhakat/stella_en_1.5B_v5_GGUF

TRANSFORMERS_EMBEDDER_QWEN2 = "Qwen/Qwen2.5"
TRANSFORMERS_EMBEDDER_QWEN2_3B = "Qwen/Qwen2.5-3B"
TRANSFORMERS_EMBEDDER_STELLA = "dunzhang/stella_en_1.5B_v5"
TRANSFORMERS_EMBEDDER_SFR = "Salesforce/SFR-Embedding-2_R"

LLM_BASE_URL = "http://10.255.240.156:11434"
LLM_MODEL_NAME = "llama3.1:latest"
LLM_CONTEXT_WINDOW = 48000
LLM_NUM_OUTPUT = 32000 # context_window + num_output <= total context length

QDRANT_BASE_URL = "http://10.255.240.18:6333"
QDRANT_LOCAL_BASE_URL = "http://localhost:6333"

# REDIS CACHE
REDIS_HOST = "10.255.240.18"
REDIS_PORT = 6379

MAX_TOKEN_LENGTH = 512
CHUNK_SIZE = 2048 # Max token size for chunk for mxbai is 512, so max character length is 3 or 4 * 512
CHUNK_OVERLAP = 256

CONFLUENCE_BASE_URL = "https://confluence.redge.com"

# GROBID SERVER
GROBID_ENABLED = True
GROBID_BASE_URL = "http://10.255.240.18:8070"
GROBID_SPLIT_SENTENCE = True
GROBID_LOAD_FIGURES = True

# Huggingface token, read from env
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")