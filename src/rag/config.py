import os

EMBEDDER_CONFIG = {
    "BASE_URL": "http://10.255.246.131:11434",  # Default embedder on c24 server
    "KS_BASE_URL": "http://10.255.240.161:11434",  # Embedder on ks server
    "MODELS": {
        "JINA": {  # Best for code embedding
            "ollama": "jina/jina-embeddings-v2-base-en:latest",
            "hf": "jinaai/jina-embeddings-v2-base-en",  # HuggingFace
        },
        "QWEN": {  # Best for textual content
            "ollama": "gte-qwen2.5-instruct-q5",
            "hf": "Alibaba-NLP/gte-Qwen2-7B-instruct",  # HuggingFace
        },
    },
    "VECTOR_SIZES": {
        768: "JINA",
        3584: "QWEN",
    },
}

QDRANT_CONFIG = {
    "BASE_URL": "http://10.255.240.18:6333",  # Qdrant server
    "BASE_URL_LOCAL": "http://localhost:6333",  # Local Qdrant server
}

LLM_CONFIG = {
    "BASE_URL": "http://10.255.240.156:11434",
    "DEFAULT_MODEL": "llama3.1:latest",
    "CODE_MODEL": "deepseek-coder-v2:latest",
    "CONTEXT_WINDOW": 96000,
}

REDIS_CONFIG = {
    "HOST": "10.255.240.18",
    "PORT": 6379,
}

CHUNK_CONFIG = {
    "SIZE": 1024,
    "OVERLAP": 128,
}

GROBID_CONFIG = {
    "ENABLED": True,
    "BASE_URL": "http://10.255.240.18:8070",
    "SPLIT_SENTENCE": True,
    "LOAD_FIGURES": True,
}

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

EXTERNAL_URLS = {
    "CONFLUENCE": "https://confluence.redge.com",
}

REPOSITORY_CONFIG = {
    "RG_EXCLUDE_PATTERNS": [
        ".vscode/**",
        "doc/**",
        "build/**",
        "build-conan/**",
        "conan/**",
        "django/**",
        "documents/**",
        "solutions/**",
        "graphs/**",
        "etc/**",
        "release/**",
        "qa/**",
        "tool/**",
        "**/*.a",
        "**/*.o",
        "**/*.so",
        "**/*.livx",
        "**/*.isml",
        "**/*.lcg",
        "*.DS_Store",
        "**/*.log",
        "**/*.tmp",
        "**/*.xml",
        "**/*.png",
        "**/*.jpg",
        "**/*.bin",
        "**/*.m3u8",
        "**/*.json",
        "**/*.md",
        "**/*.ts",
        "**/*.go",
        "**/*.js",
        "**/*.py",
        "**/*.css",
        "**/*.txt",
        "**/*.conf",
        "**/*.bat",
        "**/*.dot",
        "**/*.cfg",
        "*.code-workspace",
        "**/*.yaml",
        "**/*.yml",
        "**/*.clang-format",
        "**/*.clang-tidy",
        "**/*.jenkinsfile",
        "**/Makefile",
        "**/*.cmake",
        "**/*.sh",
        "VERSION",
        "**/tessdata/*",
        "**/*.tff",
        "**/*.sln",
        "**/*.data",
        "**/data/blur/*",
        "**/buildtools/*",
        "**/doxygen/*",
        "**/docker/*",
        "*.Dockerfile",
        "**/Dockerfile",
        "**/Dockerfile.*",
        "**/cmake/*",
        "*/clang-tidy/**",
    ],
    "CHUNK_SIZE": 1024,
}