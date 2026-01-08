# README

## Git commands to replicate environment

```text
git clone --branch dev --depth 25 ssh://git.atendesoftware.pl:29418/atds/drm/zos/red-galaxy/core product/rg
cd product/rg
git remote add changes ssh://git.atendesoftware.pl:29418/atds/drm/zos/red-galaxy/core
git fetch changes refs/changes/74/17874/6

git checkout -b ai_review FETCH_HEAD
```

### Example commands executed while processing

```text
git rebase dev
git diff dev..ai_review
git diff --name-only dev..ai_review
git diff -U10 dev..ai_review -- src/rg/core/streams/io/ConfigFrameJson.h
```


## Potential improvements from research



1. Add examples to Helper LLM system prompt

2. Move whole diff processing methods to dedicated file diffAnalyzer.py from utils

3. Verify joining identifiers from static extraction and from helper LLM

4. Think about matching git diffs with tree-sitter chunks while we are using 25 lines git context ( might be useless )

5. Cross diff check, what about the idea of passing other diffs from the same file to increase context ????
   ( this idea made loop like infinite, and increased much the processing memory usage)

---


1. Multiretriever
- Create more elastic qdrant extraction for multi collection usage
- Research regex
- Verify retries if 0 nodes found, have to discover some logic here

2. Pipeline System Prompt
- ~~Create more comprehensive system prompt for git diff analysis to fully exclude wrong interpretation~~
- Define more elastic approach to processing standard C++ methods usage in diff

3. Helper LLM
- ~~Once again research concept like Multi-Stage query to split complex queries into simpler one~~
- Reconsider system prompt for Mistral CodeMethodsExtractor in template.py

4. Main LLM
- Verify context limits, maybe possible to put more info to prompt and context

5. Other file types
- Decide if processing other files like py/yaml/sh etc. that are minority and mostly utils part of zos codebase

6. Golang support
- Go reader will be ready soon, so include here the issue to take gitlab integration into consideration
- Possible think here is that golang will have better embedded database, 
so maybe retriever per se will handle most edge cases existing in C++ flow


# Stage 3

1. Include python code reader flow processing, maybe Qdrant retriever with
option to query about any language defined in function param to avoid using mapping in pipeline or other pipelines

