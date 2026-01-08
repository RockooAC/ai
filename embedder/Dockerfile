FROM ollama/ollama

ARG FILENAME="runner.sh"
ARG MODEL="gte-Qwen2-7B-instruct-Q5_K_M.gguf"

WORKDIR /tmp

COPY ./model/${MODEL} /tmp/
COPY ./${FILENAME} /tmp/${FILENAME}

RUN chmod +x /tmp/${FILENAME}

ENTRYPOINT ["/tmp/runner.sh"]
