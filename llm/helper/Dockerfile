FROM ollama/ollama

ARG FILENAME="runner.sh"

COPY ./${FILENAME} /tmp/${FILENAME}

WORKDIR /tmp

RUN chmod +x /tmp/${FILENAME} && "./${FILENAME}"

EXPOSE 11434
