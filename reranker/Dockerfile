FROM ollama/ollama

ARG FILENAME="runner.sh"

COPY ./${FILENAME} /tmp/${FILENAME}
RUN chmod +x /tmp/${FILENAME}

ENTRYPOINT ["/tmp/runner.sh"]
