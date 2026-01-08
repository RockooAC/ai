#!/bin/bash

echo "Starting Ollama server..."
ollama serve &

echo "Waiting for Ollama server to be active..."
while [ "$(ollama list | grep 'NAME')" == "" ]; do
  sleep 1
done

ls /tmp

touch Modelfile
echo "FROM /tmp/gte-Qwen2-7B-instruct-Q5_K_M.gguf" >> Modelfile

cat Modelfile

ollama create gte-qwen2.5-instruct-q5 -f Modelfile

wait -n