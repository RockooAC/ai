#!/bin/bash

echo "Starting Ollama server..."
ollama serve &

echo "Waiting for Ollama server to be active..."
while [ "$(ollama list | grep 'NAME')" == "" ]; do
  sleep 1
done

echo "Pulling model..."
ollama pull dengcao/Qwen3-Reranker-8B:Q8_0

wait -n