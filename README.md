# Rustconf LLM Workshop

## Prerequisites
* Cargo
* Llama.cpp - [install guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#cpu-build)

## Deploying the models
```bash
cd llama.cpp/build/bin
./llama-server --embeddings --hf-repo second-state/All-MiniLM-L6-v2-Embedding-GGUF --hf-file  all-MiniLM-L6-v2-ggml-model-f16.gguf --port 8081 # embeddings model available on locahost:8081
./llama-server --jinja --hf-repo MaziyarPanahi/gemma-3-1b-it-GGUF --hf-file gemma-3-1b-it.Q5_K_M.gguf # llm available on localhost:8080

```
