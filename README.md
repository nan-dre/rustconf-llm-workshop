# Part 2 - Chat with LLM

## Overview
The Chat with LLM workshop will guide you through four essential techniques used for interacting with LLMS:
* Simple chat interactions
* RAG
* Structured outputs
* Tool calling
The final application is a CLI tool which integrates all these techniques.

## Quick Start

### Prerequisites
* Rust
* Llama.cpp - [install guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#cpu-build)

### Deploying the models
```bash
llama-server --embeddings --hf-repo second-state/All-MiniLM-L6-v2-Embedding-GGUF --hf-file  all-MiniLM-L6-v2-ggml-model-f16.gguf --port 8081 # embeddings model available on locahost:8081
llama-server --jinja --hf-repo MaziyarPanahi/gemma-3-1b-it-GGUF --hf-file gemma-3-1b-it.Q5_K_M.gguf # llm available on localhost:8080
```

## Tutorial
You will be working inside the `tutorial.rs` file. The full implementation is available in the `demo.rs` file, in case you get stuck.
In order to run the tutorial, execute:
```bash
cargo run --bin tutorial
```
### 1. Simple Chat Interaction
During the tutorial, we will be using Gemma 3 1B as our language model. The models are deployed using llama.cpp, which exposes an openai-compatible API on port 8080.

We have defined the necessary structs to interact with the model API.

Complete the TODOs to implement the chat interaction logic.
After completing the implementation, run the tutorial and select option 1.

### 2. Retrieval-Augmented Generation (RAG)

In this section, we will implement a RAG system that combines the language model with a document retrieval system.

The embeddings model is also deployed using llama.cpp and exposes a slightly different API on port 8081.

A RAG system is implemented as follows:
1. Calculate embeddings on documents inside the knowledge base.
2. Calculate the embedding of the user query.
3. Get the most similar documents from the knowledge base using the query embedding, with a metric such as cosine similarity.
4. Pass the retrieved documents as context to the language model and generate a response.


### 3. Structured Outputs
Structured outputs are a way to format the model's responses, such that they can be parsed by other systems.

This is done by defining a JSON Schema that describes the structure of the expected output.

In the background, llama.cpp parses this schema and creates a GBNF grammar that guides the model's response generation. More information in the [llama.cpp documentation](https://github.com/ggml-org/llama.cpp/tree/master/grammars).

Keep in mind that using structured outputs can degrade the capabilities of LLMs, as shown by [Tam et al.](https://arxiv.org/abs/2408.02442)

### 4. Tool Calling
Tool calling is a technique that leverages structured outputs. It allows the user to define functions that can be called by the language model and executed in the context of the conversation.


