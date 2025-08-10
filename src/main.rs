//! # LLM Workshop: Building an Agent with Tools and RAG
use dotenvy::dotenv;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::env;
use std::error::Error;
use std::io::{self, Write};
use log::info;

// --- 1. COMMON TYPES ---

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Tool {
    pub r#type: String,
    pub function: Function,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Function {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: FunctionCall,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Deserialize, Debug)]
pub struct AssistantMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Deserialize, Debug)]
pub struct Choice {
    pub message: AssistantMessage,
}

#[derive(Deserialize, Debug)]
pub struct ChatCompletion {
    pub choices: Vec<Choice>,
}

#[derive(Serialize, Debug)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub tools: Vec<Tool>,
}

// --- 2. RETRIEVAL-AUGMENTED GENERATION (RAG) ---
// We'll build a simple vector database to store and retrieve information.

mod rag {
    use super::*;

    #[derive(Deserialize)]
    struct EmbeddingResponse {
        #[serde(rename = "index")]
        _index: i32,
        embedding: Vec<Vec<f32>>,
    }

    /// A simple in-memory vector database.
    pub struct VectorDB {
        client: Client,
        embeddings_url: String,
        documents: Vec<(String, Vec<f32>)>,
    }

    impl VectorDB {
        pub fn new(embeddings_url: &str) -> Self {
            VectorDB {
                client: Client::new(),
                embeddings_url: embeddings_url.to_string(),
                documents: Vec::new(),
            }
        }

        /// Gets embeddings for a given text from an external service.
        async fn get_embeddings(&self, text: &str) -> Result<Vec<f32>, Box<dyn Error>> {
            let payload = json!({ "content": text });
            let response: Vec<EmbeddingResponse> = self
                .client
                .post(&self.embeddings_url)
                .json(&payload)
                .send()
                .await?
                .json()
                .await?;
            Ok(response.first()
                .ok_or("No embeddings found")?
                .embedding
                .first()
                .ok_or("No embedding vector found")?
                .clone())
        }

        /// Adds a document to our database, automatically creating embeddings.
        pub async fn add_document(&mut self, doc: &str) -> Result<(), Box<dyn Error>> {
            info!("Embedding document: \"{}\"", doc);
            let embedding = self.get_embeddings(doc).await?;
            self.documents.push((doc.to_string(), embedding));
            Ok(())
        }

        /// Finds the most relevant document in the DB for a given query.
        pub async fn find_most_similar(&self, query: &str) -> Result<Option<(String, f32)>, Box<dyn Error>> {
            const SIMILARITY_THRESHOLD: f32 = 0.1;

            let query_embedding = self.get_embeddings(query).await?;
            let mut best_match: Option<(&str, f32)> = None;

            for (doc, doc_embedding) in &self.documents {
            let similarity = cosine_similarity(&query_embedding, doc_embedding);

            if best_match.is_none() || similarity > best_match.unwrap().1 {
                best_match = Some((doc, similarity));
            }
            }

            // Only return the best match if its similarity score is above the threshold.
            if let Some((doc, score)) = best_match {
            if score > SIMILARITY_THRESHOLD {
                return Ok(Some((doc.to_string(), score)));
            }
            }

            Ok(None)
        }
    }

    /// TODO: Implement the Cosine Similarity function.
    /// Cosine similarity measures the cosine of the angle between two vectors.
    /// It is calculated as: (A · B) / (||A|| * ||B||)
    /// where:
    /// - A · B is the dot product of vectors A and B.
    /// - ||A|| is the magnitude (or L2 norm) of vector A.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot_product: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }
}

// --- 3. TOOLING ---
// Here we define the tools our agent can use and the logic to execute them.

mod tools {
    use super::*;

    /// TODO: Define a web search tool for the LLM.
    pub fn get_tools_definition() -> Vec<Tool> {
        vec![Tool {
            r#type: "function".to_string(),
            function: Function {
                name: "search_web".to_string(),
                description: "Search the web for up-to-date information.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to use."
                        }
                    },
                    "required": ["query"]
                }),
            },
        }]
    }

    /// Executes a web search using the Serper API.
    async fn search_web(query: &str, api_key: &str) -> Result<String, Box<dyn Error>> {
        let client = reqwest::Client::new();
        let url = "https://google.serper.dev/search";
        let payload = json!({ "q": query });

        let response = client
            .post(url)
            .header("X-API-KEY", api_key)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await?
            .json::<Value>()
            .await?;

        // Extract snippets from the top organic results
        let results = response["organic"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|r| r["snippet"].as_str())
                    .take(5)
                    .collect::<Vec<&str>>()
                    .join("; ")
            })
            .unwrap_or_else(|| "No results found.".to_string());

        Ok(results)
    }

    /// TODO: Parse the tool call from the LLM and execute the corresponding function.
    /// This function acts as a router. When the LLM requests a tool, this function
    /// calls the right Rust code and returns the result.
    pub async fn execute_tool_call(
        tool_call: &ToolCall,
        serp_api_key: &str,
        messages: &mut Vec<Message>,
    ) -> Result<String, Box<dyn Error>> {
        if tool_call.function.name == "search_web" {
            info!("> Tool call: search_web");
            let args: Value = serde_json::from_str(&tool_call.function.arguments)?;
            let query = args["query"].as_str().ok_or("Missing query for search_web")?;
            let search_result = search_web(query, serp_api_key).await?;

            messages.push(Message {
                role: "assistant".to_string(),
                content: format!("<tool call: {} with query: {}>", tool_call.function.name, query),
            });

            messages.push(Message {
                role: "user".to_string(),
                content: format!("<tool result: {}>", search_result),
            });
            Ok(format!("Search results: {}", search_result))
        } else {
            Err(format!("Unknown tool: {}", tool_call.function.name).into())
        }
    }
}

// --- 4. LLM INTERACTION ---
// This function handles the core logic of sending requests to the LLM.

/// TODO: Implement API call to the LLM server with the current conversation history and available tools.
async fn call_llm_api(
    client: &Client,
    llm_url: &str,
    model: &str,
    messages: Vec<Message>,
    use_tools: bool,
) -> Result<ChatCompletion, Box<dyn Error>> {
    let tools = if use_tools {
        tools::get_tools_definition()
    } else {
        vec![]
    };

    let request = ChatRequest {
        model: model.to_string(),
        messages,
        tools: tools,
    };

    let res = client
        .post(format!("{}/v1/chat/completions", llm_url))
        .json(&request)
        .send()
        .await?
        .json::<ChatCompletion>()
        .await?;

    Ok(res)
}

// --- 5. MAIN AGENT LOOP ---
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenv().ok();
    env_logger::init();
    let serp_api_key = env::var("SERP_API_KEY").expect("SERP_API_KEY must be set in .env file");
    let llm_url = "http://localhost:8080";
    let embeddings_url = "http://localhost:8081/embedding";
    let model = "gemma-3-1b-it";

    let client = Client::new();

    // --- Setup RAG Database ---
    let mut db = rag::VectorDB::new(embeddings_url);
    // TODO: Add relevant documents to the vector DB.
    db.add_document("The secret code to access the project is 'quantum_leap_42'.").await?;
    db.add_document("Alice is the lead engineer for the new 'Orion' feature.").await?;
    db.add_document("The project deadline has been moved to next Friday.").await?;
    info!("\nVector DB is ready.");

    // --- Main Loop ---
    println!("\nWelcome! I am a helpful assistant. How can I help you today?");
    println!("(Type 'exit' to quit)\n");

    let mut messages: Vec<Message> = Vec::new();
    loop {
        print!("USER: ");
        io::stdout().flush()?;
        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input)?;
        let user_input = user_input.trim();

        if user_input.eq_ignore_ascii_case("exit") {
            break;
        }

        // --- Let the user choose the mode ---
        println!("Choose mode: (1) RAG (Internal Docs), (2) Tool Search (Web)");
        print!("Mode: ");
        io::stdout().flush()?;
        let mut mode_input = String::new();
        io::stdin().read_line(&mut mode_input)?;
        let mode = mode_input.trim();

        match mode {
            // TODO: Implement RAG
            "1" => {
                info!("> RAG Mode: Searching internal documents...");
                let context = db.find_most_similar(user_input).await?;
                let augmented_prompt = if let Some((ctx, score)) = context {
                    info!("> Found relevant context: \"{}\", with score: {}", ctx, score);
                    format!(
                        "Answer the user's query based *only* on the following context.\n\nContext: \"{}\"\n\nUser query: {}",
                        ctx, user_input
                    )
                } else {
                    info!("> No relevant context found in internal documents for your query.");
                    // We still send the query to the LLM, which might respond that it cannot answer.
                    user_input.to_string()
                };

                let mut rag_messages = messages.clone();
                rag_messages.push(Message { role: "user".to_string(), content: augmented_prompt });

                let response = call_llm_api(&client, llm_url, model, rag_messages, false).await?;
                if let Some(content) = &response.choices[0].message.content {
                    println!("ASSISTANT: {}", content);
                    messages.push(Message { role: "user".to_string(), content: user_input.to_string() });
                    messages.push(Message { role: "assistant".to_string(), content: content.clone() });
                } else {
                    println!("ASSISTANT: <no response>");
                }
            }
            // TODO: Implement tool calling
            "2" => {
                info!("> Tool Mode: Searching the web...");
                let mut tool_messages = messages.clone();
                tool_messages.push(Message { role: "user".to_string(), content: user_input.to_string() });

                let response = call_llm_api(&client, llm_url, model, tool_messages.clone(), true).await?;
                let assistant_message = &response.choices[0].message;

                if let Some(tool_calls) = &assistant_message.tool_calls {
                    for tool_call in tool_calls {
                        let tool_result = tools::execute_tool_call(tool_call, &serp_api_key, &mut tool_messages).await?;
                        info!("> Tool result: {}", tool_result);
                    }

                    let final_response = call_llm_api(&client, llm_url, model, tool_messages, false).await?;
                    if let Some(content) = &final_response.choices[0].message.content {
                        println!("ASSISTANT: {}", content);
                        messages.push(Message { role: "user".to_string(), content: user_input.to_string() });
                        messages.push(Message { role: "assistant".to_string(), content: content.clone() });
                    } else {
                        println!("ASSISTANT: <no response after tool use>");
                    }
                } else if let Some(content) = &assistant_message.content {
                    println!("ASSISTANT: {}", content);
                    messages.push(Message { role: "user".to_string(), content: user_input.to_string() });
                    messages.push(Message { role: "assistant".to_string(), content: content.clone() });
                }
            }
            _ => {
                println!("Invalid mode selected. Please enter '1' for RAG or '2' for Tool Search.");
            }
        }
        println!();
    }

    Ok(())
}
