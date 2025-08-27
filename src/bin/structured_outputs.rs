//! # LLM Workshop: Building an Agent with Tools and RAG
//!
//! Welcome to the Rust LLM workshop! In this tutorial, you'll learn how to build a
//! sophisticated LLM-powered agent in Rust. We'll focus on three key areas:
//!
//! 1.  Retrieval-Augmented Generation (RAG): Giving the LLM access to your private documents.
//! 2.  Tooling: Allowing the LLM to use external tools like a calculator.
//! 3.  Core Logic: Combining these pieces to handle complex, multi-step queries.
//!
//! This file contains a nearly complete application. Your task is to fill in the
//! `// TODO:` sections to complete the agent's functionality.
//!

use dotenvy::dotenv;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::error::Error;
use std::io::{self, Write};
use log::info;

// --- 1. COMMON TYPES (Provided) ---
// These are the data structures for interacting with the LLM API.

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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
}

#[derive(Serialize, Debug, Clone)]
pub struct ResponseFormat {
    pub r#type: String,
    pub json_schema: JsonSchemaWrapper,
}

#[derive(Serialize, Debug, Clone)]
pub struct JsonSchemaWrapper {
    pub name: String,
    pub schema: Value,
}

#[derive(Serialize, Deserialize, Debug)]
struct UserInfo {
    name: String,
    city: String,
    age: u32,
}

// --- 2. RETRIEVAL-AUGMENTED GENERATION (RAG) ---
// Giving our LLM knowledge about private documents.

mod rag {
    use super::*;

    #[derive(Deserialize)]
    struct EmbeddingResponse {
        #[serde(rename = "index")]
        _index: i32,
        embedding: Vec<Vec<f32>>,
    }

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

        pub async fn add_document(&mut self, doc: &str) -> Result<(), Box<dyn Error>> {
            info!("Embedding document: \"{}\"", doc);
            let embedding = self.get_embeddings(doc).await?;
            self.documents.push((doc.to_string(), embedding));
            Ok(())
        }

        pub async fn find_most_similar(&self, query: &str) -> Result<Option<(String, f32)>, Box<dyn Error>> {
            const SIMILARITY_THRESHOLD: f32 = 0.1;

            let query_embedding = self.get_embeddings(query).await?;
            let mut best_match: Option<(&str, f32)> = None;

            // TODO 3: Find the document most similar to the user's query.
            // Iterate through `self.documents`. For each `(doc, doc_embedding)`, calculate the
            // `cosine_similarity` with the `query_embedding`. If the similarity is the best
            // so far, update `best_match`.
            //
            for (doc, doc_embedding) in &self.documents {
                let similarity = cosine_similarity(&query_embedding, doc_embedding);
                if best_match.is_none() || similarity > best_match.unwrap().1 {
                    best_match = Some((doc, similarity));
                }
            }

            if let Some((doc, score)) = best_match {
                if score > SIMILARITY_THRESHOLD {
                    return Ok(Some((doc.to_string(), score)));
                }
            }

            Ok(None)
        }
    }

    /// Calculates the cosine similarity between two vectors.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        // TODO 3: Calculate the cosine similarity of vectors `a` and `b`.
        // The formula is: (A · B) / (||A|| * ||B||)
        //
        // 1.  **Dot Product (A · B)**: This is the sum of the products of corresponding elements.
        //     Formula: Σ(a_i * b_i)
        //
        // 2.  **L2 Norm (Magnitude ||A||)**: This is the square root of the sum of the squares of the elements.
        //     Formula: sqrt(Σ(a_i^2))
        //
        // 3.  **Calculate**: Compute the L2 norm for both `a` and `b`, then divide the dot product
        //     by the product of the two norms.
        //
        // 4.  **Edge Case**: If either norm is 0.0, return 0.0 to prevent division by zero.
        let dot_product: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot_product / (norm_a * norm_b)
    }
}

// --- 3. TOOLING ---
// Giving our LLM the ability to use external functions.

mod tools {
    use super::*;

    /// Defines the tools our LLM can use.
    pub fn get_tools_definition() -> Vec<Tool> {
        // TODO 2: Define a tool named `calculate`.
        // Parameters are defined by a JSON schema, you can use its features, such as enums, descriptions, nested objects.
        //
        // Example structure:
        // Tool {
        //     r#type: "function".to_string(),
        //     function: Function {
        //         name: "calculate".to_string(),
        //         description: "Calculates the sum of two numbers.".to_string(),
        //         parameters: json!({
        //             "type": "object",
        //             "properties": {
        //                 "num1": { "type": "integer", "description": "The first number." },
        //                 "num2": { "type": "integer", "description": "The second number." }
        //             },
        //             "required": ["num1", "num2"]
        //         }),
        //     },
        // },
        vec![
            Tool {
                r#type: "function".to_string(),
                function: Function {
                    name: "calculate".to_string(),
                    description: "Calculates the result of a mathematical operation.".to_string(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "num1": { "type": "integer", "description": "The first number." },
                            "num2": { "type": "integer", "description": "The second number." },
                            "op": {"type": "enum", "enum": ["add", "subtract", "multiply", "divide"], "description": "The operation to perform"}
                        },
                        "required": ["num1", "num2", "op"]
                    }),
                },
            },
        ]
    }
    
    /// Example function that our tool can execute.
    async fn calculate(num1: i32, num2: i32, op: &str) -> Result<i32, Box<dyn Error>> {
        match op {
            "add" => Ok(num1 + num2),
            "subtract" => Ok(num1 - num2),
            "multiply" => Ok(num1 * num2),
            "divide" => {
                if num2 == 0 {
                    Err("Cannot divide by zero".into())
                } else {
                    Ok(num1 / num2)
                }
            },
            _ => Err(format!("Unknown operation: {}", op).into()),
        }
    }

    /// Executes a tool call requested by the LLM.
    pub async fn execute_tool_call(tool_call: &ToolCall) -> Result<String, Box<dyn Error>> {
        if tool_call.function.name == "calculate" {
            info!("> Tool call: calculate");
            let args: Value = serde_json::from_str(&tool_call.function.arguments)?;
            let num1 = args["num1"].as_i64().ok_or("Missing num1 for calculate")? as i32;
            let num2 = args["num2"].as_i64().ok_or("Missing num2 for calculate")? as i32;
            let op = args["op"].as_str().ok_or("Missing op for calculate")?;
            let result = calculate(num1, num2, op).await?;
            Ok(format!("<tool_result>{}</tool_result>", result))
        }
        else {
            Err(format!("Unknown tool: {}", tool_call.function.name).into())
        }
    }
}

// --- 4. LLM INTERACTION ---
/// TODO 1: Implement the API call to the LLM server.
/// This function will send the conversation history and available tools to the LLM.
///
/// Your task:
/// 1.  Create a `ChatRequest` struct.
///     - `model`: Use the `model` parameter.
///     - `messages`: Use the `messages` parameter.
///     - `tools`: If `use_tools` is true, call `tools::get_tools_definition()`. Otherwise, use an empty vector.
/// 2.  Use the `reqwest::Client` to make a POST request.
///     - The URL should be `"{llm_url}/v1/chat/completions"`.
///     - Send the `ChatRequest` as JSON using the `.json()` method.
///     - `await` the response.
/// 3.  Return the `ChatCompletion`.
async fn call_llm_api(
    client: &Client,
    llm_url: &str,
    model: &str,
    messages: Vec<Message>,
    use_tools: bool,
    response_format: Option<ResponseFormat>,
) -> Result<ChatCompletion, Box<dyn Error>> {
    // Your implementation here
    let tools = if use_tools {
        tools::get_tools_definition()
    } else {
        Vec::new()
    };

    info!("> Response format: {:?}", json!(response_format));

    let request = ChatRequest {
        model: model.to_string(),
        messages,
        tools,
        response_format,
    };

    let response = client.post(&format!("{}/v1/chat/completions", llm_url))
        .json(&request)
        .send()
        .await?;

    let chat_completion: ChatCompletion = response.json().await?;
    Ok(chat_completion)
}

// --- 5. AGENT LOGIC ---

/// Handles a simple response from the LLM.
async fn handle_simple_response(client: &Client, llm_url: &str, model: &str, messages: &mut Vec<Message>, user_input: &str) -> Result<(), Box<dyn Error>> {
    info!("> Simple Mode: Generating a response...");
    messages.push(Message { role: "user".to_string(), content: user_input.to_string() });

    let response = call_llm_api(&client, llm_url, model, messages.clone(), false, None).await?;
    if let Some(content) = &response.choices[0].message.content {
        println!("ASSISTANT: {}", content);
        messages.push(Message { role: "assistant".to_string(), content: content.clone() });
    } else {
        println!("ASSISTANT: <no response>");
    }
    Ok(())
}

/// Handles a query using the RAG pipeline.
async fn handle_rag_response(client: &Client, llm_url: &str, model: &str, db: &rag::VectorDB, messages: &mut Vec<Message>, user_input: &str) -> Result<(), Box<dyn Error>> {
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
        user_input.to_string()
    };

    let mut rag_messages = messages.clone();
    rag_messages.push(Message { role: "user".to_string(), content: augmented_prompt });

    let response = call_llm_api(client, llm_url, model, rag_messages, false, None).await?;
    if let Some(content) = &response.choices[0].message.content {
        println!("ASSISTANT: {}", content);
        messages.push(Message { role: "user".to_string(), content: user_input.to_string() });
        messages.push(Message { role: "assistant".to_string(), content: content.clone() });
    } else {
        println!("ASSISTANT: <no response>");
    }
    Ok(())
}

/// Handles a query that may require using tools.
async fn handle_tool_response(client: &Client, llm_url: &str, model: &str, messages: &mut Vec<Message>, user_input: &str) -> Result<(), Box<dyn Error>> {
    info!("> Tool Mode: Engaging agent...");
    let mut tool_messages = messages.clone();
    tool_messages.push(Message { role: "user".to_string(), content: user_input.to_string() });

    // First, call the LLM to see if it wants to use a tool.
    let response = call_llm_api(client, llm_url, model, tool_messages.clone(), true, None).await?;
    let assistant_message = &response.choices[0].message;

    if let Some(tool_calls) = &assistant_message.tool_calls {
        // The LLM wants to use a tool.
        tool_messages.push(Message { role: "assistant".to_string(), content: serde_json::to_string(&tool_calls)? });

        for tool_call in tool_calls {
            let tool_result = tools::execute_tool_call(tool_call).await?;
            info!("> Tool result: {}", tool_result);
            tool_messages.push(Message { role: "user".to_string(), content: tool_result });
        }

        // Now, call the LLM again with the tool results to get a final answer.
        let final_response = call_llm_api(client, llm_url, model, tool_messages, false, None).await?;
        if let Some(content) = &final_response.choices[0].message.content {
            println!("ASSISTANT: {}", content);
            messages.push(Message { role: "user".to_string(), content: user_input.to_string() });
            messages.push(Message { role: "assistant".to_string(), content: content.clone() });
        } else {
            println!("ASSISTANT: <no response after tool use>");
        }
    } else if let Some(content) = &assistant_message.content {
        // The LLM answered directly without using a tool.
        println!("ASSISTANT: {}", content);
        messages.push(Message { role: "user".to_string(), content: user_input.to_string() });
        messages.push(Message { role: "assistant".to_string(), content: content.clone() });
    }
    Ok(())
}

/// Handles a query by extracting structured data.
async fn handle_structured_output_response(client: &Client, llm_url: &str, model: &str, messages: &mut Vec<Message>, user_input: &str) -> Result<(), Box<dyn Error>> {
    info!("> Structured Output Mode: Extracting information...");

    let structured_prompt = format!(
        "From the following text, extract the requested info.\n\nText: \"{}\"",
        user_input
    );

    let structured_messages = vec![Message { role: "user".to_string(), content: structured_prompt }];
    let response_format = Some(ResponseFormat {
        r#type: "json_schema".to_string(),
        json_schema: JsonSchemaWrapper {
            name: "UserInfo".to_string(),
            schema: json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "city": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "city", "age"]
            })
        }
    });

    let response = call_llm_api(client, llm_url, model, structured_messages, false, response_format).await?;

    if let Some(content) = &response.choices[0].message.content {
        info!("> LLM raw response: {}", content);
        
        match serde_json::from_str::<UserInfo>(content) {
            Ok(user_info) => {
                println!("ASSISTANT: I've extracted the following information:");
                println!("  - Name: {}", user_info.name);
                println!("  - City: {}", user_info.city);
                println!("  - Age: {}", user_info.age);
                
                messages.push(Message { role: "user".to_string(), content: user_input.to_string() });
                let assistant_summary = format!("Extracted user info: Name: {}, City: {}, Age: {}", user_info.name, user_info.city, user_info.age);
                messages.push(Message { role: "assistant".to_string(), content: assistant_summary });
            }
            Err(e) => {
                println!("ASSISTANT: I couldn't extract the information in the required format. Here is the raw response:\n{}", content);
                eprintln!("Failed to parse JSON: {}", e);
            }
        }
    } else {
        println!("ASSISTANT: <no response>");
    }
    Ok(())
}


// --- 6. MAIN LOOP ---
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenv().ok();
    env_logger::init();
    let llm_url = "http://localhost:8080";
    let embeddings_url = "http://localhost:8081/embedding";
    let model = "gemma-3-1b-it";

    let client = Client::new();

    // --- Setup RAG Database ---
    let mut db = rag::VectorDB::new(embeddings_url);
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

        println!("Choose mode: (1) Simple response, (2) Tool Use, (3) RAG, (4) Structured Output");
        print!("Mode: ");
        io::stdout().flush()?;
        let mut mode_input = String::new();
        io::stdin().read_line(&mut mode_input)?;
        let mode = mode_input.trim();

        let result = match mode {
            "1" => handle_simple_response(&client, llm_url, model, &mut messages, user_input).await,
            "2" => handle_tool_response(&client, llm_url, model, &mut messages, user_input).await,
            "3" => handle_rag_response(&client, llm_url, model, &db, &mut messages, user_input).await,
            "4" => handle_structured_output_response(&client, llm_url, model, &mut messages, user_input).await,
            _ => {
                println!("Invalid mode selected. Please enter '1', '2', '3', or '4'.");
                Ok(())
            }
        };

        if let Err(e) = result {
            eprintln!("An error occurred: {}", e);
        }

        println!();
    }

    Ok(())
}
