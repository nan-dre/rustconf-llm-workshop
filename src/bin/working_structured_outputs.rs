use serde::{Deserialize, Serialize};
use serde_json::json;
use reqwest::Client;
use std::error::Error;

// Structs for serializing the request payload
#[derive(Debug, Serialize)]
struct RequestPayload<'a> {
    model: &'a str,
    messages: Vec<Message<'a>>,
    response_format: ResponseFormat<'a>,
}

#[derive(Debug, Serialize)]
struct Message<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Serialize)]
struct ResponseFormat<'a> {
    #[serde(rename = "type")]
    format_type: &'a str,
    json_schema: JsonSchemaWrapper<'a>,
}

#[derive(Debug, Serialize)]
struct JsonSchemaWrapper<'a> {
    name: &'a str,
    schema: serde_json::Value,
}

// Structs for deserializing the response from the Llama server
#[derive(Debug, Deserialize, Serialize)]
struct LlamaResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Debug, Deserialize, Serialize)]
struct ResponseMessage {
    content: String, // This will contain the JSON string
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // llama.cpp server endpoint (adjust host/port if needed)
    let url = "http://localhost:8080/v1/chat/completions";

    // JSON schema definition using the serde_json::json! macro
    let json_schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "city": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "city", "age"]
    });

    // Request payload
    let payload = RequestPayload {
        model: "your-model-name-here", // replace with model name in server
        messages: vec![
            Message {
                role: "system",
                content: "Extract the person's details as JSON.",
            },
            Message {
                role: "user",
                content: "In the bustling city of Cindralune, names were more than words—they were people. Among them lived Maris, a name of thirty-two years. Maris spent her mornings drifting between street signs and billboards, where younger names jostled for attention with their bright syllables and trendy spellings. She was no longer new, no longer whispered in fresh love letters, but she carried a weight of history—echoes of family dinners, childhood laughter, and tear-stained confessions.",
            },
        ],
        response_format: ResponseFormat {
            format_type: "json_schema",
            json_schema: JsonSchemaWrapper {
                name: "person_info",
                schema: json_schema,
            },
        },
    };

    // Create a reqwest client
    let client = Client::new();

    // Send request
    let response = client.post(url).json(&payload).send().await?;

    // Parse response
    let data: LlamaResponse = response.json().await?;
    println!("{}", serde_json::to_string_pretty(&data)?);


    // The actual structured data is a JSON string within the 'content' field.
    // We need to parse it again from the string.
    if let Some(choice) = data.choices.get(0) {
        println!("\nExtracted JSON: {}", &choice.message.content);
    }

    Ok(())
}