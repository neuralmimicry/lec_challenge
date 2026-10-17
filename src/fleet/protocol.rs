//! Text-only wire contracts. Fixtures test these contracts without buying tokens.
use super::*;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderApi {
    #[default]
    ChatCompletions,
    OpenaiResponses,
    GeminiGenerateContent,
    VertexGenerateContent,
    AnthropicMessages,
}

impl ProviderApi {
    pub(super) fn validate_endpoint(
        self,
        url: &reqwest::Url,
        model: &str,
        cap: &str,
    ) -> Result<()> {
        let valid = match self {
            Self::ChatCompletions => matches!(cap, "max_tokens" | "max_completion_tokens"),
            Self::OpenaiResponses => {
                cap == "max_output_tokens" && url.path().ends_with("/responses")
            }
            Self::AnthropicMessages => cap == "max_tokens" && url.path().ends_with("/messages"),
            Self::GeminiGenerateContent | Self::VertexGenerateContent => {
                cap == "maxOutputTokens"
                    && !model.is_empty()
                    && model
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b"-._".contains(&b))
                    && url
                        .path()
                        .ends_with(&format!("/models/{model}:generateContent"))
            }
        };
        ensure!(
            valid,
            "API protocol, endpoint, model and output-cap parameter disagree"
        );
        Ok(())
    }

    pub(super) fn request_body(
        self,
        model: &str,
        cap: &str,
        request: &FleetRequest,
    ) -> Result<Value> {
        let g = &request.generation;
        let body = match self {
            Self::ChatCompletions => chat_body(model, cap, request),
            Self::OpenaiResponses => {
                ensure!(
                    g.seed.is_none() && g.temperature == 1.0 && g.top_p == 1.0,
                    "Codex Responses requires seed absent and neutral temperature/top_p (1.0); sampling controls are omitted"
                );
                json!({"model":model,"instructions":request.system_prompt,"input":request.prompt,
                    "max_output_tokens":g.max_output_tokens,"reasoning":{"effort":"low"},"store":false,"stream":false})
            }
            Self::AnthropicMessages => {
                ensure!(
                    g.seed.is_none() && g.top_p == 1.0 && g.temperature <= 1.0,
                    "Messages requires seed absent, top_p=1.0 and temperature at most 1.0"
                );
                json!({"model":model,"system":request.system_prompt,
                    "messages":[{"role":"user","content":request.prompt}],
                    "max_tokens":g.max_output_tokens,"temperature":g.temperature,"stream":false})
            }
            Self::GeminiGenerateContent | Self::VertexGenerateContent => {
                ensure!(
                    g.seed.is_none_or(|seed| seed <= i32::MAX as u64),
                    "Gemini seed exceeds int32"
                );
                // This bounded adapter supports models that accept thinkingBudget=0.
                // Automatic/dynamic thinking and billable tools are not requested.
                let mut body = json!({"systemInstruction":{"parts":[{"text":request.system_prompt}]},
                    "contents":[{"role":"user","parts":[{"text":request.prompt}]}],
                    "generationConfig":{"candidateCount":1,"maxOutputTokens":g.max_output_tokens,
                        "temperature":g.temperature,"topP":g.top_p,"thinkingConfig":{"thinkingBudget":0}}});
                if let Some(seed) = g.seed {
                    body["generationConfig"]["seed"] = json!(seed);
                }
                body
            }
        };
        Ok(body)
    }
}

fn required(value: &Value, key: &str) -> Result<u64> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .context("missing or malformed required usage count")
}
fn optional(value: &Value, pointer: &str) -> Result<u64> {
    match value.pointer(pointer) {
        None => Ok(0),
        Some(value) => value.as_u64().context("malformed optional usage count"),
    }
}

pub fn extract_usage(response: &TransportResponse) -> Result<TokenUsage> {
    let body: Value =
        serde_json::from_str(&response.raw_body).context("malformed provider JSON")?;
    let usage = if let Some(u) = body.get("usageMetadata") {
        ensure!(body.get("usage").is_none(), "ambiguous usage schemas");
        ensure!(
            optional(u, "/toolUsePromptTokenCount")? == 0,
            "billable tools prohibited"
        );
        let reasoning = optional(u, "/thoughtsTokenCount")?;
        TokenUsage {
            input_tokens: required(u, "promptTokenCount")?,
            cached_input_tokens: optional(u, "/cachedContentTokenCount")?,
            output_tokens: required(u, "candidatesTokenCount")?
                .checked_add(reasoning)
                .context("token overflow")?,
            reasoning_tokens: reasoning,
        }
    } else {
        let u = body
            .get("usage")
            .context("provider response lacks usage; cost is uncertain")?;
        if body.get("object").and_then(Value::as_str) == Some("response") {
            TokenUsage {
                input_tokens: required(u, "input_tokens")?,
                cached_input_tokens: optional(u, "/input_tokens_details/cached_tokens")?,
                output_tokens: required(u, "output_tokens")?,
                reasoning_tokens: optional(u, "/output_tokens_details/reasoning_tokens")?,
            }
        } else if body.get("type").and_then(Value::as_str) == Some("message") {
            ensure!(
                optional(u, "/cache_creation_input_tokens")? == 0,
                "unexpected cache creation needs pricing reconciliation"
            );
            let cached = optional(u, "/cache_read_input_tokens")?;
            TokenUsage {
                input_tokens: required(u, "input_tokens")?
                    .checked_add(cached)
                    .context("token overflow")?,
                cached_input_tokens: cached,
                output_tokens: required(u, "output_tokens")?,
                reasoning_tokens: 0,
            }
        } else {
            let cached = optional(u, "/prompt_tokens_details/cached_tokens")?;
            let legacy = optional(u, "/prompt_cache_hit_tokens")?;
            ensure!(
                u.pointer("/prompt_tokens_details/cached_tokens").is_none()
                    || u.get("prompt_cache_hit_tokens").is_none()
                    || cached == legacy,
                "cached usage fields disagree"
            );
            TokenUsage {
                input_tokens: required(u, "prompt_tokens")?,
                cached_input_tokens: cached.max(legacy),
                output_tokens: required(u, "completion_tokens")?,
                reasoning_tokens: optional(u, "/completion_tokens_details/reasoning_tokens")?,
            }
        }
    };
    ensure!(
        usage.cached_input_tokens <= usage.input_tokens
            && usage.reasoning_tokens <= usage.output_tokens,
        "inconsistent usage fields"
    );
    Ok(usage)
}

fn text_parts(parts: &Value, native_type: Option<&str>) -> Result<String> {
    let parts = parts.as_array().context("missing response content parts")?;
    let mut text = String::new();
    for part in parts {
        if let Some(kind) = native_type {
            ensure!(
                part.get("type").and_then(Value::as_str) == Some(kind),
                "non-text or refused content prohibited"
            );
        } else {
            ensure!(
                part.as_object().is_some_and(|p| p
                    .keys()
                    .all(|k| matches!(k.as_str(), "text" | "thought" | "thoughtSignature"))),
                "non-text Gemini content prohibited"
            );
            ensure!(
                !part
                    .get("thought")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                "unexpected thinking content"
            );
        }
        text.push_str(
            part.get("text")
                .and_then(Value::as_str)
                .context("missing text content")?,
        );
    }
    Ok(text)
}

pub fn parse_recording(record: &Recording) -> Result<(String, TokenUsage)> {
    if (200..300).contains(&record.response.status) {
        let body: Value =
            serde_json::from_str(&record.response.raw_body).context("malformed provider JSON")?;
        let matches = match record.api {
            ProviderApi::ChatCompletions => body.get("choices").is_some(),
            ProviderApi::OpenaiResponses => {
                body.get("object").and_then(Value::as_str) == Some("response")
            }
            ProviderApi::AnthropicMessages => {
                body.get("type").and_then(Value::as_str) == Some("message")
            }
            ProviderApi::GeminiGenerateContent | ProviderApi::VertexGenerateContent => {
                body.get("candidates").is_some()
            }
        };
        ensure!(
            matches,
            "response schema differs from the recorded API protocol"
        );
    }
    parse_response(&record.response)
}

pub fn parse_response(response: &TransportResponse) -> Result<(String, TokenUsage)> {
    ensure!(
        (200..300).contains(&response.status),
        "provider HTTP {}",
        response.status
    );
    let body: Value =
        serde_json::from_str(&response.raw_body).context("malformed provider JSON")?;
    ensure!(
        body.get("error").is_none_or(Value::is_null),
        "provider reported an error"
    );
    let text = if body.get("candidates").is_some() {
        ensure!(
            body.pointer("/promptFeedback/blockReason").is_none(),
            "Gemini prompt blocked"
        );
        let candidates = body["candidates"]
            .as_array()
            .context("missing candidates")?;
        ensure!(
            candidates.len() == 1 && candidates[0]["finishReason"] == "STOP",
            "blocked, truncated or multiple Gemini responses"
        );
        text_parts(&candidates[0]["content"]["parts"], None)?
    } else if body.get("type").and_then(Value::as_str) == Some("message") {
        ensure!(
            body["role"] == "assistant" && body["stop_reason"] == "end_turn",
            "non-complete Messages response"
        );
        text_parts(&body["content"], Some("text"))?
    } else if body.get("output").is_some() {
        ensure!(
            body["status"] == "completed"
                && body.get("incomplete_details").is_none_or(Value::is_null),
            "incomplete Responses generation"
        );
        let mut text = String::new();
        for item in body["output"]
            .as_array()
            .context("missing Responses output")?
        {
            match item["type"].as_str() {
                Some("reasoning") => {}
                Some("message") => {
                    ensure!(
                        item["role"] == "assistant" && item["status"] == "completed",
                        "incomplete output message"
                    );
                    text.push_str(&text_parts(&item["content"], Some("output_text"))?);
                }
                _ => bail!("tool or unsupported Responses output prohibited"),
            }
        }
        text
    } else {
        let choices = body["choices"].as_array().context("missing choices")?;
        ensure!(
            choices.len() == 1 && choices[0]["finish_reason"] == "stop",
            "truncated or non-stop generation"
        );
        ensure!(
            choices[0]
                .pointer("/message/tool_calls")
                .is_none_or(Value::is_null)
                && choices[0]
                    .pointer("/message/function_call")
                    .is_none_or(Value::is_null),
            "tool completion prohibited"
        );
        choices[0]["message"]["content"]
            .as_str()
            .context("completion content must be text")?
            .to_owned()
    };
    let text = text.trim();
    let code = if text.starts_with("```") && text.ends_with("```") {
        text.find('\n')
            .map(|n| text[n + 1..text.len() - 3].trim())
            .unwrap_or(text)
    } else {
        text
    };
    ensure!(!code.trim().is_empty(), "empty completion is failure");
    Ok((code.to_owned(), extract_usage(response)?))
}
