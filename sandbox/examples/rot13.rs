pub fn solve(input: serde_json::Value) -> serde_json::Value {
    let transformed: String = input
        .as_str()
        .expect("ASCII text")
        .bytes()
        .map(|byte| match byte {
            b'a'..=b'z' => ((byte - b'a' + 13) % 26 + b'a') as char,
            b'A'..=b'Z' => ((byte - b'A' + 13) % 26 + b'A') as char,
            _ => byte as char,
        })
        .collect();
    serde_json::json!(transformed)
}
