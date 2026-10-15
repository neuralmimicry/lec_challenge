/// Sum the bounded input integers. The wrapper owns all input/output handling.
pub fn solve(input: serde_json::Value) -> serde_json::Value {
    serde_json::json!(
        input
            .as_array()
            .expect("array input")
            .iter()
            .map(|value| value.as_i64().expect("integer input"))
            .sum::<i64>()
    )
}
