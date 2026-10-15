pub fn solve(input: serde_json::Value) -> serde_json::Value {
    let points = input.as_array().expect("point array");
    if points.is_empty() {
        return serde_json::Value::Null;
    }
    let xs: Vec<i64> = points.iter().map(|p| p[0].as_i64().expect("x")).collect();
    let ys: Vec<i64> = points.iter().map(|p| p[1].as_i64().expect("y")).collect();
    serde_json::json!([
        xs.iter().min(),
        ys.iter().min(),
        xs.iter().max(),
        ys.iter().max()
    ])
}
