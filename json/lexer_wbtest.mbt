///|
fn make_dense_token_input(count : Int) -> Bytes {
  let buf = StringBuilder(size_hint=count * 2 + 2)
  buf.write_char('[')
  for i in 0..<count {
    if i > 0 {
      buf.write_char(',')
    }
    buf.write_char('0')
  }
  buf.write_char(']')
  @utf8.encode(buf.to_string()[:])
}

///|
fn make_sparse_token_input(width : Int) -> Bytes {
  let buf = StringBuilder(size_hint=width + 4)
  buf.write_string("[\"")
  for _ in 0..<width {
    buf.write_char('a')
  }
  buf.write_string("\"]")
  @utf8.encode(buf.to_string()[:])
}

///|
test "token capacity hint separates dense and sparse inputs" {
  let dense = make_dense_token_input(4096)
  let sparse = make_sparse_token_input(8192)
  let dense_hint = json_token_capacity_hint(dense)
  let sparse_hint = json_token_capacity_hint(sparse)
  @debug.assert_eq(dense_hint > dense.length() / 2, true)
  @debug.assert_eq(sparse_hint < sparse.length() / 16, true)
  @debug.assert_eq(
    json_scalar_token_starts_utf8(dense).length() <= dense_hint,
    true,
  )
  @debug.assert_eq(
    json_scalar_token_starts_utf8(sparse).length() <= sparse_hint,
    true,
  )
}

///|
fn make_dense_string_input(count : Int) -> String {
  let buf = StringBuilder(size_hint=count * 2 + 2)
  buf.write_char('[')
  for i in 0..<count {
    if i > 0 {
      buf.write_char(',')
    }
    buf.write_char('0')
  }
  buf.write_char(']')
  buf.to_string()
}

///|
test "StringView routing keeps sparse input on core and dense input indexed" {
  let dense = make_dense_string_input(40000)
  let sparse = "[\"" + String::make(70000, 'a') + "\"]"
  @debug.assert_eq(json_string_should_use_core("[0]"[:]), true)
  @debug.assert_eq(json_string_should_use_core(sparse[:]), true)
  @debug.assert_eq(json_string_should_use_core(dense[:]), false)
}
