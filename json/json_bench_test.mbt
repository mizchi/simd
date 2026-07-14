///|
let mixed_item =
  #|{"id":123456,"name":"item-123456","price":123456.125,"ratio":457e-3,"active":true,"tags":["alpha","beta","gamma"],"meta":{"count":370368,"note":null}}

///|
fn make_mixed_json(count : Int) -> String {
  let buf = StringBuilder(size_hint=count * (mixed_item.length() + 1))
  buf.write_char('[')
  for i in 0..<count {
    if i > 0 {
      buf.write_char(',')
    }
    buf.write_string(mixed_item)
  }
  buf.write_char(']')
  buf.to_string()
}

///|
fn make_plain_strings_json(count : Int, width : Int) -> String {
  let item = String::make(width, 'a')
  let buf = StringBuilder(size_hint=count * (width + 3))
  buf.write_char('[')
  for i in 0..<count {
    if i > 0 {
      buf.write_char(',')
    }
    buf.write_char('"')
    buf.write_string(item)
    buf.write_char('"')
  }
  buf.write_char(']')
  buf.to_string()
}

///|
let escaped_item =
  #|escaped\ntext\twith\u0041quotes\"and\\slashes/escaped\ntext\twith\u0042quotes\"and\\slashes

///|
fn make_escaped_strings_json(count : Int) -> String {
  let buf = StringBuilder(size_hint=count * (escaped_item.length() + 3))
  buf.write_char('[')
  for i in 0..<count {
    if i > 0 {
      buf.write_char(',')
    }
    buf.write_char('"')
    buf.write_string(escaped_item)
    buf.write_char('"')
  }
  buf.write_char(']')
  buf.to_string()
}

///|
fn make_number_json(count : Int) -> String {
  let buf = StringBuilder(size_hint=count * 13)
  buf.write_char('[')
  for i in 0..<count {
    if i > 0 {
      buf.write_char(',')
    }
    buf.write_string((i * 2654435).to_string())
    if (i & 1) == 0 {
      buf.write_string(".125e-2")
    }
  }
  buf.write_char(']')
  buf.to_string()
}

///|
fn bench_pair(b : @bench.T, name : String, input : String) -> Unit {
  let bytes = @utf8.encode(input[:])
  b.bench(name="core_json_\{name}", fn() {
    b.keep(try! @core_json.parse(input))
  })
  b.bench(name="simd_json_bytes_\{name}", fn() { b.keep(try! parse(bytes)) })
  b.bench(name="simd_json_string_\{name}", fn() {
    b.keep(try! parse_string(input[:]))
  })
}

///|
test "bench/large JSON parser comparison" (b : @bench.T) {
  bench_pair(b, "mixed_9_9mb", make_mixed_json(65536))
  bench_pair(b, "plain_strings_8_4mb", make_plain_strings_json(2048, 4096))
  bench_pair(b, "escaped_strings_12_3mb", make_escaped_strings_json(131072))
  bench_pair(b, "numbers_7_6mb", make_number_json(524288))
}
