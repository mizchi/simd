// Correctness tests for the simdjson byte-classification pipeline.
// All four targets share the same suite; the wasm / wasm-gc files
// dispatch to inline-WAT, the native / js files to scalar — identical
// public API surface.

///|
/// Build a `SimdBufferBytes` from a literal ASCII string (1 byte/char).
fn bytes_of_ascii(s : String) -> @simd_buffer.SimdBufferBytes {
  let n = s.length()
  let arr : FixedArray[Byte] = FixedArray::make(n, b'\x00')
  for i in 0..<n {
    arr[i] = s.unsafe_get(i).to_int().to_byte()
  }
  @simd_buffer.SimdBufferBytes::from_array(arr)
}

///|
/// Reference: bitmask of structural-char positions in `s`.
fn expected_struct_bits(s : String) -> FixedArray[Int] {
  let n = s.length()
  let words = (n + 31) / 32
  let out : FixedArray[Int] = FixedArray::make(words, 0)
  for i in 0..<n {
    let c = s.unsafe_get(i).to_int()
    if c == 0x7B ||
      c == 0x7D ||
      c == 0x5B ||
      c == 0x5D ||
      c == 0x2C ||
      c == 0x3A {
      out[i >> 5] = out[i >> 5] | (1 << (i & 31))
    }
  }
  out
}

///|
/// Reference: bitmask of byte positions where `s[i] == needle` (raw, no
/// escape processing).
fn expected_byte_bits(s : String, needle : Int) -> FixedArray[Int] {
  let n = s.length()
  let words = (n + 31) / 32
  let out : FixedArray[Int] = FixedArray::make(words, 0)
  for i in 0..<n {
    if s.unsafe_get(i).to_int() == needle {
      out[i >> 5] = out[i >> 5] | (1 << (i & 31))
    }
  }
  out
}

///|
/// Reference: bitmask of JSON-numeric-token character positions.
fn expected_numeric_bits(s : String) -> FixedArray[Int] {
  let n = s.length()
  let words = (n + 31) / 32
  let out : FixedArray[Int] = FixedArray::make(words, 0)
  for i in 0..<n {
    let c = s.unsafe_get(i).to_int()
    let is_digit = c >= 0x30 && c <= 0x39
    if is_digit || c == 0x2D || c == 0x2E || c == 0x2B || c == 0x65 || c == 0x45 {
      out[i >> 5] = out[i >> 5] | (1 << (i & 31))
    }
  }
  out
}

///|
/// Reference in-string mask.
fn expected_quote_bits(s : String) -> FixedArray[Int] {
  let n = s.length()
  let words = (n + 31) / 32
  let out : FixedArray[Int] = FixedArray::make(words, 0)
  let mut in_string = false
  let mut escape_pending = false
  for i in 0..<n {
    let b = s.unsafe_get(i).to_int()
    let mut bit_set = false
    if escape_pending {
      escape_pending = false
      bit_set = in_string
    } else if b == 0x22 {
      in_string = !in_string
    } else if b == 0x5C && in_string {
      escape_pending = true
      bit_set = true
    } else {
      bit_set = in_string
    }
    if bit_set {
      out[i >> 5] = out[i >> 5] | (1 << (i & 31))
    }
  }
  out
}

///|
/// Reference: byte offsets of real structural chars (outside strings).
fn expected_structural_indices(s : String) -> FixedArray[Int] {
  let n = s.length()
  let struct_bits = expected_struct_bits(s)
  let quote_bits = expected_quote_bits(s)
  let out : FixedArray[Int] = FixedArray::make(n, 0)
  let mut count = 0
  for i in 0..<n {
    let w = i >> 5
    let b = i & 31
    if (struct_bits[w] & (1 << b)) != 0 && (quote_bits[w] & (1 << b)) == 0 {
      out[count] = i
      count = count + 1
    }
  }
  let trimmed = FixedArray::make(count, 0)
  for k in 0..<count {
    trimmed[k] = out[k]
  }
  trimmed
}

///|
test "classify_structural — short JSON, scalar tail only" {
  let s = "{\"a\":1}"
  let buf = bytes_of_ascii(s)
  let out = @simd_buffer.SimdBuffer::make((s.length() + 31) / 32)
  out.clear()
  classify_structural(buf, out)
  let want = expected_struct_bits(s)
  for w in 0..<want.length() {
    if out.get(w) != want[w] {
      abort(
        "structural mismatch at word " +
        w.to_string() +
        ": want=" +
        want[w].to_string() +
        " got=" +
        out.get(w).to_string(),
      )
    }
  }
}

///|
test "classify_structural — 16-byte aligned, SIMD path only" {
  let s = "[1,2,3,{\"x\":4}]!"
  let buf = bytes_of_ascii(s)
  let out = @simd_buffer.SimdBuffer::make((s.length() + 31) / 32)
  out.clear()
  classify_structural(buf, out)
  let want = expected_struct_bits(s)
  for w in 0..<want.length() {
    if out.get(w) != want[w] {
      abort("structural mismatch at word " + w.to_string())
    }
  }
}

///|
test "classify_structural — large JSON, SIMD + tail" {
  let raw = "{\"k\":[1,2,3,4,5,6,7,8],\"n\":{\"deep\":\"v\"}},"
  let buf_w = StringBuilder::new()
  for _ in 0..<6 {
    buf_w.write_string(raw)
  }
  buf_w.write_string("END")
  let s = buf_w.to_string()
  let buf = bytes_of_ascii(s)
  let out = @simd_buffer.SimdBuffer::make((s.length() + 31) / 32)
  out.clear()
  classify_structural(buf, out)
  let want = expected_struct_bits(s)
  for w in 0..<want.length() {
    if out.get(w) != want[w] {
      abort(
        "large structural mismatch at word " +
        w.to_string() +
        ": want=" +
        want[w].to_string() +
        " got=" +
        out.get(w).to_string(),
      )
    }
  }
}

///|
test "classify_quote_raw — escaped quotes still appear (raw)" {
  let s = "{\"a\\\"b\":1}"
  let buf = bytes_of_ascii(s)
  let out = @simd_buffer.SimdBuffer::make((s.length() + 31) / 32)
  out.clear()
  classify_quote_raw(buf, out)
  let want = expected_byte_bits(s, 0x22)
  for w in 0..<want.length() {
    if out.get(w) != want[w] {
      abort("quote_raw mismatch at word " + w.to_string())
    }
  }
}

///|
test "classify_backslash — locates escape bytes" {
  let s = "{\"path\":\"C:\\\\a\\nb\"}"
  let buf = bytes_of_ascii(s)
  let out = @simd_buffer.SimdBuffer::make((s.length() + 31) / 32)
  out.clear()
  classify_backslash(buf, out)
  let want = expected_byte_bits(s, 0x5C)
  for w in 0..<want.length() {
    if out.get(w) != want[w] {
      abort("backslash mismatch at word " + w.to_string())
    }
  }
}

///|
test "classify_structural — empty buffer is a no-op" {
  let buf = @simd_buffer.SimdBufferBytes::make(0)
  let out = @simd_buffer.SimdBuffer::make(0)
  classify_structural(buf, out)
}

///|
test "classify_numeric — short input, scalar tail only" {
  let s = "-1.5e+10"
  let buf = bytes_of_ascii(s)
  let out = @simd_buffer.SimdBuffer::make((s.length() + 31) / 32)
  out.clear()
  classify_numeric(buf, out)
  let want = expected_numeric_bits(s)
  for w in 0..<want.length() {
    if out.get(w) != want[w] {
      abort(
        "numeric mismatch at word " +
        w.to_string() +
        ": want=" +
        want[w].to_string() +
        " got=" +
        out.get(w).to_string(),
      )
    }
  }
}

///|
test "classify_numeric — mixed JSON, SIMD + tail" {
  let raw = "{\"a\":123,\"b\":-4.56e+7,\"c\":[0,1,2,3,4,5,6,7,8,9],\"x\":1.0E-3}"
  let buf_w = StringBuilder::new()
  for _ in 0..<5 {
    buf_w.write_string(raw)
  }
  buf_w.write_string("end")
  let s = buf_w.to_string()
  let buf = bytes_of_ascii(s)
  let out = @simd_buffer.SimdBuffer::make((s.length() + 31) / 32)
  out.clear()
  classify_numeric(buf, out)
  let want = expected_numeric_bits(s)
  for w in 0..<want.length() {
    if out.get(w) != want[w] {
      abort(
        "numeric mismatch at word " +
        w.to_string() +
        ": want=" +
        want[w].to_string() +
        " got=" +
        out.get(w).to_string(),
      )
    }
  }
}

///|
test "classify_numeric — does not flag letters other than e/E" {
  let s = "abcDEFG1234e"
  let buf = bytes_of_ascii(s)
  let out = @simd_buffer.SimdBuffer::make((s.length() + 31) / 32)
  out.clear()
  classify_numeric(buf, out)
  let want = expected_numeric_bits(s)
  if out.get(0) != want[0] {
    abort(
      "letter-vs-digit mismatch: want=" +
      want[0].to_string() +
      " got=" +
      out.get(0).to_string(),
    )
  }
}

///|
test "compute_quote_mask — simple string" {
  let s = "{\"hello\":42}"
  let buf = bytes_of_ascii(s)
  let out = @simd_buffer.SimdBuffer::make((s.length() + 31) / 32)
  compute_quote_mask(buf, out)
  let want = expected_quote_bits(s)
  for w in 0..<want.length() {
    if out.get(w) != want[w] {
      abort(
        "quote_mask mismatch at word " +
        w.to_string() +
        ": want=" +
        want[w].to_string() +
        " got=" +
        out.get(w).to_string(),
      )
    }
  }
}

///|
test "compute_quote_mask — escaped quote stays inside string" {
  let s = "{\"a\\\"b\":1}"
  let buf = bytes_of_ascii(s)
  let out = @simd_buffer.SimdBuffer::make((s.length() + 31) / 32)
  compute_quote_mask(buf, out)
  let want = expected_quote_bits(s)
  for w in 0..<want.length() {
    if out.get(w) != want[w] {
      abort("escaped quote mismatch at word " + w.to_string())
    }
  }
}

///|
test "compute_quote_mask — backslash before backslash before quote" {
  let s = "{\"a\\\\\":1}"
  let buf = bytes_of_ascii(s)
  let out = @simd_buffer.SimdBuffer::make((s.length() + 31) / 32)
  compute_quote_mask(buf, out)
  let want = expected_quote_bits(s)
  for w in 0..<want.length() {
    if out.get(w) != want[w] {
      abort("double-backslash mismatch at word " + w.to_string())
    }
  }
}

///|
test "compute_quote_mask — large mixed JSON with multiple strings" {
  let s = "{\"name\":\"Alice\",\"path\":\"C:\\\\Users\\\\a\",\"q\":\"with \\\"quotes\\\" inside\",\"x\":[1,2,3]}"
  let buf = bytes_of_ascii(s)
  let out = @simd_buffer.SimdBuffer::make((s.length() + 31) / 32)
  compute_quote_mask(buf, out)
  let want = expected_quote_bits(s)
  for w in 0..<want.length() {
    if out.get(w) != want[w] {
      abort(
        "mixed-string mismatch at word " +
        w.to_string() +
        ": want=" +
        want[w].to_string() +
        " got=" +
        out.get(w).to_string(),
      )
    }
  }
}

///|
test "compute_quote_mask — input crossing many 32-byte words" {
  let raw = "{\"k\":\"a\\\"b\",\"l\":\"plain\",\"m\":\"\",\"n\":\"long string with \\\\ inside\"}"
  let buf_w = StringBuilder::new()
  for _ in 0..<7 {
    buf_w.write_string(raw)
  }
  let s = buf_w.to_string()
  let buf = bytes_of_ascii(s)
  let out = @simd_buffer.SimdBuffer::make((s.length() + 31) / 32)
  compute_quote_mask(buf, out)
  let want = expected_quote_bits(s)
  for w in 0..<want.length() {
    if out.get(w) != want[w] {
      abort(
        "long-input quote_mask mismatch at word " +
        w.to_string() +
        ": want=" +
        want[w].to_string() +
        " got=" +
        out.get(w).to_string(),
      )
    }
  }
}

///|
test "find_structural_indices — small JSON" {
  let s = "{\"a\":1,\"b\":[2,3]}"
  let buf = bytes_of_ascii(s)
  let indices = @simd_buffer.SimdBuffer::make(s.length())
  let count = find_structural_indices(buf, indices)
  let want = expected_structural_indices(s)
  if count != want.length() {
    abort(
      "count mismatch: want=" +
      want.length().to_string() +
      " got=" +
      count.to_string(),
    )
  }
  for k in 0..<count {
    if indices.get(k) != want[k] {
      abort(
        "index " +
        k.to_string() +
        " mismatch: want=" +
        want[k].to_string() +
        " got=" +
        indices.get(k).to_string(),
      )
    }
  }
}

///|
test "find_structural_indices — structural chars inside strings are ignored" {
  let s = "{\"key with [bracket]\":\"value, comma\"}"
  let buf = bytes_of_ascii(s)
  let indices = @simd_buffer.SimdBuffer::make(s.length())
  let count = find_structural_indices(buf, indices)
  let want = expected_structural_indices(s)
  if count != want.length() {
    abort(
      "count mismatch (strings-contain-structural): want=" +
      want.length().to_string() +
      " got=" +
      count.to_string(),
    )
  }
  for k in 0..<count {
    if indices.get(k) != want[k] {
      abort("indices mismatch at " + k.to_string())
    }
  }
}

///|
test "find_structural_indices — escaped quote does not terminate string" {
  let s = "{\"a\\\":b\":1,\"c\":2}"
  let buf = bytes_of_ascii(s)
  let indices = @simd_buffer.SimdBuffer::make(s.length())
  let count = find_structural_indices(buf, indices)
  let want = expected_structural_indices(s)
  if count != want.length() {
    abort(
      "escaped-quote count mismatch: want=" +
      want.length().to_string() +
      " got=" +
      count.to_string(),
    )
  }
  for k in 0..<count {
    if indices.get(k) != want[k] {
      abort("escaped-quote indices mismatch at " + k.to_string())
    }
  }
}

///|
test "find_structural_indices — large input crossing many words" {
  let inner = "{\"a\":1,\"b\":\"some string, with : and [ and ]\",\"c\":[1,2,3]}"
  let buf_w = StringBuilder::new()
  buf_w.write_string("[")
  for k in 0..<10 {
    if k > 0 {
      buf_w.write_string(",")
    }
    buf_w.write_string(inner)
  }
  buf_w.write_string("]")
  let s = buf_w.to_string()
  let buf = bytes_of_ascii(s)
  let indices = @simd_buffer.SimdBuffer::make(s.length())
  let count = find_structural_indices(buf, indices)
  let want = expected_structural_indices(s)
  if count != want.length() {
    abort(
      "large-input count mismatch: want=" +
      want.length().to_string() +
      " got=" +
      count.to_string(),
    )
  }
  for k in 0..<count {
    if indices.get(k) != want[k] {
      abort(
        "large-input indices mismatch at " +
        k.to_string() +
        ": want=" +
        want[k].to_string() +
        " got=" +
        indices.get(k).to_string(),
      )
    }
  }
}
