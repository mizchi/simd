// Tests run on all four targets (wasm / wasm-gc / native / js). Each asserts
// the simdcore result equals the moonbitlang/core idiom it replaces, so the
// substitution is verified to be behaviour-preserving regardless of backend.

///|
test "sum / product / dot" {
  let a : FixedArray[Int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  assert_true(sum(a) == a.iter().fold(init=0, fn(x, y) { x + y }))
  assert_true(product(a) == a.iter().fold(init=1, fn(x, y) { x * y }))
  let b : FixedArray[Int] = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
  let mut want = 0
  for i in 0..<a.length() {
    want = want + a[i] * b[i]
  }
  assert_true(dot(a, b) == want)
}

///|
test "sum empty" {
  let a : FixedArray[Int] = []
  assert_true(sum(a) == 0)
  assert_true(product(a) == 1)
}

///|
test "maximum / minimum" {
  let a : FixedArray[Int] = [5, 3, 9, 1, 7, 2, 8, 4, 6, 0, -3, 11]
  assert_true(maximum(a) == Some(11))
  assert_true(minimum(a) == Some(-3))
  let empty : FixedArray[Int] = []
  assert_true(maximum(empty) == None)
  assert_true(minimum(empty) == None)
}

///|
test "maximum_f64 / minimum_f64" {
  // odd and even lengths exercise the f64x2 body + scalar tail; negatives and
  // a value in the tail-only position guard the select push order.
  let cases : Array[FixedArray[Double]] = [
    [3.5],
    [3.5, -2.0],
    [1.0, 9.0, -4.0, 2.0, 7.0],
    [-1.0, -9.0, -4.0, -2.0, -7.0, -0.5],
    [5.0, 3.0, 9.0, 1.0, 7.0, 2.0, 8.0, 4.0, 6.0, 0.0, -3.0, 11.0],
  ]
  for arr in cases {
    let mut want_max = arr[0]
    let mut want_min = arr[0]
    for i in 1..<arr.length() {
      if arr[i] > want_max {
        want_max = arr[i]
      }
      if arr[i] < want_min {
        want_min = arr[i]
      }
    }
    assert_true(maximum_f64(arr) == Some(want_max))
    assert_true(minimum_f64(arr) == Some(want_min))
  }
  let empty : FixedArray[Double] = []
  assert_true(maximum_f64(empty) == None)
  assert_true(minimum_f64(empty) == None)
}

///|
test "search / contains" {
  let a : FixedArray[Int] = [4, 8, 15, 16, 23, 42]
  assert_true(search(a, 16) == Some(3))
  assert_true(search(a, 4) == Some(0))
  assert_true(search(a, 42) == Some(5))
  assert_true(search(a, 99) == None)
  assert_true(contains(a, 23))
  assert_false(contains(a, 24))
}

///|
test "fill" {
  let a : FixedArray[Int] = FixedArray::make(13, 0)
  fill(a, 7)
  for i in 0..<a.length() {
    assert_true(a[i] == 7)
  }
}

///|
test "sort" {
  let a : FixedArray[Int] = [
    9, 1, 8, 2, 7, 3, 6, 4, 5, 0, 15, 11, 13, 10, 12, 14, 99, -5,
  ]
  let want = a.copy()
  want.sort()
  sort(a)
  assert_true(a == want)
}

///|
test "count_nonzero" {
  let a : FixedArray[Int] = [0, 1, 0, 2, 0, 0, 3, 4, 0, 5, 0, 0, 6]
  let mut want = 0
  for i in 0..<a.length() {
    if a[i] != 0 {
      want = want + 1
    }
  }
  assert_true(count_nonzero(a) == want)
}

///|
test "int element-wise add/sub/mul/neg/abs/saxpy" {
  let a : FixedArray[Int] = [3, -7, 11, 0, 5, 9, -2, 8, 13, -1]
  let b : FixedArray[Int] = [1, 4, -6, 2, 5, -3, 7, 8, -9, 10]
  let n = a.length()
  let out = FixedArray::make(n, 0)
  add(a, b, out)
  for i in 0..<n {
    assert_true(out[i] == a[i] + b[i])
  }
  sub(a, b, out)
  for i in 0..<n {
    assert_true(out[i] == a[i] - b[i])
  }
  mul(a, b, out)
  for i in 0..<n {
    assert_true(out[i] == a[i] * b[i])
  }
  neg(a, out)
  for i in 0..<n {
    assert_true(out[i] == -a[i])
  }
  abs(a, out)
  for i in 0..<n {
    assert_true(out[i] == (if a[i] < 0 { -a[i] } else { a[i] }))
  }
  saxpy(3, a, b, out)
  for i in 0..<n {
    assert_true(out[i] == 3 * a[i] + b[i])
  }
}

///|
test "f64 reductions" {
  let a : FixedArray[Double] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  let b : FixedArray[Double] = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
  let n = a.length()
  let mut s = 0.0
  let mut d = 0.0
  for i in 0..<n {
    s = s + a[i]
    d = d + a[i] * b[i]
  }
  assert_true(sum_f64(a) == s)
  assert_true(dot_f64(a, b) == d)
  assert_true(mean_f64(a) == s / n.to_double())
}

///|
test "f64 element-wise add/sub/mul/div/sqrt/min/max" {
  let a : FixedArray[Double] = [4.0, 9.0, 16.0, 25.0, 1.0, 36.0]
  let b : FixedArray[Double] = [2.0, 3.0, 8.0, 5.0, 4.0, 6.0]
  let n = a.length()
  let out = FixedArray::make(n, 0.0)
  add_f64(a, b, out)
  for i in 0..<n {
    assert_true(out[i] == a[i] + b[i])
  }
  sub_f64(a, b, out)
  for i in 0..<n {
    assert_true(out[i] == a[i] - b[i])
  }
  mul_f64(a, b, out)
  for i in 0..<n {
    assert_true(out[i] == a[i] * b[i])
  }
  div_f64(a, b, out)
  for i in 0..<n {
    assert_true(out[i] == a[i] / b[i])
  }
  sqrt_f64(a, out)
  for i in 0..<n {
    assert_true(out[i] == a[i].sqrt())
  }
  min_elem_f64(a, b, out)
  for i in 0..<n {
    assert_true(out[i] == (if a[i] < b[i] { a[i] } else { b[i] }))
  }
  max_elem_f64(a, b, out)
  for i in 0..<n {
    assert_true(out[i] == (if a[i] > b[i] { a[i] } else { b[i] }))
  }
}

///|
test "bytes_equal (Bytes)" {
  // > 16 bytes to exercise the v128 chunk path plus the scalar tail.
  let a = b"the quick brown fox jumps"
  let b = b"the quick brown fox jumps"
  let c = b"the quick brown fox jump!"
  let d = b"the quick brown fox"
  assert_true(bytes_equal(a, b))
  assert_false(bytes_equal(a, c))
  assert_false(bytes_equal(a, d))
}

///|
test "bytes_search / bytes_count / bytes_contains (Bytes)" {
  let data = b"abacabad_abacaba_padding_xyz_padding"
  // cross-check the SIMD facade against an in-test scalar reference.
  let mut want_count = 0
  let mut want_first = -1
  for i in 0..<data.length() {
    if data[i] == b'a' {
      want_count = want_count + 1
      if want_first == -1 {
        want_first = i
      }
    }
  }
  assert_true(bytes_count(data, b'a') == want_count)
  assert_true(bytes_search(data, b'a') == Some(want_first))
  assert_true(bytes_search(data, b'_') == Some(8))
  assert_true(bytes_search(data, b'Q') == None)
  assert_true(bytes_count(data, b'Q') == 0)
  assert_true(bytes_contains(data, b'z'))
  assert_false(bytes_contains(data, b'Q'))
}

///|
test "bytes_is_ascii (Bytes)" {
  assert_true(bytes_is_ascii(b"a pure ascii string, padded out past 16 bytes"))
  // a high byte in the SIMD region (index 20).
  let hi = b"twenty ascii bytes..\xc3 then a high byte"
  assert_false(bytes_is_ascii(hi))
}

///|
test "ascii case ops (FixedArray in-place)" {
  let lo : FixedArray[Byte] = [b'H', b'E', b'L', b'L', b'O', b'1']
  to_lower_ascii(lo)
  assert_true(lo == [b'h', b'e', b'l', b'l', b'o', b'1'])
  let up : FixedArray[Byte] = [b'h', b'e', b'l', b'l', b'o', b'1']
  to_upper_ascii(up)
  assert_true(up == [b'H', b'E', b'L', b'L', b'O', b'1'])
}

///|
fn make_string(n : Int, f : (Int) -> Int) -> String {
  let arr = Array::new()
  for i in 0..<n {
    arr.push(f(i).unsafe_to_char())
  }
  String::from_array(arr)
}

///|
test "is_ascii_string" {
  // long enough (> 16 units) to cover the SIMD chunk path + scalar tail.
  let ascii = make_string(45, fn(i) { 0x41 + i % 26 })
  assert_true(is_ascii_string(ascii))
  // a non-ASCII unit at index 20, inside the SIMD region.
  let mixed = make_string(45, fn(i) { if i == 20 { 0x20AC } else { 0x41 } })
  assert_false(is_ascii_string(mixed))
  assert_true(is_ascii_string(""))
}

///|
test "encode_utf8 (ascii fast path == core)" {
  // lengths chosen to hit 16-aligned body, body+tail, and pure tail (< 16).
  for n in [0, 5, 16, 33, 100] {
    let s = make_string(n, fn(i) { 0x41 + i % 26 })
    assert_true(encode_utf8(s) == @utf8.encode(s[:]))
  }
}

///|
test "encode_utf8 (non-ascii delegates to core)" {
  // €(BMP, 3-byte) and 😀(astral, surrogate pair -> 4-byte).
  let s = make_string(40, fn(i) {
    if i == 10 {
      0x20AC
    } else if i == 25 {
      0x41
    } else {
      0x42
    }
  })
  assert_true(encode_utf8(s) == @utf8.encode(s[:]))
  let astral = "hello \u{1F600} world, padded out beyond sixteen units"
  assert_true(encode_utf8(astral) == @utf8.encode(astral[:]))
}

///|
test "encode_utf8_into" {
  let s = make_string(50, fn(i) { 0x61 + i % 26 })
  let out = FixedArray::make(64, b'\x00')
  let written = encode_utf8_into(s, out)
  assert_true(written == 50)
  let want = @utf8.encode(s[:])
  for i in 0..<written {
    assert_true(out[i] == want[i])
  }
}

///|
test "decode_utf8_unsafe (ascii == core)" {
  for n in [0, 5, 16, 33, 100] {
    let s = make_string(n, fn(i) { 0x41 + i % 26 })
    let bytes = @utf8.encode(s[:])
    assert_true(decode_utf8_unsafe(bytes) == s)
    assert_true(decode_utf8_unsafe_intrinsic(bytes) == s)
  }
}

///|
test "decode_utf8_unsafe (non-ascii == decode_lossy)" {
  // valid multi-byte UTF-8 (é, ö, ☃, 😀) padded beyond 16 bytes.
  let s = "h\u{e9}llo w\u{f6}rld \u{2603} and \u{1F600}, padded beyond sixteen"
  let bytes = @utf8.encode(s[:])
  assert_true(decode_utf8_unsafe(bytes) == @utf8.decode_lossy(bytes[:]))
  assert_true(
    decode_utf8_unsafe_intrinsic(bytes) == @utf8.decode_lossy(bytes[:]),
  )
  // round-trips back to the original for well-formed input.
  assert_true(decode_utf8_unsafe(bytes) == s)
}

///|
fn int_array(n : Int) -> Array[Int] {
  let a = Array::new(capacity=n)
  for i in 0..<n {
    a.push((i * 37 + 11) % 1000 - 500)
  }
  a
}

///|
test "array bridge: to/of_fixedarray round trip" {
  let a = int_array(20)
  let fa = to_fixedarray(a)
  assert_true(fa.length() == 20)
  let back = of_fixedarray(fa)
  assert_true(back == a)
}

///|
test "array reductions == idiom" {
  let a = int_array(50)
  assert_true(array_sum(a) == a.iter().fold(init=0, fn(x, y) { x + y }))
  assert_true(
    array_product(int_array(6)) ==
    int_array(6).iter().fold(init=1, fn(x, y) { x * y }),
  )
  assert_true(array_maximum(a) == a.iter().maximum())
  assert_true(array_minimum(a) == a.iter().minimum())
  let mut nz = 0
  for i in 0..<a.length() {
    if a[i] != 0 {
      nz = nz + 1
    }
  }
  assert_true(array_count_nonzero(a) == nz)
  let b = int_array(50)
  let mut d = 0
  for i in 0..<a.length() {
    d = d + a[i] * b[i]
  }
  assert_true(array_dot(a, b) == d)
}

///|
test "array_sort in place == Array::sort" {
  let a = int_array(100)
  let want = a.copy()
  want.sort()
  array_sort(a)
  assert_true(a == want)
  // edge cases
  let one = [42]
  array_sort(one)
  assert_true(one == [42])
  let empty : Array[Int] = []
  array_sort(empty)
  assert_true(empty == [])
}

///|
test "array f64 reductions == loop" {
  let a : Array[Double] = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
  let b : Array[Double] = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
  let mut s = 0.0
  let mut d = 0.0
  for i in 0..<a.length() {
    s = s + a[i]
    d = d + a[i] * b[i]
  }
  assert_true(array_sum_f64(a) == s)
  assert_true(array_dot_f64(a, b) == d)
  assert_true(array_mean_f64(a) == s / 8.0)
}

///|
/// Reference: structural offsets of `{}[],:` NOT inside a string literal.
fn json_indices_ref(s : String) -> Array[Int] {
  let bytes = @utf8.encode(s[:])
  let n = bytes.length()
  let result = Array::new()
  let mut in_string = false
  let mut escape_pending = false
  for i in 0..<n {
    let b = bytes[i].to_int()
    if escape_pending {
      escape_pending = false
    } else if b == 0x22 {
      in_string = !in_string
    } else if b == 0x5C && in_string {
      escape_pending = true
    } else if !in_string &&
      (
        b == 0x7B ||
        b == 0x7D ||
        b == 0x5B ||
        b == 0x5D ||
        b == 0x2C ||
        b == 0x3A
      ) {
      result.push(i)
    }
  }
  result
}

///|
test "json_structural_indices vs reference" {
  let docs = [
    "{}", "[]", "{\"a\":1,\"b\":[2,3]}", "{\"k\":\"a,b:c{}[]\",\"n\":42}", "{\"q\":\"he said \\\"hi\\\"\",\"p\":\"a\\\\b\"}",
    "[{\"id\":1,\"tags\":[\"x\",\"y\"]},{\"id\":2,\"tags\":[\"z\"]},{\"id\":3}]",
  ]
  for d in docs {
    let bytes = @utf8.encode(d[:])
    assert_true(json_structural_indices(bytes) == json_indices_ref(d))
  }
}

///|
test "json_structural_indices empty" {
  assert_true(json_structural_indices(b"") == (Array::new() : Array[Int]))
}

///|
test "json_classify_structural bitmap" {
  let bytes = b"{\"a\":[1,2]}"
  let words = (bytes.length() + 31) / 32
  let out = FixedArray::make(words, 0)
  json_classify_structural(bytes, out)
  for i in 0..<bytes.length() {
    let b = bytes[i].to_int()
    let want = b == 0x7B ||
      b == 0x7D ||
      b == 0x5B ||
      b == 0x5D ||
      b == 0x2C ||
      b == 0x3A
    let got = (out[i >> 5] & (1 << (i & 31))) != 0
    assert_true(got == want)
  }
}

///|
test "json_structural_indices_into reuse" {
  let doc = "{\"x\":[10,20,30],\"y\":{\"z\":true}}"
  let bytes = @utf8.encode(doc[:])
  let n = bytes.length()
  let words = (n + 31) / 32
  let structural = FixedArray::make(words, 0)
  let quote = FixedArray::make(words, 0)
  let indices = FixedArray::make(n, 0)
  let count = json_structural_indices_into(bytes, structural, quote, indices)
  let got = Array::new()
  for i in 0..<count {
    got.push(indices[i])
  }
  assert_true(got == json_indices_ref(doc))
  let count2 = json_structural_indices_into(bytes, structural, quote, indices)
  assert_true(count2 == count)
}

///|
/// Reference substring search (naive) for cross-checking.
fn ref_index_of(h : Bytes, n : Bytes) -> Int {
  let hl = h.length()
  let nl = n.length()
  if nl == 0 {
    return 0
  }
  if nl > hl {
    return -1
  }
  for i in 0..<=(hl - nl) {
    let mut ok = true
    for k in 0..<nl {
      if h[i + k] != n[k] {
        ok = false
        break
      }
    }
    if ok {
      return i
    }
  }
  -1
}

///|
test "bytes_index_of vs reference" {
  let h = b"the quick brown fox jumps over the lazy dog, padded out a bit more"
  let needles = [
    b"the", b"fox", b"dog", b"quick", b"more", b"t", b"e ", b"zzz", b"the quick",
    b"", b"padded out a bit more",
  ]
  for n in needles {
    let want = ref_index_of(h, n)
    let got = match bytes_index_of(h, n) {
      Some(i) => i
      None => -1
    }
    assert_true(got == want)
  }
  // overlap / repeated first byte (forces multiple candidate verifies):
  // every "a" matches the first needle byte, so the SIMD scan keeps finding
  // candidates that fail verification until the real match.
  let h2 = b"aaaaaaaaaaaaaaaaaaab"
  for n in [b"aab", b"aaab", b"ab", b"aaaa"] {
    let want = ref_index_of(h2, n)
    assert_true(
      bytes_index_of(h2, n) == (if want < 0 { None } else { Some(want) }),
    )
  }
  assert_true(bytes_contains_sub(h2, b"aab"))
  assert_false(bytes_contains_sub(h2, b"aac"))
  // needle longer than haystack
  assert_true(bytes_index_of(b"hi", b"hello") == None)
  // match at the very end, crossing a 16-byte boundary
  let h3 = b"0123456789abcdefNEEDLE"
  assert_true(bytes_index_of(h3, b"NEEDLE") == Some(16))
}

///|
test "bytes_rindex vs reference" {
  // > 32 bytes so the reverse scan covers the unaligned tail + >=2 chunks.
  let h = b"/usr/local/bin/moon; path.with.dots; trailing/slash/"
  for needle in [b'/', b'.', b';', b'z', b'm'] {
    // scalar reference: last index of needle
    let mut want = -1
    for i in 0..<h.length() {
      if h[i] == needle {
        want = i
      }
    }
    assert_true(
      bytes_rindex(h, needle) == (if want < 0 { None } else { Some(want) }),
    )
  }
  // boundary: match only in the unaligned tail
  assert_true(bytes_rindex(b"0123456789abcdefZ", b'Z') == Some(16))
  // match only in the first chunk
  assert_true(bytes_rindex(b"Z0123456789abcdef0123456789", b'Z') == Some(0))
  assert_true(bytes_rindex(b"", b'a') == None)
  // last vs first occurrence differ
  let h2 = b"abcabcabc"
  assert_true(bytes_rindex(h2, b'a') == Some(6))
  assert_true(bytes_search(h2, b'a') == Some(0))
}

///|
test "option A buffer-backed variants match scalar simdcore" {
  let a : FixedArray[Int] = FixedArray::make(300, 0)
  let b : FixedArray[Int] = FixedArray::make(300, 0)
  for i in 0..<300 {
    a[i] = (i * 48271 + 7) % 1000 - 500
    b[i] = (i * 39916 + 3) % 1000 - 500
  }
  // sum / dot parity (value-returning)
  assert_eq(sum_via_buffer(a), sum(a))
  assert_eq(dot_via_buffer(a, b), dot(a, b))
  // sort parity — both sort in place (Unit). Compare independent copies.
  let want : FixedArray[Int] = FixedArray::make(300, 0)
  let got : FixedArray[Int] = FixedArray::make(300, 0)
  for i in 0..<300 {
    want[i] = a[i]
    got[i] = a[i]
  }
  sort(want)
  sort_via_buffer(got)
  for i in 0..<300 {
    assert_eq(got[i], want[i])
  }
}
