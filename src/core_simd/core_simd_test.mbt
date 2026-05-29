// Tests run on all four targets (wasm / wasm-gc / native / js). Each asserts
// the core_simd result equals the moonbitlang/core idiom it replaces, so the
// substitution is verified to be behaviour-preserving regardless of backend.

///|
test "sum / product / dot" {
  let a : FixedArray[Int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  assert_eq(sum(a), a.iter().fold(init=0, fn(x, y) { x + y }))
  assert_eq(product(a), a.iter().fold(init=1, fn(x, y) { x * y }))
  let b : FixedArray[Int] = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
  let mut want = 0
  for i in 0..<a.length() {
    want = want + a[i] * b[i]
  }
  assert_eq(dot(a, b), want)
}

///|
test "sum empty" {
  let a : FixedArray[Int] = []
  assert_eq(sum(a), 0)
  assert_eq(product(a), 1)
}

///|
test "maximum / minimum" {
  let a : FixedArray[Int] = [5, 3, 9, 1, 7, 2, 8, 4, 6, 0, -3, 11]
  assert_eq(maximum(a), Some(11))
  assert_eq(minimum(a), Some(-3))
  let empty : FixedArray[Int] = []
  assert_eq(maximum(empty), None)
  assert_eq(minimum(empty), None)
}

///|
test "search / contains" {
  let a : FixedArray[Int] = [4, 8, 15, 16, 23, 42]
  assert_eq(search(a, 16), Some(3))
  assert_eq(search(a, 4), Some(0))
  assert_eq(search(a, 42), Some(5))
  assert_eq(search(a, 99), None)
  assert_true(contains(a, 23))
  assert_false(contains(a, 24))
}

///|
test "fill" {
  let a : FixedArray[Int] = FixedArray::make(13, 0)
  fill(a, 7)
  for i in 0..<a.length() {
    assert_eq(a[i], 7)
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
  assert_eq(a, want)
}

///|
test "bytes_equal" {
  let a : FixedArray[Byte] = [b'h', b'e', b'l', b'l', b'o', b'!', b'!', b'!']
  let b : FixedArray[Byte] = [b'h', b'e', b'l', b'l', b'o', b'!', b'!', b'!']
  let c : FixedArray[Byte] = [b'h', b'e', b'l', b'l', b'o', b'!', b'!', b'?']
  let d : FixedArray[Byte] = [b'h', b'i']
  assert_true(bytes_equal(a, b))
  assert_false(bytes_equal(a, c))
  assert_false(bytes_equal(a, d))
}

///|
test "bytes_search / bytes_count" {
  let data : FixedArray[Byte] = [
    b'a', b'b', b'a', b'c', b'a', b'b', b'a', b'd', b'a', b'b', b'a', b'c',
  ]
  assert_eq(bytes_search(data, b'c'), Some(3))
  assert_eq(bytes_search(data, b'z'), None)
  assert_eq(bytes_count(data, b'a'), 6)
  assert_eq(bytes_count(data, b'z'), 0)
}

///|
test "bytes ascii ops" {
  let ascii : FixedArray[Byte] = [b'H', b'e', b'l', b'l', b'o']
  assert_true(bytes_is_ascii(ascii))
  let hi : FixedArray[Byte] = [b'A', 0xC3, b'Z']
  assert_false(bytes_is_ascii(hi))
  let lo : FixedArray[Byte] = [b'H', b'E', b'L', b'L', b'O', b'1']
  bytes_to_lower_ascii(lo)
  assert_eq(lo, [b'h', b'e', b'l', b'l', b'o', b'1'])
  let up : FixedArray[Byte] = [b'h', b'e', b'l', b'l', b'o', b'1']
  bytes_to_upper_ascii(up)
  assert_eq(up, [b'H', b'E', b'L', b'L', b'O', b'1'])
}
