// Tests run on all four targets (wasm / wasm-gc / native / js). Each asserts
// the simdcore result equals the moonbitlang/core idiom it replaces, so the
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
test "count_nonzero" {
  let a : FixedArray[Int] = [0, 1, 0, 2, 0, 0, 3, 4, 0, 5, 0, 0, 6]
  let mut want = 0
  for i in 0..<a.length() {
    if a[i] != 0 {
      want = want + 1
    }
  }
  assert_eq(count_nonzero(a), want)
}

///|
test "int element-wise add/sub/mul/neg/abs/saxpy" {
  let a : FixedArray[Int] = [3, -7, 11, 0, 5, 9, -2, 8, 13, -1]
  let b : FixedArray[Int] = [1, 4, -6, 2, 5, -3, 7, 8, -9, 10]
  let n = a.length()
  let out = FixedArray::make(n, 0)
  add(a, b, out)
  for i in 0..<n {
    assert_eq(out[i], a[i] + b[i])
  }
  sub(a, b, out)
  for i in 0..<n {
    assert_eq(out[i], a[i] - b[i])
  }
  mul(a, b, out)
  for i in 0..<n {
    assert_eq(out[i], a[i] * b[i])
  }
  neg(a, out)
  for i in 0..<n {
    assert_eq(out[i], -a[i])
  }
  abs(a, out)
  for i in 0..<n {
    assert_eq(out[i], if a[i] < 0 { -a[i] } else { a[i] })
  }
  saxpy(3, a, b, out)
  for i in 0..<n {
    assert_eq(out[i], 3 * a[i] + b[i])
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
  assert_eq(sum_f64(a), s)
  assert_eq(dot_f64(a, b), d)
  assert_eq(mean_f64(a), s / n.to_double())
}

///|
test "f64 element-wise add/sub/mul/div/sqrt/min/max" {
  let a : FixedArray[Double] = [4.0, 9.0, 16.0, 25.0, 1.0, 36.0]
  let b : FixedArray[Double] = [2.0, 3.0, 8.0, 5.0, 4.0, 6.0]
  let n = a.length()
  let out = FixedArray::make(n, 0.0)
  add_f64(a, b, out)
  for i in 0..<n {
    assert_eq(out[i], a[i] + b[i])
  }
  sub_f64(a, b, out)
  for i in 0..<n {
    assert_eq(out[i], a[i] - b[i])
  }
  mul_f64(a, b, out)
  for i in 0..<n {
    assert_eq(out[i], a[i] * b[i])
  }
  div_f64(a, b, out)
  for i in 0..<n {
    assert_eq(out[i], a[i] / b[i])
  }
  sqrt_f64(a, out)
  for i in 0..<n {
    assert_eq(out[i], a[i].sqrt())
  }
  min_elem_f64(a, b, out)
  for i in 0..<n {
    assert_eq(out[i], if a[i] < b[i] { a[i] } else { b[i] })
  }
  max_elem_f64(a, b, out)
  for i in 0..<n {
    assert_eq(out[i], if a[i] > b[i] { a[i] } else { b[i] })
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
  assert_eq(bytes_count(data, b'a'), want_count)
  assert_eq(bytes_search(data, b'a'), Some(want_first))
  assert_eq(bytes_search(data, b'_'), Some(8))
  assert_eq(bytes_search(data, b'Q'), None)
  assert_eq(bytes_count(data, b'Q'), 0)
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
  assert_eq(lo, [b'h', b'e', b'l', b'l', b'o', b'1'])
  let up : FixedArray[Byte] = [b'h', b'e', b'l', b'l', b'o', b'1']
  to_upper_ascii(up)
  assert_eq(up, [b'H', b'E', b'L', b'L', b'O', b'1'])
}
