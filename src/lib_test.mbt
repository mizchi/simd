///|
fn make_int_array(vals : Array[Int]) -> FixedArray[Int] {
  let arr = FixedArray::make(vals.length(), 0)
  for i, v in vals {
    arr[i] = v
  }
  arr
}

///|
fn make_byte_array(vals : Array[Int]) -> FixedArray[Byte] {
  let arr = FixedArray::make(vals.length(), b'\x00')
  for i, v in vals {
    arr[i] = v.to_byte()
  }
  arr
}

// --- @internal.scalar_sum ---

///|
test "scalar_sum empty" {
  let arr : FixedArray[Int] = FixedArray::make(0, 0)
  inspect(@internal.scalar_sum(arr), content="0")
}

///|
test "scalar_sum basic" {
  let arr = make_int_array([1, 2, 3, 4, 5])
  inspect(@internal.scalar_sum(arr), content="15")
}

///|
test "scalar_sum with negatives" {
  let arr = make_int_array([10, -3, 7, -2])
  inspect(@internal.scalar_sum(arr), content="12")
}

// --- sum_i32 ---

///|
test "sum_i32 empty" {
  let arr : FixedArray[Int] = FixedArray::make(0, 0)
  inspect(sum_i32(arr), content="0")
}

///|
test "sum_i32 basic" {
  let arr = make_int_array([1, 2, 3, 4, 5, 6, 7, 8])
  inspect(sum_i32(arr), content="36")
}

///|
test "sum_i32 non-aligned" {
  let arr = make_int_array([1, 2, 3, 4, 5])
  inspect(sum_i32(arr), content="15")
}

///|
test "sum_i32 matches scalar" {
  let arr = make_int_array([10, 20, 30, 40, 50, 60, 70])
  inspect(sum_i32(arr) == @internal.scalar_sum(arr), content="true")
}

// --- @internal.scalar_dot_product ---

///|
test "scalar_dot_product basic" {
  let a = make_int_array([1, 2, 3])
  let b = make_int_array([4, 5, 6])
  inspect(@internal.scalar_dot_product(a, b), content="32")
}

// --- dot_i32 ---

///|
test "dot_i32 basic" {
  let a = make_int_array([1, 2, 3, 4])
  let b = make_int_array([5, 6, 7, 8])
  inspect(dot_i32(a, b), content="70")
}

///|
test "dot_i32 non-aligned" {
  let a = make_int_array([1, 2, 3, 4, 5])
  let b = make_int_array([2, 3, 4, 5, 6])
  inspect(dot_i32(a, b) == @internal.scalar_dot_product(a, b), content="true")
}

// --- @internal.scalar_add ---

///|
test "scalar_add basic" {
  let a = make_int_array([1, 2, 3, 4])
  let b = make_int_array([10, 20, 30, 40])
  let out = FixedArray::make(4, 0)
  @internal.scalar_add(a, b, out)
  inspect(out[0], content="11")
  inspect(out[3], content="44")
}

// --- add_i32 ---

///|
test "add_i32 basic" {
  let a = make_int_array([1, 2, 3, 4, 5, 6, 7, 8])
  let b = make_int_array([10, 20, 30, 40, 50, 60, 70, 80])
  let out = FixedArray::make(8, 0)
  add_i32(a, b, out)
  inspect(out[0], content="11")
  inspect(out[7], content="88")
}

///|
test "add_i32 non-aligned" {
  let a = make_int_array([1, 2, 3, 4, 5])
  let b = make_int_array([10, 20, 30, 40, 50])
  let out = FixedArray::make(5, 0)
  let expected = FixedArray::make(5, 0)
  add_i32(a, b, out)
  @internal.scalar_add(a, b, expected)
  for i in 0..<5 {
    assert_eq(out[i], expected[i])
  }
}

// --- @internal.scalar_adler32 ---

///|
test "scalar_adler32 empty" {
  let data : FixedArray[Byte] = FixedArray::make(0, b'\x00')
  inspect(@internal.scalar_adler32(data), content="1")
}

///|
test "scalar_adler32 wikipedia" {
  // "Wikipedia" -> Adler-32 = 0x11E60398
  let data = make_byte_array([87, 105, 107, 105, 112, 101, 100, 105, 97])
  inspect(@internal.scalar_adler32(data), content="300286872")
}

// --- adler32_bytes ---

///|
test "adler32_bytes matches scalar" {
  let data = make_byte_array([87, 105, 107, 105, 112, 101, 100, 105, 97])
  inspect(adler32_bytes(data) == @internal.scalar_adler32(data), content="true")
}

///|
test "adler32_bytes empty" {
  let data : FixedArray[Byte] = FixedArray::make(0, b'\x00')
  inspect(adler32_bytes(data), content="1")
}

// --- min / max / argmin / argmax ---

///|
test "scalar_min/max basic" {
  let arr = make_int_array([5, -3, 8, -1, 2, 7])
  inspect(@internal.scalar_min(arr), content="-3")
  inspect(@internal.scalar_max(arr), content="8")
}

///|
test "scalar_argmin/argmax basic" {
  let arr = make_int_array([5, -3, 8, -1, 2, 7])
  inspect(@internal.scalar_argmin(arr), content="1")
  inspect(@internal.scalar_argmax(arr), content="2")
}

///|
test "min_i32/max match scalar" {
  let arr = make_int_array([5, -3, 8, -1, 2, 7, 100, -50, 12, 4, 0, 9])
  assert_eq(min_i32(arr), @internal.scalar_min(arr))
  assert_eq(max_i32(arr), @internal.scalar_max(arr))
}

///|
test "argmin_i32/argmax match scalar" {
  let arr = make_int_array([5, -3, 8, -1, 2, 7, 100, -50, 12, 4, 0, 9])
  assert_eq(argmin_i32(arr), @internal.scalar_argmin(arr))
  assert_eq(argmax_i32(arr), @internal.scalar_argmax(arr))
}

///|
test "min_i32/max non-aligned" {
  let arr = make_int_array([1, 2, 3, 4, 5])
  assert_eq(min_i32(arr), 1)
  assert_eq(max_i32(arr), 5)
}

///|
test "argmin_i32/argmax with duplicates picks first" {
  let arr = make_int_array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])
  // first min (1) at index 1, first max (9) at index 5
  assert_eq(argmin_i32(arr), 1)
  assert_eq(argmax_i32(arr), 5)
}

///|
test "min_i32/max empty" {
  let arr : FixedArray[Int] = FixedArray::make(0, 0)
  inspect(min_i32(arr), content="0")
  inspect(max_i32(arr), content="0")
  inspect(argmin_i32(arr), content="-1")
  inspect(argmax_i32(arr), content="-1")
}

// --- saxpy ---

///|
test "saxpy_i32 matches scalar" {
  let a = make_int_array([1, 2, 3, 4, 5, 6, 7, 8, 9])
  let b = make_int_array([10, 20, 30, 40, 50, 60, 70, 80, 90])
  let out_simd = FixedArray::make(9, 0)
  let out_scalar = FixedArray::make(9, 0)
  saxpy_i32(3, a, b, out_simd)
  @internal.scalar_saxpy(3, a, b, out_scalar)
  for i in 0..<9 {
    assert_eq(out_simd[i], out_scalar[i])
  }
}

///|
test "saxpy_i32 basic value" {
  let a = make_int_array([1, 2, 3, 4])
  let b = make_int_array([10, 20, 30, 40])
  let out = FixedArray::make(4, 0)
  saxpy_i32(2, a, b, out)
  // 2*1+10=12, 2*2+20=24, 2*3+30=36, 2*4+40=48
  assert_eq(out[0], 12)
  assert_eq(out[3], 48)
}

// --- popcount_bytes ---

///|
test "scalar_popcount_bytes basic" {
  let data = make_byte_array([0, 1, 2, 3, 255])
  // 0+1+1+2+8 = 12
  inspect(@internal.scalar_popcount_bytes(data), content="12")
}

///|
test "popcount_bytes matches scalar" {
  let vals = Array::makei(40, fn(i) { (i * 13 + 7) % 256 })
  let data = make_byte_array(vals)
  assert_eq(popcount_bytes(data), @internal.scalar_popcount_bytes(data))
}

///|
test "popcount_bytes empty" {
  let data : FixedArray[Byte] = FixedArray::make(0, b'\x00')
  inspect(popcount_bytes(data), content="0")
}

// --- byte primitives ---

///|
test "memcpy_bytes basic" {
  let src = make_byte_array([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
  ])
  let dst = FixedArray::make(20, b'\x00')
  memcpy_bytes(src, 0, dst, 0, 20)
  for i in 0..<20 {
    assert_eq(dst[i], src[i])
  }
}

///|
test "memcpy_bytes with offsets" {
  let src = make_byte_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
  let dst = FixedArray::make(15, b'\xff')
  memcpy_bytes(src, 2, dst, 3, 6)
  // dst[3..9] = src[2..8] = 3,4,5,6,7,8; rest unchanged
  assert_eq(dst[0], b'\xff')
  assert_eq(dst[3], b'\x03')
  assert_eq(dst[8], b'\x08')
  assert_eq(dst[9], b'\xff')
}

///|
test "memset_bytes basic" {
  let dst = FixedArray::make(20, b'\x00')
  memset_bytes(dst, 0, 20, b'\x5a')
  for i in 0..<20 {
    assert_eq(dst[i], b'\x5a')
  }
}

///|
test "memset_bytes partial" {
  let dst = FixedArray::make(20, b'\x00')
  memset_bytes(dst, 5, 10, b'\x42')
  assert_eq(dst[4], b'\x00')
  assert_eq(dst[5], b'\x42')
  assert_eq(dst[14], b'\x42')
  assert_eq(dst[15], b'\x00')
}

///|
test "equal_bytes true" {
  let a = make_byte_array([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
  ])
  let b = make_byte_array([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
  ])
  inspect(equal_bytes(a, b, 17), content="true")
}

///|
test "equal_bytes false at tail" {
  let a = make_byte_array([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 99,
  ])
  let b = make_byte_array([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
  ])
  inspect(equal_bytes(a, b, 17), content="false")
}

///|
test "equal_bytes false at first SIMD block" {
  let a = make_byte_array([
    1, 2, 9, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
  ])
  let b = make_byte_array([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
  ])
  inspect(equal_bytes(a, b, 17), content="false")
}

///|
test "find_byte found in SIMD block" {
  let data = make_byte_array([0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
  inspect(find_byte(data, b'\x07'), content="5")
}

///|
test "find_byte found in tail" {
  let data = make_byte_array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9])
  inspect(find_byte(data, b'\x09'), content="16")
}

///|
test "find_byte not found" {
  let data = make_byte_array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
  inspect(find_byte(data, b'\x07'), content="-1")
}

///|
test "count_byte matches scalar" {
  let vals = Array::makei(50, fn(i) { if i % 3 == 0 { 7 } else { i % 256 } })
  let data = make_byte_array(vals)
  assert_eq(
    count_byte(data, b'\x07'),
    @internal.scalar_count_byte(data, b'\x07'),
  )
}

// --- element-wise Int ops (B) ---

///|
test "sub_i32 matches scalar" {
  let a = make_int_array([10, 20, 30, 40, 50, 60, 70])
  let b = make_int_array([1, 2, 3, 4, 5, 6, 7])
  let out = FixedArray::make(7, 0)
  let ref_arr = FixedArray::make(7, 0)
  sub_i32(a, b, out)
  @internal.scalar_sub(a, b, ref_arr)
  for i in 0..<7 {
    assert_eq(out[i], ref_arr[i])
  }
}

///|
test "mul_i32 matches scalar" {
  let a = make_int_array([1, 2, 3, 4, 5, 6, 7])
  let b = make_int_array([10, 20, 30, 40, 50, 60, 70])
  let out = FixedArray::make(7, 0)
  let ref_arr = FixedArray::make(7, 0)
  mul_i32(a, b, out)
  @internal.scalar_mul(a, b, ref_arr)
  for i in 0..<7 {
    assert_eq(out[i], ref_arr[i])
  }
}

///|
test "neg_i32 matches scalar" {
  let a = make_int_array([5, -3, 0, 17, -100, 42, -1])
  let out = FixedArray::make(7, 0)
  let ref_arr = FixedArray::make(7, 0)
  neg_i32(a, out)
  @internal.scalar_neg(a, ref_arr)
  for i in 0..<7 {
    assert_eq(out[i], ref_arr[i])
  }
}

///|
test "abs_i32 matches scalar" {
  let a = make_int_array([5, -3, 0, 17, -100, 42, -1])
  let out = FixedArray::make(7, 0)
  let ref_arr = FixedArray::make(7, 0)
  abs_i32(a, out)
  @internal.scalar_abs(a, ref_arr)
  for i in 0..<7 {
    assert_eq(out[i], ref_arr[i])
  }
}

///|
test "min_elem_i32 / max_elem_i32 match scalar" {
  let a = make_int_array([5, 1, 9, 2, -3, 7, 0])
  let b = make_int_array([3, 4, 2, 8, -1, 5, 6])
  let out = FixedArray::make(7, 0)
  let ref_arr = FixedArray::make(7, 0)
  min_elem_i32(a, b, out)
  @internal.scalar_min_elem(a, b, ref_arr)
  for i in 0..<7 {
    assert_eq(out[i], ref_arr[i])
  }
  max_elem_i32(a, b, out)
  @internal.scalar_max_elem(a, b, ref_arr)
  for i in 0..<7 {
    assert_eq(out[i], ref_arr[i])
  }
}

///|
test "eq_i32 / lt_i32 / gt_i32 produce -1/0 masks" {
  let a = make_int_array([1, 2, 3, 4, 5, 6, 7])
  let b = make_int_array([1, 5, 3, 2, 9, 6, 0])
  let out = FixedArray::make(7, 0)
  let ref_arr = FixedArray::make(7, 0)
  eq_i32(a, b, out)
  @internal.scalar_eq(a, b, ref_arr)
  for i in 0..<7 {
    assert_eq(out[i], ref_arr[i])
  }
  // eq mask values: -1 (equal) or 0
  assert_eq(out[0], -1)
  assert_eq(out[1], 0)
  lt_i32(a, b, out)
  @internal.scalar_lt(a, b, ref_arr)
  for i in 0..<7 {
    assert_eq(out[i], ref_arr[i])
  }
  gt_i32(a, b, out)
  @internal.scalar_gt(a, b, ref_arr)
  for i in 0..<7 {
    assert_eq(out[i], ref_arr[i])
  }
}

///|
test "where_i32 with computed mask" {
  let a = make_int_array([10, 20, 30, 40, 50, 60, 70])
  let b = make_int_array([1, 2, 3, 4, 5, 6, 7])
  let mask = make_int_array([-1, 0, -1, 0, -1, 0, -1])
  let out = FixedArray::make(7, 0)
  let ref_arr = FixedArray::make(7, 0)
  where_i32(mask, a, b, out)
  @internal.scalar_where(mask, a, b, ref_arr)
  for i in 0..<7 {
    assert_eq(out[i], ref_arr[i])
  }
  assert_eq(out[0], 10)
  assert_eq(out[1], 2)
}

///|
test "where_i32 end-to-end with lt_i32 as mask producer" {
  let a = make_int_array([10, 20, 30, 40, 50])
  let b = make_int_array([15, 5, 30, 100, 25])
  // clamp: out[i] = max(a[i], b[i])  via where(a < b, b, a)
  let mask = FixedArray::make(5, 0)
  let out = FixedArray::make(5, 0)
  lt_i32(a, b, mask) // mask = (a < b)
  where_i32(mask, b, a, out)
  // expected: [15, 20, 30, 100, 50]
  assert_eq(out[0], 15)
  assert_eq(out[1], 20)
  assert_eq(out[2], 30)
  assert_eq(out[3], 100)
  assert_eq(out[4], 50)
}

// --- string ops (H) ---

///|
test "is_ascii true" {
  // "Hello, World! ABCDEFGHIJK" is all ASCII
  let data = make_byte_array([
    72, 101, 108, 108, 111, 44, 32, 87, 111, 114, 108, 100, 33, 32, 65, 66, 67, 68,
    69, 70,
  ])
  inspect(is_ascii(data), content="true")
}

///|
test "is_ascii false (high byte in SIMD chunk)" {
  let data = make_byte_array([
    65, 66, 67, 68, 69, 200, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
  ])
  inspect(is_ascii(data), content="false")
}

///|
test "is_ascii false (high byte in tail)" {
  let data = make_byte_array([
    65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 200,
  ])
  inspect(is_ascii(data), content="false")
}

///|
test "to_lower_ascii matches scalar" {
  let original = [
    72, 101, 108, 108, 111, 44, 32, 87, 111, 114, 76, 68, 33, 32, 65, 66, 67, 68,
    69, 70,
  ]
  let s = make_byte_array(original)
  let r = make_byte_array(original)
  to_lower_ascii(s)
  @internal.scalar_to_lower_ascii(r)
  for i in 0..<s.length() {
    assert_eq(s[i], r[i])
  }
}

///|
test "to_upper_ascii matches scalar" {
  let original = [
    72, 101, 108, 108, 111, 44, 32, 87, 111, 114, 76, 68, 33, 32, 97, 98, 99, 100,
    101, 102,
  ]
  let s = make_byte_array(original)
  let r = make_byte_array(original)
  to_upper_ascii(s)
  @internal.scalar_to_upper_ascii(r)
  for i in 0..<s.length() {
    assert_eq(s[i], r[i])
  }
}

///|
test "to_lower_ascii leaves non-letters alone" {
  let data = make_byte_array([65, 48, 91, 96, 123, 192, 90, 97])
  // 'A'(65)→97, '0'(48) unchanged, '['(91) unchanged, '`'(96) unchanged,
  // '{'(123) unchanged, 0xC0 (>=128) unchanged, 'Z'(90)→122, 'a'(97) unchanged
  to_lower_ascii(data)
  assert_eq(data[0], b'\x61')
  assert_eq(data[1], b'\x30')
  assert_eq(data[2], b'\x5b')
  assert_eq(data[3], b'\x60')
  assert_eq(data[4], b'\x7b')
  assert_eq(data[5], b'\xc0')
  assert_eq(data[6], b'\x7a')
  assert_eq(data[7], b'\x61')
}

// --- UTF-8 validation (I) ---

///|
test "validate_utf8 pure ASCII" {
  let data = make_byte_array([
    72, 101, 108, 108, 111, 44, 32, 87, 111, 114, 108, 100, 33, 32, 49, 50, 51, 52,
    53, 54, 55, 56, 57, 48,
  ])
  inspect(validate_utf8(data), content="true")
}

///|
test "validate_utf8 valid 2-byte (é)" {
  // 'é' = U+00E9 = 0xC3 0xA9
  let data = make_byte_array([72, 0xC3, 0xA9, 108, 108, 111])
  inspect(validate_utf8(data), content="true")
}

///|
test "validate_utf8 valid 3-byte (日)" {
  // '日' = U+65E5 = 0xE6 0x97 0xA5
  let data = make_byte_array([0xE6, 0x97, 0xA5, 0xE6, 0x9C, 0xAC])
  inspect(validate_utf8(data), content="true")
}

///|
test "validate_utf8 valid 4-byte" {
  // U+1F600 (grinning face) = 0xF0 0x9F 0x98 0x80
  let data = make_byte_array([0xF0, 0x9F, 0x98, 0x80])
  inspect(validate_utf8(data), content="true")
}

///|
test "validate_utf8 lone continuation rejected" {
  let data = make_byte_array([72, 0x80, 108])
  inspect(validate_utf8(data), content="false")
}

///|
test "validate_utf8 truncated multi-byte rejected" {
  // 0xC3 expects 1 continuation but stream ends
  let data = make_byte_array([72, 0xC3])
  inspect(validate_utf8(data), content="false")
}

///|
test "validate_utf8 invalid leading byte rejected" {
  // 0xF8 is in the invalid 0xF5..0xFF range
  let data = make_byte_array([72, 0xF8, 0x80])
  inspect(validate_utf8(data), content="false")
}

///|
test "validate_utf8 long ASCII then multi-byte" {
  // 32 ASCII bytes, then a valid 2-byte
  let vals = Array::makei(32, fn(i) { 65 + i % 26 })
  let _ = vals.push(0xC3)
  let _ = vals.push(0xA9)
  let data = make_byte_array(vals)
  inspect(validate_utf8(data), content="true")
}

///|
test "validate_utf8 long ASCII then invalid" {
  let vals = Array::makei(32, fn(i) { 65 + i % 26 })
  let _ = vals.push(0x80) // lone continuation right after ASCII region
  let data = make_byte_array(vals)
  inspect(validate_utf8(data), content="false")
}

///|
test "validate_utf8 matches scalar on mixed input" {
  let data = make_byte_array([
    72, 101, 108, 108, 111, 0xE6, 0x97, 0xA5, 0xE6, 0x9C, 0xAC, 32, 0xC3, 0xA9, 33,
    0xF0, 0x9F, 0x98, 0x80,
  ])
  assert_eq(validate_utf8(data), @internal.scalar_validate_utf8(data))
}

// --- gather / scatter (G) ---

///|
test "gather_i32 basic" {
  let arr = make_int_array([10, 20, 30, 40, 50, 60, 70, 80, 90])
  let indices = make_int_array([3, 0, 7, 8, 1])
  let out = FixedArray::make(5, 0)
  gather_i32(arr, indices, out)
  // expected: arr[3], arr[0], arr[7], arr[8], arr[1] = 40, 10, 80, 90, 20
  assert_eq(out[0], 40)
  assert_eq(out[1], 10)
  assert_eq(out[2], 80)
  assert_eq(out[3], 90)
  assert_eq(out[4], 20)
}

///|
test "gather_i32 matches scalar (large)" {
  let n = 100
  let arr = FixedArray::make(n, 0)
  for i in 0..<n {
    arr[i] = i * 3
  }
  let indices = FixedArray::make(50, 0)
  for i in 0..<50 {
    indices[i] = i * 7 % n
  }
  let out_s = FixedArray::make(50, 0)
  let out_r = FixedArray::make(50, 0)
  gather_i32(arr, indices, out_s)
  @internal.scalar_gather(arr, indices, out_r)
  for i in 0..<50 {
    assert_eq(out_s[i], out_r[i])
  }
}

///|
test "scatter_i32 basic" {
  let out = FixedArray::make(10, 0)
  let indices = make_int_array([0, 3, 7, 1, 9])
  let vals = make_int_array([100, 200, 300, 400, 500])
  scatter_i32(out, indices, vals)
  assert_eq(out[0], 100)
  assert_eq(out[1], 400)
  assert_eq(out[3], 200)
  assert_eq(out[7], 300)
  assert_eq(out[9], 500)
}

// --- div (Int via f64x2 detour) ---

///|
test "div_i32 matches scalar (positive)" {
  let a = make_int_array([100, 81, 64, 49, 36, 25, 16, 9, 4])
  let b = make_int_array([2, 3, 4, 5, 6, 7, 8, 9, 1])
  let out_s = FixedArray::make(9, 0)
  let out_r = FixedArray::make(9, 0)
  div_i32(a, b, out_s)
  @internal.scalar_div(a, b, out_r)
  for i in 0..<9 {
    assert_eq(out_s[i], out_r[i])
  }
}

///|
test "div_i32 matches scalar (negatives, truncation)" {
  let a = make_int_array([-10, -7, 7, -7, 100, -100, 13, -13])
  let b = make_int_array([3, 3, -3, -3, -7, 7, -2, 2])
  let out_s = FixedArray::make(8, 0)
  let out_r = FixedArray::make(8, 0)
  div_i32(a, b, out_s)
  @internal.scalar_div(a, b, out_r)
  for i in 0..<8 {
    assert_eq(out_s[i], out_r[i])
  }
}

// --- sort (F) ---

///|
test "scalar_sort4_int sorts in place" {
  let arr = make_int_array([4, 1, 3, 2])
  @internal.scalar_sort4_int(arr, 0)
  assert_eq(arr[0], 1)
  assert_eq(arr[1], 2)
  assert_eq(arr[2], 3)
  assert_eq(arr[3], 4)
}

///|
test "sort4_i32 matches scalar" {
  let pairs = [
    [4, 1, 3, 2],
    [-5, -3, -1, -10],
    [7, 7, 7, 7],
    [1, 2, 3, 4],
    [4, 3, 2, 1],
    [100, -50, 0, 25],
  ]
  for p in pairs {
    let simd_arr = make_int_array(p)
    let ref_arr = make_int_array(p)
    sort4_i32(simd_arr, 0)
    @internal.scalar_sort4_int(ref_arr, 0)
    for i in 0..<4 {
      assert_eq(simd_arr[i], ref_arr[i])
    }
  }
}

///|
test "sort4_i32 with offset" {
  let arr = make_int_array([99, 4, 1, 3, 2, 88])
  sort4_i32(arr, 1)
  assert_eq(arr[0], 99)
  assert_eq(arr[1], 1)
  assert_eq(arr[2], 2)
  assert_eq(arr[3], 3)
  assert_eq(arr[4], 4)
  assert_eq(arr[5], 88)
}

///|
test "sort_i32 general" {
  let arr = make_int_array([5, 2, 8, 1, 9, 3, 7, 4, 6, 0])
  sort_i32(arr)
  for i in 0..<10 {
    assert_eq(arr[i], i)
  }
}

///|
test "sort_i32 large (multi-pass merge ladder)" {
  // size 1000 - not power of 2, exercises tail handling + ladder copy
  let n = 1000
  let arr = FixedArray::make(n, 0)
  let ref_arr = FixedArray::make(n, 0)
  for i in 0..<n {
    let v = (i * 31 + 17) % 991
    arr[i] = v
    ref_arr[i] = v
  }
  sort_i32(arr)
  @internal.scalar_sort_int(ref_arr)
  for i in 0..<n {
    assert_eq(arr[i], ref_arr[i])
  }
}

///|
test "sort_i32 power of two (1024)" {
  let n = 1024
  let arr = FixedArray::make(n, 0)
  let ref_arr = FixedArray::make(n, 0)
  for i in 0..<n {
    let v = (i * 13 + 7) % 257
    arr[i] = v
    ref_arr[i] = v
  }
  sort_i32(arr)
  @internal.scalar_sort_int(ref_arr)
  for i in 0..<n {
    assert_eq(arr[i], ref_arr[i])
  }
}

///|
test "sort_i32 small odd sizes (1, 2, 3, 5, 7, 11, 17)" {
  let sizes = [1, 2, 3, 5, 7, 11, 17]
  for n in sizes {
    let arr = FixedArray::make(n, 0)
    let ref_arr = FixedArray::make(n, 0)
    for i in 0..<n {
      let v = (i * 7 + 3) % 19 - 10
      arr[i] = v
      ref_arr[i] = v
    }
    sort_i32(arr)
    @internal.scalar_sort_int(ref_arr)
    for i in 0..<n {
      assert_eq(arr[i], ref_arr[i])
    }
  }
}

///|
test "bitonic_merge8_i32 interleaved" {
  // two sorted halves: [1,3,5,7] and [2,4,6,8]
  let arr = make_int_array([1, 3, 5, 7, 2, 4, 6, 8])
  bitonic_merge8_i32(arr, 0)
  for i in 0..<8 {
    assert_eq(arr[i], i + 1)
  }
}

///|
test "bitonic_merge8_i32 disjoint" {
  // first half all smaller
  let arr = make_int_array([1, 2, 3, 4, 5, 6, 7, 8])
  bitonic_merge8_i32(arr, 0)
  for i in 0..<8 {
    assert_eq(arr[i], i + 1)
  }
}

///|
test "bitonic_merge8_i32 second-half smaller" {
  let arr = make_int_array([5, 6, 7, 8, 1, 2, 3, 4])
  bitonic_merge8_i32(arr, 0)
  for i in 0..<8 {
    assert_eq(arr[i], i + 1)
  }
}

///|
test "bitonic_merge8_i32 matches scalar (random pairs)" {
  let cases = [
    [1, 4, 6, 9, 2, 3, 5, 7],
    [2, 2, 4, 8, 3, 3, 5, 9],
    [-5, -3, 0, 7, -10, -2, 1, 11],
    [10, 20, 30, 40, 5, 15, 25, 35],
  ]
  for c in cases {
    let s = make_int_array(c)
    let r = make_int_array(c)
    bitonic_merge8_i32(s, 0)
    @internal.scalar_bitonic_merge8_int(r, 0)
    for i in 0..<8 {
      assert_eq(s[i], r[i])
    }
  }
}

///|
test "bitonic_merge32_i32 interleaved" {
  let arr = FixedArray::make(32, 0)
  for i in 0..<16 {
    arr[i] = i * 2 + 1
  }
  for i in 0..<16 {
    arr[16 + i] = i * 2 + 2
  }
  bitonic_merge32_i32(arr, 0)
  for i in 0..<32 {
    assert_eq(arr[i], i + 1)
  }
}

///|
test "bitonic_merge64_i32 interleaved" {
  let arr = FixedArray::make(64, 0)
  for i in 0..<32 {
    arr[i] = i * 2 + 1
  }
  for i in 0..<32 {
    arr[32 + i] = i * 2 + 2
  }
  bitonic_merge64_i32(arr, 0)
  for i in 0..<64 {
    assert_eq(arr[i], i + 1)
  }
}

///|
test "bitonic_merge64_i32 matches scalar" {
  let h1 = FixedArray::make(32, 0)
  for i in 0..<32 {
    h1[i] = (i * 13 + 7) % 101
  }
  @internal.scalar_sort_int(h1)
  let h2 = FixedArray::make(32, 0)
  for i in 0..<32 {
    h2[i] = (i * 17 + 3) % 97
  }
  @internal.scalar_sort_int(h2)
  let arr_a = FixedArray::make(64, 0)
  let arr_b = FixedArray::make(64, 0)
  for i in 0..<32 {
    arr_a[i] = h1[i]
    arr_b[i] = h1[i]
    arr_a[32 + i] = h2[i]
    arr_b[32 + i] = h2[i]
  }
  bitonic_merge64_i32(arr_a, 0)
  @internal.scalar_bitonic_merge64_int(arr_b, 0)
  for i in 0..<64 {
    assert_eq(arr_a[i], arr_b[i])
  }
}

///|
test "bitonic_merge32_i32 matches scalar" {
  let h1 = FixedArray::make(16, 0)
  for i in 0..<16 {
    h1[i] = (i * 7 + 3) % 53
  }
  @internal.scalar_sort_int(h1)
  let h2 = FixedArray::make(16, 0)
  for i in 0..<16 {
    h2[i] = (i * 11 + 5) % 47
  }
  @internal.scalar_sort_int(h2)
  let arr_a = FixedArray::make(32, 0)
  let arr_b = FixedArray::make(32, 0)
  for i in 0..<16 {
    arr_a[i] = h1[i]
    arr_b[i] = h1[i]
    arr_a[16 + i] = h2[i]
    arr_b[16 + i] = h2[i]
  }
  bitonic_merge32_i32(arr_a, 0)
  @internal.scalar_bitonic_merge32_int(arr_b, 0)
  for i in 0..<32 {
    assert_eq(arr_a[i], arr_b[i])
  }
}

///|
test "bitonic_merge16_i32 interleaved" {
  let arr = make_int_array([
    1, 3, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8, 10, 12, 14, 16,
  ])
  bitonic_merge16_i32(arr, 0)
  for i in 0..<16 {
    assert_eq(arr[i], i + 1)
  }
}

///|
test "bitonic_merge16_i32 matches scalar (random pairs)" {
  let cases = [
    [1, 5, 9, 13, 17, 21, 25, 29, 2, 6, 10, 14, 18, 22, 26, 30],
    [-10, -5, 0, 3, 7, 11, 15, 20, -8, -3, 1, 4, 8, 12, 16, 25],
    [
      100, 200, 300, 400, 500, 600, 700, 800, 50, 150, 250, 350, 450, 550, 650, 750,
    ],
    [1, 1, 2, 3, 4, 5, 6, 6, 1, 2, 2, 3, 4, 5, 6, 7],
  ]
  for c in cases {
    let s = make_int_array(c)
    let r = make_int_array(c)
    bitonic_merge16_i32(s, 0)
    @internal.scalar_bitonic_merge16_int(r, 0)
    for i in 0..<16 {
      assert_eq(s[i], r[i])
    }
  }
}

///|
test "sort16_i32 basic" {
  let arr = make_int_array([
    15, 7, 3, 11, 1, 13, 9, 5, 14, 6, 2, 10, 0, 12, 8, 4,
  ])
  sort16_i32(arr, 0)
  for i in 0..<16 {
    assert_eq(arr[i], i)
  }
}

///|
test "sort16_i32 matches scalar (random)" {
  let arr_s = FixedArray::make(16, 0)
  let arr_r = FixedArray::make(16, 0)
  for i in 0..<16 {
    let v = (i * 17 + 5) % 31
    arr_s[i] = v
    arr_r[i] = v
  }
  sort16_i32(arr_s, 0)
  @internal.scalar_sort16_int(arr_r, 0)
  for i in 0..<16 {
    assert_eq(arr_s[i], arr_r[i])
  }
}

///|
test "sort16_i32 with offset surrounded by sentinels" {
  let arr = make_int_array([
    99, 15, 7, 3, 11, 1, 13, 9, 5, 14, 6, 2, 10, 0, 12, 8, 4, 88,
  ])
  sort16_i32(arr, 1)
  assert_eq(arr[0], 99)
  for i in 0..<16 {
    assert_eq(arr[i + 1], i)
  }
  assert_eq(arr[17], 88)
}

// --- f32 over byte buffer ---

///|
fn make_f32_buf(vals : Array[Float]) -> FixedArray[Byte] {
  let n = vals.length()
  let buf = f32_buf_alloc(n)
  for i in 0..<n {
    f32_buf_set(buf, i, vals[i])
  }
  buf
}

///|
fn assert_f32_close(
  a : Float,
  b : Float,
  eps? : Float = (1.0e-5 : Float),
) -> Unit {
  let d = (a - b).abs()
  if d > eps {
    abort(
      "expected ~" +
      b.to_double().to_string() +
      " got " +
      a.to_double().to_string(),
    )
  }
}

///|
test "f32 buf round-trip" {
  let buf = f32_buf_alloc(3)
  f32_buf_set(buf, 0, (1.5 : Float))
  f32_buf_set(buf, 1, (2.25 : Float))
  f32_buf_set(buf, 2, (-3.5 : Float))
  assert_f32_close(f32_buf_get(buf, 0), (1.5 : Float))
  assert_f32_close(f32_buf_get(buf, 1), (2.25 : Float))
  assert_f32_close(f32_buf_get(buf, 2), (-3.5 : Float))
}

///|
test "add_f32 matches scalar" {
  let a = make_f32_buf([
    (1.5 : Float),
    (2.0 : Float),
    (3.25 : Float),
    (4.0 : Float),
    (5.5 : Float),
    (6.0 : Float),
    (7.125 : Float),
  ])
  let b = make_f32_buf([
    (10.0 : Float),
    (20.5 : Float),
    (30.0 : Float),
    (40.25 : Float),
    (50.0 : Float),
    (60.5 : Float),
    (70.0 : Float),
  ])
  let out_s = f32_buf_alloc(7)
  let out_r = f32_buf_alloc(7)
  add_f32(a, b, out_s, 7)
  @internal.scalar_add_f32(a, b, out_r, 7)
  for i in 0..<7 {
    assert_f32_close(f32_buf_get(out_s, i), f32_buf_get(out_r, i))
  }
}

///|
test "mul_f32 matches scalar" {
  let a = make_f32_buf([
    (1.5 : Float),
    (2.0 : Float),
    (3.25 : Float),
    (4.0 : Float),
    (5.5 : Float),
  ])
  let b = make_f32_buf([
    (2.0 : Float),
    (3.0 : Float),
    (4.0 : Float),
    (5.0 : Float),
    (6.0 : Float),
  ])
  let out_s = f32_buf_alloc(5)
  let out_r = f32_buf_alloc(5)
  mul_f32(a, b, out_s, 5)
  @internal.scalar_mul_f32(a, b, out_r, 5)
  for i in 0..<5 {
    assert_f32_close(f32_buf_get(out_s, i), f32_buf_get(out_r, i))
  }
}

///|
test "sum_f32" {
  let a = make_f32_buf([
    (1.5 : Float),
    (2.5 : Float),
    (3.5 : Float),
    (4.5 : Float),
    (5.5 : Float),
  ])
  // 1.5+2.5+3.5+4.5+5.5 = 17.5
  assert_f32_close(sum_f32(a, 5), (17.5 : Float))
}

///|
test "dot_f32" {
  let a = make_f32_buf([
    (1.0 : Float),
    (2.0 : Float),
    (3.0 : Float),
    (4.0 : Float),
    (5.0 : Float),
  ])
  let b = make_f32_buf([
    (2.0 : Float),
    (3.0 : Float),
    (4.0 : Float),
    (5.0 : Float),
    (6.0 : Float),
  ])
  // 2+6+12+20+30 = 70
  assert_f32_close(dot_f32(a, b, 5), (70.0 : Float))
}

///|
test "sub_f32 / div_f32 / sqrt_f32 match scalar" {
  let a = make_f32_buf([
    (16.0 : Float),
    (25.0 : Float),
    (4.0 : Float),
    (9.0 : Float),
    (1.0 : Float),
  ])
  let b = make_f32_buf([
    (2.0 : Float),
    (5.0 : Float),
    (2.0 : Float),
    (3.0 : Float),
    (1.0 : Float),
  ])
  let out_s = f32_buf_alloc(5)
  let out_r = f32_buf_alloc(5)
  sub_f32(a, b, out_s, 5)
  @internal.scalar_sub_f32(a, b, out_r, 5)
  for i in 0..<5 {
    assert_f32_close(f32_buf_get(out_s, i), f32_buf_get(out_r, i))
  }
  div_f32(a, b, out_s, 5)
  @internal.scalar_div_f32(a, b, out_r, 5)
  for i in 0..<5 {
    assert_f32_close(f32_buf_get(out_s, i), f32_buf_get(out_r, i))
  }
  sqrt_f32(a, out_s, 5)
  @internal.scalar_sqrt_f32(a, out_r, 5)
  for i in 0..<5 {
    assert_f32_close(f32_buf_get(out_s, i), f32_buf_get(out_r, i))
  }
}

///|
test "min_elem_f32 / max_elem_f32 element-wise" {
  let a = make_f32_buf([
    (1.0 : Float),
    (5.0 : Float),
    (3.0 : Float),
    (8.0 : Float),
    (2.0 : Float),
  ])
  let b = make_f32_buf([
    (4.0 : Float),
    (2.0 : Float),
    (7.0 : Float),
    (6.0 : Float),
    (9.0 : Float),
  ])
  let out = f32_buf_alloc(5)
  min_elem_f32(a, b, out, 5)
  assert_f32_close(f32_buf_get(out, 0), (1.0 : Float))
  assert_f32_close(f32_buf_get(out, 1), (2.0 : Float))
  assert_f32_close(f32_buf_get(out, 3), (6.0 : Float))
  max_elem_f32(a, b, out, 5)
  assert_f32_close(f32_buf_get(out, 0), (4.0 : Float))
  assert_f32_close(f32_buf_get(out, 1), (5.0 : Float))
  assert_f32_close(f32_buf_get(out, 4), (9.0 : Float))
}

// --- reductions + scan (D) ---

///|
test "prod_i32 basic + matches scalar" {
  let arr = make_int_array([2, 3, 4, 5, 6])
  // 2*3*4*5*6 = 720
  inspect(prod_i32(arr), content="720")
  assert_eq(prod_i32(arr), @internal.scalar_prod(arr))
}

///|
test "prod_i32 empty" {
  let arr : FixedArray[Int] = FixedArray::make(0, 0)
  inspect(prod_i32(arr), content="1")
}

///|
test "mean_f64 / var_f64" {
  let arr = make_f64_array([1.0, 2.0, 3.0, 4.0, 5.0])
  // mean = 3, var (population) = ((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 5
  //      = (4 + 1 + 0 + 1 + 4) / 5 = 2
  assert_f64_close(mean_f64(arr), 3.0)
  assert_f64_close(var_f64(arr), 2.0)
}

///|
test "var_f64 matches scalar (larger)" {
  let n = 21
  let arr = FixedArray::make(n, 0.0)
  for i in 0..<n {
    arr[i] = (i + 1).to_double() * 0.5 - 3.0
  }
  assert_f64_close(var_f64(arr), @internal.scalar_var_f64(arr))
}

///|
test "count_nonzero_i32" {
  let arr = make_int_array([0, 1, 0, 2, 0, 3, 0, 4, 0, 5])
  inspect(count_nonzero_i32(arr), content="5")
}

///|
test "all_i32 / any_i32" {
  let all_ones = make_int_array([1, 2, 3, 4, 5, 6, 7])
  let has_zero = make_int_array([1, 2, 0, 4, 5])
  let all_zero = make_int_array([0, 0, 0, 0, 0])
  inspect(all_i32(all_ones), content="true")
  inspect(all_i32(has_zero), content="false")
  inspect(all_i32(all_zero), content="false")
  inspect(any_i32(all_ones), content="true")
  inspect(any_i32(has_zero), content="true")
  inspect(any_i32(all_zero), content="false")
}

///|
test "cumsum_i32 / cumprod_i32 (scalar fallback)" {
  let arr = make_int_array([1, 2, 3, 4, 5])
  let out = FixedArray::make(5, 0)
  cumsum_i32(arr, out)
  assert_eq(out[0], 1)
  assert_eq(out[1], 3)
  assert_eq(out[2], 6)
  assert_eq(out[3], 10)
  assert_eq(out[4], 15)
  cumprod_i32(arr, out)
  assert_eq(out[0], 1)
  assert_eq(out[1], 2)
  assert_eq(out[2], 6)
  assert_eq(out[3], 24)
  assert_eq(out[4], 120)
}

// --- linear algebra (C) ---

///|
test "matmul_f64 vs scalar (2x3 * 3x2)" {
  // A = [[1,2,3],[4,5,6]]  (2x3)
  // B = [[7,8],[9,10],[11,12]]  (3x2)
  // B^T (row-major) = [[7,9,11],[8,10,12]]  (2x3 in B^T layout, i.e. 2 rows of length 3)
  // Expected C = A * B = [[58,64],[139,154]]
  let a = make_f64_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let b_t = make_f64_array([7.0, 9.0, 11.0, 8.0, 10.0, 12.0])
  let c = FixedArray::make(4, 0.0)
  matmul_f64(a, b_t, c, 2, 3, 2)
  assert_f64_close(c[0], 58.0)
  assert_f64_close(c[1], 64.0)
  assert_f64_close(c[2], 139.0)
  assert_f64_close(c[3], 154.0)
}

///|
test "matmul_f64 matches scalar (5x7 * 7x4)" {
  let m = 5
  let k = 7
  let n = 4
  let a = FixedArray::make(m * k, 0.0)
  let b_t = FixedArray::make(n * k, 0.0)
  for i in 0..<(m * k) {
    a[i] = (i + 1).to_double() * 0.5
  }
  for j in 0..<(n * k) {
    b_t[j] = ((j * 3 + 1) % 17).to_double()
  }
  let c_simd = FixedArray::make(m * n, 0.0)
  let c_scalar = FixedArray::make(m * n, 0.0)
  matmul_f64(a, b_t, c_simd, m, k, n)
  @internal.scalar_matmul_f64(a, b_t, c_scalar, m, k, n)
  for i in 0..<(m * n) {
    assert_f64_close(c_simd[i], c_scalar[i])
  }
}

///|
test "gemv_f64 vs scalar" {
  // A = [[1,2,3],[4,5,6]] (2x3), x = [1,2,3]
  // y = [1+4+9, 4+10+18] = [14, 32]
  let a = make_f64_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let x = make_f64_array([1.0, 2.0, 3.0])
  let y = FixedArray::make(2, 0.0)
  gemv_f64(a, x, y, 2, 3)
  assert_f64_close(y[0], 14.0)
  assert_f64_close(y[1], 32.0)
}

///|
test "gemv_f64 matches scalar (larger)" {
  let m = 10
  let n = 13
  let a = FixedArray::make(m * n, 0.0)
  let x = FixedArray::make(n, 0.0)
  for i in 0..<(m * n) {
    a[i] = (i + 1).to_double() * 0.25
  }
  for j in 0..<n {
    x[j] = (j + 1).to_double()
  }
  let y_simd = FixedArray::make(m, 0.0)
  let y_scalar = FixedArray::make(m, 0.0)
  gemv_f64(a, x, y_simd, m, n)
  @internal.scalar_gemv_f64(a, x, y_scalar, m, n)
  for i in 0..<m {
    assert_f64_close(y_simd[i], y_scalar[i])
  }
}

///|
test "transpose_f64 (3x4)" {
  // A = [[1,2,3,4],[5,6,7,8],[9,10,11,12]] (3x4)
  // A^T = [[1,5,9],[2,6,10],[3,7,11],[4,8,12]] (4x3)
  let a = make_f64_array([
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
  ])
  let out = FixedArray::make(12, 0.0)
  transpose_f64(a, out, 3, 4)
  let expected = make_f64_array([
    1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
  ])
  for i in 0..<12 {
    assert_f64_close(out[i], expected[i])
  }
}

// --- Double (f64) ops ---

///|
fn make_f64_array(vals : Array[Double]) -> FixedArray[Double] {
  let arr = FixedArray::make(vals.length(), 0.0)
  for i, v in vals {
    arr[i] = v
  }
  arr
}

///|
fn assert_f64_close(a : Double, b : Double, eps? : Double = 1.0e-9) -> Unit {
  let d = (a - b).abs()
  if d > eps {
    abort("expected ~" + b.to_string() + " got " + a.to_string())
  }
}

///|
test "add_f64 matches scalar" {
  let a = make_f64_array([1.5, 2.0, 3.25, 4.0, 5.5, 6.0, 7.125])
  let b = make_f64_array([10.0, 20.5, 30.0, 40.25, 50.0, 60.5, 70.0])
  let out_simd = FixedArray::make(7, 0.0)
  let out_scalar = FixedArray::make(7, 0.0)
  add_f64(a, b, out_simd)
  @internal.scalar_add_f64(a, b, out_scalar)
  for i in 0..<7 {
    assert_f64_close(out_simd[i], out_scalar[i])
  }
}

///|
test "sub_i32/mul/div_f64 match scalar" {
  let a = make_f64_array([1.0, 4.0, 9.0, 16.0, 25.0])
  let b = make_f64_array([2.0, 2.0, 3.0, 4.0, 5.0])
  let out = FixedArray::make(5, 0.0)
  let ref_arr = FixedArray::make(5, 0.0)
  sub_f64(a, b, out)
  @internal.scalar_sub_f64(a, b, ref_arr)
  for i in 0..<5 {
    assert_f64_close(out[i], ref_arr[i])
  }
  mul_f64(a, b, out)
  @internal.scalar_mul_f64(a, b, ref_arr)
  for i in 0..<5 {
    assert_f64_close(out[i], ref_arr[i])
  }
  div_f64(a, b, out)
  @internal.scalar_div_f64(a, b, ref_arr)
  for i in 0..<5 {
    assert_f64_close(out[i], ref_arr[i])
  }
}

///|
test "sqrt_f64 matches scalar" {
  let a = make_f64_array([1.0, 4.0, 9.0, 16.0, 25.0])
  let out = FixedArray::make(5, 0.0)
  sqrt_f64(a, out)
  assert_f64_close(out[0], 1.0)
  assert_f64_close(out[1], 2.0)
  assert_f64_close(out[2], 3.0)
  assert_f64_close(out[3], 4.0)
  assert_f64_close(out[4], 5.0)
}

///|
test "min_elem_f64 / max_elem_f64 element-wise" {
  let a = make_f64_array([1.0, 5.0, 3.0, 8.0, 2.0])
  let b = make_f64_array([4.0, 2.0, 7.0, 6.0, 9.0])
  let out = FixedArray::make(5, 0.0)
  min_elem_f64(a, b, out)
  assert_f64_close(out[0], 1.0)
  assert_f64_close(out[1], 2.0)
  assert_f64_close(out[2], 3.0)
  assert_f64_close(out[3], 6.0)
  assert_f64_close(out[4], 2.0)
  max_elem_f64(a, b, out)
  assert_f64_close(out[0], 4.0)
  assert_f64_close(out[1], 5.0)
  assert_f64_close(out[2], 7.0)
  assert_f64_close(out[3], 8.0)
  assert_f64_close(out[4], 9.0)
}

///|
test "sum_f64 matches scalar" {
  let a = make_f64_array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
  assert_f64_close(sum_f64(a), @internal.scalar_sum_f64(a))
}

///|
test "sum_f64 empty" {
  let arr : FixedArray[Double] = FixedArray::make(0, 0.0)
  assert_f64_close(sum_f64(arr), 0.0)
}

///|
test "dot_f64 matches scalar" {
  let a = make_f64_array([1.0, 2.0, 3.0, 4.0, 5.0])
  let b = make_f64_array([2.0, 3.0, 4.0, 5.0, 6.0])
  // 2 + 6 + 12 + 20 + 30 = 70
  assert_f64_close(dot_f64(a, b), 70.0)
}

///|
test "validate_utf8_strict: valid sequences" {
  // ASCII, é (C3 A9), € (E2 82 AC), 😀 (F0 9F 98 80)
  assert_true(validate_utf8_strict([b'h', b'e', b'l', b'l', b'o']))
  assert_true(validate_utf8_strict([b'\xC3', b'\xA9']))
  assert_true(validate_utf8_strict([b'\xE2', b'\x82', b'\xAC']))
  assert_true(validate_utf8_strict([b'\xF0', b'\x9F', b'\x98', b'\x80']))
}

///|
test "validate_utf8_strict: rejects overlong / surrogate / >10FFFF" {
  // overlong 2-byte C0 80 — both strict and lax reject (b < 0xC2)
  assert_false(validate_utf8_strict([b'\xC0', b'\x80']))
  // overlong 3-byte E0 80 80 — strict rejects; lax (structural) accepts
  assert_false(validate_utf8_strict([b'\xE0', b'\x80', b'\x80']))
  assert_true(validate_utf8([b'\xE0', b'\x80', b'\x80']))
  // surrogate ED A0 80 (U+D800) — strict rejects; lax accepts
  assert_false(validate_utf8_strict([b'\xED', b'\xA0', b'\x80']))
  assert_true(validate_utf8([b'\xED', b'\xA0', b'\x80']))
  // > U+10FFFF: F4 90 80 80 — strict rejects; lax accepts
  assert_false(validate_utf8_strict([b'\xF4', b'\x90', b'\x80', b'\x80']))
  assert_true(validate_utf8([b'\xF4', b'\x90', b'\x80', b'\x80']))
  // truncated 3-byte — both reject
  assert_false(validate_utf8_strict([b'\xE2', b'\x82']))
}

///|
test "validate_utf8_strict: after ASCII SIMD-skip region" {
  // >16 ASCII bytes then a surrogate -> strict scalar pass starts past the
  // SIMD-skipped chunks and must still catch it.
  let ok = FixedArray::make(32, b'A')
  assert_true(validate_utf8_strict(ok))
  let bad = FixedArray::make(35, b'A')
  bad[32] = b'\xED'
  bad[33] = b'\xA0'
  bad[34] = b'\x80'
  assert_false(validate_utf8_strict(bad))
}
