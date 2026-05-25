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

// --- scalar_sum ---

///|
test "scalar_sum empty" {
  let arr : FixedArray[Int] = FixedArray::make(0, 0)
  inspect(scalar_sum(arr), content="0")
}

///|
test "scalar_sum basic" {
  let arr = make_int_array([1, 2, 3, 4, 5])
  inspect(scalar_sum(arr), content="15")
}

///|
test "scalar_sum with negatives" {
  let arr = make_int_array([10, -3, 7, -2])
  inspect(scalar_sum(arr), content="12")
}

// --- simd_sum ---

///|
test "simd_sum empty" {
  let arr : FixedArray[Int] = FixedArray::make(0, 0)
  inspect(simd_sum(arr), content="0")
}

///|
test "simd_sum basic" {
  let arr = make_int_array([1, 2, 3, 4, 5, 6, 7, 8])
  inspect(simd_sum(arr), content="36")
}

///|
test "simd_sum non-aligned" {
  let arr = make_int_array([1, 2, 3, 4, 5])
  inspect(simd_sum(arr), content="15")
}

///|
test "simd_sum matches scalar" {
  let arr = make_int_array([10, 20, 30, 40, 50, 60, 70])
  inspect(simd_sum(arr) == scalar_sum(arr), content="true")
}

// --- scalar_dot_product ---

///|
test "scalar_dot_product basic" {
  let a = make_int_array([1, 2, 3])
  let b = make_int_array([4, 5, 6])
  inspect(scalar_dot_product(a, b), content="32")
}

// --- simd_dot_product ---

///|
test "simd_dot_product basic" {
  let a = make_int_array([1, 2, 3, 4])
  let b = make_int_array([5, 6, 7, 8])
  inspect(simd_dot_product(a, b), content="70")
}

///|
test "simd_dot_product non-aligned" {
  let a = make_int_array([1, 2, 3, 4, 5])
  let b = make_int_array([2, 3, 4, 5, 6])
  inspect(simd_dot_product(a, b) == scalar_dot_product(a, b), content="true")
}

// --- scalar_add ---

///|
test "scalar_add basic" {
  let a = make_int_array([1, 2, 3, 4])
  let b = make_int_array([10, 20, 30, 40])
  let out = FixedArray::make(4, 0)
  scalar_add(a, b, out)
  inspect(out[0], content="11")
  inspect(out[3], content="44")
}

// --- simd_add ---

///|
test "simd_add basic" {
  let a = make_int_array([1, 2, 3, 4, 5, 6, 7, 8])
  let b = make_int_array([10, 20, 30, 40, 50, 60, 70, 80])
  let out = FixedArray::make(8, 0)
  simd_add(a, b, out)
  inspect(out[0], content="11")
  inspect(out[7], content="88")
}

///|
test "simd_add non-aligned" {
  let a = make_int_array([1, 2, 3, 4, 5])
  let b = make_int_array([10, 20, 30, 40, 50])
  let out = FixedArray::make(5, 0)
  let expected = FixedArray::make(5, 0)
  simd_add(a, b, out)
  scalar_add(a, b, expected)
  for i in 0..<5 {
    assert_eq(out[i], expected[i])
  }
}

// --- scalar_adler32 ---

///|
test "scalar_adler32 empty" {
  let data : FixedArray[Byte] = FixedArray::make(0, b'\x00')
  inspect(scalar_adler32(data), content="1")
}

///|
test "scalar_adler32 wikipedia" {
  // "Wikipedia" -> Adler-32 = 0x11E60398
  let data = make_byte_array([87, 105, 107, 105, 112, 101, 100, 105, 97])
  inspect(scalar_adler32(data), content="300286872")
}

// --- simd_adler32 ---

///|
test "simd_adler32 matches scalar" {
  let data = make_byte_array([87, 105, 107, 105, 112, 101, 100, 105, 97])
  inspect(simd_adler32(data) == scalar_adler32(data), content="true")
}

///|
test "simd_adler32 empty" {
  let data : FixedArray[Byte] = FixedArray::make(0, b'\x00')
  inspect(simd_adler32(data), content="1")
}

// --- min / max / argmin / argmax ---

///|
test "scalar_min/max basic" {
  let arr = make_int_array([5, -3, 8, -1, 2, 7])
  inspect(scalar_min(arr), content="-3")
  inspect(scalar_max(arr), content="8")
}

///|
test "scalar_argmin/argmax basic" {
  let arr = make_int_array([5, -3, 8, -1, 2, 7])
  inspect(scalar_argmin(arr), content="1")
  inspect(scalar_argmax(arr), content="2")
}

///|
test "simd_min/max match scalar" {
  let arr = make_int_array([5, -3, 8, -1, 2, 7, 100, -50, 12, 4, 0, 9])
  assert_eq(simd_min(arr), scalar_min(arr))
  assert_eq(simd_max(arr), scalar_max(arr))
}

///|
test "simd_argmin/argmax match scalar" {
  let arr = make_int_array([5, -3, 8, -1, 2, 7, 100, -50, 12, 4, 0, 9])
  assert_eq(simd_argmin(arr), scalar_argmin(arr))
  assert_eq(simd_argmax(arr), scalar_argmax(arr))
}

///|
test "simd_min/max non-aligned" {
  let arr = make_int_array([1, 2, 3, 4, 5])
  assert_eq(simd_min(arr), 1)
  assert_eq(simd_max(arr), 5)
}

///|
test "simd_argmin/argmax with duplicates picks first" {
  let arr = make_int_array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])
  // first min (1) at index 1, first max (9) at index 5
  assert_eq(simd_argmin(arr), 1)
  assert_eq(simd_argmax(arr), 5)
}

///|
test "simd_min/max empty" {
  let arr : FixedArray[Int] = FixedArray::make(0, 0)
  inspect(simd_min(arr), content="0")
  inspect(simd_max(arr), content="0")
  inspect(simd_argmin(arr), content="-1")
  inspect(simd_argmax(arr), content="-1")
}

// --- saxpy ---

///|
test "simd_saxpy matches scalar" {
  let a = make_int_array([1, 2, 3, 4, 5, 6, 7, 8, 9])
  let b = make_int_array([10, 20, 30, 40, 50, 60, 70, 80, 90])
  let out_simd = FixedArray::make(9, 0)
  let out_scalar = FixedArray::make(9, 0)
  simd_saxpy(3, a, b, out_simd)
  scalar_saxpy(3, a, b, out_scalar)
  for i in 0..<9 {
    assert_eq(out_simd[i], out_scalar[i])
  }
}

///|
test "simd_saxpy basic value" {
  let a = make_int_array([1, 2, 3, 4])
  let b = make_int_array([10, 20, 30, 40])
  let out = FixedArray::make(4, 0)
  simd_saxpy(2, a, b, out)
  // 2*1+10=12, 2*2+20=24, 2*3+30=36, 2*4+40=48
  assert_eq(out[0], 12)
  assert_eq(out[3], 48)
}

// --- popcount_bytes ---

///|
test "scalar_popcount_bytes basic" {
  let data = make_byte_array([0, 1, 2, 3, 255])
  // 0+1+1+2+8 = 12
  inspect(scalar_popcount_bytes(data), content="12")
}

///|
test "simd_popcount_bytes matches scalar" {
  let vals = Array::makei(40, fn(i) { (i * 13 + 7) % 256 })
  let data = make_byte_array(vals)
  assert_eq(simd_popcount_bytes(data), scalar_popcount_bytes(data))
}

///|
test "simd_popcount_bytes empty" {
  let data : FixedArray[Byte] = FixedArray::make(0, b'\x00')
  inspect(simd_popcount_bytes(data), content="0")
}

// --- byte primitives ---

///|
test "simd_memcpy_bytes basic" {
  let src = make_byte_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
  let dst = FixedArray::make(20, b'\x00')
  simd_memcpy_bytes(src, 0, dst, 0, 20)
  for i in 0..<20 {
    assert_eq(dst[i], src[i])
  }
}

///|
test "simd_memcpy_bytes with offsets" {
  let src = make_byte_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
  let dst = FixedArray::make(15, b'\xff')
  simd_memcpy_bytes(src, 2, dst, 3, 6)
  // dst[3..9] = src[2..8] = 3,4,5,6,7,8; rest unchanged
  assert_eq(dst[0], b'\xff')
  assert_eq(dst[3], b'\x03')
  assert_eq(dst[8], b'\x08')
  assert_eq(dst[9], b'\xff')
}

///|
test "simd_memset_bytes basic" {
  let dst = FixedArray::make(20, b'\x00')
  simd_memset_bytes(dst, 0, 20, b'\x5a')
  for i in 0..<20 {
    assert_eq(dst[i], b'\x5a')
  }
}

///|
test "simd_memset_bytes partial" {
  let dst = FixedArray::make(20, b'\x00')
  simd_memset_bytes(dst, 5, 10, b'\x42')
  assert_eq(dst[4], b'\x00')
  assert_eq(dst[5], b'\x42')
  assert_eq(dst[14], b'\x42')
  assert_eq(dst[15], b'\x00')
}

///|
test "simd_equal_bytes true" {
  let a = make_byte_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
  let b = make_byte_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
  inspect(simd_equal_bytes(a, b, 17), content="true")
}

///|
test "simd_equal_bytes false at tail" {
  let a = make_byte_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 99])
  let b = make_byte_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
  inspect(simd_equal_bytes(a, b, 17), content="false")
}

///|
test "simd_equal_bytes false at first SIMD block" {
  let a = make_byte_array([1, 2, 9, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
  let b = make_byte_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
  inspect(simd_equal_bytes(a, b, 17), content="false")
}

///|
test "simd_find_byte found in SIMD block" {
  let data = make_byte_array([0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
  inspect(simd_find_byte(data, b'\x07'), content="5")
}

///|
test "simd_find_byte found in tail" {
  let data = make_byte_array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9])
  inspect(simd_find_byte(data, b'\x09'), content="16")
}

///|
test "simd_find_byte not found" {
  let data = make_byte_array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
  inspect(simd_find_byte(data, b'\x07'), content="-1")
}

///|
test "simd_count_byte matches scalar" {
  let vals = Array::makei(50, fn(i) { if i % 3 == 0 { 7 } else { i % 256 } })
  let data = make_byte_array(vals)
  assert_eq(simd_count_byte(data, b'\x07'), scalar_count_byte(data, b'\x07'))
}

// --- element-wise Int ops (B) ---

///|
test "simd_sub matches scalar" {
  let a = make_int_array([10, 20, 30, 40, 50, 60, 70])
  let b = make_int_array([1, 2, 3, 4, 5, 6, 7])
  let out = FixedArray::make(7, 0)
  let ref_arr = FixedArray::make(7, 0)
  simd_sub(a, b, out)
  scalar_sub(a, b, ref_arr)
  for i in 0..<7 {
    assert_eq(out[i], ref_arr[i])
  }
}

///|
test "simd_mul matches scalar" {
  let a = make_int_array([1, 2, 3, 4, 5, 6, 7])
  let b = make_int_array([10, 20, 30, 40, 50, 60, 70])
  let out = FixedArray::make(7, 0)
  let ref_arr = FixedArray::make(7, 0)
  simd_mul(a, b, out)
  scalar_mul(a, b, ref_arr)
  for i in 0..<7 {
    assert_eq(out[i], ref_arr[i])
  }
}

///|
test "simd_neg matches scalar" {
  let a = make_int_array([5, -3, 0, 17, -100, 42, -1])
  let out = FixedArray::make(7, 0)
  let ref_arr = FixedArray::make(7, 0)
  simd_neg(a, out)
  scalar_neg(a, ref_arr)
  for i in 0..<7 {
    assert_eq(out[i], ref_arr[i])
  }
}

///|
test "simd_abs matches scalar" {
  let a = make_int_array([5, -3, 0, 17, -100, 42, -1])
  let out = FixedArray::make(7, 0)
  let ref_arr = FixedArray::make(7, 0)
  simd_abs(a, out)
  scalar_abs(a, ref_arr)
  for i in 0..<7 {
    assert_eq(out[i], ref_arr[i])
  }
}

///|
test "simd_min_elem / simd_max_elem match scalar" {
  let a = make_int_array([5, 1, 9, 2, -3, 7, 0])
  let b = make_int_array([3, 4, 2, 8, -1, 5, 6])
  let out = FixedArray::make(7, 0)
  let ref_arr = FixedArray::make(7, 0)
  simd_min_elem(a, b, out)
  scalar_min_elem(a, b, ref_arr)
  for i in 0..<7 {
    assert_eq(out[i], ref_arr[i])
  }
  simd_max_elem(a, b, out)
  scalar_max_elem(a, b, ref_arr)
  for i in 0..<7 {
    assert_eq(out[i], ref_arr[i])
  }
}

///|
test "simd_eq / simd_lt / simd_gt produce -1/0 masks" {
  let a = make_int_array([1, 2, 3, 4, 5, 6, 7])
  let b = make_int_array([1, 5, 3, 2, 9, 6, 0])
  let out = FixedArray::make(7, 0)
  let ref_arr = FixedArray::make(7, 0)
  simd_eq(a, b, out)
  scalar_eq(a, b, ref_arr)
  for i in 0..<7 {
    assert_eq(out[i], ref_arr[i])
  }
  // eq mask values: -1 (equal) or 0
  assert_eq(out[0], -1)
  assert_eq(out[1], 0)
  simd_lt(a, b, out)
  scalar_lt(a, b, ref_arr)
  for i in 0..<7 {
    assert_eq(out[i], ref_arr[i])
  }
  simd_gt(a, b, out)
  scalar_gt(a, b, ref_arr)
  for i in 0..<7 {
    assert_eq(out[i], ref_arr[i])
  }
}

///|
test "simd_where with computed mask" {
  let a = make_int_array([10, 20, 30, 40, 50, 60, 70])
  let b = make_int_array([1, 2, 3, 4, 5, 6, 7])
  let mask = make_int_array([-1, 0, -1, 0, -1, 0, -1])
  let out = FixedArray::make(7, 0)
  let ref_arr = FixedArray::make(7, 0)
  simd_where(mask, a, b, out)
  scalar_where(mask, a, b, ref_arr)
  for i in 0..<7 {
    assert_eq(out[i], ref_arr[i])
  }
  assert_eq(out[0], 10)
  assert_eq(out[1], 2)
}

///|
test "simd_where end-to-end with simd_lt as mask producer" {
  let a = make_int_array([10, 20, 30, 40, 50])
  let b = make_int_array([15, 5, 30, 100, 25])
  // clamp: out[i] = max(a[i], b[i])  via where(a < b, b, a)
  let mask = FixedArray::make(5, 0)
  let out = FixedArray::make(5, 0)
  simd_lt(a, b, mask) // mask = (a < b)
  simd_where(mask, b, a, out)
  // expected: [15, 20, 30, 100, 50]
  assert_eq(out[0], 15)
  assert_eq(out[1], 20)
  assert_eq(out[2], 30)
  assert_eq(out[3], 100)
  assert_eq(out[4], 50)
}

// --- reductions + scan (D) ---

///|
test "simd_prod basic + matches scalar" {
  let arr = make_int_array([2, 3, 4, 5, 6])
  // 2*3*4*5*6 = 720
  inspect(simd_prod(arr), content="720")
  assert_eq(simd_prod(arr), scalar_prod(arr))
}

///|
test "simd_prod empty" {
  let arr : FixedArray[Int] = FixedArray::make(0, 0)
  inspect(simd_prod(arr), content="1")
}

///|
test "simd_mean_f64 / simd_var_f64" {
  let arr = make_f64_array([1.0, 2.0, 3.0, 4.0, 5.0])
  // mean = 3, var (population) = ((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 5
  //      = (4 + 1 + 0 + 1 + 4) / 5 = 2
  assert_f64_close(simd_mean_f64(arr), 3.0)
  assert_f64_close(simd_var_f64(arr), 2.0)
}

///|
test "simd_var_f64 matches scalar (larger)" {
  let n = 21
  let arr = FixedArray::make(n, 0.0)
  for i in 0..<n {
    arr[i] = (i + 1).to_double() * 0.5 - 3.0
  }
  assert_f64_close(simd_var_f64(arr), scalar_var_f64(arr))
}

///|
test "simd_count_nonzero" {
  let arr = make_int_array([0, 1, 0, 2, 0, 3, 0, 4, 0, 5])
  inspect(simd_count_nonzero(arr), content="5")
}

///|
test "simd_all / simd_any" {
  let all_ones = make_int_array([1, 2, 3, 4, 5, 6, 7])
  let has_zero = make_int_array([1, 2, 0, 4, 5])
  let all_zero = make_int_array([0, 0, 0, 0, 0])
  inspect(simd_all(all_ones), content="true")
  inspect(simd_all(has_zero), content="false")
  inspect(simd_all(all_zero), content="false")
  inspect(simd_any(all_ones), content="true")
  inspect(simd_any(has_zero), content="true")
  inspect(simd_any(all_zero), content="false")
}

///|
test "simd_cumsum / simd_cumprod (scalar fallback)" {
  let arr = make_int_array([1, 2, 3, 4, 5])
  let out = FixedArray::make(5, 0)
  simd_cumsum(arr, out)
  assert_eq(out[0], 1)
  assert_eq(out[1], 3)
  assert_eq(out[2], 6)
  assert_eq(out[3], 10)
  assert_eq(out[4], 15)
  simd_cumprod(arr, out)
  assert_eq(out[0], 1)
  assert_eq(out[1], 2)
  assert_eq(out[2], 6)
  assert_eq(out[3], 24)
  assert_eq(out[4], 120)
}

// --- linear algebra (C) ---

///|
test "simd_matmul_f64 vs scalar (2x3 * 3x2)" {
  // A = [[1,2,3],[4,5,6]]  (2x3)
  // B = [[7,8],[9,10],[11,12]]  (3x2)
  // B^T (row-major) = [[7,9,11],[8,10,12]]  (2x3 in B^T layout, i.e. 2 rows of length 3)
  // Expected C = A * B = [[58,64],[139,154]]
  let a = make_f64_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let b_t = make_f64_array([7.0, 9.0, 11.0, 8.0, 10.0, 12.0])
  let c = FixedArray::make(4, 0.0)
  simd_matmul_f64(a, b_t, c, 2, 3, 2)
  assert_f64_close(c[0], 58.0)
  assert_f64_close(c[1], 64.0)
  assert_f64_close(c[2], 139.0)
  assert_f64_close(c[3], 154.0)
}

///|
test "simd_matmul_f64 matches scalar (5x7 * 7x4)" {
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
  simd_matmul_f64(a, b_t, c_simd, m, k, n)
  scalar_matmul_f64(a, b_t, c_scalar, m, k, n)
  for i in 0..<(m * n) {
    assert_f64_close(c_simd[i], c_scalar[i])
  }
}

///|
test "simd_gemv_f64 vs scalar" {
  // A = [[1,2,3],[4,5,6]] (2x3), x = [1,2,3]
  // y = [1+4+9, 4+10+18] = [14, 32]
  let a = make_f64_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let x = make_f64_array([1.0, 2.0, 3.0])
  let y = FixedArray::make(2, 0.0)
  simd_gemv_f64(a, x, y, 2, 3)
  assert_f64_close(y[0], 14.0)
  assert_f64_close(y[1], 32.0)
}

///|
test "simd_gemv_f64 matches scalar (larger)" {
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
  simd_gemv_f64(a, x, y_simd, m, n)
  scalar_gemv_f64(a, x, y_scalar, m, n)
  for i in 0..<m {
    assert_f64_close(y_simd[i], y_scalar[i])
  }
}

///|
test "simd_transpose_f64 (3x4)" {
  // A = [[1,2,3,4],[5,6,7,8],[9,10,11,12]] (3x4)
  // A^T = [[1,5,9],[2,6,10],[3,7,11],[4,8,12]] (4x3)
  let a = make_f64_array([
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
  ])
  let out = FixedArray::make(12, 0.0)
  simd_transpose_f64(a, out, 3, 4)
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
fn assert_f64_close(a : Double, b : Double, eps~ : Double = 1.0e-9) -> Unit {
  let d = (a - b).abs()
  if d > eps {
    abort("expected ~" + b.to_string() + " got " + a.to_string())
  }
}

///|
test "simd_add_f64 matches scalar" {
  let a = make_f64_array([1.5, 2.0, 3.25, 4.0, 5.5, 6.0, 7.125])
  let b = make_f64_array([10.0, 20.5, 30.0, 40.25, 50.0, 60.5, 70.0])
  let out_simd = FixedArray::make(7, 0.0)
  let out_scalar = FixedArray::make(7, 0.0)
  simd_add_f64(a, b, out_simd)
  scalar_add_f64(a, b, out_scalar)
  for i in 0..<7 {
    assert_f64_close(out_simd[i], out_scalar[i])
  }
}

///|
test "simd_sub/mul/div_f64 match scalar" {
  let a = make_f64_array([1.0, 4.0, 9.0, 16.0, 25.0])
  let b = make_f64_array([2.0, 2.0, 3.0, 4.0, 5.0])
  let out = FixedArray::make(5, 0.0)
  let ref_arr = FixedArray::make(5, 0.0)
  simd_sub_f64(a, b, out)
  scalar_sub_f64(a, b, ref_arr)
  for i in 0..<5 {
    assert_f64_close(out[i], ref_arr[i])
  }
  simd_mul_f64(a, b, out)
  scalar_mul_f64(a, b, ref_arr)
  for i in 0..<5 {
    assert_f64_close(out[i], ref_arr[i])
  }
  simd_div_f64(a, b, out)
  scalar_div_f64(a, b, ref_arr)
  for i in 0..<5 {
    assert_f64_close(out[i], ref_arr[i])
  }
}

///|
test "simd_sqrt_f64 matches scalar" {
  let a = make_f64_array([1.0, 4.0, 9.0, 16.0, 25.0])
  let out = FixedArray::make(5, 0.0)
  simd_sqrt_f64(a, out)
  assert_f64_close(out[0], 1.0)
  assert_f64_close(out[1], 2.0)
  assert_f64_close(out[2], 3.0)
  assert_f64_close(out[3], 4.0)
  assert_f64_close(out[4], 5.0)
}

///|
test "simd_min_f64 / simd_max_f64 element-wise" {
  let a = make_f64_array([1.0, 5.0, 3.0, 8.0, 2.0])
  let b = make_f64_array([4.0, 2.0, 7.0, 6.0, 9.0])
  let out = FixedArray::make(5, 0.0)
  simd_min_f64(a, b, out)
  assert_f64_close(out[0], 1.0)
  assert_f64_close(out[1], 2.0)
  assert_f64_close(out[2], 3.0)
  assert_f64_close(out[3], 6.0)
  assert_f64_close(out[4], 2.0)
  simd_max_f64(a, b, out)
  assert_f64_close(out[0], 4.0)
  assert_f64_close(out[1], 5.0)
  assert_f64_close(out[2], 7.0)
  assert_f64_close(out[3], 8.0)
  assert_f64_close(out[4], 9.0)
}

///|
test "simd_sum_f64 matches scalar" {
  let a = make_f64_array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
  assert_f64_close(simd_sum_f64(a), scalar_sum_f64(a))
}

///|
test "simd_sum_f64 empty" {
  let arr : FixedArray[Double] = FixedArray::make(0, 0.0)
  assert_f64_close(simd_sum_f64(arr), 0.0)
}

///|
test "simd_dot_f64 matches scalar" {
  let a = make_f64_array([1.0, 2.0, 3.0, 4.0, 5.0])
  let b = make_f64_array([2.0, 3.0, 4.0, 5.0, 6.0])
  // 2 + 6 + 12 + 20 + 30 = 70
  assert_f64_close(simd_dot_f64(a, b), 70.0)
}
