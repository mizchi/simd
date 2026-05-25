///|
#borrow(arr)
extern "C" fn simd_sum_ffi(arr : FixedArray[Int], len : Int) -> Int = "simd_sum_ffi"

///|
#borrow(a, b)
extern "C" fn simd_dot_product_ffi(
  a : FixedArray[Int],
  b : FixedArray[Int],
  len : Int,
) -> Int = "simd_dot_product_ffi"

///|
#borrow(a, b, out)
extern "C" fn simd_add_ffi(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit = "simd_add_ffi"

///|
#borrow(data)
extern "C" fn simd_adler32_ffi(data : FixedArray[Byte], len : Int) -> UInt = "simd_adler32_ffi"

///|
/// SIMD sum - native target: C FFI (NEON/SSE).
pub fn sum_i32(arr : FixedArray[Int]) -> Int {
  let len = arr.length()
  if len == 0 {
    return 0
  }
  simd_sum_ffi(arr, len)
}

///|
/// SIMD dot product - native target: C FFI (NEON/SSE).
pub fn dot_i32(a : FixedArray[Int], b : FixedArray[Int]) -> Int {
  let len = a.length()
  if len == 0 {
    return 0
  }
  simd_dot_product_ffi(a, b, len)
}

///|
/// SIMD element-wise add - native target: C FFI (NEON/SSE).
pub fn add_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  simd_add_ffi(a, b, out, len)
}

///|
/// SIMD Adler-32 - native target: optimized scalar with deferred modulo.
pub fn adler32_bytes(data : FixedArray[Byte]) -> UInt {
  let len = data.length()
  if len == 0 {
    return 1U
  }
  simd_adler32_ffi(data, len)
}

///|
pub fn min_i32(arr : FixedArray[Int]) -> Int {
  @internal.scalar_min(arr)
}

///|
pub fn max_i32(arr : FixedArray[Int]) -> Int {
  @internal.scalar_max(arr)
}

///|
pub fn argmin_i32(arr : FixedArray[Int]) -> Int {
  @internal.scalar_argmin(arr)
}

///|
pub fn argmax_i32(arr : FixedArray[Int]) -> Int {
  @internal.scalar_argmax(arr)
}

///|
pub fn saxpy_i32(
  k : Int,
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  @internal.scalar_saxpy(k, a, b, out)
}

///|
pub fn popcount_bytes(data : FixedArray[Byte]) -> Int {
  @internal.scalar_popcount_bytes(data)
}

///|
pub fn memcpy_bytes(
  src : FixedArray[Byte],
  si : Int,
  dst : FixedArray[Byte],
  di : Int,
  len : Int,
) -> Unit {
  @internal.scalar_memcpy_bytes(src, si, dst, di, len)
}

///|
pub fn memset_bytes(
  dst : FixedArray[Byte],
  off : Int,
  len : Int,
  value : Byte,
) -> Unit {
  @internal.scalar_memset_bytes(dst, off, len, value)
}

///|
pub fn equal_bytes(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  len : Int,
) -> Bool {
  @internal.scalar_equal_bytes(a, b, len)
}

///|
pub fn find_byte(data : FixedArray[Byte], needle : Byte) -> Int {
  @internal.scalar_find_byte(data, needle)
}

///|
pub fn count_byte(data : FixedArray[Byte], needle : Byte) -> Int {
  @internal.scalar_count_byte(data, needle)
}

///|
pub fn add_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  @internal.scalar_add_f64(a, b, out)
}

///|
pub fn sub_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  @internal.scalar_sub_f64(a, b, out)
}

///|
pub fn mul_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  @internal.scalar_mul_f64(a, b, out)
}

///|
pub fn div_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  @internal.scalar_div_f64(a, b, out)
}

///|
pub fn sqrt_f64(
  a : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  @internal.scalar_sqrt_f64(a, out)
}

///|
pub fn min_elem_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  @internal.scalar_min_f64(a, b, out)
}

///|
pub fn max_elem_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  @internal.scalar_max_f64(a, b, out)
}

///|
pub fn sum_f64(arr : FixedArray[Double]) -> Double {
  @internal.scalar_sum_f64(arr)
}

///|
pub fn dot_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
) -> Double {
  @internal.scalar_dot_f64(a, b)
}

///|
pub fn sub_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  @internal.scalar_sub(a, b, out)
}

///|
pub fn mul_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  @internal.scalar_mul(a, b, out)
}

///|
pub fn neg_i32(a : FixedArray[Int], out : FixedArray[Int]) -> Unit {
  @internal.scalar_neg(a, out)
}

///|
pub fn abs_i32(a : FixedArray[Int], out : FixedArray[Int]) -> Unit {
  @internal.scalar_abs(a, out)
}

///|
pub fn min_elem_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  @internal.scalar_min_elem(a, b, out)
}

///|
pub fn max_elem_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  @internal.scalar_max_elem(a, b, out)
}

///|
pub fn eq_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  @internal.scalar_eq(a, b, out)
}

///|
pub fn lt_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  @internal.scalar_lt(a, b, out)
}

///|
pub fn gt_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  @internal.scalar_gt(a, b, out)
}

///|
pub fn where_i32(
  mask : FixedArray[Int],
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  @internal.scalar_where(mask, a, b, out)
}

///|
pub fn matmul_f64(
  a : FixedArray[Double],
  b_t : FixedArray[Double],
  c : FixedArray[Double],
  m : Int,
  k : Int,
  n : Int,
) -> Unit {
  @internal.scalar_matmul_f64(a, b_t, c, m, k, n)
}

///|
pub fn gemv_f64(
  a : FixedArray[Double],
  x : FixedArray[Double],
  y : FixedArray[Double],
  m : Int,
  n : Int,
) -> Unit {
  @internal.scalar_gemv_f64(a, x, y, m, n)
}

///|
pub fn transpose_f64(
  a : FixedArray[Double],
  out : FixedArray[Double],
  m : Int,
  n : Int,
) -> Unit {
  @internal.scalar_transpose_f64(a, out, m, n)
}

///|
pub fn prod_i32(arr : FixedArray[Int]) -> Int {
  @internal.scalar_prod(arr)
}

///|
pub fn mean_f64(arr : FixedArray[Double]) -> Double {
  @internal.scalar_mean_f64(arr)
}

///|
pub fn var_f64(arr : FixedArray[Double]) -> Double {
  @internal.scalar_var_f64(arr)
}

///|
pub fn count_nonzero_i32(arr : FixedArray[Int]) -> Int {
  @internal.scalar_count_nonzero(arr)
}

///|
pub fn all_i32(arr : FixedArray[Int]) -> Bool {
  @internal.scalar_all(arr)
}

///|
pub fn any_i32(arr : FixedArray[Int]) -> Bool {
  @internal.scalar_any(arr)
}

///|
pub fn cumsum_i32(arr : FixedArray[Int], out : FixedArray[Int]) -> Unit {
  @internal.scalar_cumsum(arr, out)
}

///|
pub fn cumprod_i32(arr : FixedArray[Int], out : FixedArray[Int]) -> Unit {
  @internal.scalar_cumprod(arr, out)
}

///|
pub fn add_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  @internal.scalar_add_f32(a, b, out, n)
}

///|
pub fn mul_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  @internal.scalar_mul_f32(a, b, out, n)
}

///|
pub fn sum_f32(arr : FixedArray[Byte], n : Int) -> Float {
  @internal.scalar_sum_f32(arr, n)
}

///|
pub fn dot_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  n : Int,
) -> Float {
  @internal.scalar_dot_f32(a, b, n)
}

///|
pub fn div_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  @internal.scalar_div(a, b, out)
}

///|
pub fn gather_i32(
  arr : FixedArray[Int],
  indices : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  @internal.scalar_gather(arr, indices, out)
}

///|
pub fn is_ascii(data : FixedArray[Byte]) -> Bool {
  @internal.scalar_is_ascii(data)
}

///|
pub fn to_lower_ascii(data : FixedArray[Byte]) -> Unit {
  @internal.scalar_to_lower_ascii(data)
}

///|
pub fn to_upper_ascii(data : FixedArray[Byte]) -> Unit {
  @internal.scalar_to_upper_ascii(data)
}

///|
pub fn validate_utf8(data : FixedArray[Byte]) -> Bool {
  @internal.scalar_validate_utf8(data)
}

///|
pub fn scatter_i32(
  out : FixedArray[Int],
  indices : FixedArray[Int],
  vals : FixedArray[Int],
) -> Unit {
  @internal.scalar_scatter(out, indices, vals)
}

///|
pub fn sort4_i32(arr : FixedArray[Int], off : Int) -> Unit {
  @internal.scalar_sort4_int(arr, off)
}

///|
pub fn bitonic_merge8_i32(arr : FixedArray[Int], off : Int) -> Unit {
  @internal.scalar_bitonic_merge8_int(arr, off)
}

///|
pub fn bitonic_merge16_i32(arr : FixedArray[Int], off : Int) -> Unit {
  @internal.scalar_bitonic_merge16_int(arr, off)
}

///|
pub fn sort16_i32(arr : FixedArray[Int], off : Int) -> Unit {
  @internal.scalar_sort16_int(arr, off)
}

///|
pub fn sort_i32(arr : FixedArray[Int]) -> Unit {
  @internal.scalar_sort_int(arr)
}
