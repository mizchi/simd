///|
/// SIMD sum - wasm-gc target: scalar fallback (GC heap, no v128.load).
pub fn simd_sum(arr : FixedArray[Int]) -> Int {
  scalar_sum(arr)
}

///|
/// SIMD dot product - wasm-gc target: scalar fallback.
pub fn simd_dot_product(a : FixedArray[Int], b : FixedArray[Int]) -> Int {
  scalar_dot_product(a, b)
}

///|
/// SIMD element-wise add - wasm-gc target: scalar fallback.
pub fn simd_add(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  scalar_add(a, b, out)
}

///|
/// SIMD Adler-32 - wasm-gc target: scalar fallback.
pub fn simd_adler32(data : FixedArray[Byte]) -> UInt {
  scalar_adler32(data)
}

///|
pub fn simd_min(arr : FixedArray[Int]) -> Int {
  scalar_min(arr)
}

///|
pub fn simd_max(arr : FixedArray[Int]) -> Int {
  scalar_max(arr)
}

///|
pub fn simd_argmin(arr : FixedArray[Int]) -> Int {
  scalar_argmin(arr)
}

///|
pub fn simd_argmax(arr : FixedArray[Int]) -> Int {
  scalar_argmax(arr)
}

///|
pub fn simd_saxpy(
  k : Int,
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  scalar_saxpy(k, a, b, out)
}

///|
pub fn simd_popcount_bytes(data : FixedArray[Byte]) -> Int {
  scalar_popcount_bytes(data)
}

///|
pub fn simd_memcpy_bytes(
  src : FixedArray[Byte],
  si : Int,
  dst : FixedArray[Byte],
  di : Int,
  len : Int,
) -> Unit {
  scalar_memcpy_bytes(src, si, dst, di, len)
}

///|
pub fn simd_memset_bytes(
  dst : FixedArray[Byte],
  off : Int,
  len : Int,
  value : Byte,
) -> Unit {
  scalar_memset_bytes(dst, off, len, value)
}

///|
pub fn simd_equal_bytes(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  len : Int,
) -> Bool {
  scalar_equal_bytes(a, b, len)
}

///|
pub fn simd_find_byte(data : FixedArray[Byte], needle : Byte) -> Int {
  scalar_find_byte(data, needle)
}

///|
pub fn simd_count_byte(data : FixedArray[Byte], needle : Byte) -> Int {
  scalar_count_byte(data, needle)
}

///|
pub fn simd_add_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  scalar_add_f64(a, b, out)
}

///|
pub fn simd_sub_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  scalar_sub_f64(a, b, out)
}

///|
pub fn simd_mul_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  scalar_mul_f64(a, b, out)
}

///|
pub fn simd_div_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  scalar_div_f64(a, b, out)
}

///|
pub fn simd_sqrt_f64(
  a : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  scalar_sqrt_f64(a, out)
}

///|
pub fn simd_min_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  scalar_min_f64(a, b, out)
}

///|
pub fn simd_max_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  scalar_max_f64(a, b, out)
}

///|
pub fn simd_sum_f64(arr : FixedArray[Double]) -> Double {
  scalar_sum_f64(arr)
}

///|
pub fn simd_dot_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
) -> Double {
  scalar_dot_f64(a, b)
}

///|
pub fn simd_sub(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  scalar_sub(a, b, out)
}

///|
pub fn simd_mul(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  scalar_mul(a, b, out)
}

///|
pub fn simd_neg(a : FixedArray[Int], out : FixedArray[Int]) -> Unit {
  scalar_neg(a, out)
}

///|
pub fn simd_abs(a : FixedArray[Int], out : FixedArray[Int]) -> Unit {
  scalar_abs(a, out)
}

///|
pub fn simd_min_elem(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  scalar_min_elem(a, b, out)
}

///|
pub fn simd_max_elem(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  scalar_max_elem(a, b, out)
}

///|
pub fn simd_eq(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  scalar_eq(a, b, out)
}

///|
pub fn simd_lt(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  scalar_lt(a, b, out)
}

///|
pub fn simd_gt(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  scalar_gt(a, b, out)
}

///|
pub fn simd_where(
  mask : FixedArray[Int],
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  scalar_where(mask, a, b, out)
}

///|
pub fn simd_matmul_f64(
  a : FixedArray[Double],
  b_t : FixedArray[Double],
  c : FixedArray[Double],
  m : Int,
  k : Int,
  n : Int,
) -> Unit {
  scalar_matmul_f64(a, b_t, c, m, k, n)
}

///|
pub fn simd_gemv_f64(
  a : FixedArray[Double],
  x : FixedArray[Double],
  y : FixedArray[Double],
  m : Int,
  n : Int,
) -> Unit {
  scalar_gemv_f64(a, x, y, m, n)
}

///|
pub fn simd_transpose_f64(
  a : FixedArray[Double],
  out : FixedArray[Double],
  m : Int,
  n : Int,
) -> Unit {
  scalar_transpose_f64(a, out, m, n)
}

///|
pub fn simd_prod(arr : FixedArray[Int]) -> Int {
  scalar_prod(arr)
}

///|
pub fn simd_mean_f64(arr : FixedArray[Double]) -> Double {
  scalar_mean_f64(arr)
}

///|
pub fn simd_var_f64(arr : FixedArray[Double]) -> Double {
  scalar_var_f64(arr)
}

///|
pub fn simd_count_nonzero(arr : FixedArray[Int]) -> Int {
  scalar_count_nonzero(arr)
}

///|
pub fn simd_all(arr : FixedArray[Int]) -> Bool {
  scalar_all(arr)
}

///|
pub fn simd_any(arr : FixedArray[Int]) -> Bool {
  scalar_any(arr)
}

///|
pub fn simd_cumsum(arr : FixedArray[Int], out : FixedArray[Int]) -> Unit {
  scalar_cumsum(arr, out)
}

///|
pub fn simd_cumprod(arr : FixedArray[Int], out : FixedArray[Int]) -> Unit {
  scalar_cumprod(arr, out)
}
