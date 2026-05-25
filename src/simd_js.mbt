///|
/// SIMD sum - js target: scalar fallback.
pub fn sum_i32(arr : FixedArray[Int]) -> Int {
  scalar_sum(arr)
}

///|
/// SIMD dot product - js target: scalar fallback.
pub fn dot_i32(a : FixedArray[Int], b : FixedArray[Int]) -> Int {
  scalar_dot_product(a, b)
}

///|
/// SIMD element-wise add - js target: scalar fallback.
pub fn add_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  scalar_add(a, b, out)
}

///|
/// SIMD Adler-32 - js target: scalar fallback.
pub fn adler32_bytes(data : FixedArray[Byte]) -> UInt {
  scalar_adler32(data)
}

///|
pub fn min_i32(arr : FixedArray[Int]) -> Int {
  scalar_min(arr)
}

///|
pub fn max_i32(arr : FixedArray[Int]) -> Int {
  scalar_max(arr)
}

///|
pub fn argmin_i32(arr : FixedArray[Int]) -> Int {
  scalar_argmin(arr)
}

///|
pub fn argmax_i32(arr : FixedArray[Int]) -> Int {
  scalar_argmax(arr)
}

///|
pub fn saxpy_i32(
  k : Int,
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  scalar_saxpy(k, a, b, out)
}

///|
pub fn popcount_bytes(data : FixedArray[Byte]) -> Int {
  scalar_popcount_bytes(data)
}

///|
pub fn memcpy_bytes(
  src : FixedArray[Byte],
  si : Int,
  dst : FixedArray[Byte],
  di : Int,
  len : Int,
) -> Unit {
  scalar_memcpy_bytes(src, si, dst, di, len)
}

///|
pub fn memset_bytes(
  dst : FixedArray[Byte],
  off : Int,
  len : Int,
  value : Byte,
) -> Unit {
  scalar_memset_bytes(dst, off, len, value)
}

///|
pub fn equal_bytes(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  len : Int,
) -> Bool {
  scalar_equal_bytes(a, b, len)
}

///|
pub fn find_byte(data : FixedArray[Byte], needle : Byte) -> Int {
  scalar_find_byte(data, needle)
}

///|
pub fn count_byte(data : FixedArray[Byte], needle : Byte) -> Int {
  scalar_count_byte(data, needle)
}

///|
pub fn add_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  scalar_add_f64(a, b, out)
}

///|
pub fn sub_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  scalar_sub_f64(a, b, out)
}

///|
pub fn mul_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  scalar_mul_f64(a, b, out)
}

///|
pub fn div_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  scalar_div_f64(a, b, out)
}

///|
pub fn sqrt_f64(
  a : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  scalar_sqrt_f64(a, out)
}

///|
pub fn min_elem_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  scalar_min_f64(a, b, out)
}

///|
pub fn max_elem_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  scalar_max_f64(a, b, out)
}

///|
pub fn sum_f64(arr : FixedArray[Double]) -> Double {
  scalar_sum_f64(arr)
}

///|
pub fn dot_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
) -> Double {
  scalar_dot_f64(a, b)
}

///|
pub fn sub_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  scalar_sub(a, b, out)
}

///|
pub fn mul_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  scalar_mul(a, b, out)
}

///|
pub fn neg_i32(a : FixedArray[Int], out : FixedArray[Int]) -> Unit {
  scalar_neg(a, out)
}

///|
pub fn abs_i32(a : FixedArray[Int], out : FixedArray[Int]) -> Unit {
  scalar_abs(a, out)
}

///|
pub fn min_elem_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  scalar_min_elem(a, b, out)
}

///|
pub fn max_elem_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  scalar_max_elem(a, b, out)
}

///|
pub fn eq_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  scalar_eq(a, b, out)
}

///|
pub fn lt_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  scalar_lt(a, b, out)
}

///|
pub fn gt_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  scalar_gt(a, b, out)
}

///|
pub fn where_i32(
  mask : FixedArray[Int],
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  scalar_where(mask, a, b, out)
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
  scalar_matmul_f64(a, b_t, c, m, k, n)
}

///|
pub fn gemv_f64(
  a : FixedArray[Double],
  x : FixedArray[Double],
  y : FixedArray[Double],
  m : Int,
  n : Int,
) -> Unit {
  scalar_gemv_f64(a, x, y, m, n)
}

///|
pub fn transpose_f64(
  a : FixedArray[Double],
  out : FixedArray[Double],
  m : Int,
  n : Int,
) -> Unit {
  scalar_transpose_f64(a, out, m, n)
}

///|
pub fn prod_i32(arr : FixedArray[Int]) -> Int {
  scalar_prod(arr)
}

///|
pub fn mean_f64(arr : FixedArray[Double]) -> Double {
  scalar_mean_f64(arr)
}

///|
pub fn var_f64(arr : FixedArray[Double]) -> Double {
  scalar_var_f64(arr)
}

///|
pub fn count_nonzero_i32(arr : FixedArray[Int]) -> Int {
  scalar_count_nonzero(arr)
}

///|
pub fn all_i32(arr : FixedArray[Int]) -> Bool {
  scalar_all(arr)
}

///|
pub fn any_i32(arr : FixedArray[Int]) -> Bool {
  scalar_any(arr)
}

///|
pub fn cumsum_i32(arr : FixedArray[Int], out : FixedArray[Int]) -> Unit {
  scalar_cumsum(arr, out)
}

///|
pub fn cumprod_i32(arr : FixedArray[Int], out : FixedArray[Int]) -> Unit {
  scalar_cumprod(arr, out)
}

///|
pub fn add_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  scalar_add_f32(a, b, out, n)
}

///|
pub fn mul_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  scalar_mul_f32(a, b, out, n)
}

///|
pub fn sum_f32(arr : FixedArray[Byte], n : Int) -> Float {
  scalar_sum_f32(arr, n)
}

///|
pub fn dot_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  n : Int,
) -> Float {
  scalar_dot_f32(a, b, n)
}

///|
pub fn div_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  scalar_div(a, b, out)
}

///|
pub fn gather_i32(
  arr : FixedArray[Int],
  indices : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  scalar_gather(arr, indices, out)
}

///|
pub fn is_ascii(data : FixedArray[Byte]) -> Bool {
  scalar_is_ascii(data)
}

///|
pub fn to_lower_ascii(data : FixedArray[Byte]) -> Unit {
  scalar_to_lower_ascii(data)
}

///|
pub fn to_upper_ascii(data : FixedArray[Byte]) -> Unit {
  scalar_to_upper_ascii(data)
}

///|
pub fn validate_utf8(data : FixedArray[Byte]) -> Bool {
  scalar_validate_utf8(data)
}

///|
pub fn scatter_i32(
  out : FixedArray[Int],
  indices : FixedArray[Int],
  vals : FixedArray[Int],
) -> Unit {
  scalar_scatter(out, indices, vals)
}

///|
pub fn sort4_i32(arr : FixedArray[Int], off : Int) -> Unit {
  scalar_sort4_int(arr, off)
}

///|
pub fn bitonic_merge8_i32(arr : FixedArray[Int], off : Int) -> Unit {
  scalar_bitonic_merge8_int(arr, off)
}

///|
pub fn bitonic_merge16_i32(arr : FixedArray[Int], off : Int) -> Unit {
  scalar_bitonic_merge16_int(arr, off)
}

///|
pub fn sort16_i32(arr : FixedArray[Int], off : Int) -> Unit {
  scalar_sort16_int(arr, off)
}

///|
pub fn sort_i32(arr : FixedArray[Int]) -> Unit {
  scalar_sort_int(arr)
}
