///|
/// native sum: scalar wins thanks to compiler auto-vectorization;
/// going through ffi adds call overhead. The C kernel (simd_sum_ffi)
/// is kept available for explicit use.
pub fn sum_i32(arr : FixedArray[Int]) -> Int {
  @internal.scalar_sum(arr)
}

///|
/// native dot: same story as sum — scalar auto-vec wins.
pub fn dot_i32(a : FixedArray[Int], b : FixedArray[Int]) -> Int {
  @internal.scalar_dot_product(a, b)
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
/// native fill: scalar auto-vec wins; no C FFI needed.
pub fn fill_i32(arr : FixedArray[Int], value : Int) -> Unit {
  @internal.scalar_fill(arr, value)
}

///|
/// native find: scalar early-exit beats branchless SIMD here.
pub fn find_i32(arr : FixedArray[Int], needle : Int) -> Int {
  @internal.scalar_find(arr, needle)
}

///|
/// native min reduce: scalar auto-vec wins; reverted from C FFI.
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
  let len = a.length()
  if len == 0 {
    return
  }
  simd_saxpy_ffi(k, a, b, out, len)
}

///|
pub fn popcount_bytes(data : FixedArray[Byte]) -> Int {
  let len = data.length()
  if len == 0 {
    return 0
  }
  simd_popcount_bytes_ffi(data, len)
}

///|
pub fn memcpy_bytes(
  src : FixedArray[Byte],
  si : Int,
  dst : FixedArray[Byte],
  di : Int,
  len : Int,
) -> Unit {
  if len <= 0 {
    return
  }
  simd_memcpy_bytes_ffi(src, si, dst, di, len)
}

///|
pub fn memset_bytes(
  dst : FixedArray[Byte],
  off : Int,
  len : Int,
  value : Byte,
) -> Unit {
  if len <= 0 {
    return
  }
  simd_memset_bytes_ffi(dst, off, len, value)
}

///|
pub fn equal_bytes(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  len : Int,
) -> Bool {
  if len <= 0 {
    return true
  }
  simd_equal_bytes_ffi(a, b, len) != 0
}

///|
pub fn find_byte(data : FixedArray[Byte], needle : Byte) -> Int {
  let len = data.length()
  if len == 0 {
    return -1
  }
  simd_find_byte_ffi(data, len, needle)
}

///|
pub fn count_byte(data : FixedArray[Byte], needle : Byte) -> Int {
  let len = data.length()
  if len == 0 {
    return 0
  }
  simd_count_byte_ffi(data, len, needle)
}

///|
pub fn add_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  simd_add_f64_ffi(a, b, out, len)
}

///|
pub fn sub_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  simd_sub_f64_ffi(a, b, out, len)
}

///|
pub fn mul_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  simd_mul_f64_ffi(a, b, out, len)
}

///|
pub fn div_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  simd_div_f64_ffi(a, b, out, len)
}

///|
pub fn sqrt_f64(a : FixedArray[Double], out : FixedArray[Double]) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  simd_sqrt_f64_ffi(a, out, len)
}

///|
pub fn min_elem_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  simd_min_elem_f64_ffi(a, b, out, len)
}

///|
pub fn max_elem_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  simd_max_elem_f64_ffi(a, b, out, len)
}

///|
pub fn sum_f64(arr : FixedArray[Double]) -> Double {
  let len = arr.length()
  if len == 0 {
    return 0.0
  }
  simd_sum_f64_ffi(arr, len)
}

///|
pub fn dot_f64(a : FixedArray[Double], b : FixedArray[Double]) -> Double {
  let len = a.length()
  if len == 0 {
    return 0.0
  }
  simd_dot_f64_ffi(a, b, len)
}

///|
pub fn sub_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  simd_sub_ffi(a, b, out, len)
}

///|
pub fn mul_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  simd_mul_ffi(a, b, out, len)
}

///|
pub fn neg_i32(a : FixedArray[Int], out : FixedArray[Int]) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  simd_neg_i32_ffi(a, out, len)
}

///|
pub fn abs_i32(a : FixedArray[Int], out : FixedArray[Int]) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  simd_abs_i32_ffi(a, out, len)
}

///|
pub fn min_elem_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  simd_min_elem_ffi(a, b, out, len)
}

///|
pub fn max_elem_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  simd_max_elem_ffi(a, b, out, len)
}

///|
pub fn eq_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  simd_eq_i32_ffi(a, b, out, len)
}

///|
pub fn lt_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  simd_lt_i32_ffi(a, b, out, len)
}

///|
pub fn gt_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  simd_gt_i32_ffi(a, b, out, len)
}

///|
pub fn where_i32(
  mask : FixedArray[Int],
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = mask.length()
  if len == 0 {
    return
  }
  simd_where_i32_ffi(mask, a, b, out, len)
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
  if m == 0 || n == 0 {
    return
  }
  simd_matmul_f64_tile_ffi(a, b_t, c, m, k, n)
}

///|
pub fn gemv_f64(
  a : FixedArray[Double],
  x : FixedArray[Double],
  y : FixedArray[Double],
  m : Int,
  n : Int,
) -> Unit {
  if m == 0 {
    return
  }
  for i in 0..<m {
    y[i] = simd_dot_f64_range_ffi(a, i * n, x, 0, n)
  }
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
  let len = arr.length()
  if len == 0 {
    return true
  }
  simd_all_i32_ffi(arr, len) != 0
}

///|
pub fn any_i32(arr : FixedArray[Int]) -> Bool {
  let len = arr.length()
  if len == 0 {
    return false
  }
  simd_any_i32_ffi(arr, len) != 0
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
  if n == 0 {
    return
  }
  simd_add_f32_ffi(a, b, out, n)
}

///|
pub fn mul_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  if n == 0 {
    return
  }
  simd_mul_f32_ffi(a, b, out, n)
}

///|
pub fn sum_f32(arr : FixedArray[Byte], n : Int) -> Float {
  if n == 0 {
    return (0.0 : Float)
  }
  simd_sum_f32_ffi(arr, n)
}

///|
pub fn dot_f32(a : FixedArray[Byte], b : FixedArray[Byte], n : Int) -> Float {
  if n == 0 {
    return (0.0 : Float)
  }
  simd_dot_f32_ffi(a, b, n)
}

///|
pub fn sub_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  if n == 0 {
    return
  }
  simd_sub_f32_ffi(a, b, out, n)
}

///|
pub fn div_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  if n == 0 {
    return
  }
  simd_div_f32_ffi(a, b, out, n)
}

///|
pub fn sqrt_f32(a : FixedArray[Byte], out : FixedArray[Byte], n : Int) -> Unit {
  if n == 0 {
    return
  }
  simd_sqrt_f32_ffi(a, out, n)
}

///|
pub fn min_elem_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  if n == 0 {
    return
  }
  simd_min_elem_f32_ffi(a, b, out, n)
}

///|
pub fn max_elem_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  if n == 0 {
    return
  }
  simd_max_elem_f32_ffi(a, b, out, n)
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
  let len = data.length()
  if len == 0 {
    return true
  }
  simd_is_ascii_ffi(data, len) != 0
}

///|
pub fn to_lower_ascii(data : FixedArray[Byte]) -> Unit {
  let len = data.length()
  if len == 0 {
    return
  }
  simd_to_lower_ascii_ffi(data, len)
}

///|
pub fn to_upper_ascii(data : FixedArray[Byte]) -> Unit {
  let len = data.length()
  if len == 0 {
    return
  }
  simd_to_upper_ascii_ffi(data, len)
}

///|
pub fn validate_utf8(data : FixedArray[Byte]) -> Bool {
  let len = data.length()
  let start = if len >= 16 {
    simd_first_non_ascii_chunk_ffi(data, len)
  } else {
    0
  }
  @internal.scalar_validate_utf8_from(data, start)
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
pub fn bitonic_merge32_i32(arr : FixedArray[Int], off : Int) -> Unit {
  @internal.scalar_bitonic_merge32_int(arr, off)
}

///|
pub fn bitonic_merge64_i32(arr : FixedArray[Int], off : Int) -> Unit {
  @internal.scalar_bitonic_merge64_int(arr, off)
}

///|
pub fn sort16_i32(arr : FixedArray[Int], off : Int) -> Unit {
  @internal.scalar_sort16_int(arr, off)
}

///|
pub fn sort_i32(arr : FixedArray[Int]) -> Unit {
  @internal.scalar_sort_int(arr)
}
