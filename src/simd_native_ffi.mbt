///|
#borrow(a, b, out)
extern "C" fn simd_add_ffi(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit = "simd_add_ffi"

///|
#borrow(a, b, out)
extern "C" fn simd_sub_ffi(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit = "simd_sub_ffi"

///|
#borrow(a, b, out)
extern "C" fn simd_mul_ffi(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit = "simd_mul_ffi"

///|
#borrow(a, b, out)
extern "C" fn simd_min_elem_ffi(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit = "simd_min_elem_ffi"

///|
#borrow(a, b, out)
extern "C" fn simd_max_elem_ffi(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit = "simd_max_elem_ffi"

///|
#borrow(a, b, out)
extern "C" fn simd_saxpy_ffi(
  k : Int,
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit = "simd_saxpy_ffi"

///|
#borrow(a, b, out)
extern "C" fn simd_add_f64_ffi(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
  len : Int,
) -> Unit = "simd_add_f64_ffi"

///|
#borrow(a, b, out)
extern "C" fn simd_sub_f64_ffi(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
  len : Int,
) -> Unit = "simd_sub_f64_ffi"

///|
#borrow(a, b, out)
extern "C" fn simd_mul_f64_ffi(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
  len : Int,
) -> Unit = "simd_mul_f64_ffi"

///|
#borrow(a, b, out)
extern "C" fn simd_div_f64_ffi(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
  len : Int,
) -> Unit = "simd_div_f64_ffi"

///|
#borrow(a, out)
extern "C" fn simd_sqrt_f64_ffi(
  a : FixedArray[Double],
  out : FixedArray[Double],
  len : Int,
) -> Unit = "simd_sqrt_f64_ffi"

///|
#borrow(a, b, out)
extern "C" fn simd_min_elem_f64_ffi(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
  len : Int,
) -> Unit = "simd_min_elem_f64_ffi"

///|
#borrow(a, b, out)
extern "C" fn simd_max_elem_f64_ffi(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
  len : Int,
) -> Unit = "simd_max_elem_f64_ffi"

///|
#borrow(arr)
extern "C" fn simd_sum_f64_ffi(arr : FixedArray[Double], len : Int) -> Double = "simd_sum_f64_ffi"

///|
#borrow(a, b)
extern "C" fn simd_dot_f64_ffi(
  a : FixedArray[Double],
  b : FixedArray[Double],
  len : Int,
) -> Double = "simd_dot_f64_ffi"

///|
#borrow(a, b, out)
extern "C" fn simd_add_f32_ffi(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit = "simd_add_f32_ffi"

///|
#borrow(a, b, out)
extern "C" fn simd_mul_f32_ffi(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit = "simd_mul_f32_ffi"

///|
#borrow(arr)
extern "C" fn simd_sum_f32_ffi(arr : FixedArray[Byte], n : Int) -> Float = "simd_sum_f32_ffi"

///|
#borrow(a, b, out)
extern "C" fn simd_sub_f32_ffi(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit = "simd_sub_f32_ffi"

///|
#borrow(a, b, out)
extern "C" fn simd_div_f32_ffi(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit = "simd_div_f32_ffi"

///|
#borrow(a, out)
extern "C" fn simd_sqrt_f32_ffi(
  a : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit = "simd_sqrt_f32_ffi"

///|
#borrow(a, b, out)
extern "C" fn simd_min_elem_f32_ffi(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit = "simd_min_elem_f32_ffi"

///|
#borrow(a, b, out)
extern "C" fn simd_max_elem_f32_ffi(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit = "simd_max_elem_f32_ffi"

///|
#borrow(a, b)
extern "C" fn simd_dot_f32_ffi(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  n : Int,
) -> Float = "simd_dot_f32_ffi"

///|
#borrow(a, b)
extern "C" fn simd_dot_f64_range_ffi(
  a : FixedArray[Double],
  a_off : Int,
  b : FixedArray[Double],
  b_off : Int,
  k : Int,
) -> Double = "simd_dot_f64_range_ffi"

///|
#borrow(a, b_t, c)
extern "C" fn simd_matmul_f64_tile_ffi(
  a : FixedArray[Double],
  b_t : FixedArray[Double],
  c : FixedArray[Double],
  m : Int,
  k : Int,
  n : Int,
) -> Unit = "simd_matmul_f64_tile_ffi"

///|
#borrow(a, out)
extern "C" fn simd_neg_i32_ffi(
  a : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit = "simd_neg_i32_ffi"

///|
#borrow(a, out)
extern "C" fn simd_abs_i32_ffi(
  a : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit = "simd_abs_i32_ffi"

///|
#borrow(a, b, out)
extern "C" fn simd_eq_i32_ffi(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit = "simd_eq_i32_ffi"

///|
#borrow(a, b, out)
extern "C" fn simd_lt_i32_ffi(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit = "simd_lt_i32_ffi"

///|
#borrow(a, b, out)
extern "C" fn simd_gt_i32_ffi(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit = "simd_gt_i32_ffi"

///|
#borrow(mask, a, b, out)
extern "C" fn simd_where_i32_ffi(
  mask : FixedArray[Int],
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit = "simd_where_i32_ffi"

///|
#borrow(arr)
extern "C" fn simd_all_i32_ffi(arr : FixedArray[Int], len : Int) -> Int = "simd_all_i32_ffi"

///|
#borrow(arr)
extern "C" fn simd_any_i32_ffi(arr : FixedArray[Int], len : Int) -> Int = "simd_any_i32_ffi"

///|
#borrow(src, dst)
extern "C" fn simd_memcpy_bytes_ffi(
  src : FixedArray[Byte],
  si : Int,
  dst : FixedArray[Byte],
  di : Int,
  len : Int,
) -> Unit = "simd_memcpy_bytes_ffi"

///|
#borrow(dst)
extern "C" fn simd_memset_bytes_ffi(
  dst : FixedArray[Byte],
  off : Int,
  len : Int,
  value : Byte,
) -> Unit = "simd_memset_bytes_ffi"

///|
#borrow(a, b)
extern "C" fn simd_equal_bytes_ffi(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  len : Int,
) -> Int = "simd_equal_bytes_ffi"

///|
#borrow(data)
extern "C" fn simd_find_byte_ffi(
  data : FixedArray[Byte],
  len : Int,
  needle : Byte,
) -> Int = "simd_find_byte_ffi"

///|
#borrow(data)
extern "C" fn simd_count_byte_ffi(
  data : FixedArray[Byte],
  len : Int,
  needle : Byte,
) -> Int = "simd_count_byte_ffi"

///|
#borrow(data)
extern "C" fn simd_popcount_bytes_ffi(
  data : FixedArray[Byte],
  len : Int,
) -> Int = "simd_popcount_bytes_ffi"

///|
#borrow(data)
extern "C" fn simd_is_ascii_ffi(data : FixedArray[Byte], len : Int) -> Int = "simd_is_ascii_ffi"

///|
#borrow(data)
extern "C" fn simd_to_lower_ascii_ffi(
  data : FixedArray[Byte],
  len : Int,
) -> Unit = "simd_to_lower_ascii_ffi"

///|
#borrow(data)
extern "C" fn simd_to_upper_ascii_ffi(
  data : FixedArray[Byte],
  len : Int,
) -> Unit = "simd_to_upper_ascii_ffi"

///|
#borrow(data)
extern "C" fn simd_first_non_ascii_chunk_ffi(
  data : FixedArray[Byte],
  len : Int,
) -> Int = "simd_first_non_ascii_chunk_ffi"

///|
#borrow(data)
extern "C" fn simd_adler32_ffi(data : FixedArray[Byte], len : Int) -> UInt = "simd_adler32_ffi"
