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
pub fn simd_sum(arr : FixedArray[Int]) -> Int {
  let len = arr.length()
  if len == 0 {
    return 0
  }
  simd_sum_ffi(arr, len)
}

///|
/// SIMD dot product - native target: C FFI (NEON/SSE).
pub fn simd_dot_product(a : FixedArray[Int], b : FixedArray[Int]) -> Int {
  let len = a.length()
  if len == 0 {
    return 0
  }
  simd_dot_product_ffi(a, b, len)
}

///|
/// SIMD element-wise add - native target: C FFI (NEON/SSE).
pub fn simd_add(
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
pub fn simd_adler32(data : FixedArray[Byte]) -> UInt {
  let len = data.length()
  if len == 0 {
    return 1U
  }
  simd_adler32_ffi(data, len)
}
