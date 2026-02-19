///|
/// SIMD sum - js target: scalar fallback.
pub fn simd_sum(arr : FixedArray[Int]) -> Int {
  scalar_sum(arr)
}

///|
/// SIMD dot product - js target: scalar fallback.
pub fn simd_dot_product(a : FixedArray[Int], b : FixedArray[Int]) -> Int {
  scalar_dot_product(a, b)
}

///|
/// SIMD element-wise add - js target: scalar fallback.
pub fn simd_add(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  scalar_add(a, b, out)
}

///|
/// SIMD Adler-32 - js target: scalar fallback.
pub fn simd_adler32(data : FixedArray[Byte]) -> UInt {
  scalar_adler32(data)
}
