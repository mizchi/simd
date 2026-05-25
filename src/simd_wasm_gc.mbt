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
