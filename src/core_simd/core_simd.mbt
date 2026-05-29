// core_simd — drop-in faster equivalents of common moonbitlang/core
// operations, backed by mizchi/simd's SIMD kernels.
//
// This package is an *opt-in* companion to moonbitlang/core. It does not
// (and cannot) replace core transparently — instead it offers functions
// with the same shape as the core idioms they stand in for, so a hot path
// can be swapped over one call at a time:
//
//     a.iter().fold(init=0, Int::op_add)   ->   @core_simd.sum(a)
//     a.iter().maximum()                   ->   @core_simd.maximum(a)
//     a.search(x)                          ->   @core_simd.search(a, x)
//     a.sort()  (FixedArray[Int])          ->   @core_simd.sort(a)
//
// SIMD acceleration is target-conditional, inherited from the underlying
// mizchi/simd kernels:
//   - wasm     : inline-WAT v128 (the real speedup, 5-15x on these ops)
//   - native   : C FFI / scalar auto-vectorisation
//   - wasm-gc  : scalar fallback (GC-ref FixedArray can't be v128.load-ed)
//   - js       : scalar fallback (no SIMD escape hatch)
// On every target the *result* is identical to the core idiom; only the
// throughput differs. So this is always safe to substitute.
//
// Inputs are `FixedArray[Int]` / `FixedArray[Byte]` (the layouts that cross
// the wasm FFI boundary). For `Array[Int]`, call `.to_fixedarray()` first,
// or keep the hot buffer as a `FixedArray` — note that copying an
// `Array` to a `FixedArray` per call usually erases the SIMD win, so reuse
// the backing storage.

// ---------------------------------------------------------------------------
// FixedArray[Int]
// ---------------------------------------------------------------------------

///|
/// Sum of all elements. Replaces `a.iter().fold(init=0, Int::op_add)`.
/// Empty array sums to 0.
pub fn sum(a : FixedArray[Int]) -> Int {
  @simd.sum_i32(a)
}

///|
/// Product of all elements. Replaces `a.iter().fold(init=1, Int::op_mul)`.
/// Empty array has product 1.
pub fn product(a : FixedArray[Int]) -> Int {
  @simd.prod_i32(a)
}

///|
/// Dot product of two equal-length arrays. Replaces the explicit
/// `for`-loop accumulator. Caller guarantees `a.length() == b.length()`.
pub fn dot(a : FixedArray[Int], b : FixedArray[Int]) -> Int {
  @simd.dot_i32(a, b)
}

///|
/// Largest element, or `None` if empty. Replaces `a.iter().maximum()`.
pub fn maximum(a : FixedArray[Int]) -> Int? {
  if a.length() == 0 {
    None
  } else {
    Some(@simd.max_i32(a))
  }
}

///|
/// Smallest element, or `None` if empty. Replaces `a.iter().minimum()`.
pub fn minimum(a : FixedArray[Int]) -> Int? {
  if a.length() == 0 {
    None
  } else {
    Some(@simd.min_i32(a))
  }
}

///|
/// Index of the first element equal to `needle`, or `None` if absent.
/// Replaces `a.search(needle)`.
pub fn search(a : FixedArray[Int], needle : Int) -> Int? {
  let i = @simd.find_i32(a, needle)
  if i < 0 {
    None
  } else {
    Some(i)
  }
}

///|
/// Whether `needle` occurs in `a`. Replaces `a.contains(needle)`.
pub fn contains(a : FixedArray[Int], needle : Int) -> Bool {
  @simd.find_i32(a, needle) >= 0
}

///|
/// Set every element to `value`, in place. Replaces `a.fill(value)`.
pub fn fill(a : FixedArray[Int], value : Int) -> Unit {
  @simd.fill_i32(a, value)
}

///|
/// Sort the array ascending, in place. Replaces `a.sort()` for
/// `FixedArray[Int]` (bottom-up merge sort with SIMD leaf + bitonic merges).
pub fn sort(a : FixedArray[Int]) -> Unit {
  @simd.sort_i32(a)
}

// ---------------------------------------------------------------------------
// FixedArray[Byte]
// ---------------------------------------------------------------------------

///|
/// Whether two byte arrays are element-wise equal. Replaces a manual
/// length-check + `for`-loop compare. Returns `false` when lengths differ.
pub fn bytes_equal(a : FixedArray[Byte], b : FixedArray[Byte]) -> Bool {
  let len = a.length()
  if len != b.length() {
    return false
  }
  @simd.equal_bytes(a, b, len)
}

///|
/// Index of the first byte equal to `needle`, or `None` if absent.
pub fn bytes_search(data : FixedArray[Byte], needle : Byte) -> Int? {
  let i = @simd.find_byte(data, needle)
  if i < 0 {
    None
  } else {
    Some(i)
  }
}

///|
/// Number of bytes equal to `needle`.
pub fn bytes_count(data : FixedArray[Byte], needle : Byte) -> Int {
  @simd.count_byte(data, needle)
}

///|
/// Whether every byte is < 0x80 (pure ASCII).
pub fn bytes_is_ascii(data : FixedArray[Byte]) -> Bool {
  @simd.is_ascii(data)
}

///|
/// Lowercase ASCII letters (A-Z) in place; other bytes untouched.
pub fn bytes_to_lower_ascii(data : FixedArray[Byte]) -> Unit {
  @simd.to_lower_ascii(data)
}

///|
/// Uppercase ASCII letters (a-z) in place; other bytes untouched.
pub fn bytes_to_upper_ascii(data : FixedArray[Byte]) -> Unit {
  @simd.to_upper_ascii(data)
}
