// simdcore — drop-in faster equivalents of common moonbitlang/core
// operations, backed by mizchi/simd's SIMD kernels.
//
// This package is an *opt-in* companion to moonbitlang/core. It does not
// (and cannot) replace core transparently — instead it offers functions
// with the same shape as the core idioms they stand in for, so a hot path
// can be swapped over one call at a time:
//
//     a.iter().fold(init=0, Int::op_add)   ->   @simdcore.sum(a)
//     a.iter().maximum()                   ->   @simdcore.maximum(a)
//     a.search(x)                          ->   @simdcore.search(a, x)
//     a.sort()  (FixedArray[Int])          ->   @simdcore.sort(a)
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

///|
/// Number of non-zero elements. Replaces a `fold` / `for`-loop counter.
pub fn count_nonzero(a : FixedArray[Int]) -> Int {
  @simd.count_nonzero_i32(a)
}

// ---------------------------------------------------------------------------
// FixedArray[Int] — element-wise (write into a caller-supplied `out`)
//
// These replace the hand-written zip loop `for i { out[i] = a[i] OP b[i] }`.
// `out` may alias `a` (or `b`) for in-place updates. Caller guarantees the
// lengths match.
// ---------------------------------------------------------------------------

///|
/// `out[i] = a[i] + b[i]`.
pub fn add(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  @simd.add_i32(a, b, out)
}

///|
/// `out[i] = a[i] - b[i]`.
pub fn sub(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  @simd.sub_i32(a, b, out)
}

///|
/// `out[i] = a[i] * b[i]`.
pub fn mul(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  @simd.mul_i32(a, b, out)
}

///|
/// `out[i] = -a[i]`.
pub fn neg(a : FixedArray[Int], out : FixedArray[Int]) -> Unit {
  @simd.neg_i32(a, out)
}

///|
/// `out[i] = abs(a[i])`.
pub fn abs(a : FixedArray[Int], out : FixedArray[Int]) -> Unit {
  @simd.abs_i32(a, out)
}

///|
/// `out[i] = k * a[i] + b[i]` (fused scale-add, "saxpy").
pub fn saxpy(
  k : Int,
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  @simd.saxpy_i32(k, a, b, out)
}

// ---------------------------------------------------------------------------
// FixedArray[Double] — the practical numpy dtype
//
// Reductions return a `Double`; element-wise ops write into `out`. f64 is
// 2-way parallel (f64x2) so the SIMD wins are smaller than i32/byte, but the
// reductions and element-wise math still beat the scalar loop.
// ---------------------------------------------------------------------------

///|
/// Sum of all elements. Replaces `a.iter().fold(init=0.0, ...)`.
pub fn sum_f64(a : FixedArray[Double]) -> Double {
  @simd.sum_f64(a)
}

///|
/// Dot product of two equal-length arrays.
pub fn dot_f64(a : FixedArray[Double], b : FixedArray[Double]) -> Double {
  @simd.dot_f64(a, b)
}

///|
/// Arithmetic mean. Returns 0.0 for an empty array.
pub fn mean_f64(a : FixedArray[Double]) -> Double {
  @simd.mean_f64(a)
}

///|
/// Population variance.
pub fn variance_f64(a : FixedArray[Double]) -> Double {
  @simd.var_f64(a)
}

///|
/// `out[i] = a[i] + b[i]`.
pub fn add_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  @simd.add_f64(a, b, out)
}

///|
/// `out[i] = a[i] - b[i]`.
pub fn sub_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  @simd.sub_f64(a, b, out)
}

///|
/// `out[i] = a[i] * b[i]`.
pub fn mul_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  @simd.mul_f64(a, b, out)
}

///|
/// `out[i] = a[i] / b[i]`.
pub fn div_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  @simd.div_f64(a, b, out)
}

///|
/// `out[i] = sqrt(a[i])`.
pub fn sqrt_f64(a : FixedArray[Double], out : FixedArray[Double]) -> Unit {
  @simd.sqrt_f64(a, out)
}

///|
/// Element-wise minimum: `out[i] = min(a[i], b[i])`.
pub fn min_elem_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  @simd.min_elem_f64(a, b, out)
}

///|
/// Element-wise maximum: `out[i] = max(a[i], b[i])`.
pub fn max_elem_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  @simd.max_elem_f64(a, b, out)
}

// ---------------------------------------------------------------------------
// Bytes (read-only)
//
// These operate on core's immutable `Bytes` directly — no copy. On wasm,
// `Bytes` crosses the inline-WAT FFI as a linear-memory pointer, so these are
// true zero-copy SIMD scans. On wasm-gc `Bytes` is a GC ref (FFI-rejected) so
// these fall back to scalar; native/js are scalar too. Result is identical on
// all targets.
// ---------------------------------------------------------------------------

///|
/// Whether two byte sequences are equal. Replaces `a == b`.
/// Returns `false` when lengths differ.
pub fn bytes_equal(a : Bytes, b : Bytes) -> Bool {
  let len = a.length()
  if len != b.length() {
    return false
  }
  @simd.equal_bytes_b(a, b, len)
}

///|
/// Index of the first byte equal to `needle`, or `None` if absent.
pub fn bytes_search(data : Bytes, needle : Byte) -> Int? {
  let i = @simd.find_byte_b(data, needle)
  if i < 0 {
    None
  } else {
    Some(i)
  }
}

///|
/// Whether `needle` occurs in `data`.
pub fn bytes_contains(data : Bytes, needle : Byte) -> Bool {
  @simd.find_byte_b(data, needle) >= 0
}

///|
/// Number of bytes equal to `needle`.
pub fn bytes_count(data : Bytes, needle : Byte) -> Int {
  @simd.count_byte_b(data, needle)
}

///|
/// Whether every byte is < 0x80 (pure ASCII).
pub fn bytes_is_ascii(data : Bytes) -> Bool {
  @simd.is_ascii_b(data)
}

// ---------------------------------------------------------------------------
// FixedArray[Byte] (in-place mutation)
//
// `Bytes` is immutable, so the in-place ASCII case-mappings target a mutable
// `FixedArray[Byte]`. (A `Bytes -> Bytes` case-map would be allocation-bound,
// so it's deferred — see CLAUDE.md.)
// ---------------------------------------------------------------------------

///|
/// Lowercase ASCII letters (A-Z) in place; other bytes untouched.
pub fn to_lower_ascii(data : FixedArray[Byte]) -> Unit {
  @simd.to_lower_ascii(data)
}

///|
/// Uppercase ASCII letters (a-z) in place; other bytes untouched.
pub fn to_upper_ascii(data : FixedArray[Byte]) -> Unit {
  @simd.to_upper_ascii(data)
}
