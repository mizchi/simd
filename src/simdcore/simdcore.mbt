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
// Inputs are `FixedArray[Int]` / `FixedArray[Byte]` / `Bytes` / `String`
// (the layouts that cross the wasm FFI boundary). For core's growable
// `Array[T]`, use the `Array[T]` bridge at the bottom of this file
// (`to_fixedarray` / `of_fixedarray` + `array_*` one-shots): it copies once
// into a `FixedArray`, which still beats the idiomatic `a.iter().fold(...)`
// baseline — reuse the `FixedArray` if you run many ops.

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
/// Largest element, or `None` if empty. The f64 counterpart of `maximum`.
/// Replaces `a.iter().maximum()` for `FixedArray[Double]`.
pub fn maximum_f64(a : FixedArray[Double]) -> Double? {
  if a.length() == 0 {
    None
  } else {
    Some(@simd.max_f64(a))
  }
}

///|
/// Smallest element, or `None` if empty. The f64 counterpart of `minimum`.
/// Replaces `a.iter().minimum()` for `FixedArray[Double]`.
pub fn minimum_f64(a : FixedArray[Double]) -> Double? {
  if a.length() == 0 {
    None
  } else {
    Some(@simd.min_f64(a))
  }
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

///|
/// Index of the first occurrence of the `needle` byte sequence in `haystack`,
/// or `None` if absent. Empty `needle` returns `Some(0)`. (core has no
/// `Bytes` substring search.) Native uses `memmem`; wasm SIMD-scans the first
/// byte then verifies; wasm-gc / js are scalar.
pub fn bytes_index_of(haystack : Bytes, needle : Bytes) -> Int? {
  let i = @simd.find_bytes_b(haystack, needle)
  if i < 0 {
    None
  } else {
    Some(i)
  }
}

///|
/// Whether `needle` occurs as a contiguous subsequence of `haystack`.
pub fn bytes_contains_sub(haystack : Bytes, needle : Bytes) -> Bool {
  @simd.find_bytes_b(haystack, needle) >= 0
}

///|
/// Index of the *last* byte equal to `needle`, or `None` if absent. Native
/// uses `memrchr`; wasm reverse-scans with SIMD; wasm-gc / js are scalar.
pub fn bytes_rindex(data : Bytes, needle : Byte) -> Int? {
  let i = @simd.rfind_byte_b(data, needle)
  if i < 0 {
    None
  } else {
    Some(i)
  }
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

// ---------------------------------------------------------------------------
// Array[T] bridge
//
// core's growable `Array[T]` is the type most user code holds, but SIMD needs
// the contiguous `FixedArray[T]` layout and there is no public zero-copy
// `Array -> FixedArray`. So the bridge copies in (`to_fixedarray`) and wraps
// out zero-copy (`of_fixedarray`, via `Array::from_fixed_array`).
//
// The copy is cheap relative to the *idiomatic* core baseline a user writes:
// `a.iter().fold(...)` / `a.iter().maximum()` carry per-element closure
// overhead, so copy + SIMD still beats them (~6x for reductions, ~9x for
// sort on n=1024 — see CLAUDE.md). It will NOT beat a hand-tuned raw `for`
// loop on the cheapest single reductions, and the copy is wasted if you call
// many ops: in that case `to_fixedarray` once and reuse the FixedArray.
// ---------------------------------------------------------------------------

///|
/// Copy an `Array[Int]` into a fresh `FixedArray[Int]` (the SIMD-ready layout).
pub fn to_fixedarray(a : Array[Int]) -> FixedArray[Int] {
  FixedArray::from_array(a[:])
}

///|
/// Wrap a `FixedArray[Int]` as an `Array[Int]` (zero-copy via
/// `Array::from_fixed_array`). Pair with `to_fixedarray` to round-trip.
pub fn of_fixedarray(a : FixedArray[Int]) -> Array[Int] {
  Array::from_fixed_array(a)
}

///|
/// Copy an `Array[Double]` into a fresh `FixedArray[Double]`.
pub fn to_fixedarray_f64(a : Array[Double]) -> FixedArray[Double] {
  FixedArray::from_array(a[:])
}

///|
/// Wrap a `FixedArray[Double]` as an `Array[Double]` (zero-copy).
pub fn of_fixedarray_f64(a : FixedArray[Double]) -> Array[Double] {
  Array::from_fixed_array(a)
}

// --- Array[Int] one-shot convenience (copy-in; beats the iterator idiom) ---

///|
/// Sum of an `Array[Int]`. Replaces `a.iter().fold(init=0, ..)`.
pub fn array_sum(a : Array[Int]) -> Int {
  sum(to_fixedarray(a))
}

///|
/// Product of an `Array[Int]`.
pub fn array_product(a : Array[Int]) -> Int {
  product(to_fixedarray(a))
}

///|
/// Largest element of an `Array[Int]`, or `None` if empty.
pub fn array_maximum(a : Array[Int]) -> Int? {
  maximum(to_fixedarray(a))
}

///|
/// Smallest element of an `Array[Int]`, or `None` if empty.
pub fn array_minimum(a : Array[Int]) -> Int? {
  minimum(to_fixedarray(a))
}

///|
/// Dot product of two equal-length `Array[Int]`.
pub fn array_dot(a : Array[Int], b : Array[Int]) -> Int {
  dot(to_fixedarray(a), to_fixedarray(b))
}

///|
/// Number of non-zero elements of an `Array[Int]`.
pub fn array_count_nonzero(a : Array[Int]) -> Int {
  count_nonzero(to_fixedarray(a))
}

///|
/// Sort an `Array[Int]` ascending, in place (copy → SIMD merge sort → blit
/// back). ~9x `Array::sort` on n=1024 despite the round-trip copies.
pub fn array_sort(a : Array[Int]) -> Unit {
  let n = a.length()
  if n <= 1 {
    return
  }
  let fa = FixedArray::from_array(a[:])
  @simd.sort_i32(fa)
  a.unsafe_blit_fixed(0, fa, 0, n)
}

// --- Array[Double] one-shot convenience ---

///|
/// Sum of an `Array[Double]`.
pub fn array_sum_f64(a : Array[Double]) -> Double {
  sum_f64(to_fixedarray_f64(a))
}

///|
/// Dot product of two equal-length `Array[Double]`.
pub fn array_dot_f64(a : Array[Double], b : Array[Double]) -> Double {
  dot_f64(to_fixedarray_f64(a), to_fixedarray_f64(b))
}

///|
/// Arithmetic mean of an `Array[Double]`.
pub fn array_mean_f64(a : Array[Double]) -> Double {
  mean_f64(to_fixedarray_f64(a))
}

// ---------------------------------------------------------------------------
// JSON convenience (target-agnostic: delegates to the per-target
// `json_structural_indices_into`)
// ---------------------------------------------------------------------------

///|
/// Structural byte offsets of `input` (excluding those inside string
/// literals), as a fresh `Array[Int]`. Allocates scratch + indices per call —
/// use `json_structural_indices_into` with reused buffers on hot paths.
pub fn json_structural_indices(input : Bytes) -> Array[Int] {
  let n = input.length()
  if n == 0 {
    return Array::new()
  }
  let words = (n + 31) / 32
  let structural = FixedArray::make(words, 0)
  let quote = FixedArray::make(words, 0)
  let indices = FixedArray::make(n, 0)
  let count = json_structural_indices_into(input, structural, quote, indices)
  let result = Array::new(capacity=count)
  for i in 0..<count {
    result.push(indices[i])
  }
  result
}
