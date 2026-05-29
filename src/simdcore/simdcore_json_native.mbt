// JSON structural indexing — native C FFI (Bytes-direct).
//
// `Bytes` / `FixedArray[Int]` cross the native FFI as `const uint8_t*` /
// `int32_t*` under #borrow, so these bind directly to the C kernels in
// `simdcore.c`. Same public API + results as the wasm SIMD path; the C
// loop beats the per-byte MoonBit scalar fallback (and auto-vectorises
// under gcc/clang).

///|
#borrow(src, out)
extern "C" fn simdcore_json_classify_ffi(
  src : Bytes,
  len : Int,
  out : FixedArray[Int],
) -> Unit = "simdcore_json_classify_structural"

///|
#borrow(src, structural, quote, indices)
extern "C" fn simdcore_json_indices_ffi(
  src : Bytes,
  len : Int,
  structural : FixedArray[Int],
  quote : FixedArray[Int],
  indices : FixedArray[Int],
) -> Int = "simdcore_json_indices"

///|
/// Classify structural bytes (`{ } [ ] , :`) into a bitmap. `out` is zeroed
/// first. Native C FFI.
pub fn json_classify_structural(input : Bytes, out : FixedArray[Int]) -> Unit {
  let n = input.length()
  let words = (n + 31) / 32
  if out.length() < words {
    abort("simdcore.json_classify_structural: out too small")
  }
  if n == 0 {
    return
  }
  simdcore_json_classify_ffi(input, n, out)
}

///|
/// Full structural-index pipeline into caller scratch, returning the count.
/// Native C FFI (single call: classify + quote mask + extract).
pub fn json_structural_indices_into(
  input : Bytes,
  structural : FixedArray[Int],
  quote : FixedArray[Int],
  indices : FixedArray[Int],
) -> Int {
  let n = input.length()
  if n == 0 {
    return 0
  }
  let words = (n + 31) / 32
  if structural.length() < words || quote.length() < words {
    abort("simdcore.json_structural_indices_into: scratch too small")
  }
  if indices.length() < n {
    abort("simdcore.json_structural_indices_into: indices too small")
  }
  simdcore_json_indices_ffi(input, n, structural, quote, indices)
}
