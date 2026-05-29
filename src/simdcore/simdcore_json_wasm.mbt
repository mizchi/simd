// JSON structural indexing — wasm SIMD fast path, `Bytes`-direct.
//
// Ports the simdjson `find_structural_bits` pipeline to operate on core's
// `Bytes` (input) and `FixedArray[Int]` (bitmaps / indices) directly, with
// no `@simd_buffer` copy-hop. The inline-WAT bodies are the same as
// `src/simdjson/simdjson_wasm.mbt` — those take raw linear-memory addresses,
// and `Bytes` / `FixedArray[Int]` cross the FFI as exactly such pointers, so
// only the parameter types change.
//
// Honest ceiling (from the simdjson notes): the byte *classification* is
// 5–7x on wasm, but the full structural-index pipeline is only ~1.27x because
// `compute_quote_mask` has a serial bit-level dependency and wasm SIMD has no
// CLMUL to vectorise the prefix-XOR over quote positions. The classify
// primitive on its own is the real win; the full indexer is offered for
// completeness with that caveat.

///|
/// Bit `i % 32` of `out[i/32]` OR'd with 1 iff `src[i]` ∈ `{ } [ ] , :`.
/// 16-byte chunk: 6 × i8x16.eq + chained v128.or + i8x16.bitmask; scalar tail.
#borrow(src, out)
fn buf_classify_structural_b(
  src : Bytes,
  len : Int,
  out : FixedArray[Int],
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (local i32) (local i32) (local i32) (local i32) (local v128) local.get 1 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 3 block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 0 local.get 4 i32.add v128.load local.set 7 local.get 7 i32.const 123 i8x16.splat i8x16.eq local.get 7 i32.const 125 i8x16.splat i8x16.eq v128.or local.get 7 i32.const 91 i8x16.splat i8x16.eq v128.or local.get 7 i32.const 93 i8x16.splat i8x16.eq v128.or local.get 7 i32.const 44 i8x16.splat i8x16.eq v128.or local.get 7 i32.const 58 i8x16.splat i8x16.eq v128.or i8x16.bitmask local.get 4 i32.const 16 i32.and i32.shl local.set 5 local.get 2 local.get 4 i32.const 5 i32.shr_u i32.const 2 i32.shl i32.add local.tee 6 i32.load local.get 5 i32.or local.set 5 local.get 6 local.get 5 i32.store local.get 4 i32.const 16 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 1 i32.ge_s br_if 1 local.get 0 local.get 4 i32.add i32.load8_u local.set 5 local.get 5 i32.const 123 i32.eq local.get 5 i32.const 125 i32.eq i32.or local.get 5 i32.const 91 i32.eq i32.or local.get 5 i32.const 93 i32.eq i32.or local.get 5 i32.const 44 i32.eq i32.or local.get 5 i32.const 58 i32.eq i32.or local.get 4 i32.const 31 i32.and i32.shl local.set 5 local.get 2 local.get 4 i32.const 5 i32.shr_u i32.const 2 i32.shl i32.add local.tee 6 i32.load local.get 5 i32.or local.set 5 local.get 6 local.get 5 i32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
/// In-string bitmask (honours `\"` / `\\`). Fully scalar bit-walk — no
/// CLMUL on wasm. OR-in semantics, so the caller zeroes `out` first.
#borrow(src, out)
fn buf_compute_quote_mask_b(
  src : Bytes,
  len : Int,
  out : FixedArray[Int],
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (local i32) (local i32) (local i32) (local i32) (local i32) (local i32) (local i32) (local i32) (local i32) (local i32) (local i32) (local i32) block loop local.get 3 local.get 1 i32.ge_s br_if 1 local.get 0 local.get 3 i32.add i32.load8_u local.set 6 local.get 6 i32.const 34 i32.eq local.set 8 local.get 6 i32.const 92 i32.eq local.set 9 local.get 9 local.get 4 i32.and local.set 10 i32.const 1 local.get 4 local.get 10 select local.set 12 i32.const 0 local.get 12 local.get 8 select local.set 12 local.get 4 local.get 12 local.get 5 select local.set 11 i32.const 0 local.get 10 local.get 5 select local.set 14 local.get 4 i32.const 1 i32.xor local.get 4 local.get 8 select local.set 12 local.get 4 local.get 12 local.get 5 select local.set 13 local.get 13 local.set 4 local.get 14 local.set 5 local.get 2 local.get 3 i32.const 5 i32.shr_u i32.const 2 i32.shl i32.add local.set 7 local.get 7 local.get 7 i32.load local.get 11 local.get 3 i32.const 31 i32.and i32.shl i32.or i32.store local.get 3 i32.const 1 i32.add local.set 3 br 0 end end)

///|
/// Walk `structural & ~quote_mask` word-by-word, emit byte offsets via
/// i32.ctz + `eff &= eff - 1`. Returns the count written into `indices`.
#borrow(structural, quote_mask, indices)
fn buf_extract_structural_indices_b(
  structural : FixedArray[Int],
  quote_mask : FixedArray[Int],
  n : Int,
  indices : FixedArray[Int],
) -> Int =
  #|(func (param i32) (param i32) (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local i32) (local i32) local.get 2 i32.const 31 i32.add i32.const 5 i32.shr_u local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 0 local.get 4 i32.const 2 i32.shl i32.add i32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add i32.load i32.const -1 i32.xor i32.and local.set 7 block loop local.get 7 i32.eqz br_if 1 local.get 7 i32.ctz local.set 8 local.get 3 local.get 6 i32.const 2 i32.shl i32.add local.get 4 i32.const 5 i32.shl local.get 8 i32.add i32.store local.get 6 i32.const 1 i32.add local.set 6 local.get 7 local.get 7 i32.const 1 i32.sub i32.and local.set 7 br 0 end end local.get 4 i32.const 1 i32.add local.set 4 br 0 end end local.get 6)

///|
/// Zero the first `words` i32 slots of `out`.
#borrow(out)
fn buf_zero_words_b(out : FixedArray[Int], words : Int) -> Unit =
  #|(func (param i32) (param i32) (local i32) block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 0 local.get 2 i32.const 2 i32.shl i32.add i32.const 0 i32.store local.get 2 i32.const 1 i32.add local.set 2 br 0 end end)

///|
/// Classify structural bytes: bit `i` of `out` is set iff `input[i]` ∈
/// `{ } [ ] , :`. `out` must hold at least `(len + 31) / 32` i32 words; it is
/// zeroed first. This is the fast primitive (5–7x scalar on wasm); it does
/// NOT exclude structural characters that occur inside string literals — use
/// `json_structural_indices` for that.
pub fn json_classify_structural(input : Bytes, out : FixedArray[Int]) -> Unit {
  let n = input.length()
  let words = (n + 31) / 32
  if out.length() < words {
    abort("simdcore.json_classify_structural: out too small")
  }
  buf_zero_words_b(out, words)
  if n == 0 {
    return
  }
  buf_classify_structural_b(input, n, out)
}

///|
/// Full structural-index pipeline into caller-supplied scratch, returning the
/// number of indices written. `structural` / `quote` need `(len+31)/32` i32
/// words each; `indices` needs up to `len` i32 slots. Reuse the buffers
/// across calls on hot paths. NOTE: ~1.27x scalar — `compute_quote_mask` is
/// the serial, CLMUL-less floor.
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
  buf_zero_words_b(structural, words)
  buf_classify_structural_b(input, n, structural)
  buf_zero_words_b(quote, words)
  buf_compute_quote_mask_b(input, n, quote)
  buf_extract_structural_indices_b(structural, quote, n, indices)
}

///|
/// Convenience: structural byte offsets of `input` (excluding those inside
/// string literals), as a fresh `Array[Int]`. Allocates scratch + indices
/// per call — use `json_structural_indices_into` with reused buffers on hot
/// paths.
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
