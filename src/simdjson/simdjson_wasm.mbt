// simdjson-style JSON byte classification pipeline for wasm + wasm-gc.
//
// Built on top of `@simd_buffer.SimdBufferBytes` (input) and
// `@simd_buffer.SimdBuffer` (i32 bitmap output). The public API is the
// same as the scalar fallback in `simdjson_scalar.mbt`; this file
// dispatches to inline-WAT v128 primitives via `raw_addr()`.
//
// Pipeline phases:
//   1. classify_structural   — `{ } [ ] , :`     → bitmask (SIMD)
//   2. classify_numeric      — `0-9 - + . e E`   → bitmask (SIMD)
//   3. classify_quote_raw    — `"`               → bitmask (SIMD)
//   4. classify_backslash    — `\`               → bitmask (SIMD)
//   5. compute_quote_mask    — in-string bitmask (scalar bit-walk)
//   6. extract_structural_indices — bit walk → byte offsets (scalar ctz)
//
// The quote mask stays scalar: wasm SIMD has no CLMUL, so the prefix
// XOR over quote bits that simdjson's x86 path does in one instruction
// can't be vectorised here.

///|
/// Bit `i % 32` of `out[i / 32]` set iff input byte `i` ∈ `{ } [ ] , :`.
/// **OR-in** semantics — call `out.clear()` first if you want a fresh
/// mask. `out` must have `(len + 31) / 32` slots.
pub fn classify_structural(
  input : @simd_buffer.SimdBufferBytes,
  out : @simd_buffer.SimdBuffer,
) -> Unit {
  let n = input.length()
  let words = (n + 31) / 32
  if out.length() < words {
    abort("simdjson.classify_structural: out too small")
  }
  if n == 0 {
    return
  }
  buf_classify_structural(input.raw_addr(), n, out.raw_addr())
}

///|
/// Bit `i % 32` of `out[i / 32]` set iff input byte `i` ∈ `0-9 - + . e E`.
/// OR-in semantics.
pub fn classify_numeric(
  input : @simd_buffer.SimdBufferBytes,
  out : @simd_buffer.SimdBuffer,
) -> Unit {
  let n = input.length()
  let words = (n + 31) / 32
  if out.length() < words {
    abort("simdjson.classify_numeric: out too small")
  }
  if n == 0 {
    return
  }
  buf_classify_numeric(input.raw_addr(), n, out.raw_addr())
}

///|
/// Bit `i % 32` of `out[i / 32]` set iff input byte `i` == `"`. OR-in
/// semantics. Raw quote positions — backslash escapes are NOT processed
/// here; that's `compute_quote_mask`'s job.
pub fn classify_quote_raw(
  input : @simd_buffer.SimdBufferBytes,
  out : @simd_buffer.SimdBuffer,
) -> Unit {
  let n = input.length()
  let words = (n + 31) / 32
  if out.length() < words {
    abort("simdjson.classify_quote_raw: out too small")
  }
  if n == 0 {
    return
  }
  buf_classify_byte_eq(input.raw_addr(), n, out.raw_addr(), 0x22)
}

///|
/// Bit `i % 32` of `out[i / 32]` set iff input byte `i` == `\`. OR-in
/// semantics.
pub fn classify_backslash(
  input : @simd_buffer.SimdBufferBytes,
  out : @simd_buffer.SimdBuffer,
) -> Unit {
  let n = input.length()
  let words = (n + 31) / 32
  if out.length() < words {
    abort("simdjson.classify_backslash: out too small")
  }
  if n == 0 {
    return
  }
  buf_classify_byte_eq(input.raw_addr(), n, out.raw_addr(), 0x5C)
}

///|
/// Compute the JSON in-string bitmask: bit `i % 32` of `out[i / 32]` is 1
/// iff byte `i` lies strictly inside a string literal (between opening
/// and closing `"`, exclusive). Honours `\"` and `\\` escapes.
///
/// **Overwrites** `out` (caller does not need to clear).
///
/// Stays scalar — wasm SIMD has no CLMUL, so the prefix XOR over quote
/// positions runs as a branchless `select`-driven bit-walk in inline-WAT.
pub fn compute_quote_mask(
  input : @simd_buffer.SimdBufferBytes,
  out : @simd_buffer.SimdBuffer,
) -> Unit {
  let n = input.length()
  let words = (n + 31) / 32
  if out.length() < words {
    abort("simdjson.compute_quote_mask: out too small")
  }
  // Clear the first `words` slots (single inline-WAT call).
  buf_zero_words(out.raw_addr(), words)
  if n == 0 {
    return
  }
  buf_compute_quote_mask(input.raw_addr(), n, out.raw_addr())
}

///|
/// Walk `structural & ~quote_mask` word-by-word and emit byte offsets of
/// remaining bits via `i32.ctz` + `effective &= effective - 1`. Returns
/// the count written. `indices` must hold at least as many i32 slots as
/// there are real structural chars in the input — `n` (input byte
/// length) is a safe upper bound.
pub fn extract_structural_indices(
  structural : @simd_buffer.SimdBuffer,
  quote_mask : @simd_buffer.SimdBuffer,
  n : Int,
  indices : @simd_buffer.SimdBuffer,
) -> Int {
  let words = (n + 31) / 32
  if structural.length() < words || quote_mask.length() < words {
    abort("simdjson.extract_structural_indices: bitmap buffers too small")
  }
  if n == 0 {
    return 0
  }
  buf_extract_structural_indices(
    structural.raw_addr(),
    quote_mask.raw_addr(),
    n,
    indices.raw_addr(),
  )
}

///|
/// Convenience: run the full pipeline (classify_structural +
/// compute_quote_mask + extract_structural_indices). Allocates two
/// scratch SimdBuffer bitmaps internally — each `SimdBuffer::make`
/// triggers `memory.grow` (~hundreds of µs), so use the
/// `_with_scratch` variant in hot paths.
pub fn find_structural_indices(
  input : @simd_buffer.SimdBufferBytes,
  indices : @simd_buffer.SimdBuffer,
) -> Int {
  let n = input.length()
  if n == 0 {
    return 0
  }
  let words = (n + 31) / 32
  let structural = @simd_buffer.SimdBuffer::make(words)
  let quote_mask = @simd_buffer.SimdBuffer::make(words)
  find_structural_indices_with_scratch(input, structural, quote_mask, indices)
}

///|
/// Scratch-receiving variant of `find_structural_indices`. The two
/// scratch buffers must each have at least `(len + 31) / 32` i32 slots.
/// `structural_scratch` is cleared here; `compute_quote_mask` clears
/// `quote_mask_scratch` on its own.
pub fn find_structural_indices_with_scratch(
  input : @simd_buffer.SimdBufferBytes,
  structural_scratch : @simd_buffer.SimdBuffer,
  quote_mask_scratch : @simd_buffer.SimdBuffer,
  indices : @simd_buffer.SimdBuffer,
) -> Int {
  let n = input.length()
  if n == 0 {
    return 0
  }
  let words = (n + 31) / 32
  if structural_scratch.length() < words || quote_mask_scratch.length() < words {
    abort("simdjson.find_structural_indices_with_scratch: scratch too small")
  }
  buf_zero_words(structural_scratch.raw_addr(), words)
  classify_structural(input, structural_scratch)
  compute_quote_mask(input, quote_mask_scratch)
  extract_structural_indices(structural_scratch, quote_mask_scratch, n, indices)
}

// ---- inline-WAT primitives (linear-memory layouts) ----

///|
/// SIMD JSON structural classification — bit `i % 32` of `out[i / 32]`
/// is OR'd with 1 iff `src[i]` ∈ `{ } [ ] , :`. 16-byte chunk via 6 ×
/// `i8x16.eq` + chained `v128.or` + `i8x16.bitmask`. Tail is scalar.
///
/// Locals: 3 = vec_end, 4 = i, 5 = scratch, 6 = word_addr, 7 = v.
fn buf_classify_structural(addr : Int, len : Int, out : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (local i32) (local i32) (local i32) (local i32) (local v128) local.get 1 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 3 block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 0 local.get 4 i32.add v128.load local.set 7 local.get 7 i32.const 123 i8x16.splat i8x16.eq local.get 7 i32.const 125 i8x16.splat i8x16.eq v128.or local.get 7 i32.const 91 i8x16.splat i8x16.eq v128.or local.get 7 i32.const 93 i8x16.splat i8x16.eq v128.or local.get 7 i32.const 44 i8x16.splat i8x16.eq v128.or local.get 7 i32.const 58 i8x16.splat i8x16.eq v128.or i8x16.bitmask local.get 4 i32.const 16 i32.and i32.shl local.set 5 local.get 2 local.get 4 i32.const 5 i32.shr_u i32.const 2 i32.shl i32.add local.tee 6 i32.load local.get 5 i32.or local.set 5 local.get 6 local.get 5 i32.store local.get 4 i32.const 16 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 1 i32.ge_s br_if 1 local.get 0 local.get 4 i32.add i32.load8_u local.set 5 local.get 5 i32.const 123 i32.eq local.get 5 i32.const 125 i32.eq i32.or local.get 5 i32.const 91 i32.eq i32.or local.get 5 i32.const 93 i32.eq i32.or local.get 5 i32.const 44 i32.eq i32.or local.get 5 i32.const 58 i32.eq i32.or local.get 4 i32.const 31 i32.and i32.shl local.set 5 local.get 2 local.get 4 i32.const 5 i32.shr_u i32.const 2 i32.shl i32.add local.tee 6 i32.load local.get 5 i32.or local.set 5 local.get 6 local.get 5 i32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
/// SIMD single-byte classification — bit `i % 32` of `out[i / 32]` is
/// OR'd with 1 iff `src[i] == needle`. Used for `"` and `\` raw maps.
///
/// Locals: 4 = vec_end, 5 = i, 6 = scratch, 7 = word_addr, 8 = needle_v.
fn buf_classify_byte_eq(addr : Int, len : Int, out : Int, needle : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) (local i32) (local i32) (local v128) local.get 3 i8x16.splat local.set 8 local.get 1 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 4 block loop local.get 5 local.get 4 i32.ge_s br_if 1 local.get 0 local.get 5 i32.add v128.load local.get 8 i8x16.eq i8x16.bitmask local.get 5 i32.const 16 i32.and i32.shl local.set 6 local.get 2 local.get 5 i32.const 5 i32.shr_u i32.const 2 i32.shl i32.add local.tee 7 i32.load local.get 6 i32.or local.set 6 local.get 7 local.get 6 i32.store local.get 5 i32.const 16 i32.add local.set 5 br 0 end end block loop local.get 5 local.get 1 i32.ge_s br_if 1 local.get 0 local.get 5 i32.add i32.load8_u local.get 3 i32.eq local.get 5 i32.const 31 i32.and i32.shl local.set 6 local.get 2 local.get 5 i32.const 5 i32.shr_u i32.const 2 i32.shl i32.add local.tee 7 i32.load local.get 6 i32.or local.set 6 local.get 7 local.get 6 i32.store local.get 5 i32.const 1 i32.add local.set 5 br 0 end end)

///|
/// SIMD JSON-numeric classification — bit set iff byte ∈ `0-9 - + . e E`.
/// Range check (digits) via `i8x16.ge_s` / `i8x16.le_s` + `v128.and`,
/// plus 5 × `i8x16.eq` ORed in.
///
/// Locals: 3 = vec_end, 4 = i, 5 = scratch, 6 = word_addr, 7 = v.
fn buf_classify_numeric(addr : Int, len : Int, out : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (local i32) (local i32) (local i32) (local i32) (local v128) local.get 1 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 3 block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 0 local.get 4 i32.add v128.load local.set 7 local.get 7 i32.const 48 i8x16.splat i8x16.ge_s local.get 7 i32.const 57 i8x16.splat i8x16.le_s v128.and local.get 7 i32.const 45 i8x16.splat i8x16.eq v128.or local.get 7 i32.const 46 i8x16.splat i8x16.eq v128.or local.get 7 i32.const 43 i8x16.splat i8x16.eq v128.or local.get 7 i32.const 101 i8x16.splat i8x16.eq v128.or local.get 7 i32.const 69 i8x16.splat i8x16.eq v128.or i8x16.bitmask local.get 4 i32.const 16 i32.and i32.shl local.set 5 local.get 2 local.get 4 i32.const 5 i32.shr_u i32.const 2 i32.shl i32.add local.tee 6 i32.load local.get 5 i32.or local.set 5 local.get 6 local.get 5 i32.store local.get 4 i32.const 16 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 1 i32.ge_s br_if 1 local.get 0 local.get 4 i32.add i32.load8_u local.set 5 local.get 5 i32.const 48 i32.ge_s local.get 5 i32.const 57 i32.le_s i32.and local.get 5 i32.const 45 i32.eq i32.or local.get 5 i32.const 46 i32.eq i32.or local.get 5 i32.const 43 i32.eq i32.or local.get 5 i32.const 101 i32.eq i32.or local.get 5 i32.const 69 i32.eq i32.or local.get 4 i32.const 31 i32.and i32.shl local.set 5 local.get 2 local.get 4 i32.const 5 i32.shr_u i32.const 2 i32.shl i32.add local.tee 6 i32.load local.get 5 i32.or local.set 5 local.get 6 local.get 5 i32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
/// Bit-walk computing the in-string bitmask. Fully scalar — no SIMD
/// instruction can prefix-XOR the quote bits without CLMUL.
///
/// Locals: 3 = i, 4 = in_string (0/1), 5 = escape_pending (0/1),
/// 6 = b, 7 = word_addr, 8 = is_quote, 9 = is_backslash,
/// 10 = bs_and_string, 11 = bit_set, 12 = tmp, 13 = in_string_next,
/// 14 = escape_pending_next.
fn buf_compute_quote_mask(src : Int, len : Int, out : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (local i32) (local i32) (local i32) (local i32) (local i32) (local i32) (local i32) (local i32) (local i32) (local i32) (local i32) (local i32) block loop local.get 3 local.get 1 i32.ge_s br_if 1 local.get 0 local.get 3 i32.add i32.load8_u local.set 6 local.get 6 i32.const 34 i32.eq local.set 8 local.get 6 i32.const 92 i32.eq local.set 9 local.get 9 local.get 4 i32.and local.set 10 i32.const 1 local.get 4 local.get 10 select local.set 12 i32.const 0 local.get 12 local.get 8 select local.set 12 local.get 4 local.get 12 local.get 5 select local.set 11 i32.const 0 local.get 10 local.get 5 select local.set 14 local.get 4 i32.const 1 i32.xor local.get 4 local.get 8 select local.set 12 local.get 4 local.get 12 local.get 5 select local.set 13 local.get 13 local.set 4 local.get 14 local.set 5 local.get 2 local.get 3 i32.const 5 i32.shr_u i32.const 2 i32.shl i32.add local.set 7 local.get 7 local.get 7 i32.load local.get 11 local.get 3 i32.const 31 i32.and i32.shl i32.or i32.store local.get 3 i32.const 1 i32.add local.set 3 br 0 end end)

///|
/// Walk a structural-minus-quote bitmap word-by-word and emit byte
/// offsets via `i32.ctz` + `effective &= effective - 1`. Returns count.
///
/// Locals: 4 = w (word idx), 5 = num_words, 6 = count, 7 = effective,
/// 8 = bit (ctz result).
fn buf_extract_structural_indices(
  structural : Int,
  quote_mask : Int,
  n : Int,
  indices : Int,
) -> Int =
  #|(func (param i32) (param i32) (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local i32) (local i32) local.get 2 i32.const 31 i32.add i32.const 5 i32.shr_u local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 0 local.get 4 i32.const 2 i32.shl i32.add i32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add i32.load i32.const -1 i32.xor i32.and local.set 7 block loop local.get 7 i32.eqz br_if 1 local.get 7 i32.ctz local.set 8 local.get 3 local.get 6 i32.const 2 i32.shl i32.add local.get 4 i32.const 5 i32.shl local.get 8 i32.add i32.store local.get 6 i32.const 1 i32.add local.set 6 local.get 7 local.get 7 i32.const 1 i32.sub i32.and local.set 7 br 0 end end local.get 4 i32.const 1 i32.add local.set 4 br 0 end end local.get 6)

///|
/// Zero the first `words` i32 slots at `out`. Bulk clear without
/// per-slot FFI overhead from MoonBit-side loops.
fn buf_zero_words(out : Int, words : Int) -> Unit =
  #|(func (param i32) (param i32) (local i32) block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 0 local.get 2 i32.const 2 i32.shl i32.add i32.const 0 i32.store local.get 2 i32.const 1 i32.add local.set 2 br 0 end end)
