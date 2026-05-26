// Fused single-pass SIMD cosine similarity — wasm / wasm-gc only.
//
// This file demonstrates dropping down to user-side inline-WAT when the
// public SimdBuffer ops aren't expressive enough. `cosine_similarity`
// is a three-accumulator fold (dot, na, nb); chaining
// `SimdBufferF64::dot` three times walks the data three times. The
// inline-WAT kernel below threads all three accumulators through a
// single SIMD loop, paying one memory pass.
//
// Built on top of `SimdBufferF64::raw_addr()`, which exposes the
// linear-memory address of element 0. That method is only defined on
// wasm / wasm-gc; this file inherits the same target restriction.

///|
/// Fused-SIMD cosine similarity. `scratch` must have `length() >= 3` —
/// the kernel writes `[dot, na, nb]` into it, then this function
/// finishes with two scalar `sqrt` and one `div`. Pass a Ring-allocated
/// scratch (or a long-lived one) to avoid `memory.grow` per call.
pub fn cosine_similarity_fused_simd(
  a : @simd_buffer.SimdBufferF64,
  b : @simd_buffer.SimdBufferF64,
  scratch : @simd_buffer.SimdBufferF64,
) -> Double {
  let n = a.length()
  if n != b.length() {
    abort("cosine_similarity_fused_simd: length mismatch")
  }
  if scratch.length() < 3 {
    abort("cosine_similarity_fused_simd: scratch must have length >= 3")
  }
  fused_dot3_f64(a.raw_addr(), b.raw_addr(), n, scratch.raw_addr())
  let dot = scratch.get(0)
  let na = scratch.get(1)
  let nb = scratch.get(2)
  dot / (na.sqrt() * nb.sqrt())
}

///|
/// Inline-WAT kernel: walks `a[0..n]` and `b[0..n]` once, accumulating
/// `dot(a,b)` / `dot(a,a)` / `dot(b,b)` in three f64x2 registers. Tail
/// (odd `n`) falls through to a scalar block. Writes `[dot, na, nb]`
/// as three f64s starting at `out`.
///
/// Param 0 = a_addr, 1 = b_addr, 2 = n, 3 = out_addr (24 bytes).
/// Locals:
///   4 i32 chunk_i, 5 i32 vec_end (= n & ~1), 6 i32 scalar_i
///   7 v128 dot_acc, 8 v128 na_acc, 9 v128 nb_acc
///   10 v128 av, 11 v128 bv
///   12 f64 dot_scalar, 13 f64 na_scalar, 14 f64 nb_scalar
///   15 f64 ai, 16 f64 bi
fn fused_dot3_f64(a : Int, b : Int, n : Int, out : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) (local i32) (local v128) (local v128) (local v128) (local v128) (local v128) (local f64) (local f64) (local f64) (local f64) (local f64) i32.const 0 f64.convert_i32_s f64x2.splat local.set 7 i32.const 0 f64.convert_i32_s f64x2.splat local.set 8 i32.const 0 f64.convert_i32_s f64x2.splat local.set 9 local.get 2 i32.const 1 i32.shr_s i32.const 1 i32.shl local.set 5 i32.const 0 local.set 4 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 0 local.get 4 i32.const 3 i32.shl i32.add v128.load local.set 10 local.get 1 local.get 4 i32.const 3 i32.shl i32.add v128.load local.set 11 local.get 7 local.get 10 local.get 11 f64x2.mul f64x2.add local.set 7 local.get 8 local.get 10 local.get 10 f64x2.mul f64x2.add local.set 8 local.get 9 local.get 11 local.get 11 f64x2.mul f64x2.add local.set 9 local.get 4 i32.const 2 i32.add local.set 4 br 0 end end local.get 7 f64x2.extract_lane 0 local.get 7 f64x2.extract_lane 1 f64.add local.set 12 local.get 8 f64x2.extract_lane 0 local.get 8 f64x2.extract_lane 1 f64.add local.set 13 local.get 9 f64x2.extract_lane 0 local.get 9 f64x2.extract_lane 1 f64.add local.set 14 local.get 5 local.set 6 block loop local.get 6 local.get 2 i32.ge_s br_if 1 local.get 0 local.get 6 i32.const 3 i32.shl i32.add f64.load local.set 15 local.get 1 local.get 6 i32.const 3 i32.shl i32.add f64.load local.set 16 local.get 12 local.get 15 local.get 16 f64.mul f64.add local.set 12 local.get 13 local.get 15 local.get 15 f64.mul f64.add local.set 13 local.get 14 local.get 16 local.get 16 f64.mul f64.add local.set 14 local.get 6 i32.const 1 i32.add local.set 6 br 0 end end local.get 3 local.get 12 f64.store local.get 3 i32.const 8 i32.add local.get 13 f64.store local.get 3 i32.const 16 i32.add local.get 14 f64.store)
