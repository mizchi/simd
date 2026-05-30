// ===========================================================================
// "Option A": buffer-backed variants of the simdcore ops, for wasm-gc SIMD.
//
// simdcore's normal surface takes core's `FixedArray` / `Bytes` directly. On
// `wasm` that vectorises (the values cross the inline-WAT FFI as linear-memory
// pointers); on `wasm-gc` it CANNOT — a GC ref can't feed `v128.load`, so those
// ops fall back to scalar there (see the "Option A" section in CLAUDE.md).
//
// The only way a wasm-gc caller holding a `FixedArray` gets SIMD is to copy the
// data into a linear-memory `@simd_buffer.SimdBuffer`, run the SIMD kernel, and
// copy any array output back. These `*_via_buffer` functions do exactly that,
// keeping the SAME signature as their plain simdcore counterparts so they are
// drop-in. They run on all four targets (SIMD on wasm / wasm-gc, scalar-backed
// SimdBuffer on native / js) for API parity.
//
//   ┌─────────────────────────────────────────────────────────────────────┐
//   │ SLOW. Each call here allocates fresh SimdBuffer pages via             │
//   │ `memory.grow` (~hundreds of µs on wasm-gc) AND pays a per-element     │
//   │ inline-WAT FFI copy in/out. Measured on wasm-gc, this makes the light │
//   │ ops a NET LOSS vs the scalar fallback (`sum_via_buffer` ~750× slower  │
//   │ per call than `sum`). Only the compute-heavy `sort_via_buffer` beats  │
//   │ scalar, and only once the per-call allocation is amortised.           │
//   │                                                                       │
//   │ Provided for completeness / experimentation. For a real wasm-gc win,  │
//   │ use `@simd_buffer` directly: pre-allocate the buffers ONCE and reuse  │
//   │ them across calls (or carve them from a `SimdBufferRing`), so the     │
//   │ `memory.grow` cost is paid once, not per call. See the bench in       │
//   │ `simd_buffer_bench.mbt` (`optionA_*`) for the reuse-vs-naive split.   │
//   └─────────────────────────────────────────────────────────────────────┘
// ===========================================================================

///|
/// Buffer-backed `sum`. Same result as `@simdcore.sum`. SLOW — see the file
/// header; on wasm-gc this is a net loss vs the scalar `sum` (the copy hop +
/// per-call allocation outweigh a single reduction pass). Use only when you
/// already hold the data in a reused `@simd_buffer.SimdBuffer`.
pub fn sum_via_buffer(a : FixedArray[Int]) -> Int {
  let buf = @simd_buffer.SimdBuffer::from_array(a)
  buf.sum()
}

///|
/// Buffer-backed `dot`. Same result as `@simdcore.dot`. SLOW — see the file
/// header; net loss on wasm-gc for the same reason as `sum_via_buffer`.
pub fn dot_via_buffer(a : FixedArray[Int], b : FixedArray[Int]) -> Int {
  let ba = @simd_buffer.SimdBuffer::from_array(a)
  let bb = @simd_buffer.SimdBuffer::from_array(b)
  @simd_buffer.SimdBuffer::dot(ba, bb)
}

///|
/// Buffer-backed `sort`. Same result as `@simdcore.sort` (ascending). This is
/// the one op where the SIMD kernel is heavy enough to win on wasm-gc once the
/// per-call allocation is amortised (~6× over scalar in the reuse regime); with
/// the per-call allocation here it is still dominated by `memory.grow`, so for a
/// real win pre-allocate `buf` + `scratch` and call `@simd_buffer.SimdBuffer::
/// sort` directly. Provided here for API parity — like `@simdcore.sort` it
/// sorts `a` IN PLACE (returns `Unit`).
pub fn sort_via_buffer(a : FixedArray[Int]) -> Unit {
  let buf = @simd_buffer.SimdBuffer::from_array(a)
  let scratch = @simd_buffer.SimdBuffer::make(a.length())
  @simd_buffer.SimdBuffer::sort(buf, scratch)
  buf.copy_to_array(a)
}
