# `@simd_buffer` — portable SIMD buffer family

The recommended portable surface: `SimdBuffer` (i32) / `SimdBufferF32` /
`SimdBufferF64` / `SimdBufferBytes` + the `SimdBufferRing` arena allocator.
**The same public API (~50 ops) compiles on all four backends.**

```moonbit
let buf = @simd_buffer.SimdBuffer::from_array([1, 2, 3, 4, 5, 6, 7, 8])
let total = buf.sum()                          // SIMD on wasm/wasm-gc/native
let out = @simd_buffer.SimdBuffer::make(buf.length())
@simd_buffer.SimdBuffer::add(buf, buf, out)
```

## Backend comparison

| backend | storage | mechanism | accelerated? |
|---|---|---|---|
| **wasm** | linear memory (`memory.grow`) | inline-WAT `v128.*` | ✅ SIMD |
| **wasm-gc** | linear memory (`memory.grow`) | inline-WAT `v128.*` (same bodies) | ✅ SIMD |
| **native** | `FixedArray` | `@internal` / `@base64` → C FFI (NEON / SSE2 baseline) | ✅ SIMD |
| **js** | `FixedArray` | scalar via `@internal` | ❌ scalar |

This is the **only** API that gets SIMD on `wasm-gc` — the FixedArray-based
root `@simd` API falls back to scalar there (GC-ref FFI blocks `v128.load`),
whereas `SimdBuffer` owns linear-memory pages so the inline-WAT `v128` bodies
run on both wasm targets. `js` is scalar only (no SIMD escape hatch); it's
offered purely so caller code stays portable.

### Headline speedups (wasm-gc, SIMD vs the FixedArray scalar fallback)

These are what a wasm-gc user gains by porting a hot path from the
FixedArray-based `@simd` API (scalar on wasm-gc) to `SimdBuffer`:

| op | size | scalar fallback | SimdBuffer | x |
|---|---|---|---|---|
| `adler32` | 4096 B | 9.23 µs | 329 ns | **28.1** |
| `popcount` | 4096 B | 6.91 µs | 291 ns | **23.7** |
| `memset` | 4096 B | 1.00 µs | 59 ns | **16.9** |
| `memcpy` | 4096 B | 1.17 µs | 100 ns | **11.7** |
| `find_byte` | 4096 B | 1.16 µs | 230 ns | **5.0** |
| `add` (i32) | 1024 | 413 ns | 118 ns | **3.5** |
| `sum` (i32) | 1024 | 344 ns | 106 ns | **3.2** |

Byte ops show the biggest wins (the scalar fallback pays per-byte call
overhead; SIMD processes 16 bytes per `v128.load`). i32 reductions /
element-wise sit in the 3–4× range.

### Allocator note (wasm / wasm-gc only)

Linear-memory storage comes from `memory.grow` and is **never freed** — fine
for batch / request-scoped workloads. `memory.grow` is a slow host call, so:

- use the `_into` variants (write into a caller-supplied output) on hot paths;
- or carve sub-buffers from a `SimdBufferRing` (one pre-grown page, ~8 ns per
  `alloc` after `reset()` vs ~120 µs for a fresh `make`).

On native / js, allocation is GC-managed, so `Ring` is a thin shell there for
source compatibility.

Run: `moon bench --target wasm-gc -p simd_buffer` (or any target).
