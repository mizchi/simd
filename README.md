# mizchi/simd

SIMD primitives for MoonBit. The recommended portable API is the
`SimdBuffer*` family — same public surface on **wasm / wasm-gc /
native / js**, with SIMD acceleration where the target supports it and
a transparent scalar fallback where it doesn't.

```bash
just test          # all 4 targets
just bench-wasm    # wasm benchmark
just bench-native  # native benchmark
```

## Quick start (recommended)

```moonbit
let a : FixedArray[Int] = [1, 2, 3, 4, 5, 6, 7, 8]
let buf = @mizchi/simd/simd_buffer.SimdBuffer::from_array(a)
let total = buf.sum()              // SIMD on wasm / wasm-gc / native, scalar on js
let out = SimdBuffer::make(buf.length())
SimdBuffer::add(buf, buf, out)
let back : FixedArray[Int] = out.to_array()
```

The same code compiles and runs unchanged on every target.

### What you get per target

| target | storage | SIMD path | notes |
|---|---|---|---|
| `wasm` | linear memory | inline-WAT `v128.*` | fastest (3-90× over scalar) |
| `wasm-gc` | linear memory | inline-WAT `v128.*` | parity with `wasm` |
| `native` | `FixedArray` | `@internal` → C FFI (NEON/SSE) where available, else clang auto-vec | parity with the FixedArray-API native fast paths |
| `js` | `FixedArray` | **scalar only** — no SIMD acceleration. See note below |

> **js is scalar.** MoonBit on the js backend has no SIMD escape hatch.
> SimdBuffer compiles and runs on js for **API portability** (same code
> across all four targets), but throughput-critical hot paths on js
> should keep data in native JS typed arrays and call back into wasm
> where SimdBuffer is actually accelerated.

### Buffer types and operations

- `SimdBuffer` (i32): `sum`, `dot`, `add`, `sub`, `mul`, `neg`, `abs`,
  `min_elem`, `max_elem`, `eq`, `lt`, `gt`, `where_`, `saxpy`, `min`,
  `max`, `argmin`, `argmax`, `prod`, `count_nonzero`, `any`, `all`,
  `cumsum`, `cumprod`, `div`, `gather`, `scatter`, `sort`, `sort4`,
  `sort16`, `bitonic_merge8`/`16`/`32`/`64`
- `SimdBufferF32`: `add`, `sub`, `mul`, `div`, `sqrt`, `min_elem`,
  `max_elem`, `sum`, `dot`
- `SimdBufferF64`: `add`, `sub`, `mul`, `div`, `sqrt`, `min_elem`,
  `max_elem`, `sum`, `dot`, `mean`, `variance`, `matmul`, `gemv`,
  `transpose`
- `SimdBufferBytes`: `popcount`, `memcpy`, `memset`, `equal`,
  `find_byte`, `count_byte`, `is_ascii`, `to_lower_ascii`,
  `to_upper_ascii`, `validate_utf8`, `adler32`, `base64_encode` /
  `base64_decode` (+ `_into` in-place variants)
- `SimdBufferRing`: single-arena bump allocator. On wasm/wasm-gc it
  amortises `memory.grow` across sub-allocations (~120 µs → 8 ns per
  alloc). On native/js it's a thin shell — `alloc_*` just calls
  `make` because GC alloc is already cheap.
- Copy bridges: `from_array` / `to_array` / `copy_from_array` /
  `copy_to_array` for `FixedArray` ↔ `SimdBuffer` interop.

### Memory model gotcha (wasm-only)

On `wasm` / `wasm-gc`, `SimdBuffer` storage comes from `memory.grow`
and is **never freed**. Suitable for batch / request-scoped workloads.
Long-running services should use `SimdBufferRing` and `reset()` to
recycle a single grown region across calls.

```moonbit
let ring = SimdBufferRing::make(65536)
for input in inputs {
  ring.reset()
  let out = ring.alloc_bytes((input.length() + 2) / 3 * 4)
  SimdBufferBytes::base64_encode_into(input, out)
  // ... use out, then forget it ...
}
```

On native / js, allocation is GC-managed so `Ring` doesn't matter for
correctness — it's there for cross-target source compatibility.

## Alternative: FixedArray-based API

The original API surface is still available and useful when you want
GC-managed storage with no `memory.grow` lifecycle to manage:

```moonbit
let arr : FixedArray[Int] = [1, 2, 3, 4, 5, 6, 7, 8]
let total = @mizchi/simd.sum_i32(arr)            // wasm SIMD, scalar on others
```

| target | acceleration |
|---|---|
| `wasm` | inline-WAT `v128` SIMD (real SIMD) |
| `wasm-gc` | scalar fallback (GC-ref FFI blocks `v128.load`) |
| `native` | C FFI (NEON/SSE) for ~12 ops, scalar (auto-vec) otherwise |
| `js` | scalar fallback |

**Use FixedArray when**: storage lifetime is managed by GC, you don't
need wasm-gc SIMD, and you only need the ops that exist on the
FixedArray side (no `SimdBufferRing`, no transparent native parity).

**Use SimdBuffer when**: you want one API that compiles everywhere with
SIMD on three of the four targets — and you're OK with `memory.grow`
lifecycle on wasm.

## Speedup highlights

### wasm (V8, Apple Silicon, FixedArray-based API)

| op | size | scalar | SIMD | x |
|---|---|---|---|---|
| `sum_i32` | 1024 | 693 ns | 132 ns | **5.2** |
| `add_i32` | 1024 | 1.54 µs | 132 ns | **11.7** |
| `adler32` | 4096 B | 9.58 µs | 358 ns | **26.8** |
| `memcpy` | 4096 B | 5.44 µs | 86 ns | **63** |
| `memset` | 4096 B | 6.03 µs | 67 ns | **90** |
| `matmul_f64` | 64×64 | 359 µs | 65 µs | **5.5** |
| `base64_encode` | 4096 B | 7.45 µs | 2.06 µs | **3.6** |
| `sort_i32` | 1024 | 157 µs | 16.4 µs | **9.6** |

### wasm-gc (V8, Apple Silicon, SimdBuffer family)

| op | size | scalar fallback | SimdBuffer SIMD | x |
|---|---|---|---|---|
| `sum_i32` | 1024 | 344 ns | 106 ns | **3.2** |
| `add_i32` | 1024 | 413 ns | 118 ns | **3.5** |
| `popcount_bytes` | 4096 | 6.91 µs | 291 ns | **24** |
| `memcpy` | 4096 | 1.17 µs | 100 ns | **12** |
| `memset` | 4096 | 1.00 µs | 59 ns | **17** |
| `adler32` | 4096 | 9.23 µs | 329 ns | **28** |
| `matmul_f64` | 64×64 | (scalar) | 57.9 µs | ~5 |

`SimdBufferRing` brings allocation cost from ~120 µs (single
`memory.grow`) down to ~8 ns (bump pointer reset).

## File structure

```
src/
  simd_wasm_{i32,f64,f32,bytes,sort}.mbt   # FixedArray-API: wasm inline-WAT v128
  simd_native.mbt + simd_native_ffi.mbt    # FixedArray-API: native extern "C"
  simd_native.c                            # NEON / SSE intrinsics
  simd_scalar.mbt                          # FixedArray-API: js + wasm-gc fallback
  internal/scalar.mbt                      # shared scalar reference impls

  base64/                                  # RFC 4648 sub-package
    base64_common.mbt                      # tables + scalar
    base64_wasm.mbt                        # wasm SIMD encode/decode
    base64_scalar.mbt                      # other targets

  simd_buffer/                             # SimdBuffer family — portable API
    simd_buffer.mbt / _f32 / _f64 / _bytes / _sort / _ring / _copy.mbt
      # wasm + wasm-gc: linear-memory storage + inline-WAT v128
    simd_buffer_scalar.mbt
      # native + js: FixedArray storage + @internal / @base64 delegation
    simd_buffer_imports.mbt
      # cross-target import keep-alive
```

See `CLAUDE.md` for the deep dive: per-op bench tables, inline-WAT
gotchas (the `i8x16.sub` stack-order trap, `f64.min` parser hole,
wasm-gc tlsf collision), and the SimdBuffer capability matrix.

## License

MIT
