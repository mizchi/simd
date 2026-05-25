# mizchi/simd - MoonBit SIMD Abstraction

## Architecture

Target-specific SIMD implementations with scalar fallback.

- `src/scalar.mbt` - Scalar implementations (all targets)
- `src/simd_wasm.mbt` - wasm: inline WAT with v128 (i32x4 / i8x16 / i16x8)
- `src/simd_wasm_gc.mbt` - wasm-gc: scalar fallback (FixedArray is a GC array, v128.load only works on linear memory)
- `src/simd_native.mbt` - native: extern "C" FFI
- `src/simd_native.c` - C SIMD intrinsics (NEON/SSE)
- `src/simd_js.mbt` - js: scalar fallback

## Key Findings

- MoonBit's inline WAT parser **does** accept v128 instructions on the `wasm` target. Verified working: `i32x4.{splat,add,mul,extract_lane}`, `v128.load`, `v128.store`, `i16x8.extadd_pairwise_i8x16_u`, `i32x4.extadd_pairwise_i16x8_u`, `i16x8.extmul_{low,high}_i8x16_u`, `i64.extend_i32_u`. Use `block ... loop ... br_if 1 ... br 0 ... end end` for loops and indexed locals.
- The one parser gap: `v128.const i32x4 ...` literal trips the Dwarfsm parser (`Int32.of_string` failure). Build the constant via `i32.const N i32x4.splat` (or load from a precomputed `FixedArray[Byte]`).
- `wasm` FixedArray ABI: the FFI parameter pointer = address of element[0]; elements are tightly packed i32 / i8 in linear memory. Object header sits at negative offsets (`[-12]` size, `[-8]` refcount, `[-4]` type tag).
- `wasm-gc` FixedArray is passed as a GC `(ref N)`, not a linear-memory pointer, so `v128.load` is unusable. SIMD lift on wasm-gc would require a copy-to-buffer hop that erases the win. Keep scalar fallback.
- TCC (default native compiler) doesn't support NEON/SSE intrinsics, but C FFI still provides significant speedup via optimized scalar code.
- `native-stub` in moon.pkg.json links C source files for native target.
- `#borrow(param)` annotation needed for FixedArray/Bytes FFI parameters.

## wasm SIMD speedup

Bench on Apple Silicon under V8/wasm runtime, sizes as noted.

| op | size | scalar | simd (v128) | x |
|---|---|---|---|---|
| sum (Int) | 1024 | 693 ns | 132 ns | 5.2 |
| min (Int) | 1024 | 685 ns | 121 ns | 5.7 |
| max (Int) | 1024 | 1.20 µs | 239 ns | 5.0 |
| dot_product (Int) | 1024 | 1.19 µs | 182 ns | 6.6 |
| add (Int) | 1024 | 1.54 µs | 132 ns | 11.7 |
| saxpy: `k*a+b` (Int) | 1024 | 1.82 µs | 131 ns | 13.9 |
| adler32 | 4096 B | 9.58 µs | 358 ns | 26.8 |
| popcount_bytes | 4096 B | 7.78 µs | 414 ns | 18.8 |
| memcpy_bytes | 4096 B | 5.44 µs | 86 ns | 63.2 |
| memset_bytes | 4096 B | 6.03 µs | 67 ns | 90.0 |
| equal_bytes | 4096 B | 5.11 µs | 208 ns | 24.6 |
| find_byte (needle at tail) | 4096 B | 2.63 µs | 253 ns | 10.4 |
| count_byte | 4096 B | 2.62 µs | 177 ns | 14.8 |

`argmin` / `argmax` stay on the scalar path: early-exit needs `if` / `return`
opcodes that the Dwarfsm inline-WAT parser handles less reliably.

## Double (f64) ops via f64x2

Element-wise: `simd_{add,sub,mul,div,sqrt,min,max}_f64`. Reductions: `simd_{sum,dot}_f64`. All 2-way parallel.

| op | size | scalar | simd (f64x2) | x |
|---|---|---|---|---|
| add | 1024 | 1.71 µs | 278 ns | 6.2 |
| mul | 1024 | 1.94 µs | 256 ns | 7.6 |
| div | 1024 | 1.84 µs | 264 ns | 7.0 |
| sqrt | 1024 | 1.24 µs | 252 ns | 4.9 |
| sum | 1024 | 681 ns | 292 ns | 2.3 |
| dot | 1024 | 1.25 µs | 336 ns | 3.7 |

Notes specific to f64:

- `FixedArray[Float]` (f32) cannot be passed across the FFI today — the compiler rejects it as "Invalid stub type". f32 ops are deferred; f64 is the practical numpy dtype anyway.
- `f64.min` / `f64.max` are rejected by the Dwarfsm parser, and `f64.const` literals likely trip the same `Int32.of_string` path as `v128.const`. Workarounds: use `select` with `f64.lt` / `f64.gt` for the scalar tail, and synthesize `0.0` via `i32.const 0 f64.convert_i32_s` (then `f64x2.splat` for accumulators). The vector variants `f64x2.min` / `f64x2.max` parse fine.

## Inline-WAT gotchas worth remembering

- `i32.and` is bitwise, **not** logical: combining a `0/1` boolean with a
  multi-bit value (e.g. a `i8x16.bitmask` result like `0b100000 = 32`) silently
  yields `0` when bits don't overlap. Use `i32.mul` or convert both sides to
  `0/1` via `i32.eqz i32.eqz` before AND-ing as a `select` condition.
- Indexed locals: params count as `local 0..N-1`, then `(local ...)` start at
  `N`. Off-by-one (`local.get N+k` against `N+k-1` declared) compiles fine but
  fails at instantiation with `invalid local index`.
- Wasm has no `i32.min_s/max_s`; reduce horizontally with `select` over
  extracted lanes. `f64.min` / `f64.max` exist in core wasm but the Dwarfsm
  inline parser rejects them — use `select` + `f64.lt` / `f64.gt` instead.
- Float / Double constants: `f64.const X` and `v128.const ... X ...` literals
  trip the parser's int-decoder. Synthesize via `i32.const N f64.convert_i32_s`
  (and `f64x2.splat` for vectors); load shuffle / lookup tables from a
  `FixedArray[Byte]` via `v128.load`.

## Commands

```bash
just test          # All targets
just test-wasm     # wasm only
just test-native   # native only
just bench-native  # Benchmark on native
just bench-wasm    # Benchmark on wasm
```
