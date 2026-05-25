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

## Numpy-style ops (B / C / D)

### Element-wise Int + `where`

`simd_{sub,mul,neg,abs,min_elem,max_elem,eq,lt,gt}` (Int) and
`simd_where(mask, a, b, out)`. `eq` / `lt` / `gt` produce numpy-style int
masks (`-1` / `0`); pass that mask to `simd_where` (uses `v128.bitselect`).

| op | size | scalar | simd | x |
|---|---|---|---|---|
| sub | 1024 | 1.72 µs | 147 ns | 11.7 |
| mul (Int) | 1024 | 1.74 µs | 159 ns | 10.9 |
| neg | 1024 | 1.14 µs | 106 ns | 10.8 |
| abs | 1024 | 1.66 µs | 107 ns | 15.5 |
| min_elem / max_elem | 1024 | ~2.2 µs | 131 ns | ~16.7 |
| eq / lt | 1024 | 1.66 µs | 134 ns | 12.4 |
| where | 1024 | 2.10 µs | 213 ns | 9.9 |

### Linear algebra

- `simd_matmul_f64(A, B^T, C, m, k, n)` — B is passed already transposed (row-major B^T), so the inner loop is a plain dot product of two contiguous rows. Inner uses f64x2 via a private `simd_dot_f64_range(a, a_off, b, b_off, k)`.
- `simd_gemv_f64(A, x, y, m, n)` — one f64x2 dot product per output row.
- `simd_transpose_f64` — stays scalar (a SIMD 2x2 block path would need `i8x16.shuffle` with a 16-byte literal pattern; tractable but not a clear win for a memory-bound op, so deferred).

| op | size | scalar | simd | x |
|---|---|---|---|---|
| matmul_f64 | 64×64 | 359 µs | 65 µs | 5.5 |
| gemv_f64 | 256×256 | 86 µs | 16 µs | 5.3 |

### Reductions + scan

- `simd_prod` (Int), `simd_mean_f64`, `simd_var_f64`, `simd_count_nonzero`, `simd_all`, `simd_any` — all SIMD-accelerated.
- `simd_cumsum` / `simd_cumprod` stay scalar: the dependency chain across lanes makes a 4-way SIMD prefix sum require cross-lane shuffles whose 16-byte index pattern would need a precomputed `FixedArray[Byte]`. Deferred.

| op | size | scalar | simd | x |
|---|---|---|---|---|
| prod | 1024 | 751 ns | 171 ns | 4.4 |
| var_f64 | 1024 | 1.37 µs | 625 ns | 2.2 |
| count_nonzero | 1024 | 668 ns | 217 ns | 3.1 |
| all | 1024 | 642 ns | 194 ns | 3.3 |
| any (all non-zero) | 1024 | 18 ns | 204 ns | 0.09 |

`simd_any` loses on this bench because the scalar version returns on the very
first non-zero element. On a buffer of mostly zeros (the worst case for the
scalar path) the SIMD version reads every chunk at constant cost; pick the
implementation based on expected input shape.

## Extensions (argmin/argmax SIMD, transpose, f32, sort)

### argmin / argmax (Int)

Two-pass SIMD: run `simd_min_v128` / `simd_max_v128` first, then use a private `simd_find_int_v128` (i32x4.eq + bitmask + ctz with `select`-based found tracking, same shape as `simd_find_byte`).

| op | size | scalar | simd | x |
|---|---|---|---|---|
| argmin | 1024 | 761 ns | 530 ns | 1.4 |
| argmax | 1024 | 1.25 µs | 504 ns | 2.5 |

### simd_transpose_f64

2x2 f64 block via two `v128.load` + two `i8x16.shuffle` immediates + two `v128.store`. Falls back to scalar when either dimension is odd.

| op | size | scalar | simd | x |
|---|---|---|---|---|
| transpose_f64 | 128×128 | 27.3 µs | 8.71 µs | 3.1 |

### f32 over byte-buffer

`FixedArray[Float]` is rejected at the FFI boundary ("Invalid stub type"), and so is `FixedArray[UInt]`. The workable shape is `FixedArray[Byte]` with 4 bytes per element (little-endian IEEE-754) and `Float`-aware helpers (`f32_buf_alloc` / `f32_buf_get` / `f32_buf_set`). The SIMD path is `f32x4` (4-way).

Ops implemented: `simd_{add,mul,sum,dot}_f32`.

| op | size | scalar | simd | x |
|---|---|---|---|---|
| add | 1024 | 11.06 µs | 139 ns | 79.7 |
| sum | 1024 | 3.73 µs | 146 ns | 25.5 |
| dot | 1024 | 7.24 µs | 169 ns | 42.8 |

The huge ratio reflects that the scalar path pays per-element `f32_buf_get` / `f32_buf_set` (UInt bit-decode), while the SIMD path goes straight from `v128.load` to `f32x4.*` and back. Treat it as "cost of staying in byte-buffer representation," not as a pure ALU benchmark.

### simd_sort

`simd_sort4_int(arr, off)` sorts a 4-element window in place using a 3-stage sorting network: `i8x16.shuffle` + `i32x4.min_s` / `i32x4.max_s` + blend `i8x16.shuffle`. `simd_sort_int(arr)` defers to the built-in `FixedArray::sort` — a full SIMD merge sort would build on `simd_sort4_int` as the leaf kernel.

| op | size | scalar | simd | x |
|---|---|---|---|---|
| sort4_int x 256 | 1024 | 4.22 µs | 2.49 µs | 1.7 |

`v128.const` is still parser-blocked, but **`i8x16.shuffle` immediates parse fine** — that opens the door to lane-permutation tricks (sort networks, transpose, prefix sum) without needing a precomputed `FixedArray[Byte]` table.

## Prefix scan, integer div, gather/scatter, sort16

### `simd_cumsum` / `simd_cumprod`

Hillis-Steele prefix scan on i32x4: two `i8x16.shuffle` + `i32x4.add` (or `mul`) stages within a chunk, then a `i32x4.splat(running)` step to fold in the previous chunk's tail. cumprod uses lane-fill 1 (multiplicative identity) and running = 1.

| op | size | scalar | simd | x |
|---|---|---|---|---|
| cumsum | 1024 | 1.36 µs | 854 ns | 1.6 |
| cumprod | 1024 | 1.34 µs | 934 ns | 1.4 |

The gains are modest because the cross-chunk dependency (`running` updated each iteration) serializes the loop.

### `simd_div` (Int via f64x2)

wasm SIMD has no `i32x4.div_s` / `i32x4.div_u`. Workaround: widen each i32x4 to two f64x2s (lanes 0,1 via `f64x2.convert_low_i32x4_s`, lanes 2,3 via `i8x16.shuffle` swap + the same convert), do `f64x2.div`, narrow back with `i32x4.trunc_sat_f64x2_s_zero`, and recombine with one more `i8x16.shuffle`. f64 mantissa (53 bits) > i32 width (31 bits) so the round-trip is exact for representable quotients; trap semantics on `b == 0` change from `i32.div_s`'s trap to NaN-trunc-to-zero, which is the caller's responsibility.

| op | size | scalar | simd | x |
|---|---|---|---|---|
| div_int | 1024 | 2.88 µs | 458 ns | 6.3 |

### `simd_gather` / `simd_scatter`

`simd_gather` builds each output v128 with `i32.const 0 i32x4.splat` + four `i32x4.replace_lane` (each taking a scalar `i32.load` from `arr + indices[i+k] * 4`), then writes once with `v128.store`. wasm has no real gather instruction so the read side stays scalar; the win comes from fusing four stores into one.

`simd_scatter` falls through to scalar — random writes can't be vectorized without conflict detection.

| op | size | scalar | simd | x |
|---|---|---|---|---|
| gather | 1024 | 2.09 µs | 406 ns | 5.1 |
| scatter | 1024 | 2.06 µs | 2.02 µs | 1.02 |

### `simd_sort16_int`

Sort 16 elements by running `simd_sort4_int` on each of the four 4-element sub-blocks, then a scalar 3-stage merge (4+4 → 8, 4+4 → 8, 8+8 → 16) into a 16-element scratch buffer. Only the leaf sort is vectorized; merging stays scalar (bitonic SIMD merge is the next step but isn't here yet).

| op | size | scalar | simd | x |
|---|---|---|---|---|
| sort16_int × 64 | 1024 | 9.14 µs | 8.56 µs | 1.07 |

The ratio is close to 1 because the merge stage dominates and runs scalar. A future SIMD bitonic merge would bring this closer to the leaf 1.7x.

## General-purpose `sort_i32` (SIMD-leaf merge sort)

`sort_i32(arr)` is now a bottom-up merge sort with a SIMD leaf:

1. Sort every aligned 16-block with `sort16_i32` (fully SIMD).
2. Tail block (< 16 elements) falls through to the built-in scalar sort.
3. Merge ladder (16+16 → 32, 32+32 → 64, ...) is scalar `merge2_int`
   between ping-pong buffers; a future SIMD bitonic merge would replace
   this and shrink the merge cost.

| op | size | scalar (FixedArray::sort) | simd | x |
|---|---|---|---|---|
| sort_i32 | 1024 | 148.93 µs | 23.38 µs | **6.4** |

The leaf phase dominates: 64 calls to a fully-SIMD `sort16_i32` is much
cheaper than the equivalent scalar comparisons, even with the scalar
ladder on top.

## Full SIMD sort16 (no scalar merge left)

Added `simd_bitonic_merge16_int(arr, off)`: 8+8 → 16 SIMD bitonic merge in 4 stages — reverse B halves and lane-wise min/max against A halves (distance 8), distance-4 min/max between v128 pairs in each 8-half, then distance-2 and distance-1 compare-exchange within each of the 4 v128s. `simd_sort16_int` is now fully SIMD: 4 leaf `sort4` + 2 `bitonic_merge8` + 1 `bitonic_merge16`.

| op | size | scalar | simd | x |
|---|---|---|---|---|
| sort16_int × 64 | 1024 | 11.08 µs | 4.25 µs | **2.6** |

Up from 1.22x with the prior partial-scalar variant.

## SIMD-accelerated UTF-8 validation

`simd_validate_utf8(data)` runs an ASCII fast path first
(`simd_first_non_ascii_chunk_v128`, select-based tracking of the first
16-byte chunk where `i8x16.bitmask` is non-zero), then hands off to a
scalar state machine (`scalar_validate_utf8_from`) starting at that
chunk boundary. The handoff is safe because every chunk before it is
pure ASCII, so no multi-byte sequence can be in flight.

The scalar pass does *structural* validation only (start byte → expected
number of `10xxxxxx` continuations, leading byte ranges 0xC2..0xF4). It
does **not** reject overlong forms, surrogate pairs, or codepoints above
U+10FFFF — those are a stricter pass that would need shuffle-table
classifications (simdjson style).

| input | size | scalar | simd | x |
|---|---|---|---|---|
| all-ASCII | 4096 B | 3.41 µs | 334 ns | **10.2** |
| mixed (10% 2-byte) | 4000 B | 3.52 µs | 3.78 µs | 0.93 |

The mixed case loses because the very first 16-byte chunk contains a
non-ASCII byte, so the SIMD pre-scan finds no skip-able chunks and the
fast path is pure overhead. The pattern wins when the input is mostly
ASCII; for non-ASCII-heavy text, skip the wrapper and call
`scalar_validate_utf8` directly.

## SIMD bitonic merge + sort16 v2

`simd_bitonic_merge8_int(arr, off)` merges two sorted 4-element slices in
place using Batcher's bitonic merge: reverse B with `i8x16.shuffle`, take
element-wise min/max against A, then two stages of distance-2 / distance-1
compare-exchange within each half (shuffle + `i32x4.min_s`/`max_s` + blend
shuffle). `simd_sort16_int` now uses two SIMD merges plus one scalar 8+8
merge.

| op | size | scalar | simd | x |
|---|---|---|---|---|
| sort16_int × 64 | 1024 | 11.99 µs | 9.79 µs | 1.22 |

Up from 1.07x in the previous pass. A future 16-element SIMD bitonic merge
would replace the remaining scalar `merge2` and probably push this past 2x.

## String / byte ops

ASCII-focused; no UTF-8 validation yet (that needs the simdjson-style
shuffle-table approach).

- `simd_is_ascii(data) -> Bool` — fold each chunk's `i8x16.bitmask` via
  `i32.or`, then check the result is zero. Tail byte uses `& 0x80`.
- `simd_to_lower_ascii(data)` / `simd_to_upper_ascii(data)` — per-byte range
  mask: `delta = (v >= 'A') & (v <= 'Z') & 0x20`, then `v += delta` (or
  `-=` for upper). `i8x16.ge_s` / `le_s` / `v128.and` / `i8x16.add` / `sub`
  all parse fine.

| op | size | scalar | simd | x |
|---|---|---|---|---|
| is_ascii | 4096 B | 3.13 µs | 162 ns | 19.3 |
| to_lower_ascii | 4096 B | 11.27 µs | 5.90 µs | 1.9 |

The `to_lower` bench includes a per-iteration copy back to a scratch buffer
(both paths pay the same cost), which compresses the visible SIMD ratio; the
pure transform inside is closer to a 16x speedup.

## More parser surface that works

Added during these passes:

- `f64x2.convert_low_i32x4_s` / `i32x4.trunc_sat_f64x2_s_zero` (lane-width conversion)
- `i32x4.replace_lane <lane>` (immediate lane index)
- `i32x4.{neg, abs, eq, ne, lt_s, gt_s}`, `i32x4.bitmask`
- `v128.bitselect`
- `f32x4.{add, mul, extract_lane, splat}`, `f32.{add, mul, sub, div, sqrt, load, store, convert_i32_s}` — but **not** `f32.min` / `f32.max` (likely same hole as `f64.min` / `f64.max`)

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
