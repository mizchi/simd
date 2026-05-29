# mizchi/simd - MoonBit SIMD Abstraction

## Architecture

Target-specific SIMD implementations with scalar fallback.

- `src/internal/scalar.mbt` - Scalar implementations (shared across all targets via `@internal.scalar_*`)
- wasm (target `wasm`) — inline WAT with v128 (i32x4 / i8x16 / i16x8 / f64x2 / f32x4). Split by category:
  - `src/simd_wasm_i32.mbt` - i32 reductions / element-wise / comparisons / cum / gather / scatter
  - `src/simd_wasm_f64.mbt` - f64 element-wise + reductions + linear algebra (matmul / gemv / transpose)
  - `src/simd_wasm_f32.mbt` - f32 ops over byte-buffer representation (f32x4)
  - `src/simd_wasm_bytes.mbt` - byte / string / UTF-8 / Adler-32
  - `src/simd_wasm_sort.mbt` - sort networks + bitonic merges + general-purpose sort
- native (target `native`):
  - `src/simd_native.mbt` - public wrappers
  - `src/simd_native_ffi.mbt` - `extern "C"` FFI declarations
  - `src/simd_native.c` - C SIMD intrinsics (NEON/SSE)
- `src/simd_scalar.mbt` - shared scalar fallback for `js` and `wasm-gc` targets (FixedArray-on-GC-heap means `v128.load` is unusable on wasm-gc)
- `src/base64/` - sub-package providing RFC 4648 Base64 encode / decode. wasm uses inline-WAT SIMD (12-in / 16-out encode, 16-in / 12-out decode); other targets use shared scalar in `base64_common.mbt`.
- `src/simdjson/` - sub-package porting simdjson's `find_structural_bits` byte-classification pipeline to wasm SIMD. Operates on `@simd_buffer.SimdBufferBytes`. wasm / wasm-gc use inline-WAT (`i8x16.eq` + `i8x16.bitmask`); native / js use FixedArray scalar. See the "src/simdjson/ sub-package" section below.
- `src/simd_buffer/` - sub-package providing `SimdBuffer` (i32) / `SimdBufferF32` / `SimdBufferF64` / `SimdBufferBytes` + `SimdBufferRing` arena allocator. **Same public API on all four targets** (`wasm` / `wasm-gc` / `native` / `js`); this is the recommended portable surface. wasm / wasm-gc use linear-memory `v128.load` (inline-WAT). native / js share a `FixedArray` + `@internal` / `@base64` delegation impl — on native that bottoms out in C FFI (NEON / SSE) where available; **on js it's scalar only, no SIMD acceleration** (the API portability still wins, but performance does not). See the "SimdBuffer: portable SIMD across all four targets" section below.

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

## General-purpose `sort_i32` (SIMD leaf + SIMD merge32/64 + scalar tail)

`sort_i32(arr)` is a bottom-up merge sort with three SIMD phases:

1. **Leaf**: sort every aligned 16-block with `sort16_i32` (fully SIMD).
   Tail < 16 falls through to the built-in scalar sort.
2. **width = 16 pass**: in-place SIMD `bitonic_merge32_i32` for every
   aligned 32-block. Tail (16-element block + partial < 16) goes through
   scalar `merge2_int` once into `tmp` and back.
3. **width = 32 pass**: in-place SIMD `bitonic_merge64_i32` for every
   aligned 64-block. Same tail treatment as the previous pass.
4. **width ≥ 64**: scalar `merge2_int` ladder with ping-pong between
   `arr` and `tmp`. A future SIMD `bitonic_merge128` / `_256` would
   shrink this further but the working set hits 32+ v128 registers and
   the win shrinks.

| variant | size | scalar (`FixedArray::sort`) | simd | x |
|---|---|---|---|---|
| leaf only (old) | 1024 | 148.93 µs | 23.38 µs | 6.4 |
| + merge32 in ladder | 1024 | 158.02 µs | 21.08 µs | 7.5 |
| **+ merge64 in ladder** | 1024 | 157.41 µs | **16.38 µs** | **9.6** |

Each additional SIMD merge level shaves another 15-25 % off because it
replaces N scalar `merge2` calls (~O(2N) ops each) with one inline-WAT
function call doing the same work register-resident.

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

## `src/simdjson/` sub-package — JSON structural indexer

`src/simdjson/` is a separate sub-package (mirrors `src/base64/`) that
ports the SIMD core of simdjson — `find_structural_bits` — to wasm and
its scalar fallback to native / js. Layout:

```
src/simdjson/
  moon.pkg                  # imports @simd_buffer + @bench
  simdjson_wasm.mbt         # targets [wasm, wasm-gc] — inline-WAT v128
  simdjson_scalar.mbt       # targets [native, js] — FixedArray scalar
  simdjson_test.mbt         # all targets
  simdjson_bench.mbt        # all targets
```

The package operates on `@simd_buffer.SimdBufferBytes` (input) and
`@simd_buffer.SimdBuffer` (i32 bitmap / indices output), accessed
cross-package via `raw_addr()` on wasm / wasm-gc and `backing()` on
native / js (newly added accessors on the SimdBuffer family).

Public surface (identical on all four targets):

1. `classify_structural(input, out)` — bit `i` of `out` set iff byte
   `i` ∈ `{ } [ ] , :`. 16 bytes/iter via 6 × `i8x16.eq` + chained
   `v128.or` + `i8x16.bitmask`, OR-in semantics.
2. `classify_numeric(input, out)` — bit `i` set iff byte `i` ∈ `0-9 -
   + . e E`. One unsigned range check (digits) + 5 × `i8x16.eq`, same
   bitmask shape.
3. `classify_quote_raw(input, out)` — bit `i` set iff byte `i` == `"`
   (raw, no escape resolution). Single-needle `i8x16.eq` SIMD.
4. `classify_backslash(input, out)` — same shape for `\`.
5. `compute_quote_mask(input, out)` — bit `i` set iff byte `i` lies
   strictly inside a `"..."` string literal, exclusive of the
   delimiting `"`s. Honours `\"` and `\\` escapes. **Stays scalar** —
   wasm SIMD has no CLMUL, so the prefix-XOR over quote positions runs
   as a branchless `select`-driven bit-walk in inline-WAT.
6. `extract_structural_indices(structural, quote_mask, n, indices)`
   → `Int` — walks `structural & ~quote_mask` word-by-word, emits the
   ascending byte offsets via `i32.ctz` + `effective &= effective - 1`.

Plus a one-shot convenience wrapper
`find_structural_indices_with_scratch(input, struct_scratch,
quote_scratch, indices)` that chains the pipeline, and a
scratch-allocating `find_structural_indices(input, indices)` for
one-off calls (allocates two `SimdBuffer`s per call via `memory.grow`
— use the scratch variant in hot loops or pair with `SimdBufferRing`).

Usage (any target):

```moonbit
let input = @simd_buffer.SimdBufferBytes::from_array(json_bytes)
let words = (input.length() + 31) / 32
let structural = @simd_buffer.SimdBuffer::make(words)
let quote_mask = @simd_buffer.SimdBuffer::make(words)
let indices = @simd_buffer.SimdBuffer::make(input.length())
let count = @simdjson.find_structural_indices_with_scratch(
  input, structural, quote_mask, indices,
)
```

### Bench (V8, 4096-byte JSON, M-class Apple Silicon)

| op | wasm | wasm-gc | native scalar | wasm vs native |
|---|---|---|---|---|
| `classify_structural` | 428 ns | 383 ns | 2.94 µs | **6.9x** |
| `classify_numeric` | 462 ns | 400 ns | 2.60 µs | **5.6x** |
| `classify_quote_raw` | 309 ns | 261 ns | 2.04 µs | **6.6x** |
| `compute_quote_mask` | 7.16 µs | 7.24 µs | 3.28 µs | **0.46x** (loss) |
| `extract_structural_indices` | 434 ns | 432 ns | 4.79 µs | **11x** |
| `find_structural_indices_with_scratch` | 8.10 µs | 8.33 µs | 10.25 µs | **1.27x** |

The byte-classification phases (`classify_*`) are 5–7× on wasm thanks to
straightforward `i8x16.eq` lane-parallelism. `extract_structural_indices`
is 11× because the scalar baseline does manual `ctz` while the inline-WAT
gets a single `i32.ctz` instruction.

But the **pipeline as a whole is only 1.27×**, because
`compute_quote_mask` dominates and *loses* to scalar. That's the
simdjson lesson: without CLMUL there's no way to vectorise the prefix
XOR over quote positions, so the inline-WAT bit-walk pays the per-byte
inline-WAT-call overhead while the native scalar runs in a single tight
loop the compiler can auto-vectorise where possible.

### Why the full pipeline is bandwidth-bound by `compute_quote_mask`

simdjson's x86 path uses `_mm_clmulepi64_si128` (CLMUL) to prefix-XOR a
64-bit quote bitmap in one instruction. Wasm SIMD has no equivalent —
`v128` has no `clmul` family. So the quote-tracking state machine runs
scalar: one byte read + 6 `select`s + one i32 store per input byte. At
4096 bytes that's ~24,000 inline-WAT ops, dominating the 6 × 16-byte
SIMD passes (~256 ops each for classify_structural / classify_numeric /
classify_quote_raw).

Until wasm grows a CLMUL extension, the bit-walk is the floor. The
educational takeaway lines up with the cosine-similarity example: SIMD
primitives help where the algorithm is SIMD-shaped; algorithms with
serial bit-level dependencies don't get a free lunch from packed lanes.

### Inline-WAT gotcha caught here

The first pass had `find_structural_indices` allocate its two
scratch `SimdBuffer`s with `SimdBuffer::make` per call. That's two
`memory.grow` calls per invocation — ~400 µs each in the bench harness,
making the wrapper 100× slower than the sum of its parts. The fix is
the `_with_scratch` variant (same pattern as `base64_encode_into`):
caller pre-allocates the bitmaps and reuses them. Same trade-off as
the `SimdBufferRing` pattern.

## 0.3.0: byte arithmetic + image / pixel ops

Added to `SimdBufferBytes` (Tier 1, SIMD on wasm + wasm-gc) and
`SimdBuffer` (i32) — plus a Tier-2 image batch with API + scalar impl
landed in 0.3.0; their SIMD inline-WAT bodies are queued for 0.4.0.

### Tier 1 — byte-wise binary arithmetic (SIMD shipped)

| op | wasm SIMD instruction | use case |
|---|---|---|
| `SimdBufferBytes::byte_add(a, b, out)` | `i8x16.add` (mod 256) | PNG decode Up reconstruction, audio mix |
| `SimdBufferBytes::byte_sub(a, b, out)` | `i8x16.sub` | PNG encode Up filter, pixel diff |
| `SimdBufferBytes::byte_avg(a, b, out)` | `i8x16.avgr_u` (rounding +1) | midpoint color, blur kernel |
| `SimdBufferBytes::sat_add(a, b, out)` | `i8x16.add_sat_u` | font AA coverage accumulation |
| `SimdBufferBytes::sat_sub(a, b, out)` | `i8x16.sub_sat_u` | clamp-sub, audio limiting |
| `SimdBufferBytes::clamp(lo, hi, out)` | `i8x16.max_u` + `i8x16.min_u` | image normalization, audio limit |
| `SimdBufferBytes::byte_sub_offset(src, stride, out)` | shifted `v128.load` + `i8x16.sub` | PNG Sub filter (`buf[i] = src[i] - src[i - stride]`) |
| `SimdBuffer::array_equal(a, b, len) -> Bool` | `i32x4.eq` + `i8x16.bitmask` fold | pixelmatch row prefilter |

All six byte ops share the same 16-byte chunk inline-WAT skeleton:
`v128.load a → v128.load b → OP → v128.store`, plus a per-byte scalar
tail. The `clamp` variant pre-splats `lo` / `hi` once into v128 locals
outside the loop.

`byte_sub_offset` is the only one with non-trivial structure: scalar
head for `i < stride`, then SIMD body that loads `data + i` and
`data + i - stride` (16 bytes each) and subtracts. Used by image-mbt's
PNG Sub encoder, which is the pattern that motivated bundling it
in-simd rather than re-porting per consumer.

### Tier 2 — image / pixel ops (API + scalar; SIMD in 0.4.0)

`SimdBufferBytes` methods:

- `rgb_to_rgba(src, alpha, out)` — 3-byte/pixel → 4-byte/pixel with
  broadcast alpha. SIMD shape: `i8x16.shuffle` from 12-byte input to
  16-byte output with α slots; not yet shipped.
- `rgba_to_grayscale(src, out)` — Rec. 601 fixed-point
  `Y = (77*R + 150*G + 29*B) >> 8` (weights sum to 256). SIMD path
  requires `i16x8.extmul` for the weighted sum across 4 pixels.
- `channel_extract(src, ch, out)` / `channel_merge(r, g, b, a, out)` —
  interleaved RGBA ↔ planar. SIMD shape: `i8x16.shuffle` for the
  de-interleave and 4-way `i8x16.shuffle` interleave.
- `lerp(a, b, t, out)` — `out[i] = (a[i]*(256-t) + b[i]*t) >> 8`, `t ∈
  0..=256`. SIMD shape: `i16x8.mul` + add, requires lane width split.
- `histogram(self, bins[256])` — 256-bin scalar (wasm SIMD has no
  scatter; this stays scalar).
- `alpha_blend_solid(dst, sr, sg, sb, sa)` — premultiplied
  source-over, in-place over an RGBA8 destination. Scalar formula
  `out = src + dst * (255 - sa) / 255` with saturation. SIMD path
  requires fixed-point i16x8 (the canvas blend rewrite from this
  session's audit — deferred to 0.4.0).

### Why image ops aren't SIMD-shipped in 0.3.0

The byte binary ops are mechanical 16-byte-chunk-with-different-SIMD-op
skeletons. The image ops each need their own inline-WAT structure
(shuffle patterns, lane-width conversions, fixed-point rescaling). The
API surface is shipped in 0.3.0 so consumer repos can call them with
forward-compatible signatures; the 0.4.0 release fills in the wasm SIMD
bodies behind the same API.

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
- `select` push order is **val_if_true (deeper), val_if_false (shallower),
  cond (top)**. The wasm spec phrases it `[v2 v1 c] → v2 if c≠0 else v1`,
  so v2 (val_if_true) is pushed FIRST. Get this wrong and the branch
  inverts: `i8x16.sub_sat_u`'s scalar tail in 0.3.0 returned the negative
  underflow value instead of 0 because the original pushed `0` then
  `(a-b)`, making `(a-b)` end up in the val_if_true slot.
- `local.tee N` leaves the value on stack AND copies it into local `N`.
  Subsequent stack manipulation must account for that residual value
  — easy to miss when reading the WAT as if it were SSA. The clamp /
  sat_sub 0.3.0 bugs came from mixing tee with later pushes and finding
  the wrong slot consumed by `select`. Safer pattern when ambiguous:
  `local.set N` (clears stack), then `local.get N` exactly where each
  value is needed.
- MoonBit `Array[T]` has no `resize_uninit(n)` to grow without
  initialising the new range. "Pre-extend with N zero-pushes, then
  index-write the real values" doubles the push count and is **slower**
  than the original `output.push(value)` per element. Confirmed empirically
  in the inflate LZ77 refactor experiment (4.02 → 4.57 ms, 14 % regression).
  Real SIMD memcpy into the inflate output needs a custom growable byte
  buffer with manual capacity tracking; documented for a future PR.

## Base64 (RFC 4648) sub-package

Demonstrator built on the same wasm SIMD inline-WAT discipline. Lives in
`src/base64/` and exposes `encode(input) -> FixedArray[Byte]` and
`decode(input) -> FixedArray[Byte]?`. Standard alphabet only, with `=`
padding.

### Algorithm

- **Encode** processes 12 input bytes → 16 output bytes per chunk:
  1. `i8x16.shuffle [1,0,2,1, …]` lays out per 4-byte lane as `[b1,b0,b2,b1]`.
  2. Per-lane bit extraction via 4 `i16x8.shr_u`/`shl` + per-byte mask + OR
     produces 4 × 6-bit indices.
  3. ASCII offset stage is arithmetic, not a lookup: 5 chained
     `i8x16.ge_s` + `v128.and` + `i8x16.add`/`sub` apply the right delta per
     range (`A-Z`/`a-z`/`0-9`/`+`/`/`).
- **Decode** processes 16 ASCII bytes → 12 output bytes per chunk:
  1. Range masks (`i8x16.ge_s`/`le_s`/`eq`) classify each byte into one of
     five Base64 ranges; their `v128.or` is the validity bitmask.
  2. Decoded 6-bit values come from `b + (mask & offset)` summed over the
     five ranges (mutually-exclusive masks → at most one offset adds).
  3. Pack 4 × 6 → 24 bits per lane via 6 `i32x4.shl`/`shr_u` + byte-position
     masks (`0xFC` / `0x03` / `0xF0` / `0x0F` / `0xC0` / `0x3F`); a final
     `i8x16.shuffle` compacts the 12 valid bytes into the low 12 lanes for
     `v128.store`.
- Last 4-char input quad (where `=` padding may appear) always falls through
  to scalar — keeps the SIMD path branch-free.

### Speedup on wasm (V8, 4096-byte input, M-class Apple Silicon)

| op | size | scalar | simd | x |
|---|---|---|---|---|
| encode | 4096 B | 7.45 µs | 2.06 µs | **3.6** |
| decode | 5460 B | 7.97 µs | 2.28 µs | **3.5** |

### Inline-WAT gotcha caught here

`i8x16.sub` (and by extension every `iNxM.sub`) is **non-commutative** and
pops in the standard wasm stack order: stack `[..., a, b]` (with `b` on top)
produces `a - b`. So to compute `acc -= delta`, push `acc` FIRST and `delta`
SECOND, not the other way around. First pass of the encode WAT had them
reversed and produced garbage only for byte values ≥ 128 — the all-ASCII
test cases passed coincidentally because the subtraction happened to wrap
through the same modulus.

## SimdBuffer: portable SIMD across all four targets

`src/simd_buffer/` is the recommended portable API. The public surface —
`SimdBuffer` / `SimdBufferF32` / `SimdBufferF64` / `SimdBufferBytes` /
`SimdBufferRing` and ~50 ops — is **identical on `wasm`, `wasm-gc`,
`native`, and `js`**. Internally:

- **`wasm` / `wasm-gc`**: backing store is raw linear memory allocated
  via `memory.grow`; ops are inline-WAT `v128.*` (the wins listed
  below). This is the only path that gets SIMD on `wasm-gc` — the
  FixedArray-based API falls back to scalar there because GC-ref FFI
  blocks `v128.load`.
- **`native`**: backing store is `FixedArray[T]`; ops delegate to
  `@internal.scalar_*` / `@base64.*`, which themselves dispatch to the
  existing native C FFI fast paths (NEON / SSE) where one exists. No
  new C code; the native target re-uses the FixedArray-API stack
  underneath. Practical perf: parity with the FixedArray-based native
  API.
- **`js`**: backing store is `FixedArray[T]`, ops are scalar via
  `@internal`. **No SIMD acceleration on js** — MoonBit's js backend
  has no SIMD escape hatch. SimdBuffer is offered on js purely so
  caller code stays portable; throughput-critical js workloads should
  keep data in JS-native typed arrays and call into wasm.

```
src/simd_buffer/
  # wasm + wasm-gc (linear memory + inline-WAT v128)
  simd_buffer.mbt        # SimdBuffer (i32) + page-per-buffer allocator + all
                         #   i32 ops (sum / dot / add / sub / mul / neg / abs /
                         #   min_elem / max_elem / eq / lt / gt / where_ / saxpy /
                         #   min / max / argmin / argmax / prod / count_nonzero /
                         #   any / all / cumsum / cumprod / div / gather / scatter)
  simd_buffer_f32.mbt    # SimdBufferF32 + add / sub / mul / div / sqrt /
                         #   min_elem / max_elem / sum / dot
  simd_buffer_f64.mbt    # SimdBufferF64 + add / sub / mul / div / sqrt /
                         #   min_elem / max_elem / sum / dot / mean / variance /
                         #   matmul / gemv / transpose
  simd_buffer_bytes.mbt  # SimdBufferBytes + popcount / memcpy / memset / equal /
                         #   find_byte / count_byte / is_ascii / to_lower /
                         #   to_upper / validate_utf8 / adler32 / base64
  simd_buffer_sort.mbt   # SimdBuffer::sort4 / sort16 / bitonic_merge8/16/32/64 /
                         #   sort (full merge sort with caller-supplied scratch)
  simd_buffer_ring.mbt   # SimdBufferRing — single-arena bump allocator over a
                         #   pre-grown page (cheap reset, no per-buffer memory.grow)
  simd_buffer_copy.mbt   # FixedArray ↔ SimdBuffer bridges

  # native + js (FixedArray wrappers; same public API, delegates to @internal / @base64)
  simd_buffer_scalar.mbt # All of the above types + methods, scalar / C-FFI backed.
                         # native: bottoms out in NEON / SSE / clang auto-vec.
                         # js: scalar only, no SIMD escape hatch.

  # cross-target stub: keeps @base64 / @internal imports "used" on wasm targets
  simd_buffer_imports.mbt
```

### Probe findings

Verified by isolated wasm-gc probes (now removed from the tree):

- `FixedArray[Byte]` is rejected at the inline-WAT FFI on wasm-gc with
  `Invalid stub type`.
- `FixedArray[Int]` / `Bytes` *are* accepted at the FFI, but arrive as
  `(ref N)`. Linking fails with `expected i32, found (ref N)` if the WAT
  signature uses `(param i32)`.
- `(param anyref)` *is* accepted, but `array.get` needs a concrete type
  index that inline-WAT cannot reference, so we can't read elements through
  the GC ref — and even if we could, building a `v128` from 4 scalar reads
  + 4 `i32x4.replace_lane` costs ~9 ops vs 4 for plain scalar add, killing
  the SIMD win for memory-bound ops.
- `i32.store` / `i32.load` / `v128.load` *do* work inside inline-WAT on
  wasm-gc — wasm-gc modules still have linear memory. The blocker is
  getting a linear-memory address out of MoonBit, not the SIMD ops
  themselves.

### Allocation: `memory.grow` per buffer

```
fn buf_alloc(size : Int) -> Int =
  #|(func (param i32) (result i32)
       local.get 0 i32.const 65535 i32.add i32.const 16 i32.shr_u
       memory.grow
       i32.const 16 i32.shl)
```

Each `SimdBuffer::make` rounds `size` up to a multiple of the wasm page
size (64 KiB), grows linear memory by that many pages, and returns the new
region's base address. Each allocation owns its own pages, so there is no
shared bump pointer to collide with the host MoonBit runtime's tlsf
allocator on the `wasm` target.

Trade-offs of this allocator:

- **No free**: pages stay allocated for the lifetime of the wasm instance.
  Fine for batch workloads (offline data pipelines, request-scoped
  decoding); not fine for long-running services that allocate per-request.
- **Up to 64 KiB wasted per small allocation**: pages are the granularity.
- **`memory.grow` is a slow host call**: ~hundreds of microseconds per
  call. Allocating per hot-path call dominates the measured time.
  See "the `_into` pattern" below.

### The `_into` pattern

Because allocation is expensive, every non-trivial op that produces a new
buffer has an in-place sibling that writes into a caller-supplied output:

```
base64_encode(input)              → SimdBufferBytes        // allocates
base64_encode_into(input, out)    → Unit                   // no alloc, hot path
base64_decode(input)              → SimdBufferBytes?
base64_decode_into(input, out)    → Int?  // bytes written
```

Bench (4096-byte input on wasm-gc) showing the allocator cost dominating:

| variant | time |
|---|---|
| `base64_encode` (allocates output via `memory.grow`) | 395 µs |
| `base64_encode_into` (output pre-allocated) | 657 ns |

Same compute, 600× difference. **Use the `_into` variant whenever the
buffer can be reused.** Pool / reuse output buffers across calls.

### `SimdBufferRing`: arena allocator for the per-request pattern

When the workload needs a *fresh* output each call but the buffers all
share a lifetime (one HTTP request, one packet, one decode iteration),
the page-per-buffer allocator's `memory.grow` cost dominates everything.
`SimdBufferRing` carves several sub-buffers out of one pre-grown region:

```moonbit
let ring = SimdBufferRing::make(65536)  // one page, paid once
for _ in iterations {
  ring.reset()                           // O(1), invalidates all subs
  let out = ring.alloc_bytes(target_len)
  SimdBufferBytes::base64_encode_into(input, out)
  // ... use `out` ...
}
```

Bench (alloc 1024 bytes per call on wasm-gc):

| variant | per-call cost |
|---|---|
| `SimdBufferBytes::make(1024)` (one `memory.grow`) | 118 µs |
| `ring.alloc_bytes(1024)` (after `reset()`) | **8 ns** |

~15 000× cheaper. End-to-end base64 encode 4096 B with Ring reset is
702 ns — within noise of the `_into` baseline (690 ns), so the Ring
recovers the full SIMD win for the per-call-allocates pattern.

**Invariant:** `reset()` silently invalidates every sub-buffer carved
out of the Ring. There is no runtime tag — caller enforces it.

### Bench: full op surface on wasm-gc (V8, M-class Apple Silicon)

All `SimdBuffer*` ops on n = 1024 (vectors) / 4096 B (bytes) / 64×64 (mat).
Times are per-call wall clock; smaller is better.

#### i32 (`SimdBuffer`)

| op | time | op | time |
|---|---|---|---|
| `sum` | 105.7 ns | `min` | 106.1 ns |
| `add` | 117.6 ns | `max` | 105.4 ns |
| `sub` | 118.6 ns | `argmin` | 410.5 ns |
| `mul` | 118.7 ns | `argmax` | 447.7 ns |
| `neg` | 103.2 ns | `count_nonzero` | 198.1 ns |
| `abs` | 94.1 ns | `prod` (from lib bench) | ~700 ns |
| `min_elem` | 119.1 ns | `cumsum` | 850.0 ns |
| `max_elem` | 121.5 ns | `cumprod` | 943.6 ns |
| `eq` | 122.3 ns | `div` (f64x2 round-trip) | 422.7 ns |
| `lt` | 119.1 ns | `dot` | 177.3 ns |
| `gt` | 120.2 ns | `saxpy` | 124.4 ns |
| `sort` (1024, leaf + merge) | 10.4 µs | | |

#### f32 (`SimdBufferF32`)

| op | time |
|---|---|
| `add` | 127.4 ns |
| `mul` | 127.1 ns |
| `sqrt` | 160.1 ns |
| `sum` | 129.9 ns |
| `dot` | 147.9 ns |

#### f64 (`SimdBufferF64`)

| op | time | op | time |
|---|---|---|---|
| `add` | 229.3 ns | `sum` | 270.7 ns |
| `sub` | 229.4 ns | `mean` | 272.1 ns |
| `mul` | 229.3 ns | `variance` | 593.9 ns |
| `div` | 229.9 ns | `gemv` 256×256 | 18.2 µs |
| `sqrt` | 239.5 ns | `transpose` 128×128 | 8.2 µs |
| | | `matmul` 64×64 | 57.9 µs |

#### bytes (`SimdBufferBytes`)

| op (4096 B) | time | op (4096 B) | time |
|---|---|---|---|
| `popcount` | 290.6 ns | `is_ascii` | 125.5 ns |
| `memcpy` | 100.2 ns | `to_lower_ascii` | 105.1 ns |
| `memset` | 59.4 ns | `to_upper_ascii` | 118.7 ns |
| `equal` | 170.8 ns | `validate_utf8` (ASCII) | 157.1 ns |
| `find_byte` | 229.6 ns | `adler32` | 329.0 ns |
| `count_byte` | 165.1 ns | `base64_encode_into` | 647.4 ns |
| | | `base64_decode_into` (5460 B) | 1.57 µs |

### Headline ratios (wasm-gc, SimdBuffer SIMD vs scalar fallback via `FixedArray`)

These are the speedups a wasm-gc user gets by porting a hot path from the
FixedArray-based API (which falls back to scalar on wasm-gc) to the
SimdBuffer family. Scalar baselines come from running the same ops via
`@internal.scalar_*` on the FixedArray path.

| op | size | scalar (wasm-gc fallback) | SimdBuffer SIMD | x |
|---|---|---|---|---|
| `sum_i32` | 1024 | 344 ns | 106 ns | **3.2** |
| `add_i32` | 1024 | 413 ns | 118 ns | **3.5** |
| `popcount_bytes` | 4096 | 6.91 µs | 291 ns | **23.7** |
| `memcpy_bytes` | 4096 | 1.17 µs | 100 ns | **11.7** |
| `memset_bytes` | 4096 | 1.00 µs | 59 ns | **16.9** |
| `find_byte` | 4096 | 1.16 µs | 230 ns | **5.0** |
| `adler32_bytes` | 4096 | 9.23 µs | 329 ns | **28.1** |
| `matmul_f64` | 64×64 | (see note) | 57.9 µs | ~3-5 |
| `base64_encode` | 4096 B (into) | n/a (scalar via base64 sub-pkg) | 647 ns | — |

Byte ops show the biggest wins because the scalar fallback pays per-byte
function-call overhead, while the SIMD path processes 16 bytes per
`v128.load`. Numeric reductions and element-wise i32 ops sit in the 3-4×
range — still substantial.

Scalar `get(i) / set(i, v)` on SimdBuffer is **slower** than
`FixedArray[Int]` element access because each `get` goes through an
inline-WAT FFI call (no inlining). So SimdBuffer is only worth it when
the SIMD op dominates over the scalar pre/post processing. For
scalar-only access, FixedArray remains faster — see the
`from_array` / `to_array` copy helpers in `simd_buffer_copy.mbt` for the
hybrid pattern (build / consume via FixedArray, copy into SimdBuffer for
the SIMD phase only).

### When to use which

- **`SimdBuffer` family — recommended default.** Same code compiles on
  all four targets. SIMD acceleration on three (`wasm` / `wasm-gc` /
  `native`); scalar on `js` for API portability. Use this unless you
  have a specific reason to prefer FixedArray.
- **`FixedArray[Int]` + `simd_wasm_*.mbt` API** — when you specifically
  want GC-managed storage with no `memory.grow` lifecycle to think
  about, and you don't need wasm-gc SIMD. Note this API falls back to
  scalar on `wasm-gc` and `js`.

### Inline-WAT gotcha caught here (wasm-target collision)

The first SimdBuffer prototype put its bump pointer at linear-memory
address 0. That collides with the host MoonBit runtime's tlsf allocator
on the `wasm` target — every test that allocated a SimdBuffer triggered
`memory access out of bounds` inside `tlsf/removeBlock`. The fix was to
abandon shared bookkeeping entirely and have each allocation grow its
own pages via `memory.grow`.

### Capability matrix

What SimdBuffer can and can't do today. "Easy port" = same inline-WAT
shape as something already in `src/simd_wasm_*.mbt`, just rewrap with
`(addr : Int, len : Int)` params. "Hard port" = needs algorithmic
rework or extra storage gymnastics. "Out of scope" = doesn't fit the
linear-memory model or the current MoonBit / wasm-gc toolchain.

#### Can do (already shipped — full parity with the wasm FixedArray API)

- i32: `sum`, `dot`, `add`, `sub`, `mul`, `neg`, `abs`, `min_elem`,
  `max_elem`, `eq`, `lt`, `gt`, `where_`, `saxpy`, `min`, `max`,
  `argmin`, `argmax`, `prod`, `count_nonzero`, `any`, `all`, `cumsum`,
  `cumprod`, `div`, `gather`, `scatter`, `sort4`, `sort16`,
  `bitonic_merge8 / 16 / 32 / 64`, `sort` (with caller-supplied scratch
  buffer)
- f32: `add`, `sub`, `mul`, `div`, `sqrt`, `min_elem`, `max_elem`,
  `sum`, `dot`
- f64: `add`, `sub`, `mul`, `div`, `sqrt`, `min_elem`, `max_elem`,
  `sum`, `dot`, `mean`, `var`, `matmul`, `gemv`, `transpose`
- bytes: `popcount`, `memcpy`, `memset`, `equal`, `find_byte`,
  `count_byte`, `is_ascii`, `to_lower_ascii`, `to_upper_ascii`,
  `validate_utf8`, `adler32`, `base64_encode` / `base64_decode`
  (+ `_into` in-place variants)
- Arena allocator: `SimdBufferRing::make / reset / alloc_i32 / alloc_f32 /
  alloc_f64 / alloc_bytes` — cheap per-call alloc when the lifetime is
  shared
- 16-byte aligned linear-memory storage, `v128.load` / `v128.store` directly
- Separate `addr` ⇒ matmul-style 3-buffer ops (no aliasing checks; caller's
  responsibility)
- Per-buffer page ownership via `memory.grow` ⇒ no collision with the
  host MoonBit runtime's tlsf allocator on the wasm target
- Works on both `wasm` and `wasm-gc` from identical inline-WAT bodies

#### Easy ports (none remaining — all ops covered above)

#### Harder ports notes (now shipped, but worth remembering)

- `div_i32` — `i32 / i32` via `f64x2.div` round-trip. The 16-byte slack
  in `SimdBuffer::make` covers the conversion's lookahead.
- `cumsum` / `cumprod` — Hillis-Steele prefix carry threaded across
  chunks via an `i32x4.splat(running)` step at each chunk boundary.
- `gather` / `scatter` — gather lifts the store side to SIMD (4 stores
  fused via `i32x4.replace_lane` + `v128.store`); scatter stays scalar
  (no SIMD conflict detection in wasm).
- `transpose_f64` (2×2 block) — row/col strides as params; odd
  dimensions fall through to a scalar element-copy loop.
- `sort` — bottom-up merge sort with SIMD `sort16` leaf +
  `bitonic_merge32` and `bitonic_merge64` ladder. **Requires a
  caller-supplied scratch SimdBuffer of equal length**, because the
  internal `memory.grow` would dominate the cost otherwise. Pair with
  `SimdBufferRing` for cheap scratch allocation.

#### Out of scope / not viable on current MoonBit + wasm-gc

- **Free / deallocate**: bump-only allocator. Pages stay until the wasm
  instance dies. No `SimdBuffer::drop`. Long-running services need a real
  allocator (free list / slab) — not built
- **Resize in place**: `make` a new one, copy via a scalar loop
- **Read elements from inline-WAT against a GC `FixedArray`**: would need
  the array's module-internal type index, which inline-WAT can't name
- **`native` target**: needs `malloc` / `free` + C FFI. Different code
  path, not implemented. `native` keeps the FixedArray API
- **`js` target**: no linear memory exposed from MoonBit; SimdBuffer
  doesn't compile here. `js` keeps the FixedArray API + scalar fallback
- **Threading / `SharedArrayBuffer`**: current MoonBit wasm-gc target is
  single-threaded, no `memory.atomic.*` exercised
- **Multiple wasm memories**: only `memory 0` accessible from inline-WAT
- **Bounds checks at the WAT layer**: `buf_load_*` / `buf_store_*` trust
  the caller. Bounds are only enforced inside the MoonBit `get` / `set`
  wrappers. The SIMD primitive bodies (e.g., `buf_sum_i32(addr, len)`)
  walk the buffer without any guard — pass a bogus `len` and you walk
  off the page
- **Cross-instance / cross-module sharing**: addresses are
  instance-local; do not try to serialise a `SimdBuffer.addr` across the
  JS / wasm boundary

#### Open questions worth verifying when needed

- Whether SimdBuffer-style buffers can be passed as views to host JS
  code via wasm-bindgen-style mechanisms (probably yes if MoonBit
  exports `memory`; not tried)
- Whether wasm threads / atomics survive MoonBit's current wasm-gc
  toolchain
- Whether `memory64` (i64 addresses) lands in MoonBit; current code
  assumes 32-bit addresses everywhere
- Whether a `SimdBufferRing` (one big page, multiple sub-allocations
  with a shared bump pointer at the page header) gives a useful middle
  ground between "one page per buffer" and "real allocator". The
  trade-off is that all sub-buffers in the same page have the same
  lifetime, which fits many request-scoped workloads

## Downstream integrations (case studies)

How three mizchi repos picked up wasm SIMD by vendoring inline-WAT
helpers (rather than depending on `@simd_buffer` and paying the
copy-hop between their native types and `SimdBufferBytes`):

### mizchi/zlib (0.4.6 + 0.4.7)

Two SIMD ops landed via target-conditional `_simd.mbt` / `_scalar.mbt`
files mirroring this repo's layout:

- **adler32** (0.4.6): port of `simd_wasm_bytes.mbt`'s
  `adler32_chunks` inline-WAT, taking `Bytes` directly (`Bytes`-direct
  FFI works on wasm; rejected on wasm-gc, which falls through to
  scalar). Bench: 256 KB **166 µs → 21 µs ≈ 7.9×**.
- **crc32** (0.4.7): slicing-by-8 algorithm in both scalar and SIMD
  variants. Scalar version uses nested `FixedArray[FixedArray[UInt]]`
  (8 × 256 entries) for ~2.4× scalar. SIMD version uses a flattened
  `FixedArray[Int]` of 2048 entries with all 8 table lookups inlined
  as `i32.load` ops, plus `i32.load` for the input bytes (skips
  MoonBit's per-byte `Bytes[i].to_int().reinterpret_as_uint()`).
  Bench: 256 KB **728 µs → 145 µs ≈ 5.0×**.

The pattern: a `_simd.mbt` file targeted to `wasm` only that wraps the
hot inner loop in inline-WAT. wasm-gc / native / js share the
`_scalar.mbt` algorithm. **CLMUL is absent on wasm SIMD**, so slicing-
by-8 is the realistic ceiling — pure SIMD CRC32 without CLMUL is
nontrivial and we didn't pursue it.

### mizchi/pixelmatch (0.6.1)

`pixelmatch_simple_prefilter` walks rows of `FixedArray[Int]` looking
for byte-identical scanlines (the common case in VRT). The
per-element scalar walk replaced by a single inline-WAT call doing
`i32x4.eq + i8x16.bitmask` with branchless mismatch accumulation —
same pattern that lives in this repo as
`SimdBuffer::array_equal`. Bench (V8, Apple Silicon):

- identical 200×200: 195 µs → 29 µs ≈ 6.7×
- identical 500×500: 1.23 ms → 191 µs ≈ 6.4×
- 5 % diff 200×200: 215 µs → 49 µs ≈ 4.4×

### mizchi/image (0.4.2)

PNG encoder filters Up + Sub vendored as wasm-only inline-WAT:

- `apply_filter_up` (encode): `buf[i] = row[i] - prev[i]` → `i8x16.sub`
  on 16-byte chunks with scalar tail.
- `apply_filter_sub` (encode): `buf[i] = row[i] - row[i - bpp]` →
  scalar head (i < bpp), then SIMD body reading `row + i` and
  `row + i - bpp`, scalar tail.
- `reconstruct_row` decoder filter 2 (Up): `buf[i] = (row[i] +
  prev[i]) mod 256` → `i8x16.add`, same shape as encoder Up.

Average + Paeth stay scalar — Average needs floor-rounding rather
than `i8x16.avgr_u`'s round-up, Paeth has per-byte branching.

Bench (V8, 64×64 RGBA PNG): decode 894 → 700 µs (~1.3×), encode
3.02 → 2.2 ms (~1.4×). Combined with the 0.4.6 SIMD adler32 carried
through mizchi/zlib.

### What's NOT integrated (and why)

- **canvas-mbt**: blend_over per-pixel f64 math. SIMD with f64x2
  gives ≤ 2× due to byte↔f64 conversion overhead. Real 4-8× win
  requires fixed-point i16x8 blend rewrite — invasive, deferred to
  0.4.0 simd alpha_blend SIMD path + canvas refactor PR.
- **mizchi/font rasterizer**: delegates to `mizchi/svg`, so the
  SIMD-able coverage accumulation lives there, not in font itself.
- **zlib inflate LZ77 memcpy**: attempted 0.4.8 with custom
  `ByteBuf` growable-FixedArray-backed type — measured **14 %
  regression** because (a) the synthesised buffer's `push` is no
  faster than MoonBit's tuned `Array.push`, (b) `to_bytes()` adds a
  full-buffer copy at the end, (c) typical LZ77 matches are 3-30
  bytes so SIMD memcpy's 16-byte chunk barely helps. A proper win
  needs either `Array::resize_uninit` upstream support or a deeper
  refactor that swaps the inflate output type end-to-end.

### Common pattern: vendor inline-WAT rather than add `@simd_buffer` dep

All three integrations use copy-paste inline-WAT (`_simd.mbt` file +
`_scalar.mbt` fallback) targeted via `moon.pkg`'s `targets:` map. None
declare a `mizchi/simd` dep. Rationale:

- The consumer types are `Bytes` / `FixedArray[Byte]` / `FixedArray[Int]`,
  not `SimdBufferBytes`. Funneling them through `@simd_buffer`'s
  SimdBufferBytes API costs a copy hop that erases the SIMD win
  (confirmed in the zlib adler32 prototype: copy-hop variant was
  **slower** than scalar).
- inline-WAT in a single file is ~10-30 lines per op. Vendoring is
  cheaper than maintaining a wider public Bytes-direct API in
  `mizchi/simd` and managing dep version bumps across 5 repos.
- mizchi/simd remains the **reference impl + algorithm catalog**.
  Downstream repos read the algorithm here and copy the inline-WAT
  body, sometimes lightly customising for their data layout.

A future `mizchi/simd` 0.4.0 may add a Bytes-direct API surface that
lets dependent repos skip the vendoring step; the design decision is
deferred until a fourth integration would benefit.

## `src/simdcore/` sub-package — faster moonbitlang/core equivalents

`src/simdcore/` is an **opt-in companion to moonbitlang/core**. It can't
replace core transparently (no monkey-patching), so instead it exposes
functions shaped like the core idioms they stand in for, letting a hot path
be swapped one call at a time:

```moonbit
a.iter().fold(init=0, fn(x, y) { x + y })  // core
@simdcore.sum(a)                          // faster equivalent

a.iter().maximum()      ->  @simdcore.maximum(a)   // Int?
a.search(x)             ->  @simdcore.search(a, x) // Int?
a.contains(x)           ->  @simdcore.contains(a, x)
a.fill(v)               ->  @simdcore.fill(a, v)
a.sort()                ->  @simdcore.sort(a)      // FixedArray[Int]
```

It is a **thin facade over the existing kernels** (`@simd.sum_i32`,
`sort_i32`, `find_byte`, …) — no new inline-WAT except two small kernels
added to the root package to round out the core surface:

- `fill_i32(arr, value)` — `i32x4.splat` + `v128.store` (memset for i32).
- `find_i32(arr, needle) -> Int` — public wrapper over the existing private
  `find_int_i32_v128` (the kernel already backing `argmin` / `argmax`).

Surface today:

- `FixedArray[Int]` reductions / search: `sum`, `product`, `dot`,
  `maximum`/`minimum` (`Int?`), `search` (`Int?`) / `contains`,
  `count_nonzero`, `fill`, `sort`.
- `FixedArray[Int]` element-wise (write into `out`): `add`, `sub`, `mul`,
  `neg`, `abs`, `saxpy` (`out = k*a + b`).
- `FixedArray[Double]` (the practical numpy dtype): reductions `sum_f64`,
  `dot_f64`, `mean_f64`, `variance_f64`; element-wise `add_f64`, `sub_f64`,
  `mul_f64`, `div_f64`, `sqrt_f64`, `min_elem_f64`, `max_elem_f64`.
- `Bytes` (read-only, **zero-copy** on wasm): `bytes_equal`, `bytes_search`
  (`Int?`) / `bytes_contains`, `bytes_count`, `bytes_is_ascii`.
- `FixedArray[Byte]` (in-place, `Bytes` is immutable): `to_lower_ascii`,
  `to_upper_ascii`.
- `String` → UTF-8 (the FFI-boundary bottleneck): `encode_utf8` (`-> Bytes`),
  `encode_utf8_into` (`-> Int`, no alloc), `is_ascii_string`.

**Result parity is guaranteed on all four targets** — every test asserts the
`simdcore` output equals the core idiom. Only throughput is
target-conditional (SIMD on `wasm`, C-FFI/auto-vec on `native`, scalar
fallback on `wasm-gc` / `js`), so the substitution is always safe.

### `Bytes`-direct FFI (verified in this toolchain)

`bytes_*` take core's immutable `Bytes` directly — no copy to
`FixedArray[Byte]` / `SimdBufferBytes`. This works because on the `wasm`
target `Bytes` crosses the inline-WAT FFI as a **linear-memory pointer to
byte[0]** (same ABI as `FixedArray[Byte]`), so `v128.load` / `i32.load8_u`
read it directly. Probed and confirmed: both `v128.load` and a scalar
`i32.load8_u` loop over a `Bytes` param return correct values on `wasm`.

On `wasm-gc` the same inline-WAT **traps at link/instantiate** (`Bytes`
arrives as a GC ref, not an i32 address) — also confirmed — so wasm-gc, like
native / js, uses the `@internal.scalar_*_b` `Bytes` loops. The root package
carries `equal_bytes_b` / `find_byte_b` / `count_byte_b` / `is_ascii_b`
(wasm inline-WAT, identical bodies to the `FixedArray[Byte]` kernels; scalar
elsewhere). This is the zero-copy `Bytes`-direct surface the downstream-
integration notes flagged as a future option — now exercised by `simdcore`.

### String → UTF-8 conversion (the FFI-boundary bottleneck)

MoonBit's `String` is UTF-16, and crossing the FFI boundary to UTF-8 `Bytes`
is a real bottleneck: core's `@encoding/utf8.encode` runs a per-code-unit
scalar loop that the wasm backend does **not** lower to a fast intrinsic
(measured ~9 µs for 4 KiB of ASCII; `@encoding/ascii.encode` is worse, and
the `decode` side is the same scalar loop — dispatching by content buys
nothing).

Verified enabler: **`String` crosses the inline-WAT FFI as a linear-memory
pointer to its UTF-16 code-unit buffer** (unit `i` at byte offset `2*i`) —
probed by reading units with `i32.load16_u`, including a `€` (U+20AC). So the
all-ASCII case vectorises:

- `is_ascii_string` — OR-fold `(unit & 0xFF80)` over 8-unit `i16x8` chunks +
  `v128.any_true`.
- `encode_utf8` / `encode_utf8_into` — if all-ASCII, narrow UTF-16 → UTF-8
  with one `i8x16.shuffle` per 16 units (picks each unit's low byte: even
  byte lanes `0,2,…,30`), scalar tail. **Non-ASCII (multi-byte, surrogate
  pairs) delegates to `@encoding/utf8.encode`**, so the result is always
  identical to core.

`String` lives in `simdcore` directly (target-split `simdcore_str_wasm.mbt` /
`simdcore_str_fallback.mbt`), not the root catalog, because the scalar
fallback just calls `@encoding/utf8` — no root kernel needed off-wasm.

| op (4 KiB ASCII) | core | simdcore | x |
|---|---|---|---|
| `encode_utf8_into` (no alloc) | 9.13 µs | 835 ns | **10.9** |
| `encode_utf8` (`-> Bytes`) | 9.07 µs | 3.06 µs | **3.0** |

`encode_utf8` is "only" 3x because `Bytes::from_fixedarray` copies the output
(no public zero-copy `FixedArray[Byte] -> Bytes`); `encode_utf8_into` writes
straight into the caller's buffer and shows the true ~11x narrowing win.
Reuse an output buffer and prefer `_into` on hot paths (same lesson as the
`SimdBuffer` `_into` family).

**Decode (UTF-8 `Bytes` → `String`) is *not* accelerated.** There is no
public zero-copy `FixedArray[UInt16] -> String` (core's
`%string.unsafe_from_uint16_fixedarray` intrinsic is private), and
dispatching to core's `ascii`/`utf8` decoders gives no win (measured — same
scalar loop). A SIMD widen (`i8x16` → `i16x8`) is ready, but without a String
sink it can't be wired up; revisit if core exposes a UTF-16-buffer → String
constructor.

### Bench (V8 / wasm, n = 1024 ints / 4096 B, core idiom vs simdcore)

| op | core idiom | simdcore | x |
|---|---|---|---|
| `sum` | 16.33 µs | 230 ns | **71** |
| `maximum` | 19.12 µs | 236 ns | **81** |
| `sort` | 263.6 µs | 29.6 µs | **8.9** |
| `fill` | 1.28 µs | 149 ns | **8.6** |
| `search` (absent) | 1.33 µs | 612 ns | **2.2** |

The headline `sum` / `maximum` ratios are inflated by iterator-closure
overhead in `a.iter().fold(...)` / `a.iter().maximum()` — the actual code a
core user writes. Against a raw scalar `for`-loop baseline the pure-SIMD win
is ~5x (see the i32 table at the top); the 70-80x figure is "what you save by
replacing the idiomatic-but-slow iterator call." `search` is only 2.2x
because the scalar path early-exits while the branchless SIMD scan reads
every chunk — pick by expected hit position, same caveat as `simd_any`.

`Bytes` ops (4096 B, vs the loop / builtin a core user would write):

| op | core idiom | simdcore | x |
|---|---|---|---|
| `bytes_is_ascii` | 5.08 µs | 241 ns | **21** |
| `bytes_count` | 5.20 µs | 323 ns | **16** |
| `bytes_search` (absent) | 5.28 µs | 615 ns | **8.6** |
| `bytes_equal` | 452 ns | 403 ns | **1.1** |

`bytes_equal` is only ~1.1x because core's `Bytes::==` is already a tight
builtin comparison, not a per-byte MoonBit loop — there's little to beat. The
big `Bytes` wins are the ops core has *no* builtin for (`count`, `search`,
`is_ascii`), where the alternative is a hand-written `for`-loop.

Element-wise `i32` (n = 1024) and `f64` (n = 1024, f64x2 so 2-way) vs the
hand-written zip loop:

| op | core loop | simdcore | x |
|---|---|---|---|
| `add` (i32) | 3.26 µs | 302 ns | **10.8** |
| `saxpy` (i32) | 3.32 µs | 337 ns | **9.9** |
| `add_f64` | 3.16 µs | 593 ns | **5.3** |
| `dot_f64` | 2.31 µs | 668 ns | **3.5** |
| `sum_f64` | 1.30 µs | 659 ns | **2.0** |

f64 ratios are lower than i32 because f64x2 packs only 2 lanes per v128 (vs 4
for i32) and the reductions carry a serial accumulator.

Run: `moon bench --target wasm -p simdcore`.

## Commands

```bash
just test          # All targets
just test-wasm     # wasm only
just test-native   # native only
just bench-native  # Benchmark on native
just bench-wasm    # Benchmark on wasm
```
