# mizchi/simd

SIMD primitives for MoonBit, with the **same public API on all four
targets** (wasm / wasm-gc / native / js) and a transparent scalar
fallback where a target can't accelerate. Two entry points:

- **`@simdcore`** — drop-in faster equivalents of common
  `moonbitlang/core` idioms (`sum`, `sort`, `Bytes` search, UTF-8
  encode/decode, JSON structural indexing, …). Same shape as the core
  idiom it replaces; results are identical on every target, only the
  throughput differs. Start here if you just want core to be faster.
- **`@simd_buffer`** — the `SimdBuffer*` buffer family for numeric /
  byte / image pipelines that own their data across many SIMD ops.

Native is real SIMD too: the C FFI stub is gcc/clang-compiled, so it
uses **NEON on arm64 and a portable SSE2 baseline on any x86-64** (no
`-march` needed) across the i32 / f64 / byte op surface.

```bash
just test          # all 4 targets
just bench-wasm    # wasm benchmark
just bench-native  # native benchmark
```

## Packages & backend acceleration

Each package has its own README with a per-backend comparison table. At a
glance, which backends get real SIMD (✅) vs scalar fallback (·):

| package | wasm | wasm-gc | native | js | docs |
|---|:--:|:--:|:--:|:--:|---|
| [`@simdcore`](src/simdcore/) — faster core idioms | ✅ | · | ✅¹ | · | [README](src/simdcore/README.md) |
| [`@simdimage`](src/simdimage/) — image / pixel ops | ✅ | · | · | · | [README](src/simdimage/README.md) |
| [`@simd_buffer`](src/simd_buffer/) — portable buffer family | ✅ | ✅ | ✅ | · | [README](src/simd_buffer/README.md) |
| [`mizchi/simd/json`](json/) — UTF-8 JSON parser | ✅ | · | ✅ | · | [README](json/README.md) |
| [`@simdjson`](src/simdjson/) — JSON indexing | ✅ | ✅ | · | · | [README](src/simdjson/README.md) |
| [`@simdcodec`](src/simdcodec/) — byte codecs (base64) | ✅ | · | ⚠️² | · | [README](src/simdcodec/README.md) |
| [`@simdhash`](src/simdhash/) — SHA-256 / SHA-512 / SHA-1 / MD5 | ✅³ | · | ✅³ | · | [README](src/simdhash/README.md) |
| `@simd` — FixedArray root API | ✅ | · | ✅ | · | this file |

¹ native `@simdcore` uses NEON / SSE2 **and** libc (`memchr` / `memmem` /
`memrchr`) — biggest wins where a tuned libc primitive exists.
² native `@simdcodec` base64 is a gcc-compiled scalar kernel (the SSSE3 `pshufb` a
vectorised base64 needs isn't in the baseline x86-64 ABI), still ~2× over
the tcc MoonBit scalar.
³ `@simdhash`: single digests (`sha256` / `sha512` / `sha1` / `md5`) are scalar
everywhere (a hash stream is serial; no SHA-NI / CLMUL on wasm). The SIMD path
is the **batch multi-buffer** kernel: the 32-bit hashes' `*_x4` (4-way) on wasm
(inline-WAT, ~1.4–2.8×) and native (SSE2 / NEON, ~3.4–4.1×), plus `sha512_x2`
(2-way `i64x2`, ~1.8–1.9×).

## Install

Add to your `moon.mod.json`:

```json
"deps": {
  "mizchi/simd": "0.4.0"
}
```

Then in the consuming package's `moon.pkg`, import the sub-packages you need:

```
import {
  "mizchi/simd/json" @simd_json,
  "mizchi/simd/src/simdcore",
  "mizchi/simd/src/simdimage",
  "mizchi/simd/src/simd_buffer",
  "mizchi/simd/src/simdjson",
  "mizchi/simd/src/simdcodec",
  "mizchi/simd/src/simdhash",
}
```

The experimental parser is explicitly aliased as `@simd_json` to avoid a
collision with `moonbitlang/core/json`. Other imports use their last path
component — `@simdcore`, `@simdimage`, `@simd_buffer`, `@simdjson`,
`@simdcodec`, `@simdhash`. The root `mizchi/simd/src` package exports the
FixedArray-based API as `@simd`.

## Quick start (recommended)

```moonbit
let a : FixedArray[Int] = [1, 2, 3, 4, 5, 6, 7, 8]
let buf = @simd_buffer.SimdBuffer::from_array(a)
let total = buf.sum()              // SIMD on wasm / wasm-gc / native, scalar on js
let out = @simd_buffer.SimdBuffer::make(buf.length())
@simd_buffer.SimdBuffer::add(buf, buf, out)
let back : FixedArray[Int] = out.to_array()
```

The same code compiles and runs unchanged on every target.

### What you get per target

| target | storage | SIMD path | notes |
|---|---|---|---|
| `wasm` | linear memory | inline-WAT `v128.*` | fastest (3-90× over scalar) |
| `wasm-gc` | linear memory | inline-WAT `v128.*` | parity with `wasm` |
| `native` | `FixedArray` | C FFI — **NEON on arm64, SSE2 baseline on any x86-64** across i32/f64/byte ops | real SIMD; parity with the FixedArray-API native fast paths |
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
  `to_upper_ascii`, `validate_utf8` (structural), `validate_utf8_strict`
  (RFC 3629 — rejects overlong / surrogate / > U+10FFFF), `adler32`,
  `base64_encode` / `base64_decode` (+ `_into` in-place variants)
- **0.3.0 byte arithmetic** (`SimdBufferBytes`): `byte_add`,
  `byte_sub`, `byte_avg`, `sat_add`, `sat_sub`, `clamp`,
  `byte_sub_offset` (PNG Sub-filter pattern)
- **image / pixel ops** now live in their own `@simdimage` package and
  operate on `Bytes` / `FixedArray[Byte]` directly (no `SimdBuffer` hop) —
  see the `@simdimage` section below
- **0.3.0 i32**: `SimdBuffer::array_equal(a, b, len) -> Bool` —
  SIMD all-equal reduction
- `SimdBufferRing`: single-arena bump allocator. On wasm/wasm-gc it
  amortises `memory.grow` across sub-allocations (~120 µs → 8 ns per
  alloc). On native/js it's a thin shell — `alloc_*` just calls
  `make` because GC alloc is already cheap.
- Copy bridges: `from_array` / `to_array` / `copy_from_array` /
  `copy_to_array` for `FixedArray` ↔ `SimdBuffer` interop.

## `@simdcore` — faster `moonbitlang/core` equivalents

The quickest win if you just want existing core-style code to go faster.
`@simdcore` can't monkey-patch core, so it exposes functions **shaped like
the core idioms they replace** — swap a hot call one line at a time:

```moonbit
a.iter().fold(init=0, fn(x, y) { x + y })  // core
@simdcore.sum(a)                           // faster equivalent

a.iter().maximum()   ->  @simdcore.maximum(a)    // Int?
a.search(x)          ->  @simdcore.search(a, x)  // Int?
a.sort()             ->  @simdcore.sort(a)       // FixedArray[Int]
```

**Results are identical to the core idiom on all four targets** — every
test asserts equality. Only throughput is target-conditional (SIMD on
`wasm`, NEON/SSE2 C-FFI on `native`, scalar on `wasm-gc` / `js`), so the
substitution is always safe.

Surface:

- **`FixedArray[Int]`**: `sum`, `product`, `dot`, `maximum`/`minimum`,
  `search`/`contains`, `count_nonzero`, `fill`, `sort`; element-wise
  `add`, `sub`, `mul`, `neg`, `abs`, `saxpy`.
- **`FixedArray[Double]`**: `sum_f64`, `dot_f64`, `mean_f64`,
  `variance_f64`; `add_f64`, `sub_f64`, `mul_f64`, `div_f64`, `sqrt_f64`,
  `min_elem_f64`, `max_elem_f64`.
- **`Bytes` (zero-copy on wasm + native)**: `bytes_equal`,
  `bytes_search`/`bytes_contains`, `bytes_count`, `bytes_is_ascii`,
  `bytes_index_of`/`bytes_contains_sub` (substring), `bytes_rindex`
  (last byte). Native binds these to libc `memchr` / `memmem` / `memrchr`.
- **`FixedArray[Byte]` (in-place)**: `to_lower_ascii`, `to_upper_ascii`.
- **`String` ↔ UTF-8** (MoonBit's biggest FFI bottleneck):
  `encode_utf8` / `encode_utf8_into`, `is_ascii_string`,
  `decode_utf8_unsafe` / `decode_utf8_unsafe_intrinsic`. ASCII fast path
  vectorises the UTF-16↔UTF-8 narrow/widen; non-ASCII delegates to
  `@encoding/utf8` so the result always matches core.
- **`Array[Int]` / `Array[Double]` bridge**: `to_fixedarray` /
  `of_fixedarray` plus copy-in one-shots `array_sum`, `array_sort`,
  `array_dot`, `array_mean_f64`, …
- **JSON structural indexing over `Bytes`**: `json_classify_structural`
  (bitmap), `json_structural_indices` / `_into` (full pipeline).

### Headlines (wasm, V8, Apple Silicon)

| op | core idiom | `@simdcore` | x |
|---|---|---|---|
| `sum` (n=1024) | `iter().fold` 16.3 µs | 230 ns | **71** |
| `maximum` | `iter().maximum()` 19.1 µs | 236 ns | **81** |
| `sort` | `Array::sort` 264 µs | 29.6 µs | **8.9** |
| `add` (i32 element-wise) | zip loop 3.26 µs | 302 ns | **10.8** |
| `bytes_is_ascii` (4 KiB) | 5.08 µs | 241 ns | **21** |
| `bytes_index_of` (4 KiB) | 12.6 µs | 257 ns | **49** |
| `encode_utf8_into` (4 KiB ASCII) | 9.13 µs | 835 ns | **10.9** |
| `json_classify_structural` (4 KiB) | 19.0 µs | 1.81 µs | **10.5** |

The big `sum`/`maximum` ratios include the per-element closure overhead of
the *idiomatic* `iter()` call; vs a raw scalar `for`-loop the pure-SIMD win
is ~5×. `bytes_index_of` / `bytes_rindex` are also big native wins — they
bind straight to libc (`memmem` 5.5×, `memrchr` 54× over a per-byte loop).

Run: `moon bench --target wasm -p simdcore` (or `--target native`).

## `@simdimage` — SIMD image / pixel ops

Pixel-oriented byte kernels, kept in their own package so the numeric
`@simd_buffer` family stays about general storage. Like `@simdcore`,
these operate on core types directly — `Bytes` for read-only inputs,
`FixedArray[Byte]` for mutable outputs — so an image library holding
those types can call straight in with no `SimdBufferBytes` copy hop.

```moonbit
let rgb = Bytes::from_array(pixels)          // n*3 RGB bytes
let rgba : FixedArray[Byte] = FixedArray::make(n * 4, b'\x00')
@simdimage.rgb_to_rgba(rgb, 0xFF, rgba)      // expand to RGBA, opaque
```

| op | what it does |
|---|---|
| `rgb_to_rgba(src, alpha, out)` | 3 byte/px RGB → 4 byte/px RGBA, constant alpha |
| `rgba_to_grayscale(src, out)` | Rec. 601 `Y = (77R+150G+29B) >> 8` |
| `channel_extract(src, ch, out)` | pull one RGBA channel into a planar buffer |
| `channel_merge(r, g, b, a, out)` | interleave 4 planar streams → RGBA |
| `lerp(a, b, t, out)` | `(a*(256-t) + b*t) >> 8`, `t ∈ 0..=256` |
| `alpha_blend_solid(dst, r, g, b, a)` | in-place premultiplied source-over of a solid color |
| `histogram(src, bins)` | 256-bin byte histogram (scalar — no SIMD scatter) |

Same target story as the rest of the library: **SIMD on `wasm`**
(`i8x16.shuffle` / `i16x8.extmul` inline-WAT; `Bytes` / `FixedArray`
cross the FFI as linear-memory pointers), **scalar on `wasm-gc` / `native`
/ `js`** (GC-ref FFI blocks `v128.load`). Results are byte-identical on
all four.

### Bench (wasm, V8, 4096 px, vs a plain per-byte loop)

| op | scalar | SIMD | x |
|---|---|---|---|
| `rgb_to_rgba` | 25.6 µs | 1.63 µs | **15.7** |
| `alpha_blend_solid` | 57.3 µs | 3.09 µs | **18.5** |
| `lerp` (4096 B) | 12.0 µs | 866 ns | **13.9** |
| `channel_merge` | 27.0 µs | 2.54 µs | **10.6** |
| `channel_extract` | 7.33 µs | 1.10 µs | **6.7** |
| `rgba_to_grayscale` | 16.2 µs | 4.84 µs | **3.3** |

Run: `moon bench --target wasm -p simdimage`.

## `mizchi/simd/json` — indexed UTF-8 JSON parser

Experimental end-to-end parser returning MoonBit's standard `Json` value.
Native and Wasm classify 16 UTF-8 bytes at a time, carve string context from
quote/backslash masks, and feed a compact token-position tape to stage 2.

```moonbit
let bytes = @utf8.encode("{\"name\":\"moon\",\"items\":[1,2,3]}"[:])
let value = try! @simd_json.parse(bytes)
```

On the included 9.9 MB mixed benchmark, `parse(Bytes)` measured 50.32 ms vs
55.20 ms for core/json on native, and 119.69 ms vs 149.23 ms on Wasm. The
7.6 MB number-heavy input measured 14.56 ms vs 22.36 ms on native. See the
[package README](json/README.md) for all results, the pipeline, and current
diagnostic/escape-heavy limitations.

## `@simdjson` — JSON byte classification

Sub-package porting simdjson's `find_structural_bits` core to wasm SIMD.
Operates on `@simd_buffer.SimdBufferBytes` input and produces i32
bitmaps + structural-index arrays in `@simd_buffer.SimdBuffer`.

```moonbit
let input = @simd_buffer.SimdBufferBytes::from_array(json_bytes)
let words = (input.length() + 31) / 32
let structural = @simd_buffer.SimdBuffer::make(words)
let quote_mask = @simd_buffer.SimdBuffer::make(words)
let indices = @simd_buffer.SimdBuffer::make(input.length())
let count = @simdjson.find_structural_indices_with_scratch(
  input, structural, quote_mask, indices,
)
// indices.get(0..count) now hold byte offsets of `{ } [ ] , :` outside any string
```

Pipeline phases (each independently callable):

| op | what it does | wasm vs native scalar |
|---|---|---|
| `classify_structural` | bitmask of `{ } [ ] , :` positions | **6.9×** |
| `classify_numeric` | bitmask of `0-9 - + . e E` positions | **5.6×** |
| `classify_quote_raw` | bitmask of `"` positions (raw, pre-escape) | **6.6×** |
| `classify_backslash` | bitmask of `\` positions | similar |
| `compute_quote_mask` | in-string mask honouring `\"` / `\\` escapes | **0.46×** (loses — see below) |
| `extract_structural_indices` | bit-walk → byte offsets via `i32.ctz` | **11×** |
| `find_structural_indices_with_scratch` | full pipeline | **1.27×** end-to-end |

`compute_quote_mask` is the bottleneck: simdjson's x86 path uses CLMUL
to prefix-XOR a 64-bit quote bitmap in one instruction, and wasm SIMD
has no equivalent. The bit-walk stays scalar in inline-WAT (branchless
`select`), so the per-byte FFI overhead is what loses to a native
scalar tight loop. Until wasm grows CLMUL, that's the floor.

Same portable shape as `@simd_buffer`: wasm / wasm-gc do inline-WAT
v128 SIMD on the byte-classification phases; native / js fall back to
the FixedArray scalar implementation. Public API identical across all
four targets.

### Memory model gotcha (wasm-only)

On `wasm` / `wasm-gc`, `SimdBuffer` storage comes from `memory.grow`
and is **never freed**. Suitable for batch / request-scoped workloads.
Long-running services should use `SimdBufferRing` and `reset()` to
recycle a single grown region across calls.

```moonbit
let ring = @simd_buffer.SimdBufferRing::make(65536)
for input in inputs {
  ring.reset()
  let out = ring.alloc_bytes((input.length() + 2) / 3 * 4)
  @simd_buffer.SimdBufferBytes::base64_encode_into(input, out)
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
let total = @simd.sum_i32(arr)                   // wasm SIMD, scalar on others
```

| target | acceleration |
|---|---|
| `wasm` | inline-WAT `v128` SIMD (real SIMD) |
| `wasm-gc` | scalar fallback (GC-ref FFI blocks `v128.load`) |
| `native` | C FFI — **NEON (arm64) / SSE2 baseline (x86-64)** across the i32 / f64 / byte op surface; reductions, element-wise, image, base64 all wired to the FFI |
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
json/                                     # mizchi/simd/json — UTF-8 Json parser
  lexer.mbt                               # v128 context carving + scalar fallback
  parser.mbt                              # indexed stage-2 parser

src/
  # FixedArray-API root package — imported as @simd
  simd_wasm_{i32,f64,f32,bytes,sort}.mbt   # wasm inline-WAT v128
  simd_native.mbt + simd_native_ffi.mbt    # native extern "C"
  simd_native.c                            # NEON / SSE2-baseline intrinsics
  simd_scalar.mbt                          # js + wasm-gc fallback
  internal/scalar*.mbt                     # shared scalar reference impls

  simdcore/                                # @simdcore — faster core equivalents
    simdcore.mbt                           # facade (i32 / f64 / Bytes / Array)
    simdcore_str_{wasm,fallback}.mbt       # String ↔ UTF-8
    simdcore_json_{wasm,native,fallback}.mbt + simdcore.c   # JSON indexing

  simdimage/                               # @simdimage — image / pixel ops
    simdimage.mbt                          # public API (Bytes / FixedArray[Byte])
    simdimage_wasm.mbt                     # wasm inline-WAT v128
    simdimage_fallback.mbt                 # wasm-gc + native + js scalar

  simdcodec/                               # @simdcodec — byte codecs (base64)
    base64_common.mbt                      # length helpers + encode/decode wrappers
    base64_wasm.mbt                        # wasm SIMD encode_into/decode_into
    base64_fallback.mbt                    # scalar tables + reference (all but native)
    base64_scalar.mbt                      # wasm-gc + js encode_into/decode_into
    base64_native.mbt + base64.c           # native C FFI (gcc scalar, LUT decode)

  simdhash/                                # @simdhash — SHA-256 / SHA-1 / MD5
    simdhash.mbt + sha1/md5/sha512.mbt   # scalar digests + public API
    simdhash_wasm.mbt                      # wasm 4-way inline-WAT (sha256/sha1/md5 _x4)
    simdhash_native.mbt + simdhash.c       # native SSE2/NEON 4-way (sha256/sha1)

  simdjson/                                # @simdjson — JSON byte classification
    simdjson_wasm.mbt                      # wasm + wasm-gc inline-WAT
    simdjson_scalar.mbt                    # native + js scalar

  simd_buffer/                             # @simd_buffer — portable API
    simd_buffer.mbt / _f32 / _f64 / _bytes / _sort / _ring / _copy.mbt
      # wasm + wasm-gc: linear-memory storage + inline-WAT v128
    simd_buffer_scalar.mbt
      # native + js: FixedArray storage + @internal / @simdcodec delegation
    simd_buffer_imports.mbt
      # cross-target import keep-alive
```

See `CLAUDE.md` for the deep dive: per-op bench tables, inline-WAT
gotchas (the `i8x16.sub` stack-order trap, `f64.min` parser hole,
wasm-gc tlsf collision), and the SimdBuffer capability matrix.

## Examples

- [`examples/cosine_similarity/`](examples/cosine_similarity/) —
  vector / RAG search building block. **Four** implementations side by
  side (scalar, naive SIMD chained, precomputed-norm SIMD, fused
  inline-WAT) that double as a worked example of when chaining SIMD
  primitives helps and when it doesn't. precomputed and fused variants
  both land on the same memory-bandwidth × f64x2 ceiling (~4× scalar)
  — the fused variant demonstrates `SimdBufferF64::raw_addr()` as an
  escape hatch when the public ops can't compose efficiently.

## License

MIT
