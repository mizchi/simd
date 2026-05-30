# `@simdcore` — faster `moonbitlang/core` equivalents

Drop-in faster equivalents of common core idioms. Swap a hot call one line
at a time; **results are identical to the core idiom on every backend**,
only the throughput is target-conditional.

```moonbit
a.iter().fold(init=0, fn(x, y) { x + y })   // core
@simdcore.sum(a)                            // faster equivalent
```

Surface: `FixedArray[Int]` / `FixedArray[Double]` reductions + element-wise,
`Bytes` search / substring (`bytes_index_of`, `bytes_rindex`, …),
`String` ↔ UTF-8 (`encode_utf8`, `decode_utf8_unsafe`), `Array` bridge, and
JSON structural indexing.

## Backend comparison

| backend | mechanism | accelerated? |
|---|---|---|
| **wasm** | inline-WAT v128; `Bytes` / `String` / `FixedArray` cross the FFI as linear-memory pointers (zero-copy) | ✅ SIMD |
| **wasm-gc** | scalar via `@internal` (GC-ref FFI blocks `v128.load`) | ❌ scalar |
| **native** | C FFI (`simdcore.c`) — NEON / SSE2 baseline + **libc** (`memchr` / `memmem` / `memrchr`) | ✅ SIMD / libc |
| **js** | scalar via `@internal` | ❌ scalar |

The two accelerated backends shine on different ops: **wasm** wins broadly on
the byte-classification / narrow-widen kernels; **native** wins biggest where
a tuned libc primitive exists (`memchr` / `memmem` / `memrchr`).

### wasm (V8, Apple Silicon — core idiom vs `@simdcore`)

| op | core idiom | `@simdcore` | x |
|---|---|---|---|
| `sum` (n=1024) | `iter().fold` 16.3 µs | 230 ns | **71** |
| `maximum` | `iter().maximum()` 19.1 µs | 236 ns | **81** |
| `sort` | `Array::sort` 264 µs | 29.6 µs | **8.9** |
| `add` (i32 element-wise) | zip loop 3.26 µs | 302 ns | **10.8** |
| `add_f64` | zip loop 3.16 µs | 593 ns | **5.3** |
| `bytes_is_ascii` (4 KiB) | 5.08 µs | 241 ns | **21** |
| `bytes_index_of` (4 KiB) | 12.6 µs | 257 ns | **49** |
| `encode_utf8_into` (4 KiB ASCII) | 9.13 µs | 835 ns | **10.9** |
| `json_classify_structural` (4 KiB) | 19.0 µs | 1.81 µs | **10.5** |

### native (vs a per-byte MoonBit loop)

| op | speedup | how |
|---|---|---|
| `bytes_rindex` (4 KiB) | **54×** | glibc `memrchr` |
| `find_byte_b` (4 KiB) | **56×** | `memchr` |
| `is_ascii_b` (4 KiB) | **17×** | SSE2 |
| `bytes_index_of` (4 KiB) | **5.5×** | glibc `memmem` |
| `count_byte_b` (4 KiB) | **4.2×** | SSE2 |
| `bytes_equal` | ~1.1× | already `memcmp` in core |
| `json` full pipeline | ~1.15× | serial bitmap loop (no libc primitive) |

> **wasm-gc / js** run the scalar `@internal` fallback — same result, ~1.0×
> (the core-idiom baseline). Worth it only for source portability; on those
> backends keep data in native types where possible.

Run: `moon bench --target wasm -p simdcore` (or `--target native`).
