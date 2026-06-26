# Changelog

## 0.5.0

Wasm SIMD tuning with MoonBit `V128` helpers.

- **wasm: speed up byte search and UTF-8 ASCII scanning.** `popcount_bytes`,
  `find_byte`, `find_byte_b`, and the UTF-8 all-ASCII chunk skip now use small
  typed `V128` helper boundaries while preserving the faster existing inline-WAT
  kernels for paths where the helper style regressed.
- **Keep the wider V128 refactor as an experiment.** The broader rewrite is
  stored under `experiments/v128-refactor/` for reference, but is not part of
  the production package because several hot paths were substantially slower
  than the old WAT bodies.

## 0.4.1

Portability fix — no API or behaviour change.

- **native: fix `bytes_rindex` build on macOS / BSD.** `simd_rfind_byte_ffi`
  called glibc's `memrchr`, which is not declared in the macOS/BSD libc, so
  the native C stub failed to compile there (`#define _GNU_SOURCE` only helps
  on glibc). Now guarded by `__linux__` (covers glibc / musl / bionic) with a
  manual reverse-scan fallback elsewhere — native builds and tests pass on all
  platforms again (246 wasm / 246 wasm-gc / 244 native / 244 js).

## 0.4.0

A large batch since 0.3.0 — three new sub-packages, a rename, and a native
SIMD push. Same public API on all four targets (wasm / wasm-gc / native / js),
scalar fallback where a target can't accelerate.

### New packages

- **`@simdcore`** — drop-in faster equivalents of common `moonbitlang/core`
  idioms: `FixedArray[Int]` / `[Double]` reductions + element-wise, `Bytes`
  search / substring (`bytes_index_of`, `bytes_rindex`, … — native binds
  libc `memchr` / `memmem` / `memrchr`), `String` ↔ UTF-8 (`encode_utf8` /
  `decode_utf8_unsafe`), `Array` bridge, and JSON structural indexing. Results
  are identical to the core idiom on every target.
- **`@simdhash`** — cryptographic digests over `Bytes`: **SHA-256**,
  **SHA-512**, **SHA-1**, **MD5** (each `*` / `*_hex`). Multi-buffer SIMD
  batches: `sha256_x4` / `sha1_x4` / `md5_x4` (4-way; wasm inline-WAT +
  native SSE2/NEON) and `sha512_x2` (2-way `i64x2`). SHA-1 / MD5 are
  collision-broken — legacy interop only.
- **`@simdimage`** — image / pixel byte ops (`rgb_to_rgba`,
  `rgba_to_grayscale`, `channel_extract` / `channel_merge`, `lerp`,
  `alpha_blend_solid`, `histogram`) on `Bytes` / `FixedArray[Byte]` directly.
  Moved out of `SimdBufferBytes` so the buffer type stays general.

### Changed

- **Renamed `@base64` → `@simdcodec`** (byte codecs); functions are now
  `base64_*`-prefixed (`base64_encode` / `base64_decode` / …). Added in-place
  `base64_encode_into` / `base64_decode_into`; the scalar reference is no
  longer public.
- **Native is real SIMD across the board**: the root `@simd` reductions /
  element-wise / image kernels and `@simdcodec` / `@simdhash` bind to a
  gcc/clang-compiled C stub — **NEON on arm64, portable SSE2 baseline on any
  x86-64** (no `-march`).
- `@simdcore` gained `maximum_f64` / `minimum_f64`; `@simdimage`'s
  `alpha_blend_solid` takes `Byte` channel args.

### Added (root `@simd` / `@simd_buffer`)

- `validate_utf8_strict` (RFC 3629) on the root API and `SimdBufferBytes`.
- `max_f64` / `min_f64` horizontal reductions.

### Docs

- Per-package READMEs, each with a per-backend (wasm / wasm-gc / native / js)
  comparison table; a package index in the top-level README.
