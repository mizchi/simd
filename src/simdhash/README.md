# `@simdhash` — cryptographic digests

Digests over `Bytes`: **SHA-256** (FIPS 180-4), **SHA-1** (FIPS 180-4),
**MD5** (RFC 1321). Each has `*`, `*_hex` and `*_x4` (batch) forms.

```moonbit
let digest = @simdhash.sha256(data)          // -> Bytes (32 bytes)
let hex = @simdhash.sha256_hex(data)          // -> String (64 lowercase hex)
let (d0, d1, d2, d3) = @simdhash.sha256_x4(m0, m1, m2, m3)   // batch

@simdhash.sha1_hex(data)                      // 40 hex chars
@simdhash.md5_hex(data)                       // 32 hex chars
```

> **SHA-1 and MD5 are cryptographically broken** (collisions are practical).
> They are here for legacy interop — git object ids, ETags, content addressing
> of *trusted* data — never for integrity against an adversary. Use `sha256`
> for anything security-sensitive.

## Backend comparison

| backend | single (`sha256` / `sha1` / `md5`) | `*_x4` batch |
|---|---|---|
| **wasm** | scalar¹ | `sha256_x4` / `sha1_x4` / `md5_x4` **inline-WAT 4-way SIMD**² |
| **wasm-gc** | scalar | scalar |
| **native** | scalar (gcc-compiled) | `sha256_x4` / `sha1_x4` / `md5_x4` **SSE2 / NEON 4-way multi-buffer**³ |
| **js** | scalar | scalar |

Digests are **byte-identical on every backend** (verified against the FIPS
180-4 / NIST / RFC 1321 known-answer vectors plus an equal-length sweep that
asserts every lane equals the scalar digest).

¹ **A single hash stream does not vectorise.** The compression rounds are a
tight sequential dependency, and wasm SIMD has no SHA-NI / CLMUL equivalent
(same wall as `crc32` and simdjson's `compute_quote_mask`). So the single
`sha256` / `sha1` / `md5` are scalar on every backend by design.

² **The SIMD win is multi-buffer.** `*_x4` hashes four *independent* messages
in parallel — one per SIMD lane (Intel's `sha256_mb` approach). All three run
inline-WAT `i32x4` kernels on wasm (SHA-1 / MD5 as four 16/20-round group
loops; MD5's variable per-round rotation uses a dynamic `i32x4.shl` count).

³ **Native uses real SSE2 / NEON** (`simdhash.c`, gcc/clang-compiled — SSE2 is
baseline on x86-64, NEON on arm64; a portable scalar lane-struct covers other
ISAs). `sha256_x4`, `sha1_x4` **and** `md5_x4` all run the 4-way kernel. Use
the `*_x4` forms when you have many equal-length records to hash (file chunks,
Merkle leaves, …).

### Bench (four 4 KiB messages)

| | wasm | native (SSE2) |
|---|---|---|
| `sha256` × 4 (separate calls) | 527 µs | 166 µs |
| **`sha256_x4`** (multi-buffer) | **190 µs (2.8×)** | **46 µs (3.6×)** |
| `sha1` × 4 (separate calls) | 515 µs | 121 µs |
| **`sha1_x4`** (multi-buffer) | **246 µs (2.1×)** | **36 µs (3.4×)** |
| `md5` × 4 (separate calls) | 231 µs | 98 µs |
| **`md5_x4`** (multi-buffer) | **165 µs (1.4×)** | **24 µs (4.1×)** |

Not the theoretical 4×: each lane still runs the full serial round chain and
the padding/transpose costs a pass — but four lanes advance per instruction.
(MD5's 4.1× edges past the SHAs — fewer rounds, so the per-call overhead and
transpose amortise better.)

Run: `moon bench --target native -p simdhash` (or `--target wasm`).
