# `@simdhash` — cryptographic digests

Digests over `Bytes`: **SHA-256** (FIPS 180-4), **SHA-512** (FIPS 180-4),
**SHA-1** (FIPS 180-4), **MD5** (RFC 1321). Each has `*` and `*_hex` forms; the
32-bit hashes have a 4-way `*_x4` batch and SHA-512 has a 2-way `sha512_x2`
(it's 64-bit, so a 128-bit vector holds two lanes, not four).

```moonbit
let digest = @simdhash.sha256(data)          // -> Bytes (32 bytes)
let hex = @simdhash.sha256_hex(data)          // -> String (64 lowercase hex)
let (d0, d1, d2, d3) = @simdhash.sha256_x4(m0, m1, m2, m3)   // batch

@simdhash.sha512_hex(data)                    // 128 hex chars
@simdhash.sha1_hex(data)                      // 40 hex chars
@simdhash.md5_hex(data)                       // 32 hex chars
```

**SHA-512** (single) is scalar on every target (64-bit / `UInt64`, 80 rounds);
its batch `sha512_x2` is **2-way** SIMD (one message per `i64x2` lane — wasm
inline-WAT, native SSE2 / NEON), since a 128-bit vector holds only two 64-bit
words.

> **SHA-1 and MD5 are cryptographically broken** (collisions are practical).
> They are here for legacy interop — git object ids, ETags, content addressing
> of *trusted* data — never for integrity against an adversary. Use `sha256`
> for anything security-sensitive.

## Backend comparison

| backend | single | `*_x4` batch (32-bit hashes) | `sha512_x2` batch |
|---|---|---|---|
| **wasm** | scalar¹ | `sha256_x4` / `sha1_x4` / `md5_x4` **inline-WAT 4-way**² | **inline-WAT 2-way (`i64x2`)** |
| **wasm-gc** | scalar | scalar | scalar |
| **native** | scalar (gcc-compiled) | `sha256_x4` / `sha1_x4` / `md5_x4` **SSE2 / NEON 4-way**³ | **SSE2 / NEON 2-way** |
| **js** | scalar | scalar | scalar |

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
| `sha512` × 2 (separate calls) | 209 µs | 53 µs |
| **`sha512_x2`** (2-way multi-buffer) | **117 µs (1.8×)** | **27 µs (1.9×)** |

Not the theoretical 4× (2× for `sha512_x2`): each lane still runs the full
serial round chain and the padding/transpose costs a pass — but the lanes
advance per instruction. (MD5's 4.1× edges past the SHAs — fewer rounds, so the
per-call overhead and transpose amortise better; `sha512_x2` at 1.8–1.9× is
near its 2-lane ceiling.)

Run: `moon bench --target native -p simdhash` (or `--target wasm`).
