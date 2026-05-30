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

| backend | `sha256` (single) | `sha256_x4` (batch) |
|---|---|---|
| **wasm** | scalar¹ | **4-way multi-buffer SIMD**² |
| **wasm-gc** | scalar | scalar |
| **native** | scalar (gcc-compiled) | scalar |
| **js** | scalar | scalar |

Digests are **byte-identical on every backend** (verified against the FIPS
180-4 / NIST known-answer vectors).

¹ **A single SHA-256 stream does not vectorise.** The 64-round compression is
a tight sequential dependency, and wasm SIMD has no SHA-NI / CLMUL equivalent
(same wall as `crc32` and simdjson's `compute_quote_mask`). So `sha256` is
scalar on all backends by design.

² **The SIMD win is multi-buffer.** `sha256_x4` hashes four *independent*
messages in parallel — one per `i32x4` lane (Intel's `sha256_mb` approach).
On wasm with four equal-length inputs it runs the inline-WAT multi-buffer
kernel; use it when you have many equal-length records to hash (file chunks,
leaves of a Merkle tree, …).

### Bench (wasm, four 4 KiB messages)

| | time | vs 4× single |
|---|---|---|
| `sha256` × 4 (separate calls) | 527 µs | 1.0× |
| `sha256_x4` (multi-buffer SIMD) | 190 µs | **2.8×** |

(Under 2× isn't possible to beat the theoretical 4× here: each lane still runs
the same serial 64-round chain, and the transpose into lane-major layout costs
a pass — but four lanes advance per instruction, so batch throughput ~2.8×.)

**SHA-1 / MD5** are scalar on every backend (`sha1_x4` / `md5_x4` run four
scalar digests). Their single-stream compression is just as serial as
SHA-256's; a multi-buffer SIMD kernel for them is a natural follow-up on the
same pattern as `sha256_x4`.

Run: `moon bench --target wasm -p simdhash`.
