# `@simdhash` — cryptographic digests

SHA-256 (FIPS 180-4) over `Bytes`.

```moonbit
let digest = @simdhash.sha256(data)          // -> Bytes (32 bytes)
let hex = @simdhash.sha256_hex(data)          // -> String (64 lowercase hex)
let (d0, d1, d2, d3) = @simdhash.sha256_x4(m0, m1, m2, m3)   // batch
```

## Backend comparison

| backend | `sha256` (single) | `sha256_x4` (batch) |
|---|---|---|
| **wasm** | scalar¹ | scalar today; **4-way multi-buffer SIMD planned**² |
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
messages in parallel — one per `i32x4` lane — which is where v128 helps
(Intel's `sha256_mb` approach: ~4× batch throughput). The inline-WAT kernel
for this is sizeable and lands as a follow-up; the API is already stable, so
batch callers adopt the shape now and pick up the speedup transparently. Use
`sha256_x4` when you have many equal-length records to hash (file chunks,
leaves of a Merkle tree, …).

Run: `moon bench --target wasm -p simdhash`.
