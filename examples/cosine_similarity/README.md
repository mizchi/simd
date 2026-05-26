# Example — Cosine similarity

Cosine similarity of two f64 vectors. The workhorse of vector / RAG
search:

```
cos(θ) = (a · b) / (|a| * |b|)
```

This example also doubles as a **lesson about how to structure
SIMD code** — naively chaining the existing SimdBuffer primitives can
be *slower* than scalar, and the two ways to fix it (algorithmic vs
custom inline-WAT) land on the same performance ceiling.

## Four implementations

| function | passes through `a` and `b` | uses |
|---|---|---|
| `cosine_similarity_scalar` | 1 (single loop, three accumulators) | scalar MoonBit |
| `cosine_similarity_naive_simd` | 3 (`dot(a,b)`, `dot(a,a)`, `dot(b,b)`) | public `SimdBufferF64::dot` |
| `cosine_similarity_precomputed` | 1 (one `dot(q,d)`, norms are precomputed once at index time) | public `SimdBufferF64::dot` |
| `cosine_similarity_fused_simd` | 1 (fused inline-WAT kernel, three f64x2 accumulators in one loop) | `SimdBufferF64::raw_addr()` + custom `f64x2.*` |

The `precomputed` variant is the realistic RAG / embedding-search
pattern: every database vector stores its `|d|` alongside the raw
data, so the query-time cost per pair is **one** dot product.

The `fused` variant is for when you can't (or don't want to) precompute
the norms — e.g., one-off comparisons of dynamic vectors. It drops to
custom inline-WAT and hits the same single-pass memory bandwidth as
scalar, but with f64x2 SIMD inside the loop body. Available on wasm
and wasm-gc only (inline-WAT can't compile on native / js).

## Bench (Apple Silicon, V8)

`moon bench --target wasm`:

```
scalar_cosine_1024              1.27 µs
naive_simd_cosine_1024          962 ns   (1.32× — barely)
precomputed_simd_cosine_1024    331 ns   (3.84× ✓)
fused_simd_cosine_1024          350 ns   (3.63× ✓)

scalar_cosine_4096              5.58 µs
naive_simd_cosine_4096          4.21 µs  (1.32×)
precomputed_simd_cosine_4096    1.56 µs  (3.58×)
fused_simd_cosine_4096          1.38 µs  (4.04× ✓)
```

`moon bench --target wasm-gc`:

```
scalar_cosine_1024              663 ns
naive_simd_cosine_1024          924 ns   (0.72× — SIMD is SLOWER)
precomputed_simd_cosine_1024    311 ns   (2.13× ✓)
fused_simd_cosine_1024          341 ns   (1.94× ✓)

scalar_cosine_4096              2.77 µs
naive_simd_cosine_4096          3.90 µs  (0.71× — SIMD is SLOWER)
precomputed_simd_cosine_4096    1.32 µs  (2.10× ✓)
fused_simd_cosine_4096          1.34 µs  (2.07× ✓)
```

Headline: **`precomputed` and `fused` land on the same performance
ceiling.** Both run a single pass over `a` and `b`, both use f64x2
SIMD (2-way) inside. The remaining factor that varies is whether the
host runtime has any extra inlining / register-allocation headroom,
and on these benches the answer is "barely". Memory bandwidth +
f64x2 width is the wall.

## Why the naive SIMD loses

Each `SimdBufferF64::dot(x, y)` is SIMD-accelerated and faster than a
scalar pass over the same data, but chaining three of them walks the
input arrays **three times**:

```
naive_simd(a, b) := dot(a, b)   ← reads a once, b once
                  + dot(a, a)   ← reads a twice
                  + dot(b, b)   ← reads b twice
```

Scalar can interleave all three accumulators in one loop body:

```
for i in 0..<n {
    let ai = a[i]; let bi = b[i]
    dot += ai * bi
    na  += ai * ai
    nb  += bi * bi
}
```

That's three FMAs per element but only one trip through memory. Once
the vectors leave L1, **the bus dominates** and chaining SIMD
primitives is a net loss.

## How `cosine_similarity_fused_simd` is wired

```moonbit
pub fn cosine_similarity_fused_simd(
  a : @simd_buffer.SimdBufferF64,
  b : @simd_buffer.SimdBufferF64,
  scratch : @simd_buffer.SimdBufferF64,   // must have length >= 3
) -> Double {
  fused_dot3_f64(a.raw_addr(), b.raw_addr(), a.length(), scratch.raw_addr())
  let dot = scratch.get(0)
  let na  = scratch.get(1)
  let nb  = scratch.get(2)
  dot / (na.sqrt() * nb.sqrt())
}
```

The user gets the linear-memory address of each SimdBufferF64 with
`raw_addr()` (defined only on wasm / wasm-gc; on native / js it
doesn't exist, so the call site is wasm-target-only by construction).
Then `fused_dot3_f64` is a custom inline-WAT kernel that walks both
arrays once with three f64x2 accumulators in flight. See
[`cosine_similarity_fused.mbt`](cosine_similarity_fused.mbt) for the
full WAT body.

The output goes into a caller-supplied `scratch` SimdBufferF64 of
length ≥ 3 — the kernel writes `[dot, na, nb]` there as three f64s.
Pass a long-lived scratch (or a `SimdBufferRing` allocation) to avoid
`memory.grow` per call.

## When to reach for `fused_simd` vs `precomputed`

- **`precomputed`**: use whenever the workload allows it (vector
  index, persistent embedding store, fixed corpus). Cheaper, simpler,
  no inline-WAT to maintain, same performance ceiling.
- **`fused_simd`**: use when norms can't be precomputed (one-off
  pairs, dynamic vectors, streaming data). Demonstrates that
  user-code can break out of the public API and drop to inline-WAT
  when the existing ops don't compose efficiently.

## Takeaways

- SIMD primitives don't always compose into a faster algorithm. If
  your computation is memory-bound, chaining SIMD ops can multiply
  bandwidth instead of compute.
- Move shared sub-results out of the hot path. For cosine similarity
  in a vector index, that means precomputing and caching `|v|` for
  every stored vector.
- When precomputation isn't an option, drop to inline-WAT with
  `SimdBufferF64::raw_addr()` and fuse multiple accumulators into a
  single SIMD loop. You'll hit the same memory-bandwidth ceiling, but
  you'll hit it from one pass instead of three.

## Run

```bash
moon test  --target wasm-gc     # 5 correctness tests on wasm-gc (incl. fused)
moon test  --target native      # 3 correctness tests (no fused on native)
moon bench --target wasm-gc     # 4 variants × 2 sizes
moon bench --target wasm        # same, but the wasm-target numbers
```
