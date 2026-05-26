# Example — Cosine similarity

Cosine similarity of two f64 vectors. The workhorse of vector / RAG
search:

```
cos(θ) = (a · b) / (|a| * |b|)
```

This example also doubles as a **lesson about how to structure
SIMD code** — naively chaining the existing SimdBuffer primitives can
be *slower* than scalar.

## Three implementations

| function | passes through `a` and `b` | memory bandwidth (n=1024) |
|---|---|---|
| `cosine_similarity_scalar` | 1 (single loop, three accumulators) | 16 KiB |
| `cosine_similarity_naive_simd` | 3 (`dot(a,b)`, `dot(a,a)`, `dot(b,b)`) | 48 KiB |
| `cosine_similarity_precomputed` | 1 (one `dot(q,d)`, norms are precomputed once at index time) | 16 KiB |

The `precomputed` variant is the realistic RAG / embedding-search
pattern: every database vector stores its `|d|` alongside the raw
data, so the query-time cost per pair is **one** dot product.

## Bench (Apple Silicon, V8)

`moon bench --target wasm`:

```
scalar_cosine_1024              1.17 µs
naive_simd_cosine_1024          959 ns   (1.22x — barely)
precomputed_simd_cosine_1024    333 ns   (3.50x ✓)

scalar_cosine_4096              5.06 µs
naive_simd_cosine_4096          6.87 µs  (0.74x — SIMD is SLOWER)
precomputed_simd_cosine_4096    1.79 µs  (2.83x ✓)
```

`moon bench --target wasm-gc`:

```
scalar_cosine_1024              660 ns
naive_simd_cosine_1024          917 ns   (0.72x — SIMD is SLOWER)
precomputed_simd_cosine_1024    309 ns   (2.14x ✓)

scalar_cosine_4096              2.68 µs
naive_simd_cosine_4096          3.80 µs  (0.71x — SIMD is SLOWER)
precomputed_simd_cosine_4096    1.27 µs  (2.11x ✓)
```

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

The fix is structural, not algorithmic: hoist the norms out of the
hot path. In a vector index you compute `|d|` once at insert / index
time and store it. At query time you do **one** `dot(q, d)` per
candidate — back to one trip through memory.

## When does naive SIMD win?

When the vectors fit in L1 *and* there's enough arithmetic per element
to amortise the extra loads. Below n ≈ 1024 it's roughly a wash. At
n=4096 (32 KiB per vector — larger than L1 already in noisy
benchmarks) it's a clear loss everywhere we tested.

## Takeaways

- SIMD primitives don't always compose into a faster algorithm. If
  your computation is memory-bound, chaining SIMD ops can multiply
  bandwidth instead of compute.
- Move shared sub-results out of the hot path. For cosine similarity
  in a vector index, that means precomputing and caching `|v|` for
  every stored vector.
- For genuinely fused multi-accumulator kernels (computing dot, na,
  nb in a single SIMD pass), the existing public API doesn't help —
  you'd need a custom inline-WAT kernel or a new
  `SimdBufferF64::dot3` primitive. Open question whether that's
  worth shipping; the precomputed-norm pattern dodges the issue for
  the vector-search use case.

## Run

```bash
moon test  --target wasm-gc     # 4 correctness tests
moon bench --target wasm-gc     # 2 bench tests, 3 variants each
moon bench --target wasm        # same, but the wasm-target numbers
```
