// Cosine similarity of two equal-length f64 vectors.
//
//   cos(θ) = (a · b) / (|a| * |b|)
//
// Returns a value in [-1, 1]: 1 for identical directions, 0 for
// orthogonal, -1 for opposite directions. This is the workhorse of
// embedding / RAG search: rank documents by cosine distance to a query
// vector.
//
// The interesting thing about this example is that the *naive* SIMD
// translation is a regression on some targets, even though every line
// is implemented with SIMD primitives. The bench file demonstrates why
// and shows the structurally-correct fix.

///|
/// Scalar reference: single pass through the data, three running
/// accumulators (a·b, a·a, b·b). Each element of `a` and `b` is read
/// exactly once. Memory bandwidth = `2 * n * 8` bytes.
pub fn cosine_similarity_scalar(
  a : FixedArray[Double],
  b : FixedArray[Double],
) -> Double {
  let n = a.length()
  if n != b.length() {
    abort("cosine_similarity: length mismatch")
  }
  let mut dot = 0.0
  let mut na = 0.0
  let mut nb = 0.0
  for i in 0..<n {
    let ai = a[i]
    let bi = b[i]
    dot = dot + ai * bi
    na = na + ai * ai
    nb = nb + bi * bi
  }
  dot / (na.sqrt() * nb.sqrt())
}

///|
/// Naive SIMD: three `SimdBufferF64::dot` calls. Every operation along
/// the way is SIMD-accelerated, but the structure walks `a` and `b`
/// **three times each** — memory bandwidth = `6 * n * 8` bytes, three
/// times the scalar version. For workloads dominated by memory
/// bandwidth (vectors larger than L1), this loses to scalar.
///
/// Read this as the "anti-example": SIMD primitives don't compose
/// for free when the cost is memory-bound.
pub fn cosine_similarity_naive_simd(
  a : @simd_buffer.SimdBufferF64,
  b : @simd_buffer.SimdBufferF64,
) -> Double {
  let dot = @simd_buffer.SimdBufferF64::dot(a, b)
  let na = @simd_buffer.SimdBufferF64::dot(a, a)
  let nb = @simd_buffer.SimdBufferF64::dot(b, b)
  dot / (na.sqrt() * nb.sqrt())
}

///|
/// Precomputed-norm SIMD: the RAG / embedding-search pattern.
///
/// In a typical vector-search workload you normalise every database
/// vector **once** at index time, then handle queries by
///
///   cos(q, d) = dot(q, d) / |q| / |d|   ← all three known at query time
///
/// where `|q|` and `|d|` are cached. Per query you do a single dot
/// product per database vector — a single pass through the data, fully
/// SIMD-accelerated.
///
/// `q_norm` and `d_norm` are precomputed `|q|` and `|d|`; pass them in
/// from your index. The function returns the cosine.
pub fn cosine_similarity_precomputed(
  q : @simd_buffer.SimdBufferF64,
  q_norm : Double,
  d : @simd_buffer.SimdBufferF64,
  d_norm : Double,
) -> Double {
  let dot = @simd_buffer.SimdBufferF64::dot(q, d)
  dot / (q_norm * d_norm)
}

///|
/// Helper: precompute `|v|` for storage alongside a database vector.
pub fn l2_norm(v : @simd_buffer.SimdBufferF64) -> Double {
  @simd_buffer.SimdBufferF64::dot(v, v).sqrt()
}
