///|
fn make_pair(n : Int) -> (FixedArray[Double], FixedArray[Double]) {
  let a = FixedArray::make(n, 0.0)
  let b = FixedArray::make(n, 0.0)
  let mut seed_a : UInt = 0xA11CE000U
  let mut seed_b : UInt = 0xB0B0B0B0U
  for i in 0..<n {
    seed_a = seed_a * 1103515245U + 12345U
    seed_b = seed_b * 1103515245U + 67890U
    a[i] = (seed_a >> 8).reinterpret_as_int().to_double() / 100000.0
    b[i] = (seed_b >> 8).reinterpret_as_int().to_double() / 100000.0
  }
  (a, b)
}

///|
/// All three variants of cosine similarity at n = 1024. The
/// precomputed-norm variant is the RAG-realistic comparison: in a
/// real index `|d|` is stored alongside each database vector, so the
/// query-time cost is one `dot` call. The naive variant is included
/// to show that three chained SIMD primitives is *worse* than one
/// scalar single-pass loop on this workload (3× memory bandwidth).
test "bench/cosine_similarity 1024-dim" (bb : @bench.T) {
  let (a, b) = make_pair(1024)
  let simd_a = @simd_buffer.SimdBufferF64::from_array(a)
  let simd_b = @simd_buffer.SimdBufferF64::from_array(b)
  let a_norm = l2_norm(simd_a)
  let b_norm = l2_norm(simd_b)
  bb.bench(name="scalar_cosine_1024", fn() {
    bb.keep(cosine_similarity_scalar(a, b))
  })
  bb.bench(name="naive_simd_cosine_1024", fn() {
    bb.keep(cosine_similarity_naive_simd(simd_a, simd_b))
  })
  bb.bench(name="precomputed_simd_cosine_1024", fn() {
    bb.keep(cosine_similarity_precomputed(simd_a, a_norm, simd_b, b_norm))
  })
}

///|
test "bench/cosine_similarity 4096-dim" (bb : @bench.T) {
  let (a, b) = make_pair(4096)
  let simd_a = @simd_buffer.SimdBufferF64::from_array(a)
  let simd_b = @simd_buffer.SimdBufferF64::from_array(b)
  let a_norm = l2_norm(simd_a)
  let b_norm = l2_norm(simd_b)
  bb.bench(name="scalar_cosine_4096", fn() {
    bb.keep(cosine_similarity_scalar(a, b))
  })
  bb.bench(name="naive_simd_cosine_4096", fn() {
    bb.keep(cosine_similarity_naive_simd(simd_a, simd_b))
  })
  bb.bench(name="precomputed_simd_cosine_4096", fn() {
    bb.keep(cosine_similarity_precomputed(simd_a, a_norm, simd_b, b_norm))
  })
}
