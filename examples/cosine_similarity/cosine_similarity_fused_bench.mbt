// Fused-SIMD bench, wasm + wasm-gc only.

///|
test "bench/cosine_similarity 1024-dim fused" (bb : @bench.T) {
  let n = 1024
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
  let simd_a = @simd_buffer.SimdBufferF64::from_array(a)
  let simd_b = @simd_buffer.SimdBufferF64::from_array(b)
  let scratch = @simd_buffer.SimdBufferF64::make(3)
  bb.bench(name="fused_simd_cosine_1024", fn() {
    bb.keep(cosine_similarity_fused_simd(simd_a, simd_b, scratch))
  })
}

///|
test "bench/cosine_similarity 4096-dim fused" (bb : @bench.T) {
  let n = 4096
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
  let simd_a = @simd_buffer.SimdBufferF64::from_array(a)
  let simd_b = @simd_buffer.SimdBufferF64::from_array(b)
  let scratch = @simd_buffer.SimdBufferF64::make(3)
  bb.bench(name="fused_simd_cosine_4096", fn() {
    bb.keep(cosine_similarity_fused_simd(simd_a, simd_b, scratch))
  })
}
