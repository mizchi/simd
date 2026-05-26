// Fused-SIMD correctness check, wasm + wasm-gc only.

///|
test "fused SIMD matches scalar on 1024-dim random vectors" {
  let n = 1024
  let a = FixedArray::make(n, 0.0)
  let b = FixedArray::make(n, 0.0)
  let mut seed_a : UInt = 0xDEADBEEFU
  let mut seed_b : UInt = 0xCAFEF00DU
  for i in 0..<n {
    seed_a = seed_a * 1103515245U + 12345U
    seed_b = seed_b * 1103515245U + 67890U
    a[i] = (seed_a >> 8).reinterpret_as_int().to_double() / 100000.0
    b[i] = (seed_b >> 8).reinterpret_as_int().to_double() / 100000.0
  }
  let scalar_r = cosine_similarity_scalar(a, b)
  let simd_a = @simd_buffer.SimdBufferF64::from_array(a)
  let simd_b = @simd_buffer.SimdBufferF64::from_array(b)
  let scratch = @simd_buffer.SimdBufferF64::make(3)
  let r = cosine_similarity_fused_simd(simd_a, simd_b, scratch)
  if !f64_close(scalar_r, r, 1.0e-9) {
    abort(
      "fused SIMD vs scalar mismatch: scalar=" +
      scalar_r.to_string() +
      " simd=" +
      r.to_string(),
    )
  }
}

///|
test "fused SIMD handles odd length (1025) via scalar tail" {
  let n = 1025
  let a = FixedArray::make(n, 0.0)
  let b = FixedArray::make(n, 0.0)
  for i in 0..<n {
    a[i] = (i + 1).to_double() / 100.0
    b[i] = (i * 3 + 2).to_double() / 100.0
  }
  let scalar_r = cosine_similarity_scalar(a, b)
  let simd_a = @simd_buffer.SimdBufferF64::from_array(a)
  let simd_b = @simd_buffer.SimdBufferF64::from_array(b)
  let scratch = @simd_buffer.SimdBufferF64::make(3)
  let r = cosine_similarity_fused_simd(simd_a, simd_b, scratch)
  if !f64_close(scalar_r, r, 1.0e-9) {
    abort("odd-length fused mismatch")
  }
}
