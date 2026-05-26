///|
fn f64_close(a : Double, b : Double, eps : Double) -> Bool {
  let d = a - b
  let abs = if d < 0.0 { -d } else { d }
  abs < eps
}

///|
test "identical vectors → 1.0" {
  let v : FixedArray[Double] = [1.0, 2.0, 3.0, 4.0, 5.0]
  let r = cosine_similarity_scalar(v, v)
  if !f64_close(r, 1.0, 1.0e-12) {
    abort("expected 1.0, got " + r.to_string())
  }
}

///|
test "opposite vectors → -1.0" {
  let a : FixedArray[Double] = [1.0, 2.0, 3.0, 4.0]
  let b : FixedArray[Double] = [-1.0, -2.0, -3.0, -4.0]
  let r = cosine_similarity_scalar(a, b)
  if !f64_close(r, -1.0, 1.0e-12) {
    abort("expected -1.0, got " + r.to_string())
  }
}

///|
test "orthogonal vectors → 0.0" {
  let a : FixedArray[Double] = [1.0, 0.0, 0.0]
  let b : FixedArray[Double] = [0.0, 1.0, 0.0]
  let r = cosine_similarity_scalar(a, b)
  if !f64_close(r, 0.0, 1.0e-12) {
    abort("expected 0.0, got " + r.to_string())
  }
}

///|
test "naive SIMD matches scalar on 1024-dim random vectors" {
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
  let simd_r = cosine_similarity_naive_simd(simd_a, simd_b)
  if !f64_close(scalar_r, simd_r, 1.0e-9) {
    abort(
      "naive SIMD vs scalar mismatch: scalar=" +
      scalar_r.to_string() +
      " simd=" +
      simd_r.to_string(),
    )
  }
  if scalar_r < -1.0 || scalar_r > 1.0 {
    abort("cosine out of [-1, 1]: " + scalar_r.to_string())
  }
}

///|
test "precomputed-norm SIMD matches scalar" {
  let n = 1024
  let a = FixedArray::make(n, 0.0)
  let b = FixedArray::make(n, 0.0)
  let mut seed : UInt = 0x12345678U
  for i in 0..<n {
    seed = seed * 1103515245U + 12345U
    a[i] = (seed >> 8).reinterpret_as_int().to_double() / 100000.0
    seed = seed * 1103515245U + 12345U
    b[i] = (seed >> 8).reinterpret_as_int().to_double() / 100000.0
  }
  let scalar_r = cosine_similarity_scalar(a, b)
  let simd_a = @simd_buffer.SimdBufferF64::from_array(a)
  let simd_b = @simd_buffer.SimdBufferF64::from_array(b)
  let a_norm = l2_norm(simd_a)
  let b_norm = l2_norm(simd_b)
  let r = cosine_similarity_precomputed(simd_a, a_norm, simd_b, b_norm)
  if !f64_close(scalar_r, r, 1.0e-9) {
    abort(
      "precomputed-SIMD vs scalar mismatch: scalar=" +
      scalar_r.to_string() +
      " simd=" +
      r.to_string(),
    )
  }
}
