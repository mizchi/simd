// Scalar / FixedArray impl of SimdBufferF64 for native + js.
// See simd_buffer_f64.mbt for the wasm + wasm-gc version.

// ---- SimdBufferF64 ----

///|
pub struct SimdBufferF64 {
  data : FixedArray[Double]
}

///|
pub fn SimdBufferF64::make(len : Int) -> SimdBufferF64 {
  { data: FixedArray::make(len, 0.0) }
}

///|
pub fn SimdBufferF64::length(self : SimdBufferF64) -> Int {
  self.data.length()
}

///|
pub fn SimdBufferF64::get(self : SimdBufferF64, i : Int) -> Double {
  self.data[i]
}

///|
pub fn SimdBufferF64::set(self : SimdBufferF64, i : Int, v : Double) -> Unit {
  self.data[i] = v
}

///|
pub fn SimdBufferF64::sum(self : SimdBufferF64) -> Double {
  @internal.scalar_sum_f64(self.data)
}

///|
pub fn SimdBufferF64::dot(a : SimdBufferF64, b : SimdBufferF64) -> Double {
  @internal.scalar_dot_f64(a.data, b.data)
}

///|
pub fn SimdBufferF64::add(
  a : SimdBufferF64,
  b : SimdBufferF64,
  out : SimdBufferF64,
) -> Unit {
  @internal.scalar_add_f64(a.data, b.data, out.data)
}

///|
pub fn SimdBufferF64::sub(
  a : SimdBufferF64,
  b : SimdBufferF64,
  out : SimdBufferF64,
) -> Unit {
  @internal.scalar_sub_f64(a.data, b.data, out.data)
}

///|
pub fn SimdBufferF64::mul(
  a : SimdBufferF64,
  b : SimdBufferF64,
  out : SimdBufferF64,
) -> Unit {
  @internal.scalar_mul_f64(a.data, b.data, out.data)
}

///|
pub fn SimdBufferF64::div(
  a : SimdBufferF64,
  b : SimdBufferF64,
  out : SimdBufferF64,
) -> Unit {
  @internal.scalar_div_f64(a.data, b.data, out.data)
}

///|
pub fn SimdBufferF64::sqrt(a : SimdBufferF64, out : SimdBufferF64) -> Unit {
  @internal.scalar_sqrt_f64(a.data, out.data)
}

///|
pub fn SimdBufferF64::min_elem(
  a : SimdBufferF64,
  b : SimdBufferF64,
  out : SimdBufferF64,
) -> Unit {
  @internal.scalar_min_f64(a.data, b.data, out.data)
}

///|
pub fn SimdBufferF64::max_elem(
  a : SimdBufferF64,
  b : SimdBufferF64,
  out : SimdBufferF64,
) -> Unit {
  @internal.scalar_max_f64(a.data, b.data, out.data)
}

///|
pub fn SimdBufferF64::mean(self : SimdBufferF64) -> Double {
  @internal.scalar_mean_f64(self.data)
}

///|
pub fn SimdBufferF64::variance(self : SimdBufferF64) -> Double {
  @internal.scalar_var_f64(self.data)
}

///|
pub fn SimdBufferF64::matmul(
  a : SimdBufferF64,
  b_t : SimdBufferF64,
  c : SimdBufferF64,
  m : Int,
  k : Int,
  n : Int,
) -> Unit {
  @internal.scalar_matmul_f64(a.data, b_t.data, c.data, m, k, n)
}

///|
pub fn SimdBufferF64::gemv(
  a : SimdBufferF64,
  x : SimdBufferF64,
  y : SimdBufferF64,
  m : Int,
  n : Int,
) -> Unit {
  @internal.scalar_gemv_f64(a.data, x.data, y.data, m, n)
}

///|
pub fn SimdBufferF64::transpose(
  a : SimdBufferF64,
  out : SimdBufferF64,
  m : Int,
  n : Int,
) -> Unit {
  @internal.scalar_transpose_f64(a.data, out.data, m, n)
}

///|
pub fn SimdBufferF64::from_array(src : FixedArray[Double]) -> SimdBufferF64 {
  let n = src.length()
  let data = FixedArray::make(n, 0.0)
  for i in 0..<n {
    data[i] = src[i]
  }
  { data, }
}

///|
pub fn SimdBufferF64::copy_from_array(
  self : SimdBufferF64,
  src : FixedArray[Double],
) -> Unit {
  let n = src.length()
  if n != self.data.length() {
    abort("SimdBufferF64.copy_from_array: length mismatch")
  }
  for i in 0..<n {
    self.data[i] = src[i]
  }
}

///|
pub fn SimdBufferF64::to_array(self : SimdBufferF64) -> FixedArray[Double] {
  let n = self.data.length()
  let out = FixedArray::make(n, 0.0)
  for i in 0..<n {
    out[i] = self.data[i]
  }
  out
}

///|
pub fn SimdBufferF64::copy_to_array(
  self : SimdBufferF64,
  dst : FixedArray[Double],
) -> Unit {
  let n = self.data.length()
  if dst.length() != n {
    abort("SimdBufferF64.copy_to_array: length mismatch")
  }
  for i in 0..<n {
    dst[i] = self.data[i]
  }
}
