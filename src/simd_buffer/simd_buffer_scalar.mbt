// Scalar / FixedArray implementation of the SimdBuffer family.
//
// Used on the `native` and `js` targets, where inline-WAT v128 is
// unavailable. Backing storage is a GC-managed FixedArray and every
// operation delegates to `@internal.scalar_*` / `@base64.*`. The public
// API is byte-for-byte identical with the wasm / wasm-gc linear-memory
// version, so caller code using `SimdBuffer*` compiles unchanged across
// all four targets.
//
// **Performance characteristics by target:**
//   * `native` — `@internal.scalar_*` calls dispatch to the existing
//     native C FFI fast paths (NEON / SSE) where one exists. Ops without
//     an FFI helper rely on clang's auto-vectoriser via the native-stub.
//     Practical result: parity with the FixedArray-based native API.
//   * `js` — scalar only. JavaScript has no SIMD escape hatch from
//     MoonBit. SimdBuffer compiles and runs on js for **API
//     portability**, not for performance. If raw throughput on js
//     matters, keep the workload in JS-native typed arrays and call
//     into wasm where SimdBuffer is actually accelerated.

// ---- SimdBuffer (i32) ----

///|
pub struct SimdBuffer {
  data : FixedArray[Int]
}

///|
pub fn SimdBuffer::make(len : Int) -> SimdBuffer {
  { data: FixedArray::make(len, 0) }
}

///|
pub fn SimdBuffer::clear(self : SimdBuffer) -> Unit {
  let n = self.data.length()
  for i in 0..<n {
    self.data[i] = 0
  }
}

///|
pub fn SimdBuffer::length(self : SimdBuffer) -> Int {
  self.data.length()
}

///|
pub fn SimdBuffer::get(self : SimdBuffer, i : Int) -> Int {
  self.data[i]
}

///|
pub fn SimdBuffer::set(self : SimdBuffer, i : Int, v : Int) -> Unit {
  self.data[i] = v
}

///|
pub fn SimdBuffer::sum(self : SimdBuffer) -> Int {
  @internal.scalar_sum(self.data)
}

///|
pub fn SimdBuffer::dot(a : SimdBuffer, b : SimdBuffer) -> Int {
  @internal.scalar_dot_product(a.data, b.data)
}

///|
pub fn SimdBuffer::add(
  a : SimdBuffer,
  b : SimdBuffer,
  out : SimdBuffer,
) -> Unit {
  @internal.scalar_add(a.data, b.data, out.data)
}

///|
pub fn SimdBuffer::sub(
  a : SimdBuffer,
  b : SimdBuffer,
  out : SimdBuffer,
) -> Unit {
  @internal.scalar_sub(a.data, b.data, out.data)
}

///|
pub fn SimdBuffer::mul(
  a : SimdBuffer,
  b : SimdBuffer,
  out : SimdBuffer,
) -> Unit {
  @internal.scalar_mul(a.data, b.data, out.data)
}

///|
pub fn SimdBuffer::neg(a : SimdBuffer, out : SimdBuffer) -> Unit {
  @internal.scalar_neg(a.data, out.data)
}

///|
pub fn SimdBuffer::abs(a : SimdBuffer, out : SimdBuffer) -> Unit {
  @internal.scalar_abs(a.data, out.data)
}

///|
pub fn SimdBuffer::min_elem(
  a : SimdBuffer,
  b : SimdBuffer,
  out : SimdBuffer,
) -> Unit {
  @internal.scalar_min_elem(a.data, b.data, out.data)
}

///|
pub fn SimdBuffer::max_elem(
  a : SimdBuffer,
  b : SimdBuffer,
  out : SimdBuffer,
) -> Unit {
  @internal.scalar_max_elem(a.data, b.data, out.data)
}

///|
pub fn SimdBuffer::eq(a : SimdBuffer, b : SimdBuffer, out : SimdBuffer) -> Unit {
  @internal.scalar_eq(a.data, b.data, out.data)
}

///|
pub fn SimdBuffer::lt(a : SimdBuffer, b : SimdBuffer, out : SimdBuffer) -> Unit {
  @internal.scalar_lt(a.data, b.data, out.data)
}

///|
pub fn SimdBuffer::gt(a : SimdBuffer, b : SimdBuffer, out : SimdBuffer) -> Unit {
  @internal.scalar_gt(a.data, b.data, out.data)
}

///|
pub fn SimdBuffer::where_(
  mask : SimdBuffer,
  a : SimdBuffer,
  b : SimdBuffer,
  out : SimdBuffer,
) -> Unit {
  @internal.scalar_where(mask.data, a.data, b.data, out.data)
}

///|
pub fn SimdBuffer::saxpy(
  k : Int,
  a : SimdBuffer,
  b : SimdBuffer,
  out : SimdBuffer,
) -> Unit {
  @internal.scalar_saxpy(k, a.data, b.data, out.data)
}

///|
pub fn SimdBuffer::min(self : SimdBuffer) -> Int {
  @internal.scalar_min(self.data)
}

///|
pub fn SimdBuffer::max(self : SimdBuffer) -> Int {
  @internal.scalar_max(self.data)
}

///|
pub fn SimdBuffer::argmin(self : SimdBuffer) -> Int {
  @internal.scalar_argmin(self.data)
}

///|
pub fn SimdBuffer::argmax(self : SimdBuffer) -> Int {
  @internal.scalar_argmax(self.data)
}

///|
pub fn SimdBuffer::prod(self : SimdBuffer) -> Int {
  @internal.scalar_prod(self.data)
}

///|
pub fn SimdBuffer::count_nonzero(self : SimdBuffer) -> Int {
  @internal.scalar_count_nonzero(self.data)
}

///|
pub fn SimdBuffer::any(self : SimdBuffer) -> Bool {
  @internal.scalar_any(self.data)
}

///|
pub fn SimdBuffer::all(self : SimdBuffer) -> Bool {
  @internal.scalar_all(self.data)
}

///|
pub fn SimdBuffer::cumsum(arr : SimdBuffer, out : SimdBuffer) -> Unit {
  @internal.scalar_cumsum(arr.data, out.data)
}

///|
pub fn SimdBuffer::cumprod(arr : SimdBuffer, out : SimdBuffer) -> Unit {
  @internal.scalar_cumprod(arr.data, out.data)
}

///|
pub fn SimdBuffer::div(
  a : SimdBuffer,
  b : SimdBuffer,
  out : SimdBuffer,
) -> Unit {
  @internal.scalar_div(a.data, b.data, out.data)
}

///|
pub fn SimdBuffer::gather(
  arr : SimdBuffer,
  indices : SimdBuffer,
  out : SimdBuffer,
) -> Unit {
  @internal.scalar_gather(arr.data, indices.data, out.data)
}

///|
pub fn SimdBuffer::scatter(
  out : SimdBuffer,
  indices : SimdBuffer,
  vals : SimdBuffer,
) -> Unit {
  @internal.scalar_scatter(out.data, indices.data, vals.data)
}

// ---- SimdBuffer sort family ----

///|
pub fn SimdBuffer::sort4(self : SimdBuffer, off : Int) -> Unit {
  @internal.scalar_sort4_int(self.data, off)
}

///|
pub fn SimdBuffer::bitonic_merge8(self : SimdBuffer, off : Int) -> Unit {
  @internal.scalar_bitonic_merge8_int(self.data, off)
}

///|
pub fn SimdBuffer::bitonic_merge16(self : SimdBuffer, off : Int) -> Unit {
  @internal.scalar_bitonic_merge16_int(self.data, off)
}

///|
pub fn SimdBuffer::bitonic_merge32(self : SimdBuffer, off : Int) -> Unit {
  @internal.scalar_bitonic_merge32_int(self.data, off)
}

///|
pub fn SimdBuffer::bitonic_merge64(self : SimdBuffer, off : Int) -> Unit {
  @internal.scalar_bitonic_merge64_int(self.data, off)
}

///|
pub fn SimdBuffer::sort16(self : SimdBuffer, off : Int) -> Unit {
  @internal.scalar_sort16_int(self.data, off)
}

///|
/// Native sort: scratch is accepted for API parity but ignored.
pub fn SimdBuffer::sort(arr : SimdBuffer, _scratch : SimdBuffer) -> Unit {
  @internal.scalar_sort_int(arr.data)
}

// ---- SimdBuffer copy helpers ----

///|
pub fn SimdBuffer::from_array(src : FixedArray[Int]) -> SimdBuffer {
  let n = src.length()
  let data = FixedArray::make(n, 0)
  for i in 0..<n {
    data[i] = src[i]
  }
  { data, }
}

///|
pub fn SimdBuffer::copy_from_array(
  self : SimdBuffer,
  src : FixedArray[Int],
) -> Unit {
  let n = src.length()
  if n != self.data.length() {
    abort("SimdBuffer.copy_from_array: length mismatch")
  }
  for i in 0..<n {
    self.data[i] = src[i]
  }
}

///|
pub fn SimdBuffer::to_array(self : SimdBuffer) -> FixedArray[Int] {
  let n = self.data.length()
  let out = FixedArray::make(n, 0)
  for i in 0..<n {
    out[i] = self.data[i]
  }
  out
}

///|
pub fn SimdBuffer::copy_to_array(
  self : SimdBuffer,
  dst : FixedArray[Int],
) -> Unit {
  let n = self.data.length()
  if dst.length() != n {
    abort("SimdBuffer.copy_to_array: length mismatch")
  }
  for i in 0..<n {
    dst[i] = self.data[i]
  }
}

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

// ---- SimdBufferF32 ----
//
// Native scalar inline — @internal's f32 helpers take FixedArray[Byte]
// (the f32_buf encoding), so we'd have to round-trip via byte buffers to
// reuse them. The MoonBit-level scalar is small and auto-vectorises well
// under clang for the workloads where this matters.

///|
pub struct SimdBufferF32 {
  data : FixedArray[Float]
}

///|
pub fn SimdBufferF32::make(len : Int) -> SimdBufferF32 {
  { data: FixedArray::make(len, (0.0 : Float)) }
}

///|
pub fn SimdBufferF32::length(self : SimdBufferF32) -> Int {
  self.data.length()
}

///|
pub fn SimdBufferF32::get(self : SimdBufferF32, i : Int) -> Float {
  self.data[i]
}

///|
pub fn SimdBufferF32::set(self : SimdBufferF32, i : Int, v : Float) -> Unit {
  self.data[i] = v
}

///|
pub fn SimdBufferF32::sum(self : SimdBufferF32) -> Float {
  let n = self.data.length()
  let mut acc : Float = 0.0
  for i in 0..<n {
    acc = acc + self.data[i]
  }
  acc
}

///|
pub fn SimdBufferF32::dot(a : SimdBufferF32, b : SimdBufferF32) -> Float {
  let n = a.data.length()
  let mut acc : Float = 0.0
  for i in 0..<n {
    acc = acc + a.data[i] * b.data[i]
  }
  acc
}

///|
pub fn SimdBufferF32::add(
  a : SimdBufferF32,
  b : SimdBufferF32,
  out : SimdBufferF32,
) -> Unit {
  let n = a.data.length()
  for i in 0..<n {
    out.data[i] = a.data[i] + b.data[i]
  }
}

///|
pub fn SimdBufferF32::sub(
  a : SimdBufferF32,
  b : SimdBufferF32,
  out : SimdBufferF32,
) -> Unit {
  let n = a.data.length()
  for i in 0..<n {
    out.data[i] = a.data[i] - b.data[i]
  }
}

///|
pub fn SimdBufferF32::mul(
  a : SimdBufferF32,
  b : SimdBufferF32,
  out : SimdBufferF32,
) -> Unit {
  let n = a.data.length()
  for i in 0..<n {
    out.data[i] = a.data[i] * b.data[i]
  }
}

///|
pub fn SimdBufferF32::div(
  a : SimdBufferF32,
  b : SimdBufferF32,
  out : SimdBufferF32,
) -> Unit {
  let n = a.data.length()
  for i in 0..<n {
    out.data[i] = a.data[i] / b.data[i]
  }
}

///|
pub fn SimdBufferF32::sqrt(a : SimdBufferF32, out : SimdBufferF32) -> Unit {
  let n = a.data.length()
  for i in 0..<n {
    out.data[i] = a.data[i].sqrt()
  }
}

///|
pub fn SimdBufferF32::min_elem(
  a : SimdBufferF32,
  b : SimdBufferF32,
  out : SimdBufferF32,
) -> Unit {
  let n = a.data.length()
  for i in 0..<n {
    out.data[i] = if a.data[i] < b.data[i] { a.data[i] } else { b.data[i] }
  }
}

///|
pub fn SimdBufferF32::max_elem(
  a : SimdBufferF32,
  b : SimdBufferF32,
  out : SimdBufferF32,
) -> Unit {
  let n = a.data.length()
  for i in 0..<n {
    out.data[i] = if a.data[i] > b.data[i] { a.data[i] } else { b.data[i] }
  }
}

///|
pub fn SimdBufferF32::from_array(src : FixedArray[Float]) -> SimdBufferF32 {
  let n = src.length()
  let data = FixedArray::make(n, (0.0 : Float))
  for i in 0..<n {
    data[i] = src[i]
  }
  { data, }
}

///|
pub fn SimdBufferF32::copy_from_array(
  self : SimdBufferF32,
  src : FixedArray[Float],
) -> Unit {
  let n = src.length()
  if n != self.data.length() {
    abort("SimdBufferF32.copy_from_array: length mismatch")
  }
  for i in 0..<n {
    self.data[i] = src[i]
  }
}

///|
pub fn SimdBufferF32::to_array(self : SimdBufferF32) -> FixedArray[Float] {
  let n = self.data.length()
  let out : FixedArray[Float] = FixedArray::make(n, (0.0 : Float))
  for i in 0..<n {
    out[i] = self.data[i]
  }
  out
}

///|
pub fn SimdBufferF32::copy_to_array(
  self : SimdBufferF32,
  dst : FixedArray[Float],
) -> Unit {
  let n = self.data.length()
  if dst.length() != n {
    abort("SimdBufferF32.copy_to_array: length mismatch")
  }
  for i in 0..<n {
    dst[i] = self.data[i]
  }
}

// ---- SimdBufferBytes ----

///|
pub struct SimdBufferBytes {
  data : FixedArray[Byte]
}

///|
pub fn SimdBufferBytes::make(len : Int) -> SimdBufferBytes {
  { data: FixedArray::make(len, b'\x00') }
}

///|
pub fn SimdBufferBytes::length(self : SimdBufferBytes) -> Int {
  self.data.length()
}

///|
pub fn SimdBufferBytes::get(self : SimdBufferBytes, i : Int) -> Byte {
  self.data[i]
}

///|
pub fn SimdBufferBytes::set(self : SimdBufferBytes, i : Int, v : Byte) -> Unit {
  self.data[i] = v
}

///|
pub fn SimdBufferBytes::popcount(self : SimdBufferBytes) -> Int {
  @internal.scalar_popcount_bytes(self.data)
}

///|
pub fn SimdBufferBytes::memcpy(
  src : SimdBufferBytes,
  si : Int,
  dst : SimdBufferBytes,
  di : Int,
  len : Int,
) -> Unit {
  @internal.scalar_memcpy_bytes(src.data, si, dst.data, di, len)
}

///|
pub fn SimdBufferBytes::memset(
  self : SimdBufferBytes,
  off : Int,
  len : Int,
  value : Byte,
) -> Unit {
  @internal.scalar_memset_bytes(self.data, off, len, value)
}

///|
pub fn SimdBufferBytes::equal(
  a : SimdBufferBytes,
  b : SimdBufferBytes,
  len : Int,
) -> Bool {
  @internal.scalar_equal_bytes(a.data, b.data, len)
}

///|
pub fn SimdBufferBytes::find_byte(self : SimdBufferBytes, needle : Byte) -> Int {
  @internal.scalar_find_byte(self.data, needle)
}

///|
pub fn SimdBufferBytes::count_byte(
  self : SimdBufferBytes,
  needle : Byte,
) -> Int {
  @internal.scalar_count_byte(self.data, needle)
}

///|
pub fn SimdBufferBytes::is_ascii(self : SimdBufferBytes) -> Bool {
  @internal.scalar_is_ascii(self.data)
}

///|
pub fn SimdBufferBytes::to_lower_ascii(self : SimdBufferBytes) -> Unit {
  @internal.scalar_to_lower_ascii(self.data)
}

///|
pub fn SimdBufferBytes::to_upper_ascii(self : SimdBufferBytes) -> Unit {
  @internal.scalar_to_upper_ascii(self.data)
}

///|
pub fn SimdBufferBytes::validate_utf8(self : SimdBufferBytes) -> Bool {
  @internal.scalar_validate_utf8(self.data)
}

///|
pub fn SimdBufferBytes::adler32(self : SimdBufferBytes) -> UInt {
  @internal.scalar_adler32(self.data)
}

///|
pub fn SimdBufferBytes::base64_encode(
  input : SimdBufferBytes,
) -> SimdBufferBytes {
  { data: @base64.encode(input.data) }
}

///|
pub fn SimdBufferBytes::base64_decode(
  input : SimdBufferBytes,
) -> SimdBufferBytes? {
  match @base64.decode(input.data) {
    Some(d) => Some({ data: d })
    None => None
  }
}

///|
/// In-place encode. `out` must have at least `(input.length() + 2) / 3 * 4`
/// bytes; the result is written into `out.data[0..]`.
pub fn SimdBufferBytes::base64_encode_into(
  input : SimdBufferBytes,
  out : SimdBufferBytes,
) -> Unit {
  let encoded = @base64.encode(input.data)
  let n = encoded.length()
  if out.data.length() < n {
    abort("SimdBufferBytes.base64_encode_into: out too small")
  }
  for i in 0..<n {
    out.data[i] = encoded[i]
  }
}

///|
pub fn SimdBufferBytes::base64_decode_into(
  input : SimdBufferBytes,
  out : SimdBufferBytes,
) -> Int? {
  match @base64.decode(input.data) {
    Some(d) => {
      let n = d.length()
      if out.data.length() < n {
        abort("SimdBufferBytes.base64_decode_into: out too small")
      }
      for i in 0..<n {
        out.data[i] = d[i]
      }
      Some(n)
    }
    None => None
  }
}

///|
pub fn SimdBufferBytes::from_array(src : FixedArray[Byte]) -> SimdBufferBytes {
  let n = src.length()
  let data = FixedArray::make(n, b'\x00')
  for i in 0..<n {
    data[i] = src[i]
  }
  { data, }
}

///|
pub fn SimdBufferBytes::copy_from_array(
  self : SimdBufferBytes,
  src : FixedArray[Byte],
) -> Unit {
  let n = src.length()
  if n != self.data.length() {
    abort("SimdBufferBytes.copy_from_array: length mismatch")
  }
  for i in 0..<n {
    self.data[i] = src[i]
  }
}

///|
pub fn SimdBufferBytes::to_array(self : SimdBufferBytes) -> FixedArray[Byte] {
  let n = self.data.length()
  let out = FixedArray::make(n, b'\x00')
  for i in 0..<n {
    out[i] = self.data[i]
  }
  out
}

///|
pub fn SimdBufferBytes::copy_to_array(
  self : SimdBufferBytes,
  dst : FixedArray[Byte],
) -> Unit {
  let n = self.data.length()
  if dst.length() != n {
    abort("SimdBufferBytes.copy_to_array: length mismatch")
  }
  for i in 0..<n {
    dst[i] = self.data[i]
  }
}

// ---- SimdBufferRing ----
//
// On native there's no `memory.grow` cost to amortise, so the Ring is a
// thin shell: each `alloc_*` does a fresh FixedArray::make, and `reset()`
// is a no-op. API surface matches the wasm / wasm-gc version so user code
// is portable.

///|
pub struct SimdBufferRing {
  // Native Ring carries no state; the fields are only here so the public
  // surface (`capacity()` / `used()`) returns sensible numbers.
  mut total_alloc : Int
}

///|
pub fn SimdBufferRing::make(_bytes : Int) -> SimdBufferRing {
  { total_alloc: 0 }
}

///|
pub fn SimdBufferRing::capacity(self : SimdBufferRing) -> Int {
  // No fixed capacity on native — report the total bytes handed out so far.
  self.total_alloc
}

///|
pub fn SimdBufferRing::used(self : SimdBufferRing) -> Int {
  self.total_alloc
}

///|
pub fn SimdBufferRing::reset(self : SimdBufferRing) -> Unit {
  self.total_alloc = 0
}

///|
pub fn SimdBufferRing::alloc_i32(self : SimdBufferRing, n : Int) -> SimdBuffer {
  self.total_alloc = self.total_alloc + n * 4
  SimdBuffer::make(n)
}

///|
pub fn SimdBufferRing::alloc_f32(
  self : SimdBufferRing,
  n : Int,
) -> SimdBufferF32 {
  self.total_alloc = self.total_alloc + n * 4
  SimdBufferF32::make(n)
}

///|
pub fn SimdBufferRing::alloc_f64(
  self : SimdBufferRing,
  n : Int,
) -> SimdBufferF64 {
  self.total_alloc = self.total_alloc + n * 8
  SimdBufferF64::make(n)
}

///|
pub fn SimdBufferRing::alloc_bytes(
  self : SimdBufferRing,
  n : Int,
) -> SimdBufferBytes {
  self.total_alloc = self.total_alloc + n
  SimdBufferBytes::make(n)
}
