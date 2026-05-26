// Scalar / FixedArray impl of SimdBuffer (i32) for native + js.
// See simd_buffer.mbt for the wasm + wasm-gc version.

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
/// Backing FixedArray. native + js only — gives downstream packages a
/// way to read / write the underlying storage without exposing struct
/// internals. Returns the same FixedArray held by `self`, so writes
/// through the returned handle are visible via `get` / `set`.
pub fn SimdBuffer::backing(self : SimdBuffer) -> FixedArray[Int] {
  self.data
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

///|
pub fn SimdBuffer::array_equal(
  a : SimdBuffer,
  b : SimdBuffer,
  len : Int,
) -> Bool {
  if len <= 0 {
    return true
  }
  if len > a.data.length() || len > b.data.length() {
    abort("SimdBuffer.array_equal: len exceeds buffer")
  }
  for i in 0..<len {
    if a.data[i] != b.data[i] {
      return false
    }
  }
  true
}
