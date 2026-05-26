// Scalar / FixedArray impl of SimdBufferRing for native + js.
// On native / js the arena is just a list of GC-managed FixedArrays —
// no memory.grow lifecycle to think about. The reset() / alloc_* surface
// stays for cross-target source compatibility.

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
