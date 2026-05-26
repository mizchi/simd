///|
/// `SimdBufferRing` is a single-arena bump allocator backed by a
/// `memory.grow`-allocated linear-memory region.
///
/// Use case: a request- or task-scoped workload that allocates several
/// `SimdBuffer*` instances, runs SIMD over them, then discards everything
/// at once. The Ring trades the global allocator's "one page per buffer
/// (~64 KiB wasted each)" cost for a single up-front page allocation that
/// all sub-buffers share. Reset is O(1): just rewinds the bump pointer,
/// invalidating every sub-buffer carved out of the Ring.
///
/// Trade-offs:
///   * Wins: many small buffers no longer each cost a full page; no
///     `memory.grow` syscall per buffer.
///   * Loses: all sub-buffers share the Ring's lifetime — calling
///     `reset()` (or letting the Ring die) silently invalidates every
///     `SimdBuffer*` referencing it. Caller must enforce the invariant.
///     No tagging at runtime.
pub struct SimdBufferRing {
  base : Int // first byte of the Ring's memory region
  capacity : Int // total usable bytes
  mut next : Int // next-free offset from `base`, 16-byte aligned at issue
}

///|
/// Allocate a Ring with at least `bytes` of capacity. Rounded up to whole
/// wasm pages (64 KiB each).
pub fn SimdBufferRing::make(bytes : Int) -> SimdBufferRing {
  if bytes <= 0 {
    abort("SimdBufferRing.make: bytes must be positive")
  }
  // buf_alloc already rounds up to pages; we just feed it the requested
  // byte count.
  let pages = (bytes + 65535) / 65536
  let capacity = pages * 65536
  let base = buf_alloc(capacity)
  { base, capacity, next: 0 }
}

///|
pub fn SimdBufferRing::capacity(self : SimdBufferRing) -> Int {
  self.capacity
}

///|
pub fn SimdBufferRing::used(self : SimdBufferRing) -> Int {
  self.next
}

///|
/// Reset the bump pointer to 0. **Invalidates every `SimdBuffer*` that was
/// allocated from this Ring.** Caller must not touch them again.
pub fn SimdBufferRing::reset(self : SimdBufferRing) -> Unit {
  self.next = 0
}

///|
/// Raw bump: reserve `bytes` from the Ring and return its absolute address.
/// Aligns each issue to 16 bytes so v128.load/v128.store on the returned
/// address stays aligned. Aborts on out-of-capacity.
pub fn SimdBufferRing::alloc(self : SimdBufferRing, bytes : Int) -> Int {
  if bytes <= 0 {
    return self.base + self.next // zero-byte alloc is harmless; same addr
  }
  let aligned = (self.next + 15) / 16 * 16
  if aligned + bytes > self.capacity {
    abort("SimdBufferRing: out of capacity")
  }
  let addr = self.base + aligned
  self.next = aligned + bytes
  addr
}

///|
/// Carve a `SimdBuffer` (i32) of length `n` out of the Ring. Includes a
/// 16-byte trailing slack region for safe `v128.load`.
pub fn SimdBufferRing::alloc_i32(self : SimdBufferRing, n : Int) -> SimdBuffer {
  let addr = self.alloc(n * 4 + 16)
  { addr, len: n }
}

///|
pub fn SimdBufferRing::alloc_f32(
  self : SimdBufferRing,
  n : Int,
) -> SimdBufferF32 {
  let addr = self.alloc(n * 4 + 16)
  { addr, len: n }
}

///|
pub fn SimdBufferRing::alloc_f64(
  self : SimdBufferRing,
  n : Int,
) -> SimdBufferF64 {
  let addr = self.alloc(n * 8 + 16)
  { addr, len: n }
}

///|
pub fn SimdBufferRing::alloc_bytes(
  self : SimdBufferRing,
  n : Int,
) -> SimdBufferBytes {
  let addr = self.alloc(n + 16)
  { addr, len: n }
}
