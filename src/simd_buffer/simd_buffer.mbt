///|
/// SimdBuffer: a v128-friendly `i32` buffer for the wasm-gc target.
///
/// wasm-gc cannot do `v128.load` on `FixedArray[Int]` because the FFI passes
/// a GC ref, not a linear-memory pointer. SimdBuffer side-steps that by
/// allocating its storage in linear memory (via a tiny bump allocator) and
/// passing the raw `i32` address to inline-WAT helpers.
///
/// Trade-offs vs `FixedArray[Int]`:
///   * Wins: real SIMD on wasm-gc (no copy-hop overhead).
///   * Loses: no GC, no automatic free — the bump allocator only grows.
///     Suitable for batch / one-shot workloads, not long-running services.
///     Bounds are enforced in MoonBit (length check); the WAT layer trusts
///     them.
pub struct SimdBuffer {
  addr : Int // linear-memory address of element 0 (16-byte aligned)
  len : Int // number of i32 elements
}

///|
/// Allocate a `SimdBuffer` holding `len` i32 elements. The returned region
/// is 16-byte aligned and over-allocates 16 trailing bytes so that the
/// last v128.load (which always reads 16 bytes) stays within the bump
/// region.
pub fn SimdBuffer::make(len : Int) -> SimdBuffer {
  // 4 bytes per i32 + 16 trailing slack for the final v128.load.
  let bytes = len * 4 + 16
  let addr = buf_alloc(bytes)
  { addr, len }
}

///|
/// Zero-fill the buffer.
pub fn SimdBuffer::clear(self : SimdBuffer) -> Unit {
  for i in 0..<self.len {
    buf_store_i32(self.addr + i * 4, 0)
  }
}

///|
pub fn SimdBuffer::length(self : SimdBuffer) -> Int {
  self.len
}

///|
pub fn SimdBuffer::get(self : SimdBuffer, i : Int) -> Int {
  if i < 0 || i >= self.len {
    abort("SimdBuffer.get: index out of bounds")
  }
  buf_load_i32(self.addr + i * 4)
}

///|
pub fn SimdBuffer::set(self : SimdBuffer, i : Int, v : Int) -> Unit {
  if i < 0 || i >= self.len {
    abort("SimdBuffer.set: index out of bounds")
  }
  buf_store_i32(self.addr + i * 4, v)
}

///|
/// SIMD sum of an i32 SimdBuffer (wasm-gc target, v128 path).
pub fn SimdBuffer::sum(self : SimdBuffer) -> Int {
  if self.len == 0 {
    return 0
  }
  buf_sum_i32(self.addr, self.len)
}

///|
/// Element-wise SIMD add: out[i] = a[i] + b[i] for all i. The three buffers
/// must have the same length.
pub fn SimdBuffer::add(
  a : SimdBuffer,
  b : SimdBuffer,
  out : SimdBuffer,
) -> Unit {
  if a.len != b.len || a.len != out.len {
    abort("SimdBuffer.add: length mismatch")
  }
  if a.len == 0 {
    return
  }
  buf_add_i32(a.addr, b.addr, out.addr, a.len)
}

// ---- bump allocator + raw memory ops ----

///|
/// Allocator: round `size` up to a multiple of the 64 KiB wasm page size,
/// grow the linear memory by that many pages, and hand back the address of
/// the new region. Each allocation owns its own pages — no shared bump
/// pointer to collide with the MoonBit runtime's tlsf allocator on the
/// wasm target.
///
/// Trade-off: small allocations waste up to one page (~64 KiB). Suitable
/// for the batch SIMD workloads this library is designed for; not for many
/// tiny buffers.
fn buf_alloc(size : Int) -> Int =
  #|(func (param i32) (result i32) local.get 0 i32.const 65535 i32.add i32.const 16 i32.shr_u memory.grow i32.const 16 i32.shl)

///|
fn buf_load_i32(addr : Int) -> Int =
  #|(func (param i32) (result i32) local.get 0 i32.load)

///|
fn buf_store_i32(addr : Int, value : Int) -> Unit =
  #|(func (param i32) (param i32) local.get 0 local.get 1 i32.store)

// ---- SIMD primitives ----

///|
/// Sum `len` i32 values starting at `addr`. Same algorithm as the wasm
/// FixedArray-based `sum_i32_v128`: i32x4 accumulator + scalar tail.
fn buf_sum_i32(addr : Int, len : Int) -> Int =
  #|(func (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local v128) i32.const 0 i32x4.splat local.set 5 local.get 1 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 2 i32.const 2 i32.shl i32.add v128.load i32x4.add local.set 5 local.get 2 i32.const 4 i32.add local.set 2 br 0 end end local.get 5 i32x4.extract_lane 0 local.get 5 i32x4.extract_lane 1 i32.add local.get 5 i32x4.extract_lane 2 i32.add local.get 5 i32x4.extract_lane 3 i32.add local.set 4 block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.const 2 i32.shl i32.add i32.load i32.add local.set 4 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end local.get 4)

///|
/// Element-wise add: out[i] = a[i] + b[i] for `len` i32 elements.
fn buf_add_i32(a_addr : Int, b_addr : Int, out_addr : Int, len : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load i32x4.add v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add i32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add i32.load i32.add i32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

// ---- i32 element-wise + comparisons ----

///|
pub fn SimdBuffer::dot(a : SimdBuffer, b : SimdBuffer) -> Int {
  if a.len != b.len {
    abort("SimdBuffer.dot: length mismatch")
  }
  if a.len == 0 {
    return 0
  }
  buf_dot_i32(a.addr, b.addr, a.len)
}

///|
pub fn SimdBuffer::sub(
  a : SimdBuffer,
  b : SimdBuffer,
  out : SimdBuffer,
) -> Unit {
  i32_bin_check(a, b, out, "sub")
  if a.len == 0 {
    return
  }
  buf_sub_i32(a.addr, b.addr, out.addr, a.len)
}

///|
pub fn SimdBuffer::mul(
  a : SimdBuffer,
  b : SimdBuffer,
  out : SimdBuffer,
) -> Unit {
  i32_bin_check(a, b, out, "mul")
  if a.len == 0 {
    return
  }
  buf_mul_i32(a.addr, b.addr, out.addr, a.len)
}

///|
pub fn SimdBuffer::neg(a : SimdBuffer, out : SimdBuffer) -> Unit {
  if a.len != out.len {
    abort("SimdBuffer.neg: length mismatch")
  }
  if a.len == 0 {
    return
  }
  buf_neg_i32(a.addr, out.addr, a.len)
}

///|
pub fn SimdBuffer::abs(a : SimdBuffer, out : SimdBuffer) -> Unit {
  if a.len != out.len {
    abort("SimdBuffer.abs: length mismatch")
  }
  if a.len == 0 {
    return
  }
  buf_abs_i32(a.addr, out.addr, a.len)
}

///|
pub fn SimdBuffer::min_elem(
  a : SimdBuffer,
  b : SimdBuffer,
  out : SimdBuffer,
) -> Unit {
  i32_bin_check(a, b, out, "min_elem")
  if a.len == 0 {
    return
  }
  buf_min_elem_i32(a.addr, b.addr, out.addr, a.len)
}

///|
pub fn SimdBuffer::max_elem(
  a : SimdBuffer,
  b : SimdBuffer,
  out : SimdBuffer,
) -> Unit {
  i32_bin_check(a, b, out, "max_elem")
  if a.len == 0 {
    return
  }
  buf_max_elem_i32(a.addr, b.addr, out.addr, a.len)
}

///|
/// `out[i] = (a[i] == b[i]) ? -1 : 0` per i32 lane.
pub fn SimdBuffer::eq(a : SimdBuffer, b : SimdBuffer, out : SimdBuffer) -> Unit {
  i32_bin_check(a, b, out, "eq")
  if a.len == 0 {
    return
  }
  buf_eq_i32(a.addr, b.addr, out.addr, a.len)
}

///|
pub fn SimdBuffer::lt(a : SimdBuffer, b : SimdBuffer, out : SimdBuffer) -> Unit {
  i32_bin_check(a, b, out, "lt")
  if a.len == 0 {
    return
  }
  buf_lt_i32(a.addr, b.addr, out.addr, a.len)
}

///|
pub fn SimdBuffer::gt(a : SimdBuffer, b : SimdBuffer, out : SimdBuffer) -> Unit {
  i32_bin_check(a, b, out, "gt")
  if a.len == 0 {
    return
  }
  buf_gt_i32(a.addr, b.addr, out.addr, a.len)
}

///|
/// `out[i] = mask[i] ? a[i] : b[i]` (mask = -1 picks `a`, mask = 0 picks
/// `b`). Implemented via `v128.bitselect`.
pub fn SimdBuffer::where_(
  mask : SimdBuffer,
  a : SimdBuffer,
  b : SimdBuffer,
  out : SimdBuffer,
) -> Unit {
  if mask.len != a.len || a.len != b.len || a.len != out.len {
    abort("SimdBuffer.where_: length mismatch")
  }
  if mask.len == 0 {
    return
  }
  buf_where_i32(mask.addr, a.addr, b.addr, out.addr, mask.len)
}

///|
fn i32_bin_check(
  a : SimdBuffer,
  b : SimdBuffer,
  out : SimdBuffer,
  op : String,
) -> Unit {
  if a.len != b.len || a.len != out.len {
    abort("SimdBuffer." + op + ": length mismatch")
  }
}

// ---- inline-WAT primitives ----

///|
fn buf_dot_i32(a : Int, b : Int, len : Int) -> Int =
  #|(func (param i32) (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local v128) i32.const 0 i32x4.splat local.set 6 local.get 2 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 6 local.get 0 local.get 3 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 3 i32.const 2 i32.shl i32.add v128.load i32x4.mul i32x4.add local.set 6 local.get 3 i32.const 4 i32.add local.set 3 br 0 end end local.get 6 i32x4.extract_lane 0 local.get 6 i32x4.extract_lane 1 i32.add local.get 6 i32x4.extract_lane 2 i32.add local.get 6 i32x4.extract_lane 3 i32.add local.set 5 block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 3 i32.const 2 i32.shl i32.add i32.load local.get 1 local.get 3 i32.const 2 i32.shl i32.add i32.load i32.mul i32.add local.set 5 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end local.get 5)

///|
fn buf_sub_i32(a : Int, b : Int, out : Int, len : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load i32x4.sub v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add i32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add i32.load i32.sub i32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
fn buf_mul_i32(a : Int, b : Int, out : Int, len : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load i32x4.mul v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add i32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add i32.load i32.mul i32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
fn buf_neg_i32(a : Int, out : Int, len : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (local i32) (local i32) local.get 2 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 1 local.get 3 i32.const 2 i32.shl i32.add local.get 0 local.get 3 i32.const 2 i32.shl i32.add v128.load i32x4.neg v128.store local.get 3 i32.const 4 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 1 local.get 3 i32.const 2 i32.shl i32.add i32.const 0 local.get 0 local.get 3 i32.const 2 i32.shl i32.add i32.load i32.sub i32.store local.get 3 i32.const 1 i32.add local.set 3 br 0 end end)

///|
fn buf_abs_i32(a : Int, out : Int, len : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (local i32) (local i32) local.get 2 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 1 local.get 3 i32.const 2 i32.shl i32.add local.get 0 local.get 3 i32.const 2 i32.shl i32.add v128.load i32x4.abs v128.store local.get 3 i32.const 4 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 1 local.get 3 i32.const 2 i32.shl i32.add i32.const 0 local.get 0 local.get 3 i32.const 2 i32.shl i32.add i32.load i32.sub local.get 0 local.get 3 i32.const 2 i32.shl i32.add i32.load local.get 0 local.get 3 i32.const 2 i32.shl i32.add i32.load i32.const 0 i32.lt_s select i32.store local.get 3 i32.const 1 i32.add local.set 3 br 0 end end)

///|
fn buf_min_elem_i32(a : Int, b : Int, out : Int, len : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load i32x4.min_s v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add i32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add i32.load local.get 0 local.get 4 i32.const 2 i32.shl i32.add i32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add i32.load i32.lt_s select i32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
fn buf_max_elem_i32(a : Int, b : Int, out : Int, len : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load i32x4.max_s v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add i32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add i32.load local.get 0 local.get 4 i32.const 2 i32.shl i32.add i32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add i32.load i32.gt_s select i32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
fn buf_eq_i32(a : Int, b : Int, out : Int, len : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load i32x4.eq v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add i32.const 0 local.get 0 local.get 4 i32.const 2 i32.shl i32.add i32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add i32.load i32.eq i32.sub i32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
fn buf_lt_i32(a : Int, b : Int, out : Int, len : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load i32x4.lt_s v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add i32.const 0 local.get 0 local.get 4 i32.const 2 i32.shl i32.add i32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add i32.load i32.lt_s i32.sub i32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
fn buf_gt_i32(a : Int, b : Int, out : Int, len : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load i32x4.gt_s v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add i32.const 0 local.get 0 local.get 4 i32.const 2 i32.shl i32.add i32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add i32.load i32.gt_s i32.sub i32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
fn buf_where_i32(mask : Int, a : Int, b : Int, out : Int, len : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 4 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 6 block loop local.get 5 local.get 6 i32.ge_s br_if 1 local.get 3 local.get 5 i32.const 2 i32.shl i32.add local.get 1 local.get 5 i32.const 2 i32.shl i32.add v128.load local.get 2 local.get 5 i32.const 2 i32.shl i32.add v128.load local.get 0 local.get 5 i32.const 2 i32.shl i32.add v128.load v128.bitselect v128.store local.get 5 i32.const 4 i32.add local.set 5 br 0 end end block loop local.get 5 local.get 4 i32.ge_s br_if 1 local.get 3 local.get 5 i32.const 2 i32.shl i32.add local.get 1 local.get 5 i32.const 2 i32.shl i32.add i32.load local.get 2 local.get 5 i32.const 2 i32.shl i32.add i32.load local.get 0 local.get 5 i32.const 2 i32.shl i32.add i32.load select i32.store local.get 5 i32.const 1 i32.add local.set 5 br 0 end end)

// ---- i32 reductions ----

///|
pub fn SimdBuffer::min(self : SimdBuffer) -> Int {
  if self.len == 0 {
    return 0
  }
  buf_min_i32(self.addr, self.len)
}

///|
pub fn SimdBuffer::max(self : SimdBuffer) -> Int {
  if self.len == 0 {
    return 0
  }
  buf_max_i32(self.addr, self.len)
}

///|
/// 2-pass argmin: compute `min` SIMD, then SIMD-find the first matching index.
pub fn SimdBuffer::argmin(self : SimdBuffer) -> Int {
  if self.len == 0 {
    return -1
  }
  let m = buf_min_i32(self.addr, self.len)
  buf_find_int_i32(self.addr, self.len, m)
}

///|
pub fn SimdBuffer::argmax(self : SimdBuffer) -> Int {
  if self.len == 0 {
    return -1
  }
  let m = buf_max_i32(self.addr, self.len)
  buf_find_int_i32(self.addr, self.len, m)
}

///|
pub fn SimdBuffer::prod(self : SimdBuffer) -> Int {
  if self.len == 0 {
    return 1
  }
  buf_prod_i32(self.addr, self.len)
}

///|
pub fn SimdBuffer::count_nonzero(self : SimdBuffer) -> Int {
  if self.len == 0 {
    return 0
  }
  buf_count_nonzero_i32(self.addr, self.len)
}

///|
pub fn SimdBuffer::any(self : SimdBuffer) -> Bool {
  if self.len == 0 {
    return false
  }
  buf_any_i32(self.addr, self.len) != 0
}

///|
pub fn SimdBuffer::all(self : SimdBuffer) -> Bool {
  if self.len == 0 {
    return true
  }
  buf_all_i32(self.addr, self.len) != 0
}

///|
/// `out[i] = k * a[i] + b[i]`.
pub fn SimdBuffer::saxpy(
  k : Int,
  a : SimdBuffer,
  b : SimdBuffer,
  out : SimdBuffer,
) -> Unit {
  if a.len != b.len || a.len != out.len {
    abort("SimdBuffer.saxpy: length mismatch")
  }
  if a.len == 0 {
    return
  }
  buf_saxpy_i32(k, a.addr, b.addr, out.addr, a.len)
}

// ---- inline-WAT primitives ----

///|
fn buf_min_i32(addr : Int, len : Int) -> Int =
  #|(func (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local v128) local.get 0 i32.load i32x4.splat local.set 5 local.get 1 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 2 i32.const 2 i32.shl i32.add v128.load i32x4.min_s local.set 5 local.get 2 i32.const 4 i32.add local.set 2 br 0 end end local.get 5 i32x4.extract_lane 0 local.set 4 local.get 4 local.get 5 i32x4.extract_lane 1 local.get 4 local.get 5 i32x4.extract_lane 1 i32.lt_s select local.set 4 local.get 4 local.get 5 i32x4.extract_lane 2 local.get 4 local.get 5 i32x4.extract_lane 2 i32.lt_s select local.set 4 local.get 4 local.get 5 i32x4.extract_lane 3 local.get 4 local.get 5 i32x4.extract_lane 3 i32.lt_s select local.set 4 block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.const 2 i32.shl i32.add i32.load local.get 4 local.get 0 local.get 2 i32.const 2 i32.shl i32.add i32.load i32.lt_s select local.set 4 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end local.get 4)

///|
fn buf_max_i32(addr : Int, len : Int) -> Int =
  #|(func (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local v128) local.get 0 i32.load i32x4.splat local.set 5 local.get 1 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 2 i32.const 2 i32.shl i32.add v128.load i32x4.max_s local.set 5 local.get 2 i32.const 4 i32.add local.set 2 br 0 end end local.get 5 i32x4.extract_lane 0 local.set 4 local.get 4 local.get 5 i32x4.extract_lane 1 local.get 4 local.get 5 i32x4.extract_lane 1 i32.gt_s select local.set 4 local.get 4 local.get 5 i32x4.extract_lane 2 local.get 4 local.get 5 i32x4.extract_lane 2 i32.gt_s select local.set 4 local.get 4 local.get 5 i32x4.extract_lane 3 local.get 4 local.get 5 i32x4.extract_lane 3 i32.gt_s select local.set 4 block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.const 2 i32.shl i32.add i32.load local.get 4 local.get 0 local.get 2 i32.const 2 i32.shl i32.add i32.load i32.gt_s select local.set 4 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end local.get 4)

///|
fn buf_find_int_i32(addr : Int, len : Int, needle : Int) -> Int =
  #|(func (param i32) (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local i32) (local v128) local.get 2 i32x4.splat local.set 7 local.get 1 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 4 i32.const -1 local.set 5 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 0 local.get 3 i32.const 2 i32.shl i32.add v128.load local.get 7 i32x4.eq i32x4.bitmask local.set 6 local.get 3 local.get 6 i32.ctz i32.add local.get 5 local.get 5 i32.const -1 i32.eq local.get 6 i32.mul select local.set 5 local.get 3 i32.const 4 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 1 i32.ge_s br_if 1 local.get 3 local.get 5 local.get 5 i32.const -1 i32.eq local.get 0 local.get 3 i32.const 2 i32.shl i32.add i32.load local.get 2 i32.eq i32.and select local.set 5 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end local.get 5)

///|
fn buf_prod_i32(addr : Int, len : Int) -> Int =
  #|(func (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local v128) i32.const 1 i32x4.splat local.set 5 local.get 1 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 2 i32.const 2 i32.shl i32.add v128.load i32x4.mul local.set 5 local.get 2 i32.const 4 i32.add local.set 2 br 0 end end local.get 5 i32x4.extract_lane 0 local.get 5 i32x4.extract_lane 1 i32.mul local.get 5 i32x4.extract_lane 2 i32.mul local.get 5 i32x4.extract_lane 3 i32.mul local.set 4 block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.const 2 i32.shl i32.add i32.load i32.mul local.set 4 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end local.get 4)

///|
fn buf_count_nonzero_i32(addr : Int, len : Int) -> Int =
  #|(func (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local v128) i32.const 0 i32x4.splat local.set 5 local.get 1 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.const 2 i32.shl i32.add v128.load local.get 5 i32x4.ne i32x4.bitmask i32.popcnt i32.add local.set 4 local.get 2 i32.const 4 i32.add local.set 2 br 0 end end block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 4 i32.const 0 local.get 0 local.get 2 i32.const 2 i32.shl i32.add i32.load i32.ne i32.add local.set 4 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end local.get 4)

///|
fn buf_any_i32(addr : Int, len : Int) -> Int =
  #|(func (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local v128) i32.const 0 i32x4.splat local.set 5 local.get 1 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.const 2 i32.shl i32.add v128.load local.get 5 i32x4.ne i32x4.bitmask i32.or local.set 4 local.get 2 i32.const 4 i32.add local.set 2 br 0 end end block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 4 i32.const 0 local.get 0 local.get 2 i32.const 2 i32.shl i32.add i32.load i32.ne i32.or local.set 4 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end local.get 4 i32.const 0 i32.ne)

///|
fn buf_all_i32(addr : Int, len : Int) -> Int =
  #|(func (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local v128) i32.const 0 i32x4.splat local.set 5 local.get 1 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.const 2 i32.shl i32.add v128.load local.get 5 i32x4.eq i32x4.bitmask i32.or local.set 4 local.get 2 i32.const 4 i32.add local.set 2 br 0 end end block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 4 i32.const 0 local.get 0 local.get 2 i32.const 2 i32.shl i32.add i32.load i32.eq i32.or local.set 4 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end local.get 4 i32.eqz)

///|
fn buf_saxpy_i32(k : Int, a : Int, b : Int, out : Int, len : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) (local v128) local.get 0 i32x4.splat local.set 7 local.get 4 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 6 block loop local.get 5 local.get 6 i32.ge_s br_if 1 local.get 3 local.get 5 i32.const 2 i32.shl i32.add local.get 7 local.get 1 local.get 5 i32.const 2 i32.shl i32.add v128.load i32x4.mul local.get 2 local.get 5 i32.const 2 i32.shl i32.add v128.load i32x4.add v128.store local.get 5 i32.const 4 i32.add local.set 5 br 0 end end block loop local.get 5 local.get 4 i32.ge_s br_if 1 local.get 3 local.get 5 i32.const 2 i32.shl i32.add local.get 0 local.get 1 local.get 5 i32.const 2 i32.shl i32.add i32.load i32.mul local.get 2 local.get 5 i32.const 2 i32.shl i32.add i32.load i32.add i32.store local.get 5 i32.const 1 i32.add local.set 5 br 0 end end)

// ---- i32 misc: cumsum / cumprod / div / gather / scatter ----

///|
/// SIMD prefix sum (Hillis-Steele on i32x4 chunks + cross-chunk carry).
pub fn SimdBuffer::cumsum(arr : SimdBuffer, out : SimdBuffer) -> Unit {
  if arr.len != out.len {
    abort("SimdBuffer.cumsum: length mismatch")
  }
  if arr.len == 0 {
    return
  }
  buf_cumsum_i32(arr.addr, out.addr, arr.len)
}

///|
pub fn SimdBuffer::cumprod(arr : SimdBuffer, out : SimdBuffer) -> Unit {
  if arr.len != out.len {
    abort("SimdBuffer.cumprod: length mismatch")
  }
  if arr.len == 0 {
    return
  }
  buf_cumprod_i32(arr.addr, out.addr, arr.len)
}

///|
/// SIMD i32 div via f64x2 widening (wasm has no integer SIMD div).
/// Caller is responsible for ensuring no division by zero — the SIMD path
/// converts that to a NaN-trunc-to-zero result, not a trap.
pub fn SimdBuffer::div(
  a : SimdBuffer,
  b : SimdBuffer,
  out : SimdBuffer,
) -> Unit {
  i32_bin_check(a, b, out, "div")
  if a.len == 0 {
    return
  }
  buf_div_i32(a.addr, b.addr, out.addr, a.len)
}

///|
/// `out[i] = arr[indices[i]]`. Gather via SIMD store + scalar lookups
/// (wasm SIMD has no real gather instruction).
pub fn SimdBuffer::gather(
  arr : SimdBuffer,
  indices : SimdBuffer,
  out : SimdBuffer,
) -> Unit {
  if indices.len != out.len {
    abort("SimdBuffer.gather: indices/out length mismatch")
  }
  if indices.len == 0 {
    return
  }
  buf_gather_i32(arr.addr, indices.addr, out.addr, indices.len)
}

///|
/// `out[indices[i]] = vals[i]`. Scatter stays scalar — random writes can't
/// be vectorised without lock-step / conflict detection.
pub fn SimdBuffer::scatter(
  out : SimdBuffer,
  indices : SimdBuffer,
  vals : SimdBuffer,
) -> Unit {
  if indices.len != vals.len {
    abort("SimdBuffer.scatter: indices/vals length mismatch")
  }
  for i in 0..<indices.len {
    let idx = buf_load_i32(indices.addr + i * 4)
    if idx < 0 || idx >= out.len {
      abort("SimdBuffer.scatter: index out of bounds")
    }
    buf_store_i32(out.addr + idx * 4, buf_load_i32(vals.addr + i * 4))
  }
}

// ---- inline-WAT primitives ----

///|
fn buf_cumsum_i32(arr : Int, out : Int, len : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (local i32) (local i32) (local i32) (local v128) (local v128) (local v128) (local v128) i32.const 0 i32x4.splat local.set 6 i32.const 0 local.set 5 local.get 2 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 0 local.get 3 i32.const 2 i32.shl i32.add v128.load local.set 7 local.get 6 local.get 7 i8x16.shuffle 0 1 2 3 16 17 18 19 20 21 22 23 24 25 26 27 local.get 7 i32x4.add local.set 8 local.get 6 local.get 8 i8x16.shuffle 0 1 2 3 4 5 6 7 16 17 18 19 20 21 22 23 local.get 8 i32x4.add local.set 9 local.get 5 i32x4.splat local.get 9 i32x4.add local.set 7 local.get 1 local.get 3 i32.const 2 i32.shl i32.add local.get 7 v128.store local.get 7 i32x4.extract_lane 3 local.set 5 local.get 3 i32.const 4 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 3 i32.const 2 i32.shl i32.add i32.load i32.add local.set 5 local.get 1 local.get 3 i32.const 2 i32.shl i32.add local.get 5 i32.store local.get 3 i32.const 1 i32.add local.set 3 br 0 end end)

///|
fn buf_cumprod_i32(arr : Int, out : Int, len : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (local i32) (local i32) (local i32) (local v128) (local v128) (local v128) (local v128) i32.const 1 i32x4.splat local.set 6 i32.const 1 local.set 5 local.get 2 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 0 local.get 3 i32.const 2 i32.shl i32.add v128.load local.set 7 local.get 6 local.get 7 i8x16.shuffle 0 1 2 3 16 17 18 19 20 21 22 23 24 25 26 27 local.get 7 i32x4.mul local.set 8 local.get 6 local.get 8 i8x16.shuffle 0 1 2 3 4 5 6 7 16 17 18 19 20 21 22 23 local.get 8 i32x4.mul local.set 9 local.get 5 i32x4.splat local.get 9 i32x4.mul local.set 7 local.get 1 local.get 3 i32.const 2 i32.shl i32.add local.get 7 v128.store local.get 7 i32x4.extract_lane 3 local.set 5 local.get 3 i32.const 4 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 3 i32.const 2 i32.shl i32.add i32.load i32.mul local.set 5 local.get 1 local.get 3 i32.const 2 i32.shl i32.add local.get 5 i32.store local.get 3 i32.const 1 i32.add local.set 3 br 0 end end)

///|
fn buf_div_i32(a : Int, b : Int, out : Int, len : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.set 6 local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load local.set 7 local.get 6 f64x2.convert_low_i32x4_s local.set 8 local.get 7 f64x2.convert_low_i32x4_s local.set 9 local.get 8 local.get 9 f64x2.div i32x4.trunc_sat_f64x2_s_zero local.set 10 local.get 6 local.get 6 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 f64x2.convert_low_i32x4_s local.set 8 local.get 7 local.get 7 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 f64x2.convert_low_i32x4_s local.set 9 local.get 8 local.get 9 f64x2.div i32x4.trunc_sat_f64x2_s_zero local.set 11 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 10 local.get 11 i8x16.shuffle 0 1 2 3 4 5 6 7 16 17 18 19 20 21 22 23 v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add i32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add i32.load i32.div_s i32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
fn buf_gather_i32(arr : Int, indices : Int, out : Int, n : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add i32.const 0 i32x4.splat local.get 0 local.get 1 local.get 4 i32.const 2 i32.shl i32.add i32.load i32.const 2 i32.shl i32.add i32.load i32x4.replace_lane 0 local.get 0 local.get 1 local.get 4 i32.const 1 i32.add i32.const 2 i32.shl i32.add i32.load i32.const 2 i32.shl i32.add i32.load i32x4.replace_lane 1 local.get 0 local.get 1 local.get 4 i32.const 2 i32.add i32.const 2 i32.shl i32.add i32.load i32.const 2 i32.shl i32.add i32.load i32x4.replace_lane 2 local.get 0 local.get 1 local.get 4 i32.const 3 i32.add i32.const 2 i32.shl i32.add i32.load i32.const 2 i32.shl i32.add i32.load i32x4.replace_lane 3 v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 1 local.get 4 i32.const 2 i32.shl i32.add i32.load i32.const 2 i32.shl i32.add i32.load i32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)
