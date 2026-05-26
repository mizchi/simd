///|
/// f32 sibling of `SimdBuffer`. 4-byte aligned linear-memory storage.
/// Targets wasm + wasm-gc. Use when you want SIMD f32 arithmetic without the
/// byte-buffer encoding overhead the FixedArray-based `f32_buf` route pays.
pub struct SimdBufferF32 {
  addr : Int
  len : Int
}

///|
pub fn SimdBufferF32::make(len : Int) -> SimdBufferF32 {
  // 4 bytes per f32 + 16 trailing slack so the final v128.load stays in
  // bounds.
  let bytes = len * 4 + 16
  let addr = buf_alloc(bytes)
  { addr, len }
}

///|
pub fn SimdBufferF32::length(self : SimdBufferF32) -> Int {
  self.len
}

///|
pub fn SimdBufferF32::get(self : SimdBufferF32, i : Int) -> Float {
  if i < 0 || i >= self.len {
    abort("SimdBufferF32.get: index out of bounds")
  }
  buf_load_f32(self.addr + i * 4)
}

///|
pub fn SimdBufferF32::set(self : SimdBufferF32, i : Int, v : Float) -> Unit {
  if i < 0 || i >= self.len {
    abort("SimdBufferF32.set: index out of bounds")
  }
  buf_store_f32(self.addr + i * 4, v)
}

///|
pub fn SimdBufferF32::sum(self : SimdBufferF32) -> Float {
  if self.len == 0 {
    return (0.0 : Float)
  }
  buf_sum_f32(self.addr, self.len)
}

///|
pub fn SimdBufferF32::dot(a : SimdBufferF32, b : SimdBufferF32) -> Float {
  if a.len != b.len {
    abort("SimdBufferF32.dot: length mismatch")
  }
  if a.len == 0 {
    return (0.0 : Float)
  }
  buf_dot_f32(a.addr, b.addr, a.len)
}

///|
pub fn SimdBufferF32::add(
  a : SimdBufferF32,
  b : SimdBufferF32,
  out : SimdBufferF32,
) -> Unit {
  bin_check(a, b, out, "add")
  if a.len == 0 {
    return
  }
  buf_add_f32(a.addr, b.addr, out.addr, a.len)
}

///|
pub fn SimdBufferF32::sub(
  a : SimdBufferF32,
  b : SimdBufferF32,
  out : SimdBufferF32,
) -> Unit {
  bin_check(a, b, out, "sub")
  if a.len == 0 {
    return
  }
  buf_sub_f32(a.addr, b.addr, out.addr, a.len)
}

///|
pub fn SimdBufferF32::mul(
  a : SimdBufferF32,
  b : SimdBufferF32,
  out : SimdBufferF32,
) -> Unit {
  bin_check(a, b, out, "mul")
  if a.len == 0 {
    return
  }
  buf_mul_f32(a.addr, b.addr, out.addr, a.len)
}

///|
pub fn SimdBufferF32::div(
  a : SimdBufferF32,
  b : SimdBufferF32,
  out : SimdBufferF32,
) -> Unit {
  bin_check(a, b, out, "div")
  if a.len == 0 {
    return
  }
  buf_div_f32(a.addr, b.addr, out.addr, a.len)
}

///|
pub fn SimdBufferF32::sqrt(a : SimdBufferF32, out : SimdBufferF32) -> Unit {
  if a.len != out.len {
    abort("SimdBufferF32.sqrt: length mismatch")
  }
  if a.len == 0 {
    return
  }
  buf_sqrt_f32(a.addr, out.addr, a.len)
}

///|
pub fn SimdBufferF32::min_elem(
  a : SimdBufferF32,
  b : SimdBufferF32,
  out : SimdBufferF32,
) -> Unit {
  bin_check(a, b, out, "min_elem")
  if a.len == 0 {
    return
  }
  buf_min_elem_f32(a.addr, b.addr, out.addr, a.len)
}

///|
pub fn SimdBufferF32::max_elem(
  a : SimdBufferF32,
  b : SimdBufferF32,
  out : SimdBufferF32,
) -> Unit {
  bin_check(a, b, out, "max_elem")
  if a.len == 0 {
    return
  }
  buf_max_elem_f32(a.addr, b.addr, out.addr, a.len)
}

///|
fn bin_check(
  a : SimdBufferF32,
  b : SimdBufferF32,
  out : SimdBufferF32,
  op : String,
) -> Unit {
  if a.len != b.len || a.len != out.len {
    abort("SimdBufferF32." + op + ": length mismatch")
  }
}

// ---- raw memory + SIMD primitives ----
//
// Bodies copied verbatim from src/simd_wasm_f32.mbt. Both targets pass i32
// addresses at the wasm level, so the inline-WAT is identical; only the
// MoonBit-level parameter types differ (Int vs FixedArray[Byte]).

///|
fn buf_load_f32(addr : Int) -> Float =
  #|(func (param i32) (result f32) local.get 0 f32.load)

///|
fn buf_store_f32(addr : Int, v : Float) -> Unit =
  #|(func (param i32) (param f32) local.get 0 local.get 1 f32.store)

///|
fn buf_add_f32(a : Int, b : Int, out : Int, n : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load f32x4.add v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add f32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add f32.load f32.add f32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
fn buf_sub_f32(a : Int, b : Int, out : Int, n : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load f32x4.sub v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add f32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add f32.load f32.sub f32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
fn buf_mul_f32(a : Int, b : Int, out : Int, n : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load f32x4.mul v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add f32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add f32.load f32.mul f32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
fn buf_div_f32(a : Int, b : Int, out : Int, n : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load f32x4.div v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add f32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add f32.load f32.div f32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
fn buf_sqrt_f32(a : Int, out : Int, n : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (local i32) (local i32) local.get 2 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 1 local.get 3 i32.const 2 i32.shl i32.add local.get 0 local.get 3 i32.const 2 i32.shl i32.add v128.load f32x4.sqrt v128.store local.get 3 i32.const 4 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 1 local.get 3 i32.const 2 i32.shl i32.add local.get 0 local.get 3 i32.const 2 i32.shl i32.add f32.load f32.sqrt f32.store local.get 3 i32.const 1 i32.add local.set 3 br 0 end end)

///|
fn buf_min_elem_f32(a : Int, b : Int, out : Int, n : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load f32x4.min v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add f32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add f32.load local.get 0 local.get 4 i32.const 2 i32.shl i32.add f32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add f32.load f32.lt select f32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
fn buf_max_elem_f32(a : Int, b : Int, out : Int, n : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load f32x4.max v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add f32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add f32.load local.get 0 local.get 4 i32.const 2 i32.shl i32.add f32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add f32.load f32.gt select f32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
fn buf_sum_f32(arr : Int, n : Int) -> Float =
  #|(func (param i32) (param i32) (result f32) (local i32) (local i32) (local f32) (local v128) i32.const 0 f32.convert_i32_s f32x4.splat local.set 5 local.get 1 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 2 i32.const 2 i32.shl i32.add v128.load f32x4.add local.set 5 local.get 2 i32.const 4 i32.add local.set 2 br 0 end end local.get 5 f32x4.extract_lane 0 local.get 5 f32x4.extract_lane 1 f32.add local.get 5 f32x4.extract_lane 2 f32.add local.get 5 f32x4.extract_lane 3 f32.add local.set 4 block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.const 2 i32.shl i32.add f32.load f32.add local.set 4 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end local.get 4)

///|
fn buf_dot_f32(a : Int, b : Int, n : Int) -> Float =
  #|(func (param i32) (param i32) (param i32) (result f32) (local i32) (local i32) (local f32) (local v128) i32.const 0 f32.convert_i32_s f32x4.splat local.set 6 local.get 2 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 6 local.get 0 local.get 3 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 3 i32.const 2 i32.shl i32.add v128.load f32x4.mul f32x4.add local.set 6 local.get 3 i32.const 4 i32.add local.set 3 br 0 end end local.get 6 f32x4.extract_lane 0 local.get 6 f32x4.extract_lane 1 f32.add local.get 6 f32x4.extract_lane 2 f32.add local.get 6 f32x4.extract_lane 3 f32.add local.set 5 block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 3 i32.const 2 i32.shl i32.add f32.load local.get 1 local.get 3 i32.const 2 i32.shl i32.add f32.load f32.mul f32.add local.set 5 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end local.get 5)
