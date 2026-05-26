///|
/// f64 sibling of `SimdBuffer`. Same allocator + linear-memory layout, just
/// with 8-byte elements. Targets wasm + wasm-gc.
pub struct SimdBufferF64 {
  addr : Int // 16-byte aligned base address
  len : Int // number of f64 elements
}

///|
pub fn SimdBufferF64::make(len : Int) -> SimdBufferF64 {
  // 8 bytes per f64 + 16 trailing slack so the final v128.load (16 bytes
  // = 2 f64s) doesn't read past the allocated region.
  let bytes = len * 8 + 16
  let addr = buf_alloc(bytes)
  { addr, len }
}

///|
pub fn SimdBufferF64::length(self : SimdBufferF64) -> Int {
  self.len
}

///|
pub fn SimdBufferF64::get(self : SimdBufferF64, i : Int) -> Double {
  if i < 0 || i >= self.len {
    abort("SimdBufferF64.get: index out of bounds")
  }
  buf_load_f64(self.addr + i * 8)
}

///|
pub fn SimdBufferF64::set(self : SimdBufferF64, i : Int, v : Double) -> Unit {
  if i < 0 || i >= self.len {
    abort("SimdBufferF64.set: index out of bounds")
  }
  buf_store_f64(self.addr + i * 8, v)
}

///|
/// SIMD dot product (f64x2). Both buffers must have the same length.
pub fn SimdBufferF64::dot(a : SimdBufferF64, b : SimdBufferF64) -> Double {
  if a.len != b.len {
    abort("SimdBufferF64.dot: length mismatch")
  }
  if a.len == 0 {
    return 0.0
  }
  buf_dot_f64(a.addr, b.addr, a.len)
}

///|
/// SIMD sum (f64x2 accumulator + scalar tail).
pub fn SimdBufferF64::sum(self : SimdBufferF64) -> Double {
  if self.len == 0 {
    return 0.0
  }
  buf_sum_f64(self.addr, self.len)
}

///|
/// SIMD matmul. `a` is M×K, `b_t` is N×K (= transposed B), `c` is M×N.
/// Inner loop is `buf_dot_f64_range` over the K dimension (f64x2 SIMD).
pub fn SimdBufferF64::matmul(
  a : SimdBufferF64,
  b_t : SimdBufferF64,
  c : SimdBufferF64,
  m : Int,
  k : Int,
  n : Int,
) -> Unit {
  if m == 0 || n == 0 {
    return
  }
  if a.len < m * k || b_t.len < n * k || c.len < m * n {
    abort("SimdBufferF64.matmul: buffer size mismatch")
  }
  for i in 0..<m {
    let a_off = i * k
    for j in 0..<n {
      let v = buf_dot_f64_range(a.addr, a_off, b_t.addr, j * k, k)
      buf_store_f64(c.addr + (i * n + j) * 8, v)
    }
  }
}

///|
/// SIMD gemv. `a` is M×N, `x` is length N, `y` is length M (output).
pub fn SimdBufferF64::gemv(
  a : SimdBufferF64,
  x : SimdBufferF64,
  y : SimdBufferF64,
  m : Int,
  n : Int,
) -> Unit {
  if m == 0 || n == 0 {
    return
  }
  if a.len < m * n || x.len < n || y.len < m {
    abort("SimdBufferF64.gemv: buffer size mismatch")
  }
  for i in 0..<m {
    let v = buf_dot_f64_range(a.addr, i * n, x.addr, 0, n)
    buf_store_f64(y.addr + i * 8, v)
  }
}

// ---- raw memory + SIMD primitives (identical bodies to simd_wasm_f64.mbt
//      but accepting raw i32 addresses instead of FixedArray params) ----

///|
fn buf_load_f64(addr : Int) -> Double =
  #|(func (param i32) (result f64) local.get 0 f64.load)

///|
fn buf_store_f64(addr : Int, v : Double) -> Unit =
  #|(func (param i32) (param f64) local.get 0 local.get 1 f64.store)

///|
/// `dot(a[a_off..a_off+k], b[b_off..b_off+k])` as f64. Same inline-WAT
/// body as `dot_f64_range` in `simd_wasm_f64.mbt`; both target i32-address
/// params at the wasm level.
fn buf_dot_f64_range(
  a_addr : Int,
  a_off : Int,
  b_addr : Int,
  b_off : Int,
  k : Int,
) -> Double =
  #|(func (param i32) (param i32) (param i32) (param i32) (param i32) (result f64) (local i32) (local i32) (local f64) (local v128) i32.const 0 f64.convert_i32_s f64x2.splat local.set 8 local.get 4 i32.const 1 i32.shr_s i32.const 1 i32.shl local.set 6 block loop local.get 5 local.get 6 i32.ge_s br_if 1 local.get 8 local.get 0 local.get 1 local.get 5 i32.add i32.const 3 i32.shl i32.add v128.load local.get 2 local.get 3 local.get 5 i32.add i32.const 3 i32.shl i32.add v128.load f64x2.mul f64x2.add local.set 8 local.get 5 i32.const 2 i32.add local.set 5 br 0 end end local.get 8 f64x2.extract_lane 0 local.get 8 f64x2.extract_lane 1 f64.add local.set 7 block loop local.get 5 local.get 4 i32.ge_s br_if 1 local.get 7 local.get 0 local.get 1 local.get 5 i32.add i32.const 3 i32.shl i32.add f64.load local.get 2 local.get 3 local.get 5 i32.add i32.const 3 i32.shl i32.add f64.load f64.mul f64.add local.set 7 local.get 5 i32.const 1 i32.add local.set 5 br 0 end end local.get 7)

///|
fn buf_sum_f64(addr : Int, len : Int) -> Double =
  #|(func (param i32) (param i32) (result f64) (local i32) (local i32) (local f64) (local v128) i32.const 0 f64.convert_i32_s f64x2.splat local.set 5 local.get 1 i32.const 1 i32.shr_s i32.const 1 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 2 i32.const 3 i32.shl i32.add v128.load f64x2.add local.set 5 local.get 2 i32.const 2 i32.add local.set 2 br 0 end end local.get 5 f64x2.extract_lane 0 local.get 5 f64x2.extract_lane 1 f64.add local.set 4 block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.const 3 i32.shl i32.add f64.load f64.add local.set 4 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end local.get 4)

///|
fn buf_dot_f64(a_addr : Int, b_addr : Int, len : Int) -> Double =
  #|(func (param i32) (param i32) (param i32) (result f64) (local i32) (local i32) (local f64) (local v128) i32.const 0 f64.convert_i32_s f64x2.splat local.set 6 local.get 2 i32.const 1 i32.shr_s i32.const 1 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 6 local.get 0 local.get 3 i32.const 3 i32.shl i32.add v128.load local.get 1 local.get 3 i32.const 3 i32.shl i32.add v128.load f64x2.mul f64x2.add local.set 6 local.get 3 i32.const 2 i32.add local.set 3 br 0 end end local.get 6 f64x2.extract_lane 0 local.get 6 f64x2.extract_lane 1 f64.add local.set 5 block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 3 i32.const 3 i32.shl i32.add f64.load local.get 1 local.get 3 i32.const 3 i32.shl i32.add f64.load f64.mul f64.add local.set 5 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end local.get 5)

// ---- element-wise ----

///|
pub fn SimdBufferF64::add(
  a : SimdBufferF64,
  b : SimdBufferF64,
  out : SimdBufferF64,
) -> Unit {
  f64_bin_check(a, b, out, "add")
  if a.len == 0 {
    return
  }
  buf_add_f64(a.addr, b.addr, out.addr, a.len)
}

///|
pub fn SimdBufferF64::sub(
  a : SimdBufferF64,
  b : SimdBufferF64,
  out : SimdBufferF64,
) -> Unit {
  f64_bin_check(a, b, out, "sub")
  if a.len == 0 {
    return
  }
  buf_sub_f64(a.addr, b.addr, out.addr, a.len)
}

///|
pub fn SimdBufferF64::mul(
  a : SimdBufferF64,
  b : SimdBufferF64,
  out : SimdBufferF64,
) -> Unit {
  f64_bin_check(a, b, out, "mul")
  if a.len == 0 {
    return
  }
  buf_mul_f64(a.addr, b.addr, out.addr, a.len)
}

///|
pub fn SimdBufferF64::div(
  a : SimdBufferF64,
  b : SimdBufferF64,
  out : SimdBufferF64,
) -> Unit {
  f64_bin_check(a, b, out, "div")
  if a.len == 0 {
    return
  }
  buf_div_f64(a.addr, b.addr, out.addr, a.len)
}

///|
pub fn SimdBufferF64::sqrt(a : SimdBufferF64, out : SimdBufferF64) -> Unit {
  if a.len != out.len {
    abort("SimdBufferF64.sqrt: length mismatch")
  }
  if a.len == 0 {
    return
  }
  buf_sqrt_f64(a.addr, out.addr, a.len)
}

///|
pub fn SimdBufferF64::min_elem(
  a : SimdBufferF64,
  b : SimdBufferF64,
  out : SimdBufferF64,
) -> Unit {
  f64_bin_check(a, b, out, "min_elem")
  if a.len == 0 {
    return
  }
  buf_min_elem_f64(a.addr, b.addr, out.addr, a.len)
}

///|
pub fn SimdBufferF64::max_elem(
  a : SimdBufferF64,
  b : SimdBufferF64,
  out : SimdBufferF64,
) -> Unit {
  f64_bin_check(a, b, out, "max_elem")
  if a.len == 0 {
    return
  }
  buf_max_elem_f64(a.addr, b.addr, out.addr, a.len)
}

///|
pub fn SimdBufferF64::mean(self : SimdBufferF64) -> Double {
  if self.len == 0 {
    return 0.0
  }
  self.sum() / self.len.to_double()
}

///|
pub fn SimdBufferF64::variance(self : SimdBufferF64) -> Double {
  if self.len == 0 {
    return 0.0
  }
  let mu = self.mean()
  buf_var_f64_pass2(self.addr, self.len, mu) / self.len.to_double()
}

///|
fn f64_bin_check(
  a : SimdBufferF64,
  b : SimdBufferF64,
  out : SimdBufferF64,
  op : String,
) -> Unit {
  if a.len != b.len || a.len != out.len {
    abort("SimdBufferF64." + op + ": length mismatch")
  }
}

// ---- inline-WAT primitives (copied from simd_wasm_f64.mbt) ----

///|
fn buf_add_f64(a : Int, b : Int, out : Int, len : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 1 i32.shr_s i32.const 1 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 3 i32.shl i32.add local.get 0 local.get 4 i32.const 3 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 3 i32.shl i32.add v128.load f64x2.add v128.store local.get 4 i32.const 2 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 3 i32.shl i32.add local.get 0 local.get 4 i32.const 3 i32.shl i32.add f64.load local.get 1 local.get 4 i32.const 3 i32.shl i32.add f64.load f64.add f64.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
fn buf_sub_f64(a : Int, b : Int, out : Int, len : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 1 i32.shr_s i32.const 1 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 3 i32.shl i32.add local.get 0 local.get 4 i32.const 3 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 3 i32.shl i32.add v128.load f64x2.sub v128.store local.get 4 i32.const 2 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 3 i32.shl i32.add local.get 0 local.get 4 i32.const 3 i32.shl i32.add f64.load local.get 1 local.get 4 i32.const 3 i32.shl i32.add f64.load f64.sub f64.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
fn buf_mul_f64(a : Int, b : Int, out : Int, len : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 1 i32.shr_s i32.const 1 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 3 i32.shl i32.add local.get 0 local.get 4 i32.const 3 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 3 i32.shl i32.add v128.load f64x2.mul v128.store local.get 4 i32.const 2 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 3 i32.shl i32.add local.get 0 local.get 4 i32.const 3 i32.shl i32.add f64.load local.get 1 local.get 4 i32.const 3 i32.shl i32.add f64.load f64.mul f64.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
fn buf_div_f64(a : Int, b : Int, out : Int, len : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 1 i32.shr_s i32.const 1 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 3 i32.shl i32.add local.get 0 local.get 4 i32.const 3 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 3 i32.shl i32.add v128.load f64x2.div v128.store local.get 4 i32.const 2 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 3 i32.shl i32.add local.get 0 local.get 4 i32.const 3 i32.shl i32.add f64.load local.get 1 local.get 4 i32.const 3 i32.shl i32.add f64.load f64.div f64.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
fn buf_sqrt_f64(a : Int, out : Int, len : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (local i32) (local i32) local.get 2 i32.const 1 i32.shr_s i32.const 1 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 1 local.get 3 i32.const 3 i32.shl i32.add local.get 0 local.get 3 i32.const 3 i32.shl i32.add v128.load f64x2.sqrt v128.store local.get 3 i32.const 2 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 1 local.get 3 i32.const 3 i32.shl i32.add local.get 0 local.get 3 i32.const 3 i32.shl i32.add f64.load f64.sqrt f64.store local.get 3 i32.const 1 i32.add local.set 3 br 0 end end)

///|
fn buf_min_elem_f64(a : Int, b : Int, out : Int, len : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 1 i32.shr_s i32.const 1 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 3 i32.shl i32.add local.get 0 local.get 4 i32.const 3 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 3 i32.shl i32.add v128.load f64x2.min v128.store local.get 4 i32.const 2 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 3 i32.shl i32.add local.get 0 local.get 4 i32.const 3 i32.shl i32.add f64.load local.get 1 local.get 4 i32.const 3 i32.shl i32.add f64.load local.get 0 local.get 4 i32.const 3 i32.shl i32.add f64.load local.get 1 local.get 4 i32.const 3 i32.shl i32.add f64.load f64.lt select f64.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
fn buf_max_elem_f64(a : Int, b : Int, out : Int, len : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 1 i32.shr_s i32.const 1 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 3 i32.shl i32.add local.get 0 local.get 4 i32.const 3 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 3 i32.shl i32.add v128.load f64x2.max v128.store local.get 4 i32.const 2 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 3 i32.shl i32.add local.get 0 local.get 4 i32.const 3 i32.shl i32.add f64.load local.get 1 local.get 4 i32.const 3 i32.shl i32.add f64.load local.get 0 local.get 4 i32.const 3 i32.shl i32.add f64.load local.get 1 local.get 4 i32.const 3 i32.shl i32.add f64.load f64.gt select f64.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
fn buf_var_f64_pass2(addr : Int, len : Int, mu : Double) -> Double =
  #|(func (param i32) (param i32) (param f64) (result f64) (local i32) (local i32) (local f64) (local f64) (local v128) (local v128) (local v128) local.get 2 f64x2.splat local.set 7 i32.const 0 f64.convert_i32_s f64x2.splat local.set 8 local.get 1 i32.const 1 i32.shr_s i32.const 1 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 8 local.get 0 local.get 3 i32.const 3 i32.shl i32.add v128.load local.get 7 f64x2.sub local.tee 9 local.get 9 f64x2.mul f64x2.add local.set 8 local.get 3 i32.const 2 i32.add local.set 3 br 0 end end local.get 8 f64x2.extract_lane 0 local.get 8 f64x2.extract_lane 1 f64.add local.set 6 block loop local.get 3 local.get 1 i32.ge_s br_if 1 local.get 6 local.get 0 local.get 3 i32.const 3 i32.shl i32.add f64.load local.get 2 f64.sub local.tee 5 local.get 5 f64.mul f64.add local.set 6 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end local.get 6)

// ---- transpose ----

///|
/// SIMD transpose for even M and N (2×2 blocks via i8x16.shuffle). Caller
/// must allocate `out` with at least M*N f64 capacity. For odd dimensions,
/// falls back to a scalar element copy.
pub fn SimdBufferF64::transpose(
  a : SimdBufferF64,
  out : SimdBufferF64,
  m : Int,
  n : Int,
) -> Unit {
  if m == 0 || n == 0 {
    return
  }
  if a.len < m * n || out.len < m * n {
    abort("SimdBufferF64.transpose: buffer too small")
  }
  if m % 2 == 0 && n % 2 == 0 {
    for i = 0; i < m; i = i + 2 {
      for j = 0; j < n; j = j + 2 {
        buf_transpose_2x2_block_f64(
          a.addr,
          i * n + j,
          out.addr,
          j * m + i,
          n,
          m,
        )
      }
    }
  } else {
    // Scalar fallback for odd dimensions.
    for i in 0..<m {
      for j in 0..<n {
        let v = buf_load_f64(a.addr + (i * n + j) * 8)
        buf_store_f64(out.addr + (j * m + i) * 8, v)
      }
    }
  }
}

///|
fn buf_transpose_2x2_block_f64(
  a : Int,
  in_off : Int,
  out : Int,
  out_off : Int,
  n : Int,
  m : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (param i32) (param i32) (local v128) (local v128) local.get 0 local.get 1 i32.const 3 i32.shl i32.add v128.load local.set 6 local.get 0 local.get 1 local.get 4 i32.add i32.const 3 i32.shl i32.add v128.load local.set 7 local.get 2 local.get 3 i32.const 3 i32.shl i32.add local.get 6 local.get 7 i8x16.shuffle 0 1 2 3 4 5 6 7 16 17 18 19 20 21 22 23 v128.store local.get 2 local.get 3 local.get 5 i32.add i32.const 3 i32.shl i32.add local.get 6 local.get 7 i8x16.shuffle 8 9 10 11 12 13 14 15 24 25 26 27 28 29 30 31 v128.store)
