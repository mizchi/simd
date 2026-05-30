///|
pub fn mean_f64(arr : FixedArray[Double]) -> Double {
  let len = arr.length()
  if len == 0 {
    return 0.0
  }
  sum_f64(arr) / len.to_double()
}

///|
/// 2nd pass for variance: sum((arr[i] - mu)^2) using f64x2 fused-mul accumulator.
#borrow(arr)
fn var_f64_pass2(arr : FixedArray[Double], len : Int, mu : Double) -> Double =
  #|(func (param i32) (param i32) (param f64) (result f64) (local i32) (local i32) (local f64) (local f64) (local v128) (local v128) (local v128) local.get 2 f64x2.splat local.set 7 i32.const 0 f64.convert_i32_s f64x2.splat local.set 8 local.get 1 i32.const 1 i32.shr_s i32.const 1 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 8 local.get 0 local.get 3 i32.const 3 i32.shl i32.add v128.load local.get 7 f64x2.sub local.tee 9 local.get 9 f64x2.mul f64x2.add local.set 8 local.get 3 i32.const 2 i32.add local.set 3 br 0 end end local.get 8 f64x2.extract_lane 0 local.get 8 f64x2.extract_lane 1 f64.add local.set 6 block loop local.get 3 local.get 1 i32.ge_s br_if 1 local.get 6 local.get 0 local.get 3 i32.const 3 i32.shl i32.add f64.load local.get 2 f64.sub local.tee 5 local.get 5 f64.mul f64.add local.set 6 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end local.get 6)

///|
pub fn var_f64(arr : FixedArray[Double]) -> Double {
  let len = arr.length()
  if len == 0 {
    return 0.0
  }
  let mu = mean_f64(arr)
  var_f64_pass2(arr, len, mu) / len.to_double()
}

///|
/// Compute dot(a[a_off..a_off+k], b[b_off..b_off+k]) as f64.
/// Used as the inner loop for matmul and gemv.
#borrow(a, b)
fn dot_f64_range(
  a : FixedArray[Double],
  a_off : Int,
  b : FixedArray[Double],
  b_off : Int,
  k : Int,
) -> Double =
  #|(func (param i32) (param i32) (param i32) (param i32) (param i32) (result f64) (local i32) (local i32) (local f64) (local v128) i32.const 0 f64.convert_i32_s f64x2.splat local.set 8 local.get 4 i32.const 1 i32.shr_s i32.const 1 i32.shl local.set 6 block loop local.get 5 local.get 6 i32.ge_s br_if 1 local.get 8 local.get 0 local.get 1 local.get 5 i32.add i32.const 3 i32.shl i32.add v128.load local.get 2 local.get 3 local.get 5 i32.add i32.const 3 i32.shl i32.add v128.load f64x2.mul f64x2.add local.set 8 local.get 5 i32.const 2 i32.add local.set 5 br 0 end end local.get 8 f64x2.extract_lane 0 local.get 8 f64x2.extract_lane 1 f64.add local.set 7 block loop local.get 5 local.get 4 i32.ge_s br_if 1 local.get 7 local.get 0 local.get 1 local.get 5 i32.add i32.const 3 i32.shl i32.add f64.load local.get 2 local.get 3 local.get 5 i32.add i32.const 3 i32.shl i32.add f64.load f64.mul f64.add local.set 7 local.get 5 i32.const 1 i32.add local.set 5 br 0 end end local.get 7)

///|
/// SIMD matmul (B in transposed form): A is M x K, B^T is N x K, C is M x N.
/// Inner-loop dot products use f64x2 via `dot_f64_range`.
pub fn matmul_f64(
  a : FixedArray[Double],
  b_t : FixedArray[Double],
  c : FixedArray[Double],
  m : Int,
  k : Int,
  n : Int,
) -> Unit {
  if m == 0 || n == 0 {
    return
  }
  for i in 0..<m {
    let a_off = i * k
    for j in 0..<n {
      c[i * n + j] = dot_f64_range(a, a_off, b_t, j * k, k)
    }
  }
}

///|
pub fn gemv_f64(
  a : FixedArray[Double],
  x : FixedArray[Double],
  y : FixedArray[Double],
  m : Int,
  n : Int,
) -> Unit {
  if m == 0 {
    return
  }
  for i in 0..<m {
    y[i] = dot_f64_range(a, i * n, x, 0, n)
  }
}

///|
/// Transpose one 2x2 block: input rows i, i+1 starting at element `in_off`,
/// output rows j, j+1 starting at element `out_off`. `n` and `m` are the
/// row strides of input and output respectively.
#borrow(a, out)
fn transpose_2x2_block_f64(
  a : FixedArray[Double],
  in_off : Int,
  out : FixedArray[Double],
  out_off : Int,
  n : Int,
  m : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (param i32) (param i32) (local v128) (local v128) local.get 0 local.get 1 i32.const 3 i32.shl i32.add v128.load local.set 6 local.get 0 local.get 1 local.get 4 i32.add i32.const 3 i32.shl i32.add v128.load local.set 7 local.get 2 local.get 3 i32.const 3 i32.shl i32.add local.get 6 local.get 7 i8x16.shuffle 0 1 2 3 4 5 6 7 16 17 18 19 20 21 22 23 v128.store local.get 2 local.get 3 local.get 5 i32.add i32.const 3 i32.shl i32.add local.get 6 local.get 7 i8x16.shuffle 8 9 10 11 12 13 14 15 24 25 26 27 28 29 30 31 v128.store)

///|
/// SIMD transpose: when both dimensions are even, walk 2x2 f64 blocks and
/// shuffle each into the output. Odd dims fall through to scalar.
pub fn transpose_f64(
  a : FixedArray[Double],
  out : FixedArray[Double],
  m : Int,
  n : Int,
) -> Unit {
  if m > 0 && n > 0 && m % 2 == 0 && n % 2 == 0 {
    for i = 0; i < m; i = i + 2 {
      for j = 0; j < n; j = j + 2 {
        transpose_2x2_block_f64(a, i * n + j, out, j * m + i, n, m)
      }
    }
  } else {
    @internal.scalar_transpose_f64(a, out, m, n)
  }
}

///|
#borrow(a, b, out)
fn add_f64_v128(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
  len : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 1 i32.shr_s i32.const 1 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 3 i32.shl i32.add local.get 0 local.get 4 i32.const 3 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 3 i32.shl i32.add v128.load f64x2.add v128.store local.get 4 i32.const 2 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 3 i32.shl i32.add local.get 0 local.get 4 i32.const 3 i32.shl i32.add f64.load local.get 1 local.get 4 i32.const 3 i32.shl i32.add f64.load f64.add f64.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
pub fn add_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  add_f64_v128(a, b, out, len)
}

///|
#borrow(a, b, out)
fn sub_f64_v128(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
  len : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 1 i32.shr_s i32.const 1 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 3 i32.shl i32.add local.get 0 local.get 4 i32.const 3 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 3 i32.shl i32.add v128.load f64x2.sub v128.store local.get 4 i32.const 2 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 3 i32.shl i32.add local.get 0 local.get 4 i32.const 3 i32.shl i32.add f64.load local.get 1 local.get 4 i32.const 3 i32.shl i32.add f64.load f64.sub f64.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
pub fn sub_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  sub_f64_v128(a, b, out, len)
}

///|
#borrow(a, b, out)
fn mul_f64_v128(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
  len : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 1 i32.shr_s i32.const 1 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 3 i32.shl i32.add local.get 0 local.get 4 i32.const 3 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 3 i32.shl i32.add v128.load f64x2.mul v128.store local.get 4 i32.const 2 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 3 i32.shl i32.add local.get 0 local.get 4 i32.const 3 i32.shl i32.add f64.load local.get 1 local.get 4 i32.const 3 i32.shl i32.add f64.load f64.mul f64.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
pub fn mul_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  mul_f64_v128(a, b, out, len)
}

///|
#borrow(a, b, out)
fn div_f64_v128(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
  len : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 1 i32.shr_s i32.const 1 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 3 i32.shl i32.add local.get 0 local.get 4 i32.const 3 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 3 i32.shl i32.add v128.load f64x2.div v128.store local.get 4 i32.const 2 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 3 i32.shl i32.add local.get 0 local.get 4 i32.const 3 i32.shl i32.add f64.load local.get 1 local.get 4 i32.const 3 i32.shl i32.add f64.load f64.div f64.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
pub fn div_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  div_f64_v128(a, b, out, len)
}

///|
#borrow(a, out)
fn sqrt_f64_v128(
  a : FixedArray[Double],
  out : FixedArray[Double],
  len : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (local i32) (local i32) local.get 2 i32.const 1 i32.shr_s i32.const 1 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 1 local.get 3 i32.const 3 i32.shl i32.add local.get 0 local.get 3 i32.const 3 i32.shl i32.add v128.load f64x2.sqrt v128.store local.get 3 i32.const 2 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 1 local.get 3 i32.const 3 i32.shl i32.add local.get 0 local.get 3 i32.const 3 i32.shl i32.add f64.load f64.sqrt f64.store local.get 3 i32.const 1 i32.add local.set 3 br 0 end end)

///|
pub fn sqrt_f64(a : FixedArray[Double], out : FixedArray[Double]) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  sqrt_f64_v128(a, out, len)
}

///|
#borrow(a, b, out)
fn min_elem_f64_v128(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
  len : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 1 i32.shr_s i32.const 1 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 3 i32.shl i32.add local.get 0 local.get 4 i32.const 3 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 3 i32.shl i32.add v128.load f64x2.min v128.store local.get 4 i32.const 2 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 3 i32.shl i32.add local.get 0 local.get 4 i32.const 3 i32.shl i32.add f64.load local.get 1 local.get 4 i32.const 3 i32.shl i32.add f64.load local.get 0 local.get 4 i32.const 3 i32.shl i32.add f64.load local.get 1 local.get 4 i32.const 3 i32.shl i32.add f64.load f64.lt select f64.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
pub fn min_elem_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  min_elem_f64_v128(a, b, out, len)
}

///|
#borrow(a, b, out)
fn max_elem_f64_v128(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
  len : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 1 i32.shr_s i32.const 1 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 3 i32.shl i32.add local.get 0 local.get 4 i32.const 3 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 3 i32.shl i32.add v128.load f64x2.max v128.store local.get 4 i32.const 2 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 3 i32.shl i32.add local.get 0 local.get 4 i32.const 3 i32.shl i32.add f64.load local.get 1 local.get 4 i32.const 3 i32.shl i32.add f64.load local.get 0 local.get 4 i32.const 3 i32.shl i32.add f64.load local.get 1 local.get 4 i32.const 3 i32.shl i32.add f64.load f64.gt select f64.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
pub fn max_elem_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  max_elem_f64_v128(a, b, out, len)
}

///|
/// sum reduce: splat(0.0) accumulator via i32.const 0 + f64.convert_i32_s,
/// since the Dwarfsm parser may not accept `f64.const` literals.
#borrow(arr)
fn sum_f64_v128(arr : FixedArray[Double], len : Int) -> Double =
  #|(func (param i32) (param i32) (result f64) (local i32) (local i32) (local f64) (local v128) i32.const 0 f64.convert_i32_s f64x2.splat local.set 5 local.get 1 i32.const 1 i32.shr_s i32.const 1 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 2 i32.const 3 i32.shl i32.add v128.load f64x2.add local.set 5 local.get 2 i32.const 2 i32.add local.set 2 br 0 end end local.get 5 f64x2.extract_lane 0 local.get 5 f64x2.extract_lane 1 f64.add local.set 4 block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.const 3 i32.shl i32.add f64.load f64.add local.set 4 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end local.get 4)

///|
pub fn sum_f64(arr : FixedArray[Double]) -> Double {
  let len = arr.length()
  if len == 0 {
    return 0.0
  }
  sum_f64_v128(arr, len)
}

///|
/// Horizontal max over f64x2 chunks (acc seeded with arr[0]), 2-lane reduce
/// via `select` (`f64.max` is rejected by the inline-WAT parser), scalar tail.
/// Caller guarantees `len >= 1`.
/// Locals: 2 = i, 3 = vend, 4 = acc_s (f64), 5 = acc (v128).
#borrow(arr)
fn max_f64_v128(arr : FixedArray[Double], len : Int) -> Double =
  #|(func (param i32) (param i32) (result f64) (local i32) (local i32) (local f64) (local v128) local.get 0 f64.load f64x2.splat local.set 5 local.get 1 i32.const 1 i32.shr_s i32.const 1 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 2 i32.const 3 i32.shl i32.add v128.load f64x2.max local.set 5 local.get 2 i32.const 2 i32.add local.set 2 br 0 end end local.get 5 f64x2.extract_lane 0 local.get 5 f64x2.extract_lane 1 local.get 5 f64x2.extract_lane 0 local.get 5 f64x2.extract_lane 1 f64.gt select local.set 4 block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.const 3 i32.shl i32.add f64.load local.get 4 local.get 0 local.get 2 i32.const 3 i32.shl i32.add f64.load f64.gt select local.set 4 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end local.get 4)

///|
pub fn max_f64(arr : FixedArray[Double]) -> Double {
  let len = arr.length()
  if len == 0 {
    return 0.0
  }
  max_f64_v128(arr, len)
}

///|
/// Horizontal min — same shape as `max_f64_v128` with `f64x2.min` / `f64.lt`.
/// Caller guarantees `len >= 1`.
#borrow(arr)
fn min_f64_v128(arr : FixedArray[Double], len : Int) -> Double =
  #|(func (param i32) (param i32) (result f64) (local i32) (local i32) (local f64) (local v128) local.get 0 f64.load f64x2.splat local.set 5 local.get 1 i32.const 1 i32.shr_s i32.const 1 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 2 i32.const 3 i32.shl i32.add v128.load f64x2.min local.set 5 local.get 2 i32.const 2 i32.add local.set 2 br 0 end end local.get 5 f64x2.extract_lane 0 local.get 5 f64x2.extract_lane 1 local.get 5 f64x2.extract_lane 0 local.get 5 f64x2.extract_lane 1 f64.lt select local.set 4 block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.const 3 i32.shl i32.add f64.load local.get 4 local.get 0 local.get 2 i32.const 3 i32.shl i32.add f64.load f64.lt select local.set 4 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end local.get 4)

///|
pub fn min_f64(arr : FixedArray[Double]) -> Double {
  let len = arr.length()
  if len == 0 {
    return 0.0
  }
  min_f64_v128(arr, len)
}

///|
#borrow(a, b)
fn simd_dot_f64_v128(
  a : FixedArray[Double],
  b : FixedArray[Double],
  len : Int,
) -> Double =
  #|(func (param i32) (param i32) (param i32) (result f64) (local i32) (local i32) (local f64) (local v128) i32.const 0 f64.convert_i32_s f64x2.splat local.set 6 local.get 2 i32.const 1 i32.shr_s i32.const 1 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 6 local.get 0 local.get 3 i32.const 3 i32.shl i32.add v128.load local.get 1 local.get 3 i32.const 3 i32.shl i32.add v128.load f64x2.mul f64x2.add local.set 6 local.get 3 i32.const 2 i32.add local.set 3 br 0 end end local.get 6 f64x2.extract_lane 0 local.get 6 f64x2.extract_lane 1 f64.add local.set 5 block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 3 i32.const 3 i32.shl i32.add f64.load local.get 1 local.get 3 i32.const 3 i32.shl i32.add f64.load f64.mul f64.add local.set 5 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end local.get 5)

///|
pub fn dot_f64(a : FixedArray[Double], b : FixedArray[Double]) -> Double {
  let len = a.length()
  if len == 0 {
    return 0.0
  }
  simd_dot_f64_v128(a, b, len)
}
