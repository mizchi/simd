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
fn add_f64_v128(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
  len : Int,
) -> Unit {
  let vec_len = floor_f64x2_len(len)
  for i = 0; i < vec_len; {
    fixed_f64x2_store(
      out,
      i,
      f64x2_add(fixed_f64x2_load(a, i), fixed_f64x2_load(b, i)),
    )
    continue i + 2
  }
  for i in vec_len..<len {
    out[i] = a[i] + b[i]
  }
}

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
fn sub_f64_v128(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
  len : Int,
) -> Unit {
  let vec_len = floor_f64x2_len(len)
  for i = 0; i < vec_len; {
    fixed_f64x2_store(
      out,
      i,
      f64x2_sub(fixed_f64x2_load(a, i), fixed_f64x2_load(b, i)),
    )
    continue i + 2
  }
  for i in vec_len..<len {
    out[i] = a[i] - b[i]
  }
}

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
fn mul_f64_v128(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
  len : Int,
) -> Unit {
  let vec_len = floor_f64x2_len(len)
  for i = 0; i < vec_len; {
    fixed_f64x2_store(
      out,
      i,
      f64x2_mul(fixed_f64x2_load(a, i), fixed_f64x2_load(b, i)),
    )
    continue i + 2
  }
  for i in vec_len..<len {
    out[i] = a[i] * b[i]
  }
}

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
fn div_f64_v128(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
  len : Int,
) -> Unit {
  let vec_len = floor_f64x2_len(len)
  for i = 0; i < vec_len; {
    fixed_f64x2_store(
      out,
      i,
      f64x2_div(fixed_f64x2_load(a, i), fixed_f64x2_load(b, i)),
    )
    continue i + 2
  }
  for i in vec_len..<len {
    out[i] = a[i] / b[i]
  }
}

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
fn sqrt_f64_v128(
  a : FixedArray[Double],
  out : FixedArray[Double],
  len : Int,
) -> Unit {
  let vec_len = floor_f64x2_len(len)
  for i = 0; i < vec_len; {
    fixed_f64x2_store(out, i, f64x2_sqrt(fixed_f64x2_load(a, i)))
    continue i + 2
  }
  for i in vec_len..<len {
    out[i] = a[i].sqrt()
  }
}

///|
pub fn sqrt_f64(a : FixedArray[Double], out : FixedArray[Double]) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  sqrt_f64_v128(a, out, len)
}

///|
fn min_elem_f64_v128(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
  len : Int,
) -> Unit {
  let vec_len = floor_f64x2_len(len)
  for i = 0; i < vec_len; {
    fixed_f64x2_store(
      out,
      i,
      f64x2_min(fixed_f64x2_load(a, i), fixed_f64x2_load(b, i)),
    )
    continue i + 2
  }
  for i in vec_len..<len {
    out[i] = a[i].min(b[i])
  }
}

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
fn max_elem_f64_v128(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
  len : Int,
) -> Unit {
  let vec_len = floor_f64x2_len(len)
  for i = 0; i < vec_len; {
    fixed_f64x2_store(
      out,
      i,
      f64x2_max(fixed_f64x2_load(a, i), fixed_f64x2_load(b, i)),
    )
    continue i + 2
  }
  for i in vec_len..<len {
    out[i] = a[i].max(b[i])
  }
}

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
fn sum_f64_v128(arr : FixedArray[Double], len : Int) -> Double {
  let vec_len = floor_f64x2_len(len)
  let acc = for i = 0, acc = f64x2_zero(); i < vec_len; {
    continue i + 2, f64x2_add(acc, fixed_f64x2_load(arr, i))
  } nobreak {
    acc
  }
  for i in vec_len..<len; sum = f64x2_lane_sum(acc) {
    continue sum + arr[i]
  } nobreak {
    sum
  }
}

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
fn max_f64_v128(arr : FixedArray[Double], len : Int) -> Double {
  let vec_len = floor_f64x2_len(len)
  let acc = for i = 0, acc = f64x2_splat(arr[0]); i < vec_len; {
    continue i + 2, f64x2_max(acc, fixed_f64x2_load(arr, i))
  } nobreak {
    acc
  }
  for i in vec_len..<len; best = f64x2_lane_max(acc) {
    continue best.max(arr[i])
  } nobreak {
    best
  }
}

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
fn min_f64_v128(arr : FixedArray[Double], len : Int) -> Double {
  let vec_len = floor_f64x2_len(len)
  let acc = for i = 0, acc = f64x2_splat(arr[0]); i < vec_len; {
    continue i + 2, f64x2_min(acc, fixed_f64x2_load(arr, i))
  } nobreak {
    acc
  }
  for i in vec_len..<len; best = f64x2_lane_min(acc) {
    continue best.min(arr[i])
  } nobreak {
    best
  }
}

///|
pub fn min_f64(arr : FixedArray[Double]) -> Double {
  let len = arr.length()
  if len == 0 {
    return 0.0
  }
  min_f64_v128(arr, len)
}

///|
fn simd_dot_f64_v128(
  a : FixedArray[Double],
  b : FixedArray[Double],
  len : Int,
) -> Double {
  let vec_len = floor_f64x2_len(len)
  let acc = for i = 0, acc = f64x2_zero(); i < vec_len; {
    let product = f64x2_mul(fixed_f64x2_load(a, i), fixed_f64x2_load(b, i))
    continue i + 2, f64x2_add(acc, product)
  } nobreak {
    acc
  }
  for i in vec_len..<len; sum = f64x2_lane_sum(acc) {
    continue sum + a[i] * b[i]
  } nobreak {
    sum
  }
}

///|
pub fn dot_f64(a : FixedArray[Double], b : FixedArray[Double]) -> Double {
  let len = a.length()
  if len == 0 {
    return 0.0
  }
  simd_dot_f64_v128(a, b, len)
}
