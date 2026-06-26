///|
fn add_f32_v128(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  let vec_len = floor_f32x4_len(n)
  for i = 0; i < vec_len; {
    fixed_f32x4_store(
      out,
      i,
      f32x4_add(fixed_f32x4_load(a, i), fixed_f32x4_load(b, i)),
    )
    continue i + 4
  }
  for i in vec_len..<n {
    f32_buf_set(out, i, f32_buf_get(a, i) + f32_buf_get(b, i))
  }
}

///|
pub fn add_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  if n == 0 {
    return
  }
  add_f32_v128(a, b, out, n)
}

///|
fn sub_f32_v128(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  let vec_len = floor_f32x4_len(n)
  for i = 0; i < vec_len; {
    fixed_f32x4_store(
      out,
      i,
      f32x4_sub(fixed_f32x4_load(a, i), fixed_f32x4_load(b, i)),
    )
    continue i + 4
  }
  for i in vec_len..<n {
    f32_buf_set(out, i, f32_buf_get(a, i) - f32_buf_get(b, i))
  }
}

///|
pub fn sub_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  if n == 0 {
    return
  }
  sub_f32_v128(a, b, out, n)
}

///|
fn div_f32_v128(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  let vec_len = floor_f32x4_len(n)
  for i = 0; i < vec_len; {
    fixed_f32x4_store(
      out,
      i,
      f32x4_div(fixed_f32x4_load(a, i), fixed_f32x4_load(b, i)),
    )
    continue i + 4
  }
  for i in vec_len..<n {
    f32_buf_set(out, i, f32_buf_get(a, i) / f32_buf_get(b, i))
  }
}

///|
pub fn div_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  if n == 0 {
    return
  }
  div_f32_v128(a, b, out, n)
}

///|
fn sqrt_f32_v128(a : FixedArray[Byte], out : FixedArray[Byte], n : Int) -> Unit {
  let vec_len = floor_f32x4_len(n)
  for i = 0; i < vec_len; {
    fixed_f32x4_store(out, i, f32x4_sqrt(fixed_f32x4_load(a, i)))
    continue i + 4
  }
  for i in vec_len..<n {
    f32_buf_set(out, i, f32_buf_get(a, i).sqrt())
  }
}

///|
pub fn sqrt_f32(a : FixedArray[Byte], out : FixedArray[Byte], n : Int) -> Unit {
  if n == 0 {
    return
  }
  sqrt_f32_v128(a, out, n)
}

///|
fn min_elem_f32_v128(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  let vec_len = floor_f32x4_len(n)
  for i = 0; i < vec_len; {
    fixed_f32x4_store(
      out,
      i,
      f32x4_min(fixed_f32x4_load(a, i), fixed_f32x4_load(b, i)),
    )
    continue i + 4
  }
  for i in vec_len..<n {
    let av = f32_buf_get(a, i)
    let bv = f32_buf_get(b, i)
    f32_buf_set(out, i, av.min(bv))
  }
}

///|
pub fn min_elem_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  if n == 0 {
    return
  }
  min_elem_f32_v128(a, b, out, n)
}

///|
fn max_elem_f32_v128(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  let vec_len = floor_f32x4_len(n)
  for i = 0; i < vec_len; {
    fixed_f32x4_store(
      out,
      i,
      f32x4_max(fixed_f32x4_load(a, i), fixed_f32x4_load(b, i)),
    )
    continue i + 4
  }
  for i in vec_len..<n {
    let av = f32_buf_get(a, i)
    let bv = f32_buf_get(b, i)
    f32_buf_set(out, i, av.max(bv))
  }
}

///|
pub fn max_elem_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  if n == 0 {
    return
  }
  max_elem_f32_v128(a, b, out, n)
}

///|
fn mul_f32_v128(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  let vec_len = floor_f32x4_len(n)
  for i = 0; i < vec_len; {
    fixed_f32x4_store(
      out,
      i,
      f32x4_mul(fixed_f32x4_load(a, i), fixed_f32x4_load(b, i)),
    )
    continue i + 4
  }
  for i in vec_len..<n {
    f32_buf_set(out, i, f32_buf_get(a, i) * f32_buf_get(b, i))
  }
}

///|
pub fn mul_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  if n == 0 {
    return
  }
  mul_f32_v128(a, b, out, n)
}

///|
fn sum_f32_v128(arr : FixedArray[Byte], n : Int) -> Float {
  let vec_len = floor_f32x4_len(n)
  let acc = for i = 0, acc = f32x4_zero(); i < vec_len; {
    continue i + 4, f32x4_add(acc, fixed_f32x4_load(arr, i))
  } nobreak {
    acc
  }
  for i in vec_len..<n; sum = f32x4_lane_sum(acc) {
    continue sum + f32_buf_get(arr, i)
  } nobreak {
    sum
  }
}

///|
pub fn sum_f32(arr : FixedArray[Byte], n : Int) -> Float {
  if n == 0 {
    return (0.0 : Float)
  }
  sum_f32_v128(arr, n)
}

///|
fn dot_f32_v128(a : FixedArray[Byte], b : FixedArray[Byte], n : Int) -> Float {
  let vec_len = floor_f32x4_len(n)
  let acc = for i = 0, acc = f32x4_zero(); i < vec_len; {
    let product = f32x4_mul(fixed_f32x4_load(a, i), fixed_f32x4_load(b, i))
    continue i + 4, f32x4_add(acc, product)
  } nobreak {
    acc
  }
  for i in vec_len..<n; sum = f32x4_lane_sum(acc) {
    continue sum + f32_buf_get(a, i) * f32_buf_get(b, i)
  } nobreak {
    sum
  }
}

///|
pub fn dot_f32(a : FixedArray[Byte], b : FixedArray[Byte], n : Int) -> Float {
  if n == 0 {
    return (0.0 : Float)
  }
  dot_f32_v128(a, b, n)
}
