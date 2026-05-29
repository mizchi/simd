// Scalar i32 reference impls. Used by SimdBuffer (i32) on native + js,
// and as a fallback by SimdBuffer's wasm path for any op not yet vectorised.

///|
/// Sum all elements of an integer array (scalar).
pub fn scalar_sum(arr : FixedArray[Int]) -> Int {
  let len = arr.length()
  for i = 0, acc = 0; i < len; {
    continue i + 1, acc + arr[i]
  } nobreak {
    acc
  }
}

///|
/// Dot product of two integer arrays (scalar).
pub fn scalar_dot_product(a : FixedArray[Int], b : FixedArray[Int]) -> Int {
  let len = a.length()
  for i = 0, acc = 0; i < len; {
    continue i + 1, acc + a[i] * b[i]
  } nobreak {
    acc
  }
}

///|
/// Element-wise addition of two arrays, storing result in out (scalar).
pub fn scalar_add(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  for i in 0..<len {
    out[i] = a[i] + b[i]
  }
}

///|
/// Min of an integer array (scalar). Returns 0 if empty.
pub fn scalar_min(arr : FixedArray[Int]) -> Int {
  let len = arr.length()
  if len == 0 {
    return 0
  }
  for i = 1, m = arr[0]; i < len; {
    continue i + 1, if arr[i] < m { arr[i] } else { m }
  } nobreak {
    m
  }
}

///|
/// Max of an integer array (scalar). Returns 0 if empty.
pub fn scalar_max(arr : FixedArray[Int]) -> Int {
  let len = arr.length()
  if len == 0 {
    return 0
  }
  for i = 1, m = arr[0]; i < len; {
    continue i + 1, if arr[i] > m { arr[i] } else { m }
  } nobreak {
    m
  }
}

///|
/// Index of the first minimum element (scalar). Returns -1 if empty.
pub fn scalar_argmin(arr : FixedArray[Int]) -> Int {
  let len = arr.length()
  if len == 0 {
    return -1
  }
  let mut idx = 0
  let mut best = arr[0]
  for i in 1..<len {
    if arr[i] < best {
      best = arr[i]
      idx = i
    }
  }
  idx
}

///|
/// Index of the first maximum element (scalar). Returns -1 if empty.
pub fn scalar_argmax(arr : FixedArray[Int]) -> Int {
  let len = arr.length()
  if len == 0 {
    return -1
  }
  let mut idx = 0
  let mut best = arr[0]
  for i in 1..<len {
    if arr[i] > best {
      best = arr[i]
      idx = i
    }
  }
  idx
}

///|
/// out[i] = k * a[i] + b[i] (scalar SAXPY-style for Int).
pub fn scalar_saxpy(
  k : Int,
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  for i in 0..<len {
    out[i] = k * a[i] + b[i]
  }
}

///|
pub fn scalar_sub(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  for i in 0..<len {
    out[i] = a[i] - b[i]
  }
}

///|
pub fn scalar_mul(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  for i in 0..<len {
    out[i] = a[i] * b[i]
  }
}

///|
pub fn scalar_neg(a : FixedArray[Int], out : FixedArray[Int]) -> Unit {
  let len = a.length()
  for i in 0..<len {
    out[i] = -a[i]
  }
}

///|
pub fn scalar_abs(a : FixedArray[Int], out : FixedArray[Int]) -> Unit {
  let len = a.length()
  for i in 0..<len {
    out[i] = if a[i] < 0 { -a[i] } else { a[i] }
  }
}

///|
/// Element-wise min: out[i] = min(a[i], b[i]).
pub fn scalar_min_elem(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  for i in 0..<len {
    out[i] = if a[i] < b[i] { a[i] } else { b[i] }
  }
}

///|
pub fn scalar_max_elem(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  for i in 0..<len {
    out[i] = if a[i] > b[i] { a[i] } else { b[i] }
  }
}

///|
/// Element-wise eq → mask (-1 if a[i] == b[i] else 0).
pub fn scalar_eq(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  for i in 0..<len {
    out[i] = if a[i] == b[i] { -1 } else { 0 }
  }
}

///|
pub fn scalar_lt(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  for i in 0..<len {
    out[i] = if a[i] < b[i] { -1 } else { 0 }
  }
}

///|
pub fn scalar_gt(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  for i in 0..<len {
    out[i] = if a[i] > b[i] { -1 } else { 0 }
  }
}

///|
/// out[i] = mask[i] != 0 ? a[i] : b[i].
pub fn scalar_where(
  mask : FixedArray[Int],
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = mask.length()
  for i in 0..<len {
    out[i] = if mask[i] != 0 { a[i] } else { b[i] }
  }
}

///|
pub fn scalar_div(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  for i in 0..<len {
    out[i] = a[i] / b[i]
  }
}

///|
/// out[i] = arr[indices[i]] (numpy fancy indexing).
pub fn scalar_gather(
  arr : FixedArray[Int],
  indices : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let n = indices.length()
  for i in 0..<n {
    out[i] = arr[indices[i]]
  }
}

///|
/// out[indices[i]] = vals[i].
pub fn scalar_scatter(
  out : FixedArray[Int],
  indices : FixedArray[Int],
  vals : FixedArray[Int],
) -> Unit {
  let n = indices.length()
  for i in 0..<n {
    out[indices[i]] = vals[i]
  }
}

///|
pub fn scalar_prod(arr : FixedArray[Int]) -> Int {
  let len = arr.length()
  if len == 0 {
    return 1
  }
  let mut acc = 1
  for i in 0..<len {
    acc = acc * arr[i]
  }
  acc
}

///|
pub fn scalar_count_nonzero(arr : FixedArray[Int]) -> Int {
  let len = arr.length()
  let mut c = 0
  for i in 0..<len {
    if arr[i] != 0 {
      c = c + 1
    }
  }
  c
}

///|
pub fn scalar_all(arr : FixedArray[Int]) -> Bool {
  let len = arr.length()
  for i in 0..<len {
    if arr[i] == 0 {
      return false
    }
  }
  true
}

///|
pub fn scalar_any(arr : FixedArray[Int]) -> Bool {
  let len = arr.length()
  for i in 0..<len {
    if arr[i] != 0 {
      return true
    }
  }
  false
}

///|
/// Prefix sum: out[i] = sum(arr[0..=i]).
pub fn scalar_cumsum(arr : FixedArray[Int], out : FixedArray[Int]) -> Unit {
  let len = arr.length()
  let mut acc = 0
  for i in 0..<len {
    acc = acc + arr[i]
    out[i] = acc
  }
}

///|
pub fn scalar_cumprod(arr : FixedArray[Int], out : FixedArray[Int]) -> Unit {
  let len = arr.length()
  let mut acc = 1
  for i in 0..<len {
    acc = acc * arr[i]
    out[i] = acc
  }
}

///|
/// Fill every element of `arr` with `value` (scalar).
pub fn scalar_fill(arr : FixedArray[Int], value : Int) -> Unit {
  let len = arr.length()
  for i in 0..<len {
    arr[i] = value
  }
}

///|
/// Index of the first element equal to `needle`, or -1 if absent (scalar).
pub fn scalar_find(arr : FixedArray[Int], needle : Int) -> Int {
  let len = arr.length()
  for i in 0..<len {
    if arr[i] == needle {
      return i
    }
  }
  -1
}
