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
/// Total number of 1-bits across a byte buffer (scalar popcount).
pub fn scalar_popcount_bytes(data : FixedArray[Byte]) -> Int {
  let len = data.length()
  let mut acc = 0
  for i in 0..<len {
    let mut v : UInt = data[i].to_int().reinterpret_as_uint()
    while v != 0 {
      acc = acc + (v & 1).reinterpret_as_int()
      v = v >> 1
    }
  }
  acc
}

///|
/// out[di..di+len] = src[si..si+len] (scalar byte copy).
pub fn scalar_memcpy_bytes(
  src : FixedArray[Byte],
  si : Int,
  dst : FixedArray[Byte],
  di : Int,
  len : Int,
) -> Unit {
  for k in 0..<len {
    dst[di + k] = src[si + k]
  }
}

///|
/// dst[off..off+len] = value (scalar byte fill).
pub fn scalar_memset_bytes(
  dst : FixedArray[Byte],
  off : Int,
  len : Int,
  value : Byte,
) -> Unit {
  for k in 0..<len {
    dst[off + k] = value
  }
}

///|
/// True iff a[0..len] == b[0..len] byte-wise (scalar).
pub fn scalar_equal_bytes(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  len : Int,
) -> Bool {
  for k in 0..<len {
    if a[k] != b[k] {
      return false
    }
  }
  true
}

///|
/// Index of first occurrence of `needle` in `data`, or -1 if not found.
pub fn scalar_find_byte(data : FixedArray[Byte], needle : Byte) -> Int {
  let len = data.length()
  for i in 0..<len {
    if data[i] == needle {
      return i
    }
  }
  -1
}

///|
/// Count occurrences of `needle` in `data`.
pub fn scalar_count_byte(data : FixedArray[Byte], needle : Byte) -> Int {
  let len = data.length()
  let mut c = 0
  for i in 0..<len {
    if data[i] == needle {
      c = c + 1
    }
  }
  c
}

// --- element-wise Int ops (B) ---

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

// --- reductions + scan (D) ---

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
pub fn scalar_mean_f64(arr : FixedArray[Double]) -> Double {
  let len = arr.length()
  if len == 0 {
    return 0.0
  }
  scalar_sum_f64(arr) / len.to_double()
}

///|
/// Population variance (1/n * sum (x - mu)^2).
pub fn scalar_var_f64(arr : FixedArray[Double]) -> Double {
  let len = arr.length()
  if len == 0 {
    return 0.0
  }
  let mu = scalar_mean_f64(arr)
  let mut acc = 0.0
  for i in 0..<len {
    let d = arr[i] - mu
    acc = acc + d * d
  }
  acc / len.to_double()
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

// --- linear algebra (C) ---

///|
/// matmul with B already in transposed form (row-major B^T).
/// A is M x K, B^T is N x K, C is M x N - all row-major.
pub fn scalar_matmul_f64(
  a : FixedArray[Double],
  b_t : FixedArray[Double],
  c : FixedArray[Double],
  m : Int,
  k : Int,
  n : Int,
) -> Unit {
  for i in 0..<m {
    for j in 0..<n {
      let mut acc = 0.0
      for kk in 0..<k {
        acc = acc + a[i * k + kk] * b_t[j * k + kk]
      }
      c[i * n + j] = acc
    }
  }
}

///|
/// gemv: y = A * x.  A is M x N row-major, x is N, y is M.
pub fn scalar_gemv_f64(
  a : FixedArray[Double],
  x : FixedArray[Double],
  y : FixedArray[Double],
  m : Int,
  n : Int,
) -> Unit {
  for i in 0..<m {
    let mut acc = 0.0
    for j in 0..<n {
      acc = acc + a[i * n + j] * x[j]
    }
    y[i] = acc
  }
}

///|
/// Transpose: out[j*m + i] = a[i*n + j].
pub fn scalar_transpose_f64(
  a : FixedArray[Double],
  out : FixedArray[Double],
  m : Int,
  n : Int,
) -> Unit {
  for i in 0..<m {
    for j in 0..<n {
      out[j * m + i] = a[i * n + j]
    }
  }
}

// --- Double (f64) ops ---

///|
pub fn scalar_add_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  let len = a.length()
  for i in 0..<len {
    out[i] = a[i] + b[i]
  }
}

///|
pub fn scalar_sub_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  let len = a.length()
  for i in 0..<len {
    out[i] = a[i] - b[i]
  }
}

///|
pub fn scalar_mul_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  let len = a.length()
  for i in 0..<len {
    out[i] = a[i] * b[i]
  }
}

///|
pub fn scalar_div_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  let len = a.length()
  for i in 0..<len {
    out[i] = a[i] / b[i]
  }
}

///|
pub fn scalar_sqrt_f64(
  a : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  let len = a.length()
  for i in 0..<len {
    out[i] = a[i].sqrt()
  }
}

///|
pub fn scalar_sum_f64(arr : FixedArray[Double]) -> Double {
  let len = arr.length()
  let mut acc = 0.0
  for i in 0..<len {
    acc = acc + arr[i]
  }
  acc
}

///|
pub fn scalar_dot_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
) -> Double {
  let len = a.length()
  let mut acc = 0.0
  for i in 0..<len {
    acc = acc + a[i] * b[i]
  }
  acc
}

///|
/// Element-wise min (out[i] = min(a[i], b[i])).
pub fn scalar_min_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  let len = a.length()
  for i in 0..<len {
    out[i] = a[i].min(b[i])
  }
}

///|
pub fn scalar_max_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
  out : FixedArray[Double],
) -> Unit {
  let len = a.length()
  for i in 0..<len {
    out[i] = a[i].max(b[i])
  }
}

///|
/// Adler-32 checksum (scalar).
pub fn scalar_adler32(data : FixedArray[Byte]) -> UInt {
  let mod_adler = 65521U
  let len = data.length()
  for i = 0, a = 1U, b = 0U; i < len; {
    let new_a = (a + data[i].to_int().reinterpret_as_uint()) % mod_adler
    let new_b = (b + new_a) % mod_adler
    continue i + 1, new_a, new_b
  } nobreak {
    (b << 16) | a
  }
}
