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

// --- UTF-8 validation (I) ---

///|
/// Structural UTF-8 validation starting at `start`. Verifies that each
/// leading byte is followed by the right number of `10xxxxxx` continuation
/// bytes, and rejects 0x80..0xC1 leading bytes and 0xF5..0xFF entirely.
/// Does NOT reject overlong, surrogate, or > U+10FFFF codepoints — that's
/// a stricter pass than this one.
pub fn scalar_validate_utf8_from(
  data : FixedArray[Byte],
  start : Int,
) -> Bool {
  let n = data.length()
  let mut i = start
  while i < n {
    let b = data[i].to_int()
    if b < 0x80 {
      i = i + 1
    } else if b < 0xC2 {
      return false
    } else if b < 0xE0 {
      if i + 1 >= n {
        return false
      }
      if (data[i + 1].to_int() & 0xC0) != 0x80 {
        return false
      }
      i = i + 2
    } else if b < 0xF0 {
      if i + 2 >= n {
        return false
      }
      if (data[i + 1].to_int() & 0xC0) != 0x80 {
        return false
      }
      if (data[i + 2].to_int() & 0xC0) != 0x80 {
        return false
      }
      i = i + 3
    } else if b < 0xF5 {
      if i + 3 >= n {
        return false
      }
      if (data[i + 1].to_int() & 0xC0) != 0x80 {
        return false
      }
      if (data[i + 2].to_int() & 0xC0) != 0x80 {
        return false
      }
      if (data[i + 3].to_int() & 0xC0) != 0x80 {
        return false
      }
      i = i + 4
    } else {
      return false
    }
  }
  true
}

///|
pub fn scalar_validate_utf8(data : FixedArray[Byte]) -> Bool {
  scalar_validate_utf8_from(data, 0)
}

// --- string ops (H) ---

///|
/// True iff every byte in `data` is < 0x80 (pure ASCII).
pub fn scalar_is_ascii(data : FixedArray[Byte]) -> Bool {
  let len = data.length()
  for i in 0..<len {
    if (data[i].to_int() & 0x80) != 0 {
      return false
    }
  }
  true
}

///|
/// In-place ASCII A-Z → a-z conversion. Non-ASCII bytes pass through.
pub fn scalar_to_lower_ascii(data : FixedArray[Byte]) -> Unit {
  let len = data.length()
  for i in 0..<len {
    let c = data[i].to_int()
    if c >= 0x41 && c <= 0x5A {
      data[i] = (c + 0x20).to_byte()
    }
  }
}

///|
/// In-place ASCII a-z → A-Z conversion. Non-ASCII bytes pass through.
pub fn scalar_to_upper_ascii(data : FixedArray[Byte]) -> Unit {
  let len = data.length()
  for i in 0..<len {
    let c = data[i].to_int()
    if c >= 0x61 && c <= 0x7A {
      data[i] = (c - 0x20).to_byte()
    }
  }
}

// --- gather / scatter (G) ---

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

// --- sort (F) ---

///|
/// Sort the 4 elements arr[off..off+3] in place via a 5-comparator network.
pub fn scalar_sort4_int(arr : FixedArray[Int], off : Int) -> Unit {
  if off + 4 > arr.length() {
    return
  }
  let mut a = arr[off]
  let mut b = arr[off + 1]
  let mut c = arr[off + 2]
  let mut d = arr[off + 3]
  if a > b {
    let t = a
    a = b
    b = t
  }
  if c > d {
    let t = c
    c = d
    d = t
  }
  if a > c {
    let t = a
    a = c
    c = t
  }
  if b > d {
    let t = b
    b = d
    d = t
  }
  if b > c {
    let t = b
    b = c
    c = t
  }
  arr[off] = a
  arr[off + 1] = b
  arr[off + 2] = c
  arr[off + 3] = d
}

///|
/// 2-way merge of two sorted slices into `out` at `out_off`.
pub fn scalar_merge2_int(
  a : FixedArray[Int],
  a_off : Int,
  a_len : Int,
  b : FixedArray[Int],
  b_off : Int,
  b_len : Int,
  out : FixedArray[Int],
  out_off : Int,
) -> Unit {
  let mut i = 0
  let mut j = 0
  let mut k = 0
  while i < a_len && j < b_len {
    if a[a_off + i] <= b[b_off + j] {
      out[out_off + k] = a[a_off + i]
      i = i + 1
    } else {
      out[out_off + k] = b[b_off + j]
      j = j + 1
    }
    k = k + 1
  }
  while i < a_len {
    out[out_off + k] = a[a_off + i]
    i = i + 1
    k = k + 1
  }
  while j < b_len {
    out[out_off + k] = b[b_off + j]
    j = j + 1
    k = k + 1
  }
}

///|
/// Reference: 4+4 sorted → 8 sorted in place via scalar merge.
pub fn scalar_bitonic_merge8_int(arr : FixedArray[Int], off : Int) -> Unit {
  if off + 8 > arr.length() || off < 0 {
    return
  }
  let tmp = FixedArray::make(8, 0)
  scalar_merge2_int(arr, off, 4, arr, off + 4, 4, tmp, 0)
  for i in 0..<8 {
    arr[off + i] = tmp[i]
  }
}

///|
/// Reference: 16+16 sorted → 32 sorted in place via scalar merge.
pub fn scalar_bitonic_merge32_int(arr : FixedArray[Int], off : Int) -> Unit {
  if off + 32 > arr.length() || off < 0 {
    return
  }
  let tmp = FixedArray::make(32, 0)
  scalar_merge2_int(arr, off, 16, arr, off + 16, 16, tmp, 0)
  for i in 0..<32 {
    arr[off + i] = tmp[i]
  }
}

///|
/// Reference: 32+32 sorted → 64 sorted in place via scalar merge.
pub fn scalar_bitonic_merge64_int(arr : FixedArray[Int], off : Int) -> Unit {
  if off + 64 > arr.length() || off < 0 {
    return
  }
  let tmp = FixedArray::make(64, 0)
  scalar_merge2_int(arr, off, 32, arr, off + 32, 32, tmp, 0)
  for i in 0..<64 {
    arr[off + i] = tmp[i]
  }
}

///|
/// Reference: 8+8 sorted → 16 sorted in place via scalar merge.
pub fn scalar_bitonic_merge16_int(arr : FixedArray[Int], off : Int) -> Unit {
  if off + 16 > arr.length() || off < 0 {
    return
  }
  let tmp = FixedArray::make(16, 0)
  scalar_merge2_int(arr, off, 8, arr, off + 8, 8, tmp, 0)
  for i in 0..<16 {
    arr[off + i] = tmp[i]
  }
}

///|
/// Sort 16 elements via 4 sub-sorts of 4 plus a 4+4 → 8 + 4+4 → 8 + 8+8 → 16
/// scalar merge chain. Reference implementation for `simd_sort16_int`.
pub fn scalar_sort16_int(arr : FixedArray[Int], off : Int) -> Unit {
  if off + 16 > arr.length() || off < 0 {
    return
  }
  scalar_sort4_int(arr, off + 0)
  scalar_sort4_int(arr, off + 4)
  scalar_sort4_int(arr, off + 8)
  scalar_sort4_int(arr, off + 12)
  let tmp = FixedArray::make(16, 0)
  scalar_merge2_int(arr, off + 0, 4, arr, off + 4, 4, tmp, 0)
  scalar_merge2_int(arr, off + 8, 4, arr, off + 12, 4, tmp, 8)
  scalar_merge2_int(tmp, 0, 8, tmp, 8, 8, arr, off)
}

///|
pub fn scalar_sort_int(arr : FixedArray[Int]) -> Unit {
  arr.sort()
}

// --- Float (f32) ops over byte-buffer (E) ---
// Private duplicates of the byte-buffer codec; the public surface lives in
// src/f32_buf.mbt. Kept here so scalar_*_f32 reference impls stay self-
// contained within this `internal` sub-package.

///|
fn load_f32_le(buf : FixedArray[Byte], i : Int) -> Float {
  let off = i * 4
  let u : UInt = buf[off].to_int().reinterpret_as_uint() |
    (buf[off + 1].to_int().reinterpret_as_uint() << 8) |
    (buf[off + 2].to_int().reinterpret_as_uint() << 16) |
    (buf[off + 3].to_int().reinterpret_as_uint() << 24)
  Float::reinterpret_from_uint(u)
}

///|
fn store_f32_le(buf : FixedArray[Byte], i : Int, v : Float) -> Unit {
  let u = v.reinterpret_as_uint()
  let off = i * 4
  buf[off] = (u & 0xff).reinterpret_as_int().to_byte()
  buf[off + 1] = ((u >> 8) & 0xff).reinterpret_as_int().to_byte()
  buf[off + 2] = ((u >> 16) & 0xff).reinterpret_as_int().to_byte()
  buf[off + 3] = ((u >> 24) & 0xff).reinterpret_as_int().to_byte()
}

///|
pub fn scalar_add_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  for i in 0..<n {
    store_f32_le(out, i, load_f32_le(a, i) + load_f32_le(b, i))
  }
}

///|
pub fn scalar_mul_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  for i in 0..<n {
    store_f32_le(out, i, load_f32_le(a, i) * load_f32_le(b, i))
  }
}

///|
pub fn scalar_sum_f32(a : FixedArray[Byte], n : Int) -> Float {
  let mut acc : Float = (0.0 : Float)
  for i in 0..<n {
    acc = acc + load_f32_le(a, i)
  }
  acc
}

///|
pub fn scalar_sub_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  for i in 0..<n {
    store_f32_le(out, i, load_f32_le(a, i) - load_f32_le(b, i))
  }
}

///|
pub fn scalar_div_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  for i in 0..<n {
    store_f32_le(out, i, load_f32_le(a, i) / load_f32_le(b, i))
  }
}

///|
pub fn scalar_sqrt_f32(
  a : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  for i in 0..<n {
    store_f32_le(out, i, load_f32_le(a, i).sqrt())
  }
}

///|
pub fn scalar_min_elem_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  for i in 0..<n {
    let va = load_f32_le(a, i)
    let vb = load_f32_le(b, i)
    store_f32_le(out, i, if va < vb { va } else { vb })
  }
}

///|
pub fn scalar_max_elem_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  for i in 0..<n {
    let va = load_f32_le(a, i)
    let vb = load_f32_le(b, i)
    store_f32_le(out, i, if va > vb { va } else { vb })
  }
}

///|
pub fn scalar_dot_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  n : Int,
) -> Float {
  let mut acc : Float = (0.0 : Float)
  for i in 0..<n {
    acc = acc + load_f32_le(a, i) * load_f32_le(b, i)
  }
  acc
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
