// Scalar sort reference impls: sort4 / merge2 / bitonic_merge{8,16,32,64} /
// sort16 / sort_int.

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
