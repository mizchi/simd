///|
/// Sort 4 i32 elements in place at `off..off+4`. 3-stage sorting network.
pub fn SimdBuffer::sort4(self : SimdBuffer, off : Int) -> Unit {
  if off < 0 || off + 4 > self.len {
    return
  }
  buf_sort4_i32(self.addr, off)
}

///|
/// Bitonic merge of two sorted 4-element slices.
pub fn SimdBuffer::bitonic_merge8(self : SimdBuffer, off : Int) -> Unit {
  if off < 0 || off + 8 > self.len {
    return
  }
  buf_bitonic_merge8_i32(self.addr, off)
}

///|
pub fn SimdBuffer::bitonic_merge16(self : SimdBuffer, off : Int) -> Unit {
  if off < 0 || off + 16 > self.len {
    return
  }
  buf_bitonic_merge16_i32(self.addr, off)
}

///|
pub fn SimdBuffer::bitonic_merge32(self : SimdBuffer, off : Int) -> Unit {
  if off < 0 || off + 32 > self.len {
    return
  }
  buf_bitonic_merge32_i32(self.addr, off)
}

///|
pub fn SimdBuffer::bitonic_merge64(self : SimdBuffer, off : Int) -> Unit {
  if off < 0 || off + 64 > self.len {
    return
  }
  buf_bitonic_merge64_i32(self.addr, off)
}

///|
/// Sort 16 elements at `off..off+16` fully in SIMD: 4 leaf sort4 +
/// 2 bitonic_merge8 + 1 bitonic_merge16.
pub fn SimdBuffer::sort16(self : SimdBuffer, off : Int) -> Unit {
  if off < 0 || off + 16 > self.len {
    return
  }
  buf_sort4_i32(self.addr, off + 0)
  buf_sort4_i32(self.addr, off + 4)
  buf_sort4_i32(self.addr, off + 8)
  buf_sort4_i32(self.addr, off + 12)
  buf_bitonic_merge8_i32(self.addr, off + 0)
  buf_bitonic_merge8_i32(self.addr, off + 8)
  buf_bitonic_merge16_i32(self.addr, off)
}

///|
/// General-purpose i32 sort. `arr` is sorted in place; `scratch` is a
/// caller-supplied SimdBuffer used as a ping-pong buffer for the merge
/// ladder. `scratch.length()` must be `>= arr.length()`. The Ring pattern
/// makes scratch allocation almost free:
///
/// ```moonbit nocheck
/// let ring = SimdBufferRing::make(arr.length() * 4 + 65536)
/// let scratch = ring.alloc_i32(arr.length())
/// SimdBuffer::sort(arr, scratch)
/// ```
pub fn SimdBuffer::sort(arr : SimdBuffer, scratch : SimdBuffer) -> Unit {
  let n = arr.len
  if n <= 1 {
    return
  }
  if scratch.len < n {
    abort("SimdBuffer.sort: scratch must be at least as large as arr")
  }
  if n <= 16 {
    if n == 16 {
      arr.sort16(0)
    } else if n == 8 {
      arr.sort4(0)
      arr.sort4(4)
      arr.bitonic_merge8(0)
    } else if n == 4 {
      arr.sort4(0)
    } else {
      sort_scalar(arr, 0, n)
    }
    return
  }
  // 1. SIMD leaf-sort each aligned 16-block.
  let leaf_end = n / 16 * 16
  let mut i = 0
  while i < leaf_end {
    arr.sort16(i)
    i = i + 16
  }
  // 1b. Scalar sort the trailing 0..15 elements.
  let tail_len = n - leaf_end
  if tail_len > 0 {
    sort_scalar(arr, leaf_end, tail_len)
  }
  // 2a. width = 16 → bitonic_merge32 for each full 32-block; scalar merge for the tail.
  let mut pos = 0
  while pos + 32 <= n {
    arr.bitonic_merge32(pos)
    pos = pos + 32
  }
  if pos + 16 < n {
    let tail_total = n - pos
    merge2_into(arr, pos, 16, arr, pos + 16, tail_total - 16, scratch, pos)
    let mut k = pos
    while k < n {
      buf_store_i32(arr.addr + k * 4, buf_load_i32(scratch.addr + k * 4))
      k = k + 1
    }
  }
  // 2b. width = 32 → bitonic_merge64 for each full 64-block.
  let mut pos2 = 0
  while pos2 + 64 <= n {
    arr.bitonic_merge64(pos2)
    pos2 = pos2 + 64
  }
  if pos2 + 32 < n {
    let tail_total = n - pos2
    merge2_into(arr, pos2, 32, arr, pos2 + 32, tail_total - 32, scratch, pos2)
    let mut k = pos2
    while k < n {
      buf_store_i32(arr.addr + k * 4, buf_load_i32(scratch.addr + k * 4))
      k = k + 1
    }
  }
  // 2c. width ≥ 64 → scalar merge2 with ping-pong arr ↔ scratch.
  let mut width = 64
  let mut src_is_arr = true
  while width < n {
    let merged = width * 2
    let mut p = 0
    while p < n {
      let a_off = p
      let a_end = if a_off + width <= n { a_off + width } else { n }
      let b_off = a_end
      let b_end = if b_off + width <= n { b_off + width } else { n }
      if b_off >= n {
        // no B half: copy A as-is into dst.
        if src_is_arr {
          let mut k = a_off
          while k < a_end {
            buf_store_i32(scratch.addr + k * 4, buf_load_i32(arr.addr + k * 4))
            k = k + 1
          }
        } else {
          let mut k = a_off
          while k < a_end {
            buf_store_i32(arr.addr + k * 4, buf_load_i32(scratch.addr + k * 4))
            k = k + 1
          }
        }
      } else if src_is_arr {
        merge2_into(
          arr,
          a_off,
          a_end - a_off,
          arr,
          b_off,
          b_end - b_off,
          scratch,
          a_off,
        )
      } else {
        merge2_into(
          scratch,
          a_off,
          a_end - a_off,
          scratch,
          b_off,
          b_end - b_off,
          arr,
          a_off,
        )
      }
      p = p + merged
    }
    width = merged
    src_is_arr = !src_is_arr
  }
  // 3. if final pass landed in scratch, copy back.
  if !src_is_arr {
    let mut k = 0
    while k < n {
      buf_store_i32(arr.addr + k * 4, buf_load_i32(scratch.addr + k * 4))
      k = k + 1
    }
  }
}

///|
/// Scalar in-place sort of `arr[off..off+len]`. Insertion sort — only used
/// for small tail blocks (< 16 elements).
fn sort_scalar(arr : SimdBuffer, off : Int, len : Int) -> Unit {
  let mut i = 1
  while i < len {
    let v = buf_load_i32(arr.addr + (off + i) * 4)
    let mut j = i - 1
    let mut done = false
    while !done && j >= 0 {
      let prev = buf_load_i32(arr.addr + (off + j) * 4)
      if prev > v {
        buf_store_i32(arr.addr + (off + j + 1) * 4, prev)
        j = j - 1
      } else {
        done = true
      }
    }
    buf_store_i32(arr.addr + (off + j + 1) * 4, v)
    i = i + 1
  }
}

///|
/// Merge two sorted ranges (`src1[so1..so1+l1]` and `src2[so2..so2+l2]`)
/// into `dst[do..]`. All buffers are SimdBuffer (i32).
fn merge2_into(
  src1 : SimdBuffer,
  so1 : Int,
  l1 : Int,
  src2 : SimdBuffer,
  so2 : Int,
  l2 : Int,
  dst : SimdBuffer,
  do_ : Int,
) -> Unit {
  let mut i = 0
  let mut j = 0
  let mut k = do_
  while i < l1 && j < l2 {
    let a = buf_load_i32(src1.addr + (so1 + i) * 4)
    let b = buf_load_i32(src2.addr + (so2 + j) * 4)
    if a <= b {
      buf_store_i32(dst.addr + k * 4, a)
      i = i + 1
    } else {
      buf_store_i32(dst.addr + k * 4, b)
      j = j + 1
    }
    k = k + 1
  }
  while i < l1 {
    buf_store_i32(dst.addr + k * 4, buf_load_i32(src1.addr + (so1 + i) * 4))
    i = i + 1
    k = k + 1
  }
  while j < l2 {
    buf_store_i32(dst.addr + k * 4, buf_load_i32(src2.addr + (so2 + j) * 4))
    j = j + 1
    k = k + 1
  }
}

// ---- inline-WAT primitives ----

///|
fn buf_sort4_i32(arr : Int, off : Int) -> Unit =
  #|(func (param i32) (param i32) (local v128) (local v128) (local v128) (local v128) local.get 0 local.get 1 i32.const 2 i32.shl i32.add v128.load local.set 2 local.get 2 local.get 2 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 3 local.get 2 local.get 3 i32x4.min_s local.set 4 local.get 2 local.get 3 i32x4.max_s local.set 5 local.get 4 local.get 5 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 2 local.get 2 local.get 2 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 3 local.get 2 local.get 3 i32x4.min_s local.set 4 local.get 2 local.get 3 i32x4.max_s local.set 5 local.get 4 local.get 5 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 2 local.get 2 local.get 2 i8x16.shuffle 0 1 2 3 8 9 10 11 4 5 6 7 12 13 14 15 local.set 3 local.get 2 local.get 3 i32x4.min_s local.set 4 local.get 2 local.get 3 i32x4.max_s local.set 5 local.get 4 local.get 5 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 12 13 14 15 local.set 2 local.get 0 local.get 1 i32.const 2 i32.shl i32.add local.get 2 v128.store)

///|
fn buf_bitonic_merge8_i32(arr : Int, off : Int) -> Unit =
  #|(func (param i32) (param i32) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) local.get 0 local.get 1 i32.const 2 i32.shl i32.add v128.load local.set 2 local.get 0 local.get 1 i32.const 4 i32.add i32.const 2 i32.shl i32.add v128.load local.set 3 local.get 3 local.get 3 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 4 local.get 2 local.get 4 i32x4.min_s local.set 5 local.get 2 local.get 4 i32x4.max_s local.set 6 local.get 5 local.get 5 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 9 local.get 5 local.get 9 i32x4.min_s local.set 7 local.get 5 local.get 9 i32x4.max_s local.set 8 local.get 7 local.get 8 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 5 local.get 6 local.get 6 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 9 local.get 6 local.get 9 i32x4.min_s local.set 7 local.get 6 local.get 9 i32x4.max_s local.set 8 local.get 7 local.get 8 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 6 local.get 5 local.get 5 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 9 local.get 5 local.get 9 i32x4.min_s local.set 7 local.get 5 local.get 9 i32x4.max_s local.set 8 local.get 7 local.get 8 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 5 local.get 6 local.get 6 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 9 local.get 6 local.get 9 i32x4.min_s local.set 7 local.get 6 local.get 9 i32x4.max_s local.set 8 local.get 7 local.get 8 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 6 local.get 0 local.get 1 i32.const 2 i32.shl i32.add local.get 5 v128.store local.get 0 local.get 1 i32.const 4 i32.add i32.const 2 i32.shl i32.add local.get 6 v128.store)

///|
fn buf_bitonic_merge16_i32(arr : Int, off : Int) -> Unit =
  #|(func (param i32) (param i32) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) local.get 0 local.get 1 i32.const 2 i32.shl i32.add v128.load local.set 2 local.get 0 local.get 1 i32.const 4 i32.add i32.const 2 i32.shl i32.add v128.load local.set 3 local.get 0 local.get 1 i32.const 8 i32.add i32.const 2 i32.shl i32.add v128.load local.set 4 local.get 0 local.get 1 i32.const 12 i32.add i32.const 2 i32.shl i32.add v128.load local.set 5 local.get 5 local.get 5 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 6 local.get 4 local.get 4 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 7 local.get 2 local.get 6 i32x4.max_s local.set 8 local.get 2 local.get 6 i32x4.min_s local.set 2 local.get 3 local.get 7 i32x4.max_s local.set 9 local.get 3 local.get 7 i32x4.min_s local.set 3 local.get 2 local.get 3 i32x4.max_s local.set 4 local.get 2 local.get 3 i32x4.min_s local.set 2 local.get 4 local.set 3 local.get 8 local.get 9 i32x4.max_s local.set 4 local.get 8 local.get 9 i32x4.min_s local.set 8 local.get 4 local.set 9 local.get 2 local.get 2 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 4 local.get 2 local.get 4 i32x4.min_s local.set 5 local.get 2 local.get 4 i32x4.max_s local.set 6 local.get 5 local.get 6 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 2 local.get 3 local.get 3 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 4 local.get 3 local.get 4 i32x4.min_s local.set 5 local.get 3 local.get 4 i32x4.max_s local.set 6 local.get 5 local.get 6 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 3 local.get 8 local.get 8 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 4 local.get 8 local.get 4 i32x4.min_s local.set 5 local.get 8 local.get 4 i32x4.max_s local.set 6 local.get 5 local.get 6 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 8 local.get 9 local.get 9 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 4 local.get 9 local.get 4 i32x4.min_s local.set 5 local.get 9 local.get 4 i32x4.max_s local.set 6 local.get 5 local.get 6 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 9 local.get 2 local.get 2 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 4 local.get 2 local.get 4 i32x4.min_s local.set 5 local.get 2 local.get 4 i32x4.max_s local.set 6 local.get 5 local.get 6 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 2 local.get 3 local.get 3 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 4 local.get 3 local.get 4 i32x4.min_s local.set 5 local.get 3 local.get 4 i32x4.max_s local.set 6 local.get 5 local.get 6 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 3 local.get 8 local.get 8 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 4 local.get 8 local.get 4 i32x4.min_s local.set 5 local.get 8 local.get 4 i32x4.max_s local.set 6 local.get 5 local.get 6 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 8 local.get 9 local.get 9 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 4 local.get 9 local.get 4 i32x4.min_s local.set 5 local.get 9 local.get 4 i32x4.max_s local.set 6 local.get 5 local.get 6 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 9 local.get 0 local.get 1 i32.const 2 i32.shl i32.add local.get 2 v128.store local.get 0 local.get 1 i32.const 4 i32.add i32.const 2 i32.shl i32.add local.get 3 v128.store local.get 0 local.get 1 i32.const 8 i32.add i32.const 2 i32.shl i32.add local.get 8 v128.store local.get 0 local.get 1 i32.const 12 i32.add i32.const 2 i32.shl i32.add local.get 9 v128.store)

///|
fn buf_bitonic_merge32_i32(arr : Int, off : Int) -> Unit =
  #|(func (param i32) (param i32) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) local.get 0 local.get 1 i32.const 2 i32.shl i32.add v128.load local.set 2 local.get 0 local.get 1 i32.const 4 i32.add i32.const 2 i32.shl i32.add v128.load local.set 3 local.get 0 local.get 1 i32.const 8 i32.add i32.const 2 i32.shl i32.add v128.load local.set 4 local.get 0 local.get 1 i32.const 12 i32.add i32.const 2 i32.shl i32.add v128.load local.set 5 local.get 0 local.get 1 i32.const 16 i32.add i32.const 2 i32.shl i32.add v128.load local.set 6 local.get 0 local.get 1 i32.const 20 i32.add i32.const 2 i32.shl i32.add v128.load local.set 7 local.get 0 local.get 1 i32.const 24 i32.add i32.const 2 i32.shl i32.add v128.load local.set 8 local.get 0 local.get 1 i32.const 28 i32.add i32.const 2 i32.shl i32.add v128.load local.set 9 local.get 9 local.get 9 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 10 local.get 8 local.get 8 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 11 local.get 7 local.get 7 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 12 local.get 6 local.get 6 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 13 local.get 2 local.get 10 i32x4.max_s local.set 6 local.get 2 local.get 10 i32x4.min_s local.set 2 local.get 3 local.get 11 i32x4.max_s local.set 7 local.get 3 local.get 11 i32x4.min_s local.set 3 local.get 4 local.get 12 i32x4.max_s local.set 8 local.get 4 local.get 12 i32x4.min_s local.set 4 local.get 5 local.get 13 i32x4.max_s local.set 9 local.get 5 local.get 13 i32x4.min_s local.set 5 local.get 2 local.get 4 i32x4.max_s local.set 10 local.get 2 local.get 4 i32x4.min_s local.set 2 local.get 10 local.set 4 local.get 3 local.get 5 i32x4.max_s local.set 10 local.get 3 local.get 5 i32x4.min_s local.set 3 local.get 10 local.set 5 local.get 6 local.get 8 i32x4.max_s local.set 10 local.get 6 local.get 8 i32x4.min_s local.set 6 local.get 10 local.set 8 local.get 7 local.get 9 i32x4.max_s local.set 10 local.get 7 local.get 9 i32x4.min_s local.set 7 local.get 10 local.set 9 local.get 2 local.get 3 i32x4.max_s local.set 10 local.get 2 local.get 3 i32x4.min_s local.set 2 local.get 10 local.set 3 local.get 4 local.get 5 i32x4.max_s local.set 10 local.get 4 local.get 5 i32x4.min_s local.set 4 local.get 10 local.set 5 local.get 6 local.get 7 i32x4.max_s local.set 10 local.get 6 local.get 7 i32x4.min_s local.set 6 local.get 10 local.set 7 local.get 8 local.get 9 i32x4.max_s local.set 10 local.get 8 local.get 9 i32x4.min_s local.set 8 local.get 10 local.set 9 local.get 2 local.get 2 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 10 local.get 2 local.get 10 i32x4.min_s local.set 11 local.get 2 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 2 local.get 3 local.get 3 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 10 local.get 3 local.get 10 i32x4.min_s local.set 11 local.get 3 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 3 local.get 4 local.get 4 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 10 local.get 4 local.get 10 i32x4.min_s local.set 11 local.get 4 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 4 local.get 5 local.get 5 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 10 local.get 5 local.get 10 i32x4.min_s local.set 11 local.get 5 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 5 local.get 6 local.get 6 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 10 local.get 6 local.get 10 i32x4.min_s local.set 11 local.get 6 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 6 local.get 7 local.get 7 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 10 local.get 7 local.get 10 i32x4.min_s local.set 11 local.get 7 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 7 local.get 8 local.get 8 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 10 local.get 8 local.get 10 i32x4.min_s local.set 11 local.get 8 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 8 local.get 9 local.get 9 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 10 local.get 9 local.get 10 i32x4.min_s local.set 11 local.get 9 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 9 local.get 2 local.get 2 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 10 local.get 2 local.get 10 i32x4.min_s local.set 11 local.get 2 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 2 local.get 3 local.get 3 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 10 local.get 3 local.get 10 i32x4.min_s local.set 11 local.get 3 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 3 local.get 4 local.get 4 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 10 local.get 4 local.get 10 i32x4.min_s local.set 11 local.get 4 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 4 local.get 5 local.get 5 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 10 local.get 5 local.get 10 i32x4.min_s local.set 11 local.get 5 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 5 local.get 6 local.get 6 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 10 local.get 6 local.get 10 i32x4.min_s local.set 11 local.get 6 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 6 local.get 7 local.get 7 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 10 local.get 7 local.get 10 i32x4.min_s local.set 11 local.get 7 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 7 local.get 8 local.get 8 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 10 local.get 8 local.get 10 i32x4.min_s local.set 11 local.get 8 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 8 local.get 9 local.get 9 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 10 local.get 9 local.get 10 i32x4.min_s local.set 11 local.get 9 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 9 local.get 0 local.get 1 i32.const 2 i32.shl i32.add local.get 2 v128.store local.get 0 local.get 1 i32.const 4 i32.add i32.const 2 i32.shl i32.add local.get 3 v128.store local.get 0 local.get 1 i32.const 8 i32.add i32.const 2 i32.shl i32.add local.get 4 v128.store local.get 0 local.get 1 i32.const 12 i32.add i32.const 2 i32.shl i32.add local.get 5 v128.store local.get 0 local.get 1 i32.const 16 i32.add i32.const 2 i32.shl i32.add local.get 6 v128.store local.get 0 local.get 1 i32.const 20 i32.add i32.const 2 i32.shl i32.add local.get 7 v128.store local.get 0 local.get 1 i32.const 24 i32.add i32.const 2 i32.shl i32.add local.get 8 v128.store local.get 0 local.get 1 i32.const 28 i32.add i32.const 2 i32.shl i32.add local.get 9 v128.store)

///|
fn buf_bitonic_merge64_i32(arr : Int, off : Int) -> Unit =
  #|(func (param i32) (param i32) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) local.get 0 local.get 1 i32.const 2 i32.shl i32.add v128.load local.set 2 local.get 0 local.get 1 i32.const 4 i32.add i32.const 2 i32.shl i32.add v128.load local.set 3 local.get 0 local.get 1 i32.const 8 i32.add i32.const 2 i32.shl i32.add v128.load local.set 4 local.get 0 local.get 1 i32.const 12 i32.add i32.const 2 i32.shl i32.add v128.load local.set 5 local.get 0 local.get 1 i32.const 16 i32.add i32.const 2 i32.shl i32.add v128.load local.set 6 local.get 0 local.get 1 i32.const 20 i32.add i32.const 2 i32.shl i32.add v128.load local.set 7 local.get 0 local.get 1 i32.const 24 i32.add i32.const 2 i32.shl i32.add v128.load local.set 8 local.get 0 local.get 1 i32.const 28 i32.add i32.const 2 i32.shl i32.add v128.load local.set 9 local.get 0 local.get 1 i32.const 32 i32.add i32.const 2 i32.shl i32.add v128.load local.set 10 local.get 0 local.get 1 i32.const 36 i32.add i32.const 2 i32.shl i32.add v128.load local.set 11 local.get 0 local.get 1 i32.const 40 i32.add i32.const 2 i32.shl i32.add v128.load local.set 12 local.get 0 local.get 1 i32.const 44 i32.add i32.const 2 i32.shl i32.add v128.load local.set 13 local.get 0 local.get 1 i32.const 48 i32.add i32.const 2 i32.shl i32.add v128.load local.set 14 local.get 0 local.get 1 i32.const 52 i32.add i32.const 2 i32.shl i32.add v128.load local.set 15 local.get 0 local.get 1 i32.const 56 i32.add i32.const 2 i32.shl i32.add v128.load local.set 16 local.get 0 local.get 1 i32.const 60 i32.add i32.const 2 i32.shl i32.add v128.load local.set 17 local.get 17 local.get 17 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 26 local.get 2 local.get 26 i32x4.max_s local.set 18 local.get 2 local.get 26 i32x4.min_s local.set 2 local.get 16 local.get 16 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 26 local.get 3 local.get 26 i32x4.max_s local.set 19 local.get 3 local.get 26 i32x4.min_s local.set 3 local.get 15 local.get 15 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 26 local.get 4 local.get 26 i32x4.max_s local.set 20 local.get 4 local.get 26 i32x4.min_s local.set 4 local.get 14 local.get 14 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 26 local.get 5 local.get 26 i32x4.max_s local.set 21 local.get 5 local.get 26 i32x4.min_s local.set 5 local.get 13 local.get 13 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 26 local.get 6 local.get 26 i32x4.max_s local.set 22 local.get 6 local.get 26 i32x4.min_s local.set 6 local.get 12 local.get 12 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 26 local.get 7 local.get 26 i32x4.max_s local.set 23 local.get 7 local.get 26 i32x4.min_s local.set 7 local.get 11 local.get 11 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 26 local.get 8 local.get 26 i32x4.max_s local.set 24 local.get 8 local.get 26 i32x4.min_s local.set 8 local.get 10 local.get 10 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 26 local.get 9 local.get 26 i32x4.max_s local.set 25 local.get 9 local.get 26 i32x4.min_s local.set 9 local.get 2 local.get 6 i32x4.max_s local.set 26 local.get 2 local.get 6 i32x4.min_s local.set 2 local.get 26 local.set 6 local.get 3 local.get 7 i32x4.max_s local.set 26 local.get 3 local.get 7 i32x4.min_s local.set 3 local.get 26 local.set 7 local.get 4 local.get 8 i32x4.max_s local.set 26 local.get 4 local.get 8 i32x4.min_s local.set 4 local.get 26 local.set 8 local.get 5 local.get 9 i32x4.max_s local.set 26 local.get 5 local.get 9 i32x4.min_s local.set 5 local.get 26 local.set 9 local.get 18 local.get 22 i32x4.max_s local.set 26 local.get 18 local.get 22 i32x4.min_s local.set 18 local.get 26 local.set 22 local.get 19 local.get 23 i32x4.max_s local.set 26 local.get 19 local.get 23 i32x4.min_s local.set 19 local.get 26 local.set 23 local.get 20 local.get 24 i32x4.max_s local.set 26 local.get 20 local.get 24 i32x4.min_s local.set 20 local.get 26 local.set 24 local.get 21 local.get 25 i32x4.max_s local.set 26 local.get 21 local.get 25 i32x4.min_s local.set 21 local.get 26 local.set 25 local.get 2 local.get 4 i32x4.max_s local.set 26 local.get 2 local.get 4 i32x4.min_s local.set 2 local.get 26 local.set 4 local.get 3 local.get 5 i32x4.max_s local.set 26 local.get 3 local.get 5 i32x4.min_s local.set 3 local.get 26 local.set 5 local.get 6 local.get 8 i32x4.max_s local.set 26 local.get 6 local.get 8 i32x4.min_s local.set 6 local.get 26 local.set 8 local.get 7 local.get 9 i32x4.max_s local.set 26 local.get 7 local.get 9 i32x4.min_s local.set 7 local.get 26 local.set 9 local.get 18 local.get 20 i32x4.max_s local.set 26 local.get 18 local.get 20 i32x4.min_s local.set 18 local.get 26 local.set 20 local.get 19 local.get 21 i32x4.max_s local.set 26 local.get 19 local.get 21 i32x4.min_s local.set 19 local.get 26 local.set 21 local.get 22 local.get 24 i32x4.max_s local.set 26 local.get 22 local.get 24 i32x4.min_s local.set 22 local.get 26 local.set 24 local.get 23 local.get 25 i32x4.max_s local.set 26 local.get 23 local.get 25 i32x4.min_s local.set 23 local.get 26 local.set 25 local.get 2 local.get 3 i32x4.max_s local.set 26 local.get 2 local.get 3 i32x4.min_s local.set 2 local.get 26 local.set 3 local.get 4 local.get 5 i32x4.max_s local.set 26 local.get 4 local.get 5 i32x4.min_s local.set 4 local.get 26 local.set 5 local.get 6 local.get 7 i32x4.max_s local.set 26 local.get 6 local.get 7 i32x4.min_s local.set 6 local.get 26 local.set 7 local.get 8 local.get 9 i32x4.max_s local.set 26 local.get 8 local.get 9 i32x4.min_s local.set 8 local.get 26 local.set 9 local.get 18 local.get 19 i32x4.max_s local.set 26 local.get 18 local.get 19 i32x4.min_s local.set 18 local.get 26 local.set 19 local.get 20 local.get 21 i32x4.max_s local.set 26 local.get 20 local.get 21 i32x4.min_s local.set 20 local.get 26 local.set 21 local.get 22 local.get 23 i32x4.max_s local.set 26 local.get 22 local.get 23 i32x4.min_s local.set 22 local.get 26 local.set 23 local.get 24 local.get 25 i32x4.max_s local.set 26 local.get 24 local.get 25 i32x4.min_s local.set 24 local.get 26 local.set 25 local.get 2 local.get 2 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 2 local.get 26 i32x4.min_s local.set 27 local.get 2 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 2 local.get 3 local.get 3 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 3 local.get 26 i32x4.min_s local.set 27 local.get 3 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 3 local.get 4 local.get 4 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 4 local.get 26 i32x4.min_s local.set 27 local.get 4 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 4 local.get 5 local.get 5 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 5 local.get 26 i32x4.min_s local.set 27 local.get 5 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 5 local.get 6 local.get 6 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 6 local.get 26 i32x4.min_s local.set 27 local.get 6 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 6 local.get 7 local.get 7 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 7 local.get 26 i32x4.min_s local.set 27 local.get 7 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 7 local.get 8 local.get 8 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 8 local.get 26 i32x4.min_s local.set 27 local.get 8 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 8 local.get 9 local.get 9 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 9 local.get 26 i32x4.min_s local.set 27 local.get 9 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 9 local.get 18 local.get 18 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 18 local.get 26 i32x4.min_s local.set 27 local.get 18 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 18 local.get 19 local.get 19 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 19 local.get 26 i32x4.min_s local.set 27 local.get 19 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 19 local.get 20 local.get 20 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 20 local.get 26 i32x4.min_s local.set 27 local.get 20 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 20 local.get 21 local.get 21 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 21 local.get 26 i32x4.min_s local.set 27 local.get 21 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 21 local.get 22 local.get 22 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 22 local.get 26 i32x4.min_s local.set 27 local.get 22 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 22 local.get 23 local.get 23 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 23 local.get 26 i32x4.min_s local.set 27 local.get 23 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 23 local.get 24 local.get 24 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 24 local.get 26 i32x4.min_s local.set 27 local.get 24 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 24 local.get 25 local.get 25 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 25 local.get 26 i32x4.min_s local.set 27 local.get 25 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 25 local.get 2 local.get 2 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 2 local.get 26 i32x4.min_s local.set 27 local.get 2 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 2 local.get 3 local.get 3 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 3 local.get 26 i32x4.min_s local.set 27 local.get 3 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 3 local.get 4 local.get 4 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 4 local.get 26 i32x4.min_s local.set 27 local.get 4 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 4 local.get 5 local.get 5 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 5 local.get 26 i32x4.min_s local.set 27 local.get 5 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 5 local.get 6 local.get 6 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 6 local.get 26 i32x4.min_s local.set 27 local.get 6 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 6 local.get 7 local.get 7 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 7 local.get 26 i32x4.min_s local.set 27 local.get 7 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 7 local.get 8 local.get 8 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 8 local.get 26 i32x4.min_s local.set 27 local.get 8 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 8 local.get 9 local.get 9 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 9 local.get 26 i32x4.min_s local.set 27 local.get 9 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 9 local.get 18 local.get 18 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 18 local.get 26 i32x4.min_s local.set 27 local.get 18 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 18 local.get 19 local.get 19 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 19 local.get 26 i32x4.min_s local.set 27 local.get 19 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 19 local.get 20 local.get 20 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 20 local.get 26 i32x4.min_s local.set 27 local.get 20 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 20 local.get 21 local.get 21 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 21 local.get 26 i32x4.min_s local.set 27 local.get 21 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 21 local.get 22 local.get 22 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 22 local.get 26 i32x4.min_s local.set 27 local.get 22 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 22 local.get 23 local.get 23 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 23 local.get 26 i32x4.min_s local.set 27 local.get 23 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 23 local.get 24 local.get 24 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 24 local.get 26 i32x4.min_s local.set 27 local.get 24 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 24 local.get 25 local.get 25 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 25 local.get 26 i32x4.min_s local.set 27 local.get 25 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 25 local.get 0 local.get 1 i32.const 2 i32.shl i32.add local.get 2 v128.store local.get 0 local.get 1 i32.const 4 i32.add i32.const 2 i32.shl i32.add local.get 3 v128.store local.get 0 local.get 1 i32.const 8 i32.add i32.const 2 i32.shl i32.add local.get 4 v128.store local.get 0 local.get 1 i32.const 12 i32.add i32.const 2 i32.shl i32.add local.get 5 v128.store local.get 0 local.get 1 i32.const 16 i32.add i32.const 2 i32.shl i32.add local.get 6 v128.store local.get 0 local.get 1 i32.const 20 i32.add i32.const 2 i32.shl i32.add local.get 7 v128.store local.get 0 local.get 1 i32.const 24 i32.add i32.const 2 i32.shl i32.add local.get 8 v128.store local.get 0 local.get 1 i32.const 28 i32.add i32.const 2 i32.shl i32.add local.get 9 v128.store local.get 0 local.get 1 i32.const 32 i32.add i32.const 2 i32.shl i32.add local.get 18 v128.store local.get 0 local.get 1 i32.const 36 i32.add i32.const 2 i32.shl i32.add local.get 19 v128.store local.get 0 local.get 1 i32.const 40 i32.add i32.const 2 i32.shl i32.add local.get 20 v128.store local.get 0 local.get 1 i32.const 44 i32.add i32.const 2 i32.shl i32.add local.get 21 v128.store local.get 0 local.get 1 i32.const 48 i32.add i32.const 2 i32.shl i32.add local.get 22 v128.store local.get 0 local.get 1 i32.const 52 i32.add i32.const 2 i32.shl i32.add local.get 23 v128.store local.get 0 local.get 1 i32.const 56 i32.add i32.const 2 i32.shl i32.add local.get 24 v128.store local.get 0 local.get 1 i32.const 60 i32.add i32.const 2 i32.shl i32.add local.get 25 v128.store)
