///|
#borrow(arr)
fn sum_i32_v128(arr : FixedArray[Int], len : Int) -> Int =
  #|(func (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local v128) i32.const 0 i32x4.splat local.set 5 local.get 1 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 2 i32.const 2 i32.shl i32.add v128.load i32x4.add local.set 5 local.get 2 i32.const 4 i32.add local.set 2 br 0 end end local.get 5 i32x4.extract_lane 0 local.get 5 i32x4.extract_lane 1 i32.add local.get 5 i32x4.extract_lane 2 i32.add local.get 5 i32x4.extract_lane 3 i32.add local.set 4 block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.const 2 i32.shl i32.add i32.load i32.add local.set 4 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end local.get 4)

///|
/// SIMD sum - wasm target: v128 i32x4.add accumulator with scalar tail.
pub fn sum_i32(arr : FixedArray[Int]) -> Int {
  let len = arr.length()
  if len == 0 {
    return 0
  }
  sum_i32_v128(arr, len)
}

///|
#borrow(a, b)
fn dot_i32_v128(a : FixedArray[Int], b : FixedArray[Int], len : Int) -> Int =
  #|(func (param i32) (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local v128) i32.const 0 i32x4.splat local.set 6 local.get 2 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 6 local.get 0 local.get 3 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 3 i32.const 2 i32.shl i32.add v128.load i32x4.mul i32x4.add local.set 6 local.get 3 i32.const 4 i32.add local.set 3 br 0 end end local.get 6 i32x4.extract_lane 0 local.get 6 i32x4.extract_lane 1 i32.add local.get 6 i32x4.extract_lane 2 i32.add local.get 6 i32x4.extract_lane 3 i32.add local.set 5 block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 3 i32.const 2 i32.shl i32.add i32.load local.get 1 local.get 3 i32.const 2 i32.shl i32.add i32.load i32.mul i32.add local.set 5 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end local.get 5)

///|
/// SIMD dot product - wasm target: i32x4.mul + i32x4.add, scalar tail.
pub fn dot_i32(a : FixedArray[Int], b : FixedArray[Int]) -> Int {
  let len = a.length()
  if len == 0 {
    return 0
  }
  dot_i32_v128(a, b, len)
}

///|
#borrow(a, b, out)
fn add_i32_v128(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load i32x4.add v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add i32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add i32.load i32.add i32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
/// SIMD element-wise add - wasm target: i32x4.add with scalar tail.
pub fn add_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  add_i32_v128(a, b, out, len)
}

///|
/// Adler-32 weight vector [16, 15, ..., 1] for one 16-byte SIMD chunk.
let adler_weights : FixedArray[Byte] = {
  let w = FixedArray::make(16, b'\x00')
  for j in 0..<16 {
    w[j] = (16 - j).to_byte()
  }
  w
}

///|
/// Process `count16` 16-byte chunks starting at `data + offset`, returning
/// the updated (a, b) packed as ((b as u64) << 32) | (a as u64).
#borrow(data, weights)
fn adler32_chunks(
  data : FixedArray[Byte],
  offset : Int,
  count16 : Int,
  weights : FixedArray[Byte],
  a : Int,
  b : Int,
) -> Int64 =
  #|(func (param i32) (param i32) (param i32) (param i32) (param i32) (param i32) (result i64) (local i32) (local i32) (local i32) (local i32) (local v128) (local v128) (local v128) local.get 3 v128.load local.set 10 local.get 1 local.get 2 i32.const 4 i32.shl i32.add local.set 7 local.get 1 local.set 6 block loop local.get 6 local.get 7 i32.ge_s br_if 1 local.get 0 local.get 6 i32.add v128.load local.set 11 local.get 11 i16x8.extadd_pairwise_i8x16_u i32x4.extadd_pairwise_i16x8_u local.set 12 local.get 12 i32x4.extract_lane 0 local.get 12 i32x4.extract_lane 1 i32.add local.get 12 i32x4.extract_lane 2 i32.add local.get 12 i32x4.extract_lane 3 i32.add local.set 8 local.get 11 local.get 10 i16x8.extmul_low_i8x16_u i32x4.extadd_pairwise_i16x8_u local.get 11 local.get 10 i16x8.extmul_high_i8x16_u i32x4.extadd_pairwise_i16x8_u i32x4.add local.set 12 local.get 12 i32x4.extract_lane 0 local.get 12 i32x4.extract_lane 1 i32.add local.get 12 i32x4.extract_lane 2 i32.add local.get 12 i32x4.extract_lane 3 i32.add local.set 9 local.get 5 local.get 4 i32.const 4 i32.shl i32.add local.get 9 i32.add local.set 5 local.get 4 local.get 8 i32.add local.set 4 local.get 6 i32.const 16 i32.add local.set 6 br 0 end end local.get 5 i64.extend_i32_u i64.const 32 i64.shl local.get 4 i64.extend_i32_u i64.or)

///|
#borrow(arr)
fn min_i32_v128(arr : FixedArray[Int], len : Int) -> Int =
  #|(func (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local v128) local.get 0 i32.load i32x4.splat local.set 5 local.get 1 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 2 i32.const 2 i32.shl i32.add v128.load i32x4.min_s local.set 5 local.get 2 i32.const 4 i32.add local.set 2 br 0 end end local.get 5 i32x4.extract_lane 0 local.set 4 local.get 4 local.get 5 i32x4.extract_lane 1 local.get 4 local.get 5 i32x4.extract_lane 1 i32.lt_s select local.set 4 local.get 4 local.get 5 i32x4.extract_lane 2 local.get 4 local.get 5 i32x4.extract_lane 2 i32.lt_s select local.set 4 local.get 4 local.get 5 i32x4.extract_lane 3 local.get 4 local.get 5 i32x4.extract_lane 3 i32.lt_s select local.set 4 block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.const 2 i32.shl i32.add i32.load local.get 4 local.get 0 local.get 2 i32.const 2 i32.shl i32.add i32.load i32.lt_s select local.set 4 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end local.get 4)

///|
/// SIMD min - wasm target: v128 i32x4.min_s with scalar reduce.
pub fn min_i32(arr : FixedArray[Int]) -> Int {
  let len = arr.length()
  if len == 0 {
    return 0
  }
  min_i32_v128(arr, len)
}

///|
#borrow(arr)
fn max_i32_v128(arr : FixedArray[Int], len : Int) -> Int =
  #|(func (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local v128) local.get 0 i32.load i32x4.splat local.set 5 local.get 1 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 2 i32.const 2 i32.shl i32.add v128.load i32x4.max_s local.set 5 local.get 2 i32.const 4 i32.add local.set 2 br 0 end end local.get 5 i32x4.extract_lane 0 local.set 4 local.get 4 local.get 5 i32x4.extract_lane 1 local.get 4 local.get 5 i32x4.extract_lane 1 i32.gt_s select local.set 4 local.get 4 local.get 5 i32x4.extract_lane 2 local.get 4 local.get 5 i32x4.extract_lane 2 i32.gt_s select local.set 4 local.get 4 local.get 5 i32x4.extract_lane 3 local.get 4 local.get 5 i32x4.extract_lane 3 i32.gt_s select local.set 4 block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.const 2 i32.shl i32.add i32.load local.get 4 local.get 0 local.get 2 i32.const 2 i32.shl i32.add i32.load i32.gt_s select local.set 4 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end local.get 4)

///|
/// SIMD max - wasm target: v128 i32x4.max_s with scalar reduce.
pub fn max_i32(arr : FixedArray[Int]) -> Int {
  let len = arr.length()
  if len == 0 {
    return 0
  }
  max_i32_v128(arr, len)
}

///|
/// Find the first index in `arr` where the element equals `needle`. Same
/// "no early break, track found via select" pattern as `find_byte`,
/// but on i32x4 lanes (chunk = 4 elements at byte stride 4).
#borrow(arr)
fn find_int_i32_v128(arr : FixedArray[Int], len : Int, needle : Int) -> Int =
  #|(func (param i32) (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local i32) (local v128) local.get 2 i32x4.splat local.set 7 local.get 1 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 4 i32.const -1 local.set 5 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 0 local.get 3 i32.const 2 i32.shl i32.add v128.load local.get 7 i32x4.eq i32x4.bitmask local.set 6 local.get 3 local.get 6 i32.ctz i32.add local.get 5 local.get 5 i32.const -1 i32.eq local.get 6 i32.mul select local.set 5 local.get 3 i32.const 4 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 1 i32.ge_s br_if 1 local.get 3 local.get 5 local.get 5 i32.const -1 i32.eq local.get 0 local.get 3 i32.const 2 i32.shl i32.add i32.load local.get 2 i32.eq i32.and select local.set 5 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end local.get 5)

///|
/// 2-pass SIMD argmin: min_i32 for the value, then SIMD find for the first
/// matching index.
pub fn argmin_i32(arr : FixedArray[Int]) -> Int {
  let len = arr.length()
  if len == 0 {
    return -1
  }
  find_int_i32_v128(arr, len, min_i32_v128(arr, len))
}

///|
pub fn argmax_i32(arr : FixedArray[Int]) -> Int {
  let len = arr.length()
  if len == 0 {
    return -1
  }
  find_int_i32_v128(arr, len, max_i32_v128(arr, len))
}

///|
#borrow(a, b, out)
fn saxpy_i32_v128(
  k : Int,
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) (local v128) local.get 0 i32x4.splat local.set 7 local.get 4 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 6 block loop local.get 5 local.get 6 i32.ge_s br_if 1 local.get 3 local.get 5 i32.const 2 i32.shl i32.add local.get 7 local.get 1 local.get 5 i32.const 2 i32.shl i32.add v128.load i32x4.mul local.get 2 local.get 5 i32.const 2 i32.shl i32.add v128.load i32x4.add v128.store local.get 5 i32.const 4 i32.add local.set 5 br 0 end end block loop local.get 5 local.get 4 i32.ge_s br_if 1 local.get 3 local.get 5 i32.const 2 i32.shl i32.add local.get 0 local.get 1 local.get 5 i32.const 2 i32.shl i32.add i32.load i32.mul local.get 2 local.get 5 i32.const 2 i32.shl i32.add i32.load i32.add i32.store local.get 5 i32.const 1 i32.add local.set 5 br 0 end end)

///|
/// SIMD saxpy: out[i] = k * a[i] + b[i] - wasm target: v128 i32x4.mul + add.
pub fn saxpy_i32(
  k : Int,
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  saxpy_i32_v128(k, a, b, out, len)
}

///|
#borrow(data)
fn popcount_bytes_v128(data : FixedArray[Byte], len : Int) -> Int =
  #|(func (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local v128) local.get 1 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 0 local.get 2 i32.add v128.load i8x16.popcnt i16x8.extadd_pairwise_i8x16_u i32x4.extadd_pairwise_i16x8_u local.set 5 local.get 4 local.get 5 i32x4.extract_lane 0 i32.add local.get 5 i32x4.extract_lane 1 i32.add local.get 5 i32x4.extract_lane 2 i32.add local.get 5 i32x4.extract_lane 3 i32.add local.set 4 local.get 2 i32.const 16 i32.add local.set 2 br 0 end end block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.add i32.load8_u i32.popcnt i32.add local.set 4 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end local.get 4)

///|
/// SIMD popcount of a byte buffer - wasm target: i8x16.popcnt + horizontal sum.
pub fn popcount_bytes(data : FixedArray[Byte]) -> Int {
  let len = data.length()
  if len == 0 {
    return 0
  }
  popcount_bytes_v128(data, len)
}

///|
#borrow(src, dst)
fn memcpy_bytes_v128(
  src : FixedArray[Byte],
  si : Int,
  dst : FixedArray[Byte],
  di : Int,
  len : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 4 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 6 block loop local.get 5 local.get 6 i32.ge_s br_if 1 local.get 2 local.get 3 i32.add local.get 5 i32.add local.get 0 local.get 1 i32.add local.get 5 i32.add v128.load v128.store local.get 5 i32.const 16 i32.add local.set 5 br 0 end end block loop local.get 5 local.get 4 i32.ge_s br_if 1 local.get 2 local.get 3 i32.add local.get 5 i32.add local.get 0 local.get 1 i32.add local.get 5 i32.add i32.load8_u i32.store8 local.get 5 i32.const 1 i32.add local.set 5 br 0 end end)

///|
/// SIMD memcpy - wasm target: 16-byte v128.load/v128.store chunks.
pub fn memcpy_bytes(
  src : FixedArray[Byte],
  si : Int,
  dst : FixedArray[Byte],
  di : Int,
  len : Int,
) -> Unit {
  if len <= 0 {
    return
  }
  memcpy_bytes_v128(src, si, dst, di, len)
}

///|
#borrow(dst)
fn memset_bytes_v128(
  dst : FixedArray[Byte],
  off : Int,
  len : Int,
  value : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) (local v128) local.get 3 i8x16.splat local.set 6 local.get 2 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 0 local.get 1 i32.add local.get 4 i32.add local.get 6 v128.store local.get 4 i32.const 16 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 2 i32.ge_s br_if 1 local.get 0 local.get 1 i32.add local.get 4 i32.add local.get 3 i32.store8 local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
/// SIMD memset - wasm target: i8x16.splat + 16-byte v128.store chunks.
pub fn memset_bytes(
  dst : FixedArray[Byte],
  off : Int,
  len : Int,
  value : Byte,
) -> Unit {
  if len <= 0 {
    return
  }
  memset_bytes_v128(dst, off, len, value.to_int())
}

///|
#borrow(a, b)
fn equal_bytes_v128(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  len : Int,
) -> Int =
  #|(func (param i32) (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) local.get 2 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 3 i32.add v128.load local.get 1 local.get 3 i32.add v128.load i8x16.eq i8x16.bitmask i32.const 65535 i32.ne i32.or local.set 5 local.get 3 i32.const 16 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 3 i32.add i32.load8_u local.get 1 local.get 3 i32.add i32.load8_u i32.ne i32.or local.set 5 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end local.get 5 i32.eqz)

///|
/// SIMD byte equality (first `len` bytes) - wasm target: i8x16.eq + bitmask
/// folded into a mismatch flag (no early break).
pub fn equal_bytes(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  len : Int,
) -> Bool {
  if len <= 0 {
    return true
  }
  equal_bytes_v128(a, b, len) != 0
}

///|
#borrow(data)
fn find_byte_v128(data : FixedArray[Byte], len : Int, needle : Int) -> Int =
  #|(func (param i32) (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local i32) (local v128) local.get 2 i8x16.splat local.set 7 local.get 1 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 4 i32.const -1 local.set 5 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 0 local.get 3 i32.add v128.load local.get 7 i8x16.eq i8x16.bitmask local.set 6 local.get 3 local.get 6 i32.ctz i32.add local.get 5 local.get 5 i32.const -1 i32.eq local.get 6 i32.mul select local.set 5 local.get 3 i32.const 16 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 1 i32.ge_s br_if 1 local.get 3 local.get 5 local.get 5 i32.const -1 i32.eq local.get 0 local.get 3 i32.add i32.load8_u local.get 2 i32.eq i32.and select local.set 5 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end local.get 5)

///|
/// SIMD find first byte equal to `needle` - wasm target: i8x16.eq + bitmask
/// with select-based update (no early break).
pub fn find_byte(data : FixedArray[Byte], needle : Byte) -> Int {
  let len = data.length()
  if len == 0 {
    return -1
  }
  find_byte_v128(data, len, needle.to_int())
}

///|
#borrow(data)
fn count_byte_v128(data : FixedArray[Byte], len : Int, needle : Int) -> Int =
  #|(func (param i32) (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local v128) local.get 2 i8x16.splat local.set 6 local.get 1 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 3 i32.add v128.load local.get 6 i8x16.eq i8x16.bitmask i32.popcnt i32.add local.set 5 local.get 3 i32.const 16 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 1 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 3 i32.add i32.load8_u local.get 2 i32.eq i32.add local.set 5 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end local.get 5)

///|
/// SIMD count of bytes equal to `needle` - wasm target: i8x16.eq + bitmask +
/// i32.popcnt.
pub fn count_byte(data : FixedArray[Byte], needle : Byte) -> Int {
  let len = data.length()
  if len == 0 {
    return 0
  }
  count_byte_v128(data, len, needle.to_int())
}

// --- reductions + scan (D) ---

///|
#borrow(arr)
fn prod_i32_v128(arr : FixedArray[Int], len : Int) -> Int =
  #|(func (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local v128) i32.const 1 i32x4.splat local.set 5 local.get 1 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 2 i32.const 2 i32.shl i32.add v128.load i32x4.mul local.set 5 local.get 2 i32.const 4 i32.add local.set 2 br 0 end end local.get 5 i32x4.extract_lane 0 local.get 5 i32x4.extract_lane 1 i32.mul local.get 5 i32x4.extract_lane 2 i32.mul local.get 5 i32x4.extract_lane 3 i32.mul local.set 4 block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.const 2 i32.shl i32.add i32.load i32.mul local.set 4 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end local.get 4)

///|
pub fn prod_i32(arr : FixedArray[Int]) -> Int {
  let len = arr.length()
  if len == 0 {
    return 1
  }
  prod_i32_v128(arr, len)
}

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
#borrow(arr)
fn count_nonzero_i32_v128(arr : FixedArray[Int], len : Int) -> Int =
  #|(func (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local v128) i32.const 0 i32x4.splat local.set 5 local.get 1 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.const 2 i32.shl i32.add v128.load local.get 5 i32x4.ne i32x4.bitmask i32.popcnt i32.add local.set 4 local.get 2 i32.const 4 i32.add local.set 2 br 0 end end block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 4 i32.const 0 local.get 0 local.get 2 i32.const 2 i32.shl i32.add i32.load i32.ne i32.add local.set 4 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end local.get 4)

///|
pub fn count_nonzero_i32(arr : FixedArray[Int]) -> Int {
  let len = arr.length()
  if len == 0 {
    return 0
  }
  count_nonzero_i32_v128(arr, len)
}

///|
/// any/all over Int (treats lane != 0 as true). Returns 1 if true, 0 if false.
#borrow(arr)
fn any_i32_v128(arr : FixedArray[Int], len : Int) -> Int =
  #|(func (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local v128) i32.const 0 i32x4.splat local.set 5 local.get 1 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.const 2 i32.shl i32.add v128.load local.get 5 i32x4.ne i32x4.bitmask i32.or local.set 4 local.get 2 i32.const 4 i32.add local.set 2 br 0 end end block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 4 i32.const 0 local.get 0 local.get 2 i32.const 2 i32.shl i32.add i32.load i32.ne i32.or local.set 4 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end local.get 4 i32.const 0 i32.ne)

///|
pub fn any_i32(arr : FixedArray[Int]) -> Bool {
  let len = arr.length()
  if len == 0 {
    return false
  }
  any_i32_v128(arr, len) != 0
}

///|
#borrow(arr)
fn all_i32_v128(arr : FixedArray[Int], len : Int) -> Int =
  #|(func (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local v128) i32.const 0 i32x4.splat local.set 5 local.get 1 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.const 2 i32.shl i32.add v128.load local.get 5 i32x4.eq i32x4.bitmask i32.or local.set 4 local.get 2 i32.const 4 i32.add local.set 2 br 0 end end block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 4 i32.const 0 local.get 0 local.get 2 i32.const 2 i32.shl i32.add i32.load i32.eq i32.or local.set 4 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end local.get 4 i32.eqz)

///|
pub fn all_i32(arr : FixedArray[Int]) -> Bool {
  let len = arr.length()
  if len == 0 {
    return true
  }
  all_i32_v128(arr, len) != 0
}

///|
/// Hillis-Steele prefix sum on i32x4: two stages of (shuffle-left + add) per
/// chunk, then add the running sum from the previous chunk via splat.
#borrow(arr, out)
fn cumsum_i32_v128(arr : FixedArray[Int], out : FixedArray[Int], len : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (local i32) (local i32) (local i32) (local v128) (local v128) (local v128) (local v128) i32.const 0 i32x4.splat local.set 6 i32.const 0 local.set 5 local.get 2 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 0 local.get 3 i32.const 2 i32.shl i32.add v128.load local.set 7 local.get 6 local.get 7 i8x16.shuffle 0 1 2 3 16 17 18 19 20 21 22 23 24 25 26 27 local.get 7 i32x4.add local.set 8 local.get 6 local.get 8 i8x16.shuffle 0 1 2 3 4 5 6 7 16 17 18 19 20 21 22 23 local.get 8 i32x4.add local.set 9 local.get 5 i32x4.splat local.get 9 i32x4.add local.set 7 local.get 1 local.get 3 i32.const 2 i32.shl i32.add local.get 7 v128.store local.get 7 i32x4.extract_lane 3 local.set 5 local.get 3 i32.const 4 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 3 i32.const 2 i32.shl i32.add i32.load i32.add local.set 5 local.get 1 local.get 3 i32.const 2 i32.shl i32.add local.get 5 i32.store local.get 3 i32.const 1 i32.add local.set 3 br 0 end end)

///|
pub fn cumsum_i32(arr : FixedArray[Int], out : FixedArray[Int]) -> Unit {
  let len = arr.length()
  if len == 0 {
    return
  }
  cumsum_i32_v128(arr, out, len)
}

///|
/// Hillis-Steele prefix product on i32x4. Same shape as cumsum but the
/// "fill" lanes are 1 (identity for mul), the running accumulator starts at
/// 1, and chunks combine via mul.
#borrow(arr, out)
fn cumprod_i32_v128(arr : FixedArray[Int], out : FixedArray[Int], len : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (local i32) (local i32) (local i32) (local v128) (local v128) (local v128) (local v128) i32.const 1 i32x4.splat local.set 6 i32.const 1 local.set 5 local.get 2 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 0 local.get 3 i32.const 2 i32.shl i32.add v128.load local.set 7 local.get 6 local.get 7 i8x16.shuffle 0 1 2 3 16 17 18 19 20 21 22 23 24 25 26 27 local.get 7 i32x4.mul local.set 8 local.get 6 local.get 8 i8x16.shuffle 0 1 2 3 4 5 6 7 16 17 18 19 20 21 22 23 local.get 8 i32x4.mul local.set 9 local.get 5 i32x4.splat local.get 9 i32x4.mul local.set 7 local.get 1 local.get 3 i32.const 2 i32.shl i32.add local.get 7 v128.store local.get 7 i32x4.extract_lane 3 local.set 5 local.get 3 i32.const 4 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 3 i32.const 2 i32.shl i32.add i32.load i32.mul local.set 5 local.get 1 local.get 3 i32.const 2 i32.shl i32.add local.get 5 i32.store local.get 3 i32.const 1 i32.add local.set 3 br 0 end end)

///|
pub fn cumprod_i32(arr : FixedArray[Int], out : FixedArray[Int]) -> Unit {
  let len = arr.length()
  if len == 0 {
    return
  }
  cumprod_i32_v128(arr, out, len)
}

// --- linear algebra (C) ---

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

// --- Int element-wise ops (B) ---

///|
#borrow(a, b, out)
fn sub_i32_v128(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load i32x4.sub v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add i32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add i32.load i32.sub i32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
pub fn sub_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  sub_i32_v128(a, b, out, len)
}

///|
#borrow(a, b, out)
fn mul_i32_v128(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load i32x4.mul v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add i32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add i32.load i32.mul i32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
pub fn mul_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  mul_i32_v128(a, b, out, len)
}

///|
#borrow(a, out)
fn neg_i32_v128(a : FixedArray[Int], out : FixedArray[Int], len : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (local i32) (local i32) local.get 2 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 1 local.get 3 i32.const 2 i32.shl i32.add local.get 0 local.get 3 i32.const 2 i32.shl i32.add v128.load i32x4.neg v128.store local.get 3 i32.const 4 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 1 local.get 3 i32.const 2 i32.shl i32.add i32.const 0 local.get 0 local.get 3 i32.const 2 i32.shl i32.add i32.load i32.sub i32.store local.get 3 i32.const 1 i32.add local.set 3 br 0 end end)

///|
pub fn neg_i32(a : FixedArray[Int], out : FixedArray[Int]) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  neg_i32_v128(a, out, len)
}

///|
#borrow(a, out)
fn abs_i32_v128(a : FixedArray[Int], out : FixedArray[Int], len : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (local i32) (local i32) local.get 2 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 1 local.get 3 i32.const 2 i32.shl i32.add local.get 0 local.get 3 i32.const 2 i32.shl i32.add v128.load i32x4.abs v128.store local.get 3 i32.const 4 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 1 local.get 3 i32.const 2 i32.shl i32.add i32.const 0 local.get 0 local.get 3 i32.const 2 i32.shl i32.add i32.load i32.sub local.get 0 local.get 3 i32.const 2 i32.shl i32.add i32.load local.get 0 local.get 3 i32.const 2 i32.shl i32.add i32.load i32.const 0 i32.lt_s select i32.store local.get 3 i32.const 1 i32.add local.set 3 br 0 end end)

///|
pub fn abs_i32(a : FixedArray[Int], out : FixedArray[Int]) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  abs_i32_v128(a, out, len)
}

///|
#borrow(a, b, out)
fn min_elem_i32_v128(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load i32x4.min_s v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add i32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add i32.load local.get 0 local.get 4 i32.const 2 i32.shl i32.add i32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add i32.load i32.lt_s select i32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
pub fn min_elem_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  min_elem_i32_v128(a, b, out, len)
}

///|
#borrow(a, b, out)
fn max_elem_i32_v128(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load i32x4.max_s v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add i32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add i32.load local.get 0 local.get 4 i32.const 2 i32.shl i32.add i32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add i32.load i32.gt_s select i32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
pub fn max_elem_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  max_elem_i32_v128(a, b, out, len)
}

///|
/// eq → mask: out[i] = (a[i] == b[i]) ? -1 : 0.
/// Vector i32x4.eq already yields 0xFFFFFFFF/0 lanes; scalar tail uses
/// `i32.const 0 ... i32.sub` to sign-extend `0/1` → `0/-1`.
#borrow(a, b, out)
fn eq_i32_v128(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load i32x4.eq v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add i32.const 0 local.get 0 local.get 4 i32.const 2 i32.shl i32.add i32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add i32.load i32.eq i32.sub i32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
pub fn eq_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  eq_i32_v128(a, b, out, len)
}

///|
#borrow(a, b, out)
fn lt_i32_v128(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load i32x4.lt_s v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add i32.const 0 local.get 0 local.get 4 i32.const 2 i32.shl i32.add i32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add i32.load i32.lt_s i32.sub i32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
pub fn lt_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  lt_i32_v128(a, b, out, len)
}

///|
#borrow(a, b, out)
fn gt_i32_v128(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load i32x4.gt_s v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add i32.const 0 local.get 0 local.get 4 i32.const 2 i32.shl i32.add i32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add i32.load i32.gt_s i32.sub i32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
pub fn gt_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  gt_i32_v128(a, b, out, len)
}

///|
/// where(mask, a, b, out): out[i] = mask[i] ? a[i] : b[i]. Uses v128.bitselect:
/// bitselect(v1, v2, c) = (v1 & c) | (v2 & ~c), so mask=-1 picks `a`, mask=0 picks `b`.
#borrow(mask, a, b, out)
fn where_i32_v128(
  mask : FixedArray[Int],
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 4 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 6 block loop local.get 5 local.get 6 i32.ge_s br_if 1 local.get 3 local.get 5 i32.const 2 i32.shl i32.add local.get 1 local.get 5 i32.const 2 i32.shl i32.add v128.load local.get 2 local.get 5 i32.const 2 i32.shl i32.add v128.load local.get 0 local.get 5 i32.const 2 i32.shl i32.add v128.load v128.bitselect v128.store local.get 5 i32.const 4 i32.add local.set 5 br 0 end end block loop local.get 5 local.get 4 i32.ge_s br_if 1 local.get 3 local.get 5 i32.const 2 i32.shl i32.add local.get 1 local.get 5 i32.const 2 i32.shl i32.add i32.load local.get 2 local.get 5 i32.const 2 i32.shl i32.add i32.load local.get 0 local.get 5 i32.const 2 i32.shl i32.add i32.load select i32.store local.get 5 i32.const 1 i32.add local.set 5 br 0 end end)

///|
pub fn where_i32(
  mask : FixedArray[Int],
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = mask.length()
  if len == 0 {
    return
  }
  where_i32_v128(mask, a, b, out, len)
}

///|
/// SIMD i32 div: wasm has no i32x4.div, so we widen lanes (0,1) and (2,3)
/// to f64x2, do f64x2.div, truncate back via i32x4.trunc_sat_f64x2_s_zero,
/// and recombine with i8x16.shuffle. i32 range fits in f64 mantissa, so
/// the round-trip is exact for representable quotients.
#borrow(a, b, out)
fn div_i32_v128(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.set 6 local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load local.set 7 local.get 6 f64x2.convert_low_i32x4_s local.set 8 local.get 7 f64x2.convert_low_i32x4_s local.set 9 local.get 8 local.get 9 f64x2.div i32x4.trunc_sat_f64x2_s_zero local.set 10 local.get 6 local.get 6 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 f64x2.convert_low_i32x4_s local.set 8 local.get 7 local.get 7 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 f64x2.convert_low_i32x4_s local.set 9 local.get 8 local.get 9 f64x2.div i32x4.trunc_sat_f64x2_s_zero local.set 11 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 10 local.get 11 i8x16.shuffle 0 1 2 3 4 5 6 7 16 17 18 19 20 21 22 23 v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add i32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add i32.load i32.div_s i32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
pub fn div_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  div_i32_v128(a, b, out, len)
}

// --- UTF-8 validation (I) ---

///|
/// Returns the byte offset of the first 16-byte chunk that contains any
/// non-ASCII byte, or `(len/16)*16` if every aligned chunk is pure ASCII.
/// Uses select-based tracking to record the first hit without early exit.
#borrow(data)
fn first_non_ascii_chunk_v128(
  data : FixedArray[Byte],
  len : Int,
) -> Int =
  #|(func (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local i32) local.get 1 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 3 local.get 3 local.set 4 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 0 local.get 2 i32.add v128.load i8x16.bitmask local.set 5 local.get 2 local.get 4 local.get 4 local.get 3 i32.eq local.get 5 i32.mul select local.set 4 local.get 2 i32.const 16 i32.add local.set 2 br 0 end end local.get 4)

///|
/// SIMD-accelerated UTF-8 validation. SIMD fast path skips all-ASCII chunks
/// via `first_non_ascii_chunk_v128`; the scalar state machine takes
/// over at the first byte that might be non-ASCII (which is safe because
/// no multi-byte sequence can cross into that chunk from a prior all-ASCII
/// chunk).
pub fn validate_utf8(data : FixedArray[Byte]) -> Bool {
  let len = data.length()
  let start = if len >= 16 {
    first_non_ascii_chunk_v128(data, len)
  } else {
    0
  }
  @internal.scalar_validate_utf8_from(data, start)
}

// --- string ops (H) ---

///|
/// Fold each chunk's MSB-bitmask via OR, then check it's zero (no high bit
/// set means pure ASCII). Tail bytes use `data[i] & 0x80`.
#borrow(data)
fn is_ascii_v128(data : FixedArray[Byte], len : Int) -> Int =
  #|(func (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) local.get 1 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.add v128.load i8x16.bitmask i32.or local.set 4 local.get 2 i32.const 16 i32.add local.set 2 br 0 end end block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.add i32.load8_u i32.const 128 i32.and i32.or local.set 4 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end local.get 4 i32.eqz)

///|
pub fn is_ascii(data : FixedArray[Byte]) -> Bool {
  let len = data.length()
  if len == 0 {
    return true
  }
  is_ascii_v128(data, len) != 0
}

///|
/// In-place ASCII A-Z → a-z via per-byte range mask:
/// `delta = (v >= 'A') & (v <= 'Z') & 0x20`, then `v += delta`.
#borrow(data)
fn to_lower_ascii_v128(data : FixedArray[Byte], len : Int) -> Unit =
  #|(func (param i32) (param i32) (local i32) (local i32) (local v128) (local v128) (local v128) (local v128) i32.const 65 i8x16.splat local.set 4 i32.const 90 i8x16.splat local.set 5 i32.const 32 i8x16.splat local.set 6 local.get 1 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 0 local.get 2 i32.add v128.load local.set 7 local.get 0 local.get 2 i32.add local.get 7 local.get 4 i8x16.ge_s local.get 7 local.get 5 i8x16.le_s v128.and local.get 6 v128.and local.get 7 i8x16.add v128.store local.get 2 i32.const 16 i32.add local.set 2 br 0 end end block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 0 local.get 2 i32.add local.get 0 local.get 2 i32.add i32.load8_u local.get 0 local.get 2 i32.add i32.load8_u i32.const 65 i32.ge_s local.get 0 local.get 2 i32.add i32.load8_u i32.const 90 i32.le_s i32.and i32.const 32 i32.mul i32.add i32.store8 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end)

///|
pub fn to_lower_ascii(data : FixedArray[Byte]) -> Unit {
  let len = data.length()
  if len == 0 {
    return
  }
  to_lower_ascii_v128(data, len)
}

///|
/// In-place ASCII a-z → A-Z (symmetric: subtract 32 when in range).
#borrow(data)
fn to_upper_ascii_v128(data : FixedArray[Byte], len : Int) -> Unit =
  #|(func (param i32) (param i32) (local i32) (local i32) (local v128) (local v128) (local v128) (local v128) i32.const 97 i8x16.splat local.set 4 i32.const 122 i8x16.splat local.set 5 i32.const 32 i8x16.splat local.set 6 local.get 1 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 0 local.get 2 i32.add v128.load local.set 7 local.get 0 local.get 2 i32.add local.get 7 local.get 7 local.get 4 i8x16.ge_s local.get 7 local.get 5 i8x16.le_s v128.and local.get 6 v128.and i8x16.sub v128.store local.get 2 i32.const 16 i32.add local.set 2 br 0 end end block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 0 local.get 2 i32.add local.get 0 local.get 2 i32.add i32.load8_u local.get 0 local.get 2 i32.add i32.load8_u i32.const 97 i32.ge_s local.get 0 local.get 2 i32.add i32.load8_u i32.const 122 i32.le_s i32.and i32.const 32 i32.mul i32.sub i32.store8 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end)

///|
pub fn to_upper_ascii(data : FixedArray[Byte]) -> Unit {
  let len = data.length()
  if len == 0 {
    return
  }
  to_upper_ascii_v128(data, len)
}

// --- gather / scatter (G) ---

///|
/// Gather 4 elements per chunk: read 4 indices, scalar-load 4 values, build
/// a v128 via i32x4.replace_lane, then v128.store. wasm SIMD has no real
/// gather instruction so this mostly saves on the store side.
#borrow(arr, indices, out)
fn gather_i32_v128(
  arr : FixedArray[Int],
  indices : FixedArray[Int],
  out : FixedArray[Int],
  n : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add i32.const 0 i32x4.splat local.get 0 local.get 1 local.get 4 i32.const 2 i32.shl i32.add i32.load i32.const 2 i32.shl i32.add i32.load i32x4.replace_lane 0 local.get 0 local.get 1 local.get 4 i32.const 1 i32.add i32.const 2 i32.shl i32.add i32.load i32.const 2 i32.shl i32.add i32.load i32x4.replace_lane 1 local.get 0 local.get 1 local.get 4 i32.const 2 i32.add i32.const 2 i32.shl i32.add i32.load i32.const 2 i32.shl i32.add i32.load i32x4.replace_lane 2 local.get 0 local.get 1 local.get 4 i32.const 3 i32.add i32.const 2 i32.shl i32.add i32.load i32.const 2 i32.shl i32.add i32.load i32x4.replace_lane 3 v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 1 local.get 4 i32.const 2 i32.shl i32.add i32.load i32.const 2 i32.shl i32.add i32.load i32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
pub fn gather_i32(
  arr : FixedArray[Int],
  indices : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let n = indices.length()
  if n == 0 {
    return
  }
  gather_i32_v128(arr, indices, out, n)
}

///|
/// scatter is fundamentally scalar — random writes can't be vectorized
/// without lock-step / conflict-detection (which wasm SIMD lacks).
pub fn scatter_i32(
  out : FixedArray[Int],
  indices : FixedArray[Int],
  vals : FixedArray[Int],
) -> Unit {
  @internal.scalar_scatter(out, indices, vals)
}

// --- sort (F) ---

///|
/// 4-element sorting network on i32 using v128: 3 shuffle/min/max stages.
/// Stage 1: swap pairs (0,1), (2,3). Stage 2: swap halves (0,2), (1,3).
/// Stage 3: swap middle pair (1,2). Each stage = shuffle + min/max + blend.
#borrow(arr)
fn sort4_i32_v128(arr : FixedArray[Int], off : Int) -> Unit =
  #|(func (param i32) (param i32) (local v128) (local v128) (local v128) (local v128) local.get 0 local.get 1 i32.const 2 i32.shl i32.add v128.load local.set 2 local.get 2 local.get 2 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 3 local.get 2 local.get 3 i32x4.min_s local.set 4 local.get 2 local.get 3 i32x4.max_s local.set 5 local.get 4 local.get 5 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 2 local.get 2 local.get 2 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 3 local.get 2 local.get 3 i32x4.min_s local.set 4 local.get 2 local.get 3 i32x4.max_s local.set 5 local.get 4 local.get 5 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 2 local.get 2 local.get 2 i8x16.shuffle 0 1 2 3 8 9 10 11 4 5 6 7 12 13 14 15 local.set 3 local.get 2 local.get 3 i32x4.min_s local.set 4 local.get 2 local.get 3 i32x4.max_s local.set 5 local.get 4 local.get 5 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 12 13 14 15 local.set 2 local.get 0 local.get 1 i32.const 2 i32.shl i32.add local.get 2 v128.store)

///|
pub fn sort4_i32(arr : FixedArray[Int], off : Int) -> Unit {
  if off + 4 > arr.length() || off < 0 {
    return
  }
  sort4_i32_v128(arr, off)
}

///|
/// Bitonic merge of two sorted 4-element slices (arr[off..off+4] and
/// arr[off+4..off+8]) into one sorted 8-element block in place. Stage 1
/// reverses B and takes element-wise min/max against A; stages 2 and 3 are
/// distance-2 and distance-1 compare-exchanges within each half via
/// shuffle + min_s/max_s + blend shuffle.
#borrow(arr)
fn bitonic_merge8_i32_v128(arr : FixedArray[Int], off : Int) -> Unit =
  #|(func (param i32) (param i32) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) local.get 0 local.get 1 i32.const 2 i32.shl i32.add v128.load local.set 2 local.get 0 local.get 1 i32.const 4 i32.add i32.const 2 i32.shl i32.add v128.load local.set 3 local.get 3 local.get 3 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 4 local.get 2 local.get 4 i32x4.min_s local.set 5 local.get 2 local.get 4 i32x4.max_s local.set 6 local.get 5 local.get 5 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 9 local.get 5 local.get 9 i32x4.min_s local.set 7 local.get 5 local.get 9 i32x4.max_s local.set 8 local.get 7 local.get 8 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 5 local.get 6 local.get 6 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 9 local.get 6 local.get 9 i32x4.min_s local.set 7 local.get 6 local.get 9 i32x4.max_s local.set 8 local.get 7 local.get 8 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 6 local.get 5 local.get 5 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 9 local.get 5 local.get 9 i32x4.min_s local.set 7 local.get 5 local.get 9 i32x4.max_s local.set 8 local.get 7 local.get 8 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 5 local.get 6 local.get 6 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 9 local.get 6 local.get 9 i32x4.min_s local.set 7 local.get 6 local.get 9 i32x4.max_s local.set 8 local.get 7 local.get 8 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 6 local.get 0 local.get 1 i32.const 2 i32.shl i32.add local.get 5 v128.store local.get 0 local.get 1 i32.const 4 i32.add i32.const 2 i32.shl i32.add local.get 6 v128.store)

///|
pub fn bitonic_merge8_i32(arr : FixedArray[Int], off : Int) -> Unit {
  if off + 8 > arr.length() || off < 0 {
    return
  }
  bitonic_merge8_i32_v128(arr, off)
}

///|
/// Bitonic merge of two sorted 8-element slices (arr[off..off+8] and
/// arr[off+8..off+16]) into one sorted 16-element block, in place. 4 stages:
///   1) reverse B halves and lane-wise min/max with A halves (distance 8)
///   2) distance-4 min/max between v128 pairs within each 8-half
///   3) distance-2 compare-exchange within each v128
///   4) distance-1 compare-exchange within each v128
#borrow(arr)
fn bitonic_merge16_i32_v128(arr : FixedArray[Int], off : Int) -> Unit =
  #|(func (param i32) (param i32) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) local.get 0 local.get 1 i32.const 2 i32.shl i32.add v128.load local.set 2 local.get 0 local.get 1 i32.const 4 i32.add i32.const 2 i32.shl i32.add v128.load local.set 3 local.get 0 local.get 1 i32.const 8 i32.add i32.const 2 i32.shl i32.add v128.load local.set 4 local.get 0 local.get 1 i32.const 12 i32.add i32.const 2 i32.shl i32.add v128.load local.set 5 local.get 5 local.get 5 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 6 local.get 4 local.get 4 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 7 local.get 2 local.get 6 i32x4.max_s local.set 8 local.get 2 local.get 6 i32x4.min_s local.set 2 local.get 3 local.get 7 i32x4.max_s local.set 9 local.get 3 local.get 7 i32x4.min_s local.set 3 local.get 2 local.get 3 i32x4.max_s local.set 4 local.get 2 local.get 3 i32x4.min_s local.set 2 local.get 4 local.set 3 local.get 8 local.get 9 i32x4.max_s local.set 4 local.get 8 local.get 9 i32x4.min_s local.set 8 local.get 4 local.set 9 local.get 2 local.get 2 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 4 local.get 2 local.get 4 i32x4.min_s local.set 5 local.get 2 local.get 4 i32x4.max_s local.set 6 local.get 5 local.get 6 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 2 local.get 3 local.get 3 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 4 local.get 3 local.get 4 i32x4.min_s local.set 5 local.get 3 local.get 4 i32x4.max_s local.set 6 local.get 5 local.get 6 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 3 local.get 8 local.get 8 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 4 local.get 8 local.get 4 i32x4.min_s local.set 5 local.get 8 local.get 4 i32x4.max_s local.set 6 local.get 5 local.get 6 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 8 local.get 9 local.get 9 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 4 local.get 9 local.get 4 i32x4.min_s local.set 5 local.get 9 local.get 4 i32x4.max_s local.set 6 local.get 5 local.get 6 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 9 local.get 2 local.get 2 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 4 local.get 2 local.get 4 i32x4.min_s local.set 5 local.get 2 local.get 4 i32x4.max_s local.set 6 local.get 5 local.get 6 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 2 local.get 3 local.get 3 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 4 local.get 3 local.get 4 i32x4.min_s local.set 5 local.get 3 local.get 4 i32x4.max_s local.set 6 local.get 5 local.get 6 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 3 local.get 8 local.get 8 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 4 local.get 8 local.get 4 i32x4.min_s local.set 5 local.get 8 local.get 4 i32x4.max_s local.set 6 local.get 5 local.get 6 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 8 local.get 9 local.get 9 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 4 local.get 9 local.get 4 i32x4.min_s local.set 5 local.get 9 local.get 4 i32x4.max_s local.set 6 local.get 5 local.get 6 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 9 local.get 0 local.get 1 i32.const 2 i32.shl i32.add local.get 2 v128.store local.get 0 local.get 1 i32.const 4 i32.add i32.const 2 i32.shl i32.add local.get 3 v128.store local.get 0 local.get 1 i32.const 8 i32.add i32.const 2 i32.shl i32.add local.get 8 v128.store local.get 0 local.get 1 i32.const 12 i32.add i32.const 2 i32.shl i32.add local.get 9 v128.store)

///|
pub fn bitonic_merge16_i32(arr : FixedArray[Int], off : Int) -> Unit {
  if off + 16 > arr.length() || off < 0 {
    return
  }
  bitonic_merge16_i32_v128(arr, off)
}

///|
/// Sort 16 elements fully in SIMD: 4 leaf sort4 + 2 bitonic_merge8 (4+4 → 8)
/// + 1 bitonic_merge16 (8+8 → 16). No scalar merge.
pub fn sort16_i32(arr : FixedArray[Int], off : Int) -> Unit {
  if off + 16 > arr.length() || off < 0 {
    return
  }
  sort4_i32(arr, off + 0)
  sort4_i32(arr, off + 4)
  sort4_i32(arr, off + 8)
  sort4_i32(arr, off + 12)
  bitonic_merge8_i32(arr, off + 0)
  bitonic_merge8_i32(arr, off + 8)
  bitonic_merge16_i32(arr, off)
}

///|
/// Bitonic merge of two sorted 16-element slices (arr[off..off+16] and
/// arr[off+16..off+32]) into one sorted 32-element block, in place. 5
/// stages: distance 16 (reverse B + lane-wise min/max with A), distance 8
/// (between v128 pairs within each half), distance 4 (between v128 pairs
/// within each 8-block), distance 2 and 1 (within each v128).
#borrow(arr)
fn bitonic_merge32_i32_v128(arr : FixedArray[Int], off : Int) -> Unit =
  #|(func (param i32) (param i32) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) local.get 0 local.get 1 i32.const 2 i32.shl i32.add v128.load local.set 2 local.get 0 local.get 1 i32.const 4 i32.add i32.const 2 i32.shl i32.add v128.load local.set 3 local.get 0 local.get 1 i32.const 8 i32.add i32.const 2 i32.shl i32.add v128.load local.set 4 local.get 0 local.get 1 i32.const 12 i32.add i32.const 2 i32.shl i32.add v128.load local.set 5 local.get 0 local.get 1 i32.const 16 i32.add i32.const 2 i32.shl i32.add v128.load local.set 6 local.get 0 local.get 1 i32.const 20 i32.add i32.const 2 i32.shl i32.add v128.load local.set 7 local.get 0 local.get 1 i32.const 24 i32.add i32.const 2 i32.shl i32.add v128.load local.set 8 local.get 0 local.get 1 i32.const 28 i32.add i32.const 2 i32.shl i32.add v128.load local.set 9 local.get 9 local.get 9 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 10 local.get 8 local.get 8 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 11 local.get 7 local.get 7 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 12 local.get 6 local.get 6 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 13 local.get 2 local.get 10 i32x4.max_s local.set 6 local.get 2 local.get 10 i32x4.min_s local.set 2 local.get 3 local.get 11 i32x4.max_s local.set 7 local.get 3 local.get 11 i32x4.min_s local.set 3 local.get 4 local.get 12 i32x4.max_s local.set 8 local.get 4 local.get 12 i32x4.min_s local.set 4 local.get 5 local.get 13 i32x4.max_s local.set 9 local.get 5 local.get 13 i32x4.min_s local.set 5 local.get 2 local.get 4 i32x4.max_s local.set 10 local.get 2 local.get 4 i32x4.min_s local.set 2 local.get 10 local.set 4 local.get 3 local.get 5 i32x4.max_s local.set 10 local.get 3 local.get 5 i32x4.min_s local.set 3 local.get 10 local.set 5 local.get 6 local.get 8 i32x4.max_s local.set 10 local.get 6 local.get 8 i32x4.min_s local.set 6 local.get 10 local.set 8 local.get 7 local.get 9 i32x4.max_s local.set 10 local.get 7 local.get 9 i32x4.min_s local.set 7 local.get 10 local.set 9 local.get 2 local.get 3 i32x4.max_s local.set 10 local.get 2 local.get 3 i32x4.min_s local.set 2 local.get 10 local.set 3 local.get 4 local.get 5 i32x4.max_s local.set 10 local.get 4 local.get 5 i32x4.min_s local.set 4 local.get 10 local.set 5 local.get 6 local.get 7 i32x4.max_s local.set 10 local.get 6 local.get 7 i32x4.min_s local.set 6 local.get 10 local.set 7 local.get 8 local.get 9 i32x4.max_s local.set 10 local.get 8 local.get 9 i32x4.min_s local.set 8 local.get 10 local.set 9 local.get 2 local.get 2 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 10 local.get 2 local.get 10 i32x4.min_s local.set 11 local.get 2 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 2 local.get 3 local.get 3 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 10 local.get 3 local.get 10 i32x4.min_s local.set 11 local.get 3 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 3 local.get 4 local.get 4 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 10 local.get 4 local.get 10 i32x4.min_s local.set 11 local.get 4 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 4 local.get 5 local.get 5 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 10 local.get 5 local.get 10 i32x4.min_s local.set 11 local.get 5 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 5 local.get 6 local.get 6 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 10 local.get 6 local.get 10 i32x4.min_s local.set 11 local.get 6 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 6 local.get 7 local.get 7 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 10 local.get 7 local.get 10 i32x4.min_s local.set 11 local.get 7 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 7 local.get 8 local.get 8 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 10 local.get 8 local.get 10 i32x4.min_s local.set 11 local.get 8 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 8 local.get 9 local.get 9 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 10 local.get 9 local.get 10 i32x4.min_s local.set 11 local.get 9 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 9 local.get 2 local.get 2 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 10 local.get 2 local.get 10 i32x4.min_s local.set 11 local.get 2 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 2 local.get 3 local.get 3 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 10 local.get 3 local.get 10 i32x4.min_s local.set 11 local.get 3 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 3 local.get 4 local.get 4 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 10 local.get 4 local.get 10 i32x4.min_s local.set 11 local.get 4 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 4 local.get 5 local.get 5 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 10 local.get 5 local.get 10 i32x4.min_s local.set 11 local.get 5 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 5 local.get 6 local.get 6 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 10 local.get 6 local.get 10 i32x4.min_s local.set 11 local.get 6 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 6 local.get 7 local.get 7 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 10 local.get 7 local.get 10 i32x4.min_s local.set 11 local.get 7 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 7 local.get 8 local.get 8 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 10 local.get 8 local.get 10 i32x4.min_s local.set 11 local.get 8 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 8 local.get 9 local.get 9 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 10 local.get 9 local.get 10 i32x4.min_s local.set 11 local.get 9 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 9 local.get 0 local.get 1 i32.const 2 i32.shl i32.add local.get 2 v128.store local.get 0 local.get 1 i32.const 4 i32.add i32.const 2 i32.shl i32.add local.get 3 v128.store local.get 0 local.get 1 i32.const 8 i32.add i32.const 2 i32.shl i32.add local.get 4 v128.store local.get 0 local.get 1 i32.const 12 i32.add i32.const 2 i32.shl i32.add local.get 5 v128.store local.get 0 local.get 1 i32.const 16 i32.add i32.const 2 i32.shl i32.add local.get 6 v128.store local.get 0 local.get 1 i32.const 20 i32.add i32.const 2 i32.shl i32.add local.get 7 v128.store local.get 0 local.get 1 i32.const 24 i32.add i32.const 2 i32.shl i32.add local.get 8 v128.store local.get 0 local.get 1 i32.const 28 i32.add i32.const 2 i32.shl i32.add local.get 9 v128.store)

///|
pub fn bitonic_merge32_i32(arr : FixedArray[Int], off : Int) -> Unit {
  if off + 32 > arr.length() || off < 0 {
    return
  }
  bitonic_merge32_i32_v128(arr, off)
}

///|
/// Bitonic merge of two sorted 32-element slices into one sorted 64-element
/// block, in place. 6 stages on 16 v128 registers (8 L + 8 H), with 3 v128
/// scratch slots. Locals 2..9 hold L (low half), 18..25 hold H (high half);
/// locals 10..17 hold B during stage 1, then go unused.
#borrow(arr)
fn bitonic_merge64_i32_v128(arr : FixedArray[Int], off : Int) -> Unit =
  #|(func (param i32) (param i32) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) local.get 0 local.get 1 i32.const 2 i32.shl i32.add v128.load local.set 2 local.get 0 local.get 1 i32.const 4 i32.add i32.const 2 i32.shl i32.add v128.load local.set 3 local.get 0 local.get 1 i32.const 8 i32.add i32.const 2 i32.shl i32.add v128.load local.set 4 local.get 0 local.get 1 i32.const 12 i32.add i32.const 2 i32.shl i32.add v128.load local.set 5 local.get 0 local.get 1 i32.const 16 i32.add i32.const 2 i32.shl i32.add v128.load local.set 6 local.get 0 local.get 1 i32.const 20 i32.add i32.const 2 i32.shl i32.add v128.load local.set 7 local.get 0 local.get 1 i32.const 24 i32.add i32.const 2 i32.shl i32.add v128.load local.set 8 local.get 0 local.get 1 i32.const 28 i32.add i32.const 2 i32.shl i32.add v128.load local.set 9 local.get 0 local.get 1 i32.const 32 i32.add i32.const 2 i32.shl i32.add v128.load local.set 10 local.get 0 local.get 1 i32.const 36 i32.add i32.const 2 i32.shl i32.add v128.load local.set 11 local.get 0 local.get 1 i32.const 40 i32.add i32.const 2 i32.shl i32.add v128.load local.set 12 local.get 0 local.get 1 i32.const 44 i32.add i32.const 2 i32.shl i32.add v128.load local.set 13 local.get 0 local.get 1 i32.const 48 i32.add i32.const 2 i32.shl i32.add v128.load local.set 14 local.get 0 local.get 1 i32.const 52 i32.add i32.const 2 i32.shl i32.add v128.load local.set 15 local.get 0 local.get 1 i32.const 56 i32.add i32.const 2 i32.shl i32.add v128.load local.set 16 local.get 0 local.get 1 i32.const 60 i32.add i32.const 2 i32.shl i32.add v128.load local.set 17 local.get 17 local.get 17 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 26 local.get 2 local.get 26 i32x4.max_s local.set 18 local.get 2 local.get 26 i32x4.min_s local.set 2 local.get 16 local.get 16 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 26 local.get 3 local.get 26 i32x4.max_s local.set 19 local.get 3 local.get 26 i32x4.min_s local.set 3 local.get 15 local.get 15 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 26 local.get 4 local.get 26 i32x4.max_s local.set 20 local.get 4 local.get 26 i32x4.min_s local.set 4 local.get 14 local.get 14 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 26 local.get 5 local.get 26 i32x4.max_s local.set 21 local.get 5 local.get 26 i32x4.min_s local.set 5 local.get 13 local.get 13 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 26 local.get 6 local.get 26 i32x4.max_s local.set 22 local.get 6 local.get 26 i32x4.min_s local.set 6 local.get 12 local.get 12 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 26 local.get 7 local.get 26 i32x4.max_s local.set 23 local.get 7 local.get 26 i32x4.min_s local.set 7 local.get 11 local.get 11 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 26 local.get 8 local.get 26 i32x4.max_s local.set 24 local.get 8 local.get 26 i32x4.min_s local.set 8 local.get 10 local.get 10 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 26 local.get 9 local.get 26 i32x4.max_s local.set 25 local.get 9 local.get 26 i32x4.min_s local.set 9 local.get 2 local.get 6 i32x4.max_s local.set 26 local.get 2 local.get 6 i32x4.min_s local.set 2 local.get 26 local.set 6 local.get 3 local.get 7 i32x4.max_s local.set 26 local.get 3 local.get 7 i32x4.min_s local.set 3 local.get 26 local.set 7 local.get 4 local.get 8 i32x4.max_s local.set 26 local.get 4 local.get 8 i32x4.min_s local.set 4 local.get 26 local.set 8 local.get 5 local.get 9 i32x4.max_s local.set 26 local.get 5 local.get 9 i32x4.min_s local.set 5 local.get 26 local.set 9 local.get 18 local.get 22 i32x4.max_s local.set 26 local.get 18 local.get 22 i32x4.min_s local.set 18 local.get 26 local.set 22 local.get 19 local.get 23 i32x4.max_s local.set 26 local.get 19 local.get 23 i32x4.min_s local.set 19 local.get 26 local.set 23 local.get 20 local.get 24 i32x4.max_s local.set 26 local.get 20 local.get 24 i32x4.min_s local.set 20 local.get 26 local.set 24 local.get 21 local.get 25 i32x4.max_s local.set 26 local.get 21 local.get 25 i32x4.min_s local.set 21 local.get 26 local.set 25 local.get 2 local.get 4 i32x4.max_s local.set 26 local.get 2 local.get 4 i32x4.min_s local.set 2 local.get 26 local.set 4 local.get 3 local.get 5 i32x4.max_s local.set 26 local.get 3 local.get 5 i32x4.min_s local.set 3 local.get 26 local.set 5 local.get 6 local.get 8 i32x4.max_s local.set 26 local.get 6 local.get 8 i32x4.min_s local.set 6 local.get 26 local.set 8 local.get 7 local.get 9 i32x4.max_s local.set 26 local.get 7 local.get 9 i32x4.min_s local.set 7 local.get 26 local.set 9 local.get 18 local.get 20 i32x4.max_s local.set 26 local.get 18 local.get 20 i32x4.min_s local.set 18 local.get 26 local.set 20 local.get 19 local.get 21 i32x4.max_s local.set 26 local.get 19 local.get 21 i32x4.min_s local.set 19 local.get 26 local.set 21 local.get 22 local.get 24 i32x4.max_s local.set 26 local.get 22 local.get 24 i32x4.min_s local.set 22 local.get 26 local.set 24 local.get 23 local.get 25 i32x4.max_s local.set 26 local.get 23 local.get 25 i32x4.min_s local.set 23 local.get 26 local.set 25 local.get 2 local.get 3 i32x4.max_s local.set 26 local.get 2 local.get 3 i32x4.min_s local.set 2 local.get 26 local.set 3 local.get 4 local.get 5 i32x4.max_s local.set 26 local.get 4 local.get 5 i32x4.min_s local.set 4 local.get 26 local.set 5 local.get 6 local.get 7 i32x4.max_s local.set 26 local.get 6 local.get 7 i32x4.min_s local.set 6 local.get 26 local.set 7 local.get 8 local.get 9 i32x4.max_s local.set 26 local.get 8 local.get 9 i32x4.min_s local.set 8 local.get 26 local.set 9 local.get 18 local.get 19 i32x4.max_s local.set 26 local.get 18 local.get 19 i32x4.min_s local.set 18 local.get 26 local.set 19 local.get 20 local.get 21 i32x4.max_s local.set 26 local.get 20 local.get 21 i32x4.min_s local.set 20 local.get 26 local.set 21 local.get 22 local.get 23 i32x4.max_s local.set 26 local.get 22 local.get 23 i32x4.min_s local.set 22 local.get 26 local.set 23 local.get 24 local.get 25 i32x4.max_s local.set 26 local.get 24 local.get 25 i32x4.min_s local.set 24 local.get 26 local.set 25 local.get 2 local.get 2 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 2 local.get 26 i32x4.min_s local.set 27 local.get 2 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 2 local.get 3 local.get 3 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 3 local.get 26 i32x4.min_s local.set 27 local.get 3 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 3 local.get 4 local.get 4 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 4 local.get 26 i32x4.min_s local.set 27 local.get 4 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 4 local.get 5 local.get 5 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 5 local.get 26 i32x4.min_s local.set 27 local.get 5 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 5 local.get 6 local.get 6 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 6 local.get 26 i32x4.min_s local.set 27 local.get 6 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 6 local.get 7 local.get 7 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 7 local.get 26 i32x4.min_s local.set 27 local.get 7 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 7 local.get 8 local.get 8 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 8 local.get 26 i32x4.min_s local.set 27 local.get 8 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 8 local.get 9 local.get 9 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 9 local.get 26 i32x4.min_s local.set 27 local.get 9 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 9 local.get 18 local.get 18 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 18 local.get 26 i32x4.min_s local.set 27 local.get 18 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 18 local.get 19 local.get 19 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 19 local.get 26 i32x4.min_s local.set 27 local.get 19 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 19 local.get 20 local.get 20 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 20 local.get 26 i32x4.min_s local.set 27 local.get 20 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 20 local.get 21 local.get 21 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 21 local.get 26 i32x4.min_s local.set 27 local.get 21 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 21 local.get 22 local.get 22 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 22 local.get 26 i32x4.min_s local.set 27 local.get 22 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 22 local.get 23 local.get 23 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 23 local.get 26 i32x4.min_s local.set 27 local.get 23 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 23 local.get 24 local.get 24 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 24 local.get 26 i32x4.min_s local.set 27 local.get 24 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 24 local.get 25 local.get 25 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 25 local.get 26 i32x4.min_s local.set 27 local.get 25 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 25 local.get 2 local.get 2 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 2 local.get 26 i32x4.min_s local.set 27 local.get 2 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 2 local.get 3 local.get 3 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 3 local.get 26 i32x4.min_s local.set 27 local.get 3 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 3 local.get 4 local.get 4 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 4 local.get 26 i32x4.min_s local.set 27 local.get 4 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 4 local.get 5 local.get 5 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 5 local.get 26 i32x4.min_s local.set 27 local.get 5 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 5 local.get 6 local.get 6 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 6 local.get 26 i32x4.min_s local.set 27 local.get 6 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 6 local.get 7 local.get 7 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 7 local.get 26 i32x4.min_s local.set 27 local.get 7 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 7 local.get 8 local.get 8 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 8 local.get 26 i32x4.min_s local.set 27 local.get 8 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 8 local.get 9 local.get 9 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 9 local.get 26 i32x4.min_s local.set 27 local.get 9 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 9 local.get 18 local.get 18 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 18 local.get 26 i32x4.min_s local.set 27 local.get 18 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 18 local.get 19 local.get 19 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 19 local.get 26 i32x4.min_s local.set 27 local.get 19 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 19 local.get 20 local.get 20 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 20 local.get 26 i32x4.min_s local.set 27 local.get 20 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 20 local.get 21 local.get 21 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 21 local.get 26 i32x4.min_s local.set 27 local.get 21 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 21 local.get 22 local.get 22 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 22 local.get 26 i32x4.min_s local.set 27 local.get 22 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 22 local.get 23 local.get 23 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 23 local.get 26 i32x4.min_s local.set 27 local.get 23 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 23 local.get 24 local.get 24 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 24 local.get 26 i32x4.min_s local.set 27 local.get 24 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 24 local.get 25 local.get 25 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 25 local.get 26 i32x4.min_s local.set 27 local.get 25 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 25 local.get 0 local.get 1 i32.const 2 i32.shl i32.add local.get 2 v128.store local.get 0 local.get 1 i32.const 4 i32.add i32.const 2 i32.shl i32.add local.get 3 v128.store local.get 0 local.get 1 i32.const 8 i32.add i32.const 2 i32.shl i32.add local.get 4 v128.store local.get 0 local.get 1 i32.const 12 i32.add i32.const 2 i32.shl i32.add local.get 5 v128.store local.get 0 local.get 1 i32.const 16 i32.add i32.const 2 i32.shl i32.add local.get 6 v128.store local.get 0 local.get 1 i32.const 20 i32.add i32.const 2 i32.shl i32.add local.get 7 v128.store local.get 0 local.get 1 i32.const 24 i32.add i32.const 2 i32.shl i32.add local.get 8 v128.store local.get 0 local.get 1 i32.const 28 i32.add i32.const 2 i32.shl i32.add local.get 9 v128.store local.get 0 local.get 1 i32.const 32 i32.add i32.const 2 i32.shl i32.add local.get 18 v128.store local.get 0 local.get 1 i32.const 36 i32.add i32.const 2 i32.shl i32.add local.get 19 v128.store local.get 0 local.get 1 i32.const 40 i32.add i32.const 2 i32.shl i32.add local.get 20 v128.store local.get 0 local.get 1 i32.const 44 i32.add i32.const 2 i32.shl i32.add local.get 21 v128.store local.get 0 local.get 1 i32.const 48 i32.add i32.const 2 i32.shl i32.add local.get 22 v128.store local.get 0 local.get 1 i32.const 52 i32.add i32.const 2 i32.shl i32.add local.get 23 v128.store local.get 0 local.get 1 i32.const 56 i32.add i32.const 2 i32.shl i32.add local.get 24 v128.store local.get 0 local.get 1 i32.const 60 i32.add i32.const 2 i32.shl i32.add local.get 25 v128.store)

///|
pub fn bitonic_merge64_i32(arr : FixedArray[Int], off : Int) -> Unit {
  if off + 64 > arr.length() || off < 0 {
    return
  }
  bitonic_merge64_i32_v128(arr, off)
}

///|
/// General-purpose sort: SIMD-leaf bottom-up merge sort.
/// Leaf (16-element blocks): sort16_i32 (fully SIMD).
/// Tail block (< 16): fall back to the built-in scalar sort.
/// Merge ladder (16+16 → 32, 32+32 → 64, ...): scalar merge2 between
/// ping-pong buffers; a future SIMD bitonic merge would replace this.
pub fn sort_i32(arr : FixedArray[Int]) -> Unit {
  let n = arr.length()
  if n <= 1 {
    return
  }
  if n <= 16 {
    // tiny case: leaf-only path.
    if n == 16 {
      sort16_i32(arr, 0)
    } else if n == 8 {
      sort4_i32(arr, 0)
      sort4_i32(arr, 4)
      bitonic_merge8_i32(arr, 0)
    } else if n == 4 {
      sort4_i32(arr, 0)
    } else {
      @internal.scalar_sort_int(arr)
    }
    return
  }

  // 1. leaf-sort each aligned 16-block.
  let leaf_end = n / 16 * 16
  for i = 0; i < leaf_end; i = i + 16 {
    sort16_i32(arr, i)
  }
  // 1b. tail < 16: copy out, scalar sort, copy back.
  let tail_len = n - leaf_end
  if tail_len > 0 {
    let tail = FixedArray::make(tail_len, 0)
    for k in 0..<tail_len {
      tail[k] = arr[leaf_end + k]
    }
    @internal.scalar_sort_int(tail)
    for k in 0..<tail_len {
      arr[leaf_end + k] = tail[k]
    }
  }

  // 2. merge ladder.
  // 2a. width = 16 pass: in-place SIMD bitonic_merge32 for full 32-blocks.
  //     Tail (last sub-block of 16-31 elements) falls back to scalar.
  let tmp = FixedArray::make(n, 0)
  let mut pos = 0
  while pos + 32 <= n {
    bitonic_merge32_i32(arr, pos)
    pos = pos + 32
  }
  // last block: pos..n is either a single 16-block (already sorted, skip) or
  // a 16-block plus a tail of 0..15 already-sorted elements; merge into arr
  // via tmp + copy-back.
  if pos + 16 < n {
    let tail_total = n - pos
    @internal.scalar_merge2_int(
      arr, pos, 16, arr, pos + 16, tail_total - 16, tmp, pos,
    )
    for k = pos; k < n; k = k + 1 {
      arr[k] = tmp[k]
    }
  }

  // 2b. width = 32 pass: in-place SIMD bitonic_merge64 for full 64-blocks.
  let mut pos2 = 0
  while pos2 + 64 <= n {
    bitonic_merge64_i32(arr, pos2)
    pos2 = pos2 + 64
  }
  // tail: pos2..n is either a single 32-block (skip) or a 32-block plus a
  // (16-31)-element sorted tail; scalar-merge into arr via tmp + copy-back.
  if pos2 + 32 < n {
    let tail_total = n - pos2
    @internal.scalar_merge2_int(
      arr, pos2, 32, arr, pos2 + 32, tail_total - 32, tmp, pos2,
    )
    for k = pos2; k < n; k = k + 1 {
      arr[k] = tmp[k]
    }
  }

  // 2c. width >= 64 pass: scalar merge ladder with ping-pong between arr and tmp.
  let mut width = 64
  let mut src_is_arr = true
  while width < n {
    let merged = width * 2
    let mut i = 0
    while i < n {
      let a_off = i
      let a_end = if a_off + width <= n { a_off + width } else { n }
      let b_off = a_end
      let b_end = if b_off + width <= n { b_off + width } else { n }
      if b_off >= n {
        // no B half: copy A as-is into dst.
        if src_is_arr {
          for k = a_off; k < a_end; k = k + 1 {
            tmp[k] = arr[k]
          }
        } else {
          for k = a_off; k < a_end; k = k + 1 {
            arr[k] = tmp[k]
          }
        }
      } else if src_is_arr {
        @internal.scalar_merge2_int(
          arr, a_off, a_end - a_off, arr, b_off, b_end - b_off, tmp, a_off,
        )
      } else {
        @internal.scalar_merge2_int(
          tmp, a_off, a_end - a_off, tmp, b_off, b_end - b_off, arr, a_off,
        )
      }
      i = i + merged
    }
    width = merged
    src_is_arr = !src_is_arr
  }

  // 3. if final pass landed in tmp, copy back into arr.
  if !src_is_arr {
    for k in 0..<n {
      arr[k] = tmp[k]
    }
  }
}

// --- Float (f32) SIMD ops via f32x4 (4-way) over byte buffers ---

///|
#borrow(a, b, out)
fn add_f32_v128(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load f32x4.add v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add f32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add f32.load f32.add f32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

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
#borrow(a, b, out)
fn sub_f32_v128(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load f32x4.sub v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add f32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add f32.load f32.sub f32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

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
#borrow(a, b, out)
fn div_f32_v128(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load f32x4.div v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add f32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add f32.load f32.div f32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

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
#borrow(a, out)
fn sqrt_f32_v128(
  a : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (local i32) (local i32) local.get 2 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 1 local.get 3 i32.const 2 i32.shl i32.add local.get 0 local.get 3 i32.const 2 i32.shl i32.add v128.load f32x4.sqrt v128.store local.get 3 i32.const 4 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 1 local.get 3 i32.const 2 i32.shl i32.add local.get 0 local.get 3 i32.const 2 i32.shl i32.add f32.load f32.sqrt f32.store local.get 3 i32.const 1 i32.add local.set 3 br 0 end end)

///|
pub fn sqrt_f32(
  a : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  if n == 0 {
    return
  }
  sqrt_f32_v128(a, out, n)
}

///|
/// min element-wise — uses select+f32.lt for scalar tail because the
/// Dwarfsm parser rejects `f32.min` (same hole as `f64.min`).
#borrow(a, b, out)
fn min_elem_f32_v128(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load f32x4.min v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add f32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add f32.load local.get 0 local.get 4 i32.const 2 i32.shl i32.add f32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add f32.load f32.lt select f32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

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
#borrow(a, b, out)
fn max_elem_f32_v128(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load f32x4.max v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add f32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add f32.load local.get 0 local.get 4 i32.const 2 i32.shl i32.add f32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add f32.load f32.gt select f32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

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
#borrow(a, b, out)
fn mul_f32_v128(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load f32x4.mul v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add f32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add f32.load f32.mul f32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

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
#borrow(arr)
fn sum_f32_v128(arr : FixedArray[Byte], n : Int) -> Float =
  #|(func (param i32) (param i32) (result f32) (local i32) (local i32) (local f32) (local v128) i32.const 0 f32.convert_i32_s f32x4.splat local.set 5 local.get 1 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 2 i32.const 2 i32.shl i32.add v128.load f32x4.add local.set 5 local.get 2 i32.const 4 i32.add local.set 2 br 0 end end local.get 5 f32x4.extract_lane 0 local.get 5 f32x4.extract_lane 1 f32.add local.get 5 f32x4.extract_lane 2 f32.add local.get 5 f32x4.extract_lane 3 f32.add local.set 4 block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.const 2 i32.shl i32.add f32.load f32.add local.set 4 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end local.get 4)

///|
pub fn sum_f32(arr : FixedArray[Byte], n : Int) -> Float {
  if n == 0 {
    return (0.0 : Float)
  }
  sum_f32_v128(arr, n)
}

///|
#borrow(a, b)
fn dot_f32_v128(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  n : Int,
) -> Float =
  #|(func (param i32) (param i32) (param i32) (result f32) (local i32) (local i32) (local f32) (local v128) i32.const 0 f32.convert_i32_s f32x4.splat local.set 6 local.get 2 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 6 local.get 0 local.get 3 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 3 i32.const 2 i32.shl i32.add v128.load f32x4.mul f32x4.add local.set 6 local.get 3 i32.const 4 i32.add local.set 3 br 0 end end local.get 6 f32x4.extract_lane 0 local.get 6 f32x4.extract_lane 1 f32.add local.get 6 f32x4.extract_lane 2 f32.add local.get 6 f32x4.extract_lane 3 f32.add local.set 5 block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 3 i32.const 2 i32.shl i32.add f32.load local.get 1 local.get 3 i32.const 2 i32.shl i32.add f32.load f32.mul f32.add local.set 5 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end local.get 5)

///|
pub fn dot_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  n : Int,
) -> Float {
  if n == 0 {
    return (0.0 : Float)
  }
  dot_f32_v128(a, b, n)
}

// --- Double (f64) SIMD ops via f64x2 (2-way) ---

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
#borrow(a, b)
fn simd_dot_f64_v128(
  a : FixedArray[Double],
  b : FixedArray[Double],
  len : Int,
) -> Double =
  #|(func (param i32) (param i32) (param i32) (result f64) (local i32) (local i32) (local f64) (local v128) i32.const 0 f64.convert_i32_s f64x2.splat local.set 6 local.get 2 i32.const 1 i32.shr_s i32.const 1 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 6 local.get 0 local.get 3 i32.const 3 i32.shl i32.add v128.load local.get 1 local.get 3 i32.const 3 i32.shl i32.add v128.load f64x2.mul f64x2.add local.set 6 local.get 3 i32.const 2 i32.add local.set 3 br 0 end end local.get 6 f64x2.extract_lane 0 local.get 6 f64x2.extract_lane 1 f64.add local.set 5 block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 3 i32.const 3 i32.shl i32.add f64.load local.get 1 local.get 3 i32.const 3 i32.shl i32.add f64.load f64.mul f64.add local.set 5 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end local.get 5)

///|
pub fn dot_f64(
  a : FixedArray[Double],
  b : FixedArray[Double],
) -> Double {
  let len = a.length()
  if len == 0 {
    return 0.0
  }
  simd_dot_f64_v128(a, b, len)
}

///|
/// SIMD Adler-32 - wasm target: v128 16-byte chunks with scalar tail.
/// All arithmetic uses UInt so the unsigned i32 wrap inside one block
/// (zlib's NMAX=5552) carries safely; mod is applied at block boundaries.
pub fn adler32_bytes(data : FixedArray[Byte]) -> UInt {
  let mod_adler : UInt = 65521
  let len = data.length()
  let mut a : UInt = 1
  let mut b : UInt = 0
  let mut i = 0
  while i < len {
    let remaining = len - i
    let block = if remaining > 5552 { 5552 } else { remaining }
    let block_end = i + block
    let simd_chunks = block / 16
    if simd_chunks > 0 {
      let packed = adler32_chunks(
        data,
        i,
        simd_chunks,
        adler_weights,
        a.reinterpret_as_int(),
        b.reinterpret_as_int(),
      )
      a = (packed & 0xFFFFFFFFL).to_int().reinterpret_as_uint()
      b = (packed.reinterpret_as_uint64() >> 32).to_int().reinterpret_as_uint()
      i = i + simd_chunks * 16
    }
    while i < block_end {
      a = a + data[i].to_int().reinterpret_as_uint()
      b = b + a
      i = i + 1
    }
    a = a % mod_adler
    b = b % mod_adler
  }
  (b << 16) | a
}
