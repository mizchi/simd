///|
#borrow(arr)
fn simd_sum_v128(arr : FixedArray[Int], len : Int) -> Int =
  #|(func (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local v128) i32.const 0 i32x4.splat local.set 5 local.get 1 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 2 i32.const 2 i32.shl i32.add v128.load i32x4.add local.set 5 local.get 2 i32.const 4 i32.add local.set 2 br 0 end end local.get 5 i32x4.extract_lane 0 local.get 5 i32x4.extract_lane 1 i32.add local.get 5 i32x4.extract_lane 2 i32.add local.get 5 i32x4.extract_lane 3 i32.add local.set 4 block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.const 2 i32.shl i32.add i32.load i32.add local.set 4 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end local.get 4)

///|
/// SIMD sum - wasm target: v128 i32x4.add accumulator with scalar tail.
pub fn simd_sum(arr : FixedArray[Int]) -> Int {
  let len = arr.length()
  if len == 0 {
    return 0
  }
  simd_sum_v128(arr, len)
}

///|
#borrow(a, b)
fn simd_dot_v128(a : FixedArray[Int], b : FixedArray[Int], len : Int) -> Int =
  #|(func (param i32) (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local v128) i32.const 0 i32x4.splat local.set 6 local.get 2 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 6 local.get 0 local.get 3 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 3 i32.const 2 i32.shl i32.add v128.load i32x4.mul i32x4.add local.set 6 local.get 3 i32.const 4 i32.add local.set 3 br 0 end end local.get 6 i32x4.extract_lane 0 local.get 6 i32x4.extract_lane 1 i32.add local.get 6 i32x4.extract_lane 2 i32.add local.get 6 i32x4.extract_lane 3 i32.add local.set 5 block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 3 i32.const 2 i32.shl i32.add i32.load local.get 1 local.get 3 i32.const 2 i32.shl i32.add i32.load i32.mul i32.add local.set 5 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end local.get 5)

///|
/// SIMD dot product - wasm target: i32x4.mul + i32x4.add, scalar tail.
pub fn simd_dot_product(a : FixedArray[Int], b : FixedArray[Int]) -> Int {
  let len = a.length()
  if len == 0 {
    return 0
  }
  simd_dot_v128(a, b, len)
}

///|
#borrow(a, b, out)
fn simd_add_v128(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load i32x4.add v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add i32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add i32.load i32.add i32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
/// SIMD element-wise add - wasm target: i32x4.add with scalar tail.
pub fn simd_add(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  simd_add_v128(a, b, out, len)
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
fn simd_adler32_chunks(
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
fn simd_min_v128(arr : FixedArray[Int], len : Int) -> Int =
  #|(func (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local v128) local.get 0 i32.load i32x4.splat local.set 5 local.get 1 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 2 i32.const 2 i32.shl i32.add v128.load i32x4.min_s local.set 5 local.get 2 i32.const 4 i32.add local.set 2 br 0 end end local.get 5 i32x4.extract_lane 0 local.set 4 local.get 4 local.get 5 i32x4.extract_lane 1 local.get 4 local.get 5 i32x4.extract_lane 1 i32.lt_s select local.set 4 local.get 4 local.get 5 i32x4.extract_lane 2 local.get 4 local.get 5 i32x4.extract_lane 2 i32.lt_s select local.set 4 local.get 4 local.get 5 i32x4.extract_lane 3 local.get 4 local.get 5 i32x4.extract_lane 3 i32.lt_s select local.set 4 block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.const 2 i32.shl i32.add i32.load local.get 4 local.get 0 local.get 2 i32.const 2 i32.shl i32.add i32.load i32.lt_s select local.set 4 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end local.get 4)

///|
/// SIMD min - wasm target: v128 i32x4.min_s with scalar reduce.
pub fn simd_min(arr : FixedArray[Int]) -> Int {
  let len = arr.length()
  if len == 0 {
    return 0
  }
  simd_min_v128(arr, len)
}

///|
#borrow(arr)
fn simd_max_v128(arr : FixedArray[Int], len : Int) -> Int =
  #|(func (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local v128) local.get 0 i32.load i32x4.splat local.set 5 local.get 1 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 2 i32.const 2 i32.shl i32.add v128.load i32x4.max_s local.set 5 local.get 2 i32.const 4 i32.add local.set 2 br 0 end end local.get 5 i32x4.extract_lane 0 local.set 4 local.get 4 local.get 5 i32x4.extract_lane 1 local.get 4 local.get 5 i32x4.extract_lane 1 i32.gt_s select local.set 4 local.get 4 local.get 5 i32x4.extract_lane 2 local.get 4 local.get 5 i32x4.extract_lane 2 i32.gt_s select local.set 4 local.get 4 local.get 5 i32x4.extract_lane 3 local.get 4 local.get 5 i32x4.extract_lane 3 i32.gt_s select local.set 4 block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.const 2 i32.shl i32.add i32.load local.get 4 local.get 0 local.get 2 i32.const 2 i32.shl i32.add i32.load i32.gt_s select local.set 4 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end local.get 4)

///|
/// SIMD max - wasm target: v128 i32x4.max_s with scalar reduce.
pub fn simd_max(arr : FixedArray[Int]) -> Int {
  let len = arr.length()
  if len == 0 {
    return 0
  }
  simd_max_v128(arr, len)
}

///|
/// argmin: scalar fallback (early-exit needs `if`/`return` opcodes which
/// the Dwarfsm parser handles less reliably; revisit if needed).
pub fn simd_argmin(arr : FixedArray[Int]) -> Int {
  scalar_argmin(arr)
}

///|
pub fn simd_argmax(arr : FixedArray[Int]) -> Int {
  scalar_argmax(arr)
}

///|
#borrow(a, b, out)
fn simd_saxpy_v128(
  k : Int,
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) (local v128) local.get 0 i32x4.splat local.set 7 local.get 4 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 6 block loop local.get 5 local.get 6 i32.ge_s br_if 1 local.get 3 local.get 5 i32.const 2 i32.shl i32.add local.get 7 local.get 1 local.get 5 i32.const 2 i32.shl i32.add v128.load i32x4.mul local.get 2 local.get 5 i32.const 2 i32.shl i32.add v128.load i32x4.add v128.store local.get 5 i32.const 4 i32.add local.set 5 br 0 end end block loop local.get 5 local.get 4 i32.ge_s br_if 1 local.get 3 local.get 5 i32.const 2 i32.shl i32.add local.get 0 local.get 1 local.get 5 i32.const 2 i32.shl i32.add i32.load i32.mul local.get 2 local.get 5 i32.const 2 i32.shl i32.add i32.load i32.add i32.store local.get 5 i32.const 1 i32.add local.set 5 br 0 end end)

///|
/// SIMD saxpy: out[i] = k * a[i] + b[i] - wasm target: v128 i32x4.mul + add.
pub fn simd_saxpy(
  k : Int,
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  simd_saxpy_v128(k, a, b, out, len)
}

///|
#borrow(data)
fn simd_popcount_bytes_v128(data : FixedArray[Byte], len : Int) -> Int =
  #|(func (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local v128) local.get 1 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 0 local.get 2 i32.add v128.load i8x16.popcnt i16x8.extadd_pairwise_i8x16_u i32x4.extadd_pairwise_i16x8_u local.set 5 local.get 4 local.get 5 i32x4.extract_lane 0 i32.add local.get 5 i32x4.extract_lane 1 i32.add local.get 5 i32x4.extract_lane 2 i32.add local.get 5 i32x4.extract_lane 3 i32.add local.set 4 local.get 2 i32.const 16 i32.add local.set 2 br 0 end end block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.add i32.load8_u i32.popcnt i32.add local.set 4 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end local.get 4)

///|
/// SIMD popcount of a byte buffer - wasm target: i8x16.popcnt + horizontal sum.
pub fn simd_popcount_bytes(data : FixedArray[Byte]) -> Int {
  let len = data.length()
  if len == 0 {
    return 0
  }
  simd_popcount_bytes_v128(data, len)
}

///|
#borrow(src, dst)
fn simd_memcpy_bytes_v128(
  src : FixedArray[Byte],
  si : Int,
  dst : FixedArray[Byte],
  di : Int,
  len : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 4 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 6 block loop local.get 5 local.get 6 i32.ge_s br_if 1 local.get 2 local.get 3 i32.add local.get 5 i32.add local.get 0 local.get 1 i32.add local.get 5 i32.add v128.load v128.store local.get 5 i32.const 16 i32.add local.set 5 br 0 end end block loop local.get 5 local.get 4 i32.ge_s br_if 1 local.get 2 local.get 3 i32.add local.get 5 i32.add local.get 0 local.get 1 i32.add local.get 5 i32.add i32.load8_u i32.store8 local.get 5 i32.const 1 i32.add local.set 5 br 0 end end)

///|
/// SIMD memcpy - wasm target: 16-byte v128.load/v128.store chunks.
pub fn simd_memcpy_bytes(
  src : FixedArray[Byte],
  si : Int,
  dst : FixedArray[Byte],
  di : Int,
  len : Int,
) -> Unit {
  if len <= 0 {
    return
  }
  simd_memcpy_bytes_v128(src, si, dst, di, len)
}

///|
#borrow(dst)
fn simd_memset_bytes_v128(
  dst : FixedArray[Byte],
  off : Int,
  len : Int,
  value : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) (local v128) local.get 3 i8x16.splat local.set 6 local.get 2 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 0 local.get 1 i32.add local.get 4 i32.add local.get 6 v128.store local.get 4 i32.const 16 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 2 i32.ge_s br_if 1 local.get 0 local.get 1 i32.add local.get 4 i32.add local.get 3 i32.store8 local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
/// SIMD memset - wasm target: i8x16.splat + 16-byte v128.store chunks.
pub fn simd_memset_bytes(
  dst : FixedArray[Byte],
  off : Int,
  len : Int,
  value : Byte,
) -> Unit {
  if len <= 0 {
    return
  }
  simd_memset_bytes_v128(dst, off, len, value.to_int())
}

///|
#borrow(a, b)
fn simd_equal_bytes_v128(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  len : Int,
) -> Int =
  #|(func (param i32) (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) local.get 2 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 3 i32.add v128.load local.get 1 local.get 3 i32.add v128.load i8x16.eq i8x16.bitmask i32.const 65535 i32.ne i32.or local.set 5 local.get 3 i32.const 16 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 3 i32.add i32.load8_u local.get 1 local.get 3 i32.add i32.load8_u i32.ne i32.or local.set 5 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end local.get 5 i32.eqz)

///|
/// SIMD byte equality (first `len` bytes) - wasm target: i8x16.eq + bitmask
/// folded into a mismatch flag (no early break).
pub fn simd_equal_bytes(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  len : Int,
) -> Bool {
  if len <= 0 {
    return true
  }
  simd_equal_bytes_v128(a, b, len) != 0
}

///|
#borrow(data)
fn simd_find_byte_v128(data : FixedArray[Byte], len : Int, needle : Int) -> Int =
  #|(func (param i32) (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local i32) (local v128) local.get 2 i8x16.splat local.set 7 local.get 1 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 4 i32.const -1 local.set 5 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 0 local.get 3 i32.add v128.load local.get 7 i8x16.eq i8x16.bitmask local.set 6 local.get 3 local.get 6 i32.ctz i32.add local.get 5 local.get 5 i32.const -1 i32.eq local.get 6 i32.mul select local.set 5 local.get 3 i32.const 16 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 1 i32.ge_s br_if 1 local.get 3 local.get 5 local.get 5 i32.const -1 i32.eq local.get 0 local.get 3 i32.add i32.load8_u local.get 2 i32.eq i32.and select local.set 5 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end local.get 5)

///|
/// SIMD find first byte equal to `needle` - wasm target: i8x16.eq + bitmask
/// with select-based update (no early break).
pub fn simd_find_byte(data : FixedArray[Byte], needle : Byte) -> Int {
  let len = data.length()
  if len == 0 {
    return -1
  }
  simd_find_byte_v128(data, len, needle.to_int())
}

///|
#borrow(data)
fn simd_count_byte_v128(data : FixedArray[Byte], len : Int, needle : Int) -> Int =
  #|(func (param i32) (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local v128) local.get 2 i8x16.splat local.set 6 local.get 1 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 3 i32.add v128.load local.get 6 i8x16.eq i8x16.bitmask i32.popcnt i32.add local.set 5 local.get 3 i32.const 16 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 1 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 3 i32.add i32.load8_u local.get 2 i32.eq i32.add local.set 5 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end local.get 5)

///|
/// SIMD count of bytes equal to `needle` - wasm target: i8x16.eq + bitmask +
/// i32.popcnt.
pub fn simd_count_byte(data : FixedArray[Byte], needle : Byte) -> Int {
  let len = data.length()
  if len == 0 {
    return 0
  }
  simd_count_byte_v128(data, len, needle.to_int())
}

///|
/// SIMD Adler-32 - wasm target: v128 16-byte chunks with scalar tail.
/// All arithmetic uses UInt so the unsigned i32 wrap inside one block
/// (zlib's NMAX=5552) carries safely; mod is applied at block boundaries.
pub fn simd_adler32(data : FixedArray[Byte]) -> UInt {
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
      let packed = simd_adler32_chunks(
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
