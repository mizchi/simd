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

///|
/// Returns the byte offset of the first 16-byte chunk that contains any
/// non-ASCII byte, or `(len/16)*16` if every aligned chunk is pure ASCII.
/// Uses select-based tracking to record the first hit without early exit.
#borrow(data)
fn first_non_ascii_chunk_v128(data : FixedArray[Byte], len : Int) -> Int =
  #|(func (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local i32) local.get 1 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 3 local.get 3 local.set 4 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 0 local.get 2 i32.add v128.load i8x16.bitmask local.set 5 local.get 2 local.get 4 local.get 4 local.get 3 i32.eq local.get 5 i32.mul select local.set 4 local.get 2 i32.const 16 i32.add local.set 2 br 0 end end local.get 4)

///|
/// SIMD-accelerated UTF-8 validation. SIMD fast path skips all-ASCII chunks
/// via `first_non_ascii_chunk_v128`; the scalar state machine takes
/// over at the first byte that might be non-ASCII (which is safe because
/// no multi-byte sequence can cross into that chunk from a prior all-ASCII
/// chunk).
pub fn validate_utf8(data : FixedArray[Byte]) -> Bool {
  let len = data.length()
  let start = if len >= 16 { first_non_ascii_chunk_v128(data, len) } else { 0 }
  @internal.scalar_validate_utf8_from(data, start)
}

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
