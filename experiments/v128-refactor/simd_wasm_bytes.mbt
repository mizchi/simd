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
fn popcount_bytes_v128(data : FixedArray[Byte], len : Int) -> Int {
  let vec_len = floor_i8x16_len(len)
  let count = for i = 0, count = 0; i < vec_len; {
    let lane_counts = i8x16_popcnt_sum(fixed_i8x16_load(data, i))
    continue i + 16, count + i32x4_lane_sum(lane_counts)
  } nobreak {
    count
  }
  for i in vec_len..<len; count = count {
    continue count + data[i].to_int().popcnt()
  } nobreak {
    count
  }
}

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
fn memcpy_bytes_v128(
  src : FixedArray[Byte],
  si : Int,
  dst : FixedArray[Byte],
  di : Int,
  len : Int,
) -> Unit {
  let vec_len = floor_i8x16_len(len)
  for k = 0; k < vec_len; {
    fixed_i8x16_store(dst, di + k, fixed_i8x16_load(src, si + k))
    continue k + 16
  }
  for k in vec_len..<len {
    dst[di + k] = src[si + k]
  }
}

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
fn memset_bytes_v128(
  dst : FixedArray[Byte],
  off : Int,
  len : Int,
  value : Int,
) -> Unit {
  let vec_len = floor_i8x16_len(len)
  let value_v = i8x16_splat(value)
  for k = 0; k < vec_len; {
    fixed_i8x16_store(dst, off + k, value_v)
    continue k + 16
  }
  let value_b = value.to_byte()
  for k in vec_len..<len {
    dst[off + k] = value_b
  }
}

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
fn equal_bytes_v128(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  len : Int,
) -> Int {
  let vec_len = floor_i8x16_len(len)
  let mismatch = for i = 0, mismatch = false; i < vec_len; {
    let eq = i8x16_eq(fixed_i8x16_load(a, i), fixed_i8x16_load(b, i))
    continue i + 16, mismatch || i8x16_bitmask(eq) != 0xFFFF
  } nobreak {
    mismatch
  }
  let mismatch = for i in vec_len..<len; mismatch = mismatch {
    continue mismatch || a[i] != b[i]
  } nobreak {
    mismatch
  }
  if mismatch {
    0
  } else {
    1
  }
}

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
fn find_byte_v128(data : FixedArray[Byte], len : Int, needle : Int) -> Int {
  let vec_len = floor_i8x16_len(len)
  let needle_v = i8x16_splat(needle)
  let found = for i = 0, found = -1; i < vec_len; {
    let mask = i8x16_bitmask(i8x16_eq(fixed_i8x16_load(data, i), needle_v))
    let next = if found == -1 && mask != 0 { i + mask.ctz() } else { found }
    continue i + 16, next
  } nobreak {
    found
  }
  for i in vec_len..<len; found = found {
    let next = if found == -1 && data[i].to_int() == needle { i } else { found }
    continue next
  } nobreak {
    found
  }
}

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
fn count_byte_v128(data : FixedArray[Byte], len : Int, needle : Int) -> Int {
  let vec_len = floor_i8x16_len(len)
  let needle_v = i8x16_splat(needle)
  let count = for i = 0, count = 0; i < vec_len; {
    let mask = i8x16_bitmask(i8x16_eq(fixed_i8x16_load(data, i), needle_v))
    continue i + 16, count + mask.popcnt()
  } nobreak {
    count
  }
  for i in vec_len..<len; count = count {
    let inc = if data[i].to_int() == needle { 1 } else { 0 }
    continue count + inc
  } nobreak {
    count
  }
}

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
/// Strict UTF-8 validation (RFC 3629): also rejects overlong encodings,
/// surrogates, and code points > U+10FFFF. SIMD fast path skips all-ASCII
/// chunks, then a strict scalar pass takes over at the first possibly-non-
/// ASCII chunk boundary (safe — no multi-byte sequence crosses it).
pub fn validate_utf8_strict(data : FixedArray[Byte]) -> Bool {
  let len = data.length()
  let start = if len >= 16 { first_non_ascii_chunk_v128(data, len) } else { 0 }
  @internal.scalar_validate_utf8_strict_from(data, start)
}

///|
/// Fold each chunk's MSB-bitmask via OR, then check it's zero (no high bit
/// set means pure ASCII). Tail bytes use `data[i] & 0x80`.
fn is_ascii_v128(data : FixedArray[Byte], len : Int) -> Int {
  let vec_len = floor_i8x16_len(len)
  let high_bits = for i = 0, high_bits = 0; i < vec_len; {
    continue i + 16, high_bits | i8x16_bitmask(fixed_i8x16_load(data, i))
  } nobreak {
    high_bits
  }
  let high_bits = for i in vec_len..<len; high_bits = high_bits {
    continue high_bits | (data[i].to_int() & 0x80)
  } nobreak {
    high_bits
  }
  if high_bits == 0 {
    1
  } else {
    0
  }
}

///|
pub fn is_ascii(data : FixedArray[Byte]) -> Bool {
  let len = data.length()
  if len == 0 {
    return true
  }
  is_ascii_v128(data, len) != 0
}

///|
/// In-place ASCII A-Z -> a-z via per-byte range mask:
/// `delta = (v >= 'A') & (v <= 'Z') & 0x20`, then `v += delta`.
fn to_lower_ascii_v128(data : FixedArray[Byte], len : Int) -> Unit {
  let vec_len = floor_i8x16_len(len)
  let a = i8x16_splat(0x41)
  let z = i8x16_splat(0x5A)
  let delta = i8x16_splat(0x20)
  for i = 0; i < vec_len; {
    let v = fixed_i8x16_load(data, i)
    let in_range = v128_and(i8x16_ge_s(v, a), i8x16_le_s(v, z))
    fixed_i8x16_store(data, i, i8x16_add(v, v128_and(in_range, delta)))
    continue i + 16
  }
  for i in vec_len..<len {
    let c = data[i].to_int()
    if c >= 0x41 && c <= 0x5A {
      data[i] = (c + 0x20).to_byte()
    }
  }
}

///|
pub fn to_lower_ascii(data : FixedArray[Byte]) -> Unit {
  let len = data.length()
  if len == 0 {
    return
  }
  to_lower_ascii_v128(data, len)
}

///|
/// In-place ASCII a-z -> A-Z (symmetric: subtract 32 when in range).
fn to_upper_ascii_v128(data : FixedArray[Byte], len : Int) -> Unit {
  let vec_len = floor_i8x16_len(len)
  let a = i8x16_splat(0x61)
  let z = i8x16_splat(0x7A)
  let delta = i8x16_splat(0x20)
  for i = 0; i < vec_len; {
    let v = fixed_i8x16_load(data, i)
    let in_range = v128_and(i8x16_ge_s(v, a), i8x16_le_s(v, z))
    fixed_i8x16_store(data, i, i8x16_sub(v, v128_and(in_range, delta)))
    continue i + 16
  }
  for i in vec_len..<len {
    let c = data[i].to_int()
    if c >= 0x61 && c <= 0x7A {
      data[i] = (c - 0x20).to_byte()
    }
  }
}

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

// ---------------------------------------------------------------------------
// `Bytes`-direct variants. `Bytes` crosses the wasm inline-WAT FFI as a
// linear-memory pointer to byte[0] — identical ABI to FixedArray[Byte] — so
// the WAT bodies are the same as the FixedArray versions above, only the
// MoonBit parameter type differs. (On wasm-gc `Bytes` arrives as a GC ref and
// is rejected, so those targets use the scalar path.)
// ---------------------------------------------------------------------------

///|
#borrow(a, b)
fn equal_bytes_b_v128(a : Bytes, b : Bytes, len : Int) -> Int =
  #|(func (param i32) (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) local.get 2 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 3 i32.add v128.load local.get 1 local.get 3 i32.add v128.load i8x16.eq i8x16.bitmask i32.const 65535 i32.ne i32.or local.set 5 local.get 3 i32.const 16 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 3 i32.add i32.load8_u local.get 1 local.get 3 i32.add i32.load8_u i32.ne i32.or local.set 5 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end local.get 5 i32.eqz)

///|
/// SIMD byte equality over `Bytes` (first `len` bytes) - wasm target.
pub fn equal_bytes_b(a : Bytes, b : Bytes, len : Int) -> Bool {
  if len <= 0 {
    return true
  }
  equal_bytes_b_v128(a, b, len) != 0
}

///|
#borrow(data)
fn find_byte_b_v128(data : Bytes, len : Int, needle : Int) -> Int =
  #|(func (param i32) (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local i32) (local v128) local.get 2 i8x16.splat local.set 7 local.get 1 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 4 i32.const -1 local.set 5 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 0 local.get 3 i32.add v128.load local.get 7 i8x16.eq i8x16.bitmask local.set 6 local.get 3 local.get 6 i32.ctz i32.add local.get 5 local.get 5 i32.const -1 i32.eq local.get 6 i32.mul select local.set 5 local.get 3 i32.const 16 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 1 i32.ge_s br_if 1 local.get 3 local.get 5 local.get 5 i32.const -1 i32.eq local.get 0 local.get 3 i32.add i32.load8_u local.get 2 i32.eq i32.and select local.set 5 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end local.get 5)

///|
/// SIMD find first byte equal to `needle` over `Bytes` - wasm target.
pub fn find_byte_b(data : Bytes, needle : Byte) -> Int {
  let len = data.length()
  if len == 0 {
    return -1
  }
  find_byte_b_v128(data, len, needle.to_int())
}

///|
#borrow(data)
fn count_byte_b_v128(data : Bytes, len : Int, needle : Int) -> Int =
  #|(func (param i32) (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local v128) local.get 2 i8x16.splat local.set 6 local.get 1 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 3 i32.add v128.load local.get 6 i8x16.eq i8x16.bitmask i32.popcnt i32.add local.set 5 local.get 3 i32.const 16 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 1 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 3 i32.add i32.load8_u local.get 2 i32.eq i32.add local.set 5 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end local.get 5)

///|
/// SIMD count of bytes equal to `needle` over `Bytes` - wasm target.
pub fn count_byte_b(data : Bytes, needle : Byte) -> Int {
  let len = data.length()
  if len == 0 {
    return 0
  }
  count_byte_b_v128(data, len, needle.to_int())
}

///|
#borrow(data)
fn is_ascii_b_v128(data : Bytes, len : Int) -> Int =
  #|(func (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) local.get 1 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.add v128.load i8x16.bitmask i32.or local.set 4 local.get 2 i32.const 16 i32.add local.set 2 br 0 end end block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.add i32.load8_u i32.const 128 i32.and i32.or local.set 4 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end local.get 4 i32.eqz)

///|
/// Whether every byte of `data` is < 0x80 (pure ASCII) - wasm target.
pub fn is_ascii_b(data : Bytes) -> Bool {
  let len = data.length()
  if len == 0 {
    return true
  }
  is_ascii_b_v128(data, len) != 0
}

///|
/// Find the first index `>= start` where `data[i] == needle`, else -1.
/// SIMD 16-byte scan with early return on the first matching chunk (ctz gives
/// the earliest lane), then a scalar tail. Backs the substring driver below.
#borrow(data)
fn find_byte_from_b(data : Bytes, len : Int, needle : Int, start : Int) -> Int =
  #|(func (param i32) (param i32) (param i32) (param i32) (result i32) (local i32) (local i32) (local v128) local.get 2 i8x16.splat local.set 6 local.get 3 local.set 4 block loop local.get 4 i32.const 16 i32.add local.get 1 i32.gt_s br_if 1 local.get 0 local.get 4 i32.add v128.load local.get 6 i8x16.eq i8x16.bitmask local.set 5 local.get 5 i32.eqz i32.eqz if local.get 4 local.get 5 i32.ctz i32.add return end local.get 4 i32.const 16 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 1 i32.ge_s br_if 1 local.get 0 local.get 4 i32.add i32.load8_u local.get 2 i32.eq if local.get 4 return end local.get 4 i32.const 1 i32.add local.set 4 br 0 end end i32.const -1)

///|
/// SIMD substring search over `Bytes`: SIMD first-byte scan (`find_byte_from_b`)
/// to jump to candidates, scalar verify of the remaining `needle` bytes.
/// Returns the first index where `needle` occurs in `haystack`, or -1.
pub fn find_bytes_b(haystack : Bytes, needle : Bytes) -> Int {
  let hl = haystack.length()
  let nl = needle.length()
  if nl == 0 {
    return 0
  }
  if nl > hl {
    return -1
  }
  let n0 = needle[0].to_int()
  let last = hl - nl
  let mut pos = 0
  while pos <= last {
    let cand = find_byte_from_b(haystack, hl, n0, pos)
    if cand < 0 || cand > last {
      return -1
    }
    let mut k = 1
    let mut ok = true
    while k < nl {
      if haystack[cand + k] != needle[k] {
        ok = false
        break
      }
      k = k + 1
    }
    if ok {
      return cand
    }
    pos = cand + 1
  }
  -1
}

///|
/// Last index `i` where `data[i] == needle`, else -1. Scalar tail from the
/// end down to the 16-aligned boundary, then 16-byte chunks backward with
/// early return of the highest matching lane (`31 - i32.clz(bitmask)`).
/// locals: 3 = aligned-then-bitmask, 4 = pos, 5 = needle_v.
#borrow(data)
fn rfind_byte_b_v128(data : Bytes, len : Int, needle : Int) -> Int =
  #|(func (param i32) (param i32) (param i32) (result i32) (local i32) (local i32) (local v128) local.get 2 i8x16.splat local.set 5 local.get 1 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 3 local.get 1 i32.const 1 i32.sub local.set 4 block loop local.get 4 local.get 3 i32.lt_s br_if 1 local.get 0 local.get 4 i32.add i32.load8_u local.get 2 i32.eq if local.get 4 return end local.get 4 i32.const 1 i32.sub local.set 4 br 0 end end local.get 3 i32.const 16 i32.sub local.set 4 block loop local.get 4 i32.const 0 i32.lt_s br_if 1 local.get 0 local.get 4 i32.add v128.load local.get 5 i8x16.eq i8x16.bitmask local.set 3 local.get 3 i32.eqz i32.eqz if local.get 4 i32.const 31 i32.add local.get 3 i32.clz i32.sub return end local.get 4 i32.const 16 i32.sub local.set 4 br 0 end end i32.const -1)

///|
/// SIMD last occurrence of `needle` - wasm target: reverse i8x16.eq scan.
pub fn rfind_byte_b(data : Bytes, needle : Byte) -> Int {
  let len = data.length()
  if len == 0 {
    return -1
  }
  rfind_byte_b_v128(data, len, needle.to_int())
}
