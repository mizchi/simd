///|
/// SIMD encode of `chunks` × 12 input bytes → `chunks` × 16 output bytes.
/// Algorithm: Wojciech Muła's 12→16 layout. Per 24-bit input group we extract
/// four 6-bit indices, then convert each index to ASCII via range-based
/// arithmetic offsets (no swizzle lookup table required).
///
/// Param 0 = input ptr, 1 = output ptr, 2 = chunks (each chunk = 12 in / 16 out).
/// Locals: 3 = chunk_i, 4 = src_off, 5 = dst_off, 6 = v128 raw,
///   7 = v128 shuffled, 8 = v128 indices, 9 = v128 result, 10 = v128 mask.
#borrow(src, dst)
fn encode_simd_chunks(
  src : FixedArray[Byte],
  dst : FixedArray[Byte],
  src_off0 : Int,
  dst_off0 : Int,
  chunks : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) (local i32) (local v128) (local v128) (local v128) (local v128) (local v128) local.get 2 local.set 6 local.get 3 local.set 7 i32.const 0 local.set 5 block loop local.get 5 local.get 4 i32.ge_s br_if 1 local.get 0 local.get 6 i32.add v128.load local.tee 8 local.get 8 i8x16.shuffle 1 0 2 1 4 3 5 4 7 6 8 7 10 9 11 10 local.set 8 local.get 8 i32.const 10 i16x8.shr_u i32.const 63 i32x4.splat v128.and local.set 9 local.get 8 i32.const 4 i16x8.shr_u i32.const 8 i16x8.shl i32.const 16128 i32x4.splat v128.and local.get 9 v128.or local.set 9 local.get 8 i32.const 6 i16x8.shr_u i32.const 4128768 i32x4.splat v128.and local.get 9 v128.or local.set 9 local.get 8 i32.const 8 i16x8.shl i32.const 1056964608 i32x4.splat v128.and local.get 9 v128.or local.set 9 local.get 9 i32.const 65 i8x16.splat i8x16.add local.set 10 local.get 10 local.get 9 i32.const 26 i8x16.splat i8x16.ge_s i32.const 6 i8x16.splat v128.and i8x16.add local.set 10 local.get 10 local.get 9 i32.const 52 i8x16.splat i8x16.ge_s i32.const 75 i8x16.splat v128.and i8x16.sub local.set 10 local.get 10 local.get 9 i32.const 62 i8x16.splat i8x16.ge_s i32.const 15 i8x16.splat v128.and i8x16.sub local.set 10 local.get 10 local.get 9 i32.const 63 i8x16.splat i8x16.ge_s i32.const 3 i8x16.splat v128.and i8x16.add local.set 10 local.get 1 local.get 7 i32.add local.get 10 v128.store local.get 6 i32.const 12 i32.add local.set 6 local.get 7 i32.const 16 i32.add local.set 7 local.get 5 i32.const 1 i32.add local.set 5 br 0 end end)

///|
/// Encode `input` as Base64. wasm target: SIMD path (12 in / 16 out) with a
/// scalar tail.
pub fn encode(input : FixedArray[Byte]) -> FixedArray[Byte] {
  let n = input.length()
  let out = FixedArray::make(encoded_len(n), b'\x00')
  // SIMD requires reading 16 bytes per chunk; only use it when we have
  // ≥ 16 bytes of trailing input we can safely v128.load.
  let simd_chunks = if n >= 16 { (n - 16) / 12 + 1 } else { 0 }
  // re-derive: each chunk consumes 12 bytes and we must be able to read 16,
  // so the last SIMD chunk starts at `(simd_chunks - 1) * 12` and reads up to
  // `(simd_chunks - 1) * 12 + 16` bytes. Require that index ≤ n.
  let simd_chunks = if simd_chunks > 0 && (simd_chunks - 1) * 12 + 16 > n {
    simd_chunks - 1
  } else {
    simd_chunks
  }
  if simd_chunks > 0 {
    encode_simd_chunks(input, out, 0, 0, simd_chunks)
  }
  let consumed = simd_chunks * 12
  if consumed < n {
    // Build a temporary scalar input slice for the tail. The scalar encoder
    // takes the full input but writes only `encoded_len(n) - simd_chunks*16`
    // bytes of output starting at the corresponding offset, so for the tail
    // we materialise a small array.
    let tail_len = n - consumed
    let tail = FixedArray::make(tail_len, b'\x00')
    for i in 0..<tail_len {
      tail[i] = input[consumed + i]
    }
    let tail_out = FixedArray::make(encoded_len(tail_len), b'\x00')
    scalar_encode(tail, tail_out)
    let tail_out_len = tail_out.length()
    let dst_off = simd_chunks * 16
    for i in 0..<tail_out_len {
      out[dst_off + i] = tail_out[i]
    }
  }
  out
}

///|
/// SIMD decode of `chunks` × 16 input ASCII bytes → `chunks` × 12 decoded bytes.
/// Algorithm:
///   1. Range-based ASCII → 6-bit lookup with arithmetic offsets per range.
///   2. Validity mask: OR of all range masks. Returned via the result value so
///      the caller can fall back to scalar when any byte is out of alphabet.
///   3. Pack 4 × 6-bit → 3 bytes per i32 lane via shifts + per-byte masks.
///   4. Compact the 12 valid bytes per chunk into the low 12 bytes of v128 and
///      v128.store all 16 (the trailing 4 garbage bytes are overwritten by the
///      next chunk, or are out-of-bounds for the last chunk — caller handles
///      tail).
///
/// Returns 0 if all chunks decoded cleanly, non-zero if any chunk contained
/// at least one byte outside the Base64 alphabet.
#borrow(src, dst)
fn decode_simd_chunks(
  src : FixedArray[Byte],
  dst : FixedArray[Byte],
  src_off0 : Int,
  dst_off0 : Int,
  chunks : Int,
) -> Int =
  #|(func (param i32) (param i32) (param i32) (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local i32) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) local.get 2 local.set 6 local.get 3 local.set 7 i32.const 0 local.set 5 i32.const 0 local.set 8 block loop local.get 5 local.get 4 i32.ge_s br_if 1 local.get 0 local.get 6 i32.add v128.load local.set 9 local.get 9 i32.const 65 i8x16.splat i8x16.ge_s local.get 9 i32.const 90 i8x16.splat i8x16.le_s v128.and local.set 10 local.get 9 i32.const 97 i8x16.splat i8x16.ge_s local.get 9 i32.const 122 i8x16.splat i8x16.le_s v128.and local.set 11 local.get 9 i32.const 48 i8x16.splat i8x16.ge_s local.get 9 i32.const 57 i8x16.splat i8x16.le_s v128.and local.set 12 local.get 9 i32.const 43 i8x16.splat i8x16.eq local.set 13 local.get 9 i32.const 47 i8x16.splat i8x16.eq local.set 14 local.get 10 local.get 11 v128.or local.get 12 v128.or local.get 13 v128.or local.get 14 v128.or i8x16.bitmask i32.const 65535 i32.ne if local.get 8 i32.const 1 i32.or local.set 8 end local.get 9 local.get 10 i32.const 191 i8x16.splat v128.and i8x16.add local.get 11 i32.const 185 i8x16.splat v128.and i8x16.add local.get 12 i32.const 4 i8x16.splat v128.and i8x16.add local.get 13 i32.const 19 i8x16.splat v128.and i8x16.add local.get 14 i32.const 16 i8x16.splat v128.and i8x16.add local.set 9 local.get 9 i32.const 2 i32x4.shl i32.const 252 i32x4.splat v128.and local.get 9 i32.const 12 i32x4.shr_u i32.const 3 i32x4.splat v128.and v128.or local.get 9 i32.const 4 i32x4.shl i32.const 61440 i32x4.splat v128.and v128.or local.get 9 i32.const 10 i32x4.shr_u i32.const 3840 i32x4.splat v128.and v128.or local.get 9 i32.const 6 i32x4.shl i32.const 12582912 i32x4.splat v128.and v128.or local.get 9 i32.const 8 i32x4.shr_u i32.const 4128768 i32x4.splat v128.and v128.or local.tee 9 local.get 9 i8x16.shuffle 0 1 2 4 5 6 8 9 10 12 13 14 0 1 2 4 local.set 9 local.get 1 local.get 7 i32.add local.get 9 v128.store local.get 6 i32.const 16 i32.add local.set 6 local.get 7 i32.const 12 i32.add local.set 7 local.get 5 i32.const 1 i32.add local.set 5 br 0 end end local.get 8)

///|
/// Decode a Base64 buffer. wasm target: SIMD decode for the bulk, scalar for
/// the trailing 4-char group (which may contain `=` padding).
pub fn decode(input : FixedArray[Byte]) -> FixedArray[Byte]? {
  let n = input.length()
  if n == 0 {
    return Some(FixedArray::make(0, b'\x00'))
  }
  if n % 4 != 0 {
    return None
  }
  let pad = count_padding(input)
  let out = FixedArray::make(decoded_len(n, pad), b'\x00')
  // SIMD decodes 16-input/12-output per chunk; v128.store writes 16 bytes so
  // we need 4 bytes of slack at the end of `out`. Carve out at least the last
  // 4 input characters (= 3 output bytes) for the scalar tail to guarantee
  // safety.
  let bulk_chunks = if n >= 32 { (n - 4) / 16 } else { 0 }
  let mut consumed = 0
  if bulk_chunks > 0 {
    let err = decode_simd_chunks(input, out, 0, 0, bulk_chunks)
    if err != 0 {
      // Any byte outside the alphabet — fall back to full scalar decode.
      match scalar_decode(input, out) {
        Some(_) => return Some(out)
        None => return None
      }
    }
    consumed = bulk_chunks * 16
  }
  // Scalar tail: feed the remaining input (always a multiple of 4) through
  // scalar_decode by carving a slice and stitching the result back in.
  let tail_in_len = n - consumed
  let tail = FixedArray::make(tail_in_len, b'\x00')
  for i in 0..<tail_in_len {
    tail[i] = input[consumed + i]
  }
  let tail_out = FixedArray::make(decoded_len(tail_in_len, pad), b'\x00')
  match scalar_decode(tail, tail_out) {
    Some(written) => {
      let dst_off = bulk_chunks * 12
      for i in 0..<written {
        out[dst_off + i] = tail_out[i]
      }
      Some(out)
    }
    None => None
  }
}
