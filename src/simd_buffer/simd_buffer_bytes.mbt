///|
/// Byte sibling of `SimdBuffer`. Linear-memory backed, 1 byte per element.
/// Targets wasm + wasm-gc. Use when you want SIMD base64 / memcpy / scan
/// on byte buffers without paying the FixedArray-on-GC overhead.
pub struct SimdBufferBytes {
  addr : Int
  len : Int
}

///|
pub fn SimdBufferBytes::make(len : Int) -> SimdBufferBytes {
  // +16 trailing slack so final v128.load (16 bytes) stays in-bounds.
  let bytes = len + 16
  let addr = buf_alloc(bytes)
  { addr, len }
}

///|
pub fn SimdBufferBytes::length(self : SimdBufferBytes) -> Int {
  self.len
}

///|
/// Linear-memory address of byte 0. wasm + wasm-gc only — gives
/// downstream packages (e.g. `@simdjson`) a way to plug inline-WAT
/// kernels on top of SimdBufferBytes without exposing the struct
/// internals.
pub fn SimdBufferBytes::raw_addr(self : SimdBufferBytes) -> Int {
  self.addr
}

///|
pub fn SimdBufferBytes::get(self : SimdBufferBytes, i : Int) -> Byte {
  if i < 0 || i >= self.len {
    abort("SimdBufferBytes.get: index out of bounds")
  }
  buf_load_byte(self.addr + i).to_byte()
}

///|
pub fn SimdBufferBytes::set(self : SimdBufferBytes, i : Int, v : Byte) -> Unit {
  if i < 0 || i >= self.len {
    abort("SimdBufferBytes.set: index out of bounds")
  }
  buf_store_byte(self.addr + i, v.to_int())
}

///|
/// Base64 encode this buffer. Returns a freshly-allocated SimdBufferBytes
/// holding the RFC 4648 padded encoding. SIMD path (12 in / 16 out) for the
/// bulk; arithmetic scalar for the trailing 1-2 bytes.
pub fn SimdBufferBytes::base64_encode(
  input : SimdBufferBytes,
) -> SimdBufferBytes {
  let n = input.len
  let out_len = (n + 2) / 3 * 4
  let out = SimdBufferBytes::make(out_len)
  let simd_chunks_initial = if n >= 16 { (n - 16) / 12 + 1 } else { 0 }
  let simd_chunks = if simd_chunks_initial > 0 &&
    (simd_chunks_initial - 1) * 12 + 16 > n {
    simd_chunks_initial - 1
  } else {
    simd_chunks_initial
  }
  if simd_chunks > 0 {
    buf_base64_encode(input.addr, out.addr, simd_chunks)
  }
  let consumed = simd_chunks * 12
  let mut q = simd_chunks * 16
  let mut p = consumed
  // Scalar full triples.
  while p + 3 <= n {
    let b0 = buf_load_byte(input.addr + p)
    let b1 = buf_load_byte(input.addr + p + 1)
    let b2 = buf_load_byte(input.addr + p + 2)
    let v = (b0 << 16) | (b1 << 8) | b2
    buf_store_byte(out.addr + q, encode_sextet((v >> 18) & 63))
    buf_store_byte(out.addr + q + 1, encode_sextet((v >> 12) & 63))
    buf_store_byte(out.addr + q + 2, encode_sextet((v >> 6) & 63))
    buf_store_byte(out.addr + q + 3, encode_sextet(v & 63))
    p = p + 3
    q = q + 4
  }
  let rem = n - p
  if rem == 1 {
    let v = buf_load_byte(input.addr + p) << 16
    buf_store_byte(out.addr + q, encode_sextet((v >> 18) & 63))
    buf_store_byte(out.addr + q + 1, encode_sextet((v >> 12) & 63))
    buf_store_byte(out.addr + q + 2, '='.to_int())
    buf_store_byte(out.addr + q + 3, '='.to_int())
  } else if rem == 2 {
    let b0 = buf_load_byte(input.addr + p)
    let b1 = buf_load_byte(input.addr + p + 1)
    let v = (b0 << 16) | (b1 << 8)
    buf_store_byte(out.addr + q, encode_sextet((v >> 18) & 63))
    buf_store_byte(out.addr + q + 1, encode_sextet((v >> 12) & 63))
    buf_store_byte(out.addr + q + 2, encode_sextet((v >> 6) & 63))
    buf_store_byte(out.addr + q + 3, '='.to_int())
  }
  out
}

///|
/// Base64 decode. Returns `None` if the input length is not a multiple of 4
/// or if any character is out of alphabet (excluding the trailing `=`
/// padding). SIMD bulk + scalar tail for the last 4-char group.
pub fn SimdBufferBytes::base64_decode(
  input : SimdBufferBytes,
) -> SimdBufferBytes? {
  let n = input.len
  if n == 0 {
    return Some(SimdBufferBytes::make(0))
  }
  if n % 4 != 0 {
    return None
  }
  let pad = base64_count_padding(input)
  let out_len = n / 4 * 3 - pad
  let out = SimdBufferBytes::make(out_len)
  let bulk_chunks = if n >= 32 { (n - 4) / 16 } else { 0 }
  let mut consumed = 0
  if bulk_chunks > 0 {
    let err = buf_base64_decode(input.addr, out.addr, bulk_chunks)
    if err != 0 {
      // Fall back to scalar from the start.
      return base64_decode_scalar(input, out)
    }
    consumed = bulk_chunks * 16
  }
  // Scalar tail.
  let tail_q = bulk_chunks * 12
  match base64_decode_tail(input, out, consumed, tail_q, pad) {
    Some(_) => Some(out)
    None => None
  }
}

///|
/// In-place encode into a caller-supplied output buffer. Useful when you
/// want to reuse a SimdBufferBytes across many calls (avoids `memory.grow`
/// per call). `out` must have at least `(input.length() + 2) / 3 * 4` bytes.
/// Caller is responsible for sizing.
pub fn SimdBufferBytes::base64_encode_into(
  input : SimdBufferBytes,
  out : SimdBufferBytes,
) -> Unit {
  let n = input.len
  let simd_chunks_initial = if n >= 16 { (n - 16) / 12 + 1 } else { 0 }
  let simd_chunks = if simd_chunks_initial > 0 &&
    (simd_chunks_initial - 1) * 12 + 16 > n {
    simd_chunks_initial - 1
  } else {
    simd_chunks_initial
  }
  if simd_chunks > 0 {
    buf_base64_encode(input.addr, out.addr, simd_chunks)
  }
  let consumed = simd_chunks * 12
  let mut q = simd_chunks * 16
  let mut p = consumed
  while p + 3 <= n {
    let b0 = buf_load_byte(input.addr + p)
    let b1 = buf_load_byte(input.addr + p + 1)
    let b2 = buf_load_byte(input.addr + p + 2)
    let v = (b0 << 16) | (b1 << 8) | b2
    buf_store_byte(out.addr + q, encode_sextet((v >> 18) & 63))
    buf_store_byte(out.addr + q + 1, encode_sextet((v >> 12) & 63))
    buf_store_byte(out.addr + q + 2, encode_sextet((v >> 6) & 63))
    buf_store_byte(out.addr + q + 3, encode_sextet(v & 63))
    p = p + 3
    q = q + 4
  }
  let rem = n - p
  if rem == 1 {
    let v = buf_load_byte(input.addr + p) << 16
    buf_store_byte(out.addr + q, encode_sextet((v >> 18) & 63))
    buf_store_byte(out.addr + q + 1, encode_sextet((v >> 12) & 63))
    buf_store_byte(out.addr + q + 2, '='.to_int())
    buf_store_byte(out.addr + q + 3, '='.to_int())
  } else if rem == 2 {
    let b0 = buf_load_byte(input.addr + p)
    let b1 = buf_load_byte(input.addr + p + 1)
    let v = (b0 << 16) | (b1 << 8)
    buf_store_byte(out.addr + q, encode_sextet((v >> 18) & 63))
    buf_store_byte(out.addr + q + 1, encode_sextet((v >> 12) & 63))
    buf_store_byte(out.addr + q + 2, encode_sextet((v >> 6) & 63))
    buf_store_byte(out.addr + q + 3, '='.to_int())
  }
}

///|
/// In-place decode counterpart. Returns the number of decoded bytes
/// written, or `None` on invalid input. `out` must be large enough.
pub fn SimdBufferBytes::base64_decode_into(
  input : SimdBufferBytes,
  out : SimdBufferBytes,
) -> Int? {
  let n = input.len
  if n == 0 {
    return Some(0)
  }
  if n % 4 != 0 {
    return None
  }
  let pad = base64_count_padding(input)
  let bulk_chunks = if n >= 32 { (n - 4) / 16 } else { 0 }
  let mut consumed = 0
  if bulk_chunks > 0 {
    let err = buf_base64_decode(input.addr, out.addr, bulk_chunks)
    if err != 0 {
      return base64_decode_tail(input, out, 0, 0, pad)
    }
    consumed = bulk_chunks * 16
  }
  let tail_q = bulk_chunks * 12
  base64_decode_tail(input, out, consumed, tail_q, pad)
}

///|
fn base64_count_padding(input : SimdBufferBytes) -> Int {
  let n = input.len
  if n == 0 {
    return 0
  }
  if buf_load_byte(input.addr + n - 1) != '='.to_int() {
    return 0
  }
  if n >= 2 && buf_load_byte(input.addr + n - 2) == '='.to_int() {
    return 2
  }
  1
}

///|
/// Inline 6-bit → ASCII via arithmetic offsets (same recipe as the SIMD
/// encode arithmetic stage).
fn encode_sextet(v : Int) -> Int {
  let mut r = v + 65
  if v >= 26 {
    r = r + 6
  }
  if v >= 52 {
    r = r - 75
  }
  if v >= 62 {
    r = r - 15
  }
  if v >= 63 {
    r = r + 3
  }
  r
}

///|
/// Inverse: ASCII → 6-bit, or -1 if not a Base64 alphabet char.
fn decode_sextet(c : Int) -> Int {
  if c >= 65 && c <= 90 {
    return c - 65
  }
  if c >= 97 && c <= 122 {
    return c - 71
  }
  if c >= 48 && c <= 57 {
    return c + 4
  }
  if c == 43 {
    return 62
  }
  if c == 47 {
    return 63
  }
  -1
}

///|
/// Full-scalar fallback for decode when the SIMD path reported an
/// out-of-alphabet character.
fn base64_decode_scalar(
  input : SimdBufferBytes,
  out : SimdBufferBytes,
) -> SimdBufferBytes? {
  match base64_decode_tail(input, out, 0, 0, base64_count_padding(input)) {
    Some(_) => Some(out)
    None => None
  }
}

///|
/// Decode the input range `[p, n)` into the output starting at `q`, where
/// `pad` is the trailing `=` count (0, 1, 2). Returns Some(out_len) or None
/// on invalid input.
fn base64_decode_tail(
  input : SimdBufferBytes,
  out : SimdBufferBytes,
  p0 : Int,
  q0 : Int,
  pad : Int,
) -> Int? {
  let n = input.len
  if n == 0 {
    return Some(0)
  }
  let mut p = p0
  let mut q = q0
  while p + 4 <= n - 4 {
    let d0 = decode_sextet(buf_load_byte(input.addr + p))
    let d1 = decode_sextet(buf_load_byte(input.addr + p + 1))
    let d2 = decode_sextet(buf_load_byte(input.addr + p + 2))
    let d3 = decode_sextet(buf_load_byte(input.addr + p + 3))
    if (d0 | d1 | d2 | d3) < 0 {
      return None
    }
    let v = (d0 << 18) | (d1 << 12) | (d2 << 6) | d3
    buf_store_byte(out.addr + q, (v >> 16) & 0xFF)
    buf_store_byte(out.addr + q + 1, (v >> 8) & 0xFF)
    buf_store_byte(out.addr + q + 2, v & 0xFF)
    p = p + 4
    q = q + 3
  }
  // Final 4-char group.
  let d0 = decode_sextet(buf_load_byte(input.addr + p))
  let d1 = decode_sextet(buf_load_byte(input.addr + p + 1))
  if (d0 | d1) < 0 {
    return None
  }
  if pad == 2 {
    let v = (d0 << 18) | (d1 << 12)
    buf_store_byte(out.addr + q, (v >> 16) & 0xFF)
    Some(q + 1)
  } else {
    let d2 = decode_sextet(buf_load_byte(input.addr + p + 2))
    if d2 < 0 {
      return None
    }
    if pad == 1 {
      let v = (d0 << 18) | (d1 << 12) | (d2 << 6)
      buf_store_byte(out.addr + q, (v >> 16) & 0xFF)
      buf_store_byte(out.addr + q + 1, (v >> 8) & 0xFF)
      Some(q + 2)
    } else {
      let d3 = decode_sextet(buf_load_byte(input.addr + p + 3))
      if d3 < 0 {
        return None
      }
      let v = (d0 << 18) | (d1 << 12) | (d2 << 6) | d3
      buf_store_byte(out.addr + q, (v >> 16) & 0xFF)
      buf_store_byte(out.addr + q + 1, (v >> 8) & 0xFF)
      buf_store_byte(out.addr + q + 2, v & 0xFF)
      Some(q + 3)
    }
  }
}

// ---- inline-WAT primitives ----

///|
fn buf_load_byte(addr : Int) -> Int =
  #|(func (param i32) (result i32) local.get 0 i32.load8_u)

///|
fn buf_store_byte(addr : Int, value : Int) -> Unit =
  #|(func (param i32) (param i32) local.get 0 local.get 1 i32.store8)

///|
/// SIMD base64 encode of `chunks` × 12 input bytes → `chunks` × 16 output
/// bytes. Identical body to `encode_simd_chunks` in `base64/base64_wasm.mbt`
/// but takes raw i32 addresses and assumes src_off / dst_off = 0.
fn buf_base64_encode(src_addr : Int, dst_addr : Int, chunks : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (local i32) (local i32) (local i32) (local v128) (local v128) (local v128) i32.const 0 local.set 4 i32.const 0 local.set 5 i32.const 0 local.set 3 block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 0 local.get 4 i32.add v128.load local.tee 6 local.get 6 i8x16.shuffle 1 0 2 1 4 3 5 4 7 6 8 7 10 9 11 10 local.set 6 local.get 6 i32.const 10 i16x8.shr_u i32.const 63 i32x4.splat v128.and local.set 7 local.get 6 i32.const 4 i16x8.shr_u i32.const 8 i16x8.shl i32.const 16128 i32x4.splat v128.and local.get 7 v128.or local.set 7 local.get 6 i32.const 6 i16x8.shr_u i32.const 4128768 i32x4.splat v128.and local.get 7 v128.or local.set 7 local.get 6 i32.const 8 i16x8.shl i32.const 1056964608 i32x4.splat v128.and local.get 7 v128.or local.set 7 local.get 7 i32.const 65 i8x16.splat i8x16.add local.set 8 local.get 8 local.get 7 i32.const 26 i8x16.splat i8x16.ge_s i32.const 6 i8x16.splat v128.and i8x16.add local.set 8 local.get 8 local.get 7 i32.const 52 i8x16.splat i8x16.ge_s i32.const 75 i8x16.splat v128.and i8x16.sub local.set 8 local.get 8 local.get 7 i32.const 62 i8x16.splat i8x16.ge_s i32.const 15 i8x16.splat v128.and i8x16.sub local.set 8 local.get 8 local.get 7 i32.const 63 i8x16.splat i8x16.ge_s i32.const 3 i8x16.splat v128.and i8x16.add local.set 8 local.get 1 local.get 5 i32.add local.get 8 v128.store local.get 4 i32.const 12 i32.add local.set 4 local.get 5 i32.const 16 i32.add local.set 5 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end)

///|
/// SIMD base64 decode of `chunks` × 16 input ASCII bytes → `chunks` × 12
/// decoded bytes. Same body as `decode_simd_chunks` in `base64/base64_wasm.mbt`;
/// returns 0 on success, non-zero if any chunk contained an out-of-alphabet
/// byte (caller falls back to scalar in that case).
fn buf_base64_decode(src_addr : Int, dst_addr : Int, chunks : Int) -> Int =
  #|(func (param i32) (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local i32) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) i32.const 0 local.set 4 i32.const 0 local.set 5 i32.const 0 local.set 3 i32.const 0 local.set 6 block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 0 local.get 4 i32.add v128.load local.set 7 local.get 7 i32.const 65 i8x16.splat i8x16.ge_s local.get 7 i32.const 90 i8x16.splat i8x16.le_s v128.and local.set 8 local.get 7 i32.const 97 i8x16.splat i8x16.ge_s local.get 7 i32.const 122 i8x16.splat i8x16.le_s v128.and local.set 9 local.get 7 i32.const 48 i8x16.splat i8x16.ge_s local.get 7 i32.const 57 i8x16.splat i8x16.le_s v128.and local.set 10 local.get 7 i32.const 43 i8x16.splat i8x16.eq local.set 11 local.get 7 i32.const 47 i8x16.splat i8x16.eq local.set 12 local.get 8 local.get 9 v128.or local.get 10 v128.or local.get 11 v128.or local.get 12 v128.or i8x16.bitmask i32.const 65535 i32.ne if local.get 6 i32.const 1 i32.or local.set 6 end local.get 7 local.get 8 i32.const 191 i8x16.splat v128.and i8x16.add local.get 9 i32.const 185 i8x16.splat v128.and i8x16.add local.get 10 i32.const 4 i8x16.splat v128.and i8x16.add local.get 11 i32.const 19 i8x16.splat v128.and i8x16.add local.get 12 i32.const 16 i8x16.splat v128.and i8x16.add local.set 7 local.get 7 i32.const 2 i32x4.shl i32.const 252 i32x4.splat v128.and local.get 7 i32.const 12 i32x4.shr_u i32.const 3 i32x4.splat v128.and v128.or local.get 7 i32.const 4 i32x4.shl i32.const 61440 i32x4.splat v128.and v128.or local.get 7 i32.const 10 i32x4.shr_u i32.const 3840 i32x4.splat v128.and v128.or local.get 7 i32.const 6 i32x4.shl i32.const 12582912 i32x4.splat v128.and v128.or local.get 7 i32.const 8 i32x4.shr_u i32.const 4128768 i32x4.splat v128.and v128.or local.tee 7 local.get 7 i8x16.shuffle 0 1 2 4 5 6 8 9 10 12 13 14 0 1 2 4 local.set 7 local.get 1 local.get 5 i32.add local.get 7 v128.store local.get 4 i32.const 16 i32.add local.set 4 local.get 5 i32.const 12 i32.add local.set 5 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end local.get 6)

// ---- byte bulk ops ----

///|
/// SIMD popcount. Count of `1` bits across all bytes in the buffer.
pub fn SimdBufferBytes::popcount(self : SimdBufferBytes) -> Int {
  if self.len == 0 {
    return 0
  }
  buf_popcount_bytes(self.addr, self.len)
}

///|
/// SIMD memcpy: copy `len` bytes from `src` (offset `si`) to `dst` (offset
/// `di`). Buffers may not overlap.
pub fn SimdBufferBytes::memcpy(
  src : SimdBufferBytes,
  si : Int,
  dst : SimdBufferBytes,
  di : Int,
  len : Int,
) -> Unit {
  if len <= 0 {
    return
  }
  if si < 0 || di < 0 || si + len > src.len || di + len > dst.len {
    abort("SimdBufferBytes.memcpy: range out of bounds")
  }
  buf_memcpy_bytes(src.addr, si, dst.addr, di, len)
}

///|
/// SIMD memset: write `value` to `len` bytes starting at `off`.
pub fn SimdBufferBytes::memset(
  self : SimdBufferBytes,
  off : Int,
  len : Int,
  value : Byte,
) -> Unit {
  if len <= 0 {
    return
  }
  if off < 0 || off + len > self.len {
    abort("SimdBufferBytes.memset: range out of bounds")
  }
  buf_memset_bytes(self.addr, off, len, value.to_int())
}

///|
/// SIMD byte equality of the first `len` bytes of both buffers.
pub fn SimdBufferBytes::equal(
  a : SimdBufferBytes,
  b : SimdBufferBytes,
  len : Int,
) -> Bool {
  if len <= 0 {
    return true
  }
  if len > a.len || len > b.len {
    abort("SimdBufferBytes.equal: len exceeds buffer")
  }
  buf_equal_bytes(a.addr, b.addr, len) != 0
}

///|
/// SIMD find first byte equal to `needle`. Returns -1 if not present.
pub fn SimdBufferBytes::find_byte(self : SimdBufferBytes, needle : Byte) -> Int {
  if self.len == 0 {
    return -1
  }
  buf_find_byte(self.addr, self.len, needle.to_int())
}

///|
/// SIMD count of bytes equal to `needle`.
pub fn SimdBufferBytes::count_byte(
  self : SimdBufferBytes,
  needle : Byte,
) -> Int {
  if self.len == 0 {
    return 0
  }
  buf_count_byte(self.addr, self.len, needle.to_int())
}

///|
/// `true` iff every byte is < 0x80.
pub fn SimdBufferBytes::is_ascii(self : SimdBufferBytes) -> Bool {
  if self.len == 0 {
    return true
  }
  buf_is_ascii(self.addr, self.len) != 0
}

///|
/// In-place ASCII A-Z → a-z.
pub fn SimdBufferBytes::to_lower_ascii(self : SimdBufferBytes) -> Unit {
  if self.len == 0 {
    return
  }
  buf_to_lower_ascii(self.addr, self.len)
}

// ---- image / pixel ops (Tier 2 — 0.3.0) ----

///|
/// Expand a 3-byte-per-pixel RGB buffer into a 4-byte-per-pixel RGBA buffer
/// with constant `alpha`. `src.length() == n * 3` and `out.length() == n * 4`
/// where `n` is the pixel count. Use 0xFF for fully opaque.
pub fn SimdBufferBytes::rgb_to_rgba(
  src : SimdBufferBytes,
  alpha : Byte,
  out : SimdBufferBytes,
) -> Unit {
  if src.len % 3 != 0 {
    abort("SimdBufferBytes.rgb_to_rgba: src must be a multiple of 3 bytes")
  }
  let n = src.len / 3
  if out.len != n * 4 {
    abort("SimdBufferBytes.rgb_to_rgba: out length must be src/3 * 4")
  }
  if n == 0 {
    return
  }
  buf_rgb_to_rgba(src.addr, out.addr, n, alpha.to_int())
}

///|
/// Convert RGBA → grayscale Y via fixed-point weighted sum:
/// `Y = (77 * R + 150 * G + 29 * B) / 256`, an integer approximation of
/// `0.299*R + 0.587*G + 0.114*B` (Rec. 601). Alpha is dropped.
/// `src.length() == n * 4`, `out.length() == n` for pixel count `n`.
pub fn SimdBufferBytes::rgba_to_grayscale(
  src : SimdBufferBytes,
  out : SimdBufferBytes,
) -> Unit {
  if src.len % 4 != 0 {
    abort(
      "SimdBufferBytes.rgba_to_grayscale: src must be a multiple of 4 bytes",
    )
  }
  let n = src.len / 4
  if out.len != n {
    abort("SimdBufferBytes.rgba_to_grayscale: out length must be src/4")
  }
  if n == 0 {
    return
  }
  buf_rgba_to_grayscale(src.addr, out.addr, n)
}

///|
/// Extract one channel from an interleaved RGBA buffer. `ch ∈ 0..3`.
/// `src.length() == n * 4`, `out.length() == n`.
pub fn SimdBufferBytes::channel_extract(
  src : SimdBufferBytes,
  ch : Int,
  out : SimdBufferBytes,
) -> Unit {
  if ch < 0 || ch >= 4 {
    abort("SimdBufferBytes.channel_extract: ch must be 0..3")
  }
  if src.len % 4 != 0 {
    abort("SimdBufferBytes.channel_extract: src must be a multiple of 4 bytes")
  }
  let n = src.len / 4
  if out.len != n {
    abort("SimdBufferBytes.channel_extract: out length must be src/4")
  }
  if n == 0 {
    return
  }
  match ch {
    0 => buf_channel_extract_0(src.addr, out.addr, n)
    1 => buf_channel_extract_1(src.addr, out.addr, n)
    2 => buf_channel_extract_2(src.addr, out.addr, n)
    _ => buf_channel_extract_3(src.addr, out.addr, n)
  }
}

///|
/// Interleave 4 planar byte streams into an RGBA buffer. Each input has the
/// same pixel count `n`; output is `n * 4` bytes.
pub fn SimdBufferBytes::channel_merge(
  r : SimdBufferBytes,
  g : SimdBufferBytes,
  b : SimdBufferBytes,
  a : SimdBufferBytes,
  out : SimdBufferBytes,
) -> Unit {
  let n = r.len
  if g.len != n || b.len != n || a.len != n {
    abort("SimdBufferBytes.channel_merge: input lengths must match")
  }
  if out.len != n * 4 {
    abort("SimdBufferBytes.channel_merge: out length must be n * 4")
  }
  if n == 0 {
    return
  }
  buf_channel_merge(r.addr, g.addr, b.addr, a.addr, out.addr, n)
}

///|
/// Byte-wise linear interpolation: `out[i] = (a[i] * (256 - t) + b[i] * t) / 256`
/// where `t` is the integer interpolation factor in `0..=256` (0 → a only,
/// 256 → b only). Three buffers must have the same length.
pub fn SimdBufferBytes::lerp(
  a : SimdBufferBytes,
  b : SimdBufferBytes,
  t : Int,
  out : SimdBufferBytes,
) -> Unit {
  if a.len != b.len || a.len != out.len {
    abort("SimdBufferBytes.lerp: length mismatch")
  }
  let t_clamped = if t < 0 { 0 } else if t > 256 { 256 } else { t }
  let inv_t = 256 - t_clamped
  let n = a.len
  if n == 0 {
    return
  }
  // t=0 → out=a, t=256 → out=b; both weights would overflow u8, so copy.
  if t_clamped == 0 {
    buf_memcpy_bytes(a.addr, 0, out.addr, 0, n)
  } else if t_clamped == 256 {
    buf_memcpy_bytes(b.addr, 0, out.addr, 0, n)
  } else {
    buf_lerp(a.addr, b.addr, out.addr, n, inv_t, t_clamped)
  }
}

///|
/// 256-bin byte histogram. `bins.length() == 256`; each `bins[v]` is
/// incremented by the count of bytes equal to `v` in `self`. Scalar (wasm
/// SIMD has no efficient scatter for unbounded bin counts).
pub fn SimdBufferBytes::histogram(
  self : SimdBufferBytes,
  bins : SimdBuffer,
) -> Unit {
  if bins.len < 256 {
    abort("SimdBufferBytes.histogram: bins must have at least 256 slots")
  }
  for i in 0..<self.len {
    let v = buf_load_byte(self.addr + i)
    let addr = bins.addr + v * 4
    buf_store_i32(addr, buf_load_i32(addr) + 1)
  }
}

///|
/// Premultiplied source-over alpha blend with a single solid source color,
/// in-place over an RGBA8 destination buffer. Each `src_*` is `0..=255`.
/// For non-premultiplied input, the caller should pre-multiply
/// `src_r/g/b` by `src_a/255`.
///
/// Formula (per channel + per pixel): `out = src + dst * (255 - src_a) / 255`,
/// with saturation at 255. This is the standard "Porter-Duff source-over"
/// composition for premultiplied alpha; faster than the unpremultiplied
/// formula (no per-pixel divide).
///
/// Scalar for now; a SIMD i16x8 variant is planned for 0.4.0.
pub fn SimdBufferBytes::alpha_blend_solid(
  dst : SimdBufferBytes,
  src_r : Int,
  src_g : Int,
  src_b : Int,
  src_a : Int,
) -> Unit {
  if dst.len % 4 != 0 {
    abort(
      "SimdBufferBytes.alpha_blend_solid: dst must be a multiple of 4 bytes",
    )
  }
  let inv_a = 255 - src_a
  let n = dst.len / 4
  for i in 0..<n {
    let dr = buf_load_byte(dst.addr + i * 4)
    let dg = buf_load_byte(dst.addr + i * 4 + 1)
    let db = buf_load_byte(dst.addr + i * 4 + 2)
    let da = buf_load_byte(dst.addr + i * 4 + 3)
    let nr = src_r + dr * inv_a / 255
    let ng = src_g + dg * inv_a / 255
    let nb = src_b + db * inv_a / 255
    let na = src_a + da * inv_a / 255
    buf_store_byte(dst.addr + i * 4, if nr > 255 { 255 } else { nr })
    buf_store_byte(dst.addr + i * 4 + 1, if ng > 255 { 255 } else { ng })
    buf_store_byte(dst.addr + i * 4 + 2, if nb > 255 { 255 } else { nb })
    buf_store_byte(dst.addr + i * 4 + 3, if na > 255 { 255 } else { na })
  }
}

// ---- byte-wise binary arithmetic (Tier 1 — 0.3.0) ----

///|
/// Element-wise byte add (mod 256): `out[i] = (a[i] + b[i]) & 0xFF`.
/// Three buffers must have the same length.
pub fn SimdBufferBytes::byte_add(
  a : SimdBufferBytes,
  b : SimdBufferBytes,
  out : SimdBufferBytes,
) -> Unit {
  if a.len != b.len || a.len != out.len {
    abort("SimdBufferBytes.byte_add: length mismatch")
  }
  if a.len == 0 {
    return
  }
  buf_byte_add(a.addr, b.addr, out.addr, a.len)
}

///|
/// Element-wise byte subtract (mod 256): `out[i] = (a[i] - b[i]) & 0xFF`.
pub fn SimdBufferBytes::byte_sub(
  a : SimdBufferBytes,
  b : SimdBufferBytes,
  out : SimdBufferBytes,
) -> Unit {
  if a.len != b.len || a.len != out.len {
    abort("SimdBufferBytes.byte_sub: length mismatch")
  }
  if a.len == 0 {
    return
  }
  buf_byte_sub(a.addr, b.addr, out.addr, a.len)
}

///|
/// Element-wise byte average with rounding: `out[i] = (a[i] + b[i] + 1) / 2`.
/// Maps directly to `i8x16.avgr_u`. Useful for box blur, PNG Average filter,
/// midpoint color computation.
pub fn SimdBufferBytes::byte_avg(
  a : SimdBufferBytes,
  b : SimdBufferBytes,
  out : SimdBufferBytes,
) -> Unit {
  if a.len != b.len || a.len != out.len {
    abort("SimdBufferBytes.byte_avg: length mismatch")
  }
  if a.len == 0 {
    return
  }
  buf_byte_avg(a.addr, b.addr, out.addr, a.len)
}

///|
/// Saturating unsigned byte add: `out[i] = min(255, a[i] + b[i])`.
/// `i8x16.add_sat_u`. Useful for accumulator-style ops (AA coverage,
/// audio mixing) where overflow should clamp instead of wrap.
pub fn SimdBufferBytes::sat_add(
  a : SimdBufferBytes,
  b : SimdBufferBytes,
  out : SimdBufferBytes,
) -> Unit {
  if a.len != b.len || a.len != out.len {
    abort("SimdBufferBytes.sat_add: length mismatch")
  }
  if a.len == 0 {
    return
  }
  buf_sat_add(a.addr, b.addr, out.addr, a.len)
}

///|
/// Saturating unsigned byte subtract: `out[i] = max(0, a[i] - b[i])`.
/// `i8x16.sub_sat_u`.
pub fn SimdBufferBytes::sat_sub(
  a : SimdBufferBytes,
  b : SimdBufferBytes,
  out : SimdBufferBytes,
) -> Unit {
  if a.len != b.len || a.len != out.len {
    abort("SimdBufferBytes.sat_sub: length mismatch")
  }
  if a.len == 0 {
    return
  }
  buf_sat_sub(a.addr, b.addr, out.addr, a.len)
}

///|
/// Clamp each byte to `[lo, hi]`. SIMD chains `i8x16.max_u(lo)` then
/// `i8x16.min_u(hi)`. Useful for image normalization, audio limiting.
pub fn SimdBufferBytes::clamp(
  self : SimdBufferBytes,
  lo : Byte,
  hi : Byte,
  out : SimdBufferBytes,
) -> Unit {
  if self.len != out.len {
    abort("SimdBufferBytes.clamp: length mismatch")
  }
  if self.len == 0 {
    return
  }
  buf_clamp_u8(self.addr, lo.to_int(), hi.to_int(), out.addr, self.len)
}

///|
/// Subtract a shifted view of the same buffer from itself: `out[i] = (src[i] -
/// src[i - stride]) & 0xFF` for `i >= stride`, with `out[i] = src[i]` for
/// `i < stride`. This is PNG's Sub filter pattern; the SIMD body reads two
/// `v128.load`s offset by `stride`, with a scalar head + tail.
pub fn SimdBufferBytes::byte_sub_offset(
  src : SimdBufferBytes,
  stride : Int,
  out : SimdBufferBytes,
) -> Unit {
  if src.len != out.len {
    abort("SimdBufferBytes.byte_sub_offset: length mismatch")
  }
  if stride <= 0 {
    abort("SimdBufferBytes.byte_sub_offset: stride must be >= 1")
  }
  let n = src.len
  if n == 0 {
    return
  }
  // Head: i < stride → out[i] = src[i].
  let head = if stride > n { n } else { stride }
  for i in 0..<head {
    buf_store_byte(out.addr + i, buf_load_byte(src.addr + i))
  }
  if n > stride {
    buf_byte_sub_offset(src.addr, stride, out.addr, n)
  }
}

///|
/// In-place ASCII a-z → A-Z.
pub fn SimdBufferBytes::to_upper_ascii(self : SimdBufferBytes) -> Unit {
  if self.len == 0 {
    return
  }
  buf_to_upper_ascii(self.addr, self.len)
}

///|
/// Structural UTF-8 validation (per `scalar_validate_utf8_from`'s recipe).
/// SIMD ASCII fast-path skips all-ASCII 16-byte chunks; scalar state machine
/// takes over at the first chunk that contains any high-bit byte.
pub fn SimdBufferBytes::validate_utf8(self : SimdBufferBytes) -> Bool {
  let n = self.len
  let start = if n >= 16 { buf_first_non_ascii_chunk(self.addr, n) } else { 0 }
  self.validate_utf8_scalar(start)
}

///|
/// Scalar UTF-8 structural validation from `start` to end-of-buffer.
/// Inlined from `@internal.scalar_validate_utf8_from` so we don't depend on
/// FixedArray storage.
fn SimdBufferBytes::validate_utf8_scalar(
  self : SimdBufferBytes,
  start : Int,
) -> Bool {
  let n = self.len
  let mut i = start
  while i < n {
    let b = buf_load_byte(self.addr + i)
    if b < 0x80 {
      i = i + 1
    } else if b < 0xC2 {
      return false
    } else if b < 0xE0 {
      if i + 1 >= n {
        return false
      }
      if (buf_load_byte(self.addr + i + 1) & 0xC0) != 0x80 {
        return false
      }
      i = i + 2
    } else if b < 0xF0 {
      if i + 2 >= n {
        return false
      }
      if (buf_load_byte(self.addr + i + 1) & 0xC0) != 0x80 {
        return false
      }
      if (buf_load_byte(self.addr + i + 2) & 0xC0) != 0x80 {
        return false
      }
      i = i + 3
    } else if b < 0xF5 {
      if i + 3 >= n {
        return false
      }
      if (buf_load_byte(self.addr + i + 1) & 0xC0) != 0x80 {
        return false
      }
      if (buf_load_byte(self.addr + i + 2) & 0xC0) != 0x80 {
        return false
      }
      if (buf_load_byte(self.addr + i + 3) & 0xC0) != 0x80 {
        return false
      }
      i = i + 4
    } else {
      return false
    }
  }
  true
}

// ---- adler32 ----

///|
/// Adler-32 weight vector [16, 15, ..., 1]. Allocated once at module init in
/// linear memory; the address is reused by every adler32 call. Costs one
/// `memory.grow` page (~64 KiB) per process.
let adler_weights_addr : Int = {
  let buf = SimdBufferBytes::make(16)
  for i in 0..<16 {
    buf_store_byte(buf.addr + i, 16 - i)
  }
  buf.addr
}

///|
/// SIMD Adler-32 (RFC 1950 / zlib NMAX=5552). Returns the standard 32-bit
/// packed checksum.
pub fn SimdBufferBytes::adler32(self : SimdBufferBytes) -> UInt {
  let mod_adler : UInt = 65521
  let n = self.len
  let mut a : UInt = 1
  let mut b : UInt = 0
  let mut i = 0
  while i < n {
    let remaining = n - i
    let block = if remaining > 5552 { 5552 } else { remaining }
    let block_end = i + block
    let simd_chunks = block / 16
    if simd_chunks > 0 {
      let packed = buf_adler32_chunks(
        self.addr,
        i,
        simd_chunks,
        adler_weights_addr,
        a.reinterpret_as_int(),
        b.reinterpret_as_int(),
      )
      a = (packed & 0xFFFFFFFFL).to_int().reinterpret_as_uint()
      b = (packed.reinterpret_as_uint64() >> 32).to_int().reinterpret_as_uint()
      i = i + simd_chunks * 16
    }
    while i < block_end {
      a = a + buf_load_byte(self.addr + i).reinterpret_as_uint()
      b = b + a
      i = i + 1
    }
    a = a % mod_adler
    b = b % mod_adler
  }
  (b << 16) | a
}

// ---- inline-WAT primitives for the byte ops (identical bodies to the
//      FixedArray versions in simd_wasm_bytes.mbt) ----

///|
fn buf_popcount_bytes(addr : Int, len : Int) -> Int =
  #|(func (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local v128) local.get 1 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 0 local.get 2 i32.add v128.load i8x16.popcnt i16x8.extadd_pairwise_i8x16_u i32x4.extadd_pairwise_i16x8_u local.set 5 local.get 4 local.get 5 i32x4.extract_lane 0 i32.add local.get 5 i32x4.extract_lane 1 i32.add local.get 5 i32x4.extract_lane 2 i32.add local.get 5 i32x4.extract_lane 3 i32.add local.set 4 local.get 2 i32.const 16 i32.add local.set 2 br 0 end end block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.add i32.load8_u i32.popcnt i32.add local.set 4 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end local.get 4)

///|
fn buf_memcpy_bytes(
  src_addr : Int,
  si : Int,
  dst_addr : Int,
  di : Int,
  len : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 4 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 6 block loop local.get 5 local.get 6 i32.ge_s br_if 1 local.get 2 local.get 3 i32.add local.get 5 i32.add local.get 0 local.get 1 i32.add local.get 5 i32.add v128.load v128.store local.get 5 i32.const 16 i32.add local.set 5 br 0 end end block loop local.get 5 local.get 4 i32.ge_s br_if 1 local.get 2 local.get 3 i32.add local.get 5 i32.add local.get 0 local.get 1 i32.add local.get 5 i32.add i32.load8_u i32.store8 local.get 5 i32.const 1 i32.add local.set 5 br 0 end end)

///|
fn buf_memset_bytes(dst_addr : Int, off : Int, len : Int, value : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) (local v128) local.get 3 i8x16.splat local.set 6 local.get 2 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 0 local.get 1 i32.add local.get 4 i32.add local.get 6 v128.store local.get 4 i32.const 16 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 2 i32.ge_s br_if 1 local.get 0 local.get 1 i32.add local.get 4 i32.add local.get 3 i32.store8 local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
fn buf_equal_bytes(a : Int, b : Int, len : Int) -> Int =
  #|(func (param i32) (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) local.get 2 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 3 i32.add v128.load local.get 1 local.get 3 i32.add v128.load i8x16.eq i8x16.bitmask i32.const 65535 i32.ne i32.or local.set 5 local.get 3 i32.const 16 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 3 i32.add i32.load8_u local.get 1 local.get 3 i32.add i32.load8_u i32.ne i32.or local.set 5 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end local.get 5 i32.eqz)

///|
fn buf_find_byte(addr : Int, len : Int, needle : Int) -> Int =
  #|(func (param i32) (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local i32) (local v128) local.get 2 i8x16.splat local.set 7 local.get 1 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 4 i32.const -1 local.set 5 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 0 local.get 3 i32.add v128.load local.get 7 i8x16.eq i8x16.bitmask local.set 6 local.get 3 local.get 6 i32.ctz i32.add local.get 5 local.get 5 i32.const -1 i32.eq local.get 6 i32.mul select local.set 5 local.get 3 i32.const 16 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 1 i32.ge_s br_if 1 local.get 3 local.get 5 local.get 5 i32.const -1 i32.eq local.get 0 local.get 3 i32.add i32.load8_u local.get 2 i32.eq i32.and select local.set 5 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end local.get 5)

///|
fn buf_count_byte(addr : Int, len : Int, needle : Int) -> Int =
  #|(func (param i32) (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local v128) local.get 2 i8x16.splat local.set 6 local.get 1 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 3 i32.add v128.load local.get 6 i8x16.eq i8x16.bitmask i32.popcnt i32.add local.set 5 local.get 3 i32.const 16 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 1 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 3 i32.add i32.load8_u local.get 2 i32.eq i32.add local.set 5 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end local.get 5)

///|
fn buf_first_non_ascii_chunk(addr : Int, len : Int) -> Int =
  #|(func (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local i32) local.get 1 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 3 local.get 3 local.set 4 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 0 local.get 2 i32.add v128.load i8x16.bitmask local.set 5 local.get 2 local.get 4 local.get 4 local.get 3 i32.eq local.get 5 i32.mul select local.set 4 local.get 2 i32.const 16 i32.add local.set 2 br 0 end end local.get 4)

///|
fn buf_is_ascii(addr : Int, len : Int) -> Int =
  #|(func (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) local.get 1 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.add v128.load i8x16.bitmask i32.or local.set 4 local.get 2 i32.const 16 i32.add local.set 2 br 0 end end block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.add i32.load8_u i32.const 128 i32.and i32.or local.set 4 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end local.get 4 i32.eqz)

///|
fn buf_to_lower_ascii(addr : Int, len : Int) -> Unit =
  #|(func (param i32) (param i32) (local i32) (local i32) (local v128) (local v128) (local v128) (local v128) i32.const 65 i8x16.splat local.set 4 i32.const 90 i8x16.splat local.set 5 i32.const 32 i8x16.splat local.set 6 local.get 1 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 0 local.get 2 i32.add v128.load local.set 7 local.get 0 local.get 2 i32.add local.get 7 local.get 4 i8x16.ge_s local.get 7 local.get 5 i8x16.le_s v128.and local.get 6 v128.and local.get 7 i8x16.add v128.store local.get 2 i32.const 16 i32.add local.set 2 br 0 end end block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 0 local.get 2 i32.add local.get 0 local.get 2 i32.add i32.load8_u local.get 0 local.get 2 i32.add i32.load8_u i32.const 65 i32.ge_s local.get 0 local.get 2 i32.add i32.load8_u i32.const 90 i32.le_s i32.and i32.const 32 i32.mul i32.add i32.store8 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end)

///|
fn buf_to_upper_ascii(addr : Int, len : Int) -> Unit =
  #|(func (param i32) (param i32) (local i32) (local i32) (local v128) (local v128) (local v128) (local v128) i32.const 97 i8x16.splat local.set 4 i32.const 122 i8x16.splat local.set 5 i32.const 32 i8x16.splat local.set 6 local.get 1 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 0 local.get 2 i32.add v128.load local.set 7 local.get 0 local.get 2 i32.add local.get 7 local.get 7 local.get 4 i8x16.ge_s local.get 7 local.get 5 i8x16.le_s v128.and local.get 6 v128.and i8x16.sub v128.store local.get 2 i32.const 16 i32.add local.set 2 br 0 end end block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 0 local.get 2 i32.add local.get 0 local.get 2 i32.add i32.load8_u local.get 0 local.get 2 i32.add i32.load8_u i32.const 97 i32.ge_s local.get 0 local.get 2 i32.add i32.load8_u i32.const 122 i32.le_s i32.and i32.const 32 i32.mul i32.sub i32.store8 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end)

///|
fn buf_adler32_chunks(
  data_addr : Int,
  offset : Int,
  count16 : Int,
  weights_addr : Int,
  a : Int,
  b : Int,
) -> Int64 =
  #|(func (param i32) (param i32) (param i32) (param i32) (param i32) (param i32) (result i64) (local i32) (local i32) (local i32) (local i32) (local v128) (local v128) (local v128) local.get 3 v128.load local.set 10 local.get 1 local.get 2 i32.const 4 i32.shl i32.add local.set 7 local.get 1 local.set 6 block loop local.get 6 local.get 7 i32.ge_s br_if 1 local.get 0 local.get 6 i32.add v128.load local.set 11 local.get 11 i16x8.extadd_pairwise_i8x16_u i32x4.extadd_pairwise_i16x8_u local.set 12 local.get 12 i32x4.extract_lane 0 local.get 12 i32x4.extract_lane 1 i32.add local.get 12 i32x4.extract_lane 2 i32.add local.get 12 i32x4.extract_lane 3 i32.add local.set 8 local.get 11 local.get 10 i16x8.extmul_low_i8x16_u i32x4.extadd_pairwise_i16x8_u local.get 11 local.get 10 i16x8.extmul_high_i8x16_u i32x4.extadd_pairwise_i16x8_u i32x4.add local.set 12 local.get 12 i32x4.extract_lane 0 local.get 12 i32x4.extract_lane 1 i32.add local.get 12 i32x4.extract_lane 2 i32.add local.get 12 i32x4.extract_lane 3 i32.add local.set 9 local.get 5 local.get 4 i32.const 4 i32.shl i32.add local.get 9 i32.add local.set 5 local.get 4 local.get 8 i32.add local.set 4 local.get 6 i32.const 16 i32.add local.set 6 br 0 end end local.get 5 i64.extend_i32_u i64.const 32 i64.shl local.get 4 i64.extend_i32_u i64.or)

// ---- byte binary arithmetic (Tier 1 — 0.3.0) ----
//
// All six functions below share the same inline-WAT skeleton:
//   * Locals: 4 = vec_end (= len & ~15), 5 = i
//   * 16-byte chunk: v128.load a, v128.load b, OP, v128.store out
//   * Scalar tail: byte load, byte op, byte store
// Only the SIMD op (i8x16.{add,sub,avgr_u,add_sat_u,sub_sat_u}) and the
// scalar tail equivalent differ.

///|
fn buf_byte_add(a : Int, b : Int, out : Int, len : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 4 block loop local.get 5 local.get 4 i32.ge_s br_if 1 local.get 2 local.get 5 i32.add local.get 0 local.get 5 i32.add v128.load local.get 1 local.get 5 i32.add v128.load i8x16.add v128.store local.get 5 i32.const 16 i32.add local.set 5 br 0 end end block loop local.get 5 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 5 i32.add local.get 0 local.get 5 i32.add i32.load8_u local.get 1 local.get 5 i32.add i32.load8_u i32.add i32.store8 local.get 5 i32.const 1 i32.add local.set 5 br 0 end end)

///|
fn buf_byte_sub(a : Int, b : Int, out : Int, len : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 4 block loop local.get 5 local.get 4 i32.ge_s br_if 1 local.get 2 local.get 5 i32.add local.get 0 local.get 5 i32.add v128.load local.get 1 local.get 5 i32.add v128.load i8x16.sub v128.store local.get 5 i32.const 16 i32.add local.set 5 br 0 end end block loop local.get 5 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 5 i32.add local.get 0 local.get 5 i32.add i32.load8_u local.get 1 local.get 5 i32.add i32.load8_u i32.sub i32.store8 local.get 5 i32.const 1 i32.add local.set 5 br 0 end end)

///|
fn buf_byte_avg(a : Int, b : Int, out : Int, len : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 4 block loop local.get 5 local.get 4 i32.ge_s br_if 1 local.get 2 local.get 5 i32.add local.get 0 local.get 5 i32.add v128.load local.get 1 local.get 5 i32.add v128.load i8x16.avgr_u v128.store local.get 5 i32.const 16 i32.add local.set 5 br 0 end end block loop local.get 5 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 5 i32.add local.get 0 local.get 5 i32.add i32.load8_u local.get 1 local.get 5 i32.add i32.load8_u i32.add i32.const 1 i32.add i32.const 1 i32.shr_u i32.store8 local.get 5 i32.const 1 i32.add local.set 5 br 0 end end)

///|
fn buf_sat_add(a : Int, b : Int, out : Int, len : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) (local i32) local.get 3 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 4 block loop local.get 5 local.get 4 i32.ge_s br_if 1 local.get 2 local.get 5 i32.add local.get 0 local.get 5 i32.add v128.load local.get 1 local.get 5 i32.add v128.load i8x16.add_sat_u v128.store local.get 5 i32.const 16 i32.add local.set 5 br 0 end end block loop local.get 5 local.get 3 i32.ge_s br_if 1 local.get 0 local.get 5 i32.add i32.load8_u local.get 1 local.get 5 i32.add i32.load8_u i32.add local.tee 6 i32.const 255 i32.const 255 local.get 6 i32.gt_s select local.set 6 local.get 2 local.get 5 i32.add local.get 6 i32.store8 local.get 5 i32.const 1 i32.add local.set 5 br 0 end end)

///|
fn buf_sat_sub(a : Int, b : Int, out : Int, len : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) (local i32) local.get 3 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 4 block loop local.get 5 local.get 4 i32.ge_s br_if 1 local.get 2 local.get 5 i32.add local.get 0 local.get 5 i32.add v128.load local.get 1 local.get 5 i32.add v128.load i8x16.sub_sat_u v128.store local.get 5 i32.const 16 i32.add local.set 5 br 0 end end block loop local.get 5 local.get 3 i32.ge_s br_if 1 local.get 0 local.get 5 i32.add i32.load8_u local.get 1 local.get 5 i32.add i32.load8_u i32.sub local.set 6 local.get 6 i32.const 0 local.get 6 i32.const 0 i32.ge_s select local.set 6 local.get 2 local.get 5 i32.add local.get 6 i32.store8 local.get 5 i32.const 1 i32.add local.set 5 br 0 end end)

///|
/// Clamp: `out[i] = min(hi, max(lo, src[i]))` byte-wise.
/// Locals: 5 = vec_end, 6 = i, 7 = byte_tmp, 8 = lo_splat (v128), 9 = hi_splat (v128).
fn buf_clamp_u8(src : Int, lo : Int, hi : Int, out : Int, len : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) (local i32) (local v128) (local v128) local.get 1 i8x16.splat local.set 8 local.get 2 i8x16.splat local.set 9 local.get 4 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 5 block loop local.get 6 local.get 5 i32.ge_s br_if 1 local.get 3 local.get 6 i32.add local.get 0 local.get 6 i32.add v128.load local.get 8 i8x16.max_u local.get 9 i8x16.min_u v128.store local.get 6 i32.const 16 i32.add local.set 6 br 0 end end block loop local.get 6 local.get 4 i32.ge_s br_if 1 local.get 0 local.get 6 i32.add i32.load8_u local.set 7 local.get 7 local.get 1 local.get 7 local.get 1 i32.ge_s select local.set 7 local.get 7 local.get 2 local.get 7 local.get 2 i32.le_s select local.set 7 local.get 3 local.get 6 i32.add local.get 7 i32.store8 local.get 6 i32.const 1 i32.add local.set 6 br 0 end end)

///|
/// RGB → RGBA expansion: 4 pixels (12 src bytes) → 16 out bytes per iter via
/// one `i8x16.shuffle` that pulls R/G/B from the loaded vector and the alpha
/// byte from a splat in the shuffle's second operand. Scalar tail for the
/// remaining `npix % 4` pixels. The SIMD `v128.load` reads up to 3 bytes past
/// the logical RGB end into `SimdBuffer::make`'s 16-byte slack (unused lanes).
/// Locals: 4 = alpha_v (v128), 5 = p (pixel index), 6 = pix_end (multiple of 4).
fn buf_rgb_to_rgba(src : Int, out : Int, npix : Int, alpha : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local v128) (local i32) (local i32) local.get 3 i8x16.splat local.set 4 local.get 2 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 6 block loop local.get 5 local.get 6 i32.ge_s br_if 1 local.get 1 local.get 5 i32.const 2 i32.shl i32.add local.get 0 local.get 5 i32.const 1 i32.shl local.get 5 i32.add i32.add v128.load local.get 4 i8x16.shuffle 0 1 2 16 3 4 5 16 6 7 8 16 9 10 11 16 v128.store local.get 5 i32.const 4 i32.add local.set 5 br 0 end end block loop local.get 5 local.get 2 i32.ge_s br_if 1 local.get 1 local.get 5 i32.const 2 i32.shl i32.add local.get 0 local.get 5 i32.const 1 i32.shl local.get 5 i32.add i32.add i32.load8_u i32.store8 local.get 1 local.get 5 i32.const 2 i32.shl i32.add i32.const 1 i32.add local.get 0 local.get 5 i32.const 1 i32.shl local.get 5 i32.add i32.add i32.const 1 i32.add i32.load8_u i32.store8 local.get 1 local.get 5 i32.const 2 i32.shl i32.add i32.const 2 i32.add local.get 0 local.get 5 i32.const 1 i32.shl local.get 5 i32.add i32.add i32.const 2 i32.add i32.load8_u i32.store8 local.get 1 local.get 5 i32.const 2 i32.shl i32.add i32.const 3 i32.add local.get 3 i32.store8 local.get 5 i32.const 1 i32.add local.set 5 br 0 end end)

///|
/// Channel de-interleave (RGBA → planar), one specialised kernel per channel
/// because `i8x16.shuffle` lane indices are immediates. Each gathers the
/// channel byte of 4 pixels (`v128.load` of 16 src bytes) into the low 4
/// lanes via shuffle, then writes them with one `i32.store`; scalar tail.
/// Locals: 3 = p (pixel idx), 4 = pix_end, 5 = v.
fn buf_channel_extract_0(src : Int, out : Int, n : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (local i32) (local i32) (local v128) local.get 2 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 1 local.get 3 i32.add local.get 0 local.get 3 i32.const 2 i32.shl i32.add v128.load local.set 5 local.get 5 local.get 5 i8x16.shuffle 0 4 8 12 0 0 0 0 0 0 0 0 0 0 0 0 i32x4.extract_lane 0 i32.store local.get 3 i32.const 4 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 1 local.get 3 i32.add local.get 0 local.get 3 i32.const 2 i32.shl i32.add i32.load8_u i32.store8 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end)

///|
fn buf_channel_extract_1(src : Int, out : Int, n : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (local i32) (local i32) (local v128) local.get 2 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 1 local.get 3 i32.add local.get 0 local.get 3 i32.const 2 i32.shl i32.add v128.load local.set 5 local.get 5 local.get 5 i8x16.shuffle 1 5 9 13 0 0 0 0 0 0 0 0 0 0 0 0 i32x4.extract_lane 0 i32.store local.get 3 i32.const 4 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 1 local.get 3 i32.add local.get 0 local.get 3 i32.const 2 i32.shl i32.add i32.const 1 i32.add i32.load8_u i32.store8 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end)

///|
fn buf_channel_extract_2(src : Int, out : Int, n : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (local i32) (local i32) (local v128) local.get 2 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 1 local.get 3 i32.add local.get 0 local.get 3 i32.const 2 i32.shl i32.add v128.load local.set 5 local.get 5 local.get 5 i8x16.shuffle 2 6 10 14 0 0 0 0 0 0 0 0 0 0 0 0 i32x4.extract_lane 0 i32.store local.get 3 i32.const 4 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 1 local.get 3 i32.add local.get 0 local.get 3 i32.const 2 i32.shl i32.add i32.const 2 i32.add i32.load8_u i32.store8 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end)

///|
fn buf_channel_extract_3(src : Int, out : Int, n : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (local i32) (local i32) (local v128) local.get 2 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 1 local.get 3 i32.add local.get 0 local.get 3 i32.const 2 i32.shl i32.add v128.load local.set 5 local.get 5 local.get 5 i8x16.shuffle 3 7 11 15 0 0 0 0 0 0 0 0 0 0 0 0 i32x4.extract_lane 0 i32.store local.get 3 i32.const 4 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 1 local.get 3 i32.add local.get 0 local.get 3 i32.const 2 i32.shl i32.add i32.const 3 i32.add i32.load8_u i32.store8 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end)

///|
/// Channel interleave (4 planar streams → RGBA): 4 pixels per iter via two
/// zip `i8x16.shuffle`s (R+G, B+A into 16-bit pairs) then one final shuffle
/// that interleaves the pairs into `r g b a r g b a ...`; one `v128.store`.
/// Scalar tail. The 4 `v128.load`s over-read up to 12 bytes past each plane's
/// logical end into `SimdBuffer::make`'s 16-byte slack (unused lanes).
/// Locals: 6 = p, 7 = pix_end, 8 = rg (v128), 9 = ba (v128).
fn buf_channel_merge(
  r : Int,
  g : Int,
  b : Int,
  a : Int,
  out : Int,
  n : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) (local v128) (local v128) local.get 5 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 7 block loop local.get 6 local.get 7 i32.ge_s br_if 1 local.get 0 local.get 6 i32.add v128.load local.get 1 local.get 6 i32.add v128.load i8x16.shuffle 0 16 1 17 2 18 3 19 0 16 1 17 2 18 3 19 local.set 8 local.get 2 local.get 6 i32.add v128.load local.get 3 local.get 6 i32.add v128.load i8x16.shuffle 0 16 1 17 2 18 3 19 0 16 1 17 2 18 3 19 local.set 9 local.get 4 local.get 6 i32.const 2 i32.shl i32.add local.get 8 local.get 9 i8x16.shuffle 0 1 16 17 2 3 18 19 4 5 20 21 6 7 22 23 v128.store local.get 6 i32.const 4 i32.add local.set 6 br 0 end end block loop local.get 6 local.get 5 i32.ge_s br_if 1 local.get 4 local.get 6 i32.const 2 i32.shl i32.add local.get 0 local.get 6 i32.add i32.load8_u i32.store8 local.get 4 local.get 6 i32.const 2 i32.shl i32.add i32.const 1 i32.add local.get 1 local.get 6 i32.add i32.load8_u i32.store8 local.get 4 local.get 6 i32.const 2 i32.shl i32.add i32.const 2 i32.add local.get 2 local.get 6 i32.add i32.load8_u i32.store8 local.get 4 local.get 6 i32.const 2 i32.shl i32.add i32.const 3 i32.add local.get 3 local.get 6 i32.add i32.load8_u i32.store8 local.get 6 i32.const 1 i32.add local.set 6 br 0 end end)

///|
/// Byte lerp `out[i] = (a[i]*inv_t + b[i]*t) >> 8` for `inv_t`, `t` in 1..255
/// (both fit u8; t=0 / t=256 are handled by the caller as a plain copy).
/// 16 bytes per iter: two `i16x8.extmul_*_i8x16_u` weighted widenings per
/// half, `i16x8.add`, `i16x8.shr_u 8`, then a low-byte gather shuffle narrows
/// the two halves back to 16 u8; scalar tail.
/// Locals: 6 = i, 7 = vec_end, 8 = wa_v, 9 = wb_v, 10 = va, 11 = vb,
/// 12 = lo, 13 = hi.
fn buf_lerp(
  a : Int,
  b : Int,
  out : Int,
  len : Int,
  inv_t : Int,
  t : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) local.get 4 i8x16.splat local.set 8 local.get 5 i8x16.splat local.set 9 local.get 3 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 7 block loop local.get 6 local.get 7 i32.ge_s br_if 1 local.get 0 local.get 6 i32.add v128.load local.set 10 local.get 1 local.get 6 i32.add v128.load local.set 11 local.get 10 local.get 8 i16x8.extmul_low_i8x16_u local.get 11 local.get 9 i16x8.extmul_low_i8x16_u i16x8.add i32.const 8 i16x8.shr_u local.set 12 local.get 10 local.get 8 i16x8.extmul_high_i8x16_u local.get 11 local.get 9 i16x8.extmul_high_i8x16_u i16x8.add i32.const 8 i16x8.shr_u local.set 13 local.get 2 local.get 6 i32.add local.get 12 local.get 13 i8x16.shuffle 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 v128.store local.get 6 i32.const 16 i32.add local.set 6 br 0 end end block loop local.get 6 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 6 i32.add local.get 0 local.get 6 i32.add i32.load8_u local.get 4 i32.mul local.get 1 local.get 6 i32.add i32.load8_u local.get 5 i32.mul i32.add i32.const 8 i32.shr_u i32.store8 local.get 6 i32.const 1 i32.add local.set 6 br 0 end end)

///|
/// RGBA → grayscale `Y = (77*R + 150*G + 29*B) >> 8` (Rec. 601 fixed-point).
/// 4 pixels per iter: three gather `i8x16.shuffle`s pull R/G/B of 4 pixels
/// into the low lanes, each weighted-widened with `i16x8.extmul_low_i8x16_u`,
/// summed (`i16x8.add`), `>> 8`, then a low-byte gather shuffle + `i32.store`
/// writes the 4 Y bytes; scalar tail. Weights sum to 256 so the u16 lanes
/// never overflow (max 255*256 = 65280).
/// Locals: 3 = p, 4 = pix_end, 5/6/7 = wR/wG/wB, 8 = v, 9 = acc.
fn buf_rgba_to_grayscale(src : Int, out : Int, n : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (local i32) (local i32) (local v128) (local v128) (local v128) (local v128) (local v128) i32.const 77 i8x16.splat local.set 5 i32.const 150 i8x16.splat local.set 6 i32.const 29 i8x16.splat local.set 7 local.get 2 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 0 local.get 3 i32.const 2 i32.shl i32.add v128.load local.set 8 local.get 8 local.get 8 i8x16.shuffle 0 4 8 12 0 0 0 0 0 0 0 0 0 0 0 0 local.get 5 i16x8.extmul_low_i8x16_u local.get 8 local.get 8 i8x16.shuffle 1 5 9 13 0 0 0 0 0 0 0 0 0 0 0 0 local.get 6 i16x8.extmul_low_i8x16_u i16x8.add local.get 8 local.get 8 i8x16.shuffle 2 6 10 14 0 0 0 0 0 0 0 0 0 0 0 0 local.get 7 i16x8.extmul_low_i8x16_u i16x8.add i32.const 8 i16x8.shr_u local.set 9 local.get 1 local.get 3 i32.add local.get 9 local.get 9 i8x16.shuffle 0 2 4 6 0 0 0 0 0 0 0 0 0 0 0 0 i32x4.extract_lane 0 i32.store local.get 3 i32.const 4 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 1 local.get 3 i32.add local.get 0 local.get 3 i32.const 2 i32.shl i32.add i32.load8_u i32.const 77 i32.mul local.get 0 local.get 3 i32.const 2 i32.shl i32.add i32.const 1 i32.add i32.load8_u i32.const 150 i32.mul i32.add local.get 0 local.get 3 i32.const 2 i32.shl i32.add i32.const 2 i32.add i32.load8_u i32.const 29 i32.mul i32.add i32.const 8 i32.shr_u i32.store8 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end)

///|
/// Shifted-self subtraction (PNG Sub filter): for i in stride..<len,
/// `out[i] = (src[i] - src[i - stride]) & 0xFF`.
/// Locals: 4 = vec_end (= ((len - stride) & ~15) + stride), 5 = i.
fn buf_byte_sub_offset(src : Int, stride : Int, out : Int, len : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 local.get 1 i32.sub i32.const 4 i32.shr_s i32.const 4 i32.shl local.get 1 i32.add local.set 4 local.get 1 local.set 5 block loop local.get 5 local.get 4 i32.ge_s br_if 1 local.get 2 local.get 5 i32.add local.get 0 local.get 5 i32.add v128.load local.get 0 local.get 5 i32.add local.get 1 i32.sub v128.load i8x16.sub v128.store local.get 5 i32.const 16 i32.add local.set 5 br 0 end end block loop local.get 5 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 5 i32.add local.get 0 local.get 5 i32.add i32.load8_u local.get 0 local.get 5 i32.add local.get 1 i32.sub i32.load8_u i32.sub i32.store8 local.get 5 i32.const 1 i32.add local.set 5 br 0 end end)

// (Import keep-alive lives in simd_buffer_imports.mbt so it applies on
// every target including js, where no other simd_buffer file is built.)
