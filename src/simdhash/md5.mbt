// MD5 (RFC 1321), scalar on all targets. MD5 is little-endian throughout
// (word loads, length suffix, and digest output) — unlike SHA-1/256.
//
// NOTE: MD5 is cryptographically broken (collisions are trivial). Provided for
// legacy interop and non-security checksums (ETags, content addressing of
// trusted data) only — never for integrity against an adversary.

///|
/// Per-round constants: floor(2^32 * abs(sin(i + 1))).
let md5_k : FixedArray[UInt] = [
  0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
  0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
  0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
  0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
  0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
  0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
  0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244,
  0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
  0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
  0xeb86d391,
]

///|
/// Per-round left-rotation amounts.
let md5_s : FixedArray[Int] = [
  7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9,
  14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
  4, 11, 16, 23, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
]

///|
let md5_iv : FixedArray[UInt] = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476]

///|
fn load_le_u32(buf : FixedArray[Byte], off : Int) -> UInt {
  buf[off].to_uint() |
  (buf[off + 1].to_uint() << 8) |
  (buf[off + 2].to_uint() << 16) |
  (buf[off + 3].to_uint() << 24)
}

///|
/// Padded message for MD5 — same `0x80` + zero fill as SHA, but the 64-bit
/// bit-length suffix is **little-endian**.
fn md5_pad_message(data : Bytes) -> FixedArray[Byte] {
  let n = data.length()
  let buf = FixedArray::make(sha256_padded_len(n), b'\x00')
  for i in 0..<n {
    buf[i] = data[i]
  }
  buf[n] = b'\x80'
  let total = buf.length()
  let bits = n.to_int64() * 8L
  for i in 0..<8 {
    buf[total - 8 + i] = ((bits >> (i * 8)) & 0xFFL).to_int().to_byte()
  }
  buf
}

///|
fn md5_scalar(data : Bytes) -> FixedArray[Byte] {
  let buf = md5_pad_message(data)
  let blocks = buf.length() / 64
  let mut a0 = md5_iv[0]
  let mut b0 = md5_iv[1]
  let mut c0 = md5_iv[2]
  let mut d0 = md5_iv[3]
  let m : FixedArray[UInt] = FixedArray::make(16, 0)
  for blk in 0..<blocks {
    let base = blk * 64
    for j in 0..<16 {
      m[j] = load_le_u32(buf, base + j * 4)
    }
    let mut a = a0
    let mut b = b0
    let mut c = c0
    let mut d = d0
    for i in 0..<64 {
      let f = if i < 16 {
        (b & c) | (b.lnot() & d)
      } else if i < 32 {
        (d & b) | (d.lnot() & c)
      } else if i < 48 {
        b ^ c ^ d
      } else {
        c ^ (b | d.lnot())
      }
      let g = if i < 16 {
        i
      } else if i < 32 {
        (5 * i + 1) % 16
      } else if i < 48 {
        (3 * i + 5) % 16
      } else {
        7 * i % 16
      }
      let f2 = f + a + md5_k[i] + m[g]
      a = d
      d = c
      c = b
      b = b + rotl(f2, md5_s[i])
    }
    a0 = a0 + a
    b0 = b0 + b
    c0 = c0 + c
    d0 = d0 + d
  }
  let hs = [a0, b0, c0, d0]
  let out = FixedArray::make(16, b'\x00')
  for i in 0..<4 {
    // little-endian output
    out[i * 4] = (hs[i] & 0xFF).reinterpret_as_int().to_byte()
    out[i * 4 + 1] = ((hs[i] >> 8) & 0xFF).reinterpret_as_int().to_byte()
    out[i * 4 + 2] = ((hs[i] >> 16) & 0xFF).reinterpret_as_int().to_byte()
    out[i * 4 + 3] = ((hs[i] >> 24) & 0xFF).reinterpret_as_int().to_byte()
  }
  out
}

///|
/// MD5 digest of `data` (16 bytes). Scalar on every target. Legacy /
/// non-security checksums only (MD5 is collision-broken).
pub fn md5(data : Bytes) -> Bytes {
  Bytes::from_array(md5_scalar(data))
}

///|
/// MD5 digest as a 32-character lowercase hex string.
pub fn md5_hex(data : Bytes) -> String {
  to_hex(md5_scalar(data))
}

///|
/// Write the 16-byte MD5 digest of `data` into `out[off, off+16)`.
fn md5_into(data : Bytes, out : FixedArray[Byte], off : Int) -> Unit {
  let d = md5_scalar(data)
  for i in 0..<16 {
    out[off + i] = d[i]
  }
}

///|
/// Hash four messages with MD5, returning four 16-byte digests. On native
/// with four equal-length inputs this runs a 4-way multi-buffer SSE2 / NEON
/// kernel; otherwise it is four scalar digests. Always identical to four
/// `md5` calls.
pub fn md5_x4(
  m0 : Bytes,
  m1 : Bytes,
  m2 : Bytes,
  m3 : Bytes,
) -> (Bytes, Bytes, Bytes, Bytes) {
  let out = FixedArray::make(64, b'\x00')
  let n = m0.length()
  if m1.length() == n && m2.length() == n && m3.length() == n {
    imp_md5_x4_eq(m0, m1, m2, m3, out)
  } else {
    md5_into(m0, out, 0)
    md5_into(m1, out, 16)
    md5_into(m2, out, 32)
    md5_into(m3, out, 48)
  }
  (
    slice_n(out, 0, 16),
    slice_n(out, 16, 16),
    slice_n(out, 32, 16),
    slice_n(out, 48, 16),
  )
}
