// SHA-1 (FIPS 180-4), scalar on all targets. Shares the big-endian padding
// with SHA-256.
//
// NOTE: SHA-1 is cryptographically broken for collision resistance. It is
// provided for legacy interop (e.g. git object ids, older protocol
// checksums), not for new security-sensitive uses — prefer `sha256`.

///|
let sha1_iv : FixedArray[UInt] = [
  0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0,
]

///|
fn sha1_scalar(data : Bytes) -> FixedArray[Byte] {
  let buf = pad_message(data)
  let blocks = buf.length() / 64
  let h = FixedArray::make(5, 0U)
  for i in 0..<5 {
    h[i] = sha1_iv[i]
  }
  let w : FixedArray[UInt] = FixedArray::make(80, 0)
  for blk in 0..<blocks {
    let base = blk * 64
    for t in 0..<16 {
      w[t] = load_be_u32(buf, base + t * 4)
    }
    for t in 16..<80 {
      w[t] = rotl(w[t - 3] ^ w[t - 8] ^ w[t - 14] ^ w[t - 16], 1)
    }
    let mut a = h[0]
    let mut b = h[1]
    let mut c = h[2]
    let mut d = h[3]
    let mut e = h[4]
    for t in 0..<80 {
      let f = if t < 20 {
        (b & c) | (b.lnot() & d)
      } else if t < 40 {
        b ^ c ^ d
      } else if t < 60 {
        (b & c) | (b & d) | (c & d)
      } else {
        b ^ c ^ d
      }
      let k = if t < 20 {
        0x5A827999U
      } else if t < 40 {
        0x6ED9EBA1U
      } else if t < 60 {
        0x8F1BBCDCU
      } else {
        0xCA62C1D6U
      }
      let temp = rotl(a, 5) + f + e + k + w[t]
      e = d
      d = c
      c = rotl(b, 30)
      b = a
      a = temp
    }
    h[0] = h[0] + a
    h[1] = h[1] + b
    h[2] = h[2] + c
    h[3] = h[3] + d
    h[4] = h[4] + e
  }
  let out = FixedArray::make(20, b'\x00')
  for i in 0..<5 {
    out[i * 4] = ((h[i] >> 24) & 0xFF).reinterpret_as_int().to_byte()
    out[i * 4 + 1] = ((h[i] >> 16) & 0xFF).reinterpret_as_int().to_byte()
    out[i * 4 + 2] = ((h[i] >> 8) & 0xFF).reinterpret_as_int().to_byte()
    out[i * 4 + 3] = (h[i] & 0xFF).reinterpret_as_int().to_byte()
  }
  out
}

///|
/// SHA-1 digest of `data` (20 bytes). Scalar on every target. Legacy use only
/// (SHA-1 is not collision-resistant) — prefer `sha256`.
pub fn sha1(data : Bytes) -> Bytes {
  Bytes::from_array(sha1_scalar(data))
}

///|
/// SHA-1 digest as a 40-character lowercase hex string.
pub fn sha1_hex(data : Bytes) -> String {
  to_hex(sha1_scalar(data))
}

///|
/// Hash four messages with SHA-1, returning four 20-byte digests.
pub fn sha1_x4(
  m0 : Bytes,
  m1 : Bytes,
  m2 : Bytes,
  m3 : Bytes,
) -> (Bytes, Bytes, Bytes, Bytes) {
  (sha1(m0), sha1(m1), sha1(m2), sha1(m3))
}
