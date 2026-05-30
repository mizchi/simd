// SHA-512 (FIPS 180-4), scalar on all targets. 64-bit words (`UInt64`),
// 80 rounds, 128-byte blocks, big-endian. A single stream does not vectorise
// (serial round dependency); a multi-buffer SIMD path would be 2-way (i64x2)
// rather than the 4-way of the 32-bit hashes, so it's left for later.

///|
let k512 : FixedArray[UInt64] = [
  0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc,
  0x3956c25bf348b538, 0x59f111f1b605d019, 0x923f82a4af194f9b, 0xab1c5ed5da6d8118,
  0xd807aa98a3030242, 0x12835b0145706fbe, 0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2,
  0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235, 0xc19bf174cf692694,
  0xe49b69c19ef14ad2, 0xefbe4786384f25e3, 0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65,
  0x2de92c6f592b0275, 0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5,
  0x983e5152ee66dfab, 0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4,
  0xc6e00bf33da88fc2, 0xd5a79147930aa725, 0x06ca6351e003826f, 0x142929670a0e6e70,
  0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed, 0x53380d139d95b3df,
  0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b,
  0xa2bfe8a14cf10364, 0xa81a664bbc423001, 0xc24b8b70d0f89791, 0xc76c51a30654be30,
  0xd192e819d6ef5218, 0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8,
  0x19a4c116b8d2d0c8, 0x1e376c085141ab53, 0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8,
  0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb, 0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3,
  0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,
  0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915, 0xc67178f2e372532b,
  0xca273eceea26619c, 0xd186b8c721c0c207, 0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178,
  0x06f067aa72176fba, 0x0a637dc5a2c898a6, 0x113f9804bef90dae, 0x1b710b35131c471b,
  0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c,
  0x4cc5d4becb3e42b6, 0x597f299cfc657e2a, 0x5fcb6fab3ad6faec, 0x6c44198c4a475817,
]

///|
let h512_iv : FixedArray[UInt64] = [
  0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
  0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
]

///|
fn rotr64(x : UInt64, n : Int) -> UInt64 {
  (x >> n) | (x << (64 - n))
}

///|
fn load_be_u64(buf : FixedArray[Byte], off : Int) -> UInt64 {
  (buf[off].to_uint64() << 56) |
  (buf[off + 1].to_uint64() << 48) |
  (buf[off + 2].to_uint64() << 40) |
  (buf[off + 3].to_uint64() << 32) |
  (buf[off + 4].to_uint64() << 24) |
  (buf[off + 5].to_uint64() << 16) |
  (buf[off + 6].to_uint64() << 8) |
  buf[off + 7].to_uint64()
}

///|
/// Padded length for SHA-512: 0x80 + zero fill + 16-byte big-endian bit
/// length, to a multiple of 128.
fn sha512_padded_len(n : Int) -> Int {
  (n + 16) / 128 * 128 + 128
}

///|
fn sha512_scalar(data : Bytes) -> FixedArray[Byte] {
  let n = data.length()
  let buf = FixedArray::make(sha512_padded_len(n), b'\x00')
  for i in 0..<n {
    buf[i] = data[i]
  }
  buf[n] = b'\x80'
  // 128-bit big-endian bit length; the high 64 bits are 0 for any practical n.
  let bits = n.to_uint64() * 8UL
  let total = buf.length()
  for i in 0..<8 {
    buf[total - 1 - i] = ((bits >> (i * 8)) & 0xFFUL).to_int().to_byte()
  }
  let h = FixedArray::make(8, 0UL)
  for i in 0..<8 {
    h[i] = h512_iv[i]
  }
  let w : FixedArray[UInt64] = FixedArray::make(80, 0UL)
  let blocks = total / 128
  for blk in 0..<blocks {
    let base = blk * 128
    for t in 0..<16 {
      w[t] = load_be_u64(buf, base + t * 8)
    }
    for t in 16..<80 {
      let s0 = rotr64(w[t - 15], 1) ^ rotr64(w[t - 15], 8) ^ (w[t - 15] >> 7)
      let s1 = rotr64(w[t - 2], 19) ^ rotr64(w[t - 2], 61) ^ (w[t - 2] >> 6)
      w[t] = w[t - 16] + s0 + w[t - 7] + s1
    }
    let mut a = h[0]
    let mut b = h[1]
    let mut c = h[2]
    let mut d = h[3]
    let mut e = h[4]
    let mut f = h[5]
    let mut g = h[6]
    let mut hh = h[7]
    for t in 0..<80 {
      let s1 = rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41)
      let ch = (e & f) ^ (e.lnot() & g)
      let temp1 = hh + s1 + ch + k512[t] + w[t]
      let s0 = rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39)
      let maj = (a & b) ^ (a & c) ^ (b & c)
      let temp2 = s0 + maj
      hh = g
      g = f
      f = e
      e = d + temp1
      d = c
      c = b
      b = a
      a = temp1 + temp2
    }
    h[0] = h[0] + a
    h[1] = h[1] + b
    h[2] = h[2] + c
    h[3] = h[3] + d
    h[4] = h[4] + e
    h[5] = h[5] + f
    h[6] = h[6] + g
    h[7] = h[7] + hh
  }
  let out = FixedArray::make(64, b'\x00')
  for i in 0..<8 {
    for j in 0..<8 {
      out[i * 8 + j] = ((h[i] >> ((7 - j) * 8)) & 0xFFUL).to_int().to_byte()
    }
  }
  out
}

///|
/// SHA-512 digest of `data` (64 bytes). Scalar on every target.
pub fn sha512(data : Bytes) -> Bytes {
  Bytes::from_array(sha512_scalar(data))
}

///|
/// SHA-512 digest as a 128-character lowercase hex string.
pub fn sha512_hex(data : Bytes) -> String {
  to_hex(sha512_scalar(data))
}
