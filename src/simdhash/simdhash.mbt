// @simdhash — cryptographic digests over `Bytes`.
//
// SHA-256 (FIPS 180-4). The single-message `sha256` is scalar on every target
// (this file): one stream does not vectorise — the 64-round compression is a
// tight sequential dependency and wasm SIMD has no SHA-NI / CLMUL. The SIMD
// win is *multi-buffer*: `sha256_x4` hashes four independent messages in
// parallel, one per i32x4 lane (Intel `sha256_mb` style), via the inline-WAT
// kernel in `simdhash_wasm.mbt` (~2.8x batch on wasm). All paths produce
// byte-identical digests.

///|
/// SHA-256 round constants (first 32 bits of the fractional parts of the cube
/// roots of the first 64 primes).
let k256 : FixedArray[UInt] = [
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
  0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
  0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
  0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
  0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
  0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
  0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
  0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
  0xc67178f2,
]

///|
/// SHA-256 initial hash values (fractional parts of the square roots of the
/// first 8 primes).
let h256_iv : FixedArray[UInt] = [
  0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
  0x5be0cd19,
]

///|
fn rotr(x : UInt, n : Int) -> UInt {
  (x >> n) | (x << (32 - n))
}

///|
/// Length (in bytes) of the message after SHA-256 padding: append `0x80`,
/// zero-pad, then an 8-byte big-endian bit length, to a multiple of 64.
fn sha256_padded_len(n : Int) -> Int {
  // n + 1 (the 0x80) + 8 (length), rounded up to a multiple of 64.
  (n + 8) / 64 * 64 + 64
}

///|
/// Write the SHA-256 padding for a message of `n` bytes into `buf` (which
/// already holds the `n` message bytes at `[0, n)` and has length
/// `sha256_padded_len(n)`).
fn sha256_pad_into(buf : FixedArray[Byte], n : Int) -> Unit {
  let total = buf.length()
  buf[n] = b'\x80'
  for i in (n + 1)..<total {
    buf[i] = b'\x00'
  }
  // 64-bit big-endian bit length in the last 8 bytes.
  let bits = n.to_int64() * 8L
  for i in 0..<8 {
    buf[total - 1 - i] = ((bits >> (i * 8)) & 0xFFL).to_int().to_byte()
  }
}

///|
fn load_be_u32(buf : FixedArray[Byte], off : Int) -> UInt {
  (buf[off].to_uint() << 24) |
  (buf[off + 1].to_uint() << 16) |
  (buf[off + 2].to_uint() << 8) |
  buf[off + 3].to_uint()
}

///|
/// Scalar SHA-256 over `buf[0, blocks*64)` (already padded), updating the
/// 8-word state `h` in place.
fn sha256_compress(h : FixedArray[UInt], buf : FixedArray[Byte]) -> Unit {
  let blocks = buf.length() / 64
  let w : FixedArray[UInt] = FixedArray::make(64, 0)
  for blk in 0..<blocks {
    let base = blk * 64
    for t in 0..<16 {
      w[t] = load_be_u32(buf, base + t * 4)
    }
    for t in 16..<64 {
      let s0 = rotr(w[t - 15], 7) ^ rotr(w[t - 15], 18) ^ (w[t - 15] >> 3)
      let s1 = rotr(w[t - 2], 17) ^ rotr(w[t - 2], 19) ^ (w[t - 2] >> 10)
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
    for t in 0..<64 {
      let s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
      let ch = (e & f) ^ (e.lnot() & g)
      let temp1 = hh + s1 + ch + k256[t] + w[t]
      let s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
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
}

///|
/// Build the padded message buffer for `data` (length `sha256_padded_len`).
fn pad_message(data : Bytes) -> FixedArray[Byte] {
  let n = data.length()
  let buf = FixedArray::make(sha256_padded_len(n), b'\x00')
  for i in 0..<n {
    buf[i] = data[i]
  }
  sha256_pad_into(buf, n)
  buf
}

///|
/// Scalar SHA-256. Returns the 32-byte digest. Used as the all-targets
/// reference and the off-wasm path for `sha256_x4`.
fn sha256_scalar(data : Bytes) -> FixedArray[Byte] {
  let buf = pad_message(data)
  let h = FixedArray::make(8, 0U)
  for i in 0..<8 {
    h[i] = h256_iv[i]
  }
  sha256_compress(h, buf)
  let out = FixedArray::make(32, b'\x00')
  for i in 0..<8 {
    out[i * 4] = ((h[i] >> 24) & 0xFF).reinterpret_as_int().to_byte()
    out[i * 4 + 1] = ((h[i] >> 16) & 0xFF).reinterpret_as_int().to_byte()
    out[i * 4 + 2] = ((h[i] >> 8) & 0xFF).reinterpret_as_int().to_byte()
    out[i * 4 + 3] = (h[i] & 0xFF).reinterpret_as_int().to_byte()
  }
  out
}

///|
let hex_chars : FixedArray[Byte] = [
  b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9', b'a', b'b', b'c', b'd',
  b'e', b'f',
]

///|
/// Lowercase hex of a digest. Built as UTF-16LE (each ASCII hex char as a
/// `(byte, 0x00)` pair) so `Bytes::to_unchecked_string` — which reinterprets
/// the buffer as UTF-16LE — yields the correct ASCII string.
fn to_hex(digest : FixedArray[Byte]) -> String {
  let n = digest.length()
  let out = FixedArray::make(n * 4, b'\x00')
  for i in 0..<n {
    let v = digest[i].to_int()
    out[i * 4] = hex_chars[(v >> 4) & 0xF]
    out[i * 4 + 2] = hex_chars[v & 0xF]
  }
  Bytes::from_array(out).to_unchecked_string(offset=0, length=n * 4)
}

///|
/// SHA-256 digest of `data` (32 bytes). Scalar on every target (a single
/// message does not vectorise; use `sha256_x4` for batch throughput on wasm).
pub fn sha256(data : Bytes) -> Bytes {
  Bytes::from_array(sha256_scalar(data))
}

///|
/// SHA-256 digest as a 64-character lowercase hex string.
pub fn sha256_hex(data : Bytes) -> String {
  to_hex(sha256_scalar(data))
}

///|
/// Write the 32-byte SHA-256 digest of `data` into `out[off, off+32)`.
fn digest_into(data : Bytes, out : FixedArray[Byte], off : Int) -> Unit {
  let d = sha256_scalar(data)
  for i in 0..<32 {
    out[off + i] = d[i]
  }
}

///|
fn slice32(out : FixedArray[Byte], off : Int) -> Bytes {
  let a = FixedArray::make(32, b'\x00')
  for i in 0..<32 {
    a[i] = out[off + i]
  }
  Bytes::from_array(a)
}

///|
/// Hash four messages, returning four 32-byte digests. This is the batch
/// entry point: on wasm with four equal-length inputs it runs a 4-way
/// multi-buffer SIMD compression (one message per `i32x4` lane — the shape
/// where SHA-256 actually vectorises). The result is always identical to four
/// independent `sha256` calls. Unequal-length inputs (and non-wasm targets)
/// take the scalar path. On wasm, equal-length batches get ~2.8x over four
/// separate `sha256` calls.
pub fn sha256_x4(
  m0 : Bytes,
  m1 : Bytes,
  m2 : Bytes,
  m3 : Bytes,
) -> (Bytes, Bytes, Bytes, Bytes) {
  let out = FixedArray::make(128, b'\x00')
  let n = m0.length()
  if m1.length() == n && m2.length() == n && m3.length() == n {
    imp_sha256_x4_eq(m0, m1, m2, m3, out)
  } else {
    digest_into(m0, out, 0)
    digest_into(m1, out, 32)
    digest_into(m2, out, 64)
    digest_into(m3, out, 96)
  }
  (slice32(out, 0), slice32(out, 32), slice32(out, 64), slice32(out, 96))
}
