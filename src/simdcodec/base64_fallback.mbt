// Scalar Base64 tables + reference base64_encode/base64_decode. Built on every target
// EXCEPT native (which uses the C FFI kernels in base64.c). wasm uses these
// for its scalar tail; wasm-gc / js use them for the whole base64_encode/base64_decode.

///|
/// Standard Base64 alphabet (RFC 4648 §4): A-Z, a-z, 0-9, +, /.
/// Indexed by 6-bit value, gives the ASCII byte to emit.
let encode_table : FixedArray[Byte] = {
  let t = FixedArray::make(64, b'\x00')
  for i in 0..<26 {
    t[i] = (i + 65).to_byte() // 'A'..'Z'
  }
  for i in 0..<26 {
    t[26 + i] = (i + 97).to_byte() // 'a'..'z'
  }
  for i in 0..<10 {
    t[52 + i] = (i + 48).to_byte() // '0'..'9'
  }
  t[62] = b'+'
  t[63] = b'/'
  t
}

///|
/// Inverse table: ASCII byte → 6-bit value, or `INVALID` if the byte is not
/// part of the Base64 alphabet. Size 256.
let decode_table : FixedArray[Int] = {
  let t = FixedArray::make(256, INVALID_MARKER)
  for i in 0..<64 {
    t[encode_table[i].to_int()] = i
  }
  t
}

///|
/// Sentinel for "not a Base64 character" in `decode_table`. Picked as a value
/// outside the 0..63 range so callers can branch on `< 0`.
const INVALID_MARKER : Int = -1

///|
/// Scalar Base64 base64_encode. `out` must have at least `base64_encoded_len(input.length())`
/// bytes available.
fn scalar_encode(input : FixedArray[Byte], out : FixedArray[Byte]) -> Unit {
  let n = input.length()
  let triples = n / 3
  for i in 0..<triples {
    let p = i * 3
    let q = i * 4
    let b0 = input[p].to_int()
    let b1 = input[p + 1].to_int()
    let b2 = input[p + 2].to_int()
    let v = (b0 << 16) | (b1 << 8) | b2
    out[q] = encode_table[(v >> 18) & 63]
    out[q + 1] = encode_table[(v >> 12) & 63]
    out[q + 2] = encode_table[(v >> 6) & 63]
    out[q + 3] = encode_table[v & 63]
  }
  let rem = n - triples * 3
  let q = triples * 4
  if rem == 1 {
    let v = input[triples * 3].to_int() << 16
    out[q] = encode_table[(v >> 18) & 63]
    out[q + 1] = encode_table[(v >> 12) & 63]
    out[q + 2] = b'='
    out[q + 3] = b'='
  } else if rem == 2 {
    let b0 = input[triples * 3].to_int()
    let b1 = input[triples * 3 + 1].to_int()
    let v = (b0 << 16) | (b1 << 8)
    out[q] = encode_table[(v >> 18) & 63]
    out[q + 1] = encode_table[(v >> 12) & 63]
    out[q + 2] = encode_table[(v >> 6) & 63]
    out[q + 3] = b'='
  }
}

///|
/// Scalar Base64 base64_decode. Returns the number of bytes written, or `None` on
/// invalid input. `out` must have at least `base64_decoded_len(input.length(), pad)`
/// bytes; `input.length()` must be a multiple of 4.
fn scalar_decode(input : FixedArray[Byte], out : FixedArray[Byte]) -> Int? {
  let n = input.length()
  if n == 0 {
    return Some(0)
  }
  if n % 4 != 0 {
    return None
  }
  let pad = if input[n - 1] == b'=' {
    if input[n - 2] == b'=' {
      2
    } else {
      1
    }
  } else {
    0
  }
  let full_quads = (n - 4) / 4
  for i in 0..<full_quads {
    let p = i * 4
    let q = i * 3
    let d0 = decode_table[input[p].to_int()]
    let d1 = decode_table[input[p + 1].to_int()]
    let d2 = decode_table[input[p + 2].to_int()]
    let d3 = decode_table[input[p + 3].to_int()]
    if (d0 | d1 | d2 | d3) < 0 {
      return None
    }
    let v = (d0 << 18) | (d1 << 12) | (d2 << 6) | d3
    out[q] = (v >> 16).to_byte()
    out[q + 1] = (v >> 8).to_byte()
    out[q + 2] = v.to_byte()
  }
  let p = full_quads * 4
  let q = full_quads * 3
  let d0 = decode_table[input[p].to_int()]
  let d1 = decode_table[input[p + 1].to_int()]
  if (d0 | d1) < 0 {
    return None
  }
  if pad == 2 {
    let v = (d0 << 18) | (d1 << 12)
    out[q] = (v >> 16).to_byte()
    Some(q + 1)
  } else {
    let d2 = decode_table[input[p + 2].to_int()]
    if d2 < 0 {
      return None
    }
    if pad == 1 {
      let v = (d0 << 18) | (d1 << 12) | (d2 << 6)
      out[q] = (v >> 16).to_byte()
      out[q + 1] = (v >> 8).to_byte()
      Some(q + 2)
    } else {
      let d3 = decode_table[input[p + 3].to_int()]
      if d3 < 0 {
        return None
      }
      let v = (d0 << 18) | (d1 << 12) | (d2 << 6) | d3
      out[q] = (v >> 16).to_byte()
      out[q + 1] = (v >> 8).to_byte()
      out[q + 2] = v.to_byte()
      Some(q + 3)
    }
  }
}
