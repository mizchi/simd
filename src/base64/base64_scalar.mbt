///|
/// Encode `input` as Base64 (RFC 4648, standard alphabet, with `=` padding).
/// Scalar fallback for js / wasm-gc / native targets.
pub fn encode(input : FixedArray[Byte]) -> FixedArray[Byte] {
  let out = FixedArray::make(encoded_len(input.length()), b'\x00')
  scalar_encode(input, out)
  out
}

///|
/// Decode a Base64 buffer (standard alphabet, `=`-padded). Returns `None` if
/// the input length is not a multiple of 4 or if any character is not in the
/// alphabet (other than valid trailing `=`).
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
  match scalar_decode(input, out) {
    Some(_) => Some(out)
    None => None
  }
}
