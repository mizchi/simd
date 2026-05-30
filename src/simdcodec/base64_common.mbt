// Shared Base64 surface (all targets): length helpers, padding count, and the
// allocating `base64_encode` / `base64_decode` wrappers. The actual work is done by the
// per-target `base64_encode_into` / `base64_decode_into` (SIMD on wasm, C FFI on native,
// scalar on wasm-gc / js). The scalar tables + reference base64_encode/base64_decode live in
// `base64_fallback.mbt` (every target except native).

///|
/// Compute the encoded length (with padding) for an input of `n` bytes.
pub fn base64_encoded_len(n : Int) -> Int {
  (n + 2) / 3 * 4
}

///|
/// Compute the expected decoded length of a Base64-encoded buffer of `n`
/// bytes that ends in `pad` `=` characters. Caller must have validated that
/// `n` is a multiple of 4 and `pad` is 0, 1, or 2.
pub fn base64_decoded_len(n : Int, pad : Int) -> Int {
  n / 4 * 3 - pad
}

///|
/// Count trailing `=` padding bytes (0, 1, or 2). Caller has validated that
/// the input length is a multiple of 4.
pub fn base64_count_padding(input : FixedArray[Byte]) -> Int {
  let n = input.length()
  if n == 0 {
    return 0
  }
  if input[n - 1] != b'=' {
    return 0
  }
  if n >= 2 && input[n - 2] == b'=' {
    return 2
  }
  1
}

///|
/// Encode `input` as Base64 (RFC 4648, standard alphabet, `=` padding),
/// allocating the output. For a hot loop that can reuse a buffer, size one
/// with `base64_encoded_len` and call `base64_encode_into` instead (no allocation).
pub fn base64_encode(input : FixedArray[Byte]) -> FixedArray[Byte] {
  let out = FixedArray::make(base64_encoded_len(input.length()), b'\x00')
  base64_encode_into(input, out)
  out
}

///|
/// Decode a Base64 buffer, allocating the output. `None` if the length is not
/// a multiple of 4 or any non-alphabet character appears (other than valid
/// trailing `=`). For a hot loop, size a buffer with
/// `base64_decoded_len(n, base64_count_padding(input))` and call `base64_decode_into`.
pub fn base64_decode(input : FixedArray[Byte]) -> FixedArray[Byte]? {
  let n = input.length()
  if n == 0 {
    return Some(FixedArray::make(0, b'\x00'))
  }
  if n % 4 != 0 {
    return None
  }
  let pad = base64_count_padding(input)
  let out = FixedArray::make(base64_decoded_len(n, pad), b'\x00')
  match base64_decode_into(input, out) {
    Some(_) => Some(out)
    None => None
  }
}
