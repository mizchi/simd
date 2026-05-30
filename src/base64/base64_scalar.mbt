///|
/// Encode `input` as Base64 into the caller-supplied `out` (no allocation).
/// `out.length()` must be at least `encoded_len(input.length())`. Scalar
/// fallback for js / wasm-gc targets.
pub fn encode_into(input : FixedArray[Byte], out : FixedArray[Byte]) -> Unit {
  scalar_encode(input, out)
}

///|
/// Decode a Base64 buffer into the caller-supplied `out` (no allocation),
/// returning the number of bytes written or `None` on invalid input (length
/// not a multiple of 4, or a non-alphabet character other than trailing `=`).
/// `out.length()` must be at least `decoded_len(n, count_padding(input))`.
pub fn decode_into(input : FixedArray[Byte], out : FixedArray[Byte]) -> Int? {
  scalar_decode(input, out)
}
