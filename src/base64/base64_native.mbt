// Native Base64: binds to the gcc/clang-compiled C kernels in base64.c.
// Same public API + results as the scalar / wasm paths.

///|
#borrow(input, out)
extern "C" fn b64_encode_ffi(
  input : FixedArray[Byte],
  n : Int,
  out : FixedArray[Byte],
) -> Unit = "b64_encode"

///|
#borrow(input, out)
extern "C" fn b64_decode_ffi(
  input : FixedArray[Byte],
  n : Int,
  out : FixedArray[Byte],
) -> Int = "b64_decode"

///|
/// Encode `input` as Base64 (RFC 4648, standard alphabet, `=` padding).
pub fn encode(input : FixedArray[Byte]) -> FixedArray[Byte] {
  let n = input.length()
  let out = FixedArray::make(encoded_len(n), b'\x00')
  if n > 0 {
    b64_encode_ffi(input, n, out)
  }
  out
}

///|
/// Decode a Base64 buffer. `None` if the length is not a multiple of 4 or any
/// non-alphabet character appears (other than valid trailing `=`).
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
  if b64_decode_ffi(input, n, out) < 0 {
    None
  } else {
    Some(out)
  }
}
