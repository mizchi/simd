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
/// Encode `input` as Base64 into the caller-supplied `out` (no allocation).
/// `out.length()` must be at least `base64_encoded_len(input.length())`.
pub fn base64_encode_into(
  input : FixedArray[Byte],
  out : FixedArray[Byte],
) -> Unit {
  let n = input.length()
  if n > 0 {
    b64_encode_ffi(input, n, out)
  }
}

///|
/// Decode a Base64 buffer into the caller-supplied `out` (no allocation),
/// returning the number of bytes written or `None` on invalid input.
/// `out.length()` must be at least `base64_decoded_len(n, base64_count_padding(input))`.
pub fn base64_decode_into(
  input : FixedArray[Byte],
  out : FixedArray[Byte],
) -> Int? {
  let n = input.length()
  if n == 0 {
    return Some(0)
  }
  if n % 4 != 0 {
    return None
  }
  let pad = base64_count_padding(input)
  if b64_decode_ffi(input, n, out) < 0 {
    None
  } else {
    Some(base64_decoded_len(n, pad))
  }
}
