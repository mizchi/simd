// Native `*_x4` workers: bind to the SSE2 / NEON multi-buffer kernels in
// `simdhash.c`. `Bytes` crosses the FFI as a `const uint8_t*` to byte[0]
// under `#borrow` (same ABI as `FixedArray[Byte]`); the C does its own
// padding so we just pass the four message pointers + common length.

///|
#borrow(m0, m1, m2, m3, out)
extern "C" fn mb_sha256_x4_ffi(
  m0 : Bytes,
  m1 : Bytes,
  m2 : Bytes,
  m3 : Bytes,
  n : Int,
  out : FixedArray[Byte],
) -> Unit = "mb_sha256_x4"

///|
#borrow(m0, m1, m2, m3, out)
extern "C" fn mb_sha1_x4_ffi(
  m0 : Bytes,
  m1 : Bytes,
  m2 : Bytes,
  m3 : Bytes,
  n : Int,
  out : FixedArray[Byte],
) -> Unit = "mb_sha1_x4"

///|
fn imp_sha256_x4_eq(
  m0 : Bytes,
  m1 : Bytes,
  m2 : Bytes,
  m3 : Bytes,
  out : FixedArray[Byte],
) -> Unit {
  mb_sha256_x4_ffi(m0, m1, m2, m3, m0.length(), out)
}

///|
fn imp_sha1_x4_eq(
  m0 : Bytes,
  m1 : Bytes,
  m2 : Bytes,
  m3 : Bytes,
  out : FixedArray[Byte],
) -> Unit {
  mb_sha1_x4_ffi(m0, m1, m2, m3, m0.length(), out)
}
