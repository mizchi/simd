// Scalar `*_x4` workers for wasm-gc / js: four scalar digests (no SIMD escape
// hatch on these targets). Same result as the wasm / native SIMD paths.

///|
fn imp_sha256_x4_eq(
  m0 : Bytes,
  m1 : Bytes,
  m2 : Bytes,
  m3 : Bytes,
  out : FixedArray[Byte],
) -> Unit {
  digest_into(m0, out, 0)
  digest_into(m1, out, 32)
  digest_into(m2, out, 64)
  digest_into(m3, out, 96)
}

///|
fn imp_sha1_x4_eq(
  m0 : Bytes,
  m1 : Bytes,
  m2 : Bytes,
  m3 : Bytes,
  out : FixedArray[Byte],
) -> Unit {
  sha1_into(m0, out, 0)
  sha1_into(m1, out, 20)
  sha1_into(m2, out, 40)
  sha1_into(m3, out, 60)
}
