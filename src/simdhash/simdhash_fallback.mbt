// Off-wasm `sha256_x4` worker: just four scalar digests (no SIMD escape hatch
// on wasm-gc / native / js for this). Same result as the wasm SIMD path.

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
