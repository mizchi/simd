// String / Unicode conversion — scalar fallback for wasm-gc / native / js.
//
// No SIMD here: on wasm-gc `String` is a GC ref (inline-WAT FFI rejected),
// and native / js have no SIMD escape hatch. These delegate to
// `@encoding/utf8`, so the result matches the wasm fast path exactly — only
// throughput differs.

///|
/// Whether every code unit of `s` is ASCII (< 0x80).
pub fn is_ascii_string(s : String) -> Bool {
  for c in s {
    if c.to_int() >= 0x80 {
      return false
    }
  }
  true
}

///|
/// Encode `s` to UTF-8 `Bytes` (scalar; == `@encoding/utf8.encode`).
pub fn encode_utf8(s : String) -> Bytes {
  @utf8.encode(s[:])
}

///|
/// Encode `s` into `out`, returning bytes written (scalar).
pub fn encode_utf8_into(s : String, out : FixedArray[Byte]) -> Int {
  let bytes = @utf8.encode(s[:])
  let len = bytes.length()
  for i in 0..<len {
    out[i] = bytes[i]
  }
  len
}

///|
/// Decode UTF-8 `bytes` to a `String` (scalar; == `@encoding/utf8.decode_lossy`).
/// `_unsafe` mirrors the wasm signature; no SIMD here.
pub fn decode_utf8_unsafe(bytes : Bytes) -> String {
  @utf8.decode_lossy(bytes[:])
}

///|
/// Same as `decode_utf8_unsafe` off-wasm (no intrinsic fast path needed —
/// there is no SIMD widen to accelerate).
pub fn decode_utf8_unsafe_intrinsic(bytes : Bytes) -> String {
  @utf8.decode_lossy(bytes[:])
}
