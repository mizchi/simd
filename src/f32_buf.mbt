// f32 byte-buffer boundary helpers.
//
// FixedArray[Float] is rejected at the FFI boundary, so f32 buffers are
// represented as FixedArray[Byte] with 4 bytes per element (little-endian
// IEEE-754). These helpers cover the boundary; ops in simd_*.mbt just see
// byte buffers.

///|
pub fn f32_buf_alloc(n : Int) -> FixedArray[Byte] {
  FixedArray::make(n * 4, b'\x00')
}

///|
pub fn f32_buf_get(buf : FixedArray[Byte], i : Int) -> Float {
  let off = i * 4
  let u : UInt = buf[off].to_int().reinterpret_as_uint() |
    (buf[off + 1].to_int().reinterpret_as_uint() << 8) |
    (buf[off + 2].to_int().reinterpret_as_uint() << 16) |
    (buf[off + 3].to_int().reinterpret_as_uint() << 24)
  Float::reinterpret_from_uint(u)
}

///|
pub fn f32_buf_set(buf : FixedArray[Byte], i : Int, v : Float) -> Unit {
  let u = v.reinterpret_as_uint()
  let off = i * 4
  buf[off] = (u & 0xff).reinterpret_as_int().to_byte()
  buf[off + 1] = ((u >> 8) & 0xff).reinterpret_as_int().to_byte()
  buf[off + 2] = ((u >> 16) & 0xff).reinterpret_as_int().to_byte()
  buf[off + 3] = ((u >> 24) & 0xff).reinterpret_as_int().to_byte()
}
