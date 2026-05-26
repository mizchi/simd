// Copy helpers between the GC-backed FixedArray world and the linear-memory
// SimdBuffer world.
//
// All implementations are scalar element-by-element loops. They are the
// bridge users cross when they want to combine GC-friendly storage
// (FixedArray) with SIMD-friendly storage (SimdBuffer): build / consume the
// FixedArray with normal MoonBit code, then `copy_from_*` into a SimdBuffer
// to run SIMD, and `copy_to_*` back to a FixedArray when done.

///|
/// Allocate a `SimdBuffer` and fill it from `src`. Equivalent to
/// `SimdBuffer::make(src.length()) |> buf.copy_from_array(src)`.
pub fn SimdBuffer::from_array(src : FixedArray[Int]) -> SimdBuffer {
  let n = src.length()
  let buf = SimdBuffer::make(n)
  buf.copy_from_array(src)
  buf
}

///|
/// Copy `src` into `self`. Lengths must match.
pub fn SimdBuffer::copy_from_array(
  self : SimdBuffer,
  src : FixedArray[Int],
) -> Unit {
  let n = src.length()
  if n != self.len {
    abort("SimdBuffer.copy_from_array: length mismatch")
  }
  for i in 0..<n {
    buf_store_i32(self.addr + i * 4, src[i])
  }
}

///|
/// Allocate a fresh `FixedArray[Int]` and fill it with the buffer's contents.
pub fn SimdBuffer::to_array(self : SimdBuffer) -> FixedArray[Int] {
  let out = FixedArray::make(self.len, 0)
  self.copy_to_array(out)
  out
}

///|
/// Copy the buffer's contents into `dst`. Lengths must match.
pub fn SimdBuffer::copy_to_array(
  self : SimdBuffer,
  dst : FixedArray[Int],
) -> Unit {
  if dst.length() != self.len {
    abort("SimdBuffer.copy_to_array: length mismatch")
  }
  for i in 0..<self.len {
    dst[i] = buf_load_i32(self.addr + i * 4)
  }
}

// ---- SimdBufferF32 ----

///|
pub fn SimdBufferF32::from_array(src : FixedArray[Float]) -> SimdBufferF32 {
  let n = src.length()
  let buf = SimdBufferF32::make(n)
  buf.copy_from_array(src)
  buf
}

///|
pub fn SimdBufferF32::copy_from_array(
  self : SimdBufferF32,
  src : FixedArray[Float],
) -> Unit {
  let n = src.length()
  if n != self.len {
    abort("SimdBufferF32.copy_from_array: length mismatch")
  }
  for i in 0..<n {
    buf_store_f32(self.addr + i * 4, src[i])
  }
}

///|
pub fn SimdBufferF32::to_array(self : SimdBufferF32) -> FixedArray[Float] {
  let out : FixedArray[Float] = FixedArray::make(self.len, (0.0 : Float))
  self.copy_to_array(out)
  out
}

///|
pub fn SimdBufferF32::copy_to_array(
  self : SimdBufferF32,
  dst : FixedArray[Float],
) -> Unit {
  if dst.length() != self.len {
    abort("SimdBufferF32.copy_to_array: length mismatch")
  }
  for i in 0..<self.len {
    dst[i] = buf_load_f32(self.addr + i * 4)
  }
}

// ---- SimdBufferF64 ----

///|
pub fn SimdBufferF64::from_array(src : FixedArray[Double]) -> SimdBufferF64 {
  let n = src.length()
  let buf = SimdBufferF64::make(n)
  buf.copy_from_array(src)
  buf
}

///|
pub fn SimdBufferF64::copy_from_array(
  self : SimdBufferF64,
  src : FixedArray[Double],
) -> Unit {
  let n = src.length()
  if n != self.len {
    abort("SimdBufferF64.copy_from_array: length mismatch")
  }
  for i in 0..<n {
    buf_store_f64(self.addr + i * 8, src[i])
  }
}

///|
pub fn SimdBufferF64::to_array(self : SimdBufferF64) -> FixedArray[Double] {
  let out = FixedArray::make(self.len, 0.0)
  self.copy_to_array(out)
  out
}

///|
pub fn SimdBufferF64::copy_to_array(
  self : SimdBufferF64,
  dst : FixedArray[Double],
) -> Unit {
  if dst.length() != self.len {
    abort("SimdBufferF64.copy_to_array: length mismatch")
  }
  for i in 0..<self.len {
    dst[i] = buf_load_f64(self.addr + i * 8)
  }
}

// ---- SimdBufferBytes ----

///|
pub fn SimdBufferBytes::from_array(src : FixedArray[Byte]) -> SimdBufferBytes {
  let n = src.length()
  let buf = SimdBufferBytes::make(n)
  buf.copy_from_array(src)
  buf
}

///|
pub fn SimdBufferBytes::copy_from_array(
  self : SimdBufferBytes,
  src : FixedArray[Byte],
) -> Unit {
  let n = src.length()
  if n != self.len {
    abort("SimdBufferBytes.copy_from_array: length mismatch")
  }
  for i in 0..<n {
    buf_store_byte(self.addr + i, src[i].to_int())
  }
}

///|
pub fn SimdBufferBytes::to_array(self : SimdBufferBytes) -> FixedArray[Byte] {
  let out = FixedArray::make(self.len, b'\x00')
  self.copy_to_array(out)
  out
}

///|
pub fn SimdBufferBytes::copy_to_array(
  self : SimdBufferBytes,
  dst : FixedArray[Byte],
) -> Unit {
  if dst.length() != self.len {
    abort("SimdBufferBytes.copy_to_array: length mismatch")
  }
  for i in 0..<self.len {
    dst[i] = buf_load_byte(self.addr + i).to_byte()
  }
}
