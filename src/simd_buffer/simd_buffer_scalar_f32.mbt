// Scalar / FixedArray impl of SimdBufferF32 for native + js.
// See simd_buffer_f32.mbt for the wasm + wasm-gc version.

// ---- SimdBufferF32 ----
//
// Native scalar inline — @internal's f32 helpers take FixedArray[Byte]
// (the f32_buf encoding), so we'd have to round-trip via byte buffers to
// reuse them. The MoonBit-level scalar is small and auto-vectorises well
// under clang for the workloads where this matters.

///|
pub struct SimdBufferF32 {
  data : FixedArray[Float]
}

///|
pub fn SimdBufferF32::make(len : Int) -> SimdBufferF32 {
  { data: FixedArray::make(len, (0.0 : Float)) }
}

///|
pub fn SimdBufferF32::length(self : SimdBufferF32) -> Int {
  self.data.length()
}

///|
pub fn SimdBufferF32::get(self : SimdBufferF32, i : Int) -> Float {
  self.data[i]
}

///|
pub fn SimdBufferF32::set(self : SimdBufferF32, i : Int, v : Float) -> Unit {
  self.data[i] = v
}

///|
pub fn SimdBufferF32::sum(self : SimdBufferF32) -> Float {
  let n = self.data.length()
  let mut acc : Float = 0.0
  for i in 0..<n {
    acc = acc + self.data[i]
  }
  acc
}

///|
pub fn SimdBufferF32::dot(a : SimdBufferF32, b : SimdBufferF32) -> Float {
  let n = a.data.length()
  let mut acc : Float = 0.0
  for i in 0..<n {
    acc = acc + a.data[i] * b.data[i]
  }
  acc
}

///|
pub fn SimdBufferF32::add(
  a : SimdBufferF32,
  b : SimdBufferF32,
  out : SimdBufferF32,
) -> Unit {
  let n = a.data.length()
  for i in 0..<n {
    out.data[i] = a.data[i] + b.data[i]
  }
}

///|
pub fn SimdBufferF32::sub(
  a : SimdBufferF32,
  b : SimdBufferF32,
  out : SimdBufferF32,
) -> Unit {
  let n = a.data.length()
  for i in 0..<n {
    out.data[i] = a.data[i] - b.data[i]
  }
}

///|
pub fn SimdBufferF32::mul(
  a : SimdBufferF32,
  b : SimdBufferF32,
  out : SimdBufferF32,
) -> Unit {
  let n = a.data.length()
  for i in 0..<n {
    out.data[i] = a.data[i] * b.data[i]
  }
}

///|
pub fn SimdBufferF32::div(
  a : SimdBufferF32,
  b : SimdBufferF32,
  out : SimdBufferF32,
) -> Unit {
  let n = a.data.length()
  for i in 0..<n {
    out.data[i] = a.data[i] / b.data[i]
  }
}

///|
pub fn SimdBufferF32::sqrt(a : SimdBufferF32, out : SimdBufferF32) -> Unit {
  let n = a.data.length()
  for i in 0..<n {
    out.data[i] = a.data[i].sqrt()
  }
}

///|
pub fn SimdBufferF32::min_elem(
  a : SimdBufferF32,
  b : SimdBufferF32,
  out : SimdBufferF32,
) -> Unit {
  let n = a.data.length()
  for i in 0..<n {
    out.data[i] = if a.data[i] < b.data[i] { a.data[i] } else { b.data[i] }
  }
}

///|
pub fn SimdBufferF32::max_elem(
  a : SimdBufferF32,
  b : SimdBufferF32,
  out : SimdBufferF32,
) -> Unit {
  let n = a.data.length()
  for i in 0..<n {
    out.data[i] = if a.data[i] > b.data[i] { a.data[i] } else { b.data[i] }
  }
}

///|
pub fn SimdBufferF32::from_array(src : FixedArray[Float]) -> SimdBufferF32 {
  let n = src.length()
  let data = FixedArray::make(n, (0.0 : Float))
  for i in 0..<n {
    data[i] = src[i]
  }
  { data, }
}

///|
pub fn SimdBufferF32::copy_from_array(
  self : SimdBufferF32,
  src : FixedArray[Float],
) -> Unit {
  let n = src.length()
  if n != self.data.length() {
    abort("SimdBufferF32.copy_from_array: length mismatch")
  }
  for i in 0..<n {
    self.data[i] = src[i]
  }
}

///|
pub fn SimdBufferF32::to_array(self : SimdBufferF32) -> FixedArray[Float] {
  let n = self.data.length()
  let out : FixedArray[Float] = FixedArray::make(n, (0.0 : Float))
  for i in 0..<n {
    out[i] = self.data[i]
  }
  out
}

///|
pub fn SimdBufferF32::copy_to_array(
  self : SimdBufferF32,
  dst : FixedArray[Float],
) -> Unit {
  let n = self.data.length()
  if dst.length() != n {
    abort("SimdBufferF32.copy_to_array: length mismatch")
  }
  for i in 0..<n {
    dst[i] = self.data[i]
  }
}
