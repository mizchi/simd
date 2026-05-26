// Scalar f32 reference impls over byte-buffer (4 bytes per f32, little-endian).
// FixedArray[Float] cannot cross MoonBit's wasm FFI — see CLAUDE.md.

///|
fn load_f32_le(buf : FixedArray[Byte], i : Int) -> Float {
  let off = i * 4
  let u : UInt = buf[off].to_int().reinterpret_as_uint() |
    (buf[off + 1].to_int().reinterpret_as_uint() << 8) |
    (buf[off + 2].to_int().reinterpret_as_uint() << 16) |
    (buf[off + 3].to_int().reinterpret_as_uint() << 24)
  Float::reinterpret_from_uint(u)
}

///|
fn store_f32_le(buf : FixedArray[Byte], i : Int, v : Float) -> Unit {
  let u = v.reinterpret_as_uint()
  let off = i * 4
  buf[off] = (u & 0xff).reinterpret_as_int().to_byte()
  buf[off + 1] = ((u >> 8) & 0xff).reinterpret_as_int().to_byte()
  buf[off + 2] = ((u >> 16) & 0xff).reinterpret_as_int().to_byte()
  buf[off + 3] = ((u >> 24) & 0xff).reinterpret_as_int().to_byte()
}

///|
pub fn scalar_add_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  for i in 0..<n {
    store_f32_le(out, i, load_f32_le(a, i) + load_f32_le(b, i))
  }
}

///|
pub fn scalar_mul_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  for i in 0..<n {
    store_f32_le(out, i, load_f32_le(a, i) * load_f32_le(b, i))
  }
}

///|
pub fn scalar_sum_f32(a : FixedArray[Byte], n : Int) -> Float {
  let mut acc : Float = (0.0 : Float)
  for i in 0..<n {
    acc = acc + load_f32_le(a, i)
  }
  acc
}

///|
pub fn scalar_sub_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  for i in 0..<n {
    store_f32_le(out, i, load_f32_le(a, i) - load_f32_le(b, i))
  }
}

///|
pub fn scalar_div_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  for i in 0..<n {
    store_f32_le(out, i, load_f32_le(a, i) / load_f32_le(b, i))
  }
}

///|
pub fn scalar_sqrt_f32(
  a : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  for i in 0..<n {
    store_f32_le(out, i, load_f32_le(a, i).sqrt())
  }
}

///|
pub fn scalar_min_elem_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  for i in 0..<n {
    let va = load_f32_le(a, i)
    let vb = load_f32_le(b, i)
    store_f32_le(out, i, if va < vb { va } else { vb })
  }
}

///|
pub fn scalar_max_elem_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  for i in 0..<n {
    let va = load_f32_le(a, i)
    let vb = load_f32_le(b, i)
    store_f32_le(out, i, if va > vb { va } else { vb })
  }
}

///|
pub fn scalar_dot_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  n : Int,
) -> Float {
  let mut acc : Float = (0.0 : Float)
  for i in 0..<n {
    acc = acc + load_f32_le(a, i) * load_f32_le(b, i)
  }
  acc
}
