// Scalar / FixedArray impl of SimdBufferBytes for native + js.
// See simd_buffer_bytes.mbt for the wasm + wasm-gc version.

// ---- SimdBufferBytes ----

///|
pub struct SimdBufferBytes {
  data : FixedArray[Byte]
}

///|
pub fn SimdBufferBytes::make(len : Int) -> SimdBufferBytes {
  { data: FixedArray::make(len, b'\x00') }
}

///|
pub fn SimdBufferBytes::length(self : SimdBufferBytes) -> Int {
  self.data.length()
}

///|
/// Backing FixedArray. native + js only.
pub fn SimdBufferBytes::backing(self : SimdBufferBytes) -> FixedArray[Byte] {
  self.data
}

///|
pub fn SimdBufferBytes::get(self : SimdBufferBytes, i : Int) -> Byte {
  self.data[i]
}

///|
pub fn SimdBufferBytes::set(self : SimdBufferBytes, i : Int, v : Byte) -> Unit {
  self.data[i] = v
}

///|
pub fn SimdBufferBytes::popcount(self : SimdBufferBytes) -> Int {
  @internal.scalar_popcount_bytes(self.data)
}

///|
pub fn SimdBufferBytes::memcpy(
  src : SimdBufferBytes,
  si : Int,
  dst : SimdBufferBytes,
  di : Int,
  len : Int,
) -> Unit {
  @internal.scalar_memcpy_bytes(src.data, si, dst.data, di, len)
}

///|
pub fn SimdBufferBytes::memset(
  self : SimdBufferBytes,
  off : Int,
  len : Int,
  value : Byte,
) -> Unit {
  @internal.scalar_memset_bytes(self.data, off, len, value)
}

///|
pub fn SimdBufferBytes::equal(
  a : SimdBufferBytes,
  b : SimdBufferBytes,
  len : Int,
) -> Bool {
  @internal.scalar_equal_bytes(a.data, b.data, len)
}

///|
pub fn SimdBufferBytes::find_byte(self : SimdBufferBytes, needle : Byte) -> Int {
  @internal.scalar_find_byte(self.data, needle)
}

///|
pub fn SimdBufferBytes::count_byte(
  self : SimdBufferBytes,
  needle : Byte,
) -> Int {
  @internal.scalar_count_byte(self.data, needle)
}

///|
pub fn SimdBufferBytes::is_ascii(self : SimdBufferBytes) -> Bool {
  @internal.scalar_is_ascii(self.data)
}

///|
pub fn SimdBufferBytes::to_lower_ascii(self : SimdBufferBytes) -> Unit {
  @internal.scalar_to_lower_ascii(self.data)
}

///|
pub fn SimdBufferBytes::to_upper_ascii(self : SimdBufferBytes) -> Unit {
  @internal.scalar_to_upper_ascii(self.data)
}

///|
pub fn SimdBufferBytes::validate_utf8(self : SimdBufferBytes) -> Bool {
  @internal.scalar_validate_utf8(self.data)
}

///|
pub fn SimdBufferBytes::adler32(self : SimdBufferBytes) -> UInt {
  @internal.scalar_adler32(self.data)
}

///|
pub fn SimdBufferBytes::base64_encode(
  input : SimdBufferBytes,
) -> SimdBufferBytes {
  { data: @base64.encode(input.data) }
}

///|
pub fn SimdBufferBytes::base64_decode(
  input : SimdBufferBytes,
) -> SimdBufferBytes? {
  match @base64.decode(input.data) {
    Some(d) => Some({ data: d })
    None => None
  }
}

///|
/// In-place encode. `out` must have at least `(input.length() + 2) / 3 * 4`
/// bytes; the result is written into `out.data[0..]`.
pub fn SimdBufferBytes::base64_encode_into(
  input : SimdBufferBytes,
  out : SimdBufferBytes,
) -> Unit {
  let encoded = @base64.encode(input.data)
  let n = encoded.length()
  if out.data.length() < n {
    abort("SimdBufferBytes.base64_encode_into: out too small")
  }
  for i in 0..<n {
    out.data[i] = encoded[i]
  }
}

///|
pub fn SimdBufferBytes::base64_decode_into(
  input : SimdBufferBytes,
  out : SimdBufferBytes,
) -> Int? {
  match @base64.decode(input.data) {
    Some(d) => {
      let n = d.length()
      if out.data.length() < n {
        abort("SimdBufferBytes.base64_decode_into: out too small")
      }
      for i in 0..<n {
        out.data[i] = d[i]
      }
      Some(n)
    }
    None => None
  }
}

///|
pub fn SimdBufferBytes::from_array(src : FixedArray[Byte]) -> SimdBufferBytes {
  let n = src.length()
  let data = FixedArray::make(n, b'\x00')
  for i in 0..<n {
    data[i] = src[i]
  }
  { data, }
}

///|
pub fn SimdBufferBytes::copy_from_array(
  self : SimdBufferBytes,
  src : FixedArray[Byte],
) -> Unit {
  let n = src.length()
  if n != self.data.length() {
    abort("SimdBufferBytes.copy_from_array: length mismatch")
  }
  for i in 0..<n {
    self.data[i] = src[i]
  }
}

///|
pub fn SimdBufferBytes::to_array(self : SimdBufferBytes) -> FixedArray[Byte] {
  let n = self.data.length()
  let out = FixedArray::make(n, b'\x00')
  for i in 0..<n {
    out[i] = self.data[i]
  }
  out
}

///|
pub fn SimdBufferBytes::copy_to_array(
  self : SimdBufferBytes,
  dst : FixedArray[Byte],
) -> Unit {
  let n = self.data.length()
  if dst.length() != n {
    abort("SimdBufferBytes.copy_to_array: length mismatch")
  }
  for i in 0..<n {
    dst[i] = self.data[i]
  }
}
