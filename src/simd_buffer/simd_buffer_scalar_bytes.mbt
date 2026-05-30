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
/// Strict UTF-8 validation (RFC 3629): rejects overlong / surrogate /
/// > U+10FFFF.
pub fn SimdBufferBytes::validate_utf8_strict(self : SimdBufferBytes) -> Bool {
  @internal.scalar_validate_utf8_strict(self.data)
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
// ---- byte binary arithmetic (Tier 1 — 0.3.0) ----

///|
pub fn SimdBufferBytes::byte_add(
  a : SimdBufferBytes,
  b : SimdBufferBytes,
  out : SimdBufferBytes,
) -> Unit {
  if a.data.length() != b.data.length() || a.data.length() != out.data.length() {
    abort("SimdBufferBytes.byte_add: length mismatch")
  }
  for i in 0..<a.data.length() {
    out.data[i] = ((a.data[i].to_int() + b.data[i].to_int()) & 0xFF).to_byte()
  }
}

///|
pub fn SimdBufferBytes::byte_sub(
  a : SimdBufferBytes,
  b : SimdBufferBytes,
  out : SimdBufferBytes,
) -> Unit {
  if a.data.length() != b.data.length() || a.data.length() != out.data.length() {
    abort("SimdBufferBytes.byte_sub: length mismatch")
  }
  for i in 0..<a.data.length() {
    out.data[i] = ((a.data[i].to_int() - b.data[i].to_int()) & 0xFF).to_byte()
  }
}

///|
pub fn SimdBufferBytes::byte_avg(
  a : SimdBufferBytes,
  b : SimdBufferBytes,
  out : SimdBufferBytes,
) -> Unit {
  if a.data.length() != b.data.length() || a.data.length() != out.data.length() {
    abort("SimdBufferBytes.byte_avg: length mismatch")
  }
  for i in 0..<a.data.length() {
    out.data[i] = (((a.data[i].to_int() + b.data[i].to_int() + 1) >> 1) & 0xFF).to_byte()
  }
}

///|
pub fn SimdBufferBytes::sat_add(
  a : SimdBufferBytes,
  b : SimdBufferBytes,
  out : SimdBufferBytes,
) -> Unit {
  if a.data.length() != b.data.length() || a.data.length() != out.data.length() {
    abort("SimdBufferBytes.sat_add: length mismatch")
  }
  for i in 0..<a.data.length() {
    let s = a.data[i].to_int() + b.data[i].to_int()
    out.data[i] = (if s > 255 { 255 } else { s }).to_byte()
  }
}

///|
pub fn SimdBufferBytes::sat_sub(
  a : SimdBufferBytes,
  b : SimdBufferBytes,
  out : SimdBufferBytes,
) -> Unit {
  if a.data.length() != b.data.length() || a.data.length() != out.data.length() {
    abort("SimdBufferBytes.sat_sub: length mismatch")
  }
  for i in 0..<a.data.length() {
    let s = a.data[i].to_int() - b.data[i].to_int()
    out.data[i] = (if s < 0 { 0 } else { s }).to_byte()
  }
}

///|
pub fn SimdBufferBytes::clamp(
  self : SimdBufferBytes,
  lo : Byte,
  hi : Byte,
  out : SimdBufferBytes,
) -> Unit {
  if self.data.length() != out.data.length() {
    abort("SimdBufferBytes.clamp: length mismatch")
  }
  let lo_i = lo.to_int()
  let hi_i = hi.to_int()
  for i in 0..<self.data.length() {
    let v = self.data[i].to_int()
    let clamped = if v < lo_i { lo_i } else if v > hi_i { hi_i } else { v }
    out.data[i] = clamped.to_byte()
  }
}

///|
pub fn SimdBufferBytes::byte_sub_offset(
  src : SimdBufferBytes,
  stride : Int,
  out : SimdBufferBytes,
) -> Unit {
  if src.data.length() != out.data.length() {
    abort("SimdBufferBytes.byte_sub_offset: length mismatch")
  }
  if stride <= 0 {
    abort("SimdBufferBytes.byte_sub_offset: stride must be >= 1")
  }
  let n = src.data.length()
  for i in 0..<n {
    let a = src.data[i].to_int()
    let b = if i >= stride { src.data[i - stride].to_int() } else { 0 }
    out.data[i] = ((a - b) & 0xFF).to_byte()
  }
}

// ---- image / pixel ops (Tier 2 — 0.3.0) ----

///|
pub fn SimdBufferBytes::rgb_to_rgba(
  src : SimdBufferBytes,
  alpha : Byte,
  out : SimdBufferBytes,
) -> Unit {
  let src_len = src.data.length()
  if src_len % 3 != 0 {
    abort("SimdBufferBytes.rgb_to_rgba: src must be a multiple of 3 bytes")
  }
  let n = src_len / 3
  if out.data.length() != n * 4 {
    abort("SimdBufferBytes.rgb_to_rgba: out length must be src/3 * 4")
  }
  for i in 0..<n {
    out.data[i * 4] = src.data[i * 3]
    out.data[i * 4 + 1] = src.data[i * 3 + 1]
    out.data[i * 4 + 2] = src.data[i * 3 + 2]
    out.data[i * 4 + 3] = alpha
  }
}

///|
pub fn SimdBufferBytes::rgba_to_grayscale(
  src : SimdBufferBytes,
  out : SimdBufferBytes,
) -> Unit {
  let src_len = src.data.length()
  if src_len % 4 != 0 {
    abort(
      "SimdBufferBytes.rgba_to_grayscale: src must be a multiple of 4 bytes",
    )
  }
  let n = src_len / 4
  if out.data.length() != n {
    abort("SimdBufferBytes.rgba_to_grayscale: out length must be src/4")
  }
  for i in 0..<n {
    let r = src.data[i * 4].to_int()
    let g = src.data[i * 4 + 1].to_int()
    let b = src.data[i * 4 + 2].to_int()
    out.data[i] = ((77 * r + 150 * g + 29 * b) >> 8).to_byte()
  }
}

///|
pub fn SimdBufferBytes::channel_extract(
  src : SimdBufferBytes,
  ch : Int,
  out : SimdBufferBytes,
) -> Unit {
  if ch < 0 || ch >= 4 {
    abort("SimdBufferBytes.channel_extract: ch must be 0..3")
  }
  let src_len = src.data.length()
  if src_len % 4 != 0 {
    abort("SimdBufferBytes.channel_extract: src must be a multiple of 4 bytes")
  }
  let n = src_len / 4
  if out.data.length() != n {
    abort("SimdBufferBytes.channel_extract: out length must be src/4")
  }
  for i in 0..<n {
    out.data[i] = src.data[i * 4 + ch]
  }
}

///|
pub fn SimdBufferBytes::channel_merge(
  r : SimdBufferBytes,
  g : SimdBufferBytes,
  b : SimdBufferBytes,
  a : SimdBufferBytes,
  out : SimdBufferBytes,
) -> Unit {
  let n = r.data.length()
  if g.data.length() != n || b.data.length() != n || a.data.length() != n {
    abort("SimdBufferBytes.channel_merge: input lengths must match")
  }
  if out.data.length() != n * 4 {
    abort("SimdBufferBytes.channel_merge: out length must be n * 4")
  }
  for i in 0..<n {
    out.data[i * 4] = r.data[i]
    out.data[i * 4 + 1] = g.data[i]
    out.data[i * 4 + 2] = b.data[i]
    out.data[i * 4 + 3] = a.data[i]
  }
}

///|
pub fn SimdBufferBytes::lerp(
  a : SimdBufferBytes,
  b : SimdBufferBytes,
  t : Int,
  out : SimdBufferBytes,
) -> Unit {
  if a.data.length() != b.data.length() || a.data.length() != out.data.length() {
    abort("SimdBufferBytes.lerp: length mismatch")
  }
  let t_clamped = if t < 0 { 0 } else if t > 256 { 256 } else { t }
  let inv_t = 256 - t_clamped
  for i in 0..<a.data.length() {
    let av = a.data[i].to_int()
    let bv = b.data[i].to_int()
    out.data[i] = ((av * inv_t + bv * t_clamped) >> 8).to_byte()
  }
}

///|
pub fn SimdBufferBytes::histogram(
  self : SimdBufferBytes,
  bins : SimdBuffer,
) -> Unit {
  if bins.data.length() < 256 {
    abort("SimdBufferBytes.histogram: bins must have at least 256 slots")
  }
  for i in 0..<self.data.length() {
    let v = self.data[i].to_int()
    bins.data[v] = bins.data[v] + 1
  }
}

///|
pub fn SimdBufferBytes::alpha_blend_solid(
  dst : SimdBufferBytes,
  src_r : Int,
  src_g : Int,
  src_b : Int,
  src_a : Int,
) -> Unit {
  let dst_len = dst.data.length()
  if dst_len % 4 != 0 {
    abort(
      "SimdBufferBytes.alpha_blend_solid: dst must be a multiple of 4 bytes",
    )
  }
  let inv_a = 255 - src_a
  let n = dst_len / 4
  let chans = [src_r, src_g, src_b, src_a]
  // round-to-nearest div-255: (x + 128 + ((x+128) >> 8)) >> 8 (matches the
  // wasm SIMD path).
  for i in 0..<n {
    for c in 0..<4 {
      let d = dst.data[i * 4 + c].to_int()
      let t = d * inv_a + 128
      let r = (t + (t >> 8)) >> 8
      let o = chans[c] + r
      dst.data[i * 4 + c] = (if o > 255 { 255 } else { o }).to_byte()
    }
  }
}
