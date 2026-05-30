// Scalar workers for @simdimage — wasm-gc / native / js.
//
// On wasm-gc / js there is no SIMD escape hatch reachable from `Bytes` /
// `FixedArray[Byte]` (they cross the FFI as GC refs, not linear-memory
// pointers). native has no C kernel for these yet. All three share these
// plain scalar loops, which produce byte-identical results to the wasm
// SIMD bodies. Signatures match `simdimage_wasm.mbt` so the shared public
// API in `simdimage.mbt` dispatches to whichever is built for the target.

///|
fn imp_rgb_to_rgba(
  src : Bytes,
  out : FixedArray[Byte],
  npix : Int,
  alpha : Int,
) -> Unit {
  let a = alpha.to_byte()
  for i in 0..<npix {
    out[i * 4] = src[i * 3]
    out[i * 4 + 1] = src[i * 3 + 1]
    out[i * 4 + 2] = src[i * 3 + 2]
    out[i * 4 + 3] = a
  }
}

///|
fn imp_rgba_to_grayscale(src : Bytes, out : FixedArray[Byte], n : Int) -> Unit {
  for i in 0..<n {
    let r = src[i * 4].to_int()
    let g = src[i * 4 + 1].to_int()
    let b = src[i * 4 + 2].to_int()
    out[i] = ((77 * r + 150 * g + 29 * b) >> 8).to_byte()
  }
}

///|
fn imp_channel_extract_0(src : Bytes, out : FixedArray[Byte], n : Int) -> Unit {
  for i in 0..<n {
    out[i] = src[i * 4]
  }
}

///|
fn imp_channel_extract_1(src : Bytes, out : FixedArray[Byte], n : Int) -> Unit {
  for i in 0..<n {
    out[i] = src[i * 4 + 1]
  }
}

///|
fn imp_channel_extract_2(src : Bytes, out : FixedArray[Byte], n : Int) -> Unit {
  for i in 0..<n {
    out[i] = src[i * 4 + 2]
  }
}

///|
fn imp_channel_extract_3(src : Bytes, out : FixedArray[Byte], n : Int) -> Unit {
  for i in 0..<n {
    out[i] = src[i * 4 + 3]
  }
}

///|
fn imp_channel_merge(
  r : Bytes,
  g : Bytes,
  b : Bytes,
  a : Bytes,
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  for i in 0..<n {
    out[i * 4] = r[i]
    out[i * 4 + 1] = g[i]
    out[i * 4 + 2] = b[i]
    out[i * 4 + 3] = a[i]
  }
}

///|
fn imp_lerp(
  a : Bytes,
  b : Bytes,
  out : FixedArray[Byte],
  len : Int,
  inv_t : Int,
  t : Int,
) -> Unit {
  for i in 0..<len {
    let av = a[i].to_int()
    let bv = b[i].to_int()
    out[i] = ((av * inv_t + bv * t) >> 8).to_byte()
  }
}

///|
fn imp_alpha_blend_solid(
  dst : FixedArray[Byte],
  nbytes_full : Int,
  inv_a : Int,
  src_packed : Int,
) -> Unit {
  let chans = [
    src_packed & 0xFF,
    (src_packed >> 8) & 0xFF,
    (src_packed >> 16) & 0xFF,
    (src_packed >> 24) & 0xFF,
  ]
  let npix = nbytes_full / 4
  for i in 0..<npix {
    for c in 0..<4 {
      let d = dst[i * 4 + c].to_int()
      let t = d * inv_a + 128
      let r = (t + (t >> 8)) >> 8
      let o = chans[c] + r
      dst[i * 4 + c] = (if o > 255 { 255 } else { o }).to_byte()
    }
  }
}
