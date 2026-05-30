// @simdimage benches. On wasm the `*` call is the SIMD path; the
// local `scalar_*` baselines are the per-byte reference. On other targets
// both are scalar. Run: `moon bench --target wasm -p simdimage`.

///|
fn mk_bytes(n : Int, seed : Int) -> Bytes {
  let a = FixedArray::make(n, b'\x00')
  for i in 0..<n {
    a[i] = ((i * seed + 7) % 256).to_byte()
  }
  Bytes::from_array(a)
}

///|
fn scalar_rgb_to_rgba(
  src : Bytes,
  alpha : Byte,
  out : FixedArray[Byte],
) -> Unit {
  let n = src.length() / 3
  for i in 0..<n {
    out[i * 4] = src[i * 3]
    out[i * 4 + 1] = src[i * 3 + 1]
    out[i * 4 + 2] = src[i * 3 + 2]
    out[i * 4 + 3] = alpha
  }
}

///|
test "bench/rgb_to_rgba" (bb : @bench.T) {
  let npix = 4096
  let rgb = mk_bytes(npix * 3, 13)
  let rgba = FixedArray::make(npix * 4, b'\x00')
  bb.bench(name="scalar_rgb_to_rgba_4096px", fn() {
    scalar_rgb_to_rgba(rgb, b'\xFF', rgba)
    bb.keep(rgba)
  })
  bb.bench(name="simd_rgb_to_rgba_4096px", fn() {
    rgb_to_rgba(rgb, b'\xFF', rgba)
    bb.keep(rgba)
  })
}

///|
fn scalar_channel_extract(
  src : Bytes,
  ch : Int,
  out : FixedArray[Byte],
) -> Unit {
  let n = src.length() / 4
  for i in 0..<n {
    out[i] = src[i * 4 + ch]
  }
}

///|
test "bench/channel_extract" (bb : @bench.T) {
  let npix = 4096
  let rgba = mk_bytes(npix * 4, 29)
  let out = FixedArray::make(npix, b'\x00')
  bb.bench(name="scalar_channel_extract_4096px", fn() {
    scalar_channel_extract(rgba, 1, out)
    bb.keep(out)
  })
  bb.bench(name="simd_channel_extract_4096px", fn() {
    channel_extract(rgba, 1, out)
    bb.keep(out)
  })
}

///|
fn scalar_channel_merge(
  r : Bytes,
  g : Bytes,
  b : Bytes,
  a : Bytes,
  out : FixedArray[Byte],
) -> Unit {
  let n = r.length()
  for i in 0..<n {
    out[i * 4] = r[i]
    out[i * 4 + 1] = g[i]
    out[i * 4 + 2] = b[i]
    out[i * 4 + 3] = a[i]
  }
}

///|
test "bench/channel_merge" (bb : @bench.T) {
  let npix = 4096
  let r = mk_bytes(npix, 11)
  let g = mk_bytes(npix, 23)
  let b = mk_bytes(npix, 37)
  let a = mk_bytes(npix, 53)
  let out = FixedArray::make(npix * 4, b'\x00')
  bb.bench(name="scalar_channel_merge_4096px", fn() {
    scalar_channel_merge(r, g, b, a, out)
    bb.keep(out)
  })
  bb.bench(name="simd_channel_merge_4096px", fn() {
    channel_merge(r, g, b, a, out)
    bb.keep(out)
  })
}

///|
fn scalar_lerp(a : Bytes, b : Bytes, t : Int, out : FixedArray[Byte]) -> Unit {
  let inv = 256 - t
  for i in 0..<a.length() {
    out[i] = ((a[i].to_int() * inv + b[i].to_int() * t) >> 8).to_byte()
  }
}

///|
test "bench/lerp" (bb : @bench.T) {
  let n = 4096
  let a = mk_bytes(n, 11)
  let b = mk_bytes(n, 41)
  let out = FixedArray::make(n, b'\x00')
  bb.bench(name="scalar_lerp_4096", fn() {
    scalar_lerp(a, b, 96, out)
    bb.keep(out)
  })
  bb.bench(name="simd_lerp_4096", fn() {
    lerp(a, b, 96, out)
    bb.keep(out)
  })
}

///|
fn scalar_grayscale(src : Bytes, out : FixedArray[Byte]) -> Unit {
  let n = src.length() / 4
  for i in 0..<n {
    let r = src[i * 4].to_int()
    let g = src[i * 4 + 1].to_int()
    let b = src[i * 4 + 2].to_int()
    out[i] = ((77 * r + 150 * g + 29 * b) >> 8).to_byte()
  }
}

///|
test "bench/rgba_to_grayscale" (bb : @bench.T) {
  let npix = 4096
  let rgba = mk_bytes(npix * 4, 19)
  let out = FixedArray::make(npix, b'\x00')
  bb.bench(name="scalar_rgba_to_grayscale_4096px", fn() {
    scalar_grayscale(rgba, out)
    bb.keep(out)
  })
  bb.bench(name="simd_rgba_to_grayscale_4096px", fn() {
    rgba_to_grayscale(rgba, out)
    bb.keep(out)
  })
}

///|
fn scalar_alpha_blend(
  dst : FixedArray[Byte],
  sr : Int,
  sg : Int,
  sb : Int,
  sa : Int,
) -> Unit {
  let inv = 255 - sa
  let chans = [sr, sg, sb, sa]
  let n = dst.length() / 4
  for i in 0..<n {
    for c in 0..<4 {
      let d = dst[i * 4 + c].to_int()
      let t = d * inv + 128
      let r = (t + (t >> 8)) >> 8
      let o = chans[c] + r
      dst[i * 4 + c] = (if o > 255 { 255 } else { o }).to_byte()
    }
  }
}

///|
test "bench/alpha_blend_solid" (bb : @bench.T) {
  let npix = 4096
  let src = mk_bytes(npix * 4, 31)
  let dst = FixedArray::make(npix * 4, b'\x00')
  for i in 0..<(npix * 4) {
    dst[i] = src[i]
  }
  bb.bench(name="scalar_alpha_blend_4096px", fn() {
    scalar_alpha_blend(dst, 200, 100, 50, 128)
    bb.keep(dst)
  })
  bb.bench(name="simd_alpha_blend_4096px", fn() {
    alpha_blend_solid(dst, 200, 100, 50, 128)
    bb.keep(dst)
  })
}
