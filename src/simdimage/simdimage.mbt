// @simdimage — SIMD-accelerated image / pixel byte ops.
//
// A focused companion to the numeric `@simd_buffer` family: the
// pixel-oriented kernels (RGB<->RGBA, grayscale, channel de/interleave,
// lerp, solid alpha-blend, histogram) live here instead of being bolted
// onto `SimdBufferBytes`, so the buffer type stays about general byte /
// numeric storage and image code has its own home.
//
// Unlike `@simd_buffer` these operate on core types directly — `Bytes`
// for read-only inputs and `FixedArray[Byte]` for mutable outputs — the
// same zero-copy-on-wasm philosophy as `@simdcore`. No `SimdBufferBytes`
// hop, so an image library holding `Bytes` / `FixedArray[Byte]` can call
// straight in.
//
// Target behaviour (same result on all four, throughput differs):
//   - `wasm`            : inline-WAT `v128` SIMD (`Bytes` / `FixedArray`
//                         cross the FFI as linear-memory pointers).
//   - `wasm-gc` / `js`  : scalar (GC-ref FFI blocks `v128.load`).
//   - `native`          : scalar (FixedArray loops; no C kernel yet).
// The SIMD bodies live in `simdimage_wasm.mbt`; the scalar workers in
// `simdimage_fallback.mbt`. This file holds the shared, target-agnostic
// public API: argument validation, the lerp copy edge cases, the
// alpha-blend scalar tail, and the (always-scalar) histogram.

///|
/// Expand a 3-byte-per-pixel RGB buffer into a 4-byte-per-pixel RGBA buffer
/// with constant `alpha`. `src.length() == n * 3` and `out.length() == n * 4`
/// where `n` is the pixel count. Use `0xFF` for fully opaque.
pub fn rgb_to_rgba(src : Bytes, alpha : Byte, out : FixedArray[Byte]) -> Unit {
  if src.length() % 3 != 0 {
    abort("@simdimage.rgb_to_rgba: src must be a multiple of 3 bytes")
  }
  let n = src.length() / 3
  if out.length() != n * 4 {
    abort("@simdimage.rgb_to_rgba: out length must be src/3 * 4")
  }
  if n == 0 {
    return
  }
  imp_rgb_to_rgba(src, out, n, alpha.to_int())
}

///|
/// Convert RGBA -> grayscale Y via fixed-point weighted sum:
/// `Y = (77 * R + 150 * G + 29 * B) >> 8`, an integer approximation of
/// `0.299*R + 0.587*G + 0.114*B` (Rec. 601). Alpha is dropped.
/// `src.length() == n * 4`, `out.length() == n` for pixel count `n`.
pub fn rgba_to_grayscale(src : Bytes, out : FixedArray[Byte]) -> Unit {
  if src.length() % 4 != 0 {
    abort("@simdimage.rgba_to_grayscale: src must be a multiple of 4 bytes")
  }
  let n = src.length() / 4
  if out.length() != n {
    abort("@simdimage.rgba_to_grayscale: out length must be src/4")
  }
  if n == 0 {
    return
  }
  imp_rgba_to_grayscale(src, out, n)
}

///|
/// Extract one channel from an interleaved RGBA buffer. `ch in 0..3`.
/// `src.length() == n * 4`, `out.length() == n`.
pub fn channel_extract(src : Bytes, ch : Int, out : FixedArray[Byte]) -> Unit {
  if ch < 0 || ch >= 4 {
    abort("@simdimage.channel_extract: ch must be 0..3")
  }
  if src.length() % 4 != 0 {
    abort("@simdimage.channel_extract: src must be a multiple of 4 bytes")
  }
  let n = src.length() / 4
  if out.length() != n {
    abort("@simdimage.channel_extract: out length must be src/4")
  }
  if n == 0 {
    return
  }
  match ch {
    0 => imp_channel_extract_0(src, out, n)
    1 => imp_channel_extract_1(src, out, n)
    2 => imp_channel_extract_2(src, out, n)
    _ => imp_channel_extract_3(src, out, n)
  }
}

///|
/// Interleave 4 planar byte streams into an RGBA buffer. Each input has the
/// same pixel count `n`; output is `n * 4` bytes.
pub fn channel_merge(
  r : Bytes,
  g : Bytes,
  b : Bytes,
  a : Bytes,
  out : FixedArray[Byte],
) -> Unit {
  let n = r.length()
  if g.length() != n || b.length() != n || a.length() != n {
    abort("@simdimage.channel_merge: input lengths must match")
  }
  if out.length() != n * 4 {
    abort("@simdimage.channel_merge: out length must be n * 4")
  }
  if n == 0 {
    return
  }
  imp_channel_merge(r, g, b, a, out, n)
}

///|
/// Byte-wise linear interpolation: `out[i] = (a[i] * (256 - t) + b[i] * t) >> 8`
/// where `t` is the integer interpolation factor in `0..=256` (0 -> a only,
/// 256 -> b only). All three buffers must have the same length.
pub fn lerp(a : Bytes, b : Bytes, t : Int, out : FixedArray[Byte]) -> Unit {
  let n = a.length()
  if b.length() != n || out.length() != n {
    abort("@simdimage.lerp: length mismatch")
  }
  if n == 0 {
    return
  }
  let t_clamped = if t < 0 { 0 } else if t > 256 { 256 } else { t }
  // t=0 -> out=a, t=256 -> out=b; both weights would overflow u8, so copy.
  if t_clamped == 0 {
    for i in 0..<n {
      out[i] = a[i]
    }
  } else if t_clamped == 256 {
    for i in 0..<n {
      out[i] = b[i]
    }
  } else {
    imp_lerp(a, b, out, n, 256 - t_clamped, t_clamped)
  }
}

///|
/// Premultiplied source-over alpha blend with a single solid source color,
/// in-place over an RGBA8 destination buffer. Each `src_*` is `0..=255`.
/// For non-premultiplied input the caller should pre-multiply `src_r/g/b`
/// by `src_a/255`.
///
/// Formula (per channel + per pixel): `out = src + dst * (255 - src_a) / 255`,
/// saturating at 255 (Porter-Duff source-over for premultiplied alpha).
/// `/255` is round-to-nearest (`(t + (t >> 8)) >> 8`, `t = x + 128`), the
/// standard blend approximation — differs from an exact floor by at most 1 LSB.
pub fn alpha_blend_solid(
  dst : FixedArray[Byte],
  src_r : Byte,
  src_g : Byte,
  src_b : Byte,
  src_a : Byte,
) -> Unit {
  if dst.length() % 4 != 0 {
    abort("@simdimage.alpha_blend_solid: dst must be a multiple of 4 bytes")
  }
  let r = src_r.to_int()
  let g = src_g.to_int()
  let b = src_b.to_int()
  let a = src_a.to_int()
  let inv_a = 255 - a
  let n = dst.length() / 4
  if n == 0 {
    return
  }
  let src_packed = r | (g << 8) | (b << 16) | (a << 24)
  let full = n / 4
  if full > 0 {
    imp_alpha_blend_solid(dst, full * 16, inv_a, src_packed)
  }
  // scalar tail (same round-div255 formula as the SIMD body)
  let chans = [r, g, b, a]
  for i in (full * 4)..<n {
    for c in 0..<4 {
      let d = dst[i * 4 + c].to_int()
      let t = d * inv_a + 128
      let r = (t + (t >> 8)) >> 8
      let o = chans[c] + r
      dst[i * 4 + c] = (if o > 255 { 255 } else { o }).to_byte()
    }
  }
}

///|
/// 256-bin byte histogram. `bins.length() >= 256`; each `bins[v]` is
/// incremented by the count of bytes equal to `v` in `src`. Scalar on every
/// target (wasm SIMD has no efficient scatter for unbounded bin counts).
pub fn histogram(src : Bytes, bins : FixedArray[Int]) -> Unit {
  if bins.length() < 256 {
    abort("@simdimage.histogram: bins must have at least 256 slots")
  }
  for i in 0..<src.length() {
    let v = src[i].to_int()
    bins[v] = bins[v] + 1
  }
}
