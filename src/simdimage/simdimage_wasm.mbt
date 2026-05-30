// wasm SIMD workers for @simdimage.
//
// `Bytes` (input) and `FixedArray[Byte]` (output) cross the inline-WAT FFI
// as linear-memory pointers to element[0] on the `wasm` target, so these
// `v128.*` bodies (ported verbatim from the `@simd_buffer` `buf_*` kernels,
// only the MoonBit param types changed) read/write them directly. `#borrow`
// keeps the GC from moving them across the call.
//
// Over-read note: `rgb_to_rgba` and `channel_merge` issue `v128.load`s that
// read a few bytes (<=4 / <=12) past the logical end of the last full group.
// Those lanes are discarded by the following `i8x16.shuffle`, so the result
// is unaffected; the reads land in the MoonBit heap after the object and do
// not trap in practice (same pattern `@simdcore`'s String/Bytes SIMD uses).

///|
/// RGB -> RGBA, constant alpha. 4 pixels (12 src bytes) -> 16 out bytes per
/// iter via one `i8x16.shuffle` (the alpha byte comes from an `i8x16.splat`
/// in the shuffle's second operand); scalar tail for `npix % 4`.
#borrow(src, out)
fn imp_rgb_to_rgba(
  src : Bytes,
  out : FixedArray[Byte],
  npix : Int,
  alpha : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local v128) (local i32) (local i32) local.get 3 i8x16.splat local.set 4 local.get 2 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 6 block loop local.get 5 local.get 6 i32.ge_s br_if 1 local.get 1 local.get 5 i32.const 2 i32.shl i32.add local.get 0 local.get 5 i32.const 1 i32.shl local.get 5 i32.add i32.add v128.load local.get 4 i8x16.shuffle 0 1 2 16 3 4 5 16 6 7 8 16 9 10 11 16 v128.store local.get 5 i32.const 4 i32.add local.set 5 br 0 end end block loop local.get 5 local.get 2 i32.ge_s br_if 1 local.get 1 local.get 5 i32.const 2 i32.shl i32.add local.get 0 local.get 5 i32.const 1 i32.shl local.get 5 i32.add i32.add i32.load8_u i32.store8 local.get 1 local.get 5 i32.const 2 i32.shl i32.add i32.const 1 i32.add local.get 0 local.get 5 i32.const 1 i32.shl local.get 5 i32.add i32.add i32.const 1 i32.add i32.load8_u i32.store8 local.get 1 local.get 5 i32.const 2 i32.shl i32.add i32.const 2 i32.add local.get 0 local.get 5 i32.const 1 i32.shl local.get 5 i32.add i32.add i32.const 2 i32.add i32.load8_u i32.store8 local.get 1 local.get 5 i32.const 2 i32.shl i32.add i32.const 3 i32.add local.get 3 i32.store8 local.get 5 i32.const 1 i32.add local.set 5 br 0 end end)

///|
/// RGBA -> grayscale `Y = (77*R + 150*G + 29*B) >> 8` (Rec. 601 fixed-point).
/// 4 pixels per iter: three gather `i8x16.shuffle`s pull R/G/B into the low
/// lanes, each weighted-widened (`i16x8.extmul_low_i8x16_u`), summed, `>> 8`,
/// low-byte gather + `i32.store`; scalar tail.
#borrow(src, out)
fn imp_rgba_to_grayscale(src : Bytes, out : FixedArray[Byte], n : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (local i32) (local i32) (local v128) (local v128) (local v128) (local v128) (local v128) i32.const 77 i8x16.splat local.set 5 i32.const 150 i8x16.splat local.set 6 i32.const 29 i8x16.splat local.set 7 local.get 2 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 0 local.get 3 i32.const 2 i32.shl i32.add v128.load local.set 8 local.get 8 local.get 8 i8x16.shuffle 0 4 8 12 0 0 0 0 0 0 0 0 0 0 0 0 local.get 5 i16x8.extmul_low_i8x16_u local.get 8 local.get 8 i8x16.shuffle 1 5 9 13 0 0 0 0 0 0 0 0 0 0 0 0 local.get 6 i16x8.extmul_low_i8x16_u i16x8.add local.get 8 local.get 8 i8x16.shuffle 2 6 10 14 0 0 0 0 0 0 0 0 0 0 0 0 local.get 7 i16x8.extmul_low_i8x16_u i16x8.add i32.const 8 i16x8.shr_u local.set 9 local.get 1 local.get 3 i32.add local.get 9 local.get 9 i8x16.shuffle 0 2 4 6 0 0 0 0 0 0 0 0 0 0 0 0 i32x4.extract_lane 0 i32.store local.get 3 i32.const 4 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 1 local.get 3 i32.add local.get 0 local.get 3 i32.const 2 i32.shl i32.add i32.load8_u i32.const 77 i32.mul local.get 0 local.get 3 i32.const 2 i32.shl i32.add i32.const 1 i32.add i32.load8_u i32.const 150 i32.mul i32.add local.get 0 local.get 3 i32.const 2 i32.shl i32.add i32.const 2 i32.add i32.load8_u i32.const 29 i32.mul i32.add i32.const 8 i32.shr_u i32.store8 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end)

///|
#borrow(src, out)
fn imp_channel_extract_0(src : Bytes, out : FixedArray[Byte], n : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (local i32) (local i32) (local v128) local.get 2 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 1 local.get 3 i32.add local.get 0 local.get 3 i32.const 2 i32.shl i32.add v128.load local.set 5 local.get 5 local.get 5 i8x16.shuffle 0 4 8 12 0 0 0 0 0 0 0 0 0 0 0 0 i32x4.extract_lane 0 i32.store local.get 3 i32.const 4 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 1 local.get 3 i32.add local.get 0 local.get 3 i32.const 2 i32.shl i32.add i32.load8_u i32.store8 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end)

///|
#borrow(src, out)
fn imp_channel_extract_1(src : Bytes, out : FixedArray[Byte], n : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (local i32) (local i32) (local v128) local.get 2 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 1 local.get 3 i32.add local.get 0 local.get 3 i32.const 2 i32.shl i32.add v128.load local.set 5 local.get 5 local.get 5 i8x16.shuffle 1 5 9 13 0 0 0 0 0 0 0 0 0 0 0 0 i32x4.extract_lane 0 i32.store local.get 3 i32.const 4 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 1 local.get 3 i32.add local.get 0 local.get 3 i32.const 2 i32.shl i32.add i32.const 1 i32.add i32.load8_u i32.store8 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end)

///|
#borrow(src, out)
fn imp_channel_extract_2(src : Bytes, out : FixedArray[Byte], n : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (local i32) (local i32) (local v128) local.get 2 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 1 local.get 3 i32.add local.get 0 local.get 3 i32.const 2 i32.shl i32.add v128.load local.set 5 local.get 5 local.get 5 i8x16.shuffle 2 6 10 14 0 0 0 0 0 0 0 0 0 0 0 0 i32x4.extract_lane 0 i32.store local.get 3 i32.const 4 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 1 local.get 3 i32.add local.get 0 local.get 3 i32.const 2 i32.shl i32.add i32.const 2 i32.add i32.load8_u i32.store8 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end)

///|
#borrow(src, out)
fn imp_channel_extract_3(src : Bytes, out : FixedArray[Byte], n : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (local i32) (local i32) (local v128) local.get 2 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 1 local.get 3 i32.add local.get 0 local.get 3 i32.const 2 i32.shl i32.add v128.load local.set 5 local.get 5 local.get 5 i8x16.shuffle 3 7 11 15 0 0 0 0 0 0 0 0 0 0 0 0 i32x4.extract_lane 0 i32.store local.get 3 i32.const 4 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 1 local.get 3 i32.add local.get 0 local.get 3 i32.const 2 i32.shl i32.add i32.const 3 i32.add i32.load8_u i32.store8 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end)

///|
/// Channel interleave (4 planar streams -> RGBA): 4 pixels per iter via two
/// zip `i8x16.shuffle`s (R+G, B+A into 16-bit pairs) then one final shuffle
/// interleaving the pairs; one `v128.store`. Scalar tail.
#borrow(r, g, b, a, out)
fn imp_channel_merge(
  r : Bytes,
  g : Bytes,
  b : Bytes,
  a : Bytes,
  out : FixedArray[Byte],
  n : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) (local v128) (local v128) local.get 5 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 7 block loop local.get 6 local.get 7 i32.ge_s br_if 1 local.get 0 local.get 6 i32.add v128.load local.get 1 local.get 6 i32.add v128.load i8x16.shuffle 0 16 1 17 2 18 3 19 0 16 1 17 2 18 3 19 local.set 8 local.get 2 local.get 6 i32.add v128.load local.get 3 local.get 6 i32.add v128.load i8x16.shuffle 0 16 1 17 2 18 3 19 0 16 1 17 2 18 3 19 local.set 9 local.get 4 local.get 6 i32.const 2 i32.shl i32.add local.get 8 local.get 9 i8x16.shuffle 0 1 16 17 2 3 18 19 4 5 20 21 6 7 22 23 v128.store local.get 6 i32.const 4 i32.add local.set 6 br 0 end end block loop local.get 6 local.get 5 i32.ge_s br_if 1 local.get 4 local.get 6 i32.const 2 i32.shl i32.add local.get 0 local.get 6 i32.add i32.load8_u i32.store8 local.get 4 local.get 6 i32.const 2 i32.shl i32.add i32.const 1 i32.add local.get 1 local.get 6 i32.add i32.load8_u i32.store8 local.get 4 local.get 6 i32.const 2 i32.shl i32.add i32.const 2 i32.add local.get 2 local.get 6 i32.add i32.load8_u i32.store8 local.get 4 local.get 6 i32.const 2 i32.shl i32.add i32.const 3 i32.add local.get 3 local.get 6 i32.add i32.load8_u i32.store8 local.get 6 i32.const 1 i32.add local.set 6 br 0 end end)

///|
/// Byte lerp `out[i] = (a[i]*inv_t + b[i]*t) >> 8` for `inv_t`, `t` in 1..255.
/// 16 bytes per iter: two `i16x8.extmul_*_i8x16_u` weighted widenings per
/// half, `i16x8.add`, `i16x8.shr_u 8`, low-byte gather shuffle; scalar tail.
#borrow(a, b, out)
fn imp_lerp(
  a : Bytes,
  b : Bytes,
  out : FixedArray[Byte],
  len : Int,
  inv_t : Int,
  t : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) local.get 4 i8x16.splat local.set 8 local.get 5 i8x16.splat local.set 9 local.get 3 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 7 block loop local.get 6 local.get 7 i32.ge_s br_if 1 local.get 0 local.get 6 i32.add v128.load local.set 10 local.get 1 local.get 6 i32.add v128.load local.set 11 local.get 10 local.get 8 i16x8.extmul_low_i8x16_u local.get 11 local.get 9 i16x8.extmul_low_i8x16_u i16x8.add i32.const 8 i16x8.shr_u local.set 12 local.get 10 local.get 8 i16x8.extmul_high_i8x16_u local.get 11 local.get 9 i16x8.extmul_high_i8x16_u i16x8.add i32.const 8 i16x8.shr_u local.set 13 local.get 2 local.get 6 i32.add local.get 12 local.get 13 i8x16.shuffle 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 v128.store local.get 6 i32.const 16 i32.add local.set 6 br 0 end end block loop local.get 6 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 6 i32.add local.get 0 local.get 6 i32.add i32.load8_u local.get 4 i32.mul local.get 1 local.get 6 i32.add i32.load8_u local.get 5 i32.mul i32.add i32.const 8 i32.shr_u i32.store8 local.get 6 i32.const 1 i32.add local.set 6 br 0 end end)

///|
/// Premultiplied source-over of a solid color over `dst` (RGBA8, in place),
/// 4 pixels per iter. Per byte: `out = sat(src_chan + round(dst*inv_a/255))`.
/// Processes `nbytes_full` (a multiple of 16) bytes; caller handles the tail.
#borrow(dst)
fn imp_alpha_blend_solid(
  dst : FixedArray[Byte],
  nbytes_full : Int,
  inv_a : Int,
  src_packed : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) local.get 2 i8x16.splat local.set 5 i32.const 128 i16x8.splat local.set 6 local.get 3 i32x4.splat local.set 7 block loop local.get 4 local.get 1 i32.ge_s br_if 1 local.get 0 local.get 4 i32.add v128.load local.set 8 local.get 8 local.get 5 i16x8.extmul_low_i8x16_u local.get 6 i16x8.add local.set 9 local.get 9 local.get 9 i32.const 8 i16x8.shr_u i16x8.add i32.const 8 i16x8.shr_u local.set 9 local.get 8 local.get 5 i16x8.extmul_high_i8x16_u local.get 6 i16x8.add local.set 10 local.get 10 local.get 10 i32.const 8 i16x8.shr_u i16x8.add i32.const 8 i16x8.shr_u local.set 10 local.get 0 local.get 4 i32.add local.get 9 local.get 10 i8x16.shuffle 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 local.get 7 i8x16.add_sat_u v128.store local.get 4 i32.const 16 i32.add local.set 4 br 0 end end)
