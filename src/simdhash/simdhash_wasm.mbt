// wasm 4-way multi-buffer SHA-256 (one message per i32x4 lane).
//
// MoonBit transposes the four padded messages into a "v128-per-word" buffer
// `win` — for block `blk`, word `t`, the 16 bytes at `(blk*16+t)*16` hold the
// four messages' big-endian word `t` as four little-endian i32 lanes — so the
// inline-WAT just `v128.load`s each schedule word with no per-lane assembly.
// K and the initial H are passed as `FixedArray[Int]` tables (loaded +
// `i32x4.splat`, identical across lanes). The round function is the standard
// SHA-256 in `i32x4` lanes: `rotr` synthesised as `(x >>u n) | (x << 32-n)`,
// `ch` via `v128.andnot`, everything else `i32x4.add` / `v128.{and,xor,or}`.
// Output H is written to `hout` (8 v128); MoonBit reads the per-lane words
// back big-endian into the four 32-byte digests.

///|
let k256_int : FixedArray[Int] = {
  let t = FixedArray::make(64, 0)
  for i in 0..<64 {
    t[i] = k256[i].reinterpret_as_int()
  }
  t
}

///|
let iv_int : FixedArray[Int] = {
  let t = FixedArray::make(8, 0)
  for i in 0..<8 {
    t[i] = h256_iv[i].reinterpret_as_int()
  }
  t
}

///|
/// 4-way multi-buffer SHA-256 compression. See the file header for the
/// `win` / `wscratch` / `hout` layout. `blocks` = number of 64-byte blocks
/// (same for all four messages).
#borrow(win, kptr, ivptr, wscratch, hout)
fn sha256_x4_wat(
  win : FixedArray[Byte],
  blocks : Int,
  kptr : FixedArray[Int],
  ivptr : FixedArray[Int],
  wscratch : FixedArray[Byte],
  hout : FixedArray[Byte],
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) (local i32) (local i32) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) local.get 3 i32.load i32x4.splat local.set 10 local.get 3 i32.const 4 i32.add i32.load i32x4.splat local.set 11 local.get 3 i32.const 8 i32.add i32.load i32x4.splat local.set 12 local.get 3 i32.const 12 i32.add i32.load i32x4.splat local.set 13 local.get 3 i32.const 16 i32.add i32.load i32x4.splat local.set 14 local.get 3 i32.const 20 i32.add i32.load i32x4.splat local.set 15 local.get 3 i32.const 24 i32.add i32.load i32x4.splat local.set 16 local.get 3 i32.const 28 i32.add i32.load i32x4.splat local.set 17 i32.const 0 local.set 6 block loop local.get 6 local.get 1 i32.ge_s br_if 1 local.get 6 i32.const 8 i32.shl local.set 9 i32.const 0 local.set 7 block loop local.get 7 i32.const 16 i32.ge_s br_if 1 local.get 4 local.get 7 i32.const 4 i32.shl i32.add local.get 0 local.get 9 i32.add local.get 7 i32.const 4 i32.shl i32.add v128.load v128.store local.get 7 i32.const 1 i32.add local.set 7 br 0 end end i32.const 16 local.set 7 block loop local.get 7 i32.const 64 i32.ge_s br_if 1 local.get 4 local.get 7 i32.const 4 i32.shl i32.add local.set 8 local.get 8 i32.const 240 i32.sub v128.load local.set 30 local.get 30 i32.const 7 i32x4.shr_u local.get 30 i32.const 25 i32x4.shl v128.or local.get 30 i32.const 18 i32x4.shr_u local.get 30 i32.const 14 i32x4.shl v128.or v128.xor local.get 30 i32.const 3 i32x4.shr_u v128.xor local.set 26 local.get 8 i32.const 32 i32.sub v128.load local.set 31 local.get 31 i32.const 17 i32x4.shr_u local.get 31 i32.const 15 i32x4.shl v128.or local.get 31 i32.const 19 i32x4.shr_u local.get 31 i32.const 13 i32x4.shl v128.or v128.xor local.get 31 i32.const 10 i32x4.shr_u v128.xor local.set 27 local.get 8 local.get 8 i32.const 256 i32.sub v128.load local.get 26 i32x4.add local.get 8 i32.const 112 i32.sub v128.load i32x4.add local.get 27 i32x4.add v128.store local.get 7 i32.const 1 i32.add local.set 7 br 0 end end local.get 10 local.set 18 local.get 11 local.set 19 local.get 12 local.set 20 local.get 13 local.set 21 local.get 14 local.set 22 local.get 15 local.set 23 local.get 16 local.set 24 local.get 17 local.set 25 i32.const 0 local.set 7 block loop local.get 7 i32.const 64 i32.ge_s br_if 1 local.get 22 i32.const 6 i32x4.shr_u local.get 22 i32.const 26 i32x4.shl v128.or local.get 22 i32.const 11 i32x4.shr_u local.get 22 i32.const 21 i32x4.shl v128.or v128.xor local.get 22 i32.const 25 i32x4.shr_u local.get 22 i32.const 7 i32x4.shl v128.or v128.xor local.set 26 local.get 22 local.get 23 v128.and local.get 24 local.get 22 v128.andnot v128.xor local.set 27 local.get 25 local.get 26 i32x4.add local.get 27 i32x4.add local.get 2 local.get 7 i32.const 2 i32.shl i32.add i32.load i32x4.splat i32x4.add local.get 4 local.get 7 i32.const 4 i32.shl i32.add v128.load i32x4.add local.set 28 local.get 18 i32.const 2 i32x4.shr_u local.get 18 i32.const 30 i32x4.shl v128.or local.get 18 i32.const 13 i32x4.shr_u local.get 18 i32.const 19 i32x4.shl v128.or v128.xor local.get 18 i32.const 22 i32x4.shr_u local.get 18 i32.const 10 i32x4.shl v128.or v128.xor local.set 26 local.get 18 local.get 19 v128.and local.get 18 local.get 20 v128.and v128.xor local.get 19 local.get 20 v128.and v128.xor local.set 27 local.get 26 local.get 27 i32x4.add local.set 29 local.get 24 local.set 25 local.get 23 local.set 24 local.get 22 local.set 23 local.get 21 local.get 28 i32x4.add local.set 22 local.get 20 local.set 21 local.get 19 local.set 20 local.get 18 local.set 19 local.get 28 local.get 29 i32x4.add local.set 18 local.get 7 i32.const 1 i32.add local.set 7 br 0 end end local.get 10 local.get 18 i32x4.add local.set 10 local.get 11 local.get 19 i32x4.add local.set 11 local.get 12 local.get 20 i32x4.add local.set 12 local.get 13 local.get 21 i32x4.add local.set 13 local.get 14 local.get 22 i32x4.add local.set 14 local.get 15 local.get 23 i32x4.add local.set 15 local.get 16 local.get 24 i32x4.add local.set 16 local.get 17 local.get 25 i32x4.add local.set 17 local.get 6 i32.const 1 i32.add local.set 6 br 0 end end local.get 5 local.get 10 v128.store local.get 5 i32.const 16 i32.add local.get 11 v128.store local.get 5 i32.const 32 i32.add local.get 12 v128.store local.get 5 i32.const 48 i32.add local.get 13 v128.store local.get 5 i32.const 64 i32.add local.get 14 v128.store local.get 5 i32.const 80 i32.add local.get 15 v128.store local.get 5 i32.const 96 i32.add local.get 16 v128.store local.get 5 i32.const 112 i32.add local.get 17 v128.store)

///|
fn imp_sha256_x4_eq(
  m0 : Bytes,
  m1 : Bytes,
  m2 : Bytes,
  m3 : Bytes,
  out : FixedArray[Byte],
) -> Unit {
  let p0 = pad_message(m0)
  let p1 = pad_message(m1)
  let p2 = pad_message(m2)
  let p3 = pad_message(m3)
  let ps = [p0, p1, p2, p3]
  let blocks = p0.length() / 64
  // transpose: win[(blk*16 + t)*16 + k*4 ..] = LE bytes of message k's BE
  // word t (so v128.load lane k == that word).
  let win = FixedArray::make(blocks * 16 * 16, b'\x00')
  for k in 0..<4 {
    let pk = ps[k]
    for blk in 0..<blocks {
      for t in 0..<16 {
        let word = load_be_u32(pk, blk * 64 + t * 4)
        let off = (blk * 16 + t) * 16 + k * 4
        win[off] = (word & 0xFF).reinterpret_as_int().to_byte()
        win[off + 1] = ((word >> 8) & 0xFF).reinterpret_as_int().to_byte()
        win[off + 2] = ((word >> 16) & 0xFF).reinterpret_as_int().to_byte()
        win[off + 3] = ((word >> 24) & 0xFF).reinterpret_as_int().to_byte()
      }
    }
  }
  let wscratch = FixedArray::make(64 * 16, b'\x00')
  let hout = FixedArray::make(8 * 16, b'\x00')
  sha256_x4_wat(win, blocks, k256_int, iv_int, wscratch, hout)
  // read H back: lane k of Hi is the LE i32 at hout[i*16 + k*4]; emit it
  // big-endian into digest k.
  for k in 0..<4 {
    for i in 0..<8 {
      let base = i * 16 + k * 4
      let word = hout[base].to_uint() |
        (hout[base + 1].to_uint() << 8) |
        (hout[base + 2].to_uint() << 16) |
        (hout[base + 3].to_uint() << 24)
      let o = k * 32 + i * 4
      out[o] = ((word >> 24) & 0xFF).reinterpret_as_int().to_byte()
      out[o + 1] = ((word >> 16) & 0xFF).reinterpret_as_int().to_byte()
      out[o + 2] = ((word >> 8) & 0xFF).reinterpret_as_int().to_byte()
      out[o + 3] = (word & 0xFF).reinterpret_as_int().to_byte()
    }
  }
}

///|
/// SHA-1 has no wasm multi-buffer kernel (yet) — four scalar digests.
fn imp_sha1_x4_eq(
  m0 : Bytes,
  m1 : Bytes,
  m2 : Bytes,
  m3 : Bytes,
  out : FixedArray[Byte],
) -> Unit {
  sha1_into(m0, out, 0)
  sha1_into(m1, out, 20)
  sha1_into(m2, out, 40)
  sha1_into(m3, out, 60)
}
