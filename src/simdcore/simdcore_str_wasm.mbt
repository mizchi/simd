// String / Unicode conversion — wasm SIMD fast path.
//
// MoonBit's `String` is UTF-16. Crossing the FFI boundary to/from UTF-8
// `Bytes` is the common bottleneck, and core's `@encoding/utf8.encode`
// runs a per-code-unit scalar loop (~9 µs / 4 KiB ASCII on wasm).
//
// `String` crosses the inline-WAT FFI as a linear-memory pointer to its
// UTF-16 code-unit buffer (unit i at byte offset 2*i) — verified. So the
// all-ASCII case can be vectorised: detect ASCII with i16x8 masking, then
// narrow UTF-16 -> UTF-8 with a single `i8x16.shuffle` (16 units -> 16 bytes
// per iteration). Non-ASCII input falls back to `@encoding/utf8.encode`,
// which also handles surrogate pairs correctly.
//
// Decode (UTF-8 -> String) is NOT accelerated here: there is no public
// zero-copy `FixedArray[UInt16] -> String`, and dispatching to core's
// ascii/utf8 decoders gives no win (measured — they share the same scalar
// loop). See CLAUDE.md.

///|
/// Returns 1 if every UTF-16 code unit of `s` (first `n` units) is < 0x80,
/// else 0. OR-folds `(unit & 0xFF80)` across 8-unit chunks (i16x8) plus a
/// scalar tail.
#borrow(s)
fn utf16_is_ascii_w(s : String, n : Int) -> Int =
  #|(func (param i32) (param i32) (result i32) (local i32) (local i32) (local i32) (local v128) (local v128) i32.const 65408 i16x8.splat local.set 5 i32.const 0 i32x4.splat local.set 6 local.get 1 i32.const 3 i32.shr_s i32.const 3 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 6 local.get 0 local.get 2 i32.const 1 i32.shl i32.add v128.load local.get 5 v128.and v128.or local.set 6 local.get 2 i32.const 8 i32.add local.set 2 br 0 end end block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.const 1 i32.shl i32.add i32.load16_u i32.const 65408 i32.and i32.or local.set 4 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end local.get 6 v128.any_true local.get 4 i32.or i32.eqz)

///|
/// Narrow `n` ASCII UTF-16 code units of `s` into `out[0..n)` as bytes.
/// 16 units -> 16 bytes per iteration via `i8x16.shuffle` picking the low
/// byte of each unit (even byte indices); scalar tail for the remainder.
/// Caller guarantees `s` is all-ASCII over `[0, n)` and `out.length() >= n`.
#borrow(s, out)
fn utf16_narrow_ascii_w(s : String, out : FixedArray[Byte], n : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (local i32) (local i32) local.get 2 i32.const 4 i32.shr_s i32.const 4 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 1 local.get 3 i32.add local.get 0 local.get 3 i32.const 1 i32.shl i32.add v128.load local.get 0 local.get 3 i32.const 1 i32.shl i32.add i32.const 16 i32.add v128.load i8x16.shuffle 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 v128.store local.get 3 i32.const 16 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 1 local.get 3 i32.add local.get 0 local.get 3 i32.const 1 i32.shl i32.add i32.load16_u i32.store8 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end)

///|
/// Whether every code unit of `s` is ASCII (< 0x80). SIMD over the UTF-16
/// buffer.
pub fn is_ascii_string(s : String) -> Bool {
  let n = s.length()
  if n == 0 {
    return true
  }
  utf16_is_ascii_w(s, n) != 0
}

///|
/// Encode `s` to UTF-8 `Bytes`. All-ASCII strings take a SIMD narrowing fast
/// path; anything else delegates to `@encoding/utf8.encode` (which also
/// validates surrogate pairs). Result is identical to `@utf8.encode(s[:])`.
pub fn encode_utf8(s : String) -> Bytes {
  let n = s.length()
  if n == 0 {
    return @utf8.encode(s[:])
  }
  if utf16_is_ascii_w(s, n) != 0 {
    let out = FixedArray::make(n, b'\x00')
    utf16_narrow_ascii_w(s, out, n)
    Bytes::from_fixedarray(out)
  } else {
    @utf8.encode(s[:])
  }
}

///|
/// Encode `s` into a caller-supplied `out` buffer, returning the number of
/// bytes written. The all-ASCII fast path writes directly into `out` with no
/// intermediate allocation. `out.length()` must be >= the encoded length
/// (== `s.length()` for ASCII; use `@utf8` sizing otherwise). For non-ASCII
/// input this copies from a freshly-encoded `Bytes`.
pub fn encode_utf8_into(s : String, out : FixedArray[Byte]) -> Int {
  let n = s.length()
  if n == 0 {
    return 0
  }
  if utf16_is_ascii_w(s, n) != 0 {
    utf16_narrow_ascii_w(s, out, n)
    n
  } else {
    let bytes = @utf8.encode(s[:])
    let len = bytes.length()
    for i in 0..<len {
      out[i] = bytes[i]
    }
    len
  }
}
