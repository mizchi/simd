///|
fn bytes(vals : Array[Int]) -> FixedArray[Byte] {
  let arr = FixedArray::make(vals.length(), b'\x00')
  for i, v in vals {
    arr[i] = v.to_byte()
  }
  arr
}

///|
fn bytes_eq(a : FixedArray[Byte], b : FixedArray[Byte]) -> Bool {
  let n = a.length()
  if n != b.length() {
    return false
  }
  for i in 0..<n {
    if a[i] != b[i] {
      return false
    }
  }
  true
}

///|
fn check(actual : FixedArray[Byte], expected : FixedArray[Byte]) -> Unit {
  if !bytes_eq(actual, expected) {
    abort("byte arrays differ")
  }
}

///|
fn check_decode(input : FixedArray[Byte], expected : FixedArray[Byte]) -> Unit {
  match decode(input) {
    Some(b) => check(b, expected)
    None => abort("decode returned None, expected Some")
  }
}

///|
fn check_decode_none(input : FixedArray[Byte]) -> Unit {
  match decode(input) {
    Some(_) => abort("decode returned Some, expected None")
    None => ()
  }
}

///|
/// RFC 4648 §10 test vectors expressed as ASCII byte arrays.
test "encode RFC 4648 vectors" {
  check(encode(bytes([])), bytes([]))
  // "f" -> "Zg=="
  check(encode(bytes([0x66])), bytes([0x5a, 0x67, 0x3d, 0x3d]))
  // "fo" -> "Zm8="
  check(encode(bytes([0x66, 0x6f])), bytes([0x5a, 0x6d, 0x38, 0x3d]))
  // "foo" -> "Zm9v"
  check(encode(bytes([0x66, 0x6f, 0x6f])), bytes([0x5a, 0x6d, 0x39, 0x76]))
  // "foob" -> "Zm9vYg=="
  check(
    encode(bytes([0x66, 0x6f, 0x6f, 0x62])),
    bytes([0x5a, 0x6d, 0x39, 0x76, 0x59, 0x67, 0x3d, 0x3d]),
  )
  // "fooba" -> "Zm9vYmE="
  check(
    encode(bytes([0x66, 0x6f, 0x6f, 0x62, 0x61])),
    bytes([0x5a, 0x6d, 0x39, 0x76, 0x59, 0x6d, 0x45, 0x3d]),
  )
  // "foobar" -> "Zm9vYmFy"
  check(
    encode(bytes([0x66, 0x6f, 0x6f, 0x62, 0x61, 0x72])),
    bytes([0x5a, 0x6d, 0x39, 0x76, 0x59, 0x6d, 0x46, 0x79]),
  )
}

///|
test "decode RFC 4648 vectors" {
  check_decode(bytes([]), bytes([]))
  check_decode(bytes([0x5a, 0x67, 0x3d, 0x3d]), bytes([0x66]))
  check_decode(bytes([0x5a, 0x6d, 0x38, 0x3d]), bytes([0x66, 0x6f]))
  check_decode(bytes([0x5a, 0x6d, 0x39, 0x76]), bytes([0x66, 0x6f, 0x6f]))
  check_decode(
    bytes([0x5a, 0x6d, 0x39, 0x76, 0x59, 0x67, 0x3d, 0x3d]),
    bytes([0x66, 0x6f, 0x6f, 0x62]),
  )
  check_decode(
    bytes([0x5a, 0x6d, 0x39, 0x76, 0x59, 0x6d, 0x45, 0x3d]),
    bytes([0x66, 0x6f, 0x6f, 0x62, 0x61]),
  )
  check_decode(
    bytes([0x5a, 0x6d, 0x39, 0x76, 0x59, 0x6d, 0x46, 0x79]),
    bytes([0x66, 0x6f, 0x6f, 0x62, 0x61, 0x72]),
  )
}

///|
test "roundtrip random sizes" {
  // LCG-ish deterministic pseudo-random byte pattern (UInt to avoid i32 wrap).
  for
    size in [
      1, 2, 3, 4, 5, 7, 8, 12, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 96,
      127, 128,
    ] {
    let input = FixedArray::make(size, b'\x00')
    let mut seed : UInt = size.reinterpret_as_uint() * 2654435761U
    for i in 0..<size {
      seed = seed * 1103515245U + 12345U
      input[i] = ((seed >> 8) & 0xFFU).reinterpret_as_int().to_byte()
    }
    let encoded = encode(input)
    check_decode(encoded, input)
  }
}

///|
test "decode rejects non-multiple-of-4 length" {
  check_decode_none(bytes([0x61, 0x62, 0x63])) // "abc"
  check_decode_none(bytes([0x61])) // "a"
}

///|
test "decode rejects out-of-alphabet bytes" {
  check_decode_none(bytes([0x5a, 0x6d, 0x39, 0x20])) // "Zm9 "
  check_decode_none(bytes([0x5a, 0x6d, 0x39, 0x2a])) // "Zm9*"
}

///|
test "decode handles full alphabet" {
  let alphabet = FixedArray::make(64, b'\x00')
  for i in 0..<26 {
    alphabet[i] = (i + 65).to_byte()
  }
  for i in 0..<26 {
    alphabet[26 + i] = (i + 97).to_byte()
  }
  for i in 0..<10 {
    alphabet[52 + i] = (i + 48).to_byte()
  }
  alphabet[62] = b'+'
  alphabet[63] = b'/'
  match decode(alphabet) {
    Some(b) => {
      if b.length() != 48 {
        abort("expected 48 decoded bytes")
      }
      check(encode(b), alphabet)
    }
    None => abort("decode of full alphabet returned None")
  }
}

///|
/// Regression test: SIMD encode of a 16-byte buffer containing bytes > 127
/// (so the i8x16.ge_s comparisons exercise the boundary where the byte's
/// sign bit would matter under a buggy signed interpretation).
test "encode 16-byte non-ASCII round-trips" {
  let input = bytes([
    0xff, 0xfe, 0xfd, 0xfc, 0xfb, 0xfa, 0xf9, 0xf8, 0xf7, 0xf6, 0xf5, 0xf4, 0xf3,
    0xf2, 0xf1, 0xf0,
  ])
  match decode(encode(input)) {
    Some(round) => check(round, input)
    None => abort("encode/decode round-trip failed")
  }
}
