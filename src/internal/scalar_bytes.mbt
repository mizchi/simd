// Scalar byte-buffer reference impls: popcount, memcpy/set, equality, find /
// count, UTF-8 validation, ASCII case conversion, adler32.

///|
/// Total number of 1-bits across a byte buffer (scalar popcount).
pub fn scalar_popcount_bytes(data : FixedArray[Byte]) -> Int {
  let len = data.length()
  let mut acc = 0
  for i in 0..<len {
    let mut v : UInt = data[i].to_int().reinterpret_as_uint()
    while v != 0 {
      acc = acc + (v & 1).reinterpret_as_int()
      v = v >> 1
    }
  }
  acc
}

///|
/// out[di..di+len] = src[si..si+len] (scalar byte copy).
pub fn scalar_memcpy_bytes(
  src : FixedArray[Byte],
  si : Int,
  dst : FixedArray[Byte],
  di : Int,
  len : Int,
) -> Unit {
  for k in 0..<len {
    dst[di + k] = src[si + k]
  }
}

///|
/// dst[off..off+len] = value (scalar byte fill).
pub fn scalar_memset_bytes(
  dst : FixedArray[Byte],
  off : Int,
  len : Int,
  value : Byte,
) -> Unit {
  for k in 0..<len {
    dst[off + k] = value
  }
}

///|
/// True iff a[0..len] == b[0..len] byte-wise (scalar).
pub fn scalar_equal_bytes(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  len : Int,
) -> Bool {
  for k in 0..<len {
    if a[k] != b[k] {
      return false
    }
  }
  true
}

///|
/// Index of first occurrence of `needle` in `data`, or -1 if not found.
pub fn scalar_find_byte(data : FixedArray[Byte], needle : Byte) -> Int {
  let len = data.length()
  for i in 0..<len {
    if data[i] == needle {
      return i
    }
  }
  -1
}

///|
/// Count occurrences of `needle` in `data`.
pub fn scalar_count_byte(data : FixedArray[Byte], needle : Byte) -> Int {
  let len = data.length()
  let mut c = 0
  for i in 0..<len {
    if data[i] == needle {
      c = c + 1
    }
  }
  c
}

///|
/// Structural UTF-8 validation starting at `start`. Verifies that each
/// leading byte is followed by the right number of `10xxxxxx` continuation
/// bytes, and rejects 0x80..0xC1 leading bytes and 0xF5..0xFF entirely.
/// Does NOT reject overlong, surrogate, or > U+10FFFF codepoints — that's
/// a stricter pass than this one.
pub fn scalar_validate_utf8_from(data : FixedArray[Byte], start : Int) -> Bool {
  let n = data.length()
  let mut i = start
  while i < n {
    let b = data[i].to_int()
    if b < 0x80 {
      i = i + 1
    } else if b < 0xC2 {
      return false
    } else if b < 0xE0 {
      if i + 1 >= n {
        return false
      }
      if (data[i + 1].to_int() & 0xC0) != 0x80 {
        return false
      }
      i = i + 2
    } else if b < 0xF0 {
      if i + 2 >= n {
        return false
      }
      if (data[i + 1].to_int() & 0xC0) != 0x80 {
        return false
      }
      if (data[i + 2].to_int() & 0xC0) != 0x80 {
        return false
      }
      i = i + 3
    } else if b < 0xF5 {
      if i + 3 >= n {
        return false
      }
      if (data[i + 1].to_int() & 0xC0) != 0x80 {
        return false
      }
      if (data[i + 2].to_int() & 0xC0) != 0x80 {
        return false
      }
      if (data[i + 3].to_int() & 0xC0) != 0x80 {
        return false
      }
      i = i + 4
    } else {
      return false
    }
  }
  true
}

///|
pub fn scalar_validate_utf8(data : FixedArray[Byte]) -> Bool {
  scalar_validate_utf8_from(data, 0)
}

///|
/// True iff every byte in `data` is < 0x80 (pure ASCII).
pub fn scalar_is_ascii(data : FixedArray[Byte]) -> Bool {
  let len = data.length()
  for i in 0..<len {
    if (data[i].to_int() & 0x80) != 0 {
      return false
    }
  }
  true
}

///|
/// In-place ASCII A-Z → a-z conversion. Non-ASCII bytes pass through.
pub fn scalar_to_lower_ascii(data : FixedArray[Byte]) -> Unit {
  let len = data.length()
  for i in 0..<len {
    let c = data[i].to_int()
    if c >= 0x41 && c <= 0x5A {
      data[i] = (c + 0x20).to_byte()
    }
  }
}

///|
/// In-place ASCII a-z → A-Z conversion. Non-ASCII bytes pass through.
pub fn scalar_to_upper_ascii(data : FixedArray[Byte]) -> Unit {
  let len = data.length()
  for i in 0..<len {
    let c = data[i].to_int()
    if c >= 0x61 && c <= 0x7A {
      data[i] = (c - 0x20).to_byte()
    }
  }
}

///|
/// Adler-32 checksum (scalar).
pub fn scalar_adler32(data : FixedArray[Byte]) -> UInt {
  let mod_adler = 65521U
  let len = data.length()
  for i = 0, a = 1U, b = 0U; i < len; {
    let new_a = (a + data[i].to_int().reinterpret_as_uint()) % mod_adler
    let new_b = (b + new_a) % mod_adler
    continue i + 1, new_a, new_b
  } nobreak {
    (b << 16) | a
  }
}
