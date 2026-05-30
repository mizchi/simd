// SHA-256 known-answer tests (FIPS 180-4 / NIST examples) on all targets.

///|
fn bytes_of_str(s : String) -> Bytes {
  // ASCII string -> Bytes (test inputs are all ASCII).
  let a = FixedArray::make(s.length(), b'\x00')
  for i in 0..<s.length() {
    a[i] = s[i].to_int().to_byte()
  }
  Bytes::from_array(a)
}

///|
fn bytes_repeat(b : Byte, n : Int) -> Bytes {
  Bytes::from_array(FixedArray::make(n, b))
}

///|
test "sha256 canonical NIST vectors" {
  // empty string
  assert_true(
    sha256_hex(bytes_of_str("")) ==
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  )
  // "abc"
  assert_true(
    sha256_hex(bytes_of_str("abc")) ==
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
  )
  // 56-byte message — forces a second padding block (56 + 9 > 64)
  assert_true(
    sha256_hex(
      bytes_of_str("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
    ) ==
    "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
  )
  // 112-byte message (the two-block NIST example) — multi-block
  assert_true(
    sha256_hex(
      bytes_of_str(
        "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
      ),
    ) ==
    "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1",
  )
}

///|
test "sha256 digest is 32 bytes and deterministic across lengths" {
  // run a sweep of lengths spanning the padding boundaries; each digest must
  // be 32 bytes and stable when recomputed.
  for n in [0, 1, 55, 56, 63, 64, 65, 119, 128, 200] {
    let m = bytes_repeat(b'z', n)
    let d = sha256(m)
    assert_true(d.length() == 32)
    assert_true(d == sha256(m))
  }
}

///|
test "sha256_x4 matches four independent sha256" {
  let m0 = bytes_of_str("hello")
  let m1 = bytes_of_str("world")
  let m2 = bytes_repeat(b'x', 100)
  let m3 = bytes_of_str("")
  let (d0, d1, d2, d3) = sha256_x4(m0, m1, m2, m3)
  assert_true(d0 == sha256(m0))
  assert_true(d1 == sha256(m1))
  assert_true(d2 == sha256(m2))
  assert_true(d3 == sha256(m3))
  // equal-length batch (the SIMD-eligible shape on wasm)
  let a = bytes_repeat(b'a', 200)
  let b = bytes_repeat(b'b', 200)
  let c = bytes_repeat(b'c', 200)
  let d = bytes_repeat(b'd', 200)
  let (e0, e1, e2, e3) = sha256_x4(a, b, c, d)
  assert_true(e0 == sha256(a))
  assert_true(e1 == sha256(b))
  assert_true(e2 == sha256(c))
  assert_true(e3 == sha256(d))
}

///|
test "sha256_x4 equal-length sweep (exercises the wasm 4-way SIMD path)" {
  // four *distinct* equal-length messages at sizes spanning the block-count
  // and padding boundaries; every lane must match the scalar digest, proving
  // lane independence and multi-block correctness of the SIMD kernel.
  fn mk(seed : Int, n : Int) -> Bytes {
    let a = FixedArray::make(n, b'\x00')
    for i in 0..<n {
      a[i] = ((i * 31 + seed * 7 + 1) % 256).to_byte()
    }
    Bytes::from_array(a)
  }

  for n in [0, 1, 55, 56, 63, 64, 65, 119, 128, 200, 256] {
    let m0 = mk(1, n)
    let m1 = mk(2, n)
    let m2 = mk(3, n)
    let m3 = mk(4, n)
    let (d0, d1, d2, d3) = sha256_x4(m0, m1, m2, m3)
    assert_true(d0 == sha256(m0))
    assert_true(d1 == sha256(m1))
    assert_true(d2 == sha256(m2))
    assert_true(d3 == sha256(m3))
  }
}

///|
test "sha1 known vectors" {
  assert_true(
    sha1_hex(bytes_of_str("")) == "da39a3ee5e6b4b0d3255bfef95601890afd80709",
  )
  assert_true(
    sha1_hex(bytes_of_str("abc")) == "a9993e364706816aba3e25717850c26c9cd0d89d",
  )
  assert_true(
    sha1_hex(bytes_of_str("The quick brown fox jumps over the lazy dog")) ==
    "2fd4e1c67a2d28fced849ee1bb76e7391b93eb12",
  )
  // 56-byte two-block input
  assert_true(
    sha1_hex(
      bytes_of_str("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
    ) ==
    "84983e441c3bd26ebaae4aa1f95129e5e54670f1",
  )
  let d = sha1(bytes_of_str("abc"))
  assert_true(d.length() == 20)
}

///|
test "md5 known vectors" {
  assert_true(md5_hex(bytes_of_str("")) == "d41d8cd98f00b204e9800998ecf8427e")
  assert_true(
    md5_hex(bytes_of_str("abc")) == "900150983cd24fb0d6963f7d28e17f72",
  )
  assert_true(
    md5_hex(bytes_of_str("The quick brown fox jumps over the lazy dog")) ==
    "9e107d9d372bb6826bd81d3542a419d6",
  )
  // RFC 1321 test: "12345678901234567890123456789012345678901234567890123456789012345678901234567890"
  assert_true(
    md5_hex(
      bytes_of_str(
        "12345678901234567890123456789012345678901234567890123456789012345678901234567890",
      ),
    ) ==
    "57edf4a22be3c955ac49da2e2107b67a",
  )
  let d = md5(bytes_of_str("abc"))
  assert_true(d.length() == 16)
}

///|
test "sha1_x4 / md5_x4 match singles" {
  let m0 = bytes_of_str("alpha")
  let m1 = bytes_of_str("beta")
  let m2 = bytes_repeat(b'q', 130)
  let m3 = bytes_of_str("")
  let (s0, s1, s2, s3) = sha1_x4(m0, m1, m2, m3)
  assert_true(s0 == sha1(m0))
  assert_true(s1 == sha1(m1))
  assert_true(s2 == sha1(m2))
  assert_true(s3 == sha1(m3))
  let (u0, u1, u2, u3) = md5_x4(m0, m1, m2, m3)
  assert_true(u0 == md5(m0))
  assert_true(u1 == md5(m1))
  assert_true(u2 == md5(m2))
  assert_true(u3 == md5(m3))
}

///|
test "sha1_x4 equal-length sweep (native SSE2/NEON 4-way path)" {
  fn mk(seed : Int, n : Int) -> Bytes {
    let a = FixedArray::make(n, b'\x00')
    for i in 0..<n {
      a[i] = ((i * 17 + seed * 13 + 5) % 256).to_byte()
    }
    Bytes::from_array(a)
  }

  for n in [0, 1, 55, 56, 63, 64, 65, 119, 128, 200, 256] {
    let m0 = mk(1, n)
    let m1 = mk(2, n)
    let m2 = mk(3, n)
    let m3 = mk(4, n)
    let (d0, d1, d2, d3) = sha1_x4(m0, m1, m2, m3)
    assert_true(d0 == sha1(m0))
    assert_true(d1 == sha1(m1))
    assert_true(d2 == sha1(m2))
    assert_true(d3 == sha1(m3))
  }
}

///|
test "md5_x4 equal-length sweep (native SSE2/NEON 4-way path)" {
  fn mk(seed : Int, n : Int) -> Bytes {
    let a = FixedArray::make(n, b'\x00')
    for i in 0..<n {
      a[i] = ((i * 23 + seed * 11 + 3) % 256).to_byte()
    }
    Bytes::from_array(a)
  }

  for n in [0, 1, 55, 56, 63, 64, 65, 119, 128, 200, 256] {
    let m0 = mk(1, n)
    let m1 = mk(2, n)
    let m2 = mk(3, n)
    let m3 = mk(4, n)
    let (d0, d1, d2, d3) = md5_x4(m0, m1, m2, m3)
    assert_true(d0 == md5(m0))
    assert_true(d1 == md5(m1))
    assert_true(d2 == md5(m2))
    assert_true(d3 == md5(m3))
  }
}

///|
test "sha512 known vectors" {
  assert_true(
    sha512_hex(bytes_of_str("")) ==
    "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e",
  )
  assert_true(
    sha512_hex(bytes_of_str("abc")) ==
    "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
  )
  // 112-byte two-block message (NIST example)
  assert_true(
    sha512_hex(
      bytes_of_str(
        "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
      ),
    ) ==
    "8e959b75dae313da8cf4f72814fc143f8f7779c6eb9f7fa17299aeadb6889018501d289e4900f7e4331b99dec4b5433ac7d329eeb6dd26545e96e55b874be909",
  )
  let d = sha512(bytes_of_str("abc"))
  assert_true(d.length() == 64)
}

///|
test "sha512 deterministic across padding boundaries" {
  for n in [0, 1, 111, 112, 127, 128, 129, 200, 256] {
    let m = bytes_repeat(b'q', n)
    let d = sha512(m)
    assert_true(d.length() == 64)
    assert_true(d == sha512(m))
  }
}
