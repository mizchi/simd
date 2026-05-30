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
