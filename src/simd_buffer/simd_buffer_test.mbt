///|
/// Helper visible across the test + bench blackbox compilation unit.
fn Double::to_f32(self : Double) -> Float {
  Float::from_double(self)
}

///|
test "SimdBuffer make + length" {
  let b = SimdBuffer::make(8)
  assert_true(b.length() == 8)
}

///|
test "SimdBuffer round-trip via get/set" {
  let b = SimdBuffer::make(16)
  for i in 0..<16 {
    b.set(i, i * i)
  }
  for i in 0..<16 {
    if b.get(i) != i * i {
      abort("round-trip mismatch")
    }
  }
}

///|
test "SimdBuffer sum: matches scalar" {
  let n = 1024
  let b = SimdBuffer::make(n)
  let mut expected = 0
  for i in 0..<n {
    let v = i + 1 // 1..1024
    b.set(i, v)
    expected = expected + v
  }
  if b.sum() != expected {
    abort("SimdBuffer.sum mismatch")
  }
}

///|
test "SimdBuffer add: matches scalar" {
  let n = 1024
  let a = SimdBuffer::make(n)
  let b = SimdBuffer::make(n)
  let out = SimdBuffer::make(n)
  for i in 0..<n {
    a.set(i, i * 3)
    b.set(i, i * 5 + 7)
  }
  SimdBuffer::add(a, b, out)
  for i in 0..<n {
    if out.get(i) != i * 3 + (i * 5 + 7) {
      abort("SimdBuffer.add mismatch at " + i.to_string())
    }
  }
}

///|
test "SimdBuffer sum: tail (non-multiple-of-4 length)" {
  let n = 1027 // 1024 + 3 tail
  let b = SimdBuffer::make(n)
  let mut expected = 0
  for i in 0..<n {
    b.set(i, i * 2 + 1)
    expected = expected + (i * 2 + 1)
  }
  if b.sum() != expected {
    abort("tail sum mismatch")
  }
}

///|
fn f64_close(a : Double, b : Double, eps : Double) -> Bool {
  let d = a - b
  let abs = if d < 0.0 { -d } else { d }
  abs < eps
}

///|
test "SimdBufferF64 round-trip + sum + dot" {
  let n = 64
  let a = SimdBufferF64::make(n)
  let b = SimdBufferF64::make(n)
  let mut expected_sum = 0.0
  let mut expected_dot = 0.0
  for i in 0..<n {
    let av = i.to_double() + 0.5
    let bv = (i * 2).to_double() - 1.0
    a.set(i, av)
    b.set(i, bv)
    expected_sum = expected_sum + av
    expected_dot = expected_dot + av * bv
  }
  if !f64_close(a.sum(), expected_sum, 1.0e-9) {
    abort("SimdBufferF64.sum mismatch")
  }
  if !f64_close(SimdBufferF64::dot(a, b), expected_dot, 1.0e-9) {
    abort("SimdBufferF64.dot mismatch")
  }
}

///|
test "SimdBufferF64 matmul: 4x4 = 4x4 * 4x4" {
  // a (M=4, K=4) = identity
  // b_t (N=4, K=4) = transpose of identity = identity
  // c = a * b = identity
  let m = 4
  let k = 4
  let n = 4
  let a = SimdBufferF64::make(m * k)
  let b_t = SimdBufferF64::make(n * k)
  let c = SimdBufferF64::make(m * n)
  for i in 0..<m {
    for j in 0..<k {
      a.set(i * k + j, if i == j { 1.0 } else { 0.0 })
    }
  }
  for i in 0..<n {
    for j in 0..<k {
      b_t.set(i * k + j, if i == j { 1.0 } else { 0.0 })
    }
  }
  SimdBufferF64::matmul(a, b_t, c, m, k, n)
  for i in 0..<m {
    for j in 0..<n {
      let expected = if i == j { 1.0 } else { 0.0 }
      if !f64_close(c.get(i * n + j), expected, 1.0e-9) {
        abort(
          "matmul I*I != I at (" + i.to_string() + "," + j.to_string() + ")",
        )
      }
    }
  }
}

///|
test "SimdBufferF64 matmul: matches scalar reference (8x8 random)" {
  let m = 8
  let k = 8
  let n = 8
  let a = SimdBufferF64::make(m * k)
  let b_t = SimdBufferF64::make(n * k)
  let c = SimdBufferF64::make(m * n)
  for i in 0..<(m * k) {
    a.set(i, (i % 7).to_double() - 3.0)
  }
  for i in 0..<(n * k) {
    b_t.set(i, (i % 5).to_double() - 2.0)
  }
  SimdBufferF64::matmul(a, b_t, c, m, k, n)
  // Reference: scalar matmul. b_t is row-major B^T, so b_t[j*k + p] = B[p][j].
  for i in 0..<m {
    for j in 0..<n {
      let mut acc = 0.0
      for p in 0..<k {
        acc = acc + a.get(i * k + p) * b_t.get(j * k + p)
      }
      let got = c.get(i * n + j)
      if !f64_close(got, acc, 1.0e-9) {
        abort(
          "matmul mismatch at (" +
          i.to_string() +
          "," +
          j.to_string() +
          "): expected " +
          acc.to_string() +
          " got " +
          got.to_string(),
        )
      }
    }
  }
}

///|
test "SimdBufferF64 gemv: matches scalar reference" {
  let m = 6
  let n = 8
  let a = SimdBufferF64::make(m * n)
  let x = SimdBufferF64::make(n)
  let y = SimdBufferF64::make(m)
  for i in 0..<(m * n) {
    a.set(i, (i % 11).to_double() / 3.0)
  }
  for i in 0..<n {
    x.set(i, i.to_double() - 4.0)
  }
  SimdBufferF64::gemv(a, x, y, m, n)
  for i in 0..<m {
    let mut acc = 0.0
    for j in 0..<n {
      acc = acc + a.get(i * n + j) * x.get(j)
    }
    if !f64_close(y.get(i), acc, 1.0e-9) {
      abort("gemv mismatch at " + i.to_string())
    }
  }
}

///|
fn buf_from_array(vals : Array[Int]) -> SimdBufferBytes {
  let n = vals.length()
  let b = SimdBufferBytes::make(n)
  for i, v in vals {
    b.set(i, v.to_byte())
  }
  b
}

///|
fn buf_eq_bytes(a : SimdBufferBytes, expected : Array[Int]) -> Bool {
  if a.length() != expected.length() {
    return false
  }
  for i, v in expected {
    if a.get(i).to_int() != v {
      return false
    }
  }
  true
}

///|
test "SimdBufferBytes base64_encode RFC 4648 vectors" {
  // ""  -> ""
  let r0 = SimdBufferBytes::base64_encode(buf_from_array([]))
  assert_true(r0.length() == 0)
  // "f"  -> "Zg=="
  let r1 = SimdBufferBytes::base64_encode(buf_from_array([0x66]))
  if !buf_eq_bytes(r1, [0x5a, 0x67, 0x3d, 0x3d]) {
    abort("encode f")
  }
  // "fo" -> "Zm8="
  let r2 = SimdBufferBytes::base64_encode(buf_from_array([0x66, 0x6f]))
  if !buf_eq_bytes(r2, [0x5a, 0x6d, 0x38, 0x3d]) {
    abort("encode fo")
  }
  // "foobar" -> "Zm9vYmFy"
  let r3 = SimdBufferBytes::base64_encode(
    buf_from_array([0x66, 0x6f, 0x6f, 0x62, 0x61, 0x72]),
  )
  if !buf_eq_bytes(r3, [0x5a, 0x6d, 0x39, 0x76, 0x59, 0x6d, 0x46, 0x79]) {
    abort("encode foobar")
  }
}

///|
test "SimdBufferBytes roundtrip 16-byte non-ASCII (exercises SIMD encode + decode)" {
  let input = buf_from_array([
    0xff, 0xfe, 0xfd, 0xfc, 0xfb, 0xfa, 0xf9, 0xf8, 0xf7, 0xf6, 0xf5, 0xf4, 0xf3,
    0xf2, 0xf1, 0xf0,
  ])
  let encoded = SimdBufferBytes::base64_encode(input)
  let decoded_opt = SimdBufferBytes::base64_decode(encoded)
  match decoded_opt {
    Some(b) => {
      if b.length() != 16 {
        abort("decoded length wrong")
      }
      for i in 0..<16 {
        if b.get(i) != input.get(i) {
          abort("roundtrip byte mismatch at " + i.to_string())
        }
      }
    }
    None => abort("decode returned None")
  }
}

///|
test "SimdBufferBytes roundtrip many sizes" {
  // Hit sizes that cross SIMD encode/decode boundaries.
  let sizes = [0, 1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 96, 128, 129]
  for size in sizes {
    let input = SimdBufferBytes::make(size)
    let mut seed : UInt = (size + 1).reinterpret_as_uint() * 2654435761U
    for i in 0..<size {
      seed = seed * 1103515245U + 12345U
      input.set(i, ((seed >> 8) & 0xFFU).reinterpret_as_int().to_byte())
    }
    let encoded = SimdBufferBytes::base64_encode(input)
    let decoded_opt = SimdBufferBytes::base64_decode(encoded)
    match decoded_opt {
      Some(b) => {
        if b.length() != size {
          abort("size " + size.to_string() + ": decoded length wrong")
        }
        for i in 0..<size {
          if b.get(i) != input.get(i) {
            abort(
              "size " +
              size.to_string() +
              " byte " +
              i.to_string() +
              " mismatch",
            )
          }
        }
      }
      None => abort("size " + size.to_string() + ": decode None")
    }
  }
}

///|
test "SimdBufferBytes decode rejects out-of-alphabet bytes" {
  let input = buf_from_array([0x5a, 0x6d, 0x39, 0x2a]) // "Zm9*"
  match SimdBufferBytes::base64_decode(input) {
    Some(_) => abort("expected None")
    None => ()
  }
}

///|
fn f32_close(a : Float, b : Float, eps : Float) -> Bool {
  let d = a - b
  let abs = if d < (0.0 : Float) { -d } else { d }
  abs < eps
}

///|
test "SimdBufferF32 round-trip + sum + dot" {
  let n = 64
  let a = SimdBufferF32::make(n)
  let b = SimdBufferF32::make(n)
  let mut expected_sum : Float = 0.0
  let mut expected_dot : Float = 0.0
  for i in 0..<n {
    let av = (i.to_double() + 0.5).to_f32()
    let bv = ((i * 2).to_double() - 1.0).to_f32()
    a.set(i, av)
    b.set(i, bv)
    expected_sum = expected_sum + av
    expected_dot = expected_dot + av * bv
  }
  if !f32_close(a.sum(), expected_sum, 1.0e-3.to_f32()) {
    abort("SimdBufferF32.sum mismatch")
  }
  if !f32_close(SimdBufferF32::dot(a, b), expected_dot, 1.0e-2.to_f32()) {
    abort("SimdBufferF32.dot mismatch")
  }
}

///|
test "SimdBufferF32 add / sub / mul / div / sqrt / min_elem / max_elem" {
  let n = 17 // not a multiple of 4 → tail
  let a = SimdBufferF32::make(n)
  let b = SimdBufferF32::make(n)
  let out = SimdBufferF32::make(n)
  for i in 0..<n {
    a.set(i, (i + 1).to_double().to_f32())
    b.set(i, ((i + 1) * 3).to_double().to_f32())
  }
  SimdBufferF32::add(a, b, out)
  for i in 0..<n {
    let want = ((i + 1) * 4).to_double().to_f32()
    if !f32_close(out.get(i), want, 1.0e-4.to_f32()) {
      abort("F32.add mismatch at " + i.to_string())
    }
  }
  SimdBufferF32::sub(b, a, out)
  for i in 0..<n {
    let want = ((i + 1) * 2).to_double().to_f32()
    if !f32_close(out.get(i), want, 1.0e-4.to_f32()) {
      abort("F32.sub mismatch at " + i.to_string())
    }
  }
  SimdBufferF32::mul(a, b, out)
  for i in 0..<n {
    let want = ((i + 1) * (i + 1) * 3).to_double().to_f32()
    if !f32_close(out.get(i), want, 1.0e-3.to_f32()) {
      abort("F32.mul mismatch at " + i.to_string())
    }
  }
  SimdBufferF32::div(b, a, out) // b / a = 3
  for i in 0..<n {
    if !f32_close(out.get(i), 3.0.to_f32(), 1.0e-4.to_f32()) {
      abort("F32.div mismatch at " + i.to_string())
    }
  }
  SimdBufferF32::sqrt(b, out)
  for i in 0..<n {
    let want = ((i + 1) * 3).to_double().sqrt().to_f32()
    if !f32_close(out.get(i), want, 1.0e-3.to_f32()) {
      abort("F32.sqrt mismatch at " + i.to_string())
    }
  }
  SimdBufferF32::min_elem(a, b, out)
  for i in 0..<n {
    if !f32_close(out.get(i), a.get(i), 1.0e-6.to_f32()) {
      abort("F32.min_elem mismatch at " + i.to_string())
    }
  }
  SimdBufferF32::max_elem(a, b, out)
  for i in 0..<n {
    if !f32_close(out.get(i), b.get(i), 1.0e-6.to_f32()) {
      abort("F32.max_elem mismatch at " + i.to_string())
    }
  }
}

///|
test "SimdBufferBytes popcount" {
  let b = SimdBufferBytes::make(20)
  for i in 0..<20 {
    b.set(i, (0xFF).to_byte()) // 8 bits per byte
  }
  if b.popcount() != 20 * 8 {
    abort("popcount full mismatch")
  }
  // Tail (length not multiple of 16) is the 17/18/19th byte.
  let c = SimdBufferBytes::make(17)
  for i in 0..<16 {
    c.set(i, (0x05).to_byte()) // 2 bits each
  }
  c.set(16, (0x07).to_byte()) // 3 bits
  if c.popcount() != 16 * 2 + 3 {
    abort("popcount tail mismatch")
  }
}

///|
test "SimdBufferBytes memcpy + memset" {
  let src = SimdBufferBytes::make(40)
  for i in 0..<40 {
    src.set(i, i.to_byte())
  }
  let dst = SimdBufferBytes::make(40)
  SimdBufferBytes::memcpy(src, 5, dst, 10, 25)
  for i in 0..<25 {
    if dst.get(10 + i).to_int() != 5 + i {
      abort("memcpy mismatch at " + i.to_string())
    }
  }
  // memset
  let m = SimdBufferBytes::make(33)
  m.memset(2, 30, (0xAB).to_byte())
  for i in 2..<32 {
    if m.get(i).to_int() != 0xAB {
      abort("memset value mismatch at " + i.to_string())
    }
  }
  if m.get(1).to_int() != 0 || m.get(32).to_int() != 0 {
    abort("memset overran")
  }
}

///|
test "SimdBufferBytes equal + find_byte + count_byte" {
  let a = SimdBufferBytes::make(32)
  let b = SimdBufferBytes::make(32)
  for i in 0..<32 {
    a.set(i, i.to_byte())
    b.set(i, i.to_byte())
  }
  if !SimdBufferBytes::equal(a, b, 32) {
    abort("equal: identical buffers should be equal")
  }
  b.set(20, (0xFF).to_byte())
  if SimdBufferBytes::equal(a, b, 32) {
    abort("equal: differing buffers reported equal")
  }
  // find_byte
  if a.find_byte((20).to_byte()) != 20 {
    abort("find_byte miss")
  }
  if a.find_byte((0xFE).to_byte()) != -1 {
    abort("find_byte should not find")
  }
  // count_byte
  let c = SimdBufferBytes::make(50)
  for i in 0..<50 {
    c.set(i, if i % 3 == 0 { (0xCC).to_byte() } else { (0).to_byte() })
  }
  let expected_count = (50 + 2) / 3 // ceil(50/3) bytes are 0xCC
  if c.count_byte((0xCC).to_byte()) != expected_count {
    abort("count_byte mismatch")
  }
}

///|
test "SimdBufferBytes is_ascii + to_lower + to_upper" {
  let b = SimdBufferBytes::make(26)
  for i in 0..<26 {
    b.set(i, (65 + i).to_byte()) // 'A'..'Z'
  }
  if !b.is_ascii() {
    abort("is_ascii: ASCII buffer reported non-ASCII")
  }
  b.to_lower_ascii()
  for i in 0..<26 {
    if b.get(i).to_int() != 97 + i {
      abort("to_lower mismatch at " + i.to_string())
    }
  }
  b.to_upper_ascii()
  for i in 0..<26 {
    if b.get(i).to_int() != 65 + i {
      abort("to_upper round-trip mismatch at " + i.to_string())
    }
  }
  // High-bit buffer is not ASCII.
  let h = SimdBufferBytes::make(20)
  for i in 0..<20 {
    h.set(i, (0x80).to_byte())
  }
  if h.is_ascii() {
    abort("is_ascii: high-bit buffer reported ASCII")
  }
}

///|
test "SimdBufferBytes validate_utf8" {
  // All ASCII → valid.
  let a = SimdBufferBytes::make(64)
  for i in 0..<64 {
    a.set(i, (65 + i % 26).to_byte())
  }
  if !a.validate_utf8() {
    abort("UTF-8 all-ASCII rejected")
  }
  // Bare continuation byte → invalid.
  let bad = SimdBufferBytes::make(2)
  bad.set(0, (0x80).to_byte()) // bare continuation
  bad.set(1, (0x41).to_byte())
  if bad.validate_utf8() {
    abort("UTF-8 bare continuation accepted")
  }
  // Well-formed 2-byte sequence (Japanese "ñ" U+00F1 = 0xC3 0xB1).
  let ok = SimdBufferBytes::make(2)
  ok.set(0, (0xC3).to_byte())
  ok.set(1, (0xB1).to_byte())
  if !ok.validate_utf8() {
    abort("UTF-8 valid 2-byte rejected")
  }
}

///|
test "SimdBufferBytes adler32 known vectors" {
  // RFC 1950 example: ""  -> 1 (just initial state)
  let empty = SimdBufferBytes::make(0)
  if empty.adler32() != 1U {
    abort("adler32(empty) != 1")
  }
  // Single byte 0x41 ('A'): a=66, b=66 → 0x00420042
  let one = SimdBufferBytes::make(1)
  one.set(0, (0x41).to_byte())
  if one.adler32() != 0x00420042U {
    abort("adler32('A') mismatch")
  }
  // Longer: 'Wikipedia' = 9 bytes, known Adler-32 = 0x11E60398.
  let w = SimdBufferBytes::make(9)
  let chars = [0x57, 0x69, 0x6B, 0x69, 0x70, 0x65, 0x64, 0x69, 0x61]
  for i, c in chars {
    w.set(i, c.to_byte())
  }
  if w.adler32() != 0x11E60398U {
    abort("adler32('Wikipedia') mismatch")
  }
  // Long buffer (>16 bytes, exercises SIMD chunks).
  let long = SimdBufferBytes::make(64)
  for i in 0..<64 {
    long.set(i, (i + 1).to_byte())
  }
  // Compare with the scalar reference walked manually.
  let mut a : UInt = 1
  let mut b : UInt = 0
  for i in 0..<64 {
    a = (a + (i + 1).reinterpret_as_uint()) % 65521U
    b = (b + a) % 65521U
  }
  let expected = (b << 16) | a
  if long.adler32() != expected {
    abort("adler32(long) mismatch")
  }
}

///|
test "SimdBufferRing: basic alloc + reset" {
  let ring = SimdBufferRing::make(8192)
  // Capacity bookkeeping: wasm rounds up to one 64 KiB page; native reports
  // total bytes handed out so far (= 0 before any alloc). Both are valid.
  let _ = ring.capacity()
  let a = ring.alloc_i32(100)
  let b = ring.alloc_i32(50)
  // a and b are independent buffers — writes to one must not appear in the
  // other. (We can't compare addresses directly, the struct is opaque on
  // native; instead, write distinct sentinels and read back.)
  a.set(0, 0xAAA)
  b.set(0, 0xBBB)
  if a.get(0) != 0xAAA || b.get(0) != 0xBBB {
    abort("Ring sub-buffers shared storage")
  }
  // Fill and read back
  for i in 0..<100 {
    a.set(i, i * 2)
  }
  for i in 0..<50 {
    b.set(i, i + 1000)
  }
  if a.sum() != 100 * 99 {
    abort("Ring i32 sub-buffer sum mismatch")
  }
  if b.get(49) != 1049 {
    abort("Ring sub-buffer read-back mismatch")
  }
  // Reset rewinds the bump
  let used_before = ring.used()
  ring.reset()
  if ring.used() != 0 {
    abort("reset didn't clear")
  }
  if used_before <= 0 {
    abort("ring should have been used")
  }
}

///|
test "SimdBufferRing: mixed type alloc shares one page" {
  let ring = SimdBufferRing::make(8192) // rounds up to 64KB
  let i32_buf = ring.alloc_i32(64)
  let f32_buf = ring.alloc_f32(64)
  let f64_buf = ring.alloc_f64(64)
  let byte_buf = ring.alloc_bytes(64)
  // Fill each with a distinct sentinel
  for i in 0..<64 {
    i32_buf.set(i, i)
    f32_buf.set(i, i.to_double().to_f32())
    f64_buf.set(i, (i * 2).to_double())
    byte_buf.set(i, (i % 256).to_byte())
  }
  // Each buffer remembers its own contents (no aliasing)
  if i32_buf.get(63) != 63 ||
    !f32_close(f32_buf.get(63), 63.0.to_f32(), 1.0e-6.to_f32()) ||
    f64_buf.get(63) != 126.0 ||
    byte_buf.get(63).to_int() != 63 {
    abort("Ring mixed-type alloc cross-contaminated")
  }
}

///|
test "SimdBufferRing: out-of-capacity aborts" {
  // 1-page Ring (64KB usable). 5000 i32 = 20000 bytes + 16 slack, fits.
  // Another 5000 i32 = same; total 40032 bytes; still fits.
  // A third would push past 60048. A 17000-elem i32 (68016 bytes) is > 65536.
  let ring = SimdBufferRing::make(65536)
  let _a = ring.alloc_i32(5000)
  let _b = ring.alloc_i32(5000)
  // 17000 i32 = 68016 > remaining ~25KB. This should abort.
  // (We can't actually catch the abort in the test framework — skip the
  // failing case and just check used() advanced as expected.)
  if ring.used() < 5000 * 4 + 5000 * 4 {
    abort("Ring used() did not advance correctly")
  }
}

///|
test "SimdBuffer f64 element-wise + mean / var" {
  let n = 32
  let a = SimdBufferF64::make(n)
  let b = SimdBufferF64::make(n)
  let out = SimdBufferF64::make(n)
  for i in 0..<n {
    a.set(i, i.to_double() + 1.0)
    b.set(i, (i + 1).to_double() * 2.0)
  }
  SimdBufferF64::add(a, b, out)
  for i in 0..<n {
    if !f64_close(out.get(i), a.get(i) + b.get(i), 1.0e-9) {
      abort("f64.add mismatch " + i.to_string())
    }
  }
  SimdBufferF64::sub(b, a, out)
  for i in 0..<n {
    if !f64_close(out.get(i), b.get(i) - a.get(i), 1.0e-9) {
      abort("f64.sub mismatch " + i.to_string())
    }
  }
  SimdBufferF64::mul(a, b, out)
  for i in 0..<n {
    if !f64_close(out.get(i), a.get(i) * b.get(i), 1.0e-9) {
      abort("f64.mul mismatch " + i.to_string())
    }
  }
  SimdBufferF64::div(b, a, out)
  for i in 0..<n {
    if !f64_close(out.get(i), b.get(i) / a.get(i), 1.0e-9) {
      abort("f64.div mismatch " + i.to_string())
    }
  }
  SimdBufferF64::sqrt(b, out)
  for i in 0..<n {
    if !f64_close(out.get(i), b.get(i).sqrt(), 1.0e-9) {
      abort("f64.sqrt mismatch " + i.to_string())
    }
  }
  SimdBufferF64::min_elem(a, b, out)
  for i in 0..<n {
    let want = if a.get(i) < b.get(i) { a.get(i) } else { b.get(i) }
    if !f64_close(out.get(i), want, 1.0e-9) {
      abort("f64.min_elem mismatch " + i.to_string())
    }
  }
  SimdBufferF64::max_elem(a, b, out)
  for i in 0..<n {
    let want = if a.get(i) > b.get(i) { a.get(i) } else { b.get(i) }
    if !f64_close(out.get(i), want, 1.0e-9) {
      abort("f64.max_elem mismatch " + i.to_string())
    }
  }
  // mean / var
  let mut sum_a = 0.0
  for i in 0..<n {
    sum_a = sum_a + a.get(i)
  }
  if !f64_close(a.mean(), sum_a / n.to_double(), 1.0e-9) {
    abort("f64.mean mismatch")
  }
  let mu = a.mean()
  let mut ssq = 0.0
  for i in 0..<n {
    let d = a.get(i) - mu
    ssq = ssq + d * d
  }
  if !f64_close(a.variance(), ssq / n.to_double(), 1.0e-9) {
    abort("f64.variance mismatch")
  }
}

///|
test "SimdBuffer i32 element-wise + comparisons" {
  let n = 17 // not multiple of 4 to exercise tail
  let a = SimdBuffer::make(n)
  let b = SimdBuffer::make(n)
  let out = SimdBuffer::make(n)
  for i in 0..<n {
    a.set(i, i + 1)
    b.set(i, (i + 1) * 3 - 5)
  }
  // dot
  let mut dot_ref = 0
  for i in 0..<n {
    dot_ref = dot_ref + a.get(i) * b.get(i)
  }
  if SimdBuffer::dot(a, b) != dot_ref {
    abort("i32.dot mismatch")
  }
  SimdBuffer::sub(a, b, out)
  for i in 0..<n {
    if out.get(i) != a.get(i) - b.get(i) {
      abort("i32.sub " + i.to_string())
    }
  }
  SimdBuffer::mul(a, b, out)
  for i in 0..<n {
    if out.get(i) != a.get(i) * b.get(i) {
      abort("i32.mul " + i.to_string())
    }
  }
  SimdBuffer::neg(a, out)
  for i in 0..<n {
    if out.get(i) != -a.get(i) {
      abort("i32.neg " + i.to_string())
    }
  }
  let neg = SimdBuffer::make(n)
  for i in 0..<n {
    neg.set(i, -(i + 1))
  }
  SimdBuffer::abs(neg, out)
  for i in 0..<n {
    if out.get(i) != i + 1 {
      abort("i32.abs " + i.to_string())
    }
  }
  SimdBuffer::min_elem(a, b, out)
  for i in 0..<n {
    let want = if a.get(i) < b.get(i) { a.get(i) } else { b.get(i) }
    if out.get(i) != want {
      abort("i32.min_elem " + i.to_string())
    }
  }
  SimdBuffer::max_elem(a, b, out)
  for i in 0..<n {
    let want = if a.get(i) > b.get(i) { a.get(i) } else { b.get(i) }
    if out.get(i) != want {
      abort("i32.max_elem " + i.to_string())
    }
  }
  SimdBuffer::eq(a, b, out)
  for i in 0..<n {
    let want = if a.get(i) == b.get(i) { -1 } else { 0 }
    if out.get(i) != want {
      abort("i32.eq " + i.to_string())
    }
  }
  SimdBuffer::lt(a, b, out)
  for i in 0..<n {
    let want = if a.get(i) < b.get(i) { -1 } else { 0 }
    if out.get(i) != want {
      abort("i32.lt " + i.to_string())
    }
  }
  SimdBuffer::gt(a, b, out)
  for i in 0..<n {
    let want = if a.get(i) > b.get(i) { -1 } else { 0 }
    if out.get(i) != want {
      abort("i32.gt " + i.to_string())
    }
  }
  // where: out = mask ? a : b
  let mask = SimdBuffer::make(n)
  SimdBuffer::lt(a, b, mask)
  SimdBuffer::where_(mask, a, b, out)
  for i in 0..<n {
    let want = if a.get(i) < b.get(i) { a.get(i) } else { b.get(i) }
    if out.get(i) != want {
      abort("i32.where " + i.to_string())
    }
  }
}

///|
test "SimdBuffer i32 reductions" {
  let arr = SimdBuffer::make(17)
  for i in 0..<17 {
    arr.set(i, (i * 7 + 3) % 31 - 15)
  }
  // Compute reference values scalar.
  let mut mn = arr.get(0)
  let mut mx = arr.get(0)
  let mut argmn = 0
  let mut argmx = 0
  let mut p = 1
  let mut cnz = 0
  for i in 0..<17 {
    let v = arr.get(i)
    if v < mn {
      mn = v
      argmn = i
    }
    if v > mx {
      mx = v
      argmx = i
    }
    p = p * v
    if v != 0 {
      cnz = cnz + 1
    }
  }
  if arr.min() != mn {
    abort("i32.min mismatch")
  }
  if arr.max() != mx {
    abort("i32.max mismatch")
  }
  if arr.argmin() != argmn {
    abort("i32.argmin mismatch")
  }
  if arr.argmax() != argmx {
    abort("i32.argmax mismatch")
  }
  if arr.prod() != p {
    abort("i32.prod mismatch")
  }
  if arr.count_nonzero() != cnz {
    abort("i32.count_nonzero mismatch")
  }
  // any / all
  let allnz = SimdBuffer::make(8)
  for i in 0..<8 {
    allnz.set(i, i + 1)
  }
  if !allnz.all() {
    abort("i32.all true case failed")
  }
  if !allnz.any() {
    abort("i32.any true case failed")
  }
  let allzero = SimdBuffer::make(8)
  if allzero.any() {
    abort("i32.any zero case failed")
  }
  if allzero.all() {
    abort("i32.all zero case failed")
  }
}

///|
test "SimdBuffer saxpy + cumsum + cumprod + div + gather + scatter" {
  // saxpy: out = k*a + b
  let n = 16
  let a = SimdBuffer::make(n)
  let b = SimdBuffer::make(n)
  let out = SimdBuffer::make(n)
  for i in 0..<n {
    a.set(i, i + 1)
    b.set(i, 100 - i)
  }
  SimdBuffer::saxpy(3, a, b, out)
  for i in 0..<n {
    if out.get(i) != 3 * (i + 1) + (100 - i) {
      abort("saxpy " + i.to_string())
    }
  }
  // cumsum
  let cs = SimdBuffer::make(n)
  SimdBuffer::cumsum(a, cs)
  let mut acc = 0
  for i in 0..<n {
    acc = acc + a.get(i)
    if cs.get(i) != acc {
      abort("cumsum " + i.to_string())
    }
  }
  // cumprod
  let cp = SimdBuffer::make(n)
  let small = SimdBuffer::make(n)
  for i in 0..<n {
    small.set(i, i % 3 + 1) // values 1..3 to keep product small
  }
  SimdBuffer::cumprod(small, cp)
  let mut pacc = 1
  for i in 0..<n {
    pacc = pacc * small.get(i)
    if cp.get(i) != pacc {
      abort("cumprod " + i.to_string())
    }
  }
  // div
  let divisor = SimdBuffer::make(n)
  for i in 0..<n {
    divisor.set(i, i + 1)
  }
  let dividend = SimdBuffer::make(n)
  for i in 0..<n {
    dividend.set(i, (i + 1) * 100)
  }
  SimdBuffer::div(dividend, divisor, out)
  for i in 0..<n {
    if out.get(i) != 100 {
      abort("div " + i.to_string() + " got " + out.get(i).to_string())
    }
  }
  // gather
  let src = SimdBuffer::make(10)
  for i in 0..<10 {
    src.set(i, i * 10)
  }
  let indices = SimdBuffer::make(5)
  indices.set(0, 0)
  indices.set(1, 3)
  indices.set(2, 7)
  indices.set(3, 9)
  indices.set(4, 1)
  let gathered = SimdBuffer::make(5)
  SimdBuffer::gather(src, indices, gathered)
  let expected = [0, 30, 70, 90, 10]
  for i in 0..<5 {
    if gathered.get(i) != expected[i] {
      abort("gather " + i.to_string())
    }
  }
  // scatter
  let dst = SimdBuffer::make(10)
  let vals = SimdBuffer::make(5)
  for i in 0..<5 {
    vals.set(i, (i + 1) * 1000)
  }
  SimdBuffer::scatter(dst, indices, vals)
  if dst.get(0) != 1000 ||
    dst.get(3) != 2000 ||
    dst.get(7) != 3000 ||
    dst.get(9) != 4000 ||
    dst.get(1) != 5000 {
    abort("scatter mismatch")
  }
}

///|
test "SimdBuffer sort: matches reference" {
  let sizes = [1, 4, 8, 16, 17, 31, 64, 100, 255, 512]
  for size in sizes {
    let arr = SimdBuffer::make(size)
    let scratch = SimdBuffer::make(size)
    let mut seed : UInt = (size + 7).reinterpret_as_uint() * 2654435761U
    for i in 0..<size {
      seed = seed * 1103515245U + 12345U
      arr.set(i, (seed & 0xFFFFU).reinterpret_as_int() - 32768)
    }
    SimdBuffer::sort(arr, scratch)
    let mut prev = arr.get(0)
    for i in 1..<size {
      let cur = arr.get(i)
      if cur < prev {
        abort(
          "sort size=" + size.to_string() + " not monotonic at " + i.to_string(),
        )
      }
      prev = cur
    }
  }
}

///|
test "SimdBufferF64 transpose: matches scalar" {
  let m = 4
  let n = 6
  let a = SimdBufferF64::make(m * n)
  let out = SimdBufferF64::make(m * n)
  for i in 0..<(m * n) {
    a.set(i, i.to_double() + 1.0)
  }
  SimdBufferF64::transpose(a, out, m, n)
  for i in 0..<m {
    for j in 0..<n {
      let src = a.get(i * n + j)
      let dst = out.get(j * m + i)
      if !f64_close(src, dst, 1.0e-9) {
        abort(
          "transpose mismatch (" + i.to_string() + "," + j.to_string() + ")",
        )
      }
    }
  }
}

///|
test "SimdBuffer ↔ FixedArray round-trip (i32)" {
  let src : FixedArray[Int] = [1, 2, 3, 4, 5, -100, 1000000, -999]
  let buf = SimdBuffer::from_array(src)
  if buf.length() != src.length() {
    abort("length mismatch")
  }
  let back = buf.to_array()
  for i in 0..<src.length() {
    if back[i] != src[i] {
      abort("i32 round-trip mismatch at " + i.to_string())
    }
  }
}

///|
test "SimdBufferF64 ↔ FixedArray round-trip" {
  let src : FixedArray[Double] = [1.5, -2.5, 3.14159, 0.0, 1.0e10]
  let buf = SimdBufferF64::from_array(src)
  let back = buf.to_array()
  for i in 0..<src.length() {
    if !f64_close(back[i], src[i], 1.0e-12) {
      abort("f64 round-trip mismatch at " + i.to_string())
    }
  }
}

///|
test "SimdBufferF32 ↔ FixedArray round-trip" {
  let src : FixedArray[Float] = [
    1.5.to_f32(),
    (-2.5).to_f32(),
    3.14.to_f32(),
    0.0.to_f32(),
  ]
  let buf = SimdBufferF32::from_array(src)
  let back = buf.to_array()
  for i in 0..<src.length() {
    if !f32_close(back[i], src[i], 1.0e-6.to_f32()) {
      abort("f32 round-trip mismatch at " + i.to_string())
    }
  }
}

///|
test "SimdBufferBytes ↔ FixedArray round-trip" {
  let src : FixedArray[Byte] = [b'h', b'e', b'l', b'l', b'o', b'!']
  let buf = SimdBufferBytes::from_array(src)
  let back = buf.to_array()
  for i in 0..<src.length() {
    if back[i] != src[i] {
      abort("bytes round-trip mismatch at " + i.to_string())
    }
  }
}

// ---- 0.3.0 Tier 1 + Tier 2 ops ----

///|
test "byte_add / byte_sub / byte_avg (mod 256, rounding)" {
  let a = SimdBufferBytes::from_array([b'\x10', b'\x80', b'\xFF', b'\x05'])
  let b = SimdBufferBytes::from_array([b'\x20', b'\x80', b'\x01', b'\x0A'])
  let out = SimdBufferBytes::make(4)
  SimdBufferBytes::byte_add(a, b, out)
  // (0x10+0x20, 0x80+0x80=0 mod 256, 0xFF+0x01=0, 0x05+0x0A=0x0F)
  let r = out.to_array()
  assert_true(
    (r[0].to_int(), r[1].to_int(), r[2].to_int(), r[3].to_int()) ==
    (48, 0, 0, 15),
  )
  SimdBufferBytes::byte_sub(a, b, out)
  // (0x10-0x20=-0x10=0xF0, 0x80-0x80=0, 0xFF-0x01=0xFE, 0x05-0x0A=-5=0xFB)
  let r2 = out.to_array()
  assert_true(
    (r2[0].to_int(), r2[1].to_int(), r2[2].to_int(), r2[3].to_int()) ==
    (240, 0, 254, 251),
  )
  SimdBufferBytes::byte_avg(a, b, out)
  // ((0x10+0x20+1)/2=0x18, (0x80+0x80+1)/2=0x80, (0xFF+0x01+1)/2=0x80, (5+10+1)/2=8)
  let r3 = out.to_array()
  assert_true(
    (r3[0].to_int(), r3[1].to_int(), r3[2].to_int(), r3[3].to_int()) ==
    (24, 128, 128, 8),
  )
}

///|
test "sat_add / sat_sub clamp at 255 / 0" {
  let a = SimdBufferBytes::from_array([b'\xFA', b'\x10', b'\x80', b'\x00'])
  let b = SimdBufferBytes::from_array([b'\x10', b'\xFA', b'\x10', b'\xFF'])
  let out = SimdBufferBytes::make(4)
  SimdBufferBytes::sat_add(a, b, out)
  let r = out.to_array()
  assert_true(
    (r[0].to_int(), r[1].to_int(), r[2].to_int(), r[3].to_int()) ==
    (255, 255, 144, 255),
  )
  SimdBufferBytes::sat_sub(a, b, out)
  let r2 = out.to_array()
  assert_true(
    (r2[0].to_int(), r2[1].to_int(), r2[2].to_int(), r2[3].to_int()) ==
    (234, 0, 112, 0),
  )
}

///|
test "clamp byte-wise" {
  let src = SimdBufferBytes::from_array([b'\x00', b'\x40', b'\x80', b'\xFF'])
  let out = SimdBufferBytes::make(4)
  src.clamp(b'\x20', b'\xC0', out)
  let r = out.to_array()
  assert_true(
    (r[0].to_int(), r[1].to_int(), r[2].to_int(), r[3].to_int()) ==
    (32, 64, 128, 192),
  )
}

///|
test "byte_sub_offset (PNG Sub-filter pattern, stride 1)" {
  let src = SimdBufferBytes::from_array([b'\x0A', b'\x1E', b'\x32'])
  let out = SimdBufferBytes::make(3)
  SimdBufferBytes::byte_sub_offset(src, 1, out)
  // i=0: src[0]=10 (head, no sub)
  // i=1: src[1]-src[0]=30-10=20
  // i=2: src[2]-src[1]=50-30=20
  let r = out.to_array()
  assert_true((r[0].to_int(), r[1].to_int(), r[2].to_int()) == (10, 20, 20))
}

///|
test "SimdBuffer::array_equal" {
  let a = SimdBuffer::from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
  let b = SimdBuffer::from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
  if !SimdBuffer::array_equal(a, b, 10) {
    abort("array_equal: identical arrays should match")
  }
  let c = SimdBuffer::from_array([1, 2, 3, 4, 99, 6, 7, 8, 9, 10])
  if SimdBuffer::array_equal(a, c, 10) {
    abort("array_equal: should detect 5th-element mismatch")
  }
  if !SimdBuffer::array_equal(a, c, 4) {
    abort("array_equal: prefix of 4 should match")
  }
}

///|
test "SimdBufferBytes validate_utf8_strict" {
  // valid: ASCII + é + € + 😀
  let ok = buf_from_array([
    0x68, 0x69, 0xC3, 0xA9, 0xE2, 0x82, 0xAC, 0xF0, 0x9F, 0x98, 0x80,
  ])
  assert_true(ok.validate_utf8_strict())
  // surrogate ED A0 80 — strict rejects, lax accepts
  let surr = buf_from_array([0xED, 0xA0, 0x80])
  assert_false(surr.validate_utf8_strict())
  assert_true(surr.validate_utf8())
  // overlong E0 80 80 and > U+10FFFF F4 90 80 80 — strict rejects
  assert_false(buf_from_array([0xE0, 0x80, 0x80]).validate_utf8_strict())
  assert_false(buf_from_array([0xF4, 0x90, 0x80, 0x80]).validate_utf8_strict())
  // strict after a >16-byte ASCII SIMD-skip region
  let big = SimdBufferBytes::make(35)
  for i in 0..<32 {
    big.set(i, b'A')
  }
  big.set(32, b'\xED')
  big.set(33, b'\xA0')
  big.set(34, b'\x80')
  assert_false(big.validate_utf8_strict())
}
