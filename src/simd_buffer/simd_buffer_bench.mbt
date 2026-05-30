///|
fn scalar_sum(b : SimdBuffer) -> Int {
  let mut acc = 0
  let n = b.length()
  for i in 0..<n {
    acc = acc + b.get(i)
  }
  acc
}

///|
test "bench/sum" (bb : @bench.T) {
  let n = 1024
  let buf = SimdBuffer::make(n)
  for i in 0..<n {
    buf.set(i, i + 1)
  }
  bb.bench(name="scalar_sum_1024", fn() { bb.keep(scalar_sum(buf)) })
  bb.bench(name="simd_sum_1024", fn() { bb.keep(buf.sum()) })
}

///|
fn scalar_add(a : SimdBuffer, b : SimdBuffer, out : SimdBuffer) -> Unit {
  let n = a.length()
  for i in 0..<n {
    out.set(i, a.get(i) + b.get(i))
  }
}

///|
test "bench/add" (bb : @bench.T) {
  let n = 1024
  let a = SimdBuffer::make(n)
  let b = SimdBuffer::make(n)
  let out = SimdBuffer::make(n)
  for i in 0..<n {
    a.set(i, i)
    b.set(i, i * 2)
  }
  bb.bench(name="scalar_add_1024", fn() { scalar_add(a, b, out) })
  bb.bench(name="simd_add_1024", fn() { SimdBuffer::add(a, b, out) })
}

///|
fn scalar_matmul(
  a : SimdBufferF64,
  b_t : SimdBufferF64,
  c : SimdBufferF64,
  m : Int,
  k : Int,
  n : Int,
) -> Unit {
  for i in 0..<m {
    for j in 0..<n {
      let mut acc = 0.0
      for p in 0..<k {
        acc = acc + a.get(i * k + p) * b_t.get(j * k + p)
      }
      c.set(i * n + j, acc)
    }
  }
}

///|
test "bench/matmul_f64 64x64" (bb : @bench.T) {
  let m = 64
  let k = 64
  let n = 64
  let a = SimdBufferF64::make(m * k)
  let b_t = SimdBufferF64::make(n * k)
  let c = SimdBufferF64::make(m * n)
  for i in 0..<(m * k) {
    a.set(i, (i % 13).to_double() - 6.0)
  }
  for i in 0..<(n * k) {
    b_t.set(i, (i % 11).to_double() - 5.0)
  }
  bb.bench(name="scalar_matmul_64", fn() { scalar_matmul(a, b_t, c, m, k, n) })
  bb.bench(name="simd_matmul_64", fn() {
    SimdBufferF64::matmul(a, b_t, c, m, k, n)
  })
}

///|
test "bench/base64_encode_into 4096B" (bb : @bench.T) {
  let n = 4096
  let input = SimdBufferBytes::make(n)
  for i in 0..<n {
    input.set(i, (i % 256).to_byte())
  }
  let out = SimdBufferBytes::make((n + 2) / 3 * 4)
  bb.bench(name="simd_base64_encode_into_4096", fn() {
    SimdBufferBytes::base64_encode_into(input, out)
  })
}

///|
test "bench/base64_decode_into 4096B" (bb : @bench.T) {
  let n = 4095
  let input = SimdBufferBytes::make(n)
  for i in 0..<n {
    input.set(i, (i % 256).to_byte())
  }
  let encoded_tmp = SimdBufferBytes::make((n + 2) / 3 * 4)
  SimdBufferBytes::base64_encode_into(input, encoded_tmp)
  let out = SimdBufferBytes::make(n)
  bb.bench(name="simd_base64_decode_into_5460", fn() {
    bb.keep(SimdBufferBytes::base64_decode_into(encoded_tmp, out))
  })
}

///|
test "bench/base64_encode via Ring (alloc per call but in-page)" (bb : @bench.T) {
  let n = 4096
  let input = SimdBufferBytes::make(n)
  for i in 0..<n {
    input.set(i, (i % 256).to_byte())
  }
  // One big Ring; reset between calls = nearly-free allocation.
  let ring = SimdBufferRing::make(65536)
  bb.bench(name="ring_base64_encode_4096", fn() {
    ring.reset()
    let out = ring.alloc_bytes((n + 2) / 3 * 4)
    SimdBufferBytes::base64_encode_into(input, out)
  })
}

///|
test "bench/Ring alloc cost vs SimdBufferBytes::make" (bb : @bench.T) {
  let ring = SimdBufferRing::make(65536)
  bb.bench(name="ring_alloc_bytes_1024", fn() {
    ring.reset()
    bb.keep(ring.alloc_bytes(1024))
  })
  bb.bench(name="make_bytes_1024", fn() { bb.keep(SimdBufferBytes::make(1024)) })
}

// ---- i32 element-wise ----

///|
test "bench/i32 sub/mul" (bb : @bench.T) {
  let n = 1024
  let a = SimdBuffer::make(n)
  let b = SimdBuffer::make(n)
  let out = SimdBuffer::make(n)
  for i in 0..<n {
    a.set(i, i + 100)
    b.set(i, i + 1)
  }
  bb.bench(name="simd_sub_1024", fn() { SimdBuffer::sub(a, b, out) })
  bb.bench(name="simd_mul_1024", fn() { SimdBuffer::mul(a, b, out) })
}

///|
test "bench/i32 neg/abs" (bb : @bench.T) {
  let n = 1024
  let a = SimdBuffer::make(n)
  let out = SimdBuffer::make(n)
  for i in 0..<n {
    a.set(i, i - 512)
  }
  bb.bench(name="simd_neg_1024", fn() { SimdBuffer::neg(a, out) })
  bb.bench(name="simd_abs_1024", fn() { SimdBuffer::abs(a, out) })
}

///|
test "bench/i32 min_elem/max_elem/eq/lt/gt" (bb : @bench.T) {
  let n = 1024
  let a = SimdBuffer::make(n)
  let b = SimdBuffer::make(n)
  let out = SimdBuffer::make(n)
  for i in 0..<n {
    a.set(i, i)
    b.set(i, n - i)
  }
  bb.bench(name="simd_min_elem_1024", fn() { SimdBuffer::min_elem(a, b, out) })
  bb.bench(name="simd_max_elem_1024", fn() { SimdBuffer::max_elem(a, b, out) })
  bb.bench(name="simd_eq_1024", fn() { SimdBuffer::eq(a, b, out) })
  bb.bench(name="simd_lt_1024", fn() { SimdBuffer::lt(a, b, out) })
  bb.bench(name="simd_gt_1024", fn() { SimdBuffer::gt(a, b, out) })
}

///|
test "bench/i32 saxpy/dot" (bb : @bench.T) {
  let n = 1024
  let a = SimdBuffer::make(n)
  let b = SimdBuffer::make(n)
  let out = SimdBuffer::make(n)
  for i in 0..<n {
    a.set(i, i + 1)
    b.set(i, n - i)
  }
  bb.bench(name="simd_saxpy_1024", fn() { SimdBuffer::saxpy(3, a, b, out) })
  bb.bench(name="simd_dot_1024", fn() { bb.keep(SimdBuffer::dot(a, b)) })
}

///|
test "bench/i32 reductions" (bb : @bench.T) {
  let n = 1024
  let a = SimdBuffer::make(n)
  for i in 0..<n {
    a.set(i, i - 500)
  }
  bb.bench(name="simd_min_1024", fn() { bb.keep(a.min()) })
  bb.bench(name="simd_max_1024", fn() { bb.keep(a.max()) })
  bb.bench(name="simd_argmin_1024", fn() { bb.keep(a.argmin()) })
  bb.bench(name="simd_argmax_1024", fn() { bb.keep(a.argmax()) })
  bb.bench(name="simd_count_nonzero_1024", fn() { bb.keep(a.count_nonzero()) })
}

///|
test "bench/i32 cumsum/cumprod/div" (bb : @bench.T) {
  let n = 1024
  let a = SimdBuffer::make(n)
  let b = SimdBuffer::make(n)
  let out = SimdBuffer::make(n)
  for i in 0..<n {
    a.set(i, i % 3 + 1)
    b.set(i, i % 7 + 1)
  }
  bb.bench(name="simd_cumsum_1024", fn() { SimdBuffer::cumsum(a, out) })
  bb.bench(name="simd_cumprod_1024", fn() { SimdBuffer::cumprod(a, out) })
  bb.bench(name="simd_div_1024", fn() { SimdBuffer::div(a, b, out) })
}

///|
test "bench/i32 sort 1024" (bb : @bench.T) {
  let n = 1024
  let ring = SimdBufferRing::make(n * 4 * 4)
  let template = SimdBuffer::make(n)
  let mut seed : UInt = 0x12345678U
  for i in 0..<n {
    seed = seed * 1103515245U + 12345U
    template.set(i, (seed >> 8).reinterpret_as_int() % 1000)
  }
  bb.bench(name="simd_sort_1024", fn() {
    ring.reset()
    let arr = ring.alloc_i32(n)
    arr.copy_from_array(template.to_array())
    let scratch = ring.alloc_i32(n)
    SimdBuffer::sort(arr, scratch)
  })
}

// ---- f64 ----

///|
test "bench/f64 element-wise" (bb : @bench.T) {
  let n = 1024
  let a = SimdBufferF64::make(n)
  let b = SimdBufferF64::make(n)
  let out = SimdBufferF64::make(n)
  for i in 0..<n {
    a.set(i, i.to_double() + 0.5)
    b.set(i, (i + 1).to_double() * 1.5)
  }
  bb.bench(name="simd_f64_add_1024", fn() { SimdBufferF64::add(a, b, out) })
  bb.bench(name="simd_f64_sub_1024", fn() { SimdBufferF64::sub(a, b, out) })
  bb.bench(name="simd_f64_mul_1024", fn() { SimdBufferF64::mul(a, b, out) })
  bb.bench(name="simd_f64_div_1024", fn() { SimdBufferF64::div(a, b, out) })
  bb.bench(name="simd_f64_sqrt_1024", fn() { SimdBufferF64::sqrt(a, out) })
}

///|
test "bench/f64 reductions" (bb : @bench.T) {
  let n = 1024
  let a = SimdBufferF64::make(n)
  for i in 0..<n {
    a.set(i, (i % 100).to_double() - 50.0)
  }
  bb.bench(name="simd_f64_sum_1024", fn() { bb.keep(a.sum()) })
  bb.bench(name="simd_f64_mean_1024", fn() { bb.keep(a.mean()) })
  bb.bench(name="simd_f64_variance_1024", fn() { bb.keep(a.variance()) })
}

///|
test "bench/f64 gemv 256x256" (bb : @bench.T) {
  let m = 256
  let n = 256
  let a = SimdBufferF64::make(m * n)
  let x = SimdBufferF64::make(n)
  let y = SimdBufferF64::make(m)
  for i in 0..<(m * n) {
    a.set(i, (i % 13).to_double())
  }
  for i in 0..<n {
    x.set(i, i.to_double())
  }
  bb.bench(name="simd_gemv_256", fn() { SimdBufferF64::gemv(a, x, y, m, n) })
}

///|
test "bench/f64 transpose 128x128" (bb : @bench.T) {
  let m = 128
  let n = 128
  let a = SimdBufferF64::make(m * n)
  let out = SimdBufferF64::make(m * n)
  for i in 0..<(m * n) {
    a.set(i, i.to_double())
  }
  bb.bench(name="simd_transpose_128", fn() {
    SimdBufferF64::transpose(a, out, m, n)
  })
}

// ---- f32 ----

///|
test "bench/f32 element-wise + reductions" (bb : @bench.T) {
  let n = 1024
  let a = SimdBufferF32::make(n)
  let b = SimdBufferF32::make(n)
  let out = SimdBufferF32::make(n)
  for i in 0..<n {
    a.set(i, Float::from_double(i.to_double() + 0.5))
    b.set(i, Float::from_double((i + 1).to_double() * 1.5))
  }
  bb.bench(name="simd_f32_add_1024", fn() { SimdBufferF32::add(a, b, out) })
  bb.bench(name="simd_f32_mul_1024", fn() { SimdBufferF32::mul(a, b, out) })
  bb.bench(name="simd_f32_sqrt_1024", fn() { SimdBufferF32::sqrt(a, out) })
  bb.bench(name="simd_f32_sum_1024", fn() { bb.keep(a.sum()) })
  bb.bench(name="simd_f32_dot_1024", fn() { bb.keep(SimdBufferF32::dot(a, b)) })
}

// ---- bytes ----

///|
test "bench/bytes bulk 4096" (bb : @bench.T) {
  let n = 4096
  let a = SimdBufferBytes::make(n)
  let b = SimdBufferBytes::make(n)
  for i in 0..<n {
    a.set(i, (i % 256).to_byte())
    b.set(i, (i % 256).to_byte())
  }
  bb.bench(name="simd_popcount_4096", fn() { bb.keep(a.popcount()) })
  bb.bench(name="simd_find_byte_4096", fn() { bb.keep(a.find_byte(b'\xFF')) })
  bb.bench(name="simd_count_byte_4096", fn() { bb.keep(a.count_byte(b'\x42')) })
  bb.bench(name="simd_equal_4096", fn() {
    bb.keep(SimdBufferBytes::equal(a, b, n))
  })
  bb.bench(name="simd_is_ascii_4096", fn() { bb.keep(a.is_ascii()) })
  bb.bench(name="simd_adler32_4096", fn() { bb.keep(a.adler32()) })
}

///|
test "bench/bytes memcpy/memset 4096" (bb : @bench.T) {
  let n = 4096
  let src = SimdBufferBytes::make(n)
  let dst = SimdBufferBytes::make(n)
  for i in 0..<n {
    src.set(i, (i % 256).to_byte())
  }
  bb.bench(name="simd_memcpy_4096", fn() {
    SimdBufferBytes::memcpy(src, 0, dst, 0, n)
  })
  bb.bench(name="simd_memset_4096", fn() { dst.memset(0, n, b'\x42') })
}

///|
test "bench/utf8 validate (all-ASCII 4096)" (bb : @bench.T) {
  let n = 4096
  let a = SimdBufferBytes::make(n)
  for i in 0..<n {
    a.set(i, (65 + i % 26).to_byte())
  }
  bb.bench(name="simd_validate_utf8_ascii_4096", fn() {
    bb.keep(a.validate_utf8())
  })
}

///|
test "bench/ASCII case (4096)" (bb : @bench.T) {
  let n = 4096
  let a = SimdBufferBytes::make(n)
  for i in 0..<n {
    a.set(i, (65 + i % 26).to_byte())
  }
  bb.bench(name="simd_to_lower_4096", fn() { a.to_lower_ascii() })
  bb.bench(name="simd_to_upper_4096", fn() { a.to_upper_ascii() })
}

///|
/// Scalar reference rgb→rgba (per-pixel get/set), the old non-SIMD path.
fn scalar_rgb_to_rgba(
  src : SimdBufferBytes,
  alpha : Byte,
  out : SimdBufferBytes,
) -> Unit {
  let n = src.length() / 3
  for i in 0..<n {
    out.set(i * 4, src.get(i * 3))
    out.set(i * 4 + 1, src.get(i * 3 + 1))
    out.set(i * 4 + 2, src.get(i * 3 + 2))
    out.set(i * 4 + 3, alpha)
  }
}

///|
test "bench/rgb_to_rgba" (bb : @bench.T) {
  let npix = 4096
  let rgb = SimdBufferBytes::make(npix * 3)
  for i in 0..<(npix * 3) {
    rgb.set(i, (i % 256).to_byte())
  }
  let rgba = SimdBufferBytes::make(npix * 4)
  bb.bench(name="scalar_rgb_to_rgba_4096px", fn() {
    scalar_rgb_to_rgba(rgb, b'\xFF', rgba)
  })
  bb.bench(name="simd_rgb_to_rgba_4096px", fn() {
    SimdBufferBytes::rgb_to_rgba(rgb, b'\xFF', rgba)
  })
}

///|
/// Scalar reference channel extract (per-pixel get/set).
fn scalar_channel_extract(
  src : SimdBufferBytes,
  ch : Int,
  out : SimdBufferBytes,
) -> Unit {
  let n = src.length() / 4
  for i in 0..<n {
    out.set(i, src.get(i * 4 + ch))
  }
}

///|
test "bench/channel_extract" (bb : @bench.T) {
  let npix = 4096
  let rgba = SimdBufferBytes::make(npix * 4)
  for i in 0..<(npix * 4) {
    rgba.set(i, (i % 256).to_byte())
  }
  let out = SimdBufferBytes::make(npix)
  bb.bench(name="scalar_channel_extract_4096px", fn() {
    scalar_channel_extract(rgba, 1, out)
  })
  bb.bench(name="simd_channel_extract_4096px", fn() {
    SimdBufferBytes::channel_extract(rgba, 1, out)
  })
}

///|
/// Scalar reference channel merge (per-pixel get/set).
fn scalar_channel_merge(
  r : SimdBufferBytes,
  g : SimdBufferBytes,
  b : SimdBufferBytes,
  a : SimdBufferBytes,
  out : SimdBufferBytes,
) -> Unit {
  let n = r.length()
  for i in 0..<n {
    out.set(i * 4, r.get(i))
    out.set(i * 4 + 1, g.get(i))
    out.set(i * 4 + 2, b.get(i))
    out.set(i * 4 + 3, a.get(i))
  }
}

///|
test "bench/channel_merge" (bb : @bench.T) {
  let npix = 4096
  let r = SimdBufferBytes::make(npix)
  let g = SimdBufferBytes::make(npix)
  let b = SimdBufferBytes::make(npix)
  let a = SimdBufferBytes::make(npix)
  for i in 0..<npix {
    r.set(i, (i % 256).to_byte())
    g.set(i, ((i + 64) % 256).to_byte())
    b.set(i, ((i + 128) % 256).to_byte())
    a.set(i, 255)
  }
  let out = SimdBufferBytes::make(npix * 4)
  bb.bench(name="scalar_channel_merge_4096px", fn() {
    scalar_channel_merge(r, g, b, a, out)
  })
  bb.bench(name="simd_channel_merge_4096px", fn() {
    SimdBufferBytes::channel_merge(r, g, b, a, out)
  })
}

///|
fn scalar_lerp(
  a : SimdBufferBytes,
  b : SimdBufferBytes,
  t : Int,
  out : SimdBufferBytes,
) -> Unit {
  let inv = 256 - t
  for i in 0..<a.length() {
    out.set(
      i,
      ((a.get(i).to_int() * inv + b.get(i).to_int() * t) >> 8).to_byte(),
    )
  }
}

///|
test "bench/lerp" (bb : @bench.T) {
  let n = 4096
  let a = SimdBufferBytes::make(n)
  let b = SimdBufferBytes::make(n)
  for i in 0..<n {
    a.set(i, (i % 256).to_byte())
    b.set(i, ((255 - i % 256) % 256).to_byte())
  }
  let out = SimdBufferBytes::make(n)
  bb.bench(name="scalar_lerp_4096", fn() { scalar_lerp(a, b, 96, out) })
  bb.bench(name="simd_lerp_4096", fn() { SimdBufferBytes::lerp(a, b, 96, out) })
}

///|
fn scalar_grayscale(src : SimdBufferBytes, out : SimdBufferBytes) -> Unit {
  let n = src.length() / 4
  for i in 0..<n {
    let r = src.get(i * 4).to_int()
    let g = src.get(i * 4 + 1).to_int()
    let b = src.get(i * 4 + 2).to_int()
    out.set(i, ((77 * r + 150 * g + 29 * b) >> 8).to_byte())
  }
}

///|
test "bench/rgba_to_grayscale" (bb : @bench.T) {
  let npix = 4096
  let rgba = SimdBufferBytes::make(npix * 4)
  for i in 0..<(npix * 4) {
    rgba.set(i, (i % 256).to_byte())
  }
  let out = SimdBufferBytes::make(npix)
  bb.bench(name="scalar_grayscale_4096px", fn() { scalar_grayscale(rgba, out) })
  bb.bench(name="simd_grayscale_4096px", fn() {
    SimdBufferBytes::rgba_to_grayscale(rgba, out)
  })
}
