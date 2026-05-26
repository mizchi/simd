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
/// Build a JSON-shaped buffer of ~`target` bytes by repeating a fixed
/// template. Used by the JSON byte-classification benches below.
fn bench_json_buf(target : Int) -> SimdBufferBytes {
  let chunk = "{\"id\":1234,\"name\":\"alice\",\"tags\":[\"a\",\"bb\",\"ccc\"],\"meta\":{\"k\":\"v\",\"n\":-1.5e+10},\"path\":\"/x/y\\\\z\\\"q\"},"
  let buf = StringBuilder::new()
  while buf.to_string().length() < target {
    buf.write_string(chunk)
  }
  let s = buf.to_string()
  let n = if s.length() > target { target } else { s.length() }
  let arr : FixedArray[Byte] = FixedArray::make(n, b'\x00')
  for i in 0..<n {
    arr[i] = s.unsafe_get(i).to_byte()
  }
  SimdBufferBytes::from_array(arr)
}

///|
test "bench/json classify (4096 bytes)" (bb : @bench.T) {
  let buf = bench_json_buf(4096)
  let words = (buf.length() + 31) / 32
  let out = SimdBuffer::make(words)
  bb.bench(name="simd_classify_json_structural_4096", fn() {
    out.clear()
    buf.classify_json_structural(out)
  })
  bb.bench(name="simd_classify_json_numeric_4096", fn() {
    out.clear()
    buf.classify_json_numeric(out)
  })
  bb.bench(name="simd_classify_json_quote_raw_4096", fn() {
    out.clear()
    buf.classify_json_quote_raw(out)
  })
}

///|
test "bench/json quote_mask + structural_indices (4096 bytes)" (bb : @bench.T) {
  let buf = bench_json_buf(4096)
  let n = buf.length()
  let words = (n + 31) / 32
  let structural = SimdBuffer::make(words)
  let quote_mask = SimdBuffer::make(words)
  let indices = SimdBuffer::make(n)
  bb.bench(name="simd_compute_quote_mask_4096", fn() {
    buf.compute_quote_mask(quote_mask)
  })
  // For the indices bench, prebuild the inputs so we only measure extract.
  structural.clear()
  buf.classify_json_structural(structural)
  buf.compute_quote_mask(quote_mask)
  bb.bench(name="simd_extract_structural_indices_4096", fn() {
    bb.keep(json_extract_structural_indices(structural, quote_mask, n, indices))
  })
  bb.bench(name="simd_find_json_structural_indices_with_scratch_4096", fn() {
    bb.keep(
      buf.find_json_structural_indices_with_scratch(
        structural, quote_mask, indices,
      ),
    )
  })
}
