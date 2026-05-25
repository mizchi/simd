///|
fn make_bench_int_array(size : Int) -> FixedArray[Int] {
  let arr = FixedArray::make(size, 0)
  for i in 0..<size {
    arr[i] = i + 1
  }
  arr
}

///|
fn make_bench_byte_array(size : Int) -> FixedArray[Byte] {
  let arr = FixedArray::make(size, b'\x00')
  for i in 0..<size {
    arr[i] = (i % 256 + 1).to_byte()
  }
  arr
}

///|
test "bench/sum" (b : @bench.T) {
  let arr = make_bench_int_array(1024)
  b.bench(name="scalar_sum_1024", fn() { b.keep(scalar_sum(arr)) })
  b.bench(name="simd_sum_1024", fn() { b.keep(simd_sum(arr)) })
}

///|
test "bench/dot_product" (b : @bench.T) {
  let a = make_bench_int_array(1024)
  let b_arr = make_bench_int_array(1024)
  b.bench(name="scalar_dot_product_1024", fn() {
    b.keep(scalar_dot_product(a, b_arr))
  })
  b.bench(name="simd_dot_product_1024", fn() {
    b.keep(simd_dot_product(a, b_arr))
  })
}

///|
test "bench/add" (b : @bench.T) {
  let a = make_bench_int_array(1024)
  let b_arr = make_bench_int_array(1024)
  let out = FixedArray::make(1024, 0)
  b.bench(name="scalar_add_1024", fn() { scalar_add(a, b_arr, out) })
  b.bench(name="simd_add_1024", fn() { simd_add(a, b_arr, out) })
}

///|
test "bench/adler32" (b : @bench.T) {
  let data = make_bench_byte_array(4096)
  b.bench(name="scalar_adler32_4096", fn() { b.keep(scalar_adler32(data)) })
  b.bench(name="simd_adler32_4096", fn() { b.keep(simd_adler32(data)) })
}

///|
test "bench/min_max" (b : @bench.T) {
  let arr = make_bench_int_array(1024)
  b.bench(name="scalar_min_1024", fn() { b.keep(scalar_min(arr)) })
  b.bench(name="simd_min_1024", fn() { b.keep(simd_min(arr)) })
  b.bench(name="scalar_max_1024", fn() { b.keep(scalar_max(arr)) })
  b.bench(name="simd_max_1024", fn() { b.keep(simd_max(arr)) })
}

///|
test "bench/saxpy" (b : @bench.T) {
  let a = make_bench_int_array(1024)
  let b_arr = make_bench_int_array(1024)
  let out = FixedArray::make(1024, 0)
  b.bench(name="scalar_saxpy_1024", fn() { scalar_saxpy(3, a, b_arr, out) })
  b.bench(name="simd_saxpy_1024", fn() { simd_saxpy(3, a, b_arr, out) })
}

///|
test "bench/popcount_bytes" (b : @bench.T) {
  let data = make_bench_byte_array(4096)
  b.bench(name="scalar_popcount_4096", fn() {
    b.keep(scalar_popcount_bytes(data))
  })
  b.bench(name="simd_popcount_4096", fn() {
    b.keep(simd_popcount_bytes(data))
  })
}

///|
test "bench/memcpy_bytes" (b : @bench.T) {
  let src = make_bench_byte_array(4096)
  let dst = FixedArray::make(4096, b'\x00')
  b.bench(name="scalar_memcpy_4096", fn() {
    scalar_memcpy_bytes(src, 0, dst, 0, 4096)
  })
  b.bench(name="simd_memcpy_4096", fn() {
    simd_memcpy_bytes(src, 0, dst, 0, 4096)
  })
}

///|
test "bench/memset_bytes" (b : @bench.T) {
  let dst = FixedArray::make(4096, b'\x00')
  b.bench(name="scalar_memset_4096", fn() {
    scalar_memset_bytes(dst, 0, 4096, b'\x5a')
  })
  b.bench(name="simd_memset_4096", fn() {
    simd_memset_bytes(dst, 0, 4096, b'\x5a')
  })
}

///|
test "bench/equal_bytes" (b : @bench.T) {
  let a = make_bench_byte_array(4096)
  let b_arr = make_bench_byte_array(4096)
  b.bench(name="scalar_equal_4096", fn() {
    b.keep(scalar_equal_bytes(a, b_arr, 4096))
  })
  b.bench(name="simd_equal_4096", fn() {
    b.keep(simd_equal_bytes(a, b_arr, 4096))
  })
}

///|
test "bench/find_byte" (b : @bench.T) {
  // homogeneous filler with a single needle near the end so both scalar
  // and SIMD scans walk the whole buffer.
  let data = FixedArray::make(4096, b'\x01')
  data[4090] = b'\xaa'
  b.bench(name="scalar_find_byte_4096", fn() {
    b.keep(scalar_find_byte(data, b'\xaa'))
  })
  b.bench(name="simd_find_byte_4096", fn() {
    b.keep(simd_find_byte(data, b'\xaa'))
  })
}

///|
test "bench/count_byte" (b : @bench.T) {
  let data = make_bench_byte_array(4096)
  b.bench(name="scalar_count_byte_4096", fn() {
    b.keep(scalar_count_byte(data, b'\x07'))
  })
  b.bench(name="simd_count_byte_4096", fn() {
    b.keep(simd_count_byte(data, b'\x07'))
  })
}

// --- f64 ops ---

///|
fn make_bench_f64_array(size : Int) -> FixedArray[Double] {
  let arr = FixedArray::make(size, 0.0)
  for i in 0..<size {
    arr[i] = (i + 1).to_double() * 0.5
  }
  arr
}

///|
test "bench/f64 add" (b : @bench.T) {
  let a = make_bench_f64_array(1024)
  let bb = make_bench_f64_array(1024)
  let out = FixedArray::make(1024, 0.0)
  b.bench(name="scalar_add_f64_1024", fn() { scalar_add_f64(a, bb, out) })
  b.bench(name="simd_add_f64_1024", fn() { simd_add_f64(a, bb, out) })
}

///|
test "bench/f64 mul" (b : @bench.T) {
  let a = make_bench_f64_array(1024)
  let bb = make_bench_f64_array(1024)
  let out = FixedArray::make(1024, 0.0)
  b.bench(name="scalar_mul_f64_1024", fn() { scalar_mul_f64(a, bb, out) })
  b.bench(name="simd_mul_f64_1024", fn() { simd_mul_f64(a, bb, out) })
}

///|
test "bench/f64 div" (b : @bench.T) {
  let a = make_bench_f64_array(1024)
  let bb = make_bench_f64_array(1024)
  let out = FixedArray::make(1024, 0.0)
  b.bench(name="scalar_div_f64_1024", fn() { scalar_div_f64(a, bb, out) })
  b.bench(name="simd_div_f64_1024", fn() { simd_div_f64(a, bb, out) })
}

///|
test "bench/f64 sqrt" (b : @bench.T) {
  let a = make_bench_f64_array(1024)
  let out = FixedArray::make(1024, 0.0)
  b.bench(name="scalar_sqrt_f64_1024", fn() { scalar_sqrt_f64(a, out) })
  b.bench(name="simd_sqrt_f64_1024", fn() { simd_sqrt_f64(a, out) })
}

///|
test "bench/f64 sum" (b : @bench.T) {
  let a = make_bench_f64_array(1024)
  b.bench(name="scalar_sum_f64_1024", fn() { b.keep(scalar_sum_f64(a)) })
  b.bench(name="simd_sum_f64_1024", fn() { b.keep(simd_sum_f64(a)) })
}

///|
test "bench/f64 dot" (b : @bench.T) {
  let a = make_bench_f64_array(1024)
  let bb = make_bench_f64_array(1024)
  b.bench(name="scalar_dot_f64_1024", fn() { b.keep(scalar_dot_f64(a, bb)) })
  b.bench(name="simd_dot_f64_1024", fn() { b.keep(simd_dot_f64(a, bb)) })
}
