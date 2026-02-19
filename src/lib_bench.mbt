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
