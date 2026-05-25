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

// --- Int element-wise ---

///|
test "bench/sub_mul" (b : @bench.T) {
  let x = make_bench_int_array(1024)
  let y = make_bench_int_array(1024)
  let out = FixedArray::make(1024, 0)
  b.bench(name="scalar_sub_1024", fn() { scalar_sub(x, y, out) })
  b.bench(name="simd_sub_1024", fn() { simd_sub(x, y, out) })
  b.bench(name="scalar_mul_int_1024", fn() { scalar_mul(x, y, out) })
  b.bench(name="simd_mul_int_1024", fn() { simd_mul(x, y, out) })
}

///|
test "bench/neg_abs" (b : @bench.T) {
  let x = make_bench_int_array(1024)
  let out = FixedArray::make(1024, 0)
  b.bench(name="scalar_neg_1024", fn() { scalar_neg(x, out) })
  b.bench(name="simd_neg_1024", fn() { simd_neg(x, out) })
  b.bench(name="scalar_abs_1024", fn() { scalar_abs(x, out) })
  b.bench(name="simd_abs_1024", fn() { simd_abs(x, out) })
}

///|
test "bench/min_max_elem" (b : @bench.T) {
  let x = make_bench_int_array(1024)
  let y = make_bench_int_array(1024)
  let out = FixedArray::make(1024, 0)
  b.bench(name="scalar_min_elem_1024", fn() { scalar_min_elem(x, y, out) })
  b.bench(name="simd_min_elem_1024", fn() { simd_min_elem(x, y, out) })
  b.bench(name="scalar_max_elem_1024", fn() { scalar_max_elem(x, y, out) })
  b.bench(name="simd_max_elem_1024", fn() { simd_max_elem(x, y, out) })
}

///|
test "bench/compare_mask" (b : @bench.T) {
  let x = make_bench_int_array(1024)
  let y = make_bench_int_array(1024)
  let out = FixedArray::make(1024, 0)
  b.bench(name="scalar_eq_1024", fn() { scalar_eq(x, y, out) })
  b.bench(name="simd_eq_1024", fn() { simd_eq(x, y, out) })
  b.bench(name="scalar_lt_1024", fn() { scalar_lt(x, y, out) })
  b.bench(name="simd_lt_1024", fn() { simd_lt(x, y, out) })
}

// --- f32 over byte buffer ---

///|
fn make_bench_f32_buf(n : Int) -> FixedArray[Byte] {
  let buf = f32_buf_alloc(n)
  for i in 0..<n {
    f32_buf_set(buf, i, Float::from_int(i + 1) * (0.5 : Float))
  }
  buf
}

///|
test "bench/f32 add" (b : @bench.T) {
  let n = 1024
  let aa = make_bench_f32_buf(n)
  let bb = make_bench_f32_buf(n)
  let out = f32_buf_alloc(n)
  b.bench(name="scalar_add_f32_1024", fn() { scalar_add_f32(aa, bb, out, n) })
  b.bench(name="simd_add_f32_1024", fn() { simd_add_f32(aa, bb, out, n) })
}

///|
test "bench/f32 sum" (b : @bench.T) {
  let n = 1024
  let aa = make_bench_f32_buf(n)
  b.bench(name="scalar_sum_f32_1024", fn() { b.keep(scalar_sum_f32(aa, n)) })
  b.bench(name="simd_sum_f32_1024", fn() { b.keep(simd_sum_f32(aa, n)) })
}

///|
test "bench/f32 dot" (b : @bench.T) {
  let n = 1024
  let aa = make_bench_f32_buf(n)
  let bb = make_bench_f32_buf(n)
  b.bench(name="scalar_dot_f32_1024", fn() {
    b.keep(scalar_dot_f32(aa, bb, n))
  })
  b.bench(name="simd_dot_f32_1024", fn() {
    b.keep(simd_dot_f32(aa, bb, n))
  })
}

///|
test "bench/argmin_argmax" (b : @bench.T) {
  let arr = make_bench_int_array(1024)
  b.bench(name="scalar_argmin_1024", fn() { b.keep(scalar_argmin(arr)) })
  b.bench(name="simd_argmin_1024", fn() { b.keep(simd_argmin(arr)) })
  b.bench(name="scalar_argmax_1024", fn() { b.keep(scalar_argmax(arr)) })
  b.bench(name="simd_argmax_1024", fn() { b.keep(simd_argmax(arr)) })
}

// --- reductions ---

///|
test "bench/prod" (b : @bench.T) {
  let arr = FixedArray::make(1024, 1)
  for i in 0..<1024 {
    arr[i] = i % 3 + 1
  }
  b.bench(name="scalar_prod_1024", fn() { b.keep(scalar_prod(arr)) })
  b.bench(name="simd_prod_1024", fn() { b.keep(simd_prod(arr)) })
}

///|
test "bench/var_f64" (b : @bench.T) {
  let arr = make_bench_f64_array(1024)
  b.bench(name="scalar_var_f64_1024", fn() { b.keep(scalar_var_f64(arr)) })
  b.bench(name="simd_var_f64_1024", fn() { b.keep(simd_var_f64(arr)) })
}

///|
test "bench/count_nonzero" (b : @bench.T) {
  let arr = make_bench_int_array(1024)
  // sprinkle zeros
  for i in 0..<128 {
    arr[i * 8] = 0
  }
  b.bench(name="scalar_count_nonzero_1024", fn() {
    b.keep(scalar_count_nonzero(arr))
  })
  b.bench(name="simd_count_nonzero_1024", fn() {
    b.keep(simd_count_nonzero(arr))
  })
}

///|
test "bench/all_any" (b : @bench.T) {
  let arr = make_bench_int_array(1024) // all non-zero
  b.bench(name="scalar_all_1024", fn() { b.keep(scalar_all(arr)) })
  b.bench(name="simd_all_1024", fn() { b.keep(simd_all(arr)) })
  b.bench(name="scalar_any_1024", fn() { b.keep(scalar_any(arr)) })
  b.bench(name="simd_any_1024", fn() { b.keep(simd_any(arr)) })
}

// --- linear algebra ---

///|
test "bench/matmul_f64 64x64" (b : @bench.T) {
  let dim = 64
  let a = FixedArray::make(dim * dim, 0.0)
  let b_t = FixedArray::make(dim * dim, 0.0)
  let c = FixedArray::make(dim * dim, 0.0)
  for i in 0..<(dim * dim) {
    a[i] = (i + 1).to_double() * 0.001
    b_t[i] = ((i * 7 + 3) % 19).to_double() * 0.01
  }
  b.bench(name="scalar_matmul_f64_64x64", fn() {
    scalar_matmul_f64(a, b_t, c, dim, dim, dim)
  })
  b.bench(name="simd_matmul_f64_64x64", fn() {
    simd_matmul_f64(a, b_t, c, dim, dim, dim)
  })
}

// --- sort ---

///|
test "bench/sort4_int" (b : @bench.T) {
  let n = 1024 // 256 4-element chunks
  let arr = FixedArray::make(n, 0)
  for i in 0..<n {
    arr[i] = (i * 7 + 11) % 251
  }
  let scratch = FixedArray::make(n, 0)
  b.bench(name="scalar_sort4_x256", fn() {
    for i in 0..<n {
      scratch[i] = arr[i]
    }
    for j = 0; j < n; j = j + 4 {
      scalar_sort4_int(scratch, j)
    }
  })
  b.bench(name="simd_sort4_x256", fn() {
    for i in 0..<n {
      scratch[i] = arr[i]
    }
    for j = 0; j < n; j = j + 4 {
      simd_sort4_int(scratch, j)
    }
  })
}

///|
test "bench/transpose_f64 128x128" (b : @bench.T) {
  let m = 128
  let n = 128
  let a = FixedArray::make(m * n, 0.0)
  let out = FixedArray::make(m * n, 0.0)
  for i in 0..<(m * n) {
    a[i] = (i + 1).to_double() * 0.5
  }
  b.bench(name="scalar_transpose_128x128", fn() {
    scalar_transpose_f64(a, out, m, n)
  })
  b.bench(name="simd_transpose_128x128", fn() {
    simd_transpose_f64(a, out, m, n)
  })
}

///|
test "bench/gemv_f64 256x256" (b : @bench.T) {
  let m = 256
  let n = 256
  let a = FixedArray::make(m * n, 0.0)
  let x = FixedArray::make(n, 0.0)
  let y = FixedArray::make(m, 0.0)
  for i in 0..<(m * n) {
    a[i] = (i + 1).to_double() * 0.001
  }
  for j in 0..<n {
    x[j] = (j + 1).to_double()
  }
  b.bench(name="scalar_gemv_f64_256x256", fn() {
    scalar_gemv_f64(a, x, y, m, n)
  })
  b.bench(name="simd_gemv_f64_256x256", fn() {
    simd_gemv_f64(a, x, y, m, n)
  })
}

///|
test "bench/where" (b : @bench.T) {
  let mask = FixedArray::make(1024, -1)
  for i in 0..<1024 {
    if i % 3 == 0 {
      mask[i] = 0
    }
  }
  let x = make_bench_int_array(1024)
  let y = make_bench_int_array(1024)
  let out = FixedArray::make(1024, 0)
  b.bench(name="scalar_where_1024", fn() { scalar_where(mask, x, y, out) })
  b.bench(name="simd_where_1024", fn() { simd_where(mask, x, y, out) })
}
