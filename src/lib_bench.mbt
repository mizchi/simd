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
  b.bench(name="scalar_sum_1024", fn() { b.keep(@internal.scalar_sum(arr)) })
  b.bench(name="simd_sum_1024", fn() { b.keep(sum_i32(arr)) })
}

///|
test "bench/dot_product" (b : @bench.T) {
  let a = make_bench_int_array(1024)
  let b_arr = make_bench_int_array(1024)
  b.bench(name="scalar_dot_product_1024", fn() {
    b.keep(@internal.scalar_dot_product(a, b_arr))
  })
  b.bench(name="simd_dot_product_1024", fn() { b.keep(dot_i32(a, b_arr)) })
}

///|
test "bench/add" (b : @bench.T) {
  let a = make_bench_int_array(1024)
  let b_arr = make_bench_int_array(1024)
  let out = FixedArray::make(1024, 0)
  b.bench(name="scalar_add_1024", fn() { @internal.scalar_add(a, b_arr, out) })
  b.bench(name="simd_add_1024", fn() { add_i32(a, b_arr, out) })
}

///|
test "bench/adler32" (b : @bench.T) {
  let data = make_bench_byte_array(4096)
  b.bench(name="scalar_adler32_4096", fn() {
    b.keep(@internal.scalar_adler32(data))
  })
  b.bench(name="simd_adler32_4096", fn() { b.keep(adler32_bytes(data)) })
}

///|
test "bench/min_max" (b : @bench.T) {
  let arr = make_bench_int_array(1024)
  b.bench(name="scalar_min_1024", fn() { b.keep(@internal.scalar_min(arr)) })
  b.bench(name="simd_min_1024", fn() { b.keep(min_i32(arr)) })
  b.bench(name="scalar_max_1024", fn() { b.keep(@internal.scalar_max(arr)) })
  b.bench(name="simd_max_1024", fn() { b.keep(max_i32(arr)) })
}

///|
test "bench/saxpy" (b : @bench.T) {
  let a = make_bench_int_array(1024)
  let b_arr = make_bench_int_array(1024)
  let out = FixedArray::make(1024, 0)
  b.bench(name="scalar_saxpy_1024", fn() {
    @internal.scalar_saxpy(3, a, b_arr, out)
  })
  b.bench(name="simd_saxpy_1024", fn() { saxpy_i32(3, a, b_arr, out) })
}

///|
test "bench/popcount_bytes" (b : @bench.T) {
  let data = make_bench_byte_array(4096)
  b.bench(name="scalar_popcount_4096", fn() {
    b.keep(@internal.scalar_popcount_bytes(data))
  })
  b.bench(name="simd_popcount_4096", fn() { b.keep(popcount_bytes(data)) })
}

///|
test "bench/memcpy_bytes" (b : @bench.T) {
  let src = make_bench_byte_array(4096)
  let dst = FixedArray::make(4096, b'\x00')
  b.bench(name="scalar_memcpy_4096", fn() {
    @internal.scalar_memcpy_bytes(src, 0, dst, 0, 4096)
  })
  b.bench(name="simd_memcpy_4096", fn() { memcpy_bytes(src, 0, dst, 0, 4096) })
}

///|
test "bench/memset_bytes" (b : @bench.T) {
  let dst = FixedArray::make(4096, b'\x00')
  b.bench(name="scalar_memset_4096", fn() {
    @internal.scalar_memset_bytes(dst, 0, 4096, b'\x5a')
  })
  b.bench(name="simd_memset_4096", fn() { memset_bytes(dst, 0, 4096, b'\x5a') })
}

///|
test "bench/equal_bytes" (b : @bench.T) {
  let a = make_bench_byte_array(4096)
  let b_arr = make_bench_byte_array(4096)
  b.bench(name="scalar_equal_4096", fn() {
    b.keep(@internal.scalar_equal_bytes(a, b_arr, 4096))
  })
  b.bench(name="simd_equal_4096", fn() { b.keep(equal_bytes(a, b_arr, 4096)) })
}

///|
test "bench/find_byte" (b : @bench.T) {
  // homogeneous filler with a single needle near the end so both scalar
  // and SIMD scans walk the whole buffer.
  let data = FixedArray::make(4096, b'\x01')
  data[4090] = b'\xaa'
  b.bench(name="scalar_find_byte_4096", fn() {
    b.keep(@internal.scalar_find_byte(data, b'\xaa'))
  })
  b.bench(name="simd_find_byte_4096", fn() { b.keep(find_byte(data, b'\xaa')) })
}

///|
test "bench/count_byte" (b : @bench.T) {
  let data = make_bench_byte_array(4096)
  b.bench(name="scalar_count_byte_4096", fn() {
    b.keep(@internal.scalar_count_byte(data, b'\x07'))
  })
  b.bench(name="simd_count_byte_4096", fn() {
    b.keep(count_byte(data, b'\x07'))
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
  b.bench(name="scalar_add_f64_1024", fn() {
    @internal.scalar_add_f64(a, bb, out)
  })
  b.bench(name="simd_add_f64_1024", fn() { add_f64(a, bb, out) })
}

///|
test "bench/f64 mul" (b : @bench.T) {
  let a = make_bench_f64_array(1024)
  let bb = make_bench_f64_array(1024)
  let out = FixedArray::make(1024, 0.0)
  b.bench(name="scalar_mul_f64_1024", fn() {
    @internal.scalar_mul_f64(a, bb, out)
  })
  b.bench(name="simd_mul_f64_1024", fn() { mul_f64(a, bb, out) })
}

///|
test "bench/f64 div" (b : @bench.T) {
  let a = make_bench_f64_array(1024)
  let bb = make_bench_f64_array(1024)
  let out = FixedArray::make(1024, 0.0)
  b.bench(name="scalar_div_f64_1024", fn() {
    @internal.scalar_div_f64(a, bb, out)
  })
  b.bench(name="simd_div_f64_1024", fn() { div_f64(a, bb, out) })
}

///|
test "bench/f64 sqrt" (b : @bench.T) {
  let a = make_bench_f64_array(1024)
  let out = FixedArray::make(1024, 0.0)
  b.bench(name="scalar_sqrt_f64_1024", fn() {
    @internal.scalar_sqrt_f64(a, out)
  })
  b.bench(name="simd_sqrt_f64_1024", fn() { sqrt_f64(a, out) })
}

///|
test "bench/f64 sum" (b : @bench.T) {
  let a = make_bench_f64_array(1024)
  b.bench(name="scalar_sum_f64_1024", fn() {
    b.keep(@internal.scalar_sum_f64(a))
  })
  b.bench(name="simd_sum_f64_1024", fn() { b.keep(sum_f64(a)) })
}

///|
test "bench/f64 dot" (b : @bench.T) {
  let a = make_bench_f64_array(1024)
  let bb = make_bench_f64_array(1024)
  b.bench(name="scalar_dot_f64_1024", fn() {
    b.keep(@internal.scalar_dot_f64(a, bb))
  })
  b.bench(name="simd_dot_f64_1024", fn() { b.keep(dot_f64(a, bb)) })
}

// --- Int element-wise ---

///|
test "bench/sub_mul" (b : @bench.T) {
  let x = make_bench_int_array(1024)
  let y = make_bench_int_array(1024)
  let out = FixedArray::make(1024, 0)
  b.bench(name="scalar_sub_1024", fn() { @internal.scalar_sub(x, y, out) })
  b.bench(name="simd_sub_1024", fn() { sub_i32(x, y, out) })
  b.bench(name="scalar_mul_int_1024", fn() { @internal.scalar_mul(x, y, out) })
  b.bench(name="simd_mul_int_1024", fn() { mul_i32(x, y, out) })
}

///|
test "bench/neg_abs" (b : @bench.T) {
  let x = make_bench_int_array(1024)
  let out = FixedArray::make(1024, 0)
  b.bench(name="scalar_neg_1024", fn() { @internal.scalar_neg(x, out) })
  b.bench(name="simd_neg_1024", fn() { neg_i32(x, out) })
  b.bench(name="scalar_abs_1024", fn() { @internal.scalar_abs(x, out) })
  b.bench(name="simd_abs_1024", fn() { abs_i32(x, out) })
}

///|
test "bench/min_max_elem" (b : @bench.T) {
  let x = make_bench_int_array(1024)
  let y = make_bench_int_array(1024)
  let out = FixedArray::make(1024, 0)
  b.bench(name="scalar_min_elem_1024", fn() {
    @internal.scalar_min_elem(x, y, out)
  })
  b.bench(name="simd_min_elem_1024", fn() { min_elem_i32(x, y, out) })
  b.bench(name="scalar_max_elem_1024", fn() {
    @internal.scalar_max_elem(x, y, out)
  })
  b.bench(name="simd_max_elem_1024", fn() { max_elem_i32(x, y, out) })
}

///|
test "bench/compare_mask" (b : @bench.T) {
  let x = make_bench_int_array(1024)
  let y = make_bench_int_array(1024)
  let out = FixedArray::make(1024, 0)
  b.bench(name="scalar_eq_1024", fn() { @internal.scalar_eq(x, y, out) })
  b.bench(name="simd_eq_1024", fn() { eq_i32(x, y, out) })
  b.bench(name="scalar_lt_1024", fn() { @internal.scalar_lt(x, y, out) })
  b.bench(name="simd_lt_1024", fn() { lt_i32(x, y, out) })
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
  b.bench(name="scalar_add_f32_1024", fn() {
    @internal.scalar_add_f32(aa, bb, out, n)
  })
  b.bench(name="simd_add_f32_1024", fn() { add_f32(aa, bb, out, n) })
}

///|
test "bench/f32 sum" (b : @bench.T) {
  let n = 1024
  let aa = make_bench_f32_buf(n)
  b.bench(name="scalar_sum_f32_1024", fn() {
    b.keep(@internal.scalar_sum_f32(aa, n))
  })
  b.bench(name="simd_sum_f32_1024", fn() { b.keep(sum_f32(aa, n)) })
}

///|
test "bench/f32 dot" (b : @bench.T) {
  let n = 1024
  let aa = make_bench_f32_buf(n)
  let bb = make_bench_f32_buf(n)
  b.bench(name="scalar_dot_f32_1024", fn() {
    b.keep(@internal.scalar_dot_f32(aa, bb, n))
  })
  b.bench(name="simd_dot_f32_1024", fn() { b.keep(dot_f32(aa, bb, n)) })
}

///|
test "bench/argmin_argmax" (b : @bench.T) {
  let arr = make_bench_int_array(1024)
  b.bench(name="scalar_argmin_1024", fn() {
    b.keep(@internal.scalar_argmin(arr))
  })
  b.bench(name="simd_argmin_1024", fn() { b.keep(argmin_i32(arr)) })
  b.bench(name="scalar_argmax_1024", fn() {
    b.keep(@internal.scalar_argmax(arr))
  })
  b.bench(name="simd_argmax_1024", fn() { b.keep(argmax_i32(arr)) })
}

// --- reductions ---

///|
test "bench/prod" (b : @bench.T) {
  let arr = FixedArray::make(1024, 1)
  for i in 0..<1024 {
    arr[i] = i % 3 + 1
  }
  b.bench(name="scalar_prod_1024", fn() { b.keep(@internal.scalar_prod(arr)) })
  b.bench(name="simd_prod_1024", fn() { b.keep(prod_i32(arr)) })
}

///|
test "bench/var_f64" (b : @bench.T) {
  let arr = make_bench_f64_array(1024)
  b.bench(name="scalar_var_f64_1024", fn() {
    b.keep(@internal.scalar_var_f64(arr))
  })
  b.bench(name="simd_var_f64_1024", fn() { b.keep(var_f64(arr)) })
}

///|
test "bench/count_nonzero" (b : @bench.T) {
  let arr = make_bench_int_array(1024)
  // sprinkle zeros
  for i in 0..<128 {
    arr[i * 8] = 0
  }
  b.bench(name="scalar_count_nonzero_1024", fn() {
    b.keep(@internal.scalar_count_nonzero(arr))
  })
  b.bench(name="simd_count_nonzero_1024", fn() {
    b.keep(count_nonzero_i32(arr))
  })
}

///|
test "bench/all_any" (b : @bench.T) {
  let arr = make_bench_int_array(1024) // all non-zero
  b.bench(name="scalar_all_1024", fn() { b.keep(@internal.scalar_all(arr)) })
  b.bench(name="simd_all_1024", fn() { b.keep(all_i32(arr)) })
  b.bench(name="scalar_any_1024", fn() { b.keep(@internal.scalar_any(arr)) })
  b.bench(name="simd_any_1024", fn() { b.keep(any_i32(arr)) })
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
    @internal.scalar_matmul_f64(a, b_t, c, dim, dim, dim)
  })
  b.bench(name="simd_matmul_f64_64x64", fn() {
    matmul_f64(a, b_t, c, dim, dim, dim)
  })
}

///|
test "bench/div_int" (b : @bench.T) {
  let aa = make_bench_int_array(1024)
  let bb = FixedArray::make(1024, 0)
  for i in 0..<1024 {
    bb[i] = i % 7 + 1
  }
  let out = FixedArray::make(1024, 0)
  b.bench(name="scalar_div_int_1024", fn() { @internal.scalar_div(aa, bb, out) })
  b.bench(name="simd_div_int_1024", fn() { div_i32(aa, bb, out) })
}

///|
test "bench/cumsum_cumprod" (b : @bench.T) {
  let arr = make_bench_int_array(1024)
  let out = FixedArray::make(1024, 0)
  b.bench(name="scalar_cumsum_1024", fn() { @internal.scalar_cumsum(arr, out) })
  b.bench(name="simd_cumsum_1024", fn() { cumsum_i32(arr, out) })
  // small values for cumprod to avoid overflow noise
  let small = FixedArray::make(1024, 1)
  for i in 0..<1024 {
    small[i] = i % 3 + 1
  }
  b.bench(name="scalar_cumprod_1024", fn() {
    @internal.scalar_cumprod(small, out)
  })
  b.bench(name="simd_cumprod_1024", fn() { cumprod_i32(small, out) })
}

// --- UTF-8 validation ---

///|
test "bench/validate_utf8 ASCII" (b : @bench.T) {
  let n = 4096
  let data = FixedArray::make(n, b'\x00')
  for i in 0..<n {
    data[i] = (65 + i % 26).to_byte()
  }
  b.bench(name="scalar_utf8_ascii_4096", fn() {
    b.keep(@internal.scalar_validate_utf8(data))
  })
  b.bench(name="simd_utf8_ascii_4096", fn() { b.keep(validate_utf8(data)) })
}

///|
test "bench/validate_utf8 mixed" (b : @bench.T) {
  // 90% ASCII, 10% 2-byte UTF-8 (0xC3 0xA9 = é)
  let n = 4000
  let data = FixedArray::make(n, b'\x00')
  let mut i = 0
  while i < n - 1 {
    if i % 20 == 0 {
      data[i] = b'\xc3'
      data[i + 1] = b'\xa9'
      i = i + 2
    } else {
      data[i] = (65 + i % 26).to_byte()
      i = i + 1
    }
  }
  b.bench(name="scalar_utf8_mixed_4000", fn() {
    b.keep(@internal.scalar_validate_utf8(data))
  })
  b.bench(name="simd_utf8_mixed_4000", fn() { b.keep(validate_utf8(data)) })
}

// --- string ops ---

///|
test "bench/is_ascii" (b : @bench.T) {
  let n = 4096
  let data = FixedArray::make(n, b'\x00')
  for i in 0..<n {
    data[i] = (65 + i % 26).to_byte()
  }
  b.bench(name="scalar_is_ascii_4096", fn() {
    b.keep(@internal.scalar_is_ascii(data))
  })
  b.bench(name="simd_is_ascii_4096", fn() { b.keep(is_ascii(data)) })
}

///|
test "bench/to_lower_ascii" (b : @bench.T) {
  let n = 4096
  let data = FixedArray::make(n, b'\x00')
  let scratch = FixedArray::make(n, b'\x00')
  for i in 0..<n {
    data[i] = (65 + i % 26).to_byte()
  }
  b.bench(name="scalar_to_lower_4096", fn() {
    for i in 0..<n {
      scratch[i] = data[i]
    }
    @internal.scalar_to_lower_ascii(scratch)
  })
  b.bench(name="simd_to_lower_4096", fn() {
    for i in 0..<n {
      scratch[i] = data[i]
    }
    to_lower_ascii(scratch)
  })
}

// --- sort ---

///|
test "bench/sort_i32 (full)" (b : @bench.T) {
  let n = 1024
  let arr = FixedArray::make(n, 0)
  for i in 0..<n {
    arr[i] = (i * 17 + 11) % 1009
  }
  let scratch = FixedArray::make(n, 0)
  b.bench(name="scalar_sort_int_1024", fn() {
    for i in 0..<n {
      scratch[i] = arr[i]
    }
    @internal.scalar_sort_int(scratch)
  })
  b.bench(name="simd_sort_i32_1024", fn() {
    for i in 0..<n {
      scratch[i] = arr[i]
    }
    sort_i32(scratch)
  })
}

///|
test "bench/sort16_int" (b : @bench.T) {
  let n = 1024 // 64 blocks of 16
  let arr = FixedArray::make(n, 0)
  for i in 0..<n {
    arr[i] = (i * 17 + 11) % 251
  }
  let scratch = FixedArray::make(n, 0)
  b.bench(name="scalar_sort16_x64", fn() {
    for i in 0..<n {
      scratch[i] = arr[i]
    }
    for j = 0; j < n; j = j + 16 {
      @internal.scalar_sort16_int(scratch, j)
    }
  })
  b.bench(name="simd_sort16_x64", fn() {
    for i in 0..<n {
      scratch[i] = arr[i]
    }
    for j = 0; j < n; j = j + 16 {
      sort16_i32(scratch, j)
    }
  })
}

///|
test "bench/gather_scatter" (b : @bench.T) {
  let n = 1024
  let arr = make_bench_int_array(n)
  let indices = FixedArray::make(n, 0)
  for i in 0..<n {
    indices[i] = i * 31 % n
  }
  let out = FixedArray::make(n, 0)
  b.bench(name="scalar_gather_1024", fn() {
    @internal.scalar_gather(arr, indices, out)
  })
  b.bench(name="simd_gather_1024", fn() { gather_i32(arr, indices, out) })
  let target = FixedArray::make(n, 0)
  b.bench(name="scalar_scatter_1024", fn() {
    @internal.scalar_scatter(target, indices, arr)
  })
  b.bench(name="simd_scatter_1024", fn() { scatter_i32(target, indices, arr) })
}

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
      @internal.scalar_sort4_int(scratch, j)
    }
  })
  b.bench(name="simd_sort4_x256", fn() {
    for i in 0..<n {
      scratch[i] = arr[i]
    }
    for j = 0; j < n; j = j + 4 {
      sort4_i32(scratch, j)
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
    @internal.scalar_transpose_f64(a, out, m, n)
  })
  b.bench(name="simd_transpose_128x128", fn() { transpose_f64(a, out, m, n) })
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
    @internal.scalar_gemv_f64(a, x, y, m, n)
  })
  b.bench(name="simd_gemv_f64_256x256", fn() { gemv_f64(a, x, y, m, n) })
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
  b.bench(name="scalar_where_1024", fn() {
    @internal.scalar_where(mask, x, y, out)
  })
  b.bench(name="simd_where_1024", fn() { where_i32(mask, x, y, out) })
}
