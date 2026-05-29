// Benchmarks comparing the core idiom (scalar) against the core_simd
// substitute. Run with `moon bench --target wasm -p core_simd` to see the
// real SIMD speedup; on wasm-gc / js both paths are scalar (parity expected).

///|
fn bench_int_array(size : Int) -> FixedArray[Int] {
  let arr = FixedArray::make(size, 0)
  for i in 0..<size {
    arr[i] = i * 48271 % 100000
  }
  arr
}

///|
fn bench_bytes(size : Int) -> Bytes {
  let arr = FixedArray::make(size, b'\x00')
  for i in 0..<size {
    arr[i] = (i % 95 + 32).to_byte()
  }
  Bytes::from_fixedarray(arr)
}

///|
test "bench/sum" (b : @bench.T) {
  let a = bench_int_array(1024)
  b.bench(name="core_fold_sum", fn() {
    b.keep(a.iter().fold(init=0, fn(x, y) { x + y }))
  })
  b.bench(name="core_simd_sum", fn() { b.keep(sum(a)) })
}

///|
test "bench/maximum" (b : @bench.T) {
  let a = bench_int_array(1024)
  b.bench(name="core_iter_maximum", fn() { b.keep(a.iter().maximum()) })
  b.bench(name="core_simd_maximum", fn() { b.keep(maximum(a)) })
}

///|
test "bench/search" (b : @bench.T) {
  let a = bench_int_array(1024)
  // worst case: needle absent, both paths scan the whole array.
  b.bench(name="core_iter_search", fn() {
    let mut idx = -1
    for i in 0..<a.length() {
      if a[i] == -1 {
        idx = i
        break
      }
    }
    b.keep(idx)
  })
  b.bench(name="core_simd_search", fn() { b.keep(search(a, -1)) })
}

///|
test "bench/fill" (b : @bench.T) {
  let a = FixedArray::make(1024, 0)
  b.bench(name="core_fill", fn() { a.fill(7) })
  b.bench(name="core_simd_fill", fn() { fill(a, 7) })
}

///|
test "bench/sort" (b : @bench.T) {
  let src = bench_int_array(1024)
  b.bench(name="core_sort", fn() {
    let a = src.copy()
    a.sort()
    b.keep(a[0])
  })
  b.bench(name="core_simd_sort", fn() {
    let a = src.copy()
    sort(a)
    b.keep(a[0])
  })
}

///|
test "bench/bytes_count" (b : @bench.T) {
  let data = bench_bytes(4096)
  b.bench(name="core_count_loop", fn() {
    let mut c = 0
    for i in 0..<data.length() {
      if data[i] == b'A' {
        c = c + 1
      }
    }
    b.keep(c)
  })
  b.bench(name="core_simd_bytes_count", fn() { b.keep(bytes_count(data, b'A')) })
}

///|
test "bench/bytes_equal" (b : @bench.T) {
  let x = bench_bytes(4096)
  let y = bench_bytes(4096)
  b.bench(name="core_bytes_eq", fn() { b.keep(x == y) })
  b.bench(name="core_simd_bytes_equal", fn() { b.keep(bytes_equal(x, y)) })
}

///|
test "bench/bytes_search" (b : @bench.T) {
  let data = bench_bytes(4096)
  // needle absent: both paths scan the whole buffer (worst case).
  b.bench(name="core_search_loop", fn() {
    let mut idx = -1
    for i in 0..<data.length() {
      if data[i] == b'\x00' {
        idx = i
        break
      }
    }
    b.keep(idx)
  })
  b.bench(name="core_simd_bytes_search", fn() {
    b.keep(bytes_search(data, b'\x00'))
  })
}

///|
test "bench/bytes_is_ascii" (b : @bench.T) {
  let data = bench_bytes(4096)
  b.bench(name="core_is_ascii_loop", fn() {
    let mut ok = true
    for i in 0..<data.length() {
      if (data[i].to_int() & 0x80) != 0 {
        ok = false
        break
      }
    }
    b.keep(ok)
  })
  b.bench(name="core_simd_bytes_is_ascii", fn() { b.keep(bytes_is_ascii(data)) })
}
