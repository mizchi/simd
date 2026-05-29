// Benchmarks comparing the core idiom (scalar) against the simdcore
// substitute. Run with `moon bench --target wasm -p simdcore` to see the
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
fn bench_f64_array(size : Int) -> FixedArray[Double] {
  let arr = FixedArray::make(size, 0.0)
  for i in 0..<size {
    arr[i] = (i % 1000 + 1).to_double()
  }
  arr
}

///|
test "bench/sum" (b : @bench.T) {
  let a = bench_int_array(1024)
  b.bench(name="core_fold_sum", fn() {
    b.keep(a.iter().fold(init=0, fn(x, y) { x + y }))
  })
  b.bench(name="simdcore_sum", fn() { b.keep(sum(a)) })
}

///|
test "bench/maximum" (b : @bench.T) {
  let a = bench_int_array(1024)
  b.bench(name="core_iter_maximum", fn() { b.keep(a.iter().maximum()) })
  b.bench(name="simdcore_maximum", fn() { b.keep(maximum(a)) })
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
  b.bench(name="simdcore_search", fn() { b.keep(search(a, -1)) })
}

///|
test "bench/fill" (b : @bench.T) {
  let a = FixedArray::make(1024, 0)
  b.bench(name="core_fill", fn() { a.fill(7) })
  b.bench(name="simdcore_fill", fn() { fill(a, 7) })
}

///|
test "bench/sort" (b : @bench.T) {
  let src = bench_int_array(1024)
  b.bench(name="core_sort", fn() {
    let a = src.copy()
    a.sort()
    b.keep(a[0])
  })
  b.bench(name="simdcore_sort", fn() {
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
  b.bench(name="simdcore_bytes_count", fn() { b.keep(bytes_count(data, b'A')) })
}

///|
test "bench/bytes_equal" (b : @bench.T) {
  let x = bench_bytes(4096)
  let y = bench_bytes(4096)
  b.bench(name="core_bytes_eq", fn() { b.keep(x == y) })
  b.bench(name="simdcore_bytes_equal", fn() { b.keep(bytes_equal(x, y)) })
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
  b.bench(name="simdcore_bytes_search", fn() {
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
  b.bench(name="simdcore_bytes_is_ascii", fn() { b.keep(bytes_is_ascii(data)) })
}

///|
test "bench/int_add" (b : @bench.T) {
  let a = bench_int_array(1024)
  let c = bench_int_array(1024)
  let out = FixedArray::make(1024, 0)
  b.bench(name="core_add_loop", fn() {
    for i in 0..<a.length() {
      out[i] = a[i] + c[i]
    }
  })
  b.bench(name="simdcore_add", fn() { add(a, c, out) })
}

///|
test "bench/saxpy" (b : @bench.T) {
  let a = bench_int_array(1024)
  let c = bench_int_array(1024)
  let out = FixedArray::make(1024, 0)
  b.bench(name="core_saxpy_loop", fn() {
    for i in 0..<a.length() {
      out[i] = 3 * a[i] + c[i]
    }
  })
  b.bench(name="simdcore_saxpy", fn() { saxpy(3, a, c, out) })
}

///|
test "bench/f64_sum" (b : @bench.T) {
  let a = bench_f64_array(1024)
  b.bench(name="core_f64_sum_loop", fn() {
    let mut s = 0.0
    for i in 0..<a.length() {
      s = s + a[i]
    }
    b.keep(s)
  })
  b.bench(name="simdcore_sum_f64", fn() { b.keep(sum_f64(a)) })
}

///|
test "bench/f64_add" (b : @bench.T) {
  let a = bench_f64_array(1024)
  let c = bench_f64_array(1024)
  let out = FixedArray::make(1024, 0.0)
  b.bench(name="core_f64_add_loop", fn() {
    for i in 0..<a.length() {
      out[i] = a[i] + c[i]
    }
  })
  b.bench(name="simdcore_add_f64", fn() { add_f64(a, c, out) })
}

///|
test "bench/f64_dot" (b : @bench.T) {
  let a = bench_f64_array(1024)
  let c = bench_f64_array(1024)
  b.bench(name="core_f64_dot_loop", fn() {
    let mut s = 0.0
    for i in 0..<a.length() {
      s = s + a[i] * c[i]
    }
    b.keep(s)
  })
  b.bench(name="simdcore_dot_f64", fn() { b.keep(dot_f64(a, c)) })
}

///|
fn bench_ascii_string(n : Int) -> String {
  let arr = Array::new()
  for i in 0..<n {
    arr.push(Char::from_int(0x41 + i % 26))
  }
  String::from_array(arr)
}

///|
test "bench/utf8_encode" (b : @bench.T) {
  let s = bench_ascii_string(4096)
  b.bench(name="core_utf8_encode", fn() { b.keep(@utf8.encode(s[:])) })
  b.bench(name="simdcore_encode_utf8", fn() { b.keep(encode_utf8(s)) })
}

///|
test "bench/utf8_encode_into" (b : @bench.T) {
  let s = bench_ascii_string(4096)
  let out = FixedArray::make(4096, b'\x00')
  b.bench(name="core_utf8_encode_alloc", fn() { b.keep(@utf8.encode(s[:])) })
  b.bench(name="simdcore_encode_utf8_into", fn() {
    b.keep(encode_utf8_into(s, out))
  })
}

///|
test "bench/is_ascii_string" (b : @bench.T) {
  let s = bench_ascii_string(4096)
  b.bench(name="simdcore_is_ascii_string", fn() { b.keep(is_ascii_string(s)) })
}

///|
fn bench_ascii_bytes(n : Int) -> Bytes {
  let a = FixedArray::make(n, b'\x00')
  for i in 0..<n {
    a[i] = (0x41 + i % 26).to_byte()
  }
  Bytes::from_fixedarray(a)
}

///|
test "bench/utf8_decode" (b : @bench.T) {
  let bytes = bench_ascii_bytes(4096)
  b.bench(name="core_utf8_decode", fn() { b.keep(@utf8.decode_lossy(bytes[:])) })
  b.bench(name="simdcore_decode_unsafe", fn() {
    b.keep(decode_utf8_unsafe(bytes))
  })
  b.bench(name="simdcore_decode_unsafe_intrinsic", fn() {
    b.keep(decode_utf8_unsafe_intrinsic(bytes))
  })
}

///|
fn bench_int_arr(n : Int) -> Array[Int] {
  let a = Array::new(capacity=n)
  for i in 0..<n {
    a.push(i * 48271 % 100000)
  }
  a
}

///|
test "bench/array_sum" (b : @bench.T) {
  let a = bench_int_arr(1024)
  b.bench(name="core_array_iter_fold", fn() {
    b.keep(a.iter().fold(init=0, fn(x, y) { x + y }))
  })
  b.bench(name="simdcore_array_sum", fn() { b.keep(array_sum(a)) })
}

///|
test "bench/array_sort" (b : @bench.T) {
  let src = bench_int_arr(1024)
  b.bench(name="core_array_sort", fn() {
    let c = src.copy()
    c.sort()
    b.keep(c[0])
  })
  b.bench(name="simdcore_array_sort", fn() {
    let c = src.copy()
    array_sort(c)
    b.keep(c[0])
  })
}

///|
fn bench_json_bytes() -> Bytes {
  // ~4 KiB of realistic JSON: array of objects.
  let sb = StringBuilder::new()
  sb.write_string("[")
  for i in 0..<120 {
    if i > 0 {
      sb.write_string(",")
    }
    sb.write_string("{\"id\":")
    sb.write_string(i.to_string())
    sb.write_string(",\"name\":\"item ")
    sb.write_string(i.to_string())
    sb.write_string("\",\"tags\":[\"a\",\"b\",\"c\"],\"ok\":true}")
  }
  sb.write_string("]")
  @utf8.encode(sb.to_string()[:])
}

///|
test "bench/json" (b : @bench.T) {
  let bytes = bench_json_bytes()
  let n = bytes.length()
  let words = (n + 31) / 32
  let structural = FixedArray::make(words, 0)
  let quote = FixedArray::make(words, 0)
  let indices = FixedArray::make(n, 0)
  // classify-only primitive (the genuinely fast part)
  let cls = FixedArray::make(words, 0)
  b.bench(name="json_classify_scalar", fn() {
    for w in 0..<words {
      cls[w] = 0
    }
    for i in 0..<n {
      let c = bytes[i].to_int()
      if c == 0x7B ||
        c == 0x7D ||
        c == 0x5B ||
        c == 0x5D ||
        c == 0x2C ||
        c == 0x3A {
        cls[i >> 5] = cls[i >> 5] | (1 << (i & 31))
      }
    }
    b.keep(cls[0])
  })
  b.bench(name="json_classify_simdcore", fn() {
    json_classify_structural(bytes, cls)
    b.keep(cls[0])
  })
  // full pipeline (quote-mask bound)
  b.bench(name="json_indices_simdcore_into", fn() {
    b.keep(json_structural_indices_into(bytes, structural, quote, indices))
  })
}

///|
/// MoonBit scalar reference full pipeline (for a fair same-run A/B vs the
/// per-target simdcore path — C FFI on native, inline-WAT on wasm).
fn json_indices_scalar_ref(
  src : Bytes,
  structural : FixedArray[Int],
  quote : FixedArray[Int],
  indices : FixedArray[Int],
) -> Int {
  let n = src.length()
  let words = (n + 31) / 32
  for w in 0..<words {
    structural[w] = 0
    quote[w] = 0
  }
  for i in 0..<n {
    let b = src[i].to_int()
    if b == 0x7B ||
      b == 0x7D ||
      b == 0x5B ||
      b == 0x5D ||
      b == 0x2C ||
      b == 0x3A {
      structural[i >> 5] = structural[i >> 5] | (1 << (i & 31))
    }
  }
  let mut in_string = false
  let mut escape_pending = false
  for i in 0..<n {
    let b = src[i].to_int()
    let mut bit_set = false
    if escape_pending {
      escape_pending = false
      bit_set = in_string
    } else if b == 0x22 {
      in_string = not(in_string)
    } else if b == 0x5C && in_string {
      escape_pending = true
      bit_set = true
    } else {
      bit_set = in_string
    }
    if bit_set {
      quote[i >> 5] = quote[i >> 5] | (1 << (i & 31))
    }
  }
  let mut count = 0
  for w in 0..<words {
    let mut word = structural[w] & quote[w].lnot()
    while word != 0 {
      let mut x = word
      let mut bit = 0
      while (x & 1) == 0 {
        x = x >> 1
        bit = bit + 1
      }
      indices[count] = w * 32 + bit
      count = count + 1
      word = word & (word - 1)
    }
  }
  count
}

///|
test "bench/json_pipeline_ab" (b : @bench.T) {
  let bytes = bench_json_bytes()
  let n = bytes.length()
  let words = (n + 31) / 32
  let s1 = FixedArray::make(words, 0)
  let q1 = FixedArray::make(words, 0)
  let i1 = FixedArray::make(n, 0)
  b.bench(name="json_pipeline_scalar_ref", fn() {
    b.keep(json_indices_scalar_ref(bytes, s1, q1, i1))
  })
  b.bench(name="json_pipeline_simdcore", fn() {
    b.keep(json_structural_indices_into(bytes, s1, q1, i1))
  })
}
