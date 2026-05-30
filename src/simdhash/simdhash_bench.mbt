// SHA-256 benches. `sha256_x4` is the 4-way batch path (SIMD on wasm).
// Run: `moon bench --target wasm -p simdhash`.

///|
fn bench_msg(n : Int, seed : Int) -> Bytes {
  let a = FixedArray::make(n, b'\x00')
  for i in 0..<n {
    a[i] = ((i * seed + 7) % 256).to_byte()
  }
  Bytes::from_array(a)
}

///|
test "bench/sha256 single" (bb : @bench.T) {
  let m = bench_msg(4096, 13)
  bb.bench(name="sha256_4096", fn() { bb.keep(sha256(m)) })
}

///|
test "bench/sha256 x4 vs 4x single" (bb : @bench.T) {
  let m0 = bench_msg(4096, 11)
  let m1 = bench_msg(4096, 23)
  let m2 = bench_msg(4096, 37)
  let m3 = bench_msg(4096, 53)
  bb.bench(name="sha256_4x_single_4096", fn() {
    bb.keep(sha256(m0))
    bb.keep(sha256(m1))
    bb.keep(sha256(m2))
    bb.keep(sha256(m3))
  })
  bb.bench(name="sha256_x4_4096", fn() { bb.keep(sha256_x4(m0, m1, m2, m3)) })
}

///|
test "bench/sha1 x4 vs 4x single" (bb : @bench.T) {
  let m0 = bench_msg(4096, 11)
  let m1 = bench_msg(4096, 23)
  let m2 = bench_msg(4096, 37)
  let m3 = bench_msg(4096, 53)
  bb.bench(name="sha1_4x_single_4096", fn() {
    bb.keep(sha1(m0))
    bb.keep(sha1(m1))
    bb.keep(sha1(m2))
    bb.keep(sha1(m3))
  })
  bb.bench(name="sha1_x4_4096", fn() { bb.keep(sha1_x4(m0, m1, m2, m3)) })
}
