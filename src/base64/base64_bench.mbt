///|
fn make_bench_byte_array(size : Int) -> FixedArray[Byte] {
  let arr = FixedArray::make(size, b'\x00')
  for i in 0..<size {
    arr[i] = (i % 256).to_byte()
  }
  arr
}

///|
test "bench/encode" (b : @bench.T) {
  let data = make_bench_byte_array(4096)
  let out = FixedArray::make(encoded_len(4096), b'\x00')
  b.bench(name="scalar_encode_4096", fn() { scalar_encode(data, out) })
  b.bench(name="encode_4096", fn() { b.keep(encode(data)) })
}

///|
test "bench/decode" (b : @bench.T) {
  let data = make_bench_byte_array(4095) // not divisible by 3 → exercises padding
  let encoded = encode(data)
  let out = FixedArray::make(
    decoded_len(encoded.length(), count_padding(encoded)),
    b'\x00',
  )
  b.bench(name="scalar_decode_5460", fn() {
    b.keep(scalar_decode(encoded, out))
  })
  b.bench(name="decode_5460", fn() { b.keep(decode(encoded)) })
}
