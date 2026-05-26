// Benches for simdjson byte classification.

///|
/// Build a JSON-shaped buffer of ~`target` bytes by repeating a fixed
/// template.
fn bench_json_buf(target : Int) -> @simd_buffer.SimdBufferBytes {
  let chunk = "{\"id\":1234,\"name\":\"alice\",\"tags\":[\"a\",\"bb\",\"ccc\"],\"meta\":{\"k\":\"v\",\"n\":-1.5e+10},\"path\":\"/x/y\\\\z\\\"q\"},"
  let sb = StringBuilder::new()
  while sb.to_string().length() < target {
    sb.write_string(chunk)
  }
  let s = sb.to_string()
  let n = if s.length() > target { target } else { s.length() }
  let arr : FixedArray[Byte] = FixedArray::make(n, b'\x00')
  for i in 0..<n {
    arr[i] = s.unsafe_get(i).to_int().to_byte()
  }
  @simd_buffer.SimdBufferBytes::from_array(arr)
}

///|
test "bench/simdjson classify (4096 bytes)" (bb : @bench.T) {
  let buf = bench_json_buf(4096)
  let words = (buf.length() + 31) / 32
  let out = @simd_buffer.SimdBuffer::make(words)
  bb.bench(name="simdjson_classify_structural_4096", fn() {
    out.clear()
    classify_structural(buf, out)
  })
  bb.bench(name="simdjson_classify_numeric_4096", fn() {
    out.clear()
    classify_numeric(buf, out)
  })
  bb.bench(name="simdjson_classify_quote_raw_4096", fn() {
    out.clear()
    classify_quote_raw(buf, out)
  })
}

///|
test "bench/simdjson quote_mask + structural_indices (4096 bytes)" (
  bb : @bench.T,
) {
  let buf = bench_json_buf(4096)
  let n = buf.length()
  let words = (n + 31) / 32
  let structural = @simd_buffer.SimdBuffer::make(words)
  let quote_mask = @simd_buffer.SimdBuffer::make(words)
  let indices = @simd_buffer.SimdBuffer::make(n)
  bb.bench(name="simdjson_compute_quote_mask_4096", fn() {
    compute_quote_mask(buf, quote_mask)
  })
  structural.clear()
  classify_structural(buf, structural)
  compute_quote_mask(buf, quote_mask)
  bb.bench(name="simdjson_extract_structural_indices_4096", fn() {
    bb.keep(extract_structural_indices(structural, quote_mask, n, indices))
  })
  bb.bench(name="simdjson_find_structural_indices_with_scratch_4096", fn() {
    bb.keep(
      find_structural_indices_with_scratch(buf, structural, quote_mask, indices),
    )
  })
}
