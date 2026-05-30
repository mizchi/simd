///|
fn make_bench_byte_array(size : Int) -> FixedArray[Byte] {
  let arr = FixedArray::make(size, b'\x00')
  for i in 0..<size {
    arr[i] = (i % 256).to_byte()
  }
  arr
}

// Local scalar reference baselines (the package's own scalar base64_encode/base64_decode are
// package-private; these self-contained copies give the SIMD-vs-scalar bench
// its comparison point without exposing internals).

///|
fn ref_b64_char(v : Int) -> Byte {
  if v < 26 {
    (v + 65).to_byte()
  } else if v < 52 {
    (v - 26 + 97).to_byte()
  } else if v < 62 {
    (v - 52 + 48).to_byte()
  } else if v == 62 {
    b'+'
  } else {
    b'/'
  }
}

///|
fn ref_b64_val(c : Int) -> Int {
  if c >= 65 && c <= 90 {
    c - 65
  } else if c >= 97 && c <= 122 {
    c - 97 + 26
  } else if c >= 48 && c <= 57 {
    c - 48 + 52
  } else if c == 43 {
    62
  } else if c == 47 {
    63
  } else {
    -1
  }
}

///|
fn ref_encode(input : FixedArray[Byte], out : FixedArray[Byte]) -> Unit {
  let n = input.length()
  let triples = n / 3
  for i in 0..<triples {
    let p = i * 3
    let q = i * 4
    let v = (input[p].to_int() << 16) |
      (input[p + 1].to_int() << 8) |
      input[p + 2].to_int()
    out[q] = ref_b64_char((v >> 18) & 63)
    out[q + 1] = ref_b64_char((v >> 12) & 63)
    out[q + 2] = ref_b64_char((v >> 6) & 63)
    out[q + 3] = ref_b64_char(v & 63)
  }
  let rem = n - triples * 3
  let q = triples * 4
  if rem == 1 {
    let v = input[triples * 3].to_int() << 16
    out[q] = ref_b64_char((v >> 18) & 63)
    out[q + 1] = ref_b64_char((v >> 12) & 63)
    out[q + 2] = b'='
    out[q + 3] = b'='
  } else if rem == 2 {
    let v = (input[triples * 3].to_int() << 16) |
      (input[triples * 3 + 1].to_int() << 8)
    out[q] = ref_b64_char((v >> 18) & 63)
    out[q + 1] = ref_b64_char((v >> 12) & 63)
    out[q + 2] = ref_b64_char((v >> 6) & 63)
    out[q + 3] = b'='
  }
}

///|
fn ref_decode(input : FixedArray[Byte], out : FixedArray[Byte]) -> Int? {
  let n = input.length()
  if n == 0 {
    return Some(0)
  }
  if n % 4 != 0 {
    return None
  }
  let pad = if input[n - 1] == b'=' {
    if input[n - 2] == b'=' {
      2
    } else {
      1
    }
  } else {
    0
  }
  let quads = n / 4
  let mut w = 0
  for i in 0..<quads {
    let p = i * 4
    let last = i == quads - 1
    let d0 = ref_b64_val(input[p].to_int())
    let d1 = ref_b64_val(input[p + 1].to_int())
    if d0 < 0 || d1 < 0 {
      return None
    }
    let c2 = input[p + 2]
    let c3 = input[p + 3]
    let d2 = if last && pad >= 1 && c2 == b'=' {
      0
    } else {
      ref_b64_val(c2.to_int())
    }
    let d3 = if last && pad >= 1 && c3 == b'=' {
      0
    } else {
      ref_b64_val(c3.to_int())
    }
    if d2 < 0 || d3 < 0 {
      return None
    }
    let v = (d0 << 18) | (d1 << 12) | (d2 << 6) | d3
    out[w] = ((v >> 16) & 0xFF).to_byte()
    w += 1
    if !(last && pad == 2) {
      out[w] = ((v >> 8) & 0xFF).to_byte()
      w += 1
    }
    if !(last && pad >= 1) {
      out[w] = (v & 0xFF).to_byte()
      w += 1
    }
  }
  Some(w)
}

///|
test "bench/base64_encode" (b : @bench.T) {
  let data = make_bench_byte_array(4096)
  let out = FixedArray::make(base64_encoded_len(4096), b'\x00')
  b.bench(name="scalar_encode_4096", fn() { ref_encode(data, out) })
  b.bench(name="encode_4096", fn() { b.keep(base64_encode(data)) })
}

///|
test "bench/base64_decode" (b : @bench.T) {
  let data = make_bench_byte_array(4095) // not divisible by 3 → exercises padding
  let encoded = base64_encode(data)
  let out = FixedArray::make(
    base64_decoded_len(encoded.length(), base64_count_padding(encoded)),
    b'\x00',
  )
  b.bench(name="scalar_decode_5460", fn() { b.keep(ref_decode(encoded, out)) })
  b.bench(name="decode_5460", fn() { b.keep(base64_decode(encoded)) })
}
