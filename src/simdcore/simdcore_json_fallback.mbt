// JSON structural indexing — scalar fallback (wasm-gc / native / js),
// `Bytes`-direct. Same public API as the wasm SIMD path; results identical.

///|
/// Classify structural bytes (`{ } [ ] , :`) into a bitmap. `out` is zeroed
/// first. Scalar.
pub fn json_classify_structural(input : Bytes, out : FixedArray[Int]) -> Unit {
  let n = input.length()
  let words = (n + 31) / 32
  if out.length() < words {
    abort("simdcore.json_classify_structural: out too small")
  }
  for w in 0..<words {
    out[w] = 0
  }
  for i in 0..<n {
    let b = input[i].to_int()
    if b == 0x7B ||
      b == 0x7D ||
      b == 0x5B ||
      b == 0x5D ||
      b == 0x2C ||
      b == 0x3A {
      out[i >> 5] = out[i >> 5] | (1 << (i & 31))
    }
  }
}

///|
/// Full structural-index pipeline into caller scratch, returning the count.
/// Scalar.
pub fn json_structural_indices_into(
  input : Bytes,
  structural : FixedArray[Int],
  quote : FixedArray[Int],
  indices : FixedArray[Int],
) -> Int {
  let n = input.length()
  if n == 0 {
    return 0
  }
  let words = (n + 31) / 32
  if structural.length() < words || quote.length() < words {
    abort("simdcore.json_structural_indices_into: scratch too small")
  }
  if indices.length() < n {
    abort("simdcore.json_structural_indices_into: indices too small")
  }
  // classify structural
  json_classify_structural(input, structural)
  // compute quote mask (in-string bits, honouring \" and \\)
  for w in 0..<words {
    quote[w] = 0
  }
  let mut in_string = false
  let mut escape_pending = false
  for i in 0..<n {
    let b = input[i].to_int()
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
  // extract structural & ~quote
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
/// Convenience: structural byte offsets (excluding in-string) as an
/// `Array[Int]`. Scalar.
pub fn json_structural_indices(input : Bytes) -> Array[Int] {
  let n = input.length()
  if n == 0 {
    return Array::new()
  }
  let words = (n + 31) / 32
  let structural = FixedArray::make(words, 0)
  let quote = FixedArray::make(words, 0)
  let indices = FixedArray::make(n, 0)
  let count = json_structural_indices_into(input, structural, quote, indices)
  let result = Array::new(capacity=count)
  for i in 0..<count {
    result.push(indices[i])
  }
  result
}
