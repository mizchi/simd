// Scalar / FixedArray implementation of the simdjson byte-classification
// pipeline, used on the `native` and `js` targets where inline-WAT v128
// is unavailable. Public API is byte-for-byte identical with the wasm
// version in `simdjson_wasm.mbt`, so caller code compiles unchanged
// across all four targets.

///|
/// Bit `i % 32` of `out[i / 32]` set iff input byte `i` ∈ `{ } [ ] , :`.
/// OR-in semantics.
pub fn classify_structural(
  input : @simd_buffer.SimdBufferBytes,
  out : @simd_buffer.SimdBuffer,
) -> Unit {
  let n = input.length()
  let words = (n + 31) / 32
  if out.length() < words {
    abort("simdjson.classify_structural: out too small")
  }
  let data = input.backing()
  let bits = out.backing()
  for i in 0..<n {
    let b = data[i].to_int()
    if b == 0x7B ||
      b == 0x7D ||
      b == 0x5B ||
      b == 0x5D ||
      b == 0x2C ||
      b == 0x3A {
      let word = i >> 5
      let bit = i & 31
      bits[word] = bits[word] | (1 << bit)
    }
  }
}

///|
/// Bit `i % 32` of `out[i / 32]` set iff input byte `i` ∈ `0-9 - + . e E`.
/// OR-in semantics.
pub fn classify_numeric(
  input : @simd_buffer.SimdBufferBytes,
  out : @simd_buffer.SimdBuffer,
) -> Unit {
  let n = input.length()
  let words = (n + 31) / 32
  if out.length() < words {
    abort("simdjson.classify_numeric: out too small")
  }
  let data = input.backing()
  let bits = out.backing()
  for i in 0..<n {
    let b = data[i].to_int()
    let is_digit = b >= 0x30 && b <= 0x39
    if is_digit || b == 0x2D || b == 0x2E || b == 0x2B || b == 0x65 || b == 0x45 {
      let word = i >> 5
      let bit = i & 31
      bits[word] = bits[word] | (1 << bit)
    }
  }
}

///|
/// Bit `i % 32` of `out[i / 32]` set iff input byte `i` == `"`. OR-in.
pub fn classify_quote_raw(
  input : @simd_buffer.SimdBufferBytes,
  out : @simd_buffer.SimdBuffer,
) -> Unit {
  classify_byte_eq(input, out, b'\x22', "classify_quote_raw")
}

///|
/// Bit `i % 32` of `out[i / 32]` set iff input byte `i` == `\`. OR-in.
pub fn classify_backslash(
  input : @simd_buffer.SimdBufferBytes,
  out : @simd_buffer.SimdBuffer,
) -> Unit {
  classify_byte_eq(input, out, b'\x5C', "classify_backslash")
}

///|
fn classify_byte_eq(
  input : @simd_buffer.SimdBufferBytes,
  out : @simd_buffer.SimdBuffer,
  needle : Byte,
  op_name : String,
) -> Unit {
  let n = input.length()
  let words = (n + 31) / 32
  if out.length() < words {
    abort("simdjson." + op_name + ": out too small")
  }
  let data = input.backing()
  let bits = out.backing()
  for i in 0..<n {
    if data[i] == needle {
      let word = i >> 5
      let bit = i & 31
      bits[word] = bits[word] | (1 << bit)
    }
  }
}

///|
/// Compute the JSON in-string bitmask. **Overwrites** `out`.
/// Honours `\"` and `\\` escapes.
pub fn compute_quote_mask(
  input : @simd_buffer.SimdBufferBytes,
  out : @simd_buffer.SimdBuffer,
) -> Unit {
  let n = input.length()
  let words = (n + 31) / 32
  if out.length() < words {
    abort("simdjson.compute_quote_mask: out too small")
  }
  let data = input.backing()
  let bits = out.backing()
  for w in 0..<words {
    bits[w] = 0
  }
  let mut in_string = false
  let mut escape_pending = false
  for i in 0..<n {
    let b = data[i].to_int()
    let mut bit_set = false
    if escape_pending {
      escape_pending = false
      bit_set = in_string
    } else if b == 0x22 {
      in_string = !in_string
    } else if b == 0x5C && in_string {
      escape_pending = true
      bit_set = true
    } else {
      bit_set = in_string
    }
    if bit_set {
      bits[i >> 5] = bits[i >> 5] | (1 << (i & 31))
    }
  }
}

///|
/// Walk `structural & ~quote_mask` and emit byte offsets. Returns count.
pub fn extract_structural_indices(
  structural : @simd_buffer.SimdBuffer,
  quote_mask : @simd_buffer.SimdBuffer,
  n : Int,
  indices : @simd_buffer.SimdBuffer,
) -> Int {
  let words = (n + 31) / 32
  if structural.length() < words || quote_mask.length() < words {
    abort("simdjson.extract_structural_indices: bitmap buffers too small")
  }
  let struct_arr = structural.backing()
  let quote_arr = quote_mask.backing()
  let idx_arr = indices.backing()
  let mut count = 0
  let mut w = 0
  while w < words {
    let mut word = struct_arr[w] & quote_arr[w].lnot()
    while word != 0 {
      let mut x = word
      let mut bit = 0
      while (x & 1) == 0 {
        x = x >> 1
        bit = bit + 1
      }
      idx_arr[count] = w * 32 + bit
      count = count + 1
      word = word & (word - 1)
    }
    w = w + 1
  }
  count
}

///|
/// Convenience: full pipeline with internal scratch allocation. Hot
/// paths should use `find_structural_indices_with_scratch`.
pub fn find_structural_indices(
  input : @simd_buffer.SimdBufferBytes,
  indices : @simd_buffer.SimdBuffer,
) -> Int {
  let n = input.length()
  if n == 0 {
    return 0
  }
  let words = (n + 31) / 32
  let structural = @simd_buffer.SimdBuffer::make(words)
  let quote_mask = @simd_buffer.SimdBuffer::make(words)
  find_structural_indices_with_scratch(input, structural, quote_mask, indices)
}

///|
/// Scratch-receiving variant.
pub fn find_structural_indices_with_scratch(
  input : @simd_buffer.SimdBufferBytes,
  structural_scratch : @simd_buffer.SimdBuffer,
  quote_mask_scratch : @simd_buffer.SimdBuffer,
  indices : @simd_buffer.SimdBuffer,
) -> Int {
  let n = input.length()
  if n == 0 {
    return 0
  }
  let words = (n + 31) / 32
  if structural_scratch.length() < words || quote_mask_scratch.length() < words {
    abort("simdjson.find_structural_indices_with_scratch: scratch too small")
  }
  let struct_arr = structural_scratch.backing()
  for w in 0..<words {
    struct_arr[w] = 0
  }
  classify_structural(input, structural_scratch)
  compute_quote_mask(input, quote_mask_scratch)
  extract_structural_indices(structural_scratch, quote_mask_scratch, n, indices)
}
