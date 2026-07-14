// Copyright 2026 International Digital Economy Academy
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

///|
#cfg(any(target="native", target="wasm"))
priv struct JsonClassifiedBlock {
  quote : Int
  backslash : Int
  structural : Int
  whitespace : Int
  control : Int
}

///|
#cfg(any(target="native", target="wasm"))
fn json_bytes_data(input : Bytes) -> FixedArray[Byte] = "%identity"

///|
#cfg(any(target="native", target="wasm"))
let json_class_lo : V128 = @v128.i8x16_const(
  0x80, 0, 0x01, 0, 0, 0, 0, 0, 0, 0x40, 0x44, 0x18, 0x22, 0x58, 0, 0,
)

///|
#cfg(any(target="native", target="wasm"))
let json_class_hi : V128 = @v128.i8x16_const(
  0x40, 0, 0x83, 0x04, 0, 0x28, 0, 0x10, 0, 0, 0, 0, 0, 0, 0, 0,
)

///|
#cfg(any(target="native", target="wasm"))
fn classify_json_i8x16(
  data : FixedArray[Byte],
  offset : Int,
) -> JsonClassifiedBlock {
  let chunk = @v128.v128_load(data, offset)
  let low_nibbles = @v128.v128_and_(chunk, @v128.i8x16_splat(0x0F))
  let high_nibbles = @v128.v128_and_(
    @v128.i8x16_shr_u(chunk, 4),
    @v128.i8x16_splat(0x0F),
  )
  // A two-table nibble lookup classifies all JSON punctuation and whitespace
  // at once. Each class is represented by one bit present in both the low-
  // and high-nibble table entry; their intersection rejects false pairs.
  let classes = @v128.v128_and_(
    @v128.i8x16_swizzle(json_class_lo, low_nibbles),
    @v128.i8x16_swizzle(json_class_hi, high_nibbles),
  )
  let zero = @v128.i8x16_splat(0)
  let quote = @v128.i8x16_bitmask(
    @v128.i8x16_ne(@v128.v128_and_(classes, @v128.i8x16_splat(0x01)), zero),
  )
  let backslash = @v128.i8x16_bitmask(
    @v128.i8x16_ne(@v128.v128_and_(classes, @v128.i8x16_splat(0x20)), zero),
  )
  let structural = @v128.i8x16_bitmask(
    @v128.i8x16_ne(@v128.v128_and_(classes, @v128.i8x16_splat(0x1E)), zero),
  )
  let whitespace = @v128.i8x16_bitmask(
    @v128.i8x16_ne(@v128.v128_and_(classes, @v128.i8x16_splat(0xC0)), zero),
  )
  let control = @v128.i8x16_bitmask(
    @v128.i8x16_lt_u(chunk, @v128.i8x16_splat(b' ')),
  )
  { quote, backslash, structural, whitespace, control }
}

///|
#cfg(any(target="native", target="wasm"))
fn classify_json_bytes_tail(
  input : Bytes,
  offset : Int,
  count : Int,
) -> JsonClassifiedBlock {
  let mut quote = 0
  let mut backslash = 0
  let mut structural = 0
  let mut whitespace = 0
  let mut control = 0
  for lane in 0..<count {
    let byte = input.unsafe_get(offset + lane)
    let bit = 1 << lane
    if byte == b'"' {
      quote = quote | bit
    }
    if byte == b'\\' {
      backslash = backslash | bit
    }
    if byte == b'{' ||
      byte == b'}' ||
      byte == b'[' ||
      byte == b']' ||
      byte == b':' ||
      byte == b',' {
      structural = structural | bit
    }
    if byte == b' ' || byte == b'\t' || byte == b'\n' || byte == b'\r' {
      whitespace = whitespace | bit
    }
    if byte < b' ' {
      control = control | bit
    }
  }
  { quote, backslash, structural, whitespace, control }
}

///|
#cfg(any(target="native", target="wasm"))
fn json_prefix_xor16(bits : Int) -> Int {
  let bits = (bits ^ (bits << 1)) & 0xFFFF
  let bits = (bits ^ (bits << 2)) & 0xFFFF
  let bits = (bits ^ (bits << 4)) & 0xFFFF
  (bits ^ (bits << 8)) & 0xFFFF
}

///|
declare fn json_simd_token_starts_utf8(input : Bytes) -> Array[Int]

///|
// A small prefix predicts both tape density and a useful initial capacity.
// The arithmetic avoids `length * sample_starts`, which can overflow a Wasm
// Int for inputs much smaller than the maximum Bytes length.
#valtype
priv struct JsonTokenSample {
  bytes : Int
  starts : Int
}

///|
fn json_utf8_token_sample(input : Bytes) -> JsonTokenSample {
  let limit = input.length().min(4096)
  let mut starts = 0
  let mut in_string = false
  let mut escaped = false
  let mut previous_is_atom = false
  for offset in 0..<limit {
    let byte = input.unsafe_get(offset)
    if in_string {
      if escaped {
        escaped = false
      } else if byte == b'\\' {
        escaped = true
      } else if byte == b'"' {
        starts += 1
        in_string = false
      }
      previous_is_atom = false
    } else if byte == b'"' {
      starts += 1
      in_string = true
      previous_is_atom = false
    } else if byte == b'{' ||
      byte == b'}' ||
      byte == b'[' ||
      byte == b']' ||
      byte == b':' ||
      byte == b',' {
      starts += 1
      previous_is_atom = false
    } else if byte == b' ' || byte == b'\t' || byte == b'\n' || byte == b'\r' {
      previous_is_atom = false
    } else {
      if !previous_is_atom {
        starts += 1
      }
      previous_is_atom = true
    }
  }
  { bytes: limit, starts }
}

///|
fn json_token_capacity_hint(input : Bytes) -> Int {
  let sample = json_utf8_token_sample(input)
  if sample.bytes == 0 {
    return 0
  }
  let length = input.length()
  let blocks = length / sample.bytes
  let remainder = length % sample.bytes
  let estimated = blocks * sample.starts +
    (remainder * sample.starts + sample.bytes - 1) / sample.bytes
  // Small slack absorbs record-boundary bias without returning to the old
  // `length / 4` over-allocation on string-heavy documents.
  (estimated + estimated / 16 + 16).min(length)
}

///|
#warnings("-unused_value")
fn json_scalar_token_starts_utf8(input : Bytes) -> Array[Int] {
  let starts = Array::new(capacity=json_token_capacity_hint(input))
  let mut in_string = false
  let mut escaped = false
  let mut previous_is_atom = false
  for offset in 0..<input.length() {
    let byte = input.unsafe_get(offset)
    if in_string {
      if escaped {
        escaped = false
      } else if byte == b'\\' {
        escaped = true
      } else if byte == b'"' {
        starts.push(offset)
        in_string = false
      }
      previous_is_atom = false
    } else if byte == b'"' {
      starts.push(offset)
      in_string = true
      previous_is_atom = false
    } else if byte == b'{' ||
      byte == b'}' ||
      byte == b'[' ||
      byte == b']' ||
      byte == b':' ||
      byte == b',' {
      starts.push(offset)
      previous_is_atom = false
    } else if byte == b' ' || byte == b'\t' || byte == b'\n' || byte == b'\r' {
      previous_is_atom = false
    } else {
      if !previous_is_atom {
        starts.push(offset)
      }
      previous_is_atom = true
    }
  }
  starts
}

///|
// Sampling avoids choosing the Wasm SIMD classifier for punctuation-dense
// inputs, where mask extraction and sparse-event walking stop being sparse.
// Four KiB is large enough to cover many ordinary records while keeping the
// decision cost below the benchmark noise for multi-megabyte inputs.
#warnings("-unused_value")
fn json_utf8_sample_is_dense(input : Bytes) -> Bool {
  let sample = json_utf8_token_sample(input)
  if sample.bytes == 0 {
    return false
  }
  sample.starts * 4 > sample.bytes
}

///|
declare fn json_adaptive_token_starts_utf8(input : Bytes) -> Array[Int]

///|
#cfg(target="native")
fn json_adaptive_token_starts_utf8(input : Bytes) -> Array[Int] {
  json_simd_token_starts_utf8(input)
}

///|
#cfg(target="wasm")
fn json_adaptive_token_starts_utf8(input : Bytes) -> Array[Int] {
  if json_utf8_sample_is_dense(input) {
    json_scalar_token_starts_utf8(input)
  } else {
    json_simd_token_starts_utf8(input)
  }
}

///|
#cfg(any(target="js", target="wasm-gc", target="llvm"))
fn json_adaptive_token_starts_utf8(input : Bytes) -> Array[Int] {
  json_scalar_token_starts_utf8(input)
}

///|
#cfg(any(target="native", target="wasm"))
#warnings("-unused_value")
fn json_simd_token_starts_utf8(input : Bytes) -> Array[Int] {
  let length = input.length()
  let data = json_bytes_data(input)
  let starts = Array::new(capacity=json_token_capacity_hint(input))
  let mut in_string = false
  let mut escaped = false
  let mut previous_is_atom = false
  for base = 0; base < length; base = base + 16 {
    let count = (length - base).min(16)
    let block = if count == 16 {
      classify_json_i8x16(data, base)
    } else {
      classify_json_bytes_tail(input, base, count)
    }
    let valid = (1 << count) - 1
    let string_at_start = in_string
    let mut unescaped_quotes = 0
    let mut events = (block.quote | block.backslash) & valid
    let mut previous_event = -1
    // Walk only context-changing characters. Ordinary bytes never branch in
    // this phase; a gap resets backslash parity in one operation.
    while events != 0 {
      let lane = events.ctz()
      if lane > previous_event + 1 {
        escaped = false
      }
      let bit = 1 << lane
      if in_string {
        if (block.backslash & bit) != 0 {
          escaped = !escaped
        } else {
          if !escaped {
            unescaped_quotes = unescaped_quotes | bit
            in_string = false
          }
          // A quote, escaped or not, ends the preceding backslash run.
          escaped = false
        }
      } else if (block.quote & bit) != 0 {
        escaped = false
        unescaped_quotes = unescaped_quotes | bit
        in_string = true
      } else {
        escaped = false
      }
      previous_event = lane
      events = events & (events - 1)
    }
    if previous_event != count - 1 {
      escaped = false
    }
    let prefix = json_prefix_xor16(unescaped_quotes)
    let inside = if string_at_start {
      prefix.lnot() & valid
    } else {
      prefix & valid
    }
    // Kept as a separate diagnostic lane for a future stage-2 parser.
    ignore(block.control & inside)
    let structural = block.structural & inside.lnot() & valid
    let non_atom = inside | block.structural | block.whitespace | block.quote
    let atom = non_atom.lnot() & valid
    let predecessor = ((atom << 1) & valid) |
      (if previous_is_atom { 1 } else { 0 })
    let atom_starts = atom & predecessor.lnot()
    let mut start_mask = structural | unescaped_quotes | atom_starts
    while start_mask != 0 {
      let lane = start_mask.ctz()
      starts.push(base + lane)
      start_mask = start_mask & (start_mask - 1)
    }
    previous_is_atom = ((atom >> (count - 1)) & 1) != 0
  }
  starts
}

///|
#cfg(any(target="js", target="wasm-gc", target="llvm"))
#warnings("-unused_value")
fn json_simd_token_starts_utf8(input : Bytes) -> Array[Int] {
  json_scalar_token_starts_utf8(input)
}
