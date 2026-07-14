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
/// Error returned by the experimental UTF-8 parser.
pub(all) suberror ParseError {
  InvalidSyntax
  DepthLimitExceeded
} derive(Eq, @debug.Debug)

///|
pub impl Show for ParseError with fn output(self, logger) {
  match self {
    InvalidSyntax => logger.write_string("Invalid JSON syntax")
    DepthLimitExceeded =>
      logger.write_string("JSON nesting depth limit exceeded")
  }
}

///|
const SAFE_INTEGER_LIMIT : Int64 = 9007199254740991L

///|
const MAX_MANTISSA_FAST_PATH : UInt64 = 9007199254740992UL

///|
const MIN_EXPONENT_FAST_PATH : Int64 = -22L

///|
const MAX_EXPONENT_FAST_PATH : Int64 = 22L

///|
const MAX_EXPONENT_DISGUISED_FAST_PATH : Int64 = 37L

///|
const EXPONENT_CAP : Int64 = 100000L

///|
const MAX_UINT64 : UInt64 = 0xffffffffffffffffUL

///|
let pow10_table : ReadOnlyArray[Double] = [
  1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0, 100000000.0,
  1000000000.0, 10000000000.0, 100000000000.0, 1000000000000.0, 10000000000000.0,
  100000000000000.0, 1000000000000000.0, 10000000000000000.0, 100000000000000000.0,
  1000000000000000000.0, 10000000000000000000.0, 100000000000000000000.0, 1000000000000000000000.0,
  10000000000000000000000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
]

///|
let int_pow10_table : ReadOnlyArray[UInt64] = [
  1UL, 10UL, 100UL, 1000UL, 10000UL, 100000UL, 1000000UL, 10000000UL, 100000000UL,
  1000000000UL, 10000000000UL, 100000000000UL, 1000000000000UL, 10000000000000UL,
  100000000000000UL, 1000000000000000UL,
]

///|
fn json_pow10_fast_path(exponent : Int) -> Double {
  pow10_table[exponent & 31]
}

///|
fn checked_mul(a : UInt64, b : UInt64) -> UInt64? {
  if a == 0UL || b == 0UL {
    return Some(0UL)
  }
  if a == 1UL {
    return Some(b)
  }
  if b == 1UL {
    return Some(a)
  }
  if b.clz() == 0 || a.clz() == 0 {
    return None
  }
  let quotient = MAX_UINT64 / b
  if a > quotient {
    None
  } else {
    Some(a * b)
  }
}

///|
priv struct SimdJsonParseContext {
  input : Bytes
  positions : Array[Int]
  mut index : Int
}

///|
#valtype
priv struct SimdJsonNumberScan {
  negative : Bool
  is_integer : Bool
  mantissa : UInt64
  exponent : Int64
  many_digits : Bool
}

///|
#valtype
priv struct SimdJsonLexedNumber {
  repr : String?
  value : Double
}

///|
fn SimdJsonParseContext::next_position(
  ctx : SimdJsonParseContext,
) -> Int raise ParseError {
  guard ctx.index < ctx.positions.length() else { raise InvalidSyntax }
  let position = ctx.positions.unsafe_get(ctx.index)
  ctx.index += 1
  position
}

///|
fn SimdJsonParseContext::peek_byte(
  ctx : SimdJsonParseContext,
) -> Byte raise ParseError {
  guard ctx.index < ctx.positions.length() else { raise InvalidSyntax }
  ctx.input.unsafe_get(ctx.positions.unsafe_get(ctx.index))
}

///|
fn SimdJsonParseContext::expect_byte(
  ctx : SimdJsonParseContext,
  expected : Byte,
) -> Unit raise ParseError {
  let position = ctx.next_position()
  guard ctx.input.unsafe_get(position) == expected else { raise InvalidSyntax }
}

///|
fn is_simd_json_whitespace(byte : Byte) -> Bool {
  byte == b' ' || byte == b'\t' || byte == b'\n' || byte == b'\r'
}

///|
fn SimdJsonParseContext::atom_end(
  ctx : SimdJsonParseContext,
  start : Int,
) -> Int {
  let token_boundary = if ctx.index < ctx.positions.length() {
    ctx.positions.unsafe_get(ctx.index)
  } else {
    ctx.input.length()
  }
  for end = token_boundary; end > start; {
    if !is_simd_json_whitespace(ctx.input.unsafe_get(end - 1)) {
      break end
    }
    continue end - 1
  } nobreak {
    end
  }
}

///|
fn simd_json_bytes_equal(
  input : Bytes,
  start : Int,
  end : Int,
  expected : BytesView,
) -> Bool {
  guard end - start == expected.length() else { return false }
  for i in 0..<expected.length() {
    if input.unsafe_get(start + i) != expected.unsafe_get(i) {
      return false
    }
  }
  true
}

///|
// MoonBit strings can preserve lone UTF-16 surrogate code units, while a
// temporary UTF-8 buffer cannot represent them. Delegate only this rare path
// to core/json so escaped-string semantics stay identical.
fn decode_surrogate_json_string_with_core(
  input : Bytes,
  start : Int,
  end : Int,
) -> String raise ParseError {
  guard start > 0 && end < input.length() else { raise InvalidSyntax }
  let quoted = @utf8.decode(input[start - 1:end + 1]) catch {
    _ => raise InvalidSyntax
  }
  let parsed = @core_json.parse(quoted) catch { _ => raise InvalidSyntax }
  match parsed {
    String(value) => value
    _ => raise InvalidSyntax
  }
}

///|
fn decode_simd_json_string(
  input : Bytes,
  start : Int,
  end : Int,
) -> String raise ParseError {
  let mut first_escape = -1
  for i in start..<end {
    let byte = input.unsafe_get(i)
    if byte < b' ' {
      raise InvalidSyntax
    }
    if byte == b'\\' {
      first_escape = i
      break
    }
  }
  if first_escape < 0 {
    return @utf8.decode(input[start:end]) catch { _ => raise InvalidSyntax }
  }
  let buf = @buffer.Buffer(size_hint=end - start)
  if start < first_escape {
    buf.write_bytes(input[start:first_escape])
  }
  let mut segment_start = first_escape
  let mut i = first_escape
  while i < end {
    let byte = input.unsafe_get(i)
    if byte < b' ' {
      raise InvalidSyntax
    }
    if byte != b'\\' {
      i += 1
      continue
    }
    if segment_start < i {
      buf.write_bytes(input[segment_start:i])
    }
    i += 1
    guard i < end else { raise InvalidSyntax }
    match input.unsafe_get(i) {
      b'b' => buf.write_byte(b'\b')
      b'f' => buf.write_byte(b'\x0C')
      b'n' => buf.write_byte(b'\n')
      b'r' => buf.write_byte(b'\r')
      b't' => buf.write_byte(b'\t')
      b'"' => buf.write_byte(b'"')
      b'\\' => buf.write_byte(b'\\')
      b'/' => buf.write_byte(b'/')
      b'u' => {
        guard i + 4 < end else { raise InvalidSyntax }
        let mut code = 0
        for j in (i + 1)..<=(i + 4) {
          let digit = match input.unsafe_get(j) {
            b'0'..=b'9' as byte => byte.to_int() - b'0'.to_int()
            b'a'..=b'f' as byte => byte.to_int() - b'a'.to_int() + 10
            b'A'..=b'F' as byte => byte.to_int() - b'A'.to_int() + 10
            _ => raise InvalidSyntax
          }
          code = (code << 4) | digit
        }
        if code >= 0xD800 && code <= 0xDFFF {
          return decode_surrogate_json_string_with_core(input, start, end)
        }
        buf.write_char_utf8(code.unsafe_to_char())
        i += 4
      }
      _ => raise InvalidSyntax
    }
    i += 1
    segment_start = i
  }
  if segment_start < end {
    buf.write_bytes(input[segment_start:end])
  }
  @utf8.decode(buf.to_bytes()) catch {
    _ => raise InvalidSyntax
  }
}

///|
fn SimdJsonNumberScan::try_fast_double(self : SimdJsonNumberScan) -> Double {
  if self.mantissa == 0UL {
    let value = 0.0
    return if self.negative { -value } else { value }
  }
  if self.many_digits ||
    self.exponent < MIN_EXPONENT_FAST_PATH ||
    self.exponent > MAX_EXPONENT_DISGUISED_FAST_PATH ||
    self.mantissa > MAX_MANTISSA_FAST_PATH {
    return @double.not_a_number
  }
  let value = if self.exponent <= MAX_EXPONENT_FAST_PATH {
    let value = self.mantissa.to_double()
    if self.exponent < 0L {
      value / json_pow10_fast_path(-self.exponent.to_int())
    } else {
      value * json_pow10_fast_path(self.exponent.to_int())
    }
  } else {
    let shift = self.exponent - MAX_EXPONENT_FAST_PATH
    let mantissa = match
      checked_mul(self.mantissa, int_pow10_table[shift.to_int()]) {
      Some(m) => m
      None => return @double.not_a_number
    }
    if mantissa > MAX_MANTISSA_FAST_PATH {
      return @double.not_a_number
    }
    mantissa.to_double() * json_pow10_fast_path(MAX_EXPONENT_FAST_PATH.to_int())
  }
  if self.negative {
    -value
  } else {
    value
  }
}

///|
fn scan_simd_json_number(
  input : Bytes,
  start : Int,
  end : Int,
) -> SimdJsonNumberScan raise ParseError {
  let mut i = start
  let negative = input.unsafe_get(i) == b'-'
  if negative {
    i += 1
    guard i < end else { raise InvalidSyntax }
  }
  let first = input.unsafe_get(i)
  guard first >= b'0' && first <= b'9' else { raise InvalidSyntax }
  let mut mantissa = 0UL
  let mut significant_digits = 0
  let mut seen_nonzero = false
  if first == b'0' {
    i += 1
    if i < end && input.unsafe_get(i) >= b'0' && input.unsafe_get(i) <= b'9' {
      raise InvalidSyntax
    }
  } else {
    while i < end && input.unsafe_get(i) >= b'0' && input.unsafe_get(i) <= b'9' {
      let digit = input.unsafe_get(i).to_int() - b'0'.to_int()
      seen_nonzero = true
      significant_digits += 1
      if significant_digits <= 19 {
        mantissa = mantissa * 10UL +
          UInt64::extend_uint(digit.reinterpret_as_uint())
      }
      i += 1
    }
  }
  let mut has_decimal = false
  let mut fractional_digits = 0
  if i < end && input.unsafe_get(i) == b'.' {
    has_decimal = true
    i += 1
    guard i < end && input.unsafe_get(i) >= b'0' && input.unsafe_get(i) <= b'9' else {
      raise InvalidSyntax
    }
    while i < end && input.unsafe_get(i) >= b'0' && input.unsafe_get(i) <= b'9' {
      let digit = input.unsafe_get(i).to_int() - b'0'.to_int()
      fractional_digits += 1
      if digit != 0 || seen_nonzero {
        seen_nonzero = true
        significant_digits += 1
        if significant_digits <= 19 {
          mantissa = mantissa * 10UL +
            UInt64::extend_uint(digit.reinterpret_as_uint())
        }
      }
      i += 1
    }
  }
  let mut has_exponent = false
  let mut exponent_negative = false
  let mut exponent_part = 0L
  if i < end && (input.unsafe_get(i) == b'e' || input.unsafe_get(i) == b'E') {
    has_exponent = true
    i += 1
    if i < end && (input.unsafe_get(i) == b'+' || input.unsafe_get(i) == b'-') {
      exponent_negative = input.unsafe_get(i) == b'-'
      i += 1
    }
    guard i < end && input.unsafe_get(i) >= b'0' && input.unsafe_get(i) <= b'9' else {
      raise InvalidSyntax
    }
    while i < end && input.unsafe_get(i) >= b'0' && input.unsafe_get(i) <= b'9' {
      if exponent_part < EXPONENT_CAP {
        let digit = input.unsafe_get(i).to_int() - b'0'.to_int()
        exponent_part = (exponent_part * 10L + digit.to_int64()).min(
          EXPONENT_CAP,
        )
      }
      i += 1
    }
  }
  guard i == end else { raise InvalidSyntax }
  let exponent_part = if exponent_negative {
    -exponent_part
  } else {
    exponent_part
  }
  {
    negative,
    is_integer: !has_decimal && !has_exponent,
    mantissa,
    exponent: exponent_part - fractional_digits.to_int64(),
    many_digits: significant_digits > 19,
  }
}

///|
fn parse_simd_json_number(
  input : Bytes,
  start : Int,
  end : Int,
) -> SimdJsonLexedNumber raise ParseError {
  let scan = scan_simd_json_number(input, start, end)
  if scan.is_integer {
    if !scan.many_digits &&
      scan.exponent == 0L &&
      scan.mantissa <= SAFE_INTEGER_LIMIT.reinterpret_as_uint64() {
      let value = scan.mantissa.to_double()
      return { value: if scan.negative { -value } else { value }, repr: None }
    }
    let repr = @utf8.decode(input[start:end]) catch { _ => raise InvalidSyntax }
    return {
      value: if scan.negative {
        @double.neg_infinity
      } else {
        @double.infinity
      },
      repr: Some(repr),
    }
  }
  let fast = scan.try_fast_double()
  if !fast.is_nan() {
    return { value: fast, repr: None }
  }
  let repr = @utf8.decode(input[start:end]) catch { _ => raise InvalidSyntax }
  let parsed = @core_json.parse(repr) catch { _ => raise InvalidSyntax }
  match parsed {
    Number(value, repr~) => { value, repr }
    _ => raise InvalidSyntax
  }
}

///|
fn SimdJsonParseContext::parse_string_after_open(
  ctx : SimdJsonParseContext,
  open : Int,
) -> String raise ParseError {
  let close = ctx.next_position()
  guard ctx.input.unsafe_get(close) == b'"' else { raise InvalidSyntax }
  decode_simd_json_string(ctx.input, open + 1, close)
}

///|
fn SimdJsonParseContext::parse_value(
  ctx : SimdJsonParseContext,
  remaining_depth : Int,
) -> Json raise ParseError {
  let position = ctx.next_position()
  let byte = ctx.input.unsafe_get(position)
  match byte {
    b'"' => Json::string(ctx.parse_string_after_open(position))
    b'[' => ctx.parse_array(remaining_depth)
    b'{' => ctx.parse_object(remaining_depth)
    b'n' => {
      let end = ctx.atom_end(position)
      guard simd_json_bytes_equal(ctx.input, position, end, b"null") else {
        raise InvalidSyntax
      }
      null
    }
    b't' => {
      let end = ctx.atom_end(position)
      guard simd_json_bytes_equal(ctx.input, position, end, b"true") else {
        raise InvalidSyntax
      }
      Json::boolean(true)
    }
    b'f' => {
      let end = ctx.atom_end(position)
      guard simd_json_bytes_equal(ctx.input, position, end, b"false") else {
        raise InvalidSyntax
      }
      Json::boolean(false)
    }
    b'-' | b'0'..=b'9' => {
      let end = ctx.atom_end(position)
      let { value, repr } = parse_simd_json_number(ctx.input, position, end)
      Json::number(value, repr?)
    }
    _ => raise InvalidSyntax
  }
}

///|
fn SimdJsonParseContext::parse_array(
  ctx : SimdJsonParseContext,
  remaining_depth : Int,
) -> Json raise ParseError {
  guard remaining_depth > 0 else { raise DepthLimitExceeded }
  let values = []
  if ctx.peek_byte() == b']' {
    ctx.expect_byte(b']')
    return Json::array(values)
  }
  for ;; {
    values.push(ctx.parse_value(remaining_depth - 1))
    let separator = ctx.next_position()
    match ctx.input.unsafe_get(separator) {
      b']' => return Json::array(values)
      b',' => continue
      _ => raise InvalidSyntax
    }
  }
}

///|
fn SimdJsonParseContext::parse_object(
  ctx : SimdJsonParseContext,
  remaining_depth : Int,
) -> Json raise ParseError {
  guard remaining_depth > 0 else { raise DepthLimitExceeded }
  let object : Map[String, Json] = Map([])
  if ctx.peek_byte() == b'}' {
    ctx.expect_byte(b'}')
    return Json::object(object)
  }
  for ;; {
    let open = ctx.next_position()
    guard ctx.input.unsafe_get(open) == b'"' else { raise InvalidSyntax }
    let key = ctx.parse_string_after_open(open)
    ctx.expect_byte(b':')
    object[key] = ctx.parse_value(remaining_depth - 1)
    let separator = ctx.next_position()
    match ctx.input.unsafe_get(separator) {
      b'}' => return Json::object(object)
      b',' => continue
      _ => raise InvalidSyntax
    }
  }
}

///|
// Stage 1 creates an adaptive SIMD token-position tape. Stage 2 consumes the
// tape without rescanning whitespace, atom bodies, or string boundaries.
fn parse_with_positions(
  input : Bytes,
  positions : Array[Int],
  max_nesting_depth : Int,
) -> Json raise ParseError {
  let ctx : SimdJsonParseContext = { input, positions, index: 0 }
  let value = ctx.parse_value(max_nesting_depth)
  guard ctx.index == positions.length() else { raise InvalidSyntax }
  value
}

///|
/// Parse UTF-8 encoded JSON bytes into the standard MoonBit Json value.
///
/// Native and Wasm use a v128 structural-classification pass. Wasm falls back
/// to the scalar indexer for punctuation-dense inputs; JS and Wasm-GC use the
/// scalar indexer until direct v128 byte loads are available there.
pub fn parse(
  input : Bytes,
  max_nesting_depth? : Int = 1024,
) -> Json raise ParseError {
  parse_with_positions(
    input,
    json_adaptive_token_starts_utf8(input),
    max_nesting_depth,
  )
}

///|
// String input does not always benefit from a UTF-8 round trip. Short inputs
// and long string-dominated inputs stay on core/json; dense structural or
// numeric documents use the indexed UTF-8 pipeline.
fn json_string_should_use_core(input : StringView) -> Bool {
  if input.length() < 65536 {
    return true
  }
  let limit = input.length().min(4096)
  let mut starts = 0
  let mut in_string = false
  let mut escaped = false
  let mut previous_is_atom = false
  for offset in 0..<limit {
    let code_unit = input.unsafe_get(offset)
    if in_string {
      if escaped {
        escaped = false
      } else if code_unit == 0x5C {
        escaped = true
      } else if code_unit == 0x22 {
        starts += 1
        in_string = false
      }
      previous_is_atom = false
    } else if code_unit == 0x22 {
      starts += 1
      in_string = true
      previous_is_atom = false
    } else if code_unit == 0x7B ||
      code_unit == 0x7D ||
      code_unit == 0x5B ||
      code_unit == 0x5D ||
      code_unit == 0x3A ||
      code_unit == 0x2C {
      starts += 1
      previous_is_atom = false
    } else if code_unit == 0x20 ||
      code_unit == 0x09 ||
      code_unit == 0x0A ||
      code_unit == 0x0D {
      previous_is_atom = false
    } else {
      if !previous_is_atom {
        starts += 1
      }
      previous_is_atom = true
    }
  }
  starts * 20 < limit
}

///|
/// Adaptive convenience wrapper for StringView input. Small or
/// string-dominated inputs delegate to core/json; structurally dense large
/// inputs include UTF-8 encoding and use the indexed parser. Use parse(Bytes)
/// when the caller already owns UTF-8 bytes.
pub fn parse_string(
  input : StringView,
  max_nesting_depth? : Int = 1024,
) -> Json raise ParseError {
  if max_nesting_depth == 1024 && json_string_should_use_core(input) {
    return @core_json.parse(input) catch {
      @core_json.DepthLimitExceeded => raise DepthLimitExceeded
      _ => raise InvalidSyntax
    }
  }
  parse(@utf8.encode(input), max_nesting_depth~)
}

///|
/// Return whether input is valid UTF-8 JSON for this parser.
pub fn valid(input : Bytes) -> Bool {
  try {
    parse(input) |> ignore
    true
  } catch {
    _ => false
  }
}
