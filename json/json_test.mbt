///|
fn utf8_bytes(input : String) -> Bytes {
  @utf8.encode(input[:])
}

///|
fn assert_same_as_core(input : String) -> Unit raise {
  let got = try! parse(utf8_bytes(input))
  let want = try! @core_json.parse(input)
  @debug.assert_eq(got, want)
}

///|
test "parse UTF-8 JSON with the same value semantics as core/json" {
  for
    input in [
      "null", "true", "false", "0", "-42", "1.25e3", "[1,2,3]", "{\"name\":\"moon\",\"active\":true,\"items\":[1,null,3.5]}",
      "{\"escaped\":\"line\\nquote\\\"slash\\\\unicode\\u65e5\",\"raw\":\"日本語👀\"}",
      "\"\\uD83D\\uDC40\"", "\"\\uD800\"", "\"\\uDC00\"", "\"\\uD800\\u0041\"", "9007199254740993",
      "1.234567890123456789e100",
    ] {
    assert_same_as_core(input)
  }
}

///|
test "parse_string includes UTF-8 encoding convenience path" {
  let input = "{\"message\":\"こんにちは\",\"n\":123}"
  @debug.assert_eq(try! parse_string(input[:]), try! @core_json.parse(input))
}

///|
test "parse_string maps core depth errors to package errors" {
  let rejected = try {
    parse_string("[[]]"[:], max_nesting_depth=1) |> ignore
    false
  } catch {
    DepthLimitExceeded => true
    _ => false
  }
  @debug.assert_eq(rejected, true)
}

///|
test "reject invalid JSON" {
  for
    input in [
      "", "[1,]", "{\"a\":}", "{\"a\" 1}", "01", "1.", "1e+", "true false", "\"unterminated",
      "\"bad\\q\"", "\"line\nbreak\"", "\"ok\\nthen\nbad\"",
    ] {
    let rejected = try {
      parse(utf8_bytes(input)) |> ignore
      false
    } catch {
      _ => true
    }
    @debug.assert_eq(rejected, true)
  }
}

///|
test "valid mirrors parse acceptance" {
  @debug.assert_eq(valid(utf8_bytes("{\"ok\":[1,2,3]}")), true)
  @debug.assert_eq(valid(utf8_bytes("{\"bad\":]")), false)
}

///|
test "preserve negative zero" {
  let value = try! parse(utf8_bytes("-0"))
  match value {
    Number(number, ..) => @debug.assert_eq(1.0 / number, @double.neg_infinity)
    _ => fail("expected a JSON number")
  }
}

///|
test "reject invalid UTF-8 string contents" {
  let input = Bytes::from_array([b'\"', b'\xFF', b'\"'])
  let rejected = try {
    parse(input) |> ignore
    false
  } catch {
    _ => true
  }
  @debug.assert_eq(rejected, true)
}

///|
test "respect maximum nesting depth" {
  let rejected = try {
    parse(utf8_bytes("[[]]"), max_nesting_depth=1) |> ignore
    false
  } catch {
    DepthLimitExceeded => true
    _ => false
  }
  @debug.assert_eq(rejected, true)
}

///|
test "context carving crosses SIMD block boundaries" {
  for
    input in [
      "{\"0123456789abcdef\":1}", "[\"0123456789abc\\\"def\",2]", "[1234567890123456,1234567890123457]",
      "{\"padding-padding\":\"0123456789abcdef\\\\tail\"}",
    ] {
    assert_same_as_core(input)
  }
}

///|
test "escaped quote parity crosses every SIMD lane" {
  for padding in 0..<32 {
    for slash_count in 0..<8 {
      let buf = StringBuilder()
      buf.write_string("[\"")
      for _ in 0..<padding {
        buf.write_char('a')
      }
      for _ in 0..<slash_count {
        buf.write_char('\\')
      }
      buf.write_char('"')
      if (slash_count & 1) == 1 {
        buf.write_string("tail\"")
      }
      buf.write_char(']')
      assert_same_as_core(buf.to_string())
    }
  }
}
