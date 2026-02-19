///|
fn make_int_array(vals : Array[Int]) -> FixedArray[Int] {
  let arr = FixedArray::make(vals.length(), 0)
  for i, v in vals {
    arr[i] = v
  }
  arr
}

///|
fn make_byte_array(vals : Array[Int]) -> FixedArray[Byte] {
  let arr = FixedArray::make(vals.length(), b'\x00')
  for i, v in vals {
    arr[i] = v.to_byte()
  }
  arr
}

// --- scalar_sum ---

///|
test "scalar_sum empty" {
  let arr : FixedArray[Int] = FixedArray::make(0, 0)
  inspect(scalar_sum(arr), content="0")
}

///|
test "scalar_sum basic" {
  let arr = make_int_array([1, 2, 3, 4, 5])
  inspect(scalar_sum(arr), content="15")
}

///|
test "scalar_sum with negatives" {
  let arr = make_int_array([10, -3, 7, -2])
  inspect(scalar_sum(arr), content="12")
}

// --- simd_sum ---

///|
test "simd_sum empty" {
  let arr : FixedArray[Int] = FixedArray::make(0, 0)
  inspect(simd_sum(arr), content="0")
}

///|
test "simd_sum basic" {
  let arr = make_int_array([1, 2, 3, 4, 5, 6, 7, 8])
  inspect(simd_sum(arr), content="36")
}

///|
test "simd_sum non-aligned" {
  let arr = make_int_array([1, 2, 3, 4, 5])
  inspect(simd_sum(arr), content="15")
}

///|
test "simd_sum matches scalar" {
  let arr = make_int_array([10, 20, 30, 40, 50, 60, 70])
  inspect(simd_sum(arr) == scalar_sum(arr), content="true")
}

// --- scalar_dot_product ---

///|
test "scalar_dot_product basic" {
  let a = make_int_array([1, 2, 3])
  let b = make_int_array([4, 5, 6])
  inspect(scalar_dot_product(a, b), content="32")
}

// --- simd_dot_product ---

///|
test "simd_dot_product basic" {
  let a = make_int_array([1, 2, 3, 4])
  let b = make_int_array([5, 6, 7, 8])
  inspect(simd_dot_product(a, b), content="70")
}

///|
test "simd_dot_product non-aligned" {
  let a = make_int_array([1, 2, 3, 4, 5])
  let b = make_int_array([2, 3, 4, 5, 6])
  inspect(simd_dot_product(a, b) == scalar_dot_product(a, b), content="true")
}

// --- scalar_add ---

///|
test "scalar_add basic" {
  let a = make_int_array([1, 2, 3, 4])
  let b = make_int_array([10, 20, 30, 40])
  let out = FixedArray::make(4, 0)
  scalar_add(a, b, out)
  inspect(out[0], content="11")
  inspect(out[3], content="44")
}

// --- simd_add ---

///|
test "simd_add basic" {
  let a = make_int_array([1, 2, 3, 4, 5, 6, 7, 8])
  let b = make_int_array([10, 20, 30, 40, 50, 60, 70, 80])
  let out = FixedArray::make(8, 0)
  simd_add(a, b, out)
  inspect(out[0], content="11")
  inspect(out[7], content="88")
}

///|
test "simd_add non-aligned" {
  let a = make_int_array([1, 2, 3, 4, 5])
  let b = make_int_array([10, 20, 30, 40, 50])
  let out = FixedArray::make(5, 0)
  let expected = FixedArray::make(5, 0)
  simd_add(a, b, out)
  scalar_add(a, b, expected)
  for i in 0..<5 {
    assert_eq(out[i], expected[i])
  }
}

// --- scalar_adler32 ---

///|
test "scalar_adler32 empty" {
  let data : FixedArray[Byte] = FixedArray::make(0, b'\x00')
  inspect(scalar_adler32(data), content="1")
}

///|
test "scalar_adler32 wikipedia" {
  // "Wikipedia" -> Adler-32 = 0x11E60398
  let data = make_byte_array([87, 105, 107, 105, 112, 101, 100, 105, 97])
  inspect(scalar_adler32(data), content="300286872")
}

// --- simd_adler32 ---

///|
test "simd_adler32 matches scalar" {
  let data = make_byte_array([87, 105, 107, 105, 112, 101, 100, 105, 97])
  inspect(simd_adler32(data) == scalar_adler32(data), content="true")
}

///|
test "simd_adler32 empty" {
  let data : FixedArray[Byte] = FixedArray::make(0, b'\x00')
  inspect(simd_adler32(data), content="1")
}
