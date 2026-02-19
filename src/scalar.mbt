///|
/// Sum all elements of an integer array (scalar).
pub fn scalar_sum(arr : FixedArray[Int]) -> Int {
  let len = arr.length()
  for i = 0, acc = 0; i < len; {
    continue i + 1, acc + arr[i]
  } nobreak {
    acc
  }
}

///|
/// Dot product of two integer arrays (scalar).
pub fn scalar_dot_product(a : FixedArray[Int], b : FixedArray[Int]) -> Int {
  let len = a.length()
  for i = 0, acc = 0; i < len; {
    continue i + 1, acc + a[i] * b[i]
  } nobreak {
    acc
  }
}

///|
/// Element-wise addition of two arrays, storing result in out (scalar).
pub fn scalar_add(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  for i in 0..<len {
    out[i] = a[i] + b[i]
  }
}

///|
/// Adler-32 checksum (scalar).
pub fn scalar_adler32(data : FixedArray[Byte]) -> UInt {
  let mod_adler = 65521U
  let len = data.length()
  for i = 0, a = 1U, b = 0U; i < len; {
    let new_a = (a + data[i].to_int().reinterpret_as_uint()) % mod_adler
    let new_b = (b + new_a) % mod_adler
    continue i + 1, new_a, new_b
  } nobreak {
    (b << 16) | a
  }
}
