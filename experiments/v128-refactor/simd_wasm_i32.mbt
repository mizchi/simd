///|
fn sum_i32_v128(arr : FixedArray[Int], len : Int) -> Int {
  let vec_len = floor_i32x4_len(len)
  let acc = for i = 0, acc = i32x4_zero(); i < vec_len; {
    continue i + 4, i32x4_add(acc, fixed_i32x4_load(arr, i))
  } nobreak {
    acc
  }
  for i in vec_len..<len; sum = i32x4_lane_sum(acc) {
    continue sum + arr[i]
  } nobreak {
    sum
  }
}

///|
/// SIMD sum - wasm target: v128 i32x4.add accumulator with scalar tail.
pub fn sum_i32(arr : FixedArray[Int]) -> Int {
  let len = arr.length()
  if len == 0 {
    return 0
  }
  sum_i32_v128(arr, len)
}

///|
fn dot_i32_v128(a : FixedArray[Int], b : FixedArray[Int], len : Int) -> Int {
  let vec_len = floor_i32x4_len(len)
  let acc = for i = 0, acc = i32x4_zero(); i < vec_len; {
    let product = i32x4_mul(fixed_i32x4_load(a, i), fixed_i32x4_load(b, i))
    continue i + 4, i32x4_add(acc, product)
  } nobreak {
    acc
  }
  for i in vec_len..<len; sum = i32x4_lane_sum(acc) {
    continue sum + a[i] * b[i]
  } nobreak {
    sum
  }
}

///|
/// SIMD dot product - wasm target: i32x4.mul + i32x4.add, scalar tail.
pub fn dot_i32(a : FixedArray[Int], b : FixedArray[Int]) -> Int {
  let len = a.length()
  if len == 0 {
    return 0
  }
  dot_i32_v128(a, b, len)
}

///|
#borrow(a, b, out)
fn add_i32_v128(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit {
  let vec_len = floor_i32x4_len(len)
  for i = 0; i < vec_len; {
    fixed_i32x4_store(
      out,
      i,
      i32x4_add(fixed_i32x4_load(a, i), fixed_i32x4_load(b, i)),
    )
    continue i + 4
  }
  for i in vec_len..<len {
    out[i] = a[i] + b[i]
  }
}

///|
/// SIMD element-wise add - wasm target: i32x4.add with scalar tail.
pub fn add_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  add_i32_v128(a, b, out, len)
}

///|
fn min_i32_v128(arr : FixedArray[Int], len : Int) -> Int {
  let vec_len = floor_i32x4_len(len)
  let acc = for i = 0, acc = i32x4_splat(arr[0]); i < vec_len; {
    continue i + 4, i32x4_min_s(acc, fixed_i32x4_load(arr, i))
  } nobreak {
    acc
  }
  for i in vec_len..<len; best = i32x4_lane_min(acc) {
    continue i32_min_value(best, arr[i])
  } nobreak {
    best
  }
}

///|
/// SIMD min - wasm target: v128 i32x4.min_s with scalar reduce.
pub fn min_i32(arr : FixedArray[Int]) -> Int {
  let len = arr.length()
  if len == 0 {
    return 0
  }
  min_i32_v128(arr, len)
}

///|
fn max_i32_v128(arr : FixedArray[Int], len : Int) -> Int {
  let vec_len = floor_i32x4_len(len)
  let acc = for i = 0, acc = i32x4_splat(arr[0]); i < vec_len; {
    continue i + 4, i32x4_max_s(acc, fixed_i32x4_load(arr, i))
  } nobreak {
    acc
  }
  for i in vec_len..<len; best = i32x4_lane_max(acc) {
    continue i32_max_value(best, arr[i])
  } nobreak {
    best
  }
}

///|
/// SIMD max - wasm target: v128 i32x4.max_s with scalar reduce.
pub fn max_i32(arr : FixedArray[Int]) -> Int {
  let len = arr.length()
  if len == 0 {
    return 0
  }
  max_i32_v128(arr, len)
}

///|
/// Find the first index in `arr` where the element equals `needle`. Same
/// "no early break, track found via select" pattern as `find_byte`,
/// but on i32x4 lanes (chunk = 4 elements at byte stride 4).
fn find_int_i32_v128(arr : FixedArray[Int], len : Int, needle : Int) -> Int {
  let vec_len = floor_i32x4_len(len)
  let needle_v = i32x4_splat(needle)
  let found = for i = 0, found = -1; i < vec_len; {
    let mask = i32x4_bitmask(i32x4_eq(fixed_i32x4_load(arr, i), needle_v))
    continue i + 4,
      if found == -1 && mask != 0 {
        i + mask.ctz()
      } else {
        found
      }
  } nobreak {
    found
  }
  for i in vec_len..<len; found = found {
    continue if found == -1 && arr[i] == needle { i } else { found }
  } nobreak {
    found
  }
}

///|
/// 2-pass SIMD argmin: min_i32 for the value, then SIMD find for the first
/// matching index.
pub fn argmin_i32(arr : FixedArray[Int]) -> Int {
  let len = arr.length()
  if len == 0 {
    return -1
  }
  find_int_i32_v128(arr, len, min_i32_v128(arr, len))
}

///|
pub fn argmax_i32(arr : FixedArray[Int]) -> Int {
  let len = arr.length()
  if len == 0 {
    return -1
  }
  find_int_i32_v128(arr, len, max_i32_v128(arr, len))
}

///|
/// Index of the first element equal to `needle`, or -1 if absent.
/// Reuses the `find_int_i32_v128` kernel (i32x4.eq + bitmask + ctz, no early
/// break) that backs argmin / argmax.
pub fn find_i32(arr : FixedArray[Int], needle : Int) -> Int {
  let len = arr.length()
  if len == 0 {
    return -1
  }
  find_int_i32_v128(arr, len, needle)
}

///|
/// Fill `arr[0..len)` with `value`: i32x4.splat once, then v128.store per
/// 4-element chunk + i32.store scalar tail.
fn fill_i32_v128(arr : FixedArray[Int], len : Int, value : Int) -> Unit {
  let vec_len = floor_i32x4_len(len)
  let value_v = i32x4_splat(value)
  for i = 0; i < vec_len; {
    fixed_i32x4_store(arr, i, value_v)
    continue i + 4
  }
  for i in vec_len..<len {
    arr[i] = value
  }
}

///|
/// SIMD fill - wasm target: i32x4.splat + v128.store.
pub fn fill_i32(arr : FixedArray[Int], value : Int) -> Unit {
  let len = arr.length()
  if len == 0 {
    return
  }
  fill_i32_v128(arr, len, value)
}

///|
fn saxpy_i32_v128(
  k : Int,
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit {
  let vec_len = floor_i32x4_len(len)
  let k_v = i32x4_splat(k)
  for i = 0; i < vec_len; {
    fixed_i32x4_store(
      out,
      i,
      i32x4_add(i32x4_mul(k_v, fixed_i32x4_load(a, i)), fixed_i32x4_load(b, i)),
    )
    continue i + 4
  }
  for i in vec_len..<len {
    out[i] = k * a[i] + b[i]
  }
}

///|
/// SIMD saxpy: out[i] = k * a[i] + b[i] - wasm target: v128 i32x4.mul + add.
pub fn saxpy_i32(
  k : Int,
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  saxpy_i32_v128(k, a, b, out, len)
}

///|
fn prod_i32_v128(arr : FixedArray[Int], len : Int) -> Int {
  let vec_len = floor_i32x4_len(len)
  let acc = for i = 0, acc = i32x4_splat(1); i < vec_len; {
    continue i + 4, i32x4_mul(acc, fixed_i32x4_load(arr, i))
  } nobreak {
    acc
  }
  for i in vec_len..<len; product = i32x4_lane_product(acc) {
    continue product * arr[i]
  } nobreak {
    product
  }
}

///|
pub fn prod_i32(arr : FixedArray[Int]) -> Int {
  let len = arr.length()
  if len == 0 {
    return 1
  }
  prod_i32_v128(arr, len)
}

///|
fn count_nonzero_i32_v128(arr : FixedArray[Int], len : Int) -> Int {
  let vec_len = floor_i32x4_len(len)
  let count = for i = 0, count = 0; i < vec_len; {
    continue i + 4, count + i32x4_count_nonzero(fixed_i32x4_load(arr, i))
  } nobreak {
    count
  }
  for i in vec_len..<len; count = count {
    continue count + (if arr[i] != 0 { 1 } else { 0 })
  } nobreak {
    count
  }
}

///|
pub fn count_nonzero_i32(arr : FixedArray[Int]) -> Int {
  let len = arr.length()
  if len == 0 {
    return 0
  }
  count_nonzero_i32_v128(arr, len)
}

///|
/// any/all over Int (treats lane != 0 as true).
fn any_i32_v128(arr : FixedArray[Int], len : Int) -> Bool {
  let vec_len = floor_i32x4_len(len)
  let found = for i = 0, found = false; i < vec_len; {
    let mask = i32x4_bitmask(i32x4_ne(fixed_i32x4_load(arr, i), i32x4_zero()))
    continue i + 4, found || mask != 0
  } nobreak {
    found
  }
  if found {
    true
  } else {
    for i in vec_len..<len; found = false {
      continue found || arr[i] != 0
    } nobreak {
      found
    }
  }
}

///|
pub fn any_i32(arr : FixedArray[Int]) -> Bool {
  let len = arr.length()
  if len == 0 {
    return false
  }
  any_i32_v128(arr, len)
}

///|
fn all_i32_v128(arr : FixedArray[Int], len : Int) -> Bool {
  let vec_len = floor_i32x4_len(len)
  let seen_zero = for i = 0, seen_zero = false; i < vec_len; {
    let mask = i32x4_bitmask(i32x4_eq(fixed_i32x4_load(arr, i), i32x4_zero()))
    continue i + 4, seen_zero || mask != 0
  } nobreak {
    seen_zero
  }
  if seen_zero {
    false
  } else {
    for i in vec_len..<len; all = true {
      continue all && arr[i] != 0
    } nobreak {
      all
    }
  }
}

///|
pub fn all_i32(arr : FixedArray[Int]) -> Bool {
  let len = arr.length()
  if len == 0 {
    return true
  }
  all_i32_v128(arr, len)
}

///|
/// Hillis-Steele prefix sum on i32x4: two stages of (shuffle-left + add) per
/// chunk, then add the running sum from the previous chunk via splat.
#borrow(arr, out)
fn cumsum_i32_v128(
  arr : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (local i32) (local i32) (local i32) (local v128) (local v128) (local v128) (local v128) i32.const 0 i32x4.splat local.set 6 i32.const 0 local.set 5 local.get 2 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 0 local.get 3 i32.const 2 i32.shl i32.add v128.load local.set 7 local.get 6 local.get 7 i8x16.shuffle 0 1 2 3 16 17 18 19 20 21 22 23 24 25 26 27 local.get 7 i32x4.add local.set 8 local.get 6 local.get 8 i8x16.shuffle 0 1 2 3 4 5 6 7 16 17 18 19 20 21 22 23 local.get 8 i32x4.add local.set 9 local.get 5 i32x4.splat local.get 9 i32x4.add local.set 7 local.get 1 local.get 3 i32.const 2 i32.shl i32.add local.get 7 v128.store local.get 7 i32x4.extract_lane 3 local.set 5 local.get 3 i32.const 4 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 3 i32.const 2 i32.shl i32.add i32.load i32.add local.set 5 local.get 1 local.get 3 i32.const 2 i32.shl i32.add local.get 5 i32.store local.get 3 i32.const 1 i32.add local.set 3 br 0 end end)

///|
pub fn cumsum_i32(arr : FixedArray[Int], out : FixedArray[Int]) -> Unit {
  let len = arr.length()
  if len == 0 {
    return
  }
  cumsum_i32_v128(arr, out, len)
}

///|
/// Hillis-Steele prefix product on i32x4. Same shape as cumsum but the
/// "fill" lanes are 1 (identity for mul), the running accumulator starts at
/// 1, and chunks combine via mul.
#borrow(arr, out)
fn cumprod_i32_v128(
  arr : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (local i32) (local i32) (local i32) (local v128) (local v128) (local v128) (local v128) i32.const 1 i32x4.splat local.set 6 i32.const 1 local.set 5 local.get 2 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 0 local.get 3 i32.const 2 i32.shl i32.add v128.load local.set 7 local.get 6 local.get 7 i8x16.shuffle 0 1 2 3 16 17 18 19 20 21 22 23 24 25 26 27 local.get 7 i32x4.mul local.set 8 local.get 6 local.get 8 i8x16.shuffle 0 1 2 3 4 5 6 7 16 17 18 19 20 21 22 23 local.get 8 i32x4.mul local.set 9 local.get 5 i32x4.splat local.get 9 i32x4.mul local.set 7 local.get 1 local.get 3 i32.const 2 i32.shl i32.add local.get 7 v128.store local.get 7 i32x4.extract_lane 3 local.set 5 local.get 3 i32.const 4 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 3 i32.const 2 i32.shl i32.add i32.load i32.mul local.set 5 local.get 1 local.get 3 i32.const 2 i32.shl i32.add local.get 5 i32.store local.get 3 i32.const 1 i32.add local.set 3 br 0 end end)

///|
pub fn cumprod_i32(arr : FixedArray[Int], out : FixedArray[Int]) -> Unit {
  let len = arr.length()
  if len == 0 {
    return
  }
  cumprod_i32_v128(arr, out, len)
}

///|
fn sub_i32_v128(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit {
  let vec_len = floor_i32x4_len(len)
  for i = 0; i < vec_len; {
    fixed_i32x4_store(
      out,
      i,
      i32x4_sub(fixed_i32x4_load(a, i), fixed_i32x4_load(b, i)),
    )
    continue i + 4
  }
  for i in vec_len..<len {
    out[i] = a[i] - b[i]
  }
}

///|
pub fn sub_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  sub_i32_v128(a, b, out, len)
}

///|
fn mul_i32_v128(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit {
  let vec_len = floor_i32x4_len(len)
  for i = 0; i < vec_len; {
    fixed_i32x4_store(
      out,
      i,
      i32x4_mul(fixed_i32x4_load(a, i), fixed_i32x4_load(b, i)),
    )
    continue i + 4
  }
  for i in vec_len..<len {
    out[i] = a[i] * b[i]
  }
}

///|
pub fn mul_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  mul_i32_v128(a, b, out, len)
}

///|
fn neg_i32_v128(a : FixedArray[Int], out : FixedArray[Int], len : Int) -> Unit {
  let vec_len = floor_i32x4_len(len)
  for i = 0; i < vec_len; {
    fixed_i32x4_store(out, i, i32x4_neg(fixed_i32x4_load(a, i)))
    continue i + 4
  }
  for i in vec_len..<len {
    out[i] = -a[i]
  }
}

///|
pub fn neg_i32(a : FixedArray[Int], out : FixedArray[Int]) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  neg_i32_v128(a, out, len)
}

///|
fn abs_i32_v128(a : FixedArray[Int], out : FixedArray[Int], len : Int) -> Unit {
  let vec_len = floor_i32x4_len(len)
  for i = 0; i < vec_len; {
    fixed_i32x4_store(out, i, i32x4_abs(fixed_i32x4_load(a, i)))
    continue i + 4
  }
  for i in vec_len..<len {
    out[i] = if a[i] < 0 { -a[i] } else { a[i] }
  }
}

///|
pub fn abs_i32(a : FixedArray[Int], out : FixedArray[Int]) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  abs_i32_v128(a, out, len)
}

///|
fn min_elem_i32_v128(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit {
  let vec_len = floor_i32x4_len(len)
  for i = 0; i < vec_len; {
    fixed_i32x4_store(
      out,
      i,
      i32x4_min_s(fixed_i32x4_load(a, i), fixed_i32x4_load(b, i)),
    )
    continue i + 4
  }
  for i in vec_len..<len {
    out[i] = i32_min_value(a[i], b[i])
  }
}

///|
pub fn min_elem_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  min_elem_i32_v128(a, b, out, len)
}

///|
fn max_elem_i32_v128(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit {
  let vec_len = floor_i32x4_len(len)
  for i = 0; i < vec_len; {
    fixed_i32x4_store(
      out,
      i,
      i32x4_max_s(fixed_i32x4_load(a, i), fixed_i32x4_load(b, i)),
    )
    continue i + 4
  }
  for i in vec_len..<len {
    out[i] = i32_max_value(a[i], b[i])
  }
}

///|
pub fn max_elem_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  max_elem_i32_v128(a, b, out, len)
}

///|
/// eq → mask: out[i] = (a[i] == b[i]) ? -1 : 0.
/// Vector i32x4.eq already yields 0xFFFFFFFF/0 lanes; scalar tail uses
/// `i32.const 0 ... i32.sub` to sign-extend `0/1` → `0/-1`.
fn eq_i32_v128(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit {
  let vec_len = floor_i32x4_len(len)
  for i = 0; i < vec_len; {
    fixed_i32x4_store(
      out,
      i,
      i32x4_eq(fixed_i32x4_load(a, i), fixed_i32x4_load(b, i)),
    )
    continue i + 4
  }
  for i in vec_len..<len {
    out[i] = bool_mask(a[i] == b[i])
  }
}

///|
pub fn eq_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  eq_i32_v128(a, b, out, len)
}

///|
fn lt_i32_v128(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit {
  let vec_len = floor_i32x4_len(len)
  for i = 0; i < vec_len; {
    fixed_i32x4_store(
      out,
      i,
      i32x4_lt_s(fixed_i32x4_load(a, i), fixed_i32x4_load(b, i)),
    )
    continue i + 4
  }
  for i in vec_len..<len {
    out[i] = bool_mask(a[i] < b[i])
  }
}

///|
pub fn lt_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  lt_i32_v128(a, b, out, len)
}

///|
fn gt_i32_v128(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit {
  let vec_len = floor_i32x4_len(len)
  for i = 0; i < vec_len; {
    fixed_i32x4_store(
      out,
      i,
      i32x4_gt_s(fixed_i32x4_load(a, i), fixed_i32x4_load(b, i)),
    )
    continue i + 4
  }
  for i in vec_len..<len {
    out[i] = bool_mask(a[i] > b[i])
  }
}

///|
pub fn gt_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  gt_i32_v128(a, b, out, len)
}

///|
/// where(mask, a, b, out): out[i] = mask[i] ? a[i] : b[i]. Uses v128.bitselect:
/// bitselect(v1, v2, c) = (v1 & c) | (v2 & ~c), so mask=-1 picks `a`, mask=0 picks `b`.
fn where_i32_v128(
  mask : FixedArray[Int],
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit {
  let vec_len = floor_i32x4_len(len)
  for i = 0; i < vec_len; {
    fixed_i32x4_store(
      out,
      i,
      v128_bitselect(
        fixed_i32x4_load(a, i),
        fixed_i32x4_load(b, i),
        fixed_i32x4_load(mask, i),
      ),
    )
    continue i + 4
  }
  for i in vec_len..<len {
    out[i] = if mask[i] != 0 { a[i] } else { b[i] }
  }
}

///|
pub fn where_i32(
  mask : FixedArray[Int],
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = mask.length()
  if len == 0 {
    return
  }
  where_i32_v128(mask, a, b, out, len)
}

///|
/// SIMD i32 div: wasm has no i32x4.div, so we widen lanes (0,1) and (2,3)
/// to f64x2, do f64x2.div, truncate back via i32x4.trunc_sat_f64x2_s_zero,
/// and recombine with i8x16.shuffle. i32 range fits in f64 mantissa, so
/// the round-trip is exact for representable quotients.
#borrow(a, b, out)
fn div_i32_v128(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
  len : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.set 6 local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load local.set 7 local.get 6 f64x2.convert_low_i32x4_s local.set 8 local.get 7 f64x2.convert_low_i32x4_s local.set 9 local.get 8 local.get 9 f64x2.div i32x4.trunc_sat_f64x2_s_zero local.set 10 local.get 6 local.get 6 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 f64x2.convert_low_i32x4_s local.set 8 local.get 7 local.get 7 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 f64x2.convert_low_i32x4_s local.set 9 local.get 8 local.get 9 f64x2.div i32x4.trunc_sat_f64x2_s_zero local.set 11 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 10 local.get 11 i8x16.shuffle 0 1 2 3 4 5 6 7 16 17 18 19 20 21 22 23 v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add i32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add i32.load i32.div_s i32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
pub fn div_i32(
  a : FixedArray[Int],
  b : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let len = a.length()
  if len == 0 {
    return
  }
  div_i32_v128(a, b, out, len)
}

///|
/// Gather 4 elements per chunk: read 4 indices, scalar-load 4 values, build
/// a v128 via i32x4.replace_lane, then v128.store. wasm SIMD has no real
/// gather instruction so this mostly saves on the store side.
#borrow(arr, indices, out)
fn gather_i32_v128(
  arr : FixedArray[Int],
  indices : FixedArray[Int],
  out : FixedArray[Int],
  n : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add i32.const 0 i32x4.splat local.get 0 local.get 1 local.get 4 i32.const 2 i32.shl i32.add i32.load i32.const 2 i32.shl i32.add i32.load i32x4.replace_lane 0 local.get 0 local.get 1 local.get 4 i32.const 1 i32.add i32.const 2 i32.shl i32.add i32.load i32.const 2 i32.shl i32.add i32.load i32x4.replace_lane 1 local.get 0 local.get 1 local.get 4 i32.const 2 i32.add i32.const 2 i32.shl i32.add i32.load i32.const 2 i32.shl i32.add i32.load i32x4.replace_lane 2 local.get 0 local.get 1 local.get 4 i32.const 3 i32.add i32.const 2 i32.shl i32.add i32.load i32.const 2 i32.shl i32.add i32.load i32x4.replace_lane 3 v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 1 local.get 4 i32.const 2 i32.shl i32.add i32.load i32.const 2 i32.shl i32.add i32.load i32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
pub fn gather_i32(
  arr : FixedArray[Int],
  indices : FixedArray[Int],
  out : FixedArray[Int],
) -> Unit {
  let n = indices.length()
  if n == 0 {
    return
  }
  gather_i32_v128(arr, indices, out, n)
}

///|
/// scatter is fundamentally scalar — random writes can't be vectorized
/// without lock-step / conflict-detection (which wasm SIMD lacks).
pub fn scatter_i32(
  out : FixedArray[Int],
  indices : FixedArray[Int],
  vals : FixedArray[Int],
) -> Unit {
  @internal.scalar_scatter(out, indices, vals)
}
