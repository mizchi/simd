///|
fn i32x4_splat(x : Int) -> V128 =
  #|(func (param i32) (result v128) local.get 0 i32x4.splat)

///|
fn i32x4_zero() -> V128 {
  i32x4_splat(0)
}

///|
fn i32x4_add(a : V128, b : V128) -> V128 =
  #|(func (param v128) (param v128) (result v128) local.get 0 local.get 1 i32x4.add)

///|
fn i32x4_sub(a : V128, b : V128) -> V128 =
  #|(func (param v128) (param v128) (result v128) local.get 0 local.get 1 i32x4.sub)

///|
fn i32x4_mul(a : V128, b : V128) -> V128 =
  #|(func (param v128) (param v128) (result v128) local.get 0 local.get 1 i32x4.mul)

///|
fn i32x4_neg(a : V128) -> V128 =
  #|(func (param v128) (result v128) local.get 0 i32x4.neg)

///|
fn i32x4_abs(a : V128) -> V128 =
  #|(func (param v128) (result v128) local.get 0 i32x4.abs)

///|
fn i32x4_min_s(a : V128, b : V128) -> V128 =
  #|(func (param v128) (param v128) (result v128) local.get 0 local.get 1 i32x4.min_s)

///|
fn i32x4_max_s(a : V128, b : V128) -> V128 =
  #|(func (param v128) (param v128) (result v128) local.get 0 local.get 1 i32x4.max_s)

///|
fn i32x4_eq(a : V128, b : V128) -> V128 =
  #|(func (param v128) (param v128) (result v128) local.get 0 local.get 1 i32x4.eq)

///|
fn i32x4_ne(a : V128, b : V128) -> V128 =
  #|(func (param v128) (param v128) (result v128) local.get 0 local.get 1 i32x4.ne)

///|
fn i32x4_lt_s(a : V128, b : V128) -> V128 =
  #|(func (param v128) (param v128) (result v128) local.get 0 local.get 1 i32x4.lt_s)

///|
fn i32x4_gt_s(a : V128, b : V128) -> V128 =
  #|(func (param v128) (param v128) (result v128) local.get 0 local.get 1 i32x4.gt_s)

///|
fn i32x4_bitmask(v : V128) -> Int =
  #|(func (param v128) (result i32) local.get 0 i32x4.bitmask)

///|
fn v128_bitselect(true_v : V128, false_v : V128, mask : V128) -> V128 =
  #|(func (param v128) (param v128) (param v128) (result v128) local.get 0 local.get 1 local.get 2 v128.bitselect)

///|
fn v128_and(a : V128, b : V128) -> V128 =
  #|(func (param v128) (param v128) (result v128) local.get 0 local.get 1 v128.and)

///|
fn i32x4_extract_lane0(v : V128) -> Int =
  #|(func (param v128) (result i32) local.get 0 i32x4.extract_lane 0)

///|
fn i32x4_extract_lane1(v : V128) -> Int =
  #|(func (param v128) (result i32) local.get 0 i32x4.extract_lane 1)

///|
fn i32x4_extract_lane2(v : V128) -> Int =
  #|(func (param v128) (result i32) local.get 0 i32x4.extract_lane 2)

///|
fn i32x4_extract_lane3(v : V128) -> Int =
  #|(func (param v128) (result i32) local.get 0 i32x4.extract_lane 3)

///|
fn i32x4_lane_sum(v : V128) -> Int {
  i32x4_extract_lane0(v) +
  i32x4_extract_lane1(v) +
  i32x4_extract_lane2(v) +
  i32x4_extract_lane3(v)
}

///|
fn i32x4_lane_product(v : V128) -> Int {
  i32x4_extract_lane0(v) *
  i32x4_extract_lane1(v) *
  i32x4_extract_lane2(v) *
  i32x4_extract_lane3(v)
}

///|
fn i32x4_lane_min(v : V128) -> Int {
  let a = i32x4_extract_lane0(v)
  let b = i32x4_extract_lane1(v)
  let c = i32x4_extract_lane2(v)
  let d = i32x4_extract_lane3(v)
  i32_min_value(i32_min_value(a, b), i32_min_value(c, d))
}

///|
fn i32x4_lane_max(v : V128) -> Int {
  let a = i32x4_extract_lane0(v)
  let b = i32x4_extract_lane1(v)
  let c = i32x4_extract_lane2(v)
  let d = i32x4_extract_lane3(v)
  i32_max_value(i32_max_value(a, b), i32_max_value(c, d))
}

///|
fn i32x4_count_nonzero(v : V128) -> Int {
  i32x4_bitmask(i32x4_ne(v, i32x4_zero())).popcnt()
}

///|
#borrow(arr)
fn fixed_i32x4_load(arr : FixedArray[Int], i : Int) -> V128 =
  #|(func (param i32) (param i32) (result v128) local.get 0 local.get 1 i32.const 2 i32.shl i32.add v128.load)

///|
#borrow(out)
fn fixed_i32x4_store(out : FixedArray[Int], i : Int, v : V128) -> Unit =
  #|(func (param i32) (param i32) (param v128) local.get 0 local.get 1 i32.const 2 i32.shl i32.add local.get 2 v128.store)

///|
fn floor_i32x4_len(len : Int) -> Int {
  len / 4 * 4
}

///|
fn i32_min_value(a : Int, b : Int) -> Int {
  if a < b {
    a
  } else {
    b
  }
}

///|
fn i32_max_value(a : Int, b : Int) -> Int {
  if a > b {
    a
  } else {
    b
  }
}

///|
fn bool_mask(value : Bool) -> Int {
  if value {
    -1
  } else {
    0
  }
}

///|
fn f64x2_splat(x : Double) -> V128 =
  #|(func (param f64) (result v128) local.get 0 f64x2.splat)

///|
fn f64x2_zero() -> V128 =
  #|(func (result v128) i32.const 0 f64.convert_i32_s f64x2.splat)

///|
fn f64x2_add(a : V128, b : V128) -> V128 =
  #|(func (param v128) (param v128) (result v128) local.get 0 local.get 1 f64x2.add)

///|
fn f64x2_sub(a : V128, b : V128) -> V128 =
  #|(func (param v128) (param v128) (result v128) local.get 0 local.get 1 f64x2.sub)

///|
fn f64x2_mul(a : V128, b : V128) -> V128 =
  #|(func (param v128) (param v128) (result v128) local.get 0 local.get 1 f64x2.mul)

///|
fn f64x2_div(a : V128, b : V128) -> V128 =
  #|(func (param v128) (param v128) (result v128) local.get 0 local.get 1 f64x2.div)

///|
fn f64x2_sqrt(a : V128) -> V128 =
  #|(func (param v128) (result v128) local.get 0 f64x2.sqrt)

///|
fn f64x2_min(a : V128, b : V128) -> V128 =
  #|(func (param v128) (param v128) (result v128) local.get 0 local.get 1 f64x2.min)

///|
fn f64x2_max(a : V128, b : V128) -> V128 =
  #|(func (param v128) (param v128) (result v128) local.get 0 local.get 1 f64x2.max)

///|
fn f64x2_extract_lane0(v : V128) -> Double =
  #|(func (param v128) (result f64) local.get 0 f64x2.extract_lane 0)

///|
fn f64x2_extract_lane1(v : V128) -> Double =
  #|(func (param v128) (result f64) local.get 0 f64x2.extract_lane 1)

///|
fn f64x2_lane_sum(v : V128) -> Double {
  f64x2_extract_lane0(v) + f64x2_extract_lane1(v)
}

///|
fn f64x2_lane_min(v : V128) -> Double {
  let a = f64x2_extract_lane0(v)
  let b = f64x2_extract_lane1(v)
  if a < b {
    a
  } else {
    b
  }
}

///|
fn f64x2_lane_max(v : V128) -> Double {
  let a = f64x2_extract_lane0(v)
  let b = f64x2_extract_lane1(v)
  if a > b {
    a
  } else {
    b
  }
}

///|
#borrow(arr)
fn fixed_f64x2_load(arr : FixedArray[Double], i : Int) -> V128 =
  #|(func (param i32) (param i32) (result v128) local.get 0 local.get 1 i32.const 3 i32.shl i32.add v128.load)

///|
#borrow(out)
fn fixed_f64x2_store(out : FixedArray[Double], i : Int, v : V128) -> Unit =
  #|(func (param i32) (param i32) (param v128) local.get 0 local.get 1 i32.const 3 i32.shl i32.add local.get 2 v128.store)

///|
fn floor_f64x2_len(len : Int) -> Int {
  len / 2 * 2
}

///|
fn f32x4_zero() -> V128 =
  #|(func (result v128) i32.const 0 f32.convert_i32_s f32x4.splat)

///|
fn f32x4_add(a : V128, b : V128) -> V128 =
  #|(func (param v128) (param v128) (result v128) local.get 0 local.get 1 f32x4.add)

///|
fn f32x4_sub(a : V128, b : V128) -> V128 =
  #|(func (param v128) (param v128) (result v128) local.get 0 local.get 1 f32x4.sub)

///|
fn f32x4_mul(a : V128, b : V128) -> V128 =
  #|(func (param v128) (param v128) (result v128) local.get 0 local.get 1 f32x4.mul)

///|
fn f32x4_div(a : V128, b : V128) -> V128 =
  #|(func (param v128) (param v128) (result v128) local.get 0 local.get 1 f32x4.div)

///|
fn f32x4_sqrt(a : V128) -> V128 =
  #|(func (param v128) (result v128) local.get 0 f32x4.sqrt)

///|
fn f32x4_min(a : V128, b : V128) -> V128 =
  #|(func (param v128) (param v128) (result v128) local.get 0 local.get 1 f32x4.min)

///|
fn f32x4_max(a : V128, b : V128) -> V128 =
  #|(func (param v128) (param v128) (result v128) local.get 0 local.get 1 f32x4.max)

///|
fn f32x4_extract_lane0(v : V128) -> Float =
  #|(func (param v128) (result f32) local.get 0 f32x4.extract_lane 0)

///|
fn f32x4_extract_lane1(v : V128) -> Float =
  #|(func (param v128) (result f32) local.get 0 f32x4.extract_lane 1)

///|
fn f32x4_extract_lane2(v : V128) -> Float =
  #|(func (param v128) (result f32) local.get 0 f32x4.extract_lane 2)

///|
fn f32x4_extract_lane3(v : V128) -> Float =
  #|(func (param v128) (result f32) local.get 0 f32x4.extract_lane 3)

///|
fn f32x4_lane_sum(v : V128) -> Float {
  f32x4_extract_lane0(v) +
  f32x4_extract_lane1(v) +
  f32x4_extract_lane2(v) +
  f32x4_extract_lane3(v)
}

///|
#borrow(arr)
fn fixed_f32x4_load(arr : FixedArray[Byte], i : Int) -> V128 =
  #|(func (param i32) (param i32) (result v128) local.get 0 local.get 1 i32.const 2 i32.shl i32.add v128.load)

///|
#borrow(out)
fn fixed_f32x4_store(out : FixedArray[Byte], i : Int, v : V128) -> Unit =
  #|(func (param i32) (param i32) (param v128) local.get 0 local.get 1 i32.const 2 i32.shl i32.add local.get 2 v128.store)

///|
fn floor_f32x4_len(len : Int) -> Int {
  len / 4 * 4
}

///|
fn i8x16_splat(x : Int) -> V128 =
  #|(func (param i32) (result v128) local.get 0 i8x16.splat)

///|
fn i8x16_eq(a : V128, b : V128) -> V128 =
  #|(func (param v128) (param v128) (result v128) local.get 0 local.get 1 i8x16.eq)

///|
fn i8x16_ge_s(a : V128, b : V128) -> V128 =
  #|(func (param v128) (param v128) (result v128) local.get 0 local.get 1 i8x16.ge_s)

///|
fn i8x16_le_s(a : V128, b : V128) -> V128 =
  #|(func (param v128) (param v128) (result v128) local.get 0 local.get 1 i8x16.le_s)

///|
fn i8x16_add(a : V128, b : V128) -> V128 =
  #|(func (param v128) (param v128) (result v128) local.get 0 local.get 1 i8x16.add)

///|
fn i8x16_sub(a : V128, b : V128) -> V128 =
  #|(func (param v128) (param v128) (result v128) local.get 0 local.get 1 i8x16.sub)

///|
fn i8x16_bitmask(v : V128) -> Int =
  #|(func (param v128) (result i32) local.get 0 i8x16.bitmask)

///|
fn i8x16_popcnt_sum(v : V128) -> V128 =
  #|(func (param v128) (result v128) local.get 0 i8x16.popcnt i16x8.extadd_pairwise_i8x16_u i32x4.extadd_pairwise_i16x8_u)

///|
#borrow(arr)
fn fixed_i8x16_load(arr : FixedArray[Byte], i : Int) -> V128 =
  #|(func (param i32) (param i32) (result v128) local.get 0 local.get 1 i32.add v128.load)

///|
#borrow(out)
fn fixed_i8x16_store(out : FixedArray[Byte], i : Int, v : V128) -> Unit =
  #|(func (param i32) (param i32) (param v128) local.get 0 local.get 1 i32.add local.get 2 v128.store)

///|
fn floor_i8x16_len(len : Int) -> Int {
  len / 16 * 16
}
