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
fn i8x16_splat(x : Int) -> V128 =
  #|(func (param i32) (result v128) local.get 0 i8x16.splat)

///|
fn i8x16_eq(a : V128, b : V128) -> V128 =
  #|(func (param v128) (param v128) (result v128) local.get 0 local.get 1 i8x16.eq)

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
#borrow(bytes)
fn bytes_i8x16_load(bytes : Bytes, i : Int) -> V128 =
  #|(func (param i32) (param i32) (result v128) local.get 0 local.get 1 i32.add v128.load)

///|
fn floor_i8x16_len(len : Int) -> Int {
  len / 16 * 16
}
