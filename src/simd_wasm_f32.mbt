///|
#borrow(a, b, out)
fn add_f32_v128(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load f32x4.add v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add f32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add f32.load f32.add f32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
pub fn add_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  if n == 0 {
    return
  }
  add_f32_v128(a, b, out, n)
}

///|
#borrow(a, b, out)
fn sub_f32_v128(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load f32x4.sub v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add f32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add f32.load f32.sub f32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
pub fn sub_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  if n == 0 {
    return
  }
  sub_f32_v128(a, b, out, n)
}

///|
#borrow(a, b, out)
fn div_f32_v128(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load f32x4.div v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add f32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add f32.load f32.div f32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
pub fn div_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  if n == 0 {
    return
  }
  div_f32_v128(a, b, out, n)
}

///|
#borrow(a, out)
fn sqrt_f32_v128(a : FixedArray[Byte], out : FixedArray[Byte], n : Int) -> Unit =
  #|(func (param i32) (param i32) (param i32) (local i32) (local i32) local.get 2 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 1 local.get 3 i32.const 2 i32.shl i32.add local.get 0 local.get 3 i32.const 2 i32.shl i32.add v128.load f32x4.sqrt v128.store local.get 3 i32.const 4 i32.add local.set 3 br 0 end end block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 1 local.get 3 i32.const 2 i32.shl i32.add local.get 0 local.get 3 i32.const 2 i32.shl i32.add f32.load f32.sqrt f32.store local.get 3 i32.const 1 i32.add local.set 3 br 0 end end)

///|
pub fn sqrt_f32(a : FixedArray[Byte], out : FixedArray[Byte], n : Int) -> Unit {
  if n == 0 {
    return
  }
  sqrt_f32_v128(a, out, n)
}

///|
/// min element-wise — uses select+f32.lt for scalar tail because the
/// Dwarfsm parser rejects `f32.min` (same hole as `f64.min`).
#borrow(a, b, out)
fn min_elem_f32_v128(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load f32x4.min v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add f32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add f32.load local.get 0 local.get 4 i32.const 2 i32.shl i32.add f32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add f32.load f32.lt select f32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
pub fn min_elem_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  if n == 0 {
    return
  }
  min_elem_f32_v128(a, b, out, n)
}

///|
#borrow(a, b, out)
fn max_elem_f32_v128(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load f32x4.max v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add f32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add f32.load local.get 0 local.get 4 i32.const 2 i32.shl i32.add f32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add f32.load f32.gt select f32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
pub fn max_elem_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  if n == 0 {
    return
  }
  max_elem_f32_v128(a, b, out, n)
}

///|
#borrow(a, b, out)
fn mul_f32_v128(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit =
  #|(func (param i32) (param i32) (param i32) (param i32) (local i32) (local i32) local.get 3 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 5 block loop local.get 4 local.get 5 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add v128.load f32x4.mul v128.store local.get 4 i32.const 4 i32.add local.set 4 br 0 end end block loop local.get 4 local.get 3 i32.ge_s br_if 1 local.get 2 local.get 4 i32.const 2 i32.shl i32.add local.get 0 local.get 4 i32.const 2 i32.shl i32.add f32.load local.get 1 local.get 4 i32.const 2 i32.shl i32.add f32.load f32.mul f32.store local.get 4 i32.const 1 i32.add local.set 4 br 0 end end)

///|
pub fn mul_f32(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit {
  if n == 0 {
    return
  }
  mul_f32_v128(a, b, out, n)
}

///|
#borrow(arr)
fn sum_f32_v128(arr : FixedArray[Byte], n : Int) -> Float =
  #|(func (param i32) (param i32) (result f32) (local i32) (local i32) (local f32) (local v128) i32.const 0 f32.convert_i32_s f32x4.splat local.set 5 local.get 1 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 3 block loop local.get 2 local.get 3 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 2 i32.const 2 i32.shl i32.add v128.load f32x4.add local.set 5 local.get 2 i32.const 4 i32.add local.set 2 br 0 end end local.get 5 f32x4.extract_lane 0 local.get 5 f32x4.extract_lane 1 f32.add local.get 5 f32x4.extract_lane 2 f32.add local.get 5 f32x4.extract_lane 3 f32.add local.set 4 block loop local.get 2 local.get 1 i32.ge_s br_if 1 local.get 4 local.get 0 local.get 2 i32.const 2 i32.shl i32.add f32.load f32.add local.set 4 local.get 2 i32.const 1 i32.add local.set 2 br 0 end end local.get 4)

///|
pub fn sum_f32(arr : FixedArray[Byte], n : Int) -> Float {
  if n == 0 {
    return (0.0 : Float)
  }
  sum_f32_v128(arr, n)
}

///|
#borrow(a, b)
fn dot_f32_v128(a : FixedArray[Byte], b : FixedArray[Byte], n : Int) -> Float =
  #|(func (param i32) (param i32) (param i32) (result f32) (local i32) (local i32) (local f32) (local v128) i32.const 0 f32.convert_i32_s f32x4.splat local.set 6 local.get 2 i32.const 2 i32.shr_s i32.const 2 i32.shl local.set 4 block loop local.get 3 local.get 4 i32.ge_s br_if 1 local.get 6 local.get 0 local.get 3 i32.const 2 i32.shl i32.add v128.load local.get 1 local.get 3 i32.const 2 i32.shl i32.add v128.load f32x4.mul f32x4.add local.set 6 local.get 3 i32.const 4 i32.add local.set 3 br 0 end end local.get 6 f32x4.extract_lane 0 local.get 6 f32x4.extract_lane 1 f32.add local.get 6 f32x4.extract_lane 2 f32.add local.get 6 f32x4.extract_lane 3 f32.add local.set 5 block loop local.get 3 local.get 2 i32.ge_s br_if 1 local.get 5 local.get 0 local.get 3 i32.const 2 i32.shl i32.add f32.load local.get 1 local.get 3 i32.const 2 i32.shl i32.add f32.load f32.mul f32.add local.set 5 local.get 3 i32.const 1 i32.add local.set 3 br 0 end end local.get 5)

///|
pub fn dot_f32(a : FixedArray[Byte], b : FixedArray[Byte], n : Int) -> Float {
  if n == 0 {
    return (0.0 : Float)
  }
  dot_f32_v128(a, b, n)
}
