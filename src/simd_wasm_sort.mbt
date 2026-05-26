///|
/// 4-element sorting network on i32 using v128: 3 shuffle/min/max stages.
/// Stage 1: swap pairs (0,1), (2,3). Stage 2: swap halves (0,2), (1,3).
/// Stage 3: swap middle pair (1,2). Each stage = shuffle + min/max + blend.
#borrow(arr)
fn sort4_i32_v128(arr : FixedArray[Int], off : Int) -> Unit =
  #|(func (param i32) (param i32) (local v128) (local v128) (local v128) (local v128) local.get 0 local.get 1 i32.const 2 i32.shl i32.add v128.load local.set 2 local.get 2 local.get 2 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 3 local.get 2 local.get 3 i32x4.min_s local.set 4 local.get 2 local.get 3 i32x4.max_s local.set 5 local.get 4 local.get 5 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 2 local.get 2 local.get 2 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 3 local.get 2 local.get 3 i32x4.min_s local.set 4 local.get 2 local.get 3 i32x4.max_s local.set 5 local.get 4 local.get 5 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 2 local.get 2 local.get 2 i8x16.shuffle 0 1 2 3 8 9 10 11 4 5 6 7 12 13 14 15 local.set 3 local.get 2 local.get 3 i32x4.min_s local.set 4 local.get 2 local.get 3 i32x4.max_s local.set 5 local.get 4 local.get 5 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 12 13 14 15 local.set 2 local.get 0 local.get 1 i32.const 2 i32.shl i32.add local.get 2 v128.store)

///|
pub fn sort4_i32(arr : FixedArray[Int], off : Int) -> Unit {
  if off + 4 > arr.length() || off < 0 {
    return
  }
  sort4_i32_v128(arr, off)
}

///|
/// Bitonic merge of two sorted 4-element slices (arr[off..off+4] and
/// arr[off+4..off+8]) into one sorted 8-element block in place. Stage 1
/// reverses B and takes element-wise min/max against A; stages 2 and 3 are
/// distance-2 and distance-1 compare-exchanges within each half via
/// shuffle + min_s/max_s + blend shuffle.
#borrow(arr)
fn bitonic_merge8_i32_v128(arr : FixedArray[Int], off : Int) -> Unit =
  #|(func (param i32) (param i32) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) local.get 0 local.get 1 i32.const 2 i32.shl i32.add v128.load local.set 2 local.get 0 local.get 1 i32.const 4 i32.add i32.const 2 i32.shl i32.add v128.load local.set 3 local.get 3 local.get 3 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 4 local.get 2 local.get 4 i32x4.min_s local.set 5 local.get 2 local.get 4 i32x4.max_s local.set 6 local.get 5 local.get 5 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 9 local.get 5 local.get 9 i32x4.min_s local.set 7 local.get 5 local.get 9 i32x4.max_s local.set 8 local.get 7 local.get 8 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 5 local.get 6 local.get 6 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 9 local.get 6 local.get 9 i32x4.min_s local.set 7 local.get 6 local.get 9 i32x4.max_s local.set 8 local.get 7 local.get 8 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 6 local.get 5 local.get 5 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 9 local.get 5 local.get 9 i32x4.min_s local.set 7 local.get 5 local.get 9 i32x4.max_s local.set 8 local.get 7 local.get 8 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 5 local.get 6 local.get 6 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 9 local.get 6 local.get 9 i32x4.min_s local.set 7 local.get 6 local.get 9 i32x4.max_s local.set 8 local.get 7 local.get 8 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 6 local.get 0 local.get 1 i32.const 2 i32.shl i32.add local.get 5 v128.store local.get 0 local.get 1 i32.const 4 i32.add i32.const 2 i32.shl i32.add local.get 6 v128.store)

///|
pub fn bitonic_merge8_i32(arr : FixedArray[Int], off : Int) -> Unit {
  if off + 8 > arr.length() || off < 0 {
    return
  }
  bitonic_merge8_i32_v128(arr, off)
}

///|
/// Bitonic merge of two sorted 8-element slices (arr[off..off+8] and
/// arr[off+8..off+16]) into one sorted 16-element block, in place. 4 stages:
///   1) reverse B halves and lane-wise min/max with A halves (distance 8)
///   2) distance-4 min/max between v128 pairs within each 8-half
///   3) distance-2 compare-exchange within each v128
///   4) distance-1 compare-exchange within each v128
#borrow(arr)
fn bitonic_merge16_i32_v128(arr : FixedArray[Int], off : Int) -> Unit =
  #|(func (param i32) (param i32) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) local.get 0 local.get 1 i32.const 2 i32.shl i32.add v128.load local.set 2 local.get 0 local.get 1 i32.const 4 i32.add i32.const 2 i32.shl i32.add v128.load local.set 3 local.get 0 local.get 1 i32.const 8 i32.add i32.const 2 i32.shl i32.add v128.load local.set 4 local.get 0 local.get 1 i32.const 12 i32.add i32.const 2 i32.shl i32.add v128.load local.set 5 local.get 5 local.get 5 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 6 local.get 4 local.get 4 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 7 local.get 2 local.get 6 i32x4.max_s local.set 8 local.get 2 local.get 6 i32x4.min_s local.set 2 local.get 3 local.get 7 i32x4.max_s local.set 9 local.get 3 local.get 7 i32x4.min_s local.set 3 local.get 2 local.get 3 i32x4.max_s local.set 4 local.get 2 local.get 3 i32x4.min_s local.set 2 local.get 4 local.set 3 local.get 8 local.get 9 i32x4.max_s local.set 4 local.get 8 local.get 9 i32x4.min_s local.set 8 local.get 4 local.set 9 local.get 2 local.get 2 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 4 local.get 2 local.get 4 i32x4.min_s local.set 5 local.get 2 local.get 4 i32x4.max_s local.set 6 local.get 5 local.get 6 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 2 local.get 3 local.get 3 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 4 local.get 3 local.get 4 i32x4.min_s local.set 5 local.get 3 local.get 4 i32x4.max_s local.set 6 local.get 5 local.get 6 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 3 local.get 8 local.get 8 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 4 local.get 8 local.get 4 i32x4.min_s local.set 5 local.get 8 local.get 4 i32x4.max_s local.set 6 local.get 5 local.get 6 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 8 local.get 9 local.get 9 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 4 local.get 9 local.get 4 i32x4.min_s local.set 5 local.get 9 local.get 4 i32x4.max_s local.set 6 local.get 5 local.get 6 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 9 local.get 2 local.get 2 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 4 local.get 2 local.get 4 i32x4.min_s local.set 5 local.get 2 local.get 4 i32x4.max_s local.set 6 local.get 5 local.get 6 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 2 local.get 3 local.get 3 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 4 local.get 3 local.get 4 i32x4.min_s local.set 5 local.get 3 local.get 4 i32x4.max_s local.set 6 local.get 5 local.get 6 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 3 local.get 8 local.get 8 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 4 local.get 8 local.get 4 i32x4.min_s local.set 5 local.get 8 local.get 4 i32x4.max_s local.set 6 local.get 5 local.get 6 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 8 local.get 9 local.get 9 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 4 local.get 9 local.get 4 i32x4.min_s local.set 5 local.get 9 local.get 4 i32x4.max_s local.set 6 local.get 5 local.get 6 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 9 local.get 0 local.get 1 i32.const 2 i32.shl i32.add local.get 2 v128.store local.get 0 local.get 1 i32.const 4 i32.add i32.const 2 i32.shl i32.add local.get 3 v128.store local.get 0 local.get 1 i32.const 8 i32.add i32.const 2 i32.shl i32.add local.get 8 v128.store local.get 0 local.get 1 i32.const 12 i32.add i32.const 2 i32.shl i32.add local.get 9 v128.store)

///|
pub fn bitonic_merge16_i32(arr : FixedArray[Int], off : Int) -> Unit {
  if off + 16 > arr.length() || off < 0 {
    return
  }
  bitonic_merge16_i32_v128(arr, off)
}

///|
/// Sort 16 elements fully in SIMD: 4 leaf sort4 + 2 bitonic_merge8 (4+4 → 8)
/// + 1 bitonic_merge16 (8+8 → 16). No scalar merge.
pub fn sort16_i32(arr : FixedArray[Int], off : Int) -> Unit {
  if off + 16 > arr.length() || off < 0 {
    return
  }
  sort4_i32(arr, off + 0)
  sort4_i32(arr, off + 4)
  sort4_i32(arr, off + 8)
  sort4_i32(arr, off + 12)
  bitonic_merge8_i32(arr, off + 0)
  bitonic_merge8_i32(arr, off + 8)
  bitonic_merge16_i32(arr, off)
}

///|
/// Bitonic merge of two sorted 16-element slices (arr[off..off+16] and
/// arr[off+16..off+32]) into one sorted 32-element block, in place. 5
/// stages: distance 16 (reverse B + lane-wise min/max with A), distance 8
/// (between v128 pairs within each half), distance 4 (between v128 pairs
/// within each 8-block), distance 2 and 1 (within each v128).
#borrow(arr)
fn bitonic_merge32_i32_v128(arr : FixedArray[Int], off : Int) -> Unit =
  #|(func (param i32) (param i32) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) local.get 0 local.get 1 i32.const 2 i32.shl i32.add v128.load local.set 2 local.get 0 local.get 1 i32.const 4 i32.add i32.const 2 i32.shl i32.add v128.load local.set 3 local.get 0 local.get 1 i32.const 8 i32.add i32.const 2 i32.shl i32.add v128.load local.set 4 local.get 0 local.get 1 i32.const 12 i32.add i32.const 2 i32.shl i32.add v128.load local.set 5 local.get 0 local.get 1 i32.const 16 i32.add i32.const 2 i32.shl i32.add v128.load local.set 6 local.get 0 local.get 1 i32.const 20 i32.add i32.const 2 i32.shl i32.add v128.load local.set 7 local.get 0 local.get 1 i32.const 24 i32.add i32.const 2 i32.shl i32.add v128.load local.set 8 local.get 0 local.get 1 i32.const 28 i32.add i32.const 2 i32.shl i32.add v128.load local.set 9 local.get 9 local.get 9 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 10 local.get 8 local.get 8 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 11 local.get 7 local.get 7 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 12 local.get 6 local.get 6 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 13 local.get 2 local.get 10 i32x4.max_s local.set 6 local.get 2 local.get 10 i32x4.min_s local.set 2 local.get 3 local.get 11 i32x4.max_s local.set 7 local.get 3 local.get 11 i32x4.min_s local.set 3 local.get 4 local.get 12 i32x4.max_s local.set 8 local.get 4 local.get 12 i32x4.min_s local.set 4 local.get 5 local.get 13 i32x4.max_s local.set 9 local.get 5 local.get 13 i32x4.min_s local.set 5 local.get 2 local.get 4 i32x4.max_s local.set 10 local.get 2 local.get 4 i32x4.min_s local.set 2 local.get 10 local.set 4 local.get 3 local.get 5 i32x4.max_s local.set 10 local.get 3 local.get 5 i32x4.min_s local.set 3 local.get 10 local.set 5 local.get 6 local.get 8 i32x4.max_s local.set 10 local.get 6 local.get 8 i32x4.min_s local.set 6 local.get 10 local.set 8 local.get 7 local.get 9 i32x4.max_s local.set 10 local.get 7 local.get 9 i32x4.min_s local.set 7 local.get 10 local.set 9 local.get 2 local.get 3 i32x4.max_s local.set 10 local.get 2 local.get 3 i32x4.min_s local.set 2 local.get 10 local.set 3 local.get 4 local.get 5 i32x4.max_s local.set 10 local.get 4 local.get 5 i32x4.min_s local.set 4 local.get 10 local.set 5 local.get 6 local.get 7 i32x4.max_s local.set 10 local.get 6 local.get 7 i32x4.min_s local.set 6 local.get 10 local.set 7 local.get 8 local.get 9 i32x4.max_s local.set 10 local.get 8 local.get 9 i32x4.min_s local.set 8 local.get 10 local.set 9 local.get 2 local.get 2 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 10 local.get 2 local.get 10 i32x4.min_s local.set 11 local.get 2 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 2 local.get 3 local.get 3 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 10 local.get 3 local.get 10 i32x4.min_s local.set 11 local.get 3 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 3 local.get 4 local.get 4 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 10 local.get 4 local.get 10 i32x4.min_s local.set 11 local.get 4 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 4 local.get 5 local.get 5 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 10 local.get 5 local.get 10 i32x4.min_s local.set 11 local.get 5 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 5 local.get 6 local.get 6 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 10 local.get 6 local.get 10 i32x4.min_s local.set 11 local.get 6 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 6 local.get 7 local.get 7 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 10 local.get 7 local.get 10 i32x4.min_s local.set 11 local.get 7 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 7 local.get 8 local.get 8 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 10 local.get 8 local.get 10 i32x4.min_s local.set 11 local.get 8 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 8 local.get 9 local.get 9 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 10 local.get 9 local.get 10 i32x4.min_s local.set 11 local.get 9 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 9 local.get 2 local.get 2 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 10 local.get 2 local.get 10 i32x4.min_s local.set 11 local.get 2 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 2 local.get 3 local.get 3 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 10 local.get 3 local.get 10 i32x4.min_s local.set 11 local.get 3 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 3 local.get 4 local.get 4 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 10 local.get 4 local.get 10 i32x4.min_s local.set 11 local.get 4 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 4 local.get 5 local.get 5 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 10 local.get 5 local.get 10 i32x4.min_s local.set 11 local.get 5 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 5 local.get 6 local.get 6 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 10 local.get 6 local.get 10 i32x4.min_s local.set 11 local.get 6 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 6 local.get 7 local.get 7 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 10 local.get 7 local.get 10 i32x4.min_s local.set 11 local.get 7 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 7 local.get 8 local.get 8 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 10 local.get 8 local.get 10 i32x4.min_s local.set 11 local.get 8 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 8 local.get 9 local.get 9 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 10 local.get 9 local.get 10 i32x4.min_s local.set 11 local.get 9 local.get 10 i32x4.max_s local.set 12 local.get 11 local.get 12 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 9 local.get 0 local.get 1 i32.const 2 i32.shl i32.add local.get 2 v128.store local.get 0 local.get 1 i32.const 4 i32.add i32.const 2 i32.shl i32.add local.get 3 v128.store local.get 0 local.get 1 i32.const 8 i32.add i32.const 2 i32.shl i32.add local.get 4 v128.store local.get 0 local.get 1 i32.const 12 i32.add i32.const 2 i32.shl i32.add local.get 5 v128.store local.get 0 local.get 1 i32.const 16 i32.add i32.const 2 i32.shl i32.add local.get 6 v128.store local.get 0 local.get 1 i32.const 20 i32.add i32.const 2 i32.shl i32.add local.get 7 v128.store local.get 0 local.get 1 i32.const 24 i32.add i32.const 2 i32.shl i32.add local.get 8 v128.store local.get 0 local.get 1 i32.const 28 i32.add i32.const 2 i32.shl i32.add local.get 9 v128.store)

///|
pub fn bitonic_merge32_i32(arr : FixedArray[Int], off : Int) -> Unit {
  if off + 32 > arr.length() || off < 0 {
    return
  }
  bitonic_merge32_i32_v128(arr, off)
}

///|
/// Bitonic merge of two sorted 32-element slices into one sorted 64-element
/// block, in place. 6 stages on 16 v128 registers (8 L + 8 H), with 3 v128
/// scratch slots. Locals 2..9 hold L (low half), 18..25 hold H (high half);
/// locals 10..17 hold B during stage 1, then go unused.
#borrow(arr)
fn bitonic_merge64_i32_v128(arr : FixedArray[Int], off : Int) -> Unit =
  #|(func (param i32) (param i32) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) (local v128) local.get 0 local.get 1 i32.const 2 i32.shl i32.add v128.load local.set 2 local.get 0 local.get 1 i32.const 4 i32.add i32.const 2 i32.shl i32.add v128.load local.set 3 local.get 0 local.get 1 i32.const 8 i32.add i32.const 2 i32.shl i32.add v128.load local.set 4 local.get 0 local.get 1 i32.const 12 i32.add i32.const 2 i32.shl i32.add v128.load local.set 5 local.get 0 local.get 1 i32.const 16 i32.add i32.const 2 i32.shl i32.add v128.load local.set 6 local.get 0 local.get 1 i32.const 20 i32.add i32.const 2 i32.shl i32.add v128.load local.set 7 local.get 0 local.get 1 i32.const 24 i32.add i32.const 2 i32.shl i32.add v128.load local.set 8 local.get 0 local.get 1 i32.const 28 i32.add i32.const 2 i32.shl i32.add v128.load local.set 9 local.get 0 local.get 1 i32.const 32 i32.add i32.const 2 i32.shl i32.add v128.load local.set 10 local.get 0 local.get 1 i32.const 36 i32.add i32.const 2 i32.shl i32.add v128.load local.set 11 local.get 0 local.get 1 i32.const 40 i32.add i32.const 2 i32.shl i32.add v128.load local.set 12 local.get 0 local.get 1 i32.const 44 i32.add i32.const 2 i32.shl i32.add v128.load local.set 13 local.get 0 local.get 1 i32.const 48 i32.add i32.const 2 i32.shl i32.add v128.load local.set 14 local.get 0 local.get 1 i32.const 52 i32.add i32.const 2 i32.shl i32.add v128.load local.set 15 local.get 0 local.get 1 i32.const 56 i32.add i32.const 2 i32.shl i32.add v128.load local.set 16 local.get 0 local.get 1 i32.const 60 i32.add i32.const 2 i32.shl i32.add v128.load local.set 17 local.get 17 local.get 17 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 26 local.get 2 local.get 26 i32x4.max_s local.set 18 local.get 2 local.get 26 i32x4.min_s local.set 2 local.get 16 local.get 16 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 26 local.get 3 local.get 26 i32x4.max_s local.set 19 local.get 3 local.get 26 i32x4.min_s local.set 3 local.get 15 local.get 15 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 26 local.get 4 local.get 26 i32x4.max_s local.set 20 local.get 4 local.get 26 i32x4.min_s local.set 4 local.get 14 local.get 14 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 26 local.get 5 local.get 26 i32x4.max_s local.set 21 local.get 5 local.get 26 i32x4.min_s local.set 5 local.get 13 local.get 13 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 26 local.get 6 local.get 26 i32x4.max_s local.set 22 local.get 6 local.get 26 i32x4.min_s local.set 6 local.get 12 local.get 12 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 26 local.get 7 local.get 26 i32x4.max_s local.set 23 local.get 7 local.get 26 i32x4.min_s local.set 7 local.get 11 local.get 11 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 26 local.get 8 local.get 26 i32x4.max_s local.set 24 local.get 8 local.get 26 i32x4.min_s local.set 8 local.get 10 local.get 10 i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 local.set 26 local.get 9 local.get 26 i32x4.max_s local.set 25 local.get 9 local.get 26 i32x4.min_s local.set 9 local.get 2 local.get 6 i32x4.max_s local.set 26 local.get 2 local.get 6 i32x4.min_s local.set 2 local.get 26 local.set 6 local.get 3 local.get 7 i32x4.max_s local.set 26 local.get 3 local.get 7 i32x4.min_s local.set 3 local.get 26 local.set 7 local.get 4 local.get 8 i32x4.max_s local.set 26 local.get 4 local.get 8 i32x4.min_s local.set 4 local.get 26 local.set 8 local.get 5 local.get 9 i32x4.max_s local.set 26 local.get 5 local.get 9 i32x4.min_s local.set 5 local.get 26 local.set 9 local.get 18 local.get 22 i32x4.max_s local.set 26 local.get 18 local.get 22 i32x4.min_s local.set 18 local.get 26 local.set 22 local.get 19 local.get 23 i32x4.max_s local.set 26 local.get 19 local.get 23 i32x4.min_s local.set 19 local.get 26 local.set 23 local.get 20 local.get 24 i32x4.max_s local.set 26 local.get 20 local.get 24 i32x4.min_s local.set 20 local.get 26 local.set 24 local.get 21 local.get 25 i32x4.max_s local.set 26 local.get 21 local.get 25 i32x4.min_s local.set 21 local.get 26 local.set 25 local.get 2 local.get 4 i32x4.max_s local.set 26 local.get 2 local.get 4 i32x4.min_s local.set 2 local.get 26 local.set 4 local.get 3 local.get 5 i32x4.max_s local.set 26 local.get 3 local.get 5 i32x4.min_s local.set 3 local.get 26 local.set 5 local.get 6 local.get 8 i32x4.max_s local.set 26 local.get 6 local.get 8 i32x4.min_s local.set 6 local.get 26 local.set 8 local.get 7 local.get 9 i32x4.max_s local.set 26 local.get 7 local.get 9 i32x4.min_s local.set 7 local.get 26 local.set 9 local.get 18 local.get 20 i32x4.max_s local.set 26 local.get 18 local.get 20 i32x4.min_s local.set 18 local.get 26 local.set 20 local.get 19 local.get 21 i32x4.max_s local.set 26 local.get 19 local.get 21 i32x4.min_s local.set 19 local.get 26 local.set 21 local.get 22 local.get 24 i32x4.max_s local.set 26 local.get 22 local.get 24 i32x4.min_s local.set 22 local.get 26 local.set 24 local.get 23 local.get 25 i32x4.max_s local.set 26 local.get 23 local.get 25 i32x4.min_s local.set 23 local.get 26 local.set 25 local.get 2 local.get 3 i32x4.max_s local.set 26 local.get 2 local.get 3 i32x4.min_s local.set 2 local.get 26 local.set 3 local.get 4 local.get 5 i32x4.max_s local.set 26 local.get 4 local.get 5 i32x4.min_s local.set 4 local.get 26 local.set 5 local.get 6 local.get 7 i32x4.max_s local.set 26 local.get 6 local.get 7 i32x4.min_s local.set 6 local.get 26 local.set 7 local.get 8 local.get 9 i32x4.max_s local.set 26 local.get 8 local.get 9 i32x4.min_s local.set 8 local.get 26 local.set 9 local.get 18 local.get 19 i32x4.max_s local.set 26 local.get 18 local.get 19 i32x4.min_s local.set 18 local.get 26 local.set 19 local.get 20 local.get 21 i32x4.max_s local.set 26 local.get 20 local.get 21 i32x4.min_s local.set 20 local.get 26 local.set 21 local.get 22 local.get 23 i32x4.max_s local.set 26 local.get 22 local.get 23 i32x4.min_s local.set 22 local.get 26 local.set 23 local.get 24 local.get 25 i32x4.max_s local.set 26 local.get 24 local.get 25 i32x4.min_s local.set 24 local.get 26 local.set 25 local.get 2 local.get 2 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 2 local.get 26 i32x4.min_s local.set 27 local.get 2 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 2 local.get 3 local.get 3 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 3 local.get 26 i32x4.min_s local.set 27 local.get 3 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 3 local.get 4 local.get 4 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 4 local.get 26 i32x4.min_s local.set 27 local.get 4 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 4 local.get 5 local.get 5 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 5 local.get 26 i32x4.min_s local.set 27 local.get 5 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 5 local.get 6 local.get 6 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 6 local.get 26 i32x4.min_s local.set 27 local.get 6 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 6 local.get 7 local.get 7 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 7 local.get 26 i32x4.min_s local.set 27 local.get 7 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 7 local.get 8 local.get 8 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 8 local.get 26 i32x4.min_s local.set 27 local.get 8 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 8 local.get 9 local.get 9 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 9 local.get 26 i32x4.min_s local.set 27 local.get 9 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 9 local.get 18 local.get 18 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 18 local.get 26 i32x4.min_s local.set 27 local.get 18 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 18 local.get 19 local.get 19 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 19 local.get 26 i32x4.min_s local.set 27 local.get 19 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 19 local.get 20 local.get 20 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 20 local.get 26 i32x4.min_s local.set 27 local.get 20 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 20 local.get 21 local.get 21 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 21 local.get 26 i32x4.min_s local.set 27 local.get 21 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 21 local.get 22 local.get 22 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 22 local.get 26 i32x4.min_s local.set 27 local.get 22 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 22 local.get 23 local.get 23 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 23 local.get 26 i32x4.min_s local.set 27 local.get 23 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 23 local.get 24 local.get 24 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 24 local.get 26 i32x4.min_s local.set 27 local.get 24 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 24 local.get 25 local.get 25 i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 local.set 26 local.get 25 local.get 26 i32x4.min_s local.set 27 local.get 25 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 local.set 25 local.get 2 local.get 2 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 2 local.get 26 i32x4.min_s local.set 27 local.get 2 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 2 local.get 3 local.get 3 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 3 local.get 26 i32x4.min_s local.set 27 local.get 3 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 3 local.get 4 local.get 4 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 4 local.get 26 i32x4.min_s local.set 27 local.get 4 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 4 local.get 5 local.get 5 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 5 local.get 26 i32x4.min_s local.set 27 local.get 5 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 5 local.get 6 local.get 6 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 6 local.get 26 i32x4.min_s local.set 27 local.get 6 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 6 local.get 7 local.get 7 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 7 local.get 26 i32x4.min_s local.set 27 local.get 7 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 7 local.get 8 local.get 8 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 8 local.get 26 i32x4.min_s local.set 27 local.get 8 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 8 local.get 9 local.get 9 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 9 local.get 26 i32x4.min_s local.set 27 local.get 9 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 9 local.get 18 local.get 18 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 18 local.get 26 i32x4.min_s local.set 27 local.get 18 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 18 local.get 19 local.get 19 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 19 local.get 26 i32x4.min_s local.set 27 local.get 19 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 19 local.get 20 local.get 20 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 20 local.get 26 i32x4.min_s local.set 27 local.get 20 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 20 local.get 21 local.get 21 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 21 local.get 26 i32x4.min_s local.set 27 local.get 21 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 21 local.get 22 local.get 22 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 22 local.get 26 i32x4.min_s local.set 27 local.get 22 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 22 local.get 23 local.get 23 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 23 local.get 26 i32x4.min_s local.set 27 local.get 23 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 23 local.get 24 local.get 24 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 24 local.get 26 i32x4.min_s local.set 27 local.get 24 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 24 local.get 25 local.get 25 i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 local.set 26 local.get 25 local.get 26 i32x4.min_s local.set 27 local.get 25 local.get 26 i32x4.max_s local.set 28 local.get 27 local.get 28 i8x16.shuffle 0 1 2 3 20 21 22 23 8 9 10 11 28 29 30 31 local.set 25 local.get 0 local.get 1 i32.const 2 i32.shl i32.add local.get 2 v128.store local.get 0 local.get 1 i32.const 4 i32.add i32.const 2 i32.shl i32.add local.get 3 v128.store local.get 0 local.get 1 i32.const 8 i32.add i32.const 2 i32.shl i32.add local.get 4 v128.store local.get 0 local.get 1 i32.const 12 i32.add i32.const 2 i32.shl i32.add local.get 5 v128.store local.get 0 local.get 1 i32.const 16 i32.add i32.const 2 i32.shl i32.add local.get 6 v128.store local.get 0 local.get 1 i32.const 20 i32.add i32.const 2 i32.shl i32.add local.get 7 v128.store local.get 0 local.get 1 i32.const 24 i32.add i32.const 2 i32.shl i32.add local.get 8 v128.store local.get 0 local.get 1 i32.const 28 i32.add i32.const 2 i32.shl i32.add local.get 9 v128.store local.get 0 local.get 1 i32.const 32 i32.add i32.const 2 i32.shl i32.add local.get 18 v128.store local.get 0 local.get 1 i32.const 36 i32.add i32.const 2 i32.shl i32.add local.get 19 v128.store local.get 0 local.get 1 i32.const 40 i32.add i32.const 2 i32.shl i32.add local.get 20 v128.store local.get 0 local.get 1 i32.const 44 i32.add i32.const 2 i32.shl i32.add local.get 21 v128.store local.get 0 local.get 1 i32.const 48 i32.add i32.const 2 i32.shl i32.add local.get 22 v128.store local.get 0 local.get 1 i32.const 52 i32.add i32.const 2 i32.shl i32.add local.get 23 v128.store local.get 0 local.get 1 i32.const 56 i32.add i32.const 2 i32.shl i32.add local.get 24 v128.store local.get 0 local.get 1 i32.const 60 i32.add i32.const 2 i32.shl i32.add local.get 25 v128.store)

///|
pub fn bitonic_merge64_i32(arr : FixedArray[Int], off : Int) -> Unit {
  if off + 64 > arr.length() || off < 0 {
    return
  }
  bitonic_merge64_i32_v128(arr, off)
}

///|
/// General-purpose sort: SIMD-leaf bottom-up merge sort.
/// Leaf (16-element blocks): sort16_i32 (fully SIMD).
/// Tail block (< 16): fall back to the built-in scalar sort.
/// Merge ladder (16+16 → 32, 32+32 → 64, ...): scalar merge2 between
/// ping-pong buffers; a future SIMD bitonic merge would replace this.
pub fn sort_i32(arr : FixedArray[Int]) -> Unit {
  let n = arr.length()
  if n <= 1 {
    return
  }
  if n <= 16 {
    // tiny case: leaf-only path.
    if n == 16 {
      sort16_i32(arr, 0)
    } else if n == 8 {
      sort4_i32(arr, 0)
      sort4_i32(arr, 4)
      bitonic_merge8_i32(arr, 0)
    } else if n == 4 {
      sort4_i32(arr, 0)
    } else {
      @internal.scalar_sort_int(arr)
    }
    return
  }

  // 1. leaf-sort each aligned 16-block.
  let leaf_end = n / 16 * 16
  for i = 0; i < leaf_end; i = i + 16 {
    sort16_i32(arr, i)
  }
  // 1b. tail < 16: copy out, scalar sort, copy back.
  let tail_len = n - leaf_end
  if tail_len > 0 {
    let tail = FixedArray::make(tail_len, 0)
    for k in 0..<tail_len {
      tail[k] = arr[leaf_end + k]
    }
    @internal.scalar_sort_int(tail)
    for k in 0..<tail_len {
      arr[leaf_end + k] = tail[k]
    }
  }

  // 2. merge ladder.
  // 2a. width = 16 pass: in-place SIMD bitonic_merge32 for full 32-blocks.
  //     Tail (last sub-block of 16-31 elements) falls back to scalar.
  let tmp = FixedArray::make(n, 0)
  let mut pos = 0
  while pos + 32 <= n {
    bitonic_merge32_i32(arr, pos)
    pos = pos + 32
  }
  // last block: pos..n is either a single 16-block (already sorted, skip) or
  // a 16-block plus a tail of 0..15 already-sorted elements; merge into arr
  // via tmp + copy-back.
  if pos + 16 < n {
    let tail_total = n - pos
    @internal.scalar_merge2_int(
      arr,
      pos,
      16,
      arr,
      pos + 16,
      tail_total - 16,
      tmp,
      pos,
    )
    for k = pos; k < n; k = k + 1 {
      arr[k] = tmp[k]
    }
  }

  // 2b. width = 32 pass: in-place SIMD bitonic_merge64 for full 64-blocks.
  let mut pos2 = 0
  while pos2 + 64 <= n {
    bitonic_merge64_i32(arr, pos2)
    pos2 = pos2 + 64
  }
  // tail: pos2..n is either a single 32-block (skip) or a 32-block plus a
  // (16-31)-element sorted tail; scalar-merge into arr via tmp + copy-back.
  if pos2 + 32 < n {
    let tail_total = n - pos2
    @internal.scalar_merge2_int(
      arr,
      pos2,
      32,
      arr,
      pos2 + 32,
      tail_total - 32,
      tmp,
      pos2,
    )
    for k = pos2; k < n; k = k + 1 {
      arr[k] = tmp[k]
    }
  }

  // 2c. width >= 64 pass: scalar merge ladder with ping-pong between arr and tmp.
  let mut width = 64
  let mut src_is_arr = true
  while width < n {
    let merged = width * 2
    let mut i = 0
    while i < n {
      let a_off = i
      let a_end = if a_off + width <= n { a_off + width } else { n }
      let b_off = a_end
      let b_end = if b_off + width <= n { b_off + width } else { n }
      if b_off >= n {
        // no B half: copy A as-is into dst.
        if src_is_arr {
          for k = a_off; k < a_end; k = k + 1 {
            tmp[k] = arr[k]
          }
        } else {
          for k = a_off; k < a_end; k = k + 1 {
            arr[k] = tmp[k]
          }
        }
      } else if src_is_arr {
        @internal.scalar_merge2_int(
          arr,
          a_off,
          a_end - a_off,
          arr,
          b_off,
          b_end - b_off,
          tmp,
          a_off,
        )
      } else {
        @internal.scalar_merge2_int(
          tmp,
          a_off,
          a_end - a_off,
          tmp,
          b_off,
          b_end - b_off,
          arr,
          a_off,
        )
      }
      i = i + merged
    }
    width = merged
    src_is_arr = !src_is_arr
  }

  // 3. if final pass landed in tmp, copy back into arr.
  if !src_is_arr {
    for k in 0..<n {
      arr[k] = tmp[k]
    }
  }
}
