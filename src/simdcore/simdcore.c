// Native C kernels for the JSON structural pipeline (Bytes-direct).
// TCC has no SIMD intrinsics, but a tight bounds-check-free C loop (and
// auto-vectorisation under gcc/clang) still beats the per-byte MoonBit
// scalar path. `FixedArray[Int]` / `Bytes` arrive as int32_t* / const
// uint8_t* via #borrow.
#include <stdint.h>

static void classify_structural(const uint8_t* src, int32_t len, int32_t* out) {
  int32_t words = (len + 31) / 32;
  for (int32_t w = 0; w < words; w++) out[w] = 0;
  for (int32_t i = 0; i < len; i++) {
    uint8_t b = src[i];
    if (b == 0x7B || b == 0x7D || b == 0x5B || b == 0x5D || b == 0x2C ||
        b == 0x3A)
      out[i >> 5] |= (int32_t)(1u << (i & 31));
  }
}

// Classify structural bytes ({ } [ ] , :) into a bitmap (zeroed first).
void simdcore_json_classify_structural(const uint8_t* src, int32_t len,
                                       int32_t* out) {
  classify_structural(src, len, out);
}

// Full pipeline: classify + in-string quote mask + extract offsets.
// Returns the number of structural offsets written to `indices`.
int32_t simdcore_json_indices(const uint8_t* src, int32_t len,
                              int32_t* structural, int32_t* quote,
                              int32_t* indices) {
  int32_t words = (len + 31) / 32;
  classify_structural(src, len, structural);
  for (int32_t w = 0; w < words; w++) quote[w] = 0;
  int32_t in_string = 0, escape_pending = 0;
  for (int32_t i = 0; i < len; i++) {
    uint8_t b = src[i];
    int32_t bit_set = 0;
    if (escape_pending) {
      escape_pending = 0;
      bit_set = in_string;
    } else if (b == 0x22) {
      in_string = !in_string;
    } else if (b == 0x5C && in_string) {
      escape_pending = 1;
      bit_set = 1;
    } else {
      bit_set = in_string;
    }
    if (bit_set) quote[i >> 5] |= (int32_t)(1u << (i & 31));
  }
  int32_t count = 0;
  for (int32_t w = 0; w < words; w++) {
    uint32_t word = (uint32_t)structural[w] & ~(uint32_t)quote[w];
    while (word != 0u) {
      int32_t bit = 0;
      uint32_t x = word;
      while ((x & 1u) == 0u) {
        x >>= 1;
        bit++;
      }
      indices[count++] = (w << 5) + bit;
      word &= word - 1u;
    }
  }
  return count;
}
