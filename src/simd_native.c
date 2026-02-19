#include <stdint.h>
#include <stddef.h>

/* TCC (MoonBit default) doesn't support NEON/SSE intrinsics.
   Use clang/gcc with -march=native for actual SIMD. */
#if !defined(__TINYC__)
  #if defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>
    #define USE_NEON 1
  #elif defined(__SSE2__)
    #include <emmintrin.h>
    #define USE_SSE2 1
  #endif
#endif

#ifndef USE_NEON
  #define USE_NEON 0
#endif
#ifndef USE_SSE2
  #define USE_SSE2 0
#endif

int32_t simd_sum_ffi(const int32_t* arr, int32_t len) {
  int32_t result = 0;
  int32_t i = 0;

#if USE_NEON
  int32x4_t acc = vdupq_n_s32(0);
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4) {
    acc = vaddq_s32(acc, vld1q_s32(arr + i));
  }
  result = vaddvq_s32(acc);
#elif USE_SSE2
  __m128i acc = _mm_setzero_si128();
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4) {
    acc = _mm_add_epi32(acc, _mm_loadu_si128((const __m128i*)(arr + i)));
  }
  __m128i hi = _mm_shuffle_epi32(acc, _MM_SHUFFLE(1,0,3,2));
  acc = _mm_add_epi32(acc, hi);
  hi = _mm_shuffle_epi32(acc, _MM_SHUFFLE(0,1,0,1));
  acc = _mm_add_epi32(acc, hi);
  result = _mm_cvtsi128_si32(acc);
#endif

  for (; i < len; i++) {
    result += arr[i];
  }
  return result;
}

int32_t simd_dot_product_ffi(const int32_t* a, const int32_t* b, int32_t len) {
  int32_t result = 0;
  int32_t i = 0;

#if USE_NEON
  int32x4_t acc = vdupq_n_s32(0);
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4) {
    acc = vmlaq_s32(acc, vld1q_s32(a + i), vld1q_s32(b + i));
  }
  result = vaddvq_s32(acc);
#elif USE_SSE2
  __m128i acc = _mm_setzero_si128();
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4) {
    __m128i va = _mm_loadu_si128((const __m128i*)(a + i));
    __m128i vb = _mm_loadu_si128((const __m128i*)(b + i));
    __m128i lo = _mm_mul_epu32(va, vb);
    __m128i hi = _mm_mul_epu32(_mm_srli_si128(va, 4), _mm_srli_si128(vb, 4));
    __m128i prod = _mm_unpacklo_epi32(
      _mm_shuffle_epi32(lo, _MM_SHUFFLE(0,0,2,0)),
      _mm_shuffle_epi32(hi, _MM_SHUFFLE(0,0,2,0))
    );
    acc = _mm_add_epi32(acc, prod);
  }
  __m128i h = _mm_shuffle_epi32(acc, _MM_SHUFFLE(1,0,3,2));
  acc = _mm_add_epi32(acc, h);
  h = _mm_shuffle_epi32(acc, _MM_SHUFFLE(0,1,0,1));
  acc = _mm_add_epi32(acc, h);
  result = _mm_cvtsi128_si32(acc);
#endif

  for (; i < len; i++) {
    result += a[i] * b[i];
  }
  return result;
}

void simd_add_ffi(const int32_t* a, const int32_t* b, int32_t* out, int32_t len) {
  int32_t i = 0;

#if USE_NEON
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4) {
    vst1q_s32(out + i, vaddq_s32(vld1q_s32(a + i), vld1q_s32(b + i)));
  }
#elif USE_SSE2
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4) {
    __m128i va = _mm_loadu_si128((const __m128i*)(a + i));
    __m128i vb = _mm_loadu_si128((const __m128i*)(b + i));
    _mm_storeu_si128((__m128i*)(out + i), _mm_add_epi32(va, vb));
  }
#endif

  for (; i < len; i++) {
    out[i] = a[i] + b[i];
  }
}

uint32_t simd_adler32_ffi(const uint8_t* data, int32_t len) {
  uint32_t a = 1, b = 0;
  const uint32_t MOD_ADLER = 65521;
  int32_t i = 0;
  while (i < len) {
    int32_t block = len - i;
    if (block > 5552) block = 5552;
    int32_t end = i + block;
    for (; i < end; i++) {
      a += data[i];
      b += a;
    }
    a %= MOD_ADLER;
    b %= MOD_ADLER;
  }
  return (b << 16) | a;
}
