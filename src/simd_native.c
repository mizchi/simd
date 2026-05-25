#include <stdint.h>
#include <stddef.h>
#include <math.h>

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

void simd_sub_ffi(const int32_t* a, const int32_t* b, int32_t* out, int32_t len) {
  int32_t i = 0;
#if USE_NEON
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4) {
    vst1q_s32(out + i, vsubq_s32(vld1q_s32(a + i), vld1q_s32(b + i)));
  }
#elif USE_SSE2
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4) {
    __m128i va = _mm_loadu_si128((const __m128i*)(a + i));
    __m128i vb = _mm_loadu_si128((const __m128i*)(b + i));
    _mm_storeu_si128((__m128i*)(out + i), _mm_sub_epi32(va, vb));
  }
#endif
  for (; i < len; i++) out[i] = a[i] - b[i];
}

void simd_mul_ffi(const int32_t* a, const int32_t* b, int32_t* out, int32_t len) {
  int32_t i = 0;
#if USE_NEON
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4) {
    vst1q_s32(out + i, vmulq_s32(vld1q_s32(a + i), vld1q_s32(b + i)));
  }
#elif defined(__SSE4_1__)
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4) {
    __m128i va = _mm_loadu_si128((const __m128i*)(a + i));
    __m128i vb = _mm_loadu_si128((const __m128i*)(b + i));
    _mm_storeu_si128((__m128i*)(out + i), _mm_mullo_epi32(va, vb));
  }
#endif
  for (; i < len; i++) out[i] = a[i] * b[i];
}

void simd_min_elem_ffi(const int32_t* a, const int32_t* b, int32_t* out, int32_t len) {
  int32_t i = 0;
#if USE_NEON
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4) {
    vst1q_s32(out + i, vminq_s32(vld1q_s32(a + i), vld1q_s32(b + i)));
  }
#elif defined(__SSE4_1__)
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4) {
    __m128i va = _mm_loadu_si128((const __m128i*)(a + i));
    __m128i vb = _mm_loadu_si128((const __m128i*)(b + i));
    _mm_storeu_si128((__m128i*)(out + i), _mm_min_epi32(va, vb));
  }
#endif
  for (; i < len; i++) out[i] = a[i] < b[i] ? a[i] : b[i];
}

void simd_max_elem_ffi(const int32_t* a, const int32_t* b, int32_t* out, int32_t len) {
  int32_t i = 0;
#if USE_NEON
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4) {
    vst1q_s32(out + i, vmaxq_s32(vld1q_s32(a + i), vld1q_s32(b + i)));
  }
#elif defined(__SSE4_1__)
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4) {
    __m128i va = _mm_loadu_si128((const __m128i*)(a + i));
    __m128i vb = _mm_loadu_si128((const __m128i*)(b + i));
    _mm_storeu_si128((__m128i*)(out + i), _mm_max_epi32(va, vb));
  }
#endif
  for (; i < len; i++) out[i] = a[i] > b[i] ? a[i] : b[i];
}

int32_t simd_min_ffi(const int32_t* arr, int32_t len) {
  if (len <= 0) return 0;
  int32_t result = arr[0];
  int32_t i = 0;
#if USE_NEON
  if (len >= 4) {
    int32x4_t acc = vld1q_s32(arr);
    i = 4;
    int32_t end4 = (len / 4) * 4;
    for (; i < end4; i += 4) {
      acc = vminq_s32(acc, vld1q_s32(arr + i));
    }
    result = vminvq_s32(acc);
  }
#endif
  for (; i < len; i++) {
    if (arr[i] < result) result = arr[i];
  }
  return result;
}

int32_t simd_max_ffi(const int32_t* arr, int32_t len) {
  if (len <= 0) return 0;
  int32_t result = arr[0];
  int32_t i = 0;
#if USE_NEON
  if (len >= 4) {
    int32x4_t acc = vld1q_s32(arr);
    i = 4;
    int32_t end4 = (len / 4) * 4;
    for (; i < end4; i += 4) {
      acc = vmaxq_s32(acc, vld1q_s32(arr + i));
    }
    result = vmaxvq_s32(acc);
  }
#endif
  for (; i < len; i++) {
    if (arr[i] > result) result = arr[i];
  }
  return result;
}

void simd_saxpy_ffi(int32_t k, const int32_t* a, const int32_t* b, int32_t* out, int32_t len) {
  int32_t i = 0;
#if USE_NEON
  int32x4_t kv = vdupq_n_s32(k);
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4) {
    vst1q_s32(out + i, vaddq_s32(vmulq_s32(kv, vld1q_s32(a + i)), vld1q_s32(b + i)));
  }
#elif defined(__SSE4_1__)
  __m128i kv = _mm_set1_epi32(k);
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4) {
    __m128i va = _mm_loadu_si128((const __m128i*)(a + i));
    __m128i vb = _mm_loadu_si128((const __m128i*)(b + i));
    _mm_storeu_si128((__m128i*)(out + i), _mm_add_epi32(_mm_mullo_epi32(kv, va), vb));
  }
#endif
  for (; i < len; i++) out[i] = k * a[i] + b[i];
}

// --- f64 ops ---

void simd_add_f64_ffi(const double* a, const double* b, double* out, int32_t len) {
  int32_t i = 0;
#if USE_NEON
  int32_t end2 = (len / 2) * 2;
  for (; i < end2; i += 2) {
    vst1q_f64(out + i, vaddq_f64(vld1q_f64(a + i), vld1q_f64(b + i)));
  }
#elif USE_SSE2
  int32_t end2 = (len / 2) * 2;
  for (; i < end2; i += 2) {
    __m128d va = _mm_loadu_pd(a + i);
    __m128d vb = _mm_loadu_pd(b + i);
    _mm_storeu_pd(out + i, _mm_add_pd(va, vb));
  }
#endif
  for (; i < len; i++) out[i] = a[i] + b[i];
}

void simd_sub_f64_ffi(const double* a, const double* b, double* out, int32_t len) {
  int32_t i = 0;
#if USE_NEON
  int32_t end2 = (len / 2) * 2;
  for (; i < end2; i += 2) {
    vst1q_f64(out + i, vsubq_f64(vld1q_f64(a + i), vld1q_f64(b + i)));
  }
#elif USE_SSE2
  int32_t end2 = (len / 2) * 2;
  for (; i < end2; i += 2) {
    __m128d va = _mm_loadu_pd(a + i);
    __m128d vb = _mm_loadu_pd(b + i);
    _mm_storeu_pd(out + i, _mm_sub_pd(va, vb));
  }
#endif
  for (; i < len; i++) out[i] = a[i] - b[i];
}

void simd_mul_f64_ffi(const double* a, const double* b, double* out, int32_t len) {
  int32_t i = 0;
#if USE_NEON
  int32_t end2 = (len / 2) * 2;
  for (; i < end2; i += 2) {
    vst1q_f64(out + i, vmulq_f64(vld1q_f64(a + i), vld1q_f64(b + i)));
  }
#elif USE_SSE2
  int32_t end2 = (len / 2) * 2;
  for (; i < end2; i += 2) {
    __m128d va = _mm_loadu_pd(a + i);
    __m128d vb = _mm_loadu_pd(b + i);
    _mm_storeu_pd(out + i, _mm_mul_pd(va, vb));
  }
#endif
  for (; i < len; i++) out[i] = a[i] * b[i];
}

void simd_div_f64_ffi(const double* a, const double* b, double* out, int32_t len) {
  int32_t i = 0;
#if USE_NEON
  int32_t end2 = (len / 2) * 2;
  for (; i < end2; i += 2) {
    vst1q_f64(out + i, vdivq_f64(vld1q_f64(a + i), vld1q_f64(b + i)));
  }
#elif USE_SSE2
  int32_t end2 = (len / 2) * 2;
  for (; i < end2; i += 2) {
    __m128d va = _mm_loadu_pd(a + i);
    __m128d vb = _mm_loadu_pd(b + i);
    _mm_storeu_pd(out + i, _mm_div_pd(va, vb));
  }
#endif
  for (; i < len; i++) out[i] = a[i] / b[i];
}

void simd_sqrt_f64_ffi(const double* a, double* out, int32_t len) {
  int32_t i = 0;
#if USE_NEON
  int32_t end2 = (len / 2) * 2;
  for (; i < end2; i += 2) {
    vst1q_f64(out + i, vsqrtq_f64(vld1q_f64(a + i)));
  }
#elif USE_SSE2
  int32_t end2 = (len / 2) * 2;
  for (; i < end2; i += 2) {
    _mm_storeu_pd(out + i, _mm_sqrt_pd(_mm_loadu_pd(a + i)));
  }
#endif
  for (; i < len; i++) out[i] = sqrt(a[i]);
}

void simd_min_elem_f64_ffi(const double* a, const double* b, double* out, int32_t len) {
  int32_t i = 0;
#if USE_NEON
  int32_t end2 = (len / 2) * 2;
  for (; i < end2; i += 2) {
    vst1q_f64(out + i, vminq_f64(vld1q_f64(a + i), vld1q_f64(b + i)));
  }
#elif USE_SSE2
  int32_t end2 = (len / 2) * 2;
  for (; i < end2; i += 2) {
    _mm_storeu_pd(out + i, _mm_min_pd(_mm_loadu_pd(a + i), _mm_loadu_pd(b + i)));
  }
#endif
  for (; i < len; i++) out[i] = a[i] < b[i] ? a[i] : b[i];
}

void simd_max_elem_f64_ffi(const double* a, const double* b, double* out, int32_t len) {
  int32_t i = 0;
#if USE_NEON
  int32_t end2 = (len / 2) * 2;
  for (; i < end2; i += 2) {
    vst1q_f64(out + i, vmaxq_f64(vld1q_f64(a + i), vld1q_f64(b + i)));
  }
#elif USE_SSE2
  int32_t end2 = (len / 2) * 2;
  for (; i < end2; i += 2) {
    _mm_storeu_pd(out + i, _mm_max_pd(_mm_loadu_pd(a + i), _mm_loadu_pd(b + i)));
  }
#endif
  for (; i < len; i++) out[i] = a[i] > b[i] ? a[i] : b[i];
}

double simd_sum_f64_ffi(const double* arr, int32_t len) {
  double result = 0.0;
  int32_t i = 0;
#if USE_NEON
  float64x2_t acc = vdupq_n_f64(0.0);
  int32_t end2 = (len / 2) * 2;
  for (; i < end2; i += 2) {
    acc = vaddq_f64(acc, vld1q_f64(arr + i));
  }
  result = vaddvq_f64(acc);
#elif USE_SSE2
  __m128d acc = _mm_setzero_pd();
  int32_t end2 = (len / 2) * 2;
  for (; i < end2; i += 2) {
    acc = _mm_add_pd(acc, _mm_loadu_pd(arr + i));
  }
  // horizontal add lane 0 + lane 1
  double lanes[2];
  _mm_storeu_pd(lanes, acc);
  result = lanes[0] + lanes[1];
#endif
  for (; i < len; i++) result += arr[i];
  return result;
}

double simd_dot_f64_ffi(const double* a, const double* b, int32_t len) {
  double result = 0.0;
  int32_t i = 0;
#if USE_NEON
  float64x2_t acc = vdupq_n_f64(0.0);
  int32_t end2 = (len / 2) * 2;
  for (; i < end2; i += 2) {
    acc = vaddq_f64(acc, vmulq_f64(vld1q_f64(a + i), vld1q_f64(b + i)));
  }
  result = vaddvq_f64(acc);
#elif USE_SSE2
  __m128d acc = _mm_setzero_pd();
  int32_t end2 = (len / 2) * 2;
  for (; i < end2; i += 2) {
    acc = _mm_add_pd(acc, _mm_mul_pd(_mm_loadu_pd(a + i), _mm_loadu_pd(b + i)));
  }
  double lanes[2];
  _mm_storeu_pd(lanes, acc);
  result = lanes[0] + lanes[1];
#endif
  for (; i < len; i++) result += a[i] * b[i];
  return result;
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
