#include <stdint.h>
#include <stddef.h>
#include <string.h>
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

// --- Int element-wise + reduce ---

void simd_neg_i32_ffi(const int32_t* a, int32_t* out, int32_t len) {
  int32_t i = 0;
#if USE_NEON
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4) vst1q_s32(out + i, vnegq_s32(vld1q_s32(a + i)));
#elif USE_SSE2
  __m128i zero = _mm_setzero_si128();
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4)
    _mm_storeu_si128((__m128i*)(out + i), _mm_sub_epi32(zero, _mm_loadu_si128((const __m128i*)(a + i))));
#endif
  for (; i < len; i++) out[i] = -a[i];
}

void simd_abs_i32_ffi(const int32_t* a, int32_t* out, int32_t len) {
  int32_t i = 0;
#if USE_NEON
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4) vst1q_s32(out + i, vabsq_s32(vld1q_s32(a + i)));
#elif defined(__SSSE3__)
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4)
    _mm_storeu_si128((__m128i*)(out + i), _mm_abs_epi32(_mm_loadu_si128((const __m128i*)(a + i))));
#endif
  for (; i < len; i++) out[i] = a[i] < 0 ? -a[i] : a[i];
}

void simd_eq_i32_ffi(const int32_t* a, const int32_t* b, int32_t* out, int32_t len) {
  int32_t i = 0;
#if USE_NEON
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4)
    vst1q_s32(out + i, vreinterpretq_s32_u32(vceqq_s32(vld1q_s32(a + i), vld1q_s32(b + i))));
#elif USE_SSE2
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4)
    _mm_storeu_si128((__m128i*)(out + i),
      _mm_cmpeq_epi32(_mm_loadu_si128((const __m128i*)(a + i)), _mm_loadu_si128((const __m128i*)(b + i))));
#endif
  for (; i < len; i++) out[i] = a[i] == b[i] ? -1 : 0;
}

void simd_lt_i32_ffi(const int32_t* a, const int32_t* b, int32_t* out, int32_t len) {
  int32_t i = 0;
#if USE_NEON
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4)
    vst1q_s32(out + i, vreinterpretq_s32_u32(vcltq_s32(vld1q_s32(a + i), vld1q_s32(b + i))));
#elif USE_SSE2
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4)
    _mm_storeu_si128((__m128i*)(out + i),
      _mm_cmpgt_epi32(_mm_loadu_si128((const __m128i*)(b + i)), _mm_loadu_si128((const __m128i*)(a + i))));
#endif
  for (; i < len; i++) out[i] = a[i] < b[i] ? -1 : 0;
}

void simd_gt_i32_ffi(const int32_t* a, const int32_t* b, int32_t* out, int32_t len) {
  int32_t i = 0;
#if USE_NEON
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4)
    vst1q_s32(out + i, vreinterpretq_s32_u32(vcgtq_s32(vld1q_s32(a + i), vld1q_s32(b + i))));
#elif USE_SSE2
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4)
    _mm_storeu_si128((__m128i*)(out + i),
      _mm_cmpgt_epi32(_mm_loadu_si128((const __m128i*)(a + i)), _mm_loadu_si128((const __m128i*)(b + i))));
#endif
  for (; i < len; i++) out[i] = a[i] > b[i] ? -1 : 0;
}

void simd_where_i32_ffi(const int32_t* mask, const int32_t* a, const int32_t* b, int32_t* out, int32_t len) {
  int32_t i = 0;
#if USE_NEON
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4) {
    uint32x4_t m = vreinterpretq_u32_s32(vld1q_s32(mask + i));
    int32x4_t av = vld1q_s32(a + i);
    int32x4_t bv = vld1q_s32(b + i);
    vst1q_s32(out + i, vbslq_s32(m, av, bv));
  }
#elif USE_SSE2
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4) {
    __m128i m = _mm_loadu_si128((const __m128i*)(mask + i));
    __m128i av = _mm_loadu_si128((const __m128i*)(a + i));
    __m128i bv = _mm_loadu_si128((const __m128i*)(b + i));
    _mm_storeu_si128((__m128i*)(out + i), _mm_or_si128(_mm_and_si128(av, m), _mm_andnot_si128(m, bv)));
  }
#endif
  for (; i < len; i++) out[i] = mask[i] != 0 ? a[i] : b[i];
}

int32_t simd_prod_i32_ffi(const int32_t* arr, int32_t len) {
  int32_t result = 1;
  int32_t i = 0;
#if USE_NEON
  if (len >= 4) {
    int32x4_t acc = vld1q_s32(arr);
    i = 4;
    int32_t end4 = (len / 4) * 4;
    for (; i < end4; i += 4) acc = vmulq_s32(acc, vld1q_s32(arr + i));
    int32_t lanes[4];
    vst1q_s32(lanes, acc);
    result = lanes[0] * lanes[1] * lanes[2] * lanes[3];
  }
#endif
  for (; i < len; i++) result *= arr[i];
  return result;
}

int32_t simd_count_nonzero_i32_ffi(const int32_t* arr, int32_t len) {
  int32_t count = 0;
  int32_t i = 0;
#if USE_NEON
  int32x4_t zero = vdupq_n_s32(0);
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4) {
    uint32x4_t ne = vmvnq_u32(vceqq_s32(vld1q_s32(arr + i), zero));
    count += (int32_t)vaddvq_u32(vshrq_n_u32(ne, 31));
  }
#endif
  for (; i < len; i++) {
    if (arr[i] != 0) count++;
  }
  return count;
}

int32_t simd_all_i32_ffi(const int32_t* arr, int32_t len) {
  int32_t i = 0;
#if USE_NEON
  int32x4_t zero = vdupq_n_s32(0);
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4) {
    uint32x4_t eq_zero = vceqq_s32(vld1q_s32(arr + i), zero);
    if (vmaxvq_u32(eq_zero) != 0) return 0;
  }
#endif
  for (; i < len; i++) {
    if (arr[i] == 0) return 0;
  }
  return 1;
}

int32_t simd_any_i32_ffi(const int32_t* arr, int32_t len) {
  int32_t i = 0;
#if USE_NEON
  int32x4_t zero = vdupq_n_s32(0);
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4) {
    uint32x4_t ne = vmvnq_u32(vceqq_s32(vld1q_s32(arr + i), zero));
    if (vmaxvq_u32(ne) != 0) return 1;
  }
#endif
  for (; i < len; i++) {
    if (arr[i] != 0) return 1;
  }
  return 0;
}

// --- byte ops ---

void simd_memcpy_bytes_ffi(const uint8_t* src, int32_t si, uint8_t* dst, int32_t di, int32_t len) {
  memcpy(dst + di, src + si, (size_t)len);
}

void simd_memset_bytes_ffi(uint8_t* dst, int32_t off, int32_t len, uint8_t value) {
  memset(dst + off, value, (size_t)len);
}

int32_t simd_equal_bytes_ffi(const uint8_t* a, const uint8_t* b, int32_t len) {
  return memcmp(a, b, (size_t)len) == 0 ? 1 : 0;
}

int32_t simd_find_byte_ffi(const uint8_t* data, int32_t len, uint8_t needle) {
  const void* p = memchr(data, needle, (size_t)len);
  return p ? (int32_t)((const uint8_t*)p - data) : -1;
}

int32_t simd_count_byte_ffi(const uint8_t* data, int32_t len, uint8_t needle) {
  int32_t count = 0;
  int32_t i = 0;
#if USE_NEON
  uint8x16_t needle_v = vdupq_n_u8(needle);
  uint8x16_t one_v = vdupq_n_u8(1);
  int32_t end16 = (len / 16) * 16;
  for (; i < end16; i += 16) {
    uint8x16_t eq = vceqq_u8(vld1q_u8(data + i), needle_v);
    count += vaddvq_u8(vandq_u8(eq, one_v));
  }
#elif USE_SSE2
  __m128i needle_v = _mm_set1_epi8((char)needle);
  int32_t end16 = (len / 16) * 16;
  for (; i < end16; i += 16) {
    __m128i v = _mm_loadu_si128((const __m128i*)(data + i));
    __m128i eq = _mm_cmpeq_epi8(v, needle_v);
    count += __builtin_popcount(_mm_movemask_epi8(eq));
  }
#endif
  for (; i < len; i++) {
    if (data[i] == needle) count++;
  }
  return count;
}

int32_t simd_popcount_bytes_ffi(const uint8_t* data, int32_t len) {
  int32_t total = 0;
  int32_t i = 0;
#if USE_NEON
  int32_t end16 = (len / 16) * 16;
  for (; i < end16; i += 16) {
    uint8x16_t v = vld1q_u8(data + i);
    total += vaddvq_u8(vcntq_u8(v));
  }
#endif
  for (; i < len; i++) total += __builtin_popcount(data[i]);
  return total;
}

int32_t simd_is_ascii_ffi(const uint8_t* data, int32_t len) {
  int32_t i = 0;
#if USE_NEON
  int32_t end16 = (len / 16) * 16;
  uint8x16_t hi = vdupq_n_u8(0x80);
  for (; i < end16; i += 16) {
    uint8x16_t v = vld1q_u8(data + i);
    if (vmaxvq_u8(vandq_u8(v, hi)) != 0) return 0;
  }
#elif USE_SSE2
  int32_t end16 = (len / 16) * 16;
  for (; i < end16; i += 16) {
    __m128i v = _mm_loadu_si128((const __m128i*)(data + i));
    if (_mm_movemask_epi8(v) != 0) return 0;
  }
#endif
  for (; i < len; i++) {
    if (data[i] & 0x80) return 0;
  }
  return 1;
}

void simd_to_lower_ascii_ffi(uint8_t* data, int32_t len) {
  int32_t i = 0;
#if USE_NEON
  uint8x16_t a_v = vdupq_n_u8('A');
  uint8x16_t z_v = vdupq_n_u8('Z');
  uint8x16_t s_v = vdupq_n_u8(0x20);
  int32_t end16 = (len / 16) * 16;
  for (; i < end16; i += 16) {
    uint8x16_t v = vld1q_u8(data + i);
    uint8x16_t mask = vandq_u8(vcgeq_u8(v, a_v), vcleq_u8(v, z_v));
    vst1q_u8(data + i, vaddq_u8(v, vandq_u8(mask, s_v)));
  }
#endif
  for (; i < len; i++) {
    if (data[i] >= 'A' && data[i] <= 'Z') data[i] += 0x20;
  }
}

void simd_to_upper_ascii_ffi(uint8_t* data, int32_t len) {
  int32_t i = 0;
#if USE_NEON
  uint8x16_t a_v = vdupq_n_u8('a');
  uint8x16_t z_v = vdupq_n_u8('z');
  uint8x16_t s_v = vdupq_n_u8(0x20);
  int32_t end16 = (len / 16) * 16;
  for (; i < end16; i += 16) {
    uint8x16_t v = vld1q_u8(data + i);
    uint8x16_t mask = vandq_u8(vcgeq_u8(v, a_v), vcleq_u8(v, z_v));
    vst1q_u8(data + i, vsubq_u8(v, vandq_u8(mask, s_v)));
  }
#endif
  for (; i < len; i++) {
    if (data[i] >= 'a' && data[i] <= 'z') data[i] -= 0x20;
  }
}

int32_t simd_first_non_ascii_chunk_ffi(const uint8_t* data, int32_t len) {
  int32_t end16 = (len / 16) * 16;
  int32_t i = 0;
#if USE_NEON
  uint8x16_t hi = vdupq_n_u8(0x80);
  while (i < end16) {
    uint8x16_t v = vld1q_u8(data + i);
    if (vmaxvq_u8(vandq_u8(v, hi)) != 0) return i;
    i += 16;
  }
#elif USE_SSE2
  while (i < end16) {
    __m128i v = _mm_loadu_si128((const __m128i*)(data + i));
    if (_mm_movemask_epi8(v) != 0) return i;
    i += 16;
  }
#endif
  return end16;
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
