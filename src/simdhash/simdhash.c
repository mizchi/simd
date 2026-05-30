// Native 4-way multi-buffer SHA-256 / SHA-1 for @simdhash.
//
// Hashes four equal-length messages in parallel, one per SIMD lane:
//   - x86-64: SSE2 (baseline ABI, no -march needed)
//   - arm64 : NEON
//   - other : a portable scalar "4 lanes in a struct" fallback (same kernel).
// gcc/clang-compiled (the native-stub is NOT tcc), so the intrinsics activate.
// Results are byte-identical to four scalar digests.

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#if defined(__SSE2__)
#include <emmintrin.h>
typedef __m128i v4;
#define V_ADD(a, b) _mm_add_epi32((a), (b))
#define V_AND(a, b) _mm_and_si128((a), (b))
#define V_OR(a, b) _mm_or_si128((a), (b))
#define V_XOR(a, b) _mm_xor_si128((a), (b))
#define V_ANDNOT(a, b) _mm_andnot_si128((a), (b)) /* ~a & b */
#define V_SHR(a, n) _mm_srli_epi32((a), (n))
#define V_SHL(a, n) _mm_slli_epi32((a), (n))
#define V_SET1(x) _mm_set1_epi32((int)(x))
static inline v4 v_set4(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3) {
  return _mm_set_epi32((int)w3, (int)w2, (int)w1, (int)w0);
}
static inline void v_get4(v4 v, uint32_t o[4]) {
  _mm_storeu_si128((__m128i *)o, v);
}

#elif defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
typedef uint32x4_t v4;
#define V_ADD(a, b) vaddq_u32((a), (b))
#define V_AND(a, b) vandq_u32((a), (b))
#define V_OR(a, b) vorrq_u32((a), (b))
#define V_XOR(a, b) veorq_u32((a), (b))
#define V_ANDNOT(a, b) vbicq_u32((b), (a)) /* b & ~a == ~a & b */
#define V_SHR(a, n) vshrq_n_u32((a), (n))
#define V_SHL(a, n) vshlq_n_u32((a), (n))
#define V_SET1(x) vdupq_n_u32((uint32_t)(x))
static inline v4 v_set4(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3) {
  uint32_t t[4] = {w0, w1, w2, w3};
  return vld1q_u32(t);
}
static inline void v_get4(v4 v, uint32_t o[4]) { vst1q_u32(o, v); }

#else
typedef struct {
  uint32_t l[4];
} v4;
static inline v4 V_ADD(v4 a, v4 b) {
  v4 r;
  for (int i = 0; i < 4; i++) r.l[i] = a.l[i] + b.l[i];
  return r;
}
static inline v4 V_AND(v4 a, v4 b) {
  v4 r;
  for (int i = 0; i < 4; i++) r.l[i] = a.l[i] & b.l[i];
  return r;
}
static inline v4 V_OR(v4 a, v4 b) {
  v4 r;
  for (int i = 0; i < 4; i++) r.l[i] = a.l[i] | b.l[i];
  return r;
}
static inline v4 V_XOR(v4 a, v4 b) {
  v4 r;
  for (int i = 0; i < 4; i++) r.l[i] = a.l[i] ^ b.l[i];
  return r;
}
static inline v4 V_ANDNOT(v4 a, v4 b) { /* ~a & b */
  v4 r;
  for (int i = 0; i < 4; i++) r.l[i] = (~a.l[i]) & b.l[i];
  return r;
}
static inline v4 V_SHR(v4 a, int n) {
  v4 r;
  for (int i = 0; i < 4; i++) r.l[i] = a.l[i] >> n;
  return r;
}
static inline v4 V_SHL(v4 a, int n) {
  v4 r;
  for (int i = 0; i < 4; i++) r.l[i] = a.l[i] << n;
  return r;
}
static inline v4 V_SET1(uint32_t x) {
  v4 r;
  for (int i = 0; i < 4; i++) r.l[i] = x;
  return r;
}
static inline v4 v_set4(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3) {
  v4 r;
  r.l[0] = w0;
  r.l[1] = w1;
  r.l[2] = w2;
  r.l[3] = w3;
  return r;
}
static inline void v_get4(v4 v, uint32_t o[4]) {
  for (int i = 0; i < 4; i++) o[i] = v.l[i];
}
#endif

#define ROTR(x, n) V_OR(V_SHR((x), (n)), V_SHL((x), (32 - (n))))
#define ROTL(x, n) V_OR(V_SHL((x), (n)), V_SHR((x), (32 - (n))))

// ---- padding (shared shape; SHA appends a big-endian bit length) ----

static uint8_t *pad4(const uint8_t *const ms[4], int n, size_t *plen_out,
                     int *blocks_out) {
  size_t plen = ((size_t)(n + 8) / 64) * 64 + 64;
  uint8_t *pad = (uint8_t *)calloc(4, plen);
  uint64_t bits = (uint64_t)n * 8;
  for (int k = 0; k < 4; k++) {
    uint8_t *p = pad + (size_t)k * plen;
    if (n > 0) memcpy(p, ms[k], (size_t)n);
    p[n] = 0x80;
    for (int i = 0; i < 8; i++) p[plen - 1 - i] = (uint8_t)(bits >> (i * 8));
  }
  *plen_out = plen;
  *blocks_out = (int)(plen / 64);
  return pad;
}

static inline uint32_t rd_be32(const uint8_t *p) {
  return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
         ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

// ---- SHA-256 ----

static const uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
static const uint32_t IV256[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372,
                                  0xa54ff53a, 0x510e527f, 0x9b05688c,
                                  0x1f83d9ab, 0x5be0cd19};

#define S0(x) V_XOR(V_XOR(ROTR(x, 7), ROTR(x, 18)), V_SHR(x, 3))
#define S1(x) V_XOR(V_XOR(ROTR(x, 17), ROTR(x, 19)), V_SHR(x, 10))
#define B0(x) V_XOR(V_XOR(ROTR(x, 2), ROTR(x, 13)), ROTR(x, 22))
#define B1(x) V_XOR(V_XOR(ROTR(x, 6), ROTR(x, 11)), ROTR(x, 25))

void mb_sha256_x4(const uint8_t *m0, const uint8_t *m1, const uint8_t *m2,
                  const uint8_t *m3, int n, uint8_t *out) {
  const uint8_t *ms[4] = {m0, m1, m2, m3};
  size_t plen;
  int blocks;
  uint8_t *pad = pad4(ms, n, &plen, &blocks);
  v4 H[8];
  for (int i = 0; i < 8; i++) H[i] = V_SET1(IV256[i]);
  v4 W[64];
  for (int blk = 0; blk < blocks; blk++) {
    for (int t = 0; t < 16; t++) {
      uint32_t w[4];
      for (int k = 0; k < 4; k++)
        w[k] = rd_be32(pad + (size_t)k * plen + (size_t)blk * 64 + t * 4);
      W[t] = v_set4(w[0], w[1], w[2], w[3]);
    }
    for (int t = 16; t < 64; t++)
      W[t] = V_ADD(V_ADD(S1(W[t - 2]), W[t - 7]),
                   V_ADD(S0(W[t - 15]), W[t - 16]));
    v4 a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6],
       h = H[7];
    for (int t = 0; t < 64; t++) {
      v4 ch = V_XOR(V_AND(e, f), V_ANDNOT(e, g));
      v4 t1 = V_ADD(V_ADD(V_ADD(h, B1(e)), V_ADD(ch, V_SET1(K256[t]))), W[t]);
      v4 mj = V_XOR(V_XOR(V_AND(a, b), V_AND(a, c)), V_AND(b, c));
      v4 t2 = V_ADD(B0(a), mj);
      h = g;
      g = f;
      f = e;
      e = V_ADD(d, t1);
      d = c;
      c = b;
      b = a;
      a = V_ADD(t1, t2);
    }
    H[0] = V_ADD(H[0], a);
    H[1] = V_ADD(H[1], b);
    H[2] = V_ADD(H[2], c);
    H[3] = V_ADD(H[3], d);
    H[4] = V_ADD(H[4], e);
    H[5] = V_ADD(H[5], f);
    H[6] = V_ADD(H[6], g);
    H[7] = V_ADD(H[7], h);
  }
  for (int i = 0; i < 8; i++) {
    uint32_t o[4];
    v_get4(H[i], o);
    for (int k = 0; k < 4; k++) {
      uint8_t *d = out + k * 32 + i * 4;
      d[0] = (uint8_t)(o[k] >> 24);
      d[1] = (uint8_t)(o[k] >> 16);
      d[2] = (uint8_t)(o[k] >> 8);
      d[3] = (uint8_t)o[k];
    }
  }
  free(pad);
}

// ---- SHA-1 ----

static const uint32_t IV1[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476,
                                0xC3D2E1F0};

void mb_sha1_x4(const uint8_t *m0, const uint8_t *m1, const uint8_t *m2,
                const uint8_t *m3, int n, uint8_t *out) {
  const uint8_t *ms[4] = {m0, m1, m2, m3};
  size_t plen;
  int blocks;
  uint8_t *pad = pad4(ms, n, &plen, &blocks);
  v4 H[5];
  for (int i = 0; i < 5; i++) H[i] = V_SET1(IV1[i]);
  v4 W[80];
  for (int blk = 0; blk < blocks; blk++) {
    for (int t = 0; t < 16; t++) {
      uint32_t w[4];
      for (int k = 0; k < 4; k++)
        w[k] = rd_be32(pad + (size_t)k * plen + (size_t)blk * 64 + t * 4);
      W[t] = v_set4(w[0], w[1], w[2], w[3]);
    }
    for (int t = 16; t < 80; t++)
      W[t] = ROTL(V_XOR(V_XOR(W[t - 3], W[t - 8]), V_XOR(W[t - 14], W[t - 16])),
                  1);
    v4 a = H[0], b = H[1], c = H[2], d = H[3], e = H[4];
    for (int t = 0; t < 80; t++) {
      v4 f;
      uint32_t k;
      if (t < 20) {
        f = V_OR(V_AND(b, c), V_ANDNOT(b, d));
        k = 0x5A827999;
      } else if (t < 40) {
        f = V_XOR(V_XOR(b, c), d);
        k = 0x6ED9EBA1;
      } else if (t < 60) {
        f = V_OR(V_OR(V_AND(b, c), V_AND(b, d)), V_AND(c, d));
        k = 0x8F1BBCDC;
      } else {
        f = V_XOR(V_XOR(b, c), d);
        k = 0xCA62C1D6;
      }
      v4 temp = V_ADD(V_ADD(V_ADD(ROTL(a, 5), f), V_ADD(e, V_SET1(k))), W[t]);
      e = d;
      d = c;
      c = ROTL(b, 30);
      b = a;
      a = temp;
    }
    H[0] = V_ADD(H[0], a);
    H[1] = V_ADD(H[1], b);
    H[2] = V_ADD(H[2], c);
    H[3] = V_ADD(H[3], d);
    H[4] = V_ADD(H[4], e);
  }
  for (int i = 0; i < 5; i++) {
    uint32_t o[4];
    v_get4(H[i], o);
    for (int k = 0; k < 4; k++) {
      uint8_t *d = out + k * 20 + i * 4;
      d[0] = (uint8_t)(o[k] >> 24);
      d[1] = (uint8_t)(o[k] >> 16);
      d[2] = (uint8_t)(o[k] >> 8);
      d[3] = (uint8_t)o[k];
    }
  }
  free(pad);
}

// ---- MD5 (little-endian; variable per-round rotation) ----

#if defined(__SSE2__)
static inline v4 v_rotl_var(v4 x, int s) {
  return _mm_or_si128(_mm_sll_epi32(x, _mm_cvtsi32_si128(s)),
                      _mm_srl_epi32(x, _mm_cvtsi32_si128(32 - s)));
}
#elif defined(__aarch64__) || defined(__ARM_NEON)
static inline v4 v_rotl_var(v4 x, int s) {
  return vorrq_u32(vshlq_u32(x, vdupq_n_s32(s)),
                   vshlq_u32(x, vdupq_n_s32(s - 32)));
}
#else
static inline v4 v_rotl_var(v4 x, int s) {
  v4 r;
  for (int i = 0; i < 4; i++) r.l[i] = (x.l[i] << s) | (x.l[i] >> (32 - s));
  return r;
}
#endif

#define V_NOT(x) V_XOR((x), V_SET1(0xFFFFFFFFu))

static const uint32_t MD5_K[64] = {
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a,
    0xa8304613, 0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340,
    0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8,
    0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
    0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92,
    0xffeff47d, 0x85845dd1, 0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391};
static const int MD5_S[64] = {7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7,
                              12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9,
                              14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16,
                              23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15, 21,
                              6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};
static const uint32_t MD5_IV[4] = {0x67452301, 0xefcdab89, 0x98badcfe,
                                   0x10325476};

// Padding with a *little-endian* 64-bit bit length (MD5).
static uint8_t *pad4_le(const uint8_t *const ms[4], int n, size_t *plen_out,
                        int *blocks_out) {
  size_t plen = ((size_t)(n + 8) / 64) * 64 + 64;
  uint8_t *pad = (uint8_t *)calloc(4, plen);
  uint64_t bits = (uint64_t)n * 8;
  for (int k = 0; k < 4; k++) {
    uint8_t *p = pad + (size_t)k * plen;
    if (n > 0) memcpy(p, ms[k], (size_t)n);
    p[n] = 0x80;
    for (int i = 0; i < 8; i++) p[plen - 8 + i] = (uint8_t)(bits >> (i * 8));
  }
  *plen_out = plen;
  *blocks_out = (int)(plen / 64);
  return pad;
}

static inline uint32_t rd_le32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

void mb_md5_x4(const uint8_t *m0, const uint8_t *m1, const uint8_t *m2,
               const uint8_t *m3, int n, uint8_t *out) {
  const uint8_t *ms[4] = {m0, m1, m2, m3};
  size_t plen;
  int blocks;
  uint8_t *pad = pad4_le(ms, n, &plen, &blocks);
  v4 A = V_SET1(MD5_IV[0]), B = V_SET1(MD5_IV[1]), C = V_SET1(MD5_IV[2]),
     D = V_SET1(MD5_IV[3]);
  v4 M[16];
  for (int blk = 0; blk < blocks; blk++) {
    for (int j = 0; j < 16; j++) {
      uint32_t w[4];
      for (int k = 0; k < 4; k++)
        w[k] = rd_le32(pad + (size_t)k * plen + (size_t)blk * 64 + j * 4);
      M[j] = v_set4(w[0], w[1], w[2], w[3]);
    }
    v4 a = A, b = B, c = C, d = D;
    for (int i = 0; i < 64; i++) {
      v4 f;
      int g;
      if (i < 16) {
        f = V_OR(V_AND(b, c), V_ANDNOT(b, d));
        g = i;
      } else if (i < 32) {
        f = V_OR(V_AND(d, b), V_ANDNOT(d, c));
        g = (5 * i + 1) & 15;
      } else if (i < 48) {
        f = V_XOR(V_XOR(b, c), d);
        g = (3 * i + 5) & 15;
      } else {
        f = V_XOR(c, V_OR(b, V_NOT(d)));
        g = (7 * i) & 15;
      }
      v4 f2 = V_ADD(V_ADD(V_ADD(f, a), V_SET1(MD5_K[i])), M[g]);
      a = d;
      d = c;
      c = b;
      b = V_ADD(b, v_rotl_var(f2, MD5_S[i]));
    }
    A = V_ADD(A, a);
    B = V_ADD(B, b);
    C = V_ADD(C, c);
    D = V_ADD(D, d);
  }
  v4 St[4] = {A, B, C, D};
  for (int i = 0; i < 4; i++) {
    uint32_t o[4];
    v_get4(St[i], o);
    for (int k = 0; k < 4; k++) {
      uint8_t *p = out + k * 16 + i * 4;
      p[0] = (uint8_t)o[k]; // little-endian
      p[1] = (uint8_t)(o[k] >> 8);
      p[2] = (uint8_t)(o[k] >> 16);
      p[3] = (uint8_t)(o[k] >> 24);
    }
  }
  free(pad);
}
