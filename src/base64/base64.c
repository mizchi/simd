// Native scalar Base64 (RFC 4648, standard alphabet) for the `native` target.
// gcc/clang compiles this stub (not tcc), so the bounds-check-free C loop
// beats the tcc-run MoonBit scalar. A vectorised SSE/NEON base64 needs SSSE3
// pshufb (not in the baseline x86-64 ABI), so this stays scalar.
#include <stdint.h>

static const char ENC[64] =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

// 256-entry decode LUT (branchless, like the MoonBit scalar). Built once.
static int8_t DEC[256];
static int dec_ready = 0;

static void ensure_dec(void) {
  if (dec_ready) return;
  for (int i = 0; i < 256; i++) DEC[i] = -1;
  for (int i = 0; i < 26; i++) DEC['A' + i] = (int8_t)i;
  for (int i = 0; i < 26; i++) DEC['a' + i] = (int8_t)(26 + i);
  for (int i = 0; i < 10; i++) DEC['0' + i] = (int8_t)(52 + i);
  DEC['+'] = 62;
  DEC['/'] = 63;
  dec_ready = 1;
}

void b64_encode(const uint8_t* in, int32_t n, uint8_t* out) {
  int32_t triples = n / 3, q = 0;
  for (int32_t i = 0; i < triples; i++) {
    const uint8_t* p = in + i * 3;
    uint32_t v = ((uint32_t)p[0] << 16) | ((uint32_t)p[1] << 8) | p[2];
    out[q] = ENC[(v >> 18) & 63];
    out[q + 1] = ENC[(v >> 12) & 63];
    out[q + 2] = ENC[(v >> 6) & 63];
    out[q + 3] = ENC[v & 63];
    q += 4;
  }
  int32_t rem = n - triples * 3;
  const uint8_t* p = in + triples * 3;
  if (rem == 1) {
    uint32_t v = (uint32_t)p[0] << 16;
    out[q] = ENC[(v >> 18) & 63];
    out[q + 1] = ENC[(v >> 12) & 63];
    out[q + 2] = '=';
    out[q + 3] = '=';
  } else if (rem == 2) {
    uint32_t v = ((uint32_t)p[0] << 16) | ((uint32_t)p[1] << 8);
    out[q] = ENC[(v >> 18) & 63];
    out[q + 1] = ENC[(v >> 12) & 63];
    out[q + 2] = ENC[(v >> 6) & 63];
    out[q + 3] = '=';
  }
}

// Returns the number of decoded bytes, or -1 on invalid input.
int32_t b64_decode(const uint8_t* in, int32_t n, uint8_t* out) {
  if (n == 0) return 0;
  if (n % 4 != 0) return -1;
  ensure_dec();
  int pad = (in[n - 1] == '=') ? ((in[n - 2] == '=') ? 2 : 1) : 0;
  int32_t full = (n - 4) / 4, q = 0;
  for (int32_t i = 0; i < full; i++) {
    const uint8_t* p = in + i * 4;
    int d0 = DEC[p[0]], d1 = DEC[p[1]], d2 = DEC[p[2]], d3 = DEC[p[3]];
    if ((d0 | d1 | d2 | d3) < 0) return -1;
    uint32_t v = ((uint32_t)d0 << 18) | (d1 << 12) | (d2 << 6) | d3;
    out[q] = (uint8_t)(v >> 16);
    out[q + 1] = (uint8_t)(v >> 8);
    out[q + 2] = (uint8_t)v;
    q += 3;
  }
  const uint8_t* p = in + full * 4;
  int d0 = DEC[p[0]], d1 = DEC[p[1]];
  if ((d0 | d1) < 0) return -1;
  if (pad == 2) {
    out[q] = (uint8_t)(((uint32_t)d0 << 18 | d1 << 12) >> 16);
    return q + 1;
  }
  int d2 = DEC[p[2]];
  if (d2 < 0) return -1;
  if (pad == 1) {
    uint32_t v = ((uint32_t)d0 << 18) | (d1 << 12) | (d2 << 6);
    out[q] = (uint8_t)(v >> 16);
    out[q + 1] = (uint8_t)(v >> 8);
    return q + 2;
  }
  int d3 = DEC[p[3]];
  if (d3 < 0) return -1;
  uint32_t v = ((uint32_t)d0 << 18) | (d1 << 12) | (d2 << 6) | d3;
  out[q] = (uint8_t)(v >> 16);
  out[q + 1] = (uint8_t)(v >> 8);
  out[q + 2] = (uint8_t)v;
  return q + 3;
}
