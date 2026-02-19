#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#if defined(__aarch64__)
  #include <arm_neon.h>
  #define USE_NEON 1
#else
  #define USE_NEON 0
#endif

// ============================================================================
// Scalar implementations (same logic as MoonBit scalar.mbt)
// ============================================================================

int32_t scalar_sum(const int32_t* arr, int32_t len) {
  int32_t result = 0;
  for (int32_t i = 0; i < len; i++) {
    result += arr[i];
  }
  return result;
}

int32_t scalar_dot_product(const int32_t* a, const int32_t* b, int32_t len) {
  int32_t result = 0;
  for (int32_t i = 0; i < len; i++) {
    result += a[i] * b[i];
  }
  return result;
}

void scalar_add(const int32_t* a, const int32_t* b, int32_t* out, int32_t len) {
  for (int32_t i = 0; i < len; i++) {
    out[i] = a[i] + b[i];
  }
}

uint32_t scalar_adler32(const uint8_t* data, int32_t len) {
  uint32_t a = 1, b = 0;
  const uint32_t MOD = 65521;
  int32_t i = 0;
  while (i < len) {
    int32_t block = len - i;
    if (block > 5552) block = 5552;
    int32_t end = i + block;
    for (; i < end; i++) {
      a += data[i];
      b += a;
    }
    a %= MOD;
    b %= MOD;
  }
  return (b << 16) | a;
}

// ============================================================================
// NEON SIMD implementations
// ============================================================================

#if USE_NEON
int32_t neon_sum(const int32_t* arr, int32_t len) {
  int32_t result = 0;
  int32_t i = 0;
  int32x4_t acc = vdupq_n_s32(0);
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4) {
    acc = vaddq_s32(acc, vld1q_s32(arr + i));
  }
  result = vaddvq_s32(acc);
  for (; i < len; i++) {
    result += arr[i];
  }
  return result;
}

int32_t neon_dot_product(const int32_t* a, const int32_t* b, int32_t len) {
  int32_t result = 0;
  int32_t i = 0;
  int32x4_t acc = vdupq_n_s32(0);
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4) {
    acc = vmlaq_s32(acc, vld1q_s32(a + i), vld1q_s32(b + i));
  }
  result = vaddvq_s32(acc);
  for (; i < len; i++) {
    result += a[i] * b[i];
  }
  return result;
}

void neon_add(const int32_t* a, const int32_t* b, int32_t* out, int32_t len) {
  int32_t i = 0;
  int32_t end4 = (len / 4) * 4;
  for (; i < end4; i += 4) {
    vst1q_s32(out + i, vaddq_s32(vld1q_s32(a + i), vld1q_s32(b + i)));
  }
  for (; i < len; i++) {
    out[i] = a[i] + b[i];
  }
}
#endif

// ============================================================================
// Benchmark harness
// ============================================================================

typedef struct {
  double mean_ns;
  double min_ns;
  double max_ns;
} BenchResult;

static double clock_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e9 + ts.tv_nsec;
}

// Prevent dead-code elimination
static volatile int32_t sink_i32;
static volatile uint32_t sink_u32;

#define WARMUP_ITERS 1000
#define BENCH_ITERS  100000

BenchResult bench_fn_i32(int32_t (*fn)(const int32_t*, int32_t), const int32_t* data, int32_t len) {
  for (int i = 0; i < WARMUP_ITERS; i++) sink_i32 = fn(data, len);
  double total = 0, min_t = 1e18, max_t = 0;
  for (int i = 0; i < BENCH_ITERS; i++) {
    double t0 = clock_ns();
    sink_i32 = fn(data, len);
    double dt = clock_ns() - t0;
    total += dt;
    if (dt < min_t) min_t = dt;
    if (dt > max_t) max_t = dt;
  }
  return (BenchResult){ total / BENCH_ITERS, min_t, max_t };
}

typedef int32_t (*dot_fn_t)(const int32_t*, const int32_t*, int32_t);
BenchResult bench_dot(dot_fn_t fn, const int32_t* a, const int32_t* b, int32_t len) {
  for (int i = 0; i < WARMUP_ITERS; i++) sink_i32 = fn(a, b, len);
  double total = 0, min_t = 1e18, max_t = 0;
  for (int i = 0; i < BENCH_ITERS; i++) {
    double t0 = clock_ns();
    sink_i32 = fn(a, b, len);
    double dt = clock_ns() - t0;
    total += dt;
    if (dt < min_t) min_t = dt;
    if (dt > max_t) max_t = dt;
  }
  return (BenchResult){ total / BENCH_ITERS, min_t, max_t };
}

typedef void (*add_fn_t)(const int32_t*, const int32_t*, int32_t*, int32_t);
BenchResult bench_add(add_fn_t fn, const int32_t* a, const int32_t* b, int32_t* out, int32_t len) {
  for (int i = 0; i < WARMUP_ITERS; i++) fn(a, b, out, len);
  double total = 0, min_t = 1e18, max_t = 0;
  for (int i = 0; i < BENCH_ITERS; i++) {
    double t0 = clock_ns();
    fn(a, b, out, len);
    double dt = clock_ns() - t0;
    total += dt;
    if (dt < min_t) min_t = dt;
    if (dt > max_t) max_t = dt;
  }
  return (BenchResult){ total / BENCH_ITERS, min_t, max_t };
}

BenchResult bench_adler32(uint32_t (*fn)(const uint8_t*, int32_t), const uint8_t* data, int32_t len) {
  for (int i = 0; i < WARMUP_ITERS; i++) sink_u32 = fn(data, len);
  double total = 0, min_t = 1e18, max_t = 0;
  for (int i = 0; i < BENCH_ITERS; i++) {
    double t0 = clock_ns();
    sink_u32 = fn(data, len);
    double dt = clock_ns() - t0;
    total += dt;
    if (dt < min_t) min_t = dt;
    if (dt > max_t) max_t = dt;
  }
  return (BenchResult){ total / BENCH_ITERS, min_t, max_t };
}

static void print_result(const char* name, BenchResult r) {
  if (r.mean_ns >= 1000.0) {
    printf("  %-30s %8.2f µs  (min %7.2f, max %7.2f)\n",
           name, r.mean_ns / 1000.0, r.min_ns / 1000.0, r.max_ns / 1000.0);
  } else {
    printf("  %-30s %8.1f ns  (min %7.1f, max %7.1f)\n",
           name, r.mean_ns, r.min_ns, r.max_ns);
  }
}

static void print_speedup(const char* label, BenchResult base, BenchResult fast) {
  printf("  %-30s %.1fx\n", label, base.mean_ns / fast.mean_ns);
}

int main(void) {
  const int32_t N = 1024;
  const int32_t N_BYTE = 4096;

  int32_t* arr = malloc(N * sizeof(int32_t));
  int32_t* arr2 = malloc(N * sizeof(int32_t));
  int32_t* out = malloc(N * sizeof(int32_t));
  uint8_t* bdata = malloc(N_BYTE);
  for (int32_t i = 0; i < N; i++) { arr[i] = i + 1; arr2[i] = i + 1; }
  for (int32_t i = 0; i < N_BYTE; i++) { bdata[i] = (i % 256) + 1; }

  printf("=== sum (N=%d) ===\n", N);
  BenchResult r_scalar_sum = bench_fn_i32(scalar_sum, arr, N);
  print_result("clang scalar", r_scalar_sum);
#if USE_NEON
  BenchResult r_neon_sum = bench_fn_i32(neon_sum, arr, N);
  print_result("clang NEON", r_neon_sum);
  print_speedup("NEON vs scalar", r_scalar_sum, r_neon_sum);
#endif
  printf("\n");

  printf("=== dot_product (N=%d) ===\n", N);
  BenchResult r_scalar_dot = bench_dot(scalar_dot_product, arr, arr2, N);
  print_result("clang scalar", r_scalar_dot);
#if USE_NEON
  BenchResult r_neon_dot = bench_dot(neon_dot_product, arr, arr2, N);
  print_result("clang NEON", r_neon_dot);
  print_speedup("NEON vs scalar", r_scalar_dot, r_neon_dot);
#endif
  printf("\n");

  printf("=== add (N=%d) ===\n", N);
  BenchResult r_scalar_add = bench_add(scalar_add, arr, arr2, out, N);
  print_result("clang scalar", r_scalar_add);
#if USE_NEON
  BenchResult r_neon_add = bench_add(neon_add, arr, arr2, out, N);
  print_result("clang NEON", r_neon_add);
  print_speedup("NEON vs scalar", r_scalar_add, r_neon_add);
#endif
  printf("\n");

  printf("=== adler32 (N=%d) ===\n", N_BYTE);
  BenchResult r_scalar_adler = bench_adler32(scalar_adler32, bdata, N_BYTE);
  print_result("clang scalar", r_scalar_adler);
  printf("\n");

  // Summary comparison with MoonBit bench results
  printf("=== MoonBit (TCC) reference (from moon bench) ===\n");
  printf("  scalar_sum_1024:          1.60 µs\n");
  printf("  scalar_dot_product_1024:  2.07 µs\n");
  printf("  scalar_add_1024:          2.66 µs\n");
  printf("  scalar_adler32_4096:     18.33 µs\n");
  printf("  simd_sum_1024 (C FFI):    0.12 µs\n");
  printf("  simd_dot_1024 (C FFI):    0.15 µs\n");
  printf("  simd_add_1024 (C FFI):    0.08 µs\n");
  printf("  simd_adler32  (C FFI):    1.09 µs\n");

  free(arr); free(arr2); free(out); free(bdata);
  return 0;
}
