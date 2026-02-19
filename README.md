# mizchi/simd

SIMD abstraction module for MoonBit. Automatically selects optimized implementations per target.

## API

```moonbit
// Scalar implementations (all targets)
scalar_sum(arr : FixedArray[Int]) -> Int
scalar_dot_product(a : FixedArray[Int], b : FixedArray[Int]) -> Int
scalar_add(a : FixedArray[Int], b : FixedArray[Int], out : FixedArray[Int]) -> Unit
scalar_adler32(data : FixedArray[Byte]) -> UInt

// SIMD implementations (target-optimized)
simd_sum(arr : FixedArray[Int]) -> Int
simd_dot_product(a : FixedArray[Int], b : FixedArray[Int]) -> Int
simd_add(a : FixedArray[Int], b : FixedArray[Int], out : FixedArray[Int]) -> Unit
simd_adler32(data : FixedArray[Byte]) -> UInt
```

## Target Implementations

| Target  | `simd_*` implementation     | Speedup |
|---------|----------------------------|---------|
| native  | C FFI (NEON/SSE intrinsics) | 13–33x  |
| wasm    | scalar fallback             | none    |
| wasm-gc | scalar fallback             | none    |
| js      | scalar fallback             | none    |

## Benchmarks

### native (Apple Silicon M2)

```
scalar_sum_1024          1.60 µs
simd_sum_1024            0.12 µs   (13x)

scalar_dot_product_1024  2.07 µs
simd_dot_product_1024    0.15 µs   (14x)

scalar_add_1024          2.66 µs
simd_add_1024            0.08 µs   (33x)

scalar_adler32_4096     18.33 µs
simd_adler32_4096        1.09 µs   (17x)
```

### Analysis: Where does the speedup come from?

Comparing against standalone C compiled with clang:

| Implementation                        | sum (1024) | Notes                            |
|---------------------------------------|-----------|----------------------------------|
| MoonBit scalar (TCC)                  | 1600 ns   | baseline                         |
| clang -O0 scalar                      | 654 ns    | TCC ≈ clang -O0                  |
| clang -O2 -fno-vectorize scalar       | 251 ns    | clang optimization quality       |
| clang -O2 -fno-vectorize + NEON       | 102 ns    | explicit NEON gives 2.5x         |
| clang -O2 scalar (auto-vectorization) | 69 ns     | clang auto-vectorizes to NEON    |
| MoonBit C FFI (native-stub)           | 120 ns    | ≈ clang -O2 NEON                 |

**Conclusion**: The 13–33x speedup is primarily due to **TCC vs clang compiler quality**, not SIMD instructions alone. NEON SIMD itself contributes only ~2–2.5x. clang -O2 auto-vectorizes scalar loops, making hand-written NEON intrinsics of limited additional value.

### wasm / wasm-gc

Scalar fallback only — no SIMD benefit. However, wasm scalar is ~2x faster than native scalar (TCC) thanks to JIT optimization:

```
wasm scalar_sum_1024:   0.65 µs  (JIT outperforms TCC)
native scalar_sum_1024: 1.60 µs  (TCC codegen is slower)
```

## Limitations & Future Work

### wasm SIMD (v128) — blocked on Dwarfsm

MoonBit's inline WAT parser ([Dwarfsm](https://github.com/moonbitlang/moonbit-compiler)) does not fully support v128 SIMD instructions, preventing wasm SIMD optimization.

**Dwarfsm v128 support status** (verified by reading source / testing):

| Feature | Status | Notes |
|---------|--------|-------|
| `v128` as valtype (`param`/`result`) | ✅ Works | |
| `v128.load` / `v128.store` | ✅ Works | Memory access |
| `f32x4.add`, `f32x4.mul`, `f64x2.add`, `f64x2.mul` | ✅ Works | Float SIMD ops |
| `v128.const` | ❌ Parse error | Constant initialization |
| `i32x4.add`, `i32x4.mul`, etc. | ❌ Parse error | Integer SIMD ops |
| `i32x4.extract_lane`, `i8x16.shuffle` | ❌ Parse error | Lane manipulation |

**Two remaining blockers**:

1. **Missing instruction table entries**: `i32x4.*`, `v128.const`, etc. are not registered in `dwarfsm_parse.ml`. These can be added by extending the pattern table.
2. **`#external type` maps to `externref`**: `#external type V128` compiles to `externref` in wasm, causing type mismatch with inline WAT's `v128`. A language-level `v128` type or control over `#external` wasm representation is needed.

Related: [moonbitlang/core#2844](https://github.com/moonbitlang/core/issues/2844)

### wasm-gc SIMD — fundamentally blocked

GC-heap arrays cannot be addressed with `v128.load` — SIMD requires linear memory access.

### Native compiler backend

MoonBit currently uses TCC as the default native compiler. If a clang/gcc backend is supported in the future, auto-vectorization would make C FFI SIMD largely unnecessary. The `native-stub` C files are compiled separately with the system compiler (clang on macOS), which is why C FFI already benefits from optimization.

### Potential additions

- `simd_min` / `simd_max` (array min/max)
- `simd_memcmp` (byte array comparison)
- Float support (`FixedArray[Double]` → `f64x2`)

## Usage

```bash
just test          # All targets
just test-native   # native only
just bench-native  # native benchmark
just bench-wasm    # wasm benchmark
```

## File Structure

```
src/
├── moon.pkg.json       # Target-specific files + native-stub config
├── scalar.mbt          # Scalar implementations (all targets)
├── simd_wasm.mbt       # wasm: scalar fallback
├── simd_wasm_gc.mbt    # wasm-gc: scalar fallback
├── simd_native.mbt     # native: extern "C" FFI declarations
├── simd_native.c       # C SIMD (NEON/SSE + scalar fallback)
├── simd_js.mbt         # js: scalar fallback
├── lib_test.mbt        # Correctness tests (17 tests)
└── lib_bench.mbt       # Benchmarks
```

## License

MIT
