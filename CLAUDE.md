# mizchi/simd - MoonBit SIMD Abstraction

## Architecture

Target-specific SIMD implementations with scalar fallback.

- `src/scalar.mbt` - Scalar implementations (all targets)
- `src/simd_wasm.mbt` - wasm: scalar fallback (Dwarfsm parser lacks v128 support)
- `src/simd_wasm_gc.mbt` - wasm-gc: scalar fallback (GC heap, no v128.load)
- `src/simd_native.mbt` - native: extern "C" FFI
- `src/simd_native.c` - C SIMD intrinsics (NEON/SSE)
- `src/simd_js.mbt` - js: scalar fallback

## Key Findings

- MoonBit's inline WAT parser (Dwarfsm) does NOT support v128/SIMD instructions
- TCC (default native compiler) doesn't support NEON/SSE intrinsics, but C FFI still provides significant speedup via optimized scalar code
- `native-stub` in moon.pkg.json links C source files for native target
- `#borrow(param)` annotation needed for FixedArray/Bytes FFI parameters

## Commands

```bash
just test          # All targets
just test-wasm     # wasm only
just test-native   # native only
just bench-native  # Benchmark on native
just bench-wasm    # Benchmark on wasm
```
