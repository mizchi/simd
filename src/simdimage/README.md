# `@simdimage` — SIMD image / pixel ops

Pixel-oriented byte kernels that operate on core types directly — `Bytes`
(read-only input) / `FixedArray[Byte]` (mutable output) — so an image
library can call in with no `SimdBuffer` copy hop.

```moonbit
let rgb = Bytes::from_array(pixels)            // n*3 RGB bytes
let rgba : FixedArray[Byte] = FixedArray::make(n * 4, b'\x00')
@simdimage.rgb_to_rgba(rgb, 0xFF, rgba)        // -> RGBA, opaque
```

Ops: `rgb_to_rgba`, `rgba_to_grayscale`, `channel_extract`,
`channel_merge`, `lerp`, `alpha_blend_solid`, `histogram`.

## Backend comparison

| backend | mechanism | accelerated? |
|---|---|---|
| **wasm** | inline-WAT v128 (`i8x16.shuffle` / `i16x8.extmul` fixed-point); `Bytes` / `FixedArray` cross the FFI as linear-memory pointers | ✅ SIMD |
| **wasm-gc** | scalar (GC-ref FFI blocks `v128.load`) | ❌ scalar |
| **native** | scalar `FixedArray` loops (no C kernel yet — `pshufb` is SSSE3, not baseline) | ❌ scalar |
| **js** | scalar | ❌ scalar |

Results are **byte-identical on all four backends**; only `wasm` is
accelerated. `histogram` is scalar everywhere (wasm SIMD has no scatter).

### Per-op speedup (wasm, V8, 4096 px, vs a plain per-byte loop)

| op | scalar | wasm SIMD | x |
|---|---|---|---|
| `alpha_blend_solid` | 57.3 µs | 3.09 µs | **18.5** |
| `rgb_to_rgba` | 25.6 µs | 1.63 µs | **15.7** |
| `lerp` (4096 B) | 12.0 µs | 866 ns | **13.9** |
| `channel_merge` | 27.0 µs | 2.54 µs | **10.6** |
| `channel_extract` | 7.33 µs | 1.10 µs | **6.7** |
| `rgba_to_grayscale` | 16.2 µs | 4.84 µs | **3.3** |

The wasm-gc / native / js columns would all read **1.0× (scalar baseline)** —
they run the same per-byte loop the "scalar" column measures.

Run: `moon bench --target wasm -p simdimage`.
