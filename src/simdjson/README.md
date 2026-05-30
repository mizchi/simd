# `@simdjson` — JSON structural indexing

Port of simdjson's `find_structural_bits` byte-classification pipeline.
Operates on `@simd_buffer.SimdBufferBytes` (input) and
`@simd_buffer.SimdBuffer` (i32 bitmaps / indices output).

```moonbit
let input = @simd_buffer.SimdBufferBytes::from_array(json_bytes)
let words = (input.length() + 31) / 32
let structural = @simd_buffer.SimdBuffer::make(words)
let quote_mask = @simd_buffer.SimdBuffer::make(words)
let indices = @simd_buffer.SimdBuffer::make(input.length())
let count = @simdjson.find_structural_indices_with_scratch(
  input, structural, quote_mask, indices,
)
```

## Backend comparison

| backend | mechanism | accelerated? |
|---|---|---|
| **wasm** | inline-WAT v128 (`i8x16.eq` + `i8x16.bitmask`) | ✅ SIMD |
| **wasm-gc** | inline-WAT v128 (same bodies — linear-memory `SimdBuffer`) | ✅ SIMD |
| **native** | `FixedArray` scalar | ❌ scalar |
| **js** | `FixedArray` scalar | ❌ scalar |

The byte-classification phases vectorise cleanly; the pipeline as a whole is
bounded by `compute_quote_mask`, which **stays scalar on every backend** —
wasm SIMD has no CLMUL to prefix-XOR the quote bitmap (the documented floor).

### Per-phase speedup (wasm vs native scalar, 4 KiB JSON, V8, Apple Silicon)

| phase | wasm vs native scalar |
|---|---|
| `classify_structural` (`{ } [ ] , :`) | **6.9×** |
| `classify_numeric` (`0-9 - + . e E`) | **5.6×** |
| `classify_quote_raw` (`"`) | **6.6×** |
| `extract_structural_indices` (`i32.ctz` bit-walk) | **11×** |
| `compute_quote_mask` (serial prefix-XOR, no CLMUL) | **0.46×** (loss) |
| `find_structural_indices_with_scratch` (full pipeline) | **1.27×** |

Use the `classify_*` primitives where you don't need in-string exclusion
(counting, locating, minify pre-scan) — those are the 5–11× wins. Reach for
the full indexer only when you need quote-aware offsets.

Run: `moon bench --target wasm -p simdjson` (or `--target wasm-gc`).
