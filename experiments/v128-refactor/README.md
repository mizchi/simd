# V128 refactor experiment

This directory keeps the June 2026 MoonBit `V128` refactor prototype out of
the production package.

The prototype rewrote many wasm inline-WAT kernels into normal MoonBit loops
calling thin `V128` helpers. That made the code easier to read, but benchmark
results showed large slowdowns on several hot paths compared with the original
single-WAT kernels.

Adopted in `src`:

- `popcount_bytes_v128`
- `find_byte_v128`
- `find_byte_b_v128`
- `first_non_ascii_chunk_v128`

Rejected for production for now:

- i32 element-wise and reductions
- f64 element-wise and reductions
- f32 element-wise and reductions
- byte memcpy/memset/equality/count/ascii transforms

Representative wasm benchmark deltas against `HEAD` before this experiment:

| bench | old WAT | V128 helper prototype |
|---|---:|---:|
| `simd_popcount_4096` | 296ns | 183ns |
| `simd_find_byte_4096` | 280ns | 216ns |
| `simdcore_bytes_search` | 231ns | 204ns |
| `simd_utf8_ascii_4096` | 346ns | 191ns |
| `bytes_index_of_simdcore` | 162ns | rejected: 206ns |
| `simd_add_1024` | 118ns | 202ns |
| `simd_sub_1024` | 119ns | 190ns |
| `simd_add_f64_1024` | 233ns | 409ns |
| `simd_add_f32_1024` | 119ns | 190ns |
| `simd_memcpy_4096` | 85ns | 136ns |
| `simd_equal_4096` | 182ns | 237ns |

Keep this directory as reference material until MoonBit's V128 lowering or
stdlib SIMD APIs make the helper style competitive.
