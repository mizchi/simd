# `@simdcodec` — byte codecs (SIMD)

Reversible byte encodings. Today: RFC 4648 **Base64** (standard alphabet,
`=` padding). The `base64_*` prefix leaves room for more codecs (hex, …)
under the same package.

```moonbit
let enc = @simdcodec.base64_encode(input)     // FixedArray[Byte] -> FixedArray[Byte]
let dec = @simdcodec.base64_decode(enc)       // -> FixedArray[Byte]?
```

For hot loops, reuse a buffer and skip the allocation with the in-place
variants (same `_into` convention as the rest of the library):

```moonbit
let out = FixedArray::make(@simdcodec.base64_encoded_len(input.length()), b'\x00')
@simdcodec.base64_encode_into(input, out)                     // -> Unit
let n = @simdcodec.base64_decode_into(enc, dst)               // -> Int?  (bytes written)
```

## Backend comparison (base64)

| backend | mechanism | accelerated? | encode (4 KiB) | decode (5.4 KiB) |
|---|---|---|---|---|
| **wasm** | inline-WAT v128 (12-in/16-out encode, 16-in/12-out decode) | ✅ SIMD | 7.45 µs → **2.06 µs (3.6×)** | 7.97 µs → **2.28 µs (3.5×)** |
| **wasm-gc** | shared scalar (`base64_fallback.mbt`) | ❌ scalar | baseline | baseline |
| **native** | C FFI (`base64.c`), gcc/clang-compiled scalar + 256-entry decode LUT | ⚠️ no SIMD, but beats tcc MoonBit scalar | 9.2 µs → **4.15 µs (2.2×)** | 8.3 µs → **4.56 µs (1.8×)** |
| **js** | shared scalar | ❌ scalar | baseline | baseline |

- **wasm** vectorises the byte-layout shuffle + the ASCII-offset arithmetic
  (5 chained `i8x16.ge_s` + `v128.and` + `i8x16.add`/`sub`), no lookup table.
- **native** can't use the SSSE3 `pshufb` byte shuffle a vectorised base64
  needs (not in the baseline x86-64 ABI without `-march`), so the C kernel is
  scalar — but it's gcc-compiled with a decode LUT, so it still beats the
  tcc-run MoonBit scalar. `SimdBufferBytes::base64_*` on native delegates here.

Speedups are vs each backend's own scalar baseline (wasm/native: V8 / gcc on
M-class Apple Silicon). The last input quad (where `=` padding may appear)
always falls through to scalar to keep the SIMD path branch-free.

Run: `moon bench -p simdcodec --target wasm` (or `--target native`).
