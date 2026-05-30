# `@base64` — RFC 4648 Base64 (SIMD)

Standard-alphabet Base64 encode / decode with `=` padding.

```moonbit
let enc = @base64.encode(input)          // FixedArray[Byte] -> FixedArray[Byte]
let dec = @base64.decode(enc)            // -> FixedArray[Byte]?
```

## Backend comparison

| backend | mechanism | accelerated? | encode (4 KiB) | decode (5.4 KiB) |
|---|---|---|---|---|
| **wasm** | inline-WAT v128 (12-in/16-out encode, 16-in/12-out decode) | ✅ SIMD | 7.45 µs → **2.06 µs (3.6×)** | 7.97 µs → **2.28 µs (3.5×)** |
| **wasm-gc** | shared scalar (`base64_common.mbt`) | ❌ scalar | baseline | baseline |
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

Run: `moon bench -p base64 --target wasm` (or `--target native`).
