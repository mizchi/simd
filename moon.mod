name = "mizchi/simd"

version = "0.5.0"

readme = "README.md"

repository = "https://github.com/mizchi/simd"

license = "MIT"

keywords = [
  "simd",
  "wasm",
  "wasm-gc",
  "v128",
  "performance",
  "hash",
  "sha256",
]

description = "SIMD primitives for MoonBit across wasm / wasm-gc / native / js — @simdcore (faster moonbitlang/core equivalents), @simd_buffer, @simdcodec (base64), @simdhash (SHA-256/512/1, MD5 + multi-buffer), @simdimage, @simdjson."

options(
  exclude: [ "experiments" ],
)
