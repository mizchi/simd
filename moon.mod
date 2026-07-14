name = "mizchi/simd"

version = "0.6.1"

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

description = "SIMD primitives for MoonBit across wasm / wasm-gc / native / js — @simdcore, @simd_buffer, codecs, hashes, images, JSON indexing, and the experimental mizchi/simd/json parser."

options(
  exclude: [ "experiments" ],
)
