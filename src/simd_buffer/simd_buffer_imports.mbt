// The simd_buffer package imports `@simdcodec` and `@internal` at the
// package level. Native consumes both via `simd_buffer_native.mbt`, but
// on wasm / wasm-gc the inline-WAT files bypass both packages, and on
// js the package has no source files at all. To keep
// `moon check --deny-warn` happy across every target, this file lives in
// all four target lists and emits a tiny no-op reference per package.

///|
fn _imports_keep_alive() -> Int {
  let empty : FixedArray[Int] = FixedArray::make(0, 0)
  @simdcodec.base64_encoded_len(0) + @internal.scalar_sum(empty)
}
