default:
  just --list

check:
  moon check --deny-warn --target wasm
  moon check --deny-warn --target wasm-gc
  moon check --deny-warn --target native
  moon check --deny-warn --target js

test:
  moon test --target wasm
  moon test --target wasm-gc
  moon test --target native
  moon test --target js

test-wasm:
  moon test --target wasm

test-native:
  moon test --target native

bench-native:
  moon bench --target native

bench-wasm:
  moon bench --target wasm

fmt:
  moon fmt

info:
  moon info
