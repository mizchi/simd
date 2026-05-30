# Proposal: First-class SIMD for MoonBit (and unblocking wasm-gc)

This document summarises what `mizchi/simd` learned by hand-porting SIMD
kernels to MoonBit's `wasm` target via inline-WAT, what the **hard wall on
`wasm-gc`** is, and what the MoonBit compiler / toolchain could do to remove
it. It is written as input for the MoonBit team; the concrete data behind
every claim lives in this repo's `CLAUDE.md` and benches.

## TL;DR

- On the `wasm` target, `FixedArray[T]` / `Bytes` / `String` cross the
  inline-WAT FFI as **linear-memory pointers**, so `v128.load` works and we get
  3–90× speedups across reductions, byte ops, codecs, hashing, and sort.
- On the `wasm-gc` target the **same** values arrive as **GC references**.
  inline-WAT cannot obtain a linear-memory address from them, so `v128.load`
  is unusable. Every `Bytes`/`FixedArray`-direct SIMD op falls back to scalar
  on wasm-gc.
- The only portable escape today is to copy the data into a linear-memory
  `SimdBuffer` (allocated via `memory.grow`), run the kernel, and copy back.
  This wins **only** for compute-heavy kernels with buffer reuse (sort 6.4×,
  popcount 3.0×, adler32 2.6×) and **loses** for light element-wise / reduction
  ops where the copy hop dwarfs the work (sum 0.49×, dot 0.43×). A
  per-call-allocating convenience API (`@simdcore.*_via_buffer`) is a flat loss
  (~800–920×) because of the per-call `memory.grow`.
- **The root blocker is in the compiler / toolchain, and is fixable there.**
  This proposal lists the options in order of leverage.

## Background: why `wasm` works and `wasm-gc` does not

Verified facts in the current toolchain (probes preserved in `CLAUDE.md`):

| value | `wasm` FFI ABI | `wasm-gc` FFI ABI |
|---|---|---|
| `FixedArray[Int]` / `[Double]` / `[Int64]` | ptr to element[0] in linear memory | `(ref $array)` GC ref |
| `FixedArray[UInt16]` | ptr (u16-addressable) | GC ref |
| `Bytes` | ptr to byte[0] | GC ref |
| `String` | ptr to UTF-16 code-unit buffer | GC ref |
| `FixedArray[Byte]` | ptr to byte[0] | **rejected: `Invalid stub type`** |

On `wasm` the pointer feeds `v128.load` / `v128.store` directly. On `wasm-gc`:

- `(param i32)` in the WAT signature → link error `expected i32, found (ref N)`.
- `(param anyref)` is accepted, but `array.get` needs a **concrete type
  index** that inline-WAT cannot name, so elements can't be read through the
  ref. Even if they could, building a `v128` from 4 scalar `array.get` +
  `i32x4.replace_lane` costs ~9 ops vs 4 for a scalar add — the SIMD win
  evaporates for memory-bound ops.
- `i32.store` / `v128.load` etc. **do** work inside wasm-gc inline-WAT (wasm-gc
  modules still have linear memory). The blocker is purely *getting a
  linear-memory address out of a GC-managed array*.

So `wasm-gc` SIMD only happens today through `@simd_buffer.SimdBuffer*`, whose
backing store is raw linear memory (`memory.grow`) rather than a GC array.

## Measured cost of the workaround (the "Option A" data)

"Option A" = a wasm-gc caller holding `FixedArray`/`Bytes` copies into a
`SimdBuffer`, runs the SIMD kernel, copies back. Measured on Linux x86-64 /
moonrun (ratios are the point, not absolute times):

| op | size | scalar (FixedArray loop) | reuse (copy + SIMD) | x |
|---|---|---|---|---|
| `sum` (i32) | 1024 | 817 ns | 1.65 µs | **0.49** (loss) |
| `dot` (i32) | 1024 | 1.42 µs | 3.28 µs | **0.43** (loss) |
| `sort` (i32) | 1024 | 168 µs | 26.2 µs | **6.4** |
| `popcount` (bytes) | 4096 | 16.4 µs | 5.42 µs | **3.0** |
| `adler32` (bytes) | 4096 | 15.1 µs | 5.85 µs | **2.6** |

And the per-call-allocating convenience API (`@simdcore.*_via_buffer`), which
matches the simple `FixedArray`-in signature by allocating a fresh `SimdBuffer`
each call:

| op (1024, wasm-gc) | scalar simdcore | `*_via_buffer` |
|---|---|---|
| `sum` | 809 ns | 649 µs (~800× slower) |
| `dot` | 1.43 µs | 1.32 ms (~920× slower) |
| `sort` | 5.73 µs | 1.25 ms (~220× slower) |

**Takeaway:** the copy hop + `memory.grow` is the dominant cost. The user
should not have to think about this — the compiler should let a GC array feed
SIMD directly, or vectorise the loop itself.

## What the compiler / toolchain could do

In descending order of leverage.

### (A) Auto-vectorisation of plain loops — the real fix

Lower ordinary MoonBit loops over `FixedArray[T]` / `Bytes` to the wasm SIMD
proposal's `v128` ops in the backend (LLVM-style loop vectoriser, or a
pattern-based pass for the common reduction / map / scan shapes).

- **Pro:** zero user-facing API; *every* target benefits (wasm, wasm-gc, and
  the LLVM-backed native path), not just hand-written kernels. `mizchi/simd`'s
  inline-WAT becomes unnecessary for the bread-and-butter ops.
- **Pro:** sidesteps the GC-ref problem entirely — the vectoriser runs after
  the array's storage is known to the backend, not at the inline-WAT FFI
  boundary.
- **Con:** the largest effort; horizontal reductions, masked tails, and
  cross-lane scans (prefix sum) need real vectoriser smarts. But even covering
  element-wise map + simple reductions would erase most of the wasm-gc gap.
- MoonBit emits to LLVM (native) and wasm, so the infrastructure to build on
  exists.

### (B) A safe "pin / view as linear-memory pointer" intrinsic — the surgical fix

Expose a builtin that yields a linear-memory address (and length) for a GC
array's element storage, valid for the duration of a call, with the GC
pinned / move-disabled across it:

```
// strawman
fn FixedArray::with_linear_ptr[T, R](self : FixedArray[T], f : (Int) -> R) -> R
// or an unsafe lower-level:
fn FixedArray::unsafe_linear_addr[T](self : FixedArray[T]) -> Int  // pinned region
```

- **Pro:** the existing inline-WAT kernels work **unchanged** on wasm-gc — the
  address just flows into `v128.load`. The whole `SimdBuffer` copy hop and the
  `*_via_buffer` slowness disappear; `@simdcore` / `@simdimage` / `@simdhash`
  Bytes-direct surfaces light up on wasm-gc for free.
- **Pro:** much smaller than a vectoriser; it's a codegen + GC-cooperation
  feature.
- **Con:** needs GC support for pinning (or a guarantee that the wasm-gc
  collector doesn't relocate during a synchronous inline-WAT call — likely true
  today since it's single-threaded and collection points are explicit, but it
  must be *specified*, not incidental).
- **Risk:** an `unsafe` raw-address form is sharp; the scoped `with_linear_ptr`
  form is safer and still zero-copy.

### (C) Let inline-WAT read GC arrays directly — the narrow fix

Teach the Dwarfsm inline-WAT parser to resolve a MoonBit array type to its
wasm-gc concrete type index, so `array.get` / `array.len` / `array.set` are
usable on `(param (ref $T))`.

- **Pro:** no GC pinning needed; stays within the GC model.
- **Con:** as measured, building a `v128` from 4× `array.get` +
  `i32x4.replace_lane` is ~9 ops vs 4 for scalar — a **net loss for
  memory-bound ops** (sum/add/dot). Only helps compute-heavy kernels, i.e. the
  same ones (B)/(A) already cover better. Lowest leverage; listed for
  completeness.

### (D) Toolchain-local parser fixes — independent of the above

These are inline-WAT parser gaps, fixable in the toolchain without touching the
GC story, that today force awkward workarounds (documented in `CLAUDE.md`):

- `v128.const i32x4 ...` trips `Int32.of_string` — we synthesise constants via
  `i32.const N` + `i32x4.splat` or load from a `FixedArray[Byte]` table.
- `f64.const` / `f32.const` literals trip the same int-decoder — synthesised via
  `i32.const N` + `f64.convert_i32_s`.
- `f64.min` / `f64.max` / `f32.min` / `f32.max` (core wasm ops) are rejected —
  we emulate with `select` + `f64.lt`/`gt`. (The vector `f64x2.min`/`max`
  parse fine.)

Fixing these would make hand-written kernels cleaner but does **not** unblock
wasm-gc; (A) or (B) is required for that.

## What the compiler *cannot* fix (wasm spec limits)

For completeness, so the proposal isn't mistaken for a cure-all:

- **No CLMUL** in wasm SIMD → `compute_quote_mask` (simdjson) and a pure-SIMD
  CRC32 have no carry-less-multiply path; they stay serial / slicing-by-8.
- **No SHA-NI** → single-message SHA-256/512 is an irreducible serial round
  chain; the win is multi-buffer (`*_x4`), not per-message.
- **No per-byte variable shift** (worked around with SWAR) and other
  instruction-set gaps are `relaxed-simd` / future-extension territory, not
  compiler issues.

These are the algorithm-shape limits; SIMD helps where the work is SIMD-shaped
*and* large enough to amortise data movement (same boundary as the `simd_any`
and Option A notes).

## Recommendation

1. **Pursue (A) auto-vectorisation** as the strategic fix — it makes SIMD
   transparent across all targets and retires most hand-written inline-WAT.
2. **Ship (B) a scoped pin/view intrinsic** as the near-term unblock for
   wasm-gc — small, and it lights up the entire existing `Bytes`/`FixedArray`-
   direct kernel surface there with no copy hop.
3. **Land (D) parser fixes** opportunistically — low effort, improves kernel
   authoring quality regardless of (A)/(B).
4. Treat (C) as not worth it on its own.

Until any of these land, `mizchi/simd`'s guidance stands: on wasm-gc reach for
`@simd_buffer` (linear-memory backed) for compute-heavy kernels with buffer
reuse, and keep the scalar fallback for light ops — see the "Option A" and
SimdBuffer sections of `CLAUDE.md`.
