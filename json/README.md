# `mizchi/simd/json` — SIMD-assisted UTF-8 JSON parser

Experimental JSON parser that returns MoonBit's standard `Json` value. It
accepts UTF-8 `Bytes`, builds a token-position tape with context carving, then
parses from that tape without rescanning whitespace, string boundaries, or
atom bodies.

Add the containing module first:

```bash
moon add mizchi/simd
```

```moonbit
let input = @utf8.encode("{\"name\":\"moon\",\"items\":[1,2,3]}"[:])
let value = try! @simd_json.parse(input)
```

Consumer package:

```moonbit
import {
  "mizchi/simd/json" @simd_json,
  "moonbitlang/core/encoding/utf8",
}
```

`moon add` operates on the module (`mizchi/simd`), while `moon.pkg` imports
this package by its full path (`mizchi/simd/json`). The package intentionally
lives at top-level `json/`; placing it at `src/json/` would expose
`mizchi/simd/src/json` instead.

`parse_string(StringView)` is also available. It adaptively delegates small or
string-dominated documents to core/json, and only pays for UTF-8 encoding when
the sampled token density predicts a win from the indexed pipeline. Prefer
`parse(Bytes)` when the application already receives UTF-8 from a file,
network response, or protocol buffer.

## Pipeline

Stage 1 processes 16 bytes at a time on native and Wasm:

1. A two-table nibble lookup classifies quote, backslash, structural,
   whitespace, and control bytes in parallel.
2. Backslash parity identifies escaped quotes.
3. A prefix XOR over unescaped quote bits computes string context.
4. Structural bytes inside strings are carved away.
5. The remaining structural positions, string delimiters, and atom starts are
   emitted as a compact position tape.

A 4 KiB sample estimates the tape capacity before scanning. This avoids both
large over-allocation for long strings and repeated growth for dense numeric
or structural input.

Stage 2 consumes the tape:

- the next tape position bounds an atom, avoiding another body scan;
- strings decode directly from UTF-8 bytes and copy raw segments in batches;
- escaped surrogate code units delegate to core/json so lone-surrogate
  semantics remain compatible;
- numbers validate grammar and accumulate mantissa/exponent in one pass;
- arrays and objects advance by indexed structural positions.

This is why the improvement is not attributable to SIMD alone. The parser
contract and the SIMD classifier are designed together.

Background and a generalized explanation of the pipeline are available in the
[Japanese/English design notes](https://gist.github.com/mizchi/1ba06e646dbc6396a50798a2f9678d15).

## Backend strategy

| target | stage-1 strategy |
|---|---|
| native | core `V128` byte classification |
| wasm | `V128`, with scalar fallback for punctuation-dense samples |
| wasm-gc | scalar indexed fallback |
| js | scalar indexed fallback |

Wasm samples the first 4 KiB. If token starts exceed 25% of sampled bytes, it
uses the scalar indexer because mask extraction ceases to be sparse.

`parse_string` also samples the first 4 Ki UTF-16 code units. Inputs below 64
KiB or with fewer than 5% token starts use core/json; dense large documents use
UTF-8 encoding followed by the pipeline above. A non-default depth limit always
uses the indexed parser so the local `ParseError` contract remains exact.

## Synthetic large-input benchmark

Apple Silicon, release build, MoonBit 2026-07-15. Times are means and should be
treated as directional rather than universal.

### Native

| dataset | core/json | `parse(Bytes)` | `parse_string` |
|---|---:|---:|---:|
| mixed objects, 9.9 MB | 55.20 ms | **50.32 ms** | **52.47 ms** |
| long plain strings, 8.4 MB | 7.79 ms | **7.36 ms** | 7.95 ms (core route) |
| escaped strings, 12.3 MB | **31.41 ms** | 32.10 ms | 31.81 ms (core route) |
| numbers, 7.6 MB | 22.36 ms | **14.56 ms** | **15.49 ms** |

### Wasm

| dataset | core/json | `parse(Bytes)` | `parse_string` |
|---|---:|---:|---:|
| mixed objects, 9.9 MB | 149.23 ms | **119.69 ms** | **126.24 ms** |
| long plain strings, 8.4 MB | 18.68 ms | **16.53 ms** | 18.34 ms (core route) |
| escaped strings, 12.3 MB | 80.82 ms | **72.33 ms** | 80.82 ms (core route) |
| numbers, 7.6 MB | 63.73 ms | **41.00 ms** | **46.19 ms** |

Run the same comparison with:

```bash
moon bench json --target native --release
moon bench json --target wasm --release
```

## Experimental limitations

- `ParseError` intentionally reports only `InvalidSyntax` or
  `DepthLimitExceeded`; it does not yet reproduce core/json line/column
  diagnostics.
- `parse(Bytes)` can still be slower than core/json on escaped-string-heavy
  native input; `parse_string` avoids that case with adaptive routing.
- The adaptive density threshold is based on the included synthetic corpora
  and still needs validation against real-world JSON datasets.
- The token tape costs additional memory proportional to the number of token
  starts.
