# Future Work and Limitations

## Deferred: 64-bit Instruction Support

The spec defines 64-bit instruction formats (`inst64_format.adoc`), but only
32-bit is implemented. All 227 defined instructions use 32-bit encoding.

To implement:
1. Add `getInstruction64()` to `RISCVDisassembler`
2. Define `THRVMInst64` base class
3. Add 64-bit decoder list and dispatch
4. Define 64-bit instruction variants

## Deferred: Inline Asm Register Constraints

Matrix registers (`tr0-tr3`, `acc0-acc3`) cannot be used with typed inline
asm constraints (e.g., `"=vr"` style). Missing:
- GCC register names in `getGCCRegNames()`
- Constraint letters in `getRegForInlineAsmConstraint()`
- `ISD::LOAD`/`ISD::STORE` patterns for `MVT::riscvmatrix`

Pure string-template inline asm with hardcoded register names works but
cannot interact safely with the register allocator.

## Possible Improvements

- Tiling and loop-nest abstractions for common GEMM patterns
- Auto-select matmul variants based on data type
- MLIR linalg integration for automatic matrix tiling
- C++ overloaded wrapper layer

## Current Limitations

1. **No 64-bit instruction format**
2. **Matrix types cannot cross function boundaries** (no ABI support)
3. **No auto-matmul from C loops** (explicit builtins required)
4. **Limited register file (4+4)** — high pressure for multi-tile kernels
5. **Whole-register spill granularity** (8192-bit per spill)
6. **No inline asm register constraints** for matrix registers
7. **`-O0` support limited** (`RISCVLowerMatrixType` pass; `-O1`+ recommended)
8. **No `.mx`/`.d` integer EW** (spec aspirational, no RVM 0.6 instructions)
9. **No stream load/store** (`msld`/`msst` not in RVM 0.6)

## Encoding Verification

All 227 instruction encodings verified (2 independent audits, 0 discrepancies).
Known spec errata: matmul uop=01 should be 10; mfmin.s/mfmin.h names swapped.
