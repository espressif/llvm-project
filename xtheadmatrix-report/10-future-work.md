# Future Work and Limitations

## Deferred: 64-bit Instruction Support

The spec defines 64-bit instruction formats (`inst64_format.adoc`), but only
32-bit is implemented. All 227 defined instructions use 32-bit encoding.

To implement:
1. Add `getInstruction64()` to `RISCVDisassembler`
2. Define `THRVMInst64` base class
3. Add 64-bit decoder list and dispatch
4. Define 64-bit instruction variants

## Implemented: Inline Asm Register Constraints

Matrix registers now support typed inline asm constraints:
- `tr` (any matrix register), `tt` (tile only), `ta` (accumulator only)
- Named register constraints: `={tr0}`, `={acc1}`, etc.
- Clobber constraints: `"tr0"`, `"acc0"`, etc.
- `copyPhysReg` via `PTH_MMOV_MM_V` pseudo for register-to-register copies

## Implemented: Zmpanel Panel-Aware 2x2 Tiling

The Zmpanel extension has been fully implemented with 30 fire-and-forget macro
instructions for panel-aware 2x2 matrix tiling. Feature
`FeatureVendorXTHeadZmpanel` implies `FeatureVendorXTHeadMatrix`. These operate
on implicit hardware state and do not require pseudo instructions or
compiler-managed matrix register values.

### Zmpanel Future Work

- Higher-level tiling abstractions for the panel GEMM pipeline (auto-generating
  the config+load+compute+store sequence from loop-nest descriptions)
- Integration with MLIR/linalg to automatically lower tiled matmul to Zmpanel
  config+ml22+compute+msc22 sequences
- Auto-selection of Zmpanel vs base XTHeadMatrix paths based on tile dimensions
- OOB policy optimization (compiler-driven selection of zero-pad vs skip modes)

## Possible Improvements

- Auto-select matmul variants based on data type
- MLIR linalg integration for automatic matrix tiling
- C++ overloaded wrapper layer
- **Remove unnecessary tied constraint on element-wise pseudos**: EW ops
  (`madd`, `msub`, `mmul`, etc.) are pure binary `md = ms2 op ms1` — they do
  not read the old md value. The current `$src1 = $dst` tied constraint forces
  the RA to co-locate the output with the `acc` input, causing unnecessary
  spills when `acc` is still live. A new `THMI_BinaryAcc` ISel category with
  untied `(outs THRVMACC:$dst), (ins THRVMACC:$ms2, THRVMACC:$ms1)` would
  give the RA more allocation freedom.

## Current Limitations

1. **No 64-bit instruction format**
2. **Matrix types cannot cross function boundaries** (no ABI support)
3. **No auto-matmul from C loops** (explicit builtins required)
4. **Limited register file (4+4)** — high pressure for multi-tile kernels
5. **Whole-register spill granularity** (8192-bit per spill)
6. **`-O0` support limited** (`RISCVLowerMatrixType` pass; `-O1`+ recommended)
7. **No `.mx`/`.d` integer EW** (spec aspirational, no RVM 0.6 instructions)
8. **No stream load/store** (`msld`/`msst` not in RVM 0.6)
9. **No INT4 (Zmint4)** — optional `mmacc.w.q` variants not implemented
10. **No `.mv.x` element-wise variants** — not in RVM 0.6 instruction list
    (only in the intrinsic API design doc)

## Encoding Verification

All 257 instruction encodings verified (227 base + 30 Zmpanel; 3 independent audits including a full field-by-field verification of all 257 instructions, 0 implementation discrepancies).
Known spec errata: matmul uop=01 should be 10; mfmin.s/mfmin.h names swapped; pmmaccus.w.b typo.
