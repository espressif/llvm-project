# ISel / CodeGen Implementation

**Verification status**: All 267 ISel table entries verified correct. Complete
lowering chain (builtin -> intrinsic -> DAG -> MachineInstr) confirmed across
13 verification rounds. Operand ordering, type handling, register class constraints,
tied operand patterns, and CSR configuration emission all verified correct.

## Overview

ISel uses a table-driven approach via `selectTHMatrixInternal()` with ~267
entries mapping `_internal` intrinsics (plus config + panel intrinsics) to PTH_*_V
pseudo-instructions (or direct hardware instructions for Zmpanel). Post-RA,
`RISCVExpandPseudoInsts` expands each pseudo to the corresponding hardware
`TH_*` instruction.

## ISel Dispatch Categories

```
THMI_Load        (ptr, stride) → matrix
THMI_LoadWhole   (ptr) → matrix
THMI_Store       (matrix, ptr, stride) → void
THMI_StoreWhole  (matrix, ptr) → void
THMI_Zero        () → matrix
THMI_MulAcc      (acc, ms2, ms1) → acc        [tied $src1 = $dst]
THMI_MulAccImm   (acc, ms2, ms1, imm) → acc   [tied $src1 = $dst]
THMI_Unary       (ms1) → md
THMI_Binary      (ms2, ms1) → md
THMI_UnaryImm    (ms1, imm) → md
THMI_ToGPR       (matrix, idx) → gpr
THMI_FromGPR2    (matrix, data, idx) → matrix  [tied $md_in = $dst]
THMI_FromGPR     (matrix, data) → matrix       [tied $md_in = $dst]
THMI_CfgImm      (imm) → void
THMI_CfgReg      (reg) → void
THMI_NoArgs       () → void
```

## Register Class Constraints on Pseudos

| Operation | Pseudo register class | Spec constraint |
|-----------|----------------------|-----------------|
| Load A/B (mlae/mlbe) | THRVMTR | Tile (0-3) |
| Load C (mlce) | THRVMACC | Acc (4-7) |
| Load M (mlme) | THRVMMR | Any (0-7) |
| Store A/B (msae/msbe) | THRVMTR | Tile (0-3) |
| Store C (msce) | THRVMACC | Acc (4-7) |
| Store M (msme) | THRVMMR | Any (0-7) |
| Matmul (acc) | THRVMACC, tied | Acc (4-7) |
| Matmul (operands) | THRVMTR | Tile (0-3) |
| EW INT/FP .mm/.mv.i | THRVMACC, all, tied | Acc (4-7) |
| N4clip .mm/.mv.i | THRVMACC, all, tied | Acc (4-7) |
| Conversions | THRVMACC | Acc (4-7) |
| Pack | THRVMMR | Any (0-7) |
| Slide/broadcast | THRVMMR | Any (0-7) |
| Move/dup | THRVMMR | Any (0-7) |
| Zero | THRVMMR | Any (0-7) |

## Programming Model Tracking

`MatrixProgModelEnum` in `RISCVMachineFunctionInfo.h`:
- `None` — no matrix intrinsics used; all 8 matrix registers reserved
- `ManagedRA` — `_internal` intrinsics used; registers managed by RA

Set to `ManagedRA` when any non-config `_internal` intrinsic is selected.
Config intrinsics (`msettilem/k/n`, `mrelease`) do not change the model.

## Pseudo Expansion

`RISCVExpandPseudoInsts.cpp` contains a `lookupTHMatrixPseudo()` table
mapping ~220 PTH_*_V pseudos to TH_* hardware instructions:

```cpp
struct THMatrixPseudoEntry {
  unsigned PseudoOpc;
  unsigned RealOpc;
  bool SkipTiedInput;  // skip first input if tied ($src1 = $dst)
};
```

`SkipTiedInput` is set for MulAcc/MulAccImm/FromGPR/FromGPR2 pseudos
where the tied `$src1` operand duplicates `$dst` and must not be emitted
as an explicit operand in the hardware instruction.

## Spill / Reload

Matrix register spills use the RVV-style pattern (no immediate offset):
- Spill: `PTH_MSME_E8_SPILL` → `TH_MSME_E8` (whole-register store)
- Reload: `PTH_MLME_E8_RELOAD` → `TH_MLME_E8` (whole-register load)
- Registered in `isRVVSpill()` for correct stack frame handling
- `eliminateFrameIndex()` materializes the stack address into a GPR

## Clang Codegen (`RISCV.cpp`)

Spec-API builtin codegen uses lambda helpers:

| Helper | Emits | Used for |
|--------|-------|----------|
| `SpecAPILoadA(EEW)` | SetM+SetK → mlae_internal | A-tile loads |
| `SpecAPILoadB(EEW)` | SetK+SetN → mlbe_internal | B-tile loads |
| `SpecAPILoadAcc(EEW)` | SetM+SetN → mlce_internal | Acc loads |
| `SpecAPIStore(EEW)` | SetM+SetN → msce_internal | Stores |
| `SpecAPIMatmul(ID)` | SetM+SetK+SetN, swap A/B | Matmul |
| `SpecAPIZero()` | SetM+SetN → mzero_internal | Zero |
| `SpecAPIUnary(ID)` | Direct call | Conversions, move |
| `SpecAPIBinary(ID)` | Direct call | Pack |
| `SpecAPIMulAcc(ID)` | 3 matrix args | EW .mm, n4clip .mm |
| `SpecAPIMulAccImm(ID)` | 3 matrix + ZExt(imm) | EW .mv.i, n4clip .mv.i |
| `SpecAPIUnaryImm(ID)` | matrix + ZExt(imm) | Slide, broadcast |
| `SpecAPIToGPR(ID)` | Returns GPR | mmov_x_m |
| `SpecAPIFromGPR2(ID)` | Tied matrix + data + idx | mmov_m_x |
| `SpecAPIFromGPR(ID)` | Tied matrix + data | mdup_m_x |

Note: Matmul operand swap — spec formula is `md = md + ms1 × ms2`.
A maps to ms1, B maps to ms2. Codegen passes `{acc, B, A}` (not `{acc, A, B}`).

Note: `SpecAPIMatmulWiden` was removed (bug fix). It previously handled
8 widening FP matmul builtins (FP8/BF16/TF32) by passing `{acc, acc, acc}`
as all three intrinsic operands, causing register class conflicts. All
matmul dispatch (including widening) now goes through `SpecAPIMatmul`,
which correctly passes `{acc, B, A}` with the operand swap.
