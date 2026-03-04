# Verification and Bug Fixes

## Summary

Four independent verification rounds were completed against the RVM 0.6 spec.
All bugs have been fixed. The implementation is verified correct.

## Bug Fixes (all resolved)

| # | Severity | Issue | Fix |
|---|----------|-------|-----|
| 1 | HIGH | 42 conversion pseudos used THRVMMR instead of THRVMACC | Changed to THRVMACC per spec |
| 2 | HIGH | Matmul operand swap: passed `{acc, A, B}` instead of `{acc, B, A}` | Swapped to match `md = md + ms1 × ms2` |
| 3 | MEDIUM | Forward declaration of `lookupTHMatrixPseudo()` missing | Added forward declaration |
| 4 | MEDIUM | 32 name collisions between DirectReg and Spec-API in `thead_matrix.h` | Renamed Spec-API functions |
| 5 | CRITICAL | Config intrinsics forced DirectReg mode, breaking ManagedRA RA entirely | Made config intrinsics mode-neutral |
| 6 | HIGH | FP EW `.mv.i` builtin signatures missing `ms1` operand (3 args, needed 4) | Fixed signatures in builtins + header |
| 7 | MEDIUM | Immediate args (`i32`) caused type legalization crash on rv64 | ZExt to XLen in codegen |
| 8 | MEDIUM | Config intrinsics unhandled after `selectTHMatrix()` deletion | Added to ManagedRA ISel table |
| 9 | LOW | SemaRISCV.cpp referenced deleted DirectReg builtin IDs | Deleted validation section |
| 10 | LOW | Macro parameter mismatches in header | Removed unused parameters |

## Spec Errata

1. `instruction_list.adoc`: matmul `uop` field shows `01`, should be `10`
2. `instruction_list.adoc`: `mfmin.s` and `mfmin.h` names are swapped

## Register Class Verification

All operations verified to use correct register classes:

| Operation | Register class | Spec requirement |
|-----------|---------------|------------------|
| Load A/B | THRVMTR | Tile (0-3) |
| Load C | THRVMACC | Acc (4-7) |
| Matmul (acc) | THRVMACC, tied | Acc (4-7) |
| Matmul (operands) | THRVMTR | Tile (0-3) |
| EW INT/FP | THRVMACC, all, tied | Acc (4-7) |
| N4clip | THRVMACC, all, tied | Acc (4-7) |
| Conversions | THRVMACC | Acc (4-7) |
| Pack/slide/broadcast | THRVMMR | Any (0-7) |
| Move/dup/zero | THRVMMR | Any (0-7) |

## Encoding Verification

227 instruction encodings independently verified twice:
- Audit #1 (Gemini): 24/24 programmatic field checks, 0 conflicts
- Audit #2 (Claude Opus 4.6): every bit field re-verified against spec source

## Spec-API Naming Convention

| Operation | C function | Hardware instruction |
|-----------|-----------|---------------------|
| A-tile load | `__riscv_th_mld_a_*` | `th.mlae*` |
| B-tile load | `__riscv_th_mld_b_*` | `th.mlbe*` |
| Acc load | `__riscv_th_mld_acc_*` | `th.mlce*` |
| Store | `__riscv_th_mst_*` | `th.msce*` |
| INT matmul | `__riscv_th_mmaq_*` | `th.mmacc.*` |
| FP matmul | `__riscv_th_mfmaqa_*` | `th.mfmacc.*` |
| Zero | `__riscv_th_mzeros_*` | `th.mzero` |
| EW int .mm | `__riscv_th_madd_w_mm` | `th.madd.w.mm` |
| EW fp .mm | `__riscv_th_mfadd_s_mm` | `th.mfadd.s.mm` |
| Conversion | `__riscv_th_mfcvtl_s_h` | `th.mfcvtl.s.h` |
| Move | `__riscv_th_mmov_mm` | `th.mmov.mm` |
| Pack | `__riscv_th_mpack` | `th.mpack` |
| Slide | `__riscv_th_mrslidedown` | `th.mrslidedown` |
| Broadcast | `__riscv_th_mrbca` | `th.mrbca.mv.i` |

## Differences from Spec Intrinsic API (rvm-intrinsic-api.adoc v0.2)

1. **No C++ overloading**: Separate functions per type (C limitation)
2. **Role-specific loads**: `mld_a_*` / `mld_b_*` / `mld_acc_*` instead of unified `mld`
3. **Stream load/store**: Not in RVM 0.6 instruction set
4. **Matrix-scalar EW (`.mx`)**: Not in RVM 0.6 instruction set
5. **64-bit INT EW (`.d`)**: Not in RVM 0.6 instruction set
