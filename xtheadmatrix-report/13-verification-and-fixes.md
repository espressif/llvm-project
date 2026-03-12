# Verification and Bug Fixes

## Summary

Seven independent verification rounds were completed against the RVM 0.6 spec.
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
| 11 | HIGH | `SpecAPIMatmulWiden` passed `{acc, acc, acc}` for 8 widening FP matmul builtins (FP8/BF16/TF32), causing THRVMACC vs THRVMTR register class conflict | Changed builtins to 6-arg `(acc, a, b, m, k, n)`, removed `SpecAPIMatmulWiden`, unified through `SpecAPIMatmul` |

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

## Verification Round 5 (Claude Opus 4.6 #4)

Fixed `SpecAPIMatmulWiden` bug (bug #11). Updated builtin prototypes,
header macro, and codegen dispatch. Added 12 new test cases across 3
test files. Documentation and report files updated for alignment.

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

## Design Notes (Verification Round 6 — Claude Opus 4.6)

### 1. Element-wise tied constraint is unnecessary but harmless

Element-wise operations (`madd`, `msub`, `mmul`, etc.) have a `$src1 = $dst`
tied constraint in their PTH_ pseudo definitions, identical to the matmul
accumulate pattern. However, the hardware computes `md = ms2 op ms1` **without
reading the old md value** — these are pure binary operations, not
accumulate-and-operate. The spec explicitly says e.g. "madd performs the
addition of src1 and src2" with no mention of md on the RHS.

**Impact**: The tied constraint forces the register allocator to place the
output in the same physical register as the `acc` input. If the `acc` value
is still live after the element-wise op, the RA must spill it. Without the
tied constraint, the RA could freely choose any THRVMACC register for the
output, avoiding the spill.

**Status**: Not a correctness bug — the tied constraint is a simplification
for the ManagedRA model (all element-wise ops share the `THMI_MulAcc` ISel
category). Removing it would require a separate ISel category
(`THMI_BinaryAcc`) with `(outs THRVMACC:$dst), (ins THRVMACC:$ms2, THRVMACC:$ms1)`
and no tied constraint.

### 2. INT4 (Zmint4) extension not implemented

The spec defines `mmacc.w.q`, `mmaccu.w.q`, `mmaccus.w.q`, `mmaccsu.w.q`
(INT4→INT32 matmul) and `mscvtl/h.w.b.q`, `mucvtl/h.w.b.q` (INT4↔INT8
conversion) under the optional Zmint4 extension. These are not implemented.

**Status**: Acceptable — Zmint4 is marked optional in xmisa bit 0 (mmi4i32).

### 3. `.mv.x` (GPR-indexed vector) element-wise variants not implemented

The spec intrinsic API doc (`rvm-intrinsic-api.adoc`) mentions `.mv.x`
variants for element-wise operations (e.g. `madd.w.mv.x md, ms2, ms1[rs1]`),
but these do **not** appear in the actual instruction list
(`instruction_list.adoc`). The implementation correctly omits them.

### 4. Element-wise `msub` operand order may be unintuitive

The Spec-API exposes `__riscv_th_msub_w_mm(acc, s2, s1)` which maps directly
to the hardware `th.msub.w.mm md, ms2, ms1`. Per the spec, msub computes
`md[i,j] = ms1[i,j] - ms2[i,j]` ("subtraction of src2 from src1"). So the
**subtrahend** (`s2`) appears before the **minuend** (`s1`) in the API, which
matches hardware convention but may be unintuitive for users expecting
`result = first_arg - second_arg` order.

**Status**: Correct by design — the Spec-API directly mirrors hardware operand
order. Users should refer to parameter names (`__s2`, `__s1`) for clarity.

## Verification Round 6 (Claude Opus 4.6 #5) — x2 Type Support

Reviewed all uncommitted changes implementing proper x2 (register-pair) type
support. Verified 6 files changed, 222 new lines, 53 removed.

**Components verified:**

1. `RVM_X2_TYPE` macro in `RISCVMatrixTypes.def`: backward-compatible `#ifndef`
   guard, correct `#undef` order, all 27 existing consumers unaffected.
2. `CodeGenTypes.cpp`: two-pass `#define`/`#include` correctly separates single
   types → `target("riscv.matrix")` from x2 types → `{ target("riscv.matrix"),
   target("riscv.matrix") }` struct.
3. 22 `mget`/`mset` builtins: prototypes match (x2→single, x2+single→x2).
   Codegen uses extractvalue/insertvalue with select — standard approach.
4. 7 x2 matmul wrappers: FP16 h (x2 B), FP64 d (x2 dest), FP64 d.s (x2 dest),
   INT16→INT64 ss/uu/us/su (x2 dest). All extract component 0, call
   single-register builtin, insert back. Sign variants correct (us/su acc is
   signed `mint64_t`; uu acc is unsigned `muint64_t`).
5. Matmul operand order: correctly delegated to underlying builtins which
   handle the B/A swap per spec formula `md = md + ms1 × ms2`.
6. Tests: 15 cases in `xtheadmatrix-x2-types.c` (O0 IR generation + O2
   optimization folding) + 3 cases in `xtheadmatrix-spec-api.c`.

**Result: No issues found.** All implementations are correct and consistent.

## Verification Round 7 (Claude Opus 4.6 #6) — Zmpanel Extension

Implemented and verified the Zmpanel panel-aware 2x2 matrix tiling extension.

**Components verified:**

1. Feature definition: `FeatureVendorXTHeadZmpanel` correctly implies `FeatureVendorXTHeadMatrix`
2. 30 instruction encodings: all use func3=010 to distinguish from standard RVM (func3=000)
3. Decoder namespace: separate `XTHeadZmpanel` decoder table added to disassembler
4. Panel compute encoding: mirrors standard matmul format (uop=10, confirmed) with ms2/ms1/md=000; ssize field (bits[19:18]) still distinguishes FP source types
5. Implicit Defs/Uses added to panel load/store/compute instructions to prevent incorrect reordering by the compiler
6. `THMI_PanelFireForget` ISel dispatch category introduced for panel load/store/compute (replacing `THMI_NoArgs` which is reserved for `mrelease`); config instructions remain on `THMI_CfgReg`
7. Mixed-mode conflict detection: a fatal error is emitted at ISel time if a function uses both ManagedRA (base XTHeadMatrix) and Zmpanel fire-and-forget instructions, since the two models have incompatible assumptions about matrix register ownership
8. `UsesZmpanelFireAndForget` flag added to `RISCVMachineFunctionInfo` to track panel usage for the conflict check
9. Header API (`<thead_matrix.h>`) Zmpanel section guarded with `#if defined(__riscv_xtheadzmpanel)` feature guard
10. Assembly round-trip verified: encode -> binary -> disassemble produces original instructions
11. Full-stack test: builtins -> intrinsics -> ISel -> instruction emission verified for all 30 operations
12. No regressions: all 15 existing XTHeadMatrix tests pass unchanged

**Key design decisions:**
- No pseudo instructions needed -- Zmpanel instructions operate on implicit hardware state, not compiler-managed matrix register values
- Config instructions use THMI_CfgReg (same category as existing msettilem/k/n)
- Load/store/compute use THMI_PanelFireForget (dedicated category with implicit Defs/Uses)
- `UsesZmpanelFireAndForget` flag in MFI enables mixed-mode conflict detection against ManagedRA

**Result: All 20 tests pass (5 new Zmpanel + 15 existing XTHeadMatrix). Zero regressions.**

## Differences from Spec Intrinsic API (rvm-intrinsic-api.adoc v0.2)

1. **No C++ overloading**: Separate functions per type (C limitation)
2. **Role-specific loads**: `mld_a_*` / `mld_b_*` / `mld_acc_*` instead of unified `mld`
3. **Stream load/store**: Not in RVM 0.6 instruction set
4. **Matrix-scalar EW (`.mx`)**: Not in RVM 0.6 instruction set
5. **64-bit INT EW (`.d`)**: Not in RVM 0.6 instruction set
