# XuanTie RVM 0.6 (RISC-V Matrix Extension) — Implementation Report

## Project Summary

Full assembler/disassembler, LLVM IR intrinsics, Clang builtins, ISel/CodeGen,
and higher-level C API for the XuanTie RVM 0.6 (RISC-V Matrix Extension) in
LLVM, targeting AI/ML workloads. 227 instructions cover matrix multiplication,
load/store, configuration, data movement, and element-wise arithmetic across
data types from FP8 to FP64 and INT8 to INT64.

## Key Facts

| Item | Value |
|------|-------|
| Extension | `xtheadmatrix` (experimental, v0.6) |
| Decoder namespace | `XTHeadMatrix` |
| Major opcode | `OPC_CUSTOM_1` (0b0101011) |
| Instructions | 227 (7 config, 56 load/store, 27 matmul, 35 misc, 102 EW) |
| Registers | 8 matrix: tr0-tr3 (tile) + acc0-acc3 (accumulator), 3-bit encoded |
| CSRs | 13, prefixed `th.` |
| Programming model | Spec-API (ManagedRA) exclusively |
| LLVM intrinsics | ~220 `_internal` + 7 config |
| Clang builtins | ~220 Spec-API + 22 mget/mset + 22 mundef + 7 config ≈ 272 |
| Built-in types | 22 native (`__rvm_int8_t` .. `__rvm_float64x2_t`) |
| C header | `<thead_matrix.h>` with 420+ API functions/macros |
| ISel | `selectTHMatrixInternal()`, 16 dispatch categories, ~227 table entries |
| Pseudo expansion | ~220 PTH_*_V → TH_* post-RA |

## Spec-API Coverage

All 227 hardware operations are covered:

- **Loads**: A-tile `__riscv_th_mld_a_*`, B-tile `__riscv_th_mld_b_*`, accumulator `__riscv_th_mld_acc_*` (11 types each)
- **Stores**: `__riscv_th_mst_*` (11 types)
- **Matmul**: INT `__riscv_th_mmaq_*` (14 + 4 x2), FP `__riscv_th_mfmaqa_*` (13 + 3 x2)
- **Tuple**: `__riscv_th_mget_*` / `__riscv_th_mset_*` (11 types each, extract/insert from x2 pairs)
- **Zero**: `__riscv_th_mzeros_*` (11 types)
- **EW integer**: `__riscv_th_madd_w_mm` etc. (11 ops × .mm/.mv.i)
- **EW FP**: `__riscv_th_mfadd_s_mm` etc. (5 ops × 3 precisions × .mm/.mv.i)
- **Conversions**: format (26), float-int (12), packed (4)
- **N4clip**: 8 variants (signed/unsigned × low/high × .mm/.mv.i)
- **Data movement**: move, dup (4 sizes), pack (3), slide (row + 4 col sizes × up/down), broadcast (row + 4 col sizes)

## Final Test Status

| Test suite | Result |
|------------|--------|
| `xtheadmatrix-spec-api.c` (23 test cases) | PASS |
| `xtheadmatrix-x2-types.c` (15 test cases, O0+O2) | PASS |
| `xtheadmatrix-spec-api-example.c` (e2e widening matmul) | PASS |
| `xtheadmatrix-managed-ra*.ll` (4 tests) | PASS |
| `xtheadmatrix-lower-O0.ll` | PASS |
| `thead-matrix-builtin-types.c` | PASS |
| `thead-matrix-types-extended.c` | PASS |
| `xtheadmatrix-valid.s` (1154 lines) | PASS |
| `xtheadmatrix-invalid.s` | PASS |
| `xtheadmatrix-csr.s` | PASS |
| RISCV MC full suite | 555/555 PASS |
| `xtheadmatrix-inline-asm.c` (inline asm constraints) | PASS |
| End-to-end RA (EW, conversions, data movement, matmul pipeline) | PASS |
| Spill-pressure test (5 ACC values, 4 regs) | PASS |

## Verification History

Six independent verification rounds were completed:

1. **Gemini (2026-03-04)**: Found 2 HIGH bugs (conversion pseudo register classes THRVMMR→THRVMACC; matmul operand swap), filled 3 coverage gaps (B-tile load, FP/unsigned variants, matmul variants).

2. **Claude Opus 4.6 #1**: Full re-verification of all 227 encodings, 220 pseudo expansions, spec-API codegen. No new bugs. Documented limitations and spec differences.

3. **Claude Opus 4.6 #2**: End-to-end compilation + usability audit. Fixed forward declaration, 32 name collisions, type inconsistencies, config intrinsics forcing DirectReg mode (critical RA fix).

4. **Claude Opus 4.6 #3**: DirectReg removal + Spec-API completion. Removed entire DirectReg model (~3000 lines). Added ~130 new Spec-API builtins for all remaining operations. Fixed FP EW .mv.i signatures, immediate type legalization, config intrinsic ISel dispatch. All tests pass.

5. **Claude Opus 4.6 #4**: Fixed HIGH-severity `SpecAPIMatmulWiden` bug — 8 widening FP matmul builtins passed acc as all 3 intrinsic args `{acc, acc, acc}`. Changed to 6-arg `(acc, a, b, m, k, n)`, unified through `SpecAPIMatmul`. Added 11 new tests (8 widening + 3 ISel + 1 e2e example). All 8 xtheadmatrix tests pass.

6. **Claude Opus 4.6 #5**: Implemented proper x2 (register-pair) type support. Added `RVM_X2_TYPE` macro differentiation in `.def` file. x2 types now lower to `{ target("riscv.matrix"), target("riscv.matrix") }` struct IR. Added 22 `mget`/`mset` builtins (replacing mundef stubs) with extractvalue/insertvalue+select codegen. Added 7 x2 matmul wrapper functions (4 INT16→INT64 + 3 FP). Comprehensive `xtheadmatrix-x2-types.c` test file (15 test cases, O0+O2) + 3 new tests in `xtheadmatrix-spec-api.c`.

## Current Limitations

1. No 64-bit instruction format (spec defines but not implemented)
2. Matrix types cannot cross function boundaries (no ABI support)
3. No auto-matmul from C loops (explicit builtins required)
4. Limited register file (4 tile + 4 accumulator)
5. Whole-register spill granularity (8192-bit per spill)
6. `-O0` support limited (RISCVLowerMatrixType pass provides basic support)
7. Known spec errata: matmul uop=01 should be 10; mfmin.s/mfmin.h names swapped
