# File Change Summary

## Key Files (current state)

### LLVM Backend

| File | Purpose |
|------|---------|
| `llvm/lib/Target/RISCV/RISCVInstrInfoXTHeadMatrix.td` | 227 instruction definitions + ~220 PTH_*_V pseudos + spill/reload pseudos + 30 Zmpanel panel-aware instructions |
| `llvm/include/llvm/IR/IntrinsicsRISCVXTHeadMatrix.td` | ~220 `_internal` intrinsics + 7 config intrinsics + helper classes + 30 Zmpanel intrinsics |
| `llvm/lib/Target/RISCV/RISCVISelDAGToDAG.cpp` | `selectTHMatrixInternal()` with ~257 table entries, 16 dispatch categories |
| `llvm/lib/Target/RISCV/RISCVISelDAGToDAG.h` | `selectTHMatrixInternal()` declaration |
| `llvm/lib/Target/RISCV/RISCVExpandPseudoInsts.cpp` | `lookupTHMatrixPseudo()` table — ~220 PTH_*_V → TH_* expansion |
| `llvm/lib/Target/RISCV/RISCVLowerMatrixType.cpp` | `-O0` support pass for `target("riscv.matrix")` |
| `llvm/lib/Target/RISCV/RISCVMachineFunctionInfo.h` | `MatrixProgModelEnum {None, ManagedRA}` + `UsesZmpanelFireAndForget` flag for mixed-mode conflict detection |
| `llvm/lib/Target/RISCV/RISCVRegisterInfo.cpp` | Conditional matrix register reservation |
| `llvm/lib/Target/RISCV/RISCVRegisterInfo.td` | THRVMMR, THRVMTR, THRVMACC register classes |
| `llvm/lib/Target/RISCV/RISCVFeatures.td` | `FeatureVendorXTHeadMatrix` (experimental, v0.6) + `FeatureVendorXTHeadZmpanel` (implies XTHeadMatrix) |
| `llvm/lib/Target/RISCV/RISCVSystemOperands.td` | 13 CSRs with `th.` prefix |
| `llvm/lib/Target/RISCV/Disassembler/RISCVDisassembler.cpp` | Matrix register decode functions + `XTHeadZmpanel` decoder table entry |
| `llvm/include/llvm/CodeGen/ValueTypes.td` | `MVT::riscvmatrix` (8192-bit) |
| `llvm/lib/CodeGen/ValueTypes.cpp` | `riscv.matrix` ↔ `MVT` mapping |

### Clang Frontend

| File | Purpose |
|------|---------|
| `clang/include/clang/Basic/BuiltinsRISCVXTHeadMatrix.td` | ~272 Clang builtins (Spec-API + mget/mset + config + mundef) + 30 Zmpanel builtins |
| `clang/lib/CodeGen/TargetBuiltins/RISCV.cpp` | Spec-API codegen with lambda helpers + Zmpanel builtin codegen |
| `clang/lib/Headers/thead_matrix.h` | 450+ C API functions/macros (including 30+ Zmpanel wrappers) |
| `clang/include/clang/Basic/RISCVMatrixTypes.def` | 22 native built-in type definitions |

### Tests

| File | Purpose |
|------|---------|
| `clang/test/CodeGen/RISCV/xtheadmatrix-spec-api.c` | 23 Spec-API test cases |
| `clang/test/CodeGen/RISCV/xtheadmatrix-spec-api-full.c` | Extended Spec-API coverage |
| `clang/test/CodeGen/RISCV/xtheadmatrix-api-coverage.c` | Full API function coverage |
| `clang/test/CodeGen/RISCV/xtheadmatrix-verification-fixes.c` | Verification and bug fix tests |
| `clang/test/CodeGen/RISCV/xtheadmatrix-x2-types.c` | 15 x2 type test cases (O0+O2) |
| `clang/test/CodeGen/RISCV/xtheadmatrix-spec-api-example.c` | End-to-end widening matmul example |
| `clang/test/CodeGen/RISCV/xtheadmatrix-inline-asm.c` | Inline asm register constraint test |
| `clang/test/CodeGen/RISCV/xtheadmatrix-zmpanel-api.c` | Zmpanel C header API test |
| `clang/test/CodeGen/RISCV/thead-matrix-builtin-types.c` | Built-in type compilation test |
| `clang/test/CodeGen/RISCV/thead-matrix-types-extended.c` | Extended type test |
| `llvm/test/CodeGen/RISCV/xtheadmatrix-managed-ra.ll` | ManagedRA ISel test |
| `llvm/test/CodeGen/RISCV/xtheadmatrix-managed-ra-full.ll` | Comprehensive ManagedRA test |
| `llvm/test/CodeGen/RISCV/xtheadmatrix-managed-ra-spill.ll` | Spill/reload test |
| `llvm/test/CodeGen/RISCV/xtheadmatrix-managed-ra-regclass.ll` | Register class constraint test |
| `llvm/test/CodeGen/RISCV/xtheadmatrix-managed-ra-misc.ll` | Misc ManagedRA operations |
| `llvm/test/CodeGen/RISCV/xtheadmatrix-lower-O0.ll` | -O0 lowering test |
| `llvm/test/MC/RISCV/xtheadmatrix-valid.s` | 227 instruction encoding tests |
| `llvm/test/MC/RISCV/xtheadmatrix-invalid.s` | Invalid operand tests |
| `llvm/test/MC/RISCV/xtheadmatrix-csr.s` | 13 CSR resolution tests |
| `llvm/test/MC/RISCV/xtheadzmpanel-valid.s` | 30 Zmpanel instruction encoding tests |
| `llvm/test/CodeGen/RISCV/xtheadzmpanel-intrinsics.ll` | Zmpanel intrinsic codegen tests |
| `clang/test/CodeGen/RISCV/xtheadzmpanel-builtins.c` | Zmpanel builtin codegen tests |
| `clang/test/CodeGen/RISCV/xtheadzmpanel-inline-asm.c` | Zmpanel inline asm tests |
| `clang/test/CodeGen/RISCV/xtheadzmpanel-header-api.c` | Zmpanel header API tests |

## Change History

### Initial implementation
- 227 instructions, intrinsics, builtins, ISel, tests, docs
- 22 native built-in types (`__rvm_*_t`)
- `<thead_matrix.h>` header with typed API

### Verification round 1 (Gemini)
- Fixed 42 conversion pseudo register classes (THRVMMR → THRVMACC)
- Fixed matmul operand swap in Spec-API codegen
- Added B-tile loads, FP/unsigned variants, all matmul variants to Spec-API

### Verification round 2 (Claude Opus 4.6)
- Full re-verification of all 227 encodings — no new bugs

### Verification round 3 (Claude Opus 4.6)
- Fixed forward declaration of `lookupTHMatrixPseudo()`
- Fixed 32 name collisions in `thead_matrix.h`
- Fixed config intrinsics forcing DirectReg mode (critical RA fix)

### DirectReg removal + Spec-API completion
- Deleted ~3000 lines: DirectReg ISel, intrinsics, builtins, codegen, Sema, 8 test files
- Added ~1300 lines: ~130 Spec-API builtins, codegen, C wrappers for EW/conversions/data movement
- Config intrinsics moved to ManagedRA ISel table
- `MatrixProgModelEnum` simplified to `{None, ManagedRA}`
- Fixed: FP EW .mv.i signatures, immediate type legalization, macro parameters
- Renamed A-tile loads `__riscv_th_mld_*` → `__riscv_th_mld_a_*`, B-tile `__riscv_th_mldb_*` → `__riscv_th_mld_b_*`

### Widening FP matmul fix
- Fixed `SpecAPIMatmulWiden` bug: 8 widening FP matmul builtins passed `{acc, acc, acc}` instead of `{acc, b, a}`
- Changed builtin prototypes from 4-arg `(acc, m, k, n)` to 6-arg `(acc, a, b, m, k, n)`
- Updated `__THEAD_SPEC_FMMAQA_WIDEN` macro to accept `mint32_t` A/B tile args
- Removed `SpecAPIMatmulWiden` lambda, all matmul dispatch unified through `SpecAPIMatmul`
- Added 8 widening matmul tests, 3 ISel tests, 1 end-to-end example test

### x2 (register-pair) type support
- Added `RVM_X2_TYPE` macro to `RISCVMatrixTypes.def` (backward-compatible default)
- `CodeGenTypes.cpp`: x2 types now lower to `{ target("riscv.matrix"), target("riscv.matrix") }` struct
- Added 22 `mget`/`mset` builtins (`mget_spec_*` / `mset_spec_*`, 11 types each)
- `mget` codegen: extractvalue+select; `mset` codegen: insertvalue+select
- Replaced mundef stubs in `thead_matrix.h` with real builtin calls
- Added 7 x2 matmul wrapper functions:
  - `__THEAD_SPEC_MMACC_X2` macro: 4 INT16→INT64 x2 dest variants (signed/uu/us/su)
  - `__riscv_th_mfmacc_h_x2`: FP16 x2 B operand
  - `__riscv_th_mfmacc_d_x2`: FP64 x2 dest
  - `__riscv_th_mfmacc_d_s_x2`: FP64 widening x2 dest
- New `xtheadmatrix-x2-types.c` test (15 test cases at O0+O2)
- 3 new tests in `xtheadmatrix-spec-api.c` (mget/mset, FP16 x2, FP64 x2)

### Zmpanel extension (2026-03-12)
- Added `FeatureVendorXTHeadZmpanel` experimental extension (implies XTHeadMatrix)
- 30 new instructions: 12 config, 2 load, 2 store, 14 compute (10 FP + 4 INT)
- All use func3=010 (panel macro), separate decoder namespace `XTHeadZmpanel`
- No pseudos needed — fire-and-forget instructions map directly to real opcodes
- 30 intrinsics, 30 builtins, 30+ C API wrapper functions
- 5 new test files (assembly, intrinsics, builtins, inline asm, header API)

### Golden spec cross-reference audit (2026-03-19)
- Updated `xtheadmatrix-doc/RISCVXTHeadMatrix.md`: expanded "Differences from Spec
  Intrinsic API" section with full naming mapping table, signature differences
  (matmul dim order, EW CSR behavior, mzero params, CSR enum), and not-implemented
  features list. Added verification history entry #7
- Updated `xtheadmatrix-report/13-verification-and-fixes.md`: added Verification
  Round 9 findings, expanded "Differences from Spec Intrinsic API" with complete
  missing-names table, structural differences, behavioral notes, and spec internal
  inconsistencies
- Updated `xtheadmatrix-report/00-overview.md`: added verification round 8+9 to
  history, added limitations 9-11 (API naming gaps, EW CSR dependency, missing features)
- Updated `xtheadmatrix-report/12-intrinsic-api.md`: added behavioral notes (EW CSR
  state, matmul dim order), differences from spec section, updated limitations
- No code changes — documentation only

### Full independent re-verification (2026-03-20)
- Five parallel verification agents checked: instruction encodings (40+), register
  allocation constraints, Zmpanel (30 instr + 18 CSRs), stream load/store omission,
  and C API naming against spec
- **No new bugs found** across all verification areas
- Updated all report files and doc with round 10 findings
- Fixed: test file listings (added 5 previously unlisted test files), Zmpanel CSR
  count (20→18), `12-intrinsic-api.md` overview (clarified RVM 0.6 mnemonic basis),
  doc x2 example comment, doc Example 7 (replaced DirectReg builtins with ManagedRA),
  doc spec errata count (3→4)
- No code changes — documentation only
