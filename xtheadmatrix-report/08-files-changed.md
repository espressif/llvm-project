# File Change Summary

## Key Files (current state)

### LLVM Backend

| File | Purpose |
|------|---------|
| `llvm/lib/Target/RISCV/RISCVInstrInfoXTHeadMatrix.td` | 227 instruction definitions + ~220 PTH_*_V pseudos + spill/reload pseudos |
| `llvm/include/llvm/IR/IntrinsicsRISCVXTHeadMatrix.td` | ~220 `_internal` intrinsics + 7 config intrinsics + helper classes |
| `llvm/lib/Target/RISCV/RISCVISelDAGToDAG.cpp` | `selectTHMatrixInternal()` with ~227 table entries, 16 dispatch categories |
| `llvm/lib/Target/RISCV/RISCVISelDAGToDAG.h` | `selectTHMatrixInternal()` declaration |
| `llvm/lib/Target/RISCV/RISCVExpandPseudoInsts.cpp` | `lookupTHMatrixPseudo()` table — ~220 PTH_*_V → TH_* expansion |
| `llvm/lib/Target/RISCV/RISCVLowerMatrixType.cpp` | `-O0` support pass for `target("riscv.matrix")` |
| `llvm/lib/Target/RISCV/RISCVMachineFunctionInfo.h` | `MatrixProgModelEnum {None, ManagedRA}` |
| `llvm/lib/Target/RISCV/RISCVRegisterInfo.cpp` | Conditional matrix register reservation |
| `llvm/lib/Target/RISCV/RISCVRegisterInfo.td` | THRVMMR, THRVMTR, THRVMACC register classes |
| `llvm/lib/Target/RISCV/RISCVFeatures.td` | `FeatureVendorXTHeadMatrix` (experimental, v0.6) |
| `llvm/lib/Target/RISCV/RISCVSystemOperands.td` | 13 CSRs with `th.` prefix |
| `llvm/lib/Target/RISCV/Disassembler/RISCVDisassembler.cpp` | Matrix register decode functions |
| `llvm/include/llvm/CodeGen/ValueTypes.td` | `MVT::riscvmatrix` (8192-bit) |
| `llvm/lib/CodeGen/ValueTypes.cpp` | `riscv.matrix` ↔ `MVT` mapping |

### Clang Frontend

| File | Purpose |
|------|---------|
| `clang/include/clang/Basic/BuiltinsRISCVXTHeadMatrix.td` | ~272 Clang builtins (Spec-API + mget/mset + config + mundef) |
| `clang/lib/CodeGen/TargetBuiltins/RISCV.cpp` | Spec-API codegen with lambda helpers |
| `clang/lib/Headers/thead_matrix.h` | 300+ C API functions/macros |
| `clang/include/clang/Basic/RISCVMatrixTypes.def` | 22 native built-in type definitions |

### Tests

| File | Purpose |
|------|---------|
| `clang/test/CodeGen/RISCV/xtheadmatrix-spec-api.c` | 23 Spec-API test cases |
| `clang/test/CodeGen/RISCV/xtheadmatrix-x2-types.c` | 15 x2 type test cases (O0+O2) |
| `clang/test/CodeGen/RISCV/xtheadmatrix-spec-api-example.c` | End-to-end widening matmul example |
| `clang/test/CodeGen/RISCV/xtheadmatrix-inline-asm.c` | Inline asm register constraint test |
| `clang/test/CodeGen/RISCV/thead-matrix-builtin-types.c` | Built-in type compilation test |
| `clang/test/CodeGen/RISCV/thead-matrix-types-extended.c` | Extended type test |
| `llvm/test/CodeGen/RISCV/xtheadmatrix-managed-ra.ll` | ManagedRA ISel test |
| `llvm/test/CodeGen/RISCV/xtheadmatrix-managed-ra-spill.ll` | Spill/reload test |
| `llvm/test/CodeGen/RISCV/xtheadmatrix-managed-ra-regclass.ll` | Register class constraint test |
| `llvm/test/CodeGen/RISCV/xtheadmatrix-managed-ra-misc.ll` | Misc ManagedRA operations |
| `llvm/test/CodeGen/RISCV/xtheadmatrix-lower-O0.ll` | -O0 lowering test |
| `llvm/test/MC/RISCV/xtheadmatrix-valid.s` | 227 instruction encoding tests (1154 lines) |
| `llvm/test/MC/RISCV/xtheadmatrix-invalid.s` | Invalid operand tests |
| `llvm/test/MC/RISCV/xtheadmatrix-csr.s` | 13 CSR resolution tests |

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
  - `__THEAD_SPEC_MMAQA_X2` macro: 4 INT16→INT64 x2 dest variants (ss/uu/us/su)
  - `__riscv_th_mfmaqa_h_x2`: FP16 x2 B operand
  - `__riscv_th_mfmaqa_d_x2`: FP64 x2 dest
  - `__riscv_th_mfmaqa_d_s_x2`: FP64 widening x2 dest
- New `xtheadmatrix-x2-types.c` test (15 test cases at O0+O2)
- 3 new tests in `xtheadmatrix-spec-api.c` (mget/mset, FP16 x2, FP64 x2)
