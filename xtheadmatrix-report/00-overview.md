# XuanTie RVM 0.6 (RISC-V Matrix Extension) - Implementation Report

## Project Summary

Implemented full assembler/disassembler support, LLVM IR intrinsics, Clang builtins, and ISel/CodeGen for the XuanTie RVM 0.6 (RISC-V Matrix Extension) in LLVM, targeting AI/ML workloads. The extension provides 227 instructions for matrix multiplication, load/store, configuration, data movement, and element-wise arithmetic across data types from FP8 to FP64 and INT8 to INT64.

- **Extension name**: `xtheadmatrix` (experimental, version 0.6)
- **Decoder namespace**: `XTHeadMatrix`
- **Major opcode**: `OPC_CUSTOM_1` (0b0101011)
- **Instruction count**: 227 total (7 config, 56 load/store, 27 matmul, 35 misc, 102 element-wise)
- **Register class**: 8 matrix registers: `tr0-tr3` (tile) and `acc0-acc3` (accumulator), 3-bit encoded
- **CSR names**: 13 CSRs prefixed with `th.` (e.g. `th.xmcsr`, `th.mtilem`)
- **Intrinsics**: 371 lines of LLVM IR intrinsics (`int_riscv_th_*`) covering all instruction categories
- **Builtins**: 249+ Clang builtins (227 original + 22 mundef + 90+ spec-API); 121 have typed `__rvm_*_t` signatures (Phase C)
- **ISel/CodeGen**: Table-driven `selectTHMatrix()` with 227 intrinsic-to-instruction mappings, 15 dispatch categories, flexible register selection via ImmArg register index parameters (no longer hardcoded), Sema constraint validation, and all 8 matrix registers reserved. ManagedRA path: `selectTHMatrixInternal()` with 220 `_internal` intrinsic-to-pseudo mappings + post-RA expansion.
- **Intrinsic API**: `<thead_matrix.h>` header with 500+ API functions/macros, 22 matrix types backed by native built-in types, following the RVM Intrinsic API Reference Manual v0.2
- **Spec-API (ManagedRA)**: Full register-allocator-managed programming model with A/B/C-tile loads (11 types each), stores (11 types), all matmul variants (27), and zero constructors (11 types)
- **Built-in types**: 22 native Clang built-in types (`__rvm_int8_t` .. `__rvm_float64x2_t`) via `RISCVMatrixTypes.def`, integrated across ~27 Clang source files
- **Typed builtins (Phase C)**: 121 builtins accept/return `__rvm_*_t` types for Sema-level type checking; CGBuiltin filters TargetExtType args and returns PoisonValue tokens

## Final Status

- Clean build: 0 errors (2 pre-existing unused function warnings in disassembler)
- 227 instruction encodings: all verified against RVM 0.6 spec (24/24 programmatic field checks, 0 conflicts)
- 227 ISel register mappings: intrinsics accept register index arguments (ImmArg) for flexible register selection; Sema validates index constraints per instruction category
- Test files: xtheadmatrix-valid.s (1154 lines), xtheadmatrix-invalid.s (85 lines), xtheadmatrix-csr.s (63 lines)
- CodeGen tests: xtheadmatrix-isel.ll (ISel patterns), xtheadmatrix-codegen.c (end-to-end builtin compilation)
- LLVM IR intrinsics: full coverage (config, load/store, matmul, misc, element-wise)
- Clang builtins: full coverage with CGBuiltin handler (config, load/store, matmul, misc, element-wise)
- Encoding verification: all 5 original discrepancy categories resolved and second-pass verified
- 13 CSR names: all resolve correctly
- All 16 XTHeadMatrix tests pass (11 Clang CodeGen + 1 Sema + 4 LLVM); 557/557 RISCV MC tests pass (no regressions)
- Higher-level API: `<thead_matrix.h>` header compiles and all assembly checks pass
- Phase B built-in types: all 22 types compile, feature-gated, with Sema diagnostic
- Phase C typed builtins: 121 builtins accept/return `__rvm_*_t` types, 22 mundef builtins added, header migrated to native types, all tests pass
- Register index restructure: all intrinsics/builtins accept register index parameters; Sema validates constraints per instruction category
- Bug fix: `mzero` was hardcoded to tr0 (incorrect); now correctly targets acc0 via register index parameter
- Audit fixes: N4clip signed return type corrected (mint8_t), reinterpret casts accept src parameter per spec
- **Independent verification #1 (2026-03-04, Gemini)**: 2 bugs found and fixed, 3 coverage gaps eliminated (see report 13)
  - Bug fix: 42 conversion pseudo-instructions changed THRVMMR → THRVMACC per spec
  - Bug fix: spec-API matmul operand swap (A→ms1, B→ms2) for non-commutative correctness
  - New: B-tile spec-API load (mlbe), FP/unsigned type variants, all matmul variants in spec-API
- **Independent verification #2 (2026-03-04, Claude Opus 4.6)**: Full re-verification against spec source files, **no new bugs found**
  - 227 instruction encodings: every bit field re-verified against spec/instruction_list.adoc
  - 447 ISel entries (227 DirectReg + 220 ManagedRA): all correct
  - 220 pseudo expansion entries: all SkipTiedInput flags verified against .td Constraints
  - Spec-API codegen: matmul swap, load/store selection, CSR calls all confirmed correct
  - 13 CSR addresses verified, 447 intrinsic signatures verified
  - Limitations and differences from spec documented (see report 13)
