# Changelog

All notable changes to the AICS LLVM Toolchain (XTHeadMatrix / RVM 0.6) are documented here.

## [aics_llvm_toolchain_v0.2] - 2026-03-13

### Added

- **XTHeadZmpanel extension (xtheadzmpanel v0.6)**: Full LLVM/Clang support for panel-aware 2x2 matrix tiling.
  - 30 fire-and-forget macro instructions for efficient 2x2 panel GEMM pipelines.
  - 12 config instructions (address/stride/dimension), 2 panel loads, 2 panel stores, 10 FP compute, 4 INT compute variants.
  - Panel load/store/compute instructions operate on implicit hardware state with no explicit matrix register operands.
  - New `THMI_PanelFireForget` ISel dispatch category.
  - Implicit Defs/Uses on matrix registers to prevent reordering.
  - Mixed-mode conflict detection (ManagedRA vs Zmpanel fire-and-forget).
  - `XTHeadZmpanel` decoder namespace for disassembler.
  - 30 LLVM intrinsics, 30 Clang builtins, C header API wrappers.
  - Tests: MC encoding/disassembly, intrinsics, builtins, header API, inline assembly.

- **x2 matrix types and builtins for XTHeadMatrix**: Software-level x2 type support for matmul variants requiring paired matrix operands.
  - New x2 types: `mfloat16x2_t`, `mint64x2_t`, etc., mapped to `{ target("riscv.matrix"), target("riscv.matrix") }` struct at IR level.
  - `mget`/`mset` builtins for x2 component access via extractvalue/insertvalue.
  - x2 matmul variants: FP16 `mfmacc.h` (x2 B), FP64 `mfmacc.d` (x2 dest), FP64 `mfmacc.d.s` (x2 dest), INT16->INT64 all 4 sign variants (x2 dest).
  - Spec-API wrappers extract component 0 for the hardware instruction.

## [aics_llvm_toolchain_v0.1] - 2026-03-06

Initial tagged release of AICS LLVM Toolchain with XTHeadMatrix (RVM 0.6) support.
