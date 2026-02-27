# Original Implementation Plan

## Context

The XuanTie RVM 0.6 spec defines a decoupled matrix extension for RISC-V targeting AI/ML workloads. It provides ~140+ instructions for matrix multiplication, load/store, configuration, data movement, and element-wise arithmetic across data types from INT4 to FP64. This plan adds full assembler/disassembler support, intrinsics, tests, and documentation to LLVM.

## Key Design Decisions

- **Extension name**: `xtheadmatrix` (experimental, version 0.6), following the existing `xthead*` vendor prefix convention
- **Decoder namespace**: `XTHeadMatrix` (separate from existing `XTHead` which uses `OPC_CUSTOM_0`)
- **Major opcode**: `OPC_CUSTOM_1` (0b0101011) for 32-bit; `0b0111111` for 64-bit instructions
- **Register class**: 8 matrix registers: `tr0-tr3` (tile) and `acc0-acc3` (accumulator), 3-bit encoded
- **CSR names**: Prefixed with `th.` (e.g. `th.xmcsr`, `th.mtilem`)

## Implementation Phases

### Phase 1: Infrastructure
- Feature definition in `RISCVFeatures.td`
- Register definitions in `RISCVRegisterInfo.td`
- CSR definitions in `RISCVSystemOperands.td`
- Disassembler support in `RISCVDisassembler.cpp`
- Include wiring in `RISCVInstrInfo.td`

### Phase 2: Instruction Definitions
- New file `RISCVInstrInfoXTHeadMatrix.td`
- 15 instruction format classes
- Configuration (7), Load/Store (56), Matrix Multiply (~30), Misc (~30), Element-wise (~70)

### Phase 3: LLVM IR Intrinsics
- New file `IntrinsicsRISCVXTHeadMatrix.td`

### Phase 4: Clang Builtins
- New file `BuiltinsRISCVXTHeadMatrix.td`

### Phase 5: Tests
- Assembly valid/invalid tests
- CSR name tests
- Attribute/arch tests

### Phase 6: Documentation
- Entry in `RISCVUsage.rst`

### Phase 7: 64-bit Instruction Support (deferred)
- 64-bit decoder path

## Files Summary

| Action | File |
|--------|------|
| CREATE | `llvm/lib/Target/RISCV/RISCVInstrInfoXTHeadMatrix.td` |
| CREATE | `llvm/include/llvm/IR/IntrinsicsRISCVXTHeadMatrix.td` |
| CREATE | `clang/include/clang/Basic/BuiltinsRISCVXTHeadMatrix.td` |
| CREATE | `llvm/test/MC/RISCV/xtheadmatrix-valid.s` |
| CREATE | `llvm/test/MC/RISCV/xtheadmatrix-invalid.s` |
| CREATE | `llvm/test/MC/RISCV/xtheadmatrix-csr.s` |
| MODIFY | `llvm/lib/Target/RISCV/RISCVFeatures.td` |
| MODIFY | `llvm/lib/Target/RISCV/RISCVRegisterInfo.td` |
| MODIFY | `llvm/lib/Target/RISCV/RISCVSystemOperands.td` |
| MODIFY | `llvm/lib/Target/RISCV/RISCVInstrInfo.td` |
| MODIFY | `llvm/lib/Target/RISCV/Disassembler/RISCVDisassembler.cpp` |
| MODIFY | `llvm/include/llvm/IR/IntrinsicsRISCV.td` |
| MODIFY | `clang/include/clang/Basic/BuiltinsRISCV.td` |
| MODIFY | `llvm/docs/RISCVUsage.rst` |
