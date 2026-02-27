# Future Work / Not Implemented

## Phase 7: 64-bit Instruction Support (Deferred)
The original plan included 64-bit instruction support using `RVInst64` and a `getInstruction64()` method. This was deferred because:
- The current disassembler returns `Fail` for 64-bit instructions (line 851-853 in `RISCVDisassembler.cpp`)
- No infrastructure exists for 64-bit custom instruction decoding
- The 32-bit instruction set covers the core functionality

To implement later:
1. Add `getInstruction64()` method to `RISCVDisassembler` (modeled after `getInstruction48()`)
2. Define `THRVMInst64` base class using `RVInst64`
3. Add 64-bit decoder list and dispatch in `getInstruction()`
4. Define 64-bit instruction variants

## Register Allocation Flexibility
The current ISel assigns every intrinsic to fixed physical matrix registers (e.g., all matmul always uses ACC0/TR0/TR1). This is correct for single-matrix-operation sequences but limits scheduling flexibility. Future work could:
- Allow callers to specify which matrix register to target via an additional intrinsic argument
- Introduce pseudo-instructions with virtual matrix register operands that resolve during register allocation
- Enable concurrent use of multiple accumulator/tile register sets in a single function

## Higher-level Matrix API
The current interface exposes raw hardware intrinsics/builtins. A higher-level API could:
- Provide tiling and loop-nest abstractions for common GEMM patterns
- Auto-select matmul variants based on data type and accumulation semantics
- Integrate with MLIR linalg or similar frameworks for automatic matrix tiling

## Attribute/Arch Tests
The `-march` acceptance of `xtheadmatrix0p6` was not explicitly tested in a separate attribute test file. It is implicitly tested via the `--mattr=+experimental-xtheadmatrix` flag used in all test files.

## Encoding Verification Status
All 227 instruction encodings have been verified against the RVM 0.6 spec:
- 24/24 programmatic bit-field checks passed (covering all 5 categories)
- 0 encoding conflicts detected
- All 5 original discrepancy categories (CONFIG, LOAD/STORE, MATMUL, MISC, ELEMENT-WISE) resolved and second-pass verified
- Known spec errata documented: matmul uop=01 in instruction_list.adoc should be uop=10; mfmin.s/mfmin.h names swapped in spec
