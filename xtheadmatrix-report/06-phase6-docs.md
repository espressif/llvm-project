# Phase 6: Documentation

## RISCVUsage.rst Update

Added entry under the Vendor Extensions section, between `XTHeadVdot` and `XVentanaCondOps`:

```rst
``XTHeadMatrix``
  LLVM implements `version 0.6 of the XuanTie RVM (RISC-V Matrix Extension) specification` by T-HEAD of Alibaba.  This experimental extension provides decoupled matrix instructions for AI/ML workloads.  Support includes assembler/disassembler for 140+ instructions across configuration, load/store, matrix multiply (FP and integer, including widening and partial variants), data movement, and element-wise arithmetic categories.  LLVM IR intrinsics (``int_riscv_th_*``) and Clang builtins (``__builtin_riscv_th_*``) are defined for all instruction categories.  Eight matrix registers (``tr0``-``tr3`` tile, ``acc0``-``acc3`` accumulator) and 13 CSRs are supported.  All instructions are prefixed with `th.` as described in the specification.
```

## Report files

The `xtheadmatrix-report/` directory contains the full implementation report:
- `00-overview.md` - Project summary and final status
- `01-plan.md` - Original implementation plan
- `02-phase1-infrastructure.md` - Feature flags, registers, CSRs, disassembler
- `03-phase2-instructions.md` - Instruction format classes and definitions
- `04-phase3-4-intrinsics-builtins.md` - Full intrinsic and builtin coverage
- `05-phase5-tests.md` - Test coverage (278 test entries)
- `06-phase6-docs.md` - This file
- `07-build-and-issues.md` - Build history and issues resolved
- `08-files-changed.md` - Complete file change summary
- `09-instruction-encoding-reference.md` - Sample encodings
- `10-future-work.md` - Deferred items (64-bit instructions, codegen integration)
