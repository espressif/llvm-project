# Build History and Issues Resolved

## Build Configuration

```bash
cmake -G Ninja ../llvm \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_TARGETS_TO_BUILD=RISCV \
  -DLLVM_ENABLE_PROJECTS="clang" \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++
```

## Build Iterations and Issues

### Iteration 1: TableGen errors
**Error**: `could not find field for operand 'imm3'`

**Root cause**: The `THRVMInstEWBinMVI` class declared `bits<3> imm3` but never assigned it to any `Inst{...}` bits.

**Fix**:
- Changed the MVI format to have `md, ms1, imm3` (3 operands) instead of `md, ms2, ms1, imm3` (4 operands)
- Assigned `imm3` to bits [22:20] (the same position as `ms2` in the MM variant)
- Updated all MVI instruction definitions and multiclasses accordingly

### Iteration 2: AsmParser errors
**Error**: `no member named 'isTHRVMUimm10' in 'RISCVOperand'`

**Root cause**: Custom `AsmOperandClass` definitions generate `is<Name>()` predicate methods that must exist in `RISCVAsmParser.cpp`. The custom classes `THRVMUimm10AsmOperand` and `THRVMUimm3AsmOperand` had unique names.

**Fix**: Replaced custom operand classes with standard `RISCVUImmOp<N>`:
```tablegen
// BEFORE (broken):
def THRVMUimm10AsmOperand : AsmOperandClass { ... }
def thrvmuimm10 : RISCVOp { ... }

// AFTER (working):
def thrvmuimm10 : RISCVUImmOp<10>;
def thrvmuimm3  : RISCVUImmOp<3>;
```

### Iteration 3: Decoding conflicts
**Error**: `Decoding conflict encountered` - MLAE_E8 vs MLATE_E8

**Root cause**: Load/store instructions used 3-bit GPR fields (only encoding lower 3 bits of 5-bit GPR index), and the `ctrl` field differentiation was lost when restructuring.

**Fix**: Redesigned load/store encoding:
- Changed from 3-bit ctrl field to 1-bit `stride` flag (bit 25)
- Moved GPRs to standard R-type positions: rs1=[19:15], rs2=[24:20] (full 5-bit)
- This gives unique encodings: element-stride has bit 25=0, tile-stride has bit 25=1

### Iteration 4: GPR decode issues
**Error**: Configuration instructions decoded `a0` as `sp` in disassembly

**Root cause**: Config instruction `rd` field only used bits [9:7] (3 bits) for a 5-bit GPR. GPR index 10 (a0) was being truncated to 2 (sp).

**Fix**: Changed config instructions to use bits [11:7] for the full 5-bit GPR rd.

### Iteration 5: Register name issues
**Error**: `acc0` displayed as `v0` in disassembly

**Root cause**: Register decoder used arithmetic offset `RISCV::THRVM_TR0 + RegNo` which didn't correctly map to ACC registers due to enum ordering.

**Fix**: Used explicit lookup table:
```cpp
static constexpr MCPhysReg THRVMRegs[] = {
    RISCV::THRVM_TR0,  RISCV::THRVM_TR1,  RISCV::THRVM_TR2,
    RISCV::THRVM_TR3,  RISCV::THRVM_ACC0, RISCV::THRVM_ACC1,
    RISCV::THRVM_ACC2, RISCV::THRVM_ACC3};
```

### Iteration 6: Unused template parameter warnings
**Warning**: Multiple `unused template argument` warnings for `dsize`, `ssize` parameters

**Fix**: Removed unused parameters from class definitions and updated all instantiation sites.

### Iteration 7: Encoding rework (full spec alignment)
**Issue**: Systematic encoding verification against the RVM 0.6 spec revealed discrepancies across all 5 instruction categories.

**Discrepancies found and fixed**:
1. **CONFIG**: `th.mrelease` had wrong func4 (was 0001, fixed to 0000); immediate config instructions needed `bit[25]=0` enforcement
2. **LOAD/STORE**: Tile-stride mnemonic concatenation produced double-e (e.g. `th.mlatee8` instead of `th.mlate8`); fix: shorten base mnemonic from `"mlate"` to `"mlat"` so multiclass appending `"e8"` yields correct `"th.mlate8"`
3. **MATMUL**: Bypass variants needed func4=0010 (was 0001); signed/unsigned ctrl field values corrected
4. **MISC**: `th.mzero` needed func4=0000 with imm3=000; multi-register zero variants needed correct imm3 encoding (001/011/111); `th.mdupw.m.x` needed bit[25]=0 with rs1=00000
5. **ELEMENT-WISE**: Conversion source/dest size fields corrected; n4clip ctrl field values aligned with spec

**Additional issue**: A pre-commit linter reverted the tile-stride mnemonic fix twice by modifying the .td file. Required re-reading exact file content and re-applying edits with correct context after each linter pass.

**Verification**: 24/24 programmatic bit-field checks passed; 110 instructions assembled with 0 errors in comprehensive smoke test.

## Final Build Status

```
Build: 0 errors (2 pre-existing unused function warnings in disassembler)
Tests: All 3 test files pass (xtheadmatrix-valid.s, xtheadmatrix-invalid.s, xtheadmatrix-csr.s)
Instruction count: 227 total (119 standalone defs + 108 multiclass expansions)
Encoding verification: 227/227 verified against RVM 0.6 spec, 0 conflicts
```
