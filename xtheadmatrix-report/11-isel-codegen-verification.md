# ISel/CodeGen Implementation and Verification

## 1. Implementation Summary

The XTHeadMatrix ISel/CodeGen layer provides a complete path from LLVM IR intrinsics and Clang builtins down to machine instructions. The implementation uses a table-driven approach in `RISCVISelDAGToDAG.cpp`, avoiding pattern-based ISel entirely because all matrix registers are fixed (not allocatable by the register allocator).

### Architecture

- **227 intrinsic-to-instruction mappings** in a single lookup table (`THMatrixTable`)
- **15 `THMatrixCategory` dispatch cases** covering all operand formats
- **Fixed register assignment** -- every intrinsic maps to predetermined physical matrix registers
- **`hasSideEffects = 1`** on all instructions to model implicit matrix state correctly
- **THRVMMR operands in `(ins)` not `(outs)`** to prevent register allocator interaction

### Dispatch Flow

The `selectTHMatrix()` function is invoked from two points in `RISCVDAGToDAGISel::Select()`:

| ISD Opcode | Intrinsic Count | Description |
|---|---|---|
| `INTRINSIC_VOID` | 223 | All void-returning intrinsics (loads, stores, config, matmul, EW, conversions, misc) |
| `INTRINSIC_W_CHAIN` | 4 | `mmov{b,h,w,d}.x.m` -- return a GPR value from matrix register |

The function performs a linear scan of `THMatrixTable` to find the matching intrinsic ID, then dispatches on the `THMatrixCategory` enum to emit the correct operand sequence.

### THMatrixCategory Enum (15 cases)

| Category | Operand Pattern | Example Instructions |
|---|---|---|
| `THM_NoArgsOnly` | (none) | `mrelease` |
| `THM_NoArgs1Reg` | md | `mzero`, `mzero2r`, `mzero4r`, `mzero8r` |
| `THM_NoArgs2Reg` | md, ms1 | `mmov.mm`, all conversions |
| `THM_NoArgs3Reg` | md, ms2, ms1 | matmul, EW arith `.mm`, n4clip `.mm`, pack |
| `THM_LoadStrided` | md, (rs1), rs2 | `mla/b/c/at/bt/ct.e{8,16,32,64}` |
| `THM_LoadWhole` | md, (rs1) | `mlm.e{8,16,32,64}` |
| `THM_StoreStrided` | ms3, (rs1), rs2 | `msa/b/c/at/bt/ct.e{8,16,32,64}` |
| `THM_StoreWhole` | ms3, (rs1) | `msm.e{8,16,32,64}` |
| `THM_CfgImm` | uimm10 | `msettile{m,n,k}i` |
| `THM_CfgReg` | rs1 | `msettile{m,n,k}` |
| `THM_Imm2Reg` | md, ms1, uimm3 | slides, broadcasts |
| `THM_Imm3Reg` | md, ms2, ms1, uimm3 | EW arith `.mv.i`, n4clip `.mv.i` |
| `THM_ToGPR` | rd(GPR), ms2, rs1 | `mmov{b,h,w,d}.x.m` |
| `THM_FromGPR2` | md, rs2, rs1 | `mmov{b,h,w,d}.m.x` |
| `THM_FromGPR1` | md, rs2 | `mdup{b,h,w,d}.m.x` |

## 2. Register Assignment Convention

All matrix register assignments are fixed at compile time. The 8 matrix registers encode as 3-bit values: TR0=0, TR1=1, TR2=2, TR3=3, ACC0=4, ACC1=5, ACC2=6, ACC3=7.

### Load/Store Registers

| Operation | Register | Encoding | Rationale |
|---|---|---|---|
| Load A (`mla`) | TR0 | 0 | Tile register for first matmul operand |
| Load B (`mlb`) | TR1 | 1 | Tile register for second matmul operand |
| Load C (`mlc`) | ACC0 | 4 | Accumulator for matmul result |
| Load A^T (`mlat`) | TR2 | 2 | Transposed tile, separate from A |
| Load B^T (`mlbt`) | TR3 | 3 | Transposed tile, separate from B |
| Load C^T (`mlct`) | ACC1 | 5 | Transposed accumulator |
| Load M (`mlm`) | TR0 | 0 | Whole-register load to first tile |
| Stores | Mirror loads | -- | Same register as corresponding load |

### Matmul Registers

All matmul instructions (`mfmacc`, `mmacc`, `pmmacc`):
- **md = ACC0 (4)** -- accumulator destination per spec
- **ms1 = TR0 (0)** -- first tile operand
- **ms2 = TR1 (1)** -- second tile operand
- Semantics: `acc0 = acc0 + tr1 * tr0`

### Element-wise and Conversion Registers (102 instructions)

All element-wise arithmetic (integer and FP), type conversions, n4clip, and packed conversions use **accumulator registers only**, per spec:
- **md = ACC0 (4)**
- **ms1 = ACC1 (5)** (for 2-operand) or **ms1 = ACC2 (6)** (for 3-operand)
- **ms2 = ACC1 (5)** (for 3-operand)

Spec references:
- `inst32_format.adoc`: "md/ms1/ms2 should be 100-111" (accumulator range)
- `integer_ew.adoc`: "accumulation registers"
- `float_ew.adoc`: "accumulation registers"
- `type_convert.adoc`: "both accumulation registers"

### MISC Registers (slides, broadcasts, pack, mzero, mmov)

- **md = TR0 (0)**, **ms1 = TR1 (1)**, **ms2 = TR2 (2)** -- tile registers
- Pack: `md=TR0, ms2=TR2, ms1=TR1`

### GPR Move Registers

- `mmov.x.m` (ToGPR): ms2 = TR0
- `mmov.m.x` (FromGPR2): md = TR0
- `mdup.m.x` (FromGPR1): md = TR0

## 3. Critical Bug Fix: Register Assignment Correction

### Bug Description

During initial implementation, 102 ISel table entries for element-wise and conversion instructions incorrectly used **tile registers** (TR0/TR1/TR2) where the RVM 0.6 spec requires **accumulator registers** (ACC0/ACC1/ACC2). This affected:

- 26 FP format conversions (md=TR0, ms1=TR1)
- 12 float-int conversions (md=TR0, ms1=TR1)
- 8 n4clip `.mm` and `.mv.i` variants (md=TR0, ms1=TR1, ms2=TR2)
- 4 packed conversions (md=TR0, ms1=TR1)
- 22 integer EW `.w.mm` and `.w.mv.i` variants (md=TR0, ms1=TR1, ms2=TR2)
- 30 FP EW `.h/.s/.d.mm` and `.mv.i` variants (md=TR0, ms1=TR1, ms2=TR2)

### Root Cause

The initial implementation treated all non-matmul 3-register instructions identically, using the same tile register convention as MISC operations. It did not distinguish the register constraints for element-wise/conversion operations from those for slides/pack/broadcast.

### Fix

Changed all 102 affected entries from `TR0/TR1/TR2` to `ACC0/ACC1/ACC2`. Also added the `ACC2 = 6` constant which was previously unused.

Before:
```cpp
{Intrinsic::riscv_th_mfcvtl_h_e4, RISCV::TH_MFCVTL_H_E4, THM_NoArgs2Reg, TR0, TR1, 0},
{Intrinsic::riscv_th_madd_w_mm,    RISCV::TH_MADD_W_MM,    THM_NoArgs3Reg, TR0, TR1, TR2},
```

After:
```cpp
{Intrinsic::riscv_th_mfcvtl_h_e4, RISCV::TH_MFCVTL_H_E4, THM_NoArgs2Reg, ACC0, ACC1, 0},
{Intrinsic::riscv_th_madd_w_mm,    RISCV::TH_MADD_W_MM,    THM_NoArgs3Reg, ACC0, ACC1, ACC2},
```

## 4. Verification Results

### Round 1: Initial Review (4 independent agents)

- **Encoding verification**: 227/227 intrinsic-to-instruction mappings correct
- **Category dispatch**: All 15 categories produce correct operand sequences
- **Bug found**: 102 element-wise/conversion entries used tile registers instead of accumulator registers

### Round 2: Post-fix Verification (4 independent agents)

- **Register assignments**: All 227 entries verified correct against RVM 0.6 spec
- **End-to-end compilation**: All 22 instruction categories tested through `clang -> LLVM IR -> ISel -> machine code`
- **ISel test**: `xtheadmatrix-isel.ll` -- all patterns match expected machine instructions
- **E2E test**: `xtheadmatrix-codegen.c` -- all builtins compile through to correct assembly
- **XTHeadMatrix tests**: 5/5 pass (3 MC + 1 ISel + 1 E2E)
- **RISCV MC regression**: 557/557 tests pass (no regressions)

## 5. Files Changed for ISel/CodeGen

### New Files

| File | Lines | Description |
|---|---|---|
| `llvm/test/CodeGen/RISCV/xtheadmatrix-isel.ll` | ~300 | ISel pattern tests: intrinsic calls verify correct instruction selection and register assignment |
| `clang/test/CodeGen/RISCV/xtheadmatrix-codegen.c` | ~250 | End-to-end tests: builtin calls compile to expected assembly output |

### Modified Files

| File | Changes | Description |
|---|---|---|
| `llvm/lib/Target/RISCV/RISCVISelDAGToDAG.cpp` | +400 lines | `THMatrixCategory` enum, `THMatrixIntrEntry` struct, `THMatrixTable` (227 entries), `getTHMatrixReg()` helper, `lookupTHMatrixIntr()`, `selectTHMatrix()` dispatch, two call sites in `Select()` |
| `llvm/lib/Target/RISCV/RISCVISelDAGToDAG.h` | +1 line | `selectTHMatrix(SDNode *)` declaration |
| `llvm/lib/Target/RISCV/RISCVRegisterInfo.cpp` | +10 lines | Reserve all 8 THRVMMR registers when `HasVendorXTHeadMatrix` is enabled |
| `llvm/lib/Target/RISCV/RISCVInstrInfoXTHeadMatrix.td` | Modified | `hasSideEffects = 1` on all instruction classes; THRVMMR operands moved to `(ins)` from `(outs)` for ISel compatibility |
| `llvm/docs/RISCV/RISCVXTHeadMatrix.md` | Updated | Documented register constraints, ISel approach, and current limitations |

### Design Decisions

1. **Table-driven, not pattern-based**: Matrix registers are not allocatable, so standard ISel patterns (which assume virtual registers) do not apply. The table-driven approach with `CurDAG->getRegister()` for physical registers is the correct design.

2. **`hasSideEffects = 1`**: All instructions carry implicit matrix state. Without this flag, the compiler could reorder or eliminate matrix operations.

3. **THRVMMR in `(ins)` not `(outs)`**: Placing matrix register operands in `(ins)` prevents the register allocator from trying to assign virtual registers to them, since all assignments are fixed.

4. **All 8 matrix registers reserved**: `RISCVRegisterInfo::getReservedRegs()` marks TR0-TR3 and ACC0-ACC3 as reserved when the extension is enabled, ensuring no other code generator pass touches them.
