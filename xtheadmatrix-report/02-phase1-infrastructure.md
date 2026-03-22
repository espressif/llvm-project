# Phase 1: Infrastructure

**Verification status**: Register model (encodings, classes, spill/reload, calling
convention, inline asm constraints), CSR definitions (all 31 addresses), and feature
definitions all verified correct across 13 verification rounds.

## 1.1 Feature Definition (`RISCVFeatures.td`)

Added after the existing `XTHeadVdot` extension:

```tablegen
def FeatureVendorXTHeadMatrix
    : RISCVExperimentalExtension<0, 6,
                     "T-Head Matrix Extension (RVM)">;
def HasVendorXTHeadMatrix : Predicate<"Subtarget->hasVendorXTHeadMatrix()">,
                            AssemblerPredicate<(all_of FeatureVendorXTHeadMatrix),
                                "'XTHeadMatrix' (T-Head Matrix Extension)">;
```

Key decisions:
- Used `RISCVExperimentalExtension` (not `RISCVExtension`) since this is version 0.6 (not yet stable)
- This means the extension requires `+experimental-xtheadmatrix` flag in `-mattr`

## 1.2 Register Definitions (`RISCVRegisterInfo.td`)

Added 8 matrix registers organized in 3 register classes:

```tablegen
// 8 matrix registers: tr0-tr3 (tile), acc0-acc3 (accumulator)
foreach Index = 0-3 in
  def THRVM_TR#Index : RISCVReg<Index, "tr"#Index>;
foreach Index = 0-3 in
  def THRVM_ACC#Index : RISCVReg<!add(Index, 4), "acc"#Index>;

let RegInfos = XLenRI in {
  def THRVMMR  : RISCVRegisterClass<[untyped], 32, (add ...)>;  // all 8
  def THRVMTR  : RISCVRegisterClass<[untyped], 32, (add ...)>;  // tr only
  def THRVMACC : RISCVRegisterClass<[untyped], 32, (add ...)>;  // acc only
}
```

Key decisions:
- TR registers have indices 0-3, ACC registers have indices 4-7
- All three register classes are defined though only `THRVMMR` is used in the current instructions
- `THRVMTR` and `THRVMACC` are available for future use when instructions need to restrict operands

## 1.3 CSR Definitions (`RISCVSystemOperands.td`)

Added 13 CSRs gated by `FeatureVendorXTHeadMatrix`:

| CSR Name | Address | Description |
|----------|---------|-------------|
| th.xmcsr | 0x806 | Matrix control/status |
| th.mtilem | 0x807 | Tile M dimension |
| th.mtilen | 0x808 | Tile N dimension |
| th.mtilek | 0x809 | Tile K dimension |
| th.xmxrm | 0x80a | Matrix rounding mode |
| th.xmsat | 0x80b | Matrix saturation |
| th.xmfflags | 0x80c | Matrix FP flags |
| th.xmfrm | 0x80d | Matrix FP rounding mode |
| th.xmsaten | 0x80e | Matrix saturation enable |
| th.xmisa | 0xcc0 | Matrix ISA info |
| th.xtlenb | 0xcc1 | Tile length in bytes |
| th.xtrlenb | 0xcc2 | Tile register length in bytes |
| th.xalenb | 0xcc3 | Accumulator length in bytes |

## 1.4 Disassembler (`RISCVDisassembler.cpp`)

Three additions:

### Register decode functions
```cpp
static constexpr MCPhysReg THRVMRegs[] = {
    RISCV::THRVM_TR0,  RISCV::THRVM_TR1,  RISCV::THRVM_TR2,
    RISCV::THRVM_TR3,  RISCV::THRVM_ACC0, RISCV::THRVM_ACC1,
    RISCV::THRVM_ACC2, RISCV::THRVM_ACC3};

static DecodeStatus DecodeTHRVMMRRegisterClass(...)  // all 8 regs
static DecodeStatus DecodeTHRVMTRRegisterClass(...)  // tr0-tr3 only
static DecodeStatus DecodeTHRVMACCRegisterClass(...) // acc0-acc3 only
```

### Feature group
```cpp
static constexpr FeatureBitset XTHeadMatrixGroup = {
    RISCV::FeatureVendorXTHeadMatrix};
```

### Decoder list entry
```cpp
{DecoderTableXTHeadMatrix32, XTHeadMatrixGroup, "T-Head Matrix extensions"},
```

Key decisions:
- Used a lookup table `THRVMRegs[]` instead of arithmetic offset (`THRVM_TR0 + RegNo`) to guarantee correct register mapping regardless of enum ordering
- Separate feature group (not merged into existing `XTHeadGroup`) since it uses a different decoder namespace

## 1.5 Include Wiring (`RISCVInstrInfo.td`)

Added after the existing XTHead include:
```tablegen
include "RISCVInstrInfoXTHead.td"
include "RISCVInstrInfoXTHeadMatrix.td"  // NEW
```
