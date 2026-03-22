# Phase 2: Instruction Definitions

**Verification status**: All 257 instruction encodings (227 base + 30 Zmpanel) verified
correct across 13 independent verification rounds against the golden RVM 0.6 spec.
0 encoding errors ever found.

## File: `RISCVInstrInfoXTHeadMatrix.td` (~1170 lines)

## Operand Definitions

Used standard LLVM RISC-V operand infrastructure:
```tablegen
def thrvmuimm10 : RISCVUImmOp<10>;  // 10-bit unsigned immediate (config)
def thrvmuimm3  : RISCVUImmOp<3>;   // 3-bit unsigned immediate (slide/broadcast/vector index)
```

Key learning: Custom `AsmOperandClass` definitions (e.g. `THRVMUimm10AsmOperand`) generate `isTHRVMUimm10()` methods that must exist in `RISCVAsmParser.cpp`. Using the standard `RISCVUImmOp<N>` avoids this because `isUImm<N>()` already exists.

## Instruction Format Classes (15 classes)

All inherit from `THRVMInst` base class which sets `OPC_CUSTOM_1` and `DecoderNamespace = "XTHeadMatrix"`.

### Base class
```tablegen
class THRVMInst<dag outs, dag ins, string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatOther> {
  let Inst{6-0} = OPC_CUSTOM_1.Value;
  let DecoderNamespace = "XTHeadMatrix";
}
```

### Format classes

| Class | Purpose | GPR handling | Key fields |
|-------|---------|--------------|------------|
| `THRVMInstCfgImm` | Config with immediate | rd=[11:7] (5-bit) | func4, ctrl, imm[9:2] |
| `THRVMInstCfgReg` | Config with GPR source | rd=[11:7], rs1=[19:15] | func4, ctrl |
| `THRVMInstCfgRelease` | mrelease (no ops) | none | func4, ctrl |
| `THRVMInstLoad` | Matrix load | rs1=[19:15], rs2=[24:20] | func4, stride, dsize, md |
| `THRVMInstStore` | Matrix store | rs1=[19:15], rs2=[24:20] | func4, stride, dsize, ms3 |
| `THRVMInstMMul` | Matrix multiply | none (3-bit matrix regs) | func4, ctrl, ssize, dsize |
| `THRVMInstMisc1` | 1-matrix-reg ops | none | mzero variants |
| `THRVMInstMisc2` | 2-matrix-reg ops | none | mmov.mm |
| `THRVMInstMisc3` | 3-matrix-reg ops | none | mpack variants |
| `THRVMInstMiscMX` | Matrix-to-GPR | rd=[11:7] (5-bit) | mmov*.x.m |
| `THRVMInstMiscXM` | GPR-to-matrix | rs1=[19:15] (5-bit) | mmov*.m.x, mdup*.m.x |
| `THRVMInstMiscSlide` | Slide with imm3 | none | mrslide*, mcslide* |
| `THRVMInstMiscBcast` | Broadcast with imm3 | none | mrbca, mcbca* |
| `THRVMInstEWConv` | EW conversion | none | func3=001, 2-reg |
| `THRVMInstEWBinMM` | EW binary matrix-matrix | none | func3=001, 3-reg |
| `THRVMInstEWBinMVI` | EW binary with imm3 | none | func3=001, imm3=[22:20] |

### Encoding Design Decisions

1. **GPR fields**: Configuration, load/store, and move instructions use 5-bit GPR fields at standard R-type positions ([19:15] for rs1, [24:20] for rs2, [11:7] for rd). This ensures compatibility with the standard GPR decoder.

2. **Load/Store stride**: Uses a single bit (bit 25) to distinguish element-stride (0) from tile-stride (1), freeing bits [24:20] for the full 5-bit rs2 GPR.

3. **MVI format**: The `imm3` replaces the `ms2` position (bits [22:20]) since MVI instructions have `md, ms1, imm3` operands (not 4 operands).

## Instruction Categories

### Configuration (7 instructions)
```
th.mrelease                    - Release matrix state
th.msettile{m,k,n}i rd, imm   - Set tile dimension with immediate
th.msettile{m,k,n}  rd, rs1   - Set tile dimension from GPR
```

### Load/Store (56 instructions)
Using multiclasses `THRVMLoadInst` and `THRVMStoreInst`:
- Load types: mla (A-matrix), mlb (B-matrix), mlc (C-matrix), mlm (whole-register)
- Store types: msa, msb, msc, msm
- Stride modes: element-stride (e), tile-stride (te)
- Element widths: 8, 16, 32, 64-bit

### Matrix Multiply (~30 instructions)
- **FP**: mfmacc.{h,s,d}, widening (h.e4, h.e5, bf16.e4, bf16.e5, s.h, s.bf16, s.e4, s.e5, s.tf32, d.s)
- **Int**: mmacc{,u,us,su}.{w.b, d.h, w.q}
- **Partial**: pmmacc{,u,us,su}.w.b
- **Bypass**: mmacc{,u}.w.bp

### Misc (~30 instructions)
- Zero: mzero, mzero{2r,4r,8r}
- Move: mmov.mm, mmov{b,h,w,d}.{x.m, m.x}
- Duplicate: mdup{b,h,w,d}.m.x
- Pack: mpack, mpackhl, mpackhh
- Slide: mrslide{down,up}, mcslide{down,up}.{b,h,w,d}
- Broadcast: mrbca.mv.i, mcbca{b,h,w,d}.mv.i

### Element-wise (~70 instructions)
Using multiclasses for systematic generation:
- **FP conversions**: mfcvt{l,h}.* (20+ variants)
- **Int-float conversions**: m{u,s}fcvt{l,h}.*, mf{u,s}cvt.* (12 variants)
- **Fixed-point clip**: mn4clip{l,h}{,u}.w.{mm,mv.i} (8 variants)
- **Packed conversions**: m{u,s}cvt{l,h}.{b.p,w.b.q} (8 variants)
- **Int arithmetic**: m{add,sub,mul,mulh,max,umax,min,umin,srl,sll,sra}.w.{mm,mv.i} (22 variants)
- **FP arithmetic**: mf{add,sub,mul,max,min}.{h,s,d}.{mm,mv.i} (30 variants)
