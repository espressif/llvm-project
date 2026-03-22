# Instruction Encoding Reference

## Bit Layout

All 227 base instructions (+ 30 Zmpanel = 257 total) use `OPC_CUSTOM_1` (0b0101011, bits[6:0]).
All 257 encodings verified correct across 13 independent verification rounds (0 implementation errors).

```
[31:28] func4  [27:26] uop  [25:23] ctrl/size_sup  [22:20] ms2
[19:18] s_size [17:15] ms1  [14:12] func3           [11:10] d_size  [9:7] md
```

uop: 00=Config, 01=Load/Store, 10=Matmul, 11=MISC. Element-wise uses func3=001.

## Sample Encodings (verified against RVM 0.6 spec)

### Configuration Instructions (uop=00, func3=000)
Config instructions have no rd output (bits[11:7]=00000). bit[25]: 0=immediate, 1=register.
```
th.mrelease            [0x2b,0x00,0x00,0x00]  func4=0000
th.msettileki 42       [0x2b,0x00,0x15,0x10]  func4=0001, bit25=0, imm=42
th.msettilek  a0       [0x2b,0x00,0x05,0x12]  func4=0001, bit25=1
th.msettilemi 100      [0x2b,0x00,0x32,0x20]  func4=0010, bit25=0, imm=100
th.msettilem  a1       [0x2b,0x80,0x05,0x22]  func4=0010, bit25=1
th.msettileni 200      [0x2b,0x00,0x64,0x30]  func4=0011, bit25=0, imm=200
th.msettilen  a2       [0x2b,0x00,0x06,0x32]  func4=0011, bit25=1
```

### Load Instructions (uop=01, ls=0, func3=000)
bit[25]=ls: 0=load. func4: A=0000, B=0001, C=0010, M=0011, Ate=0100, Bte=0101, Cte=0110.
```
th.mlae8   tr0, (a0), a1   [0x2b,0x00,0xb5,0x04]  func4=0000, d_size=00
th.mlae32  tr2, (a0), a1   [0x2b,0x09,0xb5,0x04]  func4=0000, d_size=10
th.mlate8  tr0, (a0), a1   [0x2b,0x00,0xb5,0x44]  func4=0100, d_size=00
th.mlate32 tr2, (a0), a1   [0x2b,0x09,0xb5,0x44]  func4=0100, d_size=10
th.mlme32  tr2, (a0)       [0x2b,0x09,0x05,0x34]  func4=0011, rs2=00000
```

### Store Instructions (uop=01, ls=1, func3=000)
bit[25]=ls: 1=store. Same func4 as loads.
```
th.msae8   tr0, (a0), a1   [0x2b,0x00,0xb5,0x06]  func4=0000, d_size=00
th.msae32  tr2, (a0), a1   [0x2b,0x09,0xb5,0x06]  func4=0000, d_size=10
th.msate32 tr2, (a0), a1   [0x2b,0x09,0xb5,0x46]  func4=0100, d_size=10
th.msme32  tr2, (a0)       [0x2b,0x09,0x05,0x36]  func4=0011, rs2=00000
```

### Matrix Multiply Instructions (uop=10, func3=000)
FP: func4=0000. Int: func4=0001. Bypass: func4=0010.
```
th.mfmacc.h      acc0, tr0, tr1   [0x2b,0x86,0x04,0x08]  func4=0000, ctrl=000, s=01, d=01
th.mfmacc.s      acc0, tr0, tr1   [0x2b,0x8a,0x08,0x08]  func4=0000, ctrl=000, s=10, d=10
th.mfmacc.d      acc0, tr0, tr1   [0x2b,0x8e,0x0c,0x08]  func4=0000, ctrl=000, s=11, d=11
th.mfmacc.h.e4   acc0, tr0, tr1   [0x2b,0x86,0x80,0x08]  func4=0000, ctrl=001, s=00, d=01
th.mfmacc.s.h    acc0, tr0, tr1   [0x2b,0x8a,0x04,0x08]  func4=0000, ctrl=000, s=01, d=10
th.mfmacc.s.bf16 acc0, tr0, tr1   [0x2b,0x8a,0x84,0x08]  func4=0000, ctrl=001, s=01, d=10
th.mfmacc.d.s    acc0, tr0, tr1   [0x2b,0x8e,0x08,0x08]  func4=0000, ctrl=000, s=10, d=11
th.mmaccu.w.b    acc0, tr0, tr1   [0x2b,0x8a,0x00,0x18]  func4=0001, ctrl=000, s=00, d=10
th.mmacc.w.b     acc0, tr0, tr1   [0x2b,0x8a,0x80,0x19]  func4=0001, ctrl=011, s=00, d=10
th.mmaccus.w.b   acc0, tr0, tr1   [0x2b,0x8a,0x80,0x18]  func4=0001, ctrl=001, s=00, d=10
th.mmaccsu.w.b   acc0, tr0, tr1   [0x2b,0x8a,0x00,0x19]  func4=0001, ctrl=010, s=00, d=10
th.mmacc.d.h     acc0, tr0, tr1   [0x2b,0x8e,0x84,0x19]  func4=0001, ctrl=011, s=01, d=11
th.pmmacc.w.b    acc0, tr0, tr1   [0x2b,0x8a,0x80,0x1b]  func4=0001, ctrl=111, s=00, d=10
th.mmaccu.w.bp   acc0, tr0, tr1   [0x2b,0x8a,0x00,0x28]  func4=0010, ctrl=000, s=00, d=10
```

### Misc Instructions (uop=11, func3=000)
```
th.mzero           tr0             [0x2b,0x00,0x00,0x0c]  func4=0000, imm3=000
th.mzero2r         tr0             [0x2b,0x00,0x80,0x0c]  func4=0000, imm3=001
th.mzero4r         tr0             [0x2b,0x00,0x80,0x0d]  func4=0000, imm3=011
th.mzero8r         tr0             [0x2b,0x00,0x80,0x0f]  func4=0000, imm3=111
th.mmov.mm         tr0, tr1        [0x2b,0x80,0x00,0x1c]  func4=0001
th.mmovb.x.m       a0, tr0, a1    [0x2b,0x85,0x05,0x2c]  func4=0010, e_size=00
th.mmovw.x.m       a0, tr0, a1    [0x2b,0x85,0x05,0x2d]  func4=0010, e_size=10
th.mmovb.m.x       tr0, a0, a1    [0x2b,0x80,0xa5,0x3e]  func4=0011, bit25=1, d_size=00
th.mmovw.m.x       tr0, a0, a1    [0x2b,0x88,0xa5,0x3e]  func4=0011, bit25=1, d_size=10
th.mdupw.m.x       tr0, a0        [0x2b,0x08,0xa0,0x3c]  func4=0011, bit25=0, rs1=00000
th.mpack           tr0, tr1, tr2  [0x2b,0x00,0x11,0x4c]  func4=0100
th.mrslidedown     tr0, tr1, 3    [0x2b,0x80,0x80,0x5d]  func4=0101, uimm3=3
th.mrslideup       tr0, tr1, 2    [0x2b,0x80,0x00,0x6d]  func4=0110, uimm3=2
th.mcslidedown.b   tr0, tr1, 1    [0x2b,0x80,0x80,0x7c]  func4=0111, s=00, d=00
th.mrbca.mv.i      tr0, tr1, 3    [0x2b,0x80,0x80,0x9d]  func4=1001
th.mcbcab.mv.i     tr0, tr1, 2    [0x2b,0x80,0x00,0xad]  func4=1010
```

### Element-wise Instructions (func3=001)
```
th.mfcvtl.h.e4       tr0, tr1              [0x2b,0x94,0x00,0x00]  uop=00, func4=0000
th.mfcvtl.e4.h       tr0, tr1              [0x2b,0x90,0x04,0x00]  uop=00, func4=0000
th.mfcvtl.s.h        tr0, tr1              [0x2b,0x98,0x04,0x00]  uop=00, func4=0000
th.mfcvt.tf32.s      tr0, tr1              [0x2b,0x98,0x08,0x03]  uop=00, func4=0000
th.mufcvt.s.w        tr0, tr1              [0x2b,0x98,0x08,0x10]  uop=00, func4=0001
th.mfucvt.w.s        tr0, tr1              [0x2b,0x98,0x08,0x12]  uop=00, func4=0001
th.mn4clipl.w.mm     tr0, tr1, tr2         [0x2b,0x18,0x99,0x23]  uop=00, func4=0010, ctrl=111
th.mn4clipl.w.mv.i   tr0, tr1, tr2, 3      [0x2b,0x18,0x99,0x21]  uop=00, func4=0010, imm3=3
th.madd.w.mm         tr0, tr1, tr2         [0x2b,0x18,0x99,0x07]  uop=01, func4=0000, ctrl=111
th.madd.w.mv.i       tr0, tr1, tr2, 3      [0x2b,0x18,0x99,0x05]  uop=01, func4=0000, imm3=3
th.mfadd.h.mm        tr0, tr1, tr2         [0x2b,0x14,0x95,0x0b]  uop=10, func4=0000, ctrl=111
th.mfadd.s.mm        tr0, tr1, tr2         [0x2b,0x18,0x99,0x0b]  uop=10, func4=0000, ctrl=111
th.mfadd.d.mm        tr0, tr1, tr2         [0x2b,0x1c,0x9d,0x0b]  uop=10, func4=0000, ctrl=111
th.mfadd.s.mv.i      tr0, tr1, tr2, 3      [0x2b,0x18,0x99,0x09]  uop=10, func4=0000, imm3=3
```

## CSR Encodings
```
csrr a0, th.xmcsr    [0x73,0x25,0x60,0x80]  (addr=0x806)
csrr a0, th.mtilem   [0x73,0x25,0x70,0x80]  (addr=0x807)
csrr a0, th.mtilen   [0x73,0x26,0x80,0x80]  (addr=0x808)
csrr a0, th.mtilek   [0x73,0x26,0x90,0x80]  (addr=0x809)
csrr a0, th.xmxrm    [0x73,0x27,0xa0,0x80]  (addr=0x80a)
csrr a0, th.xmsat    [0x73,0x27,0xb0,0x80]  (addr=0x80b)
csrr a0, th.xmfflags [0x73,0x24,0xc0,0x80]  (addr=0x80c)
csrr a0, th.xmfrm    [0x73,0x24,0xd0,0x80]  (addr=0x80d)
csrr a0, th.xmsaten  [0x73,0x22,0xe0,0x80]  (addr=0x80e)
csrr a0, th.xmisa    [0x73,0x25,0x00,0xcc]  (addr=0xcc0)
csrr a0, th.xtlenb   [0x73,0x25,0x10,0xcc]  (addr=0xcc1)
csrr a0, th.xtrlenb  [0x73,0x25,0x20,0xcc]  (addr=0xcc2)
csrr a0, th.xalenb   [0x73,0x25,0x30,0xcc]  (addr=0xcc3)
```
