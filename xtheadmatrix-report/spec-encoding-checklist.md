# XTHeadMatrix RVM 0.6 — Complete Encoding Verification Reference

All instructions use `OPC_CUSTOM_1` = `0b0101011` (bits[6:0]).

Standard bit layout:
```
[31:28] func4  [27:26] uop  [25:23] ctrl/size_sup  [22:20] ms2
[19:18] s_size [17:15] ms1  [14:12] func3           [11:10] d_size  [9:7] md
```

**Verification status**: All 257 encodings verified correct across 13 independent verification rounds (0 implementation errors ever found).

**Known spec errata in instruction_list.adoc** (5 total):
1. Matmul shows uop=01 — should be **uop=10** (per inst32_format.adoc; uop=01 collides with load/store)
2. mfmin.s/mfmin.h names are swapped — use the encoding values, not the names
3. `pmmaaccus.w.b` has doubled 'a' — should be `pmmaccus.w.b`
4. `mbce{8,16,32,64}` element broadcast described in prose (broadcast.adoc) but has NO encoding assigned
5. Zmpanel compute encoding table in zmpanel.adoc mislabels bits[19:15] as `rs1=00000`; bits[19:18] carry `s_size`

---

## 1. CONFIG (uop=00, func3=000)

Special layout: `[31:28]func4 [27:26]uop=00 [25]ctrl [24:15]imm10_or_{rs2,rs1} [14:12]func3=000 [11:7]nop=00000 [6:0]opcode`

bits[11:7] = 00000 (no rd output). bit[25] = 0 for immediate, 1 for register.

| # | Mnemonic | func4 | uop | bit25 | bits[24:15] | func3 | bits[11:7] |
|---|----------|-------|-----|-------|-------------|-------|------------|
| 1 | th.mrelease | 0000 | 00 | 0 | 0000000000 | 000 | 00000 |
| 2 | th.msettileki | 0001 | 00 | 0 | uimm10[9:0] | 000 | 00000 |
| 3 | th.msettilek | 0001 | 00 | 1 | {00000,rs1} | 000 | 00000 |
| 4 | th.msettilemi | 0010 | 00 | 0 | uimm10[9:0] | 000 | 00000 |
| 5 | th.msettilem | 0010 | 00 | 1 | {00000,rs1} | 000 | 00000 |
| 6 | th.msettileni | 0011 | 00 | 0 | uimm10[9:0] | 000 | 00000 |
| 7 | th.msettilen | 0011 | 00 | 1 | {00000,rs1} | 000 | 00000 |

**Total: 7**

---

## 2. LOAD/STORE (uop=01, func3=000)

Special layout: `[31:28]func4 [27:26]uop=01 [25]ls [24:20]rs2 [19:15]rs1 [14:12]func3=000 [11:10]d_size [9:7]md/ms3 [6:0]opcode`

ls=0: Load (md), ls=1: Store (ms3). d_size: 00=e8, 01=e16, 10=e32, 11=e64.

| # | Mnemonic | func4 | uop | ls | rs2 | func3 | d_size | Notes |
|---|----------|-------|-----|----|-----|-------|--------|-------|
| 8 | th.mlae8 | 0000 | 01 | 0 | rs2 | 000 | 00 | load A elem-stride |
| 9 | th.mlae16 | 0000 | 01 | 0 | rs2 | 000 | 01 | |
| 10 | th.mlae32 | 0000 | 01 | 0 | rs2 | 000 | 10 | |
| 11 | th.mlae64 | 0000 | 01 | 0 | rs2 | 000 | 11 | |
| 12 | th.mlbe8 | 0001 | 01 | 0 | rs2 | 000 | 00 | load B elem-stride |
| 13 | th.mlbe16 | 0001 | 01 | 0 | rs2 | 000 | 01 | |
| 14 | th.mlbe32 | 0001 | 01 | 0 | rs2 | 000 | 10 | |
| 15 | th.mlbe64 | 0001 | 01 | 0 | rs2 | 000 | 11 | |
| 16 | th.mlce8 | 0010 | 01 | 0 | rs2 | 000 | 00 | load C elem-stride |
| 17 | th.mlce16 | 0010 | 01 | 0 | rs2 | 000 | 01 | |
| 18 | th.mlce32 | 0010 | 01 | 0 | rs2 | 000 | 10 | |
| 19 | th.mlce64 | 0010 | 01 | 0 | rs2 | 000 | 11 | |
| 20 | th.mlme8 | 0011 | 01 | 0 | 00000 | 000 | 00 | load M whole-reg (no rs2) |
| 21 | th.mlme16 | 0011 | 01 | 0 | 00000 | 000 | 01 | |
| 22 | th.mlme32 | 0011 | 01 | 0 | 00000 | 000 | 10 | |
| 23 | th.mlme64 | 0011 | 01 | 0 | 00000 | 000 | 11 | |
| 24 | th.mlate8 | 0100 | 01 | 0 | rs2 | 000 | 00 | load A tile-stride |
| 25 | th.mlate16 | 0100 | 01 | 0 | rs2 | 000 | 01 | |
| 26 | th.mlate32 | 0100 | 01 | 0 | rs2 | 000 | 10 | |
| 27 | th.mlate64 | 0100 | 01 | 0 | rs2 | 000 | 11 | |
| 28 | th.mlbte8 | 0101 | 01 | 0 | rs2 | 000 | 00 | load B tile-stride |
| 29 | th.mlbte16 | 0101 | 01 | 0 | rs2 | 000 | 01 | |
| 30 | th.mlbte32 | 0101 | 01 | 0 | rs2 | 000 | 10 | |
| 31 | th.mlbte64 | 0101 | 01 | 0 | rs2 | 000 | 11 | |
| 32 | th.mlcte8 | 0110 | 01 | 0 | rs2 | 000 | 00 | load C tile-stride |
| 33 | th.mlcte16 | 0110 | 01 | 0 | rs2 | 000 | 01 | |
| 34 | th.mlcte32 | 0110 | 01 | 0 | rs2 | 000 | 10 | |
| 35 | th.mlcte64 | 0110 | 01 | 0 | rs2 | 000 | 11 | |
| 36 | th.msae8 | 0000 | 01 | 1 | rs2 | 000 | 00 | store A elem-stride |
| 37 | th.msae16 | 0000 | 01 | 1 | rs2 | 000 | 01 | |
| 38 | th.msae32 | 0000 | 01 | 1 | rs2 | 000 | 10 | |
| 39 | th.msae64 | 0000 | 01 | 1 | rs2 | 000 | 11 | |
| 40 | th.msbe8 | 0001 | 01 | 1 | rs2 | 000 | 00 | store B elem-stride |
| 41 | th.msbe16 | 0001 | 01 | 1 | rs2 | 000 | 01 | |
| 42 | th.msbe32 | 0001 | 01 | 1 | rs2 | 000 | 10 | |
| 43 | th.msbe64 | 0001 | 01 | 1 | rs2 | 000 | 11 | |
| 44 | th.msce8 | 0010 | 01 | 1 | rs2 | 000 | 00 | store C elem-stride |
| 45 | th.msce16 | 0010 | 01 | 1 | rs2 | 000 | 01 | |
| 46 | th.msce32 | 0010 | 01 | 1 | rs2 | 000 | 10 | |
| 47 | th.msce64 | 0010 | 01 | 1 | rs2 | 000 | 11 | |
| 48 | th.msme8 | 0011 | 01 | 1 | 00000 | 000 | 00 | store M whole-reg (no rs2) |
| 49 | th.msme16 | 0011 | 01 | 1 | 00000 | 000 | 01 | |
| 50 | th.msme32 | 0011 | 01 | 1 | 00000 | 000 | 10 | |
| 51 | th.msme64 | 0011 | 01 | 1 | 00000 | 000 | 11 | |
| 52 | th.msate8 | 0100 | 01 | 1 | rs2 | 000 | 00 | store A tile-stride |
| 53 | th.msate16 | 0100 | 01 | 1 | rs2 | 000 | 01 | |
| 54 | th.msate32 | 0100 | 01 | 1 | rs2 | 000 | 10 | |
| 55 | th.msate64 | 0100 | 01 | 1 | rs2 | 000 | 11 | |
| 56 | th.msbte8 | 0101 | 01 | 1 | rs2 | 000 | 00 | store B tile-stride |
| 57 | th.msbte16 | 0101 | 01 | 1 | rs2 | 000 | 01 | |
| 58 | th.msbte32 | 0101 | 01 | 1 | rs2 | 000 | 10 | |
| 59 | th.msbte64 | 0101 | 01 | 1 | rs2 | 000 | 11 | |
| 60 | th.mscte8 | 0110 | 01 | 1 | rs2 | 000 | 00 | store C tile-stride |
| 61 | th.mscte16 | 0110 | 01 | 1 | rs2 | 000 | 01 | |
| 62 | th.mscte32 | 0110 | 01 | 1 | rs2 | 000 | 10 | |
| 63 | th.mscte64 | 0110 | 01 | 1 | rs2 | 000 | 11 | |

**Total: 56**

---

## 3. MATMUL (uop=10, func3=000)

Standard layout: `[31:28]func4 [27:26]uop=10 [25:23]size_sup [22:20]ms2 [19:18]s_size [17:15]ms1 [14:12]func3=000 [11:10]d_size [9:7]md [6:0]opcode`

**NOTE**: instruction_list.adoc erroneously shows uop=01 for matmul. The correct value is **uop=10** per inst32_format.adoc (uop=01 is load/store).

### 3a. FP Matmul (func4=0000)

| # | Mnemonic | func4 | uop | size_sup | ms2 | s_size | ms1 | func3 | d_size | md |
|---|----------|-------|-----|----------|-----|--------|-----|-------|--------|-----|
| 64 | th.mfmacc.h.e5 | 0000 | 10 | 000 | ms2 | 00 | ms1 | 000 | 01 | md |
| 65 | th.mfmacc.h.e4 | 0000 | 10 | 001 | ms2 | 00 | ms1 | 000 | 01 | md |
| 66 | th.mfmacc.bf16.e5 | 0000 | 10 | 100 | ms2 | 00 | ms1 | 000 | 01 | md |
| 67 | th.mfmacc.bf16.e4 | 0000 | 10 | 101 | ms2 | 00 | ms1 | 000 | 01 | md |
| 68 | th.mfmacc.s.e5 | 0000 | 10 | 000 | ms2 | 00 | ms1 | 000 | 10 | md |
| 69 | th.mfmacc.s.e4 | 0000 | 10 | 001 | ms2 | 00 | ms1 | 000 | 10 | md |
| 70 | th.mfmacc.h | 0000 | 10 | 000 | ms2 | 01 | ms1 | 000 | 01 | md |
| 71 | th.mfmacc.s.h | 0000 | 10 | 000 | ms2 | 01 | ms1 | 000 | 10 | md |
| 72 | th.mfmacc.s.bf16 | 0000 | 10 | 001 | ms2 | 01 | ms1 | 000 | 10 | md |
| 73 | th.mfmacc.s.tf32 | 0000 | 10 | 001 | ms2 | 10 | ms1 | 000 | 10 | md |
| 74 | th.mfmacc.s | 0000 | 10 | 000 | ms2 | 10 | ms1 | 000 | 10 | md |
| 75 | th.mfmacc.d.s | 0000 | 10 | 000 | ms2 | 10 | ms1 | 000 | 11 | md |
| 76 | th.mfmacc.d | 0000 | 10 | 000 | ms2 | 11 | ms1 | 000 | 11 | md |

### 3b. Integer Matmul (func4=0001)

size_sup: bit[24]=ms1_signed, bit[23]=ms2_signed, bit[25]=int4/partial mode

| # | Mnemonic | func4 | uop | size_sup | ms2 | s_size | ms1 | func3 | d_size | md |
|---|----------|-------|-----|----------|-----|--------|-----|-------|--------|-----|
| 77 | th.mmaccu.w.b | 0001 | 10 | 000 | ms2 | 00 | ms1 | 000 | 10 | md |
| 78 | th.mmaccus.w.b | 0001 | 10 | 001 | ms2 | 00 | ms1 | 000 | 10 | md |
| 79 | th.mmaccsu.w.b | 0001 | 10 | 010 | ms2 | 00 | ms1 | 000 | 10 | md |
| 80 | th.mmacc.w.b | 0001 | 10 | 011 | ms2 | 00 | ms1 | 000 | 10 | md |
| 81 | th.pmmaccu.w.b | 0001 | 10 | 100 | ms2 | 00 | ms1 | 000 | 10 | md |
| 82 | th.pmmaaccus.w.b | 0001 | 10 | 101 | ms2 | 00 | ms1 | 000 | 10 | md |
| 83 | th.pmmaccsu.w.b | 0001 | 10 | 110 | ms2 | 00 | ms1 | 000 | 10 | md |
| 84 | th.pmmacc.w.b | 0001 | 10 | 111 | ms2 | 00 | ms1 | 000 | 10 | md |
| 85 | th.mmaccu.d.h | 0001 | 10 | 000 | ms2 | 01 | ms1 | 000 | 11 | md |
| 86 | th.mmaccus.d.h | 0001 | 10 | 001 | ms2 | 01 | ms1 | 000 | 11 | md |
| 87 | th.mmaccsu.d.h | 0001 | 10 | 010 | ms2 | 01 | ms1 | 000 | 11 | md |
| 88 | th.mmacc.d.h | 0001 | 10 | 011 | ms2 | 01 | ms1 | 000 | 11 | md |

### 3c. Bypass Matmul (func4=0010)

| # | Mnemonic | func4 | uop | size_sup | ms2 | s_size | ms1 | func3 | d_size | md |
|---|----------|-------|-----|----------|-----|--------|-----|-------|--------|-----|
| 89 | th.mmaccu.w.bp | 0010 | 10 | 000 | ms2 | 00 | ms1 | 000 | 10 | md |
| 90 | th.mmacc.w.bp | 0010 | 10 | 011 | ms2 | 00 | ms1 | 000 | 10 | md |

**Total matmul: 27**

---

## 4. MISC (uop=11, func3=000)

### 4a. mzero (func4=0000)

Layout: `[31:28]0000 [27:26]11 [25:23]imm3 [22:20]000 [19:18]00 [17:15]000 [14:12]000 [11:10]00 [9:7]md`

All fields except md and imm3 are zero.

| # | Mnemonic | func4 | uop | imm3[25:23] | ms2 | s_size | ms1 | func3 | d_size | md |
|---|----------|-------|-----|-------------|-----|--------|-----|-------|--------|-----|
| 91 | th.mzero | 0000 | 11 | 000 | 000 | 00 | 000 | 000 | 00 | md |
| 92 | th.mzero2r | 0000 | 11 | 001 | 000 | 00 | 000 | 000 | 00 | md |
| 93 | th.mzero4r | 0000 | 11 | 011 | 000 | 00 | 000 | 000 | 00 | md |
| 94 | th.mzero8r | 0000 | 11 | 111 | 000 | 00 | 000 | 000 | 00 | md |

### 4b. mmov.mm (func4=0001)

| # | Mnemonic | func4 | uop | ctrl[25:23] | ms2 | s_size | ms1 | func3 | d_size | md |
|---|----------|-------|-----|-------------|-----|--------|-----|-------|--------|-----|
| 95 | th.mmov.mm | 0001 | 11 | 000 | 000 | 00 | ms1 | 000 | 00 | md |

### 4c. mmov.x.m (func4=0010) — Matrix-to-GPR

Special layout: `[31:28]0010 [27:26]11 [25]0 [24:23]e_size [22:20]ms2 [19:15]rs1(5-bit GPR) [14:12]000 [11:7]rd(5-bit GPR)`

bits[19:15] = rs1 (5-bit GPR, not split into s_size+ms1). bits[11:7] = rd (5-bit GPR, not split into d_size+md).

| # | Mnemonic | func4 | uop | bit25 | e_size[24:23] | ms2 | rs1[19:15] | func3 | rd[11:7] |
|---|----------|-------|-----|-------|---------------|-----|------------|-------|----------|
| 96 | th.mmovb.x.m | 0010 | 11 | 0 | 00 | ms2 | rs1 | 000 | rd |
| 97 | th.mmovh.x.m | 0010 | 11 | 0 | 01 | ms2 | rs1 | 000 | rd |
| 98 | th.mmovw.x.m | 0010 | 11 | 0 | 10 | ms2 | rs1 | 000 | rd |
| 99 | th.mmovd.x.m | 0010 | 11 | 0 | 11 | ms2 | rs1 | 000 | rd |

### 4d. mmov.m.x (func4=0011, ctrl=1) — GPR-to-Matrix

Special layout: `[31:28]0011 [27:26]11 [25]1 [24:20]rs2(5-bit GPR) [19:15]rs1(5-bit GPR) [14:12]000 [11:10]d_size [9:7]md`

| # | Mnemonic | func4 | uop | bit25 | rs2[24:20] | rs1[19:15] | func3 | d_size | md |
|---|----------|-------|-----|-------|------------|------------|-------|--------|-----|
| 100 | th.mmovb.m.x | 0011 | 11 | 1 | rs2 | rs1 | 000 | 00 | md |
| 101 | th.mmovh.m.x | 0011 | 11 | 1 | rs2 | rs1 | 000 | 01 | md |
| 102 | th.mmovw.m.x | 0011 | 11 | 1 | rs2 | rs1 | 000 | 10 | md |
| 103 | th.mmovd.m.x | 0011 | 11 | 1 | rs2 | rs1 | 000 | 11 | md |

### 4e. mdup.m.x (func4=0011, ctrl=0) — GPR Broadcast to Matrix

Special layout: `[31:28]0011 [27:26]11 [25]0 [24:20]rs2(5-bit GPR) [19:15]00000 [14:12]000 [11:10]d_size [9:7]md`

rs1 = 00000 (unused).

| # | Mnemonic | func4 | uop | bit25 | rs2[24:20] | rs1[19:15] | func3 | d_size | md |
|---|----------|-------|-----|-------|------------|------------|-------|--------|-----|
| 104 | th.mdupb.m.x | 0011 | 11 | 0 | rs2 | 00000 | 000 | 00 | md |
| 105 | th.mduph.m.x | 0011 | 11 | 0 | rs2 | 00000 | 000 | 01 | md |
| 106 | th.mdupw.m.x | 0011 | 11 | 0 | rs2 | 00000 | 000 | 10 | md |
| 107 | th.mdupd.m.x | 0011 | 11 | 0 | rs2 | 00000 | 000 | 11 | md |

### 4f. mpack (func4=0100)

Layout: `[31:28]0100 [27:26]11 [25]0 [24:23]pack_mode [22:20]ms2 [19:18]00 [17:15]ms1 [14:12]000 [11:10]00 [9:7]md`

| # | Mnemonic | func4 | uop | bit25 | pack[24:23] | ms2 | s_size | ms1 | func3 | d_size | md |
|---|----------|-------|-----|-------|-------------|-----|--------|-----|-------|--------|-----|
| 108 | th.mpack | 0100 | 11 | 0 | 00 | ms2 | 00 | ms1 | 000 | 00 | md |
| 109 | th.mpackhl | 0100 | 11 | 0 | 10 | ms2 | 00 | ms1 | 000 | 00 | md |
| 110 | th.mpackhh | 0100 | 11 | 0 | 11 | ms2 | 00 | ms1 | 000 | 00 | md |

### 4g. Slide: mrslidedown/mrslideup (func4=0101/0110)

Layout: `[31:28]func4 [27:26]11 [25:23]uimm3 [22:20]000 [19:18]00 [17:15]ms1 [14:12]000 [11:10]00 [9:7]md`

ms2=000 (unused). s_size=00, d_size=00 for row slides.

| # | Mnemonic | func4 | uop | uimm3[25:23] | ms2 | s_size | ms1 | func3 | d_size | md |
|---|----------|-------|-----|--------------|-----|--------|-----|-------|--------|-----|
| 111 | th.mrslidedown | 0101 | 11 | uimm3 | 000 | 00 | ms1 | 000 | 00 | md |
| 112 | th.mrslideup | 0110 | 11 | uimm3 | 000 | 00 | ms1 | 000 | 00 | md |

### 4h. Slide: mcslidedown/mcslideup (func4=0111/1000)

s_size = d_size (matching element size).

| # | Mnemonic | func4 | uop | uimm3[25:23] | ms2 | s_size | ms1 | func3 | d_size | md |
|---|----------|-------|-----|--------------|-----|--------|-----|-------|--------|-----|
| 113 | th.mcslidedown.b | 0111 | 11 | uimm3 | 000 | 00 | ms1 | 000 | 00 | md |
| 114 | th.mcslidedown.h | 0111 | 11 | uimm3 | 000 | 01 | ms1 | 000 | 01 | md |
| 115 | th.mcslidedown.w | 0111 | 11 | uimm3 | 000 | 10 | ms1 | 000 | 10 | md |
| 116 | th.mcslidedown.d | 0111 | 11 | uimm3 | 000 | 11 | ms1 | 000 | 11 | md |
| 117 | th.mcslideup.b | 1000 | 11 | uimm3 | 000 | 00 | ms1 | 000 | 00 | md |
| 118 | th.mcslideup.h | 1000 | 11 | uimm3 | 000 | 01 | ms1 | 000 | 01 | md |
| 119 | th.mcslideup.w | 1000 | 11 | uimm3 | 000 | 10 | ms1 | 000 | 10 | md |
| 120 | th.mcslideup.d | 1000 | 11 | uimm3 | 000 | 11 | ms1 | 000 | 11 | md |

### 4i. Broadcast (func4=1001/1010)

| # | Mnemonic | func4 | uop | uimm3[25:23] | ms2 | s_size | ms1 | func3 | d_size | md |
|---|----------|-------|-----|--------------|-----|--------|-----|-------|--------|-----|
| 121 | th.mrbca.mv.i | 1001 | 11 | uimm3 | 000 | 00 | ms1 | 000 | 00 | md |
| 122 | th.mcbcab.mv.i | 1010 | 11 | uimm3 | 000 | 00 | ms1 | 000 | 00 | md |
| 123 | th.mcbcah.mv.i | 1010 | 11 | uimm3 | 000 | 01 | ms1 | 000 | 01 | md |
| 124 | th.mcbcaw.mv.i | 1010 | 11 | uimm3 | 000 | 10 | ms1 | 000 | 10 | md |
| 125 | th.mcbcad.mv.i | 1010 | 11 | uimm3 | 000 | 11 | ms1 | 000 | 11 | md |

**Total MISC: 35**

---

## 5. ELEMENT-WISE (func3=001)

Standard layout: `[31:28]func4 [27:26]uop [25:23]ctrl [22:20]ms2 [19:18]s_size [17:15]ms1 [14:12]func3=001 [11:10]d_size [9:7]md`

### 5a. FP Format Conversions (uop=00, func4=0000)

ctrl bit meanings: bit[25]=bf16/e5 select, bit[24]=low(0)/high(1), bit[23]=type/variant select.
Unary ops: ms2=000 (unused), only ms1 and md are operands.

| # | Mnemonic | func4 | uop | ctrl[25:23] | ms2 | s_size | ms1 | func3 | d_size | md |
|---|----------|-------|-----|-------------|-----|--------|-----|-------|--------|-----|
| 126 | th.mfcvtl.h.e4 | 0000 | 00 | 000 | 000 | 00 | ms1 | 001 | 01 | md |
| 127 | th.mfcvth.h.e4 | 0000 | 00 | 010 | 000 | 00 | ms1 | 001 | 01 | md |
| 128 | th.mfcvtl.h.e5 | 0000 | 00 | 001 | 000 | 00 | ms1 | 001 | 01 | md |
| 129 | th.mfcvth.h.e5 | 0000 | 00 | 011 | 000 | 00 | ms1 | 001 | 01 | md |
| 130 | th.mfcvtl.e4.h | 0000 | 00 | 000 | 000 | 01 | ms1 | 001 | 00 | md |
| 131 | th.mfcvth.e4.h | 0000 | 00 | 010 | 000 | 01 | ms1 | 001 | 00 | md |
| 132 | th.mfcvtl.e5.h | 0000 | 00 | 001 | 000 | 01 | ms1 | 001 | 00 | md |
| 133 | th.mfcvth.e5.h | 0000 | 00 | 011 | 000 | 01 | ms1 | 001 | 00 | md |
| 134 | th.mfcvtl.s.h | 0000 | 00 | 000 | 000 | 01 | ms1 | 001 | 10 | md |
| 135 | th.mfcvth.s.h | 0000 | 00 | 010 | 000 | 01 | ms1 | 001 | 10 | md |
| 136 | th.mfcvtl.s.bf16 | 0000 | 00 | 001 | 000 | 01 | ms1 | 001 | 10 | md |
| 137 | th.mfcvth.s.bf16 | 0000 | 00 | 011 | 000 | 01 | ms1 | 001 | 10 | md |
| 138 | th.mfcvtl.e4.s | 0000 | 00 | 000 | 000 | 10 | ms1 | 001 | 00 | md |
| 139 | th.mfcvth.e4.s | 0000 | 00 | 010 | 000 | 10 | ms1 | 001 | 00 | md |
| 140 | th.mfcvtl.e5.s | 0000 | 00 | 100 | 000 | 10 | ms1 | 001 | 00 | md |
| 141 | th.mfcvth.e5.s | 0000 | 00 | 110 | 000 | 10 | ms1 | 001 | 00 | md |
| 142 | th.mfcvtl.h.s | 0000 | 00 | 000 | 000 | 10 | ms1 | 001 | 01 | md |
| 143 | th.mfcvth.h.s | 0000 | 00 | 010 | 000 | 10 | ms1 | 001 | 01 | md |
| 144 | th.mfcvtl.bf16.s | 0000 | 00 | 100 | 000 | 10 | ms1 | 001 | 01 | md |
| 145 | th.mfcvth.bf16.s | 0000 | 00 | 110 | 000 | 10 | ms1 | 001 | 01 | md |
| 146 | th.mfcvt.tf32.s | 0000 | 00 | 110 | 000 | 10 | ms1 | 001 | 10 | md |
| 147 | th.mfcvt.s.tf32 | 0000 | 00 | 001 | 000 | 10 | ms1 | 001 | 10 | md |
| 148 | th.mfcvtl.d.s | 0000 | 00 | 000 | 000 | 10 | ms1 | 001 | 11 | md |
| 149 | th.mfcvth.d.s | 0000 | 00 | 010 | 000 | 10 | ms1 | 001 | 11 | md |
| 150 | th.mfcvtl.s.d | 0000 | 00 | 000 | 000 | 11 | ms1 | 001 | 10 | md |
| 151 | th.mfcvth.s.d | 0000 | 00 | 010 | 000 | 11 | ms1 | 001 | 10 | md |

**Count: 26**

### 5b. Float-Int Conversions (uop=00, func4=0001)

ctrl: bit[25]=direction (0=int→float, 1=float→int), bit[24]=low(0)/high(1), bit[23]=signed(1)/unsigned(0)

| # | Mnemonic | func4 | uop | ctrl[25:23] | ms2 | s_size | ms1 | func3 | d_size | md |
|---|----------|-------|-----|-------------|-----|--------|-----|-------|--------|-----|
| 152 | th.msfcvtl.h.b | 0001 | 00 | 001 | 000 | 00 | ms1 | 001 | 01 | md |
| 153 | th.msfcvth.h.b | 0001 | 00 | 011 | 000 | 00 | ms1 | 001 | 01 | md |
| 154 | th.mufcvtl.h.b | 0001 | 00 | 000 | 000 | 00 | ms1 | 001 | 01 | md |
| 155 | th.mufcvth.h.b | 0001 | 00 | 010 | 000 | 00 | ms1 | 001 | 01 | md |
| 156 | th.msfcvt.s.w | 0001 | 00 | 001 | 000 | 10 | ms1 | 001 | 10 | md |
| 157 | th.mufcvt.s.w | 0001 | 00 | 000 | 000 | 10 | ms1 | 001 | 10 | md |
| 158 | th.mfscvt.w.s | 0001 | 00 | 101 | 000 | 10 | ms1 | 001 | 10 | md |
| 159 | th.mfucvt.w.s | 0001 | 00 | 100 | 000 | 10 | ms1 | 001 | 10 | md |
| 160 | th.mfucvtl.b.h | 0001 | 00 | 100 | 000 | 01 | ms1 | 001 | 00 | md |
| 161 | th.mfucvth.b.h | 0001 | 00 | 110 | 000 | 01 | ms1 | 001 | 00 | md |
| 162 | th.mfscvtl.b.h | 0001 | 00 | 101 | 000 | 01 | ms1 | 001 | 00 | md |
| 163 | th.mfscvth.b.h | 0001 | 00 | 111 | 000 | 01 | ms1 | 001 | 00 | md |

**Count: 12**

### 5c. Fixed-Point N4Clip (uop=00, func4=0010-0101)

Binary ops: ms2 in bits[22:20], ms1 in bits[17:15].
- `.mm` variant: ctrl=111
- `.mv.i` variant: ctrl=uimm3 (row index into ms1)

All use s_size=10, d_size=10 (32-bit).

| # | Mnemonic | func4 | uop | ctrl[25:23] | ms2 | s_size | ms1 | func3 | d_size | md |
|---|----------|-------|-----|-------------|-----|--------|-----|-------|--------|-----|
| 164 | th.mn4clipl.w.mm | 0010 | 00 | 111 | ms2 | 10 | ms1 | 001 | 10 | md |
| 165 | th.mn4clipl.w.mv.i | 0010 | 00 | uimm3 | ms2 | 10 | ms1 | 001 | 10 | md |
| 166 | th.mn4cliph.w.mm | 0011 | 00 | 111 | ms2 | 10 | ms1 | 001 | 10 | md |
| 167 | th.mn4cliph.w.mv.i | 0011 | 00 | uimm3 | ms2 | 10 | ms1 | 001 | 10 | md |
| 168 | th.mn4cliplu.w.mm | 0100 | 00 | 111 | ms2 | 10 | ms1 | 001 | 10 | md |
| 169 | th.mn4cliplu.w.mv.i | 0100 | 00 | uimm3 | ms2 | 10 | ms1 | 001 | 10 | md |
| 170 | th.mn4cliphu.w.mm | 0101 | 00 | 111 | ms2 | 10 | ms1 | 001 | 10 | md |
| 171 | th.mn4cliphu.w.mv.i | 0101 | 00 | uimm3 | ms2 | 10 | ms1 | 001 | 10 | md |

**Count: 8**

### 5d. Int4/Int8 Pack Conversions (uop=00, func4=0110)

Unary ops: ms2=000 (unused).

| # | Mnemonic | func4 | uop | ctrl[25:23] | ms2 | s_size | ms1 | func3 | d_size | md |
|---|----------|-------|-----|-------------|-----|--------|-----|-------|--------|-----|
| 172 | th.mscvtl.b.p | 0110 | 00 | 001 | 000 | 00 | ms1 | 001 | 00 | md |
| 173 | th.mscvth.b.p | 0110 | 00 | 011 | 000 | 00 | ms1 | 001 | 00 | md |
| 174 | th.mucvtl.b.p | 0110 | 00 | 000 | 000 | 00 | ms1 | 001 | 00 | md |
| 175 | th.mucvth.b.p | 0110 | 00 | 010 | 000 | 00 | ms1 | 001 | 00 | md |

**Count: 4**

### 5e. Integer Element-Wise Arithmetic (uop=01, func3=001)

Binary ops: ms2 in bits[22:20], ms1 in bits[17:15].
- `.mm` variant: ctrl=111
- `.mv.i` variant: ctrl=uimm3 (row index into ms1)

All use s_size=10, d_size=10 (32-bit).

| # | Mnemonic | func4 | uop | ctrl[25:23] | ms2 | s_size | ms1 | func3 | d_size | md |
|---|----------|-------|-----|-------------|-----|--------|-----|-------|--------|-----|
| 176 | th.madd.w.mm | 0000 | 01 | 111 | ms2 | 10 | ms1 | 001 | 10 | md |
| 177 | th.madd.w.mv.i | 0000 | 01 | uimm3 | ms2 | 10 | ms1 | 001 | 10 | md |
| 178 | th.msub.w.mm | 0001 | 01 | 111 | ms2 | 10 | ms1 | 001 | 10 | md |
| 179 | th.msub.w.mv.i | 0001 | 01 | uimm3 | ms2 | 10 | ms1 | 001 | 10 | md |
| 180 | th.mmul.w.mm | 0010 | 01 | 111 | ms2 | 10 | ms1 | 001 | 10 | md |
| 181 | th.mmul.w.mv.i | 0010 | 01 | uimm3 | ms2 | 10 | ms1 | 001 | 10 | md |
| 182 | th.mmulh.w.mm | 0011 | 01 | 111 | ms2 | 10 | ms1 | 001 | 10 | md |
| 183 | th.mmulh.w.mv.i | 0011 | 01 | uimm3 | ms2 | 10 | ms1 | 001 | 10 | md |
| 184 | th.mmax.w.mm | 0100 | 01 | 111 | ms2 | 10 | ms1 | 001 | 10 | md |
| 185 | th.mmax.w.mv.i | 0100 | 01 | uimm3 | ms2 | 10 | ms1 | 001 | 10 | md |
| 186 | th.mumax.w.mm | 0101 | 01 | 111 | ms2 | 10 | ms1 | 001 | 10 | md |
| 187 | th.mumax.w.mv.i | 0101 | 01 | uimm3 | ms2 | 10 | ms1 | 001 | 10 | md |
| 188 | th.mmin.w.mm | 0110 | 01 | 111 | ms2 | 10 | ms1 | 001 | 10 | md |
| 189 | th.mmin.w.mv.i | 0110 | 01 | uimm3 | ms2 | 10 | ms1 | 001 | 10 | md |
| 190 | th.mumin.w.mm | 0111 | 01 | 111 | ms2 | 10 | ms1 | 001 | 10 | md |
| 191 | th.mumin.w.mv.i | 0111 | 01 | uimm3 | ms2 | 10 | ms1 | 001 | 10 | md |
| 192 | th.msrl.w.mm | 1000 | 01 | 111 | ms2 | 10 | ms1 | 001 | 10 | md |
| 193 | th.msrl.w.mv.i | 1000 | 01 | uimm3 | ms2 | 10 | ms1 | 001 | 10 | md |
| 194 | th.msll.w.mm | 1001 | 01 | 111 | ms2 | 10 | ms1 | 001 | 10 | md |
| 195 | th.msll.w.mv.i | 1001 | 01 | uimm3 | ms2 | 10 | ms1 | 001 | 10 | md |
| 196 | th.msra.w.mm | 1010 | 01 | 111 | ms2 | 10 | ms1 | 001 | 10 | md |
| 197 | th.msra.w.mv.i | 1010 | 01 | uimm3 | ms2 | 10 | ms1 | 001 | 10 | md |

**Count: 22**

### 5f. FP Element-Wise Arithmetic (uop=10, func3=001)

Binary ops: ms2 in bits[22:20], ms1 in bits[17:15].
- `.mm` variant: ctrl=111
- `.mv.i` variant: ctrl=uimm3

Each op has .h (s_size=01, d_size=01), .s (10, 10), .d (11, 11) variants.

| # | Mnemonic | func4 | uop | ctrl[25:23] | ms2 | s_size | ms1 | func3 | d_size | md |
|---|----------|-------|-----|-------------|-----|--------|-----|-------|--------|-----|
| 198 | th.mfadd.h.mm | 0000 | 10 | 111 | ms2 | 01 | ms1 | 001 | 01 | md |
| 199 | th.mfadd.h.mv.i | 0000 | 10 | uimm3 | ms2 | 01 | ms1 | 001 | 01 | md |
| 200 | th.mfadd.s.mm | 0000 | 10 | 111 | ms2 | 10 | ms1 | 001 | 10 | md |
| 201 | th.mfadd.s.mv.i | 0000 | 10 | uimm3 | ms2 | 10 | ms1 | 001 | 10 | md |
| 202 | th.mfadd.d.mm | 0000 | 10 | 111 | ms2 | 11 | ms1 | 001 | 11 | md |
| 203 | th.mfadd.d.mv.i | 0000 | 10 | uimm3 | ms2 | 11 | ms1 | 001 | 11 | md |
| 204 | th.mfsub.h.mm | 0001 | 10 | 111 | ms2 | 01 | ms1 | 001 | 01 | md |
| 205 | th.mfsub.h.mv.i | 0001 | 10 | uimm3 | ms2 | 01 | ms1 | 001 | 01 | md |
| 206 | th.mfsub.s.mm | 0001 | 10 | 111 | ms2 | 10 | ms1 | 001 | 10 | md |
| 207 | th.mfsub.s.mv.i | 0001 | 10 | uimm3 | ms2 | 10 | ms1 | 001 | 10 | md |
| 208 | th.mfsub.d.mm | 0001 | 10 | 111 | ms2 | 11 | ms1 | 001 | 11 | md |
| 209 | th.mfsub.d.mv.i | 0001 | 10 | uimm3 | ms2 | 11 | ms1 | 001 | 11 | md |
| 210 | th.mfmul.h.mm | 0010 | 10 | 111 | ms2 | 01 | ms1 | 001 | 01 | md |
| 211 | th.mfmul.h.mv.i | 0010 | 10 | uimm3 | ms2 | 01 | ms1 | 001 | 01 | md |
| 212 | th.mfmul.s.mm | 0010 | 10 | 111 | ms2 | 10 | ms1 | 001 | 10 | md |
| 213 | th.mfmul.s.mv.i | 0010 | 10 | uimm3 | ms2 | 10 | ms1 | 001 | 10 | md |
| 214 | th.mfmul.d.mm | 0010 | 10 | 111 | ms2 | 11 | ms1 | 001 | 11 | md |
| 215 | th.mfmul.d.mv.i | 0010 | 10 | uimm3 | ms2 | 11 | ms1 | 001 | 11 | md |
| 216 | th.mfmax.h.mm | 0011 | 10 | 111 | ms2 | 01 | ms1 | 001 | 01 | md |
| 217 | th.mfmax.h.mv.i | 0011 | 10 | uimm3 | ms2 | 01 | ms1 | 001 | 01 | md |
| 218 | th.mfmax.s.mm | 0011 | 10 | 111 | ms2 | 10 | ms1 | 001 | 10 | md |
| 219 | th.mfmax.s.mv.i | 0011 | 10 | uimm3 | ms2 | 10 | ms1 | 001 | 10 | md |
| 220 | th.mfmax.d.mm | 0011 | 10 | 111 | ms2 | 11 | ms1 | 001 | 11 | md |
| 221 | th.mfmax.d.mv.i | 0011 | 10 | uimm3 | ms2 | 11 | ms1 | 001 | 11 | md |
| 222 | th.mfmin.h.mm | 0100 | 10 | 111 | ms2 | 01 | ms1 | 001 | 01 | md |
| 223 | th.mfmin.h.mv.i | 0100 | 10 | uimm3 | ms2 | 01 | ms1 | 001 | 01 | md |
| 224 | th.mfmin.s.mm | 0100 | 10 | 111 | ms2 | 10 | ms1 | 001 | 10 | md |
| 225 | th.mfmin.s.mv.i | 0100 | 10 | uimm3 | ms2 | 10 | ms1 | 001 | 10 | md |
| 226 | th.mfmin.d.mm | 0100 | 10 | 111 | ms2 | 11 | ms1 | 001 | 11 | md |
| 227 | th.mfmin.d.mv.i | 0100 | 10 | uimm3 | ms2 | 11 | ms1 | 001 | 11 | md |

**IMPORTANT NOTE on mfmin naming**: instruction_list.adoc shows "mfmin.s" with s_size=01 (half-precision encoding) and "mfmin.h" with s_size=10 (single-precision encoding). The **names are swapped** in the spec table. The correct mapping by encoding:
- func4=0100, s_size=01, d_size=01 → this is half-precision → should be named **mfmin.h**
- func4=0100, s_size=10, d_size=10 → this is single-precision → should be named **mfmin.s**

In the table above I use the CORRECT names (matching the encoding, not the spec text).

**Count: 30**

---

## Grand Total

| Category | Count |
|----------|-------|
| Configuration | 7 |
| Load/Store | 56 |
| Matrix Multiply | 27 |
| MISC | 35 |
| EW FP Convert (5a) | 26 |
| EW Float-Int Convert (5b) | 12 |
| EW N4Clip (5c) | 8 |
| EW Pack Convert (5d) | 4 |
| EW Int Arithmetic (5e) | 22 |
| EW FP Arithmetic (5f) | 30 |
| **TOTAL** | **227** |

---

## Quick-Reference: Encoding Discriminator Chain

```
opcode = 0101011 (OPC_CUSTOM_1)
  ├─ func3=000
  │   ├─ uop=00 → CONFIG
  │   ├─ uop=01 → LOAD/STORE (bit[25]=ls)
  │   ├─ uop=10 → MATMUL (func4 + size_sup + s_size + d_size)
  │   └─ uop=11 → MISC (func4 determines sub-type)
  └─ func3=001
      ├─ uop=00 → EW Conversions (func4=0000 FP-FP, 0001 FP-Int, 0010-0101 N4Clip, 0110 Pack)
      ├─ uop=01 → EW Integer Arithmetic (func4=operation, ctrl=111 for .mm)
      └─ uop=10 → EW FP Arithmetic (func4=operation, ctrl=111 for .mm)
```

## Key Verification Rules

1. **Config**: bits[11:7] MUST be 00000 (no rd), bit[25] selects imm(0) vs reg(1)
2. **Load/Store**: bit[25] is ls (load=0/store=1), NOT stride. func4 selects A/B/C/M/Ate/Bte/Cte
3. **Matmul**: uop MUST be 10 (not 01). func4: 0000=FP, 0001=Int, 0010=Bypass
4. **MISC slides/broadcasts**: uimm3 is in bits[25:23], ms2=000 (unused)
5. **MISC mmov.x.m**: bits[19:15] and [11:7] are 5-bit GPR fields (not split into s_size/ms1 or d_size/md)
6. **MISC mmov.m.x/mdup.m.x**: bits[24:20] and [19:15] are 5-bit GPR fields
7. **EW .mm**: ctrl MUST be 111 (not 000)
8. **EW .mv.i**: ctrl = uimm3 row index, ms2 = matrix operand, ms1 = indexed matrix
9. **EW conversions**: bit[24] = low(0)/high(1), bit[23] = type variant
10. **mzero variants**: ALL share func4=0000, distinguished by imm3 in bits[25:23] (000/001/011/111)
