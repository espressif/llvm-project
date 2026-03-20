# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-xtheadmatrix %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

# Configuration instructions
th.msettilemi 16
# CHECK-INST: th.msettilemi 16
# CHECK-ENCODING: [0x2b,0x00,0x08,0x20]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msettilem a2
# CHECK-INST: th.msettilem a2
# CHECK-ENCODING: [0x2b,0x00,0x06,0x22]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msettileki 16
# CHECK-INST: th.msettileki 16
# CHECK-ENCODING: [0x2b,0x00,0x08,0x10]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msettilek a4
# CHECK-INST: th.msettilek a4
# CHECK-ENCODING: [0x2b,0x00,0x07,0x12]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msettileni 16
# CHECK-INST: th.msettileni 16
# CHECK-ENCODING: [0x2b,0x00,0x08,0x30]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msettilen s0
# CHECK-INST: th.msettilen s0
# CHECK-ENCODING: [0x2b,0x00,0x04,0x32]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mrelease
# CHECK-INST: th.mrelease
# CHECK-ENCODING: [0x2b,0x00,0x00,0x00]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}


# Load instructions
th.mlae8 acc3, (t0), t1
# CHECK-INST: th.mlae8 acc3, (t0), t1
# CHECK-ENCODING: [0xab,0x83,0x62,0x04]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mlae16 tr0, (t1), t2
# CHECK-INST: th.mlae16 tr0, (t1), t2
# CHECK-ENCODING: [0x2b,0x04,0x73,0x04]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mlae32 tr1, (t2), t3
# CHECK-INST: th.mlae32 tr1, (t2), t3
# CHECK-ENCODING: [0xab,0x88,0xc3,0x05]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mlae64 tr2, (t3), a0
# CHECK-INST: th.mlae64 tr2, (t3), a0
# CHECK-ENCODING: [0x2b,0x0d,0xae,0x04]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mlbe8 tr3, (a0), a1
# CHECK-INST: th.mlbe8 tr3, (a0), a1
# CHECK-ENCODING: [0xab,0x01,0xb5,0x14]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mlbe16 acc0, (a1), a2
# CHECK-INST: th.mlbe16 acc0, (a1), a2
# CHECK-ENCODING: [0x2b,0x86,0xc5,0x14]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mlbe32 acc1, (a2), a3
# CHECK-INST: th.mlbe32 acc1, (a2), a3
# CHECK-ENCODING: [0xab,0x0a,0xd6,0x14]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mlbe64 acc2, (a3), a4
# CHECK-INST: th.mlbe64 acc2, (a3), a4
# CHECK-ENCODING: [0x2b,0x8f,0xe6,0x14]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mlce8 acc3, (a4), a5
# CHECK-INST: th.mlce8 acc3, (a4), a5
# CHECK-ENCODING: [0xab,0x03,0xf7,0x24]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mlce16 tr0, (a5), s0
# CHECK-INST: th.mlce16 tr0, (a5), s0
# CHECK-ENCODING: [0x2b,0x84,0x87,0x24]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mlce32 tr1, (s0), s1
# CHECK-INST: th.mlce32 tr1, (s0), s1
# CHECK-ENCODING: [0xab,0x08,0x94,0x24]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mlce64 tr2, (s1), t0
# CHECK-INST: th.mlce64 tr2, (s1), t0
# CHECK-ENCODING: [0x2b,0x8d,0x54,0x24]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mlate8 tr3, (t0), t1
# CHECK-INST: th.mlate8 tr3, (t0), t1
# CHECK-ENCODING: [0xab,0x81,0x62,0x44]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mlate16 acc0, (t1), t2
# CHECK-INST: th.mlate16 acc0, (t1), t2
# CHECK-ENCODING: [0x2b,0x06,0x73,0x44]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mlate32 acc1, (t2), t3
# CHECK-INST: th.mlate32 acc1, (t2), t3
# CHECK-ENCODING: [0xab,0x8a,0xc3,0x45]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mlate64 acc2, (t3), a0
# CHECK-INST: th.mlate64 acc2, (t3), a0
# CHECK-ENCODING: [0x2b,0x0f,0xae,0x44]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mlbte8 acc3, (a0), a1
# CHECK-INST: th.mlbte8 acc3, (a0), a1
# CHECK-ENCODING: [0xab,0x03,0xb5,0x54]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mlbte16 tr0, (a1), a2
# CHECK-INST: th.mlbte16 tr0, (a1), a2
# CHECK-ENCODING: [0x2b,0x84,0xc5,0x54]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mlbte32 tr1, (a2), a3
# CHECK-INST: th.mlbte32 tr1, (a2), a3
# CHECK-ENCODING: [0xab,0x08,0xd6,0x54]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mlbte64 tr2, (a3), a4
# CHECK-INST: th.mlbte64 tr2, (a3), a4
# CHECK-ENCODING: [0x2b,0x8d,0xe6,0x54]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mlcte8 tr3, (a4), a5
# CHECK-INST: th.mlcte8 tr3, (a4), a5
# CHECK-ENCODING: [0xab,0x01,0xf7,0x64]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mlcte16 acc0, (a5), s0
# CHECK-INST: th.mlcte16 acc0, (a5), s0
# CHECK-ENCODING: [0x2b,0x86,0x87,0x64]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mlcte32 acc1, (s0), s1
# CHECK-INST: th.mlcte32 acc1, (s0), s1
# CHECK-ENCODING: [0xab,0x0a,0x94,0x64]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mlcte64 acc2, (s1), t0
# CHECK-INST: th.mlcte64 acc2, (s1), t0
# CHECK-ENCODING: [0x2b,0x8f,0x54,0x64]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mlme8 acc3, (t0), t1
# CHECK-INST: th.mlme8 acc3, (t0), t1
# CHECK-ENCODING: [0xab,0x83,0x62,0x34]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mlme16 tr0, (t1), t2
# CHECK-INST: th.mlme16 tr0, (t1), t2
# CHECK-ENCODING: [0x2b,0x04,0x73,0x34]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mlme32 tr1, (t2), t3
# CHECK-INST: th.mlme32 tr1, (t2), t3
# CHECK-ENCODING: [0xab,0x88,0xc3,0x35]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mlme64 tr2, (t3), t4
# CHECK-INST: th.mlme64 tr2, (t3), t4
# CHECK-ENCODING: [0x2b,0x0d,0xde,0x35]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}


# Store instructions
th.msae8 tr3, (a0), a1
# CHECK-INST: th.msae8 tr3, (a0), a1
# CHECK-ENCODING: [0xab,0x01,0xb5,0x06]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msae16 acc0, (a1), a2
# CHECK-INST: th.msae16 acc0, (a1), a2
# CHECK-ENCODING: [0x2b,0x86,0xc5,0x06]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msae32 acc1, (a2), a3
# CHECK-INST: th.msae32 acc1, (a2), a3
# CHECK-ENCODING: [0xab,0x0a,0xd6,0x06]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msae64 acc2, (a3), a4
# CHECK-INST: th.msae64 acc2, (a3), a4
# CHECK-ENCODING: [0x2b,0x8f,0xe6,0x06]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msbe8 acc3, (a4), a5
# CHECK-INST: th.msbe8 acc3, (a4), a5
# CHECK-ENCODING: [0xab,0x03,0xf7,0x16]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msbe16 tr0, (a5), s0
# CHECK-INST: th.msbe16 tr0, (a5), s0
# CHECK-ENCODING: [0x2b,0x84,0x87,0x16]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msbe32 tr1, (s0), s1
# CHECK-INST: th.msbe32 tr1, (s0), s1
# CHECK-ENCODING: [0xab,0x08,0x94,0x16]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msbe64 tr2, (s1), t0
# CHECK-INST: th.msbe64 tr2, (s1), t0
# CHECK-ENCODING: [0x2b,0x8d,0x54,0x16]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msce8 tr3, (t0), t1
# CHECK-INST: th.msce8 tr3, (t0), t1
# CHECK-ENCODING: [0xab,0x81,0x62,0x26]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msce16 acc0, (t1), t2
# CHECK-INST: th.msce16 acc0, (t1), t2
# CHECK-ENCODING: [0x2b,0x06,0x73,0x26]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msce32 acc1, (t2), t3
# CHECK-INST: th.msce32 acc1, (t2), t3
# CHECK-ENCODING: [0xab,0x8a,0xc3,0x27]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msce64 acc2, (t3), a0
# CHECK-INST: th.msce64 acc2, (t3), a0
# CHECK-ENCODING: [0x2b,0x0f,0xae,0x26]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msate8 acc3, (a0), a1
# CHECK-INST: th.msate8 acc3, (a0), a1
# CHECK-ENCODING: [0xab,0x03,0xb5,0x46]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msate16 tr0, (a1), a2
# CHECK-INST: th.msate16 tr0, (a1), a2
# CHECK-ENCODING: [0x2b,0x84,0xc5,0x46]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msate32 tr1, (a2), a3
# CHECK-INST: th.msate32 tr1, (a2), a3
# CHECK-ENCODING: [0xab,0x08,0xd6,0x46]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msate64 tr2, (a3), a4
# CHECK-INST: th.msate64 tr2, (a3), a4
# CHECK-ENCODING: [0x2b,0x8d,0xe6,0x46]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msbte8 tr3, (a4), a5
# CHECK-INST: th.msbte8 tr3, (a4), a5
# CHECK-ENCODING: [0xab,0x01,0xf7,0x56]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msbte16 acc0, (a5), s0
# CHECK-INST: th.msbte16 acc0, (a5), s0
# CHECK-ENCODING: [0x2b,0x86,0x87,0x56]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msbte32 acc1, (s0), s1
# CHECK-INST: th.msbte32 acc1, (s0), s1
# CHECK-ENCODING: [0xab,0x0a,0x94,0x56]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msbte64 acc2, (s1), t0
# CHECK-INST: th.msbte64 acc2, (s1), t0
# CHECK-ENCODING: [0x2b,0x8f,0x54,0x56]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mscte8 acc3, (t0), t1
# CHECK-INST: th.mscte8 acc3, (t0), t1
# CHECK-ENCODING: [0xab,0x83,0x62,0x66]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mscte16 tr0, (t1), t2
# CHECK-INST: th.mscte16 tr0, (t1), t2
# CHECK-ENCODING: [0x2b,0x04,0x73,0x66]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mscte32 tr1, (t2), t3
# CHECK-INST: th.mscte32 tr1, (t2), t3
# CHECK-ENCODING: [0xab,0x88,0xc3,0x67]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mscte64 tr2, (t3), a0
# CHECK-INST: th.mscte64 tr2, (t3), a0
# CHECK-ENCODING: [0x2b,0x0d,0xae,0x66]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msme8 tr3, (a0), a1
# CHECK-INST: th.msme8 tr3, (a0), a1
# CHECK-ENCODING: [0xab,0x01,0xb5,0x36]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msme16 acc0, (a1), a2
# CHECK-INST: th.msme16 acc0, (a1), a2
# CHECK-ENCODING: [0x2b,0x86,0xc5,0x36]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msme32 acc1, (a2), a3
# CHECK-INST: th.msme32 acc1, (a2), a3
# CHECK-ENCODING: [0xab,0x0a,0xd6,0x36]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msme64 acc2, (a3), a4
# CHECK-INST: th.msme64 acc2, (a3), a4
# CHECK-ENCODING: [0x2b,0x8f,0xe6,0x36]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}


# Matrix multiply instructions
th.mfmacc.h.e5 acc3, tr1, tr0
# CHECK-INST: th.mfmacc.h.e5 acc3, tr1, tr0
# CHECK-ENCODING: [0xab,0x07,0x10,0x08]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfmacc.h.e4 tr0, tr2, tr1
# CHECK-INST: th.mfmacc.h.e4 tr0, tr2, tr1
# CHECK-ENCODING: [0x2b,0x84,0xa0,0x08]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfmacc.bf16.e5 tr1, tr3, tr2
# CHECK-INST: th.mfmacc.bf16.e5 tr1, tr3, tr2
# CHECK-ENCODING: [0xab,0x04,0x31,0x0a]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfmacc.bf16.e4 tr2, acc0, tr3
# CHECK-INST: th.mfmacc.bf16.e4 tr2, acc0, tr3
# CHECK-ENCODING: [0x2b,0x85,0xc1,0x0a]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfmacc.s.e5 tr3, acc1, acc0
# CHECK-INST: th.mfmacc.s.e5 tr3, acc1, acc0
# CHECK-ENCODING: [0xab,0x09,0x52,0x08]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfmacc.s.e4 acc0, acc2, acc1
# CHECK-INST: th.mfmacc.s.e4 acc0, acc2, acc1
# CHECK-ENCODING: [0x2b,0x8a,0xe2,0x08]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfmacc.h acc1, acc3, acc2
# CHECK-INST: th.mfmacc.h acc1, acc3, acc2
# CHECK-ENCODING: [0xab,0x06,0x77,0x08]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfmacc.s.h acc2, tr0, acc3
# CHECK-INST: th.mfmacc.s.h acc2, tr0, acc3
# CHECK-ENCODING: [0x2b,0x8b,0x07,0x08]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfmacc.s.bf16 acc3, tr1, tr0
# CHECK-INST: th.mfmacc.s.bf16 acc3, tr1, tr0
# CHECK-ENCODING: [0xab,0x0b,0x94,0x08]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfmacc.s tr0, tr2, tr1
# CHECK-INST: th.mfmacc.s tr0, tr2, tr1
# CHECK-ENCODING: [0x2b,0x88,0x28,0x08]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfmacc.s.tf32 tr1, tr3, tr2
# CHECK-INST: th.mfmacc.s.tf32 tr1, tr3, tr2
# CHECK-ENCODING: [0xab,0x08,0xb9,0x08]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfmacc.d.s tr2, acc0, tr3
# CHECK-INST: th.mfmacc.d.s tr2, acc0, tr3
# CHECK-ENCODING: [0x2b,0x8d,0x49,0x08]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfmacc.d tr3, acc1, acc0
# CHECK-INST: th.mfmacc.d tr3, acc1, acc0
# CHECK-ENCODING: [0xab,0x0d,0x5e,0x08]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mmacc.w.b acc0, acc2, acc1
# CHECK-INST: th.mmacc.w.b acc0, acc2, acc1
# CHECK-ENCODING: [0x2b,0x8a,0xe2,0x19]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mmaccu.w.b acc1, acc3, acc2
# CHECK-INST: th.mmaccu.w.b acc1, acc3, acc2
# CHECK-ENCODING: [0xab,0x0a,0x73,0x18]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mmaccsu.w.b acc2, tr0, acc3
# CHECK-INST: th.mmaccsu.w.b acc2, tr0, acc3
# CHECK-ENCODING: [0x2b,0x8b,0x03,0x19]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mmaccus.w.b acc3, tr1, tr0
# CHECK-INST: th.mmaccus.w.b acc3, tr1, tr0
# CHECK-ENCODING: [0xab,0x0b,0x90,0x18]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mmacc.w.bp tr0, tr2, tr1
# CHECK-INST: th.mmacc.w.bp tr0, tr2, tr1
# CHECK-ENCODING: [0x2b,0x88,0xa0,0x29]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mmaccu.w.bp tr1, tr3, tr2
# CHECK-INST: th.mmaccu.w.bp tr1, tr3, tr2
# CHECK-ENCODING: [0xab,0x08,0x31,0x28]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mmacc.d.h tr2, acc0, tr3
# CHECK-INST: th.mmacc.d.h tr2, acc0, tr3
# CHECK-ENCODING: [0x2b,0x8d,0xc5,0x19]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mmaccu.d.h tr3, acc1, acc0
# CHECK-INST: th.mmaccu.d.h tr3, acc1, acc0
# CHECK-ENCODING: [0xab,0x0d,0x56,0x18]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mmaccsu.d.h acc0, acc2, acc1
# CHECK-INST: th.mmaccsu.d.h acc0, acc2, acc1
# CHECK-ENCODING: [0x2b,0x8e,0x66,0x19]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mmaccus.d.h acc1, acc3, acc2
# CHECK-INST: th.mmaccus.d.h acc1, acc3, acc2
# CHECK-ENCODING: [0xab,0x0e,0xf7,0x18]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.pmmacc.w.b acc2, tr0, acc3
# CHECK-INST: th.pmmacc.w.b acc2, tr0, acc3
# CHECK-ENCODING: [0x2b,0x8b,0x83,0x1b]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.pmmaccu.w.b acc3, tr1, tr0
# CHECK-INST: th.pmmaccu.w.b acc3, tr1, tr0
# CHECK-ENCODING: [0xab,0x0b,0x10,0x1a]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.pmmaccsu.w.b tr0, tr2, tr1
# CHECK-INST: th.pmmaccsu.w.b tr0, tr2, tr1
# CHECK-ENCODING: [0x2b,0x88,0x20,0x1b]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.pmmaccus.w.b tr1, tr3, tr2
# CHECK-INST: th.pmmaccus.w.b tr1, tr3, tr2
# CHECK-ENCODING: [0xab,0x08,0xb1,0x1a]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}


# Misc instructions
th.mzero tr2
# CHECK-INST: th.mzero tr2
# CHECK-ENCODING: [0x2b,0x01,0x00,0x0c]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mzero2r tr3
# CHECK-INST: th.mzero2r tr3
# CHECK-ENCODING: [0xab,0x01,0x80,0x0c]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mzero4r acc0
# CHECK-INST: th.mzero4r acc0
# CHECK-ENCODING: [0x2b,0x02,0x80,0x0d]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mzero8r acc1
# CHECK-INST: th.mzero8r acc1
# CHECK-ENCODING: [0xab,0x02,0x80,0x0f]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mmov.mm acc2, acc3
# CHECK-INST: th.mmov.mm acc2, acc3
# CHECK-ENCODING: [0x2b,0x83,0x03,0x1c]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mmovb.m.x acc3, a1, a0
# CHECK-INST: th.mmovb.m.x acc3, a1, a0
# CHECK-ENCODING: [0xab,0x03,0xb5,0x3e]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mmovh.m.x tr0, a2, a1
# CHECK-INST: th.mmovh.m.x tr0, a2, a1
# CHECK-ENCODING: [0x2b,0x84,0xc5,0x3e]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mmovw.m.x tr1, a3, a2
# CHECK-INST: th.mmovw.m.x tr1, a3, a2
# CHECK-ENCODING: [0xab,0x08,0xd6,0x3e]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mmovd.m.x tr2, a4, a3
# CHECK-INST: th.mmovd.m.x tr2, a4, a3
# CHECK-ENCODING: [0x2b,0x8d,0xe6,0x3e]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mmovb.x.m a3, acc1, a4
# CHECK-INST: th.mmovb.x.m a3, acc1, a4
# CHECK-ENCODING: [0xab,0x06,0x57,0x2c]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mmovh.x.m a4, acc2, a5
# CHECK-INST: th.mmovh.x.m a4, acc2, a5
# CHECK-ENCODING: [0x2b,0x87,0xe7,0x2c]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mmovw.x.m a5, acc3, s0
# CHECK-INST: th.mmovw.x.m a5, acc3, s0
# CHECK-ENCODING: [0xab,0x07,0x74,0x2d]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mmovd.x.m s0, tr0, s1
# CHECK-INST: th.mmovd.x.m s0, tr0, s1
# CHECK-ENCODING: [0x2b,0x84,0x84,0x2d]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mdupb.m.x acc3, t1
# CHECK-INST: th.mdupb.m.x acc3, t1
# CHECK-ENCODING: [0xab,0x03,0x60,0x3c]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mduph.m.x tr0, t2
# CHECK-INST: th.mduph.m.x tr0, t2
# CHECK-ENCODING: [0x2b,0x04,0x70,0x3c]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mdupw.m.x tr1, t3
# CHECK-INST: th.mdupw.m.x tr1, t3
# CHECK-ENCODING: [0xab,0x08,0xc0,0x3d]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mdupd.m.x tr2, a0
# CHECK-INST: th.mdupd.m.x tr2, a0
# CHECK-ENCODING: [0x2b,0x0d,0xa0,0x3c]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mpack tr3, acc1, acc0
# CHECK-INST: th.mpack tr3, acc1, acc0
# CHECK-ENCODING: [0xab,0x01,0x52,0x4c]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mpackhl acc0, acc2, acc1
# CHECK-INST: th.mpackhl acc0, acc2, acc1
# CHECK-ENCODING: [0x2b,0x82,0x62,0x4d]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mpackhh acc1, acc3, acc2
# CHECK-INST: th.mpackhh acc1, acc3, acc2
# CHECK-ENCODING: [0xab,0x02,0xf3,0x4d]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mrslideup acc2, acc3, 6
# CHECK-INST: th.mrslideup acc2, acc3, 6
# CHECK-ENCODING: [0x2b,0x83,0x03,0x6f]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mrslidedown acc3, tr0, 7
# CHECK-INST: th.mrslidedown acc3, tr0, 7
# CHECK-ENCODING: [0xab,0x03,0x80,0x5f]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mcslideup.b tr0, tr1, 0
# CHECK-INST: th.mcslideup.b tr0, tr1, 0
# CHECK-ENCODING: [0x2b,0x80,0x00,0x8c]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mcslideup.h tr1, tr2, 1
# CHECK-INST: th.mcslideup.h tr1, tr2, 1
# CHECK-ENCODING: [0xab,0x04,0x85,0x8c]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mcslideup.w tr2, tr3, 2
# CHECK-INST: th.mcslideup.w tr2, tr3, 2
# CHECK-ENCODING: [0x2b,0x89,0x09,0x8d]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mcslideup.d tr3, acc0, 3
# CHECK-INST: th.mcslideup.d tr3, acc0, 3
# CHECK-ENCODING: [0xab,0x0d,0x8e,0x8d]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mcslidedown.b acc0, acc1, 4
# CHECK-INST: th.mcslidedown.b acc0, acc1, 4
# CHECK-ENCODING: [0x2b,0x82,0x02,0x7e]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mcslidedown.h acc1, acc2, 5
# CHECK-INST: th.mcslidedown.h acc1, acc2, 5
# CHECK-ENCODING: [0xab,0x06,0x87,0x7e]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mcslidedown.w acc2, acc3, 6
# CHECK-INST: th.mcslidedown.w acc2, acc3, 6
# CHECK-ENCODING: [0x2b,0x8b,0x0b,0x7f]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mcslidedown.d acc3, tr0, 7
# CHECK-INST: th.mcslidedown.d acc3, tr0, 7
# CHECK-ENCODING: [0xab,0x0f,0x8c,0x7f]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mrbca.mv.i tr0, tr1, 0
# CHECK-INST: th.mrbca.mv.i tr0, tr1, 0
# CHECK-ENCODING: [0x2b,0x80,0x00,0x9c]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mcbcab.mv.i tr1, tr2, 1
# CHECK-INST: th.mcbcab.mv.i tr1, tr2, 1
# CHECK-ENCODING: [0xab,0x00,0x81,0xac]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mcbcah.mv.i tr2, tr3, 2
# CHECK-INST: th.mcbcah.mv.i tr2, tr3, 2
# CHECK-ENCODING: [0x2b,0x85,0x05,0xad]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mcbcaw.mv.i tr3, acc0, 3
# CHECK-INST: th.mcbcaw.mv.i tr3, acc0, 3
# CHECK-ENCODING: [0xab,0x09,0x8a,0xad]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mcbcad.mv.i acc0, acc1, 4
# CHECK-INST: th.mcbcad.mv.i acc0, acc1, 4
# CHECK-ENCODING: [0x2b,0x8e,0x0e,0xae]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}


# Integer element-wise arithmetic
th.madd.w.mm acc1, acc3, acc2
# CHECK-INST: th.madd.w.mm acc1, acc3, acc2
# CHECK-ENCODING: [0xab,0x1a,0xfb,0x07]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.madd.w.mv.i acc2, tr0, acc3, 6
# CHECK-INST: th.madd.w.mv.i acc2, tr0, acc3, 6
# CHECK-ENCODING: [0x2b,0x9b,0x0b,0x07]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msub.w.mm acc3, tr1, tr0
# CHECK-INST: th.msub.w.mm acc3, tr1, tr0
# CHECK-ENCODING: [0xab,0x1b,0x98,0x17]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msub.w.mv.i tr0, tr2, tr1, 0
# CHECK-INST: th.msub.w.mv.i tr0, tr2, tr1, 0
# CHECK-ENCODING: [0x2b,0x98,0x28,0x14]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mmul.w.mm tr1, tr3, tr2
# CHECK-INST: th.mmul.w.mm tr1, tr3, tr2
# CHECK-ENCODING: [0xab,0x18,0xb9,0x27]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mmul.w.mv.i tr2, acc0, tr3, 2
# CHECK-INST: th.mmul.w.mv.i tr2, acc0, tr3, 2
# CHECK-ENCODING: [0x2b,0x99,0x49,0x25]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mmulh.w.mm tr3, acc1, acc0
# CHECK-INST: th.mmulh.w.mm tr3, acc1, acc0
# CHECK-ENCODING: [0xab,0x19,0xda,0x37]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mmulh.w.mv.i acc0, acc2, acc1, 4
# CHECK-INST: th.mmulh.w.mv.i acc0, acc2, acc1, 4
# CHECK-ENCODING: [0x2b,0x9a,0x6a,0x36]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mmax.w.mm acc1, acc3, acc2
# CHECK-INST: th.mmax.w.mm acc1, acc3, acc2
# CHECK-ENCODING: [0xab,0x1a,0xfb,0x47]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mmax.w.mv.i acc2, tr0, acc3, 6
# CHECK-INST: th.mmax.w.mv.i acc2, tr0, acc3, 6
# CHECK-ENCODING: [0x2b,0x9b,0x0b,0x47]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mumax.w.mm acc3, tr1, tr0
# CHECK-INST: th.mumax.w.mm acc3, tr1, tr0
# CHECK-ENCODING: [0xab,0x1b,0x98,0x57]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mumax.w.mv.i tr0, tr2, tr1, 0
# CHECK-INST: th.mumax.w.mv.i tr0, tr2, tr1, 0
# CHECK-ENCODING: [0x2b,0x98,0x28,0x54]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mmin.w.mm tr1, tr3, tr2
# CHECK-INST: th.mmin.w.mm tr1, tr3, tr2
# CHECK-ENCODING: [0xab,0x18,0xb9,0x67]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mmin.w.mv.i tr2, acc0, tr3, 2
# CHECK-INST: th.mmin.w.mv.i tr2, acc0, tr3, 2
# CHECK-ENCODING: [0x2b,0x99,0x49,0x65]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mumin.w.mm tr3, acc1, acc0
# CHECK-INST: th.mumin.w.mm tr3, acc1, acc0
# CHECK-ENCODING: [0xab,0x19,0xda,0x77]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mumin.w.mv.i acc0, acc2, acc1, 4
# CHECK-INST: th.mumin.w.mv.i acc0, acc2, acc1, 4
# CHECK-ENCODING: [0x2b,0x9a,0x6a,0x76]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msrl.w.mm acc1, acc3, acc2
# CHECK-INST: th.msrl.w.mm acc1, acc3, acc2
# CHECK-ENCODING: [0xab,0x1a,0xfb,0x87]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msrl.w.mv.i acc2, tr0, acc3, 6
# CHECK-INST: th.msrl.w.mv.i acc2, tr0, acc3, 6
# CHECK-ENCODING: [0x2b,0x9b,0x0b,0x87]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msll.w.mm acc3, tr1, tr0
# CHECK-INST: th.msll.w.mm acc3, tr1, tr0
# CHECK-ENCODING: [0xab,0x1b,0x98,0x97]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msll.w.mv.i tr0, tr2, tr1, 0
# CHECK-INST: th.msll.w.mv.i tr0, tr2, tr1, 0
# CHECK-ENCODING: [0x2b,0x98,0x28,0x94]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msra.w.mm tr1, tr3, tr2
# CHECK-INST: th.msra.w.mm tr1, tr3, tr2
# CHECK-ENCODING: [0xab,0x18,0xb9,0xa7]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msra.w.mv.i tr2, acc0, tr3, 2
# CHECK-INST: th.msra.w.mv.i tr2, acc0, tr3, 2
# CHECK-ENCODING: [0x2b,0x99,0x49,0xa5]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}


# FP element-wise arithmetic
th.mfadd.h.mm tr3, acc1, acc0
# CHECK-INST: th.mfadd.h.mm tr3, acc1, acc0
# CHECK-ENCODING: [0xab,0x15,0xd6,0x0b]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfadd.h.mv.i acc0, acc2, acc1, 4
# CHECK-INST: th.mfadd.h.mv.i acc0, acc2, acc1, 4
# CHECK-ENCODING: [0x2b,0x96,0x66,0x0a]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfadd.s.mm acc1, acc3, acc2
# CHECK-INST: th.mfadd.s.mm acc1, acc3, acc2
# CHECK-ENCODING: [0xab,0x1a,0xfb,0x0b]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfadd.s.mv.i acc2, tr0, acc3, 6
# CHECK-INST: th.mfadd.s.mv.i acc2, tr0, acc3, 6
# CHECK-ENCODING: [0x2b,0x9b,0x0b,0x0b]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfadd.d.mm acc3, tr1, tr0
# CHECK-INST: th.mfadd.d.mm acc3, tr1, tr0
# CHECK-ENCODING: [0xab,0x1f,0x9c,0x0b]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfadd.d.mv.i tr0, tr2, tr1, 0
# CHECK-INST: th.mfadd.d.mv.i tr0, tr2, tr1, 0
# CHECK-ENCODING: [0x2b,0x9c,0x2c,0x08]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfsub.h.mm tr1, tr3, tr2
# CHECK-INST: th.mfsub.h.mm tr1, tr3, tr2
# CHECK-ENCODING: [0xab,0x14,0xb5,0x1b]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfsub.h.mv.i tr2, acc0, tr3, 2
# CHECK-INST: th.mfsub.h.mv.i tr2, acc0, tr3, 2
# CHECK-ENCODING: [0x2b,0x95,0x45,0x19]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfsub.s.mm tr3, acc1, acc0
# CHECK-INST: th.mfsub.s.mm tr3, acc1, acc0
# CHECK-ENCODING: [0xab,0x19,0xda,0x1b]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfsub.s.mv.i acc0, acc2, acc1, 4
# CHECK-INST: th.mfsub.s.mv.i acc0, acc2, acc1, 4
# CHECK-ENCODING: [0x2b,0x9a,0x6a,0x1a]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfsub.d.mm acc1, acc3, acc2
# CHECK-INST: th.mfsub.d.mm acc1, acc3, acc2
# CHECK-ENCODING: [0xab,0x1e,0xff,0x1b]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfsub.d.mv.i acc2, tr0, acc3, 6
# CHECK-INST: th.mfsub.d.mv.i acc2, tr0, acc3, 6
# CHECK-ENCODING: [0x2b,0x9f,0x0f,0x1b]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfmul.h.mm acc3, tr1, tr0
# CHECK-INST: th.mfmul.h.mm acc3, tr1, tr0
# CHECK-ENCODING: [0xab,0x17,0x94,0x2b]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfmul.h.mv.i tr0, tr2, tr1, 0
# CHECK-INST: th.mfmul.h.mv.i tr0, tr2, tr1, 0
# CHECK-ENCODING: [0x2b,0x94,0x24,0x28]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfmul.s.mm tr1, tr3, tr2
# CHECK-INST: th.mfmul.s.mm tr1, tr3, tr2
# CHECK-ENCODING: [0xab,0x18,0xb9,0x2b]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfmul.s.mv.i tr2, acc0, tr3, 2
# CHECK-INST: th.mfmul.s.mv.i tr2, acc0, tr3, 2
# CHECK-ENCODING: [0x2b,0x99,0x49,0x29]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfmul.d.mm tr3, acc1, acc0
# CHECK-INST: th.mfmul.d.mm tr3, acc1, acc0
# CHECK-ENCODING: [0xab,0x1d,0xde,0x2b]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfmul.d.mv.i acc0, acc2, acc1, 4
# CHECK-INST: th.mfmul.d.mv.i acc0, acc2, acc1, 4
# CHECK-ENCODING: [0x2b,0x9e,0x6e,0x2a]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfmax.h.mm acc1, acc3, acc2
# CHECK-INST: th.mfmax.h.mm acc1, acc3, acc2
# CHECK-ENCODING: [0xab,0x16,0xf7,0x3b]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfmax.h.mv.i acc2, tr0, acc3, 6
# CHECK-INST: th.mfmax.h.mv.i acc2, tr0, acc3, 6
# CHECK-ENCODING: [0x2b,0x97,0x07,0x3b]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfmax.s.mm acc3, tr1, tr0
# CHECK-INST: th.mfmax.s.mm acc3, tr1, tr0
# CHECK-ENCODING: [0xab,0x1b,0x98,0x3b]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfmax.s.mv.i tr0, tr2, tr1, 0
# CHECK-INST: th.mfmax.s.mv.i tr0, tr2, tr1, 0
# CHECK-ENCODING: [0x2b,0x98,0x28,0x38]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfmax.d.mm tr1, tr3, tr2
# CHECK-INST: th.mfmax.d.mm tr1, tr3, tr2
# CHECK-ENCODING: [0xab,0x1c,0xbd,0x3b]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfmax.d.mv.i tr2, acc0, tr3, 2
# CHECK-INST: th.mfmax.d.mv.i tr2, acc0, tr3, 2
# CHECK-ENCODING: [0x2b,0x9d,0x4d,0x39]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfmin.h.mm tr3, acc1, acc0
# CHECK-INST: th.mfmin.h.mm tr3, acc1, acc0
# CHECK-ENCODING: [0xab,0x15,0xd6,0x4b]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfmin.h.mv.i acc0, acc2, acc1, 4
# CHECK-INST: th.mfmin.h.mv.i acc0, acc2, acc1, 4
# CHECK-ENCODING: [0x2b,0x96,0x66,0x4a]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfmin.s.mm acc1, acc3, acc2
# CHECK-INST: th.mfmin.s.mm acc1, acc3, acc2
# CHECK-ENCODING: [0xab,0x1a,0xfb,0x4b]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfmin.s.mv.i acc2, tr0, acc3, 6
# CHECK-INST: th.mfmin.s.mv.i acc2, tr0, acc3, 6
# CHECK-ENCODING: [0x2b,0x9b,0x0b,0x4b]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfmin.d.mm acc3, tr1, tr0
# CHECK-INST: th.mfmin.d.mm acc3, tr1, tr0
# CHECK-ENCODING: [0xab,0x1f,0x9c,0x4b]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfmin.d.mv.i tr0, tr2, tr1, 0
# CHECK-INST: th.mfmin.d.mv.i tr0, tr2, tr1, 0
# CHECK-ENCODING: [0x2b,0x9c,0x2c,0x48]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}


# Element-wise conversion instructions
th.mucvtl.b.p tr1, tr2
# CHECK-INST: th.mucvtl.b.p tr1, tr2
# CHECK-ENCODING: [0xab,0x10,0x01,0x60]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mucvth.b.p tr2, tr3
# CHECK-INST: th.mucvth.b.p tr2, tr3
# CHECK-ENCODING: [0x2b,0x91,0x01,0x61]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mscvtl.b.p tr3, acc0
# CHECK-INST: th.mscvtl.b.p tr3, acc0
# CHECK-ENCODING: [0xab,0x11,0x82,0x60]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mscvth.b.p acc0, acc1
# CHECK-INST: th.mscvth.b.p acc0, acc1
# CHECK-ENCODING: [0x2b,0x92,0x82,0x61]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfcvtl.h.s acc1, acc2
# CHECK-INST: th.mfcvtl.h.s acc1, acc2
# CHECK-ENCODING: [0xab,0x16,0x0b,0x00]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfcvth.h.s acc2, acc3
# CHECK-INST: th.mfcvth.h.s acc2, acc3
# CHECK-ENCODING: [0x2b,0x97,0x0b,0x01]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfcvtl.s.h acc3, tr0
# CHECK-INST: th.mfcvtl.s.h acc3, tr0
# CHECK-ENCODING: [0xab,0x1b,0x04,0x00]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfcvth.s.h tr0, tr1
# CHECK-INST: th.mfcvth.s.h tr0, tr1
# CHECK-ENCODING: [0x2b,0x98,0x04,0x01]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfcvtl.s.d tr1, tr2
# CHECK-INST: th.mfcvtl.s.d tr1, tr2
# CHECK-ENCODING: [0xab,0x18,0x0d,0x00]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfcvth.s.d tr2, tr3
# CHECK-INST: th.mfcvth.s.d tr2, tr3
# CHECK-ENCODING: [0x2b,0x99,0x0d,0x01]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfcvtl.d.s tr3, acc0
# CHECK-INST: th.mfcvtl.d.s tr3, acc0
# CHECK-ENCODING: [0xab,0x1d,0x0a,0x00]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfcvth.d.s acc0, acc1
# CHECK-INST: th.mfcvth.d.s acc0, acc1
# CHECK-ENCODING: [0x2b,0x9e,0x0a,0x01]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfcvtl.bf16.s acc1, acc2
# CHECK-INST: th.mfcvtl.bf16.s acc1, acc2
# CHECK-ENCODING: [0xab,0x16,0x0b,0x02]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfcvth.bf16.s acc2, acc3
# CHECK-INST: th.mfcvth.bf16.s acc2, acc3
# CHECK-ENCODING: [0x2b,0x97,0x0b,0x03]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfcvtl.s.bf16 acc3, tr0
# CHECK-INST: th.mfcvtl.s.bf16 acc3, tr0
# CHECK-ENCODING: [0xab,0x1b,0x84,0x00]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfcvth.s.bf16 tr0, tr1
# CHECK-INST: th.mfcvth.s.bf16 tr0, tr1
# CHECK-ENCODING: [0x2b,0x98,0x84,0x01]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfcvtl.e4.h tr1, tr2
# CHECK-INST: th.mfcvtl.e4.h tr1, tr2
# CHECK-ENCODING: [0xab,0x10,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfcvth.e4.h tr2, tr3
# CHECK-INST: th.mfcvth.e4.h tr2, tr3
# CHECK-ENCODING: [0x2b,0x91,0x05,0x01]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfcvtl.h.e4 tr3, acc0
# CHECK-INST: th.mfcvtl.h.e4 tr3, acc0
# CHECK-ENCODING: [0xab,0x15,0x02,0x00]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfcvth.h.e4 acc0, acc1
# CHECK-INST: th.mfcvth.h.e4 acc0, acc1
# CHECK-ENCODING: [0x2b,0x96,0x02,0x01]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfcvtl.e5.h acc1, acc2
# CHECK-INST: th.mfcvtl.e5.h acc1, acc2
# CHECK-ENCODING: [0xab,0x12,0x87,0x00]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfcvth.e5.h acc2, acc3
# CHECK-INST: th.mfcvth.e5.h acc2, acc3
# CHECK-ENCODING: [0x2b,0x93,0x87,0x01]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfcvtl.h.e5 acc3, tr0
# CHECK-INST: th.mfcvtl.h.e5 acc3, tr0
# CHECK-ENCODING: [0xab,0x17,0x80,0x00]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfcvth.h.e5 tr0, tr1
# CHECK-INST: th.mfcvth.h.e5 tr0, tr1
# CHECK-ENCODING: [0x2b,0x94,0x80,0x01]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfcvtl.e4.s tr1, tr2
# CHECK-INST: th.mfcvtl.e4.s tr1, tr2
# CHECK-ENCODING: [0xab,0x10,0x09,0x00]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfcvth.e4.s tr2, tr3
# CHECK-INST: th.mfcvth.e4.s tr2, tr3
# CHECK-ENCODING: [0x2b,0x91,0x09,0x01]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfcvtl.e5.s tr3, acc0
# CHECK-INST: th.mfcvtl.e5.s tr3, acc0
# CHECK-ENCODING: [0xab,0x11,0x0a,0x02]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfcvth.e5.s acc0, acc1
# CHECK-INST: th.mfcvth.e5.s acc0, acc1
# CHECK-ENCODING: [0x2b,0x92,0x0a,0x03]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfcvt.tf32.s acc1, acc2
# CHECK-INST: th.mfcvt.tf32.s acc1, acc2
# CHECK-ENCODING: [0xab,0x1a,0x0b,0x03]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfcvt.s.tf32 acc2, acc3
# CHECK-INST: th.mfcvt.s.tf32 acc2, acc3
# CHECK-ENCODING: [0x2b,0x9b,0x8b,0x00]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfucvt.w.s acc3, tr0
# CHECK-INST: th.mfucvt.w.s acc3, tr0
# CHECK-ENCODING: [0xab,0x1b,0x08,0x12]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfscvt.w.s tr0, tr1
# CHECK-INST: th.mfscvt.w.s tr0, tr1
# CHECK-ENCODING: [0x2b,0x98,0x88,0x12]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mufcvt.s.w tr1, tr2
# CHECK-INST: th.mufcvt.s.w tr1, tr2
# CHECK-ENCODING: [0xab,0x18,0x09,0x10]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msfcvt.s.w tr2, tr3
# CHECK-INST: th.msfcvt.s.w tr2, tr3
# CHECK-ENCODING: [0x2b,0x99,0x89,0x10]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfucvtl.b.h tr3, acc0
# CHECK-INST: th.mfucvtl.b.h tr3, acc0
# CHECK-ENCODING: [0xab,0x11,0x06,0x12]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfucvth.b.h acc0, acc1
# CHECK-INST: th.mfucvth.b.h acc0, acc1
# CHECK-ENCODING: [0x2b,0x92,0x06,0x13]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfscvtl.b.h acc1, acc2
# CHECK-INST: th.mfscvtl.b.h acc1, acc2
# CHECK-ENCODING: [0xab,0x12,0x87,0x12]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mfscvth.b.h acc2, acc3
# CHECK-INST: th.mfscvth.b.h acc2, acc3
# CHECK-ENCODING: [0x2b,0x93,0x87,0x13]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mufcvtl.h.b acc3, tr0
# CHECK-INST: th.mufcvtl.h.b acc3, tr0
# CHECK-ENCODING: [0xab,0x17,0x00,0x10]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mufcvth.h.b tr0, tr1
# CHECK-INST: th.mufcvth.h.b tr0, tr1
# CHECK-ENCODING: [0x2b,0x94,0x00,0x11]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msfcvtl.h.b tr1, tr2
# CHECK-INST: th.msfcvtl.h.b tr1, tr2
# CHECK-ENCODING: [0xab,0x14,0x81,0x10]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.msfcvth.h.b tr2, tr3
# CHECK-INST: th.msfcvth.h.b tr2, tr3
# CHECK-ENCODING: [0x2b,0x95,0x81,0x11]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mn4clipl.w.mm tr3, acc1, acc0
# CHECK-INST: th.mn4clipl.w.mm tr3, acc1, acc0
# CHECK-ENCODING: [0xab,0x19,0xda,0x23]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mn4clipl.w.mv.i acc0, acc2, acc1, 4
# CHECK-INST: th.mn4clipl.w.mv.i acc0, acc2, acc1, 4
# CHECK-ENCODING: [0x2b,0x9a,0x6a,0x22]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mn4cliph.w.mm acc1, acc3, acc2
# CHECK-INST: th.mn4cliph.w.mm acc1, acc3, acc2
# CHECK-ENCODING: [0xab,0x1a,0xfb,0x33]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mn4cliph.w.mv.i acc2, tr0, acc3, 6
# CHECK-INST: th.mn4cliph.w.mv.i acc2, tr0, acc3, 6
# CHECK-ENCODING: [0x2b,0x9b,0x0b,0x33]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mn4cliplu.w.mm acc3, tr1, tr0
# CHECK-INST: th.mn4cliplu.w.mm acc3, tr1, tr0
# CHECK-ENCODING: [0xab,0x1b,0x98,0x43]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mn4cliplu.w.mv.i tr0, tr2, tr1, 0
# CHECK-INST: th.mn4cliplu.w.mv.i tr0, tr2, tr1, 0
# CHECK-ENCODING: [0x2b,0x98,0x28,0x40]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mn4cliphu.w.mm tr1, tr3, tr2
# CHECK-INST: th.mn4cliphu.w.mm tr1, tr3, tr2
# CHECK-ENCODING: [0xab,0x18,0xb9,0x53]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}

th.mn4cliphu.w.mv.i tr2, acc0, tr3, 2
# CHECK-INST: th.mn4cliphu.w.mv.i tr2, acc0, tr3, 2
# CHECK-ENCODING: [0x2b,0x99,0x49,0x51]
# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}
