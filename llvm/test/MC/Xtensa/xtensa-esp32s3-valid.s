# RUN: llvm-mc %s -triple=xtensa  -mcpu=esp32s3 -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

.align	4
LBL0:

# CHECK-INST:  ee.clr_bit_gpio_out  52
# CHECK: encoding: [0x44,0x43,0x76]
ee.clr_bit_gpio_out 52

# CHECK-INST:  ee.get_gpio_in  a2
# CHECK: encoding: [0x24,0x08,0x65]
ee.get_gpio_in a2

# CHECK-INST:  ee.set_bit_gpio_out  18
# CHECK: encoding: [0x24,0x41,0x75]
ee.set_bit_gpio_out 18

# CHECK-INST:  ee.wr_mask_gpio_out	a3, a2
# CHECK: encoding: [0x34,0x42,0x72]
ee.wr_mask_gpio_out	a3, a2

ee.andq q5, q6, q4
# CHECK: ee.andq	 q5, q6, q4                     # encoding: [0xc4,0xb8,0xed]
ee.bitrev q2, a6
# CHECK: ee.bitrev	 q2, a6                 # encoding: [0x64,0x7b,0xdd]
ee.cmul.s16 q3, q6, q2, 3
# CHECK: ee.cmul.s16	 q3, q6, q2, 3          # encoding: [0x34,0x96,0x9e]
ee.cmul.s16.ld.incp q2, a7, q5, q1, q4, 2
# CHECK: ee.cmul.s16.ld.incp	 q2, a7, q5, q1, q4, 2 # encoding: [0xe7,0x60,0x2a,0x1c]
ee.cmul.s16.st.incp q7, a11, q1, q5, q2, 3
# CHECK: ee.cmul.s16.st.incp	 q7, a11, q1, q5, q2, 3 # encoding: [0x3b,0x57,0x83,0x1c]
ee.fft.ams.s16.ld.incp q5, a5, q3, q1, q1, q2, q5, 1
# CHECK: ee.fft.ams.s16.ld.incp	 q5, a5, q3, q1, q1, q2, q5, 1 # encoding: [0xd5,0x5a,0x4a,0x1a]
ee.fft.ams.s16.ld.incp.uaup q7, a12, q4, q1, q5, q6, q3, 0
# CHECK: ee.fft.ams.s16.ld.incp.uaup	 q7, a12, q4, q1, q5, q6, q3, 0 # encoding: [0xfc,0x66,0x87,0x1a]
ee.fft.ams.s16.ld.r32.decp q6, a5, q0, q2, q7, q2, q0, 0
# CHECK: ee.fft.ams.s16.ld.r32.decp	 q6, a5, q0, q2, q7, q2, q0, 0 # encoding: [0x65,0xc2,0x11,0x1b]
ee.fft.ams.s16.st.incp q3, q6, a7, a6, q5, q5, q1, 1
# CHECK: ee.fft.ams.s16.st.incp	 q3, q6, a7, a6, q5, q5, q1, 1 # encoding: [0x67,0x4b,0xeb,0x14]
ee.fft.cmul.s16.ld.xp q3, a12, a6, q7, q0, q7, 2
# CHECK: ee.fft.cmul.s16.ld.xp	 q3, a12, a6, q7, q0, q7, 2 # encoding: [0x6c,0x1f,0xae,0x1b]
ee.fft.cmul.s16.st.xp q4, q0, q0, a2, a8, 6, 1, 1
# CHECK: ee.fft.cmul.s16.st.xp	 q4, q0, q0, a2, a8, 6, 1, 1 # encoding: [0x82,0x30,0x51,0x15]
ee.fft.r2bf.s16 q7, q1, q3, q6, 1
# CHECK: ee.fft.r2bf.s16	 q7, q1, q3, q6, 1      # encoding: [0x54,0x9d,0xfc]
ee.fft.r2bf.s16.st.incp q7, q3, q7, a2, 2
# CHECK: ee.fft.r2bf.s16.st.incp	 q7, q3, q7, a2, 2 # encoding: [0x42,0xd7,0x1e,0x1d]
ee.fft.vst.r32.decp q3, a14, 0
# CHECK: ee.fft.vst.r32.decp	 q3, a14, 0     # encoding: [0xe4,0xb3,0xdd]
ee.ldf.128.ip f3, f5, f8, f0, a13, 64
# CHECK: ee.ldf.128.ip	 f3, f5, f8, f0, a13, 64 # encoding: [0x4d,0x80,0x35,0x10]
ee.ldf.128.xp f5, f2, f4, f4, a7, a8
# CHECK: ee.ldf.128.xp	 f5, f2, f4, f4, a7, a8 # encoding: [0x87,0x44,0x52,0x11]
ee.ldf.64.ip f6, f5, a1, 488
# CHECK: ee.ldf.64.ip	 f6, f5, a1, 488        # encoding: [0x51,0x65,0x1e,0x1c]
ee.ldf.64.xp f0, f6, a3, a8
# CHECK: ee.ldf.64.xp	 f0, f6, a3, a8         # encoding: [0x30,0x08,0x66]
ee.ldqa.s16.128.ip a11, 1904
# CHECK: ee.ldqa.s16.128.ip	 a11, 1904      # encoding: [0xb4,0x77,0x01]
ee.ldqa.s16.128.xp a6, a2
# CHECK: ee.ldqa.s16.128.xp	 a6, a2         # encoding: [0x64,0x42,0x7e]
ee.ldqa.s8.128.ip a8, 320
# CHECK: ee.ldqa.s8.128.ip	 a8, 320        # encoding: [0x84,0x14,0x11]
ee.ldqa.s8.128.xp a6, a11
# CHECK: ee.ldqa.s8.128.xp	 a6, a11        # encoding: [0x64,0x4b,0x71]
ee.ldqa.u16.128.ip a2, -1424
# CHECK: ee.ldqa.u16.128.ip	 a2, -1424      # encoding: [0x24,0x27,0x45]
ee.ldqa.u16.128.xp a3, a4
# CHECK: ee.ldqa.u16.128.xp	 a3, a4         # encoding: [0x34,0x44,0x7a]
ee.ldqa.u8.128.ip a4, 784
# CHECK: ee.ldqa.u8.128.ip	 a4, 784        # encoding: [0x44,0x31,0x15]
ee.ldqa.u8.128.xp a4, a9
# CHECK: ee.ldqa.u8.128.xp	 a4, a9         # encoding: [0x44,0x49,0x70]
ee.ldxq.32 q2, q6, a11, 2, 1
# CHECK: ee.ldxq.32	 q2, q6, a11, 2, 1      # encoding: [0xdb,0x8f,0x25,0x1c]
ee.ld.128.usar.ip q4, a8, -592
# CHECK: ee.ld.128.usar.ip	 q4, a8, -592   # encoding: [0x84,0x5b,0xe1]
ee.ld.128.usar.xp q1, a9, a7
# CHECK: ee.ld.128.usar.xp	 q1, a9, a7     # encoding: [0x94,0x87,0x8d]
ee.ld.accx.ip a2, 720
# CHECK: ee.ld.accx.ip	 a2, 720                # encoding: [0x24,0x5a,0x0e]
ee.ld.qacc_h.h.32.ip a6, -292
# CHECK: ee.ld.qacc_h.h.32.ip	 a6, -292       # encoding: [0x64,0x37,0x5e]
ee.ld.qacc_h.l.128.ip a14, 96
# CHECK: ee.ld.qacc_h.l.128.ip	 a14, 96        # encoding: [0xe4,0x06,0x06]
ee.ld.qacc_l.h.32.ip a0, -184
# CHECK: ee.ld.qacc_l.h.32.ip	 a0, -184       # encoding: [0x04,0x52,0x56]
ee.ld.qacc_l.l.128.ip a5, -352
# CHECK: ee.ld.qacc_l.l.128.ip	 a5, -352       # encoding: [0x54,0x6a,0x40]
ee.ld.ua_state.ip a3, 864
# CHECK: ee.ld.ua_state.ip	 a3, 864        # encoding: [0x34,0x36,0x10]
ee.movi.32.a q7, a0, 1
# CHECK: ee.movi.32.a	 q7, a0, 1              # encoding: [0x04,0xf5,0xfd]
ee.movi.32.q q5, a5, 3
# CHECK: ee.movi.32.q	 q5, a5, 3              # encoding: [0x54,0xbe,0xed]
ee.mov.s16.qacc q1
# CHECK: ee.mov.s16.qacc	 q1                     # encoding: [0x24,0xff,0xcd]
ee.mov.s8.qacc q7
# CHECK: ee.mov.s8.qacc	 q7                     # encoding: [0x34,0xff,0xfd]
ee.mov.u16.qacc q2
# CHECK: ee.mov.u16.qacc	 q2                     # encoding: [0x64,0x7f,0xdd]
ee.mov.u8.qacc q2
# CHECK: ee.mov.u8.qacc	 q2                     # encoding: [0x74,0x7f,0xdd]
ee.notq q7, q0
# CHECK: ee.notq	 q7, q0                         # encoding: [0x04,0xff,0xfd]
ee.orq q1, q5, q3
# CHECK: ee.orq	 q1, q5, q3                     # encoding: [0xb4,0xf4,0xcd]
ee.slci.2q q7, q4, 2
# CHECK: ee.slci.2q	 q7, q4, 2              # encoding: [0x24,0xc6,0xfc]
ee.slcxxp.2q q6, q7, a11, a4
# CHECK: ee.slcxxp.2q	 q6, q7, a11, a4        # encoding: [0xb4,0x74,0xb6]
ee.srci.2q q6, q6, 14
# CHECK: ee.srci.2q	 q6, q6, 14             # encoding: [0xe4,0x6a,0xfc]
ee.srcmb.s16.qacc q3, a7, 0
# CHECK: ee.srcmb.s16.qacc	 q3, a7, 0      # encoding: [0x74,0xf2,0xdd]
ee.srcmb.s8.qacc q4, a1, 1
# CHECK: ee.srcmb.s8.qacc	 q4, a1, 1              # encoding: [0x14,0x7e,0xed]
ee.srcq.128.st.incp q1, q4, a6
# CHECK: ee.srcq.128.st.incp	 q1, q4, a6     # encoding: [0x64,0x1e,0xec]
ee.srcxxp.2q q6, q0, a2, a14
# CHECK: ee.srcxxp.2q	 q6, q0, a2, a14        # encoding: [0x24,0x0e,0xf6]
ee.src.q q6, q7, q5
# CHECK: ee.src.q	 q6, q7, q5                     # encoding: [0x64,0xf3,0xec]
ee.src.q.ld.ip q2, a2, 1792, q6, q7
# CHECK: ee.src.q.ld.ip	 q2, a2, 1792, q6, q7   # encoding: [0x02,0xa7,0x35,0x1c]
ee.src.q.ld.xp q2, a4, a9, q1, q7
# CHECK: ee.src.q.ld.xp	 q2, a4, a9, q1, q7     # encoding: [0x94,0x47,0x04,0x1d]
ee.src.q.qup q4, q3, q7
# CHECK: ee.src.q.qup	 q4, q3, q7             # encoding: [0x44,0xb7,0xfc]
ee.srs.accx a12, a1, 0
# CHECK: ee.srs.accx	 a12, a1, 0             # encoding: [0x14,0x1c,0x7e]
ee.stf.128.ip f4, f3, f8, f2, a4, -128
# CHECK: ee.stf.128.ip	 f4, f3, f8, f2, a4, -128 # encoding: [0x84,0x82,0x43,0x12]
ee.stf.128.xp f2, f0, f5, f8, a11, a5
# CHECK: ee.stf.128.xp	 f2, f0, f5, f8, a11, a5 # encoding: [0x5b,0x58,0x20,0x13]
ee.stf.64.ip f3, f6, a10, -848
# CHECK: ee.stf.64.ip	 f3, f6, a10, -848      # encoding: [0x6a,0x36,0x4b,0x1c]
ee.stf.64.xp f2, f1, a1, a14
# CHECK: ee.stf.64.xp	 f2, f1, a1, a14        # encoding: [0x10,0x2e,0x17]
ee.stxq.32 q5, q2, a5, 0, 1
# CHECK: ee.stxq.32	 q5, q2, a5, 0, 1       # encoding: [0x05,0x8d,0xc0,0x1c]
ee.st.accx.ip a10, 24
# CHECK: ee.st.accx.ip	 a10, 24                # encoding: [0xa4,0x03,0x02]
ee.st.qacc_h.h.32.ip a14, 380
# CHECK: ee.st.qacc_h.h.32.ip	 a14, 380       # encoding: [0xe4,0x5f,0x12]
ee.st.qacc_h.l.128.ip a7, -624
# CHECK: ee.st.qacc_h.l.128.ip	 a7, -624       # encoding: [0x74,0x59,0x4d]
ee.st.qacc_l.h.32.ip a10, -20
# CHECK: ee.st.qacc_l.h.32.ip	 a10, -20       # encoding: [0xa4,0x7b,0x5d]
ee.st.qacc_l.l.128.ip a4, 1936
# CHECK: ee.st.qacc_l.l.128.ip	 a4, 1936       # encoding: [0x44,0x79,0x0c]
ee.st.ua_state.ip a4, -1728
# CHECK: ee.st.ua_state.ip	 a4, -1728      # encoding: [0x44,0x14,0x5c]
ee.vadds.s16 q5, q1, q4
# CHECK: ee.vadds.s16	 q5, q1, q4             # encoding: [0x64,0xc1,0xae]
ee.vadds.s16.ld.incp q6, a6, q1, q3, q1
# CHECK: ee.vadds.s16.ld.incp	 q6, a6, q1, q3, q1 # encoding: [0xd6,0xca,0x62,0x1c]
ee.vadds.s16.st.incp q4, a0, q1, q3, q1
# CHECK: ee.vadds.s16.st.incp	 q4, a0, q1, q3, q1 # encoding: [0x00,0xcc,0x92,0x1c]
ee.vadds.s32 q3, q5, q2
# CHECK: ee.vadds.s32	 q3, q5, q2             # encoding: [0x74,0x95,0x9e]
ee.vadds.s32.ld.incp q4, a4, q1, q6, q5
# CHECK: ee.vadds.s32.ld.incp	 q4, a4, q1, q6, q5 # encoding: [0xd4,0xab,0x43,0x1c]
ee.vadds.s32.st.incp q5, a1, q0, q6, q0
# CHECK: ee.vadds.s32.st.incp	 q5, a1, q0, q6, q0 # encoding: [0x11,0x85,0x91,0x1c]
ee.vadds.s8 q4, q4, q5
# CHECK: ee.vadds.s8	 q4, q4, q5             # encoding: [0x84,0x4c,0xae]
ee.vadds.s8.ld.incp q2, a14, q0, q3, q3
# CHECK: ee.vadds.s8.ld.incp	 q2, a14, q0, q3, q3 # encoding: [0xce,0xd9,0x20,0x1c]
ee.vadds.s8.st.incp q0, a9, q4, q7, q0
# CHECK: ee.vadds.s8.st.incp	 q0, a9, q4, q7, q0 # encoding: [0x29,0xc0,0x99,0x1c]
ee.vcmp.eq.s16 q5, q3, q0
# CHECK: ee.vcmp.eq.s16	 q5, q3, q0             # encoding: [0x94,0x83,0xae]
ee.vcmp.eq.s32 q5, q5, q4
# CHECK: ee.vcmp.eq.s32	 q5, q5, q4             # encoding: [0xa4,0xc5,0xae]
ee.vcmp.eq.s8 q0, q4, q2
# CHECK: ee.vcmp.eq.s8	 q0, q4, q2             # encoding: [0xb4,0x14,0x8e]
ee.vcmp.gt.s16 q1, q5, q2
# CHECK: ee.vcmp.gt.s16	 q1, q5, q2             # encoding: [0xc4,0x95,0x8e]
ee.vcmp.gt.s32 q4, q1, q5
# CHECK: ee.vcmp.gt.s32	 q4, q1, q5             # encoding: [0xd4,0x49,0xae]
ee.vcmp.gt.s8 q3, q6, q3
# CHECK: ee.vcmp.gt.s8	 q3, q6, q3             # encoding: [0xe4,0x9e,0x9e]
ee.vcmp.lt.s16 q3, q7, q0
# CHECK: ee.vcmp.lt.s16	 q3, q7, q0             # encoding: [0xf4,0x87,0x9e]
ee.vcmp.lt.s32 q2, q2, q1
# CHECK: ee.vcmp.lt.s32	 q2, q2, q1             # encoding: [0x04,0x2a,0x9e]
ee.vcmp.lt.s8 q7, q1, q6
# CHECK: ee.vcmp.lt.s8	 q7, q1, q6             # encoding: [0x14,0xf1,0xbe]
ee.vldbc.16 q6, a11
# CHECK: ee.vldbc.16	 q6, a11                # encoding: [0xb4,0x73,0xfd]
ee.vldbc.16.ip q6, a4, 124
# CHECK: ee.vldbc.16.ip	 q6, a4, 124            # encoding: [0x44,0x3e,0xb5]
ee.vldbc.16.xp q2, a0, a7
# CHECK: ee.vldbc.16.xp	 q2, a0, a7             # encoding: [0x04,0x47,0x9d]
ee.vldbc.32 q4, a0
# CHECK: ee.vldbc.32	 q4, a0                 # encoding: [0x04,0x77,0xed]
ee.vldbc.32.ip q6, a12, 308
# CHECK: ee.vldbc.32.ip	 q6, a12, 308           # encoding: [0xc4,0x4d,0xb2]
ee.vldbc.32.xp q1, a11, a0
# CHECK: ee.vldbc.32.xp	 q1, a11, a0            # encoding: [0xb4,0x90,0x8d]
ee.vldbc.8 q2, a3
# CHECK: ee.vldbc.8	 q2, a3                 # encoding: [0x34,0x3b,0xdd]
ee.vldbc.8.ip q3, a3, 103
# CHECK: ee.vldbc.8.ip	 q3, a3, 103            # encoding: [0x34,0xe7,0xd5]
ee.vldbc.8.xp q2, a0, a13
# CHECK: ee.vldbc.8.xp	 q2, a0, a13            # encoding: [0x04,0x5d,0x9d]
ee.vldhbc.16.incp q5, q5, a14
# CHECK: ee.vldhbc.16.incp	 q5, q5, a14    # encoding: [0xe4,0xd2,0xec]
ee.vld.128.ip q3, a14, 1248
# CHECK: ee.vld.128.ip	 q3, a14, 1248          # encoding: [0xe4,0xce,0x93]
ee.vld.128.xp q5, a10, a12
# CHECK: ee.vld.128.xp	 q5, a10, a12           # encoding: [0xa4,0xac,0xad]
ee.vld.h.64.ip q4, a14, 240
# CHECK: ee.vld.h.64.ip	 q4, a14, 240           # encoding: [0xe4,0x1e,0xa8]
ee.vld.h.64.xp q7, a4, a8
# CHECK: ee.vld.h.64.xp	 q7, a4, a8             # encoding: [0x44,0xe8,0xbd]
ee.vld.l.64.ip q1, a8, 8
# CHECK: ee.vld.l.64.ip	 q1, a8, 8              # encoding: [0x84,0x81,0x89]
ee.vld.l.64.xp q1, a2, a9
# CHECK: ee.vld.l.64.xp	 q1, a2, a9             # encoding: [0x24,0xb9,0x8d]
ee.vmax.s16 q2, q5, q6
# CHECK: ee.vmax.s16	 q2, q5, q6             # encoding: [0x24,0x75,0x9e]
ee.vmax.s16.ld.incp q0, a0, q6, q1, q2
# CHECK: ee.vmax.s16.ld.incp	 q0, a0, q6, q1, q2 # encoding: [0xd0,0x51,0x0c,0x1c]
ee.vmax.s16.st.incp q5, a10, q6, q6, q7
# CHECK: ee.vmax.s16.st.incp	 q5, a10, q6, q6, q7 # encoding: [0x3a,0xbd,0x9d,0x1c]
ee.vmax.s32 q3, q2, q7
# CHECK: ee.vmax.s32	 q3, q2, q7             # encoding: [0x34,0xfa,0x9e]
ee.vmax.s32.ld.incp q1, a3, q1, q1, q0
# CHECK: ee.vmax.s32.ld.incp	 q1, a3, q1, q1, q0 # encoding: [0xe3,0x41,0x12,0x1c]
ee.vmax.s32.st.incp q3, a12, q4, q6, q3
# CHECK: ee.vmax.s32.st.incp	 q3, a12, q4, q6, q3 # encoding: [0x0c,0x9b,0xa9,0x1c]
ee.vmax.s8 q4, q1, q6
# CHECK: ee.vmax.s8	 q4, q1, q6             # encoding: [0x44,0x71,0xae]
ee.vmax.s8.ld.incp q3, a10, q5, q1, q5
# CHECK: ee.vmax.s8.ld.incp	 q3, a10, q5, q1, q5 # encoding: [0xfa,0x69,0x3a,0x1c]
ee.vmax.s8.st.incp q3, a9, q3, q6, q7
# CHECK: ee.vmax.s8.st.incp	 q3, a9, q3, q6, q7 # encoding: [0x09,0xbb,0xb7,0x1c]
ee.vmin.s16 q6, q2, q5
# CHECK: ee.vmin.s16	 q6, q2, q5             # encoding: [0x54,0x6a,0xbe]
ee.vmin.s16.ld.incp q5, a3, q2, q4, q0
# CHECK: ee.vmin.s16.ld.incp	 q5, a3, q2, q4, q0 # encoding: [0xe3,0x02,0x55,0x1c]
ee.vmin.s16.st.incp q4, a9, q4, q6, q0
# CHECK: ee.vmin.s16.st.incp	 q4, a9, q4, q6, q0 # encoding: [0x19,0x84,0xa9,0x1c]
ee.vmin.s32 q1, q1, q6
# CHECK: ee.vmin.s32	 q1, q1, q6             # encoding: [0x64,0xf1,0x8e]
ee.vmin.s32.ld.incp q0, a1, q3, q2, q0
# CHECK: ee.vmin.s32.ld.incp	 q0, a1, q3, q2, q0 # encoding: [0xe1,0x83,0x06,0x1c]
ee.vmin.s32.st.incp q0, a12, q4, q4, q3
# CHECK: ee.vmin.s32.st.incp	 q0, a12, q4, q4, q3 # encoding: [0x1c,0x18,0xb9,0x1c]
ee.vmin.s8 q7, q6, q0
# CHECK: ee.vmin.s8	 q7, q6, q0             # encoding: [0x74,0xa6,0xbe]
ee.vmin.s8.ld.incp q2, a13, q7, q7, q3
# CHECK: ee.vmin.s8.ld.incp	 q2, a13, q7, q7, q3 # encoding: [0xfd,0xda,0x2f,0x1c]
ee.vmin.s8.st.incp q2, a4, q4, q7, q1
# CHECK: ee.vmin.s8.st.incp	 q2, a4, q4, q7, q1 # encoding: [0x24,0xca,0xa9,0x1c]
ee.vmulas.s16.accx q0, q7
# CHECK: ee.vmulas.s16.accx	 q0, q7         # encoding: [0x84,0x58,0x1a]
ee.vmulas.s16.accx.ld.ip q7, a7, -16, q2, q0
# CHECK: ee.vmulas.s16.accx.ld.ip	 q7, a7, -16, q2, q0 # encoding: [0xf7,0x80,0xf0,0x1f]
ee.vmulas.s16.accx.ld.ip.qup q5, a14, 32, q0, q2, q0, q2
# CHECK: ee.vmulas.s16.accx.ld.ip.qup	 q5, a14, 32, q0, q2, q0, q2 # encoding: [0x2e,0x10,0x54,0x00]
ee.vmulas.s16.accx.ld.xp q1, a0, a1, q2, q6
# CHECK: ee.vmulas.s16.accx.ld.xp	 q1, a0, a1, q2, q6 # encoding: [0x10,0xb1,0x10,0x1e]
ee.vmulas.s16.accx.ld.xp.qup q4, a8, a10, q4, q0, q0, q3
# CHECK: ee.vmulas.s16.accx.ld.xp.qup	 q4, a8, a10, q4, q0, q0, q3 # encoding: [0xa8,0x00,0x47,0x16]
ee.vmulas.s16.qacc q0, q6
# CHECK: ee.vmulas.s16.qacc	 q0, q6         # encoding: [0x84,0x70,0x1a]
ee.vmulas.s16.qacc.ldbc.incp q2, a6, q3, q4
# CHECK: ee.vmulas.s16.qacc.ldbc.incp	 q2, a6, q3, q4 # encoding: [0x64,0xc3,0x87]
ee.vmulas.s16.qacc.ldbc.incp.qup q0, a4, q1, q6, q4, q5
# CHECK: ee.vmulas.s16.qacc.ldbc.incp.qup	 q0, a4, q1, q6, q4, q5 # encoding: [0x84,0x74,0x0a,0x1c]
ee.vmulas.s16.qacc.ld.ip q7, a7, -64, q7, q7
# CHECK: ee.vmulas.s16.qacc.ld.ip	 q7, a7, -64, q7, q7 # encoding: [0xc7,0xf8,0xf3,0x1f]
ee.vmulas.s16.qacc.ld.ip.qup q0, a10, 48, q3, q6, q3, q6
# CHECK: ee.vmulas.s16.qacc.ld.ip.qup	 q0, a10, 48, q3, q6, q3, q6 # encoding: [0x3a,0xf3,0x0c,0x02]
ee.vmulas.s16.qacc.ld.xp q3, a11, a4, q4, q5
# CHECK: ee.vmulas.s16.qacc.ld.xp	 q3, a11, a4, q4, q5 # encoding: [0x4b,0x29,0x33,0x1e]
ee.vmulas.s16.qacc.ld.xp.qup q2, a9, a1, q3, q2, q1, q7
# CHECK: ee.vmulas.s16.qacc.ld.xp.qup	 q2, a9, a1, q3, q2, q1, q7 # encoding: [0x19,0xd1,0xae,0x16]
ee.vmulas.s8.accx q1, q0
# CHECK: ee.vmulas.s8.accx	 q1, q0         # encoding: [0xc4,0x01,0x1a]
ee.vmulas.s8.accx.ld.ip q2, a8, 80, q3, q0
# CHECK: ee.vmulas.s8.accx.ld.ip	 q2, a8, 80, q3, q0 # encoding: [0x58,0xc0,0x24,0x1e]
ee.vmulas.s8.accx.ld.ip.qup q2, a9, -80, q1, q2, q6, q3
# CHECK: ee.vmulas.s8.accx.ld.ip.qup	 q2, a9, -80, q1, q2, q6, q3 # encoding: [0xb9,0x56,0xa6,0x05]
ee.vmulas.s8.accx.ld.xp q3, a3, a4, q4, q7
# CHECK: ee.vmulas.s8.accx.ld.xp	 q3, a3, a4, q4, q7 # encoding: [0x43,0x39,0x35,0x1e]
ee.vmulas.s8.accx.ld.xp.qup q0, a3, a1, q4, q5, q3, q3
# CHECK: ee.vmulas.s8.accx.ld.xp.qup	 q0, a3, a1, q4, q5, q3, q3 # encoding: [0x13,0x2b,0x07,0x17]
ee.vmulas.s8.qacc q5, q7
# CHECK: ee.vmulas.s8.qacc	 q5, q7         # encoding: [0xc4,0x7d,0x1a]
ee.vmulas.s8.qacc.ldbc.incp q7, a1, q6, q1
# CHECK: ee.vmulas.s8.qacc.ldbc.incp	 q7, a1, q6, q1 # encoding: [0x14,0xae,0xb7]
ee.vmulas.s8.qacc.ldbc.incp.qup q3, a11, q4, q6, q5, q6
# CHECK: ee.vmulas.s8.qacc.ldbc.incp.qup	 q3, a11, q4, q6, q5, q6 # encoding: [0x9b,0x35,0x3d,0x1c]
ee.vmulas.s8.qacc.ld.ip q5, a10, -16, q0, q0
# CHECK: ee.vmulas.s8.qacc.ld.ip	 q5, a10, -16, q0, q0 # encoding: [0xfa,0x00,0xd6,0x1f]
ee.vmulas.s8.qacc.ld.ip.qup q7, a9, -48, q6, q2, q1, q2
# CHECK: ee.vmulas.s8.qacc.ld.ip.qup	 q7, a9, -48, q6, q2, q1, q2 # encoding: [0xd9,0x91,0xf5,0x07]
ee.vmulas.s8.qacc.ld.xp q1, a1, a12, q5, q0
# CHECK: ee.vmulas.s8.qacc.ld.xp	 q1, a1, a12, q5, q0 # encoding: [0xc1,0x41,0x17,0x1e]
ee.vmulas.s8.qacc.ld.xp.qup q0, a1, a14, q1, q6, q2, q4
# CHECK: ee.vmulas.s8.qacc.ld.xp.qup	 q0, a1, a14, q1, q6, q2, q4 # encoding: [0xe1,0x72,0x88,0x17]
ee.vmulas.u16.accx q7, q1
# CHECK: ee.vmulas.u16.accx	 q7, q1         # encoding: [0x84,0x0f,0x0a]
ee.vmulas.u16.accx.ld.ip q5, a8, -32, q1, q4
# CHECK: ee.vmulas.u16.accx.ld.ip	 q5, a8, -32, q1, q4 # encoding: [0xe8,0x60,0xd8,0x1f]
ee.vmulas.u16.accx.ld.ip.qup q1, a0, 48, q7, q4, q4, q0
# CHECK: ee.vmulas.u16.accx.ld.ip.qup	 q1, a0, 48, q7, q4, q4, q0 # encoding: [0x30,0xe4,0x11,0x08]
ee.vmulas.u16.accx.ld.xp q3, a14, a4, q5, q4
# CHECK: ee.vmulas.u16.accx.ld.xp	 q3, a14, a4, q5, q4 # encoding: [0x4e,0x61,0x39,0x1e]
ee.vmulas.u16.accx.ld.xp.qup q4, a3, a7, q6, q2, q4, q4
# CHECK: ee.vmulas.u16.accx.ld.xp.qup	 q4, a3, a7, q6, q2, q4, q4 # encoding: [0x73,0x94,0x49,0x18]
ee.vmulas.u16.qacc q5, q5
# CHECK: ee.vmulas.u16.qacc	 q5, q5         # encoding: [0x84,0x6d,0x0a]
ee.vmulas.u16.qacc.ldbc.incp q6, a7, q0, q3
# CHECK: ee.vmulas.u16.qacc.ldbc.incp	 q6, a7, q0, q3 # encoding: [0x74,0x98,0xd7]
ee.vmulas.u16.qacc.ldbc.incp.qup q0, a12, q6, q3, q2, q0
# CHECK: ee.vmulas.u16.qacc.ldbc.incp.qup	 q0, a12, q6, q3, q2, q0 # encoding: [0xac,0x9a,0x01,0x1c]
ee.vmulas.u16.qacc.ld.ip q4, a10, 16, q3, q2
# CHECK: ee.vmulas.u16.qacc.ld.ip	 q4, a10, 16, q3, q2 # encoding: [0x1a,0xd0,0x4a,0x1e]
ee.vmulas.u16.qacc.ld.ip.qup q2, a4, 0, q5, q4, q2, q6
# CHECK: ee.vmulas.u16.qacc.ld.ip.qup	 q2, a4, 0, q5, q4, q2, q6 # encoding: [0x04,0x62,0x2d,0x0a]
ee.vmulas.u16.qacc.ld.xp q6, a14, a2, q4, q0
# CHECK: ee.vmulas.u16.qacc.ld.xp	 q6, a14, a2, q4, q0 # encoding: [0x2e,0x01,0x6b,0x1e]
ee.vmulas.u16.qacc.ld.xp.qup q6, a12, a11, q6, q7, q4, q1
# CHECK: ee.vmulas.u16.qacc.ld.xp.qup	 q6, a12, a11, q6, q7, q4, q1 # encoding: [0xbc,0xbc,0xe3,0x18]
ee.vmulas.u8.accx q2, q1
# CHECK: ee.vmulas.u8.accx	 q2, q1         # encoding: [0xc4,0x0a,0x0a]
ee.vmulas.u8.accx.ld.ip q6, a3, -112, q2, q7
# CHECK: ee.vmulas.u8.accx.ld.ip	 q6, a3, -112, q2, q7 # encoding: [0x93,0xb8,0xec,0x1f]
ee.vmulas.u8.accx.ld.ip.qup q7, a3, -32, q3, q3, q7, q5
# CHECK: ee.vmulas.u8.accx.ld.ip.qup	 q7, a3, -32, q3, q3, q7, q5 # encoding: [0xe3,0xdf,0xfa,0x0d]
ee.vmulas.u8.accx.ld.xp q4, a4, a9, q4, q0
# CHECK: ee.vmulas.u8.accx.ld.xp	 q4, a4, a9, q4, q0 # encoding: [0x94,0x01,0x4d,0x1e]
ee.vmulas.u8.accx.ld.xp.qup q5, a7, a13, q4, q7, q2, q6
# CHECK: ee.vmulas.u8.accx.ld.xp.qup	 q5, a7, a13, q4, q7, q2, q6 # encoding: [0xd7,0x3a,0x5d,0x19]
ee.vmulas.u8.qacc q3, q6
# CHECK: ee.vmulas.u8.qacc	 q3, q6         # encoding: [0xc4,0x73,0x0a]
ee.vmulas.u8.qacc.ldbc.incp q4, a1, q0, q5
# CHECK: ee.vmulas.u8.qacc.ldbc.incp	 q4, a1, q0, q5 # encoding: [0x14,0x48,0xf7]
ee.vmulas.u8.qacc.ldbc.incp.qup q2, a1, q5, q7, q6, q4
# CHECK: ee.vmulas.u8.qacc.ldbc.incp.qup	 q2, a1, q5, q7, q6, q4 # encoding: [0xb1,0x7e,0x29,0x1c]
ee.vmulas.u8.qacc.ld.ip q2, a12, 32, q1, q4
# CHECK: ee.vmulas.u8.qacc.ld.ip	 q2, a12, 32, q1, q4 # encoding: [0x2c,0x60,0x2e,0x1e]
ee.vmulas.u8.qacc.ld.ip.qup q0, a6, 48, q0, q0, q6, q0
# CHECK: ee.vmulas.u8.qacc.ld.ip.qup	 q0, a6, 48, q0, q0, q6, q0 # encoding: [0x36,0x06,0x00,0x0e]
ee.vmulas.u8.qacc.ld.xp q6, a1, a1, q2, q5
# CHECK: ee.vmulas.u8.qacc.ld.xp	 q6, a1, a1, q2, q5 # encoding: [0x11,0xa9,0x6e,0x1e]
ee.vmulas.u8.qacc.ld.xp.qup q1, a8, a10, q3, q7, q1, q3
# CHECK: ee.vmulas.u8.qacc.ld.xp.qup	 q1, a8, a10, q3, q7, q1, q3 # encoding: [0xa8,0xf9,0x96,0x19]
ee.vmul.s16 q0, q4, q1
# CHECK: ee.vmul.s16	 q0, q4, q1             # encoding: [0x84,0x2c,0x8e]
ee.vmul.s16.ld.incp q4, a5, q1, q5, q5
# CHECK: ee.vmul.s16.ld.incp	 q4, a5, q1, q5, q5 # encoding: [0xf5,0x6b,0x43,0x1c]
ee.vmul.s16.st.incp q4, a4, q2, q5, q0
# CHECK: ee.vmul.s16.st.incp	 q4, a4, q2, q5, q0 # encoding: [0x24,0x44,0xb5,0x1c]
ee.vmul.s8 q5, q3, q2
# CHECK: ee.vmul.s8	 q5, q3, q2             # encoding: [0x94,0xb3,0xae]
ee.vmul.s8.ld.incp q6, a11, q3, q6, q4
# CHECK: ee.vmul.s8.ld.incp	 q6, a11, q3, q6, q4 # encoding: [0xcb,0xa4,0x67,0x1c]
ee.vmul.s8.st.incp q5, a5, q5, q2, q4
# CHECK: ee.vmul.s8.st.incp	 q5, a5, q5, q2, q4 # encoding: [0x35,0xa5,0xaa,0x1c]
ee.vmul.u16 q0, q0, q5
# CHECK: ee.vmul.u16	 q0, q0, q5             # encoding: [0xa4,0x68,0x8e]
ee.vmul.u16.ld.incp q4, a2, q0, q1, q1
# CHECK: ee.vmul.u16.ld.incp	 q4, a2, q0, q1, q1 # encoding: [0xc2,0x4d,0x40,0x1c]
ee.vmul.u16.st.incp q6, a5, q1, q2, q7
# CHECK: ee.vmul.u16.st.incp	 q6, a5, q1, q2, q7 # encoding: [0x35,0xbe,0xb2,0x1c]
ee.vmul.u8 q6, q4, q5
# CHECK: ee.vmul.u8	 q6, q4, q5             # encoding: [0xb4,0x6c,0xbe]
ee.vmul.u8.ld.incp q1, a5, q4, q1, q1
# CHECK: ee.vmul.u8.ld.incp	 q1, a5, q4, q1, q1 # encoding: [0xc5,0x4e,0x18,0x1c]
ee.vmul.u8.st.incp q4, a12, q5, q0, q4
# CHECK: ee.vmul.u8.st.incp	 q4, a12, q5, q0, q4 # encoding: [0x0c,0x24,0x1a,0x1d]
ee.vprelu.s16 q2, q7, q0, a1
# CHECK: ee.vprelu.s16	 q2, q7, q0, a1         # encoding: [0x14,0x07,0x9c]
ee.vprelu.s8 q5, q6, q5, a13
# CHECK: ee.vprelu.s8	 q5, q6, q5, a13        # encoding: [0xd4,0xee,0xac]
ee.vrelu.s16 q2, a14, a5
# CHECK: ee.vrelu.s16	 q2, a14, a5            # encoding: [0x54,0x1e,0xdd]
ee.vrelu.s8 q4, a14, a1
# CHECK: ee.vrelu.s8	 q4, a14, a1            # encoding: [0x14,0x5e,0xed]
ee.vsl.32 q0, q1
# CHECK: ee.vsl.32	 q0, q1                 # encoding: [0x04,0xbf,0xcd]
ee.vsmulas.s16.qacc q2, q7, 2
# CHECK: ee.vsmulas.s16.qacc	 q2, q7, 2      # encoding: [0xc4,0x7a,0x9e]
ee.vsmulas.s16.qacc.ld.incp q7, a3, q3, q4, 3
# CHECK: ee.vsmulas.s16.qacc.ld.incp	 q7, a3, q3, q4, 3 # encoding: [0xc3,0xe7,0x76,0x1c]
ee.vsmulas.s8.qacc q3, q6, 3
# CHECK: ee.vsmulas.s8.qacc	 q3, q6, 3      # encoding: [0x54,0xd3,0x8e]
ee.vsmulas.s8.qacc.ld.incp q1, a8, q1, q1, 4
# CHECK: ee.vsmulas.s8.qacc.ld.incp	 q1, a8, q1, q1, 4 # encoding: [0xc8,0x4a,0x14,0x1c]
ee.vsr.32 q4, q3
# CHECK: ee.vsr.32	 q4, q3                 # encoding: [0xc4,0xbf,0xdd]
ee.vst.128.ip q3, a6, -816
# CHECK: ee.vst.128.ip	 q3, a6, -816           # encoding: [0x64,0xcd,0xda]
ee.vst.128.xp q6, a12, a14
# CHECK: ee.vst.128.xp	 q6, a12, a14           # encoding: [0xc4,0x7e,0xbd]
ee.vst.h.64.ip q2, a5, 40
# CHECK: ee.vst.h.64.ip	 q2, a5, 40             # encoding: [0x54,0x05,0x9b]
ee.vst.h.64.xp q2, a13, a6
# CHECK: ee.vst.h.64.xp	 q2, a13, a6            # encoding: [0xd4,0x06,0xdd]
ee.vst.l.64.ip q5, a8, 16
# CHECK: ee.vst.l.64.ip	 q5, a8, 16             # encoding: [0x84,0x82,0xa4]
ee.vst.l.64.xp q0, a13, a6
# CHECK: ee.vst.l.64.xp	 q0, a13, a6            # encoding: [0xd4,0x46,0xcd]
ee.vsubs.s16 q5, q1, q4
# CHECK: ee.vsubs.s16	 q5, q1, q4             # encoding: [0xd4,0xe1,0xae]
ee.vsubs.s16.ld.incp q1, a4, q6, q0, q1
# CHECK: ee.vsubs.s16.ld.incp	 q1, a4, q6, q0, q1 # encoding: [0xd4,0x0c,0x1c,0x1c]
ee.vsubs.s16.st.incp q7, a13, q7, q5, q2
# CHECK: ee.vsubs.s16.st.incp	 q7, a13, q7, q5, q2 # encoding: [0x1d,0x57,0x1f,0x1d]
ee.vsubs.s32 q2, q7, q6
# CHECK: ee.vsubs.s32	 q2, q7, q6             # encoding: [0xe4,0x77,0x9e]
ee.vsubs.s32.ld.incp q1, a8, q1, q4, q0
# CHECK: ee.vsubs.s32.ld.incp	 q1, a8, q1, q4, q0 # encoding: [0xd8,0x05,0x13,0x1c]
ee.vsubs.s32.st.incp q1, a5, q7, q4, q0
# CHECK: ee.vsubs.s32.st.incp	 q1, a5, q7, q4, q0 # encoding: [0x25,0x01,0x1f,0x1d]
ee.vsubs.s8 q7, q1, q5
# CHECK: ee.vsubs.s8	 q7, q1, q5             # encoding: [0xf4,0xe9,0xbe]
ee.vsubs.s8.ld.incp q4, a2, q6, q1, q6
# CHECK: ee.vsubs.s8.ld.incp	 q4, a2, q6, q1, q6 # encoding: [0xd2,0x76,0x4c,0x1c]
ee.vsubs.s8.st.incp q6, a1, q6, q2, q3
# CHECK: ee.vsubs.s8.st.incp	 q6, a1, q6, q2, q3 # encoding: [0x31,0x9e,0x1c,0x1d]
ee.vunzip.16 q6, q5
# CHECK: ee.vunzip.16	 q6, q5                 # encoding: [0x84,0xe3,0xec]
ee.vunzip.32 q0, q6
# CHECK: ee.vunzip.32	 q0, q6                 # encoding: [0x94,0x03,0xfc]
ee.vunzip.8 q5, q1
# CHECK: ee.vunzip.8	 q5, q1                 # encoding: [0xa4,0xd3,0xcc]
ee.vzip.16 q2, q0
# CHECK: ee.vzip.16	 q2, q0                 # encoding: [0xb4,0x23,0xcc]
ee.vzip.32 q0, q3
# CHECK: ee.vzip.32	 q0, q3                 # encoding: [0xc4,0x83,0xdc]
ee.vzip.8 q4, q5
# CHECK: ee.vzip.8	 q4, q5                 # encoding: [0xd4,0xc3,0xec]
ee.xorq q1, q3, q4
# CHECK: ee.xorq	 q1, q3, q4                     # encoding: [0x54,0xb9,0xcd]
ee.zero.accx
# CHECK: ee.zero.accx	                        # encoding: [0x04,0x08,0x25]
ee.zero.q q0
# CHECK: ee.zero.q	 q0                     # encoding: [0xa4,0x7f,0xcd]
ee.zero.qacc
# CHECK: ee.zero.qacc	                        # encoding: [0x44,0x08,0x25]

rur.accx_0 a11
# CHECK: rur	a11, accx                       # encoding: [0xd0,0xbe,0xe3]
rur.accx_1 a11
# CHECK: rur	a11, accx                       # encoding: [0xd0,0xbe,0xe3]
rur.fft_bit_width a11
# CHECK: rur	a11, fft_bit_width              # encoding: [0xf0,0xbe,0xe3]
rur.gpio_out a3
# CHECK: rur	a3, gpio_out                    # encoding: [0x00,0x30,0xe3]
rur.qacc_h_0 a1
# CHECK: rur	a1, qacc                        # encoding: [0xe0,0x1e,0xe3]
rur.qacc_h_1 a10
# CHECK: rur	a10, qacc                       # encoding: [0xe0,0xae,0xe3]
rur.qacc_h_2 a2
# CHECK: rur	a2, qacc                        # encoding: [0xe0,0x2e,0xe3]
rur.qacc_h_3 a11
# CHECK: rur	a11, qacc                       # encoding: [0xe0,0xbe,0xe3]
rur.qacc_h_4 a13
# CHECK: rur	a13, qacc                       # encoding: [0xe0,0xde,0xe3]
rur.qacc_l_0 a8
# CHECK: rur	a8, qacc                        # encoding: [0xe0,0x8e,0xe3]
rur.qacc_l_1 a7
# CHECK: rur	a7, qacc                        # encoding: [0xe0,0x7e,0xe3]
rur.qacc_l_2 a2
# CHECK: rur	a2, qacc                        # encoding: [0xe0,0x2e,0xe3]
rur.qacc_l_3 a13
# CHECK: rur	a13, qacc                       # encoding: [0xe0,0xde,0xe3]
rur.qacc_l_4 a7
# CHECK: rur	a7, qacc                        # encoding: [0xe0,0x7e,0xe3]
rur.sar_byte a9
# CHECK: rur	a9, sar_byte                    # encoding: [0x00,0x9f,0xe3]
rur.ua_state_0 a12
# CHECK: rur	a12, ua_state                   # encoding: [0x10,0xcf,0xe3]
rur.ua_state_1 a2
# CHECK: rur	a2, ua_state                    # encoding: [0x10,0x2f,0xe3]
rur.ua_state_2 a5
# CHECK: rur	a5, ua_state                    # encoding: [0x10,0x5f,0xe3]
rur.ua_state_3 a3
# CHECK: rur	a3, ua_state                    # encoding: [0x10,0x3f,0xe3]
wur.accx_0 a6
# CHECK: wur	a6, accx                        # encoding: [0x60,0xed,0xf3]
wur.accx_1 a6
# CHECK: wur	a6, accx                        # encoding: [0x60,0xed,0xf3]
wur.fft_bit_width a13
# CHECK: wur	a13, fft_bit_width              # encoding: [0xd0,0xef,0xf3]
wur.gpio_out a0
# CHECK: wur	a0, gpio_out                    # encoding: [0x00,0x00,0xf3]
wur.qacc_h_0 a12
# CHECK: wur	a12, qacc                       # encoding: [0xc0,0xee,0xf3]
wur.qacc_h_1 a1
# CHECK: wur	a1, qacc                        # encoding: [0x10,0xee,0xf3]
wur.qacc_h_2 a2
# CHECK: wur	a2, qacc                        # encoding: [0x20,0xee,0xf3]
wur.qacc_h_3 a12
# CHECK: wur	a12, qacc                       # encoding: [0xc0,0xee,0xf3]
wur.qacc_h_4 a14
# CHECK: wur	a14, qacc                       # encoding: [0xe0,0xee,0xf3]
wur.qacc_l_0 a6
# CHECK: wur	a6, qacc                        # encoding: [0x60,0xee,0xf3]
wur.qacc_l_1 a5
# CHECK: wur	a5, qacc                        # encoding: [0x50,0xee,0xf3]
wur.qacc_l_2 a6
# CHECK: wur	a6, qacc                        # encoding: [0x60,0xee,0xf3]
wur.qacc_l_3 a6
# CHECK: wur	a6, qacc                        # encoding: [0x60,0xee,0xf3]
wur.qacc_l_4 a7
# CHECK: wur	a7, qacc                        # encoding: [0x70,0xee,0xf3]
wur.sar_byte a9
# CHECK: wur	a9, sar_byte                    # encoding: [0x90,0xf0,0xf3]
wur.ua_state_0 a8
# CHECK: wur	a8, ua_state                    # encoding: [0x80,0xf1,0xf3]
wur.ua_state_1 a14
# CHECK: wur	a14, ua_state                   # encoding: [0xe0,0xf1,0xf3]
wur.ua_state_2 a9
# CHECK: wur	a9, ua_state                    # encoding: [0x90,0xf1,0xf3]
wur.ua_state_3 a10
# CHECK: wur	a10, ua_state                   # encoding: [0xa0,0xf1,0xf3]
mv.qr q4, q7
# CHECK: mv.qr	 q4, q7                         # encoding: [0x24,0x0c,0xaf]
