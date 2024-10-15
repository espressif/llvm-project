# RUN: llvm-mc %s -triple=riscv32 -mcpu=esp32p4 -show-encoding | FileCheck -check-prefixes=CHECK %s

esp.vcmulas.s16.qacc.h q2, q2
# CHECK: esp.vcmulas.s16.qacc.h	 q2, q2         # encoding: [0x5f,0x80,0x87,0x4a]
esp.vcmulas.s16.qacc.h.ld.ip q5, a3, -16, q0, q4
# CHECK: esp.vcmulas.s16.qacc.h.ld.ip	 q5, a3, -16, q0, q4 # encoding: [0x3b,0xf7,0xfa,0x13]
esp.vcmulas.s16.qacc.h.ld.xp q0, a4, a5, q2, q3
# CHECK: esp.vcmulas.s16.qacc.h.ld.xp	 q0, a4, a5, q2, q3 # encoding: [0x7f,0x20,0x73,0x4f]
esp.vcmulas.s16.qacc.l q3, q5
# CHECK: esp.vcmulas.s16.qacc.l	 q3, q5         # encoding: [0x5f,0x80,0x83,0x76]
esp.vcmulas.s16.qacc.l.ld.ip q6, a1, -32, q6, q5
# CHECK: esp.vcmulas.s16.qacc.l.ld.ip	 q6, a1, -32, q6, q5 # encoding: [0x3b,0xfb,0xf1,0xd5]
esp.vcmulas.s16.qacc.l.ld.xp q6, a2, a0, q4, q5
# CHECK: esp.vcmulas.s16.qacc.l.ld.xp	 q6, a2, a0, q4, q5 # encoding: [0x7f,0x38,0x22,0x95]
esp.vcmulas.s8.qacc.h q4, q2
# CHECK: esp.vcmulas.s8.qacc.h	 q4, q2         # encoding: [0x5f,0x80,0x85,0x8a]
esp.vcmulas.s8.qacc.h.ld.ip q5, a1, 96, q3, q4
# CHECK: esp.vcmulas.s8.qacc.h.ld.ip	 q5, a1, 96, q3, q4 # encoding: [0x3b,0xf7,0xb1,0x72]
esp.vcmulas.s8.qacc.h.ld.xp q6, a1, a3, q1, q6
# CHECK: esp.vcmulas.s8.qacc.h.ld.xp	 q6, a1, a3, q1, q6 # encoding: [0x7f,0xb8,0x51,0x3a]
esp.vcmulas.s8.qacc.l q0, q2
# CHECK: esp.vcmulas.s8.qacc.l	 q0, q2         # encoding: [0x5f,0x80,0x81,0x0a]
esp.vcmulas.s8.qacc.l.ld.ip q4, a2, -128, q0, q2
# CHECK: esp.vcmulas.s8.qacc.l.ld.ip	 q4, a2, -128, q0, q2 # encoding: [0x3b,0x73,0xc2,0x08]
esp.vcmulas.s8.qacc.l.ld.xp q0, a5, a1, q2, q2
# CHECK: esp.vcmulas.s8.qacc.l.ld.xp	 q0, a5, a1, q2, q2 # encoding: [0x7f,0xa0,0x33,0x48]
esp.vmulas.s16.qacc q2, q3
# CHECK: esp.vmulas.s16.qacc	 q2, q3         # encoding: [0x5f,0x00,0xc7,0x4e]
esp.vmulas.s16.qacc.ld.ip q1, a5, -112, q1, q4
# CHECK: esp.vmulas.s16.qacc.ld.ip	 q1, a5, -112, q1, q4 # encoding: [0xbb,0xe6,0xe3,0x32]
esp.vmulas.s16.qacc.ld.xp q4, a4, a1, q6, q3
# CHECK: esp.vmulas.s16.qacc.ld.xp	 q4, a4, a1, q6, q3 # encoding: [0xff,0x32,0x3b,0xce]
esp.vmulas.s16.qacc.st.ip q6, a5, -96, q5, q2
# CHECK: esp.vmulas.s16.qacc.st.ip	 q6, a5, -96, q5, q2 # encoding: [0xbb,0xf8,0xeb,0xab]
esp.vmulas.s16.qacc.st.xp q4, a2, a5, q3, q1
# CHECK: esp.vmulas.s16.qacc.st.xp	 q4, a2, a5, q3, q1 # encoding: [0xff,0x32,0x7a,0x67]
esp.vmulas.s16.xacc q5, q2
# CHECK: esp.vmulas.s16.xacc	 q5, q2         # encoding: [0x5f,0x00,0xc3,0xaa]
esp.vmulas.s16.xacc.ld.ip q4, a3, 48, q3, q6
# CHECK: esp.vmulas.s16.xacc.ld.ip	 q4, a3, 48, q3, q6 # encoding: [0xbb,0xf2,0xca,0x78]
esp.vmulas.s16.xacc.ld.xp q6, a2, a5, q0, q0
# CHECK: esp.vmulas.s16.xacc.ld.xp	 q6, a2, a5, q0, q0 # encoding: [0xff,0x3a,0x7a,0x00]
esp.vmulas.s16.xacc.st.ip q1, a1, 16, q4, q3
# CHECK: esp.vmulas.s16.xacc.st.ip	 q1, a1, 16, q4, q3 # encoding: [0xbb,0xe6,0xc1,0x8d]
esp.vmulas.s16.xacc.st.xp q2, a4, a0, q5, q0
# CHECK: esp.vmulas.s16.xacc.st.xp	 q2, a4, a0, q5, q0 # encoding: [0xff,0x2a,0x2b,0xa1]
esp.vmulas.s8.qacc q4, q6
# CHECK: esp.vmulas.s8.qacc	 q4, q6         # encoding: [0x5f,0x00,0xc5,0x9a]
esp.vmulas.s8.qacc.ld.ip q5, a2, 64, q5, q0
# CHECK: esp.vmulas.s8.qacc.ld.ip	 q5, a2, 64, q5, q0 # encoding: [0xbb,0x74,0x52,0xa2]
esp.vmulas.s8.qacc.ld.xp q1, a1, a3, q5, q0
# CHECK: esp.vmulas.s8.qacc.ld.xp	 q1, a1, a3, q5, q0 # encoding: [0xff,0xa6,0x51,0xa2]
esp.vmulas.s8.qacc.st.ip q5, a3, 16, q5, q1
# CHECK: esp.vmulas.s8.qacc.st.ip	 q5, a3, 16, q5, q1 # encoding: [0xbb,0xf6,0x42,0xa7]
esp.vmulas.s8.qacc.st.xp q5, a1, a4, q4, q0
# CHECK: esp.vmulas.s8.qacc.st.xp	 q5, a1, a4, q4, q0 # encoding: [0xff,0xb6,0x61,0x83]
esp.vmulas.s8.xacc q1, q0
# CHECK: esp.vmulas.s8.xacc	 q1, q0         # encoding: [0x5f,0x00,0xc1,0x22]
esp.vmulas.s8.xacc.ld.ip q0, a4, 16, q4, q4
# CHECK: esp.vmulas.s8.xacc.ld.ip	 q0, a4, 16, q4, q4 # encoding: [0xbb,0x62,0x43,0x90]
esp.vmulas.s8.xacc.ld.xp q0, a5, a2, q4, q2
# CHECK: esp.vmulas.s8.xacc.ld.xp	 q0, a5, a2, q4, q2 # encoding: [0xff,0xa2,0x43,0x88]
esp.vmulas.s8.xacc.st.ip q3, a3, -32, q0, q5
# CHECK: esp.vmulas.s8.xacc.st.ip	 q3, a3, -32, q0, q5 # encoding: [0xbb,0xec,0x7a,0x15]
esp.vmulas.s8.xacc.st.xp q0, a5, a4, q3, q3
# CHECK: esp.vmulas.s8.xacc.st.xp	 q0, a5, a4, q3, q3 # encoding: [0xff,0xa2,0x63,0x6d]
esp.vmulas.u16.qacc q2, q0
# CHECK: esp.vmulas.u16.qacc	 q2, q0         # encoding: [0x5f,0x00,0xc6,0x42]
esp.vmulas.u16.qacc.ld.ip q5, a4, 80, q1, q3
# CHECK: esp.vmulas.u16.qacc.ld.ip	 q5, a4, 80, q1, q3 # encoding: [0xbb,0x76,0x93,0x2e]
esp.vmulas.u16.qacc.ld.xp q5, a5, a3, q6, q3
# CHECK: esp.vmulas.u16.qacc.ld.xp	 q5, a5, a3, q6, q3 # encoding: [0xff,0xb4,0x5b,0xce]
esp.vmulas.u16.qacc.st.ip q5, a3, -80, q2, q0
# CHECK: esp.vmulas.u16.qacc.st.ip	 q5, a3, -80, q2, q0 # encoding: [0xbb,0xf6,0xaa,0x43]
esp.vmulas.u16.qacc.st.xp q0, a2, a4, q5, q5
# CHECK: esp.vmulas.u16.qacc.st.xp	 q0, a2, a4, q5, q5 # encoding: [0xff,0x20,0x6a,0xb7]
esp.vmulas.u16.xacc q2, q6
# CHECK: esp.vmulas.u16.xacc	 q2, q6         # encoding: [0x5f,0x00,0xc2,0x5a]
esp.vmulas.u16.xacc.ld.ip q1, a3, 16, q5, q2
# CHECK: esp.vmulas.u16.xacc.ld.ip	 q1, a3, 16, q5, q2 # encoding: [0xbb,0xe6,0x82,0xa8]
esp.vmulas.u16.xacc.ld.xp q2, a3, a4, q1, q3
# CHECK: esp.vmulas.u16.xacc.ld.xp	 q2, a3, a4, q1, q3 # encoding: [0xff,0xa8,0x6a,0x2c]
esp.vmulas.u16.xacc.st.ip q3, a4, -112, q1, q3
# CHECK: esp.vmulas.u16.xacc.st.ip	 q3, a4, -112, q1, q3 # encoding: [0xbb,0x6e,0xa3,0x2d]
esp.vmulas.u16.xacc.st.xp q4, a0, a2, q0, q3
# CHECK: esp.vmulas.u16.xacc.st.xp	 q4, a0, a2, q0, q3 # encoding: [0xff,0x30,0x49,0x0d]
esp.vmulas.u8.qacc q6, q1
# CHECK: esp.vmulas.u8.qacc	 q6, q1         # encoding: [0x5f,0x00,0xc4,0xc6]
esp.vmulas.u8.qacc.ld.ip q0, a4, -80, q0, q3
# CHECK: esp.vmulas.u8.qacc.ld.ip	 q0, a4, -80, q0, q3 # encoding: [0xbb,0x62,0x2b,0x0e]
esp.vmulas.u8.qacc.ld.xp q0, a3, a0, q1, q3
# CHECK: esp.vmulas.u8.qacc.ld.xp	 q0, a3, a0, q1, q3 # encoding: [0xff,0xa0,0x22,0x2e]
esp.vmulas.u8.qacc.st.ip q2, a3, 64, q0, q0
# CHECK: esp.vmulas.u8.qacc.st.ip	 q2, a3, 64, q0, q0 # encoding: [0xbb,0xe8,0x12,0x03]
esp.vmulas.u8.qacc.st.xp q6, a2, a2, q3, q1
# CHECK: esp.vmulas.u8.qacc.st.xp	 q6, a2, a2, q3, q1 # encoding: [0xff,0x38,0x42,0x67]
esp.vmulas.u8.xacc q3, q3
# CHECK: esp.vmulas.u8.xacc	 q3, q3         # encoding: [0x5f,0x00,0xc0,0x6e]
esp.vmulas.u8.xacc.ld.ip q0, a5, 16, q0, q1
# CHECK: esp.vmulas.u8.xacc.ld.ip	 q0, a5, 16, q0, q1 # encoding: [0xbb,0xe2,0x03,0x04]
esp.vmulas.u8.xacc.ld.xp q4, a0, a2, q1, q5
# CHECK: esp.vmulas.u8.xacc.ld.xp	 q4, a0, a2, q1, q5 # encoding: [0xff,0x30,0x41,0x34]
esp.vmulas.u8.xacc.st.ip q4, a1, -48, q4, q6
# CHECK: esp.vmulas.u8.xacc.st.ip	 q4, a1, -48, q4, q6 # encoding: [0xbb,0xf2,0x31,0x99]
esp.vmulas.u8.xacc.st.xp q4, a2, a3, q0, q2
# CHECK: esp.vmulas.u8.xacc.st.xp	 q4, a2, a3, q0, q2 # encoding: [0xff,0x30,0x52,0x09]
esp.vmulas.s16.qacc.ldbc.incp q5, a2, q0, q6
# CHECK: esp.vmulas.s16.qacc.ldbc.incp	 q5, a2, q0, q6 # encoding: [0xbb,0x75,0x62,0x18]
esp.vmulas.s8.qacc.ldbc.incp q3, a5, q4, q2
# CHECK: esp.vmulas.s8.qacc.ldbc.incp	 q3, a5, q4, q2 # encoding: [0xbb,0xed,0x23,0x88]
esp.vmulas.u16.qacc.ldbc.incp q2, a1, q1, q3
# CHECK: esp.vmulas.u16.qacc.ldbc.incp	 q2, a1, q1, q3 # encoding: [0xbb,0xe9,0x41,0x2c]
esp.vmulas.u8.qacc.ldbc.incp q0, a1, q2, q0
# CHECK: esp.vmulas.u8.qacc.ldbc.incp	 q0, a1, q2, q0 # encoding: [0xbb,0xe1,0x01,0x40]
esp.vsmulas.s16.qacc q0, q5, 5
# CHECK: esp.vsmulas.s16.qacc	 q0, q5, 5      # encoding: [0x5f,0x80,0xf2,0x16]
esp.vsmulas.s16.qacc.ld.incp q1, a2, q4, q2, 4
# CHECK: esp.vsmulas.s16.qacc.ld.incp	 q1, a2, q4, q2, 4 # encoding: [0xbb,0x67,0xa2,0x8b]
esp.vsmulas.s8.qacc q2, q1, 13
# CHECK: esp.vsmulas.s8.qacc	 q2, q1, 13     # encoding: [0x5f,0x80,0xb6,0x46]
esp.vsmulas.s8.qacc.ld.incp q2, a5, q1, q3, 0
# CHECK: esp.vsmulas.s8.qacc.ld.incp	 q2, a5, q1, q3, 0 # encoding: [0xbb,0xeb,0x83,0x2d]
esp.vsmulas.u16.qacc q6, q1, 5
# CHECK: esp.vsmulas.u16.qacc	 q6, q1, 5      # encoding: [0x5f,0x80,0xd2,0xc6]
esp.vsmulas.u16.qacc.ld.incp q0, a0, q6, q6, 0
# CHECK: esp.vsmulas.u16.qacc.ld.incp	 q0, a0, q6, q6, 0 # encoding: [0xbb,0x63,0x81,0xda]
esp.vsmulas.u8.qacc q0, q3, 7
# CHECK: esp.vsmulas.u8.qacc	 q0, q3, 7      # encoding: [0x5f,0x80,0x93,0x0e]
esp.vsmulas.u8.qacc.ld.incp q6, a0, q6, q5, 8
# CHECK: esp.vsmulas.u8.qacc.ld.incp	 q6, a0, q6, q5, 8 # encoding: [0xbb,0x7b,0xc1,0xd4]
esp.cmul.s16 q0, q2, q4, 3
# CHECK: esp.cmul.s16	 q0, q2, q4, 3          # encoding: [0x5f,0xa4,0x07,0x50]
esp.cmul.s16.ld.incp q6, a4, q1, q1, q5, 0
# CHECK: esp.cmul.s16.ld.incp	 q6, a4, q1, q1, q5, 0 # encoding: [0xbf,0x58,0xc3,0x34]
esp.cmul.s16.st.incp q4, a0, q0, q5, q0, 0
# CHECK: esp.cmul.s16.st.incp	 q4, a0, q0, q5, q0, 0 # encoding: [0x3f,0x50,0xc1,0xa2]
esp.cmul.s8 q6, q1, q6, 2
# CHECK: esp.cmul.s8	 q6, q1, q6, 2          # encoding: [0x5f,0x27,0x03,0x38]
esp.cmul.s8.ld.incp q4, a3, q0, q4, q2, 3
# CHECK: esp.cmul.s8.ld.incp	 q4, a3, q0, q4, q2, 3 # encoding: [0x3f,0xd0,0x72,0x88]
esp.cmul.s8.st.incp q5, a1, q5, q2, q0, 3
# CHECK: esp.cmul.s8.st.incp	 q5, a1, q5, q2, q0, 3 # encoding: [0xbf,0xd6,0x71,0x42]
esp.cmul.u16 q2, q3, q5, 1
# CHECK: esp.cmul.u16	 q2, q3, q5, 1          # encoding: [0x5f,0xa5,0x04,0x74]
esp.cmul.u16.ld.incp q1, a1, q6, q5, q1, 0
# CHECK: esp.cmul.u16.ld.incp	 q1, a1, q6, q5, q1, 0 # encoding: [0x3f,0xc7,0x81,0xa4]
esp.cmul.u16.st.incp q6, a1, q2, q2, q4, 0
# CHECK: esp.cmul.u16.st.incp	 q6, a1, q2, q2, q4, 0 # encoding: [0x3f,0xd9,0x81,0x52]
esp.cmul.u8 q1, q4, q3, 3
# CHECK: esp.cmul.u8	 q1, q4, q3, 3          # encoding: [0xdf,0xa4,0x01,0x8c]
esp.cmul.u8.ld.incp q2, a5, q3, q0, q5, 1
# CHECK: esp.cmul.u8.ld.incp	 q2, a5, q3, q0, q5, 1 # encoding: [0xbf,0xc9,0x13,0x14]
esp.cmul.u8.st.incp q4, a2, q0, q4, q4, 0
# CHECK: esp.cmul.u8.st.incp	 q4, a2, q0, q4, q4, 0 # encoding: [0x3f,0x50,0x02,0x92]
esp.max.s16.a q5, a2
# CHECK: esp.max.s16.a	 q5, a2                 # encoding: [0x5b,0x52,0xc8,0x91]
esp.max.s32.a q5, a5
# CHECK: esp.max.s32.a	 q5, a5                 # encoding: [0xdb,0x53,0xa8,0x91]
esp.max.s8.a q2, a0
# CHECK: esp.max.s8.a	 q2, a0                 # encoding: [0x5b,0x51,0x40,0x92]
esp.max.u16.a q6, a3
# CHECK: esp.max.u16.a	 q6, a3                 # encoding: [0xdb,0x52,0x88,0x92]
esp.max.u32.a q0, a2
# CHECK: esp.max.u32.a	 q0, a2                 # encoding: [0x5b,0x52,0x20,0x90]
esp.max.u8.a q5, a0
# CHECK: esp.max.u8.a	 q5, a0                 # encoding: [0x5b,0x51,0x08,0x91]
esp.min.s16.a q0, a3
# CHECK: esp.min.s16.a	 q0, a3                 # encoding: [0xdb,0x52,0xd0,0x90]
esp.min.s32.a q1, a1
# CHECK: esp.min.s32.a	 q1, a1                 # encoding: [0xdb,0x51,0xb0,0x91]
esp.min.s8.a q3, a4
# CHECK: esp.min.s8.a	 q3, a4                 # encoding: [0x5b,0x53,0x50,0x93]
esp.min.u16.a q6, a0
# CHECK: esp.min.u16.a	 q6, a0                 # encoding: [0x5b,0x51,0x98,0x92]
esp.min.u32.a q2, a3
# CHECK: esp.min.u32.a	 q2, a3                 # encoding: [0xdb,0x52,0x30,0x92]
esp.min.u8.a q1, a3
# CHECK: esp.min.u8.a	 q1, a3                 # encoding: [0xdb,0x52,0x10,0x91]
esp.vabs.16 q5, q2
# CHECK: esp.vabs.16	 q5, q2                 # encoding: [0x5b,0x10,0x50,0x88]
esp.vabs.32 q0, q3
# CHECK: esp.vabs.32	 q0, q3                 # encoding: [0x5b,0x08,0x00,0x8c]
esp.vabs.8 q6, q1
# CHECK: esp.vabs.8	 q6, q1                 # encoding: [0x5b,0x00,0x60,0x84]
esp.vadd.s16 q1, q0, q3
# CHECK: esp.vadd.s16	 q1, q0, q3             # encoding: [0x5f,0x06,0x94,0x0e]
esp.vadd.s16.ld.incp q2, a3, q4, q3, q1
# CHECK: esp.vadd.s16.ld.incp	 q2, a3, q4, q3, q1 # encoding: [0x3b,0xe8,0x4a,0x65]
esp.vadd.s16.st.incp q6, a2, q1, q3, q1
# CHECK: esp.vadd.s16.st.incp	 q6, a2, q1, q3, q1 # encoding: [0x3b,0x78,0x1a,0x67]
esp.vadd.s32 q2, q5, q3
# CHECK: esp.vadd.s32	 q2, q5, q3             # encoding: [0x5f,0x05,0xa4,0xae]
esp.vadd.s32.ld.incp q5, a0, q0, q6, q2
# CHECK: esp.vadd.s32.ld.incp	 q5, a0, q0, q6, q2 # encoding: [0x3b,0x75,0x01,0xc9]
esp.vadd.s32.st.incp q5, a0, q2, q6, q1
# CHECK: esp.vadd.s32.st.incp	 q5, a0, q2, q6, q1 # encoding: [0x3b,0x75,0x21,0xc7]
esp.vadd.s8 q6, q4, q1
# CHECK: esp.vadd.s8	 q6, q4, q1             # encoding: [0x5f,0x06,0xe0,0x86]
esp.vadd.s8.ld.incp q6, a0, q3, q0, q4
# CHECK: esp.vadd.s8.ld.incp	 q6, a0, q3, q0, q4 # encoding: [0x3b,0x78,0x39,0x10]
esp.vadd.s8.st.incp q0, a5, q0, q0, q4
# CHECK: esp.vadd.s8.st.incp	 q0, a5, q0, q0, q4 # encoding: [0x3b,0xe0,0x0b,0x12]
esp.vadd.u16 q3, q5, q0
# CHECK: esp.vadd.u16	 q3, q5, q0             # encoding: [0x5f,0x04,0xb4,0xa2]
esp.vadd.u16.ld.incp q6, a1, q4, q6, q4
# CHECK: esp.vadd.u16.ld.incp	 q6, a1, q4, q6, q4 # encoding: [0x3b,0xf8,0x41,0xd1]
esp.vadd.u16.st.incp q5, a2, q4, q3, q5
# CHECK: esp.vadd.u16.st.incp	 q5, a2, q4, q3, q5 # encoding: [0x3b,0x74,0x42,0x77]
esp.vadd.u32 q5, q2, q2
# CHECK: esp.vadd.u32	 q5, q2, q2             # encoding: [0x5f,0x05,0xd0,0x4a]
esp.vadd.u32.ld.incp q3, a0, q6, q1, q2
# CHECK: esp.vadd.u32.ld.incp	 q3, a0, q6, q1, q2 # encoding: [0x3b,0x6d,0x61,0x28]
esp.vadd.u32.st.incp q1, a1, q2, q6, q6
# CHECK: esp.vadd.u32.st.incp	 q1, a1, q2, q6, q6 # encoding: [0x3b,0xe5,0x21,0xda]
esp.vadd.u8 q0, q0, q3
# CHECK: esp.vadd.u8	 q0, q0, q3             # encoding: [0x5f,0x04,0x80,0x0e]
esp.vadd.u8.ld.incp q3, a1, q0, q0, q1
# CHECK: esp.vadd.u8.ld.incp	 q3, a1, q0, q0, q1 # encoding: [0x3b,0xec,0x01,0x04]
esp.vadd.u8.st.incp q1, a1, q0, q4, q6
# CHECK: esp.vadd.u8.st.incp	 q1, a1, q0, q4, q6 # encoding: [0x3b,0xe4,0x01,0x9a]
esp.vclamp.s16 q0, q5, 4
# CHECK: esp.vclamp.s16	 q0, q5, 4              # encoding: [0x5b,0x50,0x00,0xa1]
esp.vmax.s16 q4, q1, q2
# CHECK: esp.vmax.s16	 q4, q1, q2             # encoding: [0x5f,0xae,0x06,0x28]
esp.vmax.s16.ld.incp q3, a5, q3, q4, q4
# CHECK: esp.vmax.s16.ld.incp	 q3, a5, q3, q4, q4 # encoding: [0xbf,0xcd,0x6b,0x90]
esp.vmax.s16.st.incp q4, a4, q0, q5, q6
# CHECK: esp.vmax.s16.st.incp	 q4, a4, q0, q5, q6 # encoding: [0x3f,0x50,0xeb,0xb8]
esp.vmax.s32 q4, q0, q4
# CHECK: esp.vmax.s32	 q4, q0, q4             # encoding: [0x5f,0xae,0x05,0x10]
esp.vmax.s32.ld.incp q2, a0, q2, q2, q1
# CHECK: esp.vmax.s32.ld.incp	 q2, a0, q2, q2, q1 # encoding: [0x3f,0x49,0x59,0x44]
esp.vmax.s32.st.incp q4, a1, q1, q4, q2
# CHECK: esp.vmax.s32.st.incp	 q4, a1, q1, q4, q2 # encoding: [0xbf,0xd0,0xd9,0x88]
esp.vmax.s8 q3, q1, q0
# CHECK: esp.vmax.s8	 q3, q1, q0             # encoding: [0xdf,0xad,0x02,0x20]
esp.vmax.s8.ld.incp q4, a3, q5, q2, q5
# CHECK: esp.vmax.s8.ld.incp	 q4, a3, q5, q2, q5 # encoding: [0xbf,0xd2,0x2a,0x54]
esp.vmax.s8.st.incp q0, a3, q3, q4, q5
# CHECK: esp.vmax.s8.st.incp	 q0, a3, q3, q4, q5 # encoding: [0xbf,0xc1,0xaa,0x94]
esp.vmax.u16 q6, q3, q2
# CHECK: esp.vmax.u16	 q6, q3, q2             # encoding: [0x5f,0xaf,0x04,0x68]
esp.vmax.u16.ld.incp q5, a1, q5, q5, q3
# CHECK: esp.vmax.u16.ld.incp	 q5, a1, q5, q5, q3 # encoding: [0xbf,0xd6,0x49,0xac]
esp.vmax.u16.st.incp q2, a4, q3, q2, q5
# CHECK: esp.vmax.u16.st.incp	 q2, a4, q3, q2, q5 # encoding: [0xbf,0x49,0xcb,0x54]
esp.vmax.u32 q1, q0, q3
# CHECK: esp.vmax.u32	 q1, q0, q3             # encoding: [0xdf,0xac,0x01,0x0c]
esp.vmax.u32.ld.incp q3, a1, q5, q0, q4
# CHECK: esp.vmax.u32.ld.incp	 q3, a1, q5, q0, q4 # encoding: [0xbf,0xce,0x19,0x10]
esp.vmax.u32.st.incp q1, a3, q2, q5, q5
# CHECK: esp.vmax.u32.st.incp	 q1, a3, q2, q5, q5 # encoding: [0x3f,0xc5,0x9a,0xb4]
esp.vmax.u8 q1, q4, q0
# CHECK: esp.vmax.u8	 q1, q4, q0             # encoding: [0xdf,0xac,0x00,0x80]
esp.vmax.u8.ld.incp q1, a2, q3, q4, q6
# CHECK: esp.vmax.u8.ld.incp	 q1, a2, q3, q4, q6 # encoding: [0xbf,0x45,0x0a,0x98]
esp.vmax.u8.st.incp q2, a3, q5, q3, q5
# CHECK: esp.vmax.u8.st.incp	 q2, a3, q5, q3, q5 # encoding: [0xbf,0xca,0x8a,0x74]
esp.vmin.s16 q5, q3, q2
# CHECK: esp.vmin.s16	 q5, q3, q2             # encoding: [0xdf,0x3e,0x06,0x68]
esp.vmin.s16.ld.incp q6, a3, q1, q4, q5
# CHECK: esp.vmin.s16.ld.incp	 q6, a3, q1, q4, q5 # encoding: [0xbf,0xd8,0x6a,0x95]
esp.vmin.s16.st.incp q0, a1, q5, q0, q3
# CHECK: esp.vmin.s16.st.incp	 q0, a1, q5, q0, q3 # encoding: [0xbf,0xc2,0xe9,0x0d]
esp.vmin.s32 q3, q1, q4
# CHECK: esp.vmin.s32	 q3, q1, q4             # encoding: [0xdf,0x3d,0x05,0x30]
esp.vmin.s32.ld.incp q4, a0, q1, q6, q3
# CHECK: esp.vmin.s32.ld.incp	 q4, a0, q1, q6, q3 # encoding: [0xbf,0x50,0x59,0xcd]
esp.vmin.s32.st.incp q6, a5, q4, q6, q2
# CHECK: esp.vmin.s32.st.incp	 q6, a5, q4, q6, q2 # encoding: [0x3f,0xda,0xdb,0xc9]
esp.vmin.s8 q3, q2, q1
# CHECK: esp.vmin.s8	 q3, q2, q1             # encoding: [0xdf,0x3d,0x02,0x44]
esp.vmin.s8.ld.incp q1, a4, q5, q5, q6
# CHECK: esp.vmin.s8.ld.incp	 q1, a4, q5, q5, q6 # encoding: [0xbf,0x46,0x2b,0xb9]
esp.vmin.s8.st.incp q5, a1, q6, q2, q0
# CHECK: esp.vmin.s8.st.incp	 q5, a1, q6, q2, q0 # encoding: [0x3f,0xd7,0xa9,0x41]
esp.vmin.u16 q5, q0, q1
# CHECK: esp.vmin.u16	 q5, q0, q1             # encoding: [0xdf,0x3e,0x04,0x04]
esp.vmin.u16.ld.incp q3, a5, q5, q3, q5
# CHECK: esp.vmin.u16.ld.incp	 q3, a5, q5, q3, q5 # encoding: [0xbf,0xce,0x4b,0x75]
esp.vmin.u16.st.incp q2, a3, q5, q6, q3
# CHECK: esp.vmin.u16.st.incp	 q2, a3, q5, q6, q3 # encoding: [0xbf,0xca,0xca,0xcd]
esp.vmin.u32 q5, q0, q2
# CHECK: esp.vmin.u32	 q5, q0, q2             # encoding: [0xdf,0x3e,0x01,0x08]
esp.vmin.u32.ld.incp q4, a5, q4, q4, q2
# CHECK: esp.vmin.u32.ld.incp	 q4, a5, q4, q4, q2 # encoding: [0x3f,0xd2,0x1b,0x89]
esp.vmin.u32.st.incp q4, a2, q1, q6, q3
# CHECK: esp.vmin.u32.st.incp	 q4, a2, q1, q6, q3 # encoding: [0xbf,0x50,0x9a,0xcd]
esp.vmin.u8 q0, q0, q0
# CHECK: esp.vmin.u8	 q0, q0, q0             # encoding: [0x5f,0x3c,0x00,0x00]
esp.vmin.u8.ld.incp q2, a5, q1, q0, q6
# CHECK: esp.vmin.u8.ld.incp	 q2, a5, q1, q0, q6 # encoding: [0xbf,0xc8,0x0b,0x19]
esp.vmin.u8.st.incp q1, a2, q0, q1, q1
# CHECK: esp.vmin.u8.st.incp	 q1, a2, q0, q1, q1 # encoding: [0x3f,0x44,0x8a,0x25]
esp.vmul.s16 q1, q2, q1
# CHECK: esp.vmul.s16	 q1, q2, q1             # encoding: [0xdf,0xbc,0x06,0x44]
esp.vmul.s16.ld.incp q4, a3, q1, q4, q2
# CHECK: esp.vmul.s16.ld.incp	 q4, a3, q1, q4, q2 # encoding: [0xbf,0xd0,0x6a,0x8b]
esp.vmul.s16.s8xs8 q4, q5, q1, q1
# CHECK: esp.vmul.s16.s8xs8	 q4, q5, q1, q1 # encoding: [0x5f,0x06,0xd3,0x26]
esp.vmul.s16.st.incp q2, a3, q1, q4, q0
# CHECK: esp.vmul.s16.st.incp	 q2, a3, q1, q4, q0 # encoding: [0xbf,0xc8,0xea,0x83]
esp.vmul.s32.s16xs16 q4, q5, q2, q1
# CHECK: esp.vmul.s32.s16xs16	 q4, q5, q2, q1 # encoding: [0x5f,0x06,0xd7,0x46]
esp.vmul.s8 q1, q4, q1
# CHECK: esp.vmul.s8	 q1, q4, q1             # encoding: [0xdf,0xbc,0x02,0x84]
esp.vmul.s8.ld.incp q0, a0, q4, q0, q4
# CHECK: esp.vmul.s8.ld.incp	 q0, a0, q4, q0, q4 # encoding: [0x3f,0x42,0x29,0x13]
esp.vmul.s8.st.incp q4, a3, q5, q6, q6
# CHECK: esp.vmul.s8.st.incp	 q4, a3, q5, q6, q6 # encoding: [0xbf,0xd2,0xaa,0xdb]
esp.vmul.u16 q6, q6, q1
# CHECK: esp.vmul.u16	 q6, q6, q1             # encoding: [0x5f,0xbf,0x04,0xc4]
esp.vmul.u16.ld.incp q1, a5, q5, q4, q6
# CHECK: esp.vmul.u16.ld.incp	 q1, a5, q5, q4, q6 # encoding: [0xbf,0xc6,0x4b,0x9b]
esp.vmul.u16.st.incp q3, a4, q5, q3, q3
# CHECK: esp.vmul.u16.st.incp	 q3, a4, q5, q3, q3 # encoding: [0xbf,0x4e,0xcb,0x6f]
esp.vmul.u8 q0, q1, q5
# CHECK: esp.vmul.u8	 q0, q1, q5             # encoding: [0x5f,0xbc,0x00,0x34]
esp.vmul.u8.ld.incp q5, a3, q0, q4, q5
# CHECK: esp.vmul.u8.ld.incp	 q5, a3, q0, q4, q5 # encoding: [0x3f,0xd4,0x0a,0x97]
esp.vmul.u8.st.incp q5, a0, q2, q4, q5
# CHECK: esp.vmul.u8.st.incp	 q5, a0, q2, q4, q5 # encoding: [0x3f,0x55,0x89,0x97]
esp.vprelu.s16 q0, q3, q2, a4
# CHECK: esp.vprelu.s16	 q0, q3, q2, a4         # encoding: [0x5f,0x60,0xa3,0x4e]
esp.vprelu.s8 q4, q5, q5, a3
# CHECK: esp.vprelu.s8	 q4, q5, q5, a3         # encoding: [0x5f,0xe2,0x22,0xb6]
esp.vrelu.s16 q1, a1, a4
# CHECK: esp.vrelu.s16	 q1, a1, a4             # encoding: [0x5b,0x5c,0x33,0x86]
esp.vrelu.s8 q6, a1, a5
# CHECK: esp.vrelu.s8	 q6, a1, a5             # encoding: [0x5b,0xd8,0x33,0x9a]
esp.vsadds.s16 q5, q6, a4
# CHECK: esp.vsadds.s16	 q5, q6, a4             # encoding: [0x5f,0x02,0xd3,0xda]
esp.vsadds.s8 q6, q1, a4
# CHECK: esp.vsadds.s8	 q6, q1, a4             # encoding: [0x5f,0x02,0xe3,0x2a]
esp.vsadds.u16 q0, q6, a2
# CHECK: esp.vsadds.u16	 q0, q6, a2             # encoding: [0x5f,0x02,0x82,0xd2]
esp.vsadds.u8 q3, q3, a1
# CHECK: esp.vsadds.u8	 q3, q3, a1             # encoding: [0x5f,0x82,0xb1,0x62]
esp.vsat.s16 q3, q6, a0, a4
# CHECK: esp.vsat.s16	 q3, q6, a0, a4         # encoding: [0xbb,0x59,0x61,0xd8]
esp.vsat.s32 q6, q4, a3, a2
# CHECK: esp.vsat.s32	 q6, q4, a3, a2         # encoding: [0x3b,0xd7,0x42,0x98]
esp.vsat.s8 q3, q2, a2, a1
# CHECK: esp.vsat.s8	 q3, q2, a2, a1         # encoding: [0xbb,0x49,0x32,0x58]
esp.vsat.u16 q5, q5, a0, a1
# CHECK: esp.vsat.u16	 q5, q5, a0, a1         # encoding: [0xbb,0x52,0x31,0xb8]
esp.vsat.u32 q6, q5, a3, a3
# CHECK: esp.vsat.u32	 q6, q5, a3, a3         # encoding: [0x3b,0xc7,0x52,0xb8]
esp.vsat.u8 q4, q1, a0, a3
# CHECK: esp.vsat.u8	 q4, q1, a0, a3         # encoding: [0x3b,0x42,0x51,0x38]
esp.vssubs.s16 q1, q6, a1
# CHECK: esp.vssubs.s16	 q1, q6, a1             # encoding: [0x5f,0x82,0x91,0xde]
esp.vssubs.s8 q6, q0, a3
# CHECK: esp.vssubs.s8	 q6, q0, a3             # encoding: [0x5f,0x82,0xe2,0x0e]
esp.vssubs.u16 q6, q1, a2
# CHECK: esp.vssubs.u16	 q6, q1, a2             # encoding: [0x5f,0x02,0xe2,0x36]
esp.vssubs.u8 q2, q4, a1
# CHECK: esp.vssubs.u8	 q2, q4, a1             # encoding: [0x5f,0x82,0xa1,0x86]
esp.vsub.s16 q3, q1, q2
# CHECK: esp.vsub.s16	 q3, q1, q2             # encoding: [0xdf,0x06,0xb4,0x2a]
esp.vsub.s16.ld.incp q5, a2, q6, q3, q5
# CHECK: esp.vsub.s16.ld.incp	 q5, a2, q6, q3, q5 # encoding: [0x3b,0x75,0xea,0x75]
esp.vsub.s16.st.incp q4, a0, q1, q1, q1
# CHECK: esp.vsub.s16.st.incp	 q4, a0, q1, q1, q1 # encoding: [0x3b,0x71,0x99,0x27]
esp.vsub.s32 q4, q1, q0
# CHECK: esp.vsub.s32	 q4, q1, q0             # encoding: [0xdf,0x05,0xc4,0x22]
esp.vsub.s32.ld.incp q2, a2, q5, q0, q5
# CHECK: esp.vsub.s32.ld.incp	 q2, a2, q5, q0, q5 # encoding: [0x3b,0x6b,0x52,0x15]
esp.vsub.s32.st.incp q5, a5, q4, q1, q6
# CHECK: esp.vsub.s32.st.incp	 q5, a5, q4, q1, q6 # encoding: [0x3b,0xf7,0x43,0x3b]
esp.vsub.s8 q2, q1, q3
# CHECK: esp.vsub.s8	 q2, q1, q3             # encoding: [0xdf,0x06,0xa0,0x2e]
esp.vsub.s8.ld.incp q4, a4, q5, q4, q0
# CHECK: esp.vsub.s8.ld.incp	 q4, a4, q5, q4, q0 # encoding: [0x3b,0x71,0xdb,0x80]
esp.vsub.s8.st.incp q1, a5, q0, q1, q1
# CHECK: esp.vsub.s8.st.incp	 q1, a5, q0, q1, q1 # encoding: [0x3b,0xe5,0x8b,0x26]
esp.vsub.u16 q1, q2, q1
# CHECK: esp.vsub.u16	 q1, q2, q1             # encoding: [0xdf,0x04,0x94,0x46]
esp.vsub.u16.ld.incp q6, a4, q1, q6, q1
# CHECK: esp.vsub.u16.ld.incp	 q6, a4, q1, q6, q1 # encoding: [0x3b,0x79,0x93,0xc5]
esp.vsub.u16.st.incp q1, a0, q0, q1, q6
# CHECK: esp.vsub.u16.st.incp	 q1, a0, q0, q1, q6 # encoding: [0x3b,0x65,0x81,0x3b]
esp.vsub.u32 q2, q2, q5
# CHECK: esp.vsub.u32	 q2, q2, q5             # encoding: [0xdf,0x05,0xa0,0x56]
esp.vsub.u32.ld.incp q4, a5, q2, q4, q3
# CHECK: esp.vsub.u32.ld.incp	 q4, a5, q2, q4, q3 # encoding: [0x3b,0xf3,0x23,0x8c]
esp.vsub.u32.st.incp q0, a0, q3, q3, q1
# CHECK: esp.vsub.u32.st.incp	 q0, a0, q3, q3, q1 # encoding: [0x3b,0x63,0x31,0x66]
esp.vsub.u8 q1, q4, q1
# CHECK: esp.vsub.u8	 q1, q4, q1             # encoding: [0xdf,0x04,0x90,0x86]
esp.vsub.u8.ld.incp q4, a1, q4, q5, q2
# CHECK: esp.vsub.u8.ld.incp	 q4, a1, q4, q5, q2 # encoding: [0x3b,0xf1,0xc1,0xa8]
esp.vsub.u8.st.incp q6, a4, q6, q5, q5
# CHECK: esp.vsub.u8.st.incp	 q6, a4, q6, q5, q5 # encoding: [0x3b,0x79,0xe3,0xb6]
esp.addx2 a1, a4, a2
# CHECK: esp.addx2	 a1, a4, a2             # encoding: [0xb3,0x05,0xc7,0x04]
esp.addx4 a1, a3, a3
# CHECK: esp.addx4	 a1, a3, a3             # encoding: [0xb3,0x85,0xd6,0x08]
esp.sat a5, a1, a2
# CHECK: esp.sat	 a5, a1, a2                     # encoding: [0xb3,0x25,0xf6,0x40]
esp.subx2 a0, a3, a5
# CHECK: esp.subx2	 a0, a3, a5             # encoding: [0x33,0x85,0xf6,0x44]
esp.subx4 a5, a4, a0
# CHECK: esp.subx4	 a5, a4, a0             # encoding: [0xb3,0x07,0xa7,0x48]
esp.andq q4, q2, q2
# CHECK: esp.andq	 q4, q2, q2                     # encoding: [0x5f,0x22,0x04,0x48]
esp.notq q0, q1
# CHECK: esp.notq	 q0, q1                         # encoding: [0x5f,0x20,0x06,0x20]
esp.orq q4, q3, q1
# CHECK: esp.orq	 q4, q3, q1                     # encoding: [0x5f,0x22,0x00,0x64]
esp.xorq q2, q1, q1
# CHECK: esp.xorq	 q2, q1, q1                     # encoding: [0x5f,0x21,0x02,0x24]
esp.vcmp.eq.s16 q1, q5, q1
# CHECK: esp.vcmp.eq.s16	 q1, q5, q1             # encoding: [0xdf,0xb4,0x01,0xa4]
esp.vcmp.eq.s32 q3, q3, q2
# CHECK: esp.vcmp.eq.s32	 q3, q3, q2             # encoding: [0xdf,0x2d,0x01,0x68]
esp.vcmp.eq.s8 q3, q6, q6
# CHECK: esp.vcmp.eq.s8	 q3, q6, q6             # encoding: [0xdf,0xb5,0x00,0xd8]
esp.vcmp.eq.u16 q6, q2, q5
# CHECK: esp.vcmp.eq.u16	 q6, q2, q5             # encoding: [0x5f,0x37,0x01,0x54]
esp.vcmp.eq.u32 q0, q6, q6
# CHECK: esp.vcmp.eq.u32	 q0, q6, q6             # encoding: [0x5f,0x2c,0x00,0xd8]
esp.vcmp.eq.u8 q4, q2, q4
# CHECK: esp.vcmp.eq.u8	 q4, q2, q4             # encoding: [0x5f,0x36,0x00,0x50]
esp.vcmp.gt.s16 q4, q0, q2
# CHECK: esp.vcmp.gt.s16	 q4, q0, q2             # encoding: [0x5f,0xb6,0x05,0x08]
esp.vcmp.gt.s32 q1, q1, q0
# CHECK: esp.vcmp.gt.s32	 q1, q1, q0             # encoding: [0xdf,0x2c,0x05,0x20]
esp.vcmp.gt.s8 q4, q0, q2
# CHECK: esp.vcmp.gt.s8	 q4, q0, q2             # encoding: [0x5f,0xb6,0x04,0x08]
esp.vcmp.gt.u16 q5, q6, q2
# CHECK: esp.vcmp.gt.u16	 q5, q6, q2             # encoding: [0xdf,0x36,0x05,0xc8]
esp.vcmp.gt.u32 q1, q5, q2
# CHECK: esp.vcmp.gt.u32	 q1, q5, q2             # encoding: [0xdf,0x2c,0x04,0xa8]
esp.vcmp.gt.u8 q1, q4, q4
# CHECK: esp.vcmp.gt.u8	 q1, q4, q4             # encoding: [0xdf,0x34,0x04,0x90]
esp.vcmp.lt.s16 q6, q2, q5
# CHECK: esp.vcmp.lt.s16	 q6, q2, q5             # encoding: [0x5f,0xb7,0x03,0x54]
esp.vcmp.lt.s32 q2, q3, q2
# CHECK: esp.vcmp.lt.s32	 q2, q3, q2             # encoding: [0x5f,0x2d,0x03,0x68]
esp.vcmp.lt.s8 q0, q6, q2
# CHECK: esp.vcmp.lt.s8	 q0, q6, q2             # encoding: [0x5f,0xb4,0x02,0xc8]
esp.vcmp.lt.u16 q0, q2, q5
# CHECK: esp.vcmp.lt.u16	 q0, q2, q5             # encoding: [0x5f,0x34,0x03,0x54]
esp.vcmp.lt.u32 q1, q0, q3
# CHECK: esp.vcmp.lt.u32	 q1, q0, q3             # encoding: [0xdf,0x2c,0x02,0x0c]
esp.vcmp.lt.u8 q1, q1, q4
# CHECK: esp.vcmp.lt.u8	 q1, q1, q4             # encoding: [0xdf,0x34,0x02,0x30]
esp.mov.s16.qacc q5
# CHECK: esp.mov.s16.qacc	 q5                     # encoding: [0x5b,0x14,0x60,0x10]
esp.mov.s8.qacc q3
# CHECK: esp.mov.s8.qacc	 q3                     # encoding: [0x5b,0x0c,0x20,0x10]
esp.mov.u16.qacc q6
# CHECK: esp.mov.u16.qacc	 q6                     # encoding: [0x5b,0x18,0x40,0x10]
esp.mov.u8.qacc q3
# CHECK: esp.mov.u8.qacc	 q3                     # encoding: [0x5b,0x0c,0x00,0x10]
esp.movi.16.a q5, a4, 2
# CHECK: esp.movi.16.a	 q5, a4, 2              # encoding: [0x5f,0x03,0xc1,0x14]
esp.movi.16.q q1, a3, 7
# CHECK: esp.movi.16.q	 q1, a3, 7              # encoding: [0xdf,0x83,0xe2,0x84]
esp.movi.32.a q6, a0, 2
# CHECK: esp.movi.32.a	 q6, a0, 2              # encoding: [0x5f,0x01,0xc0,0x98]
esp.movi.32.q q6, a4, 2
# CHECK: esp.movi.32.q	 q6, a4, 2              # encoding: [0x5f,0x04,0x93,0x98]
esp.movi.8.a q5, a5, 6
# CHECK: esp.movi.8.a	 q5, a5, 6              # encoding: [0xdf,0x03,0x83,0x14]
esp.movi.8.q q2, a1, 15
# CHECK: esp.movi.8.q	 q2, a1, 15             # encoding: [0xdf,0x87,0xa1,0x88]
esp.movx.r.cfg a4
# CHECK: esp.movx.r.cfg	 a4                     # encoding: [0x5f,0x03,0xd0,0x80]
esp.movx.r.fft.bit.width a2
# CHECK: esp.movx.r.fft.bit.width	 a2             # encoding: [0x5f,0x02,0xd0,0x84]
esp.movx.r.perf a4, a1
# CHECK: esp.movx.r.perf	 a4, a1                 # encoding: [0x5f,0x83,0xd1,0x8c]
esp.movx.r.sar a5
# CHECK: esp.movx.r.sar	 a5                     # encoding: [0xdf,0x03,0xb0,0x80]
esp.movx.r.sar.bytes a4
# CHECK: esp.movx.r.sar.bytes	 a4             # encoding: [0x5f,0x03,0xb0,0x88]
esp.movx.r.xacc.h a5
# CHECK: esp.movx.r.xacc.h	 a5             # encoding: [0xdf,0x03,0xb0,0x8c]
esp.movx.r.xacc.l a3
# CHECK: esp.movx.r.xacc.l	 a3             # encoding: [0xdf,0x02,0xb0,0x84]
esp.movx.w.cfg a1
# CHECK: esp.movx.w.cfg	 a1                     # encoding: [0x5f,0x80,0xd1,0x90]
esp.movx.w.fft.bit.width a1
# CHECK: esp.movx.w.fft.bit.width	 a1             # encoding: [0x5f,0x80,0xd1,0x94]
esp.movx.w.perf a1
# CHECK: esp.movx.w.perf	 a1                     # encoding: [0x5f,0x80,0xd1,0x9c]
esp.movx.w.sar a2
# CHECK: esp.movx.w.sar	 a2                     # encoding: [0x5f,0x00,0xb2,0x90]
esp.movx.w.sar.bytes a1
# CHECK: esp.movx.w.sar.bytes	 a1             # encoding: [0x5f,0x80,0xb1,0x98]
esp.movx.w.xacc.h a5
# CHECK: esp.movx.w.xacc.h	 a5             # encoding: [0x5f,0x80,0xb3,0x9c]
esp.movx.w.xacc.l a2
# CHECK: esp.movx.w.xacc.l	 a2             # encoding: [0x5f,0x00,0xb2,0x94]
esp.vext.s16 q3, q1, q5
# CHECK: esp.vext.s16	 q3, q1, q5             # encoding: [0xdb,0x59,0x18,0x19]
esp.vext.s8 q0, q0, q6
# CHECK: esp.vext.s8	 q0, q0, q6             # encoding: [0x5b,0x58,0x08,0x0a]
esp.vext.u16 q4, q2, q6
# CHECK: esp.vext.u16	 q4, q2, q6             # encoding: [0x5b,0x5a,0x28,0x12]
esp.vext.u8 q1, q2, q5
# CHECK: esp.vext.u8	 q1, q2, q5             # encoding: [0xdb,0x58,0x28,0x01]
esp.vunzip.16 q2, q3
# CHECK: esp.vunzip.16	 q2, q3                 # encoding: [0x5f,0x00,0x86,0x4e]
esp.vunzip.32 q2, q1
# CHECK: esp.vunzip.32	 q2, q1                 # encoding: [0x5f,0x80,0x84,0x46]
esp.vunzip.8 q3, q5
# CHECK: esp.vunzip.8	 q3, q5                 # encoding: [0x5f,0x00,0x84,0x76]
esp.vunzipt.16 q5, q4, q5
# CHECK: esp.vunzipt.16	 q5, q4, q5             # encoding: [0x5b,0x4c,0xc8,0xb1]
esp.vunzipt.8 q3, q6, q4
# CHECK: esp.vunzipt.8	 q3, q6, q4             # encoding: [0x5b,0x4c,0x88,0x78]
esp.vzip.16 q1, q0
# CHECK: esp.vzip.16	 q1, q0                 # encoding: [0x5f,0x00,0x82,0x22]
esp.vzip.32 q1, q1
# CHECK: esp.vzip.32	 q1, q1                 # encoding: [0x5f,0x80,0x80,0x26]
esp.vzip.8 q2, q2
# CHECK: esp.vzip.8	 q2, q2                 # encoding: [0x5f,0x00,0x80,0x4a]
esp.vzipt.16 q1, q2, q1
# CHECK: esp.vzipt.16	 q1, q2, q1             # encoding: [0x5b,0x4c,0x40,0x29]
esp.vzipt.8 q3, q3, q5
# CHECK: esp.vzipt.8	 q3, q3, q5             # encoding: [0x5b,0x4c,0x08,0x6d]
esp.zero.q q0
# CHECK: esp.zero.q	 q0                     # encoding: [0x5b,0x00,0x40,0x00]
esp.zero.qacc
# CHECK: esp.zero.qacc	                        # encoding: [0x5b,0x02,0x00,0x00]
esp.zero.xacc
# CHECK: esp.zero.xacc	                        # encoding: [0x5b,0x00,0x00,0x00]
esp.fft.ams.s16.ld.incp q5, a4, q6, q2, q3, q3, q0, 1
# CHECK: esp.fft.ams.s16.ld.incp	 q5, a4, q6, q2, q3, q3, q0, 1 # encoding: [0x7b,0x17,0xa3,0x63]
esp.fft.ams.s16.ld.incp.uaup q1, a5, q3, q5, q6, q2, q0, 0
# CHECK: esp.fft.ams.s16.ld.incp.uaup	 q1, a5, q3, q5, q6, q2, q0, 0 # encoding: [0xfb,0xa5,0x53,0xc2]
esp.fft.ams.s16.ld.r32.decp q4, a5, q3, q6, q5, q6, q6, 1
# CHECK: esp.fft.ams.s16.ld.r32.decp	 q4, a5, q3, q6, q5, q6, q6, 1 # encoding: [0xfb,0xf1,0xeb,0xba]
esp.fft.ams.s16.st.incp q5, q3, a3, a1, q6, q2, q3, 0
# CHECK: esp.fft.ams.s16.st.incp	 q5, q3, a3, a1, q6, q2, q3, 0 # encoding: [0xbf,0xb5,0x51,0xce]
esp.fft.bitrev q3, a5
# CHECK: esp.fft.bitrev	 q3, a5                 # encoding: [0x5b,0x82,0x33,0x10]
esp.fft.cmul.s16.ld.xp q3, a3, a4, q1, q3, q2, 4
# CHECK: esp.fft.cmul.s16.ld.xp	 q3, a3, a4, q1, q3, q2, 4 # encoding: [0xbf,0x8c,0x62,0x4e]
esp.fft.cmul.s16.st.xp q1, q2, q3, a4, a0, 7, 1, 1
# CHECK: esp.fft.cmul.s16.st.xp	 q1, q2, q3, a4, a0, 7, 1, 1 # encoding: [0xff,0x4f,0x23,0x45]
esp.fft.r2bf.s16 q3, q2, q5, q1, 1
# CHECK: esp.fft.r2bf.s16	 q3, q2, q5, q1, 1      # encoding: [0xdf,0x05,0xa5,0xa6]
esp.fft.r2bf.s16.st.incp q5, q4, q4, a5, 1
# CHECK: esp.fft.r2bf.s16.st.incp	 q5, q4, q4, a5, 1 # encoding: [0xdf,0xe2,0x43,0x92]
esp.fft.vst.r32.decp q1, a4, 1
# CHECK: esp.fft.vst.r32.decp	 q1, a4, 1      # encoding: [0x3b,0x24,0x03,0x80]
esp.ld.128.usar.ip q5, a4, -896
# CHECK: esp.ld.128.usar.ip	 q5, a4, -896   # encoding: [0x3b,0x34,0x43,0xc8]
esp.ld.128.usar.xp q5, a2, a5
# CHECK: esp.ld.128.usar.xp	 q5, a2, a5     # encoding: [0x5f,0x54,0x72,0x80]
esp.ld.xacc.ip a1, 616
# CHECK: esp.ld.xacc.ip	 a1, 616                # encoding: [0x3b,0xd7,0x91,0x20]
esp.ldqa.s16.128.ip a3, -672
# CHECK: esp.ldqa.s16.128.ip	 a3, -672       # encoding: [0xbb,0xcc,0xd2,0xe0]
esp.ldqa.s16.128.xp a4, a1
# CHECK: esp.ldqa.s16.128.xp	 a4, a1         # encoding: [0x5b,0x53,0x33,0x13]
esp.ldqa.s8.128.ip a4, 1808
# CHECK: esp.ldqa.s8.128.ip	 a4, 1808       # encoding: [0xbb,0x42,0x73,0x60]
esp.ldqa.s8.128.xp a0, a4
# CHECK: esp.ldqa.s8.128.xp	 a0, a4         # encoding: [0x5b,0x51,0x61,0x13]
esp.ldqa.u16.128.ip a3, -496
# CHECK: esp.ldqa.u16.128.ip	 a3, -496       # encoding: [0xbb,0xc2,0xe2,0xa0]
esp.ldqa.u16.128.xp a2, a4
# CHECK: esp.ldqa.u16.128.xp	 a2, a4         # encoding: [0x5b,0x52,0x62,0x13]
esp.ldqa.u8.128.ip a2, 1200
# CHECK: esp.ldqa.u8.128.ip	 a2, 1200       # encoding: [0xbb,0x56,0x42,0x20]
esp.ldqa.u8.128.xp a2, a5
# CHECK: esp.ldqa.u8.128.xp	 a2, a5         # encoding: [0x5b,0x50,0x72,0x13]
esp.vldbc.16.ip q4, a3, 408
# CHECK: esp.vldbc.16.ip	 q4, a3, 408            # encoding: [0x3b,0xb0,0x32,0xb6]
esp.vldbc.16.xp q5, a2, a1
# CHECK: esp.vldbc.16.xp	 q5, a2, a1             # encoding: [0x5f,0x54,0x32,0x96]
esp.vldbc.32.ip q6, a2, -176
# CHECK: esp.vldbc.32.ip	 q6, a2, -176           # encoding: [0x3b,0x38,0xa2,0xce]
esp.vldbc.32.xp q0, a1, a2
# CHECK: esp.vldbc.32.xp	 q0, a1, a2             # encoding: [0x5f,0xc0,0x41,0x8e]
esp.vldbc.8.ip q6, a0, 200
# CHECK: esp.vldbc.8.ip	 q6, a0, 200            # encoding: [0x3b,0x38,0x91,0x16]
esp.vldbc.8.xp q5, a4, a2
# CHECK: esp.vldbc.8.xp	 q5, a4, a2             # encoding: [0x5f,0x54,0x43,0x86]
esp.vldext.s16.ip q4, q1, a3, -112
# CHECK: esp.vldext.s16.ip	 q4, q1, a3, -112 # encoding: [0xbb,0xd0,0x92,0xc8]
esp.vldext.s16.xp q2, q4, a1, a0
# CHECK: esp.vldext.s16.xp	 q2, q4, a1, a0 # encoding: [0x5f,0xea,0x21,0xf0]
esp.vldext.s8.ip q3, q2, a4, 0
# CHECK: esp.vldext.s8.ip	 q3, q2, a4, 0          # encoding: [0x3b,0x4d,0x03,0x48]
esp.vldext.s8.xp q3, q6, a4, a1
# CHECK: esp.vldext.s8.xp	 q3, q6, a4, a1         # encoding: [0x5f,0x6f,0x33,0x70]
esp.vldext.u16.ip q2, q1, a1, 48
# CHECK: esp.vldext.u16.ip	 q2, q1, a1, 48 # encoding: [0xbb,0xc8,0x31,0x88]
esp.vldext.u16.xp q1, q2, a3, a1
# CHECK: esp.vldext.u16.xp	 q1, q2, a3, a1 # encoding: [0x5f,0xe5,0x32,0xb0]
esp.vldext.u8.ip q5, q6, a0, -48
# CHECK: esp.vldext.u8.ip	 q5, q6, a0, -48        # encoding: [0x3b,0x57,0xd1,0x08]
esp.vldext.u8.xp q0, q6, a3, a1
# CHECK: esp.vldext.u8.xp	 q0, q6, a3, a1         # encoding: [0x5f,0xe3,0x32,0x30]
esp.vldhbc.16.incp q4, q2, a2
# CHECK: esp.vldhbc.16.incp	 q4, q2, a2     # encoding: [0x3b,0x51,0x02,0x28]
esp.ld.qacc.h.h.128.ip a0, 816
# CHECK: esp.ld.qacc.h.h.128.ip	 a0, 816        # encoding: [0x3b,0x46,0x31,0x40]
esp.ld.qacc.h.l.128.ip a2, -496
# CHECK: esp.ld.qacc.h.l.128.ip	 a2, -496       # encoding: [0x3b,0x42,0xe2,0x60]
esp.ld.qacc.l.h.128.ip a1, -432
# CHECK: esp.ld.qacc.l.h.128.ip	 a1, -432       # encoding: [0x3b,0xca,0xe1,0x00]
esp.ld.qacc.l.l.128.ip a0, 1840
# CHECK: esp.ld.qacc.l.l.128.ip	 a0, 1840       # encoding: [0x3b,0x46,0x71,0x20]
esp.ld.ua.state.ip a4, -1392
# CHECK: esp.ld.ua.state.ip	 a4, -1392      # encoding: [0x3b,0x45,0x53,0x60]
esp.ldxq.32 q5, q4, a4, 3, 6
# CHECK: esp.ldxq.32	 q5, q4, a4, 3, 6       # encoding: [0x5f,0x34,0x8b,0x78]
esp.st.qacc.h.h.128.ip a4, 656
# CHECK: esp.st.qacc.h.h.128.ip	 a4, 656        # encoding: [0x3b,0x52,0x23,0xc0]
esp.st.qacc.h.l.128.ip a3, -1072
# CHECK: esp.st.qacc.h.l.128.ip	 a3, -1072      # encoding: [0x3b,0xda,0xb2,0xe0]
esp.st.qacc.l.h.128.ip a2, 784
# CHECK: esp.st.qacc.l.h.128.ip	 a2, 784        # encoding: [0x3b,0x42,0x32,0x80]
esp.st.qacc.l.l.128.ip a3, -736
# CHECK: esp.st.qacc.l.l.128.ip	 a3, -736       # encoding: [0x3b,0xc4,0xd2,0xa0]
esp.st.ua.state.ip a5, 1376
# CHECK: esp.st.ua.state.ip	 a5, 1376       # encoding: [0x3b,0xd9,0xa3,0xa0]
esp.stxq.32 q6, q5, a5, 3, 4
# CHECK: esp.stxq.32	 q6, q5, a5, 3, 4       # encoding: [0x5f,0xb8,0x8b,0xf1]
esp.vld.128.ip q1, a3, 560
# CHECK: esp.vld.128.ip	 q1, a3, 560            # encoding: [0x3b,0xa6,0x12,0x12]
esp.vld.128.xp q1, a5, a3
# CHECK: esp.vld.128.xp	 q1, a5, a3             # encoding: [0x5f,0xc4,0x53,0x82]
esp.vld.h.64.ip q4, a5, -568
# CHECK: esp.vld.h.64.ip	 q4, a5, -568           # encoding: [0x3b,0xb2,0xc3,0x6c]
esp.vld.h.64.xp q1, a1, a2
# CHECK: esp.vld.h.64.xp	 q1, a1, a2             # encoding: [0x5f,0xc4,0x41,0x8c]
esp.vld.l.64.ip q2, a3, -696
# CHECK: esp.vld.l.64.ip	 q2, a3, -696           # encoding: [0x3b,0xaa,0x42,0x2c]
esp.vld.l.64.xp q5, a4, a4
# CHECK: esp.vld.l.64.xp	 q5, a4, a4             # encoding: [0x5f,0x54,0x63,0x84]
esp.vst.128.ip q3, a4, 1088
# CHECK: esp.vst.128.ip	 q3, a4, 1088           # encoding: [0x3b,0x2c,0x23,0xa2]
esp.vst.128.xp q1, a4, a0
# CHECK: esp.vst.128.xp	 q1, a4, a0             # encoding: [0x5f,0x44,0x23,0x92]
esp.vst.h.64.ip q5, a4, -136
# CHECK: esp.vst.h.64.ip	 q5, a4, -136           # encoding: [0x3b,0x36,0x73,0xfc]
esp.vst.h.64.xp q1, a0, a2
# CHECK: esp.vst.h.64.xp	 q1, a0, a2             # encoding: [0x5f,0x44,0x41,0x9c]
esp.vst.l.64.ip q5, a0, -440
# CHECK: esp.vst.l.64.ip	 q5, a0, -440           # encoding: [0x3b,0x36,0x41,0xb4]
esp.vst.l.64.xp q2, a5, a3
# CHECK: esp.vst.l.64.xp	 q2, a5, a3             # encoding: [0x5f,0xc8,0x53,0x94]
esp.slci.2q q2, q6, 10
# CHECK: esp.slci.2q	 q2, q6, 10             # encoding: [0x5b,0x49,0x48,0x0a]
esp.slcxxp.2q q2, q1, a0, a3
# CHECK: esp.slcxxp.2q	 q2, q1, a0, a3         # encoding: [0x5f,0x40,0x51,0x09]
esp.src.q q5, q6, q2
# CHECK: esp.src.q	 q5, q6, q2             # encoding: [0xdb,0x02,0x2c,0x8a]
esp.src.q.ld.ip q3, a4, 48, q6, q6
# CHECK: esp.src.q.ld.ip	 q3, a4, 48, q6, q6     # encoding: [0x3b,0x2f,0x1b,0x1a]
esp.src.q.ld.xp q6, a5, a0, q6, q6
# CHECK: esp.src.q.ld.xp	 q6, a5, a0, q6, q6     # encoding: [0x3b,0x98,0x2b,0x1a]
esp.src.q.qup q1, q5, q0
# CHECK: esp.src.q.qup	 q1, q5, q0             # encoding: [0xdb,0x10,0x2c,0x81]
esp.srci.2q q5, q5, 11
# CHECK: esp.srci.2q	 q5, q5, 11             # encoding: [0xdb,0x49,0xc8,0x15]
esp.srcmb.s16.q.qacc q6, q2, 0
# CHECK: esp.srcmb.s16.q.qacc	 q6, q2, 0      # encoding: [0x5b,0x18,0x64,0x9a]
esp.srcmb.s16.qacc q2, a2, 0
# CHECK: esp.srcmb.s16.qacc	 q2, a2, 0      # encoding: [0x3b,0x28,0x02,0xd8]
esp.srcmb.s8.q.qacc q5, q3, 0
# CHECK: esp.srcmb.s8.q.qacc	 q5, q3, 0      # encoding: [0x5b,0x14,0x64,0x8b]
esp.srcmb.s8.qacc q1, a3, 1
# CHECK: esp.srcmb.s8.qacc	 q1, a3, 1      # encoding: [0x3b,0xa4,0x02,0x78]
esp.srcmb.u16.q.qacc q3, q4, 0
# CHECK: esp.srcmb.u16.q.qacc	 q3, q4, 0      # encoding: [0x5b,0x0c,0x6c,0x90]
esp.srcmb.u16.qacc q1, a0, 1
# CHECK: esp.srcmb.u16.qacc	 q1, a0, 1      # encoding: [0x3b,0x24,0x01,0xb8]
esp.srcmb.u8.q.qacc q0, q5, 1
# CHECK: esp.srcmb.u8.q.qacc	 q0, q5, 1      # encoding: [0x5b,0x00,0x6c,0x85]
esp.srcmb.u8.qacc q1, a3, 1
# CHECK: esp.srcmb.u8.qacc	 q1, a3, 1      # encoding: [0x3b,0xa4,0x02,0x38]
esp.srcq.128.st.incp q1, q2, a0
# CHECK: esp.srcq.128.st.incp	 q1, q2, a0     # encoding: [0x5b,0x40,0x01,0x09]
esp.srcxxp.2q q3, q3, a4, a4
# CHECK: esp.srcxxp.2q	 q3, q3, a4, a4         # encoding: [0x5f,0x44,0x63,0x0f]
esp.srs.s.xacc a0, a0
# CHECK: esp.srs.s.xacc	 a0, a0                 # encoding: [0x5f,0x01,0xf1,0x94]
esp.srs.u.xacc a0, a3
# CHECK: esp.srs.u.xacc	 a0, a3                 # encoding: [0x5f,0x81,0xf2,0x84]
esp.vsl.32 q6, q4
# CHECK: esp.vsl.32	 q6, q4                 # encoding: [0x5b,0x18,0x04,0x90]
esp.vsld.16 q6, q5, q1
# CHECK: esp.vsld.16	 q6, q5, q1             # encoding: [0x5f,0x18,0x20,0x15]
esp.vsld.32 q2, q4, q3
# CHECK: esp.vsld.32	 q2, q4, q3             # encoding: [0x5f,0x08,0x10,0x13]
esp.vsld.8 q1, q1, q4
# CHECK: esp.vsld.8	 q1, q1, q4             # encoding: [0x5f,0x04,0x08,0x04]
esp.vsr.s32 q1, q3
# CHECK: esp.vsr.s32	 q1, q3                 # encoding: [0x5b,0x07,0x04,0x8c]
esp.vsr.u32 q5, q2
# CHECK: esp.vsr.u32	 q5, q2                 # encoding: [0x5b,0x15,0x04,0x88]
esp.vsrd.16 q1, q5, q5
# CHECK: esp.vsrd.16	 q1, q5, q5             # encoding: [0x5f,0x04,0x68,0x15]
esp.vsrd.32 q1, q6, q3
# CHECK: esp.vsrd.32	 q1, q6, q3             # encoding: [0x5f,0x04,0x50,0x1b]
esp.vsrd.8 q0, q3, q1
# CHECK: esp.vsrd.8	 q0, q3, q1             # encoding: [0x5f,0x00,0x40,0x0d]
esp.st.s.xacc.ip a1, 304
# CHECK: esp.st.s.xacc.ip	 a1, 304                # encoding: [0xbb,0xd9,0x41,0xa0]
esp.st.u.xacc.ip a2, 976
# CHECK: esp.st.u.xacc.ip	 a2, 976                # encoding: [0xbb,0x49,0xf2,0x20]
