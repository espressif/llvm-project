# Encode/decode round-trip for esp.{vld,vst}.{h,l}.64.{ip,xp} under both
# PIE 2.1 and pure PIE 2.2. Exercises QR_64_LO / QR_64_HI operand parsing,
# encoding, and DecodeQR_64_LORegisterClass / DecodeQR_64_HIRegisterClass.
# RUN: llvm-mc -triple riscv32 -mattr=+xespv2p1 -show-encoding %s \
# RUN:   | FileCheck %s --check-prefix=ENC2P1
# RUN: llvm-mc -triple riscv32 -mattr=+xespv2p1 -filetype=obj %s -o %t2p1.o
# RUN: llvm-objdump -d --mattr=+xespv2p1 %t2p1.o \
# RUN:   | FileCheck %s --check-prefix=DIS2P1
# RUN: llvm-mc -triple riscv32 -mattr=+xespv2p2 -show-encoding %s \
# RUN:   | FileCheck %s --check-prefix=ENC2P2
# RUN: llvm-mc -triple riscv32 -mattr=+xespv2p2 -filetype=obj %s -o %t2p2.o
# RUN: llvm-objdump -d --mattr=+xespv2p2 %t2p2.o \
# RUN:   | FileCheck %s --check-prefix=DIS2P2

esp.vld.h.64.ip q0, a0, 8
esp.vld.h.64.xp q1, a1, a2
esp.vld.l.64.ip q2, a3, -16
esp.vld.l.64.xp q5, a4, a5
esp.vst.h.64.ip q4, a1, 16
esp.vst.h.64.xp q3, a3, a4
esp.vst.l.64.ip q6, a0, -8
esp.vst.l.64.xp q7, a1, a2

# ENC2P1: esp.vld.h.64.ip q0, a0, 8 # encoding: [0x3b,0x22,0x01,0x44]
# ENC2P1: esp.vld.h.64.xp q1, a1, a2 # encoding: [0x5f,0xc4,0x41,0x8c]
# ENC2P1: esp.vld.l.64.ip q2, a3, -16 # encoding: [0x3b,0xa8,0xf2,0x3c]
# ENC2P1: esp.vld.l.64.xp q5, a4, a5 # encoding: [0x5f,0x54,0x73,0x84]
# ENC2P1: esp.vst.h.64.ip q4, a1, 16 # encoding: [0x3b,0xb0,0x11,0xc4]
# ENC2P1: esp.vst.h.64.xp q3, a3, a4 # encoding: [0x5f,0xcc,0x62,0x9c]
# ENC2P1: esp.vst.l.64.ip q6, a0, -8 # encoding: [0x3b,0x3a,0xf1,0xbc]
# ENC2P1: esp.vst.l.64.xp q7, a1, a2 # encoding: [0x5f,0xdc,0x41,0x94]

# DIS2P1: 4401223b esp.vld.h.64.ip q0, a0, 8
# DIS2P1: 8c41c45f esp.vld.h.64.xp q1, a1, a2
# DIS2P1: 3cf2a83b esp.vld.l.64.ip q2, a3, -16
# DIS2P1: 8473545f esp.vld.l.64.xp q5, a4, a5
# DIS2P1: c411b03b esp.vst.h.64.ip q4, a1, 16
# DIS2P1: 9c62cc5f esp.vst.h.64.xp q3, a3, a4
# DIS2P1: bcf13a3b esp.vst.l.64.ip q6, a0, -8
# DIS2P1: 9441dc5f esp.vst.l.64.xp q7, a1, a2

# ENC2P2: esp.vld.h.64.ip q0, a0, 8 # encoding: [0x9f,0x00,0x61,0x43]
# ENC2P2: esp.vld.h.64.xp q1, a1, a2 # encoding: [0x1b,0xe4,0x49,0x41]
# ENC2P2: esp.vld.l.64.ip q2, a3, -16 # encoding: [0x1f,0x8b,0xe2,0x3f]
# ENC2P2: esp.vld.l.64.xp q5, a4, a5 # encoding: [0x1b,0x74,0x7b,0x40]
# ENC2P2: esp.vst.h.64.ip q4, a1, 16 # encoding: [0x1f,0x91,0x61,0xc3]
# ENC2P2: esp.vst.h.64.xp q3, a3, a4 # encoding: [0x1b,0xec,0x6a,0x43]
# ENC2P2: esp.vst.l.64.ip q6, a0, -8 # encoding: [0x9f,0x1b,0xe1,0xbf]
# ENC2P2: esp.vst.l.64.xp q7, a1, a2 # encoding: [0x1b,0xfc,0x49,0x42]

# DIS2P2: 4361009f esp.vld.h.64.ip q0, a0, 8
# DIS2P2: 4149e41b esp.vld.h.64.xp q1, a1, a2
# DIS2P2: 3fe28b1f esp.vld.l.64.ip q2, a3, -16
# DIS2P2: 407b741b esp.vld.l.64.xp q5, a4, a5
# DIS2P2: c361911f esp.vst.h.64.ip q4, a1, 16
# DIS2P2: 436aec1b esp.vst.h.64.xp q3, a3, a4
# DIS2P2: bfe11b9f esp.vst.l.64.ip q6, a0, -8
# DIS2P2: 4249fc1b esp.vst.l.64.xp q7, a1, a2
