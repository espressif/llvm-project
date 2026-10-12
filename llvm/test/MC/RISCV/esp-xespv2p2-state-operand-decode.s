# State operands (SAR_BYTES, QACC, XACC) of pure-2.2 codegen defs are explicit but
# have no encoding bits. The assembler and disassembler must use the state-free
# *_MC twins, otherwise llvm-objdump reads past the decoded MCInst.
# RUN: llvm-mc -triple riscv32 -mattr=+xespv2p2 -show-encoding %s \
# RUN:   | FileCheck %s --check-prefix=ENC
# RUN: llvm-mc -triple riscv32 -mattr=+xespv2p2 -filetype=obj %s -o %t.o
# RUN: llvm-objdump -d --mattr=+xespv2p2 %t.o | FileCheck %s --check-prefix=DIS

esp.ld.128.usar.ip q0, a1, 16
esp.ld.128.usar.xp q0, a1, a2
esp.ld.qacc.h.h.128.ip a1, 16
esp.ld.qacc.h.l.128.ip a1, 16
esp.ld.qacc.l.h.128.ip a1, 16
esp.ld.qacc.l.l.128.ip a1, 16
esp.ld.xacc.ip a1, 8
esp.movx.r.sar.bytes a2
esp.movx.w.sar.bytes a2
esp.srcq.128.st.incp q0, q1, a1
esp.src.q.ld.ip q2, a1, 16, q0, q1
esp.src.q.ld.xp q2, a1, a2, q0, q1
esp.src.q q0, q1, q2
esp.src.q.qup q0, q1, q2
esp.st.qacc.h.h.128.ip a1, 16
esp.st.qacc.h.l.128.ip a1, 16
esp.st.qacc.l.h.128.ip a1, 16
esp.st.qacc.l.l.128.ip a1, 16
esp.st.s.xacc.ip a1, 8
esp.st.u.xacc.ip a1, 8

# ENC: esp.ld.128.usar.ip {{.*}}# encoding: [0x1f,0x81,0xa1,0x03]
# ENC: esp.ld.128.usar.xp {{.*}}# encoding: [0x1b,0xe0,0x41,0x40]
# ENC: esp.ld.qacc.h.h.128.ip {{.*}}# encoding: [0x1f,0x82,0x01,0x08]
# ENC: esp.ld.qacc.h.l.128.ip {{.*}}# encoding: [0x1f,0x82,0x11,0x08]
# ENC: esp.ld.qacc.l.h.128.ip {{.*}}# encoding: [0x1f,0x82,0x01,0x00]
# ENC: esp.ld.qacc.l.l.128.ip {{.*}}# encoding: [0x1f,0x82,0x11,0x00]
# ENC: esp.ld.xacc.ip {{.*}}# encoding: [0x1f,0x8b,0x11,0x00]
# ENC: esp.movx.r.sar.bytes {{.*}}# encoding: [0x1b,0x22,0xc0,0x88]
# ENC: esp.movx.w.sar.bytes {{.*}}# encoding: [0x1b,0x20,0xc2,0x98]
# ENC: esp.srcq.128.st.incp {{.*}}# encoding: [0x1b,0xc0,0x41,0x84]
# ENC: esp.src.q.ld.ip {{.*}}# encoding: [0x1f,0xca,0x01,0x04]
# ENC: esp.src.q.ld.xp {{.*}}# encoding: [0x1b,0xeb,0x41,0xc4]
# ENC: esp.src.q {{.*}}# encoding: [0x1b,0x40,0x00,0x89]
# ENC: esp.src.q.qup {{.*}}# encoding: [0x1b,0x40,0x04,0x89]
# ENC: esp.st.qacc.h.h.128.ip {{.*}}# encoding: [0x1f,0x82,0x01,0x18]
# ENC: esp.st.qacc.h.l.128.ip {{.*}}# encoding: [0x1f,0x82,0x11,0x18]
# ENC: esp.st.qacc.l.h.128.ip {{.*}}# encoding: [0x1f,0x82,0x01,0x10]
# ENC: esp.st.qacc.l.l.128.ip {{.*}}# encoding: [0x1f,0x82,0x11,0x10]
# ENC: esp.st.s.xacc.ip {{.*}}# encoding: [0x9f,0x85,0x11,0x10]
# ENC: esp.st.u.xacc.ip {{.*}}# encoding: [0x9f,0x85,0x11,0x00]

# DIS: 03a1811f esp.ld.128.usar.ip q0, a1, 16
# DIS: 4041e01b esp.ld.128.usar.xp q0, a1, a2
# DIS: 0801821f esp.ld.qacc.h.h.128.ip a1, 16
# DIS: 0811821f esp.ld.qacc.h.l.128.ip a1, 16
# DIS: 0001821f esp.ld.qacc.l.h.128.ip a1, 16
# DIS: 0011821f esp.ld.qacc.l.l.128.ip a1, 16
# DIS: 00118b1f esp.ld.xacc.ip a1, 8
# DIS: 88c0221b esp.movx.r.sar.bytes a2
# DIS: 98c2201b esp.movx.w.sar.bytes a2
# DIS: 8441c01b esp.srcq.128.st.incp q0, q1, a1
# DIS: 0401ca1f esp.src.q.ld.ip q2, a1, 16, q0, q1
# DIS: c441eb1b esp.src.q.ld.xp q2, a1, a2, q0, q1
# DIS: 8900401b esp.src.q q0, q1, q2
# DIS: 8904401b esp.src.q.qup q0, q1, q2
# DIS: 1801821f esp.st.qacc.h.h.128.ip a1, 16
# DIS: 1811821f esp.st.qacc.h.l.128.ip a1, 16
# DIS: 1001821f esp.st.qacc.l.h.128.ip a1, 16
# DIS: 1011821f esp.st.qacc.l.l.128.ip a1, 16
# DIS: 1011859f esp.st.s.xacc.ip a1, 8
# DIS: 0011859f esp.st.u.xacc.ip a1, 8
