# RUN: llvm-mc -triple=riscv64 --mattr=+experimental-xtheadmatrix -show-encoding %s \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

# Check CSR names for XTHeadMatrix

# CSR read tests
csrr a0, th.xmcsr
# CHECK-INST: csrr a0, th.xmcsr

csrr a1, th.mtilem
# CHECK-INST: csrr a1, th.mtilem

csrr a2, th.mtilen
# CHECK-INST: csrr a2, th.mtilen

csrr a3, th.mtilek
# CHECK-INST: csrr a3, th.mtilek

csrr a4, th.xmxrm
# CHECK-INST: csrr a4, th.xmxrm

csrr a5, th.xmsat
# CHECK-INST: csrr a5, th.xmsat

csrr s0, th.xmfflags
# CHECK-INST: csrr s0, th.xmfflags

csrr s1, th.xmfrm
# CHECK-INST: csrr s1, th.xmfrm

csrr t0, th.xmsaten
# CHECK-INST: csrr t0, th.xmsaten

csrr t1, th.xmisa
# CHECK-INST: csrr t1, th.xmisa

csrr t2, th.xtlenb
# CHECK-INST: csrr t2, th.xtlenb

csrr t3, th.xtrlenb
# CHECK-INST: csrr t3, th.xtrlenb

csrr t4, th.xalenb
# CHECK-INST: csrr t4, th.xalenb

# CSR write tests (for writable CSRs)
csrw th.xmcsr, a0
# CHECK-INST: csrw th.xmcsr, a0

csrw th.xmxrm, a1
# CHECK-INST: csrw th.xmxrm, a1

csrw th.xmsat, a2
# CHECK-INST: csrw th.xmsat, a2

csrw th.xmfflags, a3
# CHECK-INST: csrw th.xmfflags, a3

csrw th.xmfrm, a4
# CHECK-INST: csrw th.xmfrm, a4

csrw th.xmsaten, a5
# CHECK-INST: csrw th.xmsaten, a5
