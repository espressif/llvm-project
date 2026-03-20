# RUN: llvm-mc -triple=riscv64 --mattr=+experimental-xtheadzmpanel -show-encoding %s \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

# Check CSR names for XTHeadZmpanel

# All panel CSRs are URO (user read-only), written via panel config instructions.

csrr a0, th.custom_ctrl
# CHECK-INST: csrr a0, th.custom_ctrl

csrr a1, th.base_addr_a
# CHECK-INST: csrr a1, th.base_addr_a

csrr a2, th.base_addr_b
# CHECK-INST: csrr a2, th.base_addr_b

csrr a3, th.base_addr_d
# CHECK-INST: csrr a3, th.base_addr_d

csrr a4, th.rstrideb_a
# CHECK-INST: csrr a4, th.rstrideb_a

csrr a5, th.rstrideb_b
# CHECK-INST: csrr a5, th.rstrideb_b

csrr s0, th.rstrideb_d
# CHECK-INST: csrr s0, th.rstrideb_d

csrr s1, th.panel_m
# CHECK-INST: csrr s1, th.panel_m

csrr t0, th.panel_n
# CHECK-INST: csrr t0, th.panel_n

csrr t1, th.panel_k
# CHECK-INST: csrr t1, th.panel_k

csrr t2, th.mptr_ld
# CHECK-INST: csrr t2, th.mptr_ld

csrr t3, th.nptr_ld
# CHECK-INST: csrr t3, th.nptr_ld

csrr t4, th.kptr_ld
# CHECK-INST: csrr t4, th.kptr_ld

csrr t5, th.mptr_st
# CHECK-INST: csrr t5, th.mptr_st

csrr t6, th.nptr_st
# CHECK-INST: csrr t6, th.nptr_st

csrr a0, th.addr_a
# CHECK-INST: csrr a0, th.addr_a

csrr a1, th.addr_b
# CHECK-INST: csrr a1, th.addr_b

csrr a2, th.addr_d
# CHECK-INST: csrr a2, th.addr_d
