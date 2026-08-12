# esp.sat is shared between ESP-DSP (+xespdsp) and ESP-V 2.1 (+xespv2p1); it is
# gated by HasVendorEspSAT = any_of(FeatureVendorXespdsp, FeatureXespvVersion2p1).
# Positive: it assembles under either extension. Negative: with neither enabled
# it is rejected.

# RUN: llvm-mc %s -triple=riscv32 -mattr=+xespdsp -show-encoding | FileCheck %s --check-prefix=CHECK
# RUN: llvm-mc %s -triple=riscv32 -mattr=+xespv2p1 -show-encoding | FileCheck %s --check-prefix=CHECK
# RUN: not llvm-mc %s -triple=riscv32 2>&1 | FileCheck %s --check-prefix=ERR

esp.sat a5, a1, a2
# CHECK: esp.sat	 a5, a1, a2                     # encoding: [0xb3,0x25,0xf6,0x40]
# ERR: error: instruction requires the following:
