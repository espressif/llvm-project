# esp.sat is shared between ESP-DSP (+xespdsp) and ESP-V 2.1 (+xespv2p1); it is
# gated by HasVendorEspSAT = any_of(FeatureVendorXespdsp, FeatureXespvVersion2p1).
# ESP-V 2.2 has a separate esp.sat with the same assembly form. Positive: it
# assembles under either shared extension. Negative: with none of the extensions
# enabled, both candidates are rejected.

# RUN: llvm-mc %s -triple=riscv32 -mattr=+xespdsp -show-encoding | FileCheck %s --check-prefix=CHECK
# RUN: llvm-mc %s -triple=riscv32 -mattr=+xespv2p1 -show-encoding | FileCheck %s --check-prefix=CHECK
# RUN: not llvm-mc %s -triple=riscv32 2>&1 | FileCheck %s --check-prefix=ERR

esp.sat a5, a1, a2
# CHECK: esp.sat	 a5, a1, a2                     # encoding: [0xb3,0x25,0xf6,0x40]
# ERR: error: invalid instruction, any one of the following would fix this:
# ERR: note: instruction requires the following: 'Espressif ESPDSP' or 'Espressif ESPV 2.1'
# ERR: note: instruction requires the following: 'Espressif ESPV 2.2'
