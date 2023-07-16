# RUN: llvm-mc %s -triple=xtensa -mattr=+sext -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

# Instruction format RRR
# CHECK-INST: sext a3, a2, 7
# CHECK: encoding: [0x00,0x32,0x23]
sext a3, a2, 7

# Instruction format RRR
# CHECK-INST: sext a3, a2, 22
# CHECK: encoding: [0xf0,0x32,0x23]
sext a3, a2, 22
