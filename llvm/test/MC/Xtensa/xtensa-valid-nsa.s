# RUN: llvm-mc %s -triple=xtensa -mattr=+nsa -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

# Instruction format RRR
# CHECK-INST: nsa a3, a2
# CHECK: encoding: [0x30,0xe2,0x40]
nsa a3, a2

# Instruction format RRR
# CHECK-INST: nsau a3, a2
# CHECK: encoding: [0x30,0xf2,0x40]
nsau a3, a2
