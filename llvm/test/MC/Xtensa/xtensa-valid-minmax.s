# RUN: llvm-mc %s -triple=xtensa -mattr=+minmax -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

# Instruction format RRR
# CHECK-INST: min a2, a3, a4
# CHECK: encoding: [0x40,0x23,0x43]
min a2, a3, a4

# Instruction format RRR
# CHECK-INST: max a2, a3, a4
# CHECK: encoding: [0x40,0x23,0x53]
max a2, a3, a4

# Instruction format RRR
# CHECK-INST: minu a2, a3, a4
# CHECK: encoding: [0x40,0x23,0x63]
minu a2, a3, a4

# Instruction format RRR
# CHECK-INST: maxu a2, a3, a4
# CHECK: encoding: [0x40,0x23,0x73]
maxu a2, a3, a4
