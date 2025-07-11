# RUN: llvm-mc %s -triple=xtensa  -mattr=+density -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

.align	4
LBL0:

# CHECK-INST:  ill.n
# CHECK: encoding: [0x6d,0xf0]
ill.n
