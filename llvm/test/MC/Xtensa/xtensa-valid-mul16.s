# RUN: llvm-mc %s -triple=xtensa -mattr=+mul16 -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s


.align	4
LBL0:

# CHECK-INST:  mul16s a2, a3, a4
# CHECK: encoding: [0x40,0x23,0xd1]
 mul16s a2, a3, a4

# CHECK-INST:  mul16u a2, a3, a4
# CHECK: encoding: [0x40,0x23,0xc1]
 mul16u a2, a3, a4
