# RUN: llvm-mc %s -triple=xtensa -mattr=+interrupt -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

.align	4
LBL0:

# CHECK-INST:  rfi	1
# CHECK: encoding: [0x10,0x31,0x00]
 rfi 1

# CHECK-INST:  rsil	a3, 1
# CHECK: encoding: [0x30,0x61,0x00]
 rsil a3, 1

# CHECK-INST: rsr a8, interrupt
# CHECK: encoding: [0x80,0xe2,0x03]
rsr a8, interrupt

# CHECK-INST:  waiti	1
# CHECK: encoding: [0x00,0x71,0x00]
 waiti 1

# CHECK-INST:  wsr	a0, intclear
# CHECK: encoding: [0x00,0xe3,0x13]
 wsr a0, intclear

# CHECK-INST:  wsr	a0, intclear
# CHECK: encoding: [0x00,0xe3,0x13]
 wsr.intclear a0

# CHECK-INST:  wsr	a0, intset
# CHECK: encoding: [0x00,0xe2,0x13]
 wsr a0, intset

# CHECK-INST:  wsr	a0, intset
# CHECK: encoding: [0x00,0xe2,0x13]
 wsr.intset a0

# CHECK-INST:  wsr	a0, intset
# CHECK: encoding: [0x00,0xe2,0x13]
 wsr.INTSET a0
