# RUN: llvm-mc %s -triple=xtensa  -mattr=+esp32s3 -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

.align	4
LBL0:

# CHECK-INST:  ee.clr_bit_gpio_out  52
# CHECK: encoding: [0x44,0x43,0x76]
ee.clr_bit_gpio_out 52

# CHECK-INST:  ee.get_gpio_in  a2
# CHECK: encoding: [0x24,0x08,0x65]
ee.get_gpio_in a2

# CHECK-INST:  ee.set_bit_gpio_out  18
# CHECK: encoding: [0x24,0x41,0x75]
ee.set_bit_gpio_out 18

# CHECK-INST:  ee.wr_mask_gpio_out	a3, a2
# CHECK: encoding: [0x34,0x42,0x72]
ee.wr_mask_gpio_out	a3, a2
