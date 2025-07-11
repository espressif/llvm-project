# RUN: not llvm-mc -triple xtensa %s 2>&1 | FileCheck %s

.align	4
LBL0:

rsr a0, intclear
# CHECK: :[[#@LINE-1]]:9: error: invalid operand for instruction

rsr a0, intset
# CHECK: :[[#@LINE-1]]:9: error: invalid operand for instruction

wsr a1, interrupt
# CHECK: :[[#@LINE-1]]:9: error: invalid operand for instruction

xsr a1, intset
# CHECK: :[[#@LINE-1]]:9: error: invalid operand for instruction

xsr a1, interrupt
# CHECK: :[[#@LINE-1]]:9: error: invalid operand for instruction
