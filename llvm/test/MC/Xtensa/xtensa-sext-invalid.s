# RUN: not llvm-mc %s -triple=xtensa -mattr=+sext 2>&1 | FileCheck %s

# imm7_22
sext a3, a2, 6
# CHECK: :[[#@LINE-1]]:14: error: expected immediate in range [7, 22]

# imm7_22
sext a3, a2, 23
# CHECK: :[[#@LINE-1]]:14: error: expected immediate in range [7, 22]
