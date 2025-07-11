# RUN: not llvm-mc %s -triple=xtensa -mattr=+clamps 2>&1 | FileCheck %s

# imm7_22
clamps a3, a2, 6
# CHECK: :[[#@LINE-1]]:16: error: expected immediate in range [7, 22]

# imm7_22
clamps a3, a2, 23
# CHECK: :[[#@LINE-1]]:16: error: expected immediate in range [7, 22]
