// RUN: llvm-mc %s -triple=riscv32 -mattr=+xespv -show-encoding \
// RUN:     | FileCheck -check-prefixes=CHECK-ALIAS %s
// RUN: llvm-mc %s -triple=riscv32 -mattr=+xespv -show-encoding -M no-aliases \
// RUN:     | FileCheck -check-prefixes=CHECK-INST %s

// Omitted sat defaults to 1 (sat), omitted rm defaults to 7 (dyn).

// CHECK-INST: esp.vadd.s16 q2, q4, q5, sat
// CHECK-ALIAS: esp.vadd.s16 q2, q4, q5, sat
esp.vadd.s16 q2, q4, q5

// CHECK-INST: esp.vmul.s16 q0, q2, q5, sat, dyn
// CHECK-ALIAS: esp.vmul.s16 q0, q2, q5, sat
esp.vmul.s16 q0, q2, q5

// CHECK-INST: esp.cmul.u8 q2, q1, q2, 1, sat, dyn
// CHECK-ALIAS: esp.cmul.u8 q2, q1, q2, 1, sat
esp.cmul.u8 q2, q1, q2, 1

// Explicit sat matches the default and remains visible in both forms.
// CHECK-INST: esp.vadd.s16 q2, q4, q5, sat
// CHECK-ALIAS: esp.vadd.s16 q2, q4, q5, sat
esp.vadd.s16 q2, q4, q5, sat

// Explicit trunc; aliases omit it.
// CHECK-INST: esp.vadd.s16 q2, q4, q5, trunc
// CHECK-ALIAS: esp.vadd.s16 q2, q4, q5
esp.vadd.s16 q2, q4, q5, trunc

// CHECK-INST: esp.vmul.s16 q0, q2, q5, trunc, rtz
// CHECK-ALIAS: esp.vmul.s16 q0, q2, q5, rtz
esp.vmul.s16 q0, q2, q5, 0, 3
