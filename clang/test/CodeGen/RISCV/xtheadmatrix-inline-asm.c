// REQUIRES: riscv-registered-target
//
// RUN: %clang_cc1 -triple riscv64 -target-feature +experimental-xtheadmatrix \
// RUN:     -O2 -emit-llvm %s -o - \
// RUN:     | FileCheck %s

// Test XTHeadMatrix inline assembly constraints.

#include <thead_matrix.h>

// Test 1: "tr" constraint output (any matrix register, THRVMMR)
// CHECK-LABEL: @test_tr_output
// CHECK: %0 = tail call target("riscv.matrix") asm sideeffect "th.mlae8 $0, ($1), $2", "=^tr,r,r"(ptr %base, i64 %stride)
mint32_t test_tr_output(void *base, long stride) {
  mint32_t result;
  asm volatile("th.mlae8 %0, (%1), %2"
               : "=tr"(result)
               : "r"(base), "r"(stride));
  return result;
}

// Test 2: "tr" constraint input
// CHECK-LABEL: @test_tr_input
// CHECK: tail call void asm sideeffect "th.msce8 $0, ($1), $2", "^tr,r,r"(target("riscv.matrix") %val, ptr %base, i64 %stride)
void test_tr_input(mint32_t val, void *base, long stride) {
  asm volatile("th.msce8 %0, (%1), %2"
               : /* no outputs */
               : "tr"(val), "r"(base), "r"(stride));
}

// Test 3: "tt" constraint (tile registers only, THRVMTR)
// CHECK-LABEL: @test_tt_output
// CHECK: %0 = tail call target("riscv.matrix") asm sideeffect "th.mlae8 $0, ($1), $2", "=^tt,r,r"(ptr %base, i64 %stride)
mint32_t test_tt_output(void *base, long stride) {
  mint32_t result;
  asm volatile("th.mlae8 %0, (%1), %2"
               : "=tt"(result)
               : "r"(base), "r"(stride));
  return result;
}

// Test 4: "ta" constraint (accumulator registers only, THRVMACC)
// CHECK-LABEL: @test_ta_output
// CHECK: %0 = tail call target("riscv.matrix") asm sideeffect "th.mlae8 $0, ($1), $2", "=^ta,r,r"(ptr %base, i64 %stride)
mint32_t test_ta_output(void *base, long stride) {
  mint32_t result;
  asm volatile("th.mlae8 %0, (%1), %2"
               : "=ta"(result)
               : "r"(base), "r"(stride));
  return result;
}

// Test 5: Clobber constraints
// CHECK-LABEL: @test_clobber
// CHECK: tail call void asm sideeffect "th.mcfg {{.*}}", "~{tr0},~{acc0}"()
void test_clobber(void) {
  asm volatile("th.mcfg zero, zero, zero"
               :
               :
               : "tr0", "acc0");
}

// Test 6: Read-write constraint
// CHECK-LABEL: @test_readwrite
// CHECK: %0 = tail call target("riscv.matrix") asm sideeffect "th.mmov.mm $0, $0", "=^tr,0"(target("riscv.matrix") %val)
mint32_t test_readwrite(mint32_t val) {
  asm volatile("th.mmov.mm %0, %0"
               : "+tr"(val));
  return val;
}
