// RUN: %clang_cc1 -triple xtensa -O1 -emit-llvm %s -o - \
// RUN:     | FileCheck %s

// Test Xtensa specific inline assembly constraints.

float f;
void test_f() {
// CHECK-LABEL: define dso_local void @test_f() local_unnamed_addr #0 {
// CHECK: [[FLT_ARG:%[a-zA-Z_0-9]+]] = load float, ptr @f
// CHECK: call void asm sideeffect "", "f"(float [[FLT_ARG]])
  asm volatile ("" :: "f"(f));
}

