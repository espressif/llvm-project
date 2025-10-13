// RUN: %clang_cc1 -triple xtensa -emit-llvm  -o  - %s | FileCheck %s
// RUN: %clang_cc1 -triple xtensa -S  -o  - %s | FileCheck %s --check-prefix=ASM

void foo1 (void);
void __attribute__((short_call)) foo (void);
void __attribute__((near)) bar (void) { foo1(); foo(); }

// CHECK: define{{.*}} void @bar() [[NEAR:#[0-9]+]]

// CHECK: declare void @foo() [[SHORTDECL:#[0-9]+]]

// CHECK: attributes [[NEAR]] = { {{.*}} "short-call" {{.*}} }
// CHECK: attributes [[SHORTDECL]] = { {{.*}} "short-call" {{.*}} }

// ASM: callx8 a8
// ASM: call8 foo
