// RUN: %clang_cc1 -triple riscv64 -target-feature +experimental-xtheadzmpanel \
// RUN:   -emit-llvm %s -o - | FileCheck %s

// Test inline assembly support for Zmpanel panel-aware instructions.

#include <stddef.h>

// CHECK-LABEL: @test_panel_inline_config(
// CHECK: call void asm sideeffect
void test_panel_inline_config(void *a, void *b, void *d,
                              size_t rsba, size_t m, size_t k) {
  asm volatile("th.mset22adra %0" :: "r"(a));
  asm volatile("th.mset22adrb %0" :: "r"(b));
  asm volatile("th.mset22adrd %0" :: "r"(d));
  asm volatile("th.mset22rsba %0" :: "r"(rsba));
  asm volatile("th.mset22m %0" :: "r"(m));
  asm volatile("th.mset22k %0" :: "r"(k));
  asm volatile("th.msetaccum zero");
  asm volatile("th.msetoob %0" :: "r"((size_t)2));
  asm volatile("th.msetrstptr %0" :: "r"((size_t)1));
}

// CHECK-LABEL: @test_panel_inline_gemm(
// CHECK: call void asm sideeffect
void test_panel_inline_gemm(void) {
  asm volatile("th.ml22e8");
  asm volatile("th.mmacc22.w.b");
  asm volatile("th.msc22e32");
}

// CHECK-LABEL: @test_panel_inline_fp(
// CHECK: call void asm sideeffect
void test_panel_inline_fp(void) {
  asm volatile("th.ml22e16");
  asm volatile("th.mfmacc22.h.e5");
  asm volatile("th.mfmacc22.h.e4");
  asm volatile("th.mfmacc22.bf16.e5");
  asm volatile("th.mfmacc22.bf16.e4");
  asm volatile("th.mfmacc22.s.e5");
  asm volatile("th.mfmacc22.s.e4");
  asm volatile("th.mfmacc22.h");
  asm volatile("th.mfmacc22.s.h");
  asm volatile("th.mfmacc22.s.bf16");
  asm volatile("th.mfmacc22.s");
  asm volatile("th.msc22e16");
}

// CHECK-LABEL: @test_panel_inline_int_variants(
void test_panel_inline_int_variants(void) {
  asm volatile("th.mmacc22.w.b");
  asm volatile("th.mmaccu22.w.b");
  asm volatile("th.mmaccus22.w.b");
  asm volatile("th.mmaccsu22.w.b");
}

// Test a complete panel GEMM loop using inline asm
// CHECK-LABEL: @test_panel_inline_complete(
void test_panel_inline_complete(void *a, void *b, void *d,
                                size_t rsba, size_t rsbb, size_t rsbd,
                                size_t m, size_t n, size_t k) {
  asm volatile(
    "th.mset22adra %0\n\t"
    "th.mset22adrb %1\n\t"
    "th.mset22adrd %2\n\t"
    "th.mset22rsba %3\n\t"
    "th.mset22rsbb %4\n\t"
    "th.mset22rsbd %5\n\t"
    "th.mset22m %6\n\t"
    "th.mset22n %7\n\t"
    :: "r"(a), "r"(b), "r"(d),
       "r"(rsba), "r"(rsbb), "r"(rsbd),
       "r"(m), "r"(n)
  );
  asm volatile(
    "th.mset22k %0\n\t"
    "th.msetaccum zero\n\t"
    "th.msetoob %1\n\t"
    "th.msetrstptr %2\n\t"
    "th.ml22e8\n\t"
    "th.mmacc22.w.b\n\t"
    "th.msc22e32\n\t"
    "fence\n\t"
    :: "r"(k), "r"((size_t)2), "r"((size_t)1)
  );
}
