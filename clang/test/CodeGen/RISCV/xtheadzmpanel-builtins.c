// XTHeadZmpanel (Panel-Aware 2x2 Matrix Tiling) builtins CodeGen test.
//
// Tests all 30 Zmpanel builtins: 12 config, 2 load, 2 store,
// 10 FP compute, 4 INT compute. Verifies that each builtin correctly
// lowers to its corresponding LLVM intrinsic.
//
// RUN: %clang_cc1 -triple riscv64 -target-feature +experimental-xtheadzmpanel \
// RUN:   -emit-llvm %s -o - | FileCheck %s

#include <stddef.h>

// ===================================================================
// Configuration builtins (12 total) — all take size_t, return void
// ===================================================================

// CHECK-LABEL: @test_mset22adra(
// CHECK: call void @llvm.riscv.th.mset22adra.i64(i64 %{{.*}})
void test_mset22adra(size_t val) {
  __builtin_riscv_th_mset22adra(val);
}

// CHECK-LABEL: @test_mset22adrb(
// CHECK: call void @llvm.riscv.th.mset22adrb.i64(i64 %{{.*}})
void test_mset22adrb(size_t val) {
  __builtin_riscv_th_mset22adrb(val);
}

// CHECK-LABEL: @test_mset22adrd(
// CHECK: call void @llvm.riscv.th.mset22adrd.i64(i64 %{{.*}})
void test_mset22adrd(size_t val) {
  __builtin_riscv_th_mset22adrd(val);
}

// CHECK-LABEL: @test_mset22rsba(
// CHECK: call void @llvm.riscv.th.mset22rsba.i64(i64 %{{.*}})
void test_mset22rsba(size_t val) {
  __builtin_riscv_th_mset22rsba(val);
}

// CHECK-LABEL: @test_mset22rsbb(
// CHECK: call void @llvm.riscv.th.mset22rsbb.i64(i64 %{{.*}})
void test_mset22rsbb(size_t val) {
  __builtin_riscv_th_mset22rsbb(val);
}

// CHECK-LABEL: @test_mset22rsbd(
// CHECK: call void @llvm.riscv.th.mset22rsbd.i64(i64 %{{.*}})
void test_mset22rsbd(size_t val) {
  __builtin_riscv_th_mset22rsbd(val);
}

// CHECK-LABEL: @test_mset22m(
// CHECK: call void @llvm.riscv.th.mset22m.i64(i64 %{{.*}})
void test_mset22m(size_t val) {
  __builtin_riscv_th_mset22m(val);
}

// CHECK-LABEL: @test_mset22n(
// CHECK: call void @llvm.riscv.th.mset22n.i64(i64 %{{.*}})
void test_mset22n(size_t val) {
  __builtin_riscv_th_mset22n(val);
}

// CHECK-LABEL: @test_mset22k(
// CHECK: call void @llvm.riscv.th.mset22k.i64(i64 %{{.*}})
void test_mset22k(size_t val) {
  __builtin_riscv_th_mset22k(val);
}

// CHECK-LABEL: @test_msetrstptr(
// CHECK: call void @llvm.riscv.th.msetrstptr.i64(i64 %{{.*}})
void test_msetrstptr(size_t val) {
  __builtin_riscv_th_msetrstptr(val);
}

// CHECK-LABEL: @test_msetaccum(
// CHECK: call void @llvm.riscv.th.msetaccum.i64(i64 %{{.*}})
void test_msetaccum(size_t val) {
  __builtin_riscv_th_msetaccum(val);
}

// CHECK-LABEL: @test_msetoob(
// CHECK: call void @llvm.riscv.th.msetoob.i64(i64 %{{.*}})
void test_msetoob(size_t val) {
  __builtin_riscv_th_msetoob(val);
}

// ===================================================================
// Load builtins (2 total) — no arguments, return void
// ===================================================================

// CHECK-LABEL: @test_ml22e8(
// CHECK: call void @llvm.riscv.th.ml22e8()
void test_ml22e8(void) {
  __builtin_riscv_th_ml22e8();
}

// CHECK-LABEL: @test_ml22e16(
// CHECK: call void @llvm.riscv.th.ml22e16()
void test_ml22e16(void) {
  __builtin_riscv_th_ml22e16();
}

// ===================================================================
// Store builtins (2 total) — no arguments, return void
// ===================================================================

// CHECK-LABEL: @test_msc22e16(
// CHECK: call void @llvm.riscv.th.msc22e16()
void test_msc22e16(void) {
  __builtin_riscv_th_msc22e16();
}

// CHECK-LABEL: @test_msc22e32(
// CHECK: call void @llvm.riscv.th.msc22e32()
void test_msc22e32(void) {
  __builtin_riscv_th_msc22e32();
}

// ===================================================================
// FP compute builtins (10 total) — no arguments, return void
// ===================================================================

// CHECK-LABEL: @test_mfmacc22_h_e5(
// CHECK: call void @llvm.riscv.th.mfmacc22.h.e5()
void test_mfmacc22_h_e5(void) {
  __builtin_riscv_th_mfmacc22_h_e5();
}

// CHECK-LABEL: @test_mfmacc22_h_e4(
// CHECK: call void @llvm.riscv.th.mfmacc22.h.e4()
void test_mfmacc22_h_e4(void) {
  __builtin_riscv_th_mfmacc22_h_e4();
}

// CHECK-LABEL: @test_mfmacc22_bf16_e5(
// CHECK: call void @llvm.riscv.th.mfmacc22.bf16.e5()
void test_mfmacc22_bf16_e5(void) {
  __builtin_riscv_th_mfmacc22_bf16_e5();
}

// CHECK-LABEL: @test_mfmacc22_bf16_e4(
// CHECK: call void @llvm.riscv.th.mfmacc22.bf16.e4()
void test_mfmacc22_bf16_e4(void) {
  __builtin_riscv_th_mfmacc22_bf16_e4();
}

// CHECK-LABEL: @test_mfmacc22_s_e5(
// CHECK: call void @llvm.riscv.th.mfmacc22.s.e5()
void test_mfmacc22_s_e5(void) {
  __builtin_riscv_th_mfmacc22_s_e5();
}

// CHECK-LABEL: @test_mfmacc22_s_e4(
// CHECK: call void @llvm.riscv.th.mfmacc22.s.e4()
void test_mfmacc22_s_e4(void) {
  __builtin_riscv_th_mfmacc22_s_e4();
}

// CHECK-LABEL: @test_mfmacc22_h(
// CHECK: call void @llvm.riscv.th.mfmacc22.h()
void test_mfmacc22_h(void) {
  __builtin_riscv_th_mfmacc22_h();
}

// CHECK-LABEL: @test_mfmacc22_s_h(
// CHECK: call void @llvm.riscv.th.mfmacc22.s.h()
void test_mfmacc22_s_h(void) {
  __builtin_riscv_th_mfmacc22_s_h();
}

// CHECK-LABEL: @test_mfmacc22_s_bf16(
// CHECK: call void @llvm.riscv.th.mfmacc22.s.bf16()
void test_mfmacc22_s_bf16(void) {
  __builtin_riscv_th_mfmacc22_s_bf16();
}

// CHECK-LABEL: @test_mfmacc22_s(
// CHECK: call void @llvm.riscv.th.mfmacc22.s()
void test_mfmacc22_s(void) {
  __builtin_riscv_th_mfmacc22_s();
}

// ===================================================================
// INT compute builtins (4 total) — no arguments, return void
// ===================================================================

// CHECK-LABEL: @test_mmacc22_w_b(
// CHECK: call void @llvm.riscv.th.mmacc22.w.b()
void test_mmacc22_w_b(void) {
  __builtin_riscv_th_mmacc22_w_b();
}

// CHECK-LABEL: @test_mmaccu22_w_b(
// CHECK: call void @llvm.riscv.th.mmaccu22.w.b()
void test_mmaccu22_w_b(void) {
  __builtin_riscv_th_mmaccu22_w_b();
}

// CHECK-LABEL: @test_mmaccus22_w_b(
// CHECK: call void @llvm.riscv.th.mmaccus22.w.b()
void test_mmaccus22_w_b(void) {
  __builtin_riscv_th_mmaccus22_w_b();
}

// CHECK-LABEL: @test_mmaccsu22_w_b(
// CHECK: call void @llvm.riscv.th.mmaccsu22.w.b()
void test_mmaccsu22_w_b(void) {
  __builtin_riscv_th_mmaccsu22_w_b();
}

// ===================================================================
// Combined test: INT8 panel GEMM pipeline
// ===================================================================

// CHECK-LABEL: @test_panel_gemm_int8(
// CHECK: call void @llvm.riscv.th.mset22adra.i64(i64 %{{.*}})
// CHECK: call void @llvm.riscv.th.mset22adrb.i64(i64 %{{.*}})
// CHECK: call void @llvm.riscv.th.mset22adrd.i64(i64 %{{.*}})
// CHECK: call void @llvm.riscv.th.mset22rsba.i64(i64 %{{.*}})
// CHECK: call void @llvm.riscv.th.mset22rsbb.i64(i64 %{{.*}})
// CHECK: call void @llvm.riscv.th.mset22rsbd.i64(i64 %{{.*}})
// CHECK: call void @llvm.riscv.th.mset22m.i64(i64 %{{.*}})
// CHECK: call void @llvm.riscv.th.mset22n.i64(i64 %{{.*}})
// CHECK: call void @llvm.riscv.th.mset22k.i64(i64 %{{.*}})
// CHECK: call void @llvm.riscv.th.ml22e8()
// CHECK: call void @llvm.riscv.th.mmacc22.w.b()
// CHECK: call void @llvm.riscv.th.msc22e32()
void test_panel_gemm_int8(size_t adra, size_t adrb, size_t adrd,
                           size_t rsba, size_t rsbb, size_t rsbd,
                           size_t m, size_t n, size_t k) {
  __builtin_riscv_th_mset22adra(adra);
  __builtin_riscv_th_mset22adrb(adrb);
  __builtin_riscv_th_mset22adrd(adrd);
  __builtin_riscv_th_mset22rsba(rsba);
  __builtin_riscv_th_mset22rsbb(rsbb);
  __builtin_riscv_th_mset22rsbd(rsbd);
  __builtin_riscv_th_mset22m(m);
  __builtin_riscv_th_mset22n(n);
  __builtin_riscv_th_mset22k(k);
  __builtin_riscv_th_ml22e8();
  __builtin_riscv_th_mmacc22_w_b();
  __builtin_riscv_th_msc22e32();
}

// ===================================================================
// Combined test: FP16 panel GEMM pipeline
// ===================================================================

// CHECK-LABEL: @test_panel_gemm_fp16(
// CHECK: call void @llvm.riscv.th.mset22adra.i64(i64 %{{.*}})
// CHECK: call void @llvm.riscv.th.mset22adrb.i64(i64 %{{.*}})
// CHECK: call void @llvm.riscv.th.mset22adrd.i64(i64 %{{.*}})
// CHECK: call void @llvm.riscv.th.mset22m.i64(i64 %{{.*}})
// CHECK: call void @llvm.riscv.th.mset22n.i64(i64 %{{.*}})
// CHECK: call void @llvm.riscv.th.mset22k.i64(i64 %{{.*}})
// CHECK: call void @llvm.riscv.th.ml22e16()
// CHECK: call void @llvm.riscv.th.mfmacc22.h()
// CHECK: call void @llvm.riscv.th.msc22e16()
void test_panel_gemm_fp16(size_t adra, size_t adrb, size_t adrd,
                           size_t m, size_t n, size_t k) {
  __builtin_riscv_th_mset22adra(adra);
  __builtin_riscv_th_mset22adrb(adrb);
  __builtin_riscv_th_mset22adrd(adrd);
  __builtin_riscv_th_mset22m(m);
  __builtin_riscv_th_mset22n(n);
  __builtin_riscv_th_mset22k(k);
  __builtin_riscv_th_ml22e16();
  __builtin_riscv_th_mfmacc22_h();
  __builtin_riscv_th_msc22e16();
}
