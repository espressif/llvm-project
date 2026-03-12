// RUN: %clang_cc1 -triple riscv64 -target-feature +experimental-xtheadzmpanel \
// RUN:   -emit-llvm %s -o - | FileCheck %s

// Test the thead_matrix.h header API for Zmpanel panel-aware instructions.

#include <thead_matrix.h>

// CHECK-LABEL: @test_header_config(
// CHECK: call void @llvm.riscv.th.mset22adra.i64(i64 %{{.*}})
// CHECK: call void @llvm.riscv.th.mset22adrb.i64(i64 %{{.*}})
// CHECK: call void @llvm.riscv.th.mset22adrd.i64(i64 %{{.*}})
// CHECK: call void @llvm.riscv.th.mset22rsba.i64(i64 %{{.*}})
// CHECK: call void @llvm.riscv.th.mset22rsbb.i64(i64 %{{.*}})
// CHECK: call void @llvm.riscv.th.mset22rsbd.i64(i64 %{{.*}})
// CHECK: call void @llvm.riscv.th.mset22m.i64(i64 %{{.*}})
// CHECK: call void @llvm.riscv.th.mset22n.i64(i64 %{{.*}})
// CHECK: call void @llvm.riscv.th.mset22k.i64(i64 %{{.*}})
// CHECK: call void @llvm.riscv.th.msetrstptr.i64(i64 %{{.*}})
// CHECK: call void @llvm.riscv.th.msetaccum.i64(i64 %{{.*}})
// CHECK: call void @llvm.riscv.th.msetoob.i64(i64 %{{.*}})
void test_header_config(size_t a, size_t b, size_t d,
                        size_t rsba, size_t rsbb, size_t rsbd,
                        size_t m, size_t n, size_t k) {
  __riscv_th_mset22adra(a);
  __riscv_th_mset22adrb(b);
  __riscv_th_mset22adrd(d);
  __riscv_th_mset22rsba(rsba);
  __riscv_th_mset22rsbb(rsbb);
  __riscv_th_mset22rsbd(rsbd);
  __riscv_th_mset22m(m);
  __riscv_th_mset22n(n);
  __riscv_th_mset22k(k);
  __riscv_th_msetrstptr(1);
  __riscv_th_msetaccum(0);
  __riscv_th_msetoob(2);
}

// CHECK-LABEL: @test_header_load_store(
// CHECK: call void @llvm.riscv.th.ml22e8()
// CHECK: call void @llvm.riscv.th.ml22e16()
// CHECK: call void @llvm.riscv.th.msc22e16()
// CHECK: call void @llvm.riscv.th.msc22e32()
void test_header_load_store(void) {
  __riscv_th_ml22e8();
  __riscv_th_ml22e16();
  __riscv_th_msc22e16();
  __riscv_th_msc22e32();
}

// CHECK-LABEL: @test_header_fp_compute(
// CHECK: call void @llvm.riscv.th.mfmacc22.h.e5()
// CHECK: call void @llvm.riscv.th.mfmacc22.h.e4()
// CHECK: call void @llvm.riscv.th.mfmacc22.bf16.e5()
// CHECK: call void @llvm.riscv.th.mfmacc22.bf16.e4()
// CHECK: call void @llvm.riscv.th.mfmacc22.s.e5()
// CHECK: call void @llvm.riscv.th.mfmacc22.s.e4()
// CHECK: call void @llvm.riscv.th.mfmacc22.h()
// CHECK: call void @llvm.riscv.th.mfmacc22.s.h()
// CHECK: call void @llvm.riscv.th.mfmacc22.s.bf16()
// CHECK: call void @llvm.riscv.th.mfmacc22.s()
void test_header_fp_compute(void) {
  __riscv_th_mfmacc22_h_e5();
  __riscv_th_mfmacc22_h_e4();
  __riscv_th_mfmacc22_bf16_e5();
  __riscv_th_mfmacc22_bf16_e4();
  __riscv_th_mfmacc22_s_e5();
  __riscv_th_mfmacc22_s_e4();
  __riscv_th_mfmacc22_h();
  __riscv_th_mfmacc22_s_h();
  __riscv_th_mfmacc22_s_bf16();
  __riscv_th_mfmacc22_s();
}

// CHECK-LABEL: @test_header_int_compute(
// CHECK: call void @llvm.riscv.th.mmacc22.w.b()
// CHECK: call void @llvm.riscv.th.mmaccu22.w.b()
// CHECK: call void @llvm.riscv.th.mmaccus22.w.b()
// CHECK: call void @llvm.riscv.th.mmaccsu22.w.b()
void test_header_int_compute(void) {
  __riscv_th_mmacc22_w_b();
  __riscv_th_mmaccu22_w_b();
  __riscv_th_mmaccus22_w_b();
  __riscv_th_mmaccsu22_w_b();
}

// Test a complete panel GEMM pipeline via header API
// CHECK-LABEL: @test_header_panel_gemm(
// CHECK: call void @llvm.riscv.th.mset22adra.i64
// CHECK: call void @llvm.riscv.th.mset22adrb.i64
// CHECK: call void @llvm.riscv.th.mset22adrd.i64
// CHECK: call void @llvm.riscv.th.mset22rsba.i64
// CHECK: call void @llvm.riscv.th.mset22rsbb.i64
// CHECK: call void @llvm.riscv.th.mset22rsbd.i64
// CHECK: call void @llvm.riscv.th.mset22m.i64
// CHECK: call void @llvm.riscv.th.mset22n.i64
// CHECK: call void @llvm.riscv.th.mset22k.i64
// CHECK: call void @llvm.riscv.th.msetaccum.i64
// CHECK: call void @llvm.riscv.th.msetoob.i64
// CHECK: call void @llvm.riscv.th.msetrstptr.i64
// CHECK: call void @llvm.riscv.th.ml22e8()
// CHECK: call void @llvm.riscv.th.mmacc22.w.b()
// CHECK: call void @llvm.riscv.th.msc22e32()
void test_header_panel_gemm(void *a, void *b, void *d,
                            size_t rsba, size_t rsbb, size_t rsbd,
                            size_t m, size_t n, size_t k) {
  // Setup
  __riscv_th_mset22adra((size_t)a);
  __riscv_th_mset22adrb((size_t)b);
  __riscv_th_mset22adrd((size_t)d);
  __riscv_th_mset22rsba(rsba);
  __riscv_th_mset22rsbb(rsbb);
  __riscv_th_mset22rsbd(rsbd);
  __riscv_th_mset22m(m);
  __riscv_th_mset22n(n);
  __riscv_th_mset22k(k);
  __riscv_th_msetaccum(0);    // zero mode
  __riscv_th_msetoob(2);      // load_policy=1, store_policy=0
  __riscv_th_msetrstptr(1);   // reset pointers

  // Execute: load, compute, store
  __riscv_th_ml22e8();
  __riscv_th_mmacc22_w_b();
  __riscv_th_msc22e32();
}
