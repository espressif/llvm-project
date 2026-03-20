// XTHeadMatrix Zmpanel (panel-aware 2x2 tiling) C API test.
//
// Tests all public Zmpanel C-level functions from <thead_matrix.h>:
//   - 12 configuration functions (mset22adra/b/d, mset22rsba/b/d,
//     mset22m/n/k, msetrstptr, msetaccum, msetoob)
//   - 2 load functions (ml22e8, ml22e16)
//   - 2 store functions (msc22e16, msc22e32)
//   - 10 FP compute functions (mfmacc22 variants)
//   - 4 INT compute functions (mmacc22 variants)
//
// RUN: %clang_cc1 -O2 -triple riscv64 \
// RUN:   -target-feature +experimental-xtheadmatrix \
// RUN:   -target-feature +experimental-xtheadzmpanel \
// RUN:   -emit-llvm -o - %s | FileCheck %s

#include <thead_matrix.h>

// ========================================================================
// 1. Panel Configuration - Base Addresses
// ========================================================================

// CHECK-LABEL: @test_mset22adra
// CHECK: call void @llvm.riscv.th.mset22adra(i64 %
void test_mset22adra(size_t addr) {
    __riscv_th_mset22adra(addr);
}

// CHECK-LABEL: @test_mset22adrb
// CHECK: call void @llvm.riscv.th.mset22adrb(i64 %
void test_mset22adrb(size_t addr) {
    __riscv_th_mset22adrb(addr);
}

// CHECK-LABEL: @test_mset22adrd
// CHECK: call void @llvm.riscv.th.mset22adrd(i64 %
void test_mset22adrd(size_t addr) {
    __riscv_th_mset22adrd(addr);
}

// ========================================================================
// 2. Panel Configuration - Row Strides
// ========================================================================

// CHECK-LABEL: @test_mset22rsba
// CHECK: call void @llvm.riscv.th.mset22rsba(i64 %
void test_mset22rsba(size_t stride) {
    __riscv_th_mset22rsba(stride);
}

// CHECK-LABEL: @test_mset22rsbb
// CHECK: call void @llvm.riscv.th.mset22rsbb(i64 %
void test_mset22rsbb(size_t stride) {
    __riscv_th_mset22rsbb(stride);
}

// CHECK-LABEL: @test_mset22rsbd
// CHECK: call void @llvm.riscv.th.mset22rsbd(i64 %
void test_mset22rsbd(size_t stride) {
    __riscv_th_mset22rsbd(stride);
}

// ========================================================================
// 3. Panel Configuration - Tile Dimensions
// ========================================================================

// CHECK-LABEL: @test_mset22m
// CHECK: call void @llvm.riscv.th.mset22m(i64 %
void test_mset22m(size_t m) {
    __riscv_th_mset22m(m);
}

// CHECK-LABEL: @test_mset22n
// CHECK: call void @llvm.riscv.th.mset22n(i64 %
void test_mset22n(size_t n) {
    __riscv_th_mset22n(n);
}

// CHECK-LABEL: @test_mset22k
// CHECK: call void @llvm.riscv.th.mset22k(i64 %
void test_mset22k(size_t k) {
    __riscv_th_mset22k(k);
}

// ========================================================================
// 4. Panel Configuration - Control
// ========================================================================

// CHECK-LABEL: @test_msetrstptr
// CHECK: call void @llvm.riscv.th.msetrstptr(i64 %
void test_msetrstptr(size_t val) {
    __riscv_th_msetrstptr(val);
}

// CHECK-LABEL: @test_msetaccum
// CHECK: call void @llvm.riscv.th.msetaccum(i64 %
void test_msetaccum(size_t val) {
    __riscv_th_msetaccum(val);
}

// CHECK-LABEL: @test_msetoob
// CHECK: call void @llvm.riscv.th.msetoob(i64 %
void test_msetoob(size_t val) {
    __riscv_th_msetoob(val);
}

// ========================================================================
// 5. Panel Load
// ========================================================================

// CHECK-LABEL: @test_ml22e8
// CHECK: call void @llvm.riscv.th.ml22e8()
void test_ml22e8(void) {
    __riscv_th_ml22e8();
}

// CHECK-LABEL: @test_ml22e16
// CHECK: call void @llvm.riscv.th.ml22e16()
void test_ml22e16(void) {
    __riscv_th_ml22e16();
}

// ========================================================================
// 6. Panel Store
// ========================================================================

// CHECK-LABEL: @test_msc22e16
// CHECK: call void @llvm.riscv.th.msc22e16()
void test_msc22e16(void) {
    __riscv_th_msc22e16();
}

// CHECK-LABEL: @test_msc22e32
// CHECK: call void @llvm.riscv.th.msc22e32()
void test_msc22e32(void) {
    __riscv_th_msc22e32();
}

// ========================================================================
// 7. Panel FP Compute - FP8 variants
// ========================================================================

// CHECK-LABEL: @test_mfmacc22_h_e5
// CHECK: call void @llvm.riscv.th.mfmacc22.h.e5()
void test_mfmacc22_h_e5(void) {
    __riscv_th_mfmacc22_h_e5();
}

// CHECK-LABEL: @test_mfmacc22_h_e4
// CHECK: call void @llvm.riscv.th.mfmacc22.h.e4()
void test_mfmacc22_h_e4(void) {
    __riscv_th_mfmacc22_h_e4();
}

// CHECK-LABEL: @test_mfmacc22_bf16_e5
// CHECK: call void @llvm.riscv.th.mfmacc22.bf16.e5()
void test_mfmacc22_bf16_e5(void) {
    __riscv_th_mfmacc22_bf16_e5();
}

// CHECK-LABEL: @test_mfmacc22_bf16_e4
// CHECK: call void @llvm.riscv.th.mfmacc22.bf16.e4()
void test_mfmacc22_bf16_e4(void) {
    __riscv_th_mfmacc22_bf16_e4();
}

// CHECK-LABEL: @test_mfmacc22_s_e5
// CHECK: call void @llvm.riscv.th.mfmacc22.s.e5()
void test_mfmacc22_s_e5(void) {
    __riscv_th_mfmacc22_s_e5();
}

// CHECK-LABEL: @test_mfmacc22_s_e4
// CHECK: call void @llvm.riscv.th.mfmacc22.s.e4()
void test_mfmacc22_s_e4(void) {
    __riscv_th_mfmacc22_s_e4();
}

// ========================================================================
// 8. Panel FP Compute - standard FP variants
// ========================================================================

// CHECK-LABEL: @test_mfmacc22_h
// CHECK: call void @llvm.riscv.th.mfmacc22.h()
void test_mfmacc22_h(void) {
    __riscv_th_mfmacc22_h();
}

// CHECK-LABEL: @test_mfmacc22_s_h
// CHECK: call void @llvm.riscv.th.mfmacc22.s.h()
void test_mfmacc22_s_h(void) {
    __riscv_th_mfmacc22_s_h();
}

// CHECK-LABEL: @test_mfmacc22_s_bf16
// CHECK: call void @llvm.riscv.th.mfmacc22.s.bf16()
void test_mfmacc22_s_bf16(void) {
    __riscv_th_mfmacc22_s_bf16();
}

// CHECK-LABEL: @test_mfmacc22_s
// CHECK: call void @llvm.riscv.th.mfmacc22.s()
void test_mfmacc22_s(void) {
    __riscv_th_mfmacc22_s();
}

// ========================================================================
// 9. Panel INT Compute
// ========================================================================

// CHECK-LABEL: @test_mmacc22_w_b
// CHECK: call void @llvm.riscv.th.mmacc22.w.b()
void test_mmacc22_w_b(void) {
    __riscv_th_mmacc22_w_b();
}

// CHECK-LABEL: @test_mmaccu22_w_b
// CHECK: call void @llvm.riscv.th.mmaccu22.w.b()
void test_mmaccu22_w_b(void) {
    __riscv_th_mmaccu22_w_b();
}

// CHECK-LABEL: @test_mmaccus22_w_b
// CHECK: call void @llvm.riscv.th.mmaccus22.w.b()
void test_mmaccus22_w_b(void) {
    __riscv_th_mmaccus22_w_b();
}

// CHECK-LABEL: @test_mmaccsu22_w_b
// CHECK: call void @llvm.riscv.th.mmaccsu22.w.b()
void test_mmaccsu22_w_b(void) {
    __riscv_th_mmaccsu22_w_b();
}

// ========================================================================
// 10. End-to-end: INT8 panel GEMM pipeline
// ========================================================================

// CHECK-LABEL: @test_panel_int8_gemm
// CHECK: call void @llvm.riscv.th.mset22adra(i64 %
// CHECK: call void @llvm.riscv.th.mset22adrb(i64 %
// CHECK: call void @llvm.riscv.th.mset22adrd(i64 %
// CHECK: call void @llvm.riscv.th.mset22rsba(i64 %
// CHECK: call void @llvm.riscv.th.mset22rsbb(i64 %
// CHECK: call void @llvm.riscv.th.mset22rsbd(i64 %
// CHECK: call void @llvm.riscv.th.mset22m(i64 %
// CHECK: call void @llvm.riscv.th.mset22n(i64 %
// CHECK: call void @llvm.riscv.th.mset22k(i64 %
// CHECK: call void @llvm.riscv.th.msetaccum(i64 0)
// CHECK: call void @llvm.riscv.th.msetoob(i64 2)
// CHECK: call void @llvm.riscv.th.msetrstptr(i64 1)
// CHECK: call void @llvm.riscv.th.ml22e8()
// CHECK: call void @llvm.riscv.th.mmacc22.w.b()
// CHECK: call void @llvm.riscv.th.msc22e32()
void test_panel_int8_gemm(size_t a_addr, size_t b_addr, size_t d_addr,
                          size_t stride_a, size_t stride_b, size_t stride_d,
                          size_t m, size_t n, size_t k) {
    // Setup base addresses
    __riscv_th_mset22adra(a_addr);
    __riscv_th_mset22adrb(b_addr);
    __riscv_th_mset22adrd(d_addr);
    // Setup row strides
    __riscv_th_mset22rsba(stride_a);
    __riscv_th_mset22rsbb(stride_b);
    __riscv_th_mset22rsbd(stride_d);
    // Setup panel dimensions
    __riscv_th_mset22m(m);
    __riscv_th_mset22n(n);
    __riscv_th_mset22k(k);
    // Control: zero-accumulate, OOB zero-pad loads + skip stores
    __riscv_th_msetaccum(0);
    __riscv_th_msetoob(2);    // load_policy=1, store_policy=0
    __riscv_th_msetrstptr(1); // reset all HW pointers
    // Execute one panel iteration
    __riscv_th_ml22e8();
    __riscv_th_mmacc22_w_b();
    __riscv_th_msc22e32();
}

// ========================================================================
// 11. End-to-end: FP16 panel GEMM pipeline
// ========================================================================

// CHECK-LABEL: @test_panel_fp16_gemm
// CHECK: call void @llvm.riscv.th.mset22adra
// CHECK: call void @llvm.riscv.th.mset22adrb
// CHECK: call void @llvm.riscv.th.mset22adrd
// CHECK: call void @llvm.riscv.th.mset22rsba
// CHECK: call void @llvm.riscv.th.mset22rsbb
// CHECK: call void @llvm.riscv.th.mset22rsbd
// CHECK: call void @llvm.riscv.th.mset22m
// CHECK: call void @llvm.riscv.th.mset22n
// CHECK: call void @llvm.riscv.th.mset22k
// CHECK: call void @llvm.riscv.th.msetaccum
// CHECK: call void @llvm.riscv.th.msetoob
// CHECK: call void @llvm.riscv.th.msetrstptr
// CHECK: call void @llvm.riscv.th.ml22e16()
// CHECK: call void @llvm.riscv.th.mfmacc22.h()
// CHECK: call void @llvm.riscv.th.msc22e16()
void test_panel_fp16_gemm(size_t a_addr, size_t b_addr, size_t d_addr,
                          size_t stride_a, size_t stride_b, size_t stride_d,
                          size_t m, size_t n, size_t k) {
    __riscv_th_mset22adra(a_addr);
    __riscv_th_mset22adrb(b_addr);
    __riscv_th_mset22adrd(d_addr);
    __riscv_th_mset22rsba(stride_a);
    __riscv_th_mset22rsbb(stride_b);
    __riscv_th_mset22rsbd(stride_d);
    __riscv_th_mset22m(m);
    __riscv_th_mset22n(n);
    __riscv_th_mset22k(k);
    __riscv_th_msetaccum(0);
    __riscv_th_msetoob(2);
    __riscv_th_msetrstptr(1);
    // FP16 native precision
    __riscv_th_ml22e16();
    __riscv_th_mfmacc22_h();
    __riscv_th_msc22e16();
}

// ========================================================================
// 12. Panel CSR read (verify CSR constants are accessible)
// ========================================================================

// CHECK-LABEL: @test_panel_csr_read
// CHECK: call i64 asm sideeffect "csrr $0, 0xcc4"
// CHECK: call i64 asm sideeffect "csrr $0, 0xccb"
// CHECK: call i64 asm sideeffect "csrr $0, 0xccc"
// CHECK: call i64 asm sideeffect "csrr $0, 0xccd"
unsigned long test_panel_csr_read(void) {
    unsigned long ctrl, pm, pn, pk;
    __asm__ __volatile__("csrr %0, 0xcc4" : "=r"(ctrl));
    __asm__ __volatile__("csrr %0, 0xccb" : "=r"(pm));
    __asm__ __volatile__("csrr %0, 0xccc" : "=r"(pn));
    __asm__ __volatile__("csrr %0, 0xccd" : "=r"(pk));
    return ctrl + pm + pn + pk;
}
