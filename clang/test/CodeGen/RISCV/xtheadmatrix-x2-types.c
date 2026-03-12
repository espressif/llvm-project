// Comprehensive test for x2 (register-pair) type support.
// Tests mget/mset tuple operations, x2-typed matmul wrappers, backward
// compatibility with single-register APIs, and correct IR generation.
//
// RUN: %clang_cc1 -O0 -triple riscv64 -target-feature +experimental-xtheadmatrix \
// RUN:   -emit-llvm -o - %s | FileCheck --check-prefix=O0 %s
// RUN: %clang_cc1 -O2 -triple riscv64 -target-feature +experimental-xtheadmatrix \
// RUN:   -emit-llvm -o - %s | FileCheck --check-prefix=O2 %s

#include <thead_matrix.h>

// ========================================================================
// 1. x2 type representation: struct of two target("riscv.matrix")
// ========================================================================

// O0-LABEL: @test_mundef_x2
// O0: ret { target("riscv.matrix"), target("riscv.matrix") } poison
mfloat16x2_t test_mundef_x2(void) {
    return __builtin_riscv_th_mundef_f16x2();
}

// O0-LABEL: @test_mundef_single
// O0: ret target("riscv.matrix") poison
mfloat16_t test_mundef_single(void) {
    return __builtin_riscv_th_mundef_f16();
}

// ========================================================================
// 2. mget: extract from x2 pair
// ========================================================================

// O0-LABEL: @test_mget_f16_idx0
// O0: extractvalue { target("riscv.matrix"), target("riscv.matrix") } %{{.*}}, 0
// O0: extractvalue { target("riscv.matrix"), target("riscv.matrix") } %{{.*}}, 1
// O0: select
mfloat16_t test_mget_f16_idx0(mfloat16x2_t pair) {
    return __riscv_th_mget_f16(pair, 0);
}

// O0-LABEL: @test_mget_f16_idx1
mfloat16_t test_mget_f16_idx1(mfloat16x2_t pair) {
    return __riscv_th_mget_f16(pair, 1);
}

// O0-LABEL: @test_mget_i32
// O0: extractvalue { target("riscv.matrix"), target("riscv.matrix") } %{{.*}}, 0
mint32_t test_mget_i32(mint32x2_t pair) {
    return __riscv_th_mget_i32(pair, 0);
}

// O0-LABEL: @test_mget_u64
muint64_t test_mget_u64(muint64x2_t pair) {
    return __riscv_th_mget_u64(pair, 0);
}

// ========================================================================
// 3. mset: insert into x2 pair
// ========================================================================

// O0-LABEL: @test_mset_f16_idx0
// O0: insertvalue { target("riscv.matrix"), target("riscv.matrix") } %{{.*}}, target("riscv.matrix") %{{.*}}, 0
// O0: insertvalue { target("riscv.matrix"), target("riscv.matrix") } %{{.*}}, target("riscv.matrix") %{{.*}}, 1
// O0: select
mfloat16x2_t test_mset_f16_idx0(mfloat16x2_t pair, mfloat16_t val) {
    return __riscv_th_mset_f16(pair, 0, val);
}

// O0-LABEL: @test_mset_f64_idx1
mfloat64x2_t test_mset_f64_idx1(mfloat64x2_t pair, mfloat64_t val) {
    return __riscv_th_mset_f64(pair, 1, val);
}

// ========================================================================
// 4. Round-trip: mset then mget
// ========================================================================

// At O2, optimizer should fold mset+mget into a direct passthrough.
// O2-LABEL: @test_roundtrip_f16
// O2-NOT: extractvalue
// O2-NOT: insertvalue
// O2: ret target("riscv.matrix") %
mfloat16_t test_roundtrip_f16(mfloat16_t val) {
    mfloat16x2_t pair = __riscv_th_mset_f16(
        __builtin_riscv_th_mundef_f16x2(), 0, val);
    return __riscv_th_mget_f16(pair, 0);
}

// O2-LABEL: @test_roundtrip_both_slots
// O2: ret target("riscv.matrix") %
mfloat32_t test_roundtrip_both_slots(mfloat32_t a, mfloat32_t b) {
    mfloat32x2_t pair = __riscv_th_mset_f32(
        __riscv_th_mset_f32(__builtin_riscv_th_mundef_f32x2(), 0, a), 1, b);
    return __riscv_th_mget_f32(pair, 1);
}

// ========================================================================
// 5. FP16 matmul: old single-register API still works
// ========================================================================

// O0-LABEL: @test_fp16_matmul_single
// O0: call target("riscv.matrix") @llvm.riscv.th.mfmacc.h.internal
mfloat16_t test_fp16_matmul_single(mfloat16_t acc, mfloat16_t a,
                                    mfloat16_t b,
                                    mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mfmaqa_h(acc, a, b, m, k, n);
}

// ========================================================================
// 6. FP16 matmul: new x2 B operand variant
// ========================================================================

// O0-LABEL: @test_fp16_matmul_x2_b
// O0: call target("riscv.matrix") @llvm.riscv.th.mfmacc.h.internal
mfloat16_t test_fp16_matmul_x2_b(mfloat16_t acc, mfloat16_t a,
                                   mfloat16x2_t b,
                                   mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mfmaqa_h_x2(acc, a, b, m, k, n);
}

// ========================================================================
// 7. FP32 matmul: unchanged (no x2 in spec)
// ========================================================================

// O0-LABEL: @test_fp32_matmul_single
// O0: call target("riscv.matrix") @llvm.riscv.th.mfmacc.s.internal
mfloat32_t test_fp32_matmul_single(mfloat32_t acc, mfloat32_t a,
                                    mfloat32_t b,
                                    mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mfmaqa_s(acc, a, b, m, k, n);
}

// ========================================================================
// 8. FP64 matmul: old single-register API
// ========================================================================

// O0-LABEL: @test_fp64_matmul_single
// O0: call target("riscv.matrix") @llvm.riscv.th.mfmacc.d.internal
mfloat64_t test_fp64_matmul_single(mfloat64_t c, mfloat64_t a,
                                    mfloat64_t b,
                                    mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mfmaqa_d(c, a, b, m, k, n);
}

// ========================================================================
// 9. FP64 matmul: new x2 dest variant
// ========================================================================

// O0-LABEL: @test_fp64_matmul_x2_dest
// O0: call target("riscv.matrix") @llvm.riscv.th.mfmacc.d.internal
mfloat64x2_t test_fp64_matmul_x2_dest(mfloat64x2_t c, mfloat64_t a,
                                        mfloat64_t b,
                                        mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mfmaqa_d_x2(c, a, b, m, k, n);
}

// ========================================================================
// 10. FP64 widening matmul: old single and new x2
// ========================================================================

// O0-LABEL: @test_fp64_widen_single
// O0: call target("riscv.matrix") @llvm.riscv.th.mfmacc.d.s.internal
mfloat64_t test_fp64_widen_single(mfloat64_t c, mfloat32_t a,
                                   mfloat32_t b,
                                   mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mfmaqa_d_s(c, a, b, m, k, n);
}

// O0-LABEL: @test_fp64_widen_x2
// O0: call target("riscv.matrix") @llvm.riscv.th.mfmacc.d.s.internal
mfloat64x2_t test_fp64_widen_x2(mfloat64x2_t c, mfloat32_t a,
                                  mfloat32_t b,
                                  mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mfmaqa_d_s_x2(c, a, b, m, k, n);
}

// ========================================================================
// 11. INT16->INT64 matmul: old single-register APIs
// ========================================================================

// O0-LABEL: @test_int16_ss_single
// O0: call target("riscv.matrix") @llvm.riscv.th.mmacc.d.h.internal
mint64_t test_int16_ss_single(mint64_t c, mint16_t a, mint16_t b,
                               mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mmaq_ss_d_h(c, a, b, m, k, n);
}

// O0-LABEL: @test_int16_uu_single
// O0: call target("riscv.matrix") @llvm.riscv.th.mmaccu.d.h.internal
muint64_t test_int16_uu_single(muint64_t c, muint16_t a, muint16_t b,
                                mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mmaq_uu_d_h(c, a, b, m, k, n);
}

// O0-LABEL: @test_int16_us_single
// O0: call target("riscv.matrix") @llvm.riscv.th.mmaccus.d.h.internal
mint64_t test_int16_us_single(mint64_t c, muint16_t a, mint16_t b,
                               mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mmaq_us_d_h(c, a, b, m, k, n);
}

// O0-LABEL: @test_int16_su_single
// O0: call target("riscv.matrix") @llvm.riscv.th.mmaccsu.d.h.internal
mint64_t test_int16_su_single(mint64_t c, mint16_t a, muint16_t b,
                               mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mmaq_su_d_h(c, a, b, m, k, n);
}

// ========================================================================
// 12. INT16->INT64 matmul: new x2 dest variants
// ========================================================================

// O0-LABEL: @test_int16_ss_x2
// O0: call target("riscv.matrix") @llvm.riscv.th.mmacc.d.h.internal
mint64x2_t test_int16_ss_x2(mint64x2_t c, mint16_t a, mint16_t b,
                              mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mmaq_ss_d_h_x2(c, a, b, m, k, n);
}

// O0-LABEL: @test_int16_uu_x2
// O0: call target("riscv.matrix") @llvm.riscv.th.mmaccu.d.h.internal
muint64x2_t test_int16_uu_x2(muint64x2_t c, muint16_t a, muint16_t b,
                               mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mmaq_uu_d_h_x2(c, a, b, m, k, n);
}

// O0-LABEL: @test_int16_us_x2
// O0: call target("riscv.matrix") @llvm.riscv.th.mmaccus.d.h.internal
mint64x2_t test_int16_us_x2(mint64x2_t c, muint16_t a, mint16_t b,
                              mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mmaq_us_d_h_x2(c, a, b, m, k, n);
}

// O0-LABEL: @test_int16_su_x2
// O0: call target("riscv.matrix") @llvm.riscv.th.mmaccsu.d.h.internal
mint64x2_t test_int16_su_x2(mint64x2_t c, mint16_t a, muint16_t b,
                              mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mmaq_su_d_h_x2(c, a, b, m, k, n);
}

// ========================================================================
// 13. INT8->INT32 matmul: unchanged (no x2, verify still works)
// ========================================================================

// O0-LABEL: @test_int8_ss_single
// O0: call target("riscv.matrix") @llvm.riscv.th.mmacc.w.b.internal
mint32_t test_int8_ss_single(mint32_t c, mint8_t a, mint8_t b,
                              mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mmaq_ss_w_b(c, a, b, m, k, n);
}

// ========================================================================
// 14. FP widening FP16->FP32 matmul: unchanged (no x2, verify still works)
// ========================================================================

// O0-LABEL: @test_fp_widen_s_h
// O0: call target("riscv.matrix") @llvm.riscv.th.mfmacc.s.h.internal
mfloat32_t test_fp_widen_s_h(mfloat32_t c, mfloat16_t a, mfloat16_t b,
                              mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mfmaqa_s_h(c, a, b, m, k, n);
}

// ========================================================================
// 15. End-to-end: load, build x2, call x2 matmul, extract, store
// ========================================================================

// O2-LABEL: @test_e2e_fp16_x2
// O2: call target("riscv.matrix") @llvm.riscv.th.mlae.internal16
// O2: call target("riscv.matrix") @llvm.riscv.th.mlbe.internal16
// O2: call target("riscv.matrix") @llvm.riscv.th.mlce.internal16
// O2: call target("riscv.matrix") @llvm.riscv.th.mfmacc.h.internal
// O2: call void @llvm.riscv.th.msce.internal16
void test_e2e_fp16_x2(uint16_t *a_ptr, uint16_t *b_ptr,
                       uint16_t *c_ptr, long stride,
                       mrow_t m, mcol_t k, mcol_t n) {
    mfloat16_t ta = __riscv_th_mld_a_f16(a_ptr, stride, m, k);
    mfloat16_t tb = __riscv_th_mld_b_f16(b_ptr, stride, k, n);
    mfloat16_t acc = __riscv_th_mld_acc_f16(c_ptr, stride, m, n);
    mfloat16x2_t b_pair = __riscv_th_mset_f16(
        __builtin_riscv_th_mundef_f16x2(), 0, tb);
    mfloat16_t res = __riscv_th_mfmaqa_h_x2(acc, ta, b_pair, m, k, n);
    __riscv_th_mst_f16(c_ptr, stride, res, m, n);
}

// O2-LABEL: @test_e2e_fp64_x2
// O2: call target("riscv.matrix") @llvm.riscv.th.mfmacc.d.internal
void test_e2e_fp64_x2(double *a_ptr, double *b_ptr,
                       double *c_ptr, long stride,
                       mrow_t m, mcol_t k, mcol_t n) {
    mfloat64_t ta = __riscv_th_mld_a_f64(a_ptr, stride, m, k);
    mfloat64_t tb = __riscv_th_mld_b_f64(b_ptr, stride, k, n);
    mfloat64_t c0 = __riscv_th_mld_acc_f64(c_ptr, stride, m, n);
    mfloat64x2_t c_pair = __riscv_th_mset_f64(
        __builtin_riscv_th_mundef_f64x2(), 0, c0);
    mfloat64x2_t res_pair = __riscv_th_mfmaqa_d_x2(c_pair, ta, tb, m, k, n);
    mfloat64_t res = __riscv_th_mget_f64(res_pair, 0);
    __riscv_th_mst_f64(c_ptr, stride, res, m, n);
}

// O2-LABEL: @test_e2e_int16_x2
// O2: call target("riscv.matrix") @llvm.riscv.th.mmacc.d.h.internal
void test_e2e_int16_x2(int16_t *a_ptr, int16_t *b_ptr,
                        int64_t *c_ptr, long stride,
                        mrow_t m, mcol_t k, mcol_t n) {
    mint16_t ta = __riscv_th_mld_a_i16(a_ptr, stride, m, k);
    mint16_t tb = __riscv_th_mld_b_i16(b_ptr, stride, k, n);
    mint64_t c0 = __riscv_th_mld_acc_i64(c_ptr, stride, m, n);
    mint64x2_t c_pair = __riscv_th_mset_i64(
        __builtin_riscv_th_mundef_i64x2(), 0, c0);
    mint64x2_t res_pair = __riscv_th_mmaq_ss_d_h_x2(c_pair, ta, tb, m, k, n);
    mint64_t res = __riscv_th_mget_i64(res_pair, 0);
    __riscv_th_mst_i64(c_ptr, stride, res, m, n);
}
