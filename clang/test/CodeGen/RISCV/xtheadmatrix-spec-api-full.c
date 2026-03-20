// Comprehensive Spec-API coverage test for all C API functions in thead_matrix.h
// that are NOT already covered by xtheadmatrix-spec-api.c or
// xtheadmatrix-verification-fixes.c.
//
// RUN: %clang_cc1 -O2 -triple riscv64 -target-feature +experimental-xtheadmatrix \
// RUN:   -emit-llvm -o - %s | FileCheck %s

#include <thead_matrix.h>

// ========================================================================
// 1. INT matmul missing variants
// ========================================================================

// CHECK-LABEL: @test_mmaccus_w_b
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mmaccus.w.b.internal
mint32_t test_mmaccus_w_b(mint32_t c, muint8_t a, mint8_t b,
                           mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mmaccus_w_b(c, a, b, m, k, n);
}

// CHECK-LABEL: @test_mmaccsu_w_b
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mmaccsu.w.b.internal
mint32_t test_mmaccsu_w_b(mint32_t c, mint8_t a, muint8_t b,
                           mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mmaccsu_w_b(c, a, b, m, k, n);
}

// CHECK-LABEL: @test_mmaccu_w_bp
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mmaccu.w.bp.internal
muint32_t test_mmaccu_w_bp(muint32_t c, muint8_t a, muint8_t b,
                           mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mmaccu_w_bp(c, a, b, m, k, n);
}

// CHECK-LABEL: @test_mmacc_p_uu_w_b
// CHECK: call target("riscv.matrix") @llvm.riscv.th.pmmaccu.w.b.internal
muint32_t test_mmacc_p_uu_w_b(muint32_t c, muint8_t a, muint8_t b,
                              mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_pmmaccu_w_b(c, a, b, m, k, n);
}

// CHECK-LABEL: @test_mmacc_p_us_w_b
// CHECK: call target("riscv.matrix") @llvm.riscv.th.pmmaccus.w.b.internal
mint32_t test_mmacc_p_us_w_b(mint32_t c, muint8_t a, mint8_t b,
                             mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_pmmaccus_w_b(c, a, b, m, k, n);
}

// CHECK-LABEL: @test_mmacc_p_su_w_b
// CHECK: call target("riscv.matrix") @llvm.riscv.th.pmmaccsu.w.b.internal
mint32_t test_mmacc_p_su_w_b(mint32_t c, mint8_t a, muint8_t b,
                             mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_pmmaccsu_w_b(c, a, b, m, k, n);
}

// ========================================================================
// 2. FP matmul missing variants
// ========================================================================

// CHECK-LABEL: @test_mfmacc_s_h
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmacc.s.h.internal
mfloat32_t test_mfmacc_s_h(mfloat32_t c, mfloat16_t a, mfloat16_t b,
                             mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mfmacc_s_h(c, a, b, m, k, n);
}

// CHECK-LABEL: @test_mfmacc_d_s
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmacc.d.s.internal
mfloat64_t test_mfmacc_d_s(mfloat64_t c, mfloat32_t a, mfloat32_t b,
                             mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mfmacc_d_s(c, a, b, m, k, n);
}

// CHECK-LABEL: @test_mfmacc_d
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmacc.d.internal
mfloat64_t test_mfmacc_d(mfloat64_t c, mfloat64_t a, mfloat64_t b,
                           mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mfmacc_d(c, a, b, m, k, n);
}

// CHECK-LABEL: @test_mfmacc_h_e5
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmacc.h.e5.internal
mfloat16_t test_mfmacc_h_e5(mfloat16_t c, mint32_t a, mint32_t b,
                              mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mfmacc_h_e5(c, a, b, m, k, n);
}

// CHECK-LABEL: @test_mfmacc_bf16_e4
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmacc.bf16.e4.internal
mfloat16_t test_mfmacc_bf16_e4(mfloat16_t c, mint32_t a, mint32_t b,
                                 mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mfmacc_bf16_e4(c, a, b, m, k, n);
}

// CHECK-LABEL: @test_mfmacc_bf16_e5
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmacc.bf16.e5.internal
mfloat16_t test_mfmacc_bf16_e5(mfloat16_t c, mint32_t a, mint32_t b,
                                 mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mfmacc_bf16_e5(c, a, b, m, k, n);
}

// CHECK-LABEL: @test_mfmacc_s_e4
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmacc.s.e4.internal
mfloat32_t test_mfmacc_s_e4(mfloat32_t c, mint32_t a, mint32_t b,
                              mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mfmacc_s_e4(c, a, b, m, k, n);
}

// CHECK-LABEL: @test_mfmacc_s_e5
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmacc.s.e5.internal
mfloat32_t test_mfmacc_s_e5(mfloat32_t c, mint32_t a, mint32_t b,
                              mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mfmacc_s_e5(c, a, b, m, k, n);
}

// CHECK-LABEL: @test_mfmacc_s_tf32
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmacc.s.tf32.internal
mfloat32_t test_mfmacc_s_tf32(mfloat32_t c, mint32_t a, mint32_t b,
                                mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mfmacc_s_tf32(c, a, b, m, k, n);
}

// ========================================================================
// 3. Element-wise INT .w.mm (missing 8)
// ========================================================================

// CHECK-LABEL: @test_mmulh_w_mm
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mmulh.w.mm.internal
mint32_t test_mmulh_w_mm(mint32_t acc, mint32_t s2, mint32_t s1) {
    return __riscv_th_mmulh_w_mm(acc, s2, s1);
}

// CHECK-LABEL: @test_mmax_w_mm
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mmax.w.mm.internal
mint32_t test_mmax_w_mm(mint32_t acc, mint32_t s2, mint32_t s1) {
    return __riscv_th_mmax_w_mm(acc, s2, s1);
}

// CHECK-LABEL: @test_mumax_w_mm
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mumax.w.mm.internal
mint32_t test_mumax_w_mm(mint32_t acc, mint32_t s2, mint32_t s1) {
    return __riscv_th_mumax_w_mm(acc, s2, s1);
}

// CHECK-LABEL: @test_mmin_w_mm
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mmin.w.mm.internal
mint32_t test_mmin_w_mm(mint32_t acc, mint32_t s2, mint32_t s1) {
    return __riscv_th_mmin_w_mm(acc, s2, s1);
}

// CHECK-LABEL: @test_mumin_w_mm
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mumin.w.mm.internal
mint32_t test_mumin_w_mm(mint32_t acc, mint32_t s2, mint32_t s1) {
    return __riscv_th_mumin_w_mm(acc, s2, s1);
}

// CHECK-LABEL: @test_msrl_w_mm
// CHECK: call target("riscv.matrix") @llvm.riscv.th.msrl.w.mm.internal
mint32_t test_msrl_w_mm(mint32_t acc, mint32_t s2, mint32_t s1) {
    return __riscv_th_msrl_w_mm(acc, s2, s1);
}

// CHECK-LABEL: @test_msll_w_mm
// CHECK: call target("riscv.matrix") @llvm.riscv.th.msll.w.mm.internal
mint32_t test_msll_w_mm(mint32_t acc, mint32_t s2, mint32_t s1) {
    return __riscv_th_msll_w_mm(acc, s2, s1);
}

// CHECK-LABEL: @test_msra_w_mm
// CHECK: call target("riscv.matrix") @llvm.riscv.th.msra.w.mm.internal
mint32_t test_msra_w_mm(mint32_t acc, mint32_t s2, mint32_t s1) {
    return __riscv_th_msra_w_mm(acc, s2, s1);
}

// ========================================================================
// 4. Element-wise INT .w.mv.i (missing 10)
// ========================================================================

// CHECK-LABEL: @test_madd_w_mv_i
// CHECK: call target("riscv.matrix") @llvm.riscv.th.madd.w.mv.i.internal
mint32_t test_madd_w_mv_i(mint32_t acc, mint32_t s2, mint32_t s1) {
    return __riscv_th_madd_w_mv_i(acc, s2, s1, 2);
}

// CHECK-LABEL: @test_mmul_w_mv_i
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mmul.w.mv.i.internal
mint32_t test_mmul_w_mv_i(mint32_t acc, mint32_t s2, mint32_t s1) {
    return __riscv_th_mmul_w_mv_i(acc, s2, s1, 1);
}

// CHECK-LABEL: @test_mmulh_w_mv_i
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mmulh.w.mv.i.internal
mint32_t test_mmulh_w_mv_i(mint32_t acc, mint32_t s2, mint32_t s1) {
    return __riscv_th_mmulh_w_mv_i(acc, s2, s1, 1);
}

// CHECK-LABEL: @test_mmax_w_mv_i
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mmax.w.mv.i.internal
mint32_t test_mmax_w_mv_i(mint32_t acc, mint32_t s2, mint32_t s1) {
    return __riscv_th_mmax_w_mv_i(acc, s2, s1, 0);
}

// CHECK-LABEL: @test_mumax_w_mv_i
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mumax.w.mv.i.internal
mint32_t test_mumax_w_mv_i(mint32_t acc, mint32_t s2, mint32_t s1) {
    return __riscv_th_mumax_w_mv_i(acc, s2, s1, 0);
}

// CHECK-LABEL: @test_mmin_w_mv_i
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mmin.w.mv.i.internal
mint32_t test_mmin_w_mv_i(mint32_t acc, mint32_t s2, mint32_t s1) {
    return __riscv_th_mmin_w_mv_i(acc, s2, s1, 1);
}

// CHECK-LABEL: @test_mumin_w_mv_i
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mumin.w.mv.i.internal
mint32_t test_mumin_w_mv_i(mint32_t acc, mint32_t s2, mint32_t s1) {
    return __riscv_th_mumin_w_mv_i(acc, s2, s1, 1);
}

// CHECK-LABEL: @test_msrl_w_mv_i
// CHECK: call target("riscv.matrix") @llvm.riscv.th.msrl.w.mv.i.internal
mint32_t test_msrl_w_mv_i(mint32_t acc, mint32_t s2, mint32_t s1) {
    return __riscv_th_msrl_w_mv_i(acc, s2, s1, 2);
}

// CHECK-LABEL: @test_msll_w_mv_i
// CHECK: call target("riscv.matrix") @llvm.riscv.th.msll.w.mv.i.internal
mint32_t test_msll_w_mv_i(mint32_t acc, mint32_t s2, mint32_t s1) {
    return __riscv_th_msll_w_mv_i(acc, s2, s1, 2);
}

// CHECK-LABEL: @test_msra_w_mv_i
// CHECK: call target("riscv.matrix") @llvm.riscv.th.msra.w.mv.i.internal
mint32_t test_msra_w_mv_i(mint32_t acc, mint32_t s2, mint32_t s1) {
    return __riscv_th_msra_w_mv_i(acc, s2, s1, 3);
}

// ========================================================================
// 5. Element-wise FP .mm (missing 13)
// ========================================================================

// CHECK-LABEL: @test_mfsub_h_mm
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfsub.h.mm.internal
mfloat16_t test_mfsub_h_mm(mfloat16_t acc, mfloat16_t s2, mfloat16_t s1) {
    return __riscv_th_mfsub_h_mm(acc, s2, s1);
}

// CHECK-LABEL: @test_mfmul_h_mm
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmul.h.mm.internal
mfloat16_t test_mfmul_h_mm(mfloat16_t acc, mfloat16_t s2, mfloat16_t s1) {
    return __riscv_th_mfmul_h_mm(acc, s2, s1);
}

// CHECK-LABEL: @test_mfmax_h_mm
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmax.h.mm.internal
mfloat16_t test_mfmax_h_mm(mfloat16_t acc, mfloat16_t s2, mfloat16_t s1) {
    return __riscv_th_mfmax_h_mm(acc, s2, s1);
}

// CHECK-LABEL: @test_mfmin_h_mm
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmin.h.mm.internal
mfloat16_t test_mfmin_h_mm(mfloat16_t acc, mfloat16_t s2, mfloat16_t s1) {
    return __riscv_th_mfmin_h_mm(acc, s2, s1);
}

// CHECK-LABEL: @test_mfsub_s_mm
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfsub.s.mm.internal
mfloat32_t test_mfsub_s_mm(mfloat32_t acc, mfloat32_t s2, mfloat32_t s1) {
    return __riscv_th_mfsub_s_mm(acc, s2, s1);
}

// CHECK-LABEL: @test_mfmax_s_mm
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmax.s.mm.internal
mfloat32_t test_mfmax_s_mm(mfloat32_t acc, mfloat32_t s2, mfloat32_t s1) {
    return __riscv_th_mfmax_s_mm(acc, s2, s1);
}

// CHECK-LABEL: @test_mfmin_s_mm
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmin.s.mm.internal
mfloat32_t test_mfmin_s_mm(mfloat32_t acc, mfloat32_t s2, mfloat32_t s1) {
    return __riscv_th_mfmin_s_mm(acc, s2, s1);
}

// CHECK-LABEL: @test_mfadd_d_mm
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfadd.d.mm.internal
mfloat64_t test_mfadd_d_mm(mfloat64_t acc, mfloat64_t s2, mfloat64_t s1) {
    return __riscv_th_mfadd_d_mm(acc, s2, s1);
}

// CHECK-LABEL: @test_mfsub_d_mm
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfsub.d.mm.internal
mfloat64_t test_mfsub_d_mm(mfloat64_t acc, mfloat64_t s2, mfloat64_t s1) {
    return __riscv_th_mfsub_d_mm(acc, s2, s1);
}

// CHECK-LABEL: @test_mfmul_d_mm
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmul.d.mm.internal
mfloat64_t test_mfmul_d_mm(mfloat64_t acc, mfloat64_t s2, mfloat64_t s1) {
    return __riscv_th_mfmul_d_mm(acc, s2, s1);
}

// CHECK-LABEL: @test_mfmax_d_mm
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmax.d.mm.internal
mfloat64_t test_mfmax_d_mm(mfloat64_t acc, mfloat64_t s2, mfloat64_t s1) {
    return __riscv_th_mfmax_d_mm(acc, s2, s1);
}

// CHECK-LABEL: @test_mfmin_d_mm
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmin.d.mm.internal
mfloat64_t test_mfmin_d_mm(mfloat64_t acc, mfloat64_t s2, mfloat64_t s1) {
    return __riscv_th_mfmin_d_mm(acc, s2, s1);
}

// mfadd_h_mm was already tested in verification-fixes; include here for
// completeness of the FP16 set.
// CHECK-LABEL: @test_mfadd_h_mm
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfadd.h.mm.internal
mfloat16_t test_mfadd_h_mm(mfloat16_t acc, mfloat16_t s2, mfloat16_t s1) {
    return __riscv_th_mfadd_h_mm(acc, s2, s1);
}

// ========================================================================
// 6. Element-wise FP .mv.i (missing 14)
// ========================================================================

// CHECK-LABEL: @test_mfadd_h_mv_i
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfadd.h.mv.i.internal
mfloat16_t test_mfadd_h_mv_i(mfloat16_t acc, mfloat16_t s2, mfloat16_t s1) {
    return __riscv_th_mfadd_h_mv_i(acc, s2, s1, 0);
}

// CHECK-LABEL: @test_mfsub_h_mv_i
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfsub.h.mv.i.internal
mfloat16_t test_mfsub_h_mv_i(mfloat16_t acc, mfloat16_t s2, mfloat16_t s1) {
    return __riscv_th_mfsub_h_mv_i(acc, s2, s1, 1);
}

// CHECK-LABEL: @test_mfmul_h_mv_i
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmul.h.mv.i.internal
mfloat16_t test_mfmul_h_mv_i(mfloat16_t acc, mfloat16_t s2, mfloat16_t s1) {
    return __riscv_th_mfmul_h_mv_i(acc, s2, s1, 2);
}

// CHECK-LABEL: @test_mfmax_h_mv_i
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmax.h.mv.i.internal
mfloat16_t test_mfmax_h_mv_i(mfloat16_t acc, mfloat16_t s2, mfloat16_t s1) {
    return __riscv_th_mfmax_h_mv_i(acc, s2, s1, 0);
}

// CHECK-LABEL: @test_mfadd_s_mv_i
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfadd.s.mv.i.internal
mfloat32_t test_mfadd_s_mv_i(mfloat32_t acc, mfloat32_t s2, mfloat32_t s1) {
    return __riscv_th_mfadd_s_mv_i(acc, s2, s1, 1);
}

// CHECK-LABEL: @test_mfsub_s_mv_i
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfsub.s.mv.i.internal
mfloat32_t test_mfsub_s_mv_i(mfloat32_t acc, mfloat32_t s2, mfloat32_t s1) {
    return __riscv_th_mfsub_s_mv_i(acc, s2, s1, 0);
}

// CHECK-LABEL: @test_mfmul_s_mv_i
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmul.s.mv.i.internal
mfloat32_t test_mfmul_s_mv_i(mfloat32_t acc, mfloat32_t s2, mfloat32_t s1) {
    return __riscv_th_mfmul_s_mv_i(acc, s2, s1, 2);
}

// CHECK-LABEL: @test_mfmax_s_mv_i
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmax.s.mv.i.internal
mfloat32_t test_mfmax_s_mv_i(mfloat32_t acc, mfloat32_t s2, mfloat32_t s1) {
    return __riscv_th_mfmax_s_mv_i(acc, s2, s1, 0);
}

// CHECK-LABEL: @test_mfmin_s_mv_i
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmin.s.mv.i.internal
mfloat32_t test_mfmin_s_mv_i(mfloat32_t acc, mfloat32_t s2, mfloat32_t s1) {
    return __riscv_th_mfmin_s_mv_i(acc, s2, s1, 1);
}

// CHECK-LABEL: @test_mfadd_d_mv_i
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfadd.d.mv.i.internal
mfloat64_t test_mfadd_d_mv_i(mfloat64_t acc, mfloat64_t s2, mfloat64_t s1) {
    return __riscv_th_mfadd_d_mv_i(acc, s2, s1, 0);
}

// CHECK-LABEL: @test_mfsub_d_mv_i
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfsub.d.mv.i.internal
mfloat64_t test_mfsub_d_mv_i(mfloat64_t acc, mfloat64_t s2, mfloat64_t s1) {
    return __riscv_th_mfsub_d_mv_i(acc, s2, s1, 1);
}

// CHECK-LABEL: @test_mfmul_d_mv_i
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmul.d.mv.i.internal
mfloat64_t test_mfmul_d_mv_i(mfloat64_t acc, mfloat64_t s2, mfloat64_t s1) {
    return __riscv_th_mfmul_d_mv_i(acc, s2, s1, 2);
}

// CHECK-LABEL: @test_mfmax_d_mv_i
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmax.d.mv.i.internal
mfloat64_t test_mfmax_d_mv_i(mfloat64_t acc, mfloat64_t s2, mfloat64_t s1) {
    return __riscv_th_mfmax_d_mv_i(acc, s2, s1, 0);
}

// CHECK-LABEL: @test_mfmin_d_mv_i
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmin.d.mv.i.internal
mfloat64_t test_mfmin_d_mv_i(mfloat64_t acc, mfloat64_t s2, mfloat64_t s1) {
    return __riscv_th_mfmin_d_mv_i(acc, s2, s1, 1);
}

// ========================================================================
// 7. FP format conversions (missing ~20)
// ========================================================================

// CHECK-LABEL: @test_mfcvth_s_h
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfcvth.s.h.internal
mfloat32_t test_mfcvth_s_h(mfloat16_t src) {
    return __riscv_th_mfcvth_s_h(src);
}

// CHECK-LABEL: @test_mfcvtl_h_s
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfcvtl.h.s.internal
mfloat16_t test_mfcvtl_h_s(mfloat32_t src) {
    return __riscv_th_mfcvtl_h_s(src);
}

// mfcvth_h_s already in verification-fixes; skip

// CHECK-LABEL: @test_mfcvth_d_s
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfcvth.d.s.internal
mfloat64_t test_mfcvth_d_s(mfloat32_t src) {
    return __riscv_th_mfcvth_d_s(src);
}

// CHECK-LABEL: @test_mfcvtl_s_d
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfcvtl.s.d.internal
mfloat32_t test_mfcvtl_s_d(mfloat64_t src) {
    return __riscv_th_mfcvtl_s_d(src);
}

// CHECK-LABEL: @test_mfcvth_s_d
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfcvth.s.d.internal
mfloat32_t test_mfcvth_s_d(mfloat64_t src) {
    return __riscv_th_mfcvth_s_d(src);
}

// FP8 E4 <-> FP16 (opaque)

// CHECK-LABEL: @test_mfcvtl_h_e4
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfcvtl.h.e4.internal
mint32_t test_mfcvtl_h_e4(mint32_t src) {
    return __riscv_th_mfcvtl_h_e4(src);
}

// CHECK-LABEL: @test_mfcvth_h_e4
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfcvth.h.e4.internal
mint32_t test_mfcvth_h_e4(mint32_t src) {
    return __riscv_th_mfcvth_h_e4(src);
}

// CHECK-LABEL: @test_mfcvtl_h_e5
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfcvtl.h.e5.internal
mint32_t test_mfcvtl_h_e5(mint32_t src) {
    return __riscv_th_mfcvtl_h_e5(src);
}

// CHECK-LABEL: @test_mfcvth_h_e5
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfcvth.h.e5.internal
mint32_t test_mfcvth_h_e5(mint32_t src) {
    return __riscv_th_mfcvth_h_e5(src);
}

// FP16 -> FP8 (opaque)

// CHECK-LABEL: @test_mfcvtl_e4_h
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfcvtl.e4.h.internal
mint32_t test_mfcvtl_e4_h(mint32_t src) {
    return __riscv_th_mfcvtl_e4_h(src);
}

// CHECK-LABEL: @test_mfcvth_e4_h
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfcvth.e4.h.internal
mint32_t test_mfcvth_e4_h(mint32_t src) {
    return __riscv_th_mfcvth_e4_h(src);
}

// CHECK-LABEL: @test_mfcvtl_e5_h
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfcvtl.e5.h.internal
mint32_t test_mfcvtl_e5_h(mint32_t src) {
    return __riscv_th_mfcvtl_e5_h(src);
}

// CHECK-LABEL: @test_mfcvth_e5_h
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfcvth.e5.h.internal
mint32_t test_mfcvth_e5_h(mint32_t src) {
    return __riscv_th_mfcvth_e5_h(src);
}

// BF16 <-> FP32 (opaque)

// CHECK-LABEL: @test_mfcvtl_s_bf16
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfcvtl.s.bf16.internal
mint32_t test_mfcvtl_s_bf16(mint32_t src) {
    return __riscv_th_mfcvtl_s_bf16(src);
}

// CHECK-LABEL: @test_mfcvth_s_bf16
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfcvth.s.bf16.internal
mint32_t test_mfcvth_s_bf16(mint32_t src) {
    return __riscv_th_mfcvth_s_bf16(src);
}

// CHECK-LABEL: @test_mfcvtl_bf16_s
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfcvtl.bf16.s.internal
mint32_t test_mfcvtl_bf16_s(mint32_t src) {
    return __riscv_th_mfcvtl_bf16_s(src);
}

// CHECK-LABEL: @test_mfcvth_bf16_s
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfcvth.bf16.s.internal
mint32_t test_mfcvth_bf16_s(mint32_t src) {
    return __riscv_th_mfcvth_bf16_s(src);
}

// FP8 <-> FP32 (opaque)

// CHECK-LABEL: @test_mfcvtl_e4_s
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfcvtl.e4.s.internal
mint32_t test_mfcvtl_e4_s(mint32_t src) {
    return __riscv_th_mfcvtl_e4_s(src);
}

// CHECK-LABEL: @test_mfcvth_e4_s
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfcvth.e4.s.internal
mint32_t test_mfcvth_e4_s(mint32_t src) {
    return __riscv_th_mfcvth_e4_s(src);
}

// CHECK-LABEL: @test_mfcvtl_e5_s
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfcvtl.e5.s.internal
mint32_t test_mfcvtl_e5_s(mint32_t src) {
    return __riscv_th_mfcvtl_e5_s(src);
}

// CHECK-LABEL: @test_mfcvth_e5_s
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfcvth.e5.s.internal
mint32_t test_mfcvth_e5_s(mint32_t src) {
    return __riscv_th_mfcvth_e5_s(src);
}

// TF32 <-> FP32 (opaque)

// CHECK-LABEL: @test_mfcvt_s_tf32
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfcvt.s.tf32.internal
mint32_t test_mfcvt_s_tf32(mint32_t src) {
    return __riscv_th_mfcvt_s_tf32(src);
}

// CHECK-LABEL: @test_mfcvt_tf32_s
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfcvt.tf32.s.internal
mint32_t test_mfcvt_tf32_s(mint32_t src) {
    return __riscv_th_mfcvt_tf32_s(src);
}

// ========================================================================
// 8. Float-int conversions (missing 10)
// ========================================================================

// CHECK-LABEL: @test_mufcvt_s_w
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mufcvt.s.w.internal
mfloat32_t test_mufcvt_s_w(muint32_t src) {
    return __riscv_th_mufcvt_s_w(src);
}

// CHECK-LABEL: @test_mfucvt_w_s
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfucvt.w.s.internal
muint32_t test_mfucvt_w_s(mfloat32_t src) {
    return __riscv_th_mfucvt_w_s(src);
}

// CHECK-LABEL: @test_mufcvtl_h_b
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mufcvtl.h.b.internal
mfloat16_t test_mufcvtl_h_b(muint8_t src) {
    return __riscv_th_mufcvtl_h_b(src);
}

// CHECK-LABEL: @test_mufcvth_h_b
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mufcvth.h.b.internal
mfloat16_t test_mufcvth_h_b(muint8_t src) {
    return __riscv_th_mufcvth_h_b(src);
}

// CHECK-LABEL: @test_mfucvtl_b_h
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfucvtl.b.h.internal
muint8_t test_mfucvtl_b_h(mfloat16_t src) {
    return __riscv_th_mfucvtl_b_h(src);
}

// CHECK-LABEL: @test_mfucvth_b_h
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfucvth.b.h.internal
muint8_t test_mfucvth_b_h(mfloat16_t src) {
    return __riscv_th_mfucvth_b_h(src);
}

// CHECK-LABEL: @test_msfcvtl_h_b
// CHECK: call target("riscv.matrix") @llvm.riscv.th.msfcvtl.h.b.internal
mfloat16_t test_msfcvtl_h_b(mint8_t src) {
    return __riscv_th_msfcvtl_h_b(src);
}

// CHECK-LABEL: @test_msfcvth_h_b
// CHECK: call target("riscv.matrix") @llvm.riscv.th.msfcvth.h.b.internal
mfloat16_t test_msfcvth_h_b(mint8_t src) {
    return __riscv_th_msfcvth_h_b(src);
}

// CHECK-LABEL: @test_mfscvtl_b_h
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfscvtl.b.h.internal
mint8_t test_mfscvtl_b_h(mfloat16_t src) {
    return __riscv_th_mfscvtl_b_h(src);
}

// CHECK-LABEL: @test_mfscvth_b_h
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfscvth.b.h.internal
mint8_t test_mfscvth_b_h(mfloat16_t src) {
    return __riscv_th_mfscvth_b_h(src);
}

// ========================================================================
// 9. N4clip (missing 7)
// ========================================================================

// CHECK-LABEL: @test_mn4cliph_w_mm
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mn4cliph.w.mm.internal
mint32_t test_mn4cliph_w_mm(mint32_t acc, mint32_t s2, mint32_t s1) {
    return __riscv_th_mn4cliph_w_mm(acc, s2, s1);
}

// CHECK-LABEL: @test_mn4cliplu_w_mm
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mn4cliplu.w.mm.internal
mint32_t test_mn4cliplu_w_mm(mint32_t acc, mint32_t s2, mint32_t s1) {
    return __riscv_th_mn4cliplu_w_mm(acc, s2, s1);
}

// CHECK-LABEL: @test_mn4cliphu_w_mm
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mn4cliphu.w.mm.internal
mint32_t test_mn4cliphu_w_mm(mint32_t acc, mint32_t s2, mint32_t s1) {
    return __riscv_th_mn4cliphu_w_mm(acc, s2, s1);
}

// CHECK-LABEL: @test_mn4clipl_w_mv_i
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mn4clipl.w.mv.i.internal
mint32_t test_mn4clipl_w_mv_i(mint32_t acc, mint32_t s2, mint32_t s1) {
    return __riscv_th_mn4clipl_w_mv_i(acc, s2, s1, 1);
}

// CHECK-LABEL: @test_mn4cliph_w_mv_i
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mn4cliph.w.mv.i.internal
mint32_t test_mn4cliph_w_mv_i(mint32_t acc, mint32_t s2, mint32_t s1) {
    return __riscv_th_mn4cliph_w_mv_i(acc, s2, s1, 2);
}

// CHECK-LABEL: @test_mn4cliplu_w_mv_i
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mn4cliplu.w.mv.i.internal
mint32_t test_mn4cliplu_w_mv_i(mint32_t acc, mint32_t s2, mint32_t s1) {
    return __riscv_th_mn4cliplu_w_mv_i(acc, s2, s1, 0);
}

// CHECK-LABEL: @test_mn4cliphu_w_mv_i
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mn4cliphu.w.mv.i.internal
mint32_t test_mn4cliphu_w_mv_i(mint32_t acc, mint32_t s2, mint32_t s1) {
    return __riscv_th_mn4cliphu_w_mv_i(acc, s2, s1, 3);
}

// ========================================================================
// 10. Packed conversions (all 4)
// ========================================================================

// CHECK-LABEL: @test_mucvtl_b_p
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mucvtl.b.p.internal
mint32_t test_mucvtl_b_p(mint32_t src) {
    return __riscv_th_mucvtl_b_p(src);
}

// CHECK-LABEL: @test_mscvtl_b_p
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mscvtl.b.p.internal
mint32_t test_mscvtl_b_p(mint32_t src) {
    return __riscv_th_mscvtl_b_p(src);
}

// CHECK-LABEL: @test_mucvth_b_p
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mucvth.b.p.internal
mint32_t test_mucvth_b_p(mint32_t src) {
    return __riscv_th_mucvth_b_p(src);
}

// CHECK-LABEL: @test_mscvth_b_p
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mscvth.b.p.internal
mint32_t test_mscvth_b_p(mint32_t src) {
    return __riscv_th_mscvth_b_p(src);
}

// ========================================================================
// 11. Slide missing variants
// ========================================================================

// CHECK-LABEL: @test_mcslidedown_h
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mcslidedown.h.internal
mint32_t test_mcslidedown_h(mint32_t src) {
    return __riscv_th_mcslidedown_h(src, 2);
}

// CHECK-LABEL: @test_mcslidedown_d
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mcslidedown.d.internal
mint32_t test_mcslidedown_d(mint32_t src) {
    return __riscv_th_mcslidedown_d(src, 1);
}

// CHECK-LABEL: @test_mcslideup_h
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mcslideup.h.internal
mint32_t test_mcslideup_h(mint32_t src) {
    return __riscv_th_mcslideup_h(src, 1);
}

// CHECK-LABEL: @test_mcslideup_w
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mcslideup.w.internal
mint32_t test_mcslideup_w(mint32_t src) {
    return __riscv_th_mcslideup_w(src, 2);
}

// CHECK-LABEL: @test_mcslideup_d
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mcslideup.d.internal
mint32_t test_mcslideup_d(mint32_t src) {
    return __riscv_th_mcslideup_d(src, 3);
}

// ========================================================================
// 12. Broadcast missing variants
// ========================================================================

// CHECK-LABEL: @test_mcbca_b
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mcbcab.mv.i.internal
mint32_t test_mcbca_b(mint32_t src) {
    return __riscv_th_mcbca_b(src, 0);
}

// CHECK-LABEL: @test_mcbca_h
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mcbcah.mv.i.internal
mint32_t test_mcbca_h(mint32_t src) {
    return __riscv_th_mcbca_h(src, 1);
}

// CHECK-LABEL: @test_mcbca_d
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mcbcad.mv.i.internal
mint32_t test_mcbca_d(mint32_t src) {
    return __riscv_th_mcbca_d(src, 2);
}

// ========================================================================
// 13. Move/dup missing variants
// ========================================================================

// --- Matrix-to-GPR (element extract) ---

// CHECK-LABEL: @test_mmovb_x_m
// CHECK: call i64 @llvm.riscv.th.mmovb.x.m.internal
unsigned long test_mmovb_x_m(mint32_t src, unsigned long idx) {
    return __riscv_th_mmovb_x_m(src, idx);
}

// CHECK-LABEL: @test_mmovh_x_m
// CHECK: call i64 @llvm.riscv.th.mmovh.x.m.internal
unsigned long test_mmovh_x_m(mint32_t src, unsigned long idx) {
    return __riscv_th_mmovh_x_m(src, idx);
}

// CHECK-LABEL: @test_mmovd_x_m
// CHECK: call i64 @llvm.riscv.th.mmovd.x.m.internal
unsigned long test_mmovd_x_m(mint32_t src, unsigned long idx) {
    return __riscv_th_mmovd_x_m(src, idx);
}

// --- GPR-to-matrix (element insert) ---

// CHECK-LABEL: @test_mmovb_m_x
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mmovb.m.x.internal
mint32_t test_mmovb_m_x(mint32_t dst, unsigned long data, unsigned long idx) {
    return __riscv_th_mmovb_m_x(dst, data, idx);
}

// CHECK-LABEL: @test_mmovh_m_x
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mmovh.m.x.internal
mint32_t test_mmovh_m_x(mint32_t dst, unsigned long data, unsigned long idx) {
    return __riscv_th_mmovh_m_x(dst, data, idx);
}

// CHECK-LABEL: @test_mmovd_m_x
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mmovd.m.x.internal
mint32_t test_mmovd_m_x(mint32_t dst, unsigned long data, unsigned long idx) {
    return __riscv_th_mmovd_m_x(dst, data, idx);
}

// --- Duplicate GPR to matrix column ---

// CHECK-LABEL: @test_mdupb_m_x
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mdupb.m.x.internal
mint32_t test_mdupb_m_x(mint32_t dst, unsigned long data) {
    return __riscv_th_mdupb_m_x(dst, data);
}

// CHECK-LABEL: @test_mduph_m_x
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mduph.m.x.internal
mint32_t test_mduph_m_x(mint32_t dst, unsigned long data) {
    return __riscv_th_mduph_m_x(dst, data);
}

// CHECK-LABEL: @test_mdupd_m_x
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mdupd.m.x.internal
mint32_t test_mdupd_m_x(mint32_t dst, unsigned long data) {
    return __riscv_th_mdupd_m_x(dst, data);
}

// ========================================================================
// 14. Load/store missing variants
// ========================================================================

// --- i64 loads ---

// CHECK-LABEL: @test_mld_a_i64
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mlae.internal64
mint64_t test_mld_a_i64(int64_t *base, long stride, mrow_t m, mcol_t k) {
    return __riscv_th_mld_a_i64(base, stride, m, k);
}

// CHECK-LABEL: @test_mld_b_i64
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mlbe.internal64
mint64_t test_mld_b_i64(int64_t *base, long stride, mcol_t k, mcol_t n) {
    return __riscv_th_mld_b_i64(base, stride, k, n);
}

// CHECK-LABEL: @test_mld_acc_i64
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mlce.internal64
mint64_t test_mld_acc_i64(int64_t *base, long stride, mrow_t m, mcol_t n) {
    return __riscv_th_mld_acc_i64(base, stride, m, n);
}

// --- u8/u16 loads ---

// CHECK-LABEL: @test_mld_a_u8
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mlae.internal8
muint8_t test_mld_a_u8(uint8_t *base, long stride, mrow_t m, mcol_t k) {
    return __riscv_th_mld_a_u8(base, stride, m, k);
}

// CHECK-LABEL: @test_mld_b_u16
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mlbe.internal16
muint16_t test_mld_b_u16(uint16_t *base, long stride, mcol_t k, mcol_t n) {
    return __riscv_th_mld_b_u16(base, stride, k, n);
}

// --- Transposed loads ---

// CHECK-LABEL: @test_mld_at_i8
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mlate.internal8
mint8_t test_mld_at_i8(int8_t *base, long stride, mrow_t m, mcol_t k) {
    return __riscv_th_mld_at_i8(base, stride, m, k);
}

// CHECK-LABEL: @test_mld_bt_i32
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mlbte.internal32
mint32_t test_mld_bt_i32(int32_t *base, long stride, mcol_t k, mcol_t n) {
    return __riscv_th_mld_bt_i32(base, stride, k, n);
}

// CHECK-LABEL: @test_mld_ct_f32
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mlcte.internal32
mfloat32_t test_mld_ct_f32(float *base, long stride, mrow_t m, mcol_t n) {
    return __riscv_th_mld_ct_f32(base, stride, m, n);
}

// --- i64 stores ---

// CHECK-LABEL: @test_mst_i64
// CHECK: call void @llvm.riscv.th.msce.internal64
void test_mst_i64(int64_t *base, long stride, mint64_t val,
                   mrow_t m, mcol_t n) {
    __riscv_th_mst_i64(base, stride, val, m, n);
}

// --- A/B tile stores ---

// CHECK-LABEL: @test_mst_a_i32
// CHECK: call void @llvm.riscv.th.msae.internal32
void test_mst_a_i32(int32_t *base, long stride, mint32_t val,
                     mrow_t m, mcol_t k) {
    __riscv_th_mst_a_i32(base, stride, val, m, k);
}

// CHECK-LABEL: @test_mst_b_i8
// CHECK: call void @llvm.riscv.th.msbe.internal8
void test_mst_b_i8(int8_t *base, long stride, mint8_t val,
                    mcol_t k, mcol_t n) {
    __riscv_th_mst_b_i8(base, stride, val, k, n);
}

// --- Transposed stores ---

// CHECK-LABEL: @test_mst_at_i16
// CHECK: call void @llvm.riscv.th.msate.internal16
void test_mst_at_i16(int16_t *base, long stride, mint16_t val,
                      mrow_t m, mcol_t k) {
    __riscv_th_mst_at_i16(base, stride, val, m, k);
}

// CHECK-LABEL: @test_mst_bt_i32
// CHECK: call void @llvm.riscv.th.msbte.internal32
void test_mst_bt_i32(int32_t *base, long stride, mint32_t val,
                      mcol_t k, mcol_t n) {
    __riscv_th_mst_bt_i32(base, stride, val, k, n);
}

// CHECK-LABEL: @test_mst_ct_i64
// CHECK: call void @llvm.riscv.th.mscte.internal64
void test_mst_ct_i64(int64_t *base, long stride, mint64_t val,
                      mrow_t m, mcol_t n) {
    __riscv_th_mst_ct_i64(base, stride, val, m, n);
}

// --- Whole-register ---

// CHECK-LABEL: @test_mld_m_i64
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mlme.internal64
mint64_t test_mld_m_i64(int64_t *base, long stride) {
    return __riscv_th_mld_m_i64(base, stride);
}

// CHECK-LABEL: @test_mst_m_f64
// CHECK: call void @llvm.riscv.th.msme.internal64
void test_mst_m_f64(double *base, long stride, mfloat64_t val) {
    __riscv_th_mst_m_f64(base, stride, val);
}

// ========================================================================
// 15. Config
// ========================================================================

// CHECK-LABEL: @test_msetmcol_e8
// CHECK: call void @llvm.riscv.th.msettilek(i64 %c)
mcol_t test_msetmcol_e8(mcol_t c) {
    return __riscv_th_msetmcol_e8(c);
}

// CHECK-LABEL: @test_msetmcol_e16
// CHECK: call void @llvm.riscv.th.msettilek(i64 %c)
mcol_t test_msetmcol_e16(mcol_t c) {
    return __riscv_th_msetmcol_e16(c);
}

// CHECK-LABEL: @test_msetmcol_e64
// CHECK: call void @llvm.riscv.th.msettilek(i64 %c)
mcol_t test_msetmcol_e64(mcol_t c) {
    return __riscv_th_msetmcol_e64(c);
}

// CHECK-LABEL: @test_msettilemi
// CHECK: call void @llvm.riscv.th.msettilemi(i64 4)
void test_msettilemi(void) {
    __riscv_th_msettilemi(4);
}

// CHECK-LABEL: @test_msettileki
// CHECK: call void @llvm.riscv.th.msettileki(i64 8)
void test_msettileki(void) {
    __riscv_th_msettileki(8);
}

// CHECK-LABEL: @test_msettileni
// CHECK: call void @llvm.riscv.th.msettileni(i64 16)
void test_msettileni(void) {
    __riscv_th_msettileni(16);
}
