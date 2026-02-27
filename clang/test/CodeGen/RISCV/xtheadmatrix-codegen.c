// NOTE: XTHeadMatrix end-to-end CodeGen test: C builtins -> RISC-V assembly.
// Tests a representative sample from all instruction categories.
//
// RUN: %clang_cc1 -O2 -triple riscv64 -target-feature +experimental-xtheadmatrix -S -o - %s \
// RUN:   | FileCheck %s

#include <stdint.h>

// CHECK-LABEL: test_config:
// CHECK: th.mrelease
// CHECK: th.msettilemi 8
// CHECK: th.msettilem
// CHECK: th.msettileki 16
// CHECK: th.msettilek
// CHECK: th.msettileni 4
// CHECK: th.msettilen
void test_config(long val) {
    __builtin_riscv_th_mrelease();
    __builtin_riscv_th_msettilemi(8);
    __builtin_riscv_th_msettilem(val);
    __builtin_riscv_th_msettileki(16);
    __builtin_riscv_th_msettilek(val);
    __builtin_riscv_th_msettileni(4);
    __builtin_riscv_th_msettilen(val);
}

// CHECK-LABEL: test_load_store:
// CHECK: th.mlae8 tr0
// CHECK: th.mlbe8 tr1
// CHECK: th.mlce32 acc0
// CHECK: th.mlate16 tr2
// CHECK: th.mlbte32 tr3
// CHECK: th.mlcte64 acc1
// CHECK: th.mlme8 tr0
// CHECK: th.msae8 tr0
// CHECK: th.msbe8 tr1
// CHECK: th.msce32 acc0
// CHECK: th.msme16 tr0
void test_load_store(void *p, long s) {
    __builtin_riscv_th_mlae8(p, s);
    __builtin_riscv_th_mlbe8(p, s);
    __builtin_riscv_th_mlce32(p, s);
    __builtin_riscv_th_mlate16(p, s);
    __builtin_riscv_th_mlbte32(p, s);
    __builtin_riscv_th_mlcte64(p, s);
    __builtin_riscv_th_mlme8(p);
    __builtin_riscv_th_msae8(p, s);
    __builtin_riscv_th_msbe8(p, s);
    __builtin_riscv_th_msce32(p, s);
    __builtin_riscv_th_msme16(p);
}

// CHECK-LABEL: test_matmul:
// CHECK: th.mfmacc.h acc0, tr1, tr0
// CHECK: th.mfmacc.s acc0, tr1, tr0
// CHECK: th.mfmacc.d acc0, tr1, tr0
// CHECK: th.mmacc.w.b acc0, tr1, tr0
// CHECK: th.mmaccu.w.b acc0, tr1, tr0
// CHECK: th.mmaccu.d.h acc0, tr1, tr0
// CHECK: th.pmmacc.w.b acc0, tr1, tr0
// CHECK: th.mmaccu.w.bp acc0, tr1, tr0
void test_matmul(void) {
    __builtin_riscv_th_mfmacc_h();
    __builtin_riscv_th_mfmacc_s();
    __builtin_riscv_th_mfmacc_d();
    __builtin_riscv_th_mmacc_w_b();
    __builtin_riscv_th_mmaccu_w_b();
    __builtin_riscv_th_mmaccu_d_h();
    __builtin_riscv_th_pmmacc_w_b();
    __builtin_riscv_th_mmaccu_w_bp();
}

// CHECK-LABEL: test_misc:
// CHECK: th.mzero tr0
// CHECK: th.mzero2r tr0
// CHECK: th.mzero4r tr0
// CHECK: th.mzero8r tr0
// CHECK: th.mmov.mm tr0, tr1
void test_misc(void) {
    __builtin_riscv_th_mzero();
    __builtin_riscv_th_mzero2r();
    __builtin_riscv_th_mzero4r();
    __builtin_riscv_th_mzero8r();
    __builtin_riscv_th_mmov_mm();
}

// CHECK-LABEL: test_mov:
// CHECK: th.mmovw.x.m {{[a-z0-9]+}}, tr0
// CHECK: th.mmovw.m.x tr0
// CHECK: th.mdupw.m.x tr0
void test_mov(long val, long idx) {
    long r = __builtin_riscv_th_mmovw_x_m(idx);
    __builtin_riscv_th_mmovw_m_x(val + r, idx);
    __builtin_riscv_th_mdupw_m_x(val);
}

// CHECK-LABEL: test_pack:
// CHECK: th.mpack tr0, tr2, tr1
// CHECK: th.mpackhl tr0, tr2, tr1
// CHECK: th.mpackhh tr0, tr2, tr1
void test_pack(void) {
    __builtin_riscv_th_mpack();
    __builtin_riscv_th_mpackhl();
    __builtin_riscv_th_mpackhh();
}

// CHECK-LABEL: test_slide_bcast:
// CHECK: th.mrslidedown tr0, tr1, 3
// CHECK: th.mrslideup tr0, tr1, 2
// CHECK: th.mcslidedown.b tr0, tr1, 1
// CHECK: th.mcslideup.w tr0, tr1, 4
// CHECK: th.mrbca.mv.i tr0, tr1, 5
// CHECK: th.mcbcab.mv.i tr0, tr1, 1
void test_slide_bcast(void) {
    __builtin_riscv_th_mrslidedown(3);
    __builtin_riscv_th_mrslideup(2);
    __builtin_riscv_th_mcslidedown_b(1);
    __builtin_riscv_th_mcslideup_w(4);
    __builtin_riscv_th_mrbca_mv_i(5);
    __builtin_riscv_th_mcbcab_mv_i(1);
}

// CHECK-LABEL: test_conversions:
// CHECK: th.mfcvtl.h.e4 acc0, acc1
// CHECK: th.mfcvth.s.bf16 acc0, acc1
// CHECK: th.mfcvt.tf32.s acc0, acc1
// CHECK: th.msfcvtl.h.b acc0, acc1
// CHECK: th.mfucvtl.b.h acc0, acc1
// CHECK: th.msfcvt.s.w acc0, acc1
// CHECK: th.mfscvt.w.s acc0, acc1
// CHECK: th.mucvtl.b.p acc0, acc1
void test_conversions(void) {
    __builtin_riscv_th_mfcvtl_h_e4();
    __builtin_riscv_th_mfcvth_s_bf16();
    __builtin_riscv_th_mfcvt_tf32_s();
    __builtin_riscv_th_msfcvtl_h_b();
    __builtin_riscv_th_mfucvtl_b_h();
    __builtin_riscv_th_msfcvt_s_w();
    __builtin_riscv_th_mfscvt_w_s();
    __builtin_riscv_th_mucvtl_b_p();
}

// CHECK-LABEL: test_ew_arith:
// CHECK: th.madd.w.mm acc0, acc2, acc1
// CHECK: th.madd.w.mv.i acc0, acc2, acc1, 3
// CHECK: th.mfadd.h.mm acc0, acc2, acc1
// CHECK: th.mfadd.s.mv.i acc0, acc2, acc1, 2
// CHECK: th.mn4clipl.w.mm acc0, acc2, acc1
// CHECK: th.mn4clipl.w.mv.i acc0, acc2, acc1, 1
void test_ew_arith(void) {
    __builtin_riscv_th_madd_w_mm();
    __builtin_riscv_th_madd_w_mv_i(3);
    __builtin_riscv_th_mfadd_h_mm();
    __builtin_riscv_th_mfadd_s_mv_i(2);
    __builtin_riscv_th_mn4clipl_w_mm();
    __builtin_riscv_th_mn4clipl_w_mv_i(1);
}

// CHECK-LABEL: test_gemm_int8:
// CHECK: th.msettilem
// CHECK: th.msettilen
// CHECK: th.msettilek
// CHECK: th.mzero tr0
// CHECK: th.mlae8 tr0
// CHECK: th.mlbe8 tr1
// CHECK: th.mmacc.w.b acc0, tr1, tr0
// CHECK: th.msce32 acc0
// CHECK: th.mrelease
void test_gemm_int8(void *a, void *b, void *c,
                    long sa, long sb, long sc) {
    __builtin_riscv_th_msettilem(8);
    __builtin_riscv_th_msettilen(8);
    __builtin_riscv_th_msettilek(16);
    __builtin_riscv_th_mzero();
    __builtin_riscv_th_mlae8(a, sa);
    __builtin_riscv_th_mlbe8(b, sb);
    __builtin_riscv_th_mmacc_w_b();
    __builtin_riscv_th_msce32(c, sc);
    __builtin_riscv_th_mrelease();
}

// CHECK-LABEL: test_matmul_fp_variants:
// CHECK: th.mfmacc.h.e5 acc0, tr1, tr0
// CHECK: th.mfmacc.h.e4 acc0, tr1, tr0
// CHECK: th.mfmacc.bf16.e5 acc0, tr1, tr0
// CHECK: th.mfmacc.s.h acc0, tr1, tr0
// CHECK: th.mfmacc.s.bf16 acc0, tr1, tr0
// CHECK: th.mfmacc.s.tf32 acc0, tr1, tr0
// CHECK: th.mfmacc.d.s acc0, tr1, tr0
void test_matmul_fp_variants(void) {
    __builtin_riscv_th_mfmacc_h_e5();
    __builtin_riscv_th_mfmacc_h_e4();
    __builtin_riscv_th_mfmacc_bf16_e5();
    __builtin_riscv_th_mfmacc_s_h();
    __builtin_riscv_th_mfmacc_s_bf16();
    __builtin_riscv_th_mfmacc_s_tf32();
    __builtin_riscv_th_mfmacc_d_s();
}

// CHECK-LABEL: test_matmul_int_variants:
// CHECK: th.mmaccsu.w.b acc0, tr1, tr0
// CHECK: th.mmaccus.w.b acc0, tr1, tr0
// CHECK: th.mmacc.d.h acc0, tr1, tr0
// CHECK: th.mmaccsu.d.h acc0, tr1, tr0
// CHECK: th.pmmaccsu.w.b acc0, tr1, tr0
// CHECK: th.pmmaccu.w.b acc0, tr1, tr0
// CHECK: th.mmacc.w.bp acc0, tr1, tr0
void test_matmul_int_variants(void) {
    __builtin_riscv_th_mmaccsu_w_b();
    __builtin_riscv_th_mmaccus_w_b();
    __builtin_riscv_th_mmacc_d_h();
    __builtin_riscv_th_mmaccsu_d_h();
    __builtin_riscv_th_pmmaccsu_w_b();
    __builtin_riscv_th_pmmaccu_w_b();
    __builtin_riscv_th_mmacc_w_bp();
}

// CHECK-LABEL: test_ew_int_all_ops:
// CHECK: th.madd.w.mm acc0, acc2, acc1
// CHECK: th.msub.w.mm acc0, acc2, acc1
// CHECK: th.mmul.w.mm acc0, acc2, acc1
// CHECK: th.mmulh.w.mm acc0, acc2, acc1
// CHECK: th.mmax.w.mm acc0, acc2, acc1
// CHECK: th.mumax.w.mm acc0, acc2, acc1
// CHECK: th.mmin.w.mm acc0, acc2, acc1
// CHECK: th.mumin.w.mm acc0, acc2, acc1
// CHECK: th.msrl.w.mm acc0, acc2, acc1
// CHECK: th.msll.w.mm acc0, acc2, acc1
// CHECK: th.msra.w.mm acc0, acc2, acc1
void test_ew_int_all_ops(void) {
    __builtin_riscv_th_madd_w_mm();
    __builtin_riscv_th_msub_w_mm();
    __builtin_riscv_th_mmul_w_mm();
    __builtin_riscv_th_mmulh_w_mm();
    __builtin_riscv_th_mmax_w_mm();
    __builtin_riscv_th_mumax_w_mm();
    __builtin_riscv_th_mmin_w_mm();
    __builtin_riscv_th_mumin_w_mm();
    __builtin_riscv_th_msrl_w_mm();
    __builtin_riscv_th_msll_w_mm();
    __builtin_riscv_th_msra_w_mm();
}

// CHECK-LABEL: test_ew_fp_all_sizes:
// CHECK: th.mfadd.h.mm acc0, acc2, acc1
// CHECK: th.mfsub.s.mm acc0, acc2, acc1
// CHECK: th.mfmul.d.mm acc0, acc2, acc1
// CHECK: th.mfmax.h.mv.i acc0, acc2, acc1, 1
// CHECK: th.mfmin.s.mv.i acc0, acc2, acc1, 2
// CHECK: th.mfmul.d.mv.i acc0, acc2, acc1, 3
void test_ew_fp_all_sizes(void) {
    __builtin_riscv_th_mfadd_h_mm();
    __builtin_riscv_th_mfsub_s_mm();
    __builtin_riscv_th_mfmul_d_mm();
    __builtin_riscv_th_mfmax_h_mv_i(1);
    __builtin_riscv_th_mfmin_s_mv_i(2);
    __builtin_riscv_th_mfmul_d_mv_i(3);
}

// CHECK-LABEL: test_conversions_widening:
// CHECK: th.mfcvtl.h.e5 acc0, acc1
// CHECK: th.mfcvtl.s.h acc0, acc1
// CHECK: th.mfcvtl.d.s acc0, acc1
// CHECK: th.mfcvth.h.e4 acc0, acc1
// CHECK: th.mfcvth.s.bf16 acc0, acc1
void test_conversions_widening(void) {
    __builtin_riscv_th_mfcvtl_h_e5();
    __builtin_riscv_th_mfcvtl_s_h();
    __builtin_riscv_th_mfcvtl_d_s();
    __builtin_riscv_th_mfcvth_h_e4();
    __builtin_riscv_th_mfcvth_s_bf16();
}

// CHECK-LABEL: test_conversions_narrowing:
// CHECK: th.mfcvtl.e4.h acc0, acc1
// CHECK: th.mfcvtl.h.s acc0, acc1
// CHECK: th.mfcvtl.s.d acc0, acc1
// CHECK: th.mfcvth.e5.h acc0, acc1
// CHECK: th.mfcvth.bf16.s acc0, acc1
void test_conversions_narrowing(void) {
    __builtin_riscv_th_mfcvtl_e4_h();
    __builtin_riscv_th_mfcvtl_h_s();
    __builtin_riscv_th_mfcvtl_s_d();
    __builtin_riscv_th_mfcvth_e5_h();
    __builtin_riscv_th_mfcvth_bf16_s();
}

// CHECK-LABEL: test_n4clip_all:
// CHECK: th.mn4clipl.w.mm acc0, acc2, acc1
// CHECK: th.mn4cliph.w.mm acc0, acc2, acc1
// CHECK: th.mn4cliplu.w.mm acc0, acc2, acc1
// CHECK: th.mn4cliphu.w.mm acc0, acc2, acc1
// CHECK: th.mn4cliph.w.mv.i acc0, acc2, acc1, 2
// CHECK: th.mn4cliphu.w.mv.i acc0, acc2, acc1, 4
void test_n4clip_all(void) {
    __builtin_riscv_th_mn4clipl_w_mm();
    __builtin_riscv_th_mn4cliph_w_mm();
    __builtin_riscv_th_mn4cliplu_w_mm();
    __builtin_riscv_th_mn4cliphu_w_mm();
    __builtin_riscv_th_mn4cliph_w_mv_i(2);
    __builtin_riscv_th_mn4cliphu_w_mv_i(4);
}

// CHECK-LABEL: test_all_gpr_sizes:
// CHECK: th.mmovb.x.m {{[a-z0-9]+}}, tr0
// CHECK: th.mmovh.x.m {{[a-z0-9]+}}, tr0
// CHECK: th.mmovd.x.m {{[a-z0-9]+}}, tr0
// CHECK: th.mmovb.m.x tr0
// CHECK: th.mmovh.m.x tr0
// CHECK: th.mmovd.m.x tr0
// CHECK: th.mdupb.m.x tr0
// CHECK: th.mduph.m.x tr0
// CHECK: th.mdupd.m.x tr0
long test_all_gpr_sizes(long val, long idx) {
    long r1 = __builtin_riscv_th_mmovb_x_m(idx);
    long r2 = __builtin_riscv_th_mmovh_x_m(idx);
    long r3 = __builtin_riscv_th_mmovd_x_m(idx);
    __builtin_riscv_th_mmovb_m_x(val + r1, idx);
    __builtin_riscv_th_mmovh_m_x(val + r2, idx);
    __builtin_riscv_th_mmovd_m_x(val + r3, idx);
    __builtin_riscv_th_mdupb_m_x(val);
    __builtin_riscv_th_mduph_m_x(val);
    __builtin_riscv_th_mdupd_m_x(val);
    return r1 + r2 + r3;
}

// CHECK-LABEL: test_load_store_all_eew:
// CHECK: th.mlae8 tr0
// CHECK: th.mlae16 tr0
// CHECK: th.mlae32 tr0
// CHECK: th.mlae64 tr0
// CHECK: th.msae8 tr0
// CHECK: th.msae16 tr0
// CHECK: th.msae32 tr0
// CHECK: th.msae64 tr0
void test_load_store_all_eew(void *p, long s) {
    __builtin_riscv_th_mlae8(p, s);
    __builtin_riscv_th_mlae16(p, s);
    __builtin_riscv_th_mlae32(p, s);
    __builtin_riscv_th_mlae64(p, s);
    __builtin_riscv_th_msae8(p, s);
    __builtin_riscv_th_msae16(p, s);
    __builtin_riscv_th_msae32(p, s);
    __builtin_riscv_th_msae64(p, s);
}
