// NOTE: XTHeadMatrix higher-level intrinsic API test: <thead_matrix.h> -> assembly.
// Tests all API categories using the struct-based matrix types.
//
// RUN: %clang_cc1 -O2 -triple riscv64 -target-feature +experimental-xtheadmatrix -S -o - %s \
// RUN:   | FileCheck %s

#include <thead_matrix.h>

// === Configuration ===

// CHECK-LABEL: test_msetmrow_m:
// CHECK: th.msettilem
// CHECK: ret
mrow_t test_msetmrow_m(mrow_t m) {
  return __riscv_th_msetmrow_m(m);
}

// CHECK-LABEL: test_msetmrow_n:
// CHECK: th.msettilen
// CHECK: ret
mrow_t test_msetmrow_n(mrow_t n) {
  return __riscv_th_msetmrow_n(n);
}

// CHECK-LABEL: test_msetmcol_e32:
// CHECK: th.msettilek
// CHECK: ret
mcol_t test_msetmcol_e32(mcol_t c) {
  return __riscv_th_msetmcol_e32(c);
}

// CHECK-LABEL: test_mrelease:
// CHECK: th.mrelease
// CHECK: ret
void test_mrelease(void) {
  __riscv_th_mrelease();
}

// === Zero ===

// CHECK-LABEL: test_mzero_i32:
// CHECK: th.mzero
// CHECK: ret
void test_mzero_i32(void) {
  mint32_t z = __riscv_th_mzero_i32();
  (void)z;
}

// CHECK-LABEL: test_mzero_f32:
// CHECK: th.mzero
// CHECK: ret
void test_mzero_f32(void) {
  mfloat32_t z = __riscv_th_mzero_f32();
  (void)z;
}

// CHECK-LABEL: test_mzero_i32x2:
// CHECK: th.mzero2r
// CHECK: ret
void test_mzero_i32x2(void) {
  mint32x2_t z = __riscv_th_mzero_i32x2();
  (void)z;
}

// === Load A (element-stride) ===

// CHECK-LABEL: test_mld_a_i32:
// CHECK: th.msettilem
// CHECK: th.msettilek
// CHECK: th.mlae32
// CHECK: ret
void test_mld_a_i32(const void *base, long stride, mrow_t m, mcol_t k) {
  mint32_t a = __riscv_th_mld_a_i32(base, stride, m, k);
  (void)a;
}

// === Load B (element-stride) ===

// CHECK-LABEL: test_mld_b_f16:
// CHECK: th.msettilek
// CHECK: th.msettilen
// CHECK: th.mlbe16
// CHECK: ret
void test_mld_b_f16(const void *base, long stride, mrow_t k, mcol_t n) {
  mfloat16_t b = __riscv_th_mld_b_f16(base, stride, k, n);
  (void)b;
}

// === Load C (element-stride) ===

// CHECK-LABEL: test_mld_c_i64:
// CHECK: th.msettilem
// CHECK: th.msettilen
// CHECK: th.mlce64
// CHECK: ret
void test_mld_c_i64(const void *base, long stride, mrow_t m, mcol_t n) {
  mint64_t c = __riscv_th_mld_c_i64(base, stride, m, n);
  (void)c;
}

// === Transposed Load A ===

// CHECK-LABEL: test_mld_at_i8:
// CHECK: th.msettilem
// CHECK: th.msettilek
// CHECK: th.mlate8
// CHECK: ret
void test_mld_at_i8(const void *base, long stride, mrow_t m, mcol_t k) {
  mint8_t a = __riscv_th_mld_at_i8(base, stride, m, k);
  (void)a;
}

// === Transposed Load B ===

// CHECK-LABEL: test_mld_bt_u16:
// CHECK: th.msettilek
// CHECK: th.msettilen
// CHECK: th.mlbte16
// CHECK: ret
void test_mld_bt_u16(const void *base, long stride, mrow_t k, mcol_t n) {
  muint16_t b = __riscv_th_mld_bt_u16(base, stride, k, n);
  (void)b;
}

// === Transposed Load C ===

// CHECK-LABEL: test_mld_ct_f32:
// CHECK: th.msettilem
// CHECK: th.msettilen
// CHECK: th.mlcte32
// CHECK: ret
void test_mld_ct_f32(const void *base, long stride, mrow_t m, mcol_t n) {
  mfloat32_t c = __riscv_th_mld_ct_f32(base, stride, m, n);
  (void)c;
}

// === Whole-register Load ===

// CHECK-LABEL: test_mld_whole_f64:
// CHECK: th.mlme64
// CHECK: ret
void test_mld_whole_f64(const void *base) {
  mfloat64_t w = __riscv_th_mld_whole_f64(base);
  (void)w;
}

// === Store C (element-stride) ===

// CHECK-LABEL: test_mst_c_i32:
// CHECK: th.msettilem
// CHECK: th.msettilen
// CHECK: th.msce32
// CHECK: ret
void test_mst_c_i32(void *base, long stride, mrow_t m, mcol_t n) {
  mint32_t val = __riscv_th_mundefined_i32();
  __riscv_th_mst_c_i32(base, stride, val, m, n);
}

// === Store A (element-stride) ===

// CHECK-LABEL: test_mst_a_u8:
// CHECK: th.msettilem
// CHECK: th.msettilek
// CHECK: th.msae8
// CHECK: ret
void test_mst_a_u8(void *base, long stride, mrow_t m, mcol_t k) {
  muint8_t val = __riscv_th_mundefined_u8();
  __riscv_th_mst_a_u8(base, stride, val, m, k);
}

// === Store B (transposed) ===

// CHECK-LABEL: test_mst_bt_f32:
// CHECK: th.msettilek
// CHECK: th.msettilen
// CHECK: th.msbte32
// CHECK: ret
void test_mst_bt_f32(void *base, long stride, mrow_t k, mcol_t n) {
  mfloat32_t val = __riscv_th_mundefined_f32();
  __riscv_th_mst_bt_f32(base, stride, val, k, n);
}

// === Whole-register Store ===

// CHECK-LABEL: test_mst_whole_i16:
// CHECK: th.msme16
// CHECK: ret
void test_mst_whole_i16(void *base) {
  mint16_t val = __riscv_th_mundefined_i16();
  __riscv_th_mst_whole_i16(base, val);
}

// === FP Matmul ===

// CHECK-LABEL: test_fmmacc_s:
// CHECK: th.msettilem
// CHECK: th.msettilek
// CHECK: th.msettilen
// CHECK: th.mfmacc.s
// CHECK: ret
void test_fmmacc_s(mrow_t m, mrow_t k, mcol_t n) {
  mfloat32_t c = __riscv_th_mundefined_f32();
  mfloat32_t a = __riscv_th_mundefined_f32();
  mfloat32_t b = __riscv_th_mundefined_f32();
  mfloat32_t r = __riscv_th_fmmacc_s(c, a, b, m, k, n);
  (void)r;
}

// CHECK-LABEL: test_fmmacc_h:
// CHECK: th.mfmacc.h
// CHECK: ret
void test_fmmacc_h(mrow_t m, mrow_t k, mcol_t n) {
  mfloat16_t c = __riscv_th_mundefined_f16();
  mfloat16_t a = __riscv_th_mundefined_f16();
  mfloat16_t b = __riscv_th_mundefined_f16();
  mfloat16_t r = __riscv_th_fmmacc_h(c, a, b, m, k, n);
  (void)r;
}

// CHECK-LABEL: test_fwmmacc_s_h:
// CHECK: th.mfmacc.s.h
// CHECK: ret
void test_fwmmacc_s_h(mrow_t m, mrow_t k, mcol_t n) {
  mfloat32_t c = __riscv_th_mundefined_f32();
  mfloat16_t a = __riscv_th_mundefined_f16();
  mfloat16_t b = __riscv_th_mundefined_f16();
  mfloat32_t r = __riscv_th_fwmmacc_s_h(c, a, b, m, k, n);
  (void)r;
}

// CHECK-LABEL: test_fwmmacc_d_s:
// CHECK: th.mfmacc.d.s
// CHECK: ret
void test_fwmmacc_d_s(mrow_t m, mrow_t k, mcol_t n) {
  mfloat64_t c = __riscv_th_mundefined_f64();
  mfloat32_t a = __riscv_th_mundefined_f32();
  mfloat32_t b = __riscv_th_mundefined_f32();
  mfloat64_t r = __riscv_th_fwmmacc_d_s(c, a, b, m, k, n);
  (void)r;
}

// === Integer Matmul ===

// CHECK-LABEL: test_mmaqa_ss_w_b:
// CHECK: th.mmacc.w.b
// CHECK: ret
void test_mmaqa_ss_w_b(mrow_t m, mrow_t k, mcol_t n) {
  mint32_t c = __riscv_th_mundefined_i32();
  mint8_t a = __riscv_th_mundefined_i8();
  mint8_t b = __riscv_th_mundefined_i8();
  mint32_t r = __riscv_th_mmaqa_ss_w_b(c, a, b, m, k, n);
  (void)r;
}

// CHECK-LABEL: test_mmaqa_uu_w_b:
// CHECK: th.mmaccu.w.b
// CHECK: ret
void test_mmaqa_uu_w_b(mrow_t m, mrow_t k, mcol_t n) {
  muint32_t c = __riscv_th_mundefined_u32();
  muint8_t a = __riscv_th_mundefined_u8();
  muint8_t b = __riscv_th_mundefined_u8();
  muint32_t r = __riscv_th_mmaqa_uu_w_b(c, a, b, m, k, n);
  (void)r;
}

// CHECK-LABEL: test_mmaqa_us_w_b:
// CHECK: th.mmaccus.w.b
// CHECK: ret
void test_mmaqa_us_w_b(mrow_t m, mrow_t k, mcol_t n) {
  mint32_t c = __riscv_th_mundefined_i32();
  muint8_t a = __riscv_th_mundefined_u8();
  mint8_t b = __riscv_th_mundefined_i8();
  mint32_t r = __riscv_th_mmaqa_us_w_b(c, a, b, m, k, n);
  (void)r;
}

// CHECK-LABEL: test_mmaqa_ss_d_h:
// CHECK: th.mmacc.d.h
// CHECK: ret
void test_mmaqa_ss_d_h(mrow_t m, mrow_t k, mcol_t n) {
  mint64_t c = __riscv_th_mundefined_i64();
  mint16_t a = __riscv_th_mundefined_i16();
  mint16_t b = __riscv_th_mundefined_i16();
  mint64_t r = __riscv_th_mmaqa_ss_d_h(c, a, b, m, k, n);
  (void)r;
}

// === Partial Matmul ===

// CHECK-LABEL: test_pmmaqa_ss_w_b:
// CHECK: th.pmmacc.w.b
// CHECK: ret
void test_pmmaqa_ss_w_b(mrow_t m, mrow_t k, mcol_t n) {
  mint32_t c = __riscv_th_mundefined_i32();
  mint8_t a = __riscv_th_mundefined_i8();
  mint8_t b = __riscv_th_mundefined_i8();
  mint32_t r = __riscv_th_pmmaqa_ss_w_b(c, a, b, m, k, n);
  (void)r;
}

// === Bypass Matmul ===

// CHECK-LABEL: test_mmaqa_bp_ss:
// CHECK: th.mmacc.w.bp
// CHECK: ret
void test_mmaqa_bp_ss(mrow_t m, mrow_t k, mcol_t n) {
  mint32_t c = __riscv_th_mundefined_i32();
  mint8_t a = __riscv_th_mundefined_i8();
  mint8_t b = __riscv_th_mundefined_i8();
  mint32_t r = __riscv_th_mmaqa_bp_ss(c, a, b, m, k, n);
  (void)r;
}

// === Integer EW Arithmetic ===

// CHECK-LABEL: test_madd_w_mm:
// CHECK: th.madd.w.mm
// CHECK: ret
void test_madd_w_mm(void) {
  mint32_t d = __riscv_th_mundefined_i32();
  mint32_t s1 = __riscv_th_mundefined_i32();
  mint32_t s2 = __riscv_th_mundefined_i32();
  mint32_t r = __riscv_th_madd_w_mm(d, s1, s2);
  (void)r;
}

// CHECK-LABEL: test_madd_w_mv:
// CHECK: th.madd.w.mv.i
// CHECK: ret
void test_madd_w_mv(void) {
  mint32_t d = __riscv_th_mundefined_i32();
  mint32_t s1 = __riscv_th_mundefined_i32();
  mint32_t r = __riscv_th_madd_w_mv(d, s1, 3);
  (void)r;
}

// CHECK-LABEL: test_msub_w_mm:
// CHECK: th.msub.w.mm
// CHECK: ret
void test_msub_w_mm(void) {
  mint32_t d = __riscv_th_mundefined_i32();
  mint32_t s1 = __riscv_th_mundefined_i32();
  mint32_t s2 = __riscv_th_mundefined_i32();
  mint32_t r = __riscv_th_msub_w_mm(d, s1, s2);
  (void)r;
}

// CHECK-LABEL: test_mmul_w_mm:
// CHECK: th.mmul.w.mm
// CHECK: ret
void test_mmul_w_mm(void) {
  mint32_t d = __riscv_th_mundefined_i32();
  mint32_t s1 = __riscv_th_mundefined_i32();
  mint32_t s2 = __riscv_th_mundefined_i32();
  mint32_t r = __riscv_th_mmul_w_mm(d, s1, s2);
  (void)r;
}

// CHECK-LABEL: test_msrl_w_mv:
// CHECK: th.msrl.w.mv.i
// CHECK: ret
void test_msrl_w_mv(void) {
  mint32_t d = __riscv_th_mundefined_i32();
  mint32_t s1 = __riscv_th_mundefined_i32();
  mint32_t r = __riscv_th_msrl_w_mv(d, s1, 2);
  (void)r;
}

// CHECK-LABEL: test_mmax_w_mm:
// CHECK: th.mmax.w.mm
// CHECK: ret
void test_mmax_w_mm(void) {
  mint32_t d = __riscv_th_mundefined_i32();
  mint32_t s1 = __riscv_th_mundefined_i32();
  mint32_t s2 = __riscv_th_mundefined_i32();
  mint32_t r = __riscv_th_mmax_w_mm(d, s1, s2);
  (void)r;
}

// CHECK-LABEL: test_mmin_w_mm:
// CHECK: th.mmin.w.mm
// CHECK: ret
void test_mmin_w_mm(void) {
  mint32_t d = __riscv_th_mundefined_i32();
  mint32_t s1 = __riscv_th_mundefined_i32();
  mint32_t s2 = __riscv_th_mundefined_i32();
  mint32_t r = __riscv_th_mmin_w_mm(d, s1, s2);
  (void)r;
}

// CHECK-LABEL: test_msra_w_mv:
// CHECK: th.msra.w.mv.i
// CHECK: ret
void test_msra_w_mv(void) {
  mint32_t d = __riscv_th_mundefined_i32();
  mint32_t s1 = __riscv_th_mundefined_i32();
  mint32_t r = __riscv_th_msra_w_mv(d, s1, 1);
  (void)r;
}

// === FP EW Arithmetic ===

// CHECK-LABEL: test_mfadd_s_mm:
// CHECK: th.mfadd.s.mm
// CHECK: ret
void test_mfadd_s_mm(void) {
  mfloat32_t d = __riscv_th_mundefined_f32();
  mfloat32_t s1 = __riscv_th_mundefined_f32();
  mfloat32_t s2 = __riscv_th_mundefined_f32();
  mfloat32_t r = __riscv_th_mfadd_s_mm(d, s1, s2);
  (void)r;
}

// CHECK-LABEL: test_mfadd_h_mv:
// CHECK: th.mfadd.h.mv.i
// CHECK: ret
void test_mfadd_h_mv(void) {
  mfloat16_t d = __riscv_th_mundefined_f16();
  mfloat16_t s1 = __riscv_th_mundefined_f16();
  mfloat16_t r = __riscv_th_mfadd_h_mv(d, s1, 1);
  (void)r;
}

// CHECK-LABEL: test_mfmul_d_mm:
// CHECK: th.mfmul.d.mm
// CHECK: ret
void test_mfmul_d_mm(void) {
  mfloat64_t d = __riscv_th_mundefined_f64();
  mfloat64_t s1 = __riscv_th_mundefined_f64();
  mfloat64_t s2 = __riscv_th_mundefined_f64();
  mfloat64_t r = __riscv_th_mfmul_d_mm(d, s1, s2);
  (void)r;
}

// CHECK-LABEL: test_mfmax_s_mm:
// CHECK: th.mfmax.s.mm
// CHECK: ret
void test_mfmax_s_mm(void) {
  mfloat32_t d = __riscv_th_mundefined_f32();
  mfloat32_t s1 = __riscv_th_mundefined_f32();
  mfloat32_t s2 = __riscv_th_mundefined_f32();
  mfloat32_t r = __riscv_th_mfmax_s_mm(d, s1, s2);
  (void)r;
}

// CHECK-LABEL: test_mfmin_h_mv:
// CHECK: th.mfmin.h.mv.i
// CHECK: ret
void test_mfmin_h_mv(void) {
  mfloat16_t d = __riscv_th_mundefined_f16();
  mfloat16_t s1 = __riscv_th_mundefined_f16();
  mfloat16_t r = __riscv_th_mfmin_h_mv(d, s1, 0);
  (void)r;
}

// CHECK-LABEL: test_mfsub_d_mv:
// CHECK: th.mfsub.d.mv.i
// CHECK: ret
void test_mfsub_d_mv(void) {
  mfloat64_t d = __riscv_th_mundefined_f64();
  mfloat64_t s1 = __riscv_th_mundefined_f64();
  mfloat64_t r = __riscv_th_mfsub_d_mv(d, s1, 2);
  (void)r;
}

// === FP Conversions ===

// CHECK-LABEL: test_mfcvtl_s_h:
// CHECK: th.mfcvtl.s.h
// CHECK: ret
void test_mfcvtl_s_h(void) {
  mfloat16_t src = __riscv_th_mundefined_f16();
  mfloat32_t r = __riscv_th_mfcvtl_s_h(src);
  (void)r;
}

// CHECK-LABEL: test_mfcvth_h_s:
// CHECK: th.mfcvth.h.s
// CHECK: ret
void test_mfcvth_h_s(void) {
  mfloat32_t src = __riscv_th_mundefined_f32();
  mfloat16_t r = __riscv_th_mfcvth_h_s(src);
  (void)r;
}

// CHECK-LABEL: test_mfcvt_tf32_s:
// CHECK: th.mfcvt.tf32.s
// CHECK: ret
void test_mfcvt_tf32_s(void) {
  mfloat32_t src = __riscv_th_mundefined_f32();
  mfloat32_t r = __riscv_th_mfcvt_tf32_s(src);
  (void)r;
}

// CHECK-LABEL: test_mfcvtl_d_s:
// CHECK: th.mfcvtl.d.s
// CHECK: ret
void test_mfcvtl_d_s(void) {
  mfloat32_t src = __riscv_th_mundefined_f32();
  mfloat64_t r = __riscv_th_mfcvtl_d_s(src);
  (void)r;
}

// CHECK-LABEL: test_mfcvtl_h_e4:
// CHECK: th.mfcvtl.h.e4
// CHECK: ret
void test_mfcvtl_h_e4(void) {
  muint8_t src = __riscv_th_mundefined_u8();
  mfloat16_t r = __riscv_th_mfcvtl_h_e4(src);
  (void)r;
}

// === Float-Int Conversions ===

// CHECK-LABEL: test_msfcvt_s_w:
// CHECK: th.msfcvt.s.w
// CHECK: ret
void test_msfcvt_s_w(void) {
  mint32_t src = __riscv_th_mundefined_i32();
  mfloat32_t r = __riscv_th_msfcvt_s_w(src);
  (void)r;
}

// CHECK-LABEL: test_mfucvt_w_s:
// CHECK: th.mfucvt.w.s
// CHECK: ret
void test_mfucvt_w_s(void) {
  mfloat32_t src = __riscv_th_mundefined_f32();
  muint32_t r = __riscv_th_mfucvt_w_s(src);
  (void)r;
}

// === N4Clip ===

// CHECK-LABEL: test_mn4clipl_w_mm:
// CHECK: th.mn4clipl.w.mm
// CHECK: ret
void test_mn4clipl_w_mm(void) {
  mint32_t s1 = __riscv_th_mundefined_i32();
  mint32_t s2 = __riscv_th_mundefined_i32();
  mint8_t r = __riscv_th_mn4clipl_w_mm(s1, s2);  // signed clip -> mint8_t
  (void)r;
}

// CHECK-LABEL: test_mn4cliph_w_mv:
// CHECK: th.mn4cliph.w.mv.i
// CHECK: ret
void test_mn4cliph_w_mv(void) {
  mint32_t s1 = __riscv_th_mundefined_i32();
  mint8_t r = __riscv_th_mn4cliph_w_mv(s1, 5);  // signed clip -> mint8_t
  (void)r;
}

// === Packed Conversions ===

// CHECK-LABEL: test_mucvtl_b_p:
// CHECK: th.mucvtl.b.p
// CHECK: ret
void test_mucvtl_b_p(void) {
  muint8_t src = __riscv_th_mundefined_u8();
  muint8_t r = __riscv_th_mucvtl_b_p(src);
  (void)r;
}

// === Move / Duplicate ===

// CHECK-LABEL: test_mmov_mm:
// CHECK: th.mmov.mm
// CHECK: ret
void test_mmov_mm(void) {
  __riscv_th_mmov_mm();
}

// CHECK-LABEL: test_mmov_x_m_w:
// CHECK: th.mmovw.x.m
// CHECK: ret
size_t test_mmov_x_m_w(size_t idx) {
  mint32_t src = __riscv_th_mundefined_i32();
  return __riscv_th_mmov_x_m_w(src, idx);
}

// CHECK-LABEL: test_mmov_m_x_w:
// CHECK: th.mmovw.m.x
// CHECK: ret
void test_mmov_m_x_w(size_t data, size_t idx) {
  mint32_t d = __riscv_th_mundefined_i32();
  mint32_t r = __riscv_th_mmov_m_x_w(d, data, idx);
  (void)r;
}

// CHECK-LABEL: test_mdup_m_x_w:
// CHECK: th.mdupw.m.x
// CHECK: ret
void test_mdup_m_x_w(size_t data) {
  mint32_t r = __riscv_th_mdup_m_x_w(data);
  (void)r;
}

// === Pack ===

// CHECK-LABEL: test_mpack:
// CHECK: th.mpack
// CHECK: ret
void test_mpack(void) {
  __riscv_th_mpack();
}

// CHECK-LABEL: test_mpackhl:
// CHECK: th.mpackhl
// CHECK: ret
void test_mpackhl(void) {
  __riscv_th_mpackhl();
}

// CHECK-LABEL: test_mpackhh:
// CHECK: th.mpackhh
// CHECK: ret
void test_mpackhh(void) {
  __riscv_th_mpackhh();
}

// === Slide ===

// CHECK-LABEL: test_mrslidedown:
// CHECK: th.mrslidedown
// CHECK: ret
void test_mrslidedown(void) {
  __riscv_th_mrslidedown(2);
}

// CHECK-LABEL: test_mrslideup:
// CHECK: th.mrslideup
// CHECK: ret
void test_mrslideup(void) {
  __riscv_th_mrslideup(1);
}

// CHECK-LABEL: test_mcslideup_w:
// CHECK: th.mcslideup.w
// CHECK: ret
void test_mcslideup_w(void) {
  __riscv_th_mcslideup_w(1);
}

// CHECK-LABEL: test_mcslidedown_b:
// CHECK: th.mcslidedown.b
// CHECK: ret
void test_mcslidedown_b(void) {
  __riscv_th_mcslidedown_b(3);
}

// === Broadcast ===

// CHECK-LABEL: test_mrbca:
// CHECK: th.mrbca.mv.i
// CHECK: ret
void test_mrbca(void) {
  __riscv_th_mrbca(3);
}

// CHECK-LABEL: test_mcbca_w:
// CHECK: th.mcbcaw.mv.i
// CHECK: ret
void test_mcbca_w(void) {
  __riscv_th_mcbca_w(0);
}

// CHECK-LABEL: test_mcbca_b:
// CHECK: th.mcbcab.mv.i
// CHECK: ret
void test_mcbca_b(void) {
  __riscv_th_mcbca_b(2);
}

// === CSR Access ===

// CHECK-LABEL: test_xmlenb:
// CHECK: csrr {{.*}}, th.xtlenb
// CHECK: ret
unsigned long test_xmlenb(void) {
  return __riscv_th_xmlenb();
}

// CHECK-LABEL: test_xrlenb:
// CHECK: csrr {{.*}}, th.xtrlenb
// CHECK: ret
unsigned long test_xrlenb(void) {
  return __riscv_th_xrlenb();
}

// === End-to-end: INT8 GEMM using the API ===

// CHECK-LABEL: test_int8_gemm:
// CHECK-DAG: th.msettilem
// CHECK-DAG: th.msettilek
// CHECK-DAG: th.mlae8
// CHECK-DAG: th.msettilen
// CHECK-DAG: th.mlbe8
// CHECK: th.mzero
// CHECK: th.mmacc.w.b
// CHECK: th.msce32
// CHECK: th.mrelease
// CHECK: ret
void test_int8_gemm(const int8_t *A, long a_stride,
                     const int8_t *B, long b_stride,
                     int32_t *C, long c_stride,
                     mrow_t M, mrow_t K, mcol_t N) {
  // Load A and B tiles
  mint8_t a = __riscv_th_mld_a_i8(A, a_stride, M, K);
  mint8_t b = __riscv_th_mld_b_i8(B, b_stride, K, N);

  // Zero accumulator
  mint32_t c = __riscv_th_mzero_i32();

  // Matmul: c += a * b
  c = __riscv_th_mmaqa_ss_w_b(c, a, b, M, K, N);

  // Store result
  __riscv_th_mst_c_i32(C, c_stride, c, M, N);

  // Release
  __riscv_th_mrelease();
}

// === End-to-end: FP32 Matmul using the API ===

// CHECK-LABEL: test_fp32_gemm:
// CHECK-DAG: th.msettilem
// CHECK-DAG: th.msettilek
// CHECK-DAG: th.mlae32
// CHECK-DAG: th.msettilen
// CHECK-DAG: th.mlbe32
// CHECK: th.mzero
// CHECK: th.mfmacc.s
// CHECK: th.msce32
// CHECK: th.mrelease
// CHECK: ret
void test_fp32_gemm(const float *A, long a_stride,
                     const float *B, long b_stride,
                     float *C, long c_stride,
                     mrow_t M, mrow_t K, mcol_t N) {
  mfloat32_t a = __riscv_th_mld_a_f32(A, a_stride, M, K);
  mfloat32_t b = __riscv_th_mld_b_f32(B, b_stride, K, N);
  mfloat32_t c = __riscv_th_mzero_f32();
  c = __riscv_th_fmmacc_s(c, a, b, M, K, N);
  __riscv_th_mst_c_f32(C, c_stride, c, M, N);
  __riscv_th_mrelease();
}
