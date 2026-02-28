// NOTE: XTHeadMatrix end-to-end CodeGen test: C builtins -> RISC-V assembly.
// Tests a representative sample from all instruction categories.
// Phase C: typed builtins use __rvm_*_t types; untyped builtins stay void.
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
    // Typed loads: return value is __rvm_*_t (discarded here)
    (void)__builtin_riscv_th_mlae8(0, p, s);
    (void)__builtin_riscv_th_mlbe8(1, p, s);
    (void)__builtin_riscv_th_mlce32(4, p, s);
    (void)__builtin_riscv_th_mlate16(2, p, s);
    (void)__builtin_riscv_th_mlbte32(3, p, s);
    (void)__builtin_riscv_th_mlcte64(5, p, s);
    (void)__builtin_riscv_th_mlme8(0, p);
    // Stores remain void
    __builtin_riscv_th_msae8(0, p, s);
    __builtin_riscv_th_msbe8(1, p, s);
    __builtin_riscv_th_msce32(4, p, s);
    __builtin_riscv_th_msme16(0, p);
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
    // Typed matmul builtins need matrix args (use mundef for dummy args)
    __rvm_float16_t f16 = __builtin_riscv_th_mundef_f16();
    __rvm_float32_t f32 = __builtin_riscv_th_mundef_f32();
    __rvm_float64_t f64 = __builtin_riscv_th_mundef_f64();
    __rvm_int8_t i8 = __builtin_riscv_th_mundef_i8();
    __rvm_int16_t i16 = __builtin_riscv_th_mundef_i16();
    __rvm_int32_t i32 = __builtin_riscv_th_mundef_i32();
    __rvm_int64_t i64 = __builtin_riscv_th_mundef_i64();
    __rvm_uint8_t u8 = __builtin_riscv_th_mundef_u8();
    __rvm_uint16_t u16 = __builtin_riscv_th_mundef_u16();
    __rvm_uint32_t u32 = __builtin_riscv_th_mundef_u32();
    __rvm_uint64_t u64 = __builtin_riscv_th_mundef_u64();
    (void)__builtin_riscv_th_mfmacc_h(4, 1, 0, f16, f16, f16);
    (void)__builtin_riscv_th_mfmacc_s(4, 1, 0, f32, f32, f32);
    (void)__builtin_riscv_th_mfmacc_d(4, 1, 0, f64, f64, f64);
    (void)__builtin_riscv_th_mmacc_w_b(4, 1, 0, i32, i8, i8);
    (void)__builtin_riscv_th_mmaccu_w_b(4, 1, 0, u32, u8, u8);
    (void)__builtin_riscv_th_mmaccu_d_h(4, 1, 0, u64, u16, u16);
    (void)__builtin_riscv_th_pmmacc_w_b(4, 1, 0, i32, i8, i8);
    (void)__builtin_riscv_th_mmaccu_w_bp(4, 1, 0, u32, u8, u8);
}

// CHECK-LABEL: test_misc:
// CHECK: th.mzero acc0
// CHECK: th.mzero2r acc0
// CHECK: th.mzero4r acc0
// CHECK: th.mzero8r acc0
// CHECK: th.mmov.mm tr0, tr1
void test_misc(void) {
    __builtin_riscv_th_mzero(4);
    __builtin_riscv_th_mzero2r(4);
    __builtin_riscv_th_mzero4r(4);
    __builtin_riscv_th_mzero8r(4);
    __builtin_riscv_th_mmov_mm(0, 1);
}

// CHECK-LABEL: test_mov:
// CHECK: th.mmovw.x.m {{[a-z0-9]+}}, tr0
// CHECK: th.mmovw.m.x tr0
// CHECK: th.mdupw.m.x tr0
void test_mov(long val, long idx) {
    long r = __builtin_riscv_th_mmovw_x_m(0, idx);
    __builtin_riscv_th_mmovw_m_x(0, val + r, idx);
    __builtin_riscv_th_mdupw_m_x(0, val);
}

// CHECK-LABEL: test_pack:
// CHECK: th.mpack tr0, tr2, tr1
// CHECK: th.mpackhl tr0, tr2, tr1
// CHECK: th.mpackhh tr0, tr2, tr1
void test_pack(void) {
    __builtin_riscv_th_mpack(0, 2, 1);
    __builtin_riscv_th_mpackhl(0, 2, 1);
    __builtin_riscv_th_mpackhh(0, 2, 1);
}

// CHECK-LABEL: test_slide_bcast:
// CHECK: th.mrslidedown tr0, tr1, 3
// CHECK: th.mrslideup tr0, tr1, 2
// CHECK: th.mcslidedown.b tr0, tr1, 1
// CHECK: th.mcslideup.w tr0, tr1, 4
// CHECK: th.mrbca.mv.i tr0, tr1, 5
// CHECK: th.mcbcab.mv.i tr0, tr1, 1
void test_slide_bcast(void) {
    __builtin_riscv_th_mrslidedown(0, 1, 3);
    __builtin_riscv_th_mrslideup(0, 1, 2);
    __builtin_riscv_th_mcslidedown_b(0, 1, 1);
    __builtin_riscv_th_mcslideup_w(0, 1, 4);
    __builtin_riscv_th_mrbca_mv_i(0, 1, 5);
    __builtin_riscv_th_mcbcab_mv_i(0, 1, 1);
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
    __rvm_int8_t i8 = __builtin_riscv_th_mundef_i8();
    __rvm_int32_t i32 = __builtin_riscv_th_mundef_i32();
    __rvm_float16_t f16 = __builtin_riscv_th_mundef_f16();
    __rvm_float32_t f32 = __builtin_riscv_th_mundef_f32();
    // Untyped conversions (void builtins)
    __builtin_riscv_th_mfcvtl_h_e4(4, 5);
    __builtin_riscv_th_mfcvth_s_bf16(4, 5);
    __builtin_riscv_th_mfcvt_tf32_s(4, 5);
    // Typed float-int conversions
    (void)__builtin_riscv_th_msfcvtl_h_b(4, 5, i8);
    (void)__builtin_riscv_th_mfucvtl_b_h(4, 5, f16);
    (void)__builtin_riscv_th_msfcvt_s_w(4, 5, i32);
    (void)__builtin_riscv_th_mfscvt_w_s(4, 5, f32);
    // Untyped packed conversion
    __builtin_riscv_th_mucvtl_b_p(4, 5);
}

// CHECK-LABEL: test_ew_arith:
// CHECK: th.madd.w.mm acc0, acc2, acc1
// CHECK: th.madd.w.mv.i acc0, acc2, acc1, 3
// CHECK: th.mfadd.h.mm acc0, acc2, acc1
// CHECK: th.mfadd.s.mv.i acc0, acc2, acc1, 2
// CHECK: th.mn4clipl.w.mm acc0, acc2, acc1
// CHECK: th.mn4clipl.w.mv.i acc0, acc2, acc1, 1
void test_ew_arith(void) {
    __rvm_int32_t i32 = __builtin_riscv_th_mundef_i32();
    __rvm_float16_t f16 = __builtin_riscv_th_mundef_f16();
    __rvm_float32_t f32 = __builtin_riscv_th_mundef_f32();
    // Typed INT EW MM
    (void)__builtin_riscv_th_madd_w_mm(4, 6, 5, i32, i32, i32);
    // Typed INT EW MVI
    (void)__builtin_riscv_th_madd_w_mv_i(4, 6, 5, i32, i32, 3);
    // Typed FP EW MM
    (void)__builtin_riscv_th_mfadd_h_mm(4, 6, 5, f16, f16, f16);
    // Typed FP EW MVI
    (void)__builtin_riscv_th_mfadd_s_mv_i(4, 6, 5, f32, f32, 2);
    // N4clip stays void
    __builtin_riscv_th_mn4clipl_w_mm(4, 6, 5);
    __builtin_riscv_th_mn4clipl_w_mv_i(4, 6, 5, 1);
}

// CHECK-LABEL: test_gemm_int8:
// CHECK: th.msettilem
// CHECK: th.msettilen
// CHECK: th.msettilek
// CHECK: th.mzero acc0
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
    __builtin_riscv_th_mzero(4);
    (void)__builtin_riscv_th_mlae8(0, a, sa);
    (void)__builtin_riscv_th_mlbe8(1, b, sb);
    __rvm_int32_t acc = __builtin_riscv_th_mundef_i32();
    __rvm_int8_t src = __builtin_riscv_th_mundef_i8();
    (void)__builtin_riscv_th_mmacc_w_b(4, 1, 0, acc, src, src);
    __builtin_riscv_th_msce32(4, c, sc);
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
    __rvm_float16_t f16 = __builtin_riscv_th_mundef_f16();
    __rvm_float32_t f32 = __builtin_riscv_th_mundef_f32();
    __rvm_float64_t f64 = __builtin_riscv_th_mundef_f64();
    // Untyped FP8/BF16/TF32 matmul (void builtins)
    __builtin_riscv_th_mfmacc_h_e5(4, 1, 0);
    __builtin_riscv_th_mfmacc_h_e4(4, 1, 0);
    __builtin_riscv_th_mfmacc_bf16_e5(4, 1, 0);
    // Typed widening matmul
    (void)__builtin_riscv_th_mfmacc_s_h(4, 1, 0, f32, f16, f16);
    // Untyped bf16/tf32 widening
    __builtin_riscv_th_mfmacc_s_bf16(4, 1, 0);
    __builtin_riscv_th_mfmacc_s_tf32(4, 1, 0);
    // Typed d.s widening
    (void)__builtin_riscv_th_mfmacc_d_s(4, 1, 0, f64, f32, f32);
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
    __rvm_int8_t i8 = __builtin_riscv_th_mundef_i8();
    __rvm_int16_t i16 = __builtin_riscv_th_mundef_i16();
    __rvm_int32_t i32 = __builtin_riscv_th_mundef_i32();
    __rvm_int64_t i64 = __builtin_riscv_th_mundef_i64();
    __rvm_uint8_t u8 = __builtin_riscv_th_mundef_u8();
    __rvm_uint16_t u16 = __builtin_riscv_th_mundef_u16();
    __rvm_uint32_t u32 = __builtin_riscv_th_mundef_u32();
    (void)__builtin_riscv_th_mmaccsu_w_b(4, 1, 0, i32, i8, u8);
    (void)__builtin_riscv_th_mmaccus_w_b(4, 1, 0, i32, u8, i8);
    (void)__builtin_riscv_th_mmacc_d_h(4, 1, 0, i64, i16, i16);
    (void)__builtin_riscv_th_mmaccsu_d_h(4, 1, 0, i64, i16, u16);
    (void)__builtin_riscv_th_pmmaccsu_w_b(4, 1, 0, i32, i8, u8);
    (void)__builtin_riscv_th_pmmaccu_w_b(4, 1, 0, u32, u8, u8);
    (void)__builtin_riscv_th_mmacc_w_bp(4, 1, 0, i32, i8, i8);
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
    __rvm_int32_t i32 = __builtin_riscv_th_mundef_i32();
    (void)__builtin_riscv_th_madd_w_mm(4, 6, 5, i32, i32, i32);
    (void)__builtin_riscv_th_msub_w_mm(4, 6, 5, i32, i32, i32);
    (void)__builtin_riscv_th_mmul_w_mm(4, 6, 5, i32, i32, i32);
    (void)__builtin_riscv_th_mmulh_w_mm(4, 6, 5, i32, i32, i32);
    (void)__builtin_riscv_th_mmax_w_mm(4, 6, 5, i32, i32, i32);
    (void)__builtin_riscv_th_mumax_w_mm(4, 6, 5, i32, i32, i32);
    (void)__builtin_riscv_th_mmin_w_mm(4, 6, 5, i32, i32, i32);
    (void)__builtin_riscv_th_mumin_w_mm(4, 6, 5, i32, i32, i32);
    (void)__builtin_riscv_th_msrl_w_mm(4, 6, 5, i32, i32, i32);
    (void)__builtin_riscv_th_msll_w_mm(4, 6, 5, i32, i32, i32);
    (void)__builtin_riscv_th_msra_w_mm(4, 6, 5, i32, i32, i32);
}

// CHECK-LABEL: test_ew_fp_all_sizes:
// CHECK: th.mfadd.h.mm acc0, acc2, acc1
// CHECK: th.mfsub.s.mm acc0, acc2, acc1
// CHECK: th.mfmul.d.mm acc0, acc2, acc1
// CHECK: th.mfmax.h.mv.i acc0, acc2, acc1, 1
// CHECK: th.mfmin.s.mv.i acc0, acc2, acc1, 2
// CHECK: th.mfmul.d.mv.i acc0, acc2, acc1, 3
void test_ew_fp_all_sizes(void) {
    __rvm_float16_t f16 = __builtin_riscv_th_mundef_f16();
    __rvm_float32_t f32 = __builtin_riscv_th_mundef_f32();
    __rvm_float64_t f64 = __builtin_riscv_th_mundef_f64();
    (void)__builtin_riscv_th_mfadd_h_mm(4, 6, 5, f16, f16, f16);
    (void)__builtin_riscv_th_mfsub_s_mm(4, 6, 5, f32, f32, f32);
    (void)__builtin_riscv_th_mfmul_d_mm(4, 6, 5, f64, f64, f64);
    (void)__builtin_riscv_th_mfmax_h_mv_i(4, 6, 5, f16, f16, 1);
    (void)__builtin_riscv_th_mfmin_s_mv_i(4, 6, 5, f32, f32, 2);
    (void)__builtin_riscv_th_mfmul_d_mv_i(4, 6, 5, f64, f64, 3);
}

// CHECK-LABEL: test_conversions_typed:
// CHECK: th.mfcvtl.s.h acc0, acc1
// CHECK: th.mfcvtl.d.s acc0, acc1
// CHECK: th.mfcvtl.h.s acc0, acc1
// CHECK: th.mfcvtl.s.d acc0, acc1
void test_conversions_typed(void) {
    __rvm_float16_t f16 = __builtin_riscv_th_mundef_f16();
    __rvm_float32_t f32 = __builtin_riscv_th_mundef_f32();
    __rvm_float64_t f64 = __builtin_riscv_th_mundef_f64();
    (void)__builtin_riscv_th_mfcvtl_s_h(4, 5, f16);
    (void)__builtin_riscv_th_mfcvtl_d_s(4, 5, f32);
    (void)__builtin_riscv_th_mfcvtl_h_s(4, 5, f32);
    (void)__builtin_riscv_th_mfcvtl_s_d(4, 5, f64);
}

// CHECK-LABEL: test_conversions_untyped:
// CHECK: th.mfcvtl.h.e5 acc0, acc1
// CHECK: th.mfcvth.h.e4 acc0, acc1
// CHECK: th.mfcvth.s.bf16 acc0, acc1
// CHECK: th.mfcvtl.e4.h acc0, acc1
// CHECK: th.mfcvth.bf16.s acc0, acc1
void test_conversions_untyped(void) {
    __builtin_riscv_th_mfcvtl_h_e5(4, 5);
    __builtin_riscv_th_mfcvth_h_e4(4, 5);
    __builtin_riscv_th_mfcvth_s_bf16(4, 5);
    __builtin_riscv_th_mfcvtl_e4_h(4, 5);
    __builtin_riscv_th_mfcvth_bf16_s(4, 5);
}

// CHECK-LABEL: test_n4clip_all:
// CHECK: th.mn4clipl.w.mm acc0, acc2, acc1
// CHECK: th.mn4cliph.w.mm acc0, acc2, acc1
// CHECK: th.mn4cliplu.w.mm acc0, acc2, acc1
// CHECK: th.mn4cliphu.w.mm acc0, acc2, acc1
// CHECK: th.mn4cliph.w.mv.i acc0, acc2, acc1, 2
// CHECK: th.mn4cliphu.w.mv.i acc0, acc2, acc1, 4
void test_n4clip_all(void) {
    __builtin_riscv_th_mn4clipl_w_mm(4, 6, 5);
    __builtin_riscv_th_mn4cliph_w_mm(4, 6, 5);
    __builtin_riscv_th_mn4cliplu_w_mm(4, 6, 5);
    __builtin_riscv_th_mn4cliphu_w_mm(4, 6, 5);
    __builtin_riscv_th_mn4cliph_w_mv_i(4, 6, 5, 2);
    __builtin_riscv_th_mn4cliphu_w_mv_i(4, 6, 5, 4);
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
void test_all_gpr_sizes(long val, long idx) {
    long rb = __builtin_riscv_th_mmovb_x_m(0, idx);
    long rh = __builtin_riscv_th_mmovh_x_m(0, idx);
    long rd = __builtin_riscv_th_mmovd_x_m(0, idx);
    __builtin_riscv_th_mmovb_m_x(0, rb, idx);
    __builtin_riscv_th_mmovh_m_x(0, rh, idx);
    __builtin_riscv_th_mmovd_m_x(0, rd, idx);
    __builtin_riscv_th_mdupb_m_x(0, val);
    __builtin_riscv_th_mduph_m_x(0, val);
    __builtin_riscv_th_mdupd_m_x(0, val);
}

// CHECK-LABEL: test_mundef:
// CHECK: ret
void test_mundef(void) {
    // mundef builtins return PoisonValue — no instructions emitted
    __rvm_int32_t a = __builtin_riscv_th_mundef_i32();
    __rvm_float32_t b = __builtin_riscv_th_mundef_f32();
    __rvm_int8x2_t c = __builtin_riscv_th_mundef_i8x2();
    (void)a; (void)b; (void)c;
}

// ==========================================================================
// Register flexibility tests — verify that register index parameters
// correctly select different physical registers in the output assembly.
// ==========================================================================

// CHECK-LABEL: test_multi_accumulator_matmul:
// CHECK: th.mlae8 tr0
// CHECK: th.mlbe8 tr1
// CHECK: th.mzero acc0
// CHECK: th.mfmacc.h.e4 acc0, tr1, tr0
// CHECK: th.mzero acc1
// CHECK: th.mfmacc.h.e4 acc1, tr1, tr0
// CHECK: th.msce32 acc0
// CHECK: th.msce32 acc1
void test_multi_accumulator_matmul(void *a, void *b, void *c1, void *c2,
                                   long stride) {
    // Load A into tr0, B into tr1
    __builtin_riscv_th_mlae8(0, a, stride);
    __builtin_riscv_th_mlbe8(1, b, stride);

    // Matmul into acc0
    __builtin_riscv_th_mzero(4);
    __builtin_riscv_th_mfmacc_h_e4(4, 1, 0);

    // Matmul into acc1 (different accumulator!)
    __builtin_riscv_th_mzero(5);
    __builtin_riscv_th_mfmacc_h_e4(5, 1, 0);

    // Store both accumulators
    __builtin_riscv_th_msce32(4, c1, stride);
    __builtin_riscv_th_msce32(5, c2, stride);
}

// CHECK-LABEL: test_custom_load_registers:
// CHECK: th.mlae8 tr2
// CHECK: th.mlbe8 tr3
// CHECK: th.mlce32 acc1
void test_custom_load_registers(void *a, void *b, void *c, long stride) {
    // Load A into tr2 instead of default tr0
    __builtin_riscv_th_mlae8(2, a, stride);
    // Load B into tr3 instead of default tr1
    __builtin_riscv_th_mlbe8(3, b, stride);
    // Load C into acc1 instead of default acc0
    __builtin_riscv_th_mlce32(5, c, stride);
}

// CHECK-LABEL: test_mzero_different_registers:
// CHECK: th.mzero acc0
// CHECK: th.mzero acc1
// CHECK: th.mzero acc2
// CHECK: th.mzero acc3
// CHECK: th.mzero tr0
// CHECK: th.mzero tr3
void test_mzero_different_registers(void) {
    // mzero can target any register (0-7)
    __builtin_riscv_th_mzero(4);  // acc0
    __builtin_riscv_th_mzero(5);  // acc1
    __builtin_riscv_th_mzero(6);  // acc2
    __builtin_riscv_th_mzero(7);  // acc3
    __builtin_riscv_th_mzero(0);  // tr0
    __builtin_riscv_th_mzero(3);  // tr3
}

// CHECK-LABEL: test_ew_different_acc_registers:
// CHECK: th.mn4clipl.w.mm acc1, acc3, acc2
void test_ew_different_acc_registers(void) {
    // EW operations can use different acc register combinations
    __builtin_riscv_th_mn4clipl_w_mm(5, 7, 6);  // acc1, acc3, acc2
}

// CHECK-LABEL: test_conversion_different_acc:
// CHECK: th.mfcvtl.h.e4 acc1, acc2
void test_conversion_different_acc(void) {
    // Conversions can use different acc registers
    __builtin_riscv_th_mfcvtl_h_e4(5, 6);  // acc1 ← acc2
}

// CHECK-LABEL: test_matmul_all_tile_pairs:
// CHECK: th.mfmacc.s.e4 acc0, tr3, tr2
void test_matmul_all_tile_pairs(void) {
    // Matmul with tr2/tr3 instead of default tr0/tr1
    __builtin_riscv_th_mfmacc_s_e4(4, 3, 2);  // acc0 = tr3 * tr2
}

// CHECK-LABEL: test_store_different_registers:
// CHECK: th.msae8 tr2
// CHECK: th.msbe8 tr3
// CHECK: th.msce32 acc1
void test_store_different_registers(void *a, void *b, void *c, long stride) {
    __builtin_riscv_th_msae8(2, a, stride);   // store from tr2
    __builtin_riscv_th_msbe8(3, b, stride);   // store from tr3
    __builtin_riscv_th_msce32(5, c, stride);  // store from acc1
}

// CHECK-LABEL: test_mmov_custom_registers:
// CHECK: th.mmov.mm acc0, acc1
// CHECK: th.mmov.mm tr2, tr3
void test_mmov_custom_registers(void) {
    // mmov.mm can move between any registers
    __builtin_riscv_th_mmov_mm(4, 5);  // acc0 ← acc1
    __builtin_riscv_th_mmov_mm(2, 3);  // tr2 ← tr3
}

// CHECK-LABEL: test_slide_custom_registers:
// CHECK: th.mrslidedown acc0, acc1
void test_slide_custom_registers(void) {
    // Slides can use acc registers
    __builtin_riscv_th_mrslidedown(4, 5, 2);
}

// CHECK-LABEL: test_whole_register_any:
// CHECK: th.mlme8 acc0
// CHECK: th.msme8 acc1
void test_whole_register_any(void *p) {
    // Whole-register load/store can use any register
    __builtin_riscv_th_mlme8(4, p);   // load whole into acc0
    __builtin_riscv_th_msme8(5, p);   // store whole from acc1
}

// CHECK-LABEL: test_mdup_custom_register:
// CHECK: th.mdupw.m.x acc0
void test_mdup_custom_register(long val) {
    // mdup can target any register including acc
    __builtin_riscv_th_mdupw_m_x(4, val);  // dup into acc0
}

// CHECK-LABEL: test_pack_custom_registers:
// CHECK: th.mpack acc0, acc2, acc1
void test_pack_custom_registers(void) {
    // Pack can use acc registers
    __builtin_riscv_th_mpack(4, 6, 5);  // pack acc0, acc2, acc1
}
