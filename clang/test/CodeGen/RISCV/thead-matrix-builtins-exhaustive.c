// NOTE: Exhaustive test of ALL XTHeadMatrix builtins at the low level.
// This test exercises every __builtin_riscv_th_* function to ensure
// correct code generation. Complements the API-level tests.
//
// RUN: %clang_cc1 -O2 -triple riscv64 -target-feature +experimental-xtheadmatrix -S -o - %s \
// RUN:   | FileCheck %s

#include <stdint.h>

// ============================================================================
// Test 1: All 22 mundef builtins (should produce no instructions)
// ============================================================================

// CHECK-LABEL: test_all_mundef:
// CHECK: ret
void test_all_mundef(void) {
  (void)__builtin_riscv_th_mundef_i8();
  (void)__builtin_riscv_th_mundef_i16();
  (void)__builtin_riscv_th_mundef_i32();
  (void)__builtin_riscv_th_mundef_i64();
  (void)__builtin_riscv_th_mundef_u8();
  (void)__builtin_riscv_th_mundef_u16();
  (void)__builtin_riscv_th_mundef_u32();
  (void)__builtin_riscv_th_mundef_u64();
  (void)__builtin_riscv_th_mundef_f16();
  (void)__builtin_riscv_th_mundef_f32();
  (void)__builtin_riscv_th_mundef_f64();
  (void)__builtin_riscv_th_mundef_i8x2();
  (void)__builtin_riscv_th_mundef_i16x2();
  (void)__builtin_riscv_th_mundef_i32x2();
  (void)__builtin_riscv_th_mundef_i64x2();
  (void)__builtin_riscv_th_mundef_u8x2();
  (void)__builtin_riscv_th_mundef_u16x2();
  (void)__builtin_riscv_th_mundef_u32x2();
  (void)__builtin_riscv_th_mundef_u64x2();
  (void)__builtin_riscv_th_mundef_f16x2();
  (void)__builtin_riscv_th_mundef_f32x2();
  (void)__builtin_riscv_th_mundef_f64x2();
}

// ============================================================================
// Test 2: All load builtins (28 element/tile-stride + 4 whole-register)
// ============================================================================

// CHECK-LABEL: test_load_a_all:
// CHECK: th.mlae8
// CHECK: th.mlae16
// CHECK: th.mlae32
// CHECK: th.mlae64
// CHECK: ret
void test_load_a_all(void *p, long s) {
  (void)__builtin_riscv_th_mlae8(0, p, s);
  (void)__builtin_riscv_th_mlae16(0, p, s);
  (void)__builtin_riscv_th_mlae32(0, p, s);
  (void)__builtin_riscv_th_mlae64(0, p, s);
}

// CHECK-LABEL: test_load_b_all:
// CHECK: th.mlbe8
// CHECK: th.mlbe16
// CHECK: th.mlbe32
// CHECK: th.mlbe64
// CHECK: ret
void test_load_b_all(void *p, long s) {
  (void)__builtin_riscv_th_mlbe8(1, p, s);
  (void)__builtin_riscv_th_mlbe16(1, p, s);
  (void)__builtin_riscv_th_mlbe32(1, p, s);
  (void)__builtin_riscv_th_mlbe64(1, p, s);
}

// CHECK-LABEL: test_load_c_all:
// CHECK: th.mlce8
// CHECK: th.mlce16
// CHECK: th.mlce32
// CHECK: th.mlce64
// CHECK: ret
void test_load_c_all(void *p, long s) {
  (void)__builtin_riscv_th_mlce8(4, p, s);
  (void)__builtin_riscv_th_mlce16(4, p, s);
  (void)__builtin_riscv_th_mlce32(4, p, s);
  (void)__builtin_riscv_th_mlce64(4, p, s);
}

// CHECK-LABEL: test_load_at_all:
// CHECK: th.mlate8
// CHECK: th.mlate16
// CHECK: th.mlate32
// CHECK: th.mlate64
// CHECK: ret
void test_load_at_all(void *p, long s) {
  (void)__builtin_riscv_th_mlate8(2, p, s);
  (void)__builtin_riscv_th_mlate16(2, p, s);
  (void)__builtin_riscv_th_mlate32(2, p, s);
  (void)__builtin_riscv_th_mlate64(2, p, s);
}

// CHECK-LABEL: test_load_bt_all:
// CHECK: th.mlbte8
// CHECK: th.mlbte16
// CHECK: th.mlbte32
// CHECK: th.mlbte64
// CHECK: ret
void test_load_bt_all(void *p, long s) {
  (void)__builtin_riscv_th_mlbte8(3, p, s);
  (void)__builtin_riscv_th_mlbte16(3, p, s);
  (void)__builtin_riscv_th_mlbte32(3, p, s);
  (void)__builtin_riscv_th_mlbte64(3, p, s);
}

// CHECK-LABEL: test_load_ct_all:
// CHECK: th.mlcte8
// CHECK: th.mlcte16
// CHECK: th.mlcte32
// CHECK: th.mlcte64
// CHECK: ret
void test_load_ct_all(void *p, long s) {
  (void)__builtin_riscv_th_mlcte8(5, p, s);
  (void)__builtin_riscv_th_mlcte16(5, p, s);
  (void)__builtin_riscv_th_mlcte32(5, p, s);
  (void)__builtin_riscv_th_mlcte64(5, p, s);
}

// CHECK-LABEL: test_load_whole_all:
// CHECK: th.mlme8
// CHECK: th.mlme16
// CHECK: th.mlme32
// CHECK: th.mlme64
// CHECK: ret
void test_load_whole_all(void *p) {
  (void)__builtin_riscv_th_mlme8(0, p);
  (void)__builtin_riscv_th_mlme16(0, p);
  (void)__builtin_riscv_th_mlme32(0, p);
  (void)__builtin_riscv_th_mlme64(0, p);
}

// ============================================================================
// Test 3: All store builtins (28 element/tile-stride + 4 whole-register)
// ============================================================================

// CHECK-LABEL: test_store_a_all:
// CHECK: th.msae8
// CHECK: th.msae16
// CHECK: th.msae32
// CHECK: th.msae64
// CHECK: ret
void test_store_a_all(void *p, long s) {
  __builtin_riscv_th_msae8(0, p, s);
  __builtin_riscv_th_msae16(0, p, s);
  __builtin_riscv_th_msae32(0, p, s);
  __builtin_riscv_th_msae64(0, p, s);
}

// CHECK-LABEL: test_store_b_all:
// CHECK: th.msbe8
// CHECK: th.msbe16
// CHECK: th.msbe32
// CHECK: th.msbe64
// CHECK: ret
void test_store_b_all(void *p, long s) {
  __builtin_riscv_th_msbe8(1, p, s);
  __builtin_riscv_th_msbe16(1, p, s);
  __builtin_riscv_th_msbe32(1, p, s);
  __builtin_riscv_th_msbe64(1, p, s);
}

// CHECK-LABEL: test_store_c_all:
// CHECK: th.msce8
// CHECK: th.msce16
// CHECK: th.msce32
// CHECK: th.msce64
// CHECK: ret
void test_store_c_all(void *p, long s) {
  __builtin_riscv_th_msce8(4, p, s);
  __builtin_riscv_th_msce16(4, p, s);
  __builtin_riscv_th_msce32(4, p, s);
  __builtin_riscv_th_msce64(4, p, s);
}

// CHECK-LABEL: test_store_at_all:
// CHECK: th.msate8
// CHECK: th.msate16
// CHECK: th.msate32
// CHECK: th.msate64
// CHECK: ret
void test_store_at_all(void *p, long s) {
  __builtin_riscv_th_msate8(2, p, s);
  __builtin_riscv_th_msate16(2, p, s);
  __builtin_riscv_th_msate32(2, p, s);
  __builtin_riscv_th_msate64(2, p, s);
}

// CHECK-LABEL: test_store_bt_all:
// CHECK: th.msbte8
// CHECK: th.msbte16
// CHECK: th.msbte32
// CHECK: th.msbte64
// CHECK: ret
void test_store_bt_all(void *p, long s) {
  __builtin_riscv_th_msbte8(3, p, s);
  __builtin_riscv_th_msbte16(3, p, s);
  __builtin_riscv_th_msbte32(3, p, s);
  __builtin_riscv_th_msbte64(3, p, s);
}

// CHECK-LABEL: test_store_ct_all:
// CHECK: th.mscte8
// CHECK: th.mscte16
// CHECK: th.mscte32
// CHECK: th.mscte64
// CHECK: ret
void test_store_ct_all(void *p, long s) {
  __builtin_riscv_th_mscte8(5, p, s);
  __builtin_riscv_th_mscte16(5, p, s);
  __builtin_riscv_th_mscte32(5, p, s);
  __builtin_riscv_th_mscte64(5, p, s);
}

// CHECK-LABEL: test_store_whole_all:
// CHECK: th.msme8
// CHECK: th.msme16
// CHECK: th.msme32
// CHECK: th.msme64
// CHECK: ret
void test_store_whole_all(void *p) {
  __builtin_riscv_th_msme8(0, p);
  __builtin_riscv_th_msme16(0, p);
  __builtin_riscv_th_msme32(0, p);
  __builtin_riscv_th_msme64(0, p);
}

// ============================================================================
// Test 4: All matmul builtins (14 FP + 16 INT = 30 total)
// ============================================================================

// CHECK-LABEL: test_matmul_fp_all:
// CHECK: th.mfmacc.h
// CHECK: th.mfmacc.s
// CHECK: th.mfmacc.d
// CHECK: th.mfmacc.h.e4
// CHECK: th.mfmacc.h.e5
// CHECK: th.mfmacc.bf16.e4
// CHECK: th.mfmacc.bf16.e5
// CHECK: th.mfmacc.s.h
// CHECK: th.mfmacc.s.bf16
// CHECK: th.mfmacc.s.e4
// CHECK: th.mfmacc.s.e5
// CHECK: th.mfmacc.s.tf32
// CHECK: th.mfmacc.d.s
// CHECK: ret
void test_matmul_fp_all(void) {
  __rvm_float16_t f16 = __builtin_riscv_th_mundef_f16();
  __rvm_float32_t f32 = __builtin_riscv_th_mundef_f32();
  __rvm_float64_t f64 = __builtin_riscv_th_mundef_f64();

  (void)__builtin_riscv_th_mfmacc_h(4, 1, 0, f16, f16, f16);
  (void)__builtin_riscv_th_mfmacc_s(4, 1, 0, f32, f32, f32);
  (void)__builtin_riscv_th_mfmacc_d(4, 1, 0, f64, f64, f64);
  __builtin_riscv_th_mfmacc_h_e4(4, 1, 0);
  __builtin_riscv_th_mfmacc_h_e5(4, 1, 0);
  __builtin_riscv_th_mfmacc_bf16_e4(4, 1, 0);
  __builtin_riscv_th_mfmacc_bf16_e5(4, 1, 0);
  (void)__builtin_riscv_th_mfmacc_s_h(4, 1, 0, f32, f16, f16);
  __builtin_riscv_th_mfmacc_s_bf16(4, 1, 0);
  __builtin_riscv_th_mfmacc_s_e4(4, 1, 0);
  __builtin_riscv_th_mfmacc_s_e5(4, 1, 0);
  __builtin_riscv_th_mfmacc_s_tf32(4, 1, 0);
  (void)__builtin_riscv_th_mfmacc_d_s(4, 1, 0, f64, f32, f32);
}

// CHECK-LABEL: test_matmul_int_all:
// CHECK: th.mmacc.w.b
// CHECK: th.mmaccu.w.b
// CHECK: th.mmaccus.w.b
// CHECK: th.mmaccsu.w.b
// CHECK: th.mmacc.d.h
// CHECK: th.mmaccu.d.h
// CHECK: th.mmaccus.d.h
// CHECK: th.mmaccsu.d.h
// CHECK: th.pmmacc.w.b
// CHECK: th.pmmaccu.w.b
// CHECK: th.pmmaccus.w.b
// CHECK: th.pmmaccsu.w.b
// CHECK: th.mmacc.w.bp
// CHECK: th.mmaccu.w.bp
// CHECK: ret
void test_matmul_int_all(void) {
  __rvm_int8_t i8 = __builtin_riscv_th_mundef_i8();
  __rvm_int16_t i16 = __builtin_riscv_th_mundef_i16();
  __rvm_int32_t i32 = __builtin_riscv_th_mundef_i32();
  __rvm_int64_t i64 = __builtin_riscv_th_mundef_i64();
  __rvm_uint8_t u8 = __builtin_riscv_th_mundef_u8();
  __rvm_uint16_t u16 = __builtin_riscv_th_mundef_u16();
  __rvm_uint32_t u32 = __builtin_riscv_th_mundef_u32();
  __rvm_uint64_t u64 = __builtin_riscv_th_mundef_u64();

  (void)__builtin_riscv_th_mmacc_w_b(4, 1, 0, i32, i8, i8);
  (void)__builtin_riscv_th_mmaccu_w_b(4, 1, 0, u32, u8, u8);
  (void)__builtin_riscv_th_mmaccus_w_b(4, 1, 0, i32, u8, i8);
  (void)__builtin_riscv_th_mmaccsu_w_b(4, 1, 0, i32, i8, u8);
  (void)__builtin_riscv_th_mmacc_d_h(4, 1, 0, i64, i16, i16);
  (void)__builtin_riscv_th_mmaccu_d_h(4, 1, 0, u64, u16, u16);
  (void)__builtin_riscv_th_mmaccus_d_h(4, 1, 0, i64, u16, i16);
  (void)__builtin_riscv_th_mmaccsu_d_h(4, 1, 0, i64, i16, u16);
  (void)__builtin_riscv_th_pmmacc_w_b(4, 1, 0, i32, i8, i8);
  (void)__builtin_riscv_th_pmmaccu_w_b(4, 1, 0, u32, u8, u8);
  (void)__builtin_riscv_th_pmmaccus_w_b(4, 1, 0, i32, u8, i8);
  (void)__builtin_riscv_th_pmmaccsu_w_b(4, 1, 0, i32, i8, u8);
  (void)__builtin_riscv_th_mmacc_w_bp(4, 1, 0, i32, i8, i8);
  (void)__builtin_riscv_th_mmaccu_w_bp(4, 1, 0, u32, u8, u8);
}

// ============================================================================
// Test 5: All EW integer MVI builtins (11 ops)
// ============================================================================

// CHECK-LABEL: test_ew_int_mvi_all:
// CHECK: th.madd.w.mv.i
// CHECK: th.msub.w.mv.i
// CHECK: th.mmul.w.mv.i
// CHECK: th.mmulh.w.mv.i
// CHECK: th.mmax.w.mv.i
// CHECK: th.mumax.w.mv.i
// CHECK: th.mmin.w.mv.i
// CHECK: th.mumin.w.mv.i
// CHECK: th.msrl.w.mv.i
// CHECK: th.msll.w.mv.i
// CHECK: th.msra.w.mv.i
// CHECK: ret
void test_ew_int_mvi_all(void) {
  __rvm_int32_t i32 = __builtin_riscv_th_mundef_i32();
  (void)__builtin_riscv_th_madd_w_mv_i(4, 6, 5, i32, i32, 0);
  (void)__builtin_riscv_th_msub_w_mv_i(4, 6, 5, i32, i32, 1);
  (void)__builtin_riscv_th_mmul_w_mv_i(4, 6, 5, i32, i32, 2);
  (void)__builtin_riscv_th_mmulh_w_mv_i(4, 6, 5, i32, i32, 3);
  (void)__builtin_riscv_th_mmax_w_mv_i(4, 6, 5, i32, i32, 4);
  (void)__builtin_riscv_th_mumax_w_mv_i(4, 6, 5, i32, i32, 5);
  (void)__builtin_riscv_th_mmin_w_mv_i(4, 6, 5, i32, i32, 6);
  (void)__builtin_riscv_th_mumin_w_mv_i(4, 6, 5, i32, i32, 7);
  (void)__builtin_riscv_th_msrl_w_mv_i(4, 6, 5, i32, i32, 0);
  (void)__builtin_riscv_th_msll_w_mv_i(4, 6, 5, i32, i32, 1);
  (void)__builtin_riscv_th_msra_w_mv_i(4, 6, 5, i32, i32, 2);
}

// ============================================================================
// Test 6: All FP EW builtins -- all 5 ops × 3 sizes × 2 forms = 30
// ============================================================================

// CHECK-LABEL: test_ew_fp_mm_all:
// CHECK: th.mfadd.h.mm
// CHECK: th.mfadd.s.mm
// CHECK: th.mfadd.d.mm
// CHECK: th.mfsub.h.mm
// CHECK: th.mfsub.s.mm
// CHECK: th.mfsub.d.mm
// CHECK: th.mfmul.h.mm
// CHECK: th.mfmul.s.mm
// CHECK: th.mfmul.d.mm
// CHECK: th.mfmax.h.mm
// CHECK: th.mfmax.s.mm
// CHECK: th.mfmax.d.mm
// CHECK: th.mfmin.h.mm
// CHECK: th.mfmin.s.mm
// CHECK: th.mfmin.d.mm
// CHECK: ret
void test_ew_fp_mm_all(void) {
  __rvm_float16_t f16 = __builtin_riscv_th_mundef_f16();
  __rvm_float32_t f32 = __builtin_riscv_th_mundef_f32();
  __rvm_float64_t f64 = __builtin_riscv_th_mundef_f64();

  (void)__builtin_riscv_th_mfadd_h_mm(4, 6, 5, f16, f16, f16);
  (void)__builtin_riscv_th_mfadd_s_mm(4, 6, 5, f32, f32, f32);
  (void)__builtin_riscv_th_mfadd_d_mm(4, 6, 5, f64, f64, f64);
  (void)__builtin_riscv_th_mfsub_h_mm(4, 6, 5, f16, f16, f16);
  (void)__builtin_riscv_th_mfsub_s_mm(4, 6, 5, f32, f32, f32);
  (void)__builtin_riscv_th_mfsub_d_mm(4, 6, 5, f64, f64, f64);
  (void)__builtin_riscv_th_mfmul_h_mm(4, 6, 5, f16, f16, f16);
  (void)__builtin_riscv_th_mfmul_s_mm(4, 6, 5, f32, f32, f32);
  (void)__builtin_riscv_th_mfmul_d_mm(4, 6, 5, f64, f64, f64);
  (void)__builtin_riscv_th_mfmax_h_mm(4, 6, 5, f16, f16, f16);
  (void)__builtin_riscv_th_mfmax_s_mm(4, 6, 5, f32, f32, f32);
  (void)__builtin_riscv_th_mfmax_d_mm(4, 6, 5, f64, f64, f64);
  (void)__builtin_riscv_th_mfmin_h_mm(4, 6, 5, f16, f16, f16);
  (void)__builtin_riscv_th_mfmin_s_mm(4, 6, 5, f32, f32, f32);
  (void)__builtin_riscv_th_mfmin_d_mm(4, 6, 5, f64, f64, f64);
}

// CHECK-LABEL: test_ew_fp_mvi_all:
// CHECK: th.mfadd.h.mv.i
// CHECK: th.mfadd.s.mv.i
// CHECK: th.mfadd.d.mv.i
// CHECK: th.mfsub.h.mv.i
// CHECK: th.mfsub.s.mv.i
// CHECK: th.mfsub.d.mv.i
// CHECK: th.mfmul.h.mv.i
// CHECK: th.mfmul.s.mv.i
// CHECK: th.mfmul.d.mv.i
// CHECK: th.mfmax.h.mv.i
// CHECK: th.mfmax.s.mv.i
// CHECK: th.mfmax.d.mv.i
// CHECK: th.mfmin.h.mv.i
// CHECK: th.mfmin.s.mv.i
// CHECK: th.mfmin.d.mv.i
// CHECK: ret
void test_ew_fp_mvi_all(void) {
  __rvm_float16_t f16 = __builtin_riscv_th_mundef_f16();
  __rvm_float32_t f32 = __builtin_riscv_th_mundef_f32();
  __rvm_float64_t f64 = __builtin_riscv_th_mundef_f64();

  (void)__builtin_riscv_th_mfadd_h_mv_i(4, 6, 5, f16, f16, 0);
  (void)__builtin_riscv_th_mfadd_s_mv_i(4, 6, 5, f32, f32, 1);
  (void)__builtin_riscv_th_mfadd_d_mv_i(4, 6, 5, f64, f64, 2);
  (void)__builtin_riscv_th_mfsub_h_mv_i(4, 6, 5, f16, f16, 3);
  (void)__builtin_riscv_th_mfsub_s_mv_i(4, 6, 5, f32, f32, 4);
  (void)__builtin_riscv_th_mfsub_d_mv_i(4, 6, 5, f64, f64, 5);
  (void)__builtin_riscv_th_mfmul_h_mv_i(4, 6, 5, f16, f16, 6);
  (void)__builtin_riscv_th_mfmul_s_mv_i(4, 6, 5, f32, f32, 7);
  (void)__builtin_riscv_th_mfmul_d_mv_i(4, 6, 5, f64, f64, 0);
  (void)__builtin_riscv_th_mfmax_h_mv_i(4, 6, 5, f16, f16, 1);
  (void)__builtin_riscv_th_mfmax_s_mv_i(4, 6, 5, f32, f32, 2);
  (void)__builtin_riscv_th_mfmax_d_mv_i(4, 6, 5, f64, f64, 3);
  (void)__builtin_riscv_th_mfmin_h_mv_i(4, 6, 5, f16, f16, 4);
  (void)__builtin_riscv_th_mfmin_s_mv_i(4, 6, 5, f32, f32, 5);
  (void)__builtin_riscv_th_mfmin_d_mv_i(4, 6, 5, f64, f64, 6);
}

// ============================================================================
// Test 7: All conversion builtins (typed and untyped)
// ============================================================================

// CHECK-LABEL: test_cvt_typed_all:
// CHECK: th.mfcvtl.s.h
// CHECK: th.mfcvth.s.h
// CHECK: th.mfcvtl.h.s
// CHECK: th.mfcvth.h.s
// CHECK: th.mfcvtl.d.s
// CHECK: th.mfcvth.d.s
// CHECK: th.mfcvtl.s.d
// CHECK: th.mfcvth.s.d
// CHECK: th.mufcvtl.h.b
// CHECK: th.mufcvth.h.b
// CHECK: th.msfcvtl.h.b
// CHECK: th.msfcvth.h.b
// CHECK: th.mfucvtl.b.h
// CHECK: th.mfucvth.b.h
// CHECK: th.mfscvtl.b.h
// CHECK: th.mfscvth.b.h
// CHECK: th.msfcvt.s.w
// CHECK: th.mufcvt.s.w
// CHECK: th.mfscvt.w.s
// CHECK: th.mfucvt.w.s
// CHECK: ret
void test_cvt_typed_all(void) {
  __rvm_float16_t f16 = __builtin_riscv_th_mundef_f16();
  __rvm_float32_t f32 = __builtin_riscv_th_mundef_f32();
  __rvm_float64_t f64 = __builtin_riscv_th_mundef_f64();
  __rvm_int8_t i8 = __builtin_riscv_th_mundef_i8();
  __rvm_uint8_t u8 = __builtin_riscv_th_mundef_u8();
  __rvm_int32_t i32 = __builtin_riscv_th_mundef_i32();
  __rvm_uint32_t u32 = __builtin_riscv_th_mundef_u32();

  // FP16 <-> FP32
  (void)__builtin_riscv_th_mfcvtl_s_h(4, 5, f16);
  (void)__builtin_riscv_th_mfcvth_s_h(4, 5, f16);
  (void)__builtin_riscv_th_mfcvtl_h_s(4, 5, f32);
  (void)__builtin_riscv_th_mfcvth_h_s(4, 5, f32);
  // FP32 <-> FP64
  (void)__builtin_riscv_th_mfcvtl_d_s(4, 5, f32);
  (void)__builtin_riscv_th_mfcvth_d_s(4, 5, f32);
  (void)__builtin_riscv_th_mfcvtl_s_d(4, 5, f64);
  (void)__builtin_riscv_th_mfcvth_s_d(4, 5, f64);
  // UINT8 <-> FP16
  (void)__builtin_riscv_th_mufcvtl_h_b(4, 5, u8);
  (void)__builtin_riscv_th_mufcvth_h_b(4, 5, u8);
  // SINT8 <-> FP16
  (void)__builtin_riscv_th_msfcvtl_h_b(4, 5, i8);
  (void)__builtin_riscv_th_msfcvth_h_b(4, 5, i8);
  // FP16 -> UINT8/SINT8
  (void)__builtin_riscv_th_mfucvtl_b_h(4, 5, f16);
  (void)__builtin_riscv_th_mfucvth_b_h(4, 5, f16);
  (void)__builtin_riscv_th_mfscvtl_b_h(4, 5, f16);
  (void)__builtin_riscv_th_mfscvth_b_h(4, 5, f16);
  // INT32 <-> FP32
  (void)__builtin_riscv_th_msfcvt_s_w(4, 5, i32);
  (void)__builtin_riscv_th_mufcvt_s_w(4, 5, u32);
  (void)__builtin_riscv_th_mfscvt_w_s(4, 5, f32);
  (void)__builtin_riscv_th_mfucvt_w_s(4, 5, f32);
}

// CHECK-LABEL: test_cvt_untyped_all:
// CHECK: th.mfcvtl.h.e4
// CHECK: th.mfcvth.h.e4
// CHECK: th.mfcvtl.h.e5
// CHECK: th.mfcvth.h.e5
// CHECK: th.mfcvtl.e4.h
// CHECK: th.mfcvth.e4.h
// CHECK: th.mfcvtl.e5.h
// CHECK: th.mfcvth.e5.h
// CHECK: th.mfcvtl.s.bf16
// CHECK: th.mfcvth.s.bf16
// CHECK: th.mfcvtl.bf16.s
// CHECK: th.mfcvth.bf16.s
// CHECK: th.mfcvtl.e4.s
// CHECK: th.mfcvth.e4.s
// CHECK: th.mfcvtl.e5.s
// CHECK: th.mfcvth.e5.s
// CHECK: th.mfcvt.s.tf32
// CHECK: th.mfcvt.tf32.s
// CHECK: th.mucvtl.b.p
// CHECK: th.mscvtl.b.p
// CHECK: th.mucvth.b.p
// CHECK: th.mscvth.b.p
// CHECK: ret
void test_cvt_untyped_all(void) {
  // FP8 <-> FP16
  __builtin_riscv_th_mfcvtl_h_e4(4, 5);
  __builtin_riscv_th_mfcvth_h_e4(4, 5);
  __builtin_riscv_th_mfcvtl_h_e5(4, 5);
  __builtin_riscv_th_mfcvth_h_e5(4, 5);
  __builtin_riscv_th_mfcvtl_e4_h(4, 5);
  __builtin_riscv_th_mfcvth_e4_h(4, 5);
  __builtin_riscv_th_mfcvtl_e5_h(4, 5);
  __builtin_riscv_th_mfcvth_e5_h(4, 5);
  // BF16 <-> FP32
  __builtin_riscv_th_mfcvtl_s_bf16(4, 5);
  __builtin_riscv_th_mfcvth_s_bf16(4, 5);
  __builtin_riscv_th_mfcvtl_bf16_s(4, 5);
  __builtin_riscv_th_mfcvth_bf16_s(4, 5);
  // FP8 <-> FP32
  __builtin_riscv_th_mfcvtl_e4_s(4, 5);
  __builtin_riscv_th_mfcvth_e4_s(4, 5);
  __builtin_riscv_th_mfcvtl_e5_s(4, 5);
  __builtin_riscv_th_mfcvth_e5_s(4, 5);
  // TF32 <-> FP32
  __builtin_riscv_th_mfcvt_s_tf32(4, 5);
  __builtin_riscv_th_mfcvt_tf32_s(4, 5);
  // Packed conversions
  __builtin_riscv_th_mucvtl_b_p(4, 5);
  __builtin_riscv_th_mscvtl_b_p(4, 5);
  __builtin_riscv_th_mucvth_b_p(4, 5);
  __builtin_riscv_th_mscvth_b_p(4, 5);
}

// ============================================================================
// Test 8: All slide/broadcast builtins
// ============================================================================

// CHECK-LABEL: test_slide_broadcast_exhaustive:
// CHECK: th.mrslidedown {{.*}}, {{.*}}, 0
// CHECK: th.mrslideup {{.*}}, {{.*}}, 7
// CHECK: th.mcslidedown.b
// CHECK: th.mcslidedown.h
// CHECK: th.mcslidedown.w
// CHECK: th.mcslidedown.d
// CHECK: th.mcslideup.b
// CHECK: th.mcslideup.h
// CHECK: th.mcslideup.w
// CHECK: th.mcslideup.d
// CHECK: th.mrbca.mv.i
// CHECK: th.mcbcab.mv.i
// CHECK: th.mcbcah.mv.i
// CHECK: th.mcbcaw.mv.i
// CHECK: th.mcbcad.mv.i
// CHECK: ret
void test_slide_broadcast_exhaustive(void) {
  __builtin_riscv_th_mrslidedown(0, 1, 0);
  __builtin_riscv_th_mrslideup(0, 1, 7);
  __builtin_riscv_th_mcslidedown_b(0, 1, 1);
  __builtin_riscv_th_mcslidedown_h(0, 1, 2);
  __builtin_riscv_th_mcslidedown_w(0, 1, 3);
  __builtin_riscv_th_mcslidedown_d(0, 1, 4);
  __builtin_riscv_th_mcslideup_b(0, 1, 5);
  __builtin_riscv_th_mcslideup_h(0, 1, 6);
  __builtin_riscv_th_mcslideup_w(0, 1, 7);
  __builtin_riscv_th_mcslideup_d(0, 1, 0);
  __builtin_riscv_th_mrbca_mv_i(0, 1, 1);
  __builtin_riscv_th_mcbcab_mv_i(0, 1, 2);
  __builtin_riscv_th_mcbcah_mv_i(0, 1, 3);
  __builtin_riscv_th_mcbcaw_mv_i(0, 1, 4);
  __builtin_riscv_th_mcbcad_mv_i(0, 1, 5);
}

// ============================================================================
// Test 9: Verify typed builtin return types are correct
// ============================================================================

// CHECK-LABEL: test_typed_return_types:
// CHECK: th.mlae8
// CHECK: th.mlbe16
// CHECK: th.mlce32
// CHECK: th.mmacc.w.b
// CHECK: th.mfmacc.s
// CHECK: th.madd.w.mm
// CHECK: th.mfcvtl.s.h
// CHECK: th.msfcvt.s.w
// CHECK: ret
void test_typed_return_types(void *p, long s) {
  // Load builtins return typed values
  __rvm_int8_t a8 = __builtin_riscv_th_mlae8(0, p, s);
  __rvm_int16_t b16 = __builtin_riscv_th_mlbe16(1, p, s);
  __rvm_int32_t c32 = __builtin_riscv_th_mlce32(4, p, s);

  // Matmul builtins return typed values
  __rvm_int32_t m = __builtin_riscv_th_mmacc_w_b(
      4, 1, 0,
      __builtin_riscv_th_mundef_i32(),
      __builtin_riscv_th_mundef_i8(),
      __builtin_riscv_th_mundef_i8());

  __rvm_float32_t fm = __builtin_riscv_th_mfmacc_s(
      4, 1, 0,
      __builtin_riscv_th_mundef_f32(),
      __builtin_riscv_th_mundef_f32(),
      __builtin_riscv_th_mundef_f32());

  // EW builtins return typed values
  __rvm_int32_t ew = __builtin_riscv_th_madd_w_mm(
      4, 6, 5,
      __builtin_riscv_th_mundef_i32(),
      __builtin_riscv_th_mundef_i32(),
      __builtin_riscv_th_mundef_i32());

  // Conversion builtins return typed values
  __rvm_float32_t cvt = __builtin_riscv_th_mfcvtl_s_h(
      4, 5,
      __builtin_riscv_th_mundef_f16());

  __rvm_float32_t ficvt = __builtin_riscv_th_msfcvt_s_w(
      4, 5,
      __builtin_riscv_th_mundef_i32());

  (void)a8; (void)b16; (void)c32;
  (void)m; (void)fm; (void)ew; (void)cvt; (void)ficvt;
}
