// NOTE: Comprehensive XTHeadMatrix CodeGen test exercising every instruction
// category with flexible register selection. This test verifies end-to-end
// correctness from C builtins through ISel to assembly output.
//
// RUN: %clang_cc1 -O2 -triple riscv64 -target-feature +experimental-xtheadmatrix -S -o - %s \
// RUN:   | FileCheck %s

#include <stdint.h>
#include <stddef.h>

// ============================================================================
// 1. Configuration instructions — no register indices
// ============================================================================

// CHECK-LABEL: test_all_config:
// CHECK: th.mrelease
// CHECK: th.msettilemi 1
// CHECK: th.msettileki 2
// CHECK: th.msettileni 3
// CHECK: th.msettilem
// CHECK: th.msettilek
// CHECK: th.msettilen
void test_all_config(size_t m, size_t k, size_t n) {
    __builtin_riscv_th_mrelease();
    __builtin_riscv_th_msettilemi(1);
    __builtin_riscv_th_msettileki(2);
    __builtin_riscv_th_msettileni(3);
    __builtin_riscv_th_msettilem(m);
    __builtin_riscv_th_msettilek(k);
    __builtin_riscv_th_msettilen(n);
}

// ============================================================================
// 2. Loads — every role × every EEW
// ============================================================================

// CHECK-LABEL: test_load_a_all_eew:
// CHECK: th.mlae8 tr0
// CHECK: th.mlae16 tr1
// CHECK: th.mlae32 tr2
// CHECK: th.mlae64 tr3
void test_load_a_all_eew(void *p, size_t s) {
    (void)__builtin_riscv_th_mlae8(0, p, s);
    (void)__builtin_riscv_th_mlae16(1, p, s);
    (void)__builtin_riscv_th_mlae32(2, p, s);
    (void)__builtin_riscv_th_mlae64(3, p, s);
}

// CHECK-LABEL: test_load_b_all_eew:
// CHECK: th.mlbe8 tr0
// CHECK: th.mlbe16 tr1
// CHECK: th.mlbe32 tr2
// CHECK: th.mlbe64 tr3
void test_load_b_all_eew(void *p, size_t s) {
    (void)__builtin_riscv_th_mlbe8(0, p, s);
    (void)__builtin_riscv_th_mlbe16(1, p, s);
    (void)__builtin_riscv_th_mlbe32(2, p, s);
    (void)__builtin_riscv_th_mlbe64(3, p, s);
}

// CHECK-LABEL: test_load_c_all_eew:
// CHECK: th.mlce8 acc0
// CHECK: th.mlce16 acc1
// CHECK: th.mlce32 acc2
// CHECK: th.mlce64 acc3
void test_load_c_all_eew(void *p, size_t s) {
    (void)__builtin_riscv_th_mlce8(4, p, s);
    (void)__builtin_riscv_th_mlce16(5, p, s);
    (void)__builtin_riscv_th_mlce32(6, p, s);
    (void)__builtin_riscv_th_mlce64(7, p, s);
}

// CHECK-LABEL: test_load_transposed_all:
// CHECK: th.mlate8 tr0
// CHECK: th.mlate16 tr1
// CHECK: th.mlate32 tr2
// CHECK: th.mlate64 tr3
// CHECK: th.mlbte8 tr0
// CHECK: th.mlbte16 tr1
// CHECK: th.mlbte32 tr2
// CHECK: th.mlbte64 tr3
// CHECK: th.mlcte8 acc0
// CHECK: th.mlcte16 acc1
// CHECK: th.mlcte32 acc2
// CHECK: th.mlcte64 acc3
void test_load_transposed_all(void *p, size_t s) {
    (void)__builtin_riscv_th_mlate8(0, p, s);
    (void)__builtin_riscv_th_mlate16(1, p, s);
    (void)__builtin_riscv_th_mlate32(2, p, s);
    (void)__builtin_riscv_th_mlate64(3, p, s);
    (void)__builtin_riscv_th_mlbte8(0, p, s);
    (void)__builtin_riscv_th_mlbte16(1, p, s);
    (void)__builtin_riscv_th_mlbte32(2, p, s);
    (void)__builtin_riscv_th_mlbte64(3, p, s);
    (void)__builtin_riscv_th_mlcte8(4, p, s);
    (void)__builtin_riscv_th_mlcte16(5, p, s);
    (void)__builtin_riscv_th_mlcte32(6, p, s);
    (void)__builtin_riscv_th_mlcte64(7, p, s);
}

// CHECK-LABEL: test_load_whole:
// CHECK: th.mlme8 tr0
// CHECK: th.mlme16 tr1
// CHECK: th.mlme32 acc0
// CHECK: th.mlme64 acc3
void test_load_whole(void *p) {
    (void)__builtin_riscv_th_mlme8(0, p);
    (void)__builtin_riscv_th_mlme16(1, p);
    (void)__builtin_riscv_th_mlme32(4, p);
    (void)__builtin_riscv_th_mlme64(7, p);
}

// ============================================================================
// 3. Stores — every role × every EEW
// ============================================================================

// CHECK-LABEL: test_store_a_all_eew:
// CHECK: th.msae8 tr0
// CHECK: th.msae16 tr1
// CHECK: th.msae32 tr2
// CHECK: th.msae64 tr3
void test_store_a_all_eew(void *p, size_t s) {
    __builtin_riscv_th_msae8(0, p, s);
    __builtin_riscv_th_msae16(1, p, s);
    __builtin_riscv_th_msae32(2, p, s);
    __builtin_riscv_th_msae64(3, p, s);
}

// CHECK-LABEL: test_store_b_all_eew:
// CHECK: th.msbe8 tr0
// CHECK: th.msbe16 tr1
// CHECK: th.msbe32 tr2
// CHECK: th.msbe64 tr3
void test_store_b_all_eew(void *p, size_t s) {
    __builtin_riscv_th_msbe8(0, p, s);
    __builtin_riscv_th_msbe16(1, p, s);
    __builtin_riscv_th_msbe32(2, p, s);
    __builtin_riscv_th_msbe64(3, p, s);
}

// CHECK-LABEL: test_store_c_all_eew:
// CHECK: th.msce8 acc0
// CHECK: th.msce16 acc1
// CHECK: th.msce32 acc2
// CHECK: th.msce64 acc3
void test_store_c_all_eew(void *p, size_t s) {
    __builtin_riscv_th_msce8(4, p, s);
    __builtin_riscv_th_msce16(5, p, s);
    __builtin_riscv_th_msce32(6, p, s);
    __builtin_riscv_th_msce64(7, p, s);
}

// CHECK-LABEL: test_store_transposed_all:
// CHECK: th.msate8 tr0
// CHECK: th.msate16 tr1
// CHECK: th.msate32 tr2
// CHECK: th.msate64 tr3
// CHECK: th.msbte8 tr0
// CHECK: th.msbte16 tr1
// CHECK: th.msbte32 tr2
// CHECK: th.msbte64 tr3
// CHECK: th.mscte8 acc0
// CHECK: th.mscte16 acc1
// CHECK: th.mscte32 acc2
// CHECK: th.mscte64 acc3
void test_store_transposed_all(void *p, size_t s) {
    __builtin_riscv_th_msate8(0, p, s);
    __builtin_riscv_th_msate16(1, p, s);
    __builtin_riscv_th_msate32(2, p, s);
    __builtin_riscv_th_msate64(3, p, s);
    __builtin_riscv_th_msbte8(0, p, s);
    __builtin_riscv_th_msbte16(1, p, s);
    __builtin_riscv_th_msbte32(2, p, s);
    __builtin_riscv_th_msbte64(3, p, s);
    __builtin_riscv_th_mscte8(4, p, s);
    __builtin_riscv_th_mscte16(5, p, s);
    __builtin_riscv_th_mscte32(6, p, s);
    __builtin_riscv_th_mscte64(7, p, s);
}

// CHECK-LABEL: test_store_whole:
// CHECK: th.msme8 tr0
// CHECK: th.msme16 acc0
// CHECK: th.msme32 acc2
// CHECK: th.msme64 acc3
void test_store_whole(void *p) {
    __builtin_riscv_th_msme8(0, p);
    __builtin_riscv_th_msme16(4, p);
    __builtin_riscv_th_msme32(6, p);
    __builtin_riscv_th_msme64(7, p);
}

// ============================================================================
// 4. All FP matmul variants
// ============================================================================

// CHECK-LABEL: test_fp_matmul_all:
// CHECK: th.mfmacc.h acc0, tr1, tr0
// CHECK: th.mfmacc.s acc1, tr2, tr3
// CHECK: th.mfmacc.d acc2, tr3, tr2
// CHECK: th.mfmacc.h.e4 acc0, tr1, tr0
// CHECK: th.mfmacc.h.e5 acc1, tr0, tr1
// CHECK: th.mfmacc.bf16.e4 acc2, tr2, tr3
// CHECK: th.mfmacc.bf16.e5 acc3, tr3, tr2
// CHECK: th.mfmacc.s.h acc0, tr1, tr0
// CHECK: th.mfmacc.s.bf16 acc1, tr0, tr1
// CHECK: th.mfmacc.s.e4 acc2, tr2, tr3
// CHECK: th.mfmacc.s.e5 acc3, tr3, tr2
// CHECK: th.mfmacc.s.tf32 acc0, tr0, tr1
// CHECK: th.mfmacc.d.s acc1, tr2, tr3
void test_fp_matmul_all(void) {
    __rvm_float16_t f16 = __builtin_riscv_th_mundef_f16();
    __rvm_float32_t f32 = __builtin_riscv_th_mundef_f32();
    __rvm_float64_t f64 = __builtin_riscv_th_mundef_f64();
    // Non-widening FP
    (void)__builtin_riscv_th_mfmacc_h(4, 1, 0, f16, f16, f16);
    (void)__builtin_riscv_th_mfmacc_s(5, 2, 3, f32, f32, f32);
    (void)__builtin_riscv_th_mfmacc_d(6, 3, 2, f64, f64, f64);
    // FP8 -> FP16/BF16
    __builtin_riscv_th_mfmacc_h_e4(4, 1, 0);
    __builtin_riscv_th_mfmacc_h_e5(5, 0, 1);
    __builtin_riscv_th_mfmacc_bf16_e4(6, 2, 3);
    __builtin_riscv_th_mfmacc_bf16_e5(7, 3, 2);
    // Widening to FP32
    (void)__builtin_riscv_th_mfmacc_s_h(4, 1, 0, f32, f16, f16);
    __builtin_riscv_th_mfmacc_s_bf16(5, 0, 1);
    __builtin_riscv_th_mfmacc_s_e4(6, 2, 3);
    __builtin_riscv_th_mfmacc_s_e5(7, 3, 2);
    __builtin_riscv_th_mfmacc_s_tf32(4, 0, 1);
    // Widening to FP64
    (void)__builtin_riscv_th_mfmacc_d_s(5, 2, 3, f64, f32, f32);
}

// ============================================================================
// 5. All INT matmul variants
// ============================================================================

// CHECK-LABEL: test_int_matmul_all:
// CHECK: th.mmacc.w.b acc0, tr1, tr0
// CHECK: th.mmaccu.w.b acc1, tr2, tr3
// CHECK: th.mmaccus.w.b acc2, tr0, tr1
// CHECK: th.mmaccsu.w.b acc3, tr3, tr2
// CHECK: th.mmacc.d.h acc0, tr1, tr0
// CHECK: th.mmaccu.d.h acc1, tr0, tr1
// CHECK: th.mmaccus.d.h acc2, tr2, tr3
// CHECK: th.mmaccsu.d.h acc3, tr3, tr0
// CHECK: th.pmmacc.w.b acc0, tr1, tr0
// CHECK: th.pmmaccu.w.b acc1, tr2, tr3
// CHECK: th.pmmaccus.w.b acc2, tr3, tr2
// CHECK: th.pmmaccsu.w.b acc3, tr0, tr1
// CHECK: th.mmacc.w.bp acc0, tr1, tr0
// CHECK: th.mmaccu.w.bp acc1, tr2, tr3
void test_int_matmul_all(void) {
    __rvm_int8_t i8 = __builtin_riscv_th_mundef_i8();
    __rvm_int16_t i16 = __builtin_riscv_th_mundef_i16();
    __rvm_int32_t i32 = __builtin_riscv_th_mundef_i32();
    __rvm_int64_t i64 = __builtin_riscv_th_mundef_i64();
    __rvm_uint8_t u8 = __builtin_riscv_th_mundef_u8();
    __rvm_uint16_t u16 = __builtin_riscv_th_mundef_u16();
    __rvm_uint32_t u32 = __builtin_riscv_th_mundef_u32();
    __rvm_uint64_t u64 = __builtin_riscv_th_mundef_u64();
    // INT8->INT32
    (void)__builtin_riscv_th_mmacc_w_b(4, 1, 0, i32, i8, i8);
    (void)__builtin_riscv_th_mmaccu_w_b(5, 2, 3, u32, u8, u8);
    (void)__builtin_riscv_th_mmaccus_w_b(6, 0, 1, i32, u8, i8);
    (void)__builtin_riscv_th_mmaccsu_w_b(7, 3, 2, i32, i8, u8);
    // INT16->INT64
    (void)__builtin_riscv_th_mmacc_d_h(4, 1, 0, i64, i16, i16);
    (void)__builtin_riscv_th_mmaccu_d_h(5, 0, 1, u64, u16, u16);
    (void)__builtin_riscv_th_mmaccus_d_h(6, 2, 3, i64, u16, i16);
    (void)__builtin_riscv_th_mmaccsu_d_h(7, 3, 0, i64, i16, u16);
    // Partial INT
    (void)__builtin_riscv_th_pmmacc_w_b(4, 1, 0, i32, i8, i8);
    (void)__builtin_riscv_th_pmmaccu_w_b(5, 2, 3, u32, u8, u8);
    (void)__builtin_riscv_th_pmmaccus_w_b(6, 3, 2, i32, u8, i8);
    (void)__builtin_riscv_th_pmmaccsu_w_b(7, 0, 1, i32, i8, u8);
    // Bypass
    (void)__builtin_riscv_th_mmacc_w_bp(4, 1, 0, i32, i8, i8);
    (void)__builtin_riscv_th_mmaccu_w_bp(5, 2, 3, u32, u8, u8);
}

// ============================================================================
// 6. Misc — mzero with flexible register targeting
// ============================================================================

// CHECK-LABEL: test_mzero_flexible:
// CHECK: th.mzero acc0
// CHECK: th.mzero acc1
// CHECK: th.mzero acc2
// CHECK: th.mzero acc3
// CHECK: th.mzero tr0
// CHECK: th.mzero tr3
// CHECK: th.mzero2r acc0
// CHECK: th.mzero4r acc0
// CHECK: th.mzero8r acc0
void test_mzero_flexible(void) {
    __builtin_riscv_th_mzero(4);
    __builtin_riscv_th_mzero(5);
    __builtin_riscv_th_mzero(6);
    __builtin_riscv_th_mzero(7);
    __builtin_riscv_th_mzero(0);
    __builtin_riscv_th_mzero(3);
    __builtin_riscv_th_mzero2r(4);
    __builtin_riscv_th_mzero4r(4);
    __builtin_riscv_th_mzero8r(4);
}

// ============================================================================
// 7. mmov.mm with all register combinations
// ============================================================================

// CHECK-LABEL: test_mmov_flexible:
// CHECK: th.mmov.mm tr0, tr1
// CHECK: th.mmov.mm acc0, acc1
// CHECK: th.mmov.mm tr0, acc0
// CHECK: th.mmov.mm acc3, tr3
void test_mmov_flexible(void) {
    __builtin_riscv_th_mmov_mm(0, 1);  // tile to tile
    __builtin_riscv_th_mmov_mm(4, 5);  // acc to acc
    __builtin_riscv_th_mmov_mm(0, 4);  // acc to tile
    __builtin_riscv_th_mmov_mm(7, 3);  // tile to acc
}

// ============================================================================
// 8. GPR data move (mmov.x.m, mmov.m.x, mdup.m.x)
// ============================================================================

// CHECK-LABEL: test_gpr_moves:
// CHECK: th.mmovb.x.m {{.*}}, tr0
// CHECK: th.mmovh.x.m {{.*}}, tr1
// CHECK: th.mmovw.x.m {{.*}}, acc0
// CHECK: th.mmovd.x.m {{.*}}, acc3
// CHECK: th.mmovb.m.x tr0
// CHECK: th.mmovh.m.x tr1
// CHECK: th.mmovw.m.x acc0
// CHECK: th.mmovd.m.x acc3
// CHECK: th.mdupb.m.x tr0
// CHECK: th.mduph.m.x tr2
// CHECK: th.mdupw.m.x acc1
// CHECK: th.mdupd.m.x acc3
void test_gpr_moves(size_t data, size_t idx) {
    (void)__builtin_riscv_th_mmovb_x_m(0, idx);
    (void)__builtin_riscv_th_mmovh_x_m(1, idx);
    (void)__builtin_riscv_th_mmovw_x_m(4, idx);
    (void)__builtin_riscv_th_mmovd_x_m(7, idx);
    __builtin_riscv_th_mmovb_m_x(0, data, idx);
    __builtin_riscv_th_mmovh_m_x(1, data, idx);
    __builtin_riscv_th_mmovw_m_x(4, data, idx);
    __builtin_riscv_th_mmovd_m_x(7, data, idx);
    __builtin_riscv_th_mdupb_m_x(0, data);
    __builtin_riscv_th_mduph_m_x(2, data);
    __builtin_riscv_th_mdupw_m_x(5, data);
    __builtin_riscv_th_mdupd_m_x(7, data);
}

// ============================================================================
// 9. Pack with flexible registers
// ============================================================================

// CHECK-LABEL: test_pack_flexible:
// CHECK: th.mpack tr0, tr2, tr1
// CHECK: th.mpackhl tr0, tr2, tr1
// CHECK: th.mpackhh tr0, tr2, tr1
// CHECK: th.mpack acc0, acc2, acc1
// CHECK: th.mpack tr3, tr0, tr1
void test_pack_flexible(void) {
    __builtin_riscv_th_mpack(0, 2, 1);
    __builtin_riscv_th_mpackhl(0, 2, 1);
    __builtin_riscv_th_mpackhh(0, 2, 1);
    __builtin_riscv_th_mpack(4, 6, 5);   // all acc
    __builtin_riscv_th_mpack(3, 0, 1);   // mixed tile
}

// ============================================================================
// 10. All slide variants
// ============================================================================

// CHECK-LABEL: test_slides:
// CHECK: th.mrslidedown tr0, tr1, 3
// CHECK: th.mrslideup tr2, tr3, 5
// CHECK: th.mcslidedown.b acc0, acc1, 1
// CHECK: th.mcslidedown.h tr0, tr1, 2
// CHECK: th.mcslidedown.w acc2, acc3, 3
// CHECK: th.mcslidedown.d tr2, tr3, 4
// CHECK: th.mcslideup.b tr0, tr1, 1
// CHECK: th.mcslideup.h acc0, acc1, 2
// CHECK: th.mcslideup.w tr2, tr3, 3
// CHECK: th.mcslideup.d acc2, acc3, 4
void test_slides(void) {
    __builtin_riscv_th_mrslidedown(0, 1, 3);
    __builtin_riscv_th_mrslideup(2, 3, 5);
    __builtin_riscv_th_mcslidedown_b(4, 5, 1);
    __builtin_riscv_th_mcslidedown_h(0, 1, 2);
    __builtin_riscv_th_mcslidedown_w(6, 7, 3);
    __builtin_riscv_th_mcslidedown_d(2, 3, 4);
    __builtin_riscv_th_mcslideup_b(0, 1, 1);
    __builtin_riscv_th_mcslideup_h(4, 5, 2);
    __builtin_riscv_th_mcslideup_w(2, 3, 3);
    __builtin_riscv_th_mcslideup_d(6, 7, 4);
}

// ============================================================================
// 11. All broadcast variants
// ============================================================================

// CHECK-LABEL: test_broadcasts:
// CHECK: th.mrbca.mv.i tr0, tr1, 2
// CHECK: th.mcbcab.mv.i acc0, acc1, 3
// CHECK: th.mcbcah.mv.i tr0, tr1, 4
// CHECK: th.mcbcaw.mv.i acc2, acc3, 5
// CHECK: th.mcbcad.mv.i tr2, tr3, 6
void test_broadcasts(void) {
    __builtin_riscv_th_mrbca_mv_i(0, 1, 2);
    __builtin_riscv_th_mcbcab_mv_i(4, 5, 3);
    __builtin_riscv_th_mcbcah_mv_i(0, 1, 4);
    __builtin_riscv_th_mcbcaw_mv_i(6, 7, 5);
    __builtin_riscv_th_mcbcad_mv_i(2, 3, 6);
}

// ============================================================================
// 12. FP format conversions — all 26
// ============================================================================

// CHECK-LABEL: test_fp_conversions:
// CHECK: th.mfcvtl.h.e4 acc0, acc1
// CHECK: th.mfcvth.h.e4 acc1, acc0
// CHECK: th.mfcvtl.h.e5 acc2, acc3
// CHECK: th.mfcvth.h.e5 acc3, acc2
// CHECK: th.mfcvtl.e4.h acc0, acc1
// CHECK: th.mfcvth.e4.h acc1, acc0
// CHECK: th.mfcvtl.e5.h acc2, acc3
// CHECK: th.mfcvth.e5.h acc3, acc2
void test_fp_conversions(void) {
    // FP8 <-> FP16
    __builtin_riscv_th_mfcvtl_h_e4(4, 5);
    __builtin_riscv_th_mfcvth_h_e4(5, 4);
    __builtin_riscv_th_mfcvtl_h_e5(6, 7);
    __builtin_riscv_th_mfcvth_h_e5(7, 6);
    __builtin_riscv_th_mfcvtl_e4_h(4, 5);
    __builtin_riscv_th_mfcvth_e4_h(5, 4);
    __builtin_riscv_th_mfcvtl_e5_h(6, 7);
    __builtin_riscv_th_mfcvth_e5_h(7, 6);
}

// CHECK-LABEL: test_fp_conversions_2:
// CHECK: th.mfcvtl.s.h acc0, acc1
// CHECK: th.mfcvth.s.h acc1, acc0
// CHECK: th.mfcvtl.h.s acc2, acc3
// CHECK: th.mfcvth.h.s acc3, acc2
// CHECK: th.mfcvtl.s.bf16 acc0, acc1
// CHECK: th.mfcvth.s.bf16 acc1, acc0
// CHECK: th.mfcvtl.bf16.s acc2, acc3
// CHECK: th.mfcvth.bf16.s acc3, acc2
void test_fp_conversions_2(void) {
    __rvm_float16_t f16 = __builtin_riscv_th_mundef_f16();
    __rvm_float32_t f32 = __builtin_riscv_th_mundef_f32();
    // FP16 <-> FP32 (typed)
    (void)__builtin_riscv_th_mfcvtl_s_h(4, 5, f16);
    (void)__builtin_riscv_th_mfcvth_s_h(5, 4, f16);
    (void)__builtin_riscv_th_mfcvtl_h_s(6, 7, f32);
    (void)__builtin_riscv_th_mfcvth_h_s(7, 6, f32);
    // BF16 <-> FP32
    __builtin_riscv_th_mfcvtl_s_bf16(4, 5);
    __builtin_riscv_th_mfcvth_s_bf16(5, 4);
    __builtin_riscv_th_mfcvtl_bf16_s(6, 7);
    __builtin_riscv_th_mfcvth_bf16_s(7, 6);
}

// CHECK-LABEL: test_fp_conversions_3:
// CHECK: th.mfcvtl.e4.s acc0, acc1
// CHECK: th.mfcvth.e4.s acc1, acc0
// CHECK: th.mfcvtl.e5.s acc2, acc3
// CHECK: th.mfcvth.e5.s acc3, acc2
// CHECK: th.mfcvtl.d.s acc0, acc1
// CHECK: th.mfcvth.d.s acc1, acc0
// CHECK: th.mfcvtl.s.d acc2, acc3
// CHECK: th.mfcvth.s.d acc3, acc2
// CHECK: th.mfcvt.s.tf32 acc0, acc1
// CHECK: th.mfcvt.tf32.s acc1, acc0
void test_fp_conversions_3(void) {
    __rvm_float32_t f32 = __builtin_riscv_th_mundef_f32();
    __rvm_float64_t f64 = __builtin_riscv_th_mundef_f64();
    // FP32 -> FP8
    __builtin_riscv_th_mfcvtl_e4_s(4, 5);
    __builtin_riscv_th_mfcvth_e4_s(5, 4);
    __builtin_riscv_th_mfcvtl_e5_s(6, 7);
    __builtin_riscv_th_mfcvth_e5_s(7, 6);
    // FP32 <-> FP64 (typed)
    (void)__builtin_riscv_th_mfcvtl_d_s(4, 5, f32);
    (void)__builtin_riscv_th_mfcvth_d_s(5, 4, f32);
    (void)__builtin_riscv_th_mfcvtl_s_d(6, 7, f64);
    (void)__builtin_riscv_th_mfcvth_s_d(7, 6, f64);
    // TF32 <-> FP32
    __builtin_riscv_th_mfcvt_s_tf32(4, 5);
    __builtin_riscv_th_mfcvt_tf32_s(5, 4);
}

// ============================================================================
// 13. Float-int conversions — all 12
// ============================================================================

// CHECK-LABEL: test_float_int_conversions:
// CHECK: th.mufcvtl.h.b acc0, acc1
// CHECK: th.mufcvth.h.b acc1, acc0
// CHECK: th.msfcvtl.h.b acc2, acc3
// CHECK: th.msfcvth.h.b acc3, acc2
// CHECK: th.mfucvtl.b.h acc0, acc1
// CHECK: th.mfucvth.b.h acc1, acc0
// CHECK: th.mfscvtl.b.h acc2, acc3
// CHECK: th.mfscvth.b.h acc3, acc2
// CHECK: th.msfcvt.s.w acc0, acc1
// CHECK: th.mufcvt.s.w acc1, acc0
// CHECK: th.mfscvt.w.s acc2, acc3
// CHECK: th.mfucvt.w.s acc3, acc2
void test_float_int_conversions(void) {
    __rvm_uint8_t u8 = __builtin_riscv_th_mundef_u8();
    __rvm_int8_t i8 = __builtin_riscv_th_mundef_i8();
    __rvm_float16_t f16 = __builtin_riscv_th_mundef_f16();
    __rvm_int32_t i32 = __builtin_riscv_th_mundef_i32();
    __rvm_uint32_t u32 = __builtin_riscv_th_mundef_u32();
    __rvm_float32_t f32 = __builtin_riscv_th_mundef_f32();
    (void)__builtin_riscv_th_mufcvtl_h_b(4, 5, u8);
    (void)__builtin_riscv_th_mufcvth_h_b(5, 4, u8);
    (void)__builtin_riscv_th_msfcvtl_h_b(6, 7, i8);
    (void)__builtin_riscv_th_msfcvth_h_b(7, 6, i8);
    (void)__builtin_riscv_th_mfucvtl_b_h(4, 5, f16);
    (void)__builtin_riscv_th_mfucvth_b_h(5, 4, f16);
    (void)__builtin_riscv_th_mfscvtl_b_h(6, 7, f16);
    (void)__builtin_riscv_th_mfscvth_b_h(7, 6, f16);
    (void)__builtin_riscv_th_msfcvt_s_w(4, 5, i32);
    (void)__builtin_riscv_th_mufcvt_s_w(5, 4, u32);
    (void)__builtin_riscv_th_mfscvt_w_s(6, 7, f32);
    (void)__builtin_riscv_th_mfucvt_w_s(7, 6, f32);
}

// ============================================================================
// 14. N4clip and packed conversions
// ============================================================================

// CHECK-LABEL: test_n4clip_and_packed:
// CHECK: th.mn4clipl.w.mm acc0, acc2, acc1
// CHECK: th.mn4cliph.w.mm acc1, acc3, acc0
// CHECK: th.mn4cliplu.w.mm acc2, acc0, acc3
// CHECK: th.mn4cliphu.w.mm acc3, acc1, acc2
// CHECK: th.mn4clipl.w.mv.i acc0, acc2, acc1, 3
// CHECK: th.mn4cliph.w.mv.i acc1, acc3, acc0, 5
// CHECK: th.mn4cliplu.w.mv.i acc2, acc0, acc3, 2
// CHECK: th.mn4cliphu.w.mv.i acc3, acc1, acc2, 7
// CHECK: th.mucvtl.b.p acc0, acc1
// CHECK: th.mscvtl.b.p acc1, acc0
// CHECK: th.mucvth.b.p acc2, acc3
// CHECK: th.mscvth.b.p acc3, acc2
void test_n4clip_and_packed(void) {
    __builtin_riscv_th_mn4clipl_w_mm(4, 6, 5);
    __builtin_riscv_th_mn4cliph_w_mm(5, 7, 4);
    __builtin_riscv_th_mn4cliplu_w_mm(6, 4, 7);
    __builtin_riscv_th_mn4cliphu_w_mm(7, 5, 6);
    __builtin_riscv_th_mn4clipl_w_mv_i(4, 6, 5, 3);
    __builtin_riscv_th_mn4cliph_w_mv_i(5, 7, 4, 5);
    __builtin_riscv_th_mn4cliplu_w_mv_i(6, 4, 7, 2);
    __builtin_riscv_th_mn4cliphu_w_mv_i(7, 5, 6, 7);
    __builtin_riscv_th_mucvtl_b_p(4, 5);
    __builtin_riscv_th_mscvtl_b_p(5, 4);
    __builtin_riscv_th_mucvth_b_p(6, 7);
    __builtin_riscv_th_mscvth_b_p(7, 6);
}

// ============================================================================
// 15. Integer EW arithmetic — all 11 ops, both MM and MVI
// ============================================================================

// CHECK-LABEL: test_int_ew_mm:
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
void test_int_ew_mm(void) {
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

// CHECK-LABEL: test_int_ew_mvi:
// CHECK: th.madd.w.mv.i acc0, acc2, acc1, 3
// CHECK: th.msub.w.mv.i acc0, acc2, acc1, 3
// CHECK: th.mmul.w.mv.i acc0, acc2, acc1, 3
// CHECK: th.mmulh.w.mv.i acc0, acc2, acc1, 3
// CHECK: th.mmax.w.mv.i acc0, acc2, acc1, 3
// CHECK: th.mumax.w.mv.i acc0, acc2, acc1, 3
// CHECK: th.mmin.w.mv.i acc0, acc2, acc1, 3
// CHECK: th.mumin.w.mv.i acc0, acc2, acc1, 3
// CHECK: th.msrl.w.mv.i acc0, acc2, acc1, 3
// CHECK: th.msll.w.mv.i acc0, acc2, acc1, 3
// CHECK: th.msra.w.mv.i acc0, acc2, acc1, 3
void test_int_ew_mvi(void) {
    __rvm_int32_t i32 = __builtin_riscv_th_mundef_i32();
    (void)__builtin_riscv_th_madd_w_mv_i(4, 6, 5, i32, i32, 3);
    (void)__builtin_riscv_th_msub_w_mv_i(4, 6, 5, i32, i32, 3);
    (void)__builtin_riscv_th_mmul_w_mv_i(4, 6, 5, i32, i32, 3);
    (void)__builtin_riscv_th_mmulh_w_mv_i(4, 6, 5, i32, i32, 3);
    (void)__builtin_riscv_th_mmax_w_mv_i(4, 6, 5, i32, i32, 3);
    (void)__builtin_riscv_th_mumax_w_mv_i(4, 6, 5, i32, i32, 3);
    (void)__builtin_riscv_th_mmin_w_mv_i(4, 6, 5, i32, i32, 3);
    (void)__builtin_riscv_th_mumin_w_mv_i(4, 6, 5, i32, i32, 3);
    (void)__builtin_riscv_th_msrl_w_mv_i(4, 6, 5, i32, i32, 3);
    (void)__builtin_riscv_th_msll_w_mv_i(4, 6, 5, i32, i32, 3);
    (void)__builtin_riscv_th_msra_w_mv_i(4, 6, 5, i32, i32, 3);
}

// ============================================================================
// 16. FP EW arithmetic — all 5 ops × 3 sizes, both MM and MVI
// ============================================================================

// CHECK-LABEL: test_fp_ew_mm:
// CHECK: th.mfadd.h.mm acc0, acc2, acc1
// CHECK: th.mfadd.s.mm acc0, acc2, acc1
// CHECK: th.mfadd.d.mm acc0, acc2, acc1
// CHECK: th.mfsub.h.mm acc0, acc2, acc1
// CHECK: th.mfsub.s.mm acc0, acc2, acc1
// CHECK: th.mfsub.d.mm acc0, acc2, acc1
// CHECK: th.mfmul.h.mm acc0, acc2, acc1
// CHECK: th.mfmul.s.mm acc0, acc2, acc1
// CHECK: th.mfmul.d.mm acc0, acc2, acc1
// CHECK: th.mfmax.h.mm acc0, acc2, acc1
// CHECK: th.mfmax.s.mm acc0, acc2, acc1
// CHECK: th.mfmax.d.mm acc0, acc2, acc1
// CHECK: th.mfmin.h.mm acc0, acc2, acc1
// CHECK: th.mfmin.s.mm acc0, acc2, acc1
// CHECK: th.mfmin.d.mm acc0, acc2, acc1
void test_fp_ew_mm(void) {
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

// CHECK-LABEL: test_fp_ew_mvi:
// CHECK: th.mfadd.h.mv.i acc0, acc2, acc1, 0
// CHECK: th.mfadd.s.mv.i acc0, acc2, acc1, 1
// CHECK: th.mfadd.d.mv.i acc0, acc2, acc1, 2
// CHECK: th.mfsub.h.mv.i acc0, acc2, acc1, 3
// CHECK: th.mfsub.s.mv.i acc0, acc2, acc1, 4
// CHECK: th.mfsub.d.mv.i acc0, acc2, acc1, 5
// CHECK: th.mfmul.h.mv.i acc0, acc2, acc1, 6
// CHECK: th.mfmul.s.mv.i acc0, acc2, acc1, 7
// CHECK: th.mfmul.d.mv.i acc0, acc2, acc1, 0
// CHECK: th.mfmax.h.mv.i acc0, acc2, acc1, 1
// CHECK: th.mfmax.s.mv.i acc0, acc2, acc1, 2
// CHECK: th.mfmax.d.mv.i acc0, acc2, acc1, 3
// CHECK: th.mfmin.h.mv.i acc0, acc2, acc1, 4
// CHECK: th.mfmin.s.mv.i acc0, acc2, acc1, 5
// CHECK: th.mfmin.d.mv.i acc0, acc2, acc1, 6
void test_fp_ew_mvi(void) {
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
// 17. Multi-accumulator matmul — proving the ISel flexibility
// ============================================================================

// CHECK-LABEL: test_multi_acc_matmul:
// CHECK: th.mfmacc.s acc0, tr1, tr0
// CHECK: th.mfmacc.s acc1, tr3, tr2
// CHECK: th.mfmacc.s acc2, tr1, tr0
// CHECK: th.mfmacc.s acc3, tr3, tr2
void test_multi_acc_matmul(void) {
    __rvm_float32_t f32 = __builtin_riscv_th_mundef_f32();
    // Four independent matmul accumulations targeting different acc registers
    (void)__builtin_riscv_th_mfmacc_s(4, 1, 0, f32, f32, f32); // acc0 += tr1 * tr0
    (void)__builtin_riscv_th_mfmacc_s(5, 3, 2, f32, f32, f32); // acc1 += tr3 * tr2
    (void)__builtin_riscv_th_mfmacc_s(6, 1, 0, f32, f32, f32); // acc2 += tr1 * tr0
    (void)__builtin_riscv_th_mfmacc_s(7, 3, 2, f32, f32, f32); // acc3 += tr3 * tr2
}

// ============================================================================
// 18. Mundef builtins — all 22 types
// ============================================================================

// CHECK-LABEL: test_mundef_all:
// CHECK: ret
void test_mundef_all(void) {
    // These should all compile to poison values — no instructions emitted
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
