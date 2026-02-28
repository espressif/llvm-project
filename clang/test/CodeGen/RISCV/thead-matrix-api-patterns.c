// NOTE: XTHeadMatrix API-level pattern tests. Exercises high-level API
// functions from <thead_matrix.h> to verify correct register assignments,
// dimension parameter passing, and end-to-end correctness.
//
// RUN: %clang_cc1 -O2 -triple riscv64 -target-feature +experimental-xtheadmatrix -S -o - %s \
// RUN:   | FileCheck %s

#include <thead_matrix.h>

// ============================================================================
// Pattern 1: INT8 GEMM — full pipeline (load A/B, zero, matmul, store C)
// Verifies: mzero→acc0, mla→tr0, mlb→tr1, mmacc→acc0+tr1*tr0, msc→acc0
// ============================================================================

// CHECK-LABEL: pattern_int8_gemm:
// CHECK-DAG: th.msettilem
// CHECK-DAG: th.msettilek
// CHECK: th.mlae8 tr0
// CHECK-DAG: th.msettilen
// CHECK: th.mlbe8 tr1
// CHECK: th.mzero acc0
// CHECK: th.mmacc.w.b acc0, tr1, tr0
// CHECK: th.msce32 acc0
void pattern_int8_gemm(const int8_t *A, long as, const int8_t *B, long bs,
                        int32_t *C, long cs, mrow_t M, mrow_t K, mcol_t N) {
    mint8_t a = __riscv_th_mld_a_i8(A, as, M, K);
    mint8_t b = __riscv_th_mld_b_i8(B, bs, K, N);
    mint32_t c = __riscv_th_mzero_i32();
    c = __riscv_th_mmaqa_ss_w_b(c, a, b, M, K, N);
    __riscv_th_mst_c_i32(C, cs, c, M, N);
}

// ============================================================================
// Pattern 2: FP16 widening GEMM (FP16 inputs → FP32 accumulation)
// Verifies: widening matmul uses correct registers and types
// ============================================================================

// CHECK-LABEL: pattern_fp16_widen_gemm:
// CHECK: th.mlae16 tr0
// CHECK: th.mlbe16 tr1
// CHECK: th.mlce32 acc0
// CHECK: th.mfmacc.s.h acc0, tr1, tr0
// CHECK: th.msce32 acc0
void pattern_fp16_widen_gemm(const void *A, long as, const void *B, long bs,
                              float *C, long cs,
                              mrow_t M, mrow_t K, mcol_t N) {
    mfloat16_t a = __riscv_th_mld_a_f16(A, as, M, K);
    mfloat16_t b = __riscv_th_mld_b_f16(B, bs, K, N);
    mfloat32_t c = __riscv_th_mld_c_f32(C, cs, M, N);
    c = __riscv_th_fwmmacc_s_h(c, a, b, M, K, N);
    __riscv_th_mst_c_f32(C, cs, c, M, N);
}

// ============================================================================
// Pattern 3: Unsigned INT8 GEMM
// Verifies: unsigned type variants work correctly
// ============================================================================

// CHECK-LABEL: pattern_uint8_gemm:
// CHECK: th.mlae8 tr0
// CHECK: th.mlbe8 tr1
// CHECK: th.mzero acc0
// CHECK: th.mmaccu.w.b acc0, tr1, tr0
// CHECK: th.msce32 acc0
void pattern_uint8_gemm(const uint8_t *A, long as, const uint8_t *B, long bs,
                          uint32_t *C, long cs,
                          mrow_t M, mrow_t K, mcol_t N) {
    muint8_t a = __riscv_th_mld_a_u8(A, as, M, K);
    muint8_t b = __riscv_th_mld_b_u8(B, bs, K, N);
    muint32_t c = __riscv_th_mzero_u32();
    c = __riscv_th_mmaqa_uu_w_b(c, a, b, M, K, N);
    __riscv_th_mst_c_u32(C, cs, c, M, N);
}

// ============================================================================
// Pattern 4: Transposed load pattern (A^T, B^T)
// Verifies: transposed loads use correct tile registers (tr2, tr3)
// ============================================================================

// CHECK-LABEL: pattern_transposed_loads:
// CHECK: th.mlate32 tr2
// CHECK: th.mlbte32 tr3
void pattern_transposed_loads(const void *A, long as, const void *B, long bs,
                               mrow_t M, mrow_t K, mcol_t N) {
    mfloat32_t at = __riscv_th_mld_at_f32(A, as, M, K);
    mfloat32_t bt = __riscv_th_mld_bt_f32(B, bs, K, N);
    (void)at; (void)bt;
}

// ============================================================================
// Pattern 5: C-transposed load/store
// Verifies: C^T uses acc1 register
// ============================================================================

// CHECK-LABEL: pattern_ct_load_store:
// CHECK: th.mlcte64 acc1
// CHECK: th.mscte64 acc1
void pattern_ct_load_store(const void *C_in, long cs_in,
                            void *C_out, long cs_out,
                            mrow_t M, mcol_t N) {
    mfloat64_t ct = __riscv_th_mld_ct_f64(C_in, cs_in, M, N);
    __riscv_th_mst_ct_f64(C_out, cs_out, ct, M, N);
}

// ============================================================================
// Pattern 6: EW integer arithmetic pipeline
// Verifies: EW ops use acc0/acc1/acc2 registers correctly
// ============================================================================

// CHECK-LABEL: pattern_ew_int_arith:
// CHECK: th.madd.w.mm acc0, acc2, acc1
// CHECK: th.mmul.w.mm acc0, acc2, acc1
// CHECK: th.msra.w.mm acc0, acc2, acc1
void pattern_ew_int_arith(void) {
    mint32_t d = __riscv_th_mundefined_i32();
    mint32_t s1 = __riscv_th_mundefined_i32();
    mint32_t s2 = __riscv_th_mundefined_i32();
    d = __riscv_th_madd_w_mm(d, s1, s2);
    d = __riscv_th_mmul_w_mm(d, s1, s2);
    d = __riscv_th_msra_w_mm(d, s1, s2);
    (void)d;
}

// ============================================================================
// Pattern 7: FP EW arithmetic pipeline
// Verifies: FP EW uses acc registers for all three precisions
// ============================================================================

// CHECK-LABEL: pattern_fp_ew_arith:
// CHECK: th.mfadd.h.mm acc0, acc2, acc1
// CHECK: th.mfsub.s.mm acc0, acc2, acc1
// CHECK: th.mfmul.d.mm acc0, acc2, acc1
// CHECK: th.mfmax.h.mm acc0, acc2, acc1
// CHECK: th.mfmin.s.mm acc0, acc2, acc1
void pattern_fp_ew_arith(void) {
    mfloat16_t f16 = __riscv_th_mundefined_f16();
    mfloat32_t f32 = __riscv_th_mundefined_f32();
    mfloat64_t f64 = __riscv_th_mundefined_f64();
    f16 = __riscv_th_mfadd_h_mm(f16, f16, f16);
    f32 = __riscv_th_mfsub_s_mm(f32, f32, f32);
    f64 = __riscv_th_mfmul_d_mm(f64, f64, f64);
    f16 = __riscv_th_mfmax_h_mm(f16, f16, f16);
    f32 = __riscv_th_mfmin_s_mm(f32, f32, f32);
    (void)f16; (void)f32; (void)f64;
}

// ============================================================================
// Pattern 8: Conversion pipeline (FP16 → FP32 → clip → INT8)
// Verifies: conversions chain correctly with acc registers
// ============================================================================

// CHECK-LABEL: pattern_conversion_chain:
// CHECK: th.mfcvtl.s.h acc0, acc1
// CHECK: th.mfscvt.w.s acc0, acc1
void pattern_conversion_chain(void) {
    mfloat16_t f16 = __riscv_th_mundefined_f16();
    // FP16 → FP32
    mfloat32_t f32 = __riscv_th_mfcvtl_s_h(f16);
    // FP32 → INT32
    mint32_t i32 = __riscv_th_mfscvt_w_s(f32);
    (void)i32;
}

// ============================================================================
// Pattern 9: N4clip pipeline (INT32 → INT8 via fixed-point clip)
// Verifies: N4clip signed/unsigned return correct types
// ============================================================================

// CHECK-LABEL: pattern_n4clip:
// CHECK: th.mn4clipl.w.mm acc0, acc2, acc1
// CHECK: th.mn4cliplu.w.mm acc0, acc2, acc1
void pattern_n4clip(void) {
    mint32_t src1 = __riscv_th_mundefined_i32();
    mint32_t src2 = __riscv_th_mundefined_i32();
    mint8_t clipped_s = __riscv_th_mn4clipl_w_mm(src1, src2);
    muint8_t clipped_u = __riscv_th_mn4cliplu_w_mm(src1, src2);
    (void)clipped_s; (void)clipped_u;
}

// ============================================================================
// Pattern 10: Data movement (mmov, mdup, pack)
// Verifies: move/dup/pack use correct register indices
// ============================================================================

// CHECK-LABEL: pattern_data_movement:
// CHECK: th.mmov.mm tr0, tr1
// CHECK: th.mmovw.x.m {{.*}}, tr0
// CHECK: th.mmovw.m.x tr0
// CHECK: th.mdupw.m.x tr0
// CHECK: th.mpack tr0, tr2, tr1
// CHECK: th.mpackhl tr0, tr2, tr1
void pattern_data_movement(void) {
    // mmov.mm
    __riscv_th_mmov_mm();
    // mmov.x.m
    mint32_t src = __riscv_th_mundefined_i32();
    size_t val = __riscv_th_mmov_x_m_w(src, 0);
    // mmov.m.x
    mint32_t dst = __riscv_th_mmov_m_x_w(src, 42, 0);
    // mdup.m.x
    mint32_t dup = __riscv_th_mdup_m_x_w(42);
    // pack
    __riscv_th_mpack();
    __riscv_th_mpackhl();
    (void)val; (void)dst; (void)dup;
}

// ============================================================================
// Pattern 11: Slide macros
// Verifies: slides use correct registers and immediates
// ============================================================================

// CHECK-LABEL: pattern_slides:
// CHECK: th.mrslidedown tr0, tr1, 3
// CHECK: th.mrslideup tr0, tr1, 2
// CHECK: th.mcslidedown.w tr0, tr1, 1
// CHECK: th.mcslideup.d tr0, tr1, 4
void pattern_slides(void) {
    __riscv_th_mrslidedown(3);
    __riscv_th_mrslideup(2);
    __riscv_th_mcslidedown_w(1);
    __riscv_th_mcslideup_d(4);
}

// ============================================================================
// Pattern 12: Broadcast macros
// Verifies: broadcasts use correct registers and immediates
// ============================================================================

// CHECK-LABEL: pattern_broadcasts:
// CHECK: th.mrbca.mv.i tr0, tr1, 5
// CHECK: th.mcbcab.mv.i tr0, tr1, 2
// CHECK: th.mcbcaw.mv.i tr0, tr1, 3
void pattern_broadcasts(void) {
    __riscv_th_mrbca(5);
    __riscv_th_mcbca_b(2);
    __riscv_th_mcbca_w(3);
}

// ============================================================================
// Pattern 13: Zero and undefined for all types
// Verifies: mzero targets acc0, mundef compiles cleanly
// ============================================================================

// CHECK-LABEL: pattern_zero_types:
// CHECK: th.mzero acc0
// CHECK: th.mzero acc0
// CHECK: th.mzero acc0
// CHECK: th.mzero acc0
void pattern_zero_types(void) {
    mint8_t z_i8 = __riscv_th_mzero_i8();
    mint32_t z_i32 = __riscv_th_mzero_i32();
    mfloat16_t z_f16 = __riscv_th_mzero_f16();
    mfloat32_t z_f32 = __riscv_th_mzero_f32();
    (void)z_i8; (void)z_i32; (void)z_f16; (void)z_f32;
}

// CHECK-LABEL: pattern_zero_pair_types:
// CHECK: th.mzero2r acc0
// CHECK: th.mzero2r acc0
// CHECK: th.mzero2r acc0
void pattern_zero_pair_types(void) {
    mint8x2_t z_i8x2 = __riscv_th_mzero_i8x2();
    mfloat32x2_t z_f32x2 = __riscv_th_mzero_f32x2();
    muint64x2_t z_u64x2 = __riscv_th_mzero_u64x2();
    (void)z_i8x2; (void)z_f32x2; (void)z_u64x2;
}

// ============================================================================
// Pattern 14: Reinterpret cast
// Verifies: reinterpret doesn't emit any matrix instructions
// ============================================================================

// CHECK-LABEL: pattern_reinterpret:
// CHECK-NOT: th.m
// CHECK: ret
void pattern_reinterpret(void) {
    mint32_t i32 = __riscv_th_mundefined_i32();
    mfloat32_t f32 = __riscv_th_mreinterpret_f32(i32);
    mint8_t i8 = __riscv_th_mreinterpret_i8(f32);
    (void)i8;
}

// ============================================================================
// Pattern 15: Tuple get/set
// Verifies: tuple operations don't emit matrix instructions
// ============================================================================

// CHECK-LABEL: pattern_tuple_ops:
// CHECK-NOT: th.m
// CHECK: ret
void pattern_tuple_ops(void) {
    mint32x2_t pair = __riscv_th_mundefined_i32x2();
    mint32_t elem = __riscv_th_mget_i32(pair, 0);
    pair = __riscv_th_mset_i32(pair, 1, elem);
    (void)pair;
}

// ============================================================================
// Pattern 16: CSR access
// Verifies: CSR reads and writes compile correctly
// ============================================================================

// CHECK-LABEL: pattern_csr_access:
// CHECK: csrr {{.*}}, th.xtlenb
// CHECK: csrr {{.*}}, th.xtrlenb
// CHECK: csrr {{.*}}, th.xmisa
void pattern_csr_access(void) {
    unsigned long tlenb = __riscv_th_xmlenb();
    unsigned long rlenb = __riscv_th_xrlenb();
    unsigned long xmisa = __riscv_th_xmsize();
    (void)tlenb; (void)rlenb; (void)xmisa;
}

// ============================================================================
// Pattern 17: Whole-register load/store
// Verifies: whole-register ops use TR0 by default
// ============================================================================

// CHECK-LABEL: pattern_whole_reg:
// CHECK: th.mlme32 tr0
// CHECK: th.msme32 tr0
void pattern_whole_reg(const void *in, void *out) {
    mint32_t val = __riscv_th_mld_whole_i32(in);
    __riscv_th_mst_whole_i32(out, val);
}

// ============================================================================
// Pattern 18: FP8 matmul pipeline (untyped)
// Verifies: FP8 widening matmul compiles correctly
// ============================================================================

// CHECK-LABEL: pattern_fp8_matmul:
// CHECK: th.mlae8 tr0
// CHECK: th.mlbe8 tr1
// CHECK: th.mzero acc0
// CHECK: th.mfmacc.s.e4 acc0, tr1, tr0
// CHECK: th.msce32 acc0
void pattern_fp8_matmul(const void *A, long as, const void *B, long bs,
                          float *C, long cs,
                          mrow_t M, mrow_t K, mcol_t N) {
    muint8_t a = __riscv_th_mld_a_u8(A, as, M, K);
    muint8_t b = __riscv_th_mld_b_u8(B, bs, K, N);
    mfloat32_t c = __riscv_th_mzero_f32();
    c = __riscv_th_fwmmacc_s_e4(c, a, b, M, K, N);
    __riscv_th_mst_c_f32(C, cs, c, M, N);
}

// ============================================================================
// Pattern 19: All store role variants for one type
// Verifies: store A→tr0, B→tr1, C→acc0, AT→tr2, BT→tr3, CT→acc1, whole→tr0
// ============================================================================

// CHECK-LABEL: pattern_all_stores:
// CHECK: th.msae32 tr0
// CHECK: th.msbe32 tr1
// CHECK: th.msce32 acc0
// CHECK: th.msate32 tr2
// CHECK: th.msbte32 tr3
// CHECK: th.mscte32 acc1
// CHECK: th.msme32 tr0
void pattern_all_stores(void *p, long s, mrow_t M, mrow_t K, mcol_t N) {
    mfloat32_t val = __riscv_th_mundefined_f32();
    __riscv_th_mst_a_f32(p, s, val, M, K);
    __riscv_th_mst_b_f32(p, s, val, K, N);
    __riscv_th_mst_c_f32(p, s, val, M, N);
    __riscv_th_mst_at_f32(p, s, val, M, K);
    __riscv_th_mst_bt_f32(p, s, val, K, N);
    __riscv_th_mst_ct_f32(p, s, val, M, N);
    __riscv_th_mst_whole_f32(p, val);
}

// ============================================================================
// Pattern 20: INT16→INT64 widening matmul
// Verifies: INT16 matmul uses correct types and registers
// ============================================================================

// CHECK-LABEL: pattern_int16_gemm:
// CHECK: th.mlae16 tr0
// CHECK: th.mlbe16 tr1
// CHECK: th.mzero acc0
// CHECK: th.mmacc.d.h acc0, tr1, tr0
// CHECK: th.msce64 acc0
void pattern_int16_gemm(const int16_t *A, long as, const int16_t *B, long bs,
                          int64_t *C, long cs,
                          mrow_t M, mrow_t K, mcol_t N) {
    mint16_t a = __riscv_th_mld_a_i16(A, as, M, K);
    mint16_t b = __riscv_th_mld_b_i16(B, bs, K, N);
    mint64_t c = __riscv_th_mzero_i64();
    c = __riscv_th_mmaqa_ss_d_h(c, a, b, M, K, N);
    __riscv_th_mst_c_i64(C, cs, c, M, N);
}

// ============================================================================
// Pattern 21: Partial (int4) matmul
// Verifies: partial matmul uses correct instruction
// ============================================================================

// CHECK-LABEL: pattern_partial_matmul:
// CHECK: th.pmmacc.w.b acc0, tr1, tr0
void pattern_partial_matmul(void) {
    mint8_t a = __riscv_th_mundefined_i8();
    mint8_t b = __riscv_th_mundefined_i8();
    mint32_t c = __riscv_th_mzero_i32();
    c = __riscv_th_pmmaqa_ss_w_b(c, a, b, 4, 4, 4);
    (void)c;
}

// ============================================================================
// Pattern 22: Bypass INT matmul
// Verifies: bypass matmul uses correct instruction
// ============================================================================

// CHECK-LABEL: pattern_bypass_matmul:
// CHECK: th.mmacc.w.bp acc0, tr1, tr0
// CHECK: th.mmaccu.w.bp acc0, tr1, tr0
void pattern_bypass_matmul(void) {
    mint8_t i8 = __riscv_th_mundefined_i8();
    muint8_t u8 = __riscv_th_mundefined_u8();
    mint32_t c_s = __riscv_th_mzero_i32();
    muint32_t c_u = __riscv_th_mzero_u32();
    c_s = __riscv_th_mmaqa_bp_ss(c_s, i8, i8, 4, 4, 4);
    c_u = __riscv_th_mmaqa_bp_uu(c_u, u8, u8, 4, 4, 4);
    (void)c_s; (void)c_u;
}

// ============================================================================
// Pattern 23: Packed int4 conversions
// Verifies: packed conversion API works correctly
// ============================================================================

// CHECK-LABEL: pattern_packed_conv:
// CHECK: th.mucvtl.b.p acc0, acc1
// CHECK: th.mscvth.b.p acc0, acc1
void pattern_packed_conv(void) {
    muint8_t src = __riscv_th_mundefined_u8();
    muint8_t u_out = __riscv_th_mucvtl_b_p(src);
    mint8_t s_out = __riscv_th_mscvth_b_p(src);
    (void)u_out; (void)s_out;
}

// ============================================================================
// Pattern 24: Float-int conversions round-trip
// Verifies: float→int→float conversion chain
// ============================================================================

// CHECK-LABEL: pattern_float_int_roundtrip:
// CHECK: th.mufcvtl.h.b acc0, acc1
// CHECK: th.mfucvtl.b.h acc0, acc1
// CHECK: th.msfcvt.s.w acc0, acc1
// CHECK: th.mfscvt.w.s acc0, acc1
void pattern_float_int_roundtrip(void) {
    muint8_t u8 = __riscv_th_mundefined_u8();
    mfloat16_t f16 = __riscv_th_mufcvtl_h_b(u8);
    muint8_t u8_back = __riscv_th_mfucvtl_b_h(f16);
    (void)u8_back;

    mint32_t i32 = __riscv_th_mundefined_i32();
    mfloat32_t f32 = __riscv_th_msfcvt_s_w(i32);
    mint32_t i32_back = __riscv_th_mfscvt_w_s(f32);
    (void)i32_back;
}

// ============================================================================
// Pattern 25: FP64 GEMM
// Verifies: 64-bit FP matmul works end-to-end
// ============================================================================

// CHECK-LABEL: pattern_fp64_gemm:
// CHECK: th.mlae64 tr0
// CHECK: th.mlbe64 tr1
// CHECK: th.mzero acc0
// CHECK: th.mfmacc.d acc0, tr1, tr0
// CHECK: th.msce64 acc0
void pattern_fp64_gemm(const double *A, long as, const double *B, long bs,
                         double *C, long cs,
                         mrow_t M, mrow_t K, mcol_t N) {
    mfloat64_t a = __riscv_th_mld_a_f64(A, as, M, K);
    mfloat64_t b = __riscv_th_mld_b_f64(B, bs, K, N);
    mfloat64_t c = __riscv_th_mzero_f64();
    c = __riscv_th_fmmacc_d(c, a, b, M, K, N);
    __riscv_th_mst_c_f64(C, cs, c, M, N);
}

// ============================================================================
// Pattern 26: Configuration functions
// Verifies: config API calls set correct CSRs
// ============================================================================

// CHECK-LABEL: pattern_config:
// CHECK: th.msettilem
// CHECK: th.msettilen
// CHECK: th.msettilek
// CHECK: th.mrelease
void pattern_config(mrow_t M, mrow_t K, mcol_t N) {
    __riscv_th_msetmrow_m(M);
    __riscv_th_msetmrow_n(N);
    __riscv_th_msetmcol_e32(K);
    __riscv_th_mrelease();
}
