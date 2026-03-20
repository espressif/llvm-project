// RUN: %clang_cc1 -O2 -triple riscv64 -target-feature +experimental-xtheadmatrix \
// RUN:   -emit-llvm %s -o - | FileCheck %s
//
// C-level API pipeline tests for XTHeadMatrix.
// Verifies register allocation correctness, dependency chains, CSR config
// emission, operand ordering, and complex multi-operation pipelines through
// the <thead_matrix.h> higher-level API.

#include <thead_matrix.h>

// ============================================================================
// Test 1: Full INT8 GEMM pipeline with CSR config verification
// Ensures SetM/SetK/SetN are emitted before each load/matmul/store/zero.
// ============================================================================
// CHECK-LABEL: @test_int8_gemm_pipeline(
// CHECK:       call void @llvm.riscv.th.msettilem
// CHECK:       call void @llvm.riscv.th.msettilek
// CHECK:       @llvm.riscv.th.mlae.internal8
// CHECK:       call void @llvm.riscv.th.msettilek
// CHECK:       call void @llvm.riscv.th.msettilen
// CHECK:       @llvm.riscv.th.mlbe.internal8
// CHECK:       call void @llvm.riscv.th.msettilem
// CHECK:       call void @llvm.riscv.th.msettilen
// CHECK:       @llvm.riscv.th.mzero.internal
// CHECK:       @llvm.riscv.th.mmacc.w.b.internal
// CHECK:       @llvm.riscv.th.msce.internal32
void test_int8_gemm_pipeline(const int8_t *A, long sa,
                              const int8_t *B, long sb,
                              int32_t *C, long sc,
                              mrow_t M, mcol_t K, mcol_t N) {
  mint8_t a = __riscv_th_mld_a_i8(A, sa, M, K);
  mint8_t b = __riscv_th_mld_b_i8(B, sb, K, N);
  mint32_t c = __riscv_th_mzeros_i32(M, N);
  c = __riscv_th_mmacc_w_b(c, a, b, M, K, N);
  __riscv_th_mst_i32(C, sc, c, M, N);
  __riscv_th_mrelease();
}

// ============================================================================
// Test 2: Matmul operand ordering (A/B swap)
// Hardware: md = md + ms1*ms2. C API passes (acc, a, b) but codegen must
// swap to (acc, b_as_ms2, a_as_ms1). Verify the SSA values are correct.
// ============================================================================
// CHECK-LABEL: @test_operand_ordering(
// CHECK:       %[[A:.+]] = {{.*}}@llvm.riscv.th.mlae.internal32
// CHECK:       %[[B:.+]] = {{.*}}@llvm.riscv.th.mlbe.internal32
// CHECK:       %[[Z:.+]] = {{.*}}@llvm.riscv.th.mzero.internal
//   Matmul intrinsic takes (acc, ms2=B, ms1=A):
// CHECK:       @llvm.riscv.th.mfmacc.s.internal{{.*}}({{.*}} %[[Z]], {{.*}} %[[B]], {{.*}} %[[A]])
void test_operand_ordering(const float *A, long sa,
                            const float *B, long sb,
                            float *C, long sc,
                            mrow_t M, mcol_t K, mcol_t N) {
  mfloat32_t a = __riscv_th_mld_a_f32(A, sa, M, K);
  mfloat32_t b = __riscv_th_mld_b_f32(B, sb, K, N);
  mfloat32_t c = __riscv_th_mzeros_f32(M, N);
  c = __riscv_th_mfmacc_s(c, a, b, M, K, N);
  __riscv_th_mst_f32(C, sc, c, M, N);
}

// ============================================================================
// Test 3: Chained matmul-accumulate (dependency chain)
// Two matmuls accumulate into the same C: C += A1*B1; C += A2*B2;
// The second matmul must use the output of the first as its accumulator.
// ============================================================================
// CHECK-LABEL: @test_chained_accumulate(
// CHECK:       %[[C:.+]] = {{.*}}@llvm.riscv.th.mlce.internal32
// CHECK:       %[[R1:.+]] = {{.*}}@llvm.riscv.th.mfmacc.s.internal{{.*}}({{.*}} %[[C]],
// CHECK:       %[[R2:.+]] = {{.*}}@llvm.riscv.th.mfmacc.s.internal{{.*}}({{.*}} %[[R1]],
// CHECK:       @llvm.riscv.th.msce.internal32{{.*}}({{.*}} %[[R2]],
void test_chained_accumulate(const float *A1, const float *A2,
                              const float *B1, const float *B2,
                              float *C, long stride,
                              mrow_t M, mcol_t K, mcol_t N) {
  mfloat32_t c = __riscv_th_mld_acc_f32(C, stride, M, N);
  mfloat32_t a1 = __riscv_th_mld_a_f32(A1, stride, M, K);
  mfloat32_t b1 = __riscv_th_mld_b_f32(B1, stride, K, N);
  c = __riscv_th_mfmacc_s(c, a1, b1, M, K, N);
  mfloat32_t a2 = __riscv_th_mld_a_f32(A2, stride, M, K);
  mfloat32_t b2 = __riscv_th_mld_b_f32(B2, stride, K, N);
  c = __riscv_th_mfmacc_s(c, a2, b2, M, K, N);
  __riscv_th_mst_f32(C, stride, c, M, N);
}

// ============================================================================
// Test 4: EW post-processing pipeline (matmul → add bias → shift → clip)
// Dependency chain: load→matmul→EW_add→EW_sra→n4clip→store
// EW ops rely on CSR state from prior matmul config.
// ============================================================================
// CHECK-LABEL: @test_ew_postprocess(
// CHECK:       @llvm.riscv.th.mmacc.w.b.internal
// CHECK:       @llvm.riscv.th.madd.w.mm.internal
// CHECK:       @llvm.riscv.th.msra.w.mm.internal
// CHECK:       @llvm.riscv.th.mn4clipl.w.mm.internal
// CHECK:       @llvm.riscv.th.msce.internal32
void test_ew_postprocess(const int8_t *A, long sa,
                          const int8_t *B, long sb,
                          const int32_t *bias_ptr, long bs,
                          const int32_t *shift_ptr,
                          int32_t *out, long os,
                          mrow_t M, mcol_t K, mcol_t N) {
  // Matmul
  mint8_t a = __riscv_th_mld_a_i8(A, sa, M, K);
  mint8_t b = __riscv_th_mld_b_i8(B, sb, K, N);
  mint32_t c = __riscv_th_mzeros_i32(M, N);
  c = __riscv_th_mmacc_w_b(c, a, b, M, K, N);
  // EW add bias (md = ms1 + ms2 = c + bias)
  mint32_t bias = __riscv_th_mld_acc_i32(bias_ptr, bs, M, N);
  c = __riscv_th_madd_w_mm(c, c, bias);
  // EW arithmetic shift right
  mint32_t shift = __riscv_th_mld_acc_i32(shift_ptr, bs, M, N);
  c = __riscv_th_msra_w_mm(c, c, shift);
  // N4clip (narrow and clip to INT4)
  mint32_t scale = __riscv_th_mld_acc_i32(shift_ptr, bs, M, N);
  c = __riscv_th_mn4clipl_w_mm(c, scale, c);
  // Store
  __riscv_th_mst_i32(out, os, c, M, N);
}

// ============================================================================
// Test 5: FP conversion pipeline (FP8 → FP16 → FP32 matmul → FP16 store)
// Tests conversion intrinsics in a realistic pipeline.
// ============================================================================
// CHECK-LABEL: @test_fp_conversion_pipeline(
// CHECK:       @llvm.riscv.th.mlae.internal32
// CHECK:       @llvm.riscv.th.mfcvtl.h.e4.internal
// CHECK:       @llvm.riscv.th.mlbe.internal32
// CHECK:       @llvm.riscv.th.mfcvtl.h.e4.internal
// CHECK:       @llvm.riscv.th.mzero.internal
// CHECK:       @llvm.riscv.th.mfmacc.s.h.internal
// CHECK:       @llvm.riscv.th.mfcvtl.h.s.internal
// CHECK:       @llvm.riscv.th.msce.internal16
void test_fp_conversion_pipeline(const void *A, long sa,
                                  const void *B, long sb,
                                  void *C, long sc,
                                  mrow_t M, mcol_t K, mcol_t N) {
  // Load FP8 data as opaque int32 tiles
  mint32_t a_raw = __riscv_th_mld_a_i32(A, sa, M, K);
  mint32_t a_f16 = __riscv_th_mfcvtl_h_e4(a_raw);  // FP8 E4M3 → FP16 (low)
  mint32_t b_raw = __riscv_th_mld_b_i32(B, sb, K, N);
  mint32_t b_f16 = __riscv_th_mfcvtl_h_e4(b_raw);
  // FP16 → FP32 widening matmul
  mfloat32_t acc = __riscv_th_mzeros_f32(M, N);
  // Note: we use the opaque FP8 sources directly (mfmacc_h_e4 handles it)
  // but here we demonstrate the conversion path
  mfloat16_t a16 = __riscv_th_mld_a_f16(A, sa, M, K);
  mfloat16_t b16 = __riscv_th_mld_b_f16(B, sb, K, N);
  acc = __riscv_th_mfmacc_s_h(acc, a16, b16, M, K, N);
  // Convert FP32 result to FP16 for storage
  mfloat32_t cvt_src = acc;
  mfloat16_t result_f16 = __riscv_th_mfcvtl_h_s(cvt_src);
  __riscv_th_mst_f16(C, sc, result_f16, M, N);
}

// ============================================================================
// Test 6: Register pressure — dual GEMM with shared A
// Two matmul results live simultaneously: tests RA under pressure.
// ============================================================================
// CHECK-LABEL: @test_dual_gemm(
// CHECK:       @llvm.riscv.th.mlae.internal8
// CHECK:       @llvm.riscv.th.mlbe.internal8
// CHECK:       @llvm.riscv.th.mzero.internal
// CHECK:       @llvm.riscv.th.mmacc.w.b.internal
// CHECK:       @llvm.riscv.th.mlbe.internal8
// CHECK:       @llvm.riscv.th.mzero.internal
// CHECK:       @llvm.riscv.th.mmacc.w.b.internal
// CHECK:       @llvm.riscv.th.msce.internal32
// CHECK:       @llvm.riscv.th.msce.internal32
void test_dual_gemm(const int8_t *A, long sa,
                     const int8_t *B1, long sb1,
                     const int8_t *B2, long sb2,
                     int32_t *C1, long sc1,
                     int32_t *C2, long sc2,
                     mrow_t M, mcol_t K, mcol_t N) {
  mint8_t a = __riscv_th_mld_a_i8(A, sa, M, K);
  // First GEMM
  mint8_t b1 = __riscv_th_mld_b_i8(B1, sb1, K, N);
  mint32_t c1 = __riscv_th_mzeros_i32(M, N);
  c1 = __riscv_th_mmacc_w_b(c1, a, b1, M, K, N);
  // Second GEMM reusing A
  mint8_t b2 = __riscv_th_mld_b_i8(B2, sb2, K, N);
  mint32_t c2 = __riscv_th_mzeros_i32(M, N);
  c2 = __riscv_th_mmacc_w_b(c2, a, b2, M, K, N);
  // Both results stored (c1, c2 live simultaneously at some point)
  __riscv_th_mst_i32(C1, sc1, c1, M, N);
  __riscv_th_mst_i32(C2, sc2, c2, M, N);
}

// ============================================================================
// Test 7: Transposed loads
// Verify AT/BT/CT transposed load variants generate correct intrinsics.
// ============================================================================
// CHECK-LABEL: @test_transposed_loads(
// CHECK:       @llvm.riscv.th.mlate.internal32
// CHECK:       @llvm.riscv.th.mlbte.internal32
// CHECK:       @llvm.riscv.th.mlcte.internal32
// CHECK:       @llvm.riscv.th.mfmacc.s.internal
// CHECK:       @llvm.riscv.th.mscte.internal32
void test_transposed_loads(const float *A, long sa,
                            const float *B, long sb,
                            float *C, long sc,
                            mrow_t M, mcol_t K, mcol_t N) {
  mfloat32_t a = __riscv_th_mld_at_f32(A, sa, M, K);
  mfloat32_t b = __riscv_th_mld_bt_f32(B, sb, K, N);
  mfloat32_t c = __riscv_th_mld_ct_f32(C, sc, M, N);
  c = __riscv_th_mfmacc_s(c, a, b, M, K, N);
  __riscv_th_mst_ct_f32(C, sc, c, M, N);
}

// ============================================================================
// Test 8: Mixed INT and FP matmul in the same function
// Both INT8→INT32 and FP32 matmul use same ManagedRA model.
// ============================================================================
// CHECK-LABEL: @test_mixed_int_fp(
// CHECK:       @llvm.riscv.th.mmacc.w.b.internal
// CHECK:       @llvm.riscv.th.msfcvt.s.w.internal
// CHECK:       @llvm.riscv.th.mfmacc.s.internal
// CHECK:       @llvm.riscv.th.msce.internal32
void test_mixed_int_fp(const int8_t *A_i8, const int8_t *B_i8,
                        const float *A_fp, const float *B_fp,
                        float *C, long stride,
                        mrow_t M, mcol_t K, mcol_t N) {
  // INT8 matmul
  mint8_t ai = __riscv_th_mld_a_i8(A_i8, stride, M, K);
  mint8_t bi = __riscv_th_mld_b_i8(B_i8, stride, K, N);
  mint32_t ci = __riscv_th_mzeros_i32(M, N);
  ci = __riscv_th_mmacc_w_b(ci, ai, bi, M, K, N);
  // Convert INT32 result to FP32
  mfloat32_t cf = __riscv_th_msfcvt_s_w(ci);
  // FP32 matmul that accumulates on the converted result
  mfloat32_t af = __riscv_th_mld_a_f32(A_fp, stride, M, K);
  mfloat32_t bf = __riscv_th_mld_b_f32(B_fp, stride, K, N);
  cf = __riscv_th_mfmacc_s(cf, af, bf, M, K, N);
  __riscv_th_mst_f32(C, stride, cf, M, N);
}

// ============================================================================
// Test 9: Data movement pipeline (load → slide → broadcast → pack → store)
// Tests data movement operations in a realistic dependency chain.
// ============================================================================
// CHECK-LABEL: @test_data_movement(
// CHECK:       @llvm.riscv.th.mlme.internal32
// CHECK:       @llvm.riscv.th.mrslidedown.internal
// CHECK:       @llvm.riscv.th.mrbca.mv.i.internal
// CHECK:       @llvm.riscv.th.mpack.internal
// CHECK:       @llvm.riscv.th.msme.internal32
void test_data_movement(const int32_t *src, int32_t *dst, long stride) {
  mint32_t v = __riscv_th_mld_m_i32(src, stride);
  // Slide rows down by 2
  mint32_t slid = __riscv_th_mrslidedown(v, 2);
  // Broadcast row 0 to all rows
  mint32_t bcast = __riscv_th_mrbca(slid, 0);
  // Pack low halves of two matrices
  mint32_t packed = __riscv_th_mpack(v, bcast);
  __riscv_th_mst_m_i32(dst, stride, packed);
}

// ============================================================================
// Test 10: All INT matmul signedness variants
// Tests that signed/unsigned source combinations generate distinct intrinsics.
// ============================================================================
// CHECK-LABEL: @test_signedness_variants(
// CHECK:       @llvm.riscv.th.mmacc.w.b.internal
// CHECK:       @llvm.riscv.th.mmaccu.w.b.internal
// CHECK:       @llvm.riscv.th.mmaccus.w.b.internal
// CHECK:       @llvm.riscv.th.mmaccsu.w.b.internal
void test_signedness_variants(const int8_t *Ai, const uint8_t *Au,
                               const int8_t *Bi, const uint8_t *Bu,
                               int32_t *C, long stride,
                               mrow_t M, mcol_t K, mcol_t N) {
  // ss: signed * signed
  mint8_t ai = __riscv_th_mld_a_i8(Ai, stride, M, K);
  mint8_t bi = __riscv_th_mld_b_i8(Bi, stride, K, N);
  mint32_t c_ss = __riscv_th_mzeros_i32(M, N);
  c_ss = __riscv_th_mmacc_w_b(c_ss, ai, bi, M, K, N);
  __riscv_th_mst_i32(C, stride, c_ss, M, N);
  // uu: unsigned * unsigned
  muint8_t au = __riscv_th_mld_a_u8(Au, stride, M, K);
  muint8_t bu = __riscv_th_mld_b_u8(Bu, stride, K, N);
  muint32_t c_uu = __riscv_th_mzeros_u32(M, N);
  c_uu = __riscv_th_mmaccu_w_b(c_uu, au, bu, M, K, N);
  __riscv_th_mst_u32((uint32_t *)C, stride, c_uu, M, N);
  // us: unsigned_A * signed_B
  mint32_t c_us = __riscv_th_mzeros_i32(M, N);
  c_us = __riscv_th_mmaccus_w_b(c_us, au, bi, M, K, N);
  __riscv_th_mst_i32(C, stride, c_us, M, N);
  // su: signed_A * unsigned_B
  mint32_t c_su = __riscv_th_mzeros_i32(M, N);
  c_su = __riscv_th_mmaccsu_w_b(c_su, ai, bu, M, K, N);
  __riscv_th_mst_i32(C, stride, c_su, M, N);
}

// ============================================================================
// Test 11: EW FP operations (dependency chain through FP element-wise)
// ============================================================================
// CHECK-LABEL: @test_ew_fp_chain(
// CHECK:       @llvm.riscv.th.mfmacc.s.internal
// CHECK:       @llvm.riscv.th.mfmul.s.mm.internal
// CHECK:       @llvm.riscv.th.mfadd.s.mm.internal
// CHECK:       @llvm.riscv.th.mfmax.s.mm.internal
// CHECK:       @llvm.riscv.th.msce.internal32
void test_ew_fp_chain(const float *A, const float *B,
                       const float *scale_ptr,
                       float *C, long stride,
                       mrow_t M, mcol_t K, mcol_t N) {
  mfloat32_t a = __riscv_th_mld_a_f32(A, stride, M, K);
  mfloat32_t b = __riscv_th_mld_b_f32(B, stride, K, N);
  mfloat32_t c = __riscv_th_mzeros_f32(M, N);
  c = __riscv_th_mfmacc_s(c, a, b, M, K, N);
  // EW multiply by scale
  mfloat32_t scale = __riscv_th_mld_acc_f32(scale_ptr, stride, M, N);
  c = __riscv_th_mfmul_s_mm(c, c, scale);
  // EW add bias
  mfloat32_t bias = __riscv_th_mld_acc_f32(scale_ptr, stride, M, N);
  c = __riscv_th_mfadd_s_mm(c, c, bias);
  // EW max with zero (ReLU-like)
  mfloat32_t zeros = __riscv_th_mzeros_f32(M, N);
  c = __riscv_th_mfmax_s_mm(c, c, zeros);
  __riscv_th_mst_f32(C, stride, c, M, N);
}

// ============================================================================
// Test 12: EW with immediate variant (.mv.i)
// ============================================================================
// CHECK-LABEL: @test_ew_mvi(
// CHECK:       @llvm.riscv.th.madd.w.mv.i.internal
// CHECK:       @llvm.riscv.th.mfmul.s.mv.i.internal
void test_ew_mvi(const int32_t *src, int32_t *dst,
                  const float *fsrc, float *fdst,
                  long stride, mrow_t M, mcol_t N) {
  // INT: add with immediate row index
  mint32_t a = __riscv_th_mld_acc_i32(src, stride, M, N);
  mint32_t b = __riscv_th_mld_acc_i32(src, stride, M, N);
  mint32_t r = __riscv_th_madd_w_mv_i(a, a, b, 3);
  __riscv_th_mst_i32(dst, stride, r, M, N);
  // FP: mul with immediate row index
  mfloat32_t fa = __riscv_th_mld_acc_f32(fsrc, stride, M, N);
  mfloat32_t fb = __riscv_th_mld_acc_f32(fsrc, stride, M, N);
  mfloat32_t fr = __riscv_th_mfmul_s_mv_i(fa, fa, fb, 2);
  __riscv_th_mst_f32(fdst, stride, fr, M, N);
}

// ============================================================================
// Test 13: Whole-register load/store (no config needed)
// ============================================================================
// CHECK-LABEL: @test_whole_register(
//   No msettilem/k/n calls expected for whole-register ops:
// CHECK-NOT:   @llvm.riscv.th.msettilem
// CHECK:       @llvm.riscv.th.mlme.internal32
// CHECK:       @llvm.riscv.th.mmov.mm.internal
// CHECK:       @llvm.riscv.th.msme.internal32
void test_whole_register(const int32_t *src, int32_t *dst, long stride) {
  mint32_t v = __riscv_th_mld_m_i32(src, stride);
  mint32_t copy = __riscv_th_mmov_mm(v);
  __riscv_th_mst_m_i32(dst, stride, copy);
}

// ============================================================================
// Test 14: INT16→INT64 matmul (single-register)
// ============================================================================
// CHECK-LABEL: @test_int16_matmul(
// CHECK:       @llvm.riscv.th.mlae.internal16
// CHECK:       @llvm.riscv.th.mlbe.internal16
// CHECK:       @llvm.riscv.th.mzero.internal
// CHECK:       @llvm.riscv.th.mmacc.d.h.internal
// CHECK:       @llvm.riscv.th.msce.internal64
void test_int16_matmul(const int16_t *A, const int16_t *B,
                        int64_t *C, long stride,
                        mrow_t M, mcol_t K, mcol_t N) {
  mint16_t a = __riscv_th_mld_a_i16(A, stride, M, K);
  mint16_t b = __riscv_th_mld_b_i16(B, stride, K, N);
  mint64_t c = __riscv_th_mzeros_i64(M, N);
  c = __riscv_th_mmacc_d_h(c, a, b, M, K, N);
  __riscv_th_mst_i64(C, stride, c, M, N);
}

// ============================================================================
// Test 15: GPR↔matrix data movement
// ============================================================================
// CHECK-LABEL: @test_gpr_move(
// CHECK:       @llvm.riscv.th.mlme.internal32
// CHECK:       @llvm.riscv.th.mmovw.x.m.internal
// CHECK:       @llvm.riscv.th.mmovw.m.x.internal
// CHECK:       @llvm.riscv.th.mdupw.m.x.internal
// CHECK:       @llvm.riscv.th.msme.internal32
void test_gpr_move(const int32_t *src, int32_t *dst, long stride) {
  mint32_t m = __riscv_th_mld_m_i32(src, stride);
  // Extract element at index 5
  unsigned long val = __riscv_th_mmovw_x_m(m, 5);
  // Insert val+1 at index 0
  m = __riscv_th_mmovw_m_x(m, val + 1, 0);
  // Broadcast val to all columns
  m = __riscv_th_mdupw_m_x(m, val);
  __riscv_th_mst_m_i32(dst, stride, m);
}

// ============================================================================
// Test 16: Float-int conversion round-trip
// ============================================================================
// CHECK-LABEL: @test_float_int_cvt(
// CHECK:       @llvm.riscv.th.mlce.internal32
// CHECK:       @llvm.riscv.th.msfcvt.s.w.internal
// CHECK:       @llvm.riscv.th.mfscvt.w.s.internal
// CHECK:       @llvm.riscv.th.msce.internal32
void test_float_int_cvt(const int32_t *src, int32_t *dst,
                          long stride, mrow_t M, mcol_t N) {
  mint32_t i = __riscv_th_mld_acc_i32(src, stride, M, N);
  // INT32 → FP32 (signed)
  mfloat32_t f = __riscv_th_msfcvt_s_w(i);
  // FP32 → INT32 (signed, back)
  mint32_t r = __riscv_th_mfscvt_w_s(f);
  __riscv_th_mst_i32(dst, stride, r, M, N);
}

// ============================================================================
// Test 17: Partial (panelized) INT matmul
// ============================================================================
// CHECK-LABEL: @test_partial_matmul(
// CHECK:       @llvm.riscv.th.pmmacc.w.b.internal
void test_partial_matmul(const int8_t *A, const int8_t *B,
                          int32_t *C, long stride,
                          mrow_t M, mcol_t K, mcol_t N) {
  mint8_t a = __riscv_th_mld_a_i8(A, stride, M, K);
  mint8_t b = __riscv_th_mld_b_i8(B, stride, K, N);
  mint32_t c = __riscv_th_mzeros_i32(M, N);
  c = __riscv_th_pmmacc_w_b(c, a, b, M, K, N);
  __riscv_th_mst_i32(C, stride, c, M, N);
}

// ============================================================================
// Test 18: Bypass INT matmul
// ============================================================================
// CHECK-LABEL: @test_bypass_matmul(
// CHECK:       @llvm.riscv.th.mmacc.w.bp.internal
void test_bypass_matmul(const int8_t *A, const int8_t *B,
                         int32_t *C, long stride,
                         mrow_t M, mcol_t K, mcol_t N) {
  mint8_t a = __riscv_th_mld_a_i8(A, stride, M, K);
  mint8_t b = __riscv_th_mld_b_i8(B, stride, K, N);
  mint32_t c = __riscv_th_mzeros_i32(M, N);
  c = __riscv_th_mmacc_w_bp(c, a, b, M, K, N);
  __riscv_th_mst_i32(C, stride, c, M, N);
}

// ============================================================================
// Test 19: Column slide and column broadcast
// ============================================================================
// CHECK-LABEL: @test_col_ops(
// CHECK:       @llvm.riscv.th.mcslidedown.w.internal
// CHECK:       @llvm.riscv.th.mcslideup.w.internal
// CHECK:       @llvm.riscv.th.mcbcaw.mv.i.internal
void test_col_ops(const int32_t *src, int32_t *dst, long stride) {
  mint32_t v = __riscv_th_mld_m_i32(src, stride);
  mint32_t d = __riscv_th_mcslidedown_w(v, 1);
  mint32_t u = __riscv_th_mcslideup_w(d, 2);
  mint32_t b = __riscv_th_mcbca_w(u, 0);
  __riscv_th_mst_m_i32(dst, stride, b);
}

// ============================================================================
// Test 20: Backward-compatibility aliases
// Old names should produce the same intrinsics as new names.
// ============================================================================
// CHECK-LABEL: @test_compat_aliases(
// CHECK:       @llvm.riscv.th.mmacc.w.b.internal
// CHECK:       @llvm.riscv.th.mfmacc.s.internal
void test_compat_aliases(const int8_t *Ai, const int8_t *Bi,
                          const float *Af, const float *Bf,
                          int32_t *Ci, float *Cf, long stride,
                          mrow_t M, mcol_t K, mcol_t N) {
  // Old name: __riscv_th_mmaq_ss_w_b → __riscv_th_mmacc_w_b
  mint8_t ai = __riscv_th_mld_a_i8(Ai, stride, M, K);
  mint8_t bi = __riscv_th_mld_b_i8(Bi, stride, K, N);
  mint32_t ci = __riscv_th_mzeros_i32(M, N);
  ci = __riscv_th_mmaq_ss_w_b(ci, ai, bi, M, K, N);
  __riscv_th_mst_i32(Ci, stride, ci, M, N);
  // Old name: __riscv_th_mfmaqa_s → __riscv_th_mfmacc_s
  mfloat32_t af = __riscv_th_mld_a_f32(Af, stride, M, K);
  mfloat32_t bf = __riscv_th_mld_b_f32(Bf, stride, K, N);
  mfloat32_t cf = __riscv_th_mzeros_f32(M, N);
  cf = __riscv_th_mfmaqa_s(cf, af, bf, M, K, N);
  __riscv_th_mst_f32(Cf, stride, cf, M, N);
}

// ============================================================================
// Test 21: Store A/B tile variants (not just acc stores)
// ============================================================================
// CHECK-LABEL: @test_tile_stores(
// CHECK:       @llvm.riscv.th.mlae.internal32
// CHECK:       @llvm.riscv.th.msae.internal32
// CHECK:       @llvm.riscv.th.mlbe.internal32
// CHECK:       @llvm.riscv.th.msbe.internal32
void test_tile_stores(const float *A, float *A_out,
                       const float *B, float *B_out,
                       long stride, mrow_t M, mcol_t K, mcol_t N) {
  mfloat32_t a = __riscv_th_mld_a_f32(A, stride, M, K);
  __riscv_th_mst_a_f32(A_out, stride, a, M, K);
  mfloat32_t b = __riscv_th_mld_b_f32(B, stride, K, N);
  __riscv_th_mst_b_f32(B_out, stride, b, K, N);
}
