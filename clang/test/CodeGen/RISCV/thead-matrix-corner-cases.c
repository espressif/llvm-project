// NOTE: XTHeadMatrix corner-case and real-world complex pattern tests.
// Exercises multi-accumulator pipelines, mixed-precision chains,
// tiled outer-product loops, post-processing pipelines, and other
// non-trivial usage patterns that stress the ISel, register model,
// and API correctness.
//
// RUN: %clang_cc1 -O2 -triple riscv64 -target-feature +experimental-xtheadmatrix -S -o - %s \
// RUN:   | FileCheck %s

#include <thead_matrix.h>

// ============================================================================
// Corner Case 1: Dual-accumulator GEMM
//
// Two independent matmul operations running in parallel on different
// accumulator registers. This was impossible with hardcoded ISel.
// Pattern: acc0 += tr1 * tr0, acc1 += tr3 * tr2
// ============================================================================

// CHECK-LABEL: dual_acc_gemm:
// CHECK-DAG: th.msettilem
// CHECK-DAG: th.msettilek
// CHECK: th.mlae8 tr0
// CHECK: th.mlae8 tr2
// CHECK-DAG: th.msettilen
// CHECK: th.mlbe8 tr1
// CHECK: th.mlbe8 tr3
// CHECK: th.mzero acc0
// CHECK: th.mzero acc1
// CHECK: th.mmacc.w.b acc0, tr1, tr0
// CHECK: th.mmacc.w.b acc1, tr3, tr2
// CHECK: th.msce32 acc0
// CHECK: th.msce32 acc1
void dual_acc_gemm(const int8_t *A0, long as0, const int8_t *A1, long as1,
                   const int8_t *B0, long bs0, const int8_t *B1, long bs1,
                   int32_t *C0, long cs0, int32_t *C1, long cs1,
                   mrow_t M, mrow_t K, mcol_t N) {
    // Load two A matrices into tr0, tr2
    __builtin_riscv_th_msettilem(M);
    __builtin_riscv_th_msettilek(K);
    (void)__builtin_riscv_th_mlae8(0, (void*)A0, as0);  // tr0
    (void)__builtin_riscv_th_mlae8(2, (void*)A1, as1);  // tr2

    // Load two B matrices into tr1, tr3
    __builtin_riscv_th_msettilen(N);
    (void)__builtin_riscv_th_mlbe8(1, (void*)B0, bs0);  // tr1
    (void)__builtin_riscv_th_mlbe8(3, (void*)B1, bs1);  // tr3

    // Zero two accumulators
    __builtin_riscv_th_mzero(4);  // acc0
    __builtin_riscv_th_mzero(5);  // acc1

    // Two independent matmuls
    __builtin_riscv_th_mmacc_w_b(4, 1, 0,
        __builtin_riscv_th_mundef_i32(),
        __builtin_riscv_th_mundef_i8(),
        __builtin_riscv_th_mundef_i8());  // acc0 += tr1 * tr0
    __builtin_riscv_th_mmacc_w_b(5, 3, 2,
        __builtin_riscv_th_mundef_i32(),
        __builtin_riscv_th_mundef_i8(),
        __builtin_riscv_th_mundef_i8());  // acc1 += tr3 * tr2

    // Store both results
    __builtin_riscv_th_msce32(4, (void*)C0, cs0);  // acc0
    __builtin_riscv_th_msce32(5, (void*)C1, cs1);  // acc1
}

// ============================================================================
// Corner Case 2: Mixed-precision matmul chain (FP8 → FP16 → FP32)
//
// Multi-stage widening: FP8 inputs → FP16 matmul → widen to FP32 → store.
// Tests correct register management across widening stages.
// ============================================================================

// CHECK-LABEL: mixed_precision_chain:
// CHECK: th.mfmacc.h.e4 acc0, tr1, tr0
// CHECK: th.mfcvtl.s.h acc1, acc0
// CHECK: th.msce32 acc1
void mixed_precision_chain(const void *A, long as, const void *B, long bs,
                            float *C, long cs,
                            mrow_t M, mrow_t K, mcol_t N) {
    // FP8 matmul → FP16 accumulation in acc0
    __builtin_riscv_th_msettilem(M);
    __builtin_riscv_th_msettilek(K);
    (void)__builtin_riscv_th_mlae8(0, (void*)A, as);
    __builtin_riscv_th_msettilen(N);
    (void)__builtin_riscv_th_mlbe8(1, (void*)B, bs);
    __builtin_riscv_th_mzero(4);
    __builtin_riscv_th_mfmacc_h_e4(4, 1, 0);  // acc0: FP16 result

    // Widen FP16 → FP32 (acc0 → acc1)
    __builtin_riscv_th_mfcvtl_s_h(5, 4,
        __builtin_riscv_th_mundef_f16());  // acc1 = widen(acc0)

    // Store FP32 result
    __builtin_riscv_th_msce32(5, (void*)C, cs);
}

// ============================================================================
// Corner Case 3: GEMM with post-processing (matmul + bias + activation)
//
// Complete inference pipeline: C = relu(A*B + bias)
// Uses matmul → EW add (bias) → EW max (relu) in sequence.
// ============================================================================

// CHECK-LABEL: gemm_with_postprocess:
// CHECK: th.mzero acc0
// CHECK: th.mfmacc.s acc0, tr1, tr0
// CHECK: th.mfadd.s.mm acc0, acc2, acc1
// CHECK: th.mfmax.s.mm acc0, acc2, acc1
// CHECK: th.msce32 acc0
void gemm_with_postprocess(const float *A, long as, const float *B, long bs,
                            float *C, long cs,
                            mrow_t M, mrow_t K, mcol_t N) {
    // Load A, B
    mfloat32_t a = __riscv_th_mld_a_f32(A, as, M, K);
    mfloat32_t b = __riscv_th_mld_b_f32(B, bs, K, N);

    // C = A * B (accumulate into zero)
    mfloat32_t c = __riscv_th_mzero_f32();
    c = __riscv_th_fmmacc_s(c, a, b, M, K, N);

    // c += bias (EW add with acc1 holding bias, which was loaded elsewhere)
    // In practice bias is in acc1, we simulate with mundefined
    mfloat32_t bias = __riscv_th_mundefined_f32();
    c = __riscv_th_mfadd_s_mm(c, bias, bias);

    // relu: c = max(c, 0)
    mfloat32_t zero_mat = __riscv_th_mundefined_f32();
    c = __riscv_th_mfmax_s_mm(c, zero_mat, zero_mat);

    // Store
    __riscv_th_mst_c_f32(C, cs, c, M, N);
}

// ============================================================================
// Corner Case 4: Tiled outer-product accumulation
//
// Simulates an outer loop over K dimension, accumulating into the same C tile:
//   for k_tile in range(K/TK):
//       C += A_tile[k] * B_tile[k]
// Tests that matmul accumulates correctly (not zeroed each iteration).
// ============================================================================

// CHECK-LABEL: tiled_accumulation:
// CHECK: th.mlce32 acc0
// CHECK: th.mlae32 tr0
// CHECK: th.mlbe32 tr1
// CHECK: th.mfmacc.s acc0, tr1, tr0
// CHECK: th.mlae32 tr0
// CHECK: th.mlbe32 tr1
// CHECK: th.mfmacc.s acc0, tr1, tr0
// CHECK: th.msce32 acc0
void tiled_accumulation(const float *A, long as,
                         const float *B, long bs,
                         float *C, long cs,
                         mrow_t M, mcol_t N, mrow_t TK) {
    // Load existing C accumulator (don't zero — accumulate across K tiles)
    mfloat32_t c = __riscv_th_mld_c_f32(C, cs, M, N);

    // K-tile 0
    mfloat32_t a0 = __riscv_th_mld_a_f32(A, as, M, TK);
    mfloat32_t b0 = __riscv_th_mld_b_f32(B, bs, TK, N);
    c = __riscv_th_fmmacc_s(c, a0, b0, M, TK, N);

    // K-tile 1 (different A/B pointers, same accumulator)
    const float *A1 = A + TK;
    const float *B1 = B + TK * (bs / sizeof(float));
    mfloat32_t a1 = __riscv_th_mld_a_f32(A1, as, M, TK);
    mfloat32_t b1 = __riscv_th_mld_b_f32(B1, bs, TK, N);
    c = __riscv_th_fmmacc_s(c, a1, b1, M, TK, N);

    // Store accumulated result
    __riscv_th_mst_c_f32(C, cs, c, M, N);
}

// ============================================================================
// Corner Case 5: INT8 GEMM with N4clip quantization output
//
// Full quantized inference: INT8 matmul → INT32 accumulate → N4clip → INT8
// Tests the complete quantization pipeline.
// ============================================================================

// CHECK-LABEL: quantized_gemm_n4clip:
// CHECK: th.mzero acc0
// CHECK: th.mmacc.w.b acc0, tr1, tr0
// CHECK: th.mn4clipl.w.mm acc0, acc2, acc1
void quantized_gemm_n4clip(const int8_t *A, long as,
                            const int8_t *B, long bs,
                            mrow_t M, mrow_t K, mcol_t N) {
    mint8_t a = __riscv_th_mld_a_i8(A, as, M, K);
    mint8_t b = __riscv_th_mld_b_i8(B, bs, K, N);

    // INT8 matmul → INT32
    mint32_t c = __riscv_th_mzero_i32();
    c = __riscv_th_mmaqa_ss_w_b(c, a, b, M, K, N);

    // Quantize INT32 → INT8 via N4clip
    mint32_t shift = __riscv_th_mundefined_i32();
    mint8_t result = __riscv_th_mn4clipl_w_mm(c, shift);
    (void)result;
}

// ============================================================================
// Corner Case 6: Transposed GEMM (A^T * B)
//
// Uses transposed load for A and normal load for B.
// Tests that mlate uses tr2 (not tr0).
// ============================================================================

// CHECK-LABEL: transposed_a_gemm:
// CHECK: th.mlate32 tr2
// CHECK: th.mlbe32 tr1
// CHECK: th.mzero acc0
// CHECK: th.mfmacc.s acc0, tr1, tr2
void transposed_a_gemm(const float *A, long as, const float *B, long bs,
                        mrow_t M, mrow_t K, mcol_t N) {
    // Load A-transposed into tr2
    mfloat32_t at = __riscv_th_mld_at_f32(A, as, M, K);
    // Load B into tr1
    mfloat32_t b = __riscv_th_mld_b_f32(B, bs, K, N);

    mfloat32_t c = __riscv_th_mzero_f32();

    // Matmul with A^T: must use tr2 for ms1 (instead of tr0)
    // Low-level call to verify register flexibility
    (void)__builtin_riscv_th_mfmacc_s(4, 1, 2, c, at, b);

    (void)c;
}

// ============================================================================
// Corner Case 7: Four-way accumulator GEMM
//
// All 4 accumulator registers used simultaneously for different
// sub-problems. This is the maximum parallelism the hardware supports.
// ============================================================================

// CHECK-LABEL: quad_acc_gemm:
// CHECK: th.mzero acc0
// CHECK: th.mzero acc1
// CHECK: th.mzero acc2
// CHECK: th.mzero acc3
// CHECK: th.mfmacc.s acc0, tr1, tr0
// CHECK: th.mfmacc.s acc1, tr1, tr0
// CHECK: th.mfmacc.s acc2, tr1, tr0
// CHECK: th.mfmacc.s acc3, tr1, tr0
// CHECK: th.msce32 acc0
// CHECK: th.msce32 acc1
// CHECK: th.msce32 acc2
// CHECK: th.msce32 acc3
void quad_acc_gemm(const float *A, long as, const float *B, long bs,
                    float *C0, float *C1, float *C2, float *C3, long cs,
                    mrow_t M, mrow_t K, mcol_t N) {
    mfloat32_t a = __riscv_th_mld_a_f32(A, as, M, K);
    mfloat32_t b = __riscv_th_mld_b_f32(B, bs, K, N);

    // Zero all 4 accumulators
    __builtin_riscv_th_mzero(4);  // acc0
    __builtin_riscv_th_mzero(5);  // acc1
    __builtin_riscv_th_mzero(6);  // acc2
    __builtin_riscv_th_mzero(7);  // acc3

    // Same A*B accumulated to 4 different destinations
    // (in real code, these would use different A/B tile pairs)
    (void)__builtin_riscv_th_mfmacc_s(4, 1, 0, a, a, b);
    (void)__builtin_riscv_th_mfmacc_s(5, 1, 0, a, a, b);
    (void)__builtin_riscv_th_mfmacc_s(6, 1, 0, a, a, b);
    (void)__builtin_riscv_th_mfmacc_s(7, 1, 0, a, a, b);

    // Store all 4 results
    __builtin_riscv_th_msce32(4, (void*)C0, cs);
    __builtin_riscv_th_msce32(5, (void*)C1, cs);
    __builtin_riscv_th_msce32(6, (void*)C2, cs);
    __builtin_riscv_th_msce32(7, (void*)C3, cs);
}

// ============================================================================
// Corner Case 8: INT8 unsigned-signed mixed matmul
//
// Tests all 4 sign combinations of INT8→INT32 matmul.
// Verifies correct instruction mapping for each combination.
// ============================================================================

// CHECK-LABEL: int8_sign_variants:
// CHECK: th.mmacc.w.b acc0, tr1, tr0
// CHECK: th.mmaccu.w.b acc1, tr1, tr0
// CHECK: th.mmaccus.w.b acc2, tr1, tr0
// CHECK: th.mmaccsu.w.b acc3, tr1, tr0
void int8_sign_variants(const void *A, long as, const void *B, long bs,
                         mrow_t M, mrow_t K, mcol_t N) {
    mint8_t i8a = __riscv_th_mld_a_i8(A, as, M, K);
    muint8_t u8a = __riscv_th_mld_a_u8(A, as, M, K);
    mint8_t i8b = __riscv_th_mld_b_i8(B, bs, K, N);
    muint8_t u8b = __riscv_th_mld_b_u8(B, bs, K, N);

    mint32_t z32 = __riscv_th_mzero_i32();
    muint32_t uz32 = __riscv_th_mzero_u32();

    // signed × signed → acc0
    (void)__builtin_riscv_th_mmacc_w_b(4, 1, 0, z32, i8a, i8b);
    // unsigned × unsigned → acc1
    (void)__builtin_riscv_th_mmaccu_w_b(5, 1, 0, uz32, u8a, u8b);
    // mixed us → acc2
    (void)__builtin_riscv_th_mmaccus_w_b(6, 1, 0, z32, u8a, i8b);
    // mixed su → acc3
    (void)__builtin_riscv_th_mmaccsu_w_b(7, 1, 0, z32, i8a, u8b);
}

// ============================================================================
// Corner Case 9: Full FP conversion chain
//
// FP8 → FP16 → FP32 → FP64 → FP32 → FP16 → FP8
// Tests every conversion width in both directions.
// ============================================================================

// CHECK-LABEL: fp_conversion_chain:
// CHECK: th.mfcvtl.h.e4 acc0, acc1
// CHECK: th.mfcvtl.s.h acc2, acc0
// CHECK: th.mfcvtl.d.s acc0, acc2
// CHECK: th.mfcvtl.s.d acc2, acc0
// CHECK: th.mfcvtl.h.s acc0, acc2
// CHECK: th.mfcvtl.e4.h acc2, acc0
void fp_conversion_chain(void) {
    // FP8(e4m3) → FP16 (widen low half)
    __builtin_riscv_th_mfcvtl_h_e4(4, 5);      // acc0 = widen_l(acc1)

    // FP16 → FP32 (widen low half)
    (void)__builtin_riscv_th_mfcvtl_s_h(6, 4,
        __builtin_riscv_th_mundef_f16());         // acc2 = widen_l(acc0)

    // FP32 → FP64 (widen low half)
    (void)__builtin_riscv_th_mfcvtl_d_s(4, 6,
        __builtin_riscv_th_mundef_f32());         // acc0 = widen_l(acc2)

    // FP64 → FP32 (narrow low half)
    (void)__builtin_riscv_th_mfcvtl_s_d(6, 4,
        __builtin_riscv_th_mundef_f64());         // acc2 = narrow_l(acc0)

    // FP32 → FP16 (narrow low half)
    (void)__builtin_riscv_th_mfcvtl_h_s(4, 6,
        __builtin_riscv_th_mundef_f32());         // acc0 = narrow_l(acc2)

    // FP16 → FP8(e4m3) (narrow low half)
    __builtin_riscv_th_mfcvtl_e4_h(6, 4);        // acc2 = narrow_l(acc0)
}

// ============================================================================
// Corner Case 10: EW arithmetic chain (add → mul → shift → clip)
//
// Complete INT32 post-processing: add bias, multiply scale, shift, clip.
// Tests that EW operations chain correctly on acc registers.
// ============================================================================

// CHECK-LABEL: ew_postprocess_chain:
// CHECK: th.madd.w.mm acc0, acc2, acc1
// CHECK: th.mmul.w.mm acc0, acc2, acc1
// CHECK: th.msra.w.mm acc0, acc2, acc1
// CHECK: th.mn4clipl.w.mm acc0, acc2, acc1
void ew_postprocess_chain(void) {
    mint32_t data = __riscv_th_mundefined_i32();
    mint32_t bias = __riscv_th_mundefined_i32();
    mint32_t scale = __riscv_th_mundefined_i32();

    // Add bias
    data = __riscv_th_madd_w_mm(data, bias, bias);
    // Multiply by scale
    data = __riscv_th_mmul_w_mm(data, scale, scale);
    // Arithmetic right shift
    mint32_t shift = __riscv_th_mundefined_i32();
    data = __riscv_th_msra_w_mm(data, shift, shift);
    // N4clip to INT8
    mint8_t clipped = __riscv_th_mn4clipl_w_mm(data, shift);
    (void)clipped;
}

// ============================================================================
// Corner Case 11: Slide + broadcast for data layout manipulation
//
// Uses row slide and column broadcast to rearrange matrix data.
// Tests that slide/broadcast macros work with correct register indices.
// ============================================================================

// CHECK-LABEL: data_layout_manipulation:
// CHECK: th.mrslidedown tr0, tr1, 2
// CHECK: th.mcslidedown.w tr0, tr1, 1
// CHECK: th.mrbca.mv.i tr0, tr1, 3
// CHECK: th.mcbcaw.mv.i tr0, tr1, 0
void data_layout_manipulation(void) {
    // Row slide down by 2
    __riscv_th_mrslidedown(2);
    // Column slide down by 1 element (32-bit)
    __riscv_th_mcslidedown_w(1);
    // Broadcast row 3 to all rows
    __riscv_th_mrbca(3);
    // Broadcast column 0 (32-bit) to all columns
    __riscv_th_mcbca_w(0);
}

// ============================================================================
// Corner Case 12: FP16 GEMM → mfmax → store (clamped matmul)
//
// FP16 matrix multiply then clamp output to a maximum value.
// Tests FP EW max after matmul.
// ============================================================================

// CHECK-LABEL: clamped_fp16_gemm:
// CHECK: th.mzero acc0
// CHECK: th.mfmacc.h acc0, tr1, tr0
// CHECK: th.mfmax.h.mm acc0, acc2, acc1
// CHECK: th.msce16 acc0
void clamped_fp16_gemm(const void *A, long as, const void *B, long bs,
                        void *C, long cs,
                        mrow_t M, mrow_t K, mcol_t N) {
    mfloat16_t a = __riscv_th_mld_a_f16(A, as, M, K);
    mfloat16_t b = __riscv_th_mld_b_f16(B, bs, K, N);
    mfloat16_t c = __riscv_th_mzero_f16();
    c = __riscv_th_fmmacc_h(c, a, b, M, K, N);

    // Clamp: c = max(c, max_val_matrix)
    mfloat16_t max_val = __riscv_th_mundefined_f16();
    c = __riscv_th_mfmax_h_mm(c, max_val, max_val);

    __riscv_th_mst_c_f16(C, cs, c, M, N);
}

// ============================================================================
// Corner Case 13: Matrix element extract and insert
//
// Extract elements from a matrix, modify in GPR, insert back.
// Tests mmov.x.m and mmov.m.x with different register indices.
// ============================================================================

// CHECK-LABEL: element_extract_insert:
// CHECK: th.mmovw.x.m {{.*}}, tr0
// CHECK: th.mmovw.m.x tr0
// CHECK: th.mmovw.x.m {{.*}}, acc0
// CHECK: th.mmovw.m.x acc0
void element_extract_insert(void) {
    mint32_t m = __riscv_th_mundefined_i32();

    // Extract element at index 0 from tr0
    size_t val = __riscv_th_mmov_x_m_w(m, 0);

    // Double it in GPR and insert back
    val *= 2;
    m = __riscv_th_mmov_m_x_w(m, val, 0);

    // Also test on acc register via low-level builtins
    size_t val2 = __builtin_riscv_th_mmovw_x_m(4, 0);
    __builtin_riscv_th_mmovw_m_x(4, val2 + 1, 0);
}

// ============================================================================
// Corner Case 14: Scalar broadcast (mdup) for all element sizes
//
// Broadcast a scalar value to all elements of different-width matrices.
// ============================================================================

// CHECK-LABEL: scalar_broadcast_all_sizes:
// CHECK: th.mdupb.m.x tr0
// CHECK: th.mduph.m.x tr0
// CHECK: th.mdupw.m.x tr0
// CHECK: th.mdupd.m.x tr0
void scalar_broadcast_all_sizes(void) {
    mint8_t b = __riscv_th_mdup_m_x_b(42);
    mint16_t h = __riscv_th_mdup_m_x_h(1000);
    mint32_t w = __riscv_th_mdup_m_x_w(0x12345678);
    mint64_t d = __riscv_th_mdup_m_x_d(0xDEADBEEFCAFELL);
    (void)b; (void)h; (void)w; (void)d;
}

// ============================================================================
// Corner Case 15: INT16→INT64 signed matmul
//
// Tests the 16-bit widening matmul variant.
// ============================================================================

// CHECK-LABEL: int16_matmul_full:
// CHECK: th.mlae16 tr0
// CHECK: th.mlbe16 tr1
// CHECK: th.mzero acc0
// CHECK: th.mmacc.d.h acc0, tr1, tr0
// CHECK: th.msce64 acc0
// CHECK: th.mrelease
void int16_matmul_full(const int16_t *A, long as,
                        const int16_t *B, long bs,
                        int64_t *C, long cs,
                        mrow_t M, mrow_t K, mcol_t N) {
    mint16_t a = __riscv_th_mld_a_i16(A, as, M, K);
    mint16_t b = __riscv_th_mld_b_i16(B, bs, K, N);
    mint64_t c = __riscv_th_mzero_i64();
    c = __riscv_th_mmaqa_ss_d_h(c, a, b, M, K, N);
    __riscv_th_mst_c_i64(C, cs, c, M, N);
    __riscv_th_mrelease();
}

// ============================================================================
// Corner Case 16: FP32→TF32→matmul→FP32
//
// Convert to TF32 for faster matmul, then convert back.
// Tests TF32 conversion + matmul pipeline.
// ============================================================================

// CHECK-LABEL: tf32_matmul_pipeline:
// CHECK: th.mfcvt.tf32.s acc0, acc1
// CHECK: th.mfcvt.tf32.s acc1, acc0
// CHECK: th.mzero acc2
// CHECK: th.mfmacc.s.tf32 acc2, tr1, tr0
// CHECK: th.mfcvt.s.tf32 acc0, acc2
void tf32_matmul_pipeline(void) {
    // Convert A data (in acc1) to TF32 format (→ acc0)
    __builtin_riscv_th_mfcvt_tf32_s(4, 5);  // acc0 = tf32(acc1)

    // Convert B data (in acc0) to TF32 format (→ acc1)
    __builtin_riscv_th_mfcvt_tf32_s(5, 4);  // acc1 = tf32(acc0)

    // TF32 matmul
    __builtin_riscv_th_mzero(6);  // acc2 = 0
    __builtin_riscv_th_mfmacc_s_tf32(6, 1, 0);  // acc2 += tr1 * tr0

    // Convert result back to FP32
    __builtin_riscv_th_mfcvt_s_tf32(4, 6);  // acc0 = fp32(acc2)
}

// ============================================================================
// Corner Case 17: Pack operations for data rearrangement
//
// Test all 3 pack variants (low+low, high+low, high+high).
// ============================================================================

// CHECK-LABEL: pack_all_variants:
// CHECK: th.mpack tr0, tr2, tr1
// CHECK: th.mpackhl tr0, tr2, tr1
// CHECK: th.mpackhh tr0, tr2, tr1
void pack_all_variants(void) {
    __riscv_th_mpack();    // tr0 = low(tr2) | low(tr1)
    __riscv_th_mpackhl();  // tr0 = high(tr2) | low(tr1)
    __riscv_th_mpackhh();  // tr0 = high(tr2) | high(tr1)
}

// ============================================================================
// Corner Case 18: Whole-register save/restore pattern
//
// Save all 8 matrix registers to memory, then restore.
// Tests whole-register load/store with all register indices.
// ============================================================================

// CHECK-LABEL: save_restore_all_regs:
// CHECK: th.msme32 tr0
// CHECK: th.msme32 tr1
// CHECK: th.msme32 tr2
// CHECK: th.msme32 tr3
// CHECK: th.msme32 acc0
// CHECK: th.msme32 acc1
// CHECK: th.msme32 acc2
// CHECK: th.msme32 acc3
// CHECK: th.mlme32 tr0
// CHECK: th.mlme32 tr1
// CHECK: th.mlme32 tr2
// CHECK: th.mlme32 tr3
// CHECK: th.mlme32 acc0
// CHECK: th.mlme32 acc1
// CHECK: th.mlme32 acc2
// CHECK: th.mlme32 acc3
void save_restore_all_regs(void *buf) {
    char *p = (char*)buf;
    // Save all 8 registers (ImmArg requires constant indices)
    __builtin_riscv_th_msme32(0, p);
    __builtin_riscv_th_msme32(1, p + 4096);
    __builtin_riscv_th_msme32(2, p + 8192);
    __builtin_riscv_th_msme32(3, p + 12288);
    __builtin_riscv_th_msme32(4, p + 16384);
    __builtin_riscv_th_msme32(5, p + 20480);
    __builtin_riscv_th_msme32(6, p + 24576);
    __builtin_riscv_th_msme32(7, p + 28672);
    // Restore all 8 registers
    (void)__builtin_riscv_th_mlme32(0, p);
    (void)__builtin_riscv_th_mlme32(1, p + 4096);
    (void)__builtin_riscv_th_mlme32(2, p + 8192);
    (void)__builtin_riscv_th_mlme32(3, p + 12288);
    (void)__builtin_riscv_th_mlme32(4, p + 16384);
    (void)__builtin_riscv_th_mlme32(5, p + 20480);
    (void)__builtin_riscv_th_mlme32(6, p + 24576);
    (void)__builtin_riscv_th_mlme32(7, p + 28672);
}

// ============================================================================
// Corner Case 19: FP64 widening from FP32
//
// FP32 matmul widening to FP64 accumulation.
// ============================================================================

// CHECK-LABEL: fp32_to_fp64_gemm:
// CHECK: th.mzero acc0
// CHECK: th.mfmacc.d.s acc0, tr1, tr0
// CHECK: th.msce64 acc0
void fp32_to_fp64_gemm(const float *A, long as, const float *B, long bs,
                         double *C, long cs,
                         mrow_t M, mrow_t K, mcol_t N) {
    mfloat32_t a = __riscv_th_mld_a_f32(A, as, M, K);
    mfloat32_t b = __riscv_th_mld_b_f32(B, bs, K, N);
    mfloat64_t c = __riscv_th_mzero_f64();
    c = __riscv_th_fwmmacc_d_s(c, a, b, M, K, N);
    __riscv_th_mst_c_f64(C, cs, c, M, N);
}

// ============================================================================
// Corner Case 20: Float-Int round-trip conversion
//
// INT8 → FP16 → compute → FP16 → INT8
// Tests float-int conversion + EW arithmetic + reverse conversion.
// ============================================================================

// CHECK-LABEL: float_int_roundtrip:
// CHECK: th.msfcvtl.h.b acc0, acc1
// CHECK: th.mfmul.h.mm acc0, acc2, acc1
// CHECK: th.mfscvtl.b.h acc0, acc1
void float_int_roundtrip(void) {
    mint8_t i8_data = __riscv_th_mundefined_i8();

    // INT8 → FP16 (signed widening, low half)
    mfloat16_t f16 = __riscv_th_msfcvtl_h_b(i8_data);

    // FP16 element-wise multiply
    mfloat16_t scale = __riscv_th_mundefined_f16();
    f16 = __riscv_th_mfmul_h_mm(f16, scale, scale);

    // FP16 → INT8 (signed narrowing, low half)
    mint8_t result = __riscv_th_mfscvtl_b_h(f16);
    (void)result;
}
