// NOTE: XTHeadMatrix real-world usage examples and patterns.
// Tests practical use cases that exercise the full API pipeline.
//
// RUN: %clang_cc1 -O2 -triple riscv64 -target-feature +experimental-xtheadmatrix -S -o - %s \
// RUN:   | FileCheck %s

#include <thead_matrix.h>

// ============================================================================
// Example 1: INT8 GEMM with accumulation (int8 * int8 -> int32)
//
// Pattern: Load A tiles, load B tiles, zero accumulator, matmul, store result
// ============================================================================

// CHECK-LABEL: example_int8_gemm:
// CHECK-DAG: th.msettilem
// CHECK-DAG: th.msettilek
// CHECK: th.mlae8
// CHECK-DAG: th.msettilen
// CHECK: th.mlbe8
// CHECK: th.mzero
// CHECK: th.mmacc.w.b
// CHECK: th.msce32
// CHECK: th.mrelease
// CHECK: ret
void example_int8_gemm(const int8_t *A, long a_stride,
                        const int8_t *B, long b_stride,
                        int32_t *C, long c_stride,
                        mrow_t M, mrow_t K, mcol_t N) {
  // Load input matrices into tile registers
  mint8_t a = __riscv_th_mld_a_i8(A, a_stride, M, K);
  mint8_t b = __riscv_th_mld_b_i8(B, b_stride, K, N);

  // Zero accumulator register
  mint32_t c = __riscv_th_mzero_i32();

  // Signed integer matrix multiply-accumulate: c += a * b
  c = __riscv_th_mmaqa_ss_w_b(c, a, b, M, K, N);

  // Store result from accumulator
  __riscv_th_mst_c_i32(C, c_stride, c, M, N);

  // Release matrix state
  __riscv_th_mrelease();
}

// ============================================================================
// Example 2: FP32 GEMM with pre-existing accumulator
//
// Pattern: Load-compute-store without zeroing (for tiled outer products)
// ============================================================================

// CHECK-LABEL: example_fp32_gemm_accumulate:
// CHECK-DAG: th.msettilem
// CHECK-DAG: th.msettilen
// CHECK: th.mlce32
// CHECK-DAG: th.msettilek
// CHECK: th.mlae32
// CHECK: th.mlbe32
// CHECK: th.mfmacc.s
// CHECK: th.msce32
// CHECK: ret
void example_fp32_gemm_accumulate(const float *A, long a_stride,
                                   const float *B, long b_stride,
                                   float *C, long c_stride,
                                   mrow_t M, mrow_t K, mcol_t N) {
  // Load existing C accumulator
  mfloat32_t c = __riscv_th_mld_c_f32(C, c_stride, M, N);

  // Load A and B tiles
  mfloat32_t a = __riscv_th_mld_a_f32(A, a_stride, M, K);
  mfloat32_t b = __riscv_th_mld_b_f32(B, b_stride, K, N);

  // FP matmul accumulate: c += a * b
  c = __riscv_th_fmmacc_s(c, a, b, M, K, N);

  // Store result
  __riscv_th_mst_c_f32(C, c_stride, c, M, N);
}

// ============================================================================
// Example 3: FP16 to FP32 widening matmul
//
// Pattern: Load FP16 inputs, compute in FP32, store FP32 result
// ============================================================================

// CHECK-LABEL: example_fp16_widening_gemm:
// CHECK-DAG: th.msettilem
// CHECK-DAG: th.msettilek
// CHECK: th.mlae16
// CHECK-DAG: th.msettilen
// CHECK: th.mlbe16
// CHECK: th.mzero
// CHECK: th.mfmacc.s.h
// CHECK: th.msce32
// CHECK: ret
void example_fp16_widening_gemm(const void *A, long a_stride,
                                 const void *B, long b_stride,
                                 float *C, long c_stride,
                                 mrow_t M, mrow_t K, mcol_t N) {
  // Load FP16 tiles
  mfloat16_t a = __riscv_th_mld_a_f16(A, a_stride, M, K);
  mfloat16_t b = __riscv_th_mld_b_f16(B, b_stride, K, N);

  // Zero FP32 accumulator
  mfloat32_t c = __riscv_th_mzero_f32();

  // Widening matmul: FP16 * FP16 -> FP32
  c = __riscv_th_fwmmacc_s_h(c, a, b, M, K, N);

  // Store FP32 result
  __riscv_th_mst_c_f32(C, c_stride, c, M, N);
}

// ============================================================================
// Example 4: Mixed-sign integer matmul (uint8 * int8 -> int32)
//
// Pattern: Common in quantized neural networks (unsigned activations, signed weights)
// ============================================================================

// CHECK-LABEL: example_mixed_sign_gemm:
// CHECK: th.mlae8
// CHECK: th.mlbe8
// CHECK: th.mzero
// CHECK: th.mmaccus.w.b
// CHECK: th.msce32
// CHECK: ret
void example_mixed_sign_gemm(const uint8_t *activations, long a_stride,
                              const int8_t *weights, long w_stride,
                              int32_t *output, long o_stride,
                              mrow_t M, mrow_t K, mcol_t N) {
  muint8_t a = __riscv_th_mld_a_u8(activations, a_stride, M, K);
  mint8_t  w = __riscv_th_mld_b_i8(weights, w_stride, K, N);
  mint32_t c = __riscv_th_mzero_i32();

  // Unsigned * signed matmul
  c = __riscv_th_mmaqa_us_w_b(c, a, w, M, K, N);

  __riscv_th_mst_c_i32(output, o_stride, c, M, N);
}

// ============================================================================
// Example 5: Element-wise operations pipeline
//
// Pattern: Load -> EW arithmetic -> store (bias add after matmul)
// ============================================================================

// CHECK-LABEL: example_bias_add:
// CHECK: th.mlce32
// CHECK: th.madd.w.mm
// CHECK: th.msce32
// CHECK: ret
void example_bias_add(float *output, long o_stride,
                       const float *bias, long b_stride,
                       mrow_t M, mcol_t N) {
  // Load output (int32 result from matmul)
  mint32_t out = __riscv_th_mld_c_i32(output, o_stride, M, N);

  // Load bias (also int32)
  mint32_t b = __riscv_th_mld_c_i32(bias, b_stride, M, N);

  // Element-wise add
  out = __riscv_th_madd_w_mm(out, out, b);

  // Store result
  __riscv_th_mst_c_i32(output, o_stride, out, M, N);
}

// ============================================================================
// Example 6: INT8 quantization with N4Clip
//
// Pattern: int32 matmul result -> shift right -> clip to int8
// ============================================================================

// CHECK-LABEL: example_quantize_n4clip:
// CHECK: th.mmacc.w.b
// CHECK: th.msra.w.mm
// CHECK: th.mn4clipl.w.mm
// CHECK: th.msae8
// CHECK: ret
void example_quantize_n4clip(mrow_t M, mrow_t K, mcol_t N,
                              void *out, long out_stride) {
  // Assume matmul already computed
  mint32_t acc = __riscv_th_mundefined_i32();
  mint8_t  a = __riscv_th_mundefined_i8();
  mint8_t  b = __riscv_th_mundefined_i8();

  // Compute matmul
  acc = __riscv_th_mmaqa_ss_w_b(acc, a, b, M, K, N);

  // Shift right by amount in shift_matrix
  mint32_t shift = __riscv_th_mundefined_i32();
  acc = __riscv_th_msra_w_mm(acc, acc, shift);

  // N4Clip to signed int8
  mint8_t result = __riscv_th_mn4clipl_w_mm(acc, shift);

  // Store int8 result
  __riscv_th_mst_a_i8(out, out_stride, result, M, K);
}

// ============================================================================
// Example 7: Type conversion pipeline (int32 -> fp32 -> fp16)
//
// Pattern: Common in post-processing for mixed-precision inference
// ============================================================================

// CHECK-LABEL: example_type_pipeline:
// CHECK: th.msfcvt.s.w
// CHECK: th.mfcvtl.h.s
// CHECK: ret
void example_type_pipeline(void) {
  mint32_t  i32 = __riscv_th_mundefined_i32();
  mfloat32_t f32 = __riscv_th_msfcvt_s_w(i32);   // int32 -> fp32
  mfloat16_t f16 = __riscv_th_mfcvtl_h_s(f32);   // fp32 -> fp16 (low half)
  (void)f16;
}

// ============================================================================
// Example 8: CSR-based tile auto-sizing
//
// Pattern: Read hardware capabilities, then configure appropriately
// ============================================================================

// CHECK-LABEL: example_auto_size:
// CHECK: csrr {{.*}}, th.xtlenb
// CHECK: csrr {{.*}}, th.xtrlenb
// CHECK: th.msettilem
// CHECK: th.msettilen
// CHECK: th.msettilek
// CHECK: ret
void example_auto_size(void) {
  unsigned long tlen = __riscv_th_xmlenb();
  unsigned long rlen = __riscv_th_xrlenb();

  // Calculate max rows from tlen/rlen ratio
  unsigned long max_rows = tlen / rlen;

  __riscv_th_msetmrow_m(max_rows);
  __riscv_th_msetmrow_n(max_rows);
  __riscv_th_msetmcol_e32(rlen / 4);  // 4 bytes per element
}

// ============================================================================
// Example 9: Transposed matmul (A^T * B)
//
// Pattern: Use transposed load for A matrix
// ============================================================================

// CHECK-LABEL: example_transposed_gemm:
// CHECK: th.mlate32
// CHECK: th.mlbe32
// CHECK: th.mzero
// CHECK: th.mfmacc.s
// CHECK: th.msce32
// CHECK: ret
void example_transposed_gemm(const float *AT, long at_stride,
                              const float *B, long b_stride,
                              float *C, long c_stride,
                              mrow_t M, mrow_t K, mcol_t N) {
  // Load A transposed (column-major -> row-major conversion in hardware)
  mfloat32_t a = __riscv_th_mld_at_f32(AT, at_stride, M, K);
  mfloat32_t b = __riscv_th_mld_b_f32(B, b_stride, K, N);
  mfloat32_t c = __riscv_th_mzero_f32();
  c = __riscv_th_fmmacc_s(c, a, b, M, K, N);
  __riscv_th_mst_c_f32(C, c_stride, c, M, N);
}

// ============================================================================
// Example 10: Scalar broadcast and element extraction
//
// Pattern: Fill matrix column with scalar, extract elements for verification
// ============================================================================

// CHECK-LABEL: example_scalar_ops:
// CHECK: th.mdupw.m.x
// CHECK: th.mmovw.x.m
// CHECK: th.mmovw.m.x
// CHECK: ret
void example_scalar_ops(int32_t fill_val, size_t idx) {
  // Broadcast scalar to matrix column
  mint32_t mat = __riscv_th_mdup_m_x_w((size_t)fill_val);

  // Extract an element to GPR
  size_t elem = __riscv_th_mmov_x_m_w(mat, idx);

  // Insert modified element back
  mat = __riscv_th_mmov_m_x_w(mat, elem + 1, idx);

  (void)mat;
}

// ============================================================================
// Example 11: Packed INT4 matmul
//
// Pattern: Use packed matmul for ultra-low precision inference
// ============================================================================

// CHECK-LABEL: example_int4_gemm:
// CHECK: th.mlae8
// CHECK: th.mlbe8
// CHECK: th.mzero
// CHECK: th.pmmacc.w.b
// CHECK: th.msce32
// CHECK: ret
void example_int4_gemm(const void *A, long a_stride,
                        const void *B, long b_stride,
                        int32_t *C, long c_stride,
                        mrow_t M, mrow_t K, mcol_t N) {
  // Load INT4 data packed in INT8 containers
  mint8_t a = __riscv_th_mld_a_i8(A, a_stride, M, K);
  mint8_t b = __riscv_th_mld_b_i8(B, b_stride, K, N);

  mint32_t c = __riscv_th_mzero_i32();

  // Packed INT4 matmul (processes 2x more elements per byte)
  c = __riscv_th_pmmaqa_ss_w_b(c, a, b, M, K, N);

  __riscv_th_mst_c_i32(C, c_stride, c, M, N);
}

// ============================================================================
// Example 12: FP EW max reduction (approximate softmax numerator)
//
// Pattern: mfmax to find row maximum, mfsub to subtract
// ============================================================================

// CHECK-LABEL: example_fp_ew_chain:
// CHECK: th.mfmax.s.mm
// CHECK: th.mfsub.s.mm
// CHECK: th.mfmul.s.mm
// CHECK: ret
void example_fp_ew_chain(void) {
  mfloat32_t data = __riscv_th_mundefined_f32();
  mfloat32_t ones = __riscv_th_mundefined_f32();

  // Find element-wise max
  mfloat32_t maxval = __riscv_th_mfmax_s_mm(data, data, ones);

  // Subtract max for numerical stability
  data = __riscv_th_mfsub_s_mm(data, data, maxval);

  // Scale
  data = __riscv_th_mfmul_s_mm(data, data, ones);

  (void)data;
}

// ============================================================================
// Example 13: Reinterpret cast workflow (int view of fp data)
// ============================================================================

// CHECK-LABEL: example_reinterpret:
// CHECK: th.mzero
// CHECK: ret
void example_reinterpret(void) {
  // Create an FP32 zero matrix
  mfloat32_t fzero = __riscv_th_mzero_f32();

  // Reinterpret as int32 for bitwise operations
  mint32_t izero = __riscv_th_mreinterpret_i32(fzero);

  // Reinterpret back to fp32
  mfloat32_t back = __riscv_th_mreinterpret_f32(izero);

  (void)back;
}

// ============================================================================
// Example 14: Whole-register save/restore
// ============================================================================

// CHECK-LABEL: example_save_restore:
// CHECK: th.msme32
// CHECK: th.mlme32
// CHECK: ret
void example_save_restore(void *save_area) {
  mint32_t reg = __riscv_th_mundefined_i32();

  // Save matrix register to memory (whole register, ignoring tile config)
  __riscv_th_mst_whole_i32(save_area, reg);

  // ... do other work ...

  // Restore from memory
  reg = __riscv_th_mld_whole_i32(save_area);

  (void)reg;
}

// ============================================================================
// Example 15: INT16 -> INT64 widening matmul
// ============================================================================

// CHECK-LABEL: example_int16_gemm:
// CHECK: th.mlae16
// CHECK: th.mlbe16
// CHECK: th.mzero
// CHECK: th.mmacc.d.h
// CHECK: th.msce64
// CHECK: ret
void example_int16_gemm(const int16_t *A, long a_stride,
                          const int16_t *B, long b_stride,
                          int64_t *C, long c_stride,
                          mrow_t M, mrow_t K, mcol_t N) {
  mint16_t a = __riscv_th_mld_a_i16(A, a_stride, M, K);
  mint16_t b = __riscv_th_mld_b_i16(B, b_stride, K, N);
  mint64_t c = __riscv_th_mzero_i64();

  c = __riscv_th_mmaqa_ss_d_h(c, a, b, M, K, N);

  __riscv_th_mst_c_i64(C, c_stride, c, M, N);
}

// ============================================================================
// Example 16: Row/column slide and broadcast operations
// ============================================================================

// CHECK-LABEL: example_slide_broadcast:
// CHECK: th.mrslidedown
// CHECK: th.mcslideup.w
// CHECK: th.mrbca.mv.i
// CHECK: th.mcbcaw.mv.i
// CHECK: ret
void example_slide_broadcast(void) {
  // Slide rows down by 2 positions
  __riscv_th_mrslidedown(2);

  // Slide columns up by 1 position (32-bit elements)
  __riscv_th_mcslideup_w(1);

  // Broadcast row 0 to all rows
  __riscv_th_mrbca(0);

  // Broadcast column 3 to all columns (32-bit elements)
  __riscv_th_mcbca_w(3);
}

// ============================================================================
// Example 17: Tuple operations (register pair management)
// ============================================================================

// CHECK-LABEL: example_tuple_ops:
// CHECK: ret
void example_tuple_ops(void) {
  // Create a pair of fp32 matrices
  mfloat32x2_t pair = __riscv_th_mzero_f32x2();

  // Get individual matrices from the pair
  mfloat32_t first = __riscv_th_mget_f32(pair, 0);
  mfloat32_t second = __riscv_th_mget_f32(pair, 1);

  // Set a matrix into the pair
  mfloat32_t val = __riscv_th_mundefined_f32();
  pair = __riscv_th_mset_f32(pair, 0, val);

  (void)first; (void)second; (void)pair;
}

// ============================================================================
// Example 18: Complete CSR workflow
// ============================================================================

// CHECK-LABEL: example_csr_workflow:
// CHECK: csrr {{.*}}, th.xmcsr
// CHECK: csrw th.xmfrm
// CHECK: csrw th.xmsaten
// CHECK: csrr {{.*}}, th.xmfflags
// CHECK: csrw th.xmfflags
// CHECK: ret
void example_csr_workflow(void) {
  // Read current matrix status
  unsigned long status = __riscv_th_mread_csr(RVM_CSR_XMCSR);
  (void)status;

  // Set FP rounding mode to round-to-nearest
  __riscv_th_mwrite_csr(RVM_CSR_XMFRM, 0);

  // Enable saturation for integer matmul
  __riscv_th_mwrite_csr(RVM_CSR_XMSATEN, 1);

  // Check and clear FP exception flags
  unsigned long flags = __riscv_th_mread_csr(RVM_CSR_XMFFLAGS);
  if (flags) {
    __riscv_th_mwrite_csr(RVM_CSR_XMFFLAGS, 0);
  }
}

// ==========================================================================
// Example 16: Multi-Accumulator GEMM via Low-Level Builtins
// Demonstrates register flexibility — two matmuls target different
// accumulators using explicit register index parameters.
// ==========================================================================
// CHECK-LABEL: test_dual_gemm_low_level:
// CHECK: th.msettilemi
// CHECK: th.msettileki
// CHECK: th.msettileni
// CHECK: th.mlae8 tr0
// CHECK: th.mlbe8 tr1
// CHECK: th.mzero acc0
// CHECK: th.mfmacc.h.e4 acc0, tr1, tr0
// CHECK: th.mlbe8 tr1
// CHECK: th.mzero acc1
// CHECK: th.mfmacc.h.e4 acc1, tr1, tr0
// CHECK: th.msce32 acc0
// CHECK: th.msce32 acc1
// CHECK: th.mrelease
void test_dual_gemm_low_level(const void *A, long a_stride,
                              const void *B1, long b1_stride,
                              const void *B2, long b2_stride,
                              void *C1, long c1_stride,
                              void *C2, long c2_stride) {
    // Configure dimensions
    __builtin_riscv_th_msettilemi(4);
    __builtin_riscv_th_msettileki(4);
    __builtin_riscv_th_msettileni(4);

    // Load A into tr0
    __builtin_riscv_th_mlae8(__RVM_TR0, (void *)A, a_stride);

    // GEMM 1: B1 * A → acc0
    __builtin_riscv_th_mlbe8(__RVM_TR1, (void *)B1, b1_stride);
    __builtin_riscv_th_mzero(__RVM_ACC0);
    __builtin_riscv_th_mfmacc_h_e4(__RVM_ACC0, __RVM_TR1, __RVM_TR0);

    // GEMM 2: B2 * A → acc1 (different accumulator!)
    __builtin_riscv_th_mlbe8(__RVM_TR1, (void *)B2, b2_stride);
    __builtin_riscv_th_mzero(__RVM_ACC1);
    __builtin_riscv_th_mfmacc_h_e4(__RVM_ACC1, __RVM_TR1, __RVM_TR0);

    // Store both results
    __builtin_riscv_th_msce32(__RVM_ACC0, (void *)C1, c1_stride);
    __builtin_riscv_th_msce32(__RVM_ACC1, (void *)C2, c2_stride);

    __builtin_riscv_th_mrelease();
}
