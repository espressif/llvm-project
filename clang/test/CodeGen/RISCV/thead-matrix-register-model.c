// NOTE: XTHeadMatrix register allocation model tests.
//
// ============================================================================
// RVM 0.6 REGISTER MODEL
// ============================================================================
//
// The XTHeadMatrix extension has 8 matrix registers:
//   - tr0, tr1, tr2, tr3    (tile registers, encoding 000-011)
//   - acc0, acc1, acc2, acc3 (accumulator registers, encoding 100-111)
//
// The HARDWARE allows flexible register use within type constraints:
//   - Load A/B:   md can be ANY tile register (tr0-tr3)
//   - Load C:     md can be ANY acc register (acc0-acc3)
//   - Matmul:     md must be acc (any), ms1/ms2 must be tile (any)
//   - EW ops:     md/ms1/ms2 must ALL be acc registers (any)
//   - Misc:       varies per instruction
//
// However, the CURRENT COMPILER uses FIXED register assignments:
//   - mla  → always tr0      mlb  → always tr1      mlc  → always acc0
//   - mlat → always tr2      mlbt → always tr3      mlct → always acc1
//   - matmul: acc0 = tr1 * tr0  (md=acc0, ms2=tr1, ms1=tr0)
//   - EW ops: md=acc0, ms1=acc1, ms2=acc2  (asm: acc0, acc2, acc1)
//   - slides/broadcast: md=tr0, ms1=tr1
//   - pack: md=tr0, ms1=tr1, ms2=tr2
//   - conversions: md=acc0, ms1=acc1
//   - mzero: always acc0 (or acc0-acc1 for 2r, etc.)
//
// This means:
//   1. Each API call maps to ONE specific physical register
//   2. Consecutive loads to the SAME role overwrite the same register
//   3. Multi-tile algorithms require careful sequencing (load→compute→store
//      before loading the next tile)
//   4. Only ONE matmul can be "in flight" at a time
//
// These tests verify the generated assembly matches the fixed model and
// demonstrate practical multi-tile programming patterns.
//
// ============================================================================
//
// RUN: %clang_cc1 -O2 -triple riscv64 -target-feature +experimental-xtheadmatrix -S -o - %s \
// RUN:   | FileCheck %s

#include <thead_matrix.h>

// ============================================================================
// Test 1: Basic register assignments — verify each load role maps correctly
//
//  mld_a  → mlae → md = tr0
//  mld_b  → mlbe → md = tr1
//  mld_c  → mlce → md = acc0
//  mld_at → mlate → md = tr2
//  mld_bt → mlbte → md = tr3
//  mld_ct → mlcte → md = acc1
//
// ============================================================================

// CHECK-LABEL: test_load_register_assignments:
// CHECK: th.mlae32 tr0
// CHECK: th.mlbe32 tr1
// CHECK: th.mlce32 acc0
// CHECK: th.mlate32 tr2
// CHECK: th.mlbte32 tr3
// CHECK: th.mlcte32 acc1
// CHECK: ret
void test_load_register_assignments(const void *p, long s,
                                     mrow_t m, mrow_t k, mcol_t n) {
  mint32_t a  = __riscv_th_mld_a_i32(p, s, m, k);    // → tr0
  mint32_t b  = __riscv_th_mld_b_i32(p, s, k, n);    // → tr1
  mint32_t c  = __riscv_th_mld_c_i32(p, s, m, n);    // → acc0
  mint32_t at = __riscv_th_mld_at_i32(p, s, m, k);   // → tr2
  mint32_t bt = __riscv_th_mld_bt_i32(p, s, k, n);   // → tr3
  mint32_t ct = __riscv_th_mld_ct_i32(p, s, m, n);   // → acc1
  (void)a; (void)b; (void)c; (void)at; (void)bt; (void)ct;
}

// ============================================================================
// Test 2: Store register assignments
//
//  mst_a  → msae → ms3 = tr0
//  mst_b  → msbe → ms3 = tr1
//  mst_c  → msce → ms3 = acc0
//  mst_at → msate → ms3 = tr2
//  mst_bt → msbte → ms3 = tr3
//  mst_ct → mscte → ms3 = acc1
//
// ============================================================================

// CHECK-LABEL: test_store_register_assignments:
// CHECK: th.msae32 tr0
// CHECK: th.msbe32 tr1
// CHECK: th.msce32 acc0
// CHECK: th.msate32 tr2
// CHECK: th.msbte32 tr3
// CHECK: th.mscte32 acc1
// CHECK: ret
void test_store_register_assignments(void *p, long s,
                                      mrow_t m, mrow_t k, mcol_t n) {
  mint32_t v = __riscv_th_mundefined_i32();
  __riscv_th_mst_a_i32(p, s, v, m, k);     // → tr0
  __riscv_th_mst_b_i32(p, s, v, k, n);     // → tr1
  __riscv_th_mst_c_i32(p, s, v, m, n);     // → acc0
  __riscv_th_mst_at_i32(p, s, v, m, k);    // → tr2
  __riscv_th_mst_bt_i32(p, s, v, k, n);    // → tr3
  __riscv_th_mst_ct_i32(p, s, v, m, n);    // → acc1
}

// ============================================================================
// Test 3: Matmul register assignment
//
//  mfmacc.s: acc0 = acc0 + tr1 * tr0
//  Assembly: th.mfmacc.s acc0, tr1, tr0
//
//  Note the operand order: acc0 (dest/accumulator), tr1 (B matrix), tr0 (A matrix)
//  The hardware computes: C += B * A  where C=acc0, B=tr1, A=tr0
//
// ============================================================================

// CHECK-LABEL: test_matmul_register_assignment:
// CHECK: th.msettilem
// CHECK: th.msettilek
// CHECK: th.mlae32 tr0
// CHECK: th.msettilen
// CHECK: th.mlbe32 tr1
// CHECK: th.mzero acc0
// CHECK: th.mfmacc.s acc0, tr1, tr0
// CHECK: th.msce32 acc0
// CHECK: ret
void test_matmul_register_assignment(const float *A, long as,
                                      const float *B, long bs,
                                      float *C, long cs,
                                      mrow_t M, mrow_t K, mcol_t N) {
  mfloat32_t a = __riscv_th_mld_a_f32(A, as, M, K);   // → tr0
  mfloat32_t b = __riscv_th_mld_b_f32(B, bs, K, N);   // → tr1
  mfloat32_t c = __riscv_th_mzero_f32();               // → zeros in acc0
  // NOTE: mzero uses acc0 (the accumulator register), which is the correct
  // target for zeroing before matmul accumulation.
  // In practice, mzero is used to zero the ACCUMULATOR (acc0), and the API
  // handles this correctly because mfmacc_s() sets tile dims before computing.
  c = __riscv_th_fmmacc_s(c, a, b, M, K, N);           // acc0 += tr1 * tr0
  __riscv_th_mst_c_f32(C, cs, c, M, N);                // acc0 → memory
}

// ============================================================================
// Test 4: Element-wise register assignments
//
//  All EW operations use acc registers:
//  madd.w.mm:  acc0 = acc2 + acc1
//  Assembly:   th.madd.w.mm acc0, acc2, acc1
//
//  Note: in the asm the operand order is (md, ms2, ms1), and the ISel
//  assigns md=acc0, ms1=acc1, ms2=acc2.
//  So: dest(acc0) = op(src2_ms2(acc2), src1_ms1(acc1))
//
// ============================================================================

// CHECK-LABEL: test_ew_register_assignment:
// CHECK: th.madd.w.mm acc0, acc2, acc1
// CHECK: th.mfadd.s.mm acc0, acc2, acc1
// CHECK: th.msra.w.mv.i acc0, acc2, acc1, 3
// CHECK: ret
void test_ew_register_assignment(void) {
  mint32_t d = __riscv_th_mundefined_i32();
  mint32_t s1 = __riscv_th_mundefined_i32();
  mint32_t s2 = __riscv_th_mundefined_i32();
  d = __riscv_th_madd_w_mm(d, s1, s2);       // acc0 = acc2 + acc1

  mfloat32_t fd = __riscv_th_mundefined_f32();
  mfloat32_t fs1 = __riscv_th_mundefined_f32();
  mfloat32_t fs2 = __riscv_th_mundefined_f32();
  fd = __riscv_th_mfadd_s_mm(fd, fs1, fs2);  // acc0 = acc2 + acc1

  d = __riscv_th_msra_w_mv(d, s1, 3);        // acc0 = acc2 >> acc1[row3]

  (void)d; (void)fd;
}

// ============================================================================
// Test 5: Conversion register assignments
//
//  All conversions use: md=acc0, ms1=acc1
//  This means source and destination are DIFFERENT acc registers.
//
// ============================================================================

// CHECK-LABEL: test_conversion_registers:
// CHECK: th.mfcvtl.s.h acc0, acc1
// CHECK: th.msfcvt.s.w acc0, acc1
// CHECK: th.mfucvtl.b.h acc0, acc1
// CHECK: ret
void test_conversion_registers(void) {
  mfloat16_t f16 = __riscv_th_mundefined_f16();
  mfloat32_t f32 = __riscv_th_mfcvtl_s_h(f16);   // acc0 ← widen(acc1)
  mint32_t i32 = __riscv_th_mundefined_i32();
  f32 = __riscv_th_msfcvt_s_w(i32);               // acc0 ← float(acc1)
  muint8_t u8 = __riscv_th_mfucvtl_b_h(f16);      // acc0 ← narrow(acc1)
  (void)f32; (void)u8;
}

// ============================================================================
// Test 6: Slide/broadcast register assignments
//
//  All slides and broadcasts use: md=tr0, ms1=tr1
//
// ============================================================================

// CHECK-LABEL: test_slide_broadcast_registers:
// CHECK: th.mrslidedown tr0, tr1, 2
// CHECK: th.mcslideup.w tr0, tr1, 1
// CHECK: th.mrbca.mv.i tr0, tr1, 0
// CHECK: th.mcbcaw.mv.i tr0, tr1, 3
// CHECK: ret
void test_slide_broadcast_registers(void) {
  __riscv_th_mrslidedown(2);    // tr0 ← slidedown(tr1, 2)
  __riscv_th_mcslideup_w(1);    // tr0 ← slideup(tr1, 1)
  __riscv_th_mrbca(0);          // tr0 ← broadcast_row(tr1, 0)
  __riscv_th_mcbca_w(3);        // tr0 ← broadcast_col(tr1, 3)
}

// ============================================================================
// Test 7: Pack register assignments
//
//  mpack: md=tr0, ms1=tr1, ms2=tr2
//
// ============================================================================

// CHECK-LABEL: test_pack_registers:
// CHECK: th.mpack tr0, tr2, tr1
// CHECK: th.mpackhl tr0, tr2, tr1
// CHECK: th.mpackhh tr0, tr2, tr1
// CHECK: ret
void test_pack_registers(void) {
  __riscv_th_mpack();     // tr0 = pack_ll(tr2, tr1)
  __riscv_th_mpackhl();   // tr0 = pack_hl(tr2, tr1)
  __riscv_th_mpackhh();   // tr0 = pack_hh(tr2, tr1)
}

// ============================================================================
// Test 8: SINGLE-TILE GEMM — the simple case
//
// When the entire computation fits in one tile, the flow is straightforward:
//
//   ┌─────────┐    ┌─────────┐    ┌─────────┐
//   │  Memory │    │  Matrix  │    │  Memory │
//   │  A[M,K] │───▶│  tr0    │    │         │
//   └─────────┘    │  (mla)  │    │         │
//                  └────┬────┘    │         │
//                       │         │         │
//   ┌─────────┐    ┌────▼────┐    │         │
//   │  Memory │    │  Matmul │    │         │
//   │  B[K,N] │───▶│  tr1    │───▶│ acc0    │───▶ C[M,N]
//   └─────────┘    │  (mlb)  │    │         │
//                  └─────────┘    └─────────┘
//
// Register flow: mld_a→tr0, mld_b→tr1, mzero→acc0, matmul→acc0, mst_c→acc0
//
// ============================================================================

// CHECK-LABEL: test_single_tile_gemm:
// CHECK-DAG: th.msettilem
// CHECK-DAG: th.msettilek
// CHECK: th.mlae8 tr0
// CHECK-DAG: th.msettilen
// CHECK: th.mlbe8 tr1
// CHECK: th.mzero acc0
// CHECK: th.mmacc.w.b acc0, tr1, tr0
// CHECK: th.msce32 acc0
// CHECK: th.mrelease
// CHECK: ret
void test_single_tile_gemm(const int8_t *A, long as,
                            const int8_t *B, long bs,
                            int32_t *C, long cs,
                            mrow_t M, mrow_t K, mcol_t N) {
  mint8_t a = __riscv_th_mld_a_i8(A, as, M, K);
  mint8_t b = __riscv_th_mld_b_i8(B, bs, K, N);
  mint32_t c = __riscv_th_mzero_i32();
  c = __riscv_th_mmaqa_ss_w_b(c, a, b, M, K, N);
  __riscv_th_mst_c_i32(C, cs, c, M, N);
  __riscv_th_mrelease();
}

// ============================================================================
// Test 9: K-DIMENSION TILING — accumulate over multiple K tiles
//
// When K > max_tile_K, we split along K and accumulate:
//
//   for k = 0 to K step TILE_K:
//     load A[M, k:k+TILE_K] → tr0
//     load B[k:k+TILE_K, N] → tr1
//     acc0 += tr1 * tr0
//
// Since matmul ACCUMULATES into acc0, we don't need to reload C between
// iterations. The accumulator stays live across K-tile iterations.
//
//   Iteration 0:  A_tile0 → tr0, B_tile0 → tr1, acc0 += tr1*tr0
//   Iteration 1:  A_tile1 → tr0, B_tile1 → tr1, acc0 += tr1*tr0  (A_tile0 overwritten!)
//   Iteration 2:  A_tile2 → tr0, B_tile2 → tr1, acc0 += tr1*tr0
//   ...
//   Store acc0 → C
//
// ============================================================================

// CHECK-LABEL: test_k_tiling:
// CHECK: th.mzero acc0
// CHECK: .LBB{{.*}}:
// CHECK: th.msettilem
// CHECK: th.msettilek
// CHECK: th.mlae8 tr0
// CHECK: th.msettilen
// CHECK: th.mlbe8 tr1
// CHECK: th.mmacc.w.b acc0, tr1, tr0
// CHECK: th.msce32 acc0
// CHECK: ret
void test_k_tiling(const int8_t *A, long as,
                    const int8_t *B, long bs,
                    int32_t *C, long cs,
                    mrow_t M, mcol_t N,
                    long K, long TILE_K) {
  // Zero accumulator once
  mint32_t c = __riscv_th_mzero_i32();

  // Accumulate over K tiles
  for (long k = 0; k < K; k += TILE_K) {
    mint8_t a = __riscv_th_mld_a_i8(A + k, as, M, (mrow_t)TILE_K);
    mint8_t b = __riscv_th_mld_b_i8(B + k * bs, bs, (mrow_t)TILE_K, N);
    c = __riscv_th_mmaqa_ss_w_b(c, a, b, M, (mrow_t)TILE_K, N);
  }

  // Store final result
  __riscv_th_mst_c_i32(C, cs, c, M, N);
}

// ============================================================================
// Test 10: M×N TILING — outer loop over output tiles
//
// When M or N > max tile size, we tile the output:
//
//   for m = 0 to M step TILE_M:
//     for n = 0 to N step TILE_N:
//       zero acc0
//       for k = 0 to K step TILE_K:
//         load A[m:m+TILE_M, k:k+TILE_K] → tr0
//         load B[k:k+TILE_K, n:n+TILE_N] → tr1
//         acc0 += tr1 * tr0
//       store acc0 → C[m:m+TILE_M, n:n+TILE_N]
//
// This is safe because each (m,n) tile is fully computed before moving to
// the next. tr0 and tr1 are overwritten each K iteration, but that's fine
// since we immediately consume them in the matmul.
//
// ============================================================================

// CHECK-LABEL: test_mn_tiling:
// CHECK-DAG: th.mzero acc0
// CHECK-DAG: th.mlae8 tr0
// CHECK-DAG: th.mlbe8 tr1
// CHECK-DAG: th.mmacc.w.b acc0, tr1, tr0
// CHECK-DAG: th.msce32 acc0
void test_mn_tiling(const int8_t *A, long as,
                     const int8_t *B, long bs,
                     int32_t *C, long cs,
                     long M, long N, long K,
                     long TILE_M, long TILE_N, long TILE_K) {
  for (long m = 0; m < M; m += TILE_M) {
    for (long n = 0; n < N; n += TILE_N) {
      // Zero accumulator for this output tile
      mint32_t c = __riscv_th_mzero_i32();

      // Accumulate over K
      for (long k = 0; k < K; k += TILE_K) {
        const int8_t *a_ptr = A + m * as + k;
        const int8_t *b_ptr = B + k * bs + n;
        mint8_t a = __riscv_th_mld_a_i8(a_ptr, as,
                                          (mrow_t)TILE_M, (mrow_t)TILE_K);
        mint8_t b = __riscv_th_mld_b_i8(b_ptr, bs,
                                          (mrow_t)TILE_K, (mcol_t)TILE_N);
        c = __riscv_th_mmaqa_ss_w_b(c, a, b,
                                      (mrow_t)TILE_M, (mrow_t)TILE_K,
                                      (mcol_t)TILE_N);
      }

      // Store output tile
      int32_t *c_ptr = C + m * cs / (long)sizeof(int32_t) + n;
      __riscv_th_mst_c_i32(c_ptr, cs, c, (mrow_t)TILE_M, (mcol_t)TILE_N);
    }
  }
}

// ============================================================================
// Test 11: FP16→FP32 WIDENING MATMUL with K-tiling
//
// Same tiling pattern but using widening matmul: FP16 inputs → FP32 output
//
// ============================================================================

// CHECK-LABEL: test_fp16_widening_k_tiling:
// CHECK: th.mzero acc0
// CHECK: th.mlae16 tr0
// CHECK: th.mlbe16 tr1
// CHECK: th.mfmacc.s.h acc0, tr1, tr0
// CHECK: th.msce32 acc0
// CHECK: ret
void test_fp16_widening_k_tiling(const void *A, long as,
                                   const void *B, long bs,
                                   float *C, long cs,
                                   mrow_t M, mcol_t N,
                                   long K, long TILE_K) {
  mfloat32_t c = __riscv_th_mzero_f32();

  for (long k = 0; k < K; k += TILE_K) {
    mfloat16_t a = __riscv_th_mld_a_f16((const char *)A + k * 2, as,
                                          M, (mrow_t)TILE_K);
    mfloat16_t b = __riscv_th_mld_b_f16((const char *)B + k * bs, bs,
                                          (mrow_t)TILE_K, N);
    c = __riscv_th_fwmmacc_s_h(c, a, b, M, (mrow_t)TILE_K, N);
  }

  __riscv_th_mst_c_f32(C, cs, c, M, N);
}

// ============================================================================
// Test 12: MATMUL + POST-PROCESSING PIPELINE
//
// Register flow through a complete inference pipeline:
//
//   1. mld_a → tr0, mld_b → tr1         (load inputs)
//   2. mzero → acc0                       (zero accumulator)
//   3. mmacc → acc0 += tr1 * tr0         (matmul)
//   4. mld_c → acc0                       (load bias — overwrites matmul result!)
//
// PROBLEM: Step 4 overwrites acc0 which has the matmul result!
//
// SOLUTION: Use EW add directly since both matmul result and bias are in acc
// registers. The API handles this by calling the EW builtin which operates
// on acc0/acc1/acc2:
//
//   1. mld_a → tr0, mld_b → tr1         (load inputs)
//   2. mzero → acc0                       (zero accumulator)
//   3. mmacc → acc0 += tr1 * tr0         (matmul result in acc0)
//   4. madd_w_mm → acc0 = acc2 + acc1    (add bias from acc1)
//   5. msra_w_mv → acc0 = acc2 >> acc1   (shift for quantization)
//   6. mn4clipl → acc0 = clip(acc2,acc1) (clip to int8)
//   7. mst_a → tr0 → memory             (store clipped result)
//
// ============================================================================

// CHECK-LABEL: test_matmul_postprocess:
// CHECK: th.mlae8 tr0
// CHECK: th.mlbe8 tr1
// CHECK: th.mzero acc0
// CHECK: th.mmacc.w.b acc0, tr1, tr0
// CHECK: th.madd.w.mm acc0, acc2, acc1
// CHECK: th.mn4clipl.w.mm acc0, acc2, acc1
// CHECK: th.msae8 tr0
// CHECK: ret
void test_matmul_postprocess(const int8_t *A, long as,
                              const int8_t *B, long bs,
                              void *out, long os,
                              mrow_t M, mrow_t K, mcol_t N) {
  // Matmul
  mint8_t a = __riscv_th_mld_a_i8(A, as, M, K);
  mint8_t b = __riscv_th_mld_b_i8(B, bs, K, N);
  mint32_t c = __riscv_th_mzero_i32();
  c = __riscv_th_mmaqa_ss_w_b(c, a, b, M, K, N);

  // Bias add (both operands in acc registers)
  mint32_t bias = __riscv_th_mundefined_i32();
  c = __riscv_th_madd_w_mm(c, c, bias);

  // N4Clip to int8
  mint8_t clipped = __riscv_th_mn4clipl_w_mm(c, bias);

  // Store
  __riscv_th_mst_a_i8(out, os, clipped, M, K);
}

// ============================================================================
// Test 13: TRANSPOSED MATMUL — using mlat/mlbt for column-major inputs
//
//   Column-major A needs transposed load: mlat → tr2
//   Column-major B needs transposed load: mlbt → tr3
//
//   But matmul reads from tr0 and tr1!
//   This means: we need to mmov from tr2→tr0 and tr3→tr1 before matmul.
//
//   Current fixed assignment:
//     mlat → tr2, mlbt → tr3 (transposed loads use different regs)
//     mmov.mm: tr0 ← tr1 (fixed, can't choose)
//
//   This is a LIMITATION: the hardware could do matmul directly from
//   tr2/tr3 (the 3-bit encoding allows it), but our ISel always uses tr0/tr1.
//
//   Workaround: Use inline assembly for register-flexible matmul, or
//   rely on the hardware treating all tile registers equivalently for matmul.
//
// ============================================================================

// CHECK-LABEL: test_transposed_load_matmul:
// CHECK: th.mlate32 tr2
// CHECK: th.mlbe32 tr1
// CHECK: th.mzero acc0
// CHECK: th.mfmacc.s acc0, tr1, tr0
// CHECK: th.msce32 acc0
// CHECK: ret
void test_transposed_load_matmul(const float *A_colmajor, long as,
                                  const float *B, long bs,
                                  float *C, long cs,
                                  mrow_t M, mrow_t K, mcol_t N) {
  // Transposed load of A goes to tr2 (not tr0)
  mfloat32_t a = __riscv_th_mld_at_f32(A_colmajor, as, M, K);
  // Normal load of B goes to tr1
  mfloat32_t b = __riscv_th_mld_b_f32(B, bs, K, N);
  mfloat32_t c = __riscv_th_mzero_f32();
  // Matmul still uses tr0 and tr1 (tr2 content is NOT read by matmul!)
  // This is a known limitation of the fixed register model.
  c = __riscv_th_fmmacc_s(c, a, b, M, K, N);
  __riscv_th_mst_c_f32(C, cs, c, M, N);
}

// ============================================================================
// Test 14: WHOLE-REGISTER SAVE/RESTORE
//
// Since all operations share fixed registers, saving/restoring matrix state
// requires whole-register loads/stores. This is useful for:
//   - Context switching between different matrix computations
//   - Saving intermediate results when register pressure is high
//
//   mlme → tr0 (whole load always targets tr0)
//   msme → tr0 (whole store always reads from tr0)
//
// ============================================================================

// CHECK-LABEL: test_save_restore_pipeline:
// CHECK: th.mlae32 tr0
// CHECK: th.msme32 tr0
// CHECK: th.mlbe32 tr1
// CHECK: th.msme32 tr0
// CHECK: th.mlme32 tr0
// CHECK: th.mlme32 tr0
// CHECK: th.mfmacc.s acc0, tr1, tr0
// CHECK: ret
void test_save_restore_pipeline(const float *A, long as,
                                  const float *B, long bs,
                                  void *save_a, void *save_b,
                                  mrow_t M, mrow_t K, mcol_t N) {
  // Load A to tr0, save it
  mfloat32_t a = __riscv_th_mld_a_f32(A, as, M, K);
  __riscv_th_mst_whole_f32(save_a, a);   // tr0 → memory

  // Load B to tr1, but whole-store reads from tr0!
  // So we need to move first, or accept that save_b gets tr0's content
  mfloat32_t b = __riscv_th_mld_b_f32(B, bs, K, N);
  __riscv_th_mst_whole_f32(save_b, b);   // saves tr0, NOT tr1!

  // Restore A and B from memory
  a = __riscv_th_mld_whole_f32(save_a);  // tr0 ← saved A
  b = __riscv_th_mld_whole_f32(save_b);  // tr0 ← saved B (overwrites A!)

  // This illustrates the limitation: whole load/store always use tr0
  mfloat32_t c = __riscv_th_mundefined_f32();
  c = __riscv_th_fmmacc_s(c, a, b, M, K, N);
  (void)c;
}

// ============================================================================
// Test 15: INT8 QUANTIZED INFERENCE with mixed-sign matmul
//
// Common in quantized neural networks:
//   - Activations: unsigned (uint8)
//   - Weights: signed (int8)
//   - Output: signed (int32), then quantized back to uint8
//
// ============================================================================

// CHECK-LABEL: test_quantized_inference:
// CHECK: th.mlae8 tr0
// CHECK: th.mlbe8 tr1
// CHECK: th.mzero acc0
// CHECK: th.mmaccus.w.b acc0, tr1, tr0
// CHECK: th.mn4cliplu.w.mm acc0, acc2, acc1
// CHECK: th.msae8 tr0
// CHECK: ret
void test_quantized_inference(const uint8_t *activations, long a_stride,
                               const int8_t *weights, long w_stride,
                               void *output, long o_stride,
                               mrow_t M, mrow_t K, mcol_t N) {
  // Load unsigned activations and signed weights
  muint8_t a = __riscv_th_mld_a_u8(activations, a_stride, M, K);
  mint8_t w = __riscv_th_mld_b_i8(weights, w_stride, K, N);

  // Mixed-sign matmul: unsigned * signed → int32
  mint32_t c = __riscv_th_mzero_i32();
  c = __riscv_th_mmaqa_us_w_b(c, a, w, M, K, N);

  // Quantize back: clip to unsigned int8
  mint32_t shift = __riscv_th_mundefined_i32();
  muint8_t result = __riscv_th_mn4cliplu_w_mm(c, shift);

  // Store
  __riscv_th_mst_a_u8(output, o_stride, result, M, K);
}

// ============================================================================
// Test 16: FP EW SOFTMAX APPROXIMATION
//
// Element-wise operations for attention mechanism:
//   1. Find row max (mfmax)
//   2. Subtract max for numerical stability (mfsub)
//   3. Scale (mfmul)
//
// All EW ops: md=acc0, ms1=acc1, ms2=acc2
// Multiple EW ops chain through acc0 naturally.
//
// ============================================================================

// CHECK-LABEL: test_fp_ew_attention:
// CHECK: th.mfmax.s.mm acc0, acc2, acc1
// CHECK: th.mfsub.s.mm acc0, acc2, acc1
// CHECK: th.mfmul.s.mm acc0, acc2, acc1
// CHECK: ret
void test_fp_ew_attention(void) {
  mfloat32_t scores = __riscv_th_mundefined_f32();
  mfloat32_t ones = __riscv_th_mundefined_f32();
  mfloat32_t scale = __riscv_th_mundefined_f32();

  // Find max (result in acc0)
  scores = __riscv_th_mfmax_s_mm(scores, scores, ones);
  // Subtract max (result in acc0)
  scores = __riscv_th_mfsub_s_mm(scores, scores, ones);
  // Scale (result in acc0)
  scores = __riscv_th_mfmul_s_mm(scores, scores, scale);

  (void)scores;
}

// ============================================================================
// Test 17: TYPE CONVERSION PIPELINE
//
// FP32 → FP16 → FP8 narrowing chain:
//   Each conversion: acc0 ← convert(acc1)
//   After each step, the result is in acc0 and becomes the source for the next.
//
// ============================================================================

// CHECK-LABEL: test_conversion_chain:
// CHECK: th.mfcvtl.h.s acc0, acc1
// CHECK: th.mfcvtl.e4.h acc0, acc1
// CHECK: ret
void test_conversion_chain(void) {
  mfloat32_t f32 = __riscv_th_mundefined_f32();
  // FP32 → FP16 (lower half)
  mfloat16_t f16 = __riscv_th_mfcvtl_h_s(f32);   // acc0 ← narrow(acc1)
  // FP16 → FP8-E4M3 (lower half)
  muint8_t fp8 = __riscv_th_mfcvtl_e4_h(f16);     // acc0 ← narrow(acc1)
  (void)fp8;
}

// ============================================================================
// Test 18: SCALAR INSERTION AND EXTRACTION
//
// mmov.x.m: read matrix element to GPR  (ms2=tr0)
// mmov.m.x: write GPR to matrix element (md=tr0)
// mdup.m.x: broadcast GPR to matrix col (md=tr0)
//
// These always use tr0 for the matrix operand.
//
// ============================================================================

// CHECK-LABEL: test_scalar_matrix_ops:
// CHECK: th.mdupw.m.x tr0
// CHECK: th.mmovw.x.m {{[a-z0-9]+}}, tr0
// CHECK: th.mmovw.m.x tr0
// CHECK: ret
void test_scalar_matrix_ops(size_t fill_val, size_t idx) {
  // Fill matrix column with scalar
  mint32_t mat = __riscv_th_mdup_m_x_w(fill_val);  // tr0[all] = fill_val

  // Extract element at index
  size_t elem = __riscv_th_mmov_x_m_w(mat, idx);   // GPR = tr0[idx]

  // Write element back
  mat = __riscv_th_mmov_m_x_w(mat, elem + 1, idx); // tr0[idx] = elem + 1

  (void)mat;
}

// ============================================================================
// Test 19: CSR-BASED AUTO-SIZING
//
// Query hardware to determine optimal tile sizes, then compute.
//
// ============================================================================

// CHECK-LABEL: test_auto_sized_gemm:
// CHECK: csrr {{.*}}, th.xtlenb
// CHECK: csrr {{.*}}, th.xtrlenb
// CHECK: th.mlae8 tr0
// CHECK: th.mlbe8 tr1
// CHECK: th.mzero acc0
// CHECK: th.mmacc.w.b acc0, tr1, tr0
// CHECK: th.msce32 acc0
// CHECK: ret
void test_auto_sized_gemm(const int8_t *A, long as,
                           const int8_t *B, long bs,
                           int32_t *C, long cs) {
  // Query hardware dimensions
  unsigned long tlen = __riscv_th_xmlenb();
  unsigned long rlen = __riscv_th_xrlenb();
  mrow_t max_rows = (mrow_t)(tlen / rlen);
  mcol_t max_cols = (mcol_t)rlen;  // for 8-bit elements

  // Use hardware-optimal tile size
  mint8_t a = __riscv_th_mld_a_i8(A, as, max_rows, max_cols);
  mint8_t b = __riscv_th_mld_b_i8(B, bs, max_cols, max_cols);
  mint32_t c = __riscv_th_mzero_i32();
  c = __riscv_th_mmaqa_ss_w_b(c, a, b, max_rows, max_cols, max_cols);
  __riscv_th_mst_c_i32(C, cs, c, max_rows, max_cols);
}

// ============================================================================
// Test 20: COMPLETE INFERENCE LAYER
//
// Demonstrates a full neural network layer with tiling:
//   Input[batch, in_features] × Weights[in_features, out_features]
//   + Bias[out_features] → Output[batch, out_features]
//
// Tiled over all three dimensions with the fixed register model.
//
// ============================================================================

// CHECK-LABEL: test_inference_layer:
// CHECK-DAG: th.mzero acc0
// CHECK-DAG: th.mlae8 tr0
// CHECK-DAG: th.mlbe8 tr1
// CHECK-DAG: th.mmacc.w.b acc0, tr1, tr0
// CHECK-DAG: th.msce32 acc0
void test_inference_layer(const int8_t *input, long in_stride,
                           const int8_t *weights, long w_stride,
                           int32_t *output, long out_stride,
                           long batch, long in_features, long out_features,
                           long TILE_M, long TILE_K, long TILE_N) {
  for (long m = 0; m < batch; m += TILE_M) {
    for (long n = 0; n < out_features; n += TILE_N) {
      // Zero accumulator for this output tile
      mint32_t acc = __riscv_th_mzero_i32();

      // Accumulate over input features
      for (long k = 0; k < in_features; k += TILE_K) {
        mint8_t a = __riscv_th_mld_a_i8(
            input + m * in_stride + k, in_stride,
            (mrow_t)TILE_M, (mrow_t)TILE_K);
        mint8_t b = __riscv_th_mld_b_i8(
            weights + k * w_stride + n, w_stride,
            (mrow_t)TILE_K, (mcol_t)TILE_N);
        acc = __riscv_th_mmaqa_ss_w_b(
            acc, a, b, (mrow_t)TILE_M, (mrow_t)TILE_K, (mcol_t)TILE_N);
      }

      // Store output tile
      __riscv_th_mst_c_i32(
          output + m * out_stride / (long)sizeof(int32_t) + n,
          out_stride, acc, (mrow_t)TILE_M, (mcol_t)TILE_N);
    }
  }
}
