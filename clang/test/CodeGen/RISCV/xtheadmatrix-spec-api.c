// XTHeadMatrix Spec-API (ManagedRA) end-to-end CodeGen test.
//
// Tests the public C API from <thead_matrix.h> using native matrix types
// (mint8_t, mint32_t, etc.) and Spec-API wrapper functions. Verifies that
// the wrapper functions correctly emit CSR configuration calls and
// _internal intrinsics with proper SSA dataflow.
//
// RUN: %clang_cc1 -O2 -triple riscv64 -target-feature +experimental-xtheadmatrix \
// RUN:   -emit-llvm -o - %s | FileCheck %s

#include <thead_matrix.h>

// --------------------------------------------------------------------
// Test 1: A-tile load + store round-trip
// --------------------------------------------------------------------
// CHECK-LABEL: @test_load_store_a
// CHECK: call void @llvm.riscv.th.msettilem
// CHECK: call void @llvm.riscv.th.msettilek
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mlae.internal32
// CHECK: call void @llvm.riscv.th.msettilem
// CHECK: call void @llvm.riscv.th.msettilen
// CHECK: call void @llvm.riscv.th.msce.internal32
void test_load_store_a(int32_t *base, long stride) {
    mint32_t tile = __riscv_th_mld_a_i32(base, stride, 4, 8);
    __riscv_th_mst_i32(base, stride, tile, 4, 4);
}

// --------------------------------------------------------------------
// Test 2: Full INT8 matmul pipeline (A-load + B-load + C-load + matmul + store)
// --------------------------------------------------------------------
// CHECK-LABEL: @test_int8_matmul
// CHECK: call void @llvm.riscv.th.msettilem
// CHECK: call void @llvm.riscv.th.msettilek
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mlae.internal8
// CHECK: call void @llvm.riscv.th.msettilek
// CHECK: call void @llvm.riscv.th.msettilen
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mlbe.internal8
// CHECK: call void @llvm.riscv.th.msettilem
// CHECK: call void @llvm.riscv.th.msettilen
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mlce.internal32
// CHECK: call void @llvm.riscv.th.msettilem
// CHECK: call void @llvm.riscv.th.msettilek
// CHECK: call void @llvm.riscv.th.msettilen
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mmacc.w.b.internal
// CHECK: call void @llvm.riscv.th.msettilem
// CHECK: call void @llvm.riscv.th.msettilen
// CHECK: call void @llvm.riscv.th.msce.internal32
void test_int8_matmul(int8_t *a, int8_t *b, int32_t *c, long stride,
                      mrow_t m, mcol_t k, mcol_t n) {
    mint8_t  ta = __riscv_th_mld_a_i8(a, stride, m, k);
    mint8_t  tb = __riscv_th_mld_b_i8(b, stride, k, n);
    mint32_t tc = __riscv_th_mld_acc_i32(c, stride, m, n);
    mint32_t result = __riscv_th_mmacc_w_b(tc, ta, tb, m, k, n);
    __riscv_th_mst_i32(c, stride, result, m, n);
}

// --------------------------------------------------------------------
// Test 3: Zero initialization
// --------------------------------------------------------------------
// CHECK-LABEL: @test_zero
// CHECK: call void @llvm.riscv.th.msettilem
// CHECK: call void @llvm.riscv.th.msettilen
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mzero.internal
// CHECK: call void @llvm.riscv.th.msettilem
// CHECK: call void @llvm.riscv.th.msettilen
// CHECK: call void @llvm.riscv.th.msce.internal32
void test_zero(int32_t *base, long stride) {
    mint32_t z = __riscv_th_mzeros_i32(4, 4);
    __riscv_th_mst_i32(base, stride, z, 4, 4);
}

// --------------------------------------------------------------------
// Test 4: FP32 native-precision matmul
// --------------------------------------------------------------------
// CHECK-LABEL: @test_fp32_matmul
// CHECK: call void @llvm.riscv.th.msettilem
// CHECK: call void @llvm.riscv.th.msettilek
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mlae.internal32
// CHECK: call void @llvm.riscv.th.msettilek
// CHECK: call void @llvm.riscv.th.msettilen
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mlbe.internal32
// CHECK: call void @llvm.riscv.th.msettilem
// CHECK: call void @llvm.riscv.th.msettilen
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mzero.internal
// CHECK: call void @llvm.riscv.th.msettilem
// CHECK: call void @llvm.riscv.th.msettilek
// CHECK: call void @llvm.riscv.th.msettilen
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmacc.s.internal
// CHECK: call void @llvm.riscv.th.msettilem
// CHECK: call void @llvm.riscv.th.msettilen
// CHECK: call void @llvm.riscv.th.msce.internal32
void test_fp32_matmul(float *a, float *b, float *c, long stride,
                      mrow_t m, mcol_t k, mcol_t n) {
    mfloat32_t ta  = __riscv_th_mld_a_f32(a, stride, m, k);
    mfloat32_t tb  = __riscv_th_mld_b_f32(b, stride, k, n);
    mfloat32_t acc = __riscv_th_mzeros_f32(m, n);
    mfloat32_t res = __riscv_th_mfmacc_s(acc, ta, tb, m, k, n);
    __riscv_th_mst_f32(c, stride, res, m, n);
}

// --------------------------------------------------------------------
// Test 5: Unsigned INT8 matmul (uu variant)
// --------------------------------------------------------------------
// CHECK-LABEL: @test_uint8_matmul
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mlae.internal8
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mlbe.internal8
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mzero.internal
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mmaccu.w.b.internal
// CHECK: call void @llvm.riscv.th.msce.internal32
void test_uint8_matmul(uint8_t *a, uint8_t *b, uint32_t *c, long stride,
                       mrow_t m, mcol_t k, mcol_t n) {
    muint8_t  ta  = __riscv_th_mld_a_u8(a, stride, m, k);
    muint8_t  tb  = __riscv_th_mld_b_u8(b, stride, k, n);
    muint32_t acc = __riscv_th_mzeros_u32(m, n);
    muint32_t res = __riscv_th_mmaccu_w_b(acc, ta, tb, m, k, n);
    __riscv_th_mst_u32(c, stride, res, m, n);
}

// --------------------------------------------------------------------
// Test 6: INT16 -> INT64 matmul
// --------------------------------------------------------------------
// CHECK-LABEL: @test_int16_matmul
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mlae.internal16
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mlbe.internal16
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mlce.internal64
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mmacc.d.h.internal
// CHECK: call void @llvm.riscv.th.msce.internal64
void test_int16_matmul(int16_t *a, int16_t *b, int64_t *c, long stride,
                       mrow_t m, mcol_t k, mcol_t n) {
    mint16_t ta  = __riscv_th_mld_a_i16(a, stride, m, k);
    mint16_t tb  = __riscv_th_mld_b_i16(b, stride, k, n);
    mint64_t acc = __riscv_th_mld_acc_i64(c, stride, m, n);
    mint64_t res = __riscv_th_mmacc_d_h(acc, ta, tb, m, k, n);
    __riscv_th_mst_i64(c, stride, res, m, n);
}

// --------------------------------------------------------------------
// Test 7: Shorthand alias
// --------------------------------------------------------------------
// CHECK-LABEL: @test_shorthand
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mmacc.w.b.internal
void test_shorthand(int8_t *a, int8_t *b, int32_t *c, long stride,
                    mrow_t m, mcol_t k, mcol_t n) {
    mint8_t  ta  = __riscv_th_mld_a_i8(a, stride, m, k);
    mint8_t  tb  = __riscv_th_mld_b_i8(b, stride, k, n);
    mint32_t acc = __riscv_th_mzeros_i32(m, n);
    // __riscv_th_mmacc is a shorthand for __riscv_th_mmacc_w_b
    mint32_t res = __riscv_th_mmacc(acc, ta, tb, m, k, n);
    __riscv_th_mst_i32(c, stride, res, m, n);
}

// --------------------------------------------------------------------
// Test 7: Element-wise integer operations
// --------------------------------------------------------------------
// CHECK-LABEL: @test_ew_int
// CHECK: call target("riscv.matrix") @llvm.riscv.th.madd.w.mm.internal
// CHECK: call target("riscv.matrix") @llvm.riscv.th.msub.w.mm.internal
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mmul.w.mm.internal
void test_ew_int(int32_t *base, long stride) {
    mint32_t a = __riscv_th_mld_a_i32(base, stride, 4, 4);
    mint32_t b = __riscv_th_mld_a_i32(base + 16, stride, 4, 4);
    mint32_t c = __riscv_th_mld_a_i32(base + 32, stride, 4, 4);
    mint32_t r1 = __riscv_th_madd_w_mm(a, b, c);
    mint32_t r2 = __riscv_th_msub_w_mm(r1, b, c);
    mint32_t r3 = __riscv_th_mmul_w_mm(r2, b, c);
    __riscv_th_mst_i32(base, stride, r3, 4, 4);
}

// --------------------------------------------------------------------
// Test 8: FP element-wise operations
// --------------------------------------------------------------------
// CHECK-LABEL: @test_ew_fp
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfadd.s.mm.internal
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmul.s.mm.internal
void test_ew_fp(float *base, long stride) {
    mfloat32_t a = __riscv_th_mld_a_f32(base, stride, 4, 4);
    mfloat32_t b = __riscv_th_mld_a_f32(base + 16, stride, 4, 4);
    mfloat32_t c = __riscv_th_mld_a_f32(base + 32, stride, 4, 4);
    mfloat32_t r1 = __riscv_th_mfadd_s_mm(a, b, c);
    mfloat32_t r2 = __riscv_th_mfmul_s_mm(r1, b, c);
    __riscv_th_mst_f32(base, stride, r2, 4, 4);
}

// --------------------------------------------------------------------
// Test 9: FP conversion
// --------------------------------------------------------------------
// CHECK-LABEL: @test_fp_cvt
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfcvtl.s.h.internal
void test_fp_cvt(uint16_t *base, long stride, float *out) {
    mfloat16_t src = __riscv_th_mld_a_f16(base, stride, 4, 4);
    mfloat32_t dst = __riscv_th_mfcvtl_s_h(src);
    __riscv_th_mst_f32(out, stride, dst, 4, 4);
}

// --------------------------------------------------------------------
// Test 10: Move and pack
// --------------------------------------------------------------------
// CHECK-LABEL: @test_move_pack
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mmov.mm.internal
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mpack.internal
void test_move_pack(int32_t *base, long stride) {
    mint32_t a = __riscv_th_mld_a_i32(base, stride, 4, 4);
    mint32_t b = __riscv_th_mmov_mm(a);
    mint32_t c = __riscv_th_mld_a_i32(base + 16, stride, 4, 4);
    mint32_t packed = __riscv_th_mpack(b, c);
    __riscv_th_mst_i32(base, stride, packed, 4, 4);
}

// --------------------------------------------------------------------
// Test 11: Slide and broadcast
// --------------------------------------------------------------------
// CHECK-LABEL: @test_slide_bcast
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mrslidedown.internal
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mrbca.mv.i.internal
void test_slide_bcast(int32_t *base, long stride) {
    mint32_t a = __riscv_th_mld_a_i32(base, stride, 4, 4);
    mint32_t slid = __riscv_th_mrslidedown(a, 1);
    mint32_t bcast = __riscv_th_mrbca(slid, 2);
    __riscv_th_mst_i32(base, stride, bcast, 4, 4);
}

// --------------------------------------------------------------------
// Test 12: N4clip
// --------------------------------------------------------------------
// CHECK-LABEL: @test_n4clip
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mn4clipl.w.mm.internal
void test_n4clip(int32_t *base, long stride) {
    mint32_t a = __riscv_th_mld_a_i32(base, stride, 4, 4);
    mint32_t b = __riscv_th_mld_a_i32(base + 16, stride, 4, 4);
    mint32_t c = __riscv_th_mld_a_i32(base + 32, stride, 4, 4);
    mint32_t r = __riscv_th_mn4clipl_w_mm(a, b, c);
    __riscv_th_mst_i32(base, stride, r, 4, 4);
}

// --------------------------------------------------------------------
// Test 13: Widening FP matmul — FP8 -> FP16 (h_e4)
// Verifies that A and B tile operands are distinct SSA values, not acc repeated.
// --------------------------------------------------------------------
// CHECK-LABEL: @test_widen_h_e4
// CHECK: %[[A:.+]] = {{.*}}call target("riscv.matrix") @llvm.riscv.th.mlae.internal32
// CHECK: %[[B:.+]] = {{.*}}call target("riscv.matrix") @llvm.riscv.th.mlbe.internal32
// CHECK: %[[ACC:.+]] = {{.*}}call target("riscv.matrix") @llvm.riscv.th.mzero.internal
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmacc.h.e4.internal{{.*}}(target("riscv.matrix") %[[ACC]], target("riscv.matrix") %[[B]], target("riscv.matrix") %[[A]])
void test_widen_h_e4(void *a, void *b, uint16_t *c, long stride,
                     mrow_t m, mcol_t k, mcol_t n) {
    mint32_t ta  = __riscv_th_mld_a_i32(a, stride, m, k);
    mint32_t tb  = __riscv_th_mld_b_i32(b, stride, k, n);
    mfloat16_t acc = __riscv_th_mzeros_f16(m, n);
    mfloat16_t res = __riscv_th_mfmacc_h_e4(acc, ta, tb, m, k, n);
    __riscv_th_mst_f16(c, stride, res, m, n);
}

// --------------------------------------------------------------------
// Test 14: Widening FP matmul — FP8 -> FP16 (h_e5)
// --------------------------------------------------------------------
// CHECK-LABEL: @test_widen_h_e5
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmacc.h.e5.internal
void test_widen_h_e5(void *a, void *b, uint16_t *c, long stride,
                     mrow_t m, mcol_t k, mcol_t n) {
    mint32_t ta  = __riscv_th_mld_a_i32(a, stride, m, k);
    mint32_t tb  = __riscv_th_mld_b_i32(b, stride, k, n);
    mfloat16_t acc = __riscv_th_mzeros_f16(m, n);
    mfloat16_t res = __riscv_th_mfmacc_h_e5(acc, ta, tb, m, k, n);
    __riscv_th_mst_f16(c, stride, res, m, n);
}

// --------------------------------------------------------------------
// Test 15: Widening FP matmul — BF16 -> FP16 (bf16_e4)
// --------------------------------------------------------------------
// CHECK-LABEL: @test_widen_bf16_e4
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmacc.bf16.e4.internal
void test_widen_bf16_e4(void *a, void *b, uint16_t *c, long stride,
                        mrow_t m, mcol_t k, mcol_t n) {
    mint32_t ta  = __riscv_th_mld_a_i32(a, stride, m, k);
    mint32_t tb  = __riscv_th_mld_b_i32(b, stride, k, n);
    mfloat16_t acc = __riscv_th_mzeros_f16(m, n);
    mfloat16_t res = __riscv_th_mfmacc_bf16_e4(acc, ta, tb, m, k, n);
    __riscv_th_mst_f16(c, stride, res, m, n);
}

// --------------------------------------------------------------------
// Test 16: Widening FP matmul — BF16 -> FP16 (bf16_e5)
// --------------------------------------------------------------------
// CHECK-LABEL: @test_widen_bf16_e5
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmacc.bf16.e5.internal
void test_widen_bf16_e5(void *a, void *b, uint16_t *c, long stride,
                        mrow_t m, mcol_t k, mcol_t n) {
    mint32_t ta  = __riscv_th_mld_a_i32(a, stride, m, k);
    mint32_t tb  = __riscv_th_mld_b_i32(b, stride, k, n);
    mfloat16_t acc = __riscv_th_mzeros_f16(m, n);
    mfloat16_t res = __riscv_th_mfmacc_bf16_e5(acc, ta, tb, m, k, n);
    __riscv_th_mst_f16(c, stride, res, m, n);
}

// --------------------------------------------------------------------
// Test 17: Widening FP matmul — BF16 -> FP32 (s_bf16)
// --------------------------------------------------------------------
// CHECK-LABEL: @test_widen_s_bf16
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmacc.s.bf16.internal
void test_widen_s_bf16(void *a, void *b, float *c, long stride,
                       mrow_t m, mcol_t k, mcol_t n) {
    mint32_t ta  = __riscv_th_mld_a_i32(a, stride, m, k);
    mint32_t tb  = __riscv_th_mld_b_i32(b, stride, k, n);
    mfloat32_t acc = __riscv_th_mzeros_f32(m, n);
    mfloat32_t res = __riscv_th_mfmacc_s_bf16(acc, ta, tb, m, k, n);
    __riscv_th_mst_f32(c, stride, res, m, n);
}

// --------------------------------------------------------------------
// Test 18: Widening FP matmul — FP8 -> FP32 (s_e4)
// --------------------------------------------------------------------
// CHECK-LABEL: @test_widen_s_e4
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmacc.s.e4.internal
void test_widen_s_e4(void *a, void *b, float *c, long stride,
                     mrow_t m, mcol_t k, mcol_t n) {
    mint32_t ta  = __riscv_th_mld_a_i32(a, stride, m, k);
    mint32_t tb  = __riscv_th_mld_b_i32(b, stride, k, n);
    mfloat32_t acc = __riscv_th_mzeros_f32(m, n);
    mfloat32_t res = __riscv_th_mfmacc_s_e4(acc, ta, tb, m, k, n);
    __riscv_th_mst_f32(c, stride, res, m, n);
}

// --------------------------------------------------------------------
// Test 19: Widening FP matmul — FP8 -> FP32 (s_e5)
// --------------------------------------------------------------------
// CHECK-LABEL: @test_widen_s_e5
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmacc.s.e5.internal
void test_widen_s_e5(void *a, void *b, float *c, long stride,
                     mrow_t m, mcol_t k, mcol_t n) {
    mint32_t ta  = __riscv_th_mld_a_i32(a, stride, m, k);
    mint32_t tb  = __riscv_th_mld_b_i32(b, stride, k, n);
    mfloat32_t acc = __riscv_th_mzeros_f32(m, n);
    mfloat32_t res = __riscv_th_mfmacc_s_e5(acc, ta, tb, m, k, n);
    __riscv_th_mst_f32(c, stride, res, m, n);
}

// --------------------------------------------------------------------
// Test 20: Widening FP matmul — TF32 -> FP32 (s_tf32)
// --------------------------------------------------------------------
// CHECK-LABEL: @test_widen_s_tf32
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmacc.s.tf32.internal
void test_widen_s_tf32(void *a, void *b, float *c, long stride,
                       mrow_t m, mcol_t k, mcol_t n) {
    mint32_t ta  = __riscv_th_mld_a_i32(a, stride, m, k);
    mint32_t tb  = __riscv_th_mld_b_i32(b, stride, k, n);
    mfloat32_t acc = __riscv_th_mzeros_f32(m, n);
    mfloat32_t res = __riscv_th_mfmacc_s_tf32(acc, ta, tb, m, k, n);
    __riscv_th_mst_f32(c, stride, res, m, n);
}

// --------------------------------------------------------------------
// Test: mget/mset tuple operations on x2 types
// --------------------------------------------------------------------
// At -O2, mget/mset struct ops are folded away; check the resulting dataflow.
// CHECK-LABEL: @test_mget_mset_f16
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mlae.internal16
// CHECK: call void @llvm.riscv.th.msce.internal16
void test_mget_mset_f16(uint16_t *ptr, long stride, mrow_t m, mcol_t k) {
    mfloat16_t a = __riscv_th_mld_a_f16(ptr, stride, m, k);
    mfloat16_t b = __riscv_th_mld_b_f16(ptr + 64, stride, m, k);
    mfloat16x2_t pair = __riscv_th_mset_f16(
        __riscv_th_mset_f16(__builtin_riscv_th_mundef_f16x2(), 0, a), 1, b);
    mfloat16_t got = __riscv_th_mget_f16(pair, 0);
    __riscv_th_mst_f16(ptr, stride, got, m, k);
}

// CHECK-LABEL: @test_native_fp16_matmul
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmacc.h.internal
void test_native_fp16_matmul(uint16_t *a_ptr, uint16_t *b_ptr,
                              uint16_t *c_ptr, long stride,
                              mrow_t m, mcol_t k, mcol_t n) {
    mfloat16_t ta = __riscv_th_mld_a_f16(a_ptr, stride, m, k);
    mfloat16_t tb = __riscv_th_mld_b_f16(b_ptr, stride, k, n);
    mfloat16_t acc = __riscv_th_mld_acc_f16(c_ptr, stride, m, n);
    mfloat16x2_t b_pair = __riscv_th_mset_f16(
        __builtin_riscv_th_mundef_f16x2(), 0, tb);
    mfloat16_t res = __riscv_th_mfmacc_h_x2(acc, ta, b_pair, m, k, n);
    __riscv_th_mst_f16(c_ptr, stride, res, m, n);
}

// CHECK-LABEL: @test_native_fp64_matmul
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mfmacc.d.internal
void test_native_fp64_matmul(double *a_ptr, double *b_ptr,
                              double *c_ptr, long stride,
                              mrow_t m, mcol_t k, mcol_t n) {
    mfloat64_t ta = __riscv_th_mld_a_f64(a_ptr, stride, m, k);
    mfloat64_t tb = __riscv_th_mld_b_f64(b_ptr, stride, k, n);
    mfloat64_t c0 = __riscv_th_mld_acc_f64(c_ptr, stride, m, n);
    mfloat64x2_t c_pair = __riscv_th_mset_f64(
        __builtin_riscv_th_mundef_f64x2(), 0, c0);
    mfloat64x2_t res_pair = __riscv_th_mfmacc_d_x2(c_pair, ta, tb, m, k, n);
    mfloat64_t res = __riscv_th_mget_f64(res_pair, 0);
    __riscv_th_mst_f64(c_ptr, stride, res, m, n);
}
