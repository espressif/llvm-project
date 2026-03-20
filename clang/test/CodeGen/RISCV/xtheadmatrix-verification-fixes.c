// Tests for verification round 8 fixes in thead_matrix.h:
//   1. mreinterpret_* preserves data (inline asm with tied constraints)
//   2. xmisa/xmsize CSR functions
//   3. x2 matmul operates on both elements of the pair
//   4. mzero spec-compatible aliases
//
// RUN: %clang_cc1 -O0 -triple riscv64 -target-feature +experimental-xtheadmatrix \
// RUN:   -emit-llvm -o - %s | FileCheck --check-prefix=O0 %s
// RUN: %clang_cc1 -O2 -triple riscv64 -target-feature +experimental-xtheadmatrix \
// RUN:   -emit-llvm -o - %s | FileCheck --check-prefix=O2 %s

#include <thead_matrix.h>

// ========================================================================
// 1. mreinterpret: bitwise type reinterpretation preserves register data
// ========================================================================

// Verify that mreinterpret emits inline asm with tied constraint ("0"),
// NOT a call to mundef (which would discard the data).

// O0-LABEL: @test_mreinterpret_i32_to_f32
// O0-NOT: poison
// O0: call target("riscv.matrix") asm "", "=^tr,0"(target("riscv.matrix") %
// O0: ret target("riscv.matrix")
mfloat32_t test_mreinterpret_i32_to_f32(mint32_t src) {
    return __riscv_th_mreinterpret_f32(src);
}

// O0-LABEL: @test_mreinterpret_f16_to_i8
// O0-NOT: poison
// O0: call target("riscv.matrix") asm "", "=^tr,0"(target("riscv.matrix") %
mint8_t test_mreinterpret_f16_to_i8(mfloat16_t src) {
    return __riscv_th_mreinterpret_i8(src);
}

// O0-LABEL: @test_mreinterpret_u32_to_i64
// O0-NOT: poison
// O0: call target("riscv.matrix") asm "", "=^tr,0"(target("riscv.matrix") %
mint64_t test_mreinterpret_u32_to_i64(muint32_t src) {
    return __riscv_th_mreinterpret_i64(src);
}

// O0-LABEL: @test_mreinterpret_f64_to_u16
// O0-NOT: poison
// O0: call target("riscv.matrix") asm "", "=^tr,0"(target("riscv.matrix") %
muint16_t test_mreinterpret_f64_to_u16(mfloat64_t src) {
    return __riscv_th_mreinterpret_u16(src);
}

// Test all unsigned single variants
// O0-LABEL: @test_mreinterpret_to_u8
// O0: asm "", "=^tr,0"
muint8_t test_mreinterpret_to_u8(mint32_t src) {
    return __riscv_th_mreinterpret_u8(src);
}

// O0-LABEL: @test_mreinterpret_to_u32
// O0: asm "", "=^tr,0"
muint32_t test_mreinterpret_to_u32(mint8_t src) {
    return __riscv_th_mreinterpret_u32(src);
}

// O0-LABEL: @test_mreinterpret_to_u64
// O0: asm "", "=^tr,0"
muint64_t test_mreinterpret_to_u64(mfloat32_t src) {
    return __riscv_th_mreinterpret_u64(src);
}

// O0-LABEL: @test_mreinterpret_to_f16
// O0: asm "", "=^tr,0"
mfloat16_t test_mreinterpret_to_f16(muint32_t src) {
    return __riscv_th_mreinterpret_f16(src);
}

// O0-LABEL: @test_mreinterpret_to_f64
// O0: asm "", "=^tr,0"
mfloat64_t test_mreinterpret_to_f64(mint16_t src) {
    return __riscv_th_mreinterpret_f64(src);
}

// O0-LABEL: @test_mreinterpret_to_i16
// O0: asm "", "=^tr,0"
mint16_t test_mreinterpret_to_i16(mfloat64_t src) {
    return __riscv_th_mreinterpret_i16(src);
}

// O0-LABEL: @test_mreinterpret_to_i32
// O0: asm "", "=^tr,0"
mint32_t test_mreinterpret_to_i32(mfloat16_t src) {
    return __riscv_th_mreinterpret_i32(src);
}

// Test x2 reinterpret variants
// O0-LABEL: @test_mreinterpret_x2_i32_to_f32
// O0: asm "", "=^tr,0"
mfloat32x2_t test_mreinterpret_x2_i32_to_f32(mint32x2_t src) {
    return __riscv_th_mreinterpret_f32x2(src);
}

// O0-LABEL: @test_mreinterpret_x2_f16_to_u8
// O0: asm "", "=^tr,0"
muint8x2_t test_mreinterpret_x2_f16_to_u8(mfloat16x2_t src) {
    return __riscv_th_mreinterpret_u8x2(src);
}

// O0-LABEL: @test_mreinterpret_x2_u64_to_i16
// O0: asm "", "=^tr,0"
mint16x2_t test_mreinterpret_x2_u64_to_i16(muint64x2_t src) {
    return __riscv_th_mreinterpret_i16x2(src);
}

// Verify round-trip: reinterpret preserves value through load -> reinterpret -> store
// O2-LABEL: @test_mreinterpret_roundtrip
// O2: call target("riscv.matrix") @llvm.riscv.th.mlae.internal8
// O2: call target("riscv.matrix") asm "", "=^tr,0"
// O2: call void @llvm.riscv.th.msce.internal32
void test_mreinterpret_roundtrip(int8_t *in, int32_t *out, long stride,
                                  mrow_t m, mcol_t k, mcol_t n) {
    mint8_t loaded = __riscv_th_mld_a_i8(in, stride, m, k);
    mint32_t reinterpreted = __riscv_th_mreinterpret_i32(loaded);
    __riscv_th_mst_i32(out, stride, reinterpreted, m, n);
}

// ========================================================================
// 2. xmisa / xmsize CSR functions
// ========================================================================

// O0-LABEL: @test_xmisa
// O0: call i64 asm sideeffect "csrr $0, th.xmisa", "=r"()
unsigned long test_xmisa(void) {
    return __riscv_th_xmisa();
}

// xmsize should be a compatibility alias that also reads xmisa
// O0-LABEL: @test_xmsize
// O0: call i64 asm sideeffect "csrr $0, th.xmisa", "=r"()
unsigned long test_xmsize(void) {
    return __riscv_th_xmsize();
}

// Verify both return the same value
// O2-LABEL: @test_xmisa_eq_xmsize
// O2: call i64 asm sideeffect "csrr $0, th.xmisa"
// O2: call i64 asm sideeffect "csrr $0, th.xmisa"
int test_xmisa_eq_xmsize(void) {
    return __riscv_th_xmisa() == __riscv_th_xmsize();
}

// Other CSR functions still work
// O0-LABEL: @test_xmlenb
// O0: call i64 asm sideeffect "csrr $0, th.xtlenb", "=r"()
unsigned long test_xmlenb(void) {
    return __riscv_th_xmlenb();
}

// O0-LABEL: @test_xrlenb
// O0: call i64 asm sideeffect "csrr $0, th.xtrlenb", "=r"()
unsigned long test_xrlenb(void) {
    return __riscv_th_xrlenb();
}

// ========================================================================
// 3. x2 matmul: both elements of the pair are processed
// ========================================================================

// The key verification: x2 matmul should emit TWO intrinsic calls,
// not just one. Each call processes one element of the pair.

// --- INT16->INT64 x2: verify two mmacc.d.h calls ---

// O0-LABEL: @test_x2_int_ss_both_elements
// O0: call target("riscv.matrix") @llvm.riscv.th.mmacc.d.h.internal
// O0: call target("riscv.matrix") @llvm.riscv.th.mmacc.d.h.internal
mint64x2_t test_x2_int_ss_both_elements(mint64x2_t c, mint16_t a, mint16_t b,
                                          mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mmacc_d_h_x2(c, a, b, m, k, n);
}

// O0-LABEL: @test_x2_int_uu_both_elements
// O0: call target("riscv.matrix") @llvm.riscv.th.mmaccu.d.h.internal
// O0: call target("riscv.matrix") @llvm.riscv.th.mmaccu.d.h.internal
muint64x2_t test_x2_int_uu_both_elements(muint64x2_t c, muint16_t a,
                                           muint16_t b,
                                           mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mmaccu_d_h_x2(c, a, b, m, k, n);
}

// O0-LABEL: @test_x2_int_us_both_elements
// O0: call target("riscv.matrix") @llvm.riscv.th.mmaccus.d.h.internal
// O0: call target("riscv.matrix") @llvm.riscv.th.mmaccus.d.h.internal
mint64x2_t test_x2_int_us_both_elements(mint64x2_t c, muint16_t a,
                                          mint16_t b,
                                          mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mmaccus_d_h_x2(c, a, b, m, k, n);
}

// O0-LABEL: @test_x2_int_su_both_elements
// O0: call target("riscv.matrix") @llvm.riscv.th.mmaccsu.d.h.internal
// O0: call target("riscv.matrix") @llvm.riscv.th.mmaccsu.d.h.internal
mint64x2_t test_x2_int_su_both_elements(mint64x2_t c, mint16_t a,
                                          muint16_t b,
                                          mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mmaccsu_d_h_x2(c, a, b, m, k, n);
}

// --- FP64 x2: verify two mfmacc.d calls ---

// O0-LABEL: @test_x2_fp64_both_elements
// O0: call target("riscv.matrix") @llvm.riscv.th.mfmacc.d.internal
// O0: call target("riscv.matrix") @llvm.riscv.th.mfmacc.d.internal
mfloat64x2_t test_x2_fp64_both_elements(mfloat64x2_t c, mfloat64_t a,
                                          mfloat64_t b,
                                          mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mfmacc_d_x2(c, a, b, m, k, n);
}

// --- FP64 widening (FP32->FP64) x2: verify two mfmacc.d.s calls ---

// O0-LABEL: @test_x2_fp64_widen_both_elements
// O0: call target("riscv.matrix") @llvm.riscv.th.mfmacc.d.s.internal
// O0: call target("riscv.matrix") @llvm.riscv.th.mfmacc.d.s.internal
mfloat64x2_t test_x2_fp64_widen_both_elements(mfloat64x2_t c, mfloat32_t a,
                                                mfloat32_t b,
                                                mrow_t m, mcol_t k,
                                                mcol_t n) {
    return __riscv_th_mfmacc_d_s_x2(c, a, b, m, k, n);
}

// End-to-end: load both elements, x2 matmul, store both elements
// O2-LABEL: @test_x2_e2e_both_elements
// O2: call target("riscv.matrix") @llvm.riscv.th.mmacc.d.h.internal
// O2: call target("riscv.matrix") @llvm.riscv.th.mmacc.d.h.internal
void test_x2_e2e_both_elements(int16_t *a_ptr, int16_t *b_ptr,
                                int64_t *c0_ptr, int64_t *c1_ptr,
                                long stride,
                                mrow_t m, mcol_t k, mcol_t n) {
    mint16_t ta = __riscv_th_mld_a_i16(a_ptr, stride, m, k);
    mint16_t tb = __riscv_th_mld_b_i16(b_ptr, stride, k, n);
    mint64_t c0 = __riscv_th_mld_acc_i64(c0_ptr, stride, m, n);
    mint64_t c1 = __riscv_th_mld_acc_i64(c1_ptr, stride, m, n);
    mint64x2_t c_pair = __riscv_th_mset_i64(
        __riscv_th_mset_i64(__builtin_riscv_th_mundef_i64x2(), 0, c0),
        1, c1);
    mint64x2_t res = __riscv_th_mmacc_d_h_x2(c_pair, ta, tb, m, k, n);
    mint64_t r0 = __riscv_th_mget_i64(res, 0);
    mint64_t r1 = __riscv_th_mget_i64(res, 1);
    __riscv_th_mst_i64(c0_ptr, stride, r0, m, n);
    __riscv_th_mst_i64(c1_ptr, stride, r1, m, n);
}

// ========================================================================
// 4. mzero spec-compatible aliases
// ========================================================================

// mzero_i32 alias produces same IR as mzeros_i32
// O0-LABEL: @test_mzero_alias_i32
// O0: call target("riscv.matrix") @llvm.riscv.th.mzero.internal
mint32_t test_mzero_alias_i32(mrow_t m, mcol_t n) {
    return __riscv_th_mzero_i32(m, n);
}

// O0-LABEL: @test_mzero_alias_f16
// O0: call target("riscv.matrix") @llvm.riscv.th.mzero.internal
mfloat16_t test_mzero_alias_f16(mrow_t m, mcol_t n) {
    return __riscv_th_mzero_f16(m, n);
}

// O0-LABEL: @test_mzero_alias_i8
// O0: call target("riscv.matrix") @llvm.riscv.th.mzero.internal
mint8_t test_mzero_alias_i8(mrow_t m, mcol_t n) {
    return __riscv_th_mzero_i8(m, n);
}

// O0-LABEL: @test_mzero_alias_u64
// O0: call target("riscv.matrix") @llvm.riscv.th.mzero.internal
muint64_t test_mzero_alias_u64(mrow_t m, mcol_t n) {
    return __riscv_th_mzero_u64(m, n);
}

// O0-LABEL: @test_mzero_alias_f64
// O0: call target("riscv.matrix") @llvm.riscv.th.mzero.internal
mfloat64_t test_mzero_alias_f64(mrow_t m, mcol_t n) {
    return __riscv_th_mzero_f64(m, n);
}

// Verify alias produces identical IR to original name
// O2-LABEL: @test_mzero_alias_eq_mzeros
// O2: call target("riscv.matrix") @llvm.riscv.th.mzero.internal
// O2: call target("riscv.matrix") @llvm.riscv.th.mzero.internal
// O2: call void @llvm.riscv.th.msce.internal32
// O2: call void @llvm.riscv.th.msce.internal32
void test_mzero_alias_eq_mzeros(int32_t *out1, int32_t *out2, long stride,
                                 mrow_t m, mcol_t n) {
    mint32_t z1 = __riscv_th_mzero_i32(m, n);
    mint32_t z2 = __riscv_th_mzeros_i32(m, n);
    __riscv_th_mst_i32(out1, stride, z1, m, n);
    __riscv_th_mst_i32(out2, stride, z2, m, n);
}

// ========================================================================
// 5. Additional coverage: transposed loads/stores, whole-register ops
// ========================================================================

// Transposed A-tile load
// O0-LABEL: @test_transposed_a_load
// O0: call target("riscv.matrix") @llvm.riscv.th.mlate.internal32
mint32_t test_transposed_a_load(int32_t *base, long stride, mrow_t m,
                                 mcol_t k) {
    return __riscv_th_mld_at_i32(base, stride, m, k);
}

// Transposed B-tile load
// O0-LABEL: @test_transposed_b_load
// O0: call target("riscv.matrix") @llvm.riscv.th.mlbte.internal16
mfloat16_t test_transposed_b_load(uint16_t *base, long stride, mcol_t k,
                                    mcol_t n) {
    return __riscv_th_mld_bt_f16(base, stride, k, n);
}

// Transposed C-tile load
// O0-LABEL: @test_transposed_c_load
// O0: call target("riscv.matrix") @llvm.riscv.th.mlcte.internal64
mint64_t test_transposed_c_load(int64_t *base, long stride, mrow_t m,
                                  mcol_t n) {
    return __riscv_th_mld_ct_i64(base, stride, m, n);
}

// A-tile store
// O0-LABEL: @test_a_tile_store
// O0: call void @llvm.riscv.th.msae.internal8
void test_a_tile_store(int8_t *base, long stride, mint8_t val, mrow_t m,
                        mcol_t k) {
    __riscv_th_mst_a_i8(base, stride, val, m, k);
}

// B-tile store
// O0-LABEL: @test_b_tile_store
// O0: call void @llvm.riscv.th.msbe.internal16
void test_b_tile_store(int16_t *base, long stride, mint16_t val, mcol_t k,
                        mcol_t n) {
    __riscv_th_mst_b_i16(base, stride, val, k, n);
}

// Transposed A-tile store
// O0-LABEL: @test_transposed_a_store
// O0: call void @llvm.riscv.th.msate.internal32
void test_transposed_a_store(int32_t *base, long stride, mint32_t val,
                              mrow_t m, mcol_t k) {
    __riscv_th_mst_at_i32(base, stride, val, m, k);
}

// Transposed B-tile store
// O0-LABEL: @test_transposed_b_store
// O0: call void @llvm.riscv.th.msbte.internal64
void test_transposed_b_store(int64_t *base, long stride, mint64_t val,
                              mcol_t k, mcol_t n) {
    __riscv_th_mst_bt_i64(base, stride, val, k, n);
}

// Transposed C-tile store
// O0-LABEL: @test_transposed_c_store
// O0: call void @llvm.riscv.th.mscte.internal32
void test_transposed_c_store(int32_t *base, long stride, mfloat32_t val,
                              mrow_t m, mcol_t n) {
    __riscv_th_mst_ct_f32(base, stride, val, m, n);
}

// Whole-register load
// O0-LABEL: @test_whole_register_load
// O0: call target("riscv.matrix") @llvm.riscv.th.mlme.internal32
mint32_t test_whole_register_load(int32_t *base, long stride) {
    return __riscv_th_mld_m_i32(base, stride);
}

// Whole-register store
// O0-LABEL: @test_whole_register_store
// O0: call void @llvm.riscv.th.msme.internal32
void test_whole_register_store(int32_t *base, long stride, mint32_t val) {
    __riscv_th_mst_m_i32(base, stride, val);
}

// ========================================================================
// 6. Move, duplicate, pack, slide, broadcast
// ========================================================================

// O0-LABEL: @test_mmov_mm
// O0: call target("riscv.matrix") @llvm.riscv.th.mmov.mm.internal
mint32_t test_mmov_mm(mint32_t src) {
    return __riscv_th_mmov_mm(src);
}

// O0-LABEL: @test_mmov_extract_w
// O0: call i64 @llvm.riscv.th.mmovw.x.m.internal
unsigned long test_mmov_extract_w(mint32_t src, unsigned long idx) {
    return __riscv_th_mmovw_x_m(src, idx);
}

// O0-LABEL: @test_mmov_insert_h
// O0: call target("riscv.matrix") @llvm.riscv.th.mmovh.m.x.internal
mint32_t test_mmov_insert_h(mint32_t dst, unsigned long data,
                             unsigned long idx) {
    return __riscv_th_mmovh_m_x(dst, data, idx);
}

// O0-LABEL: @test_mdup_w
// O0: call target("riscv.matrix") @llvm.riscv.th.mdupw.m.x.internal
mint32_t test_mdup_w(mint32_t dst, unsigned long data) {
    return __riscv_th_mdupw_m_x(dst, data);
}

// O0-LABEL: @test_mpack
// O0: call target("riscv.matrix") @llvm.riscv.th.mpack.internal
mint32_t test_mpack(mint32_t s2, mint32_t s1) {
    return __riscv_th_mpack(s2, s1);
}

// O0-LABEL: @test_mpackhl
// O0: call target("riscv.matrix") @llvm.riscv.th.mpackhl.internal
mint32_t test_mpackhl(mint32_t s2, mint32_t s1) {
    return __riscv_th_mpackhl(s2, s1);
}

// O0-LABEL: @test_mpackhh
// O0: call target("riscv.matrix") @llvm.riscv.th.mpackhh.internal
mint32_t test_mpackhh(mint32_t s2, mint32_t s1) {
    return __riscv_th_mpackhh(s2, s1);
}

// O0-LABEL: @test_mrslidedown
// O0: call target("riscv.matrix") @llvm.riscv.th.mrslidedown.internal
mint32_t test_mrslidedown(mint32_t src) {
    return __riscv_th_mrslidedown(src, 2);
}

// O0-LABEL: @test_mrslideup
// O0: call target("riscv.matrix") @llvm.riscv.th.mrslideup.internal
mint32_t test_mrslideup(mint32_t src) {
    return __riscv_th_mrslideup(src, 1);
}

// O0-LABEL: @test_mcslidedown_w
// O0: call target("riscv.matrix") @llvm.riscv.th.mcslidedown.w.internal
mint32_t test_mcslidedown_w(mint32_t src) {
    return __riscv_th_mcslidedown_w(src, 3);
}

// O0-LABEL: @test_mcslideup_b
// O0: call target("riscv.matrix") @llvm.riscv.th.mcslideup.b.internal
mint32_t test_mcslideup_b(mint32_t src) {
    return __riscv_th_mcslideup_b(src, 1);
}

// O0-LABEL: @test_mrbca
// O0: call target("riscv.matrix") @llvm.riscv.th.mrbca.mv.i.internal
mint32_t test_mrbca(mint32_t src) {
    return __riscv_th_mrbca(src, 0);
}

// O0-LABEL: @test_mcbca_w
// O0: call target("riscv.matrix") @llvm.riscv.th.mcbcaw.mv.i.internal
mint32_t test_mcbca_w(mint32_t src) {
    return __riscv_th_mcbca_w(src, 2);
}

// ========================================================================
// 7. FP and float-int conversions
// ========================================================================

// O0-LABEL: @test_mfcvtl_s_h
// O0: call target("riscv.matrix") @llvm.riscv.th.mfcvtl.s.h.internal
mfloat32_t test_mfcvtl_s_h(mfloat16_t src) {
    return __riscv_th_mfcvtl_s_h(src);
}

// O0-LABEL: @test_mfcvth_h_s
// O0: call target("riscv.matrix") @llvm.riscv.th.mfcvth.h.s.internal
mfloat16_t test_mfcvth_h_s(mfloat32_t src) {
    return __riscv_th_mfcvth_h_s(src);
}

// O0-LABEL: @test_mfcvtl_d_s
// O0: call target("riscv.matrix") @llvm.riscv.th.mfcvtl.d.s.internal
mfloat64_t test_mfcvtl_d_s(mfloat32_t src) {
    return __riscv_th_mfcvtl_d_s(src);
}

// O0-LABEL: @test_msfcvt_s_w
// O0: call target("riscv.matrix") @llvm.riscv.th.msfcvt.s.w.internal
mfloat32_t test_msfcvt_s_w(mint32_t src) {
    return __riscv_th_msfcvt_s_w(src);
}

// O0-LABEL: @test_mfscvt_w_s
// O0: call target("riscv.matrix") @llvm.riscv.th.mfscvt.w.s.internal
mint32_t test_mfscvt_w_s(mfloat32_t src) {
    return __riscv_th_mfscvt_w_s(src);
}

// ========================================================================
// 8. Element-wise operations
// ========================================================================

// INT element-wise .mm
// O0-LABEL: @test_madd_w_mm
// O0: call target("riscv.matrix") @llvm.riscv.th.madd.w.mm.internal
mint32_t test_madd_w_mm(mint32_t acc, mint32_t s2, mint32_t s1) {
    return __riscv_th_madd_w_mm(acc, s2, s1);
}

// INT element-wise .mv.i
// O0-LABEL: @test_msub_w_mv_i
// O0: call target("riscv.matrix") @llvm.riscv.th.msub.w.mv.i.internal
mint32_t test_msub_w_mv_i(mint32_t acc, mint32_t s2, mint32_t s1) {
    return __riscv_th_msub_w_mv_i(acc, s2, s1, 3);
}

// FP element-wise .mm
// O0-LABEL: @test_mfadd_s_mm
// O0: call target("riscv.matrix") @llvm.riscv.th.mfadd.s.mm.internal
mfloat32_t test_mfadd_s_mm(mfloat32_t acc, mfloat32_t s2, mfloat32_t s1) {
    return __riscv_th_mfadd_s_mm(acc, s2, s1);
}

// FP element-wise .mv.i
// O0-LABEL: @test_mfmin_h_mv_i
// O0: call target("riscv.matrix") @llvm.riscv.th.mfmin.h.mv.i.internal
mfloat16_t test_mfmin_h_mv_i(mfloat16_t acc, mfloat16_t s2, mfloat16_t s1) {
    return __riscv_th_mfmin_h_mv_i(acc, s2, s1, 1);
}

// N4clip .mm
// O0-LABEL: @test_mn4clipl_w_mm
// O0: call target("riscv.matrix") @llvm.riscv.th.mn4clipl.w.mm.internal
mint32_t test_mn4clipl_w_mm(mint32_t acc, mint32_t s2, mint32_t s1) {
    return __riscv_th_mn4clipl_w_mm(acc, s2, s1);
}

// ========================================================================
// 9. Widening FP matmul (opaque types)
// ========================================================================

// FP8 E4M3 -> FP16
// O0-LABEL: @test_mfmacc_h_e4
// O0: call target("riscv.matrix") @llvm.riscv.th.mfmacc.h.e4.internal
mfloat16_t test_mfmacc_h_e4(mfloat16_t c, mint32_t a, mint32_t b,
                              mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mfmacc_h_e4(c, a, b, m, k, n);
}

// BF16 -> FP32
// O0-LABEL: @test_mfmacc_s_bf16
// O0: call target("riscv.matrix") @llvm.riscv.th.mfmacc.s.bf16.internal
mfloat32_t test_mfmacc_s_bf16(mfloat32_t c, mint32_t a, mint32_t b,
                                mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mfmacc_s_bf16(c, a, b, m, k, n);
}

// ========================================================================
// 10. Configuration
// ========================================================================

// O0-LABEL: @test_config
// O0: call void @llvm.riscv.th.msettilem(i64 %m)
// O0: call void @llvm.riscv.th.msettilen(i64 %n)
// O0: call void @llvm.riscv.th.msettilek(i64 %k)
// O0: call void @llvm.riscv.th.mrelease()
void test_config(mrow_t m, mcol_t n, mcol_t k) {
    __riscv_th_msetmrow_m(m);
    __riscv_th_msetmrow_n(n);
    __riscv_th_msetmcol_e32(k);
    __riscv_th_mrelease();
}

// ========================================================================
// 11. Bypass and partial INT matmul
// ========================================================================

// O0-LABEL: @test_bypass_ss
// O0: call target("riscv.matrix") @llvm.riscv.th.mmacc.w.bp.internal
mint32_t test_bypass_ss(mint32_t c, mint8_t a, mint8_t b,
                         mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_mmacc_w_bp(c, a, b, m, k, n);
}

// O0-LABEL: @test_partial_ss
// O0: call target("riscv.matrix") @llvm.riscv.th.pmmacc.w.b.internal
mint32_t test_partial_ss(mint32_t c, mint8_t a, mint8_t b,
                          mrow_t m, mcol_t k, mcol_t n) {
    return __riscv_th_pmmacc_w_b(c, a, b, m, k, n);
}

// ========================================================================
// 12. Undefined value constructors
// ========================================================================

// O0-LABEL: @test_mundefined_i32
// O0: ret target("riscv.matrix") poison
mint32_t test_mundefined_i32(void) {
    return __riscv_th_mundefined_i32();
}

// O0-LABEL: @test_mundefined_f16x2
// O0: ret { target("riscv.matrix"), target("riscv.matrix") } poison
mfloat16x2_t test_mundefined_f16x2(void) {
    return __riscv_th_mundefined_f16x2();
}
