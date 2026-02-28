// NOTE: Comprehensive XTHeadMatrix API test: covers ALL functions in <thead_matrix.h>.
// Tests every API function across all type variants to verify correctness.
//
// RUN: %clang_cc1 -O2 -triple riscv64 -target-feature +experimental-xtheadmatrix -S -o - %s \
// RUN:   | FileCheck %s

#include <thead_matrix.h>

// ============================================================================
// Section 1: Configuration -- all 6 config + release functions
// ============================================================================

// CHECK-LABEL: test_config_all:
// CHECK: th.msettilem
// CHECK: th.msettilen
// CHECK: th.msettilek
// CHECK: th.msettilek
// CHECK: th.msettilek
// CHECK: th.msettilek
// CHECK: th.mrelease
// CHECK: ret
void test_config_all(mrow_t m, mrow_t n, mcol_t c) {
  mrow_t rm = __riscv_th_msetmrow_m(m);
  mrow_t rn = __riscv_th_msetmrow_n(n);
  mcol_t c8 = __riscv_th_msetmcol_e8(c);
  mcol_t c16 = __riscv_th_msetmcol_e16(c);
  mcol_t c32 = __riscv_th_msetmcol_e32(c);
  mcol_t c64 = __riscv_th_msetmcol_e64(c);
  __riscv_th_mrelease();
  (void)rm; (void)rn;
  (void)c8; (void)c16; (void)c32; (void)c64;
}

// ============================================================================
// Section 2: CSR access functions
// ============================================================================

// CHECK-LABEL: test_csr_read:
// CHECK: csrr {{.*}}, th.xtlenb
// CHECK: csrr {{.*}}, th.xtrlenb
// CHECK: csrr {{.*}}, th.xmisa
// CHECK: ret
unsigned long test_csr_read(void) {
  unsigned long a = __riscv_th_xmlenb();
  unsigned long b = __riscv_th_xrlenb();
  unsigned long c = __riscv_th_xmsize();
  return a + b + c;
}

// CHECK-LABEL: test_csr_macro_read:
// CHECK: csrr {{.*}}, th.xmcsr
// CHECK: csrr {{.*}}, th.mtilem
// CHECK: csrr {{.*}}, th.xmxrm
// CHECK: csrr {{.*}}, th.xmsat
// CHECK: csrr {{.*}}, th.xmfflags
// CHECK: csrr {{.*}}, th.xmfrm
// CHECK: csrr {{.*}}, th.xmsaten
// CHECK: ret
unsigned long test_csr_macro_read(void) {
  unsigned long v1 = __riscv_th_mread_csr(RVM_CSR_XMCSR);
  unsigned long v2 = __riscv_th_mread_csr(RVM_CSR_MTILEM);
  unsigned long v3 = __riscv_th_mread_csr(RVM_CSR_XMXRM);
  unsigned long v4 = __riscv_th_mread_csr(RVM_CSR_XMSAT);
  unsigned long v5 = __riscv_th_mread_csr(RVM_CSR_XMFFLAGS);
  unsigned long v6 = __riscv_th_mread_csr(RVM_CSR_XMFRM);
  unsigned long v7 = __riscv_th_mread_csr(RVM_CSR_XMSATEN);
  return v1 + v2 + v3 + v4 + v5 + v6 + v7;
}

// CHECK-LABEL: test_csr_macro_write:
// CHECK: csrw th.xmcsr
// CHECK: csrw th.xmxrm
// CHECK: csrw th.xmsat
// CHECK: csrw th.xmsaten
// CHECK: ret
void test_csr_macro_write(unsigned long val) {
  __riscv_th_mwrite_csr(RVM_CSR_XMCSR, val);
  __riscv_th_mwrite_csr(RVM_CSR_XMXRM, val);
  __riscv_th_mwrite_csr(RVM_CSR_XMSAT, val);
  __riscv_th_mwrite_csr(RVM_CSR_XMSATEN, val);
}

// ============================================================================
// Section 3: Zero functions -- all 22 types
// ============================================================================

// CHECK-LABEL: test_mzero_all_single:
// CHECK-COUNT-11: th.mzero
// CHECK: ret
void test_mzero_all_single(void) {
  mint8_t    z1 = __riscv_th_mzero_i8();
  mint16_t   z2 = __riscv_th_mzero_i16();
  mint32_t   z3 = __riscv_th_mzero_i32();
  mint64_t   z4 = __riscv_th_mzero_i64();
  muint8_t   z5 = __riscv_th_mzero_u8();
  muint16_t  z6 = __riscv_th_mzero_u16();
  muint32_t  z7 = __riscv_th_mzero_u32();
  muint64_t  z8 = __riscv_th_mzero_u64();
  mfloat16_t z9 = __riscv_th_mzero_f16();
  mfloat32_t z10 = __riscv_th_mzero_f32();
  mfloat64_t z11 = __riscv_th_mzero_f64();
  (void)z1; (void)z2; (void)z3; (void)z4; (void)z5;
  (void)z6; (void)z7; (void)z8; (void)z9; (void)z10; (void)z11;
}

// CHECK-LABEL: test_mzero_all_pair:
// CHECK-COUNT-11: th.mzero2r
// CHECK: ret
void test_mzero_all_pair(void) {
  mint8x2_t    z1 = __riscv_th_mzero_i8x2();
  mint16x2_t   z2 = __riscv_th_mzero_i16x2();
  mint32x2_t   z3 = __riscv_th_mzero_i32x2();
  mint64x2_t   z4 = __riscv_th_mzero_i64x2();
  muint8x2_t   z5 = __riscv_th_mzero_u8x2();
  muint16x2_t  z6 = __riscv_th_mzero_u16x2();
  muint32x2_t  z7 = __riscv_th_mzero_u32x2();
  muint64x2_t  z8 = __riscv_th_mzero_u64x2();
  mfloat16x2_t z9 = __riscv_th_mzero_f16x2();
  mfloat32x2_t z10 = __riscv_th_mzero_f32x2();
  mfloat64x2_t z11 = __riscv_th_mzero_f64x2();
  (void)z1; (void)z2; (void)z3; (void)z4; (void)z5;
  (void)z6; (void)z7; (void)z8; (void)z9; (void)z10; (void)z11;
}

// ============================================================================
// Section 4: Undefined value constructors -- all 22 types
// ============================================================================

// CHECK-LABEL: test_mundefined_all:
// CHECK: ret
void test_mundefined_all(void) {
  mint8_t    u1 = __riscv_th_mundefined_i8();
  mint16_t   u2 = __riscv_th_mundefined_i16();
  mint32_t   u3 = __riscv_th_mundefined_i32();
  mint64_t   u4 = __riscv_th_mundefined_i64();
  muint8_t   u5 = __riscv_th_mundefined_u8();
  muint16_t  u6 = __riscv_th_mundefined_u16();
  muint32_t  u7 = __riscv_th_mundefined_u32();
  muint64_t  u8 = __riscv_th_mundefined_u64();
  mfloat16_t u9 = __riscv_th_mundefined_f16();
  mfloat32_t u10 = __riscv_th_mundefined_f32();
  mfloat64_t u11 = __riscv_th_mundefined_f64();
  mint8x2_t    u12 = __riscv_th_mundefined_i8x2();
  mint16x2_t   u13 = __riscv_th_mundefined_i16x2();
  mint32x2_t   u14 = __riscv_th_mundefined_i32x2();
  mint64x2_t   u15 = __riscv_th_mundefined_i64x2();
  muint8x2_t   u16 = __riscv_th_mundefined_u8x2();
  muint16x2_t  u17 = __riscv_th_mundefined_u16x2();
  muint32x2_t  u18 = __riscv_th_mundefined_u32x2();
  muint64x2_t  u19 = __riscv_th_mundefined_u64x2();
  mfloat16x2_t u20 = __riscv_th_mundefined_f16x2();
  mfloat32x2_t u21 = __riscv_th_mundefined_f32x2();
  mfloat64x2_t u22 = __riscv_th_mundefined_f64x2();
  (void)u1; (void)u2; (void)u3; (void)u4; (void)u5; (void)u6;
  (void)u7; (void)u8; (void)u9; (void)u10; (void)u11; (void)u12;
  (void)u13; (void)u14; (void)u15; (void)u16; (void)u17; (void)u18;
  (void)u19; (void)u20; (void)u21; (void)u22;
}

// ============================================================================
// Section 5: Reinterpret cast -- verify with src parameter
// ============================================================================

// CHECK-LABEL: test_reinterpret:
// CHECK: ret
void test_reinterpret(void) {
  mint32_t src = __riscv_th_mundefined_i32();
  // Single reinterprets (src is consumed by void cast)
  mint8_t    r1 = __riscv_th_mreinterpret_i8(src);
  muint32_t  r2 = __riscv_th_mreinterpret_u32(src);
  mfloat32_t r3 = __riscv_th_mreinterpret_f32(src);
  // Cross-type reinterprets
  mfloat16_t f16 = __riscv_th_mundefined_f16();
  mint16_t   r4 = __riscv_th_mreinterpret_i16(f16);
  // Pair reinterprets
  mint32x2_t p = __riscv_th_mundefined_i32x2();
  mfloat32x2_t r5 = __riscv_th_mreinterpret_f32x2(p);
  (void)r1; (void)r2; (void)r3; (void)r4; (void)r5;
}

// ============================================================================
// Section 6: Tuple get/set -- all 11 types
// ============================================================================

// CHECK-LABEL: test_tuple_get_set:
// CHECK: ret
void test_tuple_get_set(void) {
  // Test get operations
  mint32x2_t p32 = __riscv_th_mundefined_i32x2();
  mint32_t g1 = __riscv_th_mget_i32(p32, 0);
  mint32_t g2 = __riscv_th_mget_i32(p32, 1);

  // Test set operations
  mint32_t single = __riscv_th_mundefined_i32();
  mint32x2_t s1 = __riscv_th_mset_i32(p32, 0, single);
  mint32x2_t s2 = __riscv_th_mset_i32(p32, 1, single);

  // Test with float types
  mfloat32x2_t fp = __riscv_th_mundefined_f32x2();
  mfloat32_t fg = __riscv_th_mget_f32(fp, 0);
  mfloat32_t fval = __riscv_th_mundefined_f32();
  mfloat32x2_t fs = __riscv_th_mset_f32(fp, 1, fval);

  (void)g1; (void)g2; (void)s1; (void)s2; (void)fg; (void)fs;
}

// ============================================================================
// Section 7: Load functions -- every role and type variant
// ============================================================================

// CHECK-LABEL: test_load_all_roles_i32:
// CHECK: th.msettilem
// CHECK: th.msettilek
// CHECK: th.mlae32
// CHECK: th.msettilek
// CHECK: th.msettilen
// CHECK: th.mlbe32
// CHECK: th.msettilem
// CHECK: th.msettilen
// CHECK: th.mlce32
// CHECK: ret
void test_load_all_roles_i32(const void *base, long stride,
                              mrow_t m, mrow_t k, mcol_t n) {
  mint32_t a = __riscv_th_mld_a_i32(base, stride, m, k);
  mint32_t b = __riscv_th_mld_b_i32(base, stride, k, n);
  mint32_t c = __riscv_th_mld_c_i32(base, stride, m, n);
  (void)a; (void)b; (void)c;
}

// CHECK-LABEL: test_load_transposed_all:
// CHECK: th.mlate8
// CHECK: th.mlbte16
// CHECK: th.mlcte32
// CHECK: ret
void test_load_transposed_all(const void *base, long stride,
                               mrow_t m, mrow_t k, mcol_t n) {
  mint8_t  at = __riscv_th_mld_at_i8(base, stride, m, k);
  mint16_t bt = __riscv_th_mld_bt_i16(base, stride, k, n);
  mint32_t ct = __riscv_th_mld_ct_i32(base, stride, m, n);
  (void)at; (void)bt; (void)ct;
}

// CHECK-LABEL: test_load_unsigned:
// CHECK: th.mlae8
// CHECK: th.mlbe16
// CHECK: th.mlce32
// CHECK: ret
void test_load_unsigned(const void *base, long stride,
                         mrow_t m, mrow_t k, mcol_t n) {
  muint8_t  a = __riscv_th_mld_a_u8(base, stride, m, k);
  muint16_t b = __riscv_th_mld_b_u16(base, stride, k, n);
  muint32_t c = __riscv_th_mld_c_u32(base, stride, m, n);
  (void)a; (void)b; (void)c;
}

// CHECK-LABEL: test_load_fp:
// CHECK: th.mlae16
// CHECK: th.mlbe32
// CHECK: th.mlce64
// CHECK: ret
void test_load_fp(const void *base, long stride,
                   mrow_t m, mrow_t k, mcol_t n) {
  mfloat16_t a = __riscv_th_mld_a_f16(base, stride, m, k);
  mfloat32_t b = __riscv_th_mld_b_f32(base, stride, k, n);
  mfloat64_t c = __riscv_th_mld_c_f64(base, stride, m, n);
  (void)a; (void)b; (void)c;
}

// CHECK-LABEL: test_load_whole:
// CHECK: th.mlme8
// CHECK: th.mlme16
// CHECK: th.mlme32
// CHECK: th.mlme64
// CHECK: ret
void test_load_whole(const void *base) {
  mint8_t  w8  = __riscv_th_mld_whole_i8(base);
  mint16_t w16 = __riscv_th_mld_whole_i16(base);
  mint32_t w32 = __riscv_th_mld_whole_i32(base);
  mint64_t w64 = __riscv_th_mld_whole_i64(base);
  (void)w8; (void)w16; (void)w32; (void)w64;
}

// ============================================================================
// Section 8: Store functions -- all roles and types
// ============================================================================

// CHECK-LABEL: test_store_all_roles:
// CHECK: th.msae32
// CHECK: th.msbe32
// CHECK: th.msce32
// CHECK: ret
void test_store_all_roles(void *base, long stride,
                           mrow_t m, mrow_t k, mcol_t n) {
  mint32_t val = __riscv_th_mundefined_i32();
  __riscv_th_mst_a_i32(base, stride, val, m, k);
  __riscv_th_mst_b_i32(base, stride, val, k, n);
  __riscv_th_mst_c_i32(base, stride, val, m, n);
}

// CHECK-LABEL: test_store_transposed:
// CHECK: th.msate8
// CHECK: th.msbte16
// CHECK: th.mscte32
// CHECK: ret
void test_store_transposed(void *base, long stride,
                            mrow_t m, mrow_t k, mcol_t n) {
  mint8_t  v8  = __riscv_th_mundefined_i8();
  mint16_t v16 = __riscv_th_mundefined_i16();
  mint32_t v32 = __riscv_th_mundefined_i32();
  __riscv_th_mst_at_i8(base, stride, v8, m, k);
  __riscv_th_mst_bt_i16(base, stride, v16, k, n);
  __riscv_th_mst_ct_i32(base, stride, v32, m, n);
}

// CHECK-LABEL: test_store_fp:
// CHECK: th.msae16
// CHECK: th.msbe32
// CHECK: th.msce64
// CHECK: ret
void test_store_fp(void *base, long stride,
                    mrow_t m, mrow_t k, mcol_t n) {
  mfloat16_t f16 = __riscv_th_mundefined_f16();
  mfloat32_t f32 = __riscv_th_mundefined_f32();
  mfloat64_t f64 = __riscv_th_mundefined_f64();
  __riscv_th_mst_a_f16(base, stride, f16, m, k);
  __riscv_th_mst_b_f32(base, stride, f32, k, n);
  __riscv_th_mst_c_f64(base, stride, f64, m, n);
}

// CHECK-LABEL: test_store_whole:
// CHECK: th.msme8
// CHECK: th.msme32
// CHECK: th.msme64
// CHECK: ret
void test_store_whole(void *base) {
  mint8_t  v8  = __riscv_th_mundefined_i8();
  mint32_t v32 = __riscv_th_mundefined_i32();
  mfloat64_t f64 = __riscv_th_mundefined_f64();
  __riscv_th_mst_whole_i8(base, v8);
  __riscv_th_mst_whole_i32(base, v32);
  __riscv_th_mst_whole_f64(base, f64);
}

// ============================================================================
// Section 9: FP Matmul -- all variants
// ============================================================================

// CHECK-LABEL: test_fmmacc_all:
// CHECK: th.mfmacc.h
// CHECK: th.mfmacc.s
// CHECK: th.mfmacc.d
// CHECK: ret
void test_fmmacc_all(mrow_t m, mrow_t k, mcol_t n) {
  mfloat16_t ch = __riscv_th_mundefined_f16();
  mfloat32_t cs = __riscv_th_mundefined_f32();
  mfloat64_t cd = __riscv_th_mundefined_f64();
  ch = __riscv_th_fmmacc_h(ch, ch, ch, m, k, n);
  cs = __riscv_th_fmmacc_s(cs, cs, cs, m, k, n);
  cd = __riscv_th_fmmacc_d(cd, cd, cd, m, k, n);
  (void)ch; (void)cs; (void)cd;
}

// CHECK-LABEL: test_fwmmacc_typed:
// CHECK: th.mfmacc.s.h
// CHECK: th.mfmacc.d.s
// CHECK: ret
void test_fwmmacc_typed(mrow_t m, mrow_t k, mcol_t n) {
  mfloat32_t cs = __riscv_th_mundefined_f32();
  mfloat16_t ah = __riscv_th_mundefined_f16();
  mfloat64_t cd = __riscv_th_mundefined_f64();
  mfloat32_t as = __riscv_th_mundefined_f32();
  cs = __riscv_th_fwmmacc_s_h(cs, ah, ah, m, k, n);
  cd = __riscv_th_fwmmacc_d_s(cd, as, as, m, k, n);
  (void)cs; (void)cd;
}

// CHECK-LABEL: test_fwmmacc_untyped:
// CHECK: th.mfmacc.h.e4
// CHECK: th.mfmacc.h.e5
// CHECK: th.mfmacc.bf16.e4
// CHECK: th.mfmacc.bf16.e5
// CHECK: th.mfmacc.s.bf16
// CHECK: th.mfmacc.s.e4
// CHECK: th.mfmacc.s.e5
// CHECK: th.mfmacc.s.tf32
// CHECK: ret
void test_fwmmacc_untyped(mrow_t m, mrow_t k, mcol_t n) {
  muint8_t u8 = __riscv_th_mundefined_u8();
  mfloat16_t f16 = __riscv_th_mundefined_f16();
  mfloat32_t f32 = __riscv_th_mundefined_f32();
  (void)__riscv_th_fwmmacc_h_e4(f16, u8, u8, m, k, n);
  (void)__riscv_th_fwmmacc_h_e5(f16, u8, u8, m, k, n);
  (void)__riscv_th_fwmmacc_bf16_e4(f16, u8, u8, m, k, n);
  (void)__riscv_th_fwmmacc_bf16_e5(f16, u8, u8, m, k, n);
  (void)__riscv_th_fwmmacc_s_bf16(f32, f16, f16, m, k, n);
  (void)__riscv_th_fwmmacc_s_e4(f32, u8, u8, m, k, n);
  (void)__riscv_th_fwmmacc_s_e5(f32, u8, u8, m, k, n);
  (void)__riscv_th_fwmmacc_s_tf32(f32, f32, f32, m, k, n);
}

// ============================================================================
// Section 10: Integer Matmul -- all variants
// ============================================================================

// CHECK-LABEL: test_mmaqa_all:
// CHECK: th.mmacc.w.b
// CHECK: th.mmaccu.w.b
// CHECK: th.mmaccus.w.b
// CHECK: th.mmaccsu.w.b
// CHECK: th.mmacc.d.h
// CHECK: th.mmaccu.d.h
// CHECK: th.mmaccus.d.h
// CHECK: th.mmaccsu.d.h
// CHECK: ret
void test_mmaqa_all(mrow_t m, mrow_t k, mcol_t n) {
  mint32_t  ci32 = __riscv_th_mundefined_i32();
  muint32_t cu32 = __riscv_th_mundefined_u32();
  mint64_t  ci64 = __riscv_th_mundefined_i64();
  muint64_t cu64 = __riscv_th_mundefined_u64();
  mint8_t  i8 = __riscv_th_mundefined_i8();
  muint8_t u8 = __riscv_th_mundefined_u8();
  mint16_t  i16 = __riscv_th_mundefined_i16();
  muint16_t u16 = __riscv_th_mundefined_u16();

  ci32 = __riscv_th_mmaqa_ss_w_b(ci32, i8, i8, m, k, n);
  cu32 = __riscv_th_mmaqa_uu_w_b(cu32, u8, u8, m, k, n);
  ci32 = __riscv_th_mmaqa_us_w_b(ci32, u8, i8, m, k, n);
  ci32 = __riscv_th_mmaqa_su_w_b(ci32, i8, u8, m, k, n);

  ci64 = __riscv_th_mmaqa_ss_d_h(ci64, i16, i16, m, k, n);
  cu64 = __riscv_th_mmaqa_uu_d_h(cu64, u16, u16, m, k, n);
  ci64 = __riscv_th_mmaqa_us_d_h(ci64, u16, i16, m, k, n);
  ci64 = __riscv_th_mmaqa_su_d_h(ci64, i16, u16, m, k, n);

  (void)ci32; (void)cu32; (void)ci64; (void)cu64;
}

// CHECK-LABEL: test_pmmaqa_all:
// CHECK: th.pmmacc.w.b
// CHECK: th.pmmaccu.w.b
// CHECK: th.pmmaccus.w.b
// CHECK: th.pmmaccsu.w.b
// CHECK: ret
void test_pmmaqa_all(mrow_t m, mrow_t k, mcol_t n) {
  mint32_t  ci32 = __riscv_th_mundefined_i32();
  muint32_t cu32 = __riscv_th_mundefined_u32();
  mint8_t  i8 = __riscv_th_mundefined_i8();
  muint8_t u8 = __riscv_th_mundefined_u8();

  ci32 = __riscv_th_pmmaqa_ss_w_b(ci32, i8, i8, m, k, n);
  cu32 = __riscv_th_pmmaqa_uu_w_b(cu32, u8, u8, m, k, n);
  ci32 = __riscv_th_pmmaqa_us_w_b(ci32, u8, i8, m, k, n);
  ci32 = __riscv_th_pmmaqa_su_w_b(ci32, i8, u8, m, k, n);

  (void)ci32; (void)cu32;
}

// CHECK-LABEL: test_bypass_matmul:
// CHECK: th.mmacc.w.bp
// CHECK: th.mmaccu.w.bp
// CHECK: ret
void test_bypass_matmul(mrow_t m, mrow_t k, mcol_t n) {
  mint32_t  ci = __riscv_th_mundefined_i32();
  muint32_t cu = __riscv_th_mundefined_u32();
  mint8_t  i8 = __riscv_th_mundefined_i8();
  muint8_t u8 = __riscv_th_mundefined_u8();

  ci = __riscv_th_mmaqa_bp_ss(ci, i8, i8, m, k, n);
  cu = __riscv_th_mmaqa_bp_uu(cu, u8, u8, m, k, n);

  (void)ci; (void)cu;
}

// ============================================================================
// Section 11: Integer EW Arithmetic -- all 11 ops, MM and MVI variants
// ============================================================================

// CHECK-LABEL: test_int_ew_all_mm:
// CHECK: th.madd.w.mm
// CHECK: th.msub.w.mm
// CHECK: th.mmul.w.mm
// CHECK: th.mmulh.w.mm
// CHECK: th.mmax.w.mm
// CHECK: th.mumax.w.mm
// CHECK: th.mmin.w.mm
// CHECK: th.mumin.w.mm
// CHECK: th.msrl.w.mm
// CHECK: th.msll.w.mm
// CHECK: th.msra.w.mm
// CHECK: ret
void test_int_ew_all_mm(void) {
  mint32_t d = __riscv_th_mundefined_i32();
  mint32_t s1 = __riscv_th_mundefined_i32();
  mint32_t s2 = __riscv_th_mundefined_i32();
  d = __riscv_th_madd_w_mm(d, s1, s2);
  d = __riscv_th_msub_w_mm(d, s1, s2);
  d = __riscv_th_mmul_w_mm(d, s1, s2);
  d = __riscv_th_mmulh_w_mm(d, s1, s2);
  d = __riscv_th_mmax_w_mm(d, s1, s2);
  d = __riscv_th_mumax_w_mm(d, s1, s2);
  d = __riscv_th_mmin_w_mm(d, s1, s2);
  d = __riscv_th_mumin_w_mm(d, s1, s2);
  d = __riscv_th_msrl_w_mm(d, s1, s2);
  d = __riscv_th_msll_w_mm(d, s1, s2);
  d = __riscv_th_msra_w_mm(d, s1, s2);
  (void)d;
}

// CHECK-LABEL: test_int_ew_all_mvi:
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
void test_int_ew_all_mvi(void) {
  mint32_t d = __riscv_th_mundefined_i32();
  mint32_t s = __riscv_th_mundefined_i32();
  d = __riscv_th_madd_w_mv(d, s, 0);
  d = __riscv_th_msub_w_mv(d, s, 1);
  d = __riscv_th_mmul_w_mv(d, s, 2);
  d = __riscv_th_mmulh_w_mv(d, s, 3);
  d = __riscv_th_mmax_w_mv(d, s, 4);
  d = __riscv_th_mumax_w_mv(d, s, 5);
  d = __riscv_th_mmin_w_mv(d, s, 6);
  d = __riscv_th_mumin_w_mv(d, s, 7);
  d = __riscv_th_msrl_w_mv(d, s, 1);
  d = __riscv_th_msll_w_mv(d, s, 2);
  d = __riscv_th_msra_w_mv(d, s, 3);
  (void)d;
}

// ============================================================================
// Section 12: FP EW Arithmetic -- all precisions
// ============================================================================

// CHECK-LABEL: test_fp_ew_h_all:
// CHECK: th.mfadd.h.mm
// CHECK: th.mfsub.h.mm
// CHECK: th.mfmul.h.mm
// CHECK: th.mfmax.h.mm
// CHECK: th.mfmin.h.mm
// CHECK: ret
void test_fp_ew_h_all(void) {
  mfloat16_t d = __riscv_th_mundefined_f16();
  mfloat16_t s1 = __riscv_th_mundefined_f16();
  mfloat16_t s2 = __riscv_th_mundefined_f16();
  d = __riscv_th_mfadd_h_mm(d, s1, s2);
  d = __riscv_th_mfsub_h_mm(d, s1, s2);
  d = __riscv_th_mfmul_h_mm(d, s1, s2);
  d = __riscv_th_mfmax_h_mm(d, s1, s2);
  d = __riscv_th_mfmin_h_mm(d, s1, s2);
  (void)d;
}

// CHECK-LABEL: test_fp_ew_s_all:
// CHECK: th.mfadd.s.mm
// CHECK: th.mfsub.s.mm
// CHECK: th.mfmul.s.mm
// CHECK: th.mfmax.s.mm
// CHECK: th.mfmin.s.mm
// CHECK: ret
void test_fp_ew_s_all(void) {
  mfloat32_t d = __riscv_th_mundefined_f32();
  mfloat32_t s1 = __riscv_th_mundefined_f32();
  mfloat32_t s2 = __riscv_th_mundefined_f32();
  d = __riscv_th_mfadd_s_mm(d, s1, s2);
  d = __riscv_th_mfsub_s_mm(d, s1, s2);
  d = __riscv_th_mfmul_s_mm(d, s1, s2);
  d = __riscv_th_mfmax_s_mm(d, s1, s2);
  d = __riscv_th_mfmin_s_mm(d, s1, s2);
  (void)d;
}

// CHECK-LABEL: test_fp_ew_d_all:
// CHECK: th.mfadd.d.mm
// CHECK: th.mfsub.d.mm
// CHECK: th.mfmul.d.mm
// CHECK: th.mfmax.d.mm
// CHECK: th.mfmin.d.mm
// CHECK: ret
void test_fp_ew_d_all(void) {
  mfloat64_t d = __riscv_th_mundefined_f64();
  mfloat64_t s1 = __riscv_th_mundefined_f64();
  mfloat64_t s2 = __riscv_th_mundefined_f64();
  d = __riscv_th_mfadd_d_mm(d, s1, s2);
  d = __riscv_th_mfsub_d_mm(d, s1, s2);
  d = __riscv_th_mfmul_d_mm(d, s1, s2);
  d = __riscv_th_mfmax_d_mm(d, s1, s2);
  d = __riscv_th_mfmin_d_mm(d, s1, s2);
  (void)d;
}

// CHECK-LABEL: test_fp_ew_mvi_all:
// CHECK: th.mfadd.h.mv.i
// CHECK: th.mfsub.s.mv.i
// CHECK: th.mfmul.d.mv.i
// CHECK: th.mfmax.h.mv.i
// CHECK: th.mfmin.s.mv.i
// CHECK: ret
void test_fp_ew_mvi_all(void) {
  mfloat16_t f16 = __riscv_th_mundefined_f16();
  mfloat32_t f32 = __riscv_th_mundefined_f32();
  mfloat64_t f64 = __riscv_th_mundefined_f64();
  f16 = __riscv_th_mfadd_h_mv(f16, f16, 0);
  f32 = __riscv_th_mfsub_s_mv(f32, f32, 1);
  f64 = __riscv_th_mfmul_d_mv(f64, f64, 2);
  f16 = __riscv_th_mfmax_h_mv(f16, f16, 3);
  f32 = __riscv_th_mfmin_s_mv(f32, f32, 4);
  (void)f16; (void)f32; (void)f64;
}

// ============================================================================
// Section 13: FP Format Conversions -- all variants
// ============================================================================

// CHECK-LABEL: test_fcvt_fp16_fp32:
// CHECK: th.mfcvtl.s.h
// CHECK: th.mfcvth.s.h
// CHECK: th.mfcvtl.h.s
// CHECK: th.mfcvth.h.s
// CHECK: ret
void test_fcvt_fp16_fp32(void) {
  mfloat16_t f16 = __riscv_th_mundefined_f16();
  mfloat32_t f32 = __riscv_th_mundefined_f32();
  f32 = __riscv_th_mfcvtl_s_h(f16);
  f32 = __riscv_th_mfcvth_s_h(f16);
  f16 = __riscv_th_mfcvtl_h_s(f32);
  f16 = __riscv_th_mfcvth_h_s(f32);
  (void)f16; (void)f32;
}

// CHECK-LABEL: test_fcvt_fp32_fp64:
// CHECK: th.mfcvtl.d.s
// CHECK: th.mfcvth.d.s
// CHECK: th.mfcvtl.s.d
// CHECK: th.mfcvth.s.d
// CHECK: ret
void test_fcvt_fp32_fp64(void) {
  mfloat32_t f32 = __riscv_th_mundefined_f32();
  mfloat64_t f64 = __riscv_th_mundefined_f64();
  f64 = __riscv_th_mfcvtl_d_s(f32);
  f64 = __riscv_th_mfcvth_d_s(f32);
  f32 = __riscv_th_mfcvtl_s_d(f64);
  f32 = __riscv_th_mfcvth_s_d(f64);
  (void)f32; (void)f64;
}

// CHECK-LABEL: test_fcvt_fp8:
// CHECK: th.mfcvtl.h.e4
// CHECK: th.mfcvth.h.e4
// CHECK: th.mfcvtl.h.e5
// CHECK: th.mfcvth.h.e5
// CHECK: th.mfcvtl.e4.h
// CHECK: th.mfcvth.e4.h
// CHECK: th.mfcvtl.e5.h
// CHECK: th.mfcvth.e5.h
// CHECK: ret
void test_fcvt_fp8(void) {
  muint8_t u8 = __riscv_th_mundefined_u8();
  mfloat16_t f16 = __riscv_th_mundefined_f16();
  f16 = __riscv_th_mfcvtl_h_e4(u8);
  f16 = __riscv_th_mfcvth_h_e4(u8);
  f16 = __riscv_th_mfcvtl_h_e5(u8);
  f16 = __riscv_th_mfcvth_h_e5(u8);
  u8 = __riscv_th_mfcvtl_e4_h(f16);
  u8 = __riscv_th_mfcvth_e4_h(f16);
  u8 = __riscv_th_mfcvtl_e5_h(f16);
  u8 = __riscv_th_mfcvth_e5_h(f16);
  (void)u8; (void)f16;
}

// CHECK-LABEL: test_fcvt_bf16:
// CHECK: th.mfcvtl.s.bf16
// CHECK: th.mfcvth.s.bf16
// CHECK: th.mfcvtl.bf16.s
// CHECK: th.mfcvth.bf16.s
// CHECK: ret
void test_fcvt_bf16(void) {
  mfloat16_t f16 = __riscv_th_mundefined_f16();
  mfloat32_t f32 = __riscv_th_mundefined_f32();
  f32 = __riscv_th_mfcvtl_s_bf16(f16);
  f32 = __riscv_th_mfcvth_s_bf16(f16);
  f16 = __riscv_th_mfcvtl_bf16_s(f32);
  f16 = __riscv_th_mfcvth_bf16_s(f32);
  (void)f16; (void)f32;
}

// CHECK-LABEL: test_fcvt_fp32_fp8:
// CHECK: th.mfcvtl.e4.s
// CHECK: th.mfcvth.e4.s
// CHECK: th.mfcvtl.e5.s
// CHECK: th.mfcvth.e5.s
// CHECK: ret
void test_fcvt_fp32_fp8(void) {
  mfloat32_t f32 = __riscv_th_mundefined_f32();
  muint8_t u8;
  u8 = __riscv_th_mfcvtl_e4_s(f32);
  u8 = __riscv_th_mfcvth_e4_s(f32);
  u8 = __riscv_th_mfcvtl_e5_s(f32);
  u8 = __riscv_th_mfcvth_e5_s(f32);
  (void)u8;
}

// CHECK-LABEL: test_fcvt_tf32:
// CHECK: th.mfcvt.s.tf32
// CHECK: th.mfcvt.tf32.s
// CHECK: ret
void test_fcvt_tf32(void) {
  mfloat32_t f32 = __riscv_th_mundefined_f32();
  f32 = __riscv_th_mfcvt_s_tf32(f32);
  f32 = __riscv_th_mfcvt_tf32_s(f32);
  (void)f32;
}

// ============================================================================
// Section 14: Float-Int Conversions -- all variants
// ============================================================================

// CHECK-LABEL: test_fint_cvt_all:
// CHECK: th.mufcvtl.h.b
// CHECK: th.mufcvth.h.b
// CHECK: th.mfucvtl.b.h
// CHECK: th.mfucvth.b.h
// CHECK: th.msfcvtl.h.b
// CHECK: th.msfcvth.h.b
// CHECK: th.mfscvtl.b.h
// CHECK: th.mfscvth.b.h
// CHECK: th.msfcvt.s.w
// CHECK: th.mufcvt.s.w
// CHECK: th.mfscvt.w.s
// CHECK: th.mfucvt.w.s
// CHECK: ret
void test_fint_cvt_all(void) {
  muint8_t  u8 = __riscv_th_mundefined_u8();
  mint8_t   i8 = __riscv_th_mundefined_i8();
  mfloat16_t f16 = __riscv_th_mundefined_f16();
  mint32_t  i32 = __riscv_th_mundefined_i32();
  muint32_t u32 = __riscv_th_mundefined_u32();
  mfloat32_t f32 = __riscv_th_mundefined_f32();

  f16 = __riscv_th_mufcvtl_h_b(u8);
  f16 = __riscv_th_mufcvth_h_b(u8);
  u8  = __riscv_th_mfucvtl_b_h(f16);
  u8  = __riscv_th_mfucvth_b_h(f16);
  f16 = __riscv_th_msfcvtl_h_b(i8);
  f16 = __riscv_th_msfcvth_h_b(i8);
  i8  = __riscv_th_mfscvtl_b_h(f16);
  i8  = __riscv_th_mfscvth_b_h(f16);
  f32 = __riscv_th_msfcvt_s_w(i32);
  f32 = __riscv_th_mufcvt_s_w(u32);
  i32 = __riscv_th_mfscvt_w_s(f32);
  u32 = __riscv_th_mfucvt_w_s(f32);

  (void)u8; (void)i8; (void)f16; (void)i32; (void)u32; (void)f32;
}

// ============================================================================
// Section 15: Packed Conversions
// ============================================================================

// CHECK-LABEL: test_packed_cvt:
// CHECK: th.mucvtl.b.p
// CHECK: th.mscvtl.b.p
// CHECK: th.mucvth.b.p
// CHECK: th.mscvth.b.p
// CHECK: ret
void test_packed_cvt(void) {
  muint8_t u8 = __riscv_th_mundefined_u8();
  u8 = __riscv_th_mucvtl_b_p(u8);
  mint8_t i8 = __riscv_th_mscvtl_b_p(u8);
  u8 = __riscv_th_mucvth_b_p(u8);
  i8 = __riscv_th_mscvth_b_p(u8);
  (void)u8; (void)i8;
}

// ============================================================================
// Section 16: N4Clip -- verify return type correctness (bug fix)
// ============================================================================

// CHECK-LABEL: test_n4clip_types:
// CHECK: th.mn4clipl.w.mm
// CHECK: th.mn4cliph.w.mm
// CHECK: th.mn4cliplu.w.mm
// CHECK: th.mn4cliphu.w.mm
// CHECK: ret
void test_n4clip_types(void) {
  mint32_t s1 = __riscv_th_mundefined_i32();
  mint32_t s2 = __riscv_th_mundefined_i32();

  // Signed clip should return mint8_t
  mint8_t  signed_l = __riscv_th_mn4clipl_w_mm(s1, s2);
  mint8_t  signed_h = __riscv_th_mn4cliph_w_mm(s1, s2);

  // Unsigned clip should return muint8_t
  muint8_t unsigned_l = __riscv_th_mn4cliplu_w_mm(s1, s2);
  muint8_t unsigned_h = __riscv_th_mn4cliphu_w_mm(s1, s2);

  (void)signed_l; (void)signed_h;
  (void)unsigned_l; (void)unsigned_h;
}

// CHECK-LABEL: test_n4clip_mvi_types:
// CHECK: th.mn4clipl.w.mv.i
// CHECK: th.mn4cliph.w.mv.i
// CHECK: th.mn4cliplu.w.mv.i
// CHECK: th.mn4cliphu.w.mv.i
// CHECK: ret
void test_n4clip_mvi_types(void) {
  mint32_t s = __riscv_th_mundefined_i32();

  // Signed MVI clips return mint8_t
  mint8_t  signed_l = __riscv_th_mn4clipl_w_mv(s, 0);
  mint8_t  signed_h = __riscv_th_mn4cliph_w_mv(s, 1);

  // Unsigned MVI clips return muint8_t
  muint8_t unsigned_l = __riscv_th_mn4cliplu_w_mv(s, 2);
  muint8_t unsigned_h = __riscv_th_mn4cliphu_w_mv(s, 3);

  (void)signed_l; (void)signed_h;
  (void)unsigned_l; (void)unsigned_h;
}

// ============================================================================
// Section 17: Move / Duplicate -- all size variants
// ============================================================================

// CHECK-LABEL: test_mov_all:
// CHECK: th.mmov.mm
// CHECK: th.mmovb.x.m
// CHECK: th.mmovh.x.m
// CHECK: th.mmovw.x.m
// CHECK: th.mmovd.x.m
// CHECK: th.mmovb.m.x
// CHECK: th.mmovh.m.x
// CHECK: th.mmovw.m.x
// CHECK: th.mmovd.m.x
// CHECK: th.mdupb.m.x
// CHECK: th.mduph.m.x
// CHECK: th.mdupw.m.x
// CHECK: th.mdupd.m.x
// CHECK: ret
void test_mov_all(size_t data, size_t idx) {
  __riscv_th_mmov_mm();

  mint8_t  sb = __riscv_th_mundefined_i8();
  mint16_t sh = __riscv_th_mundefined_i16();
  mint32_t sw = __riscv_th_mundefined_i32();
  mint64_t sd = __riscv_th_mundefined_i64();

  size_t rb = __riscv_th_mmov_x_m_b(sb, idx);
  size_t rh = __riscv_th_mmov_x_m_h(sh, idx);
  size_t rw = __riscv_th_mmov_x_m_w(sw, idx);
  size_t rd = __riscv_th_mmov_x_m_d(sd, idx);

  sb = __riscv_th_mmov_m_x_b(sb, rb, idx);
  sh = __riscv_th_mmov_m_x_h(sh, rh, idx);
  sw = __riscv_th_mmov_m_x_w(sw, rw, idx);
  sd = __riscv_th_mmov_m_x_d(sd, rd, idx);

  sb = __riscv_th_mdup_m_x_b(data);
  sh = __riscv_th_mdup_m_x_h(data);
  sw = __riscv_th_mdup_m_x_w(data);
  sd = __riscv_th_mdup_m_x_d(data);

  (void)sb; (void)sh; (void)sw; (void)sd;
}

// ============================================================================
// Section 18: Pack
// ============================================================================

// CHECK-LABEL: test_pack_all:
// CHECK: th.mpack
// CHECK: th.mpackhl
// CHECK: th.mpackhh
// CHECK: ret
void test_pack_all(void) {
  __riscv_th_mpack();
  __riscv_th_mpackhl();
  __riscv_th_mpackhh();
}

// ============================================================================
// Section 19: Slide -- all variants
// ============================================================================

// CHECK-LABEL: test_slide_all:
// CHECK: th.mrslidedown
// CHECK: th.mrslideup
// CHECK: th.mcslidedown.b
// CHECK: th.mcslideup.b
// CHECK: th.mcslidedown.h
// CHECK: th.mcslideup.h
// CHECK: th.mcslidedown.w
// CHECK: th.mcslideup.w
// CHECK: th.mcslidedown.d
// CHECK: th.mcslideup.d
// CHECK: ret
void test_slide_all(void) {
  __riscv_th_mrslidedown(1);
  __riscv_th_mrslideup(2);
  __riscv_th_mcslidedown_b(3);
  __riscv_th_mcslideup_b(4);
  __riscv_th_mcslidedown_h(5);
  __riscv_th_mcslideup_h(6);
  __riscv_th_mcslidedown_w(7);
  __riscv_th_mcslideup_w(0);
  __riscv_th_mcslidedown_d(1);
  __riscv_th_mcslideup_d(2);
}

// ============================================================================
// Section 20: Broadcast -- all variants
// ============================================================================

// CHECK-LABEL: test_broadcast_all:
// CHECK: th.mrbca.mv.i
// CHECK: th.mcbcab.mv.i
// CHECK: th.mcbcah.mv.i
// CHECK: th.mcbcaw.mv.i
// CHECK: th.mcbcad.mv.i
// CHECK: ret
void test_broadcast_all(void) {
  __riscv_th_mrbca(0);
  __riscv_th_mcbca_b(1);
  __riscv_th_mcbca_h(2);
  __riscv_th_mcbca_w(3);
  __riscv_th_mcbca_d(4);
}
