// RUN: %clang_cc1 -triple riscv64 -target-feature +experimental-xtheadmatrix \
// RUN:   -verify -fsyntax-only %s

// Test xtheadmatrix register index constraint validation.

void test_load_tile_constraint(void *ptr) {
  // Load A: md must be tile (0-3) — valid
  __builtin_riscv_th_mlae8(0, ptr, 16);  // OK: tr0
  __builtin_riscv_th_mlae8(3, ptr, 16);  // OK: tr3

  // Load A: md must be tile (0-3) — invalid
  __builtin_riscv_th_mlae8(4, ptr, 16);  // expected-error {{xtheadmatrix md register must be a tile register (index 0-3)}}
  __builtin_riscv_th_mlae8(7, ptr, 16);  // expected-error {{xtheadmatrix md register must be a tile register (index 0-3)}}

  // Load B: md must be tile (0-3) — invalid
  __builtin_riscv_th_mlbe16(5, ptr, 16); // expected-error {{xtheadmatrix md register must be a tile register (index 0-3)}}

  // Load A^T: md must be tile (0-3) — invalid
  __builtin_riscv_th_mlate32(6, ptr, 16); // expected-error {{xtheadmatrix md register must be a tile register (index 0-3)}}
}

void test_load_acc_constraint(void *ptr) {
  // Load C: md must be acc (4-7) — valid
  __builtin_riscv_th_mlce8(4, ptr, 16);  // OK: acc0
  __builtin_riscv_th_mlce8(7, ptr, 16);  // OK: acc3

  // Load C: md must be acc (4-7) — invalid
  __builtin_riscv_th_mlce8(0, ptr, 16);  // expected-error {{xtheadmatrix md register must be an accumulator register (index 4-7)}}
  __builtin_riscv_th_mlce8(3, ptr, 16);  // expected-error {{xtheadmatrix md register must be an accumulator register (index 4-7)}}

  // Load C^T: md must be acc (4-7) — invalid
  __builtin_riscv_th_mlcte16(2, ptr, 16); // expected-error {{xtheadmatrix md register must be an accumulator register (index 4-7)}}
}

void test_store_constraints(void *ptr) {
  // Store A: ms3 must be tile (0-3) — valid
  __builtin_riscv_th_msae8(0, ptr, 16);  // OK

  // Store A: ms3 must be tile (0-3) — invalid
  __builtin_riscv_th_msae8(4, ptr, 16);  // expected-error {{xtheadmatrix ms3 register must be a tile register (index 0-3)}}

  // Store C: ms3 must be acc (4-7) — valid
  __builtin_riscv_th_msce32(4, ptr, 16); // OK

  // Store C: ms3 must be acc (4-7) — invalid
  __builtin_riscv_th_msce32(0, ptr, 16); // expected-error {{xtheadmatrix ms3 register must be an accumulator register (index 4-7)}}
}

void test_matmul_constraints(void) {
  // Matmul: md must be acc, ms2/ms1 must be tile — valid
  __builtin_riscv_th_mfmacc_h_e4(4, 1, 0); // OK: acc0, tr1, tr0

  // Matmul: md must be acc — invalid (tile register)
  __builtin_riscv_th_mfmacc_h_e4(0, 1, 0); // expected-error {{xtheadmatrix md register must be an accumulator register (index 4-7)}}

  // Matmul: ms2 must be tile — invalid (acc register)
  __builtin_riscv_th_mfmacc_s_e4(4, 4, 0); // expected-error {{xtheadmatrix ms2 register must be a tile register (index 0-3)}}

  // Matmul: ms1 must be tile — invalid (acc register, using untyped variant)
  __builtin_riscv_th_mfmacc_s_tf32(4, 1, 5); // expected-error {{xtheadmatrix ms1 register must be a tile register (index 0-3)}}
}

void test_ew_constraints(void) {
  // EW: all 3 must be acc — valid
  __builtin_riscv_th_mn4clipl_w_mm(4, 5, 6); // OK: acc0, acc1, acc2

  // EW: md not acc — invalid
  __builtin_riscv_th_mn4clipl_w_mm(0, 5, 6); // expected-error {{xtheadmatrix md register must be an accumulator register (index 4-7)}}

  // EW: ms2 not acc — invalid
  __builtin_riscv_th_mn4clipl_w_mm(4, 0, 6); // expected-error {{xtheadmatrix ms2 register must be an accumulator register (index 4-7)}}

  // EW: ms1 not acc — invalid
  __builtin_riscv_th_mn4clipl_w_mm(4, 5, 0); // expected-error {{xtheadmatrix ms1 register must be an accumulator register (index 4-7)}}
}

void test_conversion_constraints(void) {
  // Conversions: md/ms1 must be acc — valid
  __builtin_riscv_th_mfcvtl_h_e4(4, 5); // OK: acc0, acc1

  // Conversions: md not acc — invalid
  __builtin_riscv_th_mfcvtl_h_e4(0, 5); // expected-error {{xtheadmatrix md register must be an accumulator register (index 4-7)}}

  // Conversions: ms1 not acc — invalid
  __builtin_riscv_th_mfcvtl_h_e4(4, 0); // expected-error {{xtheadmatrix ms1 register must be an accumulator register (index 4-7)}}
}

void test_range_constraint(void) {
  // Out of range (>7) — should trigger range error
  __builtin_riscv_th_mzero(8);  // expected-error {{argument value 8 is outside the valid range [0, 7]}}

  // Negative value (as unsigned int, but Sema checks signed range)
  __builtin_riscv_th_mmov_mm(0, 9); // expected-error {{argument value 9 is outside the valid range [0, 7]}}
}

void test_any_register(void) {
  // mzero/mmov/mdup accept any register (0-7)
  __builtin_riscv_th_mzero(0); // OK: tr0
  __builtin_riscv_th_mzero(4); // OK: acc0
  __builtin_riscv_th_mzero(7); // OK: acc3

  // Whole-register load/store accept any register
  __builtin_riscv_th_mlme8(0, (void *)0); // OK: tr0
  __builtin_riscv_th_mlme8(4, (void *)0); // OK: acc0

  // Slides/broadcasts accept any register
  __builtin_riscv_th_mrslidedown(0, 1, 3); // OK
  __builtin_riscv_th_mrslidedown(4, 5, 3); // OK: acc registers
}

// =========================================================================
// Extended Sema tests — more instruction categories and edge cases
// =========================================================================

void test_store_tile_b_constraint(void *ptr) {
  // Store B: ms3 must be tile (0-3) — valid
  __builtin_riscv_th_msbe16(1, ptr, 16); // OK: tr1

  // Store B: ms3 must be tile (0-3) — invalid
  __builtin_riscv_th_msbe16(4, ptr, 16); // expected-error {{xtheadmatrix ms3 register must be a tile register (index 0-3)}}

  // Store B^T: ms3 must be tile (0-3) — invalid
  __builtin_riscv_th_msbte32(5, ptr, 16); // expected-error {{xtheadmatrix ms3 register must be a tile register (index 0-3)}}
}

void test_store_c_transposed_constraint(void *ptr) {
  // Store C^T: ms3 must be acc (4-7) — valid
  __builtin_riscv_th_mscte64(7, ptr, 16); // OK: acc3

  // Store C^T: ms3 must be acc (4-7) — invalid
  __builtin_riscv_th_mscte64(3, ptr, 16); // expected-error {{xtheadmatrix ms3 register must be an accumulator register (index 4-7)}}
}

void test_int_matmul_constraints(void) {
  // INT matmul: all three register constraints
  __builtin_riscv_th_mfmacc_s_bf16(4, 0, 1);  // OK: acc0, tr0, tr1
  __builtin_riscv_th_mfmacc_s_bf16(3, 0, 1);  // expected-error {{xtheadmatrix md register must be an accumulator register (index 4-7)}}

  __builtin_riscv_th_mfmacc_h_e5(7, 3, 2);      // OK: acc3, tr3, tr2 (untyped)
  __builtin_riscv_th_mfmacc_bf16_e5(4, 5, 0);  // expected-error {{xtheadmatrix ms2 register must be a tile register (index 0-3)}}
}

void test_ew_mvi_constraints(void) {
  // EW MVI: all 3 reg indices must be acc
  __builtin_riscv_th_mn4clipl_w_mv_i(4, 5, 6, 3); // OK: acc0, acc1, acc2, imm=3

  __builtin_riscv_th_mn4clipl_w_mv_i(0, 5, 6, 3); // expected-error {{xtheadmatrix md register must be an accumulator register (index 4-7)}}
  __builtin_riscv_th_mn4cliph_w_mv_i(4, 0, 6, 3);  // expected-error {{xtheadmatrix ms2 register must be an accumulator register (index 4-7)}}
  __builtin_riscv_th_mn4cliplu_w_mv_i(4, 5, 0, 3); // expected-error {{xtheadmatrix ms1 register must be an accumulator register (index 4-7)}}
}

void test_packed_conversion_constraints(void) {
  // Packed conversions: md/ms1 must be acc
  __builtin_riscv_th_mucvtl_b_p(4, 5);  // OK: acc0, acc1
  __builtin_riscv_th_mucvtl_b_p(0, 5);  // expected-error {{xtheadmatrix md register must be an accumulator register (index 4-7)}}
  __builtin_riscv_th_mscvth_b_p(4, 0);  // expected-error {{xtheadmatrix ms1 register must be an accumulator register (index 4-7)}}
}

void test_float_int_conversion_constraints(void) {
  // Float-int conversions: md/ms1 must be acc
  __builtin_riscv_th_mfcvt_s_tf32(4, 5);  // OK: acc0, acc1
  __builtin_riscv_th_mfcvt_s_tf32(0, 5);  // expected-error {{xtheadmatrix md register must be an accumulator register (index 4-7)}}
  __builtin_riscv_th_mfcvt_tf32_s(4, 3);  // expected-error {{xtheadmatrix ms1 register must be an accumulator register (index 4-7)}}
}

void test_whole_register_boundary(void *ptr) {
  // Whole-register loads/stores accept any register including boundaries
  __builtin_riscv_th_mlme32(0, ptr);  // OK: tr0 (lower bound)
  __builtin_riscv_th_mlme32(7, ptr);  // OK: acc3 (upper bound)
  __builtin_riscv_th_msme16(0, ptr);  // OK: tr0
  __builtin_riscv_th_msme16(7, ptr);  // OK: acc3
}

void test_pack_any_register(void) {
  // Pack: all 3 args accept any register (0-7)
  __builtin_riscv_th_mpack(0, 1, 2);    // OK: tr0, tr1, tr2
  __builtin_riscv_th_mpack(4, 5, 6);    // OK: acc0, acc1, acc2 (also valid)
  __builtin_riscv_th_mpackhl(7, 0, 3);  // OK: acc3, tr0, tr3
}

void test_broadcast_any_register(void) {
  // Broadcasts: 2 reg indices accept any register (0-7)
  __builtin_riscv_th_mcbcab_mv_i(0, 1, 3);  // OK: tr0, tr1
  __builtin_riscv_th_mcbcab_mv_i(4, 5, 3);  // OK: acc0, acc1
  __builtin_riscv_th_mcbcaw_mv_i(7, 0, 1);  // OK: acc3, tr0
}

void test_mmov_x_m_any_register(void) {
  // mmov.x.m: ms2 register accepts any (0-7)
  (void)__builtin_riscv_th_mmovw_x_m(0, 0);  // OK: tr0
  (void)__builtin_riscv_th_mmovw_x_m(7, 0);  // OK: acc3
}

void test_mmov_m_x_any_register(void) {
  // mmov.m.x: md register accepts any (0-7)
  __builtin_riscv_th_mmovw_m_x(0, 42, 0);  // OK: tr0
  __builtin_riscv_th_mmovw_m_x(7, 42, 0);  // OK: acc3
}

void test_mdup_any_register(void) {
  // mdup: md register accepts any (0-7)
  __builtin_riscv_th_mdupw_m_x(0, 42);  // OK: tr0
  __builtin_riscv_th_mdupw_m_x(7, 42);  // OK: acc3
}

void test_load_all_eew_constraints(void *ptr) {
  // All EEW widths for load A — all must be tile
  __builtin_riscv_th_mlae8(4, ptr, 16);  // expected-error {{xtheadmatrix md register must be a tile register (index 0-3)}}
  __builtin_riscv_th_mlae16(5, ptr, 16); // expected-error {{xtheadmatrix md register must be a tile register (index 0-3)}}
  __builtin_riscv_th_mlae32(6, ptr, 16); // expected-error {{xtheadmatrix md register must be a tile register (index 0-3)}}
  __builtin_riscv_th_mlae64(7, ptr, 16); // expected-error {{xtheadmatrix md register must be a tile register (index 0-3)}}

  // All EEW widths for load C — all must be acc
  __builtin_riscv_th_mlce8(0, ptr, 16);  // expected-error {{xtheadmatrix md register must be an accumulator register (index 4-7)}}
  __builtin_riscv_th_mlce16(1, ptr, 16); // expected-error {{xtheadmatrix md register must be an accumulator register (index 4-7)}}
  __builtin_riscv_th_mlce32(2, ptr, 16); // expected-error {{xtheadmatrix md register must be an accumulator register (index 4-7)}}
  __builtin_riscv_th_mlce64(3, ptr, 16); // expected-error {{xtheadmatrix md register must be an accumulator register (index 4-7)}}
}
