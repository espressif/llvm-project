//===- IntrinsicsRISCVXTHeadMatrix.td - T-Head Matrix ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the T-Head Matrix (RVM 0.6) vendor intrinsics for RISC-V.
//
// Uses the ManagedRA (Spec-API) programming model exclusively. Matrix
// operations use target("riscv.matrix") SSA values and the register allocator
// manages matrix registers (TR0-TR3, ACC0-ACC3).
//
// Contents:
//   1. Configuration intrinsics (mrelease, msettilem/k/n variants)
//   2. _internal intrinsics returning/consuming target("riscv.matrix") values
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "riscv" in {

//===----------------------------------------------------------------------===//
// Configuration intrinsics (no matrix register operands)
//===----------------------------------------------------------------------===//

// mrelease - release matrix state
def int_riscv_th_mrelease
    : DefaultAttrsIntrinsic<[], [], [IntrHasSideEffects]>;

// msettilemi - set tile M dimension with immediate (no rd, bits[11:7]=00000)
def int_riscv_th_msettilemi
    : DefaultAttrsIntrinsic<[], [llvm_anyint_ty],
                            [IntrHasSideEffects, ImmArg<ArgIndex<0>>]>;

// msettilem - set tile M dimension from GPR (no rd, bits[11:7]=00000)
def int_riscv_th_msettilem
    : DefaultAttrsIntrinsic<[], [llvm_anyint_ty],
                            [IntrHasSideEffects]>;

// msettileki - set tile K dimension with immediate (no rd, bits[11:7]=00000)
def int_riscv_th_msettileki
    : DefaultAttrsIntrinsic<[], [llvm_anyint_ty],
                            [IntrHasSideEffects, ImmArg<ArgIndex<0>>]>;

// msettilek - set tile K dimension from GPR (no rd, bits[11:7]=00000)
def int_riscv_th_msettilek
    : DefaultAttrsIntrinsic<[], [llvm_anyint_ty],
                            [IntrHasSideEffects]>;

// msettileni - set tile N dimension with immediate (no rd, bits[11:7]=00000)
def int_riscv_th_msettileni
    : DefaultAttrsIntrinsic<[], [llvm_anyint_ty],
                            [IntrHasSideEffects, ImmArg<ArgIndex<0>>]>;

// msettilen - set tile N dimension from GPR (no rd, bits[11:7]=00000)
def int_riscv_th_msettilen
    : DefaultAttrsIntrinsic<[], [llvm_anyint_ty],
                            [IntrHasSideEffects]>;

// ManagedRA (SSA) _internal intrinsic helper classes
//===----------------------------------------------------------------------===//
// These _internal variants return/consume target("riscv.matrix") SSA values
// instead of using ImmArg register indices. Used for the ManagedRA programming
// model where the register allocator manages matrix registers.

// Load (produces matrix value): (ptr, stride) -> matrix
class THMatrix_Internal_Load
    : DefaultAttrsIntrinsic<[llvm_any_ty],
                            [llvm_ptr_ty, llvm_anyint_ty],
                            [NoCapture<ArgIndex<0>>]>;

// Load whole-register (produces matrix value): (ptr) -> matrix
class THMatrix_Internal_LoadWhole
    : DefaultAttrsIntrinsic<[llvm_any_ty],
                            [llvm_ptr_ty],
                            [NoCapture<ArgIndex<0>>]>;

// Store (consumes matrix value): (matrix, ptr, stride) -> void
class THMatrix_Internal_Store
    : DefaultAttrsIntrinsic<[],
                            [llvm_any_ty, llvm_ptr_ty, llvm_anyint_ty],
                            [NoCapture<ArgIndex<1>>]>;

// Store whole-register (consumes matrix value): (matrix, ptr) -> void
class THMatrix_Internal_StoreWhole
    : DefaultAttrsIntrinsic<[],
                            [llvm_any_ty, llvm_ptr_ty],
                            [NoCapture<ArgIndex<1>>]>;

// Zero (produces matrix value): () -> matrix
class THMatrix_Internal_Zero
    : DefaultAttrsIntrinsic<[llvm_any_ty], [], []>;

// Accumulate op (3 matrix inputs, tied acc): (acc, ms2, ms1) -> acc_out
class THMatrix_Internal_MulAcc
    : DefaultAttrsIntrinsic<[llvm_any_ty],
                            [LLVMMatchType<0>, llvm_any_ty, llvm_any_ty],
                            []>;

// Unary matrix op (1 input, 1 output): (ms1) -> md
class THMatrix_Internal_Unary
    : DefaultAttrsIntrinsic<[llvm_any_ty], [llvm_any_ty], []>;

// Binary matrix op, no accumulate (2 inputs, 1 output): (ms2, ms1) -> md
class THMatrix_Internal_Binary
    : DefaultAttrsIntrinsic<[llvm_any_ty], [llvm_any_ty, llvm_any_ty], []>;

// Matrix-to-GPR: (matrix, gpr_idx) -> gpr
class THMatrix_Internal_ToGPR
    : DefaultAttrsIntrinsic<[llvm_anyint_ty],
                            [llvm_any_ty, LLVMMatchType<0>],
                            []>;

// GPR-to-matrix element (modify): (matrix_in, gpr_data, gpr_idx) -> matrix_out
class THMatrix_Internal_FromGPR2
    : DefaultAttrsIntrinsic<[llvm_any_ty],
                            [LLVMMatchType<0>, llvm_anyint_ty, LLVMMatchType<1>],
                            []>;

// GPR-to-matrix broadcast: (matrix_in, gpr_data) -> matrix_out
class THMatrix_Internal_FromGPR
    : DefaultAttrsIntrinsic<[llvm_any_ty],
                            [LLVMMatchType<0>, llvm_anyint_ty],
                            []>;

// Unary + imm (slide/broadcast): (ms1, imm) -> md
class THMatrix_Internal_UnaryImm
    : DefaultAttrsIntrinsic<[llvm_any_ty],
                            [llvm_any_ty, llvm_anyint_ty],
                            []>;

// Accumulate + imm: (acc, ms2, ms1, imm) -> acc_out
class THMatrix_Internal_MulAccImm
    : DefaultAttrsIntrinsic<[llvm_any_ty],
                            [LLVMMatchType<0>, llvm_any_ty, llvm_any_ty,
                             llvm_anyint_ty],
                            []>;

//===----------------------------------------------------------------------===//
// ManagedRA _internal load intrinsics
//===----------------------------------------------------------------------===//

foreach eew = [8, 16, 32, 64] in {
  // Load A (element-stride)
  def int_riscv_th_mlae_internal # eew  : THMatrix_Internal_Load;
  // Load B (element-stride)
  def int_riscv_th_mlbe_internal # eew  : THMatrix_Internal_Load;
  // Load C (element-stride)
  def int_riscv_th_mlce_internal # eew  : THMatrix_Internal_Load;
  // Load A (tile-stride)
  def int_riscv_th_mlate_internal # eew : THMatrix_Internal_Load;
  // Load B (tile-stride)
  def int_riscv_th_mlbte_internal # eew : THMatrix_Internal_Load;
  // Load C (tile-stride)
  def int_riscv_th_mlcte_internal # eew : THMatrix_Internal_Load;
  // Load M (whole-register)
  def int_riscv_th_mlme_internal # eew  : THMatrix_Internal_LoadWhole;
}

//===----------------------------------------------------------------------===//
// ManagedRA _internal store intrinsics
//===----------------------------------------------------------------------===//

foreach eew = [8, 16, 32, 64] in {
  // Store A (element-stride)
  def int_riscv_th_msae_internal # eew  : THMatrix_Internal_Store;
  // Store B (element-stride)
  def int_riscv_th_msbe_internal # eew  : THMatrix_Internal_Store;
  // Store C (element-stride)
  def int_riscv_th_msce_internal # eew  : THMatrix_Internal_Store;
  // Store A (tile-stride)
  def int_riscv_th_msate_internal # eew : THMatrix_Internal_Store;
  // Store B (tile-stride)
  def int_riscv_th_msbte_internal # eew : THMatrix_Internal_Store;
  // Store C (tile-stride)
  def int_riscv_th_mscte_internal # eew : THMatrix_Internal_Store;
  // Store M (whole-register)
  def int_riscv_th_msme_internal # eew  : THMatrix_Internal_StoreWhole;
}

//===----------------------------------------------------------------------===//
// ManagedRA _internal matmul intrinsics
//===----------------------------------------------------------------------===//
// All matmul: (acc, ms2, ms1) -> acc_out

// FP matmul
def int_riscv_th_mfmacc_h_internal       : THMatrix_Internal_MulAcc;
def int_riscv_th_mfmacc_s_internal       : THMatrix_Internal_MulAcc;
def int_riscv_th_mfmacc_d_internal       : THMatrix_Internal_MulAcc;
def int_riscv_th_mfmacc_h_e4_internal    : THMatrix_Internal_MulAcc;
def int_riscv_th_mfmacc_h_e5_internal    : THMatrix_Internal_MulAcc;
def int_riscv_th_mfmacc_bf16_e4_internal : THMatrix_Internal_MulAcc;
def int_riscv_th_mfmacc_bf16_e5_internal : THMatrix_Internal_MulAcc;
def int_riscv_th_mfmacc_s_h_internal     : THMatrix_Internal_MulAcc;
def int_riscv_th_mfmacc_s_bf16_internal  : THMatrix_Internal_MulAcc;
def int_riscv_th_mfmacc_s_e4_internal    : THMatrix_Internal_MulAcc;
def int_riscv_th_mfmacc_s_e5_internal    : THMatrix_Internal_MulAcc;
def int_riscv_th_mfmacc_s_tf32_internal  : THMatrix_Internal_MulAcc;
def int_riscv_th_mfmacc_d_s_internal     : THMatrix_Internal_MulAcc;

// INT matmul
def int_riscv_th_mmacc_w_b_internal      : THMatrix_Internal_MulAcc;
def int_riscv_th_mmaccu_w_b_internal     : THMatrix_Internal_MulAcc;
def int_riscv_th_mmaccus_w_b_internal    : THMatrix_Internal_MulAcc;
def int_riscv_th_mmaccsu_w_b_internal    : THMatrix_Internal_MulAcc;
def int_riscv_th_mmacc_d_h_internal      : THMatrix_Internal_MulAcc;
def int_riscv_th_mmaccu_d_h_internal     : THMatrix_Internal_MulAcc;
def int_riscv_th_mmaccus_d_h_internal    : THMatrix_Internal_MulAcc;
def int_riscv_th_mmaccsu_d_h_internal    : THMatrix_Internal_MulAcc;

// Partial INT matmul
def int_riscv_th_pmmacc_w_b_internal     : THMatrix_Internal_MulAcc;
def int_riscv_th_pmmaccu_w_b_internal    : THMatrix_Internal_MulAcc;
def int_riscv_th_pmmaccus_w_b_internal   : THMatrix_Internal_MulAcc;
def int_riscv_th_pmmaccsu_w_b_internal   : THMatrix_Internal_MulAcc;

// Bypass INT matmul
def int_riscv_th_mmacc_w_bp_internal     : THMatrix_Internal_MulAcc;
def int_riscv_th_mmaccu_w_bp_internal    : THMatrix_Internal_MulAcc;

//===----------------------------------------------------------------------===//
// ManagedRA _internal misc intrinsics
//===----------------------------------------------------------------------===//

// Zero: () -> matrix
def int_riscv_th_mzero_internal   : THMatrix_Internal_Zero;
def int_riscv_th_mzero2r_internal : THMatrix_Internal_Zero;
def int_riscv_th_mzero4r_internal : THMatrix_Internal_Zero;
def int_riscv_th_mzero8r_internal : THMatrix_Internal_Zero;

// Move matrix-to-matrix: (ms1) -> md
def int_riscv_th_mmov_mm_internal : THMatrix_Internal_Unary;

// Move matrix-element-to-GPR: (matrix, gpr_idx) -> gpr
def int_riscv_th_mmovb_x_m_internal : THMatrix_Internal_ToGPR;
def int_riscv_th_mmovh_x_m_internal : THMatrix_Internal_ToGPR;
def int_riscv_th_mmovw_x_m_internal : THMatrix_Internal_ToGPR;
def int_riscv_th_mmovd_x_m_internal : THMatrix_Internal_ToGPR;

// Move GPR-to-matrix-element: (matrix_in, gpr_data, gpr_idx) -> matrix_out
def int_riscv_th_mmovb_m_x_internal : THMatrix_Internal_FromGPR2;
def int_riscv_th_mmovh_m_x_internal : THMatrix_Internal_FromGPR2;
def int_riscv_th_mmovw_m_x_internal : THMatrix_Internal_FromGPR2;
def int_riscv_th_mmovd_m_x_internal : THMatrix_Internal_FromGPR2;

// Duplicate GPR to matrix column: (matrix_in, gpr_data) -> matrix_out
def int_riscv_th_mdupb_m_x_internal : THMatrix_Internal_FromGPR;
def int_riscv_th_mduph_m_x_internal : THMatrix_Internal_FromGPR;
def int_riscv_th_mdupw_m_x_internal : THMatrix_Internal_FromGPR;
def int_riscv_th_mdupd_m_x_internal : THMatrix_Internal_FromGPR;

// Pack: (ms2, ms1) -> md
def int_riscv_th_mpack_internal   : THMatrix_Internal_Binary;
def int_riscv_th_mpackhl_internal : THMatrix_Internal_Binary;
def int_riscv_th_mpackhh_internal : THMatrix_Internal_Binary;

// Row slide: (ms1, imm) -> md
def int_riscv_th_mrslidedown_internal : THMatrix_Internal_UnaryImm;
def int_riscv_th_mrslideup_internal   : THMatrix_Internal_UnaryImm;

// Column slide: (ms1, imm) -> md
foreach sz = ["b", "h", "w", "d"] in {
  def int_riscv_th_mcslidedown_ # sz # _internal : THMatrix_Internal_UnaryImm;
  def int_riscv_th_mcslideup_ # sz # _internal   : THMatrix_Internal_UnaryImm;
}

// Row broadcast: (ms1, imm) -> md
def int_riscv_th_mrbca_mv_i_internal : THMatrix_Internal_UnaryImm;

// Column broadcast: (ms1, imm) -> md
def int_riscv_th_mcbcab_mv_i_internal : THMatrix_Internal_UnaryImm;
def int_riscv_th_mcbcah_mv_i_internal : THMatrix_Internal_UnaryImm;
def int_riscv_th_mcbcaw_mv_i_internal : THMatrix_Internal_UnaryImm;
def int_riscv_th_mcbcad_mv_i_internal : THMatrix_Internal_UnaryImm;

//===----------------------------------------------------------------------===//
// ManagedRA _internal element-wise intrinsics
//===----------------------------------------------------------------------===//

// FP format conversions: (ms1) -> md
def int_riscv_th_mfcvtl_h_e4_internal : THMatrix_Internal_Unary;
def int_riscv_th_mfcvth_h_e4_internal : THMatrix_Internal_Unary;
def int_riscv_th_mfcvtl_h_e5_internal : THMatrix_Internal_Unary;
def int_riscv_th_mfcvth_h_e5_internal : THMatrix_Internal_Unary;
def int_riscv_th_mfcvtl_e4_h_internal : THMatrix_Internal_Unary;
def int_riscv_th_mfcvth_e4_h_internal : THMatrix_Internal_Unary;
def int_riscv_th_mfcvtl_e5_h_internal : THMatrix_Internal_Unary;
def int_riscv_th_mfcvth_e5_h_internal : THMatrix_Internal_Unary;
def int_riscv_th_mfcvtl_s_h_internal    : THMatrix_Internal_Unary;
def int_riscv_th_mfcvth_s_h_internal    : THMatrix_Internal_Unary;
def int_riscv_th_mfcvtl_h_s_internal    : THMatrix_Internal_Unary;
def int_riscv_th_mfcvth_h_s_internal    : THMatrix_Internal_Unary;
def int_riscv_th_mfcvtl_s_bf16_internal : THMatrix_Internal_Unary;
def int_riscv_th_mfcvth_s_bf16_internal : THMatrix_Internal_Unary;
def int_riscv_th_mfcvtl_bf16_s_internal : THMatrix_Internal_Unary;
def int_riscv_th_mfcvth_bf16_s_internal : THMatrix_Internal_Unary;
def int_riscv_th_mfcvtl_e4_s_internal : THMatrix_Internal_Unary;
def int_riscv_th_mfcvth_e4_s_internal : THMatrix_Internal_Unary;
def int_riscv_th_mfcvtl_e5_s_internal : THMatrix_Internal_Unary;
def int_riscv_th_mfcvth_e5_s_internal : THMatrix_Internal_Unary;
def int_riscv_th_mfcvtl_d_s_internal : THMatrix_Internal_Unary;
def int_riscv_th_mfcvth_d_s_internal : THMatrix_Internal_Unary;
def int_riscv_th_mfcvtl_s_d_internal : THMatrix_Internal_Unary;
def int_riscv_th_mfcvth_s_d_internal : THMatrix_Internal_Unary;
def int_riscv_th_mfcvt_s_tf32_internal : THMatrix_Internal_Unary;
def int_riscv_th_mfcvt_tf32_s_internal : THMatrix_Internal_Unary;

// Float-int conversions: (ms1) -> md
def int_riscv_th_mufcvtl_h_b_internal : THMatrix_Internal_Unary;
def int_riscv_th_mufcvth_h_b_internal : THMatrix_Internal_Unary;
def int_riscv_th_msfcvtl_h_b_internal : THMatrix_Internal_Unary;
def int_riscv_th_msfcvth_h_b_internal : THMatrix_Internal_Unary;
def int_riscv_th_mfucvtl_b_h_internal : THMatrix_Internal_Unary;
def int_riscv_th_mfucvth_b_h_internal : THMatrix_Internal_Unary;
def int_riscv_th_mfscvtl_b_h_internal : THMatrix_Internal_Unary;
def int_riscv_th_mfscvth_b_h_internal : THMatrix_Internal_Unary;
def int_riscv_th_msfcvt_s_w_internal : THMatrix_Internal_Unary;
def int_riscv_th_mufcvt_s_w_internal : THMatrix_Internal_Unary;
def int_riscv_th_mfscvt_w_s_internal : THMatrix_Internal_Unary;
def int_riscv_th_mfucvt_w_s_internal : THMatrix_Internal_Unary;

// Fixed-point clip .mm: (acc, ms2, ms1) -> acc_out
def int_riscv_th_mn4clipl_w_mm_internal  : THMatrix_Internal_MulAcc;
def int_riscv_th_mn4cliph_w_mm_internal  : THMatrix_Internal_MulAcc;
def int_riscv_th_mn4cliplu_w_mm_internal : THMatrix_Internal_MulAcc;
def int_riscv_th_mn4cliphu_w_mm_internal : THMatrix_Internal_MulAcc;

// Fixed-point clip .mv.i: (acc, ms2, ms1, imm) -> acc_out
def int_riscv_th_mn4clipl_w_mv_i_internal  : THMatrix_Internal_MulAccImm;
def int_riscv_th_mn4cliph_w_mv_i_internal  : THMatrix_Internal_MulAccImm;
def int_riscv_th_mn4cliplu_w_mv_i_internal : THMatrix_Internal_MulAccImm;
def int_riscv_th_mn4cliphu_w_mv_i_internal : THMatrix_Internal_MulAccImm;

// Packed conversions: (ms1) -> md
def int_riscv_th_mucvtl_b_p_internal : THMatrix_Internal_Unary;
def int_riscv_th_mscvtl_b_p_internal : THMatrix_Internal_Unary;
def int_riscv_th_mucvth_b_p_internal : THMatrix_Internal_Unary;
def int_riscv_th_mscvth_b_p_internal : THMatrix_Internal_Unary;

// Integer EW .mm: (acc, ms2, ms1) -> acc_out
// Integer EW .mv.i: (acc, ms2, ms1, imm) -> acc_out
foreach op = ["madd", "msub", "mmul", "mmulh",
              "mmax", "mumax", "mmin", "mumin",
              "msrl", "msll", "msra"] in {
  def int_riscv_th_ # op # _w_mm_internal   : THMatrix_Internal_MulAcc;
  def int_riscv_th_ # op # _w_mv_i_internal : THMatrix_Internal_MulAccImm;
}

// FP EW .mm: (acc, ms2, ms1) -> acc_out
// FP EW .mv.i: (acc, ms2, ms1, imm) -> acc_out
foreach op = ["mfadd", "mfsub", "mfmul", "mfmax", "mfmin"] in {
  foreach sz = ["h", "s", "d"] in {
    def int_riscv_th_ # op # _ # sz # _mm_internal   : THMatrix_Internal_MulAcc;
    def int_riscv_th_ # op # _ # sz # _mv_i_internal : THMatrix_Internal_MulAccImm;
  }
}

//===----------------------------------------------------------------------===//
// Zmpanel: Panel-Aware 2x2 Matrix Tiling intrinsics
//===----------------------------------------------------------------------===//

// Panel configuration: (value) -> void
def int_riscv_th_mset22adra : DefaultAttrsIntrinsic<[], [llvm_anyint_ty], [IntrHasSideEffects]>;
def int_riscv_th_mset22adrb : DefaultAttrsIntrinsic<[], [llvm_anyint_ty], [IntrHasSideEffects]>;
def int_riscv_th_mset22adrd : DefaultAttrsIntrinsic<[], [llvm_anyint_ty], [IntrHasSideEffects]>;
def int_riscv_th_mset22rsba : DefaultAttrsIntrinsic<[], [llvm_anyint_ty], [IntrHasSideEffects]>;
def int_riscv_th_mset22rsbb : DefaultAttrsIntrinsic<[], [llvm_anyint_ty], [IntrHasSideEffects]>;
def int_riscv_th_mset22rsbd : DefaultAttrsIntrinsic<[], [llvm_anyint_ty], [IntrHasSideEffects]>;
def int_riscv_th_mset22m    : DefaultAttrsIntrinsic<[], [llvm_anyint_ty], [IntrHasSideEffects]>;
def int_riscv_th_mset22n    : DefaultAttrsIntrinsic<[], [llvm_anyint_ty], [IntrHasSideEffects]>;
def int_riscv_th_mset22k    : DefaultAttrsIntrinsic<[], [llvm_anyint_ty], [IntrHasSideEffects]>;
def int_riscv_th_msetrstptr : DefaultAttrsIntrinsic<[], [llvm_anyint_ty], [IntrHasSideEffects]>;
def int_riscv_th_msetaccum  : DefaultAttrsIntrinsic<[], [llvm_anyint_ty], [IntrHasSideEffects]>;
def int_riscv_th_msetoob    : DefaultAttrsIntrinsic<[], [llvm_anyint_ty], [IntrHasSideEffects]>;

// Panel load: () -> void
def int_riscv_th_ml22e8  : DefaultAttrsIntrinsic<[], [], [IntrHasSideEffects]>;
def int_riscv_th_ml22e16 : DefaultAttrsIntrinsic<[], [], [IntrHasSideEffects]>;

// Panel store: () -> void
def int_riscv_th_msc22e16 : DefaultAttrsIntrinsic<[], [], [IntrHasSideEffects]>;
def int_riscv_th_msc22e32 : DefaultAttrsIntrinsic<[], [], [IntrHasSideEffects]>;

// Panel FP compute: () -> void
def int_riscv_th_mfmacc22_h_e5    : DefaultAttrsIntrinsic<[], [], [IntrHasSideEffects]>;
def int_riscv_th_mfmacc22_h_e4    : DefaultAttrsIntrinsic<[], [], [IntrHasSideEffects]>;
def int_riscv_th_mfmacc22_bf16_e5 : DefaultAttrsIntrinsic<[], [], [IntrHasSideEffects]>;
def int_riscv_th_mfmacc22_bf16_e4 : DefaultAttrsIntrinsic<[], [], [IntrHasSideEffects]>;
def int_riscv_th_mfmacc22_s_e5    : DefaultAttrsIntrinsic<[], [], [IntrHasSideEffects]>;
def int_riscv_th_mfmacc22_s_e4    : DefaultAttrsIntrinsic<[], [], [IntrHasSideEffects]>;
def int_riscv_th_mfmacc22_h       : DefaultAttrsIntrinsic<[], [], [IntrHasSideEffects]>;
def int_riscv_th_mfmacc22_s_h     : DefaultAttrsIntrinsic<[], [], [IntrHasSideEffects]>;
def int_riscv_th_mfmacc22_s_bf16  : DefaultAttrsIntrinsic<[], [], [IntrHasSideEffects]>;
def int_riscv_th_mfmacc22_s       : DefaultAttrsIntrinsic<[], [], [IntrHasSideEffects]>;

// Panel INT compute: () -> void
def int_riscv_th_mmacc22_w_b   : DefaultAttrsIntrinsic<[], [], [IntrHasSideEffects]>;
def int_riscv_th_mmaccu22_w_b  : DefaultAttrsIntrinsic<[], [], [IntrHasSideEffects]>;
def int_riscv_th_mmaccus22_w_b : DefaultAttrsIntrinsic<[], [], [IntrHasSideEffects]>;
def int_riscv_th_mmaccsu22_w_b : DefaultAttrsIntrinsic<[], [], [IntrHasSideEffects]>;

} // TargetPrefix = "riscv"
