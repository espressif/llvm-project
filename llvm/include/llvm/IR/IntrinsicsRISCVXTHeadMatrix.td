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
// Matrix registers are not modeled as LLVM IR types. All matrix operations
// are modeled as side-effect-only intrinsics that operate on implicit matrix
// state. GPR operands (pointers, strides, scalars, immediates) are modeled
// as explicit intrinsic arguments.
//
// Register index parameters (i32 ImmArg) specify which matrix register (0-7)
// to use for each operand. This allows the ISel to read register assignments
// from arguments instead of using hardcoded assignments.
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "riscv" in {

//===----------------------------------------------------------------------===//
// Intrinsic helper classes
//===----------------------------------------------------------------------===//

// Matrix operation with 1 register index (md).
// Used for: mzero, mdup
class THMatrix_1Idx
    : DefaultAttrsIntrinsic<[], [llvm_i32_ty],
                            [IntrHasSideEffects, ImmArg<ArgIndex<0>>]>;

// Matrix operation with 2 register indices (md, ms1).
// Used for: mmov.mm, conversions
class THMatrix_2Idx
    : DefaultAttrsIntrinsic<[], [llvm_i32_ty, llvm_i32_ty],
                            [IntrHasSideEffects,
                             ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

// Matrix operation with 3 register indices (md, ms2, ms1).
// Used for: matmul, pack, EW arithmetic .mm, n4clip .mm
class THMatrix_3Idx
    : DefaultAttrsIntrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                            [IntrHasSideEffects,
                             ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>,
                             ImmArg<ArgIndex<2>>]>;

// Matrix load: register index + base pointer + stride.
// Used for: element-stride loads (mla, mlb, mlc), tile-stride loads
class THMatrix_Load
    : DefaultAttrsIntrinsic<[], [llvm_i32_ty, llvm_ptr_ty, llvm_anyint_ty],
                            [IntrHasSideEffects,
                             ImmArg<ArgIndex<0>>,
                             NoCapture<ArgIndex<1>>]>;

// Matrix load (whole-register): register index + base pointer only.
// Used for: mlme (whole-register load)
class THMatrix_LoadWhole
    : DefaultAttrsIntrinsic<[], [llvm_i32_ty, llvm_ptr_ty],
                            [IntrHasSideEffects,
                             ImmArg<ArgIndex<0>>,
                             NoCapture<ArgIndex<1>>]>;

// Matrix store: register index + base pointer + stride.
// Used for: element-stride stores (msa, msb, msc), tile-stride stores
class THMatrix_Store
    : DefaultAttrsIntrinsic<[], [llvm_i32_ty, llvm_ptr_ty, llvm_anyint_ty],
                            [IntrHasSideEffects, IntrWriteMem,
                             ImmArg<ArgIndex<0>>,
                             NoCapture<ArgIndex<1>>]>;

// Matrix store (whole-register): register index + base pointer only.
// Used for: msme (whole-register store)
class THMatrix_StoreWhole
    : DefaultAttrsIntrinsic<[], [llvm_i32_ty, llvm_ptr_ty],
                            [IntrHasSideEffects, IntrWriteMem,
                             ImmArg<ArgIndex<0>>,
                             NoCapture<ArgIndex<1>>]>;

// Matrix operation with 2 register indices + immediate operand.
// Used for: slide, broadcast
class THMatrix_Imm2Idx
    : DefaultAttrsIntrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_anyint_ty],
                            [IntrHasSideEffects,
                             ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>,
                             ImmArg<ArgIndex<2>>]>;

// Matrix operation with 3 register indices + immediate operand.
// Used for: EW MVI, n4clip MVI
class THMatrix_Imm3Idx
    : DefaultAttrsIntrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                                 llvm_anyint_ty],
                            [IntrHasSideEffects,
                             ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>,
                             ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

// Matrix-to-GPR with register index + element index: rd = matrix[rs1].
// Used for: mmov{b,h,w,d}.x.m (rd, ms2, rs1)
class THMatrix_ToGPR
    : DefaultAttrsIntrinsic<[llvm_anyint_ty], [llvm_i32_ty, LLVMMatchType<0>],
                            [IntrHasSideEffects, ImmArg<ArgIndex<0>>]>;

// GPR-to-matrix with register index, data and element index: matrix[rs1] = rs2.
// Used for: mmov{b,h,w,d}.m.x (md, rs2, rs1)
class THMatrix_FromGPR2
    : DefaultAttrsIntrinsic<[], [llvm_i32_ty, llvm_anyint_ty, LLVMMatchType<0>],
                            [IntrHasSideEffects, ImmArg<ArgIndex<0>>]>;

// GPR-to-matrix with register index: broadcast rs2 to matrix column.
// Used for: mdup{b,h,w,d}.m.x (md, rs2)
class THMatrix_FromGPR
    : DefaultAttrsIntrinsic<[], [llvm_i32_ty, llvm_anyint_ty],
                            [IntrHasSideEffects, ImmArg<ArgIndex<0>>]>;

//===----------------------------------------------------------------------===//
// 2.2 Configuration intrinsics (UNCHANGED — no matrix register operands)
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

//===----------------------------------------------------------------------===//
// 2.3 Load intrinsics
//===----------------------------------------------------------------------===//
// Element-stride and tile-stride loads: (i32 md_idx, ptr base, int stride) -> void.
// Whole-register loads: (i32 md_idx, ptr base) -> void.

foreach eew = [8, 16, 32, 64] in {
  // Load A (element-stride): th.mlae{8,16,32,64}
  def int_riscv_th_mlae # eew  : THMatrix_Load;
  // Load B (element-stride): th.mlbe{8,16,32,64}
  def int_riscv_th_mlbe # eew  : THMatrix_Load;
  // Load C (element-stride): th.mlce{8,16,32,64}
  def int_riscv_th_mlce # eew  : THMatrix_Load;
  // Load A (tile-stride): th.mlate{8,16,32,64}
  def int_riscv_th_mlate # eew : THMatrix_Load;
  // Load B (tile-stride): th.mlbte{8,16,32,64}
  def int_riscv_th_mlbte # eew : THMatrix_Load;
  // Load C (tile-stride): th.mlcte{8,16,32,64}
  def int_riscv_th_mlcte # eew : THMatrix_Load;
  // Load M (whole-register): th.mlme{8,16,32,64}
  def int_riscv_th_mlme # eew  : THMatrix_LoadWhole;
}

//===----------------------------------------------------------------------===//
// 2.3 Store intrinsics
//===----------------------------------------------------------------------===//
// Element-stride and tile-stride stores: (i32 ms3_idx, ptr base, int stride) -> void.
// Whole-register stores: (i32 ms3_idx, ptr base) -> void.

foreach eew = [8, 16, 32, 64] in {
  // Store A (element-stride): th.msae{8,16,32,64}
  def int_riscv_th_msae # eew  : THMatrix_Store;
  // Store B (element-stride): th.msbe{8,16,32,64}
  def int_riscv_th_msbe # eew  : THMatrix_Store;
  // Store C (element-stride): th.msce{8,16,32,64}
  def int_riscv_th_msce # eew  : THMatrix_Store;
  // Store A (tile-stride): th.msate{8,16,32,64}
  def int_riscv_th_msate # eew : THMatrix_Store;
  // Store B (tile-stride): th.msbte{8,16,32,64}
  def int_riscv_th_msbte # eew : THMatrix_Store;
  // Store C (tile-stride): th.mscte{8,16,32,64}
  def int_riscv_th_mscte # eew : THMatrix_Store;
  // Store M (whole-register): th.msme{8,16,32,64}
  def int_riscv_th_msme # eew  : THMatrix_StoreWhole;
}

//===----------------------------------------------------------------------===//
// 2.4 Matrix multiply intrinsics
//===----------------------------------------------------------------------===//
// All matmul: 3 register indices (md, ms2, ms1).

// --- FP matrix multiply-accumulate (func4=0000) ---
def int_riscv_th_mfmacc_h       : THMatrix_3Idx; // FP16
def int_riscv_th_mfmacc_s       : THMatrix_3Idx; // FP32
def int_riscv_th_mfmacc_d       : THMatrix_3Idx; // FP64

// --- Widening FP8 to FP16/BF16 ---
def int_riscv_th_mfmacc_h_e4    : THMatrix_3Idx; // FP8(e4m3)->FP16
def int_riscv_th_mfmacc_h_e5    : THMatrix_3Idx; // FP8(e5m2)->FP16
def int_riscv_th_mfmacc_bf16_e4 : THMatrix_3Idx; // FP8(e4m3)->BF16
def int_riscv_th_mfmacc_bf16_e5 : THMatrix_3Idx; // FP8(e5m2)->BF16

// --- Widening FP16/BF16/FP8/TF32 to FP32 ---
def int_riscv_th_mfmacc_s_h     : THMatrix_3Idx; // FP16->FP32
def int_riscv_th_mfmacc_s_bf16  : THMatrix_3Idx; // BF16->FP32
def int_riscv_th_mfmacc_s_e4    : THMatrix_3Idx; // FP8(e4m3)->FP32
def int_riscv_th_mfmacc_s_e5    : THMatrix_3Idx; // FP8(e5m2)->FP32
def int_riscv_th_mfmacc_s_tf32  : THMatrix_3Idx; // TF32->FP32

// --- Widening FP32 to FP64 ---
def int_riscv_th_mfmacc_d_s     : THMatrix_3Idx; // FP32->FP64

// --- INT8 to INT32 matrix multiply-accumulate (func4=0001) ---
def int_riscv_th_mmacc_w_b      : THMatrix_3Idx; // signed*signed
def int_riscv_th_mmaccu_w_b     : THMatrix_3Idx; // unsigned*unsigned
def int_riscv_th_mmaccus_w_b    : THMatrix_3Idx; // unsigned*signed
def int_riscv_th_mmaccsu_w_b    : THMatrix_3Idx; // signed*unsigned

// --- INT16 to INT64 matrix multiply-accumulate ---
def int_riscv_th_mmacc_d_h      : THMatrix_3Idx; // signed*signed
def int_riscv_th_mmaccu_d_h     : THMatrix_3Idx; // unsigned*unsigned
def int_riscv_th_mmaccus_d_h    : THMatrix_3Idx; // unsigned*signed
def int_riscv_th_mmaccsu_d_h    : THMatrix_3Idx; // signed*unsigned

// --- Partial INT8 to INT32 ---
def int_riscv_th_pmmacc_w_b     : THMatrix_3Idx; // signed*signed
def int_riscv_th_pmmaccu_w_b    : THMatrix_3Idx; // unsigned*unsigned
def int_riscv_th_pmmaccus_w_b   : THMatrix_3Idx; // unsigned*signed
def int_riscv_th_pmmaccsu_w_b   : THMatrix_3Idx; // signed*unsigned

// --- Bypass INT matrix multiply (func4=0010) ---
def int_riscv_th_mmacc_w_bp     : THMatrix_3Idx; // signed
def int_riscv_th_mmaccu_w_bp    : THMatrix_3Idx; // unsigned

//===----------------------------------------------------------------------===//
// 2.5 Misc intrinsics
//===----------------------------------------------------------------------===//

// --- Zero (1 register index: md) ---
def int_riscv_th_mzero   : THMatrix_1Idx;
def int_riscv_th_mzero2r : THMatrix_1Idx;
def int_riscv_th_mzero4r : THMatrix_1Idx;
def int_riscv_th_mzero8r : THMatrix_1Idx;

// --- Move matrix-to-matrix (2 register indices: md, ms1) ---
def int_riscv_th_mmov_mm : THMatrix_2Idx;

// --- Move matrix-element-to-GPR: rd = matrix[rs1] ---
// Intrinsic takes register index (ms2_idx) and GPR index (rs1), returns GPR value (rd)
def int_riscv_th_mmovb_x_m : THMatrix_ToGPR; // byte
def int_riscv_th_mmovh_x_m : THMatrix_ToGPR; // halfword
def int_riscv_th_mmovw_x_m : THMatrix_ToGPR; // word
def int_riscv_th_mmovd_x_m : THMatrix_ToGPR; // doubleword

// --- Move GPR-to-matrix-element: matrix[rs1] = rs2 ---
// Intrinsic takes register index (md_idx), GPR data (rs2) and GPR index (rs1)
def int_riscv_th_mmovb_m_x : THMatrix_FromGPR2; // byte
def int_riscv_th_mmovh_m_x : THMatrix_FromGPR2; // halfword
def int_riscv_th_mmovw_m_x : THMatrix_FromGPR2; // word
def int_riscv_th_mmovd_m_x : THMatrix_FromGPR2; // doubleword

// --- Duplicate GPR to matrix column ---
// Intrinsic takes register index (md_idx) and GPR data (rs2)
def int_riscv_th_mdupb_m_x : THMatrix_FromGPR; // byte
def int_riscv_th_mduph_m_x : THMatrix_FromGPR; // halfword
def int_riscv_th_mdupw_m_x : THMatrix_FromGPR; // word
def int_riscv_th_mdupd_m_x : THMatrix_FromGPR; // doubleword

// --- Pack (3 register indices: md, ms2, ms1) ---
def int_riscv_th_mpack   : THMatrix_3Idx;
def int_riscv_th_mpackhl : THMatrix_3Idx;
def int_riscv_th_mpackhh : THMatrix_3Idx;

// --- Row slide (2 register indices + 3-bit immediate) ---
def int_riscv_th_mrslidedown : THMatrix_Imm2Idx;
def int_riscv_th_mrslideup   : THMatrix_Imm2Idx;

// --- Column slide (2 register indices + 3-bit immediate, per element size) ---
foreach sz = ["b", "h", "w", "d"] in {
  def int_riscv_th_mcslidedown_ # sz : THMatrix_Imm2Idx;
  def int_riscv_th_mcslideup_ # sz   : THMatrix_Imm2Idx;
}

// --- Row broadcast (2 register indices + 3-bit immediate) ---
def int_riscv_th_mrbca_mv_i : THMatrix_Imm2Idx;

// --- Column broadcast (2 register indices + 3-bit immediate, per element size) ---
def int_riscv_th_mcbcab_mv_i : THMatrix_Imm2Idx; // byte
def int_riscv_th_mcbcah_mv_i : THMatrix_Imm2Idx; // halfword
def int_riscv_th_mcbcaw_mv_i : THMatrix_Imm2Idx; // word
def int_riscv_th_mcbcad_mv_i : THMatrix_Imm2Idx; // doubleword

//===----------------------------------------------------------------------===//
// 2.6 Element-wise intrinsics
//===----------------------------------------------------------------------===//

// --- FP format conversions (2 register indices: md, ms1) ---

// FP8 <-> FP16 (l/h widening and narrowing variants)
def int_riscv_th_mfcvtl_h_e4 : THMatrix_2Idx; // lower FP8(e4m3)->FP16
def int_riscv_th_mfcvth_h_e4 : THMatrix_2Idx; // upper FP8(e4m3)->FP16
def int_riscv_th_mfcvtl_h_e5 : THMatrix_2Idx; // lower FP8(e5m2)->FP16
def int_riscv_th_mfcvth_h_e5 : THMatrix_2Idx; // upper FP8(e5m2)->FP16
def int_riscv_th_mfcvtl_e4_h : THMatrix_2Idx; // lower FP16->FP8(e4m3)
def int_riscv_th_mfcvth_e4_h : THMatrix_2Idx; // upper FP16->FP8(e4m3)
def int_riscv_th_mfcvtl_e5_h : THMatrix_2Idx; // lower FP16->FP8(e5m2)
def int_riscv_th_mfcvth_e5_h : THMatrix_2Idx; // upper FP16->FP8(e5m2)

// FP16/BF16 <-> FP32 (l/h widening and narrowing variants)
def int_riscv_th_mfcvtl_s_h    : THMatrix_2Idx; // lower FP16->FP32
def int_riscv_th_mfcvth_s_h    : THMatrix_2Idx; // upper FP16->FP32
def int_riscv_th_mfcvtl_h_s    : THMatrix_2Idx; // lower FP32->FP16
def int_riscv_th_mfcvth_h_s    : THMatrix_2Idx; // upper FP32->FP16
def int_riscv_th_mfcvtl_s_bf16 : THMatrix_2Idx; // lower BF16->FP32
def int_riscv_th_mfcvth_s_bf16 : THMatrix_2Idx; // upper BF16->FP32
def int_riscv_th_mfcvtl_bf16_s : THMatrix_2Idx; // lower FP32->BF16
def int_riscv_th_mfcvth_bf16_s : THMatrix_2Idx; // upper FP32->BF16

// FP32 <-> FP8 (l/h narrowing variants)
def int_riscv_th_mfcvtl_e4_s : THMatrix_2Idx; // lower FP32->FP8(e4m3)
def int_riscv_th_mfcvth_e4_s : THMatrix_2Idx; // upper FP32->FP8(e4m3)
def int_riscv_th_mfcvtl_e5_s : THMatrix_2Idx; // lower FP32->FP8(e5m2)
def int_riscv_th_mfcvth_e5_s : THMatrix_2Idx; // upper FP32->FP8(e5m2)

// FP32 <-> FP64 (l/h widening and narrowing variants)
def int_riscv_th_mfcvtl_d_s : THMatrix_2Idx; // lower FP32->FP64
def int_riscv_th_mfcvth_d_s : THMatrix_2Idx; // upper FP32->FP64
def int_riscv_th_mfcvtl_s_d : THMatrix_2Idx; // lower FP64->FP32
def int_riscv_th_mfcvth_s_d : THMatrix_2Idx; // upper FP64->FP32

// TF32 <-> FP32 (same-width format change, no l/h)
def int_riscv_th_mfcvt_s_tf32 : THMatrix_2Idx; // TF32->FP32
def int_riscv_th_mfcvt_tf32_s : THMatrix_2Idx; // FP32->TF32

// --- Float-int conversions (2 register indices: md, ms1) ---

// INT8 <-> FP16 (l/h widening variants for int->float)
def int_riscv_th_mufcvtl_h_b : THMatrix_2Idx; // lower uint8->FP16
def int_riscv_th_mufcvth_h_b : THMatrix_2Idx; // upper uint8->FP16
def int_riscv_th_msfcvtl_h_b : THMatrix_2Idx; // lower sint8->FP16
def int_riscv_th_msfcvth_h_b : THMatrix_2Idx; // upper sint8->FP16

// FP16 -> INT8 (l/h narrowing variants for float->int)
def int_riscv_th_mfucvtl_b_h : THMatrix_2Idx; // lower FP16->uint8
def int_riscv_th_mfucvth_b_h : THMatrix_2Idx; // upper FP16->uint8
def int_riscv_th_mfscvtl_b_h : THMatrix_2Idx; // lower FP16->sint8
def int_riscv_th_mfscvth_b_h : THMatrix_2Idx; // upper FP16->sint8

// INT32 <-> FP32 (same-width, no l/h)
def int_riscv_th_msfcvt_s_w : THMatrix_2Idx; // sint32->FP32
def int_riscv_th_mufcvt_s_w : THMatrix_2Idx; // uint32->FP32
def int_riscv_th_mfscvt_w_s : THMatrix_2Idx; // FP32->sint32
def int_riscv_th_mfucvt_w_s : THMatrix_2Idx; // FP32->uint32

// --- Fixed-point clip (matrix-matrix, 3 register indices: md, ms2, ms1) ---
def int_riscv_th_mn4clipl_w_mm  : THMatrix_3Idx; // signed lower
def int_riscv_th_mn4cliph_w_mm  : THMatrix_3Idx; // signed upper
def int_riscv_th_mn4cliplu_w_mm : THMatrix_3Idx; // unsigned lower
def int_riscv_th_mn4cliphu_w_mm : THMatrix_3Idx; // unsigned upper

// --- Fixed-point clip (matrix-vector with immediate, 3 register indices + imm) ---
def int_riscv_th_mn4clipl_w_mv_i  : THMatrix_Imm3Idx; // signed lower
def int_riscv_th_mn4cliph_w_mv_i  : THMatrix_Imm3Idx; // signed upper
def int_riscv_th_mn4cliplu_w_mv_i : THMatrix_Imm3Idx; // unsigned lower
def int_riscv_th_mn4cliphu_w_mv_i : THMatrix_Imm3Idx; // unsigned upper

// --- Packed conversions (pair, 2 register indices: md, ms1) ---
def int_riscv_th_mucvtl_b_p : THMatrix_2Idx; // unsigned pack lower
def int_riscv_th_mscvtl_b_p : THMatrix_2Idx; // signed pack lower
def int_riscv_th_mucvth_b_p : THMatrix_2Idx; // unsigned pack upper
def int_riscv_th_mscvth_b_p : THMatrix_2Idx; // signed pack upper

// --- Integer element-wise arithmetic ---
// .w.mm variants (3 register indices: md, ms2, ms1)
// .w.mv.i variants (3 register indices + 3-bit immediate)
foreach op = ["madd", "msub", "mmul", "mmulh",
              "mmax", "mumax", "mmin", "mumin",
              "msrl", "msll", "msra"] in {
  def int_riscv_th_ # op # _w_mm   : THMatrix_3Idx;
  def int_riscv_th_ # op # _w_mv_i : THMatrix_Imm3Idx;
}

// --- FP element-wise arithmetic ---
// .{h,s,d}.mm variants (3 register indices: md, ms2, ms1)
// .{h,s,d}.mv.i variants (3 register indices + 3-bit immediate)
foreach op = ["mfadd", "mfsub", "mfmul", "mfmax", "mfmin"] in {
  foreach sz = ["h", "s", "d"] in {
    def int_riscv_th_ # op # _ # sz # _mm   : THMatrix_3Idx;
    def int_riscv_th_ # op # _ # sz # _mv_i : THMatrix_Imm3Idx;
  }
}

} // TargetPrefix = "riscv"
