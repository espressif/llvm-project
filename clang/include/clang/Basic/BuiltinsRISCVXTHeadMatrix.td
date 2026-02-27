//==- BuiltinsRISCVXTHeadMatrix.td - T-Head Matrix Builtins ----*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the T-Head Matrix (RVM 0.6) builtin function database.
// All builtins are guarded by the "experimental-xtheadmatrix" feature.
//
// Matrix registers are not exposed as C types. All matrix operations work
// on implicit matrix state. GPR operands (pointers, strides, scalars,
// immediates) are passed as explicit arguments.
//
//===----------------------------------------------------------------------===//

class RISCVXTHeadMatrixBuiltin<string prototype, string features = ""> : TargetBuiltin {
  let Spellings = ["__builtin_riscv_th_" # NAME];
  let Prototype = prototype;
  let Features = features;
}

let Attributes = [NoThrow] in {

//===----------------------------------------------------------------------===//
// Configuration builtins
//===----------------------------------------------------------------------===//
def mrelease : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def msettilemi : RISCVXTHeadMatrixBuiltin<"void(size_t)", "experimental-xtheadmatrix">;
def msettilem : RISCVXTHeadMatrixBuiltin<"void(size_t)", "experimental-xtheadmatrix">;
def msettileki : RISCVXTHeadMatrixBuiltin<"void(size_t)", "experimental-xtheadmatrix">;
def msettilek : RISCVXTHeadMatrixBuiltin<"void(size_t)", "experimental-xtheadmatrix">;
def msettileni : RISCVXTHeadMatrixBuiltin<"void(size_t)", "experimental-xtheadmatrix">;
def msettilen : RISCVXTHeadMatrixBuiltin<"void(size_t)", "experimental-xtheadmatrix">;

//===----------------------------------------------------------------------===//
// Load builtins
//===----------------------------------------------------------------------===//
// Element-stride and tile-stride loads: (void *base, size_t stride) -> void
// Whole-register loads: (void *base) -> void
foreach eew = [8, 16, 32, 64] in {
  // Element-stride loads
  def mlae # eew  : RISCVXTHeadMatrixBuiltin<"void(void *, size_t)", "experimental-xtheadmatrix">;
  def mlbe # eew  : RISCVXTHeadMatrixBuiltin<"void(void *, size_t)", "experimental-xtheadmatrix">;
  def mlce # eew  : RISCVXTHeadMatrixBuiltin<"void(void *, size_t)", "experimental-xtheadmatrix">;
  // Tile-stride loads
  def mlate # eew : RISCVXTHeadMatrixBuiltin<"void(void *, size_t)", "experimental-xtheadmatrix">;
  def mlbte # eew : RISCVXTHeadMatrixBuiltin<"void(void *, size_t)", "experimental-xtheadmatrix">;
  def mlcte # eew : RISCVXTHeadMatrixBuiltin<"void(void *, size_t)", "experimental-xtheadmatrix">;
  // Whole-register loads (no stride)
  def mlme # eew  : RISCVXTHeadMatrixBuiltin<"void(void *)", "experimental-xtheadmatrix">;
}

//===----------------------------------------------------------------------===//
// Store builtins
//===----------------------------------------------------------------------===//
// Element-stride and tile-stride stores: (void *base, size_t stride) -> void
// Whole-register stores: (void *base) -> void
foreach eew = [8, 16, 32, 64] in {
  // Element-stride stores
  def msae # eew  : RISCVXTHeadMatrixBuiltin<"void(void *, size_t)", "experimental-xtheadmatrix">;
  def msbe # eew  : RISCVXTHeadMatrixBuiltin<"void(void *, size_t)", "experimental-xtheadmatrix">;
  def msce # eew  : RISCVXTHeadMatrixBuiltin<"void(void *, size_t)", "experimental-xtheadmatrix">;
  // Tile-stride stores
  def msate # eew : RISCVXTHeadMatrixBuiltin<"void(void *, size_t)", "experimental-xtheadmatrix">;
  def msbte # eew : RISCVXTHeadMatrixBuiltin<"void(void *, size_t)", "experimental-xtheadmatrix">;
  def mscte # eew : RISCVXTHeadMatrixBuiltin<"void(void *, size_t)", "experimental-xtheadmatrix">;
  // Whole-register stores (no stride)
  def msme # eew  : RISCVXTHeadMatrixBuiltin<"void(void *)", "experimental-xtheadmatrix">;
}

//===----------------------------------------------------------------------===//
// Matrix multiply builtins: () -> void (all operands are implicit matrix regs)
//===----------------------------------------------------------------------===//

// FP matmul (func4=0000)
def mfmacc_h       : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfmacc_s       : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfmacc_d       : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;

// Widening FP8->FP16/BF16
def mfmacc_h_e4    : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfmacc_h_e5    : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfmacc_bf16_e4 : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfmacc_bf16_e5 : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;

// Widening to FP32
def mfmacc_s_h     : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfmacc_s_bf16  : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfmacc_s_e4    : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfmacc_s_e5    : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfmacc_s_tf32  : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;

// Widening FP32->FP64
def mfmacc_d_s     : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;

// INT8->INT32 (func4=0001)
def mmacc_w_b      : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mmaccu_w_b     : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mmaccus_w_b    : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mmaccsu_w_b    : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;

// INT16->INT64
def mmacc_d_h      : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mmaccu_d_h     : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mmaccus_d_h    : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mmaccsu_d_h    : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;

// Partial INT8->INT32
def pmmacc_w_b     : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def pmmaccu_w_b    : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def pmmaccus_w_b   : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def pmmaccsu_w_b   : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;

// Bypass INT (func4=0010)
def mmacc_w_bp     : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mmaccu_w_bp    : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;

//===----------------------------------------------------------------------===//
// Misc builtins
//===----------------------------------------------------------------------===//

// Zero
def mzero   : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mzero2r : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mzero4r : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mzero8r : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;

// Move matrix-to-matrix
def mmov_mm : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;

// Move matrix-element-to-GPR: rd = matrix[rs1]
// Takes index (rs1), returns element value
def mmovb_x_m : RISCVXTHeadMatrixBuiltin<"size_t(size_t)", "experimental-xtheadmatrix">;
def mmovh_x_m : RISCVXTHeadMatrixBuiltin<"size_t(size_t)", "experimental-xtheadmatrix">;
def mmovw_x_m : RISCVXTHeadMatrixBuiltin<"size_t(size_t)", "experimental-xtheadmatrix">;
def mmovd_x_m : RISCVXTHeadMatrixBuiltin<"size_t(size_t)", "experimental-xtheadmatrix">;

// Move GPR-to-matrix-element: matrix[rs1] = rs2
// Takes data (rs2) and index (rs1)
def mmovb_m_x : RISCVXTHeadMatrixBuiltin<"void(size_t, size_t)", "experimental-xtheadmatrix">;
def mmovh_m_x : RISCVXTHeadMatrixBuiltin<"void(size_t, size_t)", "experimental-xtheadmatrix">;
def mmovw_m_x : RISCVXTHeadMatrixBuiltin<"void(size_t, size_t)", "experimental-xtheadmatrix">;
def mmovd_m_x : RISCVXTHeadMatrixBuiltin<"void(size_t, size_t)", "experimental-xtheadmatrix">;

// Duplicate GPR to matrix column: takes data (rs2)
def mdupb_m_x : RISCVXTHeadMatrixBuiltin<"void(size_t)", "experimental-xtheadmatrix">;
def mduph_m_x : RISCVXTHeadMatrixBuiltin<"void(size_t)", "experimental-xtheadmatrix">;
def mdupw_m_x : RISCVXTHeadMatrixBuiltin<"void(size_t)", "experimental-xtheadmatrix">;
def mdupd_m_x : RISCVXTHeadMatrixBuiltin<"void(size_t)", "experimental-xtheadmatrix">;

// Pack
def mpack   : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mpackhl : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mpackhh : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;

// Row slide (with 3-bit immediate)
def mrslidedown : RISCVXTHeadMatrixBuiltin<"void(unsigned int)", "experimental-xtheadmatrix">;
def mrslideup   : RISCVXTHeadMatrixBuiltin<"void(unsigned int)", "experimental-xtheadmatrix">;

// Column slide (with 3-bit immediate, per element size)
foreach sz = ["b", "h", "w", "d"] in {
  def mcslidedown_ # sz : RISCVXTHeadMatrixBuiltin<"void(unsigned int)", "experimental-xtheadmatrix">;
  def mcslideup_ # sz   : RISCVXTHeadMatrixBuiltin<"void(unsigned int)", "experimental-xtheadmatrix">;
}

// Row broadcast (with 3-bit immediate)
def mrbca_mv_i : RISCVXTHeadMatrixBuiltin<"void(unsigned int)", "experimental-xtheadmatrix">;

// Column broadcast (with 3-bit immediate, per element size)
def mcbcab_mv_i : RISCVXTHeadMatrixBuiltin<"void(unsigned int)", "experimental-xtheadmatrix">;
def mcbcah_mv_i : RISCVXTHeadMatrixBuiltin<"void(unsigned int)", "experimental-xtheadmatrix">;
def mcbcaw_mv_i : RISCVXTHeadMatrixBuiltin<"void(unsigned int)", "experimental-xtheadmatrix">;
def mcbcad_mv_i : RISCVXTHeadMatrixBuiltin<"void(unsigned int)", "experimental-xtheadmatrix">;

//===----------------------------------------------------------------------===//
// Element-wise builtins
//===----------------------------------------------------------------------===//

// --- FP format conversions: () -> void ---

// FP8 <-> FP16 (l/h variants)
def mfcvtl_h_e4 : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfcvth_h_e4 : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfcvtl_h_e5 : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfcvth_h_e5 : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfcvtl_e4_h : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfcvth_e4_h : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfcvtl_e5_h : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfcvth_e5_h : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;

// FP16/BF16 <-> FP32 (l/h variants)
def mfcvtl_s_h    : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfcvth_s_h    : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfcvtl_h_s    : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfcvth_h_s    : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfcvtl_s_bf16 : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfcvth_s_bf16 : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfcvtl_bf16_s : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfcvth_bf16_s : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;

// FP32 <-> FP8 (l/h variants)
def mfcvtl_e4_s : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfcvth_e4_s : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfcvtl_e5_s : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfcvth_e5_s : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;

// FP32 <-> FP64 (l/h variants)
def mfcvtl_d_s : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfcvth_d_s : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfcvtl_s_d : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfcvth_s_d : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;

// TF32 <-> FP32 (same-width, no l/h)
def mfcvt_s_tf32 : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfcvt_tf32_s : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;

// --- Float-int conversions: () -> void ---

// INT8 <-> FP16 (l/h variants)
def mufcvtl_h_b : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mufcvth_h_b : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def msfcvtl_h_b : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def msfcvth_h_b : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfucvtl_b_h : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfucvth_b_h : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfscvtl_b_h : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfscvth_b_h : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;

// INT32 <-> FP32 (same-width, no l/h)
def msfcvt_s_w : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mufcvt_s_w : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfscvt_w_s : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mfucvt_w_s : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;

// --- Fixed-point clip (matrix-matrix): () -> void ---
def mn4clipl_w_mm  : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mn4cliph_w_mm  : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mn4cliplu_w_mm : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mn4cliphu_w_mm : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;

// --- Fixed-point clip (matrix-vector with immediate): (unsigned int) -> void ---
def mn4clipl_w_mv_i  : RISCVXTHeadMatrixBuiltin<"void(unsigned int)", "experimental-xtheadmatrix">;
def mn4cliph_w_mv_i  : RISCVXTHeadMatrixBuiltin<"void(unsigned int)", "experimental-xtheadmatrix">;
def mn4cliplu_w_mv_i : RISCVXTHeadMatrixBuiltin<"void(unsigned int)", "experimental-xtheadmatrix">;
def mn4cliphu_w_mv_i : RISCVXTHeadMatrixBuiltin<"void(unsigned int)", "experimental-xtheadmatrix">;

// --- Packed conversions (pair): () -> void ---
def mucvtl_b_p : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mscvtl_b_p : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mucvth_b_p : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def mscvth_b_p : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;

// --- Integer element-wise arithmetic ---
// .w.mm: () -> void
// .w.mv.i: (unsigned int imm3) -> void
foreach op = ["madd", "msub", "mmul", "mmulh",
              "mmax", "mumax", "mmin", "mumin",
              "msrl", "msll", "msra"] in {
  def op # _w_mm   : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
  def op # _w_mv_i : RISCVXTHeadMatrixBuiltin<"void(unsigned int)", "experimental-xtheadmatrix">;
}

// --- FP element-wise arithmetic ---
// .{h,s,d}.mm: () -> void
// .{h,s,d}.mv.i: (unsigned int imm3) -> void
foreach op = ["mfadd", "mfsub", "mfmul", "mfmax", "mfmin"] in {
  foreach sz = ["h", "s", "d"] in {
    def op # _ # sz # _mm   : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
    def op # _ # sz # _mv_i : RISCVXTHeadMatrixBuiltin<"void(unsigned int)", "experimental-xtheadmatrix">;
  }
}

} // Attributes = [NoThrow]
