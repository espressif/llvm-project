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
// Register index parameters (unsigned int) are prepended to each builtin
// prototype. These specify which matrix register (0-7) to use for each
// operand. The number of register index params depends on the instruction:
//   1 index: loads, stores, mzero, mdup, mmov.m.x, mmov.x.m
//   2 indices: mmov.mm, conversions, slides, broadcasts
//   3 indices: matmul, EW arithmetic, pack, n4clip
//   0 indices: config, mrelease, mundef
//
// Typed builtins use Qm<N> encoding for matrix types:
//   __rvm_int8_t=__rvm_int8_t,    __rvm_int16_t=__rvm_int16_t,   __rvm_int32_t=__rvm_int32_t,   __rvm_int64_t=__rvm_int64_t
//   __rvm_uint8_t=__rvm_uint8_t,   __rvm_uint16_t=__rvm_uint16_t,   __rvm_uint32_t=__rvm_uint32_t,  __rvm_uint64_t=__rvm_uint64_t
//   __rvm_float16_t=__rvm_float16_t, __rvm_float32_t=__rvm_float32_t,  __rvm_float64_t=__rvm_float64_t
//   __rvm_int8x2_t-21 = x2 pair types (same order)
//
//===----------------------------------------------------------------------===//

class RISCVXTHeadMatrixBuiltin<string prototype, string features = ""> : TargetBuiltin {
  let Spellings = ["__builtin_riscv_th_" # NAME];
  let Prototype = prototype;
  let Features = features;
}

let Attributes = [NoThrow] in {

//===----------------------------------------------------------------------===//
// Configuration builtins (UNCHANGED — no matrix register operands)
//===----------------------------------------------------------------------===//
def mrelease : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadmatrix">;
def msettilemi : RISCVXTHeadMatrixBuiltin<"void(size_t)", "experimental-xtheadmatrix">;
def msettilem : RISCVXTHeadMatrixBuiltin<"void(size_t)", "experimental-xtheadmatrix">;
def msettileki : RISCVXTHeadMatrixBuiltin<"void(size_t)", "experimental-xtheadmatrix">;
def msettilek : RISCVXTHeadMatrixBuiltin<"void(size_t)", "experimental-xtheadmatrix">;
def msettileni : RISCVXTHeadMatrixBuiltin<"void(size_t)", "experimental-xtheadmatrix">;
def msettilen : RISCVXTHeadMatrixBuiltin<"void(size_t)", "experimental-xtheadmatrix">;

//===----------------------------------------------------------------------===//
// Load builtins (1 register index: md_idx)
//===----------------------------------------------------------------------===//
// Element-stride loads: (unsigned int md_idx, void *base, size_t stride) -> typed
// Tile-stride loads: (unsigned int md_idx, void *base, size_t stride) -> typed
// Whole-register loads: (unsigned int md_idx, void *base) -> typed
def mlae8   : RISCVXTHeadMatrixBuiltin<"__rvm_int8_t(unsigned int, void *, size_t)", "experimental-xtheadmatrix">;
def mlae16  : RISCVXTHeadMatrixBuiltin<"__rvm_int16_t(unsigned int, void *, size_t)", "experimental-xtheadmatrix">;
def mlae32  : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(unsigned int, void *, size_t)", "experimental-xtheadmatrix">;
def mlae64  : RISCVXTHeadMatrixBuiltin<"__rvm_int64_t(unsigned int, void *, size_t)", "experimental-xtheadmatrix">;
def mlbe8   : RISCVXTHeadMatrixBuiltin<"__rvm_int8_t(unsigned int, void *, size_t)", "experimental-xtheadmatrix">;
def mlbe16  : RISCVXTHeadMatrixBuiltin<"__rvm_int16_t(unsigned int, void *, size_t)", "experimental-xtheadmatrix">;
def mlbe32  : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(unsigned int, void *, size_t)", "experimental-xtheadmatrix">;
def mlbe64  : RISCVXTHeadMatrixBuiltin<"__rvm_int64_t(unsigned int, void *, size_t)", "experimental-xtheadmatrix">;
def mlce8   : RISCVXTHeadMatrixBuiltin<"__rvm_int8_t(unsigned int, void *, size_t)", "experimental-xtheadmatrix">;
def mlce16  : RISCVXTHeadMatrixBuiltin<"__rvm_int16_t(unsigned int, void *, size_t)", "experimental-xtheadmatrix">;
def mlce32  : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(unsigned int, void *, size_t)", "experimental-xtheadmatrix">;
def mlce64  : RISCVXTHeadMatrixBuiltin<"__rvm_int64_t(unsigned int, void *, size_t)", "experimental-xtheadmatrix">;
def mlate8  : RISCVXTHeadMatrixBuiltin<"__rvm_int8_t(unsigned int, void *, size_t)", "experimental-xtheadmatrix">;
def mlate16 : RISCVXTHeadMatrixBuiltin<"__rvm_int16_t(unsigned int, void *, size_t)", "experimental-xtheadmatrix">;
def mlate32 : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(unsigned int, void *, size_t)", "experimental-xtheadmatrix">;
def mlate64 : RISCVXTHeadMatrixBuiltin<"__rvm_int64_t(unsigned int, void *, size_t)", "experimental-xtheadmatrix">;
def mlbte8  : RISCVXTHeadMatrixBuiltin<"__rvm_int8_t(unsigned int, void *, size_t)", "experimental-xtheadmatrix">;
def mlbte16 : RISCVXTHeadMatrixBuiltin<"__rvm_int16_t(unsigned int, void *, size_t)", "experimental-xtheadmatrix">;
def mlbte32 : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(unsigned int, void *, size_t)", "experimental-xtheadmatrix">;
def mlbte64 : RISCVXTHeadMatrixBuiltin<"__rvm_int64_t(unsigned int, void *, size_t)", "experimental-xtheadmatrix">;
def mlcte8  : RISCVXTHeadMatrixBuiltin<"__rvm_int8_t(unsigned int, void *, size_t)", "experimental-xtheadmatrix">;
def mlcte16 : RISCVXTHeadMatrixBuiltin<"__rvm_int16_t(unsigned int, void *, size_t)", "experimental-xtheadmatrix">;
def mlcte32 : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(unsigned int, void *, size_t)", "experimental-xtheadmatrix">;
def mlcte64 : RISCVXTHeadMatrixBuiltin<"__rvm_int64_t(unsigned int, void *, size_t)", "experimental-xtheadmatrix">;
def mlme8   : RISCVXTHeadMatrixBuiltin<"__rvm_int8_t(unsigned int, void *)", "experimental-xtheadmatrix">;
def mlme16  : RISCVXTHeadMatrixBuiltin<"__rvm_int16_t(unsigned int, void *)", "experimental-xtheadmatrix">;
def mlme32  : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(unsigned int, void *)", "experimental-xtheadmatrix">;
def mlme64  : RISCVXTHeadMatrixBuiltin<"__rvm_int64_t(unsigned int, void *)", "experimental-xtheadmatrix">;

//===----------------------------------------------------------------------===//
// Store builtins (1 register index: ms3_idx)
//===----------------------------------------------------------------------===//
// Element-stride and tile-stride stores: (unsigned int ms3_idx, void *base, size_t stride) -> void
// Whole-register stores: (unsigned int ms3_idx, void *base) -> void
foreach eew = [8, 16, 32, 64] in {
  // Element-stride stores
  def msae # eew  : RISCVXTHeadMatrixBuiltin<"void(unsigned int, void *, size_t)", "experimental-xtheadmatrix">;
  def msbe # eew  : RISCVXTHeadMatrixBuiltin<"void(unsigned int, void *, size_t)", "experimental-xtheadmatrix">;
  def msce # eew  : RISCVXTHeadMatrixBuiltin<"void(unsigned int, void *, size_t)", "experimental-xtheadmatrix">;
  // Tile-stride stores
  def msate # eew : RISCVXTHeadMatrixBuiltin<"void(unsigned int, void *, size_t)", "experimental-xtheadmatrix">;
  def msbte # eew : RISCVXTHeadMatrixBuiltin<"void(unsigned int, void *, size_t)", "experimental-xtheadmatrix">;
  def mscte # eew : RISCVXTHeadMatrixBuiltin<"void(unsigned int, void *, size_t)", "experimental-xtheadmatrix">;
  // Whole-register stores (no stride)
  def msme # eew  : RISCVXTHeadMatrixBuiltin<"void(unsigned int, void *)", "experimental-xtheadmatrix">;
}

//===----------------------------------------------------------------------===//
// Matrix multiply builtins (3 register indices: md_idx, ms2_idx, ms1_idx)
//===----------------------------------------------------------------------===//

// --- FP matmul (typed) ---
def mfmacc_h       : RISCVXTHeadMatrixBuiltin<"__rvm_float16_t(unsigned int, unsigned int, unsigned int, __rvm_float16_t, __rvm_float16_t, __rvm_float16_t)", "experimental-xtheadmatrix">;
def mfmacc_s       : RISCVXTHeadMatrixBuiltin<"__rvm_float32_t(unsigned int, unsigned int, unsigned int, __rvm_float32_t, __rvm_float32_t, __rvm_float32_t)", "experimental-xtheadmatrix">;
def mfmacc_d       : RISCVXTHeadMatrixBuiltin<"__rvm_float64_t(unsigned int, unsigned int, unsigned int, __rvm_float64_t, __rvm_float64_t, __rvm_float64_t)", "experimental-xtheadmatrix">;

// Widening FP8->FP16/BF16 (void — no standard types)
def mfmacc_h_e4    : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mfmacc_h_e5    : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mfmacc_bf16_e4 : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mfmacc_bf16_e5 : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int, unsigned int)", "experimental-xtheadmatrix">;

// Widening FP16->FP32 (typed)
def mfmacc_s_h     : RISCVXTHeadMatrixBuiltin<"__rvm_float32_t(unsigned int, unsigned int, unsigned int, __rvm_float32_t, __rvm_float16_t, __rvm_float16_t)", "experimental-xtheadmatrix">;

// Widening to FP32 from BF16/FP8/TF32 (void — no standard types)
def mfmacc_s_bf16  : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mfmacc_s_e4    : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mfmacc_s_e5    : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mfmacc_s_tf32  : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int, unsigned int)", "experimental-xtheadmatrix">;

// Widening FP32->FP64 (typed)
def mfmacc_d_s     : RISCVXTHeadMatrixBuiltin<"__rvm_float64_t(unsigned int, unsigned int, unsigned int, __rvm_float64_t, __rvm_float32_t, __rvm_float32_t)", "experimental-xtheadmatrix">;

// --- INT matmul (typed) ---

// INT8->INT32
def mmacc_w_b      : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(unsigned int, unsigned int, unsigned int, __rvm_int32_t, __rvm_int8_t, __rvm_int8_t)", "experimental-xtheadmatrix">;
def mmaccu_w_b     : RISCVXTHeadMatrixBuiltin<"__rvm_uint32_t(unsigned int, unsigned int, unsigned int, __rvm_uint32_t, __rvm_uint8_t, __rvm_uint8_t)", "experimental-xtheadmatrix">;
def mmaccus_w_b    : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(unsigned int, unsigned int, unsigned int, __rvm_int32_t, __rvm_uint8_t, __rvm_int8_t)", "experimental-xtheadmatrix">;
def mmaccsu_w_b    : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(unsigned int, unsigned int, unsigned int, __rvm_int32_t, __rvm_int8_t, __rvm_uint8_t)", "experimental-xtheadmatrix">;

// INT16->INT64
def mmacc_d_h      : RISCVXTHeadMatrixBuiltin<"__rvm_int64_t(unsigned int, unsigned int, unsigned int, __rvm_int64_t, __rvm_int16_t, __rvm_int16_t)", "experimental-xtheadmatrix">;
def mmaccu_d_h     : RISCVXTHeadMatrixBuiltin<"__rvm_uint64_t(unsigned int, unsigned int, unsigned int, __rvm_uint64_t, __rvm_uint16_t, __rvm_uint16_t)", "experimental-xtheadmatrix">;
def mmaccus_d_h    : RISCVXTHeadMatrixBuiltin<"__rvm_int64_t(unsigned int, unsigned int, unsigned int, __rvm_int64_t, __rvm_uint16_t, __rvm_int16_t)", "experimental-xtheadmatrix">;
def mmaccsu_d_h    : RISCVXTHeadMatrixBuiltin<"__rvm_int64_t(unsigned int, unsigned int, unsigned int, __rvm_int64_t, __rvm_int16_t, __rvm_uint16_t)", "experimental-xtheadmatrix">;

// Partial INT8->INT32
def pmmacc_w_b     : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(unsigned int, unsigned int, unsigned int, __rvm_int32_t, __rvm_int8_t, __rvm_int8_t)", "experimental-xtheadmatrix">;
def pmmaccu_w_b    : RISCVXTHeadMatrixBuiltin<"__rvm_uint32_t(unsigned int, unsigned int, unsigned int, __rvm_uint32_t, __rvm_uint8_t, __rvm_uint8_t)", "experimental-xtheadmatrix">;
def pmmaccus_w_b   : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(unsigned int, unsigned int, unsigned int, __rvm_int32_t, __rvm_uint8_t, __rvm_int8_t)", "experimental-xtheadmatrix">;
def pmmaccsu_w_b   : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(unsigned int, unsigned int, unsigned int, __rvm_int32_t, __rvm_int8_t, __rvm_uint8_t)", "experimental-xtheadmatrix">;

// Bypass INT
def mmacc_w_bp     : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(unsigned int, unsigned int, unsigned int, __rvm_int32_t, __rvm_int8_t, __rvm_int8_t)", "experimental-xtheadmatrix">;
def mmaccu_w_bp    : RISCVXTHeadMatrixBuiltin<"__rvm_uint32_t(unsigned int, unsigned int, unsigned int, __rvm_uint32_t, __rvm_uint8_t, __rvm_uint8_t)", "experimental-xtheadmatrix">;

//===----------------------------------------------------------------------===//
// Misc builtins
//===----------------------------------------------------------------------===//

// Zero (1 register index: md_idx)
def mzero   : RISCVXTHeadMatrixBuiltin<"void(unsigned int)", "experimental-xtheadmatrix">;
def mzero2r : RISCVXTHeadMatrixBuiltin<"void(unsigned int)", "experimental-xtheadmatrix">;
def mzero4r : RISCVXTHeadMatrixBuiltin<"void(unsigned int)", "experimental-xtheadmatrix">;
def mzero8r : RISCVXTHeadMatrixBuiltin<"void(unsigned int)", "experimental-xtheadmatrix">;

// Move matrix-to-matrix (2 register indices: md_idx, ms1_idx)
def mmov_mm : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int)", "experimental-xtheadmatrix">;

// Move matrix-element-to-GPR: rd = matrix[rs1] (1 register index: ms2_idx)
// Takes ms2_idx and index (rs1), returns element value
def mmovb_x_m : RISCVXTHeadMatrixBuiltin<"size_t(unsigned int, size_t)", "experimental-xtheadmatrix">;
def mmovh_x_m : RISCVXTHeadMatrixBuiltin<"size_t(unsigned int, size_t)", "experimental-xtheadmatrix">;
def mmovw_x_m : RISCVXTHeadMatrixBuiltin<"size_t(unsigned int, size_t)", "experimental-xtheadmatrix">;
def mmovd_x_m : RISCVXTHeadMatrixBuiltin<"size_t(unsigned int, size_t)", "experimental-xtheadmatrix">;

// Move GPR-to-matrix-element: matrix[rs1] = rs2 (1 register index: md_idx)
// Takes md_idx, data (rs2) and index (rs1)
def mmovb_m_x : RISCVXTHeadMatrixBuiltin<"void(unsigned int, size_t, size_t)", "experimental-xtheadmatrix">;
def mmovh_m_x : RISCVXTHeadMatrixBuiltin<"void(unsigned int, size_t, size_t)", "experimental-xtheadmatrix">;
def mmovw_m_x : RISCVXTHeadMatrixBuiltin<"void(unsigned int, size_t, size_t)", "experimental-xtheadmatrix">;
def mmovd_m_x : RISCVXTHeadMatrixBuiltin<"void(unsigned int, size_t, size_t)", "experimental-xtheadmatrix">;

// Duplicate GPR to matrix column: (1 register index: md_idx) takes data (rs2)
def mdupb_m_x : RISCVXTHeadMatrixBuiltin<"void(unsigned int, size_t)", "experimental-xtheadmatrix">;
def mduph_m_x : RISCVXTHeadMatrixBuiltin<"void(unsigned int, size_t)", "experimental-xtheadmatrix">;
def mdupw_m_x : RISCVXTHeadMatrixBuiltin<"void(unsigned int, size_t)", "experimental-xtheadmatrix">;
def mdupd_m_x : RISCVXTHeadMatrixBuiltin<"void(unsigned int, size_t)", "experimental-xtheadmatrix">;

// Pack (3 register indices: md_idx, ms2_idx, ms1_idx)
def mpack   : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mpackhl : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mpackhh : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int, unsigned int)", "experimental-xtheadmatrix">;

// Row slide (2 register indices + 3-bit immediate)
def mrslidedown : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mrslideup   : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int, unsigned int)", "experimental-xtheadmatrix">;

// Column slide (2 register indices + 3-bit immediate, per element size)
foreach sz = ["b", "h", "w", "d"] in {
  def mcslidedown_ # sz : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int, unsigned int)", "experimental-xtheadmatrix">;
  def mcslideup_ # sz   : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int, unsigned int)", "experimental-xtheadmatrix">;
}

// Row broadcast (2 register indices + 3-bit immediate)
def mrbca_mv_i : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int, unsigned int)", "experimental-xtheadmatrix">;

// Column broadcast (2 register indices + 3-bit immediate, per element size)
def mcbcab_mv_i : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mcbcah_mv_i : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mcbcaw_mv_i : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mcbcad_mv_i : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int, unsigned int)", "experimental-xtheadmatrix">;

//===----------------------------------------------------------------------===//
// Element-wise builtins
//===----------------------------------------------------------------------===//

// --- FP format conversions (2 register indices: md_idx, ms1_idx) ---

// FP8 <-> FP16 (void — no standard FP8 types)
def mfcvtl_h_e4 : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mfcvth_h_e4 : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mfcvtl_h_e5 : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mfcvth_h_e5 : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mfcvtl_e4_h : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mfcvth_e4_h : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mfcvtl_e5_h : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mfcvth_e5_h : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int)", "experimental-xtheadmatrix">;

// FP16 <-> FP32 (typed: 2 reg indices + type args)
def mfcvtl_s_h    : RISCVXTHeadMatrixBuiltin<"__rvm_float32_t(unsigned int, unsigned int, __rvm_float16_t)", "experimental-xtheadmatrix">;
def mfcvth_s_h    : RISCVXTHeadMatrixBuiltin<"__rvm_float32_t(unsigned int, unsigned int, __rvm_float16_t)", "experimental-xtheadmatrix">;
def mfcvtl_h_s    : RISCVXTHeadMatrixBuiltin<"__rvm_float16_t(unsigned int, unsigned int, __rvm_float32_t)", "experimental-xtheadmatrix">;
def mfcvth_h_s    : RISCVXTHeadMatrixBuiltin<"__rvm_float16_t(unsigned int, unsigned int, __rvm_float32_t)", "experimental-xtheadmatrix">;

// BF16 <-> FP32 (void — no standard BF16 types)
def mfcvtl_s_bf16 : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mfcvth_s_bf16 : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mfcvtl_bf16_s : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mfcvth_bf16_s : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int)", "experimental-xtheadmatrix">;

// FP32 <-> FP8 (void — no standard FP8 types)
def mfcvtl_e4_s : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mfcvth_e4_s : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mfcvtl_e5_s : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mfcvth_e5_s : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int)", "experimental-xtheadmatrix">;

// FP32 <-> FP64 (typed)
def mfcvtl_d_s : RISCVXTHeadMatrixBuiltin<"__rvm_float64_t(unsigned int, unsigned int, __rvm_float32_t)", "experimental-xtheadmatrix">;
def mfcvth_d_s : RISCVXTHeadMatrixBuiltin<"__rvm_float64_t(unsigned int, unsigned int, __rvm_float32_t)", "experimental-xtheadmatrix">;
def mfcvtl_s_d : RISCVXTHeadMatrixBuiltin<"__rvm_float32_t(unsigned int, unsigned int, __rvm_float64_t)", "experimental-xtheadmatrix">;
def mfcvth_s_d : RISCVXTHeadMatrixBuiltin<"__rvm_float32_t(unsigned int, unsigned int, __rvm_float64_t)", "experimental-xtheadmatrix">;

// TF32 <-> FP32 (void — no standard TF32 types)
def mfcvt_s_tf32 : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mfcvt_tf32_s : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int)", "experimental-xtheadmatrix">;

// --- Float-int conversions (2 register indices: md_idx, ms1_idx) ---

// UINT8 <-> FP16 (typed)
def mufcvtl_h_b : RISCVXTHeadMatrixBuiltin<"__rvm_float16_t(unsigned int, unsigned int, __rvm_uint8_t)", "experimental-xtheadmatrix">;
def mufcvth_h_b : RISCVXTHeadMatrixBuiltin<"__rvm_float16_t(unsigned int, unsigned int, __rvm_uint8_t)", "experimental-xtheadmatrix">;
def mfucvtl_b_h : RISCVXTHeadMatrixBuiltin<"__rvm_uint8_t(unsigned int, unsigned int, __rvm_float16_t)", "experimental-xtheadmatrix">;
def mfucvth_b_h : RISCVXTHeadMatrixBuiltin<"__rvm_uint8_t(unsigned int, unsigned int, __rvm_float16_t)", "experimental-xtheadmatrix">;

// SINT8 <-> FP16 (typed)
def msfcvtl_h_b : RISCVXTHeadMatrixBuiltin<"__rvm_float16_t(unsigned int, unsigned int, __rvm_int8_t)", "experimental-xtheadmatrix">;
def msfcvth_h_b : RISCVXTHeadMatrixBuiltin<"__rvm_float16_t(unsigned int, unsigned int, __rvm_int8_t)", "experimental-xtheadmatrix">;
def mfscvtl_b_h : RISCVXTHeadMatrixBuiltin<"__rvm_int8_t(unsigned int, unsigned int, __rvm_float16_t)", "experimental-xtheadmatrix">;
def mfscvth_b_h : RISCVXTHeadMatrixBuiltin<"__rvm_int8_t(unsigned int, unsigned int, __rvm_float16_t)", "experimental-xtheadmatrix">;

// INT32 <-> FP32 (typed)
def msfcvt_s_w : RISCVXTHeadMatrixBuiltin<"__rvm_float32_t(unsigned int, unsigned int, __rvm_int32_t)", "experimental-xtheadmatrix">;
def mufcvt_s_w : RISCVXTHeadMatrixBuiltin<"__rvm_float32_t(unsigned int, unsigned int, __rvm_uint32_t)", "experimental-xtheadmatrix">;
def mfscvt_w_s : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(unsigned int, unsigned int, __rvm_float32_t)", "experimental-xtheadmatrix">;
def mfucvt_w_s : RISCVXTHeadMatrixBuiltin<"__rvm_uint32_t(unsigned int, unsigned int, __rvm_float32_t)", "experimental-xtheadmatrix">;

// --- Fixed-point clip (matrix-matrix): 3 register indices ---
def mn4clipl_w_mm  : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mn4cliph_w_mm  : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mn4cliplu_w_mm : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mn4cliphu_w_mm : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int, unsigned int)", "experimental-xtheadmatrix">;

// --- Fixed-point clip (matrix-vector with immediate): 3 register indices + imm ---
def mn4clipl_w_mv_i  : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int, unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mn4cliph_w_mv_i  : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int, unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mn4cliplu_w_mv_i : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int, unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mn4cliphu_w_mv_i : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int, unsigned int, unsigned int)", "experimental-xtheadmatrix">;

// --- Packed conversions (pair): 2 register indices ---
def mucvtl_b_p : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mscvtl_b_p : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mucvth_b_p : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int)", "experimental-xtheadmatrix">;
def mscvth_b_p : RISCVXTHeadMatrixBuiltin<"void(unsigned int, unsigned int)", "experimental-xtheadmatrix">;

// --- Integer element-wise arithmetic (typed: int32) ---
// .w.mm: 3 reg indices + __rvm_int32_t args
// .w.mv.i: 3 reg indices + __rvm_int32_t args + unsigned int imm
foreach op = ["madd", "msub", "mmul", "mmulh",
              "mmax", "mumax", "mmin", "mumin",
              "msrl", "msll", "msra"] in {
  def op # _w_mm   : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(unsigned int, unsigned int, unsigned int, __rvm_int32_t, __rvm_int32_t, __rvm_int32_t)", "experimental-xtheadmatrix">;
  def op # _w_mv_i : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(unsigned int, unsigned int, unsigned int, __rvm_int32_t, __rvm_int32_t, unsigned int)", "experimental-xtheadmatrix">;
}

// --- FP element-wise arithmetic (typed by precision) ---
// .h.mm/.s.mm/.d.mm: 3 reg indices + type args
// .h.mv.i/.s.mv.i/.d.mv.i: 3 reg indices + type args + unsigned int imm
foreach op = ["mfadd", "mfsub", "mfmul", "mfmax", "mfmin"] in {
  def op # _h_mm   : RISCVXTHeadMatrixBuiltin<"__rvm_float16_t(unsigned int, unsigned int, unsigned int, __rvm_float16_t, __rvm_float16_t, __rvm_float16_t)", "experimental-xtheadmatrix">;
  def op # _h_mv_i : RISCVXTHeadMatrixBuiltin<"__rvm_float16_t(unsigned int, unsigned int, unsigned int, __rvm_float16_t, __rvm_float16_t, unsigned int)", "experimental-xtheadmatrix">;
  def op # _s_mm   : RISCVXTHeadMatrixBuiltin<"__rvm_float32_t(unsigned int, unsigned int, unsigned int, __rvm_float32_t, __rvm_float32_t, __rvm_float32_t)", "experimental-xtheadmatrix">;
  def op # _s_mv_i : RISCVXTHeadMatrixBuiltin<"__rvm_float32_t(unsigned int, unsigned int, unsigned int, __rvm_float32_t, __rvm_float32_t, unsigned int)", "experimental-xtheadmatrix">;
  def op # _d_mm   : RISCVXTHeadMatrixBuiltin<"__rvm_float64_t(unsigned int, unsigned int, unsigned int, __rvm_float64_t, __rvm_float64_t, __rvm_float64_t)", "experimental-xtheadmatrix">;
  def op # _d_mv_i : RISCVXTHeadMatrixBuiltin<"__rvm_float64_t(unsigned int, unsigned int, unsigned int, __rvm_float64_t, __rvm_float64_t, unsigned int)", "experimental-xtheadmatrix">;
}

//===----------------------------------------------------------------------===//
// Undefined value constructors (22 builtins — UNCHANGED, no register indices)
//===----------------------------------------------------------------------===//
def mundef_i8     : RISCVXTHeadMatrixBuiltin<"__rvm_int8_t()", "experimental-xtheadmatrix">;
def mundef_i16    : RISCVXTHeadMatrixBuiltin<"__rvm_int16_t()", "experimental-xtheadmatrix">;
def mundef_i32    : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t()", "experimental-xtheadmatrix">;
def mundef_i64    : RISCVXTHeadMatrixBuiltin<"__rvm_int64_t()", "experimental-xtheadmatrix">;
def mundef_u8     : RISCVXTHeadMatrixBuiltin<"__rvm_uint8_t()", "experimental-xtheadmatrix">;
def mundef_u16    : RISCVXTHeadMatrixBuiltin<"__rvm_uint16_t()", "experimental-xtheadmatrix">;
def mundef_u32    : RISCVXTHeadMatrixBuiltin<"__rvm_uint32_t()", "experimental-xtheadmatrix">;
def mundef_u64    : RISCVXTHeadMatrixBuiltin<"__rvm_uint64_t()", "experimental-xtheadmatrix">;
def mundef_f16    : RISCVXTHeadMatrixBuiltin<"__rvm_float16_t()", "experimental-xtheadmatrix">;
def mundef_f32    : RISCVXTHeadMatrixBuiltin<"__rvm_float32_t()", "experimental-xtheadmatrix">;
def mundef_f64    : RISCVXTHeadMatrixBuiltin<"__rvm_float64_t()", "experimental-xtheadmatrix">;
def mundef_i8x2   : RISCVXTHeadMatrixBuiltin<"__rvm_int8x2_t()", "experimental-xtheadmatrix">;
def mundef_i16x2  : RISCVXTHeadMatrixBuiltin<"__rvm_int16x2_t()", "experimental-xtheadmatrix">;
def mundef_i32x2  : RISCVXTHeadMatrixBuiltin<"__rvm_int32x2_t()", "experimental-xtheadmatrix">;
def mundef_i64x2  : RISCVXTHeadMatrixBuiltin<"__rvm_int64x2_t()", "experimental-xtheadmatrix">;
def mundef_u8x2   : RISCVXTHeadMatrixBuiltin<"__rvm_uint8x2_t()", "experimental-xtheadmatrix">;
def mundef_u16x2  : RISCVXTHeadMatrixBuiltin<"__rvm_uint16x2_t()", "experimental-xtheadmatrix">;
def mundef_u32x2  : RISCVXTHeadMatrixBuiltin<"__rvm_uint32x2_t()", "experimental-xtheadmatrix">;
def mundef_u64x2  : RISCVXTHeadMatrixBuiltin<"__rvm_uint64x2_t()", "experimental-xtheadmatrix">;
def mundef_f16x2  : RISCVXTHeadMatrixBuiltin<"__rvm_float16x2_t()", "experimental-xtheadmatrix">;
def mundef_f32x2  : RISCVXTHeadMatrixBuiltin<"__rvm_float32x2_t()", "experimental-xtheadmatrix">;
def mundef_f64x2  : RISCVXTHeadMatrixBuiltin<"__rvm_float64x2_t()", "experimental-xtheadmatrix">;

} // Attributes = [NoThrow]
