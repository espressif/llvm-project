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
// Uses the Spec-API (ManagedRA) programming model exclusively. No register
// index parameters — the register allocator manages matrix registers.
//
// Contents:
//   1. Configuration builtins (mrelease, msettilem/k/n variants)
//   2. Undefined value constructors
//   3. Spec-API builtins (loads, stores, matmul, zero, EW, conversions, etc.)
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

// --- Tuple get/set: extract/insert single register from x2 pair ---
// mget: (x2_pair, index) -> single
// mset: (x2_pair, index, single_val) -> x2_pair
foreach info = [["i8",  "__rvm_int8_t",     "__rvm_int8x2_t"],
                ["i16", "__rvm_int16_t",    "__rvm_int16x2_t"],
                ["i32", "__rvm_int32_t",    "__rvm_int32x2_t"],
                ["i64", "__rvm_int64_t",    "__rvm_int64x2_t"],
                ["u8",  "__rvm_uint8_t",    "__rvm_uint8x2_t"],
                ["u16", "__rvm_uint16_t",   "__rvm_uint16x2_t"],
                ["u32", "__rvm_uint32_t",   "__rvm_uint32x2_t"],
                ["u64", "__rvm_uint64_t",   "__rvm_uint64x2_t"],
                ["f16", "__rvm_float16_t",  "__rvm_float16x2_t"],
                ["f32", "__rvm_float32_t",  "__rvm_float32x2_t"],
                ["f64", "__rvm_float64_t",  "__rvm_float64x2_t"]] in {
  def mget_spec_ # info[0] : RISCVXTHeadMatrixBuiltin<
      info[1] # "(" # info[2] # ", size_t)", "experimental-xtheadmatrix">;
  def mset_spec_ # info[0] : RISCVXTHeadMatrixBuiltin<
      info[2] # "(" # info[2] # ", size_t, " # info[1] # ")", "experimental-xtheadmatrix">;
}

//===----------------------------------------------------------------------===//
// Spec-API builtins (ManagedRA: no register index, returns matrix type)
//===----------------------------------------------------------------------===//
// These builtins emit _internal intrinsics that produce/consume real
// target("riscv.matrix") SSA values for register-allocator-managed code.
//
// Load spec-API: (void *base, size_t stride, size_t m, size_t n) -> matrix
// Store spec-API: (void *base, size_t stride, matrix val, size_t m, size_t n) -> void
// Matmul spec-API: (matrix acc, matrix a, matrix b, size_t m, size_t k, size_t n) -> matrix

// --- Load A-tile (mlae): (base, stride, m, k) -> matrix ---
def mld_spec_i8  : RISCVXTHeadMatrixBuiltin<"__rvm_int8_t(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mld_spec_i16 : RISCVXTHeadMatrixBuiltin<"__rvm_int16_t(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mld_spec_i32 : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mld_spec_i64 : RISCVXTHeadMatrixBuiltin<"__rvm_int64_t(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mld_spec_u8  : RISCVXTHeadMatrixBuiltin<"__rvm_uint8_t(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mld_spec_u16 : RISCVXTHeadMatrixBuiltin<"__rvm_uint16_t(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mld_spec_u32 : RISCVXTHeadMatrixBuiltin<"__rvm_uint32_t(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mld_spec_u64 : RISCVXTHeadMatrixBuiltin<"__rvm_uint64_t(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mld_spec_f16 : RISCVXTHeadMatrixBuiltin<"__rvm_float16_t(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mld_spec_f32 : RISCVXTHeadMatrixBuiltin<"__rvm_float32_t(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mld_spec_f64 : RISCVXTHeadMatrixBuiltin<"__rvm_float64_t(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;

// --- Load B-tile (mlbe): (base, stride, k, n) -> matrix ---
def mld_b_spec_i8  : RISCVXTHeadMatrixBuiltin<"__rvm_int8_t(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mld_b_spec_i16 : RISCVXTHeadMatrixBuiltin<"__rvm_int16_t(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mld_b_spec_i32 : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mld_b_spec_i64 : RISCVXTHeadMatrixBuiltin<"__rvm_int64_t(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mld_b_spec_u8  : RISCVXTHeadMatrixBuiltin<"__rvm_uint8_t(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mld_b_spec_u16 : RISCVXTHeadMatrixBuiltin<"__rvm_uint16_t(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mld_b_spec_u32 : RISCVXTHeadMatrixBuiltin<"__rvm_uint32_t(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mld_b_spec_u64 : RISCVXTHeadMatrixBuiltin<"__rvm_uint64_t(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mld_b_spec_f16 : RISCVXTHeadMatrixBuiltin<"__rvm_float16_t(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mld_b_spec_f32 : RISCVXTHeadMatrixBuiltin<"__rvm_float32_t(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mld_b_spec_f64 : RISCVXTHeadMatrixBuiltin<"__rvm_float64_t(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;

// --- Load accumulator (mlce): (base, stride, m, n) -> matrix ---
def mld_acc_spec_i8  : RISCVXTHeadMatrixBuiltin<"__rvm_int8_t(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mld_acc_spec_i16 : RISCVXTHeadMatrixBuiltin<"__rvm_int16_t(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mld_acc_spec_i32 : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mld_acc_spec_i64 : RISCVXTHeadMatrixBuiltin<"__rvm_int64_t(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mld_acc_spec_u8  : RISCVXTHeadMatrixBuiltin<"__rvm_uint8_t(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mld_acc_spec_u16 : RISCVXTHeadMatrixBuiltin<"__rvm_uint16_t(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mld_acc_spec_u32 : RISCVXTHeadMatrixBuiltin<"__rvm_uint32_t(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mld_acc_spec_u64 : RISCVXTHeadMatrixBuiltin<"__rvm_uint64_t(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mld_acc_spec_f16 : RISCVXTHeadMatrixBuiltin<"__rvm_float16_t(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mld_acc_spec_f32 : RISCVXTHeadMatrixBuiltin<"__rvm_float32_t(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mld_acc_spec_f64 : RISCVXTHeadMatrixBuiltin<"__rvm_float64_t(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;

// --- Store (msce): (base, stride, val, m, n) -> void ---
def mst_spec_i8  : RISCVXTHeadMatrixBuiltin<"void(void *, size_t, __rvm_int8_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mst_spec_i16 : RISCVXTHeadMatrixBuiltin<"void(void *, size_t, __rvm_int16_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mst_spec_i32 : RISCVXTHeadMatrixBuiltin<"void(void *, size_t, __rvm_int32_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mst_spec_i64 : RISCVXTHeadMatrixBuiltin<"void(void *, size_t, __rvm_int64_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mst_spec_u8  : RISCVXTHeadMatrixBuiltin<"void(void *, size_t, __rvm_uint8_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mst_spec_u16 : RISCVXTHeadMatrixBuiltin<"void(void *, size_t, __rvm_uint16_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mst_spec_u32 : RISCVXTHeadMatrixBuiltin<"void(void *, size_t, __rvm_uint32_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mst_spec_u64 : RISCVXTHeadMatrixBuiltin<"void(void *, size_t, __rvm_uint64_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mst_spec_f16 : RISCVXTHeadMatrixBuiltin<"void(void *, size_t, __rvm_float16_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mst_spec_f32 : RISCVXTHeadMatrixBuiltin<"void(void *, size_t, __rvm_float32_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mst_spec_f64 : RISCVXTHeadMatrixBuiltin<"void(void *, size_t, __rvm_float64_t, size_t, size_t)", "experimental-xtheadmatrix">;

// --- Store A-tile (msae): (base, stride, val, m, k) -> void ---
foreach info = [["i8",  "__rvm_int8_t"],     ["i16", "__rvm_int16_t"],
                ["i32", "__rvm_int32_t"],     ["i64", "__rvm_int64_t"],
                ["u8",  "__rvm_uint8_t"],     ["u16", "__rvm_uint16_t"],
                ["u32", "__rvm_uint32_t"],     ["u64", "__rvm_uint64_t"],
                ["f16", "__rvm_float16_t"],   ["f32", "__rvm_float32_t"],
                ["f64", "__rvm_float64_t"]] in {
  def mst_a_spec_ # info[0] : RISCVXTHeadMatrixBuiltin<
      "void(void *, size_t, " # info[1] # ", size_t, size_t)", "experimental-xtheadmatrix">;
}

// --- Store B-tile (msbe): (base, stride, val, k, n) -> void ---
foreach info = [["i8",  "__rvm_int8_t"],     ["i16", "__rvm_int16_t"],
                ["i32", "__rvm_int32_t"],     ["i64", "__rvm_int64_t"],
                ["u8",  "__rvm_uint8_t"],     ["u16", "__rvm_uint16_t"],
                ["u32", "__rvm_uint32_t"],     ["u64", "__rvm_uint64_t"],
                ["f16", "__rvm_float16_t"],   ["f32", "__rvm_float32_t"],
                ["f64", "__rvm_float64_t"]] in {
  def mst_b_spec_ # info[0] : RISCVXTHeadMatrixBuiltin<
      "void(void *, size_t, " # info[1] # ", size_t, size_t)", "experimental-xtheadmatrix">;
}

// --- Transposed load A (mlate): (base, stride, m, k) -> matrix ---
foreach info = [["i8",  "__rvm_int8_t"],     ["i16", "__rvm_int16_t"],
                ["i32", "__rvm_int32_t"],     ["i64", "__rvm_int64_t"],
                ["u8",  "__rvm_uint8_t"],     ["u16", "__rvm_uint16_t"],
                ["u32", "__rvm_uint32_t"],     ["u64", "__rvm_uint64_t"],
                ["f16", "__rvm_float16_t"],   ["f32", "__rvm_float32_t"],
                ["f64", "__rvm_float64_t"]] in {
  def mld_at_spec_ # info[0] : RISCVXTHeadMatrixBuiltin<
      info[1] # "(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
}

// --- Transposed load B (mlbte): (base, stride, k, n) -> matrix ---
foreach info = [["i8",  "__rvm_int8_t"],     ["i16", "__rvm_int16_t"],
                ["i32", "__rvm_int32_t"],     ["i64", "__rvm_int64_t"],
                ["u8",  "__rvm_uint8_t"],     ["u16", "__rvm_uint16_t"],
                ["u32", "__rvm_uint32_t"],     ["u64", "__rvm_uint64_t"],
                ["f16", "__rvm_float16_t"],   ["f32", "__rvm_float32_t"],
                ["f64", "__rvm_float64_t"]] in {
  def mld_bt_spec_ # info[0] : RISCVXTHeadMatrixBuiltin<
      info[1] # "(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
}

// --- Transposed load C (mlcte): (base, stride, m, n) -> matrix ---
foreach info = [["i8",  "__rvm_int8_t"],     ["i16", "__rvm_int16_t"],
                ["i32", "__rvm_int32_t"],     ["i64", "__rvm_int64_t"],
                ["u8",  "__rvm_uint8_t"],     ["u16", "__rvm_uint16_t"],
                ["u32", "__rvm_uint32_t"],     ["u64", "__rvm_uint64_t"],
                ["f16", "__rvm_float16_t"],   ["f32", "__rvm_float32_t"],
                ["f64", "__rvm_float64_t"]] in {
  def mld_ct_spec_ # info[0] : RISCVXTHeadMatrixBuiltin<
      info[1] # "(void *, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
}

// --- Transposed store A (msate): (base, stride, val, m, k) -> void ---
foreach info = [["i8",  "__rvm_int8_t"],     ["i16", "__rvm_int16_t"],
                ["i32", "__rvm_int32_t"],     ["i64", "__rvm_int64_t"],
                ["u8",  "__rvm_uint8_t"],     ["u16", "__rvm_uint16_t"],
                ["u32", "__rvm_uint32_t"],     ["u64", "__rvm_uint64_t"],
                ["f16", "__rvm_float16_t"],   ["f32", "__rvm_float32_t"],
                ["f64", "__rvm_float64_t"]] in {
  def mst_at_spec_ # info[0] : RISCVXTHeadMatrixBuiltin<
      "void(void *, size_t, " # info[1] # ", size_t, size_t)", "experimental-xtheadmatrix">;
}

// --- Transposed store B (msbte): (base, stride, val, k, n) -> void ---
foreach info = [["i8",  "__rvm_int8_t"],     ["i16", "__rvm_int16_t"],
                ["i32", "__rvm_int32_t"],     ["i64", "__rvm_int64_t"],
                ["u8",  "__rvm_uint8_t"],     ["u16", "__rvm_uint16_t"],
                ["u32", "__rvm_uint32_t"],     ["u64", "__rvm_uint64_t"],
                ["f16", "__rvm_float16_t"],   ["f32", "__rvm_float32_t"],
                ["f64", "__rvm_float64_t"]] in {
  def mst_bt_spec_ # info[0] : RISCVXTHeadMatrixBuiltin<
      "void(void *, size_t, " # info[1] # ", size_t, size_t)", "experimental-xtheadmatrix">;
}

// --- Transposed store C (mscte): (base, stride, val, m, n) -> void ---
foreach info = [["i8",  "__rvm_int8_t"],     ["i16", "__rvm_int16_t"],
                ["i32", "__rvm_int32_t"],     ["i64", "__rvm_int64_t"],
                ["u8",  "__rvm_uint8_t"],     ["u16", "__rvm_uint16_t"],
                ["u32", "__rvm_uint32_t"],     ["u64", "__rvm_uint64_t"],
                ["f16", "__rvm_float16_t"],   ["f32", "__rvm_float32_t"],
                ["f64", "__rvm_float64_t"]] in {
  def mst_ct_spec_ # info[0] : RISCVXTHeadMatrixBuiltin<
      "void(void *, size_t, " # info[1] # ", size_t, size_t)", "experimental-xtheadmatrix">;
}

// --- Whole-register load (mlme): (base, stride) -> matrix ---
foreach info = [["i8",  "__rvm_int8_t"],     ["i16", "__rvm_int16_t"],
                ["i32", "__rvm_int32_t"],     ["i64", "__rvm_int64_t"],
                ["u8",  "__rvm_uint8_t"],     ["u16", "__rvm_uint16_t"],
                ["u32", "__rvm_uint32_t"],     ["u64", "__rvm_uint64_t"],
                ["f16", "__rvm_float16_t"],   ["f32", "__rvm_float32_t"],
                ["f64", "__rvm_float64_t"]] in {
  def mld_m_spec_ # info[0] : RISCVXTHeadMatrixBuiltin<
      info[1] # "(void *, size_t)", "experimental-xtheadmatrix">;
}

// --- Whole-register store (msme): (base, stride, val) -> void ---
foreach info = [["i8",  "__rvm_int8_t"],     ["i16", "__rvm_int16_t"],
                ["i32", "__rvm_int32_t"],     ["i64", "__rvm_int64_t"],
                ["u8",  "__rvm_uint8_t"],     ["u16", "__rvm_uint16_t"],
                ["u32", "__rvm_uint32_t"],     ["u64", "__rvm_uint64_t"],
                ["f16", "__rvm_float16_t"],   ["f32", "__rvm_float32_t"],
                ["f64", "__rvm_float64_t"]] in {
  def mst_m_spec_ # info[0] : RISCVXTHeadMatrixBuiltin<
      "void(void *, size_t, " # info[1] # ")", "experimental-xtheadmatrix">;
}

// --- INT matmul spec-API: (acc, a, b, m, k, n) -> acc ---
// INT8 -> INT32
def mmaqa_spec_ss_w_b : RISCVXTHeadMatrixBuiltin<
    "__rvm_int32_t(__rvm_int32_t, __rvm_int8_t, __rvm_int8_t, size_t, size_t, size_t)",
    "experimental-xtheadmatrix">;
def mmaqa_spec_uu_w_b : RISCVXTHeadMatrixBuiltin<
    "__rvm_uint32_t(__rvm_uint32_t, __rvm_uint8_t, __rvm_uint8_t, size_t, size_t, size_t)",
    "experimental-xtheadmatrix">;
def mmaqa_spec_us_w_b : RISCVXTHeadMatrixBuiltin<
    "__rvm_int32_t(__rvm_int32_t, __rvm_uint8_t, __rvm_int8_t, size_t, size_t, size_t)",
    "experimental-xtheadmatrix">;
def mmaqa_spec_su_w_b : RISCVXTHeadMatrixBuiltin<
    "__rvm_int32_t(__rvm_int32_t, __rvm_int8_t, __rvm_uint8_t, size_t, size_t, size_t)",
    "experimental-xtheadmatrix">;
// INT16 -> INT64
def mmaqa_spec_ss_d_h : RISCVXTHeadMatrixBuiltin<
    "__rvm_int64_t(__rvm_int64_t, __rvm_int16_t, __rvm_int16_t, size_t, size_t, size_t)",
    "experimental-xtheadmatrix">;
def mmaqa_spec_uu_d_h : RISCVXTHeadMatrixBuiltin<
    "__rvm_uint64_t(__rvm_uint64_t, __rvm_uint16_t, __rvm_uint16_t, size_t, size_t, size_t)",
    "experimental-xtheadmatrix">;
def mmaqa_spec_us_d_h : RISCVXTHeadMatrixBuiltin<
    "__rvm_int64_t(__rvm_int64_t, __rvm_uint16_t, __rvm_int16_t, size_t, size_t, size_t)",
    "experimental-xtheadmatrix">;
def mmaqa_spec_su_d_h : RISCVXTHeadMatrixBuiltin<
    "__rvm_int64_t(__rvm_int64_t, __rvm_int16_t, __rvm_uint16_t, size_t, size_t, size_t)",
    "experimental-xtheadmatrix">;
// Partial INT8 -> INT32
def pmmaqa_spec_ss_w_b : RISCVXTHeadMatrixBuiltin<
    "__rvm_int32_t(__rvm_int32_t, __rvm_int8_t, __rvm_int8_t, size_t, size_t, size_t)",
    "experimental-xtheadmatrix">;
def pmmaqa_spec_uu_w_b : RISCVXTHeadMatrixBuiltin<
    "__rvm_uint32_t(__rvm_uint32_t, __rvm_uint8_t, __rvm_uint8_t, size_t, size_t, size_t)",
    "experimental-xtheadmatrix">;
def pmmaqa_spec_us_w_b : RISCVXTHeadMatrixBuiltin<
    "__rvm_int32_t(__rvm_int32_t, __rvm_uint8_t, __rvm_int8_t, size_t, size_t, size_t)",
    "experimental-xtheadmatrix">;
def pmmaqa_spec_su_w_b : RISCVXTHeadMatrixBuiltin<
    "__rvm_int32_t(__rvm_int32_t, __rvm_int8_t, __rvm_uint8_t, size_t, size_t, size_t)",
    "experimental-xtheadmatrix">;
// Bypass INT
def mmaqa_spec_bp_ss : RISCVXTHeadMatrixBuiltin<
    "__rvm_int32_t(__rvm_int32_t, __rvm_int8_t, __rvm_int8_t, size_t, size_t, size_t)",
    "experimental-xtheadmatrix">;
def mmaqa_spec_bp_uu : RISCVXTHeadMatrixBuiltin<
    "__rvm_uint32_t(__rvm_uint32_t, __rvm_uint8_t, __rvm_uint8_t, size_t, size_t, size_t)",
    "experimental-xtheadmatrix">;

// --- FP matmul spec-API: (acc, a, b, m, k, n) -> acc ---
// Native-precision
def mfmaqa_spec_h : RISCVXTHeadMatrixBuiltin<
    "__rvm_float16_t(__rvm_float16_t, __rvm_float16_t, __rvm_float16_t, size_t, size_t, size_t)",
    "experimental-xtheadmatrix">;
def mfmaqa_spec_s : RISCVXTHeadMatrixBuiltin<
    "__rvm_float32_t(__rvm_float32_t, __rvm_float32_t, __rvm_float32_t, size_t, size_t, size_t)",
    "experimental-xtheadmatrix">;
def mfmaqa_spec_d : RISCVXTHeadMatrixBuiltin<
    "__rvm_float64_t(__rvm_float64_t, __rvm_float64_t, __rvm_float64_t, size_t, size_t, size_t)",
    "experimental-xtheadmatrix">;
// Widening FP16 -> FP32
def mfmaqa_spec_s_h : RISCVXTHeadMatrixBuiltin<
    "__rvm_float32_t(__rvm_float32_t, __rvm_float16_t, __rvm_float16_t, size_t, size_t, size_t)",
    "experimental-xtheadmatrix">;
// Widening FP32 -> FP64
def mfmaqa_spec_d_s : RISCVXTHeadMatrixBuiltin<
    "__rvm_float64_t(__rvm_float64_t, __rvm_float32_t, __rvm_float32_t, size_t, size_t, size_t)",
    "experimental-xtheadmatrix">;
// Widening FP8/BF16/TF32 -> various (opaque source types use __rvm_int32_t for A/B tiles)
def mfmaqa_spec_h_e4    : RISCVXTHeadMatrixBuiltin<"__rvm_float16_t(__rvm_float16_t, __rvm_int32_t, __rvm_int32_t, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mfmaqa_spec_h_e5    : RISCVXTHeadMatrixBuiltin<"__rvm_float16_t(__rvm_float16_t, __rvm_int32_t, __rvm_int32_t, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mfmaqa_spec_bf16_e4 : RISCVXTHeadMatrixBuiltin<"__rvm_float16_t(__rvm_float16_t, __rvm_int32_t, __rvm_int32_t, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mfmaqa_spec_bf16_e5 : RISCVXTHeadMatrixBuiltin<"__rvm_float16_t(__rvm_float16_t, __rvm_int32_t, __rvm_int32_t, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mfmaqa_spec_s_bf16  : RISCVXTHeadMatrixBuiltin<"__rvm_float32_t(__rvm_float32_t, __rvm_int32_t, __rvm_int32_t, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mfmaqa_spec_s_e4    : RISCVXTHeadMatrixBuiltin<"__rvm_float32_t(__rvm_float32_t, __rvm_int32_t, __rvm_int32_t, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mfmaqa_spec_s_e5    : RISCVXTHeadMatrixBuiltin<"__rvm_float32_t(__rvm_float32_t, __rvm_int32_t, __rvm_int32_t, size_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mfmaqa_spec_s_tf32  : RISCVXTHeadMatrixBuiltin<"__rvm_float32_t(__rvm_float32_t, __rvm_int32_t, __rvm_int32_t, size_t, size_t, size_t)", "experimental-xtheadmatrix">;

// --- Zero spec-API: (m, n) -> matrix ---
def mzero_spec_i8  : RISCVXTHeadMatrixBuiltin<"__rvm_int8_t(size_t, size_t)", "experimental-xtheadmatrix">;
def mzero_spec_i16 : RISCVXTHeadMatrixBuiltin<"__rvm_int16_t(size_t, size_t)", "experimental-xtheadmatrix">;
def mzero_spec_i32 : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(size_t, size_t)", "experimental-xtheadmatrix">;
def mzero_spec_i64 : RISCVXTHeadMatrixBuiltin<"__rvm_int64_t(size_t, size_t)", "experimental-xtheadmatrix">;
def mzero_spec_u8  : RISCVXTHeadMatrixBuiltin<"__rvm_uint8_t(size_t, size_t)", "experimental-xtheadmatrix">;
def mzero_spec_u16 : RISCVXTHeadMatrixBuiltin<"__rvm_uint16_t(size_t, size_t)", "experimental-xtheadmatrix">;
def mzero_spec_u32 : RISCVXTHeadMatrixBuiltin<"__rvm_uint32_t(size_t, size_t)", "experimental-xtheadmatrix">;
def mzero_spec_u64 : RISCVXTHeadMatrixBuiltin<"__rvm_uint64_t(size_t, size_t)", "experimental-xtheadmatrix">;
def mzero_spec_f16 : RISCVXTHeadMatrixBuiltin<"__rvm_float16_t(size_t, size_t)", "experimental-xtheadmatrix">;
def mzero_spec_f32 : RISCVXTHeadMatrixBuiltin<"__rvm_float32_t(size_t, size_t)", "experimental-xtheadmatrix">;
def mzero_spec_f64 : RISCVXTHeadMatrixBuiltin<"__rvm_float64_t(size_t, size_t)", "experimental-xtheadmatrix">;

//===----------------------------------------------------------------------===//
// Move / Copy spec-API
//===----------------------------------------------------------------------===//

// mmov_mm: (src) -> dst
def mmov_mm_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t)", "experimental-xtheadmatrix">;

// mmov_x_m: (matrix, index) -> gpr value
def mmovb_x_m_spec : RISCVXTHeadMatrixBuiltin<"size_t(__rvm_int32_t, size_t)", "experimental-xtheadmatrix">;
def mmovh_x_m_spec : RISCVXTHeadMatrixBuiltin<"size_t(__rvm_int32_t, size_t)", "experimental-xtheadmatrix">;
def mmovw_x_m_spec : RISCVXTHeadMatrixBuiltin<"size_t(__rvm_int32_t, size_t)", "experimental-xtheadmatrix">;
def mmovd_x_m_spec : RISCVXTHeadMatrixBuiltin<"size_t(__rvm_int32_t, size_t)", "experimental-xtheadmatrix">;

// mmov_m_x: (matrix_in, data, index) -> matrix_out
def mmovb_m_x_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mmovh_m_x_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mmovw_m_x_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t, size_t, size_t)", "experimental-xtheadmatrix">;
def mmovd_m_x_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t, size_t, size_t)", "experimental-xtheadmatrix">;

// mdup_m_x: (matrix_in, data) -> matrix_out
def mdupb_m_x_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t, size_t)", "experimental-xtheadmatrix">;
def mduph_m_x_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t, size_t)", "experimental-xtheadmatrix">;
def mdupw_m_x_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t, size_t)", "experimental-xtheadmatrix">;
def mdupd_m_x_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t, size_t)", "experimental-xtheadmatrix">;

//===----------------------------------------------------------------------===//
// Pack spec-API: (ms2, ms1) -> md
//===----------------------------------------------------------------------===//
def mpack_spec   : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t, __rvm_int32_t)", "experimental-xtheadmatrix">;
def mpackhl_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t, __rvm_int32_t)", "experimental-xtheadmatrix">;
def mpackhh_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t, __rvm_int32_t)", "experimental-xtheadmatrix">;

//===----------------------------------------------------------------------===//
// Slide spec-API: (src, imm) -> dst
//===----------------------------------------------------------------------===//
// Row slide
def mrslidedown_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t, unsigned int)", "experimental-xtheadmatrix">;
def mrslideup_spec   : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t, unsigned int)", "experimental-xtheadmatrix">;

// Column slide (per element size)
foreach sz = ["b", "h", "w", "d"] in {
  def mcslidedown_ # sz # _spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t, unsigned int)", "experimental-xtheadmatrix">;
  def mcslideup_ # sz # _spec   : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t, unsigned int)", "experimental-xtheadmatrix">;
}

//===----------------------------------------------------------------------===//
// Broadcast spec-API: (src, imm) -> dst
//===----------------------------------------------------------------------===//
// Row broadcast
def mrbca_mv_i_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t, unsigned int)", "experimental-xtheadmatrix">;

// Column broadcast (per element size)
def mcbcab_mv_i_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t, unsigned int)", "experimental-xtheadmatrix">;
def mcbcah_mv_i_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t, unsigned int)", "experimental-xtheadmatrix">;
def mcbcaw_mv_i_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t, unsigned int)", "experimental-xtheadmatrix">;
def mcbcad_mv_i_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t, unsigned int)", "experimental-xtheadmatrix">;

//===----------------------------------------------------------------------===//
// FP format conversion spec-API: (src) -> dst
//===----------------------------------------------------------------------===//

// FP8 <-> FP16
def mfcvtl_h_e4_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t)", "experimental-xtheadmatrix">;
def mfcvth_h_e4_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t)", "experimental-xtheadmatrix">;
def mfcvtl_h_e5_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t)", "experimental-xtheadmatrix">;
def mfcvth_h_e5_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t)", "experimental-xtheadmatrix">;
def mfcvtl_e4_h_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t)", "experimental-xtheadmatrix">;
def mfcvth_e4_h_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t)", "experimental-xtheadmatrix">;
def mfcvtl_e5_h_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t)", "experimental-xtheadmatrix">;
def mfcvth_e5_h_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t)", "experimental-xtheadmatrix">;

// FP16 <-> FP32
def mfcvtl_s_h_spec : RISCVXTHeadMatrixBuiltin<"__rvm_float32_t(__rvm_float16_t)", "experimental-xtheadmatrix">;
def mfcvth_s_h_spec : RISCVXTHeadMatrixBuiltin<"__rvm_float32_t(__rvm_float16_t)", "experimental-xtheadmatrix">;
def mfcvtl_h_s_spec : RISCVXTHeadMatrixBuiltin<"__rvm_float16_t(__rvm_float32_t)", "experimental-xtheadmatrix">;
def mfcvth_h_s_spec : RISCVXTHeadMatrixBuiltin<"__rvm_float16_t(__rvm_float32_t)", "experimental-xtheadmatrix">;

// BF16 <-> FP32
def mfcvtl_s_bf16_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t)", "experimental-xtheadmatrix">;
def mfcvth_s_bf16_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t)", "experimental-xtheadmatrix">;
def mfcvtl_bf16_s_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t)", "experimental-xtheadmatrix">;
def mfcvth_bf16_s_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t)", "experimental-xtheadmatrix">;

// FP32 <-> FP8
def mfcvtl_e4_s_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t)", "experimental-xtheadmatrix">;
def mfcvth_e4_s_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t)", "experimental-xtheadmatrix">;
def mfcvtl_e5_s_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t)", "experimental-xtheadmatrix">;
def mfcvth_e5_s_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t)", "experimental-xtheadmatrix">;

// FP32 <-> FP64
def mfcvtl_d_s_spec : RISCVXTHeadMatrixBuiltin<"__rvm_float64_t(__rvm_float32_t)", "experimental-xtheadmatrix">;
def mfcvth_d_s_spec : RISCVXTHeadMatrixBuiltin<"__rvm_float64_t(__rvm_float32_t)", "experimental-xtheadmatrix">;
def mfcvtl_s_d_spec : RISCVXTHeadMatrixBuiltin<"__rvm_float32_t(__rvm_float64_t)", "experimental-xtheadmatrix">;
def mfcvth_s_d_spec : RISCVXTHeadMatrixBuiltin<"__rvm_float32_t(__rvm_float64_t)", "experimental-xtheadmatrix">;

// TF32 <-> FP32
def mfcvt_s_tf32_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t)", "experimental-xtheadmatrix">;
def mfcvt_tf32_s_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t)", "experimental-xtheadmatrix">;

//===----------------------------------------------------------------------===//
// Float-int conversion spec-API: (src) -> dst
//===----------------------------------------------------------------------===//

// UINT8 <-> FP16
def mufcvtl_h_b_spec : RISCVXTHeadMatrixBuiltin<"__rvm_float16_t(__rvm_uint8_t)", "experimental-xtheadmatrix">;
def mufcvth_h_b_spec : RISCVXTHeadMatrixBuiltin<"__rvm_float16_t(__rvm_uint8_t)", "experimental-xtheadmatrix">;
def mfucvtl_b_h_spec : RISCVXTHeadMatrixBuiltin<"__rvm_uint8_t(__rvm_float16_t)", "experimental-xtheadmatrix">;
def mfucvth_b_h_spec : RISCVXTHeadMatrixBuiltin<"__rvm_uint8_t(__rvm_float16_t)", "experimental-xtheadmatrix">;

// SINT8 <-> FP16
def msfcvtl_h_b_spec : RISCVXTHeadMatrixBuiltin<"__rvm_float16_t(__rvm_int8_t)", "experimental-xtheadmatrix">;
def msfcvth_h_b_spec : RISCVXTHeadMatrixBuiltin<"__rvm_float16_t(__rvm_int8_t)", "experimental-xtheadmatrix">;
def mfscvtl_b_h_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int8_t(__rvm_float16_t)", "experimental-xtheadmatrix">;
def mfscvth_b_h_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int8_t(__rvm_float16_t)", "experimental-xtheadmatrix">;

// INT32 <-> FP32
def msfcvt_s_w_spec : RISCVXTHeadMatrixBuiltin<"__rvm_float32_t(__rvm_int32_t)", "experimental-xtheadmatrix">;
def mufcvt_s_w_spec : RISCVXTHeadMatrixBuiltin<"__rvm_float32_t(__rvm_uint32_t)", "experimental-xtheadmatrix">;
def mfscvt_w_s_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_float32_t)", "experimental-xtheadmatrix">;
def mfucvt_w_s_spec : RISCVXTHeadMatrixBuiltin<"__rvm_uint32_t(__rvm_float32_t)", "experimental-xtheadmatrix">;

//===----------------------------------------------------------------------===//
// Fixed-point clip (n4clip) spec-API
//===----------------------------------------------------------------------===//

// n4clip .mm: (acc, ms2, ms1) -> acc_out
def mn4clipl_w_mm_spec  : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t, __rvm_int32_t, __rvm_int32_t)", "experimental-xtheadmatrix">;
def mn4cliph_w_mm_spec  : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t, __rvm_int32_t, __rvm_int32_t)", "experimental-xtheadmatrix">;
def mn4cliplu_w_mm_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t, __rvm_int32_t, __rvm_int32_t)", "experimental-xtheadmatrix">;
def mn4cliphu_w_mm_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t, __rvm_int32_t, __rvm_int32_t)", "experimental-xtheadmatrix">;

// n4clip .mv.i: (acc, ms2, ms1, imm) -> acc_out
def mn4clipl_w_mv_i_spec  : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t, __rvm_int32_t, __rvm_int32_t, unsigned int)", "experimental-xtheadmatrix">;
def mn4cliph_w_mv_i_spec  : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t, __rvm_int32_t, __rvm_int32_t, unsigned int)", "experimental-xtheadmatrix">;
def mn4cliplu_w_mv_i_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t, __rvm_int32_t, __rvm_int32_t, unsigned int)", "experimental-xtheadmatrix">;
def mn4cliphu_w_mv_i_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t, __rvm_int32_t, __rvm_int32_t, unsigned int)", "experimental-xtheadmatrix">;

//===----------------------------------------------------------------------===//
// Packed conversion spec-API: (src) -> dst
//===----------------------------------------------------------------------===//
def mucvtl_b_p_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t)", "experimental-xtheadmatrix">;
def mscvtl_b_p_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t)", "experimental-xtheadmatrix">;
def mucvth_b_p_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t)", "experimental-xtheadmatrix">;
def mscvth_b_p_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t)", "experimental-xtheadmatrix">;

//===----------------------------------------------------------------------===//
// Integer element-wise spec-API: (acc, ms2, ms1) -> acc_out
//   .w.mv.i variant: (acc, ms2, ms1, imm) -> acc_out
//===----------------------------------------------------------------------===//
foreach op = ["madd", "msub", "mmul", "mmulh",
              "mmax", "mumax", "mmin", "mumin",
              "msrl", "msll", "msra"] in {
  def op # _w_mm_spec   : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t, __rvm_int32_t, __rvm_int32_t)", "experimental-xtheadmatrix">;
  def op # _w_mv_i_spec : RISCVXTHeadMatrixBuiltin<"__rvm_int32_t(__rvm_int32_t, __rvm_int32_t, __rvm_int32_t, unsigned int)", "experimental-xtheadmatrix">;
}

//===----------------------------------------------------------------------===//
// FP element-wise spec-API: (acc, ms2, ms1) -> acc_out
//   .mv.i variant: (acc, ms2, ms1, imm) -> acc_out
//===----------------------------------------------------------------------===//
foreach op = ["mfadd", "mfsub", "mfmul", "mfmax", "mfmin"] in {
  def op # _h_mm_spec   : RISCVXTHeadMatrixBuiltin<"__rvm_float16_t(__rvm_float16_t, __rvm_float16_t, __rvm_float16_t)", "experimental-xtheadmatrix">;
  def op # _h_mv_i_spec : RISCVXTHeadMatrixBuiltin<"__rvm_float16_t(__rvm_float16_t, __rvm_float16_t, __rvm_float16_t, unsigned int)", "experimental-xtheadmatrix">;
  def op # _s_mm_spec   : RISCVXTHeadMatrixBuiltin<"__rvm_float32_t(__rvm_float32_t, __rvm_float32_t, __rvm_float32_t)", "experimental-xtheadmatrix">;
  def op # _s_mv_i_spec : RISCVXTHeadMatrixBuiltin<"__rvm_float32_t(__rvm_float32_t, __rvm_float32_t, __rvm_float32_t, unsigned int)", "experimental-xtheadmatrix">;
  def op # _d_mm_spec   : RISCVXTHeadMatrixBuiltin<"__rvm_float64_t(__rvm_float64_t, __rvm_float64_t, __rvm_float64_t)", "experimental-xtheadmatrix">;
  def op # _d_mv_i_spec : RISCVXTHeadMatrixBuiltin<"__rvm_float64_t(__rvm_float64_t, __rvm_float64_t, __rvm_float64_t, unsigned int)", "experimental-xtheadmatrix">;
}

//===----------------------------------------------------------------------===//
// Zmpanel: Panel-Aware 2x2 Matrix Tiling builtins
//===----------------------------------------------------------------------===//

// --- Configuration: (size_t) -> void ---
def mset22adra : RISCVXTHeadMatrixBuiltin<"void(size_t)", "experimental-xtheadzmpanel">;
def mset22adrb : RISCVXTHeadMatrixBuiltin<"void(size_t)", "experimental-xtheadzmpanel">;
def mset22adrd : RISCVXTHeadMatrixBuiltin<"void(size_t)", "experimental-xtheadzmpanel">;
def mset22rsba : RISCVXTHeadMatrixBuiltin<"void(size_t)", "experimental-xtheadzmpanel">;
def mset22rsbb : RISCVXTHeadMatrixBuiltin<"void(size_t)", "experimental-xtheadzmpanel">;
def mset22rsbd : RISCVXTHeadMatrixBuiltin<"void(size_t)", "experimental-xtheadzmpanel">;
def mset22m    : RISCVXTHeadMatrixBuiltin<"void(size_t)", "experimental-xtheadzmpanel">;
def mset22n    : RISCVXTHeadMatrixBuiltin<"void(size_t)", "experimental-xtheadzmpanel">;
def mset22k    : RISCVXTHeadMatrixBuiltin<"void(size_t)", "experimental-xtheadzmpanel">;
def msetrstptr : RISCVXTHeadMatrixBuiltin<"void(size_t)", "experimental-xtheadzmpanel">;
def msetaccum  : RISCVXTHeadMatrixBuiltin<"void(size_t)", "experimental-xtheadzmpanel">;
def msetoob    : RISCVXTHeadMatrixBuiltin<"void(size_t)", "experimental-xtheadzmpanel">;

// --- Load: () -> void ---
def ml22e8  : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadzmpanel">;
def ml22e16 : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadzmpanel">;

// --- Store: () -> void ---
def msc22e16 : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadzmpanel">;
def msc22e32 : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadzmpanel">;

// --- FP compute: () -> void ---
def mfmacc22_h_e5    : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadzmpanel">;
def mfmacc22_h_e4    : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadzmpanel">;
def mfmacc22_bf16_e5 : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadzmpanel">;
def mfmacc22_bf16_e4 : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadzmpanel">;
def mfmacc22_s_e5    : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadzmpanel">;
def mfmacc22_s_e4    : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadzmpanel">;
def mfmacc22_h       : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadzmpanel">;
def mfmacc22_s_h     : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadzmpanel">;
def mfmacc22_s_bf16  : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadzmpanel">;
def mfmacc22_s       : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadzmpanel">;

// --- INT compute: () -> void ---
def mmacc22_w_b   : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadzmpanel">;
def mmaccu22_w_b  : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadzmpanel">;
def mmaccus22_w_b : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadzmpanel">;
def mmaccsu22_w_b : RISCVXTHeadMatrixBuiltin<"void()", "experimental-xtheadzmpanel">;

} // Attributes = [NoThrow]
