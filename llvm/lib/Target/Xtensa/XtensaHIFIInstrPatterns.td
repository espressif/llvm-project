//===- XtensaHIFIInstrPatterns.td - Tablegen patterns for Xtensa HIFI -*- tablegen -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains Tablegen code generation patterns for Xtensa HIFI extension
//
//===----------------------------------------------------------------------===//

def addr64n_56: ComplexPattern<iPTR, 2, "selectMemRegImm<-64,56,8>", [frameindex]>;

def addr32n_28: ComplexPattern<iPTR, 2, "selectMemRegImm<-32,28,4>", [frameindex]>;

let Predicates = [HasHIFI3] in {

def : Pat<(v2i32 (load (addr64n_56 AR:$a, imm64n_56:$imm))),
          (AE_L32X2_I AR:$a, imm64n_56:$imm)>;

def : Pat<(store v2i32:$v, (addr64n_56 AR:$a, imm64n_56:$imm)),
          (AE_S32X2_I v2i32:$v, AR:$a, imm64n_56:$imm)>;

def : Pat<(v1i64 (load (addr64n_56 AR:$a, imm64n_56:$imm))),
          (AE_L64_I AR:$a, imm64n_56:$imm)>;

def : Pat<(store v1i64:$v, (addr64n_56 AR:$a, imm64n_56:$imm)),
          (AE_S64_I v1i64:$v, AR:$a, imm64n_56:$imm)>;

def : Pat<(v4i16 (load (addr64n_56 AR:$a, imm64n_56:$imm))),
          (AE_L16X4_I AR:$a, imm64n_56:$imm)>;

def : Pat<(store v4i16:$v, (addr64n_56 AR:$a, imm64n_56:$imm)),
          (AE_S16X4_I v4i16:$v, AR:$a, imm64n_56:$imm)>;

def : Pat<(v1i32 (load (addr32n_28 AR:$a, imm32n_28:$imm))),
          (AE_L32_I AR:$a, imm32n_28:$imm)>;

def : Pat<(store v1i32:$v, (addr32n_28 AR:$a, imm32n_28:$imm)),
          (AE_S32_L_I v1i32:$v, AR:$a, imm32n_28:$imm)>;


def : Pat<(v8i8 (load (addr64n_56 AR:$a, imm64n_56:$imm))),
           (AE_LALIGN64_I AR:$a, imm64n_56:$imm)>;

def : Pat<(store AE_VALIGN:$v, (addr64n_56 AR:$a, imm64n_56:$imm)),
           (AE_SALIGN64_I AE_VALIGN:$v, AR:$a, imm64n_56:$imm)>;


def : Pat<(v2i32 (build_vector AR:$v1, AR:$v2)),
            (AE_MOVDA32X2 AR:$v2, AR:$v1)>;

def : Pat<(v2i32 (build_vector AR:$a, AR:$a)),
            (AE_MOVDA32 AR:$a)>;

/* Build const i64 vector when const fit in [-16,47]*/
def : Pat<(v1i64 (Xtensa_build_vec imm16n_47:$a)), 
            (AE_SRLI64 (AE_MOVI imm16n_47:$a), (i32 32))>;

/* Build const i64 vector with 32-bit const */
def : Pat<(v1i64 (Xtensa_build_vec AR:$a)), 
            (AE_SRLI64 (AE_MOVDA32 AR:$a), (i32 32))>;


def : Pat<(v1i32 (build_vector AR:$a)), 
            (AE_MOVDA32 AR:$a)>;

def : Pat<(v4i16 (build_vector AR:$a, AR:$a, AR:$a, AR:$a)),
            (AE_MOVDA16 AR:$a)>;

def : Pat<(v4i16 (build_vector AR:$v1, AR:$v2, AR:$v1, AR:$v2)),
            (AE_MOVDA16X2 AR:$v1, AR:$v2)>;

def : Pat<(v4i16 (build_vector AR:$v1, AR:$v2, AR:$v3, AR:$v4)),
            (AE_OR 
                (AE_SLAI64 (AE_MOVDA16X2 AR:$v1, AR:$v2), 32),
                (AE_MOVDA16X2 AR:$v3, AR:$v4)
                )>;

def : Pat<(i32 (extractelt v2i32:$v1, (i32 0))),
            (AE_MOVAD32_L AE_DR:$v1)>;

def : Pat<(i32 (extractelt v2i32:$v1, (i32 1))),
            (AE_MOVAD32_H AE_DR:$v1)>;

def : Pat<(i32 (extractelt v1i32:$v1, (i32 0))),
            (AE_MOVAD32_L AE_DR:$v1)>;

def : Pat<(i32 (vector_extract v4i16:$v1, (i32 0))), 
            (AE_MOVAD16_0 AE_DR:$v1)>;

def : Pat<(i32 (vector_extract v4i16:$v1, (i32 1))), 
            (AE_MOVAD16_1 AE_DR:$v1)>;

def : Pat<(i32 (vector_extract v4i16:$v1, (i32 2))), 
            (AE_MOVAD16_2 AE_DR:$v1)>;

def : Pat<(i32 (vector_extract v4i16:$v1, (i32 3))), 
            (AE_MOVAD16_3 AE_DR:$v1)>;

def : Pat<(v1i32 (extract_subvector v2i32:$v1, (i32 0))), 
             (AE_MOVDA32 (AE_MOVAD32_L AE_DR:$v1))>;
}

class CAST_PAT<ValueType dst_vt, ValueType src_vt> 
  : Pat<(dst_vt (bitconvert src_vt:$v)),
          (COPY_TO_REGCLASS AE_DR:$v, AE_DR)>;
  
def : CAST_PAT<v1i64,v4i16>;
def : CAST_PAT<v1i64,v2i32>;
def : CAST_PAT<v2i32,v4i16>;
def : CAST_PAT<v2i32,v1i64>;
def : CAST_PAT<v4i16,v2i32>;
def : CAST_PAT<v4i16,v1i64>;

def : Pat<(v1i64 (anyext v1i32:$src)),
          (AE_SRLI64 v1i32:$src, (i32 32))>;

def : Pat<(v1i64 (zext v1i32:$src)),
          (AE_SRLI64 v1i32:$src, (i32 32))>;

def : Pat<(v1i64 (sext v1i32:$src)),
          (AE_SRAI64 v1i32:$src, (i32 32))>;

/*
class BIN_PAT<SDPatternOperator node, Instruction inst,
  ValueType dst_vt, ValueType src_vt = dst_vt> 
  : Pat<(node src_vt:$f1, src_vt:$f2),
        (inst dst_vt:$f1, dst_vt:$f2)>;
*/
foreach VT = AE_DR.RegTypes in {
  def : BIN_PAT<and,AE_AND,VT>;
  def : BIN_PAT<or,AE_OR,VT>;
  def : BIN_PAT<xor,AE_XOR,VT>;
}

def : BIN_PAT<add,AE_ADD64,v1i64>;
def : BIN_PAT<add,AE_ADD32,v1i32>;
def : BIN_PAT<add,AE_ADD32,v2i32>;
def : BIN_PAT<add,AE_ADD16,v4i16>;

def : BIN_PAT<sub,AE_SUB64,v1i64>;
def : BIN_PAT<sub,AE_SUB32,v1i32>;
def : BIN_PAT<sub,AE_SUB32,v2i32>;
def : BIN_PAT<sub,AE_SUB16,v4i16>;

def : BIN_PAT<mul,AE_MULP32X2,v2i32>;
def : BIN_PAT<mul,AE_MULP32X2,v1i32>;
def : BIN_PAT<mul,AE_MUL16X4,v4i16>;

/* SELECT and SETCC patterns */
foreach VT = AE_DR.RegTypes in {
  def : Pat<(VT (select v1i1:$cc, AE_DR:$t, AE_DR:$f)),
            (AE_MOVT64 AE_DR:$t, AE_DR:$f, v1i1:$cc)>;
}
def : Pat<(f32 (int_xtensa_xt_movt_s FPR:$t, FPR:$f, v1i1:$cc)),
            (MOVT_S FPR:$t, FPR:$f, v1i1:$cc)>,Requires<[HasSingleFloat]>;

def : Pat<(f32 (select v1i1:$cc, FPR:$t, FPR:$f)),
            (MOVT_S FPR:$t, FPR:$f, v1i1:$cc)>,Requires<[HasSingleFloat]>;

class SELECTCC_VEC_INT<CondCode cond, Instruction cmp, Instruction mov, ValueType vt> 
  :  Pat<(vt (selectcc i32:$lhs, i32:$rhs, AE_DR:$t, AE_DR:$f, cond)),
         (mov AE_DR:$t, AE_DR:$f, (cmp (AE_MOVDA32 AR:$lhs),
                                             (AE_MOVDA32 AR:$rhs)))>;
  
foreach vt = [v2i32,v1i32,v1i64,v4i16] in {
        def : SELECTCC_VEC_INT<SETEQ, AE_EQ64, AE_MOVT64, vt>;
        def : SELECTCC_VEC_INT<SETOEQ, AE_EQ64, AE_MOVT64, vt>;
        def : SELECTCC_VEC_INT<SETUEQ, AE_EQ64, AE_MOVT64, vt>;
        def : SELECTCC_VEC_INT<SETNE, AE_EQ64, AE_MOVF64, vt>;
        def : SELECTCC_VEC_INT<SETUNE, AE_EQ64, AE_MOVF64, vt>;
        def : SELECTCC_VEC_INT<SETONE, AE_EQ64, AE_MOVF64, vt>;
        def : SELECTCC_VEC_INT<SETLT, AE_LT64, AE_MOVT64, vt>;
        def : SELECTCC_VEC_INT<SETULT, AE_LT64, AE_MOVT64, vt>;
        def : SELECTCC_VEC_INT<SETOLT, AE_LT64, AE_MOVT64, vt>;
        def : SELECTCC_VEC_INT<SETLE, AE_LE64, AE_MOVT64, vt>;
        def : SELECTCC_VEC_INT<SETULE, AE_LE64, AE_MOVT64, vt>;
        def : SELECTCC_VEC_INT<SETOLE, AE_LE64, AE_MOVT64, vt>;
        def : SELECTCC_VEC_INT<SETGE, AE_LT64, AE_MOVF64, vt>;
        def : SELECTCC_VEC_INT<SETUGE, AE_LT64, AE_MOVF64, vt>;
        def : SELECTCC_VEC_INT<SETOGE, AE_LT64, AE_MOVF64, vt>;
        def : SELECTCC_VEC_INT<SETGT, AE_LE64, AE_MOVF64, vt>;
        def : SELECTCC_VEC_INT<SETUGT, AE_LE64, AE_MOVF64, vt>;
        def : SELECTCC_VEC_INT<SETOGT, AE_LE64, AE_MOVF64, vt>;
}
