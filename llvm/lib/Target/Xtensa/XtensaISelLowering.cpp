//===- XtensaISelLowering.cpp - Xtensa DAG Lowering Implementation --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that Xtensa uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#include "XtensaISelLowering.h"
#include "XtensaConstantPoolValue.h"
#include "XtensaMachineFunctionInfo.h"
#include "XtensaSubtarget.h"
#include "XtensaTargetMachine.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <deque>

using namespace llvm;

#define DEBUG_TYPE "xtensa-lower"

static const MCPhysReg XtensaArgRegs[6] = {Xtensa::A2, Xtensa::A3, Xtensa::A4,
                                           Xtensa::A5, Xtensa::A6, Xtensa::A7};

static const MCPhysReg VecRegs[] = {Xtensa::AED0, Xtensa::AED1, Xtensa::AED2,
                                    Xtensa::AED3};

static const MVT VectorIntTypes[] = {
    MVT::v2i32,
    MVT::v1i32,
    MVT::v4i16,
    MVT::v1i64,
};

template <typename VT> static bool isVecVT(VT ValVT) {
  for (const auto &V : VectorIntTypes) {
    auto VV = VT(V);
    if (VV == ValVT)
      return true;
  }
  return false;
}

// Return true if we must use long (in fact, indirect) function call.
// It's simplified version, production implimentation must
// resolve a functions in ROM (usually glibc functions)
static bool isLongCall(const char *str) {
  // Currently always use long calls
  return true;
}

// The calling conventions in XtensaCallingConv.td are described in terms of the
// callee's register window. This function translates registers to the
// corresponding caller window %o register.
static unsigned toCallerWindow(unsigned Reg) {
  if (Reg >= Xtensa::A2 && Reg <= Xtensa::A7)
    return Reg - Xtensa::A2 + Xtensa::A10;
  return Reg;
}

XtensaTargetLowering::XtensaTargetLowering(const TargetMachine &TM,
                                           const XtensaSubtarget &STI)
    : TargetLowering(TM), Subtarget(STI) {
  MVT PtrVT = MVT::i32;
  // Set up the register classes.
  addRegisterClass(MVT::i32, &Xtensa::ARRegClass);

  if (Subtarget.hasSingleFloat()) {
    addRegisterClass(MVT::f32, &Xtensa::FPRRegClass);
  }

  if (Subtarget.hasBoolean()) {
    addRegisterClass(MVT::v1i1, &Xtensa::BRRegClass);
    addRegisterClass(MVT::v2i1, &Xtensa::BR2RegClass);
    addRegisterClass(MVT::v4i1, &Xtensa::BR4RegClass);
    setOperationAction(ISD::Constant, MVT::v2i1, Expand);
    setOperationAction(ISD::Constant, MVT::v1i1, Expand);
    setTargetDAGCombine(ISD::STORE);
    setTargetDAGCombine(ISD::BITCAST);
    setTargetDAGCombine(ISD::EXTRACT_SUBVECTOR);

    setOperationAction(ISD::STORE, MVT::v1i1, Legal);
    setOperationAction(ISD::STORE, MVT::v2i1, Legal);
    setOperationAction(ISD::STORE, MVT::v4i1, Legal);
    setOperationAction(ISD::LOAD, MVT::v1i1, Legal);
    setOperationAction(ISD::LOAD, MVT::v2i1, Legal);
    setOperationAction(ISD::LOAD, MVT::v4i1, Legal);
  }

  if (Subtarget.hasHIFI3()) {
    for (MVT VT : VectorIntTypes) {
      addRegisterClass(VT, &Xtensa::AE_DRRegClass);
      setOperationAction(ISD::VECTOR_SHUFFLE, VT, Expand);
      // handle bicast v8i8 to VEC_VT
      setOperationAction(ISD::BITCAST, VT, Custom);
    }
    addRegisterClass(MVT::v8i8, &Xtensa::AE_VALIGNRegClass);
    // handle bicast VEC_VT to v8i8
    setOperationAction(ISD::BITCAST, MVT::v8i8, Expand);

    setOperationAction(ISD::SIGN_EXTEND, MVT::v1i32, Expand);
    setOperationAction(ISD::ZERO_EXTEND, MVT::v1i32, Expand);
    setOperationAction(ISD::ANY_EXTEND, MVT::v1i32, Expand);
    setOperationAction(ISD::BUILD_VECTOR, MVT::v1i64, Legal);

    setTargetDAGCombine(ISD::BUILD_VECTOR);
    setOperationAction(ISD::MUL, MVT::v1i64, Expand);
  }

  // Set up special registers.
  setStackPointerRegisterToSaveRestore(Xtensa::SP);

  setSchedulingPreference(Sched::RegPressure);

  setBooleanVectorContents(ZeroOrOneBooleanContent);

  setMinFunctionAlignment(Align(4));

  setOperationAction(ISD::Constant, MVT::i32, Custom);
  setOperationAction(ISD::Constant, MVT::i64, Expand);
  setOperationAction(ISD::ConstantFP, MVT::f32, Custom);
  setOperationAction(ISD::ConstantFP, MVT::f64, Expand);

  setBooleanContents(ZeroOrOneBooleanContent);

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i8, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i16, Expand);

  setOperationAction(ISD::BITCAST, MVT::i32, Expand);
  setOperationAction(ISD::BITCAST, MVT::f32, Expand);
  setOperationAction(ISD::UINT_TO_FP, MVT::i32, Expand);
  setOperationAction(ISD::SINT_TO_FP, MVT::i32, Expand);
  setOperationAction(ISD::FP_TO_UINT, MVT::i32, Expand);
  setOperationAction(ISD::FP_TO_SINT, MVT::i32, Expand);

  // No sign extend instructions for i1
  for (MVT VT : MVT::integer_valuetypes()) {
    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::i1, Promote);
    setLoadExtAction(ISD::ZEXTLOAD, VT, MVT::i1, Promote);
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::i1, Promote);
  }

  setOperationAction(ISD::ConstantPool, PtrVT, Custom);
  setOperationAction(ISD::GlobalAddress, PtrVT, Custom);
  setOperationAction(ISD::GlobalTLSAddress, PtrVT, Custom);
  setOperationAction(ISD::BlockAddress, PtrVT, Custom);
  setOperationAction(ISD::JumpTable, PtrVT, Custom);

  // Expand jump table branches as address arithmetic followed by an
  // indirect jump.
  setOperationAction(ISD::BR_JT, MVT::Other, Custom);
  // Used by legalize types to correctly generate the setcc result.
  // AddPromotedToType(ISD::SETCC, MVT::i1, MVT::i32);
  if (!Subtarget.hasBoolean())
    setOperationPromotedToType(ISD::SETCC, MVT::i1, MVT::i32);
  setOperationPromotedToType(ISD::BR_CC, MVT::i1, MVT::i32);

  setOperationAction(ISD::BR_CC, MVT::i32, Legal);
  setOperationAction(ISD::BR_CC, MVT::i64, Expand);
  if (Subtarget.hasSingleFloat())
    setOperationAction(ISD::BR_CC, MVT::f32, Custom);
  else
    setOperationAction(ISD::BR_CC, MVT::f32, Expand);

  setOperationAction(ISD::SELECT, MVT::i32, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::i32, Custom);
  setOperationAction(ISD::SETCC, MVT::i32, Expand);
  
  setOperationAction(ISD::SELECT, MVT::f32, Expand);
  if (Subtarget.hasSingleFloat()) {
    setOperationAction(ISD::SELECT_CC, MVT::f32, Custom);
    setOperationAction(ISD::SETCC, MVT::f32, Custom);
  } else {
    setOperationAction(ISD::SELECT_CC, MVT::f32, Expand);
    setOperationAction(ISD::SETCC, MVT::f32, Expand);
  }

  setCondCodeAction(ISD::SETGT, MVT::i32, Expand);
  setCondCodeAction(ISD::SETLE, MVT::i32, Expand);
  setCondCodeAction(ISD::SETUGT, MVT::i32, Expand);
  setCondCodeAction(ISD::SETULE, MVT::i32, Expand);

  if (Subtarget.hasMul32())
    setOperationAction(ISD::MUL, MVT::i32, Legal);
  else
    setOperationAction(ISD::MUL, MVT::i32, Expand);

  if (Subtarget.hasMul32High()) {
    setOperationAction(ISD::MULHU, MVT::i32, Legal);
    setOperationAction(ISD::MULHS, MVT::i32, Legal);
  } else {
    setOperationAction(ISD::MULHU, MVT::i32, Expand);
    setOperationAction(ISD::MULHS, MVT::i32, Expand);
  }
  setOperationAction(ISD::SMUL_LOHI, MVT::i32, Expand);
  setOperationAction(ISD::UMUL_LOHI, MVT::i32, Expand);

  if (Subtarget.hasDiv32()) {
    setOperationAction(ISD::SDIV, MVT::i32, Legal);
    setOperationAction(ISD::UDIV, MVT::i32, Legal);
    setOperationAction(ISD::SREM, MVT::i32, Legal);
    setOperationAction(ISD::UREM, MVT::i32, Legal);
  } else {
    setOperationAction(ISD::SDIV, MVT::i32, Expand);
    setOperationAction(ISD::UDIV, MVT::i32, Expand);
    setOperationAction(ISD::SREM, MVT::i32, Expand);
    setOperationAction(ISD::UREM, MVT::i32, Expand);
  }
  setOperationAction(ISD::SDIVREM, MVT::i32, Expand);
  setOperationAction(ISD::UDIVREM, MVT::i32, Expand);

  setOperationAction(ISD::SDIV, MVT::i64, Expand);
  setOperationAction(ISD::UDIV, MVT::i64, Expand);
  setOperationAction(ISD::SREM, MVT::i64, Expand);
  setOperationAction(ISD::UREM, MVT::i64, Expand);

  // Xtensa doesn't support  [ADD,SUB][E,C]
  setOperationAction(ISD::ADDC, MVT::i32, Expand);
  setOperationAction(ISD::ADDE, MVT::i32, Expand);
  setOperationAction(ISD::SUBC, MVT::i32, Expand);
  setOperationAction(ISD::SUBE, MVT::i32, Expand);

  setOperationAction(ISD::ABS, MVT::i32, Legal);

  setOperationAction(ISD::ADD, MVT::i64, Expand);
  setOperationAction(ISD::SUB, MVT::i64, Expand);

  // Xtensa doesn't support s[hl,rl,ra]_parts
  setOperationAction(ISD::SHL_PARTS, MVT::i32, Custom);
  setOperationAction(ISD::SRA_PARTS, MVT::i32, Custom);
  setOperationAction(ISD::SRL_PARTS, MVT::i32, Custom);

  // Funnel shifts
  setOperationAction(ISD::FSHR, MVT::i32, Custom);
  setOperationAction(ISD::FSHL, MVT::i32, Custom);

  // Bit Manipulation
  setOperationAction(ISD::BSWAP, MVT::i32, Expand);
  setOperationAction(ISD::ROTL, MVT::i32, Expand);
  setOperationAction(ISD::ROTR, MVT::i32, Expand);
  setOperationAction(ISD::CTPOP, MVT::i32, Custom);
  setOperationAction({ISD::CTTZ, ISD::CTTZ_ZERO_UNDEF}, MVT::i32, Expand);
  if (Subtarget.hasNSA())
    setOperationAction(ISD::CTLZ, MVT::i32, Legal);
  else
    setOperationAction({ISD::CTLZ, ISD::CTLZ_ZERO_UNDEF}, MVT::i32, Expand);


  setOperationAction({ISD::SMIN, ISD::SMAX, ISD::UMIN, ISD::UMAX},
                     MVT::i32, Subtarget.hasMINMAX() ? Legal : Expand);

  setOperationAction(ISD::SMUL_LOHI, MVT::i32, Expand);
  setOperationAction(ISD::SMUL_LOHI, MVT::i64, Expand);
  setOperationAction(ISD::UMUL_LOHI, MVT::i32, Expand);
  setOperationAction(ISD::UMUL_LOHI, MVT::i64, Expand);

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i8, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i16, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i32, Expand);

  // Handle floating-point types.
  for (unsigned I = MVT::FIRST_FP_VALUETYPE; I <= MVT::LAST_FP_VALUETYPE; ++I) {
    MVT VT = MVT::SimpleValueType(I);
    if (isTypeLegal(VT)) {
      if (VT.getSizeInBits() == 32 && Subtarget.hasSingleFloat()) {
        setOperationAction(ISD::FABS, VT, Legal);
        setOperationAction(ISD::FADD, VT, Legal);
        setOperationAction(ISD::FMA, VT, Legal);
        setOperationAction(ISD::FMUL, VT, Legal);
        setOperationAction(ISD::FNEG, VT, Legal);
        setOperationAction(ISD::FSUB, VT, Legal);
      } else {
        setOperationAction(ISD::FABS, VT, Expand);
        setOperationAction(ISD::FADD, VT, Expand);
        setOperationAction(ISD::FMA, VT, Expand);
        setOperationAction(ISD::FMUL, VT, Expand);
        setOperationAction(ISD::FNEG, VT, Expand);
        setOperationAction(ISD::FSUB, VT, Expand);
      }

      // No special instructions for these.
      setOperationAction(ISD::FCBRT, VT, Expand);
      setOperationAction(ISD::FCEIL, VT, Expand);
      setOperationAction(ISD::FCOPYSIGN, VT, Expand);
      setOperationAction(ISD::FCOS, VT, Expand);
      setOperationAction(ISD::FDIV, VT, Expand);
      setOperationAction(ISD::FEXP, VT, Expand);
      setOperationAction(ISD::FEXP2, VT, Expand);
      setOperationAction(ISD::FFLOOR, VT, Expand);
      setOperationAction(ISD::FLOG, VT, Expand);
      setOperationAction(ISD::FLOG2, VT, Expand);
      setOperationAction(ISD::FLOG10, VT, Expand);
      setOperationAction(ISD::FMAXIMUM, VT, Expand);
      setOperationAction(ISD::FMINIMUM, VT, Expand);
      setOperationAction(ISD::FMAXNUM, VT, Expand);
      setOperationAction(ISD::FMINNUM, VT, Expand);
      setOperationAction(ISD::FNEARBYINT, VT, Expand);
      setOperationAction(ISD::FPOW, VT, Expand);
      setOperationAction(ISD::FPOWI, VT, Expand);
      setOperationAction(ISD::FREM, VT, Expand);
      setOperationAction(ISD::FRINT, VT, Expand);
      setOperationAction(ISD::FROUND, VT, Expand);
      setOperationAction(ISD::FSIN, VT, Expand);
      setOperationAction(ISD::FSINCOS, VT, Expand);
      setOperationAction(ISD::FSQRT, VT, Expand);
      setOperationAction(ISD::FTRUNC, VT, Expand);
      setOperationAction(ISD::LLRINT, VT, Expand);
      setOperationAction(ISD::LLROUND, VT, Expand);
      setOperationAction(ISD::LRINT, VT, Expand);
      setOperationAction(ISD::LROUND, VT, Expand);
    }
  }

  if (Subtarget.hasSingleFloat()) {
    setOperationAction(ISD::BITCAST, MVT::i32, Legal);
    setOperationAction(ISD::BITCAST, MVT::f32, Legal);
    setOperationAction(ISD::UINT_TO_FP, MVT::i32, Legal);
    setOperationAction(ISD::SINT_TO_FP, MVT::i32, Legal);
    setOperationAction(ISD::FP_TO_UINT, MVT::i32, Legal);
    setOperationAction(ISD::FP_TO_SINT, MVT::i32, Legal);
  } else {
    setOperationAction(ISD::BITCAST, MVT::i32, Expand);
    setOperationAction(ISD::BITCAST, MVT::f32, Expand);
    setOperationAction(ISD::UINT_TO_FP, MVT::i32, Expand);
    setOperationAction(ISD::SINT_TO_FP, MVT::i32, Expand);
    setOperationAction(ISD::FP_TO_UINT, MVT::i32, Expand);
    setOperationAction(ISD::FP_TO_SINT, MVT::i32, Expand);
  }

  setOperationAction(ISD::UINT_TO_FP, MVT::i64, Expand);
  setOperationAction(ISD::SINT_TO_FP, MVT::i64, Expand);
  setOperationAction(ISD::FP_TO_UINT, MVT::i64, Expand);
  setOperationAction(ISD::FP_TO_SINT, MVT::i64, Expand);

  setOperationAction(ISD::SETCC, MVT::f64, Expand);
  setOperationAction(ISD::BITCAST, MVT::i64, Expand);
  setOperationAction(ISD::BITCAST, MVT::f64, Expand);

  if (Subtarget.hasSingleFloat()) {
    setCondCodeAction(ISD::SETOGT, MVT::f32, Expand);
    setCondCodeAction(ISD::SETOGE, MVT::f32, Expand);
    setCondCodeAction(ISD::SETONE, MVT::f32, Expand);
    setCondCodeAction(ISD::SETUGE, MVT::f32, Expand);
    setCondCodeAction(ISD::SETUGT, MVT::f32, Expand);

    setTargetDAGCombine(ISD::FADD);
    setTargetDAGCombine(ISD::FSUB);
  }

  if (Subtarget.hasSingleFloat()) {
    setTargetDAGCombine(ISD::BRCOND);
  }

  if (Subtarget.hasLoop()) {
    setTargetDAGCombine(ISD::BR_CC);
  }

  // make BRCOND legal, its actually only legal for a subset of conds
  setOperationAction(ISD::BRCOND, MVT::Other, Legal);

  // Needed so that we don't try to implement f128 constant loads using
  // a load-and-extend of a f80 constant (in cases where the constant
  // would fit in an f80).
  for (MVT VT : MVT::fp_valuetypes()) {
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::f16, Expand);
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::f32, Expand);
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::f64, Expand);
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::f80, Expand);
  }

  setOperationAction(ISD::FP16_TO_FP, MVT::f64, Expand);
  setOperationAction(ISD::FP_TO_FP16, MVT::f64, Expand);
  setOperationAction(ISD::FP16_TO_FP, MVT::f32, Expand);
  setOperationAction(ISD::FP_TO_FP16, MVT::f32, Expand);

  // Floating-point truncation and stores need to be done separately.
  setTruncStoreAction(MVT::f64, MVT::f32, Expand);
  setTruncStoreAction(MVT::f64, MVT::f16, Expand);
  setTruncStoreAction(MVT::f32, MVT::f16, Expand);

  // Implement custom stack allocations
  setOperationAction(ISD::DYNAMIC_STACKALLOC, PtrVT, Custom);
  // Implement custom stack save and restore
  setOperationAction(ISD::STACKSAVE, MVT::Other, Custom);
  setOperationAction(ISD::STACKRESTORE, MVT::Other, Custom);

  // VASTART and VACOPY need to deal with the Xtensa-specific varargs
  // structure, but VAEND is a no-op.
  setOperationAction(ISD::VASTART, MVT::Other, Custom);
  // we use special va_list structure so we have to customize this
  setOperationAction(ISD::VAARG, MVT::Other, Expand);
  setOperationAction(ISD::VACOPY, MVT::Other, Custom);
  setOperationAction(ISD::VAEND, MVT::Other, Expand);

  setOperationAction(ISD::TRAP, MVT::Other, Legal);

  // to have the best chance and doing something good with fences custom lower
  // them
  setOperationAction(ISD::ATOMIC_FENCE, MVT::Other, Custom);

  if (!Subtarget.hasS32C1I()) {
    for (unsigned I = MVT::FIRST_INTEGER_VALUETYPE;
         I <= MVT::LAST_INTEGER_VALUETYPE; ++I) {
      MVT VT = MVT::SimpleValueType(I);
      if (isTypeLegal(VT)) {
        setOperationAction(ISD::ATOMIC_CMP_SWAP, VT, Expand);
        setOperationAction(ISD::ATOMIC_SWAP, VT, Expand);
        setOperationAction(ISD::ATOMIC_LOAD_ADD, VT, Expand);
        setOperationAction(ISD::ATOMIC_LOAD_SUB, VT, Expand);
        setOperationAction(ISD::ATOMIC_LOAD_AND, VT, Expand);
        setOperationAction(ISD::ATOMIC_LOAD_OR, VT, Expand);
        setOperationAction(ISD::ATOMIC_LOAD_XOR, VT, Expand);
        setOperationAction(ISD::ATOMIC_LOAD_NAND, VT, Expand);
        setOperationAction(ISD::ATOMIC_LOAD_MIN, VT, Expand);
        setOperationAction(ISD::ATOMIC_LOAD_MAX, VT, Expand);
        setOperationAction(ISD::ATOMIC_LOAD_UMIN, VT, Expand);
        setOperationAction(ISD::ATOMIC_LOAD_UMAX, VT, Expand);
      }
    }
  }

  if (Subtarget.hasS32C1I()) {
    setMaxAtomicSizeInBitsSupported(32);
    setMinCmpXchgSizeInBits(32);
  } else if (Subtarget.hasForcedAtomics()) {
    setMaxAtomicSizeInBitsSupported(32);
  } else {
    setMaxAtomicSizeInBitsSupported(0);
  }

  for (MVT VT : MVT::fixedlen_vector_valuetypes()) {
    if (isTypeLegal(VT)) {
      setOperationAction(ISD::CTPOP, VT, Expand);
      setOperationAction(ISD::SRL, VT, Expand);
      setOperationAction(ISD::SRA, VT, Expand);
      setOperationAction(ISD::SHL, VT, Expand);

      // Expand all divisions and remainders for vectors
      setOperationAction(ISD::SDIV, VT, Expand);
      setOperationAction(ISD::UDIV, VT, Expand);
      setOperationAction(ISD::SREM, VT, Expand);
      setOperationAction(ISD::UREM, VT, Expand);
    }
    setOperationAction(ISD::SDIVREM, VT, Expand);
    setOperationAction(ISD::UDIVREM, VT, Expand);

    setOperationAction(ISD::SELECT_CC, VT, Custom);
    setOperationAction(ISD::SETCC, VT, Custom);

    // Disable all narrowing stores and extending loads for vectors
    for (MVT InnerVT : MVT::fixedlen_vector_valuetypes()) {
      setTruncStoreAction(VT, InnerVT, Expand);
      setLoadExtAction(ISD::SEXTLOAD, VT, InnerVT, Expand);
      setLoadExtAction(ISD::ZEXTLOAD, VT, InnerVT, Expand);
      setLoadExtAction(ISD::EXTLOAD, VT, InnerVT, Expand);
    }
  }

  // Compute derived properties from the register classes
  computeRegisterProperties(STI.getRegisterInfo());
}

/// Return the register type for a given MVT
MVT XtensaTargetLowering::getRegisterTypeForCallingConv(LLVMContext &Context,
                                                        CallingConv::ID CC,
                                                        EVT VT) const {
  if (VT.isFloatingPoint())
    return MVT::i32;

  return TargetLowering::getRegisterTypeForCallingConv(Context, CC, VT);
}

bool XtensaTargetLowering::isFNegFree(EVT VT) const {
  if (!VT.isSimple())
    return false;

  switch (VT.getSimpleVT().SimpleTy) {
    case MVT::f32:
      return Subtarget.hasSingleFloat();
    default:
      break;
  }

  return false;
}

bool XtensaTargetLowering::isFMAFasterThanFMulAndFAdd(const MachineFunction &MF,
                                                      EVT VT) const {
  if (!VT.isSimple())
    return false;

  switch (VT.getSimpleVT().SimpleTy) {
  case MVT::f32:
    return Subtarget.hasSingleFloat();
  default:
    break;
  }

 return false;
}

bool XtensaTargetLowering::isCheapToSpeculateCtlz(Type *) const {
  return Subtarget.hasNSA();
}

bool XtensaTargetLowering::isCheapToSpeculateCttz(Type *) const {
  return Subtarget.hasNSA();
}

bool XtensaTargetLowering::isCtlzFast() const {
  return Subtarget.hasNSA();
}

/// If a physical register, this returns the register that receives the
/// exception address on entry to an EH pad.
Register XtensaTargetLowering::getExceptionPointerRegister(
    const Constant *PersonalityFn) const {
  return Xtensa::A2;
}

/// If a physical register, this returns the register that receives the
/// exception typeid on entry to a landing pad.
Register XtensaTargetLowering::getExceptionSelectorRegister(
    const Constant *PersonalityFn) const {
  return Xtensa::A3;
}

bool XtensaTargetLowering::isOffsetFoldingLegal(
    const GlobalAddressSDNode *GA) const {
  // The Xtensa target isn't yet aware of offsets.
  return false;
}

bool XtensaTargetLowering::isFPImmLegal(const APFloat &Imm, EVT VT,
                                        bool ForCodeSize) const {
  return false;
}

unsigned XtensaTargetLowering::getVaListSizeInBits(const DataLayout &DL) const {
  // 2 * sizeof(int*) + sizeof(int)
  return 3 * 4;
}

//===----------------------------------------------------------------------===//
// Inline asm support
//===----------------------------------------------------------------------===//
TargetLowering::ConstraintType
XtensaTargetLowering::getConstraintType(StringRef Constraint) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    case 'a':
    case 'd':
    case 'f':
    case 'r':
      return C_RegisterClass;

    default:
      break;
    }
  }
  return TargetLowering::getConstraintType(Constraint);
}

TargetLowering::ConstraintWeight
XtensaTargetLowering::getSingleConstraintMatchWeight(
    AsmOperandInfo &info, const char *constraint) const {
  ConstraintWeight weight = CW_Invalid;
  Value *CallOperandVal = info.CallOperandVal;
  // If we don't have a value, we can't do a match,
  // but allow it at the lowest weight.
  if (CallOperandVal == NULL)
    return CW_Default;

  Type *type = CallOperandVal->getType();

  // Look at the constraint type.
  switch (*constraint) {
  default:
    weight = TargetLowering::getSingleConstraintMatchWeight(info, constraint);
    break;

  case 'a':
  case 'd':
  case 'r':
    if (type->isIntegerTy())
      weight = CW_Register;
    break;
  case 'f':
    if (type->isFloatingPointTy())
      weight = CW_Register;
    break;

  }
  return weight;
}

std::pair<unsigned, const TargetRegisterClass *>
XtensaTargetLowering::getRegForInlineAsmConstraint(
    const TargetRegisterInfo *TRI, StringRef Constraint, MVT VT) const {
  if (Constraint.size() == 1) {
    // GCC Constraint Letters
    switch (Constraint[0]) {
    default:
      break;
    case 'a': // Address register
    case 'd': // Data register (equivalent to 'r')
    case 'r': // General-purpose register
      return std::make_pair(0U, &Xtensa::ARRegClass);
    case 'f': // Floating-point register
      if (Subtarget.hasSingleFloat())
        return std::make_pair(0U, &Xtensa::FPRRegClass);
    }
  }
  return TargetLowering::getRegForInlineAsmConstraint(TRI, Constraint, VT);
}

/// LowerAsmOperandForConstraint - Lower the specified operand into the Ops
/// vector.  If it is invalid, don't add anything to Ops.
void XtensaTargetLowering::LowerAsmOperandForConstraint(
    SDValue Op, StringRef Constraint, std::vector<SDValue> &Ops,
    SelectionDAG &DAG) const {
  SDLoc DL(Op);

  // Only support length 1 constraints for now.
  if (Constraint.size() > 1)
    return;

  TargetLowering::LowerAsmOperandForConstraint(Op, Constraint, Ops, DAG);
}

//===----------------------------------------------------------------------===//
//  DAG Combine functions
//===----------------------------------------------------------------------===//

static SDValue performMADD_MSUBCombine(SDNode *ROOTNode, SelectionDAG &CurDAG,
                                       const XtensaSubtarget &Subtarget) {
  SDValue LHS = ROOTNode->getOperand(0);
  SDValue RHS = ROOTNode->getOperand(1);

  if (LHS.getValueType() != MVT::f32 || (LHS.getOpcode() != ISD::FMUL && RHS.getOpcode() != ISD::FMUL))
    return SDValue();

  SDLoc DL(ROOTNode);
  bool IsAdd = ROOTNode->getOpcode() == ISD::FADD;

  SDValue Mult, AddOperand;
  bool Inverted;

  if (LHS.getOpcode() == ISD::FMUL)
    Mult = LHS, AddOperand = RHS, Inverted = false;
  else
    Mult = RHS, AddOperand = LHS, Inverted = true;

  if (!Mult.hasOneUse())
    return SDValue();

  SDValue MultOperand0 = Mult->getOperand(0), MultOperand1 = Mult->getOperand(1);

  if (!IsAdd) {
    if (Inverted)
      MultOperand0 = CurDAG.getNode(ISD::FNEG, DL, MVT::f32, MultOperand0);
    else
      AddOperand = CurDAG.getNode(ISD::FNEG, DL, MVT::f32, AddOperand);
  }

  SDValue FMAOps[3] = {MultOperand0, MultOperand1, AddOperand};
  EVT VTs[3] = {MVT::f32, MVT::f32, MVT::f32};

  return CurDAG.getNode(ISD::FMA, DL, VTs, FMAOps);
}

static SDValue performSUBCombine(SDNode *N, SelectionDAG &DAG,
                                 TargetLowering::DAGCombinerInfo &DCI,
                                 const XtensaSubtarget &Subtarget) {
  if (DCI.isBeforeLegalizeOps()) {
    if (Subtarget.hasSingleFloat() && N->getValueType(0) == MVT::f32)
      return performMADD_MSUBCombine(N, DAG, Subtarget);
  }
  return SDValue();
}

static SDValue performADDCombine(SDNode *N, SelectionDAG &DAG,
                                 TargetLowering::DAGCombinerInfo &DCI,
                                 const XtensaSubtarget &Subtarget) {
  if (DCI.isBeforeLegalizeOps()) {
    if (Subtarget.hasSingleFloat() && N->getValueType(0) == MVT::f32)
      return performMADD_MSUBCombine(N, DAG, Subtarget);
  }
  return SDValue();
}

static SDValue SearchLoopIntrinsic(SDValue N, ISD::CondCode &CC, int &Imm,
                                   bool &Negate) {
  switch (N->getOpcode()) {
  default:
    break;
  case ISD::XOR: {
    if (!isa<ConstantSDNode>(N.getOperand(1)))
      return SDValue();
    if (!cast<ConstantSDNode>(N.getOperand(1))->isOne())
      return SDValue();
    Negate = !Negate;
    return SearchLoopIntrinsic(N.getOperand(0), CC, Imm, Negate);
  }
  case ISD::SETCC: {
    auto *Const = dyn_cast<ConstantSDNode>(N.getOperand(1));
    if (!Const)
      return SDValue();
    if (Const->isZero())
      Imm = 0;
    else if (Const->isOne())
      Imm = 1;
    else
      return SDValue();
    CC = cast<CondCodeSDNode>(N.getOperand(2))->get();
    return SearchLoopIntrinsic(N->getOperand(0), CC, Imm, Negate);
  }
  case ISD::INTRINSIC_W_CHAIN: {
    unsigned IntOp = cast<ConstantSDNode>(N.getOperand(1))->getZExtValue();
    if (IntOp != Intrinsic::loop_decrement_reg)
      return SDValue();
    return N;
  }
  }
  return SDValue();
}

static SDValue PerformHWLoopCombine(SDNode *N, SelectionDAG &DAG,
                                    TargetLowering::DAGCombinerInfo &DCI,
                                    const XtensaSubtarget &Subtarget) {
  SDValue Chain = N->getOperand(0);
  SDLoc DL(N);
  SDValue Cond;
  SDValue Dest;
  ISD::CondCode CC = ISD::SETEQ;
  int Imm = 1;
  bool Negate = false;

  assert(N->getOpcode() == ISD::BR_CC && "Expected BR_CC!");
  CC = cast<CondCodeSDNode>(N->getOperand(1))->get();
  Cond = N->getOperand(2);
  Dest = N->getOperand(4);
  if (auto *Const = dyn_cast<ConstantSDNode>(N->getOperand(3))) {
    if (!Const->isOne() && !Const->isZero())
      return SDValue();
    Imm = Const->getZExtValue();
  } else
    return SDValue();

  SDValue Int = SearchLoopIntrinsic(Cond, CC, Imm, Negate);
  if (Int) {
    assert((N->hasOneUse() && N->use_begin()->getOpcode() == ISD::BR) &&
           "expected single br user");
    SDNode *Br = *N->use_begin();
    SDValue OtherTarget = Br->getOperand(1);

    if (Negate)
      CC = ISD::getSetCCInverse(CC, /* Integer inverse */ MVT::i32);

    auto IsTrueIfZero = [](ISD::CondCode CC, int Imm) {
      return (CC == ISD::SETEQ && Imm == 0) || (CC == ISD::SETNE && Imm == 1) ||
             (CC == ISD::SETLT && Imm == 1) || (CC == ISD::SETULT && Imm == 1);
    };

    auto IsFalseIfZero = [](ISD::CondCode CC, int Imm) {
      return (CC == ISD::SETEQ && Imm == 1) || (CC == ISD::SETNE && Imm == 0) ||
             (CC == ISD::SETGT && Imm == 0) ||
             (CC == ISD::SETUGT && Imm == 0) ||
             (CC == ISD::SETGE && Imm == 1) || (CC == ISD::SETUGE && Imm == 1);
    };

    if (IsTrueIfZero(CC, Imm)) {
      SDValue NewBrOps[] = {Br->getOperand(0), Dest};
      SDValue NewBr = DAG.getNode(ISD::BR, SDLoc(Br), MVT::Other, NewBrOps);
      DAG.ReplaceAllUsesOfValueWith(SDValue(Br, 0), NewBr);
      Dest = OtherTarget;
    } else if (!IsFalseIfZero(CC, Imm)) {
      llvm_unreachable("unsupported condition");
    }
    SDLoc dl(Int);
    SDValue Elements = Int.getOperand(2);
    SDValue Size = DAG.getTargetConstant(
        cast<ConstantSDNode>(Int.getOperand(3))->getZExtValue(), dl, MVT::i32);
    SDValue Args[] = {
        Int.getOperand(0),
        Elements,
        Size,
    };
    SDValue LoopDec = DAG.getNode(XtensaISD::LOOPDEC, dl,
                                  DAG.getVTList(MVT::i32, MVT::Other), Args);

    // We now need to make the intrinsic dead (it cannot be instruction
    // selected).
    DAG.ReplaceAllUsesWith(Int.getNode(), LoopDec.getNode());

    Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                        SDValue(LoopDec.getNode(), 1), Chain);

    SDValue EndArgs[] = {Chain, SDValue(LoopDec.getNode(), 0), Dest};
    return DAG.getNode(XtensaISD::LOOPBR, dl, MVT::Other, EndArgs);
  }
  return SDValue();
}

static SDValue PerformBRCONDCombine(SDNode *N, SelectionDAG &DAG,
                                    TargetLowering::DAGCombinerInfo &DCI,
                                    const XtensaSubtarget &Subtarget) {
  SDValue Chain = N->getOperand(0);
  SDLoc DL(N);
  SDValue Cond = N->getOperand(1);
  SDValue Dest = N->getOperand(2);
  ISD::CondCode CC = ISD::SETEQ;

  if (Cond.getOpcode() != ISD::SETCC)
    return SDValue();

  CC = cast<CondCodeSDNode>(Cond->getOperand(2))->get();
  SDValue LHS = Cond->getOperand(0);
  SDValue RHS = Cond->getOperand(1);

  if (LHS.getValueType() != MVT::i32)
    return SDValue();

  return DAG.getNode(ISD::BR_CC, DL, MVT::isVoid, Chain, DAG.getCondCode(CC),
                     LHS, RHS, Dest);
}

static SDValue PerformBUILD_VECTORCombine(SDNode *N, SelectionDAG &DAG,
                                          TargetLowering::DAGCombinerInfo &DCI,
                                          const XtensaSubtarget &Subtarget) {
  SDLoc DL(N);
  EVT VT = N->getValueType(0);
  SDValue Op0 = N->getOperand(0);
  ConstantSDNode *Const = dyn_cast<ConstantSDNode>(Op0);
  if (VT == MVT::v1i64 && Const) {
    int64_t Val = Const->getSExtValue();
    if (Val <= std::numeric_limits<uint32_t>::max())
      return DAG.getNode(XtensaISD::BUILD_VEC, DL, MVT::v1i64,
                         DAG.getConstant(Val, DL, MVT::i32));
  }
  return SDValue();
}

static SDValue PerformBITCASTCombine(SDNode *N, SelectionDAG &DAG,
                                     TargetLowering::DAGCombinerInfo &DCI,
                                     const XtensaSubtarget &Subtarget) {
  // (vNi1 (bitcast (iN (trunc i32)))) -> (vNi1 (xtensa_bitcast i32))
  SDLoc DL(N);
  SDValue Op = N->getOperand(0);

  if (N->getOpcode() != ISD::BITCAST || Op.getOpcode() != ISD::TRUNCATE)
    return SDValue();

  SDValue Int = Op.getOperand(0);
  llvm::EVT BoolVT = N->getValueType(0);

  if (!BoolVT.isVector() || BoolVT.getVectorElementType() != MVT::i1 ||
      Int.getValueType() != MVT::i32)
    return SDValue();

  SDValue Trunc = DAG.getNode(XtensaISD::TRUNC, DL, BoolVT, Int);

  return Trunc;
}

static SDValue
PerformExtractSubvectorCombine(SDNode *N, SelectionDAG &DAG,
                               TargetLowering::DAGCombinerInfo &DCI,
                               const XtensaSubtarget &Subtarget) {
  // (vNi1 (extract_subvector (v8i1 (load x))) -> (vNi1 (load x))
  SDLoc DL(N);
  SDValue Load = N->getOperand(0);

  if (N->getOpcode() != ISD::EXTRACT_SUBVECTOR)
    return SDValue();

  EVT LoadVT = Load.getValueType();
  EVT BoolVT = N->getValueType(0);

  if (!BoolVT.isVector() || BoolVT.getVectorElementType() != MVT::i1)
    return SDValue();

  if (Load.getOpcode() != ISD::LOAD)
    return SDValue();

  LoadSDNode *LdNode = cast<LoadSDNode>(Load.getNode());

  if (!LoadVT.isVector() || LoadVT.getVectorElementType() != MVT::i1)
    return SDValue();

  SDValue NewLoad =
      DAG.getLoad(BoolVT, DL, LdNode->getChain(), LdNode->getBasePtr(),
                  LdNode->getPointerInfo(), LdNode->getOriginalAlign(),
                  LdNode->getMemOperand()->getFlags());

  return NewLoad;
}
static SDValue PerformSTORECombine(SDNode *N, SelectionDAG &DAG,
                                   TargetLowering::DAGCombinerInfo &DCI,
                                   const XtensaSubtarget &Subtarget) {
  // (store (v8i1 (concat_vector (vNi1 elt) undef )) addr off)
  //  -> (store (vNi1 elt) addr off)
  SDLoc DL(N);

  if (N->getOpcode() != ISD::STORE)
    return SDValue();

  StoreSDNode *StNode = cast<StoreSDNode>(N);

  SDValue Concat = N->getOperand(1);
  EVT BoolVT = Concat.getValueType();

  if ((Concat.getOpcode() != ISD::CONCAT_VECTORS) || !BoolVT.isVector() ||
      (BoolVT.getVectorElementType() != MVT::i1))
    return SDValue();

  SDValue Val = Concat.getNode()->getOperand(0);
  EVT ValVT = Val.getValueType();

  if (!ValVT.isVector() || ValVT.getVectorElementType() != MVT::i1 ||
      ValVT.getSizeInBits() > 8) {
    return SDValue();
  }

  return DAG.getStore(StNode->getChain(), DL, Val, StNode->getBasePtr(),
                      StNode->getMemOperand());
}

SDValue XtensaTargetLowering::PerformDAGCombine(SDNode *N,
                                                DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  unsigned Opc = N->getOpcode();

  switch (Opc) {
  default:
    break;
  case ISD::FADD:
    return performADDCombine(N, DAG, DCI, Subtarget);
  case ISD::FSUB:
    return performSUBCombine(N, DAG, DCI, Subtarget);
  case ISD::BR_CC:
    return PerformHWLoopCombine(N, DAG, DCI, Subtarget);
  case ISD::BRCOND:
    return PerformBRCONDCombine(N, DAG, DCI, Subtarget);
  case ISD::BUILD_VECTOR:
    return PerformBUILD_VECTORCombine(N, DAG, DCI, Subtarget);
  case ISD::BITCAST:
    return PerformBITCASTCombine(N, DAG, DCI, Subtarget);
  case ISD::EXTRACT_SUBVECTOR:
    return PerformExtractSubvectorCombine(N, DAG, DCI, Subtarget);
  case ISD::STORE:
    return PerformSTORECombine(N, DAG, DCI, Subtarget);
  }

  return SDValue();
}

//===----------------------------------------------------------------------===//
// Calling conventions
//===----------------------------------------------------------------------===//

#include "XtensaGenCallingConv.inc"

static bool CC_Xtensa_Custom(unsigned ValNo, MVT ValVT, MVT LocVT,
                             CCValAssign::LocInfo LocInfo,
                             ISD::ArgFlagsTy ArgFlags, CCState &State) {
  static const MCPhysReg IntRegs[] = {Xtensa::A2, Xtensa::A3, Xtensa::A4,
                                      Xtensa::A5, Xtensa::A6, Xtensa::A7};
  static const MCPhysReg BoolRegs[] = {
      Xtensa::B0,  Xtensa::B1,  Xtensa::B2,  Xtensa::B3,
      Xtensa::B4,  Xtensa::B5,  Xtensa::B6,  Xtensa::B7,
      Xtensa::B8,  Xtensa::B9,  Xtensa::B10, Xtensa::B11,
      Xtensa::B12, Xtensa::B13, Xtensa::B14, Xtensa::B15};

  ArrayRef<MCPhysReg> BR2Regs(Xtensa::BR2RegClass.begin(),
                              Xtensa::BR2RegClass.end());

  ArrayRef<MCPhysReg> BR4Regs(Xtensa::BR4RegClass.begin(),
                              Xtensa::BR4RegClass.end());

  if (ArgFlags.isByVal()) {
    Align ByValAlign = ArgFlags.getNonZeroByValAlign();
    unsigned ByValSize = ArgFlags.getByValSize();
    if (ByValSize < 4) {
      ByValSize = 4;
    }
    if (ByValAlign < Align(4)) {
      ByValAlign = Align(4);
    }
    unsigned Offset = State.AllocateStack(ByValSize, ByValAlign);
    State.addLoc(CCValAssign::getMem(ValNo, ValVT, Offset, LocVT, LocInfo));
    // Mark all unused registers as allocated to avoid misuse
    // of such registers.
    while (State.AllocateReg(IntRegs))
      ;
    return false;
  }

  // Promote i8 and i16
  if (LocVT == MVT::i8 || LocVT == MVT::i16) {
    LocVT = MVT::i32;
    if (ArgFlags.isSExt())
      LocInfo = CCValAssign::SExt;
    else if (ArgFlags.isZExt())
      LocInfo = CCValAssign::ZExt;
    else
      LocInfo = CCValAssign::AExt;
  }

  unsigned Register;

  Align OrigAlign = ArgFlags.getNonZeroOrigAlign();
  bool needs64BitAlign = (ValVT == MVT::i32 && OrigAlign == Align(8));
  bool needs128BitAlign = (ValVT == MVT::i32 && OrigAlign == Align(16));

  if (ValVT == MVT::i32 || ValVT == MVT::f32) {
    Register = State.AllocateReg(IntRegs);
    // If this is the first part of an i64 arg,
    // the allocated register must be either A2, A4 or A6.
    if (needs64BitAlign && (Register == Xtensa::A3 || Register == Xtensa::A5 ||
                            Register == Xtensa::A7))
      Register = State.AllocateReg(IntRegs);
    // arguments with 16byte alignment must be passed in the first register or
    // passed via stack
    if (needs128BitAlign && (Register != Xtensa::A2))
      while ((Register = State.AllocateReg(IntRegs)))
        ;
    LocVT = MVT::i32;
  } else if (ValVT == MVT::f64) {
    // Allocate int register and shadow next int register.
    Register = State.AllocateReg(IntRegs);
    if (Register == Xtensa::A3 || Register == Xtensa::A5 ||
        Register == Xtensa::A7)
      Register = State.AllocateReg(IntRegs);
    State.AllocateReg(IntRegs);
    LocVT = MVT::i32;
  } else if (ValVT == MVT::v1i1) {
    Register = State.AllocateReg(BoolRegs);
  } else if (ValVT == MVT::v2i1) {
    Register = State.AllocateReg(BR2Regs);
    LocVT = ValVT;
  } else if (ValVT == MVT::v4i1) {
    Register = State.AllocateReg(BR4Regs);
    LocVT = ValVT;
  } else if (isVecVT(ValVT)) {
    Register = State.AllocateReg(VecRegs);
    LocVT = ValVT;
  } else
    llvm_unreachable("Cannot handle this ValVT.");

  if (!Register) {
    unsigned Offset = State.AllocateStack(ValVT.getStoreSize(), OrigAlign);
    State.addLoc(CCValAssign::getMem(ValNo, ValVT, Offset, LocVT, LocInfo));
  } else {
    State.addLoc(CCValAssign::getReg(ValNo, ValVT, Register, LocVT, LocInfo));
  }

  return false;
}

CCAssignFn *XtensaTargetLowering::CCAssignFnForCall(CallingConv::ID CC,
                                                    bool IsVarArg) const {
  return CC_Xtensa_Custom;
}

SDValue XtensaTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  XtensaFunctionInfo *XtensaFI = MF.getInfo<XtensaFunctionInfo>();
  EVT PtrVT = getPointerTy(MF.getDataLayout());

  XtensaFI->setVarArgsFrameIndex(0);

  // Used with vargs to acumulate store chains.
  std::vector<SDValue> OutChains;

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), ArgLocs,
                 *DAG.getContext());

  CCInfo.AnalyzeFormalArguments(Ins, CCAssignFnForCall(CallConv, IsVarArg));

  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    // Arguments stored on registers
    if (VA.isRegLoc()) {
      EVT RegVT = VA.getLocVT();
      const TargetRegisterClass *RC;

      if (RegVT == MVT::i32) {
        RC = &Xtensa::ARRegClass;
      } else if (RegVT == MVT::v1i1) {
        RC = &Xtensa::BRRegClass;
      } else if (RegVT == MVT::v2i1) {
        RC = &Xtensa::BR2RegClass;
      } else if (RegVT == MVT::v4i1) {
        RC = &Xtensa::BR4RegClass;
      } else if (isVecVT(RegVT)) {
        RC = &Xtensa::AE_DRRegClass;
      } else
        llvm_unreachable("RegVT not supported by FormalArguments Lowering");

      // Transform the arguments stored on
      // physical registers into virtual ones
      unsigned Register = 0;
      unsigned FrameReg = Subtarget.getRegisterInfo()->getFrameRegister(MF);

      // Argument passed in FrameReg in WinABI we save in A8 (in emitPrologue),
      // so load argument from A8
      if (Subtarget.isWinABI() && (VA.getLocReg() == FrameReg)) {
        Register = MF.addLiveIn(Xtensa::A8, RC);
        XtensaFI->setSaveFrameRegister();
      } else {
        Register = MF.addLiveIn(VA.getLocReg(), RC);
      }

      SDValue ArgValue = DAG.getCopyFromReg(Chain, DL, Register, RegVT);

      // If this is an 8 or 16-bit value, it has been passed promoted
      // to 32 bits.  Insert an assert[sz]ext to capture this, then
      // truncate to the right size.
      if (VA.getLocInfo() != CCValAssign::Full) {
        unsigned Opcode = 0;
        if (VA.getLocInfo() == CCValAssign::SExt)
          Opcode = ISD::AssertSext;
        else if (VA.getLocInfo() == CCValAssign::ZExt)
          Opcode = ISD::AssertZext;
        if (Opcode)
          ArgValue = DAG.getNode(Opcode, DL, RegVT, ArgValue,
                                 DAG.getValueType(VA.getValVT()));
        ArgValue = DAG.getNode((VA.getValVT() == MVT::f32) ? ISD::BITCAST
                                                           : ISD::TRUNCATE,
                               DL, VA.getValVT(), ArgValue);
      }

      InVals.push_back(ArgValue);

    } else {
      assert(VA.isMemLoc());

      EVT ValVT = VA.getValVT();

      // The stack pointer offset is relative to the caller stack frame.
      int FI = MFI.CreateFixedObject(ValVT.getStoreSize(), VA.getLocMemOffset(),
                                     true);

      if (Ins[VA.getValNo()].Flags.isByVal()) {
        // Assume that in this case load operation is created
        SDValue FIN = DAG.getFrameIndex(FI, MVT::i32);
        InVals.push_back(FIN);
      } else {
        // Create load nodes to retrieve arguments from the stack
        SDValue FIN =
            DAG.getFrameIndex(FI, getFrameIndexTy(DAG.getDataLayout()));
        InVals.push_back(DAG.getLoad(
            ValVT, DL, Chain, FIN,
            MachinePointerInfo::getFixedStack(DAG.getMachineFunction(), FI)));
      }
    }
  }

  if (IsVarArg) {
    ArrayRef<MCPhysReg> ArgRegs = ArrayRef(XtensaArgRegs);
    unsigned Idx = CCInfo.getFirstUnallocated(ArgRegs);
    const TargetRegisterClass *RC = &Xtensa::ARRegClass;
    MachineFrameInfo &MFI = MF.getFrameInfo();
    MachineRegisterInfo &RegInfo = MF.getRegInfo();
    unsigned RegSize = 4;
    MVT RegTy = MVT::getIntegerVT(RegSize * 8);

    XtensaFI->setVarArgsFirstGPR(Idx + 2); // 2 - number of a2 register

    XtensaFI->setVarArgsStackOffset(MFI.CreateFixedObject(
        PtrVT.getSizeInBits() / 8, CCInfo.getStackSize(), true));

    // Offset of the first variable argument from stack pointer, and size of
    // the vararg save area. For now, the varargs save area is either zero or
    // large enough to hold a0-a7.
    int VaArgOffset, VarArgsSaveSize;

    // If all registers are allocated, then all varargs must be passed on the
    // stack and we don't need to save any argregs.
    if (ArgRegs.size() == Idx) {
      VaArgOffset = CCInfo.getStackSize();
      VarArgsSaveSize = 0;
    } else {
      VarArgsSaveSize = RegSize * (ArgRegs.size() - Idx);
      VaArgOffset = -VarArgsSaveSize;
    }

    // Record the frame index of the first variable argument
    // which is a value necessary to VASTART.
    int FI = MFI.CreateFixedObject(RegSize, VaArgOffset, true);
    XtensaFI->setVarArgsFrameIndex(FI);

    // Copy the integer registers that may have been used for passing varargs
    // to the vararg save area.
    for (unsigned I = Idx; I < ArgRegs.size(); ++I, VaArgOffset += RegSize) {
      const unsigned Reg = RegInfo.createVirtualRegister(RC);
      unsigned FrameReg = Subtarget.getRegisterInfo()->getFrameRegister(MF);

      // Argument passed in FrameReg we save in A8 (in emitPrologue),
      // so load argument from A8
      if (ArgRegs[I] == FrameReg) {
        RegInfo.addLiveIn(Xtensa::A8, Reg);
        XtensaFI->setSaveFrameRegister();
      } else {
        RegInfo.addLiveIn(ArgRegs[I], Reg);
      }

      SDValue ArgValue = DAG.getCopyFromReg(Chain, DL, Reg, RegTy);
      FI = MFI.CreateFixedObject(RegSize, VaArgOffset, true);
      SDValue PtrOff = DAG.getFrameIndex(FI, getPointerTy(DAG.getDataLayout()));
      SDValue Store = DAG.getStore(Chain, DL, ArgValue, PtrOff,
                                   MachinePointerInfo::getFixedStack(MF, FI));
      cast<StoreSDNode>(Store.getNode())
          ->getMemOperand()
          ->setValue((Value *)nullptr);
      OutChains.push_back(Store);
    }
  }

  // All stores are grouped in one node to allow the matching between
  // the size of Ins and InVals. This only happens when on varg functions
  if (!OutChains.empty()) {
    OutChains.push_back(Chain);
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, OutChains);
  }

  return Chain;
}

static void fail(const SDLoc &DL, SelectionDAG &DAG, const char *Msg) {
  MachineFunction &MF = DAG.getMachineFunction();
  DAG.getContext()->diagnose(
      DiagnosticInfoUnsupported(MF.getFunction(), Msg, DL.getDebugLoc()));
}

SDValue
XtensaTargetLowering::LowerCall(CallLoweringInfo &CLI,
                                SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG = CLI.DAG;
  SDLoc &DL = CLI.DL;
  SmallVector<ISD::OutputArg, 32> &Outs = CLI.Outs;
  SmallVector<SDValue, 32> &OutVals = CLI.OutVals;
  SmallVector<ISD::InputArg, 32> &Ins = CLI.Ins;
  SDValue Chain = CLI.Chain;
  SDValue Callee = CLI.Callee;
  bool &IsTailCall = CLI.IsTailCall;
  CallingConv::ID CallConv = CLI.CallConv;
  bool IsVarArg = CLI.IsVarArg;

  MachineFunction &MF = DAG.getMachineFunction();
  EVT PtrVT = getPointerTy(DAG.getDataLayout());
  const TargetFrameLowering *TFL = Subtarget.getFrameLowering();

  // TODO: Support tail call optimization.
  if (IsTailCall) {
    if (CLI.CB && CLI.CB->isMustTailCall())
      fail(DL, DAG, "tail call is not implemented");
    IsTailCall = false;
  }

  // Analyze the operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());

  CCAssignFn *CC = CCAssignFnForCall(CallConv, IsVarArg);

  CCInfo.AnalyzeCallOperands(Outs, CC);

  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NumBytes = CCInfo.getStackSize();

  Align StackAlignment = TFL->getStackAlign();
  unsigned NextStackOffset = alignTo(NumBytes, StackAlignment);

  Chain = DAG.getCALLSEQ_START(Chain, NextStackOffset, 0, DL);

  // Copy argument values to their designated locations.
  std::deque<std::pair<unsigned, SDValue>> RegsToPass;
  SmallVector<SDValue, 8> MemOpChains;
  SDValue StackPtr;
  for (unsigned I = 0, E = ArgLocs.size(); I != E; ++I) {
    CCValAssign &VA = ArgLocs[I];
    SDValue ArgValue = OutVals[I];
    ISD::ArgFlagsTy Flags = Outs[I].Flags;

    if (VA.isRegLoc())
      // Queue up the argument copies and emit them at the end.
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), ArgValue));
    else if (Flags.isByVal()) {
      assert(VA.isMemLoc());
      assert(Flags.getByValSize() &&
             "ByVal args of size 0 should have been ignored by front-end.");
      assert(!IsTailCall &&
             "Do not tail-call optimize if there is a byval argument.");

      if (!StackPtr.getNode())
        StackPtr = DAG.getCopyFromReg(Chain, DL, Xtensa::SP, PtrVT);
      unsigned Offset = VA.getLocMemOffset();
      SDValue Address = DAG.getNode(ISD::ADD, DL, PtrVT, StackPtr,
                                    DAG.getIntPtrConstant(Offset, DL));
      SDValue SizeNode = DAG.getConstant(Flags.getByValSize(), DL, MVT::i32);
      SDValue Memcpy = DAG.getMemcpy(
          Chain, DL, Address, ArgValue, SizeNode, Flags.getNonZeroByValAlign(),
          /*isVolatile=*/false, /*AlwaysInline=*/false,
          /*CI=*/nullptr, std::nullopt, MachinePointerInfo(), MachinePointerInfo());
      MemOpChains.push_back(Memcpy);
    } else {
      assert(VA.isMemLoc() && "Argument not register or memory");

      // Work out the address of the stack slot.  Unpromoted ints and
      // floats are passed as right-justified 8-byte values.
      if (!StackPtr.getNode())
        StackPtr = DAG.getCopyFromReg(Chain, DL, Xtensa::SP, PtrVT);
      unsigned Offset = VA.getLocMemOffset();
      SDValue Address = DAG.getNode(ISD::ADD, DL, PtrVT, StackPtr,
                                    DAG.getIntPtrConstant(Offset, DL));

      // Emit the store.
      MemOpChains.push_back(
          DAG.getStore(Chain, DL, ArgValue, Address, MachinePointerInfo()));
    }
  }

  // Join the stores, which are independent of one another.
  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, MemOpChains);

  // Build a sequence of copy-to-reg nodes, chained and glued together.
  SDValue Glue;
  for (unsigned I = 0, E = RegsToPass.size(); I != E; ++I) {
    unsigned Reg = RegsToPass[I].first;
    if (Subtarget.isWinABI())
      Reg = toCallerWindow(Reg);
    Chain = DAG.getCopyToReg(Chain, DL, Reg, RegsToPass[I].second, Glue);
    Glue = Chain.getValue(1);
  }
  std::string name;
  unsigned char TF = 0;
  bool HasShortCallAttr = false;

  // Accept direct calls by converting symbolic call addresses to the
  // associated Target* opcodes.
  if (ExternalSymbolSDNode *E = dyn_cast<ExternalSymbolSDNode>(Callee)) {
    name = E->getSymbol();
    TF = E->getTargetFlags();
    if (isPositionIndependent()) {
      report_fatal_error("PIC relocations is not supported");
    } else
      Callee = DAG.getTargetExternalSymbol(E->getSymbol(), PtrVT, TF);
  } else if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
    const GlobalValue *GV = G->getGlobal();
    name = GV->getName().str();
    if (auto *F = dyn_cast<Function>(GV))
      if (F->hasFnAttribute("short-call")) {
        HasShortCallAttr = true;
        Callee = DAG.getTargetGlobalAddress(
            G->getGlobal(), DL, Callee.getValueType(), 0, 0 /* TargetFlags */);
      }
  }

  if (!name.empty() && isLongCall(name.c_str()) && !HasShortCallAttr) {
    // Create a constant pool entry for the callee address
    XtensaCP::XtensaCPModifier Modifier = XtensaCP::no_modifier;

    XtensaConstantPoolValue *CPV = XtensaConstantPoolSymbol::Create(
        *DAG.getContext(), name.c_str(), 0 /* XtensaCLabelIndex */, false,
        Modifier);

    // Get the address of the callee into a register
    SDValue CPAddr = DAG.getTargetConstantPool(CPV, PtrVT, Align(4), 0, TF);
    SDValue CPWrap = getAddrPCRel(CPAddr, DAG);
    Callee = CPWrap;
  }

  // The first call operand is the chain and the second is the target address.
  SmallVector<SDValue, 8> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  // Add a register mask operand representing the call-preserved registers.
  const TargetRegisterInfo *TRI = Subtarget.getRegisterInfo();
  const uint32_t *Mask = TRI->getCallPreservedMask(MF, CallConv);
  assert(Mask && "Missing call preserved mask for calling convention");
  Ops.push_back(DAG.getRegisterMask(Mask));

  // Add argument registers to the end of the list so that they are
  // known live into the call.
  for (unsigned I = 0, E = RegsToPass.size(); I != E; ++I) {
    unsigned Reg = RegsToPass[I].first;
    if (Subtarget.isWinABI())
      Reg = toCallerWindow(Reg);
    Ops.push_back(DAG.getRegister(Reg, RegsToPass[I].second.getValueType()));
  }

  // Glue the call to the argument copies, if any.
  if (Glue.getNode())
    Ops.push_back(Glue);

  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);
  Chain = DAG.getNode(Subtarget.isWinABI() ? XtensaISD::CALLW : XtensaISD::CALL,
                      DL, NodeTys, Ops);
  Glue = Chain.getValue(1);

  // Mark the end of the call, which is glued to the call itself.
  Chain = DAG.getCALLSEQ_END(Chain, DAG.getConstant(NumBytes, DL, PtrVT, true),
                             DAG.getConstant(0, DL, PtrVT, true), Glue, DL);
  Glue = Chain.getValue(1);

  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RetLocs;
  CCState RetCCInfo(CallConv, IsVarArg, MF, RetLocs, *DAG.getContext());
  RetCCInfo.AnalyzeCallResult(Ins, Subtarget.isWinABI() ? RetCCW_Xtensa
                                                        : RetCC_Xtensa);

  // Copy all of the result registers out of their specified physreg.
  for (unsigned I = 0, E = RetLocs.size(); I != E; ++I) {
    CCValAssign &VA = RetLocs[I];

    // Copy the value out, gluing the copy to the end of the call sequence.
    unsigned Reg = VA.getLocReg();
    SDValue RetValue = DAG.getCopyFromReg(Chain, DL, Reg, VA.getLocVT(), Glue);
    Chain = RetValue.getValue(1);
    Glue = RetValue.getValue(2);

    InVals.push_back(RetValue);
  }
  return Chain;
}

bool XtensaTargetLowering::CanLowerReturn(
    CallingConv::ID CallConv, MachineFunction &MF, bool IsVarArg,
    const SmallVectorImpl<ISD::OutputArg> &Outs, LLVMContext &Context) const {
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, RVLocs, Context);
  return CCInfo.CheckReturn(Outs, RetCC_Xtensa);
}

SDValue
XtensaTargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                                  bool IsVarArg,
                                  const SmallVectorImpl<ISD::OutputArg> &Outs,
                                  const SmallVectorImpl<SDValue> &OutVals,
                                  const SDLoc &DL, SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();

  // Assign locations to each returned value.
  SmallVector<CCValAssign, 16> RetLocs;
  CCState RetCCInfo(CallConv, IsVarArg, MF, RetLocs, *DAG.getContext());
  RetCCInfo.AnalyzeReturn(Outs, RetCC_Xtensa);

  SDValue Glue;
  // Quick exit for void returns
  if (RetLocs.empty())
    return DAG.getNode(Subtarget.isWinABI() ? XtensaISD::RETW
                                            : XtensaISD::RET,
                       DL, MVT::Other, Chain);

  // Copy the result values into the output registers.
  SmallVector<SDValue, 4> RetOps;
  RetOps.push_back(Chain);
  for (unsigned I = 0, E = RetLocs.size(); I != E; ++I) {
    CCValAssign &VA = RetLocs[I];
    SDValue RetValue = OutVals[I];

    // Make the return register live on exit.
    assert(VA.isRegLoc() && "Can only return in registers!");

    // Chain and glue the copies together.
    unsigned Register = VA.getLocReg();
    Chain = DAG.getCopyToReg(Chain, DL, Register, RetValue, Glue);
    Glue = Chain.getValue(1);
    RetOps.push_back(DAG.getRegister(Register, VA.getLocVT()));
  }

  // Update chain and glue.
  RetOps[0] = Chain;
  if (Glue.getNode())
    RetOps.push_back(Glue);

  return DAG.getNode(Subtarget.isWinABI() ? XtensaISD::RETW
                                          : XtensaISD::RET,
                     DL, MVT::Other, RetOps);
}
 
SDValue XtensaTargetLowering::LowerBR_CC(SDValue Op, SelectionDAG &DAG) const {
  SDValue Chain = Op.getOperand(0);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(1))->get();
  SDValue LHS = Op.getOperand(2);
  SDValue RHS = Op.getOperand(3);
  SDValue Dest = Op.getOperand(4);
  SDLoc DL(Op);

  if (LHS.getValueType() == MVT::f32) {
    SDValue TargetCC = DAG.getConstant(CC, DL, MVT::i32);
    return DAG.getNode(XtensaISD::BR_CC_FP, DL, Op.getValueType(), Chain,
                       TargetCC, LHS, RHS, Dest);
  } else {
    llvm_unreachable("invalid BR_CC to lower");
  }
}

static unsigned getBranchOpcode(ISD::CondCode Cond) {
  switch (Cond) {
  case ISD::SETEQ:
    return Xtensa::BEQ;
  case ISD::SETNE:
    return Xtensa::BNE;
  case ISD::SETLT:
    return Xtensa::BLT;
  case ISD::SETLE:
    return Xtensa::BGE;
  case ISD::SETGT:
    return Xtensa::BLT;
  case ISD::SETGE:
    return Xtensa::BGE;
  case ISD::SETULT:
    return Xtensa::BLTU;
  case ISD::SETULE:
    return Xtensa::BGEU;
  case ISD::SETUGT:
    return Xtensa::BLTU;
  case ISD::SETUGE:
    return Xtensa::BGEU;
  default:
    llvm_unreachable("Unknown branch kind");
  }
}

SDValue XtensaTargetLowering::LowerSELECT_CC(SDValue Op,
                                             SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT Ty = Op.getOperand(0).getValueType();
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  SDValue TrueValue = Op.getOperand(2);
  SDValue FalseValue = Op.getOperand(3);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op->getOperand(4))->get();

  SDValue TargetCC = DAG.getConstant(
      (LHS.getValueType() == MVT::f32) ? CC : getBranchOpcode(CC), DL,
      MVT::i32);

  if (LHS.getValueType() == MVT::f32 || TrueValue.getValueType() == MVT::f32)
    return DAG.getNode(XtensaISD::SELECT_CC_FP, DL, TrueValue.getValueType(),
                       LHS, RHS, TrueValue, FalseValue, TargetCC);
  else if (TrueValue.getValueType().isVector())
    return Op;

  return DAG.getNode(XtensaISD::SELECT_CC, DL, Ty, LHS, RHS, TrueValue,
                     FalseValue, TargetCC);
}

SDValue XtensaTargetLowering::LowerSETCC(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(2))->get();

  SDValue TargetCC = DAG.getConstant(
      (LHS.getValueType() == MVT::f32) ? CC : getBranchOpcode(CC), DL,
      MVT::i32);

  // Check Op SDNode users
  // If there are only CALL/CALLW nodes, don't expand Global Address
  SDNode &OpNode = *Op.getNode();
  bool Val = false;
  for (SDNode::use_iterator UI = OpNode.use_begin(); UI != OpNode.use_end();
       ++UI) {
    SDNode &User = *UI.getUse().getUser();
    unsigned OpCode = User.getOpcode();
    if (OpCode == ISD::BRCOND) {
      Val = true;
      break;
    }
  }

  // SETCC has BRCOND predecessor, return original operation
  if (Val)
    return SDValue();

  // Expand to target SELECT_CC
  SDValue TrueV = DAG.getConstant(1, DL, Op.getValueType());
  SDValue FalseV = DAG.getConstant(0, DL, Op.getValueType());

  if (LHS.getValueType() == MVT::f32 || TrueV.getValueType() == MVT::f32)
    return DAG.getNode(XtensaISD::SELECT_CC_FP, DL, TrueV.getValueType(), LHS,
                       RHS, TrueV, FalseV, TargetCC);
  else if (TrueV.getValueType().isVector())
    return SDValue();
  else
    llvm_unreachable("Unknown SETCC operand type");
}

SDValue XtensaTargetLowering::LowerRETURNADDR(SDValue Op,
                                              SelectionDAG &DAG) const {
  // check the depth
  // TODO: xtensa-gcc can handle this, by navigating through the stack, we
  // should be able to do this too
  assert((cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue() == 0) &&
         "Return address can be determined only for current frame.");

  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  MVT VT = Op.getSimpleValueType();
  unsigned RA = Xtensa::A0;
  MFI.setReturnAddressIsTaken(true);

  // Return RA, which contains the return address. Mark it an implicit
  // live-in.
  unsigned Reg = MF.addLiveIn(RA, getRegClassFor(VT));
  return DAG.getCopyFromReg(DAG.getEntryNode(), SDLoc(Op), Reg, VT);
}

SDValue XtensaTargetLowering::LowerImmediate(SDValue Op,
                                             SelectionDAG &DAG) const {
  const ConstantSDNode *CN = cast<ConstantSDNode>(Op);
  SDLoc DL(CN);
  APInt APVal = CN->getAPIntValue();
  int64_t Value = APVal.getSExtValue();
  if (Op.getValueType() == MVT::i32) {
    // Check if use node maybe lowered to the MOVI instruction
    if (Value > -2048 && Value <= 2047)
      return Op;
    Type *Ty = Type::getInt32Ty(*DAG.getContext());
    Constant *CV = ConstantInt::get(Ty, Value);
    SDValue CP = DAG.getConstantPool(CV, MVT::i32);
    return CP;
  }
  return Op;
}

SDValue XtensaTargetLowering::LowerImmediateFP(SDValue Op,
                                               SelectionDAG &DAG) const {
  const ConstantFPSDNode *CN = cast<ConstantFPSDNode>(Op);
  SDLoc DL(CN);
  APFloat apval = CN->getValueAPF();
  int64_t value = llvm::bit_cast<uint32_t>(CN->getValueAPF().convertToFloat());
  if (Op.getValueType() == MVT::f32) {
    Type *Ty = Type::getInt32Ty(*DAG.getContext());
    Constant *CV = ConstantInt::get(Ty, value);
    SDValue CP = DAG.getConstantPool(CV, MVT::i32);
    return DAG.getNode(ISD::BITCAST, DL, MVT::f32, CP);
  }
  return Op;
}

SDValue XtensaTargetLowering::LowerGlobalAddress(SDValue Op,
                                                 SelectionDAG &DAG) const {
  const GlobalAddressSDNode *G = cast<GlobalAddressSDNode>(Op);
  SDLoc DL(Op);
  auto PtrVT = Op.getValueType();
  const GlobalValue *GV = G->getGlobal();

  SDValue CPAddr = DAG.getTargetConstantPool(GV, PtrVT, Align(4));
  SDValue CPWrap = getAddrPCRel(CPAddr, DAG);

  return CPWrap;
}

SDValue XtensaTargetLowering::LowerGlobalTLSAddress(GlobalAddressSDNode *GA,
                                                    SelectionDAG &DAG) const {
  SDLoc DL(GA);
  const GlobalValue *GV = GA->getGlobal();
  EVT PtrVT = getPointerTy(DAG.getDataLayout());

  if (DAG.getTarget().useEmulatedTLS())
    return LowerToTLSEmulatedModel(GA, DAG);

  TLSModel::Model model = getTargetMachine().getTLSModel(GV);

  if (!Subtarget.hasTHREADPTR()) {
    llvm_unreachable("only emulated TLS supported");
  }

  if ((model == TLSModel::LocalExec) || (model == TLSModel::InitialExec)) {
    auto PtrVt = getPointerTy(DAG.getDataLayout());

    bool Priv = GV->isPrivateLinkage(GV->getLinkage());
    // Create a constant pool entry for the callee address
    XtensaConstantPoolValue *CPV = XtensaConstantPoolSymbol::Create(
        *DAG.getContext(), GV->getName().str().c_str() /* Sym */,
        0 /* XtensaCLabelIndex */, Priv, XtensaCP::TPOFF);

    // Get the address of the callee into a register
    SDValue CPAddr = DAG.getTargetConstantPool(CPV, PtrVt, Align(4));
    SDValue CPWrap = getAddrPCRel(CPAddr, DAG);

    SDValue TPRegister = DAG.getRegister(Xtensa::THREADPTR, MVT::i32);
    SDValue ThreadPointer =
        DAG.getNode(XtensaISD::RUR, DL, MVT::i32, TPRegister);
    return DAG.getNode(ISD::ADD, DL, PtrVT, ThreadPointer, CPWrap);
  } else
    llvm_unreachable("only local-exec and initial-exec TLS mode supported");

  return SDValue();
}

SDValue XtensaTargetLowering::LowerBlockAddress(SDValue Op,
                                                SelectionDAG &DAG) const {
  BlockAddressSDNode *Node = cast<BlockAddressSDNode>(Op);
  const BlockAddress *BA = Node->getBlockAddress();
  EVT PtrVT = Op.getValueType();

  XtensaConstantPoolValue *CPV =
      XtensaConstantPoolConstant::Create(BA, 0, XtensaCP::CPBlockAddress);
  SDValue CPAddr = DAG.getTargetConstantPool(CPV, PtrVT, Align(4));
  SDValue CPWrap = getAddrPCRel(CPAddr, DAG);

  return CPWrap;
}

SDValue XtensaTargetLowering::LowerBR_JT(SDValue Op, SelectionDAG &DAG) const {
  SDValue Chain = Op.getOperand(0);
  SDValue Table = Op.getOperand(1);
  SDValue Index = Op.getOperand(2);
  SDLoc DL(Op);
  JumpTableSDNode *JT = cast<JumpTableSDNode>(Table);
  MachineFunction &MF = DAG.getMachineFunction();
  const MachineJumpTableInfo *MJTI = MF.getJumpTableInfo();
  SDValue TargetJT = DAG.getTargetJumpTable(JT->getIndex(), MVT::i32);
  const DataLayout &TD = DAG.getDataLayout();
  EVT PtrVT = Table.getValueType();
  unsigned EntrySize = MJTI->getEntrySize(TD);

  assert((MJTI->getEntrySize(TD) == 4) && "Unsupported jump-table entry size");

  Index = DAG.getNode(
      ISD::SHL, DL, Index.getValueType(), Index,
      DAG.getConstant(Log2_32(EntrySize), DL, Index.getValueType()));

  SDValue Addr = DAG.getNode(ISD::ADD, DL, Index.getValueType(), Index, Table);
  SDValue LD =
      DAG.getLoad(PtrVT, DL, Chain, Addr,
                  MachinePointerInfo::getJumpTable(DAG.getMachineFunction()));

  return DAG.getNode(XtensaISD::BR_JT, DL, MVT::Other, LD.getValue(1), LD,
                     TargetJT);
}

SDValue XtensaTargetLowering::LowerJumpTable(SDValue Op,
                                             SelectionDAG &DAG) const {
  JumpTableSDNode *JT = cast<JumpTableSDNode>(Op);
  EVT PtrVT = Op.getValueType();

  // Create a constant pool entry for the callee address
  XtensaConstantPoolValue *CPV =
      XtensaConstantPoolJumpTable::Create(*DAG.getContext(), JT->getIndex());

  // Get the address of the callee into a register
  SDValue CPAddr = DAG.getTargetConstantPool(CPV, PtrVT, Align(4));

  return getAddrPCRel(CPAddr, DAG);
}

SDValue XtensaTargetLowering::getAddrPCRel(SDValue Op,
                                           SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT Ty = Op.getValueType();
  return DAG.getNode(XtensaISD::PCREL_WRAPPER, DL, Ty, Op);
}

SDValue XtensaTargetLowering::LowerConstantPool(SDValue Op,
                                                SelectionDAG &DAG) const {
  ConstantPoolSDNode *CP = cast<ConstantPoolSDNode>(Op);
  EVT PtrVT = Op.getValueType();
  auto C = const_cast<Constant *>(CP->getConstVal());
  auto T = const_cast<Type *>(CP->getType());
  SDValue Result;

  // Do not use constant pool for aggregate or vector constant types,
  // in such cases create global variable
  if (T->isAggregateType() || T->isVectorTy()) {
    auto AFI = DAG.getMachineFunction().getInfo<XtensaFunctionInfo>();
    auto M = const_cast<Module *>(
        DAG.getMachineFunction().getFunction().getParent());
    auto GV = new GlobalVariable(
        *M, T, /*isConstant=*/true, GlobalVariable::InternalLinkage, C,
        Twine(DAG.getDataLayout().getPrivateGlobalPrefix()) + "CP" +
            Twine(DAG.getMachineFunction().getFunctionNumber()) + "_" +
            Twine(AFI->createLabelUId()));
    Result = DAG.getTargetConstantPool(GV, PtrVT, Align(4));
  } else {
    if (CP->isMachineConstantPoolEntry())
      Result = DAG.getTargetConstantPool(CP->getMachineCPVal(), PtrVT,
                                         CP->getAlign());
    else
      Result =
          DAG.getTargetConstantPool(C, PtrVT, CP->getAlign(), CP->getOffset());
  }

  return getAddrPCRel(Result, DAG);
}

SDValue XtensaTargetLowering::LowerSTACKSAVE(SDValue Op,
                                             SelectionDAG &DAG) const {
  return DAG.getCopyFromReg(Op.getOperand(0), SDLoc(Op), Xtensa::SP,
                            Op.getValueType());
}

SDValue XtensaTargetLowering::LowerSTACKRESTORE(SDValue Op,
                                                SelectionDAG &DAG) const {
  if (Subtarget.isWinABI()) {
    SDValue NewSP =
        DAG.getNode(XtensaISD::MOVSP, SDLoc(Op), MVT::i32, Op.getOperand(1));
    return DAG.getCopyToReg(Op.getOperand(0), SDLoc(Op), Xtensa::SP, NewSP);
  } else {
    return DAG.getCopyToReg(Op.getOperand(0), SDLoc(Op), Xtensa::SP, Op.getOperand(1));
  }
}

SDValue XtensaTargetLowering::LowerFRAMEADDR(SDValue Op,
                                             SelectionDAG &DAG) const {
  // check the depth
  assert((cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue() == 0) &&
         "Frame address can only be determined for current frame.");

  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = DAG.getMachineFunction().getFrameInfo();
  MFI.setFrameAddressIsTaken(true);
  EVT VT = Op.getValueType();
  SDLoc DL(Op);

  unsigned FrameReg = Subtarget.getRegisterInfo()->getFrameRegister(MF);
  SDValue FrameAddr = DAG.getCopyFromReg(DAG.getEntryNode(), DL, FrameReg, VT);
  return FrameAddr;
}

SDValue XtensaTargetLowering::LowerDYNAMIC_STACKALLOC(SDValue Op,
                                                      SelectionDAG &DAG) const {
  SDValue Chain = Op.getOperand(0); // Legalize the chain.
  SDValue Size = Op.getOperand(1);  // Legalize the size.
  EVT VT = Size->getValueType(0);
  SDLoc DL(Op);

  // Round up Size to 32
  SDValue SizeTmp =
      DAG.getNode(ISD::ADD, DL, VT, Size, DAG.getConstant(31, DL, MVT::i32));
  SDValue SizeRoundUp = DAG.getNode(ISD::AND, DL, VT, SizeTmp,
                                    DAG.getConstant(~31, DL, MVT::i32));

  unsigned SPReg = Xtensa::SP;
  SDValue SP = DAG.getCopyFromReg(Chain, DL, SPReg, VT);
  SDValue NewSP = DAG.getNode(ISD::SUB, DL, VT, SP, SizeRoundUp); // Value
  if (Subtarget.isWinABI()) {
    SDValue NewSP1 = DAG.getNode(XtensaISD::MOVSP, DL, MVT::i32, NewSP);
    Chain = DAG.getCopyToReg(SP.getValue(1), DL, SPReg, NewSP1); // Output chain
  } else {
    Chain = DAG.getCopyToReg(SP.getValue(1), DL, SPReg, NewSP); // Output chain
  }

  SDValue NewVal = DAG.getCopyFromReg(Chain, DL, SPReg, MVT::i32);
  Chain = NewVal.getValue(1);

  SDValue Ops[2] = {NewVal, Chain};
  return DAG.getMergeValues(Ops, DL);
}

SDValue XtensaTargetLowering::LowerVASTART(SDValue Op,
                                           SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  XtensaFunctionInfo *XtensaFI = MF.getInfo<XtensaFunctionInfo>();
  EVT PtrVT = getPointerTy(DAG.getDataLayout());
  SDLoc DL(Op);

  SDValue Chain = Op.getOperand(0);
  SDValue Addr = Op.getOperand(1);

  // typedef struct __va_list_tag {
  //   int32_t *__va_stk; /* Initialized to point  to the position of the
  //                       * first argument in memory offset to account for
  //                       the
  //                       * arguments passed in registers and to account for
  //                       * the size of the argument registers not being
  //                       16-byte
  //                       * aligned.  E.G., there are 6 argument registers
  //                       * of 4 bytes each, but we want the __va_ndx for the
  //                       * first stack argument to have the maximal
  //                       * alignment of 16 bytes, so we offset the __va_stk
  //                       address by
  //                       * 32 bytes so that __va_stk[32] references the
  //                       first
  //                       * argument on the stack.
  //                       */
  //   int32_t  *__va_reg; /* Points to a stack-allocated region holding the
  //                        * contents
  //                        * of the incoming argument registers
  //                        */
  //   int32_t __va_ndx;   /* Index initialized to the position of the first
  //                        * unnamed (variable) argument.  This same index is
  //                        also
  //                        * used to address the arguments passed in memory.
  //                       */
  //  } __va_list_tag[1];

  SDValue ArgAR;
  SDValue OverflowPtrAdvance;
  SDValue StackOffsetFI =
      DAG.getFrameIndex(XtensaFI->getVarArgsStackOffset(), PtrVT);

  if (XtensaFI->getVarArgsFirstGPR() < 8) {
    ArgAR =
        DAG.getConstant(XtensaFI->getVarArgsFirstGPR() * 4 - 8, DL, MVT::i32);
    OverflowPtrAdvance = DAG.getConstant(32, DL, PtrVT);
  } else {
    OverflowPtrAdvance = DAG.getNode(ISD::AND, DL, PtrVT, StackOffsetFI,
                                     DAG.getConstant(0xf, DL, PtrVT));
    OverflowPtrAdvance = DAG.getNode(ISD::ADD, DL, PtrVT, OverflowPtrAdvance,
                                     DAG.getConstant(32, DL, PtrVT));
    ArgAR = OverflowPtrAdvance;
  }

  SDValue FR = DAG.getFrameIndex(XtensaFI->getVarArgsFrameIndex(), PtrVT);

  uint64_t FrameOffset = PtrVT.getSizeInBits() / 8;
  SDValue ConstFrameOffset1 = DAG.getConstant(FrameOffset, DL, PtrVT);
  SDValue ConstFrameOffset2 = DAG.getConstant(FrameOffset * 2, DL, PtrVT);

  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();

  // Store first word : arguments given in stack  (__va_stk)
  // Advance Argument Overflow pointer down, lest it will point to start
  // after register argument va_arg finished
  SDValue StackOffsetFICorr =
      DAG.getNode(ISD::SUB, DL, PtrVT, StackOffsetFI, OverflowPtrAdvance);
  SDValue firstStore =
      DAG.getStore(Chain, DL, StackOffsetFICorr, Addr, MachinePointerInfo(SV));

  uint64_t nextOffset = FrameOffset;
  SDValue nextPtr = DAG.getNode(ISD::ADD, DL, PtrVT, Addr, ConstFrameOffset1);

  // Store second word : arguments given on registers  (__va_reg)
  SDValue FRAdvance =
      DAG.getConstant(XtensaFI->getVarArgsFirstGPR() * 4 - 8, DL, PtrVT);
  SDValue FRDecr = DAG.getNode(ISD::SUB, DL, PtrVT, FR, FRAdvance);
  SDValue secondStore = DAG.getStore(firstStore, DL, FRDecr, nextPtr,
                                     MachinePointerInfo(SV, nextOffset));
  nextOffset += FrameOffset;
  nextPtr = DAG.getNode(ISD::ADD, DL, PtrVT, Addr, ConstFrameOffset2);

  // Store first word : number of int regs  (__va_ndx)
  return DAG.getStore(secondStore, DL, ArgAR, nextPtr,
                      MachinePointerInfo(SV, nextOffset));
}

SDValue XtensaTargetLowering::LowerVACOPY(SDValue Op, SelectionDAG &DAG) const {
  // We have to copy the entire va_list struct:
  // 2*sizeof(int*) + sizeof(int) = 12 Byte
  unsigned VAListSize = 12;
  return DAG.getMemcpy(
      Op.getOperand(0), Op, Op.getOperand(1), Op.getOperand(2),
      DAG.getConstant(VAListSize, SDLoc(Op), MVT::i32), Align(8),
      /*isVolatile=*/false, /*AlwaysInline=*/true,
      /*CI=*/nullptr, std::nullopt, MachinePointerInfo(), MachinePointerInfo());
}

SDValue XtensaTargetLowering::LowerShiftLeftParts(SDValue Op,
                                                  SelectionDAG &DAG) const {
  SDLoc DL(Op);
  MVT VT = MVT::i32;
  SDValue Lo = Op.getOperand(0), Hi = Op.getOperand(1);
  SDValue Shamt = Op.getOperand(2);

  // if Shamt - register size < 0: // Shamt < register size
  //   Lo = Lo << Shamt
  //   Hi = (Hi << Shamt) | (Lo >>u (register size - Shamt))
  // else:
  //   Lo = 0
  //   Hi = Lo << (Shamt - register size)

  SDValue MinusRegisterSize = DAG.getConstant(-32, DL, VT);
  SDValue ShamtMinusRegisterSize =
      DAG.getNode(ISD::ADD, DL, VT, Shamt, MinusRegisterSize);

  SDValue LoTrue = DAG.getNode(ISD::SHL, DL, VT, Lo, Shamt);
  SDValue HiTrue = DAG.getNode(XtensaISD::SRCL, DL, VT, Hi, Lo, Shamt);
  SDValue Zero = DAG.getConstant(0, DL, VT);
  SDValue HiFalse = DAG.getNode(ISD::SHL, DL, VT, Lo, ShamtMinusRegisterSize);

  SDValue Cond = DAG.getSetCC(DL, VT, ShamtMinusRegisterSize, Zero, ISD::SETLT);
  Lo = DAG.getNode(ISD::SELECT, DL, VT, Cond, LoTrue, Zero);
  Hi = DAG.getNode(ISD::SELECT, DL, VT, Cond, HiTrue, HiFalse);

  return DAG.getMergeValues({Lo, Hi}, DL);
}

SDValue XtensaTargetLowering::LowerShiftRightParts(SDValue Op,
                                                   SelectionDAG &DAG,
                                                   bool IsSRA) const {
  SDLoc DL(Op);
  SDValue Lo = Op.getOperand(0), Hi = Op.getOperand(1);
  SDValue Shamt = Op.getOperand(2);
  MVT VT = MVT::i32;

  // SRA expansion:
  //   if Shamt - register size < 0: // Shamt < register size
  //     Lo = (Lo >>u Shamt) | (Hi << u (register size - Shamt))
  //     Hi = Hi >>s Shamt
  //   else:
  //     Lo = Hi >>s (Shamt - register size);
  //     Hi = Hi >>s (register size - 1)
  //
  // SRL expansion:
  //   if Shamt - register size < 0: // Shamt < register size
  //     Lo = (Lo >>u Shamt) | (Hi << u (register size - Shamt))
  //     Hi = Hi >>u Shamt
  //   else:
  //     Lo = Hi >>u (Shamt - register size);
  //     Hi = 0;

  unsigned ShiftRightOp = IsSRA ? ISD::SRA : ISD::SRL;
  SDValue MinusRegisterSize = DAG.getConstant(-32, DL, VT);
  SDValue RegisterSizeMinus1 = DAG.getConstant(32 - 1, DL, VT);
  SDValue ShamtMinusRegisterSize =
      DAG.getNode(ISD::ADD, DL, VT, Shamt, MinusRegisterSize);

  SDValue LoTrue = DAG.getNode(XtensaISD::SRCR, DL, VT, Hi, Lo, Shamt);
  SDValue HiTrue = DAG.getNode(ShiftRightOp, DL, VT, Hi, Shamt);
  SDValue Zero = DAG.getConstant(0, DL, VT);
  SDValue LoFalse =
      DAG.getNode(ShiftRightOp, DL, VT, Hi, ShamtMinusRegisterSize);
  SDValue HiFalse;

  if (IsSRA) {
    HiFalse = DAG.getNode(ShiftRightOp, DL, VT, Hi, RegisterSizeMinus1);
  } else {
    HiFalse = Zero;
  }

  SDValue Cond = DAG.getSetCC(DL, VT, ShamtMinusRegisterSize, Zero, ISD::SETLT);
  Lo = DAG.getNode(ISD::SELECT, DL, VT, Cond, LoTrue, LoFalse);
  Hi = DAG.getNode(ISD::SELECT, DL, VT, Cond, HiTrue, HiFalse);

  return DAG.getMergeValues({Lo, Hi}, DL);
}

SDValue XtensaTargetLowering::LowerCTPOP(SDValue Op, SelectionDAG &DAG) const {
  auto &TLI = DAG.getTargetLoweringInfo();
  return TLI.expandCTPOP(Op.getNode(), DAG);
}

bool XtensaTargetLowering::decomposeMulByConstant(LLVMContext &Context, EVT VT,
                                                  SDValue C) const {
  APInt Imm;
  unsigned EltSizeInBits;

  if (ISD::isConstantSplatVector(C.getNode(), Imm)) {
    EltSizeInBits = VT.getScalarSizeInBits();
  } else if (VT.isScalarInteger()) {
    EltSizeInBits = VT.getSizeInBits();
    if (auto *ConstNode = dyn_cast<ConstantSDNode>(C.getNode()))
      Imm = ConstNode->getAPIntValue();
    else
      return false;
  } else {
    return false;
  }

  // Omit if data size exceeds.
  if (EltSizeInBits > 32)
    return false;

  // Convert MULT to LSL.
  if (Imm.isPowerOf2() && Imm.isIntN(5))
    return true;

  return false;
}

SDValue XtensaTargetLowering::LowerFunnelShift(SDValue Op,
                                               SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue Op0 = Op.getOperand(0);
  SDValue Op1 = Op.getOperand(1);
  SDValue Shamt = Op.getOperand(2);
  MVT VT = Op.getSimpleValueType();

  bool IsFSHR = Op.getOpcode() == ISD::FSHR;
  assert((VT == MVT::i32) && "Unexpected funnel shift type!");

  return DAG.getNode(IsFSHR ? XtensaISD::SRCR : XtensaISD::SRCL, DL, VT, Op0,
                     Op1, Shamt);
}

SDValue XtensaTargetLowering::LowerATOMIC_FENCE(SDValue Op,
                                                SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  return DAG.getNode(XtensaISD::MEMW, DL, MVT::Other, Chain);
}

SDValue XtensaTargetLowering::LowerBitVecLOAD(SDValue Op,
                                              SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  assert(VT.isVector() && VT.getSizeInBits() <= 8);
  return SDValue(); // Expand
}

SDValue XtensaTargetLowering::LowerOperation(SDValue Op,
                                             SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  case ISD::BR_CC:
    return LowerBR_CC(Op, DAG);
  case ISD::BR_JT:
    return LowerBR_JT(Op, DAG);
  case ISD::Constant:
    return LowerImmediate(Op, DAG);
  case ISD::ConstantFP:
    return LowerImmediateFP(Op, DAG);
  case ISD::RETURNADDR:
    return LowerRETURNADDR(Op, DAG);
  case ISD::GlobalAddress:
    return LowerGlobalAddress(Op, DAG);
  case ISD::GlobalTLSAddress:
    return LowerGlobalTLSAddress(cast<GlobalAddressSDNode>(Op), DAG);
  case ISD::BlockAddress:
    return LowerBlockAddress(Op, DAG);
  case ISD::JumpTable:
    return LowerJumpTable(Op, DAG);
  case ISD::CTPOP:
    return LowerCTPOP(Op, DAG);
  case ISD::ConstantPool:
    return LowerConstantPool(Op, DAG);
  case ISD::SELECT_CC:
    return LowerSELECT_CC(Op, DAG);
  case ISD::SETCC:
    return LowerSETCC(Op, DAG);
  case ISD::STACKSAVE:
    return LowerSTACKSAVE(Op, DAG);
  case ISD::STACKRESTORE:
    return LowerSTACKRESTORE(Op, DAG);
  case ISD::FRAMEADDR:
    return LowerFRAMEADDR(Op, DAG);
  case ISD::DYNAMIC_STACKALLOC:
    return LowerDYNAMIC_STACKALLOC(Op, DAG);
  case ISD::VASTART:
    return LowerVASTART(Op, DAG);
  case ISD::VACOPY:
    return LowerVACOPY(Op, DAG);
  case ISD::ATOMIC_FENCE:
    return LowerATOMIC_FENCE(Op, DAG);
  case ISD::SHL_PARTS:
    return LowerShiftLeftParts(Op, DAG);
  case ISD::SRA_PARTS:
    return LowerShiftRightParts(Op, DAG, true);
  case ISD::SRL_PARTS:
    return LowerShiftRightParts(Op, DAG, false);
  case ISD::FSHL:
  case ISD::FSHR:
    return LowerFunnelShift(Op, DAG);
  case ISD::BITCAST:
    return LowerBITCAST(Op, DAG);
  default:
    report_fatal_error("Unexpected node to lower");
  }
}

const char *XtensaTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  case XtensaISD::BR_JT:
    return "XtensaISD::BR_JT";
  case XtensaISD::CALL:
    return "XtensaISD::CALL";
  case XtensaISD::CALLW:
    return "XtensaISD::CALLW";
  case XtensaISD::EXTUI:
    return "XtensaISD::EXTUI";
  case XtensaISD::MOVSP:
    return "XtensaISD::MOVSP";
  case XtensaISD::PCREL_WRAPPER:
    return "XtensaISD::PCREL_WRAPPER";
  case XtensaISD::RET:
    return "XtensaISD::RET";
  case XtensaISD::RETW:
    return "XtensaISD::RETW";
  case XtensaISD::SELECT_CC:
    return "XtensaISD::SELECT_CC";
  case XtensaISD::SELECT_CC_FP:
    return "XtensaISD::SELECT_CC_FP";
  case XtensaISD::BR_T:
    return "XtensaISD::BR_T";
  case XtensaISD::BR_F:
    return "XtensaISD::BR_F";
  case XtensaISD::BR_CC_FP:
    return "XtensaISD::BR_CC_FP";
  case XtensaISD::SRCL:
    return "XtensaISD::SRCL";
  case XtensaISD::SRCR:
    return "XtensaISD::SRCR";
  case XtensaISD::CMPUO:
    return "XtensaISD::CMPUO";
  case XtensaISD::CMPUEQ:
    return "XtensaISD::CMPUEQ";
  case XtensaISD::CMPULE:
    return "XtensaISD::CMPULE";
  case XtensaISD::CMPULT:
    return "XtensaISD::CMPULT";
  case XtensaISD::CMPOEQ:
    return "XtensaISD::CMPOEQ";
  case XtensaISD::CMPOLE:
    return "XtensaISD::CMPOLE";
  case XtensaISD::CMPOLT:
    return "XtensaISD::CMPOLT";
  case XtensaISD::LOOPBR:
    return "XtensaISD::LOOPBR";
  case XtensaISD::LOOPDEC:
    return "XtensaISD::LOOPDEC";
  case XtensaISD::LOOPEND:
    return "XtensaISD::LOOPEND";
  case XtensaISD::MADD:
    return "XtensaISD::MADD";
  case XtensaISD::MSUB:
    return "XtensaISD::MSUB";
  case XtensaISD::MOVS:
    return "XtensaISD::MOVS";
  case XtensaISD::MEMW:
    return "XtensaISD::MEMW";
  case XtensaISD::TRUNC:
    return "XtensaISD::TRUNC";
  case XtensaISD::BUILD_VEC:
    return "XtensaISD::BUILD_VEC";
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Custom insertion
//===----------------------------------------------------------------------===//

static void GetFPBranchKind(int Cond, int &BrKind, int &CmpKind) {
  switch (Cond) {
  default:
    llvm_unreachable("Invalid condition!");
    break;
  case ISD::SETUNE:
    BrKind = Xtensa::BF;
    CmpKind = Xtensa::OEQ_S;
    break;
  case ISD::SETUO:
    BrKind = Xtensa::BT;
    CmpKind = Xtensa::UN_S;
    break;
  case ISD::SETO:
    BrKind = Xtensa::BF;
    CmpKind = Xtensa::UN_S;
    break;
  case ISD::SETUEQ:
    BrKind = Xtensa::BT;
    CmpKind = Xtensa::UEQ_S;
    break;
  case ISD::SETULE:
    BrKind = Xtensa::BT;
    CmpKind = Xtensa::ULE_S;
    break;
  case ISD::SETULT:
    BrKind = Xtensa::BT;
    CmpKind = Xtensa::ULT_S;
    break;
  case ISD::SETEQ:
  case ISD::SETOEQ:
    BrKind = Xtensa::BT;
    CmpKind = Xtensa::OEQ_S;
    break;
  case ISD::SETNE:
    BrKind = Xtensa::BF;
    CmpKind = Xtensa::OEQ_S;
    break;
  case ISD::SETLE:
  case ISD::SETOLE:
    BrKind = Xtensa::BT;
    CmpKind = Xtensa::OLE_S;
    break;
  case ISD::SETLT:
  case ISD::SETOLT:
    BrKind = Xtensa::BT;
    CmpKind = Xtensa::OLT_S;
    break;
  case ISD::SETGE:
    BrKind = Xtensa::BF;
    CmpKind = Xtensa::OLT_S;
    break;
  case ISD::SETGT:
    BrKind = Xtensa::BF;
    CmpKind = Xtensa::OLE_S;
    break;
  }
}

MachineBasicBlock *
XtensaTargetLowering::emitSelectCC(MachineInstr &MI,
                                   MachineBasicBlock *MBB) const {
  const TargetInstrInfo &TII = *Subtarget.getInstrInfo();
  DebugLoc DL = MI.getDebugLoc();

  MachineOperand &LHS = MI.getOperand(1);
  MachineOperand &RHS = MI.getOperand(2);
  MachineOperand &TrueValue = MI.getOperand(3);
  MachineOperand &FalseValue = MI.getOperand(4);
  unsigned Cond = MI.getOperand(5).getImm();

  // To "insert" a SELECT_CC instruction, we actually have to insert
  // CopyMBB and SinkMBB  blocks and add branch to MBB. We build phi
  // operation in SinkMBB like phi (TrueVakue,FalseValue), where TrueValue
  // is passed from MMB and FalseValue is passed from CopyMBB.
  //   MBB
  //   |   \
  //   |   CopyMBB
  //   |   /
  //   SinkMBB
  // The incoming instruction knows the
  // destination vreg to set, the condition code register to branch on, the
  // true/false values to select between, and a branch opcode to use.
  const BasicBlock *LLVM_BB = MBB->getBasicBlock();
  MachineFunction::iterator It = ++MBB->getIterator();

  MachineFunction *F = MBB->getParent();
  MachineBasicBlock *CopyMBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *SinkMBB = F->CreateMachineBasicBlock(LLVM_BB);

  F->insert(It, CopyMBB);
  F->insert(It, SinkMBB);

  // Transfer the remainder of MBB and its successor edges to SinkMBB.
  SinkMBB->splice(SinkMBB->begin(), MBB,
                  std::next(MachineBasicBlock::iterator(MI)), MBB->end());
  SinkMBB->transferSuccessorsAndUpdatePHIs(MBB);

  MBB->addSuccessor(CopyMBB);
  MBB->addSuccessor(SinkMBB);

  if ((MI.getOpcode() == Xtensa::SELECT_CC_FP_FP) ||
      (MI.getOpcode() == Xtensa::SELECT_CC_FP_INT)) {
    int BrKind = 0;
    int CmpKind = 0;
    unsigned b = Xtensa::B0;


    GetFPBranchKind(Cond, BrKind, CmpKind);
    BuildMI(MBB, DL, TII.get(CmpKind), b)        .addReg(LHS.getReg())
        .addReg(RHS.getReg());
    BuildMI(MBB, DL, TII.get(BrKind))
        .addReg(b, RegState::Kill)
        .addMBB(SinkMBB);
  } else {
    BuildMI(MBB, DL, TII.get(Cond))
        .addReg(LHS.getReg())
        .addReg(RHS.getReg())
        .addMBB(SinkMBB);
  }

  CopyMBB->addSuccessor(SinkMBB);

  //  SinkMBB:
  //   %Result = phi [ %FalseValue, CopyMBB ], [ %TrueValue, MBB ]
  //  ...

  BuildMI(*SinkMBB, SinkMBB->begin(), DL, TII.get(Xtensa::PHI),
          MI.getOperand(0).getReg())
      .addReg(FalseValue.getReg())
      .addMBB(CopyMBB)
      .addReg(TrueValue.getReg())
      .addMBB(MBB);

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return SinkMBB;
}

// Emit instructions for atomic_cmp_swap node for 8/16 bit operands
MachineBasicBlock *
XtensaTargetLowering::emitAtomicCmpSwap(MachineInstr &MI, MachineBasicBlock *BB,
                                        int isByteOperand) const {
  const TargetInstrInfo &TII = *Subtarget.getInstrInfo();
  DebugLoc DL = MI.getDebugLoc();

  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction::iterator It = ++BB->getIterator();

  MachineBasicBlock *thisBB = BB;
  MachineFunction *F = BB->getParent();
  MachineBasicBlock *BBLoop = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *BBExit = F->CreateMachineBasicBlock(LLVM_BB);

  F->insert(It, BBLoop);
  F->insert(It, BBExit);

  // Transfer the remainder of BB and its successor edges to BBExit.
  BBExit->splice(BBExit->begin(), BB,
                 std::next(MachineBasicBlock::iterator(MI)), BB->end());
  BBExit->transferSuccessorsAndUpdatePHIs(BB);

  BB->addSuccessor(BBLoop);

  MachineOperand &Res = MI.getOperand(0);
  MachineOperand &AtomValAddr = MI.getOperand(1);
  MachineOperand &CmpVal = MI.getOperand(2);
  MachineOperand &SwpVal = MI.getOperand(3);

  MachineFunction *MF = BB->getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  const TargetRegisterClass *RC = getRegClassFor(MVT::i32);

  unsigned R1 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::MOVI), R1).addImm(3);

  unsigned ByteOffs = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::AND), ByteOffs)
      .addReg(R1)
      .addReg(AtomValAddr.getReg());

  unsigned AddrAlign = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::SUB), AddrAlign)
      .addReg(AtomValAddr.getReg())
      .addReg(ByteOffs);

  unsigned BitOffs = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::SLLI), BitOffs)
      .addReg(ByteOffs)
      .addImm(3);

  unsigned Mask1 = MRI.createVirtualRegister(RC);
  if (isByteOperand) {
    BuildMI(*BB, MI, DL, TII.get(Xtensa::MOVI), Mask1).addImm(0xff);
  } else {
    unsigned R2 = MRI.createVirtualRegister(RC);
    BuildMI(*BB, MI, DL, TII.get(Xtensa::MOVI), R2).addImm(1);
    unsigned R3 = MRI.createVirtualRegister(RC);
    BuildMI(*BB, MI, DL, TII.get(Xtensa::SLLI), R3).addReg(R2).addImm(16);
    BuildMI(*BB, MI, DL, TII.get(Xtensa::ADDI), Mask1).addReg(R3).addImm(-1);
  }

  BuildMI(*BB, MI, DL, TII.get(Xtensa::SSL)).addReg(BitOffs);

  unsigned R2 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::MOVI), R2).addImm(-1);

  unsigned Mask2 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::SLL), Mask2).addReg(Mask1);

  unsigned Mask3 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::XOR), Mask3).addReg(Mask2).addReg(R2);

  unsigned R3 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::L32I), R3).addReg(AddrAlign).addImm(0);

  unsigned R4 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::AND), R4).addReg(R3).addReg(Mask3);

  unsigned Cmp1 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::SLL), Cmp1).addReg(CmpVal.getReg());

  unsigned Swp1 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::SLL), Swp1).addReg(SwpVal.getReg());

  BB = BBLoop;

  unsigned MaskPhi = MRI.createVirtualRegister(RC);
  unsigned MaskLoop = MRI.createVirtualRegister(RC);

  BuildMI(*BB, BB->begin(), DL, TII.get(Xtensa::PHI), MaskPhi)
      .addReg(MaskLoop)
      .addMBB(BBLoop)
      .addReg(R4)
      .addMBB(thisBB);

  unsigned Cmp2 = MRI.createVirtualRegister(RC);
  BuildMI(BB, DL, TII.get(Xtensa::OR), Cmp2).addReg(Cmp1).addReg(MaskPhi);

  unsigned Swp2 = MRI.createVirtualRegister(RC);
  BuildMI(BB, DL, TII.get(Xtensa::OR), Swp2).addReg(Swp1).addReg(MaskPhi);

  BuildMI(BB, DL, TII.get(Xtensa::WSR), Xtensa::SCOMPARE1).addReg(Cmp2);

  unsigned Swp3 = MRI.createVirtualRegister(RC);
  BuildMI(BB, DL, TII.get(Xtensa::S32C1I), Swp3)
      .addReg(Swp2)
      .addReg(AddrAlign)
      .addImm(0);

  BuildMI(BB, DL, TII.get(Xtensa::AND), MaskLoop).addReg(Swp3).addReg(Mask3);

  BuildMI(BB, DL, TII.get(Xtensa::BNE))
      .addReg(MaskLoop)
      .addReg(MaskPhi)
      .addMBB(BBLoop);

  BB->addSuccessor(BBLoop);
  BB->addSuccessor(BBExit);

  BB = BBExit;
  auto St = BBExit->begin();

  unsigned R5 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, St, DL, TII.get(Xtensa::SSR)).addReg(BitOffs);

  BuildMI(*BB, St, DL, TII.get(Xtensa::SRL), R5).addReg(Swp3);

  BuildMI(*BB, St, DL, TII.get(Xtensa::AND), Res.getReg())
      .addReg(R5)
      .addReg(Mask1);

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return BB;
}

// Emit instructions for atomic_swap node for 8/16 bit operands
MachineBasicBlock *
XtensaTargetLowering::emitAtomicSwap(MachineInstr &MI, MachineBasicBlock *BB,
                                     int isByteOperand) const {
  const TargetInstrInfo &TII = *Subtarget.getInstrInfo();
  DebugLoc DL = MI.getDebugLoc();

  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction::iterator It = ++BB->getIterator();

  MachineFunction *F = BB->getParent();
  MachineBasicBlock *BBLoop1 = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *BBLoop2 = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *BBLoop3 = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *BBLoop4 = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *BBExit = F->CreateMachineBasicBlock(LLVM_BB);

  F->insert(It, BBLoop1);
  F->insert(It, BBLoop2);
  F->insert(It, BBLoop3);
  F->insert(It, BBLoop4);
  F->insert(It, BBExit);

  // Transfer the remainder of BB and its successor edges to BBExit.
  BBExit->splice(BBExit->begin(), BB,
                 std::next(MachineBasicBlock::iterator(MI)), BB->end());
  BBExit->transferSuccessorsAndUpdatePHIs(BB);

  BB->addSuccessor(BBLoop1);
  BBLoop1->addSuccessor(BBLoop2);
  BBLoop2->addSuccessor(BBLoop3);
  BBLoop2->addSuccessor(BBLoop4);
  BBLoop3->addSuccessor(BBLoop2);
  BBLoop3->addSuccessor(BBLoop4);
  BBLoop4->addSuccessor(BBLoop1);
  BBLoop4->addSuccessor(BBExit);

  MachineOperand &Res = MI.getOperand(0);
  MachineOperand &AtomValAddr = MI.getOperand(1);
  MachineOperand &SwpVal = MI.getOperand(2);

  MachineFunction *MF = BB->getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  const TargetRegisterClass *RC = getRegClassFor(MVT::i32);

  unsigned R1 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::MOVI), R1).addImm(3);

  unsigned ByteOffs = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::AND), ByteOffs)
      .addReg(R1)
      .addReg(AtomValAddr.getReg());

  unsigned AddrAlign = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::SUB), AddrAlign)
      .addReg(AtomValAddr.getReg())
      .addReg(ByteOffs);

  unsigned BitOffs = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::SLLI), BitOffs)
      .addReg(ByteOffs)
      .addImm(3);

  unsigned Mask1 = MRI.createVirtualRegister(RC);
  if (isByteOperand) {
    BuildMI(*BB, MI, DL, TII.get(Xtensa::MOVI), Mask1).addImm(0xff);
  } else {
    unsigned R2 = MRI.createVirtualRegister(RC);
    BuildMI(*BB, MI, DL, TII.get(Xtensa::MOVI), R2).addImm(1);
    unsigned R3 = MRI.createVirtualRegister(RC);
    BuildMI(*BB, MI, DL, TII.get(Xtensa::SLLI), R3).addReg(R2).addImm(16);
    BuildMI(*BB, MI, DL, TII.get(Xtensa::ADDI), Mask1).addReg(R3).addImm(-1);
  }

  BuildMI(*BB, MI, DL, TII.get(Xtensa::SSL)).addReg(BitOffs);

  unsigned R2 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::MOVI), R2).addImm(-1);

  unsigned Mask2 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::SLL), Mask2).addReg(Mask1);

  unsigned Mask3 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::XOR), Mask3).addReg(Mask2).addReg(R2);

  unsigned R3 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::L32I), R3).addReg(AddrAlign).addImm(0);

  unsigned R4 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::AND), R4).addReg(R3).addReg(Mask3);

  unsigned SwpValShifted = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::SLL), SwpValShifted)
      .addReg(SwpVal.getReg());

  unsigned R5 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::L32I), R5).addReg(AddrAlign).addImm(0);

  unsigned AtomVal = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::AND), AtomVal).addReg(R5).addReg(Mask2);

  unsigned AtomValPhi = MRI.createVirtualRegister(RC);
  unsigned AtomValLoop = MRI.createVirtualRegister(RC);

  BuildMI(*BBLoop1, BBLoop1->begin(), DL, TII.get(Xtensa::PHI), AtomValPhi)
      .addReg(AtomValLoop)
      .addMBB(BBLoop4)
      .addReg(AtomVal)
      .addMBB(BB);

  BB = BBLoop1;

  BuildMI(BB, DL, TII.get(Xtensa::MEMW));

  unsigned R6 = MRI.createVirtualRegister(RC);
  BuildMI(BB, DL, TII.get(Xtensa::L32I), R6).addReg(AddrAlign).addImm(0);

  unsigned R7 = MRI.createVirtualRegister(RC);
  BuildMI(BB, DL, TII.get(Xtensa::AND), R7).addReg(R6).addReg(Mask3);

  unsigned MaskPhi = MRI.createVirtualRegister(RC);
  unsigned MaskLoop = MRI.createVirtualRegister(RC);

  BuildMI(*BBLoop2, BBLoop2->begin(), DL, TII.get(Xtensa::PHI), MaskPhi)
      .addReg(MaskLoop)
      .addMBB(BBLoop3)
      .addReg(R7)
      .addMBB(BBLoop1);

  BB = BBLoop2;

  unsigned Swp1 = MRI.createVirtualRegister(RC);
  BuildMI(BB, DL, TII.get(Xtensa::OR), Swp1)
      .addReg(SwpValShifted)
      .addReg(MaskPhi);

  unsigned AtomVal1 = MRI.createVirtualRegister(RC);
  BuildMI(BB, DL, TII.get(Xtensa::OR), AtomVal1)
      .addReg(AtomValPhi)
      .addReg(MaskPhi);

  BuildMI(BB, DL, TII.get(Xtensa::WSR), Xtensa::SCOMPARE1).addReg(AtomVal1);

  unsigned Swp2 = MRI.createVirtualRegister(RC);
  BuildMI(BB, DL, TII.get(Xtensa::S32C1I), Swp2)
      .addReg(Swp1)
      .addReg(AddrAlign)
      .addImm(0);

  BuildMI(BB, DL, TII.get(Xtensa::BEQ))
      .addReg(AtomVal1)
      .addReg(Swp2)
      .addMBB(BBLoop4);

  BB = BBLoop3;

  BuildMI(BB, DL, TII.get(Xtensa::AND), MaskLoop).addReg(Swp2).addReg(Mask3);

  BuildMI(BB, DL, TII.get(Xtensa::BNE))
      .addReg(MaskLoop)
      .addReg(MaskPhi)
      .addMBB(BBLoop2);

  BB = BBLoop4;

  BuildMI(BB, DL, TII.get(Xtensa::AND), AtomValLoop).addReg(Swp2).addReg(Mask2);

  BuildMI(BB, DL, TII.get(Xtensa::BNE))
      .addReg(AtomValLoop)
      .addReg(AtomValPhi)
      .addMBB(BBLoop1);

  BB = BBExit;

  auto St = BB->begin();

  unsigned R8 = MRI.createVirtualRegister(RC);

  BuildMI(*BB, St, DL, TII.get(Xtensa::SSR)).addReg(BitOffs);
  BuildMI(*BB, St, DL, TII.get(Xtensa::SRL), R8).addReg(AtomValLoop);

  if (isByteOperand) {
    BuildMI(*BB, St, DL, TII.get(Xtensa::SEXT), Res.getReg())
        .addReg(R8)
        .addImm(7);
  } else {
    BuildMI(*BB, St, DL, TII.get(Xtensa::SEXT), Res.getReg())
        .addReg(R8)
        .addImm(15);
  }

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return BB;
}

// Emit instructions for atomic_swap node for 32 bit operands
MachineBasicBlock *
XtensaTargetLowering::emitAtomicSwap(MachineInstr &MI,
                                     MachineBasicBlock *BB) const {
  const TargetInstrInfo &TII = *Subtarget.getInstrInfo();
  DebugLoc DL = MI.getDebugLoc();

  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction::iterator It = ++BB->getIterator();

  MachineFunction *F = BB->getParent();
  MachineBasicBlock *BBLoop = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *BBExit = F->CreateMachineBasicBlock(LLVM_BB);

  F->insert(It, BBLoop);
  F->insert(It, BBExit);

  // Transfer the remainder of BB and its successor edges to BBExit.
  BBExit->splice(BBExit->begin(), BB,
                 std::next(MachineBasicBlock::iterator(MI)), BB->end());
  BBExit->transferSuccessorsAndUpdatePHIs(BB);

  BB->addSuccessor(BBLoop);
  BBLoop->addSuccessor(BBLoop);
  BBLoop->addSuccessor(BBExit);

  MachineOperand &Res = MI.getOperand(0);
  MachineOperand &AtomValAddr = MI.getOperand(1);
  MachineOperand &SwpVal = MI.getOperand(2);

  MachineFunction *MF = BB->getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  const TargetRegisterClass *RC = getRegClassFor(MVT::i32);

  BuildMI(*BB, MI, DL, TII.get(Xtensa::MEMW));

  unsigned AtomVal = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::L32I), AtomVal)
      .addReg(AtomValAddr.getReg())
      .addImm(0);

  unsigned AtomValLoop = MRI.createVirtualRegister(RC);

  BuildMI(*BBLoop, BBLoop->begin(), DL, TII.get(Xtensa::PHI), Res.getReg())
      .addReg(AtomValLoop)
      .addMBB(BBLoop)
      .addReg(AtomVal)
      .addMBB(BB);

  BB = BBLoop;

  BuildMI(BB, DL, TII.get(Xtensa::WSR), Xtensa::SCOMPARE1).addReg(Res.getReg());

  BuildMI(BB, DL, TII.get(Xtensa::S32C1I), AtomValLoop)
      .addReg(SwpVal.getReg())
      .addReg(AtomValAddr.getReg())
      .addImm(0);

  BuildMI(BB, DL, TII.get(Xtensa::BNE))
      .addReg(AtomValLoop)
      .addReg(Res.getReg())
      .addMBB(BBLoop);

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return BB;
}

MachineBasicBlock *XtensaTargetLowering::emitAtomicRMW(MachineInstr &MI,
                                                       MachineBasicBlock *BB,
                                                       unsigned Opcode,
                                                       bool inv,
                                                       bool minmax) const {
  const TargetInstrInfo &TII = *Subtarget.getInstrInfo();
  DebugLoc DL = MI.getDebugLoc();

  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction::iterator It = ++BB->getIterator();

  MachineBasicBlock *ThisBB = BB;
  MachineFunction *F = BB->getParent();
  MachineBasicBlock *BBLoop = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *BBExit = F->CreateMachineBasicBlock(LLVM_BB);

  F->insert(It, BBLoop);
  F->insert(It, BBExit);

  // Transfer the remainder of BB and its successor edges to BB2.
  BBExit->splice(BBExit->begin(), BB,
                 std::next(MachineBasicBlock::iterator(MI)), BB->end());
  BBExit->transferSuccessorsAndUpdatePHIs(BB);

  BB->addSuccessor(BBLoop);

  MachineOperand &Res = MI.getOperand(0);
  MachineOperand &AtomicValAddr = MI.getOperand(1);
  MachineOperand &Val = MI.getOperand(2);
  MachineFunction *MF = BB->getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  const TargetRegisterClass *RC = getRegClassFor(MVT::i32);

  unsigned R1 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::L32I), R1)
      .addReg(AtomicValAddr.getReg())
      .addImm(0);

  BB = BBLoop;

  unsigned AtomicValPhi = MRI.createVirtualRegister(RC);
  unsigned AtomicValLoop = MRI.createVirtualRegister(RC);

  BuildMI(*BB, BB->begin(), DL, TII.get(Xtensa::PHI), AtomicValPhi)
      .addReg(AtomicValLoop)
      .addMBB(BBLoop)
      .addReg(R1)
      .addMBB(ThisBB);

  unsigned R2 = MRI.createVirtualRegister(RC);

  if (minmax) {
    MachineBasicBlock *BBLoop1 = F->CreateMachineBasicBlock(LLVM_BB);
    F->insert(It, BBLoop1);
    BB->addSuccessor(BBLoop1);
    MachineBasicBlock *BBLoop2 = F->CreateMachineBasicBlock(LLVM_BB);
    F->insert(It, BBLoop2);
    BB->addSuccessor(BBLoop2);

    BuildMI(BB, DL, TII.get(Opcode))
        .addReg(AtomicValPhi)
        .addReg(Val.getReg())
        .addMBB(BBLoop1);

    unsigned R7 = MRI.createVirtualRegister(RC);
    BuildMI(BB, DL, TII.get(Xtensa::MOV_N), R7).addReg(Val.getReg());

    BB = BBLoop1;
    unsigned R8 = MRI.createVirtualRegister(RC);
    BuildMI(BB, DL, TII.get(Xtensa::MOV_N), R8).addReg(AtomicValPhi);
    BB->addSuccessor(BBLoop2);

    BB = BBLoop2;
    unsigned R9 = MRI.createVirtualRegister(RC);

    BuildMI(*BB, BB->begin(), DL, TII.get(Xtensa::PHI), R9)
        .addReg(R7)
        .addMBB(BBLoop)
        .addReg(R8)
        .addMBB(BBLoop1);
    BuildMI(BB, DL, TII.get(Xtensa::MOV_N), R2).addReg(R9);
  } else {
    BuildMI(BB, DL, TII.get(Opcode), R2)
        .addReg(AtomicValPhi)
        .addReg(Val.getReg());
    if (inv) {
      unsigned Rtmp1 = MRI.createVirtualRegister(RC);
      BuildMI(*BB, MI, DL, TII.get(Xtensa::MOVI), Rtmp1).addImm(-1);
      unsigned Rtmp2 = MRI.createVirtualRegister(RC);
      BuildMI(*BB, MI, DL, TII.get(Xtensa::XOR), Rtmp2)
          .addReg(R2)
          .addReg(Rtmp1);
      R2 = Rtmp2;
    }
  }

  unsigned R4 = MRI.createVirtualRegister(RC);
  BuildMI(BB, DL, TII.get(Xtensa::WSR), Xtensa::SCOMPARE1).addReg(AtomicValPhi);
  BuildMI(BB, DL, TII.get(Xtensa::S32C1I), R4)
      .addReg(R2)
      .addReg(AtomicValAddr.getReg(), getKillRegState(AtomicValAddr.isDead()))
      .addImm(0);

  BuildMI(BB, DL, TII.get(Xtensa::MOV_N), AtomicValLoop).addReg(R4);

  BuildMI(BB, DL, TII.get(Xtensa::BNE))
      .addReg(AtomicValPhi)
      .addReg(R4)
      .addMBB(BBLoop);

  BB->addSuccessor(BBLoop);
  BB->addSuccessor(BBExit);

  BB = BBExit;
  auto st = BBExit->begin();

  BuildMI(*BB, st, DL, TII.get(Xtensa::MOV_N), Res.getReg()).addReg(R4);

  MI.eraseFromParent(); // The pseudo instruction is gone now.

  return BB;
}

MachineBasicBlock *
XtensaTargetLowering::emitAtomicRMW(MachineInstr &MI, MachineBasicBlock *BB,
                                    bool isByteOperand, unsigned Opcode,
                                    bool inv, bool minmax) const {
  const TargetInstrInfo &TII = *Subtarget.getInstrInfo();
  DebugLoc DL = MI.getDebugLoc();

  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction::iterator It = ++BB->getIterator();

  MachineBasicBlock *ThisBB = BB;
  MachineFunction *F = BB->getParent();
  MachineBasicBlock *BBLoop = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *BBExit = F->CreateMachineBasicBlock(LLVM_BB);

  F->insert(It, BBLoop);
  F->insert(It, BBExit);

  // Transfer the remainder of BB and its successor edges to BB2.
  BBExit->splice(BBExit->begin(), BB,
                 std::next(MachineBasicBlock::iterator(MI)), BB->end());
  BBExit->transferSuccessorsAndUpdatePHIs(BB);

  BB->addSuccessor(BBLoop);

  MachineOperand &Res = MI.getOperand(0);
  MachineOperand &AtomValAddr = MI.getOperand(1);
  MachineOperand &Val = MI.getOperand(2);

  MachineFunction *MF = BB->getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  const TargetRegisterClass *RC = getRegClassFor(MVT::i32);

  unsigned R1 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::MOVI), R1).addImm(3);

  unsigned ByteOffs = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::AND), ByteOffs)
      .addReg(R1)
      .addReg(AtomValAddr.getReg());

  unsigned AddrAlign = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::SUB), AddrAlign)
      .addReg(AtomValAddr.getReg())
      .addReg(ByteOffs);

  unsigned BitOffs = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::SLLI), BitOffs)
      .addReg(ByteOffs)
      .addImm(3);

  unsigned Mask1 = MRI.createVirtualRegister(RC);
  if (isByteOperand) {
    BuildMI(*BB, MI, DL, TII.get(Xtensa::MOVI), Mask1).addImm(0xff);
  } else {
    unsigned R2 = MRI.createVirtualRegister(RC);
    BuildMI(*BB, MI, DL, TII.get(Xtensa::MOVI), R2).addImm(1);
    unsigned R3 = MRI.createVirtualRegister(RC);
    BuildMI(*BB, MI, DL, TII.get(Xtensa::SLLI), R3).addReg(R2).addImm(16);
    BuildMI(*BB, MI, DL, TII.get(Xtensa::ADDI), Mask1).addReg(R3).addImm(-1);
  }

  BuildMI(*BB, MI, DL, TII.get(Xtensa::SSL)).addReg(BitOffs);

  unsigned R2 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::MOVI), R2).addImm(-1);

  unsigned Mask2 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::SLL), Mask2).addReg(Mask1);

  unsigned Mask3 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::XOR), Mask3).addReg(Mask2).addReg(R2);

  unsigned R3 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::L32I), R3).addReg(AddrAlign).addImm(0);

  unsigned Val1 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::SLL), Val1).addReg(Val.getReg());

  BB = BBLoop;

  unsigned AtomicValPhi = MRI.createVirtualRegister(RC);
  unsigned AtomicValLoop = MRI.createVirtualRegister(RC);

  BuildMI(*BB, BB->begin(), DL, TII.get(Xtensa::PHI), AtomicValPhi)
      .addReg(AtomicValLoop)
      .addMBB(BBLoop)
      .addReg(R3)
      .addMBB(ThisBB);

  unsigned Swp2;

  if (minmax) {
    MachineBasicBlock *BBLoop1 = F->CreateMachineBasicBlock(LLVM_BB);
    F->insert(It, BBLoop1);
    BB->addSuccessor(BBLoop1);
    MachineBasicBlock *BBLoop2 = F->CreateMachineBasicBlock(LLVM_BB);
    F->insert(It, BBLoop2);
    BB->addSuccessor(BBLoop2);

    unsigned R1 = MRI.createVirtualRegister(RC);
    unsigned R2 = MRI.createVirtualRegister(RC);
    unsigned R3 = MRI.createVirtualRegister(RC);
    unsigned R4 = MRI.createVirtualRegister(RC);

    unsigned R5 = MRI.createVirtualRegister(RC);
    BuildMI(BB, DL, TII.get(Xtensa::AND), R5)
        .addReg(AtomicValPhi)
        .addReg(Mask2);

    BuildMI(BB, DL, TII.get(Xtensa::SRL), R1).addReg(R5);
    BuildMI(BB, DL, TII.get(Xtensa::SRL), R2).addReg(Val1);

    if ((Opcode == Xtensa::BLT) || (Opcode == Xtensa::BGE)) {
      if (isByteOperand) {
        BuildMI(BB, DL, TII.get(Xtensa::SEXT), R3).addReg(R1).addImm(7);
        BuildMI(BB, DL, TII.get(Xtensa::SEXT), R4).addReg(R2).addImm(7);
      } else {
        BuildMI(BB, DL, TII.get(Xtensa::SEXT), R3).addReg(R1).addImm(15);
        BuildMI(BB, DL, TII.get(Xtensa::SEXT), R4).addReg(R2).addImm(15);
      }
    } else {
      R3 = R1;
      R4 = R2;
    }

    BuildMI(BB, DL, TII.get(Opcode)).addReg(R3).addReg(R4).addMBB(BBLoop1);

    unsigned R7 = MRI.createVirtualRegister(RC);
    BuildMI(BB, DL, TII.get(Xtensa::MOV_N), R7).addReg(Val1);

    BB = BBLoop1;
    unsigned R8 = MRI.createVirtualRegister(RC);
    BuildMI(BB, DL, TII.get(Xtensa::MOV_N), R8).addReg(AtomicValPhi);
    BB->addSuccessor(BBLoop2);

    BB = BBLoop2;
    unsigned R9 = MRI.createVirtualRegister(RC);

    BuildMI(*BB, BB->begin(), DL, TII.get(Xtensa::PHI), R9)
        .addReg(R7)
        .addMBB(BBLoop)
        .addReg(R8)
        .addMBB(BBLoop1);

    unsigned R10 = MRI.createVirtualRegister(RC);
    BuildMI(BB, DL, TII.get(Xtensa::AND), R10)
        .addReg(AtomicValPhi)
        .addReg(Mask3);

    unsigned R11 = MRI.createVirtualRegister(RC);
    BuildMI(BB, DL, TII.get(Xtensa::AND), R11).addReg(R9).addReg(Mask2);

    Swp2 = MRI.createVirtualRegister(RC);
    BuildMI(BB, DL, TII.get(Xtensa::OR), Swp2).addReg(R10).addReg(R11);
  } else {
    unsigned R4 = MRI.createVirtualRegister(RC);
    BuildMI(BB, DL, TII.get(Xtensa::AND), R4)
        .addReg(AtomicValPhi)
        .addReg(Mask2);

    unsigned Res1 = MRI.createVirtualRegister(RC);
    BuildMI(BB, DL, TII.get(Opcode), Res1).addReg(R4).addReg(Val1);

    unsigned Swp1 = MRI.createVirtualRegister(RC);
    BuildMI(BB, DL, TII.get(Xtensa::AND), Swp1).addReg(Res1).addReg(Mask2);

    unsigned R5 = MRI.createVirtualRegister(RC);
    BuildMI(BB, DL, TII.get(Xtensa::AND), R5)
        .addReg(AtomicValPhi)
        .addReg(Mask3);

    if (inv) {
      unsigned Rtmp1 = MRI.createVirtualRegister(RC);
      BuildMI(BB, DL, TII.get(Xtensa::XOR), Rtmp1)
          .addReg(AtomicValPhi)
          .addReg(Mask2);
      R5 = Rtmp1;
    }

    Swp2 = MRI.createVirtualRegister(RC);
    BuildMI(BB, DL, TII.get(Xtensa::OR), Swp2).addReg(Swp1).addReg(R5);
  }

  unsigned Swp3 = MRI.createVirtualRegister(RC);
  BuildMI(BB, DL, TII.get(Xtensa::WSR), Xtensa::SCOMPARE1).addReg(AtomicValPhi);
  BuildMI(BB, DL, TII.get(Xtensa::S32C1I), Swp3)
      .addReg(Swp2)
      .addReg(AddrAlign)
      .addImm(0);

  BuildMI(BB, DL, TII.get(Xtensa::MOV_N), AtomicValLoop).addReg(Swp3);

  BuildMI(BB, DL, TII.get(Xtensa::BNE))
      .addReg(Swp3)
      .addReg(AtomicValPhi)
      .addMBB(BBLoop);

  BB->addSuccessor(BBLoop);
  BB->addSuccessor(BBExit);
  BB = BBExit;
  auto St = BBExit->begin();

  unsigned R6 = MRI.createVirtualRegister(RC);

  BuildMI(*BB, St, DL, TII.get(Xtensa::SSR)).addReg(BitOffs);

  BuildMI(*BB, St, DL, TII.get(Xtensa::SRL), R6).addReg(AtomicValLoop);

  BuildMI(*BB, St, DL, TII.get(Xtensa::AND), Res.getReg())
      .addReg(R6)
      .addReg(Mask1);

  MI.eraseFromParent(); // The pseudo instruction is gone now.

  return BB;
}

MachineBasicBlock *XtensaTargetLowering::EmitInstrWithCustomInserter(
    MachineInstr &MI, MachineBasicBlock *MBB) const {
  const TargetInstrInfo &TII = *Subtarget.getInstrInfo();
  MachineFunction *MF = MBB->getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  DebugLoc DL = MI.getDebugLoc();

  switch (MI.getOpcode()) {
  case Xtensa::MULA_DA_LL_LDDEC_P:
  case Xtensa::MULA_DA_LH_LDDEC_P:
  case Xtensa::MULA_DA_HL_LDDEC_P:
  case Xtensa::MULA_DA_HH_LDDEC_P:
  case Xtensa::MULA_DA_LL_LDINC_P:
  case Xtensa::MULA_DA_LH_LDINC_P:
  case Xtensa::MULA_DA_HL_LDINC_P:
  case Xtensa::MULA_DA_HH_LDINC_P: {
    MachineOperand &MW = MI.getOperand(0);
    MachineOperand &S = MI.getOperand(1);
    MachineOperand &MX = MI.getOperand(2);
    MachineOperand &T = MI.getOperand(3);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned Reg1 = MRI.createVirtualRegister(RC);
    unsigned Reg2 = MRI.createVirtualRegister(RC);

    BuildMI(*MBB, MI, DL, TII.get(Xtensa::L32I), Reg1)
        .addReg(S.getReg())
        .addImm(0);

    unsigned Opc;
    switch (MI.getOpcode()) {
    case Xtensa::MULA_DA_LL_LDDEC_P:
      Opc = Xtensa::MULA_DA_LL_LDDEC;
      break;
    case Xtensa::MULA_DA_LH_LDDEC_P:
      Opc = Xtensa::MULA_DA_LH_LDDEC;
      break;
    case Xtensa::MULA_DA_HL_LDDEC_P:
      Opc = Xtensa::MULA_DA_HL_LDDEC;
      break;
    case Xtensa::MULA_DA_HH_LDDEC_P:
      Opc = Xtensa::MULA_DA_HH_LDDEC;
      break;
    case Xtensa::MULA_DA_LL_LDINC_P:
      Opc = Xtensa::MULA_DA_LL_LDINC;
      break;
    case Xtensa::MULA_DA_LH_LDINC_P:
      Opc = Xtensa::MULA_DA_LH_LDINC;
      break;
    case Xtensa::MULA_DA_HL_LDINC_P:
      Opc = Xtensa::MULA_DA_HL_LDINC;
      break;
    case Xtensa::MULA_DA_HH_LDINC_P:
      Opc = Xtensa::MULA_DA_HH_LDINC;
      break;
    }

    unsigned MWVal = MW.getImm();
    assert((MWVal < 4) && "Unexpected value of mula_da*ld* first argument, it "
                          "must be from m0..m3");
    unsigned MXVal = MX.getImm();
    assert((MXVal < 2) && "Unexpected value of mula_da*ld* third "
                          "argument, it must be m0 or m1");

    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::M0 + MWVal, RegState::Define)
        .addReg(Reg2, RegState::Define)
        .addReg(Reg1)
        .addReg(Xtensa::M0 + MXVal)
        .addReg(T.getReg());

    BuildMI(*MBB, MI, DL, TII.get(Xtensa::S32I))
        .addReg(Reg2)
        .addReg(S.getReg())
        .addImm(0);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::MULA_DD_LL_LDDEC_P:
  case Xtensa::MULA_DD_LH_LDDEC_P:
  case Xtensa::MULA_DD_HL_LDDEC_P:
  case Xtensa::MULA_DD_HH_LDDEC_P:
  case Xtensa::MULA_DD_LL_LDINC_P:
  case Xtensa::MULA_DD_LH_LDINC_P:
  case Xtensa::MULA_DD_HL_LDINC_P:
  case Xtensa::MULA_DD_HH_LDINC_P: {
    MachineOperand &MW = MI.getOperand(0);
    MachineOperand &S = MI.getOperand(1);
    MachineOperand &MX = MI.getOperand(2);
    MachineOperand &MY = MI.getOperand(3);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned Reg1 = MRI.createVirtualRegister(RC);
    unsigned Reg2 = MRI.createVirtualRegister(RC);

    BuildMI(*MBB, MI, DL, TII.get(Xtensa::L32I), Reg1)
        .addReg(S.getReg())
        .addImm(0);

    unsigned Opc;
    switch (MI.getOpcode()) {
    case Xtensa::MULA_DD_LL_LDDEC_P:
      Opc = Xtensa::MULA_DD_LL_LDDEC;
      break;
    case Xtensa::MULA_DD_LH_LDDEC_P:
      Opc = Xtensa::MULA_DD_LH_LDDEC;
      break;
    case Xtensa::MULA_DD_HL_LDDEC_P:
      Opc = Xtensa::MULA_DD_HL_LDDEC;
      break;
    case Xtensa::MULA_DD_HH_LDDEC_P:
      Opc = Xtensa::MULA_DD_HH_LDDEC;
      break;
    case Xtensa::MULA_DD_LL_LDINC_P:
      Opc = Xtensa::MULA_DD_LL_LDINC;
      break;
    case Xtensa::MULA_DD_LH_LDINC_P:
      Opc = Xtensa::MULA_DD_LH_LDINC;
      break;
    case Xtensa::MULA_DD_HL_LDINC_P:
      Opc = Xtensa::MULA_DD_HL_LDINC;
      break;
    case Xtensa::MULA_DD_HH_LDINC_P:
      Opc = Xtensa::MULA_DD_HH_LDINC;
      break;
    }

    unsigned MWVal = MW.getImm();
    assert((MWVal < 4) && "Unexpected value of mula_dd*ld* first argument, "
                          "it must be from m0..m3");
    unsigned MXVal = MX.getImm();
    assert((MXVal < 2) && "Unexpected value of mula_dd*ld* third "
                          "argument, it must be m0 or m1");
    unsigned MYVal = MY.getImm();
    assert(((MYVal > 1) && (MYVal < 4)) &&
           "Unexpected value of mula_dd*ld* fourth "
           "argument, it must be m2 or m3");

    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::M0 + MWVal, RegState::Define)
        .addReg(Reg2, RegState::Define)
        .addReg(Reg1)
        .addReg(Xtensa::M0 + MXVal)
        .addReg(Xtensa::M0 + MYVal);

    BuildMI(*MBB, MI, DL, TII.get(Xtensa::S32I))
        .addReg(Reg2)
        .addReg(S.getReg())
        .addImm(0);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::XSR_ACCLO_P:
  case Xtensa::XSR_ACCHI_P:
  case Xtensa::XSR_M0_P:
  case Xtensa::XSR_M1_P:
  case Xtensa::XSR_M2_P:
  case Xtensa::XSR_M3_P: {
    MachineOperand &T = MI.getOperand(0);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned Reg1 = MRI.createVirtualRegister(RC);
    unsigned Reg2 = MRI.createVirtualRegister(RC);

    BuildMI(*MBB, MI, DL, TII.get(Xtensa::L32I), Reg1)
        .addReg(T.getReg())
        .addImm(0);

    unsigned SReg;
    switch (MI.getOpcode()) {
    case Xtensa::XSR_ACCLO_P:
      SReg = Xtensa::ACCLO;
      break;
    case Xtensa::XSR_ACCHI_P:
      SReg = Xtensa::ACCHI;
      break;
    case Xtensa::XSR_M0_P:
      SReg = Xtensa::M0;
      break;
    case Xtensa::XSR_M1_P:
      SReg = Xtensa::M1;
      break;
    case Xtensa::XSR_M2_P:
      SReg = Xtensa::M2;
      break;
    case Xtensa::XSR_M3_P:
      SReg = Xtensa::M3;
      break;
    }

    BuildMI(*MBB, MI, DL, TII.get(Xtensa::XSR))
        .addReg(Reg2, RegState::Define)
        .addReg(SReg, RegState::Define)
        .addReg(Reg1)
        .addReg(SReg);

    BuildMI(*MBB, MI, DL, TII.get(Xtensa::S32I))
        .addReg(Reg2)
        .addReg(T.getReg())
        .addImm(0);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::WSR_ACCLO_P:
  case Xtensa::WSR_ACCHI_P:
  case Xtensa::WSR_M0_P:
  case Xtensa::WSR_M1_P:
  case Xtensa::WSR_M2_P:
  case Xtensa::WSR_M3_P: {
    MachineOperand &T = MI.getOperand(0);

    unsigned SReg;
    switch (MI.getOpcode()) {
    case Xtensa::WSR_ACCLO_P:
      SReg = Xtensa::ACCLO;
      break;
    case Xtensa::WSR_ACCHI_P:
      SReg = Xtensa::ACCHI;
      break;
    case Xtensa::WSR_M0_P:
      SReg = Xtensa::M0;
      break;
    case Xtensa::WSR_M1_P:
      SReg = Xtensa::M1;
      break;
    case Xtensa::WSR_M2_P:
      SReg = Xtensa::M2;
      break;
    case Xtensa::WSR_M3_P:
      SReg = Xtensa::M3;
      break;
    }

    BuildMI(*MBB, MI, DL, TII.get(Xtensa::WSR))
        .addReg(SReg, RegState::Define)
        .addReg(T.getReg());
    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::LDDEC_P:
  case Xtensa::LDINC_P: {
    MachineOperand &MW = MI.getOperand(0);
    MachineOperand &S = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned Reg1 = MRI.createVirtualRegister(RC);
    unsigned Reg2 = MRI.createVirtualRegister(RC);

    BuildMI(*MBB, MI, DL, TII.get(Xtensa::L32I), Reg1)
        .addReg(S.getReg())
        .addImm(0);

    unsigned Opc = Xtensa::LDDEC;

    if (MI.getOpcode() == Xtensa::LDINC_P)
      Opc = Xtensa::LDINC;

    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::M0 + MW.getImm(), RegState::Define)
        .addReg(Reg2, RegState::Define)
        .addReg(Reg1);

    BuildMI(*MBB, MI, DL, TII.get(Xtensa::S32I))
        .addReg(Reg2)
        .addReg(S.getReg())
        .addImm(0);

    MI.eraseFromParent();
    return MBB;
  }

  case Xtensa::BRCC_FP: {
    MachineOperand &Cond = MI.getOperand(0);
    MachineOperand &LHS = MI.getOperand(1);
    MachineOperand &RHS = MI.getOperand(2);
    MachineBasicBlock *TargetBB = MI.getOperand(3).getMBB();
    int BrKind = 0;
    int CmpKind = 0;
    unsigned RegB = Xtensa::B0;

    GetFPBranchKind(Cond.getImm(), BrKind, CmpKind);
    BuildMI(*MBB, MI, DL, TII.get(CmpKind), RegB)
        .addReg(LHS.getReg())
        .addReg(RHS.getReg());
    BuildMI(*MBB, MI, DL, TII.get(BrKind))
        .addReg(RegB, RegState::Kill)
        .addMBB(TargetBB);

    MI.eraseFromParent();
    return MBB;
  }

  case Xtensa::SELECT_CC_FP_FP:
  case Xtensa::SELECT_CC_FP_INT:
  case Xtensa::SELECT_CC_INT_FP:
  case Xtensa::SELECT:
    return emitSelectCC(MI, MBB);
  case Xtensa::L8I_P: {
    MachineOperand &R = MI.getOperand(0);
    MachineOperand &Op1 = MI.getOperand(1);
    MachineOperand &Op2 = MI.getOperand(2);

    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);

    const MachineMemOperand &MMO = **MI.memoperands_begin();
    if (MMO.isVolatile()) {
      BuildMI(*MBB, MI, DL, TII.get(Xtensa::MEMW));
    }

    BuildMI(*MBB, MI, DL, TII.get(Xtensa::L8UI), R1).add(Op1).add(Op2);

    if (Subtarget.hasSEXT()) {
      BuildMI(*MBB, MI, DL, TII.get(Xtensa::SEXT), R.getReg())
          .addReg(R1)
          .addImm(7);
    } else {
      unsigned R2 = MRI.createVirtualRegister(RC);
      BuildMI(*MBB, MI, DL, TII.get(Xtensa::SLLI), R2).addReg(R1).addImm(24);
      BuildMI(*MBB, MI, DL, TII.get(Xtensa::SRAI), R.getReg())
          .addReg(R2)
          .addImm(24);
    }

    MI.eraseFromParent();
    return MBB;
  }

  case Xtensa::ATOMIC_CMP_SWAP_8_P: {
    return emitAtomicCmpSwap(MI, MBB, 1);
  }

  case Xtensa::ATOMIC_CMP_SWAP_16_P: {
    return emitAtomicCmpSwap(MI, MBB, 0);
  }

  case Xtensa::ATOMIC_CMP_SWAP_32_P: {
    MachineOperand &R = MI.getOperand(0);
    MachineOperand &Addr = MI.getOperand(1);
    MachineOperand &Cmp = MI.getOperand(2);
    MachineOperand &Swap = MI.getOperand(3);

    BuildMI(*MBB, MI, DL, TII.get(Xtensa::WSR), Xtensa::SCOMPARE1)
        .addReg(Cmp.getReg());

    BuildMI(*MBB, MI, DL, TII.get(Xtensa::S32C1I), R.getReg())
        .addReg(Swap.getReg())
        .addReg(Addr.getReg())
        .addImm(0);

    MI.eraseFromParent();
    return MBB;
  }

  case Xtensa::ATOMIC_SWAP_8_P: {
    return emitAtomicSwap(MI, MBB, 1);
  }

  case Xtensa::ATOMIC_SWAP_16_P: {
    return emitAtomicSwap(MI, MBB, 0);
  }

  case Xtensa::ATOMIC_SWAP_32_P: {
    return emitAtomicSwap(MI, MBB);
  }

  case Xtensa::ATOMIC_LOAD_ADD_8_P:
    return emitAtomicRMW(MI, MBB, true, Xtensa::ADD, false, false);
  case Xtensa::ATOMIC_LOAD_SUB_8_P:
    return emitAtomicRMW(MI, MBB, true, Xtensa::SUB, false, false);
  case Xtensa::ATOMIC_LOAD_OR_8_P:
    return emitAtomicRMW(MI, MBB, true, Xtensa::OR, false, false);
  case Xtensa::ATOMIC_LOAD_XOR_8_P:
    return emitAtomicRMW(MI, MBB, true, Xtensa::XOR, false, false);
  case Xtensa::ATOMIC_LOAD_AND_8_P:
    return emitAtomicRMW(MI, MBB, true, Xtensa::AND, false, false);
  case Xtensa::ATOMIC_LOAD_NAND_8_P:
    return emitAtomicRMW(MI, MBB, true, Xtensa::AND, true, false);
  case Xtensa::ATOMIC_LOAD_MIN_8_P:
    return emitAtomicRMW(MI, MBB, true, Xtensa::BGE, false, true);
  case Xtensa::ATOMIC_LOAD_MAX_8_P:
    return emitAtomicRMW(MI, MBB, true, Xtensa::BLT, false, true);
  case Xtensa::ATOMIC_LOAD_UMIN_8_P:
    return emitAtomicRMW(MI, MBB, true, Xtensa::BGEU, false, true);
  case Xtensa::ATOMIC_LOAD_UMAX_8_P:
    return emitAtomicRMW(MI, MBB, true, Xtensa::BLTU, false, true);

  case Xtensa::ATOMIC_LOAD_ADD_16_P:
    return emitAtomicRMW(MI, MBB, false, Xtensa::ADD, false, false);
  case Xtensa::ATOMIC_LOAD_SUB_16_P:
    return emitAtomicRMW(MI, MBB, false, Xtensa::SUB, false, false);
  case Xtensa::ATOMIC_LOAD_OR_16_P:
    return emitAtomicRMW(MI, MBB, false, Xtensa::OR, false, false);
  case Xtensa::ATOMIC_LOAD_XOR_16_P:
    return emitAtomicRMW(MI, MBB, false, Xtensa::XOR, false, false);
  case Xtensa::ATOMIC_LOAD_AND_16_P:
    return emitAtomicRMW(MI, MBB, false, Xtensa::AND, false, false);
  case Xtensa::ATOMIC_LOAD_NAND_16_P:
    return emitAtomicRMW(MI, MBB, false, Xtensa::AND, true, false);
  case Xtensa::ATOMIC_LOAD_MIN_16_P:
    return emitAtomicRMW(MI, MBB, false, Xtensa::BGE, false, true);
  case Xtensa::ATOMIC_LOAD_MAX_16_P:
    return emitAtomicRMW(MI, MBB, false, Xtensa::BLT, false, true);
  case Xtensa::ATOMIC_LOAD_UMIN_16_P:
    return emitAtomicRMW(MI, MBB, false, Xtensa::BGEU, false, true);
  case Xtensa::ATOMIC_LOAD_UMAX_16_P:
    return emitAtomicRMW(MI, MBB, false, Xtensa::BLTU, false, true);

  case Xtensa::ATOMIC_LOAD_ADD_32_P:
    return emitAtomicRMW(MI, MBB, Xtensa::ADD, false, false);
  case Xtensa::ATOMIC_LOAD_SUB_32_P:
    return emitAtomicRMW(MI, MBB, Xtensa::SUB, false, false);
  case Xtensa::ATOMIC_LOAD_OR_32_P:
    return emitAtomicRMW(MI, MBB, Xtensa::OR, false, false);
  case Xtensa::ATOMIC_LOAD_XOR_32_P:
    return emitAtomicRMW(MI, MBB, Xtensa::XOR, false, false);
  case Xtensa::ATOMIC_LOAD_AND_32_P:
    return emitAtomicRMW(MI, MBB, Xtensa::AND, false, false);
  case Xtensa::ATOMIC_LOAD_NAND_32_P:
    return emitAtomicRMW(MI, MBB, Xtensa::AND, true, false);
  case Xtensa::ATOMIC_LOAD_MIN_32_P:
    return emitAtomicRMW(MI, MBB, Xtensa::BGE, false, true);
  case Xtensa::ATOMIC_LOAD_MAX_32_P:
    return emitAtomicRMW(MI, MBB, Xtensa::BLT, false, true);
  case Xtensa::ATOMIC_LOAD_UMIN_32_P:
    return emitAtomicRMW(MI, MBB, Xtensa::BGEU, false, true);
  case Xtensa::ATOMIC_LOAD_UMAX_32_P:
    return emitAtomicRMW(MI, MBB, Xtensa::BLTU, false, true);

  case Xtensa::S8I:
  case Xtensa::S16I:
  case Xtensa::S32I:
  case Xtensa::S32I_N:
  case Xtensa::SSI:
  case Xtensa::SSIP:
  case Xtensa::SSX:
  case Xtensa::SSXP:
  case Xtensa::L8UI:
  case Xtensa::L16SI:
  case Xtensa::L16UI:
  case Xtensa::L32I:
  case Xtensa::L32I_N:
  case Xtensa::LSI:
  case Xtensa::LSIP:
  case Xtensa::LSX:
  case Xtensa::LSXP: {
    if (MI.memoperands().size() > 0) {
      const MachineMemOperand &MMO = **MI.memoperands_begin();
      if (MMO.isVolatile()) {
        BuildMI(*MBB, MI, DL, TII.get(Xtensa::MEMW));
      }
    }
    return MBB;
  }
  case Xtensa::MOVBA_P:
  case Xtensa::MOVBA2_P: {
    const TargetRegisterClass *AR = getRegClassFor(MVT::i32);

    Register Dst1 = MRI.createVirtualRegister(AR);
    Register Dst2 = MRI.createVirtualRegister(AR);
    MachineOperand Breg = MI.getOperand(0);
    MachineOperand Src = MI.getOperand(1);

    /*
      MOVBA_P2 Breg, Dst1, Dest2, Src
    */
    unsigned TargetOpcode;
    switch (MI.getOpcode()) {
    case Xtensa::MOVBA_P:
      TargetOpcode = Xtensa::MOVBA_P2;
      break;
    case Xtensa::MOVBA2_P:
      TargetOpcode = Xtensa::MOVBA2_P2;
      break;
    case Xtensa::MOVBA4_P:
      TargetOpcode = Xtensa::MOVBA4_P2;
      break;
    default:
      llvm_unreachable("Unknown opcode");
    }
    BuildMI(*MBB, MI, DL, TII.get(TargetOpcode), Breg.getReg())
        .addReg(Dst1, RegState::Define | RegState::EarlyClobber)
        .addReg(Dst2, RegState::Define | RegState::EarlyClobber)
        .addReg(Src.getReg());

    MI.eraseFromParent();

    return MBB;
  }
  default:
    return EmitDSPInstrWithCustomInserter(MI, MBB, TII, MF, MRI, DL);
    // llvm_unreachable("Unexpected instr type to insert");
  }
}

SDValue XtensaTargetLowering::LowerBITCAST(SDValue Op,
                                           SelectionDAG &DAG) const {
  assert(Op.getValueType().isVector());
  if (Op.getOperand(0).getValueType() == MVT::v8i8)
    return SDValue(); // Expand
  return Op;          // Legal
}
