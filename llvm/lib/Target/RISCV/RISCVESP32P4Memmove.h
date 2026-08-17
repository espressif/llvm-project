//===- RISCVESP32P4Memmove.h - ESP32-P4 memmove opt pass -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Slice s0b: registration + unaligned-const runtime dispatch (overlap lit).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVESP32P4MEMMOVE_H
#define LLVM_LIB_TARGET_RISCV_RISCVESP32P4MEMMOVE_H

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"

#include <cstdint>
#include <functional>

namespace llvm {

extern cl::opt<bool> EnableRISCVESP32P4Memmove;

struct ESP32P4OptimizationConfig {
  static constexpr uint64_t SIMD_REGISTER_SIZE = 16;
  static constexpr uint64_t PREFERRED_ALIGNMENT = SIMD_REGISTER_SIZE;
  static constexpr uint64_t SECONDARY_ALIGNMENT = 8;
  static constexpr uint64_t SIMPLE_UNROLL_THRESHOLD = 48;
  static constexpr uint64_t MAX_UNROLL_SIZE = 128;

  static constexpr bool shouldSimpleUnroll(uint64_t Size) {
    return Size < SIMPLE_UNROLL_THRESHOLD;
  }
  static constexpr bool isWellAligned(uint64_t Alignment) {
    return Alignment >= SECONDARY_ALIGNMENT;
  }
  static constexpr bool isDivisibleBy16(uint64_t Value) {
    return (Value & (PREFERRED_ALIGNMENT - 1)) == 0;
  }
  static constexpr bool isDivisibleBy8(uint64_t Value) {
    return (Value & (SECONDARY_ALIGNMENT - 1)) == 0;
  }
};

struct RISCVESP32P4MemmovePass : PassInfoMixin<RISCVESP32P4MemmovePass> {
  enum class DstAlignment { Dst16, Dst8, DstUnalign };
  enum class SrcAlignment { Src16, Src8, SrcUnalign };
  enum class SizeType { Var, Const16, Const8, OtherConst };
  enum class MemmoveKind {
    Dst16Src16_Const16,
    Dst16Src16_Const8,
    Dst16Src16_OtherConst,
    Dst16Src16_Var,
    Dst16Src8_Var,
    Dst8Src16_Var,
    Dst8Src8_Var,
    Dst16Src8_Const,
    Dst8Src16_Const,
    Dst8Src8_Const,
    Dst16SrcUnalign_Const,
    Dst16SrcUnalign_Var,
    Dst8SrcUnalign_Const,
    Dst8SrcUnalign_Var,
    DstUnalignSrc16_Const,
    DstUnalignSrc16_Var,
    DstUnalignSrcUnalign_Const,
    DstUnalignSrcUnalign_Var
  };
  enum class AlignmentCombo { ScalarUnalignedConst };

  struct ProcessingConfig {
    uint64_t MinSize;
    std::function<void(IRBuilder<> &, Value *, Value *, uint64_t)>
        BackwardGenerator;
    ProcessingConfig(
        uint64_t MinSize,
        std::function<void(IRBuilder<> &, Value *, Value *, uint64_t)> BackGen)
        : MinSize(MinSize), BackwardGenerator(std::move(BackGen)) {}
  };

  TargetLibraryInfo *TLI = nullptr;
  AAResults *AA = nullptr;
  AssumptionCache *AC = nullptr;
  DominatorTree *DT = nullptr;
  PostDominatorTree *PDT = nullptr;
  MemorySSA *MSSA = nullptr;
  MemorySSAUpdater *MSSAU = nullptr;
  ScalarEvolution *SE = nullptr;
  Module *TheModule = nullptr;
  uint64_t SrcAlignValue = 0;
  uint64_t DstAlignValue = 0;
  uint64_t Len = 0;
  Value *SizeValue = nullptr;
  bool ChangedCFG = false;

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }

  bool runImpl(Function &F, TargetLibraryInfo *TLI_, AAResults *AA_,
               AssumptionCache *AC_, DominatorTree *DT_,
               PostDominatorTree *PDT_, MemorySSA *MSSA_, ScalarEvolution *SE_,
               FunctionAnalysisManager &AM);
  bool iterateOnFunction(Function &F);
  MemmoveKind getMemmoveKind(MemMoveInst *M);
  bool processMemmoveToSIMD(MemMoveInst *M, BasicBlock::iterator &BBI);
  bool processDstUnalignSrcUnalignConst(MemMoveInst *M,
                                        BasicBlock::iterator &BBI);
  bool processDstUnalignConstMemIntrinBypass(MemMoveInst *M,
                                             BasicBlock::iterator &BBI);
  bool processConstantSizeWithAlignment(MemMoveInst *M,
                                        BasicBlock::iterator &BBI,
                                        AlignmentCombo Combo);
  ProcessingConfig getProcessingConfig(AlignmentCombo Combo);
  bool processConstantSizeDispatcher(
      MemMoveInst *M, BasicBlock::iterator &BBI, uint64_t MinSize,
      std::function<void(IRBuilder<> &, Value *, Value *, uint64_t)>
          BackwardGenerator);
  void createRuntimeDispatch(
      MemMoveInst *M, BasicBlock::iterator &BBI, bool IsVarSize,
      std::function<void(IRBuilder<> &, Value *, Value *, Value *)>
          BackwardGenerator,
      std::function<void(IRBuilder<> &, Value *, Value *, Value *)>
          CustomForwardCopy = nullptr);
  void generateByteWiseBackwardCopy(IRBuilder<> &Builder, Value *Dst,
                                    Value *Src, uint64_t Size);
  CallInst *createOptimizedMemMove(IRBuilder<> &Builder, Value *Dst, Value *Src,
                                   Value *Size, MaybeAlign DstAlign,
                                   MaybeAlign SrcAlign, bool IsVolatile,
                                   const MemMoveInst *OriginalInst,
                                   bool NoReprocess);
  bool convertMemmoveToMemcpy(MemMoveInst *M, BasicBlock::iterator &BBI);
  bool handleInstructionDeletion(Instruction *I, BasicBlock::iterator &BBI);
};

} // namespace llvm

#endif
