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
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

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
  struct SimpleSwitchInfo {
    SwitchInst *SI;
    BasicBlock *DefaultBB;
    BasicBlock *ExitBB;
    std::vector<BasicBlock *> CaseBBs;
  };

  enum class AlignmentCombo {
    Dst16Src16,
    Dst16Src8,
    Dst8Src16,
    Dst8Src8,
    ScalarUnalignedConst
  };

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
  bool processDst16Src16Const(MemMoveInst *M, BasicBlock::iterator &BBI,
                              MemmoveKind Kind);
  void generateOptimizedBackwardCopyDst16Src16(IRBuilder<> &Builder, Value *Dst,
                                               Value *Src, uint64_t Size,
                                               Value *DstInt, Value *SrcInt);
  void generateLoopBased128BlockBackwardCopy(IRBuilder<> &Builder,
                                             Value *EndSrc, Value *EndDst,
                                             uint64_t NumBlocks);
  void generateUnrolled128BlockBackwardCopy(IRBuilder<> &Builder, Value *EndSrc,
                                            Value *EndDst, uint64_t NumBlocks);
  void generateLoopDispatcher(
      IRBuilder<> &Builder, Value *InitSrcAddr, Value *InitDstAddr,
      uint64_t NumIterations, const std::string &LoopName,
      std::function<void(IRBuilder<> &, Value *&, Value *&, Value *)>
          BodyGenerator);
  std::pair<Value *, Value *>
  emitBackwardDst16Src16OneBlock_Ptr(IRBuilder<> &Builder, Value *SrcPtr,
                                     Value *DstPtr);
  std::pair<Value *, Value *> createEspVld128IpM(IRBuilder<> &Builder,
                                                 Value *SrcPtr, int Step);
  Value *createEspVst128IpM(IRBuilder<> &Builder, Value *Vec, Value *DstPtr,
                            int Step);
  std::pair<Value *, Value *>
  createEspVld128IpMThenVst128IpM(IRBuilder<> &Builder, Value *SrcPtr,
                                  Value *DstPtr, int Step);
  std::pair<Value *, Value *> createEspVldH64IpM(IRBuilder<> &Builder,
                                                 Value *SrcPtr, int Step);
  std::pair<Value *, Value *> createEspVldL64IpM(IRBuilder<> &Builder,
                                                 Value *SrcPtr, int Step);
  Value *createEspVstH64IpM(IRBuilder<> &Builder, Value *Vec, Value *DstPtr,
                            int Step);
  Value *createEspVstL64IpM(IRBuilder<> &Builder, Value *Vec, Value *DstPtr,
                            int Step);
  CallInst *createOptimizedMemCpy(IRBuilder<> &Builder, Value *Dst, Value *Src,
                                  Value *Size, MaybeAlign DstAlign,
                                  MaybeAlign SrcAlign, bool IsVolatile = false,
                                  const MemMoveInst *OriginalInst = nullptr);
  Function *getCurrentFunction(IRBuilder<> &Builder) const;
  Value *createPtrToIntAddr(IRBuilder<> &Builder, Value *Ptr,
                            const std::string &Name = "");

  void generateOptimizedBackwardCopyDispatcher(
      IRBuilder<> &Builder, Value *Dst, Value *Src, uint64_t Size,
      uint64_t Alignment, // 8 or 16 byte alignment
      std::function<void(IRBuilder<> &, Value *, Value *, uint64_t, uint64_t,
                         uint64_t)>
          UnrollGenerator,
      std::function<void(IRBuilder<> &, Value *, Value *, uint64_t, uint64_t,
                         uint64_t)>
          LoopGenerator);
  void generateUnrolledDispatcher(
      IRBuilder<> &Builder, Value *Dst, Value *Src, uint64_t Size,
      uint64_t Remainder, uint64_t NumBlocks, uint64_t BlockSize,
      int64_t SrcOffsetFromEnd, int64_t DstOffsetFromEnd,
      const std::string &CopyName,
      std::function<void(IRBuilder<> &, Value *&, Value *&, uint64_t)>
          BlockGenerator);
  bool processDst16Src8Const(MemMoveInst *M, BasicBlock::iterator &BBI);
  void generateOptimizedBackwardCopyDst16Src8(IRBuilder<> &Builder, Value *Dst,
                                              Value *Src, uint64_t Size);
  void generateUnrolledBackwardCopyDst16Src8(IRBuilder<> &Builder, Value *Dst,
                                             Value *Src, uint64_t Size,
                                             uint64_t Remainder,
                                             uint64_t Blocks16);
  void generateLoopBackwardCopyDst16Src8(IRBuilder<> &Builder, Value *Dst,
                                         Value *Src, uint64_t Size,
                                         uint64_t Remainder, uint64_t Blocks16);
  void generateLoop128ByteBackwardCopyDst16Src8(IRBuilder<> &Builder,
                                                Value *Dst, Value *Src,
                                                uint64_t Size,
                                                uint64_t Remainder,
                                                uint64_t Blocks128);
  void generateRemaining16ByteBackwardCopyDst16Src8(
      IRBuilder<> &Builder, Value *Dst, Value *Src, uint64_t Size,
      uint64_t Remainder, uint64_t Blocks128, uint64_t Remaining16);
  std::pair<Value *, Value *>
  emitBackwardDst16Src8OneBlock_Ptr(IRBuilder<> &Builder, Value *SrcPtr,
                                    Value *DstPtr);
  std::pair<Value *, Value *>
  emitBackwardDst16Src8OneBlock_I32(IRBuilder<> &Builder, Value *SrcAddrI32,
                                    Value *DstAddrI32);

  bool processDst8Src16Const(MemMoveInst *M, BasicBlock::iterator &BBI);
  void generateOptimizedBackwardCopyDst8Src16(IRBuilder<> &Builder, Value *Dst,
                                              Value *Src, uint64_t Size);
  void generateUnrolledBackwardCopyDst8Src16(IRBuilder<> &Builder, Value *Dst,
                                             Value *Src, uint64_t Size,
                                             uint64_t Remainder,
                                             uint64_t Blocks16);
  void generateLoopBackwardCopyDst8Src16(IRBuilder<> &Builder, Value *Dst,
                                         Value *Src, uint64_t Size,
                                         uint64_t Remainder, uint64_t Blocks16);
  void generateLoop128ByteBackwardCopyDst8Src16(IRBuilder<> &Builder,
                                                Value *Dst, Value *Src,
                                                uint64_t Size,
                                                uint64_t Remainder,
                                                uint64_t Blocks128);
  void generateRemaining16ByteBackwardCopyDst8Src16(
      IRBuilder<> &Builder, Value *Dst, Value *Src, uint64_t Size,
      uint64_t Remainder, uint64_t Blocks128, uint64_t Remaining16);
  std::pair<Value *, Value *>
  emitBackwardDst8Src16OneBlock_Ptr(IRBuilder<> &Builder, Value *SrcPtr,
                                    Value *DstPtr);
  std::pair<Value *, Value *>
  emitBackwardDst8Src16OneBlock_I32(IRBuilder<> &Builder, Value *SrcAddrI32,
                                    Value *DstAddrI32);

  bool processDst8Src8Const(MemMoveInst *M, BasicBlock::iterator &BBI);
  void generateOptimizedBackwardCopyDst8Src8(IRBuilder<> &Builder, Value *Dst,
                                             Value *Src, uint64_t Size);
  void generateUnrolledBackwardCopyDst8Src8(IRBuilder<> &Builder, Value *Dst,
                                            Value *Src, uint64_t Size,
                                            uint64_t Remainder,
                                            uint64_t Blocks16);
  void generateLoopBackwardCopyDst8Src8(IRBuilder<> &Builder, Value *Dst,
                                        Value *Src, uint64_t Size,
                                        uint64_t Remainder, uint64_t Blocks16);
  void generateLoop128ByteBackwardCopyDst8Src8(IRBuilder<> &Builder, Value *Dst,
                                               Value *Src, uint64_t Size,
                                               uint64_t Remainder,
                                               uint64_t Blocks128);
  void generateRemaining16ByteBackwardCopyDst8Src8(
      IRBuilder<> &Builder, Value *Dst, Value *Src, uint64_t Size,
      uint64_t Remainder, uint64_t Blocks128, uint64_t Remaining16);
  std::pair<Value *, Value *>
  emitBackwardDst8Src8OneBlock_Ptr(IRBuilder<> &Builder, Value *SrcPtr,
                                   Value *DstPtr);
  std::pair<Value *, Value *>
  emitBackwardDst8Src8OneBlock_I32(IRBuilder<> &Builder, Value *SrcAddrI32,
                                   Value *DstAddrI32);

  void emitForwardSmallCopyBypassingMemCpyIntrinsic(IRBuilder<> &Builder,
                                                    Value *Dst, Value *Src,
                                                    uint64_t Size);
  bool processDst16SrcUnalignConst(MemMoveInst *M, BasicBlock::iterator &BBI);
  bool processDstUnalignSrc16Const(MemMoveInst *M, BasicBlock::iterator &BBI);
  bool processDst8SrcUnalignConst(MemMoveInst *M, BasicBlock::iterator &BBI);

  SimpleSwitchInfo createSimpleSwitch(IRBuilder<> &Builder, Value *TestValue,
                                      const std::string &Prefix,
                                      unsigned NumCases);
  void generateSimpleBackwardCopy(IRBuilder<> &Builder, Value *Dst, Value *Src,
                                  uint64_t Size);
  void generateSmallMemmoveBackward(IRBuilder<> &Builder, BasicBlock *BB,
                                    Value *Dst, Value *Src, Value *Size,
                                    BasicBlock *EndBB);

  void generateCorrectBackwardCopyDst16Src16(IRBuilder<> &Builder, Value *Dst,
                                             Value *Src, Value *Size32,
                                             MemmoveKind Kind);
  bool processVarMemmoveWithKind(MemMoveInst *M, BasicBlock::iterator &BBI,
                                 MemmoveKind Kind);
  bool processDst16Src16Var(MemMoveInst *M, BasicBlock::iterator &BBI);

  bool handleInstructionDeletion(Instruction *I, BasicBlock::iterator &BBI);
};

} // namespace llvm

#endif
