//=== RISCVEsp32P4MemIntrin.h
//------------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RISCVEsp32P4MemIntrin pass
//
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_RISCVESP32P4MEMINTRIN_H
#define LLVM_TRANSFORMS_UTILS_RISCVESP32P4MEMINTRIN_H

#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/LoopUnrollAnalyzer.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/ADCE.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/DeadStoreElimination.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/Reassociate.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

using namespace llvm;

namespace llvm {
class AAResults;
class AllocaInst;
class BatchAAResults;
class AssumptionCache;
class CallBase;
class CallInst;
class DominatorTree;
class Function;
class Instruction;
class LoadInst;
class MemCpyInst;
class MemMoveInst;
class MemorySSA;
class MemorySSAUpdater;
class MemSetInst;
class PostDominatorTree;
class StoreInst;
class TargetLibraryInfo;
class TypeSize;
class Value;
class RecurrenceDescriptor;
extern cl::opt<bool> EnableRISCVEsp32P4MemIntrin;
class Function;

// Source address alignment types:
// - Src16: 16-byte aligned source address
// - Src8: 16-byte unaligned,8-byte aligned source address
// - SrcUnalign: 16-byte unaligned and 8-byte unaligned source address
enum class SrcAlignment { Src16, Src8, SrcUnalign };

// Destination address alignment types:
// - Dst16: 16-byte aligned destination address
// - Dst8: 16-byte unaligned,8-byte aligned destination address
// - DstUnalign: 16-byte unaligned and 8-byte unaligned destination address
enum class DstAlignment { Dst16, Dst8, DstUnalign };

// Memory operation size types:
// - Var: Variable size
// - Const16: Constant size that is 16-byte aligned
// - Const8: Constant size that is not 16-byte aligned and  8-byte aligned
// - OtherConst: Other constant that is not 16-byte aligned and not 8-byte
// aligned
enum class SizeType { Var, Const16, Const8, OtherConst };

// Memory copy types:
// combine SrcAlignment, DstAlignment, SizeType
enum class MemCpyType {
  Src16_Dst16_Const16,
  Src16_Dst16_Const8,
  Src16_Dst16_OtherConst,
  Src16_Dst16_Var,
  Src16_Dst8_Const16,
  Src16_Dst8_Const8,
  Src16_Dst8_OtherConst,
  Src16_Dst8_Var,
  Src16_DstUnalign_Const16,
  Src16_DstUnalign_Const8,
  Src16_DstUnalign_OtherConst,
  Src16_DstUnalign_Var,
  Src8_Dst16_Const16,
  Src8_Dst16_Const8,
  Src8_Dst16_OtherConst,
  Src8_Dst16_Var,
  Src8_Dst8_Const16,
  Src8_Dst8_Const8,
  Src8_Dst8_OtherConst,
  Src8_Dst8_Var,
  Src8_DstUnalign_Const16,
  Src8_DstUnalign_Const8,
  Src8_DstUnalign_OtherConst,
  Src8_DstUnalign_Var,
  SrcUnalign_Dst16_Const16,
  SrcUnalign_Dst16_Const8,
  SrcUnalign_Dst16_OtherConst,
  SrcUnalign_Dst16_Var,
  SrcUnalign_Dst8_Const16,
  SrcUnalign_Dst8_Const8,
  SrcUnalign_Dst8_OtherConst,
  SrcUnalign_Dst8_Var,
  SrcUnalign_DstUnalign_Const16,
  SrcUnalign_DstUnalign_Const8,
  SrcUnalign_DstUnalign_OtherConst,
  SrcUnalign_DstUnalign_Var,
};

class RISCVEsp32P4MemIntrinBase {
protected:
  // Basic member variables
  TargetLibraryInfo *TLI = nullptr;
  AAResults *AA = nullptr;
  AssumptionCache *AC = nullptr;
  DominatorTree *DT = nullptr;
  PostDominatorTree *PDT = nullptr;
  MemorySSA *MSSA = nullptr;
  MemorySSAUpdater *MSSAU = nullptr;
  Module *module = nullptr;
  uint64_t SrcAlignValue = 0; // Source address alignment value
  uint64_t DstAlignValue = 0; // Destination address alignment value
  uint64_t Len = 0;           // Length of the memory copy array
  Value *SizeValue = nullptr; // Size value of the memory copy array
  bool useExistingHelperFunction(MemCpyInst *M, IRBuilder<> &Builder,
                                 const std::string &FuncName, Value *DstAddr,
                                 Value *SrcAddr, Value *Size);

  bool useExistingHelperFunction(IRBuilder<> &Builder,
                                 const std::string &FuncName, Value *DstAddr,
                                 Value *SrcAddr, Value *Size);

  Function *createMemCpyHelperFunction(IRBuilder<> &Builder,
                                       const std::string &FuncName,
                                       Value *DstAddr, Value *SrcAddr,
                                       Value *Size, bool isInline = true);

  Function *createMemCpyHelperFunction(IRBuilder<> &Builder,
                                       const std::string &FuncName,
                                       Value *DstAddr, Value *SrcAddr,
                                       bool isInline = true);

  Function *createMemCpyHelperFunctionPtr(IRBuilder<> &Builder,
                                          const std::string &FuncName,
                                          Value *DstAddr, Value *SrcAddr,
                                          Value *Size, bool isInline = true);

  Function *createMemCpyHelperFunctionGeneric(IRBuilder<> &Builder,
                                              const std::string &FuncName,
                                              Value *DstAddr, Value *SrcAddr,
                                              Value *Size, bool isInline = true,
                                              bool usePointers = false);
  void createLoopBlocks(Function *F, BasicBlock *&EntryBB,
                        BasicBlock *&ForBodyBB, BasicBlock *&ForCleanupBB);
  void setLoopMetadata(Instruction *TermInst);

  void inline createEspInlineAsm(IRBuilder<> &Builder, Value *Addr, int Index,
                                 const std::string &Prefix,
                                 const std::string &Suffix,
                                 const StringRef &Constraints);

  void preProcessIterateOnFunction(Function &F);
  MemCpyType getMemCpyType(MemCpyInst *M);
  inline bool isDivisibleBy16(uint64_t value) { return (value & 0xF) == 0; }
  inline bool isDivisibleBy8(uint64_t value) { return (value & 0x7) == 0; }
  inline bool isDivisibleBy32(uint64_t value) { return (value & 0x1F) == 0; }
  inline bool isDivisibleBy48(uint64_t value) {
    return (value & 0x0F) == 0 && (value % 3) == 0;
  }
  inline bool isDivisibleBy16ButNot8(uint64_t value) {
    return isDivisibleBy16(value) && !isDivisibleBy8(value);
  }
  inline bool isNotDivisibleBy16Or8(uint64_t value) {
    return !isDivisibleBy8(value);
  }

  void inlineEsp32P4MemCpy();
};

class RISCVEsp32P4MemIntrin : public RISCVEsp32P4MemIntrinBase {
public:
  Value *createEspVld128Ip(IRBuilder<> &Builder, Value *Src, int index);
  Value *createEspVst128Ip(IRBuilder<> &Builder, Value *Dst, int index);
  Value *createEspVldH64Ip(IRBuilder<> &Builder, Value *Src, int index);
  Value *createEspVldL64Ip(IRBuilder<> &Builder, Value *Src, int index);
  Value *createEspVstH64Ip(IRBuilder<> &Builder, Value *Dst, int index);
  Value *createEspVstL64Ip(IRBuilder<> &Builder, Value *Dst, int index);
  Value *createEspLd128UsarIp(IRBuilder<> &Builder, Value *Src, int index);
  Value *createEspSrcQLdIp(IRBuilder<> &Builder, Value *Src, int index0,
                           int index2, int index3, int index4);
  void inline createEspSrcQ(IRBuilder<> &Builder, int index0, int index1,
                            int index2);
  Value *generateLoadInstructions(IRBuilder<> &Builder, Value *SrcAddr,
                                  MemCpyType Type, int index);
  Value *generateStoreInstructions(IRBuilder<> &Builder, Value *DstAddr,
                                   MemCpyType Type, int index);
  void processDataBlock(
      IRBuilder<> &Builder, Value *&SrcAddr, Value *&DstAddr, MemCpyType Type,
      int blockSize); // will store the srcaddr and dstaddr after the loop
};

struct RISCVEsp32P4MemIntrinPass
    : public PassInfoMixin<RISCVEsp32P4MemIntrinPass>,
      public RISCVEsp32P4MemIntrin {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }

  bool runImpl(Function &F, TargetLibraryInfo *TLI_, AAResults *AA_,
               AssumptionCache *AC_, DominatorTree *DT_,
               PostDominatorTree *PDT_, MemorySSA *MSSA_,
               FunctionAnalysisManager &AM);
  bool iterateOnFunction(Function &F);

  bool processMemCpyToSIMD(MemCpyInst *M, BasicBlock::iterator &BI,
                           MemCpyType Type);
  void processMemCpyFrom0To15(MemCpyInst *M, BasicBlock::iterator &BI,
                              MemCpyType Type);

  void processSrc16Dst16From0To15Var(MemCpyInst *M, MemCpyType Type);
  bool processSrc16Dst16From1To15Const(MemCpyInst *M, BasicBlock::iterator &BI);
  void processSrc16Dst8From0To15(MemCpyInst *M, BasicBlock::iterator &BI,
                                 MemCpyType Type);
  void processSrcUnalignDstUnalignFrom0To15Var(MemCpyInst *M,
                                               BasicBlock::iterator &BI,
                                               MemCpyType Type);
  bool processSrcUnalignDst16Const(MemCpyInst *M, BasicBlock::iterator &BI);
  bool processMemCpyConstAlign(MemCpyInst *M, BasicBlock::iterator &BI,
                               bool isSrc16Aligned, bool isDst16Aligned);
  bool processSrc16Dst16Const16(MemCpyType Type, MemCpyInst *M,
                                BasicBlock::iterator &BI);
  bool processSrc16Dst16Const8(MemCpyType Type, MemCpyInst *M,
                               BasicBlock::iterator &BI);
  bool processSrc16Dst16OtherConst(MemCpyInst *M, BasicBlock::iterator &BI);
  bool processSrc16Dst16Var(MemCpyInst *M);
  bool processSrc16Dst8Const16(MemCpyType Type, MemCpyInst *M,
                               BasicBlock::iterator &BI);
  bool processSrc16Dst8Const8(MemCpyType Type, MemCpyInst *M,
                              BasicBlock::iterator &BI);
  bool processSrc16Dst8OtherConst(MemCpyInst *M, BasicBlock::iterator &BI);
  bool processSrc16Dst8Var(MemCpyInst *M);
  bool processSrc8Dst16Var(MemCpyInst *M);
  bool processSrc8Dst8Var(MemCpyInst *M);
  bool processSrc16DstUnalignConst16(MemCpyInst *M, BasicBlock::iterator &BI);
  bool processSrc16DstUnalignConst8(MemCpyInst *M, BasicBlock::iterator &BI);
  bool processSrc16DstUnalignOtherConst(MemCpyInst *M,
                                        BasicBlock::iterator &BI);
  bool processSrc16DstUnalignVar(MemCpyInst *M, BasicBlock::iterator &BI);
  bool processSrc8Dst16Const16(MemCpyType Type, MemCpyInst *M,
                               BasicBlock::iterator &BI);

  bool processSrc8Dst16Const8(MemCpyType Type, MemCpyInst *M,
                              BasicBlock::iterator &BI);
  bool processSrc8Dst16OtherConst(MemCpyInst *M, BasicBlock::iterator &BI);
  bool processSrc8Dst16Var(MemCpyInst *M, BasicBlock::iterator &BI);
  bool processSrc8Dst8Const16(MemCpyType Type, MemCpyInst *M,
                              BasicBlock::iterator &BI);
  bool processSrc8Dst8Const8(MemCpyType Type, MemCpyInst *M,
                             BasicBlock::iterator &BI);
  bool processSrc8Dst8OtherConst(MemCpyInst *M, BasicBlock::iterator &BI);
  bool processSrc8Dst8Var(MemCpyInst *M, BasicBlock::iterator &BI);

  bool processSrcUnalignDst16Const16(MemCpyInst *M, BasicBlock::iterator &BI);

  bool processSrcUnalignDst16ConstDiv48(MemCpyInst *M, BasicBlock::iterator &BI,
                                        uint64_t quotient);
  bool processSrcUnalignDst16ConstMod48From32To47(MemCpyInst *M,
                                                  BasicBlock::iterator &BI,
                                                  uint64_t quotient,
                                                  uint64_t remainder);
  bool processSrcUnalignDst16ConstMod48From16To31(MemCpyInst *M,
                                                  BasicBlock::iterator &BI,
                                                  uint64_t quotient,
                                                  uint64_t remainder);
  bool processSrcUnalignDst16ConstMod48From1To15(MemCpyInst *M,
                                                 BasicBlock::iterator &BI,
                                                 uint64_t quotient,
                                                 uint64_t remainder);
  bool processSrcUnalignDst16Const8(MemCpyInst *M, BasicBlock::iterator &BI);
  bool processSrcUnalignDst16OtherConst(MemCpyInst *M,
                                        BasicBlock::iterator &BI);
  bool processSrcUnalignDst16Var(MemCpyInst *M, BasicBlock::iterator &BI);
  bool processSrcUnalignDst8Const16(MemCpyInst *M, BasicBlock::iterator &BI);
  bool processSrcUnalignDst8Const8(MemCpyInst *M, BasicBlock::iterator &BI);
  bool processSrcUnalignDst8OtherConst(MemCpyInst *M, BasicBlock::iterator &BI);
  bool processSrcUnalignDst8Var(MemCpyInst *M, BasicBlock::iterator &BI);
  bool processSrcUnalignDstUnalignConst(MemCpyInst *M,
                                        BasicBlock::iterator &BI);

  bool processSrcUnalignDstUnalignVar(MemCpyInst *M, BasicBlock::iterator &BI);
  bool processSrcUnalignDst16ConstMod48(MemCpyInst *M, BasicBlock::iterator &BI,
                                        uint64_t quotient, uint64_t remainder,
                                        uint64_t remainderThreshold);

  bool processMemCpyWithAlignment(MemCpyType Type, MemCpyInst *M,
                                  BasicBlock::iterator &BBI,
                                  const std::string &FuncName,
                                  uint64_t blockSize = 128,
                                  uint64_t chunkSize = 16);
  bool processOtherConstAlign(MemCpyInst *M, BasicBlock::iterator &BI,
                              uint64_t dstAlign, uint64_t srcAlign);
  bool processSrcUnalignDst16Common(MemCpyInst *M, BasicBlock::iterator &BBI,
                                    bool isInline);

  bool processMemCpyWithAlignmentVar(MemCpyInst *M, std::string srcdstcase,
                                     unsigned SrcAlign, unsigned DstAlign);
  void processMemCpyVarFrom1To15(IRBuilder<> &Builder,
                                 const std::string &FuncName, Value *Dst,
                                 Value *Src, Value *Size, bool isInline,
                                 MemCpyType Type);

  void processSrc16Dst16From1To15Var(IRBuilder<> &Builder, Value *Dst,
                                     Value *Src, Value *Size, bool isInline,
                                     MemCpyType Type);
  void processSrc16Dst16From1To7Var(IRBuilder<> &Builder, Value *Dst,
                                    Value *Src, Value *Size, bool isInline);

  void processSrcUnalignOrDstUnalignFrom0To15Const(MemCpyInst *M,
                                                   BasicBlock::iterator &BBI);

  void processMemCpyVarFrom1To7(IRBuilder<> &Builder,
                                const std::string &FuncName, Value *Dst,
                                Value *Src, Value *Size, bool isInline);

  bool processFromSrcUnalignDstUnalign1To15Const(MemCpyInst *M,
                                                 BasicBlock::iterator &BBI);

  bool processSrc8DstUnalignConst16(MemCpyInst *M, BasicBlock::iterator &BBI);
  bool processSrc8DstUnalignConst8(MemCpyInst *M, BasicBlock::iterator &BBI);
  bool processSrc8DstUnalignOtherConst(MemCpyInst *M,
                                       BasicBlock::iterator &BBI);
  bool processSrc8DstUnalignVar(MemCpyInst *M, BasicBlock::iterator &BBI);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_RISCVESP32P4MEMINTRIN_H
