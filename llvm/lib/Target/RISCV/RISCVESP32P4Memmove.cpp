//===-- RISCVESP32P4Memmove.cpp - ESP32-P4 memmove (overlap slice) --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCVESP32P4Memmove.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "riscv-esp32-p4-memmove"

namespace {
static constexpr char ESP32P4MemmoveNoReprocessMDName[] =
    "riscv.esp32p4.memmove.no_reprocess";

static bool isEsp32P4MemmoveNoReprocess(const MemMoveInst *M) {
  return M->getMetadata(ESP32P4MemmoveNoReprocessMDName) != nullptr;
}

/// A memmove can only be safely replaced with memcpy if we can prove there is
/// no overlap.
static bool canProveNoOverlapDstBeforeSrc(const MemMoveInst *M,
                                          const DataLayout &DL) {
  auto *LenC = dyn_cast<ConstantInt>(M->getLength());
  if (!LenC)
    return false;
  const uint64_t Len = LenC->getZExtValue();
  if (Len == 0)
    return true;
  std::optional<int64_t> Offset =
      M->getDest()->getPointerOffsetFrom(M->getSource(), DL);
  if (!Offset)
    return false;
  if (Len > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return false;
  return *Offset <= -static_cast<int64_t>(Len);
}
} // namespace

cl::opt<bool> llvm::EnableRISCVESP32P4Memmove(
    "riscv-esp32-p4-memmove", cl::init(false),
    cl::desc("Enable ESP32-P4 memmove intrinsics optimization"));

bool RISCVESP32P4MemmovePass::handleInstructionDeletion(
    Instruction *I, BasicBlock::iterator &BBI) {
  // Update iterator to next instruction before deletion
  BBI = std::next(I->getIterator());
  // Delete the instruction
  I->eraseFromParent();
  return true;
}

CallInst *RISCVESP32P4MemmovePass::createOptimizedMemMove(
    IRBuilder<> &Builder, Value *Dst, Value *Src, Value *Size,
    MaybeAlign DstAlign, MaybeAlign SrcAlign, bool IsVolatile,
    const MemMoveInst *OriginalInst, bool NoReprocess) {
  CallInst *MemMoveCall =
      Builder.CreateMemMove(Dst, DstAlign, Src, SrcAlign, Size, IsVolatile);

  if (OriginalInst)
    MemMoveCall->copyMetadata(*OriginalInst);

  if (NoReprocess)
    MemMoveCall->setMetadata(ESP32P4MemmoveNoReprocessMDName,
                             MDNode::get(MemMoveCall->getContext(), {}));

  LLVM_DEBUG(dbgs() << "RISCVESP32P4: Created optimized memmove with "
                    << "dst_align=" << (DstAlign ? DstAlign->value() : 0)
                    << ", src_align=" << (SrcAlign ? SrcAlign->value() : 0)
                    << ", volatile=" << IsVolatile << "\n");

  return MemMoveCall;
}

void RISCVESP32P4MemmovePass::generateByteWiseBackwardCopy(IRBuilder<> &Builder,
                                                           Value *Dst,
                                                           Value *Src,
                                                           uint64_t Size) {
  using Config = ESP32P4OptimizationConfig;
  // Fully unrolling huge byte copies explodes SelectionDAG into a single
  // serialized memory chain and can make llc instruction selection hang or
  // take impractically long. Fall back to one llvm.memmove for large sizes.
  if (Size > Config::MAX_UNROLL_SIZE) {
    (void)createOptimizedMemMove(
        Builder, Dst, Src, Builder.getInt32(static_cast<uint32_t>(Size)),
        MaybeAlign(Align(1)), MaybeAlign(Align(1)), false, nullptr,
        /*NoReprocess=*/true);
    return;
  }

  Value *SrcEnd = Builder.CreateConstInBoundsGEP1_64(Builder.getInt8Ty(), Src,
                                                     Size, "src.end");
  Value *DstEnd = Builder.CreateConstInBoundsGEP1_64(Builder.getInt8Ty(), Dst,
                                                     Size, "dst.end");

  // Pure byte-level backward copy - safe and correct
  for (uint64_t I = 0; I < Size; I++) {
    Value *SrcByte = Builder.CreateConstInBoundsGEP1_64(
        Builder.getInt8Ty(), SrcEnd, -(int64_t)(I + 1), "src.byte");
    Value *DstByte = Builder.CreateConstInBoundsGEP1_64(
        Builder.getInt8Ty(), DstEnd, -(int64_t)(I + 1), "dst.byte");

    Value *Data = Builder.CreateLoad(Builder.getInt8Ty(), SrcByte, "byte.data");
    Builder.CreateStore(Data, DstByte);
  }
}

void RISCVESP32P4MemmovePass::createRuntimeDispatch(
    MemMoveInst *M, BasicBlock::iterator &BBI, bool IsVarSize,
    std::function<void(IRBuilder<> &, Value *, Value *, Value *)>
        BackwardGenerator,
    std::function<void(IRBuilder<> &, Value *, Value *, Value *)>
        CustomForwardCopy) {
  ChangedCFG = true;
  IRBuilder<> Builder(M);
  Value *Dst = M->getRawDest();
  Value *Src = M->getRawSource();
  Value *Size = M->getLength();

  // For constant size, skip zero size check (caller has handled)
  BasicBlock *CurrentBB = Builder.GetInsertBlock();
  BasicBlock *RestBB;

  if (IsVarSize) {
    // Variable size: need runtime zero size check
    Value *IsZero = Builder.CreateICmpEQ(
        Size, ConstantInt::get(Size->getType(), 0), "size.is.zero");

    // Create zero size and non-zero size branches
    BasicBlock *ZeroSizeBB = BasicBlock::Create(
        CurrentBB->getContext(), "zero.size", CurrentBB->getParent());
    BasicBlock *NonZeroSizeBB = BasicBlock::Create(
        CurrentBB->getContext(), "non.zero.size", CurrentBB->getParent());

    RestBB =
        CurrentBB->splitBasicBlock(std::next(M->getIterator()), "memmove.end");
    CurrentBB->getTerminator()->eraseFromParent();

    Builder.SetInsertPoint(CurrentBB);
    Builder.CreateCondBr(IsZero, ZeroSizeBB, NonZeroSizeBB);

    // Zero size branch: directly jump to end
    Builder.SetInsertPoint(ZeroSizeBB);
    Builder.CreateBr(RestBB);

    // Set start point for overlap check
    Builder.SetInsertPoint(NonZeroSizeBB);
  } else {
    // Constant size: directly perform overlap check
    RestBB =
        CurrentBB->splitBasicBlock(std::next(M->getIterator()), "memmove.end");
    CurrentBB->getTerminator()->eraseFromParent();
    Builder.SetInsertPoint(CurrentBB);
  }

  // Runtime overlap check: dst <= src ? (or dst < src, depending on
  // the specific situation)
  Value *DstInt = Builder.CreatePtrToInt(Dst, Builder.getInt32Ty());
  Value *SrcInt = Builder.CreatePtrToInt(Src, Builder.getInt32Ty());

  // Most cases use ULE, but some unaligned cases use ULT
  // Here we use more conservative ULE, caller can adjust according to needs
  Value *NoOverlap = Builder.CreateICmpULE(DstInt, SrcInt, "dst.leq.src");

  // Create forward and backward copy basic blocks
  Function *F = CurrentBB->getParent();
  BasicBlock *ForwardBB =
      BasicBlock::Create(F->getContext(), "forward.copy", F, RestBB);
  BasicBlock *BackwardBB =
      BasicBlock::Create(F->getContext(), "backward.copy", F, RestBB);

  Builder.CreateCondBr(NoOverlap, ForwardBB, BackwardBB);

  // Forward copy when dst <= src (low-to-high). The other branch handles dst >
  // src. Optimized forward lowering may use memcpy; memmove overlap is resolved
  // by this split.
  Builder.SetInsertPoint(ForwardBB);
  if (CustomForwardCopy)
    CustomForwardCopy(Builder, Dst, Src, Size);
  else
    (void)createOptimizedMemMove(Builder, Dst, Src, Size, M->getDestAlign(),
                                 M->getSourceAlign(), M->isVolatile(), M,
                                 /*NoReprocess=*/true);
  Builder.CreateBr(RestBB);

  // Backward copy path: use provided generator
  Builder.SetInsertPoint(BackwardBB);
  BackwardGenerator(Builder, Dst, Src, Size);
  Builder.CreateBr(RestBB);

  // Delete original memmove instruction
  handleInstructionDeletion(M, BBI);
}

bool RISCVESP32P4MemmovePass::processConstantSizeDispatcher(
    MemMoveInst *M, BasicBlock::iterator &BBI, uint64_t MinSize,
    std::function<void(IRBuilder<> &, Value *, Value *, uint64_t)>
        BackwardGenerator) {
  (void)MinSize;

  // Read length from the intrinsic; do not rely on member Len from
  // getMemmoveKind().
  auto *LenC = cast<ConstantInt>(M->getLength());
  if (LenC->isZero()) {
    return handleInstructionDeletion(M, BBI);
  }

  createRuntimeDispatch(
      M, BBI, false, // IsVarSize = false
      [this, BackwardGenerator](IRBuilder<> &Builder, Value *Dst, Value *Src,
                                Value *Size) {
        uint64_t constSize = cast<ConstantInt>(Size)->getZExtValue();
        BackwardGenerator(Builder, Dst, Src, constSize);
      });

  return true;
}

bool RISCVESP32P4MemmovePass::processConstantSizeWithAlignment(
    MemMoveInst *M, BasicBlock::iterator &BBI, AlignmentCombo Combo) {
  ProcessingConfig config = getProcessingConfig(Combo);

  return processConstantSizeDispatcher(M, BBI, config.MinSize,
                                       config.BackwardGenerator);
}

RISCVESP32P4MemmovePass::ProcessingConfig
RISCVESP32P4MemmovePass::getProcessingConfig(AlignmentCombo Combo) {
  using Config = ESP32P4OptimizationConfig;
  switch (Combo) {
  case AlignmentCombo::ScalarUnalignedConst:
    return ProcessingConfig(
        8, [this](IRBuilder<> &Builder, Value *Dst, Value *Src, uint64_t Size) {
          generateByteWiseBackwardCopy(Builder, Dst, Src, Size);
        });
  }
  llvm_unreachable("Unknown Alignment combination in overlap slice");
}

bool RISCVESP32P4MemmovePass::processDstUnalignConstMemIntrinBypass(
    MemMoveInst *M, BasicBlock::iterator &BBI) {
  if (Len == 0)
    return handleInstructionDeletion(M, BBI);
  // Overlap-slice: always ScalarUnalignedConst runtime dispatch.
  return processConstantSizeWithAlignment(M, BBI,
                                          AlignmentCombo::ScalarUnalignedConst);
}

bool RISCVESP32P4MemmovePass::processDstUnalignSrcUnalignConst(
    MemMoveInst *M, BasicBlock::iterator &BBI) {
  return processDstUnalignConstMemIntrinBypass(M, BBI);
}

RISCVESP32P4MemmovePass::MemmoveKind
RISCVESP32P4MemmovePass::getMemmoveKind(MemMoveInst *M) {
  MaybeAlign SrcAlign = M->getSourceAlign();
  MaybeAlign DstAlign = M->getDestAlign();
  SrcAlignValue = SrcAlign ? SrcAlign->value() : 1;
  DstAlignValue = DstAlign ? DstAlign->value() : 1;
  if (ConstantInt *CI = dyn_cast<ConstantInt>(M->getLength())) {
    Len = CI->getZExtValue();
    SizeValue = nullptr;
    // Overlap slice: all const-size memmoves use unaligned runtime dispatch.
    return MemmoveKind::DstUnalignSrcUnalign_Const;
  }
  Len = 0;
  SizeValue = M->getLength();
  return MemmoveKind::DstUnalignSrcUnalign_Var;
}

bool RISCVESP32P4MemmovePass::processMemmoveToSIMD(MemMoveInst *M,
                                                   BasicBlock::iterator &BBI) {
  MemmoveKind Kind = getMemmoveKind(M);
  switch (Kind) {
  case MemmoveKind::DstUnalignSrcUnalign_Const:
  case MemmoveKind::Dst16SrcUnalign_Const:
  case MemmoveKind::Dst8SrcUnalign_Const:
  case MemmoveKind::DstUnalignSrc16_Const:
    return processDstUnalignSrcUnalignConst(M, BBI);
  default:
    // Other kinds land in later stacked MRs.
    return false;
  }
}

bool RISCVESP32P4MemmovePass::convertMemmoveToMemcpy(
    MemMoveInst *M, BasicBlock::iterator &BBI) {
  LLVM_DEBUG(dbgs() << "RISCVESP32P4: Converting memmove to memcpy: " << *M
                    << "\n");

  // Create equivalent memcpy instruction
  IRBuilder<> Builder(M);

  // Create memcpy call, keep all original properties
  CallInst *NewMemcpy = Builder.CreateMemCpy(
      M->getRawDest(), M->getDestAlign(), M->getRawSource(),
      M->getSourceAlign(), M->getLength(), M->isVolatile());

  // Copy metadata
  NewMemcpy->copyMetadata(*M);

  // Copy debug information
  if (M->getDebugLoc())
    NewMemcpy->setDebugLoc(M->getDebugLoc());

  // Update MemorySSA (if needed)
  if (MSSAU) {
    auto *LastDef = cast<MemoryDef>(MSSA->getMemoryAccess(M));
    auto *NewAccess =
        MSSAU->createMemoryAccessAfter(NewMemcpy, nullptr, LastDef);
    MSSAU->insertDef(cast<MemoryDef>(NewAccess), /*RenameUses=*/true);
  }

  // Delete original memmove instruction
  handleInstructionDeletion(M, BBI);

  LLVM_DEBUG(dbgs() << "RISCVESP32P4: Successfully converted to memcpy: "
                    << *NewMemcpy << "\n");
  return true;
}

bool RISCVESP32P4MemmovePass::iterateOnFunction(Function &F) {
  bool MadeChange = false;

  // Preprocessing stage: convert memmove to memcpy only when no-overlap is
  // provable. Overlapping ranges must preserve memmove semantics.
  for (BasicBlock &BB : F) {
    for (BasicBlock::iterator BI = BB.begin(), BE = BB.end(); BI != BE;) {
      Instruction *I = &*BI++;

      if (auto *M = dyn_cast<MemMoveInst>(I)) {
        if (M->isVolatile())
          continue;

        // Convert even if the call carries no_reprocess (the replacement is
        // memcpy, so this pass will not recurse on it).
        const DataLayout &DL = F.getParent()->getDataLayout();
        if (canProveNoOverlapDstBeforeSrc(M, DL)) {
          BasicBlock::iterator TempBI = BI;
          --TempBI; // Point to current instruction
          if (convertMemmoveToMemcpy(M, TempBI)) {
            MadeChange = true;
            BI = TempBI; // Update iterator
          }
          continue;
        }

        if (isEsp32P4MemmoveNoReprocess(M))
          continue;
      }
    }
  }

  // Main processing stage: process remaining memory operations
  for (BasicBlock &BB : F) {
    for (BasicBlock::iterator BI = BB.begin(), BE = BB.end(); BI != BE;) {
      Instruction *I = &*BI++;

      if (auto *M = dyn_cast<MemMoveInst>(I)) {
        if (M->isVolatile())
          continue;
        if (isEsp32P4MemmoveNoReprocess(M))
          continue;

        // Process remaining memmove (already excluded dst < src case)
        if (processMemmoveToSIMD(M, BI)) {
          MadeChange = true;
        }
      }
    }
  }

  return MadeChange;
}

bool RISCVESP32P4MemmovePass::runImpl(Function &F, TargetLibraryInfo *TLI_,
                                      AAResults *AA_, AssumptionCache *AC_,
                                      DominatorTree *DT_,
                                      PostDominatorTree *PDT_, MemorySSA *MSSA_,
                                      ScalarEvolution *SE_,
                                      FunctionAnalysisManager &AM) {
  bool MadeChange = false;
  ChangedCFG = false;
  TLI = TLI_;
  AA = AA_;
  AC = AC_;
  DT = DT_;
  PDT = PDT_;
  MSSA = MSSA_;
  MemorySSAUpdater MSSAU_(MSSA_);
  MSSAU = &MSSAU_;
  SE = SE_; // Set ScalarEvolution

  while (true) {
    if (!iterateOnFunction(F))
      break;
    MadeChange = true;
  }

  return MadeChange;
}

PreservedAnalyses RISCVESP32P4MemmovePass::run(Function &F,
                                               FunctionAnalysisManager &AM) {
  if (!EnableRISCVESP32P4Memmove)
    return PreservedAnalyses::all();

  auto &TLI = AM.getResult<TargetLibraryAnalysis>(F);
  auto *AA = &AM.getResult<AAManager>(F);
  auto *AC = &AM.getResult<AssumptionAnalysis>(F);
  auto *DT = &AM.getResult<DominatorTreeAnalysis>(F);
  auto *PDT = &AM.getResult<PostDominatorTreeAnalysis>(F);
  auto *MSSA = &AM.getResult<MemorySSAAnalysis>(F);
  auto *SE = &AM.getResult<ScalarEvolutionAnalysis>(F); // Get ScalarEvolution
  TheModule = F.getParent();
  bool MadeChange = runImpl(F, &TLI, AA, AC, DT, PDT, &MSSA->getMSSA(), SE, AM);
  if (!MadeChange)
    return PreservedAnalyses::all();
  // Runtime dispatch splits blocks; do not claim CFG/MemorySSA.
  if (ChangedCFG)
    return PreservedAnalyses::none();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<MemorySSAAnalysis>();
  return PA;
}
