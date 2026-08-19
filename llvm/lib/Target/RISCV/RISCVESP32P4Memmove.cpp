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

namespace {
constexpr unsigned MemIntrinUnrollThresholdDefault = 150;
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
  case AlignmentCombo::Dst16Src16:
    return ProcessingConfig(
        Config::SIMD_REGISTER_SIZE,
        [this](IRBuilder<> &Builder, Value *Dst, Value *Src, uint64_t Size) {
          Value *DstInt = Builder.CreatePtrToInt(Dst, Builder.getInt64Ty());
          Value *SrcInt = Builder.CreatePtrToInt(Src, Builder.getInt64Ty());
          generateOptimizedBackwardCopyDst16Src16(Builder, Dst, Src, Size,
                                                  DstInt, SrcInt);
        });
  case AlignmentCombo::Dst16Src8:
    return ProcessingConfig(
        ESP32P4OptimizationConfig::SIMD_REGISTER_SIZE,
        [this](IRBuilder<> &Builder, Value *Dst, Value *Src, uint64_t Size) {
          generateOptimizedBackwardCopyDst16Src8(Builder, Dst, Src, Size);
        });
  case AlignmentCombo::ScalarUnalignedConst:
    return ProcessingConfig(
        8, [this](IRBuilder<> &Builder, Value *Dst, Value *Src, uint64_t Size) {
          generateByteWiseBackwardCopy(Builder, Dst, Src, Size);
        });
  }
  llvm_unreachable("Unknown Alignment combination");
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
  using Config = ESP32P4OptimizationConfig;
  MaybeAlign SrcAlign = M->getSourceAlign();
  MaybeAlign DstAlign = M->getDestAlign();
  SrcAlignValue = SrcAlign ? SrcAlign->value() : 1;
  DstAlignValue = DstAlign ? DstAlign->value() : 1;
  if (ConstantInt *CI = dyn_cast<ConstantInt>(M->getLength())) {
    Len = CI->getZExtValue();
    SizeValue = nullptr;
    if (Config::isDivisibleBy16(SrcAlignValue) &&
        Config::isDivisibleBy16(DstAlignValue)) {
      if (Config::isDivisibleBy16(Len))
        return MemmoveKind::Dst16Src16_Const16;
      if (Config::isDivisibleBy8(Len))
        return MemmoveKind::Dst16Src16_Const8;
      return MemmoveKind::Dst16Src16_OtherConst;
    }
    if (Config::isDivisibleBy8(SrcAlignValue) &&
        Config::isDivisibleBy16(DstAlignValue) &&
        !Config::isDivisibleBy16(SrcAlignValue))
      return MemmoveKind::Dst16Src8_Const;
    return MemmoveKind::DstUnalignSrcUnalign_Const;
  }
  Len = 0;
  SizeValue = M->getLength();
  if (Config::isDivisibleBy16(SrcAlignValue) &&
      Config::isDivisibleBy16(DstAlignValue))
    return MemmoveKind::Dst16Src16_Var;
  return MemmoveKind::DstUnalignSrcUnalign_Var;
}

bool RISCVESP32P4MemmovePass::processMemmoveToSIMD(MemMoveInst *M,
                                                   BasicBlock::iterator &BBI) {
  MemmoveKind Kind = getMemmoveKind(M);
  switch (Kind) {
  case MemmoveKind::Dst16Src16_Const16:
  case MemmoveKind::Dst16Src16_Const8:
  case MemmoveKind::Dst16Src16_OtherConst:
    return processDst16Src16Const(M, BBI, Kind);
  case MemmoveKind::Dst16Src8_Const:
    return processDst16Src8Const(M, BBI);
  case MemmoveKind::DstUnalignSrcUnalign_Const:
  case MemmoveKind::Dst16SrcUnalign_Const:
  case MemmoveKind::Dst8SrcUnalign_Const:
  case MemmoveKind::DstUnalignSrc16_Const:
    return processDstUnalignSrcUnalignConst(M, BBI);
  default:
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

namespace {

Value *combineEspV64LowHighToV128(IRBuilder<> &B, Value *VL, Value *VH) {
  return B.CreateShuffleVector(
      VL, VH,
      ArrayRef<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
}

Value *extractEspV128LowV64(IRBuilder<> &B, Value *V128) {
  return B.CreateShuffleVector(V128, V128,
                               ArrayRef<int>{0, 1, 2, 3, 4, 5, 6, 7});
}

Value *extractEspV128HighV64(IRBuilder<> &B, Value *V128) {
  return B.CreateShuffleVector(V128, V128,
                               ArrayRef<int>{8, 9, 10, 11, 12, 13, 14, 15});
}

} // namespace

bool RISCVESP32P4MemmovePass::processDst16Src16Const(MemMoveInst *M,
                                                     BasicBlock::iterator &BBI,
                                                     MemmoveKind Kind) {
  return processConstantSizeWithAlignment(M, BBI, AlignmentCombo::Dst16Src16);
}

void RISCVESP32P4MemmovePass::generateOptimizedBackwardCopyDst16Src16(
    IRBuilder<> &Builder, Value *Dst, Value *Src, uint64_t Size, Value *DstInt,
    Value *SrcInt) {
  LLVM_DEBUG(dbgs() << "RISCVESP32P4: Generating correct backward copy, size="
                    << Size << "\n");

  // Complete decomposition: size = 128*x + 16*y + 8*z + 4*a + 2*b + 1*c
  uint64_t Blocks128 = Size / 128;
  uint64_t RemainingAfter128 = Size % 128;

  uint64_t Blocks16 = RemainingAfter128 / 16;
  uint64_t RemainingAfter16 = RemainingAfter128 % 16;

  uint64_t Blocks8 = RemainingAfter16 / 8;
  uint64_t RemainingAfter8 = RemainingAfter16 % 8;

  uint64_t Blocks4 = RemainingAfter8 / 4;
  uint64_t RemainingAfter4 = RemainingAfter8 % 4;

  uint64_t Blocks2 = RemainingAfter4 / 2;
  uint64_t Blocks1 = RemainingAfter4 % 2;

  // Calculate the exact offset of each part (backward copy order)
  uint64_t Offset1 = Size - Blocks1;           // 1-byte end position
  uint64_t Offset2 = Offset1 - Blocks2 * 2;    // 2-byte end position
  uint64_t Offset4 = Offset2 - Blocks4 * 4;    // 4-byte end position
  uint64_t Offset8 = Offset4 - Blocks8 * 8;    // 8-byte end position
  uint64_t Offset16 = Offset8 - Blocks16 * 16; // 16-byte end position
  uint64_t Offset128 =
      Offset16 - Blocks128 * 128; // 128-byte end position (should be 0)

  assert(Offset128 == 0 && "128-byte offset should be 0");

  // Backward copy: strict order 1→2→4→8→16→128. First: 1 byte (if any).
  if (Blocks1 > 0) {
    Value *Block1Src = Builder.CreateConstInBoundsGEP1_64(
        Builder.getInt8Ty(), Src, Size - 1, "block1.src");
    Value *Block1Dst = Builder.CreateConstInBoundsGEP1_64(
        Builder.getInt8Ty(), Dst, Size - 1, "block1.dst");

    Value *Data =
        Builder.CreateLoad(Builder.getInt8Ty(), Block1Src, "load.1byte");
    Builder.CreateStore(Data, Block1Dst);

    LLVM_DEBUG(dbgs() << "RISCVESP32P4: Processed 1 byte at offset "
                      << (Size - 1) << "\n");
  }

  // Second step: process 2 bytes (if any)
  if (Blocks2 > 0) {
    Value *Block2Src = Builder.CreateConstInBoundsGEP1_64(
        Builder.getInt8Ty(), Src, Offset1 - 2, "block2.src");
    Value *Block2Dst = Builder.CreateConstInBoundsGEP1_64(
        Builder.getInt8Ty(), Dst, Offset1 - 2, "block2.dst");

    Value *SrcPtr16 = Builder.CreateBitCast(
        Block2Src, PointerType::get(Builder.getInt16Ty(), 0), "src.ptr16");
    Value *Data =
        Builder.CreateLoad(Builder.getInt16Ty(), SrcPtr16, "load.2bytes");

    Value *DstPtr16 = Builder.CreateBitCast(
        Block2Dst, PointerType::get(Builder.getInt16Ty(), 0), "dst.ptr16");
    Builder.CreateStore(Data, DstPtr16);

    LLVM_DEBUG(dbgs() << "RISCVESP32P4: Processed 2 bytes at offset "
                      << (Offset1 - 2) << "\n");
  }

  // Third step: process 4 bytes (if any)
  if (Blocks4 > 0) {
    Value *Block4Src = Builder.CreateConstInBoundsGEP1_64(
        Builder.getInt8Ty(), Src, Offset2 - 4, "block4.src");
    Value *Block4Dst = Builder.CreateConstInBoundsGEP1_64(
        Builder.getInt8Ty(), Dst, Offset2 - 4, "block4.dst");

    Value *SrcPtr32 = Builder.CreateBitCast(
        Block4Src, PointerType::get(Builder.getInt32Ty(), 0), "src.ptr32");
    Value *Data =
        Builder.CreateLoad(Builder.getInt32Ty(), SrcPtr32, "load.4bytes");

    Value *DstPtr32 = Builder.CreateBitCast(
        Block4Dst, PointerType::get(Builder.getInt32Ty(), 0), "dst.ptr32");
    Builder.CreateStore(Data, DstPtr32);

    LLVM_DEBUG(dbgs() << "RISCVESP32P4: Processed 4 bytes at offset "
                      << (Offset2 - 4) << "\n");
  }

  // Fourth step: process 8 bytes (if any)
  if (Blocks8 > 0) {
    Value *Block8Src = Builder.CreateConstInBoundsGEP1_64(
        Builder.getInt8Ty(), Src, Offset4 - 8, "block8.src");
    Value *Block8Dst = Builder.CreateConstInBoundsGEP1_64(
        Builder.getInt8Ty(), Dst, Offset4 - 8, "block8.dst");

    // 8-byte chunk: vld.h.64.ip.m then vst.h.64.ip.m (explicit v8i8)
    auto [V8, SrcNext] = createEspVldH64IpM(Builder, Block8Src, 0);
    (void)SrcNext;
    createEspVstH64IpM(Builder, V8, Block8Dst, 0);

    LLVM_DEBUG(dbgs() << "RISCVESP32P4: Processed 8 bytes at offset "
                      << (Offset4 - 8) << "\n");
  }

  // Fifth step: 16-byte blocks (if any), backward (high-offset block first).
  if (Blocks16 > 0) {
    assert(Blocks16 < 8 && "Blocks16 should be less than 8 for register limit");

    uint64_t Blocks16Offset = Offset8 - Blocks16 * 16; // i.e. Offset16

    // Highest block offset first, down to 0, so source bytes not yet copied
    // are not overwritten.
    for (uint64_t I = 0; I < Blocks16; ++I) {
      uint64_t BlockOffset =
          Blocks16Offset + (Blocks16 - 1 - I) * 16; // high block first
      Value *Blocks16SrcI = Builder.CreateConstInBoundsGEP1_64(
          Builder.getInt8Ty(), Src, BlockOffset, "Blocks16.src");
      Value *Blocks16DstI = Builder.CreateConstInBoundsGEP1_64(
          Builder.getInt8Ty(), Dst, BlockOffset, "Blocks16.dst");
      (void)emitBackwardDst16Src16OneBlock_Ptr(Builder, Blocks16SrcI,
                                               Blocks16DstI);
    }

    LLVM_DEBUG(dbgs() << "RISCVESP32P4: Processed " << Blocks16
                      << " x 16-byte blocks (backward order) from offset "
                      << Blocks16Offset << "\n");
  }

  // Sixth step: process 128-byte blocks (if any) - backward processing
  if (Blocks128 > 0) {
    // 128-byte blocks start at offset 0, end address is Blocks128 * 128
    Value *Blocks128EndSrc = Builder.CreateConstInBoundsGEP1_64(
        Builder.getInt8Ty(), Src, Blocks128 * 128 - 16, "Blocks128.end.src");
    Value *Blocks128EndDst = Builder.CreateConstInBoundsGEP1_64(
        Builder.getInt8Ty(), Dst, Blocks128 * 128 - 16, "Blocks128.end.dst");

    if (Blocks128 > static_cast<uint64_t>(MemIntrinUnrollThresholdDefault)) {
      generateLoopBased128BlockBackwardCopy(Builder, Blocks128EndSrc,
                                            Blocks128EndDst, Blocks128);
    } else {
      generateUnrolled128BlockBackwardCopy(Builder, Blocks128EndSrc,
                                           Blocks128EndDst, Blocks128);
    }
    LLVM_DEBUG(dbgs() << "RISCVESP32P4: Processed " << Blocks128
                      << " x 128B\n");
  }

  LLVM_DEBUG(dbgs() << "RISCVESP32P4: Completed backward copy decomposition\n");
}

void RISCVESP32P4MemmovePass::generateLoopBased128BlockBackwardCopy(
    IRBuilder<> &Builder, Value *EndSrc, Value *EndDst, uint64_t NumBlocks) {
  // Use loop dispatcher with ptr (no ptr2int): 8x (.m vld then vst) per
  // 128-byte block
  generateLoopDispatcher(
      Builder, EndSrc, EndDst, NumBlocks, "dst16src16.128backward",
      [this](IRBuilder<> &B, Value *&SrcPtr, Value *&DstPtr, Value *LoopIndex) {
        for (int I = 0; I < 8; ++I) {
          std::tie(SrcPtr, DstPtr) =
              emitBackwardDst16Src16OneBlock_Ptr(B, SrcPtr, DstPtr);
        }
      });

  LLVM_DEBUG(dbgs() << "RISCVESP32P4: Generated 128B backward loop for "
                    << NumBlocks << " iterations\n");
}

void RISCVESP32P4MemmovePass::generateUnrolled128BlockBackwardCopy(
    IRBuilder<> &Builder, Value *EndSrc, Value *EndDst, uint64_t NumBlocks) {
  Value *SrcPtr = EndSrc;
  Value *DstPtr = EndDst;

  // Completely unrolled: 8x (.m vld then vst) per 128-byte block (ptr in/out)
  for (uint64_t Block = 0; Block < NumBlocks; ++Block) {
    for (int I = 0; I < 8; ++I) {
      std::tie(SrcPtr, DstPtr) =
          emitBackwardDst16Src16OneBlock_Ptr(Builder, SrcPtr, DstPtr);
    }
  }

  LLVM_DEBUG(dbgs() << "RISCVESP32P4: Unrolled " << NumBlocks
                    << " x 128B blocks\n");
}

void RISCVESP32P4MemmovePass::generateLoopDispatcher(
    IRBuilder<> &Builder, Value *InitSrcAddr, Value *InitDstAddr,
    uint64_t NumIterations, const std::string &LoopName,
    std::function<void(IRBuilder<> &, Value *&, Value *&, Value *)>
        BodyGenerator) {
  LLVM_DEBUG(dbgs() << "RISCVESP32P4: Loop dispatcher for " << LoopName << ", "
                    << NumIterations << " iterations\n");

  // Create loop basic blocks
  Function *F = getCurrentFunction(Builder);
  BasicBlock *LoopHeaderBB =
      BasicBlock::Create(F->getContext(), LoopName + ".loop.header", F);
  BasicBlock *LoopBodyBB =
      BasicBlock::Create(F->getContext(), LoopName + ".loop.body", F);
  BasicBlock *LoopExitBB =
      BasicBlock::Create(F->getContext(), LoopName + ".loop.exit", F);

  // Jump to loop header
  Builder.CreateBr(LoopHeaderBB);

  // Loop header: set PHI node and loop condition (addr type = type of
  // InitSrcAddr, may be ptr or i32)
  Builder.SetInsertPoint(LoopHeaderBB);
  Type *AddrTy = InitSrcAddr->getType();
  PHINode *LoopIndex = Builder.CreatePHI(Builder.getInt32Ty(), 2, "loop.index");
  PHINode *CurrentSrc = Builder.CreatePHI(AddrTy, 2, "current.src");
  PHINode *CurrentDst = Builder.CreatePHI(AddrTy, 2, "current.dst");

  // Initial value
  LoopIndex->addIncoming(Builder.getInt32(0),
                         Builder.GetInsertBlock()->getSinglePredecessor());
  CurrentSrc->addIncoming(InitSrcAddr,
                          Builder.GetInsertBlock()->getSinglePredecessor());
  CurrentDst->addIncoming(InitDstAddr,
                          Builder.GetInsertBlock()->getSinglePredecessor());

  // Loop condition
  Value *LoopCond = Builder.CreateICmpULT(
      LoopIndex, Builder.getInt32(NumIterations), "loop.cond");
  Builder.CreateCondBr(LoopCond, LoopBodyBB, LoopExitBB);

  // Loop body: execute specific SIMD operation
  Builder.SetInsertPoint(LoopBodyBB);
  Value *NewSrcAddr = CurrentSrc;
  Value *NewDstAddr = CurrentDst;

  // Call provided loop body generator
  BodyGenerator(Builder, NewSrcAddr, NewDstAddr, LoopIndex);

  // Update loop variable
  Value *NextIndex =
      Builder.CreateAdd(LoopIndex, Builder.getInt32(1), "next.index");
  LoopIndex->addIncoming(NextIndex, LoopBodyBB);
  CurrentSrc->addIncoming(NewSrcAddr, LoopBodyBB);
  CurrentDst->addIncoming(NewDstAddr, LoopBodyBB);

  Builder.CreateBr(LoopHeaderBB);

  // Loop exit
  Builder.SetInsertPoint(LoopExitBB);

  LLVM_DEBUG(dbgs() << "RISCVESP32P4: Loop dispatcher completed for "
                    << LoopName << "\n");
}

std::pair<Value *, Value *>
RISCVESP32P4MemmovePass::emitBackwardDst16Src16OneBlock_Ptr(
    IRBuilder<> &Builder, Value *SrcPtr, Value *DstPtr) {
  Value *SrcHigh = Builder.CreateConstInBoundsGEP1_64(Builder.getInt8Ty(),
                                                      SrcPtr, 8, "src16.high");
  auto [VH, P1] = createEspVldL64IpM(Builder, SrcHigh, -8);
  auto [VL, P2] = createEspVldL64IpM(Builder, P1, -8);
  (void)P2;
  Value *V128 = combineEspV64LowHighToV128(Builder, VL, VH);
  Value *NextDst = createEspVst128IpM(Builder, V128, DstPtr, -16);
  Value *NextSrc = Builder.CreateConstInBoundsGEP1_64(
      Builder.getInt8Ty(), SrcPtr, -16, "src16.prev");
  return {NextSrc, NextDst};
}

std::pair<Value *, Value *>
RISCVESP32P4MemmovePass::createEspVld128IpM(IRBuilder<> &Builder, Value *SrcPtr,
                                            int Step) {
  Type *I32Ty = Builder.getInt32Ty();
  Function *F = Intrinsic::getOrInsertDeclaration(
      TheModule, Intrinsic::riscv_esp_vld_128_ip, {});
  Value *Args[] = {SrcPtr, ConstantInt::getSigned(I32Ty, Step)};
  CallInst *Call = Builder.CreateCall(F, Args, "vld128ip.m");
  Call->addFnAttr(Attribute::AlwaysInline);
  Value *Vec = Builder.CreateExtractValue(Call, 0, "vld128ip.m.vec");
  Value *NextPtr = Builder.CreateExtractValue(Call, 1, "vld128ip.m.nextptr");
  return {Vec, NextPtr};
}

Value *RISCVESP32P4MemmovePass::createEspVst128IpM(IRBuilder<> &Builder,
                                                   Value *Vec, Value *DstPtr,
                                                   int Step) {
  Type *I32Ty = Builder.getInt32Ty();
  Function *F = Intrinsic::getOrInsertDeclaration(
      TheModule, Intrinsic::riscv_esp_vst_128_ip, {});
  Value *Args[] = {Vec, DstPtr, ConstantInt::getSigned(I32Ty, Step)};
  CallInst *Call = Builder.CreateCall(F, Args, "vst128ip.m");
  Call->addFnAttr(Attribute::AlwaysInline);
  return Call;
}

std::pair<Value *, Value *>
RISCVESP32P4MemmovePass::createEspVld128IpMThenVst128IpM(IRBuilder<> &Builder,
                                                         Value *SrcPtr,
                                                         Value *DstPtr,
                                                         int Step) {
  auto [Vec, NextSrcPtr] = createEspVld128IpM(Builder, SrcPtr, Step);
  Value *NextDstPtr = createEspVst128IpM(Builder, Vec, DstPtr, Step);
  return {NextSrcPtr, NextDstPtr};
}

std::pair<Value *, Value *>
RISCVESP32P4MemmovePass::createEspVldH64IpM(IRBuilder<> &Builder, Value *SrcPtr,
                                            int Step) {
  Type *I32Ty = Builder.getInt32Ty();
  Function *F = Intrinsic::getOrInsertDeclaration(
      TheModule, Intrinsic::riscv_esp_vld_h_64_ip, {});
  CallInst *Call = Builder.CreateCall(
      F, {SrcPtr, ConstantInt::getSigned(I32Ty, Step)}, "vldh64ip.m");
  Call->addFnAttr(Attribute::AlwaysInline);
  Value *Vec = Builder.CreateExtractValue(Call, 0, "vldh64ip.m.vec");
  Value *NextPtr = Builder.CreateExtractValue(Call, 1, "vldh64ip.m.nextptr");
  return {Vec, NextPtr};
}

std::pair<Value *, Value *>
RISCVESP32P4MemmovePass::createEspVldL64IpM(IRBuilder<> &Builder, Value *SrcPtr,
                                            int Step) {
  Type *I32Ty = Builder.getInt32Ty();
  Function *F = Intrinsic::getOrInsertDeclaration(
      TheModule, Intrinsic::riscv_esp_vld_l_64_ip, {});
  CallInst *Call = Builder.CreateCall(
      F, {SrcPtr, ConstantInt::getSigned(I32Ty, Step)}, "vldl64ip.m");
  Call->addFnAttr(Attribute::AlwaysInline);
  Value *Vec = Builder.CreateExtractValue(Call, 0, "vldl64ip.m.vec");
  Value *NextPtr = Builder.CreateExtractValue(Call, 1, "vldl64ip.m.nextptr");
  return {Vec, NextPtr};
}

Value *RISCVESP32P4MemmovePass::createEspVstH64IpM(IRBuilder<> &Builder,
                                                   Value *Vec, Value *DstPtr,
                                                   int Step) {
  Type *I32Ty = Builder.getInt32Ty();
  Function *F = Intrinsic::getOrInsertDeclaration(
      TheModule, Intrinsic::riscv_esp_vst_h_64_ip, {});
  CallInst *Call = Builder.CreateCall(
      F, {Vec, DstPtr, ConstantInt::getSigned(I32Ty, Step)}, "vsth64ip.m");
  Call->addFnAttr(Attribute::AlwaysInline);
  return Call;
}

Value *RISCVESP32P4MemmovePass::createEspVstL64IpM(IRBuilder<> &Builder,
                                                   Value *Vec, Value *DstPtr,
                                                   int Step) {
  Type *I32Ty = Builder.getInt32Ty();
  Function *F = Intrinsic::getOrInsertDeclaration(
      TheModule, Intrinsic::riscv_esp_vst_l_64_ip, {});
  CallInst *Call = Builder.CreateCall(
      F, {Vec, DstPtr, ConstantInt::getSigned(I32Ty, Step)}, "vstl64ip.m");
  Call->addFnAttr(Attribute::AlwaysInline);
  return Call;
}

CallInst *RISCVESP32P4MemmovePass::createOptimizedMemCpy(
    IRBuilder<> &Builder, Value *Dst, Value *Src, Value *Size,
    MaybeAlign DstAlign, MaybeAlign SrcAlign, bool IsVolatile,
    const MemMoveInst *OriginalInst) {
  // Create memcpy call
  CallInst *MemcpyCall =
      Builder.CreateMemCpy(Dst, DstAlign, Src, SrcAlign, Size, IsVolatile);

  // If original instruction is provided, copy its metadata
  if (OriginalInst) {
    MemcpyCall->copyMetadata(*OriginalInst);
  }

  LLVM_DEBUG(dbgs() << "RISCVESP32P4: Created optimized memcpy with "
                    << "dst_align=" << (DstAlign ? DstAlign->value() : 0)
                    << ", src_align=" << (SrcAlign ? SrcAlign->value() : 0)
                    << ", volatile=" << IsVolatile << "\n");

  return MemcpyCall;
}

Function *
RISCVESP32P4MemmovePass::getCurrentFunction(IRBuilder<> &Builder) const {
  return Builder.GetInsertBlock()->getParent();
}

Value *RISCVESP32P4MemmovePass::createPtrToIntAddr(IRBuilder<> &Builder,
                                                   Value *Ptr,
                                                   const std::string &Name) {
  // ESP32-P4 uses 32-bit address space
  Value *IntAddr = Builder.CreatePtrToInt(Ptr, Builder.getInt32Ty(),
                                          Name.empty() ? "ptr.addr" : Name);

  return IntAddr;
}

void RISCVESP32P4MemmovePass::generateOptimizedBackwardCopyDispatcher(
    IRBuilder<> &Builder, Value *Dst, Value *Src, uint64_t Size,
    uint64_t Alignment,
    std::function<void(IRBuilder<> &, Value *, Value *, uint64_t, uint64_t,
                       uint64_t)>
        UnrollGenerator,
    std::function<void(IRBuilder<> &, Value *, Value *, uint64_t, uint64_t,
                       uint64_t)>
        LoopGenerator) {
  LLVM_DEBUG(dbgs() << "RISCVESP32P4: Backend dispatcher for " << Alignment
                    << "-byte aligned copy, size=" << Size << "\n");

  // 1. Process tail non-aligned bytes
  uint64_t Remainder = Size % Alignment;
  if (Remainder > 0) {
    Value *TailSrc = Builder.CreateConstInBoundsGEP1_64(
        Builder.getInt8Ty(), Src, Size - Remainder, "tail.src");
    Value *TailDst = Builder.CreateConstInBoundsGEP1_64(
        Builder.getInt8Ty(), Dst, Size - Remainder, "tail.dst");

    // Historically Alignment==16 used TailAlign=1 so llvm.memcpy could handle
    // mixed dst/src tail alignment (e.g. dst16/src8). On riscv32, align-1
    // memcpy may widen to i64 load/store that the ESP32-P4 backend cannot
    // select. Emit explicit i8 copies for that case (Remainder < 16).
    uint64_t TailAlign = (Alignment == 16) ? 1 : Alignment;
    if (TailAlign == 1) {
      // This is a backward copy (runtime dst > src): the tail is the highest
      // addresses, so copy high-to-low within the tail. A forward loop would
      // destroy source bytes when 0 < dst-src < Remainder (e.g. dst=src+8,
      // Size=28 -> Remainder=12) before they are read.
      for (int64_t I = (int64_t)Remainder - 1; I >= 0; --I) {
        Value *S = Builder.CreateConstInBoundsGEP1_64(Builder.getInt8Ty(),
                                                      TailSrc, I, "tail.src.b");
        Value *D = Builder.CreateConstInBoundsGEP1_64(Builder.getInt8Ty(),
                                                      TailDst, I, "tail.dst.b");
        Value *B = Builder.CreateLoad(Builder.getInt8Ty(), S, "tail.ld");
        Builder.CreateStore(B, D);
      }
      LLVM_DEBUG(dbgs() << "RISCVESP32P4: Processed " << Remainder
                        << " tail bytes (scalar i8, high-to-low)\n");
    } else {
      Builder.CreateMemCpy(TailDst, Align(TailAlign), TailSrc, Align(TailAlign),
                           Builder.getInt32(Remainder), false);
      LLVM_DEBUG(dbgs() << "RISCVESP32P4: Processed " << Remainder
                        << " Remainder bytes with " << TailAlign
                        << "-byte Alignment\n");
    }
  }

  // 2. Process aligned blocks
  uint64_t Blocks = (Size - Remainder) / Alignment;
  if (Blocks > 0) {
    // Select unroll or loop version based on block number
    // Use LLVM's default unroll threshold, but adjust for larger blocks (e.g.
    // 16 bytes)
    uint64_t Threshold = static_cast<uint64_t>(MemIntrinUnrollThresholdDefault);
    if (Alignment == 16) {
      Threshold =
          Threshold / 2; // 16-byte block is larger, reduce unroll threshold
    }

    if (Blocks <= Threshold) {
      // Small loop: fully unroll
      LLVM_DEBUG(dbgs() << "RISCVESP32P4: Using unrolled version for " << Blocks
                        << " blocks\n");
      UnrollGenerator(Builder, Dst, Src, Size, Remainder, Blocks);
    } else {
      // Large loop: generate loop structure
      LLVM_DEBUG(dbgs() << "RISCVESP32P4: Using loop version for " << Blocks
                        << " blocks\n");
      LoopGenerator(Builder, Dst, Src, Size, Remainder, Blocks);
    }
  }

  LLVM_DEBUG(dbgs() << "RISCVESP32P4: Backend dispatcher completed\n");
}

void RISCVESP32P4MemmovePass::generateUnrolledDispatcher(
    IRBuilder<> &Builder, Value *Dst, Value *Src, uint64_t Size,
    uint64_t Remainder, uint64_t NumBlocks, uint64_t BlockSize,
    int SrcOffsetFromEnd, int DstOffsetFromEnd, const std::string &CopyName,
    std::function<void(IRBuilder<> &, Value *&, Value *&, uint64_t)>
        BlockGenerator) {
  LLVM_DEBUG(dbgs() << "RISCVESP32P4: Unrolled " << CopyName << ", "
                    << NumBlocks << " x " << BlockSize << "B blocks\n");

  // Calculate starting address (from end offset)
  Value *SrcStartAddr = Builder.CreateConstInBoundsGEP1_64(
      Builder.getInt8Ty(), Src, Size - Remainder + SrcOffsetFromEnd,
      "src.start.addr");
  Value *DstStartAddr = Builder.CreateConstInBoundsGEP1_64(
      Builder.getInt8Ty(), Dst, Size - Remainder + DstOffsetFromEnd,
      "dst.start.addr");

  // Convert to integer address
  Value *SrcAddr = Builder.CreatePtrToInt(SrcStartAddr, Builder.getInt32Ty());
  Value *DstAddr = Builder.CreatePtrToInt(DstStartAddr, Builder.getInt32Ty());

  // Fully unrolled: use improved step mode
  for (uint64_t I = 0; I < NumBlocks; ++I) {
    BlockGenerator(Builder, SrcAddr, DstAddr, I);
  }

  LLVM_DEBUG(dbgs() << "RISCVESP32P4: Unrolled " << NumBlocks << " x "
                    << BlockSize << "B blocks using " << CopyName << "\n");
}

bool RISCVESP32P4MemmovePass::processDst16Src8Const(MemMoveInst *M,
                                                    BasicBlock::iterator &BBI) {
  return processConstantSizeWithAlignment(M, BBI, AlignmentCombo::Dst16Src8);
}

void RISCVESP32P4MemmovePass::generateOptimizedBackwardCopyDst16Src8(
    IRBuilder<> &Builder, Value *Dst, Value *Src, uint64_t Size) {
  // Use backend dispatcher to handle generic logic
  generateOptimizedBackwardCopyDispatcher(
      Builder, Dst, Src, Size, /*Alignment=*/16,
      // Unroll version generator
      [this](IRBuilder<> &B, Value *Dst, Value *Src, uint64_t Size,
             uint64_t Remainder, uint64_t Blocks) {
        generateUnrolledBackwardCopyDst16Src8(B, Dst, Src, Size, Remainder,
                                              Blocks);
      },
      // Loop version generator
      [this](IRBuilder<> &B, Value *Dst, Value *Src, uint64_t Size,
             uint64_t Remainder, uint64_t Blocks) {
        generateLoopBackwardCopyDst16Src8(B, Dst, Src, Size, Remainder, Blocks);
      });
}

void RISCVESP32P4MemmovePass::generateUnrolledBackwardCopyDst16Src8(
    IRBuilder<> &Builder, Value *Dst, Value *Src, uint64_t Size,
    uint64_t Remainder, uint64_t Blocks16) {
  // Use unroll dispatcher to handle generic logic
  generateUnrolledDispatcher(
      Builder, Dst, Src, Size, Remainder, Blocks16, /*BlockSize=*/16,
      /*SrcOffsetFromEnd=*/-8, /*DstOffsetFromEnd=*/-16, "dst16src8.backward",
      [this](IRBuilder<> &B, Value *&SrcAddr, Value *&DstAddr, uint64_t I) {
        (void)I;
        std::tie(SrcAddr, DstAddr) =
            emitBackwardDst16Src8OneBlock_I32(B, SrcAddr, DstAddr);
      });
}

void RISCVESP32P4MemmovePass::generateLoopBackwardCopyDst16Src8(
    IRBuilder<> &Builder, Value *Dst, Value *Src, uint64_t Size,
    uint64_t Remainder, uint64_t Blocks16) {
  LLVM_DEBUG(dbgs() << "RISCVESP32P4: Loop backward copy dst16-src8, "
                    << Blocks16 << " blocks\n");

  // Calculate number of 128-byte blocks (8 16-byte blocks) and remaining
  // 16-byte blocks
  uint64_t Blocks128 = Blocks16 / 8;   // Number of complete 128-byte blocks
  uint64_t Remaining16 = Blocks16 % 8; // Number of remaining 16-byte blocks

  LLVM_DEBUG(dbgs() << "RISCVESP32P4: " << Blocks128 << " x 128B blocks + "
                    << Remaining16 << " x 16B blocks\n");

  // First stage: process 128-byte blocks (if any)
  if (Blocks128 > 0) {
    generateLoop128ByteBackwardCopyDst16Src8(Builder, Dst, Src, Size, Remainder,
                                             Blocks128);
  }

  // Second stage: process remaining 16-byte blocks (if any)
  if (Remaining16 > 0) {
    generateRemaining16ByteBackwardCopyDst16Src8(
        Builder, Dst, Src, Size, Remainder, Blocks128, Remaining16);
  }
}

void RISCVESP32P4MemmovePass::generateLoop128ByteBackwardCopyDst16Src8(
    IRBuilder<> &Builder, Value *Dst, Value *Src, uint64_t Size,
    uint64_t Remainder, uint64_t Blocks128) {
  // 128-byte block region; last block: high 8 bytes of src (reverse starts
  // here).
  Value *Last128BlockSrcHigh = Builder.CreateConstInBoundsGEP1_64(
      Builder.getInt8Ty(), Src, Size - Remainder - 8, "last.128block.src.high");
  Value *Last128BlockDst = Builder.CreateConstInBoundsGEP1_64(
      Builder.getInt8Ty(), Dst, Size - Remainder - 16, "last.128block.dst");

  generateLoopDispatcher(
      Builder, Last128BlockSrcHigh, Last128BlockDst, Blocks128, "dst16src8.128",
      [this](IRBuilder<> &B, Value *&SrcAddr, Value *&DstAddr,
             Value *LoopIndex) {
        (void)LoopIndex;
        for (int I = 0; I < 8; ++I) {
          std::tie(SrcAddr, DstAddr) =
              emitBackwardDst16Src8OneBlock_Ptr(B, SrcAddr, DstAddr);
        }
      });

  LLVM_DEBUG(dbgs() << "RISCVESP32P4: Generated 128B loop for " << Blocks128
                    << " iterations\n");
}

void RISCVESP32P4MemmovePass::generateRemaining16ByteBackwardCopyDst16Src8(
    IRBuilder<> &Builder, Value *Dst, Value *Src, uint64_t Size,
    uint64_t Remainder, uint64_t Blocks128, uint64_t Remaining16) {
  LLVM_DEBUG(dbgs() << "RISCVESP32P4: Processing remaining " << Remaining16
                    << " x 16B blocks\n");

  // Calculate correct position of remaining 16-byte block region.
  uint64_t Blocks128Size =
      Blocks128 * 128; // Total size of 128-byte block region

  // The starting address of the last remaining 16-byte block's high 8 bytes
  Value *Remaining16SrcHigh = Builder.CreateConstInBoundsGEP1_64(
      Builder.getInt8Ty(), Src, Size - Remainder - Blocks128Size - 8,
      "remaining.src.high");
  Value *Remaining16Dst = Builder.CreateConstInBoundsGEP1_64(
      Builder.getInt8Ty(), Dst, Size - Remainder - Blocks128Size - 16,
      "remaining.dst");

  Value *SrcPtr = Remaining16SrcHigh;
  Value *DstPtr = Remaining16Dst;
  for (uint64_t I = 0; I < Remaining16; ++I) {
    (void)I;
    std::tie(SrcPtr, DstPtr) =
        emitBackwardDst16Src8OneBlock_Ptr(Builder, SrcPtr, DstPtr);
  }

  LLVM_DEBUG(dbgs() << "RISCVESP32P4: Processed remaining " << Remaining16
                    << " x 16B blocks\n");
}

std::pair<Value *, Value *>
RISCVESP32P4MemmovePass::emitBackwardDst16Src8OneBlock_Ptr(IRBuilder<> &Builder,
                                                           Value *SrcPtr,
                                                           Value *DstPtr) {
  auto [VH, P1] = createEspVldL64IpM(Builder, SrcPtr, -8);
  auto [VL, P2] = createEspVldL64IpM(Builder, P1, -8);
  Value *V128 = combineEspV64LowHighToV128(Builder, VL, VH);
  Value *NextDst = createEspVst128IpM(Builder, V128, DstPtr, -16);
  return {P2, NextDst};
}

std::pair<Value *, Value *>
RISCVESP32P4MemmovePass::emitBackwardDst16Src8OneBlock_I32(IRBuilder<> &Builder,
                                                           Value *SrcAddrI32,
                                                           Value *DstAddrI32) {
  Type *I32Ty = Builder.getInt32Ty();
  Value *SrcPtr = Builder.CreateIntToPtr(SrcAddrI32, Builder.getPtrTy());
  Value *DstPtr = Builder.CreateIntToPtr(DstAddrI32, Builder.getPtrTy());
  auto [NS, ND] = emitBackwardDst16Src8OneBlock_Ptr(Builder, SrcPtr, DstPtr);
  return {Builder.CreatePtrToInt(NS, I32Ty), Builder.CreatePtrToInt(ND, I32Ty)};
}
