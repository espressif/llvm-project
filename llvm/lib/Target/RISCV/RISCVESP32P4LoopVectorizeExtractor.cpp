//===-- RISCVESP32P4LoopVectorizeExtractor.cpp -Loop Vectorizer -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that prepares loops for ESP32-P4 specific
// vectorization by setting appropriate loop metadata and running vectorization
// passes optimized for ESP32-P4 SIMD capabilities.
//
//===----------------------------------------------------------------------===//

#include "RISCVESP32P4LoopVectorizeExtractor.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/LoopExtractor.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Scalar/LoopStrengthReduce.h"
#include "llvm/Transforms/Vectorize/LoopVectorize.h"
#include "llvm/Transforms/Vectorize/SLPVectorizer.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-esp32p4-loop-vectorize-extractor"

// Constants for ESP32-P4 specific vectorization
static constexpr unsigned ESP32P4_SIMD_BIT_WIDTH = 128;
static constexpr unsigned DEFAULT_INTERLEAVE_COUNT = 1;
static constexpr char MUSTPROGRESS_METADATA_NAME[] = "llvm.loop.mustprogress";
static constexpr char TARGET_FEATURES_ATTR_NAME[] = "target-features";

// Vectorization metadata names
static constexpr char VECTORIZE_SCALABLE_ENABLE[] =
    "llvm.loop.vectorize.scalable.enable";
static constexpr char INTERLEAVE_COUNT[] = "llvm.loop.interleave.count";
static constexpr char VECTORIZE_ENABLE[] = "llvm.loop.vectorize.enable";
static constexpr char VECTORIZE_WIDTH[] = "llvm.loop.vectorize.width";

// Command line option to enable/disable RISCVESP32P4LoopVectorizeExtractor
cl::opt<bool> llvm::EnableRISCVESP32P4LoopVectorizeExtractor(
    "riscv-esp32p4-loop-vectorize-extractor", cl::init(false),
    cl::desc("Enable RISC-V ESP32-P4 loop vectorization extractor for specific "
             "loops"));

STATISTIC(NumLoopsVectorized, "Number of loops prepared for vectorization");
STATISTIC(NumFunctionsProcessed, "Number of functions processed");
STATISTIC(NumModulesWithExtraction,
          "Number of modules requiring loop extraction");

/// Extract the element type from memory access instructions
static Type *getElementTypeFromInstruction(const Instruction &I) {
  if (const auto *LI = dyn_cast<LoadInst>(&I)) {
    return LI->getType();
  }
  if (const auto *SI = dyn_cast<StoreInst>(&I)) {
    return SI->getValueOperand()->getType();
  }
  return nullptr;
}

/// Get the minimum element bit width from loop body memory accesses
static unsigned getLoopBodyElementBitWidth(Loop *L, const DataLayout &DL) {
  if (!L || L->getBlocks().empty())
    return 0;

  TypeSize MinBitWidth = TypeSize::getFixed(UINT_MAX);

  for (BasicBlock *BB : L->getBlocks()) {
    for (const Instruction &I : *BB) {
      Type *ElTy = getElementTypeFromInstruction(I);
      if (!ElTy)
        continue;

      // Handle vector types by extracting element type
      if (ElTy->isVectorTy())
        ElTy = cast<VectorType>(ElTy)->getElementType();

      // Only consider integer and floating-point types
      if (ElTy->isIntegerTy() || ElTy->isFloatingPointTy()) {
        MinBitWidth = std::min(MinBitWidth, DL.getTypeSizeInBits(ElTy));
      }
    }
  }

  return (!MinBitWidth.isScalable() &&
          MinBitWidth.getKnownMinValue() != UINT_MAX)
             ? MinBitWidth.getKnownMinValue()
             : 0;
}

/// Check if a loop has the required mustprogress metadata
static bool hasLoopMustProgressMetadata(const Loop *L) {
  if (!L)
    return false;

  const MDNode *LoopID = L->getLoopID();
  if (!LoopID)
    return false;

  for (unsigned I = 1; I < LoopID->getNumOperands(); ++I) {
    if (const auto *MD = dyn_cast<MDNode>(LoopID->getOperand(I))) {
      if (MD->getNumOperands() >= 1) {
        if (const auto *S = dyn_cast<MDString>(MD->getOperand(0))) {
          if (S->getString() == MUSTPROGRESS_METADATA_NAME) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

/// Determine if a loop is a candidate for vectorization
static bool isLoopVectorizationCandidate(const Loop *L, const DataLayout &DL) {
  if (!L)
    return false;

  // Only process innermost loops
  if (!L->isInnermost())
    return false;

  // Check for required metadata
  if (!hasLoopMustProgressMetadata(L))
    return false;

  // Verify element bit width compatibility
  unsigned ElementBitWidth =
      getLoopBodyElementBitWidth(const_cast<Loop *>(L), DL);
  if (ElementBitWidth == 0) {
    LLVM_DEBUG(dbgs() << "Loop has no valid element bit width\n");
    return false;
  }

  // Check if SIMD width is compatible with element width
  if (ESP32P4_SIMD_BIT_WIDTH % ElementBitWidth != 0) {
    LLVM_DEBUG(dbgs() << "SIMD width " << ESP32P4_SIMD_BIT_WIDTH
                      << " not compatible with element width "
                      << ElementBitWidth << "\n");
    return false;
  }

  return true;
}

/// Create vectorization metadata for a loop
static MDNode *createVectorizationMetadata(LLVMContext &Ctx,
                                           unsigned VectorWidth,
                                           unsigned InterleaveCount) {
  // Create individual metadata nodes
  MDNode *MustProgress =
      MDNode::get(Ctx, MDString::get(Ctx, MUSTPROGRESS_METADATA_NAME));

  MDNode *NoScalable = MDNode::get(
      Ctx,
      {MDString::get(Ctx, VECTORIZE_SCALABLE_ENABLE),
       ConstantAsMetadata::get(ConstantInt::get(Type::getInt1Ty(Ctx), 0))});

  MDNode *Interleave =
      MDNode::get(Ctx, {MDString::get(Ctx, INTERLEAVE_COUNT),
                        ConstantAsMetadata::get(ConstantInt::get(
                            Type::getInt32Ty(Ctx), InterleaveCount))});

  MDNode *VecEnable = MDNode::get(
      Ctx,
      {MDString::get(Ctx, VECTORIZE_ENABLE),
       ConstantAsMetadata::get(ConstantInt::get(Type::getInt1Ty(Ctx), 1))});

  MDNode *VecWidthMD =
      MDNode::get(Ctx, {MDString::get(Ctx, VECTORIZE_WIDTH),
                        ConstantAsMetadata::get(ConstantInt::get(
                            Type::getInt32Ty(Ctx), VectorWidth))});

  // Assemble the complete metadata
  SmallVector<Metadata *, 6> MDs;
  MDs.push_back(nullptr); // Self-reference placeholder
  MDs.push_back(MustProgress);
  MDs.push_back(VecWidthMD);
  MDs.push_back(NoScalable);
  MDs.push_back(Interleave);
  MDs.push_back(VecEnable);

  MDNode *NewLoopID = MDNode::get(Ctx, MDs);
  NewLoopID->replaceOperandWith(0, NewLoopID); // Set self-reference
  return NewLoopID;
}

bool RISCVESP32P4LoopVectorizeExtractorPass::prepareLoopForVectorization(
    Function &F, FunctionAnalysisManager &AM, unsigned InterleaveCount) {

  LoopInfo &LI = AM.getResult<LoopAnalysis>(F);
  const DataLayout &DL = F.getParent()->getDataLayout();

  for (Loop *L : LI) {
    if (!isLoopVectorizationCandidate(L, DL))
      continue;

    unsigned ElementBitWidth = getLoopBodyElementBitWidth(L, DL);
    // This should not happen as we already checked in
    // isLoopVectorizationCandidate
    assert(ElementBitWidth != 0 && "Element bit width should not be zero");

    unsigned VectorWidth = ESP32P4_SIMD_BIT_WIDTH / ElementBitWidth;

    LLVM_DEBUG(dbgs() << "Vectorizing loop in " << F.getName()
                      << " with element type width " << ElementBitWidth
                      << " and calculated vector factor " << VectorWidth
                      << "\n");

    MDNode *NewLoopID = createVectorizationMetadata(F.getContext(), VectorWidth,
                                                    InterleaveCount);
    L->setLoopID(NewLoopID);
    ++NumLoopsVectorized;
    return true;
  }
  return false;
}

bool RISCVESP32P4LoopVectorizeExtractorPass::hasProcessableLoops(
    Function &F, FunctionAnalysisManager &AM) {

  // Early exit if function lacks target-features attribute
  if (!F.getFnAttribute(TARGET_FEATURES_ATTR_NAME).isValid())
    return false;

  LoopInfo &LI = AM.getResult<LoopAnalysis>(F);
  const DataLayout &DL = F.getParent()->getDataLayout();

  // Check if any loop is a vectorization candidate
  return llvm::any_of(
      LI, [&DL](const Loop *L) { return isLoopVectorizationCandidate(L, DL); });
}

bool RISCVESP32P4LoopExtractorConditionalWrapper::hasLoopsNeedingExtraction(
    Module &M, ModuleAnalysisManager &AM) {

  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  // Check if any non-declaration function has processable loops
  for (const Function &F : M) {
    if (!F.isDeclaration() &&
        RISCVESP32P4LoopVectorizeExtractorPass::hasProcessableLoops(
            const_cast<Function &>(F), FAM)) {
      LLVM_DEBUG(dbgs() << "Found function " << F.getName()
                        << " with processable loops needing extraction\n");
      return true;
    }
  }

  return false;
}

bool RISCVESP32P4LoopVectorizeExtractorPass::runVectorizationPass(
    Function &F, FunctionAnalysisManager &AM, unsigned InterleaveCount) {

  if (!F.getFnAttribute(TARGET_FEATURES_ATTR_NAME).isValid()) {
    LLVM_DEBUG(
        dbgs()
        << "Function " << F.getName()
        << " lacks target-features attribute. Skipping vectorization.\n");
    return false;
  }

  bool Changed = prepareLoopForVectorization(F, AM, InterleaveCount);
  if (!Changed)
    return false;

  // Create fresh analysis managers for vectorization passes
  // This ensures we don't interfere with the calling pass's analysis state
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  PassBuilder PB;

  // Register all required analyses
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  // Configure vectorization options
  LoopVectorizeOptions Opts;
  Opts.VectorizeOnlyWhenForced = false;
  Opts.InterleaveOnlyWhenForced = false;

  // Build and run the vectorization pipeline
  FunctionPassManager FPM;
  FPM.addPass(LoopVectorizePass(Opts));
  FPM.addPass(SLPVectorizerPass());
  FPM.addPass(createFunctionToLoopPassAdaptor(LoopStrengthReducePass()));

  // Run the pipeline with the fresh analysis manager
  FPM.run(F, FAM);

  return true;
}

PreservedAnalyses
RISCVESP32P4LoopVectorizeExtractorPass::run(Function &F,
                                            FunctionAnalysisManager &FAM) {
  if (!EnableRISCVESP32P4LoopVectorizeExtractor)
    return PreservedAnalyses::all();

  ++NumFunctionsProcessed;

  bool Changed = runVectorizationPass(F, FAM, DEFAULT_INTERLEAVE_COUNT);
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

//===----------------------------------------------------------------------===//
// RISCVESP32P4LoopExtractorConditionalWrapper Implementation
//===----------------------------------------------------------------------===//

RISCVESP32P4LoopExtractorConditionalWrapper::
    RISCVESP32P4LoopExtractorConditionalWrapper(ModulePassManager &&PM)
    : PM(std::move(PM)) {}

PreservedAnalyses
RISCVESP32P4LoopExtractorConditionalWrapper::run(Module &M,
                                                 ModuleAnalysisManager &AM) {

  if (!hasLoopsNeedingExtraction(M, AM)) {
    LLVM_DEBUG(dbgs() << "No loops needing extraction found in module "
                      << M.getName() << ", skipping LoopExtractor passes\n");
    return PreservedAnalyses::all();
  }

  ++NumModulesWithExtraction;

  LLVM_DEBUG(dbgs() << "Running LoopExtractor passes for module " << M.getName()
                    << "\n");

  return PM.run(M, AM);
}