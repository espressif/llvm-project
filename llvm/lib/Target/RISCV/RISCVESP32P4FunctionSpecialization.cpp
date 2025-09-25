//===- RISCVESP32P4FunctionSpecialization.cpp - Function Specialization   -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that specializes the dsps_firmr_s16_ansi
// function based on the value of the 'shift' field in its context struct.
//
// The pass performs the following transformations:
// 1. Identifies FIRMR struct access patterns in the function
// 2. Locates specialization conditions based on the 'shift' field
// 3. Creates specialized versions of basic blocks based on the 'interp' field
// 4. Optimizes the function by eliminating runtime checks where possible
//
//===----------------------------------------------------------------------===//

#include "RISCVESP32P4FunctionSpecialization.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/IPO/LoopExtractor.h"
#include "llvm/Transforms/IPO/SCCP.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "riscv-esp32p4-function-specialization"

STATISTIC(NumFunctionsSpecialized,
          "Number of functions specialized by ESP32P4 pass");

// Command line option to enable/disable RISCVESP32P4FunctionSpecialization
cl::opt<bool> llvm::EnableRISCVESP32P4FunctionSpecialization(
    "riscv-esp32p4-function-specialization", cl::init(false),
    cl::desc("Enable RISC-V ESP32P4 function specialization for specific "
             "functions"));

namespace {

//===----------------------------------------------------------------------===//
// Constants and Configuration
//===----------------------------------------------------------------------===//

/// FIRMR struct layout constants (byte offsets from struct base)
namespace FIRMROffsets {
static constexpr int32_t KSHIFT = 16;        ///< Shift field offset
static constexpr int32_t KROUNDING_VAL = 24; ///< Rounding value field offset
static constexpr int32_t KINTERP = 32;       ///< Interpolation field offset
static constexpr int32_t KDELAY_SIZE = 30;   ///< Delay size field offset
static constexpr int32_t KSTART_POS = 36;    ///< Start position field offset
static constexpr int32_t KDELAY = 4;         ///< Delay pointer field offset
static constexpr int32_t KPOS = 10;          ///< Position field offset
static constexpr int32_t KDECIM = 12;        ///< Decimation field offset
} // namespace FIRMROffsets

/// Specialization control parameters
namespace SpecConstants {
static constexpr int32_t KTHRESHOLD =
    15; ///< Shift value threshold for specialization
static constexpr int32_t KMAX_TRACE_DEPTH = 4; ///< Max depth for value tracing
static constexpr int32_t KINTERP_VALUE = 1;    ///< Target interpolation value
} // namespace SpecConstants

/// Type width constants
namespace TypeWidths {
static constexpr unsigned KI16 = 16; ///< 16-bit integer width
static constexpr unsigned KI32 = 32; ///< 32-bit integer width
} // namespace TypeWidths

//===----------------------------------------------------------------------===//
// Analysis Data Structures
//===----------------------------------------------------------------------===//

/// Maps FIRMR struct field information for analysis
struct FIRMRFieldMapping {
  int32_t Offset;      ///< Byte offset in struct
  Value **ResultPtr;   ///< Pointer to store found GEP instruction
  bool IsRequired;     ///< Whether this field is required for specialization
  StringRef FieldName; ///< Human-readable field name for debugging

  FIRMRFieldMapping(int32_t O, Value **R, bool Req, StringRef Name)
      : Offset(O), ResultPtr(R), IsRequired(Req), FieldName(Name) {}
};

/// Analysis result for FIRMR function specialization candidates
struct FIRMRAnalysisResult {
  bool IsSpecializable = false;

  // FIRMR-specific struct field pointers
  Value *InterpPtr = nullptr;
  Value *DelaySizePtr = nullptr;
  Value *StartPosPtr = nullptr;
  Value *ShiftPtr = nullptr;

  // Key instructions for specialization
  LoadInst *ShiftLoad = nullptr;
  ICmpInst *SpecializationCond = nullptr;
  BranchInst *SpecializationBranch = nullptr;

  /// Validates that all required fields have been found
  bool hasAllRequiredFields() const {
    return InterpPtr && DelaySizePtr && StartPosPtr && ShiftPtr;
  }

  /// Validates that specialization instructions have been found
  bool hasSpecializationInstructions() const {
    return ShiftLoad && SpecializationCond && SpecializationBranch;
  }

  /// Returns a diagnostic message for missing fields
  std::string getMissingFieldsDiagnostic() const {
    std::string Missing;
    if (!InterpPtr)
      Missing += "interp ";
    if (!DelaySizePtr)
      Missing += "delaySize ";
    if (!StartPosPtr)
      Missing += "startPos ";
    if (!ShiftPtr)
      Missing += "shift ";
    return Missing.empty() ? "none" : Missing;
  }

  /// Resets all fields to initial state
  void reset() {
    IsSpecializable = false;
    InterpPtr = DelaySizePtr = StartPosPtr = ShiftPtr = nullptr;
    ShiftLoad = nullptr;
    SpecializationCond = nullptr;
    SpecializationBranch = nullptr;
  }
};

/// Sets up analysis managers with proper cross-registration.
static void setupAnalysisManagers(LoopAnalysisManager &LAM,
                                  FunctionAnalysisManager &FAM,
                                  CGSCCAnalysisManager &CGAM,
                                  ModuleAnalysisManager &MAM) {
  PassBuilder PB;
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
}

//===----------------------------------------------------------------------===//
// Forward Declarations
//===----------------------------------------------------------------------===//

class AnalysisErrorReporter;
class BlockSpecializationValidator;

//===----------------------------------------------------------------------===//
// Utility Functions (need to be declared before use)
//===----------------------------------------------------------------------===//

/// Helper to match GEP instructions for specific field offsets.
/// \param I The instruction to check
/// \param Offset The expected byte offset
/// \returns The GEP instruction if it matches, nullptr otherwise
static GetElementPtrInst *matchFIRMRFieldAccess(Instruction *I,
                                                int32_t Offset) {
  // Quick type check first
  auto *GEP = dyn_cast<GetElementPtrInst>(I);
  if (LLVM_UNLIKELY(!GEP || GEP->getNumOperands() != 2))
    return nullptr;

  if (auto *CI = dyn_cast<ConstantInt>(GEP->getOperand(1))) {
    if (LLVM_LIKELY(CI->getSExtValue() == Offset))
      return GEP;
  }
  return nullptr;
}

/// Traces through transparent instructions to find the original value.
static Value *traceValueThroughTransparentInsts(
    Value *V, int MaxDepth = SpecConstants::KMAX_TRACE_DEPTH) {
  Value *CurrentVal = V;
  for (int i = 0; i < MaxDepth; ++i) {
    if (auto *Freeze = dyn_cast<FreezeInst>(CurrentVal)) {
      CurrentVal = Freeze->getOperand(0);
    } else if (auto *Cast = dyn_cast<CastInst>(CurrentVal)) {
      CurrentVal = Cast->getOperand(0);
    } else {
      break;
    }
  }
  return CurrentVal;
}

/// Finds a load instruction from the given pointer value.
static LoadInst *findLoadFromPointer(Value *Ptr) {
  for (User *U : Ptr->users()) {
    if (auto *LI = dyn_cast<LoadInst>(U))
      return LI;
  }
  return nullptr;
}

/// Finds a branch instruction that uses the given value.
static BranchInst *findBranchUsingValue(Value *V) {
  for (User *U : V->users()) {
    if (auto *BI = dyn_cast<BranchInst>(U))
      return BI;
  }
  return nullptr;
}

/// Checks if a basic block is in the false path of a conditional branch.
static bool isInFalsePath(BasicBlock *BB) {
  for (auto PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI) {
    BasicBlock *Pred = *PI;
    if (auto *BI = dyn_cast<BranchInst>(Pred->getTerminator())) {
      if (BI->isConditional() && BI->getSuccessor(1) == BB)
        return true;
    }
  }
  return false;
}

/// Finds the load instruction for the interp field in the appropriate block.
static LoadInst *findInterpLoad(Value *InterpPtr) {
  LoadInst *CandidateLoad = nullptr;

  for (User *U : InterpPtr->users()) {
    auto *LI = dyn_cast<LoadInst>(U);
    if (!LI)
      continue;

    BasicBlock *LoadBB = LI->getParent();

    // Prioritize loads in for.body.preheader blocks
    if (LoadBB->getName().contains("for.body.preheader"))
      return LI;

    // Consider loads in false path as candidates
    if (isInFalsePath(LoadBB)) {
      CandidateLoad = LI;
    }
  }

  return CandidateLoad;
}

/// Validates and finds the sign extension from i16 to i32.
static SExtInst *findAndValidateSextInstruction(LoadInst *InterpLoad) {
  if (!InterpLoad || !InterpLoad->getType()->isIntegerTy(TypeWidths::KI16)) {
    LLVM_DEBUG(dbgs() << "FS: InterpLoad is not i16 type.\n");
    return nullptr;
  }

  for (User *U : InterpLoad->users()) {
    if (auto *SI = dyn_cast<SExtInst>(U)) {
      if (SI->getSrcTy()->isIntegerTy(TypeWidths::KI16) &&
          SI->getDestTy()->isIntegerTy(TypeWidths::KI32))
        return SI;
    }
  }
  return nullptr;
}

/// Creates the specialization condition for interp field.
static Value *createInterpSpecializationCondition(Value *SextInst,
                                                  BasicBlock *InsertBB) {
  if (!SextInst || !InsertBB)
    return nullptr;

  Instruction *InsertionPoint = InsertBB->getTerminator();
  if (!InsertionPoint)
    return nullptr;

  IRBuilder<> Builder(InsertionPoint);
  return Builder.CreateICmpEQ(
      SextInst,
      ConstantInt::get(SextInst->getType(), SpecConstants::KINTERP_VALUE),
      "cmp.interp");
}

/// Identifies instructions whose values escape the given block.
static void
collectEscapingValues(BasicBlock *BB,
                      SmallVectorImpl<Instruction *> &EscapingValues) {
  for (Instruction &I : *BB) {
    for (User *U : I.users()) {
      if (cast<Instruction>(U)->getParent() != BB) {
        EscapingValues.push_back(&I);
        break;
      }
    }
  }
}

/// Clones instructions from source block to target block with given suffix.
static void cloneInstructionsToBlock(BasicBlock *SourceBB, BasicBlock *TargetBB,
                                     ValueToValueMapTy &VMap,
                                     StringRef Suffix) {
  IRBuilder<> Builder(TargetBB);
  for (Instruction &I : *SourceBB) {
    Instruction *NewInst = I.clone();
    if (!NewInst->getType()->isVoidTy())
      NewInst->setName(I.getName() + Suffix);

    RemapInstruction(NewInst, VMap,
                     RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);
    VMap[&I] = NewInst;
    Builder.Insert(NewInst);
  }
}

/// Creates PHI nodes for values that escape the specialized block.
static void createPhiNodesForEscapingValues(
    const SmallVectorImpl<Instruction *> &EscapingValues, BasicBlock *SuccBB,
    BasicBlock *ThenBB, BasicBlock *ElseBB, const ValueToValueMapTy &VMapThen,
    const ValueToValueMapTy &VMapElse, BasicBlock *OriginalBB) {
  IRBuilder<> Builder(SuccBB, SuccBB->begin());
  for (Instruction *EscapingValue : EscapingValues) {
    PHINode *PHI = Builder.CreatePHI(EscapingValue->getType(), 2,
                                     EscapingValue->getName() + ".phi");

    Value *ThenValue = VMapThen.lookup(EscapingValue);
    Value *ElseValue = VMapElse.lookup(EscapingValue);

    PHI->addIncoming(ThenValue, ThenBB);
    PHI->addIncoming(ElseValue, ElseBB);

    EscapingValue->replaceUsesWithIf(PHI, [OriginalBB](Use &U) {
      return cast<Instruction>(U.getUser())->getParent() != OriginalBB;
    });
  }
}

//===----------------------------------------------------------------------===//
// Analysis Error Reporter (moved up for forward declaration)
//===----------------------------------------------------------------------===//

/// RAII helper for analysis error reporting
class AnalysisErrorReporter {
  Function &F;
  StringRef PassName;
  bool HasErrors = false;

public:
  AnalysisErrorReporter(Function &Func, StringRef Name)
      : F(Func), PassName(Name) {}

  ~AnalysisErrorReporter() {
    if (HasErrors) {
      LLVM_DEBUG(dbgs() << PassName << ": Analysis failed for function "
                        << F.getName() << "\n");
    }
  }

  void reportError(StringRef Message) {
    HasErrors = true;
    LLVM_DEBUG(dbgs() << PassName << ": " << Message << "\n");
  }

  void reportInfo(StringRef Message) {
    LLVM_DEBUG(dbgs() << PassName << ": " << Message << "\n");
  }

  bool hasErrors() const { return HasErrors; }
};

//===----------------------------------------------------------------------===//
// Block Specialization Utilities
//===----------------------------------------------------------------------===//

/// Enhanced block validation with better error reporting
class BlockSpecializationValidator {
  AnalysisErrorReporter &Reporter;

public:
  explicit BlockSpecializationValidator(AnalysisErrorReporter &R)
      : Reporter(R) {}

  std::pair<BasicBlock *, BasicBlock *> validateBlock(BasicBlock *BB) {
    if (!BB) {
      Reporter.reportError("Block is null");
      return {nullptr, nullptr};
    }

    BasicBlock *PredBB = BB->getSinglePredecessor();
    if (!PredBB) {
      Reporter.reportError("Block '" + BB->getName().str() +
                           "' does not have a single predecessor");
      return {nullptr, nullptr};
    }

    BasicBlock *SuccBB = BB->getSingleSuccessor();
    if (!SuccBB) {
      Reporter.reportError("Block '" + BB->getName().str() +
                           "' does not have a single successor");
      return {nullptr, nullptr};
    }

    Instruction *Terminator = PredBB->getTerminator();
    if (!isa<BranchInst>(Terminator) ||
        cast<BranchInst>(Terminator)->isConditional()) {
      Reporter.reportError("Predecessor '" + PredBB->getName().str() +
                           "' does not end with an unconditional branch");
      return {nullptr, nullptr};
    }

    return {PredBB, SuccBB};
  }
};

/// Core transformation function for block specialization.
/// Creates specialized versions of a basic block based on a condition.
static std::pair<BasicBlock *, BasicBlock *>
specializeBlockOnCondition(Value *Condition, BasicBlock *BB,
                           DominatorTree &DT) {
  // Create a temporary error reporter for this operation
  Function *F = BB->getParent();
  AnalysisErrorReporter TempReporter(*F, "BlockSpecialization");
  BlockSpecializationValidator Validator(TempReporter);

  // Validate preconditions using the enhanced validator
  auto [PredBB, SuccBB] = Validator.validateBlock(BB);
  if (!PredBB || !SuccBB)
    return {nullptr, nullptr};

  // Identify values that escape the block and need PHI nodes
  SmallVector<Instruction *, 8> EscapingValues;
  collectEscapingValues(BB, EscapingValues);

  // Create specialized blocks
  BasicBlock *ThenBB =
      BasicBlock::Create(F->getContext(), BB->getName() + ".then", F, SuccBB);
  BasicBlock *ElseBB =
      BasicBlock::Create(F->getContext(), BB->getName() + ".else", F, SuccBB);

  // Replace predecessor's terminator with conditional branch
  PredBB->getTerminator()->eraseFromParent();
  IRBuilder<> Builder(PredBB);
  Builder.CreateCondBr(Condition, ThenBB, ElseBB);

  // Clone instructions into both specialized blocks
  ValueToValueMapTy VMapThen, VMapElse;
  cloneInstructionsToBlock(BB, ThenBB, VMapThen, ".then");
  cloneInstructionsToBlock(BB, ElseBB, VMapElse, ".else");

  // Create PHI nodes for escaping values
  createPhiNodesForEscapingValues(EscapingValues, SuccBB, ThenBB, ElseBB,
                                  VMapThen, VMapElse, BB);

  // Clean up PHI nodes in successor
  for (PHINode &PHI : SuccBB->phis()) {
    int Idx = PHI.getBasicBlockIndex(BB);
    if (Idx != -1)
      PHI.removeIncomingValue(Idx);
  }

  // Remove the original block
  BB->eraseFromParent();

  // Update dominator tree
  DT.addNewBlock(ThenBB, PredBB);
  DT.addNewBlock(ElseBB, PredBB);
  DT.changeImmediateDominator(SuccBB, PredBB);

  TempReporter.reportInfo("Successfully created specialized blocks");
  return {ThenBB, ElseBB};
}

//===----------------------------------------------------------------------===//
// Analysis Implementation
//===----------------------------------------------------------------------===//

/// Analysis pass to identify functions suitable for FIRMR specialization.
class FIRMRFunctionAnalysis {
  AnalysisErrorReporter Reporter;

public:
  explicit FIRMRFunctionAnalysis(Function &F)
      : Reporter(F, "FIRMRFunctionAnalysis") {}

  bool run(Function &F, FIRMRAnalysisResult &Result);

private:
  bool identifyFIRMRStructAccess(Function &F, FIRMRAnalysisResult &Result);
  bool findSpecializationCondition(Function &F, FIRMRAnalysisResult &Result);
  ICmpInst *findSpecializationCompare(Function &F, LoadInst *ShiftLoad);
};

bool FIRMRFunctionAnalysis::identifyFIRMRStructAccess(
    Function &F, FIRMRAnalysisResult &Result) {
  Reporter.reportInfo("Identifying FIRMR struct field access patterns in " +
                      F.getName().str());

  // Define field mappings for easier iteration
  FIRMRFieldMapping FieldMappings[] = {
      {FIRMROffsets::KINTERP, &Result.InterpPtr, true, "interp"},
      {FIRMROffsets::KDELAY_SIZE, &Result.DelaySizePtr, true, "delaySize"},
      {FIRMROffsets::KSTART_POS, &Result.StartPosPtr, true, "startPos"},
      {FIRMROffsets::KSHIFT, &Result.ShiftPtr, false, "shift"}};

  bool FoundFIRMRSignature = false;

  for (Instruction &I : instructions(F)) {
    auto *GEP = dyn_cast<GetElementPtrInst>(&I);
    if (!GEP)
      continue;

    for (auto &Mapping : FieldMappings) {
      if (auto *FieldGEP = matchFIRMRFieldAccess(GEP, Mapping.Offset)) {
        *(Mapping.ResultPtr) = FieldGEP;
        if (Mapping.IsRequired)
          FoundFIRMRSignature = true;
        Reporter.reportInfo("Found " + Mapping.FieldName.str() +
                            " field access");
        break;
      }
    }
  }

  if (!FoundFIRMRSignature) {
    Reporter.reportError(
        "Did not find FIRMR-specific field access. Skipping function.");
    return false;
  }

  if (!Result.hasAllRequiredFields()) {
    Reporter.reportError("Missing FIRMR fields: " +
                         Result.getMissingFieldsDiagnostic());
    return false;
  }

  Reporter.reportInfo("Successfully identified all FIRMR fields");
  return true;
}

ICmpInst *
FIRMRFunctionAnalysis::findSpecializationCompare(Function &F,
                                                 LoadInst *ShiftLoad) {
  // Search for the specialization condition: icmp sgt %val, 15
  for (Instruction &I : instructions(F)) {
    auto *Cmp = dyn_cast<ICmpInst>(&I);
    if (!Cmp || Cmp->getPredicate() != ICmpInst::ICMP_SGT)
      continue;

    auto *Const = dyn_cast<ConstantInt>(Cmp->getOperand(1));
    if (!Const || Const->getSExtValue() != SpecConstants::KTHRESHOLD)
      continue;

    // Trace back through transparent instructions to find the source
    Value *TracedValue = traceValueThroughTransparentInsts(Cmp->getOperand(0));

    if (TracedValue == ShiftLoad) {
      // Change predicate to avoid incorrect optimization when shift == 15
      Cmp->setPredicate(ICmpInst::ICMP_SGE);
      return Cmp;
    }
  }
  return nullptr;
}

bool FIRMRFunctionAnalysis::findSpecializationCondition(
    Function &F, FIRMRAnalysisResult &Result) {
  if (!Result.ShiftPtr)
    return false;

  // Find the load from the shift pointer
  Result.ShiftLoad = findLoadFromPointer(Result.ShiftPtr);
  if (!Result.ShiftLoad) {
    Reporter.reportError("Could not find load from shift pointer.");
    return false;
  }

  // Find the specialization compare instruction
  Result.SpecializationCond = findSpecializationCompare(F, Result.ShiftLoad);
  if (!Result.SpecializationCond) {
    Reporter.reportError("Could not find specialization condition.");
    return false;
  }

  // Find the branch that uses the compare
  Result.SpecializationBranch = findBranchUsingValue(Result.SpecializationCond);
  if (!Result.SpecializationBranch) {
    Reporter.reportError("Could not find specialization branch.");
    return false;
  }

  Reporter.reportInfo("Found specialization condition and branch.");
  return true;
}

bool FIRMRFunctionAnalysis::run(Function &F, FIRMRAnalysisResult &Result) {
  if (!identifyFIRMRStructAccess(F, Result))
    return false;

  if (!findSpecializationCondition(F, Result))
    return false;

  Result.IsSpecializable = true;
  return true;
}

//===----------------------------------------------------------------------===//
// Performance and Metrics Tracking
//===----------------------------------------------------------------------===//

/// Tracks detailed metrics about the specialization process
struct SpecializationMetrics {
  unsigned FunctionsAnalyzed = 0;
  unsigned FunctionsSpecialized = 0;
  unsigned BlocksSpecialized = 0;
  unsigned InstructionsCloned = 0;

  void recordAnalyzed() { ++FunctionsAnalyzed; }
  void recordSpecialized() { ++FunctionsSpecialized; }
  void recordBlockSpecialized() { ++BlocksSpecialized; }
  void recordInstructionCloned() { ++InstructionsCloned; }

  void dump() const {
    LLVM_DEBUG(dbgs() << "=== ESP32P4 Function Specialization Metrics ===\n"
                      << "Functions Analyzed: " << FunctionsAnalyzed << "\n"
                      << "Functions Specialized: " << FunctionsSpecialized
                      << "\n"
                      << "Blocks Specialized: " << BlocksSpecialized << "\n"
                      << "Instructions Cloned: " << InstructionsCloned << "\n");
  }
};

static SpecializationMetrics GlobalMetrics;

//===----------------------------------------------------------------------===//
// Enhanced Error Recovery and Validation
//===----------------------------------------------------------------------===//

/// Enhanced analysis error reporter with recovery suggestions
class EnhancedAnalysisErrorReporter {
  Function &F;
  StringRef PassName;
  bool HasErrors = false;
  SmallVector<std::string, 4> ErrorMessages;
  SmallVector<std::string, 4> Suggestions;

public:
  EnhancedAnalysisErrorReporter(Function &Func, StringRef Name)
      : F(Func), PassName(Name) {}

  ~EnhancedAnalysisErrorReporter() {
    if (HasErrors) {
      LLVM_DEBUG({
        dbgs() << PassName << ": Analysis failed for function " << F.getName()
               << "\n";
        dbgs() << "Errors encountered:\n";
        for (const auto &Error : ErrorMessages) {
          dbgs() << "  - " << Error << "\n";
        }
        if (!Suggestions.empty()) {
          dbgs() << "Suggestions:\n";
          for (const auto &Suggestion : Suggestions) {
            dbgs() << "  * " << Suggestion << "\n";
          }
        }
      });
    }
  }

  void reportError(StringRef Message, StringRef Suggestion = "") {
    HasErrors = true;
    ErrorMessages.push_back(Message.str());
    if (!Suggestion.empty()) {
      Suggestions.push_back(Suggestion.str());
    }
    LLVM_DEBUG(dbgs() << PassName << ": ERROR: " << Message << "\n");
  }

  void reportWarning(StringRef Message) {
    LLVM_DEBUG(dbgs() << PassName << ": WARNING: " << Message << "\n");
  }

  void reportInfo(StringRef Message) {
    LLVM_DEBUG(dbgs() << PassName << ": " << Message << "\n");
  }

  bool hasErrors() const { return HasErrors; }

  void addRecoverySuggestion(StringRef Suggestion) {
    Suggestions.push_back(Suggestion.str());
  }
};

//===----------------------------------------------------------------------===//
// Optimized Specialization Implementation
//===----------------------------------------------------------------------===//

/// Enhanced function specializer with performance optimizations
class OptimizedFIRMRFunctionSpecializer {
  Function &F;
  FunctionAnalysisManager &FAM;
  EnhancedAnalysisErrorReporter Reporter;

public:
  OptimizedFIRMRFunctionSpecializer(Function &Func, FunctionAnalysisManager &AM)
      : F(Func), FAM(AM), Reporter(Func, "OptimizedFIRMRSpecializer") {}

  /// Performs optimized specialization with early exits and caching
  bool specialize(const FIRMRAnalysisResult &AnalysisResult) {
    Reporter.reportInfo("Starting optimized FIRMR function specialization");

    // Fast validation path
    if (LLVM_UNLIKELY(!AnalysisResult.IsSpecializable)) {
      Reporter.reportError("Function is not marked as specializable",
                           "Check FIRMR struct pattern matching");
      return false;
    }

    // Pre-validate critical components
    if (LLVM_UNLIKELY(!preValidateComponents(AnalysisResult))) {
      return false;
    }

    // Perform the actual specialization
    bool Success = performOptimizedSpecialization(AnalysisResult);

    if (Success) {
      GlobalMetrics.recordSpecialized();
      Reporter.reportInfo(
          "Successfully completed optimized FIRMR specialization");
    }

    return Success;
  }

private:
  bool preValidateComponents(const FIRMRAnalysisResult &AnalysisResult) {
    if (!AnalysisResult.hasAllRequiredFields()) {
      Reporter.reportError(
          "Missing required FIRMR fields: " +
              AnalysisResult.getMissingFieldsDiagnostic(),
          "Ensure function matches expected FIRMR struct layout");
      return false;
    }

    if (!AnalysisResult.hasSpecializationInstructions()) {
      Reporter.reportError(
          "Missing specialization instructions",
          "Function may not follow expected control flow pattern");
      return false;
    }

    return true;
  }

  bool
  performOptimizedSpecialization(const FIRMRAnalysisResult &AnalysisResult) {
    // Cache frequently accessed values
    LoadInst *InterpLoad = findInterpLoad(AnalysisResult.InterpPtr);
    if (LLVM_UNLIKELY(!InterpLoad)) {
      Reporter.reportError(
          "Could not find interp load instruction",
          "Check if function contains expected interp field access");
      return false;
    }

    SExtInst *SextInst = findAndValidateSextInstruction(InterpLoad);
    if (LLVM_UNLIKELY(!SextInst)) {
      Reporter.reportError("Could not find valid sext instruction",
                           "Verify i16 to i32 sign extension pattern");
      return false;
    }

    // Perform fast block validation and specialization
    return performFastBlockSpecialization(SextInst, InterpLoad);
  }

  bool performFastBlockSpecialization(SExtInst *SextInst,
                                      LoadInst *InterpLoad) {
    BasicBlock *LoadBB = InterpLoad->getParent();
    BasicBlock *TargetBB = LoadBB->getSingleSuccessor();

    if (LLVM_UNLIKELY(!TargetBB)) {
      Reporter.reportError("Block structure not suitable for specialization",
                           "Ensure single successor relationship");
      return false;
    }

    // Create specialization condition
    Value *CmpInterp = createInterpSpecializationCondition(SextInst, LoadBB);
    if (LLVM_UNLIKELY(!CmpInterp)) {
      Reporter.reportError("Failed to create specialization condition");
      return false;
    }

    // Perform block specialization with metrics
    DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
    auto [ThenBB, ElseBB] =
        specializeBlockOnConditionWithMetrics(CmpInterp, TargetBB, DT);

    if (LLVM_LIKELY(ThenBB && ElseBB)) {
      GlobalMetrics.recordBlockSpecialized();
      Reporter.reportInfo("Block specialization completed successfully");
      return true;
    }

    Reporter.reportError("Block specialization failed");
    return false;
  }

  std::pair<BasicBlock *, BasicBlock *>
  specializeBlockOnConditionWithMetrics(Value *Condition, BasicBlock *BB,
                                        DominatorTree &DT) {
    // Use the existing function but with metrics tracking
    auto Result = specializeBlockOnCondition(Condition, BB, DT);
    if (Result.first && Result.second) {
      GlobalMetrics.recordBlockSpecialized();
    }
    return Result;
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Implementation (outside anonymous namespace)
//===----------------------------------------------------------------------===//

void RISCVESP32P4FunctionSpecializationPass::runLoopExtractor(Function &F) {
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  setupAnalysisManagers(LAM, FAM, CGAM, MAM);

  ModulePassManager MPM;
  MPM.addPass(LoopExtractorPass());
  MPM.run(*(F.getParent()), MAM);
}

PreservedAnalyses
RISCVESP32P4FunctionSpecializationPass::run(Function &F,
                                            FunctionAnalysisManager &FAM) {
  if (!EnableRISCVESP32P4FunctionSpecialization)
    return PreservedAnalyses::all();

  // Record that we analyzed this function
  GlobalMetrics.recordAnalyzed();

  EnhancedAnalysisErrorReporter PassReporter(F, "RISCVESP32P4Pass");
  PassReporter.reportInfo(
      "Starting enhanced specialization analysis for function " +
      F.getName().str());

  FIRMRAnalysisResult AnalysisResult;
  FIRMRFunctionAnalysis Analysis(F);

  if (!Analysis.run(F, AnalysisResult)) {
    PassReporter.reportInfo("Function " + F.getName().str() +
                            " is not a candidate for FIRMR specialization");
    return PreservedAnalyses::all();
  }

  PassReporter.reportInfo("Function " + F.getName().str() +
                          " is suitable for specialization");

  // Run loop extractor to outline loops into functions
  runLoopExtractor(F);

  // Perform optimized specialization
  OptimizedFIRMRFunctionSpecializer Specializer(F, FAM);
  bool Changed = Specializer.specialize(AnalysisResult);

  if (Changed) {
    ++NumFunctionsSpecialized;
    PassReporter.reportInfo("Successfully specialized function " +
                            F.getName().str());

    // Dump metrics periodically (every 10 specializations)
    if (GlobalMetrics.FunctionsSpecialized % 10 == 0) {
      GlobalMetrics.dump();
    }

    return PreservedAnalyses::none();
  }

  PassReporter.reportInfo(
      "Specialization completed without changes for function " +
      F.getName().str());
  return PreservedAnalyses::all();
}

//===----------------------------------------------------------------------===//
// ForceSpecializationWrapperPass Implementation
//===----------------------------------------------------------------------===//

ForceSpecializationWrapperPass::ScopedFlagSetter::ScopedFlagSetter(
    cl::opt<bool> *O)
    : Opt(O), OldValue(false), WasSet(false) {
  if (Opt) {
    OldValue = *Opt;
    *Opt = true;
    WasSet = true;
  }
}

ForceSpecializationWrapperPass::ScopedFlagSetter::~ScopedFlagSetter() {
  if (WasSet) {
    *Opt = OldValue;
  }
}

PreservedAnalyses
ForceSpecializationWrapperPass::run(Module &M, ModuleAnalysisManager &AM) {
  // Find the cl::opt by its string name.
  cl::opt<bool> *ForceSpecOpt = nullptr;
  auto &RegisteredOptions = cl::getRegisteredOptions();
  auto It = RegisteredOptions.find("force-specialization");
  if (It != RegisteredOptions.end()) {
    ForceSpecOpt = static_cast<cl::opt<bool> *>(It->getValue());
  }

  // Use RAII helper to ensure the flag is restored.
  ScopedFlagSetter FlagSetter(ForceSpecOpt);

  // Run the actual IPSCCP pass with function specialization enabled.
  IPSCCPPass ThePass(IPSCCPOptions(/*AllowFuncSpec=*/true));
  return ThePass.run(M, AM);
}
