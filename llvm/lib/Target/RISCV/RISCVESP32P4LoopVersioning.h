//===- RISCVESP32P4LoopVersioning.h - Loop Versioning Pass ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file declares the RISCVESP32P4LoopVersioningPass which performs loop
/// versioning optimizations specific to ESP32-P4 FIR (Finite Impulse Response)
/// filter implementations.
///
/// The pass analyzes loops that implement FIR filters and creates optimized
/// versions when certain conditions are met, particularly when coefficient
/// strides are known to be 1 and shift values are below a threshold.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_RISCVESP32P4LOOPVERSIONING_H
#define LLVM_TRANSFORMS_UTILS_RISCVESP32P4LOOPVERSIONING_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <utility> // For std::pair

namespace llvm {

// Forward declarations to reduce dependencies
class BasicBlock;
class BranchInst;
class Function;
class GetElementPtrInst;
class ICmpInst;
class LoadInst;
class Loop;
class ScalarEvolution;
class Value;

extern cl::opt<bool> EnableRISCVESP32P4LoopVersioning;

/// \brief Analysis result structure for FIR loop pattern recognition.
///
/// This structure contains all the information gathered during the analysis
/// phase of FIR loop patterns, including loop structure, field access patterns,
/// and optimization opportunities.
struct FIRLoopAnalysisResult {
  //===--------------------------------------------------------------------===//
  // Basic Loop Information
  //===--------------------------------------------------------------------===//

  /// \brief The main loop being analyzed.
  Loop *MainLoop = nullptr;

  /// \brief Key basic blocks of the loop structure.
  /// @{
  BasicBlock *LoopPreheader = nullptr; ///< Loop preheader block
  BasicBlock *LoopHeader = nullptr;    ///< Loop header block
  BasicBlock *LoopLatch = nullptr;     ///< Loop latch block
  BasicBlock *LoopExit = nullptr;      ///< Loop exit block
  /// @}

  //===--------------------------------------------------------------------===//
  // FIR Structure Field Access Patterns
  //===--------------------------------------------------------------------===//

  /// \brief GetElementPtr instructions accessing FIR struct fields.
  /// These represent access to various fields in the FIR filter structure.
  /// @{
  GetElementPtrInst *CoeffsLenPtr =
      nullptr; ///< Access to coefficients length field
  GetElementPtrInst *ShiftPtr = nullptr; ///< Access to shift amount field
  GetElementPtrInst *CoeffsPtr =
      nullptr; ///< Access to coefficients array pointer
  GetElementPtrInst *DelayPtr = nullptr; ///< Access to delay line pointer
  GetElementPtrInst *PosPtr = nullptr;   ///< Access to position field
  GetElementPtrInst *DPosPtr = nullptr;  ///< Access to delay position field
  GetElementPtrInst *DecimPtr = nullptr; ///< Access to decimation field
  /// @}

  //===--------------------------------------------------------------------===//
  // Loop Invariant Load Instructions
  //===--------------------------------------------------------------------===//

  /// \brief Load instructions that are loop invariant and can be optimized.
  /// @{
  LoadInst *CoeffsLenLoad = nullptr; ///< Load of coefficients length
  LoadInst *ShiftLoad = nullptr;     ///< Load of shift amount
  LoadInst *CoeffsLoad = nullptr;    ///< Load of coefficients array pointer
  LoadInst *DelayLoad = nullptr;     ///< Load of delay line pointer
  LoadInst *DecimLoad = nullptr;     ///< Load of decimation factor
  /// @}

  //===--------------------------------------------------------------------===//
  // Versioning Condition Analysis
  //===--------------------------------------------------------------------===//

  /// \brief Values and instructions related to versioning conditions.
  /// @{
  Value *FinalShift = nullptr;          ///< Final computed shift value
  ICmpInst *CoeffsLenGE16 = nullptr;    ///< Comparison: coeffs_len >= 16
  ICmpInst *FinalShiftLE0 = nullptr;    ///< Comparison: final_shift <= 0
  Value *VersioningCondition = nullptr; ///< Combined versioning condition
  /// @}

  //===--------------------------------------------------------------------===//
  // Loop Structure Information
  //===--------------------------------------------------------------------===//

  /// \brief Information about inner loops within the main loop.
  SmallVector<Loop *, 4> InnerLoops;

  /// \brief Whether this loop is suitable for versioning optimization.
  bool IsVersionable = false;

  //===--------------------------------------------------------------------===//
  // Transformation Result Information
  //===--------------------------------------------------------------------===//

  /// \brief Basic blocks created during loop versioning transformation.
  /// @{
  BasicBlock *VersioningConditionBB =
      nullptr;                           ///< Block containing runtime check
  BasicBlock *OptimizedPathBB = nullptr; ///< Entry to optimized path
  BasicBlock *OriginalPathBB = nullptr;  ///< Entry to original path
  BasicBlock *MergeBB = nullptr;         ///< Block where paths merge
  /// @}

  /// \brief Collections of basic blocks for original and optimized loops.
  /// @{
  SmallVector<BasicBlock *, 8> OriginalLoopBlocks; ///< Blocks in original loop
  SmallVector<BasicBlock *, 8>
      OptimizedLoopBlocks; ///< Blocks in optimized loop
  /// @}

  /// \brief Instructions created during versioning.
  /// @{
  BranchInst *VersioningBranch = nullptr; ///< Conditional branch for versioning
  Loop *OptimizedLoop = nullptr;          ///< The optimized loop structure
  /// @}

  //===--------------------------------------------------------------------===//
  // Constructors and Assignment
  //===--------------------------------------------------------------------===//

  /// \brief Default constructor.
  FIRLoopAnalysisResult() = default;

  /// \brief Deleted copy operations to prevent accidental copying.
  /// @{
  FIRLoopAnalysisResult(const FIRLoopAnalysisResult &) = delete;
  FIRLoopAnalysisResult &operator=(const FIRLoopAnalysisResult &) = delete;
  /// @}

  /// \brief Move operations are allowed.
  /// @{
  FIRLoopAnalysisResult(FIRLoopAnalysisResult &&) = default;
  FIRLoopAnalysisResult &operator=(FIRLoopAnalysisResult &&) = default;
  /// @}

  /// \brief Debug output method for analysis results.
  void dump() const {
    dbgs() << "=== FIRLoopAnalysisResult ===\n";
    dbgs() << "MainLoop: " << (MainLoop ? "Found" : "Not Found") << "\n";
    dbgs() << "IsVersionable: " << (IsVersionable ? "Yes" : "No") << "\n";
    dbgs() << "VersioningConditionBB: "
           << (VersioningConditionBB ? "Created" : "Not Created") << "\n";
    dbgs() << "OptimizedPathBB: "
           << (OptimizedPathBB ? "Created" : "Not Created") << "\n";
    dbgs() << "OriginalPathBB: " << (OriginalPathBB ? "Created" : "Not Created")
           << "\n";
    if (CoeffsLenLoad) {
      dbgs() << "CoeffsLen: ";
      CoeffsLenLoad->dump();
    }
    if (ShiftLoad) {
      dbgs() << "Shift: ";
      ShiftLoad->dump();
    }
    dbgs() << "========================\n";
  }
};

/// \brief ESP32-P4 specific loop versioning pass.
///
/// This pass performs loop versioning optimizations specifically targeted at
/// FIR (Finite Impulse Response) filter implementations on ESP32-P4. It
/// analyzes loops to identify FIR filter patterns and creates optimized
/// versions when runtime conditions permit more efficient execution.
///
/// The pass operates in several phases:
/// 1. **Analysis Phase**: Identifies FIR loop patterns and structure
/// 2. **Viability Check**: Determines if versioning is beneficial
/// 3. **Transformation Phase**: Creates optimized and original loop versions
/// 4. **Optimization Phase**: Applies ESP32-P4 specific optimizations
///
/// \see FIRLoopAnalysisResult for the analysis data structure.
struct RISCVESP32P4LoopVersioningPass
    : public PassInfoMixin<RISCVESP32P4LoopVersioningPass> {
  RISCVESP32P4LoopVersioningPass() {}

  /// \brief Main pass entry point.
  /// \param F The function to analyze and potentially transform.
  /// \param AM Function analysis manager providing required analyses.
  /// \return PreservedAnalyses indicating what analyses are still valid.
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  /// \brief Indicates this pass is always required to run.
  static bool isRequired() { return true; }

  //===--------------------------------------------------------------------===//
  // Analysis Functions
  //===--------------------------------------------------------------------===//

  /// \brief Main analysis function for FIR loop patterns.
  /// \param F Function containing the loop to analyze.
  /// \param LI Loop information analysis result.
  /// \param DT Dominator tree analysis result.
  /// \param Result Output structure for analysis results.
  /// \return true if analysis succeeded and found a viable FIR pattern.
  bool analyzeFirLoop(Function &F, LoopInfo &LI, DominatorTree &DT,
                      FIRLoopAnalysisResult &Result);

  /// \brief Identify the main loop in the function.
  /// \param LI Loop information analysis result.
  /// \return Pointer to the main loop, or nullptr if not found.
  Loop *findMainLoop(LoopInfo &LI);

  /// \brief Extract and validate loop structure information.
  /// \param L The loop to analyze.
  /// \param Result Output structure for storing loop structure info.
  /// \return true if loop structure is suitable for optimization.
  bool extractLoopStructure(Loop *L, FIRLoopAnalysisResult &Result);

  /// \brief Find FIR struct field access patterns in the function.
  /// \param F Function to search for field access patterns.
  /// \param Result Output structure for storing field access information.
  /// \return true if essential FIR field accesses were found.
  bool findFirStructAccess(Function &F, FIRLoopAnalysisResult &Result);

  /// \brief Extract loop invariant values from the loop.
  /// \param L The loop to analyze for invariants.
  /// \param Result Output structure for storing invariant information.
  /// \return true if useful loop invariants were found.
  bool extractLoopInvariants(Loop *L, FIRLoopAnalysisResult &Result);

  /// \brief Check if the analyzed loop is suitable for versioning.
  /// \param Result Analysis results to evaluate.
  /// \return true if loop versioning would be beneficial.
  bool isVersioningViable(const FIRLoopAnalysisResult &Result);

  /// \brief Build the runtime condition for loop versioning.
  /// \param Result Analysis results containing condition components.
  /// \return true if versioning condition was successfully built.
  bool buildVersioningCondition(FIRLoopAnalysisResult &Result);

  //===--------------------------------------------------------------------===//
  // Helper Methods
  //===--------------------------------------------------------------------===//

  /// \brief Match a GEP instruction against a specific FIR field offset.
  /// \param I Instruction to check (should be a GEP).
  /// \param Offset Expected field offset in bytes.
  /// \return GEP instruction if it matches the pattern, nullptr otherwise.
  GetElementPtrInst *matchFirFieldAccess(Instruction *I, int32_t Offset);

  /// \brief Check if a value is loop invariant.
  /// \param V Value to check.
  /// \param L Loop context for invariant check.
  /// \return true if value is loop invariant.
  bool isLoopInvariant(Value *V, Loop *L);

  /// \brief Print analysis results for debugging.
  /// \param Result Analysis results to print.
  void printAnalysisResults(const FIRLoopAnalysisResult &Result);

  /// \brief Print field identification summary for debugging.
  /// \param F Function being analyzed.
  /// \param Result Analysis results.
  /// \param FoundEssentialFields Whether essential fields were found.
  void printFieldIdentificationSummary(const Function &F,
                                       const FIRLoopAnalysisResult &Result,
                                       bool FoundEssentialFields);

  /// \brief Collect basic blocks for invariant analysis.
  /// \param L Loop to analyze.
  /// \return Vector of blocks to scan for invariants.
  SmallVector<BasicBlock *, 16> collectBlocksForInvariantAnalysis(Loop *L);

  /// \brief Find loads from FIR struct fields.
  /// \param BlocksToScan Blocks to search for load instructions.
  /// \param Result Output structure for storing found loads.
  /// \return true if any relevant loads were found.
  bool findStructFieldLoads(const SmallVector<BasicBlock *, 16> &BlocksToScan,
                            FIRLoopAnalysisResult &Result);

  /// \brief Validate that found loads are actually loop invariants.
  /// \param L Loop context for validation.
  /// \param Result Analysis results containing loads to validate.
  /// \return true if invariants are valid.
  bool validateLoopInvariants(Loop *L, const FIRLoopAnalysisResult &Result);

  //===--------------------------------------------------------------------===//
  // Transformation Functions
  //===--------------------------------------------------------------------===//

  /// \brief Perform the main loop versioning transformation.
  /// \param F Function containing the loop.
  /// \param Result Analysis results guiding the transformation.
  /// \param LI Loop information (will be updated).
  /// \param DT Dominator tree (will be updated).
  /// \param SE Scalar evolution analysis.
  /// \return true if transformation was successful.
  bool performLoopVersioning(Function &F, FIRLoopAnalysisResult &Result,
                             LoopInfo &LI, DominatorTree &DT,
                             ScalarEvolution &SE);

  /// \brief Create runtime check block for loop versioning.
  /// \param OrigLoop Original loop to version.
  /// \param LI Loop information.
  /// \param DT Dominator tree.
  /// \return New preheader block for the slow path.
  BasicBlock *createRuntimeCheckBlock(Loop *OrigLoop, LoopInfo &LI,
                                      DominatorTree &DT);

  /// \brief Clone loop to create fast path version.
  /// \param OrigLoop Original loop to clone.
  /// \param NewOrigPreheader New preheader for slow path.
  /// \param VMap Value mapping from original to cloned instructions.
  /// \param LI Loop information.
  /// \param DT Dominator tree.
  /// \return Cloned loop for fast path.
  Loop *cloneFastPathLoop(Loop *OrigLoop, BasicBlock *NewOrigPreheader,
                          ValueToValueMapTy &VMap, LoopInfo &LI,
                          DominatorTree &DT);

  /// \brief Create versioning condition and conditional branch.
  /// \param CheckBlock Block where runtime check is inserted.
  /// \param FastLoop Fast path loop.
  /// \param SlowPreheader Slow path preheader.
  /// \param Result Analysis results.
  /// \return true if condition was successfully created.
  bool createVersioningCondition(BasicBlock *CheckBlock, Loop *FastLoop,
                                 BasicBlock *SlowPreheader,
                                 FIRLoopAnalysisResult &Result);

  /// \brief Fix PHI nodes in shared exit block.
  /// \param OrigLoop Original loop.
  /// \param FastLoop Fast path loop.
  /// \param ExitBlock Shared exit block.
  /// \param VMap Value mapping.
  void fixExitPhiNodes(Loop *OrigLoop, Loop *FastLoop, BasicBlock *ExitBlock,
                       ValueToValueMapTy &VMap);

  /// \brief Update analysis information after transformation.
  /// \param F Function that was transformed.
  /// \param LI Loop information to update.
  /// \param DT Dominator tree to update.
  void updateAnalysisInfo(Function &F, LoopInfo &LI, DominatorTree &DT);

  //===--------------------------------------------------------------------===//
  // Optimization Functions
  //===--------------------------------------------------------------------===//

  /// \brief Insert coefficient reversal and adjust indexing for optimization.
  /// \param OptLoop Optimized loop to transform.
  /// \param VMap Value mapping.
  /// \param Result Analysis results.
  void insertCoeffReversalAndAdjustIndexing(Loop *OptLoop,
                                            ValueToValueMapTy &VMap,
                                            FIRLoopAnalysisResult &Result);

  /// \brief Get mapped coefficient values from VMap or fallback to originals.
  /// \param VMap Value mapping from loop cloning.
  /// \param Result Analysis results.
  /// \param OptLoop Optimized loop context.
  /// \return Pair of coefficient pointer and length value.
  std::pair<Value *, Value *>
  getMappedCoeffValues(ValueToValueMapTy &VMap,
                       const FIRLoopAnalysisResult &Result, Loop *OptLoop);

  /// \brief Insert coefficient reversal function call.
  /// \param OptPreheader Preheader of optimized loop.
  /// \param CoeffsPtr Pointer to coefficients array.
  /// \param CoeffsLenValue Length of coefficients array.
  /// \return true if call was successfully inserted.
  bool insertCoeffReversalCall(BasicBlock *OptPreheader, Value *CoeffsPtr,
                               Value *CoeffsLenValue);

  /// \brief Handle additions involving coeffs_len in the optimized loop.
  /// \param OptLoop Optimized loop.
  /// \param VMap Value mapping.
  /// \param Result Analysis results.
  void handleCoeffsLenAdditions(Loop *OptLoop, ValueToValueMapTy &VMap,
                                FIRLoopAnalysisResult &Result);

  //===--------------------------------------------------------------------===//
  // Utility Functions
  //===--------------------------------------------------------------------===//

  /// \brief Check if load instruction directly loads FIR coefficients.
  /// \param Load Load instruction to check.
  /// \return true if this is a direct FIR coefficients load.
  bool isDirectFirCoeffsLoad(LoadInst *Load);

  /// \brief Verify transformation results for correctness.
  /// \param F Transformed function.
  /// \param Result Transformation results.
  /// \return true if transformation appears correct.
  bool verifyTransformation(Function &F, const FIRLoopAnalysisResult &Result);

  /// \brief Apply FIR-specific optimizations to the optimized loop.
  /// \param OptLoop Optimized loop.
  /// \param VMap Value mapping.
  /// \param Result Analysis results.
  void applyFirOptimizations(Loop *OptLoop, ValueToValueMapTy &VMap,
                             FIRLoopAnalysisResult &Result);

  /// \brief Handle exit PHI nodes after loop versioning.
  /// \param OrigLoop Original loop.
  /// \param OptLoop Optimized loop.
  /// \param ExitBlock Exit block.
  /// \param VMap Value mapping.
  void handleExitPhiNodes(Loop *OrigLoop, Loop *OptLoop, BasicBlock *ExitBlock,
                          ValueToValueMapTy &VMap);

private:
  //===--------------------------------------------------------------------===//
  // Private Helper Functions
  //===--------------------------------------------------------------------===//

  /// \brief Find dominating position load using dominator tree analysis.
  /// \param LoopExitBB Loop exit basic block.
  /// \param UseInst Instruction that uses the position value.
  /// \param VMap Value mapping.
  /// \param Result Analysis results.
  /// \param DT Dominator tree for precise dominance checking.
  /// \return Dominating position load, or nullptr if not found.
  Value *findDominatingPosLoadWithDt(BasicBlock *LoopExitBB,
                                     Instruction *UseInst,
                                     ValueToValueMapTy &VMap,
                                     FIRLoopAnalysisResult &Result,
                                     DominatorTree &DT);

  /// \brief Check if one basic block can reach another.
  /// \param FromBB Source basic block.
  /// \param ToBB Target basic block.
  /// \return true if ToBB is reachable from FromBB.
  bool isReachable(BasicBlock *FromBB, BasicBlock *ToBB);

  /// \brief Fix coefficient position calculation in second loop.
  /// \param OptLoop Optimized loop.
  /// \param VMap Value mapping.
  /// \param Result Analysis results.
  /// \param DT Dominator tree.
  void fixSecondLoopCoeffPositionCalculation(Loop *OptLoop,
                                             ValueToValueMapTy &VMap,
                                             FIRLoopAnalysisResult &Result,
                                             DominatorTree &DT);

  /// \brief Replace incorrect coefficient position calculation.
  /// \param WrongCalc Incorrect calculation to replace.
  /// \param VMap Value mapping.
  /// \param Result Analysis results.
  void
  replaceWithCorrectCoeffPositionCalculation(BinaryOperator *WrongCalc,
                                             ValueToValueMapTy &VMap,
                                             FIRLoopAnalysisResult &Result);

  /// \brief Insert correct coefficient position calculation.
  /// \param LoopExitBB Loop exit basic block.
  /// \param VMap Value mapping.
  /// \param Result Analysis results.
  /// \param DT Dominator tree.
  void insertCorrectCoeffPositionCalculation(BasicBlock *LoopExitBB,
                                             ValueToValueMapTy &VMap,
                                             FIRLoopAnalysisResult &Result,
                                             DominatorTree &DT);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_RISCVESP32P4LOOPVERSIONING_H
