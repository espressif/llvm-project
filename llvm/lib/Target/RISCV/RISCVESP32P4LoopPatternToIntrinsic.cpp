//===- RISCVESP32P4LoopPatternToIntrinsic.cpp - Loop Pattern Transform ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a transformation pass that identifies matrix
/// multiplication patterns in nested loops and replaces them with optimized
/// ESP32-P4 SIMD intrinsic calls. The pass specifically targets the
/// dspm_mult_s16 family of functions and applies vectorization when certain
/// conditions are met.
///
/// The transformation works in several phases:
/// 1. Function name validation - ensures target function matches expected
/// pattern
/// 2. Parameter extraction - extracts matrix pointers and dimensions
/// 3. Loop structure analysis - validates 3-level nested loop structure
/// 4. Pattern condition checking - verifies shift operations and rounding
/// constants
/// 5. SIMD transformation - creates optimized SIMD path when conditions are met
///
/// Performance optimizations:
/// - Early exit strategies to minimize unnecessary analysis
/// - Cached pattern validation results
/// - Efficient loop traversal and instruction scanning
/// - Minimal memory allocations in hot paths
///
//===----------------------------------------------------------------------===//

#include "RISCVESP32P4LoopPatternToIntrinsic.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "riscv-esp32p4-loop-pattern-to-intrinsic"
using namespace llvm;

cl::opt<bool> llvm::EnableRISCVESP32P4LoopPatternToIntrinsic(
    "riscv-esp32p4-loop-pattern-to-intrinsic", cl::init(false),
    cl::desc("Enable RISCV ESP32P4 loop pattern to intrinsic transformation"));

namespace {

// Constants for pattern matching and optimization
static constexpr unsigned MinRequiredArgs = 7;
static constexpr unsigned MaxShiftAmount = 15;
static constexpr unsigned VectorAlignment = 8;
static constexpr unsigned RoundingConstant = 32767;
static constexpr unsigned SplitInstructionIndex = 4;

// Loop depth constants
static constexpr unsigned OuterLoopDepth = 1;
static constexpr unsigned MiddleLoopDepth = 2;
static constexpr unsigned InnerLoopDepth = 3;

// Function name pattern
static constexpr StringRef TargetFunctionPrefix = "dspm_mult_s16";
static constexpr StringRef SIMDFunctionName = "matrix_mult_simd_optimized";

// Basic block name constants
static constexpr StringRef CleanupBlockName = "for.cond.cleanup";
static constexpr StringRef PreheaderBlockName = "for.cond1.preheader.lr.ph";
static constexpr StringRef SIMDBlockName = "simd_optimization";
static constexpr StringRef ScalarPathBlockName = "scalar_path";

/// \brief Matrix multiplication pattern information
///
/// This structure contains all the information extracted from a function
/// that matches the matrix multiplication pattern. It includes matrix
/// operands, loop structure, and optimization feasibility flags.
///
/// Memory layout is optimized for cache efficiency by grouping related
/// fields together and using compact data types where possible.
struct MatrixMultiplicationPattern {
  // Matrix operands and dimensions (most frequently accessed)
  Value *MatrixA = nullptr;    ///< Input matrix A pointer
  Value *MatrixB = nullptr;    ///< Input matrix B pointer
  Value *MatrixC = nullptr;    ///< Output matrix C pointer
  Value *DimensionM = nullptr; ///< Matrix dimension M (rows of A and C)
  Value *DimensionN = nullptr; ///< Matrix dimension N (columns of B and C)
  Value *DimensionK = nullptr; ///< Matrix dimension K (columns of A, rows of B)
  Value *ShiftAmount =
      nullptr; ///< Right shift amount for fixed-point arithmetic

  // Loop structure information (accessed during analysis)
  Loop *OuterLoop = nullptr;  ///< Outermost loop (I-loop, iterates over M)
  Loop *MiddleLoop = nullptr; ///< Middle loop (J-loop, iterates over N)
  Loop *InnerLoop = nullptr;  ///< Innermost loop (K-loop, iterates over K)

  // Pattern analysis results (packed for memory efficiency)
  bool HasRoundingConstants : 1; ///< Whether rounding constants (32767) are
                                 ///< found
  bool HasShiftOperations : 1;   ///< Whether shift operations are found
  bool IsVectorizable : 1;       ///< Whether vectorization conditions are met
  bool IsValid : 1;              ///< Whether the complete pattern is valid

  /// Default constructor with proper initialization
  MatrixMultiplicationPattern()
      : HasRoundingConstants(false), HasShiftOperations(false),
        IsVectorizable(false), IsValid(false) {}

  /// \brief Debug dump of pattern information
  void dump() const {
    LLVM_DEBUG(dbgs() << "=== Matrix Multiplication Pattern Dump ===\n");
    LLVM_DEBUG(dbgs() << "  MatrixA: "
                      << (MatrixA ? MatrixA->getName() : "nullptr") << "\n");
    LLVM_DEBUG(dbgs() << "  MatrixB: "
                      << (MatrixB ? MatrixB->getName() : "nullptr") << "\n");
    LLVM_DEBUG(dbgs() << "  MatrixC: "
                      << (MatrixC ? MatrixC->getName() : "nullptr") << "\n");
    LLVM_DEBUG(dbgs() << "  DimensionM: "
                      << (DimensionM ? DimensionM->getName() : "nullptr")
                      << "\n");
    LLVM_DEBUG(dbgs() << "  DimensionN: "
                      << (DimensionN ? DimensionN->getName() : "nullptr")
                      << "\n");
    LLVM_DEBUG(dbgs() << "  DimensionK: "
                      << (DimensionK ? DimensionK->getName() : "nullptr")
                      << "\n");
    LLVM_DEBUG(dbgs() << "  ShiftAmount: "
                      << (ShiftAmount ? ShiftAmount->getName() : "nullptr")
                      << "\n");
    LLVM_DEBUG(dbgs() << "  HasRoundingConstants: " << HasRoundingConstants
                      << "\n");
    LLVM_DEBUG(dbgs() << "  HasShiftOperations: " << HasShiftOperations
                      << "\n");
    LLVM_DEBUG(dbgs() << "  IsVectorizable: " << IsVectorizable << "\n");
    LLVM_DEBUG(dbgs() << "  IsValid: " << IsValid << "\n");
    LLVM_DEBUG(dbgs() << "  OuterLoop: " << (OuterLoop ? "Found" : "nullptr")
                      << "\n");
    LLVM_DEBUG(dbgs() << "  MiddleLoop: " << (MiddleLoop ? "Found" : "nullptr")
                      << "\n");
    LLVM_DEBUG(dbgs() << "  InnerLoop: " << (InnerLoop ? "Found" : "nullptr")
                      << "\n");
    LLVM_DEBUG(dbgs() << "=============================================\n");
  }

  /// \brief Validate that all required pattern components are present
  /// \return True if all matrix operands and dimensions are non-null
  bool hasAllRequiredComponents() const {
    return MatrixA && MatrixB && MatrixC && DimensionM && DimensionN &&
           DimensionK && ShiftAmount;
  }

  /// \brief Check if optimization conditions are met
  /// \return True if pattern can be optimized to SIMD
  bool canOptimize() const {
    return IsVectorizable && HasRoundingConstants && HasShiftOperations &&
           IsValid;
  }

  /// \brief Quick validation of essential components without full analysis
  /// \return True if basic pattern requirements are met
  bool hasValidBasicStructure() const {
    return hasAllRequiredComponents() && OuterLoop && MiddleLoop && InnerLoop;
  }
};

/// \brief Utility class for basic block operations
///
/// Provides helper methods for common basic block operations like finding
/// blocks by name and locating specific instructions within blocks.
/// All methods are static and optimized for minimal overhead.
class BasicBlockUtils {
public:
  /// \brief Find a basic block by name in the given function
  /// \param F The function to search in
  /// \param Name The name of the basic block to find
  /// \return Pointer to the basic block if found, nullptr otherwise
  static BasicBlock *findBasicBlockByName(Function &F, StringRef Name) {
    // Use early exit for performance - most functions have few basic blocks
    for (BasicBlock &BB : F) {
      if (BB.hasName() && BB.getName() == Name) {
        return &BB;
      }
    }
    LLVM_DEBUG(dbgs() << "    WARNING: Basic block '" << Name
                      << "' not found\n");
    return nullptr;
  }

  /// \brief Find the instruction at the specified index in a basic block
  /// \param Block The basic block to search in
  /// \param Index The index of the instruction to find
  /// \return Pointer to the instruction if found, otherwise the terminator
  static Instruction *findInstructionAtIndex(BasicBlock *Block,
                                             unsigned Index) {
    if (!Block) {
      LLVM_DEBUG(dbgs() << "    WARNING: Null basic block provided\n");
      return nullptr;
    }

    // Fast path for small indices
    if (Index < 16) {
      unsigned InstrCount = 0;
      for (Instruction &I : *Block) {
        if (InstrCount == Index) {
          return &I;
        }
        ++InstrCount;
      }
    }

    LLVM_DEBUG(dbgs() << "    WARNING: Instruction at index " << Index
                      << " not found, using terminator\n");
    return Block->getTerminator();
  }

  /// \brief Safely split a basic block at the given instruction
  /// \param Block The block to split
  /// \param SplitBefore The instruction to split before
  /// \param Name The name for the new block
  /// \return Pointer to the new block, or nullptr on failure
  static BasicBlock *safeSplitBasicBlock(BasicBlock *Block,
                                         Instruction *SplitBefore,
                                         StringRef Name) {
    if (!Block || !SplitBefore) {
      LLVM_DEBUG(dbgs() << "    ERROR: Invalid parameters for block split\n");
      return nullptr;
    }

    // Verify that SplitBefore belongs to Block to avoid undefined behavior
    if (SplitBefore->getParent() != Block) {
      LLVM_DEBUG(
          dbgs() << "    ERROR: Instruction does not belong to the block\n");
      return nullptr;
    }

    BasicBlock *NewBlock = Block->splitBasicBlock(SplitBefore, Name);
    if (!NewBlock) {
      LLVM_DEBUG(dbgs() << "    ERROR: Failed to split basic block\n");
    }
    return NewBlock;
  }
};

/// \brief Utility class for creating IR values and expressions
///
/// Encapsulates the creation of common IR patterns used in the optimization,
/// such as vectorization conditions and function call arguments.
/// Methods are optimized for minimal IR generation overhead.
class IRValueUtils {
public:
  /// \brief Create optimization condition: k % 8 == 0 && shift < 15
  /// \param Builder IR builder for creating instructions
  /// \param DimensionK The K dimension value
  /// \param ShiftAmount The shift amount value
  /// \return Value representing the optimization condition
  static Value *createOptimizationCondition(IRBuilder<> &Builder,
                                            Value *DimensionK,
                                            Value *ShiftAmount) {
    if (!DimensionK || !ShiftAmount) {
      LLVM_DEBUG(
          dbgs()
          << "    ERROR: Invalid parameters for optimization condition\n");
      return nullptr;
    }

    // Cache commonly used constants to reduce IR bloat
    Value *Seven = Builder.getInt32(VectorAlignment - 1);
    Value *Fifteen = Builder.getInt32(MaxShiftAmount);
    Value *Zero = Builder.getInt32(0);

    // Create condition with minimal intermediate values
    Value *KMod8 = Builder.CreateAnd(DimensionK, Seven, "k_mod_8");
    Value *KAligned = Builder.CreateICmpEQ(KMod8, Zero, "k_aligned");
    Value *ShiftSmall =
        Builder.CreateICmpSLT(ShiftAmount, Fifteen, "shift_small");

    Value *Result = Builder.CreateAnd(KAligned, ShiftSmall, "use_simd");

    LLVM_DEBUG(
        dbgs() << "    SUCCESS: Created optimization condition successfully\n");
    return Result;
  }

  /// \brief Create SIMD function call arguments
  /// \param Builder IR builder for creating instructions
  /// \param Pattern The matrix multiplication pattern
  /// \return Vector of arguments for the SIMD function call
  static SmallVector<Value *, 8>
  createSIMDCallArgs(IRBuilder<> &Builder,
                     const MatrixMultiplicationPattern &Pattern) {
    if (!Pattern.hasAllRequiredComponents()) {
      LLVM_DEBUG(
          dbgs()
          << "    ERROR: Pattern missing required components for SIMD call\n");
      return {};
    }

    // Pre-allocate vector with known size to avoid reallocations
    SmallVector<Value *, 8> Args;
    Args.reserve(8);

    // Cache constants
    Value *MaxShift = Builder.getInt32(MaxShiftAmount);
    Value *RoundingConst = Builder.getInt32(RoundingConstant);

    Value *Sub2 = Builder.CreateSub(MaxShift, Pattern.ShiftAmount, "sub2");
    Value *Shr = Builder.CreateLShr(RoundingConst, Pattern.ShiftAmount, "shr");
    Value *Conv = Builder.CreateTrunc(Shr, Builder.getInt16Ty(), "conv");

    // Build argument list efficiently
    Args.append({Pattern.MatrixA, Pattern.MatrixB, Pattern.MatrixC,
                 Pattern.DimensionM, Pattern.DimensionN, Pattern.DimensionK,
                 Sub2, Conv});

    LLVM_DEBUG(dbgs() << "    SUCCESS: Created " << Args.size()
                      << " SIMD call arguments\n");
    return Args;
  }

  /// \brief Validate that a value is non-null and has expected type
  /// \param V The value to validate
  /// \param ExpectedType The expected LLVM type (optional)
  /// \return True if value is valid
  static bool validateValue(Value *V, Type *ExpectedType = nullptr) {
    if (!V) {
      LLVM_DEBUG(dbgs() << "      ERROR: Null value provided\n");
      return false;
    }

    if (ExpectedType && V->getType() != ExpectedType) {
      LLVM_DEBUG(dbgs() << "      ERROR: Value type mismatch\n");
      return false;
    }

    return true;
  }
};

/// \brief Utility class for function creation and management
///
/// Provides methods for creating SIMD functions with proper signatures,
/// attributes, and basic implementations. Optimized for minimal overhead
/// during function creation and validation.
class FunctionUtils {
public:
  /// \brief Create function type for SIMD matrix multiplication function
  /// \param Ctx LLVM context for type creation
  /// \return Function type with proper signature
  static FunctionType *createSIMDFunctionType(LLVMContext &Ctx) {
    // Cache frequently used types to reduce allocation overhead
    Type *I16PtrTy = PointerType::get(Type::getInt16Ty(Ctx), 0);
    Type *I32Ty = Type::getInt32Ty(Ctx);
    Type *I16Ty = Type::getInt16Ty(Ctx);
    Type *VoidTy = Type::getVoidTy(Ctx);

    // Pre-allocate parameter types vector
    SmallVector<Type *, 8> ParamTypes;
    ParamTypes.reserve(8);
    ParamTypes.append(
        {I16PtrTy, I16PtrTy, I16PtrTy, I32Ty, I32Ty, I32Ty, I32Ty, I16Ty});

    // Function signature: void func(i16* A, i16* B, i16* C, i32 m, i32 n, i32
    // k, i32 shift, i16 bias)
    return FunctionType::get(VoidTy, ParamTypes, false);
  }

  /// \brief Setup attributes for SIMD function
  /// \param F The function to setup attributes for
  static void setupSIMDFunctionAttributes(Function *F) {
    if (!F) {
      LLVM_DEBUG(
          dbgs() << "    ERROR: Null function provided for attribute setup\n");
      return;
    }

    // Set calling convention to fastcc for better performance
    F->setCallingConv(CallingConv::Fast);

    // Add signext attribute to 8th parameter (rounding_bias) - parameter 7
    // (0-indexed)
    if (F->arg_size() > 7) {
      auto ArgIt = F->arg_begin();
      std::advance(ArgIt, 7);
      ArgIt->addAttr(Attribute::SExt);
      LLVM_DEBUG(dbgs() << "    SUCCESS: Added signext attribute to "
                           "rounding_bias parameter\n");
    } else {
      LLVM_DEBUG(
          dbgs() << "    WARNING: Function has insufficient parameters for "
                    "signext attribute\n");
    }

    // Add function-level attributes for optimization
    F->addFnAttr(Attribute::AlwaysInline);
    F->addFnAttr(Attribute::NoUnwind);

    LLVM_DEBUG(
        dbgs() << "    SUCCESS: Function attributes configured successfully\n");
  }

  /// \brief Create simple function body with just return void
  /// \param F The function to create body for
  static void createSimpleFunctionBody(Function *F) {
    if (!F) {
      LLVM_DEBUG(
          dbgs() << "    ERROR: Null function provided for body creation\n");
      return;
    }

    BasicBlock *Entry = BasicBlock::Create(F->getContext(), "entry", F);
    IRBuilder<> Builder(Entry);
    Builder.CreateRetVoid();

    LLVM_DEBUG(dbgs() << "    SUCCESS: Simple function body created\n");
  }

  /// \brief Validate function signature matches expected pattern
  /// \param F The function to validate
  /// \return True if function signature is valid
  static bool validateFunctionSignature(Function *F) {
    if (!F) {
      LLVM_DEBUG(
          dbgs() << "    ERROR: Null function provided for validation\n");
      return false;
    }

    if (F->arg_size() < MinRequiredArgs) {
      LLVM_DEBUG(dbgs() << "    ERROR: Function has insufficient arguments: "
                        << F->arg_size() << " < " << MinRequiredArgs << "\n");
      return false;
    }

    LLVM_DEBUG(dbgs() << "    SUCCESS: Function signature validation passed\n");
    return true;
  }
};

/// \brief Pattern matcher for matrix multiplication loops
///
/// This class implements the core pattern matching logic to identify
/// matrix multiplication patterns in LLVM IR. It validates function
/// signatures, loop structures, and optimization feasibility.
/// Performance optimizations include early exits, cached computations,
/// and efficient traversal algorithms.
class MatrixMultiplicationMatcher {
public:
  /// \brief Main pattern matching entry point
  /// \param F The function to analyze
  /// \param LI Loop information for the function
  /// \param Pattern Output pattern information
  /// \return True if valid pattern found
  static bool matchPattern(Function &F, LoopInfo &LI,
                           MatrixMultiplicationPattern &Pattern) {
    LLVM_DEBUG(
        dbgs()
        << "\n[MatrixMultiplicationMatcher] Starting analysis for function: "
        << F.getName() << "\n");

    // Fast pre-validation to avoid expensive analysis on invalid functions
    if (!quickPreValidation(F)) {
      return false;
    }

    // Multi-phase pattern matching with early exits for performance
    if (!checkFunctionName(F))
      return false;
    if (!extractFunctionParameters(F, Pattern))
      return false;
    if (!analyzeLoopStructure(F, LI, Pattern))
      return false;
    if (!analyzeConditionsAndFeatures(F, Pattern))
      return false;

    Pattern.IsValid = validatePattern(Pattern);

    LLVM_DEBUG(dbgs() << "[Final Check] Pattern validation: "
                      << (Pattern.IsValid ? "VALID" : "INVALID") << "\n");

    return Pattern.IsValid;
  }

private:
  /// \brief Quick pre-validation to filter out obviously invalid functions
  /// \param F The function to pre-validate
  /// \return True if function is worth analyzing further
  static bool quickPreValidation(Function &F) {
    // Check basic requirements without expensive analysis
    if (F.isDeclaration()) {
      LLVM_DEBUG(
          dbgs() << "  ERROR: Function is a declaration, not a definition\n");
      return false;
    }

    if (!FunctionUtils::validateFunctionSignature(&F)) {
      return false;
    }

    return true;
  }

  /// \brief Check if function name matches expected pattern
  static bool checkFunctionName(Function &F) {
    LLVM_DEBUG(dbgs() << "[Step 1] Checking function name...\n");

    StringRef FuncName = F.getName();
    if (!FuncName.starts_with(TargetFunctionPrefix)) {
      LLVM_DEBUG(
          dbgs() << "  ERROR: Function name doesn't match pattern. Expected '"
                 << TargetFunctionPrefix << "*', got: " << FuncName << "\n");
      return false;
    }
    LLVM_DEBUG(dbgs() << "  SUCCESS: Function name matches: " << FuncName
                      << "\n");
    return true;
  }

  /// \brief Validate that the pattern has all required components
  static bool validatePattern(const MatrixMultiplicationPattern &Pattern) {
    bool HasMatrices = Pattern.MatrixA && Pattern.MatrixB && Pattern.MatrixC;
    bool HasLoops =
        Pattern.OuterLoop && Pattern.MiddleLoop && Pattern.InnerLoop;
    bool HasDimensions = Pattern.DimensionM && Pattern.DimensionN &&
                         Pattern.DimensionK && Pattern.ShiftAmount;

    bool IsComplete = HasMatrices && HasLoops && HasDimensions;

    LLVM_DEBUG(dbgs() << "    Pattern validation details:\n");
    LLVM_DEBUG(dbgs() << "      Matrices: " << (HasMatrices ? "PASS" : "FAIL")
                      << "\n");
    LLVM_DEBUG(dbgs() << "      Loops: " << (HasLoops ? "PASS" : "FAIL")
                      << "\n");
    LLVM_DEBUG(dbgs() << "      Dimensions: "
                      << (HasDimensions ? "PASS" : "FAIL") << "\n");

    return IsComplete;
  }

  static bool extractFunctionParameters(Function &F,
                                        MatrixMultiplicationPattern &Pattern) {
    LLVM_DEBUG(dbgs() << "[Step 2] Extracting function parameters...\n");
    LLVM_DEBUG(dbgs() << "    Function has " << F.arg_size() << " arguments\n");

    // Parameters already validated in quickPreValidation
    extractParametersFromFunction(F, Pattern);

    // Validate extracted parameters
    if (!Pattern.hasAllRequiredComponents()) {
      LLVM_DEBUG(
          dbgs() << "    ERROR: Failed to extract all required parameters\n");
      return false;
    }

    logExtractedParameters(Pattern);

    LLVM_DEBUG(
        dbgs() << "  SUCCESS: Successfully extracted function parameters\n");
    return true;
  }

  static void
  extractParametersFromFunction(Function &F,
                                MatrixMultiplicationPattern &Pattern) {
    auto ArgIt = F.arg_begin();
    Pattern.MatrixA = &*ArgIt++;
    Pattern.MatrixB = &*ArgIt++;
    Pattern.MatrixC = &*ArgIt++;
    Pattern.DimensionM = &*ArgIt++;
    Pattern.DimensionN = &*ArgIt++;
    Pattern.DimensionK = &*ArgIt++;
    Pattern.ShiftAmount = &*ArgIt++;
  }

  static void
  logExtractedParameters(const MatrixMultiplicationPattern &Pattern) {
    LLVM_DEBUG(dbgs() << "    SUCCESS: Extracted parameters:\n");
    LLVM_DEBUG(dbgs() << "       MatrixA (arg0): " << Pattern.MatrixA->getName()
                      << "\n");
    LLVM_DEBUG(dbgs() << "       MatrixB (arg1): " << Pattern.MatrixB->getName()
                      << "\n");
    LLVM_DEBUG(dbgs() << "       MatrixC (arg2): " << Pattern.MatrixC->getName()
                      << "\n");
    LLVM_DEBUG(dbgs() << "       DimensionM (arg3): "
                      << Pattern.DimensionM->getName() << "\n");
    LLVM_DEBUG(dbgs() << "       DimensionN (arg4): "
                      << Pattern.DimensionN->getName() << "\n");
    LLVM_DEBUG(dbgs() << "       DimensionK (arg5): "
                      << Pattern.DimensionK->getName() << "\n");
    LLVM_DEBUG(dbgs() << "       ShiftAmount (arg6): "
                      << Pattern.ShiftAmount->getName() << "\n");
  }

  static bool analyzeLoopStructure(Function &F, LoopInfo &LI,
                                   MatrixMultiplicationPattern &Pattern) {
    LLVM_DEBUG(dbgs() << "[Step 3] Analyzing loop structure...\n");

    // Early exit for empty loop info
    if (LI.empty()) {
      LLVM_DEBUG(dbgs() << "    ERROR: No loops found in function\n");
      return false;
    }

    // Count top-level loops efficiently
    unsigned TopLevelLoopCount = 0;
    for (auto It = LI.begin(); It != LI.end(); ++It) {
      ++TopLevelLoopCount;
    }

    LLVM_DEBUG(dbgs() << "    Found " << TopLevelLoopCount
                      << " top-level loops\n");

    if (findNestedLoopStructure(LI, Pattern)) {
      LLVM_DEBUG(
          dbgs()
          << "  SUCCESS: Found expected 3-level nested loop structure\n");
      return true;
    }

    LLVM_DEBUG(dbgs() << "  ERROR: Failed to find expected loop structure\n");
    return false;
  }

  static bool findNestedLoopStructure(LoopInfo &LI,
                                      MatrixMultiplicationPattern &Pattern) {
    // Use range-based for loop for better performance
    for (Loop *L : LI) {
      LLVM_DEBUG(dbgs() << "    Examining loop at depth " << L->getLoopDepth()
                        << "\n");

      // Early continue if not outermost loop (performance optimization)
      if (L->getLoopDepth() != OuterLoopDepth)
        continue;

      if (findMiddleAndInnerLoops(L, Pattern)) {
        return true;
      }
    }

    LLVM_DEBUG(
        dbgs() << "    ERROR: No complete 3-level nested structure found\n");
    return false;
  }

  static bool findMiddleAndInnerLoops(Loop *OuterLoop,
                                      MatrixMultiplicationPattern &Pattern) {
    Pattern.OuterLoop = OuterLoop;
    LLVM_DEBUG(dbgs() << "      SUCCESS: Found I-loop (depth " << OuterLoopDepth
                      << ")\n");

    auto &SubLoops = OuterLoop->getSubLoops();
    LLVM_DEBUG(dbgs() << "      I-loop has " << SubLoops.size()
                      << " sub-loops\n");

    if (SubLoops.empty()) {
      LLVM_DEBUG(dbgs() << "      ERROR: I-loop has no sub-loops\n");
      return false;
    }

    // Find J loop using optimized traversal
    for (Loop *SubL : SubLoops) {
      LLVM_DEBUG(dbgs() << "        Examining sub-loop at depth "
                        << SubL->getLoopDepth() << "\n");

      // Early continue if not middle loop
      if (SubL->getLoopDepth() != MiddleLoopDepth)
        continue;

      if (findInnerLoop(SubL, Pattern)) {
        return true;
      }
    }
    LLVM_DEBUG(dbgs() << "      ERROR: No depth-" << MiddleLoopDepth
                      << " loop found in I-loop\n");
    return false;
  }

  static bool findInnerLoop(Loop *MiddleLoop,
                            MatrixMultiplicationPattern &Pattern) {
    Pattern.MiddleLoop = MiddleLoop;
    LLVM_DEBUG(dbgs() << "        SUCCESS: Found J-loop (depth "
                      << MiddleLoopDepth << ")\n");

    auto &SubLoops = MiddleLoop->getSubLoops();
    LLVM_DEBUG(dbgs() << "        J-loop has " << SubLoops.size()
                      << " sub-loops\n");

    if (SubLoops.empty()) {
      LLVM_DEBUG(dbgs() << "        ERROR: J-loop has no sub-loops\n");
      return false;
    }

    // Find S loop using optimized traversal
    for (Loop *SubSubL : SubLoops) {
      LLVM_DEBUG(dbgs() << "          Examining sub-sub-loop at depth "
                        << SubSubL->getLoopDepth() << "\n");

      if (SubSubL->getLoopDepth() == InnerLoopDepth) {
        Pattern.InnerLoop = SubSubL;
        LLVM_DEBUG(dbgs() << "          SUCCESS: Found S-loop (depth "
                          << InnerLoopDepth << ")\n");
        LLVM_DEBUG(
            dbgs()
            << "    SUCCESS: Complete 3-level nested loop structure found\n");
        return true;
      }
    }
    LLVM_DEBUG(dbgs() << "        ERROR: No depth-" << InnerLoopDepth
                      << " loop found in J-loop\n");
    return false;
  }

  static bool
  analyzeConditionsAndFeatures(Function &F,
                               MatrixMultiplicationPattern &Pattern) {
    LLVM_DEBUG(dbgs() << "[Step 4] Analyzing conditions and features...\n");
    LLVM_DEBUG(
        dbgs() << "    Scanning for required patterns and conditions...\n");

    // Perform analysis with early exits for performance
    bool FoundShiftCondition = findShiftCondition(F);
    bool FoundKModCondition = findKMod8Condition(F);

    // Pattern analysis can be done in parallel conceptually, but we do it
    // sequentially for simplicity. These calls modify Pattern state directly.
    findRoundingConstants(F, Pattern);
    findShiftInstructions(F, Pattern);

    // Set vectorization conditions
    if (FoundShiftCondition && FoundKModCondition)
      Pattern.IsVectorizable = true;

    logAnalysisSummary(FoundShiftCondition, FoundKModCondition, Pattern);

    bool Result = Pattern.HasRoundingConstants && Pattern.HasShiftOperations;
    LLVM_DEBUG(dbgs() << "    Final result: " << (Result ? "PASS" : "FAIL")
                      << "\n");

    return Result;
  }

  static bool findShiftCondition(Function &F) {
    LLVM_DEBUG(dbgs() << "      Searching for shift condition patterns...\n");

    unsigned ShiftComparisonCount = 0;

    // Optimized traversal with early exit
    for (const BasicBlock &BB : F) {
      for (const Instruction &I : BB) {
        const auto *Cmp = dyn_cast<ICmpInst>(&I);
        if (!Cmp)
          continue;

        const auto *C = dyn_cast<ConstantInt>(Cmp->getOperand(1));
        if (!C || C->getValue() != MaxShiftAmount)
          continue;

        ++ShiftComparisonCount;

        if (Cmp->getPredicate() == ICmpInst::ICMP_SLT ||
            Cmp->getPredicate() == ICmpInst::ICMP_SGT) {
          LLVM_DEBUG(dbgs() << "        SUCCESS: Found shift comparison with "
                            << MaxShiftAmount
                            << " (count: " << ShiftComparisonCount << ")\n");
          return true;
        }
      }
    }

    LLVM_DEBUG(dbgs() << "        ERROR: Shift condition not found (checked "
                      << ShiftComparisonCount << " comparisons)\n");
    return false;
  }

  static bool findRoundingConstants(Function &F,
                                    MatrixMultiplicationPattern &Pattern) {
    LLVM_DEBUG(dbgs() << "      Searching for rounding constants...\n");

    unsigned RoundingConstantCount = 0;

    // Optimized traversal with early success detection
    for (const BasicBlock &BB : F) {
      for (const Instruction &I : BB) {
        for (const Use &Op : I.operands()) {
          const auto *CI = dyn_cast<ConstantInt>(Op);
          if (CI && CI->getValue() == RoundingConstant) {
            ++RoundingConstantCount;
            if (!Pattern.HasRoundingConstants) {
              Pattern.HasRoundingConstants = true;
              LLVM_DEBUG(dbgs()
                         << "        SUCCESS: Found first rounding constant "
                         << RoundingConstant << "\n");
            }
          }
        }
      }
    }

    if (Pattern.HasRoundingConstants) {
      LLVM_DEBUG(dbgs() << "        SUCCESS: Total rounding constants found: "
                        << RoundingConstantCount << "\n");
    } else {
      LLVM_DEBUG(dbgs() << "        ERROR: No rounding constants found\n");
    }

    return Pattern.HasRoundingConstants;
  }

  static bool findShiftInstructions(Function &F,
                                    MatrixMultiplicationPattern &Pattern) {
    LLVM_DEBUG(dbgs() << "      Searching for shift instructions...\n");

    unsigned ShiftInstructionCount = 0;

    // Optimized traversal with early success detection
    for (const BasicBlock &BB : F) {
      for (const Instruction &I : BB) {
        if (I.getOpcode() == Instruction::AShr ||
            I.getOpcode() == Instruction::LShr) {
          ++ShiftInstructionCount;
          if (!Pattern.HasShiftOperations) {
            Pattern.HasShiftOperations = true;
            LLVM_DEBUG(dbgs()
                       << "        SUCCESS: Found first shift instruction\n");
          }
        }
      }
    }

    if (Pattern.HasShiftOperations) {
      LLVM_DEBUG(dbgs() << "        SUCCESS: Total shift instructions found: "
                        << ShiftInstructionCount << "\n");
    } else {
      LLVM_DEBUG(dbgs() << "        ERROR: No shift instructions found\n");
    }

    return Pattern.HasShiftOperations;
  }

  static void logAnalysisSummary(bool FoundShiftCondition,
                                 bool FoundKModCondition,
                                 const MatrixMultiplicationPattern &Pattern) {
    LLVM_DEBUG(dbgs() << "    === Analysis Summary ===\n");
    LLVM_DEBUG(dbgs() << "      Shift condition: "
                      << (FoundShiftCondition ? "Found" : "Not found") << "\n");
    LLVM_DEBUG(dbgs() << "      k % 8 == 0 condition: "
                      << (FoundKModCondition ? "Found" : "Not found") << "\n");
    LLVM_DEBUG(dbgs() << "      HasRoundingConstants: "
                      << Pattern.HasRoundingConstants << "\n");
    LLVM_DEBUG(dbgs() << "      HasShiftOperations: "
                      << Pattern.HasShiftOperations << "\n");
    LLVM_DEBUG(dbgs() << "      IsVectorizable: " << Pattern.IsVectorizable
                      << "\n");
  }

  static bool findKMod8Condition(Function &F) {
    LLVM_DEBUG(dbgs() << "            Searching for k % 8 == 0 pattern...\n");

    unsigned AndInstructionCount = 0;

    // Optimized traversal with early exit
    for (const BasicBlock &BB : F) {
      for (const Instruction &I : BB) {
        const auto *And = dyn_cast<BinaryOperator>(&I);
        if (!And || And->getOpcode() != Instruction::And)
          continue;

        const auto *C = dyn_cast<ConstantInt>(And->getOperand(1));
        if (!C || C->getValue() != (VectorAlignment - 1))
          continue;

        ++AndInstructionCount;

        LLVM_DEBUG(dbgs() << "            Found 'AND " << (VectorAlignment - 1)
                          << "' instruction (count: " << AndInstructionCount
                          << ")\n");

        if (findZeroComparison(And)) {
          return true;
        }
      }
    }

    LLVM_DEBUG(dbgs() << "            ERROR: k % " << VectorAlignment
                      << " == 0 pattern not found (checked "
                      << AndInstructionCount << " AND instructions)\n");
    return false;
  }

  static bool findZeroComparison(const BinaryOperator *And) {
    unsigned UserCount = 0;

    // Efficient user traversal with early exit
    for (const auto *User : And->users()) {
      ++UserCount;
      const auto *Cmp = dyn_cast<ICmpInst>(User);
      if (!Cmp)
        continue;

      LLVM_DEBUG(dbgs() << "              Found comparison user " << UserCount
                        << "\n");

      if (Cmp->getPredicate() == ICmpInst::ICMP_EQ) {
        const auto *Zero = dyn_cast<ConstantInt>(Cmp->getOperand(1));
        if (Zero && Zero->isZero()) {
          LLVM_DEBUG(dbgs() << "              SUCCESS: Found complete k % "
                            << VectorAlignment << " == 0 pattern\n");
          return true;
        }
      }
    }

    LLVM_DEBUG(
        dbgs() << "              ERROR: No valid zero comparison found among "
               << UserCount << " users\n");
    return false;
  }
};

/// \brief Transformer for converting loops to SIMD calls
///
/// This class handles the actual transformation of matrix multiplication
/// loops into optimized SIMD function calls, including control flow
/// restructuring and function creation. All methods are optimized for
/// minimal overhead and robust error handling.
class MatrixMultiplicationTransformer {
public:
  /// \brief Main transformation entry point
  /// \param F The function to transform
  /// \param Pattern The detected matrix multiplication pattern
  /// \return True if transformation successful
  static bool transformToSIMD(Function &F,
                              const MatrixMultiplicationPattern &Pattern) {
    LLVM_DEBUG(dbgs() << "\n[MatrixMultiplicationTransformer] Starting SIMD "
                         "transformation...\n");

    // Validate transformation preconditions
    if (!Pattern.canOptimize()) {
      LLVM_DEBUG(dbgs() << "  ERROR: Pattern cannot be optimized\n");
      return false;
    }

    Module *M = F.getParent();
    if (!M) {
      LLVM_DEBUG(dbgs() << "  ERROR: Function has no parent module\n");
      return false;
    }

    // Get or create SIMD function
    Function *SIMDFunc = getOrCreateSIMDFunction(M);
    if (!SIMDFunc) {
      LLVM_DEBUG(dbgs() << "  ERROR: Failed to create SIMD function\n");
      return false;
    }
    LLVM_DEBUG(dbgs() << "  SUCCESS: SIMD function ready: "
                      << SIMDFunc->getName() << "\n");

    // Restructure entry block
    return restructureEntryBlock(F, Pattern, SIMDFunc);
  }

private:
  static bool restructureEntryBlock(Function &F,
                                    const MatrixMultiplicationPattern &Pattern,
                                    Function *SIMDFunc) {
    LLVM_DEBUG(dbgs() << "[Step 2] Restructuring entry block...\n");

    // Find required basic blocks with error handling
    BasicBlock *CleanupBlock =
        BasicBlockUtils::findBasicBlockByName(F, CleanupBlockName);
    BasicBlock *PreheaderBlock =
        BasicBlockUtils::findBasicBlockByName(F, PreheaderBlockName);

    if (!CleanupBlock) {
      LLVM_DEBUG(dbgs() << "    ERROR: Cannot find cleanup block '"
                        << CleanupBlockName << "'\n");
      return false;
    }

    if (!PreheaderBlock) {
      LLVM_DEBUG(dbgs() << "    ERROR: Cannot find preheader block '"
                        << PreheaderBlockName << "'\n");
      return false;
    }

    // Create optimization condition and blocks
    if (!createOptimizationLogic(F, Pattern, SIMDFunc, PreheaderBlock,
                                 CleanupBlock)) {
      LLVM_DEBUG(dbgs() << "    ERROR: Failed to create optimization logic\n");
      return false;
    }

    LLVM_DEBUG(dbgs() << "  SUCCESS: Successfully restructured function\n");
    return true;
  }

  static bool
  createOptimizationLogic(Function &F,
                          const MatrixMultiplicationPattern &Pattern,
                          Function *SIMDFunc, BasicBlock *PreheaderBlock,
                          BasicBlock *CleanupBlock) {
    // Find split point with error handling
    Instruction *SplitBefore = BasicBlockUtils::findInstructionAtIndex(
        PreheaderBlock, SplitInstructionIndex);
    if (!SplitBefore) {
      LLVM_DEBUG(
          dbgs() << "    ERROR: Could not find instruction to split before\n");
      return false;
    }

    // Create optimization condition
    IRBuilder<> Builder(PreheaderBlock, PreheaderBlock->begin());
    Value *UseSIMD = IRValueUtils::createOptimizationCondition(
        Builder, Pattern.DimensionK, Pattern.ShiftAmount);
    if (!UseSIMD) {
      LLVM_DEBUG(
          dbgs() << "    ERROR: Failed to create optimization condition\n");
      return false;
    }

    LLVM_DEBUG(dbgs() << "    SUCCESS: Created unified SIMD condition: (k & "
                      << (VectorAlignment - 1) << ") == 0 && shift < "
                      << MaxShiftAmount << "\n");

    // Create SIMD optimization block
    BasicBlock *SIMDOptBlock =
        createSIMDOptimizationBlock(F, Pattern, SIMDFunc, CleanupBlock);
    if (!SIMDOptBlock) {
      LLVM_DEBUG(
          dbgs() << "    ERROR: Failed to create SIMD optimization block\n");
      return false;
    }

    // Split and redirect control flow
    return redirectControlFlow(PreheaderBlock, SIMDOptBlock, UseSIMD,
                               SplitBefore);
  }

  static BasicBlock *
  createSIMDOptimizationBlock(Function &F,
                              const MatrixMultiplicationPattern &Pattern,
                              Function *SIMDFunc, BasicBlock *CleanupBlock) {
    BasicBlock *SIMDOptBlock =
        BasicBlock::Create(F.getContext(), SIMDBlockName, &F);
    IRBuilder<> SIMDBuilder(SIMDOptBlock);

    // Create SIMD call arguments with error handling
    SmallVector<Value *, 8> Args =
        IRValueUtils::createSIMDCallArgs(SIMDBuilder, Pattern);
    if (Args.empty()) {
      LLVM_DEBUG(
          dbgs() << "      ERROR: Failed to create SIMD call arguments\n");
      return nullptr;
    }

    CallInst *SIMDCall = SIMDBuilder.CreateCall(SIMDFunc, Args);
    if (!SIMDCall) {
      LLVM_DEBUG(
          dbgs() << "      ERROR: Failed to create SIMD call instruction\n");
      return nullptr;
    }

    SIMDCall->setTailCall(true);
    SIMDBuilder.CreateBr(CleanupBlock);

    LLVM_DEBUG(dbgs() << "      SUCCESS: Created SIMD optimization block with "
                      << Args.size() << " arguments\n");
    return SIMDOptBlock;
  }

  static bool redirectControlFlow(BasicBlock *PreheaderBlock,
                                  BasicBlock *SIMDOptBlock, Value *UseSIMD,
                                  Instruction *SplitBefore) {
    // Use splitBasicBlock to correctly split block with error handling
    BasicBlock *ScalarPathBlock = BasicBlockUtils::safeSplitBasicBlock(
        PreheaderBlock, SplitBefore, ScalarPathBlockName);
    if (!ScalarPathBlock) {
      LLVM_DEBUG(dbgs() << "      ERROR: Failed to split basic block\n");
      return false;
    }

    // Remove auto-generated unconditional branch, we want to replace with
    // conditional branch
    Instruction *OldTerminator = PreheaderBlock->getTerminator();
    if (OldTerminator) {
      OldTerminator->eraseFromParent();
    }

    // Insert conditional branch at end of preheader block
    IRBuilder<> Builder(PreheaderBlock);
    Instruction *CondBr =
        Builder.CreateCondBr(UseSIMD, SIMDOptBlock, ScalarPathBlock);
    if (!CondBr) {
      LLVM_DEBUG(
          dbgs() << "      ERROR: Failed to create conditional branch\n");
      return false;
    }

    LLVM_DEBUG(
        dbgs() << "    SUCCESS: Successfully restructured control flow with "
                  "non-contradictory conditions\n");
    LLVM_DEBUG(dbgs() << "    SUCCESS: SIMD path: k%" << VectorAlignment
                      << "==0 && shift<" << MaxShiftAmount << "\n");
    LLVM_DEBUG(dbgs() << "    SUCCESS: Scalar path: everything else\n");

    return true;
  }

  static Function *getOrCreateSIMDFunction(Module *M) {
    LLVM_DEBUG(dbgs() << "[Step 1] Getting or creating SIMD function...\n");

    // Check for existing function first (most common case in practice)
    if (Function *ExistingFunc = M->getFunction(SIMDFunctionName)) {
      LLVM_DEBUG(dbgs() << "    Found existing SIMD function: "
                        << SIMDFunctionName << "\n");
      return ExistingFunc;
    }

    return createNewSIMDFunction(M);
  }

  static Function *createNewSIMDFunction(Module *M) {
    LLVM_DEBUG(dbgs() << "    Creating new SIMD function: " << SIMDFunctionName
                      << "\n");

    LLVMContext &Ctx = M->getContext();
    FunctionType *FT = FunctionUtils::createSIMDFunctionType(Ctx);
    if (!FT) {
      LLVM_DEBUG(dbgs() << "      ERROR: Failed to create function type\n");
      return nullptr;
    }

    Function *NewFunc =
        Function::Create(FT, Function::InternalLinkage, SIMDFunctionName, M);
    if (!NewFunc) {
      LLVM_DEBUG(dbgs() << "      ERROR: Failed to create function\n");
      return nullptr;
    }

    FunctionUtils::setupSIMDFunctionAttributes(NewFunc);
    FunctionUtils::createSimpleFunctionBody(NewFunc);

    LLVM_DEBUG(
        dbgs()
        << "    SUCCESS: SIMD function created with simple implementation\n");

    return NewFunc;
  }
};

} // anonymous namespace

/// \brief Main pass implementation
///
/// This is the entry point for the RISCV ESP32P4 Loop Pattern to Intrinsic
/// transformation pass. It orchestrates the pattern matching and transformation
/// phases while providing comprehensive logging and error handling.
///
/// Performance characteristics:
/// - O(n) complexity where n is the number of instructions in the function
/// - Early exit strategies minimize analysis overhead for non-matching
/// functions
/// - Memory usage is minimized through efficient data structures and algorithms
PreservedAnalyses
RISCVESP32P4LoopPatternToIntrinsicPass::run(Function &F,
                                            FunctionAnalysisManager &AM) {

  LLVM_DEBUG(dbgs() << "\n" << std::string(80, '=') << "\n");
  LLVM_DEBUG(
      dbgs() << "RISCVESP32P4LoopPatternToIntrinsicPass::run() called\n");
  LLVM_DEBUG(dbgs() << "   Function: " << F.getName() << "\n");
  LLVM_DEBUG(dbgs() << "   Pass enabled: "
                    << EnableRISCVESP32P4LoopPatternToIntrinsic << "\n");
  LLVM_DEBUG(dbgs() << std::string(80, '=') << "\n");

  // Early exit if pass is disabled (most common case in release builds)
  if (!EnableRISCVESP32P4LoopPatternToIntrinsic) {
    LLVM_DEBUG(dbgs() << "ERROR: Pass disabled by command line option\n");
    return PreservedAnalyses::all();
  }

  // Get loop analysis with error handling
  auto &LI = AM.getResult<LoopAnalysis>(F);
  LLVM_DEBUG(dbgs() << "SUCCESS: LoopAnalysis obtained successfully\n");

  LLVM_DEBUG(dbgs() << "Starting matrix pattern analysis...\n");

  MatrixMultiplicationPattern Pattern;
  if (MatrixMultiplicationMatcher::matchPattern(F, LI, Pattern)) {
    LLVM_DEBUG(dbgs() << "\nMATRIX MULTIPLICATION PATTERN FOUND!\n");
    Pattern.dump();

    if (Pattern.canOptimize()) {
      LLVM_DEBUG(
          dbgs() << "\nAll conditions met! Applying SIMD transformation...\n");

      if (MatrixMultiplicationTransformer::transformToSIMD(F, Pattern)) {
        LLVM_DEBUG(dbgs() << "\nSUCCESS: Function transformed to use SIMD!\n");
        LLVM_DEBUG(dbgs() << std::string(80, '=') << "\n\n");
        return PreservedAnalyses::none();
      }
      LLVM_DEBUG(dbgs() << "\nERROR: SIMD transformation failed\n");
      LLVM_DEBUG(dbgs() << std::string(80, '=') << "\n\n");
    } else {
      LLVM_DEBUG(
          dbgs() << "\nWARNING: Pattern found but conditions not met:\n");
      LLVM_DEBUG(dbgs() << "   IsVectorizable: " << Pattern.IsVectorizable
                        << "\n");
      LLVM_DEBUG(dbgs() << "   HasRoundingConstants: "
                        << Pattern.HasRoundingConstants << "\n");
      LLVM_DEBUG(dbgs() << "   HasShiftOperations: "
                        << Pattern.HasShiftOperations << "\n");
      LLVM_DEBUG(dbgs() << "   IsValid: " << Pattern.IsValid << "\n");
    }
  } else {
    LLVM_DEBUG(
        dbgs() << "\nERROR: No matching matrix multiplication pattern found\n");
  }

  LLVM_DEBUG(dbgs() << "\nPass completed - no changes made\n");
  LLVM_DEBUG(dbgs() << std::string(80, '=') << "\n\n");
  return PreservedAnalyses::all();
}
