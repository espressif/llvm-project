#ifndef LLVM_LIB_TARGET_RISCV_RISCVESP32P4OPTUTILS_H
#define LLVM_LIB_TARGET_RISCV_RISCVESP32P4OPTUTILS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/LoopUnrollAnalyzer.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/ADCE.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/DeadStoreElimination.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/LoopStrengthReduce.h"
#include "llvm/Transforms/Scalar/LoopUnrollPass.h"
#include "llvm/Transforms/Scalar/Reassociate.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopPeel.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/SimplifyCFGOptions.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

namespace llvm {

// Check if the successors of the basic block meet the expected conditions
inline bool checkSuccessors(BasicBlock *BB, unsigned NumSuccessors,
                            BasicBlock *Succ1, BasicBlock *Succ2 = nullptr) {
  assert(BB->getTerminator() && "BB's terminator is nullptr");
  assert((NumSuccessors == 1 || NumSuccessors == 2) &&
         "NumSuccessors must be 1 or 2");
  if (succ_size(BB) != NumSuccessors)
    return false;
  if (BB->getTerminator()->getSuccessor(0) != Succ1)
    return false;
  if (NumSuccessors == 2 && BB->getTerminator()->getSuccessor(1) != Succ2)
    return false;
  return true;
}

// Get the first instruction of a specified type
template <typename T> inline T *getFirstInst(BasicBlock *BB) {
  for (Instruction &I : *BB) {
    if (T *Inst = dyn_cast<T>(&I)) {
      return Inst;
    }
  }
  return nullptr;
}

template <typename T> inline T *getLastInst(BasicBlock *BB) {
  for (Instruction &I : reverse(*BB)) {
    if (T *Inst = dyn_cast<T>(&I)) {
      return Inst;
    }
  }
  return nullptr;
}

inline PHINode *getFirstI32Phi(BasicBlock *BB) {
  for (auto &Inst : *BB) {
    if (auto *Phi = dyn_cast<PHINode>(&Inst)) {
      if (Phi->getType()->isIntegerTy(32)) {
        return Phi;
      }
    }
  }
  return nullptr;
}

// Check if fmuladd.f32 intrinsic is used
inline bool checkFMulAddUsage(Function &F) {
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (RecurrenceDescriptor::isFMulAddIntrinsic(&I)) {
        return true;
      }
    }
  }
  return false;
}

// Helper function to get a basic block by name
inline BasicBlock *getBasicBlockByName(Function &F, StringRef Name) {
  for (BasicBlock &BB : F)
    if (BB.getName() == Name)
      return &BB;
  return nullptr;
}

inline void runDeadCodeElimination(Function &F) {
  legacy::FunctionPassManager FPM(F.getParent());
  FPM.add(createDeadCodeEliminationPass());
  FPM.run(F);
}

inline void addnoalias(Function &F) {
  for (Argument &Arg : F.args()) {
    if (Arg.getType()->isPointerTy()) {
      Arg.addAttr(Attribute::NoAlias);
    }
  }
}

// Helper function to swap the successors of a terminator instruction
inline void swapTerminatorSuccessors(BasicBlock *BB) {
  if (auto *BI = dyn_cast<BranchInst>(BB->getTerminator())) {
    if (BI->isConditional() && BI->getNumSuccessors() == 2) {
      BasicBlock *TrueSuccessor = BI->getSuccessor(0);
      BasicBlock *FalseSuccessor = BI->getSuccessor(1);
      BI->setSuccessor(0, FalseSuccessor);
      BI->setSuccessor(1, TrueSuccessor);
    } else {
      llvm_unreachable("BB's terminator is not a conditional branch or doesn't "
                       "have two successors");
    }
  } else {
    llvm_unreachable("BB's terminator is not a branch instruction");
  }
}

// Helper function to clone a basic block and update its relations
inline BasicBlock *cloneBasicBlockWithRelations(BasicBlock *BB,
                                                const std::string &NameSuffix,
                                                Function *F) {
  ValueToValueMapTy VMap;
  BasicBlock *NewBB = CloneBasicBlock(BB, VMap, NameSuffix, F);

  // Update instruction references in the new block
  for (Instruction &I : *NewBB) {
    // Update operands
    for (Use &U : I.operands()) {
      Value *V = U.get();
      Value *NewV = VMap[V];
      if (NewV) {
        U.set(NewV);
      }
    }

    // Update PHI node basic block references
    if (PHINode *PN = dyn_cast<PHINode>(&I)) {
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
        BasicBlock *IncomingBB = PN->getIncomingBlock(i);
        if (IncomingBB == BB) {
          PN->setIncomingBlock(i, NewBB);
        } else if (VMap.count(IncomingBB)) {
          PN->setIncomingBlock(i, cast<BasicBlock>(VMap[IncomingBB]));
        }
      }
    }
  }

  return NewBB;
}

// Helper function to move PHI nodes to the top of a basic block
inline void movePHINodesToTop(BasicBlock &BB) {
  SmallVector<PHINode *, 8> PHIs;
  for (Instruction &I : BB) {
    if (PHINode *PHI = dyn_cast<PHINode>(&I)) {
      PHIs.push_back(PHI);
    }
  }

  // Move PHI nodes in reverse order
  for (auto it = PHIs.rbegin(); it != PHIs.rend(); ++it) {
    (*it)->moveBefore(&BB.front());
  }
}

// Helper function to get the first float PHI node in a basic block
inline PHINode *getFirstFloatPhi(BasicBlock *BB) {
  for (PHINode &Phi : BB->phis()) {
    if (Phi.getType()->isFloatTy()) {
      return &Phi;
    }
  }
  return nullptr;
}

// Helper function to get the last float PHI node in a basic block
inline PHINode *getLastFloatPhi(BasicBlock *BB) {
  for (auto it = BB->rbegin(); it != BB->rend(); ++it) {
    if (auto *Phi = dyn_cast<PHINode>(&*it)) {
      if (Phi->getType()->isFloatTy()) {
        return Phi;
      }
    }
  }
  return nullptr;
}

// Helper function to get the first i64 PHI node in a basic block
inline PHINode *getFirstI64Phi(BasicBlock *BB) {
  for (auto &Inst : *BB) {
    if (auto *Phi = dyn_cast<PHINode>(&Inst)) {
      if (Phi->getType()->isIntegerTy(64)) {
        return Phi;
      }
    }
  }
  return nullptr;
}

// Helper function to get the last i64 PHI node in a basic block
inline PHINode *getLastI64Phi(BasicBlock *BB) {
  for (auto it = BB->rbegin(); it != BB->rend(); ++it) {
    if (auto *Phi = dyn_cast<PHINode>(&*it)) {
      if (Phi->getType()->isIntegerTy(64)) {
        return Phi;
      }
    }
  }
  return nullptr;
}

// Helper function to get the last 32-bit integer PHI node in a basic block
inline PHINode *getLastI32Phi(BasicBlock *BB) {
  for (auto it = BB->rbegin(); it != BB->rend(); ++it) {
    if (auto *Phi = dyn_cast<PHINode>(&*it)) {
      if (Phi->getType()->isIntegerTy(32)) {
        return Phi;
      }
    }
  }
  return nullptr;
}

// Helper function to get the first ICmp instruction in a basic block
inline ICmpInst *getFirstICmpInst(BasicBlock *BB) {
  for (Instruction &I : *BB) {
    if (auto *CI = dyn_cast<ICmpInst>(&I)) {
      return CI;
    }
  }
  return nullptr;
}

// Helper function to get the last add nsw i64 instruction in a basic block
inline Instruction *getLastAddNswI64Inst(BasicBlock *BB) {
  for (auto it = BB->rbegin(); it != BB->rend(); ++it) {
    if (auto *BinOp = dyn_cast<BinaryOperator>(&*it)) {
      if (BinOp->getOpcode() == Instruction::Add && BinOp->hasNoSignedWrap() &&
          BinOp->getType()->isIntegerTy(64)) {
        return BinOp;
      }
    }
  }
  return nullptr;
}

// Helper function to get the first add nuw nsw i32 instruction in a basic block
inline Instruction *getFirstAddNuwNswI32Inst(BasicBlock *BB) {
  for (auto &I : *BB) {
    if (auto *BinOp = dyn_cast<BinaryOperator>(&I)) {
      if (BinOp->getOpcode() == Instruction::Add &&
          BinOp->hasNoUnsignedWrap() && BinOp->hasNoSignedWrap() &&
          BinOp->getType()->isIntegerTy(32)) {
        return BinOp;
      }
    }
  }
  return nullptr;
}

// Helper function to get the first ICmp instruction with a specific predicate
// in a basic block
inline ICmpInst *getFirstICmpInstWithPredicate(BasicBlock *BB,
                                               ICmpInst::Predicate Predicate) {
  for (Instruction &I : *BB) {
    if (auto *CI = dyn_cast<ICmpInst>(&I)) {
      if (CI->getPredicate() == Predicate) {
        return CI;
      }
    }
  }
  return nullptr;
}

// Helper function to get the last ICmp instruction with a specific predicate in
// a basic block
inline ICmpInst *getLastICmpInstWithPredicate(BasicBlock *BB,
                                              ICmpInst::Predicate Predicate) {
  ICmpInst *lastICmp = nullptr;
  for (Instruction &I : *BB) {
    if (auto *CI = dyn_cast<ICmpInst>(&I)) {
      if (CI->getPredicate() == Predicate) {
        lastICmp = CI;
      }
    }
  }
  return lastICmp;
}

// Helper function to get the first CallInst with a specific name in a basic
// block
inline CallInst *getFirstCallInstWithName(BasicBlock *BB, StringRef Name) {
  for (Instruction &I : *BB) {
    if (auto *Call = dyn_cast<CallInst>(&I)) {
      if (Call->getCalledFunction() &&
          Call->getCalledFunction()->getName() == Name) {
        return Call;
      }
    }
  }
  return nullptr;
}

// Helper function to update operands of new instructions
inline void updateOperands(SmallVector<Instruction *, 8> &NewInsts,
                           ValueToValueMapTy &ValueMap) {
  for (Instruction *inst : NewInsts) {
    for (unsigned i = 0; i < inst->getNumOperands(); i++) {
      Value *op = inst->getOperand(i);
      if (ValueMap.count(op)) {
        inst->setOperand(i, ValueMap[op]);
      }
    }
  }
}

// Helper function to get the last PHI node in a basic block
inline PHINode *getLastPhi(BasicBlock *BB) {
  for (auto it = BB->rbegin(); it != BB->rend(); ++it) {
    if (auto *Phi = dyn_cast<PHINode>(&*it)) {
      return Phi;
    }
  }
  return nullptr;
}

template <typename T> inline BinaryOperator *getFirstAddInst(BasicBlock *BB) {
  for (Instruction &I : *BB) {
    if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(&I)) {
      if (BinOp->getOpcode() == Instruction::Add &&
          BinOp->getType()->isIntegerTy(sizeof(T) * 8)) {
        return BinOp;
      }
    }
  }
  return nullptr;
}

template <typename T> inline BinaryOperator *getLastAddInst(BasicBlock *BB) {
  for (Instruction &I : reverse(*BB)) {
    if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(&I)) {
      if (BinOp->getOpcode() == Instruction::Add &&
          BinOp->getType()->isIntegerTy(sizeof(T) * 8)) {
        return BinOp;
      }
    }
  }
  return nullptr;
}

inline Instruction *getFirstFMulAddInst(BasicBlock *BB) {
  for (Instruction &I : *BB) {
    if (RecurrenceDescriptor::isFMulAddIntrinsic(&I)) {
      return &I;
    }
  }
  return nullptr;
}

// Helper function to run dead code elimination
inline void runSimplifyDcePasses(Function &F) {
  legacy::FunctionPassManager FPM(F.getParent());
  FPM.add(createDeadCodeEliminationPass());
  FPM.add(createLoopSimplifyPass());
  // Add SimplifyCFG pass with common options
  FPM.add(createCFGSimplificationPass(
      SimplifyCFGOptions()
          .bonusInstThreshold(1) // Set instruction bonus threshold
          .forwardSwitchCondToPhi(
              true) // Allow forwarding switch conditions to phi
          .convertSwitchToLookupTable(
              true)                  // Allow converting switch to lookup table
          .needCanonicalLoops(false) // Don't require canonical loop form
          .hoistCommonInsts(true)    // Hoist common instructions
          .sinkCommonInsts(true)     // Sink common instructions
      ));
  FPM.run(F);
}

inline void moveInstructionsBeforePoint(SmallVector<Instruction *> &insts,
                                        Instruction *&insertPoint) {
  for (auto *inst : insts) {
    inst->moveBefore(insertPoint);
    insertPoint = inst->getNextNode();
  }
}

inline std::pair<BasicBlock *, BasicBlock *>
cloneAndMergeLoop(Loop *L, Function &F, int unroll_count) {

  BasicBlock *loopHeader = L->getHeader();
  BasicBlock *loopHeaderClone =
      cloneBasicBlockWithRelations(loopHeader, ".clone", &F);
  loopHeaderClone->moveAfter(loopHeader);
  loopHeaderClone->getTerminator()->setSuccessor(1, loopHeader);

  std::vector<BasicBlock *> BBsToMerge;
  StringRef forBodyName = loopHeader->getName();
  for (int i = 1; i < unroll_count; ++i) {
    std::string BBName = (forBodyName + "." + std::to_string(i)).str();
    BasicBlock *clonedBB = getBasicBlockByName(F, BBName);
    if (clonedBB) {
      BBsToMerge.push_back(clonedBB);
    } else {
      llvm_unreachable("Basic block not found");
    }
  }

  if (BBsToMerge.size() == static_cast<size_t>(unroll_count - 1)) {
    for (BasicBlock *BB : BBsToMerge) {
      MergeBasicBlockIntoOnlyPred(BB);
    }
  }

  return std::make_pair(loopHeaderClone, BBsToMerge.back());
}

inline void setPHIIndexIncomingBlock(BasicBlock *BB, unsigned I,
                                     BasicBlock *NewBB) {
  for (PHINode &phi : BB->phis()) {
    phi.setIncomingBlock(I, NewBB);
  }
}

inline void setPHINodesBlock(BasicBlock *BB, BasicBlock *Pred,
                             BasicBlock *Succ) {
  for (PHINode &phi : BB->phis()) {
    phi.setIncomingBlock(0, Pred);
    phi.setIncomingBlock(1, Succ);
  }
}

inline void runPostPass(Function &F) {
  // Create necessary analysis managers
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  // Create pass builder
  PassBuilder PB;

  // Register analyses
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  FunctionPassManager FPM;
  FPM.addPass(InstCombinePass());
  FPM.addPass(GVNPass());
  FPM.run(F, FAM);
}

} // end namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVESP32P4OPTUTILS_H