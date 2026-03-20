//===- RISCVLowerMatrixType.cpp - Lower riscv.matrix for -O0 ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file Pre-ISel pass for XTHeadMatrix (RVM 0.6) ManagedRA support.
///
/// At -O0, the fast register allocator cannot handle long live ranges of
/// matrix registers (only 4 tile + 4 accumulator registers). This pass
/// inserts store+reload around matrix definitions to keep live ranges short,
/// analogous to X86's X86LowerAMXType volatileTileData().
///
/// We use the _internal whole-register store/load intrinsics
/// (msme_internal/mlme_internal) for spilling, which correctly handle the
/// full matrix register size (1024 bytes). This avoids the alloca size
/// mismatch that would occur with direct store/load of TargetExtType.
///
/// At -O2, this pass is a no-op — the greedy register allocator handles
/// matrix register pressure natively.
///
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-lower-matrix-type"

namespace {

/// Returns true if the type is target("riscv.matrix").
static bool isMatrixType(Type *Ty) {
  if (auto *TETy = dyn_cast<TargetExtType>(Ty))
    return TETy->getName() == "riscv.matrix";
  return false;
}

/// Returns true if this instruction is an _internal matrix intrinsic.
static bool isMatrixInternalIntrinsic(const IntrinsicInst *II) {
  StringRef Name = II->getCalledFunction()->getName();
  return Name.contains("riscv.th.") && Name.contains("internal");
}

/// Matrix register size in bytes (8192 bits = 1024 bytes).
static constexpr unsigned MatrixRegSizeBytes = 1024;

/// At -O0, insert whole-register store+reload around matrix definitions to
/// shorten live ranges for the fast register allocator.
///
/// For each matrix-producing _internal intrinsic:
///   %mat = call target("riscv.matrix") @llvm.riscv.th.xxx_internal(...)
///   ; ... uses of %mat ...
/// becomes:
///   %mat = call target("riscv.matrix") @llvm.riscv.th.xxx_internal(...)
///   %spill_slot = alloca [1024 x i8], align 8
///   call void @llvm.riscv.th.msme.internal8(%mat, ptr %spill_slot)
///   ; ... at each use:
///   %reloaded = call target("riscv.matrix") @llvm.riscv.th.mlme.internal8(ptr %spill_slot)
///   ; use %reloaded instead of %mat
static bool volatileMatrixData(Function &F) {
  bool Changed = false;
  auto &Ctx = F.getContext();
  Type *MatTy = TargetExtType::get(Ctx, "riscv.matrix");
  Type *SpillTy = ArrayType::get(Type::getInt8Ty(Ctx), MatrixRegSizeBytes);

  // Collect all matrix-producing intrinsic calls.
  SmallVector<IntrinsicInst *, 16> MatrixDefs;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (auto *II = dyn_cast<IntrinsicInst>(&I))
        if (isMatrixType(II->getType()) && isMatrixInternalIntrinsic(II))
          MatrixDefs.push_back(II);

  if (MatrixDefs.empty())
    return false;

  // Get XLen type for stride parameter.
  const DataLayout &DL = F.getDataLayout();
  Type *XLenTy = IntegerType::get(Ctx, DL.getPointerSizeInBits());
  Value *ZeroStride = ConstantInt::get(XLenTy, 0);

  // Get or declare the whole-register store/load intrinsics for spilling.
  Function *StoreIntrin = Intrinsic::getOrInsertDeclaration(
      F.getParent(), Intrinsic::riscv_th_msme_internal8, {MatTy, XLenTy});
  Function *LoadIntrin = Intrinsic::getOrInsertDeclaration(
      F.getParent(), Intrinsic::riscv_th_mlme_internal8, {MatTy, XLenTy});

  // For each matrix def, insert spill after def, and reload before each use.
  IRBuilder<> AllocaBuilder(&F.getEntryBlock().front());

  for (IntrinsicInst *Def : MatrixDefs) {
    // Create spill slot alloca (1024-byte aligned array) in entry block.
    AllocaInst *SpillSlot = AllocaBuilder.CreateAlloca(
        SpillTy, nullptr, Def->getName() + ".spill");
    SpillSlot->setAlignment(Align(8));

    // Store (spill) right after the def via whole-register store intrinsic.
    // Stride=0 for contiguous spill.
    IRBuilder<> StoreBuilder(Def->getNextNode());
    StoreBuilder.CreateCall(StoreIntrin, {Def, SpillSlot, ZeroStride});

    // Replace each use (except the store we just inserted) with a reload.
    SmallVector<Use *, 8> Uses;
    for (Use &U : Def->uses())
      Uses.push_back(&U);

    for (Use *U : Uses) {
      auto *UserInst = dyn_cast<Instruction>(U->getUser());
      if (!UserInst)
        continue;
      // Don't replace the spill store we just inserted.
      if (auto *CI = dyn_cast<CallInst>(UserInst))
        if (CI->getCalledFunction() == StoreIntrin &&
            CI->getArgOperand(1) == SpillSlot)
          continue;

      IRBuilder<> LoadBuilder(UserInst);
      Value *Reloaded = LoadBuilder.CreateCall(
          LoadIntrin, {SpillSlot, ZeroStride}, Def->getName() + ".reload");
      U->set(Reloaded);
    }
    Changed = true;
  }

  return Changed;
}

class RISCVLowerMatrixType : public FunctionPass {
public:
  static char ID;

  RISCVLowerMatrixType() : FunctionPass(ID) {
    initializeRISCVLowerMatrixTypePass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    // Only run at -O0 (fast regalloc).
    auto *TPC = getAnalysisIfAvailable<TargetPassConfig>();
    if (!TPC)
      return false;
    if (TPC->getOptLevel() != CodeGenOptLevel::None)
      return false;

    // Quick check: does the function use matrix types at all?
    bool HasMatrix = false;
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (isMatrixType(I.getType())) {
          HasMatrix = true;
          break;
        }
      }
      if (HasMatrix)
        break;
    }
    if (!HasMatrix)
      return false;

    return volatileMatrixData(F);
  }

  StringRef getPassName() const override {
    return "RISC-V Lower Matrix Type";
  }
};

} // namespace

char RISCVLowerMatrixType::ID = 0;

INITIALIZE_PASS(RISCVLowerMatrixType, DEBUG_TYPE,
                "RISC-V Lower Matrix Type", false, false)

FunctionPass *llvm::createRISCVLowerMatrixTypePass() {
  return new RISCVLowerMatrixType();
}
